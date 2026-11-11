// park_replay_sim: a recorded agent trace replayed through the REAL ConversationCache (include/strata/core/conversation_cache.hpp)
// to judge the eviction rules offline: the default oldest-first rule, MOD_PLAN A15 (STRATA_PARK_TTL_S, a time-to-live / idleness
// rule after Continuum, arXiv 2511.02230) and F1 (STRATA_PARK_TRUNCATE, TwoAnts/Strata 5ef4fc8b4b).  No GPU, no model: snapshots are
// synthetic images whose byte counts follow the size model fitted on the 2026-11-09 engine log (RESULTS.md 4.0.1):
//     size = 111 MiB + 14.9 KiB * tokens + 113 MiB * checkpoints
// scaled down by --scale (default 20) so that the four parked images of a 60K-token agent fit in a few hundred MiB; every byte count
// the cache sees (budget, snapshots, truncations) is scaled the same way, so the budget decisions are the real ones.
//
//     park_replay_sim --trace FILE [--budget-mib 1280] [--slots 4] [--min-tokens 0] [--truncate] [--tail-gap N]
//                     [--ttl-s S] [--hold-s 60] [--share 512] [--sub-ratio 0.5] [--tool-agent] [--verbose]
//     park_replay_sim --synthetic ROUNDS,PAR,TURNS,SHARED [options]    (the built-in main agent + parallel sub-agents trace)
//     park_replay_sim --selftest
//
// Trace: one request per line, `#` comments, fields `conv kind target out start_s dur_s tool shared`:
//   conv    the conversation's name (a request with the same name continues it; a side request repeats a name to repeat its text)
//   kind    A (an agent turn: judged for cold re-reads) or S (a side request)
//   target  the prompt's tokens; out  the generated tokens; start_s / dur_s  the request's start and duration (seconds, any epoch)
//   tool    1 when the reply ends in a tool call (the A15 hold), else 0
//   shared  leading tokens this conversation shares with every other one that names the same count (0: none): a system prompt + tools
// The flow is generate.cpp's: at a request the live session is kept when it is the same conversation, else the best parked match is
// taken (or borrowed from, ConversationCache::borrows) and the outgoing conversation is parked beside it (drop_superseded, make_room,
// put; with --truncate the deepest checkpoint prefix that fits when the whole snapshot does not).  The checkpoints are the periodic
// ones (every 16384 tokens from where a request resumed, at most 6) as in bench/results/2026-11-10-offline/park_replay.py, whose
// model this reuses; --tail-gap N adds one N tokens before the prompt's end.  The RAM floor and the disk tier are not modelled.
#include "strata/core/conversation_cache.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace strata::core;

namespace {

constexpr int32_t kToolTok = 99;
constexpr double kMiB = 1048576.0;
constexpr int64_t kPeriod = 16384;   // the periodic checkpoint spacing (--prompt-cache-every)
constexpr size_t kMaxCk = 6;         // checkpoints kept per conversation in the log's chains

struct Req {
    std::string conv;
    char kind = 'S';
    int64_t target = 0, out = 0;
    double start_s = 0.0, dur_s = 0.0;
    bool tool = false;
    int64_t shared = 0;
};

struct Opts {
    double budget_mib = 1280.0;
    size_t slots = 4;
    int64_t min_tokens = 0;
    bool truncate = false;
    int64_t tail_gap = 0;
    double ttl_s = 0.0, hold_s = 60.0, sub_ratio = 0.5;
    int64_t share = 512;
    bool tool_agent = false;
    double scale = 20.0;
    double ms_per_tok_cold = 0.41, ms_per_tok_short = 0.64, floor_s = 0.2, restore_s = 0.9;
    bool verbose = false;
};

struct Result {
    double agent_prompt_s = 0.0, all_prompt_s = 0.0, cold_s = 0.0;
    int cold = 0, parks = 0, truncated_parks = 0, skips = 0, borrows = 0, restores = 0, requests = 0;
    size_t evictions = 0, ttl_evictions = 0, truncations = 0;
    std::vector<int> cold_idx;
    std::vector<std::string> evicted_trace;   // per request: what the cache looked like before it ran (verbose)
};

uint64_t mix(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

uint64_t name_hash(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) h = (h ^ c) * 1099511628211ull;
    return h;
}

class Sim {
public:
    explicit Sim(const Opts& o)
        : o_(o), cache_((size_t) (o.budget_mib * kMiB / o.scale), o.slots, o.min_tokens, 0, o.truncate) {
        if (o.ttl_s > 0.0) {
            ConversationCache::ParkTtl pt;
            pt.ttl_ms = (int64_t) std::max(1.0, o.ttl_s * 1000.0);
            pt.hold_ms = (int64_t) std::max(0.0, o.hold_s * 1000.0);
            pt.share_min = std::max<int64_t>(1, o.share);
            pt.sub_ratio = o.sub_ratio;
            cache_.set_park_ttl(
                pt, [](const SavedConversation& e) { return !e.live.ids.empty() && e.live.ids.back() == kToolTok; },
                [this] { return now_ms_; });
        }
    }

    Result run(const std::vector<Req>& trace) {
        Result res;
        int idx = 0;
        for (const Req& r : trace) {
            serve(r, idx++, res);
        }
        res.evictions = cache_.evictions();
        res.ttl_evictions = cache_.ttl_evictions();
        res.truncations = cache_.truncations();
        return res;
    }

private:
    struct Live {
        bool valid = false;
        std::string conv;
        int64_t shared = 0;
        std::vector<int32_t> ids;       // the session's tokens: the prompt and what was generated
        std::vector<int64_t> ckpos;     // its checkpoints' depths
        int ck_total = 0;               // the checkpoint count the size model charges (>= ckpos.size())
    };

    // ---- tokens: a conversation is one stream; a prompt is its first `target` tokens, so a later turn extends an earlier one
    std::vector<int32_t>& stream(const std::string& conv, int64_t shared, int64_t n) {
        std::vector<int32_t>& s = streams_[conv];
        const uint64_t h = name_hash(conv);
        while ((int64_t) s.size() < n) {
            const int64_t pos = (int64_t) s.size();
            if (pos < shared) s.push_back((int32_t) (1000 + pos % 100000));
            else s.push_back((int32_t) (1000000 + mix(h ^ (uint64_t) (pos * 0x9E3779B97F4A7C15ull)) % 900000000ull));
        }
        return s;
    }

    // ---- images: bytes follow the fitted model, scaled; the ids' own bytes are inside the model, so the filler is the rest
    size_t scaled(double mib) const { return (size_t) std::llround(mib * kMiB / o_.scale); }

    SavedConversation make_image(const std::string& conv, int64_t shared, int64_t len, const std::vector<int64_t>& ckpos,
                                 int ck_total, double live_base_mib) {
        const std::vector<int32_t>& s = stream(conv, shared, len);
        SavedConversation img;
        img.live.ids.assign(s.begin(), s.begin() + len);
        ConversationKv kv;
        kv.cells = len;
        kv.page_size = 1;
        const size_t per_cell = (size_t) std::llround(14.9 * 1024.0 / o_.scale);
        kv.k.resize((size_t) len * (per_cell / 2));
        kv.v.resize((size_t) len * (per_cell - per_cell / 2));
        img.kv.push_back(std::move(kv));
        for (int64_t pos : ckpos) {
            ConversationCheckpoint ck;
            ck.ids.assign(s.begin(), s.begin() + pos);
            const size_t want = scaled(113.0), own = ck.ids.size() * sizeof(int32_t);
            ck.gdn.resize(want > own ? want - own : 1);
            img.checkpoints.push_back(std::move(ck));
        }
        img.geometry[0] = ck_total;
        img.geometry[1] = len;
        // the live state's filler: the whole image is model(len, ck_total), whatever the ids and the structs cost
        // (the real checkpoints' bytes are in `other` already; the model's remaining checkpoints are the filler's)
        const double total_mib = live_base_mib + 14.9 / 1024.0 * (double) len + 113.0 * (double) ck_total;
        const size_t other = img.bytes(), want = scaled(total_mib);
        img.live.gdn.resize(want > other ? want - other : 1);
        return img;
    }

    // ---- parking the outgoing conversation (generate.cpp park_current), with F1's truncated park when the whole image does not fit
    void park_current(size_t held, bool guarded_borrow, Result& res) {
        if (!live_.valid) return;
        if (!cache_.wants((int64_t) live_.ids.size())) {
            live_.valid = false;
            return;
        }
        const double base = 111.0;
        const int64_t len = (int64_t) live_.ids.size();
        std::vector<int64_t> pos = live_.ckpos;
        SavedConversation img = make_image(live_.conv, live_.shared, len, pos, std::max<int>(live_.ck_total, (int) pos.size()), base);
        cache_.drop_superseded(img.live.ids, img.live.imgs, img.checkpoints, img.cvec);
        const size_t est = img.bytes();
        (void) guarded_borrow;
        if (cache_.make_room(est, held)) {
            const bool stored = cache_.put(std::move(img), held);
            if (stored) ++res.parks;
            else ++res.skips;
        } else if (!(o_.truncate && park_truncated(held, res))) {
            ++res.skips;
        }
        live_.valid = false;
    }

    // F1: the deepest checkpoint prefix that fits (generate.cpp park_truncated, without the RAM floor)
    bool park_truncated(size_t held, Result& res) {
        std::vector<int64_t> order;
        for (int64_t p : live_.ckpos)
            if (p > 0 && p < (int64_t) live_.ids.size() && cache_.wants(p)) order.push_back(p);
        std::sort(order.begin(), order.end(), std::greater<int64_t>());
        int64_t last = -1;
        for (int64_t L : order) {
            if (L == last) continue;
            last = L;
            std::vector<int64_t> kept;
            for (int64_t p : live_.ckpos)
                if (p < L) kept.push_back(p);
            SavedConversation img = make_image(live_.conv, live_.shared, L, kept, (int) kept.size() + 1, 0.0);   // (the landing's own state is the live one: 113 MiB, counted in ck_total)
            const size_t est = img.bytes();
            if (!cache_.make_room(est, held)) continue;
            if (cache_.put(std::move(img), held)) {
                ++res.truncated_parks;
                return true;
            }
        }
        return false;
    }

    double prompt_s(int64_t read) const {
        return o_.floor_s + (double) std::max<int64_t>(0, read) * (read >= 8192 ? o_.ms_per_tok_cold : o_.ms_per_tok_short) / 1000.0;
    }

    void serve(const Req& r, int idx, Result& res) {
        now_ms_ = (int64_t) std::llround(r.start_s * 1000.0);
        ++res.requests;
        std::vector<int32_t>& s = stream(r.conv, r.shared, r.target);
        const std::vector<int32_t> prompt(s.begin(), s.begin() + r.target);
        const bool first = !seen_.count(r.conv);
        seen_.insert(r.conv);
        const bool from_live = live_.valid && live_.conv == r.conv;
        int64_t resume = from_live ? std::min<int64_t>((int64_t) live_.ids.size(), r.target - 1) : 0;
        if (o_.verbose) res.evicted_trace.push_back(describe());

        const ConversationCache::Match parked = cache_.best(prompt, {}, true);
        bool borrow = false;
        std::optional<SavedConversation> incoming;
        if (parked.tokens > resume) {
            borrow = cache_.borrows(parked);
            if (!borrow) incoming.emplace(cache_.take(parked.index));
        }
        if (borrow) {
            cache_.guard(parked.serial);
            park_current(0, true, res);
            const bool blocked = cache_.guard_blocked();
            cache_.unguard();
            if (blocked) {   // the outgoing conversation fits only in the donor's room: the donor is taken whole
                borrow = false;
                const size_t at = cache_.index_of(parked.serial);
                if (at != SIZE_MAX) incoming.emplace(cache_.take(at));
                park_current(incoming ? incoming->bytes() : 0, false, res);
            }
        } else if (!from_live || incoming) {
            park_current(incoming ? incoming->bytes() : 0, false, res);
        }

        // what the request reuses, and the checkpoints and size class it continues from
        int64_t reused = 0;
        std::vector<int64_t> pos;
        int ck = 1;
        bool restored = false;
        if (incoming || borrow) {
            const SavedConversation* donor = incoming ? &*incoming : cache_.find(parked.serial);
            reused = parked.tokens;
            if (donor != nullptr) {
                for (const auto& c : donor->checkpoints)
                    if ((int64_t) c.ids.size() <= reused) pos.push_back((int64_t) c.ids.size());   // (the restore point stays a checkpoint)
                std::sort(pos.begin(), pos.end());
                const bool truncated = donor->geometry[1] != (int64_t) donor->live.ids.size();
                const int donor_ck = truncated ? (int) donor->checkpoints.size() + 1 : (int) donor->geometry[0];
                ck = (int) std::min<size_t>(kMaxCk, (size_t) std::max(donor_ck, (int) pos.size()) + 1);
            }
            restored = true;
            if (incoming) ++res.restores;
            else ++res.borrows;
        } else if (from_live) {
            reused = resume;
            pos = live_.ckpos;
            ck = (int) std::min<size_t>(kMaxCk, (size_t) live_.ck_total + 1);
        }
        const int64_t read = r.target - reused;
        for (int64_t i = 1; i <= read / kPeriod; ++i) pos.push_back(reused + i * kPeriod);
        if (o_.tail_gap > 0 && read > 0 && r.target - o_.tail_gap > reused) pos.push_back(r.target - o_.tail_gap);
        std::sort(pos.begin(), pos.end());
        pos.erase(std::unique(pos.begin(), pos.end()), pos.end());
        while (pos.size() > kMaxCk) pos.erase(pos.begin());
        if (!restored && !from_live) ck = r.kind == 'A' ? std::max<int>(1, (int) pos.size()) : 1;
        ck = std::max<int>(ck, (int) pos.size());

        double cost = prompt_s(read);
        if (incoming) cost += o_.restore_s * (double) incoming->bytes() * o_.scale / kMiB / 1257.0;
        res.all_prompt_s += cost;
        if (r.kind == 'A') res.agent_prompt_s += cost;
        const bool cold = r.kind == 'A' && !first && r.target > 8192 && (double) reused < 0.5 * (double) r.target;
        if (cold) {
            ++res.cold;
            res.cold_s += cost;
            res.cold_idx.push_back(idx);
        }
        if (o_.verbose)
            std::printf("  %3d %-4s %c target %6lld reused %6lld read %6lld %6.1f s%s%s\n", idx, r.conv.c_str(), r.kind,
                        (long long) r.target, (long long) reused, (long long) read, cost, cold ? "  COLD" : "",
                        borrow ? "  borrowed" : incoming ? "  restored" : from_live ? "" : "  cold start");

        // the session now holds the prompt and the reply
        std::vector<int32_t>& s2 = stream(r.conv, r.shared, r.target + std::max<int64_t>(1, r.out));
        if (r.tool || (o_.tool_agent && r.kind == 'A')) s2[(size_t) (r.target + std::max<int64_t>(1, r.out) - 1)] = kToolTok;
        live_.valid = true;
        live_.conv = r.conv;
        live_.shared = r.shared;
        live_.ids.assign(s2.begin(), s2.begin() + r.target + std::max<int64_t>(1, r.out));
        live_.ckpos = pos;
        live_.ck_total = ck;
    }

    std::string describe() const {
        std::ostringstream os;
        os << "cache:";
        // (no accessor lists the entries; the sizes and the count are what the policy decisions are judged on)
        os << " n=" << cache_.size() << " bytes=" << (size_t) ((double) cache_.bytes() * o_.scale / kMiB) << "MiB";
        return os.str();
    }

    Opts o_;
    ConversationCache cache_;
    Live live_;
    int64_t now_ms_ = 0;
    std::map<std::string, std::vector<int32_t>> streams_;
    std::set<std::string> seen_;
};

bool parse_trace(std::istream& in, std::vector<Req>& out, std::string* err) {
    std::string line;
    int ln = 0;
    while (std::getline(in, line)) {
        ++ln;
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        Req r;
        std::string kind;
        int tool = 0;
        long long target = 0, out_n = 0, shared = 0;
        if (!(ls >> r.conv >> kind >> target >> out_n >> r.start_s >> r.dur_s >> tool >> shared) || kind.size() != 1) {
            if (err != nullptr) *err = "trace line " + std::to_string(ln) + ": expected `conv kind target out start_s dur_s tool shared`";
            return false;
        }
        r.kind = kind[0];
        r.target = target;
        r.out = out_n;
        r.tool = tool != 0;
        r.shared = shared;
        if (r.target < 2) {
            if (err != nullptr) *err = "trace line " + std::to_string(ln) + ": target < 2";
            return false;
        }
        out.push_back(r);
    }
    return true;
}

void print_result(const char* name, const Result& r, bool cold_list) {
    std::printf("%-34s agent prompt %7.1f s  cold %d (%6.1f s)  all %7.1f s | parks %d (+%d truncated) skips %d restores %d borrows %d | "
                "evictions %zu (ttl %zu) truncations %zu\n",
                name, r.agent_prompt_s, r.cold, r.cold_s, r.all_prompt_s, r.parks, r.truncated_parks, r.skips, r.restores, r.borrows,
                r.evictions, r.ttl_evictions, r.truncations);
    if (cold_list && !r.cold_idx.empty()) {
        std::printf("    cold requests:");
        for (int i : r.cold_idx) std::printf(" %d", i);
        std::printf("\n");
    }
}

// ------------------------------------------------------------------------------------------------ the self-test
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

// a main agent (one growing conversation that starts `par` sub-agents through a tool call) and the short sub-agent sessions that run
// side by side (their turns interleave, so each switch parks one) and share the main agent's system prompt: the shape A15 is built for.
// Times in seconds.
std::vector<Req> synthetic_agent_trace(int rounds, int par, int sub_turns, int64_t shared) {
    std::vector<Req> t;
    double now = 0.0;
    int64_t main_tokens = 30000;
    int sub = 0;
    for (int k = 0; k < rounds; ++k) {
        Req m;
        m.conv = "M"; m.kind = 'A'; m.target = main_tokens; m.out = 400; m.start_s = now; m.dur_s = 8.0; m.tool = true; m.shared = shared;
        t.push_back(m);
        now += m.dur_s + 1.0;
        std::vector<std::string> names;
        for (int i = 0; i < par; ++i) names.push_back("sub" + std::to_string(sub++));
        for (int j = 0; j < sub_turns; ++j)
            for (const std::string& name : names) {
                Req s;
                s.conv = name; s.kind = 'S'; s.target = 6000 + 1200 * j; s.out = 300; s.start_s = now; s.dur_s = 3.0;
                s.tool = j + 1 < sub_turns; s.shared = shared;
                t.push_back(s);
                now += s.dur_s + 1.0;
            }
        main_tokens += 2500;
    }
    return t;
}

int selftest() {
    // 1. the parse: a malformed line is refused with its number
    {
        std::vector<Req> t;
        std::string err;
        std::istringstream bad("A\nM A 100 5 0 1 0\n");
        check(!parse_trace(bad, t, &err) && err.find("line 1") != std::string::npos, "a short line is refused with its number");
        std::istringstream ok("# c\nM A 100 5 0 1 0 0\nX S 50 5 2 1 1 64\n");
        t.clear();
        check(parse_trace(ok, t, &err) && t.size() == 2 && t[1].tool && t[1].shared == 64, "a trace parses");
    }
    // 2. the default rule is the old oldest-first: the main agent, parked first, is the one the sub-agents' snapshots push out
    const std::vector<Req> trace = synthetic_agent_trace(6, 4, 3, 1024);
    Opts base;
    base.verbose = std::getenv("PARK_SIM_VERBOSE") != nullptr;
    base.budget_mib = 2048;   // room for the main agent (~1.1 GiB) and one or two sub-agent images
    Result r0 = Sim(base).run(trace);
    if (std::getenv("PARK_SIM_VERBOSE")) print_result("default", r0, true);
    check(r0.cold >= 2, "default: sub-agent parks push the main agent out (cold re-reads)");
    check(r0.ttl_evictions == 0 && r0.truncations == 0, "default: no TTL and no truncation counters");
    // 3. A15 on: the sub-agent branches leave first (shared system prompt, much shorter), the main agent stays
    Opts ttl = base;
    ttl.ttl_s = 5.0;
    ttl.hold_s = 0.0;   // no tool hold: only the branch ranking is under test
    Result r1 = Sim(ttl).run(trace);
    if (std::getenv("PARK_SIM_VERBOSE")) print_result("A15 ttl 5 s", r1, true);
    check(r1.cold < r0.cold, "A15: fewer cold re-reads than the default");
    check(r1.agent_prompt_s < r0.agent_prompt_s, "A15: less agent prompt time than the default");
    // 4. A15 with no shared prefix: nothing is a branch, the rule is the old one (the 11-09 trace's situation)
    const std::vector<Req> lone = synthetic_agent_trace(6, 4, 3, 0);
    Result l0 = Sim(base).run(lone);
    Result l1 = Sim(ttl).run(lone);
    check(l1.cold == l0.cold && std::fabs(l1.agent_prompt_s - l0.agent_prompt_s) < 1e-9,
          "A15 with no shared prefix equals the default (no branch to rank)");
    // 5. the tool hold alone (Continuum's pin): the main agent's reply ended in a tool call, so it is kept for the hold
    Opts hold = base;
    hold.ttl_s = 5.0;
    hold.hold_s = 600.0;
    Result h1 = Sim(hold).run(lone);
    check(h1.cold <= l0.cold, "A15 hold on a trace with no shared prefix is never worse than the default");
    // 6. F1: a conversation too large for the budget parks its deepest prefix that fits instead of nothing
    {
        std::vector<Req> t;
        double now = 0.0;
        int64_t tok = 20000;
        for (int k = 0; k < 6; ++k) {   // one long conversation interleaved with a short unrelated one
            Req a;
            a.conv = "L"; a.kind = 'A'; a.target = tok; a.out = 300; a.start_s = now; a.dur_s = 5.0;
            t.push_back(a);
            now += 6.0;
            Req b;
            b.conv = "q"; b.kind = 'S'; b.target = 80; b.out = 20; b.start_s = now; b.dur_s = 1.0;
            t.push_back(b);
            now += 2.0;
            tok += 3000;
        }
        Opts small;
        small.budget_mib = 560;   // the 40K+ conversation (~ 111 + 600 + 2 checkpoints * 113 MiB) no longer fits
        Result p0 = Sim(small).run(t);
        Opts tr = small;
        tr.truncate = true;
        Result p1 = Sim(tr).run(t);
        check(p0.skips >= 1, "F1 setup: the long conversation's snapshot does not fit the budget");
        check(p1.truncated_parks >= 1, "F1: a truncated park happens");
        check(p1.agent_prompt_s < p0.agent_prompt_s, "F1: the next turn re-reads only the tail");
        Opts off = small;   // the switch off is the old rule, bit for bit
        Result p2 = Sim(off).run(t);
        check(p2.truncated_parks == 0 && p2.truncations == 0 && p2.agent_prompt_s == p0.agent_prompt_s, "F1 off: the old rule");
    }
    if (g_fail == 0) std::printf("park_replay_sim_test: all passed\n");
    return g_fail == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    Opts o;
    std::string trace_path;
    bool self = false;
    std::string synthetic;
    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* name) { return std::strcmp(argv[i], name) == 0 && i + 1 < argc; };
        if (std::strcmp(argv[i], "--selftest") == 0) self = true;
        else if (arg("--trace")) trace_path = argv[++i];
        else if (arg("--synthetic")) synthetic = argv[++i];
        else if (arg("--budget-mib")) o.budget_mib = std::atof(argv[++i]);
        else if (arg("--slots")) o.slots = (size_t) std::atoll(argv[++i]);
        else if (arg("--min-tokens")) o.min_tokens = std::atoll(argv[++i]);
        else if (std::strcmp(argv[i], "--truncate") == 0) o.truncate = true;
        else if (arg("--tail-gap")) o.tail_gap = std::atoll(argv[++i]);
        else if (arg("--ttl-s")) o.ttl_s = std::atof(argv[++i]);
        else if (arg("--hold-s")) o.hold_s = std::atof(argv[++i]);
        else if (arg("--share")) o.share = std::atoll(argv[++i]);
        else if (arg("--sub-ratio")) o.sub_ratio = std::atof(argv[++i]);
        else if (arg("--scale")) o.scale = std::max(1.0, std::atof(argv[++i]));
        else if (std::strcmp(argv[i], "--tool-agent") == 0) o.tool_agent = true;
        else if (std::strcmp(argv[i], "--verbose") == 0) o.verbose = true;
        else {
            std::fprintf(stderr, "park_replay_sim: unknown or incomplete argument %s\n", argv[i]);
            return 2;
        }
    }
    if (self) return selftest();
    if (!synthetic.empty()) {
        int rounds = 6, par = 4, turns = 3;
        long long shared = 1024;
        std::sscanf(synthetic.c_str(), "%d,%d,%d,%lld", &rounds, &par, &turns, &shared);
        const Result r = Sim(o).run(synthetic_agent_trace(rounds, par, turns, shared));
        print_result("result", r, true);
        return 0;
    }
    if (trace_path.empty()) {
        std::fprintf(stderr, "usage: park_replay_sim --trace FILE [options] | --synthetic R,P,T,S [options] | --selftest (see the file's header)\n");
        return 2;
    }
    std::ifstream in(trace_path);
    std::vector<Req> trace;
    std::string err;
    if (!in || !parse_trace(in, trace, &err)) {
        std::fprintf(stderr, "park_replay_sim: %s\n", in ? err.c_str() : "cannot read the trace");
        return 2;
    }
    const Result r = Sim(o).run(trace);
    print_result("result", r, true);
    return 0;
}
