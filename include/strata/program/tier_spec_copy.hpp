#pragma once
// LEADS.md B147 (owner idea 2026-11-18): STRATA_ADAPT_SPEC_COPY=1, speculative early copy of the experts the adaptive tier is
// about to bring in.  Placement is not touched: the tier still decides every swap at its round, from the counts, exactly as
// before.  What changes is where the swap's bytes come from.  At the END of a round the CPU predicts the NEXT round's incoming
// experts (the tier's own selection rule run on predicted counts), and copies up to STRATA_ADAPT_SPEC_MAX of them from the host
// into a bounded spare staging area in VRAM, on a second stream, while the window runs.  At the next round an incoming expert
// that is staged (and landed) is finalised with a device-to-device copy into the victim's slot (no host read, no PCIe at that
// moment); every other incoming expert takes the normal host-to-device copy.  The slot bytes are identical either way.
//
//   STRATA_ADAPT_SPEC_COPY=1        on (default off: nothing changes, byte-identical)
//   STRATA_ADAPT_SPEC_SLOTS=S       staging slots per layer, 1..16, default 4 (VRAM: layers x S x the largest expert blob)
//   STRATA_ADAPT_SPEC_MAX=M         most staged copies per round (all layers), 1..64, default 8 (the PCIe bytes it may add)
//   STRATA_ADAPT_SPEC_CONF=C        a predicted swap needs gain (candidate count - victim count) >= C, 0..100, default 1.0
//   STRATA_ADAPT_SPEC_AGE=A         rounds a staged expert is kept (it is kept alive again whenever it is predicted again),
//                                   1..64, default 4
//
// The predictor ("ps" in tools/pancake/replay_spec_copy.py): next round's counts are guessed as decay x this round's counts + this
// window's increment (the next window routes like this one); the staged set is the swaps the tier's rule would pick on them.
// Safety (T-952 half-written slot): a staging slot is written only by the spec stream, after the adapt stream has finished
// reading the previous round's staged bytes (the caller waits on the round's event), and read only by a copy that waits for the
// spec stream's event; a victim's slot is written by the same one copy the baseline issues.  Pure logic here so a ctest covers it.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace strata::program {

class TierSpecCopy {
public:
    struct Params {
        bool on = false;
        int slots = 4;          ///< staging slots per layer
        int max_copies = 8;     ///< staged copies per round
        float conf = 1.0f;      ///< minimum predicted gain
        int age = 4;            ///< rounds a staged expert lives
    };
    struct Staged { int32_t layer, expert, slot; };   ///< slot = index inside the layer's staging slots
    struct Pred { float gain; int32_t layer, expert; };

    static bool parse_int(const char* name, int lo, int hi, int dflt, int& out) {
        const char* v = std::getenv(name);
        out = dflt;
        if (v == nullptr || v[0] == '\0') return true;
        char* end = nullptr;
        const long x = std::strtol(v, &end, 10);
        if (end == v || *end != '\0' || x < lo || x > hi) {
            std::fprintf(stderr, "strata serve: %s=%s is not an integer in %d..%d: using %d\n", name, v, lo, hi, dflt);
            return false;
        }
        out = (int) x;
        return true;
    }

    static Params params_from_env() {
        Params p;
        const char* v = std::getenv("STRATA_ADAPT_SPEC_COPY");
        p.on = v != nullptr && v[0] != '\0' && std::atoi(v) != 0;
        if (!p.on) return p;
        parse_int("STRATA_ADAPT_SPEC_SLOTS", 1, 16, 4, p.slots);
        parse_int("STRATA_ADAPT_SPEC_MAX", 1, 64, 8, p.max_copies);
        parse_int("STRATA_ADAPT_SPEC_AGE", 1, 64, 4, p.age);
        if (const char* c = std::getenv("STRATA_ADAPT_SPEC_CONF"); c != nullptr && c[0] != '\0') {
            char* end = nullptr;
            const double x = std::strtod(c, &end);
            if (end == c || *end != '\0' || !(x >= 0.0 && x <= 100.0))
                std::fprintf(stderr, "strata serve: STRATA_ADAPT_SPEC_CONF=%s is not a number in 0..100: using 1\n", c);
            else
                p.conf = (float) x;
        }
        return p;
    }

    static TierSpecCopy from_env(int64_t n_layers, int64_t n_expert) {
        TierSpecCopy t;
        t.configure(params_from_env(), n_layers, n_expert);
        return t;
    }

    void configure(const Params& p, int64_t n_layers, int64_t n_expert) {
        p_ = p;
        L_ = n_layers;
        E_ = n_expert;
        ent_.assign(p_.on ? (size_t) (n_layers * p_.slots) : 0, Entry{});
        if (p_.on)
            std::fprintf(stderr, "strata serve: STRATA_ADAPT_SPEC_COPY=1: %d staging slots per layer, at most %d staged copies per round, "
                                 "predicted gain >= %.2f, kept %d rounds; the slot bytes are the normal swap's (device copy from the staging "
                                 "slot instead of the host copy)\n", p_.slots, p_.max_copies, (double) p_.conf, p_.age);
    }

    bool on() const { return p_.on; }
    const Params& params() const { return p_; }
    int64_t staging_slots() const { return p_.on ? L_ * p_.slots : 0; }
    uint64_t rounds() const { return round_; }
    uint64_t staged_total() const { return staged_; }
    uint64_t hits() const { return hits_; }
    uint64_t misses() const { return misses_; }
    uint64_t wasted() const { return wasted_; }
    uint64_t not_landed() const { return not_landed_; }
    double mean_us() const { return round_ > 0 ? (double) ns_ / 1000.0 / (double) round_ : 0.0; }

    /// Round start (before the swaps are chosen): `raw` = the tier's counts, `decay` the tier's decay.  Keeps the counts and this
    /// window's increment (count - last count x decay, >= 0); ages the staged experts out.
    void begin_round(const std::vector<float>& raw, float decay) {
        if (!p_.on || (int64_t) raw.size() != L_ * E_) return;
        const auto t0 = std::chrono::steady_clock::now();
        ++round_;
        if (last_.empty()) {
            last_.assign(raw.size(), 0.0f);
            delta_.assign(raw.size(), 0.0f);
        }
        decay_ = decay;
        for (size_t i = 0; i < raw.size(); ++i) {
            delta_[i] = std::max(0.0f, raw[i] - last_[i] * decay);
            last_[i] = raw[i];
        }
        for (Entry& e : ent_)
            if (e.expert >= 0 && (int64_t) round_ - (int64_t) e.round > (int64_t) p_.age) { e.expert = -1; ++wasted_; }
        ns_ += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    }

    /// Index into the whole staging area (layer x slots + slot) of the staged copy of (layer, expert), or -1.
    int64_t find(int64_t layer, int32_t expert) const {
        if (!p_.on) return -1;
        for (int s = 0; s < p_.slots; ++s)
            if (ent_[(size_t) (layer * p_.slots + s)].expert == expert) return layer * p_.slots + s;
        return -1;
    }

    /// An incoming expert of this round: true (and the entry is freed) when it was staged.  `landed` = the staged copy had completed
    /// (the caller's event); a staged copy that has not landed counts as not_landed and is still used (the device copy waits for it).
    bool consume(int64_t layer, int32_t expert, int64_t* index = nullptr) {
        if (!p_.on) return false;
        const int64_t i = find(layer, expert);
        if (i < 0) { ++misses_; return false; }
        if (index) *index = i;
        ent_[(size_t) i].expert = -1;
        ++hits_;
        return true;
    }
    /// The index stays readable until the next stage() writes the slot: consume() frees the entry, the caller copies at once.

    void note_not_landed() { ++not_landed_; }

    /// A planned copy that could not be issued (no host pointer for the expert, or the copy call failed): the entry is dropped again.
    void unstage(int64_t layer, int32_t expert) {
        const int64_t i = find(layer, expert);
        if (i < 0) return;
        ent_[(size_t) i].expert = -1;
        if (staged_ > 0) --staged_;
    }

    /// The tier's selection rule on the predicted counts of one layer, run at the end of the round.  `res` = the layer's residency
    /// after this round's swaps (< 0: not in VRAM; the swapped-in experts count as resident), `mincand` / `margin` the tier's floor
    /// and --adapt-min-gain.  The prediction of expert e is decay x count + this window's increment.  Appends (gain, layer, expert).
    void predict_layer(int64_t layer, const int32_t* res, float mincand, float margin, std::vector<Pred>& out) const {
        if (!p_.on || last_.empty()) return;
        const float* u = last_.data() + layer * E_;
        const float* d = delta_.data() + layer * E_;
        std::vector<std::pair<float, int32_t>> cand, vict;
        for (int32_t e = 0; e < (int32_t) E_; ++e) {
            const float up = decay_ * u[e] + d[e];
            if (res[e] < 0) { if (up >= mincand) cand.emplace_back(up, e); }
            else vict.emplace_back(up, e);
        }
        if (cand.empty() || vict.empty()) return;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (std::ptrdiff_t) nc, vict.end(), [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + margin) break;
            const float gain = cand[i].first - vict[i].first;
            if (gain >= p_.conf) out.push_back({gain, (int32_t) layer, cand[i].second});
        }
    }

    /// The round's staging plan from all layers' predictions: best gain first, at most max_copies NEW copies.  A predicted expert
    /// that is already staged keeps its slot (kept alive, no copy).  A full layer drops its oldest entry.  Returns the copies to
    /// issue (layer, expert, slot in the layer).
    std::vector<Staged> plan(std::vector<Pred>& preds) {
        std::vector<Staged> out;
        if (!p_.on) return out;
        std::stable_sort(preds.begin(), preds.end(), [](const Pred& a, const Pred& b) { return a.gain > b.gain; });
        for (const Pred& pr : preds) {
            if ((int) out.size() >= p_.max_copies) break;
            const int64_t at = find(pr.layer, pr.expert);
            if (at >= 0) { ent_[(size_t) at].round = (int32_t) round_; continue; }
            int pick = -1;
            for (int s = 0; s < p_.slots; ++s)
                if (ent_[(size_t) (pr.layer * p_.slots + s)].expert < 0) { pick = s; break; }
            if (pick < 0) {   // full: the oldest goes (never one staged by this very plan: it carries this round)
                int32_t oldest = (int32_t) round_;
                for (int s = 0; s < p_.slots; ++s) {
                    const Entry& e = ent_[(size_t) (pr.layer * p_.slots + s)];
                    if (e.round < oldest) { oldest = e.round; pick = s; }
                }
                if (pick < 0) continue;
                ++wasted_;
            }
            Entry& e = ent_[(size_t) (pr.layer * p_.slots + pick)];
            e.expert = pr.expert;
            e.round = (int32_t) round_;
            out.push_back({pr.layer, pr.expert, pick});
            ++staged_;
        }
        return out;
    }

    void maybe_log() const {
        if (!p_.on || round_ == 0) return;
        if (round_ != 1 && round_ != 64 && round_ != 256 && (round_ & 1023) != 0) return;
        const uint64_t inc = hits_ + misses_;
        std::fprintf(stderr, "strata serve: spec copy: round %llu, %.1f us per round, staged %llu, incoming %llu of them staged %llu (%.1f%%), "
                             "staged copies unused %llu, staged but not landed %llu\n",
                     (unsigned long long) round_, mean_us(), (unsigned long long) staged_, (unsigned long long) inc,
                     (unsigned long long) hits_, inc > 0 ? 100.0 * (double) hits_ / (double) inc : 0.0,
                     (unsigned long long) wasted_, (unsigned long long) not_landed_);
    }

private:
    struct Entry { int32_t expert = -1; int32_t round = 0; };
    Params p_;
    int64_t L_ = 0, E_ = 0;
    float decay_ = 0.7f;
    std::vector<float> last_, delta_;
    std::vector<Entry> ent_;
    uint64_t round_ = 0, staged_ = 0, hits_ = 0, misses_ = 0, wasted_ = 0, not_landed_ = 0, ns_ = 0;
};

}  // namespace strata::program
