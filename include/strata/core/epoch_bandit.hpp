#pragma once
// MOD_PLAN A44 (Athena, ETH SAFARI 2026, https://arxiv.org/abs/2601.17615: an online agent that, epoch by epoch, turns
// a prefetcher and an off-chip predictor on or off from the measured system reward).  STRATA_BANDIT=<switches> (opt-in,
// default off): the "mixed" opt-ins - the ones that won on code edits and lost on short chat in the A/Bs - are turned on
// or off per window TYPE by an epoch bandit in the serve loop, instead of being fixed for the process.
//
//   switch   its own env var (must be set as well)   exactness against the switch off
//   top      STRATA_VERIFY_DEFER_CPU_TOP=<n>          LOSSY (small): which CPU rows join one layer late (A4: +20.1% edit, -6.9% short)
//   kmiss    STRATA_ADAPT_KMISS=k,W                   exact up to rounding: placement only (GPU and CPU experts round differently)
//   preread  STRATA_POOL_PREREAD=1                    bitwise exact: data movement only
//
// An arm is a subset of the switches (2^n arms, "off" included).  Each window type has its own epochs: an epoch is
// `epoch` counted windows that all run with one arm; the first `skip` windows of an epoch are not counted (a switch's
// after-effects: the adaptive tier's swaps land in the next windows, the previous arm's pre-reads are still in the
// caches).  At the end of an epoch its cost - the window time per committed token, summed over its counted windows - is
// folded into the arm's discounted mean (every arm of the type decays by `decay` per epoch: sessions drift) and the next
// arm is picked: each arm once in order ("off" first), then epsilon-greedy (STRATA_BANDIT_RULE=eps:<e>, the default
// eps:0.1) or UCB1 for a cost (STRATA_BANDIT_RULE=ucb[:c], c = 0.1, scaled by the mean cost so it has no unit; the
// bonus counts undiscounted epochs).  The arms change
// only between windows: the serve loop sets the next window's arm before the window is staged.
//
// With a lossy or rounding arm the text depends on which arm ran which window, i.e. on timings: it is not reproducible
// run to run.  STRATA_BANDIT_EXACT=1 drops the lossy switch (top), =2 the rounding one too (kmiss): only bitwise-exact
// arms are left, and the text is the switches-off text.
//
// Window types (STRATA_BANDIT_TYPE, known before the window runs):
//   req[:P]   (default, P = 30)  the request's recent lookup share (EWMA 1/8 over its windows, 0 at its start) >= P%:
//             "copy" (a code edit that repeats its input) or "chat" - the A4/A5 split, measured online
//   src       this window's own draft source: a prompt-lookup / chained window or a drafter (MTP / DFlash2) window
//   size[:N]  (N = 4) the window's tokens > N: "wide" or "narrow"
//   ctx[:N]   (N = 4096) the window's position >= N: "long" or "short" context
//   one       a single type
// A switch whose effect outlives its window (kmiss: the swaps; preread) is better judged with req / one than with src
// or size, whose types can alternate window by window.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::core {

enum BanditSwitch : uint32_t { kBanditTop = 1u, kBanditKmiss = 2u, kBanditPreread = 4u, kBanditAll = 7u };

/// The exactness class of one switch against the switch off: 0 bitwise exact, 1 exact up to rounding, 2 lossy.
inline int bandit_switch_class(uint32_t s) { return s == kBanditTop ? 2 : s == kBanditKmiss ? 1 : 0; }
inline const char* bandit_switch_name(uint32_t s) {
    return s == kBanditTop ? "top" : s == kBanditKmiss ? "kmiss" : s == kBanditPreread ? "preread" : "?";
}
/// "off", or the switches of the arm joined by '+' (top+kmiss).
inline std::string bandit_arm_name(uint32_t arm) {
    if (arm == 0) return "off";
    std::string s;
    for (uint32_t b = 1; b <= kBanditPreread; b <<= 1)
        if (arm & b) {
            if (!s.empty()) s += '+';
            s += bandit_switch_name(b);
        }
    return s;
}
/// The worst exactness class among the arm's switches (0 for "off").
inline int bandit_arm_class(uint32_t arm) {
    int c = 0;
    for (uint32_t b = 1; b <= kBanditPreread; b <<= 1)
        if ((arm & b) && bandit_switch_class(b) > c) c = bandit_switch_class(b);
    return c;
}

enum BanditTypeMode : int { kTypeOne = 0, kTypeSrc = 1, kTypeSize = 2, kTypeReq = 3, kTypeCtx = 4 };

struct BanditConfig {
    uint32_t requested = 0;   ///< the switches STRATA_BANDIT names (before availability and STRATA_BANDIT_EXACT)
    uint32_t switches = 0;    ///< the switches the bandit toggles: requested, available, within the exactness limit
    int exact = 0;            ///< STRATA_BANDIT_EXACT: 0 every arm, 1 no lossy switch, 2 bitwise-exact switches only
    int epoch = 16;           ///< STRATA_BANDIT_EPOCH: counted windows per epoch (2..4096)
    int skip = 2;             ///< STRATA_BANDIT_SKIP: windows at an epoch's start that are not counted (0..epoch)
    bool ucb = false;         ///< STRATA_BANDIT_RULE=ucb[:c] (else eps[:e])
    double eps = 0.1;
    double ucb_c = 0.1;
    double decay = 0.9;       ///< STRATA_BANDIT_DECAY: per-epoch discount of every arm's statistics, (0, 1]
    uint64_t seed = 1;        ///< STRATA_BANDIT_SEED: the explorer's random stream (deterministic)
    int type_mode = kTypeReq; ///< STRATA_BANDIT_TYPE
    double type_arg = 0.30;
    bool log = false;         ///< STRATA_BANDIT_LOG=1: one stderr line per epoch decision
    bool on() const { return switches != 0; }
};

/// The STRATA_BANDIT* variables through `get` (a getenv), the bandit's switches limited to `available` (the ones whose
/// own env var is on and that this engine can toggle).  `msg` (may be null) collects what was dropped and why.
template <class Get>
BanditConfig bandit_config(Get get, uint32_t available, std::string* msg) {
    BanditConfig c;
    auto say = [&](const std::string& s) {
        if (msg != nullptr) *msg += (msg->empty() ? "" : "; ") + s;
    };
    const char* v = get("STRATA_BANDIT");
    if (v == nullptr || v[0] == '\0') return c;
    {
        std::string list(v);
        size_t at = 0;
        while (at <= list.size()) {
            const size_t comma = list.find(',', at);
            const std::string w = list.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
            if (w == "top" || w == "rank") c.requested |= kBanditTop;
            else if (w == "kmiss") c.requested |= kBanditKmiss;
            else if (w == "preread" || w == "pre") c.requested |= kBanditPreread;
            else if (w == "all") c.requested |= kBanditAll;
            else if (!(w.empty() || w == "0" || w == "off")) say("unknown switch '" + w + "' (top, kmiss, preread, all)");
            if (comma == std::string::npos) break;
            at = comma + 1;
        }
    }
    if (const char* e = get("STRATA_BANDIT_EXACT")) c.exact = std::atoi(e) >= 2 ? 2 : std::atoi(e) >= 1 ? 1 : 0;
    uint32_t s = c.requested;
    for (uint32_t b = 1; b <= kBanditPreread; b <<= 1) {
        if (!(s & b)) continue;
        if (bandit_switch_class(b) > 2 - c.exact) {
            say(std::string(bandit_switch_name(b)) + " dropped: STRATA_BANDIT_EXACT=" + std::to_string(c.exact) +
                (bandit_switch_class(b) == 2 ? " (lossy)" : " (rounding)"));
            s &= ~b;
        } else if (!(available & b)) {
            say(std::string(bandit_switch_name(b)) + " dropped: its own switch is not on here");
            s &= ~b;
        }
    }
    c.switches = s;
    if (const char* e = get("STRATA_BANDIT_EPOCH")) {
        const int n = std::atoi(e);
        if (n >= 2 && n <= 4096) c.epoch = n;
        else say("STRATA_BANDIT_EPOCH=" + std::string(e) + " is not 2..4096: 16");
    }
    if (const char* e = get("STRATA_BANDIT_SKIP")) {
        const int n = std::atoi(e);
        if (n >= 0 && n <= c.epoch) c.skip = n;
        else say("STRATA_BANDIT_SKIP=" + std::string(e) + " is not 0..epoch: 2");
    }
    if (c.skip > c.epoch) c.skip = c.epoch;
    if (const char* e = get("STRATA_BANDIT_RULE")) {
        const char* colon = std::strchr(e, ':');
        const double x = colon != nullptr ? std::atof(colon + 1) : -1.0;
        if (std::strncmp(e, "ucb", 3) == 0) {
            c.ucb = true;
            if (colon != nullptr && x >= 0.0) c.ucb_c = x;
        } else if (std::strncmp(e, "eps", 3) == 0) {
            if (colon != nullptr && x >= 0.0 && x <= 1.0) c.eps = x;
        } else {
            say("STRATA_BANDIT_RULE=" + std::string(e) + " is not eps[:e] or ucb[:c]: eps:0.1");
        }
    }
    if (const char* e = get("STRATA_BANDIT_DECAY")) {
        const double d = std::atof(e);
        if (d > 0.0 && d <= 1.0) c.decay = d;
        else say("STRATA_BANDIT_DECAY=" + std::string(e) + " is not in (0, 1]: 0.9");
    }
    if (const char* e = get("STRATA_BANDIT_SEED")) c.seed = std::strtoull(e, nullptr, 10);
    if (const char* e = get("STRATA_BANDIT_TYPE")) {
        const char* colon = std::strchr(e, ':');
        const std::string m(e, colon != nullptr ? (size_t) (colon - e) : std::strlen(e));
        const double x = colon != nullptr ? std::atof(colon + 1) : -1.0;
        if (m == "one") c.type_mode = kTypeOne;
        else if (m == "src") c.type_mode = kTypeSrc;
        else if (m == "size") { c.type_mode = kTypeSize; c.type_arg = x >= 1.0 ? x : 4.0; }
        else if (m == "req") { c.type_mode = kTypeReq; c.type_arg = x > 0.0 && x <= 100.0 ? x / 100.0 : 0.30; }
        else if (m == "ctx") { c.type_mode = kTypeCtx; c.type_arg = x >= 1.0 ? x : 4096.0; }
        else say("STRATA_BANDIT_TYPE=" + std::string(e) + " is not one, src, size[:N], req[:P] or ctx[:N]: req:30");
    }
    if (const char* e = get("STRATA_BANDIT_LOG")) c.log = e[0] == '1';
    return c;
}

/// STRATA_BANDIT and STRATA_BANDIT_EXACT alone, read once: may the bandit own `sw`?  (verify.cpp: the rank guard's mapped
/// word is kept only then; availability is the caller's own test.)
inline bool bandit_requested(uint32_t sw) {
    static const uint32_t req = bandit_config([](const char* n) { return std::getenv(n); }, kBanditAll, nullptr).switches;
    return (req & sw) != 0;
}

/// The window's type, known before it runs (see the header comment).
class BanditTyper {
public:
    BanditTyper() = default;
    BanditTyper(int mode, double arg) : mode_(mode), arg_(arg) {}
    int types() const { return mode_ == kTypeOne ? 1 : 2; }
    /// A request starts: its lookup share starts at 0.
    void begin_request() { share_ = 0.0; }
    /// `lookup`: the window is a prompt-lookup or chained one; `T` its tokens; `pos` its first position.
    int type(bool lookup, int T, int64_t pos) const {
        switch (mode_) {
            case kTypeSrc: return lookup ? 1 : 0;
            case kTypeSize: return T > (int) arg_ ? 1 : 0;
            case kTypeReq: return share_ >= arg_ ? 1 : 0;
            case kTypeCtx: return pos >= (int64_t) arg_ ? 1 : 0;
            default: return 0;
        }
    }
    /// After a decode window of the request: whether it was a lookup / chained window.
    void note(bool lookup) { share_ += 0.125 * ((lookup ? 1.0 : 0.0) - share_); }
    double share() const { return share_; }
    const char* name(int t) const {
        switch (mode_) {
            case kTypeSrc: return t ? "lookup" : "drafter";
            case kTypeSize: return t ? "wide" : "narrow";
            case kTypeReq: return t ? "copy" : "chat";
            case kTypeCtx: return t ? "long" : "short";
            default: return "all";
        }
    }

private:
    int mode_ = kTypeReq;
    double arg_ = 0.30;
    double share_ = 0.0;
};

/// The epoch bandit: per window type, epochs of `epoch` counted windows, one arm per epoch.  Host only, deterministic for
/// a seed and a sequence of observations (core/epoch_bandit_test.cpp).
class EpochBandit {
public:
    struct Decision {
        int type = 0;
        int64_t epoch = 0;     ///< the type's epochs closed so far (this one included)
        uint32_t arm = 0;      ///< the arm the closed epoch ran
        double cost = 0.0;     ///< its window ms per committed token
        double mean = 0.0;     ///< the arm's discounted mean after it
        int windows = 0;       ///< its counted windows
        uint32_t next = 0;     ///< the arm of the type's next epoch
        const char* why = "";  ///< untried, greedy, explore, ucb
    };

    EpochBandit() = default;
    EpochBandit(const BanditConfig& c, int types) : c_(c), rng_(c.seed) {
        for (uint32_t m = 0; m <= c.switches; ++m)
            if ((m & ~c.switches) == 0) arms_.push_back(m);
        ts_.resize((size_t) (types > 0 ? types : 1));
        for (TypeState& t : ts_) {
            t.n.assign(arms_.size(), 0.0);
            t.sum.assign(arms_.size(), 0.0);
            t.pulls.assign(arms_.size(), 0);
            t.req.assign(arms_.size(), 0);
            t.total.assign(arms_.size(), 0);
        }
    }

    bool on() const { return !arms_.empty() && c_.switches != 0; }
    int n_arms() const { return (int) arms_.size(); }
    int n_types() const { return (int) ts_.size(); }
    uint32_t arm(int i) const { return arms_[(size_t) i]; }
    const BanditConfig& config() const { return c_; }

    /// The arm (a switch mask) a window of type t runs with: its type's current epoch's.
    uint32_t begin(int t) const { return arms_[(size_t) ts_[(size_t) t].cur]; }

    /// After that window: its time (ms) and committed tokens.  True when it closed the type's epoch (last_decision()
    /// then says what it measured and which arm comes next).  A window with no time or no token is not counted.
    bool end(int t, double ms, int tokens) {
        TypeState& s = ts_[(size_t) t];
        if (!(ms > 0.0) || tokens <= 0) return false;
        ++s.req[(size_t) s.cur];
        ++s.total[(size_t) s.cur];
        if (++s.seen <= c_.skip) return false;
        s.ms += ms;
        s.tok += (double) tokens;
        if (++s.counted < c_.epoch) return false;
        const double cost = s.ms / s.tok;
        for (size_t i = 0; i < arms_.size(); ++i) {
            s.n[i] *= c_.decay;
            s.sum[i] *= c_.decay;
        }
        const size_t cur = (size_t) s.cur;
        s.n[cur] += 1.0;
        s.sum[cur] += cost;
        ++s.pulls[cur];
        ++s.epochs;
        d_ = Decision{};
        d_.type = t;
        d_.epoch = s.epochs;
        d_.arm = arms_[cur];
        d_.cost = cost;
        d_.mean = s.sum[cur] / s.n[cur];
        d_.windows = s.counted;
        s.cur = choose(s, &d_.why);
        d_.next = arms_[(size_t) s.cur];
        s.seen = s.counted = 0;
        s.ms = s.tok = 0.0;
        return true;
    }
    const Decision& last_decision() const { return d_; }

    /// The arm's discounted mean cost for type t (0 when never pulled), its epochs, the windows it ran.
    double mean(int t, int i) const {
        const TypeState& s = ts_[(size_t) t];
        return s.n[(size_t) i] > 0.0 ? s.sum[(size_t) i] / s.n[(size_t) i] : 0.0;
    }
    int64_t pulls(int t, int i) const { return ts_[(size_t) t].pulls[(size_t) i]; }
    int64_t windows(int t, int i) const { return ts_[(size_t) t].total[(size_t) i]; }
    int64_t epochs(int t) const { return ts_[(size_t) t].epochs; }

    /// A request starts: its per-arm window counts (request_summary) start at 0.
    void begin_request() {
        for (TypeState& s : ts_) s.req.assign(arms_.size(), 0);
    }
    /// The request's windows by type and arm, no spaces (the DONE line's bandit= field): "chat:off*12,top*30/copy:top*41".
    template <class Typer>
    std::string request_summary(const Typer& typer) const {
        std::string out;
        for (size_t t = 0; t < ts_.size(); ++t) {
            std::string one;
            for (size_t i = 0; i < arms_.size(); ++i) {
                if (ts_[t].req[i] == 0) continue;
                if (!one.empty()) one += ',';
                one += bandit_arm_name(arms_[i]) + "*" + std::to_string(ts_[t].req[i]);
            }
            if (one.empty()) continue;
            if (!out.empty()) out += '/';
            out += std::string(typer.name((int) t)) + ":" + one;
        }
        return out.empty() ? "none" : out;
    }

private:
    struct TypeState {
        int cur = 0;            // index of the current epoch's arm
        int seen = 0;           // windows of the current epoch (skipped ones included)
        int counted = 0;        // its counted windows
        double ms = 0.0, tok = 0.0;
        int64_t epochs = 0;
        std::vector<double> n, sum;     // discounted epoch counts and cost sums
        std::vector<int64_t> pulls;     // epochs per arm
        std::vector<int64_t> req;       // windows per arm in the current request
        std::vector<int64_t> total;     // windows per arm
    };

    uint64_t next() {   // splitmix64
        uint64_t z = (rng_ += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double uniform() { return (double) (next() >> 11) * (1.0 / 9007199254740992.0); }

    int choose(const TypeState& s, const char** why) {
        const int A = (int) arms_.size();
        for (int i = 0; i < A; ++i)
            if (s.pulls[(size_t) i] == 0) {
                *why = "untried";
                return i;
            }
        auto mean_of = [&](int i) { return s.sum[(size_t) i] / s.n[(size_t) i]; };
        int best = 0;
        for (int i = 1; i < A; ++i)
            if (mean_of(i) < mean_of(best)) best = i;
        if (!c_.ucb) {
            if (uniform() < c_.eps) {
                *why = "explore";
                return (int) (next() % (uint64_t) A);
            }
            *why = "greedy";
            return best;
        }
        // UCB1 for a cost: the discounted mean minus c * scale * sqrt(ln(epochs) / epochs of the arm).  The bonus counts
        // the undiscounted epochs: with the discounted counts an arm left alone for a few epochs (0.9^k) would look
        // unexplored again and the rule would keep trying every arm.  scale = the mean of the arm means (c has no unit;
        // an epoch's cost is noisy by a few percent, hence the small default c = 0.1).
        double N = 0.0, scale = 0.0;
        for (int i = 0; i < A; ++i) {
            N += (double) s.pulls[(size_t) i];
            scale += mean_of(i);
        }
        scale /= (double) A;
        const double lnN = std::log(N > 1.0 ? N : 1.0);
        int pick = 0;
        double bs = 0.0;
        for (int i = 0; i < A; ++i) {
            const double sc = mean_of(i) - c_.ucb_c * scale * std::sqrt(lnN / (double) s.pulls[(size_t) i]);
            if (i == 0 || sc < bs) {
                bs = sc;
                pick = i;
            }
        }
        *why = pick == best ? "ucb" : "ucb-explore";
        return pick;
    }

    BanditConfig c_;
    std::vector<uint32_t> arms_;
    std::vector<TypeState> ts_;
    uint64_t rng_ = 1;
    Decision d_;
};

}  // namespace strata::core
