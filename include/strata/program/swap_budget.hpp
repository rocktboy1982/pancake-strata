#pragma once
// MOD_PLAN item 53 (Pie, UC Berkeley), STRATA_ADAPT_SWAPS_AUTO=1 (opt-in): the adaptive tier's swaps per round chosen
// online instead of a fixed --adapt-swaps.  A hill climb on the decode window time: hold a budget for `period` rounds,
// then try the next one up (x1.5); keep it while the mean window time is no worse than the best seen by more than `tol`,
// else step back and wait longer before trying again.  Bounds [lo, hi]; starts at --adapt-swaps.  Placement only.
// STRATA_TUNE_PAIRED=1 (LEADS.md Part Q7, opt-in, default off; needs STRATA_ADAPT_SWAPS_AUTO=1): the climb's point comparison is
// replaced by a paired sequential test (include/strata/core/paired_decision.hpp): base period, candidate period, base, candidate ...;
// each candidate period is paired with the base period before it, d = ln(base_ms / cand_ms), and the budget moves only after
// >= 12 pairs with mean - 2 se > ln(1 + STRATA_TUNE_MIN_EFFECT, default 3 %).  STRATA_TUNE_PAIRED_PERIOD = rounds per period
// (default 8).  Unset: the code path below is the one that always ran.
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "strata/core/paired_decision.hpp"

namespace strata::program {

// STRATA_SWAP_BUDGET_RESIDENCY=1 (LEADS.md Part S row S8, default off): a residency-gated swap budget.  radiance (hcharbonnier's
// journal W40c, W7900 gfx1100) swaps 64 experts a step until the experts of 60% of the moveable bytes are resident and a broad
// histogram pass (4,096 selections) has run, then 16 for 512 steps: -21% link bytes per token (537 to 423 MiB), decode -1.75%,
// bit-identical text; a global 16 was +6.3% worse at 92 tokens.  Ours were fixed budgets (T-199 x160, T-248 16 / 64 / 96,
// T-428 offline, T-827 16 / 64) and the online hill climb (T-221, STRATA_ADAPT_SWAPS_AUTO); a schedule keyed to residency was not tried.
// This is the schedule, pure (no engine types), so a ctest covers it:
//   fill : `hi` swaps a round (64);
//   hold : once the resident share >= `gate` AND `sel_min` selections were seen, `lo` swaps (16) for `hold_steps` decode
//          steps (512 windows); when they run out: still resident >= gate -> hold again, else back to fill.
// Two readings of ours, not in the lead's text (unverified against radiance's code): the "resident share" is the share of the
// routing mass (the usage counts) whose expert is resident now, the hit share, because our cache starts full (the profile
// fills it) while radiance fills its cache from empty, so a bytes share would not move here; and what follows the 512 steps is
// the renewal above.  The gate is STRATA_SWAP_BUDGET_RESIDENCY_GATE (default 0.60).  Placement only: the same experts compute.
class ResidencySchedule {
public:
    ResidencySchedule() = default;
    ResidencySchedule(int hi, int lo, double gate, int64_t sel_min, int64_t hold_steps)
        : hi_(hi < 1 ? 1 : hi), lo_(lo < 1 ? 1 : (lo > hi_ ? hi_ : lo)), gate_(gate), sel_min_(sel_min), hold_steps_(hold_steps < 1 ? 1 : hold_steps) {}
    int budget() const { return hold_ > 0 ? lo_ : hi_; }
    bool holding() const { return hold_ > 0; }
    int64_t holds() const { return holds_; }          ///< times a hold started or was renewed
    int64_t selections() const { return sel_; }
    double gate() const { return gate_; }
    /// One adaptive round: `windows` decode steps passed, `share` is the resident share of the routing mass now (0..1),
    /// `new_selections` the routed expert picks seen since the last round.
    void round(int windows, double share, int64_t new_selections) {
        if (new_selections > 0) sel_ += new_selections;
        if (windows < 1) windows = 1;
        if (hold_ > 0) {
            hold_ -= windows;
            if (hold_ <= 0) {
                hold_ = 0;
                if (share >= gate_) { hold_ = hold_steps_; ++holds_; }   // still resident: hold again, else fill
            }
            return;
        }
        if (share >= gate_ && sel_ >= sel_min_) { hold_ = hold_steps_; ++holds_; }
    }
private:
    int hi_ = 64, lo_ = 16;
    double gate_ = 0.60;
    int64_t sel_min_ = 4096, hold_steps_ = 512;
    int64_t hold_ = 0, holds_ = 0, sel_ = 0;
};

/// The resident share of the routing mass: sum of usage[i] over the experts resident now (host_res[i] >= 0) over the sum
/// of all usage[i] > 0; 0 when nothing was routed.  `usage` and `host_res` are indexed layer * n_expert + expert.
inline double resident_mass_share(const float* usage, const int32_t* host_res, size_t n) {
    double tot = 0.0, res = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double u = usage[i];
        if (!(u > 0.0)) continue;   // (also drops NaN)
        tot += u;
        if (host_res[i] >= 0) res += u;
    }
    return tot > 0.0 ? res / tot : 0.0;
}

/// STRATA_SWAP_BUDGET_RESIDENCY: only the text "1" switches it on (default off).
inline bool swap_budget_residency_parse(const char* v) { return v != nullptr && v[0] == '1' && v[1] == '\0'; }
/// STRATA_SWAP_BUDGET_RESIDENCY_GATE: a number in (0, 1], else the default 0.60.
inline double swap_budget_gate_parse(const char* v) {
    if (v == nullptr || v[0] == '\0') return 0.60;
    char* end = nullptr;
    const double d = std::strtod(v, &end);
    return (end == v || *end != '\0' || !(d > 0.0) || d > 1.0) ? 0.60 : d;
}

class SwapBudget {
public:
    /// `sel_per_window`: routed expert picks of one decode window at least (layers x experts per token), the unit of the
    /// residency schedule's histogram pass; used only with STRATA_SWAP_BUDGET_RESIDENCY.
    static SwapBudget from_env(int start, int hi_cap, int64_t sel_per_window = 0) {
        SwapBudget b;
        const char* v = std::getenv("STRATA_ADAPT_SWAPS_AUTO");
        b.on_ = v != nullptr && v[0] == '1';
        b.cur_ = b.best_ = std::max(1, start);
        b.hi_ = std::max(b.cur_, hi_cap);
        if (b.on_)
            std::fprintf(stderr, "strata serve: STRATA_ADAPT_SWAPS_AUTO: swaps per round chosen online in [%d, %d], from %d\n",
                         b.lo_, b.hi_, b.cur_);
        if (swap_budget_residency_parse(std::getenv("STRATA_SWAP_BUDGET_RESIDENCY"))) {   // S8 (default off)
            if (b.on_) {
                std::fprintf(stderr, "strata serve: STRATA_SWAP_BUDGET_RESIDENCY is ignored beside STRATA_ADAPT_SWAPS_AUTO\n");
            } else {
                b.res_on_ = true;
                b.sel_per_window_ = std::max<int64_t>(1, sel_per_window);
                b.res_ = ResidencySchedule(std::min(64, b.hi_), 16, swap_budget_gate_parse(std::getenv("STRATA_SWAP_BUDGET_RESIDENCY_GATE")),
                                           4096, 512);
                std::fprintf(stderr, "strata serve: STRATA_SWAP_BUDGET_RESIDENCY: %d swaps a round until %.0f%% of the routing mass is resident and "
                                     "4096 selections ran, then 16 for 512 steps\n", std::min(64, b.hi_), 100.0 * b.res_.gate());
            }
        }
        if (b.on_ && strata::core::paired_on_from(std::getenv("STRATA_TUNE_PAIRED"))) {   // LEADS.md Q7
            b.paired_ = true;
            if (const char* p = std::getenv("STRATA_TUNE_PAIRED_PERIOD")) b.ppd_ = std::max(1, std::atoi(p));
            double me = 0.03;
            if (const char* m = std::getenv("STRATA_TUNE_MIN_EFFECT")) {
                const double x = std::atof(m);
                if (x > 0.0 && std::isfinite(x)) me = x;
            }
            b.prule_ = strata::core::paired_rule(me);
            std::fprintf(stderr, "strata serve: STRATA_TUNE_PAIRED=1: the swaps climb decides on paired periods of %d rounds: "
                                 "move only when faster by %.1f %% (95 %% one-sided), >= %lld pairs, cap %lld; default keep\n",
                         b.ppd_, me * 100.0, b.prule_.min_pairs, b.prule_.cap);
        }
        return b;
    }
    bool on() const { return on_ || res_on_; }
    bool residency_on() const { return res_on_; }
    bool paired() const { return paired_; }
    int swaps() const { return res_on_ ? res_.budget() : cur_; }
    const ResidencySchedule& residency() const { return res_; }
    /// S8: the resident share of the routing mass, read by the engine before `round` (no-op unless the switch is on).
    void observe_residency(const float* usage, const int32_t* host_res, size_t n) {
        if (!res_on_) return;
        share_ = resident_mass_share(usage, host_res, n);
    }

    /// One adaptive round passed, covering `windows` decode windows.
    void round(int windows) {
        if (res_on_) {
            const bool was = res_.holding();
            res_.round(windows, share_, (int64_t) std::max(1, windows) * sel_per_window_);
            if (res_.holding() != was)
                std::fprintf(stderr, "strata serve: STRATA_SWAP_BUDGET_RESIDENCY: %s (resident share %.3f, %lld selections), %d swaps a round\n",
                             res_.holding() ? "hold" : "fill", share_, (long long) res_.selections(), res_.budget());
            return;
        }
        if (!on_) return;
        round_at(std::chrono::steady_clock::now(), windows);
    }

    /// The same with the clock given (the tests drive it).
    void round_at(std::chrono::steady_clock::time_point now, int windows) {
        if (!on_) return;
        if (started_) {
            sum_ms_ += std::chrono::duration<double, std::milli>(now - t_).count();
            win_ += std::max(1, windows);
        }
        t_ = now;
        started_ = true;
        if (++rounds_ < (paired_ ? ppd_ : period_)) return;
        const double ms = sum_ms_ / std::max(1, win_);
        rounds_ = 0; sum_ms_ = 0; win_ = 0;
        if (paired_) { paired_period(ms); return; }   // STRATA_TUNE_PAIRED
        if (best_ms_ <= 0 || cur_ == best_) {           // the held budget's measurement
            best_ms_ = best_ms_ <= 0 ? ms : (cur_ == best_ ? 0.5 * (best_ms_ + ms) : best_ms_);
        } else if (ms <= best_ms_ * (1.0 + tol_)) {     // the trial held: adopt it
            best_ = cur_;
            best_ms_ = ms;
            wait_ = 1;
        } else {                                        // the trial lost: back, and wait longer before the next one
            cur_ = best_;
            wait_ = std::min(wait_ * 2, 16);
            hold_ = wait_;
            return;
        }
        if (hold_ > 0) { --hold_; return; }
        const int next = std::min(hi_, std::max(cur_ + 1, cur_ * 3 / 2));
        if (next != cur_) cur_ = next;                  // the next trial
    }

private:
    /// A period ended with `ms` per window.  Base period: remember it and try the candidate; candidate period: one pair.
    void paired_period(double ms) {
        if (phold_ > 0) { --phold_; return; }                    // a rest after two keeps in a row
        if (pcand_ == 0) {                                       // the base period (cur_ held)
            pbase_ms_ = ms;
            int next = pdir_ > 0 ? std::min(hi_, std::max(cur_ + 1, cur_ * 3 / 2)) : std::max(lo_, std::min(cur_ - 1, cur_ * 2 / 3));
            if (next == cur_) { pdir_ = -pdir_; next = pdir_ > 0 ? std::min(hi_, std::max(cur_ + 1, cur_ * 3 / 2))
                                                                  : std::max(lo_, std::min(cur_ - 1, cur_ * 2 / 3)); }
            if (next == cur_) return;                            // no room either way
            pcand_ = next;
            pbase_v_ = cur_;
            cur_ = next;
            return;
        }
        double d = 0.0;                                          // the candidate period: pair it with the base before it
        if (strata::core::paired_diff_ms(pbase_ms_, ms, d)) pstats_.add(d);
        cur_ = pbase_v_;
        const int cand = pcand_;
        pcand_ = 0;
        const char* why = "";
        const strata::core::PairedVerdict v = strata::core::paired_decide(pstats_, prule_, &why);
        if (v == strata::core::PairedVerdict::Continue) return;   // another pair: the next base period picks the same candidate again
        std::fprintf(stderr, "strata serve: STRATA_TUNE_PAIRED: swaps %d vs %d: pairs %lld, mean d %+.4f (se %.4f) -> %s (%s)\n",
                     pbase_v_, cand, pstats_.n, pstats_.mean, pstats_.se(),
                     v == strata::core::PairedVerdict::Move ? "move" : "keep", why);
        if (v == strata::core::PairedVerdict::Move) {
            cur_ = best_ = cand;
            pfail_ = 0;
        } else {
            pdir_ = -pdir_;                                      // the other direction next
            if (++pfail_ >= 2) { pfail_ = 0; phold_ = 16; }     // both lost: rest, then try again
        }
        pstats_.reset();
    }

    bool on_ = false, started_ = false;
    bool res_on_ = false;               // STRATA_SWAP_BUDGET_RESIDENCY (S8)
    ResidencySchedule res_;
    double share_ = 0.0;
    int64_t sel_per_window_ = 1;
    int cur_ = 32, best_ = 32, lo_ = 8, hi_ = 192;
    int period_ = 32, rounds_ = 0, win_ = 0, wait_ = 1, hold_ = 0;
    double sum_ms_ = 0, best_ms_ = 0, tol_ = 0.01;
    std::chrono::steady_clock::time_point t_{};
    // STRATA_TUNE_PAIRED state (untouched while it is off)
    bool paired_ = false;
    int ppd_ = 8, pcand_ = 0, pbase_v_ = 0, pdir_ = 1, pfail_ = 0, phold_ = 0;
    double pbase_ms_ = 0;
    strata::core::PairedRule prule_;
    strata::core::PairedStats pstats_;
};

}  // namespace strata::program
