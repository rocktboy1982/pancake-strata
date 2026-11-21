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
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "strata/core/paired_decision.hpp"

namespace strata::program {

class SwapBudget {
public:
    static SwapBudget from_env(int start, int hi_cap) {
        SwapBudget b;
        const char* v = std::getenv("STRATA_ADAPT_SWAPS_AUTO");
        b.on_ = v != nullptr && v[0] == '1';
        b.cur_ = b.best_ = std::max(1, start);
        b.hi_ = std::max(b.cur_, hi_cap);
        if (b.on_)
            std::fprintf(stderr, "strata serve: STRATA_ADAPT_SWAPS_AUTO: swaps per round chosen online in [%d, %d], from %d\n",
                         b.lo_, b.hi_, b.cur_);
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
    bool on() const { return on_; }
    bool paired() const { return paired_; }
    int swaps() const { return cur_; }

    /// One adaptive round passed, covering `windows` decode windows.
    void round(int windows) {
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
