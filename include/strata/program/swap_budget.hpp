#pragma once
// MOD_PLAN item 53 (Pie, UC Berkeley), STRATA_ADAPT_SWAPS_AUTO=1 (opt-in): the adaptive tier's swaps per round chosen
// online instead of a fixed --adapt-swaps.  A hill climb on the decode window time: hold a budget for `period` rounds,
// then try the next one up (x1.5); keep it while the mean window time is no worse than the best seen by more than `tol`,
// else step back and wait longer before trying again.  Bounds [lo, hi]; starts at --adapt-swaps.  Placement only.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>

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
        return b;
    }
    bool on() const { return on_; }
    int swaps() const { return cur_; }

    /// One adaptive round passed, covering `windows` decode windows.
    void round(int windows) {
        if (!on_) return;
        const auto now = std::chrono::steady_clock::now();
        if (started_) {
            sum_ms_ += std::chrono::duration<double, std::milli>(now - t_).count();
            win_ += std::max(1, windows);
        }
        t_ = now;
        started_ = true;
        if (++rounds_ < period_) return;
        const double ms = sum_ms_ / std::max(1, win_);
        rounds_ = 0; sum_ms_ = 0; win_ = 0;
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
    bool on_ = false, started_ = false;
    int cur_ = 32, best_ = 32, lo_ = 8, hi_ = 192;
    int period_ = 32, rounds_ = 0, win_ = 0, wait_ = 1, hold_ = 0;
    double sum_ms_ = 0, best_ms_ = 0, tol_ = 0.01;
    std::chrono::steady_clock::time_point t_{};
};

}  // namespace strata::program
