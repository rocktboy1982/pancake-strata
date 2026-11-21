// STRATA_TUNE_PAIRED (LEADS.md Part Q7): the paired sequential decision (include/strata/core/paired_decision.hpp) and its use in the
// swaps-per-round climb (include/strata/program/swap_budget.hpp).  No GPU, no engine.
//   1. decision routine: A/A simulation (the false-move rate), a true gain moves, a gain under the minimum effect rarely moves,
//      the default outcome is keep, the pair cap keeps
//   2. default path: with STRATA_TUNE_PAIRED unset, SwapBudget gives the same swaps() after every round as a frozen copy of the
//      climb as it was before the switch existed (RefBudget below), over random window times
//   3. the paired climb: noise only -> the budget does not walk; a clear gain at a higher budget -> it moves there
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "strata/core/paired_decision.hpp"
#include "strata/program/swap_budget.hpp"

using namespace strata::core;
using strata::program::SwapBudget;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

static void set_env(const char* k, const char* v) {
#ifdef _WIN32
    _putenv_s(k, v == nullptr ? "" : v);
#else
    if (v == nullptr) unsetenv(k); else setenv(k, v, 1);
#endif
}

// ---- a frozen copy of SwapBudget's climb as it was before STRATA_TUNE_PAIRED (the clock given, as round_at does) ----
class RefBudget {
public:
    RefBudget(int start, int hi_cap) { cur_ = best_ = std::max(1, start); hi_ = std::max(cur_, hi_cap); }
    int swaps() const { return cur_; }
    void round_at(std::chrono::steady_clock::time_point now, int windows) {
        if (started_) {
            sum_ms_ += std::chrono::duration<double, std::milli>(now - t_).count();
            win_ += std::max(1, windows);
        }
        t_ = now;
        started_ = true;
        if (++rounds_ < period_) return;
        const double ms = sum_ms_ / std::max(1, win_);
        rounds_ = 0; sum_ms_ = 0; win_ = 0;
        if (best_ms_ <= 0 || cur_ == best_) {
            best_ms_ = best_ms_ <= 0 ? ms : (cur_ == best_ ? 0.5 * (best_ms_ + ms) : best_ms_);
        } else if (ms <= best_ms_ * (1.0 + tol_)) {
            best_ = cur_;
            best_ms_ = ms;
            wait_ = 1;
        } else {
            cur_ = best_;
            wait_ = std::min(wait_ * 2, 16);
            hold_ = wait_;
            return;
        }
        if (hold_ > 0) { --hold_; return; }
        const int next = std::min(hi_, std::max(cur_ + 1, cur_ * 3 / 2));
        if (next != cur_) cur_ = next;
    }
private:
    bool started_ = false;
    int cur_ = 32, best_ = 32, hi_ = 192;
    int period_ = 32, rounds_ = 0, win_ = 0, wait_ = 1, hold_ = 0;
    double sum_ms_ = 0, best_ms_ = 0, tol_ = 0.01;
    std::chrono::steady_clock::time_point t_{};
};

using Clock = std::chrono::steady_clock;
static Clock::time_point at_ms(double ms) {
    return Clock::time_point(std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(ms)));
}

int main() {
    // ---- 1. the decision routine
    {
        const PairedRule r = paired_rule(0.03);
        CHECK(std::fabs(r.delta - std::log(1.03)) < 1e-12 && r.min_pairs == 12 && r.cap == 160);
        // no decision before min_pairs, however clear
        PairedStats s;
        for (int i = 0; i < 11; ++i) s.add(0.5 + 0.001 * i);
        CHECK(paired_decide(s, r) == PairedVerdict::Continue);
        s.add(0.5);
        CHECK(paired_decide(s, r) == PairedVerdict::Move);
        // paired_diff_ms: lower time = positive d; bad times give no pair
        double d = 0;
        CHECK(paired_diff_ms(50.0, 40.0, d) && d > 0.22 && d < 0.23);
        CHECK(!paired_diff_ms(0.0, 40.0, d) && !paired_diff_ms(50.0, -1.0, d) && !paired_diff_ms(50.0, NAN, d));
        CHECK(!paired_on_from(nullptr) && !paired_on_from("") && !paired_on_from("0") && paired_on_from("1"));
        // the cap keeps
        PairedStats c;
        PairedRule rc = paired_rule(0.03, 12, 20);
        std::mt19937 g(7);
        std::normal_distribution<double> nd(0.03, 0.5);   // right at delta, huge noise: undecided until the cap
        PairedVerdict v = PairedVerdict::Continue;
        for (int i = 0; i < 20 && v == PairedVerdict::Continue; ++i) { c.add(nd(g)); v = paired_decide(c, rc); }
        CHECK(v == PairedVerdict::Keep);
    }
    // A/A: the false-move rate stays under 5 % (10,000 tests per sigma); a true 6 % gain moves in >= 80 % of tests
    {
        const double sigmas[] = {0.02, 0.05, 0.10};
        const PairedRule r = paired_rule(0.03);
        for (double sigma : sigmas) {
            std::mt19937 g(12345);
            std::normal_distribution<double> noise(0.0, sigma), gain(std::log(1.06), sigma), sub(std::log(1.02), sigma);
            int fm = 0, mv = 0, sm = 0;
            const int tests = 10000;
            for (int t = 0; t < tests; ++t) {
                for (int mode = 0; mode < 3; ++mode) {
                    PairedStats s;
                    PairedVerdict v = PairedVerdict::Continue;
                    for (long long i = 0; i < r.cap && v == PairedVerdict::Continue; ++i) {
                        s.add(mode == 0 ? noise(g) : mode == 1 ? gain(g) : sub(g));
                        v = paired_decide(s, r);
                    }
                    if (v == PairedVerdict::Move) (mode == 0 ? fm : mode == 1 ? mv : sm)++;
                }
            }
            std::printf("paired A/A sigma %.2f: false moves %.2f %%, 6%% gain moves %.1f %%, 2%% gain (under the minimum) moves %.2f %%\n",
                        sigma, 100.0 * fm / tests, 100.0 * mv / tests, 100.0 * sm / tests);
            CHECK(fm <= tests / 20);
            CHECK(mv >= tests * 8 / 10);
            CHECK(sm <= tests / 10);   // a real 2 % gain under the 3 % minimum effect may move now and then; it is no false move
        }
    }
    // ---- 2. the default path is the old climb
    {
        set_env("STRATA_ADAPT_SWAPS_AUTO", "1");
        set_env("STRATA_TUNE_PAIRED", nullptr);
        SwapBudget nb = SwapBudget::from_env(32, 96);
        CHECK(nb.on() && !nb.paired());
        RefBudget rb(32, 96);
        std::mt19937 g(99);
        std::uniform_real_distribution<double> ms(20.0, 90.0);
        double t = 0;
        int differs = 0, moves = 0, last = nb.swaps();
        for (int i = 0; i < 20000; ++i) {
            t += ms(g);
            const int w = 1 + (int) (g() % 4);
            nb.round_at(at_ms(t), w);
            rb.round_at(at_ms(t), w);
            if (nb.swaps() != rb.swaps()) ++differs;
            if (nb.swaps() != last) { ++moves; last = nb.swaps(); }
        }
        CHECK(differs == 0);
        CHECK(moves > 3);   // the comparison is not vacuous: the old climb moves on this input
        // switch off: round_at does nothing and swaps() stays the start
        set_env("STRATA_ADAPT_SWAPS_AUTO", nullptr);
        SwapBudget off = SwapBudget::from_env(32, 96);
        off.round_at(at_ms(1.0), 1);
        off.round_at(at_ms(900000.0), 1);
        CHECK(!off.on() && off.swaps() == 32);
        // STRATA_TUNE_PAIRED without STRATA_ADAPT_SWAPS_AUTO does nothing either
        set_env("STRATA_TUNE_PAIRED", "1");
        SwapBudget lone = SwapBudget::from_env(32, 96);
        CHECK(!lone.on() && !lone.paired());
    }
    // ---- 3. the paired climb
    {
        set_env("STRATA_ADAPT_SWAPS_AUTO", "1");
        set_env("STRATA_TUNE_PAIRED", "1");
        set_env("STRATA_TUNE_PAIRED_PERIOD", "4");
        // 3a. noise only (window time independent of the budget): never moves in 5 runs of 20,000 rounds
        for (int run = 0; run < 5; ++run) {
            SwapBudget b = SwapBudget::from_env(32, 96);
            CHECK(b.on() && b.paired());
            std::mt19937 g(1000 + run);
            std::normal_distribution<double> noise(50.0, 2.5);   // 5 % per-round noise
            double t = 0;
            int max_dev = 0;
            for (int i = 0; i < 20000; ++i) {
                t += noise(g);
                b.round_at(at_ms(t), 1);
                max_dev = std::max(max_dev, std::abs(b.swaps() - 32));
            }
            // while a candidate period runs the budget is the candidate, so look at the held value: it must still be 32
            // after the last whole pair; settle with 8 more periods on the base
            CHECK(max_dev > 0);   // candidates are tried (not vacuous)
            // find a base period: step until swaps() == 32
            int guard = 0;
            while (b.swaps() != 32 && guard++ < 100) { t += 50.0; b.round_at(at_ms(t), 1); }
            CHECK(b.swaps() == 32);
        }
        // 3b. a clear gain at a higher budget (window time 50 ms at 32, 40 ms at 48 and above): moves up
        {
            SwapBudget b = SwapBudget::from_env(32, 96);
            std::mt19937 g(4242);
            std::normal_distribution<double> noise(0.0, 1.0);
            double t = 0;
            for (int i = 0; i < 4000; ++i) {
                const double base = b.swaps() >= 48 ? 40.0 : 50.0;
                t += base + noise(g);
                b.round_at(at_ms(t), 1);
            }
            CHECK(b.swaps() >= 48);
        }
        set_env("STRATA_TUNE_PAIRED_PERIOD", nullptr);
        set_env("STRATA_TUNE_PAIRED", nullptr);
        set_env("STRATA_ADAPT_SWAPS_AUTO", nullptr);
    }
    if (failed == 0) std::printf("swap_budget_paired_test: ok\n");
    return failed == 0 ? 0 : 1;
}
