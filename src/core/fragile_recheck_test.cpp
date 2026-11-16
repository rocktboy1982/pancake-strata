// src/core/fragile_recheck_test.cpp - LEADS.md "HotRod" stand-ins: the fragile re-check's rules (core/fragile_recheck.hpp) and the
// stand-in learner's estimate (core/standin_pool.hpp), CPU only.
#include "strata/core/fragile_recheck.hpp"
#include "strata/core/standin_pool.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace strata::core;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    {   // k is learned from the first 64 windows (no cut meanwhile): the 10th percentile of the moved rows' gap / shift
        FragileState s;
        const float shift[3] = {0.0f, 1.0f, 1.0f};
        for (int w = 0; w < FragileState::kLearnWindows; ++w) {
            const float gap[3] = {5.0f, (float) (w % 10) + 1.0f, 100.0f};
            CHECK(fragile_first(s, gap, shift, 3, nullptr) == -1);
        }
        CHECK(s.k_set);
        CHECK(s.k >= 1.0f && s.k <= 2.0f);
    }
    {   // a close call cuts at its row (row 0 never), unmoved rows never cut
        FragileState s;
        s.k_set = true;
        s.k = 2.0f;
        s.eligible = 1;   // not an explore window
        const float gap[4] = {0.1f, 10.0f, 0.5f, 0.1f};
        const float shift[4] = {1.0f, 1.0f, 1.0f, 0.0f};
        bool ex = true;
        CHECK(fragile_first(s, gap, shift, 4, &ex) == 2);
        CHECK(!ex);
        const float shift0[4] = {1.0f, 0.0f, 0.0f, 0.0f};
        s.eligible = 1;
        CHECK(fragile_first(s, gap, shift0, 4, &ex) == -1);
        const float gnan[2] = {1.0f, NAN};
        const float sh2[2] = {0.0f, 1.0f};
        s.eligible = 1;
        CHECK(fragile_first(s, gnan, sh2, 2, nullptr) == 1);   // a NaN gap is a close call
    }
    {   // explore: every 16th eligible window without a normal cut cuts a moved row with ratio in [k, 8k)
        FragileState s;
        s.k_set = true;
        s.k = 1.0f;
        const float gap[2] = {1.0f, 4.0f};
        const float shift[2] = {0.0f, 1.0f};
        bool ex = false;
        CHECK(fragile_first(s, gap, shift, 2, &ex) == 1);   // eligible 0 -> explore
        CHECK(ex);
        CHECK(fragile_first(s, gap, shift, 2, &ex) == -1);
    }
    {   // settle: a flip is a changed top-1; every 50 re-checks k = 1.25 x the largest flipped ratio, bounded to 1.5 x
        FragileState s;
        s.k_set = true;
        s.k = 1.0f;
        s.brake_on = false;
        CHECK(fragile_settle(s, 7) == -1);
        for (int i = 0; i < 50; ++i) {
            fragile_cut(s, i == 3 ? 1.0f : 0.5f, 7, false);
            CHECK(fragile_settle(s, i == 3 ? 8 : 7) == (i == 3 ? 1 : 0));
        }
        CHECK(s.flips == 1 && s.rechecks == 50);
        CHECK(std::fabs(s.k - 1.25f) < 1e-6f);
        FragileState p = s;
        p.k_pinned = true;
        p.k = 3.0f;
        for (int i = 0; i < 50; ++i) { fragile_cut(p, 9.0f, 1, false); fragile_settle(p, 2); }
        CHECK(p.k == 3.0f);
    }
    {   // the brake halves the request's scale on many flips, and grows it back on clean windows
        FragileState s;
        s.k_set = true;
        fragile_request_begin(s);
        for (int i = 0; i < FragileState::kBrakeMin; ++i) { fragile_cut(s, 1.0f, 1, false); fragile_settle(s, 2); }
        CHECK(s.req_scale == 0.5f);
        CHECK(s.req_brakes == 1);
        for (int i = 0; i < FragileState::kBrakeMin; ++i) { fragile_cut(s, 1.0f, 1, false); fragile_settle(s, 1); }
        for (int w = 0; w < FragileState::kBrakeWindows; ++w) fragile_window_done(s, 8);
        CHECK(std::fabs(s.req_scale - 0.625f) < 1e-6f);
        fragile_request_begin(s);
        CHECK(s.req_scale == 1.0f && s.req_cuts == 0);
    }
    {   // the learner's estimate: an empty pair takes its parent's scale once the parent is ready
        using namespace strata::standin;
        Sums eg, lg;
        Pair empty;
        CHECK(!estimate(empty, &eg, &lg).ready);
        for (int i = 0; i < kReady; ++i) add_sample(eg, 0.5, 1.0, 1.0);   // y_e = 0.5 y_s in the mean
        const Est e = estimate(empty, &eg, &lg);
        CHECK(e.ready && e.pooled);
        CHECK(std::fabs(e.c - 0.5) < 1e-9);
        std::vector<float> ys = {1.0f, 2.0f, 3.0f}, ye = {2.0f, 4.0f, 6.0f};
        double a = 0, b = 0, c = 0;
        sample_dots(ys.data(), ye.data(), 3, a, b, c);
        CHECK(a == 28.0 && b == 14.0 && c == 56.0);
        Pair p;
        for (int i = 0; i < kReady; ++i) add_sample(p, a, b, c);
        const Est q = estimate(p, nullptr, nullptr);
        CHECK(q.ready && std::fabs(q.c - 2.0) < 1e-9 && q.err < 1e-6);
    }
    {   // STRATA_ROUTE_BUDGET_AUTO: runs of 8 alternate the arms; a faster candidate moves b, a slower one turns the direction
        AutoBudget a;
        a.on = true;
        a.b = 0.1f;
        CHECK(std::fabs(auto_arm_b(a, 1) - 0.15f) < 1e-6f);
        bool ended = false;
        for (int w = 0; w < 2 * AutoBudget::kPerArm && !ended; ++w) {
            CHECK(a.arm == (w / AutoBudget::kRun) % 2);
            ended = auto_window(a, a.arm == 1 ? 6 : 5, 0.01);   // the candidate keeps more tokens a window
        }
        CHECK(ended && a.tests == 1 && a.wins == 1);
        CHECK(std::fabs(a.b - 0.15f) < 1e-6f);
        ended = false;
        for (int w = 0; w < 2 * AutoBudget::kPerArm && !ended; ++w) ended = auto_window(a, a.arm == 1 ? 4 : 5, 0.01);
        CHECK(ended && a.tests == 2 && a.wins == 1 && a.dir == -1);
        CHECK(std::fabs(a.b - 0.15f) < 1e-6f);
        CHECK(auto_arm_b(a, 1) < a.b);
        AutoBudget off;
        CHECK(!auto_window(off, 5, 0.01));
    }
    {   // Wilson bounds
        double lo = 0, hi = 0;
        fragile_wilson90(0, 0, lo, hi);
        CHECK(lo == 0.0 && hi == 1.0);
        fragile_wilson90(10, 10, lo, hi);
        CHECK(lo > 0.7 && hi > 0.99);
    }
    if (fails) return 1;
    std::printf("fragile_recheck_test: ok\n");
    return 0;
}
