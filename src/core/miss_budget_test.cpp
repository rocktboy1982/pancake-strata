// src/core/miss_budget_test.cpp - MOD_PLAN section 6 item 20, STRATA_MISS_BUDGET: the per-token weight-budget rule
// (core/miss_budget.hpp), CPU only.  Worked tokens, then random tokens against the rule's promises: the skipped weight stays
// within the budget, only candidates are skipped, the skipped set is a lightest-first prefix of the candidates and is maximal
// (the next lightest candidate would pass the budget), and a larger budget never skips fewer entries.
#include "strata/core/miss_budget.hpp"

#include <cstdint>
#include <cstdio>
#include <algorithm>
#include <limits>
#include <memory>
#include <random>
#include <vector>

using namespace strata::core;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    // ---- the value
    CHECK(miss_budget_parse(nullptr) == 0.0f);
    CHECK(miss_budget_parse("") == 0.0f);
    CHECK(miss_budget_parse("0.05") == 0.05f);
    CHECK(miss_budget_parse("0") == 0.0f);
    CHECK(miss_budget_parse("1") == 0.0f);
    CHECK(miss_budget_parse("-0.1") == 0.0f);
    CHECK(miss_budget_parse("0.1x") == 0.0f);
    CHECK(miss_budget_parse("nan") == 0.0f);

    // ---- worked token: k 5, weights sum to 1: .4 .25 .2 .1 .05, all candidates
    {
        const float w[5] = {0.4f, 0.25f, 0.2f, 0.1f, 0.05f};
        const bool all[5] = {true, true, true, true, true};
        bool skip[5];
        CHECK(miss_budget_token(w, all, 5, 0.04f, skip) == 0);          // even the lightest (0.05) passes 4%
        CHECK(miss_budget_token(w, all, 5, 0.05f, skip) == 1 && skip[4] && !skip[3]);
        CHECK(miss_budget_token(w, all, 5, 0.14f, skip) == 1 && skip[4]);   // 0.05 + 0.10 = 0.15 > 0.14
        CHECK(miss_budget_token(w, all, 5, 0.15f, skip) == 2 && skip[4] && skip[3] && !skip[2]);
        CHECK(miss_budget_token(w, all, 5, 0.9f, skip) == 4 && !skip[0] && skip[1] && skip[2]);   // 0.05+.1+.2+.25 = .6, +.4 = 1.0 > .9
        // only the missed ones are candidates: the lightest (entry 4) is resident, so the next lightest goes first
        const bool miss[5] = {true, true, true, true, false};
        CHECK(miss_budget_token(w, miss, 5, 0.1f, skip) == 1 && skip[3] && !skip[4]);
        // a flat token: the budget skips nothing a fixed floor would not (all .2: 0.2 > 0.15)
        const float flat[5] = {0.2f, 0.2f, 0.2f, 0.2f, 0.2f};
        CHECK(miss_budget_token(flat, all, 5, 0.15f, skip) == 0);
        CHECK(miss_budget_token(flat, all, 5, 0.4f, skip) == 2 && skip[4] && skip[3]);   // ties: the later rank first
        // not a number, negative weight, empty sum: nothing
        const float nanw[3] = {0.5f, std::numeric_limits<float>::quiet_NaN(), 0.1f};
        const bool c3[3] = {true, true, true};
        bool s3[3];
        CHECK(miss_budget_token(nanw, c3, 3, 0.9f, s3) == 0);
        const float neg[3] = {0.5f, -0.1f, 0.6f};
        CHECK(miss_budget_token(neg, c3, 3, 0.9f, s3) == 0);
        const float zero[3] = {0.0f, 0.0f, 0.0f};
        CHECK(miss_budget_token(zero, c3, 3, 0.9f, s3) == 0);
        CHECK(miss_budget_token(w, all, 5, 0.0f, skip) == 0);
        CHECK(miss_budget_token(w, all, 0, 0.5f, skip) == 0);
    }

    // ---- random tokens
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> u(0.01f, 1.0f);
    int64_t tokens = 0, skipped = 0;
    for (int it = 0; it < 20000; ++it) {
        const int k = 2 + (int) (rng() % 11);
        std::vector<float> w((size_t) k);
        std::unique_ptr<bool[]> cand(new bool[(size_t) k]), s1(new bool[(size_t) k]), s2(new bool[(size_t) k]);
        double sum = 0;
        for (int j = 0; j < k; ++j) {
            w[(size_t) j] = u(rng) * u(rng);   // skewed
            sum += (double) w[(size_t) j];
            cand[(size_t) j] = rng() % 100 < 70;
        }
        const float b = 0.02f + 0.01f * (float) (rng() % 30);
        const int n1 = miss_budget_token(w.data(), cand.get(), k, b, s1.get());
        const int n2 = miss_budget_token(w.data(), cand.get(), k, b + 0.1f, s2.get());
        ++tokens;
        skipped += n1;
        double got = 0;
        int cnt = 0;
        double maxskip = 0;
        for (int j = 0; j < k; ++j)
            if (s1[(size_t) j]) {
                CHECK(cand[(size_t) j]);   // only candidates
                got += (double) w[(size_t) j];
                maxskip = std::max(maxskip, (double) w[(size_t) j]);
                ++cnt;
            }
        CHECK(cnt == n1);
        CHECK(got <= (double) b * sum * (1.0 + 1e-5));   // the weight budget
        // prefix: every candidate lighter than a skipped one is skipped too
        for (int j = 0; j < k; ++j)
            if (cand[(size_t) j] && !s1[(size_t) j]) CHECK((double) w[(size_t) j] >= maxskip || cnt == 0);
        // maximal: the lightest candidate left would pass the budget
        double next = 1e30;
        for (int j = 0; j < k; ++j)
            if (cand[(size_t) j] && !s1[(size_t) j]) next = std::min(next, (double) w[(size_t) j]);
        if (next < 1e29) CHECK(got + next > (double) b * sum * (1.0 + 1e-6) - 1e-12);
        CHECK(n2 >= n1);   // a larger budget never skips fewer
    }
    CHECK(skipped > 0);
    if (fails == 0) std::printf("miss_budget_test: ok (%lld tokens, %lld entries skipped)\n", (long long) tokens, (long long) skipped);
    return fails == 0 ? 0 : 1;
}
