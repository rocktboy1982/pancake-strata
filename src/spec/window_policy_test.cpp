// src/spec/window_policy_test.cpp - MOD_PLAN section 6 item 21, STRATA_WINDOW_POLICY: the causal last-N-windows rule
// (spec/window_policy.hpp), CPU only.  The expected kept rows below come from tools/pancake/window_policy.py causal_ks on the
// same synthetic trace (rows 7 5 6 9 4 6 3 8 5 5 6 9, accepted 3 0 1 5 2 2 0 4 1 1 3 6), so the C++ rule and the oracle
// tool's controller agree.  Then the row list against the ECAP rule: kept rows are uncapped, the rest are capped.
#include "strata/core/verify_ecap.hpp"
#include "strata/spec/window_policy.hpp"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

using namespace strata::spec;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static std::vector<int> run_causal(int n, int margin) {
    const int rows[12] = {7, 5, 6, 9, 4, 6, 3, 8, 5, 5, 6, 9};
    const int acc[12] = {3, 0, 1, 5, 2, 2, 0, 4, 1, 1, 3, 6};
    WindowPolicyCfg c;
    c.n = n;
    c.margin = margin;
    WindowPolicy p(c);
    std::vector<int> ks;
    for (int i = 0; i < 12; ++i) {
        ks.push_back(p.kept_rows(rows[i]));
        p.observe(acc[i]);
    }
    return ks;
}

int main() {
    // ---- the value
    CHECK(window_policy_parse(nullptr).n == 0);
    CHECK(window_policy_parse("").n == 0);
    {
        const WindowPolicyCfg c = window_policy_parse("4");
        CHECK(c.n == 4 && c.margin == 0 && c.cap == 0);
        const WindowPolicyCfg d = window_policy_parse("8,2,1");
        CHECK(d.n == 8 && d.margin == 2 && d.cap == 1);
    }
    CHECK(window_policy_parse("0").n == -1);
    CHECK(window_policy_parse("65").n == -1);
    CHECK(window_policy_parse("4,17").n == -1);
    CHECK(window_policy_parse("4,1,64").n == -1);
    CHECK(window_policy_parse("4,1,0,3").n == -1);
    CHECK(window_policy_parse("x").n == -1);
    CHECK(window_policy_parse("4,").n == -1);
    CHECK(window_policy_parse("-1").n == -1);

    // ---- the same kept rows as the oracle tool's causal rule (python window_policy.causal_ks)
    CHECK((run_causal(1, 0) == std::vector<int>{7, 4, 1, 2, 4, 3, 3, 1, 5, 2, 2, 4}));
    CHECK((run_causal(2, 1) == std::vector<int>{7, 5, 4, 3, 4, 6, 3, 3, 4, 5, 3, 4}));
    CHECK((run_causal(4, 0) == std::vector<int>{7, 4, 3, 3, 4, 3, 3, 4, 3, 3, 3, 4}));
    CHECK((run_causal(8, 2) == std::vector<int>{7, 5, 5, 5, 4, 6, 3, 5, 5, 5, 5, 6}));

    // ---- off: nothing changes, observe is a no-op, the window keeps every row
    {
        WindowPolicy off;
        off.observe(5);
        CHECK(!off.on() && off.kept_rows(9) == 9);
    }
    // ---- history longer than N: only the last N count (ring wraps past kMaxN)
    {
        WindowPolicyCfg c;
        c.n = 2;
        WindowPolicy p(c);
        for (int i = 0; i < 200; ++i) p.observe(0);
        p.observe(5);
        p.observe(5);
        CHECK(p.kept_rows(16) == 6);   // mean 5 -> 5 + 1
        p.observe(0);
        CHECK(p.kept_rows(16) == 4);   // mean 2.5 -> ceil 3 + 1
        CHECK(p.kept_rows(2) == 2);    // never more than T
    }
    // ---- the row list against the ECAP rule: a 6-row window, kept 3, cap 0: draft rows 1 and 2 uncapped, 3..5 capped
    {
        WindowPolicyCfg c;
        c.n = 1;
        c.cap = 0;
        WindowPolicy p(c);
        p.observe(2);   // predicted 2 accepted -> kept 3
        int8_t rows[16];
        const int n = p.rows_list(6, rows, 16);
        CHECK(n == 3 && rows[0] == -1 && rows[1] == -1 && rows[2] == 0);
        // k 1, three entries per row, every missed expert distinct: rows 1..2 read their own, rows 3..5 read nothing new
        const int64_t T = 6, K = 1;
        std::vector<int64_t> first_of((size_t) (T * K));
        std::unique_ptr<bool[]> missed(new bool[(size_t) (T * K)]), rd(new bool[(size_t) (T * K)]), dr(new bool[(size_t) (T * K)]);
        for (int64_t i = 0; i < T * K; ++i) { first_of[(size_t) i] = i; missed[(size_t) i] = true; }
        const int64_t d = strata::core::verify_ecap_drop_rows(first_of.data(), missed.get(), T, K, 1, 0, -1, rows, n, nullptr, 0,
                                                              -1.0f, rd.get(), dr.get());
        CHECK(d == 3);
        for (int i = 0; i < 6; ++i) CHECK(dr[i] == (i >= 3));
        // cap 1 on the same window keeps one new expert per capped row (k 1: each row names one)
        WindowPolicyCfg c1 = c;
        c1.cap = 1;
        WindowPolicy q(c1);
        q.observe(2);
        const int n1 = q.rows_list(6, rows, 16);
        CHECK(n1 == 3 && rows[2] == 1);
        CHECK(strata::core::verify_ecap_drop_rows(first_of.data(), missed.get(), T, K, 1, 0, -1, rows, n1, nullptr, 0, -1.0f,
                                                  rd.get(), dr.get()) == 0);
        // kept >= T: the one value -1 (overrides a scalar cap for this window)
        WindowPolicyCfg c9 = c;
        c9.margin = 9;
        WindowPolicy r(c9);
        r.observe(1);
        CHECK(r.rows_list(6, rows, 16) == 1 && rows[0] == -1);
        // cold history: the same
        WindowPolicy cold(c);
        CHECK(cold.rows_list(6, rows, 16) == 1 && rows[0] == -1);
        // kept 1 (nothing accepted lately, margin 0): the list is the cap alone, every draft row capped
        WindowPolicy z(c);
        z.observe(0);
        CHECK(z.rows_list(6, rows, 16) == 1 && rows[0] == 0);   // ceil(0) + 1 = kept 1: window row 1 on is capped
    }
    if (fails == 0) std::printf("window_policy_test: ok\n");
    return fails == 0 ? 0 : 1;
}
