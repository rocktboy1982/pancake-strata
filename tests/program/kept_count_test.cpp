// STRATA_COUNT_KEPT (LEADS.md Part Q4): the rejected rows' share of a verify window's usage counts is taken back out
// (include/strata/core/kept_count.hpp).  No GPU.  The class only changes counts it was told about, and only on drop_rejected.
#include <cmath>
#include <cstdio>
#include <vector>

#include "strata/core/kept_count.hpp"

using strata::core::KeptCount;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

constexpr int NL = 2, NE = 8, K = 3;

// what expert_pool_dispatch_multi adds to usage for a window (the pool's own two branches)
static void pool_add(std::vector<float>& u, int layer, const std::vector<int32_t>& ids, const float* w) {
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || ids[i] >= NE) continue;
        u[(size_t) layer * NE + ids[i]] += w == nullptr ? 1.0f : (w[i] == w[i] && w[i] > 0.0f ? w[i] * (float) K : 0.0f);
    }
}

int main() {
    CHECK(!KeptCount::on_from(nullptr) && !KeptCount::on_from("") && !KeptCount::on_from("0") && KeptCount::on_from("1"));
    // a window of 4 rows, 3 picks each; 2 rows kept: rows 2 and 3 come back out; row 0 and 1 stay
    {
        KeptCount kc(NL, NE);
        std::vector<float> u(NL * NE, 0.0f);
        const std::vector<int32_t> l0 = {1, 2, 3,  1, 2, 4,  5, 6, 7,  1, 5, 0};   // layer 0
        const std::vector<int32_t> l1 = {0, 0, 1,  2, 2, 2,  3, 3, 3,  4, 4, 4};   // layer 1 (repeats inside a row)
        pool_add(u, 0, l0, nullptr); kc.record(0, l0.data(), nullptr, 4, K);
        pool_add(u, 1, l1, nullptr); kc.record(1, l1.data(), nullptr, 4, K);
        const double out = kc.drop_rejected(u, 4, 2);
        CHECK(std::fabs(out - 12.0) < 1e-6);   // 2 rejected rows x 3 picks x 2 layers
        // layer 0: rows 0-1 = {1,2,3} + {1,2,4}
        CHECK(u[1] == 2.0f && u[2] == 2.0f && u[3] == 1.0f && u[4] == 1.0f);
        CHECK(u[5] == 0.0f && u[6] == 0.0f && u[7] == 0.0f && u[0] == 0.0f);
        // layer 1: rows 0-1 = {0,0,1} + {2,2,2}
        CHECK(u[NE + 0] == 2.0f && u[NE + 1] == 1.0f && u[NE + 2] == 3.0f && u[NE + 3] == 0.0f && u[NE + 4] == 0.0f);
        // the records are consumed: a second call changes nothing
        CHECK(kc.drop_rejected(u, 4, 2) == 0.0);
    }
    // all rows kept, or none to take back: no change
    {
        KeptCount kc(NL, NE);
        std::vector<float> u(NL * NE, 0.0f);
        const std::vector<int32_t> l0 = {1, 2, 3, 4, 5, 6};
        pool_add(u, 0, l0, nullptr); kc.record(0, l0.data(), nullptr, 2, K);
        const std::vector<float> before = u;
        CHECK(kc.drop_rejected(u, 2, 2) == 0.0 && u == before);
        kc.record(0, l0.data(), nullptr, 2, K);
        CHECK(kc.drop_rejected(u, 2, 0) == 0.0 && u == before);   // kept < 1 is not a window the loop reports
    }
    // a record of another window size (stale) is dropped, not applied
    {
        KeptCount kc(NL, NE);
        std::vector<float> u(NL * NE, 5.0f);
        const std::vector<int32_t> l0 = {1, 2, 3, 4, 5, 6};
        kc.record(0, l0.data(), nullptr, 2, K);
        CHECK(kc.drop_rejected(u, 4, 1) == 0.0);
        for (float v : u) CHECK(v == 5.0f);
        CHECK(kc.drop_rejected(u, 2, 1) == 0.0);   // and it is gone
    }
    // with STRATA_ADAPT_PROB_MASS weights: exactly what was added comes out
    {
        KeptCount kc(NL, NE);
        std::vector<float> u(NL * NE, 0.0f);
        const std::vector<int32_t> l0 = {1, 2, 3,  4, 5, 6};
        const std::vector<float> w = {0.5f, 0.3f, 0.2f,  0.6f, 0.0f, NAN};
        pool_add(u, 0, l0, w.data()); kc.record(0, l0.data(), w.data(), 2, K);
        CHECK(std::fabs(u[4] - 1.8f) < 1e-6f && u[5] == 0.0f && u[6] == 0.0f);
        const double out = kc.drop_rejected(u, 2, 1);
        CHECK(std::fabs(out - 1.8) < 1e-5);
        CHECK(u[4] == 0.0f && std::fabs(u[1] - 1.5f) < 1e-6f);
    }
    // counts that were decayed in between never go negative; ids out of range and -1 are skipped; a wrong-sized usage is left alone
    {
        KeptCount kc(NL, NE);
        std::vector<float> u(NL * NE, 0.0f);
        const std::vector<int32_t> l0 = {1, 2, 3,  -1, 99, 3};
        pool_add(u, 0, l0, nullptr); kc.record(0, l0.data(), nullptr, 2, K);
        for (float& v : u) v *= 0.5f;   // a tier round decayed them
        kc.drop_rejected(u, 2, 1);
        for (float v : u) CHECK(v >= 0.0f);
        std::vector<float> bad(3, 1.0f);
        kc.record(0, l0.data(), nullptr, 2, K);
        CHECK(kc.drop_rejected(bad, 2, 1) == 0.0 && bad[0] == 1.0f);
    }
    if (failed == 0) std::printf("kept_count_test: ok\n");
    return failed == 0 ? 0 : 1;
}
