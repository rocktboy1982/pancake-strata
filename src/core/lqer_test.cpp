// lqer_test: STRATA_LQER's host factorization (include/strata/core/lqer.hpp) and the TEAL / BlockFFN mask rule
// (include/strata/kernels/ffn_sparse.hpp, teal_sparsify_host).  CPU only, header only.
#include "strata/core/lqer.hpp"
#include "strata/kernels/ffn_sparse.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
}  // namespace

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    // an exact rank-4 matrix: rank 4 recovers it, rank 2 keeps most of the energy out
    {
        const int m = 96, n = 160, k = 4;
        std::vector<float> U((size_t) m * k), V((size_t) k * n), E((size_t) m * n, 0.0f);
        for (auto& x : U) x = nd(rng);
        for (auto& x : V) x = nd(rng);
        for (int i = 0; i < m; ++i)
            for (int c = 0; c < n; ++c)
                for (int j = 0; j < k; ++j) E[(size_t) i * n + c] += U[(size_t) i * k + j] * V[(size_t) j * n + c];
        double e2 = 0.0;
        for (const float e : E) e2 += (double) e * e;
        std::vector<float> A, B;
        strata::core::lqer_factor(E.data(), m, n, 4, A, B);
        check(A.size() == (size_t) m * 4 && B.size() == (size_t) 4 * n, "shapes of A and B");
        check(strata::core::lqer_residual2(E.data(), m, n, A, B, 4) < 1e-8 * e2, "rank 4 reconstructs a rank-4 matrix");
        strata::core::lqer_factor(E.data(), m, n, 2, A, B);
        const double r2 = strata::core::lqer_residual2(E.data(), m, n, A, B, 2);
        check(r2 < e2 && r2 > 1e-3 * e2, "rank 2 of a rank-4 matrix: less energy left, not none");
    }
    // a quantization-like noise matrix: the residual falls with the rank and stays near the optimum (n - r)/n of the energy
    {
        const int m = 64, n = 128;
        std::vector<float> E((size_t) m * n);
        for (auto& x : E) x = nd(rng) * 0.01f;
        double e2 = 0.0;
        for (const float e : E) e2 += (double) e * e;
        std::vector<float> A, B;
        double prev = e2 * 1.0000001;
        for (const int r : {1, 4, 16, 32}) {
            strata::core::lqer_factor(E.data(), m, n, r, A, B);
            const double res = strata::core::lqer_residual2(E.data(), m, n, A, B, r);
            check(res < prev, "the residual falls with the rank");
            check(res < e2 * (1.0 - 0.5 * r / (double) m), "at least half the optimal energy captured");
            prev = res;
        }
        strata::core::lqer_factor(E.data(), m, n, 1000, A, B);   // clamped to min(m, n)
        check(A.size() == (size_t) m * m, "rank clamped to min(m, n)");
    }
    // the TEAL / BlockFFN rule
    {
        const float nan = std::numeric_limits<float>::quiet_NaN();
        // 2 rows x 5 columns, tau 0.5
        std::vector<float> h = {0.1f, -0.7f, 0.2f, nan, 0.0f,
                                0.6f, 0.3f, -0.1f, 0.0f, 0.4f};
        std::vector<float> per = h, uni = h;
        std::vector<int32_t> idx;
        strata::kernels::teal_sparsify_host(per.data(), 5, 2, 0.5f, false, &idx);
        check(per[0] == 0.0f && per[1] == -0.7f && per[2] == 0.0f && std::isnan(per[3]) && per[4] == 0.0f, "TEAL row 0");
        check(per[5] == 0.6f && per[6] == 0.0f && per[7] == 0.0f && per[9] == 0.0f, "TEAL row 1");
        check(idx.size() == 3 && idx[0] == 0 && idx[1] == 1 && idx[2] == 3, "active columns: active in some row, NaN active");
        strata::kernels::teal_sparsify_host(uni.data(), 5, 2, 0.5f, true, &idx);
        check(uni[0] == 0.1f && uni[5] == 0.6f, "union: a column active in one row keeps every row's value");
        check(uni[2] == 0.0f && uni[7] == 0.0f && uni[4] == 0.0f && uni[9] == 0.0f, "union: columns active in no row are zero");
        check(uni[6] == 0.3f, "union: the small value of an active column stays");
        std::vector<float> all = h;
        strata::kernels::teal_sparsify_host(all.data(), 5, 2, 0.0f, false, &idx);
        check(idx.size() == 5 && all[0] == 0.1f && all[6] == 0.3f, "tau 0 keeps every value and lists every column");
    }
    std::printf("lqer_test: %d checks passed\n", checks);
    return 0;
}
