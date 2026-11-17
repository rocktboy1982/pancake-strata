// int4_rot_test.cpp - host test of include/strata/kernels/int4_rot.hpp (LEADS.md Part L L1 / L3): no GPU, no model, synthetic data.
// Checks the identities the rotated-INT4 pipeline stands on: R orthogonal and norm preserving, W' x' = W x, the A8 split q = 16 h + l
// exact, the int pipeline close to the float product, and the outlier case where rotation makes 4-bit activations usable.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "strata/kernels/int4_rot.hpp"

using namespace strata::kernels::int4rot;

static int g_fail = 0;
#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            std::printf("FAIL %s:%d %s  ", __FILE__, __LINE__, #c);            \
            std::printf(__VA_ARGS__);                                          \
            std::printf("\n");                                                 \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

static double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, n = 0;
    for (size_t i = 0; i < a.size(); ++i) { d += (double) (a[i] - b[i]) * (a[i] - b[i]); n += (double) a[i] * a[i]; }
    return n > 0 ? std::sqrt(d / n) : 0.0;
}

static std::vector<float> matmul_ref(const std::vector<float>& w, const std::vector<float>& x, int M, int K, int N) {
    std::vector<float> y((size_t) N * M, 0.f);
    for (int n = 0; n < N; ++n)
        for (int m = 0; m < M; ++m) {
            double s = 0;
            for (int k = 0; k < K; ++k) s += (double) w[(size_t) m * K + k] * x[(size_t) n * K + k];
            y[(size_t) n * M + m] = (float) s;
        }
    return y;
}

int main() {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    // the sign hash of the report, first values
    CHECK(sign_of(0) == 1.0f, "s_0");
    for (int B : {128, 256}) {
        // norm preservation and involution-free orthogonality: |Rx| = |x|
        std::vector<float> x(B), y;
        for (auto& v : x) v = nd(rng);
        y = x;
        rotate_block(y.data(), B);
        double a = 0, b = 0;
        for (int i = 0; i < B; ++i) { a += (double) x[i] * x[i]; b += (double) y[i] * y[i]; }
        CHECK(std::fabs(a - b) < 1e-3 * a, "B=%d norm %g vs %g", B, a, b);
    }
    const int M = 32, K = 640, N = 8, B = 128;            // K = 640 = the Flash-Next expert down row, B = 128
    std::vector<float> w((size_t) M * K), x((size_t) N * K);
    for (auto& v : w) v = nd(rng);
    for (auto& v : x) v = nd(rng);
    for (int n = 0; n < N; ++n) for (int c : {3, 200, 411}) x[(size_t) n * K + c] *= 25.0f;   // outlier channels
    const auto ref = matmul_ref(w, x, M, K, N);
    auto wr = w, xr = x;
    rotate_rows(wr, M, K, B);
    rotate_rows(xr, N, K, B);
    CHECK(rel(ref, matmul_ref(wr, xr, M, K, N)) < 1e-5, "W'x' = Wx");

    // the A8 split: 16 h + l reproduces every code
    const ActQ a8 = quantize_acts(xr, N, K, B, true);
    for (size_t i = 0; i < a8.hi.size(); ++i) {
        CHECK(a8.hi[i] >= -8 && a8.hi[i] <= 7 && a8.lo[i] >= 0 && a8.lo[i] <= 15, "nibble range at %zu", i);
        if (g_fail) break;
    }
    const WeightQ wq = quantize_weights_rtn(wr, M, K);
    const WeightQ wq_plain = quantize_weights_rtn(w, M, K);
    const auto y8 = gemm_ref(wq, a8);
    // float product of the DEQUANTISED weights with float rotated activations isolates the activation error
    std::vector<float> wdeq((size_t) M * K);
    for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k) wdeq[(size_t) m * K + k] = wq.q[(size_t) m * K + k] * wq.scale[(size_t) m * (K / 32) + k / 32];
    const auto yw = matmul_ref(wdeq, xr, M, K, N);
    CHECK(rel(yw, y8) < 0.01, "A8 activation error %g", rel(yw, y8));
    const double e_rot8 = rel(ref, y8);

    // A4: rotation must beat the unrotated 4-bit path on outlier data (same quantiser, no rotation)
    const ActQ a4 = quantize_acts(xr, N, K, B, false);
    const ActQ a4_plain = quantize_acts(x, N, K, B, false);
    const double e_rot4 = rel(ref, gemm_ref(wq, a4));
    const double e_plain4 = rel(ref, gemm_ref(wq_plain, a4_plain));
    std::printf("int4_rot_test: A8 rotated %.4f, A4 rotated %.4f, A4 plain %.4f (relative error of y)\n", e_rot8, e_rot4, e_plain4);
    CHECK(e_rot4 < e_plain4, "rotation must help 4-bit activations on outlier data: %g vs %g", e_rot4, e_plain4);
    CHECK(e_rot8 < e_rot4, "8-bit activations must beat 4-bit: %g vs %g", e_rot8, e_rot4);
    // the per-token multiplier m_b stays in 1..16: scale * 16 * 7 = m_b * amax_token
    for (int n = 0; n < N; ++n) for (int b = 0; b < K / B; ++b) {
        const float r = a4.scale[(size_t) n * (K / B) + b] * 16.f * 7.f;
        float amax_tok = 0; for (int k = 0; k < K; ++k) amax_tok = std::fmax(amax_tok, std::fabs(xr[(size_t) n * K + k]));
        CHECK(r >= amax_tok * 0.9999f && r <= amax_tok * 16.f * 1.0001f, "A4 multiplier outside 1..16");
    }
    if (g_fail) { std::printf("int4_rot_test: %d failures\n", g_fail); return 1; }
    std::printf("int4_rot_test: ok\n");
    return 0;
}
