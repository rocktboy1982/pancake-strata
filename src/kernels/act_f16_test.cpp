// src/kernels/act_f16_test.cpp - host reference for LEADS.md Part O O10 (STRATA_ACT_F16): how much of a quantised matvec's error is the ACTIVATION rounding?
//
// The caminus authors' claim (idea only): multiplying the dequantised weights by fp16 activations instead of llama.cpp's int8 (Q8_1, one fp16 scale per 32
// values) activations cuts the mean KLD against a float64 reference about 6x on a 4-bit model, at a decode cost of no more than 0.08 ms a token.  This test
// measures the first half on the part that is arithmetic and nothing else: for random 2-bit-class weights (what our IQ2 packs hold: 4 levels and a scale per
// block of 32) and activations of three shapes (gaussian, gaussian with a few 20x outliers per row, heavy tail), the dot product with the activation
// rounded as Q8_1 does, with it rounded to fp16, and with it exact, each against the float64 dot of the same weights.  It prints the mean relative error
// of each and asserts the ratio the hypothesis needs (fp16 several times closer).  A fixture, not a model result: whether the engine's quality moves is the
// token-forcing gate's job on the box (LEADS.md O10).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

// float -> IEEE half -> float, round to nearest even (no denormal flush; the activations here are well inside the normal range)
static float round_half(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = u & 0x80000000u;
    int exp = int((u >> 23) & 0xFF) - 127;
    uint32_t man = u & 0x7FFFFFu;
    if (exp > 15) return sign ? -65504.0f : 65504.0f;
    if (exp < -14) {   // subnormal half: a multiple of 2^-24
        const float q = std::nearbyint(f * 16777216.0f) / 16777216.0f;
        return q;
    }
    // keep 10 mantissa bits with round to nearest even
    const uint32_t keep = man >> 13, rest = man & 0x1FFFu;
    uint32_t m = keep;
    if (rest > 0x1000u || (rest == 0x1000u && (keep & 1u))) ++m;
    float out = std::ldexp(1.0f + float(m) / 1024.0f, exp);
    return sign ? -out : out;
}

// Q8_1 as llama.cpp rounds an activation block: d = amax / 127 (stored as fp16), q = round(x / d), value = d * q
static void round_q8_1(const float* x, float* out, int n) {
    for (int b = 0; b < n; b += 32) {
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(x[b + i]));
        const float d = round_half(amax / 127.0f);
        for (int i = 0; i < 32; ++i) out[b + i] = d == 0.0f ? 0.0f : d * std::nearbyint(x[b + i] / d);
    }
}

int main() {
    const int K = 2560, ROWS = 96, TRIALS = 40;
    std::mt19937 rng(2026);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> lvl(0, 3);
    std::uniform_int_distribution<int> pos(0, K - 1);
    double err_q8[3] = {0, 0, 0}, err_f16[3] = {0, 0, 0};
    const char* names[3] = {"gaussian", "outliers x20", "heavy tail (cubed)"};
    for (int shape = 0; shape < 3; ++shape) {
        for (int trial = 0; trial < TRIALS; ++trial) {
            std::vector<float> x(K), xq(K), xh(K);
            for (int i = 0; i < K; ++i) x[size_t(i)] = nd(rng);
            if (shape == 1) for (int k = 0; k < 6; ++k) x[size_t(pos(rng))] *= 20.0f;
            if (shape == 2) for (int i = 0; i < K; ++i) x[size_t(i)] = x[size_t(i)] * x[size_t(i)] * x[size_t(i)] * 0.3f;
            round_q8_1(x.data(), xq.data(), K);
            for (int i = 0; i < K; ++i) xh[size_t(i)] = round_half(x[size_t(i)]);
            for (int r = 0; r < ROWS; ++r) {
                double ref = 0, dq = 0, dh = 0;
                for (int b = 0; b < K; b += 32) {
                    const float scale = 0.05f * (1.0f + 0.5f * nd(rng) * 0.2f);
                    for (int i = 0; i < 32; ++i) {
                        const float w = scale * (float(lvl(rng)) - 1.5f);   // 4 levels, symmetric
                        ref += double(w) * x[size_t(b + i)];
                        dq += double(w) * xq[size_t(b + i)];
                        dh += double(w) * xh[size_t(b + i)];
                    }
                }
                const double den = std::fabs(ref) + 1e-3;   // relative to the result, floored (a near-zero dot has no meaningful relative error)
                err_q8[shape] += std::fabs(dq - ref) / den;
                err_f16[shape] += std::fabs(dh - ref) / den;
            }
        }
        const double n = double(ROWS) * TRIALS;
        err_q8[shape] /= n;
        err_f16[shape] /= n;
        std::printf("%-20s mean relative error: Q8_1 activation %.3e, fp16 activation %.3e, ratio %.1fx\n", names[shape], err_q8[shape], err_f16[shape],
                    err_q8[shape] / err_f16[shape]);
        CHECK(err_f16[shape] * 4.0 < err_q8[shape]);   // the hypothesis needs fp16 several times closer; the measured ratios are printed above
    }
    // the helper rounds as IEEE half does at a few known points
    CHECK(round_half(1.0f) == 1.0f && round_half(0.1f) == 0.0999755859375f && round_half(65504.0f) == 65504.0f && round_half(1.0f + 1.0f / 2048.0f) == 1.0f);
    if (g_fail) { std::printf("act_f16_test: %d FAILED\n", g_fail); return 1; }
    std::printf("act_f16_test: ok\n");
    return 0;
}
