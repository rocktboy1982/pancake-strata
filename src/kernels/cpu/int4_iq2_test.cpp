// int4_iq2_test.cpp - host test of include/strata/kernels/int4_iq2.hpp and int4_moe.hpp (LEADS.md Part L L3 / L4, STRATA_PF_IU4 host side).
// No GPU, no model: random blocks of the two daily formats.
//   1. the SWAR Q2_0 -> nibble expansion equals the loop, for every 16-bit code word;
//   2. Q2_0 -> int4 is EXACT: every value equals the engine's dequantize_q2_0 bit for bit;
//   3. IQ2_XS: the codebook has the three magnitudes {8, 25, 43}; the level mapping error at the nominal step 8.2 and at the least-squares
//      step is MEASURED and printed (weights and a matrix product); the mapped values stay inside int4;
//   4. the iu4 tile loop (emulated instruction, Q2_0 bytes as the operand source, expert gather) equals int4rot::gemm_ref bit for bit.
// What it cannot show: whether the hardware instruction reads its operands the way wmma_iu4_emulate does (box run of int4_moe_probe).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include "strata/artifact/dequant.hpp"
#include "strata/kernels/int4_moe.hpp"

using namespace strata::kernels;

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

static uint16_t float_to_half(float f) {   // round to nearest even, normal range only (scales here are 1e-3 .. 1)
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000, e = (x >> 23) & 255, m = x & 0x7fffff;
    int he = (int) e - 127 + 15;
    uint32_t hm = m >> 13;
    const uint32_t rest = m & 0x1fff;
    if (rest > 0x1000 || (rest == 0x1000 && (hm & 1))) ++hm;
    if (hm == 1024) { hm = 0; ++he; }
    return (uint16_t) (s | ((uint32_t) he << 10) | hm);
}

static double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, n = 0;
    for (size_t i = 0; i < a.size(); ++i) { d += (double) (a[i] - b[i]) * (a[i] - b[i]); n += (double) a[i] * a[i]; }
    return n > 0 ? std::sqrt(d / n) : 0.0;
}

int main() {
    std::mt19937 rng(7);
    // ---- 1. SWAR expansion, exhaustive over the 16-bit word, and the 32-bit pair on random words
    for (uint32_t w = 0; w < 65536; ++w) {
        uint32_t ref = 0;
        for (int e = 0; e < 8; ++e) ref |= ((uint32_t) (((int) ((w >> (2 * e)) & 3) - 1) & 15)) << (4 * e);
        CHECK(int4moe::q2_0_half_to_nibbles(w) == ref, "swar %u", w);
        if (g_fail) break;
    }
    for (int i = 0; i < 1000; ++i) {
        const uint32_t w = (uint32_t) rng();
        uint32_t out[2], codes[16];
        int4moe::q2_0_words_to_int4(w, out);
        for (int e = 0; e < 16; ++e) codes[e] = (w >> (2 * e)) & 3;
        for (int e = 0; e < 16; ++e) CHECK(int4iq2::unpack_nibble_signed(out, e) == (int) codes[e] - 1, "slice nibble %d", e);
    }
    // ---- 2. Q2_0 -> int4 exact
    {
        const int rows = 24, K = 640;   // K = 640 is the expert down row of the daily model
        std::vector<uint8_t> blk((size_t) rows * (K / 64) * 18);
        for (size_t b = 0; b < blk.size() / 18; ++b) {
            const uint16_t h = float_to_half(0.001f + (float) (rng() % 1000) / 4000.f);
            std::memcpy(&blk[b * 18], &h, 2);
            for (int j = 0; j < 16; ++j) blk[b * 18 + 2 + j] = (uint8_t) rng();
        }
        const auto q = int4iq2::q2_0_to_int4(blk.data(), rows, K);
        long bad = 0, nonfinite = 0;
        int qmin = 99, qmax = -99;
        for (int m = 0; m < rows; ++m)
            for (int b = 0; b < K / 64; ++b) {
                float out[64];
                strata::dequantize_q2_0(&blk[((size_t) m * (K / 64) + b) * 18], out);
                for (int j = 0; j < 64; ++j) {
                    const float v = q.value(m, b * 64 + j);
                    uint32_t x, y;
                    std::memcpy(&x, &v, 4);
                    std::memcpy(&y, &out[j], 4);
                    bad += x != y;
                    nonfinite += !std::isfinite(v);
                    qmin = std::min(qmin, (int) q.q[(size_t) m * K + b * 64 + j]);
                    qmax = std::max(qmax, (int) q.q[(size_t) m * K + b * 64 + j]);
                }
            }
        CHECK(bad == 0, "Q2_0 -> int4 not bit exact: %ld of %d values differ", bad, rows * K);
        CHECK(nonfinite == 0 && qmin == -1 && qmax == 2, "Q2_0 symbols %d..%d", qmin, qmax);
        std::printf("Q2_0 -> int4: %d values, bit-identical to dequantize_q2_0, codes %d..%d\n", rows * K, qmin, qmax);
    }
    // ---- 3. IQ2_XS levels
    const int4iq2::Iq2xsTables tab{iq2xs_grid, ksigns_iq2xs};
    const float step_best = int4iq2::best_step(tab);
    {
        const auto s82 = int4iq2::codebook_level_stats(tab, 8.2f);
        const auto sb = int4iq2::codebook_level_stats(tab, step_best);
        CHECK(s82.other == 0, "the IQ2_XS grid holds %ld magnitudes other than 8 / 25 / 43", s82.other);
        CHECK(s82.count[0] > 0 && s82.count[1] > 0 && s82.count[2] > 0, "a magnitude never occurs");
        std::printf("IQ2_XS codebook magnitudes (512 entries x 8): 8 x %ld, 25 x %ld, 43 x %ld\n", s82.count[0], s82.count[1], s82.count[2]);
        std::printf("IQ2_XS -> odd int4 levels: step 8.200 rms rel error %.4f worst %.4f; least-squares step %.4f rms %.4f worst %.4f\n",
                    s82.rms_rel, s82.worst_rel, step_best, sb.rms_rel, sb.worst_rel);
        CHECK(sb.rms_rel <= s82.rms_rel + 1e-9, "the least-squares step is worse than the nominal one");
        CHECK(sb.rms_rel < 0.05, "level mapping rms %.4f is not small", sb.rms_rel);
        // a decoded block against its mapped int4 form: weights and a product, random blocks (random indices are a stand-in for a
        // real tensor: the error per value does not depend on the index, only the magnitude mix does)
        const int rows = 32, K = 2048, N = 16;
        std::vector<uint8_t> blk((size_t) rows * (K / 256) * int4iq2::kIq2xsBlock);
        for (size_t b = 0; b < blk.size() / int4iq2::kIq2xsBlock; ++b) {
            uint8_t* p = &blk[b * int4iq2::kIq2xsBlock];
            const uint16_t h = float_to_half(0.01f + (float) (rng() % 1000) / 20000.f);
            std::memcpy(p, &h, 2);
            for (int j = 2; j < int4iq2::kIq2xsBlock; ++j) p[j] = (uint8_t) rng();
        }
        std::vector<float> w((size_t) rows * K), xs((size_t) N * K);
        for (int m = 0; m < rows; ++m)
            for (int b = 0; b < K / 256; ++b)
                int4iq2::iq2xs_decode_block(tab, &blk[((size_t) m * (K / 256) + b) * int4iq2::kIq2xsBlock], &w[(size_t) m * K + b * 256]);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : xs) v = nd(rng);
        auto product = [&](const std::vector<float>& ww) {
            std::vector<float> y((size_t) N * rows);
            for (int n = 0; n < N; ++n)
                for (int m = 0; m < rows; ++m) {
                    double s = 0;
                    for (int k = 0; k < K; ++k) s += (double) ww[(size_t) m * K + k] * xs[(size_t) n * K + k];
                    y[(size_t) n * rows + m] = (float) s;
                }
            return y;
        };
        const auto y0 = product(w);
        for (const float step : {8.2f, step_best}) {
            const auto q = int4iq2::iq2xs_to_int4(tab, blk.data(), rows, K, step);
            std::vector<float> wq((size_t) rows * K);
            int qmax = 0;
            for (int m = 0; m < rows; ++m)
                for (int k = 0; k < K; ++k) {
                    wq[(size_t) m * K + k] = q.value(m, k);
                    qmax = std::max(qmax, std::abs((int) q.q[(size_t) m * K + k]));
                }
            CHECK(qmax == 5, "odd levels reach %d", qmax);
            const double ew = rel(w, wq), ey = rel(y0, product(wq));
            std::printf("IQ2_XS block -> int4, step %.4f: weight rel error %.4f, y = W x rel error %.4f (random blocks)\n", step, ew, ey);
            CHECK(ew < 0.06 && ey < 0.06, "IQ2_XS int4 mapping error too large: w %.4f y %.4f", ew, ey);
        }
    }
    // ---- 4. the iu4 tile loop with the expert gather equals the host integer pipeline, bit for bit
    {
        const int E = 5, M = 40, K = 640, B = 128;   // 40 is not a multiple of 16 on purpose; K = 640 = 5 x 128 (the down row)
        int4moe::Q2Experts ex;
        ex.n_expert = E; ex.rows = M; ex.K = K;
        ex.bytes.resize((size_t) E * M * (K / 64) * 18);
        for (size_t b = 0; b < ex.bytes.size() / 18; ++b) {
            const uint16_t h = float_to_half(0.002f + (float) (rng() % 1000) / 8000.f);
            std::memcpy(&ex.bytes[b * 18], &h, 2);
            for (int j = 0; j < 16; ++j) ex.bytes[b * 18 + 2 + j] = (uint8_t) rng();
        }
        const std::vector<int> bounds = {0, 3, 3, 21, 22, 41};   // expert 1 gets no token; sizes are not multiples of 16
        const int N = bounds.back();
        std::vector<float> x((size_t) N * K);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : x) v = nd(rng);
        int4rot::rotate_rows(x, N, K, B);
        const auto a = int4rot::quantize_acts(x, N, K, B, true);
        const auto y = int4moe::grouped_tile_gemm(ex, bounds, a);
        double worst = 0;
        long exact = 0, total = 0;
        for (int e = 0; e < E; ++e) {
            const int n0 = bounds[e], n1 = bounds[e + 1];
            if (n1 == n0) continue;
            const auto q = int4iq2::q2_0_to_int4(ex.row_block(e, 0, 0), M, K);
            int4rot::WeightQ wq;
            wq.M = M; wq.K = K; wq.q = q.q; wq.scale = q.scale;
            int4rot::ActQ sub;
            sub.N = n1 - n0; sub.K = K; sub.B = B; sub.a8 = true;
            sub.hi.assign(a.hi.begin() + (size_t) n0 * K, a.hi.begin() + (size_t) n1 * K);
            sub.lo.assign(a.lo.begin() + (size_t) n0 * K, a.lo.begin() + (size_t) n1 * K);
            sub.scale.assign(a.scale.begin() + (size_t) n0 * (K / B), a.scale.begin() + (size_t) n1 * (K / B));
            const auto ref = int4rot::gemm_ref(wq, sub);
            for (int n = n0; n < n1; ++n)
                for (int m = 0; m < M; ++m) {
                    const float got = y[(size_t) n * M + m], want = ref[(size_t) (n - n0) * M + m];
                    ++total;
                    exact += got == want;
                    worst = std::max(worst, (double) std::fabs(got - want));
                }
        }
        CHECK(exact == total, "grouped tile GEMM differs from gemm_ref: %ld of %ld exact, worst abs %.3e", exact, total, worst);
        std::printf("iu4 tile loop (emulated instruction, Q2_0 bytes as operand source, expert gather): %ld of %ld outputs bit-equal to gemm_ref\n", exact, total);
    }
    if (g_fail) { std::printf("int4_iq2_test: %d FAILED\n", g_fail); return 1; }
    std::printf("int4_iq2_test: PASS\n");
    return 0;
}
