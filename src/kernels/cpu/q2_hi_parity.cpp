// src/kernels/cpu/q2_hi_parity.cpp - LEADS.md SliceMoE (STRATA_SLICE_MISS, opt-in, lossy): the 1-bit slice of Q2_0 rows.
//
//   1. q2_hi_plane_build + q2_hi_rows_scalar = a direct reference from the 18-byte rows, sum_i d x ((code_i & 2) - 0.5)
//      x scale_c x q_i (the high-bit midpoint of d x (code - 1));
//   2. q2_hi_rows_avx2 = q2_hi_rows_scalar (the integer part is exact; float order the same up to FMA contraction), alone
//      and at 1..8 tokens, cut into row ranges;
//   3. q2_hi_plane_cached returns the same planes for the same address.
// Synthetic data, CPU only; the AVX2 part is skipped on a CPU without AVX2.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace c = strata::kernels::cpu;

namespace {
float f16(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v = e == 0 ? std::ldexp((float) m, -24) : std::ldexp((float) (m | 0x400), (int) e - 25);
    return s ? -v : v;
}
}  // namespace

int main() {
    std::mt19937 rng(489);
    std::normal_distribution<float> nd(0.f, 1.f);
    int fail = 0;
    const int nblocks = c::FF / 64, rows = 53;
    const size_t row_bytes = (size_t) nblocks * 18;
    std::vector<uint8_t> w((size_t) rows * row_bytes);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nblocks; ++b) {
            uint8_t* blk = &w[(size_t) r * row_bytes + (size_t) b * 18];
            const uint16_t d = (uint16_t) (0x1C00 + rng() % 0x0800);
            std::memcpy(blk, &d, 2);
            for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t) rng();
        }
    std::vector<uint8_t> planes((size_t) rows * nblocks * c::kQ2HiBlock);
    c::q2_hi_plane_build(w.data(), row_bytes, nblocks, rows, planes.data());
    static c::ActQ acts[c::MAXT];
    const c::ActQ* ap[c::MAXT];
    for (int t = 0; t < c::MAXT; ++t) {
        std::vector<float> x((size_t) c::FF);
        for (float& v : x) v = nd(rng) * (t % 3 == 0 ? 4.f : 1.f);
        c::act_quant_any(x.data(), c::FF, acts[t]);
        ap[t] = &acts[t];
    }
    // (1) scalar vs the direct reference
    for (int nt = 1; nt <= c::MAXT; ++nt) {
        std::vector<std::vector<float>> got((size_t) nt, std::vector<float>((size_t) rows));
        float* gp[c::MAXT];
        for (int t = 0; t < nt; ++t) gp[t] = got[(size_t) t].data();
        c::q2_hi_rows_scalar(planes.data(), nblocks, ap, nt, gp, 0, rows);
        for (int t = 0; t < nt; ++t)
            for (int r = 0; r < rows; ++r) {
                double ref = 0.0, mag = 0.0;
                for (int b = 0; b < nblocks; ++b) {
                    const uint8_t* blk = &w[(size_t) r * row_bytes + (size_t) b * 18];
                    uint16_t dh;
                    std::memcpy(&dh, blk, 2);
                    const double d = f16(dh);
                    for (int i = 0; i < 64; ++i) {
                        const int code = (blk[2 + i / 4] >> (2 * (i % 4))) & 3;
                        const double v = d * ((double) (code & 2) - 0.5) * acts[t].scale[2 * b + i / 32] * acts[t].q[b * 64 + i];
                        ref += v;
                        mag += std::fabs(v);
                    }
                }
                if (std::fabs(ref - (double) got[(size_t) t][(size_t) r]) > 1e-4 * (mag + 1e-3)) {
                    std::printf("FAIL scalar nt=%d t=%d r=%d: %g vs ref %g\n", nt, t, r, (double) got[(size_t) t][(size_t) r], ref);
                    ++fail;
                }
            }
    }
    // (2) AVX2 vs scalar
    if (c::cpu_avx2_ok()) {
        for (int nt = 1; nt <= c::MAXT; ++nt) {
            std::vector<std::vector<float>> s((size_t) nt, std::vector<float>((size_t) rows)), v = s;
            float* sp[c::MAXT];
            float* vp[c::MAXT];
            for (int t = 0; t < nt; ++t) { sp[t] = s[(size_t) t].data(); vp[t] = v[(size_t) t].data(); }
            c::q2_hi_rows_scalar(planes.data(), nblocks, ap, nt, sp, 0, rows);
            c::q2_hi_rows_avx2(planes.data(), nblocks, ap, nt, vp, 0, 20);
            c::q2_hi_rows_avx2(planes.data(), nblocks, ap, nt, vp, 20, rows);
            for (int t = 0; t < nt; ++t)
                for (int r = 0; r < rows; ++r) {
                    const double a = s[(size_t) t][(size_t) r], b = v[(size_t) t][(size_t) r];
                    if (std::fabs(a - b) > 1e-5 * (std::fabs(a) + 1.0)) {
                        std::printf("FAIL avx2 nt=%d t=%d r=%d: %g vs %g\n", nt, t, r, b, a);
                        ++fail;
                    }
                }
        }
    } else {
        std::printf("q2_hi_parity: no AVX2, the AVX2 kernel is not checked\n");
    }
    // (3) the cache
    const uint8_t* p1 = c::q2_hi_plane_cached(w.data(), row_bytes, nblocks, rows);
    const uint8_t* p2 = c::q2_hi_plane_cached(w.data(), row_bytes, nblocks, rows);
    if (p1 == nullptr || p1 != p2 || std::memcmp(p1, planes.data(), planes.size()) != 0) {
        std::printf("FAIL cache\n");
        ++fail;
    }
    std::printf("q2_hi_parity: %s (%d failures)\n", fail ? "FAIL" : "PASS", fail);
    return fail ? 1 : 0;
}
