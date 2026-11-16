// src/kernels/cpu/q2_skip_parity.cpp - LEADS.md B1/B2 (STRATA_CPU_ACT_SPARSE / STRATA_CPU_ACT_SPARSE_UNION, opt-in, lossy).
//
//   1. act_sparse_apply: floor(p x nblk) blocks zeroed per token, the smallest-norm ones; with `uni` only the blocks
//      no token keeps; the returned mask = the blocks zero in every token;
//   2. q2_rows_skip on those activations = q2_rows_any on the same activations (the skipped blocks are zero), up to the
//      float summation order (a relative bound on the row's magnitude), alone and in windows of 1..8 tokens, cut into
//      row ranges.
// Synthetic data, CPU only; skipped (pass) on a CPU without AVX.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace c = strata::kernels::cpu;

int main() {
    if (!c::cpu_avx1_ok()) { std::printf("q2_skip_parity: no AVX on this CPU, skipped\n"); return 0; }
    std::mt19937 rng(264);
    std::normal_distribution<float> nd(0.f, 1.f);
    int fail = 0;
    const int n = c::FF;              // the down input width (10 blocks of 64)
    const int nblocks = n / 64;
    const size_t row_bytes = (size_t) nblocks * 18;
    const int rows = 77;
    std::vector<uint8_t> w((size_t) rows * row_bytes);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nblocks; ++b) {
            uint8_t* blk = &w[(size_t) r * row_bytes + (size_t) b * 18];
            const uint16_t d = (uint16_t) (0x1C00 + rng() % 0x0800);
            std::memcpy(blk, &d, 2);
            for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t) rng();
        }
    for (const bool uni : {false, true})
        for (const float p : {0.3f, 0.5f, 0.9f})
            for (int nt = 1; nt <= c::MAXT; ++nt) {
                std::vector<std::vector<float>> h((size_t) nt, std::vector<float>((size_t) n));
                for (int t = 0; t < nt; ++t)
                    for (int i = 0; i < n; ++i) h[(size_t) t][(size_t) i] = nd(rng) * (1.0f + (float) ((i / 64 + t) % 4));
                std::vector<std::vector<float>> h0 = h;
                float* hp[c::MAXT];
                for (int t = 0; t < nt; ++t) hp[t] = h[(size_t) t].data();
                const uint64_t skip = c::act_sparse_apply(hp, nt, n, p, uni);
                // (1) the zeroed blocks per token, the mask
                const int drop = (int) ((float) nblocks * p);
                uint64_t zero_all = (1ull << nblocks) - 1ull;
                for (int t = 0; t < nt; ++t) {
                    int zeroed = 0;
                    uint64_t zt = 0;
                    for (int b = 0; b < nblocks; ++b) {
                        bool z = true, same = true;
                        for (int i = 0; i < 64; ++i) {
                            const float v = h[(size_t) t][(size_t) (b * 64 + i)];
                            if (v != 0.0f) z = false;
                            if (v != h0[(size_t) t][(size_t) (b * 64 + i)]) same = false;
                        }
                        if (z) { ++zeroed; zt |= 1ull << b; }
                        else if (!same) { std::printf("FAIL p=%.1f uni=%d nt=%d: block %d changed, not zeroed\n", p, uni, nt, b); ++fail; }
                    }
                    if (!uni && zeroed != drop) { std::printf("FAIL p=%.1f nt=%d t=%d: %d blocks zeroed, want %d\n", p, nt, t, zeroed, drop); ++fail; }
                    if (uni && zeroed > drop) { std::printf("FAIL p=%.1f uni nt=%d t=%d: %d > %d zeroed\n", p, nt, t, zeroed, drop); ++fail; }
                    zero_all &= zt;
                }
                if (zero_all != skip) { std::printf("FAIL p=%.1f uni=%d nt=%d: mask %llx, want %llx\n", p, uni, nt, (unsigned long long) skip, (unsigned long long) zero_all); ++fail; }
                // (2) the skip rows against the full rows on the same (sparsified) activations
                static c::ActQ acts[c::MAXT];
                const c::ActQ* ap[c::MAXT];
                for (int t = 0; t < nt; ++t) { c::act_quant_any(h[(size_t) t].data(), n, acts[t]); ap[t] = &acts[t]; }
                std::vector<std::vector<float>> ref((size_t) nt, std::vector<float>((size_t) rows)), got = ref;
                float* rp[c::MAXT];
                float* gp[c::MAXT];
                for (int t = 0; t < nt; ++t) { rp[t] = ref[(size_t) t].data(); gp[t] = got[(size_t) t].data(); }
                c::q2_rows_any(w.data(), row_bytes, nblocks, ap, nt, rp, 0, rows);
                c::q2_rows_skip(w.data(), row_bytes, nblocks, ap, nt, gp, 0, 30, skip);
                c::q2_rows_skip(w.data(), row_bytes, nblocks, ap, nt, gp, 30, rows, skip);
                for (int t = 0; t < nt; ++t) {
                    double mag = 0.0;
                    for (int r = 0; r < rows; ++r) mag = std::fmax(mag, std::fabs((double) ref[(size_t) t][(size_t) r]));
                    for (int r = 0; r < rows; ++r) {
                        const double e = std::fabs((double) ref[(size_t) t][(size_t) r] - (double) got[(size_t) t][(size_t) r]);
                        if (e > 1e-5 * (mag + 1.0)) {
                            std::printf("FAIL p=%.1f uni=%d nt=%d t=%d r=%d: %g vs %g\n", p, uni, nt, t, r,
                                        (double) got[(size_t) t][(size_t) r], (double) ref[(size_t) t][(size_t) r]);
                            ++fail;
                            break;
                        }
                    }
                }
            }
    std::printf("q2_skip_parity: %s (%d failures)\n", fail ? "FAIL" : "PASS", fail);
    return fail ? 1 : 0;
}
