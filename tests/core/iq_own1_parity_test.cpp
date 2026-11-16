// tests/core/iq_own1_parity_test.cpp - STRATA_CPU_IQ_OWN1 (LEADS.md Part H, pszeder b2d00855; own code): the one-token IQ2_XXS /
// IQ2_XS rows (iq_own1_rows) against iq256x_rows at one token, which is ggml-cpu's AVX2 vec_dot's bits: every output must be
// BITWISE the same.  Random blocks (all byte values), fp16 block scales over a wide range, Q8_K activations with the full int8
// range (-128 included), row ranges with remainders, 1..12 blocks a row (the model's rows are 2560 / 10 blocks and 7168 / 28).
// Exit 0 = PASS, 1 = FAIL, 77 = no AVX2.  Plus an informational host timing (L2-resident), NOT a ledger number.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace c = strata::kernels::cpu;

namespace {
struct Q8K {   // ggml's block_q8_K
    float d;
    int8_t qs[256];
    int16_t bsums[16];
};
struct Rng {
    uint64_t s = 0x2545F4914F6CDD1Dull;
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
};
uint16_t half_scale(Rng& g) {   // 2^-9 .. 2^2, either sign
    return (uint16_t) (((g.next() & 1) << 15) | ((6 + g.next() % 12) << 10) | (g.next() & 0x3FF));
}
}  // namespace

int main() {
    if (!c::cpu_avx2_ok()) {
        std::printf("iq_own1_parity_test: no AVX2 - skipped\n");
        return 77;
    }
    Rng g;
    long long values = 0, bad = 0;
    for (int type : {16, 17}) {
        const size_t bb = type == 16 ? 66 : 74;
        for (int nb : {1, 2, 3, 10, 12, 28}) {
            const int rows = 131;
            const size_t rb = (size_t) nb * bb;
            std::vector<uint8_t> w(rows * rb);
            for (auto& b : w) b = (uint8_t) g.next();
            for (int r = 0; r < rows; ++r)
                for (int b = 0; b < nb; ++b) {
                    const uint16_t d = half_scale(g);
                    std::memcpy(&w[r * rb + (size_t) b * bb], &d, 2);
                }
            for (int rep = 0; rep < 8; ++rep) {
                std::vector<Q8K> y(nb);
                for (auto& blk : y) {
                    blk.d = (float) ((int) (g.next() % 2001) - 1000) * 1e-3f;
                    for (auto& q : blk.qs) q = (int8_t) (uint8_t) g.next();
                    for (auto& s : blk.bsums) s = 0;   // not read by the rows
                }
                const int r0 = rep == 0 ? 0 : (int) (g.next() % 40), r1 = rep == 0 ? rows : rows - (int) (g.next() % 40);
                std::vector<float> ref(rows, -7.f), got(rows, -7.f);
                const void* act[1] = {y.data()};
                float* po[1] = {ref.data()};
                c::iq256x_rows(type, w.data(), rb, nb * 256, act, 1, po, r0, r1);
                c::iq_own1_rows(type, w.data(), rb, nb * 256, y.data(), got.data(), r0, r1);   // out[r - r0]
                for (int r = r0; r < r1; ++r) {
                    uint32_t u1, u2;
                    std::memcpy(&u1, &ref[r], 4);
                    std::memcpy(&u2, &got[r - r0], 4);
                    ++values;
                    if (u1 != u2 && !(std::isnan(ref[r]) && std::isnan(got[r - r0]))) {
                        if (bad < 8) std::printf("  differ type %d nb %d row %d: %.9g (%08x) vs %.9g (%08x)\n", type, nb, r, ref[r], u1, got[r - r0], u2);
                        ++bad;
                    }
                }
            }
        }
    }
    std::printf("iq_own1_parity_test: %lld row outputs (IQ2_XXS, IQ2_XS), %lld differ\n", values, bad);
    for (int type : {16, 17}) {   // informational timing, one thread, L2-resident (not DRAM)
        const int nb = 10, rows = 1024;
        const size_t bb = type == 16 ? 66 : 74, rb = nb * bb;
        std::vector<uint8_t> w(rows * rb);
        for (auto& b : w) b = (uint8_t) g.next();
        std::vector<Q8K> y(nb);
        for (auto& blk : y) {
            blk.d = 0.01f;
            for (auto& q : blk.qs) q = (int8_t) (uint8_t) g.next();
        }
        const void* act[1] = {y.data()};
        std::vector<float> o(rows);
        float* po[1] = {o.data()};
        auto timeit = [&](bool own) {
            double best = 1e30;
            for (int k = 0; k < 30; ++k) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 20; ++i) {
                    if (own) c::iq_own1_rows(type, w.data(), rb, nb * 256, y.data(), o.data(), 0, rows);
                    else c::iq256x_rows(type, w.data(), rb, nb * 256, act, 1, po, 0, rows);
                }
                best = std::fmin(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 20);
            }
            return best / (rows * nb) * 1e9;
        };
        const double a = timeit(false), b = timeit(true);
        std::printf("iq_own1_parity_test: host timing type %d (L2-resident, informational): exact rows %.1f ns/block, own1 %.1f ns/block\n", type, a, b);
    }
    std::printf("iq_own1_parity_test: %s\n", bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
