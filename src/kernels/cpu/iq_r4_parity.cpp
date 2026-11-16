// src/kernels/cpu/iq_r4_parity.cpp - STRATA_CPU_R4 (opt-in, LEADS.md B123; ik_llama.cpp's row-interleaved "_R4" layouts):
// iq256_gu_rows_r4 on an R4 copy of an IQ2_XS expert (r4_repack_rows) against iq256_gu_rows on the plain blob, bit for
// bit, at 1..8 tokens and on row ranges that start and end inside a group of four.  Also checks the repack itself
// (every block where the layout says).  No GPU, no model: random blocks and random activations quantized as the engine
// quantizes them (q8k_quant_avx2).  SKIP (exit 77) without AVX2.
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace cpu = strata::kernels::cpu;

int main() {
    if (!cpu::cpu_avx2_ok()) {
        std::printf("SKIP: no AVX2\n");
        return 77;
    }
    constexpr int kH = 2560, kFF = 640, kB = 74, kMaxT = 8;
    constexpr size_t row = (size_t) (kH / 256) * kB;   // 740 bytes
    const size_t up_off = (size_t) kFF * row;
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<uint8_t> blob(2 * up_off);
    for (auto& b : blob) b = (uint8_t) byte(rng);
    for (size_t i = 0; i < blob.size() / kB; ++i) {   // a small positive f16 d at the head of every block
        const uint16_t d = (uint16_t) (0x2000 + byte(rng));
        std::memcpy(blob.data() + i * kB, &d, 2);
    }
    std::vector<uint8_t> blob4(blob.size());
    cpu::r4_repack_rows(blob.data(), row, kB, kFF, blob4.data());
    cpu::r4_repack_rows(blob.data() + up_off, row, kB, kFF, blob4.data() + up_off);
    int bad = 0;
    for (int r = 0; r < kFF && !bad; ++r)
        for (size_t i = 0; i < row / kB; ++i)
            if (std::memcmp(blob.data() + (size_t) r * row + i * kB,
                            blob4.data() + (size_t) (r / 4) * 4 * row + (i * 4 + (size_t) (r % 4)) * kB, kB) != 0) {
                std::printf("repack: row %d block %zu is not where the layout says  FAIL\n", r, i);
                ++bad;
                break;
            }

    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<std::vector<uint8_t>> acts(kMaxT, std::vector<uint8_t>((size_t) (kH / 256) * 292));
    for (auto& a : acts) {
        std::vector<float> x(kH);
        for (auto& v : x) v = nd(rng);
        cpu::q8k_quant_avx2(x.data(), a.data(), kH);
    }
    const void* act[kMaxT];
    for (int t = 0; t < kMaxT; ++t) act[t] = acts[(size_t) t].data();
    const int ranges[][2] = {{0, kFF}, {3, 77}, {5, 6}, {1, 4}, {636, 640}, {64, 192}};
    for (int nt = 1; nt <= kMaxT; ++nt)
        for (const auto& rg : ranges) {
            std::vector<std::vector<float>> a(nt, std::vector<float>(kFF, -1.f)), b(nt, std::vector<float>(kFF, -1.f));
            float* fa[kMaxT];
            float* fb[kMaxT];
            for (int t = 0; t < nt; ++t) { fa[t] = a[(size_t) t].data(); fb[t] = b[(size_t) t].data(); }
            cpu::iq256_gu_rows(17, blob.data(), row, up_off, kH, act, nt, fa, rg[0], rg[1]);
            cpu::iq256_gu_rows_r4(17, blob4.data(), row, up_off, kH, act, nt, fb, rg[0], rg[1]);
            for (int t = 0; t < nt; ++t)
                if (std::memcmp(fa[t], fb[t], kFF * sizeof(float)) != 0) {
                    std::printf("nt=%d rows [%d, %d) token %d: the R4 rows differ from iq256_gu_rows  FAIL\n", nt, rg[0], rg[1], t);
                    ++bad;
                }
        }
    std::printf("%s: IQ2_XS gate/up R4 rows against the plain rows, 1..%d tokens, %zu row ranges\n", bad ? "FAIL" : "PASS",
                kMaxT, sizeof ranges / sizeof ranges[0]);
    return bad ? 1 : 0;
}
