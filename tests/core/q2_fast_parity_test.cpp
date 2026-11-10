// tests/core/q2_fast_parity_test.cpp - the default Q2_0 AVX2 row kernel (rows_fast_nt, the cheaper unpack and R rows at
// a time) against the legacy one (STRATA_Q2_LEGACY=1), bit for bit, on random blocks.
//
// Standalone: not wired into CMake.  Build by hand, with the CPU kernel sources (AVX2 flags as in CMakeLists.txt):
//   g++ -std=c++17 -O2 -mavx2 -mfma -mf16c -Iinclude tests/core/q2_fast_parity_test.cpp \
//       src/kernels/cpu/q2_avx2.cpp src/kernels/cpu/expert_layout.cpp <other objects it needs> -o q2_fast_parity
// Run with STRATA_Q2_LEGACY and STRATA_Q2_BITPLANE unset.  Exit 0 = every output identical in every bit.
//
// Cases: token counts 1..5 and 8, short rows (10 blocks, the down projection) and long ones (40 blocks, gate/up),
// row ranges that are not multiples of the group width, and a random 2-bit code byte in every position.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <immintrin.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

namespace c = strata::kernels::cpu;

namespace {

// one Q2_0 row: nblocks x (fp16 d, 16 code bytes)
void fill_row(uint8_t* row, int nblocks, std::mt19937& rng) {
    std::uniform_real_distribution<float> ud(-0.05f, 0.05f);
    for (int b = 0; b < nblocks; ++b) {
        uint8_t* blk = row + (size_t) b * 18;
        const __m128i h = _mm_cvtps_ph(_mm_set_ss(ud(rng)), 0);
        const uint16_t bits = (uint16_t) _mm_cvtsi128_si32(h);
        std::memcpy(blk, &bits, 2);
        for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t) rng();
    }
}

bool same_bits(const float* a, const float* b, int n) { return std::memcmp(a, b, sizeof(float) * (size_t) n) == 0; }

}  // namespace

int main() {
    if (std::getenv("STRATA_Q2_LEGACY") != nullptr || std::getenv("STRATA_Q2_BITPLANE") != nullptr) {
        std::printf("q2_fast_parity: unset STRATA_Q2_LEGACY and STRATA_Q2_BITPLANE for this test\n");
        return 1;
    }
    if (!c::cpu_avx2_ok()) {
        std::printf("q2_fast_parity: no AVX2 on this CPU, skipped\n");
        return 0;
    }
    std::mt19937 rng(0x2b0ull);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int rows_total = 13;                        // not a multiple of 4: the group tail runs
    const int token_counts[] = {1, 2, 3, 4, 5, 8};
    const int block_counts[] = {10, 40, 17};          // 10 and 17 take the grouped path, 40 the one-row path
    int fail = 0, cases = 0;
    for (int nblocks : block_counts) {
        const int n = nblocks * 64;                   // one 64-value block spans two 32-value chunks
        const size_t row_bytes = (size_t) nblocks * 18;
        std::vector<uint8_t> w((size_t) rows_total * row_bytes);
        fill_row(w.data(), rows_total * nblocks, rng);
        for (int nt : token_counts) {
            std::vector<std::unique_ptr<c::ActQ>> acts;
            std::vector<const c::ActQ*> ap;
            for (int t = 0; t < nt; ++t) {
                auto a = std::make_unique<c::ActQ>();
                std::vector<float> x(n);
                for (float& v : x) v = nd(rng);
                c::act_quant_q8_1_avx2(x.data(), n, *a);
                ap.push_back(a.get());
                acts.push_back(std::move(a));
            }
            const int ranges[][2] = {{0, rows_total}, {3, 11}, {5, 6}};
            for (const auto& rg : ranges) {
                std::vector<std::vector<float>> legacy(nt, std::vector<float>(rows_total, 0.f));
                std::vector<std::vector<float>> fast(nt, std::vector<float>(rows_total, 0.f));
                std::vector<float*> lo_out(nt), fa_out(nt);
                for (int t = 0; t < nt; ++t) { lo_out[t] = legacy[t].data(); fa_out[t] = fast[t].data(); }
                c::q2_0_gguf_rows_multi_avx2_legacy(w.data(), row_bytes, nblocks, ap.data(), nt, lo_out.data(), rg[0], rg[1]);
                c::q2_0_gguf_rows_multi_avx2(w.data(), row_bytes, nblocks, ap.data(), nt, fa_out.data(), rg[0], rg[1]);
                ++cases;
                for (int t = 0; t < nt; ++t)
                    if (!same_bits(legacy[t].data() + rg[0], fast[t].data() + rg[0], rg[1] - rg[0])) {
                        ++fail;
                        std::printf("MISMATCH nblocks=%d nt=%d token=%d rows=[%d,%d)\n", nblocks, nt, t, rg[0], rg[1]);
                    }
            }
        }
    }
    std::printf("q2_fast_parity: %d cases, %d mismatching outputs\n", cases, fail);
    return fail == 0 ? 0 : 1;
}
