// src/kernels/cpu/q2_fast_parity.cpp - the opt-in STRATA_Q2_FAST AVX-2 Q2_0 row kernel (imanu86, c3fc8f34d3) against
// the legacy one (q2_0_gguf_rows_multi_avx2_legacy): every output must be BITWISE the same.  Standalone; written
// to be run on an idle machine (it is not run by the commit that added it).
//
// Random Q2_0 rows (random codes, fp16 scales over 2^-10..2^10 with zeros and subnormals), activations quantized by
// act_quant_q8_1_avx2 (zero chunks and mixed magnitudes), 1-9 tokens, row ranges with remainders, padded and
// unpadded rows, short (10 blocks: the 4-row / 2-row groups) and long (40 blocks) matrices.  The fast kernel is
// called directly, so the test does not depend on STRATA_Q2_FAST; with AVX-VNNI present both copies are checked.
// Exit 0 pass, 1 mismatch, 77 skipped (no AVX2).  Once this passes on the target CPUs STRATA_Q2_FAST becomes default.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace c = strata::kernels::cpu;

namespace {

struct Rng {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    float uni() { return (float) ((next() >> 40) & 0xFFFFFF) / 16777216.f; }
    float gauss() { float a = 0.f; for (int i = 0; i < 6; ++i) a += uni(); return a - 3.f; }
};

uint16_t rand_half(Rng& g) {
    const uint64_t r = g.next();
    const uint16_t sign = (uint16_t) ((r & 1) << 15);
    switch ((r >> 1) % 32) {
        case 0: return sign;                                       // zero
        case 1: return (uint16_t) (sign | ((r >> 8) & 0x3FF));     // subnormal
        default: {
            const uint16_t e = (uint16_t) (5 + (r >> 8) % 21);     // 2^-10 .. 2^10
            return (uint16_t) (sign | (e << 10) | ((r >> 16) & 0x3FF));
        }
    }
}

void fill_rows(Rng& g, std::vector<uint8_t>& w, int rows, int nb, size_t row_bytes) {
    w.assign((size_t) rows * row_bytes, 0);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nb; ++b) {
            uint8_t* blk = w.data() + (size_t) r * row_bytes + (size_t) b * 18;
            const uint16_t d = rand_half(g);
            std::memcpy(blk, &d, 2);
            for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t) g.next();
        }
}

void fill_act(Rng& g, c::ActQ& a, int n) {
    std::vector<float> x((size_t) n);
    for (int k = 0; k < n / 32; ++k) {
        const float mag = g.next() % 8 == 0 ? 0.f : std::ldexp(1.f, (int) (g.next() % 24) - 12);
        for (int j = 0; j < 32; ++j) x[(size_t) k * 32 + j] = mag * g.gauss();
    }
    c::act_quant_q8_1_avx2(x.data(), n, a);
}

}  // namespace

int main() {
    if (!c::cpu_avx2_ok()) {
        std::printf("q2_fast_parity: no AVX2 on this CPU - skipped\n");
        return 77;
    }
    Rng g;
    long long cases = 0, values = 0, differ = 0;
    const int shapes[][2] = {{2560, 10}, {640, 40}, {37, 1}, {64, 2}, {129, 7}, {200, 40}, {33, 3}};
    const int nvar = c::cpu_avxvnni_ok() ? 2 : 1;   // the AVX2 copy, and the AVX-VNNI copy where the CPU has it
    for (const auto& sh : shapes)
        for (int pad = 0; pad <= 6; pad += 6) {
            const int rows = sh[0], nb = sh[1];
            const size_t row_bytes = (size_t) nb * 18 + (size_t) pad;
            std::vector<uint8_t> w;
            fill_rows(g, w, rows, nb, row_bytes);
            static c::ActQ acts[9];
            for (int rep = 0; rep < 4; ++rep) {
                for (auto& a : acts) fill_act(g, a, nb * 64);
                for (int nt = 1; nt <= 9; ++nt) {
                    const int r0 = rep == 0 ? 0 : (int) (g.next() % (uint64_t) rows);
                    const int r1 = rep == 0 ? rows : r0 + (int) (g.next() % (uint64_t) (rows - r0 + 1));
                    const c::ActQ* ap[9];
                    for (int t = 0; t < nt; ++t) ap[t] = &acts[t];
                    for (int v = 0; v < nvar; ++v) {
                        const bool vnni = v == 1;
                        std::vector<float> o1((size_t) nt * rows, -7.f), o2((size_t) nt * rows, -7.f);
                        float* p1[9];
                        float* p2[9];
                        for (int t = 0; t < nt; ++t) {
                            p1[t] = o1.data() + (size_t) t * rows;
                            p2[t] = o2.data() + (size_t) t * rows;
                        }
                        c::q2_0_gguf_rows_multi_avx2_v(vnni, w.data(), row_bytes, nb, ap, nt, p1, r0, r1);
                        c::q2_0_gguf_rows_multi_avx2_fast_v(vnni, w.data(), row_bytes, nb, ap, nt, p2, r0, r1);
                        // the whole buffer, so a write outside [r0, r1) is caught too
                        for (size_t i = 0; i < o1.size(); ++i) {
                            uint32_t u1, u2;
                            std::memcpy(&u1, &o1[i], 4);
                            std::memcpy(&u2, &o2[i], 4);
                            if (u1 != u2) {
                                if (differ < 5)
                                    std::printf("  differ: vnni %d rows %d nb %d pad %d nt %d [%d,%d) idx %zu: %.9g vs %.9g\n",
                                                (int) vnni, rows, nb, pad, nt, r0, r1, i, o1[i], o2[i]);
                                ++differ;
                            }
                        }
                        values += (long long) nt * (r1 - r0);
                        ++cases;
                    }
                }
            }
        }
    std::printf("q2_fast_parity: %lld cases, %lld row outputs, %lld differ -> %s\n", cases, values, differ,
                differ ? "FAIL" : "PASS");
    return differ ? 1 : 0;
}
