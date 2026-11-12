// tests/core/q2_fast_parity_test.cpp - the fast exact AVX-2 Q2_0 row kernel (STRATA_Q2_FAST=1, ported from imanu86
// c3fc8f34d3) against the legacy one (q2_0_gguf_rows_multi_avx2_legacy): every output must be BITWISE the same, except the
// payload of a NaN output (a NaN in both; see the summary line).
//
// Random Q2_0 rows (all 256 code-byte values, fp16 scales over a wide range with zeros and subnormals), activations
// quantized by act_quant_q8_1_avx2 (zero chunks and magnitudes 2^-12..2^11), 1-9 tokens (the >4 split), the model's
// shapes (down: 10 blocks, the 4-row / 2-row groups; gate/up: 40 blocks, one row at a time) and odd ones, row
// ranges with remainders (the pool cuts rows anywhere), padded and unpadded rows.  A second pass per shape takes
// edge inputs: fp16 scales +-Inf, NaNs with random payloads and the largest normal; code bytes swept so every byte
// position sees all 256 values; activation chunks overwritten with raw scale / hx values (float subnormals, -0, +-Inf,
// NaN, FLT_MAX, random bit patterns) and raw int8 codes including -128.  Checked on the plain AVX-2 form and, where
// the CPU has AVX-VNNI, on the vpdpbusd form (fast vnni vs legacy vnni, and vs plain).  The fast kernel's corr form
// (q2_fast_corr_form) must be found, or the fast entry points would run the legacy rows and the test would prove
// nothing: -1 is a FAIL.
//
// Calls the kernels directly, so it needs no environment switch.  Exit 0 = PASS, 1 = FAIL, 77 = no AVX2 (skipped).
// Once it passes on the AVX2 machines, STRATA_Q2_FAST becomes the default (STRATA_Q2_LEGACY=1 then restores the old
// kernel for A/B).
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace c = strata::kernels::cpu;

namespace {

struct Rng {
    uint64_t s = 0x9E3779B97F4A7C15ull;
    uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    float uni() { return (float) ((next() >> 40) & 0xFFFFFF) / 16777216.f; }   // [0, 1)
    float gauss() {
        float a = 0.f;
        for (int i = 0; i < 6; ++i) a += uni();
        return a - 3.f;
    }
};

uint16_t rand_half(Rng& g, bool edge) {
    const uint64_t r = g.next();
    const uint16_t sign = (uint16_t) ((r & 1) << 15);
    switch ((r >> 1) % 32) {
        case 0: return sign;                                     // +-0
        case 1: return (uint16_t) (sign | ((r >> 8) & 0x3FF));   // subnormal
        case 2: if (edge) return (uint16_t) (sign | 0x7C00); [[fallthrough]];                          // +-Inf
        case 3: if (edge) return (uint16_t) (sign | 0x7C00 | (1 + (r >> 8) % 0x3FF)); [[fallthrough]];  // NaN
        case 4: if (edge) return (uint16_t) (sign | 0x7BFF); [[fallthrough]];                          // max normal
        default: {
            const uint16_t e = (uint16_t) (5 + (r >> 8) % 21);    // 2^-10 .. 2^10
            return (uint16_t) (sign | (e << 10) | ((r >> 16) & 0x3FF));
        }
    }
}

void fill_rows(Rng& g, std::vector<uint8_t>& w, int rows, int nblocks, size_t row_bytes, bool edge) {
    w.assign((size_t) rows * row_bytes, 0);
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < nblocks; ++b) {
            uint8_t* blk = w.data() + (size_t) r * row_bytes + (size_t) b * 18;
            const uint16_t d = rand_half(g, edge);
            std::memcpy(blk, &d, 2);
            const int n = r * nblocks + b;   // edge: 256 consecutive blocks give every byte position all 256 values
            for (int i = 0; i < 16; ++i) blk[2 + i] = edge ? (uint8_t) (n + 17 * i) : (uint8_t) g.next();
        }
}

float raw_float(Rng& g) {
    const uint64_t r = g.next();
    const float sign = r & 1 ? -1.f : 1.f;
    switch ((r >> 1) % 8) {
        case 0: return sign * 0.f;
        case 1: return sign * std::ldexp(1.f + (float) ((r >> 8) % 1000) / 1000.f, -140);   // float subnormal
        case 2: return sign * INFINITY;
        case 3: return sign * FLT_MAX;
        case 4: {
            const uint32_t u = 0x7F800000u | (r & 1 ? 0x80000000u : 0u) | (uint32_t) (1 + (r >> 8) % 0x7FFFFF);   // NaN
            float f;
            std::memcpy(&f, &u, 4);
            return f;
        }
        case 5: {
            const uint32_t u = (uint32_t) (r >> 16);   // any bit pattern
            float f;
            std::memcpy(&f, &u, 4);
            return f;
        }
        default: return sign * std::ldexp(1.f, (int) ((r >> 8) % 60) - 30);
    }
}

void fill_act(Rng& g, c::ActQ& a, int n, bool edge) {
    std::vector<float> x((size_t) n);
    for (int k = 0; k < n / 32; ++k) {
        const uint64_t mode = g.next() % 8;
        const float mag = mode == 0 ? 0.f : std::ldexp(1.f, (int) (g.next() % 24) - 12);
        for (int j = 0; j < 32; ++j) x[(size_t) k * 32 + j] = mag * g.gauss();
    }
    c::act_quant_q8_1_avx2(x.data(), n, a);
    if (!edge) return;
    for (int k = 0; k < n / 32; ++k) {   // the row kernels read only q, scale and hx
        if (g.next() % 4 != 0) continue;
        a.scale[k] = raw_float(g);
        a.hx[k] = raw_float(g);
        if (g.next() % 2 == 0)
            for (int j = 0; j < 32; ++j) a.q[k * 32 + j] = (int8_t) (uint8_t) g.next();   // -128 included
    }
}

using RowsFn = void (*)(bool, const uint8_t*, size_t, int, const c::ActQ* const*, int, float* const*, int, int);

}  // namespace

int main() {
    if (!c::cpu_avx2_ok()) {
        std::printf("q2_fast_parity_test: no AVX2 on this CPU - skipped\n");
        return 77;
    }
    const bool vnni = c::cpu_avxvnni_ok();
    const int form = c::q2_fast_corr_form(false), form_vnni = vnni ? c::q2_fast_corr_form(true) : form;
    std::printf("q2_fast_parity_test: legacy corr form %s%s\n", form == 1 ? "fma" : form == 0 ? "rounded product + add" : "NOT FOUND",
                form_vnni != form ? " (AVX-VNNI copy differs)" : "");
    if (form < 0 || form_vnni < 0) {
        std::printf("q2_fast_parity_test: the fast kernel matches neither corr form -> FAIL\n");
        return 1;
    }
    Rng g;
    long long cases = 0, values = 0, differ = 0;
    const int shapes[][2] = {{2560, 10}, {640, 40}, {37, 1}, {64, 2}, {129, 7}, {200, 40}, {33, 3}, {75, 16}, {19, 17}};
    long long edge_values = 0, nan_payload = 0;
    for (const auto& sh : shapes) {
        for (int pass = 0; pass < 4; ++pass) {
            const int pad = pass % 2 ? 6 : 0;
            const bool edge = pass >= 2;
            const int rows = sh[0], nb = sh[1];
            const size_t row_bytes = (size_t) nb * 18 + (size_t) pad;
            std::vector<uint8_t> w;
            fill_rows(g, w, rows, nb, row_bytes, edge);
            static c::ActQ acts[9];
            for (int rep = 0; rep < 8; ++rep) {
                for (auto& a : acts) fill_act(g, a, nb * 64, edge);
                for (int nt = 1; nt <= 9; ++nt) {
                    const int r0 = rep == 0 ? 0 : (int) (g.next() % (uint64_t) rows);
                    const int r1 = rep == 0 ? rows : r0 + (int) (g.next() % (uint64_t) (rows - r0 + 1));
                    const c::ActQ* ap[9];
                    for (int t = 0; t < nt; ++t) ap[t] = &acts[t];
                    // (reference, candidate) pairs: plain fast vs plain legacy; with VNNI also vnni fast vs vnni
                    // legacy and vnni fast vs plain legacy (dot4's two forms are the same integers)
                    struct Pair { RowsFn ref; bool ref_vnni; RowsFn cand; bool cand_vnni; const char* name; };
                    const Pair pairs[] = {
                        {c::q2_0_gguf_rows_multi_avx2_v, false, c::q2_0_gguf_rows_multi_avx2_fast_v, false, "plain"},
                        {c::q2_0_gguf_rows_multi_avx2_v, true, c::q2_0_gguf_rows_multi_avx2_fast_v, true, "vnni"},
                        {c::q2_0_gguf_rows_multi_avx2_v, false, c::q2_0_gguf_rows_multi_avx2_fast_v, true, "vnni/plain"},
                    };
                    for (int pi = 0; pi < (vnni ? 3 : 1); ++pi) {
                        const Pair& P = pairs[pi];
                        std::vector<float> o1((size_t) nt * rows, -7.f), o2((size_t) nt * rows, -7.f);
                        float* p1[9];
                        float* p2[9];
                        for (int t = 0; t < nt; ++t) {
                            p1[t] = o1.data() + (size_t) t * rows;
                            p2[t] = o2.data() + (size_t) t * rows;
                        }
                        P.ref(P.ref_vnni, w.data(), row_bytes, nb, ap, nt, p1, r0, r1);
                        P.cand(P.cand_vnni, w.data(), row_bytes, nb, ap, nt, p2, r0, r1);
                        for (size_t i = 0; i < o1.size(); ++i) {   // rows outside [r0, r1) must stay -7 in both
                            uint32_t u1, u2;
                            std::memcpy(&u1, &o1[i], 4);
                            std::memcpy(&u2, &o2[i], 4);
                            if (u1 != u2 && std::isnan(o1[i]) && std::isnan(o2[i])) {
                                ++nan_payload;   // both NaN: only the payload differs (see the summary line)
                            } else if (u1 != u2) {
                                if (differ < 8)
                                    std::printf("  differ [%s%s]: rows %d nb %d pad %d nt %d [%d,%d) idx %zu: %.9g (%08x) vs %.9g (%08x)\n",
                                                P.name, edge ? ", edge" : "", rows, nb, pad, nt, r0, r1, i, o1[i], u1, o2[i], u2);
                                ++differ;
                            }
                        }
                        values += (long long) nt * (r1 - r0);
                        if (edge) edge_values += (long long) nt * (r1 - r0);
                        ++cases;
                    }
                }
            }
        }
    }
    // A NaN output (a NaN or infinite scale in the inputs) is NaN in both kernels, but its payload can differ: x86 passes
    // on the first NaN operand, and the compiler orders a commutative add or an FMA form as it likes (seen with
    // STRATA_HOST_NATIVE and in the default clang build).  So the parity is bitwise except for NaN payloads, whatever the flags.
    // Nothing downstream reads a payload, so it is reported, not failed.
    std::printf("q2_fast_parity_test (%s): %lld cases, %lld row outputs (%lld with edge inputs), %lld differ, "
                "%lld NaN outputs with another payload -> %s\n", vnni ? "plain + AVX-VNNI" : "plain AVX-2", cases, values,
                edge_values, differ, nan_payload, differ ? "FAIL" : "PASS");
    return differ ? 1 : 0;
}
