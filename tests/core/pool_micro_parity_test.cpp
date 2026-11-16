// tests/core/pool_micro_parity_test.cpp - lead B110 (LEADS.md), the CPU pool micro pieces, each behind its own env, host test:
//  * STRATA_POOL_MICRO_Q2FMA2: the one-token Q2_0 row with two accumulator chains (q2_0_gguf_rows_one_split_avx2_v) against
//    the legacy one-token row.  Reassociated, so NOT bitwise: the difference must stay within 1e-5 of the largest output of
//    the case (class "rounding").
//  * STRATA_POOL_MICRO_SWIGLU: swiglu_mul_avx2 against g / (1 + std::exp(-g)) * u: relative 5e-6 (+ 1e-30 absolute) over
//    g in [-30, 30] and a sweep of the clamp range.
// Plus a micro-timing line (host, informational, NOT a ledger number: tools/pancake/probes style).  Exit 0 PASS, 1 FAIL, 77 no AVX2.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <chrono>
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
}  // namespace

int main() {
    if (!c::cpu_avx2_ok()) {
        std::printf("pool_micro_parity_test: no AVX2 - skipped\n");
        return 77;
    }
    Rng g;
    int bad = 0;
    double worst = 0;
    long long vals = 0;
    const bool vnni = c::cpu_avxvnni_ok();
    for (int nb : {10, 40, 1, 3, 17}) {
        const int rows = 257;
        const size_t rb = (size_t) nb * 18;
        std::vector<uint8_t> w(rows * rb);
        for (int r = 0; r < rows; ++r)
            for (int b = 0; b < nb; ++b) {
                uint8_t* blk = w.data() + r * rb + (size_t) b * 18;
                const uint16_t d = (uint16_t) ((8 + g.next() % 12) << 10 | (g.next() & 0x3FF));   // 2^-7..2^4, positive
                std::memcpy(blk, &d, 2);
                for (int i = 0; i < 16; ++i) blk[2 + i] = (uint8_t) g.next();
            }
        for (int rep = 0; rep < 6; ++rep) {
            static c::ActQ a;
            std::vector<float> x((size_t) nb * 64);
            for (auto& v : x) v = g.gauss() * (rep ? 1.f : 30.f);
            c::act_quant_q8_1_avx2(x.data(), nb * 64, a);
            for (int vn = 0; vn <= (vnni ? 1 : 0); ++vn) {
                std::vector<float> ref(rows), got(rows, -7.f);
                float* pr[1] = {ref.data()};
                const c::ActQ* ap[1] = {&a};
                c::q2_0_gguf_rows_multi_avx2_v(vn != 0, w.data(), rb, nb, ap, 1, pr, 0, rows);
                c::q2_0_gguf_rows_one_split_avx2_v(vn != 0, w.data(), rb, nb, &a, got.data(), 3, rows - 5);
                float mx = 0.f;
                for (float v : ref) mx = std::fmax(mx, std::fabs(v));
                for (int r = 0; r < rows; ++r) {
                    if (r < 3 || r >= rows - 5) { if (got[r] != -7.f) ++bad; continue; }   // outside [r0, r1) untouched
                    const double e = std::fabs((double) ref[r] - got[r]);
                    worst = std::fmax(worst, e / (mx + 1e-30));
                    if (e > 1e-5 * mx + 1e-30) { if (bad < 5) std::printf("  q2 differ nb %d row %d: %g vs %g\n", nb, r, ref[r], got[r]); ++bad; }
                    ++vals;
                }
            }
        }
    }
    std::printf("pool_micro_parity_test: q2 split row: %lld values, worst error %.3g of the case max, %d bad\n", vals, worst, bad);
    // SwiGLU
    const int n = 1 << 16;
    std::vector<float> gg(n), uu(n), o(n);
    for (int i = 0; i < n; ++i) {
        gg[i] = i < n / 2 ? g.gauss() * 6.f : -100.f + 200.f * (float) i / (float) n;   // gauss and a sweep over [-100, 100]
        uu[i] = g.gauss();
    }
    c::swiglu_mul_avx2(gg.data(), uu.data(), o.data(), n - 3);   // a tail of 3 (scalar)
    double sw = 0;
    int sbad = 0;
    for (int i = 0; i < n - 3; ++i) {
        const float want = (gg[i] / (1.f + std::exp(-gg[i]))) * uu[i];
        const double e = std::fabs((double) want - o[i]);
        sw = std::fmax(sw, e / (std::fabs(want) + 1e-30));
        if (e > 5e-6 * std::fabs(want) + 1e-30) { if (sbad < 5) std::printf("  swiglu g %g u %g: %g vs %g\n", gg[i], uu[i], want, o[i]); ++sbad; }
    }
    std::printf("pool_micro_parity_test: swiglu: %d values, worst relative error %.3g, %d bad\n", n - 3, sw, sbad);
    // informational timing, host only
    {
        const int nb = 40, rows = 4096;
        const size_t rb = (size_t) nb * 18;
        std::vector<uint8_t> w(rows * rb);
        for (auto& b : w) b = (uint8_t) g.next();
        for (int r = 0; r < rows; ++r)
            for (int b = 0; b < nb; ++b) { w[r * rb + (size_t) b * 18] = 0; w[r * rb + (size_t) b * 18 + 1] = 0x3C; }
        static c::ActQ a;
        std::vector<float> x((size_t) nb * 64);
        for (auto& v : x) v = g.gauss();
        c::act_quant_q8_1_avx2(x.data(), nb * 64, a);
        std::vector<float> out(rows);
        float* po[1] = {out.data()};
        const c::ActQ* ap[1] = {&a};
        auto t = [&](bool split) {
            double best = 1e30;
            for (int k = 0; k < 20; ++k) {
                const auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < 20; ++i) {
                    if (split) c::q2_0_gguf_rows_one_split_avx2_v(false, w.data(), rb, nb, &a, out.data(), 0, rows);
                    else c::q2_0_gguf_rows_multi_avx2_v(false, w.data(), rb, nb, ap, 1, po, 0, rows);
                }
                best = std::fmin(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 20);
            }
            return best / rows * 1e9;
        };
        const double a0 = t(false), a1 = t(true);
        std::printf("pool_micro_parity_test: host timing (L2-resident, %d blocks/row, informational): one chain %.1f ns/row, two chains %.1f ns/row\n",
                    nb, a0, a1);
    }
    const bool ok = bad == 0 && sbad == 0;
    std::printf("pool_micro_parity_test: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
