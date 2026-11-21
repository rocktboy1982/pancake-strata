// src/prefill/gdn_wmma_ck_test.cpp - host test of STRATA_GDN_WMMA_PREFILL (LEADS.md Part S S5): no GPU.
//   1. routing: unset / 0 never reaches the new kernels (the default path is byte-identical because it never calls them); gfx12,
//      STRATA_GDN_CHUNKED=0, short and unaligned calls are refused.
//   2. index math the kernel shares through include/strata/prefill/gdn_wmma_ck.hpp: every fragment read is 16-byte aligned and in
//      range, the T / P word staging covers every word of both matrices exactly once, the LDS budget.
//   3. binary16 rounding against known values.
//   4. the chunked algorithm (CPU reference with the kernels' operand roundings and tile order) against the sequential recurrence in
//      double, on random inputs of the real shapes (S = 128): algebra exact in FP32 at chunks 16 / 32 / 64, FP16 operands within the
//      bounds tests/hip/gdn_chunked.cpp holds the GPU kernels to (rms <= 3e-3, max <= 2% of the largest value), and chunk 32 not
//      worse than chunk 16 beyond a factor.
#include "strata/prefill/gdn_wmma_ck.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace strata::prefill::gdn_ck;

static int g_fail = 0;
#define CHECK(c)                                                                          \
    do {                                                                                  \
        if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_fail; }   \
    } while (0)

static void test_routing() {
    CHECK(parse_env(nullptr) == -1);
    CHECK(parse_env("") == -1);
    CHECK(parse_env("1") == 1);
    CHECK(parse_env("0") == 0);
    CHECK(!route(-1, -1, true, 4096, 6144));   // unset: the default path
    CHECK(!route(-1, 1, true, 4096, 6144));    // the shipped preset (STRATA_GDN_CHUNKED=1) alone
    CHECK(!route(0, 1, true, 4096, 6144));
    CHECK(route(1, 1, true, 4096, 6144));
    CHECK(route(1, -1, true, 4096, 6144));     // implies the chunked path when STRATA_GDN_CHUNKED is unset
    CHECK(!route(1, 0, true, 4096, 6144));     // an explicit STRATA_GDN_CHUNKED=0 wins
    CHECK(!route(1, 1, false, 4096, 6144));    // gfx12 / other parts keep their kernels
    CHECK(!route(1, 1, true, 63, 6144));       // below the chunked minimum
    CHECK(route(1, 1, true, 64, 6144));
    CHECK(!route(1, 1, true, 4096, 6145));     // the output rows are stored 16 bytes at a time
    CHECK(route(1, 1, true, 4096, 6152));
}

static void test_index_math() {
    // fragment reads: 16 halves from offset o: o % 8 == 0 (16-byte aligned), inside the buffer
    const int k_halves = kCk * kKP, kt_halves = kS * kKTP, tp_halves = kCk * kTPP;
    for (int u = 0; u < kNt; ++u)
        for (int c = 0; c < 16; ++c) {
            for (int j = 0; j < kS / 16; ++j) {
                const int o = off_k(u, c, j);
                CHECK(o % 8 == 0 && o >= 0 && o + 16 <= k_halves);
            }
            for (int u2 = 0; u2 < kNt; ++u2) {
                const int o = off_tp(u, c, u2);
                CHECK(o % 8 == 0 && o >= 0 && o + 16 <= tp_halves);   // T at 0, P at tp_halves: the second buffer is the same size
            }
        }
    for (int j = 0; j < kS / 16; ++j)
        for (int c = 0; c < 16; ++c)
            for (int u = 0; u < kNt; ++u) {
                const int o = off_kt(j, c, u);
                CHECK(o % 8 == 0 && o >= 0 && o + 16 <= kt_halves);
            }
    // tp_base / tp_off: the kernel's split of tp_lds_half
    for (int tid = 0; tid < kThreads; ++tid)
        for (int i = 0; i < 4; ++i) CHECK(tp_base(tid) + tp_off(i) == tp_lds_half(tid, i));
    // the staging stores of the scan: skt[(sr + i) * kKTP + st + 16u] (sr = 8 * (tid & 15), i < 8, st = tid >> 4): every (k row, token)
    // of the chunk exactly once, and a read tile of off_kt(j, c, u) covers the tokens 16u..16u+15 of row 16j + c
    {
        std::vector<int> hit(size_t(kS) * kCk, 0);
        for (int tid = 0; tid < kThreads; ++tid)
            for (int u = 0; u < kNt; ++u)
                for (int i = 0; i < 8; ++i) {
                    const int row = 8 * (tid & 15) + i, tok = (tid >> 4) + 16 * u;
                    CHECK(row < kS && tok < kCk);
                    ++hit[size_t(row) * kCk + tok];
                }
        for (int h : hit) CHECK(h == 1);
        // sk / sq: a thread writes 8 halves at (st + 16u) * kKP + sr: 16-byte aligned, every (token, 8-column slice) once
        std::vector<int> hs(size_t(kCk) * (kS / 8), 0);
        for (int tid = 0; tid < kThreads; ++tid)
            for (int u = 0; u < kNt; ++u) {
                const int o = ((tid >> 4) + 16 * u) * kKP + 8 * (tid & 15);
                CHECK(o % 8 == 0 && o + 8 <= k_halves);
                ++hs[size_t((tid >> 4) + 16 * u) * (kS / 8) + (tid & 15)];
            }
        for (int h : hs) CHECK(h == 1);
    }
    // T / P word staging: global word w of the (chunk, head) block -> LDS half; T = matrix 0, P = matrix 1; every word once, and the
    // half equals the (row, column) of the row-major kCk x kCk global matrix at the LDS pitch
    {
        std::vector<int> hit(size_t(2 * kTpWords), 0);
        for (int tid = 0; tid < kThreads; ++tid)
            for (int i = 0; i < 4; ++i) {
                const int w = tp_word(tid, i);
                CHECK(w >= 0 && w < 2 * kTpWords);
                ++hit[size_t(w)];
                const int mat = w / (kCk * kCk / 2), within = w % (kCk * kCk / 2);   // global halves 2w, 2w + 1
                const int row = (2 * within) / kCk, col = (2 * within) % kCk;
                const int want = mat * (kCk * kTPP) + row * kTPP + col;
                CHECK(tp_lds_half(tid, i) == want);
                CHECK(tp_lds_half(tid, i) % 2 == 0);   // a 4-byte LDS store
                CHECK(col % 2 == 0 && col + 1 < kCk);   // both halves of the word are in one row
            }
        // the block holds kTpWords words in total, not 2 * kTpWords: T + P = 2 * kCk * kCk halves = kCk * kCk words
        for (int w = 0; w < kTpWords; ++w) CHECK(hit[size_t(w)] == 1);
        for (int w = kTpWords; w < 2 * kTpWords; ++w) CHECK(hit[size_t(w)] == 0);
    }
    CHECK(kLdsBytes <= 65536);
    CHECK(kLdsBytes64 > 65536);
    std::printf("LDS bytes: chunk 32 %d, chunk 64 would be %d (limit 65536)\n", kLdsBytes, kLdsBytes64);
}

static void test_f16r() {
    CHECK(f16r(1.0f) == 1.0f);
    CHECK(f16r(1.0f + 0x1p-11f) == 1.0f);                 // tie to even
    CHECK(f16r(1.0f + 3 * 0x1p-11f) == 1.0f + 0x1p-9f);   // tie to even (up)
    CHECK(f16r(1.0f + 0x1p-11f + 0x1p-20f) == 1.0f + 0x1p-10f);
    CHECK(f16r(65504.0f) == 65504.0f);
    CHECK(std::isinf(f16r(65520.0f)));
    CHECK(f16r(0x1p-24f) == 0x1p-24f);
    CHECK(f16r(0x1p-25f) == 0.0f);                        // tie to even (zero)
    CHECK(f16r(-0.333251953125f) == -0.333251953125f);    // exactly representable
    float worst = 0.0f;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0, 1);
    for (int i = 0; i < 20000; ++i) {
        const float x = nd(rng);
        if (std::fabs(x) > 1e-3f) worst = std::max(worst, std::fabs(f16r(x) - x) / std::fabs(x));
    }
    CHECK(worst <= 0x1p-11f * 1.0001f);
}

struct Err {
    double sq = 0, ref = 0, mx = 0, mref = 0;
    bool finite = true;
    void add(double a, double b) {
        finite &= std::isfinite(a) && std::isfinite(b);
        sq += (a - b) * (a - b); ref += a * a; mx = std::max(mx, std::fabs(a - b)); mref = std::max(mref, std::fabs(a));
    }
    double rms() const { return std::sqrt(sq / std::max(ref, 1e-30)); }
    bool ok(double rms_bound) const { return finite && rms() <= rms_bound && mx <= 0.02 * mref + 1e-5; }
};

struct Data {
    int64_t T;
    std::vector<float> q, k, v, gate, beta, z, gamma, state;
    Data(int64_t t, float decay, float vscale, bool no_update, uint32_t seed) : T(t), q(t * kS), k(t * kS), v(t * kS), gate(t), beta(t), z(t * kS), gamma(kS), state(size_t(kS) * kS) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0, 1);
        std::uniform_real_distribution<float> ud(0, 1);
        auto unit = [&](std::vector<float>& m) {   // unit-norm rows (the conv's l2 step)
            for (int64_t r = 0; r < t; ++r) {
                double n = 0;
                for (int c = 0; c < kS; ++c) { m[r * kS + c] = nd(rng); n += double(m[r * kS + c]) * m[r * kS + c]; }
                for (int c = 0; c < kS; ++c) m[r * kS + c] /= float(std::sqrt(n + 1e-6));
            }
        };
        unit(q); unit(k);
        for (auto& x : v) x = nd(rng) * vscale;
        for (auto& x : gate) x = -decay * ud(rng);
        for (auto& x : beta) x = no_update ? 0.0f : ud(rng);
        for (auto& x : z) x = nd(rng);
        for (auto& x : gamma) x = 0.5f + ud(rng);
        for (auto& x : state) x = 0.1f * vscale * nd(rng);
    }
};

// the sequential gated delta rule in double (tests/hip/gdn_chunked.cpp cpu_check, one head), y after the norm and the z gate
static void sequential(const Data& d, std::vector<double>& y, std::vector<double>& st) {
    st.assign(d.state.begin(), d.state.end());
    y.assign(size_t(d.T) * kS, 0.0);
    std::vector<double> delta(kS), out(kS);
    for (int64_t t = 0; t < d.T; ++t) {
        const double decay = std::exp(double(d.gate[t])), beta = d.beta[t];
        for (auto& x : st) x *= decay;
        for (int c = 0; c < kS; ++c) {
            double p = 0;
            for (int r = 0; r < kS; ++r) p += double(d.k[t * kS + r]) * st[size_t(r) * kS + c];
            delta[c] = beta * (d.v[t * kS + c] - p);
        }
        double norm = 0;
        for (int c = 0; c < kS; ++c) {
            double o = 0;
            for (int r = 0; r < kS; ++r) { st[size_t(r) * kS + c] += double(d.k[t * kS + r]) * delta[c]; o += double(d.q[t * kS + r]) * st[size_t(r) * kS + c]; }
            out[c] = o / std::sqrt(double(kS));
            norm += out[c] * out[c];
        }
        for (int c = 0; c < kS; ++c)
            y[t * kS + c] = out[c] / std::sqrt(norm / kS + 1e-6) * d.gamma[c] / (1.0 + std::exp(-double(d.z[t * kS + c])));
    }
}

static void run_case(const char* label, const Data& d, int chunk, bool fp16, double rms_bound, double* y_rms_out = nullptr) {
    std::vector<double> y0, s0;
    sequential(d, y0, s0);
    RefOpts o;
    o.chunk = chunk; o.fp16 = fp16;
    std::vector<float> st(d.state), y(size_t(d.T) * kS, 0.0f);
    ref_chunked_head(o, d.q.data(), d.k.data(), d.v.data(), d.gate.data(), d.beta.data(), d.z.data(), d.gamma.data(), st.data(), y.data(), d.T);
    Err ey, es;
    for (size_t i = 0; i < y0.size(); ++i) ey.add(y0[i], y[i]);
    for (size_t i = 0; i < s0.size(); ++i) es.add(s0[i], st[i]);
    const bool ok = ey.ok(rms_bound) && es.ok(rms_bound);
    std::printf("%s %-28s T=%-5lld chunk=%-2d fp16=%d y_rms=%.3e y_max=%.3e state_rms=%.3e\n", ok ? "PASS" : "FAIL", label, (long long) d.T, chunk, fp16,
                ey.rms(), ey.mx, es.rms());
    if (!ok) ++g_fail;
    if (y_rms_out) *y_rms_out = ey.rms();
}

static void test_algorithm() {
    // algebra: FP32 arithmetic, no operand rounding: the chunked form IS the recurrence at every chunk size (tail chunks included).
    // The bound is FP32 round-off (the output is also rounded once to FP16 as the sequential double is not: so no rounding at all here)
    for (int64_t T : {1, 17, 32, 33, 64, 100, 129})
        for (int chunk : {16, 32, 64}) {
            const Data d(T, 2.0f, 1.0f, false, 11u + uint32_t(T));
            run_case("algebra fp32", d, chunk, false, 2e-5);
        }
    // FP16 operands, the kernels' roundings: the GPU bounds (rms 3e-3), tails and a nonzero carried state
    double e16 = 0, e32 = 0;
    for (int64_t T : {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 79, 257}) {
        const Data d(T, 2.0f, 1.0f, false, 21u + uint32_t(T));
        run_case("fp16 chunk 32", d, 32, true, 3e-3);
        if (T == 257) { run_case("fp16 chunk 16", d, 16, true, 3e-3, &e16); run_case("fp16 chunk 32 (again)", d, 32, true, 3e-3, &e32); }
    }
    CHECK(e32 <= 3.0 * e16 + 1e-6);   // not worse than the shipped chunk by more than a factor of three (both are f16-operand class)
    for (float decay : {0.0f, 0.01f}) {   // weak decay: the state carries over many chunks (T-1000 weak-decay case of hip_gdn_chunked)
        const Data d(1031, decay, 16.0f, false, 77u);
        run_case("weak decay x16", d, 32, true, 3e-3);
    }
    const Data nb(65, 0.01f, 1.0f, true, 91u);   // beta 0: nothing enters the state
    run_case("beta zero", nb, 32, true, 3e-3);
}

int main() {
    test_routing();
    test_index_math();
    test_f16r();
    test_algorithm();
    std::printf("%s (%d failures)\n", g_fail ? "FAIL" : "PASS", g_fail);
    return g_fail ? 1 : 0;
}
