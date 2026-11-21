// include/strata/prefill/gdn_wmma_ck.hpp - STRATA_GDN_WMMA_PREFILL=1 (LEADS.md Part S S5): the chunked gated-delta-net prefill with a
// LONGER chunk (32 tokens, two 16 x 16 x 16 WMMA token tiles) on the gfx11 matrix cores.  Pure C++ (no HIP): the constants, the
// index math and the routing rule that src/prefill/gdn_wmma_ck.inc (the kernels) uses, and the CPU reference of the chunked
// algorithm that tests/the host test src/prefill/gdn_wmma_ck_test.cpp holds against the sequential recurrence.
//
// WHY A SECOND SET OF KERNELS: STRATA_GDN_CHUNKED=1 (gdn_wmma_prep_kernel / gdn_wmma_scan_kernel, chunk 16) is the shipped gfx11
// scan (TEST_LEDGER.md: KEEP, p64 +4.2%).  Its chunk is one WMMA tile, so a 32K prompt walks 2,048 dependent steps per head, each
// with two barriers and a state -> B-operand conversion of 8 accumulators.  A chunk of 32 halves the steps, barriers and
// conversions; the products grow only where the chunk's triangular T and P are applied (2 x 2 tiles, lower half used).
// Chunk 64 (libr3d's size) does not fit: its LDS is over the 64 KB a workgroup may take (see kLdsBytes64).
//
// Not bit-exact with the chunk-16 kernels (FP16 operands, other chunk split): the same rounding class, held by the tolerances
// of tests/hip/gdn_chunked.cpp (rms <= 3e-3, max <= 2% of the largest value).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::prefill::gdn_ck {

// ---- shapes (the engine's: kernels.cu) ---------------------------------------------------------------------------------------
constexpr int kS = 128;              // head dim (keys and values)
constexpr int kHK = 16, kHV = 48;    // key heads, value heads
constexpr int kC = 10240;            // conv channels a token row: q (HK*S), k (HK*S), v (HV*S)

// ---- the chunk -----------------------------------------------------------------------------------------------------------------
constexpr int kCk = 32;              // tokens a chunk
constexpr int kNt = kCk / 16;        // 16-token tiles a chunk
constexpr int kNw = kS / 16;         // scan waves a block: one per 16 value columns
constexpr int kThreads = kNw * 32;   // 256
constexpr int kKP = kS + 8;          // LDS pitch of the FP16 k / q rows (halves)
constexpr int kKTP = kCk + 8;        // LDS pitch of the FP16 k^T rows (halves)
constexpr int kTPP = kCk + 8;        // LDS pitch of the FP16 T and P rows (halves)
constexpr int kTpWords = kCk * kCk;  // words (2 halves) of the T + P matrices of one (chunk, value head) in the global scratch

static_assert(kCk % 16 == 0 && kCk == 32, "the prep's gate scan and the T solve map one token to one lane of a wave32");
static_assert(kKP % 8 == 0 && kKTP % 8 == 0 && kTPP % 8 == 0, "a fragment is two 16-byte LDS reads: the row pitch is a multiple of 8 halves");
static_assert(kTpWords == 4 * kThreads, "four words a thread cover the chunk's T and P (2 * kCk * kCk halves)");

// LDS bytes of the scan kernel: sk and sq (FP16 rows), skt (FP16 k^T), stp (FP16 T and P), sv (FP16 v), so (FP32 o), sn (two FP32 buffers of
// the per-token sums), sg + sb, sgam.  A workgroup may take 64 KB on gfx11.
constexpr int kLdsBytes = 2 * kCk * kKP * 2 + kS * kKTP * 2 + 2 * kCk * kTPP * 2 + kCk * kS * 2 + kCk * kS * 4 + 2 * kCk * kNw * 4 + 2 * kCk * 4 + kS * 4;
static_assert(kLdsBytes <= 65536, "the scan's shared memory must fit one workgroup");
// What chunk 64 would need with the same buffers (documentation, and the reason the chunk is 32).
constexpr int kLdsBytes64 = 2 * 64 * kKP * 2 + kS * (64 + 8) * 2 + 2 * 64 * (64 + 8) * 2 + 64 * kS * 2 + 64 * kS * 4 + 2 * 64 * kNw * 4 + 2 * 64 * 4 + kS * 4;
static_assert(kLdsBytes64 > 65536, "chunk 64 does not fit a workgroup");

// ---- LDS offsets (halves) of the 16-half fragment reads: the kernel and the host test use the same functions ----------------------
constexpr int off_k(int u, int c, int j) { return (16 * u + c) * kKP + 16 * j; }     // sk / sq: token row 16u + c, k columns 16j..16j+15
constexpr int off_kt(int j, int c, int u) { return (16 * j + c) * kKTP + 16 * u; }   // skt: k row 16j + c, tokens 16u..16u+15
constexpr int off_tp(int u, int c, int u2) { return (16 * u + c) * kTPP + 16 * u2; } // T or P: token row 16u + c, tokens 16u2..
// Thread tid's i-th word (i = 0..3) of the chunk's T (i = 0, 1) and P (i = 2, 3), global word index tid + 256 i (the (chunk, head)
// block is kTpWords words: T then P, each kCk x kCk halves row-major) -> the half index in the LDS stp (T at 0, P at kCk * kTPP).
constexpr int tp_word(int tid, int i) { return tid + kThreads * i; }
constexpr int tp_lds_half(int tid, int i) {
    return (i >> 1) * (kCk * kTPP) + (((tid + kThreads * (i & 1)) >> 4) * kTPP) + 2 * ((tid + kThreads * (i & 1)) & 15);
}

// The same half index as tp_lds_half(tid, i) split into a per-thread base and a compile-time constant (the kernel adds them: constant offsets
// fold into the LDS instruction; the host test checks the identity).
constexpr int tp_base(int tid) { return (tid >> 4) * kTPP + 2 * (tid & 15); }
constexpr int tp_off(int i) { return (i >> 1) * (kCk * kTPP) + (i & 1) * 16 * kTPP; }

// ---- routing: a pure rule, so the host test can show that the default path never reaches the new kernels ----------------------
// env_ck: STRATA_GDN_WMMA_PREFILL (-1 unset or empty, else its number); chunked_env: STRATA_GDN_CHUNKED (-1 unset).  Only gfx11 (wave32
// WMMA with the replicated-row fragment layout) takes the kernels; gfx12 keeps its own.  An explicit STRATA_GDN_CHUNKED=0 wins.
inline bool route(int env_ck, int chunked_env, bool gfx11, int64_t T, int64_t ld16) {
    return env_ck >= 1 && chunked_env != 0 && gfx11 && T >= 64 && ld16 % 8 == 0;
}
inline int parse_env(const char* e) { return e == nullptr || e[0] == '\0' ? -1 : std::atoi(e); }

// ---- binary16 rounding on the host (round to nearest even), the operand rounding of the kernels ------------------------------------
inline float f16r(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    const uint32_t sign = u & 0x80000000u;
    const uint32_t a = u & 0x7fffffffu;
    if (a >= 0x7f800000u) return x;                       // inf, nan
    if (a >= 0x477ff000u) return sign ? -INFINITY : INFINITY;   // >= 65520 rounds to inf
    float r;
    if (a < 0x38800000u) {                                // subnormal half range: quantum 2^-24
        const float q = std::ldexp(1.0f, -24);
        const float m = std::nearbyint(std::fabs(x) / q);   // default rounding mode: nearest even
        r = m * q;
    } else {
        // keep 10 mantissa bits: round the float's 23-bit mantissa at bit 13, nearest even
        const uint32_t lsb = (a >> 13) & 1u;
        uint32_t b = a + 0xfffu + lsb;
        b &= ~0x1fffu;
        std::memcpy(&r, &b, 4);
    }
    return sign ? -r : r;
}

// ---- the CPU reference of the chunked algorithm for ONE value head, mirroring the kernels' operand roundings and tile order -------
// q, k: T rows of kS floats (the head's own key head), v: T rows of kS, gate / beta: T floats, z: T rows of kS, gamma: kS.
// state: kS x kS, state[r * kS + col] (r the key row), read and written.  y: T rows of kS, the FP16 value (as float) the kernel stores.
// fp16 false: every f16r is skipped (the algebra check: the chunked form is the recurrence).
struct RefOpts { int chunk = kCk; bool fp16 = true; float eps = 1e-6f; };

inline void ref_chunked_head(const RefOpts& o, const float* q, const float* k, const float* v, const float* gate, const float* beta,
                             const float* z, const float* gamma, float* state, float* y, int64_t T) {
    const int CK = o.chunk, NT = CK / 16;
    auto r16 = [&](float x) { return o.fp16 ? f16r(x) : x; };
    std::vector<float> sg(CK), sb(CK), kq(size_t(CK) * kS), qq(size_t(CK) * kS), kk(size_t(CK) * CK), qk(size_t(CK) * CK),
        tm(size_t(CK) * CK), pm(size_t(CK) * CK), yy(size_t(CK) * kS), dd(size_t(CK) * kS), oo(size_t(CK) * kS), x(size_t(CK) * kS),
        qs(size_t(CK) * kS), sa(size_t(CK) * CK), s16(size_t(kS) * kS), d16(size_t(CK) * kS);
    const float out_scale = 1.0f / std::sqrt(float(kS));
    for (int64_t t0 = 0; t0 < T; t0 += CK) {
        // ---- the chunk's gate sums, beta (past the end: gate 0, beta 0, rows zero for the gram: the kernel reads the last row, finite,
        //      and gives it beta 0 and gate 0; a zero row here is the same for every product it takes part in)
        float run = 0.0f;
        for (int t = 0; t < CK; ++t) {
            const bool ok = t0 + t < T;
            run += ok ? gate[t0 + t] : 0.0f;
            sg[t] = run;
            sb[t] = ok ? beta[t0 + t] : 0.0f;
            for (int c = 0; c < kS; ++c) {
                kq[size_t(t) * kS + c] = r16(ok ? k[(t0 + t) * kS + c] : 0.0f);
                qq[size_t(t) * kS + c] = r16(ok ? q[(t0 + t) * kS + c] : 0.0f);
            }
        }
        // ---- prep: gram and T, P (tile by tile, FP32 accumulation over the eight 16-wide k slices, as the WMMA chain)
        for (int u = 0; u < NT; ++u)
            for (int u2 = 0; u2 <= u; ++u2)
                for (int a = 0; a < 16; ++a)
                    for (int b = 0; b < 16; ++b) {
                        float g1 = 0.0f, g2 = 0.0f;
                        for (int j = 0; j < kS / 16; ++j) {
                            float p1 = 0.0f, p2 = 0.0f;
                            for (int e = 0; e < 16; ++e) {
                                p1 += kq[size_t(16 * u + a) * kS + 16 * j + e] * kq[size_t(16 * u2 + b) * kS + 16 * j + e];
                                p2 += qq[size_t(16 * u + a) * kS + 16 * j + e] * kq[size_t(16 * u2 + b) * kS + 16 * j + e];
                            }
                            g1 += p1; g2 += p2;
                        }
                        kk[size_t(16 * u + a) * CK + 16 * u2 + b] = g1;
                        qk[size_t(16 * u + a) * CK + 16 * u2 + b] = g2;
                    }
        std::fill(sa.begin(), sa.end(), 0.0f);
        std::fill(pm.begin(), pm.end(), 0.0f);
        for (int t = 0; t < CK; ++t)
            for (int i = 0; i < CK; ++i) {
                if (i > t) continue;   // tiles above the diagonal are zero (the kernel writes zeros there)
                const float dec = std::exp(sg[t] - sg[i]);
                if (i < t) sa[size_t(t) * CK + i] = sb[t] * dec * kk[size_t(t) * CK + i];
                pm[size_t(t) * CK + i] = r16(dec * qk[size_t(t) * CK + i]);
            }
        // T = (I + sa)^-1: lane l = column l, forward substitution down the rows in FP32 (tc[t] = delta - sum_{k<t} sa[t][k] tc[k])
        for (int col = 0; col < CK; ++col) {
            std::vector<float> tc(CK);
            for (int t = 0; t < CK; ++t) {
                float val = t == col ? 1.0f : 0.0f;
                for (int kx = 0; kx < t; ++kx) val = std::fma(-sa[size_t(t) * CK + kx], tc[kx], val);
                tc[t] = val;
                tm[size_t(t) * CK + col] = r16(val);
            }
        }
        // ---- scan: the state as an operand is rounded to FP16 once per chunk (gw_acc_to_b of each accumulator)
        for (size_t i = 0; i < s16.size(); ++i) s16[i] = r16(state[i]);
        for (int t = 0; t < CK; ++t)
            for (int col = 0; col < kS; ++col) {
                float a1 = 0.0f, a2 = 0.0f;
                for (int j = 0; j < kS / 16; ++j) {
                    float p1 = 0.0f, p2 = 0.0f;
                    for (int e = 0; e < 16; ++e) {
                        p1 += kq[size_t(t) * kS + 16 * j + e] * s16[size_t(16 * j + e) * kS + col];
                        p2 += qq[size_t(t) * kS + 16 * j + e] * s16[size_t(16 * j + e) * kS + col];
                    }
                    a1 += p1; a2 += p2;
                }
                x[size_t(t) * kS + col] = a1;
                qs[size_t(t) * kS + col] = a2;
                const float vv = r16(t0 + t < T ? v[(t0 + t) * kS + col] : v[(T - 1) * kS + col]);   // v is staged as FP16 (sv)
                yy[size_t(t) * kS + col] = r16(sb[t] * (vv - std::exp(sg[t]) * a1));
            }
        // d = T y over the token tiles (lower half only), o = exp(sg) qs + P d
        for (int t = 0; t < CK; ++t)
            for (int col = 0; col < kS; ++col) {
                float acc = 0.0f;
                for (int u2 = 0; u2 <= t / 16; ++u2) {
                    float p = 0.0f;
                    for (int e = 0; e < 16; ++e) p += tm[size_t(t) * CK + 16 * u2 + e] * yy[size_t(16 * u2 + e) * kS + col];
                    acc += p;
                }
                dd[size_t(t) * kS + col] = acc;
                d16[size_t(t) * kS + col] = r16(acc);
            }
        for (int t = 0; t < CK; ++t)
            for (int col = 0; col < kS; ++col) {
                float acc = std::exp(sg[t]) * qs[size_t(t) * kS + col];
                for (int u2 = 0; u2 <= t / 16; ++u2) {
                    float p = 0.0f;
                    for (int e = 0; e < 16; ++e) p += pm[size_t(t) * CK + 16 * u2 + e] * d16[size_t(16 * u2 + e) * kS + col];
                    acc += p;
                }
                oo[size_t(t) * kS + col] = acc;
            }
        // ---- state update: S = exp(gl) S + K^T (exp(gl - sg) d)
        const float gl = sg[CK - 1], eg = std::exp(gl);
        std::vector<float> w16(size_t(CK) * kS);
        for (int t = 0; t < CK; ++t)
            for (int col = 0; col < kS; ++col) w16[size_t(t) * kS + col] = r16(std::exp(gl - sg[t]) * dd[size_t(t) * kS + col]);
        for (int r = 0; r < kS; ++r)
            for (int col = 0; col < kS; ++col) {
                float acc = state[size_t(r) * kS + col] * eg;
                for (int u = 0; u < NT; ++u) {
                    float p = 0.0f;
                    for (int e = 0; e < 16; ++e) p += kq[size_t(16 * u + e) * kS + r] * w16[size_t(16 * u + e) * kS + col];
                    acc += p;
                }
                state[size_t(r) * kS + col] = acc;
            }
        // ---- the gated RMS norm and the z gate, the FP16 store
        for (int t = 0; t < CK && t0 + t < T; ++t) {
            float ss = 0.0f;
            for (int col = 0; col < kS; ++col) {
                const float ov = oo[size_t(t) * kS + col] * out_scale;
                oo[size_t(t) * kS + col] = ov;
                ss += ov * ov;
            }
            const float rn = 1.0f / std::sqrt(ss * (1.0f / kS) + o.eps);
            for (int col = 0; col < kS; ++col) {
                const float val = oo[size_t(t) * kS + col] * rn * gamma[col];
                const float zz = z[(t0 + t) * kS + col];
                y[(t0 + t) * kS + col] = r16(val * (1.0f / (1.0f + std::exp(-zz))));
            }
        }
    }
}

}  // namespace strata::prefill::gdn_ck
