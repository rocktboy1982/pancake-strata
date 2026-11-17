// include/strata/kernels/int4_rot.hpp - host reference of the rotated-INT4 pipeline of SILVERTHRONE32/llama.cpp-rdna3-int4 (MIT,
// Sachin Mahesh; LEADS.md Part L, README.md section 9).  Written here from the authors' report (docs/rdna3/papers/rdna3_int4_report.tex
// sections 3.1-3.3) and their sign hash; no code of the fork is copied.  Header only, host only, no GPU, no engine state.
//
// The pipeline for one linear layer y = W x:
//   weights  (offline): W' = W R^T per block of B values of K, GPTQ in that basis -> int4 codes in [-8, 7], one scale per 32 values;
//   acts     (run time): x' = R x per block, then
//     A8  q in [-128, 127], scale d_b = amax_b / 127 per block, split q = 16 h + l with h = q >> 4 in [-8, 7] SIGNED and l = q & 15 in
//         [0, 15] UNSIGNED: two int4 passes, acc = 16 sum(w h) + sum(w l), exact in int32;
//     A4  one pass, q in [-7, 7]: the block scale is m_b / 16 of a per-token reference amax_token, m_b = ceil(16 c amax_b / amax_token)
//         in 1..16, clip c = 0.85 (our reading of the report's "multiplier m_b of 1/16 of a per-token reference");
//   product  y = sum over 32-groups of s_w[g] d_b acc_int[g].
// R = H_B D / sqrt(B): H the Sylvester-Hadamard matrix, D = diag(s_j) with s_j = -1 when ((j * 2654435761) mod 2^32) >> 31, else +1,
// so a converter and a backend agree without storing signs; R is orthogonal, so W' x' = W x.  The authors use B = 256 (K a multiple of
// 256); any power of two works here, and the daily Flash-Next expert down projection has K = 640 = 5 x 128, so the probe runs B = 128.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace strata::kernels::int4rot {

inline float sign_of(int j) { return (((uint32_t) j * 2654435761u) >> 31) ? -1.0f : 1.0f; }

// y = R x on one block of B floats, in place: signs, the in-place butterflies of the Sylvester matrix (the order a GPU block does them in),
// then the 1/sqrt(B) scale
inline void rotate_block(float* x, int B) {
    for (int j = 0; j < B; ++j) x[j] *= sign_of(j);
    for (int h = 1; h < B; h <<= 1)
        for (int i = 0; i < B; i += 2 * h)
            for (int j = i; j < i + h; ++j) {
                const float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
    const float s = 1.0f / std::sqrt((float) B);
    for (int j = 0; j < B; ++j) x[j] *= s;
}

// every B-block of a [rows, K] matrix
inline void rotate_rows(std::vector<float>& m, int rows, int K, int B) {
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < K / B; ++b) rotate_block(&m[(size_t) r * K + (size_t) b * B], B);
}

struct ActQ {            // quantised activations of [N, K]
    int N = 0, K = 0, B = 256;
    bool a8 = true;
    std::vector<int8_t> hi;      // A8: signed high nibble value h in [-8, 7]; A4: the signed 4-bit code in [-7, 7]
    std::vector<int8_t> lo;      // A8: unsigned low nibble l in [0, 15]; A4: empty
    std::vector<float> scale;    // [N][K / B] the block scale: A8 amax_b / 127, A4 m_b * amax_token / (16 * 7)
};

// x is already rotated; A8: clip is ignored (the authors clip only the 4-bit path)
inline ActQ quantize_acts(const std::vector<float>& x, int N, int K, int B, bool a8, float clip = 0.85f) {
    ActQ q;
    q.N = N; q.K = K; q.B = B; q.a8 = a8;
    q.hi.assign((size_t) N * K, 0);
    if (a8) q.lo.assign((size_t) N * K, 0);
    q.scale.assign((size_t) N * (K / B), 0.f);
    for (int n = 0; n < N; ++n) {
        const float* row = &x[(size_t) n * K];
        float amax_tok = 0.f;
        for (int k = 0; k < K; ++k) amax_tok = std::max(amax_tok, std::fabs(row[k]));
        for (int b = 0; b < K / B; ++b) {
            float amax = 0.f;
            for (int k = 0; k < B; ++k) amax = std::max(amax, std::fabs(row[b * B + k]));
            float sc;
            if (a8) sc = amax / 127.0f;
            else {
                int m = amax_tok > 0.f ? (int) std::ceil(16.0f * clip * amax / amax_tok) : 1;
                m = std::min(16, std::max(1, m));
                sc = (float) m * amax_tok / (16.0f * 7.0f);
            }
            q.scale[(size_t) n * (K / B) + b] = sc;
            const float inv = sc > 0.f ? 1.0f / sc : 0.f;
            for (int k = 0; k < B; ++k) {
                const size_t i = (size_t) n * K + (size_t) b * B + k;
                if (a8) {
                    const int v = std::min(127, std::max(-128, (int) std::lrintf(row[b * B + k] * inv)));
                    q.hi[i] = (int8_t) (v >> 4);
                    q.lo[i] = (int8_t) (v & 15);
                } else {
                    q.hi[i] = (int8_t) std::min(7, std::max(-7, (int) std::lrintf(row[b * B + k] * inv)));
                }
            }
        }
    }
    return q;
}

struct WeightQ {         // int4 weights [M, K] with one scale per 32 values
    int M = 0, K = 0;
    std::vector<int8_t> q;       // [-8, 7]
    std::vector<float> scale;    // [M][K / 32]
};

// round-to-nearest into codes [-8, 7] with scale = max|w| / 7 per group of 32 (the starting point; rotate_gptq.py does the GPTQ search)
inline WeightQ quantize_weights_rtn(const std::vector<float>& w, int M, int K) {
    WeightQ r;
    r.M = M; r.K = K;
    r.q.assign((size_t) M * K, 0);
    r.scale.assign((size_t) M * (K / 32), 0.f);
    for (int m = 0; m < M; ++m)
        for (int g = 0; g < K / 32; ++g) {
            float amax = 0.f;
            for (int k = 0; k < 32; ++k) amax = std::max(amax, std::fabs(w[(size_t) m * K + g * 32 + k]));
            const float sc = amax / 7.0f;
            r.scale[(size_t) m * (K / 32) + g] = sc;
            const float inv = sc > 0.f ? 1.0f / sc : 0.f;
            for (int k = 0; k < 32; ++k)
                r.q[(size_t) m * K + g * 32 + k] =
                    (int8_t) std::min(7, std::max(-8, (int) std::lrintf(w[(size_t) m * K + g * 32 + k] * inv)));
        }
    return r;
}

// the integer accumulator of one 32-group for output (m, n): what the WMMA passes produce, exact in int32
inline int32_t group_acc(const WeightQ& w, const ActQ& a, int m, int n, int g) {
    int32_t hi = 0, lo = 0;
    for (int k = 0; k < 32; ++k) {
        const int kk = g * 32 + k;
        const int wq = w.q[(size_t) m * w.K + kk];
        hi += wq * a.hi[(size_t) n * a.K + kk];
        if (a.a8) lo += wq * a.lo[(size_t) n * a.K + kk];
    }
    return a.a8 ? 16 * hi + lo : hi;
}

// y[n][m] = sum_g s_w[m][g] * d[n][block(g)] * acc(m, n, g)
inline std::vector<float> gemm_ref(const WeightQ& w, const ActQ& a) {
    std::vector<float> y((size_t) a.N * w.M, 0.f);
    for (int n = 0; n < a.N; ++n)
        for (int m = 0; m < w.M; ++m) {
            float s = 0.f;
            for (int g = 0; g < w.K / 32; ++g)
                s += w.scale[(size_t) m * (w.K / 32) + g] * a.scale[(size_t) n * (a.K / a.B) + (g * 32) / a.B] * (float) group_acc(w, a, m, n, g);
            y[(size_t) n * w.M + m] = s;
        }
    return y;
}

}  // namespace strata::kernels::int4rot
