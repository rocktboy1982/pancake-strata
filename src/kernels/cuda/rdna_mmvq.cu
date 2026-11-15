// src/kernels/cuda/rdna_mmvq.cu - gfx1100 multi-column Q6_K x q8_1 decode GEMV (STRATA_RDNA_Q6K=1, opt-in).
// One wave per output row; per superblock every lane takes 8 consecutive weights (half n, quadrant qd, 8 l's), decodes
// them once (ql nibble | qh 2 bits, -32) and applies them to every column with two dp4a.  The same products and
// per-16 scales as ggml's vec_dot_q6_K_q8_1, another summation order (not bitwise the reference kernel).
#include "strata/kernels/native_mmvq.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)   // pancake: wave32 RDNA only (DPP / permlanex16 / sudot4)

namespace strata::kernels {
namespace {
struct Q81 { half2 ds; int8_t qs[32]; };
constexpr int kRdnaDefaultCols = 16;   // 9..16 columns: the verify windows of 12 rows (the engine's STRATA_VERIFY_MAX_T is at most 16)
constexpr int kRdnaMaxCols = 32;       // the widest instance built; above the default only through STRATA_RDNA_KQ_MAXCOLS (bench / parity)
// STRATA_RDNA_KQ_MAXCOLS=N caps the columns the RDNA kernels take (8 = the behaviour before 9..16 existed; an A/B knob).
inline int rdna_max_cols() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_RDNA_KQ_MAXCOLS");
        const int n = e != nullptr ? std::atoi(e) : kRdnaDefaultCols;
        return n >= 1 && n <= kRdnaMaxCols ? n : kRdnaDefaultCols;
    }();
    return v;
}
constexpr int QK = 256, BB = 210;   // Q6_K: ql[128] qh[64] scales[16] d(2)

__device__ __forceinline__ float wsum32(float v) {
#define STRATA_DPP6(v, c) __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), c, 0xf, 0xf, false))
    v += STRATA_DPP6(v, 0x161); v += STRATA_DPP6(v, 0x162); v += STRATA_DPP6(v, 0x164); v += STRATA_DPP6(v, 0x168);
    v += __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210, 0xfedcba98, false, false));
    return v;
}
__device__ __forceinline__ uint32_t ld32u(const uint8_t* p) {   // gfx11: unaligned dword loads are fine
    return *reinterpret_cast<const uint32_t*>(p);
}

template <int NC, int UNR = (NC <= 2 ? 5 : NC <= 4 ? 3 : NC <= 8 ? 2 : 1)>
__global__ void __launch_bounds__(256) rdna_q6k_kernel(const uint8_t* __restrict__ w, const Q81* __restrict__ x,
                                                       float* __restrict__ y, int n_in, int n_out) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= n_out) return;
    const int nsb = n_in / QK, xs = n_in / 32;
    const int n = lane >> 4, qd = (lane >> 2) & 3, l = (lane & 3) * 8;
    const int ql_off = n * 64 + (qd & 1) * 32 + l, qh_off = 128 + n * 32 + l, sc_off = 192 + n * 8 + (l >> 4) + 2 * qd;
    const int xb = n * 4 + qd, xi = l >> 2;     // the q8_1 block of these 8 weights inside the superblock, int index
    const int hs = 2 * qd, ns = (qd >= 2) ? 4 : 0;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    const uint8_t* wr = w + (size_t) row * nsb * BB;
#pragma unroll UNR
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* b = wr + (size_t) sb * BB;
        const uint32_t l0 = ld32u(b + ql_off), l1 = ld32u(b + ql_off + 4);
        const uint32_t h0 = ld32u(b + qh_off), h1 = ld32u(b + qh_off + 4);
        const int sc = (int) (int8_t) b[sc_off];
        const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
        // 8 six-bit weights as two packed int8x4, minus 32 (signed saturating like ggml's __vsubss4)
        const uint32_t lo0 = (l0 >> ns) & 0x0f0f0f0fu, lo1 = (l1 >> ns) & 0x0f0f0f0fu;
        const uint32_t hi0 = ((h0 >> hs) & 0x03030303u) << 4, hi1 = ((h1 >> hs) & 0x03030303u) << 4;
        // per-byte x - 32 for x in 0..63 without a borrow across bytes: set each byte's top bit, subtract, flip it back
        const int w0 = (int) ((((lo0 | hi0) | 0x80808080u) - 0x20202020u) ^ 0x80808080u);
        const int w1 = (int) ((((lo1 | hi1) | 0x80808080u) - 0x20202020u) ^ 0x80808080u);
        const float ds = d * (float) sc;
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const Q81* xb8 = x + (size_t) c * xs + sb * 8 + xb;
            const int* q = reinterpret_cast<const int*>(xb8->qs) + xi;
            int s = __builtin_amdgcn_sudot4(true, w0, true, q[0], 0, false);
            s = __builtin_amdgcn_sudot4(true, w1, true, q[1], s, false);
            acc[c] = fmaf(ds * __low2float(xb8->ds), (float) s, acc[c]);
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = wsum32(acc[c]);
        if (lane == 0) y[(size_t) c * n_out + row] = v;
    }
}

// ---- Q4_K / Q5_K / IQ4_XS: lane = (sub-block j = lane / 4, 8 weights l0 = 8 (lane % 4)); weights decoded once per
// superblock; K-quants take the min term from the lane's own activation sum (dot with ones) - same products,
// another summation order than ggml's vec_dot.
__device__ __forceinline__ int dot2(int w0, int w1, const int* q, int acc) {
    acc = __builtin_amdgcn_sudot4(true, w0, true, q[0], acc, false);
    return __builtin_amdgcn_sudot4(true, w1, true, q[1], acc, false);
}
__device__ __forceinline__ void kq_scale(const uint8_t* s, int j, int& sc, int& m) {
    if (j < 4) { sc = s[j] & 63; m = s[j + 4] & 63; }
    else { sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4); m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4); }
}
__device__ __forceinline__ uint32_t iq4_lut4(uint32_t idx) {   // 4 nibble indices (one per byte) -> 4 int8 table values
    const uint32_t t0 = 0xBFAD9881u, t1 = 0xF6EADDCFu, t2 = 0x26190D01u, t3 = 0x71594535u;   // -127..-10 | 1..113
    const uint32_t sel = idx & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(t1, t0, sel), hi = __builtin_amdgcn_perm(t3, t2, sel);
    const uint32_t msk = ((idx >> 3) & 0x01010101u) * 0xFFu;
    return (lo & ~msk) | (hi & msk);
}
template <int FMT> struct KFmt;   // 12 Q4_K, 13 Q5_K, 23 IQ4_XS
template <> struct KFmt<12> { static constexpr int B = 144; };
template <> struct KFmt<13> { static constexpr int B = 176; };
template <> struct KFmt<23> { static constexpr int B = 136; };

template <int FMT, int NC, int UNR = (NC <= 2 ? 5 : NC <= 4 ? 3 : NC <= 8 ? 2 : 1)>
__global__ void __launch_bounds__(256) rdna_kq_kernel(const uint8_t* __restrict__ w, const Q81* __restrict__ x,
                                                      float* __restrict__ y, int n_in, int n_out) {
    constexpr int BB_ = KFmt<FMT>::B;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= n_out) return;
    const int nsb = n_in / QK, xs = n_in / 32;
    const int j = lane >> 2, l0 = (lane & 3) * 8;
    float acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = 0.0f;
    const uint8_t* wr = w + (size_t) row * nsb * BB_;
#pragma unroll UNR
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* b = wr + (size_t) sb * BB_;
        int w0, w1;
        float f1, f2 = 0.0f;   // acc += d8 * (f1 * dot(w, q) - f2 * dot(1, q))
        if constexpr (FMT == 23) {
            const int hl = l0 >= 16 ? 4 : 0;
            const uint32_t q0 = ld32u(b + 8 + 16 * j + (l0 & 15)), q1 = ld32u(b + 8 + 16 * j + (l0 & 15) + 4);
            w0 = (int) iq4_lut4((q0 >> hl) & 0x0f0f0f0fu);
            w1 = (int) iq4_lut4((q1 >> hl) & 0x0f0f0f0fu);
            const uint16_t sh = *reinterpret_cast<const uint16_t*>(b + 2);
            const int ls = ((b[4 + j / 2] >> (4 * (j & 1))) & 0xf) | (((sh >> (2 * j)) & 3) << 4);
            f1 = __half2float(*reinterpret_cast<const __half*>(b)) * (float) (ls - 32);
        } else {
            const int qo = (FMT == 12 ? 16 : 48) + (j >> 1) * 32 + l0, ns = 4 * (j & 1);
            uint32_t v0 = (ld32u(b + qo) >> ns) & 0x0f0f0f0fu, v1 = (ld32u(b + qo + 4) >> ns) & 0x0f0f0f0fu;
            if constexpr (FMT == 13) {
                v0 |= ((ld32u(b + 16 + l0) >> j) & 0x01010101u) << 4;
                v1 |= ((ld32u(b + 16 + l0 + 4) >> j) & 0x01010101u) << 4;
            }
            w0 = (int) v0; w1 = (int) v1;
            // the 12 scale bytes as three dwords (the same for every lane), bytes picked arithmetically, no branches
            const uint32_t s0 = ld32u(b + 4), s1 = ld32u(b + 8), s2 = ld32u(b + 12);
            auto sb8 = [&](int k) -> int {
                const uint32_t w = k < 4 ? s0 : (k < 8 ? s1 : s2);
                return (int) ((w >> (8 * (k & 3))) & 0xffu);
            };
            int sc, m;
            if (j < 4) { sc = sb8(j) & 63; m = sb8(j + 4) & 63; }
            else { sc = (sb8(j + 4) & 0xF) | ((sb8(j - 4) >> 6) << 4); m = (sb8(j + 4) >> 4) | ((sb8(j) >> 6) << 4); }
            const float2 dm = __half22float2(*reinterpret_cast<const half2*>(b));
            f1 = dm.x * (float) sc; f2 = dm.y * (float) m;
        }
#pragma unroll
        for (int c = 0; c < NC; ++c) {
            const Q81* xb8 = x + (size_t) c * xs + sb * 8 + j;
            const int* q = reinterpret_cast<const int*>(xb8->qs) + (l0 >> 2);
            const float d8 = __low2float(xb8->ds);
            const int s1 = dot2(w0, w1, q, 0);
            if constexpr (FMT == 23) acc[c] = fmaf(d8 * f1, (float) s1, acc[c]);
            else {
                const int s2 = dot2(0x01010101, 0x01010101, q, 0);
                acc[c] = fmaf(d8, f1 * (float) s1 - f2 * (float) s2, acc[c]);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
        const float v = wsum32(acc[c]);
        if (lane == 0) y[(size_t) c * n_out + row] = v;
    }
}

// ---- Matrix-core few-column kernel (STRATA_RDNA_WMMA_MINCOLS=N: from N columns up, Q4_K / Q5_K, up to 32 columns).
// v_wmma_i32_16x16x16_iu8 (gfx11, wave32) with the roles swapped: A = the activations (row m = a column of the window, 16
// k int8 a lane: Q8_1 bytes), B = the weights (column n = a weight ROW, 16 k a lane: the quants as unsigned bytes), so
// D[m][n] holds column m of weight row n and a lane (n = lane % 16, hi = lane / 16) owns weight row n and the window
// columns 2 i + hi: its row's scales are lane-local and no cross-lane reduction ends the kernel.  A second WMMA against a
// ones fragment gives the activation sums of the min term exactly (integers).  Per 32-value sub-block j:
//   acc[col] += d8(col, j) * (d * sc_j * dot(w, x) - dmin * m_j * sum(x))     (the ggml Q4_K / Q5_K formula)
// One wave = 16 weight rows x up to 32 columns, no LDS, no barrier; each weight byte is read once for all columns.
// Fragments of gfx11: lanes 16..31 repeat lanes 0..15 for A and B; D lane l holds D[2 i + l / 16][l % 16].
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define STRATA_RDNA_WMMA_DEV 1
#else
#define STRATA_RDNA_WMMA_DEV 0
#endif
typedef int wm_i4 __attribute__((ext_vector_type(4)));
typedef int wm_i8 __attribute__((ext_vector_type(8)));

// KS: K slices a tile.  A block is 8 waves = 8 / KS tiles of 16 rows, each tile's superblocks dealt round-robin to its KS waves
// (more waves in flight hide the load latency of the serial K loop); the slices are summed through LDS at the end.
template <int FMT, int NT, int KS>
__global__ void __launch_bounds__(256) rdna_wmma_kq_kernel(const uint8_t* __restrict__ w, const Q81* __restrict__ x,
                                                           float* __restrict__ y, int n_in, int n_out, int ncols) {
#if STRATA_RDNA_WMMA_DEV
    constexpr int BB_ = KFmt<FMT>::B;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int n = lane & 15, hi = lane >> 4;
    const int tile = wave / KS, sl = wave % KS;
    const int row0 = (blockIdx.x * (8 / KS) + tile) * 16;
    const bool valid = row0 < n_out;   // (no early return: every wave reaches the barrier)
    const int row = min(row0 + n, n_out - 1);
    const int nsb = n_in / QK, xs = n_in / 32;
    const uint8_t* wr = w + (size_t) row * nsb * BB_;
    const Q81* xa[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) xa[nt] = x + (size_t) min(16 * nt + n, ncols - 1) * xs;
    float acc[NT][8];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[nt][i] = 0.0f;
    const wm_i4 ones = wm_i4{0x01010101, 0x01010101, 0x01010101, 0x01010101};
    const wm_i8 zero = wm_i8{0, 0, 0, 0, 0, 0, 0, 0};
    for (int sb = valid ? sl : nsb; sb < nsb; sb += KS) {
        const uint8_t* b = wr + (size_t) sb * BB_;
        const uint4* qsp = reinterpret_cast<const uint4*>(b + (FMT == 12 ? 16 : 48));
        uint4 qs[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) qs[i] = qsp[i];
        uint4 qh[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
        if constexpr (FMT == 13) {
            const uint4* qhp = reinterpret_cast<const uint4*>(b + 16);
            qh[0] = qhp[0];
            qh[1] = qhp[1];
        }
        const uint32_t s0 = ld32u(b + 4), s1 = ld32u(b + 8), s2 = ld32u(b + 12);
        auto sb8 = [&](int k) -> int {
            const uint32_t v = k < 4 ? s0 : (k < 8 ? s1 : s2);
            return (int) ((v >> (8 * (k & 3))) & 0xffu);
        };
        const float2 dm = __half22float2(*reinterpret_cast<const half2*>(b));
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int sc, m;
            if (j < 4) { sc = sb8(j) & 63; m = sb8(j + 4) & 63; }
            else { sc = (sb8(j + 4) & 0xF) | ((sb8(j - 4) >> 6) << 4); m = (sb8(j + 4) >> 4) | ((sb8(j) >> 6) << 4); }
            const float f1 = dm.x * (float) sc, f2 = dm.y * (float) m;
            const int sh = 4 * (j & 1);
            wm_i4 bw[2];   // the weights of this sub-block: two k16 steps of the lane's row, unsigned bytes
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint4 q = qs[2 * (j >> 1) + h];
                uint32_t v0 = (q.x >> sh) & 0x0f0f0f0fu, v1 = (q.y >> sh) & 0x0f0f0f0fu;
                uint32_t v2 = (q.z >> sh) & 0x0f0f0f0fu, v3 = (q.w >> sh) & 0x0f0f0f0fu;
                if constexpr (FMT == 13) {
                    v0 |= ((qh[h].x >> j) & 0x01010101u) << 4; v1 |= ((qh[h].y >> j) & 0x01010101u) << 4;
                    v2 |= ((qh[h].z >> j) & 0x01010101u) << 4; v3 |= ((qh[h].w >> j) & 0x01010101u) << 4;
                }
                bw[h] = wm_i4{(int) v0, (int) v1, (int) v2, (int) v3};
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const Q81* xb = xa[nt] + sb * 8 + j;
                const int* qx = reinterpret_cast<const int*>(xb->qs);
                wm_i4 a0 = wm_i4{qx[0], qx[1], qx[2], qx[3]}, a1 = wm_i4{qx[4], qx[5], qx[6], qx[7]};
                wm_i8 d1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, false, bw[0], zero, false);
                d1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, false, bw[1], d1, false);
                wm_i8 d2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, false, ones, zero, false);
                d2 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, false, ones, d2, false);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int col = 2 * i + hi + 16 * nt;
                    if (col >= ncols) continue;   // only the window's own columns: at 4 columns 2 of the 8 outputs a lane
                    const float d8 = __low2float(x[(size_t) col * xs + sb * 8 + j].ds);
                    acc[nt][i] = fmaf(d8, f1 * (float) d1[i] - f2 * (float) d2[i], acc[nt][i]);
                }
            }
        }
    }
    if constexpr (KS > 1) {
        __shared__ float red[8][NT][8][32];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 8; ++i) red[wave][nt][i][lane] = acc[nt][i];
        __syncthreads();
        if (sl == 0) {
#pragma unroll
            for (int k = 1; k < KS; ++k)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[nt][i] += red[wave + k][nt][i][lane];
        }
    }
    if (sl == 0 && valid && row0 + n < n_out) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int col = 2 * i + hi + 16 * nt;
                if (col < ncols) y[(size_t) col * n_out + row0 + n] = acc[nt][i];
            }
    }
#else
    __builtin_trap();
#endif
}
// K slices a tile: STRATA_RDNA_WMMA_KS = 1, 2 or 4 forces one; by default 4 below 400 tiles (6400 rows: few waves, latency
// bound: gdn out 2560 rows x K 6144 0.126 -> 0.046 ms) and 2 above (gdn qkv 0.086 -> 0.069 ms, the head 1.22 -> 1.15 ms).
inline int rdna_wmma_ks(int n_out) {
    static const int forced = [] {
        const char* e = std::getenv("STRATA_RDNA_WMMA_KS");
        const int n = e != nullptr ? std::atoi(e) : 0;
        return n == 1 || n == 2 || n == 4 ? n : 0;
    }();
    return forced != 0 ? forced : ((n_out + 15) / 16 < 400 ? 4 : 2);
}
inline int rdna_wmma_mincols() {
    static const int v = [] { const char* e = std::getenv("STRATA_RDNA_WMMA_MINCOLS"); return e != nullptr ? std::atoi(e) : 0; }();
    return v;
}
// The matrix-core kernel needs enough weight rows: below STRATA_RDNA_WMMA_MINROWS (default 2048) the wave-per-row kernels
// are faster (measured with K slices: 640 rows 0.026 ms against 0.015 at 9 columns; 2560 rows 0.048 against 0.058).
inline int rdna_wmma_minrows() {
    static const int v = [] { const char* e = std::getenv("STRATA_RDNA_WMMA_MINROWS"); return e != nullptr ? std::atoi(e) : 2048; }();
    return v;
}
template <int FMT>
void wmma_launch(const uint8_t* wb, const Q81* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const int ks = rdna_wmma_ks(n_out);
    const unsigned blocks = (unsigned) ((n_out + 16 * (8 / ks) - 1) / (16 * (8 / ks)));
    if (ncols <= 16) {
        if (ks == 1) rdna_wmma_kq_kernel<FMT, 1, 1><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, ncols);
        else if (ks == 2) rdna_wmma_kq_kernel<FMT, 1, 2><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, ncols);
        else rdna_wmma_kq_kernel<FMT, 1, 4><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, ncols);
    } else {
        if (ks == 1) rdna_wmma_kq_kernel<FMT, 2, 1><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, ncols);
        else if (ks == 2) rdna_wmma_kq_kernel<FMT, 2, 2><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, ncols);
        else rdna_wmma_kq_kernel<FMT, 2, 4><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, ncols);
    }
}

// Q6_K and IQ4_XS on the same matrix-core scheme: no min term, so one WMMA a k16 step (no ones fragment), weights as SIGNED
// int8 in B.  Q6_K has a scale per 16 values: each k16 step is its own sub-block (own D, own scale); IQ4_XS has one per 32.
// Both blocks are not 16-byte aligned (210 and 136 bytes): dword loads.
template <int NT, int KS>
__global__ void __launch_bounds__(256) rdna_wmma_q6k_kernel(const uint8_t* __restrict__ w, const Q81* __restrict__ x,
                                                            float* __restrict__ y, int n_in, int n_out, int ncols) {
#if STRATA_RDNA_WMMA_DEV
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int n = lane & 15, hi = lane >> 4;
    const int tile = wave / KS, sl = wave % KS;
    const int row0 = (blockIdx.x * (8 / KS) + tile) * 16;
    const bool valid = row0 < n_out;   // (no early return: every wave reaches the barrier)
    const int row = min(row0 + n, n_out - 1);
    const int nsb = n_in / QK, xs = n_in / 32;
    const uint8_t* wr = w + (size_t) row * nsb * BB;
    const Q81* xa[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) xa[nt] = x + (size_t) min(16 * nt + n, ncols - 1) * xs;
    float acc[NT][8];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[nt][i] = 0.0f;
    const wm_i8 zero = wm_i8{0, 0, 0, 0, 0, 0, 0, 0};
    for (int sb = valid ? sl : nsb; sb < nsb; sb += KS) {
        const uint8_t* b = wr + (size_t) sb * BB;
        const float d = __half2float(*reinterpret_cast<const __half*>(b + 208));
#pragma unroll
        for (int jb = 0; jb < 8; ++jb) {   // jb: the 32-value block = Q8_1 block sb * 8 + jb
            const int nn = jb >> 2, part = jb & 3;
            const int ql_off = nn * 64 + (part & 1) * 32, qh_off = 128 + nn * 32;
            const int lsh = 4 * (part >> 1), hsh = 2 * part;
            wm_i4 bw[2];
            float f[2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                int v[4];
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const uint32_t lo = (ld32u(b + ql_off + 16 * h + 4 * k) >> lsh) & 0x0f0f0f0fu;
                    const uint32_t hh = ((ld32u(b + qh_off + 16 * h + 4 * k) >> hsh) & 0x03030303u) << 4;
                    v[k] = (int) ((((lo | hh) | 0x80808080u) - 0x20202020u) ^ 0x80808080u);   // x - 32 per byte
                }
                bw[h] = wm_i4{v[0], v[1], v[2], v[3]};
                f[h] = d * (float) (int) (int8_t) b[192 + 2 * jb + h];
            }
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int* qx = reinterpret_cast<const int*>((xa[nt] + sb * 8 + jb)->qs);
                const wm_i4 a0 = wm_i4{qx[0], qx[1], qx[2], qx[3]}, a1 = wm_i4{qx[4], qx[5], qx[6], qx[7]};
                const wm_i8 d0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, bw[0], zero, false);
                const wm_i8 d1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, bw[1], zero, false);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int col = min(2 * i + hi + 16 * nt, ncols - 1);
                    const float d8 = __low2float(x[(size_t) col * xs + sb * 8 + jb].ds);
                    acc[nt][i] = fmaf(d8, f[0] * (float) d0[i] + f[1] * (float) d1[i], acc[nt][i]);
                }
            }
        }
    }
    if constexpr (KS > 1) {
        __shared__ float red[8][NT][8][32];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 8; ++i) red[wave][nt][i][lane] = acc[nt][i];
        __syncthreads();
        if (sl == 0) {
#pragma unroll
            for (int k = 1; k < KS; ++k)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[nt][i] += red[wave + k][nt][i][lane];
        }
    }
    if (sl == 0 && valid && row0 + n < n_out) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int col = 2 * i + hi + 16 * nt;
                if (col < ncols) y[(size_t) col * n_out + row0 + n] = acc[nt][i];
            }
    }
#else
    __builtin_trap();
#endif
}

template <int NT, int KS>
__global__ void __launch_bounds__(256) rdna_wmma_iq4xs_kernel(const uint8_t* __restrict__ w, const Q81* __restrict__ x,
                                                              float* __restrict__ y, int n_in, int n_out, int ncols) {
#if STRATA_RDNA_WMMA_DEV
    constexpr int BB_ = KFmt<23>::B;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int n = lane & 15, hi = lane >> 4;
    const int tile = wave / KS, sl = wave % KS;
    const int row0 = (blockIdx.x * (8 / KS) + tile) * 16;
    const bool valid = row0 < n_out;   // (no early return: every wave reaches the barrier)
    const int row = min(row0 + n, n_out - 1);
    const int nsb = n_in / QK, xs = n_in / 32;
    const uint8_t* wr = w + (size_t) row * nsb * BB_;
    const Q81* xa[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) xa[nt] = x + (size_t) min(16 * nt + n, ncols - 1) * xs;
    float acc[NT][8];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[nt][i] = 0.0f;
    const wm_i8 zero = wm_i8{0, 0, 0, 0, 0, 0, 0, 0};
    for (int sb = valid ? sl : nsb; sb < nsb; sb += KS) {
        const uint8_t* b = wr + (size_t) sb * BB_;
        const float dd = __half2float(*reinterpret_cast<const __half*>(b));
        const uint16_t sh = *reinterpret_cast<const uint16_t*>(b + 2);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int ls = ((b[4 + j / 2] >> (4 * (j & 1))) & 0xf) | (((sh >> (2 * j)) & 3) << 4);
            const float f = dd * (float) (ls - 32);
            uint32_t qd[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) qd[k] = ld32u(b + 8 + 16 * j + 4 * k);
            wm_i4 bw[2];   // k16 step 0 = the low nibbles of the 16 bytes, step 1 = the high nibbles
#pragma unroll
            for (int h = 0; h < 2; ++h)
                bw[h] = wm_i4{(int) iq4_lut4((qd[0] >> (4 * h)) & 0x0f0f0f0fu), (int) iq4_lut4((qd[1] >> (4 * h)) & 0x0f0f0f0fu),
                              (int) iq4_lut4((qd[2] >> (4 * h)) & 0x0f0f0f0fu), (int) iq4_lut4((qd[3] >> (4 * h)) & 0x0f0f0f0fu)};
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int* qx = reinterpret_cast<const int*>((xa[nt] + sb * 8 + j)->qs);
                const wm_i4 a0 = wm_i4{qx[0], qx[1], qx[2], qx[3]}, a1 = wm_i4{qx[4], qx[5], qx[6], qx[7]};
                wm_i8 d1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a0, true, bw[0], zero, false);
                d1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, bw[1], d1, false);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int col = min(2 * i + hi + 16 * nt, ncols - 1);
                    const float d8 = __low2float(x[(size_t) col * xs + sb * 8 + j].ds);
                    acc[nt][i] = fmaf(d8 * f, (float) d1[i], acc[nt][i]);
                }
            }
        }
    }
    if constexpr (KS > 1) {
        __shared__ float red[8][NT][8][32];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 8; ++i) red[wave][nt][i][lane] = acc[nt][i];
        __syncthreads();
        if (sl == 0) {
#pragma unroll
            for (int k = 1; k < KS; ++k)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[nt][i] += red[wave + k][nt][i][lane];
        }
    }
    if (sl == 0 && valid && row0 + n < n_out) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int col = 2 * i + hi + 16 * nt;
                if (col < ncols) y[(size_t) col * n_out + row0 + n] = acc[nt][i];
            }
    }
#else
    __builtin_trap();
#endif
}
inline void wmma_launch_q6k(const uint8_t* wb, const Q81* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const int ks = rdna_wmma_ks(n_out);
    const unsigned blocks = (unsigned) ((n_out + 16 * (8 / ks) - 1) / (16 * (8 / ks)));
    if (ncols <= 16) {
        if (ks == 1) rdna_wmma_q6k_kernel<1, 1><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, ncols);
        else if (ks == 2) rdna_wmma_q6k_kernel<1, 2><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, ncols);
        else rdna_wmma_q6k_kernel<1, 4><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, ncols);
    } else {
        if (ks == 1) rdna_wmma_q6k_kernel<2, 1><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, ncols);
        else if (ks == 2) rdna_wmma_q6k_kernel<2, 2><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, ncols);
        else rdna_wmma_q6k_kernel<2, 4><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, ncols);
    }
}
inline void wmma_launch_iq4xs(const uint8_t* wb, const Q81* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const int ks = rdna_wmma_ks(n_out);
    const unsigned blocks = (unsigned) ((n_out + 16 * (8 / ks) - 1) / (16 * (8 / ks)));
    if (ncols <= 16) {
        if (ks == 1) rdna_wmma_iq4xs_kernel<1, 1><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, ncols);
        else if (ks == 2) rdna_wmma_iq4xs_kernel<1, 2><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, ncols);
        else rdna_wmma_iq4xs_kernel<1, 4><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, ncols);
    } else {
        if (ks == 1) rdna_wmma_iq4xs_kernel<2, 1><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, ncols);
        else if (ks == 2) rdna_wmma_iq4xs_kernel<2, 2><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, ncols);
        else rdna_wmma_iq4xs_kernel<2, 4><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out, ncols);
    }
}
template <int FMT>
void kq_launch(const uint8_t* wb, const Q81* x, float* y, int n_in, int n_out, int ncols, cudaStream_t s) {
    const unsigned blocks = (unsigned) ((n_out + 7) / 8);
    switch (ncols) {
        case 1: rdna_kq_kernel<FMT, 1><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 2: rdna_kq_kernel<FMT, 2><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 3: rdna_kq_kernel<FMT, 3><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 4: rdna_kq_kernel<FMT, 4><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 5: rdna_kq_kernel<FMT, 5><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 6: rdna_kq_kernel<FMT, 6><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 7: rdna_kq_kernel<FMT, 7><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 8: rdna_kq_kernel<FMT, 8><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 9: rdna_kq_kernel<FMT, 9><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 10: rdna_kq_kernel<FMT, 10><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 11: rdna_kq_kernel<FMT, 11><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 12: rdna_kq_kernel<FMT, 12><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 13: rdna_kq_kernel<FMT, 13><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 14: rdna_kq_kernel<FMT, 14><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 15: rdna_kq_kernel<FMT, 15><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 16: rdna_kq_kernel<FMT, 16><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 17: rdna_kq_kernel<FMT, 17><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 18: rdna_kq_kernel<FMT, 18><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 19: rdna_kq_kernel<FMT, 19><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 20: rdna_kq_kernel<FMT, 20><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 21: rdna_kq_kernel<FMT, 21><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 22: rdna_kq_kernel<FMT, 22><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 23: rdna_kq_kernel<FMT, 23><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 24: rdna_kq_kernel<FMT, 24><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 25: rdna_kq_kernel<FMT, 25><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 26: rdna_kq_kernel<FMT, 26><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 27: rdna_kq_kernel<FMT, 27><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 28: rdna_kq_kernel<FMT, 28><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 29: rdna_kq_kernel<FMT, 29><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 30: rdna_kq_kernel<FMT, 30><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 31: rdna_kq_kernel<FMT, 31><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        default: rdna_kq_kernel<FMT, 32><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
    }
}
}  // namespace

bool rdna_kq_mmvq(int type, const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (n_in % QK != 0 || ncols < 1 || ncols > rdna_max_cols()) return false;
    const auto* wb = static_cast<const uint8_t*>(w);
    const auto* x = static_cast<const Q81*>(xq);
    cudaStream_t s = (cudaStream_t) stream;
    if (rdna_wmma_mincols() > 0 && ncols >= rdna_wmma_mincols() && ncols <= 32 && n_out >= rdna_wmma_minrows() && (type == 12 || type == 13 || (type == 23 && ncols >= 12))) {   // IQ4_XS breaks even at 12
        if (type == 12) wmma_launch<12>(wb, x, y, n_in, n_out, ncols, s);
        else if (type == 13) wmma_launch<13>(wb, x, y, n_in, n_out, ncols, s);
        else wmma_launch_iq4xs(wb, x, y, n_in, n_out, ncols, s);
        return true;
    }
    switch (type) {
        // Q4_K / Q5_K: measured slower than the native wave kernels on gfx1100 - not taken (kept for the bench)
        case 12: if (ncols < 4 && std::getenv("STRATA_RDNA_KQ_ALL") == nullptr) return false;   // faster than the wave kernel from 4 columns
            kq_launch<12>(wb, x, y, n_in, n_out, ncols, s); return true;
        case 13: if (ncols < 4 && std::getenv("STRATA_RDNA_KQ_ALL") == nullptr) return false;
            kq_launch<13>(wb, x, y, n_in, n_out, ncols, s); return true;
        case 23: kq_launch<23>(wb, x, y, n_in, n_out, ncols, s); return true;
        default: return false;
    }
}
namespace {
}  // namespace

bool rdna_q6k_enabled() {
    static const bool on = [] { const char* v = std::getenv("STRATA_RDNA_Q6K"); return v != nullptr && v[0] == '1'; }();
    return on;
}

bool rdna_q6k_mmvq(const void* w, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (n_in % QK != 0 || ncols < 1 || ncols > rdna_max_cols()) return false;
    const auto* wb = static_cast<const uint8_t*>(w);
    const auto* x = static_cast<const Q81*>(xq);
    const unsigned blocks = (unsigned) ((n_out + 7) / 8);
    cudaStream_t s = (cudaStream_t) stream;
    if (rdna_wmma_mincols() > 0 && ncols >= rdna_wmma_mincols() && ncols <= 32 && n_out >= rdna_wmma_minrows()) {
        wmma_launch_q6k(wb, x, y, n_in, n_out, ncols, s);
        return true;
    }
    switch (ncols) {
        case 1: rdna_q6k_kernel<1><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 2: rdna_q6k_kernel<2><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 3: rdna_q6k_kernel<3><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 4: rdna_q6k_kernel<4><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 5: rdna_q6k_kernel<5><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 6: rdna_q6k_kernel<6><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 7: rdna_q6k_kernel<7><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 8: rdna_q6k_kernel<8><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 9: rdna_q6k_kernel<9><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 10: rdna_q6k_kernel<10><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 11: rdna_q6k_kernel<11><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 12: rdna_q6k_kernel<12><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 13: rdna_q6k_kernel<13><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 14: rdna_q6k_kernel<14><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 15: rdna_q6k_kernel<15><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 16: rdna_q6k_kernel<16><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 17: rdna_q6k_kernel<17><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 18: rdna_q6k_kernel<18><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 19: rdna_q6k_kernel<19><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 20: rdna_q6k_kernel<20><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 21: rdna_q6k_kernel<21><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 22: rdna_q6k_kernel<22><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 23: rdna_q6k_kernel<23><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 24: rdna_q6k_kernel<24><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 25: rdna_q6k_kernel<25><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 26: rdna_q6k_kernel<26><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 27: rdna_q6k_kernel<27><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 28: rdna_q6k_kernel<28><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 29: rdna_q6k_kernel<29><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 30: rdna_q6k_kernel<30><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        case 31: rdna_q6k_kernel<31><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
        default: rdna_q6k_kernel<32><<<blocks, 256, 0, s>>>(wb, x, y, n_in, n_out); break;
    }
    return true;
}
}  // namespace strata::kernels
// CUDA and gfx906 builds compile rdna_stub.cpp instead of this file (CMakeLists.txt, MOD_PLAN 95b).
#endif
