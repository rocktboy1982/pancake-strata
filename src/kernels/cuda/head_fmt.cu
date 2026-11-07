// src/kernels/cuda/head_fmt.cu - see include/strata/kernels/head_fmt.hpp.  pancake-strata, after basalt's q5k_head.cu
// (jesdga95/basalt a9468e5c, MIT).  Each format is a traits struct (its native_mmvq.cu traits, the arithmetic
// unchanged, plus a 32-weight group's dequantization for the 2-bit copy); the kernels are templates instantiated per
// format at compile time.
#include "strata/kernels/head_fmt.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/klog.hpp"
#include "strata/kernels/pdl.hpp"
#include "head_proxy.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace strata::kernels {
namespace {

constexpr int WARP = 32;
constexpr int VT = 4;          // virtual threads per lane: native_mmvq's four warps (WARPS) of one row
constexpr int WPB = 8;         // warps, and rows, per block of the GEMV
constexpr int kMaxIn = 8192;   // the columns' activations a block keeps in shared memory: 8 x 8192 / 32 x 36 B

struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81Block) == 36);

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o, WARP);
    return x;
}

// ---------------------------------------------------------------------------------------------------------- IQ4_XS
struct IQ4XSBlock {
    half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
static_assert(sizeof(IQ4XSBlock) == 136);

__device__ __align__(4) int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                              1,    13,   25,  38,  53,  69,  89,  113};

// native_mmvq.cu's iq4_table_lookup (CUDA form)
__device__ __forceinline__ int2 iq4_lookup(int q4) {
    const uint32_t* table32 = reinterpret_cast<const uint32_t*>(kIq4nl);
    uint32_t tmp[2];
    const uint32_t sel = 0x32103210 | ((q4 & 0x88888888) >> 1);
#pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        const uint32_t low = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, sel >> shift);
    }
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), __byte_perm(tmp[0], tmp[1], 0x7531));
}

struct IQ4XS {   // native_mmvq.cu's IQ4XSTraits
    using Block = IQ4XSBlock;
    static constexpr int T = 8, BPI = 4 * VT * WARP / 32, ROW_BYTES_2560 = 10 * 136;
    __device__ static int kqs(int tid) { return 4 * (tid % 8); }
    struct W { int2 v[4]; int ls; float dw; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iq4_lookup(__ldg(reinterpret_cast<const int*>(w->qs) + iqs + j));
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = __half2float(w->d);
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j];
            const int u1 = reinterpret_cast<const int*>(x[iqs / 4].qs)[j + 4];
            sumi = STRATA_DP4A(r.v[j].x, u0, sumi);
            sumi = STRATA_DP4A(r.v[j].y, u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = r.dw * __low2float(x[iqs / 4].ds);
        return d * sumi;
    }
    // group j (0..7) of a block: the codebook times the block's 6-bit sub-scale
    __device__ static void group(const Block* b, int j, float* w) {
        const int ls = ((b->scales_l[j / 2] >> (4 * (j % 2))) & 0x0f) | (((b->scales_h >> (2 * j)) & 0x03) << 4);
        const float d = __half2float(b->d) * (float) (ls - 32);
        for (int l = 0; l < 16; ++l) {
            const uint8_t q = b->qs[16 * j + l];
            w[l] = d * (float) kIq4nl[q & 0x0f];
            w[l + 16] = d * (float) kIq4nl[q >> 4];
        }
    }
};

// ------------------------------------------------------------------------------------------------------------ Q6_K
struct Q6KBlock {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
static_assert(sizeof(Q6KBlock) == 210);

__device__ __forceinline__ int load_int_b2(const void* ptr, int i32) {   // native_mmvq.cu's
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}
__device__ __forceinline__ float q6_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const int8_t* __restrict__ scales, float d,
                                              const float* __restrict__ d8) {   // native_mmvq.cu's
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        sumf += d8[i] * (STRATA_DP4A(vi, u[i], 0) * sc);
    }
    return d * sumf;
}

struct Q6K {   // native_mmvq.cu's Q6KTraits
    using Block = Q6KBlock;
    static constexpr int T = 32, BPI = VT * WARP / 32, ROW_BYTES_2560 = 10 * 210;
    __device__ static int kqs(int tid) { return tid % 32; }
    struct W { int vl, vh; float d; const int8_t* scales; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        r.vl = load_int_b2(w->ql, iqs);
        r.vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        r.scales = w->scales + scale_offset;
        r.d = w->d;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* x, int iqs) {
        int u[2];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[i] = reinterpret_cast<const int*>(x[r.bq8_offset + 2 * i].qs)[iqs % 8];
            d8[i] = __low2float(x[r.bq8_offset + 2 * i].ds);
        }
        return q6_q8_dot_impl(r.vl, r.vh, u, r.scales, r.d, d8);
    }
    // group g (0..7) of a block: ggml's dequantize_row_q6_K, the 32 values of that group
    __device__ static void group(const Block* b, int g, float* w) {
        const int half_ = g / 4, qq = g % 4;
        const uint8_t* ql = b->ql + 64 * half_;
        const uint8_t* qh = b->qh + 32 * half_;
        const int8_t* sc = b->scales + 8 * half_;
        const float d = __half2float(b->d);
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            int q;
            if (qq == 0) q = (int) ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            else if (qq == 1) q = (int) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            else if (qq == 2) q = (int) ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            else q = (int) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            w[l] = d * (float) sc[is + 2 * qq] * (float) q;
        }
    }
};

// ------------------------------------------------------------------------------------------------------------ Q4_K
struct Q4KBlock {
    half2 dm;
    uint8_t scales[12];
    uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144);

__device__ __forceinline__ float q4_q8_dot_impl(const int* __restrict__ v, const int* __restrict__ u,
                                              const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m,
                                              const half2& dm4, const float* __restrict__ d8) {   // native_mmvq.cu's
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = STRATA_DP4A(v1i, u[2 * i + 1], STRATA_DP4A(v0i, u[2 * i], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i + 1], STRATA_DP4A(0x01010101, u[2 * i], 0));
        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

struct Q4K {   // native_mmvq.cu's Q4KTraits (QK 256, QI 32, VDR 2)
    using Block = Q4KBlock;
    static constexpr int T = 16, BPI = 2 * VT * WARP / 32, ROW_BYTES_2560 = 10 * 144;
    __device__ static int kqs(int tid) { return 2 * (tid % 16); }
    struct W { int v[2]; uint16_t aux[2]; half2 dm; int bq8_offset; };
    __device__ static W load(const Block* __restrict__ bq4, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = ql[0];
        r.v[1] = ql[4];
        const uint16_t* scales = reinterpret_cast<const uint16_t*>(bq4->scales);
        const int j = r.bq8_offset / 2;
        const int jm = j & 1;
        const uint32_t s0 = scales[jm];
        const uint32_t s2 = scales[jm + 2];
        const uint32_t s4 = scales[jm + 4];
        const uint32_t hi = uint32_t(-int32_t(j >= 2));
        r.aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
        r.aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
        r.dm = bq4->dm;
        return r;
    }
    __device__ static float apply(const W& r, const Q81Block* bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = __low2float(bq8i->ds);
            const int* q8 = reinterpret_cast<const int*>(bq8i->qs) + ((iqs / 2) % 4);
            u[2 * i] = q8[0];
            u[2 * i + 1] = q8[4];
        }
        const uint8_t* sc = reinterpret_cast<const uint8_t*>(r.aux);
        return q4_q8_dot_impl(r.v, u, sc, sc + 2, r.dm, d8);
    }
    // group g (0..7) of a block: ggml's dequantize_row_q4_K, the 32 values of that group
    __device__ static void group(const Block* b, int g, float* w) {
        int sc, m;
        if (g < 4) { sc = b->scales[g] & 63; m = b->scales[g + 4] & 63; }
        else {
            sc = (b->scales[g + 4] & 0xF) | ((b->scales[g - 4] >> 6) << 4);
            m = (b->scales[g + 4] >> 4) | ((b->scales[g] >> 6) << 4);
        }
        const float2 dm = __half22float2(b->dm);
        const float d = dm.x * (float) sc, mn = dm.y * (float) m;
        const uint8_t* q = b->qs + 32 * (g / 2);
        for (int l = 0; l < 32; ++l) w[l] = d * (float) ((g & 1) ? (q[l] >> 4) : (q[l] & 0xF)) - mn;
    }
};

// ------------------------------------------------------------------------------------------- the kernels, per format
// One row by one warp, NC columns of activations at sx (qb q8_1 blocks apart): native_mmvq's per-thread blocks (thread
// tid takes kbx = tid / T, += BPI, at kqs(tid)) and its cross-warp order, then the xor tree; every lane gets the sums.
template <class F, int NC>
__device__ __forceinline__ void row_dot(const typename F::Block* __restrict__ wr, const Q81Block* sx, int qb, int bpr,
                                        int lane, float (&out)[NC]) {
    float p[VT][NC];
#pragma unroll
    for (int v = 0; v < VT; ++v)
#pragma unroll
        for (int j = 0; j < NC; ++j) p[v][j] = 0.0f;
#pragma unroll
    for (int v = 0; v < VT; ++v) {
        const int tid = lane + WARP * v;
        const int kqs = F::kqs(tid);
        for (int kbx = tid / F::T; kbx < bpr; kbx += F::BPI) {
            const typename F::W r = F::load(wr + kbx, kqs);
            const int kby = kbx * 8;
#pragma unroll
            for (int j = 0; j < NC; ++j) p[v][j] += F::apply(r, sx + (size_t) j * qb + kby, kqs);
        }
    }
#pragma unroll
    for (int j = 0; j < NC; ++j) {
        float s = p[0][j];
#pragma unroll
        for (int v = 1; v < VT; ++v) s += p[v][j];
        out[j] = warp_sum(s);
    }
}

template <class F, int NC>
__global__ void __launch_bounds__(WPB * WARP) head_kernel(const typename F::Block* __restrict__ w,
                                                         const Q81Block* __restrict__ x, float* __restrict__ y,
                                                         int n_in, int n_out) {
    extern __shared__ int sx_raw[];   // NC columns of n_in / 32 q8_1 blocks, as native_quantize_q8_1 lays them out
    const int qb = n_in / 32;
    {
        const int words = NC * qb * (int) (sizeof(Q81Block) / 4);
        const int* src = reinterpret_cast<const int*>(x);
        for (int i = threadIdx.x; i < words; i += blockDim.x) sx_raw[i] = __ldg(src + i);
    }
    __syncthreads();
    const int lane = threadIdx.x % WARP;
    const int row = blockIdx.x * WPB + threadIdx.x / WARP;
    if (row >= n_out) return;
    const int bpr = n_in / 256;
    float s[NC];
    row_dot<F, NC>(w + (size_t) row * bpr, reinterpret_cast<const Q81Block*>(sx_raw), qb, bpr, lane, s);
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) y[(size_t) j * n_out + row] = s[j];
    }
}

template <class F, int NC>
bool launch_head(const void* w, const void* x, float* y, int n_in, int n_out, cudaStream_t s) {
    const size_t smem = (size_t) NC * (n_in / 32) * sizeof(Q81Block);
    if (smem > 48 * 1024) {
        static bool asked = false;
        if (!asked) {
            if (cudaFuncSetAttribute(head_kernel<F, NC>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int) ((size_t) 8 * (kMaxIn / 32) * sizeof(Q81Block))) != cudaSuccess)
                return false;
            asked = true;
        }
    }
    head_kernel<F, NC><<<(unsigned) ((n_out + WPB - 1) / WPB), WPB * WARP, smem, s>>>(
        (const typename F::Block*) w, (const Q81Block*) x, y, n_in, n_out);
    return true;
}

template <class F>
bool head_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!head_fmt_supported(n_in, ncols) || n_out <= 0) return false;
    const auto s = static_cast<cudaStream_t>(stream);
    bool ok = false;
    switch (ncols) {
    case 1: ok = launch_head<F, 1>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 2: ok = launch_head<F, 2>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 3: ok = launch_head<F, 3>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 4: ok = launch_head<F, 4>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 5: ok = launch_head<F, 5>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 6: ok = launch_head<F, 6>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 7: ok = launch_head<F, 7>(weights, x_q8_1, y, n_in, n_out, s); break;
    default: ok = launch_head<F, 8>(weights, x_q8_1, y, n_in, n_out, s); break;
    }
    if (!ok) return false;
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        klog_error("head_mmvq: %s", cudaGetErrorString(e));
        return false;
    }
    return true;
}

// basalt's 2-bit proxy: one thread per 32-weight group of a 2560-wide row into the copy
template <class F>
__global__ void proxy_pack_kernel(const uint8_t* __restrict__ rows, int64_t row0, int64_t n, uint4* __restrict__ wq,
                                  uint16_t* __restrict__ sc) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 80) return;
    const int64_t row = row0 + i / 80;
    const int g = (int) (i % 80);
    const typename F::Block* b = reinterpret_cast<const typename F::Block*>(rows + (size_t) row * F::ROW_BYTES_2560) + g / 8;
    float w[32];
    F::group(b, g % 8, w);
    px::pack_group<true>(w, row, g, wq, sc);
}

// One block: the rows near the best approximate score scored exactly (a warp a row, native_mmvq's bits), the exact
// best is the pick (basalt's final kernel with this format's arithmetic).
template <class F>
__global__ void __launch_bounds__(512) proxy_final_kernel(const typename F::Block* __restrict__ w,
                                                          const Q81Block* __restrict__ x, int n_tiles,
                                                          const int32_t* __restrict__ map, const float2* __restrict__ cand,
                                                          const float2* __restrict__ part, int* __restrict__ out_id,
                                                          float* __restrict__ out_prob) {
    __shared__ __align__(16) int sx_raw[80 * (int) (sizeof(Q81Block) / 4)];
    __shared__ px::Final f;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    pdl_wait();
    if (tid == 0) f.cnt = 0;
    {
        const int* src = reinterpret_cast<const int*>(x);
        for (int i = tid; i < 80 * (int) (sizeof(Q81Block) / 4); i += blockDim.x) sx_raw[i] = __ldg(src + i);
    }
    const int n = px::collect(f, n_tiles, map, cand, out_id, out_prob);
    if (n == 0) return;
    const Q81Block* sx = reinterpret_cast<const Q81Block*>(sx_raw);
    for (int c = warp; c < n; c += 16) {
        float y[1];
        row_dot<F, 1>(w + (size_t) f.list[c] * 10, sx, 80, 10, lane, y);
        if (lane == 0) f.ex[c] = y[0];
    }
    __syncthreads();
    if (warp != 0) return;
    px::pick(f, n, n_tiles, map, part, out_id, out_prob);
}

template <class F>
void proxy_pack(const void* rows, int64_t row0, int64_t n, void* proxy, int64_t n_cap, void* stream) {
    if (n <= 0) return;
    const int64_t total = n * 80;
    proxy_pack_kernel<F><<<(unsigned) ((total + 127) / 128), 128, 0, (cudaStream_t) stream>>>(
        (const uint8_t*) rows, row0, n, (uint4*) proxy, px::scales(proxy, n_cap));
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) klog_error("proxy_pack: %s", cudaGetErrorString(e));
}

template <class F>
bool proxy_top1(const void* proxy, const void* rows, const void* x_q8_1, int n_rows, int64_t n_cap, const int32_t* map,
                int* out_id, float* out_prob, void* scratch, void* stream) {
    if (n_rows <= 0 || n_rows > n_cap || !proxy || !rows || !x_q8_1 || !map || !out_id || !out_prob || !scratch)
        return false;
    const int tiles = (int) px::tiles(n_rows);
    const uint4* wq = (const uint4*) proxy;
    const uint32_t* sc = (const uint32_t*) px::scales(const_cast<void*>(proxy), n_cap);
    float2* cand = (float2*) scratch;
    float2* part = cand + (size_t) px::tiles(n_cap) * px::kPer;
    launch_pdl(px::score_kernel, dim3((unsigned) tiles), dim3(128), 0, (cudaStream_t) stream, wq, sc,
               (const px::X81*) x_q8_1, n_rows, map, cand, part);
    launch_pdl(proxy_final_kernel<F>, dim3(1), dim3(512), 0, (cudaStream_t) stream, (const typename F::Block*) rows,
               (const Q81Block*) x_q8_1, tiles, map, (const float2*) cand, (const float2*) part, out_id, out_prob);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        klog_error("proxy_top1: %s", cudaGetErrorString(e));
        return false;
    }
    return true;
}

}  // namespace

bool head_fmt_supported(int n_in, int ncols) {
    return n_in > 0 && n_in % 256 == 0 && n_in <= kMaxIn && ncols >= 1 && ncols <= 8;
}

bool iq4xs_head_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) {
    return head_mmvq<IQ4XS>(w, x, y, n_in, n_out, ncols, s);
}
bool q6k_head_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) {
    return head_mmvq<Q6K>(w, x, y, n_in, n_out, ncols, s);
}
bool q4k_head_mmvq(const void* w, const void* x, float* y, int n_in, int n_out, int ncols, void* s) {
    return head_mmvq<Q4K>(w, x, y, n_in, n_out, ncols, s);
}

size_t iq4xs_proxy_bytes(int64_t n_cap) { return px::bytes(n_cap); }
size_t iq4xs_proxy_scratch_bytes(int64_t n_cap) { return px::scratch_bytes(n_cap); }
void iq4xs_proxy_pack(const void* rows, int64_t row0, int64_t n, void* proxy, int64_t n_cap, void* s) {
    proxy_pack<IQ4XS>(rows, row0, n, proxy, n_cap, s);
}
bool iq4xs_proxy_top1(const void* proxy, const void* rows, const void* x, int n_rows, int64_t n_cap, const int32_t* map,
                      int* out_id, float* out_prob, void* scratch, void* s) {
    return proxy_top1<IQ4XS>(proxy, rows, x, n_rows, n_cap, map, out_id, out_prob, scratch, s);
}
size_t q6k_proxy_bytes(int64_t n_cap) { return px::bytes(n_cap); }
size_t q6k_proxy_scratch_bytes(int64_t n_cap) { return px::scratch_bytes(n_cap); }
void q6k_proxy_pack(const void* rows, int64_t row0, int64_t n, void* proxy, int64_t n_cap, void* s) {
    proxy_pack<Q6K>(rows, row0, n, proxy, n_cap, s);
}
bool q6k_proxy_top1(const void* proxy, const void* rows, const void* x, int n_rows, int64_t n_cap, const int32_t* map,
                    int* out_id, float* out_prob, void* scratch, void* s) {
    return proxy_top1<Q6K>(proxy, rows, x, n_rows, n_cap, map, out_id, out_prob, scratch, s);
}

size_t q4k_proxy_bytes(int64_t n_cap) { return px::bytes(n_cap); }
size_t q4k_proxy_scratch_bytes(int64_t n_cap) { return px::scratch_bytes(n_cap); }
void q4k_proxy_pack(const void* rows, int64_t row0, int64_t n, void* proxy, int64_t n_cap, void* s) {
    proxy_pack<Q4K>(rows, row0, n, proxy, n_cap, s);
}
bool q4k_proxy_top1(const void* proxy, const void* rows, const void* x, int n_rows, int64_t n_cap, const int32_t* map,
                    int* out_id, float* out_prob, void* scratch, void* s) {
    return proxy_top1<Q4K>(proxy, rows, x, n_rows, n_cap, map, out_id, out_prob, scratch, s);
}

}  // namespace strata::kernels
