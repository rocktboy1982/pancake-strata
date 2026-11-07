// src/kernels/cuda/iq4xs_head.cu - see include/strata/kernels/iq4xs_head.hpp.  pancake-strata, after basalt's
// q5k_head.cu (jesdga95/basalt a9468e5c, MIT); the IQ4_XS arithmetic is native_mmvq.cu's IQ4XSTraits, unchanged.
#include "strata/kernels/iq4xs_head.hpp"
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
constexpr int VT = 4;            // virtual threads per lane: native_mmvq's four warps (WARPS) of one row
constexpr int WPB = 8;           // warps, and rows, per block
constexpr int BPI = 4 * VT * WARP / 32;   // native_mmvq's IQ4XSTraits::BPI: blocks per iteration of a row's threads
constexpr int kMaxIn = 8192;     // the columns' activations a block keeps in shared memory: 8 x 8192 / 32 x 36 B

struct IQ4XSBlock {
    half d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[128];
};
struct Q81Block {
    half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(IQ4XSBlock) == 136 && sizeof(Q81Block) == 36);

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

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o, WARP);
    return x;
}

// IQ4XSTraits::load and ::apply, the weight half decoded once for the block's columns
struct W4 {
    int2 v[4];
    int ls;
    float dw;
};
__device__ __forceinline__ W4 iq4_load(const IQ4XSBlock* __restrict__ w, int iqs) {
    W4 r;
#pragma unroll
    for (int j = 0; j < 4; ++j) r.v[j] = iq4_lookup(__ldg(reinterpret_cast<const int*>(w->qs) + iqs + j));
    r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
    r.dw = __half2float(w->d);
    return r;
}
__device__ __forceinline__ float iq4_apply(const W4& r, const Q81Block* x, int iqs) {
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

// One row by one warp, NC columns of activations at sx (qb q8_1 blocks apart): native_mmvq's per-thread blocks and its
// cross-warp order (lane L stands for its threads L, L + 32, L + 64, L + 96), then the xor tree; every lane gets the sums.
template <int NC>
__device__ __forceinline__ void iq4_row(const IQ4XSBlock* __restrict__ wr, const Q81Block* sx, int qb, int bpr, int lane,
                                        float (&out)[NC]) {
    float p[VT][NC];
#pragma unroll
    for (int v = 0; v < VT; ++v)
#pragma unroll
        for (int j = 0; j < NC; ++j) p[v][j] = 0.0f;
#pragma unroll
    for (int v = 0; v < VT; ++v) {
        const int tid = lane + WARP * v;
        const int kqs = 4 * (tid % 8);
        for (int kbx = tid / 8; kbx < bpr; kbx += BPI) {
            const W4 r = iq4_load(wr + kbx, kqs);
            const int kby = kbx * 8;
#pragma unroll
            for (int j = 0; j < NC; ++j) p[v][j] += iq4_apply(r, sx + (size_t) j * qb + kby, kqs);
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

template <int NC>
__global__ void __launch_bounds__(WPB * WARP) iq4xs_head_kernel(const IQ4XSBlock* __restrict__ w,
                                                               const Q81Block* __restrict__ x, float* __restrict__ y,
                                                               int n_in, int n_out) {
    extern __shared__ int sx_raw[];   // NC columns of n_in / 32 q8_1 blocks, as native_quantize_q8_1 lays them out
    const int qb = n_in / 32;         // q8_1 blocks of a column
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
    iq4_row<NC>(w + (size_t) row * bpr, reinterpret_cast<const Q81Block*>(sx_raw), qb, bpr, lane, s);
    if (lane == 0) {
#pragma unroll
        for (int j = 0; j < NC; ++j) y[(size_t) j * n_out + row] = s[j];
    }
}

template <int NC>
bool launch(const void* w, const void* x, float* y, int n_in, int n_out, cudaStream_t s) {
    const size_t smem = (size_t) NC * (n_in / 32) * sizeof(Q81Block);
    if (smem > 48 * 1024) {
        static bool asked = false;   // one opt-in per instance (the size is bounded by kMaxIn)
        if (!asked) {
            if (cudaFuncSetAttribute(iq4xs_head_kernel<NC>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int) ((size_t) 8 * (kMaxIn / 32) * sizeof(Q81Block))) != cudaSuccess)
                return false;
            asked = true;
        }
    }
    const unsigned blocks = (unsigned) ((n_out + WPB - 1) / WPB);
    iq4xs_head_kernel<NC><<<blocks, WPB * WARP, smem, s>>>((const IQ4XSBlock*) w, (const Q81Block*) x, y, n_in, n_out);
    return true;
}

}  // namespace

bool iq4xs_head_supported(int n_in, int ncols) {
    return n_in > 0 && n_in % 256 == 0 && n_in <= kMaxIn && ncols >= 1 && ncols <= 8;
}

bool iq4xs_head_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    if (!iq4xs_head_supported(n_in, ncols) || n_out <= 0) return false;
    const auto s = static_cast<cudaStream_t>(stream);
    bool ok = false;
    switch (ncols) {
    case 1: ok = launch<1>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 2: ok = launch<2>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 3: ok = launch<3>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 4: ok = launch<4>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 5: ok = launch<5>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 6: ok = launch<6>(weights, x_q8_1, y, n_in, n_out, s); break;
    case 7: ok = launch<7>(weights, x_q8_1, y, n_in, n_out, s); break;
    default: ok = launch<8>(weights, x_q8_1, y, n_in, n_out, s); break;
    }
    if (!ok) return false;
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "strata: iq4xs_head_mmvq: %s\n", cudaGetErrorString(e));
        return false;
    }
    return true;
}


// ---- basalt's 2-bit proxy and two-stage argmax (q5k_head.cu's q5k_proxy_*), for IQ4_XS rows (n_in 2560)

namespace {

constexpr int kRowBytes2560 = 10 * 136;

// One thread per 32-weight group: the group's dequantized IQ4_XS values (the codebook times the block's 6-bit
// sub-scale) into the 2-bit copy.
__global__ void iq4xs_proxy_pack_kernel(const uint8_t* __restrict__ rows, int64_t row0, int64_t n,
                                        uint4* __restrict__ wq, uint16_t* __restrict__ sc) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * 80) return;
    const int64_t row = row0 + i / 80;
    const int g = (int) (i % 80);
    const IQ4XSBlock* b = reinterpret_cast<const IQ4XSBlock*>(rows + (size_t) row * kRowBytes2560) + g / 8;
    const int j = g % 8;
    const int ls = ((b->scales_l[j / 2] >> (4 * (j % 2))) & 0x0f) | (((b->scales_h >> (2 * j)) & 0x03) << 4);
    const float d = __half2float(b->d) * (float) (ls - 32);
    float w[32];
    for (int l = 0; l < 16; ++l) {
        const uint8_t q = b->qs[16 * j + l];
        w[l] = d * (float) kIq4nl[q & 0x0f];
        w[l + 16] = d * (float) kIq4nl[q >> 4];
    }
    px::pack_group<true>(w, row, g, wq, sc);
}

// One block: the rows near the best approximate score are scored exactly (a warp a row, native_mmvq's bits), the
// exact best is the pick.
__global__ void __launch_bounds__(512) iq4xs_proxy_final_kernel(const IQ4XSBlock* __restrict__ w,
                                                                const Q81Block* __restrict__ x, int n_tiles,
                                                                const int32_t* __restrict__ map,
                                                                const float2* __restrict__ cand,
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
        iq4_row<1>(w + (size_t) f.list[c] * 10, sx, 80, 10, lane, y);
        if (lane == 0) f.ex[c] = y[0];
    }
    __syncthreads();
    if (warp != 0) return;
    px::pick(f, n, n_tiles, map, part, out_id, out_prob);
}

}  // namespace

size_t iq4xs_proxy_bytes(int64_t n_cap) { return px::bytes(n_cap); }

size_t iq4xs_proxy_scratch_bytes(int64_t n_cap) { return px::scratch_bytes(n_cap); }

void iq4xs_proxy_pack(const void* rows, int64_t row0, int64_t n, void* proxy, int64_t n_cap, void* stream) {
    if (n <= 0) return;
    const int64_t total = n * 80;
    iq4xs_proxy_pack_kernel<<<(unsigned) ((total + 127) / 128), 128, 0, (cudaStream_t) stream>>>(
        (const uint8_t*) rows, row0, n, (uint4*) proxy, px::scales(proxy, n_cap));
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) klog_error("iq4xs_proxy_pack: %s", cudaGetErrorString(e));
}

bool iq4xs_proxy_top1(const void* proxy, const void* rows, const void* x_q8_1, int n_rows, int64_t n_cap,
                      const int32_t* map, int* out_id, float* out_prob, void* scratch, void* stream) {
    if (n_rows <= 0 || n_rows > n_cap || !proxy || !rows || !x_q8_1 || !map || !out_id || !out_prob || !scratch)
        return false;
    const int tiles = (int) px::tiles(n_rows);
    const uint4* wq = (const uint4*) proxy;
    const uint32_t* sc = (const uint32_t*) px::scales(const_cast<void*>(proxy), n_cap);
    float2* cand = (float2*) scratch;
    float2* part = cand + (size_t) px::tiles(n_cap) * px::kPer;
    launch_pdl(px::score_kernel, dim3((unsigned) tiles), dim3(128), 0, (cudaStream_t) stream, wq, sc,
               (const px::X81*) x_q8_1, n_rows, map, cand, part);
    launch_pdl(iq4xs_proxy_final_kernel, dim3(1), dim3(512), 0, (cudaStream_t) stream, (const IQ4XSBlock*) rows,
               (const Q81Block*) x_q8_1, tiles, map, (const float2*) cand, (const float2*) part, out_id, out_prob);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        klog_error("iq4xs_proxy_top1: %s", cudaGetErrorString(e));
        return false;
    }
    return true;
}

}  // namespace strata::kernels
