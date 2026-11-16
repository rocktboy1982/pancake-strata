// src/kernels/cuda/rkv.cu - see include/strata/kernels/rkv.hpp (R-KV eviction of decode-trace blocks from the QSA
// selection; opt-in STRATA_RKV, lossy).  CUDA and HIP; one block of 1024 threads, no warp-size assumption.
#include "strata/kernels/rkv.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

constexpr int kT = 1024;

__global__ void rkv_update_kernel(const float* __restrict__ scores, int n, long long max_blocks,
                                  const int32_t* __restrict__ steps, int step_count, const float* __restrict__ pooled,
                                  int idx_dim, const int32_t* __restrict__ start_block, int budget, int buffer, int sink,
                                  int recent, float lambda, uint32_t* __restrict__ evict) {
    __shared__ int cand[kRkvCap];
    __shared__ float nrm[kRkvCap];
    __shared__ float zv[kRkvCap];
    __shared__ int sid[kRkvCap];
    __shared__ int sc[kT];
    __shared__ float rmin[kT], rmax[kT];
    __shared__ int lo_s, hi_s, m_s;
    const int tid = (int) threadIdx.x;
    if (tid == 0) {
        const int nb = steps[(size_t) (n - 1) * step_count + 2];
        const int sb = *start_block;
        lo_s = sink > sb ? sink : sb;
        hi_s = nb - recent;
        m_s = 0;
    }
    __syncthreads();
    const int lo = lo_s, hi = hi_s;
    if (hi - lo < budget + buffer) return;   // (uniform: every thread returns)
    // the candidates: not yet evicted, ascending, at most kRkvCap
    for (int b0 = lo; b0 < hi; b0 += kT) {
        const int b = b0 + tid;
        const int live = b < hi && !((evict[b >> 5] >> (b & 31)) & 1u) ? 1 : 0;
        sc[tid] = live;
        __syncthreads();
        for (int o = 1; o < kT; o <<= 1) {
            const int v = tid >= o ? sc[tid - o] : 0;
            __syncthreads();
            sc[tid] += v;
            __syncthreads();
        }
        const int base = m_s;
        const int at = base + sc[tid] - 1;
        if (live && at < kRkvCap) cand[at] = b;
        __syncthreads();
        if (tid == kT - 1) m_s = base + sc[tid] < kRkvCap ? base + sc[tid] : kRkvCap;
        __syncthreads();
        if (m_s >= kRkvCap) break;
    }
    const int M = m_s;
    if (M < budget + buffer) return;
    // importance (max over the window's rows) and key norms
    float lmin = 3e38f, lmax = -3e38f;
    for (int i = tid; i < M; i += kT) {
        const int b = cand[i];
        float v = -3e38f;
        for (int r = 0; r < n; ++r) v = fmaxf(v, scores[(size_t) r * max_blocks + b]);
        zv[i] = v;
        lmin = fminf(lmin, v);
        lmax = fmaxf(lmax, v);
        float s = 0.0f;
        const float* k = pooled + (size_t) b * idx_dim;
        for (int d = 0; d < idx_dim; ++d) s = fmaf(k[d], k[d], s);
        nrm[i] = sqrtf(s) + 1e-12f;
    }
    rmin[tid] = lmin;
    rmax[tid] = lmax;
    __syncthreads();
    for (int o = kT / 2; o > 0; o >>= 1) {
        if (tid < o) {
            rmin[tid] = fminf(rmin[tid], rmin[tid + o]);
            rmax[tid] = fmaxf(rmax[tid], rmax[tid + o]);
        }
        __syncthreads();
    }
    const float mn = rmin[0], mx = rmax[0];
    const float span = mx - mn > 1e-30f ? mx - mn : 1.0f;
    // redundancy and the score
    for (int i = tid; i < M; i += kT) {
        const float* ki = pooled + (size_t) cand[i] * idx_dim;
        float rsum = 0.0f;
        for (int j = 0; j < M; ++j) {
            if (j == i) continue;
            const float* kj = pooled + (size_t) cand[j] * idx_dim;
            float d = 0.0f;
            for (int e = 0; e < idx_dim; ++e) d = fmaf(ki[e], kj[e], d);
            rsum += d / (nrm[i] * nrm[j]);
        }
        const float R = rsum / (float) (M - 1);
        zv[i] = fmaf(lambda, (zv[i] - mn) / span, -((1.0f - lambda) * R));   // explicit: the host reference's order
    }
    __syncthreads();
    // bitonic sort of (Z, candidate index), ascending; the padding sorts last
    for (int i = tid; i < kRkvCap; i += kT) {
        if (i >= M) zv[i] = 3.4e38f;
        sid[i] = i;
    }
    __syncthreads();
    for (int k = 2; k <= kRkvCap; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < kRkvCap; i += kT) {
                const int x = i ^ j;
                if (x > i) {
                    const float zi = zv[i], zx = zv[x];
                    const int si = sid[i], sx = sid[x];
                    const bool gt = zi > zx || (zi == zx && si > sx);   // element i after element x
                    const bool up = (i & k) == 0;
                    if (gt == up) { zv[i] = zx; zv[x] = zi; sid[i] = sx; sid[x] = si; }
                }
            }
            __syncthreads();
        }
    for (int k = tid; k < M - budget; k += kT) {
        const int b = cand[sid[k]];
        atomicOr(&evict[b >> 5], 1u << (b & 31));
    }
}

__global__ void rkv_mask_kernel(float* __restrict__ scores, long long max_blocks, const uint32_t* __restrict__ evict) {
    const long long b = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= max_blocks) return;
    if ((evict[b >> 5] >> (b & 31)) & 1u) scores[(size_t) blockIdx.y * max_blocks + b] = -3e38f;
}

}  // namespace

void rkv_update(const float* scores, int n, int64_t max_blocks, const int32_t* steps, int step_count, const float* pooled,
                int idx_dim, const int32_t* start_block, const RkvConfig& c, uint32_t* evict, void* stream) {
    if (n < 1 || c.budget <= 0) return;
    rkv_update_kernel<<<1, kT, 0, (cudaStream_t) stream>>>(scores, n, (long long) max_blocks, steps, step_count, pooled,
                                                          idx_dim, start_block, c.budget, c.buffer, c.sink, c.recent,
                                                          c.lambda, evict);
    check("rkv_update");
}

void rkv_mask_scores(float* scores, int n, int64_t max_blocks, const uint32_t* evict, void* stream) {
    if (n < 1 || max_blocks <= 0) return;
    rkv_mask_kernel<<<dim3((unsigned) ((max_blocks + 255) / 256), (unsigned) n), 256, 0, (cudaStream_t) stream>>>(
        scores, (long long) max_blocks, evict);
    check("rkv_mask_scores");
}

}  // namespace strata::kernels
