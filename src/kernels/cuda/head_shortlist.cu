// src/kernels/cuda/head_shortlist.cu - STRATA_HEAD_SHORTLIST (opt-in, LOSSY), the reference mask; see head_shortlist.hpp.
// NOT BUILT OR RUN on the box (written 2026-11-16 on a machine without the toolchain; only syntax-checked).  Two launches:
//   score:  grid (k, ncols), 256 threads: centre c . hidden j in include/strata/program/head_shortlist.hpp shortlist_dot's order
//           (256 lane sums strided by 256, then the halving tree), into scratch[j * k + c];
//   mask:   grid ncols, 256 threads: keep[c] = fewer than n_keep clusters sort before c (larger score first, lower index on a
//           tie, NaN last: shortlist_select's rule), then every logit whose cluster is not kept becomes -inf.
// Our own code from the idea only (kindi, AGPL: nothing copied).
#include "strata/kernels/head_shortlist.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace strata::kernels {
namespace {

__global__ void head_shortlist_score_kernel(const float* __restrict__ hidden, const float* __restrict__ centre,
                                            float* __restrict__ scratch, int n_embd, int k) {
    __shared__ float lane[256];
    const int c = blockIdx.x, j = blockIdx.y, i = threadIdx.x;
    const float* a = hidden + (size_t) j * n_embd;
    const float* b = centre + (size_t) c * n_embd;
    float s = 0.0f;
    for (int t = i; t < n_embd; t += 256) s += a[t] * b[t];
    lane[i] = s;
    __syncthreads();
    for (int h = 128; h >= 1; h >>= 1) {
        if (i < h) lane[i] += lane[i + h];
        __syncthreads();
    }
    if (i == 0) scratch[(size_t) j * k + c] = lane[0];
}

__global__ void head_shortlist_mask_kernel(const float* __restrict__ scratch, const int32_t* __restrict__ cluster,
                                           float* __restrict__ logits, int vocab, int k, int n_keep) {
    extern __shared__ unsigned char smem[];
    float* score = reinterpret_cast<float*>(smem);
    unsigned char* keep = smem + (size_t) k * sizeof(float);
    const int j = blockIdx.x, i = threadIdx.x;
    for (int c = i; c < k; c += 256) score[c] = scratch[(size_t) j * k + c];
    __syncthreads();
    for (int c = i; c < k; c += 256) {
        const float sc = score[c];
        const bool cn = sc != sc;
        int rank = 0;
        for (int o = 0; o < k && rank < n_keep; ++o) {
            if (o == c) continue;
            const float so = score[o];
            const bool on = so != so;
            const bool before = on != cn ? cn : (on ? o < c : (so > sc || (so == sc && o < c)));
            rank += before ? 1 : 0;
        }
        keep[c] = rank < n_keep ? 1 : 0;
    }
    __syncthreads();
    float* row = logits + (size_t) j * vocab;
    for (int v = i; v < vocab; v += 256)
        if (!keep[cluster[v]]) row[v] = -__int_as_float(0x7f800000);   // -inf
}

}  // namespace

std::size_t head_shortlist_scratch_bytes(int ncols, int k) { return (std::size_t) ncols * (std::size_t) k * sizeof(float); }

void head_shortlist_mask(const float* hidden, const float* centre, const int32_t* cluster, float* logits, int n_embd, int vocab,
                         int k, int n_keep, int ncols, float* scratch, void* stream) {
    if (n_embd <= 0 || vocab <= 0 || k <= 0 || k > kHeadShortlistMaxK || ncols <= 0 || ncols > kHeadShortlistMaxCols)
        throw std::invalid_argument("head_shortlist_mask: unsupported shape");
    n_keep = n_keep < 1 ? 1 : (n_keep > k ? k : n_keep);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    head_shortlist_score_kernel<<<dim3(unsigned(k), unsigned(ncols)), 256, 0, s>>>(hidden, centre, scratch, n_embd, k);
    head_shortlist_mask_kernel<<<unsigned(ncols), 256, (size_t) k * (sizeof(float) + 1), s>>>(scratch, cluster, logits, vocab, k, n_keep);
    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("head_shortlist_mask: launch failed");
}

}  // namespace strata::kernels
