// src/kernels/cuda/ffn_sparse.cu - see include/strata/kernels/ffn_sparse.hpp (TEAL / BlockFFN masks, the transposed
// down read, the LQER low-rank correction).  CUDA and HIP (wave32 and wave64: no warp-size assumption except in the
// one-warp reduction of lowrank_b_kernel, which uses shared memory instead of shuffles).
#include "strata/kernels/ffn_sparse.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

constexpr int kScanThreads = 1024;
constexpr int kMaxTok = 16;

__device__ __forceinline__ bool teal_active(float x, float tau) { return !(fabsf(x) < tau); }

// one block: chunks of 1024 columns; per chunk the activity flags, the zeroing, an inclusive scan of the flags in shared
// memory (Hillis-Steele) and the ascending index list
__global__ void teal_sparsify_kernel(float* __restrict__ h, int n_ff, int n_tok, float tau, int union_rows,
                                     int32_t* __restrict__ idx, int32_t* __restrict__ n_idx) {
    __shared__ int sc[kScanThreads];
    __shared__ int base_s;
    if (threadIdx.x == 0) base_s = 0;
    __syncthreads();
    for (int c0 = 0; c0 < n_ff; c0 += kScanThreads) {
        const int c = c0 + (int) threadIdx.x;
        int any = 0;
        if (c < n_ff) {
            for (int t = 0; t < n_tok; ++t) any |= teal_active(h[(size_t) t * n_ff + c], tau) ? 1 : 0;
            for (int t = 0; t < n_tok; ++t) {
                float& x = h[(size_t) t * n_ff + c];
                if (union_rows ? !any : !teal_active(x, tau)) x = 0.0f;
            }
        }
        sc[threadIdx.x] = any;
        __syncthreads();
        for (int o = 1; o < kScanThreads; o <<= 1) {
            const int v = threadIdx.x >= (unsigned) o ? sc[threadIdx.x - o] : 0;
            __syncthreads();
            sc[threadIdx.x] += v;
            __syncthreads();
        }
        const int base = base_s;
        if (any && idx != nullptr) idx[base + sc[threadIdx.x] - 1] = c;
        __syncthreads();
        if (threadIdx.x == kScanThreads - 1) base_s = base + sc[threadIdx.x];
        __syncthreads();
    }
    if (threadIdx.x == 0 && n_idx != nullptr) *n_idx = base_s;
}

// one thread per output: every row's sum over the listed columns, k order, fmaf (the host reference's order)
__global__ void down_t_gemv_kernel(const uint16_t* __restrict__ wt, const float* __restrict__ h,
                                   const int32_t* __restrict__ idx, const int32_t* __restrict__ n_idx,
                                   float* __restrict__ y, int n_ff, int n_out, int n_tok) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;
    float acc[kMaxTok];
#pragma unroll
    for (int t = 0; t < kMaxTok; ++t) acc[t] = 0.0f;
    const int n = *n_idx;
    for (int k = 0; k < n; ++k) {
        const int i = idx[k];
        const float w = __half2float(__ushort_as_half(wt[(size_t) i * n_out + o]));
#pragma unroll
        for (int t = 0; t < kMaxTok; ++t)
            if (t < n_tok) acc[t] = fmaf(w, h[(size_t) t * n_ff + i], acc[t]);
    }
    for (int t = 0; t < n_tok; ++t) y[(size_t) t * n_out + o] = acc[t];
}

struct Q81 {
    __half2 ds;
    int8_t qs[32];
};
static_assert(sizeof(Q81) == 36, "q8_1 block");

// tmp[c][j] = sum_i B[j][i] * x_c[i], one block per (j, c), a fixed-order shared-memory tree
__global__ void lowrank_b_kernel(const float* __restrict__ B, const Q81* __restrict__ x, float* __restrict__ tmp, int r,
                                 int n_in) {
    __shared__ float part[256];
    const int j = blockIdx.x, c = blockIdx.y;
    const int nb = n_in / 32;
    const Q81* xc = x + (size_t) c * nb;
    const float* Bj = B + (size_t) j * n_in;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n_in; i += blockDim.x) {
        const Q81& q = xc[i >> 5];
        const float d = __low2float(q.ds);
        s = fmaf(Bj[i], d * (float) q.qs[i & 31], s);
    }
    part[threadIdx.x] = s;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) part[threadIdx.x] += part[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) tmp[(size_t) c * r + j] = part[0];
}

// y[c][o] += sum_j A[o][j] * tmp[c][j], j order, fmaf
__global__ void lowrank_a_kernel(const float* __restrict__ A, const float* __restrict__ tmp, float* __restrict__ y, int r,
                                 int n_out) {
    const int o = blockIdx.x * blockDim.x + threadIdx.x, c = blockIdx.y;
    if (o >= n_out) return;
    const float* Ao = A + (size_t) o * r;
    const float* tc = tmp + (size_t) c * r;
    float s = 0.0f;
    for (int j = 0; j < r; ++j) s = fmaf(Ao[j], tc[j], s);
    y[(size_t) c * n_out + o] += s;
}

struct LqerEntry {
    const float* A = nullptr;
    const float* B = nullptr;
    float* tmp = nullptr;
    int r = 0, n_in = 0, n_out = 0;
};
std::unordered_map<const void*, LqerEntry>& lqer_map() {
    static std::unordered_map<const void*, LqerEntry> m;
    return m;
}
std::mutex& lqer_mu() {
    static std::mutex m;
    return m;
}
bool g_lqer_any = false;

}  // namespace

void teal_sparsify(float* h, int n_ff, int n_tok, float tau, bool union_rows, int32_t* idx, int32_t* n_idx, void* stream) {
    if (n_ff <= 0 || n_tok < 1 || n_tok > kMaxTok) { std::fprintf(stderr, "teal_sparsify: shape out of range\n"); std::exit(1); }
    teal_sparsify_kernel<<<1, kScanThreads, 0, (cudaStream_t) stream>>>(h, n_ff, n_tok, tau, union_rows ? 1 : 0, idx, n_idx);
    check("teal_sparsify");
}

void down_t_gemv(const uint16_t* wt, const float* h, const int32_t* idx, const int32_t* n_idx, float* y, int n_ff, int n_out,
                 int n_tok, void* stream) {
    if (n_out <= 0 || n_tok < 1 || n_tok > kMaxTok) { std::fprintf(stderr, "down_t_gemv: shape out of range\n"); std::exit(1); }
    down_t_gemv_kernel<<<(unsigned) ((n_out + 255) / 256), 256, 0, (cudaStream_t) stream>>>(wt, h, idx, n_idx, y, n_ff, n_out,
                                                                                           n_tok);
    check("down_t_gemv");
}

void lowrank_q81_apply(const float* A, const float* B, int r, const void* x_q8_1, float* tmp, float* y, int n_in, int n_out,
                       int ncols, void* stream) {
    if (r <= 0 || n_in % 32 != 0 || ncols < 1 || ncols > kMaxTok) {
        std::fprintf(stderr, "lowrank_q81_apply: shape out of range\n");
        std::exit(1);
    }
    cudaStream_t s = (cudaStream_t) stream;
    lowrank_b_kernel<<<dim3((unsigned) r, (unsigned) ncols), 256, 0, s>>>(B, (const Q81*) x_q8_1, tmp, r, n_in);
    lowrank_a_kernel<<<dim3((unsigned) ((n_out + 255) / 256), (unsigned) ncols), 256, 0, s>>>(A, tmp, y, r, n_out);
    check("lowrank_q81_apply");
}

bool lqer_register(const void* weights, const float* A, const float* B, int r, int n_in, int n_out) {
    if (weights == nullptr || A == nullptr || B == nullptr || r <= 0 || n_in % 32 != 0 || n_out <= 0) return false;
    float* tmp = nullptr;
    if (cudaMalloc((void**) &tmp, (size_t) kMaxTok * (size_t) r * sizeof(float)) != cudaSuccess) {
        (void) cudaGetLastError();
        return false;
    }
    std::lock_guard<std::mutex> g(lqer_mu());
    lqer_map()[weights] = LqerEntry{A, B, tmp, r, n_in, n_out};
    g_lqer_any = true;
    return true;
}

bool lqer_any() { return g_lqer_any; }

void lqer_apply_if(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    LqerEntry e;
    {
        std::lock_guard<std::mutex> g(lqer_mu());
        const auto it = lqer_map().find(weights);
        if (it == lqer_map().end()) return;
        e = it->second;
    }
    if (e.n_in != n_in || e.n_out != n_out) return;   // another view of the same bytes: not this factorization
    for (int c = 0; c < ncols; c += kMaxTok) {
        const int nc = ncols - c < kMaxTok ? ncols - c : kMaxTok;
        lowrank_q81_apply(e.A, e.B, e.r, (const Q81*) x_q8_1 + (size_t) c * (n_in / 32), e.tmp, y + (size_t) c * n_out, n_in,
                          n_out, nc, stream);
    }
}

}  // namespace strata::kernels
