// src/kernels/cuda/xattn.cu - see include/strata/kernels/xattn.hpp (XAttention antidiagonal block selection + block-sparse
// prompt attention; opt-in STRATA_XATTN, lossy).  CUDA and wave32 HIP; the attention kernel is one 32-lane wave per query
// row and head, so a wave64 build (gfx906) answers false and the caller keeps its path.
#include "strata/kernels/xattn.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

constexpr int kSelT = 256;

__device__ __forceinline__ float h2f(uint16_t u) { return __half2float(__ushort_as_half(u)); }

__device__ float block_max(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int o = kSelT / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}
__device__ float block_sum(float v, float* red) {
    red[threadIdx.x] = v;
    __syncthreads();
    for (int o = kSelT / 2; o > 0; o >>= 1) {
        if ((int) threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    const float r = red[0];
    __syncthreads();
    return r;
}

// one block per (query block, head): antidiagonal scores, the per-row softmax, the key-block mass, the threshold selection
__global__ void xattn_select_kernel(const uint16_t* __restrict__ q16, const uint16_t* __restrict__ k16, long long T,
                                    long long p0, int H, int HKV, int HD, int S, float tau, long long nkb,
                                    uint8_t* __restrict__ mask, float* __restrict__ rowbuf_all) {
    __shared__ float A[kXattnMaxKb];
    __shared__ int sid[kXattnMaxKb];
    __shared__ float red[kSelT];
    const int qb = blockIdx.x, h = blockIdx.y, nqb = gridDim.x;
    const long long P = p0 + T, B = kXattnBlock, R = B / S;
    const int kvh = h / (H / HKV);
    const long long t_end = T < (long long) qb * B + B ? T : (long long) qb * B + B;
    const long long nkq = (p0 + t_end - 1) / B + 1;
    float* rowbuf = rowbuf_all + ((size_t) h * nqb + qb) * (size_t) nkb * (size_t) R;
    for (long long kb = threadIdx.x; kb < nkq; kb += kSelT) A[kb] = 0.0f;
    __syncthreads();
    const float scale = rsqrtf((float) HD) / (float) S;
    int rows = 0;
    for (long long a = 0; a < R; ++a) {
        const long long t0 = (long long) qb * B + a * S;
        if (t0 >= T) break;
        const long long qpos = p0 + t0;
        float lmax = -INFINITY;
        for (long long idx = threadIdx.x; idx < nkq * R; idx += kSelT) {
            const long long kb = idx / R, cc = idx - kb * R, kc = kb * B + cc * S;
            float s = -INFINITY;
            if (kc <= qpos && kc < P) {
                float acc = 0.0f;
                for (int kk = 0; kk < S; ++kk) {
                    const long long t = t0 + (S - 1 - kk), cell = kc + kk;
                    if (t >= T || cell >= P) continue;
                    const uint16_t* qr = q16 + ((size_t) t * H + h) * HD;
                    const uint16_t* kr = k16 + ((size_t) kvh * P + cell) * HD;
                    for (int d = 0; d < HD; ++d) acc = fmaf(h2f(qr[d]), h2f(kr[d]), acc);
                }
                s = acc * scale;
            }
            rowbuf[idx] = s;
            lmax = fmaxf(lmax, s);
        }
        const float mx = block_max(lmax, red);   // (its barriers also order the rowbuf writes before the reads below)
        float lsum = 0.0f;
        for (long long idx = threadIdx.x; idx < nkq * R; idx += kSelT) lsum += __expf(rowbuf[idx] - mx);
        const float sum = block_sum(lsum, red);
        for (long long kb = threadIdx.x; kb < nkq; kb += kSelT) {
            float acc = 0.0f;
            for (long long cc = 0; cc < R; ++cc) acc += __expf(rowbuf[kb * R + cc] - mx);
            A[kb] += acc / sum;
        }
        __syncthreads();
        ++rows;
    }
    // descending by mass, ties to the smaller block: bitonic over the next power of two
    int N = 1;
    while (N < nkq) N <<= 1;
    for (int i = threadIdx.x; i < N; i += kSelT) {
        if (i < nkq) A[i] = rows > 0 ? A[i] / (float) rows : 0.0f;
        else A[i] = -1.0f;
        sid[i] = i;
    }
    __syncthreads();
    for (int k = 2; k <= N; k <<= 1)
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < N; i += kSelT) {
                const int x = i ^ j;
                if (x > i) {
                    const float ai = A[i], ax = A[x];
                    const int si = sid[i], sx = sid[x];
                    const bool after = ai < ax || (ai == ax && si > sx);   // element i belongs after element x
                    const bool up = (i & k) == 0;
                    if (after == up) { A[i] = ax; A[x] = ai; sid[i] = sx; sid[x] = si; }
                }
            }
            __syncthreads();
        }
    uint8_t* m = mask + ((size_t) h * nqb + qb) * (size_t) nkb;
    for (long long kb = threadIdx.x; kb < nkb; kb += kSelT) m[kb] = 0;
    __syncthreads();
    if (threadIdx.x == 0 && rows > 0) {
        float cum = 0.0f;
        for (int i = 0; i < nkq; ++i) {
            if (cum >= tau) break;
            m[sid[i]] = 1;
            cum += A[i];
        }
        m[0] = 1;
        for (long long kb = (p0 + (long long) qb * B) / B; kb < nkq; ++kb) m[kb] = 1;
    }
}

constexpr int kMaxDimLane = 8;   // HD <= 256

// one 32-lane wave per (query row, head): online softmax over the kept blocks' causal cells
__global__ void xattn_attn_kernel(const uint16_t* __restrict__ q16, const uint16_t* __restrict__ k16,
                                  const uint16_t* __restrict__ v16, long long T, long long p0, int H, int HKV, int HD,
                                  long long nkb, int nqb, const uint8_t* __restrict__ mask, float* __restrict__ out) {
    const long long t = blockIdx.x;
    const int h = blockIdx.y, lane = threadIdx.x;
    const long long P = p0 + T, B = kXattnBlock, qpos = p0 + t;
    const int kvh = h / (H / HKV), nd = HD / 32;
    const int qb = (int) (t / B);
    const uint8_t* m = mask + ((size_t) h * nqb + qb) * (size_t) nkb;
    float qv[kMaxDimLane], acc[kMaxDimLane];
    const uint16_t* qr = q16 + ((size_t) t * H + h) * HD;
    for (int i = 0; i < kMaxDimLane; ++i) {
        qv[i] = i < nd ? h2f(qr[lane + 32 * i]) : 0.0f;
        acc[i] = 0.0f;
    }
    const float scale = rsqrtf((float) HD);
    float mrun = -INFINITY, l = 0.0f;
    for (long long kb = 0; kb <= qpos / B; ++kb) {
        if (!m[kb]) continue;
        const long long c1 = kb * B + B - 1 < qpos ? kb * B + B - 1 : qpos;
        for (long long cell = kb * B; cell <= c1; ++cell) {
            const uint16_t* kr = k16 + ((size_t) kvh * P + cell) * HD;
            float s = 0.0f;
            for (int i = 0; i < nd; ++i) s = fmaf(qv[i], h2f(kr[lane + 32 * i]), s);
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
            s *= scale;
            const float mn = fmaxf(mrun, s), corr = __expf(mrun - mn), e = __expf(s - mn);
            const uint16_t* vr = v16 + ((size_t) kvh * P + cell) * HD;
            for (int i = 0; i < nd; ++i) acc[i] = fmaf(acc[i], corr, e * h2f(vr[lane + 32 * i]));
            l = l * corr + e;
            mrun = mn;
        }
    }
    float* o = out + ((size_t) t * H + h) * HD;
    for (int i = 0; i < nd; ++i) o[lane + 32 * i] = l > 0.0f ? acc[i] / l : 0.0f;
}

}  // namespace

bool xattn_prefill_attn(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t p0, int H, int HKV,
                        int HD, const XattnConfig& c, void* scratch, size_t scratch_bytes, float* out, void* stream,
                        const uint8_t** mask_out) {
#if defined(STRATA_HIP_GFX906)
    (void) q16, (void) k16, (void) v16, (void) T, (void) p0, (void) H, (void) HKV, (void) HD, (void) c, (void) scratch,
        (void) scratch_bytes, (void) out, (void) stream, (void) mask_out;
    return false;   // wave64: the attention kernel is written for 32 lanes
#else
    if (T < 1 || HD % 32 != 0 || HD > 32 * kMaxDimLane || HKV < 1 || H % HKV != 0 || scratch == nullptr) return false;
    if (c.stride != 4 && c.stride != 8 && c.stride != 16) return false;
    const int64_t P = p0 + T, nqb = (T + kXattnBlock - 1) / kXattnBlock, nkb = (P + kXattnBlock - 1) / kXattnBlock;
    if (nkb > kXattnMaxKb || scratch_bytes < xattn_scratch_bytes(T, P, H, c.stride)) return false;
    uint8_t* mask = (uint8_t*) scratch;
    float* rowbuf = (float*) ((uint8_t*) scratch + (((size_t) nqb * H * nkb + 255) & ~(size_t) 255));
    cudaStream_t s = (cudaStream_t) stream;
    xattn_select_kernel<<<dim3((unsigned) nqb, (unsigned) H), kSelT, 0, s>>>(q16, k16, (long long) T, (long long) p0, H, HKV,
                                                                            HD, c.stride, c.tau, (long long) nkb, mask, rowbuf);
    check("xattn_select");
    xattn_attn_kernel<<<dim3((unsigned) T, (unsigned) H), 32, 0, s>>>(q16, k16, v16, (long long) T, (long long) p0, H, HKV, HD,
                                                                     (long long) nkb, (int) nqb, mask, out);
    check("xattn_attn");
    if (mask_out != nullptr) *mask_out = mask;
    return true;
#endif
}

}  // namespace strata::kernels
