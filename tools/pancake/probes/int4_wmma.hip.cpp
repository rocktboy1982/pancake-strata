// int4_wmma.hip.cpp - the rotated-INT4 mechanism of SILVERTHRONE32/llama.cpp-rdna3-int4 (MIT, Sachin Mahesh; LEADS.md Part L L3, README.md
// section 9) as a standalone HIP kernel set + parity test + bench for gfx1100, written here from their report; NOT their code, NOT wired
// into the engine (see MOD_PLAN.md section 6 item 54 for why).
//
// WHAT       (1) rot_kernel: x' = R x per B-block (signs, in-LDS Sylvester butterflies, 1/sqrt(B)), the order of
//                include/strata/kernels/int4_rot.hpp; (2) quant_kernel: A8 (q = 16 h + l) or A4 (per-block multiplier of a per-token reference,
//                clip 0.85) codes; (3) pack_kernel: int8 codes -> 16 nibbles = two 32-bit words per lane operand; (4) gemm_kernel: one wave per
//                16x16 output tile, `v_wmma_i32_16x16x16_iu4` (weights signed int4 as A, activations as B: the h pass signed, the l pass
//                unsigned), the int32 accumulators of one 32-group scaled by weight scale x activation block scale in f32.  Untuned on purpose
//                (no LDS staging, no 4-wave tiles): the authors' tuned GEMM reaches 134 TOPS; this one is the check that the operand layout,
//                signedness and scale placement are right, and a floor for the bench.
// BUILD      hipcc -O2 --offload-arch=gfx1100 -Iinclude tools/pancake/probes/int4_wmma.hip.cpp -o /tmp/int4_wmma
//            (or cmake -DSTRATA_BUILD_PROBES=ON with HIP, target int4_wmma_probe)
// RUN        HIP_VISIBLE_DEVICES=0 /tmp/int4_wmma [--m 640 --k 2560 --n 512 --block 256 --a4 0|1 --reps 20 --parity-only]
//            exit 0 pass, 1 mismatch, 77 no gfx11 GPU.  Needs the card idle.  Shapes of the daily model: expert gate/up rows K = 2560,
//            expert down rows K = 640 = 5 x 128, so run `--k 640 --m 2560 --block 128` for the down projection (the authors need K % 256).
// READ       PARITY lines: device quantised activations against the host reference (identical but for float rounding of the rotation:
//            <= 0.5% of codes may differ by one) and the device GEMM against `gemm_ref` (relative error <= 2e-4).  BENCH line: ms and
//            TOPS counting each int4 pass (A8 = two passes), and the f16 WMMA peak of the card for the ratio (run wmma_rate for that).
// STATUS     syntax + device compile checked with the ROCm 6.2 SDK on the Windows PC; NEVER RUN (no gfx1100 GPU there).  Box run needed.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "strata/kernels/int4_rot.hpp"

#define CK(x)                                                                                        \
    do {                                                                                             \
        hipError_t e_ = (x);                                                                         \
        if (e_ != hipSuccess) {                                                                      \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));   \
            std::exit(1);                                                                            \
        }                                                                                            \
    } while (0)

using i2v = int __attribute__((ext_vector_type(2)));
using i8v = int __attribute__((ext_vector_type(8)));

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
#define INT4_GFX11 1
#endif

__device__ __forceinline__ float dsign(int j) { return (((unsigned) j * 2654435761u) >> 31) ? -1.0f : 1.0f; }

// grid (K / B, N), block B threads (B a power of two <= 1024): x' = R x on one block, same operation order as the host reference
__global__ void rot_kernel(const float* __restrict__ x, float* __restrict__ y, int K, int B) {
    extern __shared__ float s[];
    const int t = threadIdx.x;
    const size_t base = (size_t) blockIdx.y * K + (size_t) blockIdx.x * B;
    s[t] = x[base + t] * dsign(t);
    __syncthreads();
    for (int h = 1; h < B; h <<= 1) {
        const int j = (t / h) * 2 * h + (t % h);   // the t-th butterfly of this stage (B / 2 of them)
        if (t < B / 2) {
            const float a = s[j], b = s[j + h];
            s[j] = a + b;
            s[j + h] = a - b;
        }
        __syncthreads();
    }
    y[base + t] = s[t] * (1.0f / sqrtf((float) B));
}

// one block (256 threads) a token: amax of the token, amax and codes per B-block
__global__ void quant_kernel(const float* __restrict__ x, int8_t* __restrict__ hi, int8_t* __restrict__ lo, float* __restrict__ scale,
                             int K, int B, int a8, float clip) {
    __shared__ float red[256];
    const int n = blockIdx.x, t = threadIdx.x;
    const float* row = x + (size_t) n * K;
    float m = 0.f;
    for (int k = t; k < K; k += 256) m = fmaxf(m, fabsf(row[k]));
    red[t] = m;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (t < o) red[t] = fmaxf(red[t], red[t + o]);
        __syncthreads();
    }
    const float amax_tok = red[0];
    __syncthreads();
    for (int b = 0; b < K / B; ++b) {
        float mb = 0.f;
        for (int k = t; k < B; k += 256) mb = fmaxf(mb, fabsf(row[b * B + k]));
        red[t] = mb;
        __syncthreads();
        for (int o = 128; o > 0; o >>= 1) {
            if (t < o) red[t] = fmaxf(red[t], red[t + o]);
            __syncthreads();
        }
        const float amax = red[0];
        __syncthreads();
        float sc;
        if (a8) sc = amax / 127.0f;
        else {
            int mm = amax_tok > 0.f ? (int) ceilf(16.0f * clip * amax / amax_tok) : 1;
            mm = min(16, max(1, mm));
            sc = (float) mm * amax_tok / (16.0f * 7.0f);
        }
        if (t == 0) scale[(size_t) n * (K / B) + b] = sc;
        const float inv = sc > 0.f ? 1.0f / sc : 0.f;
        for (int k = t; k < B; k += 256) {
            const size_t i = (size_t) n * K + (size_t) b * B + k;
            const float v = row[b * B + k] * inv;
            if (a8) {
                const int q = min(127, max(-128, (int) lrintf(v)));
                hi[i] = (int8_t) (q >> 4);
                lo[i] = (int8_t) (q & 15);
            } else {
                hi[i] = (int8_t) min(7, max(-7, (int) lrintf(v)));
            }
        }
    }
}

// int8 codes (-8..15 range, low nibble kept) -> 16 nibbles = two words, element e of a slice at nibble e
__global__ void pack_kernel(const int8_t* __restrict__ in, unsigned* __restrict__ out, long long slices) {
    const long long s = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= slices) return;
    unsigned w[2] = {0u, 0u};
    for (int e = 0; e < 16; ++e) w[e >> 3] |= ((unsigned) in[s * 16 + e] & 15u) << (4 * (e & 7));
    out[s * 2] = w[0];
    out[s * 2 + 1] = w[1];
}

// grid (M / 16, N / 16), block 32: Y[n][m] = sum_g sw[m][g] * da[n][blk(g)] * acc(g), acc from iu4 WMMA (D[2i + l/16][l % 16])
template <bool A8>
__global__ void __launch_bounds__(32) gemm_kernel(const unsigned* __restrict__ wp, const float* __restrict__ ws,
                                                  const unsigned* __restrict__ hip_, const unsigned* __restrict__ lop,
                                                  const float* __restrict__ as, float* __restrict__ y, int M, int N, int K, int B) {
#if defined(INT4_GFX11)
    const int l = threadIdx.x, lr = l & 15;
    const int mrow = blockIdx.x * 16 + lr, ncol = blockIdx.y * 16 + lr;
    const int nsl = K / 16;
    float yacc[8];
    for (int i = 0; i < 8; ++i) yacc[i] = 0.f;
    for (int g = 0; g < K / 32; ++g) {
        i8v ah, al;
        for (int i = 0; i < 8; ++i) { ah[i] = 0; al[i] = 0; }
        for (int sl = 0; sl < 2; ++sl) {
            const int s = g * 2 + sl;
            const unsigned* wq = wp + ((size_t) mrow * nsl + s) * 2;
            const unsigned* aq = hip_ + ((size_t) ncol * nsl + s) * 2;
            const i2v wv = {(int) wq[0], (int) wq[1]};
            const i2v av = {(int) aq[0], (int) aq[1]};
            ah = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, wv, true, av, ah, false);
            if (A8) {
                const unsigned* lq = lop + ((size_t) ncol * nsl + s) * 2;
                const i2v lv = {(int) lq[0], (int) lq[1]};
                al = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, wv, false, lv, al, false);
            }
        }
        const float da = as[(size_t) ncol * (K / B) + (g * 32) / B];
        for (int i = 0; i < 8; ++i) {
            const int m = blockIdx.x * 16 + 2 * i + (l >> 4);
            const int v = A8 ? 16 * ah[i] + al[i] : ah[i];
            yacc[i] += ws[(size_t) m * (K / 32) + g] * da * (float) v;
        }
    }
    for (int i = 0; i < 8; ++i) y[(size_t) ncol * M + blockIdx.x * 16 + 2 * i + (l >> 4)] = yacc[i];
#else
    if (M < 0) y[0] = 0.f;
#endif
}

template <class T> static T* dev_copy(const std::vector<T>& v) {
    T* d;
    CK(hipMalloc(&d, v.size() * sizeof(T)));
    CK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    int M = 640, K = 2560, N = 512, B = 256, a4 = 0, reps = 20, device = 0;
    bool parity_only = false;
    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* k) { return !std::strcmp(argv[i], k) && i + 1 < argc; };
        if (arg("--m")) M = std::atoi(argv[++i]);
        else if (arg("--k")) K = std::atoi(argv[++i]);
        else if (arg("--n")) N = std::atoi(argv[++i]);
        else if (arg("--block")) B = std::atoi(argv[++i]);
        else if (arg("--a4")) a4 = std::atoi(argv[++i]);
        else if (arg("--reps")) reps = std::atoi(argv[++i]);
        else if (arg("--device")) device = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--parity-only")) parity_only = true;
        else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (M % 16 || N % 16 || K % 32 || K % B || (B & (B - 1)) || B > 1024 || B < 32) {
        std::fprintf(stderr, "int4_wmma: need M, N multiples of 16, K a multiple of 32 and of the block B (a power of two, 32..1024)\n");
        return 2;
    }
    hipDeviceProp_t p;
    if (hipSetDevice(device) != hipSuccess || hipGetDeviceProperties(&p, device) != hipSuccess ||
        (!std::strstr(p.gcnArchName, "gfx110") && !std::strstr(p.gcnArchName, "gfx115"))) {
        std::fprintf(stderr, "int4_wmma: no gfx11 GPU (device %d)\n", device);
        return 77;
    }
    std::printf("device %d %s, M=%d K=%d N=%d block=%d %s\n", device, p.gcnArchName, M, K, N, B, a4 ? "A4" : "A8");
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> w((size_t) M * K), x((size_t) N * K);
    for (auto& v : w) v = nd(rng);
    for (auto& v : x) v = nd(rng);
    for (int n = 0; n < N; ++n) x[(size_t) n * K + (n * 7) % K] *= 20.0f;
    using namespace strata::kernels::int4rot;
    // host reference: rotate both operands, quantise, integer GEMM
    auto wr = w, xr = x;
    rotate_rows(wr, M, K, B);
    rotate_rows(xr, N, K, B);
    const WeightQ wq = quantize_weights_rtn(wr, M, K);
    const ActQ aq = quantize_acts(xr, N, K, B, !a4);
    // device pipeline: rotate x, quantise, pack, gemm (weights are packed from the host codes)
    float *d_x = dev_copy(x), *d_xr;
    CK(hipMalloc(&d_xr, x.size() * sizeof(float)));
    rot_kernel<<<dim3(K / B, N), B, B * sizeof(float)>>>(d_x, d_xr, K, B);
    int8_t *d_hi, *d_lo;
    float* d_as;
    CK(hipMalloc(&d_hi, x.size()));
    CK(hipMalloc(&d_lo, x.size()));
    CK(hipMalloc(&d_as, (size_t) N * (K / B) * sizeof(float)));
    CK(hipMemset(d_lo, 0, x.size()));
    quant_kernel<<<N, 256>>>(d_xr, d_hi, d_lo, d_as, K, B, a4 ? 0 : 1, 0.85f);
    CK(hipDeviceSynchronize());
    std::vector<int8_t> hi_dev(x.size()), lo_dev(x.size());
    CK(hipMemcpy(hi_dev.data(), d_hi, x.size(), hipMemcpyDeviceToHost));
    CK(hipMemcpy(lo_dev.data(), d_lo, x.size(), hipMemcpyDeviceToHost));
    size_t diff = 0;
    bool big = false;
    for (size_t i = 0; i < x.size(); ++i) {
        const int a = a4 ? hi_dev[i] : 16 * hi_dev[i] + lo_dev[i], b = a4 ? aq.hi[i] : 16 * aq.hi[i] + aq.lo[i];
        if (a != b) { ++diff; if (std::abs(a - b) > 1) big = true; }
    }
    const double frac = (double) diff / (double) x.size();
    std::printf("PARITY activations: %zu of %zu codes differ (%.4f%%), %s\n", diff, x.size(), 100 * frac, big ? "some by more than 1" : "all by 1");
    unsigned *d_hip, *d_lop, *d_wp;
    const long long xs = (long long) N * K / 16, ws_ = (long long) M * K / 16;
    CK(hipMalloc(&d_hip, xs * 8));
    CK(hipMalloc(&d_lop, xs * 8));
    CK(hipMalloc(&d_wp, ws_ * 8));
    int8_t* d_wq = nullptr;
    CK(hipMalloc(&d_wq, (size_t) M * K));
    CK(hipMemcpy(d_wq, wq.q.data(), (size_t) M * K, hipMemcpyHostToDevice));
    pack_kernel<<<(unsigned) ((xs + 255) / 256), 256>>>(d_hi, d_hip, xs);
    pack_kernel<<<(unsigned) ((xs + 255) / 256), 256>>>(d_lo, d_lop, xs);
    pack_kernel<<<(unsigned) ((ws_ + 255) / 256), 256>>>(d_wq, d_wp, ws_);
    float* d_ws = dev_copy(wq.scale);
    float* d_y;
    CK(hipMalloc(&d_y, (size_t) N * M * sizeof(float)));
    auto run = [&] {
        if (a4) gemm_kernel<false><<<dim3(M / 16, N / 16), 32>>>(d_wp, d_ws, d_hip, d_lop, d_as, d_y, M, N, K, B);
        else gemm_kernel<true><<<dim3(M / 16, N / 16), 32>>>(d_wp, d_ws, d_hip, d_lop, d_as, d_y, M, N, K, B);
    };
    run();
    CK(hipDeviceSynchronize());
    std::vector<float> y((size_t) N * M);
    CK(hipMemcpy(y.data(), d_y, y.size() * sizeof(float), hipMemcpyDeviceToHost));
    // the GEMM is checked against the reference on the DEVICE's own codes (isolates the matrix pass from the rotation rounding)
    ActQ ad = aq;
    ad.hi = hi_dev;
    if (!a4) ad.lo = lo_dev;
    std::vector<float> asd((size_t) N * (K / B));
    CK(hipMemcpy(asd.data(), d_as, asd.size() * sizeof(float), hipMemcpyDeviceToHost));
    ad.scale = asd;
    const std::vector<float> ref = gemm_ref(wq, ad);
    double dn = 0, nn = 0;
    for (size_t i = 0; i < y.size(); ++i) { dn += (double) (y[i] - ref[i]) * (y[i] - ref[i]); nn += (double) ref[i] * ref[i]; }
    const double rel = nn > 0 ? std::sqrt(dn / nn) : 0.0;
    std::printf("PARITY gemm: relative error vs the host integer pipeline on the same codes %.3e\n", rel);
    const bool ok = rel <= 2e-4 && frac <= 0.005 && !big;
    std::printf("%s\n", ok ? "int4_wmma: PARITY PASS" : "int4_wmma: PARITY FAIL");
    if (!parity_only && ok) {
        hipEvent_t e0, e1;
        CK(hipEventCreate(&e0));
        CK(hipEventCreate(&e1));
        std::vector<float> ms;
        for (int r = 0; r < reps + 1; ++r) {
            CK(hipEventRecord(e0));
            run();
            CK(hipEventRecord(e1));
            CK(hipEventSynchronize(e1));
            float t;
            CK(hipEventElapsedTime(&t, e0, e1));
            if (r) ms.push_back(t);   // run 1 discarded
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        const double passes = a4 ? 1.0 : 2.0;
        std::printf("BENCH gemm: median %.4f ms, best %.4f ms, %.1f TOPS counting %g int4 pass(es) (2 M N K ops each); authors' tuned GEMM 134 TOPS\n",
                    med, (double) ms.front(), 2.0 * M * N * K * passes / (med * 1e-3) / 1e12, passes);
    }
    return ok ? 0 : 1;
}
