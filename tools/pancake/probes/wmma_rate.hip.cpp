// wmma_rate.hip.cpp - probe (LEADS.md Part L, lead L2): what rate do the gfx11 WMMA instructions really reach on the 7900 XT?
//
// QUESTION   SILVERTHRONE32/llama.cpp-rdna3-int4 (MIT) builds its whole prompt path on v_wmma_i32_16x16x16_iu4 and
//            reports, for the card, a dense peak of 206 TOPS for INT4 against 103 for INT8 and 103 TFLOPS for FP16/BF16,
//            a pure-WMMA loop at 182 TOPS (INT4) and its own GEMM at 134 TOPS effective, against 36-40 TOPS for
//            llama.cpp's MMQ (iu8) on the same card.  LEADS.md says the opposite for iu8 ("runs at the FP16 WMMA rate",
//            so int8 saves bytes, not math).  Both can be true: iu4 is the only integer path with a 2x.  This probe
//            measures all three on OUR card, so the iu4 leads of Part L have a number before anyone writes a kernel.
// WHAT       Each wave runs 8 independent accumulator chains of one WMMA instruction (f16 -> f32, iu8 -> i32, iu4 -> i32)
//            in a loop; operands live in registers (no memory traffic); the result is stored so nothing is removed.
//            TOPS = waves x iterations x 8 chains x 8192 MACs x 2 / time.  One 16x16x16 tile = 4096 MACs = 8192 ops;
//            8 chains = 65,536 ops per loop trip per wave.
// BUILD      hipcc -O2 --offload-arch=gfx1100 tools/pancake/probes/wmma_rate.hip.cpp -o /tmp/wmma_rate
//            (or cmake -DSTRATA_BUILD_PROBES=ON with HIP, target wmma_rate_probe)
// RUN        HIP_VISIBLE_DEVICES=0 /tmp/wmma_rate [--iters 20000] [--reps 7] [--blocks-per-cu 4] [--device 0]
//            Needs the GPU idle (stop the engine first); takes a few seconds.
// READ       stdout: per mode the best and median TOPS of the reps, then ratios iu8/f16 and iu4/iu8, and the card's reported
//            clock.  Decision rule for LEADS.md Part L: iu4/iu8 >= 1.8 means the 2x is real on this card, so an iu4 expert
//            kernel has a compute ceiling above the f16 path (PF_F16); a ratio near 1 closes L1/L3 for compute.
//            The probe says nothing about weights: an IQ2_XS code does not fit iu4 (see LEADS.md Part L).
// NOT        an A/B and not an engine switch: it never touches the engine.  Box run needed: nothing was run when this was
//            written (the Windows PC has no 7900 XT); a syntax check only.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x)                                                                                        \
    do {                                                                                             \
        hipError_t e_ = (x);                                                                         \
        if (e_ != hipSuccess) {                                                                      \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));   \
            std::exit(1);                                                                            \
        }                                                                                            \
    } while (0)

using h16v = _Float16 __attribute__((ext_vector_type(16)));
using f8v = float __attribute__((ext_vector_type(8)));
using i2v = int __attribute__((ext_vector_type(2)));
using i4v = int __attribute__((ext_vector_type(4)));
using i8v = int __attribute__((ext_vector_type(8)));

constexpr int kChains = 8;
constexpr int kThreads = 128;           // 4 waves of 32
constexpr double kOpsPerTrip = 8192.0 * kChains;   // per wave

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
#define RATE_GFX11 1
#endif

template <int MODE>   // 0 f16, 1 iu8, 2 iu4
__global__ void __launch_bounds__(kThreads) rate_kernel(int iters, int* __restrict__ out) {
#if defined(RATE_GFX11)
    const int t = threadIdx.x + 1;
    if (MODE == 0) {
        h16v a, b;
        for (int i = 0; i < 16; ++i) { a[i] = (_Float16) ((t + i) & 7); b[i] = (_Float16) ((t * 3 + i) & 7); }
        f8v acc[kChains];
        for (int c = 0; c < kChains; ++c) for (int i = 0; i < 8; ++i) acc[c][i] = 0.f;
#pragma unroll 1
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int c = 0; c < kChains; ++c) acc[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[c]);
        }
        float s = 0.f;
        for (int c = 0; c < kChains; ++c) for (int i = 0; i < 8; ++i) s += acc[c][i];
        if (s == 123456.789f) out[blockIdx.x] = 1;   // practically never; keeps the loop alive
    } else if (MODE == 1) {
        i4v a, b;
        for (int i = 0; i < 4; ++i) { a[i] = t * 0x01010101 + i; b[i] = t * 0x03030303 + i; }
        i8v acc[kChains];
        for (int c = 0; c < kChains; ++c) for (int i = 0; i < 8; ++i) acc[c][i] = 0;
#pragma unroll 1
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int c = 0; c < kChains; ++c) acc[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, acc[c], false);
        }
        int s = 0;
        for (int c = 0; c < kChains; ++c) for (int i = 0; i < 8; ++i) s ^= acc[c][i];
        if (s == 0x7a5a5a5a) out[blockIdx.x] = 1;
    } else {
        i2v a, b;
        for (int i = 0; i < 2; ++i) { a[i] = t * 0x01010101 + i; b[i] = t * 0x03030303 + i; }
        i8v acc[kChains];
        for (int c = 0; c < kChains; ++c) for (int i = 0; i < 8; ++i) acc[c][i] = 0;
#pragma unroll 1
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int c = 0; c < kChains; ++c) acc[c] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, a, true, b, acc[c], false);
        }
        int s = 0;
        for (int c = 0; c < kChains; ++c) for (int i = 0; i < 8; ++i) s ^= acc[c][i];
        if (s == 0x7a5a5a5a) out[blockIdx.x] = 1;
    }
#else
    // other architectures: nothing to measure (the host prints the refusal)
    if (iters < 0) out[0] = MODE;
#endif
}

struct Row {
    double best = 0, med = 0;
};

template <int MODE>
static Row run(int blocks, int iters, int reps, int* d_out) {
    hipEvent_t a, b;
    CK(hipEventCreate(&a));
    CK(hipEventCreate(&b));
    rate_kernel<MODE><<<blocks, kThreads>>>(iters / 10 + 1, d_out);   // warm up: clocks and code load
    CK(hipDeviceSynchronize());
    std::vector<double> v;
    for (int r = 0; r < reps; ++r) {
        CK(hipEventRecord(a));
        rate_kernel<MODE><<<blocks, kThreads>>>(iters, d_out);
        CK(hipEventRecord(b));
        CK(hipEventSynchronize(b));
        float ms;
        CK(hipEventElapsedTime(&ms, a, b));
        const double waves = (double) blocks * (kThreads / 32);
        v.push_back(waves * (double) iters * kOpsPerTrip / (ms * 1e-3) / 1e12);
    }
    std::sort(v.begin(), v.end());
    Row row;
    row.best = v.back();
    row.med = v[v.size() / 2];
    CK(hipEventDestroy(a));
    CK(hipEventDestroy(b));
    return row;
}

int main(int argc, char** argv) {
    int device = 0, iters = 20000, reps = 7, bpc = 4;
    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* k) { return !std::strcmp(argv[i], k) && i + 1 < argc; };
        if (arg("--device")) device = std::atoi(argv[++i]);
        else if (arg("--iters")) iters = std::atoi(argv[++i]);
        else if (arg("--reps")) reps = std::max(1, std::atoi(argv[++i]));
        else if (arg("--blocks-per-cu")) bpc = std::max(1, std::atoi(argv[++i]));
        else {
            std::fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 2;
        }
    }
    CK(hipSetDevice(device));
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, device));
    std::printf("device %d: %s (%s), %d CUs, clock %d MHz\n", device, p.name, p.gcnArchName, p.multiProcessorCount,
                p.clockRate / 1000);
    if (std::strstr(p.gcnArchName, "gfx110") == nullptr && std::strstr(p.gcnArchName, "gfx115") == nullptr) {
        std::fprintf(stderr, "wmma_rate: this probe measures the gfx11 wave32 WMMA instructions (gfx110x / gfx115x); %s is not one\n",
                     p.gcnArchName);
        return 3;
    }
    int* d_out;
    CK(hipMalloc(&d_out, 65536 * sizeof(int)));
    CK(hipMemset(d_out, 0, 65536 * sizeof(int)));
    const int blocks = p.multiProcessorCount * bpc;
    const Row f16 = run<0>(blocks, iters, reps, d_out);
    const Row i8 = run<1>(blocks, iters, reps, d_out);
    const Row i4 = run<2>(blocks, iters, reps, d_out);
    std::printf("mode      best TOPS   median TOPS   (blocks %d x %d threads, %d iterations x %d chains, %d reps)\n", blocks,
                kThreads, iters, kChains, reps);
    std::printf("f16/f32   %9.1f   %11.1f\n", f16.best, f16.med);
    std::printf("iu8/i32   %9.1f   %11.1f\n", i8.best, i8.med);
    std::printf("iu4/i32   %9.1f   %11.1f\n", i4.best, i4.med);
    std::printf("ratio iu8/f16 = %.2f   iu4/iu8 = %.2f   iu4/f16 = %.2f   (median)\n", i8.med / f16.med, i4.med / i8.med,
                i4.med / f16.med);
    std::printf("author's card (RX 7900 XT): peak 103 / 103 / 206, pure-WMMA loop 182 TOPS iu4, MMQ iu8 36-40 TOPS\n");
    std::printf("verdict rule: iu4/iu8 >= 1.8 -> the 2x is real here (LEADS.md Part L L1/L3 keep a compute ceiling)\n");
    CK(hipFree(d_out));
    return 0;
}
