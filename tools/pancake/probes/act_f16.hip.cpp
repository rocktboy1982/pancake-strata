// act_f16.hip.cpp - probe (LEADS.md Part O O10, STRATA_ACT_F16): what does multiplying 2-bit-class weights by FP16 activations cost against the read-only
// roofline of the same bytes on the 7900 XT?
//
// The caminus authors put the cost of fp16 activations (instead of Q8_1 int8 ones) at no more than 0.08 ms a token on their card.  Our decode streams its
// weights at a fraction of the DRAM peak and its IQ2 kernels are partly decode-bound (MOD_PLAN.md section 1), so the number to look at is how close a
// dequantise-and-multiply-add kernel with f16 activations gets to a kernel that only reads and xors the same words.  The weights are a synthetic 2-bit
// layout (16 values a u32, 4 levels, a scale per 32 values in a separate fp16 array): NOT our IQ2_XS format (its grid lookups cost more), so this is a
// floor for the arithmetic, not a model of the real tile.  The real tile is the future STRATA_ACT_F16 kernel (large: LEADS.md O10).
// BUILD      hipcc -O2 --offload-arch=gfx1100 tools/pancake/probes/act_f16.hip.cpp -o /tmp/act_f16
// RUN        HIP_VISIBLE_DEVICES=0 /tmp/act_f16 [--rows 4096] [--k 2560] [--experts 64] [--reps 200] [--device 0]       (engine stopped; seconds)
// READ       "read-only", "f16-act" and "q8-act" lines: us per call, GB/s of weight bytes, and the f16 / q8 ratio to read-only.  q8-act is the same kernel with the
//            activation rounded to int8 in registers (a stand-in for the Q8_1 path's rounding, same bytes, same loads, integer multiply-adds).  Decision rule:
//            f16-act within ~10% of read-only means the arithmetic is hidden by the loads and the quality hypothesis (host reference: act_f16_test) is the
//            only open question; much slower means the tile needs the packed dot (v_dot2_f32_f16) before it is worth a gate run.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CK(x)                                                                                      \
    do {                                                                                           \
        hipError_t e_ = (x);                                                                       \
        if (e_ != hipSuccess) {                                                                    \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

constexpr int kThreads = 256;

// one row per 32-thread group (wave32): each lane reads K/16/32 words strided by 32; result per row
__global__ void __launch_bounds__(kThreads) read_only_k(const uint32_t* __restrict__ w, int words_per_row, uint32_t* out, int rows) {
    const int row = blockIdx.x * (kThreads / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= rows) return;
    const uint32_t* p = w + size_t(row) * words_per_row;
    uint32_t acc = 0;
    for (int i = lane; i < words_per_row; i += 32) acc ^= p[i];
    for (int o = 16; o > 0; o >>= 1) acc ^= __shfl_xor(acc, o);
    if (lane == 0) out[row] = acc;
}

template <int MODE>   // 0: f16 activations, 1: int8-rounded activations (per 32-block scale, as Q8_1)
__global__ void __launch_bounds__(kThreads) mv_k(const uint32_t* __restrict__ w, const __half* __restrict__ scale, const __half* __restrict__ x, int k,
                                                  float* out, int rows) {
    const int words_per_row = k / 16;
    const int row = blockIdx.x * (kThreads / 32) + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= rows) return;
    const uint32_t* p = w + size_t(row) * words_per_row;
    const __half* sc = scale + size_t(row) * (k / 32);
    float acc = 0.0f;
    for (int i = lane; i < words_per_row; i += 32) {
        const uint32_t word = p[i];
        const float s = __half2float(sc[i / 2]);   // one scale per 32 values = 2 words
        float part = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const float lvl = float((word >> (2 * j)) & 3u) - 1.5f;
            float xv = __half2float(x[i * 16 + j]);
            if (MODE == 1) xv = float(int(__float2int_rn(xv * 64.0f))) * (1.0f / 64.0f);   // int8-ish rounding stand-in
            part = fmaf(lvl, xv, part);
        }
        acc = fmaf(s, part, acc);
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor(acc, o);
    if (lane == 0) out[row] = acc;
}

int main(int argc, char** argv) {
    int rows = 4096, k = 2560, experts = 64, reps = 200, dev = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--rows") rows = std::atoi(argv[i + 1]);
        else if (a == "--k") k = std::atoi(argv[i + 1]);
        else if (a == "--experts") experts = std::atoi(argv[i + 1]);
        else if (a == "--reps") reps = std::atoi(argv[i + 1]);
        else if (a == "--device") dev = std::atoi(argv[i + 1]);
    }
    CK(hipSetDevice(dev));
    const size_t wwords = size_t(rows) * (k / 16), bytes = wwords * 4;   // one expert's weights; experts rotate so every call reads cold from DRAM
    std::vector<uint32_t> hw(wwords);
    for (size_t i = 0; i < wwords; ++i) hw[i] = uint32_t(i * 2654435761u + 1u);
    std::vector<__half> hs(size_t(rows) * (k / 32), __float2half(0.05f));
    std::vector<__half> hx(size_t(k), __float2half(0.0f));
    for (int i = 0; i < k; ++i) hx[size_t(i)] = __float2half(float((i * 37) % 201 - 100) / 50.0f);
    uint32_t* dw;
    __half *ds, *dx;
    float* dout;
    uint32_t* dro;
    CK(hipMalloc(&dw, bytes * experts));
    for (int e = 0; e < experts; ++e) CK(hipMemcpy(dw + size_t(e) * wwords, hw.data(), bytes, hipMemcpyHostToDevice));
    CK(hipMalloc(&ds, hs.size() * 2));
    CK(hipMalloc(&dx, hx.size() * 2));
    CK(hipMalloc(&dout, size_t(rows) * 4));
    CK(hipMalloc(&dro, size_t(rows) * 4));
    CK(hipMemcpy(ds, hs.data(), hs.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    const dim3 grid((rows + kThreads / 32 - 1) / (kThreads / 32)), block(kThreads);
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    auto time_arm = [&](int arm) {
        std::vector<float> ms;
        for (int round = 0; round < 5; ++round) {
            CK(hipEventRecord(e0, nullptr));
            for (int r = 0; r < reps; ++r) {
                const uint32_t* w = dw + size_t(r % experts) * wwords;
                if (arm == 0) hipLaunchKernelGGL(read_only_k, grid, block, 0, nullptr, w, int(k / 16), dro, rows);
                else if (arm == 1) hipLaunchKernelGGL(mv_k<0>, grid, block, 0, nullptr, w, ds, dx, k, dout, rows);
                else hipLaunchKernelGGL(mv_k<1>, grid, block, 0, nullptr, w, ds, dx, k, dout, rows);
            }
            CK(hipEventRecord(e1, nullptr));
            CK(hipEventSynchronize(e1));
            float t = 0;
            CK(hipEventElapsedTime(&t, e0, e1));
            ms.push_back(t);
        }
        std::sort(ms.begin(), ms.end());
        return ms[2] * 1000.0 / reps;   // median us per call
    };
    time_arm(0);   // warm
    const double us_ro = time_arm(0), us_f16 = time_arm(1), us_q8 = time_arm(2);
    auto line = [&](const char* n, double us) {
        std::printf("%-9s %9.2f us  %7.1f GB/s  %5.2fx of read-only\n", n, us, double(bytes) / (us * 1e-6) / 1e9, us / us_ro);
    };
    std::printf("# rows %d, k %d, %.2f MB of weights a call, %d experts rotating\n", rows, k, double(bytes) / 1e6, experts);
    line("read-only", us_ro);
    line("f16-act", us_f16);
    line("q8-act", us_q8);
    std::printf("f16-act minus q8-act: %.2f us a call\n", us_f16 - us_q8);
    return 0;
}
