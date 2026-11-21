// reduce_f32_parity.hip.cpp - probe (LEADS.md Part U, lead U2, spec 277): the device run of tools/pancake/reduce_f32_model.py, with the DEVICE expf / sqrt / divide.
//
// QUESTION   STRATA_REDUCE_F32=1 (include/strata/kernels/reduce_f32.hpp) replaces the f64 form of silu and of the sums of squares by f32.  The host model
//            (numpy) cannot know the device expf.  This probe runs the same comparison on the card: rf32::silu against the engine's f64 original
//            `(float)(v / (1.0 + exp(-v)))` over EVERY finite f32 bit pattern (2^32, in chunks), counting f32 mismatches, the largest ulp distance, and the
//            mismatches after a bf16 and an f16 store; the same over the bf16-valued and the f16-valued inputs alone; then rows of random and worst-case
//            values through the warp sum of squares of gdn_l2_kernel (f64 against f32, the loop of gdn.cu copied here), max relative error of the scale
//            and the ulp distance of the output elements.
// READ       `exact` is claimed ONLY for a line whose mismatch counts are 0 over the whole input class; any other number is the bound.  A real run also settles
//            the unverified FP64 rate of gfx11 (U.12): the time of the f64 sweep against the f32 sweep is printed.
// BUILD      hipcc -O2 --offload-arch=gfx1100 -I include tools/pancake/probes/reduce_f32_parity.hip.cpp -o /tmp/reduce_f32_parity
// RUN        HIP_VISIBLE_DEVICES=0 /tmp/reduce_f32_parity [--chunks 16] [--rows 4096] [--device 0]   (GPU idle; the full sweep is seconds to a minute;
//            --chunks N runs the first N of the 16 chunks of 2^28 patterns, 16 = all)
// NOT        an engine switch and not an A/B.  NOT RUN when written (the Windows PC has no 7900 XT): a device compile and the host model only.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "strata/kernels/reduce_f32.hpp"

namespace rf = strata::kernels::rf32;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_)); std::exit(1); } } while (0)

__device__ __forceinline__ int64_t ordered(float f) {
    const uint32_t u = __float_as_uint(f);
    return u >= 0x80000000u ? -(int64_t) (u - 0x80000000u) : (int64_t) u;
}
__device__ __forceinline__ uint32_t bf16_rn(float f) {
    const uint32_t u = __float_as_uint(f);
    return (uint32_t) (((uint64_t) u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ uint32_t f16_bits(float f) { return (uint32_t) __half_as_ushort(__float2half_rn(f)); }

struct SiluCounts { unsigned long long n, f32_diff, bf16_diff, f16_diff, max_ulp; };

// mode 0: bit patterns base .. base + n; mode 1: every bf16 value (pattern << 16); mode 2: every f16 value
__global__ void silu_sweep(uint64_t base, uint64_t n, int mode, SiluCounts* out) {
    const uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x;
    if (mode == 0) x = __uint_as_float((uint32_t) (base + i));
    else if (mode == 1) x = __uint_as_float((uint32_t) (i << 16));
    else x = __half2float(__ushort_as_half((unsigned short) i));
    if (!isfinite(x)) return;
    const double v = (double) x;
    const float ref = (float) (v / (1.0 + exp(-v)));   // the engine's original (elementwise.cu silu_kernel, F32 = false)
    const float got = rf::silu(x);                      // STRATA_REDUCE_F32=1
    atomicAdd(&out->n, 1ull);
    if (ref != got) {
        atomicAdd(&out->f32_diff, 1ull);
        const int64_t d = ordered(ref) - ordered(got);
        atomicMax(&out->max_ulp, (unsigned long long) (d < 0 ? -d : d));
    }
    if (bf16_rn(ref) != bf16_rn(got)) atomicAdd(&out->bf16_diff, 1ull);
    if (f16_bits(ref) != f16_bits(got)) atomicAdd(&out->f16_diff, 1ull);
}

// ---- gdn_l2_kernel, both forms, one warp a row (the loop of src/kernels/cuda/gdn.cu, copied)
template <bool F32>
__device__ float l2_scale(const float* p, int cols, float eps) {
    using Acc = rf::acc_t<F32>;
    Acc acc = 0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        if constexpr (F32) acc = rf::fadd(acc, rf::fmul(p[i], p[i]));
        else acc += (double) p[i] * (double) p[i];
    }
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down(acc, off);
    __shared__ Acc ssum;
    if (threadIdx.x == 0) ssum = acc;
    __syncthreads();
    float inv;
    if constexpr (F32) inv = rf::inv_sqrt(rf::fadd(ssum, eps));
    else inv = (float) (1.0 / sqrt(ssum + (double) eps));
    __syncthreads();
    return inv;
}

__device__ uint32_t xs(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }

struct NormCounts { double max_rel; unsigned long long max_ulp, out_diff, out_n; };

__global__ void l2_rows(int rows, int cols, NormCounts* out, float* scratch) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    float* row = scratch + (size_t) r * cols;
    uint32_t s = 0x9E3779B9u * (uint32_t) (r + 1);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float u = (float) (xs(s) & 0xFFFFFFu) / 16777216.0f - 0.5f;
        const int kind = r % 4;   // 0 uniform, 1 wide dynamic range, 2 one large + small, 3 constant
        row[i] = kind == 0 ? u * 8.0f : kind == 1 ? u * exp2f((float) ((int) (xs(s) % 32u) - 16)) : kind == 2 ? (i == 0 ? 1e4f : 1e-3f * u) : 1.0f;
    }
    __syncthreads();
    const float a = l2_scale<false>(row, cols, 1e-6f);
    const float b = l2_scale<true>(row, cols, 1e-6f);
    if (threadIdx.x == 0) {
        const double rel = fabs((double) b - (double) a) / fabs((double) a);
        // atomicMax on the bit pattern of a non-negative double
        atomicMax((unsigned long long*) &out->max_rel, (unsigned long long) __double_as_longlong(rel));
    }
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float ya = row[i] * a, yb = row[i] * b;
        atomicAdd(&out->out_n, 1ull);
        if (ya != yb) {
            atomicAdd(&out->out_diff, 1ull);
            const int64_t d = ordered(ya) - ordered(yb);
            atomicMax(&out->max_ulp, (unsigned long long) (d < 0 ? -d : d));
        }
    }
}

int main(int argc, char** argv) {
    int chunks = 16, rows = 4096, dev = 0;
    for (int i = 1; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--chunks")) chunks = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--rows")) rows = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--device")) dev = std::atoi(argv[i + 1]);
    }
    CK(hipSetDevice(dev));
    hipDeviceProp_t prop;
    CK(hipGetDeviceProperties(&prop, dev));
    std::printf("device %s %s\n", prop.name, prop.gcnArchName);
    SiluCounts* d;
    CK(hipMalloc(&d, sizeof(SiluCounts)));
    auto run = [&](const char* name, int mode, uint64_t base, uint64_t n) {
        SiluCounts z{};
        CK(hipMemcpy(d, &z, sizeof z, hipMemcpyHostToDevice));
        silu_sweep<<<(unsigned) ((n + 255) / 256), 256>>>(base, n, mode, d);
        CK(hipGetLastError());
        CK(hipDeviceSynchronize());
        SiluCounts h;
        CK(hipMemcpy(&h, d, sizeof h, hipMemcpyDeviceToHost));
        std::printf("silu %-26s n %llu  f32-diff %llu  max-ulp %llu  bf16-store-diff %llu  f16-store-diff %llu  %s\n", name, h.n, h.f32_diff, h.max_ulp, h.bf16_diff,
                    h.f16_diff, (h.bf16_diff == 0 && h.f16_diff == 0) ? "(store-exact on this class)" : "(NOT exact after the store)");
        return h;
    };
    run("bf16 values", 1, 0, 65536);
    run("f16 values", 2, 0, 65536);
    hipEvent_t e0, e1;
    CK(hipEventCreate(&e0));
    CK(hipEventCreate(&e1));
    CK(hipEventRecord(e0));
    unsigned long long tot = 0, f32d = 0, bf = 0, f16 = 0, mx = 0;
    for (int c = 0; c < chunks; ++c) {
        const SiluCounts h = run("f32 patterns (chunk)", 0, (uint64_t) c << 28, 1ull << 28);
        tot += h.n; f32d += h.f32_diff; bf += h.bf16_diff; f16 += h.f16_diff; mx = h.max_ulp > mx ? h.max_ulp : mx;
    }
    CK(hipEventRecord(e1));
    CK(hipEventSynchronize(e1));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, e0, e1));
    std::printf("silu f32 patterns TOTAL n %llu  f32-diff %llu  max-ulp %llu  bf16-store-diff %llu  f16-store-diff %llu  (%.1f ms for both forms)\n", tot, f32d, mx, bf, f16, ms);
    NormCounts* nd;
    float* scratch;
    CK(hipMalloc(&nd, sizeof(NormCounts)));
    CK(hipMalloc(&scratch, (size_t) rows * 128 * sizeof(float)));
    NormCounts z{};
    CK(hipMemcpy(nd, &z, sizeof z, hipMemcpyHostToDevice));
    l2_rows<<<rows, 32>>>(rows, 128, nd, scratch);
    CK(hipGetLastError());
    CK(hipDeviceSynchronize());
    NormCounts h;
    CK(hipMemcpy(&h, nd, sizeof h, hipMemcpyDeviceToHost));
    const double rel = h.max_rel;   // the bits atomicMax kept are those of a non-negative double
    std::printf("gdn_l2 rows %d x 128: max rel err of the scale %.3e  output elements differing %llu of %llu  max-ulp %llu  (NOT exact: a sum order / precision change)\n", rows, rel,
                h.out_diff, h.out_n, h.max_ulp);
    return 0;
}
