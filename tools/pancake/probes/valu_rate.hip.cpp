// valu_rate.hip.cpp - probe (LEADS.md Part U, lead U4): instruction rates of gfx11 (RX 7900 XT) per CU, measured, not inferred.
//
// QUESTION   Nobody has measured the rate of v_dot4_i32_iu8 (sudot4) on our card: the radiance-rdna3 journal infers "dot4 is
//            reduced rate" from an end-to-end slowdown (W42m), its W46 gives 2.4 effective cycles per VALU instruction for one
//            loop, and its README says the opposite about int8 dot.  Our int8 decode kernels (rdna_kq 9.0%, the expert decode
//            kernels 6.8% + 5.8% of decode kernel time, T-029) are sudot4-based and judged bandwidth-bound only by the 85%
//            figure.  This probe gives the denominators: instructions per clock per CU for each instruction class.
// WHAT       One kernel per mode.  Every thread runs kChains = 8 independent dependency chains of ONE instruction class in a loop
//            (`#pragma unroll 1` over iterations, 8 instructions per trip, operands in registers, no memory in the VALU modes),
//            the result is stored behind an impossible test so nothing is removed.  The kernel reads the shader clock
//            (clock64) at its start and end in thread 0 of each block: per-block cycles, so the rate is per CLOCK, not per
//            nominal MHz (the card's real clock under load is 1.98-1.99 GHz, T-1006).  Grid = CUs x waves-per-SIMD blocks of
//            64 threads (2 waves), i.e. W waves per SIMD at occupancy W; the default sweep is 4 / 8 / 12 waves per SIMD.
//   modes    fma      v_fma_f32  (inline asm: never paired into a VOPD)
//            add      v_add_f32  (inline asm)
//            mul      v_mul_f32  (inline asm)
//            vopd     C++ fma + mul chains the compiler may pair as v_dual_* (the result depends on clang's pairing: the ISA
//                     check script counts the v_dual_ lines of the loop; a rate near the fma rate means no pairing)
//            dot2     v_dot2_f32_f16 (__builtin_amdgcn_fdot2)
//            dot4     v_dot4_i32_iu8 (__builtin_amdgcn_sudot4, both sign flags: the idiom of rdna_mmvq.cu / rdna_gr.cu)
//            mad24    v_mad_i32_i24 (__mul24 + add on a dependent chain)
//            perm     v_perm_b32 (__builtin_amdgcn_perm)
//            lds_r32  ds_read_b32  (volatile LDS reads, conflict-free: lane-linear)
//            lds_r128 ds_read_b128 (the same, 16 B a lane)
//            lds_w32  ds_write_b32, lds_w128 ds_write_b128
//            wmma_f16 v_wmma_f32_16x16x16_f16 (the f16 row of wmma_rate.hip.cpp; its CLI and its iu8 / iu4 rows are unchanged
//                     there, run that probe for them)
// BUILD      hipcc -O2 --offload-arch=gfx1100 tools/pancake/probes/valu_rate.hip.cpp -o valu_rate
//            (or cmake -DSTRATA_BUILD_PROBES=ON with HIP, target valu_rate_probe).  Check the instructions first (PC or box, no GPU):
//            hipcc -O2 --offload-arch=gfx1100 -S --cuda-device-only tools/pancake/probes/valu_rate.hip.cpp -o valu_rate.s
//            python3 tools/pancake/probes/valu_rate_report.py --isa valu_rate.s
// RUN        valu_rate [--device 0] [--iters 20000] [--reps 5] [--waves-per-simd 4,8,12] [--modes fma,add,...] [--cus N] [--flush-mib 512]
//            The GPU must be idle (stop the engine and every bench first); about 2 minutes with the defaults.
//            tools/pancake/box_chains/valu-rate-chainV1.sh builds and runs it on the box (idle-only).
// READ       lines `RESULT key=value ...` (one per mode and occupancy) and `DEVICE ...`; feed the stdout to
//            python3 tools/pancake/probes/valu_rate_report.py <stdout file>  (ops per clock per CU, % of theoretical on 84 CUs).
// NOT        an A/B and not an engine switch: never touches the engine.  Nothing was run when this was written (the Windows PC has
//            no gfx1100 GPU and no ROCm); the CPU-side parts (parser, report, clock arithmetic) have a fixture test.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                                        \
    do {                                                                                             \
        hipError_t e_ = (x);                                                                         \
        if (e_ != hipSuccess) {                                                                      \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));   \
            std::exit(1);                                                                            \
        }                                                                                            \
    } while (0)

constexpr int kChains = 8;
constexpr int kThreads = 64;   // 2 waves of 32

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
#define RATE_GFX11 1
#endif

using h2v = _Float16 __attribute__((ext_vector_type(2)));
using h16v = _Float16 __attribute__((ext_vector_type(16)));
using f8v = float __attribute__((ext_vector_type(8)));

enum Mode { FMA, ADD, MUL, VOPD, DOT2, DOT4, MAD24, PERM, LDS_R32, LDS_R128, LDS_W32, LDS_W128, WMMA_F16, N_MODES };
struct ModeInfo {
    const char* name;
    double inst_per_trip;   // instructions per thread (per lane) per loop trip
    double ops_per_inst;    // arithmetic ops (flop / int op, a MAC = 2) per lane per instruction; bytes per lane for LDS
    const char* unit;
};
// WMMA: one instruction = 16x16x16 MACs per wave = 8192 ops per WAVE, i.e. 256 ops per lane
static const ModeInfo kModes[N_MODES] = {
    {"fma", 8, 2, "flop"},        {"add", 8, 1, "flop"},         {"mul", 8, 1, "flop"},        {"vopd", 8, 2, "flop"},
    {"dot2", 8, 4, "flop"},       {"dot4", 8, 8, "op"},          {"mad24", 8, 2, "op"},        {"perm", 8, 1, "op"},
    {"lds_r32", 8, 4, "byte"},    {"lds_r128", 8, 16, "byte"},   {"lds_w32", 8, 4, "byte"},    {"lds_w128", 8, 16, "byte"},
    {"wmma_f16", 8, 256, "flop"},
};

template <int MODE>
__global__ void __launch_bounds__(kThreads) rate_kernel(int iters, int* __restrict__ out, uint64_t* __restrict__ cyc) {
#if defined(RATE_GFX11)
    __shared__ __align__(16) float4 lds[kThreads * 8];   // 8 KiB: lane-linear, so every LDS mode is bank-conflict-free
    const int t = threadIdx.x + 1;
    float facc = 0.f;
    int iacc = 0;
    if (MODE >= LDS_R32 && MODE <= LDS_W128) {
        for (int i = 0; i < 8; ++i) lds[threadIdx.x * 8 + i] = make_float4(t, i, t + i, 1.f);
        __syncthreads();
    }
    const uint64_t c0 = clock64();
    if (MODE == FMA) {
        float a[kChains];
        const float b = 0.999f + 1e-6f * t, k = 1e-3f * t;
        for (int c = 0; c < kChains; ++c) a[c] = 1.f + c;
#pragma unroll 1
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int c = 0; c < kChains; ++c) asm volatile("v_fma_f32 %0, %0, %1, %2" : "+v"(a[c]) : "v"(b), "v"(k));
        }
        for (int c = 0; c < kChains; ++c) facc += a[c];
    } else if (MODE == ADD) {
        float a[kChains];
        const float k = 1e-3f * t;
        for (int c = 0; c < kChains; ++c) a[c] = 1.f + c;
#pragma unroll 1
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int c = 0; c < kChains; ++c) asm volatile("v_add_f32 %0, %0, %1" : "+v"(a[c]) : "v"(k));
        }
        for (int c = 0; c < kChains; ++c) facc += a[c];
    } else if (MODE == MUL) {
        float a[kChains];
        const float b = 0.9999f + 1e-7f * t;
        for (int c = 0; c < kChains; ++c) a[c] = 1.f + c;
#pragma unroll 1
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int c = 0; c < kChains; ++c) asm volatile("v_mul_f32 %0, %0, %1" : "+v"(a[c]) : "v"(b));
        }
        for (int c = 0; c < kChains; ++c) facc += a[c];
    } else if (MODE == VOPD) {
        float a[kChains];
        const float b = 0.999f + 1e-6f * t, k = 1e-3f * t;
        for (int c = 0; c < kChains; ++c) a[c] = 1.f + c;
#pragma unroll 1
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int c = 0; c < kChains; c += 2) {   // 4 independent fma chains + 4 independent mul chains: VOPD-eligible pairs
                a[c] = __builtin_fmaf(a[c], b, k);
                a[c + 1] = a[c + 1] * b;
            }
        }
        for (int c = 0; c < kChains; ++c) facc += a[c];
    } else if (MODE == DOT2) {
        float a[kChains];
        h2v x, y;
        x[0] = (_Float16) (0.5f + 0.001f * t); x[1] = (_Float16) 0.25f;
        y[0] = (_Float16) 0.5f; y[1] = (_Float16) (0.75f + 0.001f * t);
        for (int c = 0; c < kChains; ++c) a[c] = 1.f + c;
#pragma unroll 1
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int c = 0; c < kChains; ++c) a[c] = __builtin_amdgcn_fdot2(x, y, a[c], false);
        }
        for (int c = 0; c < kChains; ++c) facc += a[c];
    } else if (MODE == DOT4) {
        int a[kChains];
        const int w = t * 0x01020304 + 0x7f7f0101, q = t * 0x03030303 + 0x01017f7f;
        for (int c = 0; c < kChains; ++c) a[c] = c;
#pragma unroll 1
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int c = 0; c < kChains; ++c) a[c] = __builtin_amdgcn_sudot4(true, w, true, q, a[c], false);
        }
        for (int c = 0; c < kChains; ++c) iacc ^= a[c];
    } else if (MODE == MAD24) {
        int a[kChains];
        const int b = t * 3 + 1, k = t + 7;
        for (int c = 0; c < kChains; ++c) a[c] = c + 1;
#pragma unroll 1
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int c = 0; c < kChains; ++c) a[c] = __mul24(a[c], b) + k;
        }
        for (int c = 0; c < kChains; ++c) iacc ^= a[c];
    } else if (MODE == PERM) {
        unsigned a[kChains];
        const unsigned b = 0x01020304u * t, sel = 0x07050301u;
        for (int c = 0; c < kChains; ++c) a[c] = 0x11111111u * (c + 1);
#pragma unroll 1
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int c = 0; c < kChains; ++c) a[c] = __builtin_amdgcn_perm(a[c], b, sel);
        }
        for (int c = 0; c < kChains; ++c) iacc ^= (int) a[c];
    } else if (MODE == LDS_R32 || MODE == LDS_R128) {
        volatile float4* v4 = lds;
        volatile float* v1 = reinterpret_cast<volatile float*>(lds);
#pragma unroll 1
        for (int it = 0; it < iters; ++it) {
            if (MODE == LDS_R128) {
#pragma unroll
                for (int c = 0; c < 8; ++c) { const float4 r = v4[c * kThreads + threadIdx.x]; facc += r.x + r.w; }
            } else {
#pragma unroll
                for (int c = 0; c < 8; ++c) facc += v1[c * kThreads * 4 + threadIdx.x];
            }
        }
    } else if (MODE == LDS_W32 || MODE == LDS_W128) {
        volatile float4* v4 = lds;
        volatile float* v1 = reinterpret_cast<volatile float*>(lds);
        const float4 val = make_float4(t, t + 1, t + 2, t + 3);
#pragma unroll 1
        for (int it = 0; it < iters; ++it) {
            if (MODE == LDS_W128) {
#pragma unroll
                for (int c = 0; c < 8; ++c) v4[c * kThreads + threadIdx.x] = val;
            } else {
#pragma unroll
                for (int c = 0; c < 8; ++c) v1[c * kThreads * 4 + threadIdx.x] = val.x;
            }
        }
    } else {   // WMMA_F16 (one wave computes the 16x16x16 tile, 8 independent accumulator chains)
        h16v a, b;
        for (int i = 0; i < 16; ++i) { a[i] = (_Float16) ((t + i) & 7); b[i] = (_Float16) ((t * 3 + i) & 7); }
        f8v acc[kChains];
        for (int c = 0; c < kChains; ++c) for (int i = 0; i < 8; ++i) acc[c][i] = 0.f;
#pragma unroll 1
        for (int it = 0; it < iters; ++it) {
#pragma unroll
            for (int c = 0; c < kChains; ++c) acc[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc[c]);
        }
        for (int c = 0; c < kChains; ++c) for (int i = 0; i < 8; ++i) facc += acc[c][i];
    }
    const uint64_t c1 = clock64();
    if (threadIdx.x == 0) cyc[blockIdx.x] = c1 - c0;
    if (facc == 123456.789f || iacc == 0x7a5a5a5a) out[blockIdx.x] = 1;   // practically never; keeps the work alive
#else
    if (iters < 0) { out[0] = MODE; cyc[0] = 0; }
#endif
}

template <int MODE>
static void launch(int blocks, int iters, int* d_out, uint64_t* d_cyc) {
    (void) hipGetLastError();   // clear: an old fault must not be blamed on this geometry (LEADS.md Part U.3)
    rate_kernel<MODE><<<blocks, kThreads>>>(iters, d_out, d_cyc);
    CK(hipGetLastError());
}
static void launch_mode(int mode, int blocks, int iters, int* d_out, uint64_t* d_cyc) {
    switch (mode) {
        case FMA: launch<FMA>(blocks, iters, d_out, d_cyc); break;
        case ADD: launch<ADD>(blocks, iters, d_out, d_cyc); break;
        case MUL: launch<MUL>(blocks, iters, d_out, d_cyc); break;
        case VOPD: launch<VOPD>(blocks, iters, d_out, d_cyc); break;
        case DOT2: launch<DOT2>(blocks, iters, d_out, d_cyc); break;
        case DOT4: launch<DOT4>(blocks, iters, d_out, d_cyc); break;
        case MAD24: launch<MAD24>(blocks, iters, d_out, d_cyc); break;
        case PERM: launch<PERM>(blocks, iters, d_out, d_cyc); break;
        case LDS_R32: launch<LDS_R32>(blocks, iters, d_out, d_cyc); break;
        case LDS_R128: launch<LDS_R128>(blocks, iters, d_out, d_cyc); break;
        case LDS_W32: launch<LDS_W32>(blocks, iters, d_out, d_cyc); break;
        case LDS_W128: launch<LDS_W128>(blocks, iters, d_out, d_cyc); break;
        default: launch<WMMA_F16>(blocks, iters, d_out, d_cyc); break;
    }
}

static std::vector<int> parse_ints(const char* s) {
    std::vector<int> v;
    for (const char* p = s; *p;) {
        v.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p == ',') ++p;
    }
    return v;
}

int main(int argc, char** argv) {
    int device = 0, iters = 20000, reps = 5, cus_override = 0, flush_mib = 512;
    std::vector<int> waves = {4, 8, 12};
    std::string modes_arg;
    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* k) { return !std::strcmp(argv[i], k) && i + 1 < argc; };
        if (arg("--device")) device = std::atoi(argv[++i]);
        else if (arg("--iters")) iters = std::atoi(argv[++i]);
        else if (arg("--reps")) reps = std::max(1, std::atoi(argv[++i]));
        else if (arg("--waves-per-simd")) waves = parse_ints(argv[++i]);
        else if (arg("--modes")) modes_arg = argv[++i];
        else if (arg("--cus")) cus_override = std::atoi(argv[++i]);
        else if (arg("--flush-mib")) flush_mib = std::max(0, std::atoi(argv[++i]));
        else {
            std::fprintf(stderr, "unknown argument %s\n", argv[i]);
            return 2;
        }
    }
    CK(hipSetDevice(device));
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, device));
    if (std::strstr(p.gcnArchName, "gfx110") == nullptr && std::strstr(p.gcnArchName, "gfx115") == nullptr) {
        std::fprintf(stderr, "valu_rate: this probe measures gfx11 wave32 instructions (gfx110x / gfx115x); %s is not one\n", p.gcnArchName);
        return 3;
    }
    const int cus = cus_override > 0 ? cus_override : p.multiProcessorCount;
    std::printf("DEVICE name=\"%s\" arch=%s cus=%d reported_clock_mhz=%d iters=%d reps=%d\n", p.name, p.gcnArchName, cus, p.clockRate / 1000, iters, reps);
    std::vector<int> run_modes;
    for (int m = 0; m < N_MODES; ++m)
        if (modes_arg.empty() || (std::string(",") + modes_arg + ",").find(std::string(",") + kModes[m].name + ",") != std::string::npos) run_modes.push_back(m);
    int* d_out;
    uint64_t* d_cyc;
    const int max_blocks = cus * (*std::max_element(waves.begin(), waves.end()));
    CK(hipMalloc(&d_out, 65536 * sizeof(int)));
    CK(hipMemset(d_out, 0, 65536 * sizeof(int)));
    CK(hipMalloc(&d_cyc, (size_t) max_blocks * sizeof(uint64_t)));
    void* d_flush = nullptr;
    if (flush_mib > 0) CK(hipMalloc(&d_flush, (size_t) flush_mib << 20));
    hipEvent_t ea, eb;
    CK(hipEventCreate(&ea));
    CK(hipEventCreate(&eb));
    for (int mode : run_modes) {
        for (int w : waves) {
            const int blocks = cus * w;   // 64-thread blocks: 2 waves each, W waves per SIMD at 2 SIMDs per CU
            launch_mode(mode, blocks, iters / 10 + 1, d_out, d_cyc);   // warm up: clocks and code load
            CK(hipDeviceSynchronize());
            std::vector<double> rate_clk, rate_s, clk_ghz;
            for (int r = 0; r < reps; ++r) {
                if (d_flush) CK(hipMemset(d_flush, r + 1, (size_t) flush_mib << 20));
                CK(hipDeviceSynchronize());
                CK(hipEventRecord(ea));
                launch_mode(mode, blocks, iters, d_out, d_cyc);
                CK(hipEventRecord(eb));
                CK(hipEventSynchronize(eb));
                float ms;
                CK(hipEventElapsedTime(&ms, ea, eb));
                std::vector<uint64_t> cyc((size_t) blocks);
                CK(hipMemcpy(cyc.data(), d_cyc, (size_t) blocks * sizeof(uint64_t), hipMemcpyDeviceToHost));
                std::sort(cyc.begin(), cyc.end());
                const double med_cyc = (double) cyc[cyc.size() / 2];
                // WMMA instructions are per wave: 8192 ops each, 2 waves a block; the other modes count per lane
                const double lanes_per_block = kThreads;
                const double ops_block = (double) iters * kModes[mode].inst_per_trip * kModes[mode].ops_per_inst * lanes_per_block;
                const double blocks_per_cu = (double) w;
                rate_clk.push_back(ops_block * blocks_per_cu / med_cyc);                       // ops (or bytes) per clock per CU
                rate_s.push_back(ops_block * (double) blocks / (ms * 1e-3));                   // ops per second, whole chip
                clk_ghz.push_back(med_cyc / (ms * 1e-3) / 1e9);                                // implied shader clock
            }
            auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
            const double inst_per_clk_cu = med(rate_clk) / kModes[mode].ops_per_inst / (mode == WMMA_F16 ? 32.0 : 1.0);
            std::printf("RESULT mode=%s waves_per_simd=%d unit=%s ops_per_clk_cu=%.3f inst_per_clk_cu=%.3f ops_per_s=%.4e implied_clock_ghz=%.4f best_ops_per_clk_cu=%.3f\n",
                        kModes[mode].name, w, kModes[mode].unit, med(rate_clk), inst_per_clk_cu, med(rate_s), med(clk_ghz),
                        *std::max_element(rate_clk.begin(), rate_clk.end()));
            std::fflush(stdout);
        }
    }
    CK(hipEventDestroy(ea));
    CK(hipEventDestroy(eb));
    if (d_flush) CK(hipFree(d_flush));
    CK(hipFree(d_cyc));
    CK(hipFree(d_out));
    return 0;
}
