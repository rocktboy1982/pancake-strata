// raw_buffer.hip.cpp - LEADS.md Part L L11: raw-buffer loads against global-pointer and flat loads on a decode-shaped weight stream, gfx1100.
// Parity (every flavour reads the same bytes: the sums are equal) and bandwidth, at a working set inside the Infinity Cache and one far
// outside it.  Decides whether STRATA_HIP_RAW_BUFFER_WEIGHTS (the 28 STRATA_GBLOB sites of src/kernels/cuda/iq_kernels.cu) is worth porting.
//
// Shape: a "matrix" of `rows` rows of `rowbytes` bytes (default 640 B: close to a Q2_0 row of 2560 weights, 40 blocks of 18 B = 720 B, and to an
// IQ2_XS row of 2560, 10 blocks of 74 B = 740 B, and 16-byte aligned), one wave a row, lane l reads 16-byte
// chunks l, l + 32, ... (the access pattern of row_dot: 512 contiguous bytes a wave step), each lane adds the four words of every chunk into a
// 32-bit checksum, the row checksums are written out.  Variants:
//   flat    generic pointer taken from device memory, like grp_ptr[g] (flat_load)
//   global  address_space(1) pointer (what STRATA_HIP_GLOBAL_WEIGHTS gives: global_load)
//   nt      __builtin_nontemporal_load on a global pointer (STRATA_EXPDEC_NT)
//   rawbuf  buffer_load_b128, aux 0
//   rawslc  buffer_load_b128, aux slc (streaming)
// BUILD   hipcc -O2 --offload-arch=gfx1100 -Iinclude tools/pancake/probes/raw_buffer.hip.cpp -o /tmp/raw_buffer   (CMake target raw_buffer_probe)
// RUN     HIP_VISIBLE_DEVICES=0 /tmp/raw_buffer [--rows 65536 --rowbytes 640 --reps 30 --flags 0x31004000]
//         the working sets are rows * rowbytes (default 40 MiB, inside the 80 MiB IC) and 16x that; exit 0 pass, 1 mismatch, 77 no gfx11 GPU.
// READ    PARITY line first (a wrong descriptor flags word reads zeros: try --flags 0x31014000).  BENCH: GB/s per variant and set; a raw-buffer
//         gain over `global` of under 3% on the far set closes L11 (the matvec is not load-instruction bound); more reopens the port.
// STATUS  syntax + device compile checked with the ROCm 6.2 SDK on the Windows PC (ISA load counts in the ledger); NEVER RUN: box run needed.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "strata/kernels/raw_buffer.hpp"

#define CK(x)                                                                                        \
    do {                                                                                             \
        hipError_t e_ = (x);                                                                         \
        if (e_ != hipSuccess) {                                                                      \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));   \
            std::exit(1);                                                                            \
        }                                                                                            \
    } while (0)

using namespace strata::kernels::rawbuf;
typedef unsigned u4v __attribute__((ext_vector_type(4)));

enum Variant { kFlat, kGlobal, kNt, kRaw, kRawSlc, kN };
static const char* kNames[kN] = {"flat", "global", "nt", "rawbuf", "rawslc"};

__device__ __forceinline__ unsigned sum4(u4v v) { return v.x + v.y + v.z + v.w; }

template <int V>
__global__ void __launch_bounds__(256) stream_kernel(const unsigned long long* __restrict__ gp_, unsigned* __restrict__ out, int rows,
                                                     int rowbytes, Rsrc rs) {
    // the base pointer comes out of device memory, like grp_ptr[g] of iq_kernels.cu: to the compiler a generic pointer, so `flat` is a flat_load
    const unsigned char* w = (const unsigned char*) gp_[0];
    const int wave = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    if (wave >= rows) return;
    const int n16 = rowbytes / 16;
    const size_t row_off = (size_t) wave * rowbytes;
    unsigned acc = 0;
    for (int c = lane; c < n16; c += 32) {
        u4v v;
        if (V == kFlat) {
            v = *reinterpret_cast<const u4v*>(w + row_off + (size_t) c * 16);
        } else if (V == kGlobal) {
            const __attribute__((address_space(1))) u4v* gp = (const __attribute__((address_space(1))) u4v*) (w + row_off + (size_t) c * 16);
            v = *gp;
        } else if (V == kNt) {
            v = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(w + row_off + (size_t) c * 16));
        } else {
            const rb_v4i r = rb_load128(rs, (int) (row_off + (size_t) c * 16), V == kRawSlc ? kAuxSlc : kAuxNone);
            v = u4v{(unsigned) r.x, (unsigned) r.y, (unsigned) r.z, (unsigned) r.w};
        }
        acc += sum4(v) * (unsigned) (c + 1);   // position-dependent so a misplaced read changes the sum
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor(acc, o, 32);
    if (lane == 0) out[wave] = acc;
}

int main(int argc, char** argv) {
    int rows = 65536, rowbytes = 640, reps = 30, device = 0;
    unsigned flags = kGfx11Word3;
    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* k) { return !std::strcmp(argv[i], k) && i + 1 < argc; };
        if (arg("--rows")) rows = std::atoi(argv[++i]);
        else if (arg("--rowbytes")) rowbytes = std::atoi(argv[++i]);
        else if (arg("--reps")) reps = std::atoi(argv[++i]);
        else if (arg("--device")) device = std::atoi(argv[++i]);
        else if (arg("--flags")) flags = (unsigned) std::strtoul(argv[++i], nullptr, 0);
        else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (rowbytes % 16 || rows < 1) { std::fprintf(stderr, "raw_buffer: rowbytes a multiple of 16\n"); return 2; }
    hipDeviceProp_t p;
    if (hipSetDevice(device) != hipSuccess || hipGetDeviceProperties(&p, device) != hipSuccess ||
        (!std::strstr(p.gcnArchName, "gfx110") && !std::strstr(p.gcnArchName, "gfx115"))) {
        std::fprintf(stderr, "raw_buffer: no gfx11 GPU (device %d)\n", device);
        return 77;
    }
    std::printf("device %d %s, rowbytes %d, descriptor word3 0x%08x\n", device, p.gcnArchName, rowbytes, flags);
    bool ok = true;
    const int sets[2] = {rows, rows * 16};
    for (int si = 0; si < 2; ++si) {
        const int R = sets[si];
        const size_t bytes = (size_t) R * rowbytes;
        if (bytes > 0xffffffffull) { std::fprintf(stderr, "raw_buffer: set above 4 GiB (the descriptor size is 32-bit)\n"); return 2; }
        std::vector<unsigned char> h(bytes);
        std::mt19937 rng(5);
        for (auto& b : h) b = (unsigned char) rng();
        unsigned char* d_w;
        unsigned* d_o;
        unsigned long long* d_gp;
        CK(hipMalloc(&d_gp, 8));
        CK(hipMalloc(&d_w, bytes));
        CK(hipMalloc(&d_o, (size_t) R * 4));
        CK(hipMemcpy(d_w, h.data(), bytes, hipMemcpyHostToDevice));
        const unsigned long long gpv = (unsigned long long) (uintptr_t) d_w;
        CK(hipMemcpy(d_gp, &gpv, 8, hipMemcpyHostToDevice));
        // host reference
        std::vector<unsigned> ref(R);
        for (int r = 0; r < R; ++r) {
            unsigned acc = 0;
            for (int c = 0; c < rowbytes / 16; ++c) {
                unsigned s = 0;
                for (int k = 0; k < 4; ++k) { unsigned w; std::memcpy(&w, &h[(size_t) r * rowbytes + c * 16 + k * 4], 4); s += w; }
                acc += s * (unsigned) (c + 1);
            }
            ref[r] = acc;   // the wave sum of lane partials equals the plain sum (addition mod 2^32 is associative)
        }
        const Rsrc rs = make_rsrc((uint64_t) (uintptr_t) d_w, (uint32_t) bytes, flags);
        const unsigned grid = (unsigned) (((size_t) R * 32 + 255) / 256);
        std::printf("set %.1f MiB:", (double) bytes / 1048576.0);
        for (int v = 0; v < kN; ++v) {
            auto run = [&] {
                switch (v) {
                    case kFlat: stream_kernel<kFlat><<<grid, 256>>>(d_gp, d_o, R, rowbytes, rs); break;
                    case kGlobal: stream_kernel<kGlobal><<<grid, 256>>>(d_gp, d_o, R, rowbytes, rs); break;
                    case kNt: stream_kernel<kNt><<<grid, 256>>>(d_gp, d_o, R, rowbytes, rs); break;
                    case kRaw: stream_kernel<kRaw><<<grid, 256>>>(d_gp, d_o, R, rowbytes, rs); break;
                    default: stream_kernel<kRawSlc><<<grid, 256>>>(d_gp, d_o, R, rowbytes, rs); break;
                }
            };
            CK(hipMemset(d_o, 0, (size_t) R * 4));
            run();
            CK(hipDeviceSynchronize());
            std::vector<unsigned> o(R);
            CK(hipMemcpy(o.data(), d_o, (size_t) R * 4, hipMemcpyDeviceToHost));
            long bad = 0;
            for (int r = 0; r < R; ++r) bad += o[r] != ref[r];
            if (bad) ok = false;
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
                if (r) ms.push_back(t);
            }
            std::sort(ms.begin(), ms.end());
            std::printf("  %s %s %.0f GB/s", kNames[v], bad ? "MISMATCH" : "ok", (double) bytes / (ms[ms.size() / 2] * 1e-3) / 1e9);
        }
        std::printf("\n");
        CK(hipFree(d_w));
        CK(hipFree(d_gp));
        CK(hipFree(d_o));
    }
    std::printf("%s\n", ok ? "raw_buffer: PARITY PASS" : "raw_buffer: PARITY FAIL (try --flags 0x31014000)");
    return ok ? 0 : 1;
}
