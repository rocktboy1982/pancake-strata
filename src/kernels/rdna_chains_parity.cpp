// src/kernels/rdna_chains_parity.cpp - STRATA_VERIFY_GEMV_CHAINS (LEADS.md Part U U7): the chain variants of the gfx1100
// decode GEMVs (rdna_mmvq.cu: rdna_q6k_chain_kernel / rdna_kq_chain_kernel, 2 and 4 chains) against the original RDNA
// kernels.  The chain split is an exact i32 re-association and the f32 stage keeps its order, so the outputs must be
// BITWISE equal (not within a tolerance) for Q4_K, Q5_K, Q6_K, IQ4_XS, every column count 1..16, three shapes.
// Random weights (fp16 fields kept finite) and activations quantized by native_quantize_q8_1, as rdna_mmvq_parity.
// Exit 77 (skip): no device, a wave64 device, or a CUDA / gfx906 build.  GPU, synthetic, no model.
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if defined(_WIN32)   // MSVC has no setenv
#define setenv(k, v, o) _putenv_s((k), (v))
#endif

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}
struct Fmt { int type; const char* name; int bytes; int d_off[2]; };
constexpr Fmt kFmts[] = {{12, "Q4_K", 144, {0, 2}}, {13, "Q5_K", 176, {0, 2}}, {14, "Q6_K", 210, {208, -1}}, {23, "IQ4_XS", 136, {0, -1}}};
struct Shape { int n_in, n_out; };
constexpr Shape kShapes[] = {{2560, 1000}, {4096, 257}, {256, 8}};
constexpr int kMaxCols = 16;
}  // namespace

int main() {
    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");
    setenv("STRATA_RDNA_KQ_ALL", "1", 1);   // Q4_K / Q5_K are taken below 4 columns too
    setenv("STRATA_RDNA_Q6K", "1", 1);
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::printf("SKIP: no GPU\n"); return 77; }
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    if (prop.warpSize != 32) { std::printf("SKIP: wave%d device\n", prop.warpSize); return 77; }
#if !defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
    std::printf("SKIP: the RDNA GEMVs are not built here (CUDA or gfx906)\n");
    return 77;
#endif
    std::mt19937 rng(41);
    std::uniform_int_distribution<int> byte(0, 255), mant(0, 1023);
    std::normal_distribution<float> nd(0.f, 1.f);
    int bad = 0, cases = 0;
    for (const Shape& sh : kShapes) {
        const int nsb = sh.n_in / 256;
        std::vector<float> x((size_t) kMaxCols * sh.n_in);
        for (auto& v : x) v = nd(rng);
        float* dx = nullptr; void* dq = nullptr; float* dy = nullptr;
        ck(cudaMalloc(&dx, x.size() * 4), "malloc x");
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "h2d x");
        ck(cudaMalloc(&dq, k::native_q8_1_bytes(sh.n_in, 1) * kMaxCols), "malloc q");
        ck(cudaMalloc(&dy, (size_t) kMaxCols * sh.n_out * 4), "malloc y");
        k::native_quantize_q8_1(dx, (uint8_t*) dq, sh.n_in, kMaxCols, st);
        ck(cudaDeviceSynchronize(), "quantize");
        for (const Fmt& f : kFmts) {
            std::vector<uint8_t> w((size_t) sh.n_out * nsb * f.bytes);
            for (auto& b : w) b = (uint8_t) byte(rng);
            for (size_t blk = 0; blk < (size_t) sh.n_out * nsb; ++blk)
                for (int off : f.d_off) {
                    if (off < 0) continue;
                    const uint16_t h = (uint16_t) (0x1C00 + mant(rng));
                    std::memcpy(w.data() + blk * f.bytes + off, &h, 2);
                }
            void* dw = nullptr;
            ck(cudaMalloc(&dw, w.size()), "malloc w");
            ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "h2d w");
            for (int nc = 1; nc <= kMaxCols; ++nc) {
                const size_t ny = (size_t) nc * sh.n_out;
                std::vector<float> out[3];
                bool taken = true;
                const int chains[3] = {0, 2, 4};
                for (int r = 0; r < 3; ++r) {
                    k::rdna_chains_override(chains[r]);
                    ck(cudaMemset(dy, 0xff, ny * 4), "memset y");   // NaN: a row never written differs
                    taken = (f.type == 14 ? k::rdna_q6k_mmvq(dw, dq, dy, sh.n_in, sh.n_out, nc, st)
                                          : k::rdna_kq_mmvq(f.type, dw, dq, dy, sh.n_in, sh.n_out, nc, st)) && taken;
                    ck(cudaGetLastError(), "launch");
                    ck(cudaDeviceSynchronize(), "gemv");
                    out[r].resize(ny);
                    ck(cudaMemcpy(out[r].data(), dy, ny * 4, cudaMemcpyDeviceToHost), "d2h y");
                }
                ++cases;
                const bool b2 = std::memcmp(out[0].data(), out[1].data(), ny * 4) == 0;
                const bool b4 = std::memcmp(out[0].data(), out[2].data(), ny * 4) == 0;
                const bool ok = taken && b2 && b4;
                if (!ok || nc == 1 || nc == kMaxCols)
                    std::printf("%s %dx%d cols %d: chains 2 %s, chains 4 %s than the original  %s\n", f.name, sh.n_in, sh.n_out, nc,
                                b2 ? "bitwise equal" : "DIFFERS", b4 ? "bitwise equal" : "DIFFERS", ok ? "ok" : "FAIL");
                if (!ok) ++bad;
            }
            ck(cudaFree(dw), "free w");
        }
        ck(cudaFree(dx), "free x");
        ck(cudaFree(dq), "free q");
        ck(cudaFree(dy), "free y");
    }
    k::rdna_chains_override(0);
    std::printf("%s: %d of %d chain cases bitwise equal to the original RDNA kernels\n", bad ? "FAIL" : "PASS", cases - bad, cases);
    return bad ? 1 : 0;
}
