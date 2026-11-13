// src/kernels/rdna_mmvq_parity.cpp - the gfx1100 multi-column decode GEMVs (rdna_mmvq.cu: rdna_q6k_mmvq for Q6_K,
// rdna_kq_mmvq for Q4_K / Q5_K / IQ4_XS; STRATA_RDNA_Q6K=1) against the generic native_mmvq kernels.  MOD_PLAN item 93.
//
// Random weight blocks (every byte random except the fp16 d / dmin fields, which are set to finite values) and random
// activations quantized by native_quantize_q8_1; three shapes (one with n_out not a multiple of the 8 rows a block
// takes) and every column count 1..8.  The RDNA kernels take the same products and scales in another summation order,
// so the check is bounded, not bitwise: finite, max |diff| within 1e-4 of the column's largest reference magnitude, and
// two runs bitwise equal.  STRATA_RDNA_KQ_ALL=1 is set so Q4_K / Q5_K are taken below 4 columns too.
// Exit 77 (skip): no device, a wave64 device, or a CUDA / gfx906 build (the RDNA kernels are stubs there).
// GPU, synthetic, no model.
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if defined(_WIN32)   // MSVC has no setenv / unsetenv
#define setenv(k, v, o) _putenv_s((k), (v))
#define unsetenv(k) _putenv_s((k), "")
#endif

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}
struct Fmt {
    int type;
    const char* name;
    int bytes;        // per 256-weight superblock
    int d_off[2];     // fp16 fields to make finite (-1: none)
};
constexpr Fmt kFmts[] = {
    {12, "Q4_K", 144, {0, 2}},
    {13, "Q5_K", 176, {0, 2}},
    {14, "Q6_K", 210, {208, -1}},
    {23, "IQ4_XS", 136, {0, -1}},
};
struct Shape { int n_in, n_out; };
constexpr Shape kShapes[] = {{2560, 1000}, {4096, 257}, {256, 8}};
constexpr int kMaxCols = 8;
}  // namespace

int main() {
    cudaStream_t st = nullptr;                 // the native kernels require an explicit stream (native_mmvq.cu validate_stream)
    ck(cudaStreamCreate(&st), "stream");
    unsetenv("STRATA_RDNA_Q6K");          // the reference: native_mmvq must keep its generic kernels
    setenv("STRATA_RDNA_KQ_ALL", "1", 1); // rdna_kq_mmvq takes Q4_K / Q5_K below 4 columns too (read per call)
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    if (prop.warpSize != 32) {
        std::printf("SKIP: wave%d device (the RDNA GEMVs are wave32 only)\n", prop.warpSize);
        return 77;
    }
#if !defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
    std::printf("SKIP: the RDNA GEMVs are not built here (CUDA or gfx906)\n");
    return 77;
#endif

    std::mt19937 rng(23);
    std::uniform_int_distribution<int> byte(0, 255), mant(0, 1023);
    std::normal_distribution<float> nd(0.f, 1.f);
    int bad = 0, cases = 0;
    for (const Shape& sh : kShapes) {
        const int nsb = sh.n_in / 256;
        std::vector<float> x((size_t) kMaxCols * sh.n_in);
        for (auto& v : x) v = nd(rng);
        float* dx = nullptr;
        void* dq = nullptr;
        float* dy = nullptr;
        ck(cudaMalloc(&dx, x.size() * 4), "malloc x");
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "h2d x");
        ck(cudaMalloc(&dq, k::native_q8_1_bytes(sh.n_in, kMaxCols)), "malloc q");
        ck(cudaMalloc(&dy, (size_t) kMaxCols * sh.n_out * 4), "malloc y");
        k::native_quantize_q8_1(dx, dq, sh.n_in, kMaxCols, st);
        ck(cudaDeviceSynchronize(), "quantize");
        for (const Fmt& f : kFmts) {
            std::vector<uint8_t> w((size_t) sh.n_out * nsb * f.bytes);
            for (auto& b : w) b = (uint8_t) byte(rng);
            for (size_t blk = 0; blk < (size_t) sh.n_out * nsb; ++blk)
                for (int off : f.d_off) {
                    if (off < 0) continue;
                    const uint16_t h = (uint16_t) (0x1C00 + mant(rng));   // fp16 2^-8 * (1 + m / 1024): 0.0039 .. 0.0078
                    std::memcpy(w.data() + blk * f.bytes + off, &h, 2);
                }
            void* dw = nullptr;
            ck(cudaMalloc(&dw, w.size()), "malloc w");
            ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "h2d w");
            for (int nc = 1; nc <= kMaxCols; ++nc) {
                const size_t ny = (size_t) nc * sh.n_out;
                std::vector<float> out[3];
                bool taken = true;
                for (int r = 0; r < 3; ++r) {
                    ck(cudaMemset(dy, 0xff, ny * 4), "memset y");   // NaN: a row never written fails
                    if (r == 0) k::native_mmvq(f.type, dw, dq, dy, sh.n_in, sh.n_out, nc, st);
                    else if (f.type == 14) taken = k::rdna_q6k_mmvq(dw, dq, dy, sh.n_in, sh.n_out, nc, st) && taken;
                    else taken = k::rdna_kq_mmvq(f.type, dw, dq, dy, sh.n_in, sh.n_out, nc, st) && taken;
                    ck(cudaGetLastError(), "launch");
                    ck(cudaDeviceSynchronize(), "gemv");
                    out[r].resize(ny);
                    ck(cudaMemcpy(out[r].data(), dy, ny * 4, cudaMemcpyDeviceToHost), "d2h y");
                }
                ++cases;
                if (!taken) {
                    std::printf("%s %dx%d cols %d: the RDNA kernel declined  FAIL\n", f.name, sh.n_in, sh.n_out, nc);
                    ++bad;
                    continue;
                }
                double rel = 0.0;
                bool finite = true;
                for (int c = 0; c < nc; ++c) {
                    double mref = 0.0, md = 0.0;
                    for (int i = 0; i < sh.n_out; ++i) {
                        const size_t j = (size_t) c * sh.n_out + i;
                        finite = finite && std::isfinite(out[0][j]) && std::isfinite(out[1][j]);
                        mref = std::fmax(mref, std::fabs((double) out[0][j]));
                        md = std::fmax(md, std::fabs((double) out[1][j] - out[0][j]));
                    }
                    rel = std::fmax(rel, md / (mref > 0 ? mref : 1.0));
                }
                const bool same = std::memcmp(out[1].data(), out[2].data(), ny * 4) == 0;
                const bool ok = finite && rel < 1e-4 && same;
                if (!ok || nc == 1 || nc == kMaxCols)
                    std::printf("%s %dx%d cols %d: max relative difference %.3g, %s, two runs %s  %s\n", f.name, sh.n_in,
                                sh.n_out, nc, rel, finite ? "finite" : "NON-FINITE", same ? "bitwise equal" : "DIFFER",
                                ok ? "ok" : "FAIL");
                if (!ok) ++bad;
            }
            ck(cudaFree(dw), "free w");
        }
        ck(cudaFree(dx), "free x");
        ck(cudaFree(dq), "free q");
        ck(cudaFree(dy), "free y");
    }
    std::printf("%s: %d of %d rdna_mmvq cases within 1e-4 of native_mmvq\n", bad ? "FAIL" : "PASS", cases - bad, cases);
    return bad ? 1 : 0;
}
