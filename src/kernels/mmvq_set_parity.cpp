// src/kernels/mmvq_set_parity.cpp - STRATA_MMVQ_SET (native_mmvq_set: two or three matrices on one q8_1 input in one
// launch; jstamagal/Strata-gfx1030 324081287b, LEADS.md "gfx1030 decode") against each matrix's own native_mmvq call,
// BIT FOR BIT: the decode window's projection types (Q4_0, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS), mixed-type sets of 2 and 3,
// n_out not a multiple of the rows a wavefront takes, 1..8 columns.  Run it with and without STRATA_TSUM=1 (the test
// table does both).  Exit 77 (skip): no device, a wave64 device, or a CUDA / gfx906 build (the set kernel is HIP wave32
// only and native_mmvq_set returns false there).  GPU, synthetic, no model.
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

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
    int block_vals;   // weights per block
    int bytes;        // bytes per block
    int d_off[2];     // fp16 fields to make finite (-1: none)
};
constexpr Fmt kFmts[] = {
    {2, "Q4_0", 32, 18, {0, -1}},
    {8, "Q8_0", 32, 34, {0, -1}},
    {12, "Q4_K", 256, 144, {0, 2}},
    {13, "Q5_K", 256, 176, {0, 2}},
    {14, "Q6_K", 256, 210, {208, -1}},
    {23, "IQ4_XS", 256, 136, {0, -1}},
};
constexpr int kNF = (int) (sizeof(kFmts) / sizeof(kFmts[0]));
constexpr int kNIn = 2560;
constexpr int kOuts[3] = {1001, 640, 257};
constexpr int kMaxCols = 8;
}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    if (prop.warpSize != 32) {
        std::printf("SKIP: wave%d device (STRATA_MMVQ_SET is wave32 only)\n", prop.warpSize);
        return 77;
    }
#if !defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
    std::printf("SKIP: STRATA_MMVQ_SET is not built here (CUDA or gfx906)\n");
    return 77;
#endif
    cudaStream_t st = nullptr;
    ck(cudaStreamCreate(&st), "stream");
    k::native_mmvq_set_tune(1);

    std::mt19937 rng(1030);
    std::uniform_int_distribution<int> byte(0, 255), mant(0, 1023);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) kMaxCols * kNIn);
    for (auto& v : x) v = nd(rng);
    float* dx = nullptr;
    void* dq = nullptr;
    ck(cudaMalloc(&dx, x.size() * 4), "malloc x");
    ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "h2d x");
    ck(cudaMalloc(&dq, k::native_q8_1_bytes(kNIn, kMaxCols)), "malloc q");
    k::native_quantize_q8_1(dx, dq, kNIn, kMaxCols, st);
    ck(cudaStreamSynchronize(st), "quantize");

    // one weight matrix per (format, output size)
    void* dw[kNF][3] = {};
    for (int f = 0; f < kNF; ++f)
        for (int o = 0; o < 3; ++o) {
            const Fmt& fm = kFmts[f];
            const size_t nblk = (size_t) kOuts[o] * (kNIn / fm.block_vals);
            std::vector<uint8_t> w(nblk * fm.bytes);
            for (auto& b : w) b = (uint8_t) byte(rng);
            for (size_t blk = 0; blk < nblk; ++blk)
                for (int off : fm.d_off) {
                    if (off < 0) continue;
                    const uint16_t h = (uint16_t) (0x1C00 + mant(rng));   // fp16 0.0039 .. 0.0078
                    std::memcpy(w.data() + blk * fm.bytes + off, &h, 2);
                }
            ck(cudaMalloc(&dw[f][o], w.size()), "malloc w");
            ck(cudaMemcpy(dw[f][o], w.data(), w.size(), cudaMemcpyHostToDevice), "h2d w");
        }
    float* dy_ref = nullptr;
    float* dy_set = nullptr;
    const size_t ymax = (size_t) kMaxCols * (kOuts[0] + kOuts[1] + kOuts[2]);
    ck(cudaMalloc(&dy_ref, ymax * 4), "malloc yr");
    ck(cudaMalloc(&dy_set, ymax * 4), "malloc ys");

    int bad = 0, cases = 0;
    for (int n_items = 2; n_items <= 3; ++n_items)
        for (int f0 = 0; f0 < kNF; ++f0)
            for (int nc = 1; nc <= kMaxCols; ++nc) {
                k::NativeMmvqItem ref[3], set[3];
                size_t off = 0;
                for (int i = 0; i < n_items; ++i) {
                    const int f = (f0 + i * 2) % kNF;   // mixed formats in one set
                    ref[i] = {kFmts[f].type, dw[f][i], dy_ref + off, kOuts[i]};
                    set[i] = {kFmts[f].type, dw[f][i], dy_set + off, kOuts[i]};
                    off += (size_t) nc * kOuts[i];
                }
                ck(cudaMemsetAsync(dy_ref, 0xff, off * 4, st), "memset r");   // NaN: a row never written fails
                ck(cudaMemsetAsync(dy_set, 0xff, off * 4, st), "memset s");
                for (int i = 0; i < n_items; ++i)
                    k::native_mmvq(ref[i].type, ref[i].weights, dq, ref[i].y, kNIn, ref[i].n_out, nc, st);
                const bool taken = k::native_mmvq_set(set, n_items, dq, kNIn, nc, st);
                ck(cudaGetLastError(), "launch");
                ck(cudaStreamSynchronize(st), "gemv");
                std::vector<float> a(off), b(off);
                ck(cudaMemcpy(a.data(), dy_ref, off * 4, cudaMemcpyDeviceToHost), "d2h r");
                ck(cudaMemcpy(b.data(), dy_set, off * 4, cudaMemcpyDeviceToHost), "d2h s");
                ++cases;
                const bool same = taken && std::memcmp(a.data(), b.data(), off * 4) == 0;
                bool finite = true;
                for (float v : a) finite = finite && std::isfinite(v);
                if (!same || !finite) {
                    size_t first = 0;
                    while (first < off && std::memcmp(&a[first], &b[first], 4) == 0) ++first;
                    std::printf("set of %d from %s, cols %d: %s%s (first difference at %zu)  FAIL\n", n_items, kFmts[f0].name, nc,
                                taken ? "" : "declined, ", finite ? "" : "non-finite reference, ", first);
                    ++bad;
                }
            }
    for (int f = 0; f < kNF; ++f)
        for (int o = 0; o < 3; ++o) cudaFree(dw[f][o]);
    cudaFree(dx);
    cudaFree(dq);
    cudaFree(dy_ref);
    cudaFree(dy_set);
    std::printf("%s: %d of %d native_mmvq_set cases bitwise native_mmvq (STRATA_TSUM %s)\n", bad ? "FAIL" : "PASS", cases - bad, cases,
                std::getenv("STRATA_TSUM") ? std::getenv("STRATA_TSUM") : "unset");
    return bad ? 1 : 0;
}
