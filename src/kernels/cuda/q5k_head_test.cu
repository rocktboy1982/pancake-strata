// src/kernels/cuda/q5k_head_test.cu - pancake-strata: basalt's q5k_head_mmvq against this tree's native_mmvq (type 13,
// Q5_K) and native_mmvq_il, bit for bit, for 1 to 8 columns in Flash-Next's head shape (2560 x 248,320), and their
// times.  Random Q5_K rows (finite half scales), activations quantized by native_quantize_q8_1.  Needs a card:
//     q5k_head_test [N_OUT]
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/q5k_head.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
constexpr int kBlockBytes = 176;   // Q5_K: d, dmin (half), scales[12], qh[32], qs[128]
}  // namespace

int main(int argc, char** argv) {
    using namespace strata::kernels;
    const int n_in = 2560, n_out = argc > 1 ? std::atoi(argv[1]) : 248320;
    const size_t row_bytes = size_t(n_in / 256) * kBlockBytes, w_bytes = row_bytes * n_out;
    std::mt19937 rng(42);
    std::vector<uint8_t> w(w_bytes);
    for (auto& b : w) b = uint8_t(rng());
    std::normal_distribution<float> nd(0.f, 1.f);
    for (size_t b = 0; b < w_bytes / kBlockBytes; ++b) {   // finite scales of a plausible size
        const __half d = __float2half(0.002f + 0.004f * std::abs(nd(rng)));
        const __half m = __float2half(0.002f * std::abs(nd(rng)));
        std::memcpy(&w[b * kBlockBytes], &d, 2);
        std::memcpy(&w[b * kBlockBytes + 2], &m, 2);
    }
    void* dw = nullptr;
    ck(cudaMalloc(&dw, w_bytes + 256), "weights");
    ck(cudaMemcpy(dw, w.data(), w_bytes, cudaMemcpyHostToDevice), "upload");
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    if (!q5k_head_supported(n_in, 1)) {
        std::printf("q5k_head_supported(2560, 1) is false\n");
        return 1;
    }
    int bad = 0;
    for (int nc = 1; nc <= 8; ++nc) {
        std::vector<float> x(size_t(n_in) * nc);
        for (auto& v : x) v = nd(rng);
        float *dx, *y_ref, *y_il, *y_new;
        void *xq, *xil;
        ck(cudaMalloc(&dx, x.size() * 4), "x");
        ck(cudaMalloc(&xq, native_q8_1_bytes(n_in, nc)), "xq");
        ck(cudaMalloc(&xil, native_q8_1_il_bytes(n_in, nc)), "xil");
        for (float** p : {&y_ref, &y_il, &y_new}) ck(cudaMalloc(p, size_t(n_out) * nc * 4), "y");
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        native_quantize_q8_1(dx, xq, n_in, nc, st);
        const bool il = native_mmvq_il_supported(13, nc, n_out);
        if (il && nc >= 2) native_q8_1_interleave(xq, xil, n_in, nc, st);
        float t[3] = {1e30f, 1e30f, 1e30f};
        for (int rep = 0; rep < 5; ++rep) {
            for (int k = 0; k < 3; ++k) {
                if (k == 1 && !il) continue;
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                cudaEventRecord(e0, st);
                if (k == 0) native_mmvq(13, dw, xq, y_ref, n_in, n_out, nc, st);
                else if (k == 1) native_mmvq_il(13, dw, xq, xil, y_il, n_in, n_out, nc, st);
                else q5k_head_mmvq(dw, xq, y_new, n_in, n_out, nc, st);
                cudaEventRecord(e1, st);
                ck(cudaStreamSynchronize(st), "run");
                float ms = 0;
                cudaEventElapsedTime(&ms, e0, e1);
                if (ms < t[k]) t[k] = ms;
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
            }
        }
        std::vector<float> a(size_t(n_out) * nc), b(a.size()), c(a.size());
        cudaMemcpy(a.data(), y_ref, a.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(c.data(), y_new, c.size() * 4, cudaMemcpyDeviceToHost);
        const bool same = std::memcmp(a.data(), c.data(), a.size() * 4) == 0;
        bool same_il = true;
        if (il) {
            cudaMemcpy(b.data(), y_il, b.size() * 4, cudaMemcpyDeviceToHost);
            same_il = std::memcmp(b.data(), c.data(), b.size() * 4) == 0;
        }
        const double gbs = double(w_bytes) / (t[2] * 1e-3) / 1e9;
        std::printf("%d column(s): native_mmvq %.3f ms, native_mmvq_il %s ms, q5k_head %.3f ms (%.0f GB/s, x%.2f) - %s%s\n",
                    nc, t[0], il ? std::to_string(t[1]).substr(0, 5).c_str() : "-", t[2], gbs,
                    (il && nc >= 2 ? std::min(t[0], t[1]) : t[0]) / t[2], same ? "the same bits" : "DIFFERENT",
                    il ? (same_il ? " (il too)" : " (il DIFFERENT)") : "");
        bad += !same || !same_il;
        cudaFree(dx); cudaFree(xq); cudaFree(xil); cudaFree(y_ref); cudaFree(y_il); cudaFree(y_new);
    }
    return bad ? 1 : 0;
}
