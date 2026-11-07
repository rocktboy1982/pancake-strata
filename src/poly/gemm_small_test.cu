// src/poly/gemm_small_test.cu - pancake-strata: gemm_f32's register-tiled small kernel against gemm_small_kernel, bit
// for bit, in the shapes a Qwen3.6 prompt batch uses (the router 2,048 -> 256, the gates 2,048 -> 32), with token
// counts that are and are not multiples of the tiles; and the time of each.  Needs a card; run by hand.
#include "strata/poly/kernels.hpp"

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
}  // namespace

int main() {
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    void* blas = strata::poly::blas_create(st);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int bad = 0;
    struct Shape { int n_in, n_out, rows; };
    for (const Shape sh : {Shape{2048, 256, 4096}, Shape{2048, 256, 1037}, Shape{2048, 32, 4096}, Shape{2048, 32, 77},
                           Shape{2000, 200, 333}}) {
        std::vector<float> w(size_t(sh.n_out) * sh.n_in), x(size_t(sh.rows) * sh.n_in);
        for (auto& v : w) v = nd(rng);
        for (auto& v : x) v = nd(rng);
        float *dw, *dx, *dy;
        ck(cudaMalloc(&dw, w.size() * 4), "w");
        ck(cudaMalloc(&dx, x.size() * 4), "x");
        ck(cudaMalloc(&dy, size_t(sh.rows) * sh.n_out * 4), "y");
        cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        std::vector<float> ys[2];
        float ms[2];
        for (int tiled = 0; tiled < 2; ++tiled) {
            strata::poly::gemm_small_set_tiled(tiled != 0);
            float best = 1e30f;
            for (int rep = 0; rep < 5; ++rep) {
                cudaEvent_t e0, e1;
                cudaEventCreate(&e0);
                cudaEventCreate(&e1);
                cudaEventRecord(e0, st);
                strata::poly::gemm_f32(blas, dw, dx, dy, sh.n_in, sh.n_out, sh.n_out, sh.rows);
                cudaEventRecord(e1, st);
                ck(cudaStreamSynchronize(st), "run");
                float t = 0;
                cudaEventElapsedTime(&t, e0, e1);
                if (t < best) best = t;
                cudaEventDestroy(e0);
                cudaEventDestroy(e1);
            }
            ms[tiled] = best;
            ys[tiled].resize(size_t(sh.rows) * sh.n_out);
            cudaMemcpy(ys[tiled].data(), dy, ys[tiled].size() * 4, cudaMemcpyDeviceToHost);
        }
        const bool same = std::memcmp(ys[0].data(), ys[1].data(), ys[0].size() * 4) == 0;
        std::printf("%d -> %d, %d tokens: plain %.3f ms, tiled %.3f ms (x%.2f), %s\n", sh.n_in, sh.n_out, sh.rows,
                    ms[0], ms[1], ms[0] / ms[1], same ? "the same bits" : "DIFFERENT");
        bad += !same;
        cudaFree(dw);
        cudaFree(dx);
        cudaFree(dy);
    }
    strata::poly::blas_destroy(blas);
    return bad ? 1 : 0;
}
