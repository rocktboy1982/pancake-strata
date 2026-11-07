// src/poly/gdn_cols_test.cu - pancake-strata: gdn_step with 2 and 4 state columns a warp against one column a warp
// (gdn_state_reg_kernel): the outputs, the state and the snapshots must be the same bits; and the time of each.
// Qwen3.6-35B-A3B's shape by default (32 value heads, 16 key heads, 128 values a head).  Needs a card; run by hand:
//     poly_gdn_cols_test [ROWS]
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
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
float* dev(const std::vector<float>& h) {
    float* p = nullptr;
    ck(cudaMalloc(&p, h.size() * 4), "cudaMalloc");
    ck(cudaMemcpy(p, h.data(), h.size() * 4, cudaMemcpyHostToDevice), "upload");
    return p;
}
std::vector<float> host(const float* d, size_t n) {
    std::vector<float> h(n);
    ck(cudaMemcpy(h.data(), d, n * 4, cudaMemcpyDeviceToHost), "download");
    return h;
}
}  // namespace

int main(int argc, char** argv) {
    const int rows = argc > 1 ? std::atoi(argv[1]) : 2048;
    const int dim = 128, n_head = 32, k_heads = 16;
    const size_t row_n = size_t(2 * k_heads + n_head) * dim, d_inner = size_t(n_head) * dim;
    const size_t state_n = size_t(n_head) * dim * dim;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto fill = [&](size_t n, float sc) { std::vector<float> v(n); for (auto& x : v) x = nd(rng) * sc; return v; };
    const auto qkv = fill(size_t(rows) * row_n, 0.5f), alpha = fill(size_t(rows) * n_head, 1.0f),
               beta = fill(size_t(rows) * n_head, 1.0f), dt = fill(n_head, 0.5f), z = fill(size_t(rows) * d_inner, 1.0f),
               o_norm = fill(dim, 1.0f), s0 = fill(state_n, 0.05f);
    std::vector<float> a(n_head);
    for (auto& x : a) x = -std::abs(nd(rng)) - 0.1f;
    // snapshots after the first tokens: bits 0, 3, 7 (a guessed window's states)
    const unsigned snaps = (1u << 0) | (1u << 3) | (1u << 7);
    const int n_snaps = 3;

    float *d_dt = dev(dt), *d_a = dev(a), *d_z = dev(z), *d_on = dev(o_norm), *d_out = nullptr, *d_snap = nullptr;
    ck(cudaMalloc(&d_out, size_t(rows) * d_inner * 4), "out");
    ck(cudaMalloc(&d_snap, n_snaps * state_n * 4), "snap");
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");

    struct Run { std::vector<float> out, s, snap; float ms; };
    auto run = [&](int cols) {
        strata::poly::gdn_set_cols(cols);
        Run r{};
        float best = 1e30f;
        for (int rep = 0; rep < 3; ++rep) {   // the inputs are overwritten: fresh copies each time
            float *d_qkv = dev(qkv), *d_al = dev(alpha), *d_be = dev(beta), *d_s = dev(s0);
            ck(cudaMemset(d_snap, 0, n_snaps * state_n * 4), "memset");
            cudaEvent_t e0, e1;
            cudaEventCreate(&e0);
            cudaEventCreate(&e1);
            cudaEventRecord(e0, st);
            strata::poly::gdn_step(d_qkv, d_al, d_be, d_dt, d_a, d_s, d_z, d_on, d_out, dim, n_head, k_heads, rows,
                                   1e-6f, st, d_snap, state_n, snaps);
            cudaEventRecord(e1, st);
            ck(cudaStreamSynchronize(st), "run");
            float ms = 0;
            cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
            if (rep == 0) {
                r.out = host(d_out, size_t(rows) * d_inner);
                r.s = host(d_s, state_n);
                r.snap = host(d_snap, n_snaps * state_n);
            }
            cudaFree(d_qkv); cudaFree(d_al); cudaFree(d_be); cudaFree(d_s);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
        r.ms = best;
        return r;
    };
    const Run ref = run(1);
    std::printf("gdn_step, %d tokens: 1 column a warp %.3f ms\n", rows, ref.ms);
    int bad = 0;
    for (int cols : {2, 4, 8}) {
        const Run r = run(cols);
        const bool same = std::memcmp(r.out.data(), ref.out.data(), r.out.size() * 4) == 0 &&
                          std::memcmp(r.s.data(), ref.s.data(), r.s.size() * 4) == 0 &&
                          std::memcmp(r.snap.data(), ref.snap.data(), r.snap.size() * 4) == 0;
        std::printf("  %d columns a warp: %.3f ms (x%.2f), %s\n", cols, r.ms, ref.ms / r.ms,
                    same ? "the same bits" : "DIFFERENT");
        bad += !same;
    }
    return bad ? 1 : 0;
}
