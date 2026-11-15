// src/kernels/rkv_parity.cpp - STRATA_RKV's kernels (include/strata/kernels/rkv.hpp) against the host rule: the evicted set of
// rkv_update (below budget + buffer nothing; above it the lowest R-KV scores until the budget; a second round only among the
// blocks still live), and rkv_mask_scores (exactly the evicted blocks' scores replaced).  GPU, synthetic, no model.
#include "strata/kernels/rkv.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
template <typename T> T* dev(size_t n) {
    T* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) {
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}

}  // namespace

int main() {
    using namespace strata::kernels;
    std::mt19937 rng(20261116);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    cudaStream_t s = nullptr;
    check(cudaStreamCreate(&s), "stream");
    int bad = 0;
    const int64_t max_blocks = 1500;
    const int D = 128, words = (int) ((max_blocks + 31) / 32);
    for (const int n : {1, 3, 8}) {
        RkvConfig c;
        c.budget = 200;
        c.buffer = 64;
        c.sink = 4;
        c.recent = 32;
        c.lambda = 0.1f;
        // keys with near-duplicates (the redundancy term has something to find)
        std::vector<float> pooled((size_t) max_blocks * D);
        for (auto& v : pooled) v = nd(rng);
        for (int b = 100; b < 400; b += 3)
            for (int d = 0; d < D; ++d) pooled[(size_t) b * D + d] = pooled[(size_t) (b - 1) * D + d] + 0.01f * nd(rng);
        std::vector<float> scores((size_t) n * max_blocks);
        for (auto& v : scores) v = nd(rng) * 3.0f;
        float* d_p = dev<float>(pooled.size());
        float* d_s = dev<float>(scores.size());
        int32_t* d_steps = dev<int32_t>((size_t) n * 4);
        int32_t* d_start = dev<int32_t>(1);
        uint32_t* d_ev = dev<uint32_t>((size_t) words);
        up(d_p, pooled);
        up(d_s, scores);
        check(cudaMemset(d_ev, 0, (size_t) words * 4), "memset");
        std::vector<uint32_t> ev((size_t) words, 0u);
        const int start = 10;
        up(d_start, std::vector<int32_t>{start});
        // three rounds: below the trigger, the first eviction, a second eviction among the live blocks
        for (const int n_bid : {250, 700, 900}) {
            std::vector<int32_t> steps((size_t) n * 4, 0);
            steps[(size_t) (n - 1) * 4 + 2] = n_bid;
            up(d_steps, steps);
            const std::vector<int> want = rkv_choose_host(scores.data(), n, max_blocks, n_bid, pooled.data(), D, start, c, ev);
            for (const int b : want) ev[(size_t) b >> 5] |= 1u << (b & 31);
            rkv_update(d_s, n, max_blocks, d_steps, 4, d_p, D, d_start, c, d_ev, s);
            check(cudaStreamSynchronize(s), "rkv sync");
            const std::vector<uint32_t> got = down(d_ev, (size_t) words);
            int diff = 0;
            for (int w = 0; w < words; ++w)
                for (uint32_t x = got[(size_t) w] ^ ev[(size_t) w]; x != 0; x &= x - 1) ++diff;
            if (diff) {
                std::fprintf(stderr, "rkv_update n %d n_bid %d: %d blocks differ (host evicts %zu)\n", n, n_bid, diff, want.size());
                ++bad;
                ev = got;   // keep the rounds comparable
            }
            if (n_bid == 250 && !want.empty()) { std::fprintf(stderr, "rkv: evicted below the trigger\n"); ++bad; }
            if (n_bid == 700 && (int) want.size() != (700 - c.recent - start) - c.budget) {
                std::fprintf(stderr, "rkv: first round should evict down to the budget\n");
                ++bad;
            }
        }
        rkv_mask_scores(d_s, n, max_blocks, d_ev, s);
        check(cudaStreamSynchronize(s), "mask sync");
        const std::vector<float> ms = down(d_s, scores.size());
        int mdiff = 0;
        for (int r = 0; r < n; ++r)
            for (int64_t b = 0; b < max_blocks; ++b) {
                const bool e = (ev[(size_t) b >> 5] >> (b & 31)) & 1u;
                const float want = e ? -3e38f : scores[(size_t) r * max_blocks + b];
                mdiff += std::memcmp(&want, &ms[(size_t) r * max_blocks + b], 4) != 0 ? 1 : 0;
            }
        if (mdiff) { std::fprintf(stderr, "rkv_mask_scores n %d: %d scores differ\n", n, mdiff); ++bad; }
        cudaFree(d_p);
        cudaFree(d_s);
        cudaFree(d_steps);
        cudaFree(d_start);
        cudaFree(d_ev);
    }
    (void) cudaStreamDestroy(s);
    std::printf("rkv_parity: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
