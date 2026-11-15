// src/kernels/qsa_page_select_recall.cpp - MOD_PLAN.md section 6 item 14: the two long-context decode selects against the
// exact one (qsa_block_scores + qsa_block_topk_ref), on synthetic keys (GPU, no model).
//   STRATA_QSA_PARTIAL_TOPK  qsa_block_topk_partial: must give the reference ids exactly (it is exact by construction).
//   STRATA_QSA_PAGE_SEL      qsa_page_select_with (16- and 32-cell pages, default kept pages): recall of the reference
//                            cells; must be a valid selection (ascending, in range, `width` cells), the reference itself
//                            while every page is kept, and the same after pooled keys are rewritten in place as with a
//                            fresh summary buffer (the fingerprint refresh).
// Two key layouts: "local" (block amplitudes drift slowly, a few strong isolated needles) and "iid" (no page structure,
// the worst case for page bounds).  Usage: qsa_page_select_recall [context=65536] [capacity_cells=262144] [reps=20]
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}
template <typename T> T* dev(size_t n) {
    T* d = nullptr;
    ck(cudaMalloc(&d, n * sizeof(T) + 64), "malloc");
    return d;
}
template <typename T> void put(T* d, const std::vector<T>& h) {
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}
template <typename T> std::vector<T> get(const T* d, size_t n) {
    std::vector<T> h(n);
    ck(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
}  // namespace

int main(int argc, char** argv) {
    const int64_t ctx_max = argc > 1 ? std::atoll(argv[1]) : 65536;
    const int64_t capacity = argc > 2 ? std::atoll(argv[2]) : 262144;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 20;
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = capacity / s.idx_block + 2, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    const int64_t nq = 4;   // a decode window's rows
    if (ctx_max + nq > capacity) { std::fprintf(stderr, "context above the capacity\n"); return 2; }
    float* d_pooled = dev<float>((size_t) (max_blocks * 128));
    float* d_pooled2 = dev<float>((size_t) (max_blocks * 128));
    float* d_dead = dev<float>(128);
    float* d_q = dev<float>((size_t) (nq * 512));
    int32_t* d_steps = dev<int32_t>((size_t) (nq * k::kStepCount));
    float* d_sc = dev<float>((size_t) (nq * max_blocks));
    int32_t* d_ref = dev<int32_t>((size_t) (nq * cap));
    int32_t* d_ids = dev<int32_t>((size_t) (nq * cap));
    int32_t* d_ids2 = dev<int32_t>((size_t) (nq * cap));
    bool ok = true;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto timed = [&](auto f) {
        f();
        ck(cudaDeviceSynchronize(), "warm");
        cudaEventRecord(e0);
        for (int r = 0; r < reps; ++r) f();
        cudaEventRecord(e1);
        ck(cudaEventSynchronize(e1), "time");
        float ms = 0;
        cudaEventElapsedTime(&ms, e0, e1);
        return 1e3f * ms / (float) reps;
    };
    for (int layout = 0; layout < 2; ++layout) {
        std::mt19937 rng(7 + layout);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> dir(128), pooled((size_t) (max_blocks * 128)), dead(128), q((size_t) (nq * 512));
        for (auto& x : dir) x = nd(rng);
        float a = 0.f;
        for (int64_t b = 0; b < max_blocks; ++b) {
            if (layout == 0) {
                a = 0.97f * a + 0.25f * nd(rng);                  // slow drift: neighbouring blocks score alike
                const float amp = (rng() % 997 == 0) ? 4.f : a;   // isolated needles
                for (int d = 0; d < 128; ++d) pooled[(size_t) (b * 128 + d)] = 0.5f * amp * dir[d] + 0.6f * nd(rng);
            } else {
                const float amp = nd(rng);
                for (int d = 0; d < 128; ++d) pooled[(size_t) (b * 128 + d)] = 0.5f * amp * dir[d] + nd(rng);
            }
        }
        for (auto& x : dead) x = nd(rng);
        for (int64_t i = 0; i < nq; ++i)
            for (int d = 0; d < 512; ++d) q[(size_t) (i * 512 + d)] = 0.2f * dir[d % 128] + 0.1f * nd(rng);
        put(d_pooled, pooled);
        put(d_dead, dead);
        put(d_q, q);
        const char* lname = layout == 0 ? "local" : "iid";
        const int64_t ctxs[3] = {4096, 12288, ctx_max};
        for (int64_t ctx : ctxs) {
            std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
            for (int64_t i = 0; i < nq; ++i) {   // qsa_step_fill's arithmetic
                int32_t* st = steps.data() + i * k::kStepCount;
                const int64_t pos = ctx - nq + i;
                st[k::kStepPos] = (int32_t) pos;
                st[k::kStepNKv] = (int32_t) (pos + 1);
                st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
                st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
            }
            put(d_steps, steps);
            auto exact = [&] {
                k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, d_sc, nullptr, -1);
                k::qsa_block_topk_ref(d_sc, d_steps, nq, max_blocks, cap, s, d_ref, nullptr);
            };
            exact();
            ck(cudaDeviceSynchronize(), "exact");
            const std::vector<int32_t> ref = get(d_ref, (size_t) (nq * cap));
            auto width_of = [&](int64_t i) { return (int64_t) steps[(size_t) (i * k::kStepCount + k::kStepWidth)]; };
            auto nkv_of = [&](int64_t i) { return (int64_t) steps[(size_t) (i * k::kStepCount + k::kStepNKv)]; };
            // ---- partial top-k: the same ids
            if (!k::qsa_block_topk_partial(d_sc, d_steps, nq, max_blocks, cap, s, d_ids, nullptr)) {
                std::printf("FAIL partial top-k refused (ctx %lld)\n", (long long) ctx);
                ok = false;
            } else {
                ck(cudaDeviceSynchronize(), "partial");
                const std::vector<int32_t> got = get(d_ids, (size_t) (nq * cap));
                int64_t same = 0;
                for (int64_t i = 0; i < nq; ++i)
                    same += std::equal(ref.begin() + i * cap, ref.begin() + i * cap + width_of(i), got.begin() + i * cap);
                const float t_ref = timed([&] { k::qsa_block_topk(d_sc, d_steps, nq, max_blocks, cap, s, d_ids2, nullptr); });
                const float t_par = timed([&] { k::qsa_block_topk_partial(d_sc, d_steps, nq, max_blocks, cap, s, d_ids2, nullptr); });
                std::printf("%s %-5s ctx %6lld: partial top-k identical %lld/%lld; top-k %.1f us -> partial %.1f us\n",
                            same == nq ? "PASS" : "FAIL", lname, (long long) ctx, (long long) same, (long long) nq, t_ref, t_par);
                ok = ok && same == nq;
            }
            // ---- page selection: validity and recall
            for (int pc : {16, 32}) {
                if (!k::qsa_page_select_with(pc, 0, d_pooled, d_dead, d_q, d_steps, nq, max_blocks, cap, s, d_ids, nullptr)) {
                    std::printf("FAIL page select refused (%d-cell pages, ctx %lld)\n", pc, (long long) ctx);
                    ok = false;
                    continue;
                }
                ck(cudaDeviceSynchronize(), "page");
                const std::vector<int32_t> got = get(d_ids, (size_t) (nq * cap));
                int64_t hit = 0, all = 0;
                bool valid = true;
                for (int64_t i = 0; i < nq; ++i) {
                    const int64_t w = width_of(i), n_kv = nkv_of(i);
                    std::vector<int32_t> x(ref.begin() + i * cap, ref.begin() + i * cap + w);
                    std::vector<int32_t> y(got.begin() + i * cap, got.begin() + i * cap + w);
                    for (int64_t j = 0; j < w; ++j)
                        if (y[(size_t) j] < 0 || y[(size_t) j] >= n_kv || (j > 0 && y[(size_t) j] <= y[(size_t) j - 1])) valid = false;
                    std::vector<int32_t> both;
                    std::set_intersection(x.begin(), x.end(), y.begin(), y.end(), std::back_inserter(both));
                    hit += (int64_t) both.size();
                    all += w;
                }
                const double recall = all ? (double) hit / (double) all : 1.0;
                // every page kept (default keep = 4x the width in pages): the candidates are every block
                const int64_t keep_cells = ((4 * cap + pc - 1) / pc) * pc;
                const bool must_match = ctx <= keep_cells;
                const bool pass = valid && (!must_match || recall >= 0.999);
                const float t_ex = timed(exact);
                const float t_pg = timed([&] {
                    k::qsa_page_select_with(pc, 0, d_pooled, d_dead, d_q, d_steps, nq, max_blocks, cap, s, d_ids2, nullptr);
                });
                std::printf("%s %-5s ctx %6lld: page select %2d-cell pages recall %.4f%s%s; scores+top-k %.1f us -> page %.1f us\n",
                            pass ? "PASS" : "FAIL", lname, (long long) ctx, pc, recall, valid ? "" : " INVALID SELECTION",
                            must_match ? " (all pages kept: must be exact)" : "", t_ex, t_pg);
                ok = ok && pass;
            }
            // ---- the fingerprint refresh: rewrite 64 blocks in place, then the same pointer vs a fresh buffer
            if (ctx == ctx_max) {
                std::vector<float> patch(64 * 128);
                for (auto& x : patch) x = 2.f * nd(rng) + 0.8f;
                const int64_t b0 = (ctx / s.idx_block) / 3;
                ck(cudaMemcpy(d_pooled + b0 * 128, patch.data(), patch.size() * 4, cudaMemcpyHostToDevice), "patch");
                ck(cudaMemcpy(d_pooled2, d_pooled, (size_t) (max_blocks * 128) * 4, cudaMemcpyDeviceToDevice), "copy");
                k::qsa_page_select_with(16, 0, d_pooled, d_dead, d_q, d_steps, nq, max_blocks, cap, s, d_ids, nullptr);
                k::qsa_page_select_with(16, 0, d_pooled2, d_dead, d_q, d_steps, nq, max_blocks, cap, s, d_ids2, nullptr);
                ck(cudaDeviceSynchronize(), "refresh");
                const std::vector<int32_t> a1 = get(d_ids, (size_t) (nq * cap)), a2 = get(d_ids2, (size_t) (nq * cap));
                bool same = true;
                for (int64_t i = 0; i < nq; ++i)
                    same = same && std::equal(a1.begin() + i * cap, a1.begin() + i * cap + width_of(i), a2.begin() + i * cap);
                std::printf("%s %-5s ctx %6lld: page summaries after an in-place rewrite equal a fresh buffer's\n",
                            same ? "PASS" : "FAIL", lname, (long long) ctx);
                ok = ok && same;
                put(d_pooled, pooled);   // the next layout starts from its own keys
            }
        }
    }
    std::printf("%s\n", ok ? "qsa_page_select_recall: PASS" : "qsa_page_select_recall: FAIL");
    return ok ? 0 : 1;
}
