// STRATA_ROUTE_PRIOR's top-10 kernel (route_prior_top10) on degenerate and ordinary router logits (GPU, synthetic).
//
// Upstream PR #5655: with every logit NaN or -inf the kernel leaked the sentinel id 1 << 30 and wrote past its score
// array (serving aborted every ~5 min under the turbo presets).  Checks: every row returns 10 distinct ids in [0, 512)
// and finite weights summing to 1; on finite logits the ids and weights match a host reference (selection on the
// biased logits, lower id first on ties; weights = the router's softmax renormalised over the 10 picked).
#include "strata/kernels/route_prior.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <vector>

namespace {
constexpr int kE = 512, kK = 10;
int checks = 0;
void check(bool ok, const char* label, int row) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s (row %d)\n", label, row); std::exit(1); }
}
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

void reference(const float* lg, const std::vector<int32_t>& res, float lambda, int32_t* ids, float* w) {
    std::vector<int> order(kE);
    std::iota(order.begin(), order.end(), 0);
    auto score = [&](int e) { return lg[e] + (res[e] >= 0 ? lambda : 0.0f); };
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score(a) > score(b); });
    const float mx = *std::max_element(lg, lg + kE);
    double sum = 0.0;
    for (int k = 0; k < kK; ++k) sum += std::exp(lg[order[k]] - mx);
    for (int k = 0; k < kK; ++k) { ids[k] = order[k]; w[k] = (float) (std::exp(lg[order[k]] - mx) / sum); }
}

// STRATA_ROUTE_PRIOR_ERANK=x: the router's first k = ceil(x exp(H)) picks (its own order, lower id first on ties) kept,
// the rest by the biased score.  Returns false when x exp(H) is within 1e-3 of an integer (the GPU's expf / logf may
// round to the other side; the row's ids are then not compared).
bool reference_erank(const float* lg, const std::vector<int32_t>& res, float lambda, float x, int32_t* ids, float* w) {
    std::vector<int> own(kE);
    std::iota(own.begin(), own.end(), 0);
    std::stable_sort(own.begin(), own.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    const float mx = *std::max_element(lg, lg + kE);
    double qs = 0.0, ql = 0.0;
    for (int k = 0; k < kK; ++k) { const double q = std::exp((double) lg[own[k]] - mx); qs += q; ql += q * std::log(q); }
    const double h = std::max(0.0, std::log(qs) - ql / qs), kf = x * std::exp(h);
    const bool clear = std::fabs(kf - std::round(kf)) > 1e-3;
    const int keep = std::min(10, std::max(1, (int) std::ceil(kf)));
    std::vector<int> order(own.begin(), own.begin() + keep), rest(kE);
    std::iota(rest.begin(), rest.end(), 0);
    auto score = [&](int e) { return lg[e] + (res[e] >= 0 ? lambda : 0.0f); };
    std::stable_sort(rest.begin(), rest.end(), [&](int a, int b) { return score(a) > score(b); });
    for (int e : rest) {
        if ((int) order.size() == kK) break;
        if (std::find(order.begin(), order.end(), e) == order.end()) order.push_back(e);
    }
    double sum = 0.0;
    for (int k = 0; k < kK; ++k) sum += std::exp(lg[order[k]] - mx);
    for (int k = 0; k < kK; ++k) { ids[k] = order[k]; w[k] = (float) (std::exp(lg[order[k]] - mx) / sum); }
    return clear;
}
}  // namespace

int main() {
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    std::mt19937 rng(5655);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    // rows: 0 all NaN, 1 all -inf, 2 NaN except 3 finite, 3 all equal, 4.. random finite
    const int n_tok = 12;
    std::vector<float> lg((size_t) n_tok * kE);
    for (int t = 0; t < n_tok; ++t)
        for (int e = 0; e < kE; ++e) {
            float v = nd(rng);
            if (t == 0) v = nan;
            if (t == 1) v = -inf;
            if (t == 2) v = (e == 7 || e == 300 || e == 511) ? v : nan;
            if (t == 3) v = 0.25f;
            lg[(size_t) t * kE + e] = v;
        }
    std::vector<int32_t> res(kE);
    for (int e = 0; e < kE; ++e) res[e] = (e % 5 == 0) ? e : -1;   // every fifth expert "in VRAM"
    const float lambda = 0.5f;

    float* d_lg = nullptr; int32_t* d_res = nullptr; int32_t* d_ids = nullptr; float* d_w = nullptr;
    ck(cudaMalloc(&d_lg, lg.size() * sizeof(float)), "malloc");
    ck(cudaMalloc(&d_res, kE * sizeof(int32_t)), "malloc");
    ck(cudaMalloc(&d_ids, n_tok * kK * sizeof(int32_t)), "malloc");
    ck(cudaMalloc(&d_w, n_tok * kK * sizeof(float)), "malloc");
    ck(cudaMemcpy(d_lg, lg.data(), lg.size() * sizeof(float), cudaMemcpyHostToDevice), "up");
    ck(cudaMemcpy(d_res, res.data(), kE * sizeof(int32_t), cudaMemcpyHostToDevice), "up");
    strata::kernels::route_prior_top10(d_lg, d_res, lambda, d_ids, d_w, n_tok, nullptr);
    ck(cudaGetLastError(), "launch");
    ck(cudaDeviceSynchronize(), "sync");
    std::vector<int32_t> ids(n_tok * kK);
    std::vector<float> w(n_tok * kK);
    ck(cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "down");
    ck(cudaMemcpy(w.data(), d_w, w.size() * sizeof(float), cudaMemcpyDeviceToHost), "down");

    for (int t = 0; t < n_tok; ++t) {
        std::set<int> seen;
        double ws = 0.0;
        for (int k = 0; k < kK; ++k) {
            const int id = ids[t * kK + k];
            check(id >= 0 && id < kE, "id in [0, 512)", t);
            seen.insert(id);
            check(std::isfinite(w[t * kK + k]) && w[t * kK + k] >= 0.0f, "finite weight", t);
            ws += w[t * kK + k];
        }
        check((int) seen.size() == kK, "10 distinct ids", t);
        check(std::fabs(ws - 1.0) < 1e-4, "weights sum to 1", t);
        if (t == 2) {   // the three finite logits come first (any NaN ranks below them)
            std::set<int> top(ids.begin() + t * kK, ids.begin() + t * kK + 3);
            check(top == std::set<int>({7, 300, 511}), "finite experts picked first", t);
        }
        if (t >= 3) {
            int32_t rid[kK]; float rw[kK];
            reference(&lg[(size_t) t * kE], res, lambda, rid, rw);
            for (int k = 0; k < kK; ++k) {
                check(ids[t * kK + k] == rid[k], "id matches the host reference", t);
                check(std::fabs(w[t * kK + k] - rw[k]) < 1e-5f, "weight matches the host reference", t);
            }
        }
    }
    // STRATA_ROUTE_PRIOR_ERANK (AcceptMoE's effective-rank utility set, LEADS.md): x = 1 and x = 0.5, then off again
    for (const float x : {1.0f, 0.5f}) {
        strata::kernels::route_prior_set_erank(x);
        strata::kernels::route_prior_top10(d_lg, d_res, lambda, d_ids, d_w, n_tok, nullptr);
        ck(cudaGetLastError(), "launch");
        ck(cudaDeviceSynchronize(), "sync");
        ck(cudaMemcpy(ids.data(), d_ids, ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost), "down");
        ck(cudaMemcpy(w.data(), d_w, w.size() * sizeof(float), cudaMemcpyDeviceToHost), "down");
        for (int t = 0; t < n_tok; ++t) {
            std::set<int> seen(ids.begin() + t * kK, ids.begin() + (t + 1) * kK);
            check((int) seen.size() == kK && *seen.begin() >= 0 && *seen.rbegin() < kE, "erank: 10 distinct ids", t);
            if (t < 4) continue;   // rows 0-3: degenerate or all equal (checked above without the set size)
            int32_t rid[kK]; float rw[kK];
            if (!reference_erank(&lg[(size_t) t * kE], res, lambda, x, rid, rw)) continue;
            for (int k = 0; k < kK; ++k) {
                check(ids[t * kK + k] == rid[k], "erank: id matches the host reference", t);
                check(std::fabs(w[t * kK + k] - rw[k]) < 1e-5f, "erank: weight matches the host reference", t);
            }
        }
    }
    strata::kernels::route_prior_set_erank(0.0f);
    for (void* p : {(void*) d_lg, (void*) d_res, (void*) d_ids, (void*) d_w}) ck(cudaFree(p), "free");
    std::printf("route_prior_test: %d checks passed\n", checks);
    return 0;
}
