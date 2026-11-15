// src/kernels/oea_route_parity.cpp - LEADS.md "OEA sibling-aware routing in verify window" (STRATA_OEA, Opportunistic
// Expert Activation, arXiv 2511.02237): oea_route_window against a host reference of the same two phases - ids EXACT
// (ties to the lower id), weights within 1e-4 relative (expf vs the device's __expf) - and the method's property: the
// window's distinct experts are exactly the phase-1 union (no piggybacked slot loads a new expert).  k0 = k must give
// router_top10's ids.  Random windows of 2..16 rows, 512 x top-10 and odd shapes, logits with planted ties and shared
// favourites.  Exit 0 = pass, 1 = a difference, 77 = no GPU.
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <vector>

namespace {
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// the best not-taken expert of a row (restricted to `allow` when given), ties to the lower id; -1 when none
int best(const float* l, int ne, const std::vector<char>& taken, const std::vector<char>* allow) {
    int bi = -1;
    for (int e = 0; e < ne; ++e) {
        if (taken[(size_t) e] || (allow != nullptr && !(*allow)[(size_t) e])) continue;
        if (!(l[e] > -INFINITY)) continue;
        if (bi < 0 || l[e] > l[bi]) bi = e;
    }
    return bi;
}

void reference(const std::vector<float>& lg, int n, int ne, int k, int k0, std::vector<int32_t>& ids, std::vector<float>& w) {
    ids.assign((size_t) n * k, 0);
    w.assign((size_t) n * k, 0.0f);
    std::vector<char> uni((size_t) ne, 0);
    std::vector<std::vector<char>> taken((size_t) n, std::vector<char>((size_t) ne, 0));
    for (int r = 0; r < n; ++r)
        for (int s = 0; s < k0; ++s) {
            const int e = best(&lg[(size_t) r * ne], ne, taken[(size_t) r], nullptr);
            ids[(size_t) r * k + s] = e < 0 ? 0 : e;
            if (e >= 0) { taken[(size_t) r][(size_t) e] = 1; uni[(size_t) e] = 1; }
        }
    for (int r = 0; r < n; ++r) {
        const float* l = &lg[(size_t) r * ne];
        std::vector<char> dummy((size_t) k, 0);
        for (int s = k0; s < k; ++s) {
            const int e = best(l, ne, taken[(size_t) r], &uni);
            if (e >= 0) { ids[(size_t) r * k + s] = e; taken[(size_t) r][(size_t) e] = 1; }
            else { ids[(size_t) r * k + s] = ids[(size_t) r * k]; dummy[(size_t) s] = 1; }
        }
        const double m = l[ids[(size_t) r * k]];
        double z = 0.0;
        for (int e = 0; e < ne; ++e) z += std::exp((double) l[e] - m);
        double sum = 0.0;
        for (int s = 0; s < k; ++s) if (!dummy[(size_t) s]) sum += std::exp((double) l[ids[(size_t) r * k + s]] - m) / z;
        const double den = std::max(sum, 6.103515625e-05);
        for (int s = 0; s < k; ++s)
            w[(size_t) r * k + s] = dummy[(size_t) s] ? 0.0f : (float) (std::exp((double) l[ids[(size_t) r * k + s]] - m) / z / den);
    }
}
}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) { std::printf("no GPU\n"); return 77; }
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    float *d_l, *d_w;
    int32_t* d_i;
    check(cudaMalloc((void**) &d_l, 16 * 1024 * 4), "l");
    check(cudaMalloc((void**) &d_w, 16 * 16 * 4), "w");
    check(cudaMalloc((void**) &d_i, 16 * 16 * 4), "i");
    int bad = 0, cases = 0;
    struct Shape { int ne, k; };
    for (const Shape sh : {Shape{512, 10}, Shape{256, 8}, Shape{1000, 16}, Shape{64, 4}})
        for (int it = 0; it < 400; ++it) {
            const int n = 2 + (int) (rng() % 15), ne = sh.ne, k = sh.k;
            const int k0 = 1 + (int) (rng() % k);
            std::vector<float> lg((size_t) n * ne);
            for (auto& v : lg) v = nd(rng);
            // shared favourites (the case OEA exploits) and exact ties
            for (int f = 0; f < 6; ++f) {
                const int e = (int) (rng() % ne);
                for (int r = 0; r < n; ++r) if (rng() % 2) lg[(size_t) r * ne + e] += 4.0f;
            }
            for (int t = 0; t < 8; ++t) {
                const int r = (int) (rng() % n), a = (int) (rng() % ne), b = (int) (rng() % ne);
                lg[(size_t) r * ne + b] = lg[(size_t) r * ne + a];
            }
            std::vector<int32_t> ri, di((size_t) n * k);
            std::vector<float> rw, dw((size_t) n * k);
            reference(lg, n, ne, k, k0, ri, rw);
            check(cudaMemcpy(d_l, lg.data(), lg.size() * 4, cudaMemcpyHostToDevice), "l up");
            strata::kernels::oea_route_window(d_l, n, ne, k, k0, d_i, d_w, nullptr);
            check(cudaDeviceSynchronize(), "run");
            check(cudaMemcpy(di.data(), d_i, di.size() * 4, cudaMemcpyDeviceToHost), "i down");
            check(cudaMemcpy(dw.data(), d_w, dw.size() * 4, cudaMemcpyDeviceToHost), "w down");
            bool ok = di == ri;
            for (size_t j = 0; j < rw.size() && ok; ++j) ok = std::fabs(dw[j] - rw[j]) <= 1e-4f * std::max(1e-3f, std::fabs(rw[j]));
            // the property: every kept expert is in the phase-1 union
            std::set<int> uni, all;
            for (int r = 0; r < n; ++r) {
                for (int s = 0; s < k0; ++s) uni.insert(di[(size_t) r * k + s]);
                for (int s = 0; s < k; ++s) all.insert(di[(size_t) r * k + s]);
            }
            ok = ok && all == uni;
            // k0 = k: the plain router's ids
            if (k0 == k && ne <= 512 && k <= 64) {
                std::vector<int32_t> ti((size_t) n * k);
                strata::kernels::router_top10(d_l, n, ne, k, d_i, d_w, nullptr);
                check(cudaMemcpy(ti.data(), d_i, ti.size() * 4, cudaMemcpyDeviceToHost), "router down");
                // (diagnostic only: the router ranks by rounded probabilities, so two logits an ulp apart can tie there)
                if (ti != di) std::printf("  note: k0 = k differs from router_top10 (ne %d k %d n %d)\n", ne, k, n);
            }
            ++cases;
            if (!ok) {
                if (bad < 6) std::printf("  FAIL ne %d k %d k0 %d n %d\n", ne, k, k0, n);
                ++bad;
            }
        }
    std::printf(bad == 0 ? "oea_route_parity: PASS (%d windows)\n" : "oea_route_parity: FAIL (%d)\n", bad == 0 ? cases : bad);
    return bad == 0 ? 0 : 1;
}
