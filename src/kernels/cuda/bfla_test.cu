// src/kernels/cuda/bfla_test.cu - BFLA block-sparse prompt attention (bfla.cu): GPU, synthetic, no model.
//
//   bfla_test                      correctness (exit 0 pass, 1 fail, 77 no gfx11 device)
//   bfla_test --bench N [T]        one attention layer of an N-token cold prompt in chunks of T (4096): the qwen35 engine's
//                                  materialised hipBLAS path, the fused kernel over every block, and BFLA sparse
//
// Checks:
//   1. dense mode (every causal block) vs an FP64 reference from the same FP16 inputs, and vs the engine's GEMM path
//   2. sparse mode with every block kept (eta 1) == dense mode, bitwise (same kernel, same block order)
//   3. sparse selection lists vs a host reimplementation of the rule (FP64 scores; a differing pick must be a near tie)
//   4. sparse attention vs the FP64 reference over exactly the selected blocks (must match like 1)
//   5. sparse vs dense FP64 (lossy: reported, loose bound on structured data), tail rule (full_from) honoured
#include "strata/kernels/bfla.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace strata::kernels;

namespace {

#define CK(x)                                                                                          \
    do {                                                                                               \
        cudaError_t e_ = (x);                                                                          \
        if (e_ != cudaSuccess) {                                                                       \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_));    \
            std::exit(1);                                                                              \
        }                                                                                              \
    } while (0)

uint16_t f2h(float f) {   // round to nearest even
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffffu;
    if (e >= 31) return (uint16_t) (sign | 0x7c00u);
    if (e <= 0) {
        if (e < -10) return (uint16_t) sign;
        m |= 0x800000u;
        const int shift = 14 - e;
        uint32_t r = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1), halfv = 1u << (shift - 1);
        if (rem > halfv || (rem == halfv && (r & 1))) ++r;
        return (uint16_t) (sign | r);
    }
    uint32_t r = ((uint32_t) e << 10) | (m >> 13);
    const uint32_t rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1))) ++r;
    return (uint16_t) (sign | r);
}
float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ffu, x;
    if (e == 0) {
        if (m == 0) x = sign;
        else {
            e = 127 - 15 + 1;
            while (!(m & 0x400u)) { m <<= 1; --e; }
            x = sign | (e << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (e == 31) x = sign | 0x7f800000u | (m << 13);
    else x = sign | ((e - 15 + 127) << 23) | (m << 13);
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

constexpr int D = kBflaDim;

struct Data {
    int H = 16, HKV = 2;
    int64_t T = 0, p0 = 0, P = 0;
    std::vector<uint16_t> q, k, v;   // q [T][H][D], k / v [HKV][P][D]
};

// Structured data: each 64-key block carries one of `topics` directions; a span of queries looks for one topic of its
// past, so real attention is concentrated on a few blocks (as BFLA assumes) but not one-hot.
Data make_data(int64_t T, int64_t p0, uint32_t seed, bool structured) {
    Data d;
    d.T = T;
    d.p0 = p0;
    d.P = p0 + T;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int topics = 24;
    std::vector<float> tv((size_t) topics * D);
    for (auto& x : tv) x = nd(rng);
    const int64_t nblk = (d.P + 63) / 64;
    std::vector<int> btopic((size_t) nblk);
    for (auto& t : btopic) t = (int) (rng() % topics);
    d.k.resize((size_t) d.HKV * d.P * D);
    d.v.resize(d.k.size());
    for (int g = 0; g < d.HKV; ++g)
        for (int64_t c = 0; c < d.P; ++c)
            for (int i = 0; i < D; ++i) {
                const float base = structured ? 0.9f * tv[(size_t) btopic[(size_t) (c / 64)] * D + i] : 0.f;
                d.k[((size_t) g * d.P + c) * D + i] = f2h(base + nd(rng));
                d.v[((size_t) g * d.P + c) * D + i] = f2h(nd(rng));
            }
    d.q.resize((size_t) T * d.H * D);
    // a query's target topic holds for a span of 32 positions (neighbouring queries look for the same thing, the
    // locality BFLA's tile-shared selection relies on); each span picks it from a block of its past
    int tgt = 0;
    for (int64_t t = 0; t < T; ++t) {
        const int64_t pos = p0 + t;
        if (t == 0 || pos % 32 == 0) tgt = btopic[(size_t) (rng() % (uint32_t) (pos / 64 + 1))];
        for (int h = 0; h < d.H; ++h)
            for (int i = 0; i < D; ++i) {
                const float base = structured ? 0.45f * tv[(size_t) tgt * D + i] : 0.f;
                d.q[((size_t) t * d.H + h) * D + i] = f2h(base + nd(rng));
            }
    }
    return d;
}

// FP64 attention of query (t, h) over the cells `cells` (all <= its position).
void ref_row(const Data& d, int64_t t, int h, const std::vector<int64_t>& cells, double* out) {
    const int g = h / (d.H / d.HKV);
    const uint16_t* q = &d.q[((size_t) t * d.H + h) * D];
    std::vector<double> s(cells.size());
    double mx = -1e300;
    for (size_t n = 0; n < cells.size(); ++n) {
        const uint16_t* k = &d.k[((size_t) g * d.P + cells[n]) * D];
        double a = 0;
        for (int i = 0; i < D; ++i) a += (double) h2f(q[i]) * (double) h2f(k[i]);
        s[n] = a / 16.0;
        mx = std::max(mx, s[n]);
    }
    double den = 0;
    for (auto& x : s) den += (x = std::exp(x - mx));
    for (int i = 0; i < D; ++i) out[i] = 0;
    for (size_t n = 0; n < cells.size(); ++n) {
        const uint16_t* v = &d.v[((size_t) g * d.P + cells[n]) * D];
        const double w = s[n] / den;
        for (int i = 0; i < D; ++i) out[i] += w * (double) h2f(v[i]);
    }
}
// The kept mass of dense attention for (t, h) on the cell set.
double kept_mass(const Data& d, int64_t t, int h, const std::vector<char>& keep_block) {
    const int g = h / (d.H / d.HKV);
    const uint16_t* q = &d.q[((size_t) t * d.H + h) * D];
    const int64_t pos = d.p0 + t;
    std::vector<double> s((size_t) pos + 1);
    double mx = -1e300;
    for (int64_t c = 0; c <= pos; ++c) {
        const uint16_t* k = &d.k[((size_t) g * d.P + c) * D];
        double a = 0;
        for (int i = 0; i < D; ++i) a += (double) h2f(q[i]) * (double) h2f(k[i]);
        s[(size_t) c] = a / 16.0;
        mx = std::max(mx, s[(size_t) c]);
    }
    double all = 0, kept = 0;
    for (int64_t c = 0; c <= pos; ++c) {
        const double e = std::exp(s[(size_t) c] - mx);
        all += e;
        if (keep_block[(size_t) (c / 64)]) kept += e;
    }
    return kept / all;
}

struct Dev {
    uint16_t *q = nullptr, *k = nullptr, *v = nullptr;
    float* out = nullptr;
    void* scratch = nullptr;
    size_t scratch_bytes = 0;
};
Dev upload(const Data& d) {
    Dev x;
    CK(cudaMalloc((void**) &x.q, d.q.size() * 2));
    CK(cudaMalloc((void**) &x.k, d.k.size() * 2 + 4096));
    CK(cudaMalloc((void**) &x.v, d.v.size() * 2 + 4096));
    CK(cudaMalloc((void**) &x.out, (size_t) d.T * d.H * D * 4));
    CK(cudaMemcpy(x.q, d.q.data(), d.q.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(x.k, d.k.data(), d.k.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(x.v, d.v.data(), d.v.size() * 2, cudaMemcpyHostToDevice));
    x.scratch_bytes = bfla_scratch_bytes(d.T, d.P, d.H, d.HKV);
    CK(cudaMalloc(&x.scratch, x.scratch_bytes));
    return x;
}
void release(Dev& x) {
    cudaFree(x.q);
    cudaFree(x.k);
    cudaFree(x.v);
    cudaFree(x.out);
    cudaFree(x.scratch);
}
std::vector<float> run(const Data& d, Dev& x, const BflaConfig& c, int64_t full_from) {
    CK(cudaMemset(x.out, 0xff, (size_t) d.T * d.H * D * 4));   // NaN: a row left unwritten shows
    if (!bfla_prefill_attn(x.q, x.k, x.v, d.T, d.p0, d.H, d.HKV, c, full_from, x.scratch, x.scratch_bytes, x.out, nullptr)) {
        std::fprintf(stderr, "bfla_prefill_attn refused\n");
        std::exit(1);
    }
    CK(cudaDeviceSynchronize());
    std::vector<float> o((size_t) d.T * d.H * D);
    CK(cudaMemcpy(o.data(), x.out, o.size() * 4, cudaMemcpyDeviceToHost));
    return o;
}

struct Err {
    double max_abs = 0, max_ref = 0, rel_l2 = 0;
    bool finite = true;
};
// GPU output vs FP64 reference rows over the cells chosen by `cells_of(t, h)`; `stride` subsamples the queries.
template <class F>
Err compare(const Data& d, const std::vector<float>& o, F cells_of, int64_t stride) {
    Err e;
    double num = 0, den = 0;
    std::vector<double> r(D);
    for (int64_t t = 0; t < d.T; t += stride)
        for (int h = 0; h < d.H; ++h) {
            ref_row(d, t, h, cells_of(t, h), r.data());
            for (int i = 0; i < D; ++i) {
                const double g = o[((size_t) t * d.H + h) * D + i];
                if (!std::isfinite(g)) e.finite = false;
                e.max_abs = std::max(e.max_abs, std::fabs(g - r[i]));
                e.max_ref = std::max(e.max_ref, std::fabs(r[i]));
                num += (g - r[i]) * (g - r[i]);
                den += r[i] * r[i];
            }
        }
    e.rel_l2 = std::sqrt(num / std::max(den, 1e-300));
    return e;
}

// Host copy of the selection rule (FP64 features and scores).
struct HostSel {
    std::vector<std::vector<int>> lists;   // [tile * HKV + g]
    std::vector<std::vector<double>> score;
};
HostSel host_select(const Data& d, const BflaConfig& c, int64_t full_from) {
    const int G = d.H / d.HKV, TT = kBflaTileRows / G;
    const int64_t nblk = (d.P + 63) / 64, n_tiles = (d.T + TT - 1) / TT;
    auto feat = [](const double* x) {   // x: 256 values (already the head mean of normalised heads for q)
        std::vector<double> r(kBflaFeat);
        double ss = 0;
        for (int i = 0; i < kBflaFeat; ++i) {
            r[i] = x[2 * i] + x[2 * i + 1];
            ss += r[i] * r[i];
        }
        const double inv = 1.0 / std::sqrt(ss / kBflaFeat + 1e-6);
        for (auto& v : r) v *= inv;
        return r;
    };
    auto normed = [](const uint16_t* x, double* o) {
        double ss = 0;
        for (int i = 0; i < D; ++i) ss += (double) h2f(x[i]) * h2f(x[i]);
        const double inv = 1.0 / std::sqrt(ss / D + 1e-6);
        for (int i = 0; i < D; ++i) o[i] = h2f(x[i]) * inv;
    };
    std::vector<double> mu((size_t) d.HKV * nblk * kBflaFeat, 0.0), phi((size_t) d.T * d.HKV * kBflaFeat);
    std::vector<double> tmp(D);
    for (int g = 0; g < d.HKV; ++g)
        for (int64_t j = 0; j < nblk; ++j) {
            const int64_t n = std::min<int64_t>(64, d.P - j * 64);
            for (int64_t c0 = 0; c0 < n; ++c0) {
                normed(&d.k[((size_t) g * d.P + j * 64 + c0) * D], tmp.data());
                const auto f = feat(tmp.data());
                for (int i = 0; i < kBflaFeat; ++i) mu[((size_t) g * nblk + j) * kBflaFeat + i] += f[i] / (double) n;
            }
        }
    std::vector<double> acc(D);
    for (int64_t t = 0; t < d.T; ++t)
        for (int g = 0; g < d.HKV; ++g) {
            std::fill(acc.begin(), acc.end(), 0.0);
            for (int hh = 0; hh < G; ++hh) {
                normed(&d.q[((size_t) t * d.H + g * G + hh) * D], tmp.data());
                for (int i = 0; i < D; ++i) acc[i] += tmp[i] / G;
            }
            const auto f = feat(acc.data());
            std::copy(f.begin(), f.end(), phi.begin() + ((size_t) t * d.HKV + g) * kBflaFeat);
        }
    HostSel hs;
    hs.lists.resize((size_t) n_tiles * d.HKV);
    hs.score.resize(hs.lists.size());
    for (int64_t tile = 0; tile < n_tiles; ++tile)
        for (int g = 0; g < d.HKV; ++g) {
            const int64_t ta = tile * TT, tb = std::min<int64_t>(d.T, ta + TT) - 1;
            const int64_t qa = d.p0 + ta, qb = d.p0 + tb;
            const int n_vis = (int) (qb / 64) + 1;
            const int64_t k = bfla_budget_blocks(qb + 1, c);
            const int sinkc = std::min(c.sink, n_vis);
            const int lo_b = std::max<int>(sinkc, (int) (qa / 64) - c.local);
            const int nc = lo_b - sinkc;
            const int64_t fq = k - (n_vis - nc);
            auto& L = hs.lists[(size_t) tile * d.HKV + g];
            auto& S = hs.score[(size_t) tile * d.HKV + g];
            S.assign((size_t) n_vis, 0.0);
            if (qb >= full_from || k >= n_vis || fq >= nc) {
                for (int j = 0; j < n_vis; ++j) L.push_back(j);
                continue;
            }
            std::vector<std::pair<double, int>> cand;
            for (int j = sinkc; j < lo_b; ++j) {
                double mx = -1e300;
                std::vector<double> u;
                for (int64_t t = ta; t <= tb; ++t) {
                    double dd = 0;
                    for (int i = 0; i < kBflaFeat; ++i)
                        dd += phi[((size_t) t * d.HKV + g) * kBflaFeat + i] * mu[((size_t) g * nblk + j) * kBflaFeat + i];
                    u.push_back(std::max(dd, 0.0) / std::sqrt((double) kBflaFeat));
                    mx = std::max(mx, u.back());
                }
                double s = 0;
                for (double x : u) s += std::exp(x - mx);
                S[(size_t) j] = mx + std::log(s);
                cand.push_back({-S[(size_t) j], j});
            }
            std::vector<char> keep((size_t) n_vis, 0);
            for (int j = 0; j < sinkc; ++j) keep[(size_t) j] = 1;
            for (int j = lo_b; j < n_vis; ++j) keep[(size_t) j] = 1;
            if (fq > 0) {
                std::sort(cand.begin(), cand.end());
                for (int64_t i = 0; i < fq; ++i) keep[(size_t) cand[(size_t) i].second] = 1;
            }
            for (int j = 0; j < n_vis; ++j)
                if (keep[(size_t) j]) L.push_back(j);
        }
    return hs;
}

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "pass" : "FAIL", what);
    if (!ok) ++g_fail;
}

int correctness() {
    // ---- 1. dense mode vs FP64 (p0 = 0 and a chunk after a prefix; T not a multiple of the tiles)
    for (auto [T, p0] : {std::pair<int64_t, int64_t>{700, 0}, {520, 1500}}) {
        const Data d = make_data(T, p0, 11u + (uint32_t) T, true);
        Dev x = upload(d);
        BflaConfig c;
        c.mode = 2;
        const auto o = run(d, x, c, std::numeric_limits<int64_t>::max());
        const Err e = compare(d, o, [&](int64_t t, int) {
            std::vector<int64_t> cells;
            for (int64_t cc = 0; cc <= d.p0 + t; ++cc) cells.push_back(cc);
            return cells;
        }, 7);
        std::printf("dense T=%lld p0=%lld: max |err| %.3e (max |ref| %.3f), rel L2 %.3e\n", (long long) T, (long long) p0,
                    e.max_abs, e.max_ref, e.rel_l2);
        check(e.finite && e.rel_l2 < 3e-3 && e.max_abs < 5e-3 * std::max(1.0, e.max_ref), "dense mode matches FP64 (FP16-operand bound)");
        release(x);
    }
    // ---- 1b. dense mode vs the engine's materialised path (FP16 GEMM -> FP32 scores -> FP16 probabilities -> GEMM)
    //       is covered by --bench (it prints the max difference); here the FP64 reference is the stricter check.

    // ---- 2-5. sparse
    const int64_t T = 1000, p0 = 6000;
    const Data d = make_data(T, p0, 7u, true);
    Dev x = upload(d);
    BflaConfig dense;
    dense.mode = 2;
    const auto od = run(d, x, dense, std::numeric_limits<int64_t>::max());
    BflaConfig all;
    all.mode = 1;
    all.eta = 1.0;
    all.kmin = 1 << 30;
    all.kmax = 1 << 30;
    const auto oa = run(d, x, all, std::numeric_limits<int64_t>::max());
    check(std::memcmp(od.data(), oa.data(), od.size() * 4) == 0, "sparse with every block kept == dense mode, bitwise");
    bfla_set_q_global(1);
    const auto oq = run(d, x, dense, std::numeric_limits<int64_t>::max());
    bfla_set_q_global(-1);
    check(std::memcmp(od.data(), oq.data(), od.size() * 4) == 0, "Q-from-global variant == LDS variant, bitwise");

    BflaConfig sp;
    sp.mode = 1;
    sp.kmin = 256;
    sp.kmax = 1024;   // forces sparsity at this small length: ~16 of ~110 blocks
    const int64_t full_from = d.P - 48;
    const auto os = run(d, x, sp, full_from);
    const BflaLists L = bfla_lists(x.scratch, d.T, d.P, d.H, d.HKV);
    std::vector<int32_t> ids((size_t) L.n_tiles * d.HKV * L.cap), cnt((size_t) L.n_tiles * d.HKV);
    CK(cudaMemcpy(ids.data(), L.ids, ids.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(cnt.data(), L.counts, cnt.size() * 4, cudaMemcpyDeviceToHost));
    const HostSel hs = host_select(d, sp, full_from);
    int same = 0, near_tie = 0, bad = 0;
    double sel_blocks = 0, vis_blocks = 0;
    bool tail_ok = true;
    for (size_t r = 0; r < hs.lists.size(); ++r) {
        const std::vector<int> gpu(ids.begin() + (long) (r * L.cap), ids.begin() + (long) (r * L.cap) + cnt[r]);
        const auto& host = hs.lists[r];
        const int64_t tile = (int64_t) r / d.HKV;
        const int64_t qb = d.p0 + std::min<int64_t>(d.T, (tile + 1) * L.tile_tokens) - 1;
        const int n_vis = (int) (qb / 64) + 1;
        sel_blocks += (double) gpu.size();
        vis_blocks += n_vis;
        if (qb >= full_from && (int) gpu.size() != n_vis) tail_ok = false;
        if (gpu == host) { ++same; continue; }
        // differing picks must be near ties of the FP64 scores (FP32 GPU scores, other summation order)
        std::set<int> a(gpu.begin(), gpu.end()), b(host.begin(), host.end());
        double lo_in = 1e300, hi_out = -1e300;
        bool sizes = a.size() == b.size();
        for (int j : a) if (!b.count(j)) hi_out = std::max(hi_out, hs.score[r][(size_t) j]);
        for (int j : b) if (!a.count(j)) lo_in = std::min(lo_in, hs.score[r][(size_t) j]);
        if (sizes && std::fabs(lo_in - hi_out) < 1e-4 * std::max(1.0, std::fabs(lo_in))) ++near_tie;
        else ++bad;
    }
    std::printf("sparse selection: %d of %zu tile lists identical to the host rule, %d near ties, %d wrong; kept %.1f%% of "
                "visible blocks\n", same, hs.lists.size(), near_tie, bad, 100.0 * sel_blocks / vis_blocks);
    check(bad == 0, "selection lists equal the host rule (up to FP32 near ties)");
    check(tail_ok, "tail tiles (full_from) keep every block");
    // the attention over exactly the GPU's lists
    const Err es = compare(d, os, [&](int64_t t, int h) {
        const int g = h / (d.H / d.HKV);
        const size_t r = (size_t) (t / L.tile_tokens) * d.HKV + g;
        std::vector<int64_t> cells;
        for (int i = 0; i < cnt[r]; ++i)
            for (int64_t cc = (int64_t) ids[r * L.cap + i] * 64; cc < (int64_t) ids[r * L.cap + i] * 64 + 64 && cc <= d.p0 + t; ++cc)
                cells.push_back(cc);
        return cells;
    }, 5);
    std::printf("sparse vs FP64 over the selected blocks: max |err| %.3e, rel L2 %.3e\n", es.max_abs, es.rel_l2);
    check(es.finite && es.rel_l2 < 3e-3 && es.max_abs < 5e-3 * std::max(1.0, es.max_ref), "sparse attention exact on its blocks");
    // lossy: vs dense FP64, and the kept attention mass
    const Err ed = compare(d, os, [&](int64_t t, int) {
        std::vector<int64_t> cells;
        for (int64_t cc = 0; cc <= d.p0 + t; ++cc) cells.push_back(cc);
        return cells;
    }, 9);
    double mass = 0, mass_min = 1;
    int nm = 0;
    for (int64_t t = 0; t < d.T; t += 37)
        for (int h = 0; h < d.H; h += 5) {
            const int g = h / (d.H / d.HKV);
            const size_t r = (size_t) (t / L.tile_tokens) * d.HKV + g;
            std::vector<char> kb((size_t) ((d.P + 63) / 64), 0);
            for (int i = 0; i < cnt[r]; ++i) kb[(size_t) ids[r * L.cap + i]] = 1;
            const double m = kept_mass(d, t, h, kb);
            mass += m;
            mass_min = std::min(mass_min, m);
            ++nm;
        }
    std::printf("sparse vs dense FP64 (lossy): rel L2 %.3e, kept attention mass mean %.4f min %.4f\n", ed.rel_l2, mass / nm, mass_min);
    check(ed.finite && ed.rel_l2 < 0.25, "sparse within the loose lossy bound on structured data (rel L2 < 0.25)");
    release(x);
    return g_fail == 0 ? 0 : 1;
}

// ---- the engine's materialised prompt attention (q36_engine.cu, T > kGemmT), for the benchmark
__global__ void k_softmax16(const float* S, __half* S16, int ld, int T, int P, int pos0, float scale) {
    __shared__ float sh[256];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * T + t) * ld;
    __half* out = S16 + ((size_t) h * T + t) * ld;
    const int n = min(P, pos0 + t + 1);
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    sh[threadIdx.x] = mx;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    mx = sh[0];
    __syncthreads();
    float sum = 0.f;
    for (int p = threadIdx.x; p < n; p += blockDim.x) sum += expf(row[p] * scale - mx);
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float inv = 1.0f / sh[0];
    for (int p = threadIdx.x; p < P; p += blockDim.x) out[p] = __float2half(p < n ? expf(row[p] * scale - mx) * inv : 0.f);
}
struct Gemm {
    cublasHandle_t blas{};
    float* att_s = nullptr;
    __half* att_p16 = nullptr;
    size_t floats = (size_t) 64 << 20;   // kAttScoreFloats
    Gemm() {
        cublasCreate(&blas);
        CK(cudaMalloc((void**) &att_s, floats * 4));
        CK(cudaMalloc((void**) &att_p16, floats * 2));
    }
    ~Gemm() {
        cudaFree(att_s);
        cudaFree(att_p16);
        cublasDestroy(blas);
    }
    // q16 [T][H][D] (chunk), k16 / v16 [HKV][P][D], out [T][H][D]
    void run(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int T, int p0, int H, int HKV, float* out) {
        const int P = p0 + T, ldp = (P + 7) / 8 * 8, G = H / HKV;
        const int tq = (int) std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) (floats / ((size_t) H * ldp))));
        const float one = 1.f, zero = 0.f;
        for (int t0 = 0; t0 < T; t0 += tq) {
            const int nt = std::min(tq, T - t0);
            for (int kvh = 0; kvh < HKV; ++kvh) {
                const __half* Kh = reinterpret_cast<const __half*>(k16) + (size_t) kvh * P * D;
                const __half* Qh = reinterpret_cast<const __half*>(q16) + ((size_t) t0 * H + kvh * G) * D;
                float* Sh = att_s + (size_t) kvh * G * nt * ldp;
                cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, P, nt, D, &one, Kh, CUDA_R_16F, D, 0, Qh, CUDA_R_16F,
                                           H * D, D, &zero, Sh, CUDA_R_32F, ldp, (long long) nt * ldp, G, CUBLAS_COMPUTE_32F,
                                           CUBLAS_GEMM_DEFAULT);
            }
            k_softmax16<<<dim3(nt, H), 256>>>(att_s, att_p16, ldp, nt, P, p0 + t0, 1.0f / 16.0f);
            for (int kvh = 0; kvh < HKV; ++kvh) {
                const __half* Vh = reinterpret_cast<const __half*>(v16) + (size_t) kvh * P * D;
                const __half* Sh = att_p16 + (size_t) kvh * G * nt * ldp;
                float* Oh = out + ((size_t) t0 * H + kvh * G) * D;
                cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, D, nt, P, &one, Vh, CUDA_R_16F, D, 0, Sh, CUDA_R_16F,
                                           ldp, (long long) nt * ldp, &zero, Oh, CUDA_R_32F, H * D, D, G, CUBLAS_COMPUTE_32F,
                                           CUBLAS_GEMM_DEFAULT);
            }
        }
    }
};

int bench(int64_t N, int64_t T) {
    // one attention layer (H 16, HKV 2, D 256) of an N-token cold prompt in chunks of T: K / V of the whole prompt
    // resident, each chunk's queries random (timing does not depend on values; selection cost is data-independent)
    const int H = 16, HKV = 2;
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> hk((size_t) HKV * N * D), hq((size_t) T * H * D);
    for (auto& x : hk) x = f2h(nd(rng));
    for (auto& x : hq) x = f2h(nd(rng));
    // K / V per chunk need the compact [HKV][P][D] layout at that P: keep [HKV][N][D] and pass P = N rows apart would
    // break the layout, so every chunk copies its prefix into the compact buffers (as k_kv_f16 does in the engine)
    uint16_t *kall, *k16, *v16, *q16;
    float* out;
    CK(cudaMalloc((void**) &kall, hk.size() * 2));
    CK(cudaMemcpy(kall, hk.data(), hk.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void**) &k16, hk.size() * 2 + 4096));
    CK(cudaMalloc((void**) &v16, hk.size() * 2 + 4096));
    CK(cudaMalloc((void**) &q16, hq.size() * 2));
    CK(cudaMemcpy(q16, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMalloc((void**) &out, (size_t) T * H * D * 4));
    void* scratch = nullptr;
    const size_t sb = bfla_scratch_bytes(T, N, H, HKV);
    CK(cudaMalloc(&scratch, sb));
    Gemm gm;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    double t_gemm = 0, t_dense = 0, t_dense_qg = 0, t_sparse = 0, t_select = 0;
    const BflaConfig& env = bfla_config();
    BflaConfig sp = env;
    sp.mode = 1;
    BflaConfig dn;
    dn.mode = 2;
    double kept = 0, vis = 0;
    double maxdiff = 0;
    for (int64_t p0 = 0; p0 < N; p0 += T) {
        const int64_t t = std::min(T, N - p0), P = p0 + t;
        for (int g = 0; g < HKV; ++g) {
            CK(cudaMemcpy(k16 + (size_t) g * P * D, kall + (size_t) g * N * D, (size_t) P * D * 2, cudaMemcpyDeviceToDevice));
            CK(cudaMemcpy(v16 + (size_t) g * P * D, kall + (size_t) g * N * D, (size_t) P * D * 2, cudaMemcpyDeviceToDevice));
        }
        float ms = 0;
        const bool gemm_fits = (size_t) H * ((P + 7) / 8 * 8) <= gm.floats;
        if (gemm_fits) {
            cudaEventRecord(e0);
            gm.run(q16, k16, v16, (int) t, (int) p0, H, HKV, out);
            cudaEventRecord(e1);
            CK(cudaEventSynchronize(e1));
            cudaEventElapsedTime(&ms, e0, e1);
            t_gemm += ms;
        }
        std::vector<float> og;
        if (p0 + t == N && gemm_fits) {
            og.resize((size_t) t * H * D);
            CK(cudaMemcpy(og.data(), out, og.size() * 4, cudaMemcpyDeviceToHost));
        }
        bfla_set_q_global(1);   // the Q-from-global variant first (its output is overwritten by the default below)
        cudaEventRecord(e0);
        bfla_prefill_attn(q16, k16, v16, t, p0, H, HKV, dn, INT64_MAX, scratch, sb, out, nullptr);
        cudaEventRecord(e1);
        CK(cudaEventSynchronize(e1));
        cudaEventElapsedTime(&ms, e0, e1);
        t_dense_qg += ms;
        bfla_set_q_global(-1);
        cudaEventRecord(e0);
        bfla_prefill_attn(q16, k16, v16, t, p0, H, HKV, dn, INT64_MAX, scratch, sb, out, nullptr);
        cudaEventRecord(e1);
        CK(cudaEventSynchronize(e1));
        cudaEventElapsedTime(&ms, e0, e1);
        t_dense += ms;
        if (!og.empty()) {
            std::vector<float> od(og.size());
            CK(cudaMemcpy(od.data(), out, od.size() * 4, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < od.size(); ++i) maxdiff = std::max(maxdiff, (double) std::fabs(od[i] - og[i]));
        }
        if (P >= sp.min_len) {
            cudaEventRecord(e0);
            bfla_prefill_attn(q16, k16, v16, t, p0, H, HKV, sp, INT64_MAX, scratch, sb, out, nullptr);
            cudaEventRecord(e1);
            CK(cudaEventSynchronize(e1));
            cudaEventElapsedTime(&ms, e0, e1);
            t_sparse += ms;
            t_select += ms;
            // the selection alone: lists from the same call, the attention again from them
            const BflaLists L = bfla_lists(scratch, t, P, H, HKV);
            std::vector<int32_t> cnt((size_t) L.n_tiles * HKV);
            CK(cudaMemcpy(cnt.data(), L.counts, cnt.size() * 4, cudaMemcpyDeviceToHost));
            for (size_t r = 0; r < cnt.size(); ++r) {
                const int64_t tile = (int64_t) r / HKV;
                kept += cnt[r];
                vis += (double) ((p0 + std::min<int64_t>(t, (tile + 1) * L.tile_tokens) - 1) / 64 + 1);
            }
            cudaEventRecord(e0);
            bfla_attn_from_lists(q16, k16, v16, t, p0, H, HKV, L.ids, L.counts, L.cap, out, nullptr);
            cudaEventRecord(e1);
            CK(cudaEventSynchronize(e1));
            cudaEventElapsedTime(&ms, e0, e1);
            t_select -= ms;   // select = sparse total - attention alone
        } else {
            t_sparse += ms;   // below the threshold the engine keeps its path; count the dense kernel's time here
        }
    }
    std::printf("bench: one attention layer (H 16 / KV 2 / D 256), %lld-token cold prompt in chunks of %lld:\n", (long long) N,
                (long long) T);
    std::printf("  engine path (hipBLAS FP16 GEMMs + materialised softmax): %9.1f ms%s\n", t_gemm,
                t_gemm == 0 ? " (scores did not fit)" : "");
    std::printf("  fused kernel, every block (STRATA_BFLA_DENSE=1):         %9.1f ms  (max |diff| vs engine path, last chunk: %.2e)\n",
                t_dense, maxdiff);
    std::printf("  fused kernel, every block, Q from global (QGLOBAL=1):    %9.1f ms\n", t_dense_qg);
    std::printf("  BFLA sparse from %lld (STRATA_BFLA_SPARSE=1, lossy):     %9.1f ms  (selection %.1f ms; kept %.1f%% of visible blocks)\n",
                (long long) sp.min_len, t_sparse, t_select, vis > 0 ? 100.0 * kept / vis : 100.0);
    std::printf("  x10 layers (Qwen3.6-35B): engine %.2f s, dense %.2f s, sparse %.2f s\n", t_gemm / 100.0, t_dense / 100.0,
                t_sparse / 100.0);
    cudaFree(kall);
    cudaFree(k16);
    cudaFree(v16);
    cudaFree(q16);
    cudaFree(out);
    cudaFree(scratch);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (!bfla_supported(16, 2, 256)) {
        std::printf("bfla_test: no gfx11 device (or not a HIP wave32 build): skipped\n");
        return 77;
    }
    if (argc >= 3 && std::string(argv[1]) == "--bench") {
        const int64_t N = std::atoll(argv[2]);
        const int64_t T = argc >= 4 ? std::atoll(argv[3]) : 4096;
        return bench(N, T);
    }
    const int rc = correctness();
    std::printf("bfla_test: %s\n", rc == 0 ? "pass" : "FAIL");
    return rc;
}
