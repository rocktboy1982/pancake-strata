// src/kernels/bfla_cpu_test.cpp - the fused prompt attention's switches, Ratio-K rule and dense-mode math, CPU only
// (strata/kernels/bfla.hpp; the kernels are in src/kernels/cuda/bfla.cu, checked on a GPU by bfla_test).
//
//   1. Switches: STRATA_BFLA_DENSE=1 is the exact mode, STRATA_BFLA_SPARSE=1 the lossy one (it wins when both are set);
//      nothing set, "0", or only the old STRATA_BFLA: off (the old switch flagged for its note); the knobs' clamps.
//   2. The Ratio-K budget (paper Eq. 9) at hand-computed points, and bfla_tile_plan (the selection kernel's own
//      function): mandatory sink / own / local blocks inside the quota, the tail rule, the budget-covers-all case.
//   3. Dense mode: a host model of k_attn's schedule - workgroups of 64/G tokens x G heads, every causal 64-key block in
//      16-key steps, the inline causal mask (only on blocks that reach the workgroup's first query), online softmax in
//      exp2 with FP32 row state, probabilities rounded to FP16 before P.V, FP32 accumulation - against plain softmax
//      attention in FP64 on small shapes (G 4 / 8 / 16, p0 = 0 and after a prefix, T not a multiple of the tiles):
//      every row attends exactly its causal cells; with FP32 probabilities the model equals FP64 to 3e-5 (the online
//      recurrence is exact), with FP16 probabilities it stays inside the bound the GPU test uses.
//   4. Lists: a list of every block walks the same steps as dense mode (bitwise the same output); a list that drops
//      blocks attends exactly its listed cells (and matches FP64 softmax over them).
// The model is the kernel's arithmetic written for the host, not the kernel; exit 0 pass, 1 fail.  No GPU, no model.
#include "strata/kernels/bfla.hpp"
#include "strata/kernels/f16_bits.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace strata::kernels;

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  [%s] %s\n", ok ? "pass" : "FAIL", what);
    if (!ok) ++g_fail;
}

// ---- 1. switches
BflaConfig parse(const std::map<std::string, std::string>& env) {
    return bfla_config_from([&](const char* name) -> const char* {
        const auto it = env.find(name);
        return it == env.end() ? nullptr : it->second.c_str();
    });
}

void test_switches() {
    std::printf("switches:\n");
    const BflaConfig none = parse({});
    check(none.mode == 0 && !none.legacy && none.min_len == 32768 && none.kmin == 2048 && none.kmax == 32768 &&
              none.sink == 2 && none.local == 4 && none.tail == -1 && std::fabs(none.eta - 1.0 / 3.0) < 1e-15,
          "nothing set: off, the paper's defaults");
    check(parse({{"STRATA_BFLA_DENSE", "1"}}).mode == 2, "STRATA_BFLA_DENSE=1: exact fused mode");
    check(parse({{"STRATA_BFLA_DENSE", "0"}}).mode == 0, "STRATA_BFLA_DENSE=0: off");
    check(parse({{"STRATA_BFLA_SPARSE", "1"}}).mode == 1, "STRATA_BFLA_SPARSE=1: lossy sparse mode");
    check(parse({{"STRATA_BFLA_DENSE", "1"}, {"STRATA_BFLA_SPARSE", "1"}}).mode == 1, "both set: sparse wins (and says LOSSY)");
    const BflaConfig old1 = parse({{"STRATA_BFLA", "1"}}), oldd = parse({{"STRATA_BFLA", "dense"}}),
                     old0 = parse({{"STRATA_BFLA", "0"}});
    check(old1.mode == 0 && old1.legacy && oldd.mode == 0 && oldd.legacy && old0.mode == 0 && !old0.legacy,
          "the old STRATA_BFLA turns nothing on (1 / dense flagged for the note, 0 silent)");
    const BflaConfig cl = parse({{"STRATA_BFLA_SPARSE", "1"}, {"STRATA_BFLA_KMIN", "10"}, {"STRATA_BFLA_KMAX", "5"},
                                 {"STRATA_BFLA_SINK", "0"}, {"STRATA_BFLA_LOCAL", "-3"}, {"STRATA_BFLA_ETA", "2"},
                                 {"STRATA_BFLA_MIN", "65536"}, {"STRATA_BFLA_TAIL", "256"}});
    check(cl.kmin == kBflaBlock && cl.kmax == kBflaBlock && cl.sink == 1 && cl.local == 0 &&
              std::fabs(cl.eta - 1.0 / 3.0) < 1e-15 && cl.min_len == 65536 && cl.tail == 256,
          "knobs: Kmin >= one block, Kmax >= Kmin, sink >= 1, local >= 0, eta in (0, 1], MIN and TAIL taken");
    check(std::strcmp(bfla_switch_name(2), "STRATA_BFLA_DENSE") == 0 && std::strcmp(bfla_switch_name(1), "STRATA_BFLA_SPARSE") == 0,
          "log names");
}

// ---- 2. budget and tile plan
void test_rule() {
    std::printf("Ratio-K rule:\n");
    const BflaConfig c;   // eta 1/3, Kmin 2048, Kmax 32768
    struct P { int64_t V, blocks; };
    // K = min(V, max(Kmin, min(Kmax, ceil(V / 3)))), k = ceil(K / 64)
    const P pts[] = {{16, 1}, {1000, 16}, {6145, 33}, {10000, 53}, {60000, 313}, {131072, 512}, {200000, 512}};
    bool ok = true;
    for (const P& p : pts) {
        const int64_t b = bfla_budget_blocks(p.V, c);
        if (b != p.blocks) {
            std::printf("    V %lld: %lld blocks, expected %lld\n", (long long) p.V, (long long) b, (long long) p.blocks);
            ok = false;
        }
    }
    check(ok, "budget blocks at V = 16, 1000, 6145, 10000, 60000, 131072, 200000");

    const int64_t none = std::numeric_limits<int64_t>::max();
    // a tile of 16 tokens at 6000..6015 (G 8), K 256..1024: 16 blocks, 94 visible, sink 2, own + local from 89
    const BflaTilePlan a = bfla_tile_plan(6000, 6015, none, 1.0 / 3.0, 256, 1024, 2, 4);
    check(a.n_vis == 94 && a.sinkc == 2 && a.lo_b == 89 && a.free_q == 16 - (94 - 87) && !a.all,
          "sparse tile: mandatory sink + own/local blocks counted in the quota of 16, 9 left for the 87 candidates");
    // the tail rule, three-valued around qb = 6015: past full_from, exactly at it (>=, not >), one before it
    check(bfla_tile_plan(6000, 6015, 6010, 1.0 / 3.0, 256, 1024, 2, 4).all, "tail rule: full_from < qb keeps every block");
    check(bfla_tile_plan(6000, 6015, 6015, 1.0 / 3.0, 256, 1024, 2, 4).all, "tail rule: full_from == qb keeps every block");
    check(!bfla_tile_plan(6000, 6015, 6016, 1.0 / 3.0, 256, 1024, 2, 4).all, "tail rule: full_from == qb + 1 stays sparse");
    // the budget, three-valued around the 94 visible blocks of qb 6015 (K = 5952 is exactly 93 blocks of 64)
    check(!bfla_tile_plan(6000, 6015, none, 1.0 / 3.0, 5952, 5952, 2, 4).all, "budget of 93 of 94 visible blocks stays sparse");
    check(bfla_tile_plan(6000, 6015, none, 1.0 / 3.0, 5953, 5953, 2, 4).all, "budget of 94 of 94 visible blocks keeps every block");
    check(bfla_tile_plan(6000, 6015, none, 1.0 / 3.0, 6016, 6016, 2, 4).all, "budget of 94 of 94 (Kmin = V) keeps every block");
    check(bfla_tile_plan(6000, 6015, none, 1.0 / 3.0, int64_t(1) << 30, int64_t(1) << 30, 2, 4).all,
          "a budget of every visible cell keeps every block");
    const BflaTilePlan m = bfla_tile_plan(6000, 6015, none, 1.0 / 3.0, 64, 64, 2, 4);
    check(!m.all && m.free_q <= 0, "a budget smaller than the mandatory blocks: the mandatory blocks only");
    check(bfla_tile_plan(0, 15, none, 1.0 / 3.0, 2048, 32768, 2, 4).all, "a prompt start below Kmin: full attention");
}

// ---- 3 / 4. the attention kernel's arithmetic
constexpr int D = kBflaDim;
struct Data {
    int G = 8, HKV = 2, H = 16;
    int64_t T = 0, p0 = 0, P = 0;
    std::vector<uint16_t> q, k, v;   // q [T][H][D], k / v [HKV][P][D], FP16 bits
    std::vector<float> qf, kf, vf;   // the same values as floats (exact)
};

// Random FP16 data with a direction both q and k share, growing along the prefix, and a few strong keys: the row maximum
// moves late and often, so the online rescale is exercised.
Data make_data(int G, int HKV, int64_t T, int64_t p0, uint32_t seed) {
    Data d;
    d.G = G;
    d.HKV = HKV;
    d.H = G * HKV;
    d.T = T;
    d.p0 = p0;
    d.P = p0 + T;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> u(D);
    for (auto& x : u) x = nd(rng) / 4.f;
    d.k.resize((size_t) d.HKV * d.P * D);
    d.v.resize(d.k.size());
    for (int g = 0; g < d.HKV; ++g)
        for (int64_t c = 0; c < d.P; ++c) {
            const float grow = 3.f * (float) c / (float) d.P, strong = c % 37 == 5 ? 2.5f : 1.f;
            for (int i = 0; i < D; ++i) {
                d.k[((size_t) g * d.P + c) * D + i] = f16_from_f32(strong * nd(rng) + grow * u[i]);
                d.v[((size_t) g * d.P + c) * D + i] = f16_from_f32(nd(rng));
            }
        }
    d.q.resize((size_t) T * d.H * D);
    for (int64_t t = 0; t < T; ++t)
        for (int h = 0; h < d.H; ++h)
            for (int i = 0; i < D; ++i) d.q[((size_t) t * d.H + h) * D + i] = f16_from_f32(nd(rng) + 4.f * u[i]);
    auto widen = [](const std::vector<uint16_t>& x) {
        std::vector<float> r(x.size());
        for (size_t i = 0; i < x.size(); ++i) r[i] = f32_from_f16(x[i]);
        return r;
    };
    d.qf = widen(d.q);
    d.kf = widen(d.k);
    d.vf = widen(d.v);
    return d;
}

// Plain softmax(q k^T / 16) v of query (t, h) in FP64 over `cells`.
void ref_row(const Data& d, int64_t t, int h, const std::vector<int64_t>& cells, double* out) {
    const int g = h / d.G;
    const float* q = &d.qf[((size_t) t * d.H + h) * D];
    std::vector<double> s(cells.size());
    double mx = -1e300;
    for (size_t n = 0; n < cells.size(); ++n) {
        const float* k = &d.kf[((size_t) g * d.P + cells[n]) * D];
        double a = 0;
        for (int i = 0; i < D; ++i) a += (double) q[i] * (double) k[i];
        s[n] = a / 16.0;
        mx = std::max(mx, s[n]);
    }
    double den = 0;
    for (auto& x : s) den += (x = std::exp(x - mx));
    std::fill(out, out + D, 0.0);
    for (size_t n = 0; n < cells.size(); ++n) {
        const float* v = &d.vf[((size_t) g * d.P + cells[n]) * D];
        for (int i = 0; i < D; ++i) out[i] += s[n] / den * (double) v[i];
    }
}

// Selection lists [tile][HKV] (tile = 128 / G tokens of the chunk), ascending block ids.
using Lists = std::vector<std::vector<int>>;

// k_attn's schedule and arithmetic for every row; lists == nullptr: dense mode.  `cells[row]` gets the cells each row
// attended (unmasked), row = t * H + h.  p16: probabilities rounded to FP16 before P.V (the kernel), else FP32.
std::vector<float> model_attn(const Data& d, const Lists* lists, bool p16, std::vector<std::vector<int64_t>>& cells) {
    const int G = d.G, WT = 64 / G, TT = kBflaTileRows / G;
    const float scale_log2 = 1.4426950408889634f / std::sqrt((float) D);
    std::vector<float> out((size_t) d.T * d.H * D, std::numeric_limits<float>::quiet_NaN());
    cells.assign((size_t) d.T * d.H, {});
    for (int g = 0; g < d.HKV; ++g)
        for (int64_t t0 = 0; t0 < d.T; t0 += WT) {   // one workgroup: tokens t0 .. tlast, rows (token, head of group g)
            const int64_t tlast = std::min<int64_t>(d.T, t0 + WT) - 1, qa = t0, qb_abs = d.p0 + tlast;
            const std::vector<int>* lst = lists ? &(*lists)[(size_t) (t0 / TT) * d.HKV + g] : nullptr;
            const int n = lst ? (int) lst->size() : (int) (qb_abs / kBflaBlock) + 1;
            for (int r = 0; r < 64; ++r) {
                const int64_t t = t0 + r / G;
                if (t > tlast) continue;   // the kernel computes these rows on a clamped token and drops them
                const int h = g * G + r % G;
                const float* q = &d.qf[((size_t) t * d.H + h) * D];
                float m = -INFINITY, l = 0.f;
                std::vector<float> O(D, 0.f);
                for (int bi = 0; bi < n; ++bi) {
                    const int64_t j = lst ? (int64_t) (*lst)[(size_t) bi] : bi;
                    const int64_t ks = j * kBflaBlock;
                    if (ks > qb_abs) break;   // ascending: every later block is past the workgroup's last query
                    const bool need_mask = ks + kBflaBlock - 1 > d.p0 + qa;
                    for (int st = 0; st < kBflaBlock / 16; ++st) {
                        const int64_t kb = ks + st * 16;
                        if (kb > qb_abs) break;
                        float v[16], p[16], mx = -INFINITY;
                        for (int c = 0; c < 16; ++c) {
                            const int64_t cell = kb + c;
                            const bool masked = cell >= d.P || (need_mask && cell - d.p0 > t);
                            float s = 0.f;
                            if (!masked) {
                                const float* k = &d.kf[((size_t) g * d.P + cell) * D];
                                for (int i = 0; i < D; ++i) s += q[i] * k[i];
                                cells[(size_t) (t * d.H + h)].push_back(cell);
                            }
                            v[c] = masked ? -INFINITY : s * scale_log2;
                            mx = std::max(mx, v[c]);
                        }
                        const float mnew = std::max(m, mx);
                        const float alpha = mnew == -INFINITY ? 1.f : std::exp2(m - mnew);
                        float ps = 0.f;
                        for (int c = 0; c < 16; ++c) {
                            p[c] = v[c] == -INFINITY ? 0.f : std::exp2(v[c] - mnew);
                            ps += p[c];
                        }
                        l = l * alpha + ps;
                        m = mnew;
                        for (int i = 0; i < D; ++i) O[(size_t) i] *= alpha;
                        for (int c = 0; c < 16; ++c) {
                            if (p[c] == 0.f) continue;
                            const float w = p16 ? f32_from_f16(f16_from_f32(p[c])) : p[c];
                            const float* vv = &d.vf[((size_t) g * d.P + kb + c) * D];
                            for (int i = 0; i < D; ++i) O[(size_t) i] += w * vv[i];
                        }
                    }
                }
                const float inv = l > 0.f ? 1.f / l : 0.f;
                for (int i = 0; i < D; ++i) out[((size_t) t * d.H + h) * D + i] = O[(size_t) i] * inv;
            }
        }
    return out;
}

struct Err {
    double max_abs = 0, max_ref = 0, rel_l2 = 0;
    bool finite = true;
};
template <class Cells>
Err compare(const Data& d, const std::vector<float>& o, Cells cells_of) {
    Err e;
    double num = 0, den = 0;
    std::vector<double> r(D);
    for (int64_t t = 0; t < d.T; ++t)
        for (int h = 0; h < d.H; ++h) {
            ref_row(d, t, h, cells_of(t, h), r.data());
            for (int i = 0; i < D; ++i) {
                const double x = o[((size_t) t * d.H + h) * D + i];
                if (!std::isfinite(x)) e.finite = false;
                e.max_abs = std::max(e.max_abs, std::fabs(x - r[(size_t) i]));
                e.max_ref = std::max(e.max_ref, std::fabs(r[(size_t) i]));
                num += (x - r[(size_t) i]) * (x - r[(size_t) i]);
                den += r[(size_t) i] * r[(size_t) i];
            }
        }
    e.rel_l2 = std::sqrt(num / std::max(den, 1e-300));
    return e;
}
std::vector<int64_t> causal(const Data& d, int64_t t) {
    std::vector<int64_t> c((size_t) (d.p0 + t + 1));
    for (size_t i = 0; i < c.size(); ++i) c[i] = (int64_t) i;
    return c;
}

void test_dense() {
    std::printf("dense mode (host model of the kernel) vs plain softmax attention in FP64:\n");
    struct Shape { int G, HKV; int64_t T, p0; };
    const Shape shapes[] = {{8, 2, 70, 0}, {4, 2, 37, 130}, {16, 1, 100, 250}, {8, 1, 1, 200}, {8, 1, 129, 64}};
    for (const Shape& s : shapes) {
        const Data d = make_data(s.G, s.HKV, s.T, s.p0, 17u + (uint32_t) (s.T * 31 + s.p0));
        std::vector<std::vector<int64_t>> seen32, seen16;
        const auto o32 = model_attn(d, nullptr, false, seen32);
        const auto o16 = model_attn(d, nullptr, true, seen16);
        bool exact_cells = true;
        for (int64_t t = 0; t < d.T; ++t)
            for (int h = 0; h < d.H; ++h) {
                auto c = seen16[(size_t) (t * d.H + h)];
                std::sort(c.begin(), c.end());
                exact_cells = exact_cells && c == causal(d, t);
            }
        const Err e32 = compare(d, o32, [&](int64_t t, int) { return causal(d, t); });
        const Err e16 = compare(d, o16, [&](int64_t t, int) { return causal(d, t); });
        std::printf("  G %d HKV %d T %lld p0 %lld: FP32 probabilities max |err| %.2e rel L2 %.2e; FP16 (kernel) max |err| %.2e "
                    "(max |ref| %.2f) rel L2 %.2e\n", s.G, s.HKV, (long long) s.T, (long long) s.p0, e32.max_abs, e32.rel_l2,
                    e16.max_abs, e16.max_ref, e16.rel_l2);
        check(exact_cells, "every row attends exactly its causal cells (mask and block walk)");
        check(e32.finite && e32.max_abs < 3e-5 * std::max(1.0, e32.max_ref), "online softmax recurrence exact (FP32 p: within 3e-5)");
        check(e16.finite && e16.rel_l2 < 3e-3 && e16.max_abs < 5e-3 * std::max(1.0, e16.max_ref),
              "FP16 probabilities within the GPU test's bound (rel L2 < 3e-3, max < 5e-3)");
    }
}

void test_lists() {
    std::printf("lists:\n");
    const Data d = make_data(8, 1, 48, 2000, 99u);
    const int TT = kBflaTileRows / d.G;
    const int64_t n_tiles = (d.T + TT - 1) / TT, none = std::numeric_limits<int64_t>::max();
    std::vector<std::vector<int64_t>> seen_d, seen_a, seen_s;
    const auto od = model_attn(d, nullptr, true, seen_d);
    Lists all((size_t) n_tiles * d.HKV), sparse(all.size());
    bool lists_ok = true;
    for (int64_t tile = 0; tile < n_tiles; ++tile)
        for (int g = 0; g < d.HKV; ++g) {
            const int64_t qa = d.p0 + tile * TT, qb = d.p0 + std::min<int64_t>(d.T, (tile + 1) * TT) - 1;
            const BflaTilePlan pl = bfla_tile_plan(qa, qb, none, 1.0 / 3.0, 256, 1024, 2, 4);
            auto& A = all[(size_t) (tile * d.HKV + g)];
            for (int j = 0; j < pl.n_vis; ++j) A.push_back(j);
            // the plan's mandatory blocks plus the free quota taken from the candidates by stride (no scores here: this
            // checks the walk over a list with gaps, not the ranking)
            auto& S = sparse[(size_t) (tile * d.HKV + g)];
            const int nc = pl.lo_b - pl.sinkc;
            for (int j = 0; j < pl.sinkc; ++j) S.push_back(j);
            for (int64_t i = 0; i < std::max<int64_t>(pl.free_q, 0); ++i) S.push_back(pl.sinkc + (int) (i * nc / pl.free_q));
            for (int j = pl.lo_b; j < pl.n_vis; ++j) S.push_back(j);
            lists_ok = lists_ok && !pl.all && (int) S.size() == (int) bfla_budget_blocks(qb + 1, 1.0 / 3.0, 256, 1024) &&
                       std::is_sorted(S.begin(), S.end()) && std::adjacent_find(S.begin(), S.end()) == S.end();
        }
    check(lists_ok, "test lists: the budget's size with gaps, ascending, no repeats");
    const auto oa = model_attn(d, &all, true, seen_a);
    check(std::memcmp(od.data(), oa.data(), od.size() * sizeof(float)) == 0, "a list of every block == dense mode, bitwise");
    const auto os = model_attn(d, &sparse, true, seen_s);
    auto listed = [&](int64_t t, int h) {
        const auto& S = sparse[(size_t) ((t / TT) * d.HKV + h / d.G)];
        std::vector<int64_t> c;
        for (int j : S)
            for (int64_t x = (int64_t) j * kBflaBlock; x < (int64_t) (j + 1) * kBflaBlock && x <= d.p0 + t; ++x) c.push_back(x);
        return c;
    };
    bool exact_cells = true;
    for (int64_t t = 0; t < d.T; ++t)
        for (int h = 0; h < d.H; ++h) {
            auto c = seen_s[(size_t) (t * d.H + h)];
            std::sort(c.begin(), c.end());
            exact_cells = exact_cells && c == listed(t, h);
        }
    check(exact_cells, "a list with gaps attends exactly its listed causal cells");
    const Err e = compare(d, os, listed);
    std::printf("  sparse list vs FP64 over its cells: max |err| %.2e, rel L2 %.2e\n", e.max_abs, e.rel_l2);
    check(e.finite && e.rel_l2 < 3e-3 && e.max_abs < 5e-3 * std::max(1.0, e.max_ref), "the listed cells' softmax, FP16 bound");
}

}  // namespace

int main() {
    test_switches();
    test_rule();
    test_dense();
    test_lists();
    std::printf("bfla_cpu_test: %s\n", g_fail == 0 ? "pass" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
