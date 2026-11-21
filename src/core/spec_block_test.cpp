// STRATA_MTP_BLOCK_VERIFY (LEADS.md S12): the block-verification rule of include/strata/core/spec_block.hpp, on the CPU.
//   1. exact enumeration: the law of an infinite stream of rounds on a toy model equals the target's joint law (to 1e-12)
//   2. fixed-seed Monte Carlo through spec_block_verify itself: chi-square and total variation of 3-token streams
//   3. temperature 0: equal to the longest matching prefix and the argmax, for any q
//   4. block verification keeps at least as many drafts as the token-by-token rule on the same drafts
//   5. the switch parse, and a point-mass q (no list)
// No GPU, no model.  The device sampler does not run the rule yet (spec_block.hpp, status).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include "strata/core/spec_block.hpp"

using namespace strata::core;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

static constexpr int V = 4;
static constexpr int G = 3;

// a toy autoregressive pair: the distribution after a prefix is a fixed pseudo-random table keyed by the prefix
struct Toy {
    uint64_t salt;
    bool sparse;   // true: the target keeps only its top-3 tokens (like a top-k cut), q keeps all
    std::vector<double> dist(const std::vector<int>& ctx, bool target) const {
        uint64_t h = salt * 0x9E3779B97F4A7C15ull + (target ? 17 : 91);
        for (int c : ctx) h = (h ^ (uint64_t) (c + 1)) * 0x100000001B3ull + 0x7F4A7C15ull;
        std::mt19937_64 g(h);
        std::vector<double> w(V);
        double s = 0;
        for (int i = 0; i < V; ++i) { w[i] = 0.05 + (double) (g() % 1000) / 1000.0; }
        if (target && sparse) {   // zero the smallest
            int m = 0;
            for (int i = 1; i < V; ++i) if (w[i] < w[m]) m = i;
            w[m] = 0;
        }
        for (double x : w) s += x;
        for (double& x : w) x /= s;
        if (!target) {   // the drafter's list is FP32 in the engine: use k / 4096 so every value is exact and the list sums to 1
            int big = 0, used = 0;
            for (int i = 0; i < V; ++i) { w[i] = std::floor(w[i] * 4096.0) / 4096.0; used += (int) (w[i] * 4096.0); if (w[i] > w[big]) big = i; }
            w[big] += (4096 - used) / 4096.0;
        }
        return w;
    }
};

struct Rows {
    int32_t ids[G + 1][V];
    double p[G + 1][V];
    float q[G][V];
    int32_t qid[G][V];
    SpecP P[G + 1];
    SpecQ Q[G];
};

static void build(const Toy& t, const std::vector<int>& ctx, const int* X, Rows& R, bool point_q) {
    std::vector<int> pre = ctx;
    for (int j = 0; j <= G; ++j) {
        const auto pt = t.dist(pre, true);
        int n = 0;
        for (int i = 0; i < V; ++i)
            if (pt[i] > 0) { R.ids[j][n] = i; R.p[j][n] = pt[i]; ++n; }
        R.P[j] = {R.ids[j], R.p[j], n};
        if (j < G) {
            const auto qt = t.dist(pre, false);
            int m = 0;
            for (int i = 0; i < V; ++i) { R.qid[j][m] = i; R.q[j][m] = (float) qt[i]; ++m; }
            R.Q[j] = point_q ? SpecQ{nullptr, nullptr, -1} : SpecQ{R.qid[j], R.q[j], m};
            pre.push_back(X[j]);
        }
    }
}

// ---- 1. exact enumeration ---------------------------------------------------------------------------------------------
// the outcome law of ONE round: {emitted tokens} -> probability, from the formulas of spec_block.hpp (ratios, residuals, h)
static std::map<std::vector<int>, double> round_law(const Toy& t, const std::vector<int>& ctx, bool point_q) {
    std::map<std::vector<int>, double> out;
    int X[G];
    const int total = 1 << (2 * G);   // V = 4: 2 bits per draft
    for (int code = 0; code < total; ++code) {
        for (int j = 0; j < G; ++j) X[j] = (code >> (2 * j)) & 3;
        Rows R;
        build(t, ctx, X, R, point_q);
        // draft probability: from q (the point-mass case drafts argmax of q: only that X has weight)
        double pX = 1.0;
        std::vector<int> pre = ctx;
        for (int j = 0; j < G; ++j) {
            const auto qt = t.dist(pre, false);
            if (point_q) {
                int am = 0;
                for (int i = 1; i < V; ++i) if (qt[i] > qt[am]) am = i;
                pX *= (X[j] == am) ? 1.0 : 0.0;
            } else {
                pX *= (double) (float) qt[X[j]];   // q is k / 4096: exact in FP32
            }
            pre.push_back(X[j]);
        }
        if (pX == 0.0) continue;
        int32_t d[G];
        for (int j = 0; j < G; ++j) d[j] = X[j];
        double w[G + 1];
        spec_block_ratios(R.P, R.Q, d, G, w);
        double rem = 1.0;
        for (int j = G; j >= 0; --j) {
            double r[64];
            const double s = spec_block_residual(R.P, R.Q, d, G, j, w[j], r);
            const double h = j == 0 ? 1.0 : spec_block_h(G, j, w[j], s);
            const double pa = rem * h;
            double sum = s;
            const bool own = !(s > 0.0);
            if (own) { sum = 0; for (int i = 0; i < R.P[j].n; ++i) { r[i] = R.P[j].p[i]; sum += r[i]; } }
            for (int i = 0; i < R.P[j].n; ++i) {
                std::vector<int> seq(X, X + j);
                seq.push_back(R.P[j].id[i]);
                out[seq] += pX * pa * r[i] / sum;
            }
            rem *= 1.0 - h;
        }
    }
    return out;
}

// the first n tokens of the stream of rounds: P(seq) summed over how the rounds cut it
static void stream(const Toy& t, const std::vector<int>& ctx, int n, bool point_q, double weight, std::map<std::vector<int>, double>& acc,
                   const std::vector<int>& made) {
    if (n == 0) { acc[made] += weight; return; }
    for (const auto& kv : round_law(t, ctx, point_q)) {
        std::vector<int> s = kv.first;
        if ((int) s.size() > n) s.resize(n);
        std::vector<int> c2 = ctx, m2 = made;
        c2.insert(c2.end(), s.begin(), s.end());
        m2.insert(m2.end(), s.begin(), s.end());
        stream(t, c2, n - (int) s.size(), point_q, weight * kv.second, acc, m2);
    }
}

static double target_prob(const Toy& t, const std::vector<int>& y) {
    double p = 1.0;
    std::vector<int> pre;
    for (int c : y) { p *= t.dist(pre, true)[c]; pre.push_back(c); }
    return p;
}

static void exact_law_test(const Toy& t, bool point_q, const char* name) {
    const int N = 4;
    std::map<std::vector<int>, double> acc;
    stream(t, {}, N, point_q, 1.0, acc, {});
    double total = 0, worst = 0;
    for (const auto& kv : acc) {
        total += kv.second;
        worst = std::max(worst, std::fabs(kv.second - target_prob(t, kv.first)));
    }
    std::printf("exact law (%s): %zu sequences, total %.15f, max |stream - target| %.3e\n", name, acc.size(), total, worst);
    CHECK(std::fabs(total - 1.0) < 1e-12);
    CHECK(worst < 1e-12);
}

// ---- 2. Monte Carlo through spec_block_verify -----------------------------------------------------------------------
static int draw(const std::vector<double>& p, double u) {
    double c = 0;
    for (int i = 0; i < (int) p.size(); ++i) { c += p[i]; if (u < c) return i; }
    return (int) p.size() - 1;
}

static void monte_carlo_test(const Toy& t, uint64_t seed, long rounds, const char* name) {
    std::mt19937_64 rng(seed);
    auto U = [&] { return (double) (rng() >> 11) * (1.0 / 9007199254740992.0); };
    const int N = 3;
    std::map<std::vector<int>, long> cnt;
    long mean_acc_sum = 0, mean_acc_rounds = 0, tok_acc_sum = 0;
    for (long k = 0; k < rounds; ++k) {
        std::vector<int> made;
        while ((int) made.size() < N) {
            int X[G];
            std::vector<int> pre = made;
            for (int j = 0; j < G; ++j) {   // draft from q (float-rounded, as the engine's list is)
                auto qt = t.dist(pre, false);
                for (double& x : qt) x = (double) (float) x;
                X[j] = draw(qt, U());
                pre.push_back(X[j]);
            }
            Rows R;
            build(t, made, X, R, false);
            int32_t d[G];
            for (int j = 0; j < G; ++j) d[j] = X[j];
            double ue[G + 1], ur = U();
            for (int j = 0; j <= G; ++j) ue[j] = U();
            const SpecBlockResult b = spec_block_verify(R.P, R.Q, d, G, ue, ur);
            int32_t out[G + 1];
            const int n_out = spec_block_fill_out(b, d, out);
            for (int i = 0; i < n_out && (int) made.size() < N; ++i) made.push_back(out[i]);
            // the same drafts through the token-by-token rule, for the acceptance comparison (first round of a stream only)
            mean_acc_sum += b.accepted;
            ++mean_acc_rounds;
            int a = 0;
            while (a < G) {
                const SpecResult sr = spec_verify_row(R.P[a], d[a], R.Q[a], (float) U(), (float) U());
                if (!sr.accepted) break;
                ++a;
            }
            tok_acc_sum += a;
        }
        ++cnt[made];
    }
    // chi-square against the target joint over the V^N cells with positive probability
    double chi = 0, tv = 0;
    int cells = 0;
    std::vector<int> y(N);
    for (int code = 0; code < (1 << (2 * N)); ++code) {
        for (int j = 0; j < N; ++j) y[j] = (code >> (2 * j)) & 3;
        const double p = target_prob(t, y);
        const long c = cnt.count(y) ? cnt[y] : 0;
        if (p <= 0) { CHECK(c == 0); continue; }   // a token the target never emits is never emitted
        const double e = p * (double) rounds;
        chi += (c - e) * (c - e) / e;
        tv += std::fabs((double) c / (double) rounds - p);
        ++cells;
    }
    tv *= 0.5;
    const double df = cells - 1;
    const double crit = df + 6.0 * std::sqrt(2.0 * df);   // about the 1e-8 tail: a fixed seed, so a wrong law fails by a mile
    std::printf("monte carlo (%s): %ld streams, %d cells, chi2 %.1f (df %.0f, limit %.1f), TV %.5f; kept drafts per round: block %.4f, token-wise %.4f\n",
                name, rounds, cells, chi, df, crit, tv, (double) mean_acc_sum / (double) mean_acc_rounds,
                (double) tok_acc_sum / (double) mean_acc_rounds);
    CHECK(chi < crit);
    CHECK(tv < 0.006);
    CHECK((double) mean_acc_sum >= (double) tok_acc_sum * 0.999);   // block verification keeps at least as many (paper, Theorem 1)
}

// ---- 3. temperature 0 ------------------------------------------------------------------------------------------------
static void greedy_test() {
    std::mt19937_64 rng(12345);
    auto U = [&] { return (double) (rng() >> 11) * (1.0 / 9007199254740992.0); };
    long bad = 0, n = 0, full = 0;
    for (int trial = 0; trial < 200000; ++trial) {
        const int gam = 1 + (int) (rng() % G);
        int32_t ids[G + 1][1];
        double p1[G + 1][1];
        SpecP P[G + 1];
        int am[G + 1];
        for (int j = 0; j <= gam; ++j) { am[j] = (int) (rng() % V); ids[j][0] = am[j]; p1[j][0] = 1.0; P[j] = {ids[j], p1[j], 1}; }
        int32_t d[G];
        for (int j = 0; j < gam; ++j) d[j] = (rng() % 3) ? am[j] : (int32_t) (rng() % V);   // mostly matching drafts
        float q[G][V];
        int32_t qid[G][V];
        SpecQ Q[G];
        for (int j = 0; j < gam; ++j) {
            double s = 0;
            double w[V];
            for (int i = 0; i < V; ++i) { w[i] = 0.01 + U(); s += w[i]; }
            for (int i = 0; i < V; ++i) { q[j][i] = (float) (w[i] / s); qid[j][i] = i; }
            Q[j] = (rng() & 1) ? SpecQ{qid[j], q[j], V} : SpecQ{nullptr, nullptr, -1};
        }
        double ue[G + 1];
        for (int j = 0; j <= gam; ++j) ue[j] = U();
        const SpecBlockResult b = spec_block_verify(P, Q, d, gam, ue, U());
        int a = 0;
        while (a < gam && d[a] == am[a]) ++a;
        ++n;
        if (a == gam) ++full;
        if (b.accepted != a || b.token != am[a]) ++bad;
    }
    std::printf("greedy: %ld trials (%ld all-match), %ld differ from prefix match + argmax\n", n, full, bad);
    CHECK(bad == 0);
    CHECK(full > 1000);
}

int main() {
    // 5. the switch
    CHECK(!spec_block_verify_on_from(nullptr));
    CHECK(!spec_block_verify_on_from("0"));
    CHECK(!spec_block_verify_on_from(""));
    CHECK(spec_block_verify_on_from("1"));

    // gamma = 0: a plain sample from P[0]
    {
        int32_t id[2] = {7, 9};
        double p[2] = {0.25, 0.75};
        SpecP P[1] = {{id, p, 2}};
        double ue[1] = {0.5};
        CHECK(spec_block_verify(P, nullptr, nullptr, 0, ue, 0.1).token == 7);
        CHECK(spec_block_verify(P, nullptr, nullptr, 0, ue, 0.3).token == 9);
        CHECK(spec_block_verify(P, nullptr, nullptr, 0, ue, 0.1).accepted == 0);
    }

    const Toy full{1, false}, cut{2, true}, third{3, false};
    exact_law_test(full, false, "full-support p, q lists");
    exact_law_test(cut, false, "p cut to its top 3, q lists");
    exact_law_test(third, true, "point-mass q (argmax drafts)");
    monte_carlo_test(full, 20261121, 400000, "full-support p, q lists");
    monte_carlo_test(cut, 7, 400000, "p cut to its top 3, q lists");
    greedy_test();

    std::printf("%s\n", failed ? "FAILED" : "spec_block_test: ok");
    return failed ? 1 : 0;
}
