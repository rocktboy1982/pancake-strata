// CPU-only test of the device form of block verification (include/strata/core/spec_block_dev.hpp, STRATA_MTP_BLOCK_VERIFY, LEADS.md S12).
//
// The CUDA/HIP kernel (src/kernels/cuda/sampler.cu spec_block_rows_kernel + spec_block_verify_kernel) cannot run here.  Its arithmetic
// and layout live in spec_block_dev.hpp, which the kernel calls; this test runs the kernel's control flow (run_serial, the same calls
// in the same order) on the flat arrays the kernel reads and compares it with the tested rule spec_block_verify on random toy
// distributions: accepted counts and tokens bit-equal, and the uniforms asked for in the same order.  It also checks the engine
// loop's view (fill_out: the host's `while (window[a+1] == out[a]) ++a`) keeps the target's output law by a fixed-seed chi-square on
// 3-token streams, the one-hot (temperature 0) case against the longest matching prefix plus the argmax, the flat layout, and the
// route (the switch off is always the token-by-token path).
#include "strata/core/spec_block_dev.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::core;
namespace sbd = strata::core::spec_block_dev;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

std::mt19937_64 rng(0x5bd1e995ull);
double uni() { return (double) (rng() >> 11) * (1.0 / 9007199254740992.0); }

// one window as the kernel sees it, and as the rule's structs see it
struct Window {
    int gamma = 0, n_q = 0;
    std::vector<int32_t> pid, d, qbuf;
    std::vector<double> pp;
    std::vector<int> pn;
    std::vector<std::vector<int32_t>> ids;   // for the SpecP / SpecQ views
    std::vector<std::vector<double>> probs;
    std::vector<std::vector<int32_t>> qids;
    std::vector<std::vector<float>> qprobs;
    std::vector<SpecP> P;
    std::vector<SpecQ> Q;
    sbd::View view() const { return {pid.data(), pp.data(), pn.data(), qbuf.data(), n_q, d.data(), gamma}; }
};

void finish(Window& w) {
    w.P.clear(); w.Q.clear();
    for (int r = 0; r <= w.gamma; ++r) w.P.push_back({w.ids[r].data(), w.probs[r].data(), (int) w.ids[r].size()});
    for (int j = 0; j < w.gamma; ++j)
        w.Q.push_back(j < w.n_q ? SpecQ{w.qids[j].data(), w.qprobs[j].data(), (int) w.qids[j].size()} : SpecQ{nullptr, nullptr, -1});
    w.pid.assign((size_t) (w.gamma + 1) * sbd::kRowCap, -1);
    w.pp.assign((size_t) (w.gamma + 1) * sbd::kRowCap, 0.0);
    w.pn.assign((size_t) w.gamma + 1, 0);
    for (int r = 0; r <= w.gamma; ++r) {
        w.pn[r] = (int) w.ids[r].size();
        for (int i = 0; i < w.pn[r]; ++i) { w.pid[(size_t) r * sbd::kRowCap + i] = w.ids[r][i]; w.pp[(size_t) r * sbd::kRowCap + i] = w.probs[r][i]; }
    }
    w.qbuf.assign((size_t) std::max(w.n_q, 1) * kSpecQStride, -1);
    for (int j = 0; j < w.n_q; ++j) {
        int32_t* ids = w.qbuf.data() + (size_t) j * kSpecQStride;
        float* pr = reinterpret_cast<float*>(ids + kSpecQEntries);
        for (size_t i = 0; i < w.qids[j].size(); ++i) { ids[i] = w.qids[j][i]; pr[i] = w.qprobs[j][i]; }
    }
}

std::vector<int32_t> pick_ids(int n, int vocab) {
    std::vector<int32_t> all(vocab);
    for (int i = 0; i < vocab; ++i) all[i] = i;
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(n);
    return all;
}

Window random_window() {
    Window w;
    w.gamma = 1 + (int) (rng() % kSpecBlockMaxGamma);
    w.n_q = (int) (rng() % (w.gamma + 1));   // 0..gamma rows with a list (the rest point masses)
    const int vocab = 80;
    for (int r = 0; r <= w.gamma; ++r) {
        const int n = 1 + (int) (rng() % 64);
        w.ids.push_back(pick_ids(n, vocab));
        std::vector<double> p(n);
        double s = 0;
        for (auto& x : p) { x = (rng() % 5 == 0) ? 0.0 : std::pow(uni(), 3.0); s += x; }
        if (!(s > 0)) { p[0] = 1; s = 1; }
        for (auto& x : p) x /= s;
        w.probs.push_back(p);
    }
    for (int j = 0; j < w.n_q; ++j) {
        const int m = 1 + (int) (rng() % 64);
        w.qids.push_back(pick_ids(m, vocab));
        std::vector<float> q(m);
        double s = 0;
        std::vector<double> t(m);
        for (auto& x : t) { x = std::pow(uni(), 2.0) + 1e-3; s += x; }
        for (int i = 0; i < m; ++i) q[i] = (float) (t[i] / s);
        w.qprobs.push_back(q);
    }
    for (int j = 0; j < w.gamma; ++j) {
        const uint64_t k = rng() % 10;
        int32_t d;
        if (k < 6) d = w.ids[j][rng() % w.ids[j].size()];
        else if (k < 9 && j < w.n_q) d = w.qids[j][rng() % w.qids[j].size()];
        else d = (int32_t) (rng() % vocab);
        w.d.push_back(d);
    }
    finish(w);
    return w;
}

void test_equals_rule() {
    size_t rejected_prefix = 0, full = 0;
    for (int trial = 0; trial < 300000; ++trial) {
        Window w = random_window();
        std::array<double, 20> eta{}, resid{};
        for (auto& x : eta) x = (double) (float) uni();   // the kernel's uniforms are floats
        for (auto& x : resid) x = (double) (float) uni();
        std::vector<int> order_a, order_b;
        const SpecBlockResult a = spec_block_verify_with(w.P.data(), w.Q.data(), w.d.data(), w.gamma,
            [&](int j) { order_a.push_back(j); return eta[j]; }, [&](int win) { order_a.push_back(1000 + win); return resid[win]; });
        double ws[kSpecBlockMaxGamma + 1], r[sbd::kRowCap];
        const SpecBlockResult b = sbd::run_serial(w.view(), ws, r,
            [&](int j) { order_b.push_back(j); return eta[j]; }, [&](int win) { order_b.push_back(1000 + win); return resid[win]; });
        check(a.accepted == b.accepted && a.token == b.token, "device flow equals spec_block_verify (accepted, token)");
        check(order_a == order_b, "the uniforms are asked for in the same order");
        // and the array form (a fixed u_resid) agrees with the lazy one for that same draw
        const SpecBlockResult c = spec_block_verify(w.P.data(), w.Q.data(), w.d.data(), w.gamma, eta.data(), resid[a.accepted]);
        check(c.accepted == a.accepted && c.token == a.token, "array form equals the lazy form");
        check(a.accepted >= 0 && a.accepted <= w.gamma, "accepted in range");
        a.accepted == w.gamma ? ++full : (a.accepted > 0 ? ++rejected_prefix : 0);
        // the engine's loop on fill_out: out[t] for t < accepted is the draft
        int32_t out[kSpecBlockMaxGamma + 1];
        sbd::fill_out(w.view(), a, [&](int t) { return resid[(t + 3) % 20]; }, out);
        for (int t = 0; t < a.accepted; ++t) check(out[t] == w.d[t], "kept rows hold their drafts");
        check(out[a.accepted] == a.token, "the winner row holds the drawn token");
    }
    check(full > 1000 && rejected_prefix > 1000, "the random windows reach full and partial accepts");
}

// A first-order toy model over 4 tokens: P(next | prev) for the target, Q for the drafter.
struct Toy {
    double P[4][4], Q[4][4];
    Toy() {
        for (int a = 0; a < 4; ++a) {
            double sp = 0, sq = 0;
            for (int b = 0; b < 4; ++b) { P[a][b] = 0.05 + uni(); Q[a][b] = 0.05 + uni(); sp += P[a][b]; sq += Q[a][b]; }
            for (int b = 0; b < 4; ++b) { P[a][b] /= sp; Q[a][b] /= sq; }
        }
        for (int b = 0; b < 4; ++b) P[1][b] = b == 3 ? 0.0 : P[1][b] / (1.0 - P[1][3]);   // a token the target never emits after 1
    }
};

// one engine round at context `last` for drafts gamma: the tokens the host loop takes (a + 1 of them)
std::vector<int> engine_round(const Toy& m, int last, int gamma) {
    Window w;
    w.gamma = gamma; w.n_q = gamma;
    int prev = last;
    std::vector<int> ctx;   // the token before each row
    for (int j = 0; j <= gamma; ++j) {
        ctx.push_back(prev);
        std::vector<int32_t> ids = {0, 1, 2, 3};
        std::vector<double> pr(4);
        for (int b = 0; b < 4; ++b) pr[b] = m.P[prev][b];
        w.ids.push_back(ids); w.probs.push_back(pr);
        if (j < gamma) {
            std::vector<float> q(4);
            for (int b = 0; b < 4; ++b) q[b] = (float) m.Q[prev][b];
            w.qids.push_back(ids); w.qprobs.push_back(q);
            double u = uni(), c = 0;   // the drafter's draw from q
            int d = 3;
            for (int b = 0; b < 4; ++b) { c += (double) q[b]; if (u < c) { d = b; break; } }
            w.d.push_back(d);
            prev = d;
        }
    }
    finish(w);
    std::array<double, 20> eta{}, resid{}, pick{};
    for (auto& x : eta) x = uni();
    for (auto& x : resid) x = uni();
    for (auto& x : pick) x = uni();
    double ws[kSpecBlockMaxGamma + 1], r[sbd::kRowCap];
    const SpecBlockResult res = sbd::run_serial(w.view(), ws, r, [&](int j) { return eta[j]; }, [&](int win) { return resid[win]; });
    int32_t out[kSpecBlockMaxGamma + 1];
    sbd::fill_out(w.view(), res, [&](int t) { return pick[t]; }, out);
    int a = 0;
    while (a < gamma && w.d[a] == out[a]) ++a;   // the engine's host loop, unchanged
    std::vector<int> emitted(out, out + a + 1);
    return emitted;
}

void test_engine_loop_law() {
    // fixed-seed chi-square: the first 3 tokens of a stream made of engine rounds follow the target chain, whatever q is
    const Toy m;
    for (int gamma : {1, 2, 3, 5}) {
        const int N = 400000;
        std::vector<double> count(64, 0.0);
        for (int n = 0; n < N; ++n) {
            std::vector<int> s;
            int last = 0;
            while (s.size() < 3) {
                for (int t : engine_round(m, last, gamma)) { s.push_back(t); last = t; }
            }
            ++count[s[0] * 16 + s[1] * 4 + s[2]];
        }
        double chi = 0;
        int df = -1;
        for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b)
                for (int c = 0; c < 4; ++c) {
                    const double e = N * m.P[0][a] * m.P[a][b] * m.P[b][c];
                    if (e <= 0.0) { check(count[a * 16 + b * 4 + c] == 0.0, "a token the target never emits is never emitted"); continue; }
                    chi += (count[a * 16 + b * 4 + c] - e) * (count[a * 16 + b * 4 + c] - e) / e;
                    ++df;
                }
        std::printf("engine loop, gamma %d: chi-square %.1f, df %d\n", gamma, chi, df);
        check(chi < df + 5.0 * std::sqrt(2.0 * df), "engine loop keeps the target's output law (chi-square within 5 sigma)");
    }
}

void test_greedy() {
    // one-hot P (temperature 0): the longest matching prefix, then the argmax
    for (int trial = 0; trial < 50000; ++trial) {
        Window w;
        w.gamma = 1 + (int) (rng() % kSpecBlockMaxGamma);
        w.n_q = (int) (rng() % (w.gamma + 1));
        const int vocab = 30;
        std::vector<int32_t> arg(w.gamma + 1);
        for (int r = 0; r <= w.gamma; ++r) {
            arg[r] = (int32_t) (rng() % vocab);
            w.ids.push_back({arg[r]});
            w.probs.push_back({1.0});
        }
        for (int j = 0; j < w.n_q; ++j) {
            w.qids.push_back(pick_ids(1 + (int) (rng() % 5), vocab));
            std::vector<float> q(w.qids[j].size(), 1.0f / (float) w.qids[j].size());
            w.qprobs.push_back(q);
        }
        for (int j = 0; j < w.gamma; ++j) w.d.push_back(rng() % 2 ? arg[j] : (int32_t) (rng() % vocab));
        finish(w);
        int expect = 0;
        while (expect < w.gamma && w.d[expect] == arg[expect]) ++expect;
        double ws[kSpecBlockMaxGamma + 1], r[sbd::kRowCap];
        const SpecBlockResult res = sbd::run_serial(w.view(), ws, r, [&](int) { return uni(); }, [&](int) { return uni(); });
        check(res.accepted == expect && res.token == arg[expect], "one-hot P: longest matching prefix, then the argmax");
        int32_t out[kSpecBlockMaxGamma + 1];
        sbd::fill_out(w.view(), res, [&](int) { return uni(); }, out);
        int a = 0;
        while (a < w.gamma && w.d[a] == out[a]) ++a;
        check(a == expect && out[a] == arg[expect], "the engine loop takes the same tokens at temperature 0");
    }
}

void test_layout_and_route() {
    for (int rows : {2, 3, 9, 17}) {
        std::vector<char> buf(sbd::rows_bytes(rows) + 8, 0x5a);
        void* base = buf.data();
        int* pn = sbd::rows_pn(base);
        int32_t* pid = sbd::rows_pid(base, rows);
        double* pp = sbd::rows_pp(base, rows);
        check((char*) pn + rows * sizeof(int) <= (char*) pid, "pn precedes pid");
        check((char*) (pid + (size_t) rows * sbd::kRowCap) <= (char*) pp, "pid precedes pp");
        check((char*) (pp + (size_t) rows * sbd::kRowCap) <= buf.data() + sbd::rows_bytes(rows), "pp ends inside rows_bytes");
        check(((reinterpret_cast<uintptr_t>(pp) - reinterpret_cast<uintptr_t>(base)) & 7) == 0, "pp is 8-byte aligned from an aligned base");
    }
    // the route: switch off is always the token-by-token path
    for (int sampled = 0; sampled < 2; ++sampled)
        for (int measure = 0; measure < 2; ++measure)
            for (int T = 0; T < 20; ++T)
                for (int nq = 0; nq < 20; ++nq)
                    check(spec_verify_route(false, sampled, measure, T, nq, 16) == VerifyRoute::TokenWise, "switch off: token-wise whatever the window");
    check(spec_verify_route(true, true, false, 4, 3, 16) == VerifyRoute::Block, "switch on, sampled window with lists: block");
    check(spec_verify_route(true, false, false, 4, 3, 16) == VerifyRoute::TokenWise, "switch on, not sampled: token-wise");
    check(spec_verify_route(true, true, true, 4, 3, 16) == VerifyRoute::TokenWise, "switch on, proposal measurement: token-wise");
    check(spec_verify_route(true, true, false, 1, 1, 16) == VerifyRoute::TokenWise, "switch on, no draft: token-wise");
    check(spec_verify_route(true, true, false, 4, 0, 16) == VerifyRoute::TokenWise, "switch on, no q lists: token-wise");
    check(spec_verify_route(true, true, false, 17, 16, 16) == VerifyRoute::TokenWise, "switch on, window above the limit: token-wise");
    check(spec_block_verify_on_from(nullptr) == false && spec_block_verify_on_from("0") == false && spec_block_verify_on_from("1") == true,
          "the switch reads on only for a value starting with 1");
}
}  // namespace

int main() {
    test_equals_rule();
    test_greedy();
    test_layout_and_route();
    test_engine_loop_law();
    std::printf("spec_block_dev_test: %d checks passed\n", checks);
    return 0;
}
