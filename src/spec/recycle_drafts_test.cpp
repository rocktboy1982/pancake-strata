// src/spec/recycle_drafts_test.cpp - STRATA_TOKEN_RECYCLE / STRATA_SSD_LITE / STRATA_HYDRA_TOP2 (recycle_drafts.hpp) and
// STRATA_SPEC_REASON's step rule (spec_reason.hpp): the matrix, the prepared outcomes (worker and synchronous agree), the const
// lookup against propose_after, and the step acceptance.  CPU only, no model.  LEADS.md section 3.
#include "strata/spec/recycle_drafts.hpp"
#include "strata/spec/spec_reason.hpp"
#include "strata/spec/suffix_drafter.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace strata::spec;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}

void test_matrix() {
    TokenRecycler r;
    check(r.first(5) == -1 && r.propose(5, 4, nullptr) == 0, "empty matrix proposes nothing");
    const int32_t win[4] = {1, 2, 3, 4};
    const int32_t out[4] = {2, 3, 9, 5};   // rows 0, 1 accepted; row 2 rejected (the target said 9 after 3)
    r.observe(win, out, 4);
    check(r.first(1) == 2 && r.first(3) == 9 && r.first(4) == 5, "every row is a pair");
    int32_t ch[8];
    const int n = r.propose(1, 8, ch);
    check(n == 3 && ch[0] == 2 && ch[1] == 3 && ch[2] == 9, "chain follows the first candidates");
    r.push(3, 7);
    check(r.first(3) == 7 && r.alt(3, 7) == 9, "most recent first, runner-up next");
    r.push(3, 9);
    check(r.first(3) == 9 && r.alt(3, 9) == 7, "an older copy moves up, no duplicate");
    for (int i = 0; i < 20; ++i) r.push(3, 100 + i);
    int32_t row[TokenRecycler::K];
    check(r.row(3, row) == TokenRecycler::K && row[0] == 119, "a row holds K candidates");
    r.push(-1, 3);
    r.push(TokenRecycler::kMaxVocab, 3);
    check(r.first(-1) == -1, "out-of-range ids ignored");
}

void test_const_lookup() {
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> d(0, 11);   // a small alphabet: many repeats
    for (int round = 0; round < 50; ++round) {
        SuffixDrafter s(3, 32, 4096);
        std::vector<int32_t> h(400);
        for (auto& x : h) x = d(rng);
        s.append(h.data(), h.size());
        for (int ne = 0; ne < 6; ++ne) {
            int32_t ex[6];
            for (int i = 0; i < ne; ++i) ex[i] = d(rng);
            int32_t a[16], b[16];
            const int ka = s.propose_after(ex, ne, 16, a);
            const int ma = s.last_match();
            int mb = -1;
            const int kb = s.propose_after_const(ex, ne, 16, b, &mb);
            bool same = ka == kb && (ka == 0 || ma == mb);
            for (int i = 0; same && i < ka; ++i) same = a[i] == b[i];
            check(same, "propose_after_const == propose_after");
            check(s.size() == h.size(), "history unchanged");
        }
    }
}

void test_outcomes() {
    // history: a repeated phrase; the window drafts part of it
    SuffixDrafter s(3, 32, 4096);
    std::vector<int32_t> h;
    for (int rep = 0; rep < 3; ++rep)
        for (int t = 10; t < 30; ++t) h.push_back(t);
    h.push_back(10);
    h.push_back(11);
    h.push_back(12);
    s.append(h.data(), h.size());
    const int32_t win[4] = {12, 13, 14, 99};   // row 0's input is the last token; drafts 13, 14, 99
    OutcomeDrafter sync;
    sync.compute(s, nullptr, win, 4, 6, false);
    // outcome a = 2 (13, 14 accepted, 99 rejected): the lookup guesses 15, then 16..
    sync.finish(2, 15, 100);   // (finish after compute alone: not started, so nothing is recorded)
    int32_t out[8];
    int m = 0;
    check(sync.take(100, 8, out, &m) == 0, "an outcome of a round not started is ignored");
    OutcomeDrafter w;
    w.start(s, nullptr, win, 4, 6, false);
    w.wait();
    w.finish(2, 15, 100);
    const int n = w.take(100, 8, out, &m);
    check(n == 6 && out[0] == 16 && out[5] == 21 && m >= 3, "the prepared draft after the outcome (worker)");
    check(w.take(100, 8, out, &m) == 0, "an outcome is taken once");
    w.start(s, nullptr, win, 4, 6, false);
    w.wait();
    w.finish(2, 15, 100);
    check(w.take(101, 8, out, &m) == 0, "another position: nothing");
    w.start(s, nullptr, win, 4, 6, false);
    w.wait();
    w.finish(2, 77, 100);
    check(w.take(100, 8, out, &m) == 0, "a bonus not prepared: nothing");
    // top-2: the matrix knows that after 14 the target said 40 (runner-up of 15) and 41 after 40
    TokenRecycler r;
    r.push(14, 40);
    r.push(14, 99);   // the draft itself first, so the runner-up must skip it
    r.push(40, 41);
    r.push(41, 42);
    w.start(s, &r, win, 4, 6, true);
    w.wait();
    w.finish(2, 40, 100);
    const int n2 = w.take(100, 8, out, &m);
    check(n2 >= 2 && out[0] == 41 && out[1] == 42 && m == 0, "the runner-up branch's continuation (matrix chain)");
    w.start(s, &r, win, 4, 6, false);
    w.wait();
    w.finish(2, 40, 100);
    check(w.take(100, 8, out, &m) == 0, "no runner-up without top2 (the first candidate is the rejected draft)");
    check(w.hits() == 2 && w.asks() == 5, "counts");
    // full acceptance (a = 3): the bonus after 99 is unknown to the lookup; nothing prepared, nothing taken
    w.start(s, nullptr, win, 4, 6, false);
    w.wait();
    w.finish(3, 5, 100);
    check(w.take(100, 8, out, &m) == 0, "no guess for an unseen context");
}

void test_spec_reason() {
    const double p1[3] = {0.5, 0.6, 0.7};
    check(spec_reason_accept(p1, 3, 0.5, 0.1) == 3, "a likely step is kept whole");
    check(spec_reason_accept(p1, 3, 0.7, 0.1) == 0, "below tau: nothing");
    const double p2[3] = {0.9, 0.05, 0.9};
    check(spec_reason_accept(p2, 3, 0.3, 0.1) == 0, "one draft below the floor rejects the step");
    check(spec_reason_accept(p1, 0, 0.5, 0.1) == 0 && spec_reason_accept(p1, 3, 0.0, 0.0) == 0, "empty / off");
    const float l[4] = {0.0f, std::log(3.0f), NAN, -INFINITY};
    check(std::fabs(row_prob(l, 4, 1) - 0.75) < 1e-6 && row_prob(l, 4, 2) == 0.0 && row_prob(l, 4, 3) == 0.0, "row_prob");
    check(row_prob(l, 4, 7) == 0.0, "row_prob out of range");
}
}  // namespace

int main() {
    test_matrix();
    test_const_lookup();
    test_outcomes();
    test_spec_reason();
    if (g_fail) { std::fprintf(stderr, "recycle_drafts_test: %d failure(s)\n", g_fail); return 1; }
    std::printf("recycle_drafts_test: ok\n");
    return 0;
}
