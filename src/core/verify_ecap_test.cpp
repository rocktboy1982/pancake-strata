// src/core/verify_ecap_test.cpp - MOD_PLAN F5 STRATA_VERIFY_ECAP: the draft-row expert cap (core/verify_ecap.hpp), CPU only.
// Worked windows, then random windows (shaped like the router: top-10 of distinct experts per token, a 64-expert layer so
// rows share experts) against the rule's promises: the committed row is never touched, only missed experts are dropped
// and always all of an expert's entries together, a draft row adds at most `cap` experts nobody else reads, a dropped
// entry has `cap` kept missed entries ranked above it in its own row, and a cap of k or more drops nothing.
#include "strata/core/verify_ecap.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <set>
#include <vector>

using namespace strata::core;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

namespace {

struct Win {
    int64_t n_tok = 0, k = 0;
    std::vector<int32_t> ids;
    std::vector<int64_t> first_of;
    std::vector<char> missed;   // per entry (vector<bool> has no data())
};

void fill_first_of(Win& w) {
    const int64_t n = w.n_tok * w.k;
    w.first_of.assign((size_t) n, 0);
    for (int64_t i = 0; i < n; ++i) {
        w.first_of[(size_t) i] = i;
        for (int64_t j = 0; j < i; ++j)
            if (w.ids[(size_t) j] == w.ids[(size_t) i]) { w.first_of[(size_t) i] = j; break; }
    }
}

// runs the rule; returns the drop flags
std::vector<char> run(const Win& w, int64_t first, int cap, int64_t* dropped = nullptr) {
    const size_t n = (size_t) (w.n_tok * w.k);
    std::unique_ptr<bool[]> missed(new bool[n]), read(new bool[n]), drop(new bool[n]);
    for (size_t i = 0; i < n; ++i) missed[i] = w.missed[i] != 0;
    const int64_t d = verify_ecap_drop(w.first_of.data(), missed.get(), w.n_tok, w.k, first, cap, read.get(), drop.get());
    if (dropped != nullptr) *dropped = d;
    std::vector<char> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = drop[i] ? 1 : 0;
    return out;
}

Win make(int64_t n_tok, int64_t k, std::vector<int32_t> ids, const std::set<int32_t>& missed_experts) {
    Win w;
    w.n_tok = n_tok;
    w.k = k;
    w.ids = std::move(ids);
    w.missed.resize(w.ids.size());
    for (size_t i = 0; i < w.ids.size(); ++i) w.missed[i] = missed_experts.count(w.ids[i]) ? 1 : 0;
    fill_first_of(w);
    return w;
}

// the scalar rule as it was before ECAP v2 (frozen copy): the new code must equal it when no list and no gate are set
int64_t ref_drop(const Win& w, int64_t first, int cap, std::vector<char>& out) {
    const int64_t n = w.n_tok * w.k;
    std::vector<char> read((size_t) n, 0);
    out.assign((size_t) n, 0);
    if (cap < 0) return 0;
    for (int64_t i = 0; i < first * w.k; ++i)
        if (w.missed[(size_t) i]) read[(size_t) w.first_of[(size_t) i]] = 1;
    for (int64_t t = first; t < w.n_tok; ++t) {
        int added = 0;
        for (int64_t j = 0; j < w.k; ++j) {
            const int64_t i = t * w.k + j;
            if (!w.missed[(size_t) i] || read[(size_t) w.first_of[(size_t) i]]) continue;
            if (added < cap) { read[(size_t) w.first_of[(size_t) i]] = 1; ++added; }
        }
    }
    int64_t d = 0;
    for (int64_t i = 0; i < n; ++i)
        if (w.missed[(size_t) i] && !read[(size_t) w.first_of[(size_t) i]]) { out[(size_t) i] = 1; ++d; }
    return d;
}

// the v2 rule on a window or one token group of it
std::vector<char> run2(const Win& w, int64_t first, int64_t base, int cap, const int8_t* rows, int nr, const float* conf,
                       int nc, float minp, int64_t* dropped = nullptr) {
    const size_t n = (size_t) (w.n_tok * w.k);
    std::unique_ptr<bool[]> missed(new bool[n]), read(new bool[n]), drop(new bool[n]);
    for (size_t i = 0; i < n; ++i) missed[i] = w.missed[i] != 0;
    const int64_t d = verify_ecap_drop_rows(w.first_of.data(), missed.get(), w.n_tok, w.k, first, base, cap, rows, nr, conf,
                                            nc, minp, read.get(), drop.get());
    if (dropped != nullptr) *dropped = d;
    std::vector<char> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = drop[i] ? 1 : 0;
    return out;
}

}  // namespace

int main() {
    // ---- ECAP v2: the list and the gate parse
    {
        int8_t r[kEcapRowsMax] = {};
        CHECK(verify_ecap_rows_parse(nullptr, r) == 0);
        CHECK(verify_ecap_rows_parse("", r) == 0);
        CHECK(verify_ecap_rows_parse("0", r) == 1 && r[0] == 0);
        CHECK(verify_ecap_rows_parse("-1,-1,0", r) == 3 && r[0] == -1 && r[1] == -1 && r[2] == 0);
        CHECK(verify_ecap_rows_parse("1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16", r) == 16);
        CHECK(verify_ecap_rows_parse("1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17", r) == -2);
        CHECK(verify_ecap_rows_parse("-2", r) == -2);
        CHECK(verify_ecap_rows_parse("64", r) == -2);
        CHECK(verify_ecap_rows_parse("1,,2", r) == -2);
        CHECK(verify_ecap_rows_parse("1,", r) == -2);
        CHECK(verify_ecap_rows_parse("1;2", r) == -2);
        CHECK(verify_ecap_minp_parse(nullptr) == -1.0f);
        CHECK(verify_ecap_minp_parse("") == -1.0f);
        CHECK(verify_ecap_minp_parse("0.5") == 0.5f);
        CHECK(verify_ecap_minp_parse("1") == 1.0f);
        CHECK(verify_ecap_minp_parse("0") == -2.0f);
        CHECK(verify_ecap_minp_parse("1.5") == -2.0f);
        CHECK(verify_ecap_minp_parse("x") == -2.0f);
        const int8_t l[3] = {-1, -1, 0};
        CHECK(verify_ecap_row_cap(l, 3, 7, 1) == -1 && verify_ecap_row_cap(l, 3, 7, 3) == 0 &&
              verify_ecap_row_cap(l, 3, 7, 9) == 0);   // the last value repeats
        CHECK(verify_ecap_row_cap(nullptr, 0, 7, 2) == 7);   // no list: the scalar
        ecap_conf_publish(nullptr, 0);
        CHECK(ecap_conf_board().n == 0);
        const float pp[3] = {0.9f, 0.2f, 0.1f};
        ecap_conf_publish(pp, 3);
        CHECK(ecap_conf_board().n == 3 && ecap_conf_board().p[1] == 0.2f);
        ecap_conf_publish(nullptr, 0);
    }
    // ---- ECAP v2 equals the old scalar rule when no list and no gate are set; a list of one repeated value equals the
    // scalar; a gate nothing falls under changes nothing (random windows)
    {
        std::mt19937 rg(11);
        for (int it = 0; it < 3000; ++it) {
            const int T = 2 + (int) (rg() % 9);
            const int64_t first = (int64_t) (rg() % 2);
            const int cap = (int) (rg() % 6) - 1;   // -1..4
            std::set<int32_t> me;
            for (int e = 0; e < 32; ++e)
                if (rg() % 100 < 50) me.insert(e);
            std::vector<int32_t> ids;
            for (int t = 0; t < T; ++t) {
                std::vector<int32_t> pool(32);
                for (int e = 0; e < 32; ++e) pool[(size_t) e] = e;
                std::shuffle(pool.begin(), pool.end(), rg);
                ids.insert(ids.end(), pool.begin(), pool.begin() + 6);
            }
            const Win w = make(T, 6, ids, me);
            std::vector<char> ref;
            const int64_t dref = ref_drop(w, first, cap, ref);
            int64_t d1 = -1, d2 = -1;
            const std::vector<char> a = run(w, first, cap, &d1);
            CHECK(a == ref && d1 == dref);
            if (cap >= 0) {
                const int8_t one[1] = {(int8_t) cap};
                const std::vector<char> b = run2(w, first, 1 - first, cap, one, 1, nullptr, 0, 0.0f, &d2);
                CHECK(b == ref && d2 == dref);
                const int8_t many[4] = {(int8_t) cap, (int8_t) cap, (int8_t) cap, (int8_t) cap};
                CHECK(run2(w, first, 1 - first, 99, many, 4, nullptr, 0, 0.0f) == ref);   // the list replaces the scalar
                const float hi[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
                CHECK(run2(w, first, 1 - first, cap, nullptr, 0, hi, 16, 0.5f) == ref);
            }
        }
    }
    // ---- ECAP v2 worked windows: 4 rows, k 2, row 0 reads expert 1; rows 1..3 each name two new missed experts
    {
        const Win w = make(4, 2, {1, 9, 2, 3, 4, 5, 6, 7}, {1, 2, 3, 4, 5, 6, 7});
        int64_t d = 0;
        // cap 0 for rows >= 3 only: rows 1 and 2 are uncapped (-1), row 3 reads nothing new
        const int8_t l3[3] = {-1, -1, 0};
        std::vector<char> o = run2(w, 1, 0, -1, l3, 3, nullptr, 0, 0.0f, &d);
        CHECK(d == 2 && o[6] == 1 && o[7] == 1 && o[2] == 0 && o[3] == 0 && o[4] == 0 && o[5] == 0);
        // rows >= 2 capped at 0: rows 2 and 3 drop their four
        const int8_t l2[2] = {-1, 0};
        (void) run2(w, 1, 0, -1, l2, 2, nullptr, 0, 0.0f, &d);
        CHECK(d == 4);
        // cap 1 on row 1 drops only its second missed expert
        const int8_t l1[2] = {1, -1};
        o = run2(w, 1, 0, -1, l1, 2, nullptr, 0, 0.0f, &d);
        CHECK(d == 1 && o[3] == 1);
        // the gate: row 2's probability is below minp, so it gets cap 0 although the list leaves it uncapped; rows 1 and 3
        // are above the gate (a mutation that gates the wrong row index shows here)
        const float conf[3] = {0.9f, 0.1f, 0.9f};   // window rows 1, 2, 3
        const int8_t ul[1] = {-1};
        o = run2(w, 1, 0, -1, ul, 1, conf, 3, 0.5f, &d);
        CHECK(d == 2 && o[4] == 1 && o[5] == 1 && o[2] == 0 && o[6] == 0);
        // the gate alone (no list, scalar -1): the same
        o = run2(w, 1, 0, -1, nullptr, 0, conf, 3, 0.5f, &d);
        CHECK(d == 2 && o[4] == 1 && o[5] == 1);
        // an unknown probability (n_conf too short) is not gated
        (void) run2(w, 1, 0, -1, nullptr, 0, conf, 1, 0.5f, &d);
        CHECK(d == 0);
        // a second token group starting at window row 2 (first 0, row_base 2): its local rows are window rows 2 and 3
        const Win g = make(2, 2, {20, 21, 22, 23}, {20, 21, 22, 23});
        o = run2(g, 0, 2, -1, l3, 3, nullptr, 0, 0.0f, &d);   // window row 2 uncapped (-1), row 3 cap 0
        CHECK(d == 2 && o[2] == 1 && o[3] == 1 && o[0] == 0);
    }

    // ---- the value
    CHECK(verify_ecap_parse(nullptr) == -1);
    CHECK(verify_ecap_parse("") == -1);
    CHECK(verify_ecap_parse("0") == 0);
    CHECK(verify_ecap_parse("2") == 2);
    CHECK(verify_ecap_parse("63") == 63);
    CHECK(verify_ecap_parse("64") == -2);
    CHECK(verify_ecap_parse("-1") == -2);
    CHECK(verify_ecap_parse("two") == -2);
    CHECK(verify_ecap_parse("2x") == -2);

    // ---- worked window A (k 4, 3 rows, cap 1): row 0 reads 11; row 1 has 11 free and adds 20 (21 over the cap);
    // row 2 adds 21 (so row 1's 21 stays), has 20 free, and 30 and 31 are over the cap: only those two are dropped
    {
        const Win w = make(3, 4, {10, 11, 12, 13, 11, 20, 21, 12, 21, 20, 30, 31}, {11, 20, 21, 30, 31});
        int64_t d = 0;
        const std::vector<char> drop = run(w, 1, 1, &d);
        CHECK(d == 2);
        for (size_t i = 0; i < drop.size(); ++i) CHECK(drop[i] == ((i == 10 || i == 11) ? 1 : 0));
        int64_t d0 = -1;
        (void) run(w, 1, 0, &d0);   // cap 0: only row 0's 11 is read, the six other missed entries go
        CHECK(d0 == 6);
        (void) run(w, 1, 2, &d0);   // cap 2: row 1 adds 20 and 21, row 2 adds 30 and 31: nothing goes
        CHECK(d0 == 0);
        (void) run(w, 1, 4, &d0);   // cap k: nothing
        CHECK(d0 == 0);
        (void) run(w, 1, -1, &d0);  // off: nothing
        CHECK(d0 == 0);
    }
    // ---- worked window B (a second token group: no committed row, first 0, cap 0): every missed entry goes
    {
        const Win w = make(2, 2, {1, 2, 1, 3}, {1, 3});
        int64_t d = 0;
        const std::vector<char> drop = run(w, 0, 0, &d);
        CHECK(d == 3);
        CHECK(drop[0] == 1 && drop[1] == 0 && drop[2] == 1 && drop[3] == 1);
    }
    // ---- worked window C (cap 0 with row 0): what row 0 reads stays, a draft row's own missed expert goes
    {
        const Win w = make(2, 2, {1, 2, 2, 3}, {1, 2, 3});
        int64_t d = 0;
        const std::vector<char> drop = run(w, 1, 0, &d);
        CHECK(d == 1);
        CHECK(drop[0] == 0 && drop[1] == 0 && drop[2] == 0 && drop[3] == 1);
    }

    // ---- random windows against the promises
    std::mt19937 rng(5);
    const int E = 64, K = 10;
    int64_t windows = 0, total_dropped = 0;
    for (int it = 0; it < 4000; ++it) {
        const int T = 2 + (int) (rng() % 15);          // 2..16 rows
        const int64_t first = (int64_t) (rng() % 2);    // the group holds row 0, or not
        const int cap = (int) (rng() % 5);              // 0..4
        std::set<int32_t> missed_experts;
        const unsigned miss_pct = 10 + rng() % 70;
        for (int e = 0; e < E; ++e)
            if (rng() % 100 < miss_pct) missed_experts.insert(e);
        std::vector<int32_t> ids;
        for (int t = 0; t < T; ++t) {
            std::vector<int32_t> pool(E);
            for (int e = 0; e < E; ++e) pool[(size_t) e] = e;
            std::shuffle(pool.begin(), pool.end(), rng);
            ids.insert(ids.end(), pool.begin(), pool.begin() + K);   // distinct per token, routing order
        }
        const Win w = make(T, K, ids, missed_experts);
        int64_t d = 0;
        const std::vector<char> drop = run(w, first, cap, &d);
        const std::vector<char> again = run(w, first, cap);
        CHECK(drop == again);   // deterministic
        ++windows;
        total_dropped += d;
        int64_t count = 0;
        std::set<int32_t> committed, read;
        for (int64_t i = 0; i < first * K; ++i)
            if (w.missed[(size_t) i]) committed.insert(w.ids[(size_t) i]);
        for (size_t i = 0; i < drop.size(); ++i) {
            count += drop[i];
            if (drop[i]) CHECK(w.missed[i]);                                  // only missed experts
            if ((int64_t) i < first * K) CHECK(!drop[i]);                     // the committed row untouched
            CHECK(drop[i] == drop[(size_t) w.first_of[i]]);                   // an expert's entries together
            if (drop[i]) CHECK(!committed.count(w.ids[i]));                   // never what row 0 reads
            if (w.missed[i] && !drop[i]) read.insert(w.ids[i]);
        }
        CHECK(count == d);
        // a draft row adds at most `cap` experts: the read set is row 0's plus at most cap per draft row
        CHECK((int64_t) read.size() <= (int64_t) committed.size() + (int64_t) cap * (T - first));
        // the top-ranked ones are kept: a dropped entry has at least `cap` kept missed entries before it in its row,
        // and a draft row keeps at least min(cap, its distinct missed experts)
        for (int64_t t = first; t < T; ++t) {
            int kept_before = 0, distinct_missed = 0, kept = 0;
            for (int64_t j = 0; j < K; ++j) {
                const size_t i = (size_t) (t * K + j);
                if (!w.missed[i]) continue;
                ++distinct_missed;   // (a token's top-k names distinct experts)
                if (drop[i]) CHECK(kept_before >= cap);
                else { ++kept_before; ++kept; }
            }
            CHECK(kept >= std::min(cap, distinct_missed));
        }
        // a cap of k or more never drops anything
        int64_t dk = -1;
        (void) run(w, first, K, &dk);
        CHECK(dk == 0);
    }
    if (total_dropped == 0) { std::fprintf(stderr, "FAIL the random windows never dropped an entry\n"); ++fails; }
    if (fails == 0) std::printf("verify_ecap_test: ok (%lld windows, %lld entries dropped)\n", (long long) windows,
                                (long long) total_dropped);
    return fails == 0 ? 0 : 1;
}
