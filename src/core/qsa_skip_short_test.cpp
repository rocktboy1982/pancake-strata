// LEADS.md Part S row S10: the host-testable part of STRATA_QSA_SKIP_SHORT (include/strata/core/qsa_skip_short.hpp).
// Validated: the parse and the clamp, the skip rule, the variant picker (default = the old sel_long_ expression), and that
// below the bound the reference top-k selection IS the identity table qsa_identity_ids writes (host reference with the
// kernel's tie rule), so a skip leaves the table byte-identical.  NOT validated: any engine wiring (there is none yet),
// the skipped kernels on a GPU, any speed.
#include "strata/core/qsa_skip_short.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

using namespace strata::core;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

// host reference of qsa_block_topk's result over CELLS: the `width` largest scores, ties to the lowest index, ascending ids
static std::vector<int32_t> ref_topk(const std::vector<float>& sc, int64_t width) {
    std::vector<int32_t> idx(sc.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) { return sc[(size_t) a] > sc[(size_t) b]; });
    idx.resize((size_t) width);
    std::sort(idx.begin(), idx.end());
    return idx;
}

int main() {
    check(!qsa_skip_short_parse(nullptr) && !qsa_skip_short_parse("") && !qsa_skip_short_parse("0") && !qsa_skip_short_parse("2") &&
              !qsa_skip_short_parse("11") && qsa_skip_short_parse("1"), "only 1 switches it on");
    check(kQsaExactBound == 2048 + 4 - 1, "the bound is idx_top_k + idx_block - 1");
    check(qsa_skip_below_parse(nullptr) == 2051 && qsa_skip_below_parse("") == 2051 && qsa_skip_below_parse("abc") == 2051 &&
              qsa_skip_below_parse("0") == 2051 && qsa_skip_below_parse("-5") == 2051 && qsa_skip_below_parse("1000") == 1000 &&
              qsa_skip_below_parse("1") == 1 && qsa_skip_below_parse("2051") == 2051 && qsa_skip_below_parse("5000") == 2051 &&
              qsa_skip_below_parse("100x") == 2051, "threshold parse: default 2051, 1..2051, larger clamped to the bound");

    // the skip rule: the LAST row decides (n_kv = pos0 + T)
    check(qsa_skip_ok(0, 1, 2051) && qsa_skip_ok(2050, 1, 2051) && !qsa_skip_ok(2051, 1, 2051), "a one-row window at pos0 2050 (n_kv 2051) may skip, at 2051 not");
    check(qsa_skip_ok(2043, 8, 2051) && !qsa_skip_ok(2044, 8, 2051), "an 8-row window: last row n_kv 2051 yes, 2052 no");
    check(!qsa_skip_ok(0, 0, 2051) && !qsa_skip_ok(-1, 1, 2051), "no rows or a negative start never skips");
    check(qsa_skip_ok(500, 8, 600) && !qsa_skip_ok(595, 8, 600), "a lower threshold is honoured");

    // the variant picker keeps today's expression when the skip is off
    const int64_t long_at = 20000;
    for (int64_t pos0 : {0, 100, 2049, 2051, 5000, 19990, 20000, 40000})
        for (int64_t T : {1, 3, 8}) {
            const int old_sel_long = (pos0 + T >= long_at) ? 1 : 0;   // select_long_on() && pos0 + T >= select_long_at()
            check(qsa_variant(true, long_at, false, 2051, pos0, T) == old_sel_long, "skip off, long on: the old variant");
            check(qsa_variant(false, long_at, false, 2051, pos0, T) == 0, "skip off, long off: variant 0");
        }
    check(qsa_variant(false, long_at, true, 2051, 100, 4) == 2 && qsa_variant(true, long_at, true, 2051, 100, 4) == 2, "short window, skip on: variant 2");
    check(qsa_variant(false, long_at, true, 2051, 5000, 4) == 0, "past the bound: variant 0");
    check(qsa_variant(true, long_at, true, 2051, 25000, 4) == 1, "long select wins past long_at");

    // below the bound the selection is the identity: reference top-k == identity table, for random scores
    uint32_t rng = 12345u;
    auto rnd = [&]() { rng = rng * 1664525u + 1013904223u; return (float) (rng >> 8) / 16777216.0f; };
    for (int64_t n_kv : {1, 2, 3, 4, 5, 63, 64, 65, 1000, 2047, 2048, 2049, 2050, 2051}) {
        std::vector<float> sc((size_t) n_kv);
        for (auto& x : sc) x = rnd();
        if (n_kv > 10) { sc[3] = sc[7] = 0.5f; }   // ties
        const int64_t width = n_kv < kQsaExactBound ? n_kv : kQsaExactBound;
        const std::vector<int32_t> ref = ref_topk(sc, width);
        std::vector<int32_t> id((size_t) width);
        qsa_identity_ids(id.data(), width);
        check(ref == id, "below / at the bound the top-k table is the identity table (byte-identical)");
    }
    // one cell past the bound it is not (a score that loses is dropped): why the threshold is clamped
    {
        std::vector<float> sc(2052, 1.0f);
        sc[10] = -1.0f;   // the worst cell is dropped
        const std::vector<int32_t> ref = ref_topk(sc, kQsaExactBound);
        std::vector<int32_t> id((size_t) kQsaExactBound);
        qsa_identity_ids(id.data(), kQsaExactBound);
        check(ref != id, "past the bound the selection is not the identity: a skip would be lossy");
    }
    std::printf("qsa_skip_short_test: %d checks passed\n", checks);
    return 0;
}
