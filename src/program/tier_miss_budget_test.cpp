// MOD_PLAN.md section 6 item 20: STRATA_TIER_MISS_BUDGET's pure rule (include/strata/program/tier_miss_budget.hpp).
#include "strata/program/tier_miss_budget.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::program;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

int main() {
    TierMissBudget m;
    check(!m.on(), "default object is off");
    check(!m.may_stay_out(0, 0), "unbuilt: nothing may stay out");
    check(!m.load_text(""), "empty text rejected");
    check(!m.load_text("WRONG 1 1 2\n1 2\n"), "bad magic rejected");
    check(!m.load_text("strata-gate-importance 2 1 2\n1 2\n"), "bad version rejected");
    check(!m.load_text("strata-gate-importance 1 2 2\n1 2\n"), "short table rejected");
    check(!m.load_text("strata-gate-importance 1 1 2\n1 -2\n"), "negative value rejected");
    // fuzz regression (local fuzz): a header that claims more values than the text can hold, or whose product overflows, is refused
    check(!m.load_text("strata-gate-importance 1 4000000000 4000000000\n1 2 3"), "4e9 x 4e9 header refused (no overflow, no allocation)");
    check(!m.load_text("strata-gate-importance 1 3037000500 3037000500\n1"), "product just past int64 refused");
    check(!m.load_text("strata-gate-importance 1 1 4000000000\n1"), "one layer of 4e9 experts in a tiny file refused");
    check(!m.on() && m.layers() == 0, "refused header leaves the object off");
    // layer 0: shares 0.05 0.05 0.30 0.60 (scale 20); layer 1: all zero; layer 2: uniform
    check(m.load_text("strata-gate-importance 1 3 4\n1 1 6 12\n0 0 0 0\n1 1 1 1\n"), "table loads");
    check(m.shape_matches(3, 4) && !m.shape_matches(3, 5), "shape check");
    m.build(0.0);
    check(!m.on(), "budget 0 marks nothing");
    m.build(0.04);
    check(!m.on(), "budget below the smallest share marks nothing");
    m.build(0.10);
    check(m.may_stay_out(0, 0) && m.may_stay_out(0, 1), "budget 0.10: the two 0.05 experts may stay out");
    check(!m.may_stay_out(0, 2) && !m.may_stay_out(0, 3), "budget 0.10: the heavier ones stay in tier reach");
    check(!m.may_stay_out(1, 0), "all-zero layer: no evidence, nothing marked");
    check(!m.may_stay_out(2, 0), "uniform layer at 0.10: one expert is 0.25 > 0.10");
    m.build(0.45);
    check(m.may_stay_out(0, 2) && !m.may_stay_out(0, 3), "budget 0.45: 0.05+0.05+0.30 = 0.40 marked, the 0.60 expert is not");
    check(m.may_stay_out(2, 0) && !m.may_stay_out(2, 1), "uniform layer at 0.45: ties break by id, one expert marked");
    check(m.n_stay_out() == 4, "count: 3 in layer 0, 1 in layer 2");
    check(!m.may_stay_out(-1, 0) && !m.may_stay_out(3, 0) && !m.may_stay_out(0, 4) && !m.may_stay_out(0, -1), "out of range is false");
    m.build(0.999);
    check(!m.may_stay_out(0, 3), "the heaviest expert (0.60) is not marked below budget 1");
    std::printf("tier_miss_budget_test: %d checks ok\n", checks);
    return 0;
}
