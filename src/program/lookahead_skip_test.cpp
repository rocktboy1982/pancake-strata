// MOD_PLAN.md section 6 (the BigMoeOnEdge switches, LEADS.md Part I): STRATA_LOOKAHEAD_SKIP_DROPPED's rule (include/strata/program/lookahead_skip.hpp).
#include "strata/program/lookahead_skip.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

using namespace strata::program;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

int main() {
    const float top[4] = {2.0f, 1.0f, 0.0f, -1.0f};
    float sh[4];
    lookahead_shares(top, 4, sh);
    float sum = 0.f;
    for (float s : sh) sum += s;
    check(std::fabs(sum - 1.0f) < 1e-6f, "shares sum to 1");
    check(sh[0] > sh[1] && sh[1] > sh[2] && sh[2] > sh[3], "shares follow the logits");
    check(std::fabs(sh[0] / sh[1] - std::exp(1.0f)) < 1e-4f, "ratio of shares = exp(logit gap)");
    float big[2] = {1000.0f, 999.0f}, sb[2];
    lookahead_shares(big, 2, sb);
    check(sb[0] == sb[0] && std::fabs(sb[0] + sb[1] - 1.0f) < 1e-6f, "large logits do not overflow");
    lookahead_shares(top, 0, sh);   // no write, no crash
    check(true, "k = 0 is a no-op");

    // _W: share under the floor drops; the rule needs share < floor strictly
    check(!lookahead_would_drop(sh, 4, 3, 0.0f, 0.0f), "no rule set: never drops");
    lookahead_shares(top, 4, sh);
    check(lookahead_would_drop(sh, 4, 3, 0.10f, 0.0f), "share 0.0321 < 0.10 drops");
    check(!lookahead_would_drop(sh, 4, 0, 0.10f, 0.0f), "the top pick (0.64) stays");
    check(!lookahead_would_drop(sh, 4, 3, sh[3], 0.0f), "share == floor stays (strict <)");
    // _CUM: the mass above reaches p
    check(!lookahead_would_drop(sh, 4, 0, 0.0f, 0.5f), "rank 0 has nothing above it");
    check(lookahead_would_drop(sh, 4, 1, 0.0f, 0.5f), "rank 1: 0.64 above >= 0.5 drops");
    check(!lookahead_would_drop(sh, 4, 1, 0.0f, 0.9f), "rank 1: 0.64 above < 0.9 stays");
    check(lookahead_would_drop(sh, 4, 3, 0.0f, 0.9f), "rank 3: 0.64 + 0.235 + 0.087 = 0.96 >= 0.9 drops");
    // either rule drops
    check(lookahead_would_drop(sh, 4, 2, 0.01f, 0.5f), "_CUM drops although _W would not");
    // edge cases
    check(!lookahead_would_drop(sh, 4, 4, 0.5f, 0.5f) && !lookahead_would_drop(sh, 4, -1, 0.5f, 0.5f), "out of range j stays");
    float nan[1] = {std::numeric_limits<float>::quiet_NaN()};
    check(!lookahead_would_drop(nan, 1, 0, 0.5f, 0.5f), "a NaN share never drops");
    std::printf("lookahead_skip_test: %d checks passed\n", checks);
    return 0;
}
