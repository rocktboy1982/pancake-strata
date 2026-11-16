// MOD_PLAN.md section 6 (the BigMoeOnEdge switches): STRATA_PREFETCH_PREV's rule (include/strata/program/prefetch_prev.hpp).
#include "strata/program/prefetch_prev.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::program;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

int main() {
    check(prefetch_prev_parse(nullptr) == 0 && prefetch_prev_parse("") == 0, "unset is off");
    check(prefetch_prev_parse("1") == 1 && prefetch_prev_parse("8") == 8, "1 and 8 are accepted");
    check(prefetch_prev_parse("0") == 0 && prefetch_prev_parse("9") == 0 && prefetch_prev_parse("-1") == 0, "0, 9 and negatives are off");
    check(prefetch_prev_parse("2x") == 0 && prefetch_prev_parse("abc") == 0 && prefetch_prev_parse("1.5") == 0, "text after or instead of a number is off");

    int32_t out[16];
    const int32_t prev[6] = {7, 3, 7, -1, 99, 5};
    check(prefetch_prev_collect(prev, 6, nullptr, 64, out) == 3 && out[0] == 7 && out[1] == 3 && out[2] == 5,
          "dedup, first-seen order, -1 and out-of-range dropped");
    int32_t res[64];
    for (int i = 0; i < 64; ++i) res[i] = -1;
    res[3] = 0;   // expert 3 resident in the GPU cache
    check(prefetch_prev_collect(prev, 6, res, 64, out) == 2 && out[0] == 7 && out[1] == 5, "a resident expert is not warmed");
    for (int i = 0; i < 64; ++i) res[i] = i;
    check(prefetch_prev_collect(prev, 6, res, 64, out) == 0, "everything resident: nothing to warm");
    check(prefetch_prev_collect(prev, 0, nullptr, 64, out) == 0, "no previous routing: nothing to warm");
    int32_t many[40];
    for (int i = 0; i < 40; ++i) many[i] = i;
    check(prefetch_prev_collect(many, 40, nullptr, 64, out) == kPrefetchPrevMaxIds, "at most 16 ids");
    std::printf("prefetch_prev_test: %d checks passed\n", checks);
    return 0;
}
