// MOD_PLAN.md section 6 items 19 / 26: STRATA_ADAPT_PROTECT's pure logic (include/strata/program/tier_protect.hpp).
#include "strata/program/tier_protect.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::program;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

int main() {
    check(tier_protect_slots(0.0, 10909) == 0, "f 0 protects nothing");
    check(tier_protect_slots(-0.5, 10909) == 0, "negative f protects nothing");
    check(tier_protect_slots(0.5, 10909) == 5455, "0.5 of 10909 rounds up to 5455");
    check(tier_protect_slots(0.75, 10909) == 8182, "0.75 of 10909 -> ceil(8181.75) = 8182");
    check(tier_protect_slots(0.9, 10909) == 9819, "0.9 of 10909 -> ceil(9818.1) = 9819");
    check(tier_protect_slots(0.5, 100) == 50, "exact product is not rounded up");
    check(tier_protect_slots(1.0, 10909) == 10909, "f 1 protects every slot");
    check(tier_protect_slots(7.0, 10) == 10, "f above 1 clamps to slots");
    check(tier_protect_slots(0.5, 0) == 0, "no slots, nothing to protect");
    check(tier_protect_slots(std::nan(""), 10) == 0, "nan is off");
    check(tier_protect_holds(0, 5), "slot 0 is below 5");
    check(tier_protect_holds(4, 5), "slot 4 is below 5");
    check(!tier_protect_holds(5, 5), "slot 5 is not below 5");
    check(!tier_protect_holds(-1, 5), "a non-resident (-1) is never protected");
    check(!tier_protect_holds(3, 0), "n_protected 0 protects nothing");
    std::printf("tier_protect_test: %d checks ok\n", checks);
    return 0;
}
