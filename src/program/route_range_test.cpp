// MOD_PLAN.md section 6 (the BigMoeOnEdge switches): STRATA_ROUTE_RANGE's rule (include/strata/program/route_range.hpp).
#include "strata/program/route_range.hpp"

#include <cstdio>
#include <initializer_list>
#include <cstdlib>
#include <cmath>

using namespace strata::program;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

int main() {
    check(route_range_bonus(0.15f, 5.0f, 1.0f) == 0.15f * 4.0f, "bonus = L x (max - min)");
    check(route_range_bonus(0.0f, 5.0f, 1.0f) == 0.0f && route_range_bonus(-1.0f, 5.0f, 1.0f) == 0.0f, "L <= 0: no bonus");
    check(route_range_bonus(0.15f, 1.0f, 5.0f) == 0.0f, "an inverted span gives no bonus");
    check(route_range_bonus(0.15f, INFINITY, 0.0f) == 0.0f && route_range_bonus(0.15f, NAN, 0.0f) == 0.0f, "non-finite span: no bonus");

    check(route_range_parse(nullptr) == 0.0f && route_range_parse("") == 0.0f, "unset is off");
    check(route_range_parse("0.15") == 0.15f, "0.15 parses");
    check(route_range_parse("0") == 0.0f && route_range_parse("-0.2") == 0.0f, "0 and negative are off");
    check(route_range_parse("5") == 0.0f && route_range_parse("abc") == 0.0f && route_range_parse("nan") == 0.0f, "above 4, text and nan are off, not clamped");
    check(route_range_parse("4") == 4.0f, "4 is the largest accepted value");

    // 8 experts, top-3; experts 5 and 6 resident, just under the router's third pick
    const float lg[8] = {3.0f, 2.0f, 1.0f, 0.9f, 0.1f, 0.8f, 0.7f, -1.0f};   // router top-3: 0 1 2
    const int32_t res[8] = {-1, -1, -1, -1, -1, 7, 8, -1};
    int32_t ids[3];
    check(route_range_select(lg, 8, res, 0.0f, 3, ids) == 0 && ids[0] == 0 && ids[1] == 1 && ids[2] == 2, "L = 0: the router's own top-3, nothing substituted");
    // range 4.0: L=0.02 -> bonus 0.08: expert 5 (0.8 + 0.08 = 0.88) stays under expert 2 (1.0)
    check(route_range_select(lg, 8, res, 0.02f, 3, ids) == 0, "a small L substitutes nothing");
    // L=0.1 -> bonus 0.4: expert 5 scores 1.2 > expert 2's 1.0
    check(route_range_select(lg, 8, res, 0.1f, 3, ids) == 1 && ids[2] == 5, "L = 0.1: the third pick becomes resident expert 5");
    // L=0.2 -> bonus 0.8: expert 5 1.6 still takes expert 2's place, expert 6 (1.5) is fourth
    check(route_range_select(lg, 8, res, 0.2f, 3, ids) == 1, "L = 0.2: still one substitution");
    // L=0.5 -> bonus 2.0: expert 5 2.8 and expert 6 2.7 both pass experts 1 (2.0) and 2 (1.0); expert 0 (3.0) holds
    check(route_range_select(lg, 8, res, 0.5f, 3, ids) == 2 && ids[0] == 0 && ids[1] == 5 && ids[2] == 6, "L = 0.5: two substitutions");

    // substitutions never decrease as L grows
    int prev = 0;
    bool mono = true;
    for (float L = 0.0f; L <= 1.0f; L += 0.05f) {
        const int s = route_range_select(lg, 8, res, L, 3, ids);
        mono = mono && s >= prev;
        prev = s;
    }
    check(mono, "substitutions are monotone in L");

    // the point of the range-relative form: scale and shift of the logits do not change the selection
    float lg2[8];
    for (int e = 0; e < 8; ++e) lg2[e] = 7.0f * lg[e] - 20.0f;
    int32_t ids2[3];
    for (const float L : {0.02f, 0.1f, 0.2f}) {
        route_range_select(lg, 8, res, L, 3, ids);
        route_range_select(lg2, 8, res, L, 3, ids2);
        check(ids[0] == ids2[0] && ids[1] == ids2[1] && ids[2] == ids2[2], "scaled and shifted logits select the same experts");
    }
    // a flat token (tiny range) gets a tiny bonus: the router's order stays; an absolute bonus of 0.4 would reorder it
    const float flat[8] = {1.000f, 0.999f, 0.998f, 0.997f, 0.990f, 0.996f, 0.995f, 0.994f};   // range 0.01, router top-3: 0 1 2
    check(route_range_select(flat, 8, res, 0.1f, 3, ids) == 0, "flat token: bonus 0.001 changes nothing");
    std::printf("route_range_test: %d checks passed\n", checks);
    return 0;
}
