// MOD_PLAN 17.1: the route prior only in lookup windows (STRATA_ROUTE_PRIOR_AUTO): the switch's parsing, the window kind, the bonus a
// window takes and the gate verdict next to STRATA_ROUTE_PRIOR_SCOPE.  No GPU, no model.
#include <cstdio>
#include <initializer_list>
#include <limits>

#include "strata/program/route_prior_auto.hpp"

using namespace strata::program;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

int main() {
    // the switch: only "1" is on; unset, empty and "0" are off and known; any other word is off and not known (the caller warns)
    {
        bool known = false;
        CHECK(!parse_prior_auto(nullptr, &known) && known);
        CHECK(!parse_prior_auto("", &known) && known);
        CHECK(!parse_prior_auto("0", &known) && known);
        CHECK(parse_prior_auto("1", &known) && known);
        CHECK(!parse_prior_auto("2", &known) && !known);
        CHECK(!parse_prior_auto("on", &known) && !known);
        CHECK(!parse_prior_auto("1 ", &known) && !known);
        CHECK(parse_prior_auto("1"));   // `known` is optional
    }
    // the minimum rows: a whole number in 0..kPriorAutoMaxMinT, anything else is 0 and not known
    {
        bool known = false;
        CHECK(parse_prior_auto_min_t(nullptr, &known) == 0 && known);
        CHECK(parse_prior_auto_min_t("", &known) == 0 && known);
        CHECK(parse_prior_auto_min_t("0", &known) == 0 && known);
        CHECK(parse_prior_auto_min_t("6", &known) == 6 && known);
        CHECK(parse_prior_auto_min_t("256", &known) == kPriorAutoMaxMinT && known);
        CHECK(parse_prior_auto_min_t("257", &known) == 0 && !known);
        CHECK(parse_prior_auto_min_t("-1", &known) == 0 && !known);
        CHECK(parse_prior_auto_min_t("6x", &known) == 0 && !known);
        CHECK(parse_prior_auto_min_t("x", &known) == 0 && !known);
        CHECK(parse_prior_auto_min_t("8") == 8);
    }
    // the window kind: a merged window is the prompt's whatever else is true; one row has no drafts; a lookup copy wins over a chain
    {
        CHECK(prior_window_kind(true, true, 3, 12) == PriorWindowKind::Prompt);
        CHECK(prior_window_kind(true, false, 0, 1) == PriorWindowKind::Prompt);
        CHECK(prior_window_kind(false, false, 0, 1) == PriorWindowKind::Plain);
        CHECK(prior_window_kind(false, true, 0, 1) == PriorWindowKind::Plain);   // no drafts: nothing was copied
        CHECK(prior_window_kind(false, true, 0, 12) == PriorWindowKind::Lookup);
        CHECK(prior_window_kind(false, true, 0, 2) == PriorWindowKind::Lookup);
        CHECK(prior_window_kind(false, false, 3, 7) == PriorWindowKind::Chain);
        CHECK(prior_window_kind(false, true, 3, 7) == PriorWindowKind::Lookup);   // (the loop never chains after a lookup)
        CHECK(prior_window_kind(false, false, 0, 4) == PriorWindowKind::Model);
    }
    const PriorWindowKind kinds[] = {PriorWindowKind::Prompt, PriorWindowKind::Plain, PriorWindowKind::Model,
                                     PriorWindowKind::Lookup, PriorWindowKind::Chain};
    const float nan = std::numeric_limits<float>::quiet_NaN();
    // auto off: every window takes the configured bonus (turbo as before); no bonus configured: none anywhere
    {
        const PriorAutoConfig off{};
        CHECK(!off.on && off.min_t == 0);
        for (PriorWindowKind k : kinds)
            for (int rows : {1, 2, 4, 12}) {
                CHECK(route_prior_auto_lambda(off, 0.5f, k, rows) == 0.5f);
                CHECK(route_prior_auto_lambda(off, 0.0f, k, rows) == 0.0f);
                CHECK(route_prior_auto_lambda(off, -1.0f, k, rows) == 0.0f);
                CHECK(route_prior_auto_lambda(off, nan, k, rows) == 0.0f);
            }
        PriorAutoConfig off_min;   // a minimum alone does nothing
        off_min.min_t = 8;
        CHECK(route_prior_auto_lambda(off_min, 0.5f, PriorWindowKind::Model, 4) == 0.5f);
        CHECK(route_prior_auto_lambda(off_min, 0.5f, PriorWindowKind::Lookup, 3) == 0.5f);
    }
    // auto on: the bonus only in lookup windows of 2 rows or more; exact routing in model, chained, plain and prompt windows
    {
        PriorAutoConfig on;
        on.on = true;
        CHECK(route_prior_auto_lambda(on, 0.5f, PriorWindowKind::Lookup, 12) == 0.5f);
        CHECK(route_prior_auto_lambda(on, 0.5f, PriorWindowKind::Lookup, 2) == 0.5f);
        CHECK(route_prior_auto_lambda(on, 1.5f, PriorWindowKind::Lookup, 5) == 1.5f);   // the configured value, not a fixed one
        CHECK(route_prior_auto_lambda(on, 0.5f, PriorWindowKind::Lookup, 1) == 0.0f);   // (a lookup kind never has 1 row; still closed)
        CHECK(route_prior_auto_lambda(on, 0.5f, PriorWindowKind::Model, 4) == 0.0f);
        CHECK(route_prior_auto_lambda(on, 0.5f, PriorWindowKind::Chain, 7) == 0.0f);
        CHECK(route_prior_auto_lambda(on, 0.5f, PriorWindowKind::Plain, 1) == 0.0f);
        CHECK(route_prior_auto_lambda(on, 0.5f, PriorWindowKind::Prompt, 12) == 0.0f);
        CHECK(route_prior_auto_lambda(on, 0.0f, PriorWindowKind::Lookup, 12) == 0.0f);
        CHECK(route_prior_auto_lambda(on, nan, PriorWindowKind::Lookup, 12) == 0.0f);
        // the minimum rows: below it a lookup window stays exact; at it and above it takes the bonus; 1 and 2 mean "any"
        PriorAutoConfig on6 = on;
        on6.min_t = 6;
        CHECK(route_prior_auto_lambda(on6, 0.5f, PriorWindowKind::Lookup, 5) == 0.0f);
        CHECK(route_prior_auto_lambda(on6, 0.5f, PriorWindowKind::Lookup, 6) == 0.5f);
        CHECK(route_prior_auto_lambda(on6, 0.5f, PriorWindowKind::Lookup, 12) == 0.5f);
        CHECK(route_prior_auto_lambda(on6, 0.5f, PriorWindowKind::Model, 12) == 0.0f);
        PriorAutoConfig on1 = on;
        on1.min_t = 1;
        CHECK(route_prior_auto_lambda(on1, 0.5f, PriorWindowKind::Lookup, 2) == 0.5f);
        CHECK(route_prior_auto_lambda(on1, 0.5f, PriorWindowKind::Lookup, 1) == 0.0f);
    }
    // the gate verdict: auto off returns the scope's verdict unchanged for every window (a run without the switch gates as before);
    // auto on can only close a window the scope left open, never open one it closed
    {
        const PriorAutoConfig off{};
        PriorAutoConfig on;
        on.on = true;
        for (PriorWindowKind k : kinds)
            for (int rows : {1, 2, 4, 12})
                for (bool scope_open : {false, true}) {
                    CHECK(prior_gate_open(scope_open, off, 0.5f, k, rows) == scope_open);
                    const bool want = scope_open && k == PriorWindowKind::Lookup && rows >= 2;
                    CHECK(prior_gate_open(scope_open, on, 0.5f, k, rows) == want);
                    if (!scope_open) CHECK(!prior_gate_open(scope_open, on, 0.5f, k, rows));
                }
        // the merged first window through the loop's own variables: always closed under auto
        CHECK(!prior_gate_open(true, on, 0.5f, prior_window_kind(true, true, 0, 12), 12));
        // a lookup window built from the loop's variables opens; the same rows drafted by the MTP do not
        CHECK(prior_gate_open(true, on, 0.5f, prior_window_kind(false, true, 0, 12), 12));
        CHECK(!prior_gate_open(true, on, 0.5f, prior_window_kind(false, false, 0, 4), 4));
        CHECK(!prior_gate_open(true, on, 0.5f, prior_window_kind(false, false, 2, 6), 6));
    }
    if (failed == 0) std::printf("route_prior_auto_test: ok\n");
    return failed == 0 ? 0 : 1;
}
