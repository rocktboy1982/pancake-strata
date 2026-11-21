// LEADS.md Part S row S8: STRATA_SWAP_BUDGET_RESIDENCY's schedule and budget function (include/strata/program/swap_budget.hpp),
// host only.  Validated: the schedule (fill 64 / hold 16 for 512 steps, the gates, the renewal), the resident-mass share, the
// parse, the budget as SwapBudget reports it, and that the default (switch unset) budget is the old one and never reaches the
// new state.  NOT validated: the wiring in src/program/generate.cpp (host_res / usage reads), the effect on link bytes or decode.
#include "strata/program/swap_budget.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::program;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

static void set_env(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v != nullptr ? v : "");
#else
    if (v != nullptr) setenv(k, v, 1); else unsetenv(k);
#endif
}

int main() {
    // parse
    check(!swap_budget_residency_parse(nullptr) && !swap_budget_residency_parse("") && !swap_budget_residency_parse("0") &&
              !swap_budget_residency_parse("2") && !swap_budget_residency_parse("11") && swap_budget_residency_parse("1"), "only 1 is on");
    check(swap_budget_gate_parse(nullptr) == 0.60 && swap_budget_gate_parse("") == 0.60 && swap_budget_gate_parse("0.8") == 0.8 &&
              swap_budget_gate_parse("1") == 1.0 && swap_budget_gate_parse("0") == 0.60 && swap_budget_gate_parse("1.5") == 0.60 &&
              swap_budget_gate_parse("abc") == 0.60 && swap_budget_gate_parse("-0.2") == 0.60, "gate parse: (0, 1], else 0.60");

    // the resident mass share
    {
        const float u[6] = {10, 0, 30, 60, -5, 0};
        const int32_t r[6] = {3, -1, -1, 7, 2, -1};   // resident: 0, 3 (4 has a negative count: ignored)
        const double s = resident_mass_share(u, r, 6);
        check(s > 0.699 && s < 0.701, "share = (10 + 60) / 100");
        const float z[3] = {0, 0, 0};
        const int32_t rz[3] = {0, 1, 2};
        check(resident_mass_share(z, rz, 3) == 0.0 && resident_mass_share(z, rz, 0) == 0.0, "nothing routed: 0");
        const int32_t all[3] = {0, 1, 2};
        const float v[3] = {1, 2, 3};
        check(resident_mass_share(v, all, 3) == 1.0, "all resident: 1");
    }

    // the schedule: fill 64, hold 16 for 512 steps
    {
        ResidencySchedule s(64, 16, 0.60, 4096, 512);
        check(s.budget() == 64 && !s.holding(), "starts in fill at 64");
        s.round(8, 0.90, 1000);   // share fine, histogram pass not yet (1,000 < 4,096)
        check(s.budget() == 64, "share alone does not open the hold: 1,000 selections");
        s.round(8, 0.50, 5000);   // 6,000 selections but 50% resident
        check(s.budget() == 64 && s.selections() == 6000, "the histogram pass alone does not open the hold: 50% resident");
        s.round(8, 0.60, 0);      // exactly the gate: opens
        check(s.budget() == 16 && s.holding() && s.holds() == 1, "share >= gate and >= 4,096 selections: hold at 16");
        for (int i = 0; i < 63; ++i) s.round(8, 0.95, 0);   // 504 steps
        check(s.budget() == 16, "held for 504 of 512 steps");
        s.round(8, 0.95, 0);      // 512: ends, still resident: renewed
        check(s.budget() == 16 && s.holds() == 2, "512 steps over and still resident: held again");
        for (int i = 0; i < 64; ++i) s.round(8, 0.95, 0);
        check(s.holds() == 3, "renewed again after 512 more steps");
        for (int i = 0; i < 63; ++i) s.round(8, 0.95, 0);
        s.round(8, 0.40, 0);      // the hold ends with the share fallen under the gate: fill
        check(s.budget() == 64 && !s.holding(), "hold over and the share fell under the gate: back to 64");
        s.round(8, 0.40, 0);
        check(s.budget() == 64, "stays in fill below the gate");
        s.round(8, 0.75, 0);      // selections are kept (>= 4,096 already): the share alone reopens it
        check(s.budget() == 16 && s.holds() == 4, "share back over the gate: hold again");
    }
    // a hold is not cut short by a share dip inside its 512 steps
    {
        ResidencySchedule s(64, 16, 0.60, 100, 512);
        s.round(1, 0.9, 200);
        check(s.budget() == 16, "held");
        s.round(100, 0.1, 0);
        check(s.budget() == 16, "a dip inside the hold changes nothing");
        // windows > 1 count as steps; an overshoot ends it
        s.round(1000, 0.1, 0);
        check(s.budget() == 64, "1,100 steps passed at 10%: fill");
    }
    // lo is clamped to hi, hi to >= 1
    {
        ResidencySchedule s(8, 16, 0.5, 1, 4);
        s.round(1, 1.0, 1);
        check(s.budget() == 8, "lo cannot exceed hi");
    }

    // SwapBudget: the default (switch unset) is the old budget and never reaches the residency state
    set_env("STRATA_SWAP_BUDGET_RESIDENCY", nullptr);
    set_env("STRATA_SWAP_BUDGET_RESIDENCY_GATE", nullptr);
    set_env("STRATA_ADAPT_SWAPS_AUTO", nullptr);
    {
        SwapBudget b = SwapBudget::from_env(32, 96, 400);
        check(!b.on() && !b.residency_on() && b.swaps() == 32, "default: off, the fixed --adapt-swaps");
        const float u[2] = {1, 1};
        const int32_t r[2] = {0, 0};
        b.observe_residency(u, r, 2);
        for (int i = 0; i < 2000; ++i) b.round(8);
        check(b.swaps() == 32 && !b.on() && b.residency().selections() == 0 && b.residency().holds() == 0 && !b.residency().holding(),
              "default: observe and round do nothing, the residency schedule is never touched");
    }
    // STRATA_ADAPT_SWAPS_AUTO alone: the hill climb as before (starts at the given budget)
    set_env("STRATA_ADAPT_SWAPS_AUTO", "1");
    {
        SwapBudget b = SwapBudget::from_env(32, 96, 400);
        check(b.on() && !b.residency_on() && b.swaps() == 32, "AUTO: online, from the start budget");
    }
    // both set: AUTO wins, the residency switch is ignored
    set_env("STRATA_SWAP_BUDGET_RESIDENCY", "1");
    {
        SwapBudget b = SwapBudget::from_env(32, 96, 400);
        check(b.on() && !b.residency_on() && b.swaps() == 32, "AUTO and RESIDENCY: AUTO wins");
    }
    set_env("STRATA_ADAPT_SWAPS_AUTO", nullptr);
    // the switch on
    {
        SwapBudget b = SwapBudget::from_env(32, 96, 400);   // one window routes at least 400 expert picks
        check(b.on() && b.residency_on() && b.swaps() == 64, "RESIDENCY: starts at 64 swaps a round");
        const float u[4] = {50, 50, 50, 50};
        const int32_t half[4] = {0, 1, -1, -1};    // 50% resident
        const int32_t most[4] = {0, 1, 2, -1};     // 75% resident
        b.observe_residency(u, half, 4);
        b.round(8);   // 3,200 selections, 50%: still filling
        check(b.swaps() == 64, "50% resident: 64");
        b.observe_residency(u, most, 4);
        b.round(8);   // 6,400 selections, 75%: hold
        check(b.swaps() == 16, "75% resident and 6,400 selections: 16");
        b.observe_residency(u, half, 4);
        for (int i = 0; i < 63; ++i) b.round(8);
        check(b.swaps() == 16, "the hold is 512 steps");
        b.round(8);
        check(b.swaps() == 64, "512 steps over at 50%: 64 again");
    }
    // a higher gate
    set_env("STRATA_SWAP_BUDGET_RESIDENCY_GATE", "0.9");
    {
        SwapBudget b = SwapBudget::from_env(32, 96, 4096);
        const float u[4] = {50, 50, 50, 50};
        const int32_t most[4] = {0, 1, 2, -1};
        b.observe_residency(u, most, 4);
        b.round(1);
        check(b.swaps() == 64, "gate 0.9: 75% does not open the hold");
        const int32_t all[4] = {0, 1, 2, 3};
        b.observe_residency(u, all, 4);
        b.round(1);
        check(b.swaps() == 16, "gate 0.9: 100% does");
    }
    set_env("STRATA_SWAP_BUDGET_RESIDENCY", nullptr);
    set_env("STRATA_SWAP_BUDGET_RESIDENCY_GATE", nullptr);
    std::printf("swap_budget_residency_test: %d checks passed\n", checks);
    return 0;
}
