// LEADS.md Part P lead L5: STRATA_FREE_SETTLE_MS's bounded wait (include/strata/program/free_settle.hpp).
#include "strata/program/free_settle.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::program;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

int main() {
    const uint64_t MiB = (uint64_t) 1 << 20;
    uint64_t now_us = 0;
    uint64_t sleeps = 0;
    auto sleep_fn = [&](uint32_t us) { now_us += us; ++sleeps; };

    {   // off: no query, no sleep, settled
        int q = 0;
        auto r = free_settle_wait(1000 * MiB, 500 * MiB, 0, 0, 100, [&] { ++q; return (uint64_t) 0; }, sleep_fn);
        check(r.settled && q == 0 && r.waited_us == 0 && r.polls == 0, "cap 0: off");
    }
    {   // already settled: one poll, no sleep
        auto r = free_settle_wait(1000 * MiB, 500 * MiB, 0, 250, 100, [] { return (uint64_t) 1500 * MiB; }, sleep_fn);
        check(r.settled && r.polls == 1 && r.waited_us == 0, "settled at once");
    }
    {   // the figure catches up after 3 steps (the lag the driver shows)
        int q = 0;
        auto r = free_settle_wait(1000 * MiB, 500 * MiB, 0, 250, 100,
                                  [&] { return (uint64_t) (q++ < 3 ? 1000 : 1500) * MiB; }, sleep_fn);
        check(r.settled && r.polls == 4 && r.waited_us == 300, "settles after the lag");
    }
    {   // never catches up: stops at the cap, not settled, total sleep == cap
        now_us = 0;
        auto r = free_settle_wait(1000 * MiB, 500 * MiB, 0, 2, 300, [] { return (uint64_t) 1000 * MiB; }, sleep_fn);
        check(!r.settled && r.waited_us == 2000 && now_us == 2000, "cap reached: waited exactly the cap");
        check(r.last_free == 1000 * MiB, "last figure reported");
    }
    {   // slack: a figure short by less than the slack counts as settled
        auto r = free_settle_wait(1000 * MiB, 500 * MiB, 64 * MiB, 250, 100, [] { return (uint64_t) 1450 * MiB; }, sleep_fn);
        check(r.settled && r.polls == 1, "within the slack");
        auto r2 = free_settle_wait(1000 * MiB, 500 * MiB, 64 * MiB, 1, 100, [] { return (uint64_t) 1400 * MiB; }, sleep_fn);
        check(!r2.settled, "short by more than the slack");
    }
    {   // a step of 0 does not loop forever; a slack larger than the target gives target 0
        auto r = free_settle_wait(0, 10, 100, 1, 0, [] { return (uint64_t) 0; }, sleep_fn);
        check(r.settled, "target 0 is reached by any figure");
        auto r2 = free_settle_wait(1000, 10, 0, 1, 0, [] { return (uint64_t) 0; }, sleep_fn);
        check(!r2.settled && r2.waited_us == 1000, "step 0 is treated as 1 us and still ends at the cap");
    }
    check(free_settle_ms_from_text(nullptr) == 0, "unset");
    check(free_settle_ms_from_text("") == 0, "empty");
    check(free_settle_ms_from_text("250") == 250, "250");
    check(free_settle_ms_from_text("-5") == 0, "negative");
    check(free_settle_ms_from_text("12x") == 0, "junk");
    check(free_settle_ms_from_text("99999") == 5000, "capped");
    std::printf("free_settle_test: %d checks passed\n", checks);
    return 0;
}
