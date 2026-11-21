// LEADS.md Part S row S7: STRATA_PCOPY_REST_USAGE's rule (include/strata/program/pcopy_rest.hpp), host only.
// Validated: the parse, the 30% / 2,048 / 8,192 rule, the empty block after a rest, and that the default (off) rule never
// reaches its state.  NOT validated: the look-ahead wiring in src/core/expert_source.cpp (a build with the engine only).
#include "strata/program/pcopy_rest.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::program;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

int main() {
    check(!pcopy_rest_parse(nullptr) && !pcopy_rest_parse("") && !pcopy_rest_parse("0") && !pcopy_rest_parse("2") &&
              !pcopy_rest_parse("11") && !pcopy_rest_parse("true") && pcopy_rest_parse("1"), "only the text 1 is on");

    // the default (off) rule never moves, whatever it is fed
    {
        PcopyRest off;
        bool all_pass = true, any_rest = false;
        for (int i = 0; i < 50000; ++i) {
            all_pass = all_pass && off.pass();
            any_rest = off.record(10, 0) || any_rest;
        }
        check(all_pass && !any_rest && !off.on() && !off.resting() && off.touched() == 0 && off.rests() == 0 && off.skipped() == 0,
              "off: pass() always true, record() does nothing, no state touched");
        PcopyRest off2(false);
        check(off2.pass() && !off2.record(5, 0) && off2.touched() == 0, "PcopyRest(false) is the same");
    }

    // constants of the rule as the lead states them
    check(kPcopyRestWindow == 2048 && kPcopyRestMinPct == 30 && kPcopyRestPasses == 8192, "2,048 / 30% / 8,192");

    // a used share of exactly 30% is kept (the rule rests on UNDER 30%)
    {
        PcopyRest r(true);
        bool started = false;
        for (int i = 0; i < 2048 / 10; ++i) started = r.record(10, 3) || started;   // 2,040 predicted: not yet judged
        check(!started && !r.resting(), "under the block size nothing is judged");
        started = r.record(10, 3);   // 2,050 predicted, 615 used = 30.0%
        check(!started && !r.resting() && r.last_predicted() == 2050 && r.last_used() == 615, "exactly 30% used keeps the copy on");
    }
    // 29.9% rests, 8,192 passes, then resumes
    {
        PcopyRest r(true);
        // 2,040 predicted with 590 used, then 10 more with none: 2,050 predicted, 590 used = 28.8%
        bool started = false;
        for (int i = 0; i < 204; ++i) started = r.record(10, i < 59 ? 10 : 0) || started;   // 2,040 predicted, 590 used = 28.9%
        check(!started, "2,040 predicted: block not complete");
        started = r.record(10, 0);   // 2,050 predicted, 590 used = 28.8%: judged
        check(started && r.resting() && r.rests() == 1 && r.last_predicted() == 2050 && r.last_used() == 590, "under 30% used rests the copy");
        int refused = 0;
        for (int i = 0; i < kPcopyRestPasses; ++i) if (!r.pass()) ++refused;
        check(refused == kPcopyRestPasses && r.skipped() == kPcopyRestPasses && !r.resting(), "8,192 passes are refused, then it resumes");
        check(r.pass(), "the pass after the rest runs");
        // feedback during a rest is ignored, the block after it starts empty
        PcopyRest q(true, 100, 30, 5);
        for (int i = 0; i < 10; ++i) q.record(10, 0);   // 100 predicted, 0 used: rest
        check(q.resting() && q.rests() == 1, "small rule: rested");
        check(!q.record(1000, 0), "a prediction recorded while resting carries no verdict");
        for (int i = 0; i < 5; ++i) check(!q.pass(), "refused while resting");
        check(q.pass() && !q.resting(), "resumed after 5 passes");
        for (int i = 0; i < 9; ++i) q.record(10, 10);   // 90 predicted, all used: the old block (empty after the rest) is not mixed in
        check(!q.resting(), "a fresh block: 90 predicted is not yet judged");
        q.record(10, 10);
        check(!q.resting() && q.rests() == 1, "100 predicted, all used: stays on");
        // a rest after good blocks, then bad: rests again
        for (int i = 0; i < 10; ++i) q.record(10, 1);
        check(q.resting() && q.rests() == 2, "10% used rests again");
    }
    // used is clamped to predicted; a prediction of nothing is ignored
    {
        PcopyRest r(true, 20, 30, 3);
        check(!r.record(0, 5) && !r.record(-3, 1), "nothing predicted: no vote");
        check(!r.record(10, 99), "used clamped to predicted");
        check(!r.record(10, 10) && !r.resting(), "all used: on");
        // 20 predicted at 100%; a block of 20 with 5 used (25%) rests
        check(r.record(10, 0) == false, "block restarts at the verdict: 10 predicted");
        check(r.record(10, 5) && r.resting(), "10 + 10 predicted, 5 used (25%): rests");
    }
    std::printf("pcopy_rest_test: %d checks passed\n", checks);
    return 0;
}
