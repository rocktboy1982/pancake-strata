// MOD_PLAN.md section 6 item 36: STRATA_KNOB_TRIAL's decision rule (include/strata/program/knob_trial.hpp).
#include "strata/program/knob_trial.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>

using namespace strata::program;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

// feed a trial: `rate_of(arm, window_index)` gives tokens/s of each window (8 tokens); `hit` a fixed hit rate
static KnobTrial run(const std::function<double(TrialArm, int)>& rate_of, double hit = 0.9, int max_windows = 40) {
    KnobTrial t;
    for (int i = 0; i < max_windows && t.verdict() == TrialVerdict::Running; ++i) {
        const TrialArm a = t.next_arm();
        t.add(a, 8, 8.0 / rate_of(a, i), hit);
    }
    return t;
}

int main() {
    // flat machine, candidate 20% faster: settles in 2 windows, 3 rounds, WIN with ratio 1.2
    KnobTrial w = run([](TrialArm a, int) { return a == TrialArm::Candidate ? 120.0 : 100.0; });
    check(w.verdict() == TrialVerdict::Win, "a flat 20% win is a win");
    check(w.ratios().size() == 3 && std::fabs(w.median_ratio() - 1.2) < 1e-9, "three rounds of ratio 1.2");
    check(w.windows() == 2 + 6, "2 settle windows (the last is I0) + C I C I C I = 8 windows");
    check(w.settle_windows() == 2, "settled after two incumbent windows");

    // a 5% win is under the 10% margin: LOSE (no adoption), but all rounds were run
    KnobTrial s = run([](TrialArm a, int) { return a == TrialArm::Candidate ? 105.0 : 100.0; });
    check(s.verdict() == TrialVerdict::Lose && s.ratios().size() == 3, "5% is under the margin: lose after the 3 rounds");
    check(std::string(s.reason()) == "median ratio below the margin", "reason: median below the margin");

    // exactly the margin wins
    KnobTrial e = run([](TrialArm a, int) { return a == TrialArm::Candidate ? 110.0 : 100.0; });
    check(e.verdict() == TrialVerdict::Win, "a ratio of exactly 1.10 wins");

    // drift cancels: the machine slows 1% per window, the candidate is genuinely 20% faster -> still ~1.2 (bracket mean)
    KnobTrial d = run([](TrialArm a, int i) { return (a == TrialArm::Candidate ? 1.2 : 1.0) * 100.0 * (1.0 - 0.01 * i); });
    check(d.verdict() == TrialVerdict::Win && std::fabs(d.median_ratio() - 1.2) < 0.01, "linear drift cancels in the bracket");
    // and a candidate that is merely equal is not made a winner by a drifting machine
    KnobTrial dn = run([](TrialArm, int i) { return 100.0 * (1.0 - 0.02 * i); });
    check(dn.verdict() == TrialVerdict::Lose && std::fabs(dn.median_ratio() - 1.0) < 0.02, "equal knob on a drifting machine: ratio 1, lose");

    // two of three rounds lost: dropped after the second lost round (round 2), before round 3
    int n_c = 0;
    KnobTrial l2 = run([&](TrialArm a, int) { if (a != TrialArm::Candidate) return 100.0; ++n_c; return n_c == 1 ? 130.0 : 95.0; });
    check(l2.verdict() == TrialVerdict::Lose, "round 1 won, rounds 2 and 3 lost: lose");
    int n_c2 = 0;
    KnobTrial l3 = run([&](TrialArm a, int) { if (a != TrialArm::Candidate) return 100.0; ++n_c2; return 95.0; });
    check(l3.verdict() == TrialVerdict::Lose && l3.ratios().size() == 2 && l3.reason() == "more than half of the rounds lost",
          "two lost rounds of three stop the trial early (2 rounds run)");

    // one catastrophic candidate window (20x slower): cut after one window
    KnobTrial c = run([](TrialArm a, int) { return a == TrialArm::Candidate ? 5.0 : 100.0; });
    check(c.verdict() == TrialVerdict::Lose && c.ratios().empty() && c.windows() == 3, "a 20x slower candidate is cut at its first window");
    check(std::string(c.reason()) == "a candidate window ran apart from the incumbent band", "reason: ran apart");

    // a noisy incumbent band is not 'apart': incumbents 80..120, a candidate window at 70 is inside the slack (below min by <= spread)
    KnobTrial nb;
    nb.add(TrialArm::Incumbent, 8, 8.0 / 80.0, 0.9);
    nb.add(TrialArm::Incumbent, 8, 8.0 / 120.0, 0.9);
    nb.add(TrialArm::Candidate, 8, 8.0 / 70.0, 0.9);   // 70 < 0.9 x 80 = 72 but 70 > 80 - 40 = 40: not apart
    check(nb.verdict() == TrialVerdict::Running, "a window inside the incumbent's own spread is not cut");

    // the settle wait: hit rates 0.50 then 0.80 do not agree, so the trial keeps measuring incumbents
    KnobTrial st;
    st.add(TrialArm::Incumbent, 8, 0.08, 0.50);
    st.add(TrialArm::Incumbent, 8, 0.08, 0.80);
    check(st.next_arm() == TrialArm::Incumbent && st.settle_windows() == 2, "hit rates 0.30 apart: still settling");
    st.add(TrialArm::Incumbent, 8, 0.08, 0.81);
    check(st.next_arm() == TrialArm::Candidate, "0.80 then 0.81 agree within 0.02: the candidate runs next");
    // unknown hit rates settle after two windows
    KnobTrial su;
    su.add(TrialArm::Incumbent, 8, 0.08);
    su.add(TrialArm::Incumbent, 8, 0.08);
    check(su.next_arm() == TrialArm::Candidate, "unknown hit rates: settled after two incumbent windows");

    // the answer ends inside a window: undecided, nothing adopted
    KnobTrial u;
    u.add(TrialArm::Incumbent, 8, 0.08, 0.9);
    u.add(TrialArm::Incumbent, 3, 0.03, 0.9);
    check(u.verdict() == TrialVerdict::Undecided, "a short window ends the trial undecided");
    u.add(TrialArm::Incumbent, 8, 0.08, 0.9);
    check(u.verdict() == TrialVerdict::Undecided, "a finished trial ignores later windows");

    // a window that ran in the wrong arm (caller ignored next_arm) is not counted
    KnobTrial wa;
    wa.add(TrialArm::Candidate, 8, 0.08, 0.9);
    check(wa.next_arm() == TrialArm::Incumbent && wa.settle_windows() == 0, "a candidate window before settling is not counted");

    // log line
    const std::string ln = w.line("adapt-swaps", "32", "64");
    check(ln == "strata knob trial: adapt-swaps 32 -> 64: WIN ratio 1.200 (rounds 1.200 1.200 1.200); 8 windows of 8 tokens, 2 to settle", "WIN log line");
    check(c.line("k", "a", "b") == "strata knob trial: k a -> b: LOSE; 3 windows of 8 tokens, 2 to settle; a candidate window ran apart from the incumbent band",
          "LOSE log line");
    std::printf("knob_trial_test: %d checks passed\n", checks);
    return 0;
}
