// src/spec/window_rule_test.cpp - which rule picks the verify window, and whose q lists judge the candidates (spec/window_rule.hpp).
//
// DFlash2 runs with the MTP layer loaded beside it as the fallback (use_mtp stays true).  Before this rule existed, the cost-aware
// window (STRATA_SPEC_COST=1, a GB10 default) read `chain_rows`, which only the MTP chain writes, so DFlash2 candidates were
// verified one row at a time; and the MTP's q lists (STRATA_SPEC_PROB=1) were attached to DFlash2's windows.  The tests:
//   1. the table of the fixed cases (DFlash2 + cost rule with and without --draft-conf / --spec-min-p);
//   2. the DEFAULT PATH IS UNCHANGED: for every input the old inline code handled correctly (MTP as the source, any flags;
//      DFlash2 with --draft-conf > 0 or with the cost rule off) a copy of the old code returns the same window;
//   3. the q rule.
#include "strata/spec/window_rule.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

using namespace strata::spec;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-78s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

// the inline code of f-0291-df2-mem 96f0307aa (the min-p rule on `draft_on`, the cost rule on use_mtp; CLI loop; the serve loop differs only in the start value and in first_window, which both callers
// overwrite afterwards)
int old_rule(const WindowRuleIn& in) {
    int T = in.start;
    if (in.use_mtp && in.spec_cost && !in.first_window) {
        T = std::clamp(in.chain_rows, 1, in.cap);
    } else if (in.draft_on && in.spec_min_p > 0.0) {
        T = 1;
        while (T < in.cap && in.dprob[T - 1] >= (float) in.spec_min_p) ++T;
    }
    if (in.df2_active && in.draft_conf > 0.0) {
        T = 1;
        double chain = 1.0;
        while (T < in.cap && (chain *= (double) in.dprob[T - 1]) >= in.draft_conf) ++T;
    }
    return T;
}
}  // namespace

int main() {
    std::printf("window_rule_test\n");
    const float p[8] = {0.9f, 0.8f, 0.3f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f};
    {   // 1. the table
        WindowRuleIn in;
        in.use_mtp = true; in.draft_on = true; in.cap = 6; in.start = 6; in.dprob = p; in.chain_rows = 1;
        in.df2_active = true; in.spec_cost = true;
        check(choose_window(in) == 6, "DFlash2 + cost rule, no min-p, no conf: the full cap (was 1)");
        in.spec_min_p = 0.5;
        check(choose_window(in) == 3, "DFlash2 + cost rule + --spec-min-p 0.5: stops at the weak draft (was 1)");
        in.spec_min_p = 0.0; in.draft_conf = 0.35;
        check(choose_window(in) == 3, "DFlash2 + cost rule + --draft-conf 0.35: the chain product rule (0.9*0.8*0.3=0.216)");
        in.draft_conf = 0.0; in.first_window = true;
        check(choose_window(in) == 6, "first window: left to the caller (it sets 1 / merge_T after)");
        in = WindowRuleIn{};
        in.use_mtp = true; in.draft_on = true; in.cap = 4; in.start = 4; in.dprob = p; in.spec_cost = true; in.chain_rows = 2;
        check(choose_window(in) == 2, "MTP + cost rule: the chain's rows");
        in.chain_rows = 9;
        check(choose_window(in) == 4, "MTP + cost rule: clamped to the cap");
        in.spec_cost = false; in.spec_min_p = 0.5;
        check(choose_window(in) == 3, "MTP + min-p 0.5: the first weak draft ends the window");
        in.spec_min_p = 0.0;
        check(choose_window(in) == 4, "MTP, no rule: the full cap");
        in.use_mtp = false; in.draft_on = false; in.start = 1;
        check(choose_window(in) == 1, "no MTP (serve): one row unless a lookup fires");
    }
    {   // 2. exhaustive over the small input space; random probabilities
        std::srand(7);
        long n = 0, same = 0, differ_df2_cost = 0, differ_other = 0;
        for (int use_mtp = 0; use_mtp < 2; ++use_mtp)
            for (int df2 = 0; df2 < 2; ++df2)
                for (int cost = 0; cost < 2; ++cost)
                    for (int first = 0; first < 2; ++first)
                        for (int cap = 1; cap <= 8; ++cap)
                            for (int rows = 1; rows <= 8; ++rows)
                                for (double mp : {0.0, 0.25, 0.5})
                                    for (double dc : {0.0, 0.15, 0.35}) {
                                        float pr[8];
                                        for (float& x : pr) x = (float) (std::rand() % 101) / 100.0f;
                                        WindowRuleIn in;
                                        in.use_mtp = use_mtp; in.draft_on = use_mtp || df2; in.df2_active = df2; in.spec_cost = cost; in.first_window = first;
                                        in.chain_rows = rows; in.cap = cap; in.start = use_mtp ? cap : 1; in.spec_min_p = mp;
                                        in.draft_conf = dc; in.dprob = pr;
                                        ++n;
                                        const int a = old_rule(in), b = choose_window(in);
                                        if (a == b) { ++same; continue; }
                                        // a difference is allowed only where the old rule used the MTP chain's rows for DFlash2's candidates
                                        if (use_mtp && df2 && cost && !first && dc == 0.0) ++differ_df2_cost; else ++differ_other;
                                    }
        std::printf("  %ld inputs: %ld identical, %ld differ (DFlash2 + cost rule, no --draft-conf), %ld differ elsewhere\n", n, same, differ_df2_cost, differ_other);
        check(differ_other == 0, "default path unchanged: no input outside the buggy case changes its window");
        check(differ_df2_cost > 0, "the buggy case does change (the test sees it)");
    }
    {   // 3. the q rule
        check(mtp_q_applies(true, false, true, false, false, 4), "MTP drew the window, sampled, probabilistic: q attached");
        check(!mtp_q_applies(true, true, true, false, false, 4), "DFlash2 drew the window: no MTP q (exact-match path)");
        check(!mtp_q_applies(true, false, true, true, false, 4), "a lookup window: point masses, no q");
        check(!mtp_q_applies(true, false, true, false, true, 4), "first window: no q");
        check(!mtp_q_applies(true, false, true, false, false, 1), "one row: no q");
        check(!mtp_q_applies(true, false, false, false, false, 4), "greedy / mode off: no q");
        check(!mtp_q_applies(false, false, true, false, false, 4), "no MTP: no q");
    }
    {   // 4. the calibrated DFlash2 gate (STRATA_DF2_GATE=1)
        AdaptiveGate g;
        float sure[8] = {0.95f, 0.95f, 0.95f, 0.95f, 0.95f, 0.95f, 0.95f, 0.95f}, weak[8] = {0.2f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f};
        check(g.choose(sure, 6) == 6, "gate, no data, confident drafts: the whole window");
        check(g.choose(weak, 6) == 1, "gate, no data, a hopeless first draft: one row");
        for (int i = 0; i < 40; ++i) g.observe(sure, 6, 1);   // drafts that keep failing at position 2 narrow the window for that bucket
        check(g.choose(sure, 6) <= 2, "gate learns that position 2 fails: the window narrows to 2 rows");
        AdaptiveGate g2;
        for (int i = 0; i < 40; ++i) g2.observe(sure, 6, 5);
        check(g2.choose(sure, 6) == 6, "gate, drafts always hold: the whole window");
        AdaptiveGate g3, g4;
        float mid[8] = {0.6f, 0.6f, 0.6f, 0.6f, 0.6f, 0.6f, 0.6f, 0.6f};
        g3.ms_per_tok = 4.0; g4.ms_per_tok = 14.0;
        check(g3.choose(mid, 6) <= g4.choose(mid, 6), "gate: a dearer token (14 ms) never gets a narrower window than a cheap one (4 ms)");
    }
    {   // 5. LEADS.md section 3 entropy stop rules (STRATA_SPEC_ENTROPY): parsing, each rule's cut, the online updates
        EntropyStop e;
        check(e.parse(nullptr) && !e.on(), "entropy: unset is off");
        check(e.parse("") && !e.on(), "entropy: empty is off");
        check(!e.parse("bogus") && !e.on(), "entropy: an unknown rule is refused and off");
        check(!e.parse("svip,h") && !e.on(), "entropy: a key without a value is refused");
        check(!e.parse("svip,zz=1") && !e.on(), "entropy: an unknown key is refused");
        check(e.parse("svip") && e.rule == EntropyStop::SVIP && e.h == 0.3, "entropy: svip, paper default h 0.3");
        check(e.keep(0.08) && !e.keep(0.1), "svip: sqrt(H) 0.28 keeps, sqrt(0.1) 0.316 > 0.3 stops");
        check(e.keep(std::nan("")), "svip: a NaN entropy (no softmax) never stops");
        check(e.parse("svip,h=0.5") && e.h == 0.5 && e.keep(0.24), "svip,h=0.5: sqrt(0.24) < 0.5 keeps");
        check(e.parse("adaedl") && e.rule == EntropyStop::AdaEDL && e.gamma == 0.2 && e.lambda == 0.4,
              "entropy: adaedl, gamma 0.2, lambda 0.4");
        // 1 - sqrt(0.2 H) >= 0.4  <=>  H <= 1.8
        check(e.keep(1.79) && !e.keep(1.81), "adaedl: the cut is at H = (1 - lambda)^2 / gamma = 1.8");
        const double l0 = e.lambda;
        e.observe(4, 1, nullptr);   // AR 0.25 < alpha 0.9: lambda goes up
        check(e.lambda > l0 && std::abs(e.lambda - (0.9 * l0 + 0.1 * (l0 + 0.01))) < 1e-12, "adaedl: low acceptance raises lambda (b2 0.9)");
        EntropyStop f;
        f.parse("adaedl,alpha=0.1");
        f.observe(4, 3, nullptr);   // AR 0.75 >= 0.1, a != n: lambda goes down
        check(f.lambda < 0.4, "adaedl: high acceptance with a rejection lowers lambda");
        const double l1 = f.lambda;
        f.observe(4, 4, nullptr);   // every draft held: lambda' = lambda (no move)
        check(std::abs(f.lambda - l1) < 1e-12, "adaedl: a full acceptance leaves lambda");
        EntropyStop d;
        check(d.parse("adasd") && d.rule == EntropyStop::AdaSD, "entropy: adasd");
        check(d.keep(100.0), "adasd: no stop before the first rejection");
        const float ent[4] = {0.1f, 0.5f, 0.9f, 2.0f};
        d.observe(4, 1, ent);   // draft 1 (H 0.5) was the first rejected
        d.observe(4, 2, ent);   // draft 2 (H 0.9)
        check(d.rej_n == 2 && std::abs(d.rej_sum - 1.4) < 1e-6, "adasd: the first rejected draft's entropy of each round is kept");
        check(d.keep(0.69) && !d.keep(0.71), "adasd: the cut is the mean rejected entropy 0.7");
        d.observe(4, 4, ent);
        check(d.rej_n == 2, "adasd: a fully accepted round adds nothing");
    }
    {   // 6. Medusa typical acceptance (STRATA_TYPICAL_THINK) and the host references the GPU parity test uses
        TypicalAccept t;
        check(parse_typical(nullptr, t) && !t.on(), "typical: unset is off");
        check(parse_typical("0.09", t) && t.on() && t.eps == 0.09f && t.alpha == 0.3f, "typical: eps 0.09, alpha 0.3 (Medusa's)");
        check(parse_typical("0.09,0.5", t) && t.alpha == 0.5f, "typical: eps,alpha");
        check(!parse_typical("1.5", t) && !t.on(), "typical: eps >= 1 refused");
        check(!parse_typical("0.1,x", t) && !t.on(), "typical: a bad alpha refused");
        check(std::abs(TypicalAccept::threshold(0.09f, 0.3f, 0.0f) - 0.09f) < 1e-7f, "typical: H 0 -> eps bounds the threshold");
        check(std::abs(TypicalAccept::threshold(0.09f, 0.3f, 3.0f) - 0.3f * std::exp(-3.0f)) < 1e-7f, "typical: high H -> alpha e^-H");
        const float flat[4] = {0.0f, 0.0f, 0.0f, 0.0f};   // p 0.25 each, H ln 4: threshold min(0.09, 0.3/4) = 0.075
        double p = 0, h = 0;
        check(typical_row_ref(flat, 4, 2, 1.0f, 0.09f, 0.3f, &p, &h) && std::abs(p - 0.25) < 1e-12 && std::abs(h - std::log(4.0)) < 1e-12,
              "typical: a flat row keeps any candidate (p 0.25 > 0.075)");
        const float peaked[3] = {10.0f, 0.0f, 0.0f};
        check(!typical_row_ref(peaked, 3, 1, 1.0f, 0.09f, 0.3f) && typical_row_ref(peaked, 3, 0, 1.0f, 0.09f, 0.3f),
              "typical: a peaked row keeps only its top token");
        check(!typical_row_ref(peaked, 3, -1, 1.0f, 0.09f, 0.3f) && !typical_row_ref(peaked, 3, 3, 1.0f, 0.09f, 0.3f),
              "typical: no candidate / out of range: not kept");
        double t2 = 0;
        const double he = row_entropy_ref(flat, 4, 0, &t2);
        check(std::abs(he - std::log(4.0)) < 1e-12 && std::abs(t2 - 0.25) < 1e-12, "row_entropy_ref: flat row, ln 4 and runner-up 0.25");
        const float ninf[3] = {1.0f, -INFINITY, -INFINITY};
        check(row_entropy_ref(ninf, 3, 0, &t2) == 0.0 && t2 == 0.0, "row_entropy_ref: -inf entries add nothing (no NaN)");
    }
    std::printf("%s\n", g_fail == 0 ? "window_rule_test: ok" : "window_rule_test: FAILED");
    return g_fail == 0 ? 0 : 1;
}
