// MOD_PLAN.md section 6 (the BigMoeOnEdge switches, LEADS.md Part I): STRATA_LOOKAHEAD_CTRL's counting rule (include/strata/program/lookahead_ctrl.hpp).
#include "strata/program/lookahead_ctrl.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

using namespace strata::program;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

int main() {
    // overlap: counted over the TRUE slots, invalid truth entries are not slots
    const int32_t pred[4] = {5, 9, 2, 7};
    const int32_t truth[4] = {9, 1, 7, -1};
    int v = 0;
    check(ctrl_overlap(pred, 4, truth, 4, 16, &v) == 2 && v == 3, "2 of 3 valid true slots found, -1 is no slot");
    check(ctrl_overlap(pred, 0, truth, 4, 16, &v) == 0 && v == 3, "empty prediction: no hit, slots still counted");
    const int32_t big[2] = {99, 3};
    check(ctrl_overlap(pred, 4, big, 2, 16, &v) == 0 && v == 1, "an id past n_expert is not a slot");

    // top-k: best first, ties to the lower id, k clamped
    const float lg[6] = {0.5f, 2.0f, 2.0f, -1.0f, 2.0f, 0.5f};
    int32_t out[6];
    check(ctrl_topk(lg, 6, 3, out) == 3 && out[0] == 1 && out[1] == 2 && out[2] == 4, "three equal 2.0: ids 1 2 4 in order");
    check(ctrl_topk(lg, 6, 4, out) == 4 && out[3] == 0, "fourth is the lower id of the two 0.5");
    check(ctrl_topk(lg, 6, 9, out) == 6 && out[5] == 3, "k clamped to n, the worst last");
    const float asc[5] = {1, 2, 3, 4, 5};
    check(ctrl_topk(asc, 5, 2, out) == 2 && out[0] == 4 && out[1] == 3, "ascending logits");

    // control on the same logits as the routing reads 100%
    LookaheadCtrlTally t;
    t.init(3);
    int32_t routed[3];
    ctrl_topk(lg, 6, 3, routed);
    int32_t ctl[3];
    const int nc = ctrl_topk(lg, 6, 3, ctl);
    t.add(1, CtrlKind::Control, ctl, nc, routed, 3, 6);
    t.add(2, CtrlKind::Control, ctl, nc, routed, 3, 6);
    check(t.rate(CtrlKind::Control, 0, 3) == 1.0, "control = 100% when it is the same function of the same input");

    // stale: layer 1 one token 1 of 3, layer 2 one token 3 of 3
    const int32_t st1[3] = {1, 8, 9}, st2[3] = {4, 1, 2};
    t.add(1, CtrlKind::StaleGpu, st1, 3, routed, 3, 6);
    t.add(2, CtrlKind::StaleGpu, st2, 3, routed, 3, 6);
    check(t.rate(CtrlKind::StaleGpu, 1) == 1.0 / 3.0 && t.rate(CtrlKind::StaleGpu, 2) == 1.0, "per layer rates");
    check(t.rate(CtrlKind::StaleGpu, 0, 3) == 4.0 / 6.0, "overall rate is hits over slots, not a mean of layer rates");
    check(t.rate(CtrlKind::StaleCpu, 0, 3) < 0.0 && t.rate(CtrlKind::Prev, 1) < 0.0, "unscored predictor reads n/a");
    t.add(7, CtrlKind::StaleGpu, st1, 3, routed, 3, 6);   // out of range layer ignored
    t.add(-1, CtrlKind::StaleGpu, st1, 3, routed, 3, 6);
    check(t.rate(CtrlKind::StaleGpu, 0, 3) == 4.0 / 6.0, "out-of-range layers are ignored");

    // a larger predicted set raises the overlap but never past 100%
    const int32_t wide[6] = {0, 1, 2, 3, 4, 5};
    LookaheadCtrlTally w;
    w.init(1);
    w.add(0, CtrlKind::StaleGpu, wide, 6, routed, 3, 6);
    check(w.rate(CtrlKind::StaleGpu, 0) == 1.0, "a prediction of every expert covers every slot, rate 1");

    // text: the format tools/pancake/lookahead_ctrl.py parses
    t.windows = 100;
    const std::string s = t.summary_line();
    check(s == "strata lookahead ctrl: windows=100 slots=6 stale_gpu=0.6667 stale_cpu=n/a control=1.0000 prev=n/a", "summary line format");
    const std::string l1 = t.layer_line(1);
    check(l1 == "strata lookahead ctrl layer 1: slots=3 stale_gpu=0.3333 stale_cpu=n/a control=1.0000 prev=n/a", "layer line format");
    check(t.layer_line(9).find("slots=0") != std::string::npos, "layer line out of range does not crash");

    LookaheadCtrlTally z;
    z.init(0);
    check(z.rate(CtrlKind::Control, 0, 4) < 0.0 && z.summary_line().find("stale_gpu=n/a") != std::string::npos, "empty tally");

    std::printf("lookahead_ctrl_test: %d checks passed\n", checks);
    return 0;
}
