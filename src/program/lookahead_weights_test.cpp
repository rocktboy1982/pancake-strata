// MOD_PLAN.md section 6 item 29: the trained look-ahead predictor's file layout and pick rule
// (include/strata/program/lookahead_weights.hpp).
#include "strata/program/lookahead_weights.hpp"

#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace strata::program;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

int main() {
    // layout constants shared with tools/pancake/fit_lookahead.py (its test pins the same numbers)
    check(kLkwMagic == 0x31574b4cu && kPredMagic == 0x32445250u, "magics");
    check(lkw_file_bytes(2, 4, 8) == 16 + 2 * (4 * 8 * 2 + 4 * 4), "LKW1 size: 16 + layers x (bf16 W + f32 bias)");
    check(pred_record_bytes(3, 2, 10, 8) == 28 + 3 * 8 * 2 + 3 * 2 * 2 + 3 * 2 + 2, "RPD2 record size");
    const uint32_t ok_head[4] = {kLkwMagic, 2, 4, 8};
    check(lkw_header_ok(ok_head, 2, 4, 8), "header ok");
    check(!lkw_header_ok(ok_head, 3, 4, 8) && !lkw_header_ok(ok_head, 2, 5, 8) && !lkw_header_ok(ok_head, 2, 4, 9), "a predictor of another shape is refused");
    const uint32_t bad_magic[4] = {0, 2, 4, 8};
    check(!lkw_header_ok(bad_magic, 2, 4, 8), "wrong magic refused");
    uint32_t ph[7];
    pred_head(ph, 5, 4, 10, 0x100000002ull, 512);
    check(ph[0] == kPredMagic && ph[1] == 5 && ph[2] == 4 && ph[3] == 10 && ph[4] == 2 && ph[5] == 1 && ph[6] == 512, "record header, window split lo/hi");

    // bf16: 1.0 exactly, round to nearest even, NaN stays NaN
    check(pred_bf16(1.0f) == 0x3f80 && pred_bf16(-2.0f) == 0xc000 && pred_bf16(0.0f) == 0, "bf16 exact values");
    check(pred_bf16(1.0f + 1.0f / 256.0f) == 0x3f80, "1 + 2^-8 is a tie, rounds to even (down)");
    check(pred_bf16(1.0f + 3.0f / 256.0f) == 0x3f82, "1 + 3 x 2^-8 is a tie, rounds to even (up)");
    check((pred_bf16(std::nanf("")) & 0x7f80) == 0x7f80 && (pred_bf16(std::nanf("")) & 0x7f) != 0, "NaN stays NaN");

    // pick: 6 experts, 2 tokens
    const float lg[12] = {0.0f, 3.0f, 1.0f, 0.0f, 2.0f, -1.0f,    // token 0: best 1, 4, 2
                          0.0f, 0.5f, 4.0f, 0.0f, 0.0f, 2.5f};   // token 1: best 2, 5, 1
    std::vector<int32_t> want, ranked;
    std::vector<float> conf;
    check(lookahead_weights_pick(lg, 2, 6, nullptr, nullptr, 3, 0.0f, 6, want, conf, ranked) == 3, "budget 3 -> 3 wanted");
    // best over tokens: e2 = 4.0, e1 = 3.0, e5 = 2.5, e4 = 2.0 ...
    check(want[0] == 2 && want[1] == 1 && want[2] == 5, "union over tokens by best score");
    check(conf[0] == 1.0f && conf[1] == 2.0f / 3.0f && conf[2] == 1.0f / 3.0f, "confidence (budget - j) / budget");
    check(ranked.size() == 6 && ranked[3] == 4, "ranking carries past the budget for the statistics");
    // bias moves the ranking
    const float bias[6] = {0, 0, 0, 0, 5.0f, 0};
    lookahead_weights_pick(lg, 2, 6, bias, nullptr, 1, 0.0f, 6, want, conf, ranked);
    check(want.size() == 1 && want[0] == 4, "a bias of 5 on expert 4 puts it first");
    // resident experts never count
    const int32_t res[6] = {-1, 7, -1, -1, -1, -1};
    lookahead_weights_pick(lg, 2, 6, nullptr, res, 3, 0.0f, 6, want, conf, ranked);
    check(want[0] == 2 && want[1] == 5 && want[2] == 4 && ranked.size() == 5, "resident expert 1 is skipped and not ranked");
    // ties go to the lower id
    const float tie[6] = {1, 1, 1, 0, 0, 0};
    lookahead_weights_pick(tie, 1, 6, nullptr, nullptr, 2, 0.0f, 6, want, conf, ranked);
    check(want[0] == 0 && want[1] == 1, "ties to the lower id");
    // tau: probability floor ends the list
    const float peak[6] = {10.0f, 0, 0, 0, 0, 0};
    lookahead_weights_pick(peak, 1, 6, nullptr, nullptr, 4, 0.5f, 6, want, conf, ranked);
    check(want.size() == 1 && want[0] == 0, "tau 0.5: only the expert with probability above 0.5");
    lookahead_weights_pick(peak, 1, 6, nullptr, nullptr, 4, 0.0f, 6, want, conf, ranked);
    check(want.size() == 4, "no floor: the full budget");
    lookahead_weights_pick(tie, 1, 6, nullptr, nullptr, 4, 0.9f, 6, want, conf, ranked);
    check(want.empty(), "tau above every probability: nothing warmed");
    // degenerate input
    check(lookahead_weights_pick(lg, 0, 6, nullptr, nullptr, 3, 0.0f, 6, want, conf, ranked) == 0 &&
          lookahead_weights_pick(lg, 2, 6, nullptr, nullptr, 0, 0.0f, 6, want, conf, ranked) == 0, "no tokens / no budget: nothing");
    const float nanlg[6] = {NAN, NAN, NAN, NAN, NAN, 1.0f};
    check(lookahead_weights_pick(nanlg, 1, 6, nullptr, nullptr, 3, 0.0f, 6, want, conf, ranked) == 1 && want[0] == 5, "NaN scores never rank");
    std::printf("lookahead_weights_test: %d checks passed\n", checks);
    return 0;
}
