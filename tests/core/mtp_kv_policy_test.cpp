// STRATA_MTP_KV_FP16 (LEADS.md Part Q5): the drafter's K/V format rule (include/strata/core/mtp_kv_policy.hpp).  No GPU.
// The default must be the rule src/core/mtp.cpp had before the switch, for every main K/V setting.
#include <cstdio>

#include "strata/core/mtp_kv_policy.hpp"

using namespace strata::core;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

// the pre-switch rule, copied from the old mtp.cpp: hybrid off; under K8V4 INT8 on; else the main setting
static bool old_int8(bool main_int8, bool main_hybrid) { return main_hybrid ? true : main_int8; }

int main() {
    CHECK(!mtp_kv_fp16_from(nullptr));
    CHECK(!mtp_kv_fp16_from(""));
    CHECK(!mtp_kv_fp16_from("0"));
    CHECK(mtp_kv_fp16_from("1"));

    // default (switch off): identical to the old rule for all 8 combinations, hybrid never on
    for (int m = 0; m < 8; ++m) {
        const bool i8 = m & 1, hy = m & 2, q4 = m & 4;
        const MtpKvPlan p = mtp_kv_plan(false, i8, hy, q4);
        CHECK(p.int8 == old_int8(i8, hy));
        CHECK(!p.hybrid);
    }
    // on: FP16 beside INT8 and K8V4 mains
    CHECK(!mtp_kv_plan(true, true, false, false).int8);
    CHECK(!mtp_kv_plan(true, false, true, false).int8);
    CHECK(!mtp_kv_plan(true, true, true, false).int8);
    CHECK(!mtp_kv_plan(true, true, false, false).hybrid);
    // on, but nothing to change: a plain FP16 main stays FP16; Q4_0 main is left as the default rule has it
    CHECK(!mtp_kv_plan(true, false, false, false).int8);
    for (int m = 0; m < 4; ++m) {
        const bool i8 = m & 1, hy = m & 2;
        CHECK(mtp_kv_plan(true, i8, hy, true).int8 == old_int8(i8, hy));
    }
    if (failed == 0) std::printf("mtp_kv_policy_test: ok\n");
    return failed == 0 ? 0 : 1;
}
