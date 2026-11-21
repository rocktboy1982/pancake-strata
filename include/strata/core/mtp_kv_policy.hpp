#pragma once
// STRATA_MTP_KV_FP16 (opt-in, default off; LEADS.md Part Q5, idea from Strata-HotRod commit 890990bbb, MIT): the draft
// layer's own K/V stays FP16 while the main model's K/V is INT8 (`--kv int8`) or K8V4 (`--kv k8v4`).  The rule is the
// HotRod one: quantise the big consumer whose output the target checks, keep the small one whose worth is its
// agreement with the target.  One draft layer, so the VRAM cost is a few MiB.  The output cannot change: the target
// verifies every draft; only the acceptance can.  Header-only so the rule is tested without a GPU
// (tests/core/mtp_kv_policy_test.cpp).
//
// Without the switch (the default) the plan is exactly what src/core/mtp.cpp did before: K8V4 never applies to the
// drafter (its attention paths handle whole formats only), it takes plain INT8 under `--kv k8v4`, and otherwise it
// takes the main model's INT8 / Q4 setting.
#include <cstdlib>

namespace strata::core {

struct MtpKvPlan {
    bool int8 = false;    ///< what qsa_set_kv_int8 holds while the drafter's state is laid out
    bool hybrid = false;  ///< what qsa_set_kv_hybrid holds then (always false: no K8V4 for the drafter)
};

/// STRATA_MTP_KV_FP16: unset, empty or 0 is off.
inline bool mtp_kv_fp16_from(const char* v) { return v != nullptr && v[0] != '\0' && std::atoi(v) != 0; }
inline bool mtp_kv_fp16_on() {
    static const bool on = mtp_kv_fp16_from(std::getenv("STRATA_MTP_KV_FP16"));
    return on;
}

/// The K/V globals the drafter's qsa_state_init sees.  `main_*` are the globals at that moment.  Q4_0 main K/V is left as
/// it is (the switch is about INT8 / K8V4 only).
inline MtpKvPlan mtp_kv_plan(bool fp16_on, bool main_int8, bool main_hybrid, bool main_q4) {
    MtpKvPlan p;
    p.hybrid = false;
    if (fp16_on && !main_q4 && (main_int8 || main_hybrid)) p.int8 = false;   // FP16 beside the INT8 / K8V4 main
    else p.int8 = main_hybrid ? true : main_int8;                             // the previous rule, unchanged
    return p;
}

}  // namespace strata::core
