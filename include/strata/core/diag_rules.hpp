// include/strata/core/diag_rules.hpp - three small rules behind the open-defect diagnostics (MOD_PLAN.md section 7):
//   T-602  il_copy_generic_read_ok: may the hyper-connection read take the generic path that writes the interleaved copy
//          (STRATA_MMVQ_IL_PLAIN)?  Not when the down / up projections exist only as int8 copies (STRATA_GR_RDNA_Q8: the
//          BF16 rows were never loaded, so the generic kernels would read a null pointer).
//   T-630  launch_error_line: the extra line STRATA_LAUNCH_CHECK_VERBOSE=1 prints before the prompt path ends the process.
//   T-613  gate_trace_line / gate_queue_blocks: the lines STRATA_ADAPT_IDLE_COPY_TRACE=1 prints around the adaptive gate's wait.
// Pure functions, no GPU: tested by src/core/diag_rules_test.cpp.  Logging and one refusal; no switch changes what a run computes.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::core::diag {

/// T-602.  false: the generic read (the one that writes the interleaved copy) cannot run because a BF16 down or up row set
/// is absent while its int8 copy is present; the caller then drops the interleaved copy and the other read runs.
inline bool il_copy_generic_read_ok(bool has_w_down, bool has_w_up, bool has_r8_down, bool has_r8_up) {
    return !((!has_w_down && has_r8_down) || (!has_w_up && has_r8_up));
}

/// The one line said (once) when il_copy_generic_read_ok refused.
inline const char* il_copy_refused_line() {
    return "strata: STRATA_MMVQ_IL_PLAIN: the hyper-connection down/up weights exist only as int8 copies (STRATA_GR_RDNA_Q8), "
           "so the generic read that writes the interleaved copy cannot run; no interleaved copy is written, the RDNA read and "
           "the plain q8_1 columns run instead (MOD_PLAN.md section 7, T-602)";
}

/// T-630.  `name` / `text`: cudaGetErrorName / cudaGetErrorString of the launch error; `kernel` the check() label;
/// grid_x / block_x the launch's first dimensions (0: not recorded at that call).  One line, no newline.
inline std::string launch_error_line(const char* kernel, const char* name, const char* text, unsigned grid_x, unsigned block_x) {
    char b[512];
    if (grid_x != 0 && block_x != 0)
        std::snprintf(b, sizeof b,
                      "strata: launch error (STRATA_LAUNCH_CHECK_VERBOSE): kernel %s: %s (%s), grid %u, block %u; the error may be "
                      "sticky from an earlier asynchronous launch (AMD_SERIALIZE_KERNEL=3 HIP_LAUNCH_BLOCKING=1 pins the kernel)",
                      kernel ? kernel : "?", text ? text : "?", name ? name : "?", grid_x, block_x);
    else
        std::snprintf(b, sizeof b,
                      "strata: launch error (STRATA_LAUNCH_CHECK_VERBOSE): kernel %s: %s (%s), grid and block not recorded at this "
                      "check; the error may be sticky from an earlier asynchronous launch (AMD_SERIALIZE_KERNEL=3 "
                      "HIP_LAUNCH_BLOCKING=1 pins the kernel)",
                      kernel ? kernel : "?", text ? text : "?", name ? name : "?");
    return b;
}

/// T-613.  True when copies the wait needs are still in the host queue: the queue holds `queued` steps, the first of them for
/// layer `front_layer`, and that layer is not past the layer waited for (the pace call should have issued it).
inline bool gate_queue_blocks(int64_t queued, int32_t front_layer, int32_t layer) {
    return queued > 0 && front_layer <= layer;
}

/// T-613.  One line about the adaptive gate's wait: `phase` "wait" (the wait starts), "landed" (it ended) or "timeout".
/// want = the layer the counter must reach, reached = the larger of the atomic and the copy signal, signal_copy = the
/// landing signal is the stream-ordered D2H copy (else a host function), queued / front_layer = the host queue now.
inline std::string gate_trace_line(const char* phase, int32_t layer, int32_t want, int32_t reached, int32_t atomic_val,
                                   int32_t copy_signal, bool signal_copy, int64_t queued, int32_t front_layer, double elapsed_ms) {
    char b[512];
    char q[96];
    if (queued > 0)
        std::snprintf(q, sizeof q, "%lld (front layer %d)", (long long) queued, (int) front_layer);
    else
        std::snprintf(q, sizeof q, "0");
    std::snprintf(b, sizeof b,
                  "strata verify: adapt idle-copy trace: %s layer %d want %d reached %d (atomic %d, copy signal %d, signal %s), "
                  "queued %s, %.1f ms%s",
                  phase ? phase : "?", (int) layer, (int) want, (int) reached, (int) atomic_val, (int) copy_signal,
                  signal_copy ? "copy" : "host", q, elapsed_ms,
                  gate_queue_blocks(queued, front_layer, layer) ? " - the queue still holds a layer the wait needs (not issued)" : "");
    return b;
}

}  // namespace strata::core::diag
