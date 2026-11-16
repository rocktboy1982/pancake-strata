// MOD_PLAN.md section 7: the rules behind the T-602 guard and the T-613 / T-630 diagnostics (include/strata/core/diag_rules.hpp).
#include "strata/core/diag_rules.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace strata::core::diag;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

int main() {
    // T-602: BF16 rows present (default, GR_RDNA_Q8 off): the generic read may write the copy
    check(il_copy_generic_read_ok(true, true, false, false), "bf16 rows only");
    check(il_copy_generic_read_ok(true, true, true, true), "bf16 and int8 copies both");
    // GR_RDNA + GR_RDNA_Q8: BF16 rows skipped, int8 copies present: refuse (null BF16 pointer in the generic kernels)
    check(!il_copy_generic_read_ok(false, false, true, true), "int8 copies only");
    check(!il_copy_generic_read_ok(false, true, true, true), "down int8 only");
    check(!il_copy_generic_read_ok(true, false, true, true), "up int8 only");
    // neither form present: not this rule's case (the read's own argument check stops it)
    check(il_copy_generic_read_ok(false, false, false, false), "nothing present is left to the argument check");
    check(std::strstr(il_copy_refused_line(), "T-602") != nullptr && std::strchr(il_copy_refused_line(), '\n') == nullptr,
          "refusal line is one line and names the defect");

    // T-630
    const std::string a = launch_error_line("moe_combine_write_norm_rs", "hipErrorIllegalAddress", "an illegal memory access was encountered", 64, 256);
    check(a.find("kernel moe_combine_write_norm_rs: an illegal memory access was encountered (hipErrorIllegalAddress), grid 64, block 256") != std::string::npos, "launch line with dims");
    check(a.find('\n') == std::string::npos, "launch line has no newline");
    const std::string b = launch_error_line("rms_rows", "x", "y", 0, 0);
    check(b.find("grid and block not recorded") != std::string::npos && b.find("grid 0") == std::string::npos, "launch line without dims");
    check(!launch_error_line(nullptr, nullptr, nullptr, 0, 0).empty(), "null strings are safe");

    // T-613
    check(gate_queue_blocks(3, 20, 25), "queued layer 20 <= wanted 25 blocks");
    check(gate_queue_blocks(1, 25, 25), "queued layer == wanted blocks");
    check(!gate_queue_blocks(3, 26, 25), "queued layer past the wanted one does not block");
    check(!gate_queue_blocks(0, 0, 25), "an empty queue does not block");
    const std::string g = gate_trace_line("timeout", 25, 25, 14, 14, 13, true, 4, 15, 20012.5);
    check(g == "strata verify: adapt idle-copy trace: timeout layer 25 want 25 reached 14 (atomic 14, copy signal 13, signal copy), "
               "queued 4 (front layer 15), 20012.5 ms - the queue still holds a layer the wait needs (not issued)", "gate line, blocked queue");
    const std::string h = gate_trace_line("landed", 25, 25, 25, 25, -1, false, 0, -1, 0.3);
    check(h == "strata verify: adapt idle-copy trace: landed layer 25 want 25 reached 25 (atomic 25, copy signal -1, signal host), "
               "queued 0, 0.3 ms", "gate line, landed");
    std::printf("diag_rules_test: %d checks ok\n", checks);
    return 0;
}
