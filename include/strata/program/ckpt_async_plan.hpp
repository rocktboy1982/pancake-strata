#pragma once
// MOD_PLAN.md section 6 item 37 (LEADS.md Part K), STRATA_CKPT_ASYNC=1 on HIP (default off, state placement only: the same bytes).
// The rules src/core/conversation_state.cpp applies to the background checkpoint copy, host-testable
// (src/program/ckpt_async_plan_test.cpp):
//   - the path is on when STRATA_CKPT_ASYNC starts with '1' and the backend has the pieces: CUDA and HIP (the HIP compat
//     header maps the stream, pinned-host, event and MemGetInfo calls it uses); never gfx906 (not measured: the old
//     card's copy engine is shared with its decode) and never Metal (the state lives in the decode window);
//   - the one-time VRAM stage buffer (one GDN checkpoint, 112.2 MiB at the daily preset) is allocated on HIP only when more
//     than `reserve` bytes stay free after it; otherwise the save is the synchronous copy it always was.  CUDA keeps its
//     old rule (no check) so the CUDA default and opt-in paths are byte-identical.
#include <cstddef>

namespace strata::program {

enum class CkptBackend { Cuda, Hip, HipGfx906, Metal };

inline bool ckpt_async_allowed(CkptBackend b, const char* env) {
    if (b == CkptBackend::HipGfx906 || b == CkptBackend::Metal) return false;
    return env != nullptr && env[0] == '1';
}

/// bytes that must stay free on the card after the stage buffer (HIP): the decode's own allocations come first
constexpr size_t kCkptStageReserve = (size_t) 3 << 29;   // 1.5 GiB, the rule of the VRAM slots

inline bool ckpt_stage_room_ok(CkptBackend b, size_t free_bytes, size_t need) {
    if (b != CkptBackend::Hip) return true;
    return free_bytes > need && free_bytes - need > kCkptStageReserve;
}

}  // namespace strata::program
