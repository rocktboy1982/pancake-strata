#pragma once
// MOD_PLAN.md section 6 item 42 (LEADS.md Part J, kindi): STRATA_CKPT_STRIDE=N (default off, state placement only: the same
// bytes).  Our mid-prompt checkpoints are `--prompt-cache-every` (16384) tokens apart; kindi spreads recurrent-state
// checkpoints at a FIXED SMALL stride from the point where the request diverged from the cache, so a later request that
// edits the middle of the prefix resumes from the nearest one below the edit instead of the root or the last turn.
//   - stride N > 0: the periodic checkpoint distance of a request that resumed at R is N (not --prompt-cache-every), for at most
//     STRATA_CKPT_STRIDE_MAX (default 3) checkpoints per request: R+N, R+2N, R+3N (the position is the first chunk end at
//     or past it).  More would evict the root and the turn checkpoint from the --prompt-cache (6) slots for one request.
//   - --prompt-cache-every 0 (periodic checkpoints off) or ckpt=0 stays off: the stride never turns checkpoints on.
//   - unset / 0: next_check == done + every, the old rule exactly.
// Pays only if turns diverge inside the prefix (MOD_PLAN item 42: the share of such requests is not recorded; a host replay
// of the turn log can count it).  Single GPU only (a layer split keeps `--prompt-cache-every`).
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace strata::program {

struct CkptStride {
    int64_t stride = 0;      ///< 0: off
    int64_t max_count = 3;   ///< checkpoints one request takes at the stride
};

inline CkptStride ckpt_stride_from_env(const char* stride, const char* max_count) {
    CkptStride c;
    if (stride != nullptr && stride[0] != '\0') c.stride = std::max<int64_t>(0, std::atoll(stride));
    if (max_count != nullptr && max_count[0] != '\0') c.max_count = std::max<int64_t>(0, std::atoll(max_count));
    return c;
}

/// The next periodic checkpoint position: the request resumed at `resume`, the prompt is read up to `done`, `every` is
/// --prompt-cache-every.  INT64_MAX: none.
inline int64_t ckpt_next_check(const CkptStride& c, int64_t resume, int64_t done, int64_t every) {
    constexpr int64_t none = std::numeric_limits<int64_t>::max();
    if (every <= 0) return none;                    // off stays off
    if (c.stride <= 0) return every > none - done ? none : done + every;   // the old rule (saturates: no signed overflow)
    const int64_t taken = done > resume ? (done - resume) / c.stride : 0;
    if (taken >= c.max_count) return none;
    return c.stride > none - done ? none : done + c.stride;   // a huge stride: none, not an overflow
}

}  // namespace strata::program
