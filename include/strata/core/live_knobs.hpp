#pragma once
// MOD_PLAN A44 (STRATA_BANDIT, core/epoch_bandit.hpp): the opt-in switches the epoch bandit turns on or off at run time.
// Only the serve loop writes them, and only between windows (before the next window is staged); the window's readers
// take them at staging (the rank guard: Verifier::stage_inputs copies it to the window's mapped word and to the plan
// sink, so the GPU's row copy and the host pool read one value) or per call (the adaptive tier, the pool pre-read).
// Their defaults are "as configured": with STRATA_BANDIT unset nothing writes them and every reader behaves as before.
#include <atomic>

namespace strata::core {

struct LiveKnobs {
    /// STRATA_VERIFY_DEFER_CPU_TOP as the bandit sets it for the next windows: -1 = not set (defer_top(), the default),
    /// 0 = the rank guard off, n > 0 = on with top n.  Taken at a window's staging (defer_rank.hpp defer_top_now()).
    std::atomic<int> top{-1};
    /// Set by a Verifier that keeps the mapped rank-guard word its window graphs read (verify.cpp): the bandit may move
    /// `top` only then (the graphs bake defer_top() into their kernel arguments otherwise).
    std::atomic<bool> top_word{false};
    /// STRATA_ADAPT_KMISS: true = the tier admits by the usage gain this round (k-th-miss paused; its miss history is
    /// still kept).  kmiss_tier.hpp admits().
    std::atomic<bool> kmiss_off{false};
    /// STRATA_POOL_PREREAD: true = the router lookahead's warm() posts nothing (expert_source.cpp).
    std::atomic<bool> preread_off{false};
};

inline LiveKnobs& live_knobs() {
    static LiveKnobs k;
    return k;
}

}  // namespace strata::core
