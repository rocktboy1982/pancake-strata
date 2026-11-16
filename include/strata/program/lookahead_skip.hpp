#pragma once
// MOD_PLAN.md section 6 (the BigMoeOnEdge switches, LEADS.md Part I) (LEADS.md Part I row I6), STRATA_LOOKAHEAD_SKIP_DROPPED=1 (opt-in, default off, only
// with STRATA_MISS_SKIP_W and / or STRATA_MISS_SKIP_CUM set): the router look-ahead leaves unwarmed (and does not hand to
// the Foresight swap) the predicted experts that the miss-skip rule would drop anyway.  BigMoeOnEdge: an early read that
// makes a to-be-dropped expert resident un-drops it (phone, depth 4, drop 0.75: 78 -> 244 MB/token and half the tok/s
// without the rule).  In our engine the skip rule looks at residency in the GPU cache, so a warm of the file / arena pages
// does not un-drop; a Foresight swap copy does, and a warm of a dropped expert is wasted bytes either way.
//
// The rule is the engine's own (expert_source.cpp, the MISS_SKIP block), applied to the PREDICTED weights:
//   token's predicted picks sorted by logit (best first), share_j = softmax over the top k picks;
//   _W:   entry j is dropped when share_j < share_lt;
//   _CUM: entry j is dropped when sum(share_i, i < j) >= cum_ge.
// An expert is left cold only when EVERY token that predicts it would drop it (the engine skips an expert only when every
// token of the window that routes it meets the rule).  STRATA_MOBILE is not modelled.  Lossy only through the skip rule
// itself, which has to be on; the look-ahead only warms, so the outputs of a run are the same with or without this switch.
#include <cmath>
#include <cstdint>

namespace strata::program {

/// Shares of one token's top-k logits (best first in `top`): softmax, written to `share[0..k)`.  k <= 0 writes nothing.
inline void lookahead_shares(const float* top, int k, float* share) {
    if (k <= 0) return;
    float mx = top[0];
    for (int j = 1; j < k; ++j) mx = top[j] > mx ? top[j] : mx;
    float sum = 0.0f;
    for (int j = 0; j < k; ++j) { share[j] = std::exp(top[j] - mx); sum += share[j]; }
    for (int j = 0; j < k; ++j) share[j] = sum > 0.0f ? share[j] / sum : 0.0f;
}

/// Would the miss-skip rule (share_lt > 0: _W, cum_ge > 0: _CUM) drop entry j of a token whose shares are `share[0..k)`
/// (best first)?  A NaN share never drops, an empty rule (both 0) never drops.
inline bool lookahead_would_drop(const float* share, int k, int j, float share_lt, float cum_ge) {
    if (j < 0 || j >= k) return false;
    const float s = share[j];
    if (s != s) return false;
    if (share_lt > 0.0f && s < share_lt) return true;
    if (cum_ge > 0.0f) {
        float above = 0.0f;
        for (int i = 0; i < j; ++i) above += share[i];
        if (above >= cum_ge) return true;
    }
    return false;
}

}  // namespace strata::program
