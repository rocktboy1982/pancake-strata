#pragma once
// MOD_PLAN.md section 6 (the BigMoeOnEdge switches, LEADS.md Part I row I4), STRATA_ROUTE_RANGE=L (opt-in, LOSSY, needs
// STRATA_ROUTE_PRIOR set to any value above 0): the residency bonus of the route prior is L x (this token's own score
// range, max - min of its 512 router logits) instead of lambda in raw logit units, so one L means the same on any router and
// any token (BigMoeOnEdge --expert-substitute L: L=0.15 took decode 2.37 -> 3.84 tok/s on their desktop host at perplexity
// +1-4%, L=0.30 +25% perplexity, theirs, never reproduced here).  The family we have: STRATA_ROUTE_PRIOR_DELTA=1 scales by
// the LAYER's running mean range (route_prior.cu); this one uses the token's own range, which is their rule exactly.
// Weights stay the router's own (the selection only changes), as in the route prior.
//
// The kernel is src/kernels/cuda/route_prior.cu (a mode of route_prior_k); this header holds the pure rule and a host
// reference of the selection so a ctest pins the semantics (scale and shift invariance, the L = 0 identity, monotone
// substitution) without a GPU.
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace strata::program {

/// The bonus for a token whose logits span [lo, hi]: L x (hi - lo); 0 for a non-finite or negative span or L <= 0.
inline float route_range_bonus(float L, float hi, float lo) {
    const float r = hi - lo;
    if (!(L > 0.0f) || !std::isfinite(L) || !std::isfinite(r) || r < 0.0f) return 0.0f;
    return L * r;
}

/// STRATA_ROUTE_RANGE parsed: L in (0, 4], else 0 (off).  An out-of-range value is off, not clamped (a typo must not arm a
/// lossy switch at a value nobody asked for).
inline float route_range_parse(const char* v) {
    if (v == nullptr || v[0] == '\0') return 0.0f;
    char* end = nullptr;
    const double f = std::strtod(v, &end);
    if (end == v || !(f > 0.0) || f > 4.0 || !std::isfinite(f)) return 0.0f;
    return (float) f;
}

/// Host reference of the selection: the top-k of (logit + bonus if resident), best first, ties to the lower id.
/// `res[e] >= 0` marks a resident expert.  Returns the number of picks that are NOT the router's own top-k (substituted).
inline int route_range_select(const float* logits, int n, const int32_t* res, float L, int k, int32_t* ids) {
    float hi = logits[0], lo = logits[0];
    for (int e = 1; e < n; ++e) { hi = logits[e] > hi ? logits[e] : hi; lo = logits[e] < lo ? logits[e] : lo; }
    const float bonus = route_range_bonus(L, hi, lo);
    // the biased pick, then the router's own pick, by repeated arg-max with a taken mask (n is small in tests)
    auto pick = [&](bool biased, int32_t* out) {
        bool taken[1024] = {false};
        for (int j = 0; j < k; ++j) {
            int best = -1;
            float bs = 0.0f;
            for (int e = 0; e < n; ++e) {
                if (taken[e]) continue;
                const float s = logits[e] + (biased && res != nullptr && res[e] >= 0 ? bonus : 0.0f);
                if (best < 0 || s > bs) { best = e; bs = s; }
            }
            taken[best] = true;
            out[j] = best;
        }
    };
    int32_t own[64];
    pick(false, own);
    pick(true, ids);
    int sub = 0;
    for (int j = 0; j < k; ++j) {
        bool in_own = false;
        for (int i = 0; i < k; ++i) in_own = in_own || own[i] == ids[j];
        if (!in_own) ++sub;
    }
    return sub;
}

}  // namespace strata::program
