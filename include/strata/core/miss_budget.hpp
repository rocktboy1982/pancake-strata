#pragma once
// MOD_PLAN section 6 item 20 (R&D-B miss budget), step 2: STRATA_MISS_BUDGET=<weight share b> (opt-in, lossy, default off).
// One controller for a missed expert instead of the separate STRATA_MISS_SKIP_W / _CUM thresholds: per token, the missed
// experts (outside the VRAM tier, so the CPU would compute them) are skipped from the lightest router weight up, as long as
// the weight skipped stays within b of the token's weight sum.  So a token loses at most b of its output weight to skipped
// experts, whatever its weights look like, where SKIP_W's fixed floor skips nothing on a flat token and a lot on a peaked one.
// The skip is the pool's tail skip (kind -2: neither copied nor computed, a zero row, no renormalisation).  An expert is
// skipped only when EVERY token of the window that routes it skips it (one skip zeroes all its rows), as SKIP_W.
// Optional: with STRATA_TIER_MISS_FILE (the per-expert gate importance table of item 20 step 1,
// include/strata/program/tier_miss_budget.hpp) and STRATA_TIER_MISS_BUDGET, only the experts that table marks low-importance
// are candidates, so the experts the tier already treats as the cheapest to lose are the only ones the controller may skip.
// Pure logic here; expert_source.cpp applies it where the SKIP_W rule runs.
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace strata::core {

/// STRATA_MISS_BUDGET's value: the share in (0, 1); 0 unset, empty or out of range (off).
inline float miss_budget_parse(const char* v) {
    if (v == nullptr || v[0] == '\0') return 0.0f;
    char* end = nullptr;
    const double b = std::strtod(v, &end);
    if (end == v || *end != '\0' || !(b > 0.0) || !(b < 1.0) || !std::isfinite(b)) return 0.0f;
    return (float) b;
}

/// One token: w[0..k) the router weights in routing order, cand[j] true when entry j is a missed expert that may be skipped.
/// Sets skip[j] for the entries to skip and returns how many.  Candidates are taken lightest first (ties: the later routing
/// rank first, the weaker choice), while the running skipped weight stays <= b * (sum of all k weights).  A candidate that
/// would pass the budget ends the walk (the budget is a prefix of the sorted candidates, not a knapsack).  A NaN weight,
/// a negative weight or an empty token skips nothing.
inline int miss_budget_token(const float* w, const bool* cand, int k, float b, bool* skip) {
    for (int j = 0; j < k; ++j) skip[j] = false;
    if (!(b > 0.0f) || k <= 0) return 0;
    double sum = 0.0;
    for (int j = 0; j < k; ++j) {
        if (!(w[j] >= 0.0f)) return 0;   // NaN or negative
        sum += (double) w[j];
    }
    if (!(sum > 0.0)) return 0;
    const double lim = (double) b * sum * (1.0 + 1e-6);
    double acc = 0.0;
    int n = 0;
    bool used[64] = {};   // k <= 64 (a router names at most a few dozen experts); more: skip nothing
    if (k > 64) return 0;
    for (;;) {
        int best = -1;
        for (int j = 0; j < k; ++j) {
            if (!cand[j] || used[j]) continue;
            if (best < 0 || w[j] < w[best] || (w[j] == w[best] && j > best)) best = j;
        }
        if (best < 0) break;
        if (acc + (double) w[best] > lim) break;
        acc += (double) w[best];
        skip[best] = true;
        used[best] = true;
        ++n;
    }
    return n;
}

}  // namespace strata::core
