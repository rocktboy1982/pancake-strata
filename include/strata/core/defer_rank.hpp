#pragma once
// MOD_PLAN item 28 (KTransformers rank guard): with STRATA_VERIFY_DEFER_CPU, a CPU entry is "heavy" (kept on time in its
// own layer instead of joining one layer late) when its routing weight is >= STRATA_VERIFY_DEFER_CPU_BELOW, or when it is
// among the token's top STRATA_VERIFY_DEFER_CPU_TOP entries by weight (0 / unset: off).  The host pool (which experts run
// first) and the GPU's row copy (copy_rows_by_weight_kernel, which repeats this rule) must agree, so both use this rank:
// the entries of the same token with a larger weight, ties broken by the lower index.
#include <cstdint>
#include <cstdlib>

namespace strata::core {

inline int defer_top() {
    static const int top = [] {
        const char* v = std::getenv("STRATA_VERIFY_DEFER_CPU_TOP");
        const int t = v != nullptr ? std::atoi(v) : 0;
        return t > 0 && t < 64 ? t : 0;
    }();
    return top;
}

inline bool defer_heavy_entry(const float* w, int64_t i, int k, float heavy, int top) {
    if (w[i] >= heavy) return true;
    if (top <= 0 || k <= 0) return false;
    const int64_t t0 = (i / k) * k;
    int rank = 0;
    for (int64_t m = t0; m < t0 + k; ++m)
        if (w[m] > w[i] || (w[m] == w[i] && m < i)) ++rank;
    return rank < top;
}

// MOD_PLAN A4 fix (the rank guard's short-chat loss, test phase 01: _TOP=2 +20.1% edit, -6.9% short): a split layer
// runs two pool passes, the heavy jobs (partial flag) and then the light ones (the CPU experts with no heavy entry, whose
// rows join one layer late).  With STRATA_VERIFY_DEFER_CPU_FOLD=<n> (0 / unset: off) a layer with at least one heavy
// job and at most n light jobs runs them all in one pass and raises the partial flag after it: no second pass (its
// start, its gate/up -> down chain on one or two experts, the first pass's idle tail), but post(l) also waits for the
// light rows.  Exact w.r.t. the split: post(l) and late_add pick their rows by the weight rule above on the GPU, not
// by when the pool wrote them, and a row's value does not depend on the pass that computed it (the pool cuts every
// phase by output rows).  A layer with no heavy job stays one pass with the partial flag first (folding it would only
// make post(l) wait).
inline int defer_fold() {
    static const int fold = [] {
        const char* v = std::getenv("STRATA_VERIFY_DEFER_CPU_FOLD");
        const int f = v != nullptr ? std::atoi(v) : 0;
        return f > 0 ? f : 0;
    }();
    return fold;
}

inline bool defer_fold_light(int heavy_jobs, int light_jobs, int fold) {
    return fold > 0 && heavy_jobs > 0 && light_jobs > 0 && light_jobs <= fold;
}

}  // namespace strata::core
