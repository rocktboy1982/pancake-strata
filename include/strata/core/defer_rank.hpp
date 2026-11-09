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

}  // namespace strata::core
