#pragma once
// MOD_PLAN.md section 6 (the BigMoeOnEdge switches, LEADS.md Part I row I1), STRATA_PREFETCH_PREV=K (opt-in, exact,
// default off): temporal prefetch.  At layer l's look-ahead submit the predictor thread also warms the experts the PREVIOUS
// window's last token routed at layers l+1 .. l+K (K in 1..8), minus those already resident in the GPU cache.  Only the
// warming source tiers (file / mmap) have pages to warm; on the pinned-RAM tiers the look-ahead is not started and the
// switch does nothing.  BigMoeOnEdge measured this as a loss where a miss is a flash read (gpt-oss prefetch 1: 828-895 ms
// per token against 414 baseline) and found the previous token's experts predict the next token's 38.0 / 35.7 / 17.9%
// (theirs); ours are PCIe / CPU misses, so this is an arm, not an expectation.  The outputs are the same with or without
// it (prefetch only).
#include <cstdint>
#include <cstdlib>

namespace strata::program {

constexpr int kPrefetchPrevMaxK = 8;     ///< layers ahead
constexpr int kPrefetchPrevMaxIds = 16;  ///< routed experts remembered per layer

/// STRATA_PREFETCH_PREV parsed: K in 1..8, else 0 (off); text, 0, a negative or a larger value are off, never clamped.
inline int prefetch_prev_parse(const char* v) {
    if (v == nullptr || v[0] == '\0') return 0;
    char* end = nullptr;
    const long k = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' || k < 1 || k > kPrefetchPrevMaxK) return 0;
    return (int) k;
}

/// The experts to warm for one layer: `prev[0..n)` (the previous token's routed ids at this layer, -1 = none) deduplicated,
/// in first-seen order, valid (0 <= e < n_expert) and not resident (`res_row[e] >= 0` marks resident; null: no filter).
/// Writes at most `kPrefetchPrevMaxIds` ids to `out`; returns the count.
inline int prefetch_prev_collect(const int32_t* prev, int n, const int32_t* res_row, int n_expert, int32_t* out) {
    int cnt = 0;
    if (n > kPrefetchPrevMaxIds) n = kPrefetchPrevMaxIds;
    for (int i = 0; i < n; ++i) {
        const int32_t e = prev[i];
        if (e < 0 || e >= n_expert) continue;
        if (res_row != nullptr && res_row[e] >= 0) continue;
        bool dup = false;
        for (int j = 0; j < cnt; ++j) dup = dup || out[j] == e;
        if (!dup) out[cnt++] = e;
    }
    return cnt;
}

}  // namespace strata::program
