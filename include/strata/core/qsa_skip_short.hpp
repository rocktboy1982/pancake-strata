#pragma once
// STRATA_QSA_SKIP_SHORT=1 (LEADS.md Part S row S10, default off), threshold STRATA_QSA_SKIP_BELOW (default 2051): skip the
// QSA indexer's launches while the whole window is below the exactness bound.  Below 2,051 cells the selection is the
// IDENTITY (qsa.hpp: qsa_selection_width = min(n_kv, idx_top_k + idx_block - 1), every complete block and the tail are
// selected whatever the scores), yet the indexer's query norm + RoPE, `qsa_block_scores` and `qsa_block_topk` still run at
// every full-attention layer (36 dependent launches a decode / verify step on Flash-Next).  radiance PR 496: skipped,
// +1.1..+1.5% below 2K tokens, +0.0% at 7.7K, bit-identical.  Ours: we already return early inside the score kernel at width
// 2,051 (STRATA_QSA_EARLY_EXIT) but still launch; the QSA scores + top-k are 1.4% of the window (T-017); expected under +0.5%.
//
// STATE OF THIS LEAD (partly coded, NOT wired into the engine): this header is the part that is exact and host-testable:
//   - the switch and threshold parse (the threshold is clamped to the bound: above it the selection is no longer the
//     identity and a skip would change the output),
//   - `qsa_skip_ok(pos0, T, below)`: a window may skip only when its LAST row's n_kv (= pos0 + T) is <= `below`,
//   - `qsa_variant(...)`: the graph-variant index (0 normal, 1 STRATA_SELECT_LONG, 2 skip) that the four places in
//     Verifier that compute `sel_long_` from the context would take instead of repeating the expression,
//   - `qsa_identity_ids`: the selection table a skipped layer must leave in `sel_` (ids[i] = i, i < width).
// The wiring needs a THIRD captured graph variant (exec_[3][T], launched_[3][T], capture_all, prelaunch / run / pl_launch,
// the eviction code): see TEST_LEDGER.md for the exact design; it was not written blind (no HIP build, no GPU here).
#include <cstdint>
#include <cstdlib>

namespace strata::core {

constexpr int64_t kQsaExactBound = 2051;   ///< idx_top_k (2048) + idx_block (4) - 1: below it the selection is the identity

/// STRATA_QSA_SKIP_SHORT: only the text "1" switches it on (default off).
inline bool qsa_skip_short_parse(const char* v) { return v != nullptr && v[0] == '1' && v[1] == '\0'; }

/// STRATA_QSA_SKIP_BELOW: a whole number in 1..2051; unset, text, 0, negative or empty = 2051; a larger number is clamped to 2051
/// (past the bound the selection is not the identity: the skip would be lossy).
inline int64_t qsa_skip_below_parse(const char* v) {
    if (v == nullptr || v[0] == '\0') return kQsaExactBound;
    char* end = nullptr;
    const long long n = std::strtoll(v, &end, 10);
    if (end == v || *end != '\0' || n < 1) return kQsaExactBound;
    return n > kQsaExactBound ? kQsaExactBound : (int64_t) n;
}

/// The two switches read once from the environment (what the wiring would call): off and 2051 when unset.
inline bool qsa_skip_short_on() {
    static const bool on = qsa_skip_short_parse(std::getenv("STRATA_QSA_SKIP_SHORT"));
    return on;
}
inline int64_t qsa_skip_below() {
    static const int64_t below = qsa_skip_below_parse(std::getenv("STRATA_QSA_SKIP_BELOW"));
    return below;
}

/// A window of `T` rows starting at cell `pos0` (row t has n_kv = pos0 + t + 1) may skip the indexer when even its last row
/// is within `below` cells.
inline bool qsa_skip_ok(int64_t pos0, int64_t T, int64_t below) { return T >= 1 && pos0 >= 0 && pos0 + T <= below; }

/// The graph variant of a window: 1 when STRATA_SELECT_LONG is on and the window reaches `long_at` cells (as `sel_long_` is
/// computed today), else 2 when the skip is on and the window is within `below`, else 0.  The long select wins when both
/// could apply (long_at is far above the bound).
inline int qsa_variant(bool select_long_on, int64_t long_at, bool skip_on, int64_t below, int64_t pos0, int64_t T) {
    if (select_long_on && pos0 + T >= long_at) return 1;
    if (skip_on && qsa_skip_ok(pos0, T, below)) return 2;
    return 0;
}

/// What a skipped layer must leave in the selection table: the identity over the first `width` cells.
inline void qsa_identity_ids(int32_t* ids, int64_t width) {
    for (int64_t i = 0; i < width; ++i) ids[i] = (int32_t) i;
}

}  // namespace strata::core
