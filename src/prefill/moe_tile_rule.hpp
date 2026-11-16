// src/prefill/moe_tile_rule.hpp - the host-testable rules of the F16 prompt-expert kernels' token tile width
// (STRATA_PF_F16_VAR_BM, STRATA_MOE_MAX_BM; Indigoshores 12d7cde5 / 0c398d60, LEADS.md Part H).  moe_f16_common.cuh
// includes this for pick_bn and work_item; moe_tile_rule_test.cpp checks it without a GPU.
#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_TILE_HD __host__ __device__
#else
#define STRATA_TILE_HD
#endif

namespace strata::prefill::fused::f16 {

constexpr int kTileRowsRule = 64;   // fused::kTileRows (include/strata/prefill/moe_fused.hpp)
constexpr int kHalfTileRule = kTileRowsRule / 2;

/// The token tile width: `forced` is STRATA_PF_F16_BN (0 unset), `var_avg` the STRATA_PF_F16_VAR_BM average-rows
/// threshold (0 off), `max_bm` STRATA_MOE_MAX_BM (32: half tiles always; anything else: no cap).
inline int pick_bn_rule(int64_t n, int n_expert, int forced, double var_avg, int max_bm) {
    if (forced == kTileRowsRule || forced == 2 * kTileRowsRule) return forced;
    const double avg = (double) n / (n_expert > 0 ? n_expert : 1);
    if (max_bm == kHalfTileRule) return kHalfTileRule;
    if (var_avg > 0.0 && avg <= var_avg) return kHalfTileRule;
    return avg >= kTileRowsRule ? 2 * kTileRowsRule : kTileRowsRule;
}

/// A half-tile launch has two work items per (64-row tile, feature block): item w -> (tile item w >> 1, half w & 1).
/// Given the 64-row tile's first row `row0` and its expert's end row `end`, the half's first row and row count; an
/// empty half (nrows <= 0) has no work.
STRATA_TILE_HD inline void half_tile_rows(int row0, int end, int half, int bn, int& r0, int& nrows) {
    r0 = row0 + half * bn;
    nrows = end - r0 < bn ? end - r0 : bn;
}

}  // namespace strata::prefill::fused::f16
