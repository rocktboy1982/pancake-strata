// include/strata/prefill/combine_write_rule.hpp - when STRATA_PF_COMBINE_WRITE (FK-36, moe_combine_write_norm_rs) may run (no GPU needed to check).
//
// The fused kernel writes the next half's xn16 / xn16_lo while its other blocks still read slot_dev, w, Dm, shared, sg and row_sd.  With
// STRATA_PF_HC_REGION=1 the xn16 set is carved from the START of the region the MoE set also starts at (Prefill::carve: `h` and `c` share
// `base`), so xn16 lies over logits, w, ids, slot_dev ... of the same chunk.  The two-kernel path is safe there (the combine has ended before
// gr_write_norm_rs writes xn16); the fused kernel is not: block t's image rows overwrite the slot / weight rows of later tokens.
// T-630 (unconfirmed static analysis until a box run): that is the engine fault of spec 119 (chunk 2304, illegal memory access).
#pragma once

#include <cstdint>

namespace strata::prefill::combine_write {

/// The fused combine + write may run only where xn16 does not share memory with the MoE set (STRATA_PF_HC_REGION off).
inline bool allowed(bool hc_region) { return !hc_region; }

/// Bytes of the xn16 image of a chunk of T tokens (token stride D + pad halves) vs the byte offset of w in the MoE set (after logits):
/// true when the image reaches the MoE inputs (hc_region only: the region's first set).
inline bool hc_image_reaches_moe_inputs(int64_t T, int64_t n_expert, int64_t D, int64_t pad) {
    return T * (D + pad) * 2 > T * n_expert * 4;
}

}  // namespace strata::prefill::combine_write
