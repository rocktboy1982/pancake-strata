// include/strata/kernels/head_shortlist.hpp - STRATA_HEAD_SHORTLIST=<file.shl> (opt-in, LOSSY; MOD_PLAN.md section 6 item 38):
// the REFERENCE form of the shortlist output head, on the full logits.  After the usual head GEMV has written logits[j * vocab + r],
// head_shortlist_mask scores the K cluster centres against column j's head input, keeps the best n clusters and writes -inf over
// every logit outside them; the existing pick (argmax_rows / sample_tokens) then runs unchanged on the masked logits and gives the
// shortlist pick (the rule of include/strata/program/head_shortlist.hpp).  It saves nothing (the full GEMV still runs): it is the
// quality arm, and the oracle the gathered-rows kernel (not written) must match.  Device pointers; capturable.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

inline constexpr int kHeadShortlistMaxK = 4096;     // clusters (scores and keep flags live in shared memory)
inline constexpr int kHeadShortlistMaxCols = 64;

/// Device scratch for the centre scores of `ncols` columns.
std::size_t head_shortlist_scratch_bytes(int ncols, int k);

/// hidden: [ncols, n_embd] floats (the head's input, head_mixed_); centre: [k, n_embd] floats; cluster: [vocab] int32 in [0, k);
/// logits: [ncols, vocab] floats, masked in place; scratch: head_shortlist_scratch_bytes.  Throws std::invalid_argument for
/// k > kHeadShortlistMaxK, ncols > kHeadShortlistMaxCols or a non-positive size.
void head_shortlist_mask(const float* hidden, const float* centre, const int32_t* cluster, float* logits, int n_embd, int vocab,
                         int k, int n_keep, int ncols, float* scratch, void* stream);

}  // namespace strata::kernels
