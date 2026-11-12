#pragma once
#include <cstdint>

namespace strata::kernels {
// Configure before session capture. Existing graphs retain their selected kernels.
// This experiment is off by default and does not modify the legacy router.
void native_router_set_enabled(bool enabled);
bool native_router_enabled();

// Pinned CUDA topk-moe contract for ONE token, 512 experts, top 10, softmax,
// no selection bias, lower normalization clamp 2^-14, and scale 1.
// Reads 512 finite F32 logits; writes 10 I32 IDs and 10 F32 weights. Equal
// computed probabilities select the lower expert index. All spans must be
// four-byte aligned and outputs disjoint from each other and the input.
// Requires a nonnull ordered CUDA stream. No allocation or synchronization.
void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream);
/// n_tok rows at once (logits [n,512], ids/weights [n,10]); each row exactly as the single call.
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream);
/// The same contract for a model family's (experts, top-k): 512 / 10 (Flash-Next: the two calls above, unchanged) and
/// 256 / 8 (Qwen3.6-35B-A3B).  logits [n_tok, n_expert], ids / weights [n_tok, k].  CUDA / HIP.
bool native_router_topk_supported(int n_expert, int k);
void native_router_topk(const float* logits, int32_t* ids, float* weights, int n_expert, int k, int n_tok, void* stream);
/// STRATA_ROUTE_PUBLISH: native_router_top10_multi then doorbell_publish_res(x, ids, d_res, ...) in one launch (1..8 rows)
void native_router_top10_publish_res(const float* logits, int32_t* ids, float* weights, int n_tok, const float* x,
                                     const int32_t* d_res, int n_expert, int64_t n, float* x_out, int32_t* ids_out,
                                     uint32_t* seq, void* stream);
/// STRATA_ROUTE_FAST=1 (opt-in, read once; off: nothing changes).
bool native_router_fast_enabled();
/// STRATA_ROUTE_FAST: the window's top-10 (route_multi's arithmetic: ids / weights bitwise native_router_top10_multi's)
/// and the CPU pool's doorbell in ONE launch, 1..8 rows: ids_out / w_out (w_out may be null) get the ids / weights,
/// x_out the n floats of x (every row; x == nullptr: the router GEMV launch before this one already copied them,
/// bf16_router_xpub_arm), then the ring d_seq is STORED as seq_value (the step's number, nonzero) instead of read back
/// over PCIe and incremented.  Ordering and fences: see native_router.cu.
void native_router_top10_publish_fast(const float* logits, int32_t* ids, float* weights, int n_tok, const float* x,
                                      int64_t n, float* x_out, int32_t* ids_out, float* w_out, uint32_t* d_seq,
                                      uint32_t seq_value, void* stream);
/// STRATA_ROUTE_RESIDENT: swap non-resident tail-rank experts for resident ones within `margin` logits (see the .cu).
/// `stats` is nullptr or 4 device uint64 counters: tail entries seen, swaps, non-resident before, non-resident after.
void native_route_resident(const float* logits, int32_t* ids, float* weights, const int32_t* res_layer, int n_tok, float margin,
                           int rank_lo, int rank_hi, unsigned long long* stats, void* stream);
}
