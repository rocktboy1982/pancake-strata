// include/strata/kernels/route_prior.hpp - STRATA_ROUTE_PRIOR=lambda (opt-in, changes the output): cache-conditioned
// routing in verify windows.  See src/kernels/cuda/route_prior.cu.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Top-10 per token over `logits` (n_tok x 512) where a resident expert (`res_layer[e] >= 0`, the layer's residency
/// table) gets `lambda` added to its logit for the SELECTION; the weights are the router's own probabilities
/// (softmax over the 512 unbiased logits) renormalised over the 10 picked, as the native router does.
/// `depth_acc` > 0 (MOD_PLAN item 23, AcceptMoE-style, STRATA_ROUTE_PRIOR_DEPTH): the bonus grows with the row's position
/// in the verify window, lambda * (1 - depth_acc^pos) with pos = row0 + the row's index - 0 at position 0 (the token
/// every window commits stays exactly routed), towards lambda for deep rows that are rarely kept.  0: lambda everywhere.
void route_prior_top10(const float* logits, const int32_t* res_layer, float lambda, int32_t* ids, float* weights,
                       int n_tok, void* stream, float depth_acc = 0.0f, int row0 = 0);

/// MOD_PLAN F13 (route-prior guards, all opt-in; call once at start-up, BEFORE any window graph is captured: the kernel's launch
/// arguments are baked into a captured graph).  Not calling it, or calling it with (nullptr, 0, 0), leaves the kernel as it was.
///  - `gate` (STRATA_ROUTE_PRIOR_SCOPE): a mapped float (device pointer) the serve loop writes before each window.  The kernel
///    reads it at run time: > 0 scales lambda by it (1 = full bonus), <= 0 makes the kernel return at once, so the router's own
///    top-10 that is already in ids / weights stands untouched (the window is routed exactly).  nullptr: no gate.
///  - `protect` (STRATA_ROUTE_PRIOR_PROTECT=N, also STRATA_ROUTE_PRIOR_KEEP=N): the router's own first N picks are never displaced;
///    the bonus only reorders ranks N..9.  0..10.
///  - `keep_p` (STRATA_ROUTE_PRIOR_KEEP_P=x): the router's picks whose probability is at least x of its top pick's are never
///    displaced either (an adaptive rank: more are protected when the router is sure).  0: off.
void route_prior_configure(const float* gate, int protect, float keep_p);

}  // namespace strata::kernels
