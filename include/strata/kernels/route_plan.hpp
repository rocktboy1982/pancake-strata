// include/strata/kernels/route_plan.hpp - STRATA_ROUTE_PLAN=1 (opt-in): the verify window's routing top-10 and the
// all-resident plan in ONE launch (one graph node fewer per MoE layer of the zero-doorbell window).  Ported from
// jstamagal/Strata-gfx1030 ef00764552 (LEADS.md "gfx1030 decode").  src/kernels/cuda/route_plan.cu has the kernel;
// route_plan_parity checks ids, weights and the plan's bytes against native_router_top10_multi + resident_plan.
#pragma once

#include <cstdint>

namespace strata::kernels {

// STRATA_ROUTE_PLAN=1: the top-10 and the plan in one launch; anything else, or unset: off.  (The fork's =2, the
// router's BF16 projection in the same launch, is not ported: see route_plan.cu.)
bool route_plan_enabled();
int route_plan_mode();

// logits [n_tok][512] -> ids / weights [n_tok][10], exactly native_router_top10_multi's (native_router_top10's for one
// token), then resident_plan(ids, n_tok * 10, 10, res_layer, n_expert, cache_base, slot_off, blob, plan, capx,
// nullptr, 0, stream, plan_err)'s plan: the all-resident form (no skip word; an expert that is not resident leaves an
// empty plan and raises *plan_err).  1 <= n_tok <= 8.  Device pointers (plan_err may be mapped host memory, or null);
// capturable.
void native_route_plan(const float* logits, int32_t* ids, float* weights, int n_tok, const int32_t* res_layer,
                       int n_expert, const uint8_t* cache_base, const unsigned long long* slot_off, long long blob,
                       int32_t* plan, long long capx, uint32_t* plan_err, void* stream);

}  // namespace strata::kernels
