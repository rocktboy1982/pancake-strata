// src/kernels/cuda/route_prior.cu - STRATA_ROUTE_PRIOR=lambda: cache-conditioned routing (opt-in, changes the output).
//
// "Mixture of Cache-Conditional Experts" (arXiv 2412.00099), without training: when a token's 10 experts are picked,
// an expert already in VRAM gets a `lambda` bonus on its logit.  The SELECTION uses the biased logits; the WEIGHTS stay
// the router's (softmax over the 512 unbiased logits), renormalised over the 10 picked like the native router.  Fewer
// missed experts to copy over PCIe or compute on the CPU.  Unlike STRATA_ROUTE_RESIDENT (a missed tail expert swapped
// for the best resident one within a logit margin), every rank can move, by an amount the logits decide.
//
// Guards (MOD_PLAN F13, all opt-in, set once at start-up through route_prior_configure; unset = this file's earlier
// kernel, bit for bit): the router's first `protect` picks (STRATA_ROUTE_PRIOR_PROTECT, mandoof1's _KEEP is the same
// rule) and every pick whose probability is at least `keep_p` of the top pick's (STRATA_ROUTE_PRIOR_KEEP_P, zsariboga
// 8d7042ad97) are taken from the router's own order and never displaced; the bonus only reorders what is left.  A
// per-window `gate` (STRATA_ROUTE_PRIOR_SCOPE: the serve loop writes 1 / 0 into a mapped float before each window) turns
// the whole kernel off for a window: it returns at once and the router's own top-10, already in ids / weights, stands.
#include "strata/kernels/route_prior.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>

namespace strata::kernels {
namespace {

__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, m, 32));
    return v;
}
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m, 32);
    return v;
}

// one warp per token; 512 experts = 16 values per lane (as the native router)
__global__ void route_prior_k(const float* __restrict__ logits, const int32_t* __restrict__ res, float lambda,
                                  int32_t* __restrict__ ids, float* __restrict__ weights, float depth_acc, int row0,
                                  const float* gate, int protect, float keep_p) {
    if (gate != nullptr) {   // STRATA_ROUTE_PRIOR_SCOPE: this window's scale, 0 = out of scope (a tool call, a code fence, ...)
        const float g = *(volatile const float*) gate;
        if (!(g > 0.0f)) return;   // the router's own top-10 stands (whole warp, before any shuffle)
        lambda *= g;
    }
    if (depth_acc > 0.0f) lambda *= 1.0f - powf(depth_acc, (float) (row0 + (int) blockIdx.x));   // item 23: 0 at position 0
    const float* lg = logits + (size_t) blockIdx.x * 512;
    int32_t* id = ids + (size_t) blockIdx.x * 10;
    float* w = weights + (size_t) blockIdx.x * 10;
    const int lane = threadIdx.x;
    float p[16], s[16];
    float mx = -INFINITY;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        float v = lg[lane + 32 * i];
        // Non-finite logits (all-ones exponent: NaN, -inf, +inf) are clamped like the native router maps NaN to -FLT_MAX
        // (native_router.cu route_multi); a finite logit passes through bit for bit.  Tested on the bits, not with
        // isnan / fmaxf, so --use_fast_math (finite-math assumptions) cannot fold it away.  Without this a window whose
        // 512 logits are all NaN / -inf left `be` at its 1<<30 sentinel below and wrote s[be >> 5] out of bounds
        // (upstream PR #5655, "fall back to the router's own top-10"; this is our own variant of that fix).
        const unsigned u = __float_as_uint(v);
        if ((u & 0x7f800000u) == 0x7f800000u) v = ((u & 0x7fffffffu) > 0x7f800000u || (u >> 31)) ? -FLT_MAX : FLT_MAX;
        p[i] = v;
        mx = fmaxf(mx, v);
    }
    mx = wmax(mx);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int e = lane + 32 * i;
        s[i] = p[i] + (res[e] >= 0 ? lambda : 0.0f);      // the selection score
        p[i] = expf(p[i] - mx);                            // the router's probability (unnormalised)
        sum += p[i];
    }
    (void) wsum(sum);
    // Upstream PR #5655 (WilliamStone; Speedy 0.4.0 "route-prior NaN guard"): with every score NaN / -inf the old
    // `s[i] > best` never fired, the sentinel id 1 << 30 leaked into ids and `s[be >> 5]` wrote 4 MiB past the array
    // (an abort every ~5 min under STRATA_ROUTE_PRIOR).  Picked experts live in a bitmask (`taken`, bit i = this lane's
    // expert lane + 32 * i), NaN scores rank as -inf and the first unpicked expert of a lane always qualifies, so all 10
    // ids are valid and distinct whatever the logits hold (on top of the input clamp above).
    // Guards: while `own_phase` the router's own picks come first - its first `protect` ranks and every expert whose
    // probability is at least `keep_p` of the top one's (p is exp(logit - max): the top one is 1.0) - in the router's
    // order; the first pick that is neither ends the phase and the rest are chosen by the biased score.  Both guards off:
    // own_phase is false from the start (the earlier kernel).  Finite logits select exactly as before.
    unsigned taken = 0;
    float sel_p = 0.0f, sel_sum = 0.0f;
    bool own_phase = protect > 0 || keep_p > 0.0f;
    for (int rank = 0; rank < 10; ++rank) {
        float best = -INFINITY, bp = 0.0f;
        int be = 1 << 30;
        bool picked = false;
        if (own_phase) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float v = p[i] == p[i] ? p[i] : -INFINITY;
                if (!((taken >> i) & 1u) && (be == (1 << 30) || v > best)) { best = v; be = lane + 32 * i; bp = p[i]; }
            }
#pragma unroll
            for (int m = 16; m; m >>= 1) {
                const float ob = __shfl_xor_sync(0xffffffffu, best, m, 32);
                const int oe = __shfl_xor_sync(0xffffffffu, be, m, 32);
                const float op = __shfl_xor_sync(0xffffffffu, bp, m, 32);
                if (ob > best || (ob == best && oe < be)) { best = ob; be = oe; bp = op; }
            }
            if (rank < protect || (keep_p > 0.0f && bp >= keep_p)) picked = true;   // (the same on every lane)
            else { own_phase = false; best = -INFINITY; bp = 0.0f; be = 1 << 30; }
        }
        if (!picked) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float v = s[i] == s[i] ? s[i] : -INFINITY;
                if (!((taken >> i) & 1u) && (be == (1 << 30) || v > best)) { best = v; be = lane + 32 * i; bp = p[i]; }
            }
#pragma unroll
            for (int m = 16; m; m >>= 1) {
                const float ob = __shfl_xor_sync(0xffffffffu, best, m, 32);
                const int oe = __shfl_xor_sync(0xffffffffu, be, m, 32);
                const float op = __shfl_xor_sync(0xffffffffu, bp, m, 32);
                if (ob > best || (ob == best && oe < be)) { best = ob; be = oe; bp = op; }
            }
        }
        if ((be & 31) == lane) taken |= 1u << (be >> 5);
        if (lane == rank) { id[rank] = be; sel_p = bp; }
        sel_sum += bp;                                      // the same on every lane
    }
    // degenerate logits (NaN / all -inf) give NaN probabilities: the 10 picked share the weight equally
    const bool ok = sel_sum > 0.0f && sel_sum <= FLT_MAX;
    if (lane < 10) w[lane] = ok ? sel_p / sel_sum : 0.1f;
}

const float* g_prior_gate = nullptr;   // set once, before any window graph is captured (launch arguments are baked in)
int g_prior_protect = 0;
float g_prior_keep_p = 0.0f;

}  // namespace

void route_prior_configure(const float* gate, int protect, float keep_p) {
    g_prior_gate = gate;
    g_prior_protect = protect < 0 ? 0 : protect > 10 ? 10 : protect;
    g_prior_keep_p = keep_p > 0.0f ? keep_p : 0.0f;
}

void route_prior_top10(const float* logits, const int32_t* res_layer, float lambda, int32_t* ids, float* weights,
                             int n_tok, void* stream, float depth_acc, int row0) {
    if (n_tok <= 0) return;
    route_prior_k<<<n_tok, 32, 0, (cudaStream_t) stream>>>(logits, res_layer, lambda, ids, weights, depth_acc, row0,
                                                           g_prior_gate, g_prior_protect, g_prior_keep_p);
}

}  // namespace strata::kernels
