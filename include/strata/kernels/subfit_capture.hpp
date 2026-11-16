// include/strata/kernels/subfit_capture.hpp - LEADS.md HotRod substitute fit (sangharshadhyeta/Strata-HotRod 870e32b4b3 / fb9b442422): the
// device side of the probe samples (STRATA_SUBFIT_LOG=1) and of the shadow samples (STRATA_SUBFIT_LEARN=1 with STRATA_FRAGILE_RECHECK).
//
// Predicated launches, all graph-safe (the gate is a host-mapped int array the host sets between windows; with the switches off none is
// launched).  gate[3] == 1: a probe window; gate[3] == 2: the exact successor of a STRATA_FRAGILE_RECHECK cut whose row listed its
// substitutions (its row 0 is the held, re-decided row).
//   - subfit_probe_pre / subfit_probe_post: in a probe window for row 0 only: the router's picks before the stand-in kernel (pre), then
//     after it: flag[k] = the pick is still the router's (not replaced), resident, and row 0 had no stand-in before this layer (clean);
//     x = row 0's layer input; w = the router weights of row 0.
//   - subfit_probe_y: the raw expert outputs of row 0 (parts, [K][N]) after the experts, before the combine (the host reads them after
//     the window).
//   - subfit_shadow_xi / subfit_shadow_y: in the exact successor, row 0's picks, input and raw expert outputs of each layer.
// PORT NOTE: HotRod's subfit_capture_x (the stand-in row's input for its tag-1 record) is not needed here: the tag-1 records come from
// the stand-in learner's CPU replay, which has the input already (verify.cpp StandinLearner).
#pragma once

#include <cstdint>

namespace strata::kernels {

/// Device buffers of the probe (allocated only with the logger).  L = layers, K = routed experts per row, N = n_embd.
struct SubfitProbeBufs {
    float* x = nullptr;        ///< [L][N]: row 0's layer input at each layer
    float* y = nullptr;        ///< [L][K][N]: row 0's raw expert outputs (before the router weight)
    float* w = nullptr;        ///< [L][K]: row 0's router weights
    int32_t* flag = nullptr;   ///< [L][K]: 1 = resident, exact pick of row 0 at a clean row
    int32_t* pre = nullptr;    ///< [L][K]: row 0's router picks before the stand-in kernel
    int32_t* clean = nullptr;  ///< [L]: 1 = row 0 had no stand-in before this layer
};

/// Row 0's picks before the stand-in kernel: pre[layer][k] = ids[k]; clean[layer] = 1 when rec[0] (row 0's first stand-in layer) is -1;
/// flag[layer][*] = 0 (a layer whose post did not run stays unsampled).
void subfit_probe_pre(const int32_t* ids, const int32_t* rec, const int32_t* gate, int layer, int K, int32_t* pre, int32_t* clean,
                      int32_t* flag, void* stream);
/// After the stand-in kernel: flag, w and x of row 0 at this layer (see the header comment).  res: the layer's d_res [n_expert].
void subfit_probe_post(const int32_t* ids, const float* w, const int32_t* res, const int32_t* pre, const int32_t* clean,
                       const int32_t* gate, const float* mixed0, int layer, int K, int64_t N, float* x, float* w_out, int32_t* flag,
                       void* stream);
/// Row 0's K raw expert outputs (parts: [K][N]) into y (the layer's [K][N] slab), while gate[3] == 1.
void subfit_probe_y(const float* parts, const int32_t* gate, int64_t K_N, float* y, void* stream);

/// Shadow samples: while gate[3] == 2, row 0's picks (ids [K]) and MoE input (mixed0 [N]) of this layer into ids_out[layer][K] and
/// x_out[layer][N] ...
void subfit_shadow_xi(const int32_t* ids, const float* mixed0, const int32_t* gate, int layer, int K, int64_t N, int32_t* ids_out,
                      float* x_out, void* stream);
/// ... and its K raw expert outputs (parts [K][N]) into y_out[layer][K][N].
void subfit_shadow_y(const float* parts, const int32_t* gate, int layer, int64_t K_N, float* y_out, void* stream);

}  // namespace strata::kernels
