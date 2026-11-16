// include/strata/kernels/subfit_apply.hpp - LEADS.md HotRod substitute fit (sangharshadhyeta/Strata-HotRod 9514498fe9 / aa797a1065 /
// b7bd02f5bf / d3b34a33a1), stage 2, steps 2-5: the device side of the per-pair corrections.
//
// Pieces:
//   - SubfitRoute: what the extended stand-in kernel (native_router.cu route_resident_ex_k) needs per call: the group's orig / flag rows
//     (written by the route kernel, consumed by subfit_apply), the layer's LUT (e*E+s -> slot or -1), the per-slot residual, the mode
//     (1 dynamic, 2 all, 3 flip), and the shadow-sample list.  All null / 0 = no wiring: the kernel does exactly what it did before.
//   - subfit_apply: one launch per layer and row group, after the experts and their join, before the combine (verify.cpp post()).  For
//     each row and slot k with orig >= 0, LUT[l][orig*E+ids] = slot >= 0 and (mode all or the slot's flag): z = A_l x_row (x = the layer's
//     MoE input), parts[row,k] = c * parts[row,k] + B z.  It also clears orig and flag for the rows it reads (they are consumed).  The
//     launch is made whenever the mode is on (the window graph is captured once); the device count nfit[l] skips a layer with no fit.
//   - subfit_tables_alloc: the device tables (LUT, A, B, c, resid, nfit).  The LUT and B always in pinned host memory the GPU reads over
//     PCIe (HotRod d3b34a33a1: in VRAM they took the margin the window graphs need), the small ones in VRAM when they fit, else pinned
//     host memory too (`host` says which).  Writes are made between windows.
//
// Layouts (E = the model's experts per layer, from the geometry): parts [rows][K][N] float, raw expert outputs (before the router
// weight); x [rows][N] float (the layer's MoE input); ids [rows][K] int32; orig [rows][K] int16 (-1: not substituted); flag [rows][K]
// uint8; lut [E*E] int16 for the layer; A [r][N] fp16; B [slots][N][r] fp16 (B[j*r+m]); c [slots] float; resid [slots] float; nfit [1].
#pragma once

#include "strata/core/subfit_rule.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// The extended stand-in kernel's view of the learner (per call; nullptr members = off).
struct SubfitRoute {
    int16_t* orig = nullptr;        ///< [rows][K] (this call's row base): the router's pick of a substituted slot, -1 otherwise
    uint8_t* flag = nullptr;        ///< [rows][K] (this call's row base): 1 = the move is corrected
    const int16_t* lut = nullptr;   ///< this layer's LUT [E*E]: e*E+s -> slot, or -1
    const float* resid = nullptr;   ///< [slots]: the held-out residual sqrt(E1/E0) of each slot's pair
    int mode = 0;                   ///< strata::core::subfit_rule::kModeDynamic (1), kModeAll (2) or kModeFlip (3); 0 = off
    int n_experts = 0;              ///< E (the geometry's n_expert): the LUT's side
    const int16_t* elut = nullptr;  ///< the fallback ladder: this layer's (layer, expert) slots [E] (used when the pair's LUT is -1)
    const int16_t* llut = nullptr;  ///< ... then this layer's own slot [1]
    unsigned long long* stats = nullptr;   ///< device [4]: corrected moves, substituted moves, refusals left (past the bound, no fit),
                                           ///< corrected moves in a held (flip re-check) row
    // shadow samples (the learner on): every substitution of a row into list [rows][list_cap][4] (layer, rank, true pick, stand-in),
    // reservoir-sampled past list_cap; list_n [rows] counts them (host-mapped; the host resets it each decode window).  nullptr: no list.
    int32_t* list = nullptr;
    uint32_t* list_n = nullptr;
    int list_cap = 0;
    uint32_t salt = 0;              ///< mixed into the reservoir's hash (the window count)
    int layer = 0;                  ///< this launch's layer (the list's and rec's layer field)
    int32_t* rec = nullptr;         ///< STRATA_SUBFIT_LOG: [rows] host-mapped, the row's first layer with a move (-1 none; the probe's clean test)
};

struct SubfitApplyArgs {
    float* parts = nullptr;             ///< [n][K][N], this call's rows
    const float* x = nullptr;           ///< [n][N] the layer's MoE input of this call's rows
    const int32_t* ids = nullptr;       ///< [n][K] the picks after the substitutions
    int16_t* orig = nullptr;            ///< [n][K] consumed (reset to -1)
    uint8_t* flag = nullptr;            ///< [n][K] consumed (reset to 0)
    const int16_t* lut = nullptr;       ///< this layer's LUT
    const uint16_t* A = nullptr;        ///< this layer's A (r x N, fp16)
    const uint16_t* B = nullptr;        ///< all slots' B (slots x N x r, fp16)
    const float* c = nullptr;           ///< [slots]
    const int32_t* nfit = nullptr;      ///< this layer's fitted count (device): 0 = nothing to apply
    const int16_t* elut = nullptr;      ///< the ladder: this layer's expert slots [E], then its layer slot [1] (nullable)
    const int16_t* llut = nullptr;
    int r = 16, N = 2560, K = 10, n = 0;
    int E = 512;                        ///< the model's experts per layer (the LUT's side)
    int mode = 1;                       ///< 1 dynamic / 3 flip (flagged moves), 2 all (every fitted move)
};

/// Launch for this layer's rows on `stream` (no host sync).  Null members or n <= 0: nothing is launched.
void subfit_apply(const SubfitApplyArgs& a, void* stream);

/// Device tables for `slots` slots of N x r, `layers` layers of `experts` experts.  The LUT is set to -1, the others zero.  Returns false
/// (nothing kept) when one allocation fails.
struct SubfitTables {
    int16_t* lut = nullptr;      ///< subfit_learn::table_size(layers, experts): [layers][E][E] pairs, [layers][E] experts, [layers] layers
    uint16_t* A = nullptr;       ///< [layers][r][N]
    uint16_t* B = nullptr;       ///< [slots][N][r]
    float* c = nullptr;          ///< [slots]
    float* resid = nullptr;      ///< [slots]
    int32_t* nfit = nullptr;     ///< [layers]
    int layers = 0, slots = 0, N = 0, r = 0, experts = 0;
    void* host[6] = {};       ///< per table (lut, A, B, c, resid, nfit): its pinned host block when it is in host memory, else nullptr
    size_t host_bytes = 0;    ///< bytes placed in pinned host memory
};
bool subfit_tables_alloc(SubfitTables& t, int layers, int slots, int N, int r, int experts);
void subfit_tables_free(SubfitTables& t);

}  // namespace strata::kernels
