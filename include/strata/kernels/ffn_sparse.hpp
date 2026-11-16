// include/strata/kernels/ffn_sparse.hpp - activation sparsity on the FFN intermediate, the transposed down read, and a
// low-rank correction of a requantized weight (pancake, all opt-in; LEADS.md rows "Activation sparsity masks",
// "BlockFFN union-over-window sparsity column", "Transposed n_ff-major down layout (PowerInfer)", "LQER/QERA low-rank
// error correction").  Status and measurements: MOD_PLAN.md / TEST_LEDGER.md.
//
// 1. TEAL (Liu et al., "Training-Free Activation Sparsity in Large Language Models", arXiv 2408.14690): an entry of a
//    hidden state whose magnitude is below a per-tensor threshold is set to zero, so the matching weight column is not
//    needed.  First version: one threshold (STRATA_TEAL=<tau>) on the SwiGLU product that feeds the shared expert's
//    down projection, in the verify window (shared_expert_multi).  LOSSY.
// 2. BlockFFN (Song et al., "BlockFFN: Towards End-Side Acceleration-Friendly Mixture-of-Experts with Chunk-Level
//    Activation Sparsity", arXiv 2507.08771): over a chunk of tokens (here: the rows of one verify window) a column is
//    skipped only when it is inactive in EVERY row, and the rows keep their own values in the union's columns
//    (STRATA_TEAL_UNION=1 with STRATA_TEAL).  Less sparse than per-row masks, but the skip is shared by the window.
// 3. PowerInfer (Song et al., "PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU", SOSP 2024):
//    the down weight stored n_ff-major (row i = intermediate column i, n_embd contiguous values), so a skipped column
//    skips contiguous bytes.  STRATA_DOWN_T=1: an FP16 transposed copy of each ffn_down_shexp is made at load
//    (native_dense.cpp) and down_t_gemv reads only the listed columns.  Not bitwise the quantized MMVQ (the activation is
//    not quantized to q8_1 and the weight is the dequantized FP16 image): KeyIfSet.
// 4. LQER (Zhang et al., "LQER: Low-Rank Quantization Error Reconstruction for LLMs", ICML 2024) / QERA (Zhang et al.,
//    arXiv 2410.06040): W ~= Wq + A B with A B a rank-r approximation of the quantization error W - Wq; y = Wq x + A (B x).
//    Only where the engine itself quantizes a full-precision weight (a BF16 projection served as Q8_0, native_dense.cpp
//    #4741): the GGUF's own quantized tensors have no full-precision original here.  STRATA_LQER=<rank> (0/unset: off),
//    STRATA_LQER_MATCH=<substring of the tensor name> (unset: every such tensor).  The factorization: lqer.hpp (host).
//    The correction is applied inside native_mmvq for a registered weight (lqer_register): the activation is the q8_1
//    image the MMVQ reads, dequantized.  Calls that bypass native_mmvq (pair / interleaved / RDNA kernels) are not
//    corrected in this first version.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::kernels {

// ---- switches (read once)
inline float teal_threshold() {   ///< STRATA_TEAL=<tau> (> 0: on)
    static const float v = [] { const char* e = std::getenv("STRATA_TEAL"); return e != nullptr && *e ? (float) std::atof(e) : 0.0f; }();
    return v > 0.0f ? v : 0.0f;
}
inline bool teal_union() {        ///< STRATA_TEAL_UNION=1: the window's rows share one column mask (BlockFFN)
    static const bool v = [] { const char* e = std::getenv("STRATA_TEAL_UNION"); return e != nullptr && e[0] == '1'; }();
    return v;
}
inline bool down_t_on() {         ///< STRATA_DOWN_T=1: transposed FP16 shared-expert down copies, masked read
    static const bool v = [] { const char* e = std::getenv("STRATA_DOWN_T"); return e != nullptr && e[0] == '1'; }();
    return v;
}
inline int lqer_rank_env() {      ///< STRATA_LQER=<rank>
    static const int v = [] { const char* e = std::getenv("STRATA_LQER"); const int r = e != nullptr ? std::atoi(e) : 0; return r > 0 && r <= 256 ? r : 0; }();
    return v;
}
inline const char* lqer_match_env() {   ///< STRATA_LQER_MATCH=<substring>
    static const char* v = std::getenv("STRATA_LQER_MATCH");
    return v;
}

/// The column test both the kernel and the host reference use: a column (or entry) is ACTIVE unless its magnitude is
/// below tau.  A NaN is active (never hidden by the mask).
inline bool teal_active_host(float x, float tau) { return !(std::fabs(x) < tau); }

/// h (n_tok, n_ff) f32, in place.  union_rows false: every entry below tau becomes 0 (TEAL).  union_rows true: a column
/// active in no row becomes 0 in every row, the other columns keep every row's value (BlockFFN).  When `idx` is not null
/// the active columns (active in at least one row) are written to idx[0..*n_idx) in ascending order.  One block,
/// n_tok <= 16, graph-capturable.  tau 0 keeps every value and lists every column.
void teal_sparsify(float* h, int n_ff, int n_tok, float tau, bool union_rows, int32_t* idx, int32_t* n_idx, void* stream);

/// y (n_tok, n_out) = for each row t: sum over k < *n_idx of wt[idx[k]][o] * h[t][idx[k]], wt (n_ff, n_out) FP16 bits
/// (the transposed down), summed in k order with fmaf.  n_tok <= 16.  Graph-capturable.
void down_t_gemv(const uint16_t* wt, const float* h, const int32_t* idx, const int32_t* n_idx, float* y, int n_ff, int n_out,
                 int n_tok, void* stream);

/// y (ncols, n_out) += A (B xq) for a q8_1 activation (ncols columns of n_in / 32 blocks of {half d, half s, int8 qs[32]}),
/// A (n_out, r) f32, B (r, n_in) f32, tmp (ncols, r) scratch.  Graph-capturable.
void lowrank_q81_apply(const float* A, const float* B, int r, const void* x_q8_1, float* tmp, float* y, int n_in, int n_out,
                       int ncols, void* stream);

/// The registry native_mmvq consults (keyed by the weight's device pointer).  A, B device copies the caller keeps alive.
/// Allocates the per-entry scratch (16 columns).  Returns false on an allocation failure (nothing registered).
bool lqer_register(const void* weights, const float* A, const float* B, int r, int n_in, int n_out);
bool lqer_any();
/// When `weights` is registered: y += A (B x) with the matching shapes (else nothing).
void lqer_apply_if(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

// ---- host references (the parity tests and the CPU test use them)

inline void teal_sparsify_host(float* h, int n_ff, int n_tok, float tau, bool union_rows, std::vector<int32_t>* idx) {
    if (idx) idx->clear();
    for (int c = 0; c < n_ff; ++c) {
        bool any = false;
        for (int t = 0; t < n_tok; ++t) any = any || teal_active_host(h[(size_t) t * n_ff + c], tau);
        for (int t = 0; t < n_tok; ++t) {
            float& x = h[(size_t) t * n_ff + c];
            if (union_rows ? !any : !teal_active_host(x, tau)) x = 0.0f;
        }
        if (any && idx) idx->push_back(c);
    }
}

}  // namespace strata::kernels
