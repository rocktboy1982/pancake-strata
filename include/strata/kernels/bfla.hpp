// include/strata/kernels/bfla.hpp - BFLA block-sparse prompt attention for dense GQA layers (pancake, opt-in, lossy).
//
// BFLA: "Block-Filtered Long-Context Attention Mechanism", Chong Wu et al. (City University of Hong Kong lead),
// arXiv 2605.12193 (May 2026); reference code github.com/Alicewithrabbit/BFLA (Apache-2.0), read, not copied.
// Design and the reasons for each deviation: docs/BFLA_RDNA3.md.
//
// One call is the attention of one prompt chunk of one dense attention layer:
//   q16  [T][H][256]   FP16, queries at absolute positions p0 .. p0 + T - 1 (normed, rotated)
//   k16  [HKV][P][256] FP16, keys of cells 0 .. P - 1 with P = p0 + T (normed, rotated), compact (row stride 256)
//   v16  [HKV][P][256] FP16
//   out  [T][H][256]   FP32, softmax(q k^T / 16) v over the kept cells (not gated)
// The query head h uses KV head h / (H / HKV).
//
// Three kernels (HIP wave32 gfx11 only; every other build and device: the functions return false and launch nothing,
// so the caller keeps its own path):
//   features  phi (per token and KV group) and mu (per 64-key block and KV head): RMS-norm per head, the group's
//             query heads averaged, adjacent channel pairs summed to 128, RMS-norm again; mu = mean over the block
//   select    per selection tile (128/G tokens) and KV head: S_j = log sum_t exp(relu(phi_t . mu_j) / sqrt(128)), the
//             Ratio-K budget k = ceil(min(V, max(Kmin, min(Kmax, ceil(eta V)))) / 64) blocks, mandatory sink, own and
//             local blocks inside the quota, the rest by score (ties: smaller id); an ascending block list
//   attn      flash attention on the gfx11 matrix cores (v_wmma_f32_16x16x16_f16) over the listed blocks, causal mask
//             inline; FP16 operands, FP32 accumulation and softmax state, FP16 probabilities
// Dense mode (mode 2) skips features / select and walks every causal block: exact attention up to rounding.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

inline constexpr int kBflaBlock = 64;    ///< keys per block (paper b = 64)
inline constexpr int kBflaDim = 256;     ///< head dim the kernels take
inline constexpr int kBflaFeat = 128;    ///< indexer feature width (paper d_I = 128)
inline constexpr int kBflaTileRows = 128;   ///< token-head rows per selection tile (paper M = 128)

struct BflaConfig {
    int mode = 0;               ///< 0 off, 1 sparse (BFLA), 2 dense (the fused kernel over every causal block)
    int64_t min_len = 32768;    ///< a chunk uses BFLA only when p0 + T >= min_len
    double eta = 1.0 / 3.0;     ///< Ratio-K fraction of the visible length
    int64_t kmin = 2048;        ///< token budget floor
    int64_t kmax = 32768;       ///< token budget cap
    int sink = 2;               ///< mandatory blocks at the start
    int local = 4;              ///< mandatory blocks before a tile's first query block
    int64_t tail = -1;          ///< tokens at the end of the final chunk with full attention; -1: 1024 / G
};

/// STRATA_BFLA (0 / 1 / sparse / dense), STRATA_BFLA_MIN, _ETA, _KMIN, _KMAX, _SINK, _LOCAL, _TAIL; read once.
const BflaConfig& bfla_config();

/// Blocks of the Ratio-K budget for a visible length V (host and device use the same rule).
inline int64_t bfla_budget_blocks(int64_t V, const BflaConfig& c) {
    int64_t e = (int64_t) (c.eta * (double) V);
    if ((double) e < c.eta * (double) V) ++e;   // ceil
    int64_t k = e < c.kmax ? e : c.kmax;
    if (k < c.kmin) k = c.kmin;
    if (k > V) k = V;
    return (k + kBflaBlock - 1) / kBflaBlock;
}

/// The geometry the kernels take: head dim 256, G = H / HKV in {4, 8, 16}, and a gfx11 device.
bool bfla_supported(int H, int HKV, int HD);

/// Scratch bytes of a call with T queries and P = p0 + T cells.
size_t bfla_scratch_bytes(int64_t T, int64_t P, int H, int HKV);

/// The attention of one chunk (layout above).  `full_from`: the first absolute position whose selection tile takes
/// every block (the tail rule; INT64_MAX for none).  Returns false, launching nothing, when the geometry, device or
/// scratch does not fit.
bool bfla_prefill_attn(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t p0, int H, int HKV,
                       const BflaConfig& c, int64_t full_from, void* scratch, size_t scratch_bytes, float* out, void* stream);

/// Tests: where the last sparse call's lists live in `scratch` (device): ids [n_tiles][HKV][cap] ascending, counts
/// [n_tiles][HKV]; a tile is 128 / G tokens of the chunk.
struct BflaLists {
    const int32_t* ids = nullptr;
    const int32_t* counts = nullptr;
    int64_t cap = 0;
    int64_t n_tiles = 0;
    int64_t tile_tokens = 0;
};
BflaLists bfla_lists(const void* scratch, int64_t T, int64_t P, int H, int HKV);

/// Tests: the same lists, the attention kernel only (features and select skipped): the caller writes the lists.
bool bfla_attn_from_lists(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t p0, int H,
                          int HKV, const int32_t* ids, const int32_t* counts, int64_t cap, float* out, void* stream);

}  // namespace strata::kernels
