// include/strata/kernels/bfla.hpp - fused prompt attention for dense GQA layers on gfx11 (pancake, opt-in): exact over
// every causal block (STRATA_BFLA_DENSE=1), or BFLA block-sparse (STRATA_BFLA_SPARSE=1, LOSSY).
//
// BFLA: "Block-Filtered Long-Context Attention Mechanism", Chong Wu et al. (City University of Hong Kong lead),
// arXiv 2605.12193 (May 2026); reference code github.com/Alicewithrabbit/BFLA (Apache-2.0), read, not copied.
// Design: docs/BFLA_RDNA3.md; where the selection differs from the paper: pancake-docs/batch4/bfla-dense.md.
//
// Switches (read once by bfla_config(); with neither set nothing here runs and the caller's path is unchanged):
//   STRATA_BFLA_DENSE=1   the fused kernel over every causal block: exact attention (no cell dropped), the numerics class
//                         of the GEMM path it replaces (FP16 operands, FP32 accumulation, FP16 probabilities), not
//                         bitwise equal to it (other summation order)
//   STRATA_BFLA_SPARSE=1  BFLA's Ratio-K block selection, the rest of the prefix dropped: LOSSY, answers can change.
//                         Wins over STRATA_BFLA_DENSE when both are set.
//   STRATA_BFLA_MIN (32768: a chunk takes either path only when p0 + T reaches it), sparse only: STRATA_BFLA_ETA,
//   _KMIN, _KMAX, _SINK, _LOCAL, _TAIL.  The old single switch STRATA_BFLA (1 / sparse / dense) is ignored, with a note.
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
// Dense mode (mode 2) skips features / select and walks every causal block.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_BFLA_HD __host__ __device__
#else
#define STRATA_BFLA_HD
#endif

namespace strata::kernels {

inline constexpr int kBflaBlock = 64;    ///< keys per block (paper b = 64)
inline constexpr int kBflaDim = 256;     ///< head dim the kernels take
inline constexpr int kBflaFeat = 128;    ///< indexer feature width (paper d_I = 128)
inline constexpr int kBflaTileRows = 128;   ///< token-head rows per selection tile (paper M = 128)

struct BflaConfig {
    int mode = 0;               ///< 0 off, 1 sparse (BFLA, lossy), 2 dense (the fused kernel over every causal block, exact)
    bool legacy = false;        ///< the old STRATA_BFLA was set (ignored)
    int64_t min_len = 32768;    ///< a chunk uses the fused kernel only when p0 + T >= min_len
    double eta = 1.0 / 3.0;     ///< Ratio-K fraction of the visible length
    int64_t kmin = 2048;        ///< token budget floor
    int64_t kmax = 32768;       ///< token budget cap
    int sink = 2;               ///< mandatory blocks at the start
    int local = 4;              ///< mandatory blocks before a tile's first query block
    int64_t tail = -1;          ///< tokens at the end of the final chunk with full attention; -1: 1024 / G
};

/// The switch that selects `mode` (for log lines).
inline const char* bfla_switch_name(int mode) {
    return mode == 1 ? "STRATA_BFLA_SPARSE" : mode == 2 ? "STRATA_BFLA_DENSE" : "STRATA_BFLA (off)";
}

/// The switches read through `get` (std::getenv in bfla_config(); a table in the tests).  No output.
template <class Get>
BflaConfig bfla_config_from(Get get) {
    BflaConfig r;
    auto on = [&](const char* name) {
        const char* v = get(name);
        return v != nullptr && std::atoi(v) != 0;
    };
    r.mode = on("STRATA_BFLA_SPARSE") ? 1 : on("STRATA_BFLA_DENSE") ? 2 : 0;
    const char* old = get("STRATA_BFLA");
    r.legacy = old != nullptr && *old != '\0' && std::strcmp(old, "0") != 0;
    auto num = [&](const char* name, auto& dst) {
        if (const char* v = get(name); v != nullptr && *v != '\0') dst = (std::remove_reference_t<decltype(dst)>) std::atof(v);
    };
    num("STRATA_BFLA_MIN", r.min_len);
    num("STRATA_BFLA_ETA", r.eta);
    num("STRATA_BFLA_KMIN", r.kmin);
    num("STRATA_BFLA_KMAX", r.kmax);
    num("STRATA_BFLA_SINK", r.sink);
    num("STRATA_BFLA_LOCAL", r.local);
    num("STRATA_BFLA_TAIL", r.tail);
    if (r.eta <= 0.0 || r.eta > 1.0) r.eta = 1.0 / 3.0;
    if (r.kmin < kBflaBlock) r.kmin = kBflaBlock;
    if (r.kmax < r.kmin) r.kmax = r.kmin;
    if (r.sink < 1) r.sink = 1;   // block 0 keeps every row's softmax non-empty
    if (r.local < 0) r.local = 0;
    return r;
}

/// The switches from the environment, read once; the first call prints what is on (and a note for the old switch).
const BflaConfig& bfla_config();

/// Blocks of the Ratio-K budget for a visible length V (paper Eq. 9; host and device).
STRATA_BFLA_HD inline int64_t bfla_budget_blocks(int64_t V, double eta, int64_t kmin, int64_t kmax) {
    const double ev = eta * (double) V;
    int64_t e = (int64_t) ev;
    if ((double) e < ev) ++e;   // ceil
    int64_t k = e < kmax ? e : kmax;
    if (k < kmin) k = kmin;
    if (k > V) k = V;
    return (k + kBflaBlock - 1) / kBflaBlock;
}
inline int64_t bfla_budget_blocks(int64_t V, const BflaConfig& c) { return bfla_budget_blocks(V, c.eta, c.kmin, c.kmax); }

/// One selection tile's plan (the selection kernel and the host tests): queries at absolute positions qa .. qb.
struct BflaTilePlan {
    int n_vis = 0;        ///< visible blocks 0 .. n_vis - 1 (the causal blocks of qb)
    int sinkc = 0;        ///< mandatory sink blocks [0, sinkc)
    int lo_b = 0;         ///< mandatory own and local blocks [lo_b, n_vis); the candidates are [sinkc, lo_b)
    int64_t free_q = 0;   ///< budget blocks left for the candidates (<= 0: the mandatory blocks only)
    bool all = false;     ///< every visible block: the tail rule (qb >= full_from), or the budget covers them
};
STRATA_BFLA_HD inline BflaTilePlan bfla_tile_plan(int64_t qa, int64_t qb, int64_t full_from, double eta, int64_t kmin,
                                                 int64_t kmax, int sink, int local) {
    BflaTilePlan p;
    p.n_vis = (int) (qb / kBflaBlock) + 1;
    const int64_t k = bfla_budget_blocks(qb + 1, eta, kmin, kmax);
    p.sinkc = sink < p.n_vis ? sink : p.n_vis;
    const int lb = (int) (qa / kBflaBlock) - local;
    p.lo_b = lb > p.sinkc ? lb : p.sinkc;
    const int nc = p.lo_b - p.sinkc;
    p.free_q = k - (int64_t) (p.n_vis - nc);
    p.all = qb >= full_from || k >= p.n_vis || p.free_q >= nc;
    return p;
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

/// Tests / bench: the attention kernel's Q from LDS (0) or global memory (1, = STRATA_BFLA_QGLOBAL=1); -1 back to the env.
void bfla_set_q_global(int on);

}  // namespace strata::kernels
