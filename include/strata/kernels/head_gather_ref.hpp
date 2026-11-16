// include/strata/kernels/head_gather_ref.hpp - STRATA_HEAD_SHORTLIST_GATHER (MOD_PLAN.md section 6 item 38, LOSSY like
// STRATA_HEAD_SHORTLIST): the host reference of the gathered-rows head.  The device kernel (head_fmt.cu head_gather_kernel)
// computes a head row only when its cluster is kept by at least one column of the chunk (the UNION of the columns' kept
// clusters), and writes -inf to every logit whose cluster the column did not keep, so its output equals, bit for bit, the full
// head GEMV followed by head_shortlist_mask's -inf over the unkept clusters.  This header states that contract on the host:
// `gathered_logits` computes only the union's rows, `masked_full_logits` is the full head then the mask, and the test
// (src/core/head_gather_test.cpp) checks they are the same floats, and the same pick, on random heads.  The
// selection rule is include/strata/program/head_shortlist.hpp's (shortlist_dot, shortlist_select).
// Row blocks: the kernel's block of `wpb` consecutive rows does nothing (and loads no activations) when none of its rows is in
// the union; `active_blocks` counts the blocks that work, the share of the head the kernel reads is rows_in_union / vocab
// (weights of 1360 contiguous bytes a row for the daily IQ4_XS head, so a gather costs no extra transactions per row).
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace strata::kernels {

/// union_[c] = 1 when any of the `ncols` columns keeps cluster c.  keep: ncols x k flags (column j at keep + j * k).
inline std::vector<uint8_t> head_gather_union(const uint8_t* keep, int ncols, int k) {
    std::vector<uint8_t> u((size_t) k, 0);
    for (int j = 0; j < ncols; ++j)
        for (int c = 0; c < k; ++c)
            if (keep[(size_t) j * k + c]) u[(size_t) c] = 1;
    return u;
}

/// Rows whose cluster is in the union.
inline int64_t head_gather_rows(const int32_t* cluster, int vocab, const std::vector<uint8_t>& u) {
    int64_t n = 0;
    for (int v = 0; v < vocab; ++v) n += u[(size_t) cluster[v]] ? 1 : 0;
    return n;
}

/// Blocks of `wpb` consecutive rows that hold at least one union row (the kernel's working blocks).
inline int64_t head_gather_active_blocks(const int32_t* cluster, int vocab, const std::vector<uint8_t>& u, int wpb) {
    int64_t n = 0;
    for (int b = 0; b * wpb < vocab; ++b) {
        bool any = false;
        for (int r = b * wpb; r < vocab && r < (b + 1) * wpb; ++r) any = any || u[(size_t) cluster[r]];
        n += any ? 1 : 0;
    }
    return n;
}

/// The dot of one head row with a hidden row, a fixed order (the stand-in for the head GEMV's exact arithmetic: the contract
/// is only that the gathered and the full head use the SAME dot).
inline float head_ref_dot(const float* row, const float* h, int d) {
    float s = 0.0f;
    for (int i = 0; i < d; ++i) s += row[i] * h[i];
    return s;
}

/// The full head (every row, every column), then -inf over the logits outside each column's kept clusters
/// (head_shortlist_mask's effect).  rows: vocab x d; hidden: ncols x d; out: ncols x vocab.
inline void masked_full_logits(const float* rows, int vocab, int d, const float* hidden, int ncols, const int32_t* cluster,
                               const uint8_t* keep, int k, float* out) {
    const float ninf = -std::numeric_limits<float>::infinity();
    for (int j = 0; j < ncols; ++j)
        for (int v = 0; v < vocab; ++v) {
            const float s = head_ref_dot(rows + (size_t) v * d, hidden + (size_t) j * d, d);
            out[(size_t) j * vocab + v] = keep[(size_t) j * k + cluster[v]] ? s : ninf;
        }
}

/// The gathered head: a row's dot is computed only when its cluster is in the union (once per column that keeps it), every
/// other logit is -inf.  Returns the number of row dots computed (columns x union rows, counting a row once per column that
/// keeps it).
inline int64_t gathered_logits(const float* rows, int vocab, int d, const float* hidden, int ncols, const int32_t* cluster,
                               const uint8_t* keep, int k, float* out) {
    const float ninf = -std::numeric_limits<float>::infinity();
    const std::vector<uint8_t> u = head_gather_union(keep, ncols, k);
    int64_t dots = 0;
    for (int v = 0; v < vocab; ++v) {
        const bool in_union = u[(size_t) cluster[v]] != 0;
        for (int j = 0; j < ncols; ++j) {
            float s = ninf;
            if (in_union && keep[(size_t) j * k + cluster[v]]) {
                s = head_ref_dot(rows + (size_t) v * d, hidden + (size_t) j * d, d);
                ++dots;
            }
            out[(size_t) j * vocab + v] = s;
        }
    }
    return dots;
}

}  // namespace strata::kernels
