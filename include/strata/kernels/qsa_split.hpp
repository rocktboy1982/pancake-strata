// include/strata/kernels/qsa_split.hpp - STRATA_QSA_SPLIT_GRID (LEADS.md Part O O9): the host rule of the decode attention's persistent-style grid.
//
// The caminus authors' note ("the grid follows the card": make the number of attention splits equal to the workgroups that run at once, and cut
// splits in whole 64-position chunks; one extra round of tasks costs a whole task) read for OUR kernel.  Ours already cuts the selected cells in
// 64-cell chunks, one block per (chunk, KV head, query) with the grid (n_chunks, n_head_kv, n_q), so nothing needs re-chunking; the lever is the
// NUMBER OF BLOCKS: with `target` workgroups wanted in all, each (KV head, query) gets `blocks = clamp(target / (n_head_kv n_q), 1, n_chunks)` blocks and
// block b visits the chunks b, b + blocks, b + 2 blocks, ... in order (attn_chunk_split_kernel).  Every chunk is still computed by the unchanged
// attn_chunk_body and writes its own partial slot, so the partials, the merge and the output are bit for bit the default's: only the schedule moves.
// Pure functions, no GPU: tested by src/kernels/qsa_split_test.cpp (every chunk exactly once, in order inside a block, the default untouched).
#pragma once

#include <cstdint>
#include <cstdlib>

namespace strata::kernels {

// the blocks per (KV head, query) for `target` workgroups in all; 0 = no split (the default grid: one block per chunk)
inline int qsa_split_blocks(int64_t target, int64_t n_head_kv, int64_t n_q, int64_t n_chunks) {
    if (target <= 0 || n_head_kv <= 0 || n_q <= 0 || n_chunks <= 1) return 0;
    int64_t b = target / (n_head_kv * n_q);
    if (b < 1) b = 1;
    return b >= n_chunks ? 0 : int(b);   // as many blocks as chunks is the default grid
}

// the value of STRATA_QSA_SPLIT_GRID: a positive integer; unset, empty, 0 or anything else means off
inline int64_t qsa_split_target(const char* env) {
    if (!env || !*env) return 0;
    char* end = nullptr;
    const long long v = std::strtoll(env, &end, 10);
    return (end && *end == '\0' && v > 0) ? int64_t(v) : 0;
}

// the chunk number block `b` of `blocks` visits at step `i` (the loop `for (c = b; c < n_chunks; c += blocks)`), -1 past the end
inline int qsa_split_chunk(int b, int blocks, int n_chunks, int i) {
    const int64_t c = int64_t(b) + int64_t(i) * blocks;
    return c < n_chunks ? int(c) : -1;
}

}  // namespace strata::kernels
