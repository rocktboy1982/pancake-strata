// include/strata/kernels/causal_order.hpp - LEADS.md Part L L8: the causal block order of the q36 WMMA flash attention (STRATA_Q36_FA_HEAVY_FIRST).
// Host only.  k_attn_fa_wmma (src/qwen35/q36_engine.cu) runs one block per 64 rows of a KV head; block bx scans c_end(bx) cells, which grows
// with bx under the causal mask, so the LAST blocks are the longest.  The hardware hands blocks out in linear order (x fastest), so with
// the natural order the longest blocks start last and form a tail where most of the card idles.  Heaviest first reverses x:
//     bx = gridDim.x - 1 - blockIdx.x
// a bijection of [0, gridDim.x), so every block still computes exactly the same rows (the output is bit for bit the natural order's).
// The kernel carries the same one-line formula; this header has it and a scheduling model, so the test can check the bijection and show the
// makespan effect without a GPU.  The model is a greedy list scheduler (a block goes to the first compute unit that frees up), the
// usual first-order model of a GPU's workgroup dispatcher; it is NOT a measurement.
#pragma once
#include <algorithm>
#include <cstdint>
#include <queue>
#include <vector>

namespace strata::kernels::causal {

inline int block_index(int linear, int grid_x, bool heavy_first) { return heavy_first ? grid_x - 1 - linear : linear; }

/// cells the block `bx` scans: the kernel's `c_end` (rows R = T G, block rows FA_ROWS, p0 cached cells before the chunk, P = p0 + T cells)
inline int64_t cells_scanned(int bx, int64_t T, int G, int64_t P, int64_t p0, int rows_per_block = 64, int tile = 16) {
    const int64_t R = T * G, r0 = (int64_t) bx * rows_per_block;
    const int64_t r_last = std::min<int64_t>(R, r0 + rows_per_block) - 1;
    const int64_t c_end = std::min<int64_t>(P, p0 + r_last / G + 1);
    return (c_end + tile - 1) / tile * tile;   // whole 16-cell tiles
}

/// makespan of the blocks of one kv-head row of the grid on `units` compute units, blocks dispatched in linear order after `block_index`
inline int64_t makespan(int grid_x, int64_t T, int G, int64_t P, int64_t p0, int units, bool heavy_first) {
    std::priority_queue<int64_t, std::vector<int64_t>, std::greater<int64_t>> free_at;
    for (int u = 0; u < units; ++u) free_at.push(0);
    int64_t end = 0;
    for (int lin = 0; lin < grid_x; ++lin) {
        const int bx = block_index(lin, grid_x, heavy_first);
        const int64_t t = free_at.top();
        free_at.pop();
        const int64_t done = t + cells_scanned(bx, T, G, P, p0);
        free_at.push(done);
        end = std::max(end, done);
    }
    return end;
}

}  // namespace strata::kernels::causal
