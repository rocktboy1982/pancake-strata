// include/strata/kernels/fold_merge.hpp - LEADS.md Part L L9: the protocol that folds the split-KV merge into the LAST chunk block
// (STRATA_ATTN_MERGE_FOLD design).  Host only.  The decode attention (src/kernels/cuda/qsa_decode_attn.cu) runs one block per 64 cells of a
// KV head's context (attn_chunk_kernel, ~10 variants), each writing a partial (m, l, acc[G][HD]) to scratch, and then a second launch
// (attn_merge_kernel / _bound_ / _v2_) combines the partials of a head in CHUNK ORDER.  The fold removes the second launch (about 2.2 us
// + 3.1 us of gap a layer in the authors' measurement, paper, not ours):
//
//   chunk block c of group q:  write the partial;  release fence;  ticket = atomicAdd(&counter[q], 1);  acquire fence;
//                              if (ticket == n_chunks - 1) { merge the group's partials in chunk order; counter[q] = 0; }
//
// Why it is safe, and the rules a kernel must keep (each is a test below):
//   R1 every chunk block takes a ticket, INCLUDING the ones that exit early because they hold no cells (they write m = -FLT_MAX, l = 0 first):
//      a block that skips the ticket leaves the count short and the merge never runs;
//   R2 the merge reads the partials in chunk index order, never in arrival order: the sums are then the separate merge's, bit for bit;
//   R3 the last block resets the counter itself, so a graph replay (the same launch again, no memset node) starts from 0 -- this needs the
//      counter to be 0 at the first launch (zeroed once per scratch owner, a hipMemsetAsync at allocation) and every launch to run to the end:
//      an aborted launch leaves a dirty counter, so the owner re-zeroes after any error path;
//   R4 one counter per (query, KV head) group and per scratch owner: the target decode and the MTP drafter, and the slots of a batch, must not
//      share a counter (their scratch strides already differ, `scratch_stride` of attn_merge_kernel);
//   R5 the ticket counts the blocks of ONE launch: n_chunks is the launch's grid, not the capacity (the _bound_ variants stop at the used chunks:
//      the fold counts all of the grid and lets the early-exit blocks arrive, R1).
// The memory-ordering part (the release / acquire pairing of the partials with the ticket) cannot be shown by a x86 host test; it is the standard
// "last block done" pattern (CUDA samples threadFenceReduction), and the probe tools/pancake/probes/attn_fold.hip.cpp checks it on the GPU with
// a bitwise comparison against the separate merge over many launches.
#pragma once
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <vector>

namespace strata::kernels::foldmerge {

struct Group {   // the partials of one (query, KV head): n_chunks x (m, l, acc[HD]) for one query head (the model keeps one head, G = 1)
    int n_chunks = 0, hd = 0;
    std::vector<float> m, l, acc;   // [n_chunks], [n_chunks], [n_chunks][hd]
    void init(int nc, int d) { n_chunks = nc; hd = d; m.assign(nc, -FLT_MAX); l.assign(nc, 0.f); acc.assign((size_t) nc * d, 0.f); }
};

/// attn_merge_kernel's arithmetic for one head: M = max m; L = sum fma(l_c, w_c, L); acc likewise, chunks in index order, skip m == -FLT_MAX.
/// `w` is a function of m - M (the device uses __expf; the model takes any pure function so both sides of a test use the same one).
template <class Exp>
inline void merge_ref(const Group& g, Exp w_of, float* out) {
    float M = -FLT_MAX;
    for (int c = 0; c < g.n_chunks; ++c) M = std::fmax(M, g.m[c]);
    float L = 0.f;
    std::vector<float> acc(g.hd, 0.f);
    for (int c = 0; c < g.n_chunks; ++c) {
        if (g.m[c] == -FLT_MAX) continue;
        const float w = w_of(g.m[c] - M);
        L = std::fmaf(g.l[c], w, L);
        for (int d = 0; d < g.hd; ++d) acc[d] = std::fmaf(g.acc[(size_t) c * g.hd + d], w, acc[d]);
    }
    for (int d = 0; d < g.hd; ++d) out[d] = L > 0.f ? acc[d] / L : 0.f;
}

/// the ticket: returns true for the block that arrived last (it must merge, then reset).  acq_rel = the release of this block's partial and the
/// acquire of everybody else's that arrived before it.
inline bool arrive(std::atomic<int>& counter, int n_chunks) { return counter.fetch_add(1, std::memory_order_acq_rel) == n_chunks - 1; }
inline void reset(std::atomic<int>& counter) { counter.store(0, std::memory_order_relaxed); }

}  // namespace strata::kernels::foldmerge
