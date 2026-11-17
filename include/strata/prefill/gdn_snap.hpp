// include/strata/prefill/gdn_snap.hpp - STRATA_GDN_SNAP_IN_CHUNK (LEADS.md Part O O6): the host rules of in-chunk recurrent-state snapshots.
//
// The caminus authors' note (the header of their prefill_gdn.hip, AGPL, idea only): the gated-delta-net prompt kernel writes state snapshots at a list of
// positions INSIDE a chunk, so a checkpoint never forces a chunk boundary.  Ours cuts a prompt into chunks at every checkpoint position (root, last user
// turn, STRATA_CKPT_STRIDE), and a chunk of `stream_min` (4096) tokens or more streams every expert the GPU does not hold, so each extra cut re-streams them.
// This header is the pure part: the chunk plan with and without in-chunk snapshots and the snapshot buffer layout; the device part is
// gdn_rec_cols_snap_kernel (src/prefill/kernels.cu, checked by src/prefill/gdn_snap_parity.cu on a card) and the host reference of its arithmetic and layout is
// src/prefill/gdn_snap_test.cpp.  The engine's prompt loop does NOT call it yet: the other layers of a snapshot position (the attention K/V rows and the conv
// history) are addressable at any position, but the checkpoint writer takes its state from the end of a chunk (MOD_PLAN.md section 6).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace strata::prefill {

// STRATA_GDN_SNAP_IN_CHUNK=1: the prompt loop plans chunks with in-chunk snapshots (gdn_chunk_plan(..., true)); read here, NOT acted on by the engine yet
inline bool gdn_snap_in_chunk_enabled(const char* env) { return env != nullptr && env[0] == '1' && env[1] == '\0'; }
inline bool gdn_snap_in_chunk_from_env() { return gdn_snap_in_chunk_enabled(std::getenv("STRATA_GDN_SNAP_IN_CHUNK")); }

struct GdnChunk {
    int64_t start = 0, len = 0;
    std::vector<int64_t> snaps;   // positions (absolute: the state AFTER that many tokens of the prompt) strictly inside this chunk
};

// `cuts`: the positions a checkpoint is wanted at (a position p means "the state after p tokens").  `chunk`: the chunk length.  Without in-chunk snapshots a
// cut inside a chunk ends the chunk there (one more chunk, one more expert stream); with them the chunks are the plain cover of [0, n) and the cut becomes a
// snapshot of its chunk (a cut on a chunk end needs none: the end state is saved anyway).
inline std::vector<GdnChunk> gdn_chunk_plan(int64_t n, int64_t chunk, std::vector<int64_t> cuts, bool in_chunk) {
    std::vector<GdnChunk> out;
    if (n <= 0 || chunk <= 0) return out;
    cuts.erase(std::remove_if(cuts.begin(), cuts.end(), [&](int64_t c) { return c <= 0 || c >= n; }), cuts.end());
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    int64_t pos = 0;
    size_t ci = 0;
    while (pos < n) {
        int64_t end = std::min(n, pos + chunk);
        GdnChunk c;
        c.start = pos;
        if (!in_chunk) {
            while (ci < cuts.size() && cuts[ci] <= pos) ++ci;
            if (ci < cuts.size() && cuts[ci] < end) end = cuts[ci];
        } else {
            for (size_t k = 0; k < cuts.size(); ++k)
                if (cuts[k] > pos && cuts[k] < end) c.snaps.push_back(cuts[k]);
        }
        c.len = end - pos;
        out.push_back(c);
        pos = end;
    }
    return out;
}

// the snapshot kernel's position argument for one chunk: positions relative to the chunk start (the state after that many of the chunk's tokens), strictly
// ascending, each in [1, len - 1] (a snapshot at len is the chunk's own end state)
inline std::vector<int64_t> gdn_snap_relative(const GdnChunk& c) {
    std::vector<int64_t> r;
    for (int64_t p : c.snaps) {
        const int64_t q = p - c.start;
        if (q >= 1 && q <= c.len - 1 && (r.empty() || q > r.back())) r.push_back(q);
    }
    return r;
}

// float index of state element (row, head, col) of snapshot k in the snapshot buffer: the layout of `state` (rows S, heads HV, columns S) per snapshot
inline int64_t gdn_snap_offset(int64_t k, int64_t row, int64_t head, int64_t col, int64_t S, int64_t HV) {
    return k * (S * HV * S) + (row * HV + head) * S + col;
}

}  // namespace strata::prefill
