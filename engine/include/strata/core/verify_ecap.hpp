#pragma once
// MOD_PLAN F5, STRATA_VERIFY_ECAP=<n> (opt-in, changes the output): cap the CPU experts that only the DRAFT rows of a
// verify window need.  After jmnargi/Strata-V100 24894e676b, 7c5041c358 and 376355453f ("relaxed MTP", upstream PR
// #3545).  That version caps the whole speculative tail on the GPU: token 0 keeps its K picks and the tail keeps the top
// N-K entries of the window.  This port caps each draft row and counts only the experts outside VRAM, because pancake's
// cost is the CPU pool's expert bytes, not the GPU's.
//
// Row 0 of a window is never touched: its input is the last committed token and its output is always committed.  Rows
// 1..T-1 take draft tokens, and their outputs count only when the drafts before them are accepted.  The rule for one pool
// call (one token group of one decode window, entry token * k + j in routing order):
//   1. every missed expert (not in the VRAM tier) of the committed rows (local rows < first) is read;
//   2. each draft row, in window order, walks its missed entries in routing order.  The native router writes a token's
//      top-k by descending routing weight (route_row: rank 0 first, ties to the lower expert id), so this is weight order.
//      Under STRATA_ROUTE_PRIOR it is the biased selection order.  An expert that is already read costs nothing and is
//      kept.  A new expert is read only while the row has added fewer than `cap` new experts;
//   3. an entry whose expert is still unread is dropped.  The expert pool neither copies nor computes it, its row is zero
//      (the STRATA_ROUTE_TAIL_SKIP path, kind -2) and the other routing weights stay as routed (no renormalisation).
// So the CPU reads at most `cap` experts per draft row on top of what the committed row needs.  A draft row always keeps
// every VRAM expert and its `cap` best missed ones, and an entry of an expert that is read anyway is never dropped.
// cap 0: a draft row reads no CPU expert of its own (it keeps its VRAM experts and the ones row 0 reads).
//
// Lossy: a dropped expert can flip a draft row's pick and the draft acceptance; the emitted token of an accepted draft row
// and the K/V and recurrent state it commits are then computed without the dropped experts.  jmnargi's lesson
// (376355453f): a cap far too tight broke 12 of 20 tool-calling turns, so the token-forcing and task gates come first.
#include <cstdint>
#include <cstdlib>

namespace strata::core {

/// STRATA_VERIFY_ECAP's value: -1 unset or empty (off), -2 not a whole number in [0, 64), else the cap per draft row.
inline int verify_ecap_parse(const char* v) {
    if (v == nullptr || v[0] == '\0') return -1;
    char* end = nullptr;
    const long c = std::strtol(v, &end, 10);
    if (end == v || *end != '\0' || c < 0 || c >= 64) return -2;
    return (int) c;
}

/// The rule above.  first_of[i]: the first entry of the call that names entry i's expert (first_of[i] == i for that
/// one).  missed[i]: entry i's expert is outside the VRAM tier (the CPU computes it or the GPU reads it over PCIe); every
/// entry of one expert has the same value.  first: the first draft row of the call (1 when it holds the window's row 0,
/// else 0).  read: n_tok * k entries of scratch.  Sets drop[i] for every dropped entry and returns how many; cap < 0
/// drops nothing.
inline int64_t verify_ecap_drop(const int64_t* first_of, const bool* missed, int64_t n_tok, int64_t k, int64_t first,
                                int cap, bool* read, bool* drop) {
    const int64_t n = n_tok * k;
    for (int64_t i = 0; i < n; ++i) {
        read[i] = false;
        drop[i] = false;
    }
    if (cap < 0 || k <= 0) return 0;
    if (first < 0) first = 0;
    if (first > n_tok) first = n_tok;
    for (int64_t i = 0; i < first * k; ++i)   // 1. the committed rows read every missed expert
        if (missed[i]) read[first_of[i]] = true;
    for (int64_t t = first; t < n_tok; ++t) {   // 2. each draft row adds at most `cap` experts nobody reads yet
        int added = 0;
        for (int64_t j = 0; j < k; ++j) {
            const int64_t i = t * k + j;
            if (!missed[i] || read[first_of[i]]) continue;
            if (added < cap) {
                read[first_of[i]] = true;
                ++added;
            }
        }
    }
    int64_t dropped = 0;   // 3. what nobody reads is dropped (a later row may have read an earlier row's extra expert)
    for (int64_t i = 0; i < n; ++i)
        if (missed[i] && !read[first_of[i]]) {
            drop[i] = true;
            ++dropped;
        }
    return dropped;
}

}  // namespace strata::core
