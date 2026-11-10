// include/strata/program/promo_chunk.hpp - MOD_PLAN Phase 16 A7 (ProMoE, SJTU 2024, https://arxiv.org/abs/2410.22134):
// chunked, cancellable promotions in the asynchronous adaptive tier.  Opt-in: STRATA_PROMO_CHUNK=<bytes> (unset or 0:
// off, the tier copies whole experts as before, byte-identical).
//
// What it does (generate.cpp, adapt_tick, arena + --adapt-async 1 on the serial loop only):
//   - a promotion (an expert copied host -> VRAM into the slot of the expert it evicts) is copied one chunk per window
//     tick instead of all at once, so a round's DMA is spread over windows (less DDR4 contention with the CPU pool at
//     once) and a promotion can be stopped between two chunks;
//   - between windows, a promotion still copying is cancelled when the incoming expert was not routed since the round's
//     pick and the evicted one was (promo_should_cancel).  The slot still holds the evicted expert past the bytes already
//     written, so a cancel copies back only those bytes (promo_cancel) and the evicted expert is resident again; a cancel
//     before the first chunk landed costs nothing;
//   - each promotion lands (becomes resident, so the GPU computes it from the next window on) as soon as its own last chunk
//     has landed, not when the whole batch has: the window-granular form of ProMoE's "cached -> finished in flight -> CPU"
//     order.  ProMoE's in-window form (the GPU picks up an expert whose copy finishes during the layer) needs a per-slot
//     ready flag read by the expert kernels and is not built (pancake-docs/batch2/A7-A46-promotions.md).
// Exact: scheduling and placement only, no output depends on it (which window first computes an expert on the GPU may
// move, as with any tier timing; the tier is not bit-exact run to run already).
// Plain arithmetic on per-swap progress, so a host test walks it without a GPU (tests/program/promo_chunk_test.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace strata::program {

/// STRATA_PROMO_CHUNK=<bytes>: 0 / unset / not a number = off.  Rounded down to a 4 KiB multiple, at least 64 KiB (a
/// smaller chunk only adds copy calls); a K, M or G suffix multiplies by 1024, 1024^2, 1024^3.
inline size_t promo_chunk_parse(const char* v) {
    if (v == nullptr || *v == '\0') return 0;
    char* end = nullptr;
    const unsigned long long n = std::strtoull(v, &end, 10);
    if (end == v) return 0;
    unsigned long long mul = 1;
    if (*end == 'k' || *end == 'K') mul = 1024ull;
    else if (*end == 'm' || *end == 'M') mul = 1024ull * 1024ull;
    else if (*end == 'g' || *end == 'G') mul = 1024ull * 1024ull * 1024ull;
    else if (*end != '\0') return 0;
    if (n == 0) return 0;
    unsigned long long b = n * mul;
    if (b / mul != n) return 0;   // overflow
    b &= ~4095ull;
    if (b < 65536ull) b = 65536ull;
    return (size_t) b;
}

/// The process's setting (read once).
inline size_t promo_chunk_env() {
    static const size_t v = [] {
        const char* e = std::getenv("STRATA_PROMO_CHUNK");
        const size_t b = promo_chunk_parse(e);
        if (e != nullptr && *e != '\0' && b == 0 && !(e[0] == '0' && e[1] == '\0'))
            std::fprintf(stderr, "strata serve: STRATA_PROMO_CHUNK=%s is not a byte count: off\n", e);
        return b;
    }();
    return v;
}

enum class PromoState : uint8_t {
    Copying,    ///< the incoming expert's bytes go into the slot, chunk by chunk
    Restoring,  ///< cancelled: the evicted expert's overwritten bytes go back
    Landed,     ///< every byte of the incoming expert is in the slot: it is resident
    Restored,   ///< the evicted expert is whole in its slot again: it is resident
};

/// One swap of a round.  `done` counts the bytes landed (of the incoming expert while Copying, of the evicted one while
/// Restoring); [done, done + posted) is the chunk in flight.
struct PromoProgress {
    PromoState st = PromoState::Copying;
    size_t total = 0;        ///< the layer's blob bytes
    size_t done = 0;
    size_t posted = 0;
    size_t target = 0;       ///< Restoring: the evicted expert's bytes to put back (the incoming bytes written over them)
    float base_in = 0.0f;    ///< the incoming / evicted expert's routing count when the round's copies started
    float base_out = 0.0f;
};

inline bool promo_moving(const PromoProgress& p) { return p.st == PromoState::Copying || p.st == PromoState::Restoring; }
inline bool promo_cancelled(const PromoProgress& p) {
    return p.st == PromoState::Restoring || p.st == PromoState::Restored;
}

/// The next chunk's length from `done` (0: nothing left).  `all`: the rest at once (a drain: the round must finish now).
inline size_t promo_next_len(const PromoProgress& p, size_t chunk, bool all) {
    const size_t end = p.st == PromoState::Copying ? p.total : p.st == PromoState::Restoring ? p.target : 0;
    if (p.done >= end) return 0;
    const size_t rest = end - p.done;
    return (all || chunk == 0 || chunk >= rest) ? rest : chunk;
}

/// The chunk in flight has landed (its copies are fenced): advance.  True when the swap just finished (Landed or
/// Restored).
inline bool promo_chunk_landed(PromoProgress& p) {
    if (!promo_moving(p) || p.posted == 0) return false;
    p.done += p.posted;
    p.posted = 0;
    if (p.st == PromoState::Copying && p.done >= p.total) { p.st = PromoState::Landed; return true; }
    if (p.st == PromoState::Restoring && p.done >= p.target) { p.st = PromoState::Restored; return true; }
    return false;
}

/// The cancel rule, between windows (never on a drain): since the round's copies started, the incoming expert was not
/// routed and the evicted one was.  `fresh_*` = routing count now minus the count at the start (> 1e-6: routed, as the
/// k-th-miss tier reads the counts).
inline bool promo_should_cancel(float fresh_in, float fresh_out) { return fresh_in <= 1e-6f && fresh_out > 1e-6f; }

/// Cancel a swap that is Copying (or Landed, when its expert became resident elsewhere): only the bytes of the incoming
/// expert written so far are copied back from the evicted expert's blob, at the same offsets.  A chunk still in flight
/// counts as written (the copies back queue behind it on the same stream).  True when nothing has to be copied (no
/// byte written: the evicted expert is whole, resident again at once).
inline bool promo_cancel(PromoProgress& p) {
    if (p.st != PromoState::Copying && p.st != PromoState::Landed) return false;
    p.target = p.done + p.posted;
    p.done = 0;
    p.posted = 0;
    if (p.target == 0) { p.st = PromoState::Restored; return true; }
    p.st = PromoState::Restoring;
    return false;
}

}  // namespace strata::program
