// include/strata/core/kv_ring.hpp - the depth-adaptive resident K/V ring (opt-in, STRATA_KV_RING_ADAPT). Header-only, CPU only.
//
// Idea and rule: xyzzing (rdna3-inference-ledger, upstream Strata PR #3919, see README.md section 9). With KV streaming
// (--kv-resident N) the K/V pool of every QSA layer is N cells in VRAM, one fixed value for every run. A ring chosen for a
// deep context is paid by shallow runs too, and the PR measured that a bigger ring lowers draft acceptance at 1K. So with
// STRATA_KV_RING_ADAPT=1 the --kv-resident value becomes a CAP and the ring is picked from the run's depth.
//
// The pool is sized by its slot count, so the caller must apply the result BEFORE the session arena is allocated and the
// graphs are captured (generate.cpp does it where --kv-resident is first applied). Nothing here touches a device.
#pragma once

#include <cstdint>
#include <cstdlib>

namespace strata::core {

/// The ring a shallow run keeps: the engine's historical --kv-resident default (what setup writes).
inline constexpr int64_t kKvRingFloorCells = 32768;

/// STRATA_KV_RING_ADAPT: unset, empty or 0 is off (today's behavior); a number other than 0 is on. A value with no
/// leading number ("yes") reads as 0, so it is off (atoll, as the other numeric switches of generate.cpp).
inline bool kv_ring_adapt_enabled(const char* env) { return env != nullptr && env[0] != '\0' && std::atoll(env) != 0; }

/// The resident ring, in cells per QSA layer, for a run that needs `cells_needed` cells (prompt plus the tokens it may
/// generate; under --serve the configured --max-context).
///  - switch off, or `cap` <= 0 (KV streaming off): `cap` unchanged.
///  - `cap` <= `floor`: `cap` (the user asked for a ring no larger than the floor; there is nothing to grow).
///  - `cells_needed` >= 2 x `cap`: `cap`. Only a context this deep earns the cap: the PR's first rule, "grow when the
///    context reaches the cap", cost 16.5% decode at 64K on its card (74.8 -> 62.5 tok/s) while 128K gained 23%.
///  - otherwise `floor`.
/// `floor` is passed in (generate.cpp: max(qsa_kv_resident_min(), kKvRingFloorCells)) so the function stays pure.
/// The PR rounds `cells_needed` up to the KV page (4 cells) first; that moves the 2 x cap edge by at most 3 cells and is
/// left out here so the function needs no model geometry.
inline int64_t kv_ring_adapt(bool enabled, int64_t cap, int64_t floor, int64_t cells_needed) {
    if (!enabled || cap <= 0) return cap;
    if (cap <= floor) return cap;
    return cells_needed / 2 >= cap ? cap : floor;   // the same as cells_needed >= 2 * cap, without the overflow
}

}  // namespace strata::core
