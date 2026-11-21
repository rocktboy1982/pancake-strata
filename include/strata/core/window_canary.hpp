// include/strata/core/window_canary.hpp - STRATA_WINDOW_CANARY=1 (default off, debug only): a capacity canary for the captured verify windows
// (LEADS.md Part U U9, idea from the capacity mode of hcharbonnier's rad-kbench: a kernel that reads the band ceiling as its row count, or writes
// the ceiling instead of the live extent, overruns a buffer that the graph was captured against).
//
// The helpers work on a word buffer that holds `cap` rows: before a window of `live` rows is staged, the rows [live, cap) are filled with a
// pattern; after the window is synchronised, the pattern must still be there.  A word that changed is a write past the live extent: the report
// names the buffer and the first word.  Host-only code (no CUDA): the engine calls it on the mapped host views of its staging buffers.
// Without the env nothing is filled and nothing is checked (enabled() is false).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>

namespace strata::core::window_canary {

constexpr int32_t kPattern = (int32_t) 0xC4A7A5E1u;

/// STRATA_WINDOW_CANARY: unset, empty or "0" = off.
inline bool enabled() {
    const char* e = std::getenv("STRATA_WINDOW_CANARY");
    return e != nullptr && e[0] != 0 && e[0] != '0';
}

/// A window of `t` rows fits a buffer captured for `max_t` rows.
inline bool window_fits(int t, int max_t) { return t >= 1 && t <= max_t; }

/// Fill the words [live, cap) with the pattern; returns how many words were filled (0 when live >= cap).
inline size_t fill(int32_t* buf, size_t live, size_t cap) {
    if (buf == nullptr || live >= cap) return 0;
    for (size_t i = live; i < cap; ++i) buf[i] = kPattern;
    return cap - live;
}

/// Index of the first word in [live, cap) that is not the pattern, or -1 when the guard is intact.
inline long first_bad(const volatile int32_t* buf, size_t live, size_t cap) {
    if (buf == nullptr) return -1;
    for (size_t i = live; i < cap; ++i)
        if (buf[i] != kPattern) return (long) i;
    return -1;
}

/// One log line for a violated guard (empty when intact): the buffer, the window size, the live extent and the first changed word.
inline std::string describe(const char* what, int t, size_t live, size_t cap, long bad) {
    if (bad < 0) return std::string();
    char b[256];
    std::snprintf(b, sizeof b, "strata verify: STRATA_WINDOW_CANARY: the %d-token window wrote past its live extent in %s (live %zu words, capacity %zu, first changed word %ld)",
                  t, what, live, cap, bad);
    return std::string(b);
}

}  // namespace strata::core::window_canary
