#pragma once
// LEADS.md Part P lead L5, STRATA_FREE_SETTLE_MS=N (default 0 = off, exact: it only waits).  hipfire (warpfront/hipfire,
// docs/methodology/lessons-0.4.5.md section 6) found that `hipMemGetInfo` lags after a free: the driver releases a freed
// buffer object asynchronously, so the free figure read right after the free is stale, and an admission decision taken
// on it refused two 16k prompts under stress.  Their fix waits, at most 250 ms (their observed maximum 2.84 ms), until
// the free figure shows the bytes that were released.  Our own rule, no code copied.
//
// Where it would apply here: a place that frees VRAM and then sizes something from the free figure (the loan's give-back
// before the next prompt's chunk is chosen, an expert-cache resize, a stage reload).  This header is the rule and a
// bounded wait over two callables so a host test can drive it; the call sites are NOT wired yet (nothing changes with
// the switch set until one is).  Header only, no device types.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <thread>

namespace strata::program {

struct FreeSettle {
    bool settled = false;       ///< the free figure reached the target (or no wait was asked for)
    uint64_t waited_us = 0;     ///< microseconds spent in `sleep_us`
    uint64_t last_free = 0;     ///< the last free figure read
    int polls = 0;
};

/// `free_before`: the free figure read before the release; `released`: bytes the caller just freed;
/// `slack`: bytes the figure may stay short of free_before + released (other processes allocate, rounding: the driver
/// frees whole granules).  cap_ms == 0: do not wait, report settled (the unset default).
/// `query` returns the current free bytes; `sleep_us` sleeps.  Polls at `step_us`, never longer than `cap_ms` in total.
inline FreeSettle free_settle_wait(uint64_t free_before, uint64_t released, uint64_t slack, uint32_t cap_ms, uint32_t step_us,
                                   const std::function<uint64_t()>& query, const std::function<void(uint32_t)>& sleep_us) {
    FreeSettle r;
    if (cap_ms == 0) { r.settled = true; return r; }
    const uint64_t target = free_before + released > slack ? free_before + released - slack : 0;
    const uint64_t cap_us = (uint64_t) cap_ms * 1000u;
    const uint32_t step = std::max<uint32_t>(step_us, 1u);
    for (;;) {
        r.last_free = query();
        ++r.polls;
        if (r.last_free >= target) { r.settled = true; return r; }
        if (r.waited_us >= cap_us) return r;
        const uint32_t s = (uint32_t) std::min<uint64_t>(step, cap_us - r.waited_us);
        sleep_us(s);
        r.waited_us += s;
    }
}

/// STRATA_FREE_SETTLE_MS text -> milliseconds: unset, empty, not a number or negative = 0 (off); capped at 5000.
inline uint32_t free_settle_ms_from_text(const char* text) {
    if (text == nullptr || *text == '\0') return 0;
    long long v = 0;
    for (const char* p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return 0;
        v = v * 10 + (*p - '0');
        if (v > 5000) return 5000;
    }
    return (uint32_t) v;
}

/// The call-site form: `cap_ms` from STRATA_FREE_SETTLE_MS, the real sleep, 200 us polls and a 64 MiB slack (the driver frees
/// whole granules and other processes allocate).  `free_before` is the free figure read BEFORE the release, `released` the
/// bytes the caller freed.  Returns at once with cap_ms == 0 (the unset default: no query, no sleep).
inline FreeSettle free_settle_after_free(uint64_t free_before, uint64_t released, uint32_t cap_ms,
                                         const std::function<uint64_t()>& query) {
    return free_settle_wait(free_before, released, (uint64_t) 64 << 20, cap_ms, 200, query,
                            [](uint32_t us) { std::this_thread::sleep_for(std::chrono::microseconds(us)); });
}

/// The switch: STRATA_FREE_SETTLE_MS read from the environment (0 = off).
inline uint32_t free_settle_ms_env() { return free_settle_ms_from_text(std::getenv("STRATA_FREE_SETTLE_MS")); }

}  // namespace strata::program
