// include/strata/core/subfit_trace.hpp - STRATA_SUBFIT_TRACE=1 (default off, diagnostic only): where the stand-in learner's CPU worker
// spends its time per replay (TEST_LEDGER.md T-632: STRATA_SUBFIT_LEARN alone cut decode 44-50% and fitted nothing).  Counters only; the
// worker takes no clock reading and does no extra work when the switch is unset.  Header-only, host tested (subfit_trace_test).
#pragma once
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::core::subfit_trace {

inline bool on_env(const char* v) { return v != nullptr && v[0] == '1' && v[1] == '\0'; }

// phases of one replay on the worker thread
enum Phase { kCopyE = 0, kCopyS, kExpertE, kExpertS, kSink, kPhases };

struct Tally {
    std::atomic<uint64_t> us[kPhases]{};
    std::atomic<uint64_t> jobs{0};
    void add(Phase p, uint64_t micros) { us[p].fetch_add(micros, std::memory_order_relaxed); }
    void job() { jobs.fetch_add(1, std::memory_order_relaxed); }
};

// one line for the request-end report; mean microseconds per replay and the total worker time
inline std::string line(const Tally& t) {
    static const char* names[kPhases] = {"copy_blob(e)", "copy_blob(s)", "expert(e)", "expert(s)", "sample sink"};
    const uint64_t n = t.jobs.load(std::memory_order_relaxed);
    uint64_t total = 0;
    for (int i = 0; i < kPhases; ++i) total += t.us[i].load(std::memory_order_relaxed);
    std::string out = "strata serve: subfit trace: " + std::to_string((unsigned long long) n) + " replays, worker " +
                      std::to_string((unsigned long long) (total / 1000)) + " ms in all;";
    for (int i = 0; i < kPhases; ++i) {
        char b[96];
        std::snprintf(b, sizeof b, " %s %.0f us/replay;", names[i],
                      n > 0 ? (double) t.us[i].load(std::memory_order_relaxed) / (double) n : 0.0);
        out += b;
    }
    out += "\n";
    return out;
}

}  // namespace strata::core::subfit_trace
