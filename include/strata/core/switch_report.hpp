#pragma once
// MOD_PLAN "Project features discovered" item 1: every opt-in switch says once that it acted, or why it will not act.
// COPY256, IC_WARM_MB and ROUTER_FUSE were silent no-ops under deferral, so the arms that set them measured nothing.
// Each line is one of
//
//     strata switch: NAME acted: DETAIL
//     strata switch: NAME will not act: REASON
//
// on stderr, NAME the switch (its environment variable or build macro), DETAIL / REASON free text on the same line (a line
// break in it is written as a space).  tools/pancake/switch_report.py reads these lines; an A/B arm whose switch never wrote
// "acted" measured nothing.  Each kind is written at most once per process and switch, so a switch can write both: a reason
// that held for a while (a short window, a fallback), then it acted.  "acted" means the switch's path ran (or was captured
// into a window graph) at least once.
//
// Call these only where the switch is set: with every switch unset nothing is written.  After a switch's first line of a
// kind, a call costs one relaxed atomic load.  The switches are listed below, one object each (inline: one per process,
// whichever file reports it).
#include <atomic>
#include <cstdarg>
#include <cstddef>
#include <cstdio>

namespace strata::switch_report {

struct Switch {
    const char* name;
    std::atomic<bool> acted_said{false};
    std::atomic<bool> refused_said{false};
};

inline Switch wait_copy256{"STRATA_WAIT_COPY256"};               // F21, verify_kernels.cu + verify.cpp's plan copy
inline Switch route_fast{"STRATA_ROUTE_FAST"};                   // F9, verify.cpp's router step
inline Switch router_fuse{"STRATA_ROUTER_FUSE"};                 // item 48, verify.cpp's router step
inline Switch ic_warm{"STRATA_IC_WARM_MB"};                      // item 47, verify.cpp
inline Switch expdec_swiglu{"STRATA_EXPDEC_SWIGLU"};             // item 48, iq_kernels.cu
inline Switch hip_global_weights{"STRATA_HIP_GLOBAL_WEIGHTS"};   // Phase 15, a build macro: iq_kernels.cu at start-up

inline constexpr const char* kActed = "acted";
inline constexpr const char* kWillNotAct = "will not act";

#if defined(__GNUC__) || defined(__clang__)
#define STRATA_SWITCH_PRINTF(fmt_arg, first_arg) __attribute__((format(printf, fmt_arg, first_arg)))
#else
#define STRATA_SWITCH_PRINTF(fmt_arg, first_arg)
#endif

// the line without its newline (snprintf's return value)
inline int format_line(char* buf, size_t size, const char* name, const char* kind, const char* text) {
    return std::snprintf(buf, size, "strata switch: %s %s: %s", name, kind, text);
}

// true: this call wrote the line (the first of its kind for this switch)
STRATA_SWITCH_PRINTF(4, 0)
inline bool say(std::atomic<bool>& said, const char* name, const char* kind, const char* fmt, va_list ap) {
    if (said.load(std::memory_order_relaxed) || said.exchange(true, std::memory_order_relaxed)) return false;
    char text[512];
    std::vsnprintf(text, sizeof text, fmt, ap);
    for (char* c = text; *c != '\0'; ++c)
        if (*c == '\n' || *c == '\r') *c = ' ';
    char line[640];
    format_line(line, sizeof line, name, kind, text);
    std::fprintf(stderr, "%s\n", line);   // one call: the line is not interleaved with another thread's output
    return true;
}

// "strata switch: NAME acted: DETAIL" once per process; DETAIL is printf's fmt + args
STRATA_SWITCH_PRINTF(2, 3) inline bool acted(Switch& s, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const bool wrote = say(s.acted_said, s.name, kActed, fmt, ap);
    va_end(ap);
    return wrote;
}

// "strata switch: NAME will not act: REASON" once per process (the first reason given)
STRATA_SWITCH_PRINTF(2, 3) inline bool will_not_act(Switch& s, const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    const bool wrote = say(s.refused_said, s.name, kWillNotAct, fmt, ap);
    va_end(ap);
    return wrote;
}

#undef STRATA_SWITCH_PRINTF

}  // namespace strata::switch_report
