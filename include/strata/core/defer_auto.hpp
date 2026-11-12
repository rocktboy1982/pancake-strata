#pragma once
// MOD_PLAN 17.1 note (STRATA_VERIFY_DEFER_AUTO=1, opt-in): expert deferral (STRATA_VERIFY_DEFER_CPU=1) only in the verify
// windows whose drafts are a lookup copy, every CPU row on time in every other window.  Header-only, no GPU, no model:
// tests/core/defer_auto_test.cpp.  Design note: pancake-docs/VERIFY_DEFER_AUTO.md.
//
// Why: on the RX 7900 XT (Q2_0, fast-hip preset, 2026-11-11, warm pass, 10 decode samples per row) removing the deferral
// gave prose +8-9% and code -1..-3%; the 2026-11-09 test phase had edit +15.8% and short no difference with it.  So it seems
// to pay in copy-heavy windows and to cost elsewhere.  These numbers are for the whole switch, not for this gate.
//
// How (verify.cpp): the deferral's window graphs already read one mapped word at run time, the rank guard of MOD_PLAN A44
// (h_top_ -> copy_rows_by_weight's top_live), and the host pool reads the same value from the plan sink (sink_.top).  A
// window staged with the rank guard at k (the model's experts per token) has every CPU entry among its token's top k, so
// every CPU row is "heavy" (defer_rank.hpp): the pool runs all its jobs in the first pass, post(l) waits for all of them
// and combines them, and the late addition in layer l + 1 copies zero rows and adds zero.  No new graph, no kernel change,
// no pool change.  What it does NOT give back: the window graphs stay the deferral's graphs, so the switches the deferral
// turns off for the whole process (STRATA_COMBINE_CPU, STRATA_WAIT_FUSE, STRATA_ROUTE_PUBLISH, STRATA_ROUTER_FUSE,
// STRATA_IC_WARM_MB) stay off in the exact windows too.
//
//   STRATA_VERIFY_DEFER_AUTO=1          deferral only in lookup windows (unset or 0: off, deferral as configured)
//   STRATA_VERIFY_DEFER_AUTO_MIN_T=N    a lookup window of fewer than N rows (head + drafts) keeps every row on time too
//                                       (unset or 0: any lookup window of 2 rows or more)
#include <cstdlib>
#include <cstring>

namespace strata::core {

/// Where a verify window's rows came from, as the serve decode loop knows it before the window runs.  (The same split as
/// STRATA_ROUTE_PRIOR_AUTO's PriorWindowKind on branch dev/p17/turbogate; kept apart so either branch merges alone.)
enum class DeferWindowKind : int {
    Prompt = 0,   ///< a merged first window: its rows are prompt tokens
    Plain = 1,    ///< one row, no drafts (a first window, or no drafter)
    Model = 2,    ///< the MTP's (or DFlash2's) drafts
    Lookup = 3,   ///< a suffix / prompt-lookup copy of earlier context (the loop's from_sfx; the cross-session index's too)
    Chain = 4,    ///< the MTP's drafts with a lookup tail appended (--lookup-chain)
};

struct DeferAutoConfig {
    bool on = false;   ///< STRATA_VERIFY_DEFER_AUTO=1 and the verifier can act on it (deferral and its heavy split on)
    int min_t = 0;     ///< STRATA_VERIFY_DEFER_AUTO_MIN_T: the fewest rows a lookup window needs to defer (0: any)
};

constexpr int kDeferAutoMaxMinT = 256;   ///< the largest STRATA_VERIFY_DEFER_AUTO_MIN_T accepted (no window holds more rows)

/// STRATA_VERIFY_DEFER_AUTO: "1" is on; nullptr, "" and "0" are off.  Anything else is off and `known` (when given) says
/// false, so the caller can warn.
inline bool parse_defer_auto(const char* v, bool* known = nullptr) {
    if (known != nullptr) *known = true;
    if (v == nullptr || v[0] == 0 || std::strcmp(v, "0") == 0) return false;
    if (std::strcmp(v, "1") == 0) return true;
    if (known != nullptr) *known = false;
    return false;
}

/// STRATA_VERIFY_DEFER_AUTO_MIN_T: a whole number in 0..kDeferAutoMaxMinT; nullptr / "" is 0.  Anything else is 0 and
/// `known` (when given) says false.
inline int parse_defer_auto_min_t(const char* v, bool* known = nullptr) {
    if (known != nullptr) *known = true;
    if (v == nullptr || v[0] == 0) return 0;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (end == v || *end != 0 || n < 0 || n > kDeferAutoMaxMinT) {
        if (known != nullptr) *known = false;
        return 0;
    }
    return (int) n;
}

/// The window's kind from the loop's own variables: `merged` (a first window carrying prompt tokens), `from_lookup` (its
/// drafts are the suffix drafter's copy), `chain_rows` (lookup tokens chained after the MTP's drafts) and `rows` (all of
/// the window's rows, head + drafts + chained tokens).
inline DeferWindowKind defer_window_kind(bool merged, bool from_lookup, int chain_rows, int rows) {
    if (merged) return DeferWindowKind::Prompt;
    if (rows <= 1) return DeferWindowKind::Plain;
    if (from_lookup) return DeferWindowKind::Lookup;
    if (chain_rows > 0) return DeferWindowKind::Chain;
    return DeferWindowKind::Model;
}

/// Does this window defer its light CPU rows?
///  - auto off: true for every window (the deferral as configured; defer_ok and the graphs decide as before);
///  - auto on: true for a lookup or chained-lookup window of at least max(2, min_t) rows, false for every other window.
/// A chained window counts as a lookup one (the A44 bandit's window type, from_sfx || chain_n > 0): it is the widest kind.
inline bool defer_window_on(const DeferAutoConfig& c, DeferWindowKind kind, int rows) {
    if (!c.on) return true;
    if (kind != DeferWindowKind::Lookup && kind != DeferWindowKind::Chain) return false;
    const int need = c.min_t > 2 ? c.min_t : 2;
    return rows >= need;
}

/// The rank guard a window stages (Verifier::stage_inputs: the mapped word its graph's row copy reads and the plan sink's
/// top, which the host pool reads).  `top_now` is what it stages without the gate (defer_top_now(): STRATA_VERIFY_DEFER_CPU_TOP
/// or the A44 bandit's arm); `k` the model's experts per token.
///  - auto off, or a window that defers: `top_now`, unchanged (so a run without the switch stages exactly what it did);
///  - auto on and a window that does not defer: `k`, so every CPU entry is among its token's top k and is kept on time.
/// A `k` that is not above 0 cannot say "every entry": `top_now` then (the window defers as configured).
inline int defer_staged_top(bool auto_on, bool window_defers, int top_now, int k) {
    if (!auto_on || window_defers || k <= 0) return top_now;
    return k;
}

}  // namespace strata::core
