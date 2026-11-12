#pragma once
// MOD_PLAN 17.1 (STRATA_ROUTE_PRIOR_AUTO=1, opt-in, lossy only through turbo): the route prior (STRATA_ROUTE_PRIOR, "turbo")
// only in the verify windows whose drafts are a lookup copy, exact routing in every other window.  Header-only, no GPU, no
// model: tests/program/route_prior_auto_test.cpp.  Design note: pancake-docs/ROUTE_PRIOR_AUTO.md.
//
// Why: on the RX 7900 XT (IQ2_XS, 2026-11-11, one warm pass per prompt) turbo was +9% decode on a copy-heavy code prompt and
// -15% / -25% on two prose prompts against the exact fast-hip preset, with the same draft acceptance, so a global switch loses.
// A copy-heavy reply is the one where the suffix drafter (prompt lookup, --suffix-draft, on by default) fills the windows.
//
// How: the serve decode loop already writes one mapped float before each verify window (MOD_PLAN F13, STRATA_ROUTE_PRIOR_SCOPE)
// and route_prior.cu reads it on the GPU at run time, inside the captured window graphs: > 0 = the bonus acts, 0 = the kernel
// returns at once and the router's own top-10 stands.  The bonus's size (lambda), the depth grading and the PROTECT / KEEP_P
// guards are launch arguments baked into the graphs and do not change per window; only that float does.  This header decides
// the float's value from what the loop knows before the window runs: where the drafts came from and how many rows there are.
//
//   STRATA_ROUTE_PRIOR_AUTO=1          the bonus only in lookup windows (unset or 0: off, turbo as before)
//   STRATA_ROUTE_PRIOR_AUTO_MIN_T=N    a lookup window of fewer than N rows (head + drafts) stays exact too (unset or 0: any)
//
// With STRATA_ROUTE_PRIOR_SCOPE also set, a window takes the bonus only when both say so (each guard can only close a window).
#include <cstdlib>
#include <cstring>

namespace strata::program {

/// Where a verify window's rows came from, as the serve decode loop knows it before the window runs.
enum class PriorWindowKind : int {
    Prompt = 0,   ///< a merged first window: its rows are prompt tokens (the prompt's K/V stays the router's own)
    Plain = 1,    ///< one row, no drafts (a first window, or no drafter)
    Model = 2,    ///< the MTP's (or DFlash2's) drafts
    Lookup = 3,   ///< a suffix / prompt-lookup copy of earlier context (the loop's from_sfx; the cross-session index's too)
    Chain = 4,    ///< the MTP's drafts with a lookup tail appended (--lookup-chain): its first rows are model drafts
};

struct PriorAutoConfig {
    bool on = false;   ///< STRATA_ROUTE_PRIOR_AUTO=1
    int min_t = 0;     ///< STRATA_ROUTE_PRIOR_AUTO_MIN_T: the fewest rows a lookup window needs to take the bonus (0: any)
};

constexpr int kPriorAutoMaxMinT = 256;   ///< the largest STRATA_ROUTE_PRIOR_AUTO_MIN_T accepted (no window holds more rows)

/// STRATA_ROUTE_PRIOR_AUTO: "1" is on; nullptr, "" and "0" are off.  Anything else is off and `known` (when given) says false,
/// so the caller can warn.
inline bool parse_prior_auto(const char* v, bool* known = nullptr) {
    if (known != nullptr) *known = true;
    if (v == nullptr || v[0] == 0 || std::strcmp(v, "0") == 0) return false;
    if (std::strcmp(v, "1") == 0) return true;
    if (known != nullptr) *known = false;
    return false;
}

/// STRATA_ROUTE_PRIOR_AUTO_MIN_T: a whole number in 0..kPriorAutoMaxMinT; nullptr / "" is 0.  Anything else is 0 and `known`
/// (when given) says false.
inline int parse_prior_auto_min_t(const char* v, bool* known = nullptr) {
    if (known != nullptr) *known = true;
    if (v == nullptr || v[0] == 0) return 0;
    char* end = nullptr;
    const long n = std::strtol(v, &end, 10);
    if (end == v || *end != 0 || n < 0 || n > kPriorAutoMaxMinT) {
        if (known != nullptr) *known = false;
        return 0;
    }
    return (int) n;
}

/// The window's kind from the loop's own variables: `merged` (a first window carrying prompt tokens), `from_lookup` (its drafts
/// are the suffix drafter's copy), `chain_rows` (lookup tokens chained after the MTP's drafts) and `rows` (head + drafts).
inline PriorWindowKind prior_window_kind(bool merged, bool from_lookup, int chain_rows, int rows) {
    if (merged) return PriorWindowKind::Prompt;
    if (rows <= 1) return PriorWindowKind::Plain;
    if (from_lookup) return PriorWindowKind::Lookup;
    if (chain_rows > 0) return PriorWindowKind::Chain;
    return PriorWindowKind::Model;
}

/// The residency bonus one verify window takes, given the configured STRATA_ROUTE_PRIOR `lambda` (already resolved).
///  - auto off: `lambda` for every window (turbo as before; the other guards still decide their own windows);
///  - auto on: `lambda` for a lookup window of at least max(2, min_t) rows, 0 (exact routing) for every other window.
/// A `lambda` that is not above 0 (off, or NaN) gives 0 either way.
inline float route_prior_auto_lambda(const PriorAutoConfig& c, float lambda, PriorWindowKind kind, int rows) {
    if (!(lambda > 0.0f)) return 0.0f;
    if (!c.on) return lambda;
    if (kind != PriorWindowKind::Lookup) return 0.0f;
    const int need = c.min_t > 2 ? c.min_t : 2;
    return rows >= need ? lambda : 0.0f;
}

/// The gate float's verdict for one window: `scope_open` is what the loop decided before this switch existed (STRATA_ROUTE_PRIOR_SCOPE
/// and the merged-window rule).  Auto off: `scope_open` unchanged, so a run without STRATA_ROUTE_PRIOR_AUTO gates exactly as before.
/// Auto on: also closed unless route_prior_auto_lambda() gives this window the bonus.
inline bool prior_gate_open(bool scope_open, const PriorAutoConfig& c, float lambda, PriorWindowKind kind, int rows) {
    if (!c.on) return scope_open;
    return scope_open && route_prior_auto_lambda(c, lambda, kind, rows) > 0.0f;
}

}  // namespace strata::program
