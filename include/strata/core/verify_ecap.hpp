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

/// ECAP v2 (MOD_PLAN section 6 item 1 of the 2026-11-17 queue): STRATA_VERIFY_ECAP_ROWS=c1,c2,... gives the cap per draft
/// row by its window row index: c1 is window row 1 (the first draft), c2 row 2, and the last value repeats for the rows
/// after the list.  Each value is -1 (that row is uncapped: it reads every expert it names) or 0..63.  The list replaces
/// the scalar STRATA_VERIFY_ECAP for every row; unset, the scalar rule runs exactly as before (the scalar function below is
/// the same code path with no list, and verify_ecap_test.cpp checks that a list of one repeated value equals it).
/// STRATA_VERIFY_ECAP_MINP=<p> adds a confidence gate: a draft row whose MTP draft probability is below p gets cap 0
/// whatever its list value.  A row without a known probability is not gated.
constexpr int kEcapRowsMax = 16;

/// The list's value: 0 unset or empty (off), -2 a bad token (a value outside -1..63, an empty field, more than
/// kEcapRowsMax values), else the number of values stored in out[].
inline int verify_ecap_rows_parse(const char* v, int8_t* out) {
    if (v == nullptr || v[0] == '\0') return 0;
    int n = 0;
    const char* p = v;
    for (;;) {
        char* end = nullptr;
        const long c = std::strtol(p, &end, 10);
        if (end == p || c < -1 || c >= 64 || n >= kEcapRowsMax) return -2;
        out[n++] = (int8_t) c;
        if (*end == '\0') return n;
        if (*end != ',') return -2;
        p = end + 1;
    }
}

/// The cap of window row `wrow` (>= 1): the list's value, the last one repeating; `scalar` when there is no list.
inline int verify_ecap_row_cap(const int8_t* rows, int n_rows, int scalar, int64_t wrow) {
    if (rows == nullptr || n_rows <= 0 || wrow < 1) return scalar;
    return rows[wrow - 1 < n_rows ? wrow - 1 : n_rows - 1];
}

/// STRATA_VERIFY_ECAP_MINP's value: -1 unset or empty (off), -2 not a number in (0, 1], else p.
inline float verify_ecap_minp_parse(const char* v) {
    if (v == nullptr || v[0] == '\0') return -1.0f;
    char* end = nullptr;
    const double p = std::strtod(v, &end);
    if (end == v || *end != '\0' || !(p > 0.0) || p > 1.0) return -2.0f;
    return (float) p;
}

/// The MTP draft probabilities of the next window, published by MtpDrafter (chain_poll, chain_outputs_ready, draft_first)
/// and taken by the Verifier: p[r] is the probability of the draft in window row r + 1.  `n` drafts are known; the
/// verifier uses the board only when n is at least its window's draft count (a chain can be longer than the window it
/// feeds) and clears it after every run, so a lookup-draft window after an unused MTP chain is the one way a stale value
/// could be read (unconfirmed static analysis).  Single-threaded use: the decode loop publishes and runs on one thread.
struct EcapConfBoard {
    float p[8] = {};
    int n = 0;
};
inline EcapConfBoard& ecap_conf_board() {
    static EcapConfBoard b;
    return b;
}
/// True when STRATA_VERIFY_ECAP_MINP names a probability: the drafter publishes only then (unset: no work, no change).
inline bool verify_ecap_minp_enabled() {
    static const bool on = verify_ecap_minp_parse(std::getenv("STRATA_VERIFY_ECAP_MINP")) > 0.0f;
    return on;
}
inline void ecap_conf_publish(const float* probs, int n) {
    EcapConfBoard& b = ecap_conf_board();
    if (probs == nullptr || n <= 0) { b.n = 0; return; }
    if (n > 8) n = 8;
    for (int i = 0; i < n; ++i) b.p[i] = probs[i];
    b.n = n;
}

/// The rule above with per-row caps.  `row_base` is the window row of this call's local row 0 (the call's token group
/// starts there).  rows/n_rows: the list (nullptr: the scalar `cap` for every row).  conf/n_conf/minp: the confidence gate,
/// conf[r] is the probability of window row r + 1 (nullptr or minp <= 0: no gate).  A row cap < 0 reads every missed
/// expert of that row.  The other arguments are the scalar rule's, below.
inline int64_t verify_ecap_drop_rows(const int64_t* first_of, const bool* missed, int64_t n_tok, int64_t k, int64_t first,
                                     int64_t row_base, int cap, const int8_t* rows, int n_rows, const float* conf,
                                     int n_conf, float minp, bool* read, bool* drop) {
    const int64_t n = n_tok * k;
    for (int64_t i = 0; i < n; ++i) {
        read[i] = false;
        drop[i] = false;
    }
    const bool gate = conf != nullptr && minp > 0.0f && n_conf > 0;
    if (k <= 0 || (cap < 0 && !(rows != nullptr && n_rows > 0) && !gate)) return 0;
    if (first < 0) first = 0;
    if (first > n_tok) first = n_tok;
    for (int64_t i = 0; i < first * k; ++i)   // 1. the committed rows read every missed expert
        if (missed[i]) read[first_of[i]] = true;
    for (int64_t t = first; t < n_tok; ++t) {   // 2. each draft row adds at most its cap of experts nobody reads yet
        const int64_t wrow = row_base + t;
        int rc = verify_ecap_row_cap(rows, n_rows, cap, wrow);
        if (gate && wrow >= 1 && wrow - 1 < n_conf && conf[wrow - 1] < minp) rc = 0;
        int added = 0;
        for (int64_t j = 0; j < k; ++j) {
            const int64_t i = t * k + j;
            if (!missed[i] || read[first_of[i]]) continue;
            if (rc < 0 || added < rc) {
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

/// The scalar rule: first_of[i]: the first entry of the call that names entry i's expert (first_of[i] == i for that
/// one).  missed[i]: entry i's expert is outside the VRAM tier (the CPU computes it or the GPU reads it over PCIe); every
/// entry of one expert has the same value.  first: the first draft row of the call (1 when it holds the window's row 0,
/// else 0).  read: n_tok * k entries of scratch.  Sets drop[i] for every dropped entry and returns how many; cap < 0
/// drops nothing.
inline int64_t verify_ecap_drop(const int64_t* first_of, const bool* missed, int64_t n_tok, int64_t k, int64_t first,
                                int cap, bool* read, bool* drop) {
    return verify_ecap_drop_rows(first_of, missed, n_tok, k, first, 0, cap, nullptr, 0, nullptr, 0, 0.0f, read, drop);
}

}  // namespace strata::core
