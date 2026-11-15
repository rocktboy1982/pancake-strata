// include/strata/spec/window_rule.hpp - which rule picks a round's verify window, by the round's ACTIVE draft source.
//
// `use_mtp` means "the MTP layer is loaded"; with DFlash2 it stays true because the MTP is the fallback.  The decisions below
// read what the MTP chain just produced (the cost-aware rule's `chain_rows`, the draft probabilities), so they apply only when
// the MTP chain is the source of this round's candidates.  With DFlash2 as the source, `chain_rows` is never written (the chain
// callback does not run) and would pin the window at one row.  Header only, no device types: the CLI loop and the serve loop of
// every backend call it.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>

namespace strata::spec {

struct WindowRuleIn {
    bool use_mtp = false;        ///< the MTP layer is loaded (a fallback beside DFlash2 counts)
    bool draft_on = false;       ///< some drafter proposes (the MTP, or DFlash2 alone with no MTP loaded)
    bool df2_active = false;     ///< DFlash2 drew this round's candidates (CLI: use_df2; serve: this request's drafter)
    bool spec_cost = false;      ///< STRATA_SPEC_COST=1: the cost-aware MTP window
    bool first_window = false;   ///< the prompt's last token alone: the caller sets its own width afterwards
    int chain_rows = 1;          ///< the window the cost rule kept in the MTP chain just drafted (MTP only)
    int cap = 1;                 ///< S_mtp: the widest window of this source
    int start = 1;               ///< the width before any rule (CLI: cap; serve: cap with the MTP, 1 without)
    double spec_min_p = 0.0;     ///< --spec-min-p: stop at the first draft whose probability is below it (0: off)
    double draft_conf = 0.0;     ///< --draft-conf: DFlash2's window ends where the product of the probabilities falls below it (0: off)
    const float* dprob = nullptr;   ///< the drafts' probabilities, dprob[j] for draft j (the row j + 1 of the window)
};

/// The cost-aware MTP window applies only to the MTP chain's own candidates.
inline bool cost_window_applies(bool use_mtp, bool spec_cost, bool df2_active, bool first_window) {
    return use_mtp && spec_cost && !df2_active && !first_window;
}

/// The MTP's q lists (STRATA_SPEC_PROB) are the proposal law only for candidates the MTP drew.
inline bool mtp_q_applies(bool use_mtp, bool df2_active, bool mtp_prob, bool from_lookup, bool first_window, int mtp_rows) {
    return use_mtp && !df2_active && mtp_prob && !from_lookup && !first_window && mtp_rows > 1;
}

/// DFlash2 with a LONG suffix match: the lookup window is taken without a DFlash2 draft round.  The choice "lookup or DFlash2" needs the draft only to
/// test that its first token agrees; when the match is long the lookup is the likelier winner and the draft (2.4 ms on Metal, 1.2 ms on the RTX PRO
/// 5000) would be thrown away.  The round's draft is then DEFERRED to the start of the next iteration, where the policy decides (lookup: the committed
/// rows only update the drafter's ring; otherwise the draft is made then, from the same taps).  min_match 0 = off.
inline bool skip_draft_for_lookup(bool df2_active, int min_match, int lookup_k, int match) {
    return df2_active && min_match > 0 && lookup_k > 0 && match >= min_match;
}

/// STRATA_DF2_LOOKUP_SKIP=N (match length in tokens; 0 = off).  `build_default`: what a build without the variable uses (Metal 8, others 0 until measured).
inline int df2_lookup_skip_min(int build_default) {
    static const int v = [build_default] {
        const char* e = std::getenv("STRATA_DF2_LOOKUP_SKIP");
        return e != nullptr && e[0] != '\0' ? std::atoi(e) : build_default;
    }();
    return v;
}

inline int choose_window(const WindowRuleIn& in) {
    int T = in.start;
    if (cost_window_applies(in.use_mtp, in.spec_cost, in.df2_active, in.first_window)) {
        T = std::clamp(in.chain_rows, 1, in.cap);
    } else if (in.draft_on && in.spec_min_p > 0.0 && in.dprob != nullptr) {
        T = 1;
        while (T < in.cap && in.dprob[T - 1] >= (float) in.spec_min_p) ++T;
    }
    if (in.df2_active && in.draft_conf > 0.0 && in.dprob != nullptr) {   // DFlash2's own confidence: every draft so far holds
        T = 1;
        double chain = 1.0;
        while (T < in.cap && (chain *= (double) in.dprob[T - 1]) >= in.draft_conf) ++T;
    }
    return T;
}

/// STRATA_DF2_GATE=1 (opt-in, measured on the Mac): DFlash2's window from a calibrated survival instead of a fixed confidence.  The selector's cumulative confidence C_j of
/// draft j over-states the real chance that drafts 1..j all hold (M5 Max UD-Q4_K_XL: C = 0.84, survival 0.70 at j = 5), and the right confidence differs by request
/// (a story wants a narrower window than JSON).  Per request this keeps, for every position j and confidence bucket, how often the first j drafts held, shrunk toward a
/// prior of C^1.3, and extends the window to row j+1 while  survival * (ms per committed token) >= row_ms  (what one more verify row costs).
struct AdaptiveGate {
    static constexpr int NB = 6;
    double n[8][NB] = {}, h[8][NB] = {};
    double ms_per_tok = 0.0;       // EMA of the round's wall ms per committed token
    double row_ms = 3.0;           // one more verify row (M5 Max: 2.65-3.3 ms)
    double prior_n = 3.0;
    static int bucket(double c) { return c < 0.15 ? 0 : c < 0.3 ? 1 : c < 0.5 ? 2 : c < 0.7 ? 3 : c < 0.85 ? 4 : 5; }
    double survival(int j, double c) const {
        const int b = bucket(c);
        const double prior = std::pow(c, 1.3);
        return (h[j][b] + prior_n * prior) / (n[j][b] + prior_n);
    }
    /// rows of the next window: 1 + the drafts kept (cap = widest window)
    int choose(const float* dprob, int cap) const {
        int T = 1;
        double chain = 1.0;
        const double v = ms_per_tok > 0.0 ? ms_per_tok : 9.0;
        while (T < cap) {
            chain *= (double) dprob[T - 1];
            if (survival(T, chain) * v < row_ms) break;
            ++T;
        }
        return T;
    }
    /// after a DFlash2 round of T rows, a drafts accepted: drafts 1..T-1 were offered, draft j survived when a >= j
    void observe(const float* dprob, int T, int a) {
        double chain = 1.0;
        for (int j = 1; j < T && j < 8; ++j) {
            chain *= (double) dprob[j - 1];
            const int b = bucket(chain);
            n[j][b] += 1.0;
            h[j][b] += a >= j ? 1.0 : 0.0;
        }
    }
    void observe_round(double round_ms, int committed) {
        if (committed < 1) return;
        const double m = round_ms / committed;
        ms_per_tok = ms_per_tok > 0.0 ? 0.9 * ms_per_tok + 0.1 * m : m;
    }
};

/// LEADS.md section 3: entropy stop rules for the MTP draft chain (STRATA_SPEC_ENTROPY=<rule>[,key=value...], opt-in; they need
/// the drafter's entropies, so they turn STRATA_DRAFT_ENTROPY on).  The rule is asked after draft j is computed whether to draft
/// j + 1, like --spec-min-p (draft j itself stays in the window).  Only how many drafts are verified changes: greedy output is
/// the same.  A NaN entropy (no full softmax: a coupled draft) never stops the chain.
///   adaedl  AdaEDL (Agrawal et al., arXiv 2410.18351, section 3 and Algorithm 1): stop when 1 - sqrt(gamma * H) < lambda.
///           After each MTP round of n drafts with a accepted: AR = b1 AR + (1 - b1) a/n; lambda' = lambda + eps if AR < alpha,
///           lambda - eps if a != n, else lambda; lambda = b2 lambda + (1 - b2) lambda'.  Paper: gamma 0.2, b1 0.5, b2 0.9,
///           eps 0.01, alpha 0.9; the initial lambda is not given (here 0.4, key lambda=).  This is LEADS.md A22.
///   svip    SVIP (Zhang et al., "Draft Model Knows When to Stop", arXiv 2411.18462, eq. 7): stop when sqrt(H) > h (h 0.3, key h=).
///   adasd   AdaSD's generation threshold (Lu, Hong, Liu, arXiv 2512.11280, section 4.3): stop when H > T_G, T_G = the mean
///           entropy of every rejected draft so far (the first draft of a round that did not hold); no stop before the first
///           rejection.  Only the length rule (LEADS.md "AdaSD adaptive length rule (as DISCO check)"); its JS-distance
///           relaxed acceptance is not built here.
struct EntropyStop {
    enum Rule { Off = 0, AdaEDL, SVIP, AdaSD };
    Rule rule = Off;
    double gamma = 0.2, lambda = 0.4, b1 = 0.5, b2 = 0.9, eps = 0.01, alpha = 0.9;   // AdaEDL
    double ar = -1.0;                                                                 // AdaEDL running acceptance (-1: none yet)
    double h = 0.3;                                                                   // SVIP
    double rej_sum = 0.0;                                                             // AdaSD
    int64_t rej_n = 0;
    bool on() const { return rule != Off; }

    /// Parse "adaedl[,gamma=..,lambda=..,eps=..,alpha=..]", "svip[,h=..]", "adasd".  Empty / null: Off.  false + Off on error.
    bool parse(const char* v) {
        *this = EntropyStop();
        if (v == nullptr || v[0] == '\0') return true;
        const char* p = v;
        auto word = [&](const char* w) {
            const size_t n = std::strlen(w);
            return std::strncmp(p, w, n) == 0 && (p[n] == '\0' || p[n] == ',') ? (p += n, true) : false;
        };
        if (word("adaedl")) rule = AdaEDL;
        else if (word("svip")) rule = SVIP;
        else if (word("adasd")) rule = AdaSD;
        else return false;
        while (*p == ',') {
            ++p;
            const char* eq = std::strchr(p, '=');
            if (eq == nullptr) { *this = EntropyStop(); return false; }
            const std::string key(p, (size_t) (eq - p));
            char* end = nullptr;
            const double x = std::strtod(eq + 1, &end);
            if (end == eq + 1 || !(x == x)) { *this = EntropyStop(); return false; }
            if (key == "gamma") gamma = x;
            else if (key == "lambda") lambda = x;
            else if (key == "eps") eps = x;
            else if (key == "alpha") alpha = x;
            else if (key == "h") h = x;
            else { *this = EntropyStop(); return false; }
            p = end;
        }
        if (*p != '\0') { *this = EntropyStop(); return false; }
        return true;
    }
    /// After draft j with entropy H (nats): draft j + 1?
    bool keep(double H) const {
        if (!(H == H)) return true;
        switch (rule) {
            case AdaEDL: return 1.0 - std::sqrt(gamma * std::max(0.0, H)) >= lambda;
            case SVIP: return std::sqrt(std::max(0.0, H)) <= h;
            case AdaSD: return rej_n == 0 || H <= rej_sum / (double) rej_n;
            default: return true;
        }
    }
    /// After a verified plain MTP round: `n` drafts offered, `a` held; `ent[j]` the entropy of draft j (may be null).
    void observe(int n, int a, const float* ent) {
        if (n <= 0) return;
        a = std::clamp(a, 0, n);
        if (rule == AdaEDL) {
            const double ari = (double) a / (double) n;
            ar = ar < 0.0 ? ari : b1 * ar + (1.0 - b1) * ari;
            double l2 = lambda;
            if (ar < alpha) l2 = lambda + eps;
            else if (a != n) l2 = lambda - eps;
            lambda = b2 * lambda + (1.0 - b2) * l2;
        } else if (rule == AdaSD && a < n && ent != nullptr && ent[a] == ent[a]) {
            rej_sum += (double) ent[a];
            ++rej_n;
        }
    }
};

/// STRATA_SPEC_ENTROPY, parsed once ("strata: STRATA_SPEC_ENTROPY=...: not understood, off" on a bad value).
inline const EntropyStop& entropy_stop_env() {
    static const EntropyStop v = [] {
        EntropyStop e;
        const char* s = std::getenv("STRATA_SPEC_ENTROPY");
        if (!e.parse(s)) std::fprintf(stderr, "strata: STRATA_SPEC_ENTROPY=%s: not understood (adaedl|svip|adasd[,k=v]), off\n", s);
        return e;
    }();
    return v;
}

/// STRATA_DRAFT_ENTROPY=1 (or any STRATA_SPEC_ENTROPY rule): the MTP draft head also computes each row's entropy and runner-up
/// probability (kernels row_entropy) for the stop rules and the window trace's record 7.  Unset: no launch, no buffer.
inline bool draft_entropy_on() {
    static const bool v = [] {
        const char* e = std::getenv("STRATA_DRAFT_ENTROPY");
        return (e != nullptr && e[0] == '1') || entropy_stop_env().on();
    }();
    return v;
}

/// LEADS.md section 3, Medusa typical acceptance on the thinking only (STRATA_TYPICAL_THINK=<eps>[,<alpha>], opt-in, LOSSY: the
/// reply is no longer a sample of the target).  Cai et al., "Medusa", arXiv 2401.10774, section 2.3.1 (reference code:
/// posterior_threshold 0.09, posterior_alpha 0.3): in a sampled request, inside a <think> span, the draft at row t + 1 is kept
/// when the target's p_t(draft) > min(eps, alpha * exp(-H(p_t))) at the request's temperature, even where the sampled token
/// differs; the window's acceptance goes on from there and the row after the last kept draft keeps its own sample.  Greedy
/// requests and the answer outside the thinking are untouched.  eps <= 0 or unset: off.
struct TypicalAccept {
    float eps = 0.0f, alpha = 0.3f;
    bool on() const { return eps > 0.0f; }
    static float threshold(float eps, float alpha, float H) { return std::min(eps, alpha * std::exp(-H)); }
};
inline bool parse_typical(const char* v, TypicalAccept& t) {
    t = TypicalAccept();
    if (v == nullptr || v[0] == '\0') return true;
    char* end = nullptr;
    const double e = std::strtod(v, &end);
    if (end == v || !(e > 0.0) || e >= 1.0) { t = TypicalAccept(); return false; }
    t.eps = (float) e;
    if (*end == ',') {
        const char* a0 = end + 1;
        const double a = std::strtod(a0, &end);
        if (end == a0 || !(a > 0.0)) { t = TypicalAccept(); return false; }
        t.alpha = (float) a;
    }
    if (*end != '\0') { t = TypicalAccept(); return false; }
    return true;
}
inline const TypicalAccept& typical_env() {
    static const TypicalAccept v = [] {
        TypicalAccept t;
        const char* s = std::getenv("STRATA_TYPICAL_THINK");
        if (!parse_typical(s, t)) std::fprintf(stderr, "strata: STRATA_TYPICAL_THINK=%s: not understood (<eps>[,<alpha>]), off\n", s);
        return t;
    }();
    return v;
}
/// Host reference of the kernel typical_accept_rows for one row (double sums): whether `cand` passes; p / H out.
inline bool typical_row_ref(const float* l, int n, int cand, float temp, float eps, float alpha, double* p_out = nullptr,
                            double* h_out = nullptr) {
    const double it = temp > 0.0f ? 1.0 / (double) temp : 1.0;
    double m = -INFINITY;
    for (int i = 0; i < n; ++i) if (l[i] == l[i]) m = std::max(m, (double) l[i] * it);
    double s = 0.0, u = 0.0;
    for (int i = 0; i < n; ++i) {
        const double d = (double) l[i] * it - m;
        const double e = std::exp(d);
        s += e;
        if (e > 0.0 || e != e) u += e * d;
    }
    const double H = std::log(s) - u / s;
    const double p = cand >= 0 && cand < n ? std::exp((double) l[cand] * it - m) / s : 0.0;
    if (p_out) *p_out = p;
    if (h_out) *h_out = H;
    return cand >= 0 && cand < n && p > (double) TypicalAccept::threshold(eps, alpha, (float) H);
}
/// Host reference of the kernel row_entropy for one row (double sums; `top` = the row's argmax).
inline double row_entropy_ref(const float* l, int n, int top, double* top2 = nullptr) {
    const double m = l[top];
    double s = 0.0, u = 0.0, m2 = -INFINITY;
    for (int i = 0; i < n; ++i) {
        const double d = (double) l[i] - m;
        const double e = std::exp(d);
        s += e;
        if (e > 0.0 || e != e) u += e * d;
        if (i != top && (double) l[i] > m2) m2 = l[i];
    }
    if (top2) *top2 = std::exp(m2 - m) / s;
    return std::log(s) - u / s;
}

inline bool df2_gate_on() {
    static const bool v = [] { const char* e = std::getenv("STRATA_DF2_GATE"); return e != nullptr && e[0] == '1'; }();
    return v;
}
inline double df2_gate_row_ms() {
    static const double v = [] { const char* e = std::getenv("STRATA_DF2_ROW_MS"); return e != nullptr && e[0] != '\0' ? std::atof(e) : 3.0; }();
    return v;
}

}  // namespace strata::spec
