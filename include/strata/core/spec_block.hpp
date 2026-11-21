// include/strata/core/spec_block.hpp - BLOCK VERIFICATION of a drafted MTP chain (STRATA_MTP_BLOCK_VERIFY=1, off by default;
// LEADS.md S12; Sun et al., "Block Verification Accelerates Speculative Decoding", arXiv 2403.10444).
//
// spec_prob.hpp judges the drafts token by token: draft i is kept with min(1, p(d_i) / q(d_i)) and the first reject ends
// the chain.  Block verification judges the whole block at once.  It keeps the same output law (every emitted token is
// distributed as the target chain p, whatever q is) and accepts at least as many drafts in expectation, because it may
// take a later prefix even when an earlier ratio was below 1.
//
// For a window with `gamma` drafts d[0..gamma-1] (draft j is the token at row j + 1), rows 0..gamma of the target
// distributions P (row j = p after the prefix d[0..j-1], the sampler's own post-chain list, as SpecP) and rows
// 0..gamma-1 of the drafter's Q (row j = the q that proposed d[j]; n < 0: a point mass at d[j], never q = 0):
//     w_0 = 1,  w_{j+1} = min(1, w_j * P_j(d_j) / Q_j(d_j))                                      (running ratio)
//     residual_j(x) = max(0, w_j * P_j(x) - Q_j(x))       (j < gamma);   residual_gamma = P_gamma  (the bonus row)
//     h_gamma = w_gamma;   h_j = S_j / (S_j + 1 - w_j),  S_j = sum_x residual_j(x)              (j < gamma; 1 when 0 / 0)
// Starting at j = gamma and going down, the first j whose uniform u_eta[j] < h_j wins: the first j drafts are kept and
// the next token is drawn from residual_j normalised (P_j itself when S_j = 0).  j = 0 always wins (w_0 = 1 gives
// h_0 = 1), so a window always yields at least one token.  The host's unchanged loop
//     `while (a < T - 1 && window[a + 1] == out[a]) ++a`
// takes exactly the kept drafts when out[j] = d[j] for j < accepted and out[accepted] = the drawn token, see fill_out.
//
// At temperature 0 (P a point mass at argmax_j) w_j is 1 while the drafts match and 0 after the first miss, h_j is 1 / 0,
// and the rule is the exact-greedy "longest matching prefix, then the argmax": tests/core spec_block_test checks this and
// the output law by exact enumeration of the rounds on a toy model (the Python check of the same formulas was exact to 1e-16)
// and by a fixed-seed chi-square on 3-token streams.
//
// Status: this is the pure rule, host-tested.  The device form (spec_block_dev.hpp: flat layout, per-entry arithmetic, a serial twin
// of the kernel flow tested against this file) runs in sampler.cu spec_block_rows_kernel + spec_block_verify_kernel, launched by
// verify.cpp only with STRATA_MTP_BLOCK_VERIFY=1 (LEADS.md S12; TEST_LEDGER.md T-1081: coded, not run on a GPU).
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "strata/core/spec_prob.hpp"

namespace strata::core {

/// STRATA_MTP_BLOCK_VERIFY as set (null: unset): on only for a value that starts with '1'.
inline bool spec_block_verify_on_from(const char* v) { return v != nullptr && v[0] == '1'; }
inline bool spec_block_verify_on() {
    static const bool on = spec_block_verify_on_from(std::getenv("STRATA_MTP_BLOCK_VERIFY"));
    return on;
}

inline constexpr int kSpecBlockMaxGamma = 16;   ///< the engine's windows hold at most 16 rows (kVerifyMaxT)

struct SpecBlockResult {
    int accepted;     ///< drafts kept (0..gamma)
    int32_t token;    ///< the token that follows them: a residual draw, or the bonus draw when accepted == gamma
};

namespace spec_block_detail {
inline double p_of(const SpecP& P, int32_t x) {
    for (int i = 0; i < P.n; ++i)
        if (P.id[i] == x) return P.p[i];
    return 0.0;
}
/// Q_j(x); a point mass (n < 0, or `d` not in the list) is 1 at d and 0 elsewhere.
inline double q_of(const SpecQ& Q, int32_t d, int32_t x) {
    double qd = -1.0;
    if (Q.n >= 0)
        for (int i = 0; i < Q.n; ++i)
            if (Q.id[i] == d) { qd = (double) Q.q[i]; break; }
    if (!(qd > 0.0)) return x == d ? 1.0 : 0.0;
    for (int i = 0; i < Q.n; ++i)
        if (Q.id[i] == x) return (double) Q.q[i];
    return 0.0;
}
}  // namespace spec_block_detail

/// The running ratios w[0..gamma] (w[0] = 1).
inline void spec_block_ratios(const SpecP* P, const SpecQ* Q, const int32_t* d, int gamma, double* w) {
    w[0] = 1.0;
    for (int j = 0; j < gamma; ++j) {
        const double pd = spec_block_detail::p_of(P[j], d[j]);
        const double qd = spec_block_detail::q_of(Q[j], d[j], d[j]);
        w[j + 1] = pd > 0.0 && qd > 0.0 ? std::min(1.0, w[j] * pd / qd) : 0.0;
    }
}

/// The residual weights of row j over P_j's support (n = P[j].n values in `r`, not normalised) and their sum S_j.
inline double spec_block_residual(const SpecP* P, const SpecQ* Q, const int32_t* d, int gamma, int j, double wj, double* r) {
    double s = 0.0;
    for (int i = 0; i < P[j].n; ++i) {
        double v;
        if (j == gamma) v = P[j].p[i];
        else {
            v = wj * P[j].p[i] - spec_block_detail::q_of(Q[j], d[j], P[j].id[i]);
            if (v < 0.0) v = 0.0;
        }
        r[i] = v;
        s += v;
    }
    return s;
}

/// The acceptance probability h_j of row j (see the header comment); `s` is spec_block_residual's sum.
inline double spec_block_h(int gamma, int j, double wj, double s) {
    if (j == gamma) return wj;
    const double den = s + 1.0 - wj;
    return den > 0.0 ? s / den : 1.0;
}

/// One block verification.  u_eta[0..gamma] are the acceptance uniforms (row j's own), u_resid the residual draw's.
/// P has gamma + 1 rows (each n <= 64), Q and d have gamma.  gamma may be 0 (a plain sample from P[0]).
inline SpecBlockResult spec_block_verify(const SpecP* P, const SpecQ* Q, const int32_t* d, int gamma, const double* u_eta,
                                         double u_resid);

/// The same rule with the uniforms asked for when they are needed, in the order the device kernel consumes them (spec_block_dev.hpp):
/// `eta(j)` for j = gamma down to 1, each only while no row has won yet (row 0 always wins and asks for none), then `resid(win)`
/// once.  spec_block_verify below is this with array lookups.
template <class Eta, class Resid>
inline SpecBlockResult spec_block_verify_with(const SpecP* P, const SpecQ* Q, const int32_t* d, int gamma, Eta&& eta, Resid&& resid) {
    double w[kSpecBlockMaxGamma + 1];
    double r[64];
    if (gamma > kSpecBlockMaxGamma) gamma = kSpecBlockMaxGamma;
    spec_block_ratios(P, Q, d, gamma, w);
    int win = 0;
    double s = 0.0;
    for (int j = gamma; j >= 0; --j) {
        s = spec_block_residual(P, Q, d, gamma, j, w[j], r);
        if (j == 0 || (double) eta(j) < spec_block_h(gamma, j, w[j], s)) { win = j; break; }
    }
    const int n = P[win].n < 64 ? P[win].n : 64;
    if (!(s > 0.0)) {   // P and Q agree (or an empty residual): draw from P itself
        s = 0.0;
        for (int i = 0; i < n; ++i) { r[i] = P[win].p[i]; s += r[i]; }
    }
    // the inverse-CDF pick over r / s (the sampler's own tail)
    double cum = 0.0;
    int last = -1;
    int32_t tok = P[win].id[n > 0 ? n - 1 : 0];
    const double u_resid = (double) resid(win);
    for (int i = 0; i < n; ++i) {
        if (!(r[i] > 0.0)) continue;
        last = i;
        cum += r[i] / s;
        if (u_resid < cum) { tok = P[win].id[i]; last = -2; break; }
    }
    if (last >= 0) tok = P[win].id[last];
    return {win, tok};
}

inline SpecBlockResult spec_block_verify(const SpecP* P, const SpecQ* Q, const int32_t* d, int gamma, const double* u_eta,
                                         double u_resid) {
    return spec_block_verify_with(P, Q, d, gamma, [&](int j) { return u_eta[j]; }, [&](int) { return u_resid; });
}

/// out[j] as the host loop expects it: the kept drafts, then the drawn token at out[accepted].  Returns accepted + 1 values written.
inline int spec_block_fill_out(const SpecBlockResult& r, const int32_t* d, int32_t* out) {
    for (int j = 0; j < r.accepted; ++j) out[j] = d[j];
    out[r.accepted] = r.token;
    return r.accepted + 1;
}

}  // namespace strata::core
