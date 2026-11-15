// include/strata/spec/spec_reason.hpp - STRATA_SPEC_REASON=<tau> (opt-in, LOSSY): step-level speculation inside the thinking
// (LEADS.md section 3 row "SpecReason step-level speculation"; MOD_PLAN.md section 6 item 18).
//
// SpecReason (Pan et al., "SpecReason: Fast and Accurate Inference-Time Compute via Speculative Reasoning", arXiv 2504.07891):
// a small model proposes a whole reasoning step, the large model only scores it (a utility judgement, 0-9 in the paper) and the
// step is kept when the score clears a threshold; otherwise the large model writes the step itself.  Reasoning tolerates an
// approximate step where the final answer does not, so the gain (1.4-3.0x in the paper) comes from the target reading a step
// instead of generating it.  This box has one model family: the step is the verify window's drafts after the first mismatch
// (the MTP or lookup proposal), the target's score is its own likelihood of them, read from the window rows the verify pass
// computed anyway (row j - 1 scores draft j).  A step whose geometric-mean target probability is at least tau, with no draft
// below tau_floor, is kept whole (all or nothing, as the paper keeps or discards a step); only inside <think>, never past
// </think> or an end token.  Greedy requests only.  Unset: nothing here runs.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

namespace strata::spec {

/// STRATA_SPEC_REASON=<tau> in (0, 1) (read once): the step threshold; 0 = off.
inline double spec_reason_tau() {
    static const double v = [] {
        const char* e = std::getenv("STRATA_SPEC_REASON");
        const double x = e != nullptr ? std::atof(e) : 0.0;
        return x > 0.0 && x < 1.0 ? x : 0.0;
    }();
    return v;
}
/// STRATA_SPEC_REASON_FLOOR=<p> (read once, default tau / 4): no draft of a kept step may score below it.
inline double spec_reason_floor(double tau) {
    static const double v = [] {
        const char* e = std::getenv("STRATA_SPEC_REASON_FLOOR");
        return e != nullptr && e[0] != '\0' ? std::atof(e) : -1.0;
    }();
    return v >= 0.0 ? v : tau * 0.25;
}

/// softmax(logits)[tok] over n values, in double (host; the row the verify pass already holds).  NaN logits count as -inf.
inline double row_prob(const float* logits, size_t n, int32_t tok) {
    if (tok < 0 || (size_t) tok >= n) return 0.0;
    double m = -INFINITY;
    for (size_t i = 0; i < n; ++i)
        if (logits[i] == logits[i] && (double) logits[i] > m) m = logits[i];
    if (!(m > -INFINITY)) return 0.0;
    double s = 0.0;
    for (size_t i = 0; i < n; ++i)
        if (logits[i] == logits[i]) s += std::exp((double) logits[i] - m);
    const double lt = logits[tok];
    return lt == lt ? std::exp(lt - m) / s : 0.0;
}

/// The step p[0..n) (target probabilities of the drafts after the first mismatch): n when it is kept (geometric mean >= tau and
/// every p >= floor), else 0.
inline int spec_reason_accept(const double* p, int n, double tau, double floor) {
    if (n <= 0 || tau <= 0.0) return 0;
    double lsum = 0.0;
    for (int i = 0; i < n; ++i) {
        if (!(p[i] >= floor) || !(p[i] > 0.0)) return 0;
        lsum += std::log(p[i]);
    }
    return lsum / n >= std::log(tau) ? n : 0;
}

}  // namespace strata::spec
