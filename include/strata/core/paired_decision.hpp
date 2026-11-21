#pragma once
// STRATA_TUNE_PAIRED (opt-in, default off; LEADS.md Part Q7, idea from Strata-HotRod include/strata/core/paired_test.hpp and
// docs/SPEC-tuner-paired-test.md, MIT; the code here is written from the method, nothing is copied): the decision routine for an
// on-line tuner that must not walk at random.
//
// A test compares a candidate setting with the base on ADJACENT runs.  Each pair gives d = ln(rate_candidate / rate_base).  The test
// moves to the candidate only when the mean of d beats the minimum effect delta = ln(1 + STRATA_TUNE_MIN_EFFECT) with ~97.7 % one-sided
// confidence (mean - z * se > delta, z = 2), keeps the base when the mean is below it with the same confidence (mean + z * se <
// delta) or at the pair cap, and otherwise asks for another pair.  No decision before `min_pairs` pairs.  The default outcome is
// KEEP.  The old rule it replaces (SwapBudget: `ms <= best_ms * 1.01`) is a point comparison with no variance and no sample size.
// Header-only, no GPU, no engine state (tests/program/swap_budget_paired_test.cpp runs the code the tuner runs).
#include <cmath>
#include <cstdlib>

namespace strata::core {

/// Welford mean and variance over the pair differences.
struct PairedStats {
    long long n = 0;
    double mean = 0.0;
    double m2 = 0.0;
    void add(double d) {
        ++n;
        const double dl = d - mean;
        mean += dl / (double) n;
        m2 += dl * (d - mean);
    }
    void reset() { n = 0; mean = 0.0; m2 = 0.0; }
    double var() const { return n > 1 ? m2 / (double) (n - 1) : 0.0; }
    double se() const { return n > 1 ? std::sqrt(var() / (double) n) : 0.0; }
};

struct PairedRule {
    long long min_pairs = 12;   ///< no decision before this many pairs
    long long cap = 160;        ///< at this many pairs without a decision: keep
    double delta = 0.0295588;   ///< the minimum effect, ln(1.03)
    double z = 2.0;
};

enum class PairedVerdict { Continue, Move, Keep };

inline PairedRule paired_rule(double min_effect, long long min_pairs = 12, long long cap = 160) {
    PairedRule r;
    r.delta = std::log(1.0 + (min_effect > 0.0 ? min_effect : 0.0));
    r.min_pairs = min_pairs < 2 ? 2 : min_pairs;
    r.cap = cap < r.min_pairs ? r.min_pairs : cap;
    return r;
}

/// One look after a pair.  `why` (optional) gets a short reason.
inline PairedVerdict paired_decide(const PairedStats& s, const PairedRule& r, const char** why = nullptr) {
    const char* w = "need more pairs";
    PairedVerdict v = PairedVerdict::Continue;
    if (s.n >= r.min_pairs) {
        const double half = r.z * s.se();
        if (s.mean - half > r.delta) {
            v = PairedVerdict::Move;
            w = "faster by more than the minimum effect";
        } else if (s.mean + half < r.delta) {
            v = PairedVerdict::Keep;
            w = "not faster by the minimum effect";
        } else if (s.n >= r.cap) {
            v = PairedVerdict::Keep;
            w = "pair cap reached";
        }
    }
    if (why != nullptr) *why = w;
    return v;
}

/// The per-pair value from two window times (lower = faster): ln(base_ms / cand_ms).  False when a time is not positive.
inline bool paired_diff_ms(double base_ms, double cand_ms, double& d) {
    if (!(base_ms > 0.0) || !(cand_ms > 0.0) || !std::isfinite(base_ms) || !std::isfinite(cand_ms)) return false;
    d = std::log(base_ms / cand_ms);
    return true;
}

/// STRATA_TUNE_PAIRED: unset, empty or 0 is off.
inline bool paired_on_from(const char* v) { return v != nullptr && v[0] != '\0' && std::atoi(v) != 0; }

}  // namespace strata::core
