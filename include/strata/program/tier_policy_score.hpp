// include/strata/program/tier_policy_score.hpp - three opt-in cache policies for the adaptive expert tier, each a score the
// tier ranks by in place of the raw decayed usage counts (the same hook as ClassLRU, classlru_score.hpp).  Placement only:
// which experts sit in VRAM changes, the outputs do not.  Env read once; unset = the tier reads the raw counts.  Set one
// policy at a time (ClassLRU, when also set, transforms first).
//
//   STRATA_ADAPT_SIEVE=1|2   LEADS.md B11.  1 = SIEVE (Zhang et al., NSDI'24 "SIEVE is simpler than LRU"): a resident
//                            expert routed since the hand last passed is kept (its visited bit), an unvisited one leaves,
//                            oldest admission first; the hand clears the visited bits every tier round.  2 = S3-FIFO
//                            (Yang et al., SOSP'23): an expert admitted less than STRATA_ADAPT_SIEVE_SMALL rounds ago
//                            (default 2: the small FIFO) that was not routed again leaves first (quick demotion); older
//                            residents keep a frequency 0..3 (main FIFO with reinsertion); a candidate evicted within the
//                            last 16 rounds (the ghost queue) is admitted at twice its count.
//   STRATA_ADAPT_FORECAST=H  LEADS.md B126 (SeqMoE: learned multi-step forecaster + probabilistic Belady eviction).  Per
//                            expert an exponentially smoothed inter-arrival gap (in tier rounds) is the forecaster; the
//                            score is P(routed within the next H rounds) = 1 - exp(-H / gap), with the gap stretched to
//                            the time since the last use once that is longer (a stale expert); the tier keeps the experts
//                            most likely to be used soon, which is Belady's rule on the forecast distribution.
//   STRATA_ADAPT_LEARNED=lr  LEADS.md A46 (learned cache policy).  An online logistic model per tier, features (1, the
//                            decayed count, 1 / (1 + age), the hits in the last 8 rounds / 8, resident), trained each round
//                            on whether the expert was routed in that round (SGD, learning rate lr), predicts the next
//                            round's use; the score is that probability.
//
// The probabilities are scaled to the usage unit (x 1 / (1 - --adapt-decay), the steady count of an expert routed every
// round), so the blocking tier's admission floor (a count of 2) keeps its meaning.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::program {

struct TierPolicyScore {
    int sieve = 0;            ///< STRATA_ADAPT_SIEVE: 0 off, 1 SIEVE, 2 S3-FIFO
    int small_rounds = 2;     ///< STRATA_ADAPT_SIEVE_SMALL (S3-FIFO's small queue, in rounds)
    float horizon = 0.0f;     ///< STRATA_ADAPT_FORECAST: 0 off
    float lr = 0.0f;          ///< STRATA_ADAPT_LEARNED: 0 off
    float decay = 0.7f;       ///< --adapt-decay
    uint64_t round = 0;
    std::vector<float> last_raw;
    std::vector<uint64_t> last_use, admitted, evicted;
    std::vector<uint8_t> visited, freq, hist;
    std::vector<int8_t> was_res;
    std::vector<float> gap;
    std::vector<float> feat;  ///< the learned policy's features of the previous round (5 per expert)
    float w[5] = {-1.0f, 0.0f, 1.0f, 1.0f, 0.0f};

    static TierPolicyScore from_env(float adapt_decay) {
        static const TierPolicyScore proto = [] {
            TierPolicyScore p;
            if (const char* v = std::getenv("STRATA_ADAPT_SIEVE")) { const int x = std::atoi(v); p.sieve = x == 1 || x == 2 ? x : 0; }
            if (const char* v = std::getenv("STRATA_ADAPT_SIEVE_SMALL")) { const int x = std::atoi(v); if (x >= 1 && x <= 64) p.small_rounds = x; }
            if (const char* v = std::getenv("STRATA_ADAPT_FORECAST")) { const double x = std::atof(v); if (x > 0.0 && std::isfinite(x)) p.horizon = (float) x; }
            if (const char* v = std::getenv("STRATA_ADAPT_LEARNED")) { const double x = std::atof(v); if (x > 0.0 && x <= 10.0) p.lr = (float) x; }
            const int n = (p.sieve ? 1 : 0) + (p.horizon > 0.0f ? 1 : 0) + (p.lr > 0.0f ? 1 : 0);
            if (n > 1) {
                std::fprintf(stderr, "strata serve: STRATA_ADAPT_SIEVE / _FORECAST / _LEARNED: set one; using the first set in that order\n");
                if (p.sieve) { p.horizon = 0.0f; p.lr = 0.0f; } else if (p.horizon > 0.0f) p.lr = 0.0f;
            }
            if (n > 0) std::fprintf(stderr, "strata serve: tier policy %s\n", p.sieve == 1 ? "SIEVE (STRATA_ADAPT_SIEVE=1)"
                                    : p.sieve == 2 ? "S3-FIFO (STRATA_ADAPT_SIEVE=2)" : p.horizon > 0.0f ? "forecast (STRATA_ADAPT_FORECAST)"
                                    : "learned (STRATA_ADAPT_LEARNED)");
            return p;
        }();
        TierPolicyScore t = proto;
        t.decay = adapt_decay;
        return t;
    }
    bool on() const { return sieve != 0 || horizon > 0.0f || lr > 0.0f; }

    /// `u` (the raw decayed counts, one per (layer, expert)) -> the policy's keep score, in place; `res[i]` >= 0 = resident.
    void transform(std::vector<float>& u, const int32_t* res) {
        if (!on()) return;
        const size_t n = u.size();
        ++round;
        if (last_raw.size() != n) {
            last_raw.assign(n, 0.0f);
            last_use.assign(n, 0);
            admitted.assign(n, 0);
            evicted.assign(n, 0);
            visited.assign(n, 0);
            freq.assign(n, 0);
            hist.assign(n, 0);
            was_res.assign(n, -1);
            gap.assign(n, 0.0f);
            feat.assign(n * 5, 0.0f);
        }
        const float scale = decay < 0.999f ? 1.0f / (1.0f - decay) : 1000.0f;
        const float big = 1.0e6f;
        if (lr > 0.0f && round > 1) {   // A46: learn from the last round's features -> was it routed in this round
            for (size_t i = 0; i < n; ++i) {
                const float* x = &feat[i * 5];
                const float y = u[i] > last_raw[i] * decay + 1e-6f ? 1.0f : 0.0f;
                float z = 0.0f;
                for (int k = 0; k < 5; ++k) z += w[k] * x[k];
                const float p = 1.0f / (1.0f + std::exp(-z));
                const float g = (y - p) * lr / (float) n * 64.0f;
                for (int k = 0; k < 5; ++k) w[k] += g * x[k];
            }
        }
        for (size_t i = 0; i < n; ++i) {
            const float raw = u[i];
            const bool used = raw > last_raw[i] * decay + 1e-6f;   // routed since the last round
            const bool r = res != nullptr && res[i] >= 0;
            if (r && was_res[i] != 1) admitted[i] = round;
            if (!r && was_res[i] == 1) evicted[i] = round;
            was_res[i] = r ? 1 : 0;
            hist[i] = (uint8_t) ((hist[i] << 1) | (used ? 1 : 0));
            if (used) {
                if (last_use[i] != 0) {
                    const float g = (float) (round - last_use[i]);
                    gap[i] = gap[i] > 0.0f ? 0.7f * gap[i] + 0.3f * g : g;
                }
                last_use[i] = round;
                visited[i] = 1;
                if (freq[i] < 3) ++freq[i];
            }
            last_raw[i] = raw;
            const float age = last_use[i] == 0 ? (float) round : (float) (round - last_use[i]);
            if (sieve == 1) {
                if (!r) continue;                                        // a candidate keeps its count
                const float ins_age = (float) (round - admitted[i]);
                u[i] = visited[i] ? raw + big : 1e-3f / (1.0f + ins_age);   // unvisited: the oldest admission leaves first
                visited[i] = 0;                                          // the hand passes every round
            } else if (sieve == 2) {
                if (!r) {
                    if (evicted[i] != 0 && round - evicted[i] <= 16) u[i] = 2.0f * raw;   // ghost hit: straight to main
                    continue;
                }
                const bool small = round - admitted[i] < (uint64_t) small_rounds;
                const bool reused = last_use[i] > admitted[i];
                if (small) u[i] = reused ? raw + big : raw;              // small FIFO: not re-routed = leaves first
                else u[i] = freq[i] > 0 ? raw + big * (float) freq[i] : 1e-3f / (1.0f + (float) (round - admitted[i]));
                if (!small && freq[i] > 0) --freq[i];                    // main FIFO: reinsertion costs one
            } else if (horizon > 0.0f) {
                if (last_use[i] == 0) { u[i] = 0.0f; continue; }
                float gg = gap[i] > 0.0f ? gap[i] : 1.0f;
                if (age > gg) gg = age;                                  // stale: the gap is at least the time since
                u[i] = scale * (1.0f - std::exp(-horizon / gg));
            } else {
                float* x = &feat[i * 5];
                x[0] = 1.0f;
                x[1] = raw / scale;
                x[2] = 1.0f / (1.0f + age);
                int hits = 0;
                for (int b = 0; b < 8; ++b) hits += hist[i] >> b & 1;
                x[3] = (float) hits / 8.0f;
                x[4] = r ? 1.0f : 0.0f;
                float z = 0.0f;
                for (int k = 0; k < 5; ++k) z += w[k] * x[k];
                u[i] = scale / (1.0f + std::exp(-z));
            }
        }
    }
};

}  // namespace strata::program
