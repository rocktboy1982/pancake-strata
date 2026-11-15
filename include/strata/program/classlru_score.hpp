// include/strata/program/classlru_score.hpp - STRATA_ADAPT_CLASSLRU=C (opt-in; unset or 0 = the tier reads the decayed usage
// counts as always).
//
// Upstream llama.cpp's ClassLRU expert cache (ggml-org/llama.cpp c4fb57969, src/llama-moe-cache.cpp; LEADS.md Part D
// "ClassLRU decay"; replayed offline by tools/replay_classlru.py) as a score the adaptive tier can rank by: per (layer,
// expert) the class = min(C - 1, floor(log2(CRF))) of its decayed use count (our usage counts are the CRF: decayed by
// --adapt-decay each round), and the victim is the slot with the largest age / 2^class.  The tier swaps a candidate in
// when its score beats the victim's, so the score is 2^class / (1 + age), age = tier rounds since the expert was last
// routed (an expert routed in this round has age 0).  Placement only: outputs unchanged.
//
// transform() rewrites a COPY of the counts the tier then ranks (the async tier's snapshot, or the blocking tier's
// counts, restored by the caller before the decay).  Env read once.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace strata::program {

struct ClassLruScore {
    int classes = 0;                  ///< C; 0 = off
    float decay = 0.7f;               ///< --adapt-decay (to tell a new use from the decay of an old count)
    uint64_t round = 0;
    std::vector<float> last_raw;      ///< the raw counts at the previous transform
    std::vector<uint64_t> last_use;   ///< the round an expert was last routed

    static ClassLruScore from_env(float adapt_decay) {
        ClassLruScore c;
        static const int n = [] {
            const char* e = std::getenv("STRATA_ADAPT_CLASSLRU");
            const int v = e ? std::atoi(e) : 0;
            return v <= 0 ? 0 : (v > 30 ? 30 : v);
        }();
        c.classes = n;
        c.decay = adapt_decay;
        return c;
    }
    bool on() const { return classes > 0; }

    /// `u` (the raw decayed counts) -> the ClassLRU keep score, in place.
    void transform(std::vector<float>& u) {
        if (!on()) return;
        ++round;
        if (last_raw.size() != u.size()) {
            last_raw.assign(u.size(), 0.0f);
            last_use.assign(u.size(), 0);
        }
        for (size_t i = 0; i < u.size(); ++i) {
            const float raw = u[i];
            if (raw > last_raw[i] * decay + 1e-6f) last_use[i] = round;   // routed since the last round
            last_raw[i] = raw;
            int cls = raw >= 1.0f ? (int) std::floor(std::log2(raw)) : 0;
            if (cls > classes - 1) cls = classes - 1;
            const double age = last_use[i] == 0 ? (double) round : (double) (round - last_use[i]);
            u[i] = raw > 0.0f ? (float) (std::ldexp(1.0, cls) / (1.0 + age)) : 0.0f;
        }
    }
};

}  // namespace strata::program
