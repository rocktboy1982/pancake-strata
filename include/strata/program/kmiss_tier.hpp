#pragma once
// MOD_PLAN item 14 (phase 13 P1b), STRATA_ADAPT_KMISS=k,W (opt-in): the adaptive tier admits a missing expert once it
// missed in k of the last W rounds (W <= 8) and evicts the resident expert routed longest ago, instead of trading by the
// decayed usage gain.  tools/pancake/time_ceiling.py prices it at +5..16% of a code-edit window serially and +45..52%
// with deferral (model) - provided the copies do not stall the windows, so it belongs with the asynchronous tier.
// Placement only: outputs unchanged.  Used by both the blocking and the asynchronous serve tier (generate.cpp).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace strata::program {

class KMissTier {
public:
    static KMissTier from_env() {
        KMissTier t;
        const char* v = std::getenv("STRATA_ADAPT_KMISS");
        int k = 0, w = 0;
        if (v != nullptr && std::sscanf(v, "%d,%d", &k, &w) == 2 && k >= 1 && w >= k && w <= 8) { t.k_ = k; t.w_ = w; }
        else if (v != nullptr && v[0] != '\0')
            std::fprintf(stderr, "strata serve: STRATA_ADAPT_KMISS=%s is not k,W with 1 <= k <= W <= 8: off\n", v);
        return t;
    }
    bool on() const { return k_ > 0; }
    int k() const { return k_; }
    int w() const { return w_; }

    /// A round starts: the counts grown since the last round's end were routed in it; a non-resident one missed.
    /// Call before anything else raises the counts (the expert seeder).
    void observe(const std::vector<float>& usage, const int32_t* res) {
        const size_t n = usage.size();
        if (prev_.size() != n) { prev_.assign(n, 0.0f); miss_.assign(n, 0); last_.assign(n, 0); }
        ++round_;
        for (size_t i = 0; i < n; ++i) {
            miss_[i] = (uint8_t) (miss_[i] << 1);
            if (usage[i] > prev_[i] + 1e-6f) {
                last_[i] = round_;
                if (res[i] < 0) miss_[i] |= 1;
            }
        }
    }
    /// The round ends: the counts as they stand now (after the decay) are the next round's baseline.
    void end_round(const std::vector<float>& usage) { prev_ = usage; }

    /// Layer l's candidates (missed >= k times in the last W rounds, `allowed(e)`) and victims (resident, not routed this
    /// round), each paired best-first: the caller swaps cand[i] for vict[i].  Scores: misses then usage; age then usage.
    template <class Allowed>
    void select(int64_t l, int64_t n_expert, const float* u, const int32_t* r, Allowed allowed,
                std::vector<std::pair<float, int32_t>>& cand, std::vector<std::pair<float, int32_t>>& vict) const {
        cand.clear();
        vict.clear();
        const uint8_t mask = (uint8_t) ((1u << w_) - 1u);
        const size_t base = (size_t) (l * n_expert);
        for (int32_t e = 0; e < (int32_t) n_expert; ++e) {
            const size_t i = base + (size_t) e;
            if (r[e] < 0) {
                const int m = __builtin_popcount((unsigned) (miss_[i] & mask));
                if (m >= k_ && allowed(e)) cand.emplace_back(1000.0f * (float) m + u[e], e);
            } else if (last_[i] != round_) {
                vict.emplace_back(1000.0f * (float) (round_ - last_[i]) - u[e], e);   // larger = evict first
            }
        }
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        std::sort(vict.begin(), vict.end(), [](auto& a, auto& b) { return a.first > b.first; });
    }

private:
    int k_ = 0, w_ = 0;
    std::vector<float> prev_;
    std::vector<uint8_t> miss_;    // per (layer, expert): bit r = missed r rounds ago
    std::vector<uint32_t> last_;   // the last round routed
    uint32_t round_ = 0;
};

}  // namespace strata::program
