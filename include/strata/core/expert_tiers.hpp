// include/strata/core/expert_tiers.hpp - STRATA_LIGHT_EXPERT_CPU (opt-in; port of tctien342/strata-x 5576b48144
// "--light-expert-cpu", LEADS.md FK-34): a soft placement preference for experts a CPU computes well.
//
// The demand prior (--expert-profile) is a ranking of (layer, expert) pairs, hottest first.  An eligible pair's score
// (rank^-1.2, a Zipf-like demand curve) is scaled by `weight` (0.75), so it moves behind an equally hot ineligible one
// while a hot eligible expert still wins GPU space.  Eligibility is the caller's predicate on the layer; this file only
// orders: routed experts, weights and quantization never change.  Disabled: the prior as given.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace strata::core {

template <class Eligible>
std::vector<std::pair<int32_t, int32_t>> rank_light_experts(const std::vector<std::pair<int32_t, int32_t>>& prior,
                                                            Eligible eligible, bool enabled, double weight = 0.75) {
    if (!enabled) return prior;
    struct Ranked { size_t rank; double score; };
    std::vector<Ranked> ranked;
    ranked.reserve(prior.size());
    for (size_t i = 0; i < prior.size(); ++i)
        ranked.push_back({i, (eligible(prior[i].first) ? weight : 1.0) * std::pow(double(i + 1), -1.2)});
    std::sort(ranked.begin(), ranked.end(), [](const Ranked& a, const Ranked& b) {
        return a.score != b.score ? a.score > b.score : a.rank < b.rank;
    });
    std::vector<std::pair<int32_t, int32_t>> result;
    result.reserve(prior.size());
    for (const auto& row : ranked) result.push_back(prior[row.rank]);
    return result;
}

}  // namespace strata::core
