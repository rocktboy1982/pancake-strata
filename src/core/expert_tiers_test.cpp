// src/core/expert_tiers_test.cpp - STRATA_LIGHT_EXPERT_CPU's ranking (expert_tiers.hpp; tctien342/strata-x 5576b48144):
// off is the identity, on is a deterministic permutation that keeps the hottest pair first and each layer's own order.
#include "strata/core/expert_tiers.hpp"

#include <algorithm>
#include <cstdio>

using namespace strata::core;

int main() {
    int failed = 0, checks = 0;
    auto check = [&](bool ok, const char* what) {
        ++checks;
        if (!ok) { ++failed; std::fprintf(stderr, "FAIL %s\n", what); }
    };
    std::vector<std::pair<int32_t, int32_t>> prior;
    for (int i = 0; i < 120; ++i) prior.emplace_back(i % 3, i / 3);
    const auto saved = prior;
    auto preferred = [](int32_t layer) { return layer == 0; };

    const auto off = rank_light_experts(prior, preferred, false);
    check(off == saved && prior == saved, "off is the identity and does not change the prior");

    const auto on = rank_light_experts(prior, preferred, true);
    check(on.size() == prior.size(), "complete result");
    check(on.front() == prior.front(), "the hottest expert still wins GPU space");
    check(on != prior, "the preference changes the ranking");

    auto sorted = on, expected = prior;
    std::sort(sorted.begin(), sorted.end());
    std::sort(expected.begin(), expected.end());
    check(sorted == expected, "the result is a permutation of the input");

    check(on == rank_light_experts(prior, preferred, true), "deterministic");
    check(prior == rank_light_experts(prior, [](int32_t) { return false; }, true), "no eligible expert: identity");
    check(prior == rank_light_experts(prior, [](int32_t) { return true; }, true), "all eligible: identity");

    for (int32_t layer = 0; layer < 3; ++layer) {
        int32_t previous = -1;
        for (const auto& pair : on)
            if (pair.first == layer) {
                check(pair.second > previous, "within-layer demand order is preserved");
                previous = pair.second;
            }
    }

    std::printf("expert_tiers_test: %d checks, %d failures\n", checks, failed);
    return failed ? 1 : 0;
}
