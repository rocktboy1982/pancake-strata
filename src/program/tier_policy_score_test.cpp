// LEADS.md B11 / B126 / A46: the tier policy scores (include/strata/program/tier_policy_score.hpp), CPU only, no model.
// Each policy is built by hand (from_env reads the switches once) and fed a synthetic routing trace.
#include "strata/program/tier_policy_score.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using strata::program::TierPolicyScore;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
// one tier round: decay, then +1 for the routed experts
void route(std::vector<float>& u, const std::vector<int>& used, float decay) {
    for (float& v : u) v *= decay;
    for (int e : used) u[(size_t) e] += 1.0f;
}
}  // namespace

int main() {
    const float decay = 0.7f;
    {   // off: nothing changes
        TierPolicyScore t;
        std::vector<float> u = {1.0f, 2.0f, 3.0f};
        const std::vector<int32_t> res = {0, -1, 1};
        t.transform(u, res.data());
        check(u[0] == 1.0f && u[1] == 2.0f && u[2] == 3.0f, "off: the counts as they were");
    }
    {   // SIEVE: a resident routed since the last round is kept over an unvisited one with a larger count
        TierPolicyScore t;
        t.sieve = 1;
        t.decay = decay;
        std::vector<float> u(4, 0.0f);
        const std::vector<int32_t> res = {0, 1, -1, -1};
        route(u, {0, 1}, decay);         // round A: both residents routed
        std::vector<float> s = u;
        t.transform(s, res.data());
        route(u, {1}, decay);            // round B: only expert 1
        s = u;
        t.transform(s, res.data());
        check(s[0] < 1.0f && s[1] > 1000.0f, "SIEVE: the resident not routed in the round leaves first");
        route(u, {0}, decay);
        s = u;
        t.transform(s, res.data());
        check(s[0] > 1000.0f, "SIEVE: a visited resident is kept");
        check(s[2] == u[2], "SIEVE: a candidate keeps its count");
    }
    {   // S3-FIFO: a new resident not routed again leaves before an old reused one
        TierPolicyScore t;
        t.sieve = 2;
        t.decay = decay;
        std::vector<float> u(3, 0.0f);
        std::vector<int32_t> res = {0, -1, -1};
        for (int r = 0; r < 4; ++r) {
            route(u, {0}, decay);
            std::vector<float> s = u;
            t.transform(s, res.data());
        }
        res[1] = 1;                      // expert 1 admitted now
        route(u, {0}, decay);
        std::vector<float> s = u;
        t.transform(s, res.data());
        check(s[0] > s[1], "S3-FIFO: the reused old resident outranks the new one");
    }
    {   // forecast: a regular expert (every round) outranks a rare one (every 4th round)
        TierPolicyScore t;
        t.horizon = 2.0f;
        t.decay = decay;
        std::vector<float> u(2, 0.0f);
        const std::vector<int32_t> res = {0, 1};
        std::vector<float> s;
        for (int r = 0; r < 16; ++r) {
            std::vector<int> used = {0};
            if (r % 4 == 0) used.push_back(1);
            route(u, used, decay);
            s = u;
            t.transform(s, res.data());
        }
        check(s[0] > s[1], "forecast: the regular expert has the higher reuse probability");
        check(s[0] <= 1.0f / (1.0f - decay) + 1e-4f, "forecast: the score is a scaled probability");
    }
    {   // learned: after training on a trace where expert 0 is always routed and 1 never, 0 scores higher
        TierPolicyScore t;
        t.lr = 1.0f;
        t.decay = decay;
        std::vector<float> u(2, 0.0f);
        const std::vector<int32_t> res = {0, 1};
        std::vector<float> s;
        for (int r = 0; r < 64; ++r) {
            route(u, {0}, decay);
            s = u;
            t.transform(s, res.data());
        }
        check(s[0] > s[1], "learned: the always-routed expert scores higher");
        check(s[0] > 0.0f && s[0] <= 1.0f / (1.0f - decay) + 1e-4f, "learned: the score is a scaled probability");
    }
    std::printf("tier_policy_score_test: %d checks passed\n", checks);
    return 0;
}
