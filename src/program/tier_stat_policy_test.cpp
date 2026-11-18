// MOD_PLAN.md section 6, LEADS.md B143 / B144: STRATA_ADAPT_FILL_FIRST and STRATA_ADAPT_TRANSITION
// (include/strata/program/tier_stat_policy.hpp), CPU only, no model.  Synthetic routing, deterministic.
#include "strata/program/tier_stat_policy.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using strata::program::TierStat;

namespace {
void check(bool ok, const char* label) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
constexpr float kDecay = 0.7f;
// one tier round's counts: decay, then +1 for each routed (layer, expert)
void route(std::vector<float>& raw, int64_t E, int64_t layer, const std::vector<int>& used) {
    for (int e : used) raw[(size_t) (layer * E + e)] += 1.0f;
}
void decay_all(std::vector<float>& raw) { for (float& v : raw) v *= kDecay; }
TierStat make(const TierStat::Params& p, int64_t L, int64_t E, int64_t slots) {
    TierStat t;
    t.configure(p, kDecay, L, E, slots, false);
    return t;
}
}  // namespace

int main() {
    {   // off: nothing changes, nothing allocated
        TierStat t = make(TierStat::Params{}, 2, 8, 8);
        check(!t.on(), "off by default");
        std::vector<float> raw(16, 1.0f), u = raw;
        const std::vector<int32_t> res(16, 0);
        t.observe(raw, res.data(), kDecay);
        t.apply(u, res.data());
        check(u == raw && t.rounds() == 0, "off: the counts as they were");
    }
    {   // f = 1 is a no-op; per-layer admission turns the fill off
        TierStat::Params p;
        p.fill_f = 1.0;
        check(!make(p, 2, 8, 8).on(), "fill_first=1 is no empty slot: off");
        p.fill_f = 0.5;
        TierStat t;
        t.configure(p, kDecay, 2, 8, 8, true);
        check(!t.on(), "per-layer admission: fill off");
    }
    {   // FILL_FIRST: layer 0 has 3 claimed + 1 empty slot, layer 1 has 4 empty slots (slot order = rank order)
        TierStat::Params p;
        p.fill_f = 3.0 / 8.0;   // core 3 slots
        TierStat t = make(p, 2, 8, 8);
        check(t.fill_on() && !t.trans_on(), "fill on");
        std::vector<int32_t> res(16, -1);
        for (int e = 0; e < 4; ++e) { res[(size_t) e] = e; res[(size_t) (8 + e)] = 4 + e; }
        std::vector<float> raw(16, 0.0f);
        raw[5] = 4.0f; raw[6] = 3.0f; raw[7] = 0.5f;
        raw[8 + 5] = 3.0f; raw[8 + 6] = 2.5f; raw[8 + 7] = 1.2f;
        t.observe(raw, res.data(), kDecay);
        check(t.empty_slots() == 5, "5 empty slots: layer 0 expert 3 and layer 1 experts 0..3");
        std::vector<float> u = raw;
        t.apply(u, res.data());
        check(u[5] == 5.0f, "layer 0: the best miss is admitted (count + 1) into the one empty slot");
        check(u[6] == 0.0f && u[7] == 0.0f, "layer 0: the other misses wait (no claimed resident is evicted)");
        check(u[3] == -1.0f && u[0] == raw[0] && u[1] == raw[1] && u[2] == raw[2], "layer 0: only the empty slot is a victim");
        check(u[8 + 5] == 4.0f && u[8 + 6] == 3.5f && u[8 + 7] >= 2.0f, "layer 1: three misses fill three of four empty slots");
        for (int e = 0; e < 4; ++e) check(u[(size_t) (8 + e)] == -1.0f, "layer 1: all residents are empty slots");
        // a miss with a count below 1 is not admitted on a first miss
        std::vector<float> raw2(16, 0.0f);
        raw2[8 + 5] = 0.4f;
        TierStat t2 = make(p, 2, 8, 8);
        t2.observe(raw2, res.data(), kDecay);
        std::vector<float> u2 = raw2;
        t2.apply(u2, res.data());
        check(u2[8 + 5] < 2.0f, "count below 1: not eligible");
        // next round: expert 3 of layer 0 is routed -> claimed; the layer is full, the normal rule is back
        decay_all(raw);
        route(raw, 8, 0, {3});
        t.observe(raw, res.data(), kDecay);
        check(t.empty_slots() == 4, "a routed placeholder is claimed");
        u = raw;
        t.apply(u, res.data());
        check(u[5] == raw[5] && u[6] == raw[6] && u[3] == raw[3], "layer 0 full: counts untouched, normal rule");
        // an arrival is claimed: expert 5 enters layer 0 in place of the claimed 0, the layer stays full
        res[5] = res[0];
        res[0] = -1;
        decay_all(raw);
        t.observe(raw, res.data(), kDecay);
        u = raw;
        t.apply(u, res.data());
        check(u[5] == raw[5], "an arrived expert is claimed, layer 0 still full");
    }
    {   // TRANSITION: A = {0,1,4}, B = {2,3,4} alternate; after B the matrix predicts A
        TierStat::Params p;
        p.trans_w = 1.0;
        p.trans_decay = 0.95;
        TierStat t = make(p, 1, 8, 4);
        check(t.trans_on() && !t.fill_on(), "transition on");
        std::vector<float> raw(8, 0.0f);
        std::vector<int32_t> res(8, -1);
        res[0] = 0; res[1] = 1; res[2] = 2; res[5] = 3;   // residents 0, 1, 2 and 5
        const std::vector<int> A = {0, 1, 4}, B = {2, 3, 4};   // expert 4 is in both: consecutive rounds overlap by 1 of 5
        for (int r = 0; r < 12; ++r) {
            decay_all(raw);
            route(raw, 8, 0, r % 2 == 0 ? A : B);
            t.observe(raw, res.data(), kDecay);
        }
        // the last round routed B (r = 11): the state is B, the next is A
        std::vector<float> u = raw;
        t.apply(u, res.data());
        check(u[0] - raw[0] > 1.5f && u[1] - raw[1] > 1.5f, "after B the residents 0, 1 (A) are protected");
        check(u[2] - raw[2] < 1.0f && u[5] == raw[5] && u[0] - raw[0] > 2.0f * (u[2] - raw[2]), "expert 2 (B, not next) gets far less than 0, 1; 5 (never used) nothing");
        check(t.switch_prob() < 0.5, "a regular alternation is not a switch");
        // a boundary: the prediction falls back to the overall use rate
        decay_all(raw);
        route(raw, 8, 0, A);
        t.boundary();
        t.observe(raw, res.data(), kDecay);
        check(t.switch_prob() >= 0.99, "boundary: switch probability 1");
        u = raw;
        t.apply(u, res.data());
        const float boost_boundary = u[2] - raw[2];   // 2 is used in half of the rounds
        check(boost_boundary > 1.0f && boost_boundary < 2.2f, "boundary: the overall rate (about 0.5) replaces the matrix");
        // a disjoint set is a switch without any boundary call
        TierStat t3 = make(p, 1, 8, 4);
        std::vector<float> raw3(8, 0.0f);
        for (int r = 0; r < 8; ++r) { decay_all(raw3); route(raw3, 8, 0, r % 2 == 0 ? A : B); t3.observe(raw3, res.data(), kDecay); }
        const double before = t3.switch_prob();
        decay_all(raw3);
        route(raw3, 8, 0, {6, 7});
        t3.observe(raw3, res.data(), kDecay);
        check(t3.switch_prob() > before + 0.2, "a routed set that shares nothing with the last round is detected as a switch");
    }
    {   // both rules on, a full-size run: cost per round (48 layers x 512 experts, about 100 experts routed per layer)
        TierStat::Params p;
        p.fill_f = 0.6;
        p.trans_w = 2.0;
        const int64_t L = 48, E = 512, slots = 10909;
        TierStat t = make(p, L, E, slots);
        std::vector<int32_t> res((size_t) (L * E), -1);
        for (int64_t s = 0; s < slots; ++s) res[(size_t) (s * (L * E) / slots)] = (int32_t) s;
        std::vector<float> raw((size_t) (L * E), 0.0f);
        uint64_t x = 88172645463325252ull;
        auto rnd = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return x; };
        const int rounds = 24;
        for (int r = 0; r < rounds; ++r) {
            decay_all(raw);
            for (int64_t l = 0; l < L; ++l)
                for (int k = 0; k < 100; ++k) raw[(size_t) (l * E) + (size_t) (rnd() % 160)] += 1.0f;   // a hot core of 160 experts
            t.observe(raw, res.data(), kDecay);
            std::vector<float> u = raw;
            t.apply(u, res.data());
        }
        std::fprintf(stderr, "tier_stat_policy_test: %.0f us per round (observe + apply), 48 x 512, about 100 routed per layer\n", t.mean_us());
        check(t.rounds() == (uint64_t) rounds && t.mean_us() < 200000.0, "the cost per round is bounded");
    }
    std::printf("tier_stat_policy_test: ok\n");
    return 0;
}
