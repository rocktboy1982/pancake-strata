// LEADS.md B145 / B146: STRATA_ADAPT_VICTIM_UNUSED and STRATA_ADAPT_DIRECT_SWAP (include/strata/program/tier_victim.hpp),
// CPU only, no model.  Synthetic routing, deterministic.
#include "strata/program/tier_victim.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using strata::program::TierVictim;
using Vic = std::vector<std::pair<float, int32_t>>;

namespace {
void check(bool ok, const char* label) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
constexpr float kDecay = 0.7f;
void route(std::vector<float>& raw, int64_t E, int64_t layer, const std::vector<int>& used) {
    for (int e : used) raw[(size_t) (layer * E + e)] += 1.0f;
}
void decay_all(std::vector<float>& raw) { for (float& v : raw) v *= kDecay; }
TierVictim make(int k, bool skip, bool direct, int64_t L, int64_t E) {
    TierVictim::Params p;
    p.unused_k = k;
    p.skip = skip;
    p.direct = direct;
    TierVictim t;
    t.configure(p, L, E);
    return t;
}
void set_env(const char* k, const char* v) {
#if defined(_WIN32)
    _putenv_s(k, v);
#else
    setenv(k, v, 1);
#endif
}
// the swap loop of generate.cpp adapt(), non-kmiss path, for one layer: the residents' scores, `n_cand` candidates
Vic pick(TierVictim& t, int64_t layer, const std::vector<float>& u, const std::vector<int>& resident, size_t n_cand) {
    Vic vict;
    for (int e : resident) vict.emplace_back(u[(size_t) e], (int32_t) e);
    if (t.unused_on()) t.arrange(layer, vict, n_cand);
    const size_t nc = std::min(n_cand, vict.size());
    if (t.unused_on())
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(), [&](auto& a, auto& b) { return t.before(layer, a, b); });
    else
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(), [](auto& a, auto& b) { return a.first < b.first; });
    vict.resize(nc);
    return vict;
}
}  // namespace

int main() {
    {   // off by default: nothing allocated, the baseline comparator
        TierVictim t = make(0, false, false, 2, 8);
        check(!t.on() && !t.unused_on() && !t.direct_on(), "off by default");
        std::vector<float> raw(16, 3.0f);
        t.observe(raw, kDecay);
        check(t.rounds() == 0 && !t.recent(0, 1), "off: observe does nothing");
        check(t.before(0, {1.0f, 3}, {2.0f, 1}) && !t.before(0, {2.0f, 3}, {1.0f, 1}), "off: the score alone");
        Vic v = {{5.0f, 1}, {1.0f, 2}};
        t.order_given(0, v, 2);
        t.arrange(0, v, 2);
        check(v.size() == 2 && v[0].second == 1, "off: the list is untouched");
    }
    {   // the env: unset = off; flag, K, fallback, invalid values
        set_env("STRATA_ADAPT_VICTIM_UNUSED", "");
        set_env("STRATA_ADAPT_DIRECT_SWAP", "");
        auto p = TierVictim::params_from_env();
        check(p.unused_k == 0 && !p.direct && !p.skip, "env unset: off");
        set_env("STRATA_ADAPT_VICTIM_UNUSED", "1");
        set_env("STRATA_ADAPT_VICTIM_K", "");
        set_env("STRATA_ADAPT_VICTIM_FALLBACK", "");
        p = TierVictim::params_from_env();
        check(p.unused_k == 4 && !p.skip, "default K = 4, fallback rule");
        set_env("STRATA_ADAPT_VICTIM_K", "8");
        set_env("STRATA_ADAPT_VICTIM_FALLBACK", "skip");
        set_env("STRATA_ADAPT_DIRECT_SWAP", "1");
        p = TierVictim::params_from_env();
        check(p.unused_k == 8 && p.skip && p.direct, "K = 8, skip, direct");
        set_env("STRATA_ADAPT_VICTIM_K", "0");
        set_env("STRATA_ADAPT_VICTIM_FALLBACK", "bogus");
        p = TierVictim::params_from_env();
        check(p.unused_k == 4 && !p.skip, "invalid K and fallback: the defaults");
        set_env("STRATA_ADAPT_VICTIM_UNUSED", "0");
        p = TierVictim::params_from_env();
        check(p.unused_k == 0 && p.direct, "VICTIM_UNUSED=0: off, K ignored; DIRECT_SWAP alone works");
        set_env("STRATA_ADAPT_VICTIM_UNUSED", "");
        set_env("STRATA_ADAPT_VICTIM_K", "");
        set_env("STRATA_ADAPT_VICTIM_FALLBACK", "");
        set_env("STRATA_ADAPT_DIRECT_SWAP", "");
    }
    {   // used = routed since the last round; K rounds of history; never used = unused
        const int64_t E = 8;
        TierVictim t = make(2, false, false, 1, E);
        std::vector<float> raw(8, 0.0f);
        check(!t.recent(0, 3), "no history: nothing is recent");
        route(raw, E, 0, {1, 2});
        t.observe(raw, kDecay);                      // round 1: 1, 2 used
        check(t.recent(0, 1) && t.recent(0, 2) && !t.recent(0, 3), "round 1: the routed ones are recent, the others unused");
        decay_all(raw); route(raw, E, 0, {1});
        t.observe(raw, kDecay);                      // round 2: 1 used, 2 not
        check(t.recent(0, 1) && t.recent(0, 2), "K = 2 round 2: 2 was used one round ago, still recent");
        decay_all(raw); route(raw, E, 0, {4});
        t.observe(raw, kDecay);                      // round 3: 4 used; 2 last used in round 1 = two rounds ago
        check(!t.recent(0, 2), "K = 2 round 3: 2 was last used two rounds ago -> unused");
        check(t.recent(0, 1) && t.recent(0, 4), "1 used in round 2 and 4 in round 3 stay recent");
        decay_all(raw);
        t.observe(raw, kDecay);                      // round 4: nothing routed
        check(t.recent(0, 4) && !t.recent(0, 1), "round 4: 4 (round 3) recent, 1 (round 2) unused");
    }
    {   // K = 1: the previous round's experts are never victims (skip), and come last (rule)
        const int64_t E = 8;
        std::vector<float> raw(8, 0.0f);
        TierVictim ts = make(1, true, false, 1, E), tr = make(1, false, false, 1, E);
        route(raw, E, 0, {0, 1, 2, 3});
        ts.observe(raw, kDecay); tr.observe(raw, kDecay);   // 0..3 used; residents 0..5, 4 and 5 unused
        std::vector<float> u(raw);
        u[4] = 5.0f; u[5] = 4.0f;                            // the unused ones have the HIGHER score
        const std::vector<int> res = {0, 1, 2, 3, 4, 5};
        Vic a = pick(ts, 0, u, res, 4);
        check(a.size() == 2 && a[0].second == 5 && a[1].second == 4, "skip: only the two unused residents are victims, lowest score first");
        Vic b = pick(tr, 0, u, res, 4);
        check(b.size() == 4 && b[0].second == 5 && b[1].second == 4, "rule: the unused first");
        check(b[2].first <= b[3].first && b[2].second != 4 && b[2].second != 5, "rule: then the used ones by score");
        check(ts.skipped() == 2, "skip counts the two candidates left without a victim");
        for (const auto& v : b) tr.note_pick(0, v.second);
        check(tr.picks() == 4 && tr.recent_picks() == 2, "rule counts the two fallback picks of used experts");
    }
    {   // the baseline victim order is reproduced when every resident is unused (nothing routed for K rounds)
        const int64_t E = 8;
        TierVictim t = make(2, false, false, 1, E), off = make(0, false, false, 1, E);
        std::vector<float> raw(8, 0.0f);
        t.observe(raw, kDecay);
        t.observe(raw, kDecay);
        const std::vector<float> u = {3, 1, 4, 1.5f, 9, 2, 6, 5};
        const std::vector<int> res = {0, 1, 2, 3, 4, 5, 6, 7};
        Vic a = pick(t, 0, u, res, 3), b = pick(off, 0, u, res, 3);
        check(a == b, "all unused: the same victims as the baseline rule");
    }
    {   // order_given (the KMiss list): unused first, each group keeps its order; skip drops the recent
        const int64_t E = 8;
        std::vector<float> raw(8, 0.0f);
        TierVictim tr = make(1, false, false, 1, E), ts = make(1, true, false, 1, E);
        route(raw, E, 0, {1, 3});
        tr.observe(raw, kDecay); ts.observe(raw, kDecay);
        Vic v = {{0.0f, 1}, {0.0f, 2}, {0.0f, 3}, {0.0f, 4}};
        Vic w = v;
        tr.order_given(0, v, 4);
        check(v[0].second == 2 && v[1].second == 4 && v[2].second == 1 && v[3].second == 3, "order_given rule: unused first, order kept");
        ts.order_given(0, w, 4);
        check(w.size() == 2 && w[0].second == 2 && w[1].second == 4 && ts.skipped() == 2, "order_given skip: the recent dropped");
    }
    {   // direct swap alone: no victim rule, a counter
        TierVictim t = make(0, false, true, 1, 8);
        check(t.on() && t.direct_on() && !t.unused_on(), "direct alone");
        t.note_direct(); t.note_direct();
        check(t.direct_swaps() == 2, "direct swaps counted");
        const std::vector<float> u = {3, 1, 4, 1.5f, 9, 2, 6, 5};
        TierVictim off = make(0, false, false, 1, 8);
        check(pick(t, 0, u, {0, 1, 2, 3}, 2) == pick(off, 0, u, {0, 1, 2, 3}, 2), "direct alone: the baseline victims");
    }
    {   // cost: 48 layers x 512 experts, 24 rounds, about 100 routed experts per layer, plus the victim ordering
        const int64_t L = 48, E = 512;
        TierVictim t = make(4, false, false, L, E);
        std::vector<float> raw((size_t) (L * E), 0.0f);
        uint32_t x = 12345;
        auto rnd = [&] { x = x * 1664525u + 1013904223u; return (x >> 8) % (uint32_t) E; };
        const auto t0 = std::chrono::steady_clock::now();
        uint64_t sink = 0;
        for (int r = 0; r < 24; ++r) {
            decay_all(raw);
            for (int64_t l = 0; l < L; ++l) for (int j = 0; j < 100; ++j) raw[(size_t) (l * E + (int64_t) rnd())] += 1.0f;
            t.observe(raw, kDecay);
            for (int64_t l = 0; l < L; ++l) {
                Vic v;
                for (int e = 0; e < 256; ++e) v.emplace_back(raw[(size_t) (l * E + e)], (int32_t) e);
                std::partial_sort(v.begin(), v.begin() + 8, v.end(), [&](auto& a, auto& b) { return t.before(l, a, b); });
                sink += (uint64_t) v[0].second;
            }
        }
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 24.0;
        std::printf("tier_victim_test: victim-unused observe + the 48 layers' victim ordering: %.1f us per round (observe alone %.1f us), sink %llu\n",
                    us, t.mean_us(), (unsigned long long) sink);
        check(us < 5000.0, "the victim logic stays under 5 ms per round");
    }
    std::printf("tier_victim_test ok\n");
    return 0;
}
