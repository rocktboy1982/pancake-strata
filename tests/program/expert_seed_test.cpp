// MOD_PLAN 11 items 3-4: the adaptive tier's seed at a request's start (include/strata/program/expert_seed.hpp).  No GPU:
// the tier's usage vector is driven the way generate.cpp drives it (routing adds, round_begin, decay, round_end).
#include <cstdio>
#include <vector>

#include "strata/program/expert_seed.hpp"

using strata::program::ExpertSeeder;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

constexpr int NL = 2, NE = 8, K = 2;

static std::vector<int32_t> toks(int n, int base) {
    std::vector<int32_t> v;
    for (int i = 0; i < n; ++i) v.push_back(base + i);
    return v;
}

// one decode window routing expert `e` of every layer `n` times, then a tier round with decay 0.5
static void window(ExpertSeeder& s, std::vector<float>& u, int e, int n) {
    for (int l = 0; l < NL; ++l) u[(size_t) (l * NE + e)] += (float) n;
    s.round_begin(u);
    for (float& v : u) v *= 0.5f;
    s.round_end(u);
}

int main() {
    // off by default: nothing changes
    {
        ExpertSeeder s(NL, NE, K, ExpertSeeder::Config{});
        CHECK(!s.on());
        std::vector<float> u(NL * NE, 1.0f);
        CHECK(s.begin_request(u, toks(100, 0), std::vector<int32_t>{}, "code").empty());
        for (float v : u) CHECK(v == 1.0f);
    }
    // item 4: the domain seed is scale * (1 - rank / top), held as a floor for `hold` rounds, then left to decay
    {
        ExpertSeeder::Config c;
        c.domain_scale = 4.0f;
        c.domain_hold = 2;
        ExpertSeeder s(NL, NE, K, c);
        s.add_domain("code", {{0, 3}, {1, 5}, {0, 1}, {1, 7}}, 2);   // top 2: (0,3) -> 4, (1,5) -> 2
        CHECK(s.on() && s.domains() == 1);
        std::vector<float> u(NL * NE, 0.0f);
        CHECK(s.begin_request(u, toks(100, 0), std::vector<int32_t>{}, "prose").empty());   // no such profile
        CHECK(u[3] == 0.0f);
        CHECK(s.begin_request(u, toks(100, 1000), std::vector<int32_t>{}, "code") == "domain profile code");
        CHECK(u[3] == 4.0f && u[NE + 5] == 2.0f && u[1] == 0.0f && u[NE + 7] == 0.0f);
        window(s, u, 0, 0);
        CHECK(u[3] == 2.0f);       // floor laid again before the round (hold 2 -> 1), then decayed
        window(s, u, 0, 0);
        CHECK(u[3] == 2.0f);       // the second held round
        window(s, u, 0, 0);
        CHECK(u[3] == 1.0f);       // released: it decays
        // the same conversation's next turn does not seed again
        std::vector<int32_t> conv = toks(100, 1000);
        std::vector<int32_t> next = toks(150, 1000);
        u.assign(NL * NE, 0.0f);
        CHECK(s.begin_request(u, next, conv, "code").empty());
        CHECK(u[3] == 0.0f);
    }
    // item 3: a conversation's counts are kept at a switch and come back, scaled, when it resumes
    {
        ExpertSeeder::Config c;
        c.conv_slots = 2;
        c.conv_tokens = 4.0f;
        ExpertSeeder s(NL, NE, K, c);
        CHECK(s.conv_on());
        std::vector<float> u(NL * NE, 0.0f);
        const std::vector<int32_t> a = toks(200, 0), b = toks(200, 5000);
        CHECK(s.begin_request(u, a, std::vector<int32_t>{}, "").empty());
        window(s, u, 2, 3);        // conversation A routes expert 2 (3 entries a layer) ...
        window(s, u, 6, 1);        // ... and expert 6 (1)
        const std::vector<float>& cur = s.current();
        CHECK(cur[2] == 3.0f && cur[6] == 1.0f && cur[NE + 2] == 3.0f);
        CHECK(s.begin_request(u, b, a, "").empty());   // B: A is kept; nothing kept matches B
        CHECK(s.kept() == 1);
        window(s, u, 4, 2);
        u.assign(NL * NE, 0.0f);
        std::vector<int32_t> a2 = a;
        for (int i = 0; i < 50; ++i) a2.push_back(7000 + i);   // A's next turn, after B
        const std::string what = s.begin_request(u, a2, b, "");
        CHECK(what.rfind("conversation working set", 0) == 0);
        CHECK(s.kept() == 2);
        // A's mass: 8 entries over 2 layers = 2 tokens (k 2); scaled to 4 tokens: x2
        CHECK(u[2] == 6.0f && u[6] == 2.0f && u[NE + 2] == 6.0f && u[4] == 0.0f);
        // A's tally goes on from its kept counts; the kept set is replaced, not duplicated, when A is kept again
        window(s, u, 1, 1);
        CHECK(s.current()[2] == 3.0f && s.current()[1] == 1.0f);
        CHECK(s.begin_request(u, b, a2, "").rfind("conversation working set", 0) == 0);   // B resumes too
        CHECK(s.kept() == 2);
    }
    if (failed == 0) std::printf("expert_seed_test: ok\n");
    return failed == 0 ? 0 : 1;
}
