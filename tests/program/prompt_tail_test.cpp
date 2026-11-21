// STRATA_PROMPT_TAIL (LEADS.md Part Q2): the prompt tail's counts and the usage seed (include/strata/program/prompt_tail.hpp).
// No GPU.  Off by default: record / apply must not touch anything.
#include <cmath>
#include <cstdio>
#include <vector>

#include "strata/program/prompt_tail.hpp"

using strata::program::PromptTail;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

constexpr int NL = 2, NE = 8, K = 4;

int main() {
    // off by default (the Config default and the empty environment)
    {
        const PromptTail::Config c = PromptTail::from_env();
        CHECK(!c.on);
        PromptTail t(NL, NE, K, PromptTail::Config{});
        CHECK(!t.on());
        const std::vector<int32_t> ids = {1, 2, 3, 4, 1, 2, 3, 4};
        t.record(0, ids.data(), (int64_t) ids.size());
        CHECK(t.layers_seen() == 0);
        std::vector<float> u(NL * NE, 1.0f);
        CHECK(t.apply(u) == 0.0);
        for (float v : u) CHECK(v == 1.0f);
    }
    PromptTail::Config c;
    c.on = true; c.ranks = 2; c.w = 10.0f;
    // ranks 0-1 only; the layer's row is replaced by each chunk, so the last chunk wins
    {
        PromptTail t(NL, NE, K, c);
        const std::vector<int32_t> first = {7, 6, 5, 4, 7, 6, 5, 4};              // an earlier chunk of layer 0
        const std::vector<int32_t> last = {1, 2, 3, 4, 1, 3, 5, 6, 1, 2, 7, 0};   // 3 tokens x 4: rank 0-1 = 1,2 | 1,3 | 1,2
        t.record(0, first.data(), (int64_t) first.size());
        t.record(0, last.data(), (int64_t) last.size());
        CHECK(t.layers_seen() == 1);
        const std::vector<int32_t> h = t.hottest(0, 3);
        CHECK(h.size() == 3 && h[0] == 1 && h[1] == 2 && h[2] == 3);   // 1: 3 picks, 2: 2, 3: 1; rank 2-3 ids ignored
        std::vector<float> u(NL * NE, 0.0f);
        const double mass = t.apply(u);
        // w * count / tokens = 10 * 3 / 3, 10 * 2 / 3, 10 * 1 / 3 on layer 0; layer 1 has no tail
        CHECK(std::fabs(u[1] - 10.0f) < 1e-4f && std::fabs(u[2] - 20.0f / 3.0f) < 1e-4f && std::fabs(u[3] - 10.0f / 3.0f) < 1e-4f);
        CHECK(u[4] == 0.0f && u[5] == 0.0f);
        for (int e = 0; e < NE; ++e) CHECK(u[NE + e] == 0.0f);
        CHECK(std::fabs(mass - 20.0) < 1e-4);   // w * ranks
    }
    // -1 and out-of-range ids are skipped; a usage vector of another shape is left alone; reset drops the tail
    {
        PromptTail t(NL, NE, K, c);
        const std::vector<int32_t> ids = {-1, 99, 3, 3};
        t.record(1, ids.data(), (int64_t) ids.size());
        std::vector<float> bad(5, 1.0f);
        CHECK(t.apply(bad) == 0.0);
        std::vector<float> u(NL * NE, 0.0f);
        CHECK(t.apply(u) == 0.0);   // the only counted ranks (0, 1) were -1 and 99
        const std::vector<int32_t> ids2 = {2, 2, 0, 0};
        t.record(1, ids2.data(), 4);
        CHECK(t.apply(u) > 0.0);
        t.reset();
        CHECK(t.layers_seen() == 0);
        std::vector<float> v(NL * NE, 0.0f);
        CHECK(t.apply(v) == 0.0);
    }
    if (failed == 0) std::printf("prompt_tail_test: ok\n");
    return failed == 0 ? 0 : 1;
}
