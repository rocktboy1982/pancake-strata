// src/prefill/moe_tile_rule_test.cpp - host check of moe_tile_rule.hpp (STRATA_PF_F16_VAR_BM, STRATA_MOE_MAX_BM): the
// tile width rule, and that the half-tile split of work_item covers exactly the rows the 64-row split covers, each once.
// No GPU.  The kernel itself (moe_f16_common.cuh work_item) is the same two lines; its bitwise claim is the GPU A/B's
// (bench/specs, drift_check).
#include "moe_tile_rule.hpp"

#include <cstdio>
#include <random>
#include <vector>

namespace f = strata::prefill::fused::f16;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    // unset: the old rule (the average against a tile's rows)
    CHECK(f::pick_bn_rule(1024 * 10, 512, 0, 0.0, 0) == 64);       // 20 rows an expert
    CHECK(f::pick_bn_rule(16384 * 10, 512, 0, 0.0, 0) == 128);     // 320
    CHECK(f::pick_bn_rule(64 * 100, 100, 0, 0.0, 0) == 128);       // exactly a tile's rows
    CHECK(f::pick_bn_rule(100, 0, 0, 0.0, 0) == 128);              // n_expert 0 counted as 1
    CHECK(f::pick_bn_rule(100, 512, 0, 0.0, 64) == 64);            // MAX_BM 64: no cap
    // forced wins over everything
    CHECK(f::pick_bn_rule(100, 512, 64, 8.0, 32) == 64);
    CHECK(f::pick_bn_rule(100, 512, 128, 8.0, 32) == 128);
    CHECK(f::pick_bn_rule(100, 512, 0, 0.0, 32) == 32);            // MAX_BM 32: always half
    CHECK(f::pick_bn_rule(16384 * 10, 512, 0, 0.0, 32) == 32);
    // VAR_BM: half tiles at or under the average, the old rule above it
    CHECK(f::pick_bn_rule(200 * 10, 512, 0, 8.0, 0) == 32);        // 3.9 rows an expert
    CHECK(f::pick_bn_rule(512 * 8, 512, 0, 8.0, 0) == 32);         // exactly 8
    CHECK(f::pick_bn_rule(512 * 9, 512, 0, 8.0, 0) == 64);         // 9
    CHECK(f::pick_bn_rule(512 * 9, 512, 0, 12.0, 0) == 32);        // threshold 12
    // the split: random experts' row counts, a tile table as fused::group builds it (64-row tiles per expert)
    std::mt19937 rng(7);
    for (int it = 0; it < 200; ++it) {
        const int E = 1 + (int) (rng() % 40);
        std::vector<int> off(E + 1, 0);
        for (int e = 0; e < E; ++e) off[e + 1] = off[e] + (int) (rng() % 4 == 0 ? 0 : rng() % 200);
        std::vector<char> seen((size_t) off[E], 0);
        long covered = 0;
        for (int e = 0; e < E; ++e) {
            for (int row0 = off[e]; row0 < off[e + 1]; row0 += 64) {
                const int whole = (off[e + 1] - row0 < 64 ? off[e + 1] - row0 : 64);
                int got = 0;
                for (int w = 0; w < 2; ++w) {   // work items 2 * tile + half
                    int r0, nr;
                    f::half_tile_rows(row0, off[e + 1], w & 1, f::kHalfTileRule, r0, nr);
                    if (nr <= 0) continue;     // work_item: no work
                    CHECK(nr <= f::kHalfTileRule && r0 >= row0 && r0 + nr <= off[e + 1]);
                    for (int r = r0; r < r0 + nr; ++r) { CHECK(!seen[(size_t) r]); seen[(size_t) r] = 1; ++covered; }
                    got += nr;
                }
                CHECK(got == whole);            // the two halves hold the 64-row tile's rows, no more
                if (whole <= 32) CHECK(got == whole);
            }
        }
        CHECK(covered == off[E]);
    }
    std::printf("moe_tile_rule_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
