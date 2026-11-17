// causal_order_test.cpp - host test of include/strata/kernels/causal_order.hpp (LEADS.md Part L L8, STRATA_Q36_FA_HEAVY_FIRST).
// The block remap is a bijection (so the output cannot change), the cost per block is non-decreasing in the block index (the premise of
// heaviest first), and in the greedy dispatcher model heaviest first never has a longer makespan than the natural order.
#include <cstdio>
#include <vector>

#include "strata/kernels/causal_order.hpp"

using namespace strata::kernels::causal;
static int g_fail = 0;
#define CHECK(c, ...)                                                          \
    do {                                                                       \
        if (!(c)) {                                                            \
            std::printf("FAIL %s:%d %s  ", __FILE__, __LINE__, #c);            \
            std::printf(__VA_ARGS__);                                          \
            std::printf("\n");                                                 \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

int main() {
    // bijection for every grid width
    for (int gx = 1; gx <= 300; ++gx)
        for (const bool hf : {false, true}) {
            std::vector<int> seen(gx, 0);
            for (int l = 0; l < gx; ++l) {
                const int b = block_index(l, gx, hf);
                CHECK(b >= 0 && b < gx, "range gx=%d l=%d", gx, l);
                if (b >= 0 && b < gx) ++seen[b];
            }
            for (int b = 0; b < gx; ++b) CHECK(seen[b] == 1, "not a bijection gx=%d b=%d", gx, b);
        }
    // premise and effect, the shapes of a q36 prompt chunk (G = 4 query heads a KV head here; chunks of T tokens at depth p0)
    struct Shape { int64_t T; int G; int64_t p0; };
    const Shape shapes[] = {{512, 4, 0}, {512, 4, 4096}, {1024, 4, 65536}, {2048, 8, 0}, {256, 4, 1000}, {64, 4, 0}, {4096, 4, 100000}};
    for (const Shape& s : shapes) {
        const int64_t P = s.p0 + s.T;
        const int gx = (int) ((s.T * s.G + 63) / 64);
        int64_t prev = -1;
        for (int b = 0; b < gx; ++b) {
            const int64_t c = cells_scanned(b, s.T, s.G, P, s.p0);
            CHECK(c >= prev, "cost not monotone: T=%lld b=%d", (long long) s.T, b);
            prev = c;
        }
        for (const int units : {6, 12, 24, 48, 96}) {
            const int64_t nat = makespan(gx, s.T, s.G, P, s.p0, units, false), hf = makespan(gx, s.T, s.G, P, s.p0, units, true);
            CHECK(hf <= nat, "heaviest first longer: T=%lld p0=%lld units=%d %lld vs %lld", (long long) s.T, (long long) s.p0, units,
                  (long long) hf, (long long) nat);
            if (units == 48)
                std::printf("model T=%lld G=%d p0=%lld blocks/head=%d units=48: makespan natural %lld, heaviest first %lld cells (%.2f%%)\n",
                            (long long) s.T, s.G, (long long) s.p0, gx, (long long) nat, (long long) hf, 100.0 * (double) (hf - nat) / (double) nat);
        }
    }
    if (g_fail) { std::printf("causal_order_test: %d FAILED\n", g_fail); return 1; }
    std::printf("causal_order_test: PASS\n");
    return 0;
}
