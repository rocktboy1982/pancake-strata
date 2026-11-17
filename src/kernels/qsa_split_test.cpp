// src/kernels/qsa_split_test.cpp - host test of include/strata/kernels/qsa_split.hpp (LEADS.md Part O O9, STRATA_QSA_SPLIT_GRID): the rule that maps a target
// workgroup count to blocks per (KV head, query) and the chunks each block visits.  No GPU: the kernel's loop is `for (c = b; c < n_chunks; c += blocks)`,
// which qsa_split_chunk spells; the test checks that this visits every chunk exactly once and that the default is untouched.
#include "strata/kernels/qsa_split.hpp"

#include <cstdio>
#include <vector>

using namespace strata::kernels;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

int main() {
    // default: off values change nothing
    CHECK(qsa_split_target(nullptr) == 0 && qsa_split_target("") == 0 && qsa_split_target("0") == 0 && qsa_split_target("-3") == 0 && qsa_split_target("12x") == 0);
    CHECK(qsa_split_target("84") == 84 && qsa_split_target("1") == 1);
    CHECK(qsa_split_blocks(0, 2, 8, 32) == 0 && qsa_split_blocks(84, 2, 8, 1) == 0 && qsa_split_blocks(84, 0, 8, 32) == 0);
    // 84 workgroups over 2 KV heads x 1 query: 42 blocks per (head, query), 32 chunks: as many blocks as chunks, so the default grid
    CHECK(qsa_split_blocks(84, 2, 1, 32) == 0);
    CHECK(qsa_split_blocks(32, 2, 1, 32) == 16);       // 16 blocks, two chunks each
    CHECK(qsa_split_blocks(8, 2, 8, 32) == 1);         // fewer targets than (head, query) pairs: one block each, never zero
    CHECK(qsa_split_blocks(168, 2, 8, 32) == 10);      // 168 / 16 = 10 blocks for 32 chunks: uneven, still exact
    // every chunk exactly once, ascending inside a block, for a range of shapes
    for (int n_chunks : {2, 3, 7, 16, 32, 33, 64}) {
        for (int blocks = 1; blocks < n_chunks; ++blocks) {
            std::vector<int> seen(size_t(n_chunks), 0);
            for (int b = 0; b < blocks; ++b) {
                int last = -1;
                for (int i = 0;; ++i) {
                    const int c = qsa_split_chunk(b, blocks, n_chunks, i);
                    if (c < 0) break;
                    CHECK(c > last && c < n_chunks && c % blocks == b);
                    last = c;
                    ++seen[size_t(c)];
                }
            }
            for (int c = 0; c < n_chunks; ++c) CHECK(seen[size_t(c)] == 1);
        }
    }
    // a block with no chunk (b >= n_chunks cannot happen when blocks < n_chunks, but the function must not index past the end)
    CHECK(qsa_split_chunk(5, 6, 5, 0) == -1);
    if (g_fail) { std::printf("qsa_split_test: %d FAILED\n", g_fail); return 1; }
    std::printf("qsa_split_test: ok\n");
    return 0;
}
