// src/core/head_gather_test.cpp - MOD_PLAN.md section 6 item 38, STRATA_HEAD_SHORTLIST_GATHER: the gathered-rows head against
// the full head plus mask (kernels/head_gather_ref.hpp), CPU only.  Random heads, random cluster tables and hidden rows: the
// gathered logits are the same floats as the masked full head (so the pick is the same), the union is what the kernel reads,
// and its blocks of 8 rows skip exactly the blocks without a union row.
#include "strata/kernels/head_gather_ref.hpp"
#include "strata/program/head_shortlist.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace strata::kernels;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    std::mt19937 rng(21);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int64_t cases = 0, union_rows = 0, vocab_total = 0, strictly_fewer = 0;
    for (int it = 0; it < 300; ++it) {
        const int V = 64 + (int) (rng() % 400), D = 8 + (int) (rng() % 40), K = 4 + (int) (rng() % 20), NC = 1 + (int) (rng() % 8);
        const int n_keep = 1 + (int) (rng() % (unsigned) K);
        std::vector<float> rows((size_t) V * D), centre((size_t) K * D), hidden((size_t) NC * D);
        for (auto& x : rows) x = nd(rng);
        for (auto& x : centre) x = nd(rng);
        for (auto& x : hidden) x = nd(rng);
        std::vector<int32_t> cluster((size_t) V);
        for (auto& c : cluster) c = (int32_t) (rng() % (unsigned) K);
        std::vector<uint8_t> keep((size_t) NC * K);
        for (int j = 0; j < NC; ++j) {
            std::vector<float> score((size_t) K);
            for (int c = 0; c < K; ++c)
                score[(size_t) c] = strata::program::shortlist_dot(hidden.data() + (size_t) j * D, centre.data() + (size_t) c * D, D);
            const std::vector<uint8_t> kp = strata::program::shortlist_select(score.data(), K, n_keep);
            std::memcpy(keep.data() + (size_t) j * K, kp.data(), (size_t) K);
        }
        std::vector<float> a((size_t) NC * V), b((size_t) NC * V);
        masked_full_logits(rows.data(), V, D, hidden.data(), NC, cluster.data(), keep.data(), K, a.data());
        const int64_t dots = gathered_logits(rows.data(), V, D, hidden.data(), NC, cluster.data(), keep.data(), K, b.data());
        // the same floats (bitwise: -inf and the same dots)
        CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
        // the same pick per column
        for (int j = 0; j < NC; ++j) {
            const std::vector<uint8_t> kj(keep.begin() + (size_t) j * K, keep.begin() + (size_t) (j + 1) * K);
            CHECK(strata::program::shortlist_pick(a.data() + (size_t) j * V, cluster.data(), V, kj) ==
                  strata::program::shortlist_pick(b.data() + (size_t) j * V, cluster.data(), V, kj));
        }
        // the union and its work
        const std::vector<uint8_t> u = head_gather_union(keep.data(), NC, K);
        const int64_t rows_in = head_gather_rows(cluster.data(), V, u);
        int64_t want_dots = 0;
        for (int j = 0; j < NC; ++j) {
            const std::vector<uint8_t> kj(keep.begin() + (size_t) j * K, keep.begin() + (size_t) (j + 1) * K);
            want_dots += strata::program::shortlist_rows_scored(cluster.data(), V, kj);
            // a column's kept rows are inside the union
            for (int v = 0; v < V; ++v)
                if (kj[(size_t) cluster[(size_t) v]]) CHECK(u[(size_t) cluster[(size_t) v]]);
        }
        CHECK(dots == want_dots);
        CHECK(rows_in <= V && rows_in >= 0);
        if (NC == 1) CHECK(rows_in == want_dots);   // one column: the union is its own kept rows
        // blocks of 8 rows: a block works exactly when one of its rows is in the union
        const int64_t blocks = head_gather_active_blocks(cluster.data(), V, u, 8);
        CHECK(blocks <= (V + 7) / 8);
        CHECK(blocks * 8 >= rows_in - 0);   // every union row sits in a working block
        ++cases;
        union_rows += rows_in;
        vocab_total += V;
        if (rows_in < V) ++strictly_fewer;
    }
    // worked case: 2 clusters, 8 rows alternating, one column keeping cluster 1 only: rows 1 3 5 7 computed, 4 blocks of
    // 2 rows all work; with blocks of 8 rows, one block works
    {
        const int32_t cluster[8] = {0, 1, 0, 1, 0, 1, 0, 1};
        const uint8_t keep[2] = {0, 1};
        const std::vector<uint8_t> u = head_gather_union(keep, 1, 2);
        CHECK(u[0] == 0 && u[1] == 1);
        CHECK(head_gather_rows(cluster, 8, u) == 4);
        CHECK(head_gather_active_blocks(cluster, 8, u, 2) == 4);
        CHECK(head_gather_active_blocks(cluster, 8, u, 8) == 1);
        const int32_t sorted[8] = {0, 0, 0, 0, 1, 1, 1, 1};   // cluster-contiguous order: half the blocks skip
        CHECK(head_gather_active_blocks(sorted, 8, u, 2) == 2);
    }
    CHECK(strictly_fewer > 0);   // the random cases do prune
    if (fails == 0)
        std::printf("head_gather_test: ok (%lld heads, union %.1f%% of the rows on average)\n", (long long) cases,
                    100.0 * (double) union_rows / (double) vocab_total);
    return fails == 0 ? 0 : 1;
}
