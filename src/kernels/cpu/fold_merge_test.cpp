// fold_merge_test.cpp - host test of the last-block-merges protocol of include/strata/kernels/fold_merge.hpp (LEADS.md Part L L9).
// Threads play the chunk blocks (random start order and delays), plain stores write the partials, the ticket picks the merging block.
// Checks R1 .. R5 of the header: the merged output equals the sequential merge BIT FOR BIT whatever the arrival order, empty chunks that exit
// early still count, the counter is back at 0 after every launch (replayed 120 times with no memset), groups with their own counters do not
// disturb each other, and a block that skips its ticket is detected (the merge never ran, the counter is short).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "strata/kernels/fold_merge.hpp"

using namespace strata::kernels::foldmerge;
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

static float w_of(float x) { return std::exp(x); }   // pure; the device's __expf is another pure function, the orchestration is what is tested

int main() {
    constexpr int HD = 64, GROUPS = 4;
    std::mt19937 seed_rng(11);
    std::atomic<int> counters[GROUPS];
    for (auto& c : counters) c.store(0);
    long launches = 0, merged = 0;
    for (int launch = 0; launch < 120; ++launch) {
        const int n_chunks = 1 + (int) (seed_rng() % 16);
        const int n_live = (int) (seed_rng() % (n_chunks + 1));   // chunks past n_live hold no cells: m = -FLT_MAX, l = 0, early exit
        std::vector<Group> grp(GROUPS);
        std::vector<std::vector<float>> out(GROUPS, std::vector<float>(HD, -1.f)), ref(GROUPS, std::vector<float>(HD, -1.f));
        std::vector<std::atomic<int>> done(GROUPS);
        for (auto& d : done) d.store(0);
        std::vector<std::vector<float>> val_m(GROUPS), val_l(GROUPS), val_a(GROUPS);
        for (int q = 0; q < GROUPS; ++q) {
            grp[q].init(n_chunks, HD);
            std::mt19937 r(launch * 17 + q);
            std::uniform_real_distribution<float> u(-4.f, 4.f);
            for (int c = 0; c < n_live; ++c) {   // the values each block will write
                val_m[q].push_back(u(r));
                val_l[q].push_back(0.5f + std::fabs(u(r)));
                for (int d = 0; d < HD; ++d) val_a[q].push_back(u(r));
            }
        }
        // sequential reference on the final partials
        for (int q = 0; q < GROUPS; ++q) {
            Group g = grp[q];
            for (int c = 0; c < n_live; ++c) {
                g.m[c] = val_m[q][c];
                g.l[c] = val_l[q][c];
                for (int d = 0; d < HD; ++d) g.acc[(size_t) c * HD + d] = val_a[q][(size_t) c * HD + d];
            }
            merge_ref(g, w_of, ref[q].data());
        }
        // the blocks, in a random order with random delays
        struct Job { int q, c; };
        std::vector<Job> jobs;
        for (int q = 0; q < GROUPS; ++q) for (int c = 0; c < n_chunks; ++c) jobs.push_back({q, c});
        std::shuffle(jobs.begin(), jobs.end(), seed_rng);
        std::vector<std::thread> th;
        for (const Job& j : jobs) {
            const int delay = (int) (seed_rng() % 50);
            th.emplace_back([&, j, delay] {
                for (volatile int spin = 0; spin < delay * 40; spin = spin + 1) {}
                if (delay % 7 == 0) std::this_thread::yield();
                Group& g = grp[j.q];
                if (j.c < n_live) {   // a block with cells writes its partial
                    g.m[j.c] = val_m[j.q][j.c];
                    g.l[j.c] = val_l[j.q][j.c];
                    for (int d = 0; d < HD; ++d) g.acc[(size_t) j.c * HD + d] = val_a[j.q][(size_t) j.c * HD + d];
                }   // an empty block's partial is the initialised m = -FLT_MAX, l = 0 (R1: it still arrives below)
                if (arrive(counters[j.q], n_chunks)) {
                    merge_ref(g, w_of, out[j.q].data());
                    reset(counters[j.q]);
                    done[j.q].fetch_add(1);
                }
            });
        }
        for (auto& t : th) t.join();
        for (int q = 0; q < GROUPS; ++q) {
            CHECK(done[q].load() == 1, "launch %d group %d: merged %d times", launch, q, done[q].load());
            CHECK(counters[q].load() == 0, "launch %d group %d: counter left at %d", launch, q, counters[q].load());
            CHECK(std::memcmp(out[q].data(), ref[q].data(), HD * sizeof(float)) == 0, "launch %d group %d: output differs from the sequential merge", launch, q);
            merged += done[q].load();
        }
        ++launches;
    }
    std::printf("fold protocol: %ld launches x %d groups, %ld merges, every output bit-identical to the sequential merge, counters back at 0\n",
                launches, GROUPS, merged);
    // negative: a block that skips its ticket (an empty chunk that returns early without arriving) leaves the merge undone and the counter short
    {
        std::atomic<int> c{0};
        const int n_chunks = 8, skipper = 5;
        int merges = 0;
        for (int b = 0; b < n_chunks; ++b)
            if (b != skipper && arrive(c, n_chunks)) ++merges;
        CHECK(merges == 0 && c.load() == n_chunks - 1, "a skipped ticket must leave the count short (merges %d count %d)", merges, c.load());
        std::printf("rule R1: a block that skips its ticket leaves the count at %d of %d and the merge never runs (detected)\n", c.load(), n_chunks);
    }
    // R2: arrival order must not matter for the SUM order: the same partials merged in two chunk orders differ in bits (float sums are not
    // associative), which is why the merge reads by chunk index
    {
        Group g;
        g.init(16, 1);
        std::mt19937 r(5);
        std::uniform_real_distribution<float> u(-2.f, 2.f);
        for (int c = 0; c < 16; ++c) { g.m[c] = u(r); g.l[c] = 1.f + u(r) * u(r); g.acc[c] = u(r) * 100.f; }
        float a = 0, b = 0;
        merge_ref(g, w_of, &a);
        Group h = g;
        for (int c = 0; c < 16; ++c) { h.m[c] = g.m[15 - c]; h.l[c] = g.l[15 - c]; h.acc[c] = g.acc[15 - c]; }
        merge_ref(h, w_of, &b);
        std::printf("rule R2: merging the same 16 partials in reverse order gives %s bits (%.9g vs %.9g)\n", std::memcmp(&a, &b, 4) ? "different" : "the same", a, b);
    }
    if (g_fail) { std::printf("fold_merge_test: %d FAILED\n", g_fail); return 1; }
    std::printf("fold_merge_test: PASS\n");
    return 0;
}
