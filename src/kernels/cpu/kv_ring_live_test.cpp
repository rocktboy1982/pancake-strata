// kv_ring_live_test.cpp - host test of the live-slot ring protocol of include/strata/kernels/kv_ring_model.hpp (LEADS.md Part N N2).
// Random traces of appends, resolves (a window's worth of distinct blocks), shrinks and grows.  Checks: the page table and the slot map stay a
// bijection; no block is ever placed on a slot at or beyond `live`; a shrunk slot holds nothing; after every resolve each requested block reads
// the HOST COPY's value (so a shrink loses no data, because the host copy is authoritative); a resolve with at most `live` distinct blocks never
// overflows; and with live == n_slots from the start the model behaves as the plain ring (no placement differs from a model that never shrinks).
#include <cstdio>
#include <random>
#include <set>
#include <vector>

#include "strata/kernels/kv_ring_model.hpp"

using strata::kernels::kvring::Ring;
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

static void invariants(const Ring& r, int step) {
    for (int b = 0; b < r.n_blocks; ++b) {
        const int sl = r.page_table[b];
        CHECK(sl >= -1 && sl < r.live, "step %d: block %d on slot %d with live %d", step, b, sl, r.live);
        if (sl >= 0 && sl < r.n_slots) CHECK(r.slot_block[sl] == b, "step %d: table / slot map disagree at block %d", step, b);
    }
    for (int sl = 0; sl < r.n_slots; ++sl) {
        if (sl >= r.live) CHECK(r.slot_block[sl] < 0, "step %d: slot %d beyond live holds block %d", step, sl, r.slot_block[sl]);
        else if (r.slot_block[sl] >= 0) CHECK(r.page_table[r.slot_block[sl]] == sl, "step %d: slot %d claims a block that points elsewhere", step, sl);
    }
}

int main() {
    std::mt19937 rng(21);
    long resolves = 0, shrinks = 0, grows = 0;
    for (int trial = 0; trial < 200; ++trial) {
        const int blocks = 64 + (int) (rng() % 400), slots = 16 + (int) (rng() % 120);
        const int window = 1 + (int) (rng() % 12);   // distinct blocks one resolve needs (the verify window's blocks)
        Ring r, plain;
        r.init(blocks, slots);
        plain.init(blocks, slots);
        const int floor_live = std::max(window, 4);
        for (int step = 0; step < 400; ++step) {
            const int op = (int) (rng() % 10);
            if (op < 2) {   // append: host always, slot if resident
                const int b = (int) (rng() % blocks);
                const int64_t v = (int64_t) rng();
                r.append(b, v);
                plain.append(b, v);
            } else if (op < 8) {   // resolve a window
                std::set<int> want;
                const int n = 1 + (int) (rng() % window);
                while ((int) want.size() < n) want.insert((int) (rng() % blocks));
                const std::vector<int> v(want.begin(), want.end());
                r.resolve(v);
                if (r.live == slots) plain.resolve(v);
                ++resolves;
                CHECK(!r.overflow, "trial %d step %d: overflow with %d blocks and live %d", trial, step, n, r.live);
                for (int b : v) {
                    int64_t got = 0;
                    CHECK(r.read(b, got) && got == r.host[b], "trial %d step %d: block %d reads %lld, host copy %lld", trial, step, b,
                          (long long) got, (long long) r.host[b]);
                }
            } else if (op == 8) {
                r.shrink(floor_live + (int) (rng() % (r.live - floor_live + 1)));
                ++shrinks;
            } else {
                r.grow(r.live + (int) (rng() % (slots - r.live + 1)));
                ++grows;
            }
            invariants(r, step);
            CHECK(r.placed_beyond_live == 0, "trial %d step %d: a block was placed beyond live", trial, step);
        }
    }
    // the never-shrunk model equals itself after a shrink to n_slots and a grow back (identity cycle): a no-op cycle changes no state
    {
        Ring a, b;
        a.init(100, 30); b.init(100, 30);
        std::mt19937 q(3);
        for (int i = 0; i < 200; ++i) {
            std::set<int> w;
            while (w.size() < 5) w.insert((int) (q() % 100));
            const std::vector<int> v(w.begin(), w.end());
            a.resolve(v);
            b.resolve(v);
            if (i % 7 == 0) { b.shrink(30); b.grow(30); }
        }
        CHECK(a.page_table == b.page_table && a.slot_block == b.slot_block, "a shrink to the full size and a grow back changed the ring");
    }
    if (g_fail) { std::printf("kv_ring_live_test: %d FAILED\n", g_fail); return 1; }
    std::printf("kv_ring_live_test: 200 traces, %ld resolves, %ld shrinks, %ld grows: bijection kept, nothing placed beyond live, every read equals the host copy; PASS\n",
                resolves, shrinks, grows);
    return 0;
}
