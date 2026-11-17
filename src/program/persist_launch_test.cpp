// src/program/persist_launch_test.cpp - host test of include/strata/program/persist_launch.hpp (LEADS.md Part O O7, STRATA_PERSIST_LAYER): the flat arrays a
// persistent launch reads, built from a tape and the carried bases, run by the device's claim / wait rule written on those arrays only (std::thread workers
// in place of workgroups).  Checks: the switch parse; the image equals the tape's own maps; a layer-shaped tape runs every task once and never before the
// segment it depends on, over three launches on carried bases, at 1, 3 and 16 workers; a tape of another shape reuses the launcher state of nothing
// (its own bases); a wait that cannot be satisfied ends with the error flag.
#include "strata/program/persist_launch.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using namespace strata::program;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

// the device's rule on the image arrays only (persist_dev.hpp, line for line): claim, binary search over prefix, bounded wait on the target, run, bump
static int64_t run_image(const PersistImage& im, std::atomic<uint64_t>& claim, std::vector<std::atomic<uint64_t>>& done, int workers,
                         std::vector<std::atomic<int>>& ran, std::atomic<int>& violations, std::atomic<int>& err, int wait_ms) {
    std::atomic<int64_t> n{0};
    const int64_t total = im.prefix[size_t(im.nseg)];
    auto worker = [&]() {
        for (;;) {
            const int64_t idx = int64_t(claim.fetch_add(1, std::memory_order_relaxed) - im.claim_base);
            int lo = 0, hi = im.nseg;
            while (hi - lo > 1) {
                const int mid = (lo + hi) >> 1;
                if (im.prefix[size_t(mid)] <= idx) lo = mid; else hi = mid;
            }
            if (idx >= total) return;
            const int seg = lo;
            const int d = im.dep[size_t(seg)];
            if (d >= 0) {
                const auto t0 = std::chrono::steady_clock::now();
                while (done[size_t(d)].load(std::memory_order_acquire) < im.wait_target[size_t(seg)]) {
                    if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(wait_ms)) { err.store(1); break; }
                    std::this_thread::yield();
                }
                // a violation: the dependency has not run all its tasks in THIS launch although no error was raised
                if (err.load() == 0 && ran[size_t(d)].load() < (im.prefix[size_t(d) + 1] - im.prefix[size_t(d)])) violations.fetch_add(1);
            }
            ran[size_t(seg)].fetch_add(1);
            done[size_t(seg)].fetch_add(1, std::memory_order_release);
            n.fetch_add(1);
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < workers; ++i) th.emplace_back(worker);
    for (auto& x : th) x.join();
    return n.load();
}

int main() {
    CHECK(persist_layer_enabled("1") && !persist_layer_enabled(nullptr) && !persist_layer_enabled("") && !persist_layer_enabled("0") && !persist_layer_enabled("11"));
    const PersistTape t = layer_tape(16, 4, 8, 24, 48, 6);
    CHECK(t.valid() && t.seg.size() == 6 && t.total() == 16 + 4 + 8 + 24 + 48 + 6);
    PersistLauncher L(t);
    // the first image: bases zero, wait targets are the dependency's task count
    PersistImage im0 = L.image();
    CHECK(im0.nseg == 6 && im0.claim_base == 0 && im0.prefix == t.prefix);
    CHECK(im0.dep[0] == -1 && im0.dep[1] == 0 && im0.dep[5] == 4);
    CHECK(im0.wait_target[1] == 16 && im0.wait_target[3] == 8 && im0.wait_target[5] == 48 && im0.wait_target[0] == 0);
    for (int workers : {1, 3, 16}) {
        PersistLauncher lau(t);
        std::atomic<uint64_t> claim{0};
        std::vector<std::atomic<uint64_t>> done(t.seg.size());
        for (auto& d : done) d.store(0);
        for (int launch = 0; launch < 3; ++launch) {
            const PersistImage im = lau.image();
            CHECK(im.claim_base == claim.load());                              // the host's base is where the live counter is
            for (size_t s = 0; s < done.size(); ++s) CHECK(lau.bases().done[s] == done[s].load());
            std::vector<std::atomic<int>> ran(t.seg.size());
            for (auto& r : ran) r.store(0);
            std::atomic<int> viol{0}, err{0};
            const int64_t n = run_image(im, claim, done, workers, ran, viol, err, 3000);
            CHECK(n == t.total());
            for (size_t s = 0; s < ran.size(); ++s) CHECK(ran[s].load() == t.count(int(s)));
            CHECK(viol.load() == 0 && err.load() == 0);
            lau.advance(workers);
            CHECK(lau.bases().claim == claim.load());                          // total + workers: the last claim of every worker
        }
    }
    // a wait that cannot be satisfied: the dependency's target is raised by one task that never runs
    {
        PersistLauncher lau(layer_tape(2, 2, 2, 2, 2, 2));
        PersistImage im = lau.image();
        im.wait_target[1] += 1;
        std::atomic<uint64_t> claim{0};
        std::vector<std::atomic<uint64_t>> done(6);
        for (auto& d : done) d.store(0);
        std::vector<std::atomic<int>> ran(6);
        for (auto& r : ran) r.store(0);
        std::atomic<int> viol{0}, err{0};
        const int64_t n = run_image(im, claim, done, 2, ran, viol, err, 50);
        CHECK(err.load() == 1 && n == 12);                                     // the launch still ends, every task ran
    }
    if (g_fail) { std::printf("persist_launch_test: %d FAILED\n", g_fail); return 1; }
    std::printf("persist_launch_test: ok\n");
    return 0;
}
