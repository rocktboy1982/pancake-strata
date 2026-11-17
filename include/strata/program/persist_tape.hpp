// include/strata/program/persist_tape.hpp - the host rules of a persistent decode launch ("one launch per pass"): LEADS.md Part O O7.
//
// The mechanism, as the caminus authors describe it (docs/dev/engine-design.md section 3.3; AGPL: the design only, written here in our own words and
// code): the host lays a TAPE of segments in a topological order.  One launch starts G workgroups; each repeatedly claims the next task number from ONE
// counter, maps it to a segment through the prefix sums of the tasks per segment, waits (bounded) until the segment it depends on has finished all its
// tasks, runs the task and bumps its segment's done counter.  Counters only grow and are never reset: every launch knows a BASE for the claim counter
// and for each done counter and waits for `base + count`, so consecutive launches of tapes with different task counts stay correct.  Progress needs
// no co-residency: a workgroup only ever waits for segments that come earlier in claim order, claims are monotonic, so every earlier task is either
// done or owned by a running workgroup and the lowest unfinished task always has its inputs: ONE resident workgroup is enough.
//
// This header is the pure part (no GPU): the tape, the claim -> (segment, task) map, the wait targets, the base update after a launch, and a
// reference run with std::thread workers used by persist_tape_test.cpp to check the protocol (each task once, no task before its dependency,
// no reset between launches, a bounded wait that gives up).  The device side is tools/pancake/probes/persist_tape.hip.cpp (the probe that prices
// the idea on gfx1100); nothing in the engine's decode path uses it yet (STRATA_PERSIST_LAYER is not written: MOD_PLAN.md section 6).
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

namespace strata::program {

struct TapeSegment {
    int tasks = 0;   // task count (>= 1)
    int dep = -1;    // the segment whose tasks must all be done first; -1 none; must be < this segment's index (claim order is topological)
};

struct PersistTape {
    std::vector<TapeSegment> seg;
    std::vector<int64_t> prefix;   // seg.size() + 1 entries: task number where each segment starts, prefix.back() = tasks per launch

    static PersistTape build(const std::vector<TapeSegment>& s) {
        PersistTape t;
        t.seg = s;
        t.prefix.assign(s.size() + 1, 0);
        for (size_t i = 0; i < s.size(); ++i) t.prefix[i + 1] = t.prefix[i] + std::max(1, s[i].tasks);
        return t;
    }
    int64_t total() const { return prefix.empty() ? 0 : prefix.back(); }
    // the segment and the task inside it for a claimed task number (binary search of the prefix sums, as the device does); false past the end
    bool locate(int64_t idx, int* segment, int* task) const {
        if (idx < 0 || idx >= total()) return false;
        const auto it = std::upper_bound(prefix.begin(), prefix.end(), idx);
        *segment = int(it - prefix.begin()) - 1;
        *task = int(idx - prefix[size_t(*segment)]);
        return true;
    }
    int count(int s) const { return int(prefix[size_t(s) + 1] - prefix[size_t(s)]); }
    // valid when every dependency points to an EARLIER segment (the progress invariant), and every segment has a task
    bool valid() const {
        for (size_t i = 0; i < seg.size(); ++i)
            if (seg[i].tasks < 1 || seg[i].dep >= int(i)) return false;
        return true;
    }
};

// the counters a launch starts from; claim and every done counter are monotonic 64-bit values that survive between launches
struct PersistBases {
    uint64_t claim = 0;
    std::vector<uint64_t> done;
    explicit PersistBases(size_t nseg = 0) : done(nseg, 0) {}
    // the value a wait on segment `dep` of the launch with these bases needs to see
    uint64_t wait_target(const PersistTape& t, int dep) const { return done[size_t(dep)] + uint64_t(t.count(dep)); }
    // after a launch that ran the whole tape by `workers` workgroups: the next launch's bases (the host adds the counts; no device-side reset, no fixed
    // epoch arithmetic).  Every workgroup makes ONE last claim that lands past the end of the tape and leaves, so the claim counter moved by total + workers.
    void advance(const PersistTape& t, int workers) {
        claim += uint64_t(t.total()) + uint64_t(workers);
        for (size_t s = 0; s < t.seg.size(); ++s) done[s] += uint64_t(t.count(int(s)));
    }
};

// Reference run: `workers` threads execute one launch of the tape.  run(segment, task) does the work of a task.  A wait that sees no progress for
// `wait_ms` gives up, sets *err and goes on (the device kernel does the same so a launch can never hang); returns the number of tasks run.
// `claim` / `dones` are the live counters (in/out: they hold the bases at the start and keep growing).
inline int64_t persist_reference_run(const PersistTape& t, PersistBases& b, std::atomic<uint64_t>& claim, std::vector<std::atomic<uint64_t>>& dones,
                                     int workers, const std::function<void(int, int)>& run, std::atomic<int>& err, int wait_ms = 2000) {
    std::atomic<int64_t> ran{0};
    const uint64_t claim_base = b.claim;
    const PersistBases at_start = b;
    auto worker = [&]() {
        for (;;) {
            const uint64_t c = claim.fetch_add(1, std::memory_order_relaxed);
            int s = 0, k = 0;
            if (!t.locate(int64_t(c - claim_base), &s, &k)) break;
            const int d = t.seg[size_t(s)].dep;
            if (d >= 0) {
                const uint64_t want = at_start.wait_target(t, d);
                const auto t0 = std::chrono::steady_clock::now();
                while (dones[size_t(d)].load(std::memory_order_acquire) < want) {
                    if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(wait_ms)) { err.store(1); break; }
                    std::this_thread::yield();
                }
            }
            run(s, k);
            dones[size_t(s)].fetch_add(1, std::memory_order_release);
            ran.fetch_add(1);
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < workers; ++i) th.emplace_back(worker);
    for (auto& x : th) x.join();
    b.advance(t, workers);
    return ran.load();
}

}  // namespace strata::program
