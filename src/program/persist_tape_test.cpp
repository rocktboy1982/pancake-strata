// LEADS.md Part O O7: the persistent-launch protocol (include/strata/program/persist_tape.hpp) on a CPU reference: every task once, no task before its
// dependency, correct with ONE worker and with many, no reset between launches of different tapes, and a bounded wait that gives up.
#include "strata/program/persist_tape.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::program;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

struct Run {
    std::atomic<uint64_t> claim{0};
    std::vector<std::atomic<uint64_t>> dones;
    explicit Run(size_t n) : dones(n) { for (auto& d : dones) d.store(0); }
};

// runs one launch; false if a task started before its dependency finished, or a task ran twice / not at all
static bool launch(const PersistTape& t, PersistBases& b, Run& r, int workers, std::atomic<int>& err) {
    const size_t n = t.seg.size();
    std::vector<std::atomic<int>> started(size_t(t.total())), finished(n);
    for (auto& x : started) x.store(0);
    for (auto& x : finished) x.store(0);
    std::atomic<int> bad{0};
    auto work = [&](int s, int k) {
        const int d = t.seg[size_t(s)].dep;
        if (d >= 0 && finished[size_t(d)].load() != t.count(d)) bad.store(1);   // a task ran before its dependency finished
        if (started[size_t(t.prefix[size_t(s)] + k)].fetch_add(1) != 0) bad.store(1);
        std::this_thread::yield();
        finished[size_t(s)].fetch_add(1);
    };
    const int64_t ran = persist_reference_run(t, b, r.claim, r.dones, workers, work, err);
    for (auto& x : started) if (x.load() != 1) bad.store(1);
    return bad.load() == 0 && ran == t.total();
}

int main() {
    {   // locate: the prefix-sum map, edges included
        const PersistTape t = PersistTape::build({{3, -1}, {1, 0}, {4, 1}});
        int s = -1, k = -1;
        check(t.total() == 8 && t.valid(), "total 8 tasks, valid");
        check(t.locate(0, &s, &k) && s == 0 && k == 0, "task 0 -> segment 0 task 0");
        check(t.locate(2, &s, &k) && s == 0 && k == 2, "task 2 is the last of segment 0");
        check(t.locate(3, &s, &k) && s == 1 && k == 0, "task 3 starts segment 1 (a one-task segment)");
        check(t.locate(4, &s, &k) && s == 2 && k == 0 && t.locate(7, &s, &k) && s == 2 && k == 3, "segment 2 holds 4..7");
        check(!t.locate(8, &s, &k) && !t.locate(-1, &s, &k), "past the end and negative are no task");
        check(!PersistTape::build({{2, -1}, {2, 1}}).valid(), "a dependency on itself or a later segment is invalid (progress invariant)");
        check(PersistTape::build({{0, -1}}).total() == 1, "a zero-task segment is raised to one task");
    }
    // a chain of dependent segments, with 1, 2 and 7 workers, 3 launches each, bases carried over (never reset)
    for (const int workers : {1, 2, 7}) {
        std::vector<TapeSegment> s;
        for (int i = 0; i < 40; ++i) s.push_back({1 + (i * 5) % 9, i ? i - 1 : -1});
        const PersistTape t = PersistTape::build(s);
        PersistBases b(s.size());
        Run r(s.size());
        std::atomic<int> err{0};
        for (int rep = 0; rep < 3; ++rep) check(launch(t, b, r, workers, err), "every task once, none before its dependency");
        check(err.load() == 0, "no wait gave up");
        check(b.claim == uint64_t(3 * (t.total() + workers)) && r.claim.load() == b.claim, "the claim base advanced by the tasks plus one exit claim per worker in each launch, never reset");
        bool done_ok = true;
        for (size_t i = 0; i < s.size(); ++i) done_ok &= r.dones[i].load() == b.done[i] && b.done[i] == uint64_t(3 * t.count(int(i)));
        check(done_ok, "every done counter equals its base: counters are monotonic across launches");
    }
    {   // tapes of DIFFERENT shape back to back, each on its own counters (the case a fixed epoch rule breaks)
        const PersistTape a = PersistTape::build({{4, -1}, {2, 0}}), c = PersistTape::build({{1, -1}, {5, 0}, {3, 1}});
        PersistBases ba(2), bc(3);
        Run ra(2), rc(3);
        std::atomic<int> err{0};
        check(launch(a, ba, ra, 3, err) && launch(a, ba, ra, 3, err) && launch(c, bc, rc, 3, err) && launch(a, ba, ra, 3, err), "different tapes still correct");
        check(err.load() == 0, "no wait gave up (different tapes)");
    }
    {   // a bounded wait gives up instead of hanging: a dependency that can never finish (the host's base is ahead of the live counter)
        const PersistTape t = PersistTape::build({{2, -1}, {2, 0}});
        PersistBases b(2);
        b.done[0] = 1000;
        Run r(2);
        std::atomic<int> err{0};
        std::atomic<int64_t> ran{0};
        persist_reference_run(t, b, r.claim, r.dones, 2, [&](int, int) { ran.fetch_add(1); }, err, 50);
        check(err.load() == 1, "the wait that cannot be satisfied sets the error word");
        check(ran.load() == 4, "the launch still completes every task (it never hangs)");
    }
    std::printf("persist_tape_test: %d checks passed\n", checks);
    return 0;
}
