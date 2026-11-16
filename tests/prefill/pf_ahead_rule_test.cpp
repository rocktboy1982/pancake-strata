// tests/prefill/pf_ahead_rule_test.cpp - host check of include/strata/prefill/pf_ahead_rule.hpp (STRATA_PF_AHEAD_DEPTH=2): the issue
// order is the depth-1 one when the depth is unset (byte-identical plan), every layer is issued exactly once into its set, a set is
// only refilled after the layer it held was combined (its readers queued), and slot ranges of the two sets never overlap.
#include "strata/prefill/pf_ahead_rule.hpp"

#include <cstdio>
#include <map>
#include <vector>

namespace p = strata::prefill::pf_ahead;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    CHECK(p::parse_depth(nullptr) == 1 && p::parse_depth("1") == 1 && p::parse_depth("2") == 2 && p::parse_depth("3") == 1 && p::parse_depth("") == 1);
    CHECK(p::sets_of(1, 160) == 1 && p::sets_of(2, 160) == 2 && p::sets_of(2, 1) == 1 && p::sets_of(2, 2) == 2);
    // depth 1: the old schedule exactly - layer lb at the start, layer l + 1 after layer l's combine
    for (int64_t lb : {0, 5}) {
        const auto v = p::plan(1, lb, lb + 6);
        CHECK(v.size() == 6);
        CHECK(v[0].after == -1 && v[0].layer == lb);
        for (int64_t l = lb; l + 1 < lb + 6; ++l) CHECK(v[(size_t) (l - lb + 1)].after == l && v[(size_t) (l - lb + 1)].layer == l + 1);
        CHECK(p::set_of(1, lb) == 0 && p::set_of(1, lb + 1) == 0 && p::set_cap(1, 160) == 160 && p::set_base(1, 160, 0) == 0);
    }
    for (int pool : {2, 3, 160, 161, 512}) {
        CHECK(p::set_cap(2, pool) * 2 <= pool);
        CHECK(p::set_base(2, pool, 0) == 0 && p::set_base(2, pool, 1) == p::set_cap(2, pool));
        CHECK(p::set_base(2, pool, 1) + p::set_cap(2, pool) <= pool);   // set 1 stays inside the pool
    }
    for (int64_t lb : {0, 3}) for (int n : {0, 1, 2, 3, 7, 48}) {
        const int64_t le = lb + n;
        const auto v = p::plan(2, lb, le);
        std::map<int64_t, int> issued;
        for (const auto& i : v) {
            ++issued[i.layer];
            CHECK(i.layer >= lb && i.layer < le);
        }
        CHECK((int64_t) issued.size() == n);                     // each layer, once
        for (const auto& kv : issued) CHECK(kv.second == 1);
        // a layer is in its pool before it is read (issued at the start, or after layer layer - 2) and the set's previous layer
        // (layer - 2) was combined before the refill; the layer before it (layer - 1) is in the OTHER set
        for (const auto& i : v) {
            if (i.after == -1) { CHECK(i.layer <= lb + 1); continue; }
            CHECK(i.layer == i.after + 2);
            CHECK(i.layer - 2 >= lb);                            // the set held layer - 2, read in layer `after`
            CHECK(p::set_of(2, i.layer) == p::set_of(2, i.after));
            CHECK(p::set_of(2, i.layer) != p::set_of(2, i.layer - 1));
        }
        // readers see their own tag: simulate the walk's tag/set state
        int64_t tag[2] = {-1, -1};
        const int64_t seq = 9;
        auto apply = [&](int64_t after_key) {
            for (const auto& i : v) if (i.after == after_key) tag[p::set_of(2, i.layer)] = p::tag_of(seq, i.layer);
        };
        apply(-1);
        for (int64_t l = lb; l < le; ++l) {
            CHECK(tag[p::set_of(2, l)] == p::tag_of(seq, l));   // layer l reads its experts from the pool
            apply(l);
        }
    }
    // STRATA_PF_COPY_BATCH: parse and the runs (constant positive host stride >= bytes, at most max_run a copy)
    CHECK(p::parse_copy_batch(nullptr) == 0 && p::parse_copy_batch("0") == 0 && p::parse_copy_batch("1") == 0 && p::parse_copy_batch("8") == 8 &&
          p::parse_copy_batch("1000") == 64 && p::parse_copy_batch("x") == 0);
    {
        auto addr = [](std::vector<long> v) { std::vector<std::intptr_t> o; for (long x : v) o.push_back((std::intptr_t) x); return o; };
        const auto r1 = p::copy_runs(addr({1000, 1100, 1200, 1300, 5000, 9000, 9100}), 100, 8);   // stride 100 x4, a pair at 4000, a lone one
        CHECK(r1.size() == 3 && r1[0].start == 0 && r1[0].count == 4 && r1[0].stride == 100);
        CHECK(r1[1].start == 4 && r1[1].count == 2 && r1[1].stride == 4000);
        CHECK(r1[2].start == 6 && r1[2].count == 1 && r1[2].stride == 0);
        const auto r2 = p::copy_runs(addr({0, 128, 256, 384, 512}), 100, 2);                       // capped at 2 a copy
        CHECK(r2.size() == 3 && r2[0].count == 2 && r2[1].count == 2 && r2[2].count == 1 && r2[0].stride == 128);
        const auto r3 = p::copy_runs(addr({0, 50, 100}), 100, 8);                                  // stride 50 < bytes: overlapping, never batched
        CHECK(r3.size() == 3 && r3[0].count == 1 && r3[1].count == 1 && r3[2].count == 1);
        const auto r4 = p::copy_runs(addr({300, 200, 100}), 50, 8);                                // descending: never batched
        CHECK(r4.size() == 3);
        const auto r5 = p::copy_runs(addr({0, 100, 300, 400}), 100, 8);                            // stride changes: 2 + 2
        CHECK(r5.size() == 2 && r5[0].count == 2 && r5[1].count == 2 && r5[1].stride == 100);
        CHECK(p::copy_runs(addr({}), 100, 8).empty());
        CHECK(p::copy_runs(addr({7}), 100, 8).size() == 1);
        CHECK(p::copy_runs(addr({0, 100, 200}), 100, 0).size() == 3);                              // max_run 0 / 1: off
        // coverage: every index once, in order
        for (const auto& rr : {r1, r2, r5}) { int next = 0; for (const auto& r : rr) { CHECK(r.start == next); next += r.count; } }
    }
    std::printf("pf_ahead_rule_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
