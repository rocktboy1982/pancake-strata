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
    std::printf("pf_ahead_rule_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
