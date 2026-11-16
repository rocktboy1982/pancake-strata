// include/strata/prefill/pf_ahead_rule.hpp - the slot / tag / issue-order rules of STRATA_PF_AHEAD_DEPTH=2 (no GPU needed to check).
//
// STRATA_PF_AHEAD (prefill.cpp pf_issue) holds ONE layer's guessed experts in a pool of `slots`; depth 1 (the default) refills it
// after every layer's combine for the next layer.  Depth 2 (MOD_PLAN.md item 47, LEADS.md Part J kindi, default off) splits the
// SAME pool into two sets of slots / 2 each and keeps two layers in flight: layer l's experts live in set (l & 1), the copies for
// layer l + 2 are queued after layer l's combine (behind its readers: pf_free), so a copy never overwrites a layer that is still
// to be read.  Slot indices stay absolute in the pool (set 1 starts at slots / 2), so the copy events and the pool address need no
// second array.  The one H2D copy engine serves the copies in issue order: l + 2 mostly queues behind l + 1, so depth 2 helps only
// where the engine idles between layers; it is judged by spec 147 against STRATA_PF_AHEAD=1.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace strata::prefill::pf_ahead {

/// STRATA_PF_AHEAD_DEPTH: 2 -> 2, anything else 1.
inline int parse_depth(const char* v) { return v != nullptr && std::strcmp(v, "2") == 0 ? 2 : 1; }
inline int depth_from_env() {
    static const int d = parse_depth(std::getenv("STRATA_PF_AHEAD_DEPTH"));
    return d;
}

/// Sets of the pool: 2 only when depth 2 is asked and the pool has at least 2 slots, else 1.
inline int sets_of(int depth, int pool_slots) { return depth == 2 && pool_slots >= 2 ? 2 : 1; }
/// The set holding layer `ln`'s experts.
inline int set_of(int sets, int64_t ln) { return sets == 2 ? (int) (ln & 1) : 0; }
/// Slots one set may take, and its first absolute slot.
inline int set_cap(int sets, int pool_slots) { return sets == 2 ? pool_slots / 2 : pool_slots; }
inline int set_base(int sets, int pool_slots, int set) { return sets == 2 ? set * (pool_slots / 2) : 0; }
/// The tag a set carries once layer `ln` of chunk `chunk_seq` is in it (-1: nothing).
inline int64_t tag_of(int64_t chunk_seq, int64_t ln) { return chunk_seq * 4096 + ln; }

/// The layer to issue after layer `l`'s combine: l + 1 at depth 1; l + 2 at depth 2 (l + 1 was issued earlier).
inline int64_t issue_after(int sets, int64_t l) { return sets == 2 ? l + 2 : l + 1; }

struct Issue {
    int64_t after;   // -1: at the chunk's start, else after this layer's combine
    int64_t layer;
};
/// The issue order of one chunk over layers [lb, le): the first layers at the start (lb; with two sets also lb + 1), then one
/// per layer after its combine while the target layer is below le.
inline std::vector<Issue> plan(int sets, int64_t lb, int64_t le) {
    std::vector<Issue> v;
    if (lb < le) v.push_back({-1, lb});
    if (sets == 2 && lb + 1 < le) v.push_back({-1, lb + 1});
    for (int64_t l = lb; l < le; ++l) {
        const int64_t t = issue_after(sets, l);
        if (t < le) v.push_back({l, t});
    }
    return v;
}

}  // namespace strata::prefill::pf_ahead
