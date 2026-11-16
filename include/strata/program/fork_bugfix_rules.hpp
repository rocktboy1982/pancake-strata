#pragma once

// Six exact bug fixes from the fork scan of 2026-11-16 (LEADS.md Part H, MOD_PLAN.md section 7), each behind a default-off
// STRATA_* switch.  The decisions are pure functions here so a host test (src/program/fork_bugfix_rules_test.cpp) proves
// "old behaviour unset, new behaviour set"; generate.cpp and verify.cpp call them.  Unset: every function returns what the
// code did before.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace strata::program {

/// One switch read: set and not "0" / empty.  Read once by the caller (a function-local static).
inline bool fork_fix_env(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}

/// STRATA_CONV_DISK_WANTS=1 (ElXreno/Strata 3da39ce1): a STRATA_CONV_DISK_ONLY save (switch away, QUIT) also asks
/// `conversations.wants(tokens)` (--conversation-cache-min-tokens) like the RAM park does, so a short side conversation is not
/// written to disk.  Unset: every conversation is saved, as before.  `wants` is ConversationCache::wants.
inline bool disk_only_save_allowed(bool switch_on, bool wants) { return !switch_on || wants; }

/// STRATA_CACHE_ROOT_PAST=1 (ElXreno/Strata 00be27a0): the prompt-cache root is the first turn boundary AT OR PAST
/// --prompt-cache-root.  Before: the FIRST turn boundary after token 0, and only when that one lies at or past the minimum
/// (a short system prompt followed by a long first message gave no root at all).  `ids`, `last_turn` (the exclusive end of
/// the search), `turn_token`, `root_min` as in generate.cpp.  Returns the position or -1.
template <class Id>
inline int64_t prompt_cache_root_at(const std::vector<Id>& ids, int64_t last_turn, int64_t turn_token, int64_t root_min,
                                    bool at_or_past) {
    const int64_t first = at_or_past ? root_min : 1;
    for (int64_t i = std::max<int64_t>(first, 1); i < last_turn && i < (int64_t) ids.size(); ++i)
        if (ids[(size_t) i] == turn_token) {
            if (at_or_past || i >= root_min) return i;
            return -1;
        }
    return -1;
}

/// STRATA_WINDOW_EVICT_LONG=1 (ElXreno/Strata c01bb2a4): the window-graph eviction looks at every captured window size up
/// to kVerifyLongT (the --lookup-max-t sizes past kWindowMaxT too), not only up to kWindowMaxT.  Returns the largest size the
/// eviction considers.  In builds where the two constants are equal nothing changes.
inline int window_evict_top(int max_t, int window_max_t, int verify_long_t, bool long_on) {
    return std::min(max_t, long_on ? verify_long_t : window_max_t);
}

/// STRATA_CAPTURE_BEAT=1 (0FL01/Strata d3a7502d): a window graph that uploaded and synchronized without error counts as
/// progress for the request watchdog (a first request captures every size one after another).  Unset: no beat.
inline bool capture_beats(bool switch_on, bool upload_ok, bool sync_ok) { return switch_on && upload_ok && sync_ok; }

/// STRATA_DISK_NO_DRAFT=1 (lorenzoaspesi/Strata f93304f8, ElXreno/Strata f16edb27): the conversation disk save / read / restore
/// passes no draft layer when no --mtp layer is loaded (the file then holds the main layers only).  Before: the (empty) draft
/// state was always passed, so a disk save or restore without --mtp failed on the draft layer.  True = pass nullptr.
inline bool disk_draft_is_null(bool switch_on, bool use_mtp) { return switch_on && !use_mtp; }

}  // namespace strata::program
