// Host test of include/strata/program/fork_bugfix_rules.hpp: with each switch off the answers are the old code's, with it on
// the fork fixes' (LEADS.md Part H, MOD_PLAN.md section 7; TEST_LEDGER.md).
#include "strata/program/fork_bugfix_rules.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::program;

static void set_env(const char* n, const char* v) {   // v == nullptr: unset
#ifdef _WIN32
    _putenv_s(n, v != nullptr ? v : "");
#else
    if (v != nullptr) setenv(n, v, 1); else unsetenv(n);
#endif
}

static int checks = 0;
#define CHECK(x) do { ++checks; if (!(x)) { std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

int main() {
    // (a) disk-only save and --conversation-cache-min-tokens
    CHECK(disk_only_save_allowed(false, false));    // unset: a short conversation is saved, as before
    CHECK(disk_only_save_allowed(false, true));
    CHECK(!disk_only_save_allowed(true, false));    // set: skipped
    CHECK(disk_only_save_allowed(true, true));

    // (b) prompt-cache root.  im_start = 9; the system prompt ends at 3 (below the root minimum 8), the next boundary is at 12.
    std::vector<int64_t> ids(20, 1);
    ids[0] = 9; ids[3] = 9; ids[12] = 9;
    CHECK(prompt_cache_root_at(ids, 20, 9, 8, false) == -1);   // old: the first boundary (3) is below the minimum: no root
    CHECK(prompt_cache_root_at(ids, 20, 9, 8, true) == 12);    // new: the first boundary at or past 8
    ids[8] = 9;
    CHECK(prompt_cache_root_at(ids, 20, 9, 8, true) == 8);     // "at": the minimum itself counts
    CHECK(prompt_cache_root_at(ids, 20, 9, 8, false) == -1);   // old: still stops at 3
    std::vector<int32_t> big(20, 1);
    big[0] = 9; big[10] = 9;
    CHECK(prompt_cache_root_at(big, 20, 9, 8, false) == 10);   // both agree when the first boundary is past the minimum
    CHECK(prompt_cache_root_at(big, 20, 9, 8, true) == 10);
    CHECK(prompt_cache_root_at(big, 10, 9, 8, true) == -1);    // last_turn is exclusive
    CHECK(prompt_cache_root_at(big, 20, 9, 30, true) == -1);   // minimum past the prompt

    // (c) window-graph eviction range (kWindowMaxT 8, kVerifyLongT 16 in the default build)
    CHECK(window_evict_top(16, 8, 16, false) == 8);
    CHECK(window_evict_top(16, 8, 16, true) == 16);
    CHECK(window_evict_top(12, 8, 16, true) == 12);            // never past --lookup-max-t
    CHECK(window_evict_top(6, 8, 16, true) == 6);
    CHECK(window_evict_top(16, 16, 16, true) == window_evict_top(16, 16, 16, false));   // equal constants: no change

    // (d) progress beat after a capture
    CHECK(!capture_beats(false, true, true));
    CHECK(capture_beats(true, true, true));
    CHECK(!capture_beats(true, false, true));
    CHECK(!capture_beats(true, true, false));

    // (g) draft layer without --mtp
    CHECK(!disk_draft_is_null(false, false));                  // unset: the draft state is passed, as before
    CHECK(!disk_draft_is_null(false, true));
    CHECK(disk_draft_is_null(true, false));
    CHECK(!disk_draft_is_null(true, true));

    // the switch reader: unset / "" / "0" are off
    set_env("STRATA_FORK_FIX_TEST", nullptr);
    CHECK(!fork_fix_env("STRATA_FORK_FIX_TEST"));
    set_env("STRATA_FORK_FIX_TEST", "");
    CHECK(!fork_fix_env("STRATA_FORK_FIX_TEST"));
    set_env("STRATA_FORK_FIX_TEST", "0");
    CHECK(!fork_fix_env("STRATA_FORK_FIX_TEST"));
    set_env("STRATA_FORK_FIX_TEST", "1");
    CHECK(fork_fix_env("STRATA_FORK_FIX_TEST"));
    set_env("STRATA_FORK_FIX_TEST", nullptr);

    std::printf("fork_bugfix_rules_test OK (%d checks)\n", checks);
    return 0;
}
