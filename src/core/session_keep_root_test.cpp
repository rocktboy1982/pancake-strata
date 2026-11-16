// CPU-only: STRATA_SESSION_KEEP_ROOT (MOD_PLAN.md section 6 item 43, LEADS.md B116): a session file keeps the root
// checkpoint too, default off and byte-identical when off (include/strata/core/conversation_file.hpp).
#include "strata/core/conversation_file.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace strata::core;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

static ConversationCheckpoint ck(size_t tokens, int seed, size_t gdn_bytes = 64) {
    ConversationCheckpoint c;
    c.ids.assign(tokens, seed);
    c.gdn.assign(gdn_bytes, (uint8_t) seed);
    return c;
}

int main() {
    std::vector<ConversationCheckpoint> chain;
    chain.push_back(ck(300, 3));    // the turn checkpoint
    chain.push_back(ck(100, 2));    // the root
    chain.push_back(ck(700, 4, 1 << 20));   // the deepest
    const auto* deep = session_deepest_checkpoint(chain);
    check(deep == &chain[2], "deepest");
    check(session_root_checkpoint(chain, deep) == &chain[1], "root: the shallowest");
    // keep_root off: exactly {deepest}, the old choice
    auto off = session_checkpoints_kept(chain, false);
    check(off.size() == 1 && off[0] == &chain[2], "off: only the deepest");
    // on: shallow first
    auto on = session_checkpoints_kept(chain, true);
    check(on.size() == 2 && on[0] == &chain[1] && on[1] == &chain[2], "on: root then deepest");
    // a single checkpoint, or root == deepest by tokens: nothing extra
    std::vector<ConversationCheckpoint> one{ck(100, 2)};
    check(session_checkpoints_kept(one, true).size() == 1, "one checkpoint: one kept");
    std::vector<ConversationCheckpoint> same{ck(100, 2), ck(100, 2)};
    check(session_checkpoints_kept(same, true).size() == 1, "same tokens: one kept");
    check(session_checkpoints_kept({}, true).empty(), "empty chain: none");
    // a pinned shallower checkpoint is the one kept as the deepest; the root is then the shallowest of the rest
    auto pinned = chain;
    pinned[0].pinned = true;
    auto pk = session_checkpoints_kept(pinned, true);
    check(pk.size() == 2 && pk[0] == &pinned[1] && pk[1] == &pinned[0], "pinned: root + the pinned one");
    // the preflight counts two copies of each kept checkpoint; one kept = the old number exactly
    SessionSaveLive live;
    live.state_bytes = 5000; live.tokens = 1000; live.images = 1; live.kv_layers = 13;
    check(session_save_peak_bytes(off, live) == session_save_peak_bytes(&chain[2], live), "peak: one kept = the old formula");
    check(session_save_peak_bytes(on, live) == session_save_peak_bytes(&chain[2], live) + 2 * (uint64_t) chain[1].bytes(),
          "peak: the root adds two copies of itself");
    // the save helpers follow the switch (unset here: the old behaviour)
    std::vector<ConversationCheckpoint> out;
    std::string why;
    uint64_t asked = 0;
    check(session_save_checkpoints(chain, live, [&](uint64_t n, std::string&) { asked = n; return true; }, out, why) &&
          out.size() == (session_keep_root_env() ? 2u : 1u), "save helper follows STRATA_SESSION_KEEP_ROOT");
    std::vector<const ConversationCheckpoint*> ptrs;
    check(session_save_stream_checkpoints(chain, live, nullptr, ptrs, why) &&
          ptrs.size() == (session_keep_root_env() ? 2u : 1u), "stream helper follows STRATA_SESSION_KEEP_ROOT");
    std::printf("session_keep_root_test: %d checks passed (keep_root env %s)\n", checks, session_keep_root_env() ? "on" : "off");
    return 0;
}
