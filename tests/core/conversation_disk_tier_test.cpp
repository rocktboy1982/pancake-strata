// CPU-only tests for MOD_PLAN item 39, the disk tier for --serve's parked conversations
// (include/strata/core/conversation_disk_tier.hpp) and the ConversationCache hooks it uses.  No CUDA, no model.
#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_disk_tier.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif

using namespace strata::core;
namespace fs = std::filesystem;

namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

constexpr int32_t kTurn = 248045, kTool = 248058, kEnd = 248046;

std::vector<int32_t> tokens(size_t n, int32_t base = 1000) {
    std::vector<int32_t> v;
    for (size_t i = 0; i < n; ++i) v.push_back(base + (int32_t) i);
    return v;
}

std::vector<uint8_t> bytes_of(size_t n, uint8_t seed) {
    std::vector<uint8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = uint8_t(i * 7u + seed);
    return v;
}

ConversationCheckpoint state(std::vector<int32_t> ids, uint8_t seed) {
    ConversationCheckpoint c;
    c.ids = std::move(ids);
    c.gdn = bytes_of(4099, seed);
    c.ple = bytes_of(77, seed + 1);
    c.tails = bytes_of(301, seed + 2);
    c.dead = bytes_of(64, seed + 3);
    c.block_pos = bytes_of(8, seed + 4);
    return c;
}

// A parked image: live = `live`, checkpoints at the given prefix lengths (chain order as given).
SavedConversation image(const std::vector<int32_t>& live, const std::vector<size_t>& at, uint8_t seed = 1) {
    SavedConversation s;
    for (size_t i = 0; i < s.geometry.size(); ++i) s.geometry[i] = int64_t(100 + i);
    s.layer_lo = 0; s.layer_hi = 48;
    s.live = state(live, seed);
    for (size_t k = 0; k < at.size(); ++k)
        s.checkpoints.push_back(state(std::vector<int32_t>(live.begin(), live.begin() + (std::ptrdiff_t) at[k]),
                                      uint8_t(seed + 10 + k)));
    ConversationKv kv;
    kv.format = 2; kv.cells = 64; kv.heads = 2; kv.head_dim = 8; kv.page_size = 16; kv.pooled_rows = 0; kv.idx_dim = 0;
    kv.k.resize(1000 + seed); kv.v.resize(999);
    s.kv.push_back(std::move(kv));
    return s;
}

// a conversation: system prompt (`sys` tokens), then a user turn and an assistant reply ending as asked
std::vector<int32_t> conversation(size_t sys, size_t body, bool tool, int32_t base = 1000) {
    std::vector<int32_t> v = tokens(sys, base);
    v.push_back(kTurn);
    for (size_t i = 0; i < body; ++i) v.push_back(base + 50000 + (int32_t) i);
    v.push_back(kTurn);   // the reply's own turn
    v.push_back(77);
    if (tool) v.push_back(kTool), v.push_back(78);
    v.push_back(kEnd);
    return v;
}

fs::path temp_dir(const char* tag) {
#ifdef _WIN32
    const fs::path p = fs::temp_directory_path() / (std::string("strata-disk-tier-") + tag);
#else
    const fs::path p = fs::temp_directory_path() / (std::string("strata-disk-tier-") + tag + "-" + std::to_string(getpid()));
#endif
    std::error_code ec;
    fs::remove_all(p, ec);
    return p;
}

std::vector<std::string> logs;
ConversationDiskTier::Settings settings(const fs::path& dir, const std::string& route_prior = "-") {
    ConversationDiskTier::Settings s;
    s.dir = dir;
    s.model = "/models/Qwen3.8-Flash-Next-IQ2_XS-00001-of-00002.gguf";
    s.id = {0x1111, 0x2222};
    std::map<std::string, std::string> env{{"STRATA_ROUTE_PRIOR", route_prior}};
    s.key = conversation_disk_key({s.id, s.model, conversation_disk_key_env([&](const char* n) -> const char* {
        const auto it = env.find(n);
        return it == env.end() || it->second == "-" ? nullptr : it->second.c_str();
    })});
    s.max_bytes = 1ull << 30;
    s.min_tokens = 0;
    s.turn_token = kTurn;
    s.tool_token = kTool;
    s.durable = false;
    s.log = [](const std::string& l) { logs.push_back(l); };
    return s;
}

size_t files_like(const fs::path& dir, const char* ext) {
    size_t n = 0;
    std::error_code ec;
    for (const auto& e : fs::recursive_directory_iterator(dir, ec))
        if (e.path().extension() == ext) ++n;
    return n;
}

void test_key() {
    auto get = [](std::map<std::string, std::string> env) {
        return conversation_disk_key_env([env](const char* n) -> const char* {
            const auto it = env.find(n);
            return it == env.end() ? nullptr : it->second.c_str();
        });
    };
    const SessionFileIdentity id{1, 2};
    const std::string exact = conversation_disk_key({id, "/m/a.gguf", get({})});
    const std::string turbo = conversation_disk_key({id, "/m/a.gguf", get({{"STRATA_ROUTE_PRIOR", "0.5"}})});
    const std::string tail = conversation_disk_key({id, "/m/a.gguf", get({{"STRATA_ROUTE_TAIL_SKIP", "0"}})});
    check(exact == conversation_disk_key({id, "/m/a.gguf", get({})}), "key: same inputs, same key");
    check(exact != turbo && exact != tail && turbo != tail, "key: a lossy routing switch changes the key");
    check(exact != conversation_disk_key({id, "/m/b.gguf", get({})}), "key: another model file, another key");
    check(exact != conversation_disk_key({{1, 3}, "/m/a.gguf", get({})}), "key: another config fingerprint, another key");
    check(conversation_disk_subfolder("/d", "/m/a.gguf", exact) != conversation_disk_subfolder("/d", "/m/a.gguf", turbo),
          "key: exact and turbo states live in different subfolders");
    check(conversation_disk_subfolder("/d", "/m/a.gguf", exact).filename().string().rfind("a-", 0) == 0,
          "key: the subfolder starts with the model's stem");
    // extra names
    const auto extra = get({{"STRATA_CONVERSATION_DISK_KEY_ENV", " STRATA_FOO ,STRATA_ROUTE_PRIOR"}, {"STRATA_FOO", "1"}});
    check(extra.size() == conversation_disk_key_env_names().size() + 1 && extra.back().first == "STRATA_FOO" &&
          extra.back().second == "1", "key: STRATA_CONVERSATION_DISK_KEY_ENV adds names once");
    check(get({}).front().second == "-", "key: an unset switch is '-'");
}

void test_choice() {
    std::vector<ConversationCheckpoint> chain;
    check(conversation_disk_checkpoints(chain).empty(), "choice: empty chain");
    const auto live = tokens(100);
    SavedConversation s = image(live, {40, 10, 90, 60});
    const auto k = conversation_disk_checkpoints(s.checkpoints);
    check(k.size() == 2 && k[0] == 1 && k[1] == 2, "choice: the root (shallowest) and the deepest, in chain order");
    s.checkpoints[0].pinned = true;   // a pin: the deepest pinned one is the resume point (session_deepest_checkpoint)
    const auto kp = conversation_disk_checkpoints(s.checkpoints);
    check(kp.size() == 2 && kp[0] == 0 && kp[1] == 1, "choice: with a pin, the root and the pinned checkpoint");
    SavedConversation one = image(live, {30});
    check(conversation_disk_checkpoints(one.checkpoints) == std::vector<size_t>{0}, "choice: one checkpoint");
}

void test_tool_policy() {
    check(conversation_ended_in_tool_call(conversation(10, 5, true), kTurn, kTool), "policy: tool call after the last turn");
    check(!conversation_ended_in_tool_call(conversation(10, 5, false), kTurn, kTool), "policy: a final answer");
    std::vector<int32_t> earlier = conversation(10, 5, true);
    earlier.push_back(kTurn); earlier.push_back(5); earlier.push_back(kEnd);   // a later reply without one
    check(!conversation_ended_in_tool_call(earlier, kTurn, kTool), "policy: only the last reply counts");
    check(!conversation_ended_in_tool_call(conversation(10, 5, true), kTurn, -1), "policy: off without a token");
}

void test_index() {
    SavedConversation s = image(conversation(20, 30, true), {5, 21});
    s.live.imgs = {{3, 0xabcdull}};
    s.checkpoints[1].imgs = {{3, 0xabcdull}};
    s.cvec = false;
    ConversationDiskEntry e = conversation_disk_entry_of(s, {0, 1});
    e.serial = 7; e.stamp = 12345; e.file_bytes = 999; e.tool = true;
    const auto b = conversation_disk_index_encode(e, 0x55);
    ConversationDiskEntry d;
    std::string why;
    check(conversation_disk_index_decode(b.data(), b.size(), 0x55, d, why), "index: round trip");
    check(d.serial == 7 && d.stamp == 12345 && d.file_bytes == 999 && d.tool && !d.cvec && d.live.ids == s.live.ids &&
          d.live.imgs == s.live.imgs && d.marks.size() == 2 && d.marks[1].ids == s.checkpoints[1].ids &&
          d.marks[1].imgs == s.checkpoints[1].imgs && d.live.gdn.empty(), "index: fields survive, no running state");
    check(!conversation_disk_index_decode(b.data(), b.size(), 0x56, d, why), "index: another key is refused");
    auto bad = b;
    bad[70] ^= 1;
    check(!conversation_disk_index_decode(bad.data(), bad.size(), 0x55, d, why), "index: a flipped byte is refused");
    check(!conversation_disk_index_decode(b.data(), b.size() - 5, 0x55, d, why), "index: a short file is refused");
    check(!conversation_disk_index_decode(b.data(), 10, 0x55, d, why), "index: a tiny file is refused");
}

void test_best_superseded_victims() {
    const auto a = tokens(100), b = tokens(100, 5000);
    std::vector<ConversationDiskEntry> es;
    es.push_back(conversation_disk_entry_of(image(a, {20, 60}), {0, 1}));
    es.back().serial = 1; es.back().stamp = 10;
    es.push_back(conversation_disk_entry_of(image(b, {30}), {0}));
    es.back().serial = 2; es.back().stamp = 20;
    std::vector<int64_t> prompt(a.begin(), a.end());
    prompt.push_back(1);
    auto m = conversation_disk_best(es, prompt, {}, true);
    check(m.index == 0 && m.tokens == 100 && m.serial == 1, "best: the live state the prompt continues");
    std::vector<int64_t> branch(a.begin(), a.begin() + 70);
    branch.push_back(-5);
    m = conversation_disk_best(es, branch, {}, true);
    check(m.index == 0 && m.tokens == 60, "best: a branch resumes at the deepest checkpoint it shares");
    std::vector<int64_t> exact(a.begin(), a.end());
    m = conversation_disk_best(es, exact, {}, true);
    check(m.tokens == 60, "best: never the whole prompt (its last token starts the next window)");
    check(conversation_disk_best(es, prompt, {}, false).tokens == 0, "best: steering must match");
    check(conversation_disk_best(es, prompt, {{5, 9}}, true).tokens == 0, "best: pictures must match");
    // ties: the most recently used
    es.push_back(es[0]);
    es.back().serial = 3; es.back().stamp = 30;
    check(conversation_disk_best(es, prompt, {}, true).serial == 3, "best: ties prefer the most recently used");
    es.pop_back();

    // superseded: the same conversation a turn later holds entry 0's deepest checkpoint (60)
    auto later = tokens(140);
    std::copy(a.begin(), a.end(), later.begin());
    const SavedConversation next = image(later, {20, 60, 120});
    const auto sup = conversation_disk_superseded(es, next);
    check(sup.size() == 1 && sup[0] == 0, "superseded: the same conversation a turn back");
    const SavedConversation sibling = image(tokens(140), {20, 110});   // shares the root (20) only
    check(conversation_disk_superseded(es, sibling).empty(), "superseded: a sibling that shares only the root is kept");

    // victims: oldest first until it fits
    check(conversation_disk_victims({10, 10, 10}, {3, 1, 2}, 5, 40).empty(), "victims: fits");
    check(conversation_disk_victims({10, 10, 10}, {3, 1, 2}, 15, 40) == std::vector<size_t>{1},
          "victims: the oldest goes first");
    check(conversation_disk_victims({10, 10, 10}, {3, 1, 2}, 25, 40) == (std::vector<size_t>{1, 2}),
          "victims: as many as needed");
}

void test_cache_hooks() {
    // without hooks: plain oldest-first, freed
    {
        ConversationCache c(1ull << 30, 8);
        check(c.put(image(conversation(10, 5, true), {}, 1)), "cache: put 1");
        check(c.put(image(conversation(10, 6, false, 3000), {}, 2)), "cache: put 2");
        check(c.evict_oldest() && c.size() == 1 && c.evictions() == 1, "cache: plain eviction");
        const auto p = conversation(10, 6, false, 3000);
        std::vector<int32_t> q(p.begin(), p.end());
        q.push_back(1);
        check(c.best(q, {}, true).tokens == (int64_t) p.size(), "cache: plain eviction removed the oldest");
    }
    // with hooks: the final-answer conversation goes first, into the sink; a tool-call one stays
    {
        ConversationCache c(1ull << 30, 8);
        std::vector<SavedConversation> sunk;
        c.set_disk_tier([&](SavedConversation&& s) { sunk.push_back(std::move(s)); },
                        [](const SavedConversation& s) { return conversation_ended_in_tool_call(s.live.ids, kTurn, kTool); });
        const auto tool = conversation(10, 5, true), fin = conversation(10, 6, false, 3000);
        check(c.put(image(tool, {}, 1)), "hooks: put tool");      // oldest
        check(c.put(image(fin, {}, 2)), "hooks: put final");
        const size_t before = c.bytes();
        check(c.evict_oldest() && sunk.size() == 1 && sunk[0].live.ids == fin, "hooks: the final answer is demoted first");
        check(c.bytes() < before && c.size() == 1 && c.evictions() == 1 && !sunk[0].live.gdn.empty(),
              "hooks: the sink gets the whole image, the cache's bytes drop");
        check(c.evict_oldest() && sunk.size() == 2 && sunk[1].live.ids == tool, "hooks: then the tool-call one");
        // drop_superseded and clear never spill
        check(c.put(image(fin, {5}, 3)), "hooks: put again");
        c.drop_superseded(fin, {}, {state(std::vector<int32_t>(fin.begin(), fin.begin() + 5), 1)}, true);
        check(c.size() == 0 && sunk.size() == 2, "hooks: a superseded copy is not spilled");
        check(c.put(image(fin, {}, 4)), "hooks: put once more");
        c.clear();
        check(sunk.size() == 2, "hooks: clear() is not spilled");
        // make_room through the slot limit spills too; pinned ones stay
        ConversationCache small(1ull << 30, 2);
        small.set_disk_tier([&](SavedConversation&& s) { sunk.push_back(std::move(s)); }, {});
        SavedConversation pinned = image(fin, {5}, 5);
        pinned.checkpoints[0].pinned = true;
        check(small.put(std::move(pinned)) && small.put(image(tool, {}, 6)), "hooks: two parked");
        check(small.put(image(conversation(10, 9, false, 7000), {}, 7)) && small.size() == 2 && sunk.size() == 3 &&
              sunk.back().live.ids == tool, "hooks: make_room spills the oldest unpinned");
    }
}

void test_tier_round_trip() {
    const fs::path dir = temp_dir("rt");
    const auto conv = conversation(40, 200, false);
    std::vector<int64_t> prompt(conv.begin(), conv.end());
    prompt.push_back(9);
    {
        ConversationDiskTier t;
        check(t.open(settings(dir)) && t.on(), "tier: opens");
        {
            ConversationDiskTier second;
            check(!second.open(settings(dir)), "tier: a second engine is locked out of the subfolder");
        }
        SavedConversation s = image(conv, {41, 100, 200, 150});   // root 41 (after the system prompt), deepest 200
        t.spill(std::move(s));
        t.drain();
        check(t.entries() == 1 && files_like(dir, ".strs") == 1 && files_like(dir, ".idx") == 1, "tier: one entry written");
        const auto m = t.best(prompt, {}, true);
        check(m.tokens == (int64_t) conv.size(), "tier: the prompt matches the live state on disk");
        std::vector<int64_t> sib(conv.begin(), conv.begin() + 41);
        sib.push_back(-1);
        check(t.best(sib, {}, true).tokens == 41, "tier: a sibling matches the kept root");
        SavedConversation back;
        size_t bytes = 0;
        std::string why;
        check(t.load(m, {}, back, bytes, why), "tier: reads back");
        const SavedConversation want = image(conv, {41, 100, 200, 150});
        check(back.live.ids == want.live.ids && back.live.gdn == want.live.gdn && back.checkpoints.size() == 2 &&
              back.checkpoints[0].ids == want.checkpoints[0].ids && back.checkpoints[0].gdn == want.checkpoints[0].gdn &&
              back.checkpoints[1].ids == want.checkpoints[2].ids && back.checkpoints[1].gdn == want.checkpoints[2].gdn &&
              back.kv.size() == 1 && back.kv[0].k.size() == want.kv[0].k.size(), "tier: live, root and deepest come back");
        check(t.entries() == 1, "tier: the entry stays on disk after a read");
    }
    // a restart finds it; a short spill is skipped; a later turn supersedes it
    {
        ConversationDiskTier t;
        check(t.open(settings(dir)) && t.entries() == 1, "tier: a restart finds the entry");
        check(t.best(prompt, {}, true).tokens == (int64_t) conv.size(), "tier: and matches it");
        auto later = conv;
        for (int i = 0; i < 50; ++i) later.push_back(3000 + i);
        t.spill(image(later, {41, 200, 240}));
        t.drain();
        check(t.entries() == 1 && files_like(dir, ".strs") == 1, "tier: the later turn supersedes the older copy");
        std::vector<int64_t> lp(later.begin(), later.end());
        lp.push_back(1);
        check(t.best(lp, {}, true).tokens == (int64_t) later.size(), "tier: the later turn matches");
    }
    {
        auto s = settings(dir);
        s.min_tokens = 100000;
        ConversationDiskTier t;
        check(t.open(s), "tier: reopens");
        t.spill(image(conversation(40, 10, false, 9000), {41}));
        t.drain();
        check(t.entries() == 1 && t.totals().skipped == 1, "tier: a conversation below min_tokens is not written");
    }
    // the turbo preset (STRATA_ROUTE_PRIOR=0.5) sees none of the exact preset's entries
    {
        ConversationDiskTier t;
        check(t.open(settings(dir, "0.5")) && t.entries() == 0, "tier: a lossy preset gets its own, empty subfolder");
        check(t.best(prompt, {}, true).tokens == 0, "tier: and never matches an exact preset's state");
    }
    // leftovers: a broken index, an orphan .strs and a temporary file are removed at open
    {
        fs::path sub;
        for (const auto& e : fs::directory_iterator(dir))
            if (e.is_directory() && fs::exists(e.path() / "key.txt")) {
                for (const auto& f : fs::directory_iterator(e.path()))
                    if (f.path().extension() == ".idx") sub = e.path();
            }
        check(!sub.empty(), "tier: found the exact subfolder");
        std::ofstream(sub / "c-00000000000000ff.idx") << "garbage";
        std::ofstream(sub / "c-00000000000000fe.strs") << "orphan";
        std::ofstream(sub / ".c-00000000000000fd.strs.1234.tmp") << "partial";
        ConversationDiskTier t;
        check(t.open(settings(dir)) && t.entries() == 1, "tier: leftovers do not count");
        check(!fs::exists(sub / "c-00000000000000ff.idx") && !fs::exists(sub / "c-00000000000000fe.strs") &&
              !fs::exists(sub / ".c-00000000000000fd.strs.1234.tmp"), "tier: leftovers are deleted");
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
}

void test_tier_cap() {
    const fs::path dir = temp_dir("cap");
    // each image is ~25 KB on disk; a cap of three of them keeps the newest three, across keys too
    uint64_t one = 0;
    {
        ConversationDiskTier t;
        auto s = settings(dir);
        check(t.open(s), "cap: opens");
        t.spill(image(conversation(40, 100, false, 10000), {41}));
        t.drain();
        one = t.bytes();
    }
    check(one > 0, "cap: an entry has a size");
    {
        auto s = settings(dir);
        s.max_bytes = one * 3 + one / 2;
        ConversationDiskTier t;
        check(t.open(s), "cap: reopens with a cap");
        for (int k = 1; k <= 4; ++k) {
            t.spill(image(conversation(40, 100, false, 10000 + 100000 * k), {41}, uint8_t(k)));
            t.drain();
        }
        check(t.entries() == 3 && t.bytes() <= s.max_bytes, "cap: the oldest entries go");
        std::vector<int64_t> first;
        const auto c0 = conversation(40, 100, false, 10000);
        first.assign(c0.begin(), c0.end());
        first.push_back(1);
        check(t.best(first, {}, true).tokens == 0, "cap: the first one went first");
    }
    {
        auto s = settings(dir, "0.5");   // another key: the other subfolder's entries count against the cap
        s.max_bytes = one * 3 + one / 2;
        ConversationDiskTier t;
        check(t.open(s), "cap: another key opens");
        t.spill(image(conversation(40, 100, false, 900000), {41}));
        t.drain();
        check(files_like(dir, ".strs") == 3, "cap: the folder's cap holds across keys (the oldest other entry went)");
    }
    {
        auto s = settings(dir);
        s.max_bytes = one / 2;   // smaller than one image: nothing is written
        ConversationDiskTier t;
        check(t.open(s), "cap: a tiny cap opens");
        t.spill(image(conversation(40, 100, false, 777777), {41}));
        t.drain();
        check(t.entries() == 0, "cap: an image larger than the cap is skipped");
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
}
}  // namespace

int main() {
    test_key();
    test_choice();
    test_tool_policy();
    test_index();
    test_best_superseded_victims();
    test_cache_hooks();
    test_tier_round_trip();
    test_tier_cap();
    std::printf("conversation_disk_tier_test: %d checks passed\n", checks);
    return 0;
}
