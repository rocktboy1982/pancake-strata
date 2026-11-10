// src/spec/suffix_index_test.cpp - MOD_PLAN A24: the cross-session suffix index (strata/spec/suffix_index.hpp).
//
//   suffix_index_test        insert, longest match, frequency scoring, the draft limits, eviction, the lead, the
//                            shared wrapper under concurrent proposals and inserts, the restart log
//                            (STRATA_SUFFIX_INDEX_FILE: reload, cut-off tail, compaction, foreign file), the opt-in switch
#include "strata/spec/suffix_index.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

using strata::spec::SharedSuffixIndex;
using strata::spec::SuffixIndex;

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); ++g_fail; }
}

using V = std::vector<int32_t>;

int draft(const SuffixIndex& idx, const V& pat, int max_k, V& out, int* match, std::vector<double>* prob = nullptr) {
    out.assign((size_t) max_k, -7);
    std::vector<double> pr((size_t) max_k, 0.0);
    const int k = idx.propose(pat.data(), (int) pat.size(), max_k, out.data(), match, pr.data());
    out.resize((size_t) k);
    if (prob != nullptr) *prob = std::vector<double>(pr.begin(), pr.begin() + k);
    return k;
}

SuffixIndex::Params params(double min_prob, double factor) {
    SuffixIndex::Params p;
    p.min_prob = min_prob;
    p.factor = factor;
    return p;
}

void test_insert_and_match() {
    SuffixIndex idx(1 << 20, params(0.1, 4.0));
    check(idx.capacity() >= 4096 && (idx.capacity() & (idx.capacity() - 1)) == 0, "capacity is a power of two");
    check(idx.bytes() <= (1u << 20), "the index stays within its bytes");
    V out;
    int m = -1;
    check(draft(idx, {1, 2, 3}, 4, out, &m) == 0 && m == 0, "an empty index drafts nothing");
    idx.insert(nullptr, 0, V{10, 11, 12, 13, 14, 15, 16}.data(), 7);
    check(idx.outputs() == 1 && idx.held() == 7, "one output of seven tokens held");
    check(draft(idx, {11, 12, 13}, 3, out, &m) == 3 && out == V({14, 15, 16}) && m == 3, "a trigram drafts what followed");
    check(draft(idx, {11, 12}, 3, out, &m) == 0, "two tokens are below the shortest match");
    check(draft(idx, {99, 12, 13}, 3, out, &m) == 0, "a context that was never seen drafts nothing");
    // the draft stops at the output's end, never runs into the next output
    idx.insert(nullptr, 0, V{20, 21, 22}.data(), 3);
    check(draft(idx, {14, 15, 16}, 5, out, &m) == 0, "the place at an output's end has nothing after it");
    check(draft(idx, {13, 14, 15}, 5, out, &m) == 1 && out == V({16}), "a draft ends with its output");
    // empty outputs are not inserted
    idx.insert(V{1, 2, 3}.data(), 3, nullptr, 0);
    check(idx.outputs() == 2, "an empty output is not inserted");
}

void test_longest_match() {
    // "5 6 7" goes on with 8 three times, but after "1 2 3 4 5 6 7" (a deeper node) it went on with 9 once
    SuffixIndex idx(1 << 20, params(0.1, 4.0));
    for (int i = 0; i < 3; ++i) idx.insert(nullptr, 0, V{40 + i, 5, 6, 7, 8}.data(), 5);
    idx.insert(nullptr, 0, V{1, 2, 3, 4, 5, 6, 7, 9}.data(), 8);
    V out;
    int m = 0;
    check(draft(idx, {1, 2, 3, 4, 5, 6, 7}, 1, out, &m) == 1 && out[0] == 9 && m == 7, "the deepest match wins over the most frequent");
    check(draft(idx, {0, 5, 6, 7}, 1, out, &m) == 1 && out[0] == 8 && m == 3, "a shallow context takes the vote of its node");
    // the 8-token chain: a context matching 10 tokens back finds the place through the K = 8 level
    SuffixIndex deep(1 << 20, params(0.1, 4.0));
    V a;
    for (int i = 0; i < 30; ++i) a.push_back(100 + i);
    deep.insert(nullptr, 0, a.data(), a.size());
    const V pat(a.begin() + 10, a.begin() + 20);   // 110..119
    check(draft(deep, pat, 4, out, &m) == 4 && out == V({120, 121, 122, 123}) && m == 10, "a 10-token match drafts 4");
    // the 24-token level and the depth cap: a long context matches at most kDepth tokens back
    V b;
    for (int i = 0; i < 200; ++i) b.push_back(1000 + i);
    deep.insert(nullptr, 0, b.data(), b.size());
    const V pat2(b.begin(), b.begin() + 150);
    check(draft(deep, pat2, 3, out, &m) == 3 && out == V({1150, 1151, 1152}) && m == SuffixIndex::kDepth, "a match is capped at kDepth");
}

void test_frequency_scoring() {
    // after "7 8 9": 10 twice, 20 once -> 10 with 2/3; then 11 from both; then 12 vs 13 (1/2): 2/3 * 1/2 = 1/3
    SuffixIndex idx(1 << 20, params(0.1, 4.0));
    idx.insert(nullptr, 0, V{7, 8, 9, 10, 11, 12}.data(), 6);
    idx.insert(nullptr, 0, V{7, 8, 9, 10, 11, 13}.data(), 6);
    idx.insert(nullptr, 0, V{7, 8, 9, 20, 21, 22}.data(), 6);
    V out;
    int m = 0;
    std::vector<double> pr;
    const int k = draft(idx, {5, 7, 8, 9}, 4, out, &m, &pr);
    check(k == 3 && out[0] == 10 && out[1] == 11 && m == 3, "the vote takes the most frequent continuation");
    check(k == 3 && out[2] == 13, "of equal counts the newest place's token wins");
    check(k == 3 && std::fabs(pr[0] - 2.0 / 3) < 1e-12 && std::fabs(pr[1] - 2.0 / 3) < 1e-12 && std::fabs(pr[2] - 1.0 / 3) < 1e-12,
          "the probabilities are the products of the shares");
    // min_prob cuts the draft where the product falls under it
    SuffixIndex strict(1 << 20, params(0.5, 4.0));
    for (const V& o : {V{7, 8, 9, 10, 11, 12}, V{7, 8, 9, 10, 11, 13}, V{7, 8, 9, 20, 21, 22}}) strict.insert(nullptr, 0, o.data(), o.size());
    check(draft(strict, {5, 7, 8, 9}, 4, out, &m) == 2 && out == V({10, 11}), "min_prob 0.5 stops before the 1/3 token");
    SuffixIndex none(1 << 20, params(0.9, 4.0));
    for (const V& o : {V{7, 8, 9, 10}, V{7, 8, 9, 20}}) none.insert(nullptr, 0, o.data(), o.size());
    check(draft(none, {7, 8, 9}, 4, out, &m) == 0 && m == 0, "a first token under min_prob: no draft, no match");
}

void test_draft_limits() {
    V a;
    for (int i = 0; i < 40; ++i) a.push_back(300 + i);
    V out;
    int m = 0;
    SuffixIndex f1(1 << 20, params(0.1, 1.0));
    f1.insert(nullptr, 0, a.data(), a.size());
    check(draft(f1, {300, 301, 302}, 10, out, &m) == 3, "factor 1: a 3-token match drafts 3 tokens");
    SuffixIndex f2(1 << 20, params(0.1, 2.0));
    f2.insert(nullptr, 0, a.data(), a.size());
    check(draft(f2, {300, 301, 302}, 10, out, &m) == 6, "factor 2: a 3-token match drafts 6 tokens");
    check(draft(f2, {300, 301, 302}, 4, out, &m) == 4, "max_k caps the draft");
    const V pat(a.begin(), a.begin() + 30);
    check(draft(f2, pat, 100, out, &m) == 10, "the output's end caps the draft");
    SuffixIndex big(1 << 20, params(0.0, 32.0));
    V c;
    for (int i = 0; i < 100; ++i) c.push_back(500 + i);
    big.insert(nullptr, 0, c.data(), c.size());
    check(draft(big, V(c.begin(), c.begin() + 20), 100, out, &m) == SuffixIndex::kMaxDraft, "kMaxDraft caps the draft");
}

void test_eviction() {
    SuffixIndex idx(0, params(0.1, 4.0));   // the smallest ring: 4096 tokens
    const size_t cap = idx.capacity();
    check(cap == 4096, "the smallest ring holds 4096 tokens");
    idx.insert(nullptr, 0, V{1, 2, 3, 4, 5}.data(), 5);
    V out;
    int m = 0;
    check(draft(idx, {1, 2, 3}, 2, out, &m) == 2, "the first output drafts before the ring wraps");
    // fill the ring with other outputs until the first one is overwritten
    V filler;
    for (int i = 0; i < 1000; ++i) filler.push_back(10000 + (i % 997));
    for (int r = 0; r < 5; ++r) idx.insert(nullptr, 0, filler.data(), filler.size());
    check(idx.held() == cap && idx.written() > cap, "the ring is full and has wrapped");
    check(draft(idx, {1, 2, 3}, 2, out, &m) == 0, "the oldest output is evicted");
    V newest;
    for (int i = 0; i < 6; ++i) newest.push_back(77 + i);
    idx.insert(nullptr, 0, newest.data(), newest.size());
    check(draft(idx, {77, 78, 79}, 3, out, &m) == 3 && out == V({80, 81, 82}), "the newest output drafts after the wrap");
    // chains that cross the evicted region stop there instead of reading overwritten places
    for (int r = 0; r < 20; ++r) idx.insert(nullptr, 0, filler.data(), filler.size());
    check(draft(idx, {10000, 10001, 10002}, 3, out, &m) == 3 && out == V({10003, 10004, 10005}), "a much-repeated trigram after many wraps");
    idx.clear();
    check(idx.held() == 0 && draft(idx, {10000, 10001, 10002}, 3, out, &m) == 0, "clear() empties the index");
}

void test_lead() {
    SuffixIndex idx(1 << 20, params(0.1, 4.0));
    const V lead{40, 41, 42, 43};
    idx.insert(lead.data(), lead.size(), V{50, 51, 52}.data(), 3);
    V out;
    int m = 0;
    check(draft(idx, {41, 42, 43}, 3, out, &m) == 3 && out == V({50, 51, 52}), "the lead's last place drafts the output's start");
    check(draft(idx, {40, 41, 42}, 3, out, &m) == 0, "other lead places are never a draft's place");
    check(draft(idx, {0, 40, 41, 42, 43, 50}, 2, out, &m) == 2 && out == V({51, 52}) && m == 5, "a match runs back into the lead");
    // only the last kLead prompt tokens are kept
    V long_lead;
    for (int i = 0; i < 200; ++i) long_lead.push_back(600 + i);
    SuffixIndex l2(1 << 20, params(0.1, 4.0));
    l2.insert(long_lead.data(), long_lead.size(), V{9, 9, 9}.data(), 3);
    check(l2.held() == (size_t) SuffixIndex::kLead + 3, "a lead keeps kLead tokens");
}

void test_shared_concurrent() {
    SharedSuffixIndex s(1 << 22, params(0.1, 1.0));
    std::atomic<bool> stop{false};
    std::atomic<long> drafts{0};
    std::vector<std::thread> readers;
    for (int r = 0; r < 3; ++r)
        readers.emplace_back([&, r] {
            int32_t out[16];
            int m = 0;
            const int32_t pat[4] = {5, 6 + r % 2, 7, 8};
            while (!stop.load()) drafts += s.propose(pat, 4, 16, out, &m) > 0;
        });
    for (int i = 0; i < 2000; ++i) {
        const int32_t o[8] = {5, 6, 7, 8, 9 + i % 3, 10, 11, 12};
        const int32_t lead[2] = {1, 2};
        s.insert(lead, 2, o, 8);
    }
    stop = true;
    for (auto& t : readers) t.join();
    size_t held = 0, cap = 0;
    uint64_t outs = 0;
    s.stats(&held, &cap, &outs);
    check(outs == 2000 && held <= cap, "concurrent inserts all landed");
    int32_t out[4];
    int m = 0;
    check(s.propose(V{6, 7, 8}.data(), 3, 1, out, &m) == 1 && out[0] == 9 + 1999 % 3, "a shared draft after the inserts");
    std::printf("suffix_index concurrent: %ld drafts while inserting\n", drafts.load());
}

void test_log() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() /
                         ("suffix_index_test_" + std::to_string((long long) std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);
    const std::string path = (dir / "sfx.log").string();
    std::string err;
    uint64_t loaded = 99;
    int32_t out[4];
    int m = 0;
    {
        SharedSuffixIndex a(1 << 20, params(0.1, 4.0));
        check(a.attach_log(path.c_str(), err, &loaded) && loaded == 0, "a new log attaches");
        a.insert(V{1, 2}.data(), 2, V{7, 8, 9, 10, 11}.data(), 5);
        a.insert(nullptr, 0, V{20, 21, 22, 23}.data(), 4);
        check(!a.attach_log(path.c_str(), err), "a second attach is refused");
    }
    {
        SharedSuffixIndex b(1 << 20, params(0.1, 4.0));
        check(b.attach_log(path.c_str(), err, &loaded) && loaded == 2, "the log reloads both outputs");
        check(b.propose(V{8, 9, 10}.data(), 3, 1, out, &m) == 1 && out[0] == 11, "a reloaded output drafts");
        check(b.propose(V{1, 2, 7}.data(), 3, 2, out, &m) == 2 && out[0] == 8 && out[1] == 9, "a reloaded lead is matched into");
    }
    {   // a record cut off by a kill: dropped, the log rewritten, and later appends load again
        std::FILE* f = std::fopen(path.c_str(), "ab");
        const uint32_t junk = 5;
        check(f != nullptr && std::fwrite(&junk, sizeof(junk), 1, f) == 1, "the test appends a cut-off record");
        if (f != nullptr) std::fclose(f);
        SharedSuffixIndex c(1 << 20, params(0.1, 4.0));
        check(c.attach_log(path.c_str(), err, &loaded) && loaded == 2, "a cut-off record is ignored");
        c.insert(nullptr, 0, V{30, 31, 32}.data(), 3);
    }
    {
        SharedSuffixIndex d(1 << 20, params(0.1, 4.0));
        check(d.attach_log(path.c_str(), err, &loaded) && loaded == 3, "appends after a repaired tail load");
    }
    // compaction: a log above twice the ring keeps only the newest outputs that fit
    const std::string path2 = (dir / "big.log").string();
    V o(1000);
    for (int i = 0; i < 1000; ++i) o[(size_t) i] = 50000 + i;
    {
        SharedSuffixIndex e(0, params(0.1, 4.0));   // 4096 tokens
        check(e.attach_log(path2.c_str(), err, &loaded), "a second log attaches");
        for (int r = 0; r < 12; ++r) e.insert(nullptr, 0, o.data(), o.size());   // 12 x 1001 tokens > 2 x 4096
    }
    const auto before = fs::file_size(path2);
    {
        SharedSuffixIndex f(0, params(0.1, 4.0));
        check(f.attach_log(path2.c_str(), err, &loaded) && loaded == 4, "only the outputs that fit the ring load");
        check(fs::file_size(path2) < before, "the oversized log is compacted");
        check(f.propose(V{50000, 50001, 50002}.data(), 3, 2, out, &m) == 2 && out[0] == 50003, "a compacted log drafts");
    }
    // a file that is not a log: refused and left alone
    const std::string path3 = (dir / "other.txt").string();
    if (std::FILE* f = std::fopen(path3.c_str(), "wb")) { std::fputs("hello", f); std::fclose(f); }
    {
        SharedSuffixIndex g(1 << 20, params(0.1, 4.0));
        check(!g.attach_log(path3.c_str(), err) && fs::file_size(path3) == 5, "a foreign file is refused and kept");
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
}

void test_switch() {
    // the process here never sets STRATA_SUFFIX_INDEX: the engine's index is off by default
    if (std::getenv("STRATA_SUFFIX_INDEX") == nullptr) check(strata::spec::shared_suffix_index() == nullptr, "off by default");
}
}  // namespace

int main() {
    test_insert_and_match();
    test_longest_match();
    test_frequency_scoring();
    test_draft_limits();
    test_eviction();
    test_lead();
    test_shared_concurrent();
    test_log();
    test_switch();
    std::printf("suffix_index unit tests: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
