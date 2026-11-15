// src/spec/ngram_mod_test.cpp - STRATA_NGRAM_MOD's shared table (ngram_mod.hpp, llama.cpp ngram-mod via tctien342/strata-x
// 05c5a841e9): a text seen once proposes its own continuation, an unproven chain proposes nothing, the table outlives a
// request's reset, and the proposal is bounded by max_k.
#include "strata/spec/ngram_mod.hpp"

#include <cstdio>
#include <vector>

using strata::spec::NgramMod;
using strata::spec::NgramModSource;

int main() {
    int failed = 0, checks = 0;
    auto check = [&](bool ok, const char* what) {
        ++checks;
        if (!ok) { ++failed; std::fprintf(stderr, "FAIL %s\n", what); }
    };
    // a 200-token text with no repeated 24-gram: token i = (i * 7919) % 50021
    std::vector<int32_t> text(200);
    for (size_t i = 0; i < text.size(); ++i) text[i] = (int32_t) ((i * 7919u) % 50021u);
    NgramModSource src(60000);
    src.reset();
    src.append(text.data(), text.size());
    // the tail: tokens 20..43 (24 of them) - the table knows what followed them (44..), more than kMin of it
    std::vector<int32_t> out(16, -1);
    int match = 0;
    int k = src.propose(text.data() + 20, 24, 0, 16, out.data(), &match);
    check(k == 16, "a proven chain fills max_k");
    check(match == (int) NgramMod::kMatch, "the match is the seed length");
    bool same = k == 16;
    for (int i = 0; i < k && same; ++i) same = out[(size_t) i] == text[(size_t) (44 + i)];
    check(same, "the proposal is the text's continuation");
    // near the end of the text the chain is shorter than kMin: nothing
    k = src.propose(text.data() + 150, 24, 0, 16, out.data(), &match);
    check(k == 0 && match == 0, "an unproven chain proposes nothing");
    // too short a tail: nothing
    k = src.propose(text.data(), 10, 0, 16, out.data(), &match);
    check(k == 0, "fewer than kMatch tail tokens: nothing");
    // a new request: the shared table still answers
    src.reset();
    k = src.propose(text.data() + 20, 24, 0, 8, out.data(), &match);
    check(k == 8 && out[0] == text[44], "the table outlives reset; max_k bounds the proposal");
    // feedback: five rounds below a quarter clear the table
    NgramMod::State st;
    for (int r = 0; r < 5; ++r) src.pool().observe(st, 0, 8);
    k = src.propose(text.data() + 20, 24, 0, 8, out.data(), &match);
    check(k == 0 && src.pool().used() == 0, "five poor rounds clear the table");
    std::printf("ngram_mod_test: %d checks, %d failures\n", checks, failed);
    return failed ? 1 : 0;
}
