// include/strata/spec/ngram_mod.hpp - STRATA_NGRAM_MOD (opt-in, LEADS.md FK-35): one bounded n-gram table shared by
// every request of the engine, the llama.cpp ngram-mod drafter (MIT; common/ngram-mod.cpp b000431a, common/speculative.cpp
// 9aa51ce4) as ported by tctien342/strata-x 05c5a841e9 / 7163271373.
//
// The table maps the hash of 24 consecutive tokens to the token that followed them (last writer wins, 4 Mi entries =
// 16 MiB).  A proposal seeds a chain with the context's last 24 tokens and follows the table one token at a time; only a
// chain of 48..64 tokens counts as proven, and the caller takes a bounded prefix of it.  Here it is one more
// DraftSource (draft_source.hpp) for --lookup-chain: the verify window checks every proposed token, so the source can
// change only the speed, never the output.  The table outlives requests (the point of sharing it); a request's own
// cursor (how much of its committed text was learned) resets with it.
#pragma once

#include "strata/spec/draft_source.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace strata::spec {

class NgramMod {
public:
    static constexpr size_t kMatch = 24, kMin = 48, kMax = 64;
    struct State { size_t learned = 0; int low = 0; };
    explicit NgramMod(int32_t vocab, size_t entries = 4 * 1024 * 1024)
        : vocab_(vocab > 0 ? vocab : 1), entries_(entries > 0 ? entries : 1, -1) {}

    /// Learn the committed text past the state's cursor (keeps the last kMatch tokens to seed the next window).
    void learn_to(State& state, const std::vector<int32_t>& history) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (history.size() <= kMatch || state.learned + kMatch >= history.size()) return;
        learn(history, state.learned);
        state.learned = history.size() - kMatch;
        if (used_ > entries_.size() / 4 * 3) clear();   // a nearly full table answers noise: start again
    }
    /// The chain after `seed` (exactly kMatch tokens): up to `budget` tokens into `out` when it is proven (>= kMin).
    int propose(const int32_t* seed, int budget, int32_t* out) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (budget <= 0) return 0;
        std::array<int32_t, kMatch + kMax> chain{};
        std::copy_n(seed, kMatch, chain.begin());
        size_t n = 0;
        for (; n < kMax; ++n) {
            const size_t i = index(chain.data() + n);
            const int32_t token = i < entries_.size() ? entries_[i] : -1;
            if (token < 0) break;
            chain[kMatch + n] = token;
        }
        if (n < kMin) return 0;
        const int count = std::min(budget, (int) n);
        std::copy_n(chain.data() + kMatch, count, out);
        return count;
    }
    /// Feedback on the verified prefix: five rounds in a row below a quarter accepted clear the table.
    void observe(State& state, int accepted, int offered) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (offered <= 0) return;
        if (accepted * 4 < offered) {
            if (++state.low >= 5) { clear(); state.low = 0; }
        } else {
            state.low = 0;
        }
    }
    size_t used() const { std::lock_guard<std::mutex> lock(mutex_); return used_; }
    size_t size_bytes() const { return entries_.size() * sizeof(int32_t); }

private:
    size_t index(const int32_t* tokens) const {
        uint64_t hash = 0;
        for (size_t j = 0; j < kMatch; ++j) {
            if (tokens[j] < 0 || tokens[j] >= vocab_) return entries_.size();
            hash = hash * 6364136223846793005ULL + (uint32_t) tokens[j];
        }
        return (size_t) (hash % entries_.size());
    }
    void learn(const std::vector<int32_t>& history, size_t start) {
        for (size_t j = start; j + kMatch < history.size(); ++j) {
            const int32_t token = history[j + kMatch];
            if (token < 0 || token >= vocab_) continue;
            const size_t i = index(history.data() + j);
            if (i == entries_.size()) continue;
            used_ += entries_[i] == -1;
            entries_[i] = token;
        }
    }
    void clear() { std::fill(entries_.begin(), entries_.end(), -1); used_ = 0; }
    int32_t vocab_;
    std::vector<int32_t> entries_;
    size_t used_ = 0;
    mutable std::mutex mutex_;
};

/// The shared table behind the DraftSource interface (registered in extra_draft_sources() by the engine).
class NgramModSource final : public DraftSource {
public:
    explicit NgramModSource(int32_t vocab) : pool_(vocab) {}
    const char* name() const override { return "ngram-mod"; }
    void reset() override { hist_.clear(); state_ = {}; }
    void append(const int32_t* tokens, size_t n) override { hist_.insert(hist_.end(), tokens, tokens + n); }
    int propose(const int32_t* tail, int n_tail, int n_pending, int max_k, int32_t* out, int* match) override {
        (void) n_pending;
        *match = 0;
        pool_.learn_to(state_, hist_);   // the committed text, before the seed (which may end in unverified drafts)
        if (n_tail < (int) NgramMod::kMatch || max_k <= 0) return 0;
        const int k = pool_.propose(tail + (n_tail - (int) NgramMod::kMatch), max_k, out);
        if (k > 0) *match = (int) NgramMod::kMatch;   // the context the chain is proven on
        return k;
    }
    NgramMod& pool() { return pool_; }

private:
    NgramMod pool_;
    NgramMod::State state_;
    std::vector<int32_t> hist_;
};

}  // namespace strata::spec
