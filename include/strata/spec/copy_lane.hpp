// include/strata/spec/copy_lane.hpp - LEADS.md Part P lead L1, STRATA_LOOKUP_COPYLANE=1 (default off, exact: a draft is only a
// guess that the verify window checks).
//
// A verbatim-copy (prompt lookup) draft source for --lookup-chain, after hipfire's native-MTP n-gram copy lane.
//
// Derived from hipfire (https://github.com/warpfront/hipfire), crates/hipfire-runtime/src/ngram_mod.rs `MtpCopyContext`
// (SPDX-License-Identifier: Apache-2.0, Copyright (c) 2026 Kaden Schutt), rewritten in C++ for this engine's DraftSource
// interface: the key ladder, the per-key direct-mapped tables, the rolling hash, the token-for-token check and the
// "copy from committed history only" rule are theirs; the lifecycle (reset / append) is ours.  The licence text and the
// notice are in LICENSES/hipfire-Apache-2.0.txt (Apache-2.0 section 4); README.md section 9 credits the source.
//
// How it differs from the chained n-gram table (NgramMod, llama.cpp ngram-mod): that table re-keys EVERY drafted token on
// the last few tokens, so a chain jumps to whichever occurrence of a short key was written last.  This source copies the
// history verbatim from ONE earlier occurrence: one table per key length (n_match and the ladder 32 / 16 / 8 where longer
// than n_match), each mapping hash(key) -> the position just past the key's most recent occurrence.  A proposal tries the
// longest key first, checks the stored position against the history token for token (a hash collision never proposes),
// and copies history[pos..] up to the cap.  Their proxy of an agentic code-edit chain: conditional per-depth acceptance
// 0.991-0.997 against 0.90-0.96 for the 5-token chain (CPU proxy, not a hardware number).  The tables are never cleared
// between requests: a slot written by an earlier request can never propose, because a proposal needs the key to equal
// history[pos - n..pos] of THIS request.
//
// Not taken (theirs, kept out on purpose): the expected-tokens-per-cost length rule and the row-0-only confirmation live
// in their window chooser; here the DraftPolicy's own `chain()` keeps the chained tokens that pay at the measured window
// costs, as for the other sources.
#pragma once

#include "strata/spec/draft_source.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::spec {

/// CopyLane's limits (outside the class: a nested struct with member initializers cannot be a default argument inside it).
struct CopyLaneConfig {
    size_t capacity = (size_t) 1 << 22;   ///< slots in total over all key tables (a power of two)
    int n_match = 5;                      ///< the shortest key
    int n_min = 1;                        ///< a copy shorter than this falls through to the next key
    int n_max = 63;                       ///< hard cap on a proposal
};

class CopyLane {
public:
    static constexpr uint64_t kHashMul = 6364136223846793005ull;
    static constexpr uint32_t kEmpty = 0xFFFFFFFFu;
    static constexpr int kLadder[3] = {32, 16, 8};   ///< tried before n_match, longest first, only those longer than n_match

    using Config = CopyLaneConfig;

    explicit CopyLane(const Config& c = Config()) { init(c); }

    bool ok() const { return ok_; }
    const Config& config() const { return cfg_; }
    const std::vector<int>& keys() const { return keys_; }
    size_t slots_per_key() const { return slots_; }
    const std::vector<int32_t>& history() const { return hist_; }

    /// A new request: the history restarts; the tables are reused without clearing (see the header).
    void begin_request() { hist_.clear(); indexed_ = 0; last_key_ = 0; last_pos_ = -1; }

    /// Committed tokens, in order (the prompt, then every accepted token).
    void append(const int32_t* tokens, size_t n) { hist_.insert(hist_.end(), tokens, tokens + n); index_new(); }

    /// What follows the longest matching suffix key of `committed ++ speculative` (the last `n_pending` tokens of `tail`
    /// are the speculative ones: never in the history, never indexed).  Needs `tail`'s last min(n_tail, key) tokens.
    /// Returns the number of tokens written to `out` (<= max_k, <= n_max, never past the history end), 0 = none;
    /// `*key_len` is the key that matched.
    int propose_after(const int32_t* tail, int n_tail, int n_pending, int max_k, int32_t* out, int* key_len) {
        if (key_len != nullptr) *key_len = 0;
        last_key_ = 0;
        last_pos_ = -1;
        if (!ok_) return 0;
        const int limit = std::min(cfg_.n_max, max_k);
        if (limit <= 0 || cfg_.n_min > limit) return 0;
        n_pending = std::max(0, std::min(n_pending, n_tail));
        const size_t end = hist_.size();
        const size_t plen = (size_t) n_pending;
        const int32_t* pref = tail + (n_tail - n_pending);
        for (size_t k = 0; k < keys_.size(); ++k) {
            const size_t n = (size_t) keys_[k];
            if (end + plen < n) continue;
            // key = (last from_hist tokens of the history) ++ (last from_pref tokens of the speculative prefix)
            const size_t from_pref = std::min(n, plen);
            const size_t from_hist = n - from_pref;
            if (from_hist > end) continue;
            const int32_t* hist_part = hist_.data() + (end - from_hist);
            const int32_t* pref_part = pref + (plen - from_pref);
            uint64_t h = 0;
            for (size_t i = 0; i < from_hist; ++i) h = h * kHashMul + (uint64_t) (uint32_t) hist_part[i];
            for (size_t i = 0; i < from_pref; ++i) h = h * kHashMul + (uint64_t) (uint32_t) pref_part[i];
            const uint32_t stored = table_[k * slots_ + slot(h)];
            if (stored == kEmpty) continue;
            const size_t pos = stored;
            if (pos >= end || pos < n) continue;
            const int32_t* cand = hist_.data() + (pos - n);
            bool same = true;
            for (size_t i = 0; i < from_hist && same; ++i) same = cand[i] == hist_part[i];
            for (size_t i = 0; i < from_pref && same; ++i) same = cand[from_hist + i] == pref_part[i];
            if (!same) continue;
            const int take = (int) std::min<size_t>((size_t) limit, end - pos);
            if (take <= 0 || take < cfg_.n_min) continue;
            std::copy(hist_.begin() + (std::ptrdiff_t) pos, hist_.begin() + (std::ptrdiff_t) pos + take, out);
            last_key_ = (int) n;
            last_pos_ = (int64_t) pos;
            if (key_len != nullptr) *key_len = (int) n;
            return take;
        }
        return 0;
    }

    int last_key() const { return last_key_; }
    int64_t last_pos() const { return last_pos_; }

private:
    void init(const Config& c) {
        cfg_ = c;
        ok_ = c.capacity != 0 && (c.capacity & (c.capacity - 1)) == 0 && c.n_match >= 1 && c.n_max >= 1 && c.n_min <= c.n_max;
        if (!ok_) return;
        for (int n : kLadder) if (n > c.n_match) keys_.push_back(n);
        keys_.push_back(c.n_match);
        const size_t per_key = std::max<size_t>(1, c.capacity / keys_.size());
        size_t s = 1;
        unsigned lg = 0;
        while ((s << 1) <= per_key) { s <<= 1; ++lg; }
        slots_ = s;
        shift_ = 64u - lg;
        table_.assign(keys_.size() * slots_, kEmpty);
        for (int n : keys_) {
            uint64_t p = 1;
            for (int i = 0; i < n - 1; ++i) p *= kHashMul;
            mul_pows_.push_back(p);
        }
    }
    size_t slot(uint64_t h) const {
        if (shift_ >= 64) return 0;   // a single slot per key
        return (size_t) (((h ^ (h >> 32)) * 0x9E3779B97F4A7C15ull) >> shift_);
    }
    /// Index every position j in indexed_..history.size(): key history[j-n..j] -> j (the token at j follows it).
    void index_new() {
        const size_t end = hist_.size();
        for (size_t k = 0; k < keys_.size(); ++k) {
            const size_t n = (size_t) keys_[k];
            const size_t start = std::max(indexed_, n);
            if (start >= end) continue;
            uint64_t h = 0;
            for (size_t i = start - n; i < start; ++i) h = h * kHashMul + (uint64_t) (uint32_t) hist_[i];
            for (size_t j = start;;) {
                table_[k * slots_ + slot(h)] = (uint32_t) j;
                if (++j >= end) break;
                h = (h - (uint64_t) (uint32_t) hist_[j - 1 - n] * mul_pows_[k]) * kHashMul + (uint64_t) (uint32_t) hist_[j - 1];
            }
        }
        indexed_ = end;
    }

    Config cfg_;
    bool ok_ = false;
    std::vector<int> keys_;
    std::vector<uint64_t> mul_pows_;
    size_t slots_ = 1;
    unsigned shift_ = 64;
    std::vector<uint32_t> table_;
    std::vector<int32_t> hist_;
    size_t indexed_ = 0;
    int last_key_ = 0;
    int64_t last_pos_ = -1;
};

/// The copy lane behind the DraftSource interface (registered in extra_draft_sources() by the engine).
class CopyLaneSource final : public DraftSource {
public:
    explicit CopyLaneSource(const CopyLane::Config& c = CopyLane::Config()) : lane_(c) {}
    const char* name() const override { return "copy-lane"; }
    void reset() override { lane_.begin_request(); }
    void append(const int32_t* tokens, size_t n) override { lane_.append(tokens, n); }
    int propose(const int32_t* tail, int n_tail, int n_pending, int max_k, int32_t* out, int* match) override {
        *match = 0;
        int key = 0;
        const int k = lane_.propose_after(tail, n_tail, n_pending, max_k, out, &key);
        if (k > 0) *match = key;   // the context the copy is proven on: the engine requires --lookup-chain-min and prefers the longest
        return k;
    }
    CopyLane& lane() { return lane_; }

private:
    CopyLane lane_;
};

}  // namespace strata::spec
