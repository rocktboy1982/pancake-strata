// src/spec/suffix_drafter.cpp - see include/strata/spec/suffix_drafter.hpp.
#include "strata/spec/suffix_drafter.hpp"

#include <algorithm>
#include <cstdlib>

namespace strata::spec {

namespace {
uint64_t mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    return x ^ (x >> 33);
}
}  // namespace

SuffixDrafter::SuffixDrafter(int min_match, int max_match, size_t capacity_tokens)
    : min_match_(std::max(3, min_match)), max_match_(std::max(min_match, max_match)) {
    size_t cap = 1;
    while (cap < capacity_tokens * 2) cap <<= 1;          // load factor <= 0.5 at the nominal capacity
    table_.assign(cap, Slot{});
    mask_ = cap - 1;
    hist_.reserve(capacity_tokens);
}

void SuffixDrafter::reset() {
    hist_.clear();
    std::fill(table_.begin(), table_.end(), Slot{});
    last_match_ = 0;
}

uint64_t SuffixDrafter::key_at(size_t end) const {
    const uint64_t a = (uint32_t) hist_[end - 2], b = (uint32_t) hist_[end - 1], c = (uint32_t) hist_[end];
    return mix(a * 0x9E3779B97F4A7C15ull ^ mix(b + 0x632BE59BD9B4E019ull) ^ (c << 1)) | 1ull;   // never 0
}

SuffixDrafter::Slot* SuffixDrafter::find_slot(uint64_t key, bool insert) {
    for (size_t i = key & mask_, probes = 0; probes <= mask_; i = (i + 1) & mask_, ++probes) {
        Slot& s = table_[i];
        if (s.key == key) return &s;
        if (s.key == 0) {
            if (!insert) return nullptr;
            s.key = key;
            return &s;
        }
    }
    return nullptr;                                         // table full: the history outgrew its capacity
}

void SuffixDrafter::append(const int32_t* tokens, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        hist_.push_back(tokens[i]);
        const size_t end = hist_.size() - 1;
        if (end < 2) continue;
        Slot* s = find_slot(key_at(end), true);
        if (s == nullptr) continue;
        for (int w = WAYS - 1; w > 0; --w) s->pos[w] = s->pos[w - 1];
        s->pos[0] = (uint32_t) end;
        if (s->n < WAYS) ++s->n;
    }
}

int SuffixDrafter::propose(int max_k, int32_t* out) {
    last_match_ = 0;
    const size_t n = hist_.size();
    if (n < 4 || max_k <= 0) return 0;
    const size_t cur = n - 1;
    const Slot* s = find_slot(key_at(cur), false);
    if (s == nullptr) return 0;
    size_t best_end = 0;
    int best_len = 0;
    size_t cand_end[WAYS] = {};
    int cand_len[WAYS] = {};
    for (int w = 0; w < s->n; ++w) {
        const size_t p = s->pos[w];
        if (p >= cur) continue;                             // the current suffix itself
        int len = 0;
        while (len < max_match_ && len <= (int) p && hist_[p - len] == hist_[cur - len]) ++len;
        cand_end[w] = p;
        cand_len[w] = len;
        if (len > best_len) { best_len = len; best_end = p; }   // most recent first, so ties keep the newer
    }
    if (best_len < min_match_) return 0;
    // STRATA_LOOKUP_OLDER=N (opt-in, read once): when the chosen match leaves fewer than min(N, max_k) tokens after it
    // (periodic text: the nearest match is one period back, so the draft stops at one period), take the most recent
    // older candidate that still matches min_match tokens and has that many after it.  Port of llama.cpp PR #29548
    // (frankekn, b8c8dad7c8: ngram-simple keeps scanning older matches).  Only the drafts change; the verify pass keeps
    // greedy output exact.  Unset or 0: the choice above, unchanged.  Lead: LEADS.md Part F (#29548).
    static const int older = [] {
        const char* v = std::getenv("STRATA_LOOKUP_OLDER");
        const int x = v != nullptr ? std::atoi(v) : 0;
        return x > 0 ? x : 0;
    }();
    if (older > 0) {
        const size_t want = (size_t) std::min(older, max_k);
        if (cur - best_end < want) {
            for (int w = 0; w < s->n; ++w) {
                if (cand_len[w] >= min_match_ && cand_end[w] < cur && cur - cand_end[w] >= want) {
                    best_end = cand_end[w];
                    best_len = cand_len[w];
                    break;
                }
            }
        }
    }
    last_match_ = best_len;
    int k = 0;
    // The continuation may run into the current suffix (periodic text); reading history up to `cur` is valid.
    for (size_t q = best_end + 1; q <= cur && k < max_k; ++q) out[k++] = hist_[q];
    return k;
}

int SuffixDrafter::propose_after(const int32_t* extra, int n_extra, int max_k, int32_t* out) {
    if (n_extra <= 0) return propose(max_k, out);
    // the index holds the history's trigrams only: appended without indexing, the extra tokens just form the suffix
    const size_t n0 = hist_.size();
    hist_.insert(hist_.end(), extra, extra + n_extra);
    const int k = propose(max_k, out);
    hist_.resize(n0);
    return k;
}

int SuffixDrafter::propose_after_const(const int32_t* extra, int n_extra, int max_k, int32_t* out, int* match) const {
    // propose() over the virtual sequence history + extra, read in place (STRATA_SSD_LITE, recycle_drafts.hpp); step for step
    // the same as propose_after(), so the same tokens and match
    *match = 0;
    if (n_extra < 0) n_extra = 0;
    const size_t n0 = hist_.size();
    const size_t n = n0 + (size_t) n_extra;
    auto at = [&](size_t i) -> int32_t { return i < n0 ? hist_[i] : extra[i - n0]; };
    if (n < 4 || max_k <= 0) return 0;
    const size_t cur = n - 1;
    const uint64_t a3 = (uint32_t) at(cur - 2), b3 = (uint32_t) at(cur - 1), c3 = (uint32_t) at(cur);
    const uint64_t key = mix(a3 * 0x9E3779B97F4A7C15ull ^ mix(b3 + 0x632BE59BD9B4E019ull) ^ (c3 << 1)) | 1ull;   // key_at's
    const Slot* s = nullptr;
    for (size_t i = key & mask_, probes = 0; probes <= mask_; i = (i + 1) & mask_, ++probes) {   // find_slot(key, false)
        const Slot& sl = table_[i];
        if (sl.key == key) { s = &sl; break; }
        if (sl.key == 0) break;
    }
    if (s == nullptr) return 0;
    size_t best_end = 0;
    int best_len = 0;
    size_t cand_end[WAYS] = {};
    int cand_len[WAYS] = {};
    for (int w = 0; w < s->n; ++w) {
        const size_t p = s->pos[w];
        if (p >= cur) continue;
        int len = 0;
        while (len < max_match_ && len <= (int) p && at(p - len) == at(cur - len)) ++len;
        cand_end[w] = p;
        cand_len[w] = len;
        if (len > best_len) { best_len = len; best_end = p; }
    }
    if (best_len < min_match_) return 0;
    static const int older = [] {   // STRATA_LOOKUP_OLDER, as propose() reads it
        const char* v = std::getenv("STRATA_LOOKUP_OLDER");
        const int x = v != nullptr ? std::atoi(v) : 0;
        return x > 0 ? x : 0;
    }();
    if (older > 0) {
        const size_t want = (size_t) std::min(older, max_k);
        if (cur - best_end < want) {
            for (int w = 0; w < s->n; ++w) {
                if (cand_len[w] >= min_match_ && cand_end[w] < cur && cur - cand_end[w] >= want) {
                    best_end = cand_end[w];
                    best_len = cand_len[w];
                    break;
                }
            }
        }
    }
    *match = best_len;
    int k = 0;
    for (size_t q = best_end + 1; q <= cur && k < max_k; ++q) out[k++] = at(q);
    return k;
}

}  // namespace strata::spec
