// include/strata/spec/suffix_index.hpp - MOD_PLAN A24: a cross-session suffix index with frequency scoring, shared
// by every session the engine serves (the main agent and its sub-agents): the global tree of SuffixDecoding
// (Oliaro et al., "SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications", arXiv 2411.04975).
//
// --lookup (SuffixDrafter) guesses from a repeat inside the request it serves: its prompt and its output so far.  An
// agent also repeats what it wrote in OTHER requests: the same tool-call skeletons, the same reasoning openers, the
// same edit shapes, and a sub-agent writes what its siblings wrote.  This index holds the finished outputs of every
// request and drafts what most of the places that end like the current context went on with.
//
// Structure (after strata-poly's SuffixIndex, src/poly/suffix_index.hpp, the FastStrata port): a ring of tokens,
// outputs one after another with a mark between them, the oldest text overwritten when the ring is full.  For three
// lengths K (24, 8, 3) every place is chained to the place before it that ended with the same K tokens, as an LZ77
// matcher chains them: a table holds the newest place for each hash of K tokens, each place the distance back to the
// one before.  A query starts at the longest K whose last K tokens were seen; the places on its chain (kWalk at most,
// newest first) are compared backwards with the context, up to kDepth tokens, and those that match furthest are
// kept - the occurrences of the deepest suffix-tree node the context ends in.  What followed them is voted on a token
// at a time: the token most of them have next is the draft's, its share of them is its probability (the paper's
// frequency score, count(child) / count(node)), the places with another token drop out, and the draft ends where the
// product of the shares falls under min_prob, where the places run out, or after factor x the match length tokens
// (the paper's MAX_SPEC = alpha * p).  The engine verifies one window, so the draft is the single most frequent path
// of the paper's speculation tree.
//
// What goes in: each finished output (insert()), after a lead of the last kLead tokens of its prompt.  A match may run
// back into the lead; only the lead's last place is a draft's place (its continuation is the output's first token),
// so the index also learns how outputs start.  The request's own prompt and output stay --lookup's job.
//
// Exact: a draft is only a guess that the verify window checks, so the index changes the speed, never the output.
// Memory: kBytesPerToken a token (ring 4 + per level a chain entry 4 and a table slot 8), the capacity a power of two.
// Thread safety: SuffixIndex itself is not synchronised; propose() is const and keeps its scratch on the stack.
// SharedSuffixIndex wraps it in a shared mutex: inserts exclusive, proposals shared.  An output goes in whole, at the
// end of its request, so concurrent requests never interleave their tokens in the ring.
#pragma once

#include "strata/spec/draft_source.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <shared_mutex>
#include <vector>

namespace strata::spec {

/// SuffixIndex's draft limits (outside the class: a nested struct with member initializers cannot be a default
/// argument inside its own enclosing class).
struct SuffixIndexParams {
    double min_prob = 0.1;   ///< a draft ends where the product of its tokens' shares falls under this
    double factor = 1.0;     ///< a draft is at most factor x the match length tokens long (at least one)
};

class SuffixIndex {
public:
    using Params = SuffixIndexParams;
    static constexpr int kLevels = 3;
    static constexpr int kLen[kLevels] = {24, 8, 3};   ///< the K of each chain, longest first; 3 is the shortest match
    static constexpr int kDepth = 64;                  ///< a match is compared this far back at most
    static constexpr int kWalk = 64;                   ///< places looked at on a chain
    static constexpr int kLead = 64;                   ///< prompt tokens kept before an output
    static constexpr int kMaxDraft = 32;
    static constexpr size_t kBytesPerToken = 4 + kLevels * (4 + 8);

    /// An index in about `bytes` (rounded down to a power of two of tokens, 4096 at least).
    explicit SuffixIndex(size_t bytes, Params p = {});

    /// One finished output, after the last tokens of its prompt (`lead`, kLead of them are kept).
    void insert(const int32_t* lead, size_t n_lead, const int32_t* out, size_t n_out);

    /// A draft after `pat` (the context's last n_pat tokens, the newest last; kDepth of them are used): writes up to
    /// max_k tokens to `out`, returns how many, sets *match to the number of trailing context tokens the draft's
    /// places matched (0: no draft) and, when `prob` is given, prob[i] to the probability that the first i + 1
    /// drafted tokens are right, as the places say.
    int propose(const int32_t* pat, int n_pat, int max_k, int32_t* out, int* match, double* prob = nullptr) const;

    void clear();
    size_t capacity() const { return tok_.size(); }   ///< tokens the ring holds at most
    size_t bytes() const { return capacity() * kBytesPerToken; }
    uint64_t written() const { return end_; }          ///< tokens and marks written since the start
    size_t held() const { return (size_t) (end_ < capacity() ? end_ : capacity()); }
    uint64_t outputs() const { return outputs_; }
    const Params& params() const { return p_; }

private:
    static constexpr int32_t kMark = -1;   // between outputs; a token is never below zero

    struct Level {
        std::vector<uint64_t> head;   // by hash: the newest place + 1 that ended with it; 0: none
        std::vector<uint32_t> back;   // by place in the ring: the distance to the place before it of the same hash
    };

    uint64_t first() const { return end_ > capacity() ? end_ - capacity() : 0; }
    uint64_t hash_ring(uint64_t at, int k) const;
    static uint64_t hash_seq(const int32_t* last, int k);   // `last` points at the newest of the k tokens
    void put(int32_t t, bool indexed);

    Params p_;
    std::vector<int32_t> tok_;
    size_t mask_ = 0;
    Level lv_[kLevels];
    uint64_t end_ = 0;   // tokens and marks written
    uint64_t run_ = 0;   // tokens since the last mark
    uint64_t outputs_ = 0;
};

/// The process-wide index behind a shared mutex, shared by every session's requests.
class SharedSuffixIndex {
public:
    SharedSuffixIndex(size_t bytes, SuffixIndex::Params p) : idx_(bytes, p) {}
    void insert(const int32_t* lead, size_t n_lead, const int32_t* out, size_t n_out) {
        std::unique_lock<std::shared_mutex> lk(mu_);
        idx_.insert(lead, n_lead, out, n_out);
    }
    int propose(const int32_t* pat, int n_pat, int max_k, int32_t* out, int* match, double* prob = nullptr) const {
        std::shared_lock<std::shared_mutex> lk(mu_);
        return idx_.propose(pat, n_pat, max_k, out, match, prob);
    }
    /// held / capacity tokens, outputs inserted (a consistent snapshot)
    void stats(size_t* held, size_t* capacity, uint64_t* outputs) const {
        std::shared_lock<std::shared_mutex> lk(mu_);
        *held = idx_.held(), *capacity = idx_.capacity(), *outputs = idx_.outputs();
    }
    size_t bytes() const { return idx_.bytes(); }   // fixed at construction

private:
    mutable std::shared_mutex mu_;
    SuffixIndex idx_;
};

/// STRATA_SUFFIX_INDEX=1 (opt-in, default off): the engine's shared index, made at the first call; nullptr when off.
///   STRATA_SUFFIX_INDEX_MB         memory cap in MiB (default 64 = 1,048,576 tokens; 1..16384)
///   STRATA_SUFFIX_INDEX_MIN_PROB   a draft ends under this probability (default 0.1, the paper's min_token_prob)
///   STRATA_SUFFIX_INDEX_FACTOR     a draft is at most factor x match tokens (default 1, the paper's max_spec_factor)
///   STRATA_SUFFIX_INDEX_RULE       "longest" (default: the index's draft replaces --lookup's when its match is longer)
///                                  or "fallback" (the index drafts only where --lookup has none)
SharedSuffixIndex* shared_suffix_index();
/// STRATA_SUFFIX_INDEX_RULE=fallback
bool suffix_index_fallback_only();

/// --lookup-chain: the shared index as a draft source beside the request's own text (draft_source.hpp).
class SuffixIndexSource final : public DraftSource {
public:
    explicit SuffixIndexSource(const SharedSuffixIndex& idx) : idx_(idx) {}
    const char* name() const override { return "suffix-index"; }
    int propose(const int32_t* tail, int n_tail, int n_pending, int max_k, int32_t* out, int* match) override {
        (void) n_pending;   // the pending drafts are part of the context the index matches
        return idx_.propose(tail, n_tail, max_k, out, match);
    }

private:
    const SharedSuffixIndex& idx_;
};

}  // namespace strata::spec
