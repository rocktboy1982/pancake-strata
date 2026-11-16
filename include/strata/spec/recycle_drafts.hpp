// include/strata/spec/recycle_drafts.hpp - three draft sources beside the suffix lookup, each behind its own default-off switch
// (LEADS.md section 3 rows "Token Recycling drafter", "SSD-lite (Saguaro)", "Staged SD / Hydra top-2 tree"; MOD_PLAN.md section 6
// item 18).  None of them changes an output under greedy verify: they only choose which drafts the verify window checks.
//
//   TokenRecycler   (STRATA_TOKEN_RECYCLE=1)  Token Recycling (Luo et al., "Turning Trash into Treasure: Accelerating Inference of
//                   Large Language Models with Token Recycling", arXiv 2408.08696): an adjacency matrix holds, for every token, the
//                   candidate tokens the target produced after it; a draft follows the matrix from the last token.  The paper fills
//                   each row with the top-k of the decode step's logits (k = 8, < 2 MB at a 32K vocabulary) and walks a static
//                   BFS tree of ~80 nodes; this tree has no top-k on the host and no tree attention, so a row holds the target's
//                   argmax of every verify row whose input was that token (accepted or rejected rows alike: the "trash" the paper
//                   recycles), most recent first, and the draft is the matrix's best chain (first version).
//   OutcomeDrafter  (STRATA_SSD_LITE=1)  Speculative speculative decoding (Kumar, Dao, May, "Speculative Speculative Decoding",
//                   Saguaro, arXiv 2603.03251): while the verify window runs, a worker thread prepares the next round's draft for
//                   the likely verification outcomes (accepted count a, bonus token b), so the draft is ready the moment the
//                   outcome is known.  Saguaro runs a dedicated drafter GPU; here the drafter is the CPU n-gram lookup
//                   (SuffixDrafter::propose_after_const) and the outcomes are, per a, the lookup's own guess of b.
//                   STRATA_HYDRA_TOP2=1 (with the worker) adds, per a, the runner-up branch: the matrix's best successor of the
//                   last accepted token other than the draft that would be rejected (Hydra top-2 / staged speculative decoding,
//                   Ankner et al. arXiv 2402.05109; Spector and Re arXiv 2308.04623).  A linear verify window cannot hold both
//                   branches of a tree, so the branch is staged: when the window is cut exactly there and the target's bonus is
//                   the runner-up, the branch's continuation is the next round's lookup draft (first version).
#pragma once

#include "strata/spec/suffix_drafter.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::spec {

/// STRATA_TOKEN_RECYCLE=1 (read once): the matrix's chain is a draft when the suffix lookup proposes nothing.
bool token_recycle_env();
/// STRATA_SSD_LITE=1 (read once): next-round drafts prepared during the verify window (OutcomeDrafter).
bool ssd_lite_env();
/// STRATA_HYDRA_TOP2=1 (read once): the runner-up branch among the prepared outcomes (needs STRATA_SSD_LITE=1).
bool hydra_top2_env();

class TokenRecycler {
public:
    static constexpr int K = 8;                    ///< candidates per token (the paper's k)
    static constexpr int32_t kMaxVocab = 1 << 20;  ///< ids at or above are ignored (the matrix grows to the largest id seen)

    void clear() { succ_.clear(); }
    /// A verify window: row t's input was window[t], its target pick out[t].  Every row is a (token, next) pair.
    void observe(const int32_t* window, const int32_t* out, int T);
    /// One pair: `next` becomes tok's first candidate (an older copy of it moves up, the oldest of K falls out).
    void push(int32_t tok, int32_t next);
    /// The chain tok -> first candidate -> ..., up to max_k tokens; returns how many.
    int propose(int32_t tok, int max_k, int32_t* out) const;
    /// tok's candidates, best first (up to K); returns how many.
    int row(int32_t tok, int32_t* out) const;
    /// tok's best candidate that is not `exclude`, or -1.
    int32_t alt(int32_t tok, int32_t exclude) const;
    /// tok's best candidate, or -1.
    int32_t first(int32_t tok) const { return alt(tok, -1); }
    size_t bytes() const { return succ_.size() * sizeof(int32_t); }

private:
    std::vector<int32_t> succ_;   // (largest id + 1) x K, -1 = empty
};

class OutcomeDrafter {
public:
    static constexpr int kMaxT = 32;        ///< windows wider than this are not prepared
    static constexpr int kMaxK = 32;        ///< draft tokens per outcome
    static constexpr int kPerA = 3;         ///< candidate bonus tokens per accepted count

    OutcomeDrafter() = default;
    ~OutcomeDrafter();
    OutcomeDrafter(const OutcomeDrafter&) = delete;
    OutcomeDrafter& operator=(const OutcomeDrafter&) = delete;

    /// Before the verify window: prepare drafts for its outcomes on the worker thread.  `sfx` and `rec` (may be null) are read
    /// only, and must not change until wait() returns; `window` is copied.  `top2`: also the runner-up branch (needs `rec`).
    void start(const SuffixDrafter& sfx, const TokenRecycler* rec, const int32_t* window, int T, int max_k, bool top2);
    /// After the verify window, before the suffix drafter or the matrix change: the worker is done.  A round that does not
    /// call start() leaves no outcome to take.
    void wait();
    /// The outcome the window committed: a drafts accepted, bonus b, so the next round starts at position `next_pos`.
    void finish(int a, int32_t b, int64_t next_pos);
    /// The next round's draft at `pos`, if the committed outcome was prepared: copies up to max_k tokens, sets *match (the
    /// lookup's match length; 0 for a matrix chain) and returns the count; 0 when nothing was prepared for it.
    int take(int64_t pos, int max_k, int32_t* out, int* match);
    /// Prepared outcomes found / asked (for the log).
    int64_t hits() const { return hits_; }
    int64_t asks() const { return asks_; }
    /// The job itself, synchronously (the unit test's entry; start() runs it on the worker).
    void compute(const SuffixDrafter& sfx, const TokenRecycler* rec, const int32_t* window, int T, int max_k, bool top2);

private:
    struct Entry {
        int a = 0;
        int32_t b = -1;
        int n = 0, match = 0;
        int32_t tok[kMaxK];
    };
    void run();

    std::vector<Entry> ent_;
    int n_ent_ = 0;
    // the job (written by start, read by the worker)
    const SuffixDrafter* sfx_ = nullptr;
    const TokenRecycler* rec_ = nullptr;
    int32_t win_[kMaxT] = {};
    int T_ = 0, max_k_ = 0;
    bool top2_ = false;
    // the committed outcome
    int fa_ = -1;
    int32_t fb_ = -1;
    int64_t fpos_ = -1;
    int64_t hits_ = 0, asks_ = 0;
    // the worker
    std::thread thr_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool job_ = false, busy_ = false, quit_ = false;
    bool started_ = false;   // start() submitted the current window (finish() records its outcome only then)
};

}  // namespace strata::spec
