// include/strata/spec/disco.hpp - DISCO: a learned stop rule for the MTP draft chain (opt-in, default off).
//
// DISCO (Mamou et al., "Dynamic Speculation Lookahead Accelerates Speculative Decoding of Large Language Models",
// Intel Labs with Weizmann / Hebrew U, arXiv 2405.04304) evaluates a small classifier after every drafted token and
// stops the chain (verifies) when it predicts the token will be rejected.  The paper reports ~10% over a tuned
// static lookahead.  Here the classifier is a logistic regression over a handful of features, fitted offline
// (tools/pancake/disco_fit.py) on traces this file's logger writes during normal decoding:
//
//   s_j = sigmoid(b + sum_i w_i * (x_ij - mean_i) / scale_i)    = P(draft j accepted | drafts 0..j-1 accepted)
//   draft j is kept and the chain goes on while s_j >= threshold; the first draft under it ends the chain and is
//   left out of the window.
//
// Features of draft j (all known on the host when the MTP returns draft j; DiscoPolicy::features):
//   0 p        the MTP's top-1 probability of draft j
//   1 surprise -log p / 5
//   2 chain    p_0 * .. * p_j (the drafter's own survival estimate)
//   3 pos      j / 8
//   4 recent   EMA of the acceptance of MTP drafts that were reached (alpha 0.1, prior 0.7)
//   5 match    log1p(the prompt-lookup match length of the context at the round) / 3 (0 without --suffix-draft)
//   6 miss     EMA of missed experts per verify row and layer (the offloaded-MoE part: what one more row wakes)
//   7 dcost    cost(j + 2) / cost(j + 1) - 1: the measured price of verifying one more row (DraftPolicy's row costs)
//   8 ctx      log2(1 + position) / 17
//
// The policy only decides how many of the drafts are verified; the verifier still accepts exactly the drafts the
// target would have produced, so greedy output is unchanged (src/spec/disco_test.cpp checks this on a mock target).
//
// Switches:  STRATA_DISCO=<weights file>        the learned rule replaces --spec-min-p / STRATA_SPEC_COST for MTP windows
//            STRATA_DISCO_TRACE=<csv file>      log one line per drafted token of every plain MTP window (training data)
// A missing or malformed weights file prints one warning and keeps the current rule (DraftPolicy / --spec-min-p).
#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::spec {

class DraftPolicy;

struct DiscoModel {
    static constexpr int kF = 9;
    static constexpr int kVersion = 1;
    double bias = 0.0;
    std::array<double, kF> w{}, mean{}, scale{};
    double threshold = 0.5;
    bool loaded = false;

    /// Parse the text weights file ("disco 1", "bias b", "w ..", "mean ..", "scale ..", "threshold t"; '#' comments).
    bool load(const std::string& path, std::string& err);
    bool parse(const std::string& text, std::string& err);
    /// P(accepted) for one feature vector.
    double score(const float* x) const;
};

class DiscoPolicy {
public:
    static constexpr int kF = DiscoModel::kF;
    static constexpr int kMaxT = 16;

    /// Read STRATA_DISCO / STRATA_DISCO_TRACE; `who` prefixes the messages ("strata generate", "strata serve").
    void init_from_env(const char* who);
    /// For tests: a model given directly (and an optional trace file).
    void set_model(const DiscoModel& m) { model_ = m; }
    bool open_trace(const std::string& path);
    ~DiscoPolicy();

    bool active() const { return model_.loaded; }          // the learned rule picks the MTP window
    bool tracing() const { return trace_ != nullptr; }
    bool enabled() const { return active() || tracing(); } // anything to keep current

    /// The feature vector of draft `idx` with probability `p` and chain survival `chain` (product of p_0..p_idx),
    /// from the state left by the last end_round().
    void features(int idx, float p, double chain, float* out) const;
    /// The chain rule (MtpDrafter::set_chain_rule): draft `idx` (0-based) with probability `p` was just computed.
    /// Returns whether to draft idx + 1; rows() is then the window to verify (1 + the drafts kept).  `cap` = the
    /// widest window (rows).
    bool keep_draft(int idx, float p, int cap);
    int rows() const { return rows_; }

    struct Round {
        const float* p = nullptr;     ///< the probabilities of the window's MTP drafts (draft j: p[j])
        int T = 1;                    ///< the verified window's MTP rows (1 + MTP drafts in it)
        int accepted = 0;             ///< drafts of the window that held
        bool mtp_window = false;      ///< a plain MTP window (not a lookup, a chain, a merged or the first window)
        double round_ms = 0.0;        ///< the round's time (0: unknown)
        int match = 0;                ///< the prompt-lookup match length seen this round (0: none)
        int64_t misses_total = 0;     ///< the expert pool's cumulative missed experts (drive.d.multi_misses)
        int n_layers = 1;
        long long pos = 0;            ///< the context position of the round's first row
        const DraftPolicy* policy = nullptr;   ///< its row costs (dcost); null keeps the last snapshot
    };
    /// After a verified round, before the next chain is drafted: the trace lines of this round's drafts (features
    /// as they were when the chain was drafted), then the state the next chain's features read.
    void end_round(const Round& r);

    double recent() const { return recent_; }
    double miss_rate() const { return miss_; }
    int64_t rounds_logged() const { return logged_; }

private:
    DiscoModel model_;
    std::FILE* trace_ = nullptr;
    int64_t logged_ = 0, round_id_ = 0;
    int rows_ = 1, kept_ = 0;
    double chain_ = 1.0;
    // state read by features()
    double recent_ = 0.7, miss_ = 0.0, match_f_ = 0.0, ctx_f_ = 0.0;
    std::array<double, kMaxT + 2> cost_{};    // cost_[t]: round ms of a window of t rows (0: unknown)
    int64_t misses_last_ = -1;
};

}  // namespace strata::spec
