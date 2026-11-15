// include/strata/spec/disco.hpp - DISCO: a learned stop rule for the MTP draft chain (opt-in, default off).
//
// DISCO (Mamou et al., "Dynamic Speculation Lookahead Accelerates Speculative Decoding of Large Language Models",
// Intel Labs with Weizmann / Hebrew U, arXiv 2405.04304) evaluates a small classifier after every drafted token and
// stops the chain (verifies) when it predicts the token will be rejected.  The paper reports ~10% over a tuned
// static lookahead.  Here the classifier is a logistic regression over a handful of features, fitted offline
// (tools/pancake/disco_fit.py) on traces this file's logger writes during normal decoding:
//
//   s_j = sigmoid(b + sum_i w_i * (x_ij - mean_i) / scale_i)    = P(draft j accepted | drafts 0..j-1 accepted)
//   draft j is kept and the chain goes on while s_j >= threshold_j; the first draft under it ends the chain and is
//   left out of the window.
//
// Features of draft j (all known on the host when the MTP returns draft j; DiscoPolicy::features):
//   0 p        the MTP's top-1 probability of draft j
//   1 surprise -log p / 5
//   2 chain    p_0 * .. * p_j (the drafter's own survival estimate)
//   3 pos      j / 8
//   4 recent   EMA of the acceptance of MTP drafts that were reached (alpha 0.1, prior 0.7)
//   5 match    log1p(the prompt-lookup match length of the context at the round) / 3 (0 without --suffix-draft)
//   6 miss     the CPU-missed experts row idx + 1 is expected to add (A22, version 2): an EMA, per window row, of the new
//              CPU misses that row added in earlier MTP windows (core/row_miss.hpp; the window trace's record 5), / 64.
//              The SpecFit cost of a draft position (arXiv 2610.15417): the CPU time of the experts it wakes.  A row not
//              seen yet takes the nearest shallower row's value (0 before any window).
//   7 dcost    cost(j + 2) / cost(j + 1) - 1: the measured price of verifying one more row (DraftPolicy's row costs)
//   8 ctx      log2(1 + position) / 17
//
// The policy only decides how many of the drafts are verified; the verifier still accepts exactly the drafts the
// target would have produced, so greedy output is unchanged (src/spec/disco_test.cpp checks this on a mock target).
//
// The drafter's entropy (DISCO's and MiMo-V2-Flash's signal; LEADS.md B24) is exposed only with STRATA_DISCO_ENTROPY: the
// draft head's softmax entropy of each draft then reaches the host (MtpDrafter::chain_entropy; -1 where the two-stage proxy
// pick never forms the full softmax).  It is not one of the kF features (version 2 files stay valid): an optional
// "went w" line adds w x H to the logit, the trace gets an "ent" column for the refit, and STRATA_DISCO_ENTROPY=<h> itself
// stops the chain at the first draft whose entropy is above h (Entropy Sentinel's rule; alone, it replaces --spec-min-p
// for MTP windows, every draft kept up to the cap otherwise).
//
// Switches:  STRATA_SPEC_CUM=<s in 0..1>        F8: stop the chain where the product of the drafts' probabilities falls under s
//                                              (a DISCO model on the chain feature alone; STRATA_DISCO wins when both are set)
//            STRATA_DISCO=<weights file>        the learned rule replaces --spec-min-p / STRATA_SPEC_COST for MTP windows
//            STRATA_DISCO_TRACE=<csv file>      log one line per drafted token of every plain MTP window (training data)
//            STRATA_DISCO_ENTROPY=<h nats>      the drafts' entropy: a stop rule (H > h ends the chain) and the "went" feature
// A missing or malformed weights file prints one warning and keeps the current rule (DraftPolicy / --spec-min-p).
// Weights are fitted by tools/pancake/disco_fit.py, from these CSV traces or from window traces (STRATA_WINDOW_TRACE),
// whose fits leave the features a window trace cannot replay (5 match, 7 dcost) at weight 0.  Version 1 files (feature 6
// was a window-level miss rate) are refused: refit.
#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

namespace strata::spec {

class DraftPolicy;

struct DiscoModel {
    static constexpr int kF = 9;
    static constexpr int kVersion = 2;   // 2: feature 6 = the row's expected new CPU misses (A22)
    double bias = 0.0;
    std::array<double, kF> w{}, mean{}, scale{};
    double threshold = 0.5;
    double went = 0.0;                   ///< optional "went w": w x the draft's entropy added to the logit (LEADS.md B24)
    /// Optional per-position thresholds ("thresholds n t_0 .. t_n-1", A22): draft j is judged against t_j (the last one
    /// for deeper drafts), so the table carries what each position costs (SpecFit) on top of how likely it holds (DISCO).
    /// 0 keeps every draft at that position, 1 none.  Absent: `threshold` everywhere.
    static constexpr int kMaxPos = 16;
    std::array<double, kMaxPos> thr_pos{};
    int n_thr_pos = 0;
    double threshold_at(int idx) const {
        return n_thr_pos > 0 ? thr_pos[(size_t) (idx < n_thr_pos ? (idx < 0 ? 0 : idx) : n_thr_pos - 1)] : threshold;
    }
    bool loaded = false;

    /// Parse the text weights file ("disco 2", "bias b", "w ..", "mean ..", "scale ..", "threshold t", optional
    /// "thresholds n t..", '#' comments).
    bool load(const std::string& path, std::string& err);
    bool parse(const std::string& text, std::string& err);
    /// P(accepted) for one feature vector.
    double score(const float* x) const;
    /// score() with went x `h` added to the logit (h < 0: unknown, nothing added).
    double score_h(const float* x, float h) const;
    /// STRATA_SPEC_CUM=s (MOD_PLAN F8, android-imdad's rule): keep a draft while the chance that every draft so far holds, the
    /// product p_0 .. p_j of the MTP's probabilities (feature 2), is >= s.  A model on that one feature: sigmoid(chain) is
    /// monotonic, so the threshold sigmoid(s) is the cut chain >= s.  Unlike the fork it has no separate --spec-min-p floor
    /// (chain <= p, so s >= min_p implies it).
    static DiscoModel cumulative(double s);
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
    bool keep_draft(int idx, float p, int cap, int32_t tok = -1);
    /// STRATA_DISCO_ENTROPY (LEADS.md B24): draft `idx`'s entropy (nats, -1 unknown), set before keep_draft(idx, ..).
    void set_entropy(int idx, float h) { if (idx >= 0 && idx < kMaxT) ent_hist_[(size_t) idx] = h; }
    /// For tests: the entropy stop threshold (0 = off) and whether the trace carries the "ent" column.
    void set_entropy_rule(double h_max, bool trace_col) { ent_max_ = h_max; ent_col_ = trace_col; }
    double entropy_max() const { return ent_max_; }
    int rows() const { return rows_; }
    /// STRATA_SPEC_LOOKUP_Q / STRATA_SPEC_DEPTH_DECAY (opt-in; android-imdad/Strata-7900xtx ef42c2fde9, LEADS.md FK-21b):
    /// the next chain's product counts `q_agree` for a draft equal to the suffix lookup's proposal at its place
    /// (`lookup`, n tokens), and p * decay^idx for any other.  Never called (q 0, decay 1): the plain product of p.
    void set_stop_hint(const int32_t* lookup, int n, double q_agree, double decay) {
        hint_n_ = n < 0 ? 0 : n > 8 ? 8 : n;
        for (int i = 0; i < hint_n_; ++i) hint_tok_[i] = lookup[i];
        hint_q_ = q_agree;
        hint_decay_ = decay;
    }

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
        const int32_t* row_miss = nullptr;     ///< the new CPU misses of each verified row (core/row_miss.hpp; null: unknown)
        int row_miss_n = 0;                    ///< rows in row_miss
    };
    /// After a verified round, before the next chain is drafted: the trace lines of this round's drafts (features
    /// as they were when the chain was drafted), then the state the next chain's features read.
    void end_round(const Round& r);

    double recent() const { return recent_; }
    double miss_rate() const { return miss_; }
    /// Feature 6 before its / 64: the expected new CPU misses of window row `row` (1 = the first draft).
    double row_miss_expect(int row) const;
    int64_t rounds_logged() const { return logged_; }

private:
    DiscoModel model_;
    std::FILE* trace_ = nullptr;
    int64_t logged_ = 0, round_id_ = 0;
    int rows_ = 1, kept_ = 0;
    double chain_ = 1.0;
    double ent_max_ = 0.0;                    // STRATA_DISCO_ENTROPY: stop above this entropy (0 = off)
    bool ent_col_ = false;                    // the trace's "ent" column
    std::array<float, kMaxT> ent_hist_{};     // the entropies of the current chain's drafts (-1 unknown)
    int32_t hint_tok_[8] = {};   // STRATA_SPEC_LOOKUP_Q (set_stop_hint)
    int hint_n_ = 0;
    double hint_q_ = 0.0, hint_decay_ = 1.0;
    // state read by features()
    double recent_ = 0.7, miss_ = 0.0, match_f_ = 0.0, ctx_f_ = 0.0;
    std::array<double, kMaxT + 2> cost_{};    // cost_[t]: round ms of a window of t rows (0: unknown)
    std::array<double, kMaxT + 1> nm_{};      // nm_[r]: EMA of the new CPU misses window row r added (MTP windows)
    std::array<int, kMaxT + 1> nm_n_{};       // windows that measured row r
    int64_t misses_last_ = -1;
};

}  // namespace strata::spec
