// include/strata/core/mtp.hpp - plan v0.3 P6: the MTP draft layer on the GPU.
//
// The model ships one multi-token-prediction layer (vLLM 0.30.0 `qwen4_exp` MTP; transcribed and validated in
// `tools/mtp_probe.py`, where it accepts 0.89 / 0.86 / 0.85 of greedy drafts at steps 1-3).  Cell i of its
// sequence pairs the main model's final multi-stream residual at position i with the token at position i+1, at
// rope position i (vLLM's convention); its output predicts the token at position i+2, and its own residual feeds
// the next draft step.
//
//   e = fc_embedding(rms(embed(tok)) * (1+w))           h = fc_hidden per stream(rms_10240(R) * (1+w))
//   R = h + e  ->  attention hyper-connection  ->  QSA attention (own K/V)  ->  MLP hyper-connection  ->
//   MoE (512 experts, top-10, shared expert)  ->  R += y * 2 sigmoid(inject / hc)  ->  final mixer  ->  head
//
// Runtime choices, each a trade of exactness for simplicity that only affects DRAFT quality, never the output
// (the verify window decides every emitted token):
//   * the large projections run as Q8_0 through the multi-column MMVQ (`tools/mtp_rt.py` quantizes them);
//   * the attention is DENSE over every cell the layer has seen - identical to the model's sparse selection
//     below 2,051 cells - so speculative cells (draft steps, rejected window rows) never touch indexer state and
//     are simply overwritten when their positions are processed again;
//   * all 512 routed experts live in VRAM (708 MB) and run through the grouped hit kernels.
#pragma once

#include "strata/core/layer.hpp"
#include "strata/core/pinned_ring.hpp"
#include "strata/core/session.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/spec/window_rule.hpp"   // EntropyStop (STRATA_SPEC_ENTROPY)

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <map>
#include <vector>

namespace strata::core {

class NativeHead;

class MtpDrafter {
public:
    MtpDrafter() = default;
    ~MtpDrafter();
    MtpDrafter(const MtpDrafter&) = delete;
    MtpDrafter& operator=(const MtpDrafter&) = delete;

    /// Loads `rt_dir` (from tools/mtp_rt.py) and allocates the layer's K/V and buffers for up to `max_t` rows.
    /// Call before the VRAM expert tier is sized: this takes ~0.9 GB.
    bool load(const std::string& rt_dir, const ModelGeometry& g, SessionState& ss, int max_t, std::string& err,
              int64_t window = 32768, const MtpDrafter* shared = nullptr);
    /// The prompt's length: prefill() skips the cells the attention window can never reach again.
    void set_prompt_len(int64_t n) { prompt_len_ = n; }
    /// STRATA_ROUTE_RESIDENT_MTP=1 (EXPERIMENTAL, opt-in, changes the output): the draft layer's router is biased like
    /// the verifier's STRATA_ROUTE_RESIDENT, with `res_row` (a device residency row of 512 experts: slot or -1) as the
    /// table.  nullptr switches it off.
    void set_route_resident(const int32_t* res_row, float margin, int lo, int hi) {
        rr_res_ = res_row; rr_margin_ = margin; rr_lo_ = lo; rr_hi_ = hi;
    }
    /// At most this many drafts per round (below max_t - 1): a window longer than the MTP's comes from elsewhere.
    void set_max_drafts(int k) { max_drafts_ = k; }
    /// --mtp-hnorm stream (opt-in; before the first draft or prefill): pre_fc_norm_hidden normalizes each
    /// hyper-connection stream on its own, as llama.cpp's qwen4exp MTP graph does, instead of one RMS over all four.
    void set_hnorm_per_stream(bool on) { hnorm_stream_ = on; }
    bool hnorm_per_stream() const { return hnorm_stream_; }
    /// --mtp-q4 (opt-in; before load): the draft layer's Q8_0 projections and its draft head run from 4-bit (Q4_0)
    /// copies made at load/bind - fewer bytes per draft step; drafts only, the verify window decides every token.
    /// (The prompt path's batched K/V pass keeps the Q8_0 originals.)
    void set_q4(bool proj, bool head) { q4_ = proj; q4_head_ = head; }
    /// --mtp-draft-vocab FILE (opt-in; before bind): the draft head's token subset from FILE instead of
    /// rt/draft_vocab.bin (e.g. data/draft_vocab_en.bin, 40K tokens)
    void set_draft_vocab(const std::string& path) { dvocab_path_ = path; }
    /// DFlash2 shares this drafter's gathered draft-head subset when it is the very same bytes (DF-05): the rows of the main head's vocabulary
    /// subset in the main head's own layout, on `dev`, gathered from `vocab_path`.  False (and the DFlash2 gathers its own) for a Q4_0 copy
    /// (--mtp-q4 head), a repacked Q6_K / Q8_0 subset, a head on another card, another vocabulary file or no subset at all.
    /// The drafter keeps the ownership: it outlives DFlash2 (the fallback), and the rows are never written after bind.
    struct DraftHeadView { const uint8_t* rows = nullptr; const int32_t* vocab_dev = nullptr; const std::vector<int32_t>* vocab_host = nullptr; int64_t n = 0; int type = -1; };
    bool draft_head_view(const NativeHead* head, const std::string& vocab_path, DraftHeadView& out) const;
    /// STRATA_MTP_TOP2=1 (diagnostic): draft j's runner-up token in the last draft() (-1 = unknown)
    static bool top2_env();
    int32_t top2(int j) const { return j >= 0 && j < (int) top2_.size() ? top2_[(size_t) j] : -1; }
    int max_t() const { return max_t_; }
    uint64_t vram_bytes() const { return vram_; }

    /// THE PROMPT ROOM (STRATA_MTP_ROOM_MIN).  The layer's 512 routed experts (708 MB) are read by the DECODE alone:
    /// `record_rest` is their only reader, and every prompt pass - the capture's K/V (record_front) and the prompt
    /// path's Prefill::draft_kv - touches the dense tensors and the K/V state instead.  So a long prompt may take
    /// their bytes: the prompt path's ring is pointed AT this block (Prefill::set_ring_room_base), and every expert
    /// blob it stages overwrites an expert.  `experts_lend` copies the block into pinned host memory first (taken
    /// once, so the `experts_reclaim` after the prompt is a DMA and not another read of experts.bin); `reclaim`
    /// copies it back IN PLACE.
    /// THE BLOCK IS NEVER FREED, NEVER REALLOCATED AND NEVER MOVES: the round and step graphs were captured with its
    /// address as the MoE kernel's argument, and lending it in place is what keeps them valid - nothing is dropped
    /// and no round pays a re-capture.
    /// While the experts are lent no decode pass may run: `draft`, `draft_first`, `chain_launch` and `record_rest`
    /// refuse (with `err`), rather than read the prompt's staged bytes as experts.
    uint64_t experts_bytes() const;    ///< 0: this drafter does not own the experts (shared weights: --batch-mtp)
    void* experts_base() const { return experts_; }   ///< the bytes the room's slots are carved from
    bool experts_resident() const { return experts_ != nullptr && !experts_lent_; }
    bool experts_lend(std::string& err);      ///< snapshot to pinned host (the block stays in place)
    bool experts_reclaim(std::string& err);   ///< the snapshot back into the block, at the address it never left

    /// The draft layer's K/V state (read-only: --serve's STRATA_STATE_HASH check hashes it)
    const QsaState& kv_state() const { return st_; }
    /// KV streaming: refill the ring of the drafter's window from its host copy for a sequence that continues at
    /// `upto` (a conversation-cache resume). No-op unless the drafter's K/V is a ring.
    void kv_restore(int64_t upto);
    /// The VRAM bind() will allocate for a native head of `head_row_bytes` per vocabulary row: the draft logits and
    /// the draft head over rt/draft_vocab.bin's subset.  The expert cache is sized before bind(), so it reserves this.
    uint64_t bind_bytes(uint64_t head_row_bytes, int64_t n_vocab) const;
    /// The main model's embedding and head, and the verify window's final residuals (T rows, hc*n_embd each).
    bool bind(const WeightTable& wt, const NativeHead* head, const float* window_R, std::string& err,
              const MtpDrafter* shared = nullptr);
    void set_window_R(const float* window_R) { window_R_ = window_R; }

    /// Prompt cells [cell0, cell0 + n): residual rows `R_rows` (device, hc*n_embd each) and `next_tokens` (host,
    /// the token at position cell+1).  Runs in batches of up to max_t rows.  `sync` false: returns without waiting
    /// for the drafter's stream (the caller orders on it: `stream()`).
    bool prefill(const float* R_rows, const int32_t* next_tokens, int64_t n, int64_t cell0, std::string& err,
                 bool sync = true);
    /// --pipeline-windows: capture the prefill graphs and size its input records for windows of up to max_t rows now,
    /// so a later prefill(sync = false) of a window neither captures nor allocates (both wait for the device).
    bool prepare_prefill(std::string& err);
    /// A prompt read beside running batch windows (STRATA_PIPED_CONCURRENT): the prompt pass's graphs captured and its
    /// input records sized for chunks of up to n rows now, with nothing in flight.  On the read's thread a capture or
    /// a free waits for the device, whose windows wait for the main thread's doorbell service (#31).
    bool reserve_prefill(int64_t n, std::string& err);
    cudaStream_t stream() const { return cs_; }

    /// Successful producer result, including a generated row that failed admission.
    /// budget is the actual producer max_steps; failure/entry resets both fields.
    /// With result supplied, n_drafts (if supplied too) is also reset on entry.
    struct DraftResult { int generated = 0; int budget = 0; };

    /// One round: catch-up over T cells from `p` (rows = the window's final residuals, `tokens` = the window's
    /// argmaxes: row t pairs R_{p+t} with the token at p+t+1), then the draft chain from row `a` (the last
    /// accepted row) for T-1 drafts at cells p+a+1 ...  `drafts` gets T-1 tokens.
    bool draft(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
               float* probs = nullptr, float min_p = 0.0f, int* n_drafts = nullptr, DraftResult* result = nullptr);

    /// STRATA_DRAFT_EARLY (mandoof1 b8004eb8d7; the serve loop): draft() in two halves.  draft_launch stages the round's
    /// inputs, launches the round (every step too on the eager path) and returns at once; draft_finish is the rest of
    /// draft() (the host rule over the steps, the final wait, the outputs).  draft() is the two back to back.  The window's
    /// commit and emit may run on the host between them: the round reads nothing the commit graph writes.  `committed`:
    /// the n_committed tokens the commit keeps, for the PLE prefetch's history (Verifier::commit advances the session's
    /// ple_prev on the host; here the commit comes later).  A launched round must always be finished.
    bool draft_launch(int T, const int32_t* tokens, int64_t p, int a, float min_p, std::string& err,
                      const int32_t* committed = nullptr, int n_committed = 0);
    bool draft_finish(int32_t* drafts, std::string& err, float* probs = nullptr, int* n_drafts = nullptr,
                      DraftResult* result = nullptr);
    bool draft_launched() const { return run_.live; }

    /// ---- STRATA_DRAFT_SUFFIX_SKIP (mandoof1 868841cb32; LEADS.md FK-33; the serve loop): a chain that ends after draft 0
    /// when the next window will take the suffix lookup whatever the chain would draft.  `fn` is asked once draft 0 and
    /// its probability are known and the host rule (min_p) would go on; true ends the round there: no step graph is
    /// launched, the drafts past 0 read 0, and the chain's state (the step inputs, the staged step rows) is kept so
    /// draft_resume() can still run the steps.  One-shot: the next draft_finish() takes and clears it.  Only on the
    /// step-by-step path without a chain rule or an entropy stop (their calls have side effects); elsewhere never asked.
    typedef bool (*SkipAfterFirstFn)(void* user, int32_t draft0, float p0);
    void set_skip_after_first(SkipAfterFirstFn fn, void* user) { skip_fn_ = fn; skip_user_ = user; }
    /// True after a draft() that ended on the predicate: draft_resume() may follow (before the next draft()).
    bool draft_resumable() const { return resume_.valid; }
    /// The steps the last draft() left out, by the same host rule, into `drafts` / `probs` (draft 0 is written again):
    /// what that draft() would have returned had the predicate said no.
    bool draft_resume(int32_t* drafts, std::string& err, float* probs = nullptr, int* n_drafts = nullptr);
    /// LEADS.md B28 STRATA_MTP_IN_VERIFY=1 (opt-in, exact under greedy; needs STRATA_DEVICE_ACCEPT=1 and STRATA_MTP_ONE_GRAPH=1): the
    /// next round rides the verify window's stream.  iv_prepare (before the window, host): true when this window's round can
    /// - the eager chain (min_p 0 or STRATA_MTP_EAGER, no chain rule, no entropy stop), not coupled, T <= max_t(), the whole
    /// chain inside the K/V - and the chain graph is captured.  iv_enqueue is the Verifier's after-commit hook: the staged
    /// inputs written on the device from the accepted row (mtp_stage_from_accept), then the chain graph, behind the commit on
    /// the window's stream.  iv_adopt (after the window, host): the queued round becomes the launched round draft_finish
    /// collects (as draft_launch's), when the device drafted from the caller's accepted row `a`; iv_drop forgets a round
    /// nobody collects (the request ended).  The round covers the window's T rows, as STRATA_MTP_CATCHUP_ALL does.
    static bool in_verify_env();
    bool iv_prepare(int T, int64_t p, float min_p, std::string& err);
    bool iv_enqueue(void* stream, const int32_t* d_out, const int32_t* d_commit, std::string& err);
    bool iv_adopt(const int32_t* tokens, int a, const int32_t* committed, int n_committed, std::string& err);
    void iv_drop() { iv_ = IvState{}; }

    /// --lookup-max-t: draft() for a window of up to kVerifyLongT rows (T may pass max_t()).  When the last accepted
    /// row is past max_t() rows, rows 0..a-1 are caught up as prefill() does and row a runs as a one-row round, so the
    /// drafter's K/V holds every accepted cell and its graphs keep max_t() rows.
    bool draft_long(int T, const int32_t* tokens, int64_t p, int a, int32_t* drafts, std::string& err,
                    float* probs = nullptr, float min_p = 0.0f, int* n_drafts = nullptr, DraftResult* result = nullptr);

    /// The first round: one cell (`cell`) from `R_row` (device) and `token` -> T-1 drafts.
    /// STRATA_SPEC_COST: called after each draft step with (index, probability) instead of the `min_p` test; false ends the chain.
    /// Also called once for the last step of a chain that reached its length (so the rule sees every draft). Empty: min_p decides.
    void set_chain_rule(std::function<bool(int, float)> f) { chain_rule_ = std::move(f); }
    /// Inside a chain rule call (index j): draft j's token, as draft() reads it (STRATA_SPEC_LOOKUP_Q).  -1: none.
    int32_t chain_draft(int j) const { return h_out_ != nullptr && j >= 0 ? ((const volatile int32_t*) h_out_)[j] : -1; }
    /// LEADS.md section 3 (STRATA_DRAFT_ENTROPY=1 / STRATA_SPEC_ENTROPY): the last draft() round's entropy (nats) and runner-up
    /// probability of draft j (window row j + 1), copied when the round returned.  NaN when the switch is off, for a row past the
    /// round's drafts, or for a draft drawn without the full softmax (coupled sampling).
    float draft_entropy(int j) const { return h_ent_ != nullptr && j >= 0 && j < 16 ? round_ent_[j] : std::numeric_limits<float>::quiet_NaN(); }
    float draft_top2(int j) const { return h_top2_ != nullptr && j >= 0 && j < 16 ? round_top2_[j] : std::numeric_limits<float>::quiet_NaN(); }
    /// STRATA_SPEC_ENTROPY (window_rule.hpp EntropyStop): a rule decides, after each draft, whether the chain goes on.
    bool entropy_stop_on() const { return ent_stop_.on(); }
    /// After a verified plain MTP window whose `n` drafts came from the last draft() round, `a` of them held (AdaEDL's
    /// threshold, AdaSD's rejected-entropy mean).  A no-op without a rule.
    void entropy_observe(int n, int a) { if (ent_stop_.on()) ent_stop_.observe(std::min(n, 16), a, round_ent_); }
    /// LEADS.md B24 (STRATA_DISCO_ENTROPY, opt-in): inside a chain rule call, draft j's softmax entropy under the draft
    /// head (nats).  -1: unknown (switch off, the proxy or coupled pick, which never form the full softmax).
    float chain_entropy(int j) const { return h_dent_ != nullptr && j >= 0 ? ((const volatile float*) h_dent_)[j] : -1.0f; }

    bool draft_first(int T, const float* R_row, int32_t token, int64_t cell, int32_t* drafts, std::string& err,
                     float* probs = nullptr, float min_p = 0.0f, int* n_drafts = nullptr, DraftResult* result = nullptr);

    /// COUPLED DRAFT SAMPLING (core/coupled_draft.hpp; STRATA_SPEC_COUPLED=1, set up by bind()): the request's
    /// sampling.  A sampled request (temperature > 0, not greedy) then drafts by SAMPLING with the target's chain and
    /// the target's Philox draw for the verifying row, and `probs` is the draft's probability under that chain; a
    /// greedy one keeps the argmax drafts and their graphs.  A no-op when the switch is off.
    void set_draft_sampling(const strata::kernels::SamplerParams& sp);
    /// Coupled mode with penalties: the history the next draft() chain starts from - what the session holds after the
    /// window's commit (`tail`) and the window's pick at its last accepted row (`next`, the next window's row 0).
    void set_draft_history(const int32_t* tail, int64_t n_tail, int32_t next);
    void set_ple_session(SessionState* s) { ple_ss_ = s; }
    std::function<bool(int, float)> chain_rule_;
    bool coupled() const { return coupled_active_; }
    /// PROBABILISTIC DRAFT ACCEPTANCE (core/spec_prob.hpp, STRATA_SPEC_PROB=1): this request's drafts are SAMPLED from
    /// q and each draft j's distribution is in spec_q() + j * kSpecQStride (ids, -1 terminated; then float bits),
    /// valid on the host once draft() has returned.  False for greedy requests and where the mode is off.
    bool prob() const { return coupled_active_ && h_q_ != nullptr; }
    const int32_t* spec_q() const { return h_q_; }

    // ---- --pipeline-windows 2: the chain as one asynchronous launch (the round and its steps back to back, no host
    // wait), its first `n_force` steps fed the given tokens instead of their own picks (teacher forcing: the drafts of
    // a window already in flight), each output read once the event after its step has completed.
    /// Before `load`: the round/step graphs get the forcing kernel (a no-op while its token is -1), and the drafter's
    /// stream the highest priority (STRATA_MTP_PRIORITY=0: the default priority).
    void set_force_capture(bool on) { force_on_ = on; }
    /// The rows the next round reads (null: the bound `window_R`), copied into the bound buffer before the round.
    void set_source_R(const float* rows) { src_R_ = rows; }
    /// Capture the round graphs for T = 1..max_t and every step graph now (a capture syncs the drafter's stream).
    bool prepare_chain(std::string& err);
    /// The catch-up over T cells from `p` (as draft()), then `n_out` outputs from row `a`: output j is the chain's
    /// own pick, and step j (1 <= j <= n_force) takes force[j - 1] as its input token instead of output j - 1.
    /// `n_early`: an event after each of the first `n_early` outputs (chain_outputs_ready).
    bool chain_launch(int T, const int32_t* tokens, int64_t p, int a, const int32_t* force, int n_force, int n_out,
                      int n_early, std::string& err);
    /// 1: every output is in chain_tok()/chain_prob(); 0: still running; -1: an error.
    int chain_poll(std::string& err);
    /// How many of the chain's first outputs (up to `n_early`; all of them once the chain is done) have landed in
    /// chain_tok()/chain_prob(); -1: an error.  A window's size is often decided by its first outputs (a draft below
    /// spec_min_p ends it), so the next launch need not wait for all of them.
    int chain_outputs_ready(std::string& err);
    bool chain_live() const { return chain_live_; }
    const int32_t* chain_tok() const { return chain_tok_; }
    const float* chain_prob() const { return chain_prob_; }

    double ms_draft = 0, ms_prefill = 0;
    double ms_src = 0;      ///< the host time spent staging the source rows (a cross-card copy when the drafter is on another card)
    int64_t n_src = 0;
    int64_t rounds = 0;

    /// E-9: the prompt path computes this layer's prompt K/V in batches (Prefill::draft_kv): its tensors, its K/V
    /// state, the first cell a round can still read, its device, and a wait for its own stream.
    const float* tensor_f32(const char* name) const { return f32(name); }
    const uint16_t* tensor_bf16(const char* name) const { return bf16(name); }
    const void* tensor_q8(const char* name) const { return q8(name); }
    QsaState& kv_state_rw() { return st_; }
    /// The first cell whose K/V the draft layer can still read: the window's start for this prompt, and never below
    /// the KVA floor (cells built from approximated rows are not read, so they are not built either).
    int64_t first_needed() const {
        const int64_t w = (window_ > 0 && prompt_len_ > 0) ? prompt_len_ - window_ - 64 : 0;
        return std::max<int64_t>(w, cell_floor_);
    }
    /// The cells the draft layer attends to (0 = every cell), as load() settled it.
    int64_t window() const { return window_; }
    /// KVA: call before load().  The draft layer's attention then reads its cell range through a floor, which stays 0
    /// (the plain window, cell for cell) until set_cell_floor() raises it.  Off (the default): the code is 0.2.6's.
    void enable_cell_floor(bool on) { floor_on_ = on; }
    bool cell_floor_enabled() const { return floor_on_ && m_floor_ != nullptr; }
    /// KVA: the first draft cell that holds a K/V built from a real (not approximated) row.  The attention starts no
    /// lower; draft_kv / prefill build no cell below it.  Between requests (the drafter idle).  0 = every cell is real.
    void set_cell_floor(int64_t cell) {
        cell_floor_ = std::max<int64_t>(0, cell);
        if (m_floor_ != nullptr) *(volatile int32_t*) h_floor_ = (int32_t) std::min<int64_t>(cell_floor_, 0x7fffffff);
    }
    int64_t cell_floor() const { return cell_floor_; }
    int device() const { return device_; }
    bool idle(std::string& err) {
        if (cs_ && cudaStreamSynchronize(cs_) != cudaSuccess) { err = "mtp: its stream failed"; return false; }
        return true;
    }

private:
    bool record_forward(int T, int step_row0, cudaStream_t cs, std::string& err);
    /// False (with `err`) while the experts are lent to a prompt: the prompt path's ring is pointed at their bytes
    /// (STRATA_MTP_ROOM_MIN), so a decode pass would read the prompt's staged blobs as experts.  Every decode entry
    /// checks this before it launches anything.
    bool decode_ready(std::string& err) const {
        if (experts_lent_) { err = "mtp: the 512 experts are lent to a prompt (STRATA_MTP_ROOM_MIN)"; return false; }
        return true;
    }
    /// The layer's front for T rows at step rows [row0, +T): the embedding, the fc projections, the attention
    /// hyper-connection read (R_, inj_, mixed_) and the K/V appended.
    bool record_front(int T, int row0, cudaStream_t cs, std::string& err);
    /// The rest for row 0 at step row `step_row`: the attention from the q projection, the MLP, the final mixer,
    /// the head, the draft and its probability.  Needs row 0's R_, inj_ and mixed_ (its q8_1 rows in xq_).
    bool record_rest(int step_row, cudaStream_t cs, std::string& err);
    void norm_rope(float* data, const float* gamma, int rows, int cols, const int32_t* p, cudaStream_t cs);
    bool capture_prefill(int T, std::string& err);
    bool capture_prefill_dev(int T, std::string& err);   ///< E-4: without the mapped staging (inputs copied on device)
    bool capture_round(int T, bool coupled, std::string& err);
    bool capture_step(int j, bool coupled, std::string& err);
    cudaGraphExec_t step_exec_[9] = {};
    /// LEADS.md B28 (STRATA_MTP_ONE_GRAPH=1): the round and steps 1..steps-1 in one graph, keyed by (coupled, T, steps)
    bool capture_chain(int T, int steps, bool coupled, std::string& err);
    std::map<int, cudaGraphExec_t> chain_exec_;
    // coupled draft sampling: its own round/step graphs (the argmax ones stay as they were), the request's
    // parameters and the penalty ring (mapped staging + device copies), the split scratch, token id -> subset index
    bool setup_coupled(std::string& err);
    cudaGraphExec_t round_exec_c_[9] = {};
    cudaGraphExec_t step_exec_c_[9] = {};
    bool coupled_ok_ = false, coupled_active_ = false;
    bool coupled_rec_ = false;   ///< record_forward: the full layer ends in the coupled sampler (draft coupled_j_)
    int coupled_j_ = 0;
    strata::kernels::SamplerParams *h_cparams_ = nullptr, *m_cparams_ = nullptr, *cparams_ = nullptr;
    int32_t *h_chist_ = nullptr, *m_chist_ = nullptr, *cring_ = nullptr, *dinv_ = nullptr;
    int32_t *h_q_ = nullptr, *m_q_ = nullptr;   ///< STRATA_SPEC_PROB: the drafts' q lists (mapped, max_t_ rows)
    void* cscratch_ = nullptr;
    const float* f32(const char* name) const;
    const uint16_t* bf16(const char* name) const;
    const void* q8(const char* name) const;

    const ModelGeometry* g_ = nullptr;
    SessionState* ss_ = nullptr;
    SessionState* ple_ss_ = nullptr;
    const WeightTable* wt_ = nullptr;
    const NativeHead* head_ = nullptr;
    const float* window_R_ = nullptr;
    int max_t_ = 0;
    int device_ = -1;   ///< the device `load` ran on: the public calls switch to it (layer split)
    int max_drafts_ = 1 << 30;
    bool hnorm_stream_ = false;
    bool q4_ = false, q4_head_ = false;
    uint8_t* dense4_ = nullptr;              ///< --mtp-q4: Q4_0 copies of the Q8_0 tensors
    std::vector<std::pair<std::string, uint64_t>> q4_off_;
    uint8_t* hcq8_ = nullptr;                ///< STRATA_MTP_HC_Q8=1: Q8_0 copies of the draft layer's hc down/up rows
    std::vector<std::pair<std::string, uint64_t>> hcq8_off_;   ///< name -> offset in hcq8_
    const uint8_t* hcq8(const char* name) const;
    bool make_hcq8(std::string& err);
    int dhead_type_ = -1;                   ///< the draft head subset's ggml type (the main head's, or Q4_0)
    std::string dvocab_path_;
    std::string vocab_file() const { return dvocab_path_.empty() ? rt_dir_ + "/draft_vocab.bin" : dvocab_path_; }
    /// a projection's weights for the draft layer's own pass: the Q4_0 copy under --mtp-q4, else the Q8_0 original
    const void* wq(const char* name, int& type) const;
    bool make_q4_dense(const std::vector<uint8_t>& blob, std::string& err);
    bool make_q4_head(std::string& err);
    void record_top2(int j);
    std::vector<int32_t> top2_, dvocab_host_;
    std::vector<float> lg_host_;
    // --pipeline-windows 2 (chain_launch)
    bool stage_source_R(int T, std::string& err);   ///< set_source_R's rows into the bound buffer
    // the round between draft_launch and draft_finish (STRATA_DRAFT_EARLY; draft() runs the two back to back)
    struct RunState {
        bool live = false, cp = false, do_ple = false, eager_path = false;
        int T = 0, max_steps = 0;
        float min_p = 0.0f;
        std::chrono::steady_clock::time_point t0;
        double ms_launch = 0.0;          ///< the launch half's host time (ms_draft counts both halves, not what lies between)
        int32_t ple_prev[2] = {0, 0};    ///< the PLE prefetch's history along the chain
    } run_;
    struct IvState {   ///< STRATA_MTP_IN_VERIFY: the round armed for / queued behind the current window
        bool armed = false, queued = false;
        int T = 0, max_steps = 0, key = 0;
        int64_t p = 0;
        float min_p = 0.0f;
    } iv_;
    cudaEvent_t iv_ev_ = nullptr;   ///< STRATA_MTP_IN_VERIFY: orders the drafter's own stream around the round
    void ple_prefetch(int32_t tok);      ///< a PLE page prefetch hint for one token of the round's chain (run_.ple_prev)
    // STRATA_DRAFT_SUFFIX_SKIP: the predicate, and the chain a draft() ended after draft 0 (draft_resume)
    SkipAfterFirstFn skip_fn_ = nullptr;
    void* skip_user_ = nullptr;
    struct ResumeState {
        bool valid = false, cp = false, do_ple = false;
        int max_steps = 0;
        int32_t tok0 = 0;
        float p0 = 0.0f, min_p = 0.0f;
        int32_t ple_prev[2] = {0, 0};
    } resume_;
    bool force_on_ = false, chain_live_ = false;
    const float* src_R_ = nullptr;
    int32_t *h_force_ = nullptr, *m_force_ = nullptr;   ///< the forced tokens (mapped), -1 = the step's own pick
    cudaEvent_t ev_chain_ = nullptr;
    cudaEvent_t ev_step_[8] = {};                       ///< after each of the first n_early outputs
    int steps_seen_ = 0, chain_n_ = 0, chain_early_ = 0;
    int32_t chain_tok_[8] = {};
    float chain_prob_[8] = {};
    int64_t n_vocab_ = 0;
    uint64_t vram_ = 0;
    cudaStream_t cs_ = nullptr;
    cudaGraphExec_t prefill_exec_[9] = {};
    cudaGraphExec_t prefill_dev_exec_[9] = {};
    PinnedRing pf_ring_;          ///< #1996: the records above go up from a pinned slot, not from the pageable vector they are built in
    int32_t* pf_dev_ = nullptr;   ///< E-4: a prompt's rows' token / step / position records, uploaded at once
    int64_t pf_cap_ = 0;          ///< its capacity in ints
    cudaStream_t side_ = nullptr;                    // the shared expert's branch of the draft graphs (CUDA)
    cudaEvent_t sh_fork_ = nullptr, sh_join_ = nullptr;
    cudaGraphExec_t round_exec_[9] = {};

    struct Tensor { std::string name, kind; int64_t rows = 0, cols = 0; uint64_t off = 0, bytes = 0; };
    std::vector<Tensor> tensors_;
    uint8_t* dense_ = nullptr;
    uint8_t* experts_ = nullptr;
    void* experts_host_map_ = nullptr;   // STRATA_MTP_HEAD_HOST=1 (LEADS.md Part S row S13): the experts' mapped host block; experts_ is its device alias
    // the prompt room: the experts' pinned host snapshot, its size, and whether a prompt has taken their contents
    // (the block itself stays where it is - see experts_lend)
    void* experts_host_ = nullptr;
    uint64_t experts_host_bytes_ = 0;
    bool experts_lent_ = false;
    // Slot drafters borrow immutable weights and head; each still owns its state and scratch.
    bool owns_weights_ = true;
    bool owns_draft_head_ = true;
    bool dhead_plain_ = false;               ///< the subset is the main head's rows as gathered (no Q4_0 copy, no repack)
    void* state_arena_ = nullptr;
    QsaState st_;
    void* arena_ = nullptr;

    // mapped staging: tokens, step records (2*max_t rows), positions per head (2*max_t rows), the selected row,
    // the drafts out
    int32_t *h_tok_ = nullptr, *m_tok_ = nullptr, *h_step_ = nullptr, *m_step_ = nullptr;
    int32_t *h_pos_ = nullptr, *m_pos_ = nullptr, *h_row_ = nullptr, *m_row_ = nullptr;
    int32_t *h_out_ = nullptr, *m_out_ = nullptr;
    float *h_prob_ = nullptr, *m_prob_ = nullptr;   // each draft's probability under the draft layer
    // STRATA_DRAFT_ENTROPY (LEADS.md section 3): each draft's entropy / runner-up probability, mapped like h_prob_ (null: off),
    // their device rows (row_entropy), the last round's host copies, and the STRATA_SPEC_ENTROPY stop rule
    float *h_ent_ = nullptr, *m_ent_ = nullptr, *h_top2_ = nullptr, *m_top2_ = nullptr;
    float *ents_ = nullptr, *top2s_ = nullptr;
    float round_ent_[16] = {}, round_top2_[16] = {};   // (NaN-filled when the buffers are made)
    strata::spec::EntropyStop ent_stop_;
    // LEADS.md B24 (STRATA_DISCO_ENTROPY, opt-in): each draft's softmax entropy (mapped; -1 = unknown) and the rows' (device)
    float *h_dent_ = nullptr, *m_dent_ = nullptr;
    float* dent_ = nullptr;
    // the draft head: the main head's rows for a token subset (rt/draft_vocab.bin), or the whole head
    uint8_t* dhead_ = nullptr;
    int32_t* dvocab_ = nullptr;
    // pancake-strata: basalt's 2-bit proxy of a Q5_K draft head (q5k_proxy_*): the greedy one-row draft step's pick
    // in two stages (every row scored from the proxy, the near-best rescored exactly); STRATA_MTP_PROXY=0 turns it off
    void* proxy_ = nullptr;
    void* proxy_scratch_ = nullptr;
    int64_t n_dvocab_ = 0;
    std::string rt_dir_;
    int64_t window_ = 0;        // attention over the last window_ cells (0 = every cell)
    int64_t cell_floor_ = 0;    // KVA: no draft cell below this one is read or built (set_cell_floor)
    bool floor_on_ = false;     // enable_cell_floor: the attention goes through window_ids_floor
    int32_t *h_floor_ = nullptr, *m_floor_ = nullptr;   // the floor, mapped: the kernel reads it when it runs
    int64_t prompt_len_ = 0;
    const int32_t* rr_res_ = nullptr;
    float rr_margin_ = 0.0f;
    int rr_lo_ = 6, rr_hi_ = 9;
    float* probs_ = nullptr;
    uint8_t* arg_scratch_ = nullptr;   ///< argmax_rows' and row_top_prob_split's partials and counters
    uint8_t* top_scratch_ = nullptr;
    // device
    int32_t *tok_ = nullptr, *step_ = nullptr, *pos_ = nullptr, *row_ = nullptr, *ident_ = nullptr;
    float *Rin_ = nullptr, *R_ = nullptr, *emb_ = nullptr, *en_ = nullptr, *e2_ = nullptr, *hn_ = nullptr, *h2_ = nullptr;
    float *mixed_ = nullptr, *inj_ = nullptr, *inj2_ = nullptr, *lo_ = nullptr, *rs_ = nullptr, *bo_ = nullptr;
    float* xn_ = nullptr;
    /// gr_read's scratch of the drafter's own (not the session's ss.block.gr: the last stage's verifier uses that one,
    /// and with --pipeline-windows a chain and that stage's window can run on the card at once)
    strata::kernels::GrWorkspace own_gr_;
    uint8_t* xq_ = nullptr;
    float *qfull_ = nullptr, *qcur_ = nullptr, *kcur_ = nullptr, *vcur_ = nullptr, *attn_ = nullptr, *attn32_ = nullptr;
    float* attn_scratch_ = nullptr;
    float *logits_ = nullptr, *w_ = nullptr, *shared_ = nullptr, *parts_ = nullptr, *y_ = nullptr, *sample_ = nullptr;
    int32_t *ids_ = nullptr, *hit_slot_ = nullptr, *hit_dst_ = nullptr, *hit_count_ = nullptr, *out_ids_ = nullptr;
    uint8_t* hit_xq_ = nullptr;
    unsigned long long* grp_ptr_ = nullptr;
    int32_t *grp_start_ = nullptr, *grp_counts_ = nullptr;
    float* hit_xs_ = nullptr;
    void* hit_scratch_ = nullptr;
    float* sh_scratch_ = nullptr;
    uint16_t* x_bf16_ = nullptr;
    float* head_logits_ = nullptr;
    float* dummy_inj_ = nullptr;
    int64_t cap_ = 0, attn_scratch_floats_ = 0;
};

}  // namespace strata::core
