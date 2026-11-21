#pragma once
// STRATA_PCOPY_REST_USAGE=1 (LEADS.md Part S row S7, default off): a usage-based resting rule for the predictive copy, the
// router look-ahead's next-layer prediction (its warm of the file / arena pages and its Foresight swap copies).
// radiance (Codeberg StillDeadcode/radiance, Apache-2.0) pauses its next-layer expert copies when under 30% of the last
// 2,048 were used, for 8,192 passes, and then resumes.  Ours never gated the copy by use: T-528 (next-layer prefetch
// -3..-4%) and T-806 (PREFETCH_PREV, no effect) ran with it always on.  This header is the rule, pure (no engine types),
// so a ctest covers it.
//
//   window  : predicted copies are counted in blocks of at least kPcopyRestWindow (2,048): `record(predicted, used)` adds
//             one layer's prediction (its not-yet-resident experts that were warmed) and how many of them the layer then
//             routed.  When the block holds >= 2,048 predictions the used share is judged: used * 100 < 30 * predicted
//             rests the copy.  (A tumbling block, not radiance's sliding one: the same threshold on the same count,
//             unconfirmed against their code.)
//   rest    : `pass()` is asked once per submit (one pass = one layer's predictive copy); while resting it returns false
//             for kPcopyRestPasses (8,192) passes, then true again with an empty block (no feedback arrives while resting).
//   off     : a default-constructed rule (the switch unset) never changes: pass() is true, record() does nothing, and
//             `touched()` stays 0 (the test uses it to show the default path never reaches the new logic).
#include <cstdint>
#include <cstdlib>

namespace strata::program {

constexpr int kPcopyRestWindow = 2048;     ///< predicted copies judged per block
constexpr int kPcopyRestMinPct = 30;       ///< a used share under this rests the copy
constexpr int kPcopyRestPasses = 8192;     ///< passes (submits) the copy rests

/// STRATA_PCOPY_REST_USAGE: only the text "1" switches it on (default off).
inline bool pcopy_rest_parse(const char* v) { return v != nullptr && v[0] == '1' && v[1] == '\0'; }

class PcopyRest {
public:
    PcopyRest() = default;
    explicit PcopyRest(bool on, int window = kPcopyRestWindow, int min_pct = kPcopyRestMinPct, int rest_passes = kPcopyRestPasses)
        : on_(on), window_(window < 1 ? 1 : window), min_pct_(min_pct), rest_passes_(rest_passes < 1 ? 1 : rest_passes) {}

    bool on() const { return on_; }
    bool resting() const { return resting_ > 0; }
    int64_t rests() const { return rests_; }          ///< rests started
    int64_t skipped() const { return skipped_; }      ///< passes refused while resting
    int64_t touched() const { return touched_; }      ///< calls that reached the rule's state (0 while off)
    int64_t last_predicted() const { return last_pred_; }   ///< the block that last started a rest
    int64_t last_used() const { return last_used_; }

    /// One pass (one submit): true = run the predictive copy.
    bool pass() {
        if (!on_) return true;
        ++touched_;
        if (resting_ <= 0) return true;
        --resting_;
        ++skipped_;
        pred_ = used_ = 0;   // the block that follows the rest starts empty
        return false;
    }

    /// One layer's prediction: `predicted` experts warmed, `used` of them routed by the layer.  True when this call started
    /// a rest.  Ignored while resting (a prediction in flight when the rest began carries no verdict).
    bool record(int predicted, int used) {
        if (!on_) return false;
        ++touched_;
        if (resting_ > 0 || predicted <= 0) return false;
        if (used < 0) used = 0;
        if (used > predicted) used = predicted;
        pred_ += predicted;
        used_ += used;
        if (pred_ < window_) return false;
        const bool rest = used_ * 100 < (int64_t) min_pct_ * pred_;
        last_pred_ = pred_;
        last_used_ = used_;
        pred_ = used_ = 0;
        if (!rest) return false;
        resting_ = rest_passes_;
        ++rests_;
        return true;
    }

private:
    bool on_ = false;
    int window_ = kPcopyRestWindow, min_pct_ = kPcopyRestMinPct, rest_passes_ = kPcopyRestPasses;
    int64_t pred_ = 0, used_ = 0, last_pred_ = 0, last_used_ = 0;
    int64_t resting_ = 0, rests_ = 0, skipped_ = 0, touched_ = 0;
};

}  // namespace strata::program
