#pragma once
// MOD_PLAN.md section 6 item 36 (LEADS.md Part I row I10), STRATA_KNOB_TRIAL=1 (default off, exact knobs only): the decision
// rule of BigMoeOnEdge's in-answer interleaved knob trial, in our form.  A lossless knob that can change with the session
// open (the per-request keys: pcie_frac, spec_min_p, aux_cpus, prefill_pipe) is timed against the incumbent inside the
// user's own first answer, in windows of 8 tokens, and the candidate is adopted only when it wins by a margin.
//
// The rule (theirs, with the choices the notes left open made here and pinned by knob_trial_test.cpp):
//   - SETTLE: incumbent windows run until two consecutive ones agree on the cache hit rate within 0.02 (the cache has settled);
//     a window with an unknown hit rate (< 0) settles after two incumbent windows.  The last settling window is the first
//     bracket I0.
//   - then I0 C1 I1 C2 I2 C3 I3 (rounds = 3): candidate window C_r is bracketed by the incumbent windows on both sides, and
//     round r's ratio = rate(C_r) / mean(rate(I_{r-1}), rate(I_r)) with rate = tokens / seconds.  A linear drift of the
//     machine (throttling, a cooling cache) cancels in the bracket; the bracket is shared between neighbouring rounds.
//   - a round LOST when its ratio is below 1; the candidate is dropped as soon as MORE THAN HALF of the rounds are lost
//     (2 of 3), or one candidate window ran apart from the incumbent band: below (1 - margin) x the slowest incumbent window
//     and below that window by more than the incumbent windows' own spread (the phone case: 8 threads ran 20x slower and
//     was cut after one window).  Dropped early, nothing more is spent on it.
//   - after the last round the MEDIAN of the round ratios decides: a WIN needs ratio >= 1 + margin (0.10), else LOSE.
//   - a window shorter than the window size (the answer ended) ends the trial UNDECIDED; nothing is adopted.
// Nothing here times anything or touches the engine: the caller feeds windows and applies the verdict (see MOD_PLAN.md item 36).
// NOT WIRED: generate.cpp has no per-window live-knob switch (the per-request keys are applied at a request's start by the
// serve loop) and the engine cannot be built here, so no STRATA_KNOB_TRIAL variable is read yet; only the rule and its log line exist.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace strata::program {

struct KnobTrialCfg {
    int window_tokens = 8;
    int rounds = 3;
    double margin = 0.10;        ///< a win needs ratio >= 1 + margin
    double settle_hit = 0.02;    ///< two incumbent windows agree on the hit rate within this
};

enum class TrialArm { Incumbent, Candidate };
enum class TrialVerdict { Running, Win, Lose, Undecided };

class KnobTrial {
public:
    explicit KnobTrial(KnobTrialCfg c = KnobTrialCfg()) : cfg_(c) {}

    /// Which arm the next window of the answer runs under.
    TrialArm next_arm() const {
        if (!settled_) return TrialArm::Incumbent;
        // after settling: windows alternate candidate, incumbent, ... starting with a candidate (I0 is the last settle window)
        return (bracket_.size() % 2 == 1) ? TrialArm::Candidate : TrialArm::Incumbent;
    }

    /// Feed one finished window.  `tokens` produced in `seconds`; `hit` the cache hit rate over it (< 0: unknown).
    void add(TrialArm arm, int tokens, double seconds, double hit = -1.0) {
        if (verdict_ != TrialVerdict::Running) return;
        ++windows_;
        if (tokens < cfg_.window_tokens || !(seconds > 0.0) || !std::isfinite(seconds)) { verdict_ = TrialVerdict::Undecided; reason_ = "the answer ended inside a window"; return; }
        const double rate = (double) tokens / seconds;
        if (!settled_) {
            if (arm != TrialArm::Incumbent) return;   // a caller that ignores next_arm(): not counted
            ++settle_windows_;
            const bool agree = last_hit_known_ ? (hit < 0.0 || std::fabs(hit - last_hit_) <= cfg_.settle_hit) : true;
            const bool second = settle_windows_ >= 2;
            last_hit_ = hit;
            last_hit_known_ = hit >= 0.0;
            incumbents_.push_back(rate);
            if (second && agree) { settled_ = true; bracket_.assign(1, rate); }   // I0
            return;
        }
        if (arm == TrialArm::Incumbent) {
            if (bracket_.size() % 2 == 1) return;   // out of turn: not counted
            incumbents_.push_back(rate);
            bracket_.push_back(rate);
            close_round();
        } else {
            if (bracket_.size() % 2 == 0) return;
            // ran apart from the incumbent band?
            const double mn = *std::min_element(incumbents_.begin(), incumbents_.end());
            const double mx = *std::max_element(incumbents_.begin(), incumbents_.end());
            if (rate < (1.0 - cfg_.margin) * mn && rate < mn - (mx - mn)) {
                cand_.push_back(rate);
                verdict_ = TrialVerdict::Lose;
                reason_ = "a candidate window ran apart from the incumbent band";
                return;
            }
            cand_.push_back(rate);
            bracket_.push_back(rate);
        }
    }

    TrialVerdict verdict() const { return verdict_; }
    const std::vector<double>& ratios() const { return ratios_; }
    int windows() const { return windows_; }
    int settle_windows() const { return settle_windows_; }
    const std::string& reason() const { return reason_; }

    double median_ratio() const {
        if (ratios_.empty()) return 0.0;
        std::vector<double> r = ratios_;
        std::sort(r.begin(), r.end());
        const size_t n = r.size();
        return n % 2 ? r[n / 2] : 0.5 * (r[n / 2 - 1] + r[n / 2]);
    }

    /// `strata knob trial: <knob> <from> -> <to>: WIN ratio 1.123 (rounds 1.100 1.130 1.120; 7 windows of 8 tokens, 2 to settle)`
    std::string line(const std::string& knob, const std::string& from, const std::string& to) const {
        const char* v = verdict_ == TrialVerdict::Win ? "WIN" : verdict_ == TrialVerdict::Lose ? "LOSE" : verdict_ == TrialVerdict::Undecided ? "UNDECIDED" : "RUNNING";
        std::string s = "strata knob trial: " + knob + " " + from + " -> " + to + ": " + v;
        char b[64];
        if (!ratios_.empty()) {
            std::snprintf(b, sizeof b, " ratio %.3f (rounds", median_ratio());
            s += b;
            for (double r : ratios_) { std::snprintf(b, sizeof b, " %.3f", r); s += b; }
            s += ")";
        }
        std::snprintf(b, sizeof b, "; %d windows of %d tokens, %d to settle", windows_, cfg_.window_tokens, settle_windows_);
        s += b;
        if (!reason_.empty()) s += "; " + reason_;
        return s;
    }

private:
    // bracket_ = I0 C1 I1 C2 I2 ...: odd length means a candidate is next
    void close_round() {
        const size_t n = bracket_.size();   // even, >= 3 after a closing incumbent: I0 C1 I1 -> n = 3
        const double c = bracket_[n - 2];
        const double base = 0.5 * (bracket_[n - 3] + bracket_[n - 1]);
        ratios_.push_back(base > 0.0 ? c / base : 0.0);
        int lost = 0;
        for (double r : ratios_) lost += r < 1.0 ? 1 : 0;
        if (2 * lost > cfg_.rounds) { verdict_ = TrialVerdict::Lose; reason_ = "more than half of the rounds lost"; return; }
        if ((int) ratios_.size() >= cfg_.rounds) {
            const double m = median_ratio();
            verdict_ = m >= 1.0 + cfg_.margin ? TrialVerdict::Win : TrialVerdict::Lose;
            if (verdict_ == TrialVerdict::Lose) reason_ = "median ratio below the margin";
        }
    }

    KnobTrialCfg cfg_;
    bool settled_ = false, last_hit_known_ = false;
    double last_hit_ = -1.0;
    int windows_ = 0, settle_windows_ = 0;
    std::vector<double> incumbents_, cand_, bracket_, ratios_;
    TrialVerdict verdict_ = TrialVerdict::Running;
    std::string reason_;
};

}  // namespace strata::program
