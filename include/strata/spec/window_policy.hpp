#pragma once
// MOD_PLAN section 6 item 21 (R&D-C window policy controller), step 2: STRATA_WINDOW_POLICY=N[,margin[,cap]] (opt-in, lossy
// like ECAP, default off).  The causal rule of tools/pancake/window_policy.py (`causal_ks`), host-testable:
//   kept rows of a window = min(T, ceil(mean accepted drafts of the last N windows) + 1 + margin)   (a cold history keeps T)
// Rows 0..kept-1 are verified as the preset does; the draft rows kept..T-1 are the ones the last windows say will mostly be
// rejected, so their CPU experts are capped at `cap` per row (default 0) by the ECAP row rule (core/verify_ecap.hpp,
// verify_ecap_drop_rows): the list is -1 for window rows 1..kept-1 (uncapped), then `cap`, repeating.  The oracle model of
// the tool is saved(k) = the CPU-missed experts only rows >= k need, which is exactly what the list drops.
// The Verifier feeds `observe` the window's accepted draft count (the matching prefix of out[] against the draft tokens)
// after every window the ECAP gate opened for, and asks `rows_list` before the next one.  The item's gate (the tool's
// oracle headroom over the best fixed rule >= 5% on a real window trace) has NOT been evaluated: no real trace exists offline.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

namespace strata::spec {

struct WindowPolicyCfg {
    int n = 0;        ///< windows of history; 0 off, -1 a bad value (off)
    int margin = 0;   ///< extra kept rows beyond the predicted acceptance
    int cap = 0;      ///< ECAP cap of the draft rows past the kept ones (0..63)
};

/// "N[,margin[,cap]]": N in 1..64, margin 0..16, cap 0..63.  Unset or empty: n 0.  Anything else: n -1.
inline WindowPolicyCfg window_policy_parse(const char* v) {
    WindowPolicyCfg c;
    if (v == nullptr || v[0] == '\0') return c;
    long x[3] = {0, 0, 0};
    const char* p = v;
    int got = 0;
    for (;;) {
        char* end = nullptr;
        x[got] = std::strtol(p, &end, 10);
        if (end == p) { c.n = -1; return c; }
        ++got;
        if (*end == '\0') break;
        if (*end != ',' || got >= 3) { c.n = -1; return c; }
        p = end + 1;
    }
    if (x[0] < 1 || x[0] > 64 || x[1] < 0 || x[1] > 16 || x[2] < 0 || x[2] > 63) { c.n = -1; return c; }
    c.n = (int) x[0];
    c.margin = (int) x[1];
    c.cap = (int) x[2];
    return c;
}

class WindowPolicy {
public:
    static constexpr int kMaxN = 64;
    WindowPolicy() = default;
    explicit WindowPolicy(const WindowPolicyCfg& c) : cfg_(c) {}
    bool on() const { return cfg_.n > 0; }
    const WindowPolicyCfg& cfg() const { return cfg_; }

    /// An outcome: the accepted draft count of a finished window.
    void observe(int accepted) {
        if (cfg_.n <= 0) return;
        hist_[head_] = accepted < 0 ? 0 : accepted;
        head_ = (head_ + 1) % kMaxN;
        if (count_ < kMaxN) ++count_;
    }
    /// Kept rows (the committed row included) for a window of T rows: T while the history is empty.
    int kept_rows(int T) const {
        if (cfg_.n <= 0 || count_ == 0) return T;
        const int m = std::min(cfg_.n, count_);
        double sum = 0;
        for (int i = 1; i <= m; ++i) sum += hist_[(head_ + kMaxN - i) % kMaxN];
        const double pred = sum / m;
        const int k = (int) std::ceil(pred - 1e-9) + 1 + cfg_.margin;
        return std::min(T, std::max(1, k));
    }
    /// The ECAP row list for this window (verify_ecap_drop_rows' `rows`): returns the number of values written to out[]
    /// (at most max): -1 for window rows 1..kept-1, then `cap`.  kept >= T (nothing to cap), or no room: the one value -1,
    /// which also overrides a scalar STRATA_VERIFY_ECAP for this window.
    int rows_list(int T, int8_t* out, int max) const {
        const int k = kept_rows(T);
        if (k >= T || k > max) {
            out[0] = -1;
            return 1;
        }
        for (int i = 0; i < k - 1; ++i) out[i] = -1;
        out[k - 1] = (int8_t) cfg_.cap;
        return k;
    }

private:
    WindowPolicyCfg cfg_;
    int hist_[kMaxN] = {};
    int head_ = 0;
    int count_ = 0;
};

}  // namespace strata::spec
