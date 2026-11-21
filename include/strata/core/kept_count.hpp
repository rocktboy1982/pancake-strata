#pragma once
// STRATA_COUNT_KEPT=1 (opt-in, default off; LEADS.md Part Q4, idea from Strata-HotRod SPEC 13.36 / 13.40, MIT; nothing copied): the
// adaptive tier's usage counts (ExpertDispatch::usage, which expert_pool_dispatch_multi bumps for EVERY row of a verify window as the
// window is planned) keep only the rows the window KEPT.  A window of T rows commits a + 1 of them (the committed row and the a
// accepted drafts); the experts the other T - a - 1 rejected draft rows routed to are experts the stream never used, yet the tier
// counted them and may swap them in.  Per layer the pool records the window's routing (record), and after the window the serial decode
// loop takes the rejected rows' share back out (drop_rejected), with exactly the amount that was added (1 per entry, or the
// STRATA_ADAPT_PROB_MASS weight x k).  Placement only: the counts steer which experts the GPU holds, never a token.  Header-only, no
// GPU (tests/core/kept_count_test.cpp).  Used only by the serial window loop (one window at a time, no --adapt-async, no --batch):
// the other loops never call drop_rejected, so for them the pointer is not set and nothing is recorded.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::core {

class KeptCount {
public:
    /// STRATA_COUNT_KEPT: unset, empty or 0 is off.
    static bool on_from(const char* v) { return v != nullptr && v[0] != '\0' && std::atoi(v) != 0; }
    static bool on_env() { return on_from(std::getenv("STRATA_COUNT_KEPT")); }

    KeptCount() = default;
    KeptCount(int64_t n_layers, int64_t n_expert) : nl_(n_layers), ne_(n_expert) {
        rec_.resize((size_t) n_layers);
    }

    /// One layer of one window: `ids` = n_tok x k router picks (row t = entries t * k .. t * k + k - 1); `w` = the per-entry
    /// STRATA_ADAPT_PROB_MASS weights or nullptr (then every entry added 1.0).  Replaces the layer's last record.
    void record(int64_t layer, const int32_t* ids, const float* w, int64_t n_tok, int64_t k) {
        if (layer < 0 || layer >= nl_ || ids == nullptr || n_tok < 1 || k < 1) return;
        Rec& r = rec_[(size_t) layer];
        const size_t n = (size_t) (n_tok * k);
        r.ids.assign(ids, ids + n);
        r.add.resize(n);
        for (size_t i = 0; i < n; ++i) {
            if (w == nullptr) r.add[i] = 1.0f;
            else r.add[i] = (w[i] == w[i] && w[i] > 0.0f) ? w[i] * (float) k : 0.0f;   // the pool's own weighting
        }
        r.n_tok = n_tok;
        r.k = k;
    }

    /// After a window of `n_tok` rows of which the first `kept_rows` were committed: subtract the rejected rows' share from `usage`
    /// (n_layers x n_expert), clamped at 0, and consume the records.  A layer whose record is missing or from a window of another
    /// size is left alone.  Returns the mass taken out.
    double drop_rejected(std::vector<float>& usage, int64_t n_tok, int64_t kept_rows) {
        double out = 0.0;
        if (usage.size() != (size_t) (nl_ * ne_)) return 0.0;
        for (int64_t l = 0; l < nl_; ++l) {
            Rec& r = rec_[(size_t) l];
            if (r.n_tok == 0) continue;
            const bool match = r.n_tok == n_tok;
            const int64_t n_tok_rec = r.n_tok, k = r.k;
            r.n_tok = 0;   // consumed either way: a stale record must not meet a later window
            if (!match || kept_rows >= n_tok_rec || kept_rows < 1) continue;
            float* u = usage.data() + (size_t) (l * ne_);
            for (int64_t i = kept_rows * k; i < n_tok_rec * k; ++i) {
                const int32_t e = r.ids[(size_t) i];
                if (e < 0 || e >= ne_) continue;
                const float a = r.add[(size_t) i];
                const float before = u[e];
                u[e] = before > a ? before - a : 0.0f;
                out += (double) (before - u[e]);
            }
        }
        total_ += out;
        ++windows_;
        return out;
    }

    int64_t windows() const { return windows_; }
    double mass_dropped() const { return total_; }

private:
    struct Rec {
        std::vector<int32_t> ids;
        std::vector<float> add;
        int64_t n_tok = 0, k = 0;
    };
    int64_t nl_ = 0, ne_ = 0;
    std::vector<Rec> rec_;
    int64_t windows_ = 0;
    double total_ = 0.0;
};

}  // namespace strata::core
