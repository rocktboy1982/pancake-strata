#pragma once
// STRATA_SWAP_GRACE=N (opt-in, default off; LEADS.md Part Q4, idea from Strata-HotRod SPEC 13.36, MIT; nothing copied): an expert the
// adaptive tier swapped IN may not be the victim of another swap for the next N tier rounds.  Without it a freshly landed expert,
// whose usage count is still low, is the cheapest resident and can be thrown out again by the next round (churn: two copies for
// nothing).  Placement only: which experts the GPU holds never changes a token.  The tier asks held(layer, expert) while it builds a
// layer's victim list and tells note_in(layer, expert) for every swap it issues; round() is called once per decode tier round (the
// STRATA_TIER_REWARM round is not a decode round).  Header-only, no GPU (tests/program/swap_grace_test.cpp).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::program {

class SwapGrace {
public:
    SwapGrace() = default;
    SwapGrace(int64_t n_layers, int64_t n_expert, int rounds) : ne_(n_expert), n_(rounds > 0 ? rounds : 0) {
        if (n_ > 0) in_.assign((size_t) (n_layers * n_expert), 0);
    }

    /// STRATA_SWAP_GRACE read once: the rounds, 0 (off) when unset, empty, 0, negative or not a number.
    static int rounds_from(const char* v) {
        if (v == nullptr || v[0] == '\0') return 0;
        const int n = std::atoi(v);
        return n > 0 ? n : 0;
    }
    static SwapGrace from_env(int64_t n_layers, int64_t n_expert) {
        const int n = rounds_from(std::getenv("STRATA_SWAP_GRACE"));
        SwapGrace g(n_layers, n_expert, n);
        if (g.on())
            std::fprintf(stderr, "strata serve: STRATA_SWAP_GRACE=%d: an expert swapped into the tier is no victim for %d rounds\n", n, n);
        return g;
    }

    bool on() const { return n_ > 0; }
    int rounds() const { return n_; }

    /// A decode tier round starts.
    void round() { if (n_ > 0) ++round_; }

    /// This expert came in during this round or one of the last N - 1 rounds before it (so N tier rounds after the swap it is free).
    bool held(int64_t layer, int64_t expert) const {
        if (n_ <= 0) return false;
        const size_t i = (size_t) (layer * ne_ + expert);
        if (i >= in_.size() || in_[i] == 0) return false;
        return round_ - in_[i] < (int64_t) n_;
    }

    /// A swap brought `expert` in.
    void note_in(int64_t layer, int64_t expert) {
        if (n_ <= 0) return;
        const size_t i = (size_t) (layer * ne_ + expert);
        if (i < in_.size()) in_[i] = round_ + 1;
    }

private:
    int64_t ne_ = 0;
    int n_ = 0;
    int64_t round_ = 0;
    std::vector<int64_t> in_;   ///< round + 1 of the expert's last swap-in (0 = never)
};

}  // namespace strata::program
