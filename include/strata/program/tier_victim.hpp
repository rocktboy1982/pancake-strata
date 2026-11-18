#pragma once
// LEADS.md B145 / B146 (owner idea 2026-11-18): two opt-in rules for how the serve tier's adapt() replaces a resident
// expert.  Both are placement only (which expert sits in VRAM and how its slot is rewritten; the math does not change).
// Env read once; unset = nothing changes (the default path is byte-identical).
//
//   STRATA_ADAPT_VICTIM_UNUSED=1  the victim must be an expert with ZERO use in the last K tier rounds (K =
//       STRATA_ADAPT_VICTIM_K, 1..64, default 4).  "Used in a round" is the TierStat test: the decayed count rose over the
//       last round's value x decay, i.e. the expert was routed (or counted by the prompt fold) since the last round.  K = 1
//       means "not used since the previous round", so an expert used in the previous round is never a victim.  Among the
//       unused ones the usual rule applies (lowest score first).  If a layer has fewer unused residents than candidates:
//       STRATA_ADAPT_VICTIM_FALLBACK=rule (default) takes the remaining victims from the used ones in the current score
//       order (the baseline rule, after the unused ones); =skip swaps only as many experts as there are unused victims.
//   STRATA_ADAPT_DIRECT_SWAP=1  the swap rewrites the victim's slot in place with ONE host-to-device copy and no
//       other step on the victim: the baseline already copies straight over the slot (no zeroing, no unmapping; generate.cpp
//       adapt() adapt_copy_h2d into device_slot(slot)), so what this adds is that the victim's host pages are not read back
//       ahead (ExpertSource::prefetch on the evicted expert: a madvise WILLNEED on a file-backed arena), because a victim
//       that was not used is not needed by the CPU soon, and a later need page-faults it in (lossless).
//
// Pure logic on the tier's own arrays (usage counts) so a ctest covers it.  Cost: observe() is one pass over layers x
// experts floats per round (about 25 us at 48 x 512); arrange() sorts nothing (a stable partition or a flag lookup).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace strata::program {

class TierVictim {
public:
    struct Params {
        int unused_k = 0;       ///< STRATA_ADAPT_VICTIM_UNUSED with STRATA_ADAPT_VICTIM_K: rounds of zero use, 0 off
        bool skip = false;      ///< STRATA_ADAPT_VICTIM_FALLBACK=skip (default rule)
        bool direct = false;    ///< STRATA_ADAPT_DIRECT_SWAP
    };

    static Params params_from_env() {
        Params p;
        auto flag = [](const char* name) {
            const char* v = std::getenv(name);
            return v != nullptr && v[0] != '\0' && std::atoi(v) != 0;
        };
        p.direct = flag("STRATA_ADAPT_DIRECT_SWAP");
        if (flag("STRATA_ADAPT_VICTIM_UNUSED")) {
            p.unused_k = 4;
            if (const char* v = std::getenv("STRATA_ADAPT_VICTIM_K"); v != nullptr && v[0] != '\0') {
                char* end = nullptr;
                const long k = std::strtol(v, &end, 10);
                if (end == v || *end != '\0' || k < 1 || k > 64)
                    std::fprintf(stderr, "strata serve: STRATA_ADAPT_VICTIM_K=%s is not an integer in 1..64: using 4\n", v);
                else
                    p.unused_k = (int) k;
            }
            if (const char* v = std::getenv("STRATA_ADAPT_VICTIM_FALLBACK"); v != nullptr && v[0] != '\0') {
                if (std::strcmp(v, "skip") == 0) p.skip = true;
                else if (std::strcmp(v, "rule") != 0)
                    std::fprintf(stderr, "strata serve: STRATA_ADAPT_VICTIM_FALLBACK=%s is not rule or skip: using rule\n", v);
            }
        }
        return p;
    }

    static TierVictim from_env(int64_t n_layers, int64_t n_expert) {
        TierVictim t;
        t.configure(params_from_env(), n_layers, n_expert);
        return t;
    }

    void configure(const Params& p, int64_t n_layers, int64_t n_expert) {
        p_ = p;
        L_ = n_layers;
        E_ = n_expert;
        if (unused_on())
            std::fprintf(stderr, "strata serve: STRATA_ADAPT_VICTIM_UNUSED=1: a victim must have had no use in the last %d tier "
                                 "rounds; with too few unused residents: %s\n",
                         p_.unused_k, p_.skip ? "the swap count shrinks (skip)" : "the baseline victim rule fills the rest (rule)");
        if (direct_on())
            std::fprintf(stderr, "strata serve: STRATA_ADAPT_DIRECT_SWAP=1: the victim's slot is rewritten in place by one copy; "
                                 "the victim's host pages are not read back ahead\n");
    }

    bool unused_on() const { return p_.unused_k > 0; }
    bool direct_on() const { return p_.direct; }
    bool on() const { return unused_on() || direct_on(); }
    int k() const { return p_.unused_k; }
    bool skip_mode() const { return p_.skip; }
    uint64_t rounds() const { return round_; }
    uint64_t picks() const { return picks_; }
    uint64_t recent_picks() const { return recent_picks_; }
    uint64_t skipped() const { return skipped_; }
    uint64_t direct_swaps() const { return direct_swaps_; }
    double mean_us() const { return round_ > 0 ? (double) ns_ / 1000.0 / (double) round_ : 0.0; }

    /// Round start: `raw` = the tier's counts before any transform.  An expert whose count rose over the last value x decay
    /// was used since the last round.
    void observe(const std::vector<float>& raw, float decay) {
        if (!unused_on() || (int64_t) raw.size() != L_ * E_) return;
        const auto t0 = std::chrono::steady_clock::now();
        const size_t n = raw.size();
        if (last_raw_.empty()) {
            last_raw_.assign(n, 0.0f);
            last_used_.assign(n, kNever);
        }
        ++round_;
        const int32_t now = (int32_t) round_;
        for (size_t i = 0; i < n; ++i) {
            if (raw[i] > last_raw_[i] * decay + 1e-6f) last_used_[i] = now;
            last_raw_[i] = raw[i];
        }
        ns_ += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    }

    /// True when (layer, expert) was used within the last K rounds (the latest round counts as the first).  Before the first
    /// observe() nothing is recent (no history: the baseline rule).
    bool recent(int64_t layer, int32_t expert) const {
        if (last_used_.empty()) return false;
        return (int64_t) round_ - (int64_t) last_used_[(size_t) (layer * E_ + expert)] < (int64_t) p_.unused_k;
    }

    /// Prepares one layer's victims (score, expert) for the swap loop: skip mode removes every recently used one (and counts the
    /// candidates it leaves without a victim, `n_cand` = the layer's candidates); rule mode keeps the list (the caller orders
    /// the unused ones first with `before`).  Off: no change.
    void arrange(int64_t layer, std::vector<std::pair<float, int32_t>>& vict, size_t n_cand) {
        if (!unused_on() || !p_.skip) return;
        const size_t before = vict.size();
        vict.erase(std::remove_if(vict.begin(), vict.end(),
                                  [&](const std::pair<float, int32_t>& v) { return recent(layer, v.second); }), vict.end());
        const size_t a = std::min(n_cand, before), b = std::min(n_cand, vict.size());
        skipped_ += (uint64_t) (a - b);
    }

    /// The victim order: unused first, then the score (lowest first).  Off: the score alone (the baseline comparator).
    bool before(int64_t layer, const std::pair<float, int32_t>& a, const std::pair<float, int32_t>& b) const {
        if (unused_on()) {
            const bool ra = recent(layer, a.second), rb = recent(layer, b.second);
            if (ra != rb) return rb;
        }
        return a.first < b.first;
    }

    /// The victims in the order the tier's own selector gave them (KMiss: least recently routed first): unused ones first, each
    /// group keeping its order; skip mode drops the recent ones.
    void order_given(int64_t layer, std::vector<std::pair<float, int32_t>>& vict, size_t n_cand) {
        if (!unused_on()) return;
        if (p_.skip) { arrange(layer, vict, n_cand); return; }
        std::stable_partition(vict.begin(), vict.end(), [&](const std::pair<float, int32_t>& v) { return !recent(layer, v.second); });
    }

    /// One chosen swap (counts, for the log line): a victim that was used within K rounds is a fallback pick.
    void note_pick(int64_t layer, int32_t victim) {
        if (!unused_on()) return;
        ++picks_;
        if (recent(layer, victim)) ++recent_picks_;
    }
    /// One swap copied under STRATA_ADAPT_DIRECT_SWAP (its victim's readback skipped).
    void note_direct() { ++direct_swaps_; }

    void maybe_log() const {
        if (!unused_on() || round_ == 0) return;
        if (round_ != 1 && round_ != 64 && round_ != 256 && (round_ & 1023) != 0) return;
        std::fprintf(stderr, "strata serve: victim unused: round %llu, %.1f us per round, picks %llu, of them used within %d rounds "
                             "%llu, candidates left without a victim (skip) %llu, direct swaps %llu\n",
                     (unsigned long long) round_, mean_us(), (unsigned long long) picks_, p_.unused_k,
                     (unsigned long long) recent_picks_, (unsigned long long) skipped_, (unsigned long long) direct_swaps_);
    }

private:
    static constexpr int32_t kNever = -(1 << 30);
    Params p_;
    int64_t L_ = 0, E_ = 0;
    std::vector<float> last_raw_;
    std::vector<int32_t> last_used_;
    uint64_t round_ = 0, picks_ = 0, recent_picks_ = 0, skipped_ = 0, direct_swaps_ = 0, ns_ = 0;
};

}  // namespace strata::program
