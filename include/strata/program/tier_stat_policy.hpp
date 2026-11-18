#pragma once
// MOD_PLAN.md section 6 (statistical expert cache), LEADS.md B143 / B144: two opt-in rules for the adaptive tier, both
// placement only (which expert sits in VRAM changes, the outputs do not).  Env read once; unset = nothing changes.
//
//   STRATA_ADAPT_FILL_FIRST=f  (0 < f < 1)  fill on demand.  Only the first ceil(f x slots) slots (the startup fill hands
//       them out in profile rank order, as STRATA_ADAPT_PROTECT reads them) start "claimed"; the other residents are
//       "empty" slots: placeholders the tier overwrites first.  While a layer has an empty slot the tier (1) admits a
//       missing expert on its first miss (count >= 1, not 2) into it with no margin, (2) admits no more experts than the
//       layer has empty slots (the rest wait), so no claimed resident is evicted until the layer's slots are full; then
//       the normal rule applies.  A placeholder that is routed, or an expert that arrived, is claimed from then on.  A
//       placeholder still serves hits until it is overwritten (a real empty slot would not): the engine keeps the bytes.
//   STRATA_ADAPT_TRANSITION=w  (0 < w <= 16)  conversation-aware eviction protection.  Per layer an expert-to-expert
//       transition matrix T[a][b] (decayed count of "a routed in tier round t, b routed in round t + 1") is built online
//       from the routing the tier already counts.  The state is the set routed in the last round: P(b used next | state)
//       = mean over a in the set of T[a][b] / N[a] (continuing the conversation), mixed with b's overall use rate
//       (conversation switched) by the estimated switch probability s.  A resident's keep score gets + w x P x 1 / (1 - decay),
//       so an expert that is likely to be needed next round is not the victim.  Switch detection: the overlap of the
//       routed sets of two rounds (Jaccard, mean over layers) below STRATA_ADAPT_TRANSITION_SWITCH x its running mean
//       (default 0.5; 0 turns the term off, the matrix then only sees continuing rounds anyway) marks the round a switch:
//       s rises, and the round's pair is not learned into the matrix.  boundary() marks the next round a switch at once
//       (a new request or session).  STRATA_ADAPT_TRANSITION_DECAY (default 0.95) is the matrix's decay per round.
//
// Pure logic on the tier's own arrays (usage counts, host_res) so a ctest covers it.  Cost: observe() is |prev set| x |set|
// scatter adds per layer, apply() |set| row adds of n_expert floats per layer; the matrix is layers x experts x experts
// floats (allocated at the first round only when the transition rule is on).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace strata::program {

class TierStat {
public:
    struct Params {
        double fill_f = 0.0;          ///< STRATA_ADAPT_FILL_FIRST: 0 off
        double trans_w = 0.0;         ///< STRATA_ADAPT_TRANSITION: 0 off
        double trans_decay = 0.95;    ///< STRATA_ADAPT_TRANSITION_DECAY
        double switch_theta = 0.5;    ///< STRATA_ADAPT_TRANSITION_SWITCH
    };

    static Params params_from_env() {
        Params p;
        auto num = [](const char* name, double lo, double hi, bool lo_open, double& out) {
            const char* v = std::getenv(name);
            if (v == nullptr || v[0] == '\0') return;
            char* end = nullptr;
            const double x = std::strtod(v, &end);
            if (end == v || !std::isfinite(x) || x > hi || x < lo || (lo_open && x == lo)) {
                std::fprintf(stderr, "strata serve: %s=%s is outside the allowed range: ignored\n", name, v);
                return;
            }
            out = x;
        };
        double f = 0.0, w = 0.0;
        num("STRATA_ADAPT_FILL_FIRST", 0.0, 1.0, true, f);
        num("STRATA_ADAPT_TRANSITION", 0.0, 16.0, true, w);
        p.fill_f = f < 1.0 ? f : 0.0;   // f = 1: no empty slot, a no-op
        p.trans_w = w;
        num("STRATA_ADAPT_TRANSITION_DECAY", 0.5, 0.999, false, p.trans_decay);
        num("STRATA_ADAPT_TRANSITION_SWITCH", 0.0, 1.0, false, p.switch_theta);
        return p;
    }

    /// `adapt_decay` = --adapt-decay; `slots` = the cache's slot count; `per_layer` = per-layer admission (the slot order
    /// is then not the profile rank order: the fill rule is off, as STRATA_ADAPT_PROTECT is).
    static TierStat from_env(float adapt_decay, int64_t n_layers, int64_t n_expert, int64_t slots, bool per_layer) {
        TierStat t;
        t.configure(params_from_env(), adapt_decay, n_layers, n_expert, slots, per_layer);
        return t;
    }

    void configure(const Params& p, float adapt_decay, int64_t n_layers, int64_t n_expert, int64_t slots, bool per_layer) {
        p_ = p;
        L_ = n_layers;
        E_ = n_expert;
        scale_ = adapt_decay < 0.999f ? 1.0f / (1.0f - adapt_decay) : 1000.0f;
        core_n_ = 0;
        if (p_.fill_f >= 1.0) p_.fill_f = 0.0;   // no empty slot: a no-op
        if (p_.fill_f > 0.0 && slots > 0) {
            if (per_layer) {
                std::fprintf(stderr, "strata serve: STRATA_ADAPT_FILL_FIRST is ignored with per-layer admission "
                                     "(--expert-cache-per-layer: the slot order is not the profile rank order)\n");
                p_.fill_f = 0.0;
            } else {
                core_n_ = (int64_t) std::ceil(p_.fill_f * (double) slots - 1e-9);
                core_n_ = std::clamp<int64_t>(core_n_, 1, slots);
            }
        }
        if (p_.trans_w > 0.0 && (double) L_ * (double) E_ * (double) E_ * 4.0 > 512.0 * 1024.0 * 1024.0) {
            std::fprintf(stderr, "strata serve: STRATA_ADAPT_TRANSITION: the matrix would exceed 512 MiB: off\n");
            p_.trans_w = 0.0;
        }
        if (fill_on())
            std::fprintf(stderr, "strata serve: STRATA_ADAPT_FILL_FIRST=%g: %lld of %lld slots start claimed, the rest fill on "
                                 "demand (first miss; no claimed resident is evicted while a layer has an empty slot)\n",
                         p_.fill_f, (long long) core_n_, (long long) slots);
        if (trans_on())
            std::fprintf(stderr, "strata serve: STRATA_ADAPT_TRANSITION=%g: residents likely to be needed next round are kept "
                                 "(matrix decay %g, switch threshold %g)\n", p_.trans_w, p_.trans_decay, p_.switch_theta);
    }

    bool fill_on() const { return core_n_ > 0; }
    bool trans_on() const { return p_.trans_w > 0.0; }
    bool on() const { return fill_on() || trans_on(); }
    /// A new request / session starts: the next round's prediction uses the overall use rate, not the matrix.
    void boundary() { boundary_ = true; }
    double switch_prob() const { return s_; }
    uint64_t rounds() const { return round_; }
    double mean_us() const { return round_ > 0 ? (double) ns_ / 1000.0 / (double) round_ : 0.0; }
    /// Empty (unclaimed resident) slots as of the last observe().
    int64_t empty_slots() const { return empty_; }

    /// Round start: `raw` = the tier's counts before any transform (an expert whose count rose over the decayed last
    /// value was routed this round), `res` = host_res.  Updates the claimed set, the switch estimate and the matrix.
    void observe(const std::vector<float>& raw, const int32_t* res, float decay) {
        if (!on() || (int64_t) raw.size() != L_ * E_) return;
        const auto t0 = std::chrono::steady_clock::now();
        const size_t n = raw.size();
        const bool first = last_raw_.empty();
        if (first) {
            last_raw_.assign(n, 0.0f);
            claimed_.assign(n, 0);
            was_res_.assign(n, 0);
            used_.assign(n, 0);
            prev_used_.assign(n, 0);
            if (trans_on()) {
                T_.assign(n * (size_t) E_, 0.0f);
                N_.assign(n, 0.0f);
                M_.assign(n, 0.0f);
                Rn_.assign((size_t) L_, 0.0f);
            }
        }
        ++round_;
        empty_ = 0;
        for (size_t i = 0; i < n; ++i) {
            const bool u = raw[i] > last_raw_[i] * decay + 1e-6f;
            used_[i] = u ? 1 : 0;
            last_raw_[i] = raw[i];
            const bool r = res[i] >= 0;
            if (first) claimed_[i] = r && (int64_t) res[i] < core_n_ ? 1 : 0;
            else if (!r) claimed_[i] = 0;
            else if (!was_res_[i] || u) claimed_[i] = 1;   // arrived, or routed: requested, so claimed
            was_res_[i] = r ? 1 : 0;
            if (r && !claimed_[i]) ++empty_;
        }
        if (trans_on()) learn();
        const auto t1 = std::chrono::steady_clock::now();
        ns_ += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        boundary_ = false;
    }

    /// Turns the counts `u` (any transform of the raw ones: it is changed in place) into the scores the tier ranks by.
    void apply(std::vector<float>& u, const int32_t* res) {
        if (!on() || last_raw_.empty() || (int64_t) u.size() != L_ * E_) return;
        const auto t0 = std::chrono::steady_clock::now();
        if (fill_on()) apply_fill(u, res);
        if (trans_on()) apply_trans(u, res);
        ns_ += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
    }

    /// One stderr line at rounds 1, 64, 256, 1024, ... (the overhead and the state, for the box runs).
    void maybe_log() const {
        if (!on() || round_ == 0) return;
        if (round_ != 1 && round_ != 64 && round_ != 256 && (round_ & 1023) != 0) return;
        std::fprintf(stderr, "strata serve: tier stat: round %llu, %.1f us per round (observe + apply), empty slots %lld, "
                             "switch estimate %.3f\n", (unsigned long long) round_, mean_us(), (long long) empty_, s_);
    }

private:
    void learn() {
        // overlap of this round's routed set with the last round's, per layer (Jaccard), mean over the layers that routed
        double ov = 0.0;
        int cnt = 0;
        for (int64_t l = 0; l < L_; ++l) {
            const size_t b = (size_t) (l * E_);
            int inter = 0, uni = 0;
            for (int64_t e = 0; e < E_; ++e) {
                const int a = prev_used_[b + (size_t) e], c = used_[b + (size_t) e];
                inter += a & c;
                uni += a | c;
            }
            if (uni > 0) { ov += (double) inter / (double) uni; ++cnt; }
        }
        ov = cnt > 0 ? ov / (double) cnt : 0.0;
        bool switched = boundary_;
        if (round_ > 1) {
            if (ov_mean_ <= 0.0) ov_mean_ = ov;   // the first pair of rounds
            else {
                if (p_.switch_theta > 0.0 && ov < p_.switch_theta * ov_mean_) switched = true;
                if (!switched) ov_mean_ = 0.8 * ov_mean_ + 0.2 * ov;   // the running mean follows continuing rounds only
            }
        }
        s_ = 0.7 * s_ + 0.3 * (switched ? 1.0 : 0.0);
        if (switched && boundary_) s_ = 1.0;
        switched_ = switched;
        g_ /= p_.trans_decay;   // growing weight = decaying history
        const float g = (float) g_;
        std::vector<int32_t> prev, cur;
        for (int64_t l = 0; l < L_; ++l) {
            const size_t b = (size_t) (l * E_);
            prev.clear();
            cur.clear();
            for (int64_t e = 0; e < E_; ++e) {
                if (prev_used_[b + (size_t) e]) prev.push_back((int32_t) e);
                if (used_[b + (size_t) e]) cur.push_back((int32_t) e);
            }
            if (!cur.empty()) {
                for (int32_t e : cur) M_[b + (size_t) e] += g;
                Rn_[(size_t) l] += g;
            }
            if (round_ > 1 && !switched && !cur.empty()) {
                for (int32_t a : prev) {
                    float* row = T_.data() + (b + (size_t) a) * (size_t) E_;
                    for (int32_t c : cur) row[c] += g;
                    N_[b + (size_t) a] += g;
                }
            }
        }
        if (g_ > 1.0e6) {
            const float k = (float) (1.0 / g_);
            for (float& v : T_) v *= k;
            for (float& v : N_) v *= k;
            for (float& v : M_) v *= k;
            for (float& v : Rn_) v *= k;
            g_ = 1.0;
        }
        prev_used_ = used_;
    }

    void apply_fill(std::vector<float>& u, const int32_t* res) {
        std::vector<std::pair<float, int32_t>> cand;
        for (int64_t l = 0; l < L_; ++l) {
            const size_t b = (size_t) (l * E_);
            int64_t free_n = 0;
            for (int64_t e = 0; e < E_; ++e)
                if (res[b + (size_t) e] >= 0 && !claimed_[b + (size_t) e]) ++free_n;
            if (free_n == 0) continue;   // the layer is full: the tier's normal rule
            cand.clear();
            for (int64_t e = 0; e < E_; ++e)
                if (res[b + (size_t) e] < 0 && u[b + (size_t) e] > 0.0f) cand.emplace_back(u[b + (size_t) e], (int32_t) e);
            std::stable_sort(cand.begin(), cand.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
            int64_t take = 0;
            for (const auto& c : cand) {
                float& v = u[b + (size_t) c.second];
                if (take < free_n && c.first >= 1.0f) { v = std::max(v + 1.0f, 2.0f); ++take; }   // first miss: eligible
                else v = 0.0f;   // waits until the layer's empty slots are used up
            }
            for (int64_t e = 0; e < E_; ++e)
                if (res[b + (size_t) e] >= 0 && !claimed_[b + (size_t) e]) u[b + (size_t) e] = -1.0f;   // the empty slots go first
        }
    }

    void apply_trans(std::vector<float>& u, const int32_t* res) {
        std::vector<float> acc((size_t) E_);
        std::vector<int32_t> set;
        const double s = std::clamp(s_, 0.0, 1.0);
        for (int64_t l = 0; l < L_; ++l) {
            const size_t b = (size_t) (l * E_);
            set.clear();
            for (int64_t e = 0; e < E_; ++e)
                if (prev_used_[b + (size_t) e]) set.push_back((int32_t) e);
            std::fill(acc.begin(), acc.end(), 0.0f);
            int have = 0;
            for (int32_t a : set) {
                const float nn = N_[b + (size_t) a];
                if (nn <= 1e-9f) continue;
                const float inv = 1.0f / nn;
                const float* row = T_.data() + (b + (size_t) a) * (size_t) E_;
                for (int64_t e = 0; e < E_; ++e) acc[(size_t) e] += row[e] * inv;
                ++have;
            }
            const float rn = Rn_[(size_t) l];
            if (have == 0 && rn <= 1e-9f) continue;
            const float ks = have > 0 ? (float) ((1.0 - s) / (double) have) : 0.0f;
            const float km = have > 0 ? (float) s / rn : 1.0f / rn;
            const float wsc = (float) p_.trans_w * scale_;
            for (int64_t e = 0; e < E_; ++e) {
                if (res[b + (size_t) e] < 0) continue;
                const float p = std::min(1.0f, acc[(size_t) e] * ks + M_[b + (size_t) e] * km);
                u[b + (size_t) e] += wsc * p;
            }
        }
    }

    Params p_;
    int64_t L_ = 0, E_ = 0, core_n_ = 0, empty_ = 0;
    float scale_ = 3.3333f;
    uint64_t round_ = 0, ns_ = 0;
    bool boundary_ = false, switched_ = false;
    double s_ = 0.1, ov_mean_ = 0.0, g_ = 1.0;
    std::vector<float> last_raw_, T_, N_, M_, Rn_;
    std::vector<uint8_t> claimed_, was_res_, used_, prev_used_;
};

}  // namespace strata::program
