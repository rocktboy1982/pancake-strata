#pragma once
// MOD_PLAN phase 11, "Left, ranked" items 3 and 4: seed the adaptive expert tier at a request's start, so it pulls the
// request's hot experts into VRAM in the first windows instead of learning them from misses.  Placement only: which
// experts the GPU or the CPU computes never changes a token (both compute the same expert exactly), so the output is the
// same with or without it.  Both parts are opt-in and off by default; header-only, no GPU (tests/program/expert_seed_test.cpp).
//
//   STRATA_DOMAIN_PROFILES=<dir>  (item 4) per-domain profiles <dir>/<name>.bin (the --expert-profile format; written by
//                                 tools/pancake/learn_domain_profiles.py).  A request names its domain with the GEN key
//                                 domain=<name> (serve/server.py's classify_domain sends it when the same variable is
//                                 set in the server's environment).  The profile's top M pairs (M = the cache's slots)
//                                 score S * (1 - rank / M), and for the first R tier rounds usage = max(usage, score).
//   STRATA_SEED_SCALE=S           the domain seed's top score (default 2), STRATA_SEED_HOLD=R its rounds (default 16).
//   STRATA_CONV_WORKSET=N         (item 3) keep the routing counts of the last N conversations (1 = 8 of them): when a
//                                 request switches conversation the outgoing one's counts are kept with its tokens, and a
//                                 request whose prompt continues a kept conversation adds them back to the tier's usage,
//                                 scaled to STRATA_CONV_WORKSET_TOKENS tokens' worth (default 32).  This works whether or not
//                                 the conversation cache parked the K/V (it is keyed by the tokens, not by the image).
//
// The numbers (tools/pancake/replay_seeding.py on bench/prefetch/routing.bin, 10,909 slots, 32 swaps a window every
// window, decay 0.7 - the fast-hip preset): plain tier 81.2% of routed entries from VRAM; domain seed (leave-one-out
// code profile, S 2, R 16) 82.9% - the code-edit request 77.3 -> 78.8, the long request 84.7 -> 90.7, the short read
// 83.1 -> 83.3; a conversation resumed after another request 79.9 -> 80.6, 90.5 -> 91.4, 86.2 -> 87.0 on its second
// part (counts x32 tokens, added once; the bound with its own future counts is 81.3 / 93.2 / 88.4).  Larger S or R
// lost on the short request (the seed evicts what it needs), and a hold on the conversation seed lost (its own routing
// moves on).  At 160 swaps a window the domain seed gains less (85.0 -> 85.5): the tier then finds the set by itself.
// CAVEAT (one trace, four requests): with the prose request seeded from its own (in-sample) profile first, the held-out
// code edit after it gets 77.0-77.1 instead of 77.3 - the seed of one domain moves the tier away from the next one's
// set, and 16 rounds at 32 swaps do not undo it.  The in-sample requests gain 1-8 points (the bound for a profile that
// fits its domain).  So the domain part needs per-domain traces (MOD_PLAN 11, item 4's commands) before it is trusted;
// the conversation part's held-out gains are small but did not go negative in any arm.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace strata::program {

class ExpertSeeder {
public:
    struct Config {
        float domain_scale = 2.0f;   ///< STRATA_SEED_SCALE
        int domain_hold = 16;        ///< STRATA_SEED_HOLD (tier rounds)
        int conv_slots = 0;          ///< STRATA_CONV_WORKSET (0 = off)
        float conv_tokens = 32.0f;   ///< STRATA_CONV_WORKSET_TOKENS
    };

    /// The switches from the environment (see the top of this file); `dir` is STRATA_DOMAIN_PROFILES or empty.
    static Config from_env(std::string& dir) {
        Config c;
        const char* d = std::getenv("STRATA_DOMAIN_PROFILES");
        dir = d != nullptr ? d : "";
        if (const char* v = std::getenv("STRATA_SEED_SCALE")) c.domain_scale = std::max(0.0f, (float) std::atof(v));
        if (const char* v = std::getenv("STRATA_SEED_HOLD")) c.domain_hold = std::max(0, std::atoi(v));
        if (const char* v = std::getenv("STRATA_CONV_WORKSET")) {
            const int n = std::atoi(v);
            c.conv_slots = n == 1 ? 8 : std::max(0, n);
        }
        if (const char* v = std::getenv("STRATA_CONV_WORKSET_TOKENS")) c.conv_tokens = std::max(0.0f, (float) std::atof(v));
        return c;
    }

    ExpertSeeder() = default;
    ExpertSeeder(int64_t n_layers, int64_t n_expert, int64_t top_k, Config c)
        : nl_(n_layers), ne_(n_expert), k_(std::max<int64_t>(1, top_k)), cfg_(c) {}

    bool on() const { return !domains_.empty() || cfg_.conv_slots > 0; }
    bool conv_on() const { return cfg_.conv_slots > 0; }
    size_t domains() const { return domains_.size(); }
    size_t kept() const { return sets_.size(); }
    const Config& config() const { return cfg_; }

    /// A domain profile: `ranked` (layer, expert) pairs in the profile's order; the first `top` (the cache's slots) score
    /// scale * (1 - rank / top).  Pairs outside the geometry are skipped.
    void add_domain(const std::string& name, const std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t top) {
        std::vector<float> s((size_t) (nl_ * ne_), 0.0f);
        const int64_t n = std::min<int64_t>(top, (int64_t) ranked.size());
        for (int64_t r = 0; r < n; ++r) {
            const auto& [l, e] = ranked[(size_t) r];
            if (l < 0 || l >= nl_ || e < 0 || e >= ne_) continue;
            s[(size_t) (l * ne_ + e)] = cfg_.domain_scale * (1.0f - (float) r / (float) top);
        }
        domains_[name] = std::move(s);
    }

    /// At a request's start, before its prompt is read.  `outgoing`: the tokens the session holds (the conversation the
    /// previous request left; empty when none).  `prompt`: this request's.  `domain`: its GEN domain= key, or empty.
    /// Returns what it did, for the log ("" = nothing).
    template<class A, class B>
    std::string begin_request(std::vector<float>& usage, const std::vector<A>& prompt, const std::vector<B>& outgoing,
                              const std::string& domain) {
        if (usage.size() != (size_t) (nl_ * ne_) || !on()) return "";
        std::string what;
        // the same conversation goes on (its next turn): the tier holds its working set already (no seed), and its counts
        // go on.  A held domain seed of an earlier turn keeps its remaining rounds.
        if (!outgoing.empty() && same_conversation(outgoing, prompt)) { mark(usage); return ""; }
        hold_ = nullptr;   // another conversation: the previous one's held seed stops
        hold_left_ = 0;
        if (conv_on()) {
            keep(outgoing);
            cur_.assign(usage.size(), 0.0f);
            const WorkSet* best = nullptr;
            size_t best_l = 0;
            for (WorkSet& w : sets_) {
                const size_t l = common_prefix(w.ids, prompt);
                if (l > best_l && covers(w.ids.size(), l)) { best = &w; best_l = l; }
            }
            if (best != nullptr) {
                double mass = 0.0;
                for (float c : best->counts) mass += c;
                const float f = mass > 0.0 ? (float) (cfg_.conv_tokens * (double) k_ * (double) nl_ / mass) : 0.0f;
                for (size_t i = 0; i < usage.size(); ++i) usage[i] += f * best->counts[i];
                cur_ = best->counts;   // the conversation's tally goes on from where it was kept
                mark(usage);
                return "conversation working set (" + std::to_string(best_l) + " tokens in common)";
            }
        }
        auto it = domain.empty() ? domains_.end() : domains_.find(domain);
        if (it != domains_.end()) {
            hold_ = &it->second;
            hold_left_ = cfg_.domain_hold;
            if (hold_left_ > 0) max_in(usage, *hold_);
            else for (size_t i = 0; i < usage.size(); ++i) usage[i] += (*hold_)[i];
            what = "domain profile " + domain;
        }
        mark(usage);
        return what;
    }

    /// Right before the tier chooses a round's swaps: the routing counted since the last round goes into the running
    /// conversation's counts, then a held domain seed is laid under the counts again.
    void round_begin(std::vector<float>& usage) {
        if (usage.size() != (size_t) (nl_ * ne_)) return;
        if (conv_on() && prev_.size() == usage.size()) {
            if (cur_.size() != usage.size()) cur_.assign(usage.size(), 0.0f);
            for (size_t i = 0; i < usage.size(); ++i) {
                const float d = usage[i] - prev_[i];
                if (d > 0.0f) cur_[i] += d;
            }
        }
        if (hold_ != nullptr && hold_left_ > 0) {
            max_in(usage, *hold_);
            if (--hold_left_ == 0) hold_ = nullptr;
        }
    }

    /// Right after the round's decay: what the next round's counts are measured from.
    void round_end(const std::vector<float>& usage) { mark(usage); }

    /// Tests: the running conversation's counts.
    const std::vector<float>& current() const { return cur_; }

private:
    struct WorkSet {
        std::vector<int32_t> ids;
        std::vector<float> counts;
        uint64_t used = 0;
    };

    // a kept conversation is this prompt's when the prompt starts with at least half of its tokens (a next turn keeps
    // all of them, except where the template drops the last turn's reasoning) and at least 64
    static bool covers(size_t kept_len, size_t common) { return common >= 64 && 2 * common >= kept_len; }

    template<class A, class B>
    static size_t common_prefix(const std::vector<A>& a, const std::vector<B>& b) {
        const size_t n = std::min(a.size(), b.size());
        size_t i = 0;
        while (i < n && (int64_t) a[i] == (int64_t) b[i]) ++i;
        return i;
    }

    template<class A, class B>
    static bool same_conversation(const std::vector<A>& outgoing, const std::vector<B>& prompt) {
        return covers(outgoing.size(), common_prefix(outgoing, prompt));
    }

    void mark(const std::vector<float>& usage) {
        if (conv_on()) prev_ = usage;
    }

    static void max_in(std::vector<float>& usage, const std::vector<float>& seed) {
        for (size_t i = 0; i < usage.size(); ++i) usage[i] = std::max(usage[i], seed[i]);
    }

    template<class B>
    void keep(const std::vector<B>& outgoing) {
        double mass = 0.0;
        for (float c : cur_) mass += c;
        if (outgoing.size() < 64 || mass <= 0.0) return;
        ++clock_;
        for (WorkSet& w : sets_)   // the same conversation, kept at an earlier turn: replaced
            if (covers(w.ids.size(), common_prefix(w.ids, outgoing))) {
                w.ids.assign(outgoing.begin(), outgoing.end());
                w.counts = cur_;
                w.used = clock_;
                return;
            }
        if ((int) sets_.size() >= cfg_.conv_slots) {   // the least recently kept one goes
            auto old = std::min_element(sets_.begin(), sets_.end(),
                                        [](const WorkSet& a, const WorkSet& b) { return a.used < b.used; });
            sets_.erase(old);
        }
        WorkSet w;
        w.ids.assign(outgoing.begin(), outgoing.end());
        w.counts = cur_;
        w.used = clock_;
        sets_.push_back(std::move(w));
    }

    int64_t nl_ = 0, ne_ = 0, k_ = 1;
    Config cfg_;
    std::map<std::string, std::vector<float>> domains_;
    std::vector<WorkSet> sets_;
    std::vector<float> cur_, prev_;
    const std::vector<float>* hold_ = nullptr;
    int hold_left_ = 0;
    uint64_t clock_ = 0;
};

}  // namespace strata::program
