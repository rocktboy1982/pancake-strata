#pragma once
// STRATA_PROMPT_TAIL (opt-in, default off; LEADS.md Part Q2, idea from Strata-HotRod commit fd41f7174, MIT): the first
// decode windows route like the END of the prompt, not like the whole prompt (STRATA_ADAPT_PROMPT) and not like an
// earlier conversation (STRATA_CONV_WORKSET).  The prompt path counts, per layer, the experts the LAST chunk routes to
// (router ranks 0 .. STRATA_PROMPT_TAIL_RANKS-1, default 2; the ids are the ones the prompt path already holds on the
// host for its grouping, so nothing is read back from the GPU).  When the prompt is read and its lent slots are back,
// the counts are added to the adaptive tier's usage as STRATA_PROMPT_TAIL_W tokens' worth (default 16: `W * count /
// tokens of the chunk`, so a layer's rows sum to W * ranks), and one blocking tier round runs before the first window
// (as STRATA_TIER_REWARM does, with its own swap cap STRATA_PROMPT_TAIL_MAX, default no cap): the tier's own rule puts
// the tail's hottest experts that are not in VRAM in first.  Placement only: which experts the GPU or the CPU computes
// never changes a token.  STRATA_PROMPT_TAIL_MIN (default 256): prompts that read fewer tokens are left alone.
// Header-only, no GPU (tests/program/prompt_tail_test.cpp).  Not recorded on the paths that keep the routing ids on the
// GPU, nor on the layer-major prompt path (Prefill::on_routes is not called there): those prompts get no tail.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace strata::program {

class PromptTail {
public:
    struct Config {
        bool on = false;                                       ///< STRATA_PROMPT_TAIL
        int ranks = 2;                                         ///< STRATA_PROMPT_TAIL_RANKS (router ranks counted, >= 1)
        float w = 16.0f;                                       ///< STRATA_PROMPT_TAIL_W (tokens' worth added to the usage)
        int64_t min_tokens = 256;                              ///< STRATA_PROMPT_TAIL_MIN (tokens the prompt reads)
        int cap = std::numeric_limits<int>::max();             ///< STRATA_PROMPT_TAIL_MAX (swaps of the round)
    };

    /// The switches from the environment; unset, empty or 0 STRATA_PROMPT_TAIL is off.
    static Config from_env() {
        Config c;
        const char* v = std::getenv("STRATA_PROMPT_TAIL");
        c.on = v != nullptr && v[0] != '\0' && std::atoi(v) != 0;
        if (const char* r = std::getenv("STRATA_PROMPT_TAIL_RANKS")) c.ranks = std::max(1, std::atoi(r));
        if (const char* w = std::getenv("STRATA_PROMPT_TAIL_W")) c.w = std::max(0.0f, (float) std::atof(w));
        if (const char* m = std::getenv("STRATA_PROMPT_TAIL_MIN")) c.min_tokens = std::max<int64_t>(1, std::atoll(m));
        if (const char* m = std::getenv("STRATA_PROMPT_TAIL_MAX")) {
            const int n = std::atoi(m);
            if (n > 0) c.cap = n;
        }
        return c;
    }

    PromptTail() = default;
    PromptTail(int64_t n_layers, int64_t n_expert, int64_t top_k, Config c)
        : nl_(n_layers), ne_(n_expert), k_(std::max<int64_t>(1, top_k)), cfg_(c) {
        cnt_.assign((size_t) (nl_ * ne_), 0.0f);
        tok_.assign((size_t) nl_, 0);
    }

    bool on() const { return cfg_.on; }
    const Config& config() const { return cfg_; }

    /// A prompt run starts: the last run's tail is dropped.
    void reset() {
        std::lock_guard<std::mutex> lk(mu_);
        std::fill(cnt_.begin(), cnt_.end(), 0.0f);
        std::fill(tok_.begin(), tok_.end(), 0);
    }

    /// One layer of one prompt chunk: `ids` = T x k router picks in rank order (-1 = none), n = T * k.  The layer's row
    /// is REPLACED, so after the run it holds the last chunk that reached the layer.  Off: nothing is touched.
    void record(int64_t layer, const int32_t* ids, int64_t n) {
        if (!cfg_.on || layer < 0 || layer >= nl_ || ids == nullptr || n <= 0) return;
        std::lock_guard<std::mutex> lk(mu_);
        float* row = cnt_.data() + (size_t) layer * (size_t) ne_;
        std::fill(row, row + ne_, 0.0f);
        for (int64_t i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            if ((int64_t) (i % k_) >= (int64_t) cfg_.ranks) continue;
            if (e >= 0 && e < ne_) row[e] += 1.0f;
        }
        tok_[(size_t) layer] = n / k_;
    }

    /// Layers that have a tail.
    int64_t layers_seen() const {
        std::lock_guard<std::mutex> lk(mu_);
        int64_t n = 0;
        for (int64_t t : tok_) n += t > 0 ? 1 : 0;
        return n;
    }

    /// Adds `w * count / tokens` of every layer that has a tail into `usage` (n_layers x n_expert).  Returns the mass
    /// added (0 = nothing: off, no tail, or a usage vector of another shape).
    double apply(std::vector<float>& usage) const {
        if (!cfg_.on || usage.size() != (size_t) (nl_ * ne_)) return 0.0;
        std::lock_guard<std::mutex> lk(mu_);
        double mass = 0.0;
        for (int64_t l = 0; l < nl_; ++l) {
            const int64_t t = tok_[(size_t) l];
            if (t <= 0) continue;
            const float f = cfg_.w / (float) t;
            const float* row = cnt_.data() + (size_t) l * (size_t) ne_;
            float* u = usage.data() + (size_t) l * (size_t) ne_;
            for (int64_t e = 0; e < ne_; ++e)
                if (row[e] > 0.0f) { u[e] += f * row[e]; mass += (double) (f * row[e]); }
        }
        return mass;
    }

    /// The `m` experts the layer's tail counts highest (ties: lowest id first), for the log and the tests.
    std::vector<int32_t> hottest(int64_t layer, int m) const {
        std::vector<std::pair<float, int32_t>> v;
        if (layer < 0 || layer >= nl_) return {};
        std::lock_guard<std::mutex> lk(mu_);
        const float* row = cnt_.data() + (size_t) layer * (size_t) ne_;
        for (int64_t e = 0; e < ne_; ++e)
            if (row[e] > 0.0f) v.emplace_back(-row[e], (int32_t) e);
        std::sort(v.begin(), v.end());
        std::vector<int32_t> out;
        for (size_t i = 0; i < v.size() && (int) i < m; ++i) out.push_back(v[i].second);
        return out;
    }

private:
    int64_t nl_ = 0, ne_ = 0, k_ = 1;
    Config cfg_;
    mutable std::mutex mu_;
    std::vector<float> cnt_;
    std::vector<int64_t> tok_;
};

}  // namespace strata::program
