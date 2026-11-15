// include/strata/kernels/rkv.hpp - R-KV redundancy-aware eviction of decode-trace blocks from the QSA selection
// (pancake, opt-in, LOSSY; LEADS.md "Reasoning KV compression (R-KV, ThinKV, TIP)").  Status: MOD_PLAN.md / TEST_LEDGER.md.
//
// R-KV (Cai et al., "R-KV: Redundancy-aware KV Cache Compression for Reasoning Models", arXiv 2505.24133): while a reasoning
// trace is decoded, once the trace's cache exceeds budget + buffer, every cached token gets
//     Z = lambda * I - (1 - lambda) * R
// with I its attention importance from the most recent queries and R its redundancy (mean cosine similarity of its key to the
// other cached keys); the lowest Z are evicted until the budget is met.
//
// Here (first version): the unit is a QSA indexer block (idx_block cells, one pooled key), the importance is the indexer's
// block score from this window's queries (max over the rows, min-max normalised over the candidates), the redundancy is the
// mean cosine similarity of the pooled keys among the candidates.  Candidates: completed blocks at or after the trace start
// (the first window after a prompt read or a rewind), past `sink` blocks and before the last `recent` blocks, not yet evicted.
// An evicted block is never selected again (its scores are set to -3e38 before the top-k); the K/V stays where it is, so this
// changes WHICH cells the attention reads, not the memory held (a page-level free is the follow-up).  ThinKV is the same
// family (thought-type budgets); TIP needs a trained model (parked).
//
// Switches (read once): STRATA_RKV=<budget blocks> (0/unset: off), STRATA_RKV_BUFFER (64 blocks), STRATA_RKV_LAMBDA (0.1, the
// paper's default), STRATA_RKV_SINK (4), STRATA_RKV_RECENT (32).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace strata::kernels {

inline constexpr int kRkvCap = 2048;   ///< candidates considered per update (budget + buffer must not exceed it)

struct RkvConfig {
    int budget = 0, buffer = 64, sink = 4, recent = 32;
    float lambda = 0.1f;
};
inline const RkvConfig& rkv_config() {
    static const RkvConfig c = [] {
        RkvConfig r;
        auto num = [](const char* n, double d) { const char* v = std::getenv(n); return v != nullptr && *v ? std::atof(v) : d; };
        r.budget = std::max(0, (int) num("STRATA_RKV", 0));
        r.buffer = std::max(1, (int) num("STRATA_RKV_BUFFER", 64));
        r.lambda = (float) num("STRATA_RKV_LAMBDA", 0.1);
        r.sink = std::max(0, (int) num("STRATA_RKV_SINK", 4));
        r.recent = std::max(0, (int) num("STRATA_RKV_RECENT", 32));
        if (r.budget + r.buffer > kRkvCap) r.budget = std::max(0, kRkvCap - r.buffer);
        return r;
    }();
    return c;
}
inline bool rkv_on() { return rkv_config().budget > 0; }

/// One QSA layer's update, inside the window graph: reads the window's block scores (n rows of max_blocks), the step record
/// of row n - 1 (n_bid at [2]), the pooled keys (rows of idx_dim), the trace's first block (*start_block, device), and sets
/// the evicted blocks' bits in `evict` (max_blocks bits).  Nothing happens below budget + buffer candidates.  One block.
void rkv_update(const float* scores, int n, int64_t max_blocks, const int32_t* steps, int step_count, const float* pooled,
                int idx_dim, const int32_t* start_block, const RkvConfig& c, uint32_t* evict, void* stream);
/// scores[r][b] = -3e38 for every evicted block b (r < n).  Graph-capturable.
void rkv_mask_scores(float* scores, int n, int64_t max_blocks, const uint32_t* evict, void* stream);

/// The host reference of rkv_update's choice (the parity test): candidate list -> the blocks to evict, ascending.
inline std::vector<int> rkv_choose_host(const float* scores, int n, int64_t max_blocks, int n_bid, const float* pooled,
                                        int idx_dim, int start_block, const RkvConfig& c, const std::vector<uint32_t>& evict) {
    std::vector<int> cand;
    const int lo = std::max(c.sink, start_block), hi = n_bid - c.recent;
    for (int b = lo; b < hi && (int) cand.size() < kRkvCap; ++b)
        if (!((evict[(size_t) b >> 5] >> (b & 31)) & 1u)) cand.push_back(b);
    const int M = (int) cand.size();
    if (M < c.budget + c.buffer) return {};
    std::vector<float> I((size_t) M), nrm((size_t) M), Z((size_t) M);
    float mn = 3e38f, mx = -3e38f;
    for (int i = 0; i < M; ++i) {
        float v = -3e38f;
        for (int r = 0; r < n; ++r) v = std::max(v, scores[(size_t) r * max_blocks + cand[(size_t) i]]);
        I[(size_t) i] = v;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
        float s = 0.0f;
        for (int d = 0; d < idx_dim; ++d) s = std::fma(pooled[(size_t) cand[(size_t) i] * idx_dim + d], pooled[(size_t) cand[(size_t) i] * idx_dim + d], s);
        nrm[(size_t) i] = std::sqrt(s) + 1e-12f;
    }
    const float span = mx - mn > 1e-30f ? mx - mn : 1.0f;
    for (int i = 0; i < M; ++i) {
        float rsum = 0.0f;
        const float* ki = pooled + (size_t) cand[(size_t) i] * idx_dim;
        for (int j = 0; j < M; ++j) {
            if (j == i) continue;
            const float* kj = pooled + (size_t) cand[(size_t) j] * idx_dim;
            float d = 0.0f;
            for (int e = 0; e < idx_dim; ++e) d = std::fma(ki[e], kj[e], d);
            rsum += d / (nrm[(size_t) i] * nrm[(size_t) j]);
        }
        const float R = rsum / (float) (M - 1);
        Z[(size_t) i] = std::fma(c.lambda, (I[(size_t) i] - mn) / span, -((1.0f - c.lambda) * R));   // the kernel's fmaf
    }
    std::vector<int> ord((size_t) M);
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b) { return Z[(size_t) a] < Z[(size_t) b] || (Z[(size_t) a] == Z[(size_t) b] && a < b); });
    std::vector<int> out;
    for (int k = 0; k < M - c.budget; ++k) out.push_back(cand[(size_t) ord[(size_t) k]]);
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace strata::kernels
