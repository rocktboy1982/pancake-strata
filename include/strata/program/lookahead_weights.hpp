#pragma once
// MOD_PLAN.md section 6 item 29 (LEADS.md Part H, fork ElXreno `2c1219eb` `e8c82e8f`): a TRAINED per-layer look-ahead
// predictor in place of the router in RouterLookahead.  The idea is the fork's; this code is written here from the
// behaviour of its public diff (a format and a rule), the diff having been read as untrusted data.
//
//   STRATA_PRED_DUMP=<dir>          append every verify-window layer to <dir>/trace.bin (measurement; for the fit)
//   STRATA_LOOKAHEAD_WEIGHTS=<file> the predictor (file format LKW1 below), used by the CPU look-ahead thread only: the GPU
//                                   look-ahead (STRATA_LOOKAHEAD_GPU, on by default) runs the model's own router in the verify
//                                   graph and cannot take other weights, so loading weights turns that path off
//   STRATA_LOOKAHEAD_B=<n>          most experts warmed per layer (default 8)
//   STRATA_LOOKAHEAD_TAU=<p>        warm only experts whose predicted probability is above p (default 0: no floor)
//
// File LKW1 (little endian), written by tools/pancake/fit_lookahead.py:
//   u32 magic 0x31574b4c ("LKW1"), u32 n_layers, u32 n_expert, u32 n_embd
//   per layer L = 0 .. n_layers-1:  bf16 W[n_expert][n_embd]  then  f32 bias[n_expert]
//   entry L predicts the experts of layer L from the MoE input of layer L - 1 (entry 0 is never used).
// Trace RPD2 (little endian), one record per window and layer, appended:
//   u32 magic 0x32445250 ("RPD2"), u32 layer, u32 n_tok, u32 k, u32 window_lo, u32 window_hi, u32 n_expert
//   bf16 x[n_tok][n_embd]   the layer's MoE input rows
//   i16 ids[n_tok][k]       the experts the layer routed (the truth)
//   u8 res[n_tok][k]        1 when that expert was resident in the GPU cache at the layer
//   u8 map[(n_expert + 7) / 8]  the layer's residency bitmap
// n_embd is not in the record: the reader takes it from the model (the fit script's --n-embd).  A pair for the fit is
// (x of layer L - 1, ids of layer L) of the same window and token.
//
// The pick rule is pure, here, so a ctest covers it (lookahead_weights_test).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace strata::program {

constexpr uint32_t kLkwMagic = 0x31574b4cu;
constexpr uint32_t kPredMagic = 0x32445250u;

/// Bytes of an LKW1 file.
inline uint64_t lkw_file_bytes(uint64_t n_layers, uint64_t n_expert, uint64_t n_embd) {
    return 16 + n_layers * (n_expert * n_embd * 2 + n_expert * 4);
}

/// Is `head[4]` the header of a predictor for this model?
inline bool lkw_header_ok(const uint32_t* head, uint64_t n_layers, uint64_t n_expert, uint64_t n_embd) {
    return head[0] == kLkwMagic && head[1] == n_layers && head[2] == n_expert && head[3] == n_embd;
}

/// Bytes of one RPD2 record.
inline uint64_t pred_record_bytes(uint64_t n_tok, uint64_t k, uint64_t n_expert, uint64_t n_embd) {
    return 28 + n_tok * n_embd * 2 + n_tok * k * 2 + n_tok * k + (n_expert + 7) / 8;
}

/// RPD2 record header.
inline void pred_head(uint32_t* out, uint32_t layer, uint32_t n_tok, uint32_t k, uint64_t window, uint32_t n_expert) {
    out[0] = kPredMagic; out[1] = layer; out[2] = n_tok; out[3] = k;
    out[4] = (uint32_t) (window & 0xffffffffu); out[5] = (uint32_t) (window >> 32); out[6] = n_expert;
}

/// Round to bf16 (nearest even), the format of the trace rows and the file weights.
inline uint16_t pred_bf16(float v) {
    uint32_t u = 0;
    std::memcpy(&u, &v, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;   // NaN stays a NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}

/// The experts to warm for one window of `nt` tokens at one layer.  `logits[t * n_expert + e]` is the predictor's score
/// before the bias.  Per expert the best score over the window's tokens counts (`tau > 0`: the best softmax probability of
/// logit + bias over the experts, per token); resident experts (`res_row[e] >= 0`; null: none) never count.  The ranking is
/// by that best score, descending, ties to the lower id, and ends at the first expert with no finite score or, with
/// `tau > 0`, a probability not above tau.  The first `budget` ranked experts are wanted, with confidence
/// (budget - j) / budget for rank j.  `ranked` (at most `kmax` ids) is the full ranking for the statistics.
/// Returns the number wanted.
inline int lookahead_weights_pick(const float* logits, int nt, int n_expert, const float* bias, const int32_t* res_row, int budget,
                                  float tau, int kmax, std::vector<int32_t>& want, std::vector<float>& conf,
                                  std::vector<int32_t>& ranked) {
    want.clear(); conf.clear(); ranked.clear();
    if (nt <= 0 || n_expert <= 0 || budget <= 0 || kmax <= 0) return 0;
    const float ninf = -std::numeric_limits<float>::infinity();
    std::vector<float> best((size_t) n_expert, ninf);
    for (int t = 0; t < nt; ++t) {
        const float* lt = logits + (size_t) t * (size_t) n_expert;
        if (tau > 0.0f) {
            float mx = ninf;
            for (int e = 0; e < n_expert; ++e) mx = std::max(mx, lt[e] + (bias != nullptr ? bias[e] : 0.0f));
            if (!std::isfinite(mx)) continue;
            double z = 0.0;
            for (int e = 0; e < n_expert; ++e) z += std::exp((double) (lt[e] + (bias != nullptr ? bias[e] : 0.0f) - mx));
            for (int e = 0; e < n_expert; ++e)
                best[(size_t) e] = std::max(best[(size_t) e], (float) (std::exp((double) (lt[e] + (bias != nullptr ? bias[e] : 0.0f) - mx)) / z));
        } else {
            for (int e = 0; e < n_expert; ++e) {
                const float s = lt[e] + (bias != nullptr ? bias[e] : 0.0f);
                if (std::isfinite(s)) best[(size_t) e] = std::max(best[(size_t) e], s);
            }
        }
    }
    if (res_row != nullptr)
        for (int e = 0; e < n_expert; ++e)
            if (res_row[e] >= 0) best[(size_t) e] = ninf;
    std::vector<int32_t> order((size_t) n_expert);
    for (int e = 0; e < n_expert; ++e) order[(size_t) e] = e;
    const int km = std::min(kmax, n_expert);
    std::partial_sort(order.begin(), order.begin() + km, order.end(), [&](int32_t a, int32_t b) {
        return best[(size_t) a] > best[(size_t) b] || (best[(size_t) a] == best[(size_t) b] && a < b);
    });
    for (int j = 0; j < km; ++j) {
        const int32_t e = order[(size_t) j];
        if (!std::isfinite(best[(size_t) e])) break;
        if (tau > 0.0f && !(best[(size_t) e] > tau)) break;
        ranked.push_back(e);
        if (j < budget) { want.push_back(e); conf.push_back((float) (budget - j) / (float) budget); }
    }
    return (int) want.size();
}

}  // namespace strata::program
