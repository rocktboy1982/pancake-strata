// include/strata/kernels/xattn.hpp - XAttention block-sparse prompt attention for long prompts (pancake, opt-in, LOSSY;
// LEADS.md "XAttention block-sparse prefill >64K").  Status and measurements: MOD_PLAN.md / TEST_LEDGER.md.
//
// XAttention (Xu, Guo, Han et al., "XAttention: Block Sparse Attention with Antidiagonal Scoring", arXiv 2503.16428, MIT
// Han lab, ICML 2025): the importance of a (query block, key block) pair of B x B attention scores is estimated from the sum
// of its ANTIDIAGONALS sampled with stride S - every row and every column of a block meets one antidiagonal, so a vertical or
// a slash pattern cannot be missed.  Per query block: the strided query rows are the S rows of each S x S sub-block read
// bottom-up (q[a*S + S-1-k]), the strided keys top-down (k[c*S + k]), their product summed over k is that sub-block's
// antidiagonal; scaled by 1/sqrt(d)/S, softmaxed over the row's causal keys, summed per key block and averaged over the rows,
// it is a distribution over the key blocks.  The key blocks are taken in decreasing order until their mass reaches the
// threshold tau; the first block and the blocks of the query block's own positions are always kept.  The attention then
// runs over the kept blocks only.
//
// Where: strata-q36 (qwen35), the dense attention layers' prompt chunks, beside BFLA (q36_engine.cu); BFLA wins when both
// are set.  One call is one chunk of one layer, the BFLA layout:
//   q16 [T][H][HD] FP16 bits (normed, rotated; positions p0 .. p0 + T - 1), k16 / v16 [HKV][P][HD] FP16 bits, P = p0 + T,
//   out [T][H][HD] FP32 = softmax(q k^T / sqrt(HD)) v over the kept cells (causal), not gated.
// First version: plain FP32 kernels (one wave per query row and head for the attention), no matrix cores; HD a multiple of
// 32 up to 256, P <= 2048 blocks of 128.
//
// Switches (read once): STRATA_XATTN=1 (off when unset), STRATA_XATTN_MIN (65536: a chunk takes it when p0 + T reaches it),
// STRATA_XATTN_TAU (0.9), STRATA_XATTN_STRIDE (8; 4, 8 or 16).  tau >= 1 keeps every causal block: dense attention, for the
// parity test.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace strata::kernels {

inline constexpr int kXattnBlock = 128;
inline constexpr int kXattnMaxKb = 2048;

struct XattnConfig {
    bool on = false;
    int64_t min_len = 65536;
    float tau = 0.9f;
    int stride = 8;
};
inline const XattnConfig& xattn_config() {
    static const XattnConfig c = [] {
        XattnConfig r;
        const char* v = std::getenv("STRATA_XATTN");
        r.on = v != nullptr && v[0] == '1';
        if (const char* e = std::getenv("STRATA_XATTN_MIN"); e && *e) r.min_len = std::atoll(e);
        if (const char* e = std::getenv("STRATA_XATTN_TAU"); e && *e) r.tau = (float) std::atof(e);
        if (const char* e = std::getenv("STRATA_XATTN_STRIDE"); e && *e) {
            const int s = std::atoi(e);
            r.stride = s == 4 || s == 8 || s == 16 ? s : 8;
        }
        return r;
    }();
    return c;
}

/// Bytes of scratch xattn_prefill_attn needs for a chunk of T rows over a context of up to P cells, H query heads.
inline size_t xattn_scratch_bytes(int64_t T, int64_t P, int H, int stride) {
    const int64_t nqb = (T + kXattnBlock - 1) / kXattnBlock, nkb = (P + kXattnBlock - 1) / kXattnBlock;
    const size_t mask = (size_t) nqb * H * nkb;
    const size_t rows = (size_t) nqb * H * (size_t) nkb * (size_t) (kXattnBlock / stride) * sizeof(float);
    return ((mask + 255) & ~(size_t) 255) + rows;
}

/// The chunk's attention by XAttention's selection.  False (nothing launched) when the shape is outside the kernels' range
/// (HD % 32, HD > 256, H % HKV, more than kXattnMaxKb key blocks, scratch too small): the caller keeps its own path.
/// `mask_out` (may be null): the scratch's mask, [H][nqb][nkb] bytes, for the parity test.
bool xattn_prefill_attn(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t p0, int H, int HKV,
                        int HD, const XattnConfig& c, void* scratch, size_t scratch_bytes, float* out, void* stream,
                        const uint8_t** mask_out = nullptr);

// ---- host references (xattn_parity)

/// The block mask of one (query block qb, head h) as the kernel computes it, from FP32 copies of q (T, H, HD) and k
/// (HKV, P, HD).  mask has nkb entries (1 = kept).
inline void xattn_select_host(const float* q, const float* k, int64_t T, int64_t p0, int H, int HKV, int HD, int qb, int h,
                              const XattnConfig& c, std::vector<uint8_t>& mask) {
    const int64_t P = p0 + T, B = kXattnBlock, S = c.stride, R = B / S;
    const int64_t nkb = (P + B - 1) / B;
    const int kvh = h / (H / HKV);
    const int64_t t_end = std::min<int64_t>(T, (int64_t) qb * B + B);
    const int64_t nkq = (p0 + t_end - 1) / B + 1;   // key blocks a query of this block can see
    std::vector<double> A((size_t) nkb, 0.0);
    int rows = 0;
    const double scale = 1.0 / std::sqrt((double) HD) / (double) S;
    for (int64_t a = 0; a < R; ++a) {
        const int64_t t0 = (int64_t) qb * B + a * S;
        if (t0 >= T) break;
        const int64_t qpos = p0 + t0;
        std::vector<double> sv;
        std::vector<int64_t> kbv;
        double mx = -1e300;
        for (int64_t kb = 0; kb < nkq; ++kb)
            for (int64_t cc = 0; cc < R; ++cc) {
                const int64_t kc = kb * B + cc * S;
                if (kc > qpos || kc >= P) continue;
                double s = 0.0;
                for (int64_t kk = 0; kk < S; ++kk) {
                    const int64_t t = t0 + (S - 1 - kk), cell = kc + kk;
                    if (t >= T || cell >= P) continue;
                    const float* qr = q + ((size_t) t * H + h) * HD;
                    const float* kr = k + ((size_t) kvh * P + cell) * HD;
                    for (int d = 0; d < HD; ++d) s += (double) qr[d] * kr[d];
                }
                s *= scale;
                sv.push_back(s);
                kbv.push_back(kb);
                mx = std::max(mx, s);
            }
        double sum = 0.0;
        for (const double s : sv) sum += std::exp(s - mx);
        for (size_t i = 0; i < sv.size(); ++i) A[(size_t) kbv[i]] += std::exp(sv[i] - mx) / sum;
        ++rows;
    }
    mask.assign((size_t) nkb, 0);
    if (rows == 0) return;
    for (auto& v : A) v /= rows;
    std::vector<int> ord((size_t) nkq);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](int x, int y) { return A[(size_t) x] > A[(size_t) y]; });
    double cum = 0.0;
    for (const int kb : ord) {
        if (cum >= (double) c.tau) break;
        mask[(size_t) kb] = 1;
        cum += A[(size_t) kb];
    }
    mask[0] = 1;
    for (int64_t kb = (p0 + (int64_t) qb * B) / B; kb < nkq; ++kb) mask[(size_t) kb] = 1;
}

/// The attention of query row t, head h over the kept blocks (FP64), out HD values.
inline void xattn_attn_host(const float* q, const float* k, const float* v, int64_t T, int64_t p0, int H, int HKV, int HD,
                            int64_t t, int h, const uint8_t* mask_qb_h /* nkb entries or null = all */, double* out) {
    const int64_t P = p0 + T, B = kXattnBlock, qpos = p0 + t;
    const int kvh = h / (H / HKV);
    const float* qr = q + ((size_t) t * H + h) * HD;
    double m = -1e300, l = 0.0;
    std::vector<double> acc((size_t) HD, 0.0);
    for (int64_t cell = 0; cell <= qpos && cell < P; ++cell) {
        if (mask_qb_h != nullptr && !mask_qb_h[cell / B]) continue;
        const float* kr = k + ((size_t) kvh * P + cell) * HD;
        double s = 0.0;
        for (int d = 0; d < HD; ++d) s += (double) qr[d] * kr[d];
        s /= std::sqrt((double) HD);
        const double mn = std::max(m, s), corr = std::exp(m - mn), e = std::exp(s - mn);
        const float* vr = v + ((size_t) kvh * P + cell) * HD;
        for (int d = 0; d < HD; ++d) acc[(size_t) d] = acc[(size_t) d] * corr + e * vr[d];
        l = l * corr + e;
        m = mn;
    }
    for (int d = 0; d < HD; ++d) out[d] = l > 0 ? acc[(size_t) d] / l : 0.0;
}

}  // namespace strata::kernels
