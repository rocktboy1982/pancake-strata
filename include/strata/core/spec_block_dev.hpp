// include/strata/core/spec_block_dev.hpp - the DEVICE form of block verification (STRATA_MTP_BLOCK_VERIFY=1, LEADS.md S12): the
// flat-array layout the second sampler kernel reads, the per-entry arithmetic it shares with the host, and a serial twin of the
// kernel's control flow that the host test runs against spec_block_verify (spec_block.hpp), the tested rule.
//
// Why a second kernel: spec_verify_merge_kernel (src/kernels/cuda/sampler.cu) is one warp per window row and rows are independent;
// block verification needs every row's P and Q at once (the running ratio w_j chains the rows, and the winner is the LAST row,
// counting down from gamma, whose uniform is below h_j).  So the device path is three launches:
//   1. sampler_split_part_kernel          (unchanged: per row, per vocabulary block, the top-k lists)
//   2. spec_block_rows_kernel             (one warp per row: the merge and sampled_probs_warp's chain, as spec_verify_merge_kernel's
//                                          first half, written to the row's P slot: ids, double probabilities, n)
//   3. spec_block_verify_kernel           (ONE warp, <<<1, 32>>>: w_j, then j = gamma .. 0, the winner's residual draw)
// Everything below the launches is this header's arithmetic.
//
// Rows and uniforms (the same position-keyed Philox streams as spec_prob.hpp, so a seed replays the same text whatever the windows):
//   window row t (0..gamma), position pos_t = counter + t; draft j (0..gamma-1) is the token at row j + 1, judged with row j's P and Q.
//   eta(j), j = 1..gamma  = philox(seed, pos_j | kSpecCtrAccept)   (consumed for j = gamma down to the winner; row 0 never asks)
//   resid(win)            = win < gamma ? philox(seed, pos_win | kSpecCtrResid) : philox(seed, pos_win)   (the plain pick of the bonus row)
// Greedy requests (temperature 0) never reach the sampled path; a one-hot P that does reach it (a tiny temperature) gives the
// longest matching prefix plus the argmax, as spec_block_test checks.
//
// Arithmetic: every product and difference of the residual is written with the round-to-nearest intrinsics on the device
// (__dmul_rn, __dsub_rn), so the compiler cannot fuse `w * p - q` into an FMA; the host sums and picks in the same index order
// (the device keeps the sums on lane 0, in order, as the sampler's own kernels do).  Bit-equality with the host is by this
// construction and by spec_block_dev_test; it is NOT confirmed on a GPU.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "strata/core/spec_block.hpp"

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_SBD_HD __host__ __device__
#else
#define STRATA_SBD_HD
#endif

namespace strata::core::spec_block_dev {

inline constexpr int kRowCap = 64;   ///< kSelMax: entries of a row's P list

/// The window the device kernel judges: P rows in three flat arrays, Q rows in the engine's qbuf layout, the drafts.
struct View {
    const int32_t* pid;    ///< [gamma + 1][kRowCap] token ids of row r's P list
    const double* pp;      ///< [gamma + 1][kRowCap] probabilities (the sampler's own `ex`)
    const int* pn;         ///< [gamma + 1] entries in row r's list
    const int32_t* qbuf;   ///< rows < n_q: kSpecQStride int32 each: kSpecQEntries ids (-1 ends the list), then the probabilities as float bits
    int n_q;
    const int32_t* d;      ///< the gamma drafts
    int gamma;
};

/// Bytes of the three P arrays for `rows` rows, in this order: pn (padded to 8), pid, pp.  The kernels carve one scratch buffer with it.
inline size_t rows_bytes(int rows) {
    const size_t n = (size_t) rows;
    return ((n * sizeof(int) + 7) & ~(size_t) 7) + n * kRowCap * sizeof(int32_t) + n * kRowCap * sizeof(double);
}
STRATA_SBD_HD inline int* rows_pn(void* base) { return static_cast<int*>(base); }
STRATA_SBD_HD inline int32_t* rows_pid(void* base, int rows) {
    return reinterpret_cast<int32_t*>(static_cast<char*>(base) + (((size_t) rows * sizeof(int) + 7) & ~(size_t) 7));
}
STRATA_SBD_HD inline double* rows_pp(void* base, int rows) {
    return reinterpret_cast<double*>(rows_pid(base, rows) + (size_t) rows * kRowCap);
}

STRATA_SBD_HD inline double mul_rn(double a, double b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __dmul_rn(a, b);
#else
    return a * b;
#endif
}
STRATA_SBD_HD inline double sub_rn(double a, double b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __dsub_rn(a, b);
#else
    return a - b;
#endif
}

/// P_row(x): 0 when x is not in the row's list.
STRATA_SBD_HD inline double p_of(const View& v, int row, int32_t x) {
    const int n = v.pn[row] < kRowCap ? v.pn[row] : kRowCap;
    for (int i = 0; i < n; ++i)
        if (v.pid[row * kRowCap + i] == x) return v.pp[row * kRowCap + i];
    return 0.0;
}

/// Q_row(x) for the row's draft d = v.d[row]: a point mass (1 at d, 0 elsewhere) when the row has no list or d is not in it with q > 0.
STRATA_SBD_HD inline double q_of(const View& v, int row, int32_t x) {
    const int32_t dr = v.d[row];
    if (row < v.n_q) {
        const int32_t* ids = v.qbuf + (size_t) row * strata::core::kSpecQStride;
        const float* pr = reinterpret_cast<const float*>(ids + strata::core::kSpecQEntries);
        double qd = -1.0;
        for (int i = 0; i < strata::core::kSpecQEntries && ids[i] >= 0; ++i)
            if (ids[i] == dr) { qd = (double) pr[i]; break; }
        if (qd > 0.0) {
            for (int i = 0; i < strata::core::kSpecQEntries && ids[i] >= 0; ++i)
                if (ids[i] == x) return (double) pr[i];
            return 0.0;
        }
    }
    return x == dr ? 1.0 : 0.0;
}

/// w_{j+1} from w_j (spec_block_ratios).
STRATA_SBD_HD inline double ratio_next(const View& v, int j, double wj) {
    const double pd = p_of(v, j, v.d[j]);
    const double qd = q_of(v, j, v.d[j]);
    if (!(pd > 0.0 && qd > 0.0)) return 0.0;
    const double x = mul_rn(wj, pd) / qd;
    return x < 1.0 ? x : 1.0;
}

/// Entry i of row j's residual weights (spec_block_residual): max(0, w_j P_j(x) - Q_j(x)), P_gamma itself for the bonus row.
STRATA_SBD_HD inline double residual_entry(const View& v, int j, double wj, int i) {
    const double p = v.pp[j * kRowCap + i];
    if (j == v.gamma) return p;
    const double r = sub_rn(mul_rn(wj, p), q_of(v, j, v.pid[j * kRowCap + i]));
    return r < 0.0 ? 0.0 : r;
}

/// h_j (spec_block_h).
STRATA_SBD_HD inline double h_of(int gamma, int j, double wj, double s) {
    if (j == gamma) return wj;
    const double den = s + 1.0 - wj;
    return den > 0.0 ? s / den : 1.0;
}

/// The inverse-CDF draw over the winner's weights r[0..n) with sum s (spec_block_verify's tail).
STRATA_SBD_HD inline int32_t draw(const View& v, int win, const double* r, double s, double u) {
    const int n = v.pn[win] < kRowCap ? v.pn[win] : kRowCap;
    double cum = 0.0;
    int last = -1;
    int32_t tok = v.pid[win * kRowCap + (n > 0 ? n - 1 : 0)];
    for (int i = 0; i < n; ++i) {
        if (!(r[i] > 0.0)) continue;
        last = i;
        cum += r[i] / s;
        if (u < cum) return v.pid[win * kRowCap + i];
    }
    if (last >= 0) tok = v.pid[win * kRowCap + last];
    return tok;
}

/// A plain sample from row `row`'s P (no draft: the sampler's own tail, as spec_sample).
STRATA_SBD_HD inline int32_t sample_plain(const View& v, int row, double u) {
    const int n = v.pn[row] < kRowCap ? v.pn[row] : kRowCap;
    double cum = 0.0;
    int32_t pick = v.pid[row * kRowCap + (n > 0 ? n - 1 : 0)];
    for (int i = 0; i < n; ++i) {
        cum += v.pp[row * kRowCap + i];
        if (u < cum) { pick = v.pid[row * kRowCap + i]; break; }
    }
    return pick;
}

/// out[0..gamma] as the engine's loop `while (a < T - 1 && window[a + 1] == out[a]) ++a` reads it: out[t] = d[t] for t < win (kept),
/// out[win] = the drawn token, and for t > win a plain sample of row t.  The last matters when the drawn token equals the draft d[win]
/// (the residual of a row can hold its own draft): the loop then goes on to row win + 1, whose P is exactly the distribution after
/// d[0..win], so a fresh sample from it (the exact-match sampler's own draw, pos_t's plain stream) is a valid continuation.
/// `u_pick(t)` is that plain draw.  Returns the number of rows written (gamma + 1).
template <class UPick>
inline int fill_out(const View& v, const SpecBlockResult& r, UPick&& u_pick, int32_t* out) {
    for (int t = 0; t <= v.gamma; ++t)
        out[t] = t < r.accepted ? v.d[t] : t == r.accepted ? r.token : sample_plain(v, t, (double) u_pick(t));
    return v.gamma + 1;
}

/// The kernel's control flow on one thread, the same calls in the same order as spec_block_verify_kernel (whose lanes split only the
/// `for i` loops of residual_entry; the sums stay ordered on lane 0).  `w` has gamma + 1 slots, `r` kRowCap.
template <class Eta, class Resid>
inline SpecBlockResult run_serial(const View& v, double* w, double* r, Eta&& eta, Resid&& resid) {
    const int gamma = v.gamma > kSpecBlockMaxGamma ? kSpecBlockMaxGamma : v.gamma;
    View vg = v;
    vg.gamma = gamma;
    w[0] = 1.0;
    for (int j = 0; j < gamma; ++j) w[j + 1] = ratio_next(vg, j, w[j]);
    int win = 0;
    double s = 0.0;
    for (int j = gamma; j >= 0; --j) {
        const int n = vg.pn[j] < kRowCap ? vg.pn[j] : kRowCap;
        for (int i = 0; i < n; ++i) r[i] = residual_entry(vg, j, w[j], i);
        s = 0.0;
        for (int i = 0; i < n; ++i) s += r[i];
        if (j == 0 || (double) eta(j) < h_of(gamma, j, w[j], s)) { win = j; break; }
    }
    const int n = vg.pn[win] < kRowCap ? vg.pn[win] : kRowCap;
    if (!(s > 0.0)) {
        s = 0.0;
        for (int i = 0; i < n; ++i) { r[i] = vg.pp[win * kRowCap + i]; s += r[i]; }
    }
    return {win, draw(vg, win, r, s, (double) resid(win))};
}

}  // namespace strata::core::spec_block_dev

namespace strata::core {

/// Where a verify window goes (verify.cpp).  The default (switch off) is always the token-by-token path, untouched; block
/// verification only for a sampled window with the drafter's lists and no proposal measurement (that path keeps its own kernel).
enum class VerifyRoute { TokenWise, Block };
inline VerifyRoute spec_verify_route(bool block_asked, bool sampled, bool measure, int T, int n_q, int max_t) {
    if (!block_asked || !sampled || measure) return VerifyRoute::TokenWise;
    if (T < 2 || T > max_t || T - 1 > kSpecBlockMaxGamma || n_q < 1) return VerifyRoute::TokenWise;
    return VerifyRoute::Block;
}

}  // namespace strata::core
