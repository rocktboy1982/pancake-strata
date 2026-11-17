// src/kernels/cuda/head_argmax_parity.cu - MOD_PLAN.md section 6 item 17 part 18 (STRATA_HEAD_ARGMAX_FUSED, LEADS.md
// "Item 18 fused argmax+accept"): the head-format GEMVs with the fused greedy pick (head_fmt.hpp *_head_mmvq_argmax +
// head_argmax_finish) against the kernels they stand in for, BITWISE:
//   1. the logits: the same bits as native_mmvq (and as the plain head_fmt GEMV), 1..8 columns, chunked like verify.cpp;
//   2. the picks: the same ids as argmax_rows over those logits (ties to the lowest row, NaN and -inf never picked,
//      0 when nothing is above -inf), on heads built to hit each case: duplicated rows (exact ties at the maximum),
//      rows with a NaN / +inf / -inf scale, rows of +0.0 and -0.0 only, a column with nothing above -inf;
//   3. the keys are back to zero after every finish (one key buffer serves every window, as in the engine).
// Exit 0 = pass, 1 = a difference, 77 = no GPU.
#include "strata/kernels/head_fmt.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

struct Fmt {
    int t, bb, d_at;
    const char* name;
    bool (*gemv)(const void*, const void*, float*, int, int, int, void*);
    bool (*gemv_am)(const void*, const void*, float*, int, int, int, unsigned long long*, void*);
};

// the scale half of super-block b of row r set to h (an IQ4_XS / Q6_K / Q4_K block's d)
void set_d(std::vector<uint8_t>& w, const Fmt& f, size_t row_bytes, int r, uint16_t h) {
    for (size_t b = 0; b < row_bytes / (size_t) f.bb; ++b) std::memcpy(&w[(size_t) r * row_bytes + b * f.bb + f.d_at], &h, 2);
}

int run_case(const Fmt& f, int n_in, int n_out, int mode, std::mt19937& rng, cudaStream_t st, unsigned long long* keys) {
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t row_bytes = (size_t) (n_in / 256) * f.bb, w_bytes = row_bytes * n_out;
    std::vector<uint8_t> w(w_bytes);
    for (auto& b : w) b = uint8_t(rng());
    for (size_t b = 0; b < w_bytes / f.bb; ++b) {
        const __half d = __float2half(0.004f * (1.0f + 0.3f * std::abs(nd(rng))));
        std::memcpy(&w[b * f.bb + f.d_at], &d, 2);
        if (f.t == 12) {   // Q4_K: dmin follows d (half2 dm)
            const __half m = __float2half(0.002f * std::abs(nd(rng)));
            std::memcpy(&w[b * f.bb + 2], &m, 2);
        }
    }
    if (mode == 1) {   // exact ties: copies of a few rows further down (the copy must lose to the lower original)
        for (int k = 0; k < 64; ++k) {
            const int a = (int) (rng() % (uint32_t) n_out), b = (int) (rng() % (uint32_t) n_out);
            std::memcpy(&w[(size_t) std::max(a, b) * row_bytes], &w[(size_t) std::min(a, b) * row_bytes], row_bytes);
        }
    } else if (mode == 2) {   // non-finite rows: NaN, +inf and -inf scales
        for (int k = 0; k < 48; ++k) {
            const uint16_t h = k % 3 == 0 ? 0x7E00 : k % 3 == 1 ? 0x7C00 : 0xFC00;
            set_d(w, f, row_bytes, (int) (rng() % (uint32_t) n_out), h);
        }
    } else if (mode == 3) {   // only zeros (+0.0 and -0.0 scales), row 0 NaN: the pick is row 1
        for (int r = 0; r < n_out; ++r) set_d(w, f, row_bytes, r, r == 0 ? 0x7E00 : (r & 1) ? 0x0000 : 0x8000);
        if (f.t == 12)   // Q4_K: also zero dmin, or the -mn term is not zero
            for (size_t b = 0; b < w_bytes / f.bb; ++b) { const uint16_t z = 0; std::memcpy(&w[b * f.bb + 2], &z, 2); }
    } else if (mode == 4) {   // nothing above -inf: every scale NaN -> the pick is 0
        for (int r = 0; r < n_out; ++r) set_d(w, f, row_bytes, r, 0x7E00);
    }
    void* dw;
    ck(cudaMalloc(&dw, w_bytes + 256), "w");
    ck(cudaMemcpy(dw, w.data(), w_bytes, cudaMemcpyHostToDevice), "upload");
    int bad = 0;
    // the build's verify window caps the column count (native_mmvq.cu MAX_NCOLS = max(8, STRATA_VERIFY_MAX_T)): more columns throw
    constexpr int kMaxCols = (STRATA_VERIFY_MAX_T > 8 ? STRATA_VERIFY_MAX_T : 8) < 12 ? (STRATA_VERIFY_MAX_T > 8 ? STRATA_VERIFY_MAX_T : 8) : 12;
    for (int nc = 1; nc <= kMaxCols; ++nc) {
        std::vector<float> x((size_t) n_in * nc);
        for (auto& v : x) v = nd(rng);
        float *dx, *ya, *yb, *yc;
        void* xq;
        int32_t *ida, *idb;
        uint8_t* scratch;
        ck(cudaMalloc(&dx, x.size() * 4), "x");
        ck(cudaMalloc(&xq, strata::kernels::native_q8_1_bytes(n_in, nc)), "xq");
        for (float** p : {&ya, &yb, &yc}) ck(cudaMalloc(p, (size_t) n_out * nc * 4), "y");
        ck(cudaMalloc(&ida, 64 * 4), "ida");
        ck(cudaMalloc(&idb, 64 * 4), "idb");
        ck(cudaMalloc(&scratch, strata::kernels::argmax_rows_scratch_bytes(nc)), "scratch");
        ck(cudaMemset(scratch, 0, strata::kernels::argmax_rows_scratch_bytes(nc)), "memset");
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x up");
        strata::kernels::native_quantize_q8_1(dx, xq, n_in, nc, st);
        strata::kernels::native_mmvq(f.t, dw, xq, ya, n_in, n_out, nc, st);
        // verify.cpp's chunks: at most 8 columns a launch, the keys of chunk h0 at keys + h0
        const size_t col_blocks = strata::kernels::native_q8_1_bytes(n_in, 1);
        for (int h0 = 0; h0 < nc; h0 += 8) {
            const int hc = std::min(8, nc - h0);
            const uint8_t* xh = static_cast<const uint8_t*>(xq) + (size_t) h0 * col_blocks;
            if (!f.gemv(dw, xh, yb + (size_t) h0 * n_out, n_in, n_out, hc, st) ||
                !f.gemv_am(dw, xh, yc + (size_t) h0 * n_out, n_in, n_out, hc, keys + h0, st)) {
                std::printf("  %s: refused at %d columns\n", f.name, hc);
                return 1;
            }
        }
        strata::kernels::head_argmax_finish(keys, nc, idb, st);
        strata::kernels::argmax_rows(ya, nc, n_out, scratch, ida, st);
        ck(cudaStreamSynchronize(st), "run");
        std::vector<float> a((size_t) n_out * nc), b(a.size()), c(a.size());
        std::vector<int32_t> pa(nc), pb(nc);
        std::vector<unsigned long long> kz(nc);
        cudaMemcpy(a.data(), ya, a.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), yb, b.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(c.data(), yc, c.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(pa.data(), ida, nc * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(pb.data(), idb, nc * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(kz.data(), keys, nc * 8, cudaMemcpyDeviceToHost);
        const bool same_y = std::memcmp(a.data(), c.data(), a.size() * 4) == 0 && std::memcmp(b.data(), c.data(), b.size() * 4) == 0;
        bool same_p = true, zero = true;
        for (int j = 0; j < nc; ++j) {
            same_p = same_p && pa[j] == pb[j];
            zero = zero && kz[j] == 0ull;
            if (mode == 3 && pb[j] != 1) same_p = false;
            if (mode == 4 && pb[j] != 0) same_p = false;
        }
        if (!(same_y && same_p && zero)) {
            std::printf("  %s mode %d cols %d: logits %s, picks %s (ref %d fused %d), keys %s\n", f.name, mode, nc,
                        same_y ? "same" : "DIFFERENT", same_p ? "same" : "DIFFERENT", pa[0], pb[0], zero ? "zero" : "NOT ZERO");
            ++bad;
        }
        cudaFree(dx); cudaFree(xq); cudaFree(ya); cudaFree(yb); cudaFree(yc); cudaFree(ida); cudaFree(idb); cudaFree(scratch);
    }
    cudaFree(dw);
    return bad;
}
}  // namespace

int main() {
    using namespace strata::kernels;
    int dev_n = 0;
    if (cudaGetDeviceCount(&dev_n) != cudaSuccess || dev_n < 1) { std::printf("no GPU\n"); return 77; }
    const Fmt fmts[] = {
        {23, 136, 0, "IQ4_XS", iq4xs_head_mmvq, iq4xs_head_mmvq_argmax},
        {14, 210, 208, "Q6_K", q6k_head_mmvq, q6k_head_mmvq_argmax},
        {12, 144, 0, "Q4_K", q4k_head_mmvq, q4k_head_mmvq_argmax},
    };
    std::mt19937 rng(7);
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    unsigned long long* keys;
    ck(cudaMalloc(&keys, 64 * 8), "keys");
    ck(cudaMemset(keys, 0, 64 * 8), "keys zero");
    int bad = 0;
    for (const Fmt& f : fmts) {
        for (int mode = 0; mode <= 4; ++mode) {
            // the daily head's width and a vocabulary that ends mid-block (n_out % 8 != 0)
            bad += run_case(f, 2560, mode == 0 ? 248320 : 40525, mode, rng, st, keys);
            bad += run_case(f, 2560, 4097, mode, rng, st, keys);
        }
        std::printf("== %s done\n", f.name);
    }
    std::printf(bad == 0 ? "head_argmax_parity: PASS\n" : "head_argmax_parity: FAIL (%d)\n", bad);
    return bad == 0 ? 0 : 1;
}
