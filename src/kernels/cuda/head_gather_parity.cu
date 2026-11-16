// src/kernels/cuda/head_gather_parity.cu - MOD_PLAN.md section 6 item 38 (STRATA_HEAD_SHORTLIST_GATHER): the gathered IQ4_XS head
// (head_fmt.hpp iq4xs_head_gather) against the full head GEMV (iq4xs_head_mmvq) followed by the shortlist mask, BITWISE:
//   every logit of a column that keeps the row's cluster is the full head's float; every other logit is -inf.  Random heads,
//   random cluster tables (also cluster-contiguous ones, so whole blocks of rows fall outside the union), random keep flags per
//   column (including a column that keeps nothing and a chunk that keeps nothing at all), 1..8 columns, a vocabulary that ends
//   mid-block.  The keep kernel (head_shortlist_keep) is checked against shortlist_select on the host for the same scores.
// NOT RUN on the box when written (no toolchain here).  Exit 0 = pass, 1 = a difference, 77 = no GPU.
#include "strata/kernels/head_fmt.hpp"
#include "strata/kernels/head_shortlist.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/program/head_shortlist.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

int run_case(int n_in, int n_out, int k, int mode, std::mt19937& rng, cudaStream_t st) {
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t row_bytes = (size_t) (n_in / 256) * 136, w_bytes = row_bytes * n_out;
    std::vector<uint8_t> w(w_bytes);
    for (auto& b : w) b = uint8_t(rng());
    for (size_t b = 0; b < w_bytes / 136; ++b) {
        const __half d = __float2half(0.004f * (1.0f + 0.3f * std::abs(nd(rng))));
        std::memcpy(&w[b * 136], &d, 2);
    }
    // mode 1: cluster-contiguous rows (whole blocks of rows fall outside the union)
    std::vector<int32_t> cluster((size_t) n_out);
    for (int v = 0; v < n_out; ++v)
        cluster[(size_t) v] = mode == 1 ? (int32_t) ((int64_t) v * k / n_out) : (int32_t) (rng() % (unsigned) k);
    void* dw;
    int32_t* dcl;
    ck(cudaMalloc(&dw, w_bytes + 256), "w");
    ck(cudaMalloc((void**) &dcl, cluster.size() * 4), "cluster");
    ck(cudaMemcpy(dw, w.data(), w_bytes, cudaMemcpyHostToDevice), "upload");
    ck(cudaMemcpy(dcl, cluster.data(), cluster.size() * 4, cudaMemcpyHostToDevice), "upload cluster");
    int bad = 0;
    for (int nc = 1; nc <= 8; ++nc) {
        std::vector<float> x((size_t) n_in * nc);
        for (auto& v : x) v = nd(rng);
        // mode 2: nothing kept anywhere; mode 3: column 0 keeps nothing
        std::vector<unsigned char> keep((size_t) nc * k);
        for (int j = 0; j < nc; ++j)
            for (int c = 0; c < k; ++c)
                keep[(size_t) j * k + c] = mode == 2 ? 0 : (rng() % 100 < (j == 0 && mode == 3 ? 0u : 20u) ? 1 : 0);
        float *dx, *yfull, *ygat;
        void* xq;
        unsigned char* dkeep;
        ck(cudaMalloc((void**) &dx, x.size() * 4), "x");
        ck(cudaMalloc(&xq, strata::kernels::native_q8_1_bytes(n_in, nc)), "xq");
        ck(cudaMalloc((void**) &yfull, (size_t) n_out * nc * 4), "y");
        ck(cudaMalloc((void**) &ygat, (size_t) n_out * nc * 4), "y");
        ck(cudaMalloc((void**) &dkeep, keep.size()), "keep");
        ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x up");
        ck(cudaMemcpy(dkeep, keep.data(), keep.size(), cudaMemcpyHostToDevice), "keep up");
        strata::kernels::native_quantize_q8_1(dx, xq, n_in, nc, st);
        if (!strata::kernels::iq4xs_head_mmvq(dw, xq, yfull, n_in, n_out, nc, st) ||
            !strata::kernels::iq4xs_head_gather(dw, xq, ygat, n_in, n_out, nc, dcl, dkeep, k, st)) {
            std::printf("  refused at %d columns\n", nc);
            return 1;
        }
        ck(cudaStreamSynchronize(st), "run");
        std::vector<float> a((size_t) n_out * nc), b(a.size());
        ck(cudaMemcpy(a.data(), yfull, a.size() * 4, cudaMemcpyDeviceToHost), "a");
        ck(cudaMemcpy(b.data(), ygat, b.size() * 4, cudaMemcpyDeviceToHost), "b");
        const float ninf = -std::numeric_limits<float>::infinity();
        bool same = true;
        for (int j = 0; j < nc && same; ++j)
            for (int r = 0; r < n_out; ++r) {
                const float want = keep[(size_t) j * k + (size_t) cluster[(size_t) r]] ? a[(size_t) j * n_out + r] : ninf;
                if (std::memcmp(&want, &b[(size_t) j * n_out + r], 4) != 0) { same = false; break; }
            }
        if (!same) {
            std::printf("  mode %d cols %d: gathered logits DIFFERENT from the full head + mask\n", mode, nc);
            ++bad;
        }
        cudaFree(dx); cudaFree(xq); cudaFree(yfull); cudaFree(ygat); cudaFree(dkeep);
    }
    cudaFree(dw);
    cudaFree(dcl);
    return bad;
}

// the keep kernel against shortlist_select on the host, on the same scores (random centres and hidden rows)
int keep_case(int n_embd, int k, int n_keep, int nc, std::mt19937& rng, cudaStream_t st) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> hid((size_t) nc * n_embd), cen((size_t) k * n_embd);
    for (auto& v : hid) v = nd(rng);
    for (auto& v : cen) v = nd(rng);
    float *dh, *dc, *dsc;
    unsigned char* dk;
    ck(cudaMalloc((void**) &dh, hid.size() * 4), "h");
    ck(cudaMalloc((void**) &dc, cen.size() * 4), "c");
    ck(cudaMalloc((void**) &dsc, strata::kernels::head_shortlist_scratch_bytes(nc, k)), "scratch");
    ck(cudaMalloc((void**) &dk, (size_t) nc * k), "keep");
    ck(cudaMemcpy(dh, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice), "h up");
    ck(cudaMemcpy(dc, cen.data(), cen.size() * 4, cudaMemcpyHostToDevice), "c up");
    strata::kernels::head_shortlist_keep(dh, dc, n_embd, k, n_keep, nc, dsc, dk, st);
    ck(cudaStreamSynchronize(st), "keep run");
    std::vector<unsigned char> got((size_t) nc * k);
    ck(cudaMemcpy(got.data(), dk, got.size(), cudaMemcpyDeviceToHost), "keep down");
    std::vector<float> sc((size_t) nc * k);
    ck(cudaMemcpy(sc.data(), dsc, sc.size() * 4, cudaMemcpyDeviceToHost), "scores down");   // the device's own scores
    int bad = 0;
    for (int j = 0; j < nc; ++j) {
        const std::vector<uint8_t> want = strata::program::shortlist_select(sc.data() + (size_t) j * k, k, n_keep);
        for (int c = 0; c < k; ++c)
            if (want[(size_t) c] != got[(size_t) j * k + c]) { ++bad; break; }
    }
    cudaFree(dh); cudaFree(dc); cudaFree(dsc); cudaFree(dk);
    return bad;
}
}  // namespace

int main() {
    int dev_n = 0;
    if (cudaGetDeviceCount(&dev_n) != cudaSuccess || dev_n < 1) { std::printf("no GPU\n"); return 77; }
    std::mt19937 rng(11);
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    int bad = 0;
    for (int mode = 0; mode <= 3; ++mode) {
        bad += run_case(2560, mode == 0 ? 248320 : 40525, 1024, mode, rng, st);
        bad += run_case(2560, 4097, 64, mode, rng, st);
    }
    for (int nc = 1; nc <= 8; nc += 3) bad += keep_case(2560, 1024, 134, nc, rng, st) + keep_case(2560, 64, 8, nc, rng, st);
    std::printf(bad == 0 ? "head_gather_parity: PASS\n" : "head_gather_parity: FAIL (%d)\n", bad);
    return bad == 0 ? 0 : 1;
}
