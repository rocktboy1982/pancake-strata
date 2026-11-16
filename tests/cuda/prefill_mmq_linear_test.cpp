// prefill_mmq_linear_test - STRATA_MMQ_LINEAR (opt-in; llama.cpp PR #31047's linearized MoE tiles, host-side form):
// mmq::Context::run_linear (the group cut into runs of experts with similar row counts, one MMQ launch a run) against
// the one-launch Context::run, on one MMQ group of 16 experts with very uneven row counts (0, 1, a few, hundreds).
// The products are the same per output; MMQ may pick another tile width per launch (and stream-k splits on some NVIDIA
// cards), so the verdict is a relative bound of 1e-5 on every output, and the line says whether the bits are equal.
// Q8_0 down [2560 x 640] and IQ2_XS gate/up [1280 x 2560] (the daily pack's experts).  GPU, synthetic, no model.
#include "strata/prefill/moe_mmq.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace mmq = strata::prefill::mmq;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

// random blocks with a small positive f16 d at the head of each block (Q8_0, IQ2_XS)
std::vector<uint8_t> weights(ggml_type t, int64_t rows, int64_t cols, int experts, std::mt19937& rng) {
    const size_t per = mmq::matrix_bytes(t, rows, cols);
    std::vector<uint8_t> w((size_t) experts * per + 4096);
    std::uniform_int_distribution<int> byte(0, 255);
    for (auto& b : w) b = (uint8_t) byte(rng);
    const size_t bs = ggml_type_size(t), nblk = (size_t) experts * per / bs;
    for (size_t i = 0; i < nblk; ++i) {
        const uint16_t d = 0x2000 + (uint16_t) byte(rng);
        std::memcpy(w.data() + i * bs, &d, 2);
    }
    return w;
}

struct Case {
    const char* name;
    ggml_type t;
    int64_t rows, cols;
};

int run_case(const Case& c, std::mt19937& rng) {
    if (!mmq::supported(c.t) || !mmq::fits(c.t, c.rows)) {
        std::printf("  %-16s not covered by MMQ here (skip)\n", c.name);
        return 0;
    }
    constexpr int E = 16;
    const int32_t rows_e[E] = {0, 1, 3, 200, 7, 0, 64, 65, 2, 300, 1, 33, 31, 0, 120, 5};
    std::vector<int32_t> bounds(E + 1, 0);
    for (int e = 0; e < E; ++e) bounds[e + 1] = bounds[e] + rows_e[e];
    const int T = bounds[E];
    int64_t maxr = 0;
    for (int e = 0; e < E; ++e) maxr = std::max<int64_t>(maxr, rows_e[e]);
    const size_t per = mmq::matrix_bytes(c.t, c.rows, c.cols);
    std::vector<uint8_t> w = weights(c.t, c.rows, c.cols, E, rng);
    std::vector<float> x((size_t) T * c.cols);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (auto& v : x) v = nd(rng);

    uint8_t* dw; float *dx, *dd; void* dq; int32_t *dids, *db;
    ck(cudaMalloc(&dw, w.size()), "w"); ck(cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice), "w");
    ck(cudaMalloc(&dx, x.size() * 4), "x"); ck(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice), "x");
    ck(cudaMalloc(&dq, mmq::q8_bytes(T, c.cols)), "xq");
    ck(cudaMalloc(&dd, (size_t) T * c.rows * 4), "dst");
    ck(cudaMalloc(&dids, (size_t) T * 4), "ids");
    ck(cudaMalloc(&db, (E + 1) * 4), "bounds");
    ck(cudaMemcpy(db, bounds.data(), (E + 1) * 4, cudaMemcpyHostToDevice), "bounds");
    mmq::iota(dids, T, nullptr);
    mmq::quantize(dx, nullptr, dq, c.t, c.cols, c.cols, T, nullptr);
    mmq::Context ctx;
    mmq::Product p;
    p.w = dw; p.type = c.t; p.w_rows = c.rows; p.w_cols = c.cols; p.expert_bytes = per; p.n = E;
    p.xq = dq; p.bounds = db; p.ids = dids; p.total_rows = T; p.max_rows = maxr; p.dst = dd; p.ld_dst = c.rows;
    std::vector<float> out[2];
    for (int k = 0; k < 2; ++k) {
        ck(cudaMemset(dd, 0, (size_t) T * c.rows * 4), "clear");
        if (k == 0) ctx.run(p, nullptr);   // host_rows unset: one launch whatever the switch says
        else {
            mmq::Product q = p;
            q.host_rows = rows_e;
            ctx.run_linear(q, nullptr);
        }
        ck(cudaDeviceSynchronize(), "run");
        out[k].resize((size_t) T * c.rows);
        ck(cudaMemcpy(out[k].data(), dd, out[k].size() * 4, cudaMemcpyDeviceToHost), "d2h");
    }
    double mref = 0.0, md = 0.0;
    bool finite = true;
    for (size_t i = 0; i < out[0].size(); ++i) {
        finite = finite && std::isfinite(out[1][i]);
        mref = std::fmax(mref, std::fabs((double) out[0][i]));
        md = std::fmax(md, std::fabs((double) out[1][i] - out[0][i]));
    }
    const double rel = md / (mref > 0 ? mref : 1.0);
    const bool same = std::memcmp(out[0].data(), out[1].data(), out[0].size() * 4) == 0;
    const bool ok = finite && rel < 1e-5;
    std::printf("  %-16s %d rows: max relative difference %.3g, %s  %s\n", c.name, T, rel,
                same ? "bitwise equal" : "not bitwise", ok ? "ok" : "FAIL");
    cudaFree(dw); cudaFree(dx); cudaFree(dq); cudaFree(dd); cudaFree(dids); cudaFree(db);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    std::mt19937 rng(31047);
    const Case cases[] = {{"q8_0 down", GGML_TYPE_Q8_0, 2560, 640}, {"iq2_xs gate/up", GGML_TYPE_IQ2_XS, 1280, 2560}};
    int bad = 0;
    for (const Case& c : cases) bad += run_case(c, rng);
    std::printf("%s: STRATA_MMQ_LINEAR split launches against the one launch\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
