// src/kernels/device_accept_parity.cpp - MOD_PLAN.md section 6 item 17, part 17 (STRATA_DEVICE_ACCEPT, LEADS.md "Item 17
// commit into window graph, device-side accept"): device_accept_commit against the host rule it stands in for - the decode
// loops' greedy prefix accept (generate.cpp) followed, when bound, by serve_stop_bound (serve_stop_bound.hpp) - and
// Verifier::commit's record [n_keep, n_keep - 1, pos0 .. pos0 + n_keep - 1, -1 ...].  Random windows of 2..16 rows with
// planted agreement runs, EOS picks at every position, budgets from -1 to past the window; `out` and `commit` in mapped host
// memory as in the engine.  Exit 0 = pass, 1 = a difference, 77 = no GPU.
#include "strata/core/serve_stop_bound.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {
void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
}  // namespace

int main() {
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) { std::printf("no GPU\n"); return 77; }
    constexpr int kMaxT = 16;
    int32_t *h_out = nullptr, *m_out = nullptr, *h_commit = nullptr, *m_commit = nullptr, *d_tok = nullptr;
    check(cudaHostAlloc((void**) &h_out, kMaxT * 4, cudaHostAllocMapped), "out");
    check(cudaHostGetDevicePointer((void**) &m_out, h_out, 0), "out ptr");
    check(cudaHostAlloc((void**) &h_commit, (2 + kMaxT) * 4 + 16, cudaHostAllocMapped), "commit");
    check(cudaHostGetDevicePointer((void**) &m_commit, h_commit, 0), "commit ptr");
    check(cudaMalloc((void**) &d_tok, kMaxT * 4), "tok");
    cudaStream_t s;
    check(cudaStreamCreate(&s), "stream");
    std::mt19937 rng(17);
    const int64_t eos[2] = {248044, 248046};
    int bad = 0, cases = 0;
    for (int it = 0; it < 20000; ++it) {
        const int T = 2 + (int) (rng() % (kMaxT - 1));
        const int max_t = kMaxT;
        std::vector<int32_t> tok(T), out(T);
        for (int t = 0; t < T; ++t) tok[t] = (int32_t) (rng() % 64);
        // plant a run of agreeing drafts of random length, then random picks (some agree by chance: small alphabet)
        const int run = (int) (rng() % T);
        for (int t = 0; t < T; ++t) out[t] = t < run && t + 1 < T ? tok[t + 1] : (int32_t) (rng() % 64);
        if (rng() % 3 == 0) out[rng() % T] = (int32_t) eos[rng() % 2];   // an EOS pick somewhere
        if (rng() % 5 == 0 && run > 0) tok[run] = out[run - 1] = (int32_t) eos[0];   // an accepted EOS draft
        strata::kernels::DeviceAcceptBound b{};
        b.on = (int) (rng() % 2);
        b.remaining = (int) (rng() % (T + 3)) - 1;
        b.n_eos = 2;
        b.eos[0] = (int32_t) eos[0];
        b.eos[1] = (int32_t) eos[1];
        const int pos0 = (int) (rng() % 100000);
        // host reference
        int a = 0;
        while (a < T - 1 && tok[(size_t) a + 1] == out[(size_t) a]) ++a;
        int keep = a + 1;
        if (b.on) {
            keep = strata::core::serve_stop_bound(a + 1, (int64_t) b.remaining, out.data(), [&](int32_t y) {
                return std::find(std::begin(eos), std::end(eos), (int64_t) y) != std::end(eos);
            });
            if (keep < 1) keep = 1;
        }
        for (int t = 0; t < T; ++t) h_out[t] = out[(size_t) t];
        for (int t = 0; t < 2 + kMaxT; ++t) h_commit[t] = 0x5a5a5a5a;
        check(cudaMemcpy(d_tok, tok.data(), (size_t) T * 4, cudaMemcpyHostToDevice), "tok up");
        strata::kernels::device_accept_commit(d_tok, m_out, T, pos0, max_t, b, m_commit, s);
        check(cudaStreamSynchronize(s), "run");
        bool ok = ((volatile int32_t*) h_commit)[0] == keep && ((volatile int32_t*) h_commit)[1] == keep - 1;
        for (int t = 0; t < max_t; ++t) ok = ok && ((volatile int32_t*) h_commit)[2 + t] == (t < keep ? pos0 + t : -1);
        ++cases;
        if (!ok) {
            if (bad < 8)
                std::printf("  T %d run %d bound %d remaining %d: host keep %d, device %d\n", T, run, b.on, b.remaining, keep,
                            ((volatile int32_t*) h_commit)[0]);
            ++bad;
        }
    }
    std::printf(bad == 0 ? "device_accept_parity: PASS (%d windows)\n" : "device_accept_parity: FAIL (%d of %d)\n", bad == 0 ? cases : bad,
                cases);
    return bad == 0 ? 0 : 1;
}
