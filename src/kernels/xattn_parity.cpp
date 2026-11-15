// src/kernels/xattn_parity.cpp - STRATA_XATTN's kernels (include/strata/kernels/xattn.hpp): tau >= 1 keeps every causal block
// and must equal dense causal attention (FP64 reference); at tau 0.9 the kernel's block mask against the host rule (FP32 vs
// FP64 masses: a few boundary rows may differ) and the attention against FP64 over the kernel's own mask.  GPU, synthetic,
// no model; exit 77 where the kernels do not run (wave64).
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/xattn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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
template <typename T> T* dev(size_t n) {
    T* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) {
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}

}  // namespace

int main() {
    using namespace strata::kernels;
    std::mt19937 rng(20261116);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    cudaStream_t s = nullptr;
    check(cudaStreamCreate(&s), "stream");
    int bad = 0;
    struct Case { int H, HKV, HD; int64_t T, p0; int stride; };
    const Case cases[] = {{4, 2, 64, 257, 0, 8}, {4, 1, 256, 384, 300, 8}, {8, 2, 128, 512, 129, 16}, {2, 2, 64, 200, 1000, 4}};
    for (const Case& k : cases) {
        const int64_t P = k.p0 + k.T, nqb = (k.T + kXattnBlock - 1) / kXattnBlock, nkb = (P + kXattnBlock - 1) / kXattnBlock;
        // a shared direction in the queries and in two key blocks: a peaked distribution the selection can act on
        std::vector<float> u((size_t) k.HD);
        for (auto& x : u) x = nd(rng);
        std::vector<uint16_t> q16((size_t) k.T * k.H * k.HD), k16((size_t) k.HKV * P * k.HD), v16(k16.size());
        std::vector<float> q(q16.size()), kf(k16.size()), vf(v16.size());
        for (int64_t t = 0; t < k.T; ++t)
            for (int h = 0; h < k.H; ++h)
                for (int d = 0; d < k.HD; ++d) {
                    const size_t i = ((size_t) t * k.H + h) * k.HD + d;
                    q16[i] = f16_from_f32(0.5f * nd(rng) + 0.5f * u[(size_t) d]);
                    q[i] = f32_from_f16(q16[i]);
                }
        for (int g = 0; g < k.HKV; ++g)
            for (int64_t c = 0; c < P; ++c)
                for (int d = 0; d < k.HD; ++d) {
                    const size_t i = ((size_t) g * P + c) * k.HD + d;
                    const bool hot = c / kXattnBlock == 1 || c / kXattnBlock == nkb / 2;
                    k16[i] = f16_from_f32(0.5f * nd(rng) + (hot ? 0.6f * u[(size_t) d] : 0.0f));
                    v16[i] = f16_from_f32(nd(rng));
                    kf[i] = f32_from_f16(k16[i]);
                    vf[i] = f32_from_f16(v16[i]);
                }
        uint16_t* d_q = dev<uint16_t>(q16.size());
        uint16_t* d_k = dev<uint16_t>(k16.size());
        uint16_t* d_v = dev<uint16_t>(v16.size());
        float* d_o = dev<float>(q16.size());
        up(d_q, q16);
        up(d_k, k16);
        up(d_v, v16);
        for (const float tau : {1.5f, 0.9f}) {
            XattnConfig c;
            c.on = true;
            c.tau = tau;
            c.stride = k.stride;
            const size_t bytes = xattn_scratch_bytes(k.T, P, k.H, k.stride);
            uint8_t* d_scr = dev<uint8_t>(bytes);
            const uint8_t* d_mask = nullptr;
            if (!xattn_prefill_attn(d_q, d_k, d_v, k.T, k.p0, k.H, k.HKV, k.HD, c, d_scr, bytes, d_o, s, &d_mask)) {
                std::printf("xattn_parity: the kernels do not run on this build/device, skipped\n");
                return 77;
            }
            check(cudaStreamSynchronize(s), "xattn sync");
            const std::vector<float> got = down(d_o, q16.size());
            const std::vector<uint8_t> mask = down(d_mask, (size_t) nqb * k.H * nkb);
            // the mask against the host rule
            int rows_diff = 0, kept = 0, causal = 0;
            for (int h = 0; h < k.H; ++h)
                for (int qb = 0; qb < nqb; ++qb) {
                    std::vector<uint8_t> want;
                    xattn_select_host(q.data(), kf.data(), k.T, k.p0, k.H, k.HKV, k.HD, qb, h, c, want);
                    const uint8_t* m = mask.data() + ((size_t) h * nqb + qb) * nkb;
                    bool same = true;
                    for (int64_t kb = 0; kb < nkb; ++kb) same = same && (m[kb] != 0) == (want[(size_t) kb] != 0);
                    rows_diff += same ? 0 : 1;
                    const int64_t t_end = std::min<int64_t>(k.T, (int64_t) qb * kXattnBlock + kXattnBlock);
                    const int64_t nkq = (k.p0 + t_end - 1) / kXattnBlock + 1;
                    for (int64_t kb = 0; kb < nkq; ++kb) { kept += m[kb] ? 1 : 0; ++causal; }
                }
            const int rows = (int) (nqb * k.H);
            if (tau >= 1.0f ? (rows_diff != 0 || kept != causal) : rows_diff * 10 > rows) {
                std::fprintf(stderr, "xattn mask H %d HD %d T %lld p0 %lld tau %g: %d of %d rows differ from the host rule, %d of %d kept\n",
                             k.H, k.HD, (long long) k.T, (long long) k.p0, tau, rows_diff, rows, kept, causal);
                ++bad;
            }
            // the attention over the kernel's own mask (tau >= 1: every causal cell, dense attention)
            double worst = 0.0;
            std::vector<double> ref((size_t) k.HD);
            for (int64_t t = 0; t < k.T; ++t)
                for (int h = 0; h < k.H; ++h) {
                    const uint8_t* m = mask.data() + ((size_t) h * nqb + t / kXattnBlock) * nkb;
                    xattn_attn_host(q.data(), kf.data(), vf.data(), k.T, k.p0, k.H, k.HKV, k.HD, t, h, tau >= 1.0f ? nullptr : m,
                                    ref.data());
                    const float* o = got.data() + ((size_t) t * k.H + h) * k.HD;
                    for (int d = 0; d < k.HD; ++d) worst = std::max(worst, std::fabs((double) o[d] - ref[(size_t) d]));
                }
            if (!(worst < 2e-3)) {
                std::fprintf(stderr, "xattn attention H %d HD %d T %lld p0 %lld tau %g: worst abs error %.3g\n", k.H, k.HD,
                             (long long) k.T, (long long) k.p0, tau, worst);
                ++bad;
            }
            if (tau < 1.0f)
                std::printf("xattn H %d HD %d T %lld p0 %lld: tau %.2f keeps %d of %d causal blocks\n", k.H, k.HD, (long long) k.T,
                            (long long) k.p0, tau, kept, causal);
            cudaFree(d_scr);
        }
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
    }
    (void) cudaStreamDestroy(s);
    std::printf("xattn_parity: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
