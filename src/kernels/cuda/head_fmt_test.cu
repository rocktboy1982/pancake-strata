// src/kernels/cuda/head_fmt_test.cu - pancake-strata: the per-format head kernels (head_fmt.cu) against native_mmvq,
// for every format the Flash-Next files' heads use here (IQ4_XS, Q6_K), at 2560 x 248,320:
//   1. the head GEMV, bit for bit against native_mmvq and native_mmvq_il, 1..8 columns, and the times;
//   2. the 2-bit proxy's pick against the exact argmax on peaked heads (an activation = a row plus noise), and its time
//      against the GEMV the draft step runs now (the 40,525-row English draft head).
#include "strata/kernels/head_fmt.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
float elapsed(cudaEvent_t a, cudaEvent_t b) { float ms = 0; cudaEventElapsedTime(&ms, a, b); return ms; }
}  // namespace

int main() {
    using namespace strata::kernels;
    struct Fmt {
        int t, bb, d_at;
        const char* name;
        bool (*gemv)(const void*, const void*, float*, int, int, int, void*);
        size_t (*pbytes)(int64_t);
        size_t (*sbytes)(int64_t);
        void (*pack)(const void*, int64_t, int64_t, void*, int64_t, void*);
        bool (*top1)(const void*, const void*, const void*, int, int64_t, const int32_t*, int*, float*, void*, void*);
    };
    const Fmt fmts[] = {
        {23, 136, 0, "IQ4_XS (ISTA/Swift IQ2_XS)", iq4xs_head_mmvq, iq4xs_proxy_bytes, iq4xs_proxy_scratch_bytes,
         iq4xs_proxy_pack, iq4xs_proxy_top1},
        {14, 210, 208, "Q6_K (ISTA/Swift IQ3_S, Coder IQ1_M, Unsloth UD-IQ4_XS)", q6k_head_mmvq, q6k_proxy_bytes,
         q6k_proxy_scratch_bytes, q6k_proxy_pack, q6k_proxy_top1},
        {12, 144, 0, "Q4_K (Unsloth UD-IQ1_M)", q4k_head_mmvq, q4k_proxy_bytes, q4k_proxy_scratch_bytes, q4k_proxy_pack,
         q4k_proxy_top1},
    };
    const int n_in = 2560, n_head = 248320, n_draft = 40525;
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    int bad = 0;
    for (const Fmt& f : fmts) {
        std::printf("== %s\n", f.name);
        const size_t row_bytes = (size_t) (n_in / 256) * f.bb, w_bytes = row_bytes * n_head;
        std::vector<uint8_t> w(w_bytes);
        for (auto& b : w) b = uint8_t(rng());
        for (size_t b = 0; b < w_bytes / f.bb; ++b) {
            const __half d = __float2half(0.004f * (1.0f + 0.3f * std::abs(nd(rng))));
            std::memcpy(&w[b * f.bb + f.d_at], &d, 2);
            if (f.t == 12) {   // Q4_K: dmin follows d (half2 dm); a random one can be inf or NaN
                const __half m = __float2half(0.002f * std::abs(nd(rng)));
                std::memcpy(&w[b * f.bb + 2], &m, 2);
            }
        }
        void* dw;
        ck(cudaMalloc(&dw, w_bytes + 256), "w");
        ck(cudaMemcpy(dw, w.data(), w_bytes, cudaMemcpyHostToDevice), "upload");
        // 1. the GEMV
        for (int nc = 1; nc <= 8; ++nc) {
            std::vector<float> x((size_t) n_in * nc);
            for (auto& v : x) v = nd(rng);
            float *dx, *ya, *yb, *yc;
            void *xq, *xil;
            ck(cudaMalloc(&dx, x.size() * 4), "x");
            ck(cudaMalloc(&xq, native_q8_1_bytes(n_in, nc)), "xq");
            ck(cudaMalloc(&xil, native_q8_1_il_bytes(n_in, nc)), "xil");
            for (float** p : {&ya, &yb, &yc}) ck(cudaMalloc(p, (size_t) n_head * nc * 4), "y");
            cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
            native_quantize_q8_1(dx, xq, n_in, nc, st);
            const bool il = native_mmvq_il_supported(f.t, nc, n_head);
            if (il && nc >= 2) native_q8_1_interleave(xq, xil, n_in, nc, st);
            float t[3] = {1e30f, 1e30f, 1e30f};
            for (int rep = 0; rep < 5; ++rep)
                for (int k = 0; k < 3; ++k) {
                    if (k == 1 && !il) continue;
                    cudaEvent_t e0, e1;
                    cudaEventCreate(&e0); cudaEventCreate(&e1);
                    cudaEventRecord(e0, st);
                    if (k == 0) native_mmvq(f.t, dw, xq, ya, n_in, n_head, nc, st);
                    else if (k == 1) native_mmvq_il(f.t, dw, xq, xil, yb, n_in, n_head, nc, st);
                    else if (!f.gemv(dw, xq, yc, n_in, n_head, nc, st)) { std::printf("refused\n"); return 1; }
                    cudaEventRecord(e1, st);
                    ck(cudaStreamSynchronize(st), "run");
                    t[k] = std::min(t[k], elapsed(e0, e1));
                    cudaEventDestroy(e0); cudaEventDestroy(e1);
                }
            std::vector<float> a((size_t) n_head * nc), b(a.size()), c(a.size());
            cudaMemcpy(a.data(), ya, a.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(c.data(), yc, c.size() * 4, cudaMemcpyDeviceToHost);
            bool same = std::memcmp(a.data(), c.data(), a.size() * 4) == 0, same_il = true;
            if (il) {
                cudaMemcpy(b.data(), yb, b.size() * 4, cudaMemcpyDeviceToHost);
                same_il = std::memcmp(b.data(), c.data(), b.size() * 4) == 0;
            }
            const float best_old = il && nc >= 2 ? std::min(t[0], t[1]) : t[0];
            std::printf("  GEMV %d col: native %.3f ms, il %s, new %.3f ms (x%.2f vs the best) - %s\n", nc, t[0],
                        il ? (std::to_string(t[1]).substr(0, 5) + " ms").c_str() : "-", t[2], best_old / t[2],
                        same && same_il ? "the same bits" : "DIFFERENT");
            bad += !(same && same_il);
            cudaFree(dx); cudaFree(xq); cudaFree(xil); cudaFree(ya); cudaFree(yb); cudaFree(yc);
        }
        // 2. the proxy on the draft head's size
        void *proxy, *scratch, *xq;
        float *dx, *y;
        int32_t* map;
        int* oid;
        float* op;
        ck(cudaMalloc(&proxy, f.pbytes(n_draft)), "proxy");
        ck(cudaMalloc(&scratch, f.sbytes(n_draft)), "scratch");
        cudaMemset(scratch, 0, f.sbytes(n_draft));
        ck(cudaMalloc(&dx, n_in * 4), "x");
        ck(cudaMalloc(&xq, native_q8_1_bytes(n_in, 1)), "xq");
        ck(cudaMalloc(&y, (size_t) n_draft * 4), "y");
        ck(cudaMalloc(&map, (size_t) n_draft * 4), "map");
        ck(cudaMalloc(&oid, 4), "id");
        ck(cudaMalloc(&op, 4), "p");
        std::vector<int32_t> ident(n_draft);
        std::iota(ident.begin(), ident.end(), 0);
        cudaMemcpy(map, ident.data(), ident.size() * 4, cudaMemcpyHostToDevice);
        f.pack(dw, 0, n_draft, proxy, n_draft, st);
        int agree = 0;
        const int trials = 200;
        float t_old = 0, t_new = 0;
        std::vector<float> hy(n_draft), x(n_in);
        for (int k = 0; k < trials; ++k) {
            const int target = (int) (rng() % (unsigned) n_draft);
            dequant_f32(f.t, dw, target, 1, n_in, dx, st);
            cudaStreamSynchronize(st);
            cudaMemcpy(x.data(), dx, n_in * 4, cudaMemcpyDeviceToHost);
            float nrm = 0.f;
            for (float v : x) nrm += v * v;
            nrm = std::sqrt(nrm / n_in) + 1e-9f;
            for (auto& v : x) v = v / nrm + 1.5f * nd(rng);
            cudaMemcpy(dx, x.data(), n_in * 4, cudaMemcpyHostToDevice);
            native_quantize_q8_1(dx, xq, n_in, 1, st);
            cudaEvent_t e0, e1, e2;
            cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
            cudaEventRecord(e0, st);
            native_mmvq(f.t, dw, xq, y, n_in, n_draft, 1, st);
            cudaEventRecord(e1, st);
            f.top1(proxy, dw, xq, n_draft, n_draft, map, oid, op, scratch, st);
            cudaEventRecord(e2, st);
            ck(cudaStreamSynchronize(st), "run");
            if (k) { t_old += elapsed(e0, e1); t_new += elapsed(e1, e2); }
            cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
            cudaMemcpy(hy.data(), y, hy.size() * 4, cudaMemcpyDeviceToHost);
            int best = 0;
            for (int r = 1; r < n_draft; ++r) if (hy[r] > hy[best]) best = r;
            int id = -1;
            cudaMemcpy(&id, oid, 4, cudaMemcpyDeviceToHost);
            agree += id == best;
        }
        std::printf("  draft pick (%d rows): two-stage = exact argmax %d/%d; %.3f ms vs the GEMV alone %.3f ms (x%.2f)\n",
                    n_draft, agree, trials, t_new / (trials - 1), t_old / (trials - 1), t_old / t_new);
        bad += agree < trials * 99 / 100;
        cudaFree(proxy); cudaFree(scratch); cudaFree(dx); cudaFree(xq); cudaFree(y); cudaFree(map); cudaFree(oid);
        cudaFree(op); cudaFree(dw);
    }
    return bad ? 1 : 0;
}
