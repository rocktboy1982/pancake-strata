// src/kernels/rdna_gr_parity.cpp - the RDNA3 hyper-connection read (rdna_gr.cu, STRATA_GR_RDNA=1) against the default
// multi read (fused_gr_read_multi).  MOD_PLAN item 93.
//
// Real geometry (n_embd 2560, hc 4, hc_lr 320), every T = 1..8, the pending write applied and the inject rows on, as
// the verifier calls it.  The RDNA read sums in another order, so it is bounded, not bitwise: every output (mixed, the
// inject outputs, the written R) must be finite and within a tolerance of the default read relative to its largest
// magnitude, and two runs of the RDNA read must agree bit for bit.
//
//   (no argument)  the BF16 read (w_down / w_up); tolerance 1e-4
//   --q8           STRATA_GR_RDNA_Q8: int8 codes + fp16 scales (rdna_gr.cu layout).  The scales are powers of two, so
//                  the BF16 image the reference reads (q * s) is exact and the 1e-4 bound still holds.  With
//                  STRATA_GR_RDNA_A8=1 the activations are int8 too: the bound is --tol (ctest passes 3e-2).
//   --tol X        override the tolerance
//
// The kernel variants are chosen by the environment (read once per process): STRATA_GR_RDNA_RW (1, 2, 4),
// STRATA_GR_RDNA_A8, STRATA_GR_INJ_UP, STRATA_GR_FUSED; CMakeLists.txt registers one test per variant.
// Exit 77 (skip): no device, or a wave64 device (the RDNA kernels are wave32 only), or a CUDA / gfx906 build (the read is
// a stub there and writes nothing).  GPU, synthetic, no model.
#include "strata/kernels/fused_gr.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if defined(_WIN32)   // MSVC has no setenv / unsetenv
#define unsetenv(k) _putenv_s((k), "")
#endif

namespace k = strata::kernels;

namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, TM = 8;
void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}
uint16_t bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    return (uint16_t) ((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
uint16_t f16_pow2(int e) {   // fp16 bits of 2^e, e in [-14, 15]
    return (uint16_t) ((e + 15) << 10);
}
float pow2(int e) { return std::ldexp(1.0f, e); }
template <typename T>
T* up(const std::vector<T>& h) {
    void* p = nullptr;
    ck(cudaMalloc(&p, h.size() * sizeof(T)), "malloc");
    ck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
    return (T*) p;
}
template <typename T>
T* dev(size_t n) {
    void* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "malloc");
    return (T*) p;
}
// rdna_gr.cu's int8 layout: n codes, then (from offset n rounded up to 256) one fp16 scale per 32 contiguous codes.
// Fills `r8` and the exact BF16 image of q * s into `img`.
void make_r8(std::mt19937& rng, size_t n, int e_lo, int e_hi, std::vector<uint8_t>& r8, std::vector<uint16_t>& img) {
    const size_t soff = (n + 255) / 256 * 256;
    r8.assign(soff + n / 32 * 2, 0);
    img.resize(n);
    std::uniform_int_distribution<int> qd(-127, 127), ed(e_lo, e_hi);
    for (size_t b = 0; b < n / 32; ++b) {
        const int e = ed(rng);
        const uint16_t s = f16_pow2(e);
        std::memcpy(r8.data() + soff + b * 2, &s, 2);
        for (size_t i = b * 32; i < b * 32 + 32; ++i) {
            const int q = qd(rng);
            r8[i] = (uint8_t) (int8_t) q;
            img[i] = bf16((float) q * pow2(e));   // |q| <= 127 fits bf16's 8-bit significand: exact
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    bool q8 = false;
    double tol = 1e-4;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--q8") == 0) q8 = true;
        else if (std::strcmp(argv[i], "--tol") == 0 && i + 1 < argc) tol = std::atof(argv[++i]);
    }
    // the reference must be the default read: STRATA_GR_RDNA=1 would route fused_gr_read_multi to rdna_gr.cu itself
    unsetenv("STRATA_GR_RDNA");
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("SKIP: no GPU\n");
        return 77;
    }
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    if (prop.warpSize != 32) {
        std::printf("SKIP: wave%d device (the RDNA read is wave32 only)\n", prop.warpSize);
        return 77;
    }
#if !defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
    std::printf("SKIP: the RDNA read is not built here (CUDA or gfx906)\n");
    return 77;
#endif
    k::fused_gr_check();
    if (q8 && !k::rdna_r8_scratch_init((long long) LR * D)) {   // also the STRATA_GR_FUSED barrier words
        std::fprintf(stderr, "rdna_r8_scratch_init failed\n");
        return 2;
    }

    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> R((size_t) TM * D), bo((size_t) TM * N), inj((size_t) TM * HC), wn(D);
    for (auto& x : R) x = nd(rng);
    for (auto& x : bo) x = 0.5f * nd(rng);
    for (auto& x : inj) x = nd(rng);
    for (auto& x : wn) x = 1.f + 0.3f * nd(rng);
    std::vector<uint16_t> wd((size_t) LR * D), wu((size_t) D * LR), wi((size_t) HC * D);
    std::vector<uint8_t> r8d, r8u;
    if (q8) {
        make_r8(rng, wd.size(), -14, -12, r8d, wd);   // |w| up to ~0.03, like the BF16 case's 0.02 sigma
        make_r8(rng, wu.size(), -13, -11, r8u, wu);
    } else {
        for (auto& x : wd) x = bf16(0.02f * nd(rng));
        for (auto& x : wu) x = bf16(0.05f * nd(rng));
    }
    for (auto& x : wi) x = bf16(0.02f * nd(rng));
    float* dR = dev<float>(R.size());
    float* dbo = up(bo);
    float* dinj = up(inj);
    float* dwn = up(wn);
    uint16_t* dwd = up(wd);
    uint16_t* dwu = up(wu);
    uint16_t* dwi = up(wi);
    const int8_t* dr8d = q8 ? (const int8_t*) up(r8d) : nullptr;
    const int8_t* dr8u = q8 ? (const int8_t*) up(r8u) : nullptr;
    float* dxn = dev<float>((size_t) TM * D);   // the default read's xn scratch; the RDNA read needs far less
    float* dlo = dev<float>((size_t) TM * LR);
    float* drs = dev<float>((size_t) TM * HC);
    float* dio = dev<float>((size_t) TM * HC);
    float* dmix = dev<float>((size_t) TM * N);

    int bad = 0;
    for (int T = 1; T <= TM; ++T) {
        k::FusedGrArgs a[TM], ar[TM];
        for (int t = 0; t < T; ++t) {
            a[t] = k::FusedGrArgs{};
            a[t].R = dR + (size_t) t * D;
            a[t].R_out = dR + (size_t) t * D;
            a[t].apply = true;
            a[t].bo_prev = dbo + (size_t) t * N;
            a[t].inj_prev = dinj + (size_t) t * HC;
            a[t].w_norm = dwn;
            a[t].w_down = dwd;
            a[t].w_up = dwu;
            a[t].w_inject = dwi;
            a[t].eps = 1e-6f;
            a[t].lo = dlo + (size_t) t * LR;
            a[t].rs = drs + (size_t) t * HC;
            a[t].inject_out = dio + (size_t) t * HC;
            a[t].mixed = dmix + (size_t) t * N;
            ar[t] = a[t];
            ar[t].r8_down = dr8d;
            ar[t].r8_up = dr8u;
        }
        // 0: the default read, 1 and 2: the RDNA read twice; each output is mixed | inject | R
        std::vector<float> out[3];
        for (int w = 0; w < 3; ++w) {
            ck(cudaMemcpy(dR, R.data(), R.size() * 4, cudaMemcpyHostToDevice), "R");
            ck(cudaMemset(dmix, 0xff, (size_t) TM * N * 4), "mix");   // NaN: an output never written fails
            ck(cudaMemset(dio, 0xff, (size_t) TM * HC * 4), "io");
            ck(cudaDeviceSynchronize(), "sync");
            if (w == 0) k::fused_gr_read_multi(a, T, dxn, nullptr);
            else k::rdna_gr_read_multi(ar, T, dxn, nullptr);
            ck(cudaGetLastError(), "launch");
            ck(cudaDeviceSynchronize(), "read");
            out[w].resize((size_t) T * (N + HC + D));
            ck(cudaMemcpy(out[w].data(), dmix, (size_t) T * N * 4, cudaMemcpyDeviceToHost), "d2h mix");
            ck(cudaMemcpy(out[w].data() + (size_t) T * N, dio, (size_t) T * HC * 4, cudaMemcpyDeviceToHost), "d2h io");
            ck(cudaMemcpy(out[w].data() + (size_t) T * (N + HC), dR, (size_t) T * D * 4, cudaMemcpyDeviceToHost), "d2h R");
        }
        // per output kind (mixed, inject, R): max |diff| relative to that kind's largest reference magnitude
        const size_t seg[4] = {0, (size_t) T * N, (size_t) T * (N + HC), (size_t) T * (N + HC + D)};
        double rel = 0.0;
        bool finite = true;
        for (int s = 0; s < 3; ++s) {
            double mref = 0.0, md = 0.0;
            for (size_t i = seg[s]; i < seg[s + 1]; ++i) {
                finite = finite && std::isfinite(out[1][i]) && std::isfinite(out[0][i]);
                mref = std::fmax(mref, std::fabs((double) out[0][i]));
                md = std::fmax(md, std::fabs((double) out[1][i] - out[0][i]));
            }
            rel = std::fmax(rel, md / (mref > 0 ? mref : 1.0));
        }
        const bool same = std::memcmp(out[1].data(), out[2].data(), out[1].size() * 4) == 0;
        const bool ok = finite && rel < tol && same;
        std::printf("T=%d: max relative difference %.3g against the default read (bound %.3g), %s, two runs %s  %s\n", T, rel,
                    tol, finite ? "finite" : "NON-FINITE", same ? "bitwise equal" : "DIFFER", ok ? "ok" : "FAIL");
        if (!ok) ++bad;
    }
    std::printf("%s: rdna_gr %s read, T = 1..%d\n", bad ? "FAIL" : "PASS", q8 ? "int8" : "bf16", TM);
    return bad ? 1 : 0;
}
