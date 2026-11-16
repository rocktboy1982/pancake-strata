// src/prefill/gated16_parity.cu - STRATA_HC_GATED16 (jstamagal/Strata-gfx1030; LEADS.md): gr_mix_r reading `gated` as the
// FP16-out GEMM left it (row t's 10240 halves at the start of its FP32 row, g_f16) against gr_mix_r on the same values
// widened to FP32 first (what Gemm::f16_inplace's widen_rows_f16 stores), bit for bit: mixed, its BF16 image (hi and lo)
// and its FP16 copy.  The switch is classed Speed in conversation_disk_env.def on that claim.  HIP only (the halves read
// is the HIP build's); exit 77 elsewhere or without a device.  Synthetic, no model.
#include "strata/prefill/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <random>
#include <vector>

namespace sp = strata::prefill;

namespace {
constexpr int64_t N = 2560, HC = 4, D = N * HC;

template <class T> T* dev(const std::vector<T>& h) {
    T* d = nullptr;
    if (cudaMalloc(&d, h.size() * sizeof(T)) != cudaSuccess) { std::printf("FAIL: cudaMalloc\n"); std::exit(2); }
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
template <class T> std::vector<T> host(const T* d, size_t n) {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
}

int run_case(int64_t T, bool lo, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    std::vector<float> R((size_t) (T * D)), rs((size_t) (T * HC)), wn((size_t) D), gw((size_t) (T * D));
    std::vector<uint16_t> gh((size_t) (T * D * 2), 0);   // T rows of D floats; the halves fill each row's first half
    for (auto& v : R) v = nd(rng) * 4.f;
    for (auto& v : rs) v = 0.2f + u01(rng);
    for (auto& v : wn) v = 0.5f + u01(rng);
    for (int64_t t = 0; t < T; ++t)
        for (int64_t i = 0; i < D; ++i) {
            const __half h = __float2half(nd(rng) * 3.f);
            uint16_t bits;
            std::memcpy(&bits, &h, 2);
            gh[(size_t) (t * D * 2 + i)] = bits;              // row t starts at float t * D = half 2 t D
            gw[(size_t) (t * D + i)] = __half2float(h);       // the value the in-place widening stores
        }
    float *dR = dev(R), *drs = dev(rs), *dwn = dev(wn), *dgw = dev(gw);
    float* dgh = reinterpret_cast<float*>(dev(gh));
    float *dm0 = nullptr, *dm1 = nullptr;
    uint16_t *db0 = nullptr, *db1 = nullptr, *dh0 = nullptr, *dh1 = nullptr, *dl0 = nullptr, *dl1 = nullptr;
    const size_t n = (size_t) (T * N);
    cudaMalloc(&dm0, n * 4); cudaMalloc(&dm1, n * 4);
    for (uint16_t** p : {&db0, &db1, &dh0, &dh1, &dl0, &dl1}) { cudaMalloc(p, n * 2); cudaMemset(*p, 0x5a, n * 2); }
    sp::gr_mix_r(dR, drs, dwn, dgw, dm0, db0, T, nullptr, dh0, lo ? dl0 : nullptr, false);
    sp::gr_mix_r(dR, drs, dwn, dgh, dm1, db1, T, nullptr, dh1, lo ? dl1 : nullptr, true);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { std::printf("  T %lld: %s\n", (long long) T, cudaGetErrorString(e)); return 1; }
    const auto m0 = host(dm0, n), m1 = host(dm1, n);
    const auto b0 = host(db0, n), b1 = host(db1, n), h0 = host(dh0, n), h1 = host(dh1, n), l0 = host(dl0, n), l1 = host(dl1, n);
    const bool same = std::memcmp(m0.data(), m1.data(), n * 4) == 0 && std::memcmp(b0.data(), b1.data(), n * 2) == 0 &&
                      std::memcmp(h0.data(), h1.data(), n * 2) == 0 && std::memcmp(l0.data(), l1.data(), n * 2) == 0;
    std::printf("  T %5lld%s: %s\n", (long long) T, lo ? " lo" : "", same ? "mixed, BF16 image, FP16 copy identical" : "DIFFER");
    for (void* p : std::initializer_list<void*>{dR, drs, dwn, dgw, dgh, dm0, dm1, db0, db1, dh0, dh1, dl0, dl1}) cudaFree(p);
    return same ? 0 : 1;
}

}  // namespace

int main() {
#if !defined(__HIPCC__)
    std::printf("gated16_parity: the FP16 read is the HIP build's (skipped)\n");
    return 77;
#else
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("gated16_parity: no device (skipped)\n");
        return 77;
    }
    int fails = 0;
    fails += run_case(1, false, 1);
    fails += run_case(333, true, 2);
    fails += run_case(1024, false, 3);
    std::printf("gated16_parity: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
#endif
}
