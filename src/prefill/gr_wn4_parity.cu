// src/prefill/gr_wn4_parity.cu - STRATA_HC_WN4 (nicejunjie/Strata-rdna3 3dbb40516): gr_write_norm_rs4, the 16-byte-access
// write + next-half norm of the prompt path's hyper-connection rows, against the scalar gr_write_norm_rs on the same
// synthetic inputs (N = 2560, 4 streams).  R, rs and the BF16 image xn16 must be the same bits (the switch is classed
// Speed in conversation_disk_env.def on that claim).  GPU, no model; exit 77 without a device.
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int N = 2560, HC = 4, D = N * HC;

template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    if (cudaMalloc(&p, n * sizeof(T)) != cudaSuccess) { std::printf("FAIL: cudaMalloc\n"); std::exit(2); }
    return p;
}

int check(int64_t T, int64_t ldx, uint32_t seed, cudaStream_t st) {
    std::mt19937 g(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const int64_t ld = ldx > 0 ? ldx : D;
    std::vector<float> R((size_t) T * D), bo((size_t) T * N), inj((size_t) T * HC), w(D);
    for (auto& x : R) x = nd(g) * 3.0f;
    for (auto& x : bo) x = nd(g);
    for (auto& x : inj) x = nd(g);
    for (auto& x : w) x = 1.0f + 0.1f * nd(g);
    float *dR0 = dalloc<float>(R.size()), *dR1 = dalloc<float>(R.size()), *dbo = dalloc<float>(bo.size());
    float *dinj = dalloc<float>(inj.size()), *dw = dalloc<float>(w.size());
    float *drs0 = dalloc<float>((size_t) T * HC), *drs1 = dalloc<float>((size_t) T * HC);
    uint16_t *dx0 = dalloc<uint16_t>((size_t) T * ld), *dx1 = dalloc<uint16_t>((size_t) T * ld);
    cudaMemcpy(dR0, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dR1, R.data(), R.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dbo, bo.data(), bo.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dinj, inj.data(), inj.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dw, w.data(), w.size() * 4, cudaMemcpyHostToDevice);
    cudaMemset(dx0, 0, (size_t) T * ld * 2);
    cudaMemset(dx1, 0, (size_t) T * ld * 2);
    const float eps = 1e-6f;
    strata::prefill::gr_write_norm_rs(dR0, dbo, dinj, HC, dw, eps, drs0, dx0, T, st, nullptr, ldx, false);
    if (!strata::prefill::gr_write_norm_rs4(dR1, dbo, dinj, HC, dw, eps, drs1, dx1, T, st, ldx)) {
        std::printf("FAIL T=%lld ldx=%lld: gr_write_norm_rs4 refused aligned buffers\n", (long long) T, (long long) ldx);
        return 1;
    }
    cudaStreamSynchronize(st);
    std::vector<float> r0(R.size()), r1(R.size()), s0((size_t) T * HC), s1((size_t) T * HC);
    std::vector<uint16_t> x0((size_t) T * ld), x1((size_t) T * ld);
    cudaMemcpy(r0.data(), dR0, r0.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(r1.data(), dR1, r1.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(s0.data(), drs0, s0.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(s1.data(), drs1, s1.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(x0.data(), dx0, x0.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(x1.data(), dx1, x1.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(dR0); cudaFree(dR1); cudaFree(dbo); cudaFree(dinj); cudaFree(dw);
    cudaFree(drs0); cudaFree(drs1); cudaFree(dx0); cudaFree(dx1);
    const bool okR = std::memcmp(r0.data(), r1.data(), r0.size() * 4) == 0;
    const bool okS = std::memcmp(s0.data(), s1.data(), s0.size() * 4) == 0;
    const bool okX = std::memcmp(x0.data(), x1.data(), x0.size() * 2) == 0;
    std::printf("%s T=%lld ldx=%lld: R %s, rs %s, xn16 %s\n", okR && okS && okX ? "ok  " : "FAIL", (long long) T,
                (long long) ldx, okR ? "same" : "DIFF", okS ? "same" : "DIFF", okX ? "same" : "DIFF");
    return okR && okS && okX ? 0 : 1;
}

}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        std::printf("gr_wn4_parity: no GPU device, skipped\n");
        return 77;
    }
    cudaStream_t st = nullptr;
    if (cudaStreamCreate(&st) != cudaSuccess) { std::printf("FAIL: stream\n"); return 1; }
    int fails = 0;
    uint32_t seed = 7;
    for (const int64_t T : {1, 3, 17, 128, 1000})
        for (const int64_t ldx : {(int64_t) 0, (int64_t) D + 64}) fails += check(T, ldx, seed++, st);
    cudaStreamDestroy(st);
    std::printf("gr_wn4_parity: %d failures\n", fails);
    return fails ? 1 : 0;
}
