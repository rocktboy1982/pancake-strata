// src/kernels/route_plan_parity.cpp - STRATA_ROUTE_PLAN's one launch against native_router_top10_multi (one token:
// native_router_top10) followed by resident_plan: the ids, the weights and every byte of the plan buffer, for 1-8
// tokens, logits with ties and tokens that share experts, a residency table with and without holes (the plan-error
// path).  Ported with the kernel from jstamagal/Strata-gfx1030 ef00764552 (LEADS.md "gfx1030 decode"); the fork's
// STRATA_ROUTE_PLAN=2 section and bench are not ported.  Exit 77: no GPU, or a wave64 device.  GPU, synthetic.
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/route_plan.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace K = strata::kernels;

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::printf("SKIP: no GPU\n"); return 77; }
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    if (prop.warpSize != 32) { std::printf("SKIP: wave%d device (the route plan kernel is wave32)\n", prop.warpSize); return 77; }
    const int NE = 512, KK = 10, MAXT = 8;
    const long long capx = (long long) MAXT * KK;
    const size_t plan_words = (size_t) (((4 + (capx + 1) + 2 * capx) + 1) & ~1ll) + 4 * capx + capx + 1 + 16;
    cudaStream_t s; ck(cudaStreamCreate(&s), "stream");
    float *dlog, *wa, *wb; int32_t *ia, *ib, *pa, *pb, *dres; unsigned long long* dso; uint32_t *ea, *eb;
    ck(cudaMalloc((void**) &dlog, MAXT * NE * 4), "log");
    ck(cudaMalloc((void**) &wa, MAXT * KK * 4), "wa"); ck(cudaMalloc((void**) &wb, MAXT * KK * 4), "wb");
    ck(cudaMalloc((void**) &ia, MAXT * KK * 4), "ia"); ck(cudaMalloc((void**) &ib, MAXT * KK * 4), "ib");
    ck(cudaMalloc((void**) &pa, plan_words * 4), "pa"); ck(cudaMalloc((void**) &pb, plan_words * 4), "pb");
    ck(cudaMalloc((void**) &dres, NE * 4), "res"); ck(cudaMalloc((void**) &dso, NE * 8), "so");
    ck(cudaMalloc((void**) &ea, 64), "ea"); ck(cudaMalloc((void**) &eb, 64), "eb");
    const uint8_t* cache_base = (const uint8_t*) 0x100000000ull;   // only its value enters the plan
    std::mt19937_64 rng(77);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::vector<float> hl(MAXT * NE), ha(MAXT * KK), hb(MAXT * KK);
    std::vector<int32_t> hia(MAXT * KK), hib(MAXT * KK), hpa(plan_words), hpb(plan_words), hres(NE);
    std::vector<unsigned long long> hso(NE);
    long long cases = 0, bad = 0, err_cases = 0;
    for (int trial = 0; trial < 600; ++trial) {
        const int T = 1 + (int) (rng() % MAXT);
        for (auto& v : hl) v = nd(rng);
        if (trial % 3 == 0)   // tokens that share most experts: one row copied with small changes
            for (int t = 1; t < MAXT; ++t)
                for (int e = 0; e < NE; ++e) hl[t * NE + e] = hl[e] + (rng() % 5 == 0 ? nd(rng) * 0.3f : 0.0f);
        if (trial % 4 == 1)   // ties: quantized logits
            for (auto& v : hl) v = std::round(v * 2.0f) / 2.0f;
        if (trial % 50 == 7) for (int e = 0; e < NE; ++e) hl[e] = 0.0f;   // an all-equal row
        for (int e = 0; e < NE; ++e) { hres[e] = (int32_t) ((e * 7919) % 100000); hso[e] = (rng() % (1ull << 34)) & ~63ull; }
        const bool holes = trial % 10 == 9;
        if (holes) for (int e = 0; e < NE; e += 1 + (int) (rng() % 3)) hres[e] = -1;
        const bool use_so = trial % 2 == 0;
        ck(cudaMemcpy(dlog, hl.data(), hl.size() * 4, cudaMemcpyHostToDevice), "log up");
        ck(cudaMemcpy(dres, hres.data(), NE * 4, cudaMemcpyHostToDevice), "res up");
        ck(cudaMemcpy(dso, hso.data(), NE * 8, cudaMemcpyHostToDevice), "so up");
        for (void* p : {(void*) wa, (void*) wb, (void*) ia, (void*) ib}) ck(cudaMemset(p, 0x5a, MAXT * KK * 4), "fill");
        ck(cudaMemset(pa, 0x3c, plan_words * 4), "fill"); ck(cudaMemset(pb, 0x3c, plan_words * 4), "fill");
        ck(cudaMemset(ea, 0, 64), "fill"); ck(cudaMemset(eb, 0, 64), "fill");
        if (T == 1) K::native_router_top10(dlog, ia, wa, s);
        else K::native_router_top10_multi(dlog, ia, wa, T, s);
        K::resident_plan(ia, T * KK, KK, dres, NE, cache_base, use_so ? dso : nullptr, 4096, pa, capx, nullptr, 0, s, ea);
        K::native_route_plan(dlog, ib, wb, T, dres, NE, cache_base, use_so ? dso : nullptr, 4096, pb, capx, eb, s);
        ck(cudaStreamSynchronize(s), "sync");
        uint32_t e1 = 0, e2 = 0;
        ck(cudaMemcpy(ha.data(), wa, MAXT * KK * 4, cudaMemcpyDeviceToHost), "wa"); ck(cudaMemcpy(hb.data(), wb, MAXT * KK * 4, cudaMemcpyDeviceToHost), "wb");
        ck(cudaMemcpy(hia.data(), ia, MAXT * KK * 4, cudaMemcpyDeviceToHost), "ia"); ck(cudaMemcpy(hib.data(), ib, MAXT * KK * 4, cudaMemcpyDeviceToHost), "ib");
        ck(cudaMemcpy(hpa.data(), pa, plan_words * 4, cudaMemcpyDeviceToHost), "pa"); ck(cudaMemcpy(hpb.data(), pb, plan_words * 4, cudaMemcpyDeviceToHost), "pb");
        ck(cudaMemcpy(&e1, ea, 4, cudaMemcpyDeviceToHost), "ea"); ck(cudaMemcpy(&e2, eb, 4, cudaMemcpyDeviceToHost), "eb");
        ++cases;
        err_cases += e1 != 0;
        const bool same = std::memcmp(ha.data(), hb.data(), MAXT * KK * 4) == 0 && std::memcmp(hia.data(), hib.data(), MAXT * KK * 4) == 0 &&
                          std::memcmp(hpa.data(), hpb.data(), plan_words * 4) == 0 && e1 == e2;
        if (!same) {
            ++bad;
            std::printf("MISMATCH trial %d T %d holes %d: ids %s, weights %s, plan %s, err %u vs %u\n", trial, T, (int) holes,
                        std::memcmp(hia.data(), hib.data(), MAXT * KK * 4) ? "DIFFER" : "equal",
                        std::memcmp(ha.data(), hb.data(), MAXT * KK * 4) ? "DIFFER" : "equal",
                        std::memcmp(hpa.data(), hpb.data(), plan_words * 4) ? "DIFFER" : "equal", e1, e2);
        }
    }
    std::printf("route_plan_parity: %lld cases, %lld mismatches (%lld with a non-resident expert)\n", cases, bad, err_cases);
    return bad == 0 ? 0 : 1;
}
