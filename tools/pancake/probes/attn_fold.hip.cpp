// attn_fold.hip.cpp - LEADS.md Part L L9: split-KV decode attention with the merge folded into the last chunk block, against the separate merge
// launch, bit for bit, plus the launch-chain time of both.  The protocol (rules R1 .. R5) is in include/strata/kernels/fold_merge.hpp and is
// tested on the host (fold_merge_test); this probe is the GPU half: the release / acquire pairing of the partials with the ticket, which a
// x86 test cannot show.  A toy attention (one query token, G = 12 query heads a KV head, head dim 256, chunks of 64 cells = the shapes of
// src/kernels/cuda/qsa_decode_attn.cu), fp32 K/V, NOT the engine's kernels: the engine port (STRATA_ATTN_MERGE_FOLD) waits for this result.
//
// BUILD   hipcc -O2 --offload-arch=gfx1100 -Iinclude tools/pancake/probes/attn_fold.hip.cpp -o /tmp/attn_fold   (CMake target attn_fold_probe)
// RUN     HIP_VISIBLE_DEVICES=0 /tmp/attn_fold [--cap 8192 --ctx 8192 --layers 40 --reps 20 --launches 200]
//         --ctx below --cap leaves chunk blocks with no cells (R1: they still take a ticket); exit 0 pass, 1 mismatch, 77 no gfx11 GPU.
// READ    PARITY: `launches` replays of the same folded graph (counter reset by the last block, no memset between) each bit-equal to the
//         separate merge; any difference or a counter not back at 0 is a FAIL.  BENCH: microseconds a layer of [chunk + merge] against
//         [chunk with the fold], in one captured graph of `layers` dependent layers (the dependency is the output feeding the next query).
//         A gain near the authors' 5 us a layer (paper) would be 0.5% of our 24.8 ms window x 40 layers: the probe decides whether the port is
//         worth a day, nothing more.
// STATUS  syntax + device compile checked with the ROCm 6.2 SDK on the Windows PC; NEVER RUN (no gfx1100 GPU there).  Box run needed.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                                        \
    do {                                                                                             \
        hipError_t e_ = (x);                                                                         \
        if (e_ != hipSuccess) {                                                                      \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));   \
            std::exit(1);                                                                            \
        }                                                                                            \
    } while (0)

constexpr int HD = 256, G = 12, CHUNK = 64, HKV = 2;

__device__ __forceinline__ float wexp(float x) { return __expf(x); }   // the one weight function of both merges

// one block a (chunk, kv head): scores, softmax partials, weighted V.  part layout as attn_merge_kernel: m / l [kvh][chunk][G], acc [kvh][chunk][G][HD]
template <bool FOLD>
__global__ void __launch_bounds__(HD) chunk_kernel(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                                                   float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pa, int ctx,
                                                   int n_chunks, float* __restrict__ out, int* __restrict__ counter) {
    __shared__ float s[G][CHUNK];
    __shared__ int s_last;
    const int c = blockIdx.x, kvh = blockIdx.y, t = threadIdx.x;
    const int c0 = c * CHUNK;
    const int n_here = max(0, min(CHUNK, ctx - c0));
    const size_t slot = (size_t) kvh * n_chunks + c;
    if (n_here <= 0) {   // R1: an empty chunk writes the identity partial and still takes a ticket below
        if (t < G) { pm[slot * G + t] = -FLT_MAX; pl[slot * G + t] = 0.f; }
    } else {
        {   // scores: thread t = (cell t / 4, heads 3 (t % 4) .. +2)
            const int cell = t / 4, h0 = 3 * (t % 4);
            if (cell < n_here) {
                float a0 = 0.f, a1 = 0.f, a2 = 0.f;
                const float* kp = k + (size_t) (c0 + cell) * HD;
                const float* q0 = q + ((size_t) kvh * G + h0) * HD;
                for (int d = 0; d < HD; ++d) {
                    const float kv_ = kp[d];
                    a0 = fmaf(q0[d], kv_, a0);
                    a1 = fmaf(q0[HD + d], kv_, a1);
                    a2 = fmaf(q0[2 * HD + d], kv_, a2);
                }
                s[h0][cell] = a0 * 0.0625f;
                s[h0 + 1][cell] = a1 * 0.0625f;
                s[h0 + 2][cell] = a2 * 0.0625f;
            }
        }
        __syncthreads();
        float mh[G];
        for (int h = 0; h < G; ++h) {
            float m = -FLT_MAX;
            for (int i = 0; i < n_here; ++i) m = fmaxf(m, s[h][i]);
            mh[h] = m;
        }
        float acc[G];
        for (int h = 0; h < G; ++h) acc[h] = 0.f;
        float lsum[G];
        for (int h = 0; h < G; ++h) lsum[h] = 0.f;
        for (int i = 0; i < n_here; ++i) {
            const float vv = v[(size_t) (c0 + i) * HD + t];
            for (int h = 0; h < G; ++h) {
                const float p = wexp(s[h][i] - mh[h]);
                acc[h] = fmaf(p, vv, acc[h]);
                if (t == 0) lsum[h] += p;
            }
        }
        for (int h = 0; h < G; ++h) pa[(slot * G + h) * HD + t] = acc[h];
        if (t == 0)
            for (int h = 0; h < G; ++h) { pm[slot * G + h] = mh[h]; pl[slot * G + h] = lsum[h]; }
    }
    if (FOLD) {
        __threadfence();   // release: this block's partial is visible device-wide before its ticket
        __syncthreads();
        if (t == 0) {
            const int ticket = atomicAdd(&counter[kvh], 1);
            s_last = ticket == n_chunks - 1;
        }
        __syncthreads();
        if (s_last) {
            __threadfence();   // acquire: every other block's partial is visible to this block
            for (int h = 0; h < G; ++h) {   // attn_merge_kernel's arithmetic, chunks in index order (R2)
                float M = -FLT_MAX;
                for (int cc = 0; cc < n_chunks; ++cc) M = fmaxf(M, pm[((size_t) kvh * n_chunks + cc) * G + h]);
                float L = 0.f, a = 0.f;
                for (int cc = 0; cc < n_chunks; ++cc) {
                    const size_t sl = (size_t) kvh * n_chunks + cc;
                    const float m = pm[sl * G + h];
                    if (m == -FLT_MAX) continue;
                    const float w = wexp(m - M);
                    L = fmaf(pl[sl * G + h], w, L);
                    a = fmaf(pa[(sl * G + h) * HD + t], w, a);
                }
                out[((size_t) kvh * G + h) * HD + t] = L > 0.f ? a / L : 0.f;
            }
            if (t == 0) counter[kvh] = 0;   // R3: the last block resets, the next launch starts at 0
        }
    }
}

__global__ void __launch_bounds__(HD) merge_kernel(const float* __restrict__ pa, const float* __restrict__ pm, const float* __restrict__ pl,
                                                   int n_chunks, float* __restrict__ out) {
    const int h = blockIdx.x, kvh = h / G, hl = h % G, t = threadIdx.x;   // grid HKV * G
    float M = -FLT_MAX;
    for (int c = 0; c < n_chunks; ++c) M = fmaxf(M, pm[((size_t) kvh * n_chunks + c) * G + hl]);
    float L = 0.f, a = 0.f;
    for (int c = 0; c < n_chunks; ++c) {
        const size_t sl = (size_t) kvh * n_chunks + c;
        const float m = pm[sl * G + hl];
        if (m == -FLT_MAX) continue;
        const float w = wexp(m - M);
        L = fmaf(pl[sl * G + hl], w, L);
        a = fmaf(pa[(sl * G + hl) * HD + t], w, a);
    }
    out[(size_t) h * HD + t] = L > 0.f ? a / L : 0.f;
}

// the output of a layer is the next layer's query (a dependent chain): q[i] = out[i] * 0.5 + base[i]
__global__ void next_q(const float* __restrict__ out, const float* __restrict__ base, float* __restrict__ q, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) q[i] = out[i] * 0.5f + base[i];
}

template <class T> static T* dev_copy(const std::vector<T>& v) {
    T* d;
    CK(hipMalloc(&d, v.size() * sizeof(T)));
    CK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    int cap = 8192, ctx = -1, layers = 40, reps = 20, launches = 200, device = 0;
    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* k) { return !std::strcmp(argv[i], k) && i + 1 < argc; };
        if (arg("--cap")) cap = std::atoi(argv[++i]);
        else if (arg("--ctx")) ctx = std::atoi(argv[++i]);
        else if (arg("--layers")) layers = std::atoi(argv[++i]);
        else if (arg("--reps")) reps = std::atoi(argv[++i]);
        else if (arg("--launches")) launches = std::atoi(argv[++i]);
        else if (arg("--device")) device = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (ctx < 0) ctx = cap;
    if (cap % CHUNK || ctx > cap || ctx < 1) { std::fprintf(stderr, "attn_fold: cap a multiple of %d, 1 <= ctx <= cap\n", CHUNK); return 2; }
    hipDeviceProp_t p;
    if (hipSetDevice(device) != hipSuccess || hipGetDeviceProperties(&p, device) != hipSuccess ||
        (!std::strstr(p.gcnArchName, "gfx110") && !std::strstr(p.gcnArchName, "gfx115"))) {
        std::fprintf(stderr, "attn_fold: no gfx11 GPU (device %d)\n", device);
        return 77;
    }
    const int n_chunks = cap / CHUNK;
    std::printf("device %d %s, cap %d (%d chunks), ctx %d (%d live), layers %d\n", device, p.gcnArchName, cap, n_chunks, ctx,
                (ctx + CHUNK - 1) / CHUNK, layers);
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> kk((size_t) cap * HD), vv((size_t) cap * HD), qb((size_t) HKV * G * HD);
    for (auto& x : kk) x = nd(rng);
    for (auto& x : vv) x = nd(rng);
    for (auto& x : qb) x = nd(rng);
    // note: K / V are shared by the two KV heads of this toy (the memory traffic is not the point, the protocol is)
    float *d_k = dev_copy(kk), *d_v = dev_copy(vv), *d_qb = dev_copy(qb);
    const size_t nq = (size_t) HKV * G * HD;
    float *d_q, *d_out_sep, *d_out_fold, *d_pm, *d_pl, *d_pa;
    CK(hipMalloc(&d_q, nq * 4));
    CK(hipMalloc(&d_out_sep, nq * 4));
    CK(hipMalloc(&d_out_fold, nq * 4));
    CK(hipMalloc(&d_pm, (size_t) HKV * n_chunks * G * 4));
    CK(hipMalloc(&d_pl, (size_t) HKV * n_chunks * G * 4));
    CK(hipMalloc(&d_pa, (size_t) HKV * n_chunks * G * HD * 4));
    int* d_cnt;
    CK(hipMalloc(&d_cnt, HKV * sizeof(int)));
    CK(hipMemset(d_cnt, 0, HKV * sizeof(int)));   // R3: zeroed once; the last block resets it from here on
    hipStream_t st;
    CK(hipStreamCreate(&st));
    // the K read of the chunk kernel indexes the cap x HD array the same for both heads (see the note above)
    auto layer_sep = [&](hipStream_t s) {
        chunk_kernel<false><<<dim3(n_chunks, HKV), HD, 0, s>>>(d_q, d_k, d_v, d_pm, d_pl, d_pa, ctx, n_chunks, d_out_sep, d_cnt);
        merge_kernel<<<HKV * G, HD, 0, s>>>(d_pa, d_pm, d_pl, n_chunks, d_out_sep);
        next_q<<<(unsigned) ((nq + 255) / 256), 256, 0, s>>>(d_out_sep, d_qb, d_q, (int) nq);
    };
    auto layer_fold = [&](hipStream_t s) {
        chunk_kernel<true><<<dim3(n_chunks, HKV), HD, 0, s>>>(d_q, d_k, d_v, d_pm, d_pl, d_pa, ctx, n_chunks, d_out_fold, d_cnt);
        next_q<<<(unsigned) ((nq + 255) / 256), 256, 0, s>>>(d_out_fold, d_qb, d_q, (int) nq);
    };
    auto build = [&](bool fold) {
        hipGraph_t g;
        hipGraphExec_t ex;
        CK(hipStreamBeginCapture(st, hipStreamCaptureModeGlobal));
        for (int l = 0; l < layers; ++l) { if (fold) layer_fold(st); else layer_sep(st); }
        CK(hipStreamEndCapture(st, &g));
        CK(hipGraphInstantiate(&ex, g, nullptr, nullptr, 0));
        return ex;
    };
    auto reset_q = [&] { CK(hipMemcpy(d_q, d_qb, nq * 4, hipMemcpyDeviceToDevice)); };
    hipGraphExec_t g_sep = build(false), g_fold = build(true);
    // parity: the final query (the output of the whole chain) and the last layer's output, separate vs folded, over many replays
    reset_q();
    CK(hipGraphLaunch(g_sep, st));
    CK(hipStreamSynchronize(st));
    std::vector<float> ref(nq), q_ref(nq);
    CK(hipMemcpy(ref.data(), d_out_sep, nq * 4, hipMemcpyDeviceToHost));
    CK(hipMemcpy(q_ref.data(), d_q, nq * 4, hipMemcpyDeviceToHost));
    bool ok = true;
    long bad = 0;
    for (int r = 0; r < launches; ++r) {
        reset_q();
        CK(hipGraphLaunch(g_fold, st));
        CK(hipStreamSynchronize(st));
        std::vector<float> o(nq), qf(nq);
        CK(hipMemcpy(o.data(), d_out_fold, nq * 4, hipMemcpyDeviceToHost));
        CK(hipMemcpy(qf.data(), d_q, nq * 4, hipMemcpyDeviceToHost));
        if (std::memcmp(o.data(), ref.data(), nq * 4) || std::memcmp(qf.data(), q_ref.data(), nq * 4)) ++bad;
    }
    int cnt[HKV];
    CK(hipMemcpy(cnt, d_cnt, sizeof cnt, hipMemcpyDeviceToHost));
    std::printf("PARITY fold: %d replays of the folded %d-layer graph, %ld differ from the separate merge; counters after: %d %d (must be 0)\n",
                launches, layers, bad, cnt[0], cnt[1]);
    ok = bad == 0 && cnt[0] == 0 && cnt[1] == 0;
    std::printf("%s\n", ok ? "attn_fold: PARITY PASS" : "attn_fold: PARITY FAIL");
    if (ok) {
        auto bench = [&](hipGraphExec_t ex) {
            hipEvent_t e0, e1;
            CK(hipEventCreate(&e0));
            CK(hipEventCreate(&e1));
            std::vector<float> ms;
            for (int r = 0; r < reps + 1; ++r) {
                reset_q();
                CK(hipStreamSynchronize(st));
                CK(hipEventRecord(e0, st));
                CK(hipGraphLaunch(ex, st));
                CK(hipEventRecord(e1, st));
                CK(hipEventSynchronize(e1));
                float t;
                CK(hipEventElapsedTime(&t, e0, e1));
                if (r) ms.push_back(t);   // run 1 discarded
            }
            std::sort(ms.begin(), ms.end());
            return ms[ms.size() / 2] * 1000.0 / layers;   // us a layer
        };
        const double us_sep = bench(g_sep), us_fold = bench(g_fold);
        std::printf("BENCH chain: separate merge %.2f us a layer, folded %.2f us a layer, difference %.2f us (%.1f%%); authors' gain 5.3 us (paper, not ours)\n",
                    us_sep, us_fold, us_sep - us_fold, 100.0 * (us_sep - us_fold) / us_sep);
    }
    return ok ? 0 : 1;
}
