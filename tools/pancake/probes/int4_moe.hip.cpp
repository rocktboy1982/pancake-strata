// int4_moe.hip.cpp - LEADS.md Part L L3 / L4: the expert-gather iu4 GEMM for Q2_0 experts (the kernel behind STRATA_PF_IU4), standalone, gfx1100.
// Written here from the host side in include/strata/kernels/int4_moe.hpp; NOT wired into the engine (MOD_PLAN.md section 6 item 54 says why).
//
// WHAT    One wave a 16 x 16 output tile of one expert: the weights are read straight from the Q2_0 blocks (18 bytes: fp16 d, 16 code bytes),
//         16 codes = one 32-bit word are spread to a 16-nibble operand by SWAR (no repack pass, no int4 copy), the activations are the A8
//         pair of the int4_wmma probe (h signed, l unsigned, q = 16 h + l), `v_wmma_i32_16x16x16_iu4` twice a slice, the int32 of a
//         32-group scaled by d x the activation block scale in f32.  Token rows are grouped by expert (bounds[e]..bounds[e + 1], the order the
//         engine sorts routed entries in), a tile list (expert, first row) is built on the host.  Untuned on purpose: a floor.
// BUILD   hipcc -O2 --offload-arch=gfx1100 -Iinclude tools/pancake/probes/int4_moe.hip.cpp -o /tmp/int4_moe   (or CMake target int4_moe_probe)
// RUN     HIP_VISIBLE_DEVICES=0 /tmp/int4_moe [--m 512 --k 2560 --tokens 2048 --experts 64 --block 256 --reps 20 --parity-only]
//         exit 0 pass, 1 mismatch, 77 no gfx11 GPU.  Daily shapes: gate/up K = 2560, down K = 640 (use --k 640 --block 128).
// READ    PARITY: the device GEMM against int4rot::gemm_ref on the same codes, expert by expert (relative error <= 2e-4; the integers are exact,
//         the difference is the f32 order of the group sum).  BENCH: median ms, TOPS counting both int4 passes, weight bytes read a second
//         (the Q2_0 bytes of the experts that got tokens: 2.25 bits a weight), for the box comparison with STRATA_PF_F16's kernel.
//         The decision it gives: iu4 tile rate on a Q2_0 expert shape against the f16 path (T-820) decides whether an engine kernel is worth
//         writing; the parity line decides whether the operand layout (the open point of L3) is right.
// STATUS  syntax + device compile checked with the ROCm 6.2 SDK on the Windows PC; NEVER RUN (no gfx1100 GPU there).  Box run needed.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "strata/kernels/int4_moe.hpp"

#define CK(x)                                                                                        \
    do {                                                                                             \
        hipError_t e_ = (x);                                                                         \
        if (e_ != hipSuccess) {                                                                      \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));   \
            std::exit(1);                                                                            \
        }                                                                                            \
    } while (0)

using i2v = int __attribute__((ext_vector_type(2)));
using i8v = int __attribute__((ext_vector_type(8)));

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__)
#define INT4_GFX11 1
#endif

// the same lines as int4moe::q2_0_half_to_nibbles (include/strata/kernels/int4_moe.hpp, tested on the host for every 16-bit word)
__device__ __forceinline__ unsigned q2_half_to_nibbles(unsigned c16) {
    unsigned x = c16 & 0xffffu;
    x = (x | (x << 8)) & 0x00ff00ffu;
    x = (x | (x << 4)) & 0x0f0f0f0fu;
    x = (x | (x << 2)) & 0x33333333u;
    return ((x | 0x88888888u) - 0x11111111u) ^ 0x88888888u;
}

__global__ void pack_kernel(const int8_t* __restrict__ in, unsigned* __restrict__ out, long long slices) {
    const long long s = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= slices) return;
    unsigned w[2] = {0u, 0u};
    for (int e = 0; e < 16; ++e) w[e >> 3] |= ((unsigned) in[s * 16 + e] & 15u) << (4 * (e & 7));
    out[s * 2] = w[0];
    out[s * 2 + 1] = w[1];
}

// grid (M / 16, n_tiles), block 32.  tile t = (tile_e[t], tile_n0[t]): rows n0 .. n0 + 15 of the sorted activations, those >= bounds[e + 1] are padding.
__global__ void __launch_bounds__(32) gemm_q2_kernel(const unsigned char* __restrict__ wq, const int* __restrict__ tile_e,
                                                     const int* __restrict__ tile_n0, const int* __restrict__ bounds,
                                                     const unsigned* __restrict__ hip_, const unsigned* __restrict__ lop,
                                                     const float* __restrict__ as, float* __restrict__ y, int M, int K, int B) {
#if defined(INT4_GFX11)
    const int l = threadIdx.x, lr = l & 15;
    const int e = tile_e[blockIdx.y], n0 = tile_n0[blockIdx.y], nend = bounds[e + 1];
    const int mrow = min(blockIdx.x * 16 + lr, M - 1), ncol = n0 + lr;
    const bool nok = ncol < nend;
    const int nsl = K / 16, nb = K / 64;
    const unsigned char* wrow = wq + ((size_t) e * M + mrow) * nb * 18;
    float yacc[8];
    for (int i = 0; i < 8; ++i) yacc[i] = 0.f;
    for (int g = 0; g < K / 32; ++g) {
        i8v ah, al;
        for (int i = 0; i < 8; ++i) { ah[i] = 0; al[i] = 0; }
        for (int sl = 0; sl < 2; ++sl) {
            const int k0 = g * 32 + sl * 16, s = k0 / 16;
            const unsigned char* blk = wrow + (size_t) (k0 / 64) * 18;
            const unsigned short* c16 = reinterpret_cast<const unsigned short*>(blk + 2 + ((k0 % 64) / 16) * 4);   // 2-byte aligned only
            const unsigned c32 = (unsigned) c16[0] | ((unsigned) c16[1] << 16);
            const i2v wv = {(int) q2_half_to_nibbles(c32), (int) q2_half_to_nibbles(c32 >> 16)};
            i2v av = {0, 0}, lv = {0, 0};
            if (nok) {
                const unsigned* aq = hip_ + ((size_t) ncol * nsl + s) * 2;
                const unsigned* lq = lop + ((size_t) ncol * nsl + s) * 2;
                av = i2v{(int) aq[0], (int) aq[1]};
                lv = i2v{(int) lq[0], (int) lq[1]};
            }
            ah = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, wv, true, av, ah, false);
            al = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, wv, false, lv, al, false);
        }
        // the scale of the weight row that lane l's OUTPUT m belongs to is read below per output (rows differ across the 8 outputs)
        const float da = nok ? as[(size_t) ncol * (K / B) + (g * 32) / B] : 0.f;
        for (int i = 0; i < 8; ++i) {
            const int m = min(blockIdx.x * 16 + 2 * i + (l >> 4), M - 1);
            const unsigned short hd = *reinterpret_cast<const unsigned short*>(wq + (((size_t) e * M + m) * nb + (g * 32) / 64) * 18);
            const float sw = (float) __builtin_bit_cast(_Float16, hd);
            yacc[i] += sw * da * (float) (16 * ah[i] + al[i]);
        }
    }
    if (nok)
        for (int i = 0; i < 8; ++i) y[(size_t) ncol * M + blockIdx.x * 16 + 2 * i + (l >> 4)] = yacc[i];
#else
    if (M < 0) y[0] = 0.f;
#endif
}

template <class T> static T* dev_copy(const std::vector<T>& v) {
    T* d;
    CK(hipMalloc(&d, v.size() * sizeof(T)));
    CK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}

static uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000, e = (x >> 23) & 255, m = x & 0x7fffff;
    int he = (int) e - 127 + 15;
    uint32_t hm = m >> 13;
    const uint32_t rest = m & 0x1fff;
    if (rest > 0x1000 || (rest == 0x1000 && (hm & 1))) ++hm;
    if (hm == 1024) { hm = 0; ++he; }
    return (uint16_t) (s | ((uint32_t) he << 10) | hm);
}

int main(int argc, char** argv) {
    int M = 512, K = 2560, tokens = 2048, E = 64, B = 256, reps = 20, device = 0;
    bool parity_only = false;
    for (int i = 1; i < argc; ++i) {
        auto arg = [&](const char* k) { return !std::strcmp(argv[i], k) && i + 1 < argc; };
        if (arg("--m")) M = std::atoi(argv[++i]);
        else if (arg("--k")) K = std::atoi(argv[++i]);
        else if (arg("--tokens")) tokens = std::atoi(argv[++i]);
        else if (arg("--experts")) E = std::atoi(argv[++i]);
        else if (arg("--block")) B = std::atoi(argv[++i]);
        else if (arg("--reps")) reps = std::atoi(argv[++i]);
        else if (arg("--device")) device = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--parity-only")) parity_only = true;
        else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
    }
    if (M % 16 || K % 64 || K % B || (B & (B - 1)) || B > 1024 || B < 32 || E < 1 || tokens < 1) {
        std::fprintf(stderr, "int4_moe: need M a multiple of 16, K a multiple of 64 and of the block B (a power of two, 32..1024)\n");
        return 2;
    }
    hipDeviceProp_t p;
    if (hipSetDevice(device) != hipSuccess || hipGetDeviceProperties(&p, device) != hipSuccess ||
        (!std::strstr(p.gcnArchName, "gfx110") && !std::strstr(p.gcnArchName, "gfx115"))) {
        std::fprintf(stderr, "int4_moe: no gfx11 GPU (device %d)\n", device);
        return 77;
    }
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0.f, 1.f);
    // experts: random Q2_0 blocks; tokens: a random expert each (skewed: expert id ~ min of two draws), rows sorted by expert
    strata::kernels::int4moe::Q2Experts ex;
    ex.n_expert = E; ex.rows = M; ex.K = K;
    ex.bytes.resize((size_t) E * M * (K / 64) * 18);
    for (size_t b = 0; b < ex.bytes.size() / 18; ++b) {
        const uint16_t h = float_to_half(0.002f + (float) (rng() % 1000) / 8000.f);
        std::memcpy(&ex.bytes[b * 18], &h, 2);
        for (int j = 0; j < 16; ++j) ex.bytes[b * 18 + 2 + j] = (uint8_t) rng();
    }
    std::vector<int> cnt(E, 0);
    for (int t = 0; t < tokens; ++t) ++cnt[std::min(rng() % E, rng() % E)];
    std::vector<int> bounds(E + 1, 0);
    for (int e = 0; e < E; ++e) bounds[e + 1] = bounds[e] + cnt[e];
    const int N = bounds[E];
    std::vector<float> x((size_t) N * K);
    for (auto& v : x) v = nd(rng);
    strata::kernels::int4rot::rotate_rows(x, N, K, B);
    const auto a = strata::kernels::int4rot::quantize_acts(x, N, K, B, true);
    std::vector<int> tile_e, tile_n0;
    int used = 0;
    for (int e = 0; e < E; ++e) {
        used += cnt[e] > 0;
        for (int n0 = bounds[e]; n0 < bounds[e + 1]; n0 += 16) { tile_e.push_back(e); tile_n0.push_back(n0); }
    }
    std::printf("device %s, M=%d K=%d tokens=%d experts=%d (%d with tokens) tiles=%zu block=%d A8\n", p.gcnArchName, M, K, N, E, used,
                tile_e.size(), B);
    // device buffers
    unsigned char* d_w;
    CK(hipMalloc(&d_w, ex.bytes.size()));
    CK(hipMemcpy(d_w, ex.bytes.data(), ex.bytes.size(), hipMemcpyHostToDevice));
    int *d_te = dev_copy(tile_e), *d_tn = dev_copy(tile_n0), *d_b = dev_copy(bounds);
    int8_t *d_hi = dev_copy(a.hi), *d_lo = dev_copy(a.lo);
    float* d_as = dev_copy(a.scale);
    const long long xs = (long long) N * K / 16;
    unsigned *d_hip, *d_lop;
    CK(hipMalloc(&d_hip, xs * 8));
    CK(hipMalloc(&d_lop, xs * 8));
    pack_kernel<<<(unsigned) ((xs + 255) / 256), 256>>>(d_hi, d_hip, xs);
    pack_kernel<<<(unsigned) ((xs + 255) / 256), 256>>>(d_lo, d_lop, xs);
    float* d_y;
    CK(hipMalloc(&d_y, (size_t) N * M * sizeof(float)));
    CK(hipMemset(d_y, 0, (size_t) N * M * sizeof(float)));
    auto run = [&] { gemm_q2_kernel<<<dim3(M / 16, (unsigned) tile_e.size()), 32>>>(d_w, d_te, d_tn, d_b, d_hip, d_lop, d_as, d_y, M, K, B); };
    run();
    CK(hipDeviceSynchronize());
    std::vector<float> y((size_t) N * M);
    CK(hipMemcpy(y.data(), d_y, y.size() * sizeof(float), hipMemcpyDeviceToHost));
    // reference: expert by expert on the same codes
    double dn = 0, nn = 0;
    for (int e = 0; e < E; ++e) {
        const int n0 = bounds[e], n1 = bounds[e + 1];
        if (n1 == n0) continue;
        const auto q = strata::kernels::int4iq2::q2_0_to_int4(ex.row_block(e, 0, 0), M, K);
        strata::kernels::int4rot::WeightQ wq;
        wq.M = M; wq.K = K; wq.q = q.q; wq.scale = q.scale;
        strata::kernels::int4rot::ActQ sub;
        sub.N = n1 - n0; sub.K = K; sub.B = B; sub.a8 = true;
        sub.hi.assign(a.hi.begin() + (size_t) n0 * K, a.hi.begin() + (size_t) n1 * K);
        sub.lo.assign(a.lo.begin() + (size_t) n0 * K, a.lo.begin() + (size_t) n1 * K);
        sub.scale.assign(a.scale.begin() + (size_t) n0 * (K / B), a.scale.begin() + (size_t) n1 * (K / B));
        const auto ref = strata::kernels::int4rot::gemm_ref(wq, sub);
        for (int n = n0; n < n1; ++n)
            for (int m = 0; m < M; ++m) {
                const double r = ref[(size_t) (n - n0) * M + m], g = y[(size_t) n * M + m];
                dn += (g - r) * (g - r);
                nn += r * r;
            }
    }
    const double rel = nn > 0 ? std::sqrt(dn / nn) : 0.0;
    std::printf("PARITY gemm: relative error vs the host integer pipeline on the same codes %.3e\n", rel);
    const bool ok = rel <= 2e-4;
    std::printf("%s\n", ok ? "int4_moe: PARITY PASS" : "int4_moe: PARITY FAIL");
    if (!parity_only && ok) {
        hipEvent_t e0, e1;
        CK(hipEventCreate(&e0));
        CK(hipEventCreate(&e1));
        std::vector<float> ms;
        for (int r = 0; r < reps + 1; ++r) {
            CK(hipEventRecord(e0));
            run();
            CK(hipEventRecord(e1));
            CK(hipEventSynchronize(e1));
            float t;
            CK(hipEventElapsedTime(&t, e0, e1));
            if (r) ms.push_back(t);   // run 1 discarded
        }
        std::sort(ms.begin(), ms.end());
        const double med = ms[ms.size() / 2];
        // work counted per tile (16 padded rows), 2 int4 passes
        const double ops = 2.0 * 2.0 * 16.0 * (double) M * (double) K * (double) tile_e.size();
        const double wbytes = (double) used * M * (K / 64) * 18;
        std::printf("BENCH gemm: median %.4f ms, best %.4f ms, %.1f TOPS counting both int4 passes of padded tiles, weights %.1f GB/s "
                    "(Q2_0 bytes of the %d experts with tokens)\n", med, (double) ms.front(), ops / (med * 1e-3) / 1e12, wbytes / (med * 1e-3) / 1e9, used);
    }
    return ok ? 0 : 1;
}
