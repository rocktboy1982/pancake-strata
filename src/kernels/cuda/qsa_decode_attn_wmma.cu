// src/kernels/cuda/qsa_decode_attn_wmma.cu - see include/strata/kernels/qsa_decode_attn_wmma.hpp.
// pancake: BitDecoding-style (Edinburgh, HPCA 2026, MIT) decode attention over low-bit KV on RDNA3 matrix cores.
#include "strata/kernels/qsa_decode_attn_wmma.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#if defined(__HIPCC__)
#include "strata/kernels/gfx_arch.hpp"
#endif

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

namespace strata::kernels {

#if defined(__HIPCC__)
namespace {

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
#define STRATA_DW_GFX11 1
#else
#define STRATA_DW_GFX11 0
#endif

constexpr int HD = 256;       // head_dim
constexpr int G = 12;         // query heads per KV head
constexpr int CHUNK = kQsaDecodeAttnChunk;
constexpr int THREADS = 256;  // 8 wave32
constexpr int Q4B = (int) sizeof(block_q4_0);       // 18
constexpr int Q4ROW = (HD / QK4_0) * Q4B;           // 144 bytes per head row
constexpr int VROW = HD;                            // staged V row: 256 int8 codes, or 144 q4_0 bytes
static_assert(CHUNK == 64 && THREADS == 256, "the wave split below assumes 64 cells and 8 waves");

typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));

// gfx11 wave32 16x16x16 layout: A lane l holds row l % 16, all 16 k (lanes 16..31 repeat 0..15); B lane l holds
// column l % 16, all 16 k; C/D lane l holds D[2 * i + l / 16][l % 16], i = 0..7.
__device__ __forceinline__ f8 wmma(h16 a, h16 b, f8 c) {
#if STRATA_DW_GFX11
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    __builtin_trap();
    return c;
#endif
}

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

struct alignas(16) Smem {
    uint8_t v[CHUNK][VROW];      // the chunk's V rows (int8 codes or raw q4_0 blocks)
    float part[4][G][CHUNK + 1]; // q.k per 64-dim group, K scale applied
    float p[G][CHUNK + 1];       // probabilities
    float vs[8][CHUNK];          // V scale per (group or q4 block, cell); 0 for a masked cell
    long long row[CHUNK];        // pool row of each cell, -1 masked
    float qmax[8];
};

// KV_MODE 1: int8 K and V; 2: q4_0 K and V (both rotated by the caller); 3: int8 K, q4_0 V
template <int KV_MODE>
__global__ void __launch_bounds__(THREADS) decode_attn_wmma_kernel(const float* __restrict__ q, QsaAttnPools p,
                                                                   const int32_t* __restrict__ ids,
                                                                   const int32_t* __restrict__ step, int n_kv_heads,
                                                                   int page_size, float scale,
                                                                   float* __restrict__ part_acc,
                                                                   float* __restrict__ part_m,
                                                                   float* __restrict__ part_l, int n_chunks, int cap,
                                                                   long long stride) {
#if STRATA_DW_GFX11
    constexpr bool K_Q4 = KV_MODE == 2, V_Q4 = KV_MODE != 1;
    __shared__ Smem S;
    q += (size_t) blockIdx.z * (size_t) (n_kv_heads * G) * HD;
    ids += (size_t) blockIdx.z * (size_t) cap;
    step += (size_t) blockIdx.z * kStepCount;
    part_acc += (size_t) blockIdx.z * (size_t) stride;
    part_m += (size_t) blockIdx.z * (size_t) stride;
    part_l += (size_t) blockIdx.z * (size_t) stride;
    const int n_ids = __ldg(step + kStepWidth);
    const int chunk = blockIdx.x, kvh = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const int col = lane & 15, half = lane >> 4;
    const int c0 = chunk * CHUNK;
    const int n_here = min(CHUNK, n_ids - c0);
    const int slot = kvh * n_chunks + chunk;
    if (n_here <= 0) {   // the FP32 kernel's exit: the merge skips this chunk
        if (t < G) { part_m[slot * G + t] = -FLT_MAX; part_l[slot * G + t] = 0.0f; }
        return;
    }
    q += (size_t) kvh * G * HD;

    // the chunk's pool rows, and |q| max for the FP16 split's exponent
    if (t < CHUNK) {
        long long r = -1;
        if (t < n_here) {
            const int cell = __ldg(ids + c0 + t);
            const long long page = (long long) __ldg(p.page_table + cell / page_size);
            if (page >= 0) r = (page * n_kv_heads + kvh) * page_size + (cell % page_size);
        }
        S.row[t] = r;
    }
    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q[i]));
    qm = wmax(qm);
    if (lane == 0) S.qmax[warp] = qm;
    __syncthreads();
    qm = 0.0f;
#pragma unroll
    for (int w = 0; w < 8; ++w) qm = fmaxf(qm, S.qmax[w]);
    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);
    const float qup = ldexpf(1.0f, 14 - qe), qdown = ldexpf(scale, qe - 14);

    // stage V (coalesced 16-byte loads) and its scales; masked cells get zero codes and scale 0
    {
        constexpr int VEC = V_Q4 ? Q4ROW / 16 : HD / 16;   // 9 or 16 uint4 a row
        const uint8_t* vbase = V_Q4 ? p.v_q4 : reinterpret_cast<const uint8_t*>(p.v_q);
        for (int i = t; i < CHUNK * VEC; i += THREADS) {
            const int c = i / VEC, j = i % VEC;
            const long long r = S.row[c];
            uint4 x = make_uint4(0, 0, 0, 0);
            if (r >= 0) x = __ldg(reinterpret_cast<const uint4*>(vbase + r * (V_Q4 ? Q4ROW : HD)) + j);
            *reinterpret_cast<uint4*>(&S.v[c][j * 16]) = x;
        }
        constexpr int NG = V_Q4 ? HD / QK4_0 : HD / KV_Q8_GROUP;   // 8 or 4 scales a row
        for (int i = t; i < CHUNK * NG; i += THREADS) {
            const int c = i / NG, g = i % NG;
            const long long r = S.row[c];
            float sc = 0.0f;
            if (r >= 0) {
                const uint16_t bits = V_Q4 ? __ldg(reinterpret_cast<const uint16_t*>(p.v_q4 + r * Q4ROW + g * Q4B))
                                           : __ldg(p.v_scale + r * (HD / KV_Q8_GROUP) + g);
                sc = __half2float(__ushort_as_half(bits));
            }
            S.vs[g][c] = sc;
        }
    }

    // q.k: wave w takes dim group g = w & 3 (64 dims, 4 k-steps) and the cell tiles 2 * (w >> 2) + {0, 1}
    {
        const int g = warp & 3;
        h16 qh[4], ql[4];
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            float x[16];
            if (col < G) {
                const float4* src = reinterpret_cast<const float4*>(q + (size_t) col * HD + g * 64 + kk * 16);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float4 a = src[j];
                    x[4 * j] = a.x; x[4 * j + 1] = a.y; x[4 * j + 2] = a.z; x[4 * j + 3] = a.w;
                }
            } else {
#pragma unroll
                for (int i = 0; i < 16; ++i) x[i] = 0.0f;
            }
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float v = x[i] * qup;
                const _Float16 hi = (_Float16) v;
                qh[kk][i] = hi;
                ql[kk][i] = (_Float16) (v - (float) hi);
            }
        }
#pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int nt = 2 * (warp >> 2) + u;
            const long long r = S.row[nt * 16 + col];
            f8 sa = f8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, sb = sa;
            float ka = 0.0f, kb = 0.0f;   // K scales: int8 one per group (ka); q4_0 one per 32-dim block (ka, kb)
#pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                h16 b;
                if (r < 0) {
#pragma unroll
                    for (int i = 0; i < 16; ++i) b[i] = (_Float16) 0.0f;
                } else if constexpr (K_Q4) {
                    const int d0 = g * 64 + kk * 16, blk = d0 / QK4_0;
                    const uint8_t* bp = p.k_q4 + r * Q4ROW + blk * Q4B;
                    const uint16_t* qs = reinterpret_cast<const uint16_t*>(bp + 2);   // 2-byte aligned
                    const int sh = (d0 % QK4_0) ? 4 : 0;   // dims 16..31 of a block are the high nibbles
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const uint16_t w2 = __ldg(qs + i);
                        b[2 * i] = (_Float16) (int) (((w2 >> sh) & 0xF) - 8);
                        b[2 * i + 1] = (_Float16) (int) (((w2 >> (8 + sh)) & 0xF) - 8);
                    }
                    if ((kk & 1) == 0) {
                        const float d = __half2float(__ushort_as_half(__ldg(reinterpret_cast<const uint16_t*>(bp))));
                        if (kk == 0) ka = d; else kb = d;
                    }
                } else {
                    const uint4 x = __ldg(reinterpret_cast<const uint4*>(p.k_q + r * HD + g * 64 + kk * 16));
                    const uint32_t w4[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
                    for (int i = 0; i < 16; ++i) b[i] = (_Float16) (int) (int8_t) (w4[i / 4] >> (8 * (i % 4)));
                    if (kk == 0) ka = __half2float(__ushort_as_half(__ldg(p.k_scale + r * (HD / KV_Q8_GROUP) + g)));
                }
                if (K_Q4 && kk >= 2) {
                    sb = wmma(qh[kk], b, sb);
                    sb = wmma(ql[kk], b, sb);
                } else {
                    sa = wmma(qh[kk], b, sa);
                    sa = wmma(ql[kk], b, sa);
                }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = 2 * i + half;
                if (row < G) S.part[g][row][nt * 16 + col] = K_Q4 ? fmaf(sa[i], ka, sb[i] * kb) : sa[i] * ka;
            }
        }
    }
    __syncthreads();

    // per-head chunk max and exp-sum (natural units, as attn_chunk_kernel): wave w takes heads w and w + 8
    for (int h = warp; h < G; h += 8) {
        float x[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int c = lane + 32 * j;
            x[j] = (c < n_here && S.row[c] >= 0)
                       ? (((S.part[0][h][c] + S.part[1][h][c]) + S.part[2][h][c]) + S.part[3][h][c]) * qdown
                       : -FLT_MAX;
        }
        const float m = wmax(fmaxf(x[0], x[1]));
        const float ea = (lane < n_here && S.row[lane] >= 0) ? __expf(x[0] - m) : 0.0f;
        const float eb = (lane + 32 < n_here && S.row[lane + 32] >= 0) ? __expf(x[1] - m) : 0.0f;
        S.p[h][lane] = ea;
        S.p[h][lane + 32] = eb;
        const float l = wsum(ea + eb);
        if (lane == 0) { part_m[slot * G + h] = m; part_l[slot * G + h] = l; }
    }
    __syncthreads();

    // p.V: wave w owns dims [32w, 32w + 32) (two 16-dim tiles); its V scale is int8 group w / 2 or q4_0 block w.
    // The scale rides in the A operand (p * vscale, per cell), the codes enter B exactly.
    {
        const int sg = V_Q4 ? warp : warp >> 1;
        float vmax = fmaxf(S.vs[sg][lane], S.vs[sg][lane + 32]);
        vmax = wmax(vmax);
        const float vup = vmax > 0.0f ? 16384.0f / vmax : 0.0f, vdown = vmax * (1.0f / 16384.0f);
        f8 acc[2];
        acc[0] = acc[1] = f8{0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
            const int cb = ks * 16;
            h16 ah, al;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float pv = (col < G ? S.p[col][cb + i] : 0.0f) * (S.vs[sg][cb + i] * vup);
                const _Float16 hi = (_Float16) pv;
                ah[i] = hi;
                al[i] = (_Float16) (pv - (float) hi);
            }
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int d = warp * 32 + j * 16 + col;
                h16 b;
                if constexpr (V_Q4) {
                    const int off = (d / QK4_0) * Q4B + 2 + (d % 16);
                    const int sh = (d % QK4_0) >= 16 ? 4 : 0;
#pragma unroll
                    for (int i = 0; i < 16; ++i) b[i] = (_Float16) (int) (((S.v[cb + i][off] >> sh) & 0xF) - 8);
                } else {
#pragma unroll
                    for (int i = 0; i < 16; ++i) b[i] = (_Float16) (int) (int8_t) S.v[cb + i][d];
                }
                acc[j] = wmma(ah, b, acc[j]);
                acc[j] = wmma(al, b, acc[j]);
            }
        }
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = 2 * i + half;
                if (row < G) part_acc[((size_t) slot * G + row) * HD + warp * 32 + j * 16 + col] = acc[j][i] * vdown;
            }
    }
#else
    __builtin_trap();
#endif
}

std::atomic<bool> g_failed{false};

}  // namespace

bool qsa_decode_attn_wmma_on() {
    static const bool env = [] { const char* v = std::getenv("STRATA_ATTN_WMMA"); return v && v[0] == '1'; }();
    if (!env || g_failed.load(std::memory_order_relaxed)) return false;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return false;
    static std::atomic<int> cls[16];   // (zero: static storage) 0 unknown, 1 gfx11 matrix cores, 2 other
    int c = cls[dev].load(std::memory_order_relaxed);
    if (c == 0) {
        cudaDeviceProp prop{};
        c = (cudaGetDeviceProperties(&prop, dev) == cudaSuccess && gfx_arch_is_gfx11_wmma(prop.gcnArchName)) ? 1 : 2;
        cls[dev].store(c, std::memory_order_relaxed);
        static std::atomic<bool> said{false};
        if (!said.exchange(true))
            std::fprintf(stderr,
                         "strata: STRATA_ATTN_WMMA=1 - decode attention chunks on %s matrix cores (BitDecoding-style; "
                         "FP32-level, not bitwise equal to the default kernel)%s\n",
                         prop.gcnArchName, c == 1 ? "" : ": not a gfx11 part, ignored");
    }
    return c == 1;
}

bool qsa_decode_attn_wmma_chunks(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                                 int64_t cap, const QsaShapes& s, float* scratch, int64_t n_q, void* stream) {
    if (n_q <= 0) return true;
    if (s.head_dim != HD || s.n_head != (int64_t) G * s.n_head_kv || cap <= 0 || !scratch || !q || !ids || !steps ||
        !pools.page_table || n_q > 65535)
        return false;
    const int kv_mode = pools.k_q4 != nullptr ? 2 : (pools.k_q != nullptr && pools.v_q4 != nullptr ? 3
                        : (pools.k_q != nullptr ? 1 : 0));
    if (kv_mode == 0) return false;
    if (kv_mode == 1 && (!pools.v_q || !pools.k_scale || !pools.v_scale)) return false;
    if (kv_mode == 2 && !pools.v_q4) return false;
    if (kv_mode == 3 && !pools.k_scale) return false;
    // the kernel reads K/V rows with 16-byte loads (int8 rows of 256 B, q4_0 rows of 144 B)
    auto mis = [](const void* x) { return x != nullptr && ((uintptr_t) x & 15) != 0; };
    if (mis(pools.k_q) || mis(pools.v_q) || mis(pools.k_q4) || mis(pools.v_q4)) return false;
    const int n_chunks = (int) ((cap + CHUNK - 1) / CHUNK);
    const long long stride = (long long) qsa_decode_attn_scratch_floats(cap, s);
    float* part_acc = scratch;
    float* part_m = scratch + (size_t) n_chunks * s.n_head * HD;
    float* part_l = part_m + (size_t) n_chunks * s.n_head;
    const float scale = 1.0f / sqrtf((float) HD);
    const dim3 grid((unsigned) n_chunks, (unsigned) s.n_head_kv, (unsigned) n_q);
    cudaStream_t st = (cudaStream_t) stream;
#define STRATA_DW_L(M) decode_attn_wmma_kernel<M><<<grid, THREADS, 0, st>>>(q, pools, ids, steps, (int) s.n_head_kv, \
        (int) s.page_size, scale, part_acc, part_m, part_l, n_chunks, (int) cap, stride)
    if (kv_mode == 3) STRATA_DW_L(3);
    else if (kv_mode == 2) STRATA_DW_L(2);
    else STRATA_DW_L(1);
#undef STRATA_DW_L
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_decode_attn (wmma): %s; the FP32 decode attention runs from here on\n",
                     cudaGetErrorString(e));
        g_failed.store(true);
        return false;
    }
    return true;
}

#else   // CUDA (and any non-HIP build): the gfx11 kernel does not exist

bool qsa_decode_attn_wmma_on() { return false; }
bool qsa_decode_attn_wmma_chunks(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t,
                                 const QsaShapes&, float*, int64_t, void*) {
    return false;
}

#endif

}  // namespace strata::kernels
