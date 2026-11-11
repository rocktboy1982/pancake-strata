// src/kernels/cuda/bfla.cu - BFLA block-sparse prompt attention (pancake, opt-in, lossy).  See include/strata/kernels/bfla.hpp
// and docs/BFLA_RDNA3.md.  BFLA: Chong Wu et al., arXiv 2605.12193 (CityU); reference code Apache-2.0
// (github.com/Alicewithrabbit/BFLA), read for the algorithm only - this file is written from the paper's equations.
//
// HIP wave32 gfx11 only (v_wmma_f32_16x16x16_f16 with the gfx11 fragment layout, documented and checked on the card in
// qsa_prompt_attn.cu S6).  The CUDA and gfx906 builds compile the stubs at the end: every entry point returns false.
#include "strata/kernels/bfla.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

namespace strata::kernels {

const BflaConfig& bfla_config() {
    static const BflaConfig c = [] {
        BflaConfig r;
        if (const char* e = std::getenv("STRATA_BFLA")) {
            const std::string v(e);
            if (v == "1" || v == "sparse" || v == "on") r.mode = 1;
            else if (v == "dense" || v == "2") r.mode = 2;
        }
        auto num = [](const char* name, auto& dst) {
            if (const char* e = std::getenv(name); e != nullptr && *e != '\0') dst = (std::remove_reference_t<decltype(dst)>) std::atof(e);
        };
        num("STRATA_BFLA_MIN", r.min_len);
        num("STRATA_BFLA_ETA", r.eta);
        num("STRATA_BFLA_KMIN", r.kmin);
        num("STRATA_BFLA_KMAX", r.kmax);
        num("STRATA_BFLA_SINK", r.sink);
        num("STRATA_BFLA_LOCAL", r.local);
        num("STRATA_BFLA_TAIL", r.tail);
        if (r.eta <= 0.0 || r.eta > 1.0) r.eta = 1.0 / 3.0;
        r.kmin = std::max<int64_t>(r.kmin, kBflaBlock);
        r.kmax = std::max<int64_t>(r.kmax, r.kmin);
        r.sink = std::max(r.sink, 1);   // block 0 keeps every row's softmax non-empty
        r.local = std::max(r.local, 0);
        if (r.mode != 0)
            std::fprintf(stderr,
                         "strata: STRATA_BFLA=%s - %s prompt attention on the dense attention layers above %lld tokens%s "
                         "(eta %.3f, K %lld..%lld, sink %d, local %d blocks of %d)\n",
                         r.mode == 1 ? "sparse" : "dense", r.mode == 1 ? "LOSSY block-sparse (BFLA)" : "fused (every block)",
                         (long long) r.min_len, r.mode == 1 ? ": answers can change" : "", r.eta, (long long) r.kmin,
                         (long long) r.kmax, r.sink, r.local, kBflaBlock);
        return r;
    }();
    return c;
}

namespace {
inline int64_t bfla_tile_tokens(int G) { return kBflaTileRows / G; }
inline int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
struct Layout {
    size_t phi = 0, mu = 0, keys = 0, ids = 0, counts = 0, total = 0;
    int64_t nblk = 0, n_tiles = 0, cap = 0;
};
Layout layout(int64_t T, int64_t P, int H, int HKV) {
    Layout l;
    const int G = H / HKV;
    l.nblk = cdiv(P, kBflaBlock);
    l.cap = l.nblk;
    l.n_tiles = cdiv(T, bfla_tile_tokens(G));
    auto al = [](size_t x) { return (x + 255) & ~(size_t) 255; };
    size_t o = 0;
    l.phi = o;
    o = al(o + (size_t) T * HKV * kBflaFeat * 4);
    l.mu = o;
    o = al(o + (size_t) HKV * l.nblk * kBflaFeat * 4);
    l.keys = o;
    o = al(o + (size_t) l.n_tiles * HKV * l.cap * 4);
    l.ids = o;
    o = al(o + (size_t) l.n_tiles * HKV * l.cap * 4);
    l.counts = o;
    o = al(o + (size_t) l.n_tiles * HKV * 4);
    l.total = o;
    return l;
}
[[maybe_unused]] bool geometry_ok(int H, int HKV, int HD) {
    if (HD != kBflaDim || HKV <= 0 || H <= 0 || H % HKV != 0) return false;
    const int G = H / HKV;
    return G == 4 || G == 8 || G == 16;
}
}  // namespace

size_t bfla_scratch_bytes(int64_t T, int64_t P, int H, int HKV) {
    if (T <= 0 || P <= 0 || HKV <= 0 || H % HKV != 0) return 0;
    return layout(T, P, H, HKV).total;
}

BflaLists bfla_lists(const void* scratch, int64_t T, int64_t P, int H, int HKV) {
    BflaLists r;
    if (scratch == nullptr || T <= 0 || P <= 0 || HKV <= 0 || H % HKV != 0) return r;
    const Layout l = layout(T, P, H, HKV);
    const uint8_t* b = static_cast<const uint8_t*>(scratch);
    r.ids = reinterpret_cast<const int32_t*>(b + l.ids);
    r.counts = reinterpret_cast<const int32_t*>(b + l.counts);
    r.cap = l.cap;
    r.n_tiles = l.n_tiles;
    r.tile_tokens = bfla_tile_tokens(H / HKV);
    return r;
}

}  // namespace strata::kernels

#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906)
#include "strata/kernels/gfx_arch.hpp"

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || \
    defined(__gfx1151__) || defined(__gfx1152__)
#define STRATA_BFLA_GFX11 1
#else
#define STRATA_BFLA_GFX11 0
#endif

namespace strata::kernels {
namespace {

typedef _Float16 bh8 __attribute__((ext_vector_type(8)));
typedef _Float16 bh16 __attribute__((ext_vector_type(16)));
typedef float bf8 __attribute__((ext_vector_type(8)));

constexpr float kEps = 1e-6f;
constexpr int QS = kBflaDim + 8;   // LDS row stride of Q and K tiles in halves (528 B: 16-byte aligned, banks spread)
constexpr int VS = 16 + 8;         // LDS row stride of the transposed V tile (one dim's 16 keys)
constexpr int PS = 16 + 8;         // LDS row stride of a wave's probability tile

__device__ __forceinline__ bf8 wmma16(bh16 a, bh16 b, bf8 c) {
#if STRATA_BFLA_GFX11
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    __builtin_trap();
    return c;
#endif
}
__device__ __forceinline__ bh16 lds16(const _Float16* p) {   // 16 halves from two 16-byte reads
    const bh8 lo = *reinterpret_cast<const bh8*>(p);
    const bh8 hi = *reinterpret_cast<const bh8*>(p + 8);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
__device__ __forceinline__ float wave_sum(float x) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor(x, o, 32);
    return x;
}
__device__ __forceinline__ float half_max(float x) {   // over the 16 lanes of a half wave
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor(x, o, 32));
    return x;
}
__device__ __forceinline__ float half_sum(float x) {
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) x += __shfl_xor(x, o, 32);
    return x;
}
__device__ __forceinline__ uint32_t order_key(float s) {   // larger float -> larger key
    const uint32_t u = __float_as_uint(s);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// ---- features.  phi[t][g][128]: lane l owns dims 8l .. 8l+7 of each head, i.e. feature channels 4l .. 4l+3.
__global__ void __launch_bounds__(32) k_phi(const _Float16* __restrict__ q16, int H, int HKV, float* __restrict__ phi) {
    const int t = blockIdx.x, g = blockIdx.y, lane = threadIdx.x;
    const int G = H / HKV;
    float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int hh = 0; hh < G; ++hh) {
        const bh8 x = *reinterpret_cast<const bh8*>(q16 + ((size_t) t * H + g * G + hh) * kBflaDim + lane * 8);
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) ss += (float) x[i] * (float) x[i];
        const float inv = rsqrtf(wave_sum(ss) / (float) kBflaDim + kEps);
#pragma unroll
        for (int i = 0; i < 8; ++i) a[i] += (float) x[i] * inv;
    }
    float r[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        r[i] = (a[2 * i] + a[2 * i + 1]) / (float) G;
        ss += r[i] * r[i];
    }
    const float inv = rsqrtf(wave_sum(ss) / (float) kBflaFeat + kEps);
    float* o = phi + ((size_t) t * HKV + g) * kBflaFeat + lane * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) o[i] = r[i] * inv;
}
// mu[g][j][128]: the mean over block j's cells of the same transform of K.  grid (nblk, HKV), block 128 (4 waves).
__global__ void __launch_bounds__(128) k_mu(const _Float16* __restrict__ k16, int64_t P, float* __restrict__ mu, int64_t nblk) {
    __shared__ float sh[4][kBflaFeat];
    const int j = blockIdx.x, g = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int64_t c0 = (int64_t) j * kBflaBlock;
    const int n = (int) std::min<int64_t>(kBflaBlock, P - c0);
    float acc[4] = {0, 0, 0, 0};
    for (int c = w; c < n; c += 4) {
        const bh8 x = *reinterpret_cast<const bh8*>(k16 + ((size_t) g * P + c0 + c) * kBflaDim + lane * 8);
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 8; ++i) ss += (float) x[i] * (float) x[i];
        const float inv = rsqrtf(wave_sum(ss) / (float) kBflaDim + kEps);
        float r[4], s2 = 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            r[i] = ((float) x[2 * i] + (float) x[2 * i + 1]) * inv;
            s2 += r[i] * r[i];
        }
        const float inv2 = rsqrtf(wave_sum(s2) / (float) kBflaFeat + kEps);
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[i] += r[i] * inv2;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) sh[w][lane * 4 + i] = acc[i];
    __syncthreads();
    const int f = threadIdx.x;
    mu[((size_t) g * nblk + j) * kBflaFeat + f] = (sh[0][f] + sh[1][f] + sh[2][f] + sh[3][f]) / (float) n;
}

// ---- 256-thread block helpers (8 waves)
__device__ __forceinline__ int block_count(int v, int* sh) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    int s = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) s += sh[i];
    return s;
}
__device__ __forceinline__ int block_excl_scan(int v, int* sh, int& total) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up(x, o, 32);
        if (lane >= o) x += y;
    }
    __syncthreads();
    if (lane == 31) sh[w] = x;
    __syncthreads();
    int off = 0;
    total = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        off += i < w ? sh[i] : 0;
        total += sh[i];
    }
    return off + x - v;
}

struct SelArgs {
    int64_t T, p0, nblk, cap, full_from;
    int H, HKV, tile_tokens, sink, local;
    double eta;
    int64_t kmin, kmax;
};

// ---- selection.  grid (n_tiles, HKV), block 256.
__global__ void __launch_bounds__(256) k_select(const float* __restrict__ phi, const float* __restrict__ mu, SelArgs a,
                                                uint32_t* __restrict__ keys, int32_t* __restrict__ ids, int32_t* __restrict__ counts) {
    __shared__ float4 ph[32 * kBflaFeat / 4];
    __shared__ int red[8];
    const int tile = blockIdx.x, g = blockIdx.y, tid = threadIdx.x;
    const int64_t ta = (int64_t) tile * a.tile_tokens;
    const int64_t tb = std::min<int64_t>(a.T, ta + a.tile_tokens) - 1;
    const int64_t qa = a.p0 + ta, qb = a.p0 + tb;
    const int n_vis = (int) (qb / kBflaBlock) + 1;
    const size_t row = (size_t) tile * a.HKV + g;
    int32_t* out = ids + row * a.cap;
    // the Ratio-K budget (bfla_budget_blocks)
    const double ev = a.eta * (double) (qb + 1);
    int64_t e = (int64_t) ev;
    if ((double) e < ev) ++e;
    int64_t kt = std::min<int64_t>(e, a.kmax);
    kt = std::max<int64_t>(kt, a.kmin);
    kt = std::min<int64_t>(kt, qb + 1);
    const int64_t k = (kt + kBflaBlock - 1) / kBflaBlock;
    const int sinkc = std::min(a.sink, n_vis);
    const int lo_b = std::max<int>(sinkc, (int) (qa / kBflaBlock) - a.local);   // [lo_b, n_vis) mandatory, own + local
    const int nc = lo_b - sinkc;                                                   // candidates [sinkc, lo_b)
    const int64_t free_q = k - (int64_t) (n_vis - nc);
    if (qb >= a.full_from || k >= n_vis || free_q >= nc) {
        for (int j = tid; j < n_vis; j += 256) out[j] = j;
        if (tid == 0) counts[row] = n_vis;
        return;
    }
    if (free_q <= 0) {   // only the mandatory blocks fit the quota
        const int n_tail = n_vis - lo_b;
        for (int j = tid; j < sinkc; j += 256) out[j] = j;
        for (int j = tid; j < n_tail; j += 256) out[sinkc + j] = lo_b + j;
        if (tid == 0) counts[row] = sinkc + n_tail;
        return;
    }
    const int need = (int) free_q;
    // the tile's query features into LDS
    const int nrows = (int) (tb - ta + 1);
    for (int i = tid; i < nrows * kBflaFeat / 4; i += 256)
        ph[i] = reinterpret_cast<const float4*>(phi + ((size_t) (ta + i / (kBflaFeat / 4)) * a.HKV + g) * kBflaFeat)[i % (kBflaFeat / 4)];
    __syncthreads();
    uint32_t* key = keys + row * a.cap;
    const float rs = rsqrtf((float) kBflaFeat);
    for (int j = sinkc + tid; j < lo_b; j += 256) {
        float4 m4[kBflaFeat / 4];
        const float4* mp = reinterpret_cast<const float4*>(mu + ((size_t) g * a.nblk + j) * kBflaFeat);
#pragma unroll
        for (int c = 0; c < kBflaFeat / 4; ++c) m4[c] = mp[c];
        float mx = -INFINITY, sum = 0.f;
        for (int t = 0; t < nrows; ++t) {
            float d = 0.f;
#pragma unroll
            for (int c = 0; c < kBflaFeat / 4; ++c) {
                const float4 p = ph[t * (kBflaFeat / 4) + c];
                d += p.x * m4[c].x + p.y * m4[c].y + p.z * m4[c].z + p.w * m4[c].w;
            }
            const float u = fmaxf(d, 0.f) * rs;
            if (u > mx) {
                sum = sum * __expf(mx - u) + 1.f;
                mx = u;
            } else {
                sum += __expf(u - mx);
            }
        }
        key[j] = order_key(mx + __logf(sum));
    }
    __syncthreads();
    // the need-th largest key: the largest x with #{key >= x} >= need
    uint32_t x = 0;
    for (int bit = 31; bit >= 0; --bit) {
        const uint32_t cand = x | (1u << bit);
        int c = 0;
        for (int j = sinkc + tid; j < lo_b; j += 256) c += key[j] >= cand;
        if (block_count(c, red) >= need) x = cand;
    }
    int cgt = 0;
    for (int j = sinkc + tid; j < lo_b; j += 256) cgt += key[j] > x;
    const int need_eq = need - block_count(cgt, red);   // >= 1 by the choice of x
    // ascending emission: mandatory, above x, and the first need_eq equal to x
    int pos = 0, eq_seen = 0;
    for (int j0 = 0; j0 < n_vis; j0 += 256) {
        const int j = j0 + tid;
        const bool cand = j >= sinkc && j < lo_b;
        const uint32_t kj = cand ? key[j] : 0u;
        const bool is_eq = cand && kj == x;
        int eq_total = 0;
        const int eq_rank = eq_seen + block_excl_scan(is_eq ? 1 : 0, red, eq_total);
        const bool keep = j < n_vis && (!cand || kj > x || (is_eq && eq_rank < need_eq));
        int keep_total = 0;
        const int at = pos + block_excl_scan(keep ? 1 : 0, red, keep_total);
        if (keep) out[at] = j;
        pos += keep_total;
        eq_seen += eq_total;
    }
    if (tid == 0) counts[row] = pos;
}

// ---- block-sparse flash attention on the gfx11 matrix cores.  grid (ceil(T / (64/G)), HKV), block 128 (4 waves).
// Rows r = 0..63 of a workgroup are (token t0 + r / G, head g*G + r % G); wave w owns rows 16w .. 16w+15 and all 256
// dims of their output (16 C tiles).  ids == nullptr: every causal block (dense).
// QL = false (STRATA_BFLA_QGLOBAL=1, experiment): Q fragments are read from global memory (L0/L1) every step instead of
// being staged in LDS: 24 KB of LDS a workgroup instead of 57.6 KB, so ~5 waves a SIMD can be resident instead of 2.
template <int G, bool QL>
__global__ void __launch_bounds__(128) k_attn(const _Float16* __restrict__ q16, const _Float16* __restrict__ k16,
                                              const _Float16* __restrict__ v16, int T, int64_t p0, int H, int HKV, int64_t P,
                                              const int32_t* __restrict__ ids, const int32_t* __restrict__ counts, int64_t cap,
                                              float scale_log2, float* __restrict__ out) {
    constexpr int WT = 64 / G, TT = kBflaTileRows / G;
    __shared__ __attribute__((aligned(16))) _Float16 Qs[QL ? 64 * QS : 8];
    __shared__ __attribute__((aligned(16))) _Float16 Ks[16 * QS];
    __shared__ __attribute__((aligned(16))) _Float16 Vt[kBflaDim * VS];
    __shared__ __attribute__((aligned(16))) _Float16 Ps[4][16 * PS];
    const int g = blockIdx.y, tid = threadIdx.x, lane = tid & 31, w = tid >> 5, hl = lane >> 4, ln = lane & 15;
    const int t0 = blockIdx.x * WT;
    const int tlast = min(T, t0 + WT) - 1;
    const int qa = t0, qb = tlast;   // chunk-local; key positions are compared as (cell - p0)
    if constexpr (QL) {
        for (int i = tid; i < 64 * (kBflaDim / 8); i += 128) {
            const int r = i / (kBflaDim / 8), c = (i % (kBflaDim / 8)) * 8;
            const int t = min(t0 + r / G, T - 1);
            *reinterpret_cast<bh8*>(&Qs[r * QS + c]) =
                *reinterpret_cast<const bh8*>(q16 + ((size_t) t * H + g * G + r % G) * kBflaDim + c);
        }
    }
    const int my_row = 16 * w + ln;
    const _Float16* qg = q16 + ((size_t) min(t0 + my_row / G, T - 1) * H + g * G + my_row % G) * kBflaDim;
    int qrel[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) qrel[i] = min(t0 + (16 * w + 2 * i + hl) / G, tlast);
    bf8 O[16];
#pragma unroll
    for (int nt = 0; nt < 16; ++nt) O[nt] = bf8{0, 0, 0, 0, 0, 0, 0, 0};
    float mrow[8], lrow[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        mrow[i] = -INFINITY;
        lrow[i] = 0.f;
    }
    const int64_t qb_abs = p0 + qb;
    const int tile = t0 / TT;
    const int n = ids ? counts[(size_t) tile * HKV + g] : (int) (qb_abs / kBflaBlock) + 1;
    const int32_t* lst = ids ? ids + ((size_t) tile * HKV + g) * cap : nullptr;
    const _Float16* kb0 = k16 + (size_t) g * P * kBflaDim;
    const _Float16* vb0 = v16 + (size_t) g * P * kBflaDim;
    const bh8 zero8 = bh8{0, 0, 0, 0, 0, 0, 0, 0};
    for (int bi = 0; bi < n; ++bi) {
        const int64_t j = lst ? (int64_t) lst[bi] : (int64_t) bi;
        const int64_t ks = j * kBflaBlock;
        if (ks > qb_abs) break;   // ascending lists: every later block is past this workgroup's last query
        const bool need_mask = ks + kBflaBlock - 1 > p0 + qa;
        for (int st = 0; st < kBflaBlock / 16; ++st) {
            const int64_t kb = ks + st * 16;
            if (kb > qb_abs) break;
            __syncthreads();   // the previous step's K / V / P reads are done (and Q is staged)
            for (int i = tid; i < 16 * (kBflaDim / 8); i += 128) {
                const int r = i / (kBflaDim / 8), c = (i % (kBflaDim / 8)) * 8;
                const int64_t cell = kb + r;
                const bool ok = cell < P;
                const bh8 kv = ok ? *reinterpret_cast<const bh8*>(kb0 + (size_t) cell * kBflaDim + c) : zero8;
                const bh8 vv = ok ? *reinterpret_cast<const bh8*>(vb0 + (size_t) cell * kBflaDim + c) : zero8;
                *reinterpret_cast<bh8*>(&Ks[r * QS + c]) = kv;
#pragma unroll
                for (int e = 0; e < 8; ++e) Vt[(c + e) * VS + r] = vv[e];
            }
            __syncthreads();
            bf8 s = bf8{0, 0, 0, 0, 0, 0, 0, 0};
            const _Float16* qrow = QL ? &Qs[(16 * w + ln) * QS] : qg;
            const _Float16* krow = &Ks[ln * QS];
#pragma unroll
            for (int kk = 0; kk < kBflaDim / 16; ++kk) s = wmma16(lds16(qrow + kk * 16), lds16(krow + kk * 16), s);
            const int64_t cell = kb + ln;
            const int krel = (int) (cell - p0);
            float alpha[8];
            _Float16* ps = Ps[w];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                float v = s[i] * scale_log2;
                if (cell >= P || (need_mask && krel > qrel[i])) v = -INFINITY;
                const float mnew = fmaxf(mrow[i], half_max(v));
                const float p = v == -INFINITY ? 0.f : exp2f(v - mnew);
                alpha[i] = mnew == -INFINITY ? 1.f : exp2f(mrow[i] - mnew);
                lrow[i] = lrow[i] * alpha[i] + half_sum(p);
                mrow[i] = mnew;
                ps[(2 * i + hl) * PS + ln] = (_Float16) p;
            }
#pragma unroll
            for (int nt = 0; nt < 16; ++nt)
#pragma unroll
                for (int i = 0; i < 8; ++i) O[nt][i] *= alpha[i];
            __syncthreads();   // P is in LDS (every wave reaches this: the loop bounds are uniform)
            const bh16 pa = lds16(ps + ln * PS);
#pragma unroll
            for (int nt = 0; nt < 16; ++nt) O[nt] = wmma16(pa, lds16(&Vt[(nt * 16 + ln) * VS]), O[nt]);
        }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int r = 16 * w + 2 * i + hl;
        const int t = t0 + r / G;
        if (t > tlast) continue;
        const float inv = lrow[i] > 0.f ? 1.f / lrow[i] : 0.f;
        float* o = out + ((size_t) t * H + g * G + r % G) * kBflaDim + ln;
#pragma unroll
        for (int nt = 0; nt < 16; ++nt) o[nt * 16] = O[nt][i] * inv;
    }
}

int g_q_global = -1;   // -1: STRATA_BFLA_QGLOBAL decides
bool q_global() {
    if (g_q_global >= 0) return g_q_global != 0;
    static const bool v = [] { const char* e = std::getenv("STRATA_BFLA_QGLOBAL"); return e != nullptr && e[0] == '1'; }();
    return v;
}

bool device_gfx11() {
    static int cached[64];
    static bool init[64];
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) return false;
    if (!init[dev]) {
        cudaDeviceProp prop{};
        cached[dev] = cudaGetDeviceProperties(&prop, dev) == cudaSuccess && gfx_arch_is_gfx11_wmma(prop.gcnArchName);
        init[dev] = true;
    }
    return cached[dev] != 0;
}

bool launch_attn(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t p0, int H, int HKV,
                 const int32_t* ids, const int32_t* counts, int64_t cap, float* out, cudaStream_t st) {
    const int G = H / HKV;
    const float scale_log2 = 1.4426950408889634f / std::sqrt((float) kBflaDim);
    const auto* q = reinterpret_cast<const _Float16*>(q16);
    const auto* k = reinterpret_cast<const _Float16*>(k16);
    const auto* v = reinterpret_cast<const _Float16*>(v16);
    const int64_t P = p0 + T;
    const dim3 grid((unsigned) cdiv(T, 64 / G), (unsigned) HKV);
    const bool ql = !q_global();
#define BFLA_L(GG)                                                                                                       \
    (ql ? k_attn<GG, true><<<grid, 128, 0, st>>>(q, k, v, (int) T, p0, H, HKV, P, ids, counts, cap, scale_log2, out)        \
        : k_attn<GG, false><<<grid, 128, 0, st>>>(q, k, v, (int) T, p0, H, HKV, P, ids, counts, cap, scale_log2, out))
    switch (G) {
        case 4: BFLA_L(4); break;
        case 8: BFLA_L(8); break;
        case 16: BFLA_L(16); break;
        default: return false;
    }
#undef BFLA_L
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace

bool bfla_supported(int H, int HKV, int HD) { return geometry_ok(H, HKV, HD) && device_gfx11(); }

void bfla_set_q_global(int on) { g_q_global = on; }

bool bfla_attn_from_lists(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t p0, int H,
                          int HKV, const int32_t* ids, const int32_t* counts, int64_t cap, float* out, void* stream) {
    if (T <= 0 || p0 < 0 || p0 + T >= ((int64_t) 1 << 31) || !bfla_supported(H, HKV, kBflaDim)) return false;
    return launch_attn(q16, k16, v16, T, p0, H, HKV, ids, counts, cap, out, (cudaStream_t) stream);
}

bool bfla_prefill_attn(const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t T, int64_t p0, int H, int HKV,
                       const BflaConfig& c, int64_t full_from, void* scratch, size_t scratch_bytes, float* out, void* stream) {
    if (c.mode == 0 || T <= 0 || p0 < 0 || p0 + T >= ((int64_t) 1 << 31) || !bfla_supported(H, HKV, kBflaDim)) return false;
    cudaStream_t st = (cudaStream_t) stream;
    if (c.mode == 2) return launch_attn(q16, k16, v16, T, p0, H, HKV, nullptr, nullptr, 0, out, st);
    const int64_t P = p0 + T;
    const Layout l = layout(T, P, H, HKV);
    if (scratch == nullptr || scratch_bytes < l.total) return false;
    uint8_t* b = static_cast<uint8_t*>(scratch);
    float* phi = reinterpret_cast<float*>(b + l.phi);
    float* mu = reinterpret_cast<float*>(b + l.mu);
    uint32_t* keys = reinterpret_cast<uint32_t*>(b + l.keys);
    int32_t* ids = reinterpret_cast<int32_t*>(b + l.ids);
    int32_t* counts = reinterpret_cast<int32_t*>(b + l.counts);
    const int G = H / HKV;
    k_phi<<<dim3((unsigned) T, (unsigned) HKV), 32, 0, st>>>(reinterpret_cast<const _Float16*>(q16), H, HKV, phi);
    k_mu<<<dim3((unsigned) l.nblk, (unsigned) HKV), 128, 0, st>>>(reinterpret_cast<const _Float16*>(k16), P, mu, l.nblk);
    SelArgs a{};
    a.T = T;
    a.p0 = p0;
    a.nblk = l.nblk;
    a.cap = l.cap;
    a.full_from = full_from;
    a.H = H;
    a.HKV = HKV;
    a.tile_tokens = (int) bfla_tile_tokens(G);
    a.sink = c.sink;
    a.local = c.local;
    a.eta = c.eta;
    a.kmin = c.kmin;
    a.kmax = c.kmax;
    k_select<<<dim3((unsigned) l.n_tiles, (unsigned) HKV), 256, 0, st>>>(phi, mu, a, keys, ids, counts);
    if (cudaGetLastError() != cudaSuccess) return false;
    return launch_attn(q16, k16, v16, T, p0, H, HKV, ids, counts, l.cap, out, st);
}

}  // namespace strata::kernels
#else
// CUDA and gfx906 builds: no BFLA kernels; every caller keeps its own attention.
namespace strata::kernels {
bool bfla_supported(int, int, int) { return false; }
bool bfla_attn_from_lists(const uint16_t*, const uint16_t*, const uint16_t*, int64_t, int64_t, int, int, const int32_t*,
                          const int32_t*, int64_t, float*, void*) {
    return false;
}
bool bfla_prefill_attn(const uint16_t*, const uint16_t*, const uint16_t*, int64_t, int64_t, int, int, const BflaConfig&, int64_t,
                       void*, size_t, float*, void*) {
    return false;
}
void bfla_set_q_global(int) {}
}  // namespace strata::kernels
#endif
