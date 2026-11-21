// Included by kernels.cu right after gdn_wmma12.inc (it uses the gfx11 chunked kernels' constants, types and helpers: GW_*, gw_*).
//
// STRATA_GDN_WMMA_PREFILL=1 (LEADS.md Part S S5, gfx11 only): the chunked gated-delta-net recurrence of gdn_wmma_prep_kernel /
// gdn_wmma_scan_kernel with a chunk of 32 tokens (two 16 x 16 x 16 WMMA token tiles) instead of 16.  Same algebra (the file comment
// above gdn_wmma_prep_kernel in kernels.cu), same operand roundings (k, q, T, P, D and the state as an operand in FP16, FP32 accumulate
// and state), the output norm and the z gate fused; what changes:
//   * half the dependent steps, barriers and state -> B-operand conversions per prompt token;
//   * T = (I + A)^-1 over 32 tokens by FP32 forward substitution (a lane a column, 496 FMAs), not a blocked inverse; P and T are
//     applied tile by tile over the lower half only (token tile u takes tiles u2 <= u);
//   * v is staged as FP16 (it is rounded to FP16 before it enters Y anyway), the output rows go through an FP32 `so` tile and 16-byte stores;
//     LDS is 60,160 bytes (limit 65,536); chunk 64 would need 125,952 with the same buffers (include/strata/prefill/gdn_wmma_ck.hpp).
// THE KERNELS HAVE NEVER BEEN RUN (the PC that wrote them has no ROCm card).  They were compiled for gfx1100 with the ROCm 6.2 SDK clang 19
// (`--offload-arch=gfx1100 -O3`, kernel-resource-usage remarks): the prep kernel 120 VGPRs, no spill; the scan kernel uses all 256 VGPRs and
// SPILLS to scratch (the state alone is 64 and two token tiles double the products of the chunk-16 kernel, which needs 234): the ledger row
// of this lead has the numbers.  If the box A/B shows the spills cost more than the saved barriers, the way forward is the token tiles in
// sequence (half the live products, the state conversion repeated).  The host reference of the algorithm and the index math are tested
// (src/prefill/gdn_wmma_ck_test.cpp); the GPU parity is tests/hip/gdn_chunked.cpp --ck (ctest hip_gdn_chunked_ck).
//
// Memory layout and fragment conventions are those of the chunk-16 kernels (checked on gfx1100 there): an accumulator lane l holds
// column l & 15, rows 2v + (l >> 4); gw_acc_to_b turns it into the B operand (and an A operand: both are "lane l = row / column
// l & 15, 16 k values").
namespace gdn_ckc = ::strata::prefill::gdn_ck;
constexpr int GC_CK = gdn_ckc::kCk;      // 32 tokens a chunk
constexpr int GC_NT = gdn_ckc::kNt;      // 2 token tiles
constexpr int GC_AHEAD = 48;             // tokens ahead of the chunk whose rows are touched (a chunk and a half)
constexpr int GC_KP = gdn_ckc::kKP;      // FP16 k / q row pitch (= GW_KP)
constexpr int GC_KTP = gdn_ckc::kKTP;    // FP16 k^T row pitch
constexpr int GC_TPP = gdn_ckc::kTPP;    // FP16 T / P row pitch
static_assert(GC_KP == GW_KP && GW_NW == gdn_ckc::kNw && GW_VPK == HV / HK, "constants of the chunk-16 kernels");
static_assert(S == gdn_ckc::kS && HK == gdn_ckc::kHK && HV == gdn_ckc::kHV && C == gdn_ckc::kC, "shapes");

// A block per (key head, chunk) and a wave per value head of the key head.  T then P of the chunk (32 x 32 FP16 each, row-major, T
// lower triangular, the blocks above the diagonal zero) at tp[(chunk * HV + value head) * 2048].
__global__ void __launch_bounds__(32 * GW_VPK) gdn_wmma_ck_prep_kernel(const float* __restrict__ h, const float* __restrict__ gate,
                                                                        const float* __restrict__ beta, _Float16* __restrict__ tp,
                                                                        int64_t T) {
    __shared__ float sa[GW_VPK][GC_CK * GC_CK], sg[GW_VPK][GC_CK], sb[GW_VPK][GC_CK];
    const int qh = blockIdx.x, ch = blockIdx.y, w = threadIdx.x >> 5, l = threadIdx.x & 31, c = l & 15, hi = l >> 4;
    const int vh = qh + w * HK;
    const int64_t t0 = (int64_t) ch * GC_CK;
    gw_f8 kk[GC_NT][GC_NT], qk[GC_NT][GC_NT];   // [u][u2], u2 <= u: k_t.k_i and q_t.k_i (lane: column i = 16 u2 + c, rows t = 16 u + 2v + hi)
#pragma unroll
    for (int u = 0; u < GC_NT; ++u)
#pragma unroll
        for (int u2 = 0; u2 < GC_NT; ++u2) { kk[u][u2] = gw_f8{}; qk[u][u2] = gw_f8{}; }
    {
        const float* ht[GC_NT];
        bool ok[GC_NT];
#pragma unroll
        for (int u = 0; u < GC_NT; ++u) {
            ok[u] = t0 + 16 * u + c < T;   // past the end: zero rows
            ht[u] = h + (ok[u] ? t0 + 16 * u + c : 0) * C;
        }
#pragma unroll 1   // (unrolled, the compiler hoists the loads of every slice: 256 VGPRs and a spill, measured with the SDK compile)
        for (int j = 0; j < S / 16; ++j) {
            gw_h16 ka[GC_NT], qa[GC_NT];
#pragma unroll
            for (int u = 0; u < GC_NT; ++u) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const gw_f4 k4 = *reinterpret_cast<const gw_f4*>(ht[u] + HK * S + qh * S + 16 * j + 4 * i);
                    const gw_f4 q4 = *reinterpret_cast<const gw_f4*>(ht[u] + qh * S + 16 * j + 4 * i);
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        ka[u][4 * i + e] = (_Float16) (ok[u] ? k4[e] : 0.0f);
                        qa[u][4 * i + e] = (_Float16) (ok[u] ? q4[e] : 0.0f);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < GC_NT; ++u)
#pragma unroll
                for (int u2 = 0; u2 <= u; ++u2) {   // B = K^T: lane i holds column i of K^T, row i of K, the A operand's own layout
                    kk[u][u2] = gw_mma(ka[u], ka[u2], kk[u][u2]);
                    qk[u][u2] = gw_mma(qa[u], ka[u2], qk[u][u2]);
                }
        }
    }
    {   // lane l: token l (past the end: gate 0, beta 0); the inclusive scan of the gate over the 32 tokens (the whole wave)
        const bool ok = t0 + l < T;
        float g = ok ? gate[(t0 + l) * HV + vh] : 0.0f;
#pragma unroll
        for (int o = 1; o < GC_CK; o <<= 1) {
            const float u = __shfl_up(g, o, GC_CK);
            if (l >= o) g += u;
        }
        sg[w][l] = g;
        sb[w][l] = ok ? beta[(t0 + l) * HV + vh] : 0.0f;
    }
    gw_lds_wait();   // the wave's own LDS writes, read across its lanes
    _Float16* out = tp + ((int64_t) ch * HV + vh) * (2 * GC_CK * GC_CK);
#pragma unroll
    for (int u = 0; u < GC_NT; ++u)
#pragma unroll
        for (int u2 = 0; u2 < GC_NT; ++u2)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int t = 16 * u + 2 * v + hi, i = 16 * u2 + c;
                float av = 0.0f, pv = 0.0f;
                if (u2 <= u) {   // (the compile-time test: tiles above the diagonal stay zero)
                    av = i < t ? sb[w][t] * __expf(sg[w][t] - sg[w][i]) * kk[u][u2][v] : 0.0f;
                    pv = i <= t ? __expf(sg[w][t] - sg[w][i]) * qk[u][u2][v] : 0.0f;
                }
                sa[w][t * GC_CK + i] = av;
                out[GC_CK * GC_CK + t * GC_CK + i] = (_Float16) pv;
            }
    gw_lds_wait();
    {   // lane l: column l of T = (I + A)^-1 by forward substitution, FP32
        float tc[GC_CK];
#pragma unroll
        for (int t = 0; t < GC_CK; ++t) {
            float v = t == l ? 1.0f : 0.0f;
#pragma unroll
            for (int k = 0; k < t; ++k) v = fmaf(-sa[w][t * GC_CK + k], tc[k], v);
            tc[t] = v;
            out[t * GC_CK + l] = (_Float16) v;
        }
    }
}

// A thread's slice of a chunk in the scan: tokens st + 16u (st = threadIdx / 16, u < 2), columns sr .. sr + 7 (sr = (threadIdx % 16) * 8):
// k and q, loaded a chunk ahead; four words of the chunk's T and P; lanes 0..31 of wave 0 also gate and beta of token l.
// Rows past the end read the last row (finite: their gate and beta become 0 at staging, which keeps them out of the state, of the real
// rows' products and of the output).
struct GcChunk {
    float g, b;
    unsigned tp[4];
};
// The prefetch of the NEXT chunk is T / P / gate / beta only (6 registers): k, q and v of two token tiles are 48 registers, and with the
// state (64), the products and their operands the kernel needed over 256 VGPRs and spilled 425 (ROCm 6.2 SDK compile, gfx1100).  Their
// rows are brought into the cache a chunk and a half ahead (gc_touch) and loaded at staging.
__device__ __forceinline__ void gc_load(GcChunk& in, const float* __restrict__ gate, const float* __restrict__ beta,
                                        const unsigned* __restrict__ tp, int64_t t0, int64_t T, int vh) {
#pragma unroll
    for (int i = 0; i < 4; ++i)
        in.tp[i] = tp[((t0 / GC_CK) * HV + vh) * gdn_ckc::kTpWords + gdn_ckc::tp_word(threadIdx.x, i)];
    if (threadIdx.x < GC_CK) {
        const int64_t tl = std::min<int64_t>(t0 + threadIdx.x, T - 1);
        in.g = gate[tl * HV + vh];
        in.b = beta[tl * HV + vh];
    }
}
// One line (128 bytes) a thread of the rows of 16 tokens from t0 (k, q, v, z of this block's heads: 16 tokens x 16 parts), with a load whose
// result is consumed one chunk later by an empty asm (gw_touch of the chunk-16 kernels, the same pattern).
__device__ __forceinline__ unsigned gc_touch(const float* __restrict__ h, const float* __restrict__ z, int64_t t0, int64_t T, int vh, int qh) {
    const int t = threadIdx.x >> 4, part = threadIdx.x & 15, kind = part >> 2;
    if (t0 + t >= T) return 0u;
    const float* p = kind == 0 ? h + (t0 + t) * C + HK * S + qh * S
                   : kind == 1 ? h + (t0 + t) * C + qh * S
                   : kind == 2 ? h + (t0 + t) * C + 2 * HK * S + vh * S
                               : z + (t0 + t) * HV * S + vh * S;
    return __float_as_uint(p[32 * (part & 3)]);
}
// The chunk's z values in the same layout (read for the output rows after the norm).
__device__ __forceinline__ void gc_load_z(float (&zr)[GC_NT][8], const float* __restrict__ z, int64_t t0, int64_t T, int vh, int col,
                                          int hi) {
#pragma unroll
    for (int u = 0; u < GC_NT; ++u)
#pragma unroll
        for (int v = 0; v < 8; ++v) zr[u][v] = z[std::min<int64_t>(t0 + 16 * u + 2 * v + hi, T - 1) * HV * S + vh * S + col];
}

// Per chunk: stage it (k / q FP16, k also transposed, v FP16, the gate's running sum, beta, T and P); barrier (1); store the previous chunk's
// output rows (norm, z gate, 16 bytes a thread); X = K S0, QS = Q S0; Y, D, the state update, o and the wave's sums of o^2; barrier (2); o to
// LDS.  LDS is single-buffered except sn: an array written in the staging of chunk c + 1 was last read before every wave reached barrier (2)
// of c (sk, sq, skt, sv, stp: only the compute of c reads them); `so` is written after barrier (2) of c and read after barrier (1) of c + 1
// (the next write is after barrier (2) of c + 1); `sn` has two buffers (written in chunk c, read after barrier (1) of c + 1, while chunk
// c + 1's sums go to the other).
__global__ void __launch_bounds__(GW_NW * 32) gdn_wmma_ck_scan_kernel(float* __restrict__ state, const float* __restrict__ h,
                                                                       const float* __restrict__ gate, const float* __restrict__ beta,
                                                                       const float* __restrict__ z, const float* __restrict__ gamma,
                                                                       const unsigned* __restrict__ tp, float eps,
                                                                       uint16_t* __restrict__ y16, int64_t T, int64_t ld16) {
    __shared__ __attribute__((aligned(16))) _Float16 sk[GC_CK * GC_KP], sq[GC_CK * GC_KP], skt[S * GC_KTP], stp[2 * GC_CK * GC_TPP], sv[GC_CK * S];
    __shared__ __attribute__((aligned(16))) float so[GC_CK * S], sn[2][GC_CK * GW_NW];
    __shared__ float sg[GC_CK], sb[GC_CK], sgam[S];
    const _Float16 *st_ = stp, *sp_ = stp + GC_CK * GC_TPP;   // T, P
    const int vh = blockIdx.x, qh = vh % HK, w = threadIdx.x >> 5, l = threadIdx.x & 31, c = l & 15, hi = l >> 4;
    const int col = 16 * w + c;                                       // this lane's column in the products
    const int st = threadIdx.x >> 4, sr = (threadIdx.x & 15) * 8;    // this thread's staging / output slice
    const size_t rs = (size_t) HV * S;
    const GwSel ps = gw_sel(hi);
    float* sbase = state + (size_t) hi * rs + (size_t) vh * S + col;   // the state: element (16 j + 2 v) rows on, this lane's column
    gw_f8 s[S / 16];   // the state: rows 16j + 2v + hi, column col
#pragma unroll
    for (int j = 0; j < S / 16; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) s[j][v] = *(sbase + (size_t) (16 * j + 2 * v) * rs);
    if (threadIdx.x < S) sgam[threadIdx.x] = gamma[threadIdx.x];
    const float out_scale = rsqrtf((float) S);
    // Per-lane LDS base pointers; element offsets are compile-time constants (16 u + 2 v rows), so each address folds into the instruction's
    // immediate.  Written as t = 16 u + 2 v + hi arithmetic the compiler hoisted ~50 loop-invariant address registers out of the chunk loop and
    // spilled them (SDK compile of an earlier form of this kernel: 337 spilled VGPRs).
    const _Float16* svp = sv + hi * S + col;
    const float* sgp = sg + hi;
    const float* sbp = sb + hi;
    float* sop = so + hi * S + col;
    _Float16* skts = skt + sr * GC_KTP + st;
    const int tbase = gdn_ckc::tp_base(threadIdx.x);
    GcChunk cur;   // the next chunk's T / P / gate / beta land during the current one
    gc_load(cur, gate, beta, tp, 0, T, vh);
    unsigned touched0 = 0, touched1 = 0;
    gw_f4 zq[GC_NT][2] = {};   // the z slice (tokens st + 16 u, columns sr .. sr + 7) of the chunk whose output rows are stored next
    auto store_rows = [&](int64_t c0, int par) {   // the output rows of the chunk at c0: o from so, its sum of squares from sn[par]
#pragma unroll
        for (int u = 0; u < GC_NT; ++u) {
            const int t = st + 16 * u;
            if (c0 + t < T) {
                const gw_f4 n0 = *reinterpret_cast<const gw_f4*>(&sn[par][t * GW_NW]), n1 = *reinterpret_cast<const gw_f4*>(&sn[par][t * GW_NW + 4]);
                const float ss = n0[0] + n0[1] + n0[2] + n0[3] + n1[0] + n1[1] + n1[2] + n1[3];
                const float rn = rsqrtf(ss * (1.0f / S) + eps);
                const gw_f4 o0 = *reinterpret_cast<const gw_f4*>(&so[t * S + sr]), o1 = *reinterpret_cast<const gw_f4*>(&so[t * S + sr + 4]);
                gw_h8 r;
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    r[i] = (_Float16) ((i < 4 ? o0[i] : o1[i - 4]) * rn * sgam[sr + i] * sigm(i < 4 ? zq[u][0][i] : zq[u][1][i - 4]));
                *reinterpret_cast<uint4*>(y16 + (c0 + t) * ld16 + vh * S + sr) = __builtin_bit_cast(uint4, r);
            }
        }
    };
    for (int64_t t0 = 0, par = 0; t0 < T; t0 += GC_CK, par ^= 1) {
        {   // stage this chunk: the rows of k, q and v are loaded here (they were touched a chunk and a half ago)
#pragma unroll
            for (int u = 0; u < GC_NT; ++u) {
                const float* ht = h + std::min<int64_t>(t0 + st + 16 * u, T - 1) * C;
                const gw_f4 k0 = *reinterpret_cast<const gw_f4*>(ht + HK * S + qh * S + sr), k1 = *reinterpret_cast<const gw_f4*>(ht + HK * S + qh * S + sr + 4);
                const gw_f4 q0 = *reinterpret_cast<const gw_f4*>(ht + qh * S + sr), q1 = *reinterpret_cast<const gw_f4*>(ht + qh * S + sr + 4);
                const gw_f4 v0 = *reinterpret_cast<const gw_f4*>(ht + 2 * HK * S + vh * S + sr), v1 = *reinterpret_cast<const gw_f4*>(ht + 2 * HK * S + vh * S + sr + 4);
                gw_h8 kh, qq, vv;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    kh[i] = (_Float16) (i < 4 ? k0[i] : k1[i - 4]);
                    qq[i] = (_Float16) (i < 4 ? q0[i] : q1[i - 4]);
                    vv[i] = (_Float16) (i < 4 ? v0[i] : v1[i - 4]);
                    skts[i * GC_KTP + 16 * u] = kh[i];
                }
                *reinterpret_cast<uint4*>(&sk[(st + 16 * u) * GC_KP + sr]) = __builtin_bit_cast(uint4, kh);
                *reinterpret_cast<uint4*>(&sq[(st + 16 * u) * GC_KP + sr]) = __builtin_bit_cast(uint4, qq);
                *reinterpret_cast<uint4*>(&sv[(st + 16 * u) * S + sr]) = __builtin_bit_cast(uint4, vv);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
                *reinterpret_cast<unsigned*>(stp + tbase + gdn_ckc::tp_off(i)) = cur.tp[i];
            if (threadIdx.x < GC_CK) {   // past the end: gate 0, beta 0
                const bool ok = t0 + l < T;
                float g = ok ? cur.g : 0.0f;
#pragma unroll
                for (int o = 1; o < GC_CK; o <<= 1) {
                    const float u = __shfl_up(g, o, GC_CK);
                    if (l >= o) g += u;
                }
                sg[l] = g;
                sb[l] = ok ? cur.b : 0.0f;
            }
        }
        if (t0 + GC_CK < T) gc_load(cur, gate, beta, tp, t0 + GC_CK, T, vh);   // lands during this chunk
        const unsigned touch0 = gc_touch(h, z, t0 + GC_AHEAD, T, vh, qh), touch1 = gc_touch(h, z, t0 + GC_AHEAD + 16, T, vh, qh);
        gw_lds_barrier();   // (1) this chunk staged; the previous chunk's o in so and its sums in sn[par ^ 1]
        if (t0 > 0) store_rows(t0 - GC_CK, (int) par ^ 1);
        __asm__ volatile("" ::"v"(touched0), "v"(touched1));   // the previous touches, consumed here
        touched0 = touch0;
        touched1 = touch1;
        gw_f8 x[GC_NT], qs[GC_NT];
#pragma unroll
        for (int u = 0; u < GC_NT; ++u) { x[u] = gw_f8{}; qs[u] = gw_f8{}; }
#pragma unroll
        for (int j = 0; j < S / 16; ++j) {   // K S0 and Q S0 for both token tiles, the state operand converted once
            const gw_h16 b = gw_acc_to_b(s[j], ps);
#pragma unroll
            for (int u = 0; u < GC_NT; ++u) {
                x[u] = gw_mma(gw_ld(sk + gdn_ckc::off_k(u, c, j)), b, x[u]);
                qs[u] = gw_mma(gw_ld(sq + gdn_ckc::off_k(u, c, j)), b, qs[u]);
            }
        }
        gw_h16 yb[GC_NT];
#pragma unroll
        for (int u = 0; u < GC_NT; ++u) {
            gw_f8 y;
#pragma unroll
            for (int v = 0; v < 8; ++v)
                y[v] = sbp[16 * u + 2 * v] * ((float) svp[(16 * u + 2 * v) * S] - __expf(sgp[16 * u + 2 * v]) * x[u][v]);
            yb[u] = gw_acc_to_b(y, ps);
        }
        gw_f8 o[GC_NT];
        gw_h16 db[GC_NT], ddb[GC_NT];
        const float gl = sg[GC_CK - 1];
#pragma unroll
        for (int u = 0; u < GC_NT; ++u) {   // D = T Y over the lower token tiles; its B form for P D and its decayed B form for the state
            gw_f8 d = gw_f8{};
#pragma unroll
            for (int u2 = 0; u2 <= u; ++u2) d = gw_mma(gw_ld(st_ + gdn_ckc::off_tp(u, c, u2)), yb[u2], d);
            db[u] = gw_acc_to_b(d, ps);
            gw_f8 dd;
#pragma unroll
            for (int v = 0; v < 8; ++v) dd[v] = __expf(gl - sgp[16 * u + 2 * v]) * d[v];
            ddb[u] = gw_acc_to_b(dd, ps);
        }
        const float eg = __expf(gl);
#pragma unroll
        for (int j = 0; j < S / 16; ++j) {   // S = exp(gl) S + K^T (exp(gl - gamma) D)
            gw_f8 acc = s[j] * eg;
#pragma unroll
            for (int u = 0; u < GC_NT; ++u) acc = gw_mma(gw_ld(skt + gdn_ckc::off_kt(j, c, u)), ddb[u], acc);
            s[j] = acc;
        }
#pragma unroll
        for (int u = 0; u < GC_NT; ++u) {   // o = exp(gamma) Q S0 + P D
            gw_f8 acc;
#pragma unroll
            for (int v = 0; v < 8; ++v) acc[v] = __expf(sgp[16 * u + 2 * v]) * qs[u][v];
#pragma unroll
            for (int u2 = 0; u2 <= u; ++u2) acc = gw_mma(gw_ld(sp_ + gdn_ckc::off_tp(u, c, u2)), db[u2], acc);
            o[u] = acc;
#pragma unroll
            for (int v = 0; v < 8; ++v) {   // the wave's 16 columns of o^2 per token, then the 8 waves' through LDS
                o[u][v] *= out_scale;
                float p = o[u][v] * o[u][v];
#pragma unroll
                for (int m = 1; m < 16; m <<= 1) p += __shfl_xor(p, m);
                if (c == 0) sn[par][(16 * u + 2 * v + hi) * GW_NW + w] = p;
            }
        }
#pragma unroll
        for (int u = 0; u < GC_NT; ++u)   // this chunk's z slice, for its output rows after barrier (1) of the next chunk
#pragma unroll
            for (int i = 0; i < 2; ++i)
                zq[u][i] = t0 + st + 16 * u < T ? *reinterpret_cast<const gw_f4*>(z + (t0 + st + 16 * u) * (int64_t) (HV * S) + vh * S + sr + 4 * i) : gw_f4{};
        gw_lds_barrier();   // (2) the sums of o^2
#pragma unroll
        for (int u = 0; u < GC_NT; ++u)
#pragma unroll
            for (int v = 0; v < 8; ++v) sop[(16 * u + 2 * v) * S] = o[u][v];
    }
    __asm__ volatile("" ::"v"(touched0), "v"(touched1));
    gw_lds_barrier();
    store_rows((T - 1) / GC_CK * GC_CK, (int) (((T - 1) / GC_CK) & 1));
    __asm__ volatile("" : "+v"(sbase));   // the state's addresses recomputed here, not kept (spilled) since the start
#pragma unroll
    for (int j = 0; j < S / 16; ++j)
#pragma unroll
        for (int v = 0; v < 8; ++v) *(sbase + (size_t) (16 * j + 2 * v) * rs) = s[j][v];
}

// (g_gdn_ck_launches, the launch counter of the default-path test, is defined at the top of kernels.cu.)
// Launches the chunk-32 prep and scan for T >= 64 tokens (gfx11, checked by the caller).  y is the FP32 scratch of the other kernels
// (T x 6144 floats) and holds T and P: 3 KB a token.
static void gdn_rec_wmma_ck(float* state, const float* h, const float* gate, const float* beta, const float* z, const float* gamma, float eps,
                            float* y, uint16_t* y16, int64_t T, cudaStream_t s, int64_t ld16) {
    const int64_t nch = (T + GC_CK - 1) / GC_CK;
    _Float16* tp = reinterpret_cast<_Float16*>(y);
    ++g_gdn_ck_launches;
    gdn_wmma_ck_prep_kernel<<<dim3(HK, (unsigned) nch), 32 * GW_VPK, 0, s>>>(h, gate, beta, tp, T);
    gdn_wmma_ck_scan_kernel<<<HV, GW_NW * 32, 0, s>>>(state, h, gate, beta, z, gamma, reinterpret_cast<const unsigned*>(tp), eps, y16, T, ld16);
}
