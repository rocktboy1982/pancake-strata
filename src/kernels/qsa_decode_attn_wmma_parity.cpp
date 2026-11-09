// src/kernels/qsa_decode_attn_wmma_parity.cpp - pancake: the matrix-core decode attention chunk pass
// (qsa_decode_attn_wmma.hpp, STRATA_ATTN_WMMA) against the default FP32 kernel and an FP64 host reference over the same
// dequantized pools (GPU, synthetic, no model).
//
// Covers: int8 (mode 1), q4_0 (mode 2) and K8V4 (mode 3) pools; a permuted page table with unresident (-1) pages, so some
// cells and some whole chunks are masked; widths 1, 37, 64, 65, 1000 and 2051 (the real selection at cap 2051), alone and
// mixed in one batch, n_q 1 and 3; queries with an outlier channel. Pass: the WMMA output's error against FP64 is at most
// max(2e-5, 4 x the FP32 kernel's) in relative L2 per query head and 1e-4 in max abs relative to the head's max |ref|,
// and two runs are bit-identical (determinism). Exit 0 pass, 1 fail, 77 no gfx11 device.
//
//   hip_decode_attn_wmma_parity           the checks
//   hip_decode_attn_wmma_parity --bench   plus time per call (chunk pass + merge), FP32 kernel vs WMMA, cap 2051
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_decode_attn_wmma.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "hip_decode_attn_wmma_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
template <typename T> T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    ck(cudaMalloc((void**) &d, h.size() * sizeof(T) + 256), "cudaMalloc");
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

constexpr int HD = 256, NKV = 2, G = 12, NH = 24;
constexpr int64_t CAP = 2051;

struct Pools {
    int mode = 1;
    int64_t rows = 0;
    std::vector<int8_t> kq, vq;            // int8 codes [row][256]
    std::vector<uint16_t> ks, vs;          // fp16 scales [row][4]
    std::vector<uint8_t> k4, v4;           // q4_0 [row][144]
    // dequantized host copies [row][256]
    std::vector<float> kf, vf;
};

void fill_q8(std::mt19937& rng, int64_t rows, std::vector<int8_t>& c, std::vector<uint16_t>& sc, std::vector<float>& f) {
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> scale(0.002f, 0.06f);
    c.resize(rows * HD); sc.resize(rows * (HD / 64)); f.resize(rows * HD);
    for (int64_t r = 0; r < rows; ++r) {
        for (int g = 0; g < HD / 64; ++g) sc[r * 4 + g] = k::f16_from_f32(scale(rng));
        for (int d = 0; d < HD; ++d) {
            c[r * HD + d] = (int8_t) code(rng);
            f[r * HD + d] = (float) c[r * HD + d] * k::f32_from_f16(sc[r * 4 + d / 64]);
        }
    }
}
void fill_q4(std::mt19937& rng, int64_t rows, std::vector<uint8_t>& b, std::vector<float>& f) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.01f, 0.3f);
    const int row_bytes = (HD / 32) * 18;
    b.resize(rows * row_bytes); f.resize(rows * HD);
    for (int64_t r = 0; r < rows; ++r)
        for (int blk = 0; blk < HD / 32; ++blk) {
            uint8_t* p = b.data() + r * row_bytes + blk * 18;
            const uint16_t d = k::f16_from_f32(scale(rng));
            std::memcpy(p, &d, 2);
            for (int j = 0; j < 16; ++j) p[2 + j] = (uint8_t) byte(rng);
            const float df = k::f32_from_f16(d);
            for (int j = 0; j < 32; ++j) {
                const int nib = j < 16 ? (p[2 + j] & 0xF) : (p[2 + j - 16] >> 4);
                f[r * HD + blk * 32 + j] = (float) (nib - 8) * df;
            }
        }
}

struct Case {
    std::vector<int> widths;   // one per query
};

double rel_l2(const float* a, const double* ref, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; ++i) { num += (a[i] - ref[i]) * (a[i] - ref[i]); den += ref[i] * ref[i]; }
    return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}
}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    int n_dev = 0;
    if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev == 0) { std::printf("no GPU: skipped\n"); return 77; }
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "props");
    if (std::strncmp(prop.gcnArchName, "gfx11", 5) != 0) {
        std::printf("%s is not a gfx11 part: skipped\n", prop.gcnArchName);
        return 77;
    }
    const k::QsaShapes s = k::qsa_real_shapes();
    const int ps = (int) s.page_size;
    const int64_t n_cells = 8192, pages = n_cells / ps, rows = pages * NKV * ps;
    std::mt19937 rng(20261109);

    // page table: a permutation of physical pages, ~4% unresident, plus one run of 32 pages (128 cells: two whole chunks
    // of a dense selection)
    std::vector<int32_t> table(pages);
    for (int64_t i = 0; i < pages; ++i) table[i] = (int32_t) i;
    std::shuffle(table.begin(), table.end(), rng);
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    for (int64_t i = 0; i < pages; ++i) if (u01(rng) < 0.04f) table[i] = -1;
    for (int64_t i = 256; i < 256 + 32; ++i) table[i] = -1;
    int32_t* d_table = upload(table);

    const std::vector<Case> cases = {{{1}}, {{37}}, {{64}}, {{65}}, {{1000}}, {{2051}}, {{2051, 65, 1}}, {{2051, 2051, 2051}}};
    int fails = 0;
    double worst_w = 0, worst_f = 0;
    for (int mode = 1; mode <= 3; ++mode) {
        Pools P; P.mode = mode; P.rows = rows;
        if (mode == 2) { fill_q4(rng, rows, P.k4, P.kf); fill_q4(rng, rows, P.v4, P.vf); }
        else {
            fill_q8(rng, rows, P.kq, P.ks, P.kf);
            if (mode == 1) fill_q8(rng, rows, P.vq, P.vs, P.vf); else fill_q4(rng, rows, P.v4, P.vf);
        }
        k::QsaAttnPools pools{};
        pools.page_table = d_table;
        if (mode == 2) { pools.k_q4 = upload(P.k4); pools.v_q4 = upload(P.v4); }
        else {
            pools.k_q = upload(P.kq); pools.k_scale = upload(P.ks);
            if (mode == 1) { pools.v_q = upload(P.vq); pools.v_scale = upload(P.vs); }
            else pools.v_q4 = upload(P.v4);
        }
        for (const Case& cs : cases) {
            const int nq = (int) cs.widths.size();
            std::vector<float> q((size_t) nq * NH * HD);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (auto& x : q) x = nd(rng);
            for (int z = 0; z < nq; ++z) q[(size_t) z * NH * HD + 5 * HD + 17] = 40.0f;   // an outlier channel
            std::vector<int32_t> ids((size_t) nq * CAP, 0), steps((size_t) nq * k::kStepCount, 0);
            for (int z = 0; z < nq; ++z) {
                const int w = cs.widths[z];
                std::vector<int32_t> all(n_cells);
                for (int i = 0; i < n_cells; ++i) all[i] = i;
                std::shuffle(all.begin(), all.end(), rng);
                std::vector<int32_t> sel(all.begin(), all.begin() + w);
                if (w == 2051 && z == 1) for (int i = 0; i < w; ++i) sel[i] = 1000 + i;   // dense: hits the 128-cell hole
                std::sort(sel.begin(), sel.end());
                std::copy(sel.begin(), sel.end(), ids.begin() + (size_t) z * CAP);
                steps[(size_t) z * k::kStepCount + k::kStepPos] = (int32_t) n_cells - 1;
                steps[(size_t) z * k::kStepCount + k::kStepNKv] = (int32_t) n_cells;
                steps[(size_t) z * k::kStepCount + k::kStepNBid] = (int32_t) n_cells / 4;
                steps[(size_t) z * k::kStepCount + k::kStepWidth] = w;
            }
            float* d_q = upload(q);
            int32_t* d_ids = upload(ids);
            int32_t* d_steps = upload(steps);
            const size_t sf = k::qsa_decode_attn_scratch_floats(CAP, s);
            float *d_scr = nullptr, *d_o32 = nullptr, *d_ow = nullptr, *d_ow2 = nullptr;
            ck(cudaMalloc((void**) &d_scr, sf * nq * sizeof(float)), "scratch");
            ck(cudaMalloc((void**) &d_o32, (size_t) nq * NH * HD * 4), "o");
            ck(cudaMalloc((void**) &d_ow, (size_t) nq * NH * HD * 4), "o");
            ck(cudaMalloc((void**) &d_ow2, (size_t) nq * NH * HD * 4), "o");
            k::qsa_decode_attn_batch(d_q, pools, d_ids, d_steps, CAP, s, d_scr, d_o32, nq, nullptr);
            for (float* out : {d_ow, d_ow2}) {
                ck(cudaMemset(d_scr, 0xFF, sf * nq * sizeof(float)), "nan fill");   // NaN: an unwritten partial shows
                if (!k::qsa_decode_attn_wmma_chunks(d_q, pools, d_ids, d_steps, CAP, s, d_scr, nq, nullptr)) {
                    std::printf("FAIL mode %d: the WMMA chunk pass did not launch\n", mode);
                    return 1;
                }
                k::qsa_decode_attn_merge_only(d_steps, CAP, s, d_scr, out, nq, 0, nullptr);
            }
            ck(cudaDeviceSynchronize(), "sync");
            std::vector<float> o32((size_t) nq * NH * HD), ow(o32.size()), ow2(o32.size());
            ck(cudaMemcpy(o32.data(), d_o32, o32.size() * 4, cudaMemcpyDeviceToHost), "dl");
            ck(cudaMemcpy(ow.data(), d_ow, ow.size() * 4, cudaMemcpyDeviceToHost), "dl");
            ck(cudaMemcpy(ow2.data(), d_ow2, ow2.size() * 4, cudaMemcpyDeviceToHost), "dl");
            const bool det = std::memcmp(ow.data(), ow2.data(), ow.size() * 4) == 0;
            // FP64 reference
            double e_w = 0, e_f = 0, e_wmax = 0;
            bool bad = !det;
            for (int z = 0; z < nq; ++z) {
                const int w = cs.widths[z];
                for (int h = 0; h < NH; ++h) {
                    const int kvh = h / G;
                    std::vector<double> sc(w), ref(HD, 0.0);
                    std::vector<int64_t> rr(w, -1);
                    double m = -1e300;
                    for (int i = 0; i < w; ++i) {
                        const int cell = ids[(size_t) z * CAP + i];
                        const int32_t page = table[cell / ps];
                        if (page < 0) continue;
                        rr[i] = ((int64_t) page * NKV + kvh) * ps + cell % ps;
                        double dot = 0;
                        for (int d = 0; d < HD; ++d)
                            dot += (double) q[((size_t) z * NH + h) * HD + d] * P.kf[rr[i] * HD + d];
                        sc[i] = dot / 16.0;
                        m = std::max(m, sc[i]);
                    }
                    double l = 0;
                    for (int i = 0; i < w; ++i) {
                        if (rr[i] < 0) continue;
                        const double e = std::exp(sc[i] - m);
                        l += e;
                        for (int d = 0; d < HD; ++d) ref[d] += e * P.vf[rr[i] * HD + d];
                    }
                    double rmax = 0;
                    for (int d = 0; d < HD; ++d) { ref[d] = l > 0 ? ref[d] / l : 0.0; rmax = std::max(rmax, std::fabs(ref[d])); }
                    const size_t o = ((size_t) z * NH + h) * HD;
                    const double ew = rel_l2(&ow[o], ref.data(), HD), ef = rel_l2(&o32[o], ref.data(), HD);
                    double emax = 0;
                    for (int d = 0; d < HD; ++d) {
                        if (!std::isfinite(ow[o + d])) bad = true;
                        emax = std::max(emax, std::fabs(ow[o + d] - ref[d]));
                    }
                    emax = rmax > 0 ? emax / rmax : emax;
                    e_w = std::max(e_w, ew); e_f = std::max(e_f, ef); e_wmax = std::max(e_wmax, emax);
                    if (ew > std::max(2e-5, 4 * ef) || emax > 1e-4) bad = true;
                }
            }
            worst_w = std::max(worst_w, e_w); worst_f = std::max(worst_f, e_f);
            std::printf("%s mode %d widths", bad ? "FAIL" : "ok  ", mode);
            for (int w : cs.widths) std::printf(" %d", w);
            std::printf(": rel L2 vs FP64 wmma %.2e fp32 %.2e, max abs/max %.2e, deterministic %s\n", e_w, e_f, e_wmax,
                        det ? "yes" : "NO");
            fails += bad;
            if (bench && cs.widths.size() == 3 && cs.widths[1] == 2051) {
                cudaEvent_t a, b;
                cudaEventCreate(&a); cudaEventCreate(&b);
                for (int nqb : {1, 3}) {
                    const int iters = 500;
                    float t32 = 0, tw = 0;
                    for (int rep = 0; rep < 2; ++rep) {
                        cudaEventRecord(a);
                        for (int i = 0; i < iters; ++i)
                            k::qsa_decode_attn_batch(d_q, pools, d_ids, d_steps, CAP, s, d_scr, d_o32, nqb, nullptr);
                        cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&t32, a, b);
                        cudaEventRecord(a);
                        for (int i = 0; i < iters; ++i) {
                            k::qsa_decode_attn_wmma_chunks(d_q, pools, d_ids, d_steps, CAP, s, d_scr, nqb, nullptr);
                            k::qsa_decode_attn_merge_only(d_steps, CAP, s, d_scr, d_ow, nqb, 0, nullptr);
                        }
                        cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&tw, a, b);
                    }
                    std::printf("bench mode %d n_q %d width 2051: fp32 %.1f us/call, wmma %.1f us/call (x%.2f)\n", mode,
                                nqb, 1000.f * t32 / iters, 1000.f * tw / iters, t32 / tw);
                }
                cudaEventDestroy(a); cudaEventDestroy(b);
            }
            cudaFree(d_q); cudaFree(d_ids); cudaFree(d_steps); cudaFree(d_scr); cudaFree(d_o32); cudaFree(d_ow); cudaFree(d_ow2);
        }
    }
    std::printf("%s: %d failing case(s); worst rel L2 vs FP64: wmma %.2e, fp32 kernel %.2e\n", fails ? "FAIL" : "PASS",
                fails, worst_w, worst_f);
    return fails ? 1 : 0;
}
