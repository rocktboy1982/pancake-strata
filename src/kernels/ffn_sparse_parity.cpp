// src/kernels/ffn_sparse_parity.cpp - the opt-in sparse-FFN kernels (include/strata/kernels/ffn_sparse.hpp) against host
// references: teal_sparsify (TEAL per row and BlockFFN union, bitwise, its ascending index list), down_t_gemv over the listed
// columns (bitwise: the same fmaf order), its degenerate all-columns case against the dense FP16 product, and
// lowrank_q81_apply (STRATA_LQER) against an FP64 reference.  GPU, synthetic, no model.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/ffn_sparse.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}
template <typename T> T* dev(size_t n) {
    T* p = nullptr;
    check(cudaMalloc(&p, (n ? n : 1) * sizeof(T)), "cudaMalloc");
    return p;
}
template <typename T> void up(T* d, const std::vector<T>& h) {
    check(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
}
template <typename T> std::vector<T> down(const T* d, size_t n) {
    std::vector<T> h(n);
    check(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return h;
}
bool same_bits(float a, float b) { return std::memcmp(&a, &b, 4) == 0; }

int test_teal(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    int bad = 0;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (const int n_ff : {640, 2048, 3001})
        for (const int n_tok : {1, 3, 8, 16})
            for (const int uni : {0, 1})
                for (const float tau : {0.0f, 0.5f, 1.2f}) {
                    std::vector<float> h((size_t) n_ff * n_tok);
                    for (float& v : h) v = nd(rng);
                    h[(size_t) (rng() % h.size())] = std::nanf("");
                    h[(size_t) (rng() % h.size())] = -0.0f;
                    std::vector<float> ref = h;
                    std::vector<int32_t> idx_ref;
                    teal_sparsify_host(ref.data(), n_ff, n_tok, tau, uni != 0, &idx_ref);
                    float* d_h = dev<float>(h.size());
                    int32_t* d_idx = dev<int32_t>((size_t) n_ff + 1);
                    up(d_h, h);
                    teal_sparsify(d_h, n_ff, n_tok, tau, uni != 0, d_idx, d_idx + n_ff, s);
                    check(cudaStreamSynchronize(s), "teal sync");
                    const std::vector<float> got = down(d_h, h.size());
                    const std::vector<int32_t> gi = down(d_idx, (size_t) n_ff + 1);
                    int diff = 0;
                    for (size_t i = 0; i < h.size(); ++i) diff += same_bits(got[i], ref[i]) ? 0 : 1;
                    const int n = gi[(size_t) n_ff];
                    bool idx_ok = n == (int) idx_ref.size();
                    for (int k = 0; idx_ok && k < n; ++k) idx_ok = gi[(size_t) k] == idx_ref[(size_t) k];
                    if (diff || !idx_ok) {
                        std::fprintf(stderr, "teal_sparsify n_ff %d n_tok %d union %d tau %g: %d values differ, index list %s\n",
                                     n_ff, n_tok, uni, tau, diff, idx_ok ? "ok" : "differs");
                        ++bad;
                    }
                    cudaFree(d_h);
                    cudaFree(d_idx);
                }
    return bad;
}

int test_down_t(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    int bad = 0;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (const int n_ff : {640, 1536})
        for (const int n_out : {1000, 2048})
            for (const int n_tok : {1, 2, 8, 16})
                for (const float tau : {0.0f, 0.8f}) {
                    std::vector<uint16_t> wt((size_t) n_ff * n_out);
                    for (auto& w : wt) w = f16_from_f32(nd(rng) * 0.05f);
                    std::vector<float> h((size_t) n_ff * n_tok);
                    for (float& v : h) v = nd(rng);
                    std::vector<int32_t> idx;
                    teal_sparsify_host(h.data(), n_ff, n_tok, tau, true, &idx);
                    std::vector<float> ref((size_t) n_tok * n_out), dense((size_t) n_tok * n_out);
                    for (int t = 0; t < n_tok; ++t)
                        for (int o = 0; o < n_out; ++o) {
                            float acc = 0.0f, accd = 0.0f;
                            for (const int i : idx)
                                acc = std::fma(f32_from_f16(wt[(size_t) i * n_out + o]), h[(size_t) t * n_ff + i], acc);
                            for (int i = 0; i < n_ff; ++i)
                                accd = std::fma(f32_from_f16(wt[(size_t) i * n_out + o]), h[(size_t) t * n_ff + i], accd);
                            ref[(size_t) t * n_out + o] = acc;
                            dense[(size_t) t * n_out + o] = accd;
                        }
                    std::vector<int32_t> idx_n = idx;
                    idx_n.resize((size_t) n_ff + 1, 0);
                    idx_n[(size_t) n_ff] = (int32_t) idx.size();
                    uint16_t* d_w = dev<uint16_t>(wt.size());
                    float* d_h = dev<float>(h.size());
                    int32_t* d_i = dev<int32_t>(idx_n.size());
                    float* d_y = dev<float>(ref.size());
                    up(d_w, wt);
                    up(d_h, h);
                    up(d_i, idx_n);
                    down_t_gemv(d_w, d_h, d_i, d_i + n_ff, d_y, n_ff, n_out, n_tok, s);
                    check(cudaStreamSynchronize(s), "down_t sync");
                    const std::vector<float> got = down(d_y, ref.size());
                    int diff = 0;
                    for (size_t i = 0; i < ref.size(); ++i) diff += same_bits(got[i], ref[i]) ? 0 : 1;
                    // tau 0 lists every column: the masked read is the dense product, bit for bit
                    int ddiff = 0;
                    if (tau == 0.0f)
                        for (size_t i = 0; i < ref.size(); ++i) ddiff += same_bits(got[i], dense[i]) ? 0 : 1;
                    if (diff || ddiff) {
                        std::fprintf(stderr, "down_t_gemv n_ff %d n_out %d n_tok %d tau %g: %d differ from the reference, %d from dense\n",
                                     n_ff, n_out, n_tok, tau, diff, ddiff);
                        ++bad;
                    }
                    cudaFree(d_w);
                    cudaFree(d_h);
                    cudaFree(d_i);
                    cudaFree(d_y);
                }
    return bad;
}

int test_lowrank(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    int bad = 0;
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (const int n_in : {256, 2048})
        for (const int n_out : {300, 1024})
            for (const int r : {1, 8, 32})
                for (const int ncols : {1, 5, 16}) {
                    std::vector<float> A((size_t) n_out * r), B((size_t) r * n_in), y0((size_t) ncols * n_out);
                    for (float& v : A) v = nd(rng) * 0.1f;
                    for (float& v : B) v = nd(rng) * 0.1f;
                    for (float& v : y0) v = nd(rng);
                    const int nb = n_in / 32;
                    std::vector<uint8_t> xq((size_t) ncols * nb * 36);
                    std::vector<double> x((size_t) ncols * n_in);
                    for (int c = 0; c < ncols; ++c)
                        for (int b = 0; b < nb; ++b) {
                            uint8_t* blk = xq.data() + ((size_t) c * nb + b) * 36;
                            const uint16_t dh = f16_from_f32(0.01f + 0.02f * (float) (rng() % 100) / 100.0f);
                            int sum = 0;
                            for (int j = 0; j < 32; ++j) {
                                const int8_t q = (int8_t) ((int) (rng() % 255) - 127);
                                blk[4 + j] = (uint8_t) q;
                                sum += q;
                                x[(size_t) c * n_in + b * 32 + j] = (double) f32_from_f16(dh) * q;
                            }
                            const uint16_t sh = f16_from_f32(f32_from_f16(dh) * (float) sum);
                            std::memcpy(blk, &dh, 2);
                            std::memcpy(blk + 2, &sh, 2);
                        }
                    std::vector<double> ref((size_t) ncols * n_out);
                    for (int c = 0; c < ncols; ++c) {
                        std::vector<double> tmp((size_t) r, 0.0);
                        for (int j = 0; j < r; ++j)
                            for (int i = 0; i < n_in; ++i) tmp[(size_t) j] += (double) B[(size_t) j * n_in + i] * x[(size_t) c * n_in + i];
                        for (int o = 0; o < n_out; ++o) {
                            double acc = y0[(size_t) c * n_out + o];
                            for (int j = 0; j < r; ++j) acc += (double) A[(size_t) o * r + j] * tmp[(size_t) j];
                            ref[(size_t) c * n_out + o] = acc;
                        }
                    }
                    float* d_A = dev<float>(A.size());
                    float* d_B = dev<float>(B.size());
                    float* d_y = dev<float>(y0.size());
                    float* d_t = dev<float>((size_t) ncols * r);
                    uint8_t* d_x = dev<uint8_t>(xq.size());
                    up(d_A, A);
                    up(d_B, B);
                    up(d_y, y0);
                    up(d_x, xq);
                    lowrank_q81_apply(d_A, d_B, r, d_x, d_t, d_y, n_in, n_out, ncols, s);
                    check(cudaStreamSynchronize(s), "lowrank sync");
                    const std::vector<float> got = down(d_y, y0.size());
                    double worst = 0.0;
                    for (size_t i = 0; i < ref.size(); ++i)
                        worst = std::max(worst, std::fabs((double) got[i] - ref[i]) / (1.0 + std::fabs(ref[i])));
                    if (!(worst < 1e-4)) {
                        std::fprintf(stderr, "lowrank_q81_apply n_in %d n_out %d r %d ncols %d: worst rel error %.3g\n", n_in, n_out,
                                     r, ncols, worst);
                        ++bad;
                    }
                    cudaFree(d_A);
                    cudaFree(d_B);
                    cudaFree(d_y);
                    cudaFree(d_t);
                    cudaFree(d_x);
                }
    return bad;
}

}  // namespace

int main() {
    std::mt19937 rng(20261116);
    cudaStream_t s = nullptr;
    check(cudaStreamCreate(&s), "stream");
    int bad = 0;
    bad += test_teal(rng, s);
    bad += test_down_t(rng, s);
    bad += test_lowrank(rng, s);
    (void) cudaStreamDestroy(s);
    std::printf("ffn_sparse_parity: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
