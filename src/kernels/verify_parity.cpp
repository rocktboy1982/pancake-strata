// src/kernels/verify_parity.cpp - batched / split verify-window kernels against the kernels they replace, BITWISE.
//
// A verify window's token t must come out bit for bit as it would in a window of any other size (the drafts are
// accepted exactly when greedy decode would have produced them), so every kernel here is checked against the
// one it stands in for, on random inputs that include -0.0, denormals, NaN, -inf and large values.  The tests
// follow eddoursul's fork (MIT; verify_parity.cpp), ported to this tree's kernels.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/spec/window_rule.hpp"   // the host references of row_entropy / typical_accept_rows (LEADS.md section 3)

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
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
    check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
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

// argmax_rows against sample_tokens' greedy pick and row_top_prob_split against row_top_prob, bitwise: rows of both
// vocabularies with ties at the maximum (the lowest index wins), NaN, -inf, a row with no value above -inf, and
// launches repeated on one scratch (each must leave its counters at zero)
int test_argmax(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    int bad = 0;
    std::normal_distribution<float> nd(0.0f, 4.0f);
    uint8_t* sa = dev<uint8_t>(argmax_rows_scratch_bytes(8));   // one scratch for every row count, as in the engine
    uint8_t* st = dev<uint8_t>(row_top_prob_scratch_bytes(8));
    check(cudaMemset(sa, 0, argmax_rows_scratch_bytes(8)), "memset");
    check(cudaMemset(st, 0, row_top_prob_scratch_bytes(8)), "memset");
    for (const int n : {248320, 40525, 4097, 33}) {
        for (const int rows : {4, 8, 1, 3, 8, 2}) {
            std::vector<float> h((size_t) rows * n);
            for (float& v : h) v = nd(rng);
            for (int r = 0; r < rows; ++r) {
                float* row = h.data() + (size_t) r * n;
                const int a = (int) (rng() % n), b = (int) (rng() % n);
                row[a] = row[b] = 40.0f;                       // a tie at the maximum
                row[(int) (rng() % n)] = std::nanf("");        // never picked
                row[(int) (rng() % n)] = -INFINITY;
                if (r == 2)                                    // no value above -inf: 0
                    for (int i = 0; i < n; ++i) row[i] = i % 7 ? -INFINITY : std::nanf("");
            }
            float* d_l = dev<float>(h.size());
            up(d_l, h);
            int* d_ref = dev<int>(rows);
            int32_t* d_out = dev<int32_t>(rows);
            float* d_p1 = dev<float>(rows);
            float* d_p2 = dev<float>(rows);
            SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            sample_tokens(d_l, rows, n, nullptr, 0, sp, d_ref, s);
            row_top_prob(d_l, rows, n, d_ref, d_p1, s);
            for (int rep = 0; rep < 3; ++rep) {
                argmax_rows(d_l, rows, n, sa, d_out, s);
                row_top_prob_split(d_l, rows, n, d_out, d_p2, st, s);
                check(cudaStreamSynchronize(s), "argmax");
                const std::vector<int> ref = down(d_ref, rows);
                const std::vector<int32_t> out = down(d_out, rows);
                const std::vector<float> p1 = down(d_p1, rows), p2 = down(d_p2, rows);
                for (int r = 0; r < rows; ++r)
                    if (ref[r] != out[r] || std::memcmp(&p1[r], &p2[r], 4) != 0) {
                        if (bad < 5)
                            std::printf("  argmax n %d rows %d row %d: %d / %d, p %.9g / %.9g\n", n, rows, r, ref[r],
                                        out[r], p1[r], p2[r]);
                        ++bad;
                    }
            }
            (void) cudaFree(d_l); (void) cudaFree(d_ref); (void) cudaFree(d_out); (void) cudaFree(d_p1); (void) cudaFree(d_p2);
        }
    }
    (void) cudaFree(sa);
    (void) cudaFree(st);
    std::printf("argmax_rows, row_top_prob_split: %s\n",
                bad ? "MISMATCH" : "bitwise equal to the one-block kernels (ties, NaN, -inf, 1-8 rows on one scratch)");
    return bad;
}

int test_mapped_multi(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    int bad = 0;
    // 7-copy round layout + non-multiple-of-4 large copy (1025 words) + 4B-aligned non-16B-aligned large copy (1028 words)
    const int64_t words[7] = {5, 40, 1025, 1028, 51200, 4, 16};
    const int64_t off[7]   = {0,  0,    0,    1,     0, 0,  0};   // entry 3 is +1 word (+4 B) offset -> 4B-aligned, not 16B
    int32_t* h_src[7] = {};
    int32_t* d_src[7] = {};
    int32_t* d_ref[7] = {};
    int32_t* d_out[7] = {};
    MappedCopy mc[7] = {};
    for (int i = 0; i < 7; ++i) {
        const size_t alloc_w = (size_t) (words[i] + off[i]);
        check(cudaHostAlloc((void**) &h_src[i], alloc_w * 4, cudaHostAllocMapped), "cudaHostAlloc");
        check(cudaHostGetDevicePointer((void**) &d_src[i], h_src[i], 0), "cudaHostGetDevicePointer");
        for (size_t j = 0; j < alloc_w; ++j) h_src[i][j] = (int32_t) rng();
        d_ref[i] = dev<int32_t>(alloc_w);
        d_out[i] = dev<int32_t>(alloc_w);
        check(cudaMemset(d_ref[i], 0, alloc_w * 4), "memset");
        if (words[i] > 1024 && (words[i] & 3) == 0 && off[i] == 0)
            copy_from_mapped((float*) (d_ref[i] + off[i]), (const float*) (d_src[i] + off[i]), words[i], s);
        else
            copy_i32_from_mapped(d_ref[i] + off[i], d_src[i] + off[i], words[i], s);
        mc[i] = {d_out[i] + off[i], d_src[i] + off[i], words[i]};
    }
    const char* prev = std::getenv("STRATA_MTP_MULTI_COPY");
    const std::string saved = prev ? prev : "";
    for (const char* mode : {"1", "0"}) {
#if defined(_WIN32)
        _putenv_s("STRATA_MTP_MULTI_COPY", mode);
#else
        setenv("STRATA_MTP_MULTI_COPY", mode, 1);
#endif
        for (int i = 0; i < 7; ++i) check(cudaMemsetAsync(d_out[i], 0x5a, (size_t) (words[i] + off[i]) * 4, s), "poison");
        copy_from_mapped_multi(mc, 7, s);
        check(cudaStreamSynchronize(s), "mapped_multi");
        for (int i = 0; i < 7; ++i) {
            const std::vector<int32_t> r = down(d_ref[i] + off[i], (size_t) words[i]);
            const std::vector<int32_t> o = down(d_out[i] + off[i], (size_t) words[i]);
            if (r != o) ++bad;
        }
    }
#if defined(_WIN32)
    _putenv_s("STRATA_MTP_MULTI_COPY", prev ? saved.c_str() : "");
#else
    if (prev) setenv("STRATA_MTP_MULTI_COPY", saved.c_str(), 1);
    else unsetenv("STRATA_MTP_MULTI_COPY");
#endif
    for (int i = 0; i < 7; ++i) {
        (void) cudaFreeHost(h_src[i]);
        (void) cudaFree(d_ref[i]);
        (void) cudaFree(d_out[i]);
    }
    std::printf("copy_from_mapped_multi: %s\n",
                bad ? "MISMATCH" : "bitwise equal to individual mapped copies (multi=1 and fallback=0, aligned + 1025w + 4B-aligned)");
    return bad;
}

int test_quantize_q8_0_scaled(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    int bad = 0;
    std::normal_distribution<float> nd(0.0f, 3.0f);
    for (const int64_t n : {2560, 10240, 25600}) {
        const int64_t nb = n / 32;
        std::vector<float> hx((size_t) n);
        for (int64_t i = 0; i < n; ++i) hx[(size_t) i] = nd(rng);
        for (int64_t j = 0; j < 32 && j < n; ++j) hx[(size_t) j] = 0.0f;   // all-zero block
        if (nb > 2) {
            hx[64] = 127.0f;
            for (int j = 1; j < 32; ++j) hx[64 + j] = (float) j - 0.5f;    // exact .5 half-away ties
        }
        float* d_x = dev<float>((size_t) n);
        uint8_t* d_b = dev<uint8_t>((size_t) nb * 34);
        float* d_s = dev<float>((size_t) nb);
        up(d_x, hx);
        quantize_q8_0_scaled(d_x, d_b, d_s, n, s);
        check(cudaStreamSynchronize(s), "q8_0_scaled");
        const std::vector<uint8_t> gb = down(d_b, (size_t) nb * 34);
        const std::vector<float> gs = down(d_s, (size_t) nb);
        for (int64_t b = 0; b < nb; ++b) {
            const float* xb = hx.data() + b * 32;
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::fmax(amax, std::fabs(xb[i]));
            const float sc = amax > 0.f ? amax / 127.f : 0.f;
            const float inv = sc > 0.f ? 1.f / sc : 0.f;
            if (std::memcmp(&gs[(size_t) b], &sc, 4) != 0) ++bad;
            const uint16_t d16 = f16_from_f32(sc);
            const uint8_t* ob = gb.data() + b * 34;
            if (ob[0] != (uint8_t) (d16 & 0xFF) || ob[1] != (uint8_t) (d16 >> 8)) ++bad;
            for (int i = 0; i < 32; ++i) {
                const float t = xb[i] * inv;
                const float r = t + (t >= 0.f ? 0.5f : -0.5f);
                int v = (int) r;
                v = v < -127 ? -127 : (v > 127 ? 127 : v);
                if (ob[2 + i] != (uint8_t) (int8_t) v) ++bad;
            }
        }
        (void) cudaFree(d_x);
        (void) cudaFree(d_b);
        (void) cudaFree(d_s);
    }
    std::printf("quantize_q8_0_scaled: %s\n", bad ? "MISMATCH" : "bitwise equal to CPU act_quant_q8_1 reference (zero, .5 ties, normal)");
    return bad;
}

void bench_head_ops(cudaStream_t s) {
    using namespace strata::kernels;
    uint8_t* sa = dev<uint8_t>(argmax_rows_scratch_bytes(8));
    uint8_t* st = dev<uint8_t>(row_top_prob_scratch_bytes(8));
    check(cudaMemset(sa, 0, argmax_rows_scratch_bytes(8)), "memset");
    check(cudaMemset(st, 0, row_top_prob_scratch_bytes(8)), "memset");
    cudaEvent_t e0, e1;
    check(cudaEventCreate(&e0), "ev");
    check(cudaEventCreate(&e1), "ev");
    constexpr int ITERS = 200;
    auto time_fn = [&](auto&& fn) {
        for (int i = 0; i < 20; ++i) fn();
        check(cudaStreamSynchronize(s), "warm");
        check(cudaEventRecord(e0, s), "e0");
        for (int i = 0; i < ITERS; ++i) fn();
        check(cudaEventRecord(e1, s), "e1");
        check(cudaEventSynchronize(e1), "sync");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
        return ms * 1000.0 / ITERS;
    };
    for (const int n : {106299, 248320}) {
        for (const int rows : {1, 4, 5}) {
            if (n == 106299 && rows > 1) continue;
            std::vector<float> h((size_t) rows * n, 0.1f);
            for (int r = 0; r < rows; ++r) h[(size_t) r * n + (r * 1337 + 42) % n] = 10.0f;
            float* d_l = dev<float>(h.size());
            up(d_l, h);
            int32_t* d_out = dev<int32_t>(rows);
            float* d_p = dev<float>(rows);
            SamplerParams sp;
            sp.greedy = true;
            sp.temperature = 0.0f;
            const double us_g1 = time_fn([&] { sample_tokens(d_l, rows, n, nullptr, 0, sp, d_out, s); });
            const double us_gm = time_fn([&] { argmax_rows(d_l, rows, n, sa, d_out, s); });
            const double us_p1 = time_fn([&] { row_top_prob(d_l, rows, n, d_out, d_p, s); });
            const double us_pm = time_fn([&] { row_top_prob_split(d_l, rows, n, d_out, d_p, st, s); });
            std::printf("  V=%6d rows=%d: greedy 1-blk %6.2f us -> argmax_rows %6.2f us (x%.2f) | top_prob 1-blk %6.2f us -> split %6.2f us (x%.2f)\n",
                        n, rows, us_g1, us_gm, us_g1 / us_gm, us_p1, us_pm, us_p1 / us_pm);
            (void) cudaFree(d_l); (void) cudaFree(d_out); (void) cudaFree(d_p);
        }
    }
    (void) cudaEventDestroy(e0);
    (void) cudaEventDestroy(e1);
    (void) cudaFree(sa);
    (void) cudaFree(st);
}

// LEADS.md section 3: row_entropy (STRATA_DRAFT_ENTROPY: the draft stop rules, window-trace record 7) and typical_accept_rows
// (STRATA_TYPICAL_THINK, Medusa typical acceptance) against the double-precision host references in spec/window_rule.hpp.
// Not bitwise (the kernels sum FP32 in row_top_prob's order with __expf): entropy and probabilities within a relative 2e-4
// (+1e-6 absolute), NaN exactly where the reference is NaN, and the typical flags equal wherever p is not within 1e-3 of the
// threshold.  Rows: normal logits (sd 4) with a tie at the top, a flat row, a row of mostly -inf, a row holding a NaN; both
// vocabularies; candidates = the argmax, a random token, -1 and n (out of range); temperatures 1.0 and 0.6.
int test_entropy_typical(std::mt19937& rng, cudaStream_t s) {
    using namespace strata::kernels;
    int bad = 0;
    std::normal_distribution<float> nd(0.0f, 4.0f);
    auto close = [](double g, double r) {
        if (r != r) return g != g;
        return std::abs(g - r) <= 2e-4 * std::abs(r) + 1e-6;
    };
    for (const int n : {248320, 40525, 33}) {
        const int rows = 6;
        std::vector<float> h((size_t) rows * n);
        for (float& v : h) v = nd(rng);
        for (int r = 0; r < rows; ++r) {
            float* row = h.data() + (size_t) r * n;
            if (r == 1) for (int i = 0; i < n; ++i) row[i] = 0.5f;                                  // flat
            if (r == 2) for (int i = 0; i < n; ++i) row[i] = i % 5 ? -INFINITY : nd(rng);           // mostly -inf
            if (r == 3) row[(int) (rng() % n)] = std::nanf("");                                     // NaN: entropy NaN, nothing kept
            if (r == 4) { const int a = (int) (rng() % n), b = (int) (rng() % n); row[a] = row[b] = 30.0f; }   // a tie at the top
        }
        float* d_l = dev<float>(h.size());
        up(d_l, h);
        int* d_ids = dev<int>(rows);
        float* d_ent = dev<float>(rows);
        float* d_t2 = dev<float>(rows);
        SamplerParams sp;
        sp.greedy = true;
        sp.temperature = 0.0f;
        sample_tokens(d_l, rows, n, nullptr, 0, sp, d_ids, s);
        row_entropy(d_l, rows, n, (const int32_t*) d_ids, d_ent, d_t2, s);
        check(cudaStreamSynchronize(s), "row_entropy");
        const std::vector<int> ids = down(d_ids, rows);
        const std::vector<float> ent = down(d_ent, rows), t2 = down(d_t2, rows);
        for (int r = 0; r < rows; ++r) {
            double rt2 = 0.0;
            const double re = strata::spec::row_entropy_ref(h.data() + (size_t) r * n, n, ids[r], &rt2);
            if (!close(ent[r], re) || !close(t2[r], rt2)) {
                if (bad < 8) std::printf("  row_entropy n %d row %d: H %.9g / %.9g, top2 %.9g / %.9g\n", n, r, ent[r], re, t2[r], rt2);
                ++bad;
            }
        }
        for (const float temp : {1.0f, 0.6f}) {
            std::vector<int32_t> cand(rows);
            for (int r = 0; r < rows; ++r) cand[r] = r == 0 ? ids[r] : r == 1 ? -1 : r == 5 ? n : (int32_t) (rng() % n);
            cand[4] = ids[4];
            int32_t* d_c = dev<int32_t>(rows);
            int32_t* d_ok = dev<int32_t>(rows);
            float* d_pc = dev<float>(rows);
            float* d_hc = dev<float>(rows);
            up(d_c, cand);
            typical_accept_rows(d_l, rows, n, d_c, temp, 0.09f, 0.3f, d_ok, d_pc, d_hc, s);
            check(cudaStreamSynchronize(s), "typical_accept_rows");
            const std::vector<int32_t> ok = down(d_ok, rows);
            const std::vector<float> pc = down(d_pc, rows), hc = down(d_hc, rows);
            for (int r = 0; r < rows; ++r) {
                double rp = 0.0, rh = 0.0;
                const bool rok = strata::spec::typical_row_ref(h.data() + (size_t) r * n, n, cand[r], temp, 0.09f, 0.3f, &rp, &rh);
                const double thr = rh == rh ? (double) strata::spec::TypicalAccept::threshold(0.09f, 0.3f, (float) rh) : 0.0;
                const bool near = rp == rp && std::abs(rp - thr) <= 1e-3 * thr;
                if (!close(pc[r], rp) || !close(hc[r], rh) || (!near && (ok[r] != 0) != rok)) {
                    if (bad < 8)
                        std::printf("  typical n %d T %.1f row %d cand %d: ok %d / %d, p %.9g / %.9g, H %.9g / %.9g\n", n, (double) temp,
                                    r, cand[r], ok[r], (int) rok, pc[r], rp, hc[r], rh);
                    ++bad;
                }
            }
            (void) cudaFree(d_c); (void) cudaFree(d_ok); (void) cudaFree(d_pc); (void) cudaFree(d_hc);
        }
        (void) cudaFree(d_l); (void) cudaFree(d_ids); (void) cudaFree(d_ent); (void) cudaFree(d_t2);
    }
    std::printf("row_entropy, typical_accept_rows: %s\n",
                bad ? "MISMATCH" : "match the host references (flat, -inf, NaN, tie rows; both vocabularies; T 1.0 / 0.6)");
    return bad;
}

}  // namespace

int main(int argc, char** argv) {
    bool do_bench = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--bench") do_bench = true;
        else if (std::string(argv[i]) != "--selftest") {
            std::fprintf(stderr, "usage: verify_parity [--selftest] [--bench]\n");
            return 2;
        }
    }
    std::mt19937 rng(20260925);
    cudaStream_t s = nullptr;
    check(cudaStreamCreate(&s), "stream");
    int bad = 0;
    bad += test_argmax(rng, s);
    bad += test_mapped_multi(rng, s);
    bad += test_quantize_q8_0_scaled(rng, s);
    bad += test_entropy_typical(rng, s);
    if (do_bench && bad == 0) bench_head_ops(s);
    (void) cudaStreamDestroy(s);
    std::printf("verify_parity: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
