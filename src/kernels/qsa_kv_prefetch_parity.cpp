// src/kernels/qsa_kv_prefetch_parity.cpp - LEADS.md Part U U8: the register-prefetch decode attention chunk kernel
// (attn_chunk_pf_kernel, a build made with -DSTRATA_ATTN_KV_PREFETCH, switched by qsa_attn_kv_prefetch_override / the env
// STRATA_ATTN_KV_PREFETCH=1) against the default chunk kernel, BITWISE, through the production entries qsa_decode_attn_step
// (n_q 1) and qsa_decode_attn_batch (n_q 2..4).  Prefetch only moves where loads are issued: every output bit must match.
// Four KV formats (fp16, int8, q4_0, k8v4), page tables with unresident (-1) pages (scattered and one 64-page run, so
// whole chunks and single cells are masked), widths {0, 1, 63, 64, 65, 1023, 2051, cap, INT_MAX-like}, selections shaped
// like the drafter's window, the indexer's and unordered with repeats.  Scratch and outputs NaN-filled before each run.
// Exit 77: no GPU, or the build has no prefetch kernel.  GPU, synthetic, no model.
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_kv_prefetch_parity: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
template <typename T> T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc((void**) &p, n * sizeof(T) + 256), "cudaMalloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "cudaMemset");
    return p;
}
template <typename T> T* upload(const std::vector<T>& h) {
    T* d = dalloc<T>(h.size());
    ck(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return d;
}

// fp16 bits drawn directly: a random sign, exponent 11..15 (|x| in [2^-4, 2)), a random mantissa; finite by
// construction. Scales are positive: INT8 [1/64, 1/32) (codes up to 127 give |x| < 4), Q4_0 [1/4, 1/2).
uint16_t f16_value(uint32_t r) { return (uint16_t) ((r & 0x8000u) | ((11u + (r >> 16) % 5u) << 10) | (r & 0x3FFu)); }
uint16_t f16_scale_q8(uint32_t r) { return (uint16_t) ((9u << 10) | (r & 0x3FFu)); }
uint16_t f16_scale_q4(uint32_t r) { return (uint16_t) ((13u << 10) | (r & 0x3FFu)); }

constexpr int kFmtF16 = 0, kFmtInt8 = 1, kFmtQ4 = 2, kFmtK8V4 = 3;
const char* fmt_name(int f) { return f == kFmtF16 ? "fp16" : f == kFmtInt8 ? "int8" : f == kFmtQ4 ? "q4_0" : "k8v4"; }

struct DevicePools {   // every format's pools over the same physical rows; `view` picks what a KV mode reads
    uint16_t *k16 = nullptr, *v16 = nullptr, *k8s = nullptr, *v8s = nullptr;
    int8_t *k8 = nullptr, *v8 = nullptr;
    uint8_t *k4 = nullptr, *v4 = nullptr;
    k::QsaAttnPools view(int fmt, const int32_t* table) const {
        k::QsaAttnPools p;
        p.page_table = table;
        if (fmt == kFmtF16) { p.k_pool = k16; p.v_pool = v16; }
        else if (fmt == kFmtInt8) { p.k_q = k8; p.v_q = v8; p.k_scale = k8s; p.v_scale = v8s; }
        else if (fmt == kFmtQ4) { p.k_q4 = k4; p.v_q4 = v4; }
        else { p.k_q = k8; p.k_scale = k8s; p.v_q4 = v4; }   // K8V4: INT8 K, Q4_0 V (qsa_decode_attn.cu kv_mode 3)
        return p;
    }
};

DevicePools make_pools(int64_t rows, int64_t D, std::mt19937& rng) {
    DevicePools p;
    {
        std::vector<uint16_t> h((size_t) (rows * D));
        for (auto& x : h) x = f16_value(rng());
        p.k16 = upload(h);
        for (auto& x : h) x = f16_value(rng());
        p.v16 = upload(h);
    }
    {
        std::vector<int8_t> c((size_t) (rows * D));
        for (auto& x : c) x = (int8_t) ((int) (rng() % 255u) - 127);
        p.k8 = upload(c);
        for (auto& x : c) x = (int8_t) ((int) (rng() % 255u) - 127);
        p.v8 = upload(c);
        std::vector<uint16_t> sc((size_t) (rows * (D / k::KV_Q8_GROUP)));
        for (auto& x : sc) x = f16_scale_q8(rng());
        p.k8s = upload(sc);
        for (auto& x : sc) x = f16_scale_q8(rng());
        p.v8s = upload(sc);
    }
    {
        const size_t blocks = (size_t) (rows * (D / k::QK4_0));
        std::vector<uint8_t> b(blocks * sizeof(k::block_q4_0));
        for (int side = 0; side < 2; ++side) {
            for (size_t i = 0; i < blocks; ++i) {
                k::block_q4_0 blk;
                blk.d = f16_scale_q4(rng());
                for (auto& q : blk.qs) q = (uint8_t) (rng() & 0xFFu);
                std::memcpy(b.data() + i * sizeof(k::block_q4_0), &blk, sizeof(blk));
            }
            (side == 0 ? p.k4 : p.v4) = upload(b);
        }
    }
    return p;
}

// the cells this harness can address, and where the 64 unresident pages in a row sit (cells [kHole, kHole + 256))
constexpr int64_t kCells = 40960;
constexpr int64_t kHole = 8192;

// query `i`'s `cap` selection ids. Every entry is a valid cell: a width above the cap makes the chunk kernel read
// all `cap` of them.
void selection(int32_t* ids, int64_t cap, int width, int shape, std::mt19937& rng) {
    const int64_t w = std::max<int64_t>(0, std::min<int64_t>(width, cap));
    if (shape == 0) {   // the drafter's window: consecutive cells (window_ids), often across the unresident run
        int64_t start;
        if (rng() % 2 == 0) start = std::max<int64_t>(0, kHole - 64 * (int64_t) (rng() % 8));   // chunk-aligned on it
        else start = (int64_t) (rng() % (uint32_t) (kCells - std::max<int64_t>(w, 1)));
        if (start + w > kCells) start = kCells - w;
        for (int64_t j = 0; j < cap; ++j) ids[j] = (int32_t) ((start + j) % kCells);
    } else if (shape == 1) {   // the indexer's: ascending with gaps
        std::vector<int32_t> v((size_t) cap);
        for (auto& x : v) x = (int32_t) (rng() % (uint32_t) kCells);
        std::sort(v.begin(), v.begin() + w);
        std::copy(v.begin(), v.end(), ids);
    } else {   // unordered, with repeats
        for (int64_t j = 0; j < cap; ++j) ids[j] = (int32_t) (rng() % (uint32_t) kCells);
    }
}


}  // namespace

namespace {
struct Run {
    k::QsaShapes s = k::qsa_real_shapes();
    DevicePools pools;
    int32_t* table = nullptr;
    float *q = nullptr, *scratch = nullptr, *out[2] = {nullptr, nullptr};
    int32_t *ids = nullptr, *steps = nullptr;
    static constexpr int64_t kMaxQ = 4, kMaxCap = 4096;
    void init(std::mt19937& rng) {
        const int64_t PS = s.page_size, pages = kCells / PS, H = s.n_head_kv, D = s.head_dim;
        std::vector<int32_t> tab((size_t) pages);
        for (int64_t i = 0; i < pages; ++i) tab[(size_t) i] = (int32_t) i;
        std::shuffle(tab.begin(), tab.end(), rng);
        for (int64_t i = 0; i < pages; ++i)
            if (rng() % 32u == 0) tab[(size_t) i] = -1;
        for (int64_t i = kHole / PS; i < (kHole + 256) / PS; ++i) tab[(size_t) i] = -1;
        table = upload(tab);
        pools = make_pools(pages * H * PS, D, rng);
        q = dalloc<float>((size_t) (kMaxQ * s.n_head * D));
        ids = dalloc<int32_t>((size_t) (kMaxQ * kMaxCap));
        steps = dalloc<int32_t>((size_t) (kMaxQ * k::kStepCount));
        scratch = dalloc<float>((size_t) (k::qsa_decode_attn_scratch_floats(kMaxCap, s) * (uint64_t) kMaxQ));
        for (auto& o : out) o = dalloc<float>((size_t) (kMaxQ * s.n_head * D));
    }
    // one case: both runs, true when bitwise equal
    bool one(int fmt, int64_t cap, int n_q, const std::vector<int32_t>& widths, std::mt19937& rng) {
        const int64_t D = s.head_dim, NH = s.n_head;
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> hq((size_t) (n_q * NH * D));
        for (auto& x : hq) x = nd(rng);
        ck(cudaMemcpy(q, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "q");
        std::vector<int32_t> hid((size_t) (n_q * cap)), hst((size_t) (n_q * k::kStepCount), 0);
        for (int i = 0; i < n_q; ++i) {
            selection(hid.data() + (size_t) i * cap, cap, widths[(size_t) i], (int) (rng() % 3u), rng);
            hst[(size_t) i * k::kStepCount + k::kStepWidth] = widths[(size_t) i];
        }
        ck(cudaMemcpy(ids, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice), "ids");
        ck(cudaMemcpy(steps, hst.data(), hst.size() * 4, cudaMemcpyHostToDevice), "steps");
        const size_t n = (size_t) (n_q * NH * D);
        std::vector<float> r[2];
        for (int on = 0; on < 2; ++on) {
            if (!k::qsa_attn_kv_prefetch_override(on)) { std::puts("SKIP: build without -DSTRATA_ATTN_KV_PREFETCH"); std::exit(77); }
            ck(cudaMemset(scratch, 0xFF, (size_t) k::qsa_decode_attn_scratch_floats(cap, s) * (size_t) n_q * 4), "scratch NaN");
            ck(cudaMemset(out[on], 0xFF, n * 4), "out NaN");
            if (n_q == 1) k::qsa_decode_attn_step(q, pools.view(fmt, table), ids, steps, cap, s, scratch, out[on], nullptr);
            else k::qsa_decode_attn_batch(q, pools.view(fmt, table), ids, steps, cap, s, scratch, out[on], n_q, nullptr);
            ck(cudaDeviceSynchronize(), "attention");
            r[on].resize(n);
            ck(cudaMemcpy(r[on].data(), out[on], n * 4, cudaMemcpyDeviceToHost), "d2h");
        }
        k::qsa_attn_kv_prefetch_override(0);
        const bool same = std::memcmp(r[0].data(), r[1].data(), n * 4) == 0;
        if (!same) {
            std::printf("FAIL %s cap %lld n_q %d widths", fmt_name(fmt), (long long) cap, n_q);
            for (int32_t w : widths) std::printf(" %d", w);
            std::printf("\n");
        }
        return same;
    }
};
}  // namespace

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) { std::puts("SKIP: no GPU"); return 77; }
    if (!k::qsa_attn_kv_prefetch_override(0)) { std::puts("SKIP: build without -DSTRATA_ATTN_KV_PREFETCH"); return 77; }
    std::mt19937 rng(88);
    Run run;
    run.init(rng);
    const int32_t wl[] = {0, 1, 63, 64, 65, 1023, 2051, 4096, 1 << 30};
    int cases = 0, bad = 0;
    for (int fmt = 0; fmt < 4; ++fmt)
        for (int64_t cap : {1000LL, 2051LL, 2112LL, 4096LL})
            for (int n_q : {1, 2, 4})
                for (int rep = 0; rep < 6; ++rep) {
                    std::vector<int32_t> widths((size_t) n_q);
                    for (auto& w : widths) w = wl[rng() % (sizeof(wl) / sizeof(wl[0]))];
                    ++cases;
                    if (!run.one(fmt, cap, n_q, widths, rng)) ++bad;
                }
    std::printf("%s: %d of %d prefetch attention cases bitwise equal to the default\n", bad ? "FAIL" : "PASS", cases - bad, cases);
    return bad ? 1 : 0;
}
