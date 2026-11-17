// src/prefill/gdn_snap_test.cpp - host test of STRATA_GDN_SNAP_IN_CHUNK (LEADS.md Part O O6): the chunk plan with and without in-chunk snapshots, the
// snapshot layout, and a host reference of the snapshot kernel's recurrence (tiny sizes, the arithmetic of gdn_rec_cols_pipe_kernel in the same order): a snapshot
// at position p must be bit for bit the state of a run of p tokens, and taking snapshots must not change the state the run ends in.
#include "strata/prefill/gdn_snap.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace strata::prefill;

static int g_fail = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++g_fail; } \
    } while (0)

// the gated delta rule per token for one head (columns independent): kv = sum_r s[r][col] k[r]; delta = (v[col] - g kv) beta; s = fma(g, s, k delta)
// (the test needs the same function on both sides, not the GPU's row-group tree order)
struct Rec {
    int S, HV;
    std::vector<float> state;   // [row][head][col]
    Rec(int s, int hv) : S(s), HV(hv), state(size_t(s) * size_t(hv) * size_t(s), 0.0f) {}
    float& at(int r, int h, int c) { return state[(size_t(r) * size_t(HV) + size_t(h)) * size_t(S) + size_t(c)]; }
    void step(const std::vector<float>& k, const std::vector<float>& v, float g, float beta, int h) {
        for (int c = 0; c < S; ++c) {
            float kv = 0.0f;
            for (int r = 0; r < S; ++r) kv = std::fmaf(at(r, h, c), k[size_t(r)], kv);
            const float delta = (v[size_t(c)] - g * kv) * beta;
            for (int r = 0; r < S; ++r) at(r, h, c) = std::fmaf(g, at(r, h, c), k[size_t(r)] * delta);
        }
    }
};

static float rnd(uint32_t& x) {
    x = x * 1664525u + 1013904223u;
    return float((x >> 8) & 0xFFFF) / 65536.0f - 0.5f;
}

int main() {
    // plan: a 20000-token prompt, chunk 8192, checkpoints at 5000, 12000, one on a chunk end (8192), one at the prompt end, a zero and a duplicate
    const std::vector<int64_t> cuts = {12000, 5000, 8192, 20000, 0, 5000};
    auto plain = gdn_chunk_plan(20000, 8192, cuts, false);
    auto snap = gdn_chunk_plan(20000, 8192, cuts, true);
    CHECK(snap.size() == 3);    // ceil(20000 / 8192): no extra chunk
    CHECK(plain.size() == 4);   // [0,5000) [5000,8192) [8192,12000) [12000,20000): the cuts at 5000, 8192 and 12000 each end a chunk
    int64_t total = 0;
    for (auto& c : plain) { CHECK(c.start == total); total += c.len; }
    CHECK(total == 20000);
    total = 0;
    for (auto& c : snap) { CHECK(c.start == total); total += c.len; }
    CHECK(total == 20000);
    CHECK(snap[0].snaps == std::vector<int64_t>({5000}));
    CHECK(snap[1].snaps == std::vector<int64_t>({12000}));   // 8192 is the first chunk's end: no snapshot
    CHECK(snap[2].snaps.empty());
    CHECK(gdn_snap_relative(snap[1]) == std::vector<int64_t>({12000 - 8192}));
    CHECK(gdn_chunk_plan(0, 8, {}, true).empty() && gdn_chunk_plan(5, 0, {}, true).empty());
    CHECK(gdn_chunk_plan(100, 8192, {50}, true).size() == 1 && gdn_chunk_plan(100, 8192, {50}, false).size() == 2);
    // the layout is the state's, per snapshot
    CHECK(gdn_snap_offset(0, 0, 0, 0, 128, 48) == 0 && gdn_snap_offset(1, 0, 0, 0, 128, 48) == 128LL * 48 * 128);
    CHECK(gdn_snap_offset(0, 2, 3, 5, 8, 4) == (2 * 4 + 3) * 8 + 5);
    // recurrence: snapshot == truncated run; taking snapshots changes nothing
    const int S = 8, HV = 3, T = 37;
    uint32_t seed = 7;
    std::vector<std::vector<float>> K(T, std::vector<float>(S)), V(T, std::vector<float>(S));
    std::vector<float> G(T), B(T);
    for (int t = 0; t < T; ++t) {
        for (int i = 0; i < S; ++i) { K[size_t(t)][size_t(i)] = rnd(seed) * 0.4f; V[size_t(t)][size_t(i)] = rnd(seed); }
        G[size_t(t)] = 0.5f + 0.5f * (rnd(seed) + 0.5f);
        B[size_t(t)] = 0.3f + 0.4f * (rnd(seed) + 0.5f);
    }
    Rec start(S, HV);
    for (int h = 0; h < HV; ++h)
        for (int r = 0; r < S; ++r)
            for (int c = 0; c < S; ++c) start.at(r, h, c) = rnd(seed) * 0.1f;
    const std::vector<int64_t> want = {1, 5, 16, 36};
    std::vector<std::vector<float>> snaps(want.size());
    size_t next = 0;
    Rec snap_run = start;
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < HV; ++h) snap_run.step(K[size_t(t)], V[size_t(t)], G[size_t(t)], B[size_t(t)], h);
        while (next < want.size() && want[next] == t + 1) { snaps[next] = snap_run.state; ++next; }
    }
    for (size_t k = 0; k < want.size(); ++k) {
        Rec trunc = start;
        for (int t = 0; t < want[k]; ++t)
            for (int h = 0; h < HV; ++h) trunc.step(K[size_t(t)], V[size_t(t)], G[size_t(t)], B[size_t(t)], h);
        CHECK(snaps[k].size() == trunc.state.size() && std::memcmp(snaps[k].data(), trunc.state.data(), snaps[k].size() * sizeof(float)) == 0);
    }
    Rec plain_run = start;
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < HV; ++h) plain_run.step(K[size_t(t)], V[size_t(t)], G[size_t(t)], B[size_t(t)], h);
    CHECK(std::memcmp(plain_run.state.data(), snap_run.state.data(), plain_run.state.size() * sizeof(float)) == 0);
    if (g_fail) { std::printf("gdn_snap_test: %d FAILED\n", g_fail); return 1; }
    std::printf("gdn_snap_test: ok\n");
    return 0;
}
