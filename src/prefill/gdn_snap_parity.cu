// src/prefill/gdn_snap_parity.cu - on a card: gdn_recurrence_snapshots (LEADS.md Part O O6, STRATA_GDN_SNAP_IN_CHUNK) against gdn_recurrence_cols, bit for bit.
//
// One layer's recurrence over T tokens from a non-zero state, synthetic inputs of the model's shape (q/k L2-normalised per key head, gates in (0, 1)).
//   1. the snapshot kernel's final state and FP32 output are the plain kernel's, bit for bit (taking snapshots changes nothing);
//   2. snapshot k equals the state of a plain run of snap_pos[k] tokens from the same start, bit for bit, for positions 1, 2, 7, 128, T - 1 (and a list with
//      none, and a list with T itself, which the kernel treats as an ordinary position: its snapshot equals the final state).
// Exit 0: all equal; 1: a difference or a CUDA error.  Run: build/gdn_snap_parity  (needs the card alone, ~20 MB, a second).
#include "strata/prefill/gdn_snap.hpp"
#include "strata/prefill/kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

constexpr int S = 128, HK = 16, HV = 48, C = 10240;   // as kernels.cu: head size, key heads, value heads, q|k|v width

#define CK(x)                                                                                      \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) {                                                                   \
            std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while (0)

float rnd(uint32_t& x) {
    x = x * 1664525u + 1013904223u;
    return float((x >> 8) & 0xFFFF) / 65536.0f - 0.5f;
}

}  // namespace

int main() {
    const int T = 300;
    uint32_t seed = 12345;
    std::vector<float> h(size_t(T) * C), gate(size_t(T) * HV), beta(size_t(T) * HV);
    for (int t = 0; t < T; ++t) {
        float* ht = h.data() + size_t(t) * C;
        for (int i = 0; i < C; ++i) ht[i] = rnd(seed);
        for (int kh = 0; kh < HK; ++kh)   // q and k L2-normalised per key head
            for (int off : {0, HK * S}) {
                double ss = 0;
                for (int i = 0; i < S; ++i) ss += double(ht[off + kh * S + i]) * ht[off + kh * S + i];
                const float inv = float(1.0 / std::sqrt(ss + 1e-6));
                for (int i = 0; i < S; ++i) ht[off + kh * S + i] *= inv;
            }
        for (int v = 0; v < HV; ++v) {
            gate[size_t(t) * HV + v] = -0.05f - 0.5f * (rnd(seed) + 0.5f);   // log decay (negative)
            beta[size_t(t) * HV + v] = 0.2f + 0.6f * (rnd(seed) + 0.5f);
        }
    }
    const size_t sf = size_t(strata::prefill::gdn_snapshot_state_floats());
    std::vector<float> st0(sf);
    for (auto& v : st0) v = rnd(seed) * 0.05f;
    float *dh, *dg, *db, *dstate, *dy, *dsnap;
    int64_t* dpos;
    CK(cudaMalloc(&dh, h.size() * 4));
    CK(cudaMalloc(&dg, gate.size() * 4));
    CK(cudaMalloc(&db, beta.size() * 4));
    CK(cudaMalloc(&dstate, sf * 4));
    CK(cudaMalloc(&dy, size_t(T) * HV * S * 4));
    CK(cudaMalloc(&dsnap, sf * 4 * 8));
    CK(cudaMalloc(&dpos, 8 * sizeof(int64_t)));
    CK(cudaMemcpy(dh, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dg, gate.data(), gate.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, beta.data(), beta.size() * 4, cudaMemcpyHostToDevice));

    auto run_plain = [&](int64_t tokens, std::vector<float>& state, std::vector<float>& y) {
        CK(cudaMemcpy(dstate, st0.data(), sf * 4, cudaMemcpyHostToDevice));
        strata::prefill::gdn_recurrence_cols(dstate, dh, dg, db, dy, tokens, nullptr);
        CK(cudaDeviceSynchronize());
        state.resize(sf);
        y.resize(size_t(tokens) * HV * S);
        CK(cudaMemcpy(state.data(), dstate, sf * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
    };
    int bad = 0;
    std::vector<float> ref_state, ref_y;
    run_plain(T, ref_state, ref_y);
    const std::vector<std::vector<int64_t>> lists = {{1, 2, 7, 128, T - 1}, {}, {5, T}};
    for (const auto& pos : lists) {
        CK(cudaMemcpy(dstate, st0.data(), sf * 4, cudaMemcpyHostToDevice));
        CK(cudaMemset(dsnap, 0x7F, sf * 4 * 8));
        if (!pos.empty()) CK(cudaMemcpy(dpos, pos.data(), pos.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
        strata::prefill::gdn_recurrence_snapshots(dstate, dh, dg, db, dy, T, dpos, int(pos.size()), dsnap, nullptr);
        CK(cudaDeviceSynchronize());
        std::vector<float> state(sf), y(size_t(T) * HV * S);
        CK(cudaMemcpy(state.data(), dstate, sf * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(y.data(), dy, y.size() * 4, cudaMemcpyDeviceToHost));
        if (std::memcmp(state.data(), ref_state.data(), sf * 4) != 0 || std::memcmp(y.data(), ref_y.data(), y.size() * 4) != 0) {
            std::printf("FAIL snapshot kernel (%zu positions): final state or output differs from the plain kernel\n", pos.size());
            ++bad;
        }
        for (size_t k = 0; k < pos.size(); ++k) {
            std::vector<float> st, yy, snap(sf);
            run_plain(pos[k], st, yy);
            CK(cudaMemcpy(snap.data(), dsnap + k * sf, sf * 4, cudaMemcpyDeviceToHost));
            if (std::memcmp(snap.data(), st.data(), sf * 4) != 0) {
                std::printf("FAIL snapshot %zu at %lld differs from a run of that many tokens\n", k, (long long) pos[k]);
                ++bad;
            }
        }
    }
    std::printf("gdn_snap_parity: %s\n", bad ? "FAILED" : "ok");
    return bad ? 1 : 0;
}
