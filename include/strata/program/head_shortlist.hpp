#pragma once
// MOD_PLAN.md section 6 item 38 / LEADS.md Part J, STRATA_HEAD_SHORTLIST=<file.shl> (opt-in, LOSSY): the shortlist output
// head.  The vocabulary is clustered offline (tools/pancake/head_cluster.py writes the table); per column the K cluster
// centres are scored against the head's input, the best `n` clusters are kept and the greedy pick is the largest logit among
// the rows of the kept clusters.  A token outside the kept clusters can never win, so the pick can differ from the full
// head's.  Our own code from the idea only (kindi, AGPL: nothing copied).  Pure host logic, no engine types, so a ctest
// covers the table format and the selection rule; the device twin is src/kernels/cuda/head_shortlist.cu (not built here).
// TABLE (little endian, written by head_cluster.py save_table): "SHL1", u32 vocab, u32 n_embd, u32 K, i32 cluster[vocab],
// f32 centre[K * n_embd].
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace strata::program {

struct ShortlistTable {
    uint32_t vocab = 0, n_embd = 0, k = 0;
    std::vector<int32_t> cluster;   // [vocab]: the cluster of each head row, in [0, k)
    std::vector<float> centre;      // [k * n_embd]: the mean raw row of each cluster
};

/// Parse a table image.  False with `err` set for a wrong magic, a short or long file, a zero dimension or a cluster id
/// outside [0, k).
inline bool shortlist_parse(const uint8_t* b, size_t n, ShortlistTable& t, std::string& err) {
    if (n < 16 || std::memcmp(b, "SHL1", 4) != 0) { err = "not a SHL1 table"; return false; }
    uint32_t v, d, k;
    std::memcpy(&v, b + 4, 4); std::memcpy(&d, b + 8, 4); std::memcpy(&k, b + 12, 4);
    if (v == 0 || d == 0 || k == 0) { err = "a zero dimension"; return false; }
    const uint64_t want = 16ull + 4ull * v + 4ull * k * (uint64_t) d;
    if ((uint64_t) n != want) { err = "size " + std::to_string(n) + " but the header says " + std::to_string(want); return false; }
    t.vocab = v; t.n_embd = d; t.k = k;
    t.cluster.resize(v);
    std::memcpy(t.cluster.data(), b + 16, 4ull * v);
    for (uint32_t i = 0; i < v; ++i)
        if (t.cluster[i] < 0 || (uint32_t) t.cluster[i] >= k) { err = "cluster id out of range at row " + std::to_string(i); return false; }
    t.centre.resize((size_t) k * d);
    std::memcpy(t.centre.data(), b + 16 + 4ull * v, 4ull * k * d);
    return true;
}

inline bool shortlist_load(const char* path, ShortlistTable& t, std::string& err) {
    FILE* f = std::fopen(path, "rb");
    if (!f) { err = std::string("cannot open ") + path; return false; }
    std::vector<uint8_t> buf;
    uint8_t chunk[1 << 16];
    for (size_t r; (r = std::fread(chunk, 1, sizeof chunk, f)) > 0;) buf.insert(buf.end(), chunk, chunk + r);
    std::fclose(f);
    return shortlist_parse(buf.data(), buf.size(), t, err);
}

/// How many clusters to keep: env_n (STRATA_HEAD_SHORTLIST_CLUSTERS_SCORED) when > 0, clamped to [1, k], else
/// ceil(frac * k) (head_cluster.py shortlist_argmax's rule; frac 0.13 there), at least 1.
inline int shortlist_clusters_scored(int k, int env_n, double frac = 0.13) {
    if (k <= 0) return 0;
    if (env_n > 0) return env_n > k ? k : env_n;
    const int n = (int) std::ceil(frac * (double) k - 1e-9);
    return n < 1 ? 1 : (n > k ? k : n);
}
inline int shortlist_clusters_scored_from_env(int k) {
    const char* e = std::getenv("STRATA_HEAD_SHORTLIST_CLUSTERS_SCORED");
    return shortlist_clusters_scored(k, e != nullptr ? std::atoi(e) : 0);
}

/// The centre score: 256 lane sums (lane i adds a[j] * c[j] for j = i, i + 256, ... in order), then a halving tree
/// (lane i += lane i + s for s = 128 ... 1).  The device kernel keeps this order; a build that contracts to fma may still
/// differ in the last bit, so a tie between two centres is not promised to break the same way on the device.
inline float shortlist_dot(const float* a, const float* c, int n) {
    float lane[256];
    for (int i = 0; i < 256; ++i) {
        float s = 0.0f;
        for (int j = i; j < n; j += 256) s += a[j] * c[j];
        lane[i] = s;
    }
    for (int s = 128; s >= 1; s >>= 1)
        for (int i = 0; i < s; ++i) lane[i] += lane[i + s];
    return lane[0];
}

/// Keep the best `n` of the `k` cluster scores: the larger score first, the lower cluster index on a tie, a NaN score last.
/// -> keep[k] (1 = kept).  n is clamped to [1, k].
inline std::vector<uint8_t> shortlist_select(const float* score, int k, int n) {
    std::vector<uint8_t> keep((size_t) (k > 0 ? k : 0), 0);
    if (k <= 0) return keep;
    n = n < 1 ? 1 : (n > k ? k : n);
    for (int c = 0; c < k; ++c) {
        int rank = 0;   // the clusters that sort before c; c is kept when fewer than n do
        for (int o = 0; o < k && rank < n; ++o) {
            if (o == c) continue;
            const bool on = std::isnan(score[o]), cn = std::isnan(score[c]);
            const bool before = on != cn ? cn : (on ? o < c : (score[o] > score[c] || (score[o] == score[c] && o < c)));
            rank += before ? 1 : 0;
        }
        if (k > 0) keep[(size_t) c] = rank < n ? 1 : 0;
    }
    return keep;
}

/// The greedy pick among the rows of the kept clusters: the lowest row index of the largest value above -inf (NaN never),
/// 0 when no kept row qualifies (the engine's argmax_rows answer for a row with nothing above -inf).
inline int32_t shortlist_pick(const float* logits, const int32_t* cluster, int vocab, const std::vector<uint8_t>& keep) {
    int32_t best = 0;
    float bv = -INFINITY;
    for (int v = 0; v < vocab; ++v) {
        if (!keep[(size_t) cluster[v]]) continue;
        const float x = logits[v];
        if (x > bv) { bv = x; best = v; }
    }
    return best;
}

/// Rows scored in exact arithmetic for this keep set (the work a gathered GEMV would do), without the K centres.
inline int64_t shortlist_rows_scored(const int32_t* cluster, int vocab, const std::vector<uint8_t>& keep) {
    int64_t n = 0;
    for (int v = 0; v < vocab; ++v) n += keep[(size_t) cluster[v]] ? 1 : 0;
    return n;
}

}  // namespace strata::program
