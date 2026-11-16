// LEADS.md A29: include/strata/core/expert_entropy.hpp round trips bit for bit (empty, one symbol, uniform, skewed
// 2-bit-code-like bytes, a skewed histogram that needs the 15-bit length limit) and codes skewed bytes smaller.  CPU only.
// STRATA_EXPERT_ENTROPY_TIER (expert_entropy_tier.hpp): segmented streams of real block layouts round trip through the
// single-thread decoder and the decode pool, malformed streams are refused, and the staging ring's slot rule holds.
#include "strata/core/expert_entropy.hpp"
#include "strata/core/expert_entropy_tier.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace e = strata::core::entropy;

namespace {
int fails = 0;
void round_trip(const std::vector<uint8_t>& v, const char* what, double max_ratio) {
    const std::vector<uint8_t> c = e::encode(v.data(), v.size());
    std::vector<uint8_t> d;
    if (!e::decode(c.data(), c.size(), d) || d != v) { std::fprintf(stderr, "FAIL %s: round trip\n", what); ++fails; return; }
    const double r = v.empty() ? 0.0 : (double) c.size() / (double) v.size();
    if (max_ratio > 0.0 && r > max_ratio) { std::fprintf(stderr, "FAIL %s: ratio %.3f > %.3f\n", what, r, max_ratio); ++fails; }
    std::printf("%s: %zu -> %zu bytes (%.3f)\n", what, v.size(), c.size(), r);
}

// ---- STRATA_EXPERT_ENTROPY_TIER (core/expert_entropy_tier.hpp): segmented streams on blobs in the real block layouts
// (ggml's Q8_0 and Q4_0 bytes from Gaussian weights, and IQ-like grid indices), the decode pool, the ring's slot rule.

uint16_t to_half(float f) {   // IEEE binary16, round to nearest even (enough for block scales)
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t) ((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (exp <= 0) return (uint16_t) sign;
    if (exp >= 31) return (uint16_t) (sign | 0x7c00u);
    uint32_t h = sign | ((uint32_t) exp << 10) | (man >> 13);
    const uint32_t rest = man & 0x1fffu;
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
    return (uint16_t) h;
}

// n_blocks of ggml block_q8_0 (half d, int8 qs[32]) from N(0, 1) weights
std::vector<uint8_t> q8_0_blob(std::mt19937& rng, int n_blocks) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<uint8_t> out;
    for (int b = 0; b < n_blocks; ++b) {
        float x[32], amax = 0.0f;
        for (float& v : x) { v = g(rng); amax = std::max(amax, std::fabs(v)); }
        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
        const uint16_t dh = to_half(d);
        out.push_back((uint8_t) dh);
        out.push_back((uint8_t) (dh >> 8));
        for (float v : x) out.push_back((uint8_t) (int8_t) std::lround(v * id));
    }
    return out;
}

// n_blocks of ggml block_q4_0 (half d, uint8 qs[16]: low nibbles weights 0..15, high nibbles 16..31), ggml's rounding
std::vector<uint8_t> q4_0_blob(std::mt19937& rng, int n_blocks) {
    std::normal_distribution<float> g(0.0f, 1.0f);
    std::vector<uint8_t> out;
    for (int b = 0; b < n_blocks; ++b) {
        float x[32], amax = 0.0f, mx = 0.0f;
        for (float& v : x) {
            v = g(rng);
            if (std::fabs(v) > amax) { amax = std::fabs(v); mx = v; }
        }
        const float d = mx / -8.0f, id = d != 0.0f ? 1.0f / d : 0.0f;
        const uint16_t dh = to_half(d);
        out.push_back((uint8_t) dh);
        out.push_back((uint8_t) (dh >> 8));
        for (int j = 0; j < 16; ++j) {
            const int q0 = std::min(15, (int) (x[j] * id + 8.5f));
            const int q1 = std::min(15, (int) (x[j + 16] * id + 8.5f));
            out.push_back((uint8_t) (q0 | (q1 << 4)));
        }
    }
    return out;
}

// IQ2_XS-like: half d, 32 uint16 grid indices (9-bit index + 7 sign bits, near uniform), 8 scale bytes
std::vector<uint8_t> iq_like_blob(std::mt19937& rng, int n_blocks) {
    std::vector<uint8_t> out;
    std::uniform_int_distribution<int> u(0, 65535), s(0, 255);
    for (int b = 0; b < n_blocks; ++b) {
        const uint16_t dh = to_half(0.01f + 0.001f * (float) (b % 7));
        out.push_back((uint8_t) dh);
        out.push_back((uint8_t) (dh >> 8));
        for (int j = 0; j < 32; ++j) { const int v = u(rng); out.push_back((uint8_t) v); out.push_back((uint8_t) (v >> 8)); }
        for (int j = 0; j < 8; ++j) out.push_back((uint8_t) (s(rng) & 0x77));
    }
    return out;
}

void seg_round_trip(const std::vector<uint8_t>& v, const char* what, uint32_t seg, double max_ratio,
                    e::SegDecodePool* pool) {
    const std::vector<uint8_t> c = e::encode_segmented(v.data(), v.size(), seg);
    std::vector<uint8_t> d(v.size(), 0xAB);
    if (!e::decode_segmented(c.data(), c.size(), d.data(), d.size()) || d != v) {
        std::fprintf(stderr, "FAIL %s (seg %u): segmented round trip\n", what, seg);
        ++fails;
        return;
    }
    if (e::seg_decoded_bytes(c.data(), c.size()) != v.size()) { std::fprintf(stderr, "FAIL %s: decoded size\n", what); ++fails; }
    if (pool != nullptr) {
        std::vector<uint8_t> p(v.size(), 0xCD);
        if (!pool->decode(c.data(), c.size(), p.data(), p.size()) || p != v) {
            std::fprintf(stderr, "FAIL %s (seg %u): pool decode (%d helpers)\n", what, seg, pool->helpers());
            ++fails;
        }
    }
    const double r = v.empty() ? 0.0 : (double) c.size() / (double) v.size();
    if (max_ratio > 0.0 && r > max_ratio) { std::fprintf(stderr, "FAIL %s: ratio %.3f > %.3f\n", what, r, max_ratio); ++fails; }
    std::printf("tier %s (seg %u%s): %zu -> %zu bytes (%.3f)\n", what, seg, pool ? ", pool" : "", v.size(), c.size(), r);
}

void tier_tests(std::mt19937& rng) {
    const std::vector<uint8_t> q8 = q8_0_blob(rng, 8192);      // 278,528 B
    const std::vector<uint8_t> q4 = q4_0_blob(rng, 16384);     // 294,912 B
    const std::vector<uint8_t> iq = iq_like_blob(rng, 4096);   // 303,104 B
    std::vector<uint8_t> mixed = q4;                           // a blob of gate/up/down in different formats
    mixed.insert(mixed.end(), q8.begin(), q8.end());
    mixed.insert(mixed.end(), iq.begin(), iq.end());
    for (int helpers : {0, 1, 3, 7}) {
        e::SegDecodePool pool(helpers);
        for (uint32_t seg : {1u << 10, 1u << 16, 100000u, 1u << 20}) {
            seg_round_trip(q8, "q8_0", seg, 0.0, &pool);
            seg_round_trip(q4, "q4_0", seg, 0.0, &pool);
            seg_round_trip(iq, "iq-like", seg, 0.0, &pool);
            seg_round_trip(mixed, "mixed", seg, 0.0, &pool);
        }
        // repeated decodes through one pool (the park/wake of the helpers), and a one-segment stream
        for (int r = 0; r < 50; ++r) seg_round_trip(q4, "q4_0 repeat", 1u << 14, 0.0, &pool);
        seg_round_trip(std::vector<uint8_t>(5000, 3), "one symbol", 1u << 16, 0.2, &pool);
        seg_round_trip({}, "empty", 1u << 16, 0.0, &pool);
        seg_round_trip(std::vector<uint8_t>(1, 9), "one byte", 1u << 16, 0.0, &pool);
    }
    // Q4_0's nibbles of Gaussian weights are not uniform: the coder must save something on them; Q8_0 too
    seg_round_trip(q4, "q4_0 saves", 1u << 16, 0.97, nullptr);
    seg_round_trip(q8, "q8_0 saves", 1u << 16, 0.97, nullptr);
    // malformed streams are refused, never half-decoded into a success
    {
        std::vector<uint8_t> c = e::encode_segmented(q4.data(), q4.size(), 1u << 16);
        std::vector<uint8_t> d(q4.size());
        std::vector<uint8_t> t = c;
        t.resize(t.size() - 200);
        if (e::decode_segmented(t.data(), t.size(), d.data(), d.size())) { std::fprintf(stderr, "FAIL truncated tier stream decoded\n"); ++fails; }
        t = c;
        t[0] ^= 1;
        if (e::decode_segmented(t.data(), t.size(), d.data(), d.size())) { std::fprintf(stderr, "FAIL bad magic decoded\n"); ++fails; }
        if (e::decode_segmented(c.data(), c.size(), d.data(), d.size() - 1)) { std::fprintf(stderr, "FAIL wrong size decoded\n"); ++fails; }
        t = c;
        for (int k = 4; k < 132; ++k) t[(size_t) k] = 0x11;   // every length 1: not a prefix code
        if (e::decode_segmented(t.data(), t.size(), d.data(), d.size())) { std::fprintf(stderr, "FAIL non-prefix lengths decoded\n"); ++fails; }
        e::SegDecodePool pool(3);
        t = c;
        t.resize(t.size() - 200);
        if (pool.decode(t.data(), t.size(), d.data(), d.size())) { std::fprintf(stderr, "FAIL pool decoded a truncated stream\n"); ++fails; }
        if (!pool.decode(c.data(), c.size(), d.data(), d.size()) || d != q4) { std::fprintf(stderr, "FAIL pool after a refusal\n"); ++fails; }
    }
    // the ring's slot rule
    {
        e::SlotState s[4];
        int hit = 0;
        if (e::pick_slot(s, 4, 7, 10, 4, &hit) != 0 || hit) { std::fprintf(stderr, "FAIL pick_slot: free slot\n"); ++fails; }
        for (int i = 0; i < 4; ++i) { s[i].idx = 100 + i; s[i].seq = 10; }
        if (e::pick_slot(s, 4, 102, 11, 4, &hit) != 2 || !hit) { std::fprintf(stderr, "FAIL pick_slot: hit\n"); ++fails; }
        if (e::pick_slot(s, 4, 7, 13, 4, &hit) != -1) { std::fprintf(stderr, "FAIL pick_slot: too young\n"); ++fails; }
        s[1].seq = 8;
        s[3].seq = 9;
        if (e::pick_slot(s, 4, 7, 13, 4, &hit) != 1 || hit) { std::fprintf(stderr, "FAIL pick_slot: oldest\n"); ++fails; }
        s[1].busy = true;
        if (e::pick_slot(s, 4, 7, 13, 4, &hit) != 3) { std::fprintf(stderr, "FAIL pick_slot: busy skipped\n"); ++fails; }
        s[3].busy = true;
        if (e::pick_slot(s, 4, 7, 13, 4, &hit) != -1) { std::fprintf(stderr, "FAIL pick_slot: all busy\n"); ++fails; }
    }
}
}  // namespace

int main() {
    std::mt19937 rng(503);
    round_trip({}, "empty", 0.0);
    round_trip(std::vector<uint8_t>(4096, 7), "one symbol", 0.2);
    std::vector<uint8_t> u(1 << 16);
    for (auto& b : u) b = (uint8_t) rng();
    round_trip(u, "uniform", 1.01);
    std::vector<uint8_t> s(1 << 16);   // four 2-bit codes a byte, codes 1/2 twice as likely as 0/3
    std::discrete_distribution<int> cd({1, 2, 2, 1});
    for (auto& b : s) b = (uint8_t) (cd(rng) | cd(rng) << 2 | cd(rng) << 4 | cd(rng) << 6);
    round_trip(s, "skewed codes", 0.99);
    std::vector<uint8_t> g;   // geometric histogram: unlimited Huffman lengths would exceed 15
    for (int sym = 0; sym < 40; ++sym)
        for (uint64_t k = 0; k < (1ull << (sym < 22 ? 22 - sym : 0)) / 64 + 1; ++k) g.push_back((uint8_t) sym);
    std::shuffle(g.begin(), g.end(), rng);
    round_trip(g, "length limit", 0.0);
    std::vector<uint8_t> bad = e::encode(s.data(), s.size());
    bad.resize(140);
    std::vector<uint8_t> d;
    if (e::decode(bad.data(), bad.size(), d)) { std::fprintf(stderr, "FAIL truncated stream decoded\n"); ++fails; }
    tier_tests(rng);
    if (fails) return 1;
    std::printf("expert_entropy_test: passed\n");
    return 0;
}
