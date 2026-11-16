// LEADS.md A29: include/strata/core/expert_entropy.hpp round trips bit for bit (empty, one symbol, uniform, skewed
// 2-bit-code-like bytes, a skewed histogram that needs the 15-bit length limit) and codes skewed bytes smaller.  CPU only.
#include "strata/core/expert_entropy.hpp"

#include <algorithm>
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
    if (fails) return 1;
    std::printf("expert_entropy_test: passed\n");
    return 0;
}
