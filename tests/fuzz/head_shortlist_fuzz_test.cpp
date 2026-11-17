// Fuzz of the SHL1 shortlist table parser (include/strata/program/head_shortlist.hpp shortlist_parse): prefixes, byte flips and
// random blobs of a valid table. Asserts: no sanitizer report, no allocation over 256 MiB, and a true return only for a
// consistent table (cluster ids in [0, k), sizes matching the header).
// Run (host-only, no engine, no GPU):
//   clang++ -std=c++20 -fsanitize=address,undefined -fno-sanitize-recover=undefined -g -O1 -Iinclude -Itests/fuzz tests/fuzz/head_shortlist_fuzz_test.cpp -o head_shortlist_fuzz_test && ./head_shortlist_fuzz_test
#include "fuzz_common.hpp"
#include "strata/program/head_shortlist.hpp"

using namespace strata::program;

static void put32(std::vector<uint8_t>& b, uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back((uint8_t) (v >> (8 * i))); }

int main() {
    std::vector<uint8_t> valid = {'S', 'H', 'L', '1'};
    const uint32_t vocab = 40, d = 6, k = 5;
    put32(valid, vocab); put32(valid, d); put32(valid, k);
    for (uint32_t i = 0; i < vocab; ++i) put32(valid, i % k);
    for (uint32_t i = 0; i < k * d; ++i) { float f = 0.25f * (float) i; uint32_t u; std::memcpy(&u, &f, 4); put32(valid, u); }
    size_t ok_count = 0, bad_count = 0;
    {   // the valid sample itself parses
        ShortlistTable t; std::string err;
        if (!shortlist_parse(valid.data(), valid.size(), t, err)) fuzz::fail("valid sample refused", err.c_str());
    }
    // a header claiming 4 billion rows, columns and clusters, with a file that is only the header, is refused
    {
        std::vector<uint8_t> h = {'S', 'H', 'L', '1'};
        put32(h, 0xffffffffu); put32(h, 0xffffffffu); put32(h, 0xffffffffu);
        ShortlistTable t; std::string err;
        if (shortlist_parse(h.data(), h.size(), t, err)) fuzz::fail("4-billion header accepted", "header only");
    }
    const size_t cases = fuzz::run_all(valid, [&](const uint8_t* p, size_t n) {
        ShortlistTable t; std::string err;
        bool ok = false;
        try { ok = shortlist_parse(p, n, t, err); } catch (const std::exception& e) { fuzz::fail("exception", e.what()); }
        if (!ok) { ++bad_count; return; }
        ++ok_count;
        if (t.cluster.size() != t.vocab || t.centre.size() != (size_t) t.k * t.n_embd) fuzz::fail("size mismatch", "accepted table");
        for (int32_t c : t.cluster) if (c < 0 || (uint32_t) c >= t.k) fuzz::fail("cluster out of range", "accepted table");
    });
    if (fuzz::g_big_allocs != 0) fuzz::fail("allocation over the cap", "refused allocations seen");
    std::printf("head_shortlist_fuzz_test: PASS, %zu cases (%zu accepted, %zu refused), %zu over-cap allocations\n", cases, ok_count, bad_count, fuzz::g_big_allocs);
    return 0;
}
