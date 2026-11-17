#pragma once
// Shared pieces of the host fuzz harnesses (tests/fuzz/*_fuzz_test.cpp): a fixed-seed generator, an allocation guard (any
// single allocation over kAllocCap is counted and refused, so a header that claims billions of entries shows up as a failure
// of the parser, not as an out-of-memory) and the three input families every harness runs: every prefix of a valid sample,
// every position of it with a few byte values, and kBlobs pseudo-random blobs.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <vector>

namespace fuzz {

constexpr size_t kAllocCap = 256u << 20;   // 256 MiB: no parser input here may need more
constexpr int kBlobs = 2000;
inline size_t g_big_allocs = 0;            // allocations over the cap (refused)

struct Rng {
    uint64_t s = 0x9e3779b97f4a7c15ull;
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
};

inline void fail(const char* what, const char* detail) {
    std::fprintf(stderr, "FUZZ FAIL: %s (%s)\n", what, detail);
    std::exit(1);
}

/// Calls f(const uint8_t*, size_t) on every input; returns how many were run.  `f` must return normally.
template <class F>
size_t run_all(const std::vector<uint8_t>& valid, F f) {
    size_t n = 0;
    for (size_t len = 0; len <= valid.size(); ++len) { f(valid.data(), len); ++n; }                 // (a) every prefix
    static const uint8_t vals[] = {0x00, 0xff, 0x80, 0x7f, 0x01};
    std::vector<uint8_t> b;
    for (size_t i = 0; i < valid.size(); ++i)                                                       // (b) single-byte flips
        for (uint8_t v : vals) { b = valid; b[i] = v; f(b.data(), b.size()); ++n; b = valid; b[i] ^= 0x55; f(b.data(), b.size()); ++n; }
    Rng r;
    for (int k = 0; k < kBlobs; ++k) {                                                              // (c) random blobs
        b.assign((size_t) (r.next() % 600), 0);
        for (auto& c : b) c = (uint8_t) r.next();
        if (k % 4 == 0 && !valid.empty())                                                           // a quarter keep the valid head
            for (size_t i = 0; i < 16 && i < valid.size() && i < b.size(); ++i) b[i] = valid[i];
        f(b.data(), b.size()); ++n;
    }
    return n;
}

}  // namespace fuzz

// Allocation guard: one global operator new for the whole program.
void* operator new(size_t n) {
    if (n > fuzz::kAllocCap) { ++fuzz::g_big_allocs; throw std::bad_alloc(); }
    void* p = std::malloc(n ? n : 1);
    if (!p) throw std::bad_alloc();
    return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
