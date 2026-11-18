// The elastic K/V's VMM ranges (vmm.hpp): chunks map readable and writable, a chunk unmapped from one range and mapped
// into another keeps its bytes (what the expert cache hands the K/V), and a range releases what it holds.
#include "strata/core/vmm.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using strata::core::VmmChunk;
using strata::core::VmmRange;

static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

int main() {
    if (cudaFree(nullptr) != cudaSuccess || !strata::core::vmm_available()) {
        std::printf("vmm_test: no CUDA device with virtual memory - skipped\n");
        return 0;
    }
    const uint64_t G = strata::core::vmm_granularity();
    CHECK(G >= 4096);
    VmmRange a, b;
    CHECK(a.reserve(8 * G + 1));   // rounds up
    CHECK(a.chunks() == 9);
    CHECK(!a.zero_backed() && !b.zero_backed());   // default: byte-identical to before, nothing is mapped until asked
    CHECK(b.reserve(4 * G));
    CHECK(a.mapped_count() == 0);
    CHECK(a.map_range(0, 6, [] { return (VmmChunk) 0; }));
    CHECK(a.mapped_count() == 6 && a.mapped(5) && !a.mapped(6));
    std::vector<uint8_t> h(6 * G), back(6 * G);
    for (size_t i = 0; i < h.size(); ++i) h[i] = (uint8_t) (i * 2654435761u >> 13);
    CHECK(cudaMemcpy(a.base(), h.data(), h.size(), cudaMemcpyHostToDevice) == cudaSuccess);
    // chunks 4 and 5 move to b's chunks 1 and 0. The host-to-device copy above may still be in flight (a pageable
    // cudaMemcpy returns once the data is staged), and a chunk unmapped under a copy faults it (an illegal memory access
    // that sticks to every later call), so the device is drained first, as generate.cpp does before it unmaps
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    const VmmChunk c4 = a.unmap(4), c5 = a.unmap(5);
    CHECK(c4 != 0 && c5 != 0 && !a.mapped(4) && a.unmap(4) == 0);
    std::vector<VmmChunk> give = {c4, c5};   // map_range takes them from the back: chunk 0 <- c5, chunk 1 <- c4
    CHECK(b.map_range(0, 2, [&] { VmmChunk x = give.back(); give.pop_back(); return x; }));
    CHECK(cudaMemcpy(back.data(), b.base(), G, cudaMemcpyDeviceToHost) == cudaSuccess);
    bool same = true;
    for (uint64_t i = 0; i < G; ++i) same = same && back[i] == h[5 * G + i];
    CHECK(same);
    CHECK(cudaMemcpy(back.data(), b.base() + G, G, cudaMemcpyDeviceToHost) == cudaSuccess);
    same = true;
    for (uint64_t i = 0; i < G; ++i) same = same && back[i] == h[4 * G + i];
    CHECK(same);
    // a's first chunks are untouched, and map_range fills a hole with new memory, skipping mapped chunks
    CHECK(cudaMemcpy(back.data(), a.base(), 4 * G, cudaMemcpyDeviceToHost) == cudaSuccess);
    same = true;
    for (uint64_t i = 0; i < 4 * G; ++i) same = same && back[i] == h[i];
    CHECK(same);
    CHECK(a.map_range(3, 7, [] { return (VmmChunk) 0; }) && a.mapped_count() == 7);
    CHECK(cudaMemset(a.base() + 4 * G, 7, 3 * G) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess);
    size_t f0 = 0, f1 = 0, t = 0;
    cudaMemGetInfo(&f0, &t);
    a.release();
    b.release();
    for (int i = 0; i < 20; ++i) {   // the free-memory counter may lag a release by a moment on Windows (WDDM)
        cudaMemGetInfo(&f1, &t);
        if (f1 >= f0 + 8 * G) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    CHECK(f1 >= f0 + 8 * G);   // 7 + 2 chunks went back (the driver may round)
    CHECK(cudaGetLastError() == cudaSuccess);

    // STRATA_VMM_ZERO_BACK=1 (LEADS.md Part N lead N6, kvcached's zero page): unmapped chunks read zeros through one shared
    // read-only chunk, a real chunk mapped over one hides it, and an unmapped chunk reads zeros again (never the bytes the
    // real chunk held).  Needs the same memory as a plain range (one extra chunk).
    // ZERO_BACK diag: every step below is independent and names itself on failure ("zero-back step N name"); a wrong read
    // prints the first wrong byte, its chunk, its value and the wrong-byte count per chunk.  When the read after unmap fails,
    // step 5 also re-reads (after a sync and a pause), copies device-to-device first, and reads the neighbours, so one run
    // tells stale mapping from a copy-path problem from a lost zero mapping.  Every step prints PASS / FAIL / SKIPPED, and the
    // lever's own decisions (granularity, map / access / memset results, read-back) are on stderr.
#ifdef _WIN32
    _putenv_s("STRATA_VMM_ZERO_BACK", "1");
#else
    setenv("STRATA_VMM_ZERO_BACK", "1", 1);
#endif
    {
        // fell: the lever undid itself (its start-up read-back or its check after an unmap found a chunk that does not read zeros, or a
        // call failed; vmm.cpp says why on stderr): the range is then a plain one, which is a safe outcome and not a test failure, but
        // the steps that need zeros are SKIPPED and the run says FALLBACK, so it is never mistaken for a pass of the zero backing.
        bool fell = false;
        auto done = [&](int n, const char* name, bool ok, bool always = false) {
            if (fell && !always) { std::printf("zero-back step %d %s: SKIPPED (fell back)\n", n, name); return false; }
            if (!ok) { std::fprintf(stderr, "FAIL zero-back step %d %s\n", n, name); std::printf("zero-back step %d %s: FAIL\n", n, name); ++fails; }
            else std::printf("zero-back step %d %s: PASS\n", n, name);
            return ok;
        };
        // copies n bytes at device address p to the host; prints the error name when the copy fails
        auto d2h = [&](std::vector<uint8_t>& dst, const uint8_t* p, size_t n, const char* what) {
            std::fill(dst.begin(), dst.begin() + (std::ptrdiff_t) n, (uint8_t) 0xAB);   // a copy that writes nothing cannot pass for zeros
            const cudaError_t e = cudaMemcpy(dst.data(), p, n, cudaMemcpyDeviceToHost);
            if (e != cudaSuccess) std::fprintf(stderr, "zero-back diag: %s copy failed: %s\n", what, cudaGetErrorString(e));
            return e == cudaSuccess;
        };
        // every byte of chunk c (c counted from chunk0) must equal expect(c); reports where and how many are wrong
        auto bytes_ok = [&](const std::vector<uint8_t>& buf, size_t n, uint64_t chunk0, auto expect) {
            std::vector<unsigned long long> bad(n / G + 1, 0);
            unsigned long long first = ~0ull;
            for (size_t i = 0; i < n; ++i)
                if (buf[i] != expect(chunk0 + i / G)) { if (first == ~0ull) first = i; ++bad[i / G]; }
            if (first == ~0ull) return true;
            const uint64_t c = chunk0 + first / G;
            std::fprintf(stderr, "zero-back diag: first wrong byte at +%llu (chunk %llu, offset %llu in it) is 0x%02x, expected 0x%02x; wrong bytes per chunk:",
                         first, (unsigned long long) c, (unsigned long long) (first % G), (unsigned) buf[first], (unsigned) expect(c));
            for (size_t k = 0; k < n / G; ++k) std::fprintf(stderr, " c%llu=%llu", (unsigned long long) (chunk0 + k), bad[k]);
            std::fprintf(stderr, "\n");
            return false;
        };
        auto zero = [](uint64_t) { return 0; };
        VmmRange z;
        std::vector<uint8_t> zb(4 * G, 0xAB);
        const bool r1 = z.reserve(4 * G);
        if (r1 && !z.zero_backed() && z.zero_fell_back()) {
            std::printf("zero-back step 1 reserve zero-backed: FALLBACK (the start-up self-check undid the zero backing, see the stderr lines above)\n");
            fell = true;
        }
        const bool s1 = done(1, "reserve zero-backed", r1 && z.zero_backed() && z.mapped_count() == 0 && !z.mapped(0));
        // 2: reading unmapped chunks is legal and gives zeros
        done(2, "unmapped range reads zeros", s1 && d2h(zb, z.base(), 4 * G, "range") && bytes_ok(zb, 4 * G, 0, zero));
        // 3: a real chunk maps over chunk 1
        const bool s3 = done(3, "map a real chunk over chunk 1", s1 && z.map_range(1, 2, [] { return (VmmChunk) 0; }) && z.mapped_count() == 1 && z.mapped(1));
        // 4: a write lands in chunk 1 only
        bool s4 = false;
        if (s3) {
            const bool w = cudaMemset(z.base() + G, 9, G) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
            s4 = done(4, "real chunk 1 holds 9s, the others zeros",
                      w && d2h(zb, z.base(), 4 * G, "range") && bytes_ok(zb, 4 * G, 0, [](uint64_t c) { return c == 1 ? 9 : 0; }));
        } else done(4, "real chunk 1 holds 9s, the others zeros (skipped: step 3 failed)", false);
        // 5: unmap chunk 1: it reads zeros again, not the 9s
        VmmChunk c1 = 0;
        if (s3) {
            c1 = z.unmap(1);
            if (z.zero_fell_back() && !z.zero_backed()) {
                std::printf("zero-back step 5 unmapped chunk 1 reads zeros again: FALLBACK (the check after the unmap undid the zero backing, see the stderr lines above)\n");
                fell = true;
            }
            const bool un = c1 != 0 && z.mapped_count() == 0;
            const bool rd = un && d2h(zb, z.base() + G, G, "chunk 1") && bytes_ok(zb, G, 1, zero);
            if (!done(5, "unmapped chunk 1 reads zeros again", rd) && un && !fell) {
                std::vector<uint8_t> t(G, 0xAB);
                const bool sy = cudaDeviceSynchronize() == cudaSuccess;
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                const bool again = d2h(t, z.base() + G, G, "chunk 1 retry") && bytes_ok(t, G, 1, zero);
                std::fprintf(stderr, "zero-back diag: retry after sync (%s) and 200 ms: %s\n", sy ? "ok" : "failed", again ? "zeros (the first read was early or stale)" : "still wrong (the mapping itself is wrong)");
                void* scratch = nullptr;
                if (cudaMalloc(&scratch, G) == cudaSuccess) {
                    const bool cp = cudaMemcpy(scratch, z.base() + G, G, cudaMemcpyDeviceToDevice) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
                    const bool dd = cp && d2h(t, (const uint8_t*) scratch, G, "scratch") && bytes_ok(t, G, 1, zero);
                    std::fprintf(stderr, "zero-back diag: device-to-device copy of chunk 1 then read: %s\n", !cp ? "the copy failed" : dd ? "zeros (only the direct host copy path sees the wrong bytes)" : "still wrong (the device sees the wrong bytes too)");
                    (void) cudaFree(scratch);
                }
                const bool nb = d2h(zb, z.base(), 4 * G, "range") && bytes_ok(zb, 4 * G, 0, zero);
                std::fprintf(stderr, "zero-back diag: whole range after the unmap: %s\n", nb ? "all zeros" : "not all zeros (counts above)");
                (void) cudaGetLastError();
            }
        } else done(5, "unmapped chunk 1 reads zeros again (skipped: step 3 failed)", false);
        // 6: the real chunk goes back to the driver, the range goes away, no sticky error is left
        if (c1 != 0) strata::core::vmm_chunk_free(c1);
        z.release();
        done(6, "release without a sticky error", cudaGetLastError() == cudaSuccess, true);
        (void) s4;
    }
#ifdef _WIN32
    _putenv_s("STRATA_VMM_ZERO_BACK", "");
#else
    unsetenv("STRATA_VMM_ZERO_BACK");
#endif
    std::printf("vmm_test: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
