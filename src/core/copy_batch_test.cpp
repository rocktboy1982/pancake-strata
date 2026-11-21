// LEADS.md Part T row T7: STRATA_DECODE_COPY_BATCH's host logic (include/strata/core/copy_batch.hpp) against a fake copy backend.
// Validated here: the parse, the compile guard's version rule, the grouping, the driver (off = the loop and never a batch call,
// on = groups, fallback on a refused batch, the first failing single ends the run), and that the batched copies move the same
// bytes as the single copies (a hash of the destination).  NOT validated here: hipMemcpyBatchAsync itself (no ROCm on this PC).
#include "strata/core/copy_batch.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace strata::core;

static int checks = 0;
static void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}

struct FakeBackend {
    int singles = 0, batches = 0, batched_copies = 0;
    int fail_batch_at = -1;    // the Nth batch call (0-based) is refused
    int fail_single_at = -1;   // the Nth single call (0-based) fails with code 7
    std::vector<size_t> batch_sizes;
    int single(void* d, const void* s, size_t b) {
        if (singles++ == fail_single_at) return 7;
        std::memcpy(d, s, b);
        return 0;
    }
    int batch(void* const* d, const void* const* s, const size_t* sz, size_t n) {
        const int at = batches++;
        batch_sizes.push_back(n);
        if (at == fail_batch_at) return 1;   // refused: nothing copied
        for (size_t i = 0; i < n; ++i) std::memcpy(d[i], s[i], sz[i]);
        batched_copies += (int) n;
        return 0;
    }
};

static uint64_t fnv(const std::vector<uint8_t>& v) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t b : v) { h ^= b; h *= 1099511628211ull; }
    return h;
}

struct Rig {   // n blobs of `bytes` each from `host`, into slots of a device image, in a shuffled slot order
    size_t n, bytes;
    std::vector<uint8_t> host, dev;
    std::vector<void*> dst;
    std::vector<const void*> src;
    std::vector<size_t> sz;
    Rig(size_t n_, size_t bytes_) : n(n_), bytes(bytes_), host(n_ * bytes_), dev(n_ * bytes_, 0) {
        for (size_t i = 0; i < host.size(); ++i) host[i] = (uint8_t) (i * 131u + (i >> 7) * 17u + 3u);
        for (size_t i = 0; i < n; ++i) {
            const size_t slot = (i * 7 + 3) % n;   // a permutation when gcd(7, n) == 1 (the cases below keep it so)
            dst.push_back(dev.data() + slot * bytes);
            src.push_back(host.data() + i * bytes);
            sz.push_back(bytes);
        }
    }
};

int main() {
    // switch parse: only "1" is on
    check(!decode_copy_batch_parse(nullptr) && !decode_copy_batch_parse("") && !decode_copy_batch_parse("0") &&
              !decode_copy_batch_parse("2") && !decode_copy_batch_parse("11") && !decode_copy_batch_parse("on"),
          "only the text 1 switches it on");
    check(decode_copy_batch_parse("1"), "1 is on");

    // compile guard: HIP 7.17.0 and newer, or forced; never when forbidden
    check(!hip_memcpy_batch_compiled(70150000L, false, false), "ROCm 10.0 (HIP 7.15): not compiled");
    check(!hip_memcpy_batch_compiled(70169999L, false, false), "HIP 7.16.x: not compiled");
    check(hip_memcpy_batch_compiled(70170000L, false, false), "ROCm 10.2 (HIP 7.17): compiled");
    check(hip_memcpy_batch_compiled(80000000L, false, false), "a later HIP: compiled");
    check(hip_memcpy_batch_compiled(70150000L, true, false), "forced on an older HIP");
    check(!hip_memcpy_batch_compiled(70170000L, true, true) && !hip_memcpy_batch_compiled(80000000L, false, true), "forbidden: never");

    // grouping
    check(plan_groups(0, 128).empty(), "no copies, no groups");
    {
        const auto g = plan_groups(300, 128);
        check(g.size() == 3 && g[0].first == 0 && g[0].count == 128 && g[1].first == 128 && g[1].count == 128 &&
                  g[2].first == 256 && g[2].count == 44, "300 copies: 128 + 128 + 44");
        size_t cover = 0;
        for (const auto& x : g) { check(x.first == cover, "groups are contiguous and in order"); cover += x.count; }
        check(cover == 300, "groups cover every copy exactly once");
    }
    check(plan_groups(1, 128).size() == 1 && plan_groups(1, 128)[0].count == 1, "one copy: one group of one");
    check(plan_groups(5, 1).size() == 5 && plan_groups(5, 0).size() == 5, "max_group < 2: groups of one");
    check(plan_groups(6, 3).size() == 2 && plan_groups(7, 3).size() == 3 && plan_groups(7, 3)[2].count == 1, "7 by 3: 3 + 3 + 1");

    // the reference: the loop of single copies
    Rig ref(37, 4096);
    {
        FakeBackend be;
        BatchLatch latch;
        check(run_copies(be, ref.dst.data(), ref.src.data(), ref.sz.data(), ref.n, false, latch) == 0, "reference loop ok");
        check(be.singles == 37 && be.batches == 0, "off: 37 singles, never a batch call");
    }
    const uint64_t want = fnv(ref.dev);

    // default path (off): the batch call is never made, whatever the size, even with the latch fine and a huge group cap
    for (size_t n : {(size_t) 1, (size_t) 2, (size_t) 37, (size_t) 200}) {
        Rig r(n, 512);
        FakeBackend be;
        BatchLatch latch;
        size_t batches = 0;
        check(run_copies(be, r.dst.data(), r.src.data(), r.sz.data(), n, false, latch, 128, &batches) == 0, "off run ok");
        check(be.batches == 0 && batches == 0 && be.singles == (int) n, "default (off) path never calls the batch backend");
    }

    // on: the same bytes in the same slots as the loop
    {
        Rig r(37, 4096);
        FakeBackend be;
        BatchLatch latch;
        size_t batches = 0;
        check(run_copies(be, r.dst.data(), r.src.data(), r.sz.data(), r.n, true, latch, 16, &batches) == 0, "batched run ok");
        check(fnv(r.dev) == want, "batched copies leave the same bytes as the single copies (hash of the slots)");
        check(be.batches == 3 && batches == 3 && be.batch_sizes[0] == 16 && be.batch_sizes[1] == 16 && be.batch_sizes[2] == 5,
              "37 copies by 16: 16 + 16 + 5");
        check(be.singles == 0 && be.batched_copies == 37, "no single copy beside the batches");
    }
    // on, group of one at the end goes single
    {
        Rig r(33, 256);
        FakeBackend be;
        BatchLatch latch;
        check(run_copies(be, r.dst.data(), r.src.data(), r.sz.data(), r.n, true, latch, 16) == 0, "33 by 16 ok");
        check(be.batches == 2 && be.singles == 1 && be.batched_copies == 32, "33 by 16: two batches and one single");
    }
    // on, one copy: a single (no batch of one)
    {
        Rig r(1, 256);
        FakeBackend be;
        BatchLatch latch;
        check(run_copies(be, r.dst.data(), r.src.data(), r.sz.data(), 1, true, latch) == 0 && be.batches == 0 && be.singles == 1,
              "one copy is a single copy even when on");
    }
    // a refused batch: redone as singles, latched off, the bytes still right, nothing lost
    {
        Rig r(37, 4096);
        FakeBackend be;
        be.fail_batch_at = 0;
        BatchLatch latch;
        size_t batches = 0;
        check(run_copies(be, r.dst.data(), r.src.data(), r.sz.data(), r.n, true, latch, 16, &batches) == 0, "refused batch: run ok");
        check(fnv(r.dev) == want, "refused batch: the same bytes land in the same slots (loop redo)");
        check(!latch.ok && latch.failures == 1 && be.batches == 1 && batches == 1 && be.singles == 37,
              "refused batch: one batch call, then every copy one at a time, latch off");
        // the next call with the latch off never batches
        Rig r2(37, 4096);
        FakeBackend be2;
        check(run_copies(be2, r2.dst.data(), r2.src.data(), r2.sz.data(), r2.n, true, latch, 16) == 0 && be2.batches == 0 &&
                  be2.singles == 37 && fnv(r2.dev) == want, "latched off: the loop");
    }
    // the second batch refused: the first stays done, the rest goes by singles
    {
        Rig r(37, 4096);
        FakeBackend be;
        be.fail_batch_at = 1;
        BatchLatch latch;
        check(run_copies(be, r.dst.data(), r.src.data(), r.sz.data(), r.n, true, latch, 16) == 0, "second batch refused: ok");
        check(fnv(r.dev) == want && be.batched_copies == 16 && be.singles == 21 && be.batches == 2,
              "second batch refused: 16 batched, 21 single, same bytes");
    }
    // a failing single ends the run with its code
    {
        Rig r(10, 128);
        FakeBackend be;
        be.fail_single_at = 3;
        BatchLatch latch;
        check(run_copies(be, r.dst.data(), r.src.data(), r.sz.data(), r.n, false, latch) == 7 && be.singles == 4,
              "a failing single stops the loop with its code");
    }
    {
        Rig r(10, 128);
        FakeBackend be;
        be.fail_batch_at = 0;
        be.fail_single_at = 2;
        BatchLatch latch;
        check(run_copies(be, r.dst.data(), r.src.data(), r.sz.data(), r.n, true, latch, 4) == 7 && be.singles == 3,
              "a failing single in the redo of a refused batch stops it with its code");
    }
    // n = 0
    {
        FakeBackend be;
        BatchLatch latch;
        check(run_copies(be, nullptr, nullptr, nullptr, 0, true, latch) == 0 && be.batches == 0 && be.singles == 0, "no copies: nothing called");
    }
    std::printf("copy_batch_test: %d checks passed\n", checks);
    return 0;
}
