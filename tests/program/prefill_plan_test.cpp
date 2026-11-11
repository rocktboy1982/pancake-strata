// The prompt planner's host-memory model (#2824): no GPU, no model.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "strata/program/prefill_plan.hpp"

using namespace strata::program::prefill_plan;

static int failed = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++failed; } } while (0)

static uint64_t gib(double g) { return (uint64_t) (g * kGiB); }

// options at the lend sizes the reporter's engine printed: 2048 1,688 slots 2.3 GiB, 4096 2.97K 4.0, 6144 5.8, 8192 7.5
static std::vector<Option> issue_options(int64_t full_ring) {
    std::vector<Option> v;
    const struct { int64_t c; double g; } L[] = {{1024, 1.6}, {2048, 2.3}, {3072, 3.1}, {4096, 4.0}, {5120, 4.9}, {6144, 5.8}, {7168, 6.6}, {8192, 7.5}};
    for (const auto& l : L) v.push_back({l.c, full_ring, full_ring, gib(l.g), 0});
    return v;
}

int main() {
    // #2824: 2x RTX 3060, 32 GB, IQ2_XS: the experts total 33.0 GiB, the GPU caches 13.84, the RAM copy 12.0 of the rest
    Host h;
    h.page_cache = gib(14.5);
    h.h2d_gbps = 8.0;
    h.ssd_gbps = 0.45;
    Tier t;
    t.experts_total = gib(33.0);
    t.gpu_cache = gib(13.84);
    t.file_tier = gib(7.18);
    t.extra_file = gib(0.35);
    const auto opts = issue_options(64);
    // the file sets the reporter computed: 15.1 GiB at 8192, 13.3 at 6144, 11.6 at 4096
    CHECK(std::abs(price(h, t, 29000, opts[7]).file_set / kGiB - 15.03) < 0.15);
    CHECK(std::abs(price(h, t, 29000, opts[5]).file_set / kGiB - 13.33) < 0.15);
    CHECK(std::abs(price(h, t, 29000, opts[3]).file_set / kGiB - 11.53) < 0.15);
    // measured drive reads per 29K prompt: 8192: 28-45 GB, 4096: 0.9-1.4 GB, 6144: 1.2-2.8 GB
    const Cost c8 = price(h, t, 28700, opts[7]), c6 = price(h, t, 28700, opts[5]), c4 = price(h, t, 28700, opts[3]);
    std::printf("8192: set %.1f GiB miss %.2f reads %.1f GB %.0f s | 6144: %.1f GiB %.2f %.1f GB %.0f s | 4096: %.1f GiB %.2f %.1f GB %.0f s\n",
                c8.file_set / kGiB, c8.miss, c8.read_bytes / 1e9, c8.seconds, c6.file_set / kGiB, c6.miss, c6.read_bytes / 1e9,
                c6.seconds, c4.file_set / kGiB, c4.miss, c4.read_bytes / 1e9, c4.seconds);
    CHECK(c8.read_bytes / 1e9 > 28 && c8.read_bytes / 1e9 < 46);
    CHECK(c4.read_bytes == 0 && c4.fits);
    CHECK(c6.read_bytes / 1e9 < 15);   // measured 1.2-2.8 GB: the model is conservative here
    // the planner leaves 8192 for the one that fits
    const Pick p = choose(h, t, 28700, opts, 7);
    std::printf("%s\n", describe(p, h, 28700).c_str());
    CHECK(p.changed);
    CHECK(p.best.opt.chunk >= 3072 && p.best.opt.chunk <= 5120);
    CHECK(p.best.read_bytes < 3e9);

    // P2003 (RTX 3060, IQ2_XS, cgroup 32G, NVMe): 8192 was the fastest of 2048/3328/4096/6144/8192 although it thrashes (measured 480 vs 391 tok/s at 4096).
    // file sets 19.1 / 18.2 / 17.5 / 17.0 / 16.4 GiB against a 17.0 GiB cache; the fast drive makes the extra passes the dearer cost
    Host nv;
    nv.page_cache = gib(17.0);
    nv.h2d_gbps = 25.6;
    nv.ssd_gbps = 2.5;
    Tier t2;
    t2.experts_total = gib(33.0);
    t2.gpu_cache = gib(6.0);
    t2.file_tier = gib(15.0);
    t2.extra_file = gib(0.35);
    std::vector<Option> o4 = {{2048, 96, 96, gib(1.1), 0}, {3328, 96, 96, gib(1.63), 0}, {4096, 96, 96, gib(2.0), 0}, {6144, 96, 96, gib(2.9), 0}, {8192, 96, 96, gib(3.8), 0}};
    const Pick n = choose(nv, t2, 28887, o4, 4);
    std::puts(describe(n, nv, 28887).c_str());
    CHECK(!n.changed);

    // a box where everything fits keeps today's choice
    Host big = h;
    big.page_cache = gib(40);
    const Pick q = choose(big, t, 28700, opts, 7);
    CHECK(!q.changed && q.best.opt.chunk == 8192 && q.best.read_bytes == 0);

    // a box whose file set can never fit (8 GB card + 32 GB RAM): no refusal, no blind shrink: the fewest passes win
    Tier hard;
    hard.experts_total = gib(48);
    hard.gpu_cache = gib(3.0);
    hard.file_tier = gib(40);
    hard.extra_file = gib(0.4);
    Host small = h;
    small.page_cache = gib(22);
    std::vector<Option> o2;
    for (int64_t c = 1024; c <= 8192; c += 1024) o2.push_back({c, 64, 64, gib(0.9 * (double) c / 1024.0), 0});
    const Pick r = choose(small, hard, 28700, o2, 7);
    std::printf("%s\n", describe(r, small, 28700).c_str());
    CHECK(r.best.opt.chunk == 8192);
    // and from a small today's choice it moves UP to fewer passes when nothing fits either way
    const Pick r2 = choose(small, hard, 28700, o2, 1);
    CHECK(r2.best.opt.chunk > 2048);

    // unbuffered reads (no page cache): every chunk reads its set; the biggest chunk reads least
    Host ub = h;
    ub.page_cache = 0;
    CHECK(price(ub, t, 28700, opts[7]).read_bytes < price(ub, t, 28700, opts[3]).read_bytes);

    // lent experts the RAM copy keeps leave the file set: 8192 with all 7.5 GiB kept is 7.5 GiB smaller
    Option kept = opts[7];
    kept.lend_kept = kept.lend_bytes;
    CHECK(price(h, t, 28700, kept).file_set < price(h, t, 28700, opts[7]).file_set - gib(7.4));
    CHECK(price(h, t, 28700, kept).fits);

    // a smaller ring only wins when it brings the set into the cache
    Tier edge = t;
    edge.file_tier = gib(6.2);   // 8192 with the 64-slot ring: 6.2+7.5+.35 = 14.05 GiB, 0.97 of the cache
    std::vector<Option> o3 = {{8192, 64, 64, gib(7.5), 0}, {8192, 16, 64, gib(7.0), 0}};
    const Pick s = choose(h, edge, 28700, o3, 0);
    std::printf("%s\n", describe(s, h, 28700).c_str());
    // request-sized chunks: a prompt that fits today's 8192 stays one chunk; a longer one is cut into equal chunks of at most the plan's
    CHECK(balanced_chunk(4101, 8192, 3840) == 4352);
    CHECK(balanced_chunk(8192, 8192, 3840) == 8192);
    CHECK(balanced_chunk(300, 8192, 3840) == 512);
    CHECK(balanced_chunk(20400, 8192, 3840) == 3584);    // 6 equal chunks (last 2,480), no stub
    CHECK(balanced_chunk(28887, 8192, 3840) == 3840);    // 8 chunks: 7 x 3840 + 2,007
    CHECK(balanced_chunk(8200, 8192, 3840) == 2816);     // 3 chunks of 2,816 instead of 3,840 + 3,840 + 520
    CHECK(balanced_chunk(20400, 8192, 0) == 8192);       // no plan: today's cap
    for (int64_t t = 8193; t < 40000; t += 37) {
        const int64_t c = balanced_chunk(t, 8192, 3840);
        const int64_t n = (t + c - 1) / c;
        CHECK(c % 256 == 0 && c <= 8192);
        CHECK((t - (n - 1) * c) * 5 >= c * 2 || n == 1);   // the last chunk is at least 40% of a chunk
        CHECK(c <= 3840);
        CHECK(n <= (t + 3839) / 3840 + 1);
    }

    // MOD_PLAN 17.5: the chunks of one segment (Prefill::run_impl: full chunks, then the rest)
    CHECK(chunk_lengths(0, 32768).empty());
    CHECK(chunk_lengths(100, 0).empty());
    CHECK((chunk_lengths(32765, 32768) == std::vector<int64_t>{32765}));
    CHECK((chunk_lengths(99997, 32768) == std::vector<int64_t>{32768, 32768, 32768, 1693}));   // a 100,000-token prompt today
    CHECK((chunk_lengths(102397, 32768) == std::vector<int64_t>{32768, 32768, 32768, 4093}));  // ttft_bench's "100K" (102,400)
    CHECK((chunk_lengths(69997, 32768) == std::vector<int64_t>{32768, 32768, 4461}));
    for (int64_t n = 1; n < 140000; n += 997) {
        const auto v = chunk_lengths(n, 32768);
        int64_t sum = 0;
        for (size_t i = 0; i < v.size(); ++i) {
            sum += v[i];
            CHECK(v[i] > 0 && (i + 1 == v.size() || v[i] == 32768));
        }
        CHECK(sum == n);
    }

    // STRATA_PREFILL_EQUAL_TAIL: the pancake preset's chunk 32768 and staged limit 4096 (STRATA_PREFILL_CPU_SHARE_MAX); a
    // single-message prompt of N tokens reads a segment of N - 3 (the assistant header goes through the windows)
    const int64_t C = 32768, S = 4096;
    CHECK(equal_tail_chunk(99997, C, S) == 25088);                 // 32768 x 3 + 1,693 staged -> 25,088 x 3 + 24,733
    CHECK((chunk_lengths(99997, 25088) == std::vector<int64_t>{25088, 25088, 25088, 24733}));
    CHECK(equal_tail_chunk(102397, C, S) == 25600);                // 32768 x 3 + 4,093 staged -> 25,600 x 3 + 25,597
    CHECK(equal_tail_chunk(65997, C, S) == 22016);                 // 32768 x 2 + 461 staged -> 22,016 x 2 + 21,965
    CHECK(equal_tail_chunk(34997, C, S) == 17664);                 // 32768 + 2,229 staged -> 17,664 + 17,333
    CHECK((chunk_lengths(34997, 17664) == std::vector<int64_t>{17664, 17333}));
    CHECK(equal_tail_chunk(32765, C, S) == 0);                     // one chunk
    CHECK(equal_tail_chunk(32768, C, S) == 0);
    CHECK(equal_tail_chunk(49149, C, S) == 0);                     // the last chunk (16,381) streams: STRATA_PREFILL_EQUAL's case
    CHECK(equal_tail_chunk(69997, C, S) == 0);                     // 4,461 streams too
    CHECK(equal_tail_chunk(127997, C, S) == 0);
    CHECK(equal_tail_chunk(65537, C, S) == 0);                     // a 1-token tail is left alone (kEqualTailMinTail)
    CHECK(equal_tail_chunk(98305, C, S) == 0);
    CHECK(equal_tail_chunk(2 * C + 255, C, S) == 0);               // 255 < 256
    CHECK(equal_tail_chunk(2 * C + 256, C, S) == 22016);           // 256: evened
    CHECK(equal_tail_chunk(65537, C, S, 1) == 22016);              // STRATA_PREFILL_EQUAL_TAIL_MIN=1 takes even a 1-token tail
    CHECK(equal_tail_chunk(8192 + 1000, 8192, 1024) == 0);         // evened chunks of 4,608 < kEqualTailMinChunk: left alone
    CHECK(equal_tail_chunk(8192 + 25, 8192, 1024) == 0);           // upstream's 25%-slower case stays as it was
    CHECK(equal_tail_chunk(2 * C + 5000, C, 20000) == 23552);      // a large staged limit: the evened chunks still stream
    CHECK(equal_tail_chunk(2 * C + 5000, C, 24000) == 0);          // ...and when they would not, nothing changes
    CHECK(equal_tail_chunk(0, C, S) == 0 && equal_tail_chunk(-5, C, S) == 0 && equal_tail_chunk(70000, 0, S) == 0);
    CHECK(equal_tail_chunk(99997, C, S, 0) == 0);                  // a minimum tail below 1 is off
    // another start's auto chunk (11-09: 31,488): the plan moves with it
    CHECK(equal_tail_chunk(99997, 31488, S) == 0);                 // 31488 x 3 + 5,533: the tail streams
    CHECK(equal_tail_chunk(3 * 31488 + 2000, 31488, S) == 24320);  // 31488 x 3 + 2,000 staged -> 24,320 x 3 + 23,504
    CHECK((chunk_lengths(3 * 31488 + 2000, 24320) == std::vector<int64_t>{24320, 24320, 24320, 23504}));
    for (const int64_t cap : {8192, 16384, 31488, 32768}) {
        for (const int64_t smin : {1024, 4096}) {
            for (int64_t n = 1; n < 140000; n += 13) {
                const int64_t c = equal_tail_chunk(n, cap, smin);
                const int64_t k = (n + cap - 1) / cap, last = n - (k - 1) * cap;
                const bool staged_tail = k >= 2 && last < smin && last >= kEqualTailMinTail;
                if (c == 0) {
                    // only where the rule does not apply, or the evened chunks would be too short to stream well
                    const int64_t even = std::min(cap, ((n + k - 1) / k + 255) / 256 * 256);
                    CHECK(!staged_tail || even < kEqualTailMinChunk || even < smin || n - (k - 1) * even < smin);
                    continue;
                }
                CHECK(staged_tail);
                CHECK(c % 256 == 0 && c <= cap && c >= kEqualTailMinChunk && c >= smin);
                const int64_t k2 = (n + c - 1) / c, last2 = n - (k2 - 1) * c;
                CHECK(k2 == k);                     // the same number of passes
                CHECK(last2 >= smin && last2 <= c); // every chunk streams
            }
        }
    }

    // the switches as set
    CHECK(equal_tail_min_from(nullptr, nullptr) == 0);
    CHECK(equal_tail_min_from("0", "512") == 0);
    CHECK(equal_tail_min_from("", nullptr) == 0);
    CHECK(equal_tail_min_from("1", nullptr) == kEqualTailMinTail);
    CHECK(equal_tail_min_from("1", "") == kEqualTailMinTail);
    CHECK(equal_tail_min_from("1", "1024") == 1024);
    CHECK(equal_tail_min_from("1", "1") == 1);
    CHECK(equal_tail_min_from("1", "0") == kEqualTailMinTail);
    CHECK(equal_tail_min_from("1", "x") == kEqualTailMinTail);
    std::printf("%s\n", failed ? "FAILED" : "prefill_plan_test: ok");
    return failed ? 1 : 0;
}
