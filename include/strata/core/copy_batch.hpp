#pragma once
// copy_batch.hpp - STRATA_DECODE_COPY_BATCH=1 (LEADS.md Part T row T7, default off): the decode side's independent
// expert H2D copies as hipMemcpyBatchAsync groups on ROCm.  This header is the part the host can test: no GPU header
// is included.  It holds
//   - the switch's parse (`decode_copy_batch_parse`), the compile-time guard (`kHipMemcpyBatchCompiled`) and the
//     once-latched runtime state (`BatchLatch`),
//   - the grouping (`plan_groups`): the copies in their given order, cut into groups of at most `max_group`,
//   - `run_copies`: the driver, written against a backend with `single(dst, src, bytes)` and
//     `batch(dsts, srcs, sizes, n)` (both return 0 on success), so a test can run it with a fake backend that
//     copies into plain memory and records its calls.
// Rules the driver keeps (the test checks each):
//   - off (the default): every copy goes through `single`, in order, and `batch` is never called: the loop the
//     engine always ran;
//   - on: groups of 2..max_group go through `batch`; a group of one goes through `single`; the copies keep their
//     order inside a group and the groups keep theirs, so the stream order is that of the loop;
//   - a batch the backend refuses (non-zero) is redone as singles for that group, and the latch turns the batch
//     path off for the rest of the process (one warning from the caller): nothing is dropped and nothing is copied
//     twice into a different place (the same bytes land in the same slots, so a redo is harmless);
//   - the first failing single ends the run and its code is returned.
// The engine's own backend is in dma_batch.hpp (`copy_blobs`, HIP branch); ROCm 10.2 (HIP 7.17) is the first
// release that fixed the host block behind a long kernel and split batches per device (MOD_PLAN.md / LEADS.md
// Part T row T7), so the guard below compiles the HIP call only from HIP 7.17 on (or with -DSTRATA_HIP_MEMCPY_BATCH=1).
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace strata::core {

/// STRATA_DECODE_COPY_BATCH: only the text "1" switches it on (default off).
inline bool decode_copy_batch_parse(const char* v) { return v != nullptr && v[0] == '1' && v[1] == '\0'; }

/// The compile-time guard: the HIP call is compiled only when it is known to exist with the batched, per-device
/// implementation.  `hip_version` is HIP_VERSION (major * 10^7 + minor * 10^5 + patch); 70170000 is HIP 7.17.0
/// (ROCm 10.2.0).  `force` (-DSTRATA_HIP_MEMCPY_BATCH=1) compiles it on any HIP, `forbid`
/// (-DSTRATA_NO_HIP_MEMCPY_BATCH) never.  A build that does not compile it treats the switch as off (one stderr line).
constexpr bool hip_memcpy_batch_compiled(long hip_version, bool force, bool forbid) {
    return !forbid && (force || hip_version >= 70170000L);
}

/// The most copies in one batch group (the same cap as the CUDA 13 path in dma_batch.hpp).
constexpr size_t kCopyBatchMaxGroup = 128;

struct CopyGroup {
    size_t first;   ///< index of the group's first copy
    size_t count;   ///< copies in the group (1: a single copy)
};

/// `n` copies cut, in order, into groups of at most `max_group` (>= 2; a smaller value gives groups of one).  The last
/// group may be shorter; a group of one is a single copy.  Covers 0..n-1 exactly once.
inline std::vector<CopyGroup> plan_groups(size_t n, size_t max_group) {
    std::vector<CopyGroup> g;
    if (max_group < 2) max_group = 1;
    for (size_t i = 0; i < n;) {
        const size_t c = (n - i) < max_group ? (n - i) : max_group;
        g.push_back({i, c});
        i += c;
    }
    return g;
}

/// Whether the batch path is still allowed: starts true, `fail()` latches it off for good (a refused batch).
struct BatchLatch {
    bool ok = true;
    int failures = 0;
    void fail() { ok = false; ++failures; }
};

/// The driver.  `Backend`: `int single(void* dst, const void* src, size_t bytes)` and
/// `int batch(void* const* dsts, const void* const* srcs, const size_t* sizes, size_t n)`; 0 = success.
/// `on`: the switch (decode_copy_batch_parse) AND the build has the call AND the runtime has it.
/// Returns 0, or the first non-zero code of a failing single.  `batches` (optional) counts the batch calls made.
template <class Backend>
inline int run_copies(Backend& be, void* const* dst, const void* const* src, const size_t* bytes, size_t n, bool on,
                      BatchLatch& latch, size_t max_group = kCopyBatchMaxGroup, size_t* batches = nullptr) {
    if (n == 0) return 0;
    if (!on || !latch.ok || n < 2) {   // the loop, exactly: one copy each, in order
        for (size_t i = 0; i < n; ++i)
            if (const int e = be.single(dst[i], src[i], bytes[i]); e != 0) return e;
        return 0;
    }
    for (const CopyGroup& g : plan_groups(n, max_group)) {
        bool done = false;
        if (g.count >= 2 && latch.ok) {
            if (batches != nullptr) ++*batches;
            if (be.batch(dst + g.first, src + g.first, bytes + g.first, g.count) == 0) done = true;
            else latch.fail();   // refused: this group and every later one go one copy at a time
        }
        if (!done)
            for (size_t i = g.first; i < g.first + g.count; ++i)
                if (const int e = be.single(dst[i], src[i], bytes[i]); e != 0) return e;
    }
    return 0;
}

}  // namespace strata::core
