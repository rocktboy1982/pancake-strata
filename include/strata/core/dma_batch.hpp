#pragma once
// dma_batch.hpp - independent host-to-device blob copies as one submission: copy_blobs.
//
// The batched path is midhatn's (PR #807, MIT-compatible contribution): cudaMemcpyBatchAsync for independent
// pinned expert uploads on CUDA 13+, with the Windows alias lookup it found necessary.  eddoursul's fork (F12) uses
// the same call for the adaptive tier's and the prefetch's copies: under WDDM a copy submitted alone raises an
// interrupt when it lands, on the processor its GPU's interrupts go to; a batch raises one.
//
// Modes (STRATA_DMA_BATCH, read once; `dma_batch_mode()`):
//   0  one cudaMemcpyAsync per blob: exactly the loop the engine always ran
//   1  cudaMemcpyBatchAsync (CUDA 13+, groups of 2..128), the copies in any order, the batch in the stream's
//   2  as 1, plus CUDA 13.4's cudaMemcpyFlagPreferOverlapWithCompute (a hint)
//   3  HIP only (set by STRATA_DECODE_COPY_BATCH=1 when STRATA_DMA_BATCH is unset or 0; LEADS.md Part T row T7):
//      hipMemcpyBatchAsync, groups of 2..128 (copy_batch.hpp), compiled from HIP 7.17 (ROCm 10.2) on
// Every other build (HIP without mode 3, CUDA before 13) and every single copy or group over 128 takes the loop.  A batch the
// runtime refuses is redone as the loop (with one warning), never dropped: readiness is signalled only for
// uploads that were submitted.
#include <cuda_runtime.h>

#include "strata/core/copy_batch.hpp"

// The compile-time guard of mode 3 (copy_batch.hpp: hip_memcpy_batch_compiled is the same rule, tested on the host; keep the
// two numbers in step): HIP 7.17.0 (ROCm 10.2.0) or newer, or -DSTRATA_HIP_MEMCPY_BATCH=1; never with
// -DSTRATA_NO_HIP_MEMCPY_BATCH.  -DSTRATA_HIP_MEMCPY_BATCH_NO_FAILIDX selects the 8-argument prototype (no failIdx, the CUDA 13
// shape) if the header of the build has that one.  UNVERIFIED on a real header: the prototype below follows CUDA 12.8's
// cudaMemcpyBatchAsync with failIdx; a mismatch is a compile error (the redeclaration below), never a wrong call.
#if defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && !defined(STRATA_NO_HIP_MEMCPY_BATCH) && defined(HIP_VERSION) && \
    (defined(STRATA_HIP_MEMCPY_BATCH) || HIP_VERSION >= 70170000)
#define STRATA_HIP_MEMCPY_BATCH_COMPILED 1
// weak: a build made on 10.2 that runs on a runtime without the symbol still starts (the pointer test below)
extern "C" hipError_t hipMemcpyBatchAsync(void** dsts, void** srcs, size_t* sizes, size_t count, hipMemcpyAttributes* attrs,
                                          size_t* attrsIdxs, size_t numAttrs,
#if !defined(STRATA_HIP_MEMCPY_BATCH_NO_FAILIDX)
                                          size_t* failIdx,
#endif
                                          hipStream_t stream) __attribute__((weak));
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace strata::core {

/// STRATA_DECODE_COPY_BATCH=1 (default off): the mode it asks for in this build, 3 or 0 (never batched).  Not 3 where the
/// build has no hipMemcpyBatchAsync (CUDA, HIP before 7.17, the guard above) or the runtime lacks the symbol: one stderr line.
inline int decode_copy_batch_mode() {
    if (!decode_copy_batch_parse(std::getenv("STRATA_DECODE_COPY_BATCH"))) return 0;
#if defined(STRATA_HIP_MEMCPY_BATCH_COMPILED)
    if (&hipMemcpyBatchAsync != nullptr) return 3;
    std::fprintf(stderr, "strata: STRATA_DECODE_COPY_BATCH=1: this HIP runtime has no hipMemcpyBatchAsync: off\n");
#else
    std::fprintf(stderr, "strata: STRATA_DECODE_COPY_BATCH=1: this build has no hipMemcpyBatchAsync (needs HIP 7.17 / ROCm 10.2): off\n");
#endif
    return 0;
}

/// The mode: STRATA_DMA_BATCH (0, 1, 2); unset = the engine's default (`default_mode`, set at the call site's first
/// use; 0 until a measured win makes it 1).  With STRATA_DECODE_COPY_BATCH=1 and a mode of 0: 3 on a HIP build that has the
/// call (see decode_copy_batch_mode).  Switch unset: exactly the old result.
inline int dma_batch_mode(int default_mode = 0) {
    static const int mode = [default_mode] {
        const char* e = std::getenv("STRATA_DMA_BATCH");
        const int m = e != nullptr && e[0] != '\0' ? std::atoi(e) : default_mode;
        if (m == 0 && std::getenv("STRATA_DECODE_COPY_BATCH") != nullptr) return decode_copy_batch_mode();
        return m;
    }();
    return mode;
}

namespace detail {

// A registered Windows host pointer may differ from its device-visible alias; the batch API needs the alias
// (cudaMemcpyAsync(kind=HostToDevice) resolves it itself).  The experts' blobs are fixed, so each is looked up once.
inline cudaError_t host_alias(const void* p, const void** out) {
    static std::mutex mu;
    static std::unordered_map<const void*, const void*> cache;
    {
        std::lock_guard<std::mutex> lk(mu);
        const auto it = cache.find(p);
        if (it != cache.end()) { *out = it->second; return cudaSuccess; }
    }
    void* alias = nullptr;
    const cudaError_t e = cudaHostGetDevicePointer(&alias, const_cast<void*>(p), 0);
    if (e != cudaSuccess) return e;
    std::lock_guard<std::mutex> lk(mu);
    cache.emplace(p, alias);
    *out = alias;
    return cudaSuccess;
}

}  // namespace detail

/// `n` independent copies, dst[i] <- src[i] (bytes[i] each), on `stream`.  The sources are immutable pinned host
/// blobs and the destinations disjoint; the caller keeps them alive until the stream completes.  Ordering with
/// earlier and later work on the stream is that of the copies one by one.  Returns the first error (the loop
/// continues past none: the first failing copy ends it).
inline cudaError_t copy_blobs(void* const* dst, const void* const* src, const size_t* bytes, size_t n,
                              cudaStream_t stream, int batch_mode) {
    if (n == 0) return cudaSuccess;
#if !defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906) && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
    constexpr size_t cap = 128;
    if (batch_mode != 0 && n > 1 && n <= cap) {
        std::array<const void*, cap> srcs{};
        bool aliased = true;
        for (size_t i = 0; i < n; ++i)
            if (detail::host_alias(src[i], &srcs[i]) != cudaSuccess) { aliased = false; break; }
        if (aliased) {
            cudaMemcpyAttributes attr{};
            attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
#if CUDART_VERSION >= 13040
            if (batch_mode == 2) attr.flags = cudaMemcpyFlagPreferOverlapWithCompute;
#endif
            size_t first = 0;
            std::array<void*, cap> dsts{};
            std::array<size_t, cap> sizes{};
            for (size_t i = 0; i < n; ++i) { dsts[i] = dst[i]; sizes[i] = bytes[i]; }
            const cudaError_t e = cudaMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), n, &attr, &first, 1, stream);
            if (e == cudaSuccess) return cudaSuccess;
            static bool warned = false;
            if (!warned) {
                warned = true;
                std::fprintf(stderr, "strata: cudaMemcpyBatchAsync failed (%s); the uploads go one copy at a time\n",
                             cudaGetErrorString(e));
            }
            (void) cudaGetLastError();
        }
    }
#elif defined(STRATA_HIP_MEMCPY_BATCH_COMPILED)
    if (batch_mode == 3) {   // STRATA_DECODE_COPY_BATCH=1: groups of 2..128 through hipMemcpyBatchAsync (copy_batch.hpp's driver)
        struct HipBackend {
            cudaStream_t stream;
            int single(void* d, const void* s, size_t b) {
                return (int) cudaMemcpyAsync(d, s, b, cudaMemcpyHostToDevice, stream);
            }
            int batch(void* const* d, const void* const* s, const size_t* sz, size_t k) {
                hipMemcpyAttributes attr{};
                attr.srcAccessOrder = (decltype(attr.srcAccessOrder)) 1;   // hipMemcpySrcAccessOrderStream (the host blobs are immutable)
                size_t idx = 0;
#if !defined(STRATA_HIP_MEMCPY_BATCH_NO_FAILIDX)
                size_t fail = 0;
#endif
                const hipError_t e = hipMemcpyBatchAsync(const_cast<void**>(d), const_cast<void**>(s), const_cast<size_t*>(sz), k,
                                                         &attr, &idx, 1,
#if !defined(STRATA_HIP_MEMCPY_BATCH_NO_FAILIDX)
                                                         &fail,
#endif
                                                         stream);
                if (e != hipSuccess) {
                    static bool warned = false;
                    if (!warned) {
                        warned = true;
                        std::fprintf(stderr, "strata: hipMemcpyBatchAsync failed (%s); the uploads go one copy at a time\n",
                                     hipGetErrorString(e));
                    }
                    (void) hipGetLastError();
                }
                return (int) e;
            }
        } be{stream};
        static BatchLatch latch;   // a refused batch turns the batch path off for the rest of the process
        return (cudaError_t) run_copies(be, dst, src, bytes, n, true, latch);
    }
#else
    (void) batch_mode;
#endif
    for (size_t i = 0; i < n; ++i) {
        const cudaError_t e = cudaMemcpyAsync(dst[i], src[i], bytes[i], cudaMemcpyHostToDevice, stream);
        if (e != cudaSuccess) return e;
    }
    return cudaSuccess;
}

/// The verifier's form: `n` blobs of `bytes` each into consecutive slots of `dst`.
inline cudaError_t copy_expert_blobs(uint8_t* dst, const uint8_t* const* src, int n, size_t bytes,
                                     cudaStream_t stream, int batch_mode) {
    if (n <= 0) return cudaSuccess;
    constexpr size_t cap = 128;
    if (batch_mode == 0 || (size_t) n > cap) {   // the old loop, with no staging of its own
        for (int i = 0; i < n; ++i) {
            const cudaError_t e = cudaMemcpyAsync(dst + (size_t) i * bytes, src[i], bytes, cudaMemcpyHostToDevice, stream);
            if (e != cudaSuccess) return e;
        }
        return cudaSuccess;
    }
    void* d[cap] = {};   // plain arrays: nvcc on Linux (13.0, g++) rejects std::array<T, cap> here ("template argument 2 is invalid")
    const void* s[cap] = {};
    size_t b[cap] = {};
    for (int i = 0; i < n; ++i) { d[i] = dst + (size_t) i * bytes; s[i] = src[i]; b[i] = bytes; }
    return copy_blobs(d, s, b, (size_t) n, stream, batch_mode);
}

}  // namespace strata::core
