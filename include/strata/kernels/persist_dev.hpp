// include/strata/kernels/persist_dev.hpp - the device side of a persistent launch (LEADS.md Part O O7, STRATA_PERSIST_LAYER; HIP, gfx11): the protocol of
// include/strata/program/persist_tape.hpp as a reusable kernel template.  Header only, included by the one translation unit that instantiates it
// (src/kernels/persist_layer_parity.cu today; the decode's stage kernels once they have task-number bodies, see persist_launch.hpp).
//
// One launch starts G workgroups.  Each claims the next task number from ONE 64-bit counter (relaxed add), maps it to (segment, task) by binary search over
// the prefix sums, waits until the segment's dependency has finished all its tasks (thread 0 polls the done counter with relaxed loads and a short sleep,
// bounded: on a timeout the error word is set and the wait ends), takes one acquire fence per wave after the wait, runs `Body::run(seg, task)` with the
// whole workgroup, releases and bumps the done counter.  A workgroup only ever waits for earlier segments in claim order, so ONE resident workgroup is
// enough (no co-residency assumption).  Counters are never reset: the host passes the claim base and, per segment, the value the dependency must reach.
// Our own code, from the description of the protocol only (AGPL source not used).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace strata::kernels {

struct PersistDevArgs {
    const int64_t* prefix;                 // nseg + 1
    const int32_t* dep;                    // nseg, -1 none
    const unsigned long long* wait_target; // nseg
    unsigned long long* claim;
    unsigned long long claim_base;
    unsigned long long* done;              // one counter per segment, `done_stride` unsigned long longs apart (16 = one 128 B line)
    int done_stride;
    int nseg;
    int* err;
    long long limit_ticks;                 // bounded wait, in wall_clock64 ticks
};

// Body: a type with `__device__ void run(int seg, int task)`; every thread of the workgroup calls it (it may use __syncthreads inside, all threads of the
// workgroup take the same task).
template <class Body>
__global__ void persist_kernel(PersistDevArgs a, Body body) {
    __shared__ int s_task, s_seg;
    const int64_t total = a.prefix[a.nseg];
    for (;;) {
        if (threadIdx.x == 0) {
            const int64_t idx = int64_t(__hip_atomic_fetch_add(a.claim, 1ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) - a.claim_base);
            int lo = 0, hi = a.nseg;
            while (hi - lo > 1) {
                const int mid = (lo + hi) >> 1;
                if (a.prefix[mid] <= idx) lo = mid; else hi = mid;
            }
            s_seg = idx >= total ? -1 : lo;
            s_task = int(idx - a.prefix[lo]);
        }
        __syncthreads();
        const int seg = s_seg, task = s_task;
        if (seg < 0) return;
        const int d = a.dep[seg];
        if (d >= 0 && threadIdx.x == 0) {
            const unsigned long long want = a.wait_target[seg];
            const long long t0 = wall_clock64();
            unsigned long long* dc = a.done + size_t(d) * size_t(a.done_stride);
            while (__hip_atomic_load(dc, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < want) {
                __builtin_amdgcn_s_sleep(1);
                if (wall_clock64() - t0 > a.limit_ticks) { atomicExch(a.err, 1); break; }
            }
        }
        __syncthreads();
        __atomic_thread_fence(__ATOMIC_ACQUIRE);
        body.run(seg, task);
        __atomic_thread_fence(__ATOMIC_RELEASE);
        __syncthreads();
        if (threadIdx.x == 0) __hip_atomic_fetch_add(a.done + size_t(seg) * size_t(a.done_stride), 1ull, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

}  // namespace strata::kernels
