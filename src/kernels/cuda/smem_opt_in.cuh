// Helpers for the per-device dynamic shared-memory opt-in of a launch family.  A layer split launches the same
// pancake-strata: ported from basalt (jesdga95/basalt a9468e5c, MIT, Jesus Gulfo) src/models/qwen4exp/kernels/smem_opt_in.cuh; namespace, includes and
// logging renamed for Strata, the kernels unchanged.
// kernels from one thread per card, so the lazy per-device state is set under a lock and published with a release
// store; a failed opt-in is fatal because the launch would otherwise fail later as invalid.
#pragma once
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include "strata/kernels/klog.hpp"

namespace strata::kernels {

inline int opt_in_device(const char* who) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) {
        klog_error("%s: device %d out of range", who, dev);
        std::abort();
    }
    return dev;
}

inline void opt_in_check(cudaError_t e, const char* who) {
    if (e != cudaSuccess) {
        klog_error("%s: shared-memory opt-in failed: %s", who, cudaGetErrorString(e));
        std::abort();
    }
}

inline std::mutex& opt_in_mutex() {
    static std::mutex m;
    return m;
}

}  // namespace strata::kernels
