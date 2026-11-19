// src/kernels/cuda/rdna_stub.cpp - the RDNA kernels' answers in CUDA and gfx906 builds.
// rdna_mmvq.cu uses DPP, permlanex16 and sudot4 and builds on wave32 HIP only; the cmake in strata_gpu_engine.cmake compiles it
// in a STRATA_ENABLE_HIP build and this file everywhere else.  The kernels do not exist here: every caller keeps its own
// path (native_mmvq).
#include "strata/kernels/native_mmvq.hpp"

#include <cstdint>

#if !(defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906))
namespace strata::kernels {
bool rdna_q6k_enabled() { return false; }
bool rdna_q6k_mmvq(const void*, const void*, float*, int, int, int, void*) { return false; }
bool rdna_kq_mmvq(int, const void*, const void*, float*, int, int, int, void*) { return false; }
}  // namespace strata::kernels
#endif
