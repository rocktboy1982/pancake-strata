// src/kernels/cuda/rdna_stub.cpp - the RDNA kernels' answers in CUDA and gfx906 builds (MOD_PLAN 95b).
// rdna_gr.cu and rdna_mmvq.cu use DPP, permlanex16 and sudot4 and build on wave32 HIP only; CMakeLists.txt compiles them
// in a STRATA_ENABLE_HIP build and this file everywhere else.  The kernels do not exist here: every caller keeps its own
// path (fused_gr_read_multi, native_mmvq).  Moved out of the two .cu files' #else branches, bodies unchanged.
// tools/pancake/check_stubs.py checks these against fused_gr.hpp / native_mmvq.hpp.
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cstdint>

#if !(defined(STRATA_USE_HIP) && !defined(STRATA_HIP_GFX906))
// rdna_gr.cu: the RDNA read does not exist here; every caller keeps its own path.
namespace strata::kernels {
bool rdna_gr_enabled() { return false; }
bool rdna_r8_scratch_init(long long) { return false; }
const uint16_t* rdna_r8_to_bf16(const int8_t*, long long, long long, int, void*) { return nullptr; }
void rdna_gr_read_multi(const FusedGrArgs*, int, float*, void*, unsigned long long*, int) {}
}  // namespace strata::kernels

// rdna_mmvq.cu: the RDNA GEMVs do not exist here; the callers keep native_mmvq.
namespace strata::kernels {
bool rdna_q6k_enabled() { return false; }
bool rdna_q6k_mmvq(const void*, const void*, float*, int, int, int, void*) { return false; }
bool rdna_kq_mmvq(int, const void*, const void*, float*, int, int, int, void*) { return false; }
}  // namespace strata::kernels
#endif
