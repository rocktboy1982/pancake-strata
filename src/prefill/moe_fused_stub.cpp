// src/prefill/moe_fused_stub.cpp - fused:: (moe_fused.hpp, moe_fused_iq.hpp) in a build without the fused int8 experts.
// Moved out of prefill.cpp (MOD_PLAN 95b), bodies unchanged.  In strata_prefill's sources in every build, empty when
// STRATA_PREFILL_FUSED is set.  tools/pancake/check_stubs.py checks it against the headers (the want_dm_f16 break of
// 2026-11-09 was a header function without its answer here).
#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_iq.hpp"

#include <cstddef>
#include <cstdint>

#ifndef STRATA_PREFILL_FUSED
// #136: the fused int8 experts are CUDA-only (HIP and builds without MMQ keep the MMQ / FP16 paths)
namespace strata::prefill::fused {
bool built() { return false; }
bool gfx12_iq2s_default() { return false; }   // AMD-Q: a HIP build without the fused kernels (STRATA_PREFILL_MMQ off) still links prefill.cpp's share-limit call
bool available() { return false; }
bool enabled() { return false; }
bool requested() { return false; }
size_t act_bytes(int64_t, int64_t) { return 0; }
size_t group_bytes(int64_t, int) { return 0; }
void quantize_act(const float*, int64_t, int64_t, void*, void*) {}
void group(const int32_t*, int64_t, int, int, void*, int32_t*, int32_t*, void*) {}
void experts(const Batch&, int, int64_t, const void*, const void*, const int32_t*, void*, float*, void*) {}
bool native_supported(int, int) { return false; }
void quantize_act_native(const float*, int64_t, int64_t, void*, void*) {}
void experts_native(const Batch&, const NativeGeom&, int, int64_t, const void*, const void*, const int32_t*, void*,
                    float*, void*) {}
bool native_wide() { return false; }
void experts_native(const WideBatch&, const NativeGeom&, int, int64_t, const void*, const void*, const int32_t*, void*,
                    float*, void*) {}
void want_dm_f16(bool) {}                    // STRATA_PF_DM16 (6e31946cb): no fused kernels, the down rows stay FP32
bool dm_f16_used() { return false; }
}  // namespace strata::prefill::fused
#endif
