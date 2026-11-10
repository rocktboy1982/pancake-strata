// src/prefill/moe_mmq_stub.cpp - mmq:: (moe_mmq.hpp) in a build without the MMQ library (strata_mmq).
// Moved out of prefill.cpp (MOD_PLAN 95b), bodies unchanged.  The file is in strata_prefill's sources in every build and is
// empty when the library is linked (STRATA_PREFILL_MMQ set on strata_prefill).  tools/pancake/check_stubs.py checks that every
// mmq:: function the always-built code calls has an answer here, with the header's parameter types.
#include "strata/prefill/moe_mmq.hpp"

#include <cstddef>
#include <cstdint>

#ifndef STRATA_PREFILL_MMQ
// A build without the llama.cpp sources (no STRATA_NATIVE_EXPERTS): no MMQ, the FP16 expert path everywhere.
namespace strata::prefill::mmq {
bool built() { return false; }
bool supported(int) { return false; }
bool fits(int, int64_t) { return false; }
size_t matrix_bytes(int, int64_t, int64_t) { return 0; }
size_t q8_bytes(int64_t, int64_t) { return 0; }
void quantize(const float*, const int32_t*, void*, int, int64_t, int64_t, int64_t, void*, float*) {}
bool fp4_activations(int) { return false; }
void quantize_scatter(const float*, const int32_t*, const int32_t*, void*, int, int64_t, int64_t, int64_t, int, void*) {}
Context::Context() {}
Context::~Context() {}
void Context::run(const Product&, void*) {}
void gather_native(const void*, const void*, size_t, const void*, size_t, void*, void*, void*, const void*, void*, size_t) {}
bool gather_native_group(const GatherGroup&, size_t, size_t, size_t, size_t, void*, size_t, void*, size_t, void*, size_t,
                         void*, size_t) { return false; }
void gather_strata_q2(const uint8_t*, void*, void*, void*) {}
void swiglu(const float*, float*, int64_t, int64_t, bool, void*) {}
bool swiglu_quant_ok(int) { return false; }
void swiglu_quant(const float*, void*, int64_t, bool, void*) {}
void swiglu_scaled(const float*, float*, int64_t, int64_t, const int32_t*, int, const float*, int64_t, void*) {}
void down_row_scales(float*, const int32_t*, int, const float*, int64_t, void*) {}
void iota(int32_t*, int64_t, void*) {}
bool tab_built() { return false; }
bool tab_fits(int, int64_t) { return false; }
size_t overread_bytes(int, int64_t) { return 0; }
bool whole_tiles(int, int64_t, int) { return false; }
int direct_mode() { return 0; }
size_t direct_slot_guard() { return 0; }
void prep_group(const GuardTab&, size_t, void*, void*, size_t, void*) {}
}  // namespace strata::prefill::mmq
#endif
