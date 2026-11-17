// include/strata/kernels/raw_buffer.hpp - LEADS.md Part L L11: buffer-resource ("raw buffer") loads for gfx11, the weight read of the authors'
// decode matvec (int4-wmma.cu:48-60 builds the descriptor with __builtin_amdgcn_make_buffer_rsrc and loads with raw_buffer_load_b128 and a
// streaming cache policy).  Written here from the ISA documentation's descriptor layout; nothing of the fork is copied.
//
// A buffer load takes a 4-dword descriptor (base, size, flags) in scalar registers and a per-lane byte offset: the address is base + offset
// with the bounds check done by the hardware (offset >= size reads 0), the offset is a plain 32-bit VGPR (no 64-bit address arithmetic per
// load), and the load counts against vmcnt only, as global_load does.  The question for our decode matvec is whether it beats the global-
// pointer form of STRATA_HIP_GLOBAL_WEIGHTS (T-050: flat_load 13,119 -> 108) on the Q2_0 / IQ2_XS row streams: tools/pancake/probes/raw_buffer.hip.cpp
// measures that; the engine port (STRATA_HIP_RAW_BUFFER_WEIGHTS) waits for it (28 STRATA_GBLOB call sites in iq_kernels.cu take plain
// pointers into templated row_dot helpers: the port is a load-wrapper type threaded through them, not a macro).
//
// The compiler of the Windows SDK (clang 19.0.0git, ROCm 6.2) has no __builtin_amdgcn_make_buffer_rsrc; the LLVM intrinsic reached through
// an asm label compiles and emits buffer_load_b32 / b64 / b128 (verified, T-ledger row of this lead).  A box with a newer compiler may use the
// builtin with the same descriptor words.
#pragma once
#include <cstdint>

namespace strata::kernels::rawbuf {

struct Rsrc { uint32_t w[4]; };

/// the fourth descriptor word for gfx11 (DST_SEL_XYZW, a 32-bit raw format, bounds check by num_records in bytes): 0x31004000, the value
/// Composable Kernel uses for gfx11 / gfx12 `[unverified: from memory of that header, not read here]`; gfx103x used 0x31014000.  The probe
/// takes it as --flags so the box run can try both; a wrong word shows as zeros or a fault in the parity line.
constexpr uint32_t kGfx11Word3 = 0x31004000u;
constexpr uint32_t kGfx10Word3 = 0x31014000u;

/// descriptor of [base, base + bytes): word0 = base low, word1 = base bits 47:32 (stride 0 in 63:48), word2 = size in bytes, word3 = flags
inline Rsrc make_rsrc(uint64_t base, uint32_t bytes, uint32_t word3 = kGfx11Word3) {
    return Rsrc{{(uint32_t) (base & 0xffffffffu), (uint32_t) ((base >> 32) & 0xffffu), bytes, word3}};
}

/// the cache-policy operand (aux): glc = 1, slc = 2, dlc = 4 on gfx10 / gfx11; slc marks a streamed (read once) line
constexpr int kAuxNone = 0, kAuxGlc = 1, kAuxSlc = 2, kAuxDlc = 4;

#if defined(__HIPCC__)
typedef int rb_v2i __attribute__((ext_vector_type(2)));
typedef int rb_v4i __attribute__((ext_vector_type(4)));
typedef int rb_rsrc_t __attribute__((ext_vector_type(4)));
extern "C" __device__ int strata_rb_load_i32(rb_rsrc_t, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.i32");
extern "C" __device__ rb_v2i strata_rb_load_v2i32(rb_rsrc_t, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v2i32");
extern "C" __device__ rb_v4i strata_rb_load_v4i32(rb_rsrc_t, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

__device__ __forceinline__ rb_rsrc_t rb_vec(const Rsrc& r) { return rb_rsrc_t{(int) r.w[0], (int) r.w[1], (int) r.w[2], (int) r.w[3]}; }
__device__ __forceinline__ int rb_load32(const Rsrc& r, int byte_off, int aux = 0) { return strata_rb_load_i32(rb_vec(r), byte_off, 0, aux); }
__device__ __forceinline__ rb_v2i rb_load64(const Rsrc& r, int byte_off, int aux = 0) { return strata_rb_load_v2i32(rb_vec(r), byte_off, 0, aux); }
__device__ __forceinline__ rb_v4i rb_load128(const Rsrc& r, int byte_off, int aux = 0) { return strata_rb_load_v4i32(rb_vec(r), byte_off, 0, aux); }
#endif

}  // namespace strata::kernels::rawbuf
