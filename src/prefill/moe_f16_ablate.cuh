// src/prefill/moe_f16_ablate.cuh - BENCH-ONLY ablation switches of the F16 prompt-expert kernels (LEADS.md Part U, lead U5; method: the
// per-stage ablation ladder of the radiance-rdna3 study).  Every macro below REMOVES ONE STAGE of native_f16_kernel (and, where the stage is
// shared, of the K-quant kernels in moe_f16_kq.cu), so the output is WRONG BY DESIGN: the binary only measures how much time a stage owns.
// Nothing here is defined in the engine's build: a macro needs -DSTRATA_BENCH_ABLATION as well, else the compile stops (the #error below),
// so a serving build can never contain one.  Default (no macro): this header is empty and the kernels are the source they were.
//
//   STRATA_ABL_NO_GLOBAL_LOAD  (alias STRATA_ABL_NO_STAGE_LOAD)  no global read of the weight unit or of the staged activations
//   STRATA_ABL_NO_DEQUANT      native_f16_kernel: no format decode (convert<>) of the weight sub-block into int8 codes + scales
//   STRATA_ABL_NO_LDS_STORE    no LDS write of the weight tile, the scales or the activation tile (commit_stage / ActStage::commit)
//   STRATA_ABL_NO_LDS_READ     no LDS read of the weight fragment, the scales or the activation (B) fragments
//   STRATA_ABL_NO_WIDEN        native_f16_kernel: no int8 -> F16 widening and scale multiply of the A fragment in compute_stage
//   STRATA_ABL_NO_WMMA         no v_wmma_f32_16x16x16_f16 (the accumulators stay opaque so the epilogue is not folded)
//   STRATA_ABL_NO_EPILOGUE     no gate/up or down epilogue (the accumulators are consumed by an empty asm so the WMMAs are kept)
//
// Macros combine (a cumulative ladder = several -D together).  How the stage is removed without letting the compiler delete its neighbours:
// the values a removed stage would have produced are made OPAQUE (an empty `asm volatile` with a "+v" operand: zero instructions, but the
// compiler cannot fold the value or drop the code that consumes it), and values a later stage would have consumed are SUNK the same way
// (`"v"` input only), so the work before them stays live.  What is NOT held fixed: the LDS and VGPR allocation (it can change with a
// stage removed, so occupancy can change), the routing of later layers (wrong hidden states choose other experts: the ladder is
// approximate, say so when you read it), and the next layers' work.  tools/pancake/f16_ablate.py report turns the times into shares.
#pragma once

#include <cstdint>

#if defined(STRATA_ABL_NO_STAGE_LOAD) && !defined(STRATA_ABL_NO_GLOBAL_LOAD)
#define STRATA_ABL_NO_GLOBAL_LOAD 1
#endif

#if defined(STRATA_ABL_NO_GLOBAL_LOAD) || defined(STRATA_ABL_NO_DEQUANT) || defined(STRATA_ABL_NO_LDS_STORE) || \
    defined(STRATA_ABL_NO_LDS_READ) || defined(STRATA_ABL_NO_WIDEN) || defined(STRATA_ABL_NO_WMMA) || defined(STRATA_ABL_NO_EPILOGUE)
#define STRATA_ABL_ANY 1
#if !defined(STRATA_BENCH_ABLATION)
#error A STRATA_ABL_* ablation macro makes the prompt-expert kernels compute wrong results on purpose: define STRATA_BENCH_ABLATION too, in a bench-only build, never in the engine
#endif
#else
#define STRATA_ABL_ANY 0
#endif

#if STRATA_ABL_ANY
namespace strata::prefill::fused::f16::abl {

// (the asm is for the gfx11 device pass only: the host pass of a .cu sees empty bodies, as the kernels' own #if STRATA_F16_DEVICE)
#if STRATA_F16_DEVICE
#define STRATA_ABL_ASM_IO(x) asm volatile("" : "+v"(x))
#define STRATA_ABL_ASM_IN(x) asm volatile("" ::"v"(x))
#else
#define STRATA_ABL_ASM_IO(x) ((void) (x))
#define STRATA_ABL_ASM_IN(x) ((void) (x))
#endif
__device__ __forceinline__ void opaque(uint32_t& x) { STRATA_ABL_ASM_IO(x); }
__device__ __forceinline__ void opaque(float& x) { STRATA_ABL_ASM_IO(x); }
__device__ __forceinline__ void opaque(uint4& v) {
    opaque(v.x);
    opaque(v.y);
    opaque(v.z);
    opaque(v.w);
}
__device__ __forceinline__ void opaque(uint2& v) {
    opaque(v.x);
    opaque(v.y);
}
__device__ __forceinline__ void use(uint32_t x) { STRATA_ABL_ASM_IN(x); }
__device__ __forceinline__ void use(float x) { STRATA_ABL_ASM_IN(x); }
__device__ __forceinline__ void use(const uint4& v) {
    use(v.x);
    use(v.y);
    use(v.z);
    use(v.w);
}
/// Any trivially copyable vector (v16h, v8f, ...) as 32-bit words.
template <class V>
__device__ __forceinline__ void use_vec(const V& v) {
    static_assert(sizeof(V) % 4 == 0, "whole 32-bit words");
    uint32_t w[sizeof(V) / 4];
    __builtin_memcpy(w, &v, sizeof(V));
#pragma unroll
    for (unsigned i = 0; i < sizeof(V) / 4; ++i) use(w[i]);
}
template <class V>
__device__ __forceinline__ void opaque_vec(V& v) {
    static_assert(sizeof(V) % 4 == 0, "whole 32-bit words");
    uint32_t w[sizeof(V) / 4];
    __builtin_memcpy(w, &v, sizeof(V));
#pragma unroll
    for (unsigned i = 0; i < sizeof(V) / 4; ++i) opaque(w[i]);
    __builtin_memcpy(&v, w, sizeof(V));
}

}  // namespace strata::prefill::fused::f16::abl
#endif
