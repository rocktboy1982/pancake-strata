// include/strata/kernels/reduce_f32.hpp - STRATA_REDUCE_F32=1 (default OFF; LEADS.md Part U, lead U2): the f64 accumulators and f64 libm calls of the
// legacy (non-native) device kernels replaced by f32 arithmetic.  Idea: radiance-rdna3's tanh-in-double finding (gfx11 has no fast FP64: their
// README says soft-float at 1:64, AMD lists 1:32 - both unverified here); the code is ours.
//
// Kernels with a switch (each is a template on `bool F32`; the default instantiation `false` is the source it was, so the default path is unchanged):
//   src/kernels/cuda/elementwise.cu  silu_kernel                (silu_inplace)
//   src/kernels/cuda/gdn.cu          gdn_l2_kernel, gdn_out_norm_kernel
//   src/kernels/cuda/ple.cu          gnorm_kernel, gate_kernel, bf16_gemv_kernel, block_sum
//   src/kernels/cuda/qsa.cu          indexer_key_append_kernel   (the k-norm sum of squares and the pooled mean's norm)
// NOT bit-exact: an f32 sum differs from the f64 sum in its last bits, so the output differs by a few ulp; the kernels are judged by exactness, so
// the switch is classed KeyIfSet and needs the quality gate before any default.  tools/pancake/reduce_f32_model.py states, per kernel, what is proven
// (host emulation of the exact device operation order with correctly rounded f32 operations) and what is not (the device expf is not the host's).
// Which of these kernels run on the served path is UNVERIFIED (LEADS.md U.12): under --native (the daily preset) layer.cpp takes the native f32 kernels
// instead (native_gdn_*, native_qsa_indexer_*, native_ple_postops, native_bf16), so on a static reading none of the kernels above runs there; the
// dispatch census (lead U3) decides whether an A/B is meaningful.
#pragma once

#include <cmath>
#include <cstdlib>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_RF32_FN __host__ __device__ inline
#else
#define STRATA_RF32_FN inline
#endif

namespace strata::kernels::rf32 {

/// STRATA_REDUCE_F32=1 (value starting with '1'); read once.  Host side only: the kernels take the choice as a template argument.
inline bool requested() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_REDUCE_F32");
        return v != nullptr && v[0] == '1';
    }();
    return on;
}

/// The accumulator type of a kernel instantiation: double (the original) or float.
template <bool F32>
using acc_t = std::conditional_t<F32, float, double>;

/// x / (1 + exp(-x)), f32.  Below -87 the plain form overflows expf(-x) (> 3.4e38 at x < -88.72) and returns -0 where the f64 original returns a tiny
/// negative value, so x * expf(x) is used there (1 + exp(-x) is exp(-x) to f32 precision for x < -17).
/// expf is the device's (ocml, within 1 ulp as documented, not correctly rounded): tools/pancake/reduce_f32_model.py bounds the result for a correctly
/// rounded expf and for one that is 1 ulp off.
STRATA_RF32_FN float silu(float x) {
    return x < -87.0f ? x * expf(x) : x / (1.0f + expf(-x));
}

/// 1 / sqrt(s) in f32 with correctly rounded operations (explicit intrinsics on a device: no contraction, no fast-math): the host emulation is exact.
#if defined(__CUDACC__) || defined(__HIPCC__)
__device__ inline float inv_sqrt(float s) { return __fdiv_rn(1.0f, __fsqrt_rn(s)); }
__device__ inline float fadd(float a, float b) { return __fadd_rn(a, b); }
__device__ inline float fmul(float a, float b) { return __fmul_rn(a, b); }
#endif

}  // namespace strata::kernels::rf32
