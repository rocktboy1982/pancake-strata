#pragma once
// LEADS.md Part U U7: independent i32 accumulator chains in the verify-window GEMV (STRATA_VERIFY_GEMV_CHAINS=2|4,
// default unset = the code path of rdna_mmvq.cu unchanged).  This header holds the integer arithmetic the chain
// variants share between the device kernels and the host test (tests/kernels/rdna_chain_split_test.cpp).
//
// The lane dot of one superblock is two v_dot4_i32_iu8 steps over 8 int8 pairs.  Today they are ONE dependent chain
// (acc -> sudot4 -> sudot4).  The variants issue two independent sudot4 from zero and join them with an integer add.
// Integer addition modulo 2^32 is associative and commutative and sudot4 here has no clamp, so the joined result is
// bit-identical to the chain for every input; the f32 stage that follows is untouched (same values, same order).
#include <cstdint>

namespace strata::kernels::rdna_chain {

// D::dot4(w, q, acc) = four signed int8 products added to acc modulo 2^32 (the device: __builtin_amdgcn_sudot4 with
// both sign flags set, no clamp; the host test: an emulation).
template <int CH, class D>
#if defined(__HIPCC__) || defined(__CUDACC__)
__host__ __device__ __forceinline__
#else
inline
#endif
int dot8(int w0, int w1, int q0, int q1) {
    if constexpr (CH <= 1) {
        return D::dot4(w1, q1, D::dot4(w0, q0, 0));   // the existing chain, kept as the reference form
    } else {
        const int a = D::dot4(w0, q0, 0);
        const int b = D::dot4(w1, q1, 0);
        return (int) ((uint32_t) a + (uint32_t) b);
    }
}

// the chain count a switch value selects: 0 = off (current code), 2 or 4 = the variants; anything else = off
inline int parse_chains(const char* e) {
    if (e == nullptr || e[0] == '\0' || e[1] != '\0') return 0;
    return e[0] == '2' ? 2 : e[0] == '4' ? 4 : 0;
}

}  // namespace strata::kernels::rdna_chain
