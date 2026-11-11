// include/strata/kernels/iq_signs.hpp - MOD_PLAN F12 (STRATA_IQ_SIGN_FAST): the IQ2/IQ3 grid-sign step without byte
// compares, for the device kernels (iq_kernels.cu, iq_signed<true>) and the host test (tests/core/iq_sign_fast_test.cpp)
// from one definition.
#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define STRATA_IQ_SIGNS_HD __host__ __device__
#else
#define STRATA_IQ_SIGNS_HD
#endif

namespace strata::kernels {

/// g (four codebook bytes) with the bytes negated (mod 256) where t has a bit; t holds at most one set bit per byte.
/// llama.cpp's form is s = __vcmpne4(t, 0), __vsub4(g ^ s, s); this one is 7 plain 32-bit ops and no multiply:
///   m = (t + 0x7F7F7F7F) & 0x80808080: a byte of t is 0 or a single bit (<= 0x80), so byte + 0x7F <= 0xFF carries into
///       no other byte, and its top bit is set iff the byte was nonzero: m is 0x80 in exactly the bytes to negate;
///   c = m >> 7 is 0x01 there; m - c is 0x7F there and 0 elsewhere (no borrow: each byte of m is >= that of c), so
///       m | (m - c) is 0xFF there and 0 elsewhere - the mask __vcmpne4 returns;
///   (g ^ s) + c: a negated byte becomes (g ^ 0xFF) + 1 = 256 - g, which is <= 0xFF - no carry into the next byte -
///       because no IQ2/IQ3 codebook byte is 0; the other bytes add 0.  Every byte is __vsub4's wrapped byte.
/// The two preconditions (one bit per byte of t, no zero byte in g) are what the host test checks exhaustively.
STRATA_IQ_SIGNS_HD inline uint32_t iq_signs_fast(uint32_t g, uint32_t t) {
    const uint32_t m = (t + 0x7F7F7F7Fu) & 0x80808080u;
    const uint32_t c = m >> 7;
    return (g ^ (m | (m - c))) + c;
}

}  // namespace strata::kernels
