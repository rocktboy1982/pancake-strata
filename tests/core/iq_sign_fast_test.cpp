// tests/core/iq_sign_fast_test.cpp - MOD_PLAN F12 (STRATA_IQ_SIGN_FAST): iq_signs_fast (include/strata/kernels/iq_signs.hpp,
// the function iq_kernels.cu's iq_signed<true> calls) against llama.cpp's sign step s = __vcmpne4(t, 0), __vsub4(g ^ s, s)
// in its CUDA semantics (byte-wise) and in the HIP build's SWAR emulation (hip_compat/intrinsics.hpp, transcribed below),
// on the host, exhaustively over the domain the kernels use:
//   * g: every 32-bit word of the five codebooks (iq2xxs 256 x 2, iq2xs 512 x 2, iq2s 1024 x 2, iq3xxs 256, iq3s 512);
//   * t: every word with at most one set bit per byte (9^4 = 6561), a superset of the sign fields the formats produce
//     (unpack_ksigns(v) & 0x08040201 / 0x80402010 for v < 128; the IQ2_S / IQ3_S shifts of a sign byte), which are
//     checked to be in it.
// Also checks the precondition the identity needs (no codebook byte is 0) and, as a negative control, that the test
// sees the difference where a byte of g is 0.  Exit 0 = PASS, 1 = FAIL.
#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include "strata/kernels/iq_signs.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

// CUDA: __vcmpne4 = 0xff per unequal byte; __vsub4 = per-byte wrapped difference
uint32_t cuda_signed(uint32_t g, uint32_t t) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t s = ((t >> (8 * i)) & 0xFF) != 0 ? 0xFF : 0;
        const uint32_t gb = (g >> (8 * i)) & 0xFF;
        r |= (((gb ^ s) - s) & 0xFF) << (8 * i);
    }
    return r;
}

// HIP: strata::hip_compat::vcmpne4 / vsub4 (include/strata/hip_compat/intrinsics.hpp), device-only there
constexpr uint32_t kHigh = 0x80808080u;
uint32_t hip_vcmpne4(uint32_t a, uint32_t b) {
    const uint32_t t = a ^ b;
    const uint32_t nonzero = (((t & ~kHigh) + ~kHigh) | t) & kHigh;
    return (nonzero >> 7) * 0xffu;
}
uint32_t hip_vsub4(uint32_t a, uint32_t b) { return ((a | kHigh) - (b & ~kHigh)) ^ ((a ^ ~b) & kHigh); }
uint32_t hip_signed(uint32_t g, uint32_t t) {
    const uint32_t s = hip_vcmpne4(t, 0);
    return hip_vsub4(g ^ s, s);
}

bool one_bit_per_byte(uint32_t t) {
    for (int i = 0; i < 4; ++i) {
        const uint32_t b = (t >> (8 * i)) & 0xFF;
        if ((b & (b - 1)) != 0) return false;
    }
    return true;
}

uint32_t unpack_ksigns(uint8_t v) {   // iq_kernels.cu
    const uint32_t p = (uint32_t) __builtin_popcount(v) & 1;
    return (v ^ p << 7) * 0x01010101u;
}

}  // namespace

int main() {
    struct Grid { const char* name; std::vector<uint32_t> words; };
    Grid grids[5] = {{"iq2xxs", {}}, {"iq2xs", {}}, {"iq2s", {}}, {"iq3xxs", {}}, {"iq3s", {}}};
    auto add64 = [](Grid& g, const uint64_t* p, int n) {
        for (int i = 0; i < n; ++i) {
            g.words.push_back((uint32_t) p[i]);
            g.words.push_back((uint32_t) (p[i] >> 32));
        }
    };
    add64(grids[0], iq2xxs_grid, 256);
    add64(grids[1], iq2xs_grid, 512);
    add64(grids[2], iq2s_grid, 1024);
    grids[3].words.assign(iq3xxs_grid, iq3xxs_grid + 256);
    grids[4].words.assign(iq3s_grid, iq3s_grid + 512);

    long long fails = 0;
    for (const Grid& g : grids)
        for (uint32_t w : g.words)
            for (int i = 0; i < 4; ++i)
                if (((w >> (8 * i)) & 0xFF) == 0) {
                    if (fails < 8) std::printf("  precondition: %s word %08x has a zero byte\n", g.name, w);
                    ++fails;
                }

    // every t with at most one set bit per byte
    std::vector<uint32_t> ts;
    const uint32_t byte_vals[9] = {0, 1, 2, 4, 8, 16, 32, 64, 128};
    for (int a = 0; a < 9; ++a)
        for (int b = 0; b < 9; ++b)
            for (int c = 0; c < 9; ++c)
                for (int d = 0; d < 9; ++d)
                    ts.push_back(byte_vals[a] | byte_vals[b] << 8 | byte_vals[c] << 16 | byte_vals[d] << 24);

    // the formats' sign fields lie in that set
    long long fields = 0;
    for (int v = 0; v < 128; ++v)
        for (uint32_t m : {0x08040201u, 0x80402010u}) {
            fails += !one_bit_per_byte(unpack_ksigns((uint8_t) v) & m);
            ++fields;
        }
    for (int s = 0; s < 256; ++s) {
        fails += !one_bit_per_byte(((s & 0x03) << 7) | ((s & 0x0C) << 21));
        fails += !one_bit_per_byte(((s & 0x30) << 3) | ((s & 0xC0) << 17));
        fields += 2;
    }

    long long cases = 0, words = 0;
    for (const Grid& g : grids) {
        words += (long long) g.words.size();
        for (uint32_t w : g.words)
            for (uint32_t t : ts) {
                const uint32_t ref = cuda_signed(w, t);
                const uint32_t hip = hip_signed(w, t), fast = strata::kernels::iq_signs_fast(w, t);
                if (hip != ref || fast != ref) {
                    if (fails < 8)
                        std::printf("  differ: %s g %08x t %08x: cuda %08x hip %08x fast %08x\n", g.name, w, t, ref, hip, fast);
                    ++fails;
                }
                ++cases;
            }
    }

    // negative control: with a zero byte in g the identity breaks (the carry), and the comparison above would see it
    const bool control = strata::kernels::iq_signs_fast(0x01010100u, 0x00000080u) != cuda_signed(0x01010100u, 0x00000080u);
    if (!control) {
        std::printf("  negative control: no difference on a zero byte - the check cannot see a carry\n");
        ++fails;
    }

    std::printf("iq_sign_fast_test: %lld codebook words x %zu sign words = %lld cases (%lld format sign fields in the set), "
                "%lld failures -> %s\n", words, ts.size(), cases, fields, fails, fails ? "FAIL" : "PASS");
    return fails ? 1 : 0;
}
