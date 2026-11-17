// include/strata/kernels/int4_moe.hpp - LEADS.md Part L L3 / L4, the host side of STRATA_PF_IU4 (Q2_0 experts on the iu4 WMMA).
// Host only, header only.  What it fixes, for the kernel in tools/pancake/probes/int4_moe.hip.cpp (and for an engine kernel later):
//   1. q2_0_words_to_int4: 16 Q2_0 codes (32 bits of a block) -> the 16 signed nibbles of one WMMA operand slice, by SWAR (the same
//      lines run on the device): spread the 2-bit fields to nibbles, subtract 1 per nibble.  Q2_0 needs no repack pass and no int4
//      copy of the weights: the packed 2-bit bytes ARE the operand source.
//   2. wmma_iu4_emulate: the semantics of v_wmma_i32_16x16x16_iu4 as the probe feeds it (A: lane l holds row l % 16, 16 k as two
//      words; B: lane l holds column l % 16; D: lane l holds D[2 i + l / 16][l % 16], i = 0..7), so the tile loop of the kernel is
//      checked on the host to the last integer.  The emulator is OUR reading of the operand layout: the box run of the probe decides
//      whether the hardware agrees (the same open point as int4_wmma.hip.cpp).
//   3. grouped_tile_gemm: the expert-gather GEMM, token rows grouped by expert (bounds[e]..bounds[e + 1], as the engine sorts the
//      routed entries), weights read straight from the Q2_0 blocks of expert e, activations the A8 pair of int4rot (h signed, l
//      unsigned), run through the emulator; equals int4rot::gemm_ref on the same codes.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

#include "strata/kernels/int4_iq2.hpp"
#include "strata/kernels/int4_rot.hpp"

namespace strata::kernels::int4moe {

/// 8 Q2_0 codes (16 bits, code e at bits 2e) -> 8 signed nibbles in one word (code - 1 in two's complement, element e at nibble e)
inline uint32_t q2_0_half_to_nibbles(uint32_t codes16) {
    uint32_t x = codes16 & 0xffffu;
    x = (x | (x << 8)) & 0x00ff00ffu;
    x = (x | (x << 4)) & 0x0f0f0f0fu;
    x = (x | (x << 2)) & 0x33333333u;
    return ((x | 0x88888888u) - 0x11111111u) ^ 0x88888888u;
}
/// 16 Q2_0 codes (one 32-bit word of a block's code bytes) -> the operand slice's two words
inline void q2_0_words_to_int4(uint32_t codes32, uint32_t out[2]) {
    out[0] = q2_0_half_to_nibbles(codes32);
    out[1] = q2_0_half_to_nibbles(codes32 >> 16);
}

/// v_wmma_i32_16x16x16_iu4 on 32 lanes: a[l] / b[l] are the lane's two words; the signedness is per operand.  d[l][i] = D[2 i + l / 16][l % 16].
inline void wmma_iu4_emulate(const uint32_t a[32][2], bool a_signed, const uint32_t b[32][2], bool b_signed, int32_t d[32][8]) {
    auto nib = [](const uint32_t w[2], int e, bool sgn) {
        const int v = (int) ((w[e >> 3] >> (4 * (e & 7))) & 15u);
        return sgn && v >= 8 ? v - 16 : v;
    };
    for (int l = 0; l < 32; ++l)
        for (int i = 0; i < 8; ++i) {
            const int row = 2 * i + l / 16, col = l % 16;
            int32_t acc = d[l][i];
            for (int k = 0; k < 16; ++k) acc += nib(a[row], k, a_signed) * nib(b[col], k, b_signed);   // A row `row` lives in lane `row`
            d[l][i] = acc;
        }
}

/// Q2_0 expert weights, rows x K/64 blocks of 18 bytes per expert, concatenated
struct Q2Experts {
    int n_expert = 0, rows = 0, K = 0;
    std::vector<uint8_t> bytes;
    const uint8_t* row_block(int e, int m, int b) const {
        return bytes.data() + (((size_t) e * rows + m) * (K / 64) + b) * 18;
    }
};

/// y[r][m] for the sorted token rows r in [bounds[e], bounds[e + 1]) against expert e's rows, A8 activations (a.N = total rows)
inline std::vector<float> grouped_tile_gemm(const Q2Experts& w, const std::vector<int>& bounds, const int4rot::ActQ& a) {
    const int M = w.rows, K = w.K, N = a.N;
    std::vector<float> y((size_t) N * M, 0.f);
    for (int e = 0; e < w.n_expert; ++e)
        for (int n0 = bounds[e]; n0 < bounds[e + 1]; n0 += 16)
            for (int m0 = 0; m0 < M; m0 += 16) {
                float acc[16][16] = {};   // [m][n]
                for (int g = 0; g < K / 32; ++g) {
                    uint32_t aw[32][2], hw[32][2], lw[32][2];
                    int32_t dh[32][8] = {}, dl[32][8] = {};
                    for (int sl = 0; sl < 2; ++sl) {   // the two 16-slices of the 32-group
                        const int k0 = g * 32 + sl * 16;
                        for (int lane = 0; lane < 32; ++lane) {
                            const int mm = std::min(m0 + lane % 16, M - 1), nn = n0 + lane % 16;
                            const uint8_t* blk = w.row_block(e, mm, k0 / 64);
                            uint32_t c32;
                            std::memcpy(&c32, blk + 2 + ((k0 % 64) / 16) * 4, 4);
                            q2_0_words_to_int4(c32, aw[lane]);
                            int8_t h[16] = {}, lo[16] = {};
                            if (nn < bounds[e + 1])
                                for (int k = 0; k < 16; ++k) {
                                    h[k] = a.hi[(size_t) nn * K + k0 + k];
                                    lo[k] = a.lo[(size_t) nn * K + k0 + k];
                                }
                            int4iq2::pack_nibbles16(h, hw[lane]);
                            int4iq2::pack_nibbles16(lo, lw[lane]);
                        }
                        wmma_iu4_emulate(aw, true, hw, true, dh);
                        wmma_iu4_emulate(aw, true, lw, false, dl);
                    }
                    for (int lane = 0; lane < 32; ++lane)
                        for (int i = 0; i < 8; ++i) {
                            const int mi = 2 * i + lane / 16, ni = lane % 16;
                            const int mm = std::min(m0 + mi, M - 1), nn = n0 + ni;
                            if (m0 + mi >= M || nn >= bounds[e + 1]) continue;
                            uint16_t hd;
                            std::memcpy(&hd, w.row_block(e, mm, (g * 32) / 64), 2);
                            const float sw = int4iq2::half_to_float(hd);
                            acc[mi][ni] += sw * a.scale[(size_t) nn * (K / a.B) + (g * 32) / a.B] * (float) (16 * dh[lane][i] + dl[lane][i]);
                        }
                }
                for (int mi = 0; mi < 16; ++mi)
                    for (int ni = 0; ni < 16; ++ni)
                        if (m0 + mi < M && n0 + ni < bounds[e + 1]) y[(size_t) (n0 + ni) * M + m0 + mi] = acc[mi][ni];
            }
    return y;
}

}  // namespace strata::kernels::int4moe
