// include/strata/kernels/int4_iq2.hpp - LEADS.md Part L L4: the daily weight formats as int4 operands of the iu4 WMMA (lead L3 / L4).
// Host only, header only, no GPU, no engine state; the tables of IQ2_XS (grid, signs) come in as pointers so the test, not this file,
// includes ggml-common.h.
//
//   Q2_0   (GGUF type 42, 64 values in 18 bytes: fp16 d, 2-bit codes, value = (code - 1) d, so the symbols are {-1, 0, 1, 2}):
//          EXACT in int4.  q = code - 1 is in [-1, 2], the scale of a 32-group is d of its 64-block; the half-blocks repeat d.
//   IQ2_XS (256 values in 74 bytes: fp16 d, 32 x u16 = 9-bit index into a 512-entry grid of 8 magnitudes + 7-bit sign index, 8 bytes
//          of 4-bit scales, value = d (0.5 + s4) 0.25 grid_j sign_j): the magnitudes are {8, 25, 43}, so the value is NOT an int4
//          grid.  Mapped to the odd levels k in {+-1, +-3, +-5} of one step t (int4 holds -8..7): the error is
//          |g - t k| per weight before the group scale.  t = 8.2 is the lead's nominal step; best_step() finds the least-squares step
//          for the codebook's own magnitude frequencies.  Lossy; the scale of the int4 operand is the 16-value scale times t.
// The WMMA operand packing (16 nibbles = two 32-bit words, element e at nibble e of the slice) is pack_nibbles16.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace strata::kernels::int4iq2 {

inline float half_to_float(uint16_t h) {   // IEEE binary16, host
    const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = std::ldexp((float) m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((float) (m + 1024), (int) e - 25);
    return s ? -v : v;
}

/// int4 codes in [-8, 7], one per value, row-major [rows][K], one float scale per `group` values: value ~ q * scale.
struct Int4Rows {
    int rows = 0, K = 0, group = 32;
    std::vector<int8_t> q;
    std::vector<float> scale;   // [rows][K / group]
    float value(int r, int k) const { return (float) q[(size_t) r * K + k] * scale[(size_t) r * (K / group) + k / group]; }
};

/// Q2_0 rows (rows x K/64 blocks of 18 bytes) -> int4, exact.  K a multiple of 64.
inline Int4Rows q2_0_to_int4(const uint8_t* blocks, int rows, int K) {
    Int4Rows r;
    r.rows = rows; r.K = K; r.group = 32;
    r.q.assign((size_t) rows * K, 0);
    r.scale.assign((size_t) rows * (K / 32), 0.f);
    const int nb = K / 64;
    for (int m = 0; m < rows; ++m)
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = blocks + ((size_t) m * nb + b) * 18;
            uint16_t h;
            std::memcpy(&h, blk, 2);
            const float d = half_to_float(h);
            r.scale[(size_t) m * (K / 32) + 2 * b] = d;
            r.scale[(size_t) m * (K / 32) + 2 * b + 1] = d;
            for (int j = 0; j < 64; ++j) {
                const int code = (blk[2 + j / 4] >> ((j % 4) * 2)) & 3;
                r.q[(size_t) m * K + b * 64 + j] = (int8_t) (code - 1);   // {-1, 0, 1, 2}
            }
        }
    return r;
}

/// The Q2_0 value of the engine's own dequantiser, for the test: (code - 1) * d (strata::dequantize_q2_0)
inline float q2_0_value(const uint8_t* blk, int j) {
    uint16_t h;
    std::memcpy(&h, blk, 2);
    return (float) (((blk[2 + j / 4] >> ((j % 4) * 2)) & 3) - 1) * half_to_float(h);
}

// ---- IQ2_XS --------------------------------------------------------------------------------------------------------------------
constexpr int kIq2xsBlock = 74;   // bytes of 256 values: 2 (d) + 64 (qs) + 8 (scales)
struct Iq2xsTables {
    const uint64_t* grid;   // 512 entries, 8 magnitude bytes each (little endian)
    const uint8_t* ksigns;  // 128 entries: bit j set = value j is negative (bit 7 is the parity of the others)
};

/// the 8 magnitudes of grid entry i (bytes of the table word) and the sign mask of sign index s
inline void iq2xs_entry(const Iq2xsTables& t, int gi, int si, float out[8]) {
    uint64_t g = t.grid[gi & 511];
    for (int j = 0; j < 8; ++j) {
        const float mag = (float) ((g >> (8 * j)) & 0xff);
        out[j] = (t.ksigns[si & 127] >> j & 1) ? -mag : mag;
    }
}

/// the exact decoded value of value j of the block (llama.cpp's dequantize_row_iq2_xs)
inline void iq2xs_decode_block(const Iq2xsTables& t, const uint8_t* blk, float out[256]) {
    uint16_t hd;
    std::memcpy(&hd, blk, 2);
    const float d = half_to_float(hd);
    const uint8_t* sc = blk + 2 + 64;
    for (int ib = 0; ib < 8; ++ib) {
        const float db[2] = {d * (0.5f + (float) (sc[ib] & 0xf)) * 0.25f, d * (0.5f + (float) (sc[ib] >> 4)) * 0.25f};
        for (int l = 0; l < 4; ++l) {
            uint16_t q;
            std::memcpy(&q, blk + 2 + 2 * (4 * ib + l), 2);
            float e[8];
            iq2xs_entry(t, q & 511, q >> 9, e);
            for (int j = 0; j < 8; ++j) out[32 * ib + 8 * l + j] = db[l / 2] * e[j];
        }
    }
}

/// the odd level k in {1, 3, 5} nearest to magnitude g / t
inline int iq2xs_level(float mag, float t) {
    int best = 1;
    float be = 1e30f;
    for (int k = 1; k <= 5; k += 2) {
        const float e = std::fabs(mag - t * (float) k);
        if (e < be) { be = e; best = k; }
    }
    return best;
}

/// IQ2_XS rows (rows x K/256 blocks of 74 bytes) -> int4 odd levels, one scale per 16 values (d (0.5 + s4) 0.25 t).  Lossy.
inline Int4Rows iq2xs_to_int4(const Iq2xsTables& t, const uint8_t* blocks, int rows, int K, float step) {
    Int4Rows r;
    r.rows = rows; r.K = K; r.group = 16;
    r.q.assign((size_t) rows * K, 0);
    r.scale.assign((size_t) rows * (K / 16), 0.f);
    const int nb = K / 256;
    for (int m = 0; m < rows; ++m)
        for (int b = 0; b < nb; ++b) {
            const uint8_t* blk = blocks + ((size_t) m * nb + b) * kIq2xsBlock;
            uint16_t hd;
            std::memcpy(&hd, blk, 2);
            const float d = half_to_float(hd);
            const uint8_t* sc = blk + 2 + 64;
            for (int ib = 0; ib < 8; ++ib)
                for (int l = 0; l < 4; ++l) {
                    const float db = d * (0.5f + (float) ((l / 2) ? (sc[ib] >> 4) : (sc[ib] & 0xf))) * 0.25f;
                    r.scale[(size_t) m * (K / 16) + b * 16 + 2 * ib + l / 2] = db * step;
                    uint16_t q;
                    std::memcpy(&q, blk + 2 + 2 * (4 * ib + l), 2);
                    float e[8];
                    iq2xs_entry(t, q & 511, q >> 9, e);
                    for (int j = 0; j < 8; ++j) {
                        const int k = iq2xs_level(std::fabs(e[j]), step);
                        r.q[(size_t) m * K + b * 256 + 32 * ib + 8 * l + j] = (int8_t) (e[j] < 0 ? -k : k);
                    }
                }
        }
    return r;
}

/// How the codebook sits on the odd int4 levels: frequency of each magnitude over all 512 grid entries x 8 positions, the
/// root-mean-square relative error of the level mapping (value-weighted), and the worst per-magnitude relative error.
struct LevelStats {
    long count[3] = {0, 0, 0};   // magnitudes 8, 25, 43 (the grid's three values; anything else is counted in `other`)
    long other = 0;
    double rms_rel = 0, worst_rel = 0;
    float step = 0;
};
inline LevelStats codebook_level_stats(const Iq2xsTables& t, float step) {
    LevelStats s;
    s.step = step;
    double num = 0, den = 0;
    for (int gi = 0; gi < 512; ++gi)
        for (int j = 0; j < 8; ++j) {
            const int mag = (int) ((t.grid[gi] >> (8 * j)) & 0xff);
            if (mag == 8) ++s.count[0]; else if (mag == 25) ++s.count[1]; else if (mag == 43) ++s.count[2]; else ++s.other;
            const int k = iq2xs_level((float) mag, step);
            const double err = (double) mag - (double) step * k;
            num += err * err;
            den += (double) mag * mag;
            if (mag > 0) s.worst_rel = std::max(s.worst_rel, std::fabs(err) / mag);
        }
    s.rms_rel = den > 0 ? std::sqrt(num / den) : 0.0;
    return s;
}

/// the step that minimises the squared error of the three magnitudes weighted by the codebook's own frequencies (least squares of
/// g against t k with k = 1, 3, 5; k of a magnitude is its nearest level, iterated to a fixed point)
inline float best_step(const Iq2xsTables& t) {
    float step = 8.2f;
    for (int it = 0; it < 50; ++it) {
        double sgk = 0, skk = 0;
        for (int gi = 0; gi < 512; ++gi)
            for (int j = 0; j < 8; ++j) {
                const int mag = (int) ((t.grid[gi] >> (8 * j)) & 0xff);
                const int k = iq2xs_level((float) mag, step);
                sgk += (double) mag * k;
                skk += (double) k * k;
            }
        const float next = (float) (sgk / skk);
        if (std::fabs(next - step) < 1e-6f) { step = next; break; }
        step = next;
    }
    return step;
}

/// 16 consecutive codes of a row -> two 32-bit words, code e at nibble e (the operand layout of v_wmma_i32_16x16x16_iu4 as the probe
/// int4_wmma.hip.cpp feeds it); a negative code keeps its low 4 bits (two's complement)
inline void pack_nibbles16(const int8_t* q, uint32_t out[2]) {
    out[0] = out[1] = 0;
    for (int e = 0; e < 16; ++e) out[e >> 3] |= ((uint32_t) q[e] & 15u) << (4 * (e & 7));
}
inline int8_t unpack_nibble_signed(const uint32_t w[2], int e) {
    const int v = (int) ((w[e >> 3] >> (4 * (e & 7))) & 15u);
    return (int8_t) (v >= 8 ? v - 16 : v);
}

}  // namespace strata::kernels::int4iq2
