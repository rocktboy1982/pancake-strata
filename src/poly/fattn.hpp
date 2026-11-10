// src/poly/fattn.hpp - a prompt batch's attention with llama.cpp's flash-attention MMA kernel (fattn_mma.cuh, MIT)
// over a q8_0 or fp16 cache: heads of 256 values, 8 or 16 heads a group (Qwen3.6 / Ornith 122B).
// The keys go in blocks: under q8_0 each block's K and V are dequantized to fp16 in the scratch, the kernel attends
// to them with the causal mask computed from the positions, and the blocks' results are merged by their softmax max
// and sum.  An fp16 cache is read where it lies, but for the last block (the batch's own rows), which is copied.
// fdec_q8() and fdec_f16() are the decode side (a few tokens), a kernel of the fork's own.
//
// MOD_PLAN F17 (= item 8.8).  Ported from RandoTeam/4gbVramStrata (MIT, Copyright (c) 2026 Niko1221 and the Strata
// contributors, VecSzn and the PolyStrata contributors): commits 79b1e756 (STRATA_FATTN), 79acffe5 (decided per card
// and per scratch), 67f60c12 (STRATA_FDEC), 7fb9d79d (groups of 16), ac6402df (the copy in step with the llama.cpp
// pin, which pancake shares: 3cf03257f), bee07744 (the 4,096-row start of the decode kernel).  pancake's changes:
//  - Opt-in (Qwen35::attn): STRATA_FATTN=1 and STRATA_FDEC=1; unset, the old kernels run and nothing changes.  The
//    fork made both default on 10-28; pancake waits for its KL gate (the kernels sum in halves: rounding-level).
//  - pancake's q8_0 cache keeps a row's codes and its scales together (store_q8(): width int8, then width / 32
//    fp16 scales), the fork's in two arrays: Kvq8 carries the row strides of both, q8_cache() makes one.
//  - A batch whose results do not fit the scratch is read in slices (fattn_q8(): --elastic-prefill's 4,096-row
//    batches with Qwen3.6's 16 heads fill all of Qwen35's score_ with results alone; the fork refused them).
// CUDA only (mma.sync, Ampere or later): on HIP and Turing every function says no and the old path runs.
//
// Every call enqueues on the given CUDA stream.  Pointers are device pointers.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::poly {

/// A q8_0 attention cache: row r's int8 codes from codes + r * code_stride bytes, its fp16 scales (one a 32 values,
/// in the order of the values) from scales + r * scale_stride halves.
struct Kvq8 {
    const int8_t* codes = nullptr;
    const uint16_t* scales = nullptr;
    size_t code_stride = 0;    ///< bytes a row
    size_t scale_stride = 0;   ///< halves a row
};

/// The rows store_q8() writes (strata/poly/kernels.hpp), width values a row (kv_heads * dim, a multiple of 32): a
/// row is its width codes and then their width / 32 scales, width + width / 16 bytes.
inline Kvq8 q8_cache(const void* cache, int width) {
    if (!cache || width <= 0 || width % 32) return Kvq8{};
    const auto* p = static_cast<const uint8_t*>(cache);
    const size_t row = (size_t) width + (size_t) width / 16;
    return Kvq8{reinterpret_cast<const int8_t*>(p), reinterpret_cast<const uint16_t*>(p + width), row, row / 2};
}

/// This build has the kernel for heads of `dim` values, n_head / kv_heads heads a group, on the current card, and a
/// scratch of scratch_floats floats holds a block for batches of up to `rows` tokens, in slices where the whole
/// batch does not fit: fattn_q8() and fattn_f16() then take every such batch.
bool fattn_fits(int dim, int n_head, int kv_heads, int rows, size_t scratch_floats);

/// Causal attention of the n tokens from position pos to the cache rows 0 to pos + n - 1, as attend_batch()
/// (no window, no ring): q and out n_head * dim floats a token.  scratch (scratch_floats floats) holds a block's
/// results and its K and V; a batch whose block does not fit goes in slices of half as many tokens (a whole number
/// of the kernel's tiles), each attending as its own batch; false, with nothing enqueued, when not even a tile's
/// tokens fit, and then the caller's old path writes all of `out`.
bool fattn_q8(const float* q, Kvq8 kq8, Kvq8 vq8, int64_t pos, int n, float* out, int dim, int n_head, int kv_heads,
              float scale, float* scratch, size_t scratch_floats, void* stream);

/// fattn_q8() over an fp16 cache (keys and vals, kv_heads * dim halves a row).
bool fattn_f16(const float* q, const uint16_t* keys, const uint16_t* vals, int64_t pos, int n, float* out, int dim,
               int n_head, int kv_heads, float scale, float* scratch, size_t scratch_floats, void* stream);

/// This build has the decode kernel for this shape on the current card (Ampere or later), and a scratch of
/// scratch_floats floats holds its splits for `rows` cache rows: fdec_q8() and fdec_f16() then take every call up to
/// that many.
bool fdec_fits(int dim, int n_head, int kv_heads, int rows, size_t scratch_floats);

/// attend() for n tokens (fewer than kStageRows) with no window, ring or `whole`, over a q8_0 cache: token t
/// attends the rows 0 to pos + t, with the position from `at` on the card when it is given, and the launch sized
/// for `room` rows then (a recorded launch).  The keys go in splits of 1,024 rows on the tensor cores, read once
/// for up to 4 tokens (group 8) or 2 (group 16), and the splits are merged; a token gets the same bits alone or
/// with others. scratch holds the splits' results; false, with nothing enqueued, where fdec_fits() is false or the
/// caches are not laid out as the kernel reads them (codes 8-byte and scales 16-byte aligned, the same strides for
/// keys and values).
bool fdec_q8(const float* q, Kvq8 kq8, Kvq8 vq8, int64_t pos, int n, const long long* at, int room, float* out,
             int dim, int n_head, int kv_heads, float scale, float* scratch, size_t scratch_floats, void* stream);

/// fdec_q8() over an fp16 cache (keys and vals, kv_heads * dim halves a row, 16-byte aligned), read where it lies.
bool fdec_f16(const float* q, const uint16_t* keys, const uint16_t* vals, int64_t pos, int n, const long long* at,
              int room, float* out, int dim, int n_head, int kv_heads, float scale, float* scratch,
              size_t scratch_floats, void* stream);

}  // namespace strata::poly
