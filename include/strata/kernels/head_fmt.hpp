// include/strata/kernels/head_fmt.hpp - pancake-strata: specialized kernels for the output heads of the models this
// engine runs, one per head format (every Qwen3.8-Flash-Next file has a 2560-wide head):
//
//   IQ4_XS  ISTA-DASLab / Swift IQ2_XS                               (the daily model)
//   Q6_K    ISTA-DASLab / Swift IQ3_S, the Coder IQ1_M, Unsloth UD-IQ4_XS
//   Q4_K    Unsloth UD-IQ1_M
//   (Q5_K   ISTA-DASLab / Swift Q2_0 and IQ3_XXS: basalt's q5k_head.hpp)
//
// For each format: the head GEMV for 1..8 columns, bitwise equal to native_mmvq (the exact layout) - one warp per row
// holds what native_mmvq's four warps did (lane L stands for its threads L, L + 32, L + 64, L + 96, whose sums it keeps
// apart and adds in native_mmvq's cross-warp order before the same xor tree), and a block reads its columns' activations
// once into shared memory; and basalt's 2-bit proxy for the MTP draft head's greedy pick: every row scored from a 2-bit
// copy, the rows near the best rescored exactly in the same arithmetic, the exact best is the pick.  The design is
// basalt's q5k_head (jesdga95/basalt, MIT); each format's arithmetic is native_mmvq.cu's traits, unchanged.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// The shapes the head GEMVs take: n_in a multiple of 256, at most 8192 (the columns' activations in shared memory),
/// 1..8 columns.  False from the GEMVs: an unsupported shape, nothing launched.
bool head_fmt_supported(int n_in, int ncols);
bool iq4xs_head_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);
bool q6k_head_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);
bool q4k_head_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

/// The 2-bit proxy of a draft head of 2560-wide rows: pack rows [row0, row0 + n) (n_cap rows of capacity), then the
/// argmax of rows [0, n_rows) . x through `map` (a negative entry skips a row) and its softmax probability.  `scratch`:
/// the *_scratch_bytes, zeroed once.  The picks feed drafts, which the verify window checks.
size_t iq4xs_proxy_bytes(int64_t n_cap);
size_t iq4xs_proxy_scratch_bytes(int64_t n_cap);
void iq4xs_proxy_pack(const void* rows, int64_t row0, int64_t n, void* proxy, int64_t n_cap, void* stream);
bool iq4xs_proxy_top1(const void* proxy, const void* rows, const void* x_q8_1, int n_rows, int64_t n_cap,
                      const int32_t* map, int* out_id, float* out_prob, void* scratch, void* stream);
size_t q6k_proxy_bytes(int64_t n_cap);
size_t q6k_proxy_scratch_bytes(int64_t n_cap);
void q6k_proxy_pack(const void* rows, int64_t row0, int64_t n, void* proxy, int64_t n_cap, void* stream);
bool q6k_proxy_top1(const void* proxy, const void* rows, const void* x_q8_1, int n_rows, int64_t n_cap,
                    const int32_t* map, int* out_id, float* out_prob, void* scratch, void* stream);

size_t q4k_proxy_bytes(int64_t n_cap);
size_t q4k_proxy_scratch_bytes(int64_t n_cap);
void q4k_proxy_pack(const void* rows, int64_t row0, int64_t n, void* proxy, int64_t n_cap, void* stream);
bool q4k_proxy_top1(const void* proxy, const void* rows, const void* x_q8_1, int n_rows, int64_t n_cap,
                    const int32_t* map, int* out_id, float* out_prob, void* scratch, void* stream);

}  // namespace strata::kernels
