// include/strata/kernels/iq4xs_head.hpp - pancake-strata: the IQ4_XS x q8_1 GEMV of a wide head (Flash-Next IQ2_XS's
// output head and its MTP draft head subset) for 1..8 columns, bitwise equal to native_mmvq (type 23, the exact layout).
//
// The design is basalt's q5k_head (jesdga95/basalt, MIT): one warp per row holds what native_mmvq's four warps did -
// lane L stands for its threads L, L + 32, L + 64, L + 96, keeps their sums apart and adds them in native_mmvq's
// cross-warp order before the same xor tree - and a block reads its columns' activations once into shared memory for
// all its rows.  At 2560 inputs native_mmvq keeps only 80 of a row's 128 threads busy.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::kernels {

/// The shapes the kernel takes: n_in a multiple of 256 with its columns' activations in shared memory (n_in <= 8192),
/// 1..8 columns.
bool iq4xs_head_supported(int n_in, int ncols);
/// y[j * n_out + r] = row r of `weights` (IQ4_XS, n_in / 256 blocks per row) . column j of `x_q8_1`; the bits of
/// native_mmvq(23, ...).  False, with nothing launched, for an unsupported shape.
bool iq4xs_head_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream);

/// basalt's 2-bit proxy (q5k_proxy_*) for IQ4_XS rows of 2560 inputs: pack rows [row0, row0 + n) of the IQ4_XS rows
/// into the copy (n_cap rows of capacity), then the head's argmax and its softmax probability over rows [0, n_rows):
/// every row scored from the copy, the rows near the best rescored exactly (native_mmvq's arithmetic), the exact best
/// is the pick (through `map`; rows with a negative entry are skipped).  `scratch`: iq4xs_proxy_scratch_bytes, zeroed once.
size_t iq4xs_proxy_bytes(int64_t n_cap);
size_t iq4xs_proxy_scratch_bytes(int64_t n_cap);
void iq4xs_proxy_pack(const void* rows, int64_t row0, int64_t n, void* proxy, int64_t n_cap, void* stream);
bool iq4xs_proxy_top1(const void* proxy, const void* rows, const void* x_q8_1, int n_rows, int64_t n_cap,
                      const int32_t* map, int* out_id, float* out_prob, void* scratch, void* stream);

}  // namespace strata::kernels
