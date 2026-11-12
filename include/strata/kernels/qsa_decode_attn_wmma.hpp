// include/strata/kernels/qsa_decode_attn_wmma.hpp - pancake: the split-K decode attention's chunk pass on RDNA3 matrix
// cores (v_wmma_f32_16x16x16_f16, wave32), opt-in with STRATA_ATTN_WMMA=1. HIP gfx11 only; everywhere else every
// entry point says "not available" and qsa_decode_attn keeps its FP32 kernels.
//
// Design from BitDecoding (Du, ..., Luo Mai; University of Edinburgh; HPCA 2026, arXiv 2503.18773,
// github.com/OpenBitSys/BitDecoding, MIT, read at f2caa89): low-bit KV codes go into the tensor-core B fragment
// as exact FP16 integers, the per-group scales stay in FP32 outside the matrix product, and the GQA query heads that
// share one KV head fill the M dimension of the tile (their "query transformation"; here 12 heads of 16 rows). No
// BitDecoding code is copied: its kernels are CuTe/CUTLASS sm80 (mma.sync m16n8k16, ldmatrix, cp.async, lop3 PTX
// dequant) and none of that exists on gfx11. The fragment layout and the hi+lo FP16 split follow this repository's
// own gfx11 prompt kernel (qsa_prompt_attn.cu, prompt_attn_wmma_body).
//
// What it computes: exactly what qsa_decode_attn's attn_chunk_kernel computes for one 64-cell chunk and one KV head -
// the chunk's scores max m, exp-sum l and unnormalised p.V - in the same scratch layout, so the unchanged merge
// kernels finish it. q and p are carried as FP16 hi + lo pairs (FP32-level accuracy) and the four 64-dim groups' q.k
// are added in a fixed order: deterministic, close to the FP32 kernel (~1e-6 relative) but NOT bitwise equal to it,
// so greedy output can part from the default path after many tokens. KV formats: int8 (mode 1), q4_0 (mode 2) and
// K8V4 (mode 3); fp16 pools are not taken (false: the caller runs the FP32 kernel).
#pragma once

#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"

#include <cstdint>

namespace strata::kernels {

/// The decode attention's chunk size in cells; qsa_decode_attn.cu static_asserts its own CHUNK against it.
inline constexpr int kQsaDecodeAttnChunk = 64;

/// STRATA_ATTN_WMMA=1, the build has the gfx11 kernel and the current device is a gfx11 matrix-core part, and no
/// launch has failed. Read once per device.
bool qsa_decode_attn_wmma_on();

/// The chunk pass of qsa_decode_attn_batch on matrix cores: `n_q` queries' partials into `scratch` (the layout and
/// stride of qsa_decode_attn_scratch_floats). False and nothing launched when the pools are fp16, the geometry is not
/// the compiled one, or the kernel is not available (not built for gfx11, CUDA). Ignores the environment, so a test
/// can call it directly; production goes through qsa_decode_attn_* which check qsa_decode_attn_wmma_on() first.
bool qsa_decode_attn_wmma_chunks(const float* q, const QsaAttnPools& pools, const int32_t* ids, const int32_t* steps,
                                 int64_t cap, const QsaShapes& s, float* scratch, int64_t n_q, void* stream);

}  // namespace strata::kernels
