#pragma once
#include <cstdint>

namespace strata::kernels {
// Configure before constructing/capturing sessions. Existing captures keep their
// selected kernels; changing this flag does not rewrite an existing graph.
void native_moe_combine_set_enabled(bool enabled);
bool native_moe_combine_enabled();

// Pinned CUDA weighted-reduction contract for one token, k in [1, 15].
// The backend fuses k=2..15; k=1 is its ordinary multiply/contiguous/add path.
// parts: k contiguous F32 rows of n_embd; weights: k F32 values. The first
// product rounds to F32, following products accumulate with FMA in expert order,
// and optional shared is added once afterward. Shared is not router weighted.
// Requires a nonnull ordered stream and disjoint output. No allocation or sync.
void native_moe_combine(const float* parts, const float* weights, const float* shared,
                        float* output, int64_t n_embd, int64_t k, void* stream);
/// n_tok rows (parts [n,k,N], weights [n,k], shared/output [n,N]) in one launch, each as the single call.
/// 143-amdk: native_moe_combine_multi with moe_hit_add folded in: for the rows listed in dst[0, *count) (entry index = token * k + expert,
/// relative to `parts` and `hits`) the expert's part is parts + hits (the add moe_hit_add made in place), then the combine as before.
/// Only the k = 10 vec4 case; `native_moe_combine_hits_supported` says whether a call may use it.
bool native_moe_combine_hits_supported(const float* parts, const float* shared, const float* output, int64_t n_embd, int64_t k, int n_tok);
void native_moe_combine_multi_hits_added(const float* parts, const float* hits, const int32_t* dst, const int32_t* count,
                                         const float* weights, const float* shared, float* output, int64_t n_embd, int64_t k,
                                         int n_tok, void* stream);
/// STRATA_FUSE_COMBINE (this-cute-world/Strata ed87ad46af): native_moe_combine_multi_hits_added with the CPU share's wait
/// and copy folded in - the parts are read from `mapped_parts` (the pool's mapped rows, device alias) once every block has
/// seen *flag >= value:
///   skip == nullptr: as copy_rows_from_mapped_after(flag, value, parts, mapped_parts, n * k, n_embd, dst, count) then
///                    the combine (a hit row's part +0.0, the others the mapped row);
///   skip != nullptr: as wait_flag_ge_or + copy_or_zero_from_mapped(parts, mapped_parts, .., skip, value) then the
///                    combine (*skip == value: every part +0.0, no wait; else every part the mapped row).
/// shared_gate != nullptr (STRATA_FUSE_SHGATE, 93c97cc230): `shared` unscaled, scaled by sigmoid(shared_gate[t]) as
/// shared_expert_multi's sigmoid_scale_rows did.  Bitwise the launches it replaces; one kernel node for two or three.
void native_moe_combine_multi_hits_mapped(const float* mapped_parts, const float* hits, const int32_t* dst,
                                          const int32_t* count, const float* weights, const float* shared, float* output,
                                          int64_t n_embd, int64_t k, int n_tok, const uint32_t* flag, uint32_t value,
                                          const uint32_t* skip, const float* shared_gate, void* stream);
/// android-imdad caa9b00115 (its STRATA_COMBINE_HITS): native_moe_combine_multi_hits_added under the fork's name and
/// contract - k = 10, 1-8 tokens, 16-byte aligned rows only: false (nothing launched) otherwise.  Bitwise the fork's.
bool native_moe_combine_multi_add_hits(const float* parts, const float* hit, const int32_t* dst, const int32_t* count,
                                       const float* weights, const float* shared, float* output, int64_t n_embd,
                                       int64_t k, int n_tok, void* stream);
/// android-imdad caa9b00115 (its STRATA_WAIT_COMBINE): wait_copy_rows_from_mapped (the CPU's rows after `flag` reaches
/// `value`) and the add-hits combine in one launch - this tree's native_moe_combine_multi_cpu kernel with every block
/// waiting (no gate, no presum); the same output bit for bit.  `cpu_rows` is the mapped rows.
bool native_moe_wait_combine_ok(const float* cpu_rows, const float* hit, const float* shared, const float* output,
                                int64_t n_embd, int64_t k, int n_tok);
void native_moe_wait_combine_multi_add_hits(const uint32_t* flag, uint32_t value, const float* cpu_rows, const float* hit,
                                            const int32_t* dst, const int32_t* count, const float* weights,
                                            const float* shared, float* output, int64_t n_embd, int64_t k, int n_tok,
                                            void* stream);
void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream);
/// STRATA_VERIFY_RESIDENT: the same, reading every part as `0.0f + hit[i]` - bitwise what a zeroed part row plus
/// moe_hit_add (parts += hit) gave, without the zero fill and the add launch.
void native_moe_combine_multi_hits(const float* hits, const float* weights, const float* shared, float* output,
                                   int64_t n_embd, int64_t k, int n_tok, void* stream);
/// S26 STRATA_LFUSE: the same with `shared` unscaled and shared_gate[t] the shared expert's raw gate logit: the
/// row is scaled by sigmoid(shared_gate[t]) here, bitwise what shared_expert_multi's sigmoid + scale gave.
/// STRATA_COMBINE_CPU (k = 10): the verify window's combine with the CPU rows read straight from the mapped `cpu`
/// rows and the GPU's own rows (hit_rows[0, *count)) from `hit` as 0.0f + hit - bitwise copy_rows_from_mapped +
/// moe_hit_add + native_moe_combine_multi; a nonnull shared_gate also gates `shared` as the _gated combine does.
void native_moe_combine_multi_cpu(const float* cpu, const float* hit, const int32_t* hit_rows, const int32_t* count,
                                  const float* weights, const float* shared, const float* shared_gate, float* output,
                                  int64_t n_embd, int n_tok, void* stream, const uint32_t* wait_flag = nullptr,
                                  uint32_t wait_value = 0, unsigned long long* t_waited = nullptr, bool presum = false,
                                  const float* presum_rows = nullptr);
/// (item 52) `presum_rows`: the pre-summed rows are presum_rows + t * n_embd (VRAM) instead of the mapped buffer.
/// (MOD_PLAN item 51) `presum`: the CPU entries of token t are ONE row, cpu + (t * 10) * n_embd, already weighted and
/// summed by the host; it is added once (when the token has a CPU entry) after the hits' weighted sum.
/// (MOD_PLAN item 49) `t_waited`, with `wait_flag`: block (0,0) writes the GPU clock (ns) there once the CPU-share wait
/// has ended - the verify profile's "waitCPU" stamp, which the fused wait otherwise folds into "combine".
void native_moe_combine_multi_hits_gated(const float* hits, const float* weights, const float* shared,
                                         const float* shared_gate, float* output, int64_t n_embd, int64_t k, int n_tok,
                                         void* stream);
}
