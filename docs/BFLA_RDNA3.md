# BFLA on the RX 7900 XT (gfx1100): design note

Branch `feat/bfla-hip` (pancake-strata). Status: design (this note), then an opt-in HIP path. Nothing here changes a
default.

Source: BFLA, "Block-Filtered Long-Context Attention Mechanism" (Chong Wu et al., City University of Hong Kong lead;
arXiv 2605.12193 v2, May 2026). Code: github.com/Alicewithrabbit/BFLA, **Apache-2.0** (read at commit 46907c2; nothing
copied: it is a vLLM 0.19.1 Triton patch, and this port is new HIP code written from the paper's equations).

## 1. What BFLA is

A training-free sparse **prefill** attention. Decoding is untouched. Two separate parts:

**A. Block selection (the "indexer").** Per attention layer and request, from the post-RoPE Q and K the layer already
has (no learned weights):

1. Each head vector is RMS-normalised, N(x) = x / sqrt(|x|^2 / d + eps), and heads are averaged into one summary per
   token (paper Eq. 3).
2. Adjacent channels are reduced to d_I = 128, normalised again, and rotated by an orthonormal Walsh-Hadamard
   transform (Eq. 4-5).
3. Keys are grouped into blocks of b = 64 tokens; a block's key state is the mean feature, stored in FP8 with one scale
   per block (Eq. 6).
4. Queries are grouped into tiles of M = 128 token-head rows (one KV-head group, ordered by token then head). A tile's
   score for block j is S_j = log sum_r exp(u_rj), u_rj = (a_j / sqrt(d_I)) * relu(phi_r . mu_j) (Eq. 8). One
   selection is shared by the whole tile and by the G query heads of the KV head.
5. Budget ("Ratio-K", Eq. 9): K_nom = min(V, max(K_min, min(K_max, ceil(eta V)))) tokens, V = visible length,
   eta = 1/3, K_min = 2048, K_max = 32768; k = ceil(K_nom / b) blocks.
6. Selection (Eq. 10): mandatory blocks first (sinks, the blocks the tile's queries sit in, an aligned local band;
   the paper points to FlashPrefill V2's defaults: 2 sink blocks, 4 local blocks), counted in the same quota; the rest
   of the quota goes to the best-scoring other blocks, ties to the smaller block id.
7. Full attention where the approximation is weakest: every query when N <= K_min, and the last r_tail = 8 groups of
   M_tail = 128 query vectors per KV-head group (the prompt's last 1024/G tokens).

**B. Block-sparse attention.** Exact softmax attention over every causal position of the selected blocks (Eq. 11-12),
original Q/K/V in BF16, no correction for the dropped mass. On H200 it is FlashPrefill V2's FA3-derived pipeline (TMA,
wgmma, warp specialisation).

Error bound (Theorem 4.1): the omitted mass delta <= (1 - beta) min{1, e^(2 eps_rank) tau(k_free)}, and
|o_dense - o| <= D_V delta. It is lossy; how lossy depends on how concentrated the real attention is.

The published code differs from the paper: its torch mask builder keeps blocks by a cumulative-mass target
(`VLLM_BFLA_KEEP_MASS=0.99`) on mean- or "flat64"-pooled Q/K blocks of 256, local 8 blocks, no Ratio-K and no FP8
key state. This port follows **the paper's** deterministic Ratio-K rule, because the paper's Qwen3.8-Flash-Next
numbers are for it.

## 2. How it meets pancake's attention, and why it does not help Flash-Next here

**Qwen3.8-Flash-Next (main engine, the daily model).** Its 12 full-attention layers are QSA layers. Pancake's prompt
path already runs QSA sparse (src/prefill/prefill.cpp, "QSA" section): per **query** the indexer (4 heads, 128 dims,
one pooled key per 4 cells) scores every completed block (`qsa_block_scores[_tc|_two_pass]`), `qsa_block_topk` keeps
`qsa_selection_width` = 2048 + 3 cells, and `qsa_prompt_attn_batch` (gfx11 WMMA with STRATA_HIP_WMMA=1) attends only to
those cells. Attention cost is therefore **linear** in the prompt (2051 cells per query); only the indexer and the
select grow with N^2.

BFLA's Flash-Next gains are against **dense FA3**, and the paper's own QSA baseline is as fast or faster than BFLA
(LongBench-v2 TTFT 1.490x QSA vs 1.435x BFLA; RULER 128K 1.319x vs 1.273x; operator level BFLA wins at 8-64K and
QSA at 96-128K). BFLA's win there is accuracy (RULER 94.36 vs QSA 92.39, dense 94.65), not speed. Put into pancake's
QSA layers, BFLA's budget (V/3 up to 32,768 tokens) would read ~16x more cells per query than QSA's 2051 at 128K.
So: **no speed ceiling on Flash-Next**; not implemented there. The one transferable idea - one selection per tile of
queries instead of per query, to cut QSA's N^2 select cost - is a different, lossy change to the model's own sparse
attention; its overlap can be measured with the existing `STRATA_SEL_OVERLAP` diagnostic first (MOD_PLAN phase 11, "BFLA (RDNA3)").

**Qwen3.6-35B-A3B / Qwen3.8-35B-A3B-Distill (src/qwen35 engine, `strata` switches to it for qwen35moe files; builds on
HIP).** These have 10 dense gated full-attention layers (16 query heads, 2 KV heads, head dim 256, GQA 8). Their
prompt chunk attention today (q36_engine.cu, `layer()`, T > kGemmT) is: K/V of the whole prefix converted to FP16
(`k_kv_f16`), Q.K^T as an FP16 hipBLAS strided GEMM into **materialised FP32 scores** over the whole prefix, a
causal softmax kernel writing FP16 probabilities, P.V as a second GEMM. This is exactly the attention BFLA targets
(the authors ran Qwen3.6-27B and Qwen3.5/3.6 MoE), and it is quadratic both in FLOPs and in score traffic. This is
where the port goes.

## 3. Expected share of prefill time (estimates, not measured yet)

Dense attention work for an N-token cold prompt, Qwen3.6-35B (10 attention layers, H = 16, D = 256):
2 N^2 H D per layer = 8192 N^2 FLOP, x10 = 81,920 N^2: 1.4 PFLOP at 128K, 5.6 PFLOP at 256K. The current path
also writes and reads the scores: sum over chunks of T x P x H values in FP32 + FP16, ~12 bytes per score:
~1.6 TB per layer at 128K (6.6 TB at 256K). At ~50 TFLOP/s of FP16 GEMM and ~800 GB/s that is roughly 45 s at
128K and ~180 s at 256K, against perhaps 60-130 s for everything else at 1,000-2,000 tok/s (Qwen3.6 prompt reading on
this card was never measured in pancake). Attention's share should therefore be ~30-45% at 128K and ~50-60% at 256K
(it was 38% at 42K on strata-poly / RTX 3060).

Kept work under the Ratio-K budget (sum over query positions of min(max(p/3, 2048), 32768) vs p): 3.2x less at 128K,
4.9x less at 256K, 1.8x at 64K, ~1.4x at 32K (the floor). The fused kernel also removes the score traffic, which is a
gain even with every block selected (dense mode = MOD_PLAN 8.8's fused prompt kernel for this engine).

## 4. RDNA3 plan

gfx1100 facts used (checked: `qsa_prompt_attn.cu` S6 comment, verified on the card by earlier work; hip_backend.cmake):
wave32; `v_wmma_f32_16x16x16_f16` (gfx11 layout: A lane l holds all 16 k of row l%16, lanes 16-31 repeat 0-15; B lane l
all 16 k of column l%16; C/D lane l holds D[2i + l/16][l%16], i = 0..7); 64 KiB LDS per workgroup (128 KiB per WGP);
no FP8 matrix instructions (gfx12 has them), no TMA / wgmma / warp specialisation. Following the HipKittens lesson,
every wave does the same work (no producer/consumer waves).

Three kernels, all opt-in, in `src/kernels/cuda/bfla.cu` (HIP wave32 only; the CUDA and gfx906 builds get stubs that
return false, so callers keep their path):

1. **Features** (`bfla_features`): per token and KV group, phi = N(mean over the G query heads of N(q_h)) reduced
   to 128 channels by adjacent-pair sums and normalised; per key block the mean of the same transform of K. Both from
   the FP16 Q/K the attention reads. Key states are kept in FP32 (no FP8 on gfx11). The Hadamard rotation is left
   out: relu(phi . mu) is invariant under an orthonormal rotation of both sides, so it only matters for FP8 storage.
   The key states are recomputed for the whole prefix every chunk (O(P), ~1/4 of the existing `k_kv_f16` pass), so a
   restored conversation cache can never leave them stale.
2. **Scores + select** (`bfla_select`): one workgroup per (selection tile = 128/G tokens = 128 rows, KV head) computes
   S_j for every visible block (online log-sum-exp over the tile's rows), then the Ratio-K selection with 2 sink blocks,
   the tile's own blocks and 4 local blocks mandatory, a 32-step bisection for the k-th score and the smaller-id tie
   rule, and writes an ascending block list. Deterministic.
3. **Block-sparse flash attention** (`bfla_attn`): one workgroup of 4 waves per (64/G tokens x G heads = 64 rows,
   KV head); wave w owns 16 rows and all 256 dims. Q (64 x 256 FP16) is staged in LDS once; per selected block, 16-key
   steps: K rows to LDS, V transposed to LDS (so a B fragment is one 32-byte read), S = Q K^T in 16 WMMAs, causal mask
   inline only on blocks that reach the diagonal, online softmax in exp2 with FP32 row state (row max/sum by
   half-wave xor shuffles), P to FP16 through a per-wave LDS tile, O += P V in 16 WMMAs (O is 16 FP32 C tiles = 128
   VGPRs a lane). Output FP32 in the engine's `att` layout; the gate multiply stays the existing `k_gate_mul`.
   Numerics class equal to the current path (FP16 operands, FP32 accumulation, FP16 probabilities), different
   summation order: not bitwise.

Dense mode (`STRATA_BFLA=dense`) feeds every causal block to kernel 3: the same kernel as exact (up to rounding)
attention. It is the "all blocks selected" correctness check and a candidate replacement for the GEMM path on its own.

## 5. Switches

- `STRATA_BFLA=1` (or `sparse`): BFLA on the qwen35 engine's prompt chunks; `dense`: the fused kernel with every
  block; unset or `0`: today's path, byte-identical.
- `STRATA_BFLA_MIN` (32768): a chunk takes the BFLA path only when its last position + 1 reaches this.
- `STRATA_BFLA_ETA` (0.333), `STRATA_BFLA_KMIN` (2048), `STRATA_BFLA_KMAX` (32768), `STRATA_BFLA_SINK` (2),
  `STRATA_BFLA_LOCAL` (4), `STRATA_BFLA_TAIL` (1024/G tokens of the final chunk, full attention).
- A start-up warning says the option is lossy.

## 6. Checks

- `bfla_test` (GPU, synthetic, no model): kernel 3 with every block vs an FP64 CPU reference and vs the
  materialised dense path; kernel 2's lists vs a CPU reimplementation of Eq. 8-10; sparse mode vs the reference
  restricted to the selected blocks (must match) and vs dense (tolerance, reports the dropped mass).
- End to end: the long-context recall / token-forcing gate MOD_PLAN uses for lossy switches (needles at 64K/128K,
  teacher-forced top-1 agreement vs the dense engine), then TTFT A/B at 64K and 128K.
