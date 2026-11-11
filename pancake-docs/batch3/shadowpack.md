# Batch 3, shadowpack: F4 / A27 sub-2-bit shadow pack, REAP (order 5), S2-MoE KV rollback (order 4), HOBBIT precision swap

Branch `dev/p16/shadowpack` (from `dev/phase16-academic` 3f4ba97cf). CPU and offline only: no GPU, no engine run, no model
loaded. Model bytes read: about 120 MB (5 layers x 16 experts of the IQ2_XS shard, read through the GGUF mapping).
The engine's default path is unchanged: the only build change adds two targets (`strata_shadow_pack`, `shadow_pack_test`).

## F4 / A27: sub-2-bit shadow of the gate/up rows

### Source check

- **HOBBIT** (arXiv 2411.01433v2, read in HTML): the low copy is int4 for an fp16 model (RTX 4090) or int2 for an int8 model
  (Jetson AGX Orin), a 4x step. It is GPU-centric: on a miss the chosen-precision expert is loaded over PCIe 4.0 and the GPU
  computes it, so the low copy saves PCIe bytes. Both copies live in 256 GB of host RAM. Models: Mixtral-8x7B and Phi-MoE
  (8 or 16 experts, top-2). Accuracy loss at most 1% (GSM8K, TruthfulQA). The quantizer for the low copy is not named.
- **Mismatch with our setting.** Our misses are computed on the CPU from RAM (DDR4), so a shadow saves DDR4 bytes, not PCIe
  bytes. The only step down from our files is IQ1_M: 1.46x fewer gate/up bytes than IQ2_S, 1.18x fewer than IQ2_XXS, none
  for the 3 layers that are already IQ1_M, and none for down (Q2_0 in all 48 layers). The source is already 2-bit, not
  fp16/int8. HOBBIT's own numbers therefore do not transfer.
- **MoBiLE** (arXiv 2510.12357, abstract): it halves the number of experts for unimportant tokens. That is a top-k reduction
  (the A1/A4 family), not a low-precision copy. Citing it for A27 is a mismatch.

### What was built

- `include/strata/artifact/shadow_pack.hpp`: the converter, header-only. It decodes a layer's gate/up rows with ggml's own
  dequantizer and encodes them again with ggml's own quantizer (IQ1_M by default, or IQ1_S / IQ2_XXS). It writes a GGUF that
  holds only the shadow tensors (`blk.L.ffn_{gate,up}_exps.weight`, `[2560, 640, 512]`). For a full conversion it also writes
  a `native_experts.txt` v4. In that file, a converted layer names the shadow GGUF for gate/up (a path relative to the
  --native shard's folder) and the model shard for down. So `expert_layout_load` and the CPU expert kernels read the
  shadow without any change. A layer is converted only when the shadow is smaller (the min rule of `shadow_bytes.py`), unless
  `--layers` lists it.
- `src/artifact/shadow_pack.cpp`: the `strata_shadow_pack` CLI. `--stride s` converts experts 0, s, 2s, ... (a measurement
  slice, with no index). `--eval n` measures the expert-output error on n random N(0,1) activations per expert. `--flat`
  encodes with an importance matrix of ones instead of ggml's no-imatrix weighting (each weight weighted by its square). No
  imatrix ships with the model.
- `src/artifact/shadow_pack_test.cpp` (ctest, label `cpu`): it converts a synthetic 2-layer shard with the model's widths. It
  then checks: (1) `expert_layout_load` reads the shadow index (v4; layer 0 IQ1_M/Q2_0, layer 1 unchanged); (2) every
  file/offset names its tensor (the `check_experts_gguf` conditions); (3) the shadow rows are bit-exact
  `ggml IQ1_M(decode(IQ2_S))`; (4) `native_gu_rows` / `native_down_rows` on a shadow blob match the float expert
  (rel 0.013, which is q8 activation rounding).

Run: `strata_shadow_pack --pack <Strata-data>/packs/iq2_xs --gguf <models>/IQ2_XS/...-00001-of-00002.gguf --out DIR
--layers 0,1,8 --stride 32 --eval 4 [--flat]`. A full conversion drops `--stride` and `--layers`. To load a full
conversion, the shadow GGUF must stay where the relative path in the index points.

### Measured offline (real pack, 16 experts per layer, stride 32; relative L2 error against the IQ2 source)

| layer (source) | gate/up bytes | weights x^2 / flat | expert output x^2 / flat |
| --- | --- | --- | --- |
| 0 (IQ2_S) | -31.7% | 0.539 / 0.452 | 0.765 / 0.553 |
| 1 (IQ2_XXS) | -15.2% | 0.527 / 0.442 | 0.794 / 0.579 |
| 24 (IQ2_S) | -31.7% | 0.536 / 0.441 | 0.883 / 0.616 |
| 47 (IQ2_S) | -31.7% | 0.538 / 0.444 | 0.915 / 0.625 |
| 8 (IQ1_M round trip) | 0% | 0.115 / 0.114 | 0.262 / 0.261 |
| reference: layer 0 IQ2_S -> IQ2_XXS | -19.5% | 0.355 | 0.443 |

- The expert output error uses random activations. Real inputs are better aligned with the weights, so these numbers are
  pessimistic, but they are the only inputs available (no MoE-input dump exists).
- On synthetic Gaussian weights (scratch check), float -> IQ1_M directly gives 0.58 (x^2) / 0.45 (flat), and float -> IQ2_S ->
  IQ1_M gives 0.65 / 0.52. Most of the error comes from IQ1_M without an imatrix, not from the second requant. The real
  weights match the Gaussian case.
- **CPU time is not proportional to bytes.** The engine's `native_gu_rows`, one token, one thread, rows streamed from DRAM
  (scratch bench, 3 runs, shared machine at load 6-7): IQ2_S 208-217 us per expert (4.8-5.0 GB/s), IQ2_XXS 190-198 us,
  IQ1_M 191-200 us (3.6-3.8 GB/s). Per thread, IQ1_M saves about 8-10% of IQ2_S's time and nothing against IQ2_XXS. The rows
  are compute-bound per thread. The byte saving (-18.2% overall) only becomes time when the pool is bandwidth-bound: 8 threads
  x ~5 GB/s is about DDR4's practical limit, so the truth is somewhere between the two. The time estimate in RESULTS.md
  (3.8-8.5% of decode, byte-proportional) is an upper bound. With compute-bound rows it is about 1-3%.
- RAM: with the min rule, 45 layers x 512 x 716,800 B = **16.5 GB** (the plan says 17.6 GB, which counts the 3 IQ1_M layers
  twice).
- Conversion cost: about 1.7 thread-seconds per expert (IQ1_M encode, 2 threads) -> 45 x 512 experts = about 39,000
  thread-seconds (5.4 h on 2 threads).

**Verdict: do not build the engine switch now.** The time gain is at most the byte-proportional 3.8-8.5%, and probably
1-3% given the measured kernel. The quality cost per missed expert is large (weights 0.44-0.54, expert output 0.55-0.9 on
random inputs). The shadow also adds 16.5 GB to a 57 GB box. If it is ever pursued: (1) collect an imatrix (needs a model
run); (2) run the KL / task gate with the full shadow loaded as a replacement pack, which today's loader already accepts and
which gives an upper bound on the quality loss; (3) only then add a CPU-miss-only read path.

## REAP (order 5): saliency on recorded routing traces

### Source check

- REAP (arXiv 2510.13999, ICLR 2026, read in HTML): `S_j = 1/|X_j| sum_{x in X_j} g_j(x) ||f_j(x)||_2`. Here X_j is the set of
  tokens routed to j, g_j is the gate normalized over the top-k, and the router is renormalized over the experts left after
  pruning. It prunes 25% / 50% of each layer. Calibration: 1,024 samples x 2,048 tokens for models up to 110B (C4 +
  evol-codealpaca; more data for larger models). Models: 8-384 experts, top-1 to top-8. Code accuracy drop 1.9% (25%) and
  6.9% (50%) for models up to 110B; 0.16% / 1.2% for Qwen3-Coder-480B. Pruning needs no re-quantization (FP8 and W4A16 models
  were tested).
- **Mismatches.** Our trace has 10,960 routed entries per layer, about 1,000x less than REAP's calibration (per layer, 69
  experts are never routed and 164 are under 5 entries). The trace has no gate weights: `--dump-routing` writes zeros from the
  serve path on purpose (the multi dispatch fuses the router weights, see generate.cpp `drive_pool_multi_impl`). Only
  `STRATA_WINDOW_TRACE` records them, and no such trace exists. Expert output norms are in no trace, and the existing hidden
  dump (A19) is the head's input, not the MoE input. Whether pruning compounds badly with 2-bit experts is untested. Our model
  has 512 experts and top-10, more than any model in the paper.

### What was built

`tools/pancake/reap_saliency.py` (5 pytest cases, `tools/pancake/test_reap_saliency.py`). It computes REAP's score where the
trace allows: the gate part from weights, output norms from an optional `--act-norm` .npy. Without weights it falls back to
routed frequency and prints that it did. Per prune ratio it reports the routed entries and gate mass that land on pruned
experts, and the CPU-missed experts per window through `replay_cache_policy`'s adaptive tier. There are two bounds, because
the trace holds only the top-k: `drop` (pruned entries vanish) and `miss` (each pruned expert is replaced by a non-resident
one). `--holdout` scores on even windows and evaluates on odd ones.

### Result on the 2026-11-08 trace (sha256 37b87e16...aa9240, profile `pancake-profile-iq2_xs.bin`, 10,909 slots; frequency score only)

| | entries on pruned | CPU misses per window (drop / miss bound) | hit rate (drop / miss) |
| --- | --- | --- | --- |
| no pruning (in sample / held out) | - | 362.7 / 390.1 | 79.8% / 76.9% |
| 25%, in sample | 1.3% | -12.3% / -4.5% | 81.8% / 80.8% |
| 25%, held out | 2.4% | -18.5% / -6.3% | 80.5% / 78.6% |
| 50%, in sample | 9.1% | -79.2% / -27.2% | 95.1% / 86.4% |
| 50%, held out | 11.1% | -79.2% / -24.0% | 94.3% / 83.8% |

CPU experts are 21-46.5% of a window (time_ceiling), so 25% pruning is worth roughly 1-9% of decode time and 50% roughly 5-37%.
Both are lossy, and the quality cost on this model is unknown. **No pruning is proposed.** The held-out 50% row says the
pruned experts still carry 11% of the routing. The A28 candidate (the ISTA-DASLab Coder file with 256 of 512 experts) is
already a 50%-pruned model made by the quant authors with their own calibration. It is the cheaper way to test the idea.
Next: one `STRATA_WINDOW_TRACE` run gives REAP's gate part. Output norms need a per-layer MoE-input dump, which would be an
engine change.

## S2-MoE (order 4): context-aligned shared-KV drafting with rollback; design note only

Source: arXiv 2608.15018v2 (HTML) and github.com/angerybob/S2-MoE (README, `docs/qwen-hybrid-s2moe.md`).

- **The paper's draft** is the target itself with fewer experts per token (OLMoE 8->2, DeepSeek 6->1, Qwen3 8->3, GPT-OSS
  4->1; no layer skipping). It reads the target's KV cache; its KV writes are temporary, rolled back before verification, and
  only accepted tokens are committed. Context alignment raises acceptance, for example from 36.08 to 48.71 (DeepSeek GSM8K)
  and from 33.01 to 59.09 (OLMoE GSM8K). Rollback cost is not reported. Batch size and greedy vs sampling are not stated.
  NVIDIA only (Orin, RTX 4090), CUDA gating kernel.
- **Hybrid models in the repo.** The Qwen3-Next / Qwen3.5 driver is marked experimental: greedy, one request, no MTP. It
  restores recurrent snapshots per fork depth. It reports no speedup or acceptance numbers.
- **Strata already has the exact part.** The native MTP drafter reads the target's hidden state, writes its own K/V cell per
  draft, and leaves rejected cells to be overwritten by the next round (`src/core/mtp.cpp`, "the K/V of the rejected rows is
  not caught up ... the next round writes it first"). The target's GDN state is replayed after a partial accept. The MTP
  layer's 512 experts are resident in VRAM (`include/strata/core/mtp.hpp`, 708 MB), so drafting causes no expert misses.
- **The paper's draft model does not fit.**
  - Our MTP accepts 71% of drafts on short chat and 94% on edits. S2-MoE's best context-aligned acceptance is 40-59%.
  - A full-depth reduced-expert draft costs the dense part of one decode step per draft token: 48 layers of attention/GDN,
    shared experts and router. MTP costs 2.2-2.5 ms for a whole window.
  - Each draft step would need a GDN state snapshot, not a KV trim.
  - Draft experts outside VRAM would miss to the CPU. That is the naive shared/top-r self-drafting the plan says not to build.
- **The transferable piece** is S2-MoE 4.1 (admit a draft while p_i * T_AR >= new experts x T_exp + C_draft). That is EcoSpec
  / A22 DISCO / A23, owned by dev/p16/disco-suffix. Reuse-aware gating (4.2) is lossy and is order 6, not started.

**Verdict:** no code. There is nothing to replay offline: no window trace with drafts exists, and a self-draft's acceptance
needs model runs.

## HOBBIT mixed precision: the precision swap does not transfer (evidence)

- `bench/results/2026-11-10-offline/gguf_table.json` (header only): gate/up are IQ2_S in 34 layers, IQ2_XXS in 11 and IQ1_M in
  3; down is Q2_0 (2.25 bpw) in all 48.
- `Strata-data/models/` holds only this file and the Q2_0 file. Q2_0 is 2.25 bpw, so it is no lower copy for gate/up, and it
  is the same type as down.
- There is no copy below IQ2_XS to swap to. The only step is a new IQ1_M requant of gate/up, which is F4 above: 1.18-1.46x
  fewer bytes instead of HOBBIT's 4x, a per-thread kernel time saving of only 0-10%, and 0.44-0.54 relative weight error.
- HOBBIT's gate-magnitude proxy (order 2) is owned by dev/p16/rankguard. It needs a weighted trace, as REAP's gate part does.

## Tests run

- CPU build (`-DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_CUDA=OFF`, ggml from the pinned commit 3cf03257f copied into the build
  dir): `strata_shadow_pack` and `shadow_pack_test` compile without warnings. `ctest -L cpu -R shadow_pack`: passed (27 s).
- `python -m pytest tools/pancake/test_reap_saliency.py`: 5 passed.
- Offline measurements above: the CLI on the real pack slice; `reap_saliency.py` on the trace in sample and held out.
- Not run: anything on the GPU, the engine, a quality gate, or a full 48-layer conversion.

## Risk

- None to the default path: the new targets and the tool only. The shadow format reuses native_experts.txt v4, so a future
  loader change is not needed to read it, but nothing reads it today.
