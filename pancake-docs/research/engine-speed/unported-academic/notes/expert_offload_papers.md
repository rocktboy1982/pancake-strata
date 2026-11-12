# Expert-offload papers behind MOD_PLAN rows A27, A36, A44

Research date: 2026-11-12. Read-only pass: no code edits, no builds, no GPU. Full-text content was fetched with WebFetch, which returns a small-model summary, not verbatim text. Numbers below are as the summary reported them and should be spot-checked against the PDF before any figure is quoted in a paper or the README.

Two MOD_PLAN descriptions are wrong and this pass corrects them:
- **A44**: arXiv 2601.17615 is Athena (HPCA 2026), an online RL coordinator for CPU prefetchers and an off-chip predictor. It is not an MoE paper and not an epoch bandit over window types. The MOD_PLAN ledger already says the code is a stateless MAB, not Athena.
- **A27 and A36**: the cited MoBiLE paper is not a precision paper (it halves the expert count for unimportant tokens), and MonoMoE is a FP8 fused kernel for a datacenter GPU, not a weight layout for offloaded or VRAM-hit experts.

---

## A27: low-precision copy of experts read only on CPU misses

### Takeaway
HOBBIT (arXiv 2411.01433) is the paper behind A27 and does what the row says: a cache miss loads a low-precision copy of a low-importance expert instead of the full one, and a second threshold skips some experts entirely. The gain is reported on edge boards with Mixtral-8x7B and Phi-MoE, against baselines that are not the Pancake engine. The method is lossy and the paper's own ablation attributes only 1.19x to 1.57x to dynamic loading. MoBiLE (arXiv 2510.12357) is not a precision method; it is the reduced-expert fallback family that already sits in the parked turbo variants.

### Cited Findings

**HOBBIT, arXiv 2411.01433 [paper, full text read via WebFetch summary]**
- Title and authors: "HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference", Tang et al. (arXiv, cs.LG; no peer-reviewed venue listed). Source: [arXiv HTML v2](https://arxiv.org/html/2411.01433v2), [abstract](https://arxiv.org/abs/2411.01433).
- Storage: all experts stay in CPU memory; a subset is kept on the GPU at multiple precisions. Low-precision copies are read from CPU memory or SSD on a miss. The paper evaluates pre-existing model pairs (Float16 with Int4, Int8 with Int2) and does not say how the low-precision versions are produced (separate quantized files or conversion at load time is not described). Source: [HOBBIT HTML v2](https://arxiv.org/html/2411.01433v2).
- Selection rule on a miss: the K selected experts are ranked by normalized gate weight. Each expert's unimportance score is the sum of gate weights of the experts ranked above it; the top expert always loads at high precision. Score at most T1 loads high precision, otherwise low precision; a second threshold T2 skips the expert. Mixtral-8x7B uses T1 = 0.6 and T2 = 0.9, which the paper reports as a split of about 67% high, 30% low and 3% skipped. Source: [HOBBIT HTML v2](https://arxiv.org/html/2411.01433v2).
- Justification for gate weight as the proxy: reported Pearson correlation 0.99 with expert output magnitude. Source: [HOBBIT HTML v2](https://arxiv.org/html/2411.01433v2). MOD_PLAN line 1639 already flags that this is weight times expert norm on Mixtral only.
- Prefetch and cache (brief): the current layer's gating input predicts the next 1 to 3 layers; reported top-1 accuracy about 96% for the next layer and about 90% for the next two or three. The cache is a weighted sum of LRU, LFU, a new LHU score and farthest-layer distance, with separate high- and low-precision caches. Source: [HOBBIT HTML v2](https://arxiv.org/html/2411.01433v2).
- Hardware: RTX 4090 (24 GB, 256 GB host RAM, PCIe 4.0) and Jetson AGX Orin (32 GB unified memory), Samsung 980 PRO SSD. Models: Mixtral-8x7B and Phi-MoE. Baselines: Transformers, DeepSpeed-Inference, Llama.cpp, MoE-Offloading, MoE-Infinity, Fiddler (not all on every platform). Source: [HOBBIT HTML v2](https://arxiv.org/html/2411.01433v2).
- Speedups (decode): Orin vs Llama.cpp 13.0x (Mixtral) and 18.9x (Phi-MoE); Orin vs MoE-Infinity 3.64x and 9.93x; RTX 4090 vs MoE-Offloading 3.21x and 3.29x; RTX 4090 + CPU vs Llama.cpp 1.31x and 1.42x; vs Fiddler 0.99x and 1.46x. The abstract's "up to 9.93x" is, per the body, an average over configurations for Phi-MoE on Orin against one baseline, so "up to" overstates it. Source: [HOBBIT HTML v2](https://arxiv.org/html/2411.01433v2), [abstract](https://arxiv.org/abs/2411.01433).
- Ablations: dynamic loading 1.19x to 1.57x; prefetch about 1.05x when combined with mixed-precision loading. Source: [HOBBIT HTML v2](https://arxiv.org/html/2411.01433v2).
- Accuracy (Table 3): Mixtral GSM8K 0.52 to 0.51 under mixed precision; Phi-MoE GSM8K unchanged at 0.83; paper says accuracy drops by no more than 1%. Source: [HOBBIT HTML v2](https://arxiv.org/html/2411.01433v2).

**MoBiLE, arXiv 2510.12357 [paper, full text read via WebFetch summary]**
- Title: "MoBiLE: Efficient Mixture-of-Experts Inference on Consumer GPU with Mixture of Big Little Experts", Zhao et al., submitted 14 Oct 2025, comments say accepted to ASP-DAC 2026. Source: [arXiv HTML](https://arxiv.org/html/2510.12357), [abstract](https://arxiv.org/abs/2510.12357).
- Mechanism: no quantization or reduced precision anywhere in the paper. "Little" mode runs with half the default experts (top-K/2). Token importance is a confidence check on the little model's top output score: if above gamma (0.7 by default) the little token is kept; otherwise the big model regenerates the position. Fallback router outputs from the discarded pass drive cross-layer prefetch, which is training-free. Source: [MoBiLE HTML](https://arxiv.org/html/2510.12357).
- Hardware: RTX 4080 (16 GB), i5-12400, 64 GB DRAM, PCIe 4.0. Models in the body: OLMoE and Qwen1.5-MoE only (the abstract's "four architectures" is not supported by the body). Tasks: GSM8K (8-shot, 256 tokens) and HumanEval (0-shot, 512 tokens). Source: [MoBiLE HTML](https://arxiv.org/html/2510.12357).
- Speedup vs full-model offload baseline: 1.72x (Qwen MoE) and 1.60x (OLMoE) on average; vs Pre-gated MoE 1.28x; vs AdapMoE 1.21x. Source: [MoBiLE HTML](https://arxiv.org/html/2510.12357).
- Accuracy: OLMoE GSM8K full 0.515, reduced 0.371, MoBiLE 0.448 (about 6.7 points below full, which is not "negligible"). Qwen MoE GSM8K full 0.641, MoBiLE 0.608. Source: [MoBiLE HTML](https://arxiv.org/html/2510.12357).
- Exact or lossy: lossy by construction (approximates the top-K expert set for tokens judged unimportant). Expert counts are inconsistent inside the paper (Qwen 4 of 60 in the text versus 4/64 in Table II).

### Inferences
- A27 as written (HOBBIT) is a lossy substitute for a missing high-precision miss, and the quality cost in the paper is small on Mixtral but not zero on OLMoE-like models. Pancake's experts are already IQ2_XS, so the HOBBIT gap is larger: a per-miss copy would need to be below 2 bits, which is exactly the F4 shadow-pack regime.
- The F4 shadow pack already measured this regime: IQ1_M shadow weight error 0.53 and expert output error 0.55 to 0.91 on 16 experts (MOD_PLAN ledger, shadowpack row). That is a direct negative for the lossy form of A27 on this model, unless the CPU-miss copy is gated by the same importance rule HOBBIT uses (not yet tested).
- MoBiLE is closer to A4 (rank guard) and to the parked "BuddyMoE / MoBiLE / AdapMoE" re-selections (MOD_PLAN line 1804). It does not belong under A27.

### Pancake state (grep, read-only)
- Not implemented: no hits for `hobbit`, `mobile`, `big_little`, or a low-precision miss copy in `src/` or `include/`. The only existing fallback knobs are the A4 rank guard (`_TOP`) and the turbo re-selection (MOD_PLAN line 1256 says only one re-selection runs in `verify.cpp`).
- Partial precedents: `STRATA_SHADOW_EXPERTS` and `STRATA_SHADOW_MEMFD` exist in the tree (F4 shadow pack; MOD_PLAN marks F4 REJECT as an engine switch). setup.py already lists UD-IQ1_M shards (setup.py lines 488, 615), but those are whole-model downloads, not per-expert miss copies.

### Gaps
- The HOBBIT summary does not give the exact format of the low-precision copy, and the paper's quantization pipeline is not described. A reproduction would need the HOBBIT code (not checked in this pass).
- No direct evidence on a model with a 256-expert top-8 layout and IQ2_XS base. Both HOBBIT and MoBiLE test 8 to 64 experts.

### Minimal A/B arm for A27 (proposal, not run)
- Arm off: current path (IQ2_XS experts, A4 rank guard off).
- Arm on: on a CPU miss, if the expert's HOBBIT unimportance score (sum of higher-ranked normalized gate weights) is at most T1 keep the IQ2_XS read; otherwise read a sub-2-bit copy (IQ1_M-class) for that expert only; skip the expert above T2.
- Gate: needs a routing trace with gate weights (MOD_PLAN item 6 says none exists yet; the 219-window trace has zero weights), a `--dump-routing` run, and the task gate (A55) for quality. Blocked on a sub-2-bit source and disk space, as MOD_PLAN already says.

---

## A36: weight-major layout for VRAM-hit experts / Q2_0 down

### Takeaway
MonoMoE (arXiv 2609.04244) is a FP8 fused persistent megakernel for datacenter decode on H200. "Weight-major" means CTAs are partitioned over expert-weight tiles with all decode tokens placed on the tensor-core N dimension (SwapAB). It does not change the weight layout in memory, has no VRAM-hit or offload split, and does not touch Q2_0. The idea does not transfer to Pancake as a layout change. What might transfer is the per-layer question of whether VRAM-hit kernel time sits on the critical path, which MOD_PLAN already plans to check with `probes/hit_critical.py`.

### Cited Findings

**MonoMoE, arXiv 2609.04244 [paper, full text read via WebFetch summary]**
- Title: "MonoMoE: An Efficient Fused Mega-kernel for Quantized MoE Decoding", Gong et al., cs.AR, submitted 19 Aug 2026 (preprint, no venue). Code in the FlashInfer repo, `csrc/fused_moe/monomoe`. Source: [arXiv HTML](https://arxiv.org/html/2609.04244), [abstract](https://arxiv.org/abs/2609.04244).
- Tile mapping: CTA partition over expert-weight tiles; each CTA owns an output-row stripe and streams its weight panels against the whole decode-step token tile. Tokens sit on the N dimension of the MMA via a SwapAB formulation, so no transpose kernel and no padded [E, M, H] dispatch. The paper does not change the expert weight memory layout. Source: [MonoMoE HTML](https://arxiv.org/html/2609.04244).
- VRAM-resident versus offloaded split: not discussed; all MoE weights are FP8 on H200 HBM3e (141 GB). Source: [MonoMoE HTML](https://arxiv.org/html/2609.04244).
- Quantization: FP8 E4M3, 128x128 block scales for weights, 128-element activation quantization inside the kernel. No 2-bit or Q2 format. Source: [MonoMoE HTML](https://arxiv.org/html/2609.04244).
- Exact or lossy: lossy relative to an FP32 reference (FP8 casts). The paper does not claim bit-exact equivalence with the staged baseline; error reported as matrix-level cosine similarity at or above 0.998. Source: [MonoMoE HTML](https://arxiv.org/html/2609.04244).
- Evaluation: H200 with vLLM and CUDA Graphs. Models: Qwen3.5-35B-A3B, Qwen3.5-122B-A10B, GLM-5.2, DeepSeek-V3.1 (all 256 experts, top-8). Baselines: vLLM Triton grouped GEMM, and an FP8 adaptation of FlashMoE. Kernel speedup vs grouped GEMM 1.02x to 1.54x (Qwen3.5-35B 1.17x to 1.54x). End-to-end TPOT reduction vs grouped GEMM 2.9% to 18.7% depending on model and batch (B = 1 to 8). FlashMoE-FP8 is 2.2x to 3.8x slower on the kernel. Accuracy (GSM8K, HumanEval, B = 1): differences reported as within binomial SE. Source: [MonoMoE HTML](https://arxiv.org/html/2609.04244), [abstract](https://arxiv.org/abs/2609.04244).
- Internal inconsistency: the Introduction gives DeepSeek-V3.1 TPOT as a 1.2% regression to 3.7% reduction versus Section 4.4 at 3.5% to 12.8%. Source: [MonoMoE HTML](https://arxiv.org/html/2609.04244).

### Inferences
- The name collision is the issue: "weight-major" here is a kernel scheduling choice on a FP8 GPU, so it has no direct analogue in the Pancake Q2_0/IQ2_XS RDNA3 decode path, where the question is whether the VRAM-hit kernel is on the critical path at all.
- The kernel gain is measured on H200 with FP8. Its ceiling for Pancake is the hit-side share of wall time, which MOD_PLAN measured earlier as 1.58/2.47/0.30% (pre-deferral profile). That is below the 2% bar in the plan. A re-check needs a deferral-on trace.

### Pancake state (grep, read-only)
- No code hits for `weight-major`, `swapab`, or `monomoe`. `STRATA_HIP_GLOBAL_WEIGHTS` exists (flat loads, a separate item in MOD_PLAN F7/F12 rows); it is not the same mapping and was not read in depth.
- MOD_PLAN line 1549 already records the SwapAB point. The kernel-gate verdict for A36 is MEASURE (MOD_PLAN kernel_gate table).

### Gaps
- No Q2_0 or RDNA3 numbers in the paper. No measurement of the weight-major tile map on consumer or AMD hardware.
- The VRAM-hit path in Pancake is not described in the paper, so the proposed layout cannot be evaluated from the source.

### Minimal A/B arm for A36 (proposal, not run)
- Step 0 (no new kernel): run `probes/hit_critical.py` on a deferral-on gpu_profile trace (MOD_PLAN already plans two traces). If the hit share on the critical path is under 2% of wall time, stop.
- Step 1 only if step 0 passes: an A/B of a token-major versus expert-weight-major grid for the VRAM-hit kernel behind a build flag, bit-exact parity (`iq_multi_parity`) before any timing, timed with the paired-start method (A B B A).

---

## A44: epoch bandit that turns a mixed opt-in on or off per window type

### Takeaway
Athena (arXiv 2601.17615, HPCA 2026, Bera et al., ETH SAFARI group with Mutlu) is an online RL controller for CPU prefetchers and an off-chip predictor. It is not an MoE paper, not a bandit, and has no per-window-type on/off decision for expert features. The row's citation id is right but the description is wrong. The research question behind A44 (choose per window type whether a lossy mixed opt-in is on) is still valid and is what Pancake's `STRATA_BANDIT` already tries; the MOD_PLAN ledger says the code is a stateless MAB, not Athena.

### Cited Findings

**Athena, arXiv 2601.17615 [paper, abstract only]**
- Title: "Athena: Synergizing Data Prefetching and Off-Chip Prediction via Online Reinforcement Learning", Bera, Lang, Hengartner, Kanellopoulos, Kumar, Sadrosadati, Mutlu; HPCA 2026; submitted 24 Jan 2026 (cs.AR). Source: [arXiv abstract](https://arxiv.org/abs/2601.17615).
- Mechanism (abstract): each epoch Athena observes system features (prefetcher and off-chip predictor accuracy, bandwidth usage), picks an action (enable or disable the prefetcher and/or predictor, or change prefetcher aggressiveness), and receives a reward from changes in epoch metrics such as cycles per epoch. Source: [arXiv abstract](https://arxiv.org/abs/2601.17615).
- Epoch length, the exact state and action encoding, and the learning rule are not in the abstract. MOD_PLAN line 1641 cites a SARSA design with alpha 0.6, gamma 0.6, epsilon 0, 4 actions, 2K-instruction epochs, 3 KB Q-table, ChampSim, and +5.0 to +7.0% geomean IPC over a stateless MAB. Those numbers are not verified in this pass (body not read) and should be treated as [unverified] until the PDF is checked.
- Evaluation: abstract says it outperforms prior coordination policies across configurations, prefetcher/OCP combinations and bandwidths, with modest storage. No benchmark names or numbers in the abstract. Source: [arXiv abstract](https://arxiv.org/abs/2601.17615).
- WebSearch for "Athena MoE expert offloading" returned no Athena-MoE paper; the nearest MoE offloading results were different papers (expert eviction, FlashMoE, ExpertFlow, local routing consistency). Source: WebSearch, 2026-11-12.

### Inferences
- Exact or lossy: a policy that picks among existing settings does not change output bits by itself. Its effect is only through the lossy arms it toggles (A4 rank guard, A5 k-th miss, pool pre-read). The lossy part is in those arms, not in the bandit.
- The mismatch matters for the A44 claim "+2 to 5% on mixed sessions" (MOD_PLAN). That estimate is the project's own, not Athena's; the MOD_PLAN ledger already says so.

### Pancake state (grep, read-only)
- `STRATA_BANDIT` (and `_TYPE`, `_EXACT`, `_RULE`, `_MARGIN` variants) is read in `src/program/generate.cpp` (the only file matching the bandit search). MOD_PLAN line 1724 gives the arms: `STRATA_BANDIT=top,kmiss,preread` (+ `_EXACT`, `_RULE`, `_TYPE`).
- MOD_PLAN ledger (policy row, line 1511): offline replay shows the lossy arm ran 44.5 to 49.6% of epochs; `STRATA_BANDIT_MARGIN` 0.02 cuts it to 10.3 to 31.4%. Bandit default path is byte-identical over 640 configs.
- `STRATA_POOL_PREREAD` is read in `src/kernels/cpu/pool.cpp:786` and `include/strata/core/expert_source.hpp:1405` (the pre-read arm of the bandit).

### Gaps
- Athena body not read (epoch length, state features, reward, learning rule unconfirmed).
- No evidence in this pass that a per-window-type RL controller beats the stateless MAB on Pancake; the ledger's own finding (Athena-style state adds +1.4 to +1.7 points in the paper's CPU setting, reward correction +1.0) is unverified here.

### Minimal A/B arm for A44 (proposal, not run)
- Arm fixed-off: current default (mixed opt-ins off).
- Arm fixed-on: `STRATA_BANDIT=top,kmiss,preread` with the exact arm forced on.
- Arm bandit: `STRATA_BANDIT=top,kmiss,preread` with `STRATA_BANDIT_MARGIN=0.02` (the ledger's proposed cut), per window type.
- Measured with A B B A paired starts on the edit and short windows, as the plan requires. Pass criterion: the bandit arm keeps the edit gain of the best fixed arm without the short-chat loss. No new controller code is needed for this test.

---

## Summary table

| Row | Paper (verified id) | Read level | Mechanism in paper | Exact or lossy | In Pancake now? | Minimal A/B arm |
|---|---|---|---|---|---|---|
| A27 | HOBBIT 2411.01433 | full text read (summary) | low-precision copy on miss, gate-rank threshold, skip band | lossy | no; F4 shadow pack (lossy, REJECT) exists as partial precedent | on a miss: IQ2_XS unless importance <= T1, then IQ1_M-class copy; gated by routing trace + task gate |
| A27 (wrong ref) | MoBiLE 2510.12357 | full text read (summary) | big/little expert count, confidence gate gamma 0.7 | lossy | no; parked as re-selection variant | not a precision test; belongs with A4 |
| A36 | MonoMoE 2609.04244 | full text read (summary) | FP8 fused megakernel, token tile on N dim, weight-tile CTAs | lossy (FP8) | no; SwapAB point already in MOD_PLAN | step 0: `hit_critical.py` on deferral-on trace; build only if hit share >= 2% |
| A44 | Athena 2601.17615 | abstract only | online RL for CPU prefetch and off-chip predictor | policy only; effect via lossy arms | partly: `STRATA_BANDIT` stateless MAB | fixed-off vs fixed-on vs bandit with margin 0.02, A B B A starts |

## Source-check debts
- HOBBIT 9.93x: body says it is an average over configurations for one baseline; abstract says "up to". Use the body wording.
- MoBiLE "four models" claim in the abstract versus two models in the body. Use the body.
- MonoMoE DeepSeek-V3.1 TPOT range: intro and Section 4.4 disagree.
- Athena numbers (+5.0 to +7.0% geomean, SARSA parameters) are from MOD_PLAN, not from the paper body. Mark [unverified] until the PDF is read.
