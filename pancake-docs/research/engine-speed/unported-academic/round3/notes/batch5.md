# Round-3 batch 5: A30, A31, A33, A35, A36 (read-only, 2026-11-12)

Scope: check the five MOD_PLAN rows against the papers and the repo. No code edits, no builds, no GPU. Read levels are given per paper. Tool note: the PRESERVE full-text fetch returned an error from the fetch tool, so PRESERVE is abstract only. Quotes for MonoMoE come from a summarizing fetch and must be checked against the paper before use.

## A30 Fiddler: CPU share in prefill (token-count threshold)

### Takeaway
Fiddler has no token-count threshold. Its switch is a per-expert latency comparison (CPU compute time against GPU compute plus weight transfer time). Pancake's `auto` share already uses that latency model, and the 4096 cut in the plan row is not from the paper. On this 7900 XT (gfx11) the share is OFF by default, so the A30 test is an env A/B and not a tuning of an existing default.

### Cited Findings
- Claim (abstract): Fiddler decides CPU or GPU execution per expert; reports 1.26x single-batch, 1.30x long-prefill, 11.57x beam search speedups. Read level: abstract only for the speedup list — [Fiddler arXiv 2402.07033](https://arxiv.org/abs/2402.07033).
- Full text (HTML, read level: full text, sections 3.2, 3.3, 4.2): the paper says CPU execution is better for small inputs and GPU is better above "some threshold", but gives no number. Algorithm 1 is a per-expert check: run on GPU if the weights are resident, or if CPU latency exceeds GPU latency plus transfer latency; the latency constants are measured at init and not reported — [Fiddler HTML](https://arxiv.org/html/2402.07033).
- Prefill numbers in the full text: average TTFT for 512 to 4096 input tokens is 1.07x faster than DeepSpeed-MII and 1.65x faster than Mixtral-Offloading. Model is Mixtral-8x7B at 16-bit. Hardware is Quadro RTX 6000 (24 GB) with Xeon Gold 6126, or RTX 6000 Ada (48 GB) with Xeon Platinum 8480+. The text does not say which environment each number comes from — [Fiddler HTML](https://arxiv.org/html/2402.07033).
- Repo: `src/prefill/prefill.cpp` lines 270-330. The share (`STRATA_PREFILL_CPU_SHARE`, `auto`) is ON by default only where it was measured (CUDA, one GPU, no batch slots, 0.1.41). `STRATA_PREFILL_CPU_SHARE_MAX` default is 3072 in `cpu_share_max()`, and the default-armed path returns `share_rules::default_share_limit()`, which is `kShareLimitNew` (3072 per the `arch_defaults.hpp` comment) with `kShareLimitOld = 1024` at `include/strata/prefill/share_rules.hpp` lines 21-24. The `auto` gate already compares CPU time per expert with GPU time per streamed expert (`prefill.cpp` lines 270-276).
- Repo: `include/strata/core/arch_defaults.hpp` line 33: the architecture rule is false for gfx11 (RX 7000 included). On this 7900 XT the share is therefore off unless set in the environment. Measured HIP wins are on gfx12 (R9700, RX 9070, RX 9060) only.
- Repo header numbers (not Fiddler): chunks below 1024 tokens were -20% to -35% at 512 and 1000 tokens on RTX 3060, P100 and RTX 5070 (8/8 pairs each), mean KL about 0.004.

### Inferences
- The plan row "deltas < 4096 skip the staged path" does not match the code (default limit 3072 when armed, 1024 measured). Fix the row to say: the threshold is the existing `STRATA_PREFILL_CPU_SHARE_MAX`, and the paper adds no threshold.
- The Fiddler mechanism is what `auto` already does. The new information from the paper is a sanity check that a cutoff by size is the wrong tool; the gate should stay the latency gate.
- The "~0.4 s per agent turn" estimate in the plan has no source. It is an estimate built from the 2.4K tok/s figure in the plan, not from a measurement on this card.

### Gaps
- No Fiddler numbers on AMD or on RDNA3. Mixtral-8x7B FP16 on Quadro/RTX 6000 is not comparable to Qwen-class MoE on a 20 GB card.
- No evidence in either source for the 4096 cut.

### Pancake status
- Coded and default-off on gfx11: `STRATA_PREFILL_CPU_SHARE` (`=auto`, `=0`), `STRATA_PREFILL_CPU_SHARE_MAX` (`prefill.cpp` 309-330).
- Plan row: "planned (env A/B: `CPU_SHARE=0`, `SHARE_MAX=1024`)". The env names in the plan are short forms; the real names carry the `STRATA_PREFILL_` prefix.

### Smallest test
- Env A/B on the 7900 XT with a 512, 1000, 2000 and 3000-token prompt, arms `STRATA_PREFILL_CPU_SHARE=0` and `STRATA_PREFILL_CPU_SHARE=auto` (default limit) and `auto` plus `STRATA_PREFILL_CPU_SHARE_MAX=1024`. A B B A, 4 starts per cell. Prefill timing line from `strata prefill timing`. GPU time: about 25 to 40 min including model load, and it needs the night queue (CPU pool competes with the decode host).
- Exact or lossy: the share changes which device computes an expert, not the arithmetic. Check mean KL against the share off (the code header uses ~0.004 as its bar); it is lossy only if the bit-level output changes.

---

## A31 GDN prefill chunk 64 with blockwise inverse and fused prep

### Takeaway
Chunk 64 is already implemented in `src/prefill/gdn_chunk.cu` (CH = 64, WMMA path default). The paper gives no kernel speedup on any AMD or on a 2024-era GPU. Its only throughput figure is an H100 training-throughput plot, and the text summary says Gated DeltaNet is slightly slower than Mamba2 there. The 5-11% gain in the plan is unsourced. The decision stays with the stage bench (plan row).

### Cited Findings
- Claim (abstract, read level: abstract only for the speedup claim): Gated DeltaNet combines a gate for fast memory erasure with the delta rule for targeted updates, and beats Mamba2 and DeltaNet on several benchmarks — [Gated DeltaNet arXiv 2412.06464](https://arxiv.org/abs/2412.06464).
- Full text (HTML v3, read level: full text for sections 2 to 4, first 100K of 118K characters read via summary tool): chunkwise form splits tokens into chunks of size C, and the paper gives no numeric chunk size for experiments. The intra-chunk solve is written as a C x C inverse of a strictly lower-triangular term (UT transform, Joffrain et al. 2006). No blockwise 16x16 scheme is described — [Gated DeltaNet HTML](https://arxiv.org/html/2412.06464v3).
- Throughput figure: caption says single H100 GPU. Text says Transformer++ is fastest at 2K "thanks to" FlashAttention-2, and Gated DeltaNet is slightly slower than Mamba2 (2-3K tokens/s, as summarized). The summary tool could not read the per-point values reliably — [Gated DeltaNet HTML](https://arxiv.org/html/2412.06464v3).
- Parallel DeltaNet (arXiv 2406.06484) and Mamba-2 (arXiv 2405.21060): not fetched. Gap.
- Repo: `src/prefill/gdn_chunk.cu` line 29 `constexpr int CH = 64`; a Kogge-Stone scan over 64 values per block (line 48); a per-chunk 64x64 A and P matrix in f32 (lines 66-72); the WMMA path is default and `STRATA_GDN_CHUNK=2` selects the FP32-only scan (line 542). Whether the solve is a 16x16 blockwise inverse was not confirmed here; the solve is a column solve in shared memory (line 119).
- Repo: `tools/pancake/test_gdn_chunk64_rule.py` tests the decision rule with made-up stage-bench output (its header says so). It is not a measurement.
- Internal timing (RTX 3060, 41,811 tokens, 2026-11-08, cited from the test file): GDN 11.9% of prefill, recurrence 6.0%, conv and gates 1.3%, out proj 4.4%. This is a 3060 log, not a 7900 XT number.

### Inferences
- A31 as written ("chunk 64") is not a new idea for this repo. The open question is the one the plan already names: whether the blockwise inverse and fused prep beat the current solve. That needs the 7900 XT stage bench, not the paper.
- Decision rule in the plan (build only if recurrence over WMMA floor >= 2 at 32K and GDN >= 6% of prefill) is consistent with the paper's lack of a number. The 3060 share (11.9%) suggests GDN may pass the 6% bar on this card but this is unmeasured.

### Gaps
- No source for "5-11% of cold prefill". The plan row should carry "unsourced" beside it.
- No AMD prefill data in either paper.

### Pancake status
- Chunk 64 coded (`src/prefill/gdn_chunk.cu`), WMMA default. Blockwise 16x16 solve: not verified in code.
- Stage bench `hip_gdn_stage_bench` in `tests/hip/gdn_stage_bench.cpp` (already in the repo; MOD_PLAN says "blocked-on-GPU").

### Smallest test
- `hip_gdn_stage_bench 4096 16384 32768` on the 7900 XT, then `tools/pancake/probes/gdn_chunk64_rule.py` on the output. GPU time: minutes. Also one `strata prefill timing` at 32K to read the GDN share on this card (bar: 6%). Lossy: FP16-class (plan row); parity against the FP32 scan (`STRATA_GDN_CHUNK=2`) is needed before use.

---

## A33 Small-M fusion of pre-doorbell chains

### Takeaway
The SGLang paper does not make a small-M fusion claim. Its abstract is about the SGLang frontend and runtime (RadixAttention, compressed FSM decoding, up to 6.4x throughput). The Qwen3 report did not show a gated residual in the part read. The fusion itself is coded and opt-in, and the GPU screen already found about +1% at best (noise-level in the confirm run). This is a HOLD, not a new item.

### Cited Findings
- SGLang (arXiv 2312.07104), read level: abstract only. Claims up to 6.4x higher throughput than prior systems on several models and tasks; RadixAttention and compressed FSM decoding are its runtime optimizations. No mention of operator fusion or small-M kernels — [SGLang arXiv](https://arxiv.org/abs/2312.07104).
- Qwen3 technical report (arXiv 2505.09388), read level: summary of full text for the first 100K of 134.5K characters, appendix not read. No sentence about gated residual connections was found. The only gating mentioned is SwiGLU in the feed-forward block — [Qwen3 HTML](https://arxiv.org/html/2505.09388). The "gated residual" citation in the plan is not supported by the part read; it may be in the unread appendix.
- Repo: `STRATA_EXPDEC_SWIGLU` in `src/kernels/cuda/iq_kernels.cu` line 4227 (opt-in, `=1`). `STRATA_ROUTER_FUSE` in `src/kernels/cuda/native_bf16.cu` line 472 (opt-in, `=1`). Both only act with `STRATA_ROUTE_PUBLISH=1` and no deferral (plan notes).
- Plan GPU results (from the MOD_PLAN kern-hip rows, not re-run): expdec-swiglu short +0.6% [-1.6, +2.8], edit +1.1% [-4.7, +6.9]; all3 no difference. Verdict HOLD.

### Inferences
- The SGLang citation adds nothing to A33. The source to cite for the idea would be a small-M fused-MoE paper with a measured number, and none was found in this batch.
- With a ceiling of about 1% per chain and a measurement interval wider than 1%, a screen cannot settle this. Kernel-level measurement (below) is the only cheap route.

### Gaps
- No primary source found for "SGLang small-M fusion". Search was limited to the item's own arXiv link.
- The gated-residual claim is not verified (appendix unread).

### Pancake status
- Coded, opt-in: `STRATA_EXPDEC_SWIGLU`, `STRATA_ROUTER_FUSE`. Measured HOLD (MOD_PLAN kern-hip). Note: plan says `STRATA_ROUTER_FUSE` never acts in the fast-hip or turbo arms.

### Smallest test
- `expdec_bench` (kernel microbench, minutes) with `STRATA_EXPDEC_SWIGLU` 0 and 1, A B B A, gate/up plus SwiGLU time per layer. Kernel gain times window share against the 2% bar (kernel_gate). GPU time: about 10 min. Exact: the fused kernel is bit-identical only if the q8_1 quantization order matches; check with `iq_multi_parity` bitwise.

---

## A35 Warm the Infinity Cache with next-layer weights (PRESERVE)

### Takeaway
PRESERVE, as far as the abstract says, prefetches weights and KV cache from HBM into an accelerator's on-chip cache during inter-device communication. It reports up to 1.6x end-to-end on unnamed accelerators with open LLMs. It does not name the Infinity Cache, and it has no single-GPU result. The transfer to a 7900 XT warm-up during a CPU wait is our own inference, not the paper's claim.

### Cited Findings
- PRESERVE (arXiv 2501.08192), read level: abstract only (the full-text fetch returned a tool error). Claim: prefetch model weights and KV cache from off-chip HBM to on-chip cache during communication, to overlap memory and communication; up to 1.6x end-to-end; a design-space exploration adds 1.25x performance per cost by choosing L2 cache size. Hardware and models are unnamed in the summary — [PRESERVE arXiv](https://arxiv.org/abs/2501.08192).
- The abstract summary does not mention Infinity Cache, AMD, or any specific accelerator. The "80 MiB Infinity Cache" in the plan is therefore not from this source.

### Inferences
- The paper's mechanism (overlap a memory read with a communication wait) is close to Pancake's `STRATA_IC_WARM_MB`, which reads next-layer pre-doorbell weights during the CPU combine wait. The overlap logic is the same; the hardware and the size of the win are not transferable from the abstract.
- Plan row "0-2 ms/window (est.)" has no source.

### Gaps
- Full text not read. Hardware, cache size, the warm-up method and any single-device ablation are unknown.
- No source for the 80 MiB size beyond the plan; the number needs checking against the 7900 XT spec (plan treats it as known).

### Pancake status
- Coded, opt-in: `STRATA_IC_WARM_MB` in `src/core/verify.cpp` line 1028 (warm_ thread during the CPU wait; falls back off on setup failure, line 1057). Measured "n.s." in plan 04.

### Smallest test
- The existing knob, with the new harness: `STRATA_IC_WARM_MB` at 0, 16, 40 and 80, A B B A, decode trace on an edit window. GPU time: about 30 to 60 min, night queue only. Exact: prefetch changes timing, not values. Gate: only if a 1 ms saving or more shows in the window trace.

---

## A36 Weight-major layout for VRAM-hit experts (MonoMoE)

### Takeaway
The round-1 note is right: MonoMoE is a persistent FP8 (E4M3, block-scaled) megakernel evaluated on H200, and "weight-major" means how CTAs are partitioned over expert weight tiles, not a change to how weights are stored. It does not touch Q2_0 or IQ weights. The plan row "weight-major layout for VRAM-hit experts / Q2_0 down" should be dropped. The ceiling the repo already measured (about +2% on an edit window if all VRAM-hit GPU work vanished) is below the bar.

### Cited Findings
- MonoMoE (arXiv 2609.04244), read level: full text via summary of the HTML version. Abstract: "weight-major persistent megakernel for block-wise quantized MoE decode"; integrated with vLLM and open-sourced in FlashInfer — [MonoMoE arXiv](https://arxiv.org/abs/2609.04244), [MonoMoE HTML](https://arxiv.org/html/2609.04244).
- Definition (Introduction and section 3.1, paraphrased by the summary): the kernel partitions expert weights into output-row stripes and places the whole decode-step token tile on the tensor core's small N dimension. The grid is split over weight tiles, not over expert-local token blocks. Tokens are not sorted or padded per expert.
- No weight repacking or offline relayout was found. Offline work is schedule tuning. Compile-time specialization fixes TMA descriptors and shared memory layouts.
- SwapAB is mentioned in section 3.1: it swaps the operand roles and yields a transposed output layout, without a transpose kernel. So the round-1 phrase "SwapAB tile mapping" is correct.
- Precision: all MoE weights and activations are E4M3 FP8 with 128x128 block scales on weights and 128-element activation blocks in the kernel (section 4.1). The abstract says "block-wise quantized", not FP8 only.
- Hardware: NVIDIA H200 only. Results: up to 1.54x over vLLM's Triton grouped GEMM for the routed MoE operator; 2.20 to 3.84x over a FlashMoE-FP8 adaptation; up to 18.7% lower time per output token end to end, accuracy preserved (from the summary of the abstract and tables).
- Repo: `tools/pancake/kernel_gate.py` line 514 (`STRATA_HIP_GLOBAL_WEIGHTS`, "global instead of flat weight loads") is a different idea and is not a layout change either. Plan notes say removing all VRAM-hit GPU work gains +1.99% on an edit window (kernel_gate), and the plan's bar is 2%.

### Inferences
- Verdict on the round-1 note: correct on the layout point. The only correction is the wording: "FP8" should read "block-wise quantized, E4M3 FP8 in the evaluated configuration".
- The MonoMoE idea does not transfer to this engine: Pancake's VRAM-hit experts are Q2_0 and IQ formats, and the paper's gains come from FP8 tensor-core shapes on H200 that a 7900 XT (no FP8 tensor path used here) does not share.

### Gaps
- No RDNA3 or Q2_0 number. No layout ablation in the paper.
- The summary tool paraphrased the quotes; they must be checked against the PDF before any use in a report.

### Pancake status
- No layout change coded. Relevant knobs: `STRATA_HIP_GLOBAL_WEIGHTS` (kernel_gate candidate FLAT, build flag), `tools/pancake/probes/hit_critical.py` (rule tool).

### Smallest test
- Two gpu_profile traces with deferral on, then `probes/hit_critical.py`. Build only if the 25%-faster-hit ceiling is at least 2%, as the plan says. GPU time: about 20 min for two traces. Expected outcome from existing numbers: no-go on edits, likely below the bar on short windows as well.

---

## Summary of corrections to MOD_PLAN

- A30: the "deltas < 4096" threshold is not from Fiddler; Fiddler has a latency gate, which `auto` already implements. Share is off by default on gfx11. Env names need the `STRATA_PREFILL_` prefix.
- A31: chunk 64 is already in the code. The 5-11% gain is unsourced. Paper numbers are H100 training throughput, not prefill on AMD.
- A33: SGLang paper has no fusion claim (abstract). Qwen3 "gated residual" not found in the part read.
- A35: PRESERVE does not mention the Infinity Cache; hardware unnamed; full text not read.
- A36: round-1 note is correct; the plan's "weight-major layout" wording is wrong. Drop the layout idea.
