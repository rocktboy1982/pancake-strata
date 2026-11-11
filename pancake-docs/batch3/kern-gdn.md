# A31 / A36 / 16.11: GDN chunk 64, weight-major hit experts, attention and long-context (batch 3, 2026-11-11)

Agent `kern-gdn`, branch `dev/p16/kern-gdn` (worktree `Strata-p16-kern-gdn`, from `dev/phase16-academic` 3f4ba97cf).
Kernel wave: this branch merges last. The deep-research did not cover these items, so each primary source was read here.
No GPU was used and no model was loaded. HIP code was compile-checked only (one object at a time, `-j1`, `MemoryMax=6G`).

## Decisions

| Item | Decision | Why in one line |
| --- | --- | --- |
| A31 GDN chunk 64 | **blocked-on-GPU** (leaning close) | Rule inputs do not exist for the 7900 XT; the one prefill timing is an RTX 3060 CUDA run. One stage-bench run decides. |
| A36 weight-major hit experts | **do not build** (re-check on a deferral-on trace) | MonoMoE's "weight-major" is a tile mapping plus a megakernel, not a weight layout; our hit kernels are already weight-row-major. The one recorded trace gives a ceiling of 0.3 to 2.5% of the profiled wall. |
| 16.11 BFLA (qwen35 engine) | **keep on its branch, opt-in, lossy part not default** | 3.15x is against a materialised-softmax baseline; the sparse part adds 1.58x at 128K and fails the plan's own KL gate. |
| 16.11 XAttention | **do not build** | Attention-only number on a dense 8B model; BFLA already covers the case; nothing to win on Flash-Next. |
| 16.11 A49 (decode vs context) | **blocked-on-GPU** | Spec `a49-decode-context.json` exists; no data for 64K and 128K. |
| 16.11 A50 (window decode attention, F11) | **blocked-on-GPU** (now compiles) | `q36_engine.cu` with `STRATA_ATTN_WINDOW` builds for gfx1100; bit-equality needs `--dump-full-logits` on a GPU. |
| 16.11 A51 Quest / Twilight; KV compression (LServe, ShadowKV, DuoAttention); sliding window | **do not build** | Premise (KV read dominates decode) does not hold here: attention is 3.0% of a Flash-Next window. |

New files (all new, nothing existing edited): `tools/pancake/probes/gdn_chunk64_rule.py`, `tools/pancake/probes/gdn_chunk_model.py`,
their tests `tools/pancake/test_gdn_chunk64_rule.py` and `tools/pancake/test_gdn_chunk_model.py`, and this note.
No engine switch was added; the default path is untouched.

## A31: GDN prefill chunk 64

### Source check

- Gated DeltaNet (2412.06464) and parallel DeltaNet (2406.06484) write the chunk size as a symbol `C`. Neither names 64, neither has a
  chunk-size ablation, and neither measures prefill on a consumer GPU. Both measure training throughput of 1.3B models on one H100
  (Gated DeltaNet Fig. 3: same as DeltaNet, 2-3K tokens/s under Mamba2; parallel DeltaNet Fig. 6), and parallel DeltaNet also shows the
  chunkwise-against-recurrent speed-up of its Triton kernels at batch x length = 16K. The inverse of `I + tril(...)` is "solved efficiently
  using forward substitution" (parallel DeltaNet, eq. 10); no 16x16 blockwise scheme is described.
- Chunk 64 and the fused kkt + solve_tril + recompute_w_u come from the flash-linear-attention code (`chunk_size` default 64, 16 and 32
  accepted), not from the papers. That is a Triton/NVIDIA training kernel.
- The plan's "5-11% of cold prefill" has no source in either paper. It looks derived from the "GDN 12%" of the 3060 timing (below), which is
  all four GDN phases, not the recurrence. The V100 chunk-64 attempt (4-6x slower, `bench/results/2026-10-02-gdn-chunk`) is a different
  kernel (128 threads, 84 KB staging); it neither supports nor refutes the gfx11 design.
- Verdict: the method is real but nothing in the sources transfers a speed-up to gfx1100. It stays a measure-first item.

### What the current code does (read, `src/prefill/kernels.cu` 1803-2175)

`gdn_wmma_prep_kernel` (T and P, once per chunk) then `gdn_wmma_scan_kernel` (48 blocks of 8 waves, one chunk of 16 tokens per loop turn, 2
LDS barriers per chunk, next chunk's loads issued early, `GW_AHEAD = 3` cache warm-up). Per chunk the dependent chain is about 11 WMMAs
(8 for X / QS, then D, o, state) plus three accumulator-to-B conversions. 48 heads on 42 WGPs means 6 WGPs run two heads and 36 run one, so
the kernel time is set by the two-head WGPs. Static expectation (unconfirmed): the two-head WGPs are WMMA-pipe-bound, so rec_out / floor
is about 1 to 2. That would close chunk 64 (+12% WMMAs) and leave only the head-balance (column split) idea.

### The rule and the numbers that exist

Rule (batch 2): build only if (a) `rec_out / WMMA floor >= 2` at 32K and (b) recurrence `>= 6%` of the cold-prefill GPU timeline.

| Input | Value | Applies to the 7900 XT? |
| --- | --- | --- |
| Stage bench `STAGE` / `A31` lines on gfx1100 | none recorded (the A31 line was added 11-10 23:54, never run) | no data |
| WMMA floor, formula of `gdn_stage_bench.cpp` (1664 clocks a chunk, 2 heads on the busiest WGP, about 2.4 GHz) | about 0.36 / 1.42 / 2.84 ms a layer at 4K / 16K / 32K | model |
| Prompt speed, final suites, 7900 XT (`bench/results/2026-11-08-7900xt-final-suites.txt`) | 16K: 2,456 tok/s (6.5 s); 32K: 2,580 tok/s (12.45 s) | yes (meas.) |
| `STRATA_PREFILL_TIMING` line, 41,811 tokens: gdn 11.9% (all four GDN phases), **gdn recurrence 2,515 ms = 6.0%**, conv+gates 1.3%, out proj 4.4% | `bench/prefetch/long-timing.long-timing.engine.log` on `port/rdna3-decode`: GPU 0 is an **RTX 3060, CUDA sm_86**, token-serial recurrence | **no** (CUDA, other card, other kernel) |
| gfx1151, chunked path (`docs/AMD_HIP.md`: 32K +1.4%/+2.1% prompt, recurrence 1.6x) | recurrence about 2.3 to 3.7% of the 32K prompt (derived, rough) | no |
| gfx12 R9700 (comment in `kernels.cu`: chunked +10.1% at 32K, 1.7-2.5x on the layer) | chunked recurrence about 6 to 13% of the old 32K time (derived, rough) | no |

Two plan labels are wrong and should be fixed by whoever edits the plan: `pancake-docs/RESEARCH_AND_FINDINGS.md` ("Prefill path findings
(fast-hip ...)") and MOD_PLAN 16.6 present the 3060 timing as the daily 7900 XT preset (the Grouped-technologies report and item 79 already
say 3060). The README 4C rows for Mamba-2/SSD and Mamba-3 say "Adopted (GDN chunk-64 prefill line)"; chunk 64 is not built.

### What the two parts of the rule mean together (new)

(b) needs `rec_out >= 0.06 x prompt time / 36 GDN layers`: **20.8 ms a layer at 32K and 10.8 ms at 16K**, which is 7.3x and 7.6x the floor.
So (b) holding implies (a) holding, and a stage bench far below 21 ms closes A31 without any timing run. `gdn_chunk64_rule.py` does this
arithmetic (`--prefill-tok-s 2580`). Two caveats on (a): `rec_out_ms` is prep + scan + out norm while the floor is the scan only, so the
printed ratio is a little high; and the clock in the floor is the device's reported clock, not the sustained one.

### What to run when the GPU is free (one command decides)

    HIP_VISIBLE_DEVICES=0 STRATA_GDN_CHUNKED=1 build-hip/hip_gdn_stage_bench 4096 16384 32768 | tee /tmp/a31-stage.txt
    python tools/pancake/probes/gdn_chunk64_rule.py --stage /tmp/a31-stage.txt --prefill-tok-s 2580

Optional: the same prompt with `STRATA_PREFILL_TIMING=1` and `--timing ENGINE_LOG`. If rec/floor is near 1 the scan is WMMA-bound: close
chunk 64 and look at the head balance only. If rec/floor is about 7 or more, find out what bounds it before writing 300 lines of chunk 64.

### Checks run

- `hip_gdn_stage_bench` object (the batch-2 A31 line was never compiled): `make -f CMakeFiles/hip_gdn_stage_bench.dir/build.make
  .../tests/hip/gdn_stage_bench.cpp.o` in `build-hip` (HIP on, gfx1100): **compiles, no warnings**. Not linked, not run.
- Numerical model of the design on the CPU (`gdn_chunk_model.py`, numpy, seeded): the blockwise 16x16 inverse is exact in FP64
  (error 7e-16, same as chunk 16), and with every WMMA operand rounded to FP16 chunk 64 gives y 4.73e-4 / state 3.60e-4 against chunk
  16's 4.72e-4 / 3.49e-4, also with the inverse itself done in FP16 (4.73e-4 / 3.60e-4) and with a weak gate (0.05 scale: 4.83e-4 against
  4.74e-4). So the design is numerically sound for unit-norm keys and beta in (0, 1); near-collinear keys were not tried. This says nothing
  about speed.
- `python tools/pancake/test_gdn_chunk_model.py` (3 tests) and `python tools/pancake/test_gdn_chunk64_rule.py` (7 tests): pass.

## A36: weight-major layout for the VRAM-hit experts

### Source check (MonoMoE, 2609.04244, Amazon AGI, 2026-08-19; body read)

- "Weight-major" is a schedule: SwapAB puts the whole decode-step token tile on the small N axis of the tensor-core MMA (Hopper WGMMA, N = 8
  to 256) and splits CTAs over expert-weight tiles. It does **not** change how weights are stored ("does not launch a transpose kernel or move
  an additional tensor").
- The baseline is vLLM's Triton token-major grouped GEMM, which pads token tiles to M = 16 and reaches 10.6% of peak DRAM bandwidth at
  batch 1. Removing the padding and preprocessing saves 5.3 to 5.7 us a layer (13 to 27% of a 20 to 43 us operator).
- The rest of the gain (up to 1.54x operator, up to 18.7% time per output token) comes from the persistent megakernel: one launch for routing,
  top-k, quantization, both projections, activation and reduction, with warp specialization. There is no ablation of the tile mapping alone.
- Setting: H200, FP8 E4M3 with 128x128 block scales, 256 experts top-8, all experts in HBM, batch 1 to 8, Qwen3.5/GLM-5.2/DeepSeek-V3.1.
  No IQ2/Q2_0, no AMD, no CPU-offload hit/miss split.
- Mismatch with our setting: `expdec_gu_kernel` and `native_down_multi_kernel` already take weight rows as the parallel axis with the
  activations in LDS and q8_1 dots; there are no token tiles to pad. The plan row ("weight-major layout for VRAM-hit experts / Q2_0 down")
  reads the term as a storage layout. citecheck B5 already notes the megakernel link; the mismatch above is in addition.

### Rule run on a recorded trace

`probes/hit_critical.py` needs a per-dispatch rocprofv3 trace with kernel names. The repository has only aggregate stats
(`2026-11-09-7900xt-gpuprof`, `2026-11-09-gpuprof-defer`). The one per-dispatch trace on disk is a raw rocprofv3 buffer,
`/home/dan/clone-of-d/src/Strata/.rocprofv3/281900-281969-kernel_trace.dat` (136 MB, 11-09 08:52, read only, not copied into the repo):
655,360 dispatch records (the buffer cap, so the third request is cut), three decode phases, kernel ids but **no names**. I decoded the
record layout (208-byte stride, start/end ns at 8-byte fields 5 and 6, kernel id at field 10) and named the six kernels the rule needs
by their duration signature against `run_kernel_stats.csv` and by call counts (route 21,714 = combine 21,714; wait_flag_copy, swiglu and
down 23,250 each; gate/up 3c2 + 46d in the 3.1 : 1 ratio of the stats): wait_flag_copy `253` (opens each layer segment), gate/up `3c2`
(type 22) and `46d` (type 16), swiglu `353`, native down `291`, combine_cpu `c3b`. Per layer the order is 253, 3c2 or 46d, 353, 291, c3b.
This is a reverse-engineered mapping, so treat the numbers as indicative.

Result of `hit_critical.analyze` per decode phase (25% faster hit kernels; % of the profiled wall):

| Phase | Layer segments | GPU-bound | Wall | Hit kernels (down) | Wait | Exposed hit (down) | Ceiling at 25% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8,495 | 4,398 (52%) | 5,685 ms | 711 ms (302) | 1,053 ms | 359 ms (151) | **1.58%** |
| 2 | 9,647 | 6,712 (70%) | 5,604 ms | 832 ms (354) | 442 ms | 553 ms (234) | **2.47%** |
| 3 | 5,105 | 621 (12%) | 4,971 ms | 468 ms (202) | 1,994 ms | 59 ms (25) | **0.30%** |

Reading: the rule (>= 2% on both the short and the edit trace) is not met (one of three phases passes). Caveats that pull the other way:
the trace was taken with the profiler (decode 79 against about 126 tok/s without it, so the profiled wall is stretched by launch gaps;
scaling the wall by 1/1.6 would give about 2.5 / 3.9 / 0.5%), and it predates deferral (the wait kernel is combine_cpu, once a layer),
and deferral hides CPU time and should raise the exposed share. Which phase is `short` and which is `edit` is not recorded. The ceiling is
an upper bound (CPU margin in GPU-bound layers unknown) and assumes a 25% faster hit kernel exists; nothing in MonoMoE supports that for
our kernels. The Q2_0 down half alone is 0.66 / 1.04 / 0.13%.

Decision: **do not build.** Re-run `hit_critical.py` on `gpu_profile.py ... --case short` and `--case edit` with deferral on when the GPU is
free (the plan's reopen check); build only if both reach 2% and a concrete way to a 25% faster hit kernel exists (the A37 bank-conflict
counters are the nearer lever).

## 16.11 Attention, KV and long-context decode

### BFLA on `feat/bfla-hip` against its source (2605.12193 v2, CityU, 2026-10-26)

- The branch follows the paper's Ratio-K rule (eta 1/3, K_min 2048, K_max 32768, blocks of 64, tiles of 128 rows, 2 sink + 4 local blocks,
  tail kept dense), with FP32 key states and no Hadamard rotation or FP8 (gfx11 has no FP8). That matches the paper's method; the paper's
  public code differs from the paper (cumulative-mass rule) and the branch documents it.
- What the paper measured (H200, FA3 or Triton dense baselines): up to about 4x TTFT, but that is Gemma4-E4B (dense Triton, 128K: 3.995x).
  On Qwen3.6-27B at 128K it is **1.244x** (64K 1.112x, 32K 1.041x); on Qwen3.8-Flash-Next 1.27x at 128K, and the paper's own QSA
  baseline is faster than BFLA there (1.319x at 128K; Terminal Bench 1.391x against 1.342x). BFLA's edge on Flash-Next is accuracy, not speed.
- What we measured (7900 XT, Qwen3.6-35B-A3B UD-Q4_K_XL, noisy, PANCAKE_STATUS P2b): 128K prompt 386.9 off, 771.2 dense, 1,219.8 sparse tok/s;
  64K 1,058.2 dense, 1,284.1 sparse. The "3.15x" is sparse against **off**, whose attention is a materialised-FP32-score hipBLAS path.
  The fused dense kernel alone gives 2.0x and is exact (max diff 3.7e-5 to 1.3e-4 against FP64 reference). The part due to sparsity is
  1,219.8 / 771.2 = **1.58x at 128K** and 1,284.1 / 1,058.2 = **1.21x at 64K**. Those are consistent with the single-layer kernel data
  (128K, 10 layers: 283 s, 96 s dense, 33 s sparse; e2e 338 s, 170 s, 107 s, so non-attention time is about 74 s either way) and
  higher than the paper's 1.24x on a 27B at 128K, plausible for 10 attention layers of head dim 256 on a card with slower other work.
  The 3.15x must not be set against the paper's 1.34-1.44x: other model, other baseline.
- Quality gate: the recall probe says top-1 48/48 and mean KL 0.0014 (64K) / 0.0045 (128K). MOD_PLAN's own rule for a lossy proposal
  (line 2115) is mean KL <= 1e-3 and top-1 >= 99%. Both KL values are above it, and 48 tokens cannot show 99% (all-agree on 48 gives a
  95% one-sided lower bound of 93.9%; about 300 tokens are needed).
- Compile check of the integration (new): `git merge-tree` of `feat/bfla-hip` into this tree conflicts only in `MOD_PLAN.md`;
  `CMakeLists.txt`, `prefill.cpp` and `q36_engine.cu` merge cleanly. From that merged tree, `bfla.cu`, `bfla_test.cu` and `q36_engine.cu`
  (with the BFLA hooks) each **compile for gfx1100 with no errors** (scratch build, not committed). `bfla_test` was not run (GPU).
- Decision: Flash-Next keeps QSA (closed 11-09, unchanged). For the qwen35 engine merge the **dense mode first** (`STRATA_BFLA=dense`, exact,
  2.0x at 128K); keep the sparse selection opt-in and lossy until it passes the KL gate on enough tokens (300+), at 64K and 128K.

### XAttention (2503.16428; the id question is settled in `batch3/citecheck.md`: XAttention is 2503.16428, 2502.17421 is LongSpec; the wrong
ids are `pancake-docs/README.md` line 180 and `RESEARCH_AND_FINDINGS.md` line 208, not the root README)

Antidiagonal block scoring (stride 8 or 16, threshold tau 0.9 with per-head tuning). Measured on Llama-3.1-8B (all 32 layers dense),
Qwen2-VL-7B and HunyuanVideo on NVIDIA. "Up to 13.5x" is the prefill attention operator at 256K (9.8x at the next point), not time to first
token. For us: Flash-Next prompt attention is QSA already (2,051 cells a row); on the qwen35 engine BFLA already keeps 31 to 33% of blocks, and
attention at 128K is about 33 of 107 s, so even halving the sparse attention again would give at most about 15% of that prompt (less at
64K). **Do not build.** (Static estimate from the measured kernel times.)

### A49 / A50 / A51

- A49: spec `bench/specs/a49-decode-context.json` and the `longdec32k/64k/128k` cases exist; no run. Existing evidence: decode after a
  13.8K prompt is 155.9 tok/s against 156.3 after a 2K prompt (`2026-11-09-decode-curve`), so no attention effect at 14K. For Flash-Next the
  QSA cap makes attention flat above about 2K cells (0.74 ms of a 24.8 ms window, 3.0%, latency-bound). For Qwen3.6-35B-A3B the KV read at
  128K is about 1.34 GB with int8 KV (10 layers x 2 KV heads x 256 x 2 x 1 B x 131,072), about 1.7 ms a step at 800 GB/s: the share depends
  on a decode step time that was not measured at 128K. Blocked-on-GPU.
- A50 (LongSpec idea, F11 on the qwen35 engine): LongSpec (2502.17421, ACL 2025) is a trained long-context draft model (constant-size
  draft KV, Anchor-Offset Indices) plus Hybrid Tree Attention (prefix through FlashAttention, tree through a masked kernel, merged by
  log-sum-exp). Its "up to 3.26x" is end-to-end speculative-decoding speed over Flash-Attention-2 baselines (Vicuna, LongChat,
  LLaMA-3.1-8B, QwQ-32B) and the attention kernel alone is "about 75% less latency" than the HuggingFace implementation, not than a
  split-K kernel like ours. Only the kernel idea applies (no draft model is trained here). The plan's 48.6 to 88.9 tok/s at 200K is a fork
  figure on an RTX 5080, not in the paper (citecheck agrees). The branch's `k_attn_window` / `STRATA_ATTN_WINDOW=1` was never built: `q36_engine.cu` now **compiles for gfx1100 with no
  errors** (warnings are old: sign-compare, unused `D`, unused `k_add` / `k_gate_rows`). Exactness ("bit for bit" against `k_attn_part`)
  needs `--dump-full-logits` on and off on a GPU; spec `18-f11-attn-window.json`.
- A51 (Quest 2406.10774, Twilight 2502.02770): both were measured on dense MHA 7-8B models (LongChat-7B, Llama-2/3.1-8B) on an RTX 4090 /
  A100, where the KV cache (16 GB in Quest's example) is over half of decode latency. Here the KV is 1.5 GiB at 131K, the experts set the
  step time, and Flash-Next's QSA already is a query-aware top-k. The plan's parking rule (attention > 15% of decode, from A49) is right
  and not met on any measured case. **Do not build.**
- KV compression and sliding window (README 4C: LServe 2502.14866, ShadowKV 2410.21465, DuoAttention 2410.10819): LServe (prefill up to
  2.9x, decode 1.3-2.1x over vLLM), ShadowKV (A100, up to 3.04x throughput through 6x larger batches, low-rank keys plus offloaded values) and
  DuoAttention (decode up to 2.18x MHA / 1.50x GQA, an optimisation pass with synthetic data to find retrieval heads) all target dense
  models, mostly batched, on A100. Flash-Next has 12 attention layers that are already sparse and an int8 KV with 32K of 131K cells
  resident; the Qwen3.6 engine has 10 layers with 2 KV heads. Nothing here is KV-capacity or KV-bandwidth bound at batch 1.
  **Do not build.** (Abstract-level read for these three; not re-derived.)

## Tests run

| Check | Result |
| --- | --- |
| `python tools/pancake/test_hit_critical.py` (A36 tool, 3 cases) | pass |
| `python tools/pancake/test_gdn_chunk64_rule.py` (new, 7 cases) | pass |
| `python tools/pancake/test_gdn_chunk_model.py` (new, 3 cases, about 1 s) | pass |
| HIP configure `-DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1100` (Unix Makefiles) | ok |
| `hip_gdn_stage_bench` object (A31 line) | compiles, no warnings |
| `src/qwen35/q36_engine.cu` object, this tree (F11 `STRATA_ATTN_WINDOW`) | compiles, 0 errors, old warnings |
| `bfla.cu`, `bfla_test.cu`, `q36_engine.cu` from `merge-tree(dev/phase16-academic, feat/bfla-hip)` | each compiles, 0 errors (scratch) |
| `hit_critical.analyze` on the decoded 11-09 rocprofv3 buffer | numbers above (names matched by signature) |
| Not run: `hip_gdn_stage_bench`, `hip_gdn_chunked`, `bfla_test`, anything on the GPU; no link of the full HIP targets | GPU rules |

## Risk and next

- A31: low risk (nothing changed). The cost of a wrong "build" would be about 300 lines of lane-layout code for a gain the 3060 data
  cannot support; the stage bench is a 2-minute run.
- A36: the trace mapping is reverse-engineered; the verdict should not rest on it. Re-run on a named trace with deferral on.
- BFLA: merging the branch last is clean in code; only the plan text conflicts. The first thing to land is the exact dense mode.
- Plan text to fix (not edited here): A31 "5-11%" (no source; 12% is the 3060's whole-GDN share), the "fast-hip" label on the 3060 timing,
  README 4C "Adopted" on the chunk-64 line, A36's "layout" wording, BFLA "3.15x" without its baseline.
- Next on a free GPU, in order: `hip_gdn_stage_bench` (A31), two `gpu_profile.py` traces with deferral (A36 re-check), `a49-decode-context`
  then `18-f11-attn-window` (A49/A50), `bfla_test` and the BFLA recall on 300+ tokens (BFLA gate).
