# Strata (github.com/Niko1221/Strata): distinctive features and proposed improvements worth porting to a llama.cpp server (Avalon)

Scope and method. I read the local checkout /home/dan/Strata (HEAD 7695859, 2026-10-11) and fetched upstream main into a separate ref (`refs/remotes/upstream-research/main`, head 146df3c4, 2026-11-07). The working tree and the running services were not touched. Upstream has **4,081 commits** since our checkout, with releases up to v0.2.12 and engine 0.1.58 (commit 7be42963). I also used `gh` to read the open PRs and issues as of 2026-11-08. Links: D = docs/DETAILS.md on main (https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md), A = docs/AMD_HIP.md on main (https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP.md). Commit links take the form https://github.com/Niko1221/Strata/commit/<hash>.

Baseline for comparison:
- **Avalon on our RX 7900 XT (20 GB):** about 100 t/s decode on short prompts and about 1,800 t/s prefill at 16-32K.
- **Strata IQ2_XS on an RX 7900 XTX (24 GB, engine 0.1.57), from a community report:** prompt 1,051/1,385/1,381 t/s at fresh 4K/16K/32K and chat decode 80.3 t/s ([PR #5415](https://github.com/Niko1221/Strata/pull/5415)).
- **Gyro-M TQ2_0 on the same card:** 1,184/1,566/1,549 t/s prompt and 75.0 t/s decode, with the power cap raised to 321 W ([PR #5584](https://github.com/Niko1221/Strata/pull/5584)).

So even current upstream Strata on a bigger RDNA3 card does not beat Avalon outright. The value lies in specific techniques, not in switching back.

## Q1. What does Strata do that llama.cpp/Avalon does not? (memory tiers, PLE, quantization, scheduling)

### Takeaway
Strata's core difference is a **dynamic** memory tier system:
- an adaptive VRAM expert cache, with the CPU computing missed experts in place at the same time;
- a tuned PCIe share for the remaining misses;
- router-predicted prefetch of experts from the page cache or NVMe;
- the PLE (n-gram) table handled as its own tier;
- speculation decoding that is aware of cost.

llama.cpp's expert placement is static (`-ot` / `--n-cpu-moe`). The pieces that port cheaply to llama.cpp are:
- the mmap/readahead hints;
- a static "resident-expert router bias";
- the PLE-in-RAM policy;
- the tuning tables;
- the speculation-window policies.

### Cited Findings
**Tiering model**
- GPU holds attention/DeltaNet, routers, shared experts, the output head, the MTP layer and the KV cache (beyond 64K only the most-read part; the rest streams from RAM). The rest of VRAM is an **expert cache of the most-used experts that adapts to the conversation**. RAM holds all 24,576 experts pinned, and the CPU computes misses in place, concurrently with the GPU (AVX-512/AVX2, ggml i-quant kernels). The 28.8 GB n-gram/PLE table lives on SSD, read a few rows per token, unbuffered (`--ple-io direct`) — [D "How it works"](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#how-it-works); [HOW_IT_WORKS.md](https://github.com/Niko1221/Strata/blob/main/docs/HOW_IT_WORKS.md)
- "Every extra GB holds about 700 more experts." An expert profile ranks expert pairs, and `tools/make_profile.py` builds one from your own `--dump-routing` trace — [D "Speed (measured)"](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md)

**PCIe share of missed experts** (default on CUDA)
- Some missed experts are copied to the GPU over PCIe instead of being computed on the CPU. The share starts from a probe and is refined from the CPU pool's time per miss: `L / (L + k R)`, rounded to 0.05, clamped to 0.10-0.95, changed at most 3 times. k = 1.85 at a pool rate of 30 GB/s, falling to 0.8 at 12 GB/s — [D "Speed settings / On by default"](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#speed-settings)
- On HIP outside gfx12 the in-window refinement (PcieAuto) is opt-in, because "a second gfx1100 report saw it never engage" — [commit b317c23f](https://github.com/Niko1221/Strata/commit/b317c23f) (#2224)

**Route tail skip** (`STRATA_ROUTE_TAIL_SKIP=7`, default on CUDA and gfx1200/1201, not gfx11)
- A missed expert that every token of a verify window routes at rank 7 or lower is neither copied nor computed. It is not applied when 90% or more of the experts are cached. Answers differ slightly — [D](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#speed-settings)

**Router bias toward resident experts** (quality trade-off)
- `STRATA_ROUTE_PRIOR=1.5` on an RTX 3060 12 GB with IQ3_XXS: story +15%, code +37% decode. But agent tasks averaged 4.5/10 with it against 5.6/10 without. It was reverted to opt-in on 2026-10-29 because of repetition loops in long agent sessions (#4402) — [D](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md)
- Open [PR #5596](https://github.com/Niko1221/Strata/pull/5596) (DeepSeek engine, `STRATA_DS4_CACHE_ROUTE=<ram>,<vram>`, 2026-11-08) adds a selection-only bonus to the router bias for experts held in VRAM or RAM:
  - speed: decode 7.0 -> 10.9 t/s and 7 agent turns in 55 s instead of 109 s on a 12 GB laptop;
  - quality: top-1 agreement 93.2-95.8% and PPL +0.5%;
  - the mixing weights are untouched, and the bias copy is rebuilt every 16 decode steps (61 KB upload).

**PLE table in RAM**
- `STRATA_PLE_AUTO_RAM` (default on where RAM fits) keeps the n-gram table resident. On an R9700 with IQ3_S at an 8K prompt, the PLE stage went 845 -> 132 ms and the prompt 2,125 -> 2,611 t/s (+22.5%), with identical answers — [commit 3b34b6c7](https://github.com/Niko1221/Strata/commit/3b34b6c7); [A "Speed switches"](https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP.md#speed-switches)
- Setup picks `--ple-io ram` only when RAM covers the model's need plus the table plus 16 GB (Q2_0 needs 93 GB, IQ3_S 107 GB). Otherwise it keeps the direct SSD reads — [D](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md)
- The PLE reader has also been optimised: a small batch is folded in a stack buffer instead of a hash table — [commit 1ed923f5](https://github.com/Niko1221/Strata/commit/1ed923f5). The PLE table can be IQ4_NL, Q4_0, Q5_0, Q5_1, Q8_0, FP8 or BF16 — [D](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#how-it-works)

**Linux file tier** (experts read from mmapped GGUF when RAM is short)
- On by default:
  - zero-copy compute from the page cache (`STRATA_COLD_OVERLAP=4`);
  - I/O-thread whole-blob `pread` prefetch of the current layer's uncached experts **and the router-predicted experts of the next layers** (`STRATA_IO_PREFETCH`, 8 threads, depth 2);
  - a decode gate that runs the prefetch only while it pays.
- The prompt chunk is planned from the page cache size, drive speed and PCIe link (`STRATA_PREFILL_HOST_PLAN`) — [D "Linux file tier"](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#linux-file-tier)
- Open [PR #5561](https://github.com/Niko1221/Strata/pull/5561) (`STRATA_EXPERTS_SEQ_READAHEAD=1`) adds `madvise(MADV_SEQUENTIAL)` on the mapped expert files right after mmap:
  - five follow-up turns took 23.5-28.4 s instead of 65.2/76.2 s;
  - an 800-token answer decoded at 42.0-52.2 instead of 32.8/43.9 t/s;
  - answers were byte-identical (RX 6900 XT, 32 GB RAM, Gyro-M 29.9 GiB experts).
  - The author's explanation: without the hint, the kernel's `MMAP_LOTSAMISS` logic stops read-around and every fault reads 4 KiB (up to 1.35 M major faults per turn).
- An `STRATA_FILE_TIER_HONEST` re-check counts only the page cache the group can really have — [commit 0a0e4463](https://github.com/Niko1221/Strata/commit/0a0e4463)

**Prefill**
- Prefill streams experts over PCIe in chunks of up to 8,192 tokens by default (32,768 opt-in, and the default on gfx12). The ring is sized in bytes — [D "How it works"](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#how-it-works)
- On gfx12, 32K chunks gave +12% at a 20K prompt (UD-Q4_K_XL +26%) against 16K chunks — [D "On by default"](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#speed-settings)

**gfx1100 hipBLASLt tuning tables** (shipped: `tools/hip/gfx1100-hipblaslt-*.txt`, including 100401 for ROCm 10.0.0)

| Card / setup | Prompt | Before -> after (t/s) | Gain |
| --- | --- | --- | --- |
| 7900 XTX, IQ2_XS | 3,151 tokens | 711 -> 938 | +31.9% |
| 7900 XTX, Swift IQ3_XXS | 6.5K | 857 -> 1,524 | +78% |
| 7900 XTX, engine 0.1.38 lineage | 131K | 926 -> 1,687 | +82% |

— [A "Tuning table"](https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP.md#tuning-table). Our own config uses `gfx1100-hipblaslt-100200.txt` (~/Strata/strata-iq2_xs.json).

**Own WMMA dense GEMM on RDNA3**
- The commit message reads "4.97 vs 20-40 TFLOPS at our shapes" (stated as hipBLAS vs own kernel), and the engine prefers its own `src/prefill/wmma_gemm.cu` over hipBLAS. The commit is in the DeepSeek engine context — [commit cca3dd2e](https://github.com/Niko1221/Strata/commit/cca3dd2e)

**KVA (approximate late-layer prefill, opt-in `STRATA_KVA=1`)**
- How it works: layers 36-47 build only their recurrent state, GDN state and KV/indexer pages, skipping attention output and MoE. A 2048-token exact tail follows.
- Speed:
  - P100: +12% at 14K and +20% at 30K prompts;
  - R9700: 27K prompt 2,411 -> 3,086 t/s (+28%), but **decode -11%** (94.7 -> 84.3, lower MTP acceptance).
- Quality: an earlier split-24 variant lost facts and failed agent runs. Split 36 matched exact on extraction (585/606) — [docs/KVA.md](https://github.com/Niko1221/Strata/blob/main/docs/KVA.md)

**Quant formats** (llama.cpp already has most of these)
- Strata reads GSQ-RCO IQ2/IQ3 packs, a native Q2_0, Unsloth UD K-quants and the v3.1 sets (UD-IQ1_M to UD-IQ4_XS-v31, added 2026-11-05) — [commit 9ed3c778](https://github.com/Niko1221/Strata/commit/9ed3c778)
- Gyro trellis types TQ2_T/TQK6/TQK7 in a Hadamard-rotated basis: "about IQ3_XXS quality with about 20% less GPU memory" — [docs/GYRO.md](https://github.com/Niko1221/Strata/blob/main/docs/GYRO.md)
- Also NVFP4 ([docs/NVFP4.md](https://github.com/Niko1221/Strata/blob/main/docs/NVFP4.md)).

**Experimental "speed projection"**
- A control vector (a refusal-direction projection on layers 4-44) costing 0.2-0.4% per token. It is not a real speed feature and it removes safety behaviour — [D](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#experimental-speed-projection-experimental-off-by-default). Not recommended.

### Inferences
**Static resident-expert router bias (worth an A/B)**
- In llama.cpp, placement is static. A per-expert router bias toward the GPU-resident experts would therefore be a **constant** tensor added before top-k. It is cheap to implement, a small patch to the MoE graph build in `llama-graph.cpp`, adding a bias only to the selection logits as DeepSeek's `exp_probs_b` does.
- Caveat: Strata's own data shows agent-quality losses (#4402). Treat it as an opt-in to A/B, not a default.
- Inferred benefit: Avalon on 20 GB VRAM + 57 GB RAM keeps most experts in RAM/CPU, so steering near-ties to GPU experts could raise decode notably, but at a measurable KL cost.

**MADV_SEQUENTIAL / WILLNEED hints (easiest high-value port)**
- Avalon's GGUF (~89 GiB) exceeds RAM (57 GB) plus VRAM headroom, so some expert pages fault from NVMe. PR #5561's one-line `madvise(MADV_SEQUENTIAL)` on the expert mmap, plus Strata's `posix_fadvise WILLNEED` start-up read-ahead, are the easiest ports here: a few lines in `llama-mmap.cpp`.
- The fault storm PR #5561 describes (4 KiB faults at a few hundred MB/s) is plausibly hitting Avalon too. To check: measure major faults per turn with `perf stat -e major-faults` or /proc/<pid>/stat.

**Router-predicted next-layer expert prefetch**
- Issue `pread`/WILLNEED for the experts the next layer's router is likely to pick. This is a moderate port: it needs a hook after the router top-k and an I/O thread pool. It matters only when experts spill out of the page cache.

**PLE in RAM: probably not for us**
- With 57 GB RAM and an ~89 GiB GGUF it is unlikely to fit next to the CPU-resident experts. It is only worth it if Avalon's PLE is IQ4_NL (~27 GiB for UD-IQ4_XS) and expert residency is traded for it. The bigger lesson is reading PLE rows with direct I/O and a dedicated reader thread, overlapped with compute (Strata prefetches the next chunk's PLE rows on a thread; Metal commit [687a7d79](https://github.com/Niko1221/Strata/commit/687a7d79)).

**hipBLASLt tuning tables**
- Relevant only if Avalon's prefill uses hipBLAS/hipBLASLt for dequantized F16 GEMMs (llama.cpp's ROCm path uses MMQ for most quant types and rocBLAS/hipBLASLt for large F16 GEMMs). Easy to try: set `HIPBLASLT_TUNING_OVERRIDE_FILE` to a table built on this card for this ROCm.

**Dynamic expert cache and CPU/GPU concurrent miss compute**
- This is the hardest port (the architecture of `ggml_mul_mat_id` across backends). llama.cpp's scheduler already runs CPU experts and GPU layers, but not as a per-expert cache that swaps.

### Gaps
- I did not verify Avalon's own source for which of these it already has (n-cpu-moe split, PLE handling, mmap hints).
- No Strata measurement exists for router-bias or file-tier features on gfx1100 specifically.
- I did not read the Strata paper (docs/paper/Strata-Paper.pdf).

## Q2. What is proposed or in progress, and what changed upstream since 2026-10-11?

### Takeaway
Upstream moved very fast (4,081 commits in 4 weeks). The main themes:
- a DFlash2 diffusion drafter as an alternative to MTP;
- fused int8/F16 matrix-core prefill kernels per architecture, with gfx12 and gfx1151 favoured and gfx1100 mostly left opt-in;
- KVA approximate prefill;
- a large hardening of the tool-call parser;
- new config keys for agent robustness.

The open PRs add thinking-skip for short answers, in-place retune with cache routing, MADV_SEQUENTIAL, a cache-route router bonus, and a gfx1100 int8-WMMA attention study.

### Cited Findings
**DFlash2 drafter**
- What it is: a tuned copy of incoai/Qwen3.8-Flash-Next-DFlash2 (Apache-2.0), with an optional Q4_0 selector that saves 174 MiB — [D "Credits"](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#credits-and-licenses)
- Measurements:
  - M5 Max UD-Q4_K_XL: +14.3% over MTP spec 2 — [commit 66b346c2](https://github.com/Niko1221/Strata/commit/66b346c2)
  - gfx1201: +7.2% short, +1.35% long — [commit 29e8a47a](https://github.com/Niko1221/Strata/commit/29e8a47a)
  - DGX Spark: slower than MTP (37 vs 45 t/s) — [docs/DGX_SPARK_LINUX.md](https://github.com/Niko1221/Strata/blob/main/docs/DGX_SPARK_LINUX.md)
- Confidence threshold: scaled by how many expert slots remain (9000+ slots: 0.35; 3000+: 0.7; below: 0.8) — [commit 29e8a47a](https://github.com/Niko1221/Strata/commit/29e8a47a)
- An opt-in calibrated verify-window gate replaces the fixed `--draft-conf` (`STRATA_DF2_GATE=1`, per-request survival by position x confidence bucket against the cost of one more row) — [commit b97433df](https://github.com/Niko1221/Strata/commit/b97433df)
- HIP gfx1201 auto went back to MTP "until the DFlash2 prompt stall is fixed" — [commit aba4ddac](https://github.com/Niko1221/Strata/commit/aba4ddac)

**Longer prompt-lookup windows**
- `--lookup-max-t` up to 16 (default 12 on 8-row CUDA builds) gave +11 to +14% decode on file rewrites and agent edit turns (RTX A6000 IQ3_S) — [D "Opt-in switches"](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#opt-in-switches)

**gfx1100 work since checkout**

| Change | Status | Measured effect | Source |
| --- | --- | --- | --- |
| Exact long-prompt QSA scorer `STRATA_SCORES_THREAD=1` | opt-in on gfx1100 | +10-13% prompt at ~131K and +22% at 240K on a 7900 XTX; no gain on cached follow-ups or coding | [commit e5d91bf2](https://github.com/Niko1221/Strata/commit/e5d91bf2); [A](https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP.md#exact-long-prompt-qsa-scorer) |
| Chunked GDN on WMMA `STRATA_GDN_CHUNKED=1` | opt-in | only +1.5-2% (measured on gfx1151) | [commit 1b1bd4b4](https://github.com/Niko1221/Strata/commit/1b1bd4b4); [A](https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP.md#opt-in-chunked-prompt-gdn-on-gfx11) |
| K-quant / other formats on the gfx11 F16 matrix cores (`STRATA_PF_F16_KQ`, `STRATA_PF_F16`) | opt-in | not stated | [commit df218bd8](https://github.com/Niko1221/Strata/commit/df218bd8), [commit bd5cc1f2](https://github.com/Niko1221/Strata/commit/bd5cc1f2) |
| gr_upmix with 256 tokens per block | merged | halves w_up staging per token | [commit 09dc5c52](https://github.com/Niko1221/Strata/commit/09dc5c52) |
| gfx11 WMMA expert kernels | default on the 780M iGPU | "decode x1.53" there | [commit 34458600](https://github.com/Niko1221/Strata/commit/34458600) |

- A gfx11 WMMA prompt-attention path exists but is opt-in (`STRATA_HIP_WMMA=1`) and not bitwise — [A "Speed switches"](https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP.md#speed-switches)

**Open gfx1100 measurement reports**
- [PR #5458](https://github.com/Niko1221/Strata/pull/5458) (RX 7900 XTX, IQ3_S, 94K deep prefill):
  - gains: `STRATA_PA_FAST=1` +5.43%; `STRATA_PF_FUSED=1` + `STRATA_PREFILL_RING=512` +9.08%; combined +13.82%;
  - decode: `STRATA_TSUM=1` -0.91% decode time (bitwise);
  - negative results: expert layout modes 2/4 3.0x slower, mode 8 2.1x slower, `STRATA_EXPERT_V2=1` 1.31x slower, `STRATA_EXPDEC=1` 1.09x slower, the mmvq upstream multi-column layout 17-22% slower at ncols 8.
- [Issue #5459](https://github.com/Niko1221/Strata/issues/5459) proposes a gfx1100 defaults row containing only TSUM (open).
- [PR #5535](https://github.com/Niko1221/Strata/pull/5535) (open, docs-only): an int8 WMMA (`V_WMMA_I32_16X16X16_IU8`) prompt-attention kernel for gfx1100 with exact i32 accumulation and an fp32 online softmax. Kernel time is -32.1% against the two-image f16 WMMA class, but end-to-end 94K prefill is **0.995x (no win)** against the single-image fast config.

**Other open engine PRs** (all as of 2026-11-07/08)
- [PR #5512](https://github.com/Niko1221/Strata/pull/5512): `--thinking-policy on` skips thinking for short direct-answer prompts. At max_tokens=24, 0/6 -> 5/6 answered (median wall 0.72 -> 0.50 s). An alternative draft, PR #5189 `short_answer_room`, reserves half the cap instead.
- [PR #5516](https://github.com/Niko1221/Strata/pull/5516): `POST /v1/tune` changes runtime knobs (prompt_cache, spec_min_p, pcie_frac, prefill_chunk, adapt_*) without the 93-100 s reload. `--conv-route` turns the conversation cache off for single-shot traffic:
  - short single-shot TTFT 0.522 -> 0.304 s (-42%);
  - with the cache off, prefill +10.1% on 3.5K code and +4.0% on prose.
- [PR #5561](https://github.com/Niko1221/Strata/pull/5561) (MADV_SEQUENTIAL, see Q1) and [PR #5596](https://github.com/Niko1221/Strata/pull/5596) (cache-route router bonus, see Q1).
- PR #5566, opt-in cold store for the expert arena (#5391); PR #5593, `STRATA_PROMPT_DIRECT=1` without a RAM budget; PR #5580, `continue_final_message` (llama.cpp / HF semantics). All from `gh pr list -R Niko1221/Strata`, 2026-11-08.

**Other merged speed items since checkout**
- **Fetch-admission:** the PCIe share's fetched misses are admitted into the GPU cache (`STRATA_ADAPT_FETCH`, opt-in) — [commit 24c6c2f6](https://github.com/Niko1221/Strata/commit/24c6c2f6)
- **QSA prompt selection:** exact two-pass selection (opt-in) — [commit 386f9959](https://github.com/Niko1221/Strata/commit/386f9959)
- **Strix Halo F16 fused K-quant prompt experts:** 8K 885 -> 1,387 t/s and 4K 611 -> 1,316 t/s on gfx1151 — [commit 4dbaf703](https://github.com/Niko1221/Strata/commit/4dbaf703)
- **V2K grouped decode experts:** -4.0% window time on gfx1151 — [commit e431a4a8](https://github.com/Niko1221/Strata/commit/e431a4a8)
- **Equal-size prompt chunks** (`STRATA_PREFILL_EQUAL`) and pipelined windows with batch (`STRATA_PW_WITH_BATCH`, +14% lone-request code on 2x A6000) — [D "Opt-in switches"](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#opt-in-switches)
- **Overlapped KV upload during prefill** (`STRATA_KV_PREFETCH=1`, opt-in): the next attention layer's RAM-resident KV is uploaded on a separate stream — [docs/KV_PREFETCH.md](https://github.com/Niko1221/Strata/blob/main/docs/KV_PREFETCH.md) (in our checkout; no number stated in the doc head)

**Open performance and regression issues relevant to us**
- [#5490](https://github.com/Niko1221/Strata/issues/5490): decode throughput drops after several requests, and a reload restores 100+ t/s.
- [#5327](https://github.com/Niko1221/Strata/issues/5327): 1M-context decode is 2.4x slower since 0.1.56.
- [#5441](https://github.com/Niko1221/Strata/issues/5441): RX 9070 XT, 50 -> 14 t/s after an update because of excessive SSD reads.
- [#5134](https://github.com/Niko1221/Strata/issues/5134): gfx1100 illegal memory access in the MTP draft layer K/V after very long contexts.
- [#5372](https://github.com/Niko1221/Strata/issues/5372): GPU spins while idle.

### Inferences
- **DFlash2 port: hard, not first.** It needs a new drafter architecture in llama.cpp. Its win over MTP is backend-dependent (+1-14%, negative on Spark). Our MTP + n-gram setup already gives 169-181 t/s on code edits.
- **Longer n-gram windows: cheap.** Raising the n-gram draft window (`--draft-max` 12-16 for the ngram-simple type) is a cheap test.
- **Calibrated per-position acceptance gating: moderate.** DF2_GATE and the cost-aware prompt lookup are a moderate server-side port into llama.cpp's speculative loop.
- **gfx1100 kernels: little to take.** Strata itself found that most kernel experiments are neutral or negative on 7900-class cards (PR #5458 / #5535). The exception is the long-context QSA scorer: +10-22% only beyond 131K, and only if Avalon implements QSA scoring similarly.
- **`/v1/tune` and conv-route are easy server-side ports.** llama-server already has `cache_prompt` per request. Turning it off by traffic shape matters less for llama.cpp, where caching is cheaper than Strata's "parking".

### Gaps
- I did not check the 4,081 commits one by one. Themes come from keyword filtering (`git log HEAD..upstream-research/main` grepped for gfx11, DFlash, tool, PLE and similar).
- No public roadmap or TODO doc was found. The closest are docs/work/IMPLEMENTATION_STATUS.md (+1,260 lines) and the paper, neither read in full.
- GitHub Discussions were not checked.

## Q3. Which give measured speed gains on AMD RDNA3, or for MoE models larger than VRAM?

### Takeaway
On RDNA3 the measured, defensible gains are:
- the hipBLASLt tuning table (+32% to +82% prompt);
- PA_FAST plus fused prefill with a bigger ring (+13.8% deep prefill, approximate);
- the exact QSA scorer (+10-22% beyond 131K);
- TSUM (about 1% decode).

For MoE larger than VRAM (and larger than RAM), the gains are in memory and I/O policy:
- MADV_SEQUENTIAL (follow-up turns 2.5-3x faster, decode +20-28%, on RDNA2 with 32 GB RAM);
- PLE in RAM (+22.5% prompt at 8K, RDNA4);
- router residency bias (+15-56% decode, with a quality cost);
- PCIe share and tail skip (CUDA defaults).

### Cited Findings
**RDNA3 / gfx1100**

| Feature | Measured on | Effect | Source |
| --- | --- | --- | --- |
| hipBLASLt table | 7900 XTX | +31.9% (3.1K IQ2_XS), +78% (6.5K), +82% (131K) | [A](https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP.md#tuning-table) |
| PA_FAST + PF_FUSED + RING=512 | 7900 XTX, 94K deep prefill | +13.8% combined (not bitwise) | [PR #5458](https://github.com/Niko1221/Strata/pull/5458) |
| Exact QSA scorer | 7900 XTX | +10-13% at 131K, +22% at 240K | [A](https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP.md#exact-long-prompt-qsa-scorer) |
| TSUM | 7900 XTX | -0.91% decode time (bitwise) | [Issue #5459](https://github.com/Niko1221/Strata/issues/5459) |
| int8 WMMA attention | 7900 XTX | end-to-end 0.995x (no gain) | [PR #5535](https://github.com/Niko1221/Strata/pull/5535) |
| Chunked GDN | gfx1151 | +1.5-2% | [A](https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP.md#opt-in-chunked-prompt-gdn-on-gfx11) |

**Larger-than-VRAM MoE (any GPU)**

| Feature | Measured on | Effect | Source |
| --- | --- | --- | --- |
| MADV_SEQUENTIAL | RX 6900 XT, 32 GB RAM | turns 65-76 s -> 23-28 s; decode 33-44 -> 42-52 t/s | [PR #5561](https://github.com/Niko1221/Strata/pull/5561) |
| PLE resident in RAM | R9700 | 8K prompt +22.5% | [commit 3b34b6c7](https://github.com/Niko1221/Strata/commit/3b34b6c7) |
| ROUTE_PRIOR=1.5 | RTX 3060 12 GB | story +15%, code +37% decode; agent pass rate 5.6 -> 4.5/10 | [D](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md) |
| CACHE_ROUTE 0.2,0.3 | 12 GB laptop, DeepSeek | 7.0 -> 10.9 t/s (+56%); top-1 93.2% | [PR #5596](https://github.com/Niko1221/Strata/pull/5596) |
| `STRATA_ADAPT_OVERLAP=1` (adaptive-tier copies overlapped with the next window) | A4000, P100 / RTX 5070 Windows | +4 to +8% / loses | [D "Opt-in switches"](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#opt-in-switches) |
| `STRATA_PCIE_BALANCE=1` | RTX A4000 | +5.6% (others slightly worse) | [D](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#opt-in-switches) |
| `STRATA_DISJOINT_ADAPT=1` (two GPUs: no duplicate expert slots, ~1,900-2,000 duplicates seen) | multi-GPU only | not relevant to us | [docs/DISJOINT_EXPERT_CACHE.md](https://github.com/Niko1221/Strata/blob/main/docs/DISJOINT_EXPERT_CACHE.md) |

### Inferences
**Candidates for Avalon on RX 7900 XT + 57 GB RAM, in order of expected value per effort**
1. `MADV_SEQUENTIAL`/`WILLNEED` on expert mmaps. Trivial; only matters if the experts don't stay in the page cache, which is likely with ~89 GiB of GGUF against 57 GB RAM.
2. A hipBLASLt/rocBLAS tuning table for the GEMM shapes Avalon's prefill actually uses. Easy if those GEMMs go through hipBLASLt.
3. A static resident-expert router bias. Small code; needs a quality A/B on agent tasks.
4. A bigger prefill ubatch with overlapped expert upload, if Avalon's prefill is upload-bound. llama.cpp's `--ubatch-size` and op-offload already partly do this.
5. The QSA exact scorer, only for prompts over 131K.

**What Strata's 7900 XTX runs suggest about llama.cpp's kernels on gfx1100**
- Strata's per-arch kernel results on gfx1100 (most variants slower; mmvq multi-column slower than on NVIDIA) suggest llama.cpp's existing gfx1100 mmvq/MMQ choices are already near a local optimum.

### Gaps
- No Strata number exists for gfx1100 decode with the adaptive cache against llama.cpp's static `--n-cpu-moe` on the same model.
- The cross-engine comparison rests on different cards (XTX 24 GB vs our XT 20 GB) and different engine versions.

## Q4. Which agent-robustness features are worth porting as server-side logic?

### Takeaway
Strata's server has a rich, measured layer of agent-robustness logic. The best candidates for porting into llama-server or an Avalon proxy:
- tolerant tool-call parsing for Qwen's XML format (recovery: 98.3% vs 92.9% read correctly on 1,462 real agent turns);
- `reasoning_close_retry`;
- `answer_room` and the reasoning-budget wrap-up;
- repeat and loop stops;
- `fit_max_tokens`;
- serving any model name;
- a no-progress watchdog;
- prefix-cache-friendly prompt normalisation (`env_to_user` for opencode).

All of these are pure server or template logic and need no kernel work.

### Cited Findings
**Tool-call parsing** (config table in [D "Config keys"](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#config-keys-strata-modeljson-outside-args))
- `tool_call_recovery` (default on since 0.2.8) also reads a declared tool's call written:
  - as bare `<parameter=NAME>`;
  - as JSON inside `<tool_call>`, including inside a Markdown fence (#2037, [commit de603bb8](https://github.com/Niko1221/Strata/commit/de603bb8));
  - as `<function=NAME>` without `<tool_call>`;
  - as a second call inside one `<tool_call>`.
- Measured: 98.3% read as intended with it and 92.9% without, on 1,462 agent turns (Qwen3.8 under Claude Code). "Only a declared tool's call is ever recovered."

Additions since our checkout (all merged):
- **Misspelt openers** (`<toolcall>`, `<tool-call>`) followed by a declared `<function=NAME>` open a call (#5276) — [commit 63fd7592](https://github.com/Niko1221/Strata/commit/63fd7592)
- **Name variants:**
  - `tools.write`, `functions.write` and `default_api.write` map to the declared tool `write` (#4751) — [commit 35057071](https://github.com/Niko1221/Strata/commit/35057071)
  - `<function=tools NAME>`, and a bare `<function=tools>` matched to the one tool whose parameters fit (#5409) — [commit 68ab15dd](https://github.com/Niko1221/Strata/commit/68ab15dd)
  - A namespaced MCP tool written by its last segment (get_asset_info for mcp__assets__get_asset_info) maps when the match is unique (#5281) — [commit 05506f95](https://github.com/Niko1221/Strata/commit/05506f95)
- **A tool the conversation already called counts as declared** (#4663) — [commit 3282c613](https://github.com/Niko1221/Strata/commit/3282c613), [commit 6a8752f9](https://github.com/Niko1221/Strata/commit/6a8752f9)
- **Damaged arguments, a stray `</think>` and a missing `</function>`** no longer cost the call (#4427, #4077) — [commit d7113273](https://github.com/Niko1221/Strata/commit/d7113273)
- **False positives avoided:**
  - Quoted `<tool_call>` / `</tool_call>` in messages, and calls inside fences or indented code, stay text (#2026, #4702, #4703) — [commit a6830008](https://github.com/Niko1221/Strata/commit/a6830008), [commit 209dcadb](https://github.com/Niko1221/Strata/commit/209dcadb)
  - A file argument that itself contains closing tags is kept whole (#4701): a closed call is held until the turn ends, a new call starts, or 1,500 characters pass — [commit 49b880c0](https://github.com/Niko1221/Strata/commit/49b880c0)
- **Parser speed:** the opener checks scan only the last 256 characters, so a long call is not rescanned on every token — [commit b1541ab2](https://github.com/Niko1221/Strata/commit/b1541ab2)
- **`tool_choice: "required"` with several tools:** the server pre-writes only `"<tool_call>\n"`. Pre-writing `<function=` made the model invent names (PR #4828) — [commit d7e555bd](https://github.com/Niko1221/Strata/commit/d7e555bd)
- **Anthropic tool results** are reordered into call order by `tool_use_id` before the template pairs them by position (PR #4832) — [commit 95381770](https://github.com/Niko1221/Strata/commit/95381770)
- **`undeclared_tool_calls: "call"`** (#4622) returns a well-formed call to an undeclared tool as a tool call, so the client can reply with its valid names. The default `"text"` keeps it as text, with `strata_rejected_tool_calls` beside it — [commit 75acfca3](https://github.com/Niko1221/Strata/commit/75acfca3)

**Reasoning control** (all keys from [D "Config keys"](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#config-keys-strata-modeljson-outside-args) unless noted)
- `reasoning_close_retry` (off by default; we used it): when a reply ends inside its thinking with no answer and no tool call, close the thinking once and continue (#1053).
- `reasoning_budget_tokens`: the server ends thinking with a short wrap-up line plus `</think>`. It accepts llama.cpp's `thinking_budget_tokens`.
- `answer_room` (default on; new since our checkout):
  - with thinking on, no budget and max_tokens >= 1,024, thinking is wrapped up when a quarter of the cap (at least 512 tokens) is left;
  - requests without max_tokens get 3/4 of the context.
- `reasoning_loop_recovery`: `"stop"` or `"recover"` (restart once) when the reasoning repeats whole passages, checked every 512 tokens. It can misfire on repeated code (#728).
- `repeat_stop_tokens` (256): the reply ends as `length` after 256 repeats of one token.
- Open [PR #5512](https://github.com/Niko1221/Strata/pull/5512): skip thinking on short direct-answer prompts (numbers in Q2).
- `anthropic_thinking: "on_request"`: renders without thinking when an Anthropic request asks for none (Claude Code helper calls).
- `effort_position: "end"`: puts the effort level in a late system turn so effort changes reuse the cache.

**Serving and reliability**
- `fit_max_tokens` (default true): a max_tokens past the context is shortened, not refused with 400.
- `aliases`: extra names in `/v1/models`, "Any other name is still served".
- The Anthropic route serves the one loaded model for an unknown name (Claude Desktop sends claude-5-5-sonnet). With several models loaded it still returns 400 (#5312) — [commit ea765a57](https://github.com/Niko1221/Strata/commit/ea765a57)
- Watchdog: a request that makes no progress for 2 minutes ends with an error and a stall report, and the next request restarts the engine (`STRATA_WATCHDOG_S`, issue #29). `engine_silence_s` (300 s) ends the engine after that long without a line during a request; while a prompt is read, each chunk may take 3x the previous one's time — [D "Troubleshooting"](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#troubleshooting), [D "Config keys"](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#config-keys-strata-modeljson-outside-args)
- NaN retry: the prompt is re-read in batch mode (#3465) — [commit e0cfb80f](https://github.com/Niko1221/Strata/commit/e0cfb80f)

**Prefix-cache friendliness** (important for a hybrid recurrent model, where state cannot be truncated)
- `env_to_user` moves opencode's `<env>` block (working directory, date) into the first user message. The system turn then stays identical across sessions and the disk prefix cache can restore it — [D "Config keys"](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#config-keys-strata-modeljson-outside-args)
- `STRATA_CACHE_MESSAGE_BOUNDARY=1` saves a checkpoint at the previous turn boundary, so an edited final user message does not force a re-read (prefixes of at least 8,192 tokens with a tail of at most 1,024) — [docs/MESSAGE_BOUNDARY_CACHE.md](https://github.com/Niko1221/Strata/blob/main/docs/MESSAGE_BOUNDARY_CACHE.md)
- `--prompt-cache-tail` adds one extra checkpoint near the prompt end, for branches that diverge late — [docs/PROMPT_CACHE_TAIL.md](https://github.com/Niko1221/Strata/blob/main/docs/PROMPT_CACHE_TAIL.md)

### Inferences
**High value, low to moderate effort** (as a patch to llama.cpp's Qwen3-Coder/XML tool parser in `common/chat*.cpp`, or as a proxy in front of Avalon)
1. Tolerant tool-call recovery, including the misspelt-opener, prefix and namespaced-name rules, "already called counts as declared", and the false-positive guards (quoted, fenced, closing tags in file content).
2. `reasoning_close_retry` and `answer_room`. These need server-side re-entry: inject `</think>` and continue. llama-server has `--reasoning-budget`, but no "closed with no answer -> retry" logic, as far as I know.
3. `fit_max_tokens` and the any-name/alias fallback (a trivial proxy feature).
4. `env_to_user` for opencode. Trivial in a proxy, and it directly improves prefix-cache hits for our opencode use.
5. Message-boundary and tail checkpoints. llama.cpp's `--ctx-checkpoints` for recurrent/hybrid models exists, but the checkpoint placement policy (at the last user-turn boundary) would be a small, valuable change.
6. A watchdog with no-progress detection and auto-restart. Can be done in a systemd/launcher wrapper; Avalon already runs under avalon.service.
7. `repeat_stop_tokens` and loop detection. Easy.
8. Anthropic `tool_use_id` reordering. Only if Avalon serves `/v1/messages`.

**Lower priority**
- `undeclared_tool_calls: "call"` and `tool_choice: required` prefill tweaks are niche.
- PR #5512 thinking-skip is a heuristic classifier, still open and not merged.

### Gaps
- I did not inspect Avalon's or current llama.cpp's tool parser to list which recovery cases it already handles.
- Strata's 98.3% vs 92.9% figure comes from one corpus (1,462 turns under Claude Code), not from opencode.
- No measured completion-rate numbers were found for `reasoning_close_retry` or `answer_room`.
