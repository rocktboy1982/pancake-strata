# Pull Avalon main first, then cherry-pick
The biggest available speed-up for our setup does not come from upstream llama.cpp or Strata. It comes from Avalon's own repo, which is **48 commits ahead of our installed build c24c03d**. Those commits add a registered-RAM expert arena (on by default on Linux), a hosted-MoE limit computed from the measured link rate, the IQ3_KS MMA MMQ path, and a fix that launches q8_0/q4_0 sparse FA with the right tile. There is also an open PR that makes our own `ngram-simple,draft-mtp` launcher the default ([compare c24c03d...main](https://github.com/freriadoc/Avalon/compare/c24c03d...main), [Avalon PR #41](https://github.com/freriadoc/Avalon/pull/41)).

Upstream llama.cpp is 191 commits past Avalon's base, and upstream's headline MoE work is mostly an expert cache that Avalon replaced with its own buun-derived cache. #29887, #31233, #30966, #31888 and #31497 therefore matter as ideas, not as patches. Most upstream server and speculative features apply only partly too, either because Avalon already has its own version (FR-Spec trimmed MTP head, `LLAMA_MTP_KV_ONLY`, route tail skip, hosted MoE) or because they target NVIDIA. Four upstream items are worth porting now or testing right away:
- **#31751**, a one-pragma fix for RDNA3 MMQ register spills: +36% at 17-32-token batches on gfx1100, which is our n-gram verify size.
- **#30137**, reference-counted checkpoints (merged).
- **#24891 and #30745**, checkpoint fixes for hybrid models in multi-turn tool use. Opencode sessions hit exactly that case.
- **#29548**, an ngram-simple fix for periodic text.

From Strata, the cheap wins are policy and server logic, not kernels:
- `madvise(MADV_SEQUENTIAL)` on the expert mmap: follow-up turns 2.5-3x faster on an RDNA2 box whose RAM was too small for the model ([Strata PR #5561](https://github.com/Niko1221/Strata/pull/5561));
- a hipBLASLt tuning table;
- tolerant Qwen XML tool-call recovery: 98.3% vs 92.9% of agent turns read correctly;
- `env_to_user`, which keeps opencode's system prompt cacheable.

Two prerequisites come before any of this. The model folder is gone and avalon.service is disabled, so no number in this report has been measured on our box. And the D drive has too little free space to re-download the model.

## Prerequisite: restore the model before anything is measured
The baseline check on 2026-11-08 found three problems:
- **`/home/dan/clone-of-d/Avalon-models/Qwen3.8-Flash-Next-Avalon-GGUF` no longer exists.** That means the 33 GGUF parts, `MTP/` and `moe-cache-ranking.bin` are all missing ([download.log](file:///home/dan/clone-of-d/Avalon-models/download.log)).
- **avalon.service is inactive and disabled.** It stopped after 1,230 restarts that timed out waiting for part 00033. The service log also shows `ROCm error: invalid device function` aborts left over from the RTX 3060 period ([avalon-service.log](file:///home/dan/clone-of-d/Avalon/avalon-service.log)).
- **The D partition has 78 GB free, but the re-download is ~89 GiB (~96 GB).** About 20 GB must be freed on D before the download can finish. Every gain below is an author's claim from other hardware until the model is back and the bench rerun.

Our own reference numbers come from before the model went missing. They are the bar every item below has to beat:
- ~100 t/s decode on short prompts;
- 1,771/1,805 t/s prefill at 16K/32K;
- a median of 168-181 t/s on code edits with `ngram-simple,draft-mtp`, against 118 t/s with MTP alone ([avalon-guide.md](file:///home/dan/clone-of-d/llama-cpp-7900xt-optimization/docs/avalon-guide.md), [bench-edit-ngram.txt](file:///home/dan/clone-of-d/Avalon/bench-edit-ngram.txt)).

## Avalon already owns most of what upstream is building
The installed binary is built from Avalon commit **c24c03d**. It contains upstream llama.cpp up to **0ec4027 (PR #31434, 2026-10-31)**, applied by linear cherry-pick/rebase. Avalon left a few upstream commits out on purpose; #31380, for example, is missing ([build-update.log](file:///home/dan/clone-of-d/Avalon/build-update.log), [Avalon README](https://github.com/freriadoc/Avalon/blob/main/README.md), [PR #31434](https://github.com/ggml-org/llama.cpp/pull/31434)). Upstream master is **191 commits ahead** of that base ([compare](https://github.com/ggml-org/llama.cpp/compare/0ec4027a2a0183de87c007069adddc06e665f7de...master)).

Two research notes assumed different cutoffs:
- One looked at commit a91372e.
- One estimated, from the server code, that Avalon tracks upstream only to about 2026-10-20, because Avalon has #28554 but lacks [#30856](https://github.com/ggml-org/llama.cpp/pull/30856) and [#31256](https://github.com/ggml-org/llama.cpp/pull/31256).

The second estimate mostly reflects server files Avalon trimmed or dropped (realtime, connect, model scheduling), not a missing rebase. In practice, an upstream PR merged between 10-20 and 10-31 is probably present unless Avalon dropped it, and anything merged after 10-31 has to be ported by hand. Check presence before porting #31271, #30557, #31256 and #29948/#31188.

Avalon already ships its own version of many upstream and Strata ideas, which knocks several items out of the catalogue ([Avalon README](https://github.com/freriadoc/Avalon/blob/main/README.md)):

| Avalon already has | Overlaps upstream or Strata item |
|---|---|
| GPU expert cache with LFU admission and a shipped heat ranking | Upstream `--moe-cache-mib` with LRU/SLRU/ClassLRU; Strata's adaptive cache and preload |
| Hosted MoE: the CPU computes missing experts inside the HIP graph, and 1-2 misses can go to the GPU | Upstream `--moe-cache-cpu-share`; Strata's PCIe share |
| Tail skip of rank ≥7 missed experts (on by default) | Strata `STRATA_ROUTE_TAIL_SKIP=7` |
| Expert prefetch during prefill | Strata I/O prefetch (in part) |
| QSA sparse attention over q8_0/q4_0 KV | Upstream quantized-KV FA PRs; Strata QSA scorer |
| MTP head with its LM head cut to 65k tokens (FR-Spec), sharing the expert cache | Upstream #31037/#31031 trimmed draft head |
| `LLAMA_MTP_KV_ONLY` env knob | Upstream #31750 KV-only MTP catch-up |
| RDNA3 WMMA chunked GDN prefill, fused qwen4exp kernels | Upstream #29353 GDN chunked; Strata `STRATA_GDN_CHUNKED` |

Upstream expert-cache patches target `src/llama-moe-cache.cpp`, which **does not exist in Avalon** ([avalon_baseline local diff](https://github.com/freriadoc/Avalon)). So #31233 (ClassLRU), #31853 (CRF miss selection), #31888 and #31497 are policy ideas to compare with Avalon's LFU, not patches to merge. #31497 even works against a deliberate Avalon design: Avalon makes the MTP head share the expert cache.

## Upstream llama.cpp catalogue: 60-odd items, a handful aimed at RDNA3
The tables list every upstream item the research found that adds a feature or claims a speed gain, plus bug fixes that bring speed. Claimed gains are the PR authors' numbers on their own hardware. "In Avalon?" marks status against base 0ec4027.

### GPU kernels (HIP/RDNA3 and CUDA code that compiles into HIP)
| PR | Status | What it does | Claimed gain (author's hardware) | In Avalon? |
|---|---|---|---|---|
| [#31751](https://github.com/ggml-org/llama.cpp/pull/31751) | open draft, 11-05 | Stops J=32 MMQ register spills on RDNA3 (`#pragma unroll 1` in Q8_0-layout kernels) | **PP17 +36.5%, PP32 +36.1%**, W7800 gfx1100, ROCm 10.1 | No |
| [#31746](https://github.com/ggml-org/llama.cpp/pull/31746) | open draft, 11-05 | Tunes D256 MMA flash attention for gfx1100 | Attention op time 0.70-0.84x of master | No |
| [#31234](https://github.com/ggml-org/llama.cpp/pull/31234) | open, 10-27 | Native q4/q8 KV in the HIP FA tile kernel | q4_0 tg **+13% @64k, +24% @131k**, R9700 RDNA4 | No (overlaps QSA) |
| [#30792](https://github.com/ggml-org/llama.cpp/pull/30792) | open, 10-19 | Native q8_0 KV in the tile kernel | Tested on gfx1151; numbers not captured | No |
| [#31802](https://github.com/ggml-org/llama.cpp/pull/31802) | open, 11-06 | Fused q8/q4 KV dequant in MMA FA | Decode +34% @64k, +63% @131k, sm_89 only | No |
| [#26419](https://github.com/ggml-org/llama.cpp/pull/26419) | open, 08-01 | WMMA MMA FA for D256 on RDNA | Still 12-20% slower than the old rocWMMA kernel | No (superseded by #31746) |
| [#28714](https://github.com/ggml-org/llama.cpp/pull/28714) | open | MMQ mma/load split for RDNA3.5 | pp +2-4.8%, gfx115x only | No |
| [#31625](https://github.com/ggml-org/llama.cpp/pull/31625) | open, 11-03 | Fast-math on HIP | No numbers | No |
| [#26301](https://github.com/ggml-org/llama.cpp/pull/26301) | open | Dequant-float matvec for K-quants | +7-10% decode on small dense models, RDNA3.5 | No |
| [#26592](https://github.com/ggml-org/llama.cpp/pull/26592) | open | hipCUB for argsort/top-k | pp +0.5% | No |
| [#30489](https://github.com/ggml-org/llama.cpp/pull/30489) | open, 10-14 | Fuses GLU into the Q8_1 activation quantize (includes upstream's cache slot map) | No HIP numbers | No |
| [#31898](https://github.com/ggml-org/llama.cpp/pull/31898) | open, 11-08 | Scratch memory in graph_optimize (infrastructure) | None yet | No |
| [#31838](https://github.com/ggml-org/llama.cpp/pull/31838) | open, 11-07 | Fixed rocBLAS workspace | Not tested for speed | No |
| [#30370](https://github.com/ggml-org/llama.cpp/pull/30370) | open draft | HIP sparse fattn | "TODO add benchmark" | No |
| [#31047](https://github.com/ggml-org/llama.cpp/pull/31047) | merged 11-05 | Linearized MoE layout for MMQ/MMF | 2-4% | No (conflicts with Avalon's MMQ MUL_MAT_ID changes) |
| [#29948](https://github.com/ggml-org/llama.cpp/pull/29948) / [#31188](https://github.com/ggml-org/llama.cpp/pull/31188) | merged 10-25 / 10-28 | Gate/up+GLU fused into one MMQ launch | pp16384 +6-17% on NVIDIA; **gfx906 regressed 15-50%** ([#31599](https://github.com/ggml-org/llama.cpp/issues/31599)) | Probably (pre-base) |
| [#29353](https://github.com/ggml-org/llama.cpp/pull/29353) | merged 10-14 | Chunked GDN kernel for CUDA/HIP | PP +11-21% on NVIDIA | Yes (Avalon also has its own gdn-wmma) |
| [#28552](https://github.com/ggml-org/llama.cpp/pull/28552) | merged 09-09 | Routed-MoE MMQ N-tiles sized for RDNA3 | Large routed prefill gains, W7800 | Yes |
| [#29184](https://github.com/ggml-org/llama.cpp/pull/29184), [#25952](https://github.com/ggml-org/llama.cpp/pull/25952), [#29720](https://github.com/ggml-org/llama.cpp/pull/29720), [#30517](https://github.com/ggml-org/llama.cpp/pull/30517), [#30553](https://github.com/ggml-org/llama.cpp/pull/30553) | merged 09-10 | Shared-expert MMVQ fusion, MoE reduction fusion, rms_norm float4, low-bit MMVQ, Stream-K cap | 0.4-16%, mostly NVIDIA or MI100; RDNA "more neutral" | Yes |
| [#31818](https://github.com/ggml-org/llama.cpp/commits/master) | merged after base | SSM_SCAN fast path for d_state=64 | Not stated | No |

There is one compiler caveat. A Discussion found **~126 kernels regressed in ROCm 10.1 vs 7.2.4** (on gfx906), and maintainers are moving CI to 10.1 anyway ([Discussion #30622](https://github.com/ggml-org/llama.cpp/discussions/30622), [PR #30652](https://github.com/ggml-org/llama.cpp/pull/30652)). Avalon was tuned on ROCm 7.2.4, and we run 10.0, so any ROCm upgrade needs its own A/B test.

### MoE offload, expert caching and I/O
| PR | Status | What it does | Claimed gain | Relevance to Avalon |
|---|---|---|---|---|
| [#29887](https://github.com/ggml-org/llama.cpp/pull/29887) | merged 10-07 | Upstream GPU expert cache (`--moe-cache-mib`) | QFN 1.57-2.2x decode on 4090/5090; R9700 HIP 1.3-2.5x | Removed in Avalon; it already has its own cache |
| [#30637](https://github.com/ggml-org/llama.cpp/pull/30637) → [#31233](https://github.com/ggml-org/llama.cpp/pull/31233) | merged 10-19 / 11-06 | SLRU, then ClassLRU (LRFU/MQ hybrid) eviction | **+6-7% decode on QFN**, 8% fewer uploads | Idea only; compare with Avalon's LFU |
| [#30966](https://github.com/ggml-org/llama.cpp/pull/30966) + [#31853](https://github.com/ggml-org/llama.cpp/pull/31853) | merged 11-07 | Run cache misses on the CPU (`--moe-cache-cpu-share`) plus a tuner; misses picked by CRF score | QFN at a 6.4 GB cache: 24.2 → 47.6 t/s | Avalon's hosted MoE already does this inside the graph |
| [#31888](https://github.com/ggml-org/llama.cpp/pull/31888) | merged 11-08 | The CPU part stops waiting for cached experts | +7-9% TG on Gemma 4, none on QFN | Check whether hosted MoE already overlaps |
| [#31497](https://github.com/ggml-org/llama.cpp/pull/31497) | merged 11-08 | Keeps the MoE cache away from the draft model | Crash fix | Skip (Avalon shares the cache on purpose) |
| [#31286](https://github.com/ggml-org/llama.cpp/pull/31286) | open draft | Automatic cpu-share selection | Image-only results | Avalon main has a link-rate limit |
| [#31469](https://github.com/ggml-org/llama.cpp/pull/31469) | open RFC, 10-31 | Out-of-core expert streaming: O_DIRECT pager, VRAM slot pool, RAM victim cache | **TG 1.5 → 16-22 t/s**, 68 GB model on 16 GB RAM, NVIDIA | Concept only; conflicts with any cache |
| [#25294](https://github.com/ggml-org/llama.cpp/pull/25294) | open | Streams routed experts from disk | No numbers | Concept only |
| [#30594](https://github.com/ggml-org/llama.cpp/pull/30594) | open | Profile-prefilled hybrid cache | 1.4-1.7x, but **1-token batches only; MTP verify can get slower** | Poor fit |
| [#30971](https://github.com/ggml-org/llama.cpp/pull/30971) | open draft | Adaptive single-graph MoE offload, built "to beat Strata" on an R9700 | No consolidated numbers | Avalon already runs MoE inside one graph |
| [#30685](https://github.com/ggml-org/llama.cpp/pull/30685) | open draft | Async input copies in the scheduler | None | Watch |
| [#27861](https://github.com/ggml-org/llama.cpp/pull/27861), [#30059](https://github.com/ggml-org/llama.cpp/pull/30059) | open | Older expert caches | QFN +31% on 2x 3090 | Superseded |
| [#21067](https://github.com/ggml-org/llama.cpp/pull/21067) | open PoC | Prefetches the next layer's override weights | CUDA only | No |
| [#26414](https://github.com/ggml-org/llama.cpp/pull/26414) | open | mlocks the hottest experts in RAM | None | Avalon main's registered-RAM arena does the same job |
| [#28055](https://github.com/ggml-org/llama.cpp/pull/28055) / [#29195](https://github.com/ggml-org/llama.cpp/pull/29195) | open | No MAP_POPULATE / mmap on demand | Startup only | Startup time only |
| [#31886](https://github.com/ggml-org/llama.cpp/pull/31886) | open, 11-08 | ROCm mmap load hang fix: stages uploads through ≤64 MiB pinned buffers | Fixes a hang | Same symptom as Avalon issue [#3](https://github.com/freriadoc/Avalon/issues) and upstream [#29149](https://github.com/ggml-org/llama.cpp/issues/29149) |
| [#31139](https://github.com/ggml-org/llama.cpp/pull/31139) | open | Splits MoE cache banks above the max allocation size | Vulkan | No |
| [#29599](https://github.com/ggml-org/llama.cpp/pull/29599) | merged 09-30 | `llama_prefetch_rows`: madvise prefetch of PLE rows for qwen4exp/Gemma4 | pp512 177 → 383 t/s, 104 GiB qwen4exp on Strix Halo ROCm | Probably in Avalon (pre-base); see note below |
| [#31321](https://github.com/ggml-org/llama.cpp/pull/31321) | merged 10-29 | Reuses the staging buffer at load | Load time | Probably in Avalon |

The notes conflict on PLE. The upstream-performance note called QFN "not a PLE model". But Strata serves Qwen3.8-Flash-Next with a **28.8 GB n-gram/PLE table** ([Strata DETAILS](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#how-it-works)), and #29599 names qwen4exp, which is QFN's architecture. So PLE row access probably does matter for our model. That makes it worth checking how Avalon stores and reads that table before ruling out PLE-related ideas.

### Speculative decoding and qwen4exp/MTP graph work
| PR | Status | What it does | Claimed gain | Relevance |
|---|---|---|---|---|
| [#31443](https://github.com/ggml-org/llama.cpp/pull/31443) | merged 11-03 | qwen4exp writes the conv state tail without an intermediate cont | "**Free 2%**" on QFN generation | Avalon's qwen4exp.cpp is heavily rewritten (+1,088 lines), so check whether its fused kernels already remove the copy |
| [#31660](https://github.com/ggml-org/llama.cpp/pull/31660) | merged 11-05 | MTP conv snapshots in one set_rows instead of up to 12 copies per layer | Not stated | Conflict-prone; same check |
| [#30141](https://github.com/ggml-org/llama.cpp/pull/30141) | merged 11-03 | Saves and restores the MTP hidden state across checkpoints | Correctness | Partly present (`pending_h`) |
| [#31569](https://github.com/ggml-org/llama.cpp/pull/31569) | merged 11-05 | `--fit` accounts for skipped NextN layers (fixes "one layer on CPU, performance halves") | Up to 2x if affected | Our ~100 t/s matches the README's 98, so we are likely not affected; Avalon rewrote fit.cpp |
| [#28118](https://github.com/ggml-org/llama.cpp/pull/28118) | open draft | Keeps spec recurrent-state checkpoints on the device | gfx1151: spec **6.2 → 41-57 t/s** | Avalon has ON_DEVICE in memory-recurrent, and our spec runs at 168-181 t/s, so the collapse is not happening here |
| [#31669](https://github.com/ggml-org/llama.cpp/pull/31669) | open WIP | Replays GDN steps instead of rolling back | Flat single-seq; 73-91% less recurrent memory | Freeing memory gave nothing on this box |
| [#31750](https://github.com/ggml-org/llama.cpp/pull/31750) | open draft | KV-only MTP catch-up graphs | -2.2% step time | Avalon has `LLAMA_MTP_KV_ONLY` |
| [#27210](https://github.com/ggml-org/llama.cpp/pull/27210) | open | Adaptive MTP depth | Up to 151 t/s on recall at depth 12 | Needs n-max ≥7; deep drafts touch many experts under offload |
| [#27173](https://github.com/ggml-org/llama.cpp/pull/27173) | open | Drafts all tokens in one decode, with a top-32k draft vocab | +10% at depth 5, 2x 5090 | FR-Spec already trims the head |
| [#31037](https://github.com/ggml-org/llama.cpp/pull/31037) / [#31031](https://github.com/ggml-org/llama.cpp/pull/31031) | open | Trimmed MTP output head | +7.4% on SYCL | Already in Avalon (FR-Spec 65k) |
| [#30196](https://github.com/ggml-org/llama.cpp/pull/30196) | open | Reads context outputs with one sync | Table truncated | Small; watch |
| [#27489](https://github.com/ggml-org/llama.cpp/pull/27489), [#31415](https://github.com/ggml-org/llama.cpp/pull/31415) | open | MTP compute-buffer sharing; fewer realloc passes for dense Qwen3.5 | VRAM; dense only | No |
| [#31533](https://github.com/ggml-org/llama.cpp/pull/31533) | open | Few-row MMQ for batch sizes 2-24 | Ampere+ only | No (#31751 is the RDNA3 equivalent) |
| [#29548](https://github.com/ggml-org/llama.cpp/pull/29548) | open | ngram-simple keeps scanning older matches on periodic text | 0 → nonzero drafts on repeated text | **Directly our drafter** |
| [#30723](https://github.com/ggml-org/llama.cpp/pull/30723), [#28431](https://github.com/ggml-org/llama.cpp/pull/28431) | open | Faster ngram-cache; ngram-map-k4v fix | Up to 1.9x drafting speed | Other n-gram types, not ours |
| [#26283](https://github.com/ggml-org/llama.cpp/pull/26283), [#30931](https://github.com/ggml-org/llama.cpp/pull/30931), [#27692](https://github.com/ggml-org/llama.cpp/pull/27692), [#31732](https://github.com/ggml-org/llama.cpp/pull/31732) | open | Suffix decoding, tree speculation, speculative prefill (lossy), self-speculation (diffusion models only) | Unmeasured or N/A | Watch suffix and tree |
| [#27694](https://github.com/ggml-org/llama.cpp/pull/27694), [#28549](https://github.com/ggml-org/llama.cpp/pull/28549), [#30459](https://github.com/ggml-org/llama.cpp/pull/30459), [#29924](https://github.com/ggml-org/llama.cpp/pull/29924) | merged before base | Rejection sampling, CUDA graph for the MTP draft, draft n-ubatch, n-gram fix at temp>0 | — | Present |
| [#29963](https://github.com/ggml-org/llama.cpp/pull/29963) | open | Pipeline parallelism with host experts | 1.5-2x prefill, multi-GPU | No (single GPU) |

### Prompt cache, checkpoints, slots and server scheduling
| PR | Status | What it does | Claimed gain | Relevance for opencode |
|---|---|---|---|---|
| [#30137](https://github.com/ggml-org/llama.cpp/pull/30137) | merged 11-05 | Reference-counted checkpoints instead of deep copies | Grows with context | Hybrid model with 256k context: yes |
| [#24891](https://github.com/ggml-org/llama.cpp/pull/24891) | open | Fixes checkpoint invalidation after tool requests in multi-turn chats ([#24055](https://github.com/ggml-org/llama.cpp/issues/24055)) | Avoids re-prefill | **Exactly opencode's pattern** |
| [#30745](https://github.com/ggml-org/llama.cpp/pull/30745) | open draft | Last-user-end checkpoint supersedes earlier ones | Reuse **91% → 94%**, save file 3.6 → 0.8 GB | Yes |
| [#30992](https://github.com/ggml-org/llama.cpp/pull/30992) | open | `--prefill-budget-busy`: chunked prefill, shortest job first | Latency for small requests | Only with `-np 2` (title requests) |
| [#30970](https://github.com/ggml-org/llama.cpp/pull/30970) | open | `--slot-attach`: slots share prefix cells | No copy, no recompute | Only with 2 slots |
| [#28092](https://github.com/ggml-org/llama.cpp/pull/28092) | open, +1,706 lines | `--cache-disk` persistent prompt cache, hybrid-aware | Avoids a ~70 s cold prefill at 131k | Yes; must point at D |
| [#29738](https://github.com/ggml-org/llama.cpp/pull/29738) | open | Session-keyed automatic slot save/restore | — | Only if opencode sends session headers (unverified) |
| [#28532](https://github.com/ggml-org/llama.cpp/pull/28532) | open | `--slot-linger-ms` | Fewer KV swaps | Multi-session only |
| [#30188](https://github.com/ggml-org/llama.cpp/pull/30188) | draft | `cache_prompt:false` requests discard their state | Title requests stop evicting the agent prompt | Yes |
| [#30997](https://github.com/ggml-org/llama.cpp/pull/30997), [#30866](https://github.com/ggml-org/llama.cpp/pull/30866) | open | Retry RAM-cache restore; pinned-slot NaN that forced cold reprocessing | Avoids full reprocessing | Small fixes |
| [#29918](https://github.com/ggml-org/llama.cpp/pull/29918), [#29578](https://github.com/ggml-org/llama.cpp/pull/29578), [#26204](https://github.com/ggml-org/llama.cpp/pull/26204), [#28046](https://github.com/ggml-org/llama.cpp/pull/28046) | open | Hybrid cache reuse after mid-prompt edits, pinned prefix preload, KV clone between slots, per-entry cache limit | 2.12x on branching (M3 Max) | Watch |
| [#27140](https://github.com/ggml-org/llama.cpp/pull/27140) | open | Fixes slow prefill with q4_0 KV on CUDA | 74 → 1,182 t/s on a 3090 | Our 1.8k t/s says HIP is not affected |
| [#31391](https://github.com/ggml-org/llama.cpp/pull/31391), [#31073](https://github.com/ggml-org/llama.cpp/pull/31073), [#31539](https://github.com/ggml-org/llama.cpp/pull/31539) | merged / merged / open | Gathered KV state reads, GPU-resident rotation constants, faster find_slot | Faster slot save | Small |
| [#30492](https://github.com/ggml-org/llama.cpp/pull/30492) | open | Keep-alive kept after streamed responses | Saves a TCP reconnect per turn | Small |
| [#30689](https://github.com/ggml-org/llama.cpp/pull/30689), [#31489](https://github.com/ggml-org/llama.cpp/pull/31489) | draft / open (ngxson) | Engine library split; service mode | Architecture | **Biggest future rebase risk** |

### Agent-facing parser and reasoning features
None of these are speed items in the kernel sense. They count here because a dropped tool call or a reply stuck in thinking costs a full agent turn. The candidates:
- tool arguments accepted in any order: [#30193](https://github.com/ggml-org/llama.cpp/pull/30193) (JSON) and [#31040](https://github.com/ggml-org/llama.cpp/pull/31040) (XML `<parameter=>`), with overlaps [#30382](https://github.com/ggml-org/llama.cpp/pull/30382) and [#30476](https://github.com/ggml-org/llama.cpp/pull/30476);
- named `tool_choice` ([#31564](https://github.com/ggml-org/llama.cpp/pull/31564));
- withholding tools when `tool_choice:none` ([#30434](https://github.com/ggml-org/llama.cpp/pull/30434));
- a reasoning-aware `--json-schema` ([#31862](https://github.com/ggml-org/llama.cpp/pull/31862));
- template reasoning-effort discovery ([#31271](https://github.com/ggml-org/llama.cpp/pull/31271), merged 10-27; it is not confirmed to clamp "high" to a supported level, so it does not yet replace our template mapping);
- a soft reasoning-budget wrap-up ([#27578](https://github.com/ggml-org/llama.cpp/pull/27578));
- `reasoning_effort` overriding `--reasoning` ([#27221](https://github.com/ggml-org/llama.cpp/pull/27221)).

## Strata catalogue: memory policy and agent logic, not kernels
Strata upstream is 4,081 commits past our old checkout. Strata itself does not beat Avalon on RDNA3. A community 7900 XTX run on engine 0.1.57 gets 80.3 t/s decode and 1,381 t/s prompt at 32K ([Strata PR #5415](https://github.com/Niko1221/Strata/pull/5415)). Our smaller 7900 XT does ~100 t/s and 1,805 t/s on Avalon. What Strata offers is techniques worth copying. Its own gfx1100 kernel experiments came out mostly neutral or negative: expert layout modes 2-3x slower, int8 WMMA attention at 0.995x end to end ([PR #5458](https://github.com/Niko1221/Strata/pull/5458), [PR #5535](https://github.com/Niko1221/Strata/pull/5535)). That suggests llama.cpp's gfx1100 kernels are already near a local optimum.

| Strata item | Status | Measured effect | Avalon equivalent |
|---|---|---|---|
| `MADV_SEQUENTIAL` on the expert mmap ([PR #5561](https://github.com/Niko1221/Strata/pull/5561)) | open | Follow-up turns **65-76 s → 23-28 s**; decode 33-44 → 42-52 t/s; RX 6900 XT, 32 GB RAM, byte-identical output | None known |
| Router-predicted next-layer `pread` prefetch from the page cache ([DETAILS](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#linux-file-tier)) | default | Not isolated | Prefill prefetch only |
| hipBLASLt tuning table for gfx1100, including ROCm 10.0.0 ([AMD_HIP](https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP.md#tuning-table)) | shipped | Prompt **+32% / +78% / +82%** at 3K/6.5K/131K on a 7900 XTX | Unknown whether Avalon's prefill GEMMs go through hipBLASLt |
| PCIe share of missed experts ([DETAILS](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#speed-settings)) | CUDA default, opt-in on gfx1100 | — | Hosted MoE + link-rate limit (Avalon main) |
| Route tail skip at rank ≥7 | CUDA default | Answers differ slightly | Already in Avalon (on by default) |
| Fetch admission of PCIe-fetched misses ([24c6c2f6](https://github.com/Niko1221/Strata/commit/24c6c2f6)) | opt-in | — | Avalon 517ca1b admission rule |
| Router bias toward resident experts ([DETAILS](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md), [PR #5596](https://github.com/Niko1221/Strata/pull/5596)) | opt-in; reverted from default | Code +37% decode, but **agent score 5.6 → 4.5/10** and repetition loops; DeepSeek variant +56% at 93-96% top-1 agreement | None |
| PLE table resident in RAM ([3b34b6c7](https://github.com/Niko1221/Strata/commit/3b34b6c7)) | default where it fits | 8K prompt +22.5%, R9700 | Needs ~model + table + 16 GB RAM; does not fit 57 GB |
| KVA approximate late-layer prefill ([KVA.md](https://github.com/Niko1221/Strata/blob/main/docs/KVA.md)) | opt-in | Prefill +28%, but **decode -11%** | None |
| Exact long-prompt QSA scorer ([AMD_HIP](https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP.md#exact-long-prompt-qsa-scorer)) | opt-in on gfx1100 | +10-13% at 131K, +22% at 240K; nothing on cached follow-ups | Avalon QSA (different scorer) |
| PA_FAST + fused prefill + ring 512 ([PR #5458](https://github.com/Niko1221/Strata/pull/5458)) | open report | +13.8% at 94K deep prefill, not bitwise | Strata-internal kernels |
| TSUM ([Issue #5459](https://github.com/Niko1221/Strata/issues/5459)) | proposed | -0.9% decode time | — |
| DFlash2 drafter ([66b346c2](https://github.com/Niko1221/Strata/commit/66b346c2), [29e8a47a](https://github.com/Niko1221/Strata/commit/29e8a47a), [aba4ddac](https://github.com/Niko1221/Strata/commit/aba4ddac)) | default on some backends | +14% on M5 Max, +7% short / +1% long on gfx1201, slower on Spark; gfx1201 HIP reverted to MTP | Avalon has the upstream `dflash` spec types; upstream [#30887](https://github.com/ggml-org/llama.cpp/issues/30887) reports QFN+DFlash2 crashes |
| Longer prompt-lookup window, up to 16 ([DETAILS](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#opt-in-switches)) | opt-in | **+11-14%** on file rewrites and agent edits | Our ngram-simple settings |
| Calibrated verify-window gate `STRATA_DF2_GATE` ([b97433df](https://github.com/Niko1221/Strata/commit/b97433df)) | opt-in | — | None |
| `/v1/tune` live knobs, conv-route ([PR #5516](https://github.com/Niko1221/Strata/pull/5516)) | open | Single-shot TTFT -42% | llama.cpp caching is cheaper; low value |
| Thinking skip for short answers ([PR #5512](https://github.com/Niko1221/Strata/pull/5512)) | open | Median 0.72 → 0.50 s | Heuristic |
| Tolerant Qwen XML tool-call recovery ([DETAILS config](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#config-keys-strata-modeljson-outside-args), [63fd7592](https://github.com/Niko1221/Strata/commit/63fd7592), [35057071](https://github.com/Niko1221/Strata/commit/35057071), [49b880c0](https://github.com/Niko1221/Strata/commit/49b880c0)) | default | **98.3% vs 92.9%** of 1,462 Qwen3.8 agent turns read as intended | Upstream strict parser |
| `reasoning_close_retry`, `answer_room`, loop and repeat stops | default/opt-in | Unmeasured | Partly `--reasoning-budget` |
| `env_to_user` (moves opencode's `<env>` block out of the system turn) | key | Keeps the system prefix identical across sessions | None |
| Message-boundary and tail checkpoints ([MESSAGE_BOUNDARY_CACHE.md](https://github.com/Niko1221/Strata/blob/main/docs/MESSAGE_BOUNDARY_CACHE.md), [PROMPT_CACHE_TAIL.md](https://github.com/Niko1221/Strata/blob/main/docs/PROMPT_CACHE_TAIL.md)) | opt-in | — | Analogue of upstream #30745 |
| No-progress watchdog, `fit_max_tokens`, serve any model name ([Troubleshooting](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#troubleshooting)) | default | Reliability | systemd restart only |

## Prioritized recommendation for Avalon on the 7900 XT
Expected gains are inferred from the authors' figures and adjusted for our setup: experts mostly in RAM, a ~89 GiB model on 57 GB RAM, MTP + n-gram drafting, and long opencode prompts. Effort means porting work into Avalon. None of it is measured here yet.

### Add now (after the model is restored)
| Item | Expected gain | Effort | Risk | Reason |
|---|---|---|---|---|
| **Pull Avalon main (48 commits)** ([compare](https://github.com/freriadoc/Avalon/compare/c24c03d...main)) | Registered-RAM arena should cut the "slow first session" and page-fault cost; link-rate hosted-MoE limit; IQ3_KS MMQ and sparse-FA tile fixes. A 7900 XTX user reports 2,113 t/s prefill with `--moe-cache-cpu-overlap 2` ([README](https://github.com/freriadoc/Avalon/blob/main/README.md)) | Rebuild only | Medium: open MTP hang [#40](https://github.com/freriadoc/Avalon/issues) (workaround `GGML_CUDA_MOE_CACHE_DEFER=0`); several WIP features (off by default); `install.sh` overwrites our launcher | It is the maintainer's own tuning for exactly this model; keep the c24c03d binaries as a rollback, as was done for a91372e |
| **Untried knobs already in the build:** `--moe-cache-cpu-overlap 2`, `GGML_SCHED_PREFETCH_EXPERTS=6`, larger `-ub` if it fits in 20 GB ([README](https://github.com/freriadoc/Avalon/blob/main/README.md)) | Prefill +10-20% (XTX report) | Launcher edit | Low | No porting; reversible |
| **#31751 RDNA3 J=32 MMQ spill fix** ([PR](https://github.com/ggml-org/llama.cpp/pull/31751)) | Up to +36% on 17-32-token verify batches; our n-gram drafts accept 27-35 tokens | One pragma | Low; scoped to Q8_0-layout kernels | Hits our hottest decode case. It is not the global "unroll flag" we tested; it is a targeted per-kernel spill fix. Helps the IQ4_NL MTP head now, and IQ3_KS if Avalon main routes it through q8_0 MMQ on HIP (to check) |
| **#29548 ngram-simple periodic-text fix** ([PR](https://github.com/ggml-org/llama.cpp/pull/29548)) | Drafts where ngram-simple now gives up | Small, common/ only (plain upstream code in Avalon) | Low | Our main speed lever is ngram-simple |
| **#30137 reference-counted checkpoints** ([PR](https://github.com/ggml-org/llama.cpp/pull/30137)) | Less copy time per checkpoint at 256k context | Small-medium (Avalon's trimmed server) | Low (merged upstream) | A hybrid model checkpoints constantly |

### Test first
| Item | Expected gain | Effort | Risk | Reason |
|---|---|---|---|---|
| **#24891 tool-request checkpoint fix + #30745 last-user-end checkpoint** ([#24891](https://github.com/ggml-org/llama.cpp/pull/24891), [#30745](https://github.com/ggml-org/llama.cpp/pull/30745)) | Avoids seconds of re-prefill per agent turn; reuse 91 → 94% | Medium | Medium: unreviewed, touches slot logic | The earlier "checkpoint tuning" test changed counts, not placement or invalidation; this is a new reason. Measure the reused-token % in the logs first |
| **Strata `MADV_SEQUENTIAL` / `WILLNEED`** ([PR #5561](https://github.com/Niko1221/Strata/pull/5561)) | 2-3x faster follow-up turns *if* experts fault from NVMe | A few lines in llama-mmap.cpp | Low | Measure `major-faults` per turn first; the registered-RAM arena in Avalon main may already remove most faults |
| **hipBLASLt tuning table** ([AMD_HIP](https://github.com/Niko1221/Strata/blob/main/docs/AMD_HIP.md#tuning-table)) | Prompt +30-80% *if* prefill GEMMs use hipBLASLt | Env var + table built on this card | Low | Only helps if a profile shows rocBLAS/hipBLASLt kernels in prefill; Avalon is mostly MMQ + gdn-wmma |
| **Longer n-gram draft window** ([Strata](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#opt-in-switches)) | +11-14% on edit turns | Flag change | Low | Separate from the MTP n-max that was tested |
| **#31234 native quantized KV in the FA tile kernel** ([PR](https://github.com/ggml-org/llama.cpp/pull/31234)) | +13-24% decode at 64-131k on RDNA4 | Medium | Medium: overlaps QSA's q8_0/q4_0 gather | First check which FA path Avalon's dense decode uses below the QSA threshold |
| **#31746 D256 MMA FA tuning** ([PR](https://github.com/ggml-org/llama.cpp/pull/31746)) | 16-30% less attention time in prefill/verify | Small-medium | Medium | Only if QFN's full-attention layers are D=256 (check the GGUF) |
| **#31443 / #31660 qwen4exp copies** ([#31443](https://github.com/ggml-org/llama.cpp/pull/31443), [#31660](https://github.com/ggml-org/llama.cpp/pull/31660)) | ~2% decode | Medium (conflicts with the qwen4exp rewrite) | Low | Check whether Avalon's fused kernels already removed these copies |
| **Strata tolerant tool-call recovery + `env_to_user` + `reasoning_close_retry`**, as a parser patch or proxy ([Strata config](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md#config-keys-strata-modeljson-outside-args)) | Fewer failed turns (92.9 → 98.3% on Claude Code traffic); stable system prefix keeps the cache warm | Small (proxy) to medium (common/chat) | Low-medium; risk of false-positive recovery | Pure logic; check opencode logs for parse failures first. #31040/#30193 are the upstream equivalents |
| **#28092 disk prompt cache** ([PR](https://github.com/ggml-org/llama.cpp/pull/28092)) | Removes a ~70 s cold prefill at 131k after restarts | Large (+1,706 lines) | Medium; disk use | Point it at D, never /; useful while restarts are frequent |
| **#31886 ROCm staged mmap uploads** ([PR](https://github.com/ggml-org/llama.cpp/pull/31886)) | Fixes load hangs | Small | Low | Only if loading hangs (Avalon issue #3); the arena may supersede it |
| **ClassLRU / CRF ideas from #31233 / #31853** ([#31233](https://github.com/ggml-org/llama.cpp/pull/31233)) | +6-7% decode on QFN vs SLRU (gain vs Avalon's LFU unknown) | Large (reimplement in moe-cache.cu) | Medium | Policy idea only; first compare cache hit rates in Avalon's stats |

### Watch
These are not ready, too broad, or do not fit our box yet:
- **#31802** fused-dequant MMA FA: NVIDIA only so far.
- **#31625** HIP fast-math: no numbers yet.
- **#31469 / #25294** out-of-core streaming: an RFC that conflicts with every cache.
- **#28118** on-device spec checkpoints: Avalon already gets 168-181 t/s with spec.
- **#27210 / #27173** adaptive or deeper MTP drafting: deep drafts touch more experts under offload.
- **#30992 / #30970 / #29738** multi-slot scheduling: only if we move to `-np 2` for title requests.
- **#30188**: keeps `cache_prompt:false` title requests from evicting the agent prompt.
- **#27578** soft reasoning budget.
- **#31271** reasoning-effort discovery: keep our template mapping until it is shown to clamp levels.
- **Strata's DFlash2**: an upstream crash report on QFN exists, and Strata itself reverted to MTP on HIP.
- **Strata's exact QSA scorer**: only above 131K tokens.
- **Strata's calibrated verify gate.**
- **Upstream #30689 engine split**: the biggest rebase hazard for any server patch.

Watch any ROCm 10.1 upgrade for the regression reported in [Discussion #30622](https://github.com/ggml-org/llama.cpp/discussions/30622).

### Skip
| Item | Reason |
|---|---|
| Upstream expert cache (#29887, #30637, #30966, #31888, #31286, #30594, #27861) | Avalon removed that code; hosted MoE already covers CPU miss compute, and #30594 hurts MTP verify |
| #31497 (no MoE cache for the draft) | Contradicts Avalon's deliberate shared MTP cache |
| #31750 KV-only catch-up; #31037/#31031 trimmed draft head | Already in Avalon (`LLAMA_MTP_KV_ONLY`, FR-Spec 65k) |
| #31669 GDN replay, #27489 buffer sharing, anything sold mainly as "frees VRAM" | Freeing VRAM through q4_0 K made no difference on this box |
| #31533, #28714, #28195, #26301, #26592 | NVIDIA-only or RDNA3.5-only, or no gain for 512-expert top-k |
| #27140 q4_0 prefill fix | CUDA problem; our HIP prefill already runs at 1.8k t/s |
| Strata router residency bias (`ROUTE_PRIOR`, `CACHE_ROUTE`) | Strata's own data shows lower agent quality and loops in long sessions; we run an agent |
| Strata KVA approximate prefill | Decode -11% and a quality risk; opencode decode matters more |
| Strata PLE-in-RAM | Needs model + 28.8 GB table + 16 GB headroom; 57 GB RAM cannot hold it |
| Strata gfx1100 kernel switches (int8 WMMA attention, TSUM, expert layouts) | Neutral or negative on 7900-class cards in Strata's own tests |
| Already tested with no gain: HIP graphs, the global unroll flag, COMPUTE power profile, draft p-min, checkpoint count tuning, AMD-Ecosystem fork, Vulkan, turbo4/TurboQuant KV | Retest only with a new reason; none of the items above supplies one, except #31751 (per-kernel spill fix) and #24891/#30745 (checkpoint logic, not counts) |

## Conclusion
Upstream llama.cpp and Avalon are converging on the same problem from different ends. Upstream is rebuilding, one generic piece at a time, the expert-cache-plus-CPU-miss design that Avalon already ships as one model-specific HIP graph. Upstream's own author benchmarks on QFN (47-67 t/s on RTX 4090/5090) still trail Avalon's ~100 t/s on a smaller AMD card. Upstream's cache work is therefore a source of policy ideas, not code. The upstream code that does transfer cleanly is in areas Avalon left as plain upstream: speculative drivers, server checkpoints and chat parsing. That is also where opencode's real costs sit, in re-prefilling after tool turns and in dropped tool calls.

The order of work follows from that. First restore the model, after freeing ~20 GB on D. Then rebuild on Avalon main and rerun `bench.py` and the edit bench as the new baseline. Then land the one-pragma #31751 and the small n-gram and checkpoint fixes. Choose between the I/O-side items (MADV hints, disk cache) by measuring page faults and reused-token percentages first, because the registered-RAM arena may already have solved what they target. The biggest unknown is how much of our ~89 GiB working set really faults from NVMe per turn on 57 GB of RAM. That one number decides whether the I/O items are worth doing at all.
