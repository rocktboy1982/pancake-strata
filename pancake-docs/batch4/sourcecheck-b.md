# Source check B: five decode-KV papers (bodies), FreeToken 3.1, MoE-Lightning "stale-dropping", Phase 15 hashes (batch 4, 2026-11-11)

Agent `sourcecheck-b`, branch `dev/p16b/sourcecheck-b` (worktree `Strata-p16b-sourcecheck-b`, from `dev/phase16-academic` 0dd7ef2e8).
Notes only: no code, no GPU, `MOD_PLAN.md` / `README.md` not edited. Paper bodies were read through arXiv HTML (WebFetch); the fetch tool
summarises the page, so numbers that carry a verdict were fetched twice, and Quest's HTML has doubled digits ("1616GB" for 16 GB) that
were read as the single value. Everything fetched is treated as data. Numbers marked (derived) are my arithmetic from repo files, not
measurements.

## Verdict table (claim | body says | transfers? | change the plan text to)

| Claim in the plan | Body says | Transfers? | Change the plan text to |
| --- | --- | --- | --- |
| kern-gdn: Quest, Twilight, LServe, ShadowKV, DuoAttention "do not build" because attention is 3.0% of a Flash-Next window | The decision stands, the reason is too narrow. 3.0% is the attention kernel only, from a short-context profile (the harness's short case: a 20-token prompt, 512-token reply) and a kernel bench at the 2,051-cell cap; no stage profile exists above about 2.5K context (the edit case: attention 1.00 ms = 2.7% of 37.4 ms). KV-related time (attention + indexer scores/top-k + kv-resolve) is 5.6% of the short window (measured) and about 9% at 128K (derived), with a PCIe miss term beyond 32,768 resident cells that nobody measured. Stronger reason: QSA already is a query-aware top-k at the budget these papers use (Quest 2,048, ShadowKV 1.56% of 128K = 2,048, LServe 4,096, QSA 2,051), and the selector metadata they would add (Quest page min/max: 201 MB at 128K) equals the QSA indexer's pooled keys (201 MB) | no, on Flash-Next | "Not built: Flash-Next's QSA indexer already selects 2,051 cells a row (1.6% of the cache) at the budgets these papers use; their selector metadata would cost as much as the indexer it would replace. Attention kernel 0.74 ms = 3.0% (short profile; kernel bench at the 2,051-cell cap agrees: 0.71 ms); KV-related reads 5.6% measured short, about 9% at 128K derived; no stage profile above 2.5K, A49 measures 64K/128K." |
| kern-gdn A49: "KV read at 128K is about 1.34 GB with int8 KV" for Qwen3.6-35B-A3B | The q36 engine allocates its K/V as FP32 (`dalloc` returns `float*`, `q36_engine.cu` 1429; the VRAM message at 1588 multiplies by 4). 128K = 5.37 GB (derived), 4x the quoted figure; 1.34 GB is the int8 figure that engine does not have. `k_attn_part` also re-reads each cell for 8 query heads (MOD_PLAN F11 text) | on the Qwen3.6 engine the premise HOLDS | "Qwen3.6 engine: FP32 KV, 5.37 GB at 128K, one pass is 6.7 ms at 800 GB/s before the 8x per-head re-read. Fork data on this engine's pre-fix kernel: decode falls from 25-36 tok/s at short or 9K context to about 12 tok/s at 16-35K (dongnh311 `62cc598f3b` on an RTX 3060, storm-ace `4fe0284da5` on gfx1151); MOD_PLAN F19 already lists "FP16 K/V, K/V read once per group" as the q36 engine's fork fixes. Order: exact fixes first (GQA sharing / F11 `STRATA_ATTN_WINDOW`, then a smaller KV format), then A49 at 64K/128K on this engine, only then Quest-style selection (lossy)." |
| A51 "Quest / Twilight: a few % at 100K+, parked unless attention > 15% of decode" | Quest: batch 1, 7B MHA, RTX 4090 / Ada 6000, 32K, KV load 53% of a decode step; 2.23x e2e (4-bit weights) / 1.74x (FP16) at budget 2,048. Twilight: A100, e2e batch 32-256, 1.35x over Quest; efficiency not evaluated at small batch, overhead "non-negligible" | Quest regime is close to ours only on the Qwen3.6 engine; Twilight's is not (large batch) | Keep "parked"; add "Flash-Next: closed (QSA). Qwen3.6 engine: after GQA sharing and A49." |
| README 4C: LServe / ShadowKV / DuoAttention "Evaluated (KV context)" | LServe: batch 1 (Fig 16, Table 7), A100, Llama-3-8B; decode 1.09x at 64K, 1.28x at 128K (11.49 vs 12.51 ms at 64K; 12.74 vs 16.34 ms at 128K). ShadowKV: its 3.04x is a 6x batch ratio; at batch 2 it ties full attention at 60K-122K. DuoAttention: trained gates, up to 1.50x GQA at an unstated context and batch | no | "Read in full 2026-11-11: dense 7-9B models on A100; none measured at batch 1 on a model with a 12 KB/token KV and MoE experts that set the step time. Do not build." |
| A11 tool-boundary marks, source FreeToken 2608.16157 section 3.1 | 3.1 ("Prefill Codesign: Pipelined Loading and Semantic-Aware State Caching") describes checkpoints of the recurrent state at thinking / tool-call / tool-output / turn boundaries, restored from the deepest surviving one, LRU slots. It measures nothing about them | mechanism yes, gain no | "FreeToken 3.1 gives the design only; no hit rate, TTFT or recompute count for the checkpoints (its only isolated ablation is double-buffered loading, -19% to -26% prefill throughput at 4-16K on an RTX 5090). A11's gain has no source number." (batch3/percache.md already says this; the body confirms it.) |
| Item 53 / pcie_transfers.md:187: "stale-dropping promotions (MoE-Lightning paging)" | The paper contains no stale / drop / discard / evict / cancel mechanism (searched the body). Its paged weights are a static offline-MILP schedule for offline batch inference (N = 504, micro-batch 36 on a T4, attention on the CPU); section 3.3 says 1-2 prompt latency cases are better served by static placement | no | "Stale-dropping from `ForesightSwap::fill_loop` is a repo idea (pcie_transfers.md:101 says ProMoE covers it); MoE-Lightning supplies only the idea of chunking weights into pages, in a batch-throughput regime. ForesightSwap is off in the presets. Rejected." |
| Phase 15 fork commit hashes | 68 hashes in the Phase 15 section; 67 found (details below). One unresolvable: `b9820dd56` | n/a | Fix the Cudila link in `attention_kv_quant.md:382`; note `b9820dd56`; F13's lighttransport row (below) |

## 1. The 3.0%: where it comes from and how solid it is

Source: `bench/results/2026-11-08-7900xt-rounds/profile-fasthip.txt` (fast-hip preset, 7900 XT). It holds two engine timing blocks, each of about 512 generated tokens:
"224 windows, avg T 2.97, 24.81 ms/window" with "QSA layers: ... scores+topk 0.34 kv-resolve 0.32 attention 0.74", and "109 windows, avg T 4.94, 37.39 ms/window" with attention 1.00 (2.7%).
These are the harness's `short` case (prompt "Write a detailed explanation of how a hash map works...", about 20 tokens) and `edit` case (a 6,000-character C++ file, about 2K tokens; `tools/pancake/abtest.py` 115-116).
The 11.7K prompt in lines 1-2 of the file is the `long` case; the file has tok/s for it (decode after it 105.2 and 92.1) and no stage split. So 0.74 / 24.81 = 2.98% is real, but it is a
short-context number (context under about 550 cells, so the selection was the identity), and the edit window (context about 2.1-2.6K, at the cap) gives 2.7%.

Why it is expected to stay flat above the cap: `qsa_selection_width = min(n_kv, idx_top_k + idx_block - 1)` = 2,051 (`qsa.hpp` 127-135), and the chunk-pass microbench at width 2,051, n_q 3
is 58.9-59.9 us a call (`bench/results/2026-11-09-bitdecoding-ab.txt` in 8860719dd) x 12 layers = 0.71 ms, which equals the short profile (0.74). Bytes are 26.0 MB a row, about 0.1 ms for a 3-row window at
800 GB/s, so the kernel is latency-bound (the commit says the same). Per window row the edit case (1.00 ms / 4.94 rows = 0.20) is not above the short case (0.74 / 2.97 = 0.25). That argument is sound for the attention kernel.

What it leaves out (and what "at 32K / 131K / 262K alike" glosses):
- Measured beyond the edit case: decode tok/s only. A 36,124-token prompt (A/B in 8860719dd): window 24.23 / 27.48 ms for the base arms, no stage split. Decode after a 13.8K prompt equals decode after 2K (`2026-11-09-decode-curve`).
  Nothing at 64K or 128K on Flash-Next: A49 (`longdec64k`, `longdec128k`) has not run.
- The indexer reads the whole pooled-key cache every row: 128 B per cell per layer, 12 layers (`qsa.hpp` 77-85). That is 201 MB a row at 128K, 8 times the 26 MB the attention reads. The measured
  `scores+topk` is 0.34 ms at a context of a few hundred cells, where its bytes are about nothing, so 0.34 ms is fixed cost (top-k, launches) and the bytes come on top.
- kv-resolve (0.32 ms with everything resident) turns into PCIe traffic beyond `--kv-resident 32768`: a page is 4 cells, 4,224 B a layer (`kv_stream.hpp`); with 75% of the cells non-resident at 128K, a row
  that missed everything would pull 19.5 MB, about 1.6 ms at the 12.2 GB/s the repo measures on PCIe 3.0 (1.4 MB promotion in 115 us). The real miss rate depends on how stable the top-k is from step to step;
  `KvStreamMap.ctl` has u64 miss and lookup counters (`kv_stream.hpp`), so A49 can log it.

KV-related reads in the window, 3 rows (derived; indexer = 0.34 ms fixed + its bytes at 800 GB/s for 3 rows; resolve held at 0.32 ms; PCIe misses excluded):

| Context | attention | indexer scores+top-k | kv-resolve | total | share of 24.8 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| short case (measured) | 0.74 | 0.34 | 0.32 | 1.40 | 5.6% |
| 12K | 0.74 | 0.41 | 0.32 | 1.47 | 5.9% |
| 32K | 0.74 | 0.53 | 0.32 | 1.59 | 6.4% |
| 64K | 0.74 | 0.72 | 0.32 | 1.78 | 7.2% |
| 128K | 0.74 | 1.09 | 0.32 | 2.15 | 8.7% |
| 262K | 0.74 | 1.85 | 0.32 | 2.91 | 11.7% |

By bytes the same picture: per token the model reads about 3.2 GB (GGUF table, `bench/results/2026-11-10-offline/gguf_table.json`: attention/SSM 2.07 GB, shared experts 0.13, routers 0.13,
about 0.73 GB of routed experts for 10 a layer x 48 layers, plus the head) and QSA + indexer add 227 MB at 128K, about 7%. So "3.0%" is solid for the attention kernel, not for the KV-related share,
which is 6-9% at 32-128K (derived) and still under the plan's own 15% parking line. The one term that could push it over is the unmeasured PCIe miss cost.

## 2. KV bytes at 128K on this hardware

Model (`qsa.hpp` header, `flashnext-iq2_xs-fast-hip.json`: `--max-context 131072 --kv int8 --kv-resident 32768`): 48 layers, 12 QSA layers, 24 query / 2 KV heads, head dim 256,
idx 4 heads x 128, idx_block 4, top-k 2,048. Page = 4 cells x 2 heads x (K + V) = 4,224 B in int8 with fp16 scales, so 12,672 B a token over 12 layers.
Hardware: RX 7900 XT 20 GB (800 GB/s spec, the figure the plan uses), Ryzen 7 5700G, root port PCIe 3.0 x16 (`gpu_sync.md:24`), about 57 GB RAM (`pancake-docs/README.md`).

| Read at 128K, per decode row | Bytes | At 800 GB/s | Note |
| --- | ---: | ---: | --- |
| Whole KV cache, int8 + scales (dense attention, hypothetical) | 1.66 GB | 2.1 ms | fp16 would be 3.2 GB |
| ... of which beyond the 32,768 resident cells | 1.25 GB | 104 ms over PCIe at 12.2 GB/s | why dense attention cannot work here past 32K |
| QSA attention, 2,051 cells x 12 layers | 26.0 MB (1.56%) | 0.03 ms (0.1 ms for 3 rows) | what runs |
| QSA indexer, 32,768 pooled blocks x 512 B x 12 layers | 201 MB | 0.25 ms (0.75 ms for 3 rows) | the context-growing term |
| Quest page metadata if it replaced the indexer (min + max, 16-cell pages, fp16) | 201 MB | same | no saving |
| Llama-3-8B fp16 for comparison (the papers' typical model) | 17.2 GB | - | about 10x our bytes a token |

The KV cache of this model is 12.7 KB a token, a tenth of the models the five papers measured, and the part that is read is already 1.6% of it.

Qwen3.6-35B-A3B on the q36 engine (`q36_engine.cu`, 10 attention layers, 2 KV heads, head dim 256, FP32): 40,960 B a token, 5.37 GB at 128K, 6.7-8.9 ms for one pass at 800-600 GB/s,
and `k_attn_part` re-reads for 8 query heads with no GQA sharing (MOD_PLAN: about 40 GB a token before L2/MALL). Here KV reads are the decode cost, as the
fork logs show. This is the one setting where the papers' premise holds.

## 3. What each paper measured

| Paper | Models (attention) | Hardware | Batch | Context | Attention / KV share of decode in the paper | Headline | Quality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Quest 2406.10774 | LongChat-7B-32k, Yarn-Llama-2-7B-128k, Llama-2-7B (MHA implied by 32 heads; never named) | RTX 4090 (kernel), Ada 6000 (e2e) | 1 ("single-batch") | e2e 32K, budget 2,048; passkey 10K and 100K | KV load 53% of a decode step at 32K; 16 GB cache, at least 11 ms to read (RTX 4090) | self-attention 7.03x vs FlashInfer; e2e 2.23x with 4-bit weights, 1.74x FP16; first 2 layers dense | passkey 100% at 2,048 on 100K; LongBench near full at 1/5-1/10 sparsity |
| Twilight 2502.02770 | LongChat-7B (MHA), Llama-2-7B (MHA), Llama-3.1-8B (GQA) | A100 | e2e 32-256; breakdown at 64 | 10-30K prompts (breakdown 32K) | none reported (attention breakdown only) | self-attention up to 15.8x vs FA2, 2.4x over FlashInfer, 1.4x over Quest; TPOT up to 3.9x vs FlashInfer, 1.35x vs Quest | Llama-3.1-8B LongBench -0.7%; RULER -0.8% |
| LServe 2502.14866 | Llama-3-8B (GQA), Llama-2-7B, Minitron-4B, R1-Distill-Llama-8B; no MoE | 8x A100 80GB, one L40S | decode Fig 16 and Table 7: 1; Table 1 uses 32; Fig 10 not stated | 4K-320K | attention at least 50% above 64K, 75% at 128K (Fig 2, batch 1, prefill and decode together) | prefill up to 2.9x; decode 1.3-2.1x (1.09x at 64K, 1.28x at 128K, 1.82x at 320K in Table 7); selector 0.24 ms vs 0.12 ms kernel at 128K | LongBench -0.3; RULER 256K -3.7 at budget 4,096 |
| ShadowKV 2410.21465 | Llama-3-8B-1M, GLM-4-9B-1M, Llama-3.1-8B, Yi-9B-200K (GQA, 8 or 4 KV heads) | A100; PCIe 31.5 GB/s assumed | max that fits: 4 full vs 24 ShadowKV at 122K; smallest tested 2 | 60K-512K | none; decode per block at 24 x 128K: fetch V 1.66 ms vs attention 0.2 ms | 3.04x throughput = 6x batch; at batch 2 equal to full attention at 60K and 122K, 1.2x at 244K | RULER -2.0 worst case (Llama-3.1-8B) |
| DuoAttention 2410.10819 | Llama-2-7B (MHA), Llama-3-8B (GQA), Llama-3-70B, Mistral-7B; no MoE | A100 (80G for 3.3M tokens) | not stated | up to 3.3M; prefill 100K / 320K | none | decode up to 2.18x MHA, 1.50x GQA at an unstated context; 50% of heads kept full for GQA; gates trained 2,000 steps on 8x A100 | LongBench -3.0 on Llama-2-7B-32K at 25%; about 0 on Llama-3-8B-1M at 50% |

Cross-check of the share: LServe's vLLM latency goes 12.51 ms (64K) to 16.34 ms (128K), a slope of 3.8 ms per 64K tokens, so the KV read is about 7.7 ms of 16.3 ms (47%) at
128K for a model whose cache is 17 GB. Ours is 2.15 ms of a 24.8 ms window (9%, derived) with a cache a tenth the size and a step time set by experts, not by the cache.

## 4. Does any of them pay at 64K-128K here? (gate: at least 2% of modeled decode time)

| Method | What it could touch on Flash-Next | Upper bound at 128K (derived) | Verdict |
| --- | --- | --- | --- |
| Quest | replace the QSA indexer with page min/max: same 201 MB, a different (untrained) selection | 0 | no |
| Twilight | prune the 2,051 cells to a few hundred; needs an INT4 K copy (1/8 of an fp16 cache by the paper; about 0.4 GB on our int8 cache at 128K) and an SpGEMV estimate; its gain is over a 2K-8K base budget at batch 32+ | the 0.74 ms kernel is latency-bound, so at most about 0.25 ms = 1.0% | no (below the gate, lossy, selection trained at 2,048) |
| LServe | streaming heads (50%) and a reusable page selector (reuse every 4 steps, accuracy holds to 8) | streaming heads: bytes are 0.1 ms, under 0.5%. Selector reuse is the only idea with a size: skipping scores + top-k on 3 of 4 steps saves at most 0.82 ms = 3.3% at 128K, 0.54 ms = 2.2% at 64K, 0.40 ms = 1.6% at 32K (derived from section 1) | do not build as published; reuse of the QSA selection is a new, lossy idea (the ids are an exact contract in the engine, `qsa.hpp` header) for A49 to price |
| ShadowKV | low-rank keys on the GPU, values fetched from host | negative: V over PCIe 3.0 (12 GB/s here, 31.5 assumed) at batch 1; its gain is batch capacity | no |
| DuoAttention | static retrieval / streaming heads | bytes 0.1 ms; VRAM saved about 0.2 GB at most (about 150 experts), and the plan already found that VRAM does not buy decode here | no |

The ceiling for everything together is the whole KV-related row of section 1 (2.15 ms, 8.7% at 128K), and the part a method could remove without replacing QSA is the
attention kernel (0.74 ms). Cheaper context-growing levers exist and are not these papers: a fp16 or int8 pooled-key cache for the indexer (MOD_PLAN's own "long-context decode lever";
half or a quarter of 0.75 ms = 1.5-2.3% at 128K, lossy and needing a selection-agreement test) and a PCIe miss counter from A49.

## 5. FreeToken 2608.16157, section 3.1 (A11)

Outline of the body: 1 Introduction; 2 Challenges (2.1 prefill, 2.2 decode, 2.3 resource management); 3 Design (3.1 Prefill Codesign: Pipelined Loading and Semantic-Aware State Caching,
3.2 Decode Codesign, 3.3 Elastic Memory Management); 4 Implementation; 5 Evaluation (5.1 setup, 5.2 main results, 5.3 breakdown); 6 Related Work; 7 Conclusion.
- Checkpointed: the complete recurrent state of the linear-attention layers (Gated DeltaNet in Qwen3.6-35B-A3B, Kimi Delta Attention); full-attention KV is separate, in a radix prefix tree.
- When: at special-token boundaries (thinking segments, tool calls and outputs, turns), because harnesses edit context there. Restore: deepest checkpoint that survives the edit; only the new suffix is re-read.
- Slots: LRU, independent of the KV pool; a checkpoint is "as much memory as the KV cache of hundreds of tokens", "only a few can be held"; no byte size or slot count.
- Measured: nothing about the checkpoints. The only ablation is pipelined loading (double buffering off: -19% / -25% / -26% throughput at 4K / 8K / 16K on an RTX 5090, Qwen3.6-35B BF16).
  End to end: decode 77-83 tok/s on Qwen3.6-35B-A3B (22-25 on DeepSeek-V4-Flash), 1.5-2.3x the best baseline (llama.cpp, Ollama, KTransformers, MoE-Infinity), worst turn under 44 s;
  these bundle expert residency, the CPU-GPU split and the state cache, and no number isolates the checkpoints.
- Match with the code: `STRATA_CACHE_TOOL_BOUNDARY` / `=2` implements the same marks; the source supports the mechanism and gives no gain. percache's "keep off" (about 0 in steady growth, upper bound 0.5%) stands.

## 6. MoE-Lightning 2411.11217

- Searched the body for stale, staleness, drop, dropping, discard, evict, cancel, outdated: none. The one near hit is request admission (Algorithm 2, `aborted_requests` carried to the next batch), unrelated to weights.
- What it has: CGOPipe, a fixed CPU-GPU-I/O launch order (Algorithm 1) where the next layer's weights move in n pages (n = number of micro-batches; an expert is two pages, section 4.1 and App A.1);
  the policy (batch N, micro-batch, placement ratios) comes from the HRM model solved offline as a MILP. Static, no reaction to the router, no cancelled transfer.
- Setting: offline throughput inference, Mixtral 8x7B on a 16 GB T4 / 24 GB L4, attention on the CPU, N = 504 and micro-batch 36 in the MTBench example; up to 10.3x throughput. Section 3.3 says latency cases with one or two prompts are better served by static placement.
- Our tier: single stream, ForesightSwap is off in the presets, a promotion is 115 us on the link (about 13.8 swaps a window = 1.6 ms, 6% of a window), the async tier lost 9.6% in A8. The origin note (`pcie_transfers.md:101`) credits cancellation to ProMoE and
  only the 256 KB chunking to MoE-Lightning; MOD_PLAN line 216 / 1072 / 1915 merged the two. ProMoE's own body is still a listed debt.

## 7. Phase 15 fork commit hashes

Method: every hex token of 7-40 characters (with a digit and a letter) in the Phase 15 section of `MOD_PLAN.md` (68 hashes), in `research_notes/Speed ideas from all forks/*.md` and in `reports/Speed ideas from all forks.md`
(768 distinct in all), put through `git cat-file --batch-check` (the batch form of `git cat-file -t`, accepts abbreviations; no "ambiguous" result) in the main checkout and the fork dirs: Strata, StrataFast, FastStrata, PolyStrata, Speedy, colibri,
colibri-fork, llamAmpere, rocWMMA, plus aviary-hy3, basalt, Strata-rdna3. The rest were checked against the fork knowledge base `data/fork-kb/forks.sqlite.gz` (commits table: sha, repo, date, message). Scratch files were kept outside the repo.

| Phase 15 hashes (68) | Count | Where |
| --- | ---: | --- |
| Our own commits, `commit` in Strata | 8 | `d5b58f669`, `89da64b93`, `c19f82b53`, `8860719dd`, `6c68d7586`, `146df3c46`, `f15ad2b51`, `dd8bf619b` |
| Resolve only in PolyStrata (a simplexai-labs clone; the commits are RandoTeam/4gbVramStrata's) | 9 | `79b1e756`, `67f60c12`, `79acffe5`, `7fb9d79d`, `ac6402df`, `bee07744`, `44b58c6a`, `4636dbe3`, `6a991bce` |
| Resolve only in StrataFast (eddoursul's clone; the commits are eddoursul/Lamina's) | 3 | `13d90a38ec`, `ab808b133e`, `83f40aee62` |
| Resolve only in llamAmpere | 1 | `3cf03257f` (llama.cpp "CUDA: enable sparse fa for qwen4", the pin MOD_PLAN names) |
| Not in any local dir, present in the fork KB under the named author | 46 | TwoAnts, svgop, mandoof1, jmnargi, riceharvest, imanu86, android-imdad, Ben1078, zsariboga, Asofwar, yqYo1, RX5950XT, lighttransport, dongnh311, storm-ace, AdrianBM96, KhaledElzarw, unimatrix099, maoyufeng1985, bsorensen110 and the rest |
| Neither | 1 | `b9820dd56` |

FastStrata, colibri, colibri-fork and rocWMMA hold none of the cited hashes; Speedy, basalt and Strata-rdna3 hold only the history we share with them. Of the 46 KB-only hashes, the owner's name is in the
plan row for 39; for the other 7 (`9af8f1fa14`, `4d462ccff3`, `0af4328ec8`, `691fa9d191`, `287cd207ad` KhaledElzarw; `4638045c` RandoTeam; `2f7020f2` maoyufeng1985) it is in the section heading or the next line. Where the row names an env var, the commit message contains it (`STRATA_PARK_TRUNCATE`, `STRATA_VERIFY_ECAP`,
`STRATA_SPEC_PROB`, `STRATA_Q2_LEGACY`, `STRATA_SPEC_CUM`, `STRATA_ROUTE_FAST`, `STRATA_ADAPT_PROMPT`, `STRATA_ATTN_WINDOW`, `STRATA_ROUTE_PRIOR_PROTECT`, `STRATA_FETCH_ADMIT`, `STRATA_FATTN`, `STRATA_FDEC`, `STRATA_HIP_WMMA_Q4`).

Findings to fix:
1. `b9820dd56` (MOD_PLAN line 2307, "cherry-pick of `b9820dd56`"): no local object and not in the KB. The text comes from the cherry-pick trailer of bsorensen110's `19fc0e48` (full id `b9820dd56755e8a18f0b879eedcf50f322cbe854`), so it is a real id on a branch no scan saw. Write "source commit not in the fork KB".
2. F13 row cites lighttransport `2ea0c1e042` for "route-prior guards": that commit is "residency-aware pruning and cache-aware substitution", whose own message says they "do not beat uniform top-3". It exists, but it is not a route-prior guard; the guards are zsariboga `8d7042ad97` and mandoof1 `3e0489e118`. Drop or reword.
3. `MOD_PLAN.md` lines 1985-1988 are F12, F13, F12, F13: each row is in the table twice (1985 and 1988 plain, 1986 and 1987 with the "coded" note). Drop the plain copies.
4. Outside Phase 15 but in the same notes: `research_notes/Speed ideas from all forks/attention_kv_quant.md:382` links Cudila `911187e2` with the full id `911187e2b5f5ad0cdc0b5ff6e08bc74a7c66d78c`; the KB has `911187e249dfba1a61c66dc3c4beaf9b307ab32f`
   ("strata-nemotron 3/5"). The first 8 characters agree, the rest do not: that URL 404s. Of 377 commit URLs in these notes, this is the only sha missing from the KB. 65 URLs use `Niko1221/Strata/commit/<fork sha>`: they resolve through the fork network only, they are not on an upstream branch.
5. `0a20260930` (`server_system.md:261`) is not a hash, it is part of the ROCm nightly name `10.2.0a20260930`.

## What was not verified

- The fork commit bodies (diffs) were not read, only that the commit exists and what its message says; no claim that a diff does what the plan row says was checked.
- Paper figures (Quest Fig 10, Twilight Fig 7/8/10, LServe Fig 2/10) are not in the extracted text; the batch size of LServe Table 7 is "unit" only through the Figure 16 caption.
- The KV-related table in section 1 is arithmetic on a short-case and an edit-case profile (no stage profile of Flash-Next exists above about 2.5K context); the indexer's real achieved bandwidth and top-k scaling are unknown, and the PCIe miss term is excluded. A49 (`longdec64k`, `longdec128k`, `STRATA_DECODE_TIMING=1`, with the kv-stream miss counters logged) replaces it with data.
- ProMoE, HOBBIT, DALI bodies (other debts in MOD_PLAN line 1602) were not in scope.
