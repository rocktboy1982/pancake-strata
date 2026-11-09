# 🥞 pancake-strata

**A fast local LLM engine, built one layer at a time.**

Why *pancake*? A good stack of pancakes is built one at a time: each one goes on top of the last, warm, and the stack
only gets better as it grows. This project is built the same way. It starts from **[Strata](https://github.com/Niko1221/Strata)**,
a fast mixture-of-experts engine for home GPUs. On top of it go the newest speed techniques the open-source
community has published, one after another: an engine from FastStrata, kernels from llama.cpp, ideas from basalt,
colibri, SGLang and others, and our own measured kernels. Every layer is checked before it goes on the stack: it has
to be **faster on a real benchmark** and keep the **same answers**. The result is a full load of pancakes: a stack of
the latest LLM inference technologies, each adding speed.

The target is single-user agent serving (long, growing prompts and tool calls) on ordinary hardware: an
**RTX 3060 12 GB** (CUDA) and an **RX 7900 XT 20 GB** (HIP), a Ryzen 7 5700G and about 57 GB of RAM.

> The original Strata README is kept as [README.strata.md](README.strata.md); its translations are the `README.*.md` files.

## Results so far

Qwen3.6-35B-A3B (UD-Q4_K_XL with its MTP layer) on an RTX 3060 12 GB, greedy, cold runs:

| | Strata (`strata`) | pancake-strata (`strata-poly`) | Gain |
| --- | ---: | ---: | ---: |
| Short chat, writing speed | 64.6 tok/s | **100.7 tok/s** | x1.56 |
| Code edit (5.3K prompt), writing speed | 40.1 tok/s | **151.1 tok/s** | x3.8 |
| 42K-token prompt: reading | 42.1 s | **32.9 s** | x1.28 |
| 42K-token prompt: writing speed after it | 9.7 tok/s | **71.7 tok/s** | x7.4 |
| Largest context on the 12 GB card | 64K (the 131K setup default runs out of VRAM) | **131K** | |

Every test, profile and dead end: [docs/PANCAKE_FINDINGS.md](docs/PANCAKE_FINDINGS.md). The plan:
[MOD_PLAN.md](MOD_PLAN.md).

## What pancake-strata does beyond Strata

Everything below is on top of upstream Strata v0.2.12 and measured on the RTX 3060 12 GB unless the row says RX 7900 XT
(numbers and method: [docs/PANCAKE_FINDINGS.md](docs/PANCAKE_FINDINGS.md), MOD_PLAN phase 11). "Exact" means the same tokens as before, checked.

| | Strata | pancake-strata |
| --- | --- | --- |
| **Qwen3.6-35B-A3B engine** | `strata` (fp32 KV cache) | adds `strata-poly` (FastStrata's engine): **x1.6 short answers, x3.8 code edits, x7.4 writing after 42K tokens** |
| **Context on a 12 GB card (Qwen3.6)** | 64K (setup's 131K runs out of VRAM) | **131K** (Q8 KV cache that grows with the conversation) |
| **Prompt reading (Qwen3.6)** | - | **9-13% faster, exact**: DeltaNet recurrence 4 columns a warp (x1.4), register-tiled router GEMM (x2.0-2.3), attention softmax read once |
| **Output head** | `native_mmvq` | **a specialized exact kernel per head format** of the Flash-Next files - IQ4_XS (IQ2_XS), Q5_K (Q2_0, IQ3_XXS; basalt's), Q6_K (IQ3_S, Coder), Q4_K (UD-IQ1_M): one warp a row, activations in shared memory, x1.02-1.27 at 1-2 and 5-8 columns |
| **MTP draft head pick** | GEMV + argmax + probability | **basalt's 2-bit two-stage pick, per head format** (IQ4_XS, Q5_K, Q6_K, Q4_K): x1.6-2.3 against the GEMV alone, the exact argmax 200/200 in tests; end to end the same tokens and drafts accepted, +3-5% decode (noisy run) |
| **Default quality** | expert tail skip on (`STRATA_ROUTE_TAIL_SKIP=7`, answers change) | **no expert tail skip**; the default fast preset's switches passed the token-forcing check (within the engine's own run-to-run noise), `exact` is bit-for-bit; the tail skip and other lossy switches are opt-in |
| **Ready presets** | setup's config | `./run-pancake.sh` (fast, the default: Speedy's switches, +13% writing after 42K) and `./run-pancake.sh exact`; `*-hip` presets on the RX 7900 XT |
| **MTP draft vocabulary** | 106K tokens with CJK | **English/code subset by default** (+5% decode, ~215 MiB more VRAM for experts); Strata-rdna3's 59.6K list included |
| **Upstream PRs not in Strata** | - | `STRATA_VERIFY_DEFER_CPU` (#2689/#2690, opt-in, lossy; non-finite logits on HIP traced to `STRATA_ROUTE_PUBLISH`, now off under deferral (d5b58f669); the overlap model prices deferral at +33% on edits, not yet measured) |
| **Expert cache admission (RX 7900 XT)** | first-miss LRU churn | **admit on the k-th miss** (`kmiss_tier.hpp`, `STRATA_ADAPT_KMISS`, items 14/27/30): a miss is promoted the second time it is needed, with Belady- and frequency-aware admission, instead of churning the tier on every first miss; the time-ceiling simulator prices online 2nd-miss admission at +5-16% of CPU-expert time saved on code edits (oracle ceiling +27-48%), model, A/B queued |
| **PCIe split and decode tuning (RX 7900 XT)** | stock PcieAuto share | `--pcie-frac 0` and the tuned `*-hip` presets: +6% short chat (111.7 -> 118.4 tok/s) and +13% code edits (92.5 -> 104.3 tok/s) measured on the RX 7900 XT; the `turbo-hip` preset adds +30-50% decode (111.7 -> 149-168 tok/s), lossy (2.9% token disagreement), quality gate pending |
| **Flash-Next expert routing** | - | GPU router lookahead on by default (CPU router cost 30 -> 0.6 ms a window, +10% decode on mmap'd experts) and a learned expert profile in the IQ2_XS presets (+6.7% decode, every run faster) |
| **Measurement tooling** | bench harness | **20 tools under `tools/pancake/`**: rocprofv3 kernel profiling (`gpu_profile.py`), the time-budget ceiling simulator (`time_ceiling.py`), paired load-gated A/B (`abtest.py`), token-forcing quality gates (`quality_gate.py`), regression watch (`regression_watch.py`), decode-curve diagnostics (`decode_curve.py`), Belady ceiling replays (`belady_ceiling.py`), expert-routing traces (`conv_trace.py`) |
| **Server** | - | sends `turn_at` to `strata-poly`, so a chat's next turn resumes from the cache |
| **Agent robustness** | #606 one-token stop, opt-in #728 | **Project Maya's loop guard** (MIT): exact loops broken as they start (drop + short penalty, close the thinking, stop), repeated tool calls noted then blocked, on by default; tool calls that do not parse never fail or swallow the answer; opt-in soft thinking-budget close at a sentence end |
| **Thinking caps per effort level** | one `reasoning_budget_tokens` for every request | `reasoning_budget_by_effort` (opt-in, 5dd620fb9): a thinking cap per effort level (s1 budget forcing), a request's own budget still wins; not in any preset yet (opencode sends `xhigh` by default) |
| **Route prior graded by draft depth** | - | `STRATA_ROUTE_PRIOR_DEPTH=a` (item 23, AcceptMoE-style, opt-in, lossy; 4f56ce535), quality gate pending |
| **Side branches (on branches, not merged)** | - | `feat/disco`: DISCO, a learned stop rule for the MTP draft chain (opt-in), untested on the GPU. `feat/contextpilot`: ContextPilot block dedup + cache-aware admission; dedup measured 1.6% savings on 40 real opencode sessions, keep off. `feat/bitdecoding-hip`: `STRATA_ATTN_WMMA` decode attention on RDNA3 matrix cores; attention is only ~3% of decode, ~2-3% gain unconfirmed. `feat/bfla-hip`: BFLA block-sparse long-prompt attention plus a fused RDNA3 flash-attention kernel on the `qwen35` engine (Qwen3.6/3.8-35B): one 32K attention layer 1238.7 ms (hipBLAS path) -> 581.1 ms exact (`STRATA_BFLA=dense`) -> 500.2 ms sparse (`STRATA_BFLA=1`, lossy); at 64K 5224 -> 2321 (dense) -> 1090 ms (sparse, 4.8x); at 128K 28287 -> 9553 (dense, 3.0x) -> 3335 ms (sparse, 8.5x); Flash-Next keeps its QSA layers (end-to-end on Qwen3.6 UD-Q4_K_XL at 128K: 386.9 off -> 771.2 dense, 2.0x -> 1219.8 sparse tok/s, 3.15x, needle-recall top-1 48/48; Flash-Next 131K tile-shared QSA overlap measured: 42-74% shared cells, 229.1 tok/s prefill) |
| **Proof** | - | bench harness (`tools/pancake/bench_engines.py`, every run kept as JSON), bitwise kernel tests (`poly_gdn_cols_test`, `poly_gemm_small_test`, `q5k_head_test`, `iq4xs_head_test`), and a findings log with every test, profile and rejected idea |
| **RX 7900 XT (HIP)** | gfx1100 builds | Strata-rdna3's RDNA3 kernels and fusions ported (branch `port/rdna3-decode`), `*-hip` presets: ~112 tok/s decode vs Strata-rdna3 109-119, Avalon 95-100, stock Strata 75-77 (MOD_PLAN phase 11) |
| **Ahead** | - | MOD_PLAN phase 13 by measured gain: CPU-expert time first (expert deferral on HIP, time-based cache simulator), wiring pancake to opencode, Qwen 4 plan (Flash-Next is its official architecture preview); then phase 14's processes: nightly A/B, quality gates, self-tuning from the owner's own sessions |

## How pancake-strata is built: the process

Every layer on the stack goes through the same four steps. No switch goes in on a hunch: it is priced first, in
milliseconds of the real decode window, and kept only if it is faster and just as exact.

1. **Broad survey and deep research, before any code.** Upstream Strata and llama.cpp PRs, every community fork
   (FastStrata, PolyStrata, Strata-rdna3, Speedy, StrataFast, basalt, colibri, Aviary, llamAmpere, Project Maya,
   gpu-unlock), seven global university-paper sweeps (Stanford; MIT/Harvard; the rest of the US; Europe; Asia; Canada;
   Pacific/Africa/Latin America), the 2026 LLM runtime architectures, MoE engine speed-up deep dives (llama.cpp,
   ik_llama.cpp, KTransformers, SGLang, vLLM, TensorRT-LLM, FlashInfer, BitDecoding), hardware data-path and consumer
   platform limits, and real `opencode` session telemetry: 2,063 Flash-Next turns analyzed (served by the old
   Strata/Avalon backend), turn median 6.2 s, p90 50 s, p99 224 s, about 18% of turn time spent re-reading prompts the
   cache already had.
2. **Time-budget profiling before coding.** On the RX 7900 XT the actual decode window was measured and decomposed:
   37.4 ms on a code edit (GPU floor 15.8 ms, CPU expert misses 17.4 ms, MTP draft 2.5 ms) and 24.8 ms on short chat
   (GPU 15.9, CPU 5.2, draft 2.2). The CPU Q2_0 expert kernel already runs at ~80% of the 33 GB/s DDR4, and the GPU
   spends 38% of its kernel time in `combine_cpu`, almost all of it waiting for CPU rows. That proved MoE decode on
   consumer hardware is a scheduling, overlap and cache-admission problem, not a raw CPU-kernel-bandwidth problem, and
   it set the ~20 ms perfect-overlap floor every candidate is priced against.
3. **Expected-gain planning and the Not-Adopted Ledger** ([MOD_PLAN.md](MOD_PLAN.md)). Every candidate is ranked into
   Tiers A-D and substitute groups S1-S7 by measured milliseconds saved per decode window (gains inside a group do not
   add), combined into items 63-73 where a combination beats its parts, and everything looked at and not taken goes
   into the Not-Adopted Ledger with the exact reason and a reopen trigger.
4. **Isolated prototyping behind a two-stage gate.** Heavy or experimental features grow on their own branches
   (`feat/bfla-hip`, `feat/disco`, `feat/contextpilot`, `feat/bitdecoding-hip`, `exp/expert-prefetch`). Kernels are
   verified first with bitwise unit tests (`poly_gdn_cols_test`, `poly_gemm_small_test`, `q5k_head_test`,
   `iq4xs_head_test`), then paired cold A/B benchmarks (`bench_engines.py`, `abtest.py`) and token-forcing quality
   checks (`quality_gate.py`, `regression_watch.py`) decide. A layer joins the stack only if it is faster on real
   workloads and keeps the answers exact by default; lossy switches stay strictly opt-in.

## Daily use: Qwen3.8-Flash-Next

The model in daily use is **Qwen3.8-Flash-Next**, the official preview of the Qwen 4 architecture: a 125B MoE plus a
51B hashed n-gram (PLE) table plus a ~4B MTP draft layer, **~176.9B parameters in all** (ISTA-DASLab's model cards;
earlier notes here said "125B"), ~6B active per token. Each block is 3 Gated DeltaNet layers + 1 QSA attention layer;
each MoE layer has 512 experts, 10 routed + 1 shared per token. The files are ISTA-DASLab's GSQ-RCO GGUFs, run on
Strata's main engine with the RAM-resident expert arena (RX 7900 XT 20 GB now, RTX 3060 12 GB before; 57 GB RAM):

- **Q2_0** (shard 1 37.6 GB): every routed expert tensor Q2_0. DASLab's card: **93.8 vs 70.3 tok/s decode** for Q2_0 vs
  IQ2_XS in llama.cpp, task average 89.07 vs 89.16 (**-0.09**). [MOD_PLAN.md](MOD_PLAN.md) treats Q2_0 as the daily
  file (item 6 closed).
- **"IQ2_XS"**: the name is the bit budget, not the tensor type. The experts' gate/up are IQ2_S (34
  layers), IQ2_XXS (11) and IQ1_M (3), and every down projection is Q2_0 (640-wide rows rule out K- and I-quants). The 3
  IQ1_M layers take the generic ggml-cpu path on CPU misses (item 24).
- The n-gram table (shard 2, 28.8 GB IQ4_NL) is the same file for every variant and stays on the SSD.

**Known inconsistency:** `run-pancake.sh` still defaults to `iq2_xs` and its comment calls Q2_0 "a lower-quality quant"
(80.6% same top token as Q8_0, [docs/QUANTS.md](docs/QUANTS.md)); the plan and DASLab's task scores say Q2_0. Pass
`q2_0` explicitly until the script is changed. Pancake also listens on port 8080, while opencode points at 11437
(`strata.service` / `avalon.service`): it is not yet wired to opencode.

```sh
./run-pancake.sh               # Flash-Next "IQ2_XS" (the script's default), fast preset, English/code draft vocabulary
./run-pancake.sh q2_0          # Flash-Next Q2_0: the plan's daily file, the fastest Flash-Next file here
./run-pancake.sh iq2_xs exact  # the exact preset: the same answers as plain decoding (~13% slower after long prompts)
./run-pancake.sh qwen36        # Qwen3.6-35B-A3B UD-Q4_K_XL on strata-poly
./run-pancake.sh qwen36-q3     # Qwen3.6-35B-A3B UD-Q3_K_M on strata-poly: 150+ tok/s edits, 83 tok/s after 42K
./run-pancake.sh qwen38-35b    # Qwen3.8-35B-A3B-Distill Q4_K_M (empero; Qwen3.6 tuned on Qwen3.8 answers): ~98 / 152 / 74 tok/s
```

OpenAI API on `http://127.0.0.1:8080/v1` (Anthropic: `/v1/messages`). Measured on the RTX 3060 with "IQ2_XS" (cold runs,
2 rounds; the 7900 XT numbers are in MOD_PLAN phase 11, ~112-135 tok/s decode): short answers
~50-55 tok/s, code edits ~74 tok/s, after a 42K-token prompt 41-47 tok/s, a 42K prompt read in ~37-38 s. Upstream's
default expert tail skip would add 18-22% writing speed but changes answers: off here, an opt-in.
Qwen3.6-35B-A3B on `strata-poly` (below) remains available and is faster (100-150 tok/s) but a smaller model.

## Research and plan

- **Plan:** [MOD_PLAN.md](MOD_PLAN.md), items 1-73 across phases 0-14. Phase 11 (RX 7900 XT) holds the research index
  (one row per report, what each gave the plan) and items 13-62; the development plan is ordered by measured gain
  (phase 13, Tiers A-D, substitute groups S1-S7, combined items 63-73); the **not-adopted ledger** (section
  "Not adopted (ledger, 2026-11-09)") records everything looked at and not taken, with the reason and a reopen trigger.
- **Status:** [PANCAKE_STATUS.md](PANCAKE_STATUS.md), one row per item: how much of the code exists and builds (Dev)
  and how far it is from decided (Test), ordered by phase 13's expected gain.
- **Reports** ([reports/](reports/), 14, copies of the owner's research series; most are in Romanian):
  - Engine and platform: [Pancake MoE engine speedup](reports/Pancake%20MoE%20engine%20speedup.md),
    [LLM runtime architectures 2026](reports/LLM%20runtime%20architectures%202026.md),
    [pain points and solutions](reports/Pancake%20pain%20points%20%C8%99i%20solu%C8%9Bii.md),
    [technology families](reports/Tehnologii%20grupate%20pentru%20pancake.md),
    [hardware data path](reports/Tehnologii%20drum%20de%20date%20hardware.md) and
    [consumer platform capabilities](reports/Capabilit%C4%83%C8%9Bi%20platforme%20consumer.md).
  - Upstream survey: [Avalon upgrades from upstream projects](reports/Avalon%20upgrades%20from%20upstream%20projects.md).
  - University papers, seven sweeps: [Stanford](reports/Lucr%C4%83ri%20Stanford%20utile%20pentru%20Avalon.md),
    [MIT/Harvard](reports/Lucr%C4%83ri%20MIT%20Harvard%20pentru%20Avalon.md),
    [rest of the US](reports/Lucr%C4%83ri%20universit%C4%83%C8%9Bi%20SUA%20pentru%20Avalon.md),
    [Europe](reports/Lucr%C4%83ri%20Europa%20pentru%20Avalon.md),
    [Asia](reports/Lucr%C4%83ri%20Asia%20pentru%20Avalon.md),
    [Canada](reports/Lucr%C4%83ri%20Canada%20pentru%20Avalon.md) and
    [Pacific/Africa/Latin America](reports/Lucr%C4%83ri%20Pacific%20Africa%20America%20Latin%C4%83.md).
- **Notes** ([research_notes/](research_notes/), in English, the per-topic layer under the reports): the six MoE
  engine-speedup deep dives ([cpu_expert_compute](research_notes/Pancake%20MoE%20engine%20speedup/cpu_expert_compute.md),
  [engines_labs_companies](research_notes/Pancake%20MoE%20engine%20speedup/engines_labs_companies.md),
  [expert_caching_prediction](research_notes/Pancake%20MoE%20engine%20speedup/expert_caching_prediction.md),
  [moe_speculative_decoding](research_notes/Pancake%20MoE%20engine%20speedup/moe_speculative_decoding.md),
  [rdna3_gpu_decode](research_notes/Pancake%20MoE%20engine%20speedup/rdna3_gpu_decode.md),
  [transfer_and_kv](research_notes/Pancake%20MoE%20engine%20speedup/transfer_and_kv.md)); the pain-point notes
  ([engine](research_notes/Pancake%20pain%20points%20%C8%99i%20solu%C8%9Bii/engine.md),
  [server_agent_ops](research_notes/Pancake%20pain%20points%20%C8%99i%20solu%C8%9Bii/server_agent_ops.md)); and eleven
  further folders, one per report above.
- **What changed in our understanding** (measured vs estimated, with files):
  [docs/PANCAKE_FINDINGS.md section 13](docs/PANCAKE_FINDINGS.md#13-research-series-2026-11-09). In short: a code-edit
  decode window is the plain sum of GPU, CPU-expert and draft time (a scheduling problem, not a bandwidth one), and
  pancake is not yet serving opencode.

## The layers of the stack, and what each does

### From FastStrata: the `strata-poly` engine

A second engine beside Strata's, tuned for Qwen3.6-35B-A3B on small NVIDIA cards and measured on an RTX 3060 Laptop.
Ported whole (`src/poly/`), with the kernels it needs merged into Strata's newer ones.

| Technology | What it does | Benefit |
| --- | --- | --- |
| Hybrid expert placement with swaps | Keeps the most-used experts in VRAM and computes the rest on the CPU; swaps experts in while it writes | Runs a 23 GB model on a 12 GB card |
| `--kv-q8` | The attention cache in 8 bits instead of 16 (11.7 KB a token), growing with the conversation | 131K context on 12 GB; more VRAM left for experts |
| `--attn-split` | Decode attention as one pass over the cache on the tensor cores (split-K, flash-decoding style) | Writing stays fast after long prompts (FastStrata: +30% at 37K) |
| `--graph-experts` | Each layer of a decode step recorded as one CUDA graph, card and CPU taking turns through pinned memory | Fewer launch costs: 7-10% less time a step |
| MTP drafting + verification windows | The model's own multi-token-prediction layer guesses 3 tokens ahead; one window verifies them | More tokens a step, same output |
| `--suffix-drafts` | Guesses from every earlier prompt and answer (SuffixDecoding), in a 160 MiB index | Repeated tasks and code edits written much faster |
| `--cpu-multi` | The CPU's K-quant experts read once for all tokens of a window that use them (AVX2) | CPU experts at 83-98% of RAM bandwidth |
| `--elastic-prefill`, `--lend-staging`, `--stage-next` | Prompt batches borrow VRAM from the expert cache, and the next layer's experts are copied while a layer runs | Faster prompt reading on a small card |
| `--conversation-disk`, conversation cache, turn checkpoints | Conversations parked in RAM or on disk, resumed at the last turn | An agent's next turn does not re-read its history |

### From Strata and llama.cpp / ggml

Strata gives the whole base: model loading, the OpenAI/Anthropic-compatible server, setup, the CUDA and HIP builds, the
Flash-Next and DeepSeek engines, and the multi-GPU, router and caching work. The prompt path multiplies the experts
with **llama.cpp's MMQ kernels** (ggml-cuda, quantized matrix products on int8 tensor cores), compiled from the
pinned llama.cpp checkout Strata already uses.

### Our own layers (pancake-strata)

| Technology | What it does | Benefit |
| --- | --- | --- |
| DeltaNet recurrence, 4 columns a warp | Each warp walks 4 state columns of a head through the prompt, sharing their loads; ~1 wave of blocks instead of ~5 on a 28-SM card | `gdn_step` x1.4, the same bits |
| Register-tiled small GEMM | The router and gate products computed 4x4 per thread from registers, each output's sum in the same order | Router x2.0-2.3, the same bits |
| Register softmax for prompt attention | Each thread keeps its scores in registers: a score row is read once, not twice | ~1.5% faster long reads, the same bits |
| Prompt reads, together | The two above | 9-12% faster prompt reading (16K: 10.7 -> 9.6 s, 42K: 35.8 -> 32.9 s) |
| `turn_at` in the server | The server tells `strata-poly` where the last chat turn starts | The next turn resumes from the cache |
| Bench harness `tools/pancake/bench_engines.py` | The same prompts through any engine, every run kept as JSON lines | Every layer is measured before it stays |
| Bitwise kernel tests | `poly_gdn_cols_test`, `poly_gemm_small_test` | A speedup only stays when it gives the same bits |

### Where the inspiration comes from

Three pillars, each with a different role:

1. **Strata and its community forks**, the code pancake starts from. Strata (`Niko1221/Strata`) is the base;
   FastStrata (`Jarson1408/FastStrata`) and PolyStrata (`simplexai-labs/PolyStrata`) give the `strata-poly` engine;
   Strata-rdna3 (`nicejunjie/Strata-rdna3`) the RDNA3 kernels; Speedy (`DevDock-AI/Speedy`), StrataFast
   (`eddoursul/StrataFast`), basalt (`jesdga95/basalt`), colibri (`JustVugg/colibri`), Aviary (`SensAI-PT/aviary-hy3`),
   llamAmpere (`JakeATX/llamAmpere`), Project Maya (`mw00/project-maya`) and gpu-unlock (`Zagitta/gpu-unlock`) are
   measured, studied or ported piece by piece (table below).
2. **Production inference engines and runtimes**, ideas more than code: `llama.cpp`/`ggml` (MMQ kernels, the CPU
   expert dot products), `ik_llama.cpp` (repacked MoE quants, fused MoE and DeltaNet ops), `KTransformers` (CPU-GPU
   hybrid MoE scheduling; its measured deferral ceiling, up to 1.45x, prices item 8), `SGLang` (overlap scheduler, tree
   drafting, HiCache, GPU grammar masks), `vLLM`, `TensorRT-LLM` (grammar inside speculative decoding, auto n-gram
   enablement), `FlashInfer` and `BitDecoding` (studied; BitDecoding's HIP port is opt-in on `feat/bitdecoding-hip`).
3. **Academic research and model labs**, the papers behind the mechanisms and the files in use: ISTA-DASLab (the
   GSQ-RCO Q2_0 / IQ2_XS GGUFs for Qwen3.8-Flash-Next), Unsloth (UD dynamic quants with the MTP layer), the Qwen team
   at Alibaba, and the research lines: **AcceptMoE** (depth-graded router prior, item 23), **Belady / k-th-miss cache
   admission** and the **MoE-Infinity / ProMoE / Pre-gated MoE / ExpertFlow** expert-caching and prefetching line,
   **Apto (2611.01859) and Apex (2610.23006)** (CPU-GPU hybrid expert serving and caching),
   **BFLA** (block-sparse flash attention on RDNA3), **DISCO** (learned draft stopping), **ContextPilot** (context-block
   dedup and cache-aware admission), **SuffixDecoding** (training-free drafts from prompt and history) and **s1**
   (reasoning budget forcing by effort level).

basalt (two-stage output head, grammar-constrained decoding, drafting), colibri (cache-aware routing, multitier
memory), SGLang (tree drafting, GPU grammar masks), TensorRT-LLM's blogs (grammar with speculative decoding),
Aviary and lilt's colibri fork: what was taken, planned or rejected, and why, is in
[docs/PANCAKE_FINDINGS.md](docs/PANCAKE_FINDINGS.md#5-tried-and-rejected-with-the-reason) and in MOD_PLAN's
Not-Adopted Ledger.

## Build and run

```sh
export PATH=/usr/local/cuda/bin:$PATH
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86-real
cmake --build build-cuda --target strata strata-poly -j8
./setup.sh --family qwen36 --model UD-Q4_K_XL --build --no-start   # model, tokenizer, a config
```

Then point the server at `strata-poly` with a config whose `"exe"` is `build-cuda/strata-poly` and whose `"args"` are
`--model <gguf> --max-context 131072 --threads <cores> --kv-q8 --attn-split --graph-experts --elastic-prefill 1024
--lend-staging --stage-next --suffix-drafts 256 --cpu-multi`, and start it with
`.venv/bin/python serve/server.py --engine strata --config <that config> --port 8091`.

## Thank you 🙏

This stack exists because these people shared their work openly. Thank you!

| Project | Who | License | What pancake-strata uses |
| --- | --- | --- | --- |
| **[Strata](https://github.com/Niko1221/Strata)** | Niko1221 and the Strata contributors | MIT | The base of everything: engines, server, setup, builds |
| **[FastStrata](https://github.com/Jarson1408/FastStrata)** | Jarson1408 | MIT | The `strata-poly` engine for Qwen3.6, its speed options and its measured optimization log |
| **[PolyStrata](https://github.com/simplexai-labs/PolyStrata)** | simplexai-labs | MIT | FastStrata's parent: the `src/poly` multi-architecture engine |
| **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)** | Georgi Gerganov and the ggml contributors | MIT | MMQ quantized matrix kernels, the GGUF format, the CPU expert dot products |
| **[basalt](https://github.com/jesdga95/basalt)** | Jesús Gulfo | MIT | The Q5_K head kernels and the 2-bit two-stage draft pick (`q5k_head.cu`, `head_proxy.cuh`), and their design for the IQ4_XS ports; grammar and drafting ideas (planned) |
| **[colibri](https://github.com/JustVugg/colibri)** | JustVugg and contributors | Apache-2.0 | Studied: cache-aware routing, multitier memory ideas |
| **[Aviary](https://github.com/SensAI-PT/aviary-hy3)** | SensAI-PT | Apache-2.0 | Studied: distributed expert placement |
| **[llamAmpere](https://github.com/JakeATX/llamAmpere)** | JakeATX | MIT | Studied: adaptive MTP depth, SJ-KVaRN KV cache, mb-MTP (candidates) |
| **[gpu-unlock](https://github.com/Zagitta/gpu-unlock)** | Zagitta | GPL-2.0 | Recommended (not bundled): full-rate FP16 tensor cores on GeForce, which speeds up attention |
| **[Strata-rdna3](https://github.com/nicejunjie/Strata-rdna3)** | nicejunjie | MIT | Studied: RDNA3 (7900 XT/XTX) kernels and tensor parallel, for the HIP half of the plan |
| **[Speedy](https://github.com/DevDock-AI/Speedy)** | DevDock-AI | MIT | Studied: RTX 3080 (sm_86) Flash-Next presets and the upstream PRs it measured |
| **[StrataFast](https://github.com/eddoursul/StrataFast)** | eddoursul | MIT | Studied: two-stage prompts, a self-trained draft layer |
| **[SGLang](https://github.com/sgl-project/sglang)** | the SGLang team | Apache-2.0 | Ideas: tree drafting, speculative sampling, GPU token masks (planned) |
| **[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)** | ikawrakow | MIT | Studied: repacked MoE quants, fused MoE and DeltaNet ops, DFlash 2 drafting for Flash-Next |
| **[KTransformers](https://github.com/kvcache-ai/ktransformers)** | the KTransformers team | Apache-2.0 | Studied: CPU-GPU hybrid MoE scheduling; its measured deferral ceiling (up to 1.45x) prices item 8 |
| **[vLLM](https://github.com/vllm-project/vllm)** | the vLLM team | Apache-2.0 | Studied: checked and rejected as a base engine here (keeps the model in VRAM); overlap-scheduler and KV-store ideas |
| **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)** | NVIDIA | Apache-2.0 | Ideas from the tech blogs: grammar inside speculative decoding, auto n-gram enablement |
| **[FlashInfer](https://github.com/flashinfer-ai/flashinfer)** | the FlashInfer team | Apache-2.0 | Studied: its ROCm port is CDNA only, so nothing to run here |
| **[BitDecoding](https://arxiv.org/abs/2503.18773)** | Edinburgh-led, with MSR | - | Studied: NVIDIA tensor-core decode attention; the HIP port is opt-in on `feat/bitdecoding-hip` |
| **[ISTA-DASLab](https://huggingface.co/ISTA-DASLab)** | IST Austria DAS Lab | see the model cards | The Qwen3.8-Flash-Next GSQ-RCO GGUFs (Q2_0, "IQ2_XS") in daily use, and the composition and quality numbers quoted here |
| **[Project Maya](https://github.com/mw00/project-maya)** | mw00 | MIT | The loop guard, soft thinking-budget close and tolerant tool-call parsing in `serve/` |
| **[Unsloth](https://huggingface.co/unsloth)** | Unsloth AI | - | The Qwen3.6 GGUFs with the MTP layer used for every measurement |
| **[Qwen](https://huggingface.co/Qwen)** | the Qwen team, Alibaba | Apache-2.0 | Qwen3.6-35B-A3B |

Code taken from another project keeps its license and credit in the commit that brought it in. pancake-strata itself
is MIT, as Strata is.
