# pancake-strata: plan

A fork of [Strata](https://github.com/Niko1221/Strata) tuned for speed on two home cards: an **RTX 3060 12 GB** (CUDA,
sm_86) and an **RX 7900 XT 20 GB** (HIP, gfx1100), with a Ryzen 7 5700G (AVX2, 16 threads) and about 57 GB of RAM.
Features are ported by hand from projects that forked Strata earlier and measured their gains.

Status (2026-11-07): driver up; CUDA sm_86 build of strata, strata-dsv41 and strata-poly. FastStrata's strata-poly engine is ported (branch port/faststrata-poly). First numbers on the RTX 3060 12 GB, Qwen3.6-35B UD-Q4_K_XL, greedy, cold run (bench/results/2026-11-07-pancake-baseline.jsonl, tools/pancake/bench_engines.py):

| Case | strata (16K ctx) | strata (64K ctx) | strata-poly, FastStrata flags (131K ctx) |
| --- | ---: | ---: | ---: |
| short chat, 256 new | 64.6 tok/s | 53.2 tok/s | 100.7 tok/s |
| code edit, 5.3K prompt, 512 new | 40.1 tok/s | 39.4 tok/s | 151.1 tok/s |
| 41.8K prompt: read / 128 new | - | 42.1 s / 9.7 tok/s | 35.8 s / 71.7 tok/s |

strata keeps fp32 KV (45 KB a token): setup's 131072 context runs out of VRAM on 12 GB, and at 64K no expert fits in VRAM. strata-poly: --threads 8 --kv-q8 --attn-split --graph-experts --elastic-prefill 1024 --lend-staging --stage-next --suffix-drafts 256 --cpu-multi (no lossy --tail-skip). The two engines' greedy texts differ after 25-81 tokens (different kernels, Q8 KV), as expected for two engines; the exactness rule applies to base vs mod of one engine.

Since then (commit 1c182934a): DeltaNet 4 columns a warp and a tiled small GEMM, the same bits, prompt reads 9-12%
faster. Every test, profile and rejected idea is in [docs/PANCAKE_FINDINGS.md](docs/PANCAKE_FINDINGS.md); phase 1's
profile is there too (prompt reading is GPU-bound; attention is 38% of a 42K read; CPU experts are ~half of decode).

## Goal

The fastest single-user agent serving (long, growing prompts and tool calls) for three models:

| Model | Engine in Strata | Packs to tune |
| --- | --- | --- |
| Qwen3.8-Flash-Next (125B MoE, ~6B active, 51B n-gram table, 4B MTP) | main engine (`src/core`, `src/program`) | the fastest: Q2_0, IQ2_XS |
| Qwen3.6-35B-A3B | `src/qwen35` | Unsloth UD-Q4_K_XL / Q4_K_M with the MTP layer |
| DeepSeek V4.1-Flash-1017 (latest, 2026-10-17) | `src/dsv41` | ISTA-DAS 2.00 / 2.25 bpw |

Speed counts as: decode tok/s (new text and code edits), prefill tok/s at 4K/16K/32K, and the time to the first token
of an agent turn after a long conversation. Output must stay identical to plain decoding unless a flag says it is lossy.

## Non-goals

- Qwen3.8-27B (dense). Strata's expert offload does nothing for it; llama.cpp or TabbyAPI on the 7900 XT serves it.
- NVFP4 packs and basalt's `.basalt` format: they need Blackwell (sm_120).
- Vulkan, SYCL, Intel and Mac paths: left as upstream has them, not tuned.
- Other engines as the base. Checked and rejected for this hardware: vLLM, SGLang, KTransformers (needs a 24 GB+
  card), TensorRT-LLM, LMDeploy, MLC-LLM, ExLlamaV2 (archived), PowerInfer, TGI (archived), LightLLM, NanoFlow,
  DeepSpeed-MII, TensorFold. They keep the model in VRAM or need far more VRAM or RAM.

## Sources

| Source | Commit | License | What it gives |
| --- | --- | --- | --- |
| [Strata](https://github.com/Niko1221/Strata) (base) | `146df3c4` (v0.2.12) | MIT | the engines, CUDA and HIP builds |
| [FastStrata](https://github.com/Jarson1408/FastStrata) | `df91582` | MIT | Qwen3.6-35B options measured on an RTX 3060 Laptop, `docs/OPTIMIZATION_LOG.md` |
| [PolyStrata](https://github.com/simplexai-labs/PolyStrata) | `a82999fd` | MIT | FastStrata's parent, the `src/poly` engine |
| [basalt](https://github.com/jesdga95/basalt) | `a9468e5c` | MIT | two-stage head, grammar, MTP work, idle sleep |
| [SGLang](https://github.com/sgl-project/sglang) kernels | main | Apache-2.0 | ideas: tree drafts, speculative sampling, GPU token mask |
| TensorRT-LLM tech blogs | - | - | ideas: grammar inside speculative decoding, auto n-gram enablement |

basalt rewrote its history, so it shares no git ancestry with Strata: it forked at Strata `822a5c783` (2026-09-27,
engine 0.1.8). Both diverged by thousands of commits, so nothing is merged; each feature is ported and credited.

## Ground rules

1. **Measure first, every step.** A change stays only if it is faster on the bench below and passes the exactness check.
2. **Exact by default.** Greedy output equals the base engine token for token. Lossy options are opt-in flags with a
   start-up warning (as basalt and FastStrata do).
3. **Both cards.** CUDA and HIP from the same source. A change measured only on one card says so; a CUDA-only change
   must not break the HIP build.
4. **Small commits** with the measured numbers in the message, crediting the source project and commit.

## Bench

- Harness: `~/bench` (existing Python harness and jsonl results), extended for Strata's OpenAI API.
- Cases: short chat (new text), code edit (copy-heavy), 16K and 32K prompts, an agent session replay (growing prompt,
  tool calls), a repeated task in a new conversation (cross-conversation drafts).
- Each case: greedy, 3 runs, interleaved base and mod, median. Record VRAM peak and RAM use.
- Exactness: greedy text of the base and mod engines compared for 900+ tokens per case.

## Phases

### 0. Bring-up (after the reboot)

- Driver 595 loads, `nvidia-smi` sees the 3060; optional gpu-unlock checked in `journalctl`.
- CUDA 13.1 build of Strata for sm_86 on the D drive; HIP build for gfx1100 compiles.
- Download the fastest Flash-Next pack, Qwen3.6-35B UD-Q4 with MTP, DeepSeek 2.00 bpw as disk allows.
- **Done when:** all three models answer on the 3060 and the base numbers are in `bench/results`.

### 1. Baseline and profile

- Base Strata on the 3060: MTP on/off, n-gram on/off, both, at each context length.
- Profile where a decode step goes: head, attention, GPU experts, CPU experts, PCIe copies, drafting, sampling.
- Measure where Flash-Next's n-gram table sits (VRAM, RAM, disk) and what reading it costs.
- **Done when:** a written breakdown per model says which phase below targets the biggest cost.

### 2. FastStrata's measured options, Qwen3.6-35B first

Port from FastStrata's `src/poly` into Strata's `src/qwen35`, in its measured order, then try the same ideas on Flash-Next:

| Option | Gain FastStrata measured (3060 Laptop 6 GB) |
| --- | --- |
| `--attn-split`: decode attention in one pass on the tensor cores | +30% decode at 37K |
| `--graph-experts`: each layer as one CUDA graph | -7 to -10% time per window |
| `--elastic-prefill`, `--lend-staging`, `--stage-next`: wider prompt batches, overlap the next layer's expert copy | agent turn after 38K: 6.4 s -> 3.9 s |
| `--suffix-drafts`: n-gram drafts from every past conversation | repeated task 90 -> 98-107 tok/s |
| `--cpu-multi`: CPU K-quant experts read once per window | +1 to +5% |
| `--kv-q8` | longer context in the same VRAM |

HIP: `--graph-experts` maps to HIP graphs; the tensor-core attention needs a WMMA path for gfx1100 or stays CUDA-only.

### 3. Drafting: MTP and n-gram (the key point)

- From basalt: the 2-bit draft-head screen (cheaper draft scan), the K/V append beside the next draft query,
  `--relaxed-drafting` as a lossy opt-in.
- Auto n-gram: turn suffix/n-gram drafts on when they pay (code edits), off when they do not (TensorRT-LLM's idea).
  Target: keep the old llama.cpp result on code edits (118 -> 175 tok/s with ngram + MTP) or beat it.
- Speculative sampling (SGLang `speculative_sampling.cu`): keep acceptance high at temperature > 0.
- Small draft trees (SGLang `tree.cuh`) against chains. On a MoE with experts in RAM, each branch wakes more experts,
  so this stays only if measured faster.
- **Done when:** each item is measured on the code-edit, chat and agent cases, kept or dropped with its numbers.

### 4. Exact two-stage output head (basalt)

- Port basalt's `q6k_head.cu`, `q5k_head.cu`, `decode/verify/head_split.cpp` to Flash-Next and Qwen3.6-35B. No
  Blackwell intrinsics in them; HIP through the existing hipify path. The 3060's 360 GB/s should make the head a large
  share of a decode step (phase 1 confirms).

### 5. DeepSeek V4.1-Flash on the 7900 XT

- Strata's DeepSeek engine runs on RTX 30 but on AMD only gfx1201 (R9700) and gfx1151 (Strix Halo). Add gfx1100.
- Needs the 7900 XT in the machine to run; until then compile-only.

### 6. Grammar inside speculative decoding

- basalt's grammar engine (`src/program/grammar`: GBNF, JSON schema, tool-call forms; CPU, portable).
- The token mask applied on the GPU (SGLang `apply_token_bitmask_inplace_cuda.cu`), checked per verify window so MTP
  keeps drafting under a grammar.
- **Done when:** tool calls are always valid JSON for the schema and decode speed with a grammar is within a few
  percent of without.

### 7. Cache tiers

- FastStrata's `--conversation-disk` (70K conversation after a restart: 83 s -> 7.2 s), for all three engines.
- One shared cached prefix (system prompt and tools) across agent conversations, as SGLang's RadixAttention does.
- basalt's `--idle-sleep`: free VRAM between requests, wake in seconds instead of a cold load.

### 8. Additions from the forks and reports read on 2026-11-07 (owner: "add them to the plan and bring them in")

Each lands as its own commit, measured (the bench and, where numbers change, exactness or KL), credited to its source.

| # | Item | Source | Engine | Exact? | Expected |
| --- | --- | --- | --- | --- | --- |
| 8.1 | Flash-Next IQ2_XS running on the 3060 with Speedy's sm_86 presets as the starting config; base numbers | Speedy (DevDock-AI) | main | yes (config) | the plan's first model, finally measured |
| 8.2 | Turn on the upstream opt-ins Speedy measured, A/B each on this PC: `STRATA_PF_GROUP_SCRATCH`/`_HC_REGION` (#4039), `STRATA_PREFILL_ROUTED_STREAM` (#4283), `STRATA_PROMPT_ATTN_IMMA`, `STRATA_KV_PREFETCH` | Speedy, upstream PRs | main | per its PR | prefill +5-33% on the 3080 |
| 8.3 | Port the ones upstream lacks: `STRATA_VERIFY_DEFER_CPU` (#2689/#2690, light CPU expert rows a layer late in verify), `STRATA_PSI_SHED` | Speedy | main | KL measured by the PR | decode GPU-wait cut |
| 8.4 | The 59.6K-token MTP draft vocabulary (English/code + CJK), lossless | Strata-rdna3 | main (and poly's `--draft-vocab`) | yes | draft step ~12% cheaper, decode +4% |
| 8.5 | Adaptive MTP depth (3-4 by measured cost) | llamAmpere | poly | yes | decode |
| 8.6 | A learned expert profile loaded at start | llama.cpp MoE cache PR | poly, main | yes | start-of-session decode |
| 8.7 | Two-stage prompts (`--prompt-split`) | StrataFast | main | check | long prompts 1.24-1.55x there |
| 8.8 | Fused flash-attention prompt kernel (D=256, GQA 8, Q8 cache, causal inline); INT8-IMMA attention variant | own; Speedy's IMMA idea | poly | numerics change: KL | the 38% at 42K |
| 8.9 | Opt-in lossy: cache-aware routing (colibri `CACHE_ROUTE`), SJ-KVaRN-style KV body (keys 8-bit, recent window), mb-MTP | colibri, llamAmpere | poly | no: opt-in, KL | decode / VRAM |
| 8.10 | RDNA3 kernels, tensor parallel, SDMA, idle-queue parking for the 7900 XT | Strata-rdna3 | main, HIP | per kernel parity | 2-3x stock on gfx1100; waits for the card |
| 8.11 | basalt's head kernels (owner's request): **done 2026-11-08** - `q5k_head.cu` ported (Q5_K heads: Q2_0 / IQ3_XXS files) and the same designs written for IQ4_XS (the IQ2_XS head in daily use): exact head GEMV, 2-bit two-stage draft pick | basalt | main | yes | +3-5% decode (draft pick), head x1.07-1.27 in its test |

### Priority change (owner, 2026-11-08): Qwen3.8-Flash-Next for daily use

Daily use moves from Qwen3.6 (`strata-poly`) to Qwen3.8-Flash-Next IQ2_XS on the main engine (`run-pancake.sh`).
Main-engine items come first now: 8.2-8.4 (done or measured), basalt's two-stage head (8.11) on the engine's native
Q5_K head (`q5k_head.cu`, the same model family and width), 8.7 two-stage prompts, 8.6 a learned expert profile, and
the Qwen 4 readiness of phase 9. The poly-only items (8.5 adaptive depth: poly already has the cost-based `Guesses`
rule up to 3 guesses; 8.8 its flash-attention kernel) wait.

### 9. Qwen 4 (owner's request, 2026-11-07)

Status, checked on the official pages on 2026-11-07: **no Qwen 4 weights are out.** The Qwen organization on Hugging
Face lists Qwen3.8 (Flash-Next, 27B, 2.4T-A95B) and Qwen3.6 as its newest models. Qwen3.8-Flash-Next's own model card
calls it an experimental preview of the architecture that will underpin Qwen4: 125B total / 6B active, 48 layers in 12
blocks of three Gated DeltaNet + MoE layers and one Qwen Sparse Attention + MoE layer, a 20M-entry n-gram embedding
table (51B parameters) at layer 2, one MTP layer, 262K native context (1M extended). Third-party reports (not
confirmed by Alibaba) name Max / Plus / Flash API tiers and a 27B open-weights tier.

So the work is:

1. **Now:** every Flash-Next item of phase 8 (8.1-8.4, 8.7) runs on the Qwen 4 Flash architecture: the main engine's
   GDN, QSA, n-gram table and MTP paths are what Qwen 4 Flash will need.
2. **When the weights appear:** read the GGUF's architecture name and tensor list against Flash-Next's
   (`strata --families`, the layout guards in `src/core`), add the family to setup (`setup.py` families table, hashes),
   pack it, run the bench and the exactness/KL checks, and add its row to the README. Expected: little new code if the
   architecture matches the preview.
3. **A Qwen 4 27B dense tier**, if it ships open: Strata's expert offload does nothing for a dense model; serve it
   with a dense engine (llamAmpere-style KV and quant tricks are the reference for 12 GB cards) or keep it out of
   scope, decided when its size is known.
4. Watch the Qwen organization on Hugging Face and the Qwen blog for the release.

Disk (2026-11-07): Flash-Next IQ2_XS (shard 1 37 GB + n-gram table 27 GB + MTP 6.5 GB) is already in
`/home/dan/clone-of-d/Strata-data`; `~/Strata-data` on the root disk was a duplicate and is gone; the unused Qwen3.6
UD-IQ3_S file was deleted. DeepSeek V4.1 (239-274 GB) still does not fit.

### 10. CPU, RAM, PCIe and GPU in sync (owner, 2026-11-08)

Decode on the 3060 is a relay: per layer the router picks experts, the GPU runs the cached ones, the CPU reads the
missing ones from RAM (33 GB/s), the GPU waits, then the next layer starts; PCIe is mostly idle inside a step.  Everything
slow starts after the router decides.  Make the parts overlap by knowing earlier which experts come next.  None of these
changes which experts are used: the output stays exact.

| # | Item | Source | Measure first |
| --- | --- | --- | --- |
| 10.1 | Instrument a decode step: GPU idle waiting for the CPU, CPU expert time, PCIe busy, per layer | own (existing STRATA_* timers) | the ceiling on this PC |
| 10.2 | Router lookahead: predict layer L+1's experts from layer L's output; start the CPU's RAM reads and the PCIe copies early | colibri PILOT (idea) | prediction hit rate |
| 10.3 | Draft-informed prefetch: the MTP draft tokens' routing predicts the verify window's experts; copy them into VRAM during drafting | new | prediction hit rate |
| 10.4 | Finish together: when the CPU's share would end later than the GPU's, move one or two missing experts to the GPU | Avalon "hosted MoE" | per-layer finish times |
| 10.5 | Avalon's VRAM lending: unused KV-cache tail and idle compute buffers lent to the expert cache | Avalon | experts in VRAM |

Order: 10.1, then 10.2 / 10.3 by their measured hit rates, then 10.4, 10.5.

## Risks

- Code drift: FastStrata's options live in `src/poly`, not in `src/qwen35`; porting may mean rewriting, not copying.
- HIP parity: several gains rely on NVIDIA tensor cores or CUDA graphs.
- The 7900 XT is not detected today; AMD work is compile-only until it is back.
- Disk: the root disk is tight; builds, models and caches go on the D drive (`/home/dan/clone-of-d`).
- Upstream moves fast (thousands of commits a month); rebase on Strata releases, not on every commit.
