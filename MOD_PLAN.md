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

## Status (2026-11-08, branches exp/expert-prefetch -> feat/grammar)

| Item | State | Evidence |
| --- | --- | --- |
| 10.1 decode breakdown | done | CPU experts 38 ms of a 66 ms window; GPU waits 17.5 ms |
| 10.2 router lookahead | done, on by default on the GPU | same picks, CPU cost 30 -> 0.6 ms/window, +10% decode on mmap'd experts |
| 10.2 prefetch consumer (Foresight) | measured, off | 0.97 at 4 slots; 8 slots run out of VRAM on long prompts |
| 10.3 draft-informed prefetch | covered by 10.2 | the lookahead routes the window's draft tokens |
| 10.4 finish together | already in Strata | PcieAuto, share 0.20-0.25 on all 48 layers |
| 10.5 VRAM lending, item 7 per-layer budgets | dropped | +5% cache slots: no decode change; cross-layer swaps 48.3% vs 48.4% |
| 8.6 learned expert profile | done, in the IQ2_XS presets | +6.7% decode, every run faster |
| 8.7 two-stage prompts | blocked | needs a second GPU |
| 8.8 fused attention prompt kernel | low ceiling | attention is 6.7% of a 42K prompt on the main engine |
| 8.9 cache-aware routing | opt-in (`turbo`) | 0.5: +25% decode, 2.9% tokens differ (stock 1.2-1.3%) |
| 8.9 KV body, mb-MTP | dropped | VRAM does not buy decode here; drafting is 2-5% of a window |
| 8.10, phase 5 | blocked | the 7900 XT is not detected |
| phase 3 drafting | in the engine / low ceiling | auto n-gram (DraftPolicy), spec sampling exist; draft step 2-5% |
| phase 6 grammar in spec decoding | done, on in the IQ2_XS presets | tool calls / strict schemas always valid, 0.1-0.2% per window |
| phase 7 cache tiers | mostly in the engine | open: sleep-in-place with the RAM complement (basalt 64167ba73), automatic disk sessions |
| phase 9 Qwen 4 | waiting | no weights published |
| prompt path kernels | measured, FP16 kept | WMMA worse; fused int8 -7% at 42K but +12% at 5K |
| vs stock Strata (upstream 146df3c46) | measured | Flash-Next IQ2_XS, experts read from disk on both: decode 54.5 -> 69.2 tok/s (+27%), 5.3K edit prompt 23.3 -> 11.0 s; ours with the RAM arena 75.5 |

**Next session (left 2026-11-08):**
1. Decide the load default: building the 34 GB expert arena now often trips systemd-oomd (`user@.service`
   memory-pressure limit 50% for 30 s; only the capped scope dies). `STRATA_RAM_KEEP_GIB=20` (mmap tier) loads every
   time but decodes ~8% slower (69 vs 75 tok/s). Either make 20 the default in `capped.sh`, or keep the arena and pin
   it in smaller steps so the stall time stays under the limit.
2. Phase 7 sleep-in-place (basalt 64167ba73, ~815 lines): only if the GPU is wanted for other work while idle.
3. Merge `port/faststrata-poly -> exp/expert-prefetch -> feat/grammar` when wanted (all pushed, none merged).
Stock comparison build: `../Strata-stock` (upstream 146df3c46, `build-cuda/strata`); results in FINDINGS 10f.

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

Status 2026-11-08 (branch `feat/grammar`): **done**.  basalt's grammar engine ported (d64768699 GBNF / JSON-schema /
tool-call builders and 02e1006fe the per-request guard; the engine credited to llama.cpp, MIT), `Verifier::resample_row`
(c7813a6c3), the engine's `--grammar on` and request `grammar=` key (011e13d72: after each verify window the first kept pick
the grammar refuses is drawn again under its mask and the window ends there, so MTP keeps drafting), and the server's key
for tool calls and JSON response formats (`serve/grammar.py`).  CPU tests: grammar_test, grammar_schema_test,
grammar_guard_test, serve/test_grammar.py.  On the 3060: 3 strict json_schema requests (enum + required, two of them
off-topic prompts) and a tool call with an enum argument all valid; 0.3-0.8 ms of grammar code per request, 35-110 us per
50-130 ms window (0.1-0.2%), 0-1 masked redraws per request.  The IQ2_XS presets run with `--grammar on`.  Not covered:
`--batch` slots (refused with an error, as basalt's first version), a layer split's head stage (resample_row forwards).

### 7. Cache tiers

- FastStrata's `--conversation-disk` (70K conversation after a restart: 83 s -> 7.2 s), for all three engines.
- One shared cached prefix (system prompt and tools) across agent conversations, as SGLang's RadixAttention does.
- basalt's `--idle-sleep`: free VRAM between requests, wake in seconds instead of a cold load.

Status 2026-11-08: the shared prefix is in the engine (a new conversation that starts inside a parked one borrows its
checkpoint); conversations to disk exist as session files (`slot_save_path`, `/slots/0?action=save|restore`), by hand
rather than FastStrata's automatic `--conversation-disk`; VRAM back between requests exists as `--vram-elastic` +
`POST /v1/vram` (no reload), but the engine refuses it with the pinned RAM complement the daily IQ2_XS preset uses (the
complement holds only the experts the VRAM cache does not), and `--idle-unload` is a cold reload (~1 min).  basalt's
sleep-in-place (64167ba73, ~815 lines: drop and redo the RAM copy's registration, refill on wake) is what closes that gap;
left for after the speed and quality items.

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

8.9 measured 2026-11-08, cache-aware routing (`STRATA_ROUTE_PRIOR`, the resident experts' logits get a bonus; changes
the output), token-forcing check against the exact preset (stock noise 1.17-1.30% mismatch, 0 large-margin events) and
decode against the learned-profile preset (75.2 tok/s mean, short x2 + edit x2):

| lambda | mismatch | vs stock (90% CI) | large-margin events | verdict | decode |
| --- | ---: | ---: | ---: | --- | ---: |
| 2.0 | 9.90% | +8.67 pts (5.71, 11.95) | 8 | REJECT | 88.1 tok/s (+17%) |
| 0.5 | 2.89% | +1.66 pts (1.00, 2.41) | 0 | REJECT (rate) | 94.1 tok/s (+25%) |

0.5 changes about one token in 60 more than the engine's own noise, never with a confident wrong pick: an opt-in, not a
default.  `./run-pancake.sh iq2_xs turbo` (configs/pancake/flashnext-iq2_xs-turbo.json) runs it.

Prompt reading on the daily preset, measured 2026-11-08 (STRATA_PREFILL_TIMING=1, the 41,818-token prompt): 42.9 s
(976 tok/s; the 100 s of the RAM-capped trace run read lent experts from the SSD).  GPU timeline: expert GEMMs 29% (gate/up
17.8, down 8.5, dequant 2.9 - the FP16 path, not MMQ, on this card), GDN ~12%, PLE 10.6% (4.3 s on the host), QSA
attention 6.7% + projections 5.5% + select 2.6%.  So 8.8's fused attention kernel can give a few % here (it was 38% on
strata-poly), and the expert GEMMs are the prompt's target.  `--ple-io mmap`: long prompt 42.9 -> 40.4 s, PLE host 4.3 ->
3.3 s, decode not better (edit 75.9/88.6 vs 80.9/91.4): not adopted.
The prompt path's other expert kernels, same prompts (time to first token, 5.2K / 41.8K tokens; FP16 default 6.51 /
42.9 s): `STRATA_PF_WMMA=1` 8.76 / 47.9 s (worse both); `STRATA_PF_FUSED=1` (int8) 7.29 / 40.0 s (+12% / -7%).  Agent turns
mostly read short deltas after the conversation cache, so the fused kernels stay off; a chunk-size switch between the
two would be the next step if long cold prompts matter.

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

Predictors to compare (owner, 2026-11-08), all scored offline on one `--dump-routing` trace before any prefetch is built
(`tools/pancake/predict_experts.py`, branch `exp/expert-prefetch`): recall (predicted & used / used), miss cover (the
share of the cache's misses predicted) and waste (predicted, not held, not used), per budget of experts per layer.
1. Historical: the previous window's experts; decayed frequency (an EMA); both topped up.
2. Next-layer router lookahead (Strata's RouterLookahead; `STRATA_LOOKAHEAD_STATS=1` scores it in the engine).
3. Dictionary: (layer, last 1-4 tokens incl. the draft tokens) -> expert distribution, learned online from every verify
   window (EMA), backed off to shorter keys and then to the global frequency (a Beta prior); top-16 experts per layer per
   key to bound memory.  Strata's suffix index is the model for the lookup.
4. MTP-based: each main layer's router applied to the draft's hidden state.  Only if 1-3 leave enough misses uncovered:
   the MTP block's hidden state is not each layer's input, so it is the weakest-founded and the costliest to build.
5. Hybrid: the best of 1-4 merged by confidence.
Prior art (reported, not yet read here): SP-MoE, MoE-SpeQ, "Speculating Experts" (2026), DoMoE, and an Oct 2026 study
of expert bytes per speculative window.  The aim is speed on this PC, not a novelty claim.

Two more ideas from the owner's reading (2026-11-08), new for this engine and testable on the same trace:
6. Expert co-activation: per layer, which experts fire together (a conditional-probability graph); when one expert is
   predicted or seen, its partners are candidates too.  Scored as predictor 6 and as an add-on to 1-3.
7. Per-layer cache budgets: router entropy differs by layer; give the cache's slots to layers by measured miss cost
   instead of equally.  Scored in the replay simulation (replay_cache_policy.py), then built if it wins.

Already in Strata or in this plan (the owner's lists checked against the code, 2026-11-08): adaptive draft length and
the MTP-vs-suffix choice (DraftPolicy prices each window size and source by measured cost and acceptance), early draft
cut-off (`--spec-min-p`), token-grouped expert execution in a verify window (`native_expert_grouped`; `--cpu-multi` on
poly), fused dequantization in the matrix kernels (native_mmvq, MMQ), INT8 KV cache (`--kv int8`), prefix / conversation
caching and checkpoints, staging buffers that overlap copies with compute (prefill; FastStrata's `--stage-next`), an
adaptive expert tier (usage-based swaps).  FP8 KV: no FP8 tensor cores on sm_86.  Expert-sensitivity mixed-precision
quantization: needs the BF16 model (~250 GB) to quantize from, out of reach on this disk; Avalon's IQ3_KS hybrid is the
evidence it pays.  Age-based KV precision is 8.9 (SJ-KVaRN style).  An adaptive controller over all of it: later, once
each part is measured.

Design rules for 10.2 / 10.3 (owner's input, 2026-11-08, adjusted by measurements):
- Memory: the prefetch window comes only from memory nobody uses now (the KV cache past the current context, idle
  compute buffers: Avalon's lending), never carved out of the expert cache (upstream's Foresight slots lost on every box).
  A hard floor of free KV room: near it, low-confidence prefetches stop (no OOM, no KV offload).
- Confidence: prefetch only experts whose predicted routing weight passes a threshold; weight early draft tokens over
  late ones (token 1 is the most predictable).  The threshold and depth are set from the hit rates measured in 10.1-10.3.
- Streams: transfers on their own copy stream, apart from compute and KV writes; when verification rejects draft token k,
  unstarted transfers for tokens > k are dropped.
- Timing: stage layer N+2 / N+3 while layer N runs; prefetch an expert only when its PCIe time (PCIe 3.0 x16, ~12 GB/s)
  fits before the layer that needs it.

Results, 2026-11-08 (fast preset, IQ2_XS, 4236 cache slots, 4 prompts; `bench/prefetch/`):
- 10.1 done.  A decode window (3.05 tokens) takes 66 ms: verify 63.3 = GPU waiting to reach the CPU's results 17.5 +
  per-layer host 44.4, of which CPU expert compute 37.7.  Per layer and window: 17 VRAM hits, 7.5-12 CPU experts, 1.2-3.3
  PCIe copies; PCIe probe 13.4 GB/s.  The CPU's share is the bottleneck, as the plan assumed.
- 10.2 / 10.3 offline (`predict_experts.py`, tests in `test_predict_experts.py`): of the experts the cache misses, a
  prefetch of 16 per layer covers transition (routing-keyed dictionary) 29% at 65% waste, hybrid 27%, co-activation 26%,
  prev+freq 26%, frequency 25%, prev-window 20%.  History-only predictors are weak.
- The engine's RouterLookahead (`STRATA_LOOKAHEAD_STATS=1`) is about 2x better: its top-8 votes cover 24-35% of the misses
  at 60-69% precision, top-12 33-46%; k=14 recall 58-69%.  Lookahead is the predictor to build on (10.2 first);
  transition is the best add-on to test merged with it.
- Its cost: the predictor thread takes 30-35 ms of CPU per window, on the same cores as the CPU experts (37.7 ms).
  Before 10.2 can pay, the lookahead must get cheaper or move to the GPU; measure decode with it on vs off.
- PCIe budget: at 13.4 GB/s a 1.4 ms layer moves ~18 MB, about 12-15 experts, so a prefetch must stay within ~8-12
  confident experts per layer; the confidence cutoff from the design rules is required, not optional.
- Lookahead cost, measured (arena tier, 2 runs x short/edit, `bench/prefetch/la-*.jsonl`): decode 70.5 tok/s mean
  without the predictor thread, 61.4 with it scoring only (-13%; short 52.8/63.4 -> 43.8/59.2, edit 75.8/90.0 -> 66.5/76.0).
  CPU expert time per window rises 18-33% (the thread's 30-34 ms of router GEMVs compete with the 7 pool workers for
  cores and RAM bandwidth).  So 10.2 starts by moving the lookahead to the GPU: the next layer's router on this layer's
  input is a 512 x n_embd GEMV per token, microseconds there, and it frees the CPU entirely.
- 10.2 step 1 done: the lookahead runs on the GPU (`STRATA_LOOKAHEAD_GPU=1`, commit 7ba03d6fe): the verify window routes
  layer l + 1 on layer l's input and publishes the top-10 before layer l's doorbell.  Same predictions as the CPU thread
  (k=10 recall/precision 0.576/0.548 vs 0.576/0.553, 0.544/0.505 vs 0.535/0.501), predictor thread 30-34 -> 0.6-0.7 ms
  per window, decode with the scorer on 61.4 -> 66.2 tok/s (lookahead off 70.5; the rest is the scorer on the pool's
  path and run noise of ~20% between a prompt's first and second run).  On the arena tier the predictions have no
  consumer yet: step 2 is acting on them (Foresight's swap space first, which measured 0.91 on a 3060 with the CPU
  predictor, i.e. less than the predictor's own 0.87 cost).
- 10.2 step 2, Foresight's swap space fed by the GPU lookahead (`STRATA_FS_SLOTS`): 4 slots per layer, 32 copies per
  window: decode 49.9/63.8 short, 74.4/85.6 edit = 68.4 mean vs 70.5 off (0.97; upstream's 0.91 had the CPU predictor's
  0.87 inside).  The prefetch works (16140 experts served from 8790 copies, 97% of them the lookahead's) but starves:
  486k requests found no free slot, 100k over the copy budget.  8 slots / 64 copies: short 52.6/67.2 (vs 52.8/63.4) but
  the 5.3K-token prompt then failed (cuBLAS status 14, out of VRAM): the swap space is allocated after the cache is sized
  and eats the prompt path's headroom.  So the window has to come from idle memory (design rule 1, 10.5), not on top.
- 10.4 is already in Strata: PcieAuto measures the pool's time per missed expert (53-55 us here) against the link (13.4
  GB/s) and sets the share of every layer's misses the GPU takes over PCIe (0.20-0.25, all 48 layers) so CPU and GPU
  finish together.  Nothing to build; a per-layer refinement is not worth it at a 0.20 share.
- 10.2 step 3: the GPU lookahead is on by default (STRATA_LOOKAHEAD_GPU=0: the CPU thread).  On the mmap'd-expert tier,
  where the lookahead runs by default to warm pages: decode 52.0/64.1/72.6/79.2 -> 56.5/67.6/81.0/90.0 tok/s (+10%,
  every run faster; `bench/prefetch/mmap-*.jsonl`).  Outputs: the arms differ only where the baseline arms differ among
  themselves (near-tie tokens, e.g. la-off's second short run splits at token 133 from every other arm).
- 8.7 (two-stage prompts) needs a second GPU: StrataFast's `--prompt-split` runs each chunk's late layers on another
  card while the main one starts the next chunk.  Blocked on hardware, like 8.10.
- 8.6 done: a profile learned on agent-style requests over source files the bench does not use
  (`tools/pancake/learn_profile.py`, `data/expert-profile-pancake.bin`) against the shipped one: decode
  52.8/63.4/75.8/90.0 -> 57.8/70.8/80.9/91.4 tok/s (+6.7%, every run faster, most on the cold first runs).  The IQ2_XS
  presets use it now; the Q2_0 presets keep the shipped one until measured.
- Phase 3, measured ceiling: the draft step is 2.5-3.2 ms of a 46-130 ms window (2-5%), so basalt's draft-head items (2-bit
  screen of a Q8_0 draft head, K/V append beside the draft query) can give 1-2% at most here.  Speculative sampling at
  temperature > 0 is in the engine (`STRATA_SPEC_PROB`, coupled, Gumbel; measured upstream on a 3060: the default path is
  as fast or faster), and the MTP/suffix choice is DraftPolicy's.  Lower priority than anything on the CPU experts.
- 10.3 is covered by 10.2: the GPU lookahead routes every token of the verify window, and those are the MTP draft's
  tokens; the separate MTP-head router predictor (rated weakest above) is not needed.
- 10.5 does not pay here: `--kv-resident 8192` gives the expert cache the idle KV VRAM a lend would (4188 -> 4406 slots,
  +5%); CPU experts per layer-window fall ~4%, decode 68.5 tok/s vs 70.5 (noise).  Dropped (`bench/prefetch/kv8k.jsonl`).
- Item 7 (per-layer budgets) does not pay: in the replay, swaps across layers give 48.3% of routed entries from the cache
  against 48.4% for the engine's per-layer swaps (`replay_cache_policy.py --cross`, `bench/prefetch/cross-layer.txt`); the
  profile already spreads 62-110 slots per layer.  Dropped.
- Memory: a run without `STRATA_RAM_KEEP_GIB` used the whole 57 GB and killed the desktop session.  Runs now use
  `STRATA_RAM_KEEP_GIB=12` inside `systemd-run --user --scope -p MemoryMax=54G` (owner, 2026-11-08).  The trace above ran
  under a 42G cap, which fell back to mmap'd experts (the arena needs 45 GiB): speed differs, routing does not.


## Phase 11: RX 7900 XT (gfx1100), started 2026-11-08

The owner switched back to the RX 7900 XT. Owner requests this phase (all in scope): resume on the 7900 XT; port Strata-rdna3's
RDNA3 work ("any percent is good": no fusion skipped); try again everything that was dropped on the 3060; make the CPU side
faster without losing the fast preset's gain; research in depth (5 layers: survey -> techniques -> mechanisms -> code ->
expected gain; papers, MIT/Harvard/Stanford/Berkeley/CMU/Tsinghua..., Chinese and Western companies, open-source engines;
memory, caching, prediction, statistics, direct GPU commands); compare with Avalon and stock Strata (numbers from the owner's
repos llama-cpp-7900xt-optimization and strata-7900xt-optimization); keep the RTX 3060 (CUDA) line unchanged.

Research report: [reports/Pancake MoE engine speedup.md](reports/Pancake%20MoE%20engine%20speedup.md) (notes in
research_notes/Pancake MoE engine speedup/).

Done (branch port/rdna3-decode, local):
- HIP build (`build-hip`, gfx1100); run-pancake.sh picks it when no NVIDIA card is present.
- Ported from Strata-rdna3: RDNA hyper-connection read (+Q8, A8, INJ_UP, FUSED), wave32 multi-row GEMVs (RDNA_Q6K, MMVQ_WAVE),
  COMBINE_CPU, WAIT_FUSE, ROUTE_PUBLISH.  rdna_gr.cu / rdna_mmvq.cu build on wave32 HIP only; CUDA gets stubs.  Not ported:
  long-context top-k (= upstream STRATA_SELECT_LONG), PF_GROUP_RES (= upstream group_resident), PA_KLDS (upstream rewrote the
  kernel with STRATA_PA_RPF), RDNA_PAIR (neutral, off upstream), GDN megakernel (0.0 ms on one GPU: the window waits on CPU rows).
- Decode 87-101 -> 104-108 (kernels) -> ~112 tok/s (fusions); Strata-rdna3 itself 109-119; Avalon 95-100; stock Strata 75-77.

Round A (load-gated, base 111.7 / 92.5 after an 11.7K prompt): --pcie-frac 0 118.4 / 104.3; + adapt every window, 160 swaps
121.2 / 100.1; rdna3's adapt settings 116.4; --kv-resident 8192 114.4; STRATA_SPEC_COST 112.0 / 98.3; DFlash2 115.9 / 84.3;
lossy: turbo 149.3, tail skip 7 129.7.  Worse: upstream group A switches together 101.2 (bisecting), ADAPT_LAG 87, zh60k vocab
92.5, rdna3 profile 105.6, upstream HC_Q8 107.2, kv-resident 131072 106.2, mmap tier 108.6.  Flat: --spec 5/6, full vocab,
SELECT_LONG, mtp-window, hw queues.  VERIFY_DEFER_CPU gave non-finite logits with COMBINE_CPU (fixed: cfuse off when deferring).

Replay ceilings at 10,909 cached experts (44%): engine tier 79.6%, per-layer LRU 89.5%, Belady 94.7%; tier every window x160 85.0%.

Done 2026-11-08 evening: rounds B/C, presets fast-hip (default) / hip (exact) / turbo-hip (commit f0f9c5788); token forcing
INCONCLUSIVE +0.4 pts, 0 large-margin events; strata_bench --long and strata-7900xt bench.py beat Avalon and stock Strata
(bench/results/2026-11-08-7900xt-final-suites.txt).  VERIFY_DEFER_CPU still non-finite on HIP with the ported kernels.

Status 2026-11-08 late.  Done: CUDA build check (compiles; GPU run needs the 3060), final comparison (beats Avalon and
stock Strata), and measured with no gain: reuse-aware admission (PCIe 3.0 cost), L3 prefetch (STRATA_ARENA_WARM, worse), CPU pool
settings (defaults best), IQ2_S two-half decode (STRATA_IQ2S_PAIR, neutral; the kernels are compute-bound), window thresholds and
--spec 3/5/6, tandem.cpp's MTP_DEFER (already here).  Added from tandem.cpp: STRATA_POOL_PREREAD (idle pool workers pre-read the
lookahead's next-layer CPU experts); A/B pending with the Q2_0 model test.

### Phase 11 research index (2026-11-08)

| Research | Where | What it produced for the plan |
| --- | --- | --- |
| qwen38-megakernel (cutlerbenjamin1-cmd, AGPL, CUDA dense 27B) | this session, notes in memory | ideas only (AGPL vs MIT): GDN replay already here; in-context draft-vocab tokens -> tested full vs en vs zh60k vocab (no gain) |
| Strata-rdna3 fork (nicejunjie) | MOD_PLAN phase 11 "Done" | RDNA3 kernels + COMBINE_CPU / WAIT_FUSE / ROUTE_PUBLISH ported (87-101 -> ~112 tok/s); megakernel GDN = 0.0 ms on one GPU (reverted, revisit after the GPU profile) |
| Six-thread deep research: CPU experts, caching/prediction, MoE speculative decoding, RDNA3 GPU decode, PCIe/KV, engines/labs/companies (MIT, Harvard, Stanford, Tsinghua, DeepSeek, Moonshot, Alibaba, ByteDance, Huawei, OpenAI, Anthropic, xAI, NASA: none relevant from the last four) | reports/Pancake MoE engine speedup.md, research_notes/Pancake MoE engine speedup/ | --pcie-frac 0 (+6/+13%), adaptive tier every window (+16% total), Belady ceiling replay (79.6 / 89.5 / 94.7%), expert deferral (lossy, broken on HIP), DraftExpert-style miss-aware windows (tested: policy already optimal), tail-row drop (item 5) |
| Avalon / stock Strata numbers (owner's repos llama-cpp-7900xt-optimization, strata-7900xt-optimization) | bench/results/2026-11-08-7900xt-final-suites.txt | final comparison: pancake beats both |
| tandem.cpp (ewhacc, MIT) + tandem.cpp-amd (itlab-cl) | this section | STRATA_POOL_PREREAD ported (A/B pending); MTP_DEFER already here; MTP head Q4_K requant not needed (ours already ~2-bit, 787 MB) |
| Pattern recognition (ExpertFlow, ST-MoE CCT/HT, MoE-Beyond, FineMoE) | this section | items 3-4: per-conversation expert working set, per-domain profiles |
| Stanford engines (Tokasaurus, Hazy Research megakernels, Cohere megakernel 2026) | this section | no new 2026 Stanford engine found; megakernels matter only for the GPU part (item 1) |
| Q2_0 quality (docs/QUANTS.md) | this section | Q2_0 = opt-in presets (134.7-136.1 tok/s; 80.6% same top token as Q8_0) |
| Strategy (where the time goes, ranked steps) | https://claude.ai/artifact/KS4sEzwMLwEGhVEPdi3xyF | the step order below; results log kept there |
| Avalon upgrades report (owner's /home/dan/reports/Avalon upgrades from upstream projects.md) | this section | for pancake: the gfx1100 hipBLASLt table must match the runtime (the preset loaded 100200 on hipBLASLt 100401 -> untuned; 100401 table A/B queued; Strata measured +32/+78/+82% prompt on a 7900 XTX); longer prompt-lookup windows (--lookup-max-t 12/16, +11-14% on edits per Strata) A/B queued; tool-call recovery, env_to_user, reasoning_close_retry already in serve/; for Avalon itself: source fast-forwarded c24c03d -> 88313b6 (48 commits; rollback branch local/c24c03d-baseline), rebuild after the benchmarks, model needs ~20 GB more free on D |
| llama.cpp CPU share (PR #30966, merged 2026-11-07; #31888; #31233 ClassLRU; #31286 open) | this section | already here: Strata's PCIe share is the same split (--pcie-frac 0 = their share ~0.95, measured best); ClassLRU's long-memory scoring (half-life 128 tokens) replayed on our tier: decay 0.7..0.986 within 0.5 points at 32 swaps/window - no gain, not ported |
| Runtime architectures 2026 (disaggregated serving, control/execution plane, heterogeneous runtimes, runtime intelligence, single-node applicability) | reports/LLM runtime architectures 2026.md, research_notes/Arhitecturi runtime LLM 2026/ (notes in English) | pasted overview: names real, details inflated (TokenSpeed "zero runtime errors" unsourced; xLLM = jd-opensource/xllm, Ascend-only benchmarks; Exo = agent runtime, 12-53% not "up to 53%"). For pancake: P/D disaggregation, chunked-prefill interleave, overlap scheduler, distributed KV stores do not apply; adopt items 10-12 below |
| Project Maya (mw00/project-maya, MIT, a Strata fork; 2026-11-09) | reports/Avalon upgrades from upstream projects.md, research_notes/Avalon upgrades from upstream projects/project_maya.md | not faster for us (its Qwen engine is a Strata 0.1.27 snapshot); ported loop guard, soft budget close, tolerant tool calls (section "Agent robustness from Project Maya"); SSD direct reads and route-share expert skip left as test ideas |
| Stanford papers 2022-2026 (reports/Lucrări Stanford utile pentru Avalon.md) | research_notes/Lucrări Stanford utile pentru Avalon/ | per-level thinking cap (done, 5dd620fb9); multi-entry prefix cache (RadixAttention); CATS (superseded, item 13); measurement harness |
| MIT/Harvard papers 2022-2026 (reports/Lucrări MIT Harvard pentru Avalon.md) | research_notes/Lucrări MIT Harvard pentru Avalon/ | TEAL over CATS (item 13); GDN chunk 64 (item 18); XAttention opt-in (item 19); learned thinking stop (item 21); bit allocation guidance (Scaling Laws for Precision) |
| Rest of US universities 2022-2026: Berkeley, CMU, UW, Princeton, Cornell, UCSD, Duke, Rice, Stevens, ... (reports/Lucrări universități SUA pentru Avalon.md) | research_notes/Lucrări universități SUA pentru Avalon/ | items 13-17, 20, 22; SuffixDrafter/DraftPolicy already cover TurboSpec/Sequoia-style length choice; QTIP/IQ2_KT not worth a requant; Europe, Asia, Canada, Pacific/Africa/Latin America reports follow |
| European universities 2022-2026: ETH/EPFL, ISTA + Central Europe, UK, France/South/Nordics/Israel (reports/Lucrări Europa pentru Avalon.md) | research_notes/Lucrări Europa pentru Avalon/ | daily files are ISTA-DASLab GSQ-RCO (Q2_0 93.8 vs IQ2_XS 70.3 tok/s on DASLab's card, -0.09 task avg: Q2_0 confirmed); base is 176.9B incl. the 51B n-gram table; "IQ2_XS" experts are IQ2_S/IQ2_XXS/IQ1_M + Q2_0 down; items 23-27; cache-conditional routing (Qualcomm/UvA) = STRATA_ROUTE_PRIOR, finish item 7; security: GGUF-quantization backdoors (ETH "Mind the Gap") -> official files + SHA-256 vs HF |
| Asian universities + labs 2022-2026: Tsinghua/PKU, rest of mainland China, HK/Taiwan/Singapore, Korea/Japan/India, Qwen/DeepSeek/Moonshot/ByteDance/MiniMax/Xiaomi (reports/Lucrări Asia pentru Avalon.md) | research_notes/Lucrări Asia pentru Avalon/ | Qwen3.8 tech report checked against our code: MTP depth already --spec 4 (official 4.07 accepted at 4 steps), drafter has no PLE block and dense SWA attention (SGLang shortcuts already here), PLE via --ple-io direct (not mmap; Europe report corrected), grammar deferred after </think>; Avalon still at --spec-draft-n-max 2 (A/B 3/4); official thinking sampling T=1.0 noted, owner keeps 0.1; items 28-32 |
| Canadian universities 2022-2026: Mila/McGill/Polytechnique/Concordia, Toronto/Vector, Waterloo, Alberta, UBC, Queen's (reports/Lucrări Canada pentru Avalon.md) | research_notes/Lucrări Canada pentru Avalon/ | little for engine speed; "Displacement Is Not Direction" (Toronto): KLD/top-1 stop predicting task score among near-baseline quants -> token forcing only rejects large deviations, a fixed task set picks between close candidates (quality-gate change for items 8, 13, 23); items 33-34; SpAx = 4th mask in item 13; Cactus parked (does nothing under greedy, ~0 at T=0.1) |
| Australia/NZ, Africa, Latin America 2022-2026 (reports/Lucrări Pacific Africa America Latină.md) | research_notes/Lucrări Pacific Africa America Latină/ | little fits: SpecFit marginal new-miss cost (DISCO feature), UNSW restore-equivalence test, UFMG coordinate-descent kernel sweep, UBA top-k entropy signal; items 35-37; the report ends with the series-wide ranking below |

Development pass done 2026-11-08 night (owner: develop everything first, then test and optimize):
- items 3-4 merged from feat/pattern-cache and built (HIP and CUDA compile; expert_seed_test ok; 7/7 Python tests);
- 7900 XT presets use the matching hipBLASLt table (100401);
- item 11 = existing opt-ins STRATA_CACHE_MESSAGE_BOUNDARY=1 + --prompt-cache-tail (anticipatory boundary with the 32K chunks
  the preset already uses; author: first edit -33%) -> test arms;
- item 12: the thinking budget already exists (reasoning_budget_tokens, #123); Dynasor-style early exit deferred (needs a
  forked probe of the single decode stream, lossy, pays on long math reasoning, not agent coding);
- item 10 (MTP draft inside the verify graph) waits for a draft timing run (launch overhead vs the MTP layer's compute);
- item 8 (VERIFY_DEFER_CPU on HIP) waits for the bisect runs;
- Avalon source 88313b6 built (binary rollback: Avalon/bin-c24c03d-rollback; the model is still missing).
Test phase queue: scratchpad queue.sh order - GPU profile + draft timing first (they steer development), then the A/B arms:
hipBLASLt 100401, --lookup-max-t 12/16, pool pre-read, prompt lending/chunks, the small-gains stack, --mtp-q4, item 11
switches, items 3-4 seeds (needs conv_trace runs), the deferral bisect, then token forcing of the final preset.

Left, ranked (details of the measurements above stay in this section's history):
1. GPU path: the GPU's own work is now 15.9 of a 24.8 ms short window - profile (rocprofv3), then GDN megakernel (rdna3),
   group A switches one at a time, hyper-connection reads (~2.7 ms), the VRAM-hit expert phase (3.3 ms).  Exact.
2. Keep the decode cache through long prompts: the prompt path borrows 9,412 cache slots (12.6 GiB); decode after an 11.7K
   prompt drops ~126 -> ~105 tok/s while it refills.  Exact.
3. Per-conversation expert working set (restore a resumed conversation's hot experts during its prompt).  Exact; needs
   per-session routing traces first.  CODE DONE on feat/pattern-cache (STRATA_CONV_WORKSET), see "Items 3-4" below.
4. Per-domain expert profiles (code / prose / tools, chosen from the prompt).  Exact.  CODE DONE on feat/pattern-cache
   (STRATA_DOMAIN_PROFILES); the offline gain is mixed until per-domain traces exist, see "Items 3-4" below.
5. Tail-row drop: replay with real MTP confidences before building.  Exact.
6. Q2_0 as a 7900 XT preset if its test wins (quality check first; a different quant of the model).
7. Token-forcing check of turbo-hip on the 7900 XT.
8. VERIFY_DEFER_CPU on HIP (non-finite with the ported kernels; lossy opt-in).
9. Phase 7 leftovers: sleep-in-place, automatic disk sessions.
10. MTP drafting inside the captured verify graph (9% of a window): first measure how much of the 2.2 ms is launch overhead.
11. Agent-turn-aware cache policy: anticipatory checkpoints at turn boundaries, one system/tool prompt prefix shared across
    sessions (with the coding agent's conversation working set, item 3).
12. Thinking budget with a natural transition phrase, and Dynasor-style early exit of reasoning (opt-in: changes the text,
    shortens replies rather than speeding tokens).
    2026-11-09: per-level caps (reasoning_budget_by_effort, 5dd620fb9) and Maya's soft close at a sentence end are in;
    the learned stop (item 21) replaces Dynasor as the early-exit route.
Added 2026-11-09 from the university paper series (Stanford, MIT/Harvard, rest of the US; reports/ "Lucrări ..."):
13. CPU-expert activation sparsity, lossy opt-in: one offline FP32 pass on opencode traces with three masks - Berkeley's
    intra-expert gate mask (arXiv 2605.08575: Qwen3.5 MoE at 84-87% sparsity, 95% of score), TEAL (MIT; holds 40-50% on
    dense) and CATS (Stanford; collapses at 40% on Llama-3-8B) - at 50/70/85%, gated by token forcing.  Build the AVX2
    gather kernel (+ W_down repacked n_ff-major at load) only if the gate mask passes at >= 70%.  Ceiling ~1.25x decode
    (8.9 ms of CPU-bound time above the 15.9 ms GPU floor), realistic 10-20%.
14. Expert cache, time-based ceiling first: extend the replay with PCIe 3.0 promotion cost (~115 us / 1.4 MB vs ~42 us CPU
    compute), DDR contention and GPU slack; compare tier, LRU-with-promotion and a Belady that skips non-paying promotions
    (Baleen).  Only if >= 5% of decode is on the table: SIEVE/S3-FIFO dynamic segment beside the frequency tier, admission
    priced by that cost; then OpenEvolve policy search on the simulator scored on decode time (Berkeley ADRS) or a
    Hawkeye-style Belady-imitation classifier.  (Replay showed tier 79.6% < LRU 89.5% < Belady 94.7%, promotions free.)
15. Cross-session suffix index for SuffixDrafter (CMU SuffixDecoding: tree over prompt + all earlier outputs, frequency
    scoring, shared by sub-agents) - simulate with the drafter's --simulate first; check DraftPolicy's learned MTP table
    against Berkeley's measured MTP decay (0.92 / 0.68 / 0.38).
16. Checkpoint triggers at tool-call and tool-result boundaries (FreeToken, Marconi) and Marconi's FLOP-saved-per-byte
    eviction across the parked tiers; a checkpoint after </think> only if the template keeps earlier thinking.
17. Pre-stage the expected expert set while a tool runs (KVFlow applied to experts) - simulator first (L3 prefetch and
    DraftExpert windows failed; this targets the idle tool window, not the decode window).
18. GDN prefill: chunk 64 with blockwise 16x16 inverse + fused prep kernels (MIT GDN/FLA line; 5-11% of prefill).
19. XAttention block-sparse prefill above ~64K, lossy opt-in, gated by code-recall tests (20-40% prefill at 131-262K, est.).
20. ReplaySSM audit of the GDN replay (1-3% decode; up to 8% on speculation only if a snapshot is copied per draft token).
21. Learned thinking stop: a linear probe on hidden states trained offline on logged turns (Thought Calibration, OS-Pruner,
    Learning How Hard to Think; 20-40% fewer thinking tokens on routine turns, untested on agents).
22. FreeToken's PCIe/CPU miss split vs --pcie-frac: read the code, very likely the same mechanism - close unless its
    copied experts are also promoted into the cache.
Added 2026-11-09 from the European report:
23. AcceptMoE-style depth-graded verify prior (Imperial): no bonus at draft position 0 (the committed token stays exact),
    growing cache-residency bonus with depth, weighted by DraftPolicy's measured acceptance; lossy opt-in, token forcing.
    Different from the DraftExpert windows: it changes the verify expert set, not the window length.
24. IQ1_M expert rows (3 layers of the "IQ2_XS" file's gate/up) take the ggml-cpu path in src/kernels/cpu/iq_avx2.cpp:
    add a native AVX2 IQ1_M row kernel (1-4% decode on the IQ2_XS preset, 0 on Q2_0; iq_avx512.cpp's comment is wrong).
25. ISTA-DASLab Coder variant (256 of 512 experts, 29.6 GB resident, SWE-bench Verified 95.7% of BF16, ~3.5 bpw LUT types):
    A/B one evening; compare shard-2 hash first so only ~29.6 GB is fetched; regenerate profile 8.6 and items 3-4 seeds
    (expert ids change).  D space competes with the Avalon model (~20 GB short).
26. Semantic constraint on the edit tool's oldString (must occur in the file) and grammar-constrained tool-call JSON inside
    speculative decoding (ETH type-constrained decoding, PLDI 2025: >50% fewer compile errors) - fewer failed turns.
27. Item 14 simulator gets Harmonia/Athena (SAFARI, online-learned placement) and MoE-Infinity (activation-matrix match)
    as candidate policies; item 13 gets the Polish/Sapienza per-layer thresholds (no new item).
Also: presence_penalty must stay 0 for tool calling (DASLab MoESQ card: SWE-bench 74.7 -> 70.6 with it on).
Added 2026-11-09 from the Asian report:
28. Item 8 rank guard (KTransformers, SOSP'25): STRATA_VERIFY_DEFER_CPU_BELOW (0.1) and "never the last layer" exist; add "always
    immediate when in the token's top-2", measure at verify windows 1/2/4/8 after the HIP non-finite bisect.  Hand-off stays the
    doorbell/partial-flag protocol (no HIP graphs).
29. Per-expert barriers in the CPU pool instead of the global run_phase sync between gate/up and down, SiLU*mul fused into gate/up
    (Tsinghua OSDI'26) - a few percent of CPU expert time.
30. Offline recall test of three next-expert predictors on routing traces (next-layer router on the current hidden state; the
    pre-attention state; the MTP hidden state), recall@10/16/24; use only with PCIe 3.0 promotion cost counted (feeds item 14).
    ProMoE scheduling (chunked, cancellable transfers; cached -> in-flight -> CPU order) goes into the item 14 simulator first.
31. preserve_thinking end to end: server and template honour it; check that opencode resends earlier reasoning, else the
    conversation cache re-reads from the first dropped block every turn.
32. Gated Residual kernel count at batch 1 (SGLang small-M fusion idea) inside the item 1 GPU profile; FP8 irrelevant on RDNA3.
Item 13 also gets CHESS per-channel thresholds, union sparsity over verify windows, transposed down layout.  Item 19: BFLA
(CityU; benchmarks Flash-Next itself, 1.34-1.44x TTFT at 128-256K) is the candidate instead of XAttention - port in progress on
feat/bfla-hip.  Any-Precision for draft passes does not apply (MTP drafter keeps all 512 experts in VRAM, 835 MiB).
Added 2026-11-09 from the Canadian report:
33. Loop guard behaviour counters (UBC/Queen's: failed agent runs cost 1.4-4.4x the tokens, agents do not fail fast): N edits
    without a build/test, M reads/searches without an edit - log-only first, then a note like the tool-repeat note.
34. Rumination signal in thinking (McGill/Mila Thoughtology): rolling 5-gram repetition rate feeding the soft close
    (reasoning_budget_grace) and the learned stop (item 21).
Quality gates (revised): token forcing rejects large deviations only; choice between close variants by a fixed task set
(LiveCodeBench subset + BFCL v4 multi-turn + own-repo checkpoint grading), repeated runs as distributions.
Added 2026-11-09 from the Pacific/Africa/Latin America report:
35. Restore-equivalence test on the real engine: a full-hit restore of a parked Qwen conversation must match a cold recompute
    token for token (check cache_n/prompt_n off-by-one; UNSW found such a bug in a hybrid-state cache).  Precondition for
    items 3, 11, 16, 31.
36. DISCO (feat/disco) features: per-position marginal NEW missed experts (SpecFit, Beihang/Monash; log the variance first)
    and top-k logprob entropy (UBA Entropy Sentinel), the latter also as a log-only loop-guard signal (items 33-34).
37. UFMG coordinate-descent sweep of HIP launch/tile parameters and AVX2 block sizes, after the item 1 GPU profile (1-3%).
Series-wide ranking (all seven university reports; precondition: the revised quality gates):
  1) GPU profile + kernel tuning (1, 32, 37)  2) turbo-hip decision (7)  3) offline sparsity masks + per-expert pool barriers
  (13, 29)  4) correct, cheap per-turn cache with the restore test (35, 31, 16, 11)  5) keep the decode cache through long
  prompts (2)  6) fix item 8, then the rank guard (8, 28)  7) DISCO / SpecFit / AcceptMoE (15, 36, 23)  8) effort and thinking
  (12, 21, 33, 34)  9) time-based cache simulator with candidate policies (14, 27, 30, 3-4, 17)  10) tool reliability (26,
  presence_penalty 0).  Item 6 closed (Q2_0 is the daily file); item 24 gives 0 on Q2_0.  feat/contextpilot: keep off
  (1.6% dedup savings measured on 40 real opencode sessions).
Agent-harness items from the same reports (opencode, outside the engine): low/medium default effort with escalation on a new
external signal, ACE-style delta compaction / Slipstream async compaction in tool-idle windows, Recursive-LM tool for huge
inputs, AsyncFC, GEPA on the compaction prompt.  Quality gates for lossy items: token forcing -> BFCL v4 multi-turn +
LiveCodeBench -> HELMET RAG/re-rank -> Terminal-Bench subset.
(Item 8, VERIFY_DEFER_CPU on HIP, is the highest-ceiling lossy option per both research reports: up to 1.45x in KTransformers.)
Blocked: the 3060 GPU run (card out), two-stage prompts (second GPU), DeepSeek V4.1 on gfx1100 (257 GB, D too small).
Owner's call: merging port/faststrata-poly -> exp/expert-prefetch -> feat/grammar -> port/rdna3-decode into main; BIOS iGPU
memory 4 -> 2 GB; DDR4-3600 at FCLK 1800.

### Agent robustness from Project Maya (MIT, mw00/project-maya main 7bf15ac, v1.1.56; 2026-11-09)

Server-side only (no engine change, no speed effect until a loop is seen).  Copied and adapted, tests in
`serve/test_loop_guard.py` (Maya's module tests as they are + pancake server tests) and `serve/test_tolerant_tool_calls.py`.

- **Loop guard** (`serve/loop_guard.py`, on by default; `STRATA_LOOP_GUARD=0` / `MAYA_LOOP_GUARD=0`, config
  `"loop_guard": false`, or a request's `"loop_guard": false|true`).  In the token loop: an exact block repeated back to
  back is seen as it starts; the stream holds back at most 64 tokens of a tail that repeats; 1st loop in the answer
  (1st/2nd in the thinking): the repeats not sent are dropped and the engine continues with repetition penalty 1.15 over
  2 x period (GEN `pen_from` so the window counts from the reply's start); 3rd loop in the thinking closes it with
  `</think>`; a loop that comes back in the answer ends it ("stop", a note if no answer text yet).  Tool calls are never
  cut.  Across turns: a note after a tool result repeated 3x with the same result, the call blocked (answered by the
  server in the prompt, thinking steered, slightly hotter) after 3 notes, "keep going" when a looping conversation would
  end the turn with nothing new (3x max), and the prompt of a looping conversation under 32K chars leaves out what the
  model already wrote.  Counters: `/metrics` JSON `loop_guard`, Prometheus `strata:loop_guard_*_total`.
  Precedence: one-token runs stay #606's (`repeat_stop_tokens`, "length"); thinking loops are #728's when
  `reasoning_loop_recovery` is configured; the guard handles everything else.  DeepSeek (DSML) gets notes but no blocking.
- **Soft thinking budget** (Maya #613), opt-in `"reasoning_budget_grace": N` (Maya uses 512): past the #123 budget the
  thinking closes at its first sentence or line end, at most N tokens (and half the remaining room) later.  Default 0
  keeps the exact #123/#984/DeepSeek small-cap close positions the tests pin.
- **Tolerant tool calls** (Maya #613/#706): an unclosed `<parameter=` tag is malformed (pancake waited for a
  `</parameter>` that never came and lost every word after a streamed call); a malformed call never raises (also with
  `"tool_call_recovery": false`): announced -> left unfinished (#211), else text.  Not ported: Maya's resume-the-thinking
  after an end-of-turn token inside it - both Qwen turn-end tokens are the engine's own EOS here, and the empty-answer case
  is already #1053 (`reasoning_close_retry`).

### Items 3-4: seeding the adaptive tier at a request's start (branch feat/pattern-cache, 2026-11-08)

Why seeding and not longer memory: a replay showed the tier's hit rate within 0.5 points for usage decay 0.7..0.986 a
window at 32 swaps, so the gain has to come from putting the right experts in early.  Placement only: the GPU and the
CPU compute an expert the same way, so no output changes.  Both opt-in, default off (the RTX 3060 line is untouched).

- Engine: include/strata/program/expert_seed.hpp (header-only, tests/program/expert_seed_test.cpp), hooked in the
  --serve loop of src/program/generate.cpp (blocking and asynchronous tier; not the batch slots or the pipelined loop).
  STRATA_DOMAIN_PROFILES=<dir> loads <dir>/<name>.bin; a GEN line's domain=<name> picks one; its top M pairs (M = cache
  slots) score S * (1 - rank / M) and stay a floor under the counts for R rounds (STRATA_SEED_SCALE=2, STRATA_SEED_HOLD=16).
  STRATA_CONV_WORKSET=N (1 = 8) keeps the outgoing conversation's routing counts at every conversation switch (keyed by its
  tokens: works with or without the parked K/V) and adds them back, scaled to STRATA_CONV_WORKSET_TOKENS=32 tokens'
  worth, when a prompt continues a kept conversation.  A continuing turn is never seeded; ckpt=0 side requests are skipped.
- Server: serve/server.py classify_domain (code fences / diffs / code-shaped lines -> code; tools offered or a tool turn
  -> tool; else prose), sent as domain=<name> when STRATA_DOMAIN_PROFILES is in the server's environment
  (serve/test_domain_classify.py).
- Tools: tools/pancake/learn_domain_profiles.py (per-domain profiles from collect_profile_traces.py's corpus traces and/or
  DOMAIN=TRACE specs), tools/pancake/replay_seeding.py (offline replay of both, incl. --requests for conv_trace.py runs),
  tools/pancake/conv_trace.py (interleaved multi-turn conversations with --dump-routing; also the live A/B driver).
- Offline (bench/prefetch/routing.bin, 10,909 slots, 32 swaps every window): plain 81.17%; domain leave-one-out (code)
  82.86% (code edit 77.3 -> 78.8, long 84.7 -> 90.7) but with the prose request seeded in-sample first the held-out edit
  gets 77.1; oracle domain (own counts) 84.03%.  Conversation resumed after another request: +0.7..+1.0 points on its
  rest (bound +1.4..+2.7).  At 160 swaps the domain seed gains 0.5.  One trace of four requests: not enough to trust the
  domain part; the conversation part never went negative.
- Next (do not run beside a benchmark): build, collect traces, replay, then the live A/B:

      cmake --build build-hip -j16 --target strata expert_seed_test && build-hip/expert_seed_test
      cmake --build build-cuda -j16 --target strata          # must still compile (the 3060 line)
      sed -e "s#<STRATA_DATA>#/home/dan/clone-of-d/Strata-data#g" -e "s#<REPO>#$PWD#g" -e "s#/build-cuda/#/build-hip/#" \
          configs/pancake/flashnext-iq2_xs-fast-hip.json > pancake-flashnext-iq2_xs-fast-hip.json   # (what run-pancake.sh writes)
      export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib HIP_VISIBLE_DEVICES=0
      .venv/bin/python tools/collect_profile_traces.py pancake-flashnext-iq2_xs-fast-hip.json bench/domain-traces --mode routing
      .venv/bin/python tools/pancake/conv_trace.py pancake-flashnext-iq2_xs-fast-hip.json bench/conv-trace --tag base
      .venv/bin/python tools/pancake/learn_domain_profiles.py --out data/domain-profiles --traces bench/domain-traces --slots 10909
      python3 tools/pancake/replay_seeding.py bench/conv-trace/base.trace data/expert-profile-pancake.bin 10909 \
          --requests bench/conv-trace/base.requests.json --domain-dir data/domain-profiles
      STRATA_CONV_WORKSET=1 STRATA_DOMAIN_PROFILES=$PWD/data/domain-profiles \
          .venv/bin/python tools/pancake/conv_trace.py pancake-flashnext-iq2_xs-fast-hip.json bench/conv-trace --tag seeded --domain-keys

  (compare base vs seeded decode_tok_s per request in <tag>.requests.json; the corpus traces are short decode-only
  replies, so add code-edit traces with long context, e.g. bench_engines.py runs with --dump-routing, as code=TRACE.)

## Phase 12 (future): a second GPU

The owner plans a second card.  These were judged "does not apply" on one GPU (research reports above) and become the work
list once two cards are in the box.  Each needs its own A/B on the real pair; the gains quoted are other people's hardware.

| Technology | Source | Why it waits for the second card | Expected value with two cards |
| --- | --- | --- | --- |
| Tensor parallel decode (every layer on both cards) | Strata-rdna3 src/core/tp.cpp, --tensor-parallel 2 | needs two cards | rdna3: IQ3_S 2x 7900 XTX 129-170 tok/s vs 112-143 on one |
| Layer split + pipelined windows (stage 0 runs the next window while stage 1 verifies) | upstream --layer-split, --pipeline-windows | needs two cards | overlapped verify windows |
| Two-stage prompts (late layers of a chunk on the second card) | StrataFast --prompt-split, MOD_PLAN 8.7 | needs two cards | prompt reading |
| Peer expert transfer / RCCL, idle-queue parking (kfdq) | Strata-rdna3 peer_experts.cpp, peer_transport.inc, kfdq.cpp | 2-GPU only; RCCL headers needed (librccl.so present) | expert traffic between cards; idle power 98+91 W -> 5+13 W |
| Prefill / decode disaggregation (one card reads prompts, the other decodes) | llm-d 1.0, NVIDIA Dynamo, Mooncake, DistServe, SGLang PD | Strata's own SPLIT_SERVE lost 4-48% on two GPUs; transfer over PCIe 3.0 without RDMA | test only for agent sessions with long prompts while a reply decodes |
| KV streaming between cards (NIXL / Mooncake transfer engine style) | vLLM connectors, Mooncake | no second device | pairs with the item above |
| Chunked-prefill interleave, overlap scheduler, reasoning-aware scheduling (PASCAL) | vLLM V1, SGLang, TensorRT-LLM | protect OTHER users' decode; one user here | only if pancake serves several users or agents at once |
| GDN megakernel (one persistent kernel per GDN layer) | Strata-rdna3 megakernel_gdn.cu | 0.0 ms on one GPU while CPU experts gate the window; measured 1.6% on rdna3's TP config | revisit with TP |
| Heterogeneous-accelerator decoupling (xLLM service/engine split, vLLM platform plugins) | jd-opensource/xllm, vllm-ascend | single AMD card | only for a non-AMD second card |

## Risks

- Code drift: FastStrata's options live in `src/poly`, not in `src/qwen35`; porting may mean rewriting, not copying.
- HIP parity: several gains rely on NVIDIA tensor cores or CUDA graphs.
- The 7900 XT is not detected today; AMD work is compile-only until it is back.
- Disk: the root disk is tight; builds, models and caches go on the D drive (`/home/dan/clone-of-d`).
- Upstream moves fast (thousands of commits a month); rebase on Strata releases, not on every commit.
