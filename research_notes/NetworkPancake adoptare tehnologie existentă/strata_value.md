# Strata engine value vs llama.cpp / ik_llama.cpp / KTransformers (as of 2026-11-10)

Repo = C:\Users\iancu\pancake-main (read-only; nothing built or run). Tags: [M] measured (by the cited source), [D] derived from cited numbers, [E] estimate/inference, [U] unverified. "Strata" = pancake-strata (MIT fork of Niko1221/Strata). Repo paths are relative to the repo root.

## 1. Is Strata's advantage on a 6 GB card (GTX 1660 Ti, Qwen3.6-35B-A3B UD-IQ2_XXS) real, unproven, or absent?

### Takeaway
UNPROVEN on the exact question. The repo has no llama.cpp number on the 1660 Ti and says so itself. The only same-machine Strata-vs-llama.cpp A/B in the repo is for a different, much larger model (Qwen3.8-Flash-Next 177B, IQ3_XXS) on 12-16 GB cards, where Strata decoded 3-5x faster than a lightly tuned llama.cpp v0.6.0; that is real evidence of an engine-level gap for a RAM-bound MoE, but it does not transfer to a 35B model that fits in RAM. Public llama.cpp reports on 4-6 GB cards (about 17-20 tok/s on Qwen3.6-35B-A3B Q4_K_M) are in the same range as Strata's 34 +/- 14 tok/s at IQ2, so the 1660 Ti advantage is plausible at best, absent at worst.

### Cited Findings

Strata's own 1660 Ti numbers [M]:
- Hardware: GTX 1660 Ti 6 GB (sm_75, no tensor cores), 13.9 GB RAM, Windows 11, CUDA 12.8.61, MSVC 14.44. CPU model is NOT stated (only "system CPU 4-7%"); "8 CPU threads". — `docs/WINDOWS_CUDA_SM75.md:5-6, 36`
- Model: unsloth/Qwen3.6-35B-A3B-MTP-GGUF UD-IQ2_XXS, 11.01 GB, arch qwen35moe; 2319 of 10496 routed experts in VRAM (2.1 GiB), rest on 8 CPU threads. — `docs/WINDOWS_CUDA_SM75.md:34-36`
- Decode (`tools/pancake/bench_win.py 3 3 3 96`): 3 fresh-server runs, 3 warm-up requests uncounted, 3 measured per prompt, 96 new tokens, timed first-to-last streamed token. Short prompt 34.3 +/- 14.1 tok/s (39.8, 34.7, 28.4); code 34.2 +/- 12.5 (37.6, 28.4, 36.5); TTFT 0.62 / 0.94 s. — `docs/WINDOWS_CUDA_SM75.md:40-47`
- Repo's own caveats: "three runs do not separate a 5 tok/s change"; first requests 2.0-15 tok/s, rising ~10x over the first requests (file cache and swaps, not separated); RAM full gave 2.0 tok/s (683 ms/token on CPU experts); only two prompts, greedy, "shows the output is not broken and nothing more"; no same-day `main` comparison (main does not build on MSVC); no quiet-machine gate; no long context, parallel, quality gate. Date: written during the 2026-11 pancake work (no explicit date in the file). — `docs/WINDOWS_CUDA_SM75.md:49-73`
- No llama.cpp run on this machine appears anywhere in the repo for this model. Grep of README, docs, MOD_PLAN, bench for 1660/llama.cpp gives only the 3060/P100/5070 head-to-head below. [M: absence verified by grep, 2026-11-10]
- The 7900 XT decode numbers in README Table 2A are on a different card and backend and the repo says so (`docs/WINDOWS_CUDA_SM75.md:73`).

Strata vs llama.cpp same-machine A/B in the repo [M], different model:
- Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS GGUF (70.6 GiB, 177B, ~3B active), same GGUF read by both engines, llama.cpp v0.6.0 (tag b11429) built -DGGML_CUDA=ON, `-fa on -ctk q8_0 -ctv q8_0`; Strata setup config (MTP, int8 KV, expert cache auto). llama.cpp "tuned" = `-ngl 99 -ncmoe N -b 2048 -ub 2048` with N from a small llama-bench sweep (2-3 values). — `bench/results/2026-10-06-llamacpp-0.6.0/README.md:1-33`
- RTX 3060 12 GB, 46 GB RAM, Core Ultra 7 265, Linux: Strata decode 41.6-41.9 tok/s (4K) and 38.8-41.8 (32K); llama.cpp tuned 9.1-9.8 (4K) and 8.7-8.8 (32K) = 4.3-4.7x. Prompt 915 vs 337 (4K), 1050 vs 338 (32K). Tesla P100: 31.1 vs 10.3 decode (3.0x). RTX 5070 Windows, ONE round, llama `--fit on` untuned: 50-58 vs 14-16. — `bench/results/2026-10-06-llamacpp-0.6.0/README.md:35-102`
- n = 3 rounds per engine (32K), 6-8 at 4K, engines alternated per round, medians. Repo caveats: small tuning sweep only (-ncmoe, -ub), no `-ot` patterns, no thread tuning, no `--no-mmap` trial; MTP head for llama.cpp was a separate unsloth head (different from Strata's native head), and llama.cpp MTP was slower than no-MTP; "heavier tune may add some percent, not 4x"; model larger than RAM so page-cache effects; P100 shared with other jobs; no perplexity/KL. — `bench/results/2026-10-06-llamacpp-0.6.0/README.md:104-120`
- Repo's own explanation of the gap: CPU-resident experts are the bottleneck; Strata keeps a learned/adaptive set in VRAM, uses MTP with no extra VRAM penalty, avoids copying. [D from the README text, not an ablation]

llama.cpp on 4-6 GB cards with Qwen3.6-35B-A3B (external) [all Q4_K_M, none IQ2, none a 1660 Ti]:
- GTX 1650 4 GB, i7-7700 (4 threads used), 31 GB DDR4-2133, Ubuntu, llama.cpp v0.4.0 commit 5266f24, UD-Q4_K_M 22.1 GB, `-ngl 99 -ncmoe 40 -fa 1`: 20.2 tok/s decode (llama-bench tg128, -r 3); `-ncmoe 38`: 20.6; server prefill 66 tok/s on a 2,343-token prompt; `-ncmoe 37` OOM. Same box RTX 3060 12 GB: 28.1 (ncmoe 40), 38.6 (ncmoe 24). [M, third-party blog] — [InsiderLLM GTX 1650 vs RTX 3060](https://insiderllm.com/guides/qwen3-6-35b-a3b-gtx-1650-4gb-vs-rtx-3060/)
- GTX 1060 6 GB, i3-8100 (4 cores), 24 GB DDR4: baseline `--n-gpu-layers 20` 3 tok/s; `--n-cpu-moe 41` 10; `--no-mmap` 13.5; `--n-cpu-moe 35` 17; 256K ctx with turbo KV 17. How measured not stated; page says it is mostly AI-generated, flags inconsistent (35 vs 36). [U] — [AI-Suka summary of a Codacus video](https://ai-suka.pages.dev/2026/05/09-llama-cpp-35b-moe-6gb-vram/); InsiderLLM lists the same ~17 as community, flags unrecorded — [InsiderLLM best way](https://insiderllm.com/guides/best-way-run-qwen-3-6-35b-moe-locally/)
- RTX 3070 Ti 8 GB, Qwen3.6-35B IQ3_XXS, `-ncmoe`: ~8.7 to >40 tok/s (member-only Medium post; body not retrievable, 403) [U] — [Medium](https://medium.com/coding-nexus/one-flag-tripled-my-tokens-per-second-on-an-8gb-gpu-238f55d1e3bd) (snippet via search only)
- Estimator pages (willitrunai: 1060 6 GB 2.0-2.6 tok/s, 1660 Ti 3.6 tok/s for Qwen3-30B-A3B Q4_K_M) are calculators, not measurements. [U] — [willitrunai 1660 Ti](https://willitrunai.com/de/can-run/qwen-3-30b-a3b-on-gtx-1660-ti-6gb)
- I found no measurement of ANY engine on Qwen3-30B-A3B or Qwen3.6-35B-A3B at IQ2 on a 6 GB card, and none on a GTX 1660 Ti specifically.

llama.cpp expert cache PR #29887 (mainline now):
- "llama : add a GPU cache for MoE experts kept in host memory", author am17an, merged by ggerganov (page shows 2026-10-07; opened 2026-10-03). `--moe-cache-mib N`; LRU cache for MUL_MAT_ID on host-resident experts, only misses uploaded over PCIe, applies to batches <= 32 tokens (decode only), start ~10% of expert size. Author benchmark: Qwen3.8-Flash-Next Q4_0 (93.7 GiB), EPYC 7742, PCIe 4.0 x16: RTX 4090 25.0 -> 39.4-40.7 tok/s (1.57-1.62x, 72-77% hit); RTX 5090 30.8 -> 54.5-67.8 (1.77-2.20x, 77-89% hit). — [PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887)
- Limits: single device at first; multi-GPU via #30112 (2x4090: generation 1.74x with `--fit` but prompt 0.73x with `-cmoe`); one report of GLM 5.3 Flash 2-GPU generation falling 33 -> 16; Vulkan 23 -> 18.1, ROCm layer split 25 -> 23; RTX 3090 cache only 1-5% faster than hand-tuned `--n-cpu-moe 44`; R9700 11.0 -> 27.6 simulated small card; RTX 5060 Ti 16 GB 14.95 -> 19.26 (fork); T4 16 GB Qwen3-30B-A3B 22.6 -> 43.2 (third party). No card under 16 GB benchmarked; 8 GB sizing question unanswered. PR commenter break-even ~85% hit rate on PCIe 4.0 x8 (calculation, not measurement). — [DevelopersDigest](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026); [RunAIHome](https://runaihome.com/blog/llama-cpp-v0-7-0-moe-cache-gpu-guide-2026/) (says it did not reproduce)
- A CPU share for cache misses (#30966, merged 2026-11-07 per the repo) is also in mainline. — `MOD_PLAN.md:833`; third-party mention in search result list ([DevelopersDigest / release notes](https://newreleases.io/project/github/ggml-org/llama.cpp/release/v0.7.0))
- The repo itself says Strata "already does both" (hybrid CPU/GPU experts, swaps, `--expert-profile`) and that its PCIe share is the same split as #30966. — `docs/PANCAKE_FINDINGS.md:154`, `MOD_PLAN.md:833`
- The repo's own simulator says online expert-cache admission gains only +5..16% over a plain second-miss policy (oracle +27..48%); predictor lookahead recall 19-21% so "predicted promotions do not pay". These are model/simulator numbers on the 7900 XT box, not 1660 Ti. [D] — `PANCAKE_STATUS.md:16, 19`

### Inferences
- [D] Strata's 34 tok/s at IQ2_XXS (11 GB file) vs llama.cpp's ~17-20 at Q4_K_M (22 GB file) on 4-6 GB cards: the files differ 2x in bytes; CPU-resident expert bytes scale roughly with file size, so most of the gap could be quant size alone. A llama.cpp IQ2_XXS run would plausibly land well above 17-20 tok/s. [E] the same-file result is the one that matters and does not exist.
- [E] The honest prior: llama.cpp `-ncmoe` with IQ2_XXS on a 1660 Ti with a decent DDR4/DDR5 CPU is plausibly in the 20-35 tok/s range; Strata's 34.3 mean sits at the upper edge, and its +/-14 CI overlaps all of it. Strata's advantage on this card is therefore unproven (and could be zero).
- [D] Where the advantage is real: a model whose experts do not fit in RAM or where CPU experts dominate (the 177B/3060 A/B shows 3-5x). For 35B-A3B at Q4 or IQ2 on 12 GB+ cards, the repo claims "100+ tok/s on the 3060" for Qwen3.6 (`docs/PANCAKE_FINDINGS.md:108`, quant not stated in that line) but no llama.cpp baseline.
- Strata's cold-start behaviour (2-15 tok/s for the first requests, ~10x climb) is a weakness vs static `-ncmoe` placement, which is fast from request 1 (the 1650/1060 reports show no warm-up curve). [D]
- Strata's learned-cache design is directionally the same idea as #29887 (hot experts in VRAM, misses to CPU), which is now upstream. The moat is thinner than the README table implies; the remaining difference is policy (k-th miss admission, deferral, seeds) and the kernel set, not the concept.

Minimal fair same-machine A/B (required, none exists):
1. Same GTX 1660 Ti PC, same Windows state, quiet machine; record CPU model, RAM speed/channels, thread count (set both engines to the same -t, test 4/6/8).
2. Same file: the exact `Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf`, sha256 recorded. Also repeat with UD-Q4_K_M only if RAM allows (22 GB does not fit 13.9 GB RAM; mmap paging would dominate).
3. llama.cpp: current mainline release with CUDA, built for sm_75 (`-DCMAKE_CUDA_ARCHITECTURES=75`; note no tensor cores, try `GGML_CUDA_FORCE_MMQ`), `-ngl 99 -fa on --n-cpu-moe N` sweep (N from the OOM edge up to 40, ~6 values), `--no-mmap`/`--load-mode none` vs mmap, `-ub`/`-b` 512 vs 2048, then `--moe-cache-mib` sweep (e.g. 1000/2000/3000 MiB with `-cmoe`). Also ik_llama.cpp with `-ot exps=CPU` plus selective layers if it has the quant.
4. Metric parity: decode tok/s from the same timing source (server timings), same prompts (short + code + a ~4K one), 96-256 new tokens, greedy, warm-up of 3+ requests for BOTH engines, report the first-request curve too, plus TTFT and prefill tok/s.
5. n >= 8 alternating rounds (engine order alternated), report median and 95% CI; a gap is only claimed if CIs separate. Add a RAM-pressure variant (other apps open) since Strata collapsed to 2 tok/s when RAM was full.
6. Quality check: same-token forcing or KL on 14K tokens (repo has `tools/pancake/quality_gate.py`), because Strata defaults may include lossy options (the repo notes upstream TAIL_SKIP is lossy; pancake presets set it to 0).

### Gaps
- No llama.cpp, ik_llama.cpp or KTransformers number for Qwen3.6-35B-A3B IQ2_XXS or Qwen3-30B-A3B on any 6-8 GB card; no 1660 Ti measurement of anything but Strata. Searched r/LocalLLaMA mirrors (reddit.sentinel-team.org snapshots appeared but I did not open them), blogs and estimator sites.
- Strata's 1660 Ti CPU/RAM-speed unknown, so llama.cpp results from other CPUs cannot be normalised.
- Strata CI +/-14 tok/s with n=3 cannot support any claim finer than "20-50".
- Strata engine repo has no date stamp on WINDOWS_CUDA_SM75.md; ordering inferred from README (2026-11).

## 2. Which Strata features are additive to a distributed platform, which are single-node only?

### Takeaway
The additive pieces are the ones that live on one node's expert-resident path (per-node learned expert cache, expert profiles, conversation/prefix cache, quant kernels). Strata itself has NO network or distributed mode: the repo says LAN round trips per layer "would cost more than a whole token takes now" and treats multi-GPU as one PC. Speculative windows hiding network latency is a plausible, but unbuilt, use of Strata's speculation machinery. Strata's bf16/fp16 wire formats are not documented as a network feature.

### Cited Findings
- Strata multi-GPU is layer split within one PC, same-make cards ("two or more graphics cards of the same make in one PC"). — `docs/MULTI_GPU.md:1-4`
- Split serving (prefill on one card, decode on another) hands sessions over via RAM (`/dev/shm`), same PC; measured slower than layer split on two alike cards. — `docs/SPLIT_SERVE.md:3-6, 15-25`
- Aviary (SensAI-PT/aviary-hy3, Apache-2.0): multi-machine cluster on colibri, activations over LAN. Repo verdict: "No gain on one PC; 40 layers of LAN round trips would cost more than a whole token takes now." Listed as "Distributed expert placement across nodes; surveyed, single-machine focus". — `docs/PANCAKE_FINDINGS.md:109`, `README.md:111-130` (table row)
- KTransformers and others rejected as base: "keep the model in VRAM or need far more VRAM or RAM". — `MOD_PLAN.md:39-41`
- Per-node expert cache pieces that exist: `--expert-cache N|auto`, learned profile save `expert_profile_save`, `--expert-profile` at start, k-th-miss admission (`STRATA_ADAPT_KMISS`), async tier, domain/conversation seeds. — `docs/DETAILS.md:390-400`, `PANCAKE_STATUS.md:16-23`, `PANCAKE_STATUS.md:18`
- Speculation: MTP head (own and unsloth), n-gram/suffix, DFlash2 gate idea from ik_llama.cpp PR 2761, DISCO learned draft stop (branch `feat/disco`, not merged), `--spec N`, cost-aware window pricing (`DraftPolicy`, `STRATA_SPEC_COST_TABLE`). — `README.md:98-101, 129`, `PANCAKE_STATUS.md:53`, `docs/PANCAKE_FINDINGS.md:139-150`
- Conversation cache + disk tier: `STRATA_CONVERSATION_DISK_DIR` (opt-in, exact vs re-read), RAM-budget conv cache, tool-boundary checkpoints, turn checkpoints; restore equivalence tested by `tools/pancake/restore_check.py`; "Test 0%" status for several. — `PANCAKE_STATUS.md:76-89`
- KV: `--kv-q8` default (q8 KV essentially free per cited KLD study), `--kv-q4` keeps keys Q8, `--kv int8`. — `docs/PANCAKE_FINDINGS.md` section 8 (KV KLD row, ~lines 160-162)
- Long context BFLA: block-sparse prompt attention, 128K end-to-end prompt 386.9 -> 771.2 (dense) -> 1219.8 (sparse) tok/s on Qwen3.6-35B-A3B UD-Q4_K_XL, 7900 XT; recall 48/48, KL 0.0014/0.0045; branch `feat/bfla-hip` UNMERGED, HIP only, test 10%. — `README.md:62-77`, `PANCAKE_STATUS.md:26`
- CPU sparsity masks (TEAL/CATS) REJECTED offline (experts dense). Predictor lookahead recall 19-21%, closed. — `PANCAKE_STATUS.md:19, 21`
- CUDA graphs: the status table mentions a "captured graph" tied to CPU pre-summing (`STRATA_CPU_PRESUM`) but I did not find a standalone CUDA-graph speed measurement. [U] — `PANCAKE_STATUS.md:48`
- MMQ for IQ2/IQ3/Q2K/Q3K: all tensor types of the UD-IQ2_XXS file (IQ2_XXS, IQ3_XXS, IQ4_XS, IQ2_S, Q2_K, Q3_K for experts; Q4_K-Q8_0 elsewhere) have a kernel in `src/poly/mmq.cu`. These are largely ggml-lineage kernels (README credits llama.cpp for formats and MMVQ lineage). No per-kernel speedup vs ggml measured in the files I read for the 1660 Ti. — `docs/WINDOWS_CUDA_SM75.md:34-36`, `README.md:128`
- Wire formats: I found no bf16/fp16 network wire format in Strata. Split serving and layer split use PCIe/RAM. The speculative "Aviary" ref is the only LAN design and is rejected. [U for any bf16/fp16 wire claim] 
- MOD_PLAN's simulator suggests deferral (CPU experts finish while next GPU layer runs) +33% on code edits and +45-52% with k-th-miss, as a MODEL (alpha 0.25-0.5), not a measurement; measured deferral +15.8% +/-3.9 on edits (HIP). — `PANCAKE_STATUS.md:20`, `MOD_PLAN.md:1499`

Classification [D/E]:
- Additive (node-local, composes with a mesh): per-node learned expert cache and profile (each node sees only its experts/queries: profile is local), k-th-miss admission, conversation/prefix cache + disk tier (per node or per session affinity), KV quantisation, quant kernels (MMQ), BFLA (once merged and for CUDA), speculative decoding on the node that holds the draft head.
- Additive with new work (not present): speculative windows hiding network latency (Strata's DraftPolicy/window cost tables could be extended with RTT), expert-parallel placement using learned profiles to decide which node owns which experts, activation compression for the wire.
- Not additive / single-node: expert deferral and rank guard (assume CPU and GPU on the same node sharing one routing step), PCIe share (`--pcie-frac`), `--shared-expert-arena` (shared RAM copy across engines in one machine), split serving over `/dev/shm`, layer split's same-make-cards requirement, CUDA graph capture tied to the local window.

### Inferences
- [E] A mesh node that serves a layer slice (pipeline) sees all tokens' routing for its layers, so a per-layer cache is well defined; the repo's measured hit-rate gains (e.g. hit 54-61% -> 71-78% with adapt-fetch, RTX PRO 5000, `docs/DETAILS.md:1078`) were measured single-node only.
- [E] Network latency hiding via speculation is most credible with MTP/n-gram drafts verified in batch windows; Strata's own decode wins come from windows of ~4 tokens, so one RTT per window rather than per token. No measurement of this exists.

### Gaps
- No distributed measurements in the repo. bf16/fp16 wire format: none found. CUDA graph benefit: not isolated.
- Not read: `docs/paper/Strata-Paper.pdf`, `pancake-docs/*.md`, `docs/DETAILS.md` in full (large).

## 3. Model coverage limits and licences

### Takeaway
Coverage is narrow: Qwen3.8-Flash-Next (qwen4exp), Qwen3.6-35B-A3B (qwen35moe), DeepSeek V4.1-Flash, plus files for gemma4 and glm5next in src/poly; Qwen3-30B-A3B (qwen3moe) and generic llama-family GGUFs are not supported as far as I found. Licence is MIT (Strata) and all bundled third-party code is MIT; combining with MIT/Apache projects is straightforward with notice retention; model weights have separate licences.

### Cited Findings
- AGENTS.md: "public model families are DeepSeekV4.1-Flash-1017, Qwen3.8-Flash-Next and Qwen3.6-35B-A3B". — `AGENTS.md:3-5`
- Model files in the poly engine: `qwen35`, `deepseek4`, `gemma4`, `glm5next`. Gemma4/glm5next are not in the user-facing model lists. — `README.md:557`, `src/poly/gemma4.cpp`, `src/poly/deepseek4.cpp`, `src/poly/qwen35.cpp` (file existence verified)
- Qwen3.6-35B-A3B: Strata runs only Unsloth GGUFs with the MTP layer ("chosen by the file"); Intel Arc and Mac engines do not run it; listed sizes UD-Q4_K_XL 22.9 GB, UD-Q6_K_XL, Q8_0. — `docs/MODELS.md:283-295`
- No match for qwen3moe or Qwen3-30B-A3B in src, include, docs, README (grep, 2026-11-10). [M: absence]
- Non-goal: dense models (Qwen3.8-27B): "Strata's expert offload does nothing for it". — `MOD_PLAN.md:36`
- Licence: MIT, "Copyright (c) 2026 Niko1221 and the Strata contributors". — `LICENSE:1-21`; README keeps it: `README.md:739-740`
- No NOTICE file at repo root (root listing); per-component notices in `third_party/`: antirez-ds4 (MIT, Metal vision), libstrix (MIT, notice only; licence text upstream), project-maya (MIT, DeepSeek engine `src/dsv41`, retains Maya licence plus Strata copyright), sfnx (MIT, `src/split`), ggml (MIT, `third_party/ggml`, pinned 3cf0325, 2026-09-20). — `third_party/*/NOTICE.md`, `third_party/*/LICENSE`, `third_party/ggml/VERSION.txt`
- Other licences in the repo: web app font SIL OFL 1.1; experimental speed-projection data under Qwen Community License 1.0. — `docs/HOW_IT_WORKS.md:72-76`
- Ideas-only (not code) from: ik_llama.cpp (MIT), KTransformers (Apache-2.0), SGLang, vLLM, FlashInfer (Apache-2.0), colibri (Apache-2.0), Aviary (Apache-2.0). Code ported from: FastStrata (MIT), tandem.cpp (MIT), Avalon (MIT, ideas), basalt, llamAmpere (MIT, ideas). AGPL qwen38-megakernel and GPL-2.0 gpu-unlock: ideas only, no code. — `README.md:105-130`
- KTransformers licence Apache-2.0. — [KTransformers repo](https://github.com/kvcache-ai/ktransformers)

### Inferences
- [D] MIT -> MIT/Apache: a combined work is fine; keep the MIT copyright and notice text for Strata and per-component notices; Apache-2.0 requires its own NOTICE/patent grant handling for the Apache parts. No copyleft was found in the code path.
- [E] "Licence check" is limited: I did not audit every source file header or Python dependency in `requirements*.txt`, and model weights (Qwen Apache 2.0 per `docs/MODELS.md:285`; DeepSeek, Unsloth quant terms) are separate.
- [E] Coverage risk for a mesh: a platform that serves arbitrary GGUFs would have to fall back to llama.cpp for everything outside the four architectures; Strata would be an optional per-model engine, not a general one.

### Gaps
- Strata's engine/architecture list for "4 architectures" is my count (qwen4exp/Flash-Next, qwen35moe, dsv41, plus gemma4/glm5next in poly); not stated by the repo as a single list.
- Per-dependency licences not enumerated.

## 4. Honest ranking: gain from Strata inside a mesh vs plain llama.cpp, and the minimal experiment

### Takeaway
Ranking by evidence strength: (1) large MoE where experts live in RAM on 12-16 GB cards (measured 3-5x decode, 2.7-3x prefill vs lightly tuned llama.cpp, repo's own A/B); (2) agent workloads with long, growing prompts (conversation cache, disk tier, BFLA) - features exist, mostly measured on other hardware or untested; (3) low-VRAM 6-8 GB cards on 35B-A3B - unproven; (4) network-latency hiding - not built. Plain llama.cpp now has the expert cache (#29887) and CPU share (#30966), narrowing (1) in decode; the repo has not rerun against those.

### Cited Findings
- Ranking evidence (1): `bench/results/2026-10-06-llamacpp-0.6.0/README.md:89-104`.
- llama.cpp mainline caught up on the cache concept (#29887, merged ~2026-10-07; #30112 multi-GPU; #30966 CPU share, 2026-11-07). The 2026-10-06 A/B predates the merge, so it compared Strata with llama.cpp WITHOUT `--moe-cache-mib`. [D] — [PR #29887](https://github.com/ggml-org/llama.cpp/pull/29887); `MOD_PLAN.md:833`
- ik_llama.cpp: focus is CPU/hybrid performance with fused MoE ops and tensor overrides; HF guide: its GPU prompt-offload batch threshold is 32 * total_experts / active_experts vs 32 in llama.cpp. One March 2026 Reddit benchmark: ik_llama.cpp 35-40% faster prompt processing on some Qwen3, generation nearly identical; others slower on a 5060 Ti. A dasroot claim of ~110 vs 80 tok/s on 12 GB is without methodology [U]. — [HF blog](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide); [dasroot](https://dasroot.net/posts/2026/05/qwen-3-6-27b-24gb-vram-benchmarking-llama-cpp-ik-llama-cpp-beellama-vllm/)
- KTransformers: AVX2-only CPU backend added March 2026, kernels tuned for AMX/AVX512, INT4/INT8 CPU weights and GPTQ GPU weights; named models include Qwen3MoE, Qwen3-Next, DeepSeek, Kimi, GLM, MiniMax; Qwen3.6 not mentioned; GGUF not explicitly mentioned; no 6-8 GB consumer numbers. The repo rejected it as "needs 24 GB+ card". — [KTransformers](https://github.com/kvcache-ai/ktransformers); `MOD_PLAN.md:39`
- SGLang-KT integration gives comparable performance to native KTransformers on single GPU + CPU (no Qwen3-30B-A3B numbers). — [LMSYS blog](https://lmsys.org/blog/2025-10-22-KTransformers)
- Strata's other claimed per-card results are single-box, e.g. RTX 3060 12 GB IQ2_XS 20.5 GiB experts in RAM, decode 43.26 -> 50.40 tok/s (+16.5%) from `STRATA_RAM_ADAPT` vs Strata v0.1.43 (self-baseline), `docs/DETAILS.md:1484`; `PANCAKE_STATUS.md` explicitly says "All speed numbers measured today are provisional ... other work ran on the machine" and that most items are 0% tested. — `PANCAKE_STATUS.md:9-10`

### Inferences
- [E] Probable gains inside a mesh (ordered): 
  1. Strata as the engine for MoE nodes with a large RAM/small VRAM split, if the model is one of its four (largest and best-measured; but baseline pre-dates #29887, rerun needed).
  2. Prompt/prefix cache and disk tier for agent sessions (sticky routing needed); measured as exact vs re-read, gains in seconds per turn; one number: "90.8% of turns hit the cache". — `PANCAKE_STATUS.md:97-98`
  3. BFLA for long-prompt TTFT (3.15x at 128K, unmerged, HIP only).
  4. IQ2-quant kernels on Turing cards (sm_75) enabling a 35B on 6 GB + 12-16 GB RAM: capability gain (it runs) more than proven speed gain.
  5. Learned profiles/co-activation: repo's own results small (+0.6-1.5 hit points for seeds, lookahead recall 19-21%); lowest-value item for a mesh.
- [E] Risks: Windows/MSVC build needed fixes and four tests do not build; Strata-poly's long warm-up; 4 architectures; single maintainer fork lineage; many features "coded, 0% tested".

Minimal experiment that would prove value (single machine first, then mesh):
- E1 (decisive, 1 day): the same-machine A/B above on the 1660 Ti PC and on a 12 GB card, with llama.cpp mainline >= v0.7 including `--moe-cache-mib`, same GGUF, n >= 8 alternating rounds, median + CI. Pass criterion: Strata decode CI lower bound above llama.cpp's best tuned arm by >= 20%, after warm-up, with RAM-pressure variant; otherwise Strata adds nothing at that tier.
- E2 (mesh): two nodes (or node + client) running llama.cpp RPC/mesh with Strata vs llama.cpp as the per-node engine on the SAME slice; measure end-to-end tok/s at 1-5 ms and 20 ms injected RTT, with spec windows 1/4/8, to test whether Strata's windowed verification hides RTT. No such data exists today.
- E3 (agent): 20-turn agent trace with growing prompts, sticky vs non-sticky routing, TTFT per turn with and without Strata's conversation cache (repo has `restore_check.py`).

### Gaps
- No distributed test exists; no baseline against llama.cpp >= v0.7 cache; no ik_llama.cpp or KTransformers numbers on 6-8 GB cards found; Reddit/Phoronix not searched in depth (search engine returned mirror snapshots I did not open).
- Phoronix coverage of #29887: not searched.
