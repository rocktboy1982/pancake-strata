# Strata companion tools: benchmark methodology, published results, speed-relevant settings (state 2026-11-10)

Scope: the 10 companion repos listed in the assignment, read via shallow `git clone` + `gh api` on 2026-11-10. Where a companion repo only re-publishes upstream Niko1221/Strata community reports (strata-benchmarks), the upstream report folders were cross-checked in the local mirror `/home/dan/clone-of-d/src/Strata/bench/results/`. Our comparison point: RX 7900 XT 20 GB (gfx1100) + Ryzen 7 5700G, 57 GB RAM, `tools/pancake/abtest.py` (ABBA, 3 reps, 95% CI, quiet gate).

## Q1. Per-repo inventory: purpose, last commit, relevance

### Takeaway
Only three repos carry real measurement content: strata-manager (a full benchmark/calibration engine that reimplements upstream `tools/strata_bench.py` and `tools/calibrate.py`), strata-context-benchmark (one heavily documented RTX 4080 study), and strata-benchmarks (a 1,552-row JSONL scrape of upstream community reports, last updated 2026-10-18). The other seven are launchers/containers/monitors whose only speed-relevant content is a few runtime settings (memlock, VRAM reserve, prompt-cache cadence, I/O priority).

### Cited Findings
| Repo | Purpose | Last commit (UTC) | Speed/measurement content |
|---|---|---|---|
| street-tripy/strata-benchmarks | 1,552 measurements, one JSON line each, "collected by the Strata Launcher from Niko1221/Strata's reports"; every row has `source: upstream:<folder>` | 2026-10-18 01:21 ("bench: 1552 measurements") | Results aggregation only; no own method — [repo](https://github.com/street-tripy/strata-benchmarks) |
| rduffyuk/strata-context-benchmark | Reproducible study of one patched Strata deployment on a 16 GiB RTX 4080 up to 256K context; offline replay `reproduce_all.py` | 2026-11-10 00:06 (v1.0.1, docs only) | Detailed method + results — [repo](https://github.com/rduffyuk/strata-context-benchmark) |
| hongfaqiu/strata-manager | Tauri/Rust desktop manager (Win/Linux), benchmark page, "Official benchmark", quick calibration, speed health check, AMD sysfs telemetry; 56 stars | 2026-11-09 17:43 (v0.2.44) | Most complete methodology — [repo](https://github.com/hongfaqiu/strata-manager) |
| Raboo/strata-container | Prebuilt GHCR images (CUDA/ROCm gfx120x+gfx110x+gfx1151) + env wrapper | 2026-11-09 22:37 (documents upstream v0.2.13) | Launch flags only — [README](https://github.com/Raboo/strata-container/blob/main/README.md) |
| HinataSoft/strata-docker | NVIDIA-only Docker builder (pinned default `v0.1.21`) | 2026-09-29 10:34 | Runtime tips (memlock, NUMA) — [README](https://github.com/HinataSoft/strata-docker/blob/main/README.md) |
| dongzn2017/strata-launcher | Windows JSON-config launcher (start/check/stop), "routing learning" profiles | 2026-11-10 00:26 (1.0.0) | Exposes `STRATA_ROUTE_PRIOR`, `STRATA_KVA`, `STRATA_RESIDENT_HEADROOM_GIB`; no benchmarks — [repo](https://github.com/dongzn2017/strata-launcher) |
| VKVstudio/StrataSmartToggle | Win11 taskbar launcher/VRAM release (C#) | 2026-10-29 15:53 | None — [README](https://github.com/VKVstudio/StrataSmartToggle/blob/main/README.md) |
| vodesoft/comfyui-strata-nodes | ComfyUI node that spawns a Strata server | 2026-11-01 20:05 | None — [README](https://github.com/vodesoft/comfyui-strata-nodes/blob/master/README.md) |
| sognix/strata-supervisor | Docker web UI to load/unload model configs; NVIDIA + AMD images; tested locally on one RX 7900 XTX | 2026-11-09 16:22 | One AMD stability finding (VRAM eviction), RAM modes — [repo](https://github.com/sognix/strata-supervisor) |
| yosephyne/stratoscope | Menu-bar/tray live gauge (prefill/decode, cache hits) reading Strata `/metrics` on :8095 | 2026-11-09 21:46 | Metric definitions only — [README](https://github.com/yosephyne/stratoscope/blob/main/README.md) |

- strata-benchmarks schema fields: gpu, vram_gb, cpu, ram_gb, os, driver, model, quant, context, kv, prompt_tokens, decode_tok_s, prefill_tok_s, ttft_s, generated_tokens, n_runs, label, notes, raw; machine names/paths/keys redacted — [README](https://github.com/street-tripy/strata-benchmarks/blob/main/README.md)
- No companion repo has a run log newer than upstream 0.2.13 for AMD; strata-supervisor's AMD tests were on `0.2.9-rocm-gfx110x` (2026-11-04) — [IMPLEMENTATION.md §13](https://github.com/sognix/strata-supervisor/blob/main/AI/IMPLEMENTATION.md)

### Inferences
- For new numbers to compare against, strata-manager's "Official benchmark" is the de facto shared yardstick (it is what its leaderboard ranks on), so adopting the same request set would make pancake numbers directly comparable to Manager users' uploads.
- strata-benchmarks is useful as a pre-parsed index into upstream reports but is a month stale (stops at 2026-10-10 reports) and has many null fields; it should not be cited as a primary source.

### Gaps
- strata-manager's website leaderboard (rduffy-style public results) was not fetched; contents/AMD rows unknown.
- The "Strata Launcher" that strata-benchmarks credits is not obviously dongzn2017/strata-launcher (which has no collection code); provenance of the scrape unclear.

## Q2. Measurement methodology we can compare to or adopt

### Takeaway
Two reusable methods exist: (1) strata-manager's port of upstream `tools/strata_bench.py` (fixed request set, 1 discarded warm-up + calibration + 1K prompt, 3 runs, median with min/max, engine-reported timings) and its port of upstream `calibrate.py` (each value visited 3 times in alternating order, median decides, 2 warm-up rounds, final interleaved confirmation, change accepted only if >3% better); (2) rduffyuk's seeded, shuffled, 5-session design with n=50 per cell, explicit reset/cold/cached sequence, min-max ranges (not CIs), and full exclusion accounting. None of them uses confidence intervals or ABBA explicitly; our harness is statistically stricter, but the alternation/confirmation idea and the >3% gate match it.

### Cited Findings
**strata-manager "Official benchmark"** (`packages/core/src/official-bench.ts`) — [source](https://github.com/hongfaqiu/strata-manager/blob/main/packages/core/src/official-bench.ts)
- Reproduces upstream `tools/strata_bench.py` v0.2.7.1 (commit 46c152a1) "request for request": three short questions from `tools/ab_engine.py` (≤100 new tokens); fresh word-list prompts of ~4,096 / 16,384 (/ 32,768 with `--long`) tokens sized by a two-point calibration (24 new tokens); a coding-agent session of five 1,500-word turns with system prompt + four tools from `tools/agent_turns_bench.py` (64 new tokens per turn).
- Three runs each, medians with min/max (Python `statistics.median` semantics), thinking off, temperature 0. "One warm-up chat, the calibration and a 1K prompt are discarded first." Speeds are the engine's own `prompt_per_second` / `predicted_per_second`; TTFT is client-measured. Order: warmup, chat, 4K, 16K, agent, then 32K. Default parameters = `python tools/strata_bench.py --runs 3 --long` (upstream docs/STRIX_HALO_WINDOWS.md §6).
- Reference table `official-bench-reference.json` copies maintainer-published numbers "as printed (no rounding, no derived values)", each with a link to the source line — [source](https://github.com/hongfaqiu/strata-manager/blob/main/packages/core/src/official-bench-reference.json). Example (only AMD row family present): Ryzen AI Max+ 395 / Radeon 8060S, Windows 11, Qwen3.8-Flash-Next IQ3_S, 131,072 ctx, int8 KV, Strata 0.2.7 RC (engine 0.1.51): chat decode median 44.0 (40.6–59.5) tok/s, TTFT 0.60 s; 4K prefill 755 tok/s; 16K 821; 32K 792; agent hit 69%, agent decode 47 tok/s, agent turn TTFT 4.5 s (measured 2026-10-26).

**strata-manager quick calibration** (`crates/runtime-core/src/tuning_calibrate.rs`) — [source](https://github.com/hongfaqiu/strata-manager/blob/main/crates/runtime-core/src/tuning_calibrate.rs)
- Follows upstream `tools/calibrate.py` / `setup.sh --calibrate`: request-level sweeps through one running engine, "every value visited `SWEEP_ROUNDS` times in alternating order, the median deciding, and a final interleaved confirmation against the current settings."
- Constants: `MIN_GAIN = 0.03`, `SWEEP_ROUNDS = 3`, `WARMUP_ROUNDS = 2`, `MAX_NEW = 128`, prompts = Code / Chat / Chinese scenes, plus long answers (>512 tokens) because "short 128-token chats alone hide the expert-tier and long-answer effects (upstream #2722)". Sweep order reverses each round (`sweep_order([0.3,0.5,0.7],1) == [0.7,0.5,0.3]`), i.e. an ABC/CBA pattern.
- Grids: `PCIE_FRACS = [0, 0.2, 0.35, 0.55, 0.75, 0.9, 1.0]`, `SPEC_MIN_PS = [0.3, 0.5, 0.7]`, `PIPE_VALUES = [0, 1, 768, 1536, 3072, 6144]` at `PIPE_TOKENS = [2048, 8192]`.
- A current value is kept unless another beats it by >3% in the median (fix for upstream #2465). Restart-bound settings (CPU pool workers, expert tier, "AMD shared-expert stream") are not swept live; they become ≤4 restart candidates: two worker counts (default/2, default/4 when in range) and expert-tier presets `adapt_every=1, adapt_swaps ∈ {80,160}, adapt_decay=0.97`.
- A calibration is flagged stale when the engine version changes (upstream #4841: 0.1.55 changed parallel defaults and GDN kernels) — [calibration.ts](https://github.com/hongfaqiu/strata-manager/blob/main/packages/core/src/calibration.ts)

**strata-manager general benchmark** (`benchmark.ts`, `benchmark-suite.ts`) — [source](https://github.com/hongfaqiu/strata-manager/blob/main/packages/core/src/benchmark.ts)
- Optional single warm-up (excluded), `unique` mode prepends `[benchmark <id>/<index>]` to defeat prefix-cache reuse vs `repeat` mode; reports mean, p50, p95 (suite adds p99) of prefill/decode/TTFT and cache-hit rate; warm-up resource observations also excluded. Suite default: 2 rounds, 256 max tokens, 8,192 input chars, 3 conversation turns, steady/ramp stress patterns.

**rduffyuk/strata-context-benchmark** — [README](https://github.com/rduffyuk/strata-context-benchmark/blob/main/README.md), [METHODS.md](https://github.com/rduffyuk/strata-context-benchmark/blob/main/studies/v7/METHODS.md)
- Five consecutive sessions with a deterministic schedule: per session and level, seed `2353 + session*1000000 + level` shuffles ten fixture indices; one warm-up pair then ten measured pairs; each pair runs reset → cold → cached. 1,375 calls = 850 measurements + 85 warm-ups + 440 resets, over 6 h 41 min 22 s; "no unresolved calls or inference retry".
- Eight input lengths up to 261,112 tokens; n=50 per context/cache cell, explicitly "repeated measurement of ten prompts, not 50 independent prompts". Ranges reported are observed min–max, "not confidence intervals". "Cold means zero KV reuse, not a global OS cache flush."
- Rules: retain failures and raw responses, verify input/output/KV-reuse counters independently, never silently retry, disclose any binary difference; evidence manifests are SHA-256 checked.
- Explicitly disclaims cross-day repeatability: "These runs do not establish repeatability across different days."

**Upstream community reports re-published in strata-benchmarks (AMD-relevant methods)**
- RX 7900 XTX ROCm 7.1.1 vs 10.2 nightly: one identical ~61-token request repeated back-to-back, 200 generated; request 1 treated as cold (fresh engine + cold expert tier) and excluded; warm = requests 4–9 (ROCm 7) / 6–10 (nightly) — [local mirror README](../../../../../bench/results/2026-10-03-community-rx7900xtx-rocm102-nightly/README.md), upstream folder `bench/results/2026-10-03-community-rx7900xtx-rocm102-nightly` in [Niko1221/Strata](https://github.com/Niko1221/Strata)
- RX 7900 XTX hipBLASLt 100500: two arms differ only in `STRATA_HIPBLASLT_TUNING`; "medians of 3 clean cells with range" — local mirror `bench/results/2026-10-04-community-rx7900xtx-hipblaslt-100500/README.md`
- 4x RX 7900 XT: per arm, cold first request (32K) + warm-ups at 4K and 32K not counted, then 3 runs each at 4,096 / 32,768 / 128,000; "one process start per arm; no variance study across reboots" — local mirror `bench/results/2026-10-09-community-4x-rx-7900-xt/README.md`

### Inferences
- Our ABBA + 95% CI + quiet gate is stricter than every published Strata method (all use median + min/max of 3, or n=50 min/max). For comparability, report median [min–max] alongside our CI.
- The Manager/upstream calibrate pattern (2 warm-up rounds, alternating order, >3% median gain gate, final interleaved confirmation) is a cheap addition to abtest.py for parameter sweeps (as opposed to A/B of builds).
- Upstream practice of discarding the first request after load (cold expert tier) and treating warm steady state as starting only after several requests (requests 4+ or 6+ on the XTX) suggests our 3-rep design may still contain warm-up-curve effects unless the gate covers expert-cache convergence.

### Gaps
- No companion repo publishes run-to-run variance statistics (SD/CV) or reboot-to-reboot replication.

## Q3. Published results tables usable for comparison (AMD and single-GPU + CPU expert offload)

### Takeaway
No companion repo has its own RX 7900 XT single-card numbers. The closest comparators, all upstream community reports re-indexed by strata-benchmarks, are RX 7900 XTX (warm short-prompt decode 80.4 tok/s on ROCm 7.1.1, 88.4 on ROCm 10.2 nightly, IQ3_S), RX 7900 GRE + Ryzen 7 5700X3D (decode 65.3 IQ2_XS / 59.1 IQ3_XXS on fresh prompts), and an RX 6700 XT on a Ryzen 7 5700G (our CPU; 32.1 tok/s decode, ~300 tok/s prefill, IQ2_XS). The only companion-own result is the RTX 4080 study (41.6 tok/s sustained decode, IQ2_XS, 200 W cap).

### Cited Findings
**rduffyuk RTX 4080 16 GiB (200 W cap), i9-13900KF, ~64 GB RAM, Linux 7.0.0-34, NVIDIA 615.71.09, patched engine 0.1.54, Qwen3.8-Flash-Next GSQ-RCO IQ2_XS, int8 KV, 262,144 ctx** — [README](https://github.com/rduffyuk/strata-context-benchmark/blob/main/README.md), [article](https://github.com/rduffyuk/strata-context-benchmark/blob/main/article/article.md), [results/strong-json-context.csv](https://github.com/rduffyuk/strata-context-benchmark/blob/main/results/strong-json-context.csv), [results/v7.csv](https://github.com/rduffyuk/strata-context-benchmark/blob/main/results/v7.csv)
- Sustained decode: 50 input + 1,024 output tokens, 50 samples, median 41.614 tok/s.
- Short-answer decode at 2K cold: median 43.4 tok/s, min 21.2, max 49.4 (n=50); 2K cached: 44.5 (38.0–50.8); 8K cold: 42.0 (22.6–50.4).
- Cold prefill at 261,112 tokens: 131.039 s median (~1,993 tok/s); cached (all but 7 tokens reused): 96 ms.
- Strong-JSON follow-up: 2K cold prefill 3.77 s, decode 39.8; 8K cold 5.56 s, decode 44.3.
- Memory: 8.00 GiB pinned expert copies + 3.09 GiB pinned KV in RAM; cgroup peak 34.0 GiB; GPU ~14.48 of 15.99 GiB occupied at startup.

**Upstream AMD reports indexed by strata-benchmarks** ([bench.jsonl](https://github.com/street-tripy/strata-benchmarks/blob/main/bench.jsonl); details from local mirror of upstream `bench/results/`)
| Report | Hardware | Model/quant | Result |
|---|---|---|---|
| 2026-10-03 rx7900xtx-rocm102-nightly | RX 7900 XTX 24 GB, Fedora 44 | Flash-Next IQ3_S | ~61-token prompt, 200 gen: ROCm 7.1.1 warm median 80.4 (78.8–81.6, n=6), cold request 1: 59.3; ROCm 10.2 nightly warm 88.4 (83.8–88.9, n=5) = +10.0%; 128K one-shot decode 62.0 vs 61.0 |
| 2026-10-04 rx7900xtx-hipblaslt-100500 | same | IQ3_S | 131,071-token prefill 1,687.0 (1,679.6–1,694.0) with gfx1100-100500 hipBLASLt table vs 926.1 (917.4–926.2) without; decode 62.26 vs 60.69 (55.24–61.02); 1K prompt: 645.8 prefill / 69.38 decode |
| 2026-10-04 rx-7900-gre | RX 7900 GRE 16 GB + Ryzen 7 5700X3D | Coder IQ1_M / IQ2_XS / IQ3_XXS | fresh-prompt decode median 65.3 (IQ2_XS), 59.1 (IQ3_XXS); Coder IQ1_M 4,210-tok prompt 1,224.1 [1,210.4–1,273.1] prefill, 48.5 [46.3–52.4] decode; hipBLASLt table `gfx1100-hipblaslt-100202.txt` active |
| 2026-10-06 2x-rx-7900-gre | 2x RX 7900 GRE | Coder IQ1_M | decode 68.5 (range 67.6–75.7 in notes) |
| 2026-10-09 4x-rx-7900-xt | 4x RX 7900 XT 20 GiB | IQ3_S, 262K ctx, `layer_split 12,24,36`, `STRATA_HIPBLASLT_TUNING=tools/hip/gfx1100-hipblaslt-100401.txt` | prompt 2,196.5 tok/s (median indexed); notes 3,430 / 963 tok/s at different sizes |
| 2026-10-05 rx-6700-xt | RX 6700 XT 12 GB + **Ryzen 7 5700G**, 64 GB | IQ2_XS, 131,072 ctx, `HSA_OVERRIDE_GFX_VERSION=10.3.0` | decode 32.1 / 30.7 / 30.2 at 4K / 32K / 128K; prefill 301.5 / 309.4 / 298.2 (3 runs each, median) |

### Inferences
- Rough expectations for a single 7900 XT (20 GB, ~800 GB/s; between GRE and XTX): warm short-prompt decode in the 60–85 tok/s band depending on quant, ROCm version and expert-cache hit rate; long-prompt prefill ~900 vs ~1,700 tok/s depending on whether a matching hipBLASLt table is loaded. Pancake numbers outside that band are worth explaining.
- The 4080 study's decode min of ~21 vs median ~43 (2× spread within a cell of n=50) shows that large intra-cell spread also occurs on NVIDIA under Strata; it was not investigated there.

### Gaps
- No RX 7900 XT single-card result exists in any of these repos.
- The strata-benchmarks scrape has many null fields (model/quant missing for the 0.1.31 XTX rows: decode 4.9–64.7 tok/s), so those rows are not usable without the upstream report.

## Q4. Tuned settings / env vars / launch options / OS tuning that gave speed gains

### Takeaway
The only quantified gains in companion repos are inherited from upstream: hipBLASLt tuning table (+82% long-prompt prefill on 7900 XTX), ROCm 10.2 nightly (+10% decode on XTX), `--kv-resident 32768` (50.9→62.6 tok/s at 262K, ~+6% at 128K), Linux read-ahead (cold start ~920 s → 70 s), I/O priority (Windows: 15× slower model reads / −14% generation when inherited low priority), NUMA interleave (+4–10% decode on dual-socket). Containers all set `--ulimit memlock=-1` (otherwise the expert arena silently cannot pin and "every copy to the GPU crawls"). Nobody sets CPU governor, amdgpu power level/profile, or pinning.

### Cited Findings
- `--ulimit memlock=-1:-1 --shm-size 2g` in every container recipe; AMD needs `--device /dev/kfd --device /dev/dri` — [strata-container README](https://github.com/Raboo/strata-container/blob/main/README.md), [strata-supervisor README](https://github.com/sognix/strata-supervisor/blob/main/README.md)
- "Without it the engine cannot pin its 50 GB expert arena, the failure is *silent*, and every copy to the GPU crawls. This is the single most likely cause of 'it works but it is slow'." (paraphrased from the `RLIMIT_MEMLOCK` troubleshooting entry) — [strata-docker README](https://github.com/HinataSoft/strata-docker/blob/main/README.md)
- Multi-die CPUs: set UMA/Distributed interleave rather than NPS2; populate every memory channel ("bandwidth matters more here than clock speed") — [strata-docker README](https://github.com/HinataSoft/strata-docker/blob/main/README.md). Quantified upstream (indexed in strata-benchmarks, 2026-10-03 z840-rtx3090ti): `numactl --interleave=all` 98.8 tok/s (95.9–100.6) vs node0-bound 94.9 vs default 90.1, with hugepages split 14000/14000 — [bench.jsonl rows 518–520](https://github.com/street-tripy/strata-benchmarks/blob/main/bench.jsonl)
- `STRATA_PROMPT_CACHE=24` (engine default 6, ~118 MB RAM each) and `STRATA_PROMPT_CACHE_EVERY=8192` (engine default 16384) for agent workloads — [strata-docker README](https://github.com/HinataSoft/strata-docker/blob/main/README.md)
- `DRAFT_VOCAB=en` frees ~215 MiB VRAM (348→133 MiB) for MTP models; `PREFILL=auto:32768`, `CONVERSATION_CACHE_MIB` wrapper overrides — [strata-container README](https://github.com/Raboo/strata-container/blob/main/README.md). Upstream 0.2.7 notes `--mtp-draft-vocab draft_vocab_en.bin` +2.7 to +3.1% decode for English — local `docs/release-notes-0.2.7.md`
- strata-manager health check: `--kv-resident 32768` at ≥64K context ("Upstream measured 50.9 -> 62.6 tok/s at 262K with Q2_0 and about +6% at 128K"; ~13.7 KB RAM per streamed token, keep 4 GiB margin); `--vram-reserve-mib` + `--expert-cache auto` when free VRAM < 512 MiB (1 GiB on Windows); leave `--pcie-frac` unset since engine 0.1.43 (RDNA4) / 0.1.42 (CUDA) refines it automatically; never use `pcie_frac 0` on AMD ("it wedges AMD gfx1201 at boot", upstream #2498) — [speed-tuning.ts](https://github.com/hongfaqiu/strata-manager/blob/main/packages/core/src/speed-tuning.ts), [tuning_calibrate.rs](https://github.com/hongfaqiu/strata-manager/blob/main/crates/runtime-core/src/tuning_calibrate.rs)
- `STRATA_ROUTE_PRIOR=1.5`: "faster on ≤12 GB NVIDIA" switch (upstream #4735), off by default, cost: long chats may repeat text; Qwen + CUDA only — [strata-manager CHANGELOG](https://github.com/hongfaqiu/strata-manager/blob/main/CHANGELOG.md). Also exposed with `STRATA_KVA` in [strata-launcher engine-options.mjs](https://github.com/dongzn2017/strata-launcher/blob/main/scripts/strata-panel/engine-options.mjs)
- Concurrency env `STRATA_BATCH_ATTN_ROWS=1`, `STRATA_BATCH_GDN_GROUPS=1`, `STRATA_PIPED_CONCURRENT=1`: since engine 0.1.55 the engine enables the first two itself only on Linux+CUDA with ≥2 concurrent requests, "not ... AMD/HIP" (upstream #4841) — [strata-manager CHANGELOG](https://github.com/hongfaqiu/strata-manager/blob/main/CHANGELOG.md), [concurrency.ts](https://github.com/hongfaqiu/strata-manager/blob/main/packages/core/src/concurrency.ts)
- Windows I/O priority: engines started from a low-priority parent inherited low I/O priority; upstream #4733 measured model reads ~15× slower and generation 14% slower; Manager now resets to normal — [strata-manager CHANGELOG](https://github.com/hongfaqiu/strata-manager/blob/main/CHANGELOG.md)
- `STRATA_RESIDENT_HEADROOM_GIB` (launcher default 8; rduffyuk used 10); Strata page-locks the RAM copy only with 4 GiB to spare per upstream `docs/AMD_HIP_PERFORMANCE.md` — [strata-launcher core.mjs](https://github.com/dongzn2017/strata-launcher/blob/main/scripts/strata-panel/core.mjs), [strata-supervisor app.js](https://github.com/sognix/strata-supervisor/blob/main/app/frontend/app.js)
- rduffyuk's full tuned profile (RTX 4080): `--expert-cache auto --resident-budget-gib 8 --prefill auto --spec 2 --kv int8 --pool-workers 6 --vram-reserve-mib 1536 --kv-resident 32768 --prompt-cache 1`, env `STRATA_PREFILL_LAYER_MAJOR=1 STRATA_PREFILL_LM_TOKENS=24576 STRATA_PREFILL_LM_MARGIN_MIB=512 STRATA_RESIDENT_HEADROOM_GIB=10 STRATA_POSTTOUCH=0 STRATA_ADVISORY_PLAN=0`; but "stock_baseline_comparison: NOT_PERFORMED; per-option difference from every upstream default is not established" — [environment.json](https://github.com/rduffyuk/strata-context-benchmark/blob/main/environment.json)
- strata-supervisor records the desktop-app IQ3_S config seen on the 7900 XTX box: `--max-context 524288 --rope-scaling yarn --rope-scale 2 --kv int8 --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --pool-workers 15 --vram-reserve-mib 4915 --resident-budget-gib 40` (no speed numbers) — [IMPLEMENTATION.md](https://github.com/sognix/strata-supervisor/blob/main/AI/IMPLEMENTATION.md)
- AMD-specific gains from the upstream reports indexed by strata-benchmarks: `STRATA_HIPBLASLT_TUNING=<gfx1100 table matching the hipBLASLt version>` 926→1,687 tok/s prefill at 131K on XTX (decode unchanged); table solution IDs are scoped to one hipBLASLt build, so a ROCm update invalidates it; ROCm 10.2 nightly +10% warm decode on XTX; the 4x 7900 XT run used `gfx1100-hipblaslt-100401.txt` (its "preparation commands were not recorded") — local mirror reports `2026-10-04-community-rx7900xtx-hipblaslt-100500`, `2026-10-03-community-rx7900xtx-rocm102-nightly`, `2026-10-09-community-4x-rx-7900-xt`. The XTX hipBLASLt arms ran with `STRATA_PREFILL_RING=96 STRATA_GR_V3=1`.
- Linux start time (upstream docs, local mirror): the engine requests kernel read-ahead of model files ("a cold start went from ~920 s to 70 s on one PC"; `STRATA_READ_AHEAD=0` disables) — `/home/dan/clone-of-d/src/Strata/docs/TROUBLESHOOTING.md` line ~120.
- Hugepages: 7900 GRE report: "The engine asked for 2 MiB hugepages for the host expert arena and could not get them (`vm.nr_hugepages=0`), so the arena used 4 KB pages" (IQ2_XS arena 16,907 × 2 MiB = 33.0 GiB) — local mirror `2026-10-04-community-rx-7900-gre/README.md`. Upstream 0.2.7: reserved 2 MB hugepages for DeepSeek: 8K prompt 651→694 tok/s — local `docs/release-notes-0.2.7.md`.

### Inferences
- Our box has `vm.nr_hugepages=0`, THP `madvise`, defrag `madvise`, and amdgpu `power_dpm_force_performance_level=auto` (checked 2026-11-10). So pancake's arena gets 4 KB pages unless THP madvise succeeds; reserving ~34–40 GiB of 2 MiB hugepages for an IQ2_XS/IQ3 arena is untested on our side and may not fit with 57 GB RAM.
- The hipBLASLt table is the single largest documented prefill lever on gfx1100; verify pancake logs "hipBLASLt tuning enabled (N rows, gfx1100, version X)" with X matching the ROCm 10.0.0 hipBLASLt.

### Gaps
- No companion repo documents CPU governor, `amd_pstate` mode, core pinning/`taskset`, GPU `power_dpm_force_performance_level`, `pp_power_profile_mode` COMPUTE, or ROCm env (`HSA_*`, `GPU_MAX_HW_QUEUES`, etc.) beyond `HSA_OVERRIDE_GFX_VERSION` for RDNA2.

## Q5. Run-to-run variance (bimodal slow runs, thermal, compaction, clock states): documented explanations and mitigations

### Takeaway
No companion repo documents a bimodal ~25% slowdown or controls GPU clock state/power profile. The relevant documented mechanisms are: (1) amdgpu evicting Strata's VRAM expert cache to system RAM when the card is nearly full and the desktop needs VRAM (seen on a 7900 XTX; fixed by a larger `--vram-reserve-mib`); (2) cold expert tier / warm-up curve (first requests 20–30% slower on XTX); (3) THP compaction stalls on fragmented memory (upstream #771, `STRATA_NO_ARENA_THP=1`); (4) clocks still ramping on the first timed case (upstream AMD_HIP.md); (5) hugepage reservation failure leading to 4 KB pages. Mechanisms 1, 3 and 5 can produce on/off (bimodal) behaviour that affects both decode and prefill.

### Cited Findings
- **amdgpu VRAM eviction (7900 XTX, 0.2.9-rocm-gfx110x, 2026-11-07):** with Coder IQ1_M loaded, free host RAM dropped ~15 GiB for 1–2 s every 10–35 s; with Strata's default reserve (700 MiB) only ~850 MiB VRAM stayed free, and when the desktop needed VRAM (a playing video) "`amdgpu` evicts the expert cache (15.44 GiB) to system RAM (kernel memory, not the container's)". With `vram_reserve_mib = 1000` the spikes stopped (user's test) — [strata-supervisor IMPLEMENTATION.md §13](https://github.com/sognix/strata-supervisor/blob/main/AI/IMPLEMENTATION.md)
- **Cold vs warm:** on the XTX, request 1 (fresh engine + cold expert tier) decoded 59.3 tok/s vs 80.4 warm median (−26%); the report deliberately measures the warm-up curve and only counts requests 4–9 — local mirror `bench/results/2026-10-03-community-rx7900xtx-rocm102-nightly/README.md` (indexed in [strata-benchmarks](https://github.com/street-tripy/strata-benchmarks/blob/main/bench.jsonl)). Same report: nightly decode range 83.8–88.9 vs ROCm7 78.8–81.6.
- **THP compaction:** "With transparent huge pages on `always`, the arena does not ask for `MADV_HUGEPAGE` on top (a fragmented machine spent minutes compacting memory, #771); `STRATA_NO_ARENA_THP=1` skips that request on any setting" — local `/home/dan/clone-of-d/src/Strata/docs/TROUBLESHOOTING.md` (upstream docs).
- **Clock ramp:** "the first case of a fresh `tune_hipblaslt` process times its baseline while the clocks are still ramping. The first fresh prompt after a model loads is also not a measurement." — local `/home/dan/clone-of-d/src/Strata/docs/AMD_HIP.md` (~line 659).
- **Thermal/power observed but not controlled:** RX 6700 XT + Ryzen 7 5700G report logged 99% load, median 173 W (max 188 W), median 82 °C (max 85–86 °C) via an amdgpu-sysfs monitor; GRE report: "GPU power limits and clocks were neither set nor measured", "a sustained thermal run [was] not evaluated" — local mirror `2026-10-05-community-rx-6700-xt/README.md`, `2026-10-04-community-rx-7900-gre/README.md`.
- **Telemetry available:** strata-manager reads Linux amdgpu sysfs (`pp_dpm_sclk` / `pp_dpm_mclk` active level marked `*`, hwmon temps) for AMD cards, so active clock level can be logged per sample — [amd_gpu/sysfs.rs](https://github.com/hongfaqiu/strata-manager/blob/main/crates/runtime-core/src/amd_gpu/sysfs.rs)
- **Intra-cell spread on NVIDIA:** rduffyuk 2K-cold decode min 21.2 vs median 43.4 tok/s (n=50) with a 200 W cap; not analysed — [results/v7.csv](https://github.com/rduffyuk/strata-context-benchmark/blob/main/results/v7.csv)
- **Windows I/O priority** (not Linux): inherited low priority gave −14% generation — [strata-manager CHANGELOG](https://github.com/hongfaqiu/strata-manager/blob/main/CHANGELOG.md)

### Inferences
- For our bimodal ~25% runs (both decode and prompt read, no foreign CPU load), the best-supported candidate from these sources is VRAM pressure/eviction: if the 7900 XT also drives a display (or anything else allocates VRAM) while pancake leaves <1 GiB free, amdgpu can evict part of the expert cache to GTT/system RAM, which would slow decode and prefill together. Check: `cat /sys/class/drm/card*/device/mem_info_vram_used` and `mem_info_gtt_used` per run, plus `dmesg` for evictions; mitigation: raise `--vram-reserve-mib` (≥1000 MiB worked on XTX) and keep the desktop on the 5700G iGPU.
- Second candidate: whether the arena got THP/hugepages that run. With THP=madvise and no reserved pool, a fragmented page cache after earlier runs could make the arena 4 KB on some starts and 2 MB on others (per-start bimodality); log `AnonHugePages` / `smaps` for the engine PID per run, or run `echo 1 > /proc/sys/vm/compact_memory` before start, or reserve `vm.nr_hugepages`.
- Third: GPU DPM state. Nobody in these repos pins it, but logging `pp_dpm_sclk`/`pp_dpm_mclk` (as strata-manager does) per sample would show whether slow runs sit at a lower mclk/sclk level; `power_dpm_force_performance_level=high` or `pp_power_profile_mode` COMPUTE is the standard A/B (not documented by any Strata repo — our own experiment).
- Warm-up curve: upstream counts warm decode only after 4–6 identical requests; if abtest.py's per-arm warm-up is shorter, cold-expert-tier runs could land in the measured set and look like a slow mode (though that would hit first runs systematically, not randomly).

### Gaps
- No source found that reports bimodal (two-mode) Strata timings on any GPU, or that tests `power_dpm_force_performance_level`, `pp_power_profile_mode`, CPU governor, or memory compaction as variance controls. These would need our own experiments.
