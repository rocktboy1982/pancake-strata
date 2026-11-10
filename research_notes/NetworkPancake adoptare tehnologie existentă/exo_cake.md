# exo and Cake as candidate bases for NetworkPancake (as of 2026-11-10)

Method: GitHub REST API, raw.githubusercontent.com files (main branch), one web search per open question. Nothing was built or run. The unauthenticated GitHub API rate limit (60/h) ran out near the end, so some counts are partial and marked. Prior notes (existing_systems.md, community_discovery.md, config_ui_orchestration.md) are not repeated except where this note corrects or extends them.

## 1. License (exact terms) and what each forbids

### Takeaway
exo is Apache-2.0 and can be forked, embedded and sold with no restriction beyond notices. Cake uses a custom "FAIR License v1.0.0" that is NOT open source: it is free only for non-commercial use, and ANY use by or for a business entity needs a separately signed commercial agreement with the author. For a project that may ever be commercial or used inside a company, Cake is effectively unusable as a base without a deal with Simone Margaritelli.

### Cited Findings
- exo: GitHub reports license Apache-2.0 (SPDX), repo `archived: false` — [GitHub API](https://api.github.com/repos/exo-explore/exo); README badge "License: Apache-2.0" — [README](https://github.com/exo-explore/exo)
- Cake: GitHub reports SPDX `NOASSERTION` / "Other"; the LICENSE file is titled "FAIR License (Free Access for Individuals and Researchers) Version 1.0.0", copyright 2024-present Simone Margaritelli, "All Rights Reserved" — [LICENSE](https://raw.githubusercontent.com/evilsocket/cake/main/LICENSE); [API](https://api.github.com/repos/evilsocket/cake)
- Naming inconsistency: the README expands FAIR as "Free for Attribution and Individual Rights", the LICENSE as "Free Access for Individuals and Researchers" — [README](https://raw.githubusercontent.com/evilsocket/cake/main/README.md) vs [LICENSE](https://raw.githubusercontent.com/evilsocket/cake/main/LICENSE)
- §2 Non-commercial grant: use, copy, modify, distribute software and derivatives for "personal, educational, academic research, or non-profit purposes, where no Commercial Use or Business Use is involved", provided you keep notices, mark modifications, and distribute any Derivative Work under the same license (copyleft-like) — [LICENSE](https://raw.githubusercontent.com/evilsocket/cake/main/LICENSE)
- "Commercial Use" is defined broadly: paid product or service, SaaS, anything monetised through subscriptions, ads, in-app purchases, licensing or consulting fees, or internal use inside a for-profit org to support revenue-generating operations (§1) — same LICENSE
- "Business Use" = any use by or on behalf of a legal entity (corporation, LLC, sole proprietorship, etc.) in its business operations "regardless of whether the specific use directly generates revenue" (§1). §5: Business Use requires a separate written commercial agreement signed by the author (evilsocket@gmail.com); use without it "constitutes a violation ... and an infringement of the Author's copyright" — same LICENSE
- §3-4: any Commercial Use also needs visible attribution: the text "Powered by Cake" or "Built with Cake", a link to https://github.com/evilsocket/cake, and "Created by Simone Margaritelli", in the UI, landing/docs page, or README, not hidden or in fine print, for the whole duration of use — same LICENSE
- §6 Contributions: contributors grant the author a perpetual, irrevocable right to relicense their code under any terms; this is a one-way (dual-licensing) rights grab — same LICENSE
- §7 Termination: automatic on any violation; a 30-day cure applies only to a first violation after written notice — same LICENSE
- Because Derivative Works must stay under FAIR (§2c), embedding Cake code in NetworkPancake would force the whole combined work to FAIR. Linking out via a process boundary may avoid this, but the license text does not address it; [UNVERIFIED legal reading].

### Inferences
- If NetworkPancake is meant to be freely usable (including by companies, or monetisable), Cake as a code base is a blocker. Cake as inspiration (design ideas are not copyrightable) is fine; copying code is not.
- A hobby, non-profit, personal-use fork would be allowed, but the author could still relicense contributions and the same-license rule would bind Strata-adjacent code that is merged in.

### Gaps
- No lawyer's reading; no statement from the author on whether talking to a Cake worker over a network protocol counts as a Derivative Work.

## 2. Activity, releases, archive status (conflict resolved)

### Takeaway
exo is archived = FALSE and very active (daily commits, pushed 2026-11-06). The "archived" claim does not hold for exo-explore/exo; I found no source supporting it. Cake is NOT archived but has been quiet for about 6.5 months (last push 2026-04-24, zero commits in the last 90 days) and has no releases or tags.

### Cited Findings
- exo: `archived: false`, `disabled: false`, pushed_at 2026-11-06T22:00:57Z, created 2024-06-24, 48,276 stars, 3,605 forks, 458 "open_issues_count" (issues+PRs combined; earlier notes had 206 issues / 252 PRs) — [API](https://api.github.com/repos/exo-explore/exo)
- exo commit volume: the API call for commits since 2026-08-10 returned the 100-item page cap, and those 100 span only 2026-10-13 to 2026-11-06 (about 25 days, all committed by AlexCheema, who merges PRs). Therefore at least 100 commits in 25 days, likely several hundred in 90 days. The second and third pages failed from rate limiting; exact 90-day total NOT obtained — [API commits](https://api.github.com/repos/exo-explore/exo/commits?since=2026-08-10T00:00:00Z&per_page=100)
- Recent commit subjects (Nov 2026): "Serve requests without the master", "Launched models survive a change of master, a heal and a full restart (#2602)", "Keep the models users launch running: deployments and one reconciler (#2601)" — [API commits](https://api.github.com/repos/exo-explore/exo/commits?per_page=6) (PR numbers near #2600 show high PR throughput)
- exo contributors: 98 on page 1 of the contributors list (list may continue); top: AlexCheema 1,505, Evanev7 147, rltakashige 141, JakeHillion 74, cadenmackenzie 64, blindcrone 63. Heavily dominated by one maintainer (bus factor) — [API contributors](https://api.github.com/repos/exo-explore/exo/contributors?per_page=100)
- exo releases (GitHub): v1.0.64 2026-01-23, v1.0.65 01-24, v1.0.66 01-26, v1.0.67 01-28, v1.0.68 02-25, v1.0.69 03-27, v1.0.70 04-17, v1.0.71 04-23. No GitHub release after April 2026, though main keeps moving (source version in pyproject.toml is 0.3.70). The macOS app has its own download channel (EXO-latest.dmg, Homebrew cask) — [API releases](https://api.github.com/repos/exo-explore/exo/releases?per_page=8); [pyproject](https://raw.githubusercontent.com/exo-explore/exo/main/pyproject.toml); [README](https://github.com/exo-explore/exo)
- exo "archived" conflict: the repo is not archived per the API. A search for "exo archived / v2 rewrite" found no archive notice; results that did appear (SourceForge mirror, TomeVault) only show activity. The likely source of the confusion is an unrelated repository, or the old exo v1 codebase that was rewritten in place (current code is a different architecture from 2024 exo with tinygrad); [UNVERIFIED which source made the claim]. One open PR title in the repo ("Tinygrad runner for LLM inference on Linux", #1660) shows tinygrad is no longer in main — [issues search](https://api.github.com/search/issues?q=repo:exo-explore/exo+linux+gpu+in:title)
- Cake: `archived: false`, pushed_at 2026-04-24T15:11:28Z, created 2024-07-07, 3,131 stars, 206 forks, 17 open issues+PRs — [API](https://api.github.com/repos/evilsocket/cake)
- Cake commits since 2026-08-10: empty list (0) — [API](https://api.github.com/repos/evilsocket/cake/commits?since=2026-08-10T00:00:00Z). Last commit 2026-04-24 is a merge of PR #84 (Copilot-authored fix for iOS TCP failures, issue #79) — [API commits](https://api.github.com/repos/evilsocket/cake/commits?per_page=5)
- Cake contributors: evilsocket 1,042 commits, "Copilot" 44, then four people with 1 commit each (8 listed in total). Effectively a single-author project — [API contributors](https://api.github.com/repos/evilsocket/cake/contributors?per_page=100)
- Cake releases and tags: none (latest-release endpoint 404; tags empty). The README has a release badge pointing to /releases/latest, which resolves to nothing — [API](https://api.github.com/repos/evilsocket/cake/releases/latest)
- Earlier note stated Cake "15 open issues" and 1,117 commits; this run saw 17 and about 1,086 contributions (1,042 + 44), consistent within a few days of drift — [community_discovery.md](C:\Users\iancu\pancake-main\research_notes\Plan NetworkPancake inferență distribuită\community_discovery.md)

### Inferences
- exo is a live product with a commercial company (exo labs) behind it; divergence risk is high (internal APIs churn weekly) but abandonment risk is low. Cake's near-zero activity for 6+ months plus no releases means a fork would be a fork of a stalled project.

### Gaps
- Exact exo 90-day commit count and PR-merge rate (rate limit).
- Origin of the "exo is archived" claim: not found.

## 3. Architecture and where an engine plugs in

### Takeaway
exo has a real engine abstraction (`Engine` / `Builder` ABCs) but it is hard-wired to MLX and mflux: the choice is an if-statement in `bootstrap.py` and the instance types (MlxRing, MlxJaccl) are MLX-specific. Cake has a Rust `Forwarder` trait (shard of layers) and a `Generator` trait (model), both tightly bound to candle `Tensor`s; backends (cuda/metal/vulkan/rocm/cpu) sit below candle via a custom `backends/` layer. Neither natively accommodates an opaque external engine like `strata-poly --serve`.

### Cited Findings: exo
- Language Python (requires `==3.13.*`) plus Rust crates for networking (`rust/networking`: discovery.rs, membership.rs, swarm.rs; `rust/exo_rs` PyO3 bindings) — [pyproject](https://raw.githubusercontent.com/exo-explore/exo/main/pyproject.toml); [file tree](https://api.github.com/repos/exo-explore/exo/git/trees/main?recursive=1)
- Roles: `src/exo/master/` (main.py, placement.py, placement_utils.py, deployments.py), `src/exo/worker/` (main.py, plan.py, serving.py, runner/supervisor.py, runner/runner.py), `src/exo/shared/topology.py`, `src/exo/api/` (OpenAI, Claude, Ollama, Responses adapters) — [file tree](https://api.github.com/repos/exo-explore/exo/git/trees/main?recursive=1)
- Engine abstraction, `src/exo/worker/engines/base.py` (about 55 lines): `class Engine(ABC)` with abstract `warmup()`, `submit(task: GenerationTask)`, `step() -> Iterable[(TaskId, Chunk | CancelledResponse | FinishedResponse)]`, `close()`, `serve_prefill(request, wfile)`; `class Builder(ABC)` with `connect(bound_instance)`, `load(bound_instance) -> Generator[ModelLoadingResponse]`, `build() -> Engine`, `close()` — [base.py](https://raw.githubusercontent.com/exo-explore/exo/main/src/exo/worker/engines/base.py)
- Engine selection is hard-coded in `src/exo/worker/runner/bootstrap.py` `entrypoint()`: `if bound_instance.is_image_model: MfluxBuilder(...) else: MlxBuilder(model_id=..., ...)`. No registry or plugin mechanism; adding a third engine means editing this branch — [bootstrap.py](https://raw.githubusercontent.com/exo-explore/exo/main/src/exo/worker/runner/bootstrap.py)
- Runner (`runner/runner.py`) asserts `isinstance(self.generator, Engine)` and drives it via submit/step; each runner is a separate spawned process supervised by `supervisor.py` (38 KB) — [runner.py](https://raw.githubusercontent.com/exo-explore/exo/main/src/exo/worker/runner/runner.py)
- Instance types: `InstanceMeta = {MlxRing, MlxJaccl}`; `Instance = MlxRingInstance | MlxJacclInstance`; a source comment says `# TODO: Single node instance`. Placement and instance model assume MLX collectives (ring over TCP, or JACCL over RDMA) — [instances.py](https://raw.githubusercontent.com/exo-explore/exo/main/src/exo/shared/types/worker/instances.py)
- Sharding is MLX-specific: `engines/mlx/sharding/pipeline.py`, `tensor.py`, plus per-family `models/<family>/sharding.py`. Model families present: deepseek_v3/v4/v41, glm, gpt_oss, kimi, llama, minimax, nemotron_h, qwen3 (incl. qwen3 MoE sharding, 13 KB sharding.py), gemma, step, laguna, inkling, hy_v3, muse_spark, flint — [file tree](https://api.github.com/repos/exo-explore/exo/git/trees/main?recursive=1)
- A "disaggregated" prefill/decode path exists (`worker/disaggregated/`, `engines/mlx/disaggregated/`), showing the maintainers can ship cross-node protocols, but it is also MLX-tied — [file tree](https://api.github.com/repos/exo-explore/exo/git/trees/main?recursive=1)
- Discovery is a Rust libp2p-style swarm/membership layer (the README mentions a Zenoh namespace in earlier notes) and requires identical version/wire-version/build across nodes: "Devices running different versions don't join each other" — [README](https://github.com/exo-explore/exo); [config_ui_orchestration.md](C:\Users\iancu\pancake-main\research_notes\Plan NetworkPancake inferență distribuită\config_ui_orchestration.md)
- No tinygrad and no llama.cpp in current main: pyproject deps are mlx, mlx-lm, mlx-vlm, torch; a tinygrad runner is only an open PR (#1660, 4 comments). The prior note's "MLX/tinygrad/llama.cpp abstraction" premise is therefore outdated — [pyproject](https://raw.githubusercontent.com/exo-explore/exo/main/pyproject.toml); [issue search](https://api.github.com/search/issues?q=repo:exo-explore/exo+linux+gpu+in:title)

### Cited Findings: Cake
- Language Rust, built on candle (`candle_core::Tensor` in all trait signatures). Workspace crates: `cake-core` (library), `cake-cli`, `cake-mobile` (UniFFI lib for iOS/Android), `cake-mobile-app` (Kotlin Multiplatform app) — [Cargo.toml / tree](https://api.github.com/repos/evilsocket/cake/git/trees/main?recursive=1); [cake-mobile/src/lib.rs](https://raw.githubusercontent.com/evilsocket/cake/main/cake-mobile/src/lib.rs)
- Shard abstraction, `cake-core/src/cake/mod.rs` ~line 509: `#[async_trait] pub trait Forwarder: Debug + Send + Sync + Display` with `load(name, ctx)`, `forward(&self, x: &Tensor, index_pos, block_idx, ctx)`, `forward_mut`, `forward_batch`, `goodbye`, `layer_name`, `ident` — [cake/mod.rs](https://raw.githubusercontent.com/evilsocket/cake/main/cake-core/src/cake/mod.rs)
- Model abstraction, `cake-core/src/models/mod.rs`: `Generator` (`type Shardable: Forwarder; load(context)`), `TextGenerator` (`add_message`, `reset`, `goodbye`, `next_token`, `generated_tokens`), `ImageGenerator`, audio — [models/mod.rs](https://raw.githubusercontent.com/evilsocket/cake/main/cake-core/src/models/mod.rs)
- Distribution layer: `cake-core/src/cake/sharding/` (master.rs, worker.rs 30 KB, client.rs, discovery.rs 53 KB mDNS, auth.rs HMAC, topology.rs, proto/message.rs 37 KB custom binary protocol, api/ HTTP + index.html web UI 36 KB) — [tree](https://api.github.com/repos/evilsocket/cake/git/trees/main?recursive=1)
- Compute backends: `cake-core/src/backends/{cpu,cuda,metal,rocm,vulkan}/` (cuda `ops.cu` 26 KB; metal `ops.msl`; vulkan `ops.wgsl` via wgpu; rocm ffi) with `backends/mod.rs` (28 KB) as the dispatch trait. Granularity is tensor ops, not whole engines — [tree](https://api.github.com/repos/evilsocket/cake/git/trees/main?recursive=1)
- Layer assignment is proportional to each worker's VRAM (via nvidia-smi, or system memory for Metal/CPU); master streams only needed safetensors shards to workers (zstd + CRC32), workers cache in `~/.cache/cake/` — [clustering.md](https://raw.githubusercontent.com/evilsocket/cake/main/docs/clustering.md); [README](https://raw.githubusercontent.com/evilsocket/cake/main/README.md)

### Inferences
- exo's `Engine` contract (submit task, step yields chunks) maps cleanly onto a "send GEN, read T lines until DONE" subprocess protocol, so a StrataEngine could be a thin Python class around `subprocess.Popen`. The hard parts are outside the ABC: `bootstrap.py` branch, `Instance`/placement types and model cards assume MLX sharding, so Strata would at best be a single-node (full model per node) engine and never take part in exo's pipeline/tensor sharding.
- In Cake a Strata integration would be a new `Generator` + `Forwarder` implementing `forward` over candle tensors, which does not fit a process that owns the whole model and speaks tokens; the natural fit is a different layer (replace a "model" with an opaque token-level worker), meaning edits in master/worker protocol code, in a repo whose license blocks reuse anyway.

### Gaps
- exo's `placement.py` was not read line by line in this pass for how a node with a non-MLX engine would be filtered (prior note has placement_utils details).
- Cake's `backends/mod.rs` trait names not enumerated.

## 4. Supported models, MoE

### Takeaway
Both support Qwen3 MoE. exo's MoE support is large-model, high-memory Apple-first (Qwen3-235B, Kimi K2, DeepSeek) with tensor/pipeline sharding. Cake supports Qwen3 MoE 30B-A3B and Qwen3.5 MoE (35B-A3B GPTQ-Int4) but expert handling is layer-sharded, not expert-parallel; it also has an experimental disk-backed expert provider. Neither uses GGUF-native quantised kernels comparable to Strata's.

### Cited Findings
- Cake text families (15): LLaMA 3.x, SmolLM2, Qwen2/2.5, Qwen3 dense, Qwen3 MoE (`Qwen/Qwen3-30B-A3B`, 128 experts / top-8), Qwen3.5, Qwen3.5 MoE (`Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`, 256 experts / top-8), Phi-4(-mini), Mistral, Gemma 3, Falcon3, OLMo 2, EXAONE 4.0, DeepSeek-R1 distills. Also SD/FLUX image and VibeVoice/LuxTTS TTS — [models.md](https://raw.githubusercontent.com/evilsocket/cake/main/docs/models.md)
- Cake docs state Qwen3 MoE 235B-A22B is a target "requiring multiple nodes to hold all expert weights" — [models.md](https://raw.githubusercontent.com/evilsocket/cake/main/docs/models.md)
- Cake MoE infra files: `models/common/expert_provider.rs`, `disk_expert_provider.rs` (35 KB), `qwen3_moe/moe.rs`, `qwen3_5_moe/moe.rs`, plus `autoresearch/models/flash-moe/` — [tree](https://api.github.com/repos/evilsocket/cake/git/trees/main?recursive=1)
- Cake quant handling: GGUF Q4_K_M loads but is dequantised to F16 for compute (no native quantised matmul): Qwen2.5-0.5B 184 tok/s vs ollama 421 tok/s on an RTX 3080 laptop, VRAM 3.1 GB vs 0.6 GB. Authors' own benchmark (single GPU, 0.5B) — [benchmarks/README.md](https://raw.githubusercontent.com/evilsocket/cake/main/docs/benchmarks/README.md)
- exo: Qwen3 family has `models/qwen3/` (family.py, sharding.py 13 KB, qwen4_exp.py), plus deepseek_v3/v4, kimi, glm_moe_dsa, gpt_oss, minimax, nemotron_h; models are MLX format (mlx-lm) from Hugging Face — [tree](https://api.github.com/repos/exo-explore/exo/git/trees/main?recursive=1)
- exo measured numbers (third party): 4x Mac Studio M3 Ultra with RDMA, Qwen3-235B 32 tok/s, Kimi K2 about 30 tok/s — [Geerling](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5) (via earlier note)

### Gaps
- GGUF support in exo: not found (MLX formats only per deps).
- No independent Cake MoE benchmark found.

## 5. Hardware: CUDA, AMD, Vulkan, Windows, old PCs

### Takeaway
Cake is the broader hardware story (CUDA, Metal, Vulkan, ROCm backend code, CPU, Windows CUDA, pre-Volta patch script), but only on paper and with dated, hand-run docs. exo is Apple-first: Linux needs MLX's CUDA wheels (CUDA 12/13, effectively modern NVIDIA only), no Vulkan, no AMD, no Windows. Neither is a practical base for "old PCs" except Cake with its Vulkan/Pascal paths, which is blocked by license.

### Cited Findings: exo
- README still says "Currently, exo runs on CPU on Linux. GPU support for Linux platforms is under development" (stated twice) — [README](https://github.com/exo-explore/exo)
- But pyproject.toml on main already defines extras `mlx-cpu`, `mlx-cuda12`, `mlx-cuda13` (mlx-cuda-12/13 0.32.4 on linux plus matching torch indexes), and the README install snippet says "NVIDIA: --extra mlx-cuda13 or --extra mlx-cuda12". So NVIDIA CUDA via MLX is real but not advertised in prose; README is inconsistent — [pyproject](https://raw.githubusercontent.com/exo-explore/exo/main/pyproject.toml); [README](https://github.com/exo-explore/exo)
- Tracking issue #913 "General Linux CUDA Support" still open; maintainer Evanev7 on 2025-12-20: "I consider this issue solved with pipeline CUDA but RDMA should follow"; DGX Spark not production-ready (2026-01-27); reports from Jetson, GTX 1060 (0.0 TFLOPS) and CUDA OOM were closed as duplicates; last comment 2026-02-02 asked whether CUDA work happens in a private repo, no reply — [issue 913](https://github.com/exo-explore/exo/issues/913)
- Open PRs/issues: #2103 native CUDA, #2216 report CUDA VRAM as node memory, #2129 3-node heterogeneous Metal+CUDA, #2180 heterogeneous scheduling may assign inference to a CPU worker despite a CUDA GPU, #2053 CUDA Docker image, #2449 Vulkan legacy iGPU engine (open, 0 comments), #934 Vulkan support (open) — [issues search](https://api.github.com/search/issues?q=repo:exo-explore/exo+(cuda+OR+vulkan+OR+windows+OR+tinygrad)+in:title)
- Windows: not supported. #606 [TRACKING] Windows Support open (22 comments), #1983 request, #2298 PR "Windows mlx-cpu runtime support" open; maintainer said Windows pipeline support is feasible but RDMA unlikely (2025-12-19) — [issues search](https://api.github.com/search/issues?q=repo:exo-explore/exo+(cuda+OR+vulkan+OR+windows+OR+tinygrad)+in:title); [issue 913](https://github.com/exo-explore/exo/issues/913)
- Source install requires Python 3.13, Rust toolchain, Node for the dashboard build, uv — [README](https://github.com/exo-explore/exo); [pyproject](https://raw.githubusercontent.com/exo-explore/exo/main/pyproject.toml)

### Cited Findings: Cake
- Platform table: Linux arm/arm64/x86_64 (CPU, CUDA, BLAS); Windows x86_64 CUDA "Supported", BLAS "Untested"; macOS x86_64 and aarch64 (+Metal); Linux x86_64 Vulkan "Supported"; Android arm/arm64/x86_64 CPU "Supported", CUDA untested; iOS aarch64 CPU, and Metal on A13+/M-series. CUDA >= 12.2 required — [install.md](https://raw.githubusercontent.com/evilsocket/cake/main/docs/install.md)
- Vulkan is "portable but less optimized than Metal/CUDA", via wgpu / WGSL shaders, any Vulkan 1.3+ device (Steam Deck, AMD, Intel). The CLAUDE.md dev guide lists the author's own cluster: RTX 3080 laptop, 2x TITAN X Pascal (CUDA 12.4), M3 Pro Mac — [CLAUDE.md](https://raw.githubusercontent.com/evilsocket/cake/main/CLAUDE.md)
- Pre-Volta GPUs (Pascal/Maxwell) need patching candle-kernels via `cake-core/src/backends/cuda/compat/patch.sh` — [install.md](https://raw.githubusercontent.com/evilsocket/cake/main/docs/install.md)
- Metal speed: about 42 tok/s on M3 Pro with the 0.8B model — [install.md](https://raw.githubusercontent.com/evilsocket/cake/main/docs/install.md)

### Inferences
- For "old PCs", Cake's Vulkan + Pascal-patch combination is the only stack here that targets them, but speed is likely poor (F16 dequant, "less optimized") and no numbers exist; exo effectively excludes old non-NVIDIA PCs and all Windows machines today.

### Gaps
- No user-reported Vulkan speeds for Cake. ROCm backend maturity (code exists: `backends/rocm/`) is unknown; install.md does not list ROCm as a feature.

## 6. Cake iOS/Android worker: real or aspirational

### Takeaway
It is real code but unreleased and unproven: there is a Kotlin Multiplatform worker app in the repo that you must build yourself (APK or Xcode), CI build jobs exist, and there are bug-fix PRs from actual iOS use; there is no App Store or Play Store listing, no release binaries, and no phone benchmark anywhere I could find. Treat as demo-level.

### Cited Findings
- `cake-mobile-app/` is a Kotlin Multiplatform / Compose Multiplatform "worker app" for iOS and Android; Android build via `make mobile_android` (needs cargo-ndk, NDK) producing a debug APK; iOS via `make mobile_ios` then Xcode build/deploy. Metal on A13+; older devices fall back to CPU — [install.md](https://raw.githubusercontent.com/evilsocket/cake/main/docs/install.md)
- `cake-mobile/src/lib.rs` (22 KB) uses UniFFI to expose `stop_worker`, status JSON (stage, progress, model, layers, backend) and starts a `cake_core` Worker — the app only acts as a worker serving assigned layers; it is not a chat client — [lib.rs](https://raw.githubusercontent.com/evilsocket/cake/main/cake-mobile/src/lib.rs)
- iOS Swift code is a minimal host: `ContentView.swift` embeds the Compose UI (652 bytes) — [ContentView.swift](https://raw.githubusercontent.com/evilsocket/cake/main/cake-mobile-app/iosApp/iosApp/ContentView.swift)
- Recent mobile work (April 2026): PR #84 "fix: iOS worker TCP connection failures (issue #79)" (retry logic, spawn_blocking), PR #81 "configurable iOS jetsam memory limits", PR #60 CI jobs for Android aarch64 and iOS arm64, PR #14 "Standardized Android client", and a fix for madvise on Android — [issues search](https://api.github.com/search/issues?q=repo:evilsocket/cake+iphone+OR+android+OR+tok/s&per_page=10); [commits](https://api.github.com/repos/evilsocket/cake/commits?per_page=5). Indicates iOS memory-kill (jetsam) and connection problems were real issues in practice.
- No App Store / Play Store link, no GitHub release assets, no tags — [API releases](https://api.github.com/repos/evilsocket/cake/releases/latest)
- No phone tok/s numbers: the only benchmark file (2026-03-18) is on an RTX 3080 laptop for 0.5B text, FLUX, VibeVoice; the README example topology listing iPhone 15 Pro Max / iPad / M1 Max gives no speeds. The author said "no benchmarks at the moment" in a July 2024 Reddit thread (search-result summary, not fetched) — [benchmarks/README.md](https://raw.githubusercontent.com/evilsocket/cake/main/docs/benchmarks/README.md); [clustering.md](https://raw.githubusercontent.com/evilsocket/cake/main/docs/clustering.md); [Reddit via redlib](https://redlib.ssps.io/r/deeplearning/comments/1e4kxkr/cake_a_rust_distributed_llm_inference_for_mobile)
- Model sizes/quants usable on phones: undocumented. Given F16 dequant compute and docs examples (0.6B-1.7B models), expect small models only [UNVERIFIED inference].
- Security (all workers): mutual HMAC-SHA256 challenge-response with `--cluster-key`; traffic and weights are NOT encrypted ("use a VPN or SSH tunnel"); only an 8-hex-char hash of the key goes into mDNS adverts — [clustering.md](https://raw.githubusercontent.com/evilsocket/cake/main/docs/clustering.md)

### Inferences
- A phone worker in the Cake sense adds capacity for tiny layer slices over Wi-Fi; for MoE with Strata-sized models it will not help. The iOS worker is technically the most complete phone path found, but a license plus stalled maintenance means NetworkPancake would need its own phone client regardless.

### Gaps
- No user-reported speeds, App Store, TestFlight or demo video found. Issue bodies (#79, #14) were not read (rate limit).

## 7. UI and security comparison

### Takeaway
exo has the stronger, maintained UI (dashboard on :52415, macOS menu-bar app) but no authentication story in the README; Cake has a single-file web UI plus a TUI and a cluster key with HMAC auth but no encryption.

### Cited Findings
- exo: dashboard at http://localhost:52415 for cluster management and chat; macOS menu-bar app (.dmg, Homebrew cask); the README's only security note concerns `trust_remote_code`, default off for custom models. No API auth or TLS is described in the README (grep for auth/encrypt/tls found nothing) — [README](https://github.com/exo-explore/exo)
- exo version-lock: nodes must match version, wire version and build; `EXO_*` env vars and `--no-worker` coordinator-only mode exist — [README](https://github.com/exo-explore/exo)
- Cake: web UI is `cake-core/src/cake/sharding/api/index.html` (36 KB) served with an OpenAI-compatible REST API, plus `cake chat` TUI (`cake-cli/src/chat.rs`) — [tree](https://api.github.com/repos/evilsocket/cake/git/trees/main?recursive=1); [README](https://raw.githubusercontent.com/evilsocket/cake/main/README.md)

### Gaps
- exo network-layer security (libp2p/noise encryption in `rust/networking`) not verified; earlier notes cite a secondary source about security gaps.

## 8. Effort estimate and recommendation for adding Strata

### Takeaway
Rough effort (my estimates, not sourced): exo single-node Strata engine about 300-600 lines Python and 3-5 days to get working; honest end-to-end (placement, model cards, dashboard, tests, tracking upstream churn) 2-3 weeks. Cake: not recommended because of the license; engineering would be larger (about 1,000+ lines Rust, plus protocol changes) and tied to a stalled repo.

### Inferences (all estimates, [UNVERIFIED])
- exo path: (1) write `StrataBuilder(Builder)` and `StrataEngine(Engine)` that spawns `strata-poly --serve`, translates `GenerationTask` to `GEN`, parses `T` lines into `Chunk`s and `DONE` into `FinishedResponse`, supports cancel by restarting or a CANCEL line; (2) add a branch in `bootstrap.py` (about 5 lines); (3) add an `InstanceMeta` value / single-node instance type (the `# TODO: Single node instance` comment indicates exo itself lacks one, so expect touching `placement.py`, `instances.py`, the dashboard, and the node capability reporting) ; (4) model cards for Strata-supported GGUFs (exo models are MLX repos, so GGUF/HF download paths would be new). Strata would run whole-model per node: exo gives discovery, dashboard and OpenAI/Claude/Ollama APIs for free, but not distribution of Strata.
- Python 3.13 pin, Rust build for networking wheels, and Mac-first CI make exo a heavy base on Windows (unsupported) where NetworkPancake's old PCs live. A fork must also keep rebasing against fast main (100+ commits per 25 days).
- Simplest honest alternative: keep Strata's own `GEN/T/DONE` worker and put only a thin adapter in front, using exo or Cake as a design reference (exo placement math, Cake mDNS+HMAC cluster-key auth and weight streaming) rather than as a code base. That keeps license and divergence risk at zero.
- Ranking for "base to build on": exo (usable license, active, but MLX/Apple-centric, Windows unsupported); Cake (broadest hardware and phone story on paper, but FAIR license plus stalled repo make it a reference only).

### Gaps
- No prototype was built; estimates are judgment calls from file sizes and interfaces.
