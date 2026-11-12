# Mesh-LLM as a base for NetworkPancake (source-level assessment, 2026-11-10)

Method: shallow clone (depth 300) of https://github.com/Mesh-LLM/mesh-llm at main commit f26843e (2026-11-10), reading README, ROADMAP, docs and Rust sources; GitHub API for repo stats; WebFetch for issues #1236 and #987. The GitHub API rate-limited me partway, so some counts are from search queries made earlier. File paths below are relative to the repo root. Line numbers are from that clone. "(est.)" marks my own estimates. Prior notes in `Plan NetworkPancake inferență distribuită/` (existing_systems.md, mainstream_actors.md, community_discovery.md) only had README-level facts. Everything below goes deeper.

## 1. License, activity, bus factor

### Takeaway
Permissive and commercial-friendly, but the license metadata is internally inconsistent: LICENSE is Apache-2.0 while the Cargo workspace says "MIT OR Apache-2.0". Activity is extremely high: about 300 commits in roughly 25 days, 33 open PRs, release roughly every 1-3 weeks. Core maintainers are few (about 3 account for most history), and the project is young (created 2026-02-11).

### Cited Findings
- LICENSE is the stock Apache License 2.0 text, with an appendix notice "Copyright 2024 Block, Inc." No NOTICE file exists at the root (`ls NOTICE*` returned nothing). — [LICENSE](https://github.com/Mesh-LLM/mesh-llm/blob/main/LICENSE)
- Workspace manifest declares `license = "MIT OR Apache-2.0"`, `edition = "2024"`, `version = "0.78.0"`. This conflicts with the Apache-only LICENSE file. GitHub's API reports Apache-2.0. Either way both options are permissive. No commercial-use limits found in LICENSE or README. — [Cargo.toml](https://github.com/Mesh-LLM/mesh-llm/blob/main/Cargo.toml); API `repos/Mesh-LLM/mesh-llm`
- Repo stats: created 2026-02-11, about 4,013 stars, 505 forks, language Rust, last push 2026-11-10. — GitHub API `repos/Mesh-LLM/mesh-llm`
- Issues: 132 open, 438 closed. PRs: 33 open, 2,305 closed. — GitHub search API (`type:issue/pr state:open/closed`)
- Commits: my depth-300 clone spans 2026-10-16 to 2026-11-10, so there are at least 300 commits in about 25 days (about 12/day). The exact 90-day count was not obtained. Authors in those 300 commits: Virgile Pourchet 73, James Dumay 61, Martin Kleinberger 43, Nick DiZazzo 34, Daniel Winter-Wijntjes 31, Michael Neale 18, Ethan Zhang 14. — `git log` on the clone
- All-time GitHub contributors (by contributions): i386 1157, michaelneale 834, ndizazzo 442, Virgile-pct 153, IvGolovach 98, Copilot 79, danielwinterw 67, kleinbem 47, StevenMih 38, benthecarman 26. — API `contributors`
- Releases: v0.76.0 (2026-09-10), v0.76.1 (09-12), v0.76.2 (09-14), v0.77.0 (09-28), v0.78.0 (10-04), v0.78.1 (10-05), a 0.78.0-preview build (10-30). 0.79.0-rc1 was being prepared (PR #2857, issue #2775 "Release 0.79.0 candidate validation"). — API `releases`; [PR 2857](https://github.com/Mesh-LLM/mesh-llm/pull/2857)
- Size: about 940k lines of Rust in `mesh/` plus `skippy/`, and about 70 workspace crates. — `find mesh skippy -name '*.rs' | xargs cat | wc -l` on the clone; [Cargo.toml](https://github.com/Mesh-LLM/mesh-llm/blob/main/Cargo.toml)
- Backers and structure: the LICENSE copyright and a Block engineering blog post indicate Block's backing. Several contributors (i386, michaelneale, ndizazzo) account for most history. A Co-owner of the org was not verified.

### Inferences
- Bus factor is roughly 3-6 people, but they are productive, and the commit rate shows a funded team rather than a hobby project.
- The 940k-line codebase with a pace of about 12 commits/day is a fast-moving target. A fork would drift quickly. Prefer integration through the plugin interface over patching core.
- Before adopting, ask maintainers or counsel which license applies (MIT OR Apache-2.0 vs Apache-2.0 only).

### Gaps
- Exact 90-day commit count and unique-contributor count (API rate limit; clone depth too shallow).
- Whether a NOTICE or per-crate license files exist in sub-directories (only root checked).

## 2. Architecture and how it wraps llama.cpp

### Takeaway
Rust host (mesh) around a Rust+C++ engine called Skippy. Skippy is a patched llama.cpp, carried as a queue of 16 patches over a pinned upstream commit and exposed through an experimental C ABI. It does not use llama.cpp RPC. The OpenAI frontend sits behind a Rust trait (`InferenceBackend`) that is the natural engine boundary.

### Cited Findings
- Two product trees share one Cargo workspace: `skippy/` (model management, native inference, split execution, OpenAI serving) and `mesh/` (peer discovery, transport, routing, plugins, console). Dependencies point from mesh to skippy only. — [README.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/README.md); [skippy/docs/ARCHITECTURE.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/skippy/docs/ARCHITECTURE.md)
- Call chain from the architecture doc: client -> shared inference frontend -> serving + scheduler + cache -> Rust runtime -> Skippy C ABI -> patched llama.cpp. — skippy/docs/ARCHITECTURE.md
- Pinned upstream llama.cpp commit: `372dc5ca157276beb6c67b2e92477e8d662e964a` (file `skippy/llama_cpp/upstream.txt`). Patch queue `skippy/llama_cpp/patches/0001..0016` (graph-derived native ABI, staged graph realization, MTP state, MoE host placement in 0012, etc.). C ABI version in 0012's diff is 0.1.69, described as "experimental version-0". — files named
- Key crates: `skippy/crates/skippy-ffi` (C ABI bindings), `skippy-runtime`, `skippy-serving`, `skippy-inference-api`, `skippy-protocol` (stage wire protocol, about 2.7k lines), `skippy-coordinator`/`skippy-topology`/`skippy-scheduler` (placement), `skippy-package-*` (layer packages). Mesh side: `mesh/crates/mesh-llm-host-runtime` (main host), `mesh-llm-plugin`, `mesh-llm-transport`, `mesh-llm-ui`, `mesh-llm-sdk`, `mesh-llm-ffi`. — Cargo.toml workspace members
- Stage protocol: "Internal, generation-gated stage traffic. Mixed generations fail closed; it is not a backwards-compatible public network API." Current generation 16, peers must advertise `skippy-stage/2` and `stage-generation-16`. — skippy/docs/ARCHITECTURE.md; skippy/docs/DATA_FLOW.md
- It is described as intentionally independent of llama.cpp RPC: stock RPC transfers about 16.88 GB on connect; the fork loads local GGUF on peers, 0 bytes transferred. — [mesh/docs/BENCHMARKS.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/mesh/docs/BENCHMARKS.md)

### Inferences
- A non-llama.cpp engine cannot join the stage protocol. That protocol is tied to the patched llama.cpp graph (activation frames, graph filtering).

### Gaps
- I did not build the code or verify the patch queue applies cleanly. How much of upstream llama.cpp diverges (patch sizes) was not measured.

## 3. Engine abstraction: can an external process be a node type?

### Takeaway
Yes, and there is direct precedent. Two shipped external engines (DwarfStar/ds4 and Flash-MoE) are integrated as plugins that register an OpenAI-compatible HTTP endpoint and optionally let mesh-llm manage the process. There is also an in-process Rust trait (`InferenceBackend`). Plugin-hosted engines are single-node only: they are routed to and gossiped, but never split.

### Cited Findings
- `InferenceBackend` trait (async_trait) at `skippy/crates/skippy-inference-api/src/backend.rs:180`. Methods: `models`, `chat_completion`, `chat_completion_stream`, `completion`, `completion_stream`, `embeddings`, `rerank`, `count_chat_tokens`, `system_one`; non-chat methods default to "unsupported". Production implementor: `SkippyModelHandle` at `mesh/crates/mesh-llm-host-runtime/src/inference/skippy/mod.rs:477`. Other implementors are test or wrapper types (e.g. `MembershipBackend` in `runtime/shared_serving/backend.rs:97`, `CompactingInferenceBackend` in `skippy-inference-api/src/guardrails/compact.rs:65`).
- Plugin inference registration: plugin DSL `inference::openai_http(endpoint_id, address)` and `inference::provider(...)` (managed by plugin) in `mesh/crates/mesh-llm-plugin/src/dsl.rs:473-490` (builder at `dsl.rs:293`); manifest helper `openai_http_inference_endpoint` at `mesh/crates/mesh-llm-plugin/src/manifest.rs:1351`.
- Host side: `PluginManager::inference_models()`, `inference_endpoint_for_model()`, `inference_endpoint_for_request()` in `mesh/crates/mesh-llm-host-runtime/src/plugin/inference_routing.rs:44-100`. Route struct `InferenceEndpointRoute {plugin_name, endpoint_id, protocol, address, models}` at `plugin/types.rs:276`. Endpoint health is probed via `GET /v1/models` (tests in `plugin/health.rs:699-760`).
- `protocol` values handled: `openai_compatible` (or none) = chat; also `document_ocr_v1`, `transcription_v1`, `image_text_similarity_v1`, `depth_estimation_v1`. — inference_routing.rs:14-40
- Mesh-wide serving of plugin-hosted models: `mesh/crates/mesh-llm-host-runtime/src/runtime/plugin_host_role.rs` polls every 5 s and claims `NodeRole::Host` when plugins advertise models. Wired at `runtime/run_auto.rs:1838`. The design note states a live test covered only a single node and did not exercise an inbound mesh tunnel from another peer or mesh-wide routing. — [plugin-hosted-mesh-serving-design.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/mesh/docs/plugins/plugin-hosted-mesh-serving-design.md)
- Plugin launch contract: host spawns the plugin with env vars `MESH_LLM_PLUGIN_ENDPOINT` (local IPC: unix socket or Windows pipe), `MESH_LLM_PLUGIN_TRANSPORT`, `MESH_LLM_PLUGIN_NAME`, `MESH_LLM_PLUGIN_URL`; control protocol is protobuf (`mesh/crates/mesh-llm-plugin/proto/plugin.proto`, 1035 lines). — [mesh/docs/plugins/README.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/mesh/docs/plugins/README.md)
- Precedent 1, Flash-MoE plugin: external repo Mesh-LLM/flash-moe; config `[[plugin]] name="flash-moe"` with `--backend-command` and args; plugin appends `--serve <port>` and registers `http://127.0.0.1:<port>/v1`; or attach to an existing `url`. "This is intentionally a single-node backend adapter." — [mesh/docs/plugins/flash-moe.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/mesh/docs/plugins/flash-moe.md). Note Strata's `--serve` flag name collides conceptually with this convention but speaks a line protocol, not HTTP.
- Precedent 2, DwarfStar: `mesh-llm plugins install Mesh-LLM/ds4-plugin`, runs antirez/ds4 as "an alternative engine to Mesh's built-in runtime", Apple Silicon, "DwarfStar runs on one node; Mesh does not split it across machines." Needs `[runtime] mode = "on_demand"`. — [mesh/docs/plugins/dwarfstar.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/mesh/docs/plugins/dwarfstar.md)
- A separate speculative-decoding native plugin ABI exists (`mesh/crates/mesh-native-serving-plugin-api/src/lib.rs`, `NATIVE_SERVING_PLUGIN_ABI_V2`, C-ABI dylib with tokenizer inventory and proposal polling). This is for proposers into the Skippy decode loop, not for whole engines. — that file

### Inferences
- Integration paths for Strata (est.):
  - Path A, plugin (recommended): write a small shim that turns Strata's stdin/stdout `GEN`/`T`/`DONE` protocol into `/v1/chat/completions` (SSE) plus `/v1/models`, then a plugin that spawns `strata-poly --serve` plus the shim and registers an `openai_http` endpoint. Shim about 500-1000 lines (tokenizer/chat-template handling, streaming, cancel, concurrency queue: Strata presumably serves one stream at a time, unverified); plugin glue about 200-400 lines of Rust against `mesh-llm-plugin` DSL. Effort 1-2 weeks including Windows testing, assuming Strata exposes its own tokenizer/chat template; add a week if not. No fork of Mesh needed.
  - Path B, in-tree `InferenceBackend` impl: about 1.5-3k lines, 3-4 weeks, requires touching host runtime and tracking an upstream moving about 12 commits/day. Not advised.
  - Path C, use Mesh only for discovery, transport and console and bypass Skippy entirely: this is what Path A gives you.
- Risk: the multi-peer routing of plugin-hosted models has been tested only for single-node ingress per the design note, so verify with 2 machines early.
- Strata's expert-offload cache, speculative decoding and IQ2/IQ3 kernels would stay internal to Strata, because Mesh sees only an OpenAI endpoint.

### Gaps
- I did not read the `Mesh-LLM/flash-moe` or `Mesh-LLM/ds4-plugin` repositories (the actual plugin source), so the real plugin LOC is unknown.
- Whether the plugin API is versioned/stable (the docs suggest plugin ABI is evolving; PR #2907 and 2913 touch plugins).

## 4. Splitting modes and what works with a non-llama.cpp engine

### Takeaway
Layer-stage pipeline splitting (the headline feature) requires llama.cpp internals. Only whole-model routing, replication and Mixture-of-Agents fan-out work with an external OpenAI-speaking engine.

### Cited Findings
- Single-node first: "If one node can host the full model, it serves the model locally without stage traffic." Requests are routed by `model` field to a peer serving it. — README.md
- Skippy stage splits: contiguous layer ranges planned by a coordinator, downstream stages started first, stage 0 published last. Decode sends activation frames of 8 KiB/token (f32, a 4-stage reference run), prefill frames up to about 2 MiB (prefill chunk 256). Prediction goes back directly from the final stage to the driver stage. — [skippy/docs/DATA_FLOW.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/skippy/docs/DATA_FLOW.md); [skippy/docs/SKIPPY_SPLITS.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/skippy/docs/SKIPPY_SPLITS.md)
- Split admission is gated by a certification roster of GGUF `general.architecture` values; override flag `--allow-uncertified-split`. — SKIPPY_SPLITS.md
- Staged GGUF fragment fetch: "layer package" = Hugging Face repo with `model-package.json` plus `shared/{metadata,embeddings,output}.gguf` and `layers/layer-NNNNN.gguf`, each with size and SHA-256; peers download only shared artifacts plus their stage's layers. Example `mesh-llm serve --model hf://meshllm/Qwen3-235B-A22B-UD-Q4_K_XL-layers@<rev> --split`. — [skippy/docs/LAYER_PACKAGE_REPOS.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/skippy/docs/LAYER_PACKAGE_REPOS.md)
- Other multi-model features that are engine-agnostic at the HTTP level: smart router (task classifier, multi-model per node), demand-based rebalancing (partial), and `model: "mesh"` Mixture-of-Agents fan-out (experimental). — [ROADMAP.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/ROADMAP.md); README.md
- Speculative decoding: PR #567 experimental; MTP/DFlash/ngram proposers live inside Skippy. — ROADMAP.md; skippy docs

### Inferences
- Works with an external engine: model-level routing, health/gossip advertisement, router, MoA, console listing, API proxy, owner/trust controls.
- Needs llama.cpp internals: stage splits, layer packages (GGUF fragments), activation-wire dtype, stage-level KV cache, MTP in-engine speculation, `n_cpu_moe` host placement.
- A Strata-native pipeline split would have to be designed separately; Mesh's stage protocol is gated to generation 16 and "not a backwards-compatible public network API", so do not implement it.

### Gaps
- Whether the router (task classifier) can use capability metadata from a plugin-registered engine beyond model name was not checked.

## 5. MoE handling and the 35B-A3B case

### Takeaway
Mesh does NOT place experts across nodes in the shipped path. MoE models are split by layers like dense models; experts can be pushed to host RAM per node (`cpu_moe`, `n_cpu_moe`), and a Flash-MoE SSD streaming plugin exists as a single-node path. Maintainers' own research concludes expert parallelism over LAN is not worthwhile. A 35B-A3B model that fits one node runs there whole.

### Cited Findings
- Config knobs `cpu_moe = "auto"` and `n_cpu_moe = N` (routed experts of first N blocks in host memory) in [mesh/docs/USAGE.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/mesh/docs/USAGE.md) lines about 531-540 and 831-840. Implemented by patch 0012 "place routed experts of a block range in host memory": overrides `ffn_{up,down,gate,gate_up}_exps` of those blocks to the CPU buffer type, "the same mechanism as upstream --n-cpu-moe. ... Routing and precision are unchanged." — skippy/llama_cpp/patches/0012-feat-skippy-place-routed-experts-of-a-block-range-in.patch (author Daniel Winter, 2026-10-16)
- ROADMAP says "MoE expert sharding: Implemented. Auto-detects MoE, computes overlapping expert assignments, splits locally ... zero cross-node expert traffic" but also "Best thought of as experimental, most results show this doesn't perform as well as one would hope." — [ROADMAP.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/ROADMAP.md). In the current code I only found `SurveyLaunchKind::MoeShard`/`MoeFallback` labels used for forced vs capacity-driven splits (`runtime/startup_handles.rs:749-754`, `runtime/survey.rs:167`), not an expert-assignment planner. Issue #987 says its design has "No remote expert calls, all-to-all traffic, or restored partial expert-island serving", which implies expert-island serving was removed. So the ROADMAP entry looks stale.
- Issue #1236 (open, Task): "Do not build EP split serving for batch-1 decode." Per-layer expert work about 0.13-0.17 ms vs LAN round trips of 8-11 ms. Pipeline parallelism is the practical path, boundary about 14-16 KiB per token versus about 20-29 MiB per token for EP worst case. Hot-set residency needs about 99% of routing mass. Figures are mostly arithmetic from bandwidth specs, not measurements. `min_experts_per_node` is in protobuf but read by no planning code. — [issue 1236](https://github.com/Mesh-LLM/mesh-llm/issues/1236)
- Issue #987 (open, no comments): proposal for MoE-aware elastic batching inside a stage, expected gain in multi-sequence throughput not single-stream. — [issue 987](https://github.com/Mesh-LLM/mesh-llm/issues/987). Issue #1185 "Track expert usage stats for MoE models across the mesh" is open (title only read).
- Supported MoE families certified for stage splits include Qwen3-MoE, Qwen3-VL-MoE (30B-A3B MXFP4 GGUF), GLM4-MoE (GLM-4.7-Flash-REAP-23B-A3B), Kimi Linear 48B-A3B, OLMoE, Hunyuan-A13B, EXAONE-MoE 236B-A23B. Qwen3.6/3.8 load as the `qwen35`/`qwen35moe` arch pair. — [skippy/docs/FAMILY_STATUS.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/skippy/docs/FAMILY_STATUS.md)
- Flash-MoE plugin = SSD expert streaming, single-node only; "Not included: mesh-distributed SSD expert streaming". — flash-moe.md

### Inferences
- For a 35B-A3B model (about 17-22 GB at Q4-class quants), Mesh would run it whole on one capable node, optionally with `n_cpu_moe` for VRAM-limited GPUs, or split by contiguous layers over multiple weaker nodes with 8-16 KiB per token per hop. Nothing in Mesh does expert-level placement across nodes. Strata's expert-offload cache would be a differentiator rather than overlapping functionality, since Mesh only has the static `n_cpu_moe` placement.

### Gaps
- Whether any expert-assignment code survives in `mesh-llm-host-runtime/src/inference/` (I only grepped for strings).
- Content of issue #1185 beyond its title.

## 6. Transport, discovery, security

### Takeaway
QUIC via iroh with relays, Nostr for public discovery, signed invite tokens and owner-key admission for private meshes. Security is layered but the public mesh is open to any member, and the maintainers say so.

### Cited Findings
- Transport: QUIC (iroh 1.4.1 in Cargo.lock) end-to-end encrypted, including inference requests and split activations; relays forward encrypted packets. — README.md; Cargo.lock
- Discovery: Nostr relays by default (`nostr-sdk 0.45.1` in `mesh/crates/mesh-llm-host-runtime/Cargo.toml:89`); `--publish` advertises a mesh; private meshes use invite tokens (`--join`, `--join-file`, `MESH_LLM_JOIN`). A mode exists that contacts neither Nostr relays nor public iroh relays (LAN/mDNS). — [mesh/docs/MESHES.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/mesh/docs/MESHES.md)
- Admission: default token encodes endpoint identity and addresses. Signed bootstrap tokens bind mesh genesis policy and can expire. "QUIC encryption protects traffic in transit; it does not decide which peers" are admitted. Owner-key and `--trust-policy require-owned` provide allowlisting; release attestation can require peer binaries signed by a trusted release signer. — MESHES.md
- Separate owner-control plane `mesh-llm-control/2`; `/mesh/load`, `/mesh/drop` removed (410 Gone). — README.md; plugin-hosted design doc
- Open work: verifiable inference receipts auditing which model a peer served (PRs #2851, #2981; issue #1233). Payments/Lightning docs and `mesh-llm-payments`/`wallet` crates exist (PR #2551). — PR list
- Security reporting via GitHub private advisories (SECURITY.md); warns not to test against the public mesh.

### Inferences
- Because peers on the public mesh can send inference requests and activations to each other, any engine behind a plugin endpoint inherits an untrusted-requester threat model. Strata's line protocol parser is not exposed directly, only the shim, so harden the shim.

### Gaps
- No independent security audit found. Details of the Nostr event kinds and relay list not read.

## 7. Web console, OpenAI endpoint, platforms, mobile

### Takeaway
OpenAI-compatible API on :9337 and a React console/management API on :3131 (the headless flag hides the UI but keeps the management API). Broad desktop GPU matrix including Windows Vulkan; mobile is client-only via SDKs, with no evidence of phones serving.

### Cited Findings
- API: `http://localhost:9337/v1` (`/v1/models`, chat completions, completions, Responses; Anthropic Messages route also in `skippy-inference-api`); `--auto` joins best public mesh and starts console on 3131. — README.md; [mesh/docs/EXO_COMPARISON.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/mesh/docs/EXO_COMPARISON.md); skippy ARCHITECTURE.md
- Console: React/Vite app in `mesh/crates/mesh-llm-ui` embedded into the Rust binary; also MCP server at `:3131/mcp`; `GET /api/status`, `/api/runtime/stages`. A feature-by-feature inventory of the console was not extracted. — mesh-llm-ui/README.md; EXO_COMPARISON.md; SKIPPY_SPLITS.md
- Platforms (release bundles): macOS (Metal); Linux CPU, ARM64 CPU, ARM64 CUDA, CUDA, CUDA Blackwell, ROCm, Vulkan; Windows CPU, CUDA, ROCm, Vulkan. Each is a backend-neutral host plus a versioned native runtime (`mesh-llm runtime install cuda`). — README.md
- Windows notes: native CUDA works (verified on driver 610.74, RTX 4070 Ti); RTX 50-series (SM 120) not covered by stable native Windows because the only Windows CUDA runtime is CUDA 12 up to SM 90, so WSL2 is advised for them; `mesh-llm.exe` not code-signed; open issue #2768 "Qualify Windows ROCm BLAS support". — README.md
- Mobile: roadmap "Mobile chat app (exemplar)": client-only iOS-first app joining by QR code; Swift, Kotlin, Python, Node and Rust SDKs expose an embedded node with client/serve/combined modes (`mesh/docs/SDK.md`). Roadmap item is a plan, not a shipped app. — ROADMAP.md; SDK.md
- Android/Termux, iOS serving: no source found.

### Inferences
- Your GTX/RTX Windows machine is covered via CUDA 12 runtime (check SM level), and Vulkan builds cover the rest. A phone could join as a client but there is no sign it can serve a stage.

### Gaps
- Console features, screenshots and ROCm/Vulkan real-world reports were not examined; Metal/Vulkan open PRs relevant to phones: none found (search for "android/ios/phone/mobile" returned only unrelated PRs).

## 8. Performance claims and roadmap

### Takeaway
Published numbers show splitting costs speed: GLM-4.7-Flash-Q4_K_M went from 68 tok/s solo to 21 on two Macs over Wi-Fi. Mesh is for fitting big models, not speeding them up. The roadmap is breadth (router, MoA, multimodal, SDKs, payments, receipts), with MoE sharding flagged as unproven.

### Cited Findings
- Benchmarks doc: GLM-4.7-Flash-Q4_K_M (17 GB) on M4 Max + Mac mini M4 over Wi-Fi: solo 68 tok/s; 2-node split (85/15) 21; 3-node (62/31/8) 12-13. Sydney-to-Queensland about 20 ms RTT: 10-25 tok/s, "overhead was dominated by per-token RPC latency". These are described as "a quick reality check, not a universal promise." Method (prompt length, runs) is not stated. — [mesh/docs/BENCHMARKS.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/mesh/docs/BENCHMARKS.md)
- A structured competitive benchmark vs raw llama.cpp is defined but results were not read: CUDA and Metal, Llama 3.2 1B, DeepSeek Coder V2 Lite MoE, Falcon-H1, Granite 4.0 H 1B, llama-benchy pp=512 tg=8/64/256, concurrency 1 to 256, 256 prompts from a Thoughtworks agentic dataset, 576 cells. — [skippy/docs/COMPETITIVE_BENCHMARK.md](https://github.com/Mesh-LLM/mesh-llm/blob/main/skippy/docs/COMPETITIVE_BENCHMARK.md)
- Roadmap items: smart router (done), MoA (done, experimental), mobile chat app (planned), multimodal (partly done), speculative decoding (experimental, MTP ongoing), demand-based rebalancing (partial), MoE expert sharding (marked implemented but experimental), SDK/desktop packaging. — ROADMAP.md
- Open PRs of note (2026-11-10): #2907 port patch queue 0011-0016 into a typed facade (`skippy-api`); #2856 fair single-model routing with MoA as an opt-in plugin; #2851/#2981 inference receipts; #2746 shared prefix retention; #2920 portable Linux x86 CPU baseline; #2947 certify new model families; #2962 drop CacheGen KV disk codec. No open PR found for Vulkan-specific or phone-specific work; issue #1246 proposes a macOS Core AI backend (a second whole-model engine, relevant precedent). — GitHub search API

### Gaps
- Competitive benchmark results and the speculative-decoding speedups were not read.
- No independent third-party benchmarks of Mesh found.

## Bottom line for NetworkPancake (my judgment)
- Adopt Mesh-LLM as the network/discovery/API/console layer and add Strata as a plugin-hosted OpenAI engine (Path A, about 1-2 weeks est.). The mechanism is shipped, documented and used by two engines already.
- Do not expect it to split Strata across machines. Strata would be single-node per mesh peer; cross-node pooling would only work for llama.cpp/Skippy models.
- Treat these as risks: license metadata mismatch, project age (9 months), fast-moving unstable internals (ABI v0, stage generation 16), and plugin-hosted multi-peer routing tested on one node only.
