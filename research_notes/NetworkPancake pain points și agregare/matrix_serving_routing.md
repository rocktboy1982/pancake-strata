# Matrix: NetworkPancake plan vs existing serving/gateway/router projects (as of 2026-11-10)

Method: GitHub REST API (unauthenticated; rate limit hit after 8 repos), raw README/LICENSE files, a few docs pages, 3 web searches. "Commits 90d" = count from the `Link: rel=last` page of `commits?since=2026-08-12&per_page=1` (includes merge/bot commits; contributor counts were NOT fetched, so they are a gap everywhere). Percent coverage = my estimate [E] against the plan's 7 requirements: (a) one OpenAI endpoint, (b) strong node + small workers, (c) router, (d) per-model routing, (e) mesh of heterogeneous nodes, (f) session stickiness / shared memory, (g) job portal (async queue, quotas). Not repeated here: see mesh_llm.md, llamacpp_rpc_localai.md, mainstream_actors.md in the sibling folders.

## Q1. Per-candidate coverage, license, activity, extension interface, Strata plug-in, what remains

### Takeaway
No single project covers the plan. Mesh-LLM covers the most of (a)(d)(e) and now ships a mixture-of-agents (MoA) fan-out; LocalAI covers the most of (g) plus a documented gRPC backend interface; LiteLLM covers (c)(d)(f)(g) as a pure OpenAI-API front but has no mesh. Strata can plug into all of them through its OpenAI-compatible server; only LocalAI (gRPC) and Mesh (plugin) need adapter code, and Mesh's new URL-based `openai-endpoint` plugin may make Strata integration near-zero code [inference, untested].

### Cited Findings

**Matrix (summary).** Coverage = % of plan [E]; Strata plug = how Strata attaches.

| Project | License | Last push / latest release | Commits 90d | Coverage [E] | Extension interface | Strata plug | Remains for us |
|---|---|---|---|---|---|---|---|
| Mesh-LLM | Apache-2.0 (API) | 2026-11-10 / v0.78.1 (2026-10-05) | 1045 | ~55% (a,d,e ok; b via MoA; c partial; f no; g no) | plugins (`[[plugin]]`, `openai-endpoint` by URL, external binary plugins), `model:"mesh"` MoA | OpenAI server via `openai-endpoint` plugin (URL) | portal/queue, orchestrator-worker with verification, stickiness, Strata A/B gate |
| LocalAI | MIT | 2026-11-10 / v4.12.1 (2026-10-23) | 2229 | ~45% (a,d ok; c,e via distributed mode; g partial: API keys, quotas, jobs) | gRPC `backend.proto`, backend OCI gallery | write gRPC backend wrapper around `strata` stdin/stdout, or register OpenAI upstream | heterogeneous LAN mesh without Postgres/NATS, orchestrator-worker, verification |
| LiteLLM proxy | MIT outside `enterprise/` (GitHub shows NOASSERTION) | 2026-11-10 / v1.108.4 LTS (2026-11-09) | 13065 (suspiciously high, bot/merge heavy) | ~40% (a,c,d,f,g-lite ok; no node mesh) | `CustomLLM` class, `custom_provider_map`, `CustomRoutingStrategyBase`, OpenAI-compatible `api_base` | OpenAI upstream (zero code) | node discovery, hardware awareness, parallel fan-out/verify |
| llama-swap | MIT | 2026-11-10 / v272 | 148 | ~20% (a,d; swaps local models; no mesh, no router logic) | YAML `cmd` per model; any OpenAI/Anthropic-compatible upstream | add Strata as a `cmd` model (zero code) | everything beyond one-box model switching |
| vLLM production-stack | Apache-2.0 | 2026-11-07 / vllm-stack-0.1.14 | 37 | ~15% (c,f for vLLM only, K8s/Helm) | Helm values, router keyed by routing key/session ID | no (vLLM-only, Kubernetes) | n/a, wrong target (GPU-datacenter) |
| llm-d | Apache-2.0, CNCF sandbox | 2026-11-10 / v1.0.0 (2026-10-20) | 480 | ~15% (c,f best-in-class, but K8s + vLLM/SGLang) | Kubernetes Gateway API / inference scheduler, Helm well-lit paths | no (needs vLLM/SGLang model servers) | n/a for LAN; borrow ideas (prefix/session-aware routing) |
| Ray Serve LLM | Apache-2.0 (Ray) | 2026-11-10 / ray-2.60.0 (2026-11-04) | 1159 (whole Ray repo) | ~15% (c,f; Ray cluster) | pluggable request routers (`PrefixCacheAffinityRouter`, alpha), `build_openai_app` | no (vLLM engines, Ray cluster) | n/a, heavy dependency |
| vLLM Semantic Router | Apache-2.0 | 2026-11-10 / v0.4.0 (2026-09-27) | 1389 | ~25% of (c,d): signal/decision model-selection layer; no nodes/queue | Go service, YAML policies/signals, installs via script; "selects or combines models" | sits in front of OpenAI backends; Strata as backend | node mesh, queue, verification (could be the router brain) |
| RouteLLM | Apache-2.0 | last push 2024-08-10 (stale 27 months) | 0 | ~10% of (c) (strong/weak classifier) | Python lib + OpenAI-compatible server | OpenAI upstream | do not depend; copy the idea only |
| Open WebUI | custom BSD-style + branding clause (GitHub NOASSERTION) | 2026-11-10 / v0.13.3 (2026-11-08) | 2618 | ~25% of UI/(g) (multi-model chat, arena, tools; not a router) | Python Pipelines/Functions, OpenAI-compatible connections | OpenAI connection (zero code) | everything backend-side; license limits rebranding >50 users |
| Ollama | MIT | not fetched (rate limit) | n/a | ~15% (a,d on one box) | OpenAI-compatible API, `ollama launch`, cloud models | OpenAI upstream | all |
| LM Studio (llmster) | proprietary app (unverified) | n/a | n/a | ~15% (a,d one box; LM Link remote via Tailscale, preview) | `lms` CLI, OpenAI-compatible endpoint | OpenAI upstream | all |
| GPUStack | Apache-2.0 | not fetched | n/a | ~25% (b,c,e for GPU clusters; engines vLLM/SGLang/TRT-LLM/custom) | pluggable "custom inference engines" | custom engine entry (feasibility unverified) | LAN/consumer iGPU/Vulkan nodes unclear |
| Envoy AI Gateway | Apache-2.0 | not fetched | n/a | ~10% (c,d at K8s edge) | Envoy/K8s CRDs | OpenAI upstream | brief, not LAN-relevant |
| Kong AI gateway | Apache-2.0 (core) | not fetched | n/a | ~10% | Kong plugins (Lua/Go) | OpenAI upstream | brief |
| Portkey gateway | MIT (per LICENSE file) | not fetched | n/a | ~15% (c,d, fallbacks) | OpenAI-compatible config JSON | OpenAI upstream | brief |
| Helicone | Apache-2.0 | not fetched | n/a | ~5% (observability) | proxy / logging | OpenAI upstream | brief |
| OpenRouter-style clones | AGPL (LLM Gateway), Apache-2.0 (Bifrost) per vendor blogs | not checked | n/a | ~10-15% | OpenAI-compatible | OpenAI upstream | vendor-authored sources, unverified |

Sources per row (license/activity from `api.github.com/repos/<owner>/<repo>`, `.../commits?since=2026-08-12`, `.../releases/latest`; license text from `https://raw.githubusercontent.com/<owner>/<repo>/HEAD/LICENSE`).

**Mesh-LLM.**
- README: pools GPUs/memory across machines and exposes one OpenAI-compatible API at `localhost:9337/v1`; the mesh decides local, route to peer, or Skippy stage split — [README](https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/HEAD/README.md)
- Routing is by the `model` field to the peer that can serve it; QUIC end-to-end encrypted peer transport; private meshes are invite-token based — [README](https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/HEAD/README.md)
- New since earlier notes: Mixture-of-Agents gateway. `model:"mesh"` fans a request to every model in the mesh, a deterministic arbiter picks/merges, a reducer LLM only on conflict; marked experimental ("behavior, routing heuristics, error shapes may change"); needs >=2 distinct models — [README](https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/HEAD/README.md), [MOA_GATEWAY.md](https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/HEAD/mesh/docs/design/MOA_GATEWAY.md)
- MoA design states its own aim is "not to beat split-large on quality" but to keep the mesh useful on slow links; each worker runs fully local — [MOA_GATEWAY.md](https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/HEAD/mesh/docs/design/MOA_GATEWAY.md). This is fan-out + arbitration, not the plan's plan-then-delegate-then-verify orchestrator-worker; closest overlap is plan phase 3 (voting).
- Plugins: `[[plugin]]` blocks in config; `mesh-llm plugins install <GitHub org/repo>`; an `openai-endpoint` plugin routes to any OpenAI-compatible server by `url` (documented with Lemonade at `http://localhost:8000/api/v1`); other plugins: blackboard, flash-moe, ds4 — [USAGE.md](https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/HEAD/mesh/docs/USAGE.md)
- License: GitHub API reports Apache-2.0 — [API](https://api.github.com/repos/Mesh-LLM/mesh-llm); plan notes the Cargo inconsistency (not rechecked here).
- Activity: 4015 stars, 165 open issues, 1045 commits since 2026-08-12, release cadence not checked beyond latest v0.78.1 (2026-10-05, five weeks before today while push was today) — [API](https://api.github.com/repos/Mesh-LLM/mesh-llm). Repo now has separate `skippy/` (native inference + OpenAI serving) and `mesh/` trees; this differs from the "llama.cpp-based" framing in older notes (README).

**LocalAI.**
- MIT; 50286 stars; 2229 commits in 90 days; v4.12.1 on 2026-10-23 — [API](https://api.github.com/repos/mudler/LocalAI)
- "A small core": each backend wraps an engine in its own image, "build your own backend in any language against an open interface"; 60+ backends incl. llama.cpp, vLLM, SGLang, MLX; OpenAI, Anthropic and Ollama-compatible APIs — [README](https://raw.githubusercontent.com/mudler/LocalAI/HEAD/README.md)
- Backend interface: gRPC `service Backend` with `Health`, `LoadModel`, `Predict`, `PredictStream`, `Embedding`, `TokenizeString`, `Status`, `GetMetrics`, plus image/audio/etc. RPCs — [backend.proto](https://raw.githubusercontent.com/mudler/LocalAI/HEAD/backend/backend.proto). A Strata wrapper needs mainly `Health/LoadModel/PredictStream/Free` [inference].
- Distributed mode: stateless frontends with a SmartRouter (idle-first, free-VRAM, LRU eviction), workers self-register, state in PostgreSQL (SQLite not supported), coordination over NATS, S3-style file staging, NATS auth/mTLS — [distributed-mode.md](https://raw.githubusercontent.com/mudler/LocalAI/HEAD/docs/content/features/distributed-mode.md)
- 2026 news: v4.3 per-request replica routing, per-API-key usage attribution; June: prefix-cache-aware routing, production request router; v4.1: multi-user, OIDC, API keys, per-user quotas; v4.9 deny-by-default auth — [README news](https://raw.githubusercontent.com/mudler/LocalAI/HEAD/README.md). P2P/federated mode is documented separately and is distinct from distributed mode (distributed-mode.md).
- Not verified: whether LocalAI distributed workers run on Vulkan iGPUs/old cards (backends exist for Vulkan per README).

**LiteLLM proxy.**
- License file: MIT except `enterprise/` directory under a separate license — [LICENSE](https://raw.githubusercontent.com/BerriAI/litellm/HEAD/LICENSE). One vendor blog calls it BSL 1.1; contradicted by the LICENSE file (I trust the file) — [llmgateway.io](https://llmgateway.io/blog/open-source-openrouter-alternatives).
- 63043 stars, 5047 open issues, v1.108.4 LTS on 2026-11-09 with an `lts` image tag; releases are signed with cosign — [API/release](https://api.github.com/repos/BerriAI/litellm/releases/latest)
- Routing strategies: simple-shuffle (default), usage-based-routing-v2, latency-based-routing, least-busy, cost-based-routing; fallbacks with `order` priority, context-window fallbacks, weighted failover; cooldowns (default 5 s); session affinity via `x-litellm-session-id` header or `metadata.session_id`, `deployment_affinity_ttl_seconds` default 3600, re-pin on failover; custom strategy via `CustomRoutingStrategyBase` — [routing docs](https://docs.litellm.ai/docs/routing)
- Custom provider: subclass `CustomLLM` (`completion/acompletion/streaming/astreaming`, image methods), register via `custom_provider_map`; proxy routes `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, `/v1/messages` map to it — [custom_llm_server](https://docs.litellm.ai/docs/providers/custom_llm_server)
- Budgets, virtual keys, batch: listed in docs navigation but I did not open those pages [unverified detail here; mainstream_actors.md may cover].
- Caveat: pace of change is extreme (release notes show regressions like a missing `mcp` module "broken since v1.106.0"), so pin LTS — [release notes](https://api.github.com/repos/BerriAI/litellm/releases/latest).

**llama-swap.** MIT, 6229 stars, v272 published 2026-11-10 (very frequent integer releases), 148 commits in 90 days. Go single binary + YAML; swaps any OpenAI/Anthropic-compatible server including llama.cpp forks and vLLM; `ttl`, profiles, concurrent-model swap matrix DSL, API keys, `/metrics`, `/upstream/:model_id`, `hooks`, `setParams` filters. README mentions no peer/multi-host feature (grep for "peer" returned nothing) — [README](https://raw.githubusercontent.com/mostlygeek/llama-swap/HEAD/README.md), [API](https://api.github.com/repos/mostlygeek/llama-swap).

**vLLM production-stack.** Apache-2.0; Helm-based; request router "based on routing keys or session IDs to maximize KV cache reuse"; vLLM engines only; vllm-stack-0.1.14 on 2026-11-07; 37 commits in 90 days (slow), 280 open issues; 2026 roadmap issue #855 — [README](https://raw.githubusercontent.com/vllm-project/production-stack/HEAD/README.md), [API](https://api.github.com/repos/vllm-project/production-stack).

**llm-d.** Apache-2.0, CNCF sandbox (Red Hat, Google Cloud, IBM Research, CoreWeave, NVIDIA); v1.0.0 on 2026-10-20; 480 commits in 90 days; features: prefix-cache and load-aware routing, tiered KV cache, P/D disaggregation, session-aware routing for agents, flow control with priority/fairness, OpenAI-compatible Batch API; runs above vLLM/SGLang on Kubernetes — [README](https://raw.githubusercontent.com/llm-d/llm-d/HEAD/README.md), [API](https://api.github.com/repos/llm-d/llm-d).

**Ray Serve LLM.** Ray is Apache-2.0, ray-2.60.0 on 2026-11-04. Request routing is pluggable; `PrefixCacheAffinityRouter` (character-level prefix tree, falls back to power-of-two choices when load is imbalanced) is documented as alpha; session-affinity (consistent hashing needing client session ID) and KV-aware routers exist — [Ray docs](https://docs.ray.io/en/latest/serve/llm/prefix-aware-request-router.html), [API](https://api.github.com/repos/ray-project/ray).

**vLLM Semantic Router.** Apache-2.0, 6444 stars, 614 open issues, v0.4.0 on 2026-09-27, 1389 commits in 90 days (Go 1.25); "decision layer" that selects or combines models per call under policy (quality/latency/cost, location constraints), keeps the harness's agent loop; ships its own trained "Decision" models; v0.3 added "stateful production routing" — [README](https://raw.githubusercontent.com/vllm-project/semantic-router/HEAD/README.md). Extension interface details (Envoy ext_proc?) not verified in README.

**RouteLLM.** Apache-2.0; last push 2024-08-10; 45 open issues; no releases found — [API](https://api.github.com/repos/lm-sys/RouteLLM). Effectively unmaintained; use as an idea source only.

**Open WebUI.** 158296 stars, v0.13.3 (2026-11-08), 2618 commits in 90 days. LICENSE is BSD-3-like plus clause 4: do not remove/alter "Open WebUI" branding unless <=50 end users per 30 days, written permission, or enterprise license; CLA required — [LICENSE](https://raw.githubusercontent.com/open-webui/open-webui/HEAD/LICENSE). Release notes show terminals, coding agents, automations, arena candidates — [release](https://api.github.com/repos/open-webui/open-webui/releases/latest). Fine as a front-end client for a <=50-user LAN; not a router.

**Ollama.** MIT; README now advertises "locally or in the cloud", `ollama launch claude`/`openclaw` integrations, and Ollama Cloud clients — [LICENSE](https://raw.githubusercontent.com/ollama/ollama/HEAD/LICENSE), [README](https://raw.githubusercontent.com/ollama/ollama/HEAD/README.md). Activity/releases not fetched (rate limit).

**LM Studio.** `llmster` is the headless daemon since 0.4.0 (`lms daemon up`); LM Link gives remote use of models on another machine, end-to-end encrypted, over Tailscale, in Preview — [LM Studio docs](https://lmstudio.ai/docs/advanced/headless), [DGX Station post](https://lmstudio.ai/blog/dgx-station), [NVIDIA playbook](https://build.nvidia.com/playbooks/lm-studio/instructions.md). Free-tier limits conflict across sources ([respan](https://www.respan.ai/market-map/lm-studio) vs [besthub](https://www.besthub.dev/articles/unlock-the-full-power-of-lm-studio-for-local-llm-deployment-fa920e83f08f)); license of the app is not verified (I did not find an open-source repo claim).

**GPUStack.** Apache-2.0; cluster manager for on-prem, K8s and cloud GPUs; pluggable engines vLLM, SGLang, TensorRT-LLM "or your own"; accelerators NVIDIA, AMD, Ascend, Hygon (list truncated at fetch); built-in auth, load balancing, benchmarking; some features (cluster topology) enterprise-only — [README](https://raw.githubusercontent.com/gpustack/gpustack/HEAD/README.md), [LICENSE](https://raw.githubusercontent.com/gpustack/gpustack/HEAD/LICENSE).

**Brief tier.** LICENSE files: Envoy AI Gateway Apache-2.0, Kong Apache-2.0, Portkey gateway MIT (a search summary said Apache-2.0; the LICENSE file says MIT), Helicone Apache-2.0 — raw LICENSE URLs under `https://raw.githubusercontent.com/{envoyproxy/ai-gateway,Kong/kong,Portkey-AI/gateway,Helicone/helicone}/HEAD/LICENSE`. KServe not checked. OpenRouter-alternative lists are vendor-authored (Bifrost, LLM Gateway) — [llmgateway.io](https://llmgateway.io/blog/open-source-openrouter-alternatives).

### Inferences
- Plan coverage by requirement: (a) endpoint is solved by everyone; (d) per-model routing by Mesh, LocalAI, LiteLLM, llama-swap; (e) heterogeneous mesh only by Mesh-LLM (QUIC, discovery) and LocalAI distributed (needs Postgres+NATS, heavier); (f) stickiness only by LiteLLM, vLLM/llm-d/Ray (all assume vLLM-style KV), LocalAI (prefix-cache-aware routing); (g) portal/queue/quotas only partly by LocalAI (users, quotas, jobs) and llm-d (batch API, flow control) — nobody gives the plan's "job number, queue position, result retention" portal for LAN.
- What truly remains for us regardless of base: orchestrator-worker with verification (plan phase 2), parallel-thinking voting with quorum (phase 3; Mesh MoA is the nearest), portal MVP (phase 1.5), and the Strata A/B gate. These are the differentiated parts; routing and endpoint plumbing are commodity.
- The K8s/vLLM family (production-stack, llm-d, Ray Serve, KServe, Envoy) does not fit a LAN of old consumer GPUs and Strata/llama.cpp nodes; track them only for routing ideas (prefix/session affinity, flow control priority).
- Strata attaches to everything via its Python OpenAI-compatible server; the line protocol GEN/T/DONE is only needed for a LocalAI gRPC backend or a native Mesh plugin, and is not needed if `openai-endpoint` suffices [untested; plan already flags the `/v1/models` + SSE probe risk].

### Gaps
- Contributor counts and bus-factor for every repo (contributors API not called; rate limit).
- Activity for Ollama, GPUStack, Envoy AI Gateway, Kong, Portkey, Helicone, KServe (API 403 after 8 repos); retry with a token.
- LiteLLM budgets/virtual keys/batch pages; Open WebUI Pipelines/arena details; Semantic Router extension interface; Ray Serve LLM license of `serve.llm` module specifically; whether llama-swap has any multi-host support beyond proxying an upstream URL.
- Commit counts are indicative only (merge/bot commits not excluded; LiteLLM's 13065 looks inflated).
- Whether Mesh-LLM's `openai-endpoint` plugin supports SSE streaming, `/v1/models` listing, and per-plugin model advertising to the mesh (the Lemonade doc example suggests listing works, not tested). Mesh plugin protocol (`openai_http` vs `openai-endpoint` naming) not reconciled with earlier notes.

## Q2. Most stable external interface to avoid divergence

### Takeaway
The OpenAI HTTP API (chat/completions, models, SSE) is the most stable surface and is supported by every candidate; the next most stable is LocalAI's gRPC `backend.proto` (versioned in-repo, many backends depend on it) and LiteLLM's `CustomLLM`; Mesh-LLM's plugin protocol is the least stable (explicit pre-1.0 churn, MoA marked experimental).

### Cited Findings
- All candidates in the matrix advertise OpenAI-compatible endpoints (READMEs cited in Q1); llama-swap and LocalAI additionally expose Anthropic-compatible `/v1/messages`; LocalAI also an Ollama-compatible API — [llama-swap README](https://raw.githubusercontent.com/mostlygeek/llama-swap/HEAD/README.md), [LocalAI README](https://raw.githubusercontent.com/mudler/LocalAI/HEAD/README.md)
- LocalAI `backend.proto` has grown with new RPCs (Stores, Rerank, Face/Voice, Live audio) while the core `Predict/PredictStream/LoadModel/Health` remain; the proto is additive in the HEAD copy — [backend.proto](https://raw.githubusercontent.com/mudler/LocalAI/HEAD/backend/backend.proto). Backward-compatibility policy not found.
- Mesh-LLM warns its MoA behavior and error shapes may change between versions; plan notes ABI v0 — [README](https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/HEAD/README.md)
- Ray's prefix-aware router doc is marked alpha, API may change — [Ray docs](https://docs.ray.io/en/latest/serve/llm/prefix-aware-request-router.html)
- LiteLLM ships LTS tags to give a stable channel — [release](https://api.github.com/repos/BerriAI/litellm/releases/latest)

### Inferences
- Rank of stability for a thin layer (best to worst): OpenAI HTTP > LocalAI gRPC proto > LiteLLM `CustomLLM`/OpenAI upstream > llama-swap YAML (stable but only config) > Mesh plugin protocol > Ray/Semantic Router policy DSLs (fast-moving).
- Strategy: make Strata speak OpenAI HTTP only (already does), and treat every project as a replaceable consumer of that surface.

### Gaps
- No formal compatibility/deprecation policy found for backend.proto or Mesh plugins.
- Strata's `/v1/models` and SSE compliance against each consumer not tested here.

## Q3. Which could replace our planned coordinator/router

### Takeaway
Only partially, and not one project: Mesh-LLM already is a router+mesh (replaces discovery, per-model routing, transport); LiteLLM or vLLM Semantic Router can replace policy-level routing/fallbacks/session pinning if put in front; none replaces orchestrator-worker verification or the job portal. LocalAI distributed mode is the only full alternative stack with queue/quotas but demands Postgres + NATS.

### Cited Findings
- Mesh: routing by model field, local-first, Skippy splits, MoA fan-out/arbiter — [README](https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/HEAD/README.md)
- LiteLLM: strategies, fallbacks, cooldowns, session affinity, custom strategy plug — [routing docs](https://docs.litellm.ai/docs/routing)
- Semantic Router: select or combine models per call under policy, harness keeps the loop — [README](https://raw.githubusercontent.com/vllm-project/semantic-router/HEAD/README.md)
- LocalAI SmartRouter: idle-first scheduling, LRU eviction, replica routing — [distributed-mode.md](https://raw.githubusercontent.com/mudler/LocalAI/HEAD/docs/content/features/distributed-mode.md)

### Inferences
- Candidate architectures: (A) Mesh-LLM as the mesh + our thin orchestrator/portal service in front speaking OpenAI to `localhost:9337` (matches plan, least code); (B) LiteLLM in front of Strata/llama.cpp nodes on the LAN, no mesh (fastest to get phase 1 with stickiness and keys, but node discovery is manual); (C) LocalAI distributed (most features, heaviest ops). My recommendation [E]: A primary, B as the no-risk fallback if Mesh churn bites, since B needs no custom code and gives keys/budgets/fallbacks needed for the portal.
- Our own coordinator should be limited to the part nobody has: plan → parallel subtasks → verify → merge, and the async job queue.

### Gaps
- Not tested live; the plan's phase 0 two-machine Mesh test remains the real verification.

## Q4. Ranking by ease of aggregating upstream changes for a thin layer maintained by us

### Takeaway
Easiest: consume projects as unmodified binaries via OpenAI HTTP or YAML (llama-swap, LiteLLM-LTS, Ollama, Open WebUI). Medium: LocalAI via a gRPC backend we own; Mesh-LLM via an external plugin repo. Hardest: anything in Kubernetes-land or forks.

### Cited Findings
- Release/activity signals: llama-swap v272 (frequent small releases), LiteLLM LTS channel with ~daily releases, LocalAI minor releases roughly monthly (v4.9.0 Aug, v4.12.1 Oct 23), Mesh v0.78.1 with 1045 commits in 90 days, llm-d v1.0.0 on 2026-10-20 (first 1.0), production-stack chart 0.1.x — [APIs cited in Q1].
- Mesh plugins install by `mesh-llm plugins install <org/repo>`, so a plugin can live in our own repo and track releases independently — [USAGE.md](https://raw.githubusercontent.com/Mesh-LLM/mesh-llm/HEAD/mesh/docs/USAGE.md)
- LocalAI backends are separate OCI images in a gallery, built against the open interface — [README](https://raw.githubusercontent.com/mudler/LocalAI/HEAD/README.md)

### Inferences
Ranking, easiest to hardest for a thin layer [E]:
1. llama-swap (config-only, MIT, tiny surface; but low coverage).
2. LiteLLM proxy on LTS tag (config + optional `CustomLLM`; pin LTS; watch license `enterprise/`).
3. LocalAI (stable gRPC proto + MIT + OCI backends; heavier runtime, huge change volume mostly irrelevant to us).
4. Mesh-LLM (Apache-2.0, plugin can live in our repo; but pre-1.0 pace, MoA experimental, 1045 commits/90d means frequent rebase of any custom glue).
5. vLLM Semantic Router (Apache-2.0, fast-moving, its own models; use only if we need semantic routing).
6. Open WebUI (client only; license branding clause).
7. llm-d / production-stack / Ray Serve / GPUStack (Kubernetes/vLLM assumptions; only track ideas).
8. RouteLLM (stale; do not depend).

What to track upstream (watch releases): Mesh-LLM (plugin API, MoA, `openai-endpoint`), LocalAI (distributed + `backend.proto`), LiteLLM LTS notes (routing/affinity). What to adopt: Mesh as base per plan, LiteLLM-LTS as fallback/portal-keys front, llama-swap for single-box model switching on non-Mesh nodes. Ideas only: llm-d/Ray session-aware and prefix-aware routing for plan phase 4.

### Gaps
- Release cadence not computed from release lists (only latest tags fetched).
- No measurement of integration effort; all effort figures are my estimates.
