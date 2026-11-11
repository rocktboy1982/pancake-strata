# Similarity matrix: NetworkPancake orchestration layer and portal vs existing projects (as of 2026-11-10)

Method: GitHub REST API (license, pushed_at, stars, archived; unauthenticated, rate limit hit after about 60 calls, so latest-release data is partial), raw LICENSE and README files, protocol spec pages. Convention: [E] estimate, [U] unverified (background knowledge or not checked this session). "Pushed" = `pushed_at` from the API on 2026-11-10 (any branch, so it overstates release activity). Builds on, and does not repeat, `Plan NetworkPancake inferență distribuită/config_ui_orchestration.md` (exo dashboard, Petals monitor, LM Link) and the Mesh-LLM note.

Plan parts used as columns: **P1** orchestrator-worker with verification; **P2** parallel thinking with vote and quorum stop; **P3** layer of several orchestrators; **P4** shared memory; **P5** async multi-user portal (queue, quotas, results with confidence and evidence); **OBS** traces/metrics.

## 1. Per-project matrix: coverage, license, activity, extension point, OpenAI-compatible endpoint

### Takeaway
No project covers P2 (vote with quorum stop) or "result with confidence and evidence" out of the box. Agent frameworks (LangGraph, Microsoft Agent Framework, OpenAI Agents SDK, Pydantic AI, ADK, CrewAI) cover P1/P3 plumbing and all talk to a custom `base_url`. Queues (Temporal, Celery, pg-boss, Procrastinate, NATS) cover the P5 queue. Two licence traps for a multi-user portal: Dify (no multi-tenant without written permission) and n8n (Sustainable Use). Two dead or dying: AutoGen (archived) and Flowise (archived 2026-08-13).

### Cited Findings

Agent and orchestration frameworks

| Project | Covers | License | Activity (2026-11-10) | Extension point | OpenAI-compatible endpoint without custom code |
|---|---|---|---|---|---|
| LangGraph | P1, P3 (graph, subgraphs), P4 (checkpoint + store), durable execution, human-in-loop | MIT | pushed 2026-11-10, 44.8k stars, `sdk==0.4.14` tagged 2026-11-06 | graph nodes, checkpointer interface, store; commercial LangSmith Deployment is separate | Yes through LangChain `ChatOpenAI(base_url=...)` [U, background] |
| AutoGen (Microsoft) | multi-agent chat | CC-BY-4.0 reported by API (code license not rechecked) | **Archived**, no bug or security fixes; last release python-v0.7.5 2025-09-30; successor Microsoft Agent Framework | none, do not adopt | n/a |
| Microsoft Agent Framework | P1, P3: graph workflows (sequential, concurrent, handoff, group), checkpointing, human-in-loop, OpenTelemetry built in, DevUI; Python, .NET, Go | license not fetched [U] | README live; repo activity not fetched (rate limit) | workflow graph, OTel | OpenAI-compatible clients likely [U] |
| AG2 (AutoGen fork) | multi-agent chat | Apache-2.0 | pushed 2026-11-09, v1.3.0 2026-11-04, 5.1k stars | agents, group chat patterns | via config `base_url` [U] |
| CrewAI | P1 (Crews, Flows event-driven) | MIT | pushed 2026-11-10, release 1.15.39 on 2026-11-09, 60.7k stars; commercial AMP control plane is separate | Flows, custom tools | via LiteLLM-style model strings [U] |
| OpenAI Agents SDK | P1 (handoffs, guardrails, tracing) | MIT | pushed 2026-11-10, v0.24.0 2026-10-15, 30.6k stars | `Model` interface, `OpenAIChatCompletionsModel`, `MultiProvider`; Responses-only features rejected on Chat Completions and non-Responses backends | Yes for Chat Completions; Responses-only features (tool search, reasoning mode/context) do not work on non-Responses backends |
| smolagents (HF) | P1 light, code-writing agents | Apache-2.0 | pushed 2026-11-03, 30.3k stars | `LiteLLMModel`, other model classes | Yes via LiteLLM or OpenAI-style model classes [U for exact class] |
| Pydantic AI | P1, typed outputs (schema verification), graph | MIT | pushed 2026-11-10, 21.1k stars | `OpenAIProvider(base_url=...)`, custom `AsyncOpenAI` client | Yes, documented as "OpenAI-compatible APIs" |
| Google ADK | P1, P3 (workflow runtime: graph-based, nested workflows), A2A support [U] | Apache-2.0 | pushed 2026-11-10, 22.2k stars | agent/workflow classes | "Model-agnostic" per README; LiteLLM route [U] |

Protocols

| Project | Role | License | Status |
|---|---|---|---|
| MCP | tools/resources for agents | Apache-2.0 (transition from MIT, mixed during transition) | spec 2026-07-28 current; see section 2 |
| A2A | agent-to-agent | Apache-2.0 | v1.0.0 2026-03-12, v1.0.1 2026-05-28, 26.5k stars, pushed 2026-11-10 |

Queues and durable execution (P5)

| Project | Covers | License | Activity | Extension point | Note |
|---|---|---|---|---|---|
| Temporal | durable workflows, retries, signals, visibility; strong fit for job resume after node failure | MIT | pushed 2026-11-10, 24.3k stars | Python/Go SDKs, activities | Heavy to run (server + DB) [U on footprint] |
| Celery | task queue, priorities per queue, rate limits | BSD-3 | pushed 2026-11-10, 29.2k stars | broker Redis/RabbitMQ | Priority and fairness are per-queue; per-user quota not built in [U] |
| RQ | simple Redis queue | BSD (Vincent Driessen) | pushed 2026-11-09, 10.7k stars | workers, job classes | |
| Arq | asyncio Redis queue | MIT | **last push 2026-04-16**, 3.0k stars | | slow, 7 months quiet |
| pg-boss | Postgres queue (Node), priorities, retries, singleton keys | MIT | pushed 2026-11-10, 4.1k stars | | Fits "one Postgres, no new infra" |
| Procrastinate | Postgres queue (Python), priorities, periodic, locks | MIT | pushed 2026-11-10, 1.4k stars | | Same idea for Python |
| NATS JetStream | durable streams, work queues, KV, object store | Apache-2.0 | nats-server pushed 2026-11-10, 21.0k stars | | Natural mesh-wide transport; more ops than Postgres [E] |

Visual platforms and UI

| Project | Covers | License (exact terms from LICENSE) | Activity |
|---|---|---|---|
| Dify | workflow builder, app portal | Modified Apache-2.0: **may not operate multi-tenant environment (one tenant = one workspace) without written authorization; may not remove logo/copyright in `web/` frontend**; commercial use otherwise allowed | pushed 2026-11-10, 160.8k stars |
| n8n | workflow automation | Sustainable Use License: use or modify only for **own internal business purposes or non-commercial/personal use**; distribute only free for non-commercial purposes; `.ee.` files need Enterprise license | pushed 2026-11-10, 210k stars |
| Flowise | visual agent builder | Apache-2.0 except enterprise dir | **Archived 2026-08-13**; wind-down announced 2026-07-29, end of life 2026-08-31, no successor named ([discussion](https://github.com/FlowiseAI/Flowise/discussions/6727)) |
| Langflow | visual flows | MIT | pushed 2026-11-10, 156.5k stars |
| Open WebUI | chat UI, multiple OpenAI connections | "Open WebUI License" (BSD-3 style + clause 4): **branding may not be removed or altered unless <= 50 end users in a rolling 30 days, or written permission, or enterprise license**; CLA | pushed 2026-11-10, 158k stars |

Observability

| Project | Role | License | Activity |
|---|---|---|---|
| Langfuse | LLM traces, evals, prompt mgmt | MIT Expat except `ee/` dirs; copyright now ClickHouse, Inc. | pushed 2026-11-10, 37.3k stars |
| Phoenix (Arize) | traces, evals | **Elastic License 2.0**: no offering as hosted/managed service giving access to substantial features | pushed 2026-11-10, 12.2k stars |
| OTel GenAI conventions | trace schema | Apache-2.0 (semantic-conventions repo) | **moved** to `open-telemetry/semantic-conventions-genai`; old repo page says "no longer maintained here" |

Sources: GitHub API per repo `https://api.github.com/repos/<owner>/<repo>` (e.g. [langgraph](https://api.github.com/repos/langchain-ai/langgraph)); [AutoGen README](https://raw.githubusercontent.com/microsoft/autogen/main/README.md); [Flowise README](https://raw.githubusercontent.com/FlowiseAI/Flowise/main/README.md); [Dify LICENSE](https://raw.githubusercontent.com/langgenius/dify/main/LICENSE); [n8n LICENSE](https://raw.githubusercontent.com/n8n-io/n8n/master/LICENSE.md); [Open WebUI LICENSE](https://raw.githubusercontent.com/open-webui/open-webui/main/LICENSE); [Langfuse LICENSE](https://raw.githubusercontent.com/langfuse/langfuse/main/LICENSE); [Phoenix LICENSE](https://raw.githubusercontent.com/Arize-ai/phoenix/main/LICENSE); [Celery LICENSE](https://raw.githubusercontent.com/celery/celery/main/LICENSE); [MCP LICENSE](https://raw.githubusercontent.com/modelcontextprotocol/modelcontextprotocol/main/LICENSE); [Agents SDK models doc](https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/models/index.md); [Pydantic AI OpenAI doc](https://raw.githubusercontent.com/pydantic/pydantic-ai/main/docs/models/openai.md); [smolagents models](https://raw.githubusercontent.com/huggingface/smolagents/main/docs/source/en/reference/models.md); [ADK README](https://raw.githubusercontent.com/google/adk-python/main/README.md); [CrewAI README](https://raw.githubusercontent.com/crewAIInc/crewAI/main/README.md); [LangGraph README](https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md); [Agent Framework README](https://raw.githubusercontent.com/microsoft/agent-framework/main/README.md); [Temporal README](https://raw.githubusercontent.com/temporalio/temporal/main/README.md); [A2A releases](https://github.com/a2aproject/A2A/releases); [OTel moved page](https://raw.githubusercontent.com/open-telemetry/semantic-conventions/main/docs/gen-ai/README.md).

Open-source "LLM batch job portal / async LLM job queue" projects found
- llm-d Batch Gateway: Kubernetes system with OpenAI-compatible `/v1/batches` and `/v1/files`; PostgreSQL 12+ for metadata, priority queue and events; Redis/Valkey optional (only for `redis-sortedset` dispatch; the Postgres SQL transport for async queues is labelled experimental); S3 or filesystem for files; Apache-2.0 badge; the processor sends to an "llm-d Router" with mTLS support; 28 stars, 468 commits, no GitHub release shown. Whether arbitrary OpenAI-compatible URLs work as backend is not confirmed. — [README](https://raw.githubusercontent.com/llm-d/llm-d-batch-gateway/main/README.md), [repo page](https://github.com/llm-d/llm-d-batch-gateway)
- AIBrix (vLLM project): Apache-2.0, v0.8.0 dated 2026-10-09, 5.2k stars. Its docs state a Batch API fully compatible with OpenAI Batch API, run as Kubernetes Jobs, storage on S3, Redis or local filesystem (from a search snippet; the repo main page did not mention batch). Kubernetes-bound. — [AIBrix batch docs](https://aibrix.readthedocs.io/latest/features/batch-api.html), [repo](https://github.com/vllm-project/aibrix)
- "Local Batch API" (PHP 8.4+), drop-in for OpenAI/Anthropic batch API on Ollama or LM Studio; tiny, license and activity not checked [U]. — [listing](https://awesome.ecosyste.ms/projects/github.com%2Fdigitaldreams%2Flocal-batch-api)
- Bifrost (Go gateway): forwards batch calls to hosted provider batch APIs, not a portal for your own nodes. — [article](https://www.getmaxim.ai/articles/top-4-llm-gateways-with-openai-batch-api-and-anthropic-batch-support-in-2026/)
- The search found no mature project that is both a multi-user portal (quotas, per-user results) and a self-hosted OpenAI-compatible batch runner. The search tool's own summary says so.

### Inferences
- OpenAI-compatible reach: every agent framework above can point at Mesh-LLM or Strata `serve/server.py` if the server implements `/v1/chat/completions` with tools and SSE. The risk is not connection but features: Agents SDK rejects Responses-only features on non-Responses backends, so build the contract on Chat Completions.
- Dify, n8n, Open WebUI are poor bases for a shared multi-user portal: Dify forbids multi-tenant use, n8n forbids third-party commercial hosting, Open WebUI needs branding kept above 50 users. For 3-5 trusted LAN users (plan phase 1.5) all three are usable internally [E; not legal advice].
- Phoenix (ELv2) and Langfuse (MIT core) are fine self-hosted; prefer Langfuse or plain OTel collector + Jaeger/Tempo to avoid ELv2.
- Weak recent signal on Arq (no push since April) and AutoGen/Flowise (archived) means: exclude.

### Gaps
- Latest release tags and dates for smolagents, Pydantic AI, ADK, Temporal, Celery, RQ, pg-boss, Procrastinate, NATS, Dify, n8n, Langflow, Open WebUI, Langfuse, Phoenix: API rate limit reached; only `pushed_at` known.
- Contributors counts, key open issues per project: not collected.
- Microsoft Agent Framework license and OpenAI-compatible endpoint support: not fetched.
- Langflow, NATS JetStream, Procrastinate, pg-boss feature claims come from background knowledge [U].
- No test was run against Mesh-LLM or Strata endpoints; the plan itself says `/v1/models` and SSE from Strata's server are unverified with Mesh probe.

## 2. Stable interfaces as our contract: maturity and risk

### Takeaway
Use Chat Completions (not Responses) as the model contract, MCP for tools but pin to a spec revision and avoid deprecated features, OpenTelemetry for traces with a version-pinned attribute set, and keep A2A optional (HTTP/JSON adapter only). gRPC `backend.proto` is internal to the LocalAI fallback and must not leak into portal or orchestrator code.

### Cited Findings
- OpenAI Chat Completions: not deprecated, no sunset date given, but "Responses is recommended for all new projects"; Assistants API was sunset 2026-08-26. — [OpenAI migration guide](https://developers.openai.com/api/docs/guides/migrate-to-responses)
- OpenAI Agents SDK keeps `OpenAIChatCompletionsModel`; some features (reasoning mode/context, tool search) are Responses-only and rejected on Chat Completions / non-Responses backends. — [Agents SDK models](https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/models/index.md)
- exo already serves `/v1/chat/completions` and `/v1/responses` (existing note), so Responses is implementable by local servers, but not shown for Mesh-LLM or Strata here. — [exo README](https://github.com/exo-explore/exo)
- MCP spec 2026-07-28 (previous: 2025-11-25) removes protocol sessions and `Mcp-Session-Id`, makes protocol stateless (no `initialize` handshake; version and capabilities in each request `_meta`), adds `server/discover`, replaces GET stream with `subscriptions/listen`, removes SSE resumability (broken stream loses the request; client must reissue), moves tasks to extension `io.modelcontextprotocol/tasks` (polling `tasks/get`, `tasks/update`, `tasks/list` removed), introduces Multi Round-Trip Requests replacing server-initiated sampling/elicitation/roots requests, adds `resultType`. — [MCP 2026-07-28 changes](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- MCP deprecates Roots, Sampling and Logging; HTTP+SSE transport formally Deprecated; DCR (RFC 7591) deprecated in favor of Client ID Metadata Documents; new policy with minimum 12-month deprecation window; OpenTelemetry trace context (`traceparent`, `tracestate`, `baggage`) documented in `_meta`. — [same page](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- A draft after 2026-07-28 only updates the Client ID Metadata Document reference to draft-03 and restores `_meta` fields. — [MCP draft changelog](https://modelcontextprotocol.io/specification/draft/changelog)
- MCP repo license is in transition from MIT to Apache-2.0 (contributions without relicensing consent remain MIT). — [LICENSE](https://raw.githubusercontent.com/modelcontextprotocol/modelcontextprotocol/main/LICENSE)
- A2A: v1.0.0 on 2026-03-12 with many breaking changes (push notification config merged, enum formats aligned with ProtoJSON, OAuth implicit/password removed, device code and PKCE added, `tasks/list` added, gRPC multi-tenancy); v1.0.1 on 2026-05-28 bug fixes only; v0.3.0 (2025-07-30) and v0.2.x had repeated breaking changes; moved from google-a2a/google to a2aproject. — [A2A releases](https://github.com/a2aproject/A2A/releases)
- OpenTelemetry GenAI conventions moved to `open-telemetry/semantic-conventions-genai`; the fetched pages gave no stability label, no release, and covers GenAI clients, MCP and provider conventions. — [moved page](https://raw.githubusercontent.com/open-telemetry/semantic-conventions/main/docs/gen-ai/README.md), [new repo](https://github.com/open-telemetry/semantic-conventions-genai)
- LocalAI backend gRPC and llama.cpp RPC: llama.cpp `rpc-server` README calls itself "fragile and insecure". — [llama.cpp RPC README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/rpc/README.md)

Risk assessment (mine)

| Interface | Use | Maturity | Risk | Mitigation |
|---|---|---|---|---|
| OpenAI Chat Completions (+SSE, tools, `response_format`) | worker and verifier calls, node endpoint | Very high, de facto; local servers implement subsets | Low; vendor-recommended direction is Responses [E: stay on Chat for 12+ months] | Keep an `LLMClient` adapter; add Responses adapter only if a needed feature requires it |
| OpenAI Responses | optional second adapter | Stable at OpenAI, partial in local servers | Medium for local backends | Do not make required |
| MCP | tool access for workers | Spec revises fast: two incompatible-in-practice transport shapes (stateful 2025-11-25, stateless 2026-07-28) | Medium-high on transport; low on tool schema | Pin revision, use official SDK, avoid Sampling/Roots/Logging, avoid server-initiated requests, expose tools via thin gateway so swap is local |
| A2A | agents across orchestrators (P3) | v1.0 Mar 2026, only one patch since | Medium; plan limits depth to 2, so internal HTTP contract (task -> result + confidence + evidence) is enough | Define our own JSON schema first, add A2A adapter later [E] |
| gRPC `backend.proto` (LocalAI) | fallback backend only | Stable inside LocalAI [U] | Medium-high if it leaks | Confine to backend adapter; no portal dependency |
| OpenTelemetry | traces, metrics, span per delegated subtask | Core SDKs stable; GenAI attributes had no stability label in fetched pages | Medium for attribute names | Pin semconv version, wrap in `tracing.py`, store our own attributes under a `pancake.*` namespace |
| Own job/result schema | portal and orchestrator | ours | Low | Version field, JSON Schema in repo |

### Inferences
- MCP 2026-07-28 breaks the session model of 2025-11-25; SDKs may support one or both. Pinning protocol version per worker pool is necessary.
- A2A and MCP adoption by agent frameworks (ADK mentions workflows; A2A in ADK is [U]) makes both likely to persist, but the plan's depth-2 hierarchy does not need A2A on day one.
- Because Chat Completions has no announced sunset and every candidate framework supports it, it is the lowest-risk model contract.

### Gaps
- Whether Mesh-LLM and Strata `serve/server.py` implement `tools` with streaming, `response_format`/JSON schema, `n>1` and `logprobs` (needed for quorum and confidence): not verified.
- OTel GenAI stability status and attribute list: pages fetched had no content; check `semantic-conventions-genai` docs/CHANGELOG.
- Current official MCP SDK support level for 2026-07-28 (Python/TypeScript): not checked.

## 3. Thin layer we must maintain, and closest existing projects

### Takeaway
We must own four things: delegation policy, verifiers plus quorum/vote, per-user quotas and fairness, and the result store with confidence and evidence. No candidate implements all of it. Closest three by coverage: LangGraph (orchestrator-worker/graph, durable state), Microsoft Agent Framework (concurrent and group workflows, checkpointing, OTel), llm-d Batch Gateway (batch portal semantics: files, jobs, priority queue, Postgres). Temporal is the best durability substrate but not an LLM project.

### Cited Findings
- LangGraph advertises durable execution, human-in-the-loop, short- and long-term memory; production deployment is a separate commercial LangSmith Deployment. — [LangGraph README](https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md)
- Microsoft Agent Framework: graph workflows with sequential, concurrent, handoff and group patterns, checkpointing, streaming, HITL, time-travel; built-in OpenTelemetry; DevUI; successor to archived AutoGen. — [README](https://raw.githubusercontent.com/microsoft/agent-framework/main/README.md)
- Google ADK: graph-based workflow runtime with HITL and nested workflows, model-agnostic. — [ADK README](https://raw.githubusercontent.com/google/adk-python/main/README.md)
- CrewAI: Crews (autonomous collaboration) and Flows (event-driven control); commercial AMP control plane for deployment, observability, governance. — [CrewAI README](https://raw.githubusercontent.com/crewAIInc/crewAI/main/README.md)
- llm-d Batch Gateway: see section 1 (OpenAI `/v1/batches`, `/v1/files`, Postgres priority queue). — [README](https://raw.githubusercontent.com/llm-d/llm-d-batch-gateway/main/README.md)
- Pydantic AI exposes typed outputs and an OpenAI-compatible provider. — [doc](https://raw.githubusercontent.com/pydantic/pydantic-ai/main/docs/models/openai.md)

Our thin layer (what to own) [E sizes]

| Piece | Why nothing covers it | Interface it sits behind | Size [E] |
|---|---|---|---|
| Orchestrator policy (delegate or answer directly; safety rule "never worse than best single node") | Frameworks give mechanisms (handoff, graph), not Mesh-aware policy with node capacity | `Policy.decide(task, cluster_state) -> Plan`; cluster state from Mesh `/v1/models` and console | 500-1500 lines |
| Verification (tests, JSON schema, double-worker compare) | Schema validation exists (Pydantic AI); test execution and compare-merge do not | `Verifier.check(subtask_result) -> verdict + evidence` | 500-1500 lines |
| Vote and quorum stop (10-200 chains) | Not found in any candidate | `Voter` consuming N async chat completions, cancels at quorum | 300-800 lines |
| Quotas, fairness, priorities | Celery/pg-boss give priorities; per-user token budgets and fairness are ours | `QuotaStore` in Postgres | 300-800 lines |
| Result store (confidence, evidence, provenance) | llm-d stores batch outputs, not confidence/evidence | Postgres tables + JSON schema `Result{answer, confidence, evidence[], trace_id}` | 300-600 lines |
| Portal UI (submit, position, status, result) | Open WebUI / Dify license limits or wrong shape | small web app over REST | 1000-2500 lines |

### Inferences
- Reasonable starting stack with lowest licence risk: Postgres + Procrastinate or pg-boss for the queue (no new infra), LangGraph or plain asyncio for orchestrator, our own voter and result store, OTel to Langfuse. Temporal only if automatic resume of multi-hour jobs proves necessary [E].
- Closest by feature overlap: (1) LangGraph, (2) Microsoft Agent Framework, (3) llm-d Batch Gateway. None is close to "most of it"; orchestration frameworks omit the portal, and the portal-like projects omit orchestration. This is a judgment from README claims, not from code reading.
- llm-d Batch Gateway is Kubernetes-oriented and 28 stars; adopting it for 3-5 LAN users is probably over-engineering, but its OpenAI Batch-compatible API is a good shape to copy for the portal REST contract.

### Gaps
- Not read at source: LangGraph's parallel `Send`/map-reduce API, voting examples, or any "self-consistency with early stop" library. A targeted search for such a library was not done.
- No code-level review of any project's extension points beyond README and docs lines.
- Licence of Microsoft Agent Framework not retrieved.

## 4. Strategy to avoid a divergent fork

### Takeaway
Do not fork; depend on released versions behind adapters, pin exact versions, watch a short list of repos via release feeds, and upstream only small docs/compat fixes. Highest-churn items to watch: MCP spec, Mesh-LLM, A2A, OTel GenAI conventions.

### Cited Findings
- MCP defined a feature lifecycle with a minimum twelve-month deprecation window and a registry of deprecated features, which gives a predictable change signal. — [MCP changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- AutoGen archived and Flowise archived within 2026 show component death is a real risk (Flowise code freeze 2026-07-29, archive 2026-08-13, end of life 2026-08-31, forks encouraged; AutoGen archived, redirected to Agent Framework). — [Flowise discussion](https://github.com/FlowiseAI/Flowise/discussions/6727), [AutoGen README](https://raw.githubusercontent.com/microsoft/autogen/main/README.md)
- Mesh-LLM LICENSE vs Cargo inconsistency and v0 ABI are already recorded in the plan. — Plan NetworkPancake revizuit.md

Strategy (my recommendations; GitHub feature details are standard behavior [U, not rechecked])

1. Watch list and mechanism
   - Tier 1 (could break us): Mesh-LLM, Strata (`serve/`), llama.cpp (the commit pinned by Mesh), MCP spec repo (`seps/` PRs, `deprecated` registry), A2A releases, `semantic-conventions-genai` CHANGELOG, chosen agent framework (LangGraph or Agent Framework), chosen queue.
   - Tier 2 (alternates): OpenAI Agents SDK, Pydantic AI, Temporal, llm-d batch gateway, LocalAI.
   - Mechanism: GitHub "Releases only" watch or `https://github.com/<o>/<r>/releases.atom` into one feed reader; for MCP and A2A also watch labeled issues/PRs (SEP labels per the MCP SEP workflow). Renovate or Dependabot for pinned versions; weekly 15-minute triage [E].
2. Adapter boundaries (all custom code behind five ports): `LLMClient` (Chat Completions), `ToolGateway` (MCP), `JobQueue`, `ResultStore`, `Tracer`. Orchestrator policy, verifier, voter, quotas depend only on these ports and on our `Task/Result` JSON Schema. Backend `backend.proto` stays inside the LocalAI adapter.
3. Versioning and pinning: pin exact versions and lockfile hashes; pin MCP protocol revision (`2026-07-28`) and A2A major (`1.x`); pin OTel semconv version and copy attribute names to one module; pin llama.cpp commit used by Mesh and record in plan. Our own contract versioned `pancake.task/v1`, `pancake.result/v1` with additive-only changes.
4. Contract tests: a small conformance suite (chat, streaming, tool call, JSON schema, `n>1`, logprobs, cancellation) run against Mesh-LLM and Strata endpoints in CI on every upstream bump; a second suite replays a recorded MCP tool session. A failing suite is the signal to hold the bump.
5. Exit rules: if a dependency shows archive notice, no release for 6 months [E], or a licence change (cf. Flowise, MCP licence transition, Langfuse now under ClickHouse), switch to the Tier 2 alternate; adapters keep the cost to one module.
6. Upstream contribution where cheap: docs and compatibility fixes (for example Strata `/v1/models` and SSE conformance, Mesh probe behavior), reproducible issues with the conformance suite as evidence; follow Strata's CONTRIBUTING (area tag in title, one problem per PR, opt-in features) per its AGENTS.md. No large upstream features before phase 0 results.

### Inferences
- Because the orchestrator and portal depend on ports, replacing LangGraph by Agent Framework or plain asyncio is a one-module change; replacing Mesh-LLM by LocalAI changes only the cluster-state source in the `Policy` input.
- The conformance suite is the best "cheap upstream contribution": it doubles as acceptance test for plan phase 1 ("a normal OpenAI client works").

### Gaps
- No data on how frequently each Tier 1 repo releases (rate limit); the triage cadence is an estimate [E].
- Mesh-LLM and Strata maintainers' receptiveness to PRs not assessed.
- Security posture of MCP servers and queue workers on a LAN (authz, tool sandboxing) not researched here.
