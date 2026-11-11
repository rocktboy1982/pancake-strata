# Pain points of a multi-user async job portal over LLM serving (queues, multi-tenancy, reliability, latency, observability, cost, security)

Date: 2026-11-10. Tag legend: [M] = confirmed from a primary source (official docs / paper) fetched in this session; [C] = community, vendor-blog or secondary report (anecdotal); [U] = unverified or my inference.
Scope note: I did not exhaustively diff against config_ui_orchestration.md / internet_networking.md; topics here are limited to queue/tenancy/ops. The plan's phase 1.5 portal grep only surfaced "do not open public portal / public Mesh network" (line 42 of the revised plan), so recommendations are written for a closed-beta, trusted-user portal.

## 1. Batch / async APIs: how they work and what users complain about

### Takeaway
The de-facto contract (OpenAI, Anthropic, LiteLLM) is: upload many requests, one 24h window, poll status, fetch result + error files keyed by custom_id, partial results on expiry. It is simple enough to copy 1:1 for the portal. Complaints centre on opaque queue limits, stuck/expired jobs without diagnosis, and no ordering guarantee.

### Cited Findings
- OpenAI Batch: 50% cheaper; only window `24h`; states validating, failed, in_progress, finalizing, completed, expired, cancelling (can take up to 10 min), cancelled; max 50,000 requests and 200 MB per file; per-model queued-token limit; 2,000 batches/hour; results in `output_file_id`, failures in `error_file_id`; order may differ so match on `custom_id`; output files deleted 30 days after completion [M] — [OpenAI Batch guide](https://developers.openai.com/api/docs/guides/batch)
- OpenAI on expiry: unfinished requests are cancelled, completed ones stay in the output file and are billed, each unfinished request goes to error file with code `batch_expired` [M] — same source
- Anthropic Message Batches: 50% discount; limit 100,000 requests or 256 MB; status only `in_progress` then `ended`; per-request result types (succeeded, errored, canceled, expired); expired = not sent before 24h, not billed; results downloadable 29 days; per-request param validation is asynchronous and errors appear only after the whole batch ended; stream results rather than download whole; ZDR not eligible (29-day retention) [M] — [Anthropic batch docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing)
- User complaints on OpenAI forum: "Enqueued token limit reached" while no batches active or all completed/failed; reported enqueued count 0 yet rejection; suspicion that max_completion_tokens or failed batches count; requests for estimated token count at submission/rejection; batches moving validating -> in_progress -> expired with no error; batches expiring in ~10 minutes; no official OpenAI acknowledgement found [C] — [thread 1](https://community.openai.com/t/no-batches-in-progress-but-get-enqueued-token-limit-reached/1033872), [thread 2](https://community.openai.com/t/the-enqueued-tokens-bug-is-still-active/1293702), [expiry thread](https://community.openai.com/t/batch-api-consistently-expires-without-completion-despite-valid-jsonl-need-investigation/1366718)
- Azure OpenAI batch jobs stuck in "validating" (a May 2026 report tied by Microsoft to a multi-region outage) [C] — [MS Q&A](https://learn.microsoft.com/en-us/answers/questions/5551588/azure-openai-batch-jobs-are-getting-stuck-in-valid)
- LiteLLM adds Files + Batches over engines lacking them: for `hosted_vllm/` deployments without `/v1/files` the proxy stores the input, runs each line itself (default concurrency `LITELLM_EXECUTED_BATCH_CONCURRENCY`=4), supports validating/in_progress/finalizing/completed, error file, cancel keeps completed output, `batch_expired` at 24h; needs DB; only chat/completions/embeddings/responses; no streaming lines; batch lines skip pre-call guardrails; results visible only to creating key; if the proxy replica running a batch restarts, batch is marked `failed` with `runner_lost` after >3 min without progress and must be resubmitted [M] — [LiteLLM vLLM batches](https://docs.litellm.ai/docs/providers/vllm_batches)
- LiteLLM managed batches can load-balance across several Azure deployments by writing files to every matching deployment; a doc FAQ asks whether later GETs hit the same deployment, answer not found; docs list `provider`/`target_model_names` filters unsupported for managed batches (400) and "pre-GA limitations" [C/M mixed] — [managed batches](https://docs.litellm.ai/docs/proxy/managed_batches), [managed files](https://docs.litellm.ai/docs/proxy/litellm_managed_files)
- LiteLLM does not pre-reserve budget for batch submissions (cost unknown up front); cost recorded on completion, so a tenant can over-submit [M] — [LiteLLM budgets](https://docs.litellm.ai/docs/proxy/users)

### Inferences
- [U] Copy the schema: job states incl. `expired`/`cancelling`, `custom_id` per item, separate result and error files, per-item `batch_expired`, partial results retained. This is cheap and familiar to users.
- [U] Fix known complaints by design: return estimated token/cost on submit and on reject, expose queue position and a reason for every non-progress state, validate per item synchronously at submit (Anthropic's deferred validation is a pain point), and make `runner_lost` auto-requeue instead of requiring resubmit.
- [U] On unreliable volunteer nodes a 24h window with partial results is the natural fit; item-level (not file-level) scheduling is essential.

### Gaps
- vLLM's own `run-batch` CLI and its failure semantics were not fetched.
- No primary complaint data for Anthropic batches (no forum search done).

## 2. Queueing and fairness (head-of-line blocking, quotas, preemption)

### Takeaway
Per-client RPM/TPM caps are the industry default but waste idle capacity; research (VTC, Llumnix) gives token-weighted fair scheduling and migration/priority. LiteLLM gives practical budgets/limits but with multi-instance, Redis and reservation caveats the portal must replicate or avoid.

### Cited Findings
- VTC (Sheng et al.): providers cap requests per client as a "rudimentary notion of fairness" that wastes capacity; VTC tracks per-client weighted input+output tokens served, built on continuous batching, work-conserving, service difference between two backlogged clients bounded by factor 2; code released [M] — [arXiv 2401.00588](https://arxiv.org/abs/2401.00588)
- Llumnix (OSDI 24): live migration of requests with their KV state across instances; treats priorities/SLOs; claims up to 1.5x faster high-priority requests and up to 36% cost saving at similar tail latency (arXiv numbers; OSDI numbers not checked); supports vLLM [M] — [arXiv 2406.03243](https://arxiv.org/abs/2406.03243), [USENIX](https://www.usenix.org/conference/osdi24/presentation/sun-biao)
- LiteLLM proxy: budgets need Postgres (global max_budget fails open w/o DB); scopes key, team, team-member, end-customer, per-model (enterprise); `budget_duration`; reset check every 10 min by default; tpm/rpm/max_parallel_requests per key/user/team; reserves estimated max cost per request and reconciles after (disabling lets concurrent requests overshoot); without Redis each instance counts separately (N instances admit up to N times limit); Redis restart can reload stale lower counter; `fail_closed_*` options return 503; `default_estimated_output_tokens` when max_tokens absent [M] — [LiteLLM users/budgets](https://docs.litellm.ai/docs/proxy/users)
- vLLM production-stack docs index lists KV-cache-aware, load-aware and prefix-aware routing and a distributed-tracing guide (details pages not read) [M-partial] — [production-stack docs](https://docs.vllm.ai/projects/production-stack/en/latest/)

### Inferences
- [U] Head-of-line blocking: separate interactive and batch lanes; chunk long jobs into items so one tenant's 50k-item file cannot occupy all nodes; dispatch round-robin or VTC-style by tokens served per tenant (cheap to implement at job-dispatch level rather than inside the engine).
- [U] Preemption on weak/volunteer nodes: only at item boundaries (requeue the item) since live KV migration (Llumnix) is infeasible over the internet.
- [U] Run limits in one authoritative place (the portal DB) rather than per-gateway counters, to avoid the LiteLLM multi-instance overshoot class of bugs.

### Gaps
- Ray Serve priority/fairness features were not researched; llm-d scheduler details not fetched; no measured HOL-blocking numbers on weak GPUs.

## 3. Reliability: durable workflows, retries, idempotency, node churn, poison jobs

### Takeaway
Every mainstream queue gives at-least-once execution; correctness comes from idempotent items and explicit timeouts/heartbeats. Celery+Redis has a well-known duplicate-execution trap for long tasks; Temporal and Postgres-based engines make retry state explicit at the cost of operating more infrastructure.

### Cited Findings
- Temporal activities: recorded complete once but code may run multiple times (e.g. worker crash after work, before reporting); recommended idempotency key = Workflow Run ID + Activity ID; must set a timeout (typically start-to-close) because a crashed worker cannot be detected directly; finish within minutes or heartbeat; do not hand-roll retries inside activities [M] — [Temporal activity docs](https://docs.temporal.io/activity-definition)
- Celery on Redis: unacked tasks reappear after `visibility_timeout` (default 1h); with late ack a task running longer can be run twice; advice: timeout above longest task, check all apps sharing the broker (shortest wins), make tasks idempotent [C, no official docs fetched] — [dev.to](https://dev.to/rayshell_bowman_24ed5836b/celery-retries-keep-duplicating-jobs-after-redis-visibility-timeout-56h7), [dev.to reliability layer](https://dev.to/akoladefaj/celery-loses-8-of-your-tasks-by-default-heres-the-reliability-layer-i-built-to-fix-that-40mc)
- Practitioner comparisons: Celery = good task queue, poor workflow engine, no durable execution; Temporal strong for long multi-step with waits but needs its own cluster; Postgres-based durable execution (Hatchet, DBOS, hand-rolled with SKIP LOCKED/advisory locks) suits small teams but lacks cross-service workflow semantics; no head-to-head LLM benchmark; many authors sell competing products [C] — [dev.to Celery vs Temporal](https://dev.to/uaslimcreate/task-queues-in-production-celery-vs-temporal-vs-native-fastapi-16hl), [dev.to Postgres durable](https://dev.to/pvgomes/postgres-is-all-you-need-for-durable-execution-454b), [Hatchet](https://www.mintlify.com/hatchet-dev/hatchet/introduction)
- LiteLLM's own batch runner shows the node-loss pattern: no progress >3 min -> `runner_lost`, batch failed [M] — [LiteLLM vLLM batches](https://docs.litellm.ai/docs/providers/vllm_batches)

### Inferences
- [U] For the portal (single Postgres, small team, closed beta) a Postgres item table with lease (`locked_until`), heartbeat, attempt counter and `FOR UPDATE SKIP LOCKED` covers most of what Temporal gives; reserve Temporal for multi-step "parallel thinking / orchestrator-worker" flows if they grow.
- [U] Poison jobs: cap attempts per item (e.g. 3), then dead-letter into the error file with reason, and quarantine the model/prompt combo, never block the batch. Also cap by node: an item that killed two different nodes is likely poison, not a flaky node.
- [U] Idempotency: item key = (batch_id, custom_id, attempt-independent); store first-completed result and ignore late duplicates (volunteer node may return after lease expiry). Node churn means lease timeout must scale with expected item duration on weak GPUs; use heartbeats with token-progress rather than fixed long timeouts.

### Gaps
- No primary docs fetched for Celery, Arq/RQ, pg-boss, Procrastinate, NATS JetStream (ack-wait, max-deliver would map directly to lease/attempt caps); no source on result verification of untrusted nodes (covered by other notes if at all).

## 4. Latency / throughput: KV-cache reuse, stickiness, cold start, model swap, weak GPUs

### Takeaway
Cache reuse and routing affinity give large wins but trade against fairness and isolation; model swapping on a node is slow and unmeasured in docs, so the scheduler should batch jobs by model.

### Cited Findings
- LMCache: KV cache layer extracting from vLLM/SGLang, offload to CPU/storage/network, PD disaggregation; authors report up to 15x throughput on multi-round QA/document analysis (their workloads); remote fetch costs prefill delay; context truncation can roughly halve prefix-cache hit ratio [M, authors' claims] — [arXiv 2510.09665](https://arxiv.org/abs/2510.09665)
- llama-swap: proxy that starts the upstream server matching the request's `model`, replacing the running one; by default one model at a time; `ttl` auto-unload; `matrix` for concurrent models; README documents no queueing behaviour and gives no swap-time numbers [M] — [llama-swap](https://github.com/mostlygeek/llama-swap)
- vLLM production-stack lists prefix-aware, KV-aware and load-aware routing; shared remote KV storage benchmark marked work-in-progress [M-partial] — [docs](https://docs.vllm.ai/projects/production-stack/en/latest/)

### Inferences
- [U] A job queue knows the whole batch ahead, so it can group items by (model, shared prefix) and send groups to a single node: session stickiness for free, fewer swaps. Prefer this to a generic round-robin.
- [U] Because llama-swap has no queue, concurrent requests for different models thrash the GPU; put the queue in the portal/agent, not in the swap proxy.

### Gaps
- No measured cold-start/swap times, no weak-GPU observations (consumer GPUs, small VRAM) found in this session; llm-d not fetched; Ray Serve not covered.

## 5. Observability and cost

### Takeaway
Langfuse and OTel GenAI conventions cover per-call tokens/cost; the typical gaps are missing intermediate spans, wrong model/price matching, token-type bucket mistakes, and no join between job, tenant and node.

### Cited Findings
- OpenTelemetry GenAI semantic conventions have moved out of the main semconv site to the repo github.com/open-telemetry/semantic-conventions-genai; the old page is no longer maintained; stability status not confirmed [M] — [OTel semconv gen-ai](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- Langfuse: ingested usage/cost takes priority over inferred; model matching via regex, custom model definitions possible; cannot infer cost for o1-family from strings; exclusive-bucket token types so mis-mapping silently applies wrong price; check every intermediate tool/LLM call is a span or cost is undercounted; tool output often causes context blowup cost [M] — [Langfuse cost management](https://langfuse.com/resources/engineering/llm-cost-management), [token usage docs](https://langfuse.com/docs/token-usage)
- A practitioner analysis of 100k Langfuse traces reports dashboards below actual cloud bill and projects with no cost tracking [C] — [substack](https://lebensold.substack.com/p/we-analyzed-100k-langfuse-traces)
- LiteLLM spend logs record 0 when no price is configured for a model [M] — [LiteLLM vLLM batches](https://docs.litellm.ai/docs/providers/vllm_batches)

### Inferences
- [U] For volunteer nodes "cost" is token counts and node-seconds, not dollars: set custom per-model prices (token-equivalent credits) or spend logs show zero.
- [U] Minimum schema: tenant_id, batch_id, item_id, attempt, node_id, model, prompt/completion/cached tokens, queue wait, run time, outcome. Roll multi-step token totals up by batch/item id, not trace id, because retries create several traces for one item.

### Gaps
- Helicone, Phoenix not researched; Langfuse self-host/ClickHouse scaling not found; current OTel GenAI stability status unverified.

## 6. Multi-tenant security and GDPR

### Takeaway
Shared prefix caches are a demonstrated cross-tenant timing leak; vLLM offers per-tenant `cache_salt`. Sending third parties' data to volunteer nodes is a legal question I found no direct source for.

### Cited Findings
- "The Early Bird Catches the Leak" (arXiv 2409.20002): timing differences reveal prefix-cache hits, enabling recovery of another user's prompt token by token (~234 queries/token, 92.3% accuracy in early version; later version reports 99% per-token accuracy); black-box study says Claude, DeepSeek, Azure OpenAI use KV/semantic cache sharing and are susceptible; mitigation proposals include sharing cache only for blocks of at least k tokens, and anonymizing inputs for semantic caches; arXiv preprint, numbers differ between versions [M for paper, claims about named vendors are the authors'] — [arXiv 2409.20002](https://arxiv.org/html/2409.20002v5)
- vLLM `cache_salt`: mixed into first-block hash so only same-salt requests share KV blocks; intended to block timing attacks in shared deployments; since v0.11 default hash is sha256; non-cryptographic hash options can "leak private information in multi-tenant environments" [M] — [vLLM prefix caching design](https://docs.vllm.ai/en/latest/design/prefix_caching/)
- LiteLLM: batch results visible only to creating key; batch lines bypass pre-call guardrails [M] — [LiteLLM vLLM batches](https://docs.litellm.ai/docs/providers/vllm_batches)
- Anthropic batches are not ZDR-eligible (29-day retention) — shows providers retain batch data longer than sync data [M] — [Anthropic batch docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing)
- EDPB-related commentary: processor's identifiability assessment follows the controller's, so data passed to a processor remains personal data; EDPB Opinion 28/2024 treats model anonymity case-by-case; EDPB draft blockchain guidelines address controller/processor roles in decentralised setups (draft status) [C, law-firm commentary] — [Mondaq](https://www.mondaq.com/data-protection/1857784/privacy-corner), [Slaughter and May](https://thelens.slaughterandmay.com/post/102ju1c/ai-development-meets-gdpr-key-takeaways-from-the-edpbs-latest-opinion), [Mondaq blockchain](https://www.mondaq.com/unitedstates/fin-tech/1616728/edpb-publishes-draft-guidelines-on-personal-data-in-blockchain-and-a-report-on-ai-privacy-risks-mitigations-in-large-language-models)

### Inferences
- [U] Set `cache_salt` = tenant id (or disable prefix caching across tenants) on every node; do not use LMCache cross-tenant sharing without salting.
- [U] A volunteer node operator sees plaintext prompts and outputs (no confidential inference on consumer GPUs). If jobs contain other people's personal data, the portal operator is probably a controller/processor needing a processor agreement with node operators; simplest mitigation for the closed beta: route personal-data jobs only to trusted/own nodes, tag jobs with a sensitivity flag, and keep result retention short and deletable. This is not legal advice and needs a lawyer.
- [U] Per-tenant API keys, results scoped by key (as LiteLLM does), signed short-lived job leases for nodes so a node can only fetch items it is leased.

### Gaps
- No source specific to GDPR roles of volunteer-computing nodes; no data on prompt leakage via LMCache/remote KV sharing; no source on node-side exfiltration or result-tampering in volunteer inference.
