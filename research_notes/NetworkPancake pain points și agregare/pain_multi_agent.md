# Pain points of orchestrator-worker / multi-agent LLM systems (as of 2026-11-10)

Tags: [M] measured in a cited paper/eval, [C] claimed (blog, forum, vendor), [U] unverified or secondary only. Plan read first (Plan NetworkPancake revizuit.md): phase 2 gate (>=1.5x faster at same correctness), phase 2b (orchestrator layer, depth 2, token budget, loop protection), phase 3 (voting over many small chains). The notes below are mapped to those. Prior research notes were not repeated.

## 1. Failure taxonomies and measured failure rates; multi-agent vs single-agent gains/losses

### Takeaway
Multi-agent systems fail mostly from specification/design, inter-agent misalignment and weak verification, not from raw model weakness. Gains are large only on decomposable, parallelizable tasks; on sequential tasks multi-agent is measurably worse; centralized orchestration with verification contains error amplification. This supports NetworkPancake's "strong node plans and verifies" and its safety rule (delegate only decomposable work).

### Cited Findings
- [M] MAST (Cemri et al.): 1,642 annotated traces, 7 open-source MAS frameworks, 14 failure modes in 3 categories, failure rates of 41% to 86.7% across the systems; annotator kappa 0.88; LLM judge (o1) 94% accuracy, kappa 0.77 — [arXiv abs](https://arxiv.org/abs/2503.13657), [HTML](https://arxiv.org/html/2503.13657)
- [M] MAST category shares (my sums of per-mode %): FC1 system design ~44% (step repetition 15.7%, loss of termination awareness 12.4%, disobeying task spec 11.8%); FC2 inter-agent misalignment ~32% (reasoning-action mismatch 13.2%, task derailment 7.4%, not asking for clarification 6.8%); FC3 task verification ~24% (premature termination 6.2%, incomplete verification 8.2%, incorrect verification 9.1%) — [arXiv HTML](https://arxiv.org/html/2503.13657)
- [M] MAST interventions: clarified roles + CEO as final decider +9.4% success on ChatDev; high-level objective check +15.6% on ProgramDev; authors say gains inconsistent, many failures remain. Verifiers in MetaGPT/ChatDev were often superficial (e.g. "code compiles"). MetaGPT had 60-68% fewer FC1/FC2 failures than ChatDev but 1.56x more FC3 failures — [arXiv HTML](https://arxiv.org/html/2503.13657)
- [M] Google/MIT "Towards a Science of Scaling Agent Systems" (arXiv 2512.08296, Dec 2025; exists, verified): single agent vs 4 multi-agent architectures. Blog version: 180 configs, 3 model families; +80.9% on Finance-Agent (decomposable) with centralized coordination; all multi-agent variants -39% to -70% on PlanCraft (sequential); independent agents amplify errors 17.2x vs 4.4x for centralized (orchestrator as "validation bottleneck"); predictive model R^2=0.513, picks best architecture for 87% of held-out configs; tool-coordination "tax" grows with tool count (qualitative only) — [Google Research blog](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)
- Version conflict: arXiv abstract page (later version) says 260 configs, 6 benchmarks, 5 architectures, +80.8% / -70.0%, R^2 0.373 (0.413 with capability metric), 87% — [arXiv abs](https://arxiv.org/abs/2512.08296); the blog/earlier version says 180 configs and R^2 0.513. Quote with version caveat.
- [U] A secondary summary claims a "45% single-agent accuracy" threshold beyond which more agents hurt; not confirmed in primary source — [search summary via zby.github.io](https://zby.github.io/commonplace/sources/towards-a-science-of-scaling-agent-systems)
- [M] Tran and Kiela (Stanford, arXiv 2604.02460, Apr 2026): at equal thinking-token budget, single agent matches or beats multi-agent on multi-hop reasoning (Qwen3, DeepSeek-R1-distill, Gemini 2.5); apparent multi-agent gains partly artifacts of unaccounted compute and API budget control; MAS become competitive when single-agent context use degrades or more compute is spent — [arXiv abs](https://arxiv.org/abs/2604.02460). Secondary numbers (e.g. 0.327 vs 0.418 on 4-hop MuSiQue, "2-6x efficiency penalty" with >10 tools) are [U] — [DEV summary](https://dev.to/greza_dev/why-single-agents-beat-multi-agent-systems-at-equal-token-budgets-445c)
- [M] LLMCompiler (planner + parallel executor DAG) vs ReAct: up to 3.7x latency speedup, up to 6.7x cost saving, ~9% accuracy gain — [arXiv abs](https://arxiv.org/abs/2312.04511)
- [C] Cognition "Don't build multi-agents": principles "share full agent traces" and "actions carry implicit decisions; conflicting decisions carry bad results"; parallel subagents without shared context produced mismatched Flappy-Bird pieces; architecture called "very fragile"; 2024 edit-apply split (small model applying big model's edit instructions) misread ambiguities, a single model doing both was more reliable — [Cognition](https://cognition.com/blog/dont-build-multi-agents)
- [C] Anthropic: lead Opus 4 + Sonnet 4 subagents beat single Opus 4 by 90.2% on internal research eval (~20 initial queries; small, internal); parallelization cut research time up to 90% on complex queries — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)

### Inferences
- NetworkPancake's task choice (multi-file code with tests, extraction from ~50 documents) is on the decomposable side; but multi-file code has coupling, where Cognition and Anthropic both warn (shared context, dependencies). Phase 2 must classify tasks by independence before delegating; fall back to single strong node otherwise (already the plan's safety rule).
- The orchestrator is the single point that contains error amplification (4.4x vs 17.2x). Plan phase 2b's layer of several orchestrators with no verifier above them resembles "independent" topologies; keep a verifying node above them.
- FC3 (verification, ~24%) and FC1 (repetition, termination) justify deterministic checks (tests, schema) rather than LLM-only verification.
- Gate of >=1.5x speedup at equal correctness is consistent with the literature: wall-clock gains are real (3.7x, up to 90%) but only on parallel structure; equal-budget comparisons often erase quality gains.

### Gaps
- Per-failure frequencies for planner-small-worker topologies specifically: MAST covers general frameworks, not a big-planner/small-worker split.
- Full numbers for BrowseComp-Plus and Workbench in the Google paper (figure only).
- Not read: MAST final 61k characters of HTML (appendix); Tran and Kiela full text.

## 2. Framework pain points (LangGraph, AutoGen/AG2, CrewAI, OpenAI Agents SDK, smolagents, Claude Code subagents)

### Takeaway
Reported issues cluster into: runaway loops/recursion and cost, state/checkpoint bugs in nested or parallel graphs (re-running completed nodes, lost writes), delegation not working in hierarchical modes (CrewAI), and non-determinism making debugging hard. My search found rich evidence for LangGraph, CrewAI and Claude Code; nothing usable for AutoGen/AG2, OpenAI Agents SDK, smolagents (not found, not a statement of absence).

### Cited Findings
- [C] LangGraph #9276 (open at time of search): resuming a compiled subgraph re-runs a node that already completed in the interrupted/failed super-step; reporter says sibling side effect ran twice in 20/20 runs without a patch; flat graph avoids it — [issue 9276](https://github.com/langchain-ai/langgraph/issues/9276)
- [C] LangGraph #9487: update_state on head of a failed run drops a finished parallel sibling's writes — [issue 9487](https://github.com/langchain-ai/langgraph/issues/9487)
- [C] LangGraph #9346 and #3206: subgraph with checkpointer=True returns previous run's output or is skipped after resume (the latter ended in recursion-limit error) — [9346](https://github.com/langchain-ai/langgraph/issues/9346), [3206](https://github.com/langchain-ai/langgraph/issues/3206)
- [C] langgraphjs #3307: tracking issue listing 16 open durable-execution bugs (interrupts, retries, error handlers, subgraphs, caching), 14 with fix PRs awaiting review — [issue 3307](https://github.com/langchain-ai/langgraphjs/issues/3307)
- [C] LangGraph #7117: main agent loses memory of previous tool invocations when invoking tool-call subgraph — [issue 7117](https://github.com/langchain-ai/langgraph/issues/7117)
- [C] deepagents #573: subagents lack checkpoint persistence; state query truncates tool history — [issue 573](https://github.com/langchain-ai/deepagents/issues/573)
- [C] CrewAI hierarchical process: manager cannot delegate ("coworker not found"), runs all agents instead of routing, delegates to wrong agent, only manager appears in self.agents; blog argues auto-created manager lacks domain context; fix suggestion: custom manager_agent, careful allow_delegation — [community 6302](https://community.crewai.com/t/hierarchical-process-manager-cannot-delegate-issue-coworker-not-found/6302), [community 3179](https://community.crewai.com/t/manager-agent-delegates-task-to-wrong-agent-in-a-hierarchical-process/3179), [community 7010](https://community.crewai.com/t/issue-only-manager-agent-appears-in-self-agents-when-using-process-hierarchical/7010). A TDS article on CrewAI manager-worker failure exists but returned 403, so unread — [TDS](https://towardsdatascience.com/why-crewais-manager-worker-architecture-fails-and-how-to-fix-it/)
- [C] Recursive subagent spawning in Claude Code: sub-agents spawning without depth limit, 50+ levels observed; ~800k tokens / $27.60 from one uncontrolled loop; background subagents burned usage limit after host exited — [68619](https://claudeissues.com/issue/68619-critical-subagent-spawning-and-subagent-pattern-bugs-trigger-infinite-recursion), [69578](https://claudeissues.com/issue/69578-bug-report-uncontrolled-sub-agent-recursive-loop-caused-800k-token-consumption-2), [72080](https://claudeissues.com/issue/72080-bug-sub-agents-stuck-in-infinite-loops-consuming-excessive-tokens)
- [C] Loops: agent re-reading same file range repeatedly with no circuit breaker (Cursor) — [forum](https://forum.cursor.com/t/agent-enters-infinite-loop-re-reading-the-exact-same-file-range-with-auto-mode/170975); text-to-SQL agent re-running near-identical queries until LangGraph recursion limit; LangChain forum points to model-call-limit middleware and recursion limits — [LangChain forum](https://forum.langchain.com/t/how-are-you-handling-runaway-agent-costs/3913). A guide notes a 12-step cap still burns 12 full turns, so detect lack of progress (repeated identical calls) earlier — [DEV](https://dev.to/isabelle_dubuis_d858453d7/when-agents-loop-forever-4-root-causes-and-how-to-stop-them-2ib1) [U, blog]
- [C] Anthropic: non-deterministic runs hard to debug; errors compound in long stateful runs; restarts costly; fixed via full tracing, checkpoints with resume, retries, rainbow deployments — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)

### Inferences
- Nested/parallel state + resume is where frameworks are buggiest. NetworkPancake's phase 1.5 "jobs resumed after node failure" should keep workers stateless and subtasks idempotent, with the orchestrator owning state, rather than relying on a framework's nested checkpointing.
- Depth limit 2, hard token budget and identical-call detection from the plan are directly supported; recursion caps alone are insufficient.

### Gaps
- No GitHub-issue evidence retrieved for AutoGen/AG2, OpenAI Agents SDK or smolagents (searches returned only Claude Code, LangGraph, CrewAI). Frequencies of each issue type across trackers not measured.
- Issue status (open/closed) as of Nov 2026 not re-verified beyond search snippets.

## 3. Small-model workers delegated by a big model

### Takeaway
Small workers work when subtasks are tiny, well-specified and redundantly voted with format checks (MAKER), and fail on loose instructions and missing context (Cognition). Planning quality of small models is much worse than large ones, so keep planning on the strong node. No benchmark found that measures small-worker instruction-following under a large planner directly.

### Cited Findings
- [M] MAKER (arXiv 2511.09030): 20-disk Towers of Hanoi, >1M LLM steps, zero errors using maximal decomposition (one step per agent), first-to-ahead-by-k voting, and red-flagging (discard over-long or malformed output). Per-step error: gpt-4.1-mini ~0.22%, o3-mini(low) ~0.18%, gpt-oss-20B ~3.6%, kimi-k2 ~4%, gpt-4.1-nano 36%, Llama-3.2-3B failed every step. Required k grows ~ln(steps); cost Theta(s ln s). Error rises sharply past ~700 output tokens; red-flagging reduced correlated errors; no step erred in both of two independent 10K-step runs. Projected cost ~$3.5K (gpt-4.1-mini). Without correction a 1% per-step error fails after ~100 steps — [arXiv HTML](https://arxiv.org/html/2511.09030v1)
- [M] Caveat on MAKER: task is a deterministic, fully specified puzzle with an exact checker; not shown for open-ended work (my reading; the paper's own scope).
- [M] Xu et al. (arXiv 2506.16411): chunked processing by weaker workers can beat a single strong model on long inputs because single-model error grows superlinearly with input length (search summary; paper not fully read) — [arXiv](https://arxiv.org/html/2506.16411v2)
- [M, abstract-level] HybridFlow (ICML 2026, arXiv 2512.22137): per-query dependency DAG, newly unlocked subtasks run in parallel, learned benefit-cost model routes each subtask to edge or cloud; reports lower latency and less cloud API use at competitive accuracy on GPQA, MMLU-Pro, AIME24, LiveBench-Reasoning; abstract gives no numbers — [arXiv abs](https://arxiv.org/abs/2512.22137). [U] search summary says small models (e.g. Llama-3-8B) plan much worse than GPT-5; not confirmed in text I read.
- [U] TDAG: pre-decomposed fixed plans suffer from error propagation (one early failure sinks whole task); proposes dynamic decomposition — [arXiv](https://arxiv.org/html/2402.10178v2)
- [C] Cognition: small model misapplied a large model's edit instructions; more reliable to have small model rewrite the whole file — [Cognition](https://cognition.com/blog/dont-build-multi-agents)
- [C] Anthropic: vague delegation caused duplicated work and gaps; fix: each subagent gets objective, output format, tool/source guidance and boundaries — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)

### Inferences
- Design contract: workers receive self-contained, short-output, schema-constrained subtasks; orchestrator discards malformed or over-long responses and retries (red-flagging) before any LLM-based verification. Phase 3 (voting) maps to MAKER but needs verifiable correctness; MAKER's low correlation of errors is only shown for one model family, so use diverse workers cautiously (two same-model workers may share errors).
- Verification cost: deterministic checks (tests, schema) are cheap; LLM verification by strong node costs tokens comparable to generation; MAST shows LLM verifiers are often superficial. Prefer executable checks.

### Gaps
- No direct measurement of verification cost vs generation cost in planner-worker setups, nor of context-transfer loss. No data for sub-10B open models as workers under a big planner beyond MAKER's per-step errors.

## 4. Token/cost multiples and when delegation pays off

### Takeaway
Anthropic reports ~4x tokens for agents vs chat and ~15x for multi-agent. Cost is economically justified only for high-value, parallelizable tasks; token usage explains most quality variance, so a good part of "multi-agent gain" is just more compute.

### Cited Findings
- [C] Anthropic: agents ~4x tokens of chat, multi-agent ~15x (verified in the post); token usage alone explained 80% of variance on BrowseComp; upgrading to Sonnet 4 beat doubling Sonnet 3.7 token budget; effort scaling: 1 agent / 3-10 calls for simple, 2-4 subagents for comparisons, >10 subagents for complex; multi-agent unsuited to tasks needing shared context, many dependencies, or most coding tasks; real-time delegation is something LLM agents are "not yet great at" — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- [M] Equal-budget studies reduce or erase multi-agent advantage on multi-hop reasoning — [arXiv 2604.02460](https://arxiv.org/abs/2604.02460)
- [M] LLMCompiler: up to 6.7x cost saving vs ReAct (because of fewer LLM calls and context re-sending, per planner-DAG design) — [arXiv](https://arxiv.org/abs/2312.04511)
- [C] Early Anthropic agents spawned 50 subagents for simple queries and searched endlessly — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)

### Inferences
- NetworkPancake workers are local/cheap, so a 15x token multiple costs time and energy, not dollars; the binding limit becomes wall-clock and strong-node bottleneck (planning plus verification serial work). Amdahl: the plan's >=1.5x gate is sensible; compare against strong model alone at equal token budget as well, per Tran and Kiela.
- Add an effort-scaling rule in the router (simple query -> no delegation).

### Gaps
- Anthropic's 4x/15x figures are single-vendor, internal; no independent replication found. No cost data for local-hardware swarms.

## 5. Mitigations practitioners report

### Takeaway
Consistent across sources: bounded effort (budgets, depth, call caps), explicit task contracts, executable verification, checkpoint-and-resume with idempotent steps, tracing, and context handoff via external storage.

### Cited Findings
- [C] Anthropic: effort-scaling rules in prompts; explicit subagent briefs; tool descriptions tested/rewritten by an agent (-40% task completion time); persist plan in external memory and spawn fresh subagents with clean context; subagents write outputs to external storage and pass references to cut copy overhead; checkpoints with retry; full tracing; LLM judge plus human eval; rainbow deployments — [Anthropic](https://www.anthropic.com/engineering/multi-agent-research-system)
- [M] MAKER: voting with margin k, red-flagging on length/format, maximal decomposition — [arXiv](https://arxiv.org/html/2511.09030v1)
- [M] MAST: single accountable final decider and high-level objective checks help modestly (+9.4%, +15.6%) — [arXiv HTML](https://arxiv.org/html/2503.13657)
- [M] Centralized orchestrator with validation cuts error amplification (17.2x -> 4.4x) — [Google blog](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)
- [C] Cognition: share full traces or keep single-threaded; avoid parallel writers making conflicting decisions — [Cognition](https://cognition.com/blog/dont-build-multi-agents)
- [C] Loop guards: iteration caps, duplicate-call detection, hard cost ceilings, model-call-limit middleware — [LangChain forum](https://forum.langchain.com/t/how-are-you-handling-runaway-agent-costs/3913), [DEV](https://dev.to/isabelle_dubuis_d858453d7/when-agents-loop-forever-4-root-causes-and-how-to-stop-them-2ib1)
- [C] CrewAI: explicit custom manager with domain role instead of auto-generated manager — [community](https://community.crewai.com/t/hierarchical-process-manager-cannot-delegate-issue-coworker-not-found/6302)

### Inferences
- Checklist for NetworkPancake phase 2/2b: (1) independence classifier before delegation; (2) per-job token and wall-clock budget, depth<=2, spawn cap, identical-call detector; (3) JSON-schema worker outputs, length cap, reject-and-retry; (4) executable verification first, strong-node review second; (5) idempotent, stateless workers with orchestrator-owned state and artifact store; (6) trace every delegation (non-determinism); (7) always compare to strong-node-alone at equal tokens; (8) a verifying node above the multi-orchestrator layer.

### Gaps
- Retry/idempotency practice is mostly argued from framework bugs (LangGraph re-running completed nodes) rather than a measured study. No measured effect sizes for budgets or depth limits.
