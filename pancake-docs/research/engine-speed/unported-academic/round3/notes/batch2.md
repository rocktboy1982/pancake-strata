# Round 3 batch 2: A8-A12 (online swap budget, pre-staging, conversation working set, checkpoints/FLOP eviction, multi-entry prefix store)

Read level for all five papers: **abstract only** (arXiv abstract page via WebFetch). No full text, HTML or PDF was read, so no paper's setup, per-step rule, or ablation is verified here. Where a claim below comes from the repo's own MOD_PLAN rows, it is labeled as the repo's record, not as a paper finding.

Repo grep (read-only, `/home/dan/clone-of-d/src/Strata`) and MOD_PLAN rows (lines 1795-1815, 1643, 1649, 1651, 1534-1590) were read.

## A8: online swap budget on the blocking tier (Pie, arXiv 2411.09317)

### Takeaway
The abstract gives no per-step swap-budget rule, no model, no GPU, and no exact-vs-lossy statement, so it cannot be used to predict the A8 gain. The repo already codes an online budget (`STRATA_ADAPT_SWAPS_AUTO`); the open question is the blocking-tier arm, which needs the night GPU confirm.

### Cited Findings
- Pie is "Pooling CPU Memory for LLM Inference", Yi Xu, Ziming Mao, Xiangxi Mo, Shu Liu, Ion Stoica, submitted 14 Nov 2024 — [arXiv abstract](https://arxiv.org/abs/2411.09317)
- Abstract claims "performance-transparent swapping" and "adaptive expansion" that "dynamically adjusts CPU memory allocation based on real-time information". This is CPU-memory sizing, not a per-step swap count — [arXiv abstract](https://arxiv.org/abs/2411.09317)
- Abstract names GH200 Grace Hopper as the platform whose bandwidth Pie uses; it does not name the experimental hardware or model — [arXiv abstract](https://arxiv.org/abs/2411.09317)
- Baselines: vLLM and FlexGen. Headline: "up to 1.9X in throughput and 2X in latency" vs vLLM; "up to 1.67X" GPU memory reduction; "9.4X higher throughput" vs FlexGen — [arXiv abstract](https://arxiv.org/abs/2411.09317)
- Repo record (MOD_PLAN 53, row A8, line 1795): the budget is coded as `STRATA_ADAPT_SWAPS_AUTO`; the async tier was rejected at -9.6%; the blocking arm is open; estimate 0-5% over the best fixed `--adapt-swaps` (est., not measured). Code: `src/program/generate.cpp:10537` lets the budget grow to 96 swaps a round when the variable is set, otherwise uses `o.adapt_swaps` capped at 96.
- Repo record (MOD_PLAN line 1534): the daytime confirm `25-a8` was invalid (ran during compilation); rerun is at night (NIGHT_QUEUE item 2). Screen `--adapt-swaps 64|128` is queued for the GPU phase (line 1512).

### Inferences
- The paper supports only the general idea of adjusting memory/swap effort online. Its headline numbers are vs vLLM/FlexGen on a different system, so they are not an estimate for our tier.
- The repo's own record (MOD_PLAN) already says the async version lost 9.6%. A blocking-tier gain can only be judged on our hardware.

### Gaps
- Per-step swap-count rule, exact/lossy status, model and hardware: not in the abstract. Full text needed before citing any mechanism detail.
- Pancake status: coded opt-in. Blocking-tier value unmeasured.

### Smallest test on our engine
- Env-only: `--adapt-swaps 64` and `--adapt-swaps 128` screen, then `STRATA_ADAPT_SWAPS_AUTO=1` vs best fixed, on the blocking tier with an A B B A run (per the repo's benchmark rule).
- GPU time: not measured. Night slot only. Our estimate is one screen plus one confirm (n=6), roughly 1-2 hours of GPU time (estimate). Quality gate applies (token forcing, task gate).

---

## A9: pre-stage next-turn experts while a tool runs (KVFlow arXiv 2507.07400; InferCept arXiv 2402.01869)

### Takeaway
Neither abstract supports "pre-stage while a tool runs". KVFlow's abstract describes background CPU-to-GPU loading for agents scheduled in the *next step*, with no mention of tool execution. InferCept's abstract describes the cost of recomputation during interceptions but not the mechanism. The idea rests on the repo's own framing.

### Cited Findings
- KVFlow: "Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows", Pan et al., 10 Jul 2025. Mechanism: "proactively loads required tensors from CPU to GPU in background threads for agents scheduled in the next step". Baseline: SGLang hierarchical radix cache. Headline: up to 1.83x (single workflow, large prompts), up to 2.19x (many concurrent workflows). Hardware and models not stated — [arXiv abstract](https://arxiv.org/abs/2507.07400)
- InferCept: "Efficient Intercept Support for Augmented LLM Inference", Abhyankar et al. Abstract: recomputation of already-computed contexts "accounts for 37-40% of total model forwarding time"; throughput 1.6x-2x; "2x more requests per second". Preservation/swap mechanism not described. Hardware, model, exact/lossy not stated — [arXiv abstract](https://arxiv.org/abs/2402.01869)
- Repo record (MOD_PLAN 17, line 1796): estimate 1-3% of the next turn's decode (est.); bench n.s. because there are no idle gaps. Code: `src/program/generate.cpp:14064` `STRATA_IDLE_ADAPT=N` (opt-in, adapts the tier between requests while no request waits). MOD_PLAN line 1560: no deciding test today.

### Inferences
- The abstract-level evidence supports "prefetch for the next scheduled agent", which is close to what `STRATA_IDLE_ADAPT` does, not "pre-stage experts during a tool call". The tool-time window is a repo hypothesis.
- The 37-40% recomputation figure is about KV recompute, not experts, so it does not transfer to A9 numbers.

### Gaps
- Whether KVFlow or InferCept do prefetch during tool execution: unknown from abstracts. Needs full text.
- Expert (not KV) pre-staging evidence from either paper: none.

### Smallest test on our engine
- Agent replay with a turn log that has real tool gaps: `STRATA_IDLE_ADAPT=N` vs off, measuring next-turn expert hit rate and decode time. Repo says the bench has no idle gaps, so the replay must have them.
- GPU time: not measured. Estimate about 1 hour of replay GPU time (estimate). Night only.

---

## A10: conversation working set / activation-matrix match at request start (MoE-Infinity, arXiv 2401.14361)

### Takeaway
MoE-Infinity's abstract supports sparsity-aware expert caching driven by traced expert activations, for batch-1 personal machines. Its headline is 3.1-16.7x per-token latency vs listed baselines. It does not describe request-start matching at the granularity the repo uses; that match is not confirmed by the abstract.

### Cited Findings
- MoE-Infinity: "Efficient MoE Inference on Personal Machines with Sparsity-Aware Expert Cache", Xue, Fu, Lu, Mai, Marina (v3, 12 Mar 2025). Mechanism: traces expert activations, selects the trace representing the sparsity pattern, uses it to guide replacement and prefetching. Targets batch size one. — [arXiv abstract](https://arxiv.org/abs/2401.14361)
- Models: DeepSeek and Mixtral. Baselines: vLLM, Ollama, DeepSpeed, BrainStorm. Hardware not in the abstract. Headline: "3.1-16.7x per-token latency improvements" — [arXiv abstract](https://arxiv.org/abs/2401.14361)
- Exact/lossy: not stated. Caching and prefetch do not change outputs by design, but the abstract does not say so.
- Repo record (MOD_PLAN 3-4 (+84), row A10, line 1797): +0.6-1.5 hit points (est.); inert under k-th miss until item 84 gives seeds virtual misses; status "done opt-in; item 84 open". MOD_PLAN line 1512 (finemoe replay): MoE-Infinity baseline was -19 to -32% on the only trace.

### Inferences
- The repo's own offline replay already found MoE-Infinity-style prefetch weak on our single trace (-19 to -32%), so A10's paper evidence is not the decider; the repo's trace is.

### Gaps
- Activation-matrix match at request start: not in the abstract. Needs full text.
- Our replay had one trace with no weights (bench/prefetch/routing.bin); the estimate is not a GPU result.

### Smallest test on our engine
- Item 84 (virtual misses for seeding) must land first, then a replay on more than one trace, then a GPU screen with the opt-in flag.
- GPU time: not measured. Estimate under 1 hour once item 84 is in (estimate).

---

## A11: checkpoints at tool boundaries and FLOP-saved-per-byte eviction (Marconi arXiv 2411.19379; FreeToken arXiv 2608.16157)

### Takeaway
Marconi's abstract supports a hybrid-model prefix cache with admission/eviction that weighs recency, forecast reuse, and compute saved vs memory footprint. It gives no formula and no FLOP-per-byte rule. The FreeToken paper was not opened in this batch; the repo's own MOD_PLAN reading (line 1651) says it checkpoints recurrent state only and has no checkpoint gain measurement.

### Cited Findings
- Marconi: "Prefix Caching for the Era of Hybrid LLMs", Pan, Wang, Jia, Karakus, Zancato, Dao, Wang, Netravali; MLSys 2025 camera-ready. Recurrent layers' in-place state updates prevent rolling back for partial overlaps, so only exact-match hits. Headline in abstract: up to 34.4x higher token hit rate; "71.1% or 617 ms" lower TTFT vs state-of-the-art prefix caching. Hardware, models, baselines not in the abstract — [arXiv abstract](https://arxiv.org/abs/2411.19379)
- Repo record (MOD_PLAN line 1643): no alpha numbers in the body; alpha = 0 is LRU; eviction-only vs LRU hit rate +19.0% to +219.7%; P95 TTFT -24.7% on SWE-Bench; 7B hybrid, 4xA100. The abstract's "-71%" merges two datasets (71.1% is ShareGPT, 617 ms / 46.8% SWE-Bench). This was checked against the full body in an earlier round, not in this batch.
- Repo record (MOD_PLAN line 1651, FreeToken 2608.16157 s3.1): checkpoints recurrent state only (not KV) at thinking/tool/turn boundaries; the only ablation is double-buffered loading (-19 to -26% prefill throughput on RTX 5090); no checkpoint gain measured.
- Repo record (MOD_PLAN line 1515, percache): tool-boundary marks expected ~0; FLOP eviction 0.5-0.8% at best; try `--prompt-cache-spread` first; A11 marks/FLOP/A13 HOLD/REJECT as engine items.
- Code: `--prompt-cache-spread` at `src/program/generate.cpp:964` and help text at 1161. The tool-boundary flag (527e06429 per MOD_PLAN line 1809) was not found by my grep (output was truncated at 60 lines); it needs a targeted grep before anyone cites it.

### Inferences
- The repo's offline replay on the real opencode store puts A11 marks at about 0 and FLOP eviction at 0.5-0.8% at best. The 34.4x and 71.1% numbers are Marconi's workloads, not ours.
- FreeToken's mechanism is a source for checkpoints, not a gain source.

### Gaps
- Marconi's eviction formula and alpha: not in abstract; repo says alpha is fitted by replay grid search.
- FreeToken: abstract not read in this batch; nothing measured for checkpoints.
- Tool-boundary code path: not confirmed by grep.

### Smallest test on our engine
- Run `--prompt-cache-spread` on the agent replay first (the repo says so), then the tool-boundary mark on `restore_check --tool-boundary` (A14 gate first).
- GPU time: not measured. Estimate under 1 hour per arm on the replay (estimate). Night only.

---

## A12: multi-entry prefix store with pinned system+tools prefix (RadixAttention arXiv 2312.07104)

### Takeaway
The RadixAttention abstract (the SGLang paper) supports prefix KV reuse with "up to 6.4x higher throughput" vs baselines; it does not describe the radix tree, the pinning rule, or the workload. The pinned-prefix idea is already implemented as an opt-in and the repo's record says not to enable it for Qwen.

### Cited Findings
- SGLang paper "Efficient Execution of Structured Language Model Programs", Zheng et al. (v2, 6 Jun 2024). Abstract: KV-cache reuse runtime; "up to 6.4x higher throughput" vs state-of-the-art. Radix structure and workloads are not described in the abstract — [arXiv abstract](https://arxiv.org/abs/2312.07104)
- Exact/lossy: not stated in abstract (prefix reuse is generally exact, but the abstract does not say so).
- Repo record (MOD_PLAN line 1796-1815, row A12, status "done opt-in": `STRATA_AUTO_PIN`, `STRATA_PIN_PARK_TOKENS`). Up to ~7% of turn time (sub-agent switches, est.).
- Code: `serve/server.py:6517` (`STRATA_AUTO_PIN` gate, opt-in, pins system+tools prefix, parks the branch left at the pin); `src/program/generate.cpp:16703` reads `STRATA_PIN_PARK_TOKENS` (default 2048).
- Repo record (MOD_PLAN line 1515, percache): do NOT enable `STRATA_AUTO_PIN` for Qwen; pinned parked entries are never evicted; six sessions: new ones are not parked; -1.3% modeled. Only `--conversation-cache-mib 2048 --conversation-cache-min-tokens 8192` was above 1% (+1.89%, 734 s).

### Inferences
- The abstract's 6.4x is an SGLang-vs-baseline throughput number; it does not transfer to the turn-time saving our record estimates (~7%, est.).
- The repo's own modeled result (-1.3% with pinning on Qwen) argues against enabling the pin without a fix for pinned-entry eviction.

### Gaps
- RadixAttention's tree and eviction details: not in abstract.
- Pinned-entry eviction fix: no design in the record yet; a prerequisite for enabling it on Qwen.

### Smallest test on our engine
- Fix or bound pinned-entry eviction first, then `STRATA_AUTO_PIN=1` vs off on the sub-agent switch replay, with the restore gate (A14) on.
- GPU time: not measured. Estimate under 1 hour for a replay pair (estimate). Night only.

---

## Cross-item notes

- All five papers were read at abstract level. No number in this file comes from a full-text read. For any engine decision, full text or the authors' own artifacts are needed.
- The repo already has opt-in code for A8 (auto budget), A9 (idle adapt), A10 (opt-in, item 84 open), A11 (prompt-cache-spread; tool-boundary to confirm), A12 (auto pin). Only A8 has an open GPU arm; A12 is blocked on pinned-entry eviction for Qwen.
- Per the repo's rules (MOD_PLAN notes): GPU runs at night only, A B B A order, quality gate on any default change.
- GPU-time figures above are estimates, not measurements.
