# Round 3 batch 3: A14, A15, A17, A18, A20

Scope: MOD_PLAN rows A14, A15, A17, A18, A20. Read-only research, no code, builds or GPU. Read levels are stated per item. Most sources were read from the arXiv abstract page only; where the full text was read it is marked. Tool-call budget was slightly exceeded (about 20 calls) to chase the s1 table values and the Continuum body.

Sources used (all fetched 2026-11-12):
- Hybrid-state cache recovery: https://arxiv.org/abs/2609.15030 (abstract page)
- Continuum: https://arxiv.org/abs/2511.02230 (abstract page); https://arxiv.org/html/2511.02230 (body, first 100k of 108k chars)
- s1: https://arxiv.org/abs/2501.19393 (abstract page); https://arxiv.org/html/2501.19393 (body, first 100k of 142k chars; budget forcing section and Tables 1-5 read)
- Are LRMs Interruptible: https://arxiv.org/abs/2510.11713 (abstract page)
- Just Enough Thinking / ALP: https://arxiv.org/abs/2506.05256 (abstract page)
- Danger of Overthinking: https://arxiv.org/abs/2502.08235 (abstract page)
- Thoughtology: https://arxiv.org/abs/2504.07128 (abstract page)
- Entropy Sentinel: https://arxiv.org/abs/2601.09001 (abstract page)
- Qwen3 report: https://arxiv.org/abs/2505.09388 (abstract page)
- TokenCake (related, not an A15 source): https://arxiv.org/abs/2510.18586 (abstract page)
- MORI, search snippet only: https://newx.sg/paper/detail/29025108-5e59-11f1-b84c-00163e10baa7 (sign-in page, no content); the search result summary lists it as a May 2026 academic paper, not verified.

## A14: restore-equivalence test for cache recovery with hybrid state

### Takeaway
The cited paper is a single-author, single-model integration repair (GLM-5.3-Flash on vLLM + LMCache, four-way TP) whose equivalence test compares restored-cache output against a modified recomputation control, not against an unmodified run. Its evidence supports the test shape the plan already uses (restore arm vs recompute control, per boundary), not a general determinism claim. The Pancake restore_check gate is coded; its one GPU run was not decisive for noise reasons, and the paper's control-arm idea is the fix the plan already names.

### Cited Findings
- Claim: a fully cached prompt restored the whole prompt state, but the scheduler credited one token fewer; the fix was strict-prefix lookup alignment, shared-computation corrections, matched checkpoint scheduling and fixed per-rank kernel configs. Experimentally validated for one model revision and one configuration; no general determinism or quality claim. [Hybrid-state abstract](https://arxiv.org/abs/2609.15030)
- Equivalence result: agreement with the modified recomputation control rose from 34/36 to 36/36 generations, 64 token IDs each. [Hybrid-state abstract](https://arxiv.org/abs/2609.15030)
- Additional validation: 72 paired 256-token continuations over two fresh-container runs for three synthetic templates passed; an instrumented run passed transfer-page, effective-tail and delayed-save checks. [Hybrid-state abstract](https://arxiv.org/abs/2609.15030)
- Speed: across 120 requests, output equality held; CPU reload cut TTFT by 46-64% and total request time by 1.9-7.0%, against modified cold recomputation. [Hybrid-state abstract](https://arxiv.org/abs/2609.15030)
- Pancake side (from MOD_PLAN, not an external source): restore_check fixed to exit 2 when a should-hit turn reports 0 cached tokens; the first GPU run gave exit 1, every turn differed from the re-read arm including `first` (no cache involved), the base binary gives the same pattern, and only `restore repeat == tool-retry` and `disk repeat == tool-retry` match. [MOD_PLAN, notes row for A14, line ~1530 and ~1818]

### Inferences
- Mechanism: the paper's own method is a comparison against a control that goes through the same modified code path. Pancake's first run compared against a free-running re-read arm, so it measured greedy-decode noise (the README gives 1.2-1.3% token divergence run to run), not cache correctness. The fix is what the paper implies: compare restore against a second recompute arm and report divergence position, or use teacher forcing.
- Exact or lossy: the test itself is exact (token equality or logit comparison). The cache path it guards is exact in the paper's setup (equality is claimed for the tested configuration only).
- Read level: abstract page only for the paper. The test procedure beyond "modified recomputation control" is not in the abstract.
- Existing status in Pancake: restore_check.py exists with per-arm engine logs and tests (13 passing per the plan). Not yet a decisive gate. The determinism concern in the plan is consistent with the paper's noise-aware framing.
- Smallest test: a teacher-forced check. Feed one fixed token sequence (a cached prefix plus a continuation), run once with the restored cache and once with cold recompute, and compare argmax and logit distance at every position. This removes free-running divergence. Covers a full hit, a partial hit and each checkpoint boundary the plan lists, with MTP on and off.
- GPU time: one prefill-only pass per arm per boundary, a few thousand tokens at most. Estimated at minutes, not hours. This is an estimate from the shape of the test, not a measured number.

### Gaps
- The paper's full text (test procedure, per-boundary list, MTP interaction) was not read; only the abstract. Single author, single model, so it does not establish generality.
- No external number exists for an equivalence test on a Qwen MoE or on the Pancake hybrid (RDNA3) path. Pancake's own result is the only evidence for that path.
- The paper's 34/36 and 36/36 counts are small samples; the gap between them is 2 generations and has no stated confidence interval.

## A15: TTL and idleness rule for parked sub-agent sessions

### Takeaway
Continuum's TTL rule is a tool-call pin whose value comes from a queueing term that is zero with one user, so the paper's gain does not transfer to single-user Pancake. Continuum does not cover sub-agents (the paper names them as future work). Pancake's park TTL is already merged; the offline replay found no gain from A15 alone, and the rank-3 eviction bug is still open. The "MORI" attribution in the plan is not supported by the Continuum paper and the MORI paper could not be read.

### Cited Findings
- Continuum pins a request's KV cache for a TTL when its output is a tool call; TTL tau* = argmax over candidate durations of P(tau, f) x (T x eta + PrefillReload) - tau, where P is the empirical CDF of that tool's recorded durations, T is a sliding-window queueing delay, and eta is a memoryfulness factor. Cold start uses a default T with Exp(1) durations and eta = 1. [Continuum body](https://arxiv.org/html/2511.02230)
- Stated gain: over 8x average job completion time (abstract); 1.12x to 3.66x lower delay and 1.10x to 3.22x higher throughput (introduction); up to 8.18x on a real SWE-agent deployment. [Continuum abstract](https://arxiv.org/abs/2511.02230); [Continuum body](https://arxiv.org/html/2511.02230)
- Conflict in the source: Section 6.2 says up to 2x reduction in average response time for Llama-3.1-8B against vanilla vLLM, which does not match the 3.66x range. The paper does not explain the difference. [Continuum body](https://arxiv.org/html/2511.02230)
- Hardware and models: A100, H100 and B200 GPUs; Llama-3.1 8B and 70B, Gemma-3 12B; GLM-4.5 appears only in an RL micro-benchmark. [Continuum body](https://arxiv.org/html/2511.02230)
- Sub-agents and idle time: not analysed. Non-linear flows and multi-agent coordination are listed as untested future work. No single-user load ablation was found. [Continuum body](https://arxiv.org/html/2511.02230)
- Related: TokenCake offloads idle KV during function calls with predictive upload and reports up to 47.06% end-to-end latency reduction in memory-constrained settings (abstract only). [TokenCake abstract](https://arxiv.org/abs/2510.18586)
- MORI (unverified): a search summary describes a May 2026 paper that estimates idleness from tool-invocation durations, uses hierarchical admission control and cross-tier KV migration, and reports 20-71% higher throughput and 18-43% lower TTFT on coding-agent workloads. I could not open the page (sign-in wall) and could not confirm the arXiv ID or its authors. [search summary, newx.sg link](https://newx.sg/paper/detail/29025108-5e59-11f1-b84c-00163e10baa7)

### Inferences
- Continuum's formula reduces to the reload term when queueing delay T is near zero. With one user (Pancake's case) T is near zero, so the TTL is driven by reload cost against tool-duration tail, and the gain comes from the same kind of over-budget and eviction-order effects that the Pancake replay already measured.
- This matches the plan's replay result: A15 alone gave no change, because 6 of 7 cold re-reads were over-budget skips, not evictions. The rule can only help once the eviction order is correct, which is why the rank-3 bug (evicts the whole held set under a byte shortfall, conversation_cache.hpp:754) matters.
- Pancake already has the knobs: `STRATA_PARK_TTL_S`, `_HOLD_S`, `_SHARE`, `_SUB_RATIO`. The plan lists A15 as exact (eviction order) and the sub-agent ranks as Pancake's own design, not Continuum's.
- Mechanism that Continuum does not test and Pancake would need: sub-agent parks where the parent is still active. That case is outside Continuum's evidence.
- Exact or lossy: exact for eviction order; the TTL value is a policy choice.
- Read level: Continuum full body for the formula and hardware (first 100k chars); abstract only for MORI and TokenCake.
- Smallest test: no new GPU run. Fix the rank-3 eviction (smallest entry whose bytes cover the shortfall) with a unit test, then rerun the offline park replay on the 2026-11-09 session (75 requests, 7 cold re-reads, 162.7 s). That needs CPU only. A GPU check is only needed after the replay shows a gain, and the plan already gates that on restore_check.

### Gaps
- MORI: the paper, its arXiv ID and its authors are not confirmed. The plan's "(MORI, same group)" attribution is not supported by any source I read, and the Continuum body does not mention MORI. Drop the attribution until the paper is read.
- No source reports TTL behaviour for a parked sub-agent session with a live parent.
- Continuum's per-model delay numbers exist only in figures (8-12), not in text; not read.

## A17: per-effort budget forcing with graceful close at the cap

### Takeaway
s1's budget forcing is a test-time scaling method, not a latency method: the forced end (appending the end-of-thinking delimiter and "Final Answer:") is used to cap thinking at about 30,000 tokens in the reported scores, and the "Wait" extension is what lifts AIME24 from 50% to 57%. The interruption paper shows that a forced close can leak reasoning into the answer or make the model panic, with drops up to 60% when information arrives late. Pancake's per-effort cap should therefore be treated as lossy, gated by a close-quality check, and sized from effort-tagged data that does not yet exist.

### Cited Findings
- Budget forcing: the end-of-thinking delimiter is appended to exit thinking early, optionally followed by "Final Answer:", to make the model give its current best answer. The paper's explicit cap for Table 2 is "around 30,000 thinking tokens". [s1 body](https://arxiv.org/html/2501.19393)
- Extension: the end-of-thinking delimiter is suppressed and "Wait" appended, up to 2, 4 or 6 times; gains flatten at about six. [s1 body](https://arxiv.org/html/2501.19393)
- Scores (s1-32B, with vs without budget forcing): AIME24 56.7 vs 50.0; MATH500 93.0 vs 92.6; GPQA Diamond 59.6 vs 56.6. [s1 body, Table 1](https://arxiv.org/html/2501.19393)
- String choice matters: 2x "Wait" gives AIME24 53.3, but 2x without any string gives MATH500 90.2 and GPQA 55.1. [s1 body, Table 4](https://arxiv.org/html/2501.19393)
- Abstract: "exceeds o1-preview ... by up to 27%" on MATH and AIME24; AIME24 raised "from 50% to 57%" by budget forcing. [s1 abstract](https://arxiv.org/abs/2501.19393)
- Interruption: under budget-constrained output, reasoning models show reasoning leakage (folding reasoning into the answer), panic (abandoning reasoning under time pressure), and self-doubt; performance drops by up to 60% when updates arrive late in reasoning. Models are not named in the abstract. [Are LRMs Interruptible, abstract](https://arxiv.org/abs/2510.11713)
- Abstract does not describe a graceful close procedure. [Are LRMs Interruptible, abstract](https://arxiv.org/abs/2510.11713)

### Inferences
- The s1 cap is a quality tool (cap at a fixed budget), not a time saving per turn. The time saving only comes from the tail above the cap, which for Pancake is the tail above p99 (about 1,061 thinking tokens per the plan). That is consistent with the plan's "tail only" framing.
- Lossy by construction: the paper's own evidence is accuracy, and the interruption paper shows failure modes at a cap. A per-effort cap needs an answer-quality check after the forced close, not just a token count.
- Pancake state (from MOD_PLAN): grace 512 and close_retry are in the presets; per-level caps are not applied (provisional {low 1024, medium 1536, high 3072}); `effort` is now written to the turn log (new key). The plan's 34-turn analysis found no cap that saves time on the current data.
- Read level: s1 full body for the mechanism and tables (first 100k of 142k chars); abstract only for the interruption paper.
- Smallest test: client-side, no GPU. Replay the existing and new turn logs with each candidate cap and count turns where thinking exceeds the cap and the final answer changes shape (leakage check: does the answer contain thinking-style text). Then, when 200+ effort-tagged turns exist, run the task gate with the cap on and off (the same A/B as A18).
- GPU time: the replay needs none. The live check needs agent turns, not a separate benchmark run, so cost is the wall-clock of the gate runs. Not estimated further here.

### Gaps
- The s1 scaling curve (Figure 4a) has no text values beyond the endpoints; not read in numeric form.
- The interruption paper's full text (method, per-model numbers, how the interruption is implemented) was not read.
- No source reports a per-effort cap for an agent workload; all the numbers are math or QA benchmarks.

## A18: effort per task rather than per turn (low vs high effort on total time to success)

### Takeaway
None of the cited sources measures time to success for an effort choice, so the plan's "-20-40% thinking (est.)" has no source and should be removed or marked as an assumption. The closest evidence is: ALP cuts tokens by 50% on a 1.5B math model (no time-to-success reported), and Danger of Overthinking links overthinking to lower agent performance on SWE-Bench Verified but its 43% saving comes from picking a candidate with lower overthinking, not from changing effort. The decision still has to be made on Pancake's own A/B with a time-to-success metric.

### Cited Findings
- ALP (Adaptive Length Penalty) ties length to each prompt's solve rate and reports a 50% cut in average token usage on DeepScaleR-1.5B without a large accuracy drop; benchmarks are not named in the abstract; time to success is not reported. [ALP abstract](https://arxiv.org/abs/2506.05256)
- Danger of Overthinking: reasoning models show three agentic failure patterns (Analysis Paralysis, Rogue Actions, Premature Disengagement); higher overthinking scores correlate with lower performance; choosing the lower-overthinking candidate improves performance by "almost 30%" and reduces compute by 43%. Benchmark SWE-Bench Verified, 4,018 trajectories, models not named in the abstract. [Danger of Overthinking abstract](https://arxiv.org/abs/2502.08235)
- Qwen3: a thinking budget mechanism lets users allocate compute at inference, to balance latency and performance; the abstract gives no effect size. [Qwen3 abstract](https://arxiv.org/abs/2505.09388)
- Plan metric: area under the time-to-success curve (SWE-Effi, A57). Not read here; cited from MOD_PLAN.

### Inferences
- The "-20-40%" in MOD_PLAN is not supported by any source read here. Mark it as an assumption, not an estimate from the literature.
- The 43% figure is a candidate-selection result on SWE-Bench, not an effort-per-task result. It does not transfer to a low versus high effort choice.
- The counter-evidence named in the plan (Qwen3 report) gives a budget mechanism, not a measured penalty; the abstract alone does not show that high effort hurts time to success.
- Pancake state (from MOD_PLAN): `xhigh` vs `medium` A/B is planned (owner decision); the task gate exists but has 18 offline tasks, not the BFCL / LiveCodeBench subsets. Per the plan's notes, 18 tasks x 5 reps gives a 95% CI of about 25 points, so a 10-15 point effect cannot be resolved at that size.
- Metric: use area under the time-to-success curve (SWE-Effi), not mean tokens.
- Read level: abstract only for ALP, Danger of Overthinking, Qwen3. No full text read in this batch.
- Smallest test: client-side A/B on the task gate, effort chosen per task type (not per turn), with time to success as the metric. Needs about 100 tasks or a 10-15 point margin to decide (plan's own sizing).
- GPU time: the runs are agent runs on the local engine. Cost = tasks x reps x median turn time (from the turn log). Not estimated here; the plan has no measured turn time for the gate yet.

### Gaps
- No source measures time to success for effort per task on an agent workload.
- ALP full text (benchmarks, accuracy numbers, any latency numbers) was not read.
- The "-20-40%" figure: no source; the plan needs either a citation or removal.

## A20: rumination rate and top-k entropy as a close signal

### Takeaway
Thoughtology describes DeepSeek-R1 rumination qualitatively, with no measurement method or numbers in the abstract, so it cannot set a threshold. Entropy Sentinel uses top-k logprob entropy for domain-level quality monitoring, not for stopping reasoning, and its abstract does not give k or any early-stop result. Pancake already logs rumination and entropy signals with a host-side cost (1.84 ms per token); the close signal is unmeasured and should stay log-only until the logs show separation.

### Cited Findings
- Thoughtology: DeepSeek-R1 "persistently ruminate[s] on previously explored problem formulations, obstructing further exploration"; the paper covers thought-length control, with a "sweet spot" where extra inference time can hurt. Only DeepSeek-R1 and an unnamed non-reasoning counterpart are named; no benchmarks or numbers in the abstract. [Thoughtology abstract](https://arxiv.org/abs/2504.07128)
- Entropy Sentinel: uses top-k logprobs and an output-entropy profile per response to predict instance quality; averages into a domain-level estimate for monitoring and intervention. Abstract numbers: on subjective tasks several models track the judge's slice scores with correlation up to r = 0.91; several models detect the worst categories near-perfectly; the coupling does not hold for all LLMs. Nine LLMs on verifiable tasks; models and k not named. [Entropy Sentinel abstract](https://arxiv.org/abs/2601.09001)
- Pancake (from MOD_PLAN): host-side logprobs cost 1.84 ms per token; no GPU entropy path; `rumination_entropy` and `entropy_below` exist in thinking_caps.py; an opt-in close is coded on feat/item-34-rumination; the entropy log is new. [MOD_PLAN, rows A20 and A20 status]

### Inferences
- Neither source gives a threshold, a rolling window or a close outcome that Pancake could copy. The rumination rate in the plan (rolling 5-grams) is Pancake's own design; the Thoughtology abstract does not say how repetition is measured, so it cannot be cited as a validation of 5-grams.
- Entropy Sentinel's evidence is monitoring quality at the domain level, which is a different question from whether to stop one reasoning trace. Do not cite it as a close signal.
- Cost: at 1.84 ms per token on the host, the entropy signal is cheap enough to log on every turn, but a close decision made on it adds a per-token host round trip unless computed on the device. That is a cost question for the close, not for logging.
- Read level: abstract only for both sources.
- Smallest test: no GPU. Replay the turn logs that already carry logprobs (or add the entropy log) and check whether top-k entropy and rumination rate separate turns that ended in a wrong answer from turns that ended right. Only if they separate, test a close on a small task set (A/B on the task gate).
- GPU time: none for the replay. A close test needs gate runs; no estimate here.

### Gaps
- Thoughtology: measurement method and numbers are not in the abstract; full text not read.
- Entropy Sentinel: k value and any early-stop experiment are not in the abstract; full text not read.
- No source gives a rumination threshold for a close on agent or coding tasks.
- The entropy log has no data yet (plan: new), so no threshold can be set from Pancake's own traces.

## Summary of what changes in MOD_PLAN (for the owner, not applied)
- A14: the test should be teacher-forced or use a second recompute arm; the first run's noise result is explained by the base binary's identical divergence pattern.
- A15: drop "(MORI, same group)" until the MORI paper is read; Continuum's gain is queueing-driven and the sub-agent part is not covered by it.
- A17: s1 is a test-time scaling method; cite its table values with the caveat that the cap is a quality tool, not a latency one.
- A18: remove or mark "-20-40% (est.)" as an assumption; no source supports it.
- A20: Thoughtology and Entropy Sentinel do not validate a close signal; keep log-only.
