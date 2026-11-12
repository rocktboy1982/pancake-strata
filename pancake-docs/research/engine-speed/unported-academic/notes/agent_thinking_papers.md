# Agent and thinking control papers (MOD_PLAN A19, A22, A40, A41)

Research date 2026-11-12. Read-only: no code, builds or GPU runs. Sources are arXiv HTML pages fetched through a summarizing fetch tool, so numbers are as the fetch returned them and were not checked against the paper's tables. Verification marks: [paper, full text read], [paper, abstract only], [unverified].

Summary:

| Item | Paper | Verification | Exact or lossy (text) | Placement | Verdict on the plan row |
| --- | --- | --- | --- | --- | --- |
| A19 thinking stop | Thought Calibration 2505.18404 | [paper, full text read] | lossy | engine hook (already built as opt-in) plus offline fit | Plan's "-20-40% on routine turns" is not in the paper; the ICLR citation is a different method |
| A22 chain stop | DISCO 2405.04304, SpecFit 2610.15417, MiMo-V2-Flash 2601.02780 | all three [paper, full text read] | exact | engine (spec path, STRATA_DISCO exists) | Plan's attribution is split: entropy is in DISCO, the per-position cost is in SpecFit, and MiMo has no stop rule |
| A40 keep decoding during tools | AsyncFC 2605.15077 | [paper, full text read] via summarizing fetch | lossy | agent client or a proxy in front of the engine; not the engine | Worth a log-only overlap measurement before any build |
| A41 read-before-edit and test-before-done | ECLoop 2607.28815 | [paper, full text read] via summarizing fetch | lossy | agent client (harness hook); not the engine | Read-before-edit is already enforced by opencode (batch-3 notes, [unverified] today); the test gate is the only new part |

## A19 - learned thinking stop (linear probe on hidden states)

### Takeaway
Thought Calibration gives a real, risk-controlled early-exit rule for reasoning models, with a 60% in-distribution and 20% out-of-distribution cut in thinking tokens on math and science benchmarks. It is not evidence for agent turns, and the ICLR paper in the plan row does not study thinking stops at all.

### Cited Findings
- Method: each reasoning step is split at a "\n\n" boundary that contains "wait" or "but". Each step uses the mean last-layer hidden state of its tokens, reduced to d=256 by PCA, and a linear probe averaged over a 10-step window; labels come from Qwen 3 32B as a judge (correct, consistent, leaf, novel). Stop when the probe score passes a threshold chosen by fixed-sequence testing with binomial-tail p-values (Learn-then-Test). [Thought Calibration](https://arxiv.org/abs/2505.18404)
- Results: on s1K-1.1 test (in-distribution, DeepSeek-R1-distill Qwen 2.5 32B, R1-distill Llama 3.3 70B, QwQ 32B), the probes cut thinking tokens by up to 60% while keeping accuracy near full. Out-of-distribution (Qwen 32B probes on AIME 24, GPQA Diamond, MATH-500) the cut is up to 20%; it beats budget forcing on AIME 24 and GPQA and matches it on MATH-500. [Thought Calibration](https://arxiv.org/abs/2505.18404)
- Calibration: the supervised-label probe is over-confident out of distribution; the "consistent" probe generalizes better. [Thought Calibration](https://arxiv.org/abs/2505.18404)
- Limits stated by the authors: the guarantee holds only over draws of the calibration set, so calibration data must resemble deployment; small datasets; single run; early exit only. [Thought Calibration](https://arxiv.org/abs/2505.18404)
- The ICLR "Learning How Hard to Think" paper (arXiv 2410.04707, the arXiv version of the plan's ICLR link) is about per-query compute allocation before generation, not a thinking stop. A two-layer MLP or LoRA predicts marginal reward gains per budget from the query; extra best-of-k samples or calls to a strong model are routed to queries predicted to benefit. Reported: math (Mathstral-7B, oracle verifier) 25-50% less compute at the same success rate as best-of-k; chat 0-10% saving on the full set; routing 25-40% fewer strong-model calls. [Learning How Hard to Think](https://arxiv.org/abs/2410.04707) [paper, full text read]
- OS-Pruner (plan's A19 row, https://www.alphaxiv.org/abs/2607.11089): the abstract frames chain-of-thought pruning as an optimal stopping problem and claims 20-60% shorter generation "with minimal accuracy sacrifice". No accuracy numbers, and the abstract does not say whether a probe is used. [OS-Pruner](https://arxiv.org/abs/2607.11089) [paper, abstract only]
- Plan row correction: the row says "-20-40% thinking on routine turns (authors; untested on agents)". The authors' numbers are 60% in-distribution and up to 20% out-of-distribution, on math and science, not agent turns. The citecheck entry in MOD_PLAN.md already records this.
- Plan row correction: the row and batch-2 notes use a probe every 64 tokens or at "\n\n". The paper's checkpoint is the "\n\n" step containing "wait" or "but". The 64-token trigger is Strata's own default (`STRATA_HIDDEN_DUMP ... every=64`, `STRATA_THINK_PROBE` check points), not the paper's.
- Plan context (MOD_PLAN.md, A19 HOLD and agent-loop rows): agent turns have thinking p99 1,061 tokens, thinking is 22% of turn time, and there are only 34 logged turns. The HOLD line says the probe saves at most about 4% of turn time.

### Inferences
- Applying the paper's own ranges to Strata's agent turns gives 22% x 20% = 4.4% to 22% x 60% = 13.2% of turn time, at best. The "<= ~4%" HOLD figure matches the out-of-distribution end. The probe is worth building only if agent turns carry much more thinking than the logged 34 turns show.
- The paper trains on step labels from a judge LLM on math. Strata would need labels from agent turns (a forced-stop replay on logged turns). Without that, the probe measures nothing about agents.
- A linear probe on the hidden state is an engine-side change (the hook reads hidden states); it does not change output format, but it changes text whenever it fires. It is lossy.
- The paper's "\n\n + wait/but" boundary is a different checkpoint rule from Strata's. A faithful replication would test that rule as well as the 64-token one.

### Placement and smallest test
- Placement: engine. The hook (`include/strata/program/think_probe.hpp`), the dump, and `STRATA_THINK_PROBE` already exist as opt-in code per the batch-2 doc. Offline training and threshold choice are in `tools/pancake/thinking_probe.py`.
- Smallest test: no new code. Collect dumps on a few dozen real agent turns with `STRATA_HIDDEN_DUMP` (GPU, so the night queue), label them with the forced-stop replay (`thinking_probe.py labels-plan`), then fit and check premature-fire rate. Stop there unless the thinking share of logged turns is well above the current 22%.
- Gaps: no agent-turn labels exist; OS-Pruner full text was not read; no results for any model besides the paper's three reasoning models.

## A22 - learned chain stop (drafter entropy and per-position new-miss cost)

### Takeaway
DISCO is the only paper of the three that uses drafter entropy for a stop rule, and it is exact (it only changes when to verify). SpecFit has the per-position cost idea, but it is a tree-subtree search whose gains are on a different target and baseline. MiMo-V2-Flash has no stop rule, only an entropy-to-acceptance fit. The plan's combined rule is therefore Strata's own proposal, not a published method.

### Cited Findings
- DISCO: after each draft token, a shallow 2-layer FFN scores confidence from the draft's top-k probabilities (k=10), its entropy, and the position; drafting stops when the score falls below a threshold, with a cap SL_max, both tuned on validation. The classifier is trained to predict whether the draft token matches the target. Reported: 10.3% average latency gain over the best static lookahead and 31.4% over a dynamic heuristic; "generating the exact same text". Hardware: one A100 80GB; targets Starcoder-15B and Vicuna-13B. [DISCO](https://arxiv.org/abs/2405.04304) [paper, full text read]
- DISCO limits: on MBPP its latency (12.58 ms) is still well above the oracle (10.18 ms); on CNN-DM the validation-tuned static lookahead is slower on test than SL-5 (authors attribute it to validation-test mismatch); only two model families; non-zero temperature tested only on CNN-DM at temperature 1. [DISCO](https://arxiv.org/abs/2405.04304)
- DISCO exactness: the paper says the verification adapts rejection sampling, which preserves the target distribution; the main results are greedy. [DISCO](https://arxiv.org/abs/2405.04304)
- SpecFit: chooses which draft tokens the offloaded MoE target verifies, by maximizing expected committed tokens over estimated cycle time. The cycle-time cost is the per-layer union of experts activated by the selected nodes, fitted by non-negative least squares on 144 probe points from GSM8K calibration questions. Chain stop: drafting stops when the selected subtree has no node from the newest level. The paper does not use drafter entropy and has no per-position cost term. [SpecFit](https://arxiv.org/abs/2610.15417) [paper, full text read]
- SpecFit results (decode tokens/s; Qwen3-30B-A3B target, RedHatAI EAGLE-3 drafter, one A100 PCIe 40GB, 48 LRU expert slots): summarization +21.6% over cached EAGLE-3 (+31.6% without cache); GSM8K +7.0% (+13.1% without cache); RTX 4090 subset +10.8%. [SpecFit](https://arxiv.org/abs/2610.15417)
- SpecFit exactness: output preservation rests on greedy verification; the paper makes no formal equivalence claim. Route predictions are simulated at 60% accuracy; the deployable predictor is future work. [SpecFit](https://arxiv.org/abs/2610.15417)
- The plan's "SpecFit authors' ceiling +7-22%" is not in the paper. The paper's numbers are the +7% to +32% measured gains above, against EAGLE-3, not against DISCO or a static chain.
- MiMo-V2-Flash: fixed three-layer MTP draft, no chain stop or dynamic depth rule. The only entropy statement is an empirical fit, acceptance length = 4(1 - 0.58 x^0.58) with R^2 = 0.995, where x is next-token cross-entropy. Speedup 1.82x to 2.70x at acceptance lengths 2.8 to 3.8 (batch 32 to 128, 16K in, 1K out). [MiMo-V2-Flash](https://arxiv.org/abs/2601.02780) [paper, full text read; the first 100,000 of about 120,000 characters, which covers the cited sections]
- Strata-side facts (repo notes, batch-2 A22 doc): the MTP selection kernel returns only the top-1 probability, so draft entropy is not available cheaply; a per-position threshold table and `STRATA_DISCO` already exist as opt-in code; the internal ceiling for any stop rule is 1.0% short, 7.9% edit and 3.4% long (MOD_PLAN.md, disco-suffix row).

### Inferences
- The "drafter entropy and per-position new-miss cost" combination is split across two papers: entropy from DISCO, the cost term from SpecFit (whose cost is a whole-round expert union, not a per-position term). Nothing published combines them.
- DISCO's exactness argument applies to Strata's chain (`--spec 4`): a stop rule only changes when to verify, so it stays exact under greedy decoding. Text parity still needs the noise-aware check the plan already requires.
- SpecFit's tree search does not transfer: the plan rejects draft trees (Staged SD, SpecInfer, Sequoia). Only its cost-term idea transfers, and that is A23, already coded.
- Given the internal ceiling (at most about 8% on edits, about 1% on short turns), a learned stop is likely to give 0-3% in practice. The paper's 10% is against a static lookahead on a different model pair.

### Placement and smallest test
- Placement: engine, in the speculative path. The feature (entropy or top-k probabilities at each draft position) needs the engine to expose them; the fit is offline.
- Smallest test: add top-10 draft probabilities and entropy at draft positions to the window trace (record 5), by a host-side read of the draft logits where it exists, or a small reduction kernel otherwise. Then run `disco_fit.py` on the trace with features the existing rule does not have. This needs one window-trace GPU run (night queue), not a new decode path. Engine change is limited to the trace.
- Gaps: no entropy feature exists in the engine; the trace has not been collected at `--spec-min-p 0`; SpecFit's route predictor is untested.

## A40 - keep decoding while tools run (symbolic futures)

### Takeaway
AsyncFC runs tool calls in the background and lets the model keep decoding with placeholder futures, with no model change. The reported speedup is 1.40x on SWE-bench Lite with accuracy unchanged, and 1.17x to 1.26x on the BFCL and HotpotQA workloads. The mechanism lives in the function-call runtime between the model and the tools, so it is a client or proxy change, not an engine change.

### Cited Findings
- Mechanism: function schemas are rewritten so arguments may be future IDs; a call returns a placeholder at once; the model can pass unresolved futures to later calls; an `await_future` function stops decoding when the model needs a value; completed results are inserted as user messages at turn boundaries; calls are queued and checked against access labels so independent ones run concurrently; a failed call cancels its dependents. [AsyncFC](https://arxiv.org/abs/2605.15077) [paper, full text read via summarizing fetch]
- Results: SWE-bench Lite 1.40x over SWE-agent, accuracy 79.3% vs 79.0%; BFCL v4 Web Search 1.26x over sequential function calling, accuracy 53.2% vs 54.8%; HotpotQA 1.24x, accuracy unchanged at 75.0%; BFCL v3 with Gemini 3.1 Pro 1.17x with accuracy 62.0% to 65.3%. Against parallel function calling on SWE-bench the latency gain is 1.04x in the baseline comparison, with a significance claim and no number given. [AsyncFC](https://arxiv.org/abs/2605.15077)
- Exactness: the paper does not classify output as exact or lossy. Tool outputs are inserted unchanged at turn boundaries, but the model decodes past values it has not seen, and no rollback is described for a wrong guess about a placeholder. The evidence is task accuracy, not token equality. [AsyncFC](https://arxiv.org/abs/2605.15077)
- Limits: gain depends on workload (strictly sequential tasks and near-zero-latency tools gain little); decode overhead when a call completes during `await_future`; the scheduler serializes unless the tools are annotated; some results are not significant; few models; hardware not specified. [AsyncFC](https://arxiv.org/abs/2605.15077)
- Plan row says "up to 1.4x end to end on code (authors)" (matches SWE-bench Lite, one benchmark) and "L (opencode fork)".

### Inferences
- Lossy: the decode conditions on values the model has not seen, so text can change. The plan's "lossy" label is right.
- The gain is bounded by the overlap of tool wall time with decode time. For a local engine the decode step may be slower than the paper's hosted models, so the overlap share matters more than the speedup figure.
- The engine needs no change beyond normal chat decoding; the runtime that holds the futures is the client or a proxy in front of Strata's OpenAI-compatible endpoint.

### Placement and smallest test
- Placement: agent client (opencode) or a proxy in front of the engine. Not the engine.
- Smallest test (log-only, no GPU, no fork): from existing opencode session and turn logs, measure per turn the wall time of tool calls that do not feed the very next model step (reads, greps, test runs whose output the model does not need to continue) against total decode time. If that overlappable share is under about 10% of turn time, drop A40. Only if it is larger, prototype the runtime as a proxy and measure on one SWE-bench-style task set.
- Gaps: no evaluation with local models or the Strata engine; no rollback data; full tables not checked (summarizing fetch).

## A41 - read-before-edit and test-before-done loop (ECLoop)

### Takeaway
ECLoop's gain comes from a hard gate at the harness level: edits and submissions are held until task-specific evidence is recorded, with up to three retries. Reported gains are +4.8 to +11.8 points on SWE-bench Verified with two hosted models, and up to 12.1% fewer tokens. The generic natural-language nudge in its own ablation does worse than no guidance, so the plan's "generic end-of-turn nudge" is not the mechanism being tested.

### Cited Findings
- Mechanism: once per task an LLM proposes evidence conditions, which static analysis (AST, call graph, class hierarchy) grounds to code entities; deterministic checks mark conditions satisfied from the recorded trajectory; unmet conditions are added to context; when the agent proposes an edit or submission, the conditions that apply to that action are checked, the action is postponed with the missing evidence, up to three retries per target, then an audited fallback. Information-gathering actions are never blocked. Enforced by a harness hook (Codex CLI), not by prompting. [ECLoop](https://arxiv.org/abs/2607.28815) [paper, full text read via summarizing fetch]
- Results (SWE-bench Verified, 500 instances, Pass@1): mini-swe-agent GPT-5-mini 56.2% to 68.0% (+11.8 pp); mini-swe-agent MiniMax-M2.5 75.8% to 80.6% (+4.8 pp); Codex CLI GPT-5-mini 40.4% to 50.8% (+10.4 pp); Codex CLI MiniMax-M2.5 74.8% to 79.8% (+5.0 pp). Significance: exact McNemar, p < 0.001 for the mini-swe-agent runs. Self-Refine lowered Pass@1 by 1.4 and 1.8 pp. [ECLoop](https://arxiv.org/abs/2607.28815)
- Tokens: average tokens fell in all four configurations, by 12.1% at most (mini-swe-agent GPT-5-mini, cost -10.2%) and 1.4% at least (Codex CLI MiniMax-M2.5). Totals include the specification call. [ECLoop](https://arxiv.org/abs/2607.28815)
- Ablation (100-instance subset, mini-swe-agent GPT-5-mini; full ECLoop 68 solved): baseline 47; no guidance 63; no commitment check 58; no evidence-state update 59; natural-language summary instead of structured conditions 58. [ECLoop](https://arxiv.org/abs/2607.28815)
- Limits stated: depends on issue quality; the same model writes the spec, judges gaps and drives the agent; only SWE-bench Verified, two models, two scaffolds; some regressions when the fallback releases an under-supported patch; no non-Python or feature-addition tasks. [ECLoop](https://arxiv.org/abs/2607.28815)
- Read-before-edit: the batch-3 agent-loop notes (repo) say opencode already enforces it and that `edits_unread` would show a change. Re-verified today: [unverified] (no opencode check run in this session).

### Inferences
- Lossy: the gate changes the trajectory and sometimes the final patch. The plan's label is right.
- Harness-side. The engine's only role is to be the model. Nothing in the engine can gate tool calls.
- The ablation says a natural-language summary of missing evidence is worse than no guidance (58 vs 63), so a generic nudge probably does not help. The gain depends on structured, checked conditions and the hard hold.
- The compile step is one extra model call per task; on a local engine that is added latency that the paper's totals include but the plan has not costed.

### Placement and smallest test
- Placement: agent client (opencode plugin or hook), with the existing counter switch `STRATA_AGENT_COUNTERS` as the log. Not the engine.
- Smallest test, in two steps, both log-only first: (1) count edits and submissions with no prior matching read or with no test run after the last edit, from existing opencode logs (the A38 counters already record "edits without test" and "reads without edit"); if that count is near zero on real sessions, drop A41. (2) Only if it is not, add one hook that holds the first "task done" until a test command has run after the last edit, with three retries and an audit line. Do not build the full ECLoop compile step until (1) and (2) show a gap.
- Gaps: no test on non-SWE-bench tasks or on local models; compile-step cost on Strata not measured; read-before-edit enforcement in opencode not re-checked today.
