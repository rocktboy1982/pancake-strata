# Agent loop and thinking control, round 2

Research date 2026-11-12. Read-only: no code, builds, downloads or GPU runs. Round 1 is in `../../notes/agent_thinking_papers.md`.

Read levels: **full text** = arXiv HTML fetched and read through a summarizing fetch (numbers as returned, not checked against PDF tables); **model card** = Hugging Face page read through the same fetch; **abstract** = abstract or search-snippet only; **unverified** = search summary only.

## Key Question 1: A19 thinking stop on agentic or coding tasks

### Takeaway
No published thinking-stop or thinking-cap method has been measured on SWE-bench or on an agent loop with a stop-accuracy cost. The closest result is ThinkBrake (NeurIPS 2025): on BFCL (single-turn and live tool calls, Qwen3-4B-Thinking-2507) a training-free margin stop cuts thinking tokens about 25% with average accuracy flat (non-live 89.8 vs 89.6). OS-Pruner (2607.11089) is math-only, so it does not answer the agentic question.

### Cited Findings
- ThinkBrake stop rule: at each end-of-sentence boundary, stop when log(p(top token) / p(`</think>`)) <= 0.25. Training-free. [ThinkBrake](https://arxiv.org/html/2510.00546v1) — **full text**
- ThinkBrake on BFCL non-live, Qwen3-4B-Thinking-2507: base avg acc 89.6; ThinkBrake 89.8 with -24.9% average thinking tokens. Base avg thinking tokens 1,214 (per split 985 to 1,622). Per split, multi-parallel loses nothing (90.5 to 90.5, -28.2% tokens), multiple -17.5% tokens with 95.5 flat. [ThinkBrake](https://arxiv.org/html/2510.00546v1) — **full text**
- ThinkBrake on BFCL live: average 82.4 vs base 82.9, so it is not "never degrades" on live. [ThinkBrake](https://arxiv.org/html/2510.00546v1) — **full text**
- Simple fixed-cut baselines cost accuracy on BFCL: `NoWait` (-54.8% tokens) avg 87.9 (-1.7); `ThinkLess` (-100% tokens) avg 87.7 but parallel drops to 79.5 and multi-parallel to 83.0; the probability-space ThinkBrake variant drops to 75.9 avg with multi-parallel at 42.5. [ThinkBrake](https://arxiv.org/html/2510.00546v1) — **full text**
- Oracle termination on BFCL (search summary): 85.8 to 94.2 average accuracy with 80-94% fewer tokens, which shows the gain available from knowing when to stop. [ThinkBrake](https://arxiv.org/abs/2510.00546v1) — **abstract** (via search summary)
- A 2026 preprint on reasoning budgets on BFCL v3 Multiple (arXiv 2604.02155): Qwen2.5-1.5B, 32-token budget 64.0% vs 44.0% no reasoning; 256-token budget 25.0%. Non-monotonic. Small instruction models, not thinking models. [2604.02155](https://api.emergentmind.com/papers/2604.02155) — **abstract** (via summary, unverified)
- OS-Pruner (2607.11089): stopping rule is a policy stop probability sigma(W^T h + b) on the last hidden state of the reasoning prefix, last two attention layers fine-tuned; reward A - lambda*L; trained on DeepScaleR-Preview. Evaluated only on GSM8K, MATH-500, AIME24/25. **SWE-bench, BFCL and tool-use are not evaluated.** Headline: 20-60% length cut "with minimal accuracy sacrifice"; the text limits that claim to "<2%" accuracy loss. [OS-Pruner HTML](https://arxiv.org/html/2607.11089v1) — **full text**
- OS-Pruner Table 1 (forced answer), DeepSeek-R1-Distill-Qwen-7B: baseline avg 71.4 / 4823 tokens; OS-Pruner 71.1 / 3671. Table 2 at lambda = 5e-5: avg 69.8 / 2834 (-49%), so the cut costs about 1.6 points. AIME at lambda = 1e-4: 33.0 vs 48.0 (-15 points). [OS-Pruner HTML](https://arxiv.org/html/2607.11089v1) — **full text**
- OS-Pruner Table 1 on DRPO-7B: AIME 44.4 to 36.9 (-7.5 points), which the paper's own "minimal" framing does not cover. Table 1 is not from one lambda per row (paper inconsistency noted by the fetch). [OS-Pruner HTML](https://arxiv.org/html/2607.11089v1) — **full text**
- Agent-run early termination (not thinking stop): EarlyEval (2609.02783) stops about 35% of SWE-bench Verified runs at 95% prediction accuracy, cutting 26% of steps and about 30% of tokens. [EarlyEval](https://huggingface.co/papers/2609.02783.md) — **abstract** (via aiweekly summary; figure unverified). EET (2601.05777): 19-55% cost cut on SWE-bench Verified across three SWE agents, resolution down at most 0.2 points. [EET](https://fugumt.com/fugumt/paper_check/2601.05777v1_enmode) — **unverified** (search summary)
- Reasoning-only early stops measured on math, not agents (for scale): ESTAR 4799 to 1290 tokens at 98.9% of accuracy [ESTAR](https://arxiv.org/html/2602.10004v1); REFRAIN 20-55% fewer tokens [REFRAIN](https://arxiv.org/html/2510.10103v1). Both **unverified** (search summary).

### Inferences
- The answer to "does thinking-stop cost accuracy on agent tasks" is: no direct measurement exists in the sources read. The only agentic-style evidence is BFCL (tool-call accuracy), where a margin stop keeps accuracy and a fixed cut does not.
- ThinkBrake fits Strata better than the probe in round 1. It is training-free and uses only the next-token logits at sentence boundaries, which Strata already computes on the sampling path. It is not the same as the existing probe (`think_probe.hpp`). It still needs the engine to expose the `</think>` logit at each boundary, which is cheap (one extra value per boundary).
- OS-Pruner has no tool-use evidence, so it stays HOLD. Its AIME results show that a lambda that cuts length by half can drop hard-task accuracy a lot.
- Round-1 HOLD figure (at most about 4% of turn time) still stands. Using ThinkBrake's ~25% token cut on the 22% thinking share from round 1 gives about 5.5% of turn time at best.

### Measurement that settles it on our side
Replay logged agent turns (opencode session logs, or Strata turn log) with the `</think>` margin rule applied offline at each sentence boundary, if the logits can be recovered; otherwise run the rule on a few dozen BFCL-style or real tool-call prompts through the Strata server with the margin rule at threshold 0.25. Score tool-call name and argument match against the unstopped run. Stop if the mismatch rate is above 0 on tool-call turns, or if the thinking share of logged turns is below ~10%.

Estimate: offline replay on existing logs without GPU: about 1 day to write the replay and read logits, if the turn log keeps per-token logprobs (check first; if not, a night-queue GPU run is needed, 1-2 days including labels).

### Gaps
- No SWE-bench or agent-loop measurement of a thinking stop or cap was found in any source read.
- OS-Pruner full text read; no tool-use numbers exist in it.
- The 2604.02155 budget paper and the ESTAR/REFRAIN/EET numbers were not read in full.

---

## Key Question 2: A40 decoding during tool calls, and speculative tool execution

### Takeaway
Tool wall time is a large share of agent-turn time within a request (59.8% of response time per request in TraceLab's Claude Code and Codex traces), but across a whole session most time is human. The overlap that AsyncFC needs is what matters, and no source measures the overlappable share for coding agents. Speculative tool execution with rollback exists as design work (Atomix, ICLR 2026 speculative actions) but has no coding-agent timing measurement in the sources read.

### Cited Findings
- TraceLab (2606.30560), 4,265 sessions (2,676 Claude Code from 37 users; 1,589 Codex from 22 users), Sep 2025 to Jun 2026, 23 models. Per session with human gaps uncapped: human 92.3%, LLM generation 3.3%, tool execution 4.8%. Capped at 1 h: human 64.3%, LLM 14.5%, tool 21.2%. Per request: LLM 41.0%, tool 59.8%. Per step: LLM 40.7%, tool 59.3%. [TraceLab](https://arxiv.org/html/2606.30560) — **full text** (first 100,000 of 133,630 characters; Section 4.4 table and measurement method read)
- Method: Codex reports tool execution time directly; for Claude the paper derives it from call and return timestamps. Human time is the gap between events. [TraceLab](https://arxiv.org/html/2606.30560) — **full text**
- Mean per step: LLM 11.5 s, tool 16.8 s. Mean per request 4.3 min (median 38.3 s). [TraceLab](https://arxiv.org/html/2606.30560) — **full text**
- Heavy tail (search summary): calls over one minute are about 4% of invocations and about 85% of tool time. [TraceLab](https://syfi.cs.washington.edu/blog/2026-06-25-tracelab/) — **unverified** (secondary write-up)
- Per-request figures sum to 100.8%; the paper does not explain this. [TraceLab](https://arxiv.org/html/2606.30560) — **full text**
- Overlap (search summary): only 7.7% of aggregate tool wall time overlaps LLM calls in GitHub Copilot production traces. [Copilot trace write-up](https://codex.danielvaughan.com/2026/09/01/agentic-coding-in-the-wild-github-copilot-production-traces-kv-cache-codex-cli/) — **unverified**
- AsyncFC (2605.15077, round 1): SWE-bench Lite 1.40x with accuracy 79.3 vs 79.0; a placeholder-future runtime, no rollback described. [AsyncFC](https://arxiv.org/abs/2605.15077) — **full text via summary** (round 1)
- Speculative tool execution with rollback: Atomix (2602.14849) treats tool calls as transactions with Saga-style compensation, aborting losing branches. [Atomix](https://arxiv.org/html/2602.14849v1) — **unverified** (search summary). Speculative actions (ICLR 2026, Columbia) runs predicted actions in parallel and requires idempotent, reversible or sandboxed side effects. [Speculative actions](https://www.cs.columbia.edu/~kkaffes/papers/speculative-actions-iclr26.pdf) — **unverified**. PASTE (2603.18897) reports a 48.5% task-time cut (summary only). [PASTE via agentwiki](https://agentwiki.org/_export/xhtml/speculative_tool_execution) — **unverified**
- Ghost tool calls: a speculative call can leak intent to an external service before commit, and that cannot be rolled back. [Modelwire summary](https://themodelwire.com/article/ghost-tool-calls-issue-time-privacy-for-speculative-agent-tools-01KT36B63E6MQ74N8PZNYSC0AR) — **unverified**

### Inferences
- Within a request, tools are the larger share in these traces, so overlapping them with decode has room to help. Across a whole session, human time dominates, so the end-to-end gain is capped.
- Only read-only or reversible tools are safe to run speculatively on our side (reads, greps, test runs in a scratch copy). Writes wait for the model's commit.
- A per-request tool share near 60% means the overlap ceiling is high, but the 7.7% overlap figure says the actual overlap is small. These two numbers together say the gain depends on the share of tools whose output the next decode step does not need.

### Measurement that settles it on our side
From opencode session logs or the Strata turn log, for each request: (a) tool wall time; (b) decode time; (c) the share of tool calls whose result is not consumed by the very next model step (reads or searches followed by unrelated output, test runs the model does not wait on). Overlappable share = sum of (c) over total turn time. Drop A40 if under ~10% (round-1 threshold). Also count tool calls that are read-only, to bound speculative execution.

Estimate: log-only analysis of existing traces, 0.5-1 day, no GPU. A runtime prototype (proxy plus futures) is 1-2 weeks and only if the share is above 10%.

### Gaps
- No coding-agent measurement of speculative tool execution with rollback was found.
- TraceLab full text read up to 100,000 of 133,630 characters; the remaining sections were not checked.
- TraceLab covers Claude Code and Codex only, not opencode or local models.

---

## Key Question 3: A41 read-before-edit and test-before-done (ECLoop and harness gates)

### Takeaway
ECLoop (2607.28815) is a hard harness gate: edits and submissions are held until structured evidence conditions are met (up to three holds, then an audited fallback). Full text was read. The ablation is strong: dropping the commitment check costs 10 solved, and a natural-language summary of the same conditions does worse than no guidance. No replication or code release was found.

### Cited Findings
- Condition compilation: an LLM proposes conditions from the issue and repo structure; each is grounded with AST, call graph and class hierarchy to tuples (action type, entity, required event pattern, satisfaction predicate). Unresolvable conditions are removed. Satisfaction is computed deterministically from the trajectory, not from the model's claims. [ECLoop](https://arxiv.org/html/2607.28815v1) — **full text**
- Gating: commitment actions are source edits and final submission. File reads, searches and diagnostics are never blocked. A commitment is postponed when its evidence gap is not empty; the missing evidence is returned to the agent. [ECLoop](https://arxiv.org/html/2607.28815v1) — **full text**
- Retry: hold budget of three per target, then an "audited fallback" that proceeds with unsatisfied conditions recorded. The paper attributes the few regressions to this path. [ECLoop](https://arxiv.org/html/2607.28815v1) — **full text**
- Ablation, 100-instance subset, mini-swe-agent v2, GPT-5-mini, solved count: full 68; minus guidance 63; minus commitment check 58; minus state update 59; baseline (minus all) 47; natural-language spec instead of structured 58. The text says the NL variant falls "below even no guidance (Pass@1 of 63%)". [ECLoop](https://arxiv.org/html/2607.28815v1) — **full text**
- Full benchmark (SWE-bench Verified, 500): gains +4.8 to +11.8 points across two models and two scaffolds (round 1, full text via summary). Cost on GPT-5-mini: $0.047 base vs $0.042 ECLoop per instance (-10.2%), total $23.60 vs $21.18. [ECLoop](https://arxiv.org/html/2607.28815v1) — **full text**
- The compile step's cost is not reported separately; the reported totals include it. [ECLoop](https://arxiv.org/html/2607.28815v1) — **full text**
- Limits as stated: the same model writes the spec, judges gaps and drives the agent (authors suggest an independent model as future work); only SWE-bench Verified, two models, two scaffolds; no non-Python or feature tasks. [ECLoop](https://arxiv.org/html/2607.28815v1) — **full text**
- No code or data release link on the paper page. [ECLoop](https://arxiv.org/html/2607.28815v1) — **full text** (absence noted)
- Harness-level verification gates in coding agents: no source was read in this round that documents a shipped gate beyond ECLoop. The Strata A38 counters ("edits without test", "reads without edit") are the internal equivalent (from repo notes, not re-verified today).
- Frequency of edit-without-read in logs: no external measurement found. TraceLab gives no edit-level breakdown.

### Inferences
- The gain in ECLoop comes from a hard, checked hold, not from a prompt. A generic "run tests before done" nudge is not the tested mechanism and the ablation argues against natural-language guidance.
- For opencode, a test-before-done hold can be a plugin hook: hold the first "done" until a test command ran after the last edit, three retries, audit line. The compile step (one extra model call per task) is the cost the plan has not priced for a local engine.
- The paper's ablation is on one model and one subset (100 instances), so the effect size has wide error bars.

### Measurement that settles it on our side
Count, from existing opencode session logs (the A38 counters if present): (1) edits with no prior read of that file in the same session; (2) "task done" messages with no test command run after the last edit. If (1) or (2) is near zero in real sessions, drop A41. If not, add the one hold hook and compare solved-task outcomes on a fixed set of 20 local tasks, same model, same day, two orderings (A B B A, per owner rule).

Estimate: log count, 0.5 day, no GPU. Hook plus A/B: 2-3 days of code, then night-queue runs.

### Gaps
- No replication or ablation outside ECLoop's own paper found.
- No external log study of edit-without-read frequency found.
- ECLoop compile-step cost on a local model not measured anywhere.

---

## Key Question 4: A28 ISTA-DASLab "Coder" (primary source)

### Takeaway
The primary source exists publicly. ISTA-DASLab publishes two Hugging Face models named "...GSQ-RCO-Coder-GGUF": DeepSeek-V4.1-Flash-1017 and Qwen3.8-Flash-Next. "Coder" is a capability-targeted expert-pruning build (half the routed experts removed by RCO, weights quantized by GSQ, calibrated on code, agentic and vision data), not a separate code fine-tune. Both are relevant to Strata's two public families (AGENTS.md). Both cards report SWE-bench Verified and LiveCodeBench v6 only.

### Cited Findings
- Org page lists the two models among 144 models and 12 collections; org has 97 repos. [ISTA-DASLab org](https://huggingface.co/ISTA-DASLab) — **model card** (org page, truncated listing)
- Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF: base Qwen3.8-Flash-Next (card gives 176.9B params; HF metadata says 117B, unexplained). Half the 512 experts removed (256 per layer, 48 layers, 10 active). Transformer 3.5 bpw; n-gram table IQ4_NL 4.5 bpw. Total 58.4 GB vs 354 GB BF16; transformer shard 29.6 GB; card says it fits one 32 GB accelerator. SWE-bench Verified 82.80 base vs 79.20 (95.7% retained) at xhigh; LiveCodeBench v6 87.43 vs 87.43. October 17 update: SWE-bench 75.60 earlier build, 86.28 LCB; re-download first shard. Vision only qualitative. [Qwen3.8 card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) — **model card**
- DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF: base 763.2B total (551.9B transformer), 384 experts, 6 active. Half removed (192 per layer). Kept experts at 2.00 bpw (IQ2_XXS build). Transformer 72.1 GB (vs 138.2 GB unpruned IQ2_XXS); Engram tables 110.6 GB; mmproj 0.97 GB; IQ4_NL file listed as 111 GB. SWE-bench Verified 81.20 base vs 77.00 (94.8%); LiveCodeBench v6 89.71 vs 88.57 (98.7%), reasoning effort 50. MIT license. [DeepSeek card](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF) — **model card**
- GSQ: arXiv 2604.18556, Dadgarnia, Tabesh, Nikdan, Helcig, Kurtic, Alistarh; post-training Gumbel-Softmax scalar quantization; symmetric group-wise grid compatible with scalar kernels; code stated to be at github.com/IST-DASLab/GSQ. [GSQ abstract](https://arxiv.org/abs/2604.18556) — **abstract** (search summary; full HTML listed but not read)
- RCO: arXiv 2605.00649 (Helcig and Alistarh): expert selection by minimizing KL to the unpruned model on calibration data, one budget per layer. [Qwen3.8 card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) — **model card** (RCO paper itself not read)
- GitHub: the org page lists 97 repos (10 shown). None is named "coder" or "ISTA". Public repos include GPTQ-2D, MatGPTQ, MoESQ, SLQ, qutlass, disaggregated-llama.cpp, llmq. A GSQ repo is stated on the GSQ paper page but was not confirmed on the org listing. [IST-DASLab GitHub](https://github.com/IST-DASLab) — **model card** (org page)
- Earlier "ISTA-DASLab Coder" searches returned only the unrelated ISTA sparse-coding algorithm and IQuest-Coder. Those are not this model. [search summary] — **unverified**

### Inferences
- The "Coder" name means capability-targeted pruning. It does not imply a separate code fine-tune, and the cards do not state this in those words. The card's own calibration description supports it.
- Both cards report only two coding/agentic benchmarks, with retention 94.8-100%. General-task regression is not measured, and the cards say degradation outside the targeted domains is accepted. This matters for a general assistant on Strata.
- Card inconsistencies to note: Qwen3.8 license badge says Apache-2.0 but text says base license applies; HF metadata size 117B vs card 176.9B.
- Strata-relevant: the card's claim of running on 8 GB VRAM via Strata is unverified by the card. Do not repeat it without a test.
- The two cards list different reasoning-effort settings (xhigh for Qwen, 50 for DeepSeek), so the numbers are not comparable across the two models.

### Measurement that settles it on our side
Run the Coder GGUF and the stock build of the same family through Strata on the same day with the same agent prompts: one SWE-bench-style small set (for example 20 tasks from our own fixture set) plus a general-chat regression set. Compare tool-call success and general answer quality. This needs a download, which the owner must approve.

Estimate: download is 58 GB (Qwen Coder) or 111 GB (DeepSeek Coder). **The D drive has about 29 GB free (97% full, per memory), so neither fits without freeing space**, and the root disk is about 16 GB free. Download needs owner approval (file size and source stated). Once on disk, the eval is about 1 day of runs on the night queue.

### Gaps
- Calibration dataset names, sizes and composition not given on either card.
- RCO and GSQ full text not read; GSQ GitHub repo not confirmed on the org listing.
- No independent evaluation of either Coder model was found.
- Full DeepSeek file table not read (elided in fetch).

---

## Summary of read levels
| Source | Read level |
| --- | --- |
| ThinkBrake 2510.00546 | full text |
| OS-Pruner 2607.11089 | full text (HTML) |
| TraceLab 2606.30560 | full text, first 100k of 133k chars |
| ECLoop 2607.28815 | full text (HTML) |
| ISTA-DASLab org, Qwen3.8 and DeepSeek Coder cards | model card |
| GSQ 2604.18556 | abstract |
| 2604.02155, EarlyEval, EET, ESTAR, REFRAIN, Atomix, PASTE, speculative actions, Copilot traces | abstract or search summary (unverified) |
| AsyncFC 2605.15077 | full text via summary (round 1) |
