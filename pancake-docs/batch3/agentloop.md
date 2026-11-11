# Batch 3, agentloop: thinking control, learned thinking stop, Fiddler threshold, agent-loop switches, task gate

Branch `dev/p16/agentloop` (from `dev/phase16-academic` 727eb8a62). Items: A17, A18, A20 (16.3), A19 (16.3), A30 (16.6), A38,
A39, A41, A42, A43 (16.8), A55 (16.13). Nothing here ran the engine or loaded a model; everything is Python / host C++ checked
offline, plus one single-thread CPU microbench (`q2_rows_bench`, nothing else running on the box at the time). Sources were read
as abstract pages and the HTML of the papers (2026-11-11); where only the abstract could be read it says so.

Verdicts in one line each (details below):

| Item | Cited method matches what the code does? | Right for a coding agent? | Code state |
| --- | --- | --- | --- |
| A17 per-effort cap | yes (s1 upper cap + own close) | cap above p99 is safe by construction, worth 0 s on today's log | tool + config exist; waits for 200+ effort-tagged turns |
| A18 effort per task | none of the three papers is an inference-time policy | the A/B itself is fine | spec + harness exist, dry run ok |
| A20 rumination + entropy | 5-gram rate: yes (Thoughtology, math only, no threshold). Entropy as a stop signal: no source | rate close: unvalidated heuristic. Entropy: costs >= ~2 ms/token on the host | log + opt-in close exist; do not build the GPU entropy |
| A19 learned stop | partly: features, labels and threshold rule differ from Thought Calibration | small gain at best (p99 1,061 thinking tokens); no ground-truth labels yet | dump hook compiles; threshold rule fixed here |
| A30 Fiddler threshold | Fiddler's rule is per expert, not per chunk; the engine already has it | expected gain ~0 at 2-3K tokens, not 0.4 s | spec got an `auto` arm |
| A38 counters | paper is observational, signals are agent-specific | as a measurement, yes | log-only, records what the plan says |
| A39 async compaction | design note holds | needs a spare slot we do not have | design only |
| A41 test-before-done | no: ECLoop holds edits until task-specific evidence exists | the generic end-of-turn nudge is closer to the baseline that did not help | built, opt-in |
| A42 externalize | no: RLM keeps the prompt in a REPL and recurses; tested on long inputs | mechanism is a plain prompt shrink; opencode already truncates big outputs to a file | built, opt-in |
| A43 oldString constraint | no: the paper constrains TypeScript types | check + log is the right first step | built (check/log/note), constrained decoding not built |
| A55 task gate | tasks are our own 18, not BFCL / LiveCodeBench | the gate cannot decide anything at 18 tasks x 5 reps | interval option added, see below |

## Tests run

- `python3 -m pytest serve` : 1712 passed, 14 skipped, 0 failed (15 min; the slow ones are `test_stream_ownership`).
- `python3 -m pytest tools/pancake serve/test_agent_loop.py serve/test_rumination.py serve/test_think_probe.py serve/test_budget_by_effort.py serve/test_turn_log.py` : 181 passed after the changes below (tools/pancake alone was 109 before them).
- `think_probe_test` (host C++, CPU-only configure `-DSTRATA_BUILD_TESTS=ON`, no GPU): builds, prints `ok`.
- HIP (gfx1100) compile-only: `src/core/verify.cpp.o` builds (A19's `copy_head_row`). `src/program/generate.cpp` does NOT compile on this branch:
  21 errors, all in the A45 block (`CalibMeasure`, `calib_enabled`, `calib_cache_parse`, ... at lines 2865-3018 and 8278-8329), because
  `generate.cpp` never includes `strata/core/calibrate_ratio.hpp`. Not mine (A45); with `-include strata/core/calibrate_ratio.hpp`
  `-fsyntax-only` passes, so A19's hook (lines 7614-7700, 19403-19440, 20584-20640) has no error. Whoever owns A45 must add the include.
- `task_gate.py --dry-run` for taskgate-effort / -caps / -rumination / -lossy: all ok (18 tasks, every checker passes its reference and fails its wrong answer).
- Offline simulations (scripts in the session scratchpad, not committed): gate power and interval coverage; host cost of the logprobs line.

## A17 - per-effort thinking cap

Source: s1 (2501.19393): budget forcing = at the cap append the end-of-thinking so the model answers; models trained on 1K math/science
examples. "Are LRMs Interruptible?" (2510.11713): an interrupted model leaks reasoning into the answer or panics and answers wrong; up to 60% worse
when the update comes late. Both are math/science, not tool calls. Code = the same mechanism (`reasoning_budget_by_effort` + grace, the
server closes at a sentence end). Match: yes. What the interruptible paper adds: the stress arm in `taskgate-caps.json` (cap 512) is the right test of whether a
forced close breaks a tool call; a cap above the p99 only touches the tail. The log: 878 lines, 34 agent turns (no effort key), thinking p50/p90/p99/max =
208/440/1,061/1,154 tokens on those; `rumination_max` is filled on 1 of the 73 thinking rows. So no cap can be fitted yet, as the plan says.
Caveat to remember: `thinking_caps.py` must be fed a log made WITHOUT caps; with caps on, the logged lengths are censored and the p99 only shrinks.

## A18 - effort per task

Papers: ALP 2506.05256 is an RL objective (DeepScaleR-1.5B, -50% tokens), "Don't Overthink it" 2505.17813 needs k parallel samples (short-m@k), Danger of
Overthinking 2502.08235 is on SWE-bench Verified but its remedy is picking the lowest-overthinking of several runs (+30% performance, -43% compute). None is a
single-sample inference-time effort policy, so "-20-40% thinking" is not a measured gain of any of them. The experiment (arms medium / xhigh / low / off on total
time to success) is still the right way to answer it. The effort reaches the model through the chat template's `reasoning_effort` (and, with `effort_end`, a short effort system turn before the answer, `render_prompt`);
whether the arms really differ in thinking length is what the first run shows (check `reasoning_tokens` per arm before reading the success numbers).
Changed here: `time_curve()` in `task_gate.py` computed its own area under the curve next to A57's `time_auc()` (same number); it now calls `time_auc` and
`success_within` (one definition).

## A20 - rumination and entropy

Thoughtology (2504.07128), read in the HTML: rumination is "the frequency of redundant reconsiderations", measured as the proportion of repeated 5-grams in the
text (our `RuminationMeter` matches, window and rolling form are ours); "some evidence" that it goes with errors, on AIME-24, MATH500, GSM8k with DeepSeek-R1; no
threshold and no stopping rule. So the close threshold is ours and must come from `rumination_max` of real agent turns (none yet). Entropy Sentinel (2601.09001):
a classifier on the entropy profile of a whole response to estimate accuracy per domain; not a thinking-stop signal. No source supports `entropy_below`.
Cost, measured: the engine's `emit_lp_line` copies the whole vocabulary row and runs `exp` in double over it plus a partial sort: 1.84 ms per token on the 5700G
(`g++ -O2`, 248,320 floats, k 8, host part only, D2H copy not included), against ~11 ms per decoded token. Per accepted token the thinking gets ~17% slower
before the copy. Verdict: keep `rumination_entropy` as a measuring arm only (the spec's `entropy-log` arm times it), do not write a GPU version without a signal.

## A19 - learned thinking stop

What Thought Calibration (2505.18404) does (HTML): steps are `\n\n`-delimited paragraphs that contain "wait" or "but"; the feature is the MEAN last-layer representation of the
step's tokens reduced to 256 dimensions by PCA; probes: `consistent` (the answer at this truncation equals the answer at full budget; no correctness labels), `correct` (graded
by a judge) and `novel leaf`; probe outputs averaged over 10 steps; threshold by Learn-then-Test with fixed-sequence testing over a descending grid and binomial tail
p-values; ID up to -60% thinking, OOD (AIME, GPQA, MATH-500) up to -20%, the supervised probe overconfident; models R1-distill Qwen 32B / Llama 70B, QwQ 32B; calibration set
450 (s1K-1.1); "guarantees need calibration data close to the deployment data". "Learning How Hard to Think" (ICLR'25) allocates best-of-k samples or routes a query; it does not
stop a generation. OS-Pruner (2607.11089, abstract only): stops by a utility of accuracy vs length, 20-60%, probe not stated.
Ours vs theirs:
- State: ONE token's `head_mixed_` (n_embd 2560, the vector the LM head reads) at every 64 thinking tokens and after every blank line; theirs is a per-step mean, PCA-reduced.
  Single-token last-layer probes are a common choice, but it is not the paper's feature. The engine cannot average over a step without copying every row.
- Labels: the default is a proxy ("the model is about to close its own thinking", within 128 tokens of `</think>` or past 90%). It is not in the paper, it can save at most ~10%
  of the thinking, and it never sees "the answer is already decided". The paper's `consistent` label needs the forced-stop replay, which is not written ("needs a running server").
- Threshold: the tool took the LOWEST passing threshold of a 50-point sweep, i.e. the luckiest of 50 correlated tests. Fixed here: `pick_threshold()` walks down from the highest
  threshold and stops at the first that fails the Hoeffding bound (fixed-sequence testing, as the paper). Still a simplification: Hoeffding instead of the exact binomial tail, the
  bound is on the per-fire premature rate (the paper bounds the per-example risk), and the held-out requests are both the calibration set and the report set.
- Value here: thinking is ~22% of turn time and p99 1,061 tokens (34 turns): even -20% of it is ~4% of turn time, lossy, and the probe has no calibration data from agent turns.
Code state: `think_probe_test` ok; `verify.cpp.o` and the `generate.cpp` hook compile (see Tests run). Keep `STRATA_HIDDEN_DUMP` (exact, one 10 KB copy per check point) to collect data
if the owner wants it; do not spend more on `STRATA_THINK_PROBE` before the replay driver and agent dumps exist.

## A30 - Fiddler threshold for delta prefill

Fiddler (2402.07033, HTML): per expert and layer, CPU if the CPU latency for the s tokens routed to it is below the GPU latency plus the weight copy; CPU latency linear in s, the copy
constant, both measured at start; Mixtral-8x7B 16-bit (~300 MB per expert), Quadro RTX 6000 on PCIe 3.0 x16 and RTX 6000 Ada on Gen4; long prefill 1.07x-1.65x vs offloading baselines
(1.30x in the abstract). There is no numeric threshold in the paper.
Our numbers: expert blob 1,382,400 B (gate/up 640 rows x 720 B each, down 2,560 rows x 180 B), promotion 115 us on PCIe 3.0 (RESEARCH_AND_FINDINGS), CPU 42 us per expert at one row on
the pool. `q2_rows_bench` (5700G, one thread, legacy kernel, bit-exact vs fast): gate/up 107 ns per row, down 31 ns per row at one token, x2.5 / x2.4 at four tokens, the same in L2 and in
DRAM, i.e. the Q2_0 CPU kernel is compute-bound at ~6 GB/s per thread, not byte-bound (the "cost barely grows with tokens per expert" line in the 16.1 rejected-list is wrong; its conclusion for
decode stands because decode has 1-2 tokens per expert). That is ~220 us per expert-token on one thread, ~113 + 107 s us for s tokens; scaling to the measured 42 us pool time at s = 1
(5.2x, an assumption about the pool) gives t_cpu(s) ~ 21.7 + 20.6 s us. Against 115 us: break-even s ~ 4.5 tokens per expert.
The engine already does this per expert: in `prefill.cpp` only non-resident experts with at most `MAXT` = 8 tokens are CPU candidates, fewest tokens first, `share x streamed` of them
(`auto` measures c and g per layer and gates itself). So the A30 row's "token-count threshold, deltas < 4096" is a chunk limit, not Fiddler's rule. Mean tokens per expert = N x 10 / 512: 20 at 1K, 58 at
3K, so at 3K only the cold tail can qualify while the whole activation matrix (T x 2560 floats = 31 MB per layer, ~126 ms per 3K chunk) goes to the host. Upstream's own numbers
(`prefill.cpp` comment, RTX 5060 Ti PCIe 3.0 x8 + 3060): the share's effect is -20..-35% at 500-1,000 tokens, 4.33 -> 4.22 s at 2,012, unchanged at 3.2-3.5K. The plan's "~0.4 s per 3K delta" assumes the
3K chunk could reach the 32K chunk's 2,580 tok/s; that gap is expert-streaming amortisation over a bigger chunk, not the CPU share. Expect ~0 at 2-3K. Done: spec `a30-fiddler-cpu-share.json` got
arm `auto`; its note carries this. Run it only as the cheap env screen it is.

## A38 / A41 / A42 / A43 / A39 - agent loop

A38 (Beyond Accuracy 2511.11012, HTML): four agents on 404 multi-hunk bugs. Failed repairs use +33% to +440% input tokens and per-bug cost ratios of 1.26x-5.42x (the plan's "1.4-4.4x" is a loose
transcription). The behaviour signals are observational and agent-specific: edits without testing for Gemini-cli, searching without acting for Claude Code, repeated rebuilds for Codex /
Qwen Code; Codex's reads stay at ~5% whatever the outcome; "no universal pass/fail pattern", no intervention recommended. So counters are the right first step and no threshold may come from the paper.
Recorded vs plan: `STRATA_AGENT_COUNTERS=1` writes an `agent` object per request in the turn log (kinds, both streaks, their maxima, `code_edits_since_check`, `files_edited`, `edits_unread`, the A43 states,
`edit_failed`, `last_round_*`, `ends_unchecked`); numbers only; no key without the switch; no object for a request without a tool call in its turn. That is what the plan asks. Per-request A41/A42 events
are only in `/metrics` (`finish_nudges`, `externalized*`), which is enough for an arm-per-server A/B.
A41 (ECLoop 2607.28815, HTML): an execution layer that compiles task conditions from the issue and repository (inspect the callers before editing, reproduce before editing, check which tests exist before
submitting), checks them deterministically against the agent's events and HOLDS the edit or the submission until they hold (released after 3 holds); SWE-bench Verified x {mini-swe-agent, Codex CLI} x
{GPT-5-mini, MiniMax-M2.5}: Pass@1 +4.8..+11.8 pp, tokens -1.4..-12.1%. Ablation: structured conditions replaced by a natural-language summary 58 vs 63 without guidance (47 with nothing);
Self-Refine (post-hoc review) slightly lowered Pass@1. Our nudge is one generic sentence when the model ends its turn after code edits with no build / test / run since: it is the weak form (closest to the
natural-language and post-hoc baselines), not the commitment hold. Read-before-edit: opencode already enforces it (batch-2 read the binary; re-read here), `edits_unread` shows if that ever changes.
Keep it opt-in and judge it with the task gate; expect no transfer of the paper's gain.
A42 (RLM 2512.24601, HTML): the prompt lives in a REPL variable, the root model writes code to peek / grep / chunk and calls sub-models recursively; GPT-5 (+ GPT-5-mini sub-calls) and Qwen3-Coder-480B; tasks CodeQA,
BrowseComp+, OOLONG (inputs up to millions of tokens); "+26% median vs compaction" is on GPT-5 over four long-input tasks; small models struggle as RLMs (Qwen3-8B) and Qwen3-Coder made more syntax errors.
Ours: tool results over N tokens go to a file, 400 + 400 token excerpt plus the path. That is "CodeAct + files" at best, with no recursion and no REPL, on an IQ2 model. As a prompt shrink it is sound (prefill
is the bottleneck) and byte-stable. New fact: the installed opencode (2026-10-28 binary) already truncates big tool output itself ("The tool call succeeded but the output was truncated. Full output saved to: ...")
with a hint to use Grep / Read ranges; I could not read its limits. So A42 only matters below opencode's own limit; set N from the A38 counters, not from the paper.
A43 (Type-Constrained Code Generation 2504.09246, abstract): prefix automata over a type system, TypeScript, HumanEval / MBPP, compile errors cut by more than half. It does not constrain edits to substrings
of existing text. The oldString check is our idea; the paper only supports "constrained decoding removes a class of invalid outputs". The check / log / note is right; constrained decoding stays unbuilt.
Edit-failure texts in the installed opencode checked: "Could not find oldString in the file...", "Found multiple matches for oldString...", "oldString not found in content": all covered by `FAIL_MARKERS`.
A39 (Slipstream 2605.08580, abstract): the compactor runs in parallel with the agent on the pre-compaction context and a judge checks the summary; up to -39.7% latency on SWE-bench Verified and BrowseComp, baseline not
named in the abstract, serving setup not stated. ACE (2510.04618, abstract): an evolving prompt playbook with incremental updates (+10.6% agents, +8.6% finance); it is about prompt content, not KV reuse. The design note's
reasons (one GPU slot, the swap must coincide with the client's own compaction) hold; the paper's gain needs spare capacity we do not have. Stays parked.
Added a test (`serve/test_agent_loop.py`): a file line that itself looks like a line number keeps its digits in all three read-output styles.

## A55 - task gate

BFCL, LiveCodeBench and TheAgentCompany are not in the runner: it grades our own 18 tasks (8 code edits, 7 tool calls, 3 QA) with checkers that pass the reference and fail the wrong answer.
Power, simulated with the runner's own `boot_diff` and `verdict` (task difficulty 55% at p .97, 25% at .75, 20% at .4; 150-200 trials; margin 5 pts):

| tasks x reps | true loss | mean interval width | verdicts (current two-stage bootstrap) |
| --- | --- | --- | --- |
| 18 x 5 | 0 | 25 pts | INCONCLUSIVE 90%, PASS 10% |
| 18 x 5 | 5 pts | 27 | INCONCLUSIVE 99% |
| 18 x 5 | 10 pts | 28 | INCONCLUSIVE 90%, REJECT 10% |
| 18 x 10 | 10 pts | 20 | INCONCLUSIVE 91%, REJECT 9% |
| 60 x 5 | 10 pts | 16 | INCONCLUSIVE 79%, REJECT 21% |

The default interval resamples the tasks and then the repetitions inside each task, which counts the repetition noise twice: coverage of the true difference was 97-100% for a nominal 95%, and the interval
was ~35-40% wider than resampling tasks alone (coverage 91-94%, so that one is slightly liberal at 18 tasks). Added `--tasks-only` (`boot_diff(resample_reps=False)`, report line "over tasks only"); the default
path gives byte-identical numbers (a test keeps the old function as the reference). Even tasks-only, 18 tasks cannot certify a 5-point loss. A gate for A1/A2/A4/A17-A20 needs ~100+ tasks or a loss margin of
~10-15 pts, and the first run should be read as "REJECT or not", never "PASS". `build_report`'s info-only time-AUC (A57) is unchanged.

## Files changed

`tools/pancake/thinking_probe.py` (+ test), `tools/pancake/task_gate.py` (+ test), `bench/specs/a30-fiddler-cpu-share.json`, `serve/test_agent_loop.py`, this note.

## Risk and next

- All changes are offline tooling; engine and server code were not touched; no default path changed (`boot_diff` default checked against the previous implementation in a test).
- Left for the owner: the missing include in `generate.cpp` (A45) blocks any engine build of this branch; the A19 replay driver; agent-turn dumps and effort-tagged turns; a task set of 100+ tasks; a decision on `--tasks-only`
  as the default; the A30 / A18 / A41 screens when the GPU is free.
