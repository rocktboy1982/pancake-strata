# A17 / A18 / A20 - thinking control (batch 2, branch work/a17-thinking)

Written 2026-11-10 night. Everything here is opt-in and default off; the default request path, the default turn log and
the default task-gate report are unchanged. Lossy parts are marked LOSSY.

## What was already there (found first, reused, not duplicated)

- `reasoning_budget_tokens` (#123) with the soft close `reasoning_budget_grace` (Maya #613, 512 in the presets): the cap
  closes the thinking at the next sentence or line end, at most `grace` tokens late, with the server's wrap-up and the
  model's own end-of-thinking (`serve/server.py`, `Service.reasoning_budget`, the `wrap` path in `_run`).
- `reasoning_budget_by_effort` (commit 5dd620fb9, `serve/test_budget_by_effort.py`): the per-effort cap table. This IS the
  A17 mechanism ("graceful close at the cap"); no second one was written.
- The turn log's `effort` field (f1df90c5a) and `rumination_max` / `rumination_mean` / `rumination_closed` per request, the
  opt-in `rumination_close` (item 34, `serve/rumination.py`, `serve/test_rumination.py`).
- `tools/pancake/turn_thinking.py` (A17 analysis of one log, no effort split) and `tools/pancake/task_gate.py` (A55).

## What was added

### A17 - `tools/pancake/thinking_caps.py` (new, read-only) + `test_thinking_caps.py`

p99 of the thinking tokens per effort level from the turn log; suggested cap = `ceil(p99 * 1.25 / 128) * 128`; what the cap
would have cut in the log you give it (turns hit, tokens, seconds at the turn's own decode speed); a level with fewer than
200 thinking turns is marked PROVISIONAL and gets no config line; ends with the line to paste:

    "reasoning_budget_by_effort": {"low": ..., "medium": ..., "high": ...}, "reasoning_budget_grace": 512

(`high`, `xhigh`, `max` are one template level: list `high` once.) It also prints, per level, the p50/p90/p99 of
`rumination_max` and the p01/p10/p50 of `entropy_mean`, which is where the A20 thresholds come from.
Run: `python tools/pancake/thinking_caps.py [pancake-turns.jsonl] [--min-turns 200] [--margin 1.25] [--json]`.
Today's log still has no effort-tagged agent turns in numbers that matter (34 untagged ones), so it prints "no thinking turn
carries an effort level yet". Applying a cap is LOSSY for a turn that thinks past it (tail only: the cap sits above the p99).

### A18 - effort per task: `bench/specs/taskgate-effort.json` + task-gate support

`task_gate.py` got, all opt-in (a spec without them runs and reports exactly as before):
- per-arm `"request": {"thinking", "reasoning_effort", "max_tokens", "temperature"}`: an arm per effort level on the same
  preset and tasks (the server is restarted per arm anyway; the request carries `reasoning_effort`);
- per-arm `"config": {...}` (in `abtest.make_config`, also usable by abtest specs): top-level server-config keys, `null`
  removes one. This is how an arm turns on `reasoning_budget_by_effort`, `rumination_close`, `rumination_entropy`;
- spec `"time_curve": true`: per arm the share of runs that succeeded within 10/30/60/120/300 s, the area under that curve
  up to `time_horizon_s` (120; mean of `max(0, 1 - t/T)` over successes, 0 for failures: the SWE-Effi idea, A57) and
  tokens per run, next to the existing `t_success median` and `s/success` (all request seconds / successes).
Arms of the spec: medium (base), xhigh, low, off. The question is total time to success, not tokens: an xhigh arm that
succeeds more but costs more seconds per success loses on this report, and the report says so. The default 18 tasks are
easy (the base passes most of them); if medium and xhigh tie on success the answer is "medium is cheaper", which is a
valid result, but a harder set (BFCL multi-turn, own-repo checkpoints; A55 follow-up) is what separates them.

### A20 - entropy as the second signal (the 5-gram log and close already existed)

`serve/rumination.py`: `EntropyMeter`, `lp_entropy` (entropy in nats of the top-K list of an engine `LP` line, renormalised
over the K ids), `entropy_setting`; `close_setting` takes an optional `entropy_below`.
`serve/server.py` (Service): config `"rumination_entropy": K` (2..20) makes the server add `logprobs=K` to the engine
request of a thinking request and read the engine's `LP` lines (`engine.lp_lines`) for the thinking tokens (parser state
`reasoning`). Per request, only when on: `entropy_mean`, `entropy_min_window` (lowest 64-token rolling mean) in the
history and the turn log, `thinking_entropy` {mean, min_window, tokens} in the final `done` event, `rumination_entropy` on the
status page switches. With `"rumination_close": {"threshold": .., "entropy_below": H}` the close needs the 5-gram rate high
AND the rolling mean entropy at or below H (a model that repeats itself while sure of itself, not one that repeats while
unsure); a request that cannot be measured never closes on this rule. LOSSY (the close), opt-in; `entropy_below` without
`rumination_entropy` is refused at start.
Limits, by design: only Strata's own engine, one request at a time (parallel batch slots carry no LP lines, the split engine
has none), and not for a request that asked for logprobs itself (the lines are one stream). The 5-gram rate is per request,
not per thinking segment: a reply has one thinking block, so per request is per segment in practice; a continued pass
(budget wrap-up, loop recovery) keeps the same meters. The parser state at the time a line is read lags the token by the
loop guard's held tokens (a few), which only blurs the first and last thinking tokens.
COST WARNING: the engine's logprobs path (`emit_lp_line`) copies the whole vocabulary's logits to the host, exponentiates them
on the CPU and partially sorts them, for every token. That is about a megabyte copy plus a few ms of CPU per token against
about 11 ms per token of decode, so "cheaply available" is NOT true on this box: it is unmeasured and must be measured (the
spec's `entropy-log` arm does exactly that: no close, only the speed). A cheap version would compute the top-K entropy on the
GPU inside the engine; not written (C++/HIP, no builds tonight).

Specs: `taskgate-caps.json` (A17: no cap vs a 512 stress cap vs a p99 cap), `taskgate-rumination.json` (A20: base, entropy
log only, rate close, rate + entropy close). Thresholds in it are placeholders until real turns give the p99 of
`rumination_max` and the p10 of `entropy_mean`.

## Files

serve/rumination.py, serve/server.py, serve/test_rumination.py, tools/pancake/thinking_caps.py,
tools/pancake/test_thinking_caps.py, tools/pancake/task_gate.py, tools/pancake/abtest.py (make_config: arm `config`),
tools/pancake/test_task_gate_effort.py, bench/specs/taskgate-{effort,caps,rumination}.json, this note.

## Tests run

NOT RUN. `wait_quiet.sh` returned 1 three times (a benchmark, `abtest.py bench/specs/09-prefill.json`, ran from 23:11 to past
23:44), so by the rules no test was executed. Done instead: `python -m py_compile` on every touched Python file (all OK, parse
only) and a line-by-line trace of every new test against the code. The coordinator should run, at 02:00:

    nice -n 19 python -m pytest tools/pancake/test_thinking_caps.py tools/pancake/test_task_gate_effort.py tools/pancake/test_task_gate.py -q
    nice -n 19 python -m unittest serve.test_rumination serve.test_budget_by_effort serve.test_turn_log -v
    python tools/pancake/task_gate.py bench/specs/taskgate-effort.json --dry-run     (also -caps, -rumination)

## Spec to run (needs the GPU, not run tonight)

`python tools/pancake/task_gate.py bench/specs/taskgate-effort.json --out bench/results/<day>-taskgate-effort` (4 arms x 18
tasks x 5 reps with thinking: expect an hour or more; `--reps 3` for a first look). Then `taskgate-caps.json` and
`taskgate-rumination.json`. `--dry-run` validates a spec and runs every checker without a server.

## Expected gain and risk

- A17: 0 s per turn on the current log (p99 caps cut nothing, A6_A17.md); the value is tail protection (a runaway 8K think
  at ~90 tok/s is ~90 s, capped at 3072 + 512 about 40 s). Risk: a cap cuts a legitimate long think (tail only by
  construction); mitigated by the margin and the grace.
- A18: tens of seconds per turn if medium matches xhigh on the tasks that matter (-20..40% thinking, estimate from the
  papers); counter-evidence in the Qwen3 report that more thinking helps on hard tasks. The spec decides it on our tasks.
- A20: unmeasured. Risk: a false close cuts good thinking (the rate threshold has no paper value; two signals lower that
  risk, the entropy readout costs decode speed).

## What is left

- Run the three specs; collect 200+ effort-tagged agent turns, then `thinking_caps.py` and apply the caps it prints.
- Measure the per-token cost of `rumination_entropy` (the `entropy-log` arm); if it is large, compute the top-K entropy in the
  engine on the GPU.
- A harder task set for the A18 comparison.
- MOD_PLAN rows (coordinator): A17 = tool + spec written; A18 = spec + harness support written, not run; A20 = entropy log
  and the entropy-gated close coded, not run.
