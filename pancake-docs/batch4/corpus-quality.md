# Bigger quality sets: token forcing 20 -> 48 requests, task gate 18 -> 150 tasks (batch 4, agent `corpus-quality`)

Branch `dev/p16b/corpus-quality` (worktree `../Strata-p16b-corpus-quality`), base `dev/phase16-academic` 0dd7ef2e8. Python and data only: no
engine, no model, no GPU was started. The opencode database was opened read-only (`mode=ro`) and only numbers were taken from it; no text.

Problem (MOD_PLAN "Test-protocol findings", pancake-docs/batch3/promotion.md): the 18-task gate passes an identical arm ~3% of the time and rejects a
true 10-point loss ~3%; the 20-request token-forcing set is INCONCLUSIVE 22% of the time for an identical arm. No lossy item could reach PROMOTE.

## What was built

| File | What |
| --- | --- |
| `tools/pancake/build_task_set.py` | Repo history + tests -> verifiable coding tasks in task_gate's format. `--count N`, `--check FILE`, `--rev`, `--seed`, `--families`, `--per-file`, `--per-function`. |
| `bench/tasks/repo-history.json` | 150 tasks built at `0dd7ef2e8151` (history 18, bugfix 66, implement 66; 57 files, 98 functions). Order is round-robin over the families: every prefix is a mix. |
| `tools/pancake/task_gate.py` | Three small, default-neutral additions: task field `expect.setup` (definitions run before the answer), `--limit N` (first N tasks), `run_python(tail=)`. |
| `tools/pancake/build_quality_set.py` | The 28 new token-forcing requests (agent-shaped, from the repo's own files at a pinned commit, the model's own chat template and tokenizer). |
| `bench/quality/set-v2.json` | 48 requests: the 20 old ones verbatim (first) + 28 new with `"cont": null` (see "What the night run must do"). 2.1 MB. |
| `tools/pancake/quality_gate.py` | `set_problems / set_summary / pending / merge_conts` (pure), `--fill`, `--dry-run`, a refusal for a set with pending continuations. The `--set` default is unchanged (bench/quality/set.json). |
| `tools/pancake/gate_power.py` | Monte Carlo of both gates with their own code (`force_check.check`, `task_gate.boot_diff + verdict`). |
| tests | `test_build_task_set.py` (12), `test_quality_gate.py` (11), `test_gate_power.py` (5). |

`tools/pancake/power_sim.py` (harness-sim, commit 1216cf60f on `dev/p16b/harness-sim`) is the speed rule of promote.py on abtest's noise; it does not cover
the two quality gates, so `gate_power.py` is separate and uses the gates' real functions.

## Source check

Nothing here changes a statistic: the rate rule (+0.5 points at 95%), the event rule, the paired bootstrap and the margin are those of `force_check.py`
(SpecPilot, arXiv 2611.02501) and `task_gate.py`. The numbers below re-derive the promotion agent's power findings with the same code and add two things
the old simulation did not have: the real noise structure of the stock arms and a task mix picked for information.

## Real session shapes (numbers only; 2,036 requests of qwen3.8-flash-next-iq2_xs in 25 sessions)

Prompt tokens per request p10/p25/p50/p75/p90/p99 = 33.8K / 45.1K / 64.5K / 81.6K / 93.0K / 107K. First request of a session p25/p50/p75 = 11.4K / 19.1K /
37.9K (the system + tools prefix is ~11K). Growth between two requests of a session: median 525 tokens, p90 2.3K, p99 16K. Reply tokens p10/p50/p90/p99 =
75 / 115 / 229 / 1,025; by step: bash median 256 (p90 914), edit or write 336 (p90 1,379), read 157, text-only 556, todowrite 737. Steps by tool: bash 57%,
caveman_retrieve 13%, edit/write 11%, read 7%, text-only 6%. An edit's `oldString` median 137 chars, `newString` 197; a `write` 3,054 chars (~1,000 tokens).
The old set has prompts of 655-1,486 tokens: two orders of magnitude below the real median, and none crosses `--kv-resident 32768`.

## Token-forcing set (`bench/quality/set-v2.json`)

| kind | n | prompt tokens (min / median / max) | shape |
| --- | --- | --- | --- |
| agent, code, docs, english, multilingual | 20 | 655 / 1,085 / 1,486 | the old set, verbatim, same ids, same order |
| edit-short | 6 | 321 / 530 / 766 | one Python function + "rename X" / "add an early return", no tools, copy-heavy (like abtest's `edit`) |
| first | 4 | 10,560 / 10,562 / 10,570 | system (agent prompt + the repo's AGENTS.md / CONTRIBUTING.md / docs, cut to 10.5K) + 9 tools + a first request, thinking on |
| toolres | 6 | 11.5K / 13.3K / 16.9K | ... + assistant `read` call + the file as the tool result; thinking on x3, off x3 |
| edit-ctx | 6 | 13.3K / 14.4K / 16.3K | ... + "rename X in <file>" with the file read: the reply is an edit call (copy-heavy), thinking off |
| deep32k | 4 | 27.3K / 30.5K / 33.1K | a session of 5-10 read / git grep / git log exchanges, then the next step |
| deep64k | 2 | 64.1K / 64.1K / 64.2K | the same, 19-21 exchanges (crosses `--kv-resident 32768`) |

Built from the repository's own files at 0dd7ef2e8151 (`git show`, `git grep`, `git log`), never from the history's text. Rendered with the pack's
`chat_template.jinja` (Qwen3.x tool format `<function=..><parameter=..>`) and tokenized with its tokenizer; `--check` verifies ids < 248,320, the old 20 verbatim,
unique ids. The new items carry `meta` (shape, thinking, files, rev, prompt_tokens). Prompt tokens in all: 486,974 (466,342 new).

**The continuations are not here.** A token-forcing continuation is the stock engine's own greedy output (`force_check.py gen`); human text or the history's
sampled replies would make the mismatch statistic measure something else (the old arms flip ~1.2% of positions only because the forced text IS the argmax).
That needs the GPU, so the 28 new requests have `"cont": null`. `quality_gate.py` refuses such a set unless `--fill` is given (and `--dry-run` says so).

### What the night run must do (one engine start, ~10 minutes, then the normal gate)

```sh
python tools/pancake/quality_gate.py bench/specs/<any quality spec>.json --out bench/quality-hip/<day>-fill --set bench/quality/set-v2.json --fill
# gen with the first (stock) arm -> <out>/set.filled.json, then scores every arm with it.  Afterwards:
cp bench/quality-hip/<day>-fill/set.filled.json bench/quality/set-v2.json && git add bench/quality/set-v2.json
# and in tools/pancake/quality_gate.py change the `--set` default to "bench/quality/set-v2.json" (test_phase.sh passes no --set;
# keep it a plain string literal: spec_lint.py reads that line)
```
Freeze the filled file: every later run must use the same tokens. Old results stay comparable on the 20 common requests (`force_check.py check` pairs the
requests present in both arms). Checked offline: `quality_gate.py SPEC --set bench/quality/set-v2.json --dry-run` fails with "28 requests have no
continuation", with `--fill --dry-run` it prints "5 engine starts ... gen of 28 continuations" and "dry run: ok"; the old set still passes. Estimate printed:
~5 minutes of scoring per arm (prefill line of percache_sim + ~100 tok/s decode, an estimate), +1 start for the fill.

## Task set (`bench/tasks/repo-history.json`)

A task is a pure top-level Python function of this repo (stdlib only) that the repo's own tests call with literal arguments. The checker replays those calls
against what the function at the pinned commit returns for them (recorded twice in separate processes, so hash-randomised or clock-dependent results are
dropped; floats compared with 1e-9; the result must be a literal or a builtin exception). The module-level definitions the function needs are the task's
`expect.setup`. Three families, three broken starting points, 1-3 cases shown and the rest hidden:

| family | n | the broken start | the instruction |
| --- | --- | --- | --- |
| history | 18 | the function as it was before its last change(s) in `git log` (diff <= 30 lines, 15 tasks with one commit, 3 with two) | the commit subjects + 2 failing calls |
| bugfix | 66 | one AST mutation the cases detect: negated condition 14, comparison 11, off-by-one 10, operator 10, and/or 10, call swap 5, flipped boolean 4, other 2 | 35 with one failing call, 31 (documented functions) with only the docstring |
| implement | 66 | signature + docstring, body `raise NotImplementedError` | 3 expected results |

Funnel (`functions` is every top-level function with at least one literal test call): 487 functions -> 174 eligible (undecorated, 4-45 lines, no I/O or
global state) and >= 2 cases -> 123 with a usable context (stdlib imports, constants, helpers) -> 109 with >= 2 stable recorded cases -> 291 candidate tasks
(bugfix 195, history 18, implement 78) -> **291 validated** (the reference passes and the broken answer fails the real checker in task_gate's sandbox:
`task_gate.selftest`) -> 150 emitted after the caps (<= 2 tasks per function, <= 8 per file). Why history is only 18: of the 109 functions with recorded
cases, 29 have an older version within 300 commits of their file, and for 18 of those the older version gets a recorded case wrong (the other 11 changed only
what the cases do not test); most functions were written once.

Self-tests: `build_task_set.py --check bench/tasks/repo-history.json` -> 150 tasks, 0 problems; every reference is the function `0dd7ef2e8151` holds (AST equal);
the checker also passes on the **whole original module** at that commit for 79 tasks, 71 are skipped because the module does not run in the sandbox (non-stdlib
or relative import, module-level work), 0 fail. `task_gate.py bench/specs/taskgate-lossy.json --tasks bench/tasks/repo-history.json --limit 110 --dry-run` ->
"110 tasks, code_edit 110 ... dry run: ok". The build is deterministic: two runs gave byte-identical files (38-45 s, 4 jobs, 6 GB cap).

Run the gate on it (the task-gate specs have a `"tasks"` key; or on the command line):

```sh
python tools/pancake/task_gate.py bench/specs/taskgate-lossy.json --tasks bench/tasks/repo-history.json --out bench/results/<day>-taskgate --reps 5 [--tasks-only]
# cost: tasks x reps requests per server start (150 x 3 = 450, ~5-8 s each); --limit 100 for the first 100 (a mix of the three families)
```

## Power (gate_power.py, the gates' own code)

### Token forcing (`python tools/pancake/gate_power.py tokens`, 400 trials per cell, `force_check.check`)

Share PASS / INCONCLUSIVE / REJECT in %; the arm adds +x points of mismatches to stock; the old set uses its real 20 request lengths (5,322 tokens), the grown set
adds 28 requests of 240 scored tokens (12,042 tokens; the real figure is known after the fill run). Two noise models: **ties**, fitted to the real stock arms
(`stock-a.tsv` / `stock-b.tsv`: identical config, 62 and 69 of 5,322 positions mismatch, 36 of them the same position where independent arms would share 0.8,
so the flips are coin tosses at near-tie positions; request counts are over-dispersed, variance 11.8 against 3.1 for a binomial), and **iid**, the model of
promotion.md (every position independent) at the real 1.2% rate.

| set | model | identical to stock | +0.5 pts | +1.0 pts | +1.5 pts |
| --- | --- | --- | --- | --- | --- |
| 20 requests | ties | 98 / 2 / 0 | 10 / 87 / 3 | 0 / 33 / 67 | 0 / 3 / 97 |
| 48 requests | ties | 100 / 0 / 0 | 10 / 86 / 4 | 0 / 4 / 96 | 0 / 0 / 100 |
| 20 requests | iid | 86 / 14 / 0 | 9 / 87 / 4 | 0 / 26 / 74 | 0 / 0 / 100 |
| 48 requests | iid | 99 / 1 / 0 | 4 / 90 / 6 | 0 / 6 / 94 | 0 / 0 / 100 |

The promotion note (identical arm INCONCLUSIVE 22%, +1.0 rejected 62% at 20 requests; ~92% at 40) is reproduced within the difference between the models (iid at
1.2%: 14% and 74%). With the real noise structure an identical arm is not the problem (98% PASS at 20 requests: its flips are the same positions as stock's); a
real +1.0 point deviation is: rejected 67% of the time at 20 requests, **94-96% at 48**. +0.5 points can never be settled: the rule's margin IS +0.5 (PASS needs the
upper bound <= 0.5, REJECT the lower bound > 0.5), so it stays INCONCLUSIVE ~87% whatever the size. Real data agree: tailskip7 (+0.84 [0.44, 1.24]) was INCONCLUSIVE.

### Task gate (`python tools/pancake/gate_power.py tasks`, 100 trials per cell, `task_gate.boot_diff + verdict`, margin 5 points, default interval)

Every arm run draws every task independently from its pass rate (a lossy arm leaves the sampled path at its first different token, so same-seed pairing does not
help in the case that matters); a true loss of d points scales every rate by 1 - d / mean rate. Share PASS(+BETTER) / INCONCLUSIVE / REJECT in %. Mix A =
promotion.md's (56% of the tasks at 95% pass rate, 17% at 80%, 17% at 50%, 11% at 10%); mix B = 20% at 95%, 20% at 80%, 40% at 50%, 20% at 20%.

| tasks x reps | mix | identical | loss 5 pts | loss 10 pts |
| --- | --- | --- | --- | --- |
| 18 x 5 (today) | A | 9 / 91 / 0 | 1 / 98 / 1 | 0 / 93 / 7 |
| 18 x 5 | B | 2 / 98 / 0 | 3 / 97 / 0 | 0 / 98 / 2 |
| 100 x 5 | A | 39 / 61 / 0 | 0 / 100 / 0 | 0 / 60 / 40 |
| 100 x 5 | B | 28 / 72 / 0 | 1 / 98 / 1 | 0 / 82 / 18 |
| 100 x 10 | A | 82 / 18 / 0 | 2 / 98 / 0 | 0 / 43 / 57 |
| 100 x 10 | B | 47 / 53 / 0 | 1 / 99 / 0 | 0 / 49 / 51 |
| 150 x 3 | A | 49 / 51 / 0 | 1 / 99 / 0 | 0 / 70 / 30 |
| 150 x 3 | B | 26 / 74 / 0 | 1 / 98 / 1 | 0 / 78 / 22 |
| 150 x 5 | A | 64 / 36 / 0 | 0 / 100 / 0 | 0 / 47 / 53 |
| 150 x 5 | B | 41 / 59 / 0 | 0 / 100 / 0 | 0 / 64 / 36 |
| 150 x 10 | A | 90 / 10 / 0 | 0 / 99 / 1 | 0 / 20 / 80 |
| 150 x 10 | B | 78 / 22 / 0 | 0 / 99 / 1 | 0 / 35 / 65 |

(The whole grid, 18/100/150 tasks x 3/5/10 reps, is `gate_power.py tasks`.) Findings:
- Going from 18 to 150 tasks lifts "identical arm passes" from 2-9% to 41-64% at 5 reps and to 78-90% at 10 reps, and "true 10-point loss rejected" from 0-7% to 36-53%
  at 5 reps and 65-80% at 10 reps. **The grown set helps a lot but is still not decisive at 3-5 repetitions**: the PASS needs the 95% lower bound above -5 points.
- A 5-point loss (the margin) is never rejected, by construction: REJECT needs the upper bound below -5 points.
- **Picking mid-difficulty tasks does not buy power** with this test: mix B (more tasks the base solves only sometimes) is *worse* than mix A for an identical arm,
  because a task near 50% adds the most variance to the paired difference, and about equal for a true loss. What buys power is tasks and repetitions (and the
  interval, below); saturated tasks are harmless noise-wise but dilute a loss.
- Cost: 150 tasks x 5 reps = 750 requests per server start (~1 h at ~5 s each, an estimate: 0.3-1.5K-token prompts, 100-400-token replies at ~100 tok/s), x4 starts (base
  twice, two arms) = 4 h; 10 reps doubles it.

### The same with `task_gate.py --tasks-only` (`gate_power.py tasks --tasks-only`; the interval from resampling the tasks alone, which task_gate's own notes call the
one-stage interval: the default resamples the repetitions too and counts their noise twice, ~35% wider)

| tasks x reps | mix | identical | loss 5 pts | loss 10 pts |
| --- | --- | --- | --- | --- |
| 18 x 5 (today) | A | 19 / 81 / 0 | 5 / 90 / 5 | 0 / 81 / 19 |
| 100 x 5 | A | 72 / 28 / 0 | 3 / 95 / 2 | 0 / 33 / 67 |
| 100 x 5 | B | 56 / 44 / 0 | 4 / 93 / 3 | 0 / 52 / 48 |
| 100 x 10 | A | 94 / 6 / 0 | 3 / 97 / 0 | 0 / 13 / 87 |
| 100 x 10 | B | 77 / 23 / 0 | 4 / 93 / 3 | 0 / 29 / 71 |
| 150 x 5 | A | 86 / 14 / 0 | 0 / 100 / 0 | 0 / 23 / 77 |
| 150 x 5 | B | 64 / 36 / 0 | 4 / 95 / 1 | 0 / 33 / 67 |
| 150 x 10 | A | 100 / 0 / 0 | 3 / 91 / 6 | 0 / 4 / 96 |
| 150 x 10 | B | 89 / 11 / 0 | 2 / 96 / 2 | 0 / 11 / 89 |

With the one-stage interval 150 tasks x 5 reps passes an identical arm 64-86% of the time and rejects a true 10-point loss 67-77%; at 10 reps 89-100% and 89-96%.
That is the cheapest way to a decisive gate (no extra requests). Caveat: a percentile bootstrap of few tasks is slightly liberal at the margin (a loss equal to the
5-point margin is rejected 0-6% here at 100-150 tasks, 0-9% at 18, against 2.5% nominal). The default stays as it is; whether the lossy items use `--tasks-only` is
the owner's call (promote.py reads the verdict either way).

## Risks and limits

- **Difficulty is unknown.** No model ran. How the task gate behaves depends on how many tasks the base solves only sometimes; mixes A and B above bracket it and
  differ by less than the effect of reps. The bugfix tasks with a shown example are probably easy for the served model, the history and implement ones harder. A task
  the base always or never solves adds no noise but costs requests; dropping such tasks saves time and does not raise the power (see Findings), and it must be decided
  from a **separate** calibration run, never from the runs being judged (selection on them biases the gate). `task_gate.py --from-runs DIR` prints the per-task rates.
- **Tasks on one function share their checker** (98 functions behind 150 tasks) and `strata_control` supplies 64 of them: the effective number of independent
  units is nearer 100 than 150; the three families per function are capped at two.
- Python only, stdlib-only functions: it does not exercise tool calls (keep `bench/tasks/default.json` for tool-call validity) or C++.
- The new token-forcing requests are synthetic (shape, not content, of the real sessions). Thinking-on requests score reasoning text, thinking-off ones
  tool calls and code; the mismatch rate of the new kinds is unknown until the fill run (the power numbers assume 240 scored tokens each, the old set's real
  lengths for the 20). A 64K request prefills for ~25 s (percache_sim's line), 47 requests are ~5 minutes of prefill per arm.
- The chat template is rendered with jinja2 and the HF-style `tojson` (no HTML escaping); the server's renderer may differ in whitespace. That cannot bias a
  comparison between arms (same ids for every arm) but the ids are not guaranteed to equal what the server would send.
- The ties model is fitted to one pair of real stock arms (62 and 69 mismatches of 5,322): treat the identical-arm column as an estimate, not a guarantee.

## Next

1. Night: the fill run above, commit `set.filled.json` as `set-v2.json`, change the `--set` default in quality_gate.py.
2. Night: one base-only task run (3 reps) on `repo-history.json` to measure the per-task rates and the real request time (the cost estimate above is a guess), then
   choose `--limit` and `--reps` from the power table (10 reps are what makes the gate decisive).
3. Point the lossy specs (`bench/specs/taskgate-*.json`, presets-specs' files) at `bench/tasks/repo-history.json`; `promote.py` needs no change (it reads the
   same `summary.json`).
4. A C++ / tool-call family would need a compiler or schema checker in the sandbox; not started.
