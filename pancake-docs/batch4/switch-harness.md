# switch-harness: switch evidence and tier check in promote.py (batch 4, agent `switch-harness`)

Branch `dev/p16b/switch-harness` (worktree `../Strata-p16b-switch-harness`). Python and data only: no C++, no GPU, no engine, no model was
started. Plan: MOD_PLAN.md "Project features discovered" 1 (switch-acted evidence) and 4 (tier-aware runs).

## What was built

| File | What |
| --- | --- |
| `tools/pancake/switch_report.py` (new) | Parses the engine's `strata switch: NAME acted: ...` / `strata switch: NAME will not act: ...` lines (`SWITCH_LINE`, the one place that knows the format; matches `include/strata/core/switch_report.hpp` of `dev/p16b/switch-cpp`) and reports per arm which of its env switches acted. Command line: `switch_report.py bench/results/<day>/<stem>`. |
| `tools/pancake/promote.py` | Two new checks, run before anything else is judged (see below). |
| `bench/promotion/switch-exempt.json` (new) | The exemption list: 111 switches that print no line yet. |
| `tools/pancake/test_switch_report.py` (new) | 25 tests on synthetic logs and jsonl. `test_promote.py` is unchanged: its 36 tests still pass. |

`bench/promotion/phase16-gpu-2026-11-11.json` is unchanged (nothing needed).

## The switch rule

An arm's switches are the `STRATA_*` keys of its env overlay in the spec that are not removed (`null`) or set to an off value
(`0`, empty, `off`, `false`, `no`). Status over the runs promote.py uses (the first clean attempt per repetition, as abtest does):

| Status | Meaning | Verdict |
| --- | --- | --- |
| ACTED | every run logged `acted` | judged as before |
| VOID | no run logged `acted` | HOLD `switch never acted: the arm measured nothing (NAME: will not act: reason / no line in N engine logs)`; the arm's numbers, parity and gates are not judged and a REJECT it would have had is dropped (it measured the base) |
| PARTIAL | some runs only, or some runs without a log | HOLD, numbers still shown |
| NO_LOG | no run has an engine log (result from before abtest kept it) | HOLD `not verified`, numbers still shown |
| EXEMPT | on the exemption list | listed in one line, not checked |

A switch that logs both lines (a reason that held for a while, then it acted) counts as acted. Names are compared without `STRATA_`
and without a `=value` suffix. An item can add its own exemptions with `"switch_exempt": {"reason": ["STRATA_NAME"]}`.

**Exemption list.** `bench/promotion/switch-exempt.json` = every env key of `bench/specs` on 2026-11-11 except the six switches that log
since `switch_report.hpp` (WAIT_COPY256, ROUTE_FAST, ROUTER_FUSE, IC_WARM_MB, EXPDEC_SWIGLU, HIP_GLOBAL_WEIGHTS). Many of the others are tuning
values (`STRATA_ADAPT_KMISS=2,8`) that will never log. Delete a name when its switch starts to log. A new switch in a new spec that logs
nothing yet is not on the list, so its arm is VOID with the hint to add it: that is the point of the rule.

## The tier rule

The jsonl rows carry `tier` (written by abtest). promote.py collects the tiers of every run the comparison uses (the reference arm of the
spec, the `vs` arm and the judged arm): more than one value -> HOLD `tiers differ: base on arena, x on mmap ...` and nothing else is
judged. Rows without a tier (every result so far) are unknown: one `WARNING: <file>: N of M runs carry no tier` at the end of the report
(once per file however many arms and items use it), no hold. A tier given as a dict is compared as JSON.

## Contract with abtest (owner: harness-core): NOT met yet, the lead must check

1. **Where the engine log is.** The per-arm `<arm>-r<rep>-a<attempt>.out` files under the results folder are the *server's narration*
   (`start_server` runs `serve/server.py` with stdout there). The engine's stderr, where the switch lines go, is written to the preset's `log`
   file (`pancake-flashnext.log`, opened in append mode by `serve/server.py`), shared by every start and every arm. Checked on
   `bench/results/2026-11-11-gpu/25-kern-hip-confirm/*.out`: no engine lines in them (not even the `STRATA_ROUTE_FAST=1: ...` lines the spec note
   promises as log proof). So the switch lines are not kept per run today. promote.py reads
   `<results>/<stem>/<arm>-r<rep>-a<attempt>.engine.log` (`switch_report.LOG_SUFFIX`); abtest must write it: in `run_arm`, count the lines of
   `cfg["log"]` before `start_server` and write `splitlines()[n_before:]` after the requests, before `stop_servers()`. Until it does, every arm
   with a non-exempt switch is HOLD `not verified`.
2. **The tier field.** promote.py expects `tier` as a string (`arena` / `mmap`) on every attempt row; absent, `null`, empty and `unknown` all
   mean unknown.

## Tests run

- `python -m pytest -q tools/pancake/test_promote.py tools/pancake/test_switch_report.py`: 61 passed (36 existing + 25 new).
- `python tools/pancake/promote.py bench/promotion/phase16-gpu-2026-11-11.json` on the real results: both items stay HOLD; the new
  lines are `not verified` for the live switches WAIT_COPY256, ROUTE_FAST and EXPDEC_SWIGLU (the results predate the engine log) and the tier warning. `phase16.json`: 35 HOLD
  before and after (the 2 items with a deferral switch print the exempt line; the tier warning appears).
- New tests: good arm, void arm (will not act / silent / REJECT-by-numbers becomes HOLD), partly acting arm, no kept log, `.out` is not an
  engine log, only the attempts abtest used are read, exempt switch (file and item), malformed `switch_exempt`, mixed tiers (including the
  `vs` arm and one run inside an arm), consistent tier, rows without a tier (warned once per file), part of the rows without a tier, the
  read-only promise for `switch_report.py`, the command line.

## Risk and next

- With no `.engine.log` kept, arms of the six live switches are HOLD even when they acted. Fail-safe, but tonight's runs need item 1 above.
- The exemption list is a snapshot; prune it as switch-cpp's successors instrument more switches. A stale name only weakens the check.
- Not done: the same check inside abtest itself (`abtest` could print VOID per arm while it runs; it is harness-core's file).
