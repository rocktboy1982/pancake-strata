# Batch 5, promote-block: promote.py reads robust and block abtest files

Branch `dev/p16c/promote-block` (from `dev/phase16-academic` 712aea988). Files: `tools/pancake/promote.py`,
`tools/pancake/test_promote.py`, new `tools/pancake/block_reading_sim.py` and `tools/pancake/test_block_reading_sim.py`.
`switch_report.py` is unchanged (see Risks). No GPU, no engine, no abtest run. What harness-core asked of promote.py is in
`pancake-docs/batch4/harness-core.md`, "For other agents / the lead".

## What changed (default files are read exactly as before)

- **Reader.** `read_records` reads the jsonl (an attempt that appears twice still gives the "appears twice" problem; the later one
  counts). `select_runs` calls `abtest.select(records)`, so promote.py uses the attempts abtest's own summary uses, in every mode. For an
  `--outlier robust` file "first clean, else quietest" was wrong: there `clean` only records the proxies. `read_runs` keeps its
  `(runs, attempts not used, problems)` result, so `--noise`, `power_sim.py` and `test_abtest_e2e.py` are unaffected.
- **Robust files** are not held for foreign CPU or reclaim noise. The report says what the rule left out, per arm of the comparison:
  `abtest   outlier rule robust, left out: base slow starts 0, outlier passes 0; x slow starts 1, outlier passes 0`. Gate files keep the
  noisy-run HOLD and the line "N noisy attempts were discarded and repeated".
- **Options.** The arms of a comparison must come from runs with the same `block`, `passes`, `warmup` and `outlier` (the `opts` of the
  jsonl rows). Different -> HOLD "abtest options differ between the runs compared (block: base 0, x 3; ...)", nothing else judged (the
  same treatment as mixed tiers). A row without `opts` is the old default run (block 0, passes 1, warmup 0, gate), so old files, and a
  default run written with opts, mix freely. `test_tier` and `per_run` are not compared: a screen has one start per arm and is not a
  promote input.
- **Tier.** Unchanged, it already did what was asked: known tiers that differ -> HOLD (also one run inside an arm); a missing or
  `"unknown"` tier counts as unknown, one warning per file, no check (tests: `test_switch_report.py`, "tier rule").
- **Block files.** A start is one pair: the mean of its passes (`case_value` already averaged rows), paired by round, t with
  starts - 1 degrees of freedom, MIN_PAIRS (3) starts per arm. Fewer -> HOLD "fewer than 3 clean pairs (engine starts of 3 passes) on
  every claim case (edit n=2); run with reps >= 3 x 3". A "gain open" HOLD in a block file counts starts and gives the reps
  (`about 16 pairs (engine starts of 3 passes) would settle it ...; run with reps >= 16 x 3`). `--noise` names the unit of a block file.
  `--block 3 --reps 6` runs 2 starts per arm, so it can only be read through abtest's own summary, never promoted; `--reps 9` is the
  smallest promotable block run at N = 3.

## Decision: the start-level reading stays (power_sim.py)

The question was whether abtest's own pass-level reading (every pass a pair, interval widened by sqrt(1 + (N - 1) x BLOCK_ICC)) keeps
the false PROMOTE of a true 0% item at or below 1%. `tools/pancake/block_reading_sim.py` runs promote.py's rule on three readings of the
same simulated data (power_sim.py's noise fitted on `bench/results/2026-11-11-gpu/*.jsonl`, the `--outlier robust` procedure: slow starts
run again, outlier passes dropped; 20,000 trials per cell, seed 1; k = 3 / 4 starts x N = 3 / 4 passes; ICC 0 / 0.3 / 0.6 / 0.9; slow
starts as fitted / none; 64 cells). `python3 tools/pancake/block_reading_sim.py` (11 s) prints every cell.

| exact item (need 1%) | start (promote.py) | pass (abtest, design effect) | pass, t over k - 1 df |
| --- | --- | --- | --- |
| false PROMOTE at a true 0%, worst cell | 0.9% | 7.2% (edit, ICC 0.9, k 3 x 4) | 2.0% |
| cells above 1% | 0 of 64 | 35 of 64 | 8 of 64 |
| P(PROMOTE) at a true 3%, mean of the cells | 28% (short 33 / 43, edit 16 / 22, slow starts as fitted / none) | 51% | 21% |
| lossy item (need 3%): worst false PROMOTE / cells above 1% / power at a true 5% | 0.1% / 0 / 28% | 2.1% / 14 / 51% | 1.1% / 1 / 20% |

Worst false PROMOTE by ICC (slow starts as fitted / none), `pass` reading: short 1.6 / 0.0 (ICC 0), 1.7 / 0.1 (0.3), 2.4 / 0.5 (0.6),
3.7 / 1.7 (0.9); edit 2.2 / 0.0, 2.9 / 0.3, 4.3 / 1.4, 7.2 / 4.0; `start` reading: 0.1-0.4 (short), 0.3-0.9 (edit) at every ICC.

Reading: abtest's pass-level reading is wider than a naive one but still over-promotes: because of slow starts that survive
the rerun (short, ICC 0: 1.6% with them, 0.0% without) and, at high ICC, because 0.45 is only an assumed share (the fit says 0.61
for `short`, 0 for `edit`; harness-sim's own table shows the same: 4.0% at ICC 0.9 on edit). Its extra power is the same thing seen from the other side. The one clustered variant tried
(the same pass-level sd and widening, but t over the k starts) is below the start reading in power at ICC <= 0.3, above it at ICC 0.9
(P at a true 3%, short: 37 / 43 against 23 / 29), and its false PROMOTE then reaches 2.0% (8 cells above 1%), so it fails the 1% test too.
The start-level reading is the only one that holds the 1% bound in every cell, so promote.py keeps it and asks for 3 x N reps. Not shown here: that the start-level interval covers at its nominal 95% (only the false-PROMOTE side was
simulated), and any ICC measurement from real passes (needs a block run).

## Replay check on the real files (old promote.py against this one)

`replay_items.json` (scratch): every abtest jsonl of `bench/results/2026-11-09 .. 2026-11-11` that has a spec in `bench/specs`
(19 files, all old-mode rows without `opts`), one item per non-reference arm, as an exact and as a lossy item: 114 items.
Old = `dev/phase16-academic` promote.py, new = this one, same `--root`. The two outputs are **identical** (1,252 lines; 0 PROMOTE, 95 HOLD,
19 REJECT). Also identical: `bench/promotion/phase16.json` (35 items), `bench/promotion/phase16-gpu-2026-11-11.json` (2 items), and
`--noise` on the 2026-11-11-gpu and 2026-11-10-test-phase files. Nothing changes for old-mode rows; the only verdict changes are the
ones above, for robust, block and mixed-option files, which no real file is yet.

What-if on `25-kern-hip-confirm.jsonl` (27 attempts), copied with `opts.outlier = "robust"` on every record, so promote.py reads the
recorded attempts as a robust run (a real robust run would have run the all3 rep 0 slot again, n = 6): the file is picked by
`abtest.select`, 3 slow starts of all3 are left out and none for base or expdec-swiglu, nothing is held for noise. all3 short **+3.1%
[+1.7, +4.4] n=5** (Bonferroni over the 2 claim cases), edit +0.9% [-4.1, +5.9]; expdec-swiglu short +0.1% [-1.3, +1.6], edit +2.5%
[-2.3, +7.3] (HOLD, about 35 pairs). It matches harness-core's replay (all3 short +3.1% +-1.1). The gain test for all3 is then
"shown"; its HOLDs would be the two unrelated ones: the base text hash differs from run to run (parity not checkable) and the
engine log is missing (switch not verified). This is a replay of recorded attempts, not a measurement: rerun before acting on it.

## Tests

- `python3 -I -m pytest -q tools/pancake/test_promote.py tools/pancake/test_switch_report.py tools/pancake/test_abtest_core.py
  tools/pancake/test_abtest_e2e.py tools/pancake/test_power_sim.py tools/pancake/test_block_reading_sim.py`: **130 passed, 17 skipped**.
  The 17 skips are all of `test_abtest_e2e.py`: its own guard skips while a real engine server runs (pid 596375, `strata`
  `pancake-flashnext-iq2_xs-fast-hip.json`, the night session's GPU run), because the fake abtest's `stop_servers()` would end it. I did not
  override the guard. The one e2e test that touches promote.py (`test_promote_reads_a_block_start_as_one_pair...`) uses `read_runs`
  (3-tuple), `case_pairs`, `case_value` and `MIN_PAIRS`, which this change keeps; run the file once the server is gone.
- `test_promote.py` 45 passed (was 36): robust file not held for proxies; robust picks the attempt `abtest.select` picks (a clean slow
  start is left out, gate keeps it); options differ -> HOLD and not judged; a run inside an arm with other options; old rows and a
  default run written with opts mix; a block start is one pair (same line as the old-mode file, n=4 not 12); 2 starts per arm HOLD with
  "run with reps >= 3 x 3", 3 starts PROMOTE; block "gain open" counts starts; robust block drops an outlier pass and still pairs the start;
  `--noise` names the block unit. `World.runs` gained `passes` and `opts` and `World.rewrite`; every old test is untouched and passes.
- `test_block_reading_sim.py` 3 passed (draw shape and NaN passes, 2 starts, the false-PROMOTE ordering at a fixed seed).

## Risks and limits

- The robust rule can hide a slowdown that the switch itself causes (harness-core's risk). promote.py prints the drops per arm but does not
  judge them; read `slow starts N` of the arm against the reference before cherry-picking. A rule such as "arm lost more starts than the
  reference" needs more than the 60 recorded attempts to set a threshold.
- `switch_report.py` is unchanged. Its command line reads every `.engine.log` in the folder, including starts that died or that a robust
  run left out as slow, while promote.py checks only the attempts used; a dead start therefore shows PARTIAL on the command line but not in
  promote.py (said in its docstring).
- The simulation rests on 45 usable starts per case and on an assumed ICC; its numbers are Monte Carlo (20,000 trials, +-0.2 points at 1%).
- Mixed known and unknown tiers still only warn (existing test); a `tier_source` of "guess" is trusted as the tier.

## Next

- One night run with `--block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 9` (3 starts per arm) on `25-kern-hip-confirm`: the first
  promotable block file. `abtest_replay.py --components` on it gives the real ICC; feed it to `block_reading_sim.py --icc`.
- Re-run `block_reading_sim.py` if `BLOCK_ICC` or the outlier rule changes.
