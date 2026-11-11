# Batch 4, harness-core: abtest.py block mode, warm-up passes, outcome-based outlier rule, tier and engine-log recording

Branch `dev/p16b/harness-core` (from `dev/phase16-academic` 0dd7ef2e8). Files: `tools/pancake/abtest.py` (only this agent edits
it), new `tools/pancake/abtest_replay.py`, new `tools/pancake/test_abtest_core.py`. No engine, no GPU, no abtest run: everything
below is checked by replaying the 60 recorded attempts in `bench/results/2026-11-11-gpu/*.jsonl` and by unit tests with a fake
server.

## What changed (all opt-in; the default run is unchanged)

- `--block N`: N passes (every case once) per engine start. Starts in A B B A order, `max(2, ceil(reps / N))` starts per arm.
  Pass k >= 1 gets a `[pass k]` first line on short / edit, so it reads its prompt cold like pass 0. The summary pairs the same
  round and pass, and widens its intervals by the design effect `sqrt(1 + (passes - 1) x BLOCK_ICC)`, with BLOCK_ICC = 0.45.
- `--warmup W`: W throwaway passes per start, instead of the single short warm-up request. The throwaway rows go to
  `warmup_rows`. In block or warm-up runs every row carries `pass` and `warm`.
- `--outlier robust` (confirm tier only): no repeats because of the proxies; `clean` still records the proxies' verdict. Two
  outcome rules instead:
  1. Slow start: more than 8.75% below its arm's reference start on some case. The reference is the faster of two starts, or
     the median from three starts on. Its slot runs again in the same round, up to `--retries` times.
  2. Outlier pass: dropped whole when a request in it is a robust outlier among the same arm, case and request slot:
     `|x - median| > 3.5 x max(1.4826 MAD, 2.5% of median)`, judged only with 3 or more values.
- Every attempt row now records `tier` (arena / mmap / unknown), `tier_source` (engine / capped / guess / none), `keep_gib`,
  `big_free_gib_start` and `opts` (test_tier, reps, block, passes, warmup, outlier, foreign_cpu, reclaim_max, retries,
  early_stop, per_run). The tier comes from three sources, best first:
  - the engine log lines: `experts via mmap` / `not enough RAM for the expert arena` mean mmap, `expert arena:` means arena;
    the .out line `experts loaded:` also means arena;
  - capped.sh's `capped: tier=... keep_gib=...` line (tier-tools branch), which lands in the .out;
  - capped.sh's rule: arena when keep <= 12 GiB and big_free >= 33 GiB.

  Before the summary, abtest warns when the attempts it used mix tiers.
- Each attempt writes `<stem>/<arm>-r<rep>-a<attempt>.engine.log`: this start's lines of the shared, appended engine log. The
  switch-harness agent asked for it. It is written for starts that died too.
- `select(records)` decides what a summary uses, for both rules. It reads the mode from the records' `opts`; it is used by
  main, early stop and the replay. Pairing is now by key (rep, pass, request slot) instead of list position. That gives the same
  pairs when every slot is there, which matches the code's own comment "paired by repetition" and promote.py's pairing.
- The summary format is unchanged. The new lines (robust drops, block design effect, tier warning) are printed before
  `summary`, so queue_run.sh's `summary.txt` is unchanged.

## Default path: byte-identical

I compared the old abtest (git `dev/phase16-academic`) with the new one on every abtest jsonl in bench/results (23 files,
2026-11-10 and 11-11). The check ran the old main's online selection plus the old summary / text_check / screen_verdict
against the new `select` and the same printers. All 23 are identical, and `decided()` agrees on all of them. The default jsonl
rows gain only the new top-level keys (`tier`, `tier_source`, `keep_gib`, `big_free_gib_start`, `opts`); the `rows` keys are
unchanged.

## Replay of the 2026-11-11 data (`python3 -I tools/pancake/abtest_replay.py bench/results/2026-11-11-gpu/*.jsonl --components`)

| rule | attempts started | requests used | dropped |
| --- | --- | --- | --- |
| gate as run (foreign 25%, reclaim 1.0) | 60 | 96 | - |
| gate, reclaim 3.0 | 38 | 96 | - |
| robust | 36 (34 slots + 2 re-runs stood in by recorded attempts) | 94 | 6 (3 slow starts) |

- Slow starts in the data: 7 of 60 attempts, 10-35% slow on the whole start. The gate's proxies discarded 5 of them but kept 2
  proxy-clean ones:
  - 25-a8-swaps base rep 0 a1 (short 100 vs 125). Every arm of that screen was compared against it: base short came out
    109.9 +-123.
  - 25-kern-hip-confirm all3 rep 0 a2 (111 / 131 vs 127 / 143).

  The gate also discarded 21 attempts that were normal. The slow-start rule catches all 7 (wherever an arm has two starts)
  and discards none of the normal ones.
- **25-kern-hip-confirm under robust: all3 short +3.1% +-1.1% (better).** The gate as run gave +0.7% +-6.2 (no difference),
  because of the kept slow attempt. Every non-slow all3 start decoded short at 127.1-127.5 tok/s; every base start was at or
  below 125.1. Other results: expdec-swiglu short +0.1% +-1.2 (no difference), edit no difference for both arms. A real robust
  run would have run the all3 rep 0 slot again instead of leaving it out (n=5). This changes the plan's verdict for all3
  (`HOLD`, "the screen's +2% was noise") on the short case. The owner or lead should rerun before acting on it.
- Screens: robust is not allowed with `--tier screen`. A screen has one start per arm, and replayed under robust, the top2
  slow start (short 93 vs 124) would have stayed in and shown -23.6% "worse".
- Warm-up: the first measured short is +5.0% (median of 60 starts) over the warm-up short in the same start, and a screen's
  second short is +2.0% over its first. Starts keep warming over the first passes, so block runs should use `--warmup 1`.
- Same prompt again in the same server: the edit is +33% (the prompt comes from the conversation cache; expert cache hits go
  from 76% to 88%). That is why block passes get the `[pass k]` line.

## Validity: block mode is less than the "6x" of the plan

- Start-shared noise (one-way ANOVA over the screens' two requests per start, slow starts left out, 22 starts): short within-start SD
  1.22% and between-start SD 0.51% (ICC 0.15); edit 2.16% and 1.94% (ICC 0.45). The estimator was checked on simulated data
  (true ICC 0 / 0.3 / 0.7 gives medians 0.01 / 0.31 / 0.69). So passes in one start are worth less than independent starts:
  2 starts x 3 passes is worth about 3.2 independent starts (design effect 1.9 at ICC 0.45).
- Modeled for 25-kern-hip-confirm (3 arms x 6 reps, 67 min recorded, 27 attempts). The model uses the recorded 105 s start
  overhead and 13 s cold pass, `--warmup 1`, and no re-runs of slow starts (add about 12% of starts):

  | block | same reps | same power |
  | --- | --- | --- |
  | 2 | 21 min | 36 min |
  | 3 | 16 min (6 starts, 4.2x) | 31 min (12 starts, 2.2x) |
  | 4 | 17 min | 34 min |
  | 6 | 19 min | 39 min |

  Block 3 is the best. The plan's "45 minutes to 5-8 minutes" is not reachable without losing validity: 2 starts per arm at
  about 105 s each is the floor.
- Recommended use: `--block 3 --warmup 1 --outlier robust --reclaim-max 3`. Use reps 6 for a quick read (about 4x faster,
  intervals about 1.4x wider), or reps 12 for the old power (about 2x faster).

## Tests (CPU only)

- `python3 -I -m pytest -q tools/pancake/test_abtest_core.py`: 38 passed. Covers:
  - the scheduler (ABBA order, at least 2 starts per arm, slot balance);
  - `[pass k]` prompts;
  - robust bounds, floor and slot groups, a pass dropped whole, prefill cases on prompt tok/s;
  - slow starts (two / three / block);
  - robust picks and selection for gate, robust and replayed thresholds;
  - keyed pairing;
  - the unchanged summary format and the design-effect widening;
  - `decided` with `least`;
  - tier and keep_gib order;
  - the mixed-tier warning;
  - run_arm against a fake server (default rows unchanged, block plus warm-up rows, the warm-up request kept without
    `--warmup`, the engine log of a start that died);
  - main against a fake run_arm (the gate repeats; robust block runs two starts per arm and never repeats; a slow start is
    run again in its own round; screen plus block or robust is refused);
  - replay counts on the real 25-kern-hip-confirm and 10-rank-guard-fold files.
- `python3 -I -m pytest -q tools/pancake`: 331 passed, 1 skipped.
- The old-vs-new identity check on the 23 result files (above).

## Risk

- An outlier rule can hide an intermittent slowdown caused by the switch itself. Every drop is printed with its arm; look at
  which arms lose starts.
- BLOCK_ICC rests on 22 screen starts, and the edit figure comes from the cached second request. Re-estimate it from the
  first real block run: `abtest_replay.py --components` reads block files.
- In robust mode a slow start's re-run comes right after the start that showed it, so it can be minutes away from its
  round partner.
- Block passes measure warm steady state, not "the first request after a start": absolute tok/s and some cache-related
  effects can differ from old-mode results. Do not mix them (`opts.block` / `opts.warmup` are recorded).

## For other agents / the lead

- promote.py (switch-harness) should:
  1. Use `abtest.select(records)` instead of its own read_runs. For robust files, "first clean else quietest" is wrong: `clean`
     only records the proxies. Do not HOLD robust runs for proxy noise.
  2. Reject pairs whose `tier` differs ("unknown" / missing counts as unknown), and pairs with different `opts`
     (block / warmup / outlier).
  3. For block files: promote's `case_value` averages a start's passes and pairs by round, which is valid but gives only
     `rounds` pairs (2 means HOLD with MIN_PAIRS 3). Either use `abtest.paired()` with
     `abtest.design_effect(opts["passes"])`, or ask for at least 3 rounds.
- harness-sim: my tests are in `test_abtest_core.py`; please use other file names.
- tier-tools: the `capped:` line is parsed from the .out (capped.sh's stderr goes there). The engine's own line wins when
  both are present.

## Next

- One night run with `--block 3 --warmup 1 --outlier robust --reclaim-max 3` on 25-kern-hip-confirm. That rechecks all3
  short, measures BLOCK_ICC from real passes, and checks the modeled minutes.
