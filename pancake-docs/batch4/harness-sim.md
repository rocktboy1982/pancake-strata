# Batch 4, harness-sim: a fake engine for abtest / queue_run.sh, end-to-end tests, and how many starts a small gain needs

Branch `dev/p16b/harness-sim` (from `dev/phase16-academic` 0dd7ef2e8). New files only: `tools/pancake/probes/fake_engine.py`,
`tools/pancake/test_abtest_e2e.py`, `tools/pancake/test_fake_engine.py`, `tools/pancake/power_sim.py`,
`tools/pancake/test_power_sim.py`. abtest.py, queue_run.sh and promote.py are not touched. No GPU, no model, no engine. The e2e
tests do run abtest.py and queue_run.sh, but against the fake (below).

## 1. The fake engine (`probes/fake_engine.py`)

What abtest really starts is not the `strata` exe. It is `capped.sh <root>/.venv/bin/python serve/server.py --engine strata
--config CFG --port N`, and that Python server then starts the exe over a stdin/stdout line protocol (INFO / READY / GEN lines;
`serve/server.py` is 11,000+ lines and needs a tokenizer). A fake exe would mean running all of that. Everything abtest looks at
is on the server's outside, so the fake replaces the **server**:

- stdout: `ready: http://...` when up (`EngineDied` and exit 1 when it died before that); SIGTERM ends it.
- HTTP: `/v1/chat/completions` (OpenAI shape, the text is hashed) and `/v1/messages/count_tokens`.
- the engine log (config key `log`, appended): `strata serve: prompt P tokens = 0 reused + P read in X ms (A tok/s), G generated in Y ms
  (B tok/s), drafts accepted a of b, 1 checkpoints` per request (the line `abtest.LINE` reads), `[strata] ... engine started: ...` per
  start, optionally the tier line (`expert arena:` / `experts via mmap`) and one `strata switch: NAME acted: ...` line per `STRATA_*`
  env key on the first request (`FAKE_SWITCH_SILENT` / `FAKE_SWITCH_WILLNOT` make a key silent / "will not act").

Figures per request = `base[case] x (1 + gain) x start factor x cold factor x repeat factor x (1 + noise)`. Profile in the config's
`"fake"` dict (defaults in `DEFAULTS`, fitted to the 2026-11-11 data), per arm through env keys in the spec: `FAKE_GAIN_PCT` (the
arm's true effect on the judged figure; `FAKE_GAIN_PCT_<CASE>`), `FAKE_TEXT_SALT` (another text = a lossy arm), `FAKE_SD_START_PCT`,
`FAKE_SD_REQUEST_PCT`, `FAKE_OUTLIER_PROB`, `FAKE_COLD_PCT`, `FAKE_DIE`, `FAKE_TIER`, `FAKE_SEED`. Profile-only: `slow_starts`
(start indices that run slow: a deterministic outlier), `die_starts`, `repeat_pct`, `startup_s`. Random numbers come from (seed,
start index, request index); the start index is the number of `engine started` lines in the log, so a run is reproducible. A repeat
of the **exact same text** inside a start is faster (`edit` +35%, `short` +2%: measured on the real engine, the screens' second
request); abtest's `[pass k]` / `[run ...]` first lines make a request new again, as on the real engine.

**Fake repo root** (`make_fake_root(dir)`, or `python tools/pancake/probes/fake_engine.py --make-root DIR`): a copy of abtest.py /
queue_run.sh (and the other `tools/pancake/*.py|sh`) whose ROOT is a temp directory holding `.venv/bin/python` (a wrapper that adds a
`sitecustomize` compressing every `time.sleep` by `FAKE_TIME_SCALE`, default 0.05, so abtest's 5 s quiet sample and 3 s ready poll
cost 0.25 s and 0.15 s; `FAKE_TIME_SCALE=1` for real time), `tools/pancake/capped.sh` (no systemd, no RAM wait; prints the real one's
`capped: tier=... keep_gib=...` stderr line), `serve/server.py` (a symlink to the fake) and a synthetic `src/` (the `long` prompt has the
real 34,264 characters; enough corpus for every long* case). abtest runs **unmodified**; a 6-start old-mode run takes about 4 s.

**Hazard, found while building it.** `abtest.stop_servers()` ends every process on the machine whose command line contains
`server.py --engine strata` and waits (up to 2 min) for any process named `strata`. An abtest from a fake root therefore stops a real
server of another session. `fake_engine.real_run_active()` looks at argv (not substrings) for a real abtest, restore_check, server or
queue and returns the first it finds; the e2e tests call it before every start and **skip** while one runs (a real 25-a8-confirm
was running when I started: I did not run any abtest until it ended, and ran my CPU work under `CPUQuota=10-15%` meanwhile).

## 2. Tests run

| File | What | baseline (this branch's abtest / queue_run) | harness-core 7afc716ad + tier-tools ce52e022e |
| --- | --- | --- | --- |
| `test_fake_engine.py` (8) | profile merge, classify (short / edit / each long* case, tie by max_tokens), start state, speeds (gain on the judged figure only, cold, repeat), log line vs `abtest.LINE`, Engine repeat by exact text and switch lines, `real_run_active` argv logic, fake root layout | 8 passed | 8 passed |
| `test_abtest_e2e.py` (17) | old mode: A B C / C B A order, record and row schema (no block fields), text parity (exact vs salted), cold warm-up request not in the rows, promote.py reads the file; screen tier (2 requests per case, the second edit +35%); early stop after 2 reps; one server death retried, two abort with rc 2; the gate keeps a slow start that no proxy flagged | 5 passed | 5 passed (default path unchanged) |
| | block (passes per start, A B B A, `[pass k]` keeps every pass cold), `--warmup` (throwaway rows, no extra warm-up request), `--outlier robust` (slow start run again, also with 2 starts per arm), tier record (engine line / capped.sh line), per-start engine log with switch lines, promote.py reads a block start as one pair | 7 skipped (no such flags) | 7 passed |
| | queue_run.sh: dry run; two specs (4 and 2 run lines, `:--tier=screen` arrives, summary.txt); second call skips both and rewrites nothing; busy refusal (exit 3); night window (exit 5) and `--drop-cache` once | 2 skipped (see below), 2 skipped (no `BUSY_PATTERN`, no `--night-only`) | 5 passed |
| | xfail: queue_run's busy check ignores a shell that only mentions abtest | xfail | xfail |
| `test_power_sim.py` (19) | pairwise sd, fit on synthetic attempts and on the real results, promote.py rule vs `promote.gain_ci` on 300 random samples, Bonferroni and MIN_PAIRS, designs, outlier / slow-start rules, `min_units`, `--icc`, a full small report | 19 passed | - |

Run: `python -m pytest tools/pancake/test_fake_engine.py tools/pancake/test_abtest_e2e.py tools/pancake/test_power_sim.py`. Point the
e2e at another build: `ABTEST_UNDER_TEST=/path/abtest.py QUEUE_RUN_UNDER_TEST=/path/queue_run.sh`.

## 3. Findings for the other owners

1. **queue_run.sh's busy check false-positives (tier-tools).** `pgrep -f 'tools/pancake/abtest.py|restore_check.py|bin/strata '`
   matches any command line that merely contains the text. Right now pid 395230 (`bash -c 'until ! pgrep -f "tools/pancake/abtest.py";
   do sleep 20; done; echo A8 confirm finished ...'`) is a watcher that **never ends: it matches itself**. Tonight `queue_run.sh` (old
   or new) would stop with "another test or engine is running" (exit 3) until it is killed. Fix: anchor the pattern at argv, e.g.
   `^[^ ]*python[^ ]* [^ ]*tools/pancake/abtest\.py`. The xfail test passes once it is.
2. **Block mode with the default start count can never PROMOTE (harness-core / switch-harness).** `--block N --reps R` runs
   `max(2, ceil(R / N))` starts per arm. promote.py makes one pair per start (it averages the rows of a case in a record, pairs by rep)
   and needs `MIN_PAIRS` = 3. So `--block 3 --reps 6` gives 2 pairs and always HOLDs ("fewer than 3 clean pairs"). Ask for `reps >= 3 x N`
   (test: `test_promote_reads_a_block_start_as_one_pair...`). In the simulation the k = 2 column is 0.0 everywhere.
3. **The second identical request is a different regime** (edit +35%, expert cache 76% -> 88%): harness-core's `[pass k]` prefix handles
   it, and the e2e test pins it (pass 1 and 2 stay within 15% of pass 0). A "warm passes only" mode without that prefix would measure
   the lookup-draft regime, not the first-pass figure the old confirms report.
4. abtest's `stop_servers` hazard above (a real run and a fake run must never overlap).

## 4. How many starts does a small gain need (`power_sim.py`)

Monte Carlo of promote.py's rule (95% lower bound of the paired gain >= 1% exact / 3% lossy, >= 3 pairs; thresholds imported from
promote.py) on noise fitted from the 60 recorded attempts (`bench/results/2026-11-11-gpu/*.jsonl`, 45 usable starts per case; the
25-a8-confirm file was still growing when I ran it). `python tools/pancake/power_sim.py` prints every table (about 2 minutes);
`--icc`, `--sd-between`, `--sd-within`, `--outlier-prob` run sensitivities.

**Fitted noise** (robust: median / pairwise differences; slow starts = more than 3 robust sds and 5% below the arm's median):

| case | starts kept | sd of one start's figure | between / within a start | slow starts |
| --- | --- | --- | --- | --- |
| short, every start | 45 | 1.22% | 0.95% / 0.77% (ICC 0.61) | 9% of starts, x0.77 |
| edit, every start | 45 | 1.94% | 0 / 1.94% (the split is not identifiable, see risks) | 13% of starts, x0.89 |
| short / edit, the starts abtest's gates kept (clean) | 21 | 0.76% / 2.32% | - | 5% x0.87 / 5% x0.91 (one event each) |

The gates as run discarded 48% of the attempts (30 of 62 in the plan, 29 of 60 here); `--reclaim-max 3.0` and foreign CPU <= 25% would
have discarded 8%. Two procedures are simulated: **gate** (old mode, only clean starts, the discards cost time) and **robust**
(abtest `--outlier robust`: slow starts, more than 8.75% below the arm's reference start, are run again up to 2 times; a pass that is a
robust outlier is dropped). Block columns: k starts per arm x 4 passes; `start` = promote.py's reading (k pairs, t with k - 1 df),
`+p` = abtest's own pass-level summary with the design effect (BLOCK_ICC 0.45), `+n` = every pass a pair, no widening.

**P(PROMOTE) %, exact item (need 1%), `short`** (10,000 trials per cell):

| gain | gate n=3 | n=6 | n=10 | n=20 | robust n=3 | n=6 | n=10 | n=20 | n=40 | block k=3 | k=4 | k=6 | k=4 +p | k=6 +n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.1 | 0.0 | 0.0 | 0.0 | 0.4 | 0.1 | 0.0 | 0.0 | 0.0 | 0.4 | 0.2 | 0.1 | 0.6 | 0.4 |
| 1 | 1.7 | 1.6 | 1.0 | 0.9 | 2.7 | 2.6 | 2.5 | 2.5 | 2.4 | 2.8 | 2.5 | 2.6 | 6.3 | 13.0 |
| 2 | 11.9 | 25.5 | 30.0 | 36.2 | 9.2 | 20.9 | 36.7 | 67.8 | 93.5 | 10.6 | 15.8 | 28.6 | 36.5 | 69.3 |
| 3 | 31.2 | 53.2 | 58.7 | 58.5 | 20.6 | 60.4 | 89.0 | 99.8 | 100 | 26.2 | 45.1 | 75.7 | 80.7 | 98.3 |
| 5 | 64.8 | 72.9 | 68.1 | 95.7 | 53.2 | 98.9 | 100 | 100 | 100 | 64.5 | 92.4 | 99.7 | 99.5 | 99.9 |
| 10 | 75.7 | 92.4 | 99.4 | 100 | 93.8 | 99.9 | 100 | 100 | 100 | 96.0 | 99.4 | 99.9 | 99.6 | 99.9 |

**`edit`, exact item:** robust n=6 / 10 / 20 / 40 at +2%: 9 / 15 / 26 / 43; at +3%: 24 / 40 / 68 / 92; at +5%: 63 / 85 / 99 / 100;
block k=3 / 4 / 6 at +3%: 20 / 31 / 50, at +5%: 49 / 67 / 75. A gain equal to the need promotes alpha/2 = 2.5% of the time (row 1%
exact): the rule asks for a gain **above** the threshold, not at it.

**Minimum for 80% power** (n pairs = 2n starts; block k = starts per arm; `-` = not within 60; minutes = two arms at 109 s per start +
13 s per pass, the gate column with its 48% discards, the robust columns with the slow-start reruns):

| case | true gain | item | gate n | robust n | block k (4 passes, promote.py) | block k (abtest's pass-level reading) | minutes: gate / robust / block |
| --- | --- | --- | --- | --- | --- | --- | --- |
| short | 2% | exact | - | 26 | 20 | 16 | - / 116 / 118 |
| short | 3% | exact | 35 | 9 | 7 | 4 | 275 / 40 / 41 |
| short | 5% | exact | 12 | 4 | 4 | 3 | 94 / 18 / 23 |
| short | 10% | exact | 4 | 3 | 3 | 3 | 31 / 13 / 17 |
| short | 5% | lossy | 37 | 9 | 7 | 4 | 291 / 40 / 41 |
| edit | 2% | exact | - | - | 60 | 54 | - / - / 350 |
| edit | 3% | exact | 39 | 29 | 17 | 15 | 307 / 129 / 99 |
| edit | 5% | exact | 12 | 9 | 7 | 5 | 94 / 40 / 41 |
| edit | 10% | exact | 5 | 4 | 4 | 3 | 39 / 18 / 23 |
| edit | 5% | lossy | 39 | 29 | 18 | 15 | 307 / 129 / 105 |

**Answer for a 2% item:** an exact item that really gains 2% needs about **26 pairs (52 starts, ~2 hours) in old mode with the robust
rule** on `short`, 14-25 starts per arm in block mode depending on ICC (ICC 0.3 / 0.6 / 0.9: 14 / 19 / 25), and on `edit` it is out of
reach (60 starts per arm at the fitted split, never at ICC >= 0.3): the decode noise of `edit` plus its slow starts is larger than the
margin. A 3% exact item needs 7-9 starts per arm on `short` and 17-29 on `edit`. A lossy item (need 3%) has no power at a true 3% (2.5%: the gain must be
above the need); at 5% it needs 9 (`short`) to 29 (`edit`) pairs. Today's gate procedure needs 35-39 pairs for 3% (4.5-5 hours) and
never settles 2%: its 48% discards and the slow starts it keeps are the cost. What cuts the time is the outcome rule (no discards,
slow starts run again), not block mode: a 3% spec takes about 40 / 129 minutes (`short` / `edit`) in old mode with the robust rule and
41 / 99 minutes in block mode, against 275-307 now. Starts are not the whole noise, and more passes per start remove only the
within-start part (4 -> 8 passes changes k by 0-2). The plan's target (a 45-minute spec in 5-8 minutes) is not reached for two arms at
these gains: 13-23 minutes for a 5-10% item, 40-130 for 3%, about 2 hours for 2% on `short`.

**ICC sensitivity** (robust procedure; the total sd per start is kept, only its split between starts and inside a start changes;
abtest assumes 0.45, the fit says 0.61 for `short`, 0 for `edit`):

| case | ICC | 2% exact: old n / block k (4 passes) / block k (8) / pass-level k | 3% exact: same | false PROMOTE at 0%, k=4 x 4: start / pass-level / naive |
| --- | --- | --- | --- | --- |
| short | 0.3 | 27 / 14 / 12 / 14 | 9 / 6 / 5 / 4 | 0.2 / 0.3 / 0.6 |
| short | 0.6 | 27 / 19 / 18 / 15 | 9 / 7 / 7 / 4 | 0.2 / 0.6 / 1.1 |
| short | 0.9 | 27 / 25 / 24 / 17 | 9 / 8 / 8 / 5 | 0.4 / 1.2 / 2.6 |
| short, no slow starts | 0.3 | 27 / 14 / 12 / 14 | 9 / 6 / 5 / 4 | 0.1 / 0.0 / 0.1 |
| edit | 0.3 | - / - / - / 58 | 28 / 20 / 20 / 16 | 0.4 / 1.4 / 4.1 |
| edit | 0.9 | - / - / - / - | 28 / 28 / 27 / 18 | 0.7 / 4.0 / 8.7 |
| edit, no slow starts | 0.3 | - / 32 / 26 / 34 | 18 / 10 / 8 / 9 | 0.2 / 0.1 / 0.9 |

Reading: promote.py's start-level reading keeps the false-PROMOTE rate at or below 0.7% for every ICC. abtest's pass-level summary
with the design effect is close to nominal at ICC 0.3-0.6 and about 4% at ICC 0.9 on `edit` (a calibrated interval gives at most 2.5%),
and reading every pass as a pair without widening reaches 4-9%: do not let block-mode passes be counted as independent pairs
anywhere outside abtest's own (widened) summary. Without slow starts the same items are 20-50% cheaper, which is what the
outlier rule and `drop_cache.py` are for.

## 5. Risks and limits

- 45 usable starts per case from 5 files is a small sample. The slow-start probability (9% / 13%, 5% for the clean subset) rests on 4-6
  events; the gate-procedure columns are lumpy (not monotone in n) for that reason. Treat 20% differences as noise.
- The split of a start's variance between starts and inside a start is the weakest number: only screens have two requests per start,
  and the second request is the warm-pass regime (its ratio to the first varies 1.18-1.41 on `edit`). That ratio spread overstates the
  cold within-start noise, so the `edit` fit puts everything inside a start (ICC 0). Run `--icc` before trusting a block-mode k.
  A real 2-pass block run (`--block 2`) on the next GPU window would measure it.
- Starts are independent in the model (pairing by repetition and A B B A drift cancellation are not credited), the warm-pass level
  is not modeled (only relative noise), and the dropped passes of abtest's rule are paired by key in the model (harness-core pairs by
  key too since 7afc716ad).
- The e2e tests need no GPU but start processes and bind ports: they skip while a real run is active, and they cannot test abtest's
  noise gates (the fake's CPU is idle; the host's load is switched off with `--foreign-cpu 1000000 --reclaim-max 1000000`).
- The fake is not the engine: it checks harness logic (order, records, parsing, retries, rules), not any speed claim.

## 6. Next

- When harness-core / switch-harness / tier-tools merge, run the e2e against the merged tree with no override; update the constants in
  `power_sim.py` automatically (it reads `abtest.SLOW_START`, `ROBUST_*`, `BLOCK_ICC` when present).
- Tier-tools: anchor queue_run.sh's busy pattern (finding 1) and kill pid 395230 before tonight's queue.
- Harness-core: require or default to at least 3 starts per arm in block mode (finding 2), or have promote.py say so.
- First GPU window: one `--block 2 --warmup 1` spec on `short` / `edit` with 6 starts per arm (about 25 minutes) gives the ICC and the
  cold within-start noise; feed it to `power_sim.py --icc`.
