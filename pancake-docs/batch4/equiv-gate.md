# Batch 4, equiv-gate: a noise-aware restore-equivalence gate (A14)

Branch `dev/p16b/equiv-gate` (from `dev/phase16-academic` 0dd7ef2e8). Files: `tools/pancake/restore_check.py`,
`tools/pancake/test_restore_check.py`. Nothing here ran the engine; the two saved GPU runs in
`bench/results/2026-11-11-gpu/` (`restore-check.json`, `restore-check-BASEBINARY.json`) are the evidence and the test fixtures.

## Problem

The strict gate (every turn must match the re-read arm token for token) said DIFFERENT on every turn on both binaries, `first`
(no cache involved) included, and `reread repeat vs tool-retry`. What held: `restore` and `disk` `repeat == tool-retry`.
What the saved runs show about the noise (text units, see Limits; two runs of the same turn, all pairs inside each file):

- 31 divergences in 2,549 watched positions: h = 0.0122 per unit. README quotes 1.2-1.3% of tokens for stock run-to-run noise
  (token-forcing gate); same size, so the free-running divergence is what that noise predicts.
- On the cold `first` turn, with no cache anywhere, the new binary's re-read leaves the other five runs at unit 26 and those
  five differ from each other at unit 92.
- On `tool-retry` the four runs (restore and re-read of each binary) split at unit 2 into two groups that follow neither the
  arm nor the binary ({new restore, base re-read} against {base restore, new re-read}): a near-tie that any numeric change flips
  (a cache restore, or a different binary). So the 2-unit split is noise-like, not a restore signature.

## Source check

- MOD_PLAN A14 cites UNSW 2026, arXiv 2609.15030 (read as the abstract page only, 2026-11-11; the full text was not read). It
  compares generated token ids (64 per generation) with a *modified recomputation control* that matches the checkpoint
  scheduling and the per-rank kernel configuration (34/36 to 36/36 identical), no logits, no tolerance. So its text-equality
  gate works because it removes the numeric differences first. Ours cannot (stock engine, 1.2-1.3% noise), which is why the
  comparison needs a control that measures the noise instead. The re-read arm is only an approximate analogue of their
  control: a re-read prefix is computed in other batches than a restored one.
- `tools/force_check.py` (SpecPilot, token forcing) already judges against >= 2 stock arms for the noise. Same idea; there
  the continuation is forced through the engine's STRATA_FOLLOW.
- Teacher forcing through the server: not supported. `/v1/chat/completions` returns logprobs of the tokens the engine
  generated (`logprobs: true`, `top_logprobs` 0-20, from the engine's LP lines, `serve/test_logprobs.py`), has no `echo`, no
  prompt logprobs and no way to give a continuation. The engine can force one (STRATA_FOLLOW, STRATA_FOLLOW_LOGPOS, STRATA_LOGPOS),
  but STRATA_FOLLOW takes one line per engine request in decode order and the server does not map chat calls 1:1 to engine
  requests (`reasoning_close_retry`, which the preset sets, continues a reply in a second one; `--disk` adds side requests).
  Not wired: it cannot be checked without running. What the server does give is enough for a teacher-forced comparison on the
  common prefix (below).

## What was built (all opt-in; without `--control` / `--teacher-forced` the run, the arms and the exit codes are the old ones)

| Switch | Does |
| --- | --- |
| `--control` | arm `control`: a second independent re-read arm (same env as `reread`, own engine start) |
| `--teacher-forced` | `logprobs: true, top_logprobs: 5` on every request of every arm; answers keep token, logprob and the listed alternatives (no bytes) |
| `--alpha` `--lp-tol` | level of the rate test (0.05) and the logprob floor (0.05 nats) |
| `--analyze F [--control-from G]` | judge a saved `--out` file offline; G's re-read arm can stand in for a missing control |
| `--simulate F...` | error rates of the rule on a noise model fitted to saved files |

**Position rule (needs the control).** For each of the five turns that must hit the cache (`HIT_TURNS`), the first differing
text unit between the cache arm and the re-read arm, and between the control and the re-read arm; a turn is a stretch of watched
positions (position + 1 if it diverged, else its length) with at most one divergence. The cache arm FAILS when it leaves the
re-read arm faster per watched position than the control does: one-sided exact conditional test of two Poisson rates
(`binom_tail(n, events_cache, watched_cache / watched_total)`), level 0.05. If a cache arm that splits after three matching
units in every turn would not be flagged against this control, a pass proves nothing: exit 3 (noise floor too high).

**Exact evidence, reported apart.** `repeat == tool-retry` inside each cache arm (restore, disk): a difference is a real
difference (exit 1) whatever the noise. The same pair inside the re-read arm and the control is printed as a noise sample.

**Logprob rule (`--teacher-forced`).** Up to the first differing token two runs have the same context, so the chosen-token
logprobs there are teacher-forced for free. Per hit turn: the largest logprob difference on the shared tokens, and at the first
differing token the larger of the two runs' margins over the other run's token (both listed in the top 5; a split that float
noise caused is a near-tie in both). A cache arm passes when none exceeds max(0.05 nats, 2 x the control's own worst); fewer
than 32 shared tokens over the five turns cannot clear it (exit 3). When it ran, it decides; the position rule is printed.

Exit codes: 0 consistent with noise, 1 real difference, 2 setup error or cache never exercised (checked first), 3 inconclusive.

## Error rates of the position rule (`restore_check.py --simulate`, seeded, 4,000 runs per cell, one cache arm)

Noise model: a near-tie at each position with probability 2h; each run still sharing the prefix takes one of two candidates
by a fair coin, so a pair splits with probability h per position and, of the three runs (cache, re-read, control), the cache
arm is the odd one out as often as the control. h = 0.0122 fitted to the two saved runs, hit-turn lengths 112-175 units.

| cache arm | h/2 | h | 2h |
| --- | --- | --- | --- |
| equivalent: rate rule fails (false failure) | 0.7% | 1.2% | 1.3% |
| equivalent: rate rule inconclusive | 0.0% | 0.2% | 2.9% |
| equivalent: **turn-majority rule fails** (the first rule tried) | 9.0% | 17.5% | 20.3% |
| splits at unit 3 in every turn: rate rule fails | 100% | 99.9% | 97.6% |
| 4x the divergence rate: rate rule fails | 50% | 58% | 58% |
| 11x the divergence rate: rate rule fails | 96% | 97% | 97% |

At h = 0.4 per unit (a control that splits at once) the rule says inconclusive in 99.7% of runs, fails 0.3%.

The first rule tried (pass if the cache arm is not earlier than the control in a majority of turns) fails an equivalent cache
9-20% of the time, because the odd one out is the cache arm in a third of the splits and five turns are few. The rate rule uses
how early each split is (a split at unit 3 is a lot of evidence, one at unit 98 almost none) and keeps the false failure near
1%; the turn-majority count stays in the printout (`earlier than the control in k of 5 turns`) and in `simulate()`. Two cache
arms (restore, disk) double the run-level false failure to about 2-3%.

## The two real runs as fixtures (`RealRuns` in the test file)

Neither saved run has a control arm. Each run's cache arms are judged against its own re-read arm with the OTHER run's re-read
arm as the control (an independent engine start; it also carries what the two binaries differ by, which only makes the
control noisier, so this tests that the rule does not flag real runs, not what a same-binary control would show).

| run | cache arm | earlier than control | p | verdict | exact `repeat == tool-retry` |
| --- | --- | --- | --- | --- | --- |
| new binary | restore | 1 of 5 | 0.507 | consistent with noise | MATCH |
| new binary | disk | 1 of 5 | 0.515 | consistent with noise | MATCH |
| base binary | restore | 1 of 5 | 0.673 | consistent with noise | MATCH |
| base binary | disk | 1 of 5 | 0.673 | consistent with noise | MATCH |

Exit 0 for both. The control resolves an early bug in both (planting a split at unit 3 in every hit turn gives exit 1; a
control that splits at unit 1 gives exit 3). The strict comparison still says DIFFERENT on those files (also tested).

## Tests run

- `python3 -m unittest tools/pancake/test_restore_check.py`: 55 tests, OK (13 old, 42 new): the fixtures above, synthetic runs
  (a bug at token 3 in every turn FAILS, noise as large as a second re-read PASSES, a control splitting at once is exit 3, the
  exact evidence fails a run the positions pass), the `--analyze` exit codes 0/1/2/3 and the strict form, the logprob rule
  (deviation, flip, tolerance from the control, too few tokens), `chat()` against a local stand-in server (request body and
  trimmed entries), and the simulation bounds.
- `restore_check.py --analyze ... --control-from ...` and `--simulate` on the two real files: output as in the tables above.
- Not run: the engine. The `--teacher-forced` request shape was checked against `serve/server.py` (`top_logprobs`,
  `openai_collect`) and `serve/test_logprobs.py`, not against a live server.

## Limits and risk

- Positions are text units (`\w+ | spaces | marks`, about 1.3 per token) because the saved runs hold text, not token ids.
- The noise model is fair-coin, exchangeable and homogeneous in h, with five independent turns. h was fitted to dependent
  pairs of two runs (31 divergences); the table spans 0.5x-2x. It cannot represent a systematic shift: if the engine is
  deterministic for two re-reads (control equal to re-read everywhere) and a cache arm differs from them in a few turns, the
  rule fails the arm, as the strict gate would; whether that is float rounding or a bug is what `--teacher-forced` is for.
- Five turns give little power for a mild problem (4x the divergence rate is caught about 57% of the time).
- `LP_TOL` (0.05 nats), the factor 2 and the 32-token minimum are guesses with no logprob data behind them; the first GPU run
  prints every number, and `--analyze --lp-tol X` re-judges a saved file without the engine. Asking for logprobs may change the
  engine's decode path (it copies the logits to the host), so a `--teacher-forced` run is not comparable to one without; every
  arm of one run asks the same.
- The control is another engine start of the same config. If it ends in exit 3, repeat the run; there is no pooling of several
  controls yet.

## Next

1. GPU, alone, at night: `restore_check.py --tool-boundary --disk --control --teacher-forced --out <dir>/restore-check-noise.json`
   (restore, re-read, control and disk arms: four engine starts and the disk restart). Exit 0 clears the cache arms for the
   items that wait on A14 (F1 park-truncate, A11-A13).
2. Calibrate `LP_TOL` from that run's printed control deviations, then re-judge with `--analyze`.
3. If real logits are wanted beyond the common prefix, extend `tools/force_check.py` (it drives the engine directly, so
   STRATA_FOLLOW maps 1:1) rather than this script.
