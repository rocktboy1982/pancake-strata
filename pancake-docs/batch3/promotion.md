# Promotion gate: which tested items go into port/rdna3-decode (batch 3, agent `promotion`)

Branch `dev/p16/promotion` (worktree `../Strata-p16-promotion`). Python and data only: no C++, no GPU, no engine, no model was started.
Owner rule (2026-11-11): after testing, only items that show value go into the rdna3 branch.

## What was built

| File | What |
| --- | --- |
| `tools/pancake/promote.py` | Reads the result directories and prints PROMOTE / HOLD / REJECT per item with the numbers; prints the `git cherry-pick -x` line for PROMOTE items. It only reads: no `subprocess`, no `os`, no file write, never touches port/rdna3-decode (a test checks the source). `--noise FILES` prints the observed noise and the pairs needed. |
| `tools/pancake/test_promote.py` | 36 tests on synthetic result directories (every verdict, every reader, the cherry-pick lines, the read-only promise, the catalog). |
| `bench/promotion/phase16.json` | The catalog: 35 items (spec, arms, claim cases, exact flag, parity, gates, branch, commits). Items carry the commits named in the MOD_PLAN status table or read from `git log dev/batch-1..dev/batch-2` (A17, A18, A20, A19, A37); every hash was checked with `git cat-file`. Four commits named by the plan are already ancestors of port/rdna3-decode (A11 527e06429, A13 070fe8512, A2 4f56ce535, A23 663107a56; `git merge-base --is-ancestor`): those items have `commits: []`, because PROMOTE for them means switching on by default, not a merge. The other batch-2 items have `commits: []` and the `git log --grep` line in `note` (a prefix grep misses commits such as "DISCO v2:", so a guessed list would be wrong without anyone seeing it). |
| `bench/specs/25-f1-park-truncate.json`, `25-a11-a13-cache.json`, `25-a8-swaps-auto-blocking.json`, `25-a4-w-sweep.json`, `taskgate-a19-probe.json` | The five missing specs (not run). Checked offline: `task_gate.py --dry-run` ok on the task gate spec; `replay_turns.py --synthetic --dry-run` ok on both replay specs. |

Use (run by the owner, from the repo root; the tool needs nothing but the result files):

```sh
python tools/pancake/promote.py bench/promotion/phase16.json            # every item
python tools/pancake/promote.py bench/promotion/phase16.json --only F7-q2-fast,A4-fold
python tools/pancake/promote.py --noise bench/results/<day>-test-phase/*.jsonl
```

Result layout it reads (next to `rev.txt`): `<stem>.jsonl` (abtest), `<stem>-quality/` (quality_gate), `<stem>-taskgate/summary.json`,
`<stem>-replay/summary.json`, `restore-check*.txt`, `ctest-gpu.txt`. `test_phase.sh` writes the first two and the parity files only.
**It does not run `task_gate.py` or `replay_turns.py`** (TEST_PLAN tier 3 says it should), so run them by hand into the same folder:

```sh
OUT=bench/results/<day>-test-phase
.venv/bin/python tools/pancake/task_gate.py bench/specs/<stem>.json --out $OUT/<stem>-taskgate
.venv/bin/python tools/pancake/replay_turns.py bench/specs/<stem>.json --out $OUT/<stem>-replay --reps 3 --arms <arms>
```

I did not edit `test_phase.sh`: bash reads a script while it runs, and another session may be running it.

## Rules (the thresholds, each with its source)

Any REJECT reason gives REJECT, else any HOLD reason gives HOLD, else PROMOTE. An item with several arms gets its best arm.

| Check | Rule | Source / reason |
| --- | --- | --- |
| Gain | 95% lower bound of the paired gain (Student t over runs, paired by repetition) on at least one declared claim case >= 1% of base (exact) or >= 3% (lossy) | SpecPilot (arXiv 2611.02501, sec. 3.2): promote only if the gain's 95% lower bound exceeds 1% plus twice the stock-vs-stock noise, "~3%" if the outputs diverge. Its noise term is < 0.1% on an H100; here the interval half-width is 2-7%, so the paired interval is the noise term |
| Several comparisons | Bonferroni over (arms x claim cases); the claim cases are declared in the item, in advance | A gain picked from 2 cases x 3 arms is a 1-in-3 false positive without it. Test: the same data promotes with `claim: [edit]` and holds without |
| No gain | every claim case has an upper bound < the threshold -> REJECT | abtest's own stop rule ("no effect worth having") |
| Pairs | >= 3 clean pairs per claim case (item may set `min_pairs`; the report prints it) | n = 2 gives t = 12.7 and a deceptively small SD by chance |
| Noise | a used run that abtest flagged noisy (foreign CPU, reclaim) -> HOLD | the quiet-machine gate exists for this |
| Regression | any case: upper bound < -1% or point estimate <= -3% -> HOLD | -3% is `abtest --tier screen`'s rule; mixed results (A4: +20% edit, -6.9% short) must be fixed or decided by the owner |
| Parity (exact) | text hash equal to base on every run of every non-prefill case, else REJECT ("declared exact, text differs"); `restore_check` all MATCH; named ctest parity tests passed; `waived` with a reason | an exact item whose text differs is a bug or a rounding-level item; flag it lossy and it goes through the lossy gates |
| Token forcing (lossy) | PASS needed; REJECT -> REJECT; INCONCLUSIVE -> HOLD; the two stock runs must differ by < 0.5 pts | `force_check.py` = SpecPilot sec. 3.3 (rate rule +0.5 pts at 95%, large-margin event rule) |
| Task gate (lossy) | BETTER or PASS needed; REJECT -> REJECT; INCONCLUSIVE -> HOLD; base drift warning -> HOLD | Displacement Is Not Direction (arXiv 2606.19558) |
| Replay items | gain = total prompt time of the replay (lower is better) with the same gain rule; abtest short/edit only checked for regressions | `replay_turns.py` verdict metric |
| Task gate items | gain = the time-to-success AUC difference lower bound (points) with the same rule | A57 |
| Provenance | `tested_rev` (optional) must match `rev.txt`; an appended rerun in the jsonl (same arm/rep/attempt twice) -> HOLD | stale or mixed results must not promote |

Item flags that change the rules: `vs` (gain against another arm: F21 over `fusewait`), `gates` (narrow the lossy gates, printed),
`min_gain`, `min_pairs` (printed as an override), `parity`.

## Source check: do the cited methods apply here?

SpecPilot (read the PDF text, sec. 3.2-3.3, 6.2):
- Token forcing is a **bug / reward-hacking test for lossless systems** ("losslessness holds by construction"). We use it on lossy
  arms: a PASS means "no large deviation", never "no quality loss". The 0.5-point margin was set against a stock mismatch rate of
  0.72%; ours is 1.7-2.0% (IQ2/Q2_0, MTP), so the same margin is a smaller relative bound.
- The paper scores the system's own outputs with a stock FP32-head engine. `quality_gate.py` scores one fixed text with the
  candidate engine (force_check's "change of the scorer" mode): valid for the routing/numerics arms, not what the paper tests.
- The paper audits large-margin events (explained if stock reproduces them); `force_check.py` counts them raw (documented there). It can
  over-reject, never under-reject.
- Its promotion rule (lower bound > 1% + 2x noise; ~3% when outputs diverge) is the model for the gain thresholds above. The paper
  measures on 2 GPUs with swapped roles; we pair by ABBA repetition on one GPU.

Displacement Is Not Direction (read the PDF text, sec. 4, App. C, E):
- Verified: in the near-baseline "silent zone" (KLD <= 0.064 on Qwen3.6-35B-A3B, <= 0.027 on Devstral-Small-2-24B) rho(KLD, score) is
  +0.00 / -0.24, and the bootstrap intervals "are wide and straddle zero". The paper reads this as no *detectable* ranking signal,
  not as proof of none: `task_gate.py`'s docstring ("carry no information") is a little stronger than the source.
- The cohorts are 28 / 41 **quantized** models. Routing changes (deferral, turbo prior) are another kind of change; the transfer is
  plausible, not shown. The paper gives no protocol for a task gate (tasks, repeats): see the power numbers below.

Our own tools (read as source, then checked on real data):
- `abtest.T95` falls back to 2.0 above df 10 (the exact value at df 11 is 2.20): intervals are too narrow for reps >= 12. promote.py
  recomputes exact t quantiles; it matches abtest's printed numbers on `05c-lookup12.jsonl` (edit +8.8% +-6.8, short -3.5% +-3.6; a test).
- abtest's early stop can end at 2 pairs; promote needs 3. abtest's "better" means the interval excludes 0; promote needs the threshold.
- Result files from before the text-hash feature (2026-11-09 and 04c) carry no `texts`: exact items on them HOLD ("no text hashes").

### How many repeats for the observed noise

`promote.py --noise` on the 95 comparisons with >= 3 pairs in the 2026-11-09/10 test phases (decode tok/s, short and edit):
paired SD of the difference, % of base: **median 3.9, p25 1.6, p75 13.5, p90 21.4** (memory-reclaim and noisy-neighbour outliers).
Share of comparisons that could show a gain within N pairs: 1% gain: 9/95 within 4, 24/95 within 12; 3% gain: 27/95 within 4, 51/95 within 12.
So the default 3-4 reps prove only large gains (deferral +15.8% at SD 1.6 qualifies; a 1-3% item does not).
Pairs needed for the lower bound to reach the threshold (t interval, 95%):

| true gain \ paired SD | 1.0% | 1.5% | 2.0% | 4.0% |
| --- | --- | --- | --- | --- |
| exact, 2% | 7 | 12 | 18 | 64 |
| exact, 3% | 4 | 5 | 7 | 18 |
| exact, 5% | 3 | 3 | 4 | 7 |
| lossy (>= 3%), 5% | 4 | 5 | 7 | 18 |
| lossy, 8% | 3 | 3 | 3 | 5 |

Rule of thumb: for an exact item with a 1-3% claim run `ABTEST_ARGS="--no-early-stop --reps 8"` (or 12) on a quiet machine
(`os_tune.sh`); a noisy base (CV 13% in 03-cpu-pool) needs the machine fixed, not more reps. Every HOLD line prints the pairs that
would settle it at the observed SD, or says it is more than 200 (too close to the threshold or too noisy).

### Token forcing power at our set size (Monte Carlo with `force_check.check`, stock mismatch 1.8%, 266 tokens/request)

The real set (`bench/quality/set.json`) has 20 requests and 5,322 tokens. Verdict shares over 40 trials (PASS/INCONCLUSIVE/REJECT):

| requests | identical to stock | +0.5 pts | +1.0 pts | +1.5 pts |
| --- | --- | --- | --- | --- |
| 20 | 31/9/0 | 2/36/2 | 0/15/25 | 0/1/39 |
| 40 | 37/3/0 | 3/36/1 | 0/3/37 | 0/0/40 |
| 80 | 39/1/0 | 2/36/2 | 0/0/40 | 0/0/40 |

At 20 requests an arm identical to stock gets INCONCLUSIVE 22% of the time and a real +1.0 pt deviation is rejected only 62% of the
time (turbo's +0.83 [0.42, 1.27] on 2026-11-10 is that band). **Use >= 40 requests (about 10,600 tokens) in the quality set.**

### Task gate power (Monte Carlo with `task_gate.boot_diff` + `verdict`, margin 5 pts, independent draws = worst case)

Task mix: 10 tasks at 95% pass, 3 at 80%, 3 at 50%, 2 at 10%. Share (PASS or BETTER / INCONCLUSIVE / REJECT), 30-60 trials:

| gate size | true loss 0 | 5 pts | 10 pts |
| --- | --- | --- | --- |
| 18 tasks x 5 reps (today) | 2/58/0 | 1/59/0 | 0/58/2 |
| 18 tasks x 10 reps | 6/54/0 | 1/57/2 | 0/54/6 |
| 36 tasks x 10 reps | 18/42/0 | 0/60/0 | 0/44/16 |
| 72 tasks x 10 reps | 20/10/0 | 0/29/1 | 0/17/13 |
| 108 tasks x 10 reps | 26/4/0 | 0/30/0 | 0/16/14 |

**The 18-task gate cannot pass an arm that is identical to base (3%) and cannot reject a 10-point loss (3%).** Repetitions help little;
tasks help. Same-seed pairing makes exactly identical arms equal, but a lossy arm changes the sampled path after its first different
token, so the worst case is the realistic one. Consequence for this tool: with the default gates no lossy item can reach PROMOTE until the
task set is about 100 tasks (A55 plans BFCL / LiveCodeBench subsets) or the owner narrows `gates` for an item or widens the margin.
That is a decision for the owner; the tool keeps the strict default and prints the numbers.

### Real-data run (existing 2026-11-09/10 results, read-only; `rev` as in each `rev.txt`)

| Item | Verdict | Why |
| --- | --- | --- |
| A3 deferral (token forcing only) | PROMOTE | edit +15.8% [+11.9, +19.7] n=3, short/long no regression, token forcing PASS |
| A4 `_TOP=2` | HOLD | edit +20.1%, short -6.9% [-10.5, -3.4], long point -8.8% (the A4 fold exists for this) |
| lookup12 | HOLD | edit +8.8% [+2.0, +15.6] straddles 3%, short -3.5%, 5 of 8 runs noisy |
| expdec-down (exact) | HOLD | short +1.7% [+1.2, +2.2] but the file has no text hashes |
| turbo | HOLD | edit +29.7% [-23.9, +83.4]; token forcing INCONCLUSIVE |

These match the decisions the owner took by hand, except lookup12, which the tool holds for noise and the short-chat point estimate.

## Mapping: every Phase 16 item to the spec and the test that decides it

Abbreviations: AB = `abtest.py` (confirm; `--reps 8` for small gains), TF = `quality_gate.py` (token forcing), TG = `task_gate.py`, RP =
`replay_turns.py`, RC = `restore_check.py` (plain, `--tool-boundary`, `--disk`), CT = ctest (`-L gpu` or `-L cpu`), OFF = offline tool (no
GPU). "Catalog" = the name in `bench/promotion/phase16.json`. **NEW** = spec written by this agent. **GAP** = no spec yet.

Implementation and test status table (dev/batch-1):

| Item | Exact | Spec (arm) | Deciding test | Catalog | Flag |
| --- | --- | --- | --- | --- | --- |
| A3 deferral | lossy-small | 01-deferral (defer) | AB + TF (+ TG) | done, in preset | - |
| POOL_FUSE_PHASES | exact | 03-cpu-pool, 03b-fuse-min | AB (+ TF on 03b) | done, in preset | - |
| EXPDEC_DOWN | exact | 04c-gpu-retest (expdec-down) | AB, text parity | done, in preset | file lacks text hashes |
| lookup window 12 | lossy | 05c-lookup12 | AB + TF + TG | - | in rdna3 already (dan-99); promote.py says HOLD |
| small-flag stack | - | 05s-screen | AB screen | closed | - |
| A4 `_TOP=2` | lossy-small | 01-deferral (defer-top2) | AB + TF | - | HOLD (short -6.9%); superseded by the fold |
| A4 fold | lossy-small | 10-rank-guard-fold (top2-fold1/2) | AB + TF + TG | A4-fold | claim edit, short must not regress |
| A4 w sweep | lossy-small | 25-a4-w-sweep (**NEW**) | AB + TF + TG | - | tuning of the above: a config decision, no commits |
| A5 k-th miss | rounding | 15-kmiss (kmiss28), 02b | AB + TF + TG; offline trace replay (cache_sim) | A5-kmiss | the bench hit rate is 89%: AB cannot show the agent-traffic gain |
| async tier, POOL_TAIL4, MWAITX, LAZY_PARK, COMBINE_POLL1, DEVICE_SPIN | - | 02, 03, 04, 04c | AB | closed | rejected / n.s. |
| 4.0.1 conversation cache flags | exact | 11-conv-cache (cache2048-min8192) | RP (decides) + RC + AB side check | 4.0.1-conversation-cache | flags exist: a preset change |
| F1 PARK_TRUNCATE | exact | 25-f1-park-truncate (**NEW**) | RP (vs the two cache flags) + RC + unit test | F1-park-truncate | was GAP |
| F7 Q2_FAST | exact | 23-batch1-confirm (q2fast) | AB text parity; q2_rows_bench (bit-exact, done) | F7-q2-fast | - |
| F21 WAIT_COPY256 | exact | 23-batch1-confirm (fusewait-copy256 vs fusewait) | AB text parity | F21-wait-copy256 | the old spec compared with base; `vs` fixes the reference |
| F9 ROUTE_FAST | exact | 17-f9-route-fast (route-fast) | AB + CT native_multi_parity (bitwise lines) | F9-route-fast | - |
| F12 IQ_SIGN_FAST | exact | 22-f12-sign-fast | AB + CT iq_multi_parity + expdec_bench | F12-sign-fast | spec reps is 1: use 8 |
| F13 turbo guards | lossy | 19-turbo-guards | AB + TF + TG | F13-turbo-guards | - |
| F5 VERIFY_ECAP | lossy | 20-verify-ecap | AB + TF + TG | F5-verify-ecap | - |
| F14 EXP_MODE=9 | exact | 21-f14-exp9-3060 | AB (RTX 3060) | F14-exp9-3060 | needs the 3060, not installed |
| A55 task gate | - | taskgate-*.json | `test_task_gate.py` (13 tests), `--dry-run` | - | tool; see the power numbers: it needs more tasks |
| disk cache key v2 | exact | none | unit test (95 checks), CT -L cpu | - | correctness fix: no speed spec by design |
| A17 turn log `effort` | - | none | python tests (8) | - | tool |
| F6 exact spec sampling | exact (distribution) | 12-spec-sampling (T 1.0), a21-spec-lowtemp (T 0.1) | AB | F6-spec-sampling, A21-spec-lowtemp | **GAP: no tool checks the output distribution**; parity is `waived` with a reason |
| F8 SPEC_CUM | exact | none | OFF `tail_replay.py` on a window trace (>= 2% to port) | - | **GAP by design**: not ported; the offline ceiling is 1.0% short / 7.9% edit |
| F4 shadow pack | lossy | none | OFF 4.0.5 (done), then TG on the requantized file | - | **GAP**: needs the pack file; blocked on RAM and D space |
| A6 time model | exact (offline) | none | OFF `time_ceiling.py --validate` | - | tool |
| A7 chunked promotions | rounding | 22-a7-promo-chunk (reference = async) | AB + TF + TG | A7-promo-chunk | - |
| A8 swap budget, blocking tier | rounding | 25-a8-swaps-auto-blocking (**NEW**) | AB + TF + TG vs base and vs the best fixed arm | A8-swaps-auto | was GAP; the A6 model expects no gain |
| A46 learned admission | - | none | OFF `cache_learn.py` | - | closed |

Batch 2 table:

| Item | Exact | Spec (arm) | Deciding test | Catalog | Flag |
| --- | --- | --- | --- | --- | --- |
| A15 park TTL | exact | 24-a15-park-ttl (ttl30, ttl5) | RP + RC + AB side check | A15-park-ttl | - |
| A17 caps | lossy tail | taskgate-caps (cap-p99) | TG (time AUC) | A17-caps | caps need 200+ effort-tagged turns |
| A18 effort | lossy | taskgate-effort (low, xhigh, off) | TG (time AUC) | A18-effort | a policy decision, no code |
| A20 rumination | close lossy | taskgate-rumination | TG (time AUC) | A20-rumination | thresholds are placeholders |
| A19 hidden dump | exact | 16-a19-hidden-dump (dump) | AB text parity (a prerequisite, no gain of its own) | - | goes in only with the probe |
| A19 probe | lossy | taskgate-a19-probe (**NEW**) | TG (time AUC) | A19-think-probe | was GAP; needs trained weights (path in the spec is a placeholder) |
| A21 | exact (distribution) | a21-spec-lowtemp | AB | A21-spec-lowtemp | as F6 |
| A22 DISCO | exact | 24-a22-disco (disco) | AB text parity (needs a fitted weights file) | A22-disco | traces first (`disco_fit.py`) |
| A23 spec cost | exact | 24-a22-disco (spec-cost) | AB | A23-spec-cost | - |
| A25 / A26 | exact (offline) | none | OFF `draft_report.py` | - | report tool |
| A24 suffix index | exact | 24-a24-suffix-index (sidx, sidx-fallback) | AB; RP block for agent shape | A24-suffix-index | the spec says lossy: if the text differs, set exact false |
| A29 entropy | - | none | OFF `expert_entropy.py` | - | closed |
| A30 CPU share | rounding | a30-fiddler-cpu-share | AB (long1k, long3k) + TF + TG | A30-prefill-cpu-share | - |
| A33 EXPDEC_SWIGLU | exact | a33-a35-fusion-retest | AB text parity | A33-expdec-swiglu | - |
| A33 ROUTER_FUSE | exact | a33-router-fuse-nodefer (vs defer-preset) | AB | A33-router-fuse | skipped while deferral is on: expect REJECT |
| A35 IC_WARM | exact | a33-a35-fusion-retest | AB | A35-ic-warm | - |
| A34 sweep | exact | `sweep-knobs-fast-hip.json` (input of `sweep.py`) | AB on the winner | - | the spec is generated from the sweep's best point |
| A31 GDN chunk 64 | rounding | none | OFF `gdn_stage_bench` (build only if rec/floor >= 2) | - | **GAP by design**: no switch until the bench says build |
| A36 weight-major | exact | none | OFF `probes/hit_critical.py` | - | **GAP by design** (rule first) |
| A37 LDS LUT | bitwise | 24-a37-lut-dup (lut16, lut8, lut16-rb128) | AB + rocprof counters | A37-lut-dup | spec reps is 1: use 8 |
| A38 counters | log exact | a38-agent-loop (base) | python tests | - | log-only: value is the signal in real sessions |
| A41 / A42 / A43 nudges | lossy | a38-agent-loop (finish-nudge, externalize8k, oldstring-note) | **none**: the spec only shows no speed cost | - | **GAP: no deciding test.** The task gate has no agent loop with tests and the replay runs no tools; needs a real-session replay with the A57 AUC |
| A44 bandit | lossy-small | 24-a44-bandit (reference = defer) | AB + TF + TG | A44-bandit | - |
| A45 calibration | exact | 24-a45-calibrate (calibrate vs static-default) | AB | A45-calibrate | expected share 0 on this box (presets set `--pcie-frac 0`) |
| A52 direct loader | exact | none | `load_time.py` (start time), `STRATA_VERIFY_ARENA` | - | **GAP**: the gain is start time; promote.py has no reader for it (decode-unchanged is covered by any AB) |
| A53 IOMMU guard | exact | none | `os_tune.sh check` | - | startup note |
| A56 / A57 | - | none | `force_check.py selftest`, `test_task_gate.py` | - | tools |
| upstream sync | exact | none | build + `ple_residency_test` | - | correctness |

Method-validity order (batch 3) and the rest of Phase 16:

| Item | Spec | Deciding test | Flag |
| --- | --- | --- | --- |
| FineMoE (dev/p16/finemoe) | none | OFF hit-rate and time model on gate traces; an engine switch only if modeled decode time gains >= 2% | **GAP until the switch has a name** (the agent's note will give it) |
| HOBBIT gate-magnitude proxy (rankguard) | none; the skip rule would reuse 10 / 25-a4-w-sweep | OFF correlation and threshold re-fit | **GAP**, lossy skip only |
| EcoSpec cost-aware drafts (disco-suffix) | 24-a22-disco (spec-cost) is the nearest existing arm | OFF replay of draft trees + routing traces | **GAP**: the new switch is not named yet |
| S2-MoE KV rollback (shadowpack) | none | OFF acceptance on traces | **GAP**: design note only |
| REAP pruning (shadowpack) | none | OFF saliency, then TG on a pruned file | **GAP**: needs the pruned model file |
| S2-MoE reuse-aware gating, DraftExpert | none | - | not started |
| Kernels (F7, F12, F21, A37, A31, F9) | rows above | AB + parity | second wave |
| A1 turbo | 06-turbo-depth (turbo), 19-turbo-guards | AB + TF + TG | A1-turbo (preset decision) |
| A2 depth prior | 06-turbo-depth (turbo-depth75) | AB + TF + TG | A2-depth-prior |
| A9 idle adapt | 02-tier (idle-adapt4) | **none**: the bench has no idle gaps and the replay caps think time at 2 s | **GAP**: needs real sessions with tool gaps |
| A10 seeding | none | OFF `replay_seeding.py` | inert until item 84 |
| A11 / A12 / A13 | 25-a11-a13-cache (**NEW**) | RP + RC (plain, `--tool-boundary`, `--disk`) + AB side check | A11-tool-boundary, A12-auto-pin, A13-disk-tier; were GAP |
| A14 restore test | none | `restore_check.py` (gates A11-A15) | tool; its txt files are parity inputs |
| A16, A47 | done | - | in presets |
| A27 / A28 | none | TG on another model / shadow pack file | **GAP**: needs the files and an owner decision |
| A32 prefill streaming | none | code read first | - |
| A39, A40, A50, A51 | parked | A49 (a49-decode-context, one arm: a measurement) decides A50/A51 | no promotion |
| A48 | none | OFF `q2_rows_bench` | decided F7 |
| A58 | done | - | abtest |
| Phase 15 specs outside these tables | 18-f11-attn-window, 18-f15-fetch-admit, 18-f17-poly-fattn-3060, 18-f22-wmma-q4-gfx11 | AB + parity | not in the catalog (their branches are not in batch 3) |

## Tests run (this worktree, `systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 nice -n 19`)

Passed:
- `python -m pytest tools/pancake/test_promote.py` : 36 passed (also `python tools/pancake/test_promote.py`). Mutation check by hand: a
  lossy threshold of 1%, regression rules off, `MIN_PAIRS` 1, Bonferroni off: each makes a test fail.
- `promote.py` on the real 2026-11-09/10 results: abtest's printed intervals reproduced (a test), table above.
- `promote.py bench/promotion/phase16.json`: 35 items parse and validate against their specs; all HOLD "not run".
- `python tools/force_check.py selftest` ok (no engine); `task_gate.py bench/specs/taskgate-a19-probe.json --dry-run` ok;
  `replay_turns.py bench/specs/25-f1-park-truncate.json` and `25-a11-a13-cache.json` with `--synthetic --dry-run` ok (57 and 113 min planned).
- ast-grep over `tools/pancake/promote.py`: no `subprocess`, `os.system`, `Popen`, `open(..., "w")` or `write_text`.

Not run: any abtest, quality_gate, task_gate (live), replay, restore_check or ctest (no GPU, as ordered). The five new specs have not run.
The two Monte Carlo tables used `force_check.check` and `task_gate.boot_diff` on synthetic data.

## Risks and next

- The thresholds are conventions from one paper's setting (1% / 3%); the owner can override per item (`min_gain`, printed).
- Strict gates mean few PROMOTEs until the quality set has >= 40 requests and the task set about 100 tasks. Decision for the owner: grow
  them, or narrow `gates` for items with no behavior risk (rounding-level), or accept token forcing alone.
- `commits` is empty for batch-2 items on purpose; fill it from the `git log` line in each `note` after reading each commit.
- The cherry-pick line assumes the commits are plain (non-merge) and apply in the listed order; CMake test lists conflict (batch 2 had
  union merges): expect to resolve by hand.
- Next: a loader reader (start time, A52) and a replay for A9 need data that does not exist yet; fill the engine-switch specs for
  FineMoE / EcoSpec / HOBBIT after the agents name their switches.
