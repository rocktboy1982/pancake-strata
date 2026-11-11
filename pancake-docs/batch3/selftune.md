# selftune (Phase 16, MOD_PLAN 16.9 item 70 and the Python tooling around it)

Branch `dev/p16/selftune` (from `dev/phase16-academic` 3f4ba97cf). Python only; no engine, no GPU, no C++ touched, the default
path of the engine is unchanged. `time_ceiling.py`, `cache_learn.py`, `promote.py` were not edited (finemoe / promotion own them).
Everything below was run offline on `bench/prefetch/routing.bin` (219 decode windows, 81% tier hits, the trace the A6 fit used;
untracked in the main checkout, copied read-only) and on synthetic traces.

## What was wrong, what changed

| Item | Finding | Change |
| --- | --- | --- |
| 70 `selftune.py` | The 2026-11-09 dry run (candidate profile +8.6%, + k-th miss +23.1% "modeled") priced policies with `time_ceiling.window_ms` serial and promotions free: the model A6 refuted (k-th miss +14% modeled, +2.0 +-1.6 measured). Judged on one half of one trace (the two halves of the trace correlate 0.36 per expert pair; blocks 0.04-0.67). The A/B spec had `--adapt-async 1` (measured and rejected, the model priced the blocking tier), the rdna3test base (no deferral) and `"lossy"` absent (placement moves work between GPU and CPU kernels whose rounding differs; abtest prints "placement-dependent: quality gate"). Prompt windows (thousands of rows) would have counted as routing. The candidate ranked only the pairs of the old ranking (a short ranking caps the cache, issue #46). | `chain_model.py` (A6 `window_ms2` + the `--validate` constants as one function; deferral and FUSE_PHASES read from the base config's env). Held-out folds: leave one session out, or 4 blocks of one session; a policy goes on only with mean >= 2% and >= 75% of folds gaining and >= 200 windows. Spec: fast-hip base, blocking k-th miss, cases short + edit, `lossy: true`. Decode windows only (<= 16 rows). Every pair ranked. |
| A6 `time_ceiling --validate` | Reproduced byte-identical to `bench/results/2026-11-10-offline/A6_validate_output.txt`. The 3.4-point mean error is 3 out-of-sample arms (+0.7, -1.4, -8.1) and one fit point; see "validation" below. No edit (not my file). | `time_model_bootstrap.py` (new): the spread of each prediction when the trace is resampled. |
| A49 | `decode_context` spec rule (10% / 25% of ms@10K) disagreed with the plan's 15% attention share. Spec and cases are valid (`longdec*` are in abtest). | `decode_context.py` (new): line fit ms = a + b x context, attention share per case, plan verdict, draft acceptance per case; spec note points to it. |
| A52 `load_time.py` | Log regexes match `src/core/expert_source.cpp` (checked), `STRATA_UNBUFFERED_LOAD` 1/0 semantics match, cold = pack dir walked (experts.bin included). Bug: `capped.sh` waits up to 10 min for 48 GiB MemAvailable inside the timed start. "Warm" rows inherit what the run before left (a direct load leaves the cache cold). | RAM wait done first (`waited_s`), `WAIT_GIB=0` to capped.sh, `after` column, docstring estimate marked "not measured". |
| A33-A35 / A34 `sweep.py` | The source is a compiler tuner; see source check. Old rule: one run per point, `t > best`. Simulated on a flat landscape (3% run noise): left the start in 95% of sweeps and reported +4.7% for nothing. `DEFAULT` started at MINBLK 400 / MINB 2 (engine defaults 800 / 4) and `--pool-tasks 48` (engine 0 = automatic). `STRATA_IQ_PREFETCH` (the gate/up CPU kernels of the IQ2_XS preset, default 2048) was not a knob; `STRATA_Q2_PREFETCH` only reaches the Q2_0 down rows. | Noise rule of the source (3 runs, Mann-Whitney, keep only a statistically faster value), per-case normalised samples, no re-measuring, final point confirmed against the start in fresh A B B A runs, start = base config's value. Knob file = single source (first value = engine default, checked in the engine source), + IQ_PREFETCH. |
| `regression_watch.py` | Printed "no regression" and stored an empty last-good record when every run was noisy; compared absolute numbers of different days with a bare 5% (same preset: 116.6 and 136.8 tok/s in two test-phase groups); default config was the pre-deferral rdna3test. | Record = mean + 95% interval; REGRESSION needs drop > tolerance and > both intervals, inside the noise = SUSPECT (exit 3), no clean data = exit 2, never stored; fast-hip default, 4 reps, history file. |
| A33 / A35 specs | `STRATA_EXPDEC_SWIGLU`, `STRATA_ROUTER_FUSE`, `STRATA_IC_WARM_MB` exist; the claim that ROUTER_FUSE is inert under deferral is right (`rpub` needs `!defer_`, verify.cpp 2483); `STRATA_DOORBELL_STORE` (also disables it on HIP) is opt-in, off in the preset. ROUTER_FUSE only applies to windows of 2-8 rows (`--lookup-max-t 12` windows of 9-12 rows keep the plain publish). | none |

## Source check (the deep-research did not cover these)

| Source | Verdict |
| --- | --- |
| Vidur, MLSys'24, 2405.05465 (A6, 45, 70) | Sound as a method: operators profiled and regressed (random forest), request-level latency predicted and validated on 4 models x 3 traces, "< 9% error" is the relative error of normalised end-to-end latency (P50/P95; max 3.33% on the static P95), at 85% of capacity. A6's "3.4 points" is an error in gain points on three arms, equivalent to about the same percent of window time, but from one fitted trace and n = 3, not 12 held-out scenarios; the 5-9% target is borrowed, not shared. |
| Verma, Canesche, Pereira (UFMG), arXiv 2609.03114 (A34) | Mismatch of scope, ok on method. It tunes tile sizes of Pluto-generated CPU kernels and thread-block dimensions on an A100 (+5.5-8.5% over the compiler's default, 1.06-1.28x geomean on CPUs); the number does not transfer to hand-written decode kernels (plan said 1-3%, unmeasured). Its search is neighbourhood hill climbing (footnote 4: explicitly not coordinate descent), 3 runs per kernel, Mann-Whitney U to stop (no statistically faster neighbour), expanded neighbourhoods gave no gain. `sweep.py` is a line search per knob (cheaper) and now takes over the noise rule, the one part that matters. |
| Twilight, 2502.02770 (A49 "method from Twilight") | Mismatch of attribution. Top-p sparse attention (lossy), 15.4x attention operator / 3.9x per-token latency on one A100 at batch 32-256; no share-versus-context curve at batch 1. A49 is a plain measurement and stands without it; the 15% rule is ours. |
| ServerlessLLM, OSDI'24, 2401.14351 (A52) | The loader design matches: chunked bulk reads (16 MB), direct I/O, multi-thread, pinned chunk pool, pipelining; 4 CPU cores saturate the SSD. Numbers: 3.6-4.7x vs Safetensors, 6-8.2x vs PyTorch on a 12 GB/s RAID0 PCIe 4 NVMe (A5000 host); ablation direct I/O 2.1x, threads 2.3x, pinned 1.4x, pipeline 1.5x. The plan's "28-65 s -> 10-14 s" is not in the paper; this engine's buffered path is already chunked and multi-threaded, so expect the direct-I/O part and the page-cache pressure relief, not 3-8x. Measure. |
| SGLang, 2312.07104 (A33 "small-M fusion") | Citation mismatch: RadixAttention and compressed FSMs, nothing on kernel fusion. The fusions are ordinary launch-count reductions; no gain assumed (04 run was n.s.). |
| PRESERVE, 2501.08192 (A35) | Different mechanism: weights/KV prefetched into an accelerator's on-chip cache during inter-device communication in distributed serving (1.6x e2e, simulated accelerators). The Infinity-Cache warm during the combine wait is an adaptation; its spec says to drop it if it does nothing. |
| Item 70 itself | A composition (64 traces, 4/8.6 profiles, 43, 14, 63, 67) with no single paper; the check is the offline numbers below. |

## Modeled against measured

| What | Modeled | Measured | Note |
| --- | --- | --- | --- |
| A6 deferral (edit) | +15.8% | +15.8 +-3.9 | fitted (deferred-layer cost 91 us) |
| A6 deferral, group-02 base | +15.8 | +17.3 +-2.9 | error -1.5 |
| A6 k-th miss 2,8 over deferral | +2.7 | +2.0 +-1.6 | error +0.7; resampled trace -6.3 .. +1.9 (5-95%): the sign is not settled by the trace |
| A6 FUSE_PHASES over deferral | +1.1 | +2.5 +-1.7 | error -1.4; resampled +1.0 .. +1.2 |
| A6 rank guard TOP=2 over serial | +12.0 | +20.1 +-3.3 | error -8.1; resampled +11.6 .. +13.6: a model miss, not trace noise |
| A6 mean error, out-of-sample arms | 3.4 points (n = 3) | | one structural fit point; alpha 0.5 chosen from the DDR bandwidth ratio before KMISS, but the measurement then bounded it to 0.4-0.75 |
| 8.6 learned profile (pancake vs shipped), chain model on routing.bin | +5.4% whole trace (blocks +11.2, +10.7, +8.8, +0.6; first 30 windows +8.9%; static hit 60.5 -> 71.9%) | +6.7% (52.8/63.4/75.8/90.0 -> 57.8/70.8/80.9/91.4 tok/s, 4 runs, an earlier engine, no deferral) | same order; the trace predates the profile (not trained on it); the gain is a cold-start gain (largest early, ~0 when the hit rate is 93%) |
| 8.6 against the coder profile | +11.5% whole, +26.5% first 30 windows | - | not measured |
| selftune candidate profile (learned on routing.bin, 4 blocks held out) | +4.6% mean; blocks +1.4, +8.2, +11.5, -2.5 | not measured (A/B spec written) | goes to the A/B (mean >= 2%, 3 of 4 blocks gain) |
| selftune k-th miss 2/8 on the current profile | +2.0% mean; blocks -0.4, -3.2, +11.8, 0.0 | +2.0 +-1.6 edit, -6.4 short (02b) | does not go on (1 of 4 blocks gains): "mixed", as measured |
| the 2026-11-09 dry run, same policies | +20.6% (k-th miss), +23.1% (both) | +2.0 +-1.6 | refuted; reproduced with the old code |

Learning curve of the candidate profile (8 blocks of the trace, other blocks as training, random subsets): 27 training windows
+1.7% (42% of held-out blocks gain), 54 +4.7% (54%), 109 +9.7% (62%), 191 +9.9% (75%); the spread between blocks is 10-14 points. So:
a profile needs on the order of 100-200 decode windows (500-1000 tokens at 4.7 tokens per window) before its mean gain stops
growing, and even then one block in four loses. `--min-windows 200`, and prefer several sessions (leave-one-session-out) over
blocks of one. `--blend 0.3` was the best of {0, 0.3, 0.6, 0.9} on this trace (mean +4.0 / +4.6 / +2.7 / +0.9): a prior toward the
current ranking helps; the grid is one trace, so it is not tuned further.

## Not covered / limits

- The model is an edit-window model; short chat has no routing trace to validate on (A6 note). selftune prints nothing about it:
  it prices whatever windows the traces hold with the edit-window constants.
- One recorded trace (bench routing, not agent traffic): the real test of item 70 needs `STRATA_WINDOW_TRACE` sessions of
  daily use (item 64); until then the loop is built and its first run is the A/B of `bench/selftune/<day>/abtest-spec.json`.
- sweep.py cannot resolve a 1% knob with 6 samples per point (simulated power: 3% -> 42%, 5% -> 85%; with 24 samples 2% -> 65%,
  3% -> 92%, 1% -> 28%); use `--reps 12` for a night run or the confirm tier of abtest.py for the survivors.
- regression_watch with 4 reps sees ~+-5% intervals: it catches 10% regressions. The drift-free form is an interleaved A/B of the
  last good build (`{"exe": ...}` arm) against the new one; needs the old binary kept.
- Nothing here was run against the engine. `chain_model.py` copies the `--validate` constants; finemoe's branch lifts
  `fit_model` / `model_ms` out of `time_ceiling.py` (window_ms2 unchanged): after both merge, `chain_model` can call those.

## Tests run (CPU, python 3.14, `systemd-run -p MemoryMax=6G`)

`python3 -m pytest -q tools/pancake` -> 141 passed (new: test_selftune 12, test_sweep 9, test_regression_watch 4,
test_load_time +1 (8), test_decode_context 4, test_time_model_bootstrap 2). Offline runs: `time_ceiling.py ... --validate`
(identical to the recorded output), `selftune.py --routing routing.bin` (5 s), `time_model_bootstrap.py` (60 resamples), the
sweep simulation (old vs new rule, power). Not run: abtest, the sweep itself, load_time, regression_watch (they start the server).
