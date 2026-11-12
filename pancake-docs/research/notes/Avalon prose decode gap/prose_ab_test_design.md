# Prose A/B test design for Pancake (read-only research)

Scope: read-only. Nothing built, no engine run, no abtest run, nothing written under the D drive.
Labels: [code] read from a named file; [measured] computed from a named result file; [unverified] not checked here.

Corrections to the brief, found while reading:
- `tools/pancake/prose_gap.py` is NOT in the main tree (`port/rdna3-decode`). It exists only as an untracked file in seven agent worktrees under `/home/dan/clone-of-d/src/Strata/.claude/worktrees/`, all with the same md5 (f161d199, 335 lines). The night-queue configs point `exe` and `cwd` at the worktree `agent-ad25cdf1dc7ea704a`.
- The "sd 1.1-7.7 tok/s" figure covers only `prose-base-01A`. Across the 25 Pancake prose series (5 prompts x 5 jobs) the kept-run sd runs from 0.65 to 25.5 tok/s. The `prose-var-04A` repeat is the noisy one.
- `--spec-draft-n-max` is not a Pancake flag in this tree (no match in any .cpp/.hpp/.h/.py). It appears only in Avalon / llama.cpp research notes. Do not use it as an arm.

## Q1. How text identity is checked, and why identical-looking arms gave different text

### Takeaway
The abtest "exact" check is a sha1 of the answer text compared with base. It can only say "identical" or "different", not "exact" or "lossy". The night queue shows that the same config gives different text across server starts and sometimes within one server, so a "different text" result alone proves nothing.

### Cited Findings
- [code] `tools/pancake/abtest.py:530-542` (`text_check`): each arm's text is compared with base's FIRST rep only (`bt[0]`). "identical on every case -> exact"; otherwise "DIFFERENT text -> lossy or placement-dependent: quality gate".
- [code] `tools/pancake/abtest.py:459-460, 483`: one sha1 per request, stored in `texts[c]`. With `--tier screen` (2 requests per case) only the last request of each case is kept. long* cases are excluded (unique prefix line).
- [code] `tools/pancake/abtest.py:4-8, 52-54` (docstring): "an arm whose text is identical to base on every case is exact, so it needs no quality gate."
- [code] `tools/pancake/prose_gap.py:106-108, 234-236`: per-run `text_sha256` (16 hex chars). The summary prints "N distinct of M kept runs ... greedy on one server should repeat".
- [code] `tools/pancake/quality_gate.py:1-12`: scores the stock arm twice (noise floor) and judges other arms with `tools/force_check.py` rules (PASS / REJECT / INCONCLUSIVE). Its docstring: a placement-only switch "does not need this"; lossy ones (route prior, deferral rank guard, depth-graded prior) do.
- [measured] `night-queue-run/configs/prose-base-01A.json` and `prose-var-04A.json` are identical except the `log` and `turn_log` paths (diff run for this note). Both are "preset as is" in `RESULTS.md`.
- [measured] Text hashes, `raw/prose-base-01A-p1.json` vs `raw/prose-var-04A-p1.json` (story): the six hashes are identical in order (48f2ad, e0b596, a1b249, cba6e9, 0676c8, 3e6ae5). For tips they match only on runs 1 and 6 (bd7fcd ... c56f6e). Same config, different text, on the same prompt.
- [measured] Within one server, greedy text is not stable: `prose-base-01A-p1` (story) has 6 distinct hashes in 6 runs; `prose-base-01A-p4` (js) has 5 of 6 equal (f0dab9) and one different. Token counts differ too (story 482 vs 512 in base runs 3-4), so the content differs, not only whitespace.
- [measured, quoted in repo] `MOD_PLAN.md:2456-2457` (Phase 17 intro): "the exact engine is already not run-to-run deterministic (0 of 6 outputs identical exact-vs-exact)".

### Inferences
- The "different text" result from abtest is a weak signal. Two arms with the same effect can differ in text, and an arm that is truly lossy can match base by chance on a short prompt.
- [unverified] The likely cause is timing-dependent placement, not the flags themselves: the base config has `--adapt-every 1`, `--adapt-swaps 32`, `--pool-affinity measure`, `STRATA_PREFILL_CPU_SHARE=0.6`, `STRATA_VERIFY_DEFER_CPU=1`, and `--spec-min-p 0.5`. Any of these can change which tokens go into a verify window, and float reduction order then flips near-tie argmax picks. The engine source was not read to confirm this.
- Fix to the check before any arm is read: add a control arm `base_rep` (same config as base, run in the same batch). Count how often each arm's per-request hashes match ANY base hash, not only base rep 1. An arm counts as "no worse than noise" only when its mismatch rate is no higher than the `base` vs `base_rep` mismatch rate. Otherwise send it to `quality_gate.py`.
- [unverified] Whether abtest's `texts` store is fixed to keep every request (it currently keeps only the last per case).

Proposed spec (structure from `bench/specs/*.json` and `abtest.py:12-23`; file is a plan, not written):
```json
{
 "base": "pancake-flashnext-iq2_xs-fast-hip.json",
 "arms": {
   "base": {},
   "base_rep": {},
   "minp0": {"args": {"--spec-min-p": "0"}}
 },
 "cases": ["short", "edit"],
 "reps": 12,
 "note": "exact-by-design candidates; text check uses base_rep as the noise floor"
}
```
Caveats for this spec:
- [code] `abtest.py` does not read the `"lossy"` key used in existing specs. It is documentation only.
- [code] abtest has no prose case. `short` (a hash-map explanation) is the nearest prose-like case; `edit` is code. Prose needs new cases (e.g. the five `prose_gap` prompts), which means a code change in `abtest.prompts()` (`abtest.py:110-124`). Not done here.
- [unverified] The main-tree base `pancake-flashnext-iq2_xs-fast-hip.json` was not diffed against `configs/prose-base-01A.json`. Diff before use: the night-queue base has `--spec-min-p 0.5` and a worktree `exe`.

### Gaps
- No A-A text data exists on the abtest path. The A-A rate has to be measured first.
- Engine determinism mechanism not checked in source.

## Q2. Reps needed to see 3% at 95% confidence, from the night-queue prose runs

### Takeaway
A 3% difference needs roughly 12 server starts per arm if the between-start CV is 2.5%, about 45 if it is 5%, and about 110 if it is 8%. The night-queue data suggests the between-start CV is about 5% (one pair of runs, five prompts), so 3% is not reachable with the current queue. A 5-6% bar needs roughly 12-20 starts per arm.

### Cited Findings
- [measured] Kept runs are runs 2-6 of each 6-run server start (run 1 discarded). Kept-run CV: 0.6-5.3% for the 20 series of 01A / 01B / 02C / 03D; 3.5-21.5% for the five 04A series. Source: `raw/prose-*-p*.json`, computed here.
- [measured] Same config, two server starts (01A vs 04A), per-prompt medians of kept runs:

  | prompt | 01A | 04A | 04A vs 01A |
  | --- | --- | --- | --- |
  | story | 100.6 | 80.8 | -19.7% |
  | tips | 109.7 | 104.8 | -4.5% |
  | moe | 106.7 | 100.8 | -5.5% |
  | js | 149.2 | 132.0 | -11.5% |
  | lru | 149.0 | 122.0 | -18.1% |

  Standard deviation of the five differences: about 7.0 points, so about 5% per server start if the two starts are equal in noise (computed here, n = 5).
- [measured] `RESULTS.md` (Prose profile): the `sd` column is the sd of kept runs of one prompt in one start (matches the raw data).
- [code] `abtest.py:578-592`: foreign-CPU gate (25% of a core), reclaim gate (1.0% of a core), per-attempt noise discard and retry. `abtest.py:1-8` says the load average is not used as the gate.
- [measured] Every prose_gap run is flagged `noisy: true` (1-min load above 2.0 before or after): `raw/prose-*-p*.json` (`loadavg_before/after`). `abtest.py:210-213` (`quiet_gate` docstring) says the 1-min load right after a run is the finished server's own 16 threads decaying, so it is not a foreign-load signal; `queue.log` guard lines are load-based. The queue waited for load below 1 (`queue.log` guard lines), not for foreign CPU.
- [code] `abtest.py:496-512` (`decided`): early stop once an interval excludes 0 or is below 1.5% of base. `abtest.py:487-493`: CI uses a t table to df 9 and 2.0 above.
- [measured] Wall time per prose start: 3.8-5.6 min (`RESULTS.md` jobs table, `queue.log`).

### Inferences
- Formula: n per arm = 2 (t_{0.975} + t_{0.8})^2 (s / delta)^2, two-sided alpha 0.05, power 0.8, delta = 3% (about 3 tok/s at 100 tok/s). Table (normal quantiles, since scipy is not installed here; t-quantiles for small n add about 5-10%):

  | between-start CV | reps per arm |
  | --- | --- |
  | 1% | 2 |
  | 2% | 7 |
  | 2.5% | 12 |
  | 3% | 16 |
  | 5% | 45 |
  | 8% | 112 |
  | 11% | 212 |

- The screen tier (2 requests per case, one start) cannot judge 3%. Its promote bar is +2% (`abtest.py:515-527`) and it has one start per arm.
- Time: at about 4-5 min per start, 12 starts x 3 arms is about 3 h; 45 starts x 3 arms is about 11 h. Runs belong on the night queue (owner rule, `MOD_PLAN.md` Phase 17).
- Prose gap itself: Avalon minus Pancake base on prose medians is +3.6% (story), +4.5% (tips), +8.4% (moe), -15.8% (js), -10.9% (lru) (`RESULTS.md`, prose-avalon vs prose-base-01A). The prose part of that gap (3.6-8.4%) is inside the between-start spread seen above, so the gap is not yet established. This matches MOD_PLAN's "5-10%" claim only on three of five prompts, and Avalon is behind on the two code prompts.

### Gaps
- Between-start CV rests on one pair of starts and five prompts. More same-config starts (A-A) are needed to pin it.
- No scipy in this environment, so exact t-quantiles were not computed.

## Q3. Candidate arms, and what is already measured as worse

### Takeaway
The existing flags show no prose gain of 2% or more. `--spec-min-p 0` is flat to slightly worse on prose; `--spec 2` with min-p 0 is clearly worse on decode; `STRATA_SPEC_COST=1` is flat on prose and worse on code. `--spec-draft-n-max` and any "verify batch" flag are not confirmed to exist in Pancake. The first arm to test is the A-A control; a new lever from the Phase 17.2 profile is needed for a real prose gain.

### Cited Findings (prose medians, pancake, kept runs; delta vs 01A)
- [measured] `prose-var-01B` (`--spec-min-p 0`): story 95.7 (-4.9%), tips 110.2 (+0.5%), moe 100.3 (-6.0%), js 147.1 (-1.4%), lru 152.4 (+2.3%). Acceptance falls on prose (story 52.8% -> 33.7%). `RESULTS.md` prose profile; config `configs/prose-var-01B.json`.
- [measured] `prose-var-02C` (`--spec 2 --spec-min-p 0`): story 95.9 (-4.7%), tips 103.5 (-5.6%), moe 98.0 (-8.2%), js 113.2 (-24.1%), lru 114.7 (-23.0%). Acceptance rises (story 59.2%). Decode is lower. `RESULTS.md`.
- [measured] `prose-var-03D` (`STRATA_SPEC_COST=1`): story 100.5 (-0.1%), tips 111.9 (+2.0%), moe 104.3 (-2.3%), js 134.2 (-10.0%), lru 139.1 (-6.6%). `RESULTS.md`.
- [measured] Earlier: `MOD_PLAN.md:1150`: `STRATA_SPEC_COST` 112.0 / 98.3 vs base 111.7 in "Round A (worse/flat)". `MOD_PLAN.md:2460`: "spec 2 is slower than spec 4 on both pancake and Strata-rdna3 (-5%)".
- [measured] `MOD_PLAN.md:1680`: `spec-min-p 0.7` was in a "small-flag stack" that screened -20%, so the flag alone is not isolated.
- [measured] `MOD_PLAN.md:2437-2445`: route prior (`STRATA_ROUTE_PRIOR=0.5`, turbo) is lossy and -15 to -25% on two prose prompts. Drop for prose.
- [measured] `MOD_PLAN.md:2445, 2457-2460`: deferral (`STRATA_VERIFY_DEFER_CPU` off) is +7-8% on Q2_0 but -3% on IQ2_XS, and "NOT established" (overlapping agent load). A deferral retest is in the night queue (`RESULTS.md` 1-defer jobs); read it before adding it as an arm.
- [code] `--spec-draft-n-max`: no match in .cpp / .hpp / .h / .py in the main tree. Only Avalon/llama.cpp research notes (`research_notes/Avalon upgrades from upstream projects/`).
- [code] "verify batch": no flag of that name found. `--verify` (17 hits) and `--verify-rows` (only in `tools/dsv41/engram_iq4nl_bench.py:533`) were the only candidates; neither is a verified Pancake knob. `verify_batch_parity` appears only in `graphify-out/` (generated graph).
- [measured] Current draft depth is `--spec 4` (`MOD_PLAN.md:852`); `--spec 5` is an open item in 17.4, not measured.

### Inferences
- Recommended arm set: `base` (preset, `--spec-min-p 0.5`), `base_rep` (A-A control), and optionally `minp0` if the A-A control shows the between-start noise is small enough to read a 2-3% effect. Do not include `--spec 2` again (already measured -5% to -24%).
- Any arm that is not exact by construction (route prior, deferral rank guard, depth-graded prior) goes to `quality_gate.py`, not to the text hash.
- The Phase 17.2 profile (`gpu_profile.py`, `time_ceiling.py`) is the tool to find a lever. A flag sweep is unlikely to close a gap of 3-8%.

### Gaps
- `--spec-draft-n-max` and a verify-batch flag: no pancake equivalent found. Unverified whether the engine arg parser (`serve/`, `src/`) defines them under another name; not searched in full.
- No prose-specific measurement of deferral on IQ2_XS.

## Q4. What must be avoided

### Takeaway
Avoid: running abtest or prose jobs in daytime; putting logs, results or work dirs on the D drive (ntfs3 deadlock risk); reading the screen tier as a 3% result; treating the text hash as an exactness proof. The GPU clock is currently "high" (user constraint confirmed by sysfs for one read). The 85 C junction guard was not found anywhere in the repo or memory.

### Cited Findings
- [measured, read-only sysfs 2026-11-12] Card1 (device 0x744c, the RX 7900 XT): `power_dpm_force_performance_level = high`; `pp_dpm_sclk` line marked `*` read `0Mhz` at that moment. Card2 is the Vega iGPU (0x1638) on `auto`. The 2026-11-10 note in `research_notes/Strata forks speed gains/tools_benchmarks.md:129` recorded `auto` on the same box, so the level is not fixed over time: check it before each series.
- [measured] Junction read by the first hwmon found: 44 C, which is the iGPU's hwmon, not the 7900 XT's. The 7900 XT junction was not read.
- [code] Junction and clock are only logged in `abtest.py:272-296, 312, 654` (`t_junc`, `sclk`). No junction limit or guard appears in `tools/pancake/`, `MOD_PLAN.md`, `docs/` or the memory notes (grep). The "85 C junction guard" in the brief is therefore [unverified]; abtest does not stop on temperature.
- [memory] `project_ntfs-hang-2026-11-11-evening.md`: two ntfs3 deadlocks on the D drive, triggered by many file creates or deletes at once (a `git reset --hard` with parallel checkout workers). Its fix direction: keep repos and worktrees off NTFS.
- [code] `abtest.py:614`: `work = Path(a.out).with_suffix("")`, so each run writes per-arm config JSON, one server log per start and the jsonl under the `--out` path. Night-queue configs already put logs on root (`/home/dan/night-queue-run/raw/`).
- [code] `abtest.py:406`: the engine is started with `LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib` and `HIP_VISIBLE_DEVICES=0`. Model packs are on D (`/home/dan/clone-of-d/Strata-data/`), read-only use, so no write load there.
- [measured] Root disk: 14 GB free of 107 GB (`df`, 2026-11-12). Keep result files small and on root.
- [memory] Owner rule: benchmarks only at night through the night queue, on a quiet machine, one engine at a time, nothing else running (`MOD_PLAN.md` Phase 17; `feedback_follow-plan-day-night`).
- [memory] Method rule: discard the first run, keep the second pass (`feedback_discard-first-run`); ABBA interleaving (`abtest.py:617`).

### Inferences
- Set `--out` and the config `log` to root disk paths for any abtest or prose series, and keep the number of files written per start small.
- Read the GPU level and clock (`sysfs`) at the start of each series and store it in the result, so a "high" vs "auto" difference is visible, not guessed.
- For a junction guard, ask the owner for the threshold and where it was defined; do not rely on the 85 C figure until it is found.

### Gaps
- 7900 XT junction temperature and clock not read at a loaded moment (no benchmark run here, as required).
- Whether the owner's "pinned high" state persists through a night queue was not checked.

## Summary test plan (for the night queue, after the owner approves)

1. A-A control first: `base` vs `base_rep` on the existing prose prompts, 12 starts each, with the foreign-CPU gate on. This sets the between-start CV and the text-mismatch rate. Cost about 1 h.
2. Decide the reps from the measured CV using the Q2 table. A 3% bar at CV 5% needs about 45 starts per arm; if that is too long, state the bar as 5-6% (about 12-20 starts).
3. Arms: `minp0` only if step 1 shows CV below about 3%. Never `--spec 2`. Never route prior or deferral without `quality_gate.py`.
4. Add prose cases to abtest (the five `prose_gap` prompts) before running. Keep `--out` and logs on root disk.
5. Report per content class (prose vs code), with the A-A row next to every arm.
