# Night test queue (single source of truth for the unattended night run)

Owner rule (2026-11-11): development by day, GPU testing by night.  The night cron (21:07) follows this file; edit this file
(not the cron) when the queue changes.  Repo root for every command: `/home/dan/clone-of-d/src/Strata-p16`.

## 0. Preconditions (stop and report if any fails)
- No `abtest.py`, `restore_check.py` or `strata` process running; `MemAvailable >= 40 GiB`; nobody else using the GPU.
- Rebuild the HIP binary if C++ merged since the last build (build dir `/home/dan/.cache/strata-p16-final-hip2`, configure flags in
  `/tmp/claude-1000/final-build.sh`: `-DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0 -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.0/lib/llvm/bin/clang++
  -DSTRATA_BUILD_TESTS=ON -DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=/home/dan/clone-of-d/src/Strata-p16-compilegate/build-cpu/llama.cpp-src`,
  ninja at `/home/dan/clone-of-d/src/Strata/.venv/bin/ninja`, targets `strata strata_tests`), then `ctest -L cpu` must stay >= 130 passed / 0 failed / 8 skipped.
  The repo root needs the symlinks `.venv -> /home/dan/clone-of-d/src/Strata/.venv` and `build-hip -> /home/dan/.cache/strata-p16-final-hip2`.
- Regenerate the base configs: `/home/dan/clone-of-d/src/Strata-p16/tools/pancake/gen_bases.sh`.
- Check the memory tier: `/home/dan/clone-of-d/src/Strata-p16/tools/pancake/os_tune.sh`.  Free 2 MiB blocks >= 35 GiB -> arena tier
  (`STRATA_RAM_KEEP_GIB=12`) for the arena-dependent specs; otherwise the mmap tier (`STRATA_RAM_KEEP_GIB=20`, queue_run's default).
  Never compare results across tiers; the tier goes into the result notes.
- Optional, non-root: `python3 /home/dan/clone-of-d/src/Strata-p16/tools/pancake/drop_cache.py` (more free blocks); the owner's
  `sudo /home/dan/clone-of-d/src/Strata-p16/tools/pancake/os_tune.sh compact` helps more.

## 1. Order (one GPU job at a time; the lead makes NO other tool calls while a job runs, only a check every ~25 min)
1. `restore_check.py --tool-boundary --disk --control --teacher-forced --out bench/results/<date>-night/restore-check-noise.json`
   (noise-aware gate; exit 0 clears the cache arms, exit 3 = repeat, exit 1 = real difference; then calibrate LP_TOL and re-judge with `--analyze`).
2. A8 swap budget: the daytime confirm of 2026-11-11 is INVALID (stopped at 09:50: it ran while 15 agents compiled, base had 1 clean attempt of 6, 5 discarded). Rerun `25-a8-confirm.json:--no-early-stop` (arms base / swaps64 / swaps-auto) on a quiet night machine, ideally in block mode; only if an arm wins, run `quality_gate.py` for it. Do not start any daytime GPU job while agents build.
3. Queue (use `/home/dan/clone-of-d/src/Strata-p16/tools/pancake/queue_run.sh --night-only 21-06 --drop-cache <OUTDIR> <specs...>`):
   `19-turbo-guards.json`, `20-verify-ecap.json`, `15-q2_0.json`, `25-f-env-rerun.json` (arena tier) or `25-f-env-rerun-mmap.json` (mmap tier),
   `26-stack-small-exact.json:--no-early-stop` (run `ctest -R 'native_multi_parity|iq_multi_parity|hip_wait_copy|iq_sign_fast|q2_fast'` first),
   `25-f1-park-truncate.json`, `25-a11-a13-cache.json`, `a30-fiddler-cpu-share.json`, the selftune candidate spec, `a49-decode-context.json`.
   Use block mode (`--block`) and `--outlier robust` once harness-core's branch is merged and its tests pass.
4. Kernel level (minutes each, GPU alone), then `python3 tools/pancake/kernel_gate.py item KEY ...` with each measurement BEFORE any end-to-end screen: F7 `q2_rows_bench --threads 8 --nt 1` and `--nt 2` (taskset -c 0-7, idle box); F12 and flat loads `expdec_bench 16/42,22/42 def,dec:s0,dec:s1` (and a `-DSTRATA_HIP_GLOBAL_WEIGHTS=1` build); F9 and A33 `gpu_profile.py --env STRATA_ROUTE_FAST=1` / `STRATA_EXPDEC_SWIGLU=1`; A36 `hit_critical.py` on gpu_profile traces with deferral on; A31 `hip_gdn_stage_bench 4096 16384 32768`; `empty_grid` probe.
5. Real-trace recording: smoke `python3 /home/dan/clone-of-d/src/Strata-p16/tools/pancake/record_session.py plan --preset flashnext-iq2_xs-fast-hip --out <DIR> --repo /home/dan/clone-of-d/src/Strata-p16 --run` (few minutes, Ctrl-C, then `check <DIR> --min-windows 5 --min-turns 1`); if no real weights, the owner of verify.cpp applies pancake-docs/batch4/corpus-recorder-engine.patch; then a ~1 hour session, `check`, `convert`, and the printed replay commands (FineMoE, HOBBIT proxy, REAP, EcoSpec, A44/A46 re-judged on real traces).

## 1b. Added 2026-11-11 (afternoon): harness flags, reruns, fill steps
- Use `--block 3 --warmup 1 --outlier robust --reclaim-max 3` with reps >= 9 (3 x N) for every promote-capable spec (reps 6 gives 2 starts per arm and promote.py always HOLDs); block results are warm steady state: do not mix with old-mode results.
- FIRST rerun `25-kern-hip-confirm.json` (all3 short is +3.1% +-1.1% under the robust rule vs +0.7% +-6.2% as run: the HOLD for all3 may be wrong) with the flags above; then `25-a8-confirm.json`.
- `quality_gate.py <spec> --out bench/quality-hip/<day>-fill --set bench/quality/set-v2.json --fill` (one engine start, ~10 min), commit `set.filled.json` as set-v2.json, flip the `--set` default literal in quality_gate.py.
- A base-only task run (3 reps, `task_gate.py --tasks bench/tasks/repo-history.json --limit 110`) to measure real request time; add `"tasks": "bench/tasks/repo-history.json"` to bench/specs/taskgate-*.json; use `--tasks-only` for lossy items.
- Run the pin-hazard confirm in BOTH orders: `A B C D E F A B` (the old rule wins: it keeps A-D forever) and `A B C D E F E F` (the hazard shows: newer sessions return) with `STRATA_AUTO_PIN=1` (now evict-last) vs base.
- Never run a fake-engine abtest test while a real run is active (`stop_servers` kills every strata server until the abtest-safety follow-up lands).

## 2. After the queue
- `python3 tools/pancake/promote.py <items.json>` on every result; commit results to `dev/phase16-academic` by path; update the plan table
  (Dev / Tested / Cut); `python3 -I tools/pancake/plan_html.py`; `/home/dan/backup-pancake-2026-11-11/backup.sh <label>`.
- Promote ONLY `PROMOTE` verdicts into `port/rdna3-decode` (local commits by path).  Everything else stays on the dev branches for re-testing.
- Never sudo, never bypass the pre-push hook, never push without the owner's go-ahead for anything but dev branches.
