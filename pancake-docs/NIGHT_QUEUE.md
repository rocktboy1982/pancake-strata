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
4. Kernel level (minutes each, GPU alone): `expdec_bench`, `hip_gdn_stage_bench 4096 16384 32768`, `empty_grid` probe; feed `kernel_gate.py`.
5. One real-trace recording session if `record_session.py` is merged (see its notes).

## 2. After the queue
- `python3 tools/pancake/promote.py <items.json>` on every result; commit results to `dev/phase16-academic` by path; update the plan table
  (Dev / Tested / Cut); `python3 -I tools/pancake/plan_html.py`; `/home/dan/backup-pancake-2026-11-11/backup.sh <label>`.
- Promote ONLY `PROMOTE` verdicts into `port/rdna3-decode` (local commits by path).  Everything else stays on the dev branches for re-testing.
- Never sudo, never bypass the pre-push hook, never push without the owner's go-ahead for anything but dev branches.
