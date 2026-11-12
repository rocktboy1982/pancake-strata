# Pancake night log (started 2026-11-11 evening, for the morning of 2026-11-12)
Owner request: after 17.6, ONLY testing; backup to the USB at each step so another ntfs3 hang cannot lose data; note everything.
Times are local. Facts only; no numbers here are measurements unless marked.

- 22:3x  Rebooted machine verified (fsck, bundle). Repo stays on D; git checkout.workers=1. USB PANCAKE now mounted read-write.
- Done today: PANCAKE_STATUS commits pushed (port/rdna3-decode 2ba5a9459); prefilllevers committed (7639dbf76) and merged into dev/p17/integration (a35f6b243), pushed; HIP build of strata + strata_tests OK, 150 cpu ctests pass (9 skipped).
- 17.6 written: pancake-docs/PRESET_BUNDLES.md (commit a375429c5 on dev/p17/integration, not pushed yet at this line).
- 22:55  17.6 pushed (dev/p17/integration a375429c5). Backup script ~/bin/pancake-backup.sh (root bundle + USB copy, verified), first run OK (149M bundle).
- 22:57  Night queue preflight clean (build-hip -> build-int, .venv -> main repo .venv symlinks in the integration worktree, ignored by git). 9router STOPPED for the quiet-machine guard (restart: nohup 9router -n -H 127.0.0.1 --skip-update -l > ~/.9router/server.log 2>&1 &).
- 22:57  Handoff prepared: ~/pancake-handoff-2026-11-12.md + ~/bin/pancake-continue.sh [pool] (starts 9router only when run). Owner: if the main session hits its limit, switch to the 9r-* / coding-* pools.
- 22:58  Guard blocked job 2 for ~2 min (load 8.3 after job 1 + my backup). Backup script now runs niced/ionice idle so periodic backups perturb measurements less. Job 1 (defer-iq2xs-01A) DONE in 2.3 min (est 8).

## Interim results, deferral arm (23:20, PRELIMINARY: 5 of 43 jobs, kept passes only, from RESULTS.md)
Decode tok/s, median of kept passes (A = deferral on, B = deferral removed, C = gate auto-closed):
- A deferral on   (01A, n=2): overall 105.7 (104.5-106.9) | code 131.2 | prose 103.2 | accept 68.7% | TTFT 0.18 s
- B deferral off  (02B+05B, n=4): overall 103.2 (100.1-104.9) | code 121.0 | prose 99.5 | accept 67.7% | TTFT 0.20 s
- C gate auto     (03C+04C, n=4): overall 93.3 (86.8-96.5) | code 103.2 | prose 83.2 | accept 68.3% | TTFT 0.19 s
Read: no clear verdict yet. A vs B overlaps (about +2%, within run-to-run spread). C is lower, but its two jobs are far apart
in spread and the machine was not fully quiet at the start of the run (load guard only needs < 1.0 at job start).
Contrast with MOD_PLAN 17.1 (deferral off = -3% on IQ2_XS): this run does not reproduce that sign. Decide only after all arms.
Source: /home/dan/night-queue-run/RESULTS.md (per-pass tables) and raw/*.json. Not yet written into the repo plan.
- 00:46  PROBLEM found: job ttft-avalon-iq2xs is marked `done` (3.7 min) but has NO measurements. Every prompt failed (ConnectionRefused). Server log raw/srv-ttft-avalon-iq2xs.log: "ROCm error: out of memory" during the first request (Avalon llama-server, -c 131072, -b 4096 -ub 4096). The queue treats it as done, so --retry-failed will NOT pick it up; it needs a manual rerun or a state.json edit after the queue finishes. Pancake and rdna3 TTFT jobs have real numbers (8K/32K/64K measured).
  Likely cause (unverified): GPU memory not freed from the previous job or the Avalon config too big for 20 GB. Check before rerunning: rocm-smi memory before start.
- 02:05  Inventory: incomplete 2026-11-11 A/B runs (jsonl, no summary): 10-rank-guard-fold (10 of 24 runs), 25-kern-hip (10 of 40 runs). Never run: 25-a8-swaps-auto-blocking. Confirm runs (25-kern-hip-confirm, 25-a8-confirm) wait for the screens.
- 02:05  Chain ~/bin/abtest-chain.sh queued: starts after night queue pid 69999 exits; runs the 3 specs in order into /home/dan/abtest-2026-11-12 (root disk). Log: /home/dan/abtest-2026-11-12/chain.log
- 02:43  NIGHT QUEUE FINISHED: 43 of 43 jobs marked done. Only ttft-avalon-iq2xs has failed passes (ROCm out of memory, no measurements). prose-avalon and the other kvring/lever/defer jobs have real numbers (checked the raw logs).
- 02:47  A/B chain (~/bin/abtest-chain.sh) ran but produced NOTHING: "server died before ready (free RAM in 2 MiB blocks: 28 GiB; fix the machine first: sudo tools/pancake/os_tune.sh compact)". Needs memory compaction (sudo) before any engine start. Spec copies fixed in ~/abtest-specs (base path configs/pancake/flashnext-*.json). Outputs in ~/abtest-2026-11-12 (empty results).
- Next (needs owner): run `sudo tools/pancake/os_tune.sh compact` from the worktree, then rerun ~/bin/abtest-chain.sh (remove its outputs first). Avalon TTFT rerun still pending.
- 09:39  Owner decision: skip the A/B chain (failed: server died before ready even after compaction); keep only the Avalon TTFT rerun (~/night-queue-rerun-avalon). Owner declined giving the assistant sudo; compaction is run by the owner.
- 09:50  Avalon TTFT rerun finished (2.0 min): ONLY the 8K prompt measured (8157 tok, TTFT 6.38 s, sd 6.26-6.43). 32K, 64K, 100K failed: Avalon ROCm out of memory on GPU (not system RAM; compaction did not help). Owner decision: keep the 8K result only; the long-prompt TTFT comparison vs Pancake is NOT measured. Avalon config (-c 131072 -b 4096 -ub 4096) is the probable cause, unverified.
- Status: night queue 43/43 jobs done (ttft-avalon-iq2xs in the main run is the failed one; the rerun in ~/night-queue-rerun-avalon supersedes its 8K row only). A/B chain (10-rank-guard-fold, 25-kern-hip, 25-a8-swaps-auto-blocking) skipped by owner: produced nothing.
- 10:36  Owner: redo 25-kern-hip. Queued ~/bin/kern-redo.sh after the a8 run; quiet gates max-load 1.0, foreign-cpu 5, max-wait 1800. Old run kept as 25-kern-hip.jsonl (noisy: base short 82-129 tok/s).
- 12:07  25-kern-hip redo: speed inconclusive (all arms within noise); quality flipped vs run 1 (route-fast/expdec/all3 DIFFERENT then identical). Rerunning only the odd arms: ~/bin/kern-odd.sh -> 25-kern-hip-odd.jsonl
- 12:29  STOPPED odd-arms rerun at owner request (killed by PID); partial output in stopped-odd-partial/. Full kern-hip retest waits for the owner's clock pin.
- 14:19  Scheduled tonight 23:00: ~/bin/tonight.sh = recheck-kern (edit base + copy256 quality, 6 reps), confirm-kern (base/expdec/all3, 6 reps), confirm-a8 (base/swaps64/swaps-auto, 6 reps). Skipped: 10-rank-guard-fold (needs rdna3 engine not built here). Clock: owner pinned 'high'; restore auto after runs. Log: ~/abtest-2026-11-12/tonight/tonight.log
- 14:20  Queued rank-guard (10-rank-guard-fold, HIP build-int, screen tier) to start after tonight chain: ~/bin/rank-guard-after-tonight.sh; log tonight/rank-guard.log
- 14:28  Queued TTFT baseline rerun after rank-guard: ~/bin/ttft-tonight.sh -> ~/night-queue-rerun-ttft (log tonight/ttft.log)
- 14:55  Owner (now, not night): run controls: ctrl-avalon-tailoff, ctrl-avalon-iq3ks-main, ctrl-pancake-base-rep -> /home/dan/night-queue-controls (plan ~/night-queue-controls-plan.json). Deferral off also applied to port/rdna3-decode (9f0bdf6b1), 17.2 rewritten.
- 15:30  Owner: run full prose set (12 starts/arm, A B B A) after tonight's runs: ~/bin/fullset-after-tonight.sh -> /home/dan/night-queue-fullset (log night-queue-fullset.log). Controls done: tail-off 97.3, iq3ks-main 90.7, pancake repeat 95.0 (prose means, single starts).
