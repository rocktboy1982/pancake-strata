# Night queue runner: one command, one quiet machine, a morning table

MOD_PLAN Phase 17, "Night queue" (this page is the runner; `NIGHT_QUEUE.md` is the older cron queue). `tools/pancake/night_queue.py` (stdlib only) runs the five night items one engine at a
time, unattended, and writes `RESULTS.md`. **Status: the runner and its plan are written and tested with a fake server
only (`tools/pancake/test_night_queue.py`, 22 tests). No engine was started, nothing was built, and no number below is a
measurement.** The durations are the plan's guesses.

## Before you start

1. Run it from the checkout that has this branch merged (`dev/p17/integration` plus `dev/p17/nightqueue`), **built** (`build-hip/strata`
   from that tree, the 7900 XT build) and with `.venv`. `--dry-run` ends with a "preflight" block that lists what is missing (binary,
   model and pack files, scripts, `.venv`); a real start refuses to begin while anything is missing (`--skip-preflight` overrides).
2. The machine must be quiet: no agents, builds, tests or other engines (also stop `pancake.service` / `strata.service` if running).
   The queue checks this itself before every job, see "The guard".
3. Look first: `python3 tools/pancake/night_queue.py --dry-run` prints every job in order with its engine command, the env / args
   delta against the preset, the client commands per pass, the preflight result and the estimated total. `--list` is the short form.

## Start

```sh
cd /home/dan/clone-of-d/src/Strata
setsid nohup python3 tools/pancake/night_queue.py > /tmp/night-queue.out 2>&1 &
```

Everything goes to `bench/results/night-queue/` (change with `--run-dir DIR`): `queue.log`, `state.json`, `RESULTS.md`,
`night_queue.pid`, `configs/` (the rendered config of every job), `raw/` (per job: the engine log `srv-<job>.log`, the engine's
own log `engine-<job>.log`, the turn log, one `<job>-p<N>.json` and `.log` per client pass). A second start in a folder that has a
`state.json` refuses unless you say `--resume` or `--fresh` (moves the old folder aside).

Useful variants: `--group 1-defer` (or `2-ttft`, `3-lever`, `4-kvring`, `5-prose`), `--skip-group 4-kvring`, `--only 'defer-q2-*'`
(names, globs, comma lists), `--fake` (the whole loop on a stdlib fake server, seconds, no GPU: a rehearsal of the plumbing, its
numbers mean nothing).

## What it runs (estimated 7 h 26 min in total; 43 engine starts)

Each line is one engine start with the client passes named. "pancake" = `configs/pancake/flashnext-iq2_xs-fast-hip.json` (the daily
IQ2_XS preset) unless the Q2_0 one is named (`flashnext-q2_0-fast-hip-adopted.json`, the Q2_0 preset with the adopted items, as the
2026-11-11 matrix ran it). Deltas are against that preset.

| Group (MOD_PLAN item) | Jobs | What | Est. |
| --- | --- | --- | --- |
| `1-defer` (item 1) | `defer-iq2xs-01A` ... `-09C`, `defer-q2-01A` ... `-09C` | Per model 9 servers in the order **A B C C B A A B C**: A = the preset (`STRATA_VERIFY_DEFER_CPU=1`), B = `STRATA_VERIFY_DEFER_CPU` removed, C = the gate's `auto-closed` arm (`STRATA_VERIFY_DEFER_AUTO=1`, `_MIN_T=256`, from `bench/specs/p17-defer-auto.json`). Every arm has 3 servers, so each pair meets in both orders. Each server: 4 passes of the five cmp_bench prompts (thinking off, greedy, 512 tokens); pass 1 is the warm-up (excluded), the last 2 passes are kept. | 2 h 28 min |
| `2-ttft` (item 2) | `ttft-pancake-iq2xs-a`, `ttft-pancake-iq2xs-100k-trace`, `ttft-rdna3-iq2xs`, `ttft-avalon-iq2xs` | `ttft_bench.py` scenario A (8K / 32K / 64K prompts, run 1 discarded) plus scenario B (the 10-turn session, 3 sessions) on pancake; the 100K prompt on pancake in its own server with `STRATA_TRACE=1` (chunk-handoff fault check, MOD_PLAN 17.5); then Strata-rdna3 and Avalon with 8K / 32K / 64K / 100K and the session. | 1 h 07 min |
| `3-lever` (item 3) | `lever-01A` ... `-07A` | Order **A B C D E F A** on pancake, each with 3 agent sessions of 10 turns **with a short side request between turns** (`nq_clients.py session`), then `ttft_audit.py` on that server's engine and turn log. A = control; B = `--conversation-cache-min-tokens 8192`; C = a budget floor, `--conversation-cache-mib 6144`; D = `STRATA_PARK_TRUNCATE=1`; E = B + C + D; F = `STRATA_PARK_REWIND_MAX=8192` with `STRATA_PARK_DIVERGENCE_LOG=1`. | 1 h 17 min |
| `4-kvring` (item 4) | `kvring-01S` ... `-08S` | Order **S T R P P R T S**, single prompts at depth then a 256-token decode (`nq_clients.py depth`, 3 runs per size, run 1 discarded): S = static `--kv-resident 32768` (the preset), T = static 49152, R = `STRATA_KV_RING_ADAPT=1` + `--kv-resident 49152` (sizes 1K / 32K / 64K / 120K), P = the same with `--max-context 65536` (sizes 1K / 32K / 56K). | 1 h 44 min |
| `5-prose` (item 5, 17.2) | `prose-base-01A`, `prose-avalon`, `prose-var-01B` ... `-04A` | `prose_gap.py` (5 prompts, 6 runs each, run 1 discarded) on pancake, then Avalon, then the flag-only pancake variants B = `--spec-min-p 0`, C = `--spec 2 --spec-min-p 0`, D = `STRATA_SPEC_COST=1`, then the preset again (drift check). | 0 h 50 min |

Not in the queue: the 17.3 greedy ablation (`python tools/pancake/ablate_greedy.py bench/specs/ablation-17-3-plan.json --group core`,
hours, it starts its own servers and must run alone afterwards) and the agent "edit" case of item 1 (use
`bench/specs/p17-defer-auto.json` with `abtest.py` for the paired edit numbers).

Choices that are mine, not the plan's, and worth a look before the night:

- **Ring (item 4).** Under `--serve` the adaptive rule uses `--max-context`, not the request's depth
  (`src/program/generate.cpp`, the "KV ring adaptive" line; `include/strata/core/kv_ring.hpp`): with the preset's `--max-context 131072`
  and a cap of 49152 the ring is 49152 for every request, so arm R should behave like T (the engine log line "KV ring adaptive: ... ring
  N cells" says what each server got). Only P (`--max-context 65536`) makes the rule pick the 32768 floor. So "ADAPT on vs off" is
  T vs R here; the per-depth switching the PR measured is not what serve mode does. 128K cannot be tested with `--max-context 131072`
  (prompt plus the new tokens must fit), hence 120K and 56K.
- **Budget floor (lever 1b).** I read "a conversation-cache budget floor" as an explicit `--conversation-cache-mib 6144` (the default is
  sized from the free RAM up to 8192 MiB). If the start-up log of the control server shows the automatic budget was already at least that
  big, arm C is a no-op.
- **Side requests.** The session client sends one short unrelated conversation (about 150 words, 16 new tokens) between two turns.
  That is my approximation of a title request or a sub-agent; without it the conversation cache never has to choose and levers 1 and 2
  would show nothing. It is not a recording of real traffic.
- **Strata-rdna3 on IQ2_XS.** Its only config in the repo (`bench/results/2026-11-11-vs-avalon/raw/r3-q2.json`) is the Q2_0 one; the plan
  derives the IQ2_XS config by replacing the pack and the two GGUF names (`engines.rdna3-iq2xs.replace`). Whether that config starts was
  not tried; if it does not, that job shows `failed` and the queue goes on. Avalon runs IQ2_XS only (Q2_0 does not run in its build).
  The Avalon command is copied from `bench/results/2026-11-11-vs-avalon/raw/matrix.sh`.
- Estimates assume about 4 min per Strata start, 3 min for Avalon, and the per-pass guesses in the plan. The first start after a cold
  page cache or a long uptime can take much longer (`--start-timeout-min`, default 25, then the job is `timeout`).

## The guard (before every job)

A job starts only when all of these hold: no process named `strata`, `llama-server`, `cc1plus`, `ninja`, `cmake`, `hipcc` or
`ld.lld` and no python running `abtest.py`, `ablate_greedy.py`, `restore_check.py`, `ttft_bench.py`, `prose_gap.py`, `nq_clients.py`,
`gpu_profile.py`, `task_gate.py`, `quality_gate.py` or a Strata `server.py --engine` (the queue's own process tree is ignored);
1-minute load below 1.0; MemAvailable at least 48 GiB; the job's port free. Otherwise it waits (polls every 10 s, logs the reasons)
up to `--guard-wait-min` (default 30). Still not quiet: the job is **skipped** (`status: skipped`, the reasons in `queue.log` and
`RESULTS.md`) and the queue goes on with the next one; `--resume` tries skipped jobs again. Thresholds: `--load-max`, `--mem-gib`;
`--no-guard` turns it off (not for the night).

Start and stop: the engine is started in its own session (`setsid`) and remembered by PID. Readiness = `GET /health` 200. Stop =
SIGTERM to the leader PID, then SIGTERM to the remaining members of that session by PID, SIGKILL by PID only as the last resort; then
the queue waits until the session is gone, the port is free and MemAvailable is back to 48 GiB. It never uses `pkill -f` or kills by
name. A job that leaves a process or the port behind is marked `stop problem` in `RESULTS.md` and logged.

Other rules: a job timeout (default max(30, 2.5 x the estimate) minutes; `timeout_min` in the plan) covers start, passes and nothing
else; a failed start, a client that exits non-zero or a timeout marks that job and the queue continues.

## Stop and resume

- Stop cleanly: `kill -TERM $(cat bench/results/night-queue/night_queue.pid)` (or Ctrl-C in the foreground). The running engine is
  stopped the same way as after a job, `state.json` and `RESULTS.md` are written, exit code 130.
- Continue: `python3 tools/pancake/night_queue.py --resume`. Jobs that are `done`, `failed` or `timeout` are skipped (a failed job
  would fail the same way; `--retry-failed` runs them again); `skipped`, `interrupted` and never-started jobs run.
- After a crash of the queue itself or a power cut: the job that was `running` counts as unfinished; check that no engine is left
  (`pgrep -x strata`, `pgrep -x llama-server`) before resuming, the guard will refuse to start otherwise.

## In the morning

1. `bench/results/night-queue/queue.log`: START / DONE / SKIP / FAILED / TIMEOUT per job, and any `WARNING ... still there after the stop`.
2. `RESULTS.md` (rewritten after every job; `--results-only` rebuilds it from `state.json`). Top table: every job with its status.
   Then, by kind of client:
   - **Decode** (item 1 and `defer-*`): per pass with the role (pass 1 = warm-up, excluded; kept; extra) and the overall / code /
     prose medians, acceptance and the engine's prompt tok/s and TTFT on those short prompts; then per job (kept passes), then a
     **per-arm table** that pools the kept requests of all servers of an arm and gives the change against arm A, plus the overall
     number of each server in run order (a trend across the order A B C C B A A B C means drift, not the switch). Read code and
     prose separately: the earlier finding is model-dependent and opposite in sign between classes.
   - **TTFT**: per size the client TTFT median / min / max with the engine's prompt ms and tok/s; the session table with total TTFT,
     reuse share and per-turn rows. **Agent session with side requests** (item 3): the same per arm, plus the median TTFT of the side
     requests; the `ttft_audit.py` text and JSON per job are in `raw/lever-*-post1.txt` / `-audit.json`; the divergence lines of arm F
     are in `raw/srv-lever-06F.log` / `engine-lever-06F.log` (they contain prompt text).
   - **Depth** (item 4): per arm and size TTFT, prompt tok/s, decode tok/s and acceptance; check the engine log of each server for the
     "KV ring adaptive" line to see which ring it really got.
   - **Prose**: per prompt the tok/s median and sd, acceptance, tokens and ms per window.
3. Failed or timed-out jobs: the last lines of `raw/srv-<job>.log`. For `ttft-pancake-iq2xs-100k-trace` a crash of the engine at the
   chunk handoff is itself the result (MOD_PLAN 17.5).
4. For arm C of item 1 the engine log must show the `STRATA_VERIFY_DEFER_AUTO=1 (MOD_PLAN 17.1 note` start-up line; without it the arm
   did not act (the same rule as the paired gate).
5. Method: a number is only a result when the pass-by-pass rows agree; with 3 servers per arm and 2 kept passes per server the decode
   medians carry a noise of a few percent (the 2026-11-11 spread was 12-17 tok/s between runs). Decide with the paired block harness
   (`abtest.py`, bar >= 2%), not from this table alone, and update MOD_PLAN / the docs and push after each research step as usual.

## Files

- `tools/pancake/night_queue.py` the runner (plan loading and A B B A expansion, guard, start / stop by PID, state, `RESULTS.md`, dry run, fake server).
- `tools/pancake/night_queue_plan.json` the engines and the 43 jobs.
- `tools/pancake/nq_clients.py` the three small clients (`decode`, `session`, `depth`); `ttft_bench.py` and `prose_gap.py` are used as they are.
- `tools/pancake/test_night_queue.py` tests: `nice -n 19 python3 -I -m pytest -q tools/pancake/test_night_queue.py`.
