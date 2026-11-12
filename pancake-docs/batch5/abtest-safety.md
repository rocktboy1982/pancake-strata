# Batch 5, abtest-safety: abtest stops only its own servers and refuses to start beside another test or engine

Branch `dev/p16c/abtest-safety` (from `dev/phase16-academic` 712aea988). Files: `tools/pancake/abtest.py` and its tests
(`test_abtest_core.py`, `test_abtest_e2e.py`). No GPU, no real engine: the e2e tests run abtest against the fake engine.

## The problem (batch 4, harness-sim)

`abtest.stop_servers()` sent SIGTERM to every python process on the machine whose command line contained
`server.py --engine strata`, then waited for any process named `strata`. A fake or a second abtest therefore killed a real server of
another session, and an abtest started while another test ran corrupted it. (The wait never saw the real engine anyway: its process
name is `strata-host`, so the old wait returned at once.)

## What changed (default output and records unchanged)

- `OWN` holds the `Popen` of every server `start_server` creates. `stop_servers(wait=120)` sends SIGTERM to those only (the server stops
  its engine and exits), waits for them, then waits until the engine exe below them is gone (pids taken from the process tree before the
  signal, so a reparented engine is still waited for). After `wait` s a server that ignored SIGTERM is killed (it is a `Popen` of ours: no
  pid-reuse risk), an engine that is still there is reported. Nothing is signalled by name or pid scan; a leftover or another session's
  server is never touched.
- `main()` refuses to start (message on stderr naming pid and command line, **exit 3**, nothing created: no out file, no work dir) when
  `busy_processes()` finds another process: the same anchored test as `queue_run.sh`'s BUSY pattern (a real python running
  `tools/pancake/abtest.py` / `restore_check.py`, or the `build-hip/strata` binary; a shell that only mentions them does not match;
  `test_busy_is_the_anchored_pattern_of_queue_run_sh` pins that the two strings are identical) plus the Python server of an engine
  (`serve/server.py --engine strata`), which is what a crashed run leaves behind. Its own process tree is skipped. Exit 3 is also what
  `queue_run.sh` uses for "another test or engine running".
- `--force-busy`: start anyway (prints one line, `abtest: --force-busy: running beside N other test / engine process(es)`); for the
  fake-engine tests, e.g. two fake abtests side by side.
- The old `stop_servers()` at the start of `main` is gone (it killed leftovers); a leftover is now reported and refused.
- `__main__`: SIGTERM becomes `sys.exit(143)` and `stop_servers()` runs in a `finally`, so Ctrl-C, a crash, an abort or a `kill` leaves no
  server of this abtest behind (a leftover now stops the next start, so this matters more than before). SIGKILL cannot be handled.

## Tests

- `test_abtest_core.py` (66, was 38): BUSY equals queue_run.sh's; BUSY / SERVER match real command lines and ignore mentions (the
  `until ! pgrep -f ...` watcher, pytest, `abtest_replay.py`, `strata-host`, ...); `busy_processes()` lists an orphaned decoy and skips
  a mention and its own child; `stop_servers` ends its own server and the engine below it in under 10 s while a decoy
  `server.py --engine strata` and a decoy process named `strata` stay alive (the old wait sat on that for 2 minutes), kills a server
  that ignores SIGTERM after `wait`, signals nobody when it started nothing; `start_server` registers its Popen; `main` refuses (exit 3,
  nothing touched), goes on with `--force-busy`, says nothing extra on a free machine. An autouse fixture sets `busy_processes` to
  "nothing runs" for the other unit tests (they must not depend on what runs on the machine).
- `test_abtest_e2e.py` (24, was 17), fake engine: two concurrent abtests (A slow, B started once A's server is up and run to the end) both
  finish with every row and each ends only its own server; refusal for a decoy abtest / restore_check / `build-hip/strata` (exit 3, no
  files, then `--force-busy` runs and the decoy is left alone); a shell that only mentions abtest.py does not block; a leftover
  `server.py --engine strata` is reported with its pid, not killed, and still alive after a `--force-busy` run; SIGTERM to a running
  abtest ends its server too (exit 143). All skip for an abtest without `--force-busy`, and (as before) while a real run is active.
- Regression check: the same new e2e tests against the OLD abtest (plus an ignored `--force-busy`) fail: A "lost its server to B"
  (connection error), the leftover was killed (exit 0 instead of 3), SIGTERM left the server (exit -15).
- Replay: `abtest_replay.py` over the 32 recorded abtest jsonl files in `bench/results` (23 replay, the 9 older ones fail identically
  with a KeyError before and after) gives byte-identical output before and after (1569 lines compared with `cmp`).
- Result: `python3 -I -m pytest -q tools/pancake/test_abtest_core.py tools/pancake/test_abtest_e2e.py tools/pancake/test_abtest_cases.py`
  = 72 passed, 24 skipped while another session's real server was up (every e2e test skips, as required); the same command inside a private
  PID namespace (`bwrap --unshare-pid --proc /proc`, where no real process is visible or reachable) = 95 passed, 1 xpassed. The xpass is
  `test_queue_busy_check_ignores_a_shell_that_only_mentions_abtest` (xfail, strict=False): queue_run.sh's BUSY is anchored now.

## Risk

- A stale server (crashed run) now makes abtest refuse with exit 3 instead of being killed silently: stop it by hand (`kill PID`, the message
  lists it). `queue_run.sh` then logs exit 3 for every spec until it is gone (each spec's summary stays empty, so a rerun runs it).
- An engine exe still tearing down after the 120 s wait is reported and also makes the next start refuse (BUSY matches `build-hip/strata`):
  correct, the machine is not quiet.
- `ready_wait` expiry ("not ready, carry on") leaves the server in `OWN`; it is stopped at the end of the run as before.

## Next / for other owners

- `quality_gate.py`, `task_gate.py` and `sweep.py` call `abtest.stop_servers()` at their start to pre-clean: it now stops only servers
  started in the same process by `abtest.start_server`, so that pre-clean is a no-op and they have no busy refusal. They can call
  `abtest.busy_processes()` (list of `(pid, command line)`) and exit 3 the same way. `restore_check.py` has its own start / stop
  (`start_new_session`) and no refusal either.
- `probes/fake_engine.py` (docstring, hazard paragraph) still describes the old name-based kill; `real_run_active()` stays useful as the guard.
