#!/bin/sh
# bench-sched.sh - scheduler that puts the two updated-strata benchmarks BEFORE the rest of chain AA on the box (owner order 2026-11-19, autonomy:
# "prioritize those benchmarks on the box - make sure they are not taking multiple hours").  The file name has no "win-chain", "redo-chain" or "tail-chain":
# no busy() pattern of any chain sees this script (T-944).
#
# What it does (nothing is killed, aborted, deleted; no sudo; no sudoers edit; no box file is edited in place):
#   a. finds the AA shell, EXACTLY `sh ./redo-chainAA.sh` (cmdline checked; zero matches = AA is not running, several = ERROR and stop) and sends SIGSTOP to
#      that one pid.  AA is a sequential `sh` script: it freezes BETWEEN its steps; the child that runs now (abtest, a build) is not touched and finishes;
#   b. waits until no measured job runs (the busy() patterns of the chains, WITHOUT the frozen AA shell): 2 clean polls of 15 s = 30 idle seconds (owner 2026-11-19), once;
#   c. runs the two benchmark chains one after the other, in this shell (so their end is known): bench-updatedAA.sh (plain upstream Strata v0.2.16 against
#      updated-strata, marker WINBENCHUS_DONE) and bench-updatedRD.sh (plain Strata-rdna3 against updated-strata-rdna3, marker WINBENCHRD_DONE), both with
#      BENCH_SCHED=1 (no waiting for AA), BENCH_PREIDLED=1 (quick idle only), BENCH_KEEP_SERVICE_DOWN=1 (pancake.service stays down: AA needs the box without
#      it: its busy() sees `strata --serve`).  A chain that aborts or fails writes its marker (ERROR + ..._ABORTED + ..._DONE); a missing marker is logged as ERROR;
#      the next chain runs in every case;
#   d. SIGCONT to the AA shell (after a cmdline check that the pid is still `sh ./redo-chainAA.sh`), on every way out of this script (normal end, error,
#      HUP / INT / TERM).  AA continues exactly where it was (its first wait_idle sees nothing of this script or the chains: they have ended).
# Only SIGKILL of this script leaves AA frozen: then run on the box   kill -CONT <pid of sh ./redo-chainAA.sh>   (pgrep -fx 'sh ./redo-chainAA.sh').
# Log: /home/dan/clone-of-d/src/bench-sched.log, last line BENCHSCHED_DONE.  Start:  cd /home/dan/clone-of-d/src && setsid nohup sh ./bench-sched.sh > /dev/null 2>&1 &
# Environment: BENCH_US / BENCH_RD = the chain scripts (default ./bench-updatedAA.sh ./bench-updatedRD.sh), SIGCMD = the kill command (tests).
# BENCH_EXTRA=<script> (a third slot, after US and RD): run that chain too, with the same three variables; its log is <script name without .sh>.log in this
# directory and its marker is the line its own header names (bench-exact.sh: bench-exact.log, WINBENCHEXACT_DONE).  BENCH_ONLY=EXTRA runs the extra chain alone.
# A versioned copy (bench-exact.v1.sh, bench-rdgate.v1.sh) keeps the log of its family (bench-exact.log / bench-rdgate.log, WINBENCHRDGATE_DONE).
S=/home/dan/clone-of-d/src; L=$S/bench-sched.log
US=${BENCH_US:-./bench-updatedAA.sh}; RD=${BENCH_RD:-./bench-updatedRD.sh}
AA_CMD="sh ./redo-chainAA.sh"; AAPID=
cd $S || exit 1
say() { echo "$*" >> $L; }
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain[A-Za-z0-9]*\.sh|[x-z]tail-chain[A-Z]\.sh|[a]fter7" >/dev/null 2>&1; }
sendsig() { ${SIGCMD:-kill} "$@"; }
aa_pids() { ps -eo pid=,args= | awk '{ a = $0; sub(/^ *[0-9]+ +/, "", a); if (a == "sh ./redo-chainAA.sh") print $1 }'; }
aa_args() { ps -o args= -p "$1" 2>/dev/null | sed 's/^ *//; s/ *$//'; }
resume() {
  if [ "${BENCH_NO_RESUME:-0}" = 1 ]; then say "AA stays frozen (BENCH_NO_RESUME=1: owner order, benchmarks only)"; return 0; fi
  [ -n "$AAPID" ] || return 0
  a=$(aa_args $AAPID)
  if [ "$a" = "$AA_CMD" ]; then sendsig -CONT $AAPID; say "AA_RESUMED pid $AAPID (state now $(ps -o stat= -p $AAPID 2>/dev/null)) $(date +%T)"
  else say "AA pid $AAPID is gone or is not '$AA_CMD' any more (cmdline '$a'): nothing to resume"; fi
  AAPID=
}
trap resume EXIT
trap 'say "signal received: leaving (AA is resumed first)"; exit 1' HUP INT TERM
say "START $(date +%F_%T) (service: $(systemctl --user is-active pancake.service 2>&1 | head -n 1))"

# ------------------------------------------------------------------ a. freeze the AA shell
set -- $(aa_pids)
if [ $# -eq 1 ]; then
  [ "$(aa_args $1)" = "$AA_CMD" ] || { say "ERROR pid $1 changed its cmdline: nothing stopped"; echo BENCHSCHED_DONE >> $L; exit 1; }
  sendsig -STOP $1; AAPID=$1
  say "AA_PAUSED pid $AAPID cmdline '$AA_CMD' (state $(ps -o stat= -p $AAPID 2>/dev/null)) $(date +%T)"
elif [ $# -eq 0 ]; then say "AA shell not running: nothing to freeze (the benchmarks run alone)"
else say "ERROR $# AA shells ($*): nothing stopped, nothing run"; echo BENCHSCHED_DONE >> $L; exit 1; fi

# ------------------------------------------------------------------ b. wait for the box to be idle: the current AA step ends, then 3 clean polls
q=0; n=0
while [ $q -lt 2 ]; do
  if busy || pgrep -f "[b]ench-(updated(AA|RD)|diagUS|exact|rdgate)[.a-z0-9]*\.sh" >/dev/null 2>&1; then q=0; n=$((n+1)); [ $((n % 40)) -eq 1 ] && say "WAIT a measured job of the frozen step is still running $(date +%T)"; else q=$((q+1)); fi
  sleep 15
done
say "IDLE the box is idle $(date +%T)"

# ------------------------------------------------------------------ c. the two benchmark chains, one after the other
run_chain() {  # run_chain NAME SCRIPT LOG MARKER
  say "START $1 $(date +%T)"
  BENCH_SCHED=1 BENCH_PREIDLED=1 BENCH_KEEP_SERVICE_DOWN=1 sh $2; rc=$?
  last=$(tail -n 1 $3 2>/dev/null)
  say "END $1 exit=$rc last-log-line=$last $(date +%T)"
  [ "$last" = "$4" ] || say "ERROR $1 ended without its marker $4 (exit $rc): see $3"
}
# BENCH_ONLY=US or RD runs one chain (a rerun of one benchmark); BENCH_ONLY=EXTRA only the BENCH_EXTRA chain
case "${BENCH_ONLY:-}" in RD|EXTRA) ;; *) run_chain US "$US" $S/bench-updatedAA.log WINBENCHUS_DONE ;; esac
case "${BENCH_ONLY:-}" in US|EXTRA) ;; *) run_chain RD "$RD" $S/bench-updatedRD.log WINBENCHRD_DONE ;; esac
if [ -n "${BENCH_EXTRA:-}" ]; then
  case "${BENCH_ONLY:-}" in US|RD) ;; *)
    case "$(basename "$BENCH_EXTRA")" in
      bench-exact*.sh) run_chain EXTRA "$BENCH_EXTRA" $S/bench-exact.log WINBENCHEXACT_DONE ;;
      bench-rdgate*.sh) run_chain EXTRA "$BENCH_EXTRA" $S/bench-rdgate.log WINBENCHRDGATE_DONE ;;
      *) xl=$S/$(basename "$BENCH_EXTRA" .sh).log; run_chain EXTRA "$BENCH_EXTRA" "$xl" "$(grep -o -m1 'WIN[A-Z]*_DONE' "$BENCH_EXTRA" 2>/dev/null)" ;;
    esac ;;
  esac
elif [ "${BENCH_ONLY:-}" = EXTRA ]; then say "ERROR BENCH_ONLY=EXTRA without BENCH_EXTRA=<script>: nothing run"; fi

# ------------------------------------------------------------------ d. AA goes on
resume
echo BENCHSCHED_DONE >> $L
