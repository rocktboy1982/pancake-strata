#!/bin/sh
# prof-common.sh v1 - sourced by prof-t11.v1.sh, trace-t12.v1.sh, att-t13.v1.sh (LEADS.md Part T T11 / T12 / T13).  Not run on its own.
# Provides: ROCM, ROCPROF, newout TAG (a NEW numbered directory under /home/dan/auto-runs/prof, never reuses or deletes one), log, refuse_if_busy,
# need_run NAME (default OFF gate: the script only prints its plan unless STRATA_<NAME>_RUN=1), have_flag FLAG (is FLAG in `rocprofv3 --help`).
# RUN ONLY WHEN THE BOX IS IDLE: refuse_if_busy exits when a measured job, build or server runs (it never kills anything).  No sudo.
ROCM=${ROCM:-/opt/rocm/core-10.0}
ROCPROF=${ROCPROF:-$ROCM/bin/rocprofv3}
PROF_ROOT=${PROF_ROOT:-/home/dan/auto-runs/prof}
BUSY_RX='abtest|bench-|strata --serve|serve/server.py|quality_gate|task_gate|ninja|clang|ctest|replay_turns'

need_run() {   # $1 = T11 | T12 | T13
  eval "v=\${STRATA_$1_RUN:-0}"
  if [ "$v" != 1 ]; then
    echo "default OFF: nothing started.  Set STRATA_$1_RUN=1 (box idle, PROF_CMD set) to run.  Plan below:"
    return 1
  fi
  return 0
}

newout() {     # $1 = tag -> sets OUT (a new directory), LOG
  d=$(date +%Y-%m-%d)
  base=$PROF_ROOT/$d-$1
  n=0; OUT=$base
  while [ -e "$OUT" ]; do n=$((n+1)); OUT=$base.$n; done
  mkdir -p "$OUT" || { echo "cannot create $OUT"; exit 1; }
  LOG=$OUT/run.log
}

log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

refuse_if_busy() {   # $1 = marker word for the last log line
  if [ "${PROF_SKIP_BUSY:-0}" != 1 ] && pgrep -f "$BUSY_RX" > /dev/null 2>&1; then
    log "ERROR box busy (a busy-pattern process runs): not profiling"; echo "$1" >> "$LOG"; exit 1
  fi
}

have_flag() { LD_LIBRARY_PATH=$ROCM/lib:$LD_LIBRARY_PATH "$ROCPROF" --help 2>&1 | grep -q -e "$1"; }

run_prof() {   # $1 = marker; rest = rocprofv3 options (before the --); PROF_CMD is the engine command line
  m=$1; shift
  [ -n "$PROF_CMD" ] || { log "ERROR PROF_CMD is not set (the engine command to profile)"; echo "$m" >> "$LOG"; exit 1; }
  [ -x "$ROCPROF" ] || { log "ERROR $ROCPROF not found"; echo "$m" >> "$LOG"; exit 1; }
  log "rocprofv3 $* -d $OUT/prof -- $PROF_CMD"
  # shellcheck disable=SC2086
  LD_LIBRARY_PATH=$ROCM/lib:$LD_LIBRARY_PATH timeout "${PROF_TIMEOUT:-1800}" "$ROCPROF" "$@" -d "$OUT/prof" -- sh -c "$PROF_CMD" > "$OUT/rocprof.stdout" 2>&1
  rc=$?
  log "rocprofv3 exit $rc (124 = timeout; partial files are kept)"
  return $rc
}
