#!/bin/sh
# trace-t12.v1.sh - LEADS.md Part T T12: timeline capture to find the CPU wait behind wait_flag_ge (T-021: 31% of kernel time) and the >1 ms gaps (T-020).
# Default OFF (prints the plan).  RUN ONLY WHEN THE BOX IS IDLE; refuses when a busy-pattern process runs; deletes nothing; no sudo.
#   cd /home/dan/clone-of-d/src
#   PROF_CMD='<the engine command, one short request>' STRATA_T12_RUN=1 sh tools/pancake/box_chains/trace-t12.v1.sh
# Mode (TRACE_MODE): "hip" (default) = --kernel-trace --hip-trace --hip-event-trace --hip-graph-trace --memory-copy-trace --marker-trace (event trace needs
#   ROCm 10.1+, graph-node attribution 10.0: an unknown flag is dropped with a WARN after a `rocprofv3 --help` check);  "sys" = --sys-trace (everything:
#   hip + hsa + kernel + memory copy + marker); "rocsys" = `rocsys -- <cmd>` (Systems Profiler, ROCm 10.2 only: needs an installed 10.2, skipped otherwise).
#   TRACE_PFTRACE=1 adds `--output-format csv json pftrace` (Perfetto file, open it at ui.perfetto.dev on the PC; big).
#   Marker ranges: if the engine calls roctxRangePush/Pop (it does not today), --marker-trace shows them; nothing is added to the engine here.
# Other environment: PROF_TAG (default t12), PROF_TIMEOUT (default 1800 s), ROCM, PROF_ROOT (default /home/dan/auto-runs/prof).
# Output: $PROF_ROOT/<date>-<tag>[.n]/prof/..., report.txt (timeline_gaps.py: gaps between kernels, host wait share, wait_flag share), run.log (PROF_T12_DONE).
# The profiler slows the engine (~1.6x): read shares and counts.  A failed capture is logged; the report still runs on whatever files exist.
# OWNER ACTIONS (root): only for rocsys / a 10.2 install (`sudo apt install amdrocm-core10.2-gfx1100`, T1): not done by this script.
HERE=$(dirname "$0")
. "$HERE/prof-common.sh"
MODE=${TRACE_MODE:-hip}
echo "T12 plan (mode $MODE): rocprofv3 --kernel-trace --hip-trace --hip-event-trace --hip-graph-trace --memory-copy-trace --marker-trace -d <new dir> -- \$PROF_CMD ; python3 tools/pancake/timeline_gaps.py <dir>"
need_run T12 || exit 0
newout "${PROF_TAG:-t12}"
refuse_if_busy PROF_T12_DONE
FMT="csv json"
[ "${TRACE_PFTRACE:-0}" = 1 ] && FMT="csv json pftrace"
case "$MODE" in
  sys)
    # shellcheck disable=SC2086
    run_prof PROF_T12_DONE --sys-trace --output-format $FMT || log "capture failed, report still runs"
    ;;
  rocsys)
    if [ -x "$ROCM/bin/rocsys" ]; then
      mkdir -p "$OUT/prof"; log "rocsys -- $PROF_CMD"
      (cd "$OUT/prof" && timeout "${PROF_TIMEOUT:-1800}" "$ROCM/bin/rocsys" -- sh -c "$PROF_CMD" > "$OUT/rocsys.stdout" 2>&1) || log "rocsys failed"
    else
      log "rocsys not installed under $ROCM (ROCm 10.2 only): skipped"
    fi
    ;;
  *)
    OPTS="--kernel-trace --hip-trace --memory-copy-trace"
    for f in hip-event-trace hip-graph-trace marker-trace; do
      if have_flag "--$f"; then OPTS="$OPTS --$f"; else log "WARN --$f not in this rocprofv3: dropped"; fi
    done
    # shellcheck disable=SC2086
    run_prof PROF_T12_DONE $OPTS --output-format $FMT || log "capture failed, report still runs"
    ;;
esac
${PYTHON:-python3} "$HERE/../timeline_gaps.py" "$OUT" > "$OUT/report.txt" 2>&1 || log "timeline_gaps exit $? (see report.txt)"
log "results in $OUT"
echo PROF_T12_DONE >> "$LOG"
