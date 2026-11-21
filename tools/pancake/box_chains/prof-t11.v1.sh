#!/bin/sh
# prof-t11.v1.sh - LEADS.md Part T T11: rocprofv3 kernel trace + PMC counters (gfx1100) with optional kernel replay (beta, ROCm 10.1+), then the report.
# Default OFF (prints the plan).  RUN ONLY WHEN THE BOX IS IDLE; refuses when a busy-pattern process runs; deletes nothing; no sudo.
#   cd /home/dan/clone-of-d/src
#   PROF_CMD='<the engine command, one short request>' STRATA_T11_RUN=1 sh tools/pancake/box_chains/prof-t11.v1.sh
# Environment (all optional except PROF_CMD):
#   PROF_PMC      counters, space separated (default "SQ_WAVES GRBM_GUI_ACTIVE"); the names must exist: `rocprofv3-avail list --pmc` first (read-only look)
#   PROF_REPLAY   1 = add `--replay-mode kernel --kernel-replay-beta-enabled` (ALL counter groups in one run; ROCm 10.1+, beta; on 10.0.0 the script
#                 checks `rocprofv3 --help` and says so instead of passing an unknown flag).  The replay snapshots device memory: a 20 GB engine may not
#                 replay faithfully -> default 0 = several short runs, one counter group each (PROF_GROUPS="SQ_WAVES GRBM_GUI_ACTIVE;SQ_INSTS_VALU ..." splits by ';')
#   PROF_GROUPS   ';'-separated counter groups, one rocprofv3 run each (default: PROF_PMC as one group)
#   PROF_TAG      tag of the output directory (default t11), PROF_TIMEOUT seconds per run (default 1800), ROCM, PROF_ROOT (default /home/dan/auto-runs/prof)
# Output: $PROF_ROOT/<date>-<tag>[.n]/prof*/ (kernel_trace.csv, counter_collection.csv, ...), report.txt (prof_report.py), run.log (last line PROF_T11_DONE).
# A failed run is logged and the next group still runs.  The profiler slows the engine (~1.6x with --kernel-trace): read shares, not absolute speed.
# OWNER ACTIONS (root): none expected.  If rocprofv3 fails with a permission error on /dev/kfd or perf counters, the fix needs root (add the user to the
#   render/video groups, or `echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid`): ask the owner, do not run it from this script.
HERE=$(dirname "$0")
. "$HERE/prof-common.sh"
echo "T11 plan: rocprofv3 --kernel-trace --stats --pmc <counters> [replay] -d <new dir> -- \$PROF_CMD ; python3 tools/pancake/prof_report.py <dir>"
need_run T11 || exit 0
newout "${PROF_TAG:-t11}"
refuse_if_busy PROF_T11_DONE
GROUPS_LIST=${PROF_GROUPS:-${PROF_PMC:-SQ_WAVES GRBM_GUI_ACTIVE}}
REPLAY=
if [ "${PROF_REPLAY:-0}" = 1 ]; then
  if have_flag kernel-replay-beta-enabled; then REPLAY="--replay-mode kernel --kernel-replay-beta-enabled"
  else log "WARN kernel replay flag not in this rocprofv3 (needs ROCm 10.1+): running one group per run instead"; fi
fi
i=0
OLDIFS=$IFS; IFS=';'
for grp in $GROUPS_LIST; do
  IFS=$OLDIFS
  i=$((i+1))
  OUTBASE=$OUT
  log "group $i: $grp ${REPLAY:+(replay)}"
  # shellcheck disable=SC2086
  OUT=$OUTBASE/g$i; mkdir -p "$OUT"
  # run_prof writes to $OUT/prof
  # shellcheck disable=SC2086
  run_prof PROF_T11_DONE --kernel-trace --stats --pmc $grp $REPLAY --output-format csv json || log "group $i failed, continuing"
  OUT=$OUTBASE
  IFS=';'
done
IFS=$OLDIFS
${PYTHON:-python3} "$HERE/../prof_report.py" "$OUT" > "$OUT/report.txt" 2>&1 || log "prof_report exit $? (see report.txt)"
log "results in $OUT"
echo PROF_T11_DONE >> "$LOG"
