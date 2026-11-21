#!/bin/sh
# att-t13.v1.sh - LEADS.md Part T T13: Advanced Thread Trace (ATT) of ONE hot kernel with rocprofv3 --att.  gfx1100 support is [unverified]: the script
# FIRST checks (read-only) that this rocprofv3 knows --att and that a trace decoder library is installed, and otherwise exits with a clear message (code 3)
# without starting the engine.  Default OFF (prints the plan).  RUN ONLY WHEN THE BOX IS IDLE; deletes nothing; no sudo.
#   cd /home/dan/clone-of-d/src
#   ATT_KERNEL='rdna_kq' PROF_CMD='<engine command, one short request>' STRATA_T13_RUN=1 sh tools/pancake/box_chains/att-t13.v1.sh
#   ATT_CHECK_ONLY=1 STRATA_T13_RUN=1 sh tools/pancake/box_chains/att-t13.v1.sh     only the availability checks (no engine, no GPU use)
# Environment: ATT_KERNEL (regex of ONE kernel name; required for a run), ATT_ITER (which dispatch, default 5: `--kernel-iteration-range [5,5]`, one dispatch
#   only: an ATT buffer of a whole decode run would be huge), ATT_EXTRA (extra rocprofv3 options, e.g. "--att-target-cu 1 --att-buffer-size 0x6000000":
#   read `rocprofv3 --help` first, the names are [unverified] on 10.0), ATT_DECODER_DIR (directory holding librocprof-trace-decoder.so, default
#   $ROCM/lib and $ROCM/lib/rocprofiler-sdk), PROF_TAG (default t13), PROF_TIMEOUT (default 900 s), ROCM, PROF_ROOT.
# Output: $PROF_ROOT/<date>-<tag>[.n]/ (check.txt, prof/ with the ATT output), run.log (last line PROF_T13_DONE).  Decoding the trace into stall reasons
#   needs the rocprof-trace-decoder Python API: not run here; the raw ATT files are kept for a later look.
# Exit codes: 0 done (or default-off plan), 1 busy or bad arguments, 3 ATT or decoder not available (message in check.txt and run.log).
# OWNER ACTIONS (root): if the decoder is missing, installing it is a package step (`apt-cache search rocprof-trace-decoder`, then sudo apt install): not done here.
HERE=$(dirname "$0")
. "$HERE/prof-common.sh"
echo "T13 plan: check rocprofv3 --help for --att and the decoder library; then rocprofv3 --att --kernel-include-regex \$ATT_KERNEL --kernel-iteration-range [N,N] -d <new dir> -- \$PROF_CMD"
need_run T13 || exit 0
newout "${PROF_TAG:-t13}"
refuse_if_busy PROF_T13_DONE
CHK=$OUT/check.txt
ok=1
if [ ! -x "$ROCPROF" ]; then echo "NO: $ROCPROF not found" >> "$CHK"; ok=0
elif have_flag "--att"; then echo "OK: rocprofv3 lists --att" >> "$CHK"
else echo "NO: this rocprofv3 has no --att option" >> "$CHK"; ok=0; fi
found=
for dir in ${ATT_DECODER_DIR:-$ROCM/lib $ROCM/lib/rocprofiler-sdk}; do
  for f in "$dir"/librocprof-trace-decoder.so* "$dir"/*trace-decoder*.so*; do
    [ -f "$f" ] && { found=$f; break 2; }
  done
done
if [ -n "$found" ]; then echo "OK: decoder library $found" >> "$CHK"
else echo "NO: no rocprof-trace-decoder library under ${ATT_DECODER_DIR:-$ROCM/lib $ROCM/lib/rocprofiler-sdk}" >> "$CHK"; ok=0; fi
echo "NOTE: gfx1100 ATT support is not stated in the ROCm 10 notes; a run may still fail on the device (see rocprof.stdout)" >> "$CHK"
cat "$CHK" | while read -r l; do log "$l"; done
if [ "$ok" != 1 ]; then
  log "ATT not available: not running the engine (see check.txt)"; echo PROF_T13_DONE >> "$LOG"; exit 3
fi
if [ "${ATT_CHECK_ONLY:-0}" = 1 ]; then log "check only: done"; echo PROF_T13_DONE >> "$LOG"; exit 0; fi
[ -n "$ATT_KERNEL" ] || { log "ERROR ATT_KERNEL (kernel name regex) is not set"; echo PROF_T13_DONE >> "$LOG"; exit 1; }
OPTS="--att"
if have_flag "--kernel-include-regex"; then OPTS="$OPTS --kernel-include-regex $ATT_KERNEL"; else log "WARN no --kernel-include-regex: ATT of EVERY kernel is too big, stopping"; echo PROF_T13_DONE >> "$LOG"; exit 3; fi
if have_flag "--kernel-iteration-range"; then OPTS="$OPTS --kernel-iteration-range [${ATT_ITER:-5},${ATT_ITER:-5}]"; else log "WARN no --kernel-iteration-range: every dispatch of the kernel is traced"; fi
# shellcheck disable=SC2086
PROF_TIMEOUT=${PROF_TIMEOUT:-900} run_prof PROF_T13_DONE $OPTS $ATT_EXTRA || log "ATT run failed (partial files kept)"
log "results in $OUT"
echo PROF_T13_DONE >> "$LOG"
