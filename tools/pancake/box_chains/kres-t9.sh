#!/bin/sh
# kres-t9.sh v1 - LEADS.md Part T T9 + T10 input: build the engine once for gfx1100 with the kernel-resource remarks, extract the device code object, dump
# its notes and disassembly, and run kernel_resources_obj.py + isa_audit.py on them.  Default OFF: without STRATA_T9_RUN=1 it only prints the plan.
# RUN ONLY WHEN THE BOX IS IDLE (a build uses all cores): the script refuses when a busy-pattern process runs (it never kills anything).
#   cd /home/dan/clone-of-d/src && STRATA_T9_RUN=1 sh tools/pancake/box_chains/kres-t9.sh
# Environment: SRC (default $PWD), ROCM (default /opt/rocm/core-10.0), OUT (default /home/dan/auto-runs/kres/<date>; never deleted or overwritten: a numbered
#   suffix is used), BUILD (default $OUT/build: a NEW build directory, existing builds are not touched), JOBS (default 8).
# Writes only under $OUT.  Failure-safe: every step logs and the next runs; the last line of $OUT/kres.log is KRES_T9_DONE.  No sudo.
# OWNER ACTIONS (root): none.
S=${SRC:-$PWD}
ROCM=${ROCM:-/opt/rocm/core-10.0}
D=$(date +%Y-%m-%d)
OUT=${OUT:-/home/dan/auto-runs/kres/$D}
JOBS=${JOBS:-8}
if [ "$STRATA_T9_RUN" != 1 ]; then
  echo "PLAN (STRATA_T9_RUN=1 to run, box idle only): cmake -S $S -B \$OUT/build -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_FLAGS=-Rpass-analysis=kernel-resource-usage"
  echo "  cmake --build (log kept); roc-obj-extract the device ELF; llvm-readelf --notes; llvm-objdump -d --mcpu=gfx1100"
  echo "  python3 tools/pancake/kernel_resources_obj.py --notes ... --log ...; python3 tools/pancake/isa_audit.py ..."
  exit 0
fi
n=0; O=$OUT; while [ -e "$O" ]; do n=$((n+1)); O=$OUT.$n; done; OUT=$O
mkdir -p "$OUT" || { echo "cannot create $OUT"; exit 1; }
L=$OUT/kres.log
log() { echo "$(date +%H:%M:%S) $*" | tee -a "$L"; }
if [ "${KRES_SKIP_BUSY:-0}" != 1 ] && pgrep -f 'abtest|bench-|strata --serve|serve/server.py|quality_gate|task_gate|ninja|clang|ctest|replay_turns' >/dev/null 2>&1; then
  log "ERROR box busy (a busy-pattern process runs): not building"; echo KRES_T9_DONE >> "$L"; exit 1
fi
B=${BUILD:-$OUT/build}
log "configure $B"
cmake -S "$S" -B "$B" -G Ninja -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_FLAGS="-Rpass-analysis=kernel-resource-usage" > "$OUT/configure.log" 2>&1 || log "configure FAILED (see configure.log)"
log "build (log $OUT/build.log)"
cmake --build "$B" -j "$JOBS" > "$OUT/build.log" 2>&1 || log "build FAILED (partial log kept)"
BIN=$(find "$B" -maxdepth 2 -type f -name 'strata' 2>/dev/null | head -1)
[ -n "$BIN" ] || BIN=$(find "$B" -maxdepth 2 -type f -name '*.so' 2>/dev/null | head -1)
log "binary: ${BIN:-none}"
if [ -n "$BIN" ]; then
  "$ROCM/bin/roc-obj-extract" -o "$OUT/co" "$BIN" > "$OUT/extract.log" 2>&1 || log "roc-obj-extract failed (see extract.log)"
  for co in "$OUT"/co/*gfx1100* "$OUT"/co/*.co; do
    [ -f "$co" ] || continue
    b=$(basename "$co")
    "$ROCM/lib/llvm/bin/llvm-readelf" --notes "$co" > "$OUT/$b.notes.txt" 2>&1 || log "readelf failed on $b"
    "$ROCM/lib/llvm/bin/llvm-objdump" -d --mcpu=gfx1100 "$co" > "$OUT/$b.dis.txt" 2>&1 || log "objdump failed on $b"
    python3 "$S/tools/pancake/kernel_resources_obj.py" --notes "$OUT/$b.notes.txt" --log "$OUT/build.log" --csv "$OUT/$b.resources.csv" > "$OUT/$b.resources.txt" 2>&1 || log "kernel_resources_obj exit $?"
    python3 "$S/tools/pancake/isa_audit.py" "$OUT/$b.dis.txt" --csv "$OUT/$b.isa.csv" > "$OUT/$b.isa.txt" 2>&1 || log "isa_audit exit $?"
  done
fi
log "results in $OUT"
echo KRES_T9_DONE >> "$L"
