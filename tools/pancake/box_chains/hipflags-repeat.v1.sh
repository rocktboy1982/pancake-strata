#!/bin/sh
# hipflags-repeat.v1.sh - LEADS.md Part S S4: repeatability + VOPD census of the HIP code generation (box job; NOT RUN yet; compile only, no GPU, no engine start).
# Two checks on a build made by hipflags-build.v1.sh (its compile_commands.json):
#   1. UNIT LEVEL (minutes): hip_repeat_check.py compiles N HIP translation units twice with identical flags (device side, gfx1100) and compares the bytes;
#      it counts v_dual_* (VOPD) instructions with llvm-objdump, and, with HF_VARIANT (default "-mllvm -amdgpu-enable-vopd=false"), in a third compile.
#   2. BUILD LEVEL (HF_FULL=1): after two builds of the same tree (HF_TAG=rep-a and HF_TAG=rep-b through hipflags-build.v1.sh) every object file of
#      build-rep-a is compared (cmp) with the same relative path in build-rep-b; the BUILDDIFF lines name the objects that differ between two identical builds.
#   HF_BUILD=<configured build dir> (default $HF_ROOT/build-default), HF_MATCH=<regex of unit files, default rdna_kq|moe_mmq|gr_fused|head_fmt>,
#   HF_LIMIT=6, HF_OUT=$HF_ROOT/repeat-<time>
# Refuses to start while the box is busy.  Deletes nothing.  Exit 0 repeatable, 1 differs, 2 error.  Lines: REPEAT ..., REPEATDONE units=N differ=M, ERROR ...
ROCM=${ROCM:-/opt/rocm/core-10.0}; HF_ROOT=${HF_ROOT:-/mnt/pancake/hipflags}; HERE=$(cd "$(dirname "$0")" && pwd)
HF_BUILD=${HF_BUILD:-$HF_ROOT/build-default}; HF_MATCH=${HF_MATCH:-rdna_kq|moe_mmq|gr_fused|head_fmt}; HF_LIMIT=${HF_LIMIT:-6}
HF_OUT=${HF_OUT:-$HF_ROOT/repeat-$(date +%m%d-%H%M%S)}; HF_VARIANT=${HF_VARIANT:--mllvm -amdgpu-enable-vopd=false}
fail() { echo "ERROR $*"; exit 2; }
[ -f "$HF_BUILD/compile_commands.json" ] || fail "no $HF_BUILD/compile_commands.json (run hipflags-build.v1.sh first)"
command -v python3 >/dev/null 2>&1 || fail "no python3"
if pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[b]ench-(sched|sweep|retest|final|exact|updated)[.a-z0-9]*\.sh|[r]un-queue-us[0-9a-z-]*\.sh" >/dev/null 2>&1; then
  fail "the box is busy: not starting, nothing changed"
fi
mkdir -p "$HF_OUT" || fail "cannot create $HF_OUT"
OBJDUMP=$ROCM/lib/llvm/bin/llvm-objdump; [ -x "$OBJDUMP" ] || OBJDUMP=
python3 "$HERE/hip_repeat_check.py" --compile-commands "$HF_BUILD/compile_commands.json" --out "$HF_OUT" --match "$HF_MATCH" --limit "$HF_LIMIT" \
  --clang "$ROCM/lib/llvm/bin/clang++" ${OBJDUMP:+--objdump "$OBJDUMP"} --variant-flags "$HF_VARIANT"
rc=$?
if [ "${HF_FULL:-0}" = 1 ] && [ $rc -le 1 ]; then
  A=$HF_ROOT/build-rep-a; B=$HF_ROOT/build-rep-b
  [ -d "$A" ] && [ -d "$B" ] || fail "HF_FULL=1 needs both builds first: HF_TAG=rep-a sh hipflags-build.v1.sh ; HF_TAG=rep-b sh hipflags-build.v1.sh"
  nd=0
  for f in $(cd "$A" && find . -name '*.o' | sort); do
    [ -f "$B/$f" ] || { echo "BUILDDIFF missing in rep-b: $f"; nd=$((nd+1)); continue; }
    cmp -s "$A/$f" "$B/$f" || { echo "BUILDDIFF $f"; nd=$((nd+1)); }
  done
  echo "BUILDREPEATDONE differing_objects=$nd"
  [ $nd -eq 0 ] || rc=1
fi
exit $rc
