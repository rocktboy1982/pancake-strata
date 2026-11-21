#!/bin/sh
# hipflags-probe.v1.sh - LEADS.md Part T T8: which candidate compiler flags does THIS box's HIP compiler accept? (box job, NOT RUN yet; one tiny compile per
# flag, no engine, no GPU use; every candidate name is UNVERIFIED for LLVM 24 and this script is how they get verified.)  Compiles an empty device kernel for
# gfx1100 once per candidate and prints ACCEPT / REJECT / WARN (accepted with a diagnostic such as "unused argument" or "unknown"), plus the compiler version.
# Writes only into $HF_OUT (default a mktemp dir under /mnt/pancake/hipflags).  More candidates: HF_CANDIDATES="flag1|tok flag2" (one group per word, | joins tokens).
ROCM=${ROCM:-/opt/rocm/core-10.0}; CXX=${CXX:-$ROCM/lib/llvm/bin/clang++}; HF_ROOT=${HF_ROOT:-/mnt/pancake/hipflags}
[ -x "$CXX" ] || { echo "ERROR no compiler at $CXX"; exit 2; }
mkdir -p "$HF_ROOT" 2>/dev/null
OUT=${HF_OUT:-$(mktemp -d "$HF_ROOT/probe.XXXXXX")} || { echo "ERROR no scratch dir"; exit 2; }
if pgrep -f "[n]inja|[a]btest.py|[s]trata --serve|[s]erve/server.py|[q]uality_gate.py|[t]ask_gate.py|[r]eplay_turns.py|[c]test" >/dev/null 2>&1; then echo "ERROR the box is busy: not starting"; exit 2; fi
printf '#include <hip/hip_runtime.h>\n__global__ void k(float* p){ p[threadIdx.x] = (float)threadIdx.x; }\n' > "$OUT/empty.hip"
"$CXX" --version | head -2
DEF="-mllvm|-amdgpu-enable-vopd=false -mllvm|-amdgpu-enable-vopd=true -mcumode -mno-cumode -mllvm|-amdgpu-sched-strategy=max-ilp -mllvm|-amdgpu-sched-strategy=max-memory-clause -foffload-lto -foffload-lto|-fgpu-rdc --offload-compress"
for c in ${HF_CANDIDATES:-$DEF}; do
  flags=$(echo "$c" | tr '|' ' ')
  if "$CXX" -x hip --offload-arch=gfx1100 --offload-device-only -c -O3 $flags -o "$OUT/probe.o" "$OUT/empty.hip" > "$OUT/probe.txt" 2>&1; then
    if grep -qiE "warning|unknown|unused|ignor|not supported" "$OUT/probe.txt"; then
      echo "WARN   $flags :: $(grep -iE 'warning|unknown|unused|ignor|not supported' "$OUT/probe.txt" | head -1)"
    else echo "ACCEPT $flags"; fi
  else
    echo "REJECT $flags :: $(grep -iE 'error|unknown|unsupported' "$OUT/probe.txt" | head -1)"
  fi
done
echo "HIPFLAGS_PROBE_DONE (files in $OUT)"
