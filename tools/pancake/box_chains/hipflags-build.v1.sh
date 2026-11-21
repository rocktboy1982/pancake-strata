#!/bin/sh
# hipflags-build.v1.sh - LEADS.md Part S S4 / Part T T8: build the engine with ONE compiler-flag variant into its OWN build directory
# (box job; NOT RUN yet: written on the Windows PC, never compiled or executed there).  No "bench-", "abtest" or "win-chain" in the name; the
# cmake / ninja / clang children it starts are seen by every chain's busy(), so it REFUSES to start while a measured job or a build runs.
#   HF_TAG=<name>            required: the build lands in $HF_ROOT/build-<name> (same tag = incremental rebuild)
#   HF_VOPD=default|off|on   STRATA_HIP_VOPD (default: default = no flag, the unchanged build)
#   HF_EXTRA="..."           STRATA_HIP_EXTRA_FLAGS (default empty); HF_EXTRA_SCOPE=mmq|all
#   HF_SRC=<tree>            source tree (default /mnt/pancake/hipflags/src: a committed tree synced with `tar xmf` semantics, mtimes reset, else ninja keeps stale objects)
#   HF_ROOT=/mnt/pancake/hipflags   (the root disk is full: builds stay on /mnt/pancake)
#   HF_CMAKE_EXTRA="-DSTRATA_PREFILL_MMQ=ON"   more cmake -D options (the prompt MMQ library is where HF_EXTRA applies at scope mmq)
#   JOBS=8
# Rules: never deletes anything; an existing $BUILD/strata is copied to $BUILD/strata.prev.<time> before the build; any failed step prints ERROR and
# exits 1; stops when /mnt/pancake has < 2 GB free.  Log: $HF_ROOT/build-<tag>.log; last line HIPFLAGS_BUILD_DONE rc=<n>.  Start (later, box idle):
#   HF_TAG=default sh hipflags-build.v1.sh ; HF_TAG=default-b sh hipflags-build.v1.sh ; HF_TAG=vopd-off HF_VOPD=off sh hipflags-build.v1.sh
ROCM=${ROCM:-/opt/rocm/core-10.0}; CM=${CM:-/home/linuxbrew/.linuxbrew/bin/cmake}; NINJA=${NINJA:-/home/dan/clone-of-d/src/Strata/.venv/bin/ninja}
HF_ROOT=${HF_ROOT:-/mnt/pancake/hipflags}; HF_SRC=${HF_SRC:-$HF_ROOT/src}; JOBS=${JOBS:-8}
HF_VOPD=${HF_VOPD:-default}; HF_EXTRA=${HF_EXTRA:-}; HF_EXTRA_SCOPE=${HF_EXTRA_SCOPE:-mmq}; HF_CMAKE_EXTRA=${HF_CMAKE_EXTRA:--DSTRATA_PREFILL_MMQ=ON}
fail() { echo "ERROR $*"; [ -n "${LOG:-}" ] && echo "HIPFLAGS_BUILD_DONE rc=1" >> "$LOG"; exit 1; }
[ -n "${HF_TAG:-}" ] || { echo "ERROR HF_TAG is required"; exit 1; }
case "$HF_TAG" in *[!A-Za-z0-9._-]*) echo "ERROR HF_TAG may hold letters, digits, . _ - only"; exit 1;; esac
case "$HF_VOPD" in default|off|on) ;; *) echo "ERROR HF_VOPD must be default, off or on"; exit 1;; esac
case "$HF_EXTRA_SCOPE" in mmq|all) ;; *) echo "ERROR HF_EXTRA_SCOPE must be mmq or all"; exit 1;; esac
mkdir -p "$HF_ROOT" || { echo "ERROR cannot create $HF_ROOT"; exit 1; }
BUILD=$HF_ROOT/build-$HF_TAG; LOG=$HF_ROOT/build-$HF_TAG.log
say() { echo "$*" | tee -a "$LOG"; }
[ -f "$HF_SRC/CMakeLists.txt" ] || fail "no source tree at $HF_SRC (sync a committed tree there first, tar xmf)"
[ -f "$HF_SRC/cmake/strata_hip_flags.cmake" ] || fail "$HF_SRC has no cmake/strata_hip_flags.cmake (tree is older than lead/S4-T8)"
for f in "$CM" "$NINJA" "$ROCM/lib/llvm/bin/clang++"; do [ -x "$f" ] || fail "missing $f"; done
FREE=$(df -Pk /mnt/pancake 2>/dev/null | awk 'NR==2{print $4}'); [ "${FREE:-0}" -ge 2097152 ] || fail "/mnt/pancake has ${FREE:-0} KB free (< 2 GB); nothing was deleted"
if pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[b]ench-(sched|sweep|retest|final|exact|updated)[.a-z0-9]*\.sh|[r]un-queue-us[0-9a-z-]*\.sh" >/dev/null 2>&1; then
  fail "the box is busy (a measured job, a build or a queue runs): not starting, nothing changed"
fi
mkdir -p "$BUILD"
say "HIPFLAGS_BUILD start $(date +%T) tag=$HF_TAG vopd=$HF_VOPD extra='$HF_EXTRA' scope=$HF_EXTRA_SCOPE src=$HF_SRC build=$BUILD"
[ -x "$BUILD/strata" ] && cp -p "$BUILD/strata" "$BUILD/strata.prev.$(date +%H%M%S)" && say "backup of the previous binary"
env HIP_PLATFORM=amd ROCM_PATH=$ROCM HIP_PATH=$ROCM "$CM" -S "$HF_SRC" -B "$BUILD" -G Ninja -DCMAKE_MAKE_PROGRAM="$NINJA" -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=$ROCM \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DSTRATA_HIP_VOPD="$HF_VOPD" -DSTRATA_HIP_EXTRA_FLAGS="$HF_EXTRA" -DSTRATA_HIP_EXTRA_FLAGS_SCOPE="$HF_EXTRA_SCOPE" \
  $HF_CMAKE_EXTRA >> "$LOG" 2>&1 || fail "configure failed (see $LOG)"
nice -n 10 env HIP_PLATFORM=amd ROCM_PATH=$ROCM HIP_PATH=$ROCM "$CM" --build "$BUILD" --target strata -j"$JOBS" >> "$LOG" 2>&1 || fail "build failed (see $LOG); the previous binary is untouched"
[ -x "$BUILD/strata" ] || fail "build ended without $BUILD/strata"
say "binary $BUILD/strata sha256 $(sha256sum "$BUILD/strata" | cut -d' ' -f1)"
echo "HIPFLAGS_BUILD_DONE rc=0" >> "$LOG"; echo "HIPFLAGS_BUILD_DONE rc=0"
