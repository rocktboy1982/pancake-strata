#!/bin/sh
# build_updated.sh - the optimised HIP (or CUDA) build of the updated-strata branch for one machine class.
#
#   tools/pancake/build_updated.sh [SRC_DIR] [BUILD_DIR]       SRC_DIR defaults to this checkout, BUILD_DIR to SRC_DIR/build-updated
#   environment: BACKEND=hip (default) | cuda      ROCM=/opt/rocm/core-10.0      GFX=gfx1100      NINJA=ninja      JOBS=8
#                LTO=1 (off by default)            TESTS=1 (build the tests, off by default)     CUDA_ARCH=75
# The box chain (tools/pancake/box_chains/bench-updatedAA.sh) calls it with BUILD_DIR outside / (disk / is nearly full).
# Exit 0 only when ninja built the `strata` target and the binary exists; every failure prints ERROR and exits non-zero.
#
# Baseline: what box chain AA configures (redo-chainAA.sh vbuild: Release, gfx1100, clang++ of ROCm core-10.0, Ninja,
# STRATA_AMD_COMMUNITY_TUNING=ON, STRATA_WINDOW_MAX_T=auto); cmake's cache of that box shows CMAKE_HIP_FLAGS_RELEASE and
# CMAKE_CXX_FLAGS_RELEASE = "-O3 -DNDEBUG".  Each flag and why (the code reference is in engine/):
#   -DCMAKE_BUILD_TYPE=Release, -O3 -DNDEBUG (HIP and C++)     engine/CMakeLists.txt defaults to Release; stated so a stale cache cannot
#                                                               leave -O0 (cmake/hip_backend.cmake describes that failure on an RX 6800)
#   -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF            the two backends are exclusive (CMakeLists.txt: FATAL_ERROR if both)
#   -DCMAKE_HIP_ARCHITECTURES=gfx1100 (only)                    no second target: smaller binary, faster link; the runtime device check
#                                                               (src/core/device.cu) stops a binary on a card it has no code for
#   -DSTRATA_AMD_COMMUNITY_TUNING=ON                            compiles the community SDWA dot (option default ON; explicit so it cannot drift)
#   -DSTRATA_WINDOW_MAX_T=16                                    the HIP default ("auto" = 16 on HIP): the 16-row verify window the
#                                                               RDNA kernels (rdna_mmvq.cu, 4..16 columns) and --spec 4 windows need
#   -DSTRATA_NATIVE_EXPERTS=ON, STRATA_OWN_CPU_QUANT default   CPU expert rows; GGML_NATIVE=ON (the non-portable default,
#                                                               cmake/strata_ggml.cmake) already compiles ggml-cpu with -march=native
#   host -march=native NOT added globally                       cmake/backends/cpu.cmake and quant.cmake keep AVX-only source files with
#                                                               AVX2/FMA explicitly off: a global flag would leak wider code into them
#   no -ffast-math / --use_fast_math for HIP                    the tree sets -ffp-contract=off on elementwise.cu and quantize_act.cu
#                                                               on purpose (Q8_K rounding ties); fast-math would change numerics, so the
#                                                               default stays exact. (The CUDA branch already uses --use_fast_math on the
#                                                               native_* files; unchanged.)
#   hipBLASLt                                                   found by find_package(hipblaslt) in hip_backend.cmake; the gfx1100 solution table
#                                                               is a RUN-TIME setting (STRATA_HIPBLASLT_TUNING, set by the tuned flags
#                                                               for gfx1100 / ROCm 10.x), not a build flag
#   LTO=1 -> -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON            opt-in only: the HIP device code is compiled per file; host LTO is untested here and
#                                                               not claimed faster. Build with LTO=1 into another BUILD_DIR to compare.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=${1:-$(cd "$HERE/../.." && pwd)}
BUILD=${2:-$SRC/build-updated}
BACKEND=${BACKEND:-hip}
ROCM=${ROCM:-/opt/rocm/core-10.0}
GFX=${GFX:-gfx1100}
JOBS=${JOBS:-8}
NINJA=${NINJA:-$(command -v ninja || true)}
CM=${CMAKE:-$(command -v cmake || echo /home/linuxbrew/.linuxbrew/bin/cmake)}
fail() { echo "ERROR build_updated: $*" >&2; exit 1; }
[ -f "$SRC/engine/CMakeLists.txt" ] || fail "no engine/CMakeLists.txt under $SRC"
[ -n "$NINJA" ] || fail "ninja not found (set NINJA=)"
mkdir -p "$BUILD" || fail "cannot create $BUILD"

OPT="-O3 -DNDEBUG"
set -- -S "$SRC/engine" -B "$BUILD" -G Ninja "-DCMAKE_MAKE_PROGRAM=$NINJA" -DCMAKE_BUILD_TYPE=Release \
       "-DCMAKE_CXX_FLAGS_RELEASE=$OPT" -DSTRATA_AMD_COMMUNITY_TUNING=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_WINDOW_MAX_T=16 \
       -DSTRATA_BUILD_TESTS="$([ "${TESTS:-0}" = 1 ] && echo ON || echo OFF)"
case "$BACKEND" in
  hip)  set -- "$@" -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF "-DCMAKE_HIP_ARCHITECTURES=$GFX" "-DCMAKE_HIP_FLAGS_RELEASE=$OPT" \
               "-DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++" "-DCMAKE_PREFIX_PATH=$ROCM" ;;
  cuda) set -- "$@" -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-75}" ;;
  *)    fail "BACKEND must be hip or cuda (got $BACKEND)" ;;
esac
[ "${LTO:-0}" = 1 ] && set -- "$@" -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
echo "build_updated: $CM $*"
"$CM" "$@" > "$BUILD/configure.log" 2>&1 || { tail -20 "$BUILD/configure.log" >&2; fail "configure failed (log $BUILD/configure.log)"; }
"$CM" --build "$BUILD" -j"$JOBS" --target strata > "$BUILD/build.log" 2>&1
rc=$?
[ $rc -eq 0 ] || { grep -m8 -E "error:|undefined|CMake Error" "$BUILD/build.log" >&2; fail "build exit $rc (log $BUILD/build.log)"; }
[ -x "$BUILD/strata" ] || fail "build exit 0 but $BUILD/strata is missing"
echo "build_updated: ok $BUILD/strata"
