#!/bin/sh
# build_updated_rdna3.sh - the optimised build of updated-strata-rdna3 for the RX 7900 XT (gfx1100), ROCm /opt/rocm/core-10.0, clang 23, gcc 15, Ninja.
# The baseline it is measured against is the PLAIN Strata-rdna3 build with its own documented flags (README.md "Manual build"): Release, HIP on,
# STRATA_PREFILL_MMQ, STRATA_MMQ_KQUANTS, gfx1100.  Every flag below is either one of those, or listed with the reason it is on / off.
#
#   tools/pancake/build_updated_rdna3.sh SRC BUILD_DIR [--print-flags] [--configure-only] [--target T]...   (default target: strata)
#   BACKEND=cuda  tools/pancake/build_updated_rdna3.sh SRC BUILD_DIR    CUDA stays buildable: STRATA_ENABLE_CUDA=ON, CMAKE_CUDA_ARCHITECTURES (default 75)
#   Environment: WINDOW_MAX_T (default 16), ROCM (default /opt/rocm/core-10.0), GGML_DIR (offline llama.cpp checkout), JOBS (default 8), MARCH_NATIVE=1, LTO=1, EXTRA_HIP_FLAGS, EXTRA_CXX_FLAGS
#
# Flags and why
#   CMAKE_BUILD_TYPE=Release            -O3 -DNDEBUG for host code (CMake's Release); the HIP device code is -O3 too.  Same as the plain build, stated.
#   CMAKE_HIP_ARCHITECTURES=gfx1100     the only GPU of the box: one code object, no fat binary for other arches (compile time and binary size; the code
#                                       is identical to a multi-arch build for gfx1100).
#   STRATA_ENABLE_HIP=ON, _CUDA=OFF     the HIP backend (wave32); CUDA off for the box build (BACKEND=cuda switches it).
#   STRATA_PREFILL_MMQ / STRATA_MMQ_KQUANTS / STRATA_MMQ_TQ_OVERLAY / STRATA_NATIVE_EXPERTS=ON
#                                       the prompt kernels and the native (IQ) expert path the daily model needs; same as the plain build.
#   -mtune=native (host CXX)            schedules for the build host's CPU (Ryzen 5700G, Zen 3) and changes NO instruction set: output bit for bit
#                                       the same as without it.  The box builds and runs on the same machine.
#   -march=native                       OFF by default (MARCH_NATIVE=1 turns it on): it lets gcc contract a*b+c into FMA in the C++ CPU code, which changes
#                                       rounding of the CPU expert rows ("exact" arms stop being bitwise equal) and the per-file kernels already carry their own
#                                       ISA flags and run-time CPU checks.  Only a measured gain on the box would justify it.
#   LTO / IPO                           OFF by default (LTO=1 turns on CMAKE_INTERPROCEDURAL_OPTIMIZATION): generate.cpp is 18k lines and the box has little
#                                       disk and memory to spare during a chain; not measured to be faster, so not default.
#   fast-math                           OFF: -ffast-math / -fno-honor-nans change rounding, NaN handling (the sampler rule) and exactness claims.
#   hipBLASLt / rocBLAS                 no build-time switch exists in this tree: the lever is the run-time tuning table STRATA_HIPBLASLT_TUNING
#                                       (tools/hip/gfx1100-hipblaslt-100401.txt), a diagnostic candidate of tools/pancake/flag_diag.sh, not a build flag.
#   STRATA_AMD_COMMUNITY_TUNING=ON      ported from the pancake tree (README-PANCAKE.md 2.16) and passed as the pancake box build passes it, so both builds
#                                       match.  It is the CMake default anyway; OFF would only add STRATA_AMD_COMMUNITY_DOT_OFF, which is read inside a
#                                       __gfx1010__ branch of the device code and by the gfx1010 log line: the gfx1100 code object is the same either way.
#   STRATA_ROCWMMA_INCLUDE_DIR          passed when rocWMMA headers are found ($ROCM/include/rocwmma or ROCWMMA_INCLUDE): optional gfx1100 QSA selection kernels.
#   STRATA_GGML_DIR                     the pinned llama.cpp checkout the box already has (no network during a chain).
#   STRATA_WINDOW_MAX_T=16              verify windows of up to 16 rows (ported: upstream 9d3c8c366 + 03627326d; "auto" is 16 for HIP too, stated explicitly):
#                                       --lookup-max-t 12 runs its 9..12-row windows as one group and STRATA_VERIFY_ECAP_MIN_ROWS=9 can open (pancake
#                                       T-126/T-127, T-588); the pancake daily build is the same (STRATA_WINDOW_MAX_T=auto, HIP => 16).  The plain
#                                       Strata-rdna3 build has no such option (its windows are 8 rows).  Windows of up to 8 rows run the same kernels.
#   STRATA_WERROR=OFF, STRATA_BUILD_TESTS=ON   warnings do not stop a measured build; the ctests (verify_ecap_test, sampler_parity, ...) exist for the diagnostics.
set -u
SRC=${1:-}; BUILD=${2:-}
[ -n "$SRC" ] && [ -n "$BUILD" ] || { sed -n 2,5p "$0" | sed 's/^# \{0,1\}//'; exit 2; }
shift 2
PRINT=0; CONFIGURE_ONLY=0; TARGETS=""
while [ $# -gt 0 ]; do
  case $1 in
    --print-flags) PRINT=1 ;;
    --configure-only) CONFIGURE_ONLY=1 ;;
    --target) shift; TARGETS="$TARGETS $1" ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$TARGETS" ] || TARGETS=" strata"
ROCM=${ROCM:-/opt/rocm/core-10.0}
CM=${CMAKE:-$(command -v cmake || echo /home/linuxbrew/.linuxbrew/bin/cmake)}
NINJA=${NINJA:-$(command -v ninja || echo "")}
GGML_DIR=${GGML_DIR:-/home/dan/clone-of-d/build-int/_deps/strata_llamacpp-src}
JOBS=${JOBS:-8}
CXXF="-O3 -DNDEBUG -mtune=native ${EXTRA_CXX_FLAGS:-}"
[ "${MARCH_NATIVE:-0}" = 1 ] && CXXF="$CXXF -march=native"
set -- -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CXX_FLAGS_RELEASE=$CXXF" -DSTRATA_WERROR=OFF -DSTRATA_BUILD_TESTS=ON \
       -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_TQ_OVERLAY=ON -DSTRATA_WINDOW_MAX_T=${WINDOW_MAX_T:-16} \
       -DSTRATA_AMD_COMMUNITY_TUNING=ON
[ -d "$GGML_DIR" ] && set -- "$@" "-DSTRATA_GGML_DIR=$GGML_DIR"
[ -n "$NINJA" ] && set -- "$@" "-DCMAKE_MAKE_PROGRAM=$NINJA"
[ "${LTO:-0}" = 1 ] && set -- "$@" -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
if [ "${BACKEND:-hip}" = cuda ]; then
  set -- "$@" -DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF "-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-75}"
else
  RW=${ROCWMMA_INCLUDE:-}
  [ -z "$RW" ] && [ -d "$ROCM/include/rocwmma" ] && RW=$ROCM/include
  set -- "$@" -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1100 \
       "-DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++" "-DCMAKE_PREFIX_PATH=$ROCM" "-DCMAKE_HIP_FLAGS_RELEASE=-O3 -DNDEBUG ${EXTRA_HIP_FLAGS:-}"
  [ -n "$RW" ] && set -- "$@" "-DSTRATA_ROCWMMA_INCLUDE_DIR=$RW"
  export HIP_PLATFORM=amd ROCM_PATH=$ROCM HIP_PATH=$ROCM PATH=$ROCM/bin:$ROCM/lib/llvm/bin:$PATH
fi
if [ $PRINT = 1 ]; then echo "$CM $*"; exit 0; fi
"$CM" "$@" || { echo "ERROR: configure failed (exit $?)" >&2; exit 1; }
[ $CONFIGURE_ONLY = 1 ] && exit 0
# shellcheck disable=SC2086
nice -n 10 "$CM" --build "$BUILD" -j"$JOBS" --target $TARGETS -- -k 0 || { echo "ERROR: build failed (exit $?)" >&2; exit 1; }
