#!/bin/sh
# pancake-strata: serve a model with its measured preset.
#   ./run-pancake.sh [MODEL] [fast|exact|turbo]
# MODEL (default iq2_xs):
#   iq2_xs      Qwen3.8-Flash-Next, ISTA-DASLab IQ2_XS - main engine (the daily model)
#   q2_0        Qwen3.8-Flash-Next, ISTA-DASLab Q2_0   - main engine (Strata's fastest file; basalt's Q5_K head kernels)
#   qwen36      Qwen3.6-35B-A3B, Unsloth UD-Q4_K_XL    - strata-poly (FastStrata's engine)
#   qwen36-q3   Qwen3.6-35B-A3B, Unsloth UD-Q3_K_M     - strata-poly, smaller and faster
#   qwen38-35b  Qwen3.8-35B-A3B-Distill Q4_K_M (empero: Qwen3.6 fine-tuned on Qwen3.8 answers) - strata-poly, the
#               MTP drafting layer borrowed from the Qwen3.6 Q4_K_XL file
# Flash-Next defaults to the fast preset (Speedy's switches: ~13% faster writing after long prompts; it passed the
# token-forcing quality check, docs/PANCAKE_FINDINGS.md 10e).  `exact`: the same answers as plain decoding.
# `turbo` (IQ2_XS only, lossy opt-in): fast + cache-aware routing (STRATA_ROUTE_PRIOR=0.5), +25% decode on the 3060
# but 2.9% of tokens differ from exact decoding (stock noise 1.2-1.3%, no large-margin errors; MOD_PLAN 8.9).
# RX 7900 XT (no NVIDIA card): build-hip and the *-hip presets where they exist - iq2_xs fast (default, ~126 tok/s), exact,
# turbo (~168, lossy); q2_0 fast (~135 tok/s) / turbo: a lower-quality quant (80.6% same top token as Q8_0, docs/QUANTS.md).
# STRATA_DATA (default /home/dan/clone-of-d/Strata-data) and MODELS (default /home/dan/clone-of-d/models) hold the
# files; PORT defaults to 8080.
cd "$(dirname "$0")" || exit 1
REPO=$(pwd)
DATA=${STRATA_DATA:-/home/dan/clone-of-d/Strata-data}
MODELS=${MODELS:-/home/dan/clone-of-d/models}
MODEL=iq2_xs
FAST=-fast
for a in "$@"; do
  case "$a" in
    fast) FAST=-fast ;;
    exact) FAST= ;;
    turbo) FAST=-turbo ;;
    iq2_xs|q2_0|qwen36|qwen36-q3|qwen38-35b) MODEL=$a ;;
    *) echo "usage: $0 [iq2_xs|q2_0|qwen36|qwen36-q3|qwen38-35b] [fast|exact|turbo]" >&2; exit 2 ;;
  esac
done
case "$MODEL" in
  iq2_xs) PRESET=flashnext-iq2_xs$FAST ;;
  q2_0) PRESET=flashnext-q2_0$FAST ;;
  qwen36) PRESET=qwen36-ud-q4_k_xl ;;  # (fast/exact apply to Flash-Next only)
  qwen36-q3) PRESET=qwen36-ud-q3_k_m ;;
  qwen38-35b) PRESET=qwen38-35b-distill-q4_k_m ;;
esac
[ -f "configs/pancake/$PRESET.json" ] || { echo "no preset configs/pancake/$PRESET.json" >&2; exit 1; }
# The GPU picks the build: an NVIDIA card runs build-cuda, otherwise the HIP build (build-hip, RX 7900 XT gfx1100).
# strata-poly (the Qwen3.6 / 35B presets) is CUDA-only.
if nvidia-smi -L >/dev/null 2>&1; then
  BUILD=build-cuda LIBS=/usr/local/cuda/lib64
else
  BUILD=build-hip LIBS=/opt/rocm/core-10.0/lib
  export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}  # ROCm GPU 1 is the Vega iGPU
  grep -q strata-poly "configs/pancake/$PRESET.json" && { echo "$MODEL needs strata-poly, which is CUDA-only" >&2; exit 1; }
fi
# the 7900 XT has its own measured presets (configs/pancake/*-hip.json, MOD_PLAN phase 11) where one exists
[ "$BUILD" = build-hip ] && [ -f "configs/pancake/$PRESET-hip.json" ] && PRESET=$PRESET-hip
CFG="$REPO/pancake-$PRESET.json"
sed -e "s#<STRATA_DATA>#$DATA#g" -e "s#<REPO>#$REPO#g" -e "s#<MODELS>#$MODELS#g" -e "s#/build-cuda/#/$BUILD/#" \
  "configs/pancake/$PRESET.json" > "$CFG"
export LD_LIBRARY_PATH=$LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# Linux memory tuning for the 33 GiB expert arena (tools/pancake/os_tune.sh): without background compaction a long-up
# machine's fragmented RAM stalls the arena lock and systemd-oomd can kill the engine during its load
if [ "$MODEL" = iq2_xs ] || [ "$MODEL" = q2_0 ]; then
  [ "$(cat /proc/sys/vm/compact_unevictable_allowed 2>/dev/null || echo 0)" = 1 ] &&
    echo "pancake: tip - run 'sudo tools/pancake/os_tune.sh install' once (background memory compaction; prevents the" \
         "engine being killed while it loads on a long-up PC). 'tools/pancake/os_tune.sh' shows the state and the optional" \
         "huge-page reservation." >&2
fi
# NOCAP=1: the caller already caps memory (pancake.service sets MemoryMax itself; capped.sh's systemd-run --scope
# would move the engine out of the unit's cgroup, so stop/restart/Conflicts= would no longer reach it)
if [ "${NOCAP:-0}" = 1 ]; then
  # wait until the RAM of a previous run is given back, as capped.sh does
  for _ in $(seq 120); do [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -ge "${WAIT_GIB:-48}" ] && break; sleep 5; done
  export STRATA_RAM_KEEP_GIB="${STRATA_RAM_KEEP_GIB:-12}"
  exec "$REPO/.venv/bin/python" "$REPO/serve/server.py" --engine strata --config "$CFG" --port "${PORT:-8080}"
fi
exec "$REPO/tools/pancake/capped.sh" "$REPO/.venv/bin/python" "$REPO/serve/server.py" --engine strata --config "$CFG" --port "${PORT:-8080}"
