#!/bin/sh
# pancake-strata: serve a model with its measured preset.
#   ./run-pancake.sh [MODEL] [fast]
# MODEL (default iq2_xs):
#   iq2_xs      Qwen3.8-Flash-Next, ISTA-DASLab IQ2_XS - main engine (the daily model)
#   q2_0        Qwen3.8-Flash-Next, ISTA-DASLab Q2_0   - main engine (Strata's fastest file; basalt's Q5_K head kernels)
#   qwen36      Qwen3.6-35B-A3B, Unsloth UD-Q4_K_XL    - strata-poly (FastStrata's engine)
#   qwen36-q3   Qwen3.6-35B-A3B, Unsloth UD-Q3_K_M     - strata-poly, smaller and faster
#   qwen38-35b  Qwen3.8-35B-A3B-Distill Q4_K_M (empero: Qwen3.6 fine-tuned on Qwen3.8 answers) - strata-poly, the
#               MTP drafting layer borrowed from the Qwen3.6 Q4_K_XL file
# fast (Flash-Next only): + Speedy's switches, ~13% faster writing after long prompts; the numbers move slightly.
# Without it every preset is exact: the same answers as plain decoding (docs/PANCAKE_FINDINGS.md).
# STRATA_DATA (default /home/dan/clone-of-d/Strata-data) and MODELS (default /home/dan/clone-of-d/models) hold the
# files; PORT defaults to 8080.
cd "$(dirname "$0")" || exit 1
REPO=$(pwd)
DATA=${STRATA_DATA:-/home/dan/clone-of-d/Strata-data}
MODELS=${MODELS:-/home/dan/clone-of-d/models}
MODEL=iq2_xs
FAST=
for a in "$@"; do
  case "$a" in
    fast) FAST=-fast ;;
    iq2_xs|q2_0|qwen36|qwen36-q3|qwen38-35b) MODEL=$a ;;
    *) echo "usage: $0 [iq2_xs|q2_0|qwen36|qwen36-q3|qwen38-35b] [fast]" >&2; exit 2 ;;
  esac
done
case "$MODEL" in
  iq2_xs) PRESET=flashnext-iq2_xs$FAST ;;
  q2_0) PRESET=flashnext-q2_0$FAST ;;
  qwen36) PRESET=qwen36-ud-q4_k_xl ;;
  qwen36-q3) PRESET=qwen36-ud-q3_k_m ;;
  qwen38-35b) PRESET=qwen38-35b-distill-q4_k_m ;;
esac
[ -f "configs/pancake/$PRESET.json" ] || { echo "no preset configs/pancake/$PRESET.json" >&2; exit 1; }
CFG="$REPO/pancake-$PRESET.json"
sed -e "s#<STRATA_DATA>#$DATA#g" -e "s#<REPO>#$REPO#g" -e "s#<MODELS>#$MODELS#g" "configs/pancake/$PRESET.json" > "$CFG"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec "$REPO/.venv/bin/python" "$REPO/serve/server.py" --engine strata --config "$CFG" --port "${PORT:-8080}"
