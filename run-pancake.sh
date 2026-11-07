#!/bin/sh
# pancake-strata: serve Qwen3.8-Flash-Next IQ2_XS on the strata engine built in build-cuda/.
#   ./run-pancake.sh          exact preset (no tail skip, English draft vocabulary): the same answers as plain decoding
#   ./run-pancake.sh fast     + Speedy's switches: ~13% faster writing after long prompts; INT8 prompt attention and
#                             Q8 hyper-connection weights move the numbers slightly (docs/PANCAKE_FINDINGS.md section 10)
# STRATA_DATA (default /home/dan/clone-of-d/Strata-data) holds the model, pack and MTP files; PORT defaults to 8080.
cd "$(dirname "$0")" || exit 1
REPO=$(pwd)
DATA=${STRATA_DATA:-/home/dan/clone-of-d/Strata-data}
PRESET=flashnext-iq2_xs
[ "$1" = "fast" ] && PRESET=flashnext-iq2_xs-fast
CFG="$REPO/pancake-$PRESET.json"
sed -e "s#<STRATA_DATA>#$DATA#g" -e "s#<REPO>#$REPO#g" "configs/pancake/$PRESET.json" > "$CFG"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
exec "$REPO/.venv/bin/python" "$REPO/serve/server.py" --engine strata --config "$CFG" --port "${PORT:-8080}"
