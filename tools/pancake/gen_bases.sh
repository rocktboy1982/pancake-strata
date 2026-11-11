#!/bin/sh
# gen_bases.sh - write the generated pancake-X.json base configs that the specs name, from configs/pancake/X.json,
# the way run-pancake.sh writes them (same placeholders).  The specs, abtest.py and queue_run.sh read these files from the
# repo root; only test_phase.sh and run-pancake.sh write them otherwise, so a queue run can start on a stale file.
# Without an NVIDIA card a template that runs build-cuda (the RTX 3060 and strata-poly presets) is skipped: run-pancake.sh
# never writes it there either.
#   tools/pancake/gen_bases.sh [SPEC ...]      (default: every bench/specs/*.json)
# pancake-flashnext-iq2_xs-rdna3test.json is written from configs/pancake/flashnext-iq2_xs-fast-hip.json, as test_phase.sh
# does (the template of that name is an older preset and lacks the adopted items).
# STRATA_DATA (default /home/dan/clone-of-d/Strata-data) and MODELS (default /home/dan/clone-of-d/models) as in run-pancake.sh.
cd "$(dirname "$0")/../.." || exit 1
REPO=$(pwd)
DATA=${STRATA_DATA:-/home/dan/clone-of-d/Strata-data}
MODELS=${MODELS:-/home/dan/clone-of-d/models}
nvidia-smi -L >/dev/null 2>&1 && NVIDIA=1 || NVIDIA=0
[ $# -gt 0 ] || set -- bench/specs/*.json
bases=$(python3 -c '
import json, sys
bases = {json.load(open(path)).get("base", "") for path in sys.argv[1:]}
print("\n".join(sorted(b for b in bases if b.startswith("pancake-") and "/" not in b)))' "$@") || exit 1
rc=0
for base in $bases; do
  name=${base#pancake-}; name=${name%.json}
  [ "$name" = flashnext-iq2_xs-rdna3test ] && name=flashnext-iq2_xs-fast-hip
  tpl=configs/pancake/$name.json
  [ -f "$tpl" ] || { echo "$base: no template $tpl" >&2; rc=1; continue; }
  if [ $NVIDIA = 0 ] && grep -q /build-cuda/ "$tpl"; then
    echo "$base: skipped, $tpl runs build-cuda and there is no NVIDIA card"; continue
  fi
  sed -e "s#<STRATA_DATA>#$DATA#g" -e "s#<REPO>#$REPO#g" -e "s#<MODELS>#$MODELS#g" "$tpl" > "$base" || { rc=1; continue; }
  echo "$base <- $tpl"
done
exit $rc
