#!/bin/bash
# test_phase.sh - the whole test phase in one command (owner: develop first, then test): every spec in bench/specs/ in
# order, each through tools/pancake/abtest.py (quiet-machine gate, noisy runs repeated, 95% intervals) and, for specs
# marked "lossy", tools/pancake/quality_gate.py (token forcing).  Results: bench/results/<day>-test-phase/.
# Before starting: tell the other Claude sessions to stop builds / GPU jobs.  Resumable: a spec with a summary is skipped.
#   tools/pancake/test_phase.sh [SPEC ...]      (default: all of bench/specs/*.json)
cd "$(dirname "$0")/../.." || exit 1
DAY=$(date +%Y-%m-%d)
OUT=bench/results/$DAY-test-phase
mkdir -p "$OUT"
PY=.venv/bin/python
# the base config is regenerated from the repo's 7900 XT preset, as run-pancake.sh writes it
sed -e "s#<STRATA_DATA>#/home/dan/clone-of-d/Strata-data#g" -e "s#<REPO>#$PWD#g" -e "s#<MODELS>#/home/dan/clone-of-d/models#g" \
    -e "s#/build-cuda/#/build-hip/#" configs/pancake/flashnext-iq2_xs-fast-hip.json > pancake-flashnext-iq2_xs-rdna3test.json
git rev-parse --short HEAD > "$OUT/rev.txt"
specs=("$@"); [ ${#specs[@]} -eq 0 ] && specs=(bench/specs/*.json)
# items 35/83 (dan-9f): restore-equivalence gate first - a parked/restored conversation must match a cold re-read token
# for token; it gates the checkpoint/cache items (39, 16, 31, 11, disk tier).  Needs the GPU alone; skipped on resume.
if [ $# -eq 0 ] && [ ! -s "$OUT/restore-check.txt" ]; then
  echo "== restore check ($(date +%H:%M))"
  $PY tools/pancake/restore_check.py > "$OUT/restore-check.txt" 2>&1; echo "restore check exit $?" | tee -a "$OUT/restore-check.txt"
  tail -n 5 "$OUT/restore-check.txt"
fi
for sp in "${specs[@]}"; do
  n=$(basename "$sp" .json)
  if [ -s "$OUT/$n.summary.txt" ]; then echo "== $n: done before, skipped"; continue; fi
  echo "== $n ($(date +%H:%M))"
  $PY tools/pancake/abtest.py "$sp" --out "$OUT/$n.jsonl" 2>&1 | tee "$OUT/$n.log" | sed -n '/^summary/,$p' > "$OUT/$n.summary.txt"
  cat "$OUT/$n.summary.txt"
  if python3 -c "import json,sys; sys.exit(0 if json.load(open('$sp')).get('lossy') else 1)"; then
    $PY tools/pancake/quality_gate.py "$sp" --out "$OUT/$n-quality" 2>&1 | tee "$OUT/$n.quality.txt" | grep -E 'PASS|REJECT|INCONCLUSIVE|FAILED'
  fi
done
# item 76: the device-spin arm is void unless the engine log says the scheduling flag was set
grep -h "STRATA_DEVICE_SPIN=1" "$OUT"/04-gpu-launches/device-spin-*.out 2>/dev/null | sort | uniq -c | sed 's/^/device-spin check: /'
echo "TEST PHASE DONE: $OUT"
