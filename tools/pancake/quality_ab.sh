#!/bin/bash
# pancake-strata: judge the lossy speed switches with tools/force_check.py (SpecPilot's token-forcing rules) against
# the exact Flash-Next preset.  Output: bench/quality/*.tsv and the verdicts in bench/quality/verdicts.txt.
cd "$(dirname "$0")/../.." || exit 1
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
Q=bench/quality; PY=.venv/bin/python
envs(){ python3 -c "import json; print(' '.join('--env '+k+'='+v for k,v in (json.load(open('$1')).get('env') or {}).items()))"; }
EX=$Q/flashnext-iq2_xs.json; FA=$Q/flashnext-iq2_xs-fast.json
[ -f $Q/set.json ] || $PY tools/force_check.py gen $EX $Q/set.json --corpus $Q/held.jsonl --offset 0 --per-kind 4 --new 300 $(envs $EX) || exit 1
$PY tools/force_check.py score $EX $Q/set.json $Q/stock-a.tsv $(envs $EX)
$PY tools/force_check.py score $EX $Q/set.json $Q/stock-b.tsv $(envs $EX)
$PY tools/force_check.py score $EX $Q/set.json $Q/tailskip7.tsv $(envs $EX | sed 's/STRATA_ROUTE_TAIL_SKIP=0/STRATA_ROUTE_TAIL_SKIP=7/')
$PY tools/force_check.py score $FA $Q/set.json $Q/fast.tsv $(envs $FA)
{ for c in tailskip7 fast; do echo "== $c"; $PY tools/force_check.py check --stock $Q/stock-a.tsv $Q/stock-b.tsv --cand $Q/$c.tsv --json $Q/$c.verdict.json; echo "exit $?"; done; } | tee $Q/verdicts.txt
