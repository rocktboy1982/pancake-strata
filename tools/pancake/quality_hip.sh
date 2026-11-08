#!/bin/bash
# pancake-strata: token-forcing check (tools/force_check.py) of the RDNA3 decode kernels on the RX 7900 XT.
# stock = the HIP preset without the ported Strata-rdna3 kernels / fusions (two arms: the noise); cand = with them.
# Reuses bench/quality/set.json (fixed text).  Output: bench/quality-hip/*.tsv and verdict.txt.
cd "$(dirname "$0")/../.." || exit 1
export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib HIP_VISIBLE_DEVICES=0
Q=bench/quality-hip; PY=.venv/bin/python; SET=bench/quality/set.json
envs(){ python3 -c "import json; print(' '.join('--env '+k+'='+v for k,v in (json.load(open('$1')).get('env') or {}).items()))"; }
run(){ tools/pancake/capped.sh $PY tools/force_check.py score "$1" $SET "$2" $(envs "$1"); }
run $Q/stock.json $Q/stock-a.tsv
run $Q/stock.json $Q/stock-b.tsv
run $Q/cand.json $Q/cand.tsv
$PY tools/force_check.py check --stock $Q/stock-a.tsv $Q/stock-b.tsv --cand $Q/cand.tsv --json $Q/cand.verdict.json | tee $Q/verdict.txt
echo "exit ${PIPESTATUS[0]}" | tee -a $Q/verdict.txt
