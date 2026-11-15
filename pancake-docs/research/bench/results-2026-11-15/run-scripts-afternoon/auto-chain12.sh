#!/bin/sh
L=/home/dan/auto-chain.log
until grep -q 'END 56-verify-ecap-confirm' "$L" 2>/dev/null; do sleep 30; done
cd /home/dan/clone-of-d/src/Strata
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START qgate-ecap $(date +%T)" >> $L
python3 tools/pancake/quality_gate.py bench/specs/56-verify-ecap-confirm.json --out bench/quality-hip/2026-11-15-ecap --set bench/quality/set-v2.json --fill > /home/dan/auto-runs/qgate-ecap.stdout 2>&1
echo "END qgate-ecap exit=$? $(date +%T)" >> $L
