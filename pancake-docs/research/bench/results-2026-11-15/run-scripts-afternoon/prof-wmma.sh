#!/bin/sh
cd /home/dan/clone-of-d/src/Strata
export PATH=/home/dan/clone-of-d/src/Strata/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib
echo "START prof-wmma $(date +%T)" >> /home/dan/auto-chain.log
python3 tools/pancake/gpu_profile.py configs/pancake/flashnext-iq2_xs-fast-hip.json /home/dan/prof-wmma-base --case edit --env STRATA_RDNA_WMMA_MINCOLS=0 > /home/dan/auto-runs/prof-wmma-base.stdout 2>&1
python3 tools/pancake/gpu_profile.py configs/pancake/flashnext-iq2_xs-fast-hip.json /home/dan/prof-wmma-on --case edit --env STRATA_RDNA_WMMA_MINCOLS=9 > /home/dan/auto-runs/prof-wmma-on.stdout 2>&1
echo "END prof-wmma $(date +%T)" >> /home/dan/auto-chain.log
