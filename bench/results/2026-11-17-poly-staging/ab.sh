#!/bin/bash
# interleaved A/B: off,on,off,on ; same prompts (BENCH_CTR) so ids_hash must match
cd /c/Users/iancu/repos/pancake-strata/bench/results/2026-11-17-poly-staging
E='C:\Users\iancu\build-latest\build-sm75\strata-poly.exe'
export BENCH_CTR=5000
for i in 1 2; do
 ./arm.sh ab_off$i $E 1 4 500,1000,1800
 ./arm.sh ab_on$i $E 1 4 500,1000,1800 --stage-ahead 128
done
touch ab.done
