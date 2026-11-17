#!/bin/bash
# interleaved A/B on the rebuilt binary: default (stage_ahead 128) against --stage-ahead 1024, same prompts
cd /c/Users/iancu/repos/pancake-strata/bench/results/2026-11-17-poly-staging
E='C:\Users\iancu\build-latest\build-sm75\strata-poly.exe'
export BENCH_CTR=7000
for i in 1 2; do
 ./arm.sh d1024_$i $E 1 4 500,1000,1800 --stage-ahead 1024
 ./arm.sh d128_$i $E 1 4 500,1000,1800
done
touch ab2.done
