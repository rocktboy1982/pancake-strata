#!/bin/bash
# usage: arm.sh LABEL EXE SESSIONS REPS SIZES [extra strata-poly args...]   (env of caller reaches the server)
L=$1; EXE=$2; S=$3; R=$4; SZ=$5; shift 5
D=/c/Users/iancu/repos/pancake-strata/bench/results/2026-11-17-poly-staging
export STRATA_POLY_EXE="$EXE"
export STRATA_MODEL='C:\Users\iancu\models\Qwen3.6-35B-A3B-UD-IQ2_XXS.gguf'
export STRATA_TOKENIZER='C:\Users\iancu\build-latest\tok\tokenizer.json'
export CUDA_BIN='C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin'
export BENCH_OUT="C:/Users/iancu/repos/pancake-strata/bench/results/2026-11-17-poly-staging" BENCH_LABEL=$L
{ date; nvidia-smi --query-gpu=utilization.gpu,memory.used,clocks.sm --format=csv,noheader; } > $D/${L}_pre.txt
/c/Users/iancu/build-latest/venv/Scripts/python.exe -I /c/Users/iancu/repos/pancake-strata/tools/pancake/bench_win_prompt.py $S $R 16 4096 $SZ "$@" > $D/${L}_stdout.txt 2>&1
