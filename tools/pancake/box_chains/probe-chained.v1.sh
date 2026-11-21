#!/bin/sh
# probe-chained.v1.sh - LEADS.md Part S lead S3 (chained launches): the ceiling probe of tools/pancake/probes/retained_replay.hip.cpp with its
# arm (d) (no AQL barrier bit, in-kernel counter wait).  NEW FILE, NOT RUN (written on the Windows PC; the box was busy with the flag screen).
#
# Builds the probe twice (the code object for the raw AQL arms, then the host program with -DSTRATA_PROBE_HSA) and runs it on the idle card:
# stream / graph / AQL barrier replay / chained (HIP stream) / chained (AQL, no barrier bit), N = 2,248 and 2,648 nodes, then the ceiling lines
# (rule: below 3% of the 24.8 ms window stop, no engine step).  Seconds of GPU time; compiles in about a minute.
# Run:   cd <repo on the box> && sh tools/pancake/box_chains/probe-chained.v1.sh        (output: $OUT/probe-chained.out)
# It refuses to run beside any measured job (exit 3): stop pancake.service and wait for the queue yourself; it never touches the queue.
# Read the result with the lines "chain check ... FAILED" first: a failed check means the chain did not run in order, the timings are void.
set -u
HIPCC=${HIPCC:-/opt/rocm/core-10.0/bin/hipcc}
SRC=${SRC:-tools/pancake/probes/retained_replay.hip.cpp}
OUT=${OUT:-/home/dan/auto-runs/probe-chained}
TMP=${TMP:-/mnt/pancake/probe-chained}
mkdir -p "$OUT" "$TMP" || exit 2
L="$OUT/probe-chained.out"
say() { echo "$(date +%T) $*" | tee -a "$L"; }
if pgrep -f 'abtest|strata --serve|serve/server.py|quality_gate|task_gate|ninja|clang|ctest|replay_turns|bench-sched|bench-sweep|run-queue' >/dev/null 2>&1; then
  say "BUSY: a measured job or a build is running; nothing started"
  exit 3
fi
say "PROBE_CHAINED start (hipcc $HIPCC)"
if ! "$HIPCC" --offload-arch=gfx1100 --genco "$SRC" -o "$TMP/empty.co" >"$TMP/genco.log" 2>&1; then say "ERROR genco failed: $(tail -3 "$TMP/genco.log")"; exit 4; fi
if ! "$HIPCC" -O2 --offload-arch=gfx1100 -DSTRATA_PROBE_HSA "$SRC" -o "$TMP/retained_replay" >"$TMP/build.log" 2>&1; then
  say "NOTE: the HSA build failed ($(tail -2 "$TMP/build.log" | tr '\n' ' ')); building without the AQL arms"
  if ! "$HIPCC" -O2 --offload-arch=gfx1100 "$SRC" -o "$TMP/retained_replay" >>"$TMP/build.log" 2>&1; then say "ERROR build failed"; exit 4; fi
fi
"$TMP/retained_replay" --n 2248,2648 --blocks 96 --threads 256 --reps 20 --window-ms 24.8 --hsaco "$TMP/empty.co" 2>&1 | tee -a "$L"
say "PROBE_CHAINED_DONE exit=$?"
