#!/bin/sh
# hipblaslt-arms-run.sh - the box commands of LEADS.md Part T leads T3 (spec 255: no table / hipBLASLt heuristic / table) and T4 (warm-up, first-request time).
# NEVER RUN BY THE AUTHOR (coded on lead/T2-T4 from a Windows PC with no ROCm): TEST_LEDGER.md rows say "coded, not run".  Run by hand ON THE BOX, only when it is
# idle and an engine of THIS branch is built (the HIP build with src/prefill/gemm.cu of lead/T2-T4).  It refuses to start while any other job runs, never touches
# pancake.service or a queue, and has no "bench-", "win-chain" or "abtest" in its own name (the abtest.py it starts is reached through tools/pancake/./abtest.py,
# as the other chains do, so its busy check does not take this script for another test).
#
# Parts (PART="T3 T4" by default):
#   T3  spec 255 screen: arms base-no-table (STRATA_HIPBLASLT_TABLE=off), heuristic (+ STRATA_HIPBLASLT_HEURISTIC=1), heuristic-top4, table (the 100401 table, or
#       TABLE_FILE=<a freshly tuned 100500 file> on ROCm 10.2), cases long10k / long32k / long64k, 1 request per case and arm, drift check on.  Output
#       $OUT/255.jsonl + $OUT/255.stdout.  The verdict is the owner's: heuristic vs base-no-table (does the library's own choice beat hipBLASEx), heuristic vs
#       table (can the per-version table go).  Quality gate / KLD check BEFORE any adoption (a heuristic algorithm may differ in the last bits).
#   T4  first-request latency of the compressed hipBLASLt catalogs (ROCm 10.2's hipBLASLt 1.5.0 unpacks a catalog on first use): the same engine started 3 times
#       with STRATA_HIPBLASLT_WARMUP=0 and 3 times with =1, ENGINE_CMD below, one prompt request each, the time of the FIRST request and the engine's own
#       "STRATA_HIPBLASLT_WARMUP: ... s" line.  Writes $OUT/warm-<0|1>-<n>.log.  The warm-up cost moves to the start (before 'ready'); the saving is the first
#       request's time.  Compare 10.0 against 10.2 builds the same way (ROCM and BUILD differ).
#
# Usage on the box:
#   cd <pancake checkout of lead/T2-T4>
#   BU=/mnt/pancake/benchRD/build-updated CONFIG_PLAIN=/home/dan/auto-runs/benchRD/config-plain.json \
#     ROCM=/opt/rocm/core-10.0 sh tools/pancake/box_chains/hipblaslt-arms-run.sh
# Variables: BU (dir holding the built `strata`, required), CONFIG_PLAIN (required for T3, the plain run config of spec 251), ROCM (default /opt/rocm/core-10.0),
#   OUT (default $HOME/auto-runs/hipblaslt-arms), TABLE_FILE (default tools/hip/gfx1100-hipblaslt-100401.txt of this checkout), PART, HIP_VISIBLE_DEVICES (default 0),
#   ENGINE_CMD (T4: the command that starts the engine with the daily config and answers one request; default `python3 serve/server.py --config $CONFIG_PLAIN`
#   plus the request by curl - read the T4 block below and adjust it to the box's way of starting a server before the first run).
set -u
SRC=${SRC:-$(cd "$(dirname "$0")/../../.." && pwd)}
ROCM=${ROCM:-/opt/rocm/core-10.0}
OUT=${OUT:-$HOME/auto-runs/hipblaslt-arms}
BU=${BU:-}
CONFIG_PLAIN=${CONFIG_PLAIN:-}
TABLE_FILE=${TABLE_FILE:-$SRC/tools/hip/gfx1100-hipblaslt-100401.txt}
PART=${PART:-T3 T4}
LOG=$OUT/run.log
export PATH=$ROCM/bin:$PATH LD_LIBRARY_PATH=$ROCM/lib HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

say() { echo "[hipblaslt-arms] $(date +%T) $*" | tee -a "$LOG"; }
die() { say "ERROR $1"; echo "HIPBLASLT_ARMS_DONE" >> "$LOG"; exit 1; }
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[r]un-queue-us|[b]ench-sched|[b]ench-sweep" > /dev/null 2>&1; }

mkdir -p "$OUT" || { echo "cannot create $OUT"; exit 1; }
: >> "$LOG"
busy && die "the box is busy; nothing was started"
[ -n "$BU" ] && [ -x "$BU/strata" ] || die "BU must be the directory of a built strata (the HIP build of this branch); got '$BU'"
[ -f "$SRC/bench/specs/255-hipblaslt-heuristic-arms.json" ] || die "spec 255 is missing in $SRC"
# the build must be THIS branch's: its binary carries the new switch names
strings "$BU/strata" 2>/dev/null | grep -q "STRATA_HIPBLASLT_HEURISTIC" || die "$BU/strata does not know STRATA_HIPBLASLT_HEURISTIC: build lead/T2-T4 first"

for p in $PART; do
  case $p in
  T3)
    [ -f "$CONFIG_PLAIN" ] || die "T3 needs CONFIG_PLAIN (the plain run config, as spec 251's)"
    [ -f "$TABLE_FILE" ] || die "T3 table file missing: $TABLE_FILE"
    sed "s#@UPDATED_SRC@#$SRC#g; s#@UPDATED@#$BU#g; s#@CONFIG_PLAIN@#$CONFIG_PLAIN#g; s#$SRC/tools/hip/gfx1100-hipblaslt-100401.txt#$TABLE_FILE#g" \
      "$SRC/bench/specs/255-hipblaslt-heuristic-arms.json" > "$OUT/255.json" || die "T3 spec fill"
    say "T3 spec 255 screen: table arm = $TABLE_FILE, build $BU"
    busy && die "T3: the box became busy"
    python3 "$SRC/tools/pancake/./abtest.py" "$OUT/255.json" --out "$OUT/255.jsonl" --tier screen > "$OUT/255.stdout" 2>&1
    rc=$?
    say "T3 abtest exit $rc; look for 'Lt heuristic' / 'status=' lines in the engine logs of the run; an arm without 'STRATA_HIPBLASLT_HEURISTIC=1' in its engine log is inert"
    grep -E "vs base|identical|DIFFERENT" "$OUT/255.stdout" | cut -c1-200 >> "$LOG"
    ;;
  T4)
    # ENGINE_CMD starts the engine as the daily setup does and must print its log to stdout/stderr; the request is a one-token-out prompt of 4000 tokens
    # (a table bucket), so the time is the prompt read, where the first-use unpack lands.  Adjust REQUEST_CMD to the box's server port.
    ENGINE_CMD=${ENGINE_CMD:-"python3 $SRC/serve/server.py --config ${CONFIG_PLAIN:-/dev/null} --port 8099"}
    REQUEST_CMD=${REQUEST_CMD:-"python3 $SRC/tools/pancake/./abtest.py --help"}
    say "T4 warm-up first-request timing: 3 starts each, WARMUP=0 then 1 (interleaved). ENGINE_CMD=$ENGINE_CMD"
    say "T4 NOTE: REQUEST_CMD defaults to a placeholder; set it to a curl of one ~4000-token chat completion against the port, then re-run PART=T4"
    case "$REQUEST_CMD" in *"--help"*) die "T4: set REQUEST_CMD (a curl of one ~4000-token completion); not run" ;; esac
    for n in 1 2 3; do
      for w in 0 1; do
        busy && die "T4: the box became busy"
        f=$OUT/warm-$w-$n.log
        say "T4 start WARMUP=$w run $n -> $f"
        ( STRATA_HIPBLASLT_WARMUP=$w sh -c "$ENGINE_CMD" > "$f" 2>&1 & echo $! > "$OUT/engine.pid" )
        i=0
        while [ $i -lt 600 ]; do
          grep -qi "ready\|listening" "$f" 2>/dev/null && break
          sleep 2; i=$((i + 2))
        done
        t0=$(date +%s.%N)
        sh -c "$REQUEST_CMD" > "$OUT/warm-$w-$n.request" 2>&1
        t1=$(date +%s.%N)
        echo "first_request_seconds=$(echo "$t1 $t0" | awk '{printf "%.2f", $1 - $2}') warmup=$w run=$n" | tee -a "$LOG" "$f" > /dev/null
        grep "STRATA_HIPBLASLT_WARMUP" "$f" | tail -1 >> "$LOG"
        kill "$(cat "$OUT/engine.pid")" 2> /dev/null
        sleep 10
      done
    done
    ;;
  *) say "unknown part $p (T3 or T4)";;
  esac
done
say "done; archive $OUT in bench/results/<date>-hipblaslt-arms/ and write the ledger rows (T-nnn) from $OUT/255.stdout and $LOG"
echo "HIPBLASLT_ARMS_DONE" >> "$LOG"
