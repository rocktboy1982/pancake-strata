#!/bin/sh
# LEADS.md Part U lead U4 (instruction-rate probe), box script v1.  NEW file; it never edits or kills anything and never deletes a file.
# IDLE ONLY: the GPU must be idle for about 2 minutes.  The script REFUSES (exit 3, one log line) while any busy-pattern process runs, and it
# does not wait for one: start it by hand when the flag screen / the queue is done, then read the log.
#   sh tools/pancake/box_chains/u4-valu-rate-chain.v1.sh [SRC_TREE]      (default SRC_TREE = /home/dan/clone-of-d/src/Strata-win)
# Steps: 0 idle + disk check | 1 hipcc build of the probe into a NEW directory | 2 ISA check (every mode's mnemonic is in its own kernel) |
#        3 the probe (valu_rate, defaults: 4 / 8 / 12 waves per SIMD, about 2 minutes) | 4 the report (ops per clock per CU, % of theoretical on 84 CUs) |
#        5 the existing wmma_rate probe (iu8 / iu4 rows, a few seconds; its CLI is unchanged).
# Output: $OUT = /home/dan/auto-runs/u4-valu-rate-<timestamp>/ (valu_rate.s, isa.txt, valu_rate.out, report.txt, wmma_rate.out, chain.log).
# Read the verdict lines of report.txt, then record the numbers in TEST_LEDGER.md (rows T-1062..), never in a new document.
SRC=${1:-/home/dan/clone-of-d/src/Strata-win}
TS=$(date +%Y%m%d_%H%M%S)
OUT=/home/dan/auto-runs/u4-valu-rate-$TS
HIPCC=${HIPCC:-/opt/rocm/core-10.0/bin/hipcc}
export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[b]ench-sweep|[b]ench-sched|[r]un-queue-us|[w]in-chain[A-Za-z0-9]*\.sh|[v]alu_rate|[w]mma_rate" >/dev/null; }
if busy; then
  echo "REFUSED $(date +%F_%T): a busy-pattern process is running (flag screen / queue / build / test); the probe needs an idle GPU. Nothing was done." >&2
  exit 3
fi
mkdir -p "$OUT" || exit 2
L=$OUT/chain.log
say() { echo "$(date +%T) $*" >> "$L"; echo "$*"; }
say "START u4-valu-rate-chain.v1 src=$SRC out=$OUT"
[ -f "$SRC/tools/pancake/probes/valu_rate.hip.cpp" ] || { say "FATAL no valu_rate.hip.cpp in $SRC (sync the branch lead/U4-U7-U8 or port/rdna3-decode after the merge with tar xmf first)"; exit 2; }
[ -x "$HIPCC" ] || { say "FATAL no hipcc at $HIPCC"; exit 2; }
FREE=$(df -Pk "$OUT" | awk 'NR==2{print $4}')
[ "$FREE" -gt 204800 ] || { say "SKIPPED: less than 200 MB free under $OUT ($FREE KB); nothing deleted"; exit 2; }
# 1 build (a few seconds; one translation unit, no ninja, no cmake)
say "BUILD valu_rate"
"$HIPCC" -O2 --offload-arch=gfx1100 "$SRC/tools/pancake/probes/valu_rate.hip.cpp" -o "$OUT/valu_rate" >> "$L" 2>&1 || { say "BUILD FAILED (see chain.log)"; exit 2; }
"$HIPCC" -O2 --offload-arch=gfx1100 "$SRC/tools/pancake/probes/wmma_rate.hip.cpp" -o "$OUT/wmma_rate" >> "$L" 2>&1 || say "wmma_rate build failed (continuing)"
# 2 ISA check
say "ISA check"
"$HIPCC" -O2 --offload-arch=gfx1100 -S --cuda-device-only "$SRC/tools/pancake/probes/valu_rate.hip.cpp" -o "$OUT/valu_rate.s" >> "$L" 2>&1
python3 "$SRC/tools/pancake/probes/valu_rate_report.py" --isa "$OUT/valu_rate.s" > "$OUT/isa.txt" 2>&1; ISA_RC=$?
say "ISA check exit=$ISA_RC (isa.txt); a NOT EMITTED row means that mode did NOT measure its instruction: read it before the numbers"
# 3 run (idle re-check first: the build is the only thing that ran)
if busy; then say "REFUSED before the run: something started meanwhile"; exit 3; fi
say "RUN valu_rate"
"$OUT/valu_rate" > "$OUT/valu_rate.out" 2>> "$L"; RC=$?
say "valu_rate exit=$RC"
# 4 report
python3 "$SRC/tools/pancake/probes/valu_rate_report.py" "$OUT/valu_rate.out" --cus 84 > "$OUT/report.txt" 2>&1
say "report written: $OUT/report.txt"
cat "$OUT/report.txt"
# 5 wmma_rate (iu8 / iu4 rows of Part L L2; a few seconds)
if [ -x "$OUT/wmma_rate" ] && ! busy; then
  say "RUN wmma_rate"
  "$OUT/wmma_rate" > "$OUT/wmma_rate.out" 2>> "$L"; say "wmma_rate exit=$?"
fi
say "END u4-valu-rate-chain.v1 (U4_DONE)"
