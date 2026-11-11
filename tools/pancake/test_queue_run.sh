#!/bin/bash
# test_queue_run.sh - tests for queue_run.sh and capped.sh (dry run only).  Nothing here starts an engine or a GPU job:
# queue_run.sh runs a fake abtest (ABTEST=) and a fake drop_cache (DROP_CACHE=) with PYTHON=python3, the clock is a fake
# `date` on PATH (it answers `date +%H` from $FAKE_HOUR_FILE), and capped.sh only runs with -n.  The real-run tests use
# BUSY_PATTERN to ignore a real abtest or engine (a GPU run in progress must not make them fail); only the busy test
# uses queue_run.sh's default pattern, against a sleeping stand-in process.
# Run: bash tools/pancake/test_queue_run.sh     (exit 0: all pass)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
QR=$HERE/queue_run.sh
CAPPED=$HERE/capped.sh
T=$(mktemp -d)
busy=""
trap 'rm -rf "$T"; [ -n "$busy" ] && kill "$busy" 2>/dev/null' EXIT
nfail=0
pass() { echo "ok   $*"; }
bad() { echo "FAIL $*"; nfail=$((nfail + 1)); }
expect() {                       # expect WHAT COMMAND...: pass when COMMAND succeeds
  local what=$1; shift
  if "$@"; then pass "$what"; else bad "$what"; fi
}
expect_rc() {                    # expect_rc CODE WHAT COMMAND...: pass when COMMAND exits with CODE
  local want=$1 what=$2 got; shift 2
  "$@" > /dev/null 2>&1; got=$?
  if [ "$got" = "$want" ]; then pass "$what (exit $got)"; else bad "$what: exit $got, want $want"; fi
}

# fake tools: a date that can be told the hour, an abtest that logs its argv and writes what abtest.py writes, a drop_cache
mkdir -p "$T/bin" "$T/specs"
export REAL_DATE FAKE_LOG=$T/calls.log FAKE_HOUR_FILE=$T/hour
REAL_DATE=$(command -v date)
cat > "$T/bin/date" <<'EOF'
#!/bin/bash
# fake date: "date +%H" prints the hour in $FAKE_HOUR_FILE; anything else is the real date
if [ "$1" = "+%H" ]; then cat "$FAKE_HOUR_FILE"; else exec "$REAL_DATE" "$@"; fi
EOF
chmod +x "$T/bin/date"
cat > "$T/fake_abtest.py" <<'EOF'
# fake abtest.py: logs its argv, writes the jsonl, the rep lines and the summary that abtest.py prints; no engine
import os, pathlib, sys
argv = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write("abtest " + " ".join(argv) + "\n")
name = pathlib.Path(argv[0]).stem
if "fail" in name:
    print("Traceback (fake): the engine did not start")
    sys.exit(3)
pathlib.Path(argv[argv.index("--out") + 1]).write_text("{}\n")
for i in range(int(os.environ.get("FAKE_REPS", "2"))):
    print(f"rep {i} short 100.0 tok/s")
print("summary")
print("  base 100.0 drafts accepted 3")
if name == os.environ.get("FAKE_CLOSE_WINDOW_AFTER"):   # the night ends while this spec runs
    pathlib.Path(os.environ["FAKE_HOUR_FILE"]).write_text("13\n")
EOF
cat > "$T/fake_drop.py" <<'EOF'
import os
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write("drop_cache\n")
print("before: MemFree 1.0 GiB, free 2 MiB blocks 0.1 GiB")
EOF
export PATH="$T/bin:$PATH"
S=$T/specs
for s in a b c fail-d; do echo '{}' > "$S/$s.json"; done
echo 12 > "$FAKE_HOUR_FILE"

qr() { PYTHON=python3 ABTEST=$T/fake_abtest.py DROP_CACHE=$T/fake_drop.py MIN_GIB=0 BUSY_PATTERN="no-such-job-$$" "$QR" "$@"; }
reset() { : > "$FAKE_LOG"; rm -rf "$T"/out*; echo 12 > "$FAKE_HOUR_FILE"; }
calls() { wc -l < "$FAKE_LOG"; }
at_hour() { echo "$1" > "$FAKE_HOUR_FILE"; }

echo "== syntax and lint"
expect "bash -n queue_run.sh" bash -n "$QR"
expect "bash -n capped.sh" bash -n "$CAPPED"
if command -v shellcheck > /dev/null; then
  expect "shellcheck queue_run.sh" shellcheck "$QR"
  expect "shellcheck capped.sh" shellcheck "$CAPPED"
else
  echo "skip shellcheck (not installed)"
fi

echo "== dry run and arguments"
reset
qr -n "$T/out-dry" "$S/a.json" "$S/b.json:--tier=screen" "$S/c.json:--reclaim-max=2.0" > /dev/null
expect "dry run: a gets the default --reclaim-max 3.0" grep -q 'would run: .*a.json .*--reclaim-max 3.0$' "$T/out-dry/queue.log"
expect "dry run: b keeps its tier and gets the default" grep -q 'b.json --out .* --tier screen --reclaim-max 3.0$' "$T/out-dry/queue.log"
expect "dry run: c sets its own reclaim once" test "$(grep c.json "$T/out-dry/queue.log" | grep -c -- '--reclaim-max')" = 1
expect "dry run: c keeps its own 2.0" grep -q 'c.json .*--reclaim-max 2.0$' "$T/out-dry/queue.log"
expect "dry run starts nothing" test "$(calls)" = 0
reset
qr -n --reclaim-max 1.5 "$T/out-dry2" "$S/a.json" > /dev/null
expect "--reclaim-max X sets the default" grep -q -- '--reclaim-max 1.5$' "$T/out-dry2/queue.log"
env -u ABTEST -u PYTHON "$QR" -n "$T/out-dry3" "$S/a.json" > /dev/null
expect "default abtest is tools/pancake/abtest.py under .venv" grep -q '.venv/bin/python tools/pancake/abtest.py' "$T/out-dry3/queue.log"
reset
qr -n --drop-cache "$T/out-dry4" "$S/a.json" "$S/b.json" > /dev/null
expect "dry run --drop-cache names drop_cache once" test "$(grep -c 'would run: python3 .*fake_drop.py' "$T/out-dry4/queue.log")" = 1
expect_rc 2 "usage: bad --night-only" qr --night-only 25-07 "$T/x" "$S/a.json"
expect_rc 2 "usage: --night-only needs HH-HH" qr --night-only 7-9 "$T/x" "$S/a.json"
expect_rc 2 "usage: bad --reclaim-max" qr --reclaim-max fast "$T/x" "$S/a.json"
expect_rc 2 "usage: unknown option" qr --bogus "$T/x" "$S/a.json"
expect_rc 2 "usage: no spec" qr "$T/x"
expect_rc 2 "usage: no arguments" qr

echo "== real run with a fake abtest"
reset
qr "$T/out1" "$S/a.json" "$S/b.json:--tier=screen" "$S/c.json:--reclaim-max=2.0" "$S/nosuch.json" > /dev/null; rc=$?
expect "queue exits 0" test "$rc" = 0
expect "a has jsonl, log and summary" test -s "$T/out1/a.summary.txt" -a -f "$T/out1/a.jsonl" -a -f "$T/out1/a.log"
expect "summary keeps only the summary block" test "$(head -1 "$T/out1/a.summary.txt")" = summary
expect "queue.summary.txt: one line per spec that ran" test "$(wc -l < "$T/out1/queue.summary.txt")" = 3
expect "queue.summary.txt: exit, runs, minutes" grep -q '^a exit=0 runs=2 minutes=[0-9.]*$' "$T/out1/queue.summary.txt"
expect "abtest got the default reclaim" grep -qF -- "abtest $S/a.json --out $T/out1/a.jsonl --reclaim-max 3.0" "$FAKE_LOG"
expect "abtest got the tier and the default reclaim" grep -qF -- "abtest $S/b.json --out $T/out1/b.jsonl --tier screen --reclaim-max 3.0" "$FAKE_LOG"
expect "abtest got c's own reclaim only" test "$(grep c.json "$FAKE_LOG" | grep -c -- '--reclaim-max')" = 1
expect "missing spec is reported" grep -q 'nosuch: no such spec, skipped' "$T/out1/queue.log"
before=$(calls)
qr "$T/out1" "$S/a.json" "$S/b.json:--tier=screen" "$S/c.json:--reclaim-max=2.0" > /dev/null
expect "resume: done specs are skipped" grep -q 'a: done before, skipped' "$T/out1/queue.log"
expect "resume: nothing runs again" test "$(calls)" = "$before"
expect "resume: queue.summary.txt unchanged" test "$(wc -l < "$T/out1/queue.summary.txt")" = 3

echo "== a failing spec does not stop the queue"
reset
qr "$T/out2" "$S/fail-d.json" "$S/a.json" > /dev/null; rc=$?
expect "queue goes on after exit 3 and exits 0" test "$rc" = 0
expect "failing spec: exit=3 runs=0" grep -q '^fail-d exit=3 runs=0 ' "$T/out2/queue.summary.txt"
expect "the next spec ran" grep -q '^a exit=0 runs=2 ' "$T/out2/queue.summary.txt"

echo "== drop-cache step"
reset
qr --drop-cache "$T/out3" "$S/a.json" "$S/b.json" > /dev/null
expect "drop_cache ran once" test "$(grep -c '^drop_cache$' "$FAKE_LOG")" = 1
expect "drop_cache ran before the first spec" test "$(head -1 "$FAKE_LOG")" = drop_cache
expect "drop_cache output is in the queue log" grep -q 'before: MemFree' "$T/out3/queue.log"

echo "== night window"
reset; at_hour 13
expect_rc 5 "13h is outside 23-07: exit 5 at once" qr --night-only 23-07 "$T/out4" "$S/a.json"
expect "...and nothing ran" test "$(calls)" = 0
expect "...and it says why" grep -q 'outside the night window 23-07' "$T/out4/queue.log"
expect_rc 5 "13h is outside 23-07 in a dry run too" qr -n --night-only 23-07 "$T/out5" "$S/a.json"
for h in 0 23; do
  reset; at_hour "$h"
  expect_rc 0 "${h}h is inside 23-07 (wraps midnight)" qr --night-only 23-07 "$T/out6" "$S/a.json"
done
reset; at_hour 7
expect_rc 5 "7h is outside 23-07 (end hour is not in it)" qr --night-only 23-07 "$T/out7" "$S/a.json"
reset; at_hour 13
expect_rc 0 "13h is inside 10-16" qr --night-only 10-16 "$T/out8" "$S/a.json"
reset; at_hour 16
expect_rc 5 "16h is outside 10-16" qr --night-only 10-16 "$T/out9" "$S/a.json"
reset; at_hour 13
expect_rc 0 "equal hours mean all day" qr --night-only 09-09 "$T/out10" "$S/a.json"
reset; at_hour 23
export FAKE_CLOSE_WINDOW_AFTER=a
expect_rc 5 "the night ends during a spec: the next spec is not started" \
  qr --night-only 23-07 "$T/out11" "$S/a.json" "$S/b.json"
unset FAKE_CLOSE_WINDOW_AFTER
expect "...only a ran" test "$(calls)" = 1
expect "...and its line is in the summary" grep -q '^a exit=0 ' "$T/out11/queue.summary.txt"

echo "== a running test or engine stops the queue"
reset
bash -c 'sleep 60; :' _ tools/pancake/abtest.py &
busy=$!
sleep 1
expect_rc 3 "another abtest running (default pattern): exit 3" env -u BUSY_PATTERN PYTHON=python3 \
  ABTEST="$T/fake_abtest.py" MIN_GIB=0 "$QR" "$T/out12" "$S/a.json"
kill "$busy" 2>/dev/null; busy=""
expect "...and nothing was started" test "$(calls)" = 0

echo "== capped.sh (dry run: no wait, no start)"
line=$(STRATA_RAM_KEEP_GIB=20 timeout 20 "$CAPPED" -n true 2>&1 | head -1)
expect "keep 20: mmap tier line" test "$(echo "$line" | sed -E 's/big_free_gib=[0-9]+$//')" = "capped: tier=mmap keep_gib=20 "
line=$(env -u STRATA_RAM_KEEP_GIB timeout 20 "$CAPPED" -n true 2>&1)
first=$(head -1 <<< "$line"); cmd=$(tail -1 <<< "$line")
expect "default keep 12: the capped line has the four fields" grep -qE '^capped: tier=(arena|mmap) keep_gib=12 big_free_gib=[0-9]+$' <<< "$first"
big=${first##*big_free_gib=}; tier=$(sed -E 's/.*tier=([a-z]+) .*/\1/' <<< "$first")
want=mmap; [ "$big" -ge 33 ] && want=arena
expect "tier follows the rule (arena: keep <= 12 and >= 33 GiB in 2 MiB blocks)" test "$tier" = "$want"
expect "the command carries STRATA_TIER_GUESS and the scope cap" grep -qF "STRATA_TIER_GUESS=$tier" <<< "$cmd"
expect "the command starts systemd-run in a scope" grep -qF "would run: systemd-run --user --scope" <<< "$cmd"
line=$(STRATA_RAM_KEEP_GIB=12.5 timeout 20 "$CAPPED" -n true 2>&1 | head -1)
expect "keep above 12: mmap whatever the free blocks" grep -q 'tier=mmap keep_gib=12.5 ' <<< "$line"

echo
if [ "$nfail" = 0 ]; then echo "all passed"; else echo "$nfail failed"; fi
[ "$nfail" = 0 ]
