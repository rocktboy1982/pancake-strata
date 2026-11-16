#!/bin/sh
# host_tune_ab.sh - independent host-level A/B of every host_tune.sh step on the Linux box (MOD_PLAN.md section 5, host tuning).
#
# Per step: wait for 3 idle minutes, stop pancake.service (only after 3 minutes without a client on port 11437), apply ONE step
# with tools/pancake/host_tune.sh, run the self-vs-self screen spec bench/specs/125-host-tune-ab.json (cases short, edit, long1k,
# long3k; arms base and base-again on the same binary, drift_check on), revert the step, wait for idle again.  Plus a baseline
# run before the first step and one after the last (drift of the box over the whole session), and for the NVMe read-ahead step a
# cold pack-load timing (page cache dropped with `sudo tee /proc/sys/vm/drop_caches`, then tools/pancake/load_time.py).
# Judge a step by its base arm against winH-baseline: the base-again arm of the same file is that file's noise floor.
#
#   tools/pancake/host_tune_ab.sh plan                 print the order, files and the duration estimate (changes nothing)
#   tools/pancake/host_tune_ab.sh run [STEPS]          the whole session; STEPS = comma list, default
#                                                      nvme_ra,thp,swappiness,cpu_perf,amd_pstate,gpu_dpm,aspm,cstate,all
#                                                      ('all' = every step that was not already at target, applied together)
#   tools/pancake/host_tune_ab.sh mlock                restart pancake.service idle and check what the engine pinned (VmLck)
#   tools/pancake/host_tune_ab.sh table                base-arm means of every winH-*.jsonl and the change against the baseline
# Options: --out DIR (default /home/dan/auto-runs; env AB_OUT)  --spec FILE  --dry-run  --idle-s N (default 180)
#          --no-baseline  --expect-gib N (mlock, default 33)
# Result files in DIR: winH-<step>.stdout (abtest summary), .jsonl (rows), .check.txt, .apply.txt, and for the cold load
#          winH-<step>.coldload.stdout and winH-<step>-coldload-<n>/rows.jsonl; winH-mlock.stdout.
# Box rules kept: never overlaps another box job (pgrep of abtest.py, replay_turns.py, task_gate.py, quality_gate.py, cmake, ninja,
# strata --serve outside pancake.service); stops pancake.service only after IDLE_S seconds with no established connection on port
# 11437 and starts it again at the end (also on Ctrl-C or error, together with the revert of the step in flight).  Steps that are
# already at target (probe) are skipped and logged: no A/B is possible there.  memlock has no runtime form: it is tested by the
# `mlock` command, not applied here.  Nothing here edits sudoers, permissions or the boot loader.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
TUNE="$HERE/host_tune.sh"
OUT=${AB_OUT:-/home/dan/auto-runs}
SPEC=${AB_SPEC:-bench/specs/125-host-tune-ab.json}
PORT=${AB_PORT:-11437}
SERVICE=${AB_SERVICE:-pancake.service}
IDLE_S=${AB_IDLE_S:-180}
MAX_WAIT_S=${AB_MAX_WAIT_S:-21600}
COLD_STEPS=${AB_COLD_STEPS:-nvme_ra}
PYTHON=${PYTHON:-python3}
DEFAULT_STEPS="nvme_ra,thp,swappiness,cpu_perf,amd_pstate,gpu_dpm,aspm,cstate,all"
RUNTIME_STEPS="nvme_ra thp swappiness cpu_perf amd_pstate gpu_dpm aspm cstate"
BUSY=${AB_BUSY_PATTERN:-'^[^ ]*python[0-9.]* ([^ ]+ )*[^ ]*(abtest|replay_turns|task_gate|quality_gate)\.py( |$)|^[^ ]*/(cmake|ninja)( |$)|(^|/)strata --serve'}
DRY=0; BASELINE=1; EXPECT_GIB=33; CMD=""; STEPARG=""
WAS_ACTIVE=0; STOPPED=0; CURRENT=""
SUDO=sudo; [ "$(id -u)" = 0 ] && SUDO=""

say() { echo "[$(date +%H:%M:%S)] $*"; }
die() { echo "host_tune_ab: $*" >&2; exit 1; }
run() { if [ "$DRY" = 1 ]; then say "[dry-run] $*"; else "$@"; fi; }

busy_pids() {   # engines / tests / builds that are not pancake.service
  for p in $(pgrep -f "$BUSY" 2>/dev/null); do
    grep -q "$SERVICE" "/proc/$p/cgroup" 2>/dev/null && continue
    echo "$p"
  done; }
clients() { ss -Htn state established "( sport = :$PORT )" 2>/dev/null | wc -l; }

wait_quiet() {   # $1 = what to watch (busy|clients), $2 = seconds it must stay quiet
  what=$1; need=$2; quiet=0; waited=0
  [ "$DRY" = 1 ] && { say "[dry-run] wait $need s with no $what"; return 0; }
  while [ "$quiet" -lt "$need" ]; do
    if [ "$what" = busy ]; then n=$(busy_pids | wc -l); else n=$(clients); fi
    if [ "$n" -gt 0 ]; then
      [ "$quiet" -gt 0 ] && say "$what seen again ($n): idle clock restarts"
      quiet=0
    else quiet=$((quiet + 10)); fi
    [ "$quiet" -lt "$need" ] && { sleep 10; waited=$((waited + 10)); }
    [ "$waited" -gt "$MAX_WAIT_S" ] && die "waited more than $MAX_WAIT_S s for no $what: giving up (nothing was changed or it was reverted)"
  done
  say "no $what for $need s"; }

wait_idle() { wait_quiet busy "$IDLE_S"; }

stop_service() {
  [ "$STOPPED" = 1 ] && return 0
  if [ "$DRY" != 1 ] && systemctl --user is-active --quiet "$SERVICE"; then WAS_ACTIVE=1; fi
  wait_quiet clients "$IDLE_S"
  if [ "$DRY" = 1 ] || [ "$WAS_ACTIVE" = 1 ]; then run systemctl --user stop "$SERVICE"; say "$SERVICE stopped"; fi
  STOPPED=1; }

start_service() {
  if [ "$STOPPED" = 1 ] && { [ "$WAS_ACTIVE" = 1 ] || [ "$DRY" = 1 ]; }; then
    run systemctl --user start "$SERVICE"; say "$SERVICE started again"
  fi
  STOPPED=0; }

cleanup() {
  rc=$?
  trap - EXIT HUP INT TERM
  if [ -n "$CURRENT" ]; then say "interrupted in step $CURRENT: reverting"; "$TUNE" --revert all > /dev/null 2>&1 || say "REVERT FAILED: run $TUNE --revert all"; fi
  start_service
  exit $rc; }

drop_caches() { run sync; if [ "$DRY" = 1 ]; then say "[dry-run] echo 3 | sudo tee /proc/sys/vm/drop_caches"; else echo 3 | $SUDO tee /proc/sys/vm/drop_caches > /dev/null; fi; }

coldload() {   # $1 = tag: three cold starts (page cache dropped before each); load_time.py prints start seconds and the load line
  tag=$1
  say "cold pack load: $tag (3 starts)"
  for i in 1 2 3; do
    drop_caches
    run "$PYTHON" tools/pancake/load_time.py --out "$OUT/$tag-coldload-$i" --modes default --cache warm --reps 1 --no-verify
  done > "$OUT/$tag.coldload.stdout" 2>&1; }

ab_run() {   # $1 = tag
  tag=$1
  say "abtest $SPEC -> $OUT/$tag.stdout"
  if [ "$DRY" = 1 ]; then say "[dry-run] $PYTHON tools/pancake/abtest.py $SPEC --tier screen --out $OUT/$tag.jsonl"; return 0; fi
  "$PYTHON" tools/pancake/abtest.py "$SPEC" --tier screen --out "$OUT/$tag.jsonl" > "$OUT/$tag.stdout" 2>&1
  say "abtest exit $?"; }

is_cold() { case ",$COLD_STEPS," in *",$1,"*) return 0 ;; esac; return 1; }

baseline() {   # $1 = tag (baseline | baseline-end)
  tag=winH-$1
  wait_idle; stop_service
  "$TUNE" --check > "$OUT/$tag.check.txt" 2>&1
  ab_run "$tag"
  [ -n "$COLD_STEPS" ] && coldload "$tag"
  return 0; }

step_list() {   # $1 = comma list; 'all' expanded to the steps that are not at target
  echo "$1" | tr ',' ' '; }

one_step() {   # $1 = step name or all
  s=$1; tag=winH-$s; apply=$s
  if [ "$s" = all ]; then
    apply=""
    for x in $RUNTIME_STEPS; do "$TUNE" --probe "$x" > /dev/null 2>&1; [ $? = 1 ] && apply="${apply:+$apply,}$x"; done
    [ -n "$apply" ] || { say "all: every step is already at target or not applicable: skipped"; echo "SKIPPED: nothing to apply" > "$OUT/$tag.stdout"; return 0; }
  else
    "$TUNE" --probe "$s" > /dev/null 2>&1; pr=$?
    if [ "$pr" != 1 ]; then
      why="already at target"; [ "$pr" = 2 ] && why="not applicable on this host"
      say "$s: $why: skipped (no A/B possible)"
      echo "SKIPPED: $why (host_tune.sh --probe exit $pr)" > "$OUT/$tag.stdout"
      "$TUNE" --check > "$OUT/$tag.check.txt" 2>&1
      return 0
    fi
  fi
  say "=== step $s ($apply)"
  wait_idle; stop_service
  "$TUNE" --check > "$OUT/$tag.check.txt" 2>&1
  CURRENT=$s
  if ! run "$TUNE" --apply "$apply" > "$OUT/$tag.apply.txt" 2>&1; then
    say "$s: apply failed (see $OUT/$tag.apply.txt); reverting"
    run "$TUNE" --revert all >> "$OUT/$tag.apply.txt" 2>&1; CURRENT=""; echo "FAILED: apply (see .apply.txt)" > "$OUT/$tag.stdout"; return 0
  fi
  is_cold "$s" && coldload "$tag"
  ab_run "$tag"
  run "$TUNE" --revert all >> "$OUT/$tag.apply.txt" 2>&1 || say "REVERT FAILED: run $TUNE --revert all"
  CURRENT=""
  say "step $s reverted"; }

do_run() {
  mkdir -p "$OUT" 2> /dev/null || [ "$DRY" = 1 ] || die "cannot create $OUT"
  cd "$REPO" || die "no repo"
  [ -f "$SPEC" ] || die "spec $SPEC not found (run from the repo or pass --spec)"
  [ -x "$TUNE" ] || [ "$DRY" = 1 ] || die "$TUNE is not executable"
  trap cleanup EXIT HUP INT TERM
  list=${STEPARG:-$DEFAULT_STEPS}
  for s in $(step_list "$list"); do
    case " $RUNTIME_STEPS all " in *" $s "*) ;; *) die "step '$s' cannot be A/B tested here (memlock: use the mlock command)" ;; esac
  done
  [ "$BASELINE" = 1 ] && baseline baseline
  for s in $(step_list "$list"); do one_step "$s"; done
  [ "$BASELINE" = 1 ] && baseline baseline-end
  start_service
  say "done; compare with: $0 table --out $OUT"; }

do_plan() {
  list=${STEPARG:-$DEFAULT_STEPS}
  echo "host A/B plan (nothing is changed); out dir $OUT; spec $SPEC; idle wait ${IDLE_S}s before every step"
  [ "$BASELINE" = 1 ] && echo "  winH-baseline          no step (+ cold load $COLD_STEPS)"
  for s in $(step_list "$list"); do
    "$TUNE" --probe "$s" > /dev/null 2>&1; pr=$?
    case $s in all) st="every step not at target, together" ;; *) st=$([ $pr = 1 ] && echo "will run" || echo "skipped here (at target or n/a, probe exit $pr)") ;; esac
    echo "  winH-$s: $st$(is_cold "$s" && echo ' + cold pack load x3')"
  done
  [ "$BASELINE" = 1 ] && echo "  winH-baseline-end      no step (drift of the session)"
  echo "estimate: about 20 minutes per step (3 idle + 2 arms x (engine start 40-65 s + cases) + revert + 3 idle), about 4 minutes more"
  echo "for each cold load, so 8 steps + all + 2 baselines = roughly 3.5 to 4 hours; the box is exclusive for that time."; }

do_mlock() {
  cd "$REPO" || die "no repo"
  mkdir -p "$OUT" 2> /dev/null || true
  trap cleanup EXIT HUP INT TERM
  wait_idle; stop_service
  # the daily preset = pancake.service (run-pancake.sh iq2_xs fast); start it idle and let it load the arena
  run systemctl --user start "$SERVICE"; STOPPED=0
  [ "$DRY" = 1 ] && { say "[dry-run] wait until port $PORT listens, then host_tune.sh --mlock-check"; return 0; }
  i=0; until ss -Hltn "( sport = :$PORT )" 2> /dev/null | grep -q .; do
    i=$((i + 5)); [ "$i" -gt 900 ] && die "the engine did not listen on $PORT within 15 minutes"; sleep 5
  done
  sleep 20
  {
    say "mlock check, idle daily preset ($SERVICE), expected arena ${EXPECT_GIB} GiB"
    "$TUNE" --mlock-check --expect-gib "$EXPECT_GIB"; echo "exit $?"
    log=${AB_ENGINE_LOG:-$HOME/clone-of-d/src/Strata/pancake-service.log}
    if [ -r "$log" ]; then echo "engine log $log (last 600 lines): $(tail -n 600 "$log" | grep -c 'mlock failed') line(s) with 'mlock failed'"; tail -n 600 "$log" | grep -i 'mlock' | tail -n 5; fi
  } > "$OUT/winH-mlock.stdout" 2>&1
  cat "$OUT/winH-mlock.stdout"; }

do_table() {
  "$PYTHON" - "$OUT" <<'PY'
import glob, json, os, statistics, sys
out = sys.argv[1]
def means(path):
    m = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not r.get("clean", True):
            continue
        for x in r.get("rows", []):
            key = "prompt_tps" if x["case"].startswith("long") and "dec" not in x["case"] else "decode_tps"
            m.setdefault((r["arm"], x["case"]), []).append(x[key])
    return {k: statistics.mean(v) for k, v in m.items()}
files = sorted(glob.glob(os.path.join(out, "winH-*.jsonl")))
base = {}
bp = os.path.join(out, "winH-baseline.jsonl")
if os.path.exists(bp):
    base = means(bp)
cases = ["short", "edit", "long1k", "long3k"]
print("%-22s %-6s %s" % ("file", "arm", "  ".join("%-18s" % c for c in cases)))
for f in files:
    mm = means(f)
    name = os.path.basename(f)[5:-6]
    for arm in ("base", "base-again"):
        cells = []
        for c in cases:
            v = mm.get((arm, c))
            b = base.get(("base", c))
            if v is None:
                cells.append("%-18s" % "-")
            elif b and not (name == "baseline" and arm == "base"):
                cells.append("%-18s" % ("%.1f (%+.1f%%)" % (v, 100 * (v - b) / b)))
            else:
                cells.append("%-18s" % ("%.1f" % v))
        print("%-22s %-6s %s" % (name, arm, "  ".join(cells)))
print("short, edit = decode tok/s; long1k, long3k = prompt tok/s; (+x%) = against the base arm of winH-baseline;")
print("the base-again row of the same file is that file's noise floor: a step counts only above it.")
PY
}

# ---- command line
while [ $# -gt 0 ]; do
  case $1 in
    plan|run|mlock|table) [ -z "$CMD" ] && CMD=$1 || STEPARG=$1 ;;
    --out) OUT=$2; shift ;; --spec) SPEC=$2; shift ;; --idle-s) IDLE_S=$2; shift ;; --expect-gib) EXPECT_GIB=$2; shift ;;
    --dry-run|-n) DRY=1 ;; --no-baseline) BASELINE=0 ;;
    -h|--help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) die "unknown option $1" ;;
    *) STEPARG=$1 ;;
  esac
  shift
done
[ -n "$CMD" ] || { sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
case $IDLE_S in ""|*[!0-9]*) die "--idle-s needs a number" ;; esac
case $STEPARG in *[!a-z_,]*) die "bad step list" ;; esac
case $CMD in
  plan) do_plan ;; run) do_run ;; mlock) do_mlock ;; table) do_table ;;
esac
