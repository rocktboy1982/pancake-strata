#!/bin/sh
# bench-ablate-f16.sh - STAGE ABLATION LADDER of the F16 prompt-expert kernel (LEADS.md Part U, lead U5; TEST_LEDGER.md T-1070; spec 276).  NOT RUN: written on the
# Windows PC, the box was busy with the flag screen.  IDLE ONLY: it waits (quick idle, as the other chains) until none of the benchmark / build jobs of the box
# runs, starts nothing while the sweep queue (run-queue-us*.sh, bench-sweep*.sh, bench-sched*.sh) is alive, deletes NOTHING (every directory it makes is new; an
# existing run directory stops it), and never edits a queue file.  The name has no "win-chain" / "tail-chain": it is started by hand (or by the scheduler as
# BENCH_EXTRA), and its steps are seen as busy by the other chains through abtest.py / ninja / cmake like any benchmark step.
#
# What it does (ABL_PHASE=build|run|report|all, default all):
#   build : ONE new build directory $PM/build (configure like the chain-X variant builds, gfx1100, HIP, ROCm core-10.0), one variant after the other:
#           cmake -DSTRATA_BENCH_ABLATION_DEFS="<macros>" and `cmake --build --target strata`; only moe_fused_iq.cu and moe_f16_kq.cu recompile (the macros are
#           defined for those two translation units alone, CMakeLists.txt), then the binary is COPIED to $PM/bin/<arm>/strata (+ symlinks to the build directory's
#           shared libraries, the macros in MACROS, a sha256).  Arm `base` = empty DEFS = the engine unchanged.  14 arms (python3 tools/pancake/f16_ablate.py
#           variants): base, noload nodeq nowiden nostore noread nowmma noepi (one stage removed each), cum2..cum7 (cumulative ladder).  ABL_SET=singles: the first 8.
#   run   : spec 276 (bench/specs/276-f16-ablation-ladder.json: base-late drift arm, cases long3k / long / long32k / long64k, STRATA_PF_F16=1 in every arm) with
#           tools/pancake/abtest.py --tier screen, ABL_RUNS times (default 2: the owner protocol is a two-run verdict), to $OUT/run<N>.jsonl.  The text of every
#           ablated arm is WRONG BY DESIGN: only prompt tok/s is read; no drift check of text and no quality gate belongs to this spec.
#   report: python3 tools/pancake/f16_ablate.py report $OUT/run1.jsonl $OUT/run2.jsonl > $OUT/ladder.txt (and ladder.json).
# Start (idle box, nothing else queued):  cd <clone of the branch> && ABL_BRANCH=lead/U5-U2 setsid nohup sh tools/pancake/box_chains/bench-ablate-f16.sh > /dev/null 2>&1 &
# Log $S/bench-ablate-f16.log, last line ABLATEF16_DONE (also after an abort: the line before is ABLATEF16_ABORTED <why>).
# ABL_DRY=1 prints the plan (BUILD / RUN lines) and exits: nothing is built, no pgrep, no service touched (the host test uses it).
# Rules kept: no box job overlaps (busy() sees every other job); pancake.service is stopped only with ABL_STOP_SERVICE=1 and only after 3 minutes without a client,
# and started again only if this script stopped it; the AA shell is never signalled.
S=${ABL_S:-/home/dan/clone-of-d/src}; L=${ABL_LOG:-$S/bench-ablate-f16.log}; PM=${ABL_PM:-/mnt/pancake/ablate-f16}; OUTROOT=${ABL_OUTROOT:-/home/dan/auto-runs/ablate-f16}
BRANCH=${ABL_BRANCH:-lead/U5-U2}; REPO_URL=${ABL_REPO_URL:-https://github.com/rocktboy1982/pancake-strata.git}
ROCM=${ABL_ROCM:-/opt/rocm/core-10.0}; CM=${ABL_CMAKE:-/home/linuxbrew/.linuxbrew/bin/cmake}; NINJA=${ABL_NINJA:-$S/Strata/.venv/bin/ninja}
GGML=${ABL_GGML:-/home/dan/clone-of-d/build-int/_deps/strata_llamacpp-src}; HPY=${ABL_PY:-$S/Strata-win/.venv/bin/python}
PY3=${ABL_PY3:-python3}; PHASE=${ABL_PHASE:-all}; SET=${ABL_SET:-full}; RUNS=${ABL_RUNS:-2}; DRY=${ABL_DRY:-0}; TAG=${ABL_TAG:-$(date +%F)}
OUT=$OUTROOT/$TAG; SRC=$PM/src; BLD=$PM/build; BIN=$PM/bin; SPEC=$SRC/bench/specs/276-f16-ablation-ladder.json
HERE=$(cd "$(dirname "$0")" && pwd); TOOL=$HERE/../f16_ablate.py
say() { if [ "$DRY" = 1 ]; then echo "$*"; else echo "$*" >> $L; fi; }
abort() { say "ABLATEF16_ABORTED $*"; say "ABLATEF16_DONE"; exit 1; }
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[t]tft_bench.py|[w]in-chain[A-Za-z0-9]*\.sh|[x-z]tail-chain[A-Z]\.sh|[r]un-queue-us[0-9a-z-]*\.sh|[b]ench-(sweep|sched|retest|final-us|exact|diagUS|updated(AA|RD))[.a-z0-9]*\.sh" >/dev/null 2>&1; }
quick_idle() { q=0; while [ $q -lt 2 ]; do if busy; then q=0; sleep 15; else q=$((q+1)); sleep 15; fi; done; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }
freekb() { df -Pk "$1" | awk 'NR==2{print $4}'; }
STOPPED=0
TAB=$(printf '	')

variants() { "$PY3" "$TOOL" variants --set "$SET" | tr -d '\r'; }   # name<TAB>macro;macro

if [ "$DRY" = 1 ]; then
  echo "PLAN branch=$BRANCH set=$SET runs=$RUNS phase=$PHASE pm=$PM out=$OUT"
  variants | while IFS="$TAB" read -r name defs; do echo "BUILD $name DEFS=[$defs] -> $BIN/$name/strata"; done
  r=1; while [ $r -le $RUNS ]; do echo "RUN abtest.py $SPEC --tier screen --out $OUT/run$r.jsonl"; r=$((r+1)); done
  echo "REPORT f16_ablate.py report $OUT/run*.jsonl"
  exit 0
fi

echo "START $(date +%F_%T) phase=$PHASE set=$SET runs=$RUNS branch=$BRANCH" >> $L
[ -x "$HPY" ] || abort "harness python $HPY missing (ABL_PY)"
[ -x "$NINJA" ] || abort "ninja $NINJA missing (ABL_NINJA)"
[ -d "$GGML" ] || abort "ggml source $GGML missing (ABL_GGML)"
mkdir -p $PM $OUTROOT || abort "cannot make $PM / $OUTROOT"

# ---- source: a new clone of the branch (an existing one is fetched and checked out detached; nothing is deleted)
if [ ! -d $SRC/.git ]; then
  git clone --quiet $REPO_URL $SRC >> $L 2>&1 || abort "clone of $REPO_URL failed"
fi
git -C $SRC fetch --quiet origin "$BRANCH" >> $L 2>&1 || abort "fetch of $BRANCH failed"
git -C $SRC checkout --quiet --detach FETCH_HEAD >> $L 2>&1 || abort "checkout of $BRANCH failed"
say "SOURCE $(git -C $SRC rev-parse --short HEAD) ($BRANCH)"
[ -f $SPEC ] || abort "spec 276 missing in the checkout"

build_one() {  # build_one NAME DEFS(;-separated)
  nm=$1; defs=$2
  if [ -x $BIN/$nm/strata ]; then say "BUILD $nm: already built, kept"; return 0; fi
  if [ "$(freekb $PM)" -lt 8000000 ]; then say "BUILD $nm: $PM has < 8 GB free"; return 1; fi
  quick_idle; say "BUILD $nm DEFS=[$defs] $(date +%T)"
  $CM -S $SRC -B $BLD -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
    -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=$ROCM/lib/llvm/bin/clang++ \
    -DCMAKE_MAKE_PROGRAM=$NINJA -DCMAKE_PREFIX_PATH=$ROCM \
    -DSTRATA_GGML_DIR=$GGML -DSTRATA_AMD_COMMUNITY_TUNING=ON \
    -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_TQ_OVERLAY=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=ON \
    -DSTRATA_WINDOW_MAX_T=auto -DSTRATA_PLE_FIXTURE_DIR=$SRC/bench/micro "-DSTRATA_BENCH_ABLATION_DEFS=$defs" > $PM/configure-$nm.log 2>&1 \
    && nice -n 10 $CM --build $BLD -j8 --target strata > $PM/build-$nm.log 2>&1
  rc=$?; say "BUILD_END $nm exit=$rc $(date +%T)"
  if [ $rc -ne 0 ]; then grep -m6 -E "error:|undefined|CMake Error" $PM/build-$nm.log $PM/configure-$nm.log >> $L; return $rc; fi
  # the configure line above must be the one of the box's build-win (the same switches): the proof is a DEFS warning in the configure log for ablated arms only
  if [ -n "$defs" ]; then grep -q "STRATA_BENCH_ABLATION_DEFS=$defs" $PM/configure-$nm.log || { say "BUILD $nm: the ablation warning is missing in configure-$nm.log"; return 1; }
  else grep -q "STRATA_BENCH_ABLATION_DEFS=" $PM/configure-$nm.log && { say "BUILD $nm: base must not be ablated"; return 1; }; fi
  mkdir -p $BIN/$nm && cp $BLD/strata $BIN/$nm/strata.tmp && mv $BIN/$nm/strata.tmp $BIN/$nm/strata || return 1
  for lib in $BLD/*.so*; do [ -e "$lib" ] && ln -sfn "$lib" $BIN/$nm/$(basename "$lib"); done
  echo "$defs" > $BIN/$nm/MACROS; sha256sum $BIN/$nm/strata > $BIN/$nm/SHA256
  say "BIN $nm $(cut -c1-16 $BIN/$nm/SHA256) ldd-missing: $(ldd $BIN/$nm/strata 2>/dev/null | grep -c 'not found')"
  return 0
}

if [ "$PHASE" = build ] || [ "$PHASE" = all ]; then
  OKB=1
  variants | while IFS="$TAB" read -r name defs; do echo "$name$TAB$defs"; done > $PM/variants.txt
  while IFS="$TAB" read -r name defs; do build_one "$name" "$defs" || OKB=0; done < $PM/variants.txt
  # base and every ablation must have DIFFERENT binaries (a failed define would give 14 copies of the base)
  n=$(cat $BIN/*/SHA256 2>/dev/null | cut -c1-64 | sort -u | wc -l); say "DISTINCT binaries: $n"
  [ $OKB -eq 1 ] || abort "a variant build failed (see the log); the run needs every arm"
fi

if [ "$PHASE" = run ] || [ "$PHASE" = all ]; then
  "$PY3" - "$SPEC" "$BIN" <<'PY' || abort "spec 276 names an arm whose binary is missing"
import json, os, sys
spec = json.load(open(sys.argv[1]))
missing = [a for a, v in spec["arms"].items() if not os.path.exists(v["exe"].replace("/mnt/pancake/ablate-f16/bin", sys.argv[2]))]
sys.exit(1 if missing else 0)
PY
  mkdir -p $OUT || abort "cannot make $OUT"
  if systemctl --user is-active --quiet pancake.service 2>/dev/null; then
    [ "${ABL_STOP_SERVICE:-0}" = 1 ] || abort "pancake.service is running (set ABL_STOP_SERVICE=1 to let this script stop it when it has no client)"
    q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; say "WAIT service has clients $(date +%T)"; else q=$((q+1)); fi; sleep 60; done
    systemctl --user stop pancake.service && STOPPED=1; say "service stopped (idle) $(date +%T)"
  fi
  export ABTEST_PYTHON=$HPY
  # the spec's exe paths are under /mnt/pancake/ablate-f16/bin; ABL_PM elsewhere: a rewritten copy goes to $OUT
  RSPEC=$SPEC
  if [ "$PM" != /mnt/pancake/ablate-f16 ]; then RSPEC=$OUT/spec-276.json; sed "s#/mnt/pancake/ablate-f16/bin#$BIN#g" $SPEC > $RSPEC; fi
  cd $SRC || abort "cd $SRC"
  r=1
  while [ $r -le $RUNS ]; do
    [ -e $OUT/run$r.jsonl ] && abort "$OUT/run$r.jsonl exists (never overwritten: set ABL_TAG)"
    quick_idle; say "RUN $r START $(date +%T)"
    "$PY3" tools/pancake/abtest.py $RSPEC --tier screen --out $OUT/run$r.jsonl > $OUT/run$r.stdout 2>&1; rc=$?
    say "RUN $r END exit=$rc $(date +%T)"
    r=$((r+1))
  done
  [ $STOPPED -eq 1 ] && { systemctl --user start pancake.service; say "service started again"; }
fi

if [ "$PHASE" = report ] || [ "$PHASE" = all ]; then
  set -- $OUT/run*.jsonl
  [ -e "$1" ] || abort "no $OUT/run*.jsonl to report"
  "$PY3" "$TOOL" report --set "$SET" "$@" > $OUT/ladder.txt 2>&1
  "$PY3" "$TOOL" report --set "$SET" --json "$@" > $OUT/ladder.json 2>&1
  say "REPORT $OUT/ladder.txt ($(grep -c '^verdict' $OUT/ladder.txt) verdict lines)"
  grep '^verdict' $OUT/ladder.txt | cut -c1-200 >> $L
fi
say "ABLATEF16_DONE"
