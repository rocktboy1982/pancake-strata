#!/bin/sh
# Redo chain AA (owner standing order 2026-11-19, "be autonomous and redo this"): everything that never ran because the HIP engine of origin 2e2d2cc69 did not link (TEST_LEDGER.md T-965,
# T-968, T-969; MOD_PLAN.md section 5 "BOX STATUS 2026-11-19 09:00" re-queue list) -- chain V's recording and replays, specs 85-il-plain-diag / 88-idle-copy-diag / 88-rerun, all of tail chain X,
# and vmm_test with the per-step ZERO_BACK diagnostics.  The file name has no "win-chain" so that no other busy() pattern ever sees this script (T-944).
# Rules (standing): box jobs never overlap (3 idle minutes before the first step, 30 s between this chain's own later steps, busy() sees every other job); nothing is killed or deleted; pancake.service is
# stopped only after 3 minutes without a client on port 11437 and started again at the end; the sudoers file and permissions are never touched; code is synced with `tar xmf` semantics (mtimes reset, else
# ninja keeps stale objects); the tree synced is origin port/rdna3-decode as of the start.
# BUILD GUARD (T-965 / T-968: a failed link REMOVED build-win/strata and three later jobs died on the missing file): the old binary is copied to /mnt/pancake/chainAA/bak/ first; when the build exits
# non-zero the copy is put back (cp over, nothing removed), an ERROR line and WINCHAINAA_ABORTED are logged, the service is started and the chain STOPS with the DONE marker.  strata is checked before any later step.
# Order: 0 wait for the earlier chains' markers, stop service, sync + header | 1 build (strata, then the probe targets, build-win-il) | 2 vmm_test (STRATA_VMM_ZERO_BACK=1, per-step diagnostics) + vmm_probe probe 5b |
#        3 chain V: record real4 / real5 (window trace with router weights) and the six replays | 4 specs 85-il-plain-diag, 88-idle-copy-diag (serialized diagnostics, no confirm), 88-rerun (screen, two-pass confirm) |
#        5 tail chain X: probes, expdec ablation, 194 / 195, 196, 212 / 213, 167 soak (two-pass protocol: chainT-decide.py; pass 2 only for arms >= 1.0 % lossless / >= 1.5 % lossy; promotion bars
#        unchanged; a lossy arm is never promoted without the quality gate).
# Shortened against the old scripts (owner target <= 15 h): the duplicate pcie_link --json run is dropped (same measurement as pcie_link), the 167 race soak runs 40 starts per arm instead of 60 (a copy of
# the spec in chainAA/, the repo spec is untouched), the two diagnostic specs run their screen only.
# Completion marker: the last line of /home/dan/clone-of-d/src/redo-chainAA.log is WINCHAINAA_DONE (also after an abort; the line before it then is WINCHAINAA_ABORTED).
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/redo-chainAA.log; O=/home/dan/auto-runs; T=$O/chainAA; PM=/mnt/pancake/chainAA; C=$PM/conv
CM=/home/linuxbrew/.linuxbrew/bin/cmake; HIPCC=/opt/rocm/core-10.0/bin/hipcc
DEC=$S/chainT-decide.py; B=$S/build-win; BI=$S/build-win-il; PROF=data/expert-profile-pancake.bin
REPO_URL=https://github.com/rocktboy1982/pancake-strata.git
export PATH=$R/.venv/bin:$PATH LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib HIP_VISIBLE_DEVICES=0
mkdir -p $T $PM $PM/bak $C
echo "START $(date +%F_%T)" >> $L
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain[A-Za-z0-9]*\.sh|[x-z]tail-chain[A-Z]\.sh|[a]fter7" >/dev/null 2>&1; }
wait_idle() { if [ "${QI:-0}" = 1 ]; then q=0; while [ $q -lt 2 ]; do if busy; then q=0; sleep 15; else q=$((q+1)); sleep 15; fi; done; else q=0; while [ $q -lt 3 ]; do if busy; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; QI=1; fi; }   # quick idle: the full 3 minutes only before the first step, 30 s between this chain's own later steps
quick_idle() { while busy; do sleep 20; done; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }
freekb() { df -Pk "$1" | awk 'NR==2{print $4}'; }
say() { echo "$*" >> $L; }
lowdisk() { [ "$(freekb $S)" -lt 307200 ]; }   # < 300 MB on / : no engine step starts
abort() {  # abort MESSAGE: ERROR line + marker, service back, STOP
  say "ERROR $*"; say "WINCHAINAA_ABORTED: $*"
  systemctl --user start pancake.service; say "service started $(date +%T); disk / free $(freekb /) KB"
  echo WINCHAINAA_DONE >> $L; exit 1
}
# ------------------------------------------------------------------ helpers
pstep() {  # probe step (no 3-minute wait, only until nothing else runs); timeout 15 min
  sn=$1; so=$2; shift 2
  quick_idle; say "START $sn $(date +%T)"
  timeout 900 "$@" > $so 2>&1; rc=$?; say "END $sn exit=$rc $(date +%T)"; return $rc
}
ab() {  # ab SPECFILE PREFIX flags... : abtest with the chain-N retry on 'already running'
  sp=$1; pfx=$2; shift 2; tries=0
  while :; do
    if lowdisk; then say "SKIP $pfx: / has < 300 MB free"; return 9; fi
    wait_idle; say "START $pfx $(date +%F_%T)"
    python3 tools/pancake/abtest.py $sp "$@" --out $O/winAA-$pfx.jsonl > $O/winAA-$pfx.stdout 2>&1; rc=$?
    say "END $pfx exit=$rc $(date +%T)"
    if [ $rc -eq 3 ] && grep -q "already running" $O/winAA-$pfx.stdout && [ $tries -lt 12 ]; then tries=$((tries+1)); sleep 300; continue; fi
    return $rc
  done
}
gate() {  # gate NAME SPEC [extra]
  gn=$1; gs=$2; shift 2
  if lowdisk; then say "SKIP gate $gn: / has < 300 MB free"; return 9; fi
  wait_idle; say "START gate $gn $(date +%F_%T)"
  python3 tools/pancake/quality_gate.py bench/specs/$gs.json --out $O/winAA-$gn/gate --near-tie "$@" > $O/winAA-$gn.stdout 2>&1; rc=$?
  say "END gate $gn exit=$rc $(date +%T)"
  grep -E "PASS|REJECT|INCONCLUSIVE|near-tie|top-1" $O/winAA-$gn.stdout | tail -12 | cut -c1-200 >> $L
}
proof() {  # proof DIR ARMPREFIX GOODPATTERN [BADPATTERN]: 0 if every engine log of the arm has the good line and none the bad one
  d=$1; a=$2; g=$3; bd=$4; ok=0; n=0
  for f in $d/$a*-r*-a*.engine.log; do [ -f "$f" ] || continue; n=$((n+1))
    grep -q -- "$g" $f || { say "PROOF $a: '$g' missing in $(basename $f)"; ok=1; }
    [ -n "$bd" ] && grep -q -- "$bd" $f && { say "PROOF $a: '$bd' present in $(basename $f)"; ok=1; }
  done
  [ $n -eq 0 ] && { say "PROOF $a: no engine log under $d"; return 1; }
  [ $ok -eq 0 ] && say "PROOF $a: ok ($n logs, '$g')"; return $ok
}
autof() {  # autof NAME SPECFILE SIBLINGFILE|- REF|- [screen flags...]: screen, decision rule, confirm only for winners (two-pass)
  nm=$1; file=$2; sib=$3; rf=$4; shift 4
  [ -f "$file" ] || { say "SKIP $nm: spec file $file missing"; return 1; }
  ab $file $nm-screen "$@" || { say "screen $nm failed or skipped: no confirm"; return 1; }
  extra=""; [ "$sib" != "-" ] && extra="--confirm-spec $sib"
  [ "$rf" != "-" ] && extra="$extra --ref $rf"
  rm -f $T/$nm-confirm.json
  python3 $DEC $O/winAA-$nm-screen.stdout $file $T/$nm-confirm.json $extra > $T/$nm-decide.txt 2>&1
  cat $T/$nm-decide.txt | grep -E "WIN|CONFIRM|NONE|dropped" | cut -c1-230 >> $L
  [ -s $T/$nm-confirm.json ] || return 0
  [ -n "$PG" ] && { proof $O/winAA-$nm-screen "$PA" "$PG" "$PB" || say "PROOF $nm: informational only, the confirm still runs"; }
  [ "$PROOF_FAIL" = "1" ] && { say "confirm $nm skipped: the screen's proof line was missing (the arm is the base)"; return 0; }
  ab $T/$nm-confirm.json $nm-confirm --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5
  grep -E "vs .*(better|worse|no difference)" $O/winAA-$nm-confirm.stdout | cut -c1-200 >> $L
  return 0
}
auto() {  # auto NAME SCREENSPEC-NUMBER SIBLING-NUMBER|- REF|- [screen flags...]
  nm=$1; sp=$2; sib=$3; rf=$4; shift 4
  spec_ok $sp || { say "SKIP $nm: spec $sp missing or flagged by spec_lint"; return 1; }
  sf=""; [ "$sib" != "-" ] && sf=$(ls $R/bench/specs/$sib-*.json | head -1); [ -z "$sf" ] && sf="-"
  autof $nm $(ls $R/bench/specs/$sp-*.json | head -1) $sf $rf "$@"
}

# ------------------------------------------------------------------ 0. wait for the earlier chains, stop the service, sync, header
until grep -q WINCHAINV_DONE $S/win-chainV.log 2>/dev/null && grep -q AFTER7_DONE $O/after7.log 2>/dev/null && grep -q WINCHAINX_DONE $S/xtail-chainX.log 2>/dev/null \
      && grep -q WINCHAINY_DONE $S/ytail-chainY.log 2>/dev/null && grep -q WINCHAINZ_DONE $S/ztail-chainZ.log 2>/dev/null; do sleep 300; done
say "chains V, after7, X, Y and Z are done $(date +%T)"
PG=; PA=; PB=; PROOF_FAIL=0
q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; say "WAIT service has clients $(date +%T)"; else q=$((q+1)); fi; sleep 60; done
systemctl --user stop pancake.service; say "service stopped (idle) $(date +%T)"
cd $R; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null
wait_idle
say "SYNC start $(date +%T) df / = $(freekb /) KB, /mnt/pancake = $(freekb /mnt/pancake) KB"
SHA=$(timeout 60 git ls-remote $REPO_URL refs/heads/port/rdna3-decode 2>/dev/null | cut -f1)
SRC=$PM/src.tgz; rm -f $SRC.part
if [ -n "$SHA" ] && timeout 1200 curl -sfL -o $SRC.part https://codeload.github.com/rocktboy1982/pancake-strata/tar.gz/$SHA && tar tzf $SRC.part >/dev/null 2>&1; then
  mv $SRC.part $SRC; say "SYNC source: origin port/rdna3-decode $SHA (codeload)"
else
  abort "SYNC download failed (no fallback: an old tree lacks the HIP link fix 4011f6435 and the replay options)"
fi
[ "$(freekb /)" -gt 1000000 ] || abort "SYNC: / has < 1 GB free ($(freekb /) KB)"
tar xzmf $SRC --no-same-owner --strip-components=1 -C $R; rc=$?; say "SYNC exit=$rc ($(tar tzf $SRC | wc -l) entries) $(date +%T)"
[ $rc -eq 0 ] || abort "SYNC tar exit=$rc"
[ -f $R/tools/pancake/box_chains/chainT-decide.py ] && DEC=$R/tools/pancake/box_chains/chainT-decide.py
say "HEADER decision script: $DEC"
HAVE_TOOL=0
grep -q -- "--requests" tools/pancake/replay_fill_transition.py 2>/dev/null && grep -q -- "--order" tools/pancake/conv_trace.py 2>/dev/null && HAVE_TOOL=1
say "HEADER replay tool with --requests/--stwt and conv_trace --order present: $HAVE_TOOL"
python3 tools/pancake/spec_lint.py --dir bench/specs --quiet > $T/spec_lint.txt 2>&1; say "HEADER spec_lint exit=$? ($(grep -c PHANTOM $T/spec_lint.txt) phantom lines)"
lint_bad() { grep -Eq "^[A-Za-z-]+ +$1-" $T/spec_lint.txt; }
spec_ok() { ls $R/bench/specs/$1-*.json >/dev/null 2>&1 && ! lint_bad $1; }
python3 tools/pancake/test_disk_env_table.py > $T/disk_env_table.txt 2>&1; say "HEADER disk_env_table exit=$?"
python3 -m pytest -q tools/pancake/test_replay_fill_transition.py > $T/pytest.txt 2>&1; say "HEADER pytest replay_fill_transition exit=$? $(tail -1 $T/pytest.txt | cut -c1-120)"

# ------------------------------------------------------------------ 1. build (the old binaries are backed up; a failed build puts them back and stops the chain)
[ "$(freekb /)" -ge 2097152 ] || abort "BUILD: / has $(freekb /) KB free (< 2 GB); nothing was deleted"
[ -x $B/strata ] && cp -p $B/strata $PM/bak/strata.prev && say "BUILD backup of the previous build-win/strata: $(stat -c %s $PM/bak/strata.prev) bytes"
wait_idle; say "BUILD build-win strata $(date +%T) (/ free $(freekb /) KB)"
nice -n 10 $CM --build $B -j8 --target strata > $S/build-win.chainAA.log 2>&1; rc=$?
say "BUILD_END build-win exit=$rc $(date +%T) (/ free $(freekb /) KB)"
if [ $rc -ne 0 ]; then
  grep -m8 -E "error:|undefined|CMake Error" $S/build-win.chainAA.log >> $L
  if [ -f $PM/bak/strata.prev ]; then cp -p $PM/bak/strata.prev $B/strata && say "BUILD FAILED: the previous build-win/strata was restored from $PM/bak/strata.prev"; fi
  abort "BUILD FAILED exit=$rc (log $S/build-win.chainAA.log); the chain stops, nothing after it ran"
fi
[ -x $B/strata ] || abort "BUILD exit=0 but $B/strata is missing"
ENGINE=0; [ -x $B/strata ] && ENGINE=1
if [ $ENGINE -eq 1 ]; then say "ENGINE=1 (build-win/strata present: yes)"; else say "ENGINE=0 (build-win/strata present: NO)"; fi
say "HEADER STRATA_WINDOW_TRACE_WEIGHTS strings in build-win/strata: $(grep -c STRATA_WINDOW_TRACE_WEIGHTS $B/strata)"
# the probe targets and vmm_test (their old binaries stay when a target fails: a failure here is logged, not fatal)
PTARGETS="vmm_test act_f16_probe link_touch_probe wmma_rate_probe int4_wmma_probe int4_moe_probe attn_fold_probe raw_buffer_probe persist_tape_probe persist_layer_parity retained_replay_probe expdec_bench"
for t in $PTARGETS; do
  [ -f $B/build.ninja ] && ! grep -q "^build $t:" $B/build.ninja && { say "BUILD target $t: not a target of this tree, existing binary kept"; continue; }
  quick_idle; nice -n 10 $CM --build $B -j8 --target $t > $T/build-$t.log 2>&1; rc=$?; say "BUILD_END target $t exit=$rc"
  [ $rc -ne 0 ] && grep -m4 -E "error:|undefined|CMake Error" $T/build-$t.log >> $L
done
IL=0
if [ -d $BI ]; then
  [ -x $BI/strata ] && cp -p $BI/strata $PM/bak/strata-il.prev
  [ -x $BI/mmvq_il_parity ] && cp -p $BI/mmvq_il_parity $PM/bak/mmvq_il_parity.prev
  quick_idle; say "BUILD build-win-il strata mmvq_il_parity $(date +%T)"
  nice -n 10 $CM --build $BI -j8 --target strata mmvq_il_parity > $S/build-win-il.chainAA.log 2>&1; rc=$?
  say "BUILD_END build-win-il exit=$rc $(date +%T)"
  if [ $rc -eq 0 ] && [ -x $BI/strata ]; then IL=1
  else
    grep -m6 -E "error:|undefined|CMake Error" $S/build-win-il.chainAA.log >> $L
    [ -f $PM/bak/strata-il.prev ] && cp -p $PM/bak/strata-il.prev $BI/strata
    [ -f $PM/bak/mmvq_il_parity.prev ] && cp -p $PM/bak/mmvq_il_parity.prev $BI/mmvq_il_parity
    say "WARN build-win-il failed: spec 85-il-plain-diag is skipped (the old il binaries were put back)"
  fi
else say "WARN $BI missing: spec 85-il-plain-diag is skipped"; fi

# ------------------------------------------------------------------ 2. vmm_test with the per-step ZERO_BACK diagnostics (T-950), vmm_probe probe 5b
grep -q "ZERO_BACK diag" $R/tests/core/vmm_test.cpp 2>/dev/null || say "vmm_test: the synced tree has no ZERO_BACK diagnostics; the old test runs"
if [ -x $B/vmm_test ]; then
  pstep vmm_test_zero $T/vmm_test_zero.out env STRATA_VMM_ZERO_BACK=1 $B/vmm_test
  grep -E "zero-back|vmm_test:|FAIL|strata:" $T/vmm_test_zero.out | head -80 | cut -c1-220 >> $L
else say "vmm_test_zero skipped: $B/vmm_test is missing"; fi
VP=/mnt/pancake/chainT/vmm_probe
if [ -f $R/tools/pancake/vmm_probe.cpp ] && $HIPCC -O2 -std=c++17 --offload-arch=gfx1100 $R/tools/pancake/vmm_probe.cpp -o $PM/vmm_probe > $T/vmm_probe_build.log 2>&1 && [ -x $PM/vmm_probe ]; then VP=$PM/vmm_probe; say "vmm_probe rebuilt from the tree"; else say "vmm_probe: rebuild failed or missing, using chain T's binary"; fi
if [ -x $VP ]; then pstep vmm_probe $T/vmm_probe.out $VP 64 512; grep -E "5b" $T/vmm_probe.out | head -20 | cut -c1-220 >> $L
else say "vmm_probe skipped: $VP is missing"; fi

# ------------------------------------------------------------------ 3. chain V: multi-conversation recording with router weights, then the replays (CPU only)
rec() {  # rec TAG [conv_trace flags]: dump + window trace with weights + sidecar
  tag=$1; shift
  if lowdisk; then say "SKIP record $tag: / has < 300 MB free"; return 9; fi
  wait_idle; say "START record $tag $(date +%F_%T)"
  timeout 5400 python3 tools/pancake/conv_trace.py $C/config.json $C --max-new 384 --tag $tag \
      --window-trace $C/$tag.stwt --engine-env STRATA_WINDOW_TRACE_WEIGHTS=1 "$@" > $C/$tag.stdout 2>&1; rc=$?
  say "END record $tag exit=$rc $(date +%T) ($(wc -l < $C/$tag.requests.json 2>/dev/null) lines of sidecar, trace $(stat -c %s $C/$tag.trace 2>/dev/null) bytes, stwt $(stat -c %s $C/$tag.stwt 2>/dev/null) bytes)"
  grep "STRATA_WINDOW_TRACE" $C/$tag.engine.log 2>/dev/null | head -3 | cut -c1-200 >> $L
  return $rc
}
rep() {  # rep NAME TRACE REQUESTS STWT-or-none SLOTS [more flags]
  nm=$1; tr=$2; rq=$3; st=$4; sl=$5; shift 5
  [ -s "$tr" ] && [ -s "$rq" ] || { say "SKIP replay $nm: no trace or sidecar"; return 9; }
  stw=""; [ "$st" != none ] && [ -s "$st" ] && stw="--stwt $st"
  wait_idle; say "START replay $nm $(date +%T)"
  python3 tools/pancake/replay_fill_transition.py $tr $PROF $sl --requests $rq $stw --json $O/winAA-replay-$nm.json "$@" > $O/winAA-replay-$nm.txt 2>&1; rc=$?
  say "END replay $nm exit=$rc $(date +%T)"
  head -1 $O/winAA-replay-$nm.txt | cut -c1-330 >> $L
  grep -E "^(base|pm|pmw|ff:0.7|tr:2.0|trb:2.0|ff:0.7\+tr:2.0|ff:0.7\+pmw|tr:2.0\+pmw|trb:2.0\+pmw) " $O/winAA-replay-$nm.txt | cut -c1-90 >> $L
  return $rc
}
if [ $ENGINE -eq 1 ] && [ $HAVE_TOOL -eq 1 ]; then
  sed 's#/Strata/build-hip/strata#/build-win/strata#' $S/Strata/pancake-flashnext-iq2_xs-fast-hip.json > $C/config.json
  rec real4
  rec real5 --order AAABBCCDD
else
  say "SKIP record: no engine build or no new tools in the tree"
fi
if [ $HAVE_TOOL -eq 1 ]; then
  for t in real4 real5; do
    rep $t-main $C/$t.trace $C/$t.requests.json $C/$t.stwt 10909 --sweep
    rep $t-every1 $C/$t.trace $C/$t.requests.json $C/$t.stwt 10909 --every 1
    rep $t-slots5000 $C/$t.trace $C/$t.requests.json $C/$t.stwt 5000
  done
  W16=$O/win16-conv
  rep real1-chain16a $W16/real1.trace $W16/real1.requests.json none 10909 --sweep
fi
RD=$R/bench/results/2026-11-19-chainAA; mkdir -p $RD
for f in $O/winAA-replay-*.txt $O/winAA-replay-*.json $C/real4.requests.json $C/real5.requests.json $C/real4.stdout $C/real5.stdout \
         $C/real4.engine.log $C/real5.engine.log $T/pytest.txt $T/disk_env_table.txt $T/spec_lint.txt; do
  [ -f "$f" ] && cp -n "$f" $RD/ 2>/dev/null
done
say "COPY small outputs to $RD ($(ls $RD | wc -l) files); big files stay: $C/real4.trace real4.stwt real5.trace real5.stwt"

# ------------------------------------------------------------------ 4. specs 85-il-plain-diag, 88-idle-copy-diag, 88-rerun (the box agent's spec files of after7.sh, read in place)
cd $R
AS=$O/after7
# the two diagnostics run with serialized kernels + verify trace as after7.sh had them: their timings are NOT comparable, so the screen only, no decision, no confirm
if [ $IL -eq 1 ] && [ -f $BI/mmvq_il_parity ]; then
  pstep mmvq_il_parity_plain $T/mmvq_il_parity-plain.log env STRATA_MMVQ_IL_PLAIN=1 HIP_LAUNCH_BLOCKING=1 $BI/mmvq_il_parity --selftest
  tail -n 2 $T/mmvq_il_parity-plain.log | cut -c1-160 >> $L
fi
export HIP_LAUNCH_BLOCKING=1 AMD_SERIALIZE_KERNEL=3 STRATA_VERIFY_TRACE=1
if [ $IL -eq 1 ] && [ -f $AS/85-il-plain-diag.json ]; then
  ab $AS/85-il-plain-diag.json 85-il-plain-diag --tier screen
  grep -E "vs .*(better|worse|no difference)|crash|already running|no clean runs" $O/winAA-85-il-plain-diag.stdout | cut -c1-200 >> $L
else say "SKIP 85-il-plain-diag: build-win-il not usable or $AS/85-il-plain-diag.json missing"; fi
if [ -f $AS/88-idle-copy-diag.json ]; then
  ab $AS/88-idle-copy-diag.json 88-idle-copy-diag --tier screen
  grep -E "vs .*(better|worse|no difference)|crash|already running|no clean runs" $O/winAA-88-idle-copy-diag.stdout | cut -c1-200 >> $L
else say "SKIP 88-idle-copy-diag: $AS/88-idle-copy-diag.json missing"; fi
unset HIP_LAUNCH_BLOCKING AMD_SERIALIZE_KERNEL STRATA_VERIFY_TRACE
# 88-rerun: normal environment; lossy arms (speed only: a promoted arm needs the quality gate before adoption), two-pass confirm for arms >= 1.5 %
autof 88-rerun $AS/88-rerun.json - - --tier screen

# ------------------------------------------------------------------ 5. tail chain X: probes, expdec ablation, 194 / 195, 196, 212 / 213, 167
if [ $ENGINE -eq 1 ]; then
  wait_idle
  pstep act_f16 $T/act_f16.out $B/act_f16_probe --reps 200
  pstep link_touch $T/link_touch.out $B/link_touch_probe --us 4000 --reps 400 --touch-us 5
  pstep pcie_link $T/pcie_link.out python3 tools/pancake/pcie_link.py
  pstep wmma_rate $T/wmma_rate.out $B/wmma_rate_probe --iters 20000 --reps 7
  pstep int4_wmma_parity $T/int4_wmma_parity.out $B/int4_wmma_probe --parity-only; rcp=$?
  if [ $rcp -eq 0 ]; then
    pstep int4_wmma_gateup_a8 $T/int4_wmma_gateup_a8.out $B/int4_wmma_probe --m 640 --k 2560 --n 512 --block 256 --a4 0 --reps 20
    pstep int4_wmma_gateup_a4 $T/int4_wmma_gateup_a4.out $B/int4_wmma_probe --m 640 --k 2560 --n 512 --block 256 --a4 1 --reps 20
    pstep int4_wmma_down_a8 $T/int4_wmma_down_a8.out $B/int4_wmma_probe --m 2560 --k 640 --n 512 --block 128 --a4 0 --reps 20
    pstep int4_wmma_down_a4 $T/int4_wmma_down_a4.out $B/int4_wmma_probe --m 2560 --k 640 --n 512 --block 128 --a4 1 --reps 20
  else say "int4_wmma bench skipped: parity exit=$rcp"; fi
  pstep int4_moe_parity $T/int4_moe_parity.out $B/int4_moe_probe --parity-only; rcp=$?
  if [ $rcp -eq 0 ]; then
    pstep int4_moe_gateup $T/int4_moe_gateup.out $B/int4_moe_probe --m 512 --k 2560 --tokens 2048 --experts 64 --block 256 --reps 20
    pstep int4_moe_down $T/int4_moe_down.out $B/int4_moe_probe --m 512 --k 640 --tokens 2048 --experts 64 --block 128 --reps 20
  else say "int4_moe bench skipped: parity exit=$rcp"; fi
  pstep attn_fold $T/attn_fold.out $B/attn_fold_probe --cap 8192 --ctx 8192 --layers 40 --reps 20 --launches 200
  pstep raw_buffer $T/raw_buffer.out $B/raw_buffer_probe --rows 65536 --rowbytes 640 --reps 30; rcp=$?
  [ $rcp -ne 0 ] && pstep raw_buffer_flags2 $T/raw_buffer_flags2.out $B/raw_buffer_probe --rows 65536 --rowbytes 640 --reps 30 --flags 0x31014000
  pstep persist_tape $T/persist_tape.out $B/persist_tape_probe --segments 432 --tasks 192 --wbytes 16384 --grids 96,192,288,384 --reps 20
  pstep persist_layer_parity $T/persist_layer_parity.out $B/persist_layer_parity
  pstep retained_replay $T/retained_replay.out $B/retained_replay_probe --n 2248,2648 --blocks 96 --threads 256 --reps 20 --window-ms 24.8
  # expdec ablation (LEADS O14): default = build-win's expdec_bench; the two ablation builds go to /mnt/pancake
  AB=$PM/ablate; mkdir -p $AB
  vbuild() {  # vbuild DIR TARGET cmake-extra... (configure like chain 11's variants, build one target)
    vd=$1; vt=$2; shift 2
    if [ "$(freekb /mnt/pancake)" -lt 8000000 ]; then say "VARIANT $vd skipped: /mnt/pancake has < 8 GB free"; return 1; fi
    wait_idle; say "CONFIGURE+BUILD $vd $(date +%T)"
    $CM -S $R -B $vd -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
      -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_HIP_COMPILER=/opt/rocm/core-10.0/lib/llvm/bin/clang++ \
      -DCMAKE_MAKE_PROGRAM=$S/Strata/.venv/bin/ninja -DCMAKE_PREFIX_PATH=/opt/rocm/core-10.0 \
      -DSTRATA_GGML_DIR=/home/dan/clone-of-d/build-int/_deps/strata_llamacpp-src -DSTRATA_AMD_COMMUNITY_TUNING=ON \
      -DSTRATA_MMQ_KQUANTS=ON -DSTRATA_MMQ_TQ_OVERLAY=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PREFILL_MMQ=ON \
      -DSTRATA_WINDOW_MAX_T=auto -DSTRATA_PLE_FIXTURE_DIR=$R/bench/micro "$@" > $vd.log 2>&1 && \
      nice -n 10 $CM --build $vd -j8 --target $vt >> $vd.log 2>&1
    rc=$?; say "BUILD_END $vd exit=$rc $(date +%T)"; [ $rc -ne 0 ] && grep -m6 -E "error:|undefined|CMake Error" $vd.log >> $L
    return $rc
  }
  PAIRS=16/42,17/42,22/42
  pstep ablate_default $T/ablate-default.txt $B/expdec_bench 300 0 20 $PAIRS def,dec
  vbuild $AB/nomac expdec_bench "-DCMAKE_HIP_FLAGS=-DSTRATA_KERNEL_ABLATE=1" && pstep ablate_nomac $T/ablate-nomac.txt $AB/nomac/expdec_bench 300 0 20 $PAIRS def,dec
  vbuild $AB/samerow expdec_bench "-DCMAKE_HIP_FLAGS=-DSTRATA_KERNEL_ABLATE=2" && pstep ablate_samerow $T/ablate-samerow.txt $AB/samerow/expdec_bench 300 0 20 $PAIRS def,dec
  say "START ablate_expdec report $(date +%T)"
  python3 tools/pancake/ablate_expdec.py report $T/ablate-default.txt $T/ablate-nomac.txt $T/ablate-samerow.txt > $T/ablate-report.txt 2>&1; say "END ablate_expdec report exit=$?"

  # 194 / 195: build macros, one binary per arm (variant builds on /mnt/pancake, linked as $S/build-<name>)
  if spec_ok 194; then
    OKV=1
    for v in "ldsbar -DCMAKE_HIP_FLAGS=-DSTRATA_LDS_ONLY_BARRIER" "lazy2 -DCMAKE_HIP_FLAGS=-DSTRATA_ATTN_LAZY_RESCALE=2" \
             "lazy8 -DCMAKE_HIP_FLAGS=-DSTRATA_ATTN_LAZY_RESCALE=8" "lazy2ns -DCMAKE_HIP_FLAGS=-DSTRATA_ATTN_LAZY_RESCALE=2+-DSTRATA_ATTN_LAZY_NOSKIP" \
             "xlds -DCMAKE_HIP_FLAGS=-DSTRATA_PF_MOE_XLDS"; do
      nm=${v%% *}; fl=${v#* }; fl=$(echo "$fl" | tr '+' ' ')
      [ -e $S/build-$nm ] && [ ! -L $S/build-$nm ] && { say "VARIANT build-$nm: a real directory already exists at $S/build-$nm, left alone"; OKV=0; continue; }
      # always rebuilt from the synced tree (an old variant binary would be a stale arm); the directory is reused, so ninja only recompiles what changed
      mkdir -p $PM/builds; vbuild $PM/builds/build-$nm strata "$fl" || OKV=0
      [ -x $PM/builds/build-$nm/strata ] && ln -sfn $PM/builds/build-$nm $S/build-$nm
    done
    if [ $OKV -eq 1 ]; then
      auto 194 194 - - --tier screen --drift-check
      grep -q "lazy" $T/194-confirm.json 2>/dev/null && spec_ok 195 && gate 195 195-prompt-lazy-quality
    else say "SKIP 194 / 195: a variant build failed (arms need one binary each)"; fi
  fi
  # 196: the adaptive machinery on a repeated vs a rotated prompt (A/A), two confirm runs
  if spec_ok 196; then
    ab $(ls $R/bench/specs/196-*.json) 196-plain --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5
    grep -E "vs .*(better|worse|no difference)" $O/winAA-196-plain.stdout | cut -c1-200 >> $L
    ab $(ls $R/bench/specs/196-*.json) 196-rotated --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5 --rotate-prompts
    grep -E "vs .*(better|worse|no difference)" $O/winAA-196-rotated.stdout | cut -c1-200 >> $L
  fi
  # 212 / 213: statistical expert cache, screen then confirm for winners only (the offline replay T-929..T-931 predicts no gain)
  if spec_ok 212 && spec_ok 213; then
    SPEC=$(ls $R/bench/specs/212-*.json | head -1); SIB=$(ls $R/bench/specs/213-*.json | head -1)
    ab $SPEC 212-screen --tier screen --drift-check
    PROOF_FAIL=0
    proof $O/winAA-212-screen ff70 "STRATA_ADAPT_FILL_FIRST=0.7:" || PROOF_FAIL=1
    proof $O/winAA-212-screen tr2 "STRATA_ADAPT_TRANSITION=2:" || PROOF_FAIL=1
    grep -h "tier stat:" $O/winAA-212-screen/*.engine.log 2>/dev/null | sort | uniq -c | sort -rn | head -6 | cut -c1-200 >> $L
    rm -f $T/212-confirm.json
    python3 $DEC $O/winAA-212-screen.stdout $SPEC $T/212-confirm.json --confirm-spec $SIB > $T/212-decide.txt 2>&1
    grep -E "WIN|CONFIRM|NONE|dropped" $T/212-decide.txt | cut -c1-230 >> $L
    if [ -s $T/212-confirm.json ] && [ $PROOF_FAIL -eq 0 ]; then
      ab $T/212-confirm.json 212-confirm --tier confirm --block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 5
      grep -E "vs .*(better|worse|no difference)" $O/winAA-212-confirm.stdout | cut -c1-200 >> $L
    elif [ -s $T/212-confirm.json ]; then say "confirm 212 skipped: a proof line was missing (an arm is the base)"; fi
  fi
  # 167: race soak, the longest step: last.  40 starts of the edit case per arm (the repo spec says 60: shortened for the 15 h target), same binary twice
  if spec_ok 167; then
    S167=$(ls $R/bench/specs/167-*.json | head -1)
    sed 's/"reps": 60,/"reps": 40,/' $S167 > $T/167-soak40.json
    grep -q '"reps": 40,' $T/167-soak40.json && S167=$T/167-soak40.json
    say "167 soak spec: $S167"
    ab $S167 167-soak --tier confirm --outlier robust --no-early-stop
    grep -A4 "generated text vs base" $O/winAA-167-soak.stdout | cut -c1-200 >> $L
  fi
fi
systemctl --user start pancake.service; say "service started $(date +%T); disk / free $(freekb /) KB"
echo WINCHAINAA_DONE >> $L
