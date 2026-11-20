#!/bin/sh
# bench-sweep.sh - FULL FLAG SWEEP of one updated branch (owner order 2026-11-20, autonomy: "test all the coded flags once, then triage which to retest,
# don't keep low performers").  The file name has no "win-chain", "redo-chain" or "tail-chain": no busy() pattern of another chain sees it (T-944).
# Log /home/dan/clone-of-d/src/bench-sweep.log, last line WINBENCHSWEEP_DONE (also after an abort or a signal; the line before it is then
# WINBENCHSWEEP_ABORTED).  Results: /home/dan/auto-runs/benchsweep/ (<branch>-p1-<slice>.jsonl, <branch>-p2-<slice>.jsonl, <branch>-triage-<slice>.json and
# the merged <branch>-triage.json: per flag the pass-1 and pass-2 effects and KEEP / DROP / RETEST-GATE / RETEST / UNTESTED).
#
# SWEEP_BRANCH=us|rd picks the branch (one branch per run), SWEEP_SLICE=K/N one part of its flag list (default 1/1).  No build, no engine-source download:
#   us: base = the pancake harness's configs/pancake/flashnext-iq2_xs-plain-upstream.json, arm `updated` = spec 220's updated arm (build
#       /mnt/pancake/benchus/build-us/strata, the profile and table of /mnt/pancake/benchus/src-us);
#   rd: base = /home/dan/auto-runs/benchRD/config-plain.json, arm `updated` = what /home/dan/auto-runs/benchRD/config-updated.json adds to it
#       (build /mnt/pancake/benchRD/build-updated/strata; the plus arms' files from /mnt/pancake/benchRD/src-updated).  That config is from the run
#       before the flag cleanup (TEST_LEDGER.md T-988): the updated arm is built from it MINUS SWEEP_RD_DROP (default STRATA_POOL_FUSE_PHASES, out of
#       the shipped RD file) and WITH SWEEP_RD_SET (default STRATA_VERIFY_ECAP=0 STRATA_VERIFY_ECAP_MIN_ROWS=9, ECAP as shipped, T-990); every change
#       is logged as an ARMFIX line.
# PRIORITY ORDER (owner order 2026-11-20): slice 1 of a branch = all [arm] + [named] switches (us 36: the 35 of the updated arm + AMD_COMMUNITY_TUNING;
#   rd 17: the 11 of the updated arm + GR_RDNA_Q8 / A8 / INJ_UP / FUSED, ADAPT_PERSIST, AMD_COMMUNITY_TUNING, filled up with [code] arms to the even
#   slice size), [code] after them; every slice is triaged, confirmed (pass 2) and merged on its own: usable after slice 1.
# SWEEP_HARNESS_SHA=<commit> pins the synced harness (sweep.py and its registry: the slices of one branch must cut the same list); default origin HEAD.
# PASS 1 (tools/pancake/box_chains/sweep.py spec): only the coded switches NEVER TESTED on gfx1100 (owner clarification 2026-11-20): no pancake
#   TEST_LEDGER.md row naming them on the box and no 2026-11-19 box verdict; already-measured switches keep their verdict and get no arm.  One engine
#   start per arm, abtest --tier screen --drift-check (the base again at the end), cases short, edit, longdec32k; minus-X for an untested switch the
#   updated arm turns on, plus-X for any other at its non-default value.  TRIAGE (sweep.py triage): promising = >= 1.0 % (exact) / >= 1.5 % (lossy) on a
#   metric and none below -3 %; the rest DROP (low performers).
# PASS 2: the promising flags only (at most SWEEP_PASS2_MAX, default 6, the strongest first): abtest --tier confirm, 3 reps; KEEP >= 2 % (exact),
#   RETEST-GATE >= 3 % (lossy: a quality gate is due), else DROP.  The triage file is what tools/pancake/apply_triage.py applies to the shipped files.
# BUDGET: 150 s per engine start (sweep.py estimate); a run must stay <= SWEEP_MAX_S (9000 s, 2.5 h) or it stops before any start and names the
#   slice count it needs.  us: 153 untested switches = 5 runs (SWEEP_SLICE=1/5 .. 5/5: 36 / 30 / 30 / 30 / 27 arms, <= 8850 s each); rd: 109 untested
#   switches, the ones Strata-rdna3's own setup already runs at that value get no arm (98 arms with the box's config-updated.json of 2026-11-19) = 3 runs
#   (1/3 .. 3/3: 33 / 33 / 32 arms, <= 8400 s each).  8 runs, at most about 18 h of box time (pass 2 at its cap in every run).  BENCH_BUDGET_S overrides
#   the computed budget (estimate + 900 s); pass 2 starts only when its estimate fits the budget left.
# THE CANDIDATES (tools/pancake/box_chains/sweep_candidates.json, generated 2026-11-20 from the branch trees; why untested: no pancake TEST_LEDGER.md
#   measurement row naming it on the box and no 2026-11-19 box verdict; the MOD_PLAN.md section 6 items 24 / 25 / 38 / 42 / 44 / 46 / 47 / 48 / 50 / 51 / 53
#   name no switch either branch's engine reads except STRATA_DENSE_MMQ, which is measured (T-615)):
#   us, 153 never tested ([arm] set by the updated arm: minus arm; [named] owner list; [code] other coded on/off switch; * = lossy bars):
#     ROUTE_TAIL_SKIP=0*[arm], PIN_BY_NEED=1[arm], EXPDEC=1[arm], EXCHANGE_ROTATE=1[arm], ATTN_DAILY=1[arm], PROMPT_ATTN_IMMA=1*[arm], KV_PREFETCH=1[arm],
#     PF_GROUP_SCRATCH=1[arm], SELECT_WMMA=1*[arm], SH_STREAM=0[arm], QFUSE=1[arm], VERIFY_QDEDUP=1[arm], STAGER_RING=64[arm],
#     PREFILL_CPU_SHARE_MAX=4096[arm], GDN_HEAD=1[arm], GDN_CHUNKED=1[arm], GDN_CONVL2=1*[arm], GDN_NOY=1[arm], PF_GEMM=1[arm], PF_PAD=1[arm],
#     HC_UPMIX=1*[arm], PF_HCDOWN=1*[arm], FUSE_CONVAB=1[arm], HEAD_MIX_MULTI=1[arm], KV_HOST_DMA=1[arm], QSA_ROWS=1[arm], GR_RDNA_Q8=1*[arm],
#     GR_RDNA_A8=1*[arm], GR_INJ_UP=1*[arm], GR_FUSED=1*[arm], ENGINE_NICE=0[arm], --expert-profile=/mnt/pancake/benchus/src-us/data/expert-profile-
#     pancake.bin[arm], --pool-affinity=measure[arm], --adapt-every=1[arm], --conversation-cache-min-tokens=8192[arm], AMD_COMMUNITY_TUNING=0[named],
#     ADAPT_CALLBACK_WARMUP=0[code], ADAPT_NOWAIT=1[code], ADAPT_OVERLAP_RESIDENT=1[code], ADMIT_MERGE_FIRST=0[code], ATTN_LANECELL=1[code],
#     ATTN_MERGE_V2=1[code], ATTN_V3=0*[code], CKPT_ASYNC=1[code], CKPT_GATHER=1[code], CKPT_POOL=1[code], CKPT_THREADS=0[code], CPU_IQ_ALL_EXACT1=1[code],
#     CVEC_FUSE=1[code], DEQUANT_SIDE=1[code], DOWN_WS=1[code], DRAFT_KEEP=1[code], EARLY_REMOTE_CONTEXTS=1[code], EMBED_VRAM=1[code],
#     EMB_REUSE_ACCOUNT=1[code], EXPERT_V2=1[code], EXPERT_V2K=1[code], EXP_MODE=0*[code], FS_SLOTS=1[code], FUSE_DECODE=1[code], FUSE_HEAD_GR=1*[code],
#     GDN_CHUNKED_MIN=0*[code], GDN_TC=1*[code], GROUPED_V1=1*[code], GR_Q32_G=1[code], GR_Q8_PFS=1[code], HCD_EXACT=1[code], HC_Q8_ALL_STAGES=1*[code],
#     HC_Q8_INJECT=1*[code], HIP_ADAPT_KERNEL_COPY=1[code], HIP_SWIGLU_FUSED=1[code], IQ_STAGE_GRID=0[code], IQ_STAGE_GRID_MMVQ=1[code],
#     IQ_STAGE_TABLES=0[code], KVA=0*[code], KVA_ANY_MODEL=1*[code], KVA_DRAFT_FLOOR=1[code], KVA_MIN_PROMPT=1*[code], KVA_ORACLE_FP32=1*[code],
#     KVA_SUFFIX=1*[code], KV_GROW=0[code], KV_HOST_REGISTER=1[code], LFUSE=1[code], LFUSE_GATE=0[code], LFUSE_PAIR=0[code], LM_HIP=1*[code],
#     MMQ_KQ=1*[code], MMVQ_IL=0[code], MMVQ_IL_120=1*[code], MMVQ_IL_Q3K=1*[code], MMVQ_IL_T1=1*[code], MMVQ_IL_WIDE=1*[code], MMVQ_WAVE=0[code],
#     MTP_CATCHUP_ALL=0[code], MTP_FULL_HEAD=1[code], MTP_TOP2=1[code], NORM_ROPE=0[code], NO_MMVQ_ROWS2=0*[code], NO_MULTI_GR=1[code],
#     NO_NORM_ROPE=0[code], NO_SUB16_GU=1[code], OLD_GROUPED=1*[code], OLD_IQ_MMVQ=1[code], OLD_Q80_SCALED=1*[code], PA_RPF=0[code], PA_TRIM=0[code],
#     PA_V2T=1*[code], PCIE_AUTO_CANON=1[code], PF_BK64=0[code], PF_FUSED_F16=1*[code], PF_FUSED_NATIVE=0*[code], PF_FUSED_Q2N=1*[code],
#     PF_FUSED_Q50=1*[code], PF_Q8D_F16=0*[code], PF_WMMA2=1*[code], PIPED_MTP=0[code], PLE_COMPACT=1[code], PLE_COPY_STREAM=1[code],
#     POOL_UMWAIT_C02=1[code], PREFILL_EQUAL=1*[code], PREFILL_HELP=1*[code], PREFILL_HOST_PLAN=0*[code], PREFILL_LM_EVEN=0*[code], PROMPT_ATTN_Q4=0*[code],
#     PROMPT_ATTN_Q4_TC=1*[code], Q2_BITPLANE=1*[code], Q2_WARP=0*[code], Q6_PACKED=1[code], Q8_GATHER_WIDE=0*[code], Q8_PACKED=1[code],
#     QSA_FUSE_GATE=0*[code], QSA_FUSE_NR=0[code], QSA_SLOTS=1[code], RING_BYTES=0[code], ROPE_TABLE=1*[code], S2_SWIGLU_Q8=0*[code], SELECT_SIMT=1*[code],
#     SH_FORK_LATE=0[code], SIDE_MIXER=1[code], SLOT_CKPT_MOVE=0[code], SMALL_UNR=1[code], SPEC_GUMBEL=1[code], STAGE_TRIM=0[code], TOPK_ACTIVE_ANY=1[code],
#     TOPK_STREAM=0*[code], TQ_STAGE=0*[code], TQ_SUB8=0*[code], TSUM=1[code], VERIFY_ASYNC_LAUNCH=1[code], VERIFY_COHERENT=1[code], VNNI_TIER=1*[code],
#     WMMA_BF16=0*[code], XN_F16=0[code]
#   us, measured already (no arm, the verdict stays):
#     PF_HC_REGION (ledger T-800), HC_Q8 (ledger T-012), PREFILL_CPU_SHARE (ledger T-827), HIP_WMMA (ledger T-072), PREFILL_AUTO_MAX (ledger T-106, T-614),
#     PF_FUSED (ledger T-803), GPU_MAX_HW_QUEUES (ledger T-010), GDN_PP (ledger T-474, T-527, T-642, T-921), PA_FAST (ledger T-768), PF_F16 (diag rerun
#     2026-11-19 (plain build) NO-GAIN), PREFILL_RING (ledger T-920), ADAPT_OVERLAP (ledger T-845), HIPBLASLT_TUNING (diag rerun 2026-11-19 (plain build)),
#     RDNA_MMVQ_WAVE (diag rerun 2026-11-19), GR_RDNA (diag rerun 2026-11-19), RDNA_Q6K (diag rerun 2026-11-19 (UNSUPPORTED, text; owner: invalid, kept)),
#     VERIFY_ECAP (diag rerun 2026-11-19 NO-GAIN short; spec 222 updated vs updated-exact), COMBINE_CPU (diag rerun 2026-11-19), WAIT_FUSE (diag rerun
#     2026-11-19 (with COMBINE_CPU)), ROUTE_PUBLISH (ledger T-009, T-549, T-839), POOL_FUSE_PHASES (ledger T-224), EXPDEC_DOWN (ledger T-037, T-040, T-222),
#     --adapt-swaps (ledger T-248, T-827), --pcie-frac (ledger T-197, T-268, T-605, T-640 (+1)), --lookup-max-t (ledger T-116, T-126, T-127, T-561),
#     --conversation-cache-mib (ledger T-690), DENSE_MMQ (ledger T-615)
#   rd, 109 never tested ([arm] set by the updated arm: minus arm; [named] owner list; [code] other coded on/off switch; * = lossy bars):
#     PIN_BY_NEED=1[arm], EXPDEC=1[arm], EXCHANGE_ROTATE=1[arm], KV_PREFETCH=1[arm], PROMPT_ATTN_IMMA=1*[arm], ATTN_DAILY=1[arm], PF_GROUP_SCRATCH=1[arm],
#     ROUTE_TAIL_SKIP=0*[arm], --grammar=on[arm], --pool-affinity=measure[arm], --conversation-cache-min-tokens=8192[arm], GR_RDNA_Q8=1*[named],
#     GR_RDNA_A8=1*[named], GR_INJ_UP=1*[named], GR_FUSED=1*[named], ADAPT_PERSIST=0[named], AMD_COMMUNITY_TUNING=0[named], ADAPT_CALLBACK_WARMUP=0[code],
#     ADAPT_NOWAIT=1[code], ADAPT_OVERLAP_RESIDENT=1[code], ADMIT_MERGE_FIRST=0[code], ATTN_LANECELL=1[code], ATTN_V3=0*[code], CKPT_POOL=0[code],
#     CKPT_THREADS=0[code], CPU_IQ_ALL_EXACT1=1[code], CVEC_FUSE=1[code], EARLY_REMOTE_CONTEXTS=1[code], EMB_REUSE_ACCOUNT=1[code], EXPERT_V2=1[code],
#     EXPERT_V2K=1[code], EXP_MODE=0*[code], FS_SLOTS=1[code], FUSE_DECODE=1[code], FUSE_HEAD_GR=1*[code], GROUPED_V1=1*[code], HCD_EXACT=1[code],
#     HC_Q8_INJECT=1*[code], HC_REQ8=1*[code], HC_UPMIX=1*[code], HC_WN4=0[code], HIP_ADAPT_KERNEL_COPY=1[code], HIP_SWIGLU_FUSED=1[code],
#     IQ_STAGE_GRID=0[code], IQ_STAGE_GRID_MMVQ=1[code], IQ_STAGE_TABLES=0[code], KV_HOST_REGISTER=1[code], LFUSE=1[code], LFUSE_GATE=0[code],
#     LFUSE_PAIR=0[code], MMQ_KQ=1*[code], MMVF_AUX=1[code], MMVQ_IL=0[code], MMVQ_IL_120=1*[code], MMVQ_IL_Q3K=1*[code], MMVQ_IL_T1=1*[code],
#     MMVQ_IL_WIDE=1*[code], MMVQ_WAVE=0[code], MTP_CATCHUP_ALL=0[code], MTP_FULL_HEAD=1[code], MTP_TOP2=1[code], NORM_ROPE=0[code], NO_MMVQ_ROWS2=0*[code],
#     NO_MULTI_GR=1[code], NO_NORM_ROPE=0[code], NO_SUB16_GU=1[code], OLD_GROUPED=1*[code], OLD_IQ_MMVQ=1[code], PA_V2T=1*[code], PCIE_AUTO_CANON=1[code],
#     PCIE_OFF=0[code], PF_BK64=0[code], PF_FUSED_NATIVE=0*[code], PF_GEMM=1*[code], PF_HCDOWN=1*[code], PF_PAD=1[code], PF_WMMA2=1*[code],
#     POOL_UMWAIT_C02=1[code], PREFILL_EQUAL=1*[code], PREFILL_HELP=1*[code], PREFILL_HOST_PLAN=0*[code], PROMPT_ATTN_Q4=0*[code],
#     PROMPT_ATTN_Q4_TC=1*[code], Q2_BITPLANE=1*[code], Q2_WARP=0*[code], Q6_PACKED=1[code], Q8_GATHER_WIDE=0*[code], Q8_PACKED=1[code], QSA_ROWS=1[code],
#     QSA_SLOTS=1[code], RESIDENT_PIN=0[code], RING_BYTES=0[code], ROPE_TABLE=1*[code], S2_SWIGLU_Q8=0*[code], SELECT_SIMT=1*[code], SELECT_WMMA=1*[code],
#     SH_FORK_LATE=0[code], SH_STREAM=0[code], SLOT_CKPT_MOVE=0[code], SPEC_GUMBEL=1[code], TOPK_ACTIVE_ANY=1[code], TOPK_STREAM=0*[code],
#     TQ_STAGE=0*[code], TQ_SUB8=0*[code], TSUM=1[code], VERIFY_ASYNC_LAUNCH=1[code], VERIFY_COHERENT=1[code], WMMA_BF16=0*[code], XN_F16=0[code]
#   rd, measured already (no arm, the verdict stays):
#     VERIFY_ECAP (benchRD-diag-updated; spec 252 screen), EXPDEC_DOWN (ledger T-037, T-040, T-222), PF_HC_REGION (ledger T-800), PREFILL_EQUAL_TAIL (ledger
#     T-679, T-810, T-822), HC_Q8 (ledger T-012), --lookup-max-t (ledger T-116, T-126, T-127, T-561), GR_RDNA (ledger T-649, T-650, T-678), RDNA_MMVQ_WAVE
#     (benchRD-diag-updated GAIN), COMBINE_CPU (ledger T-009, T-313), WAIT_FUSE (ledger T-009), ROUTE_PUBLISH (ledger T-009, T-549, T-839), POOL_FUSE_PHASES
#     (benchRD-diag-updated 2026-11-19 -5.97%), PF_F16 (benchRD-diag-updated GAIN; spec 250 updated+pf16-1), RDNA_KQ_MAXCOLS (ledger T-080, T-699, T-767),
#     DENSE_MMQ (ledger T-615)
# Rules (as bench-exact.sh): box jobs never overlap (busy() sees every other job and the other benchmark chains; a FROZEN AA shell is not a job); ~30 s
# quick idle before each step; pancake.service stays DOWN (never started; stopped only after 3 minutes without a client); AA stays frozen (no signal is
# ever sent to it); nothing is deleted; every abtest run is in its OWN SESSION under a watchdog (run_group: TERM at the limit, KILL 100 s later, REAP of
# every process it had): no orphan engine survives a step.  The harness runs with the box's venv python (ABTEST_PYTHON).
# Start: /home/dan/clone-of-d/src/run-queue-sweep.sh runs the 8 slices (rd 1/3, us 1/5, rd 2/3, rd 3/3, us 2/5 .. 5/5) one after the other; by hand:
#   cd /home/dan/clone-of-d/src && SWEEP_BRANCH=rd SWEEP_SLICE=1/3 BENCH_ONLY=EXTRA BENCH_EXTRA=./bench-sweep.sh BENCH_NO_RESUME=1 setsid nohup sh ./bench-sched.sh > /dev/null 2>&1 &
#   (then SWEEP_SLICE=2/3, 3/3; then SWEEP_BRANCH=us SWEEP_SLICE=1/5 .. 5/5; each run merges its slice into <branch>-triage.json)
S=/home/dan/clone-of-d/src; R=$S/Strata-win; L=$S/bench-sweep.log; O=/home/dan/auto-runs/benchsweep; T=$O/work; PM=/mnt/pancake/benchsweep; HN=$PM/harness
USB=/mnt/pancake/benchus; RDB=/mnt/pancake/benchRD; RDC=/home/dan/auto-runs/benchRD/config-plain.json; RDU=/home/dan/auto-runs/benchRD/config-updated.json
DATA=/home/dan/clone-of-d/Strata-data; ROCM=/opt/rocm/core-10.0; REPO_URL=https://github.com/rocktboy1982/pancake-strata.git
T0=$(date +%s); BUDGET=${BENCH_BUDGET_S:-9900}; POLL=5; MAXS=${SWEEP_MAX_S:-9000}; P2MAX=${SWEEP_PASS2_MAX:-6}
B=${SWEEP_BRANCH:-}; SL=${SWEEP_SLICE:-1/1}; SLN=$(echo "$SL" | tr / o | sed 's/o/of/')
export PATH=$ROCM/bin:$R/.venv/bin:$PATH LD_LIBRARY_PATH=$ROCM/lib HIP_VISIBLE_DEVICES=0
HPY=${BENCH_HARNESS_PY:-$R/.venv/bin/python}
export HPY ABTEST_PYTHON=$HPY
mkdir -p $O $T $PM
echo "START $(date +%F_%T) branch ${B:-?} slice $SL" >> $L
say() { echo "$*" >> $L; }
busy() { pgrep -f "[c]make --build|[n]inja|[c]lang|[c]c1plus|[c]test|[a]btest.py|[t]ask_gate.py|[r]eplay_turns.py|[q]uality_gate.py|[n]ight_queue.py|[s]erve/server.py|[s]trata --serve|[c]onv_trace.py|[a]uto-chain|[a]uto-post|[i]q_avx2_parity|[t]tft_bench.py|[w]in-chain[A-Za-z0-9]*\.sh|[x-z]tail-chain[A-Z]\.sh|[a]fter7|[b]ench-(updated(AA|RD)|diagUS|exact|rdgate)[.a-z0-9]*\.sh" >/dev/null 2>&1; }
aa_live() { for p in $(pgrep -f "[r]edo-chainAA\.sh" 2>/dev/null); do case "$(ps -o stat= -p $p 2>/dev/null)" in T*) ;; *) return 0;; esac; done; return 1; }
bz() { busy || aa_live; }
QI=${BENCH_PREIDLED:-0}
wait_idle() { if [ "$QI" = 1 ]; then q=0; while [ $q -lt 2 ]; do if bz; then q=0; sleep 15; else q=$((q+1)); sleep 15; fi; done; else q=0; while [ $q -lt 3 ]; do if bz; then q=0; sleep 60; else q=$((q+1)); sleep 60; fi; done; QI=1; fi; }
elapsed() { echo $(( $(date +%s) - T0 )); }
left() { echo $(( BUDGET - $(elapsed) )); }
left_ok() { [ "$(left)" -ge "$1" ]; }
clients() { ss -Htn state established '( sport = :11437 )' 2>/dev/null | wc -l; }
freekb() { df -Pk "$1" | awk 'NR==2{print $4}'; }
finish() {
  say "service stays down (owner order 2026-11-19: only the benchmarks run; this chain never starts pancake.service) $(date +%T)"
  say "/ free $(freekb /) KB, /mnt/pancake free $(freekb /mnt/pancake) KB, box time $(elapsed) s"
  echo WINBENCHSWEEP_DONE >> $L
}
abort() { say "ERROR $*"; say "WINBENCHSWEEP_ABORTED: $*"; finish; exit 1; }
errline() {
  pat='error|fatal|abort|died|failed|needs|cannot|exception|terminate|no such|not found|unrecognized|unknown'
  a=$(grep -i -m1 -E "$pat" "$1" 2>/dev/null | cut -c1-240); [ -n "$a" ] || a=$(grep -m1 . "$1" 2>/dev/null | cut -c1-240)
  e=; for f in "$2"/*.engine.log "$2"/*.out; do [ -f "$f" ] || continue; e=$(grep -i -E "$pat" "$f" 2>/dev/null | tail -n 1 | cut -c1-240); [ -n "$e" ] && break; done
  echo "first error line of the stdout: [$a]; engine log: [$e]"
}

# ------------------------------------------------------------------ process groups: a watchdog per step, no orphan afterwards (the same functions as bench-exact.sh)
# ptree SID: SID, every descendant (by parent pid) and every process of session SID.  stamps PIDS: "pid start-time" lines of the live ones (pid reuse safe).
ptree() { ps -eo pid=,ppid=,sid= 2>/dev/null | awk -v r="$1" '{ c[$2] = c[$2] " " $1; if ($3 == r) s[$1] = 1 } END { q = r; while (q != "") { n = split(q, a, " "); q = ""; for (i = 1; i <= n; i++) if (!(a[i] in seen)) { seen[a[i]] = 1; q = q c[a[i]] } } for (p in s) seen[p] = 1; for (p in seen) print p }'; }
stamps() { [ -n "$1" ] && ps -o pid=,lstart= -p "$(echo $1 | tr ' ' ',')" 2>/dev/null | sed 's/^ *//; s/  */ /g'; }
alive() { kill -0 "$1" 2>/dev/null || return 1; case "$(ps -o stat= -p "$1" 2>/dev/null)" in Z*) return 1;; esac; return 0; }
sendsig() { ${REAPCMD:-kill} "$@" 2>/dev/null; }
CUR_SID=; CUR_SEEN=
run_group() {  # run_group NAME LIMIT_S LOG CMD...: CMD in its own session; at min(LIMIT, budget left) TERM to its tree, KILL 100 s later; then REAP
  nm=$1; lim=$2; log=$3; shift 3
  lb=$(left); [ "$lb" -lt 60 ] && lb=60; [ "$lb" -lt "$lim" ] && lim=$lb
  seen=$T/$nm.pids; : > "$seen"
  setsid "$@" > "$log" 2>&1 &   # a background job of a non-interactive sh is no group leader: setsid does not fork, $! is the session id
  sid=$!; CUR_SID=$sid; CUR_SEEN=$seen; t=0; killed=0
  while alive $sid; do
    stamps "$(ptree $sid)" >> "$seen"
    if [ $killed -eq 0 ] && [ $t -ge $lim ]; then
      tp=$(ptree $sid); say "TIMEOUT $nm after $t s (limit $lim s, budget left $(left) s): TERM to its process tree: $(echo $tp)"
      for p in $tp; do sendsig -TERM $p; done; killed=1; tk=$t
    elif [ $killed -eq 1 ] && [ $t -ge $((tk + 100)) ]; then
      tp=$(ptree $sid); say "TIMEOUT $nm: still running 100 s after TERM: KILL: $(echo $tp)"
      for p in $tp; do sendsig -KILL $p; done; killed=2
    elif [ $killed -eq 2 ] && [ $t -ge $((tk + 130)) ]; then say "ERROR $nm: the session leader $sid survives SIGKILL"; break; fi
    sleep $POLL; t=$((t + POLL))
  done
  wait $sid 2>/dev/null; rc=$?; [ $killed -gt 0 ] && rc=124
  reap $nm "$seen"; CUR_SID=; CUR_SEEN=
  return $rc
}
reap() {  # reap NAME SEEN_FILE: every process the step had that still runs (same pid AND start time) gets TERM, then KILL after 30 s; each is logged
  pids=$(awk '{print $1}' "$2" 2>/dev/null | sort -u | tr '\n' ' ')
  [ -n "$pids" ] || return 0
  left_over=$(stamps "$pids" | while read -r p st; do grep -qxF "$p $st" "$2" && echo "$p"; done | tr '\n' ' ')
  [ -n "$(echo $left_over)" ] || return 0
  say "REAP $1: still running after the step ended, TERM: $(for p in $left_over; do printf '%s [%s] ' $p "$(ps -o args= -p $p 2>/dev/null | cut -c1-100)"; done)"
  for p in $left_over; do sendsig -TERM $p; done
  still=$left_over; n=0; while [ $n -lt 6 ]; do sleep 5; n=$((n + 1)); still=$(stamps "$left_over" | while read -r p st; do grep -qxF "$p $st" "$2" && echo "$p"; done | tr '\n' ' '); [ -n "$(echo $still)" ] || break; done
  if [ -n "$(echo $still)" ]; then say "REAP $1: KILL $still"; for p in $still; do sendsig -KILL $p; done; sleep 3
    gone=$(stamps "$still" | while read -r p st; do grep -qxF "$p $st" "$2" && echo "$p"; done | tr '\n' ' ')
    [ -z "$(echo $gone)" ] || say "ERROR REAP $1: processes survive SIGKILL: $gone"
  fi
  say "REAP $1 done"
}
on_signal() {
  say "signal received $(date +%T): the running step's tree gets TERM, then the chain ends"
  if [ -n "$CUR_SID" ]; then for p in $(ptree $CUR_SID); do sendsig -TERM $p; done; sleep 5; reap signal "$CUR_SEEN"; fi
  say "WINBENCHSWEEP_ABORTED: signal"; finish; exit 1
}
trap on_signal HUP INT TERM

# ------------------------------------------------------------------ steps
ab() {  # ab NAME LIMIT SPEC flags...: one abtest run from the synced harness, in its own session
  nm=$1; lim=$2; sp=$3; shift 3
  [ "$(freekb /)" -ge 307200 ] || { say "ERROR $nm: / has < 300 MB free: not run"; return 1; }
  say "START $nm $(date +%F_%T) (limit $lim s, budget left $(left) s)"
  run_group $nm $lim $O/$nm.stdout sh -c 'cd "$1" && shift && exec "$HPY" tools/pancake/abtest.py "$@"' ab $HN $sp "$@" --out $O/$nm.jsonl; rc=$?
  say "END $nm exit=$rc $(date +%T)"
  [ $rc -eq 0 ] || say "ERROR $nm exit=$rc: $(errline $O/$nm.stdout $O/$nm) (the triage uses the rows that were measured)"
  grep -E "ORDER EFFECT|DRIFT" $O/$nm.stdout | cut -c1-220 >> $L
  return $rc
}
sw() { "$HPY" $HN/tools/pancake/box_chains/sweep.py "$@"; }

# ------------------------------------------------------------------ 0. branch, (standalone: wait for its benchmark), service down, idle, prerequisites
case "$B" in us|rd) ;; *) abort "SWEEP_BRANCH must be us or rd (got '${B}')";; esac
case "$SL" in [1-9]/[1-9]) ;; *) abort "SWEEP_SLICE must be K/N (got '$SL')";; esac
if [ "${BENCH_SCHED:-0}" != 1 ]; then
  if [ $B = us ]; then mk=WINBENCHUS_DONE; ml=$S/bench-updatedAA.log; else mk=WINBENCHRD_DONE; ml=$S/bench-updatedRD.log; fi
  until grep -q $mk $ml 2>/dev/null; do sleep 300; done
  say "the $B benchmark chain is done $(date +%T)"
else say "started by the scheduler: the AA shell is frozen, the box was idle $(date +%T)"; fi
if systemctl --user is-active --quiet pancake.service; then
  q=0; while [ $q -lt 3 ]; do if [ "$(clients)" -gt 0 ]; then q=0; say "WAIT service has clients $(date +%T)"; else q=$((q+1)); fi; sleep 60; done
  systemctl --user stop pancake.service; say "service stopped (idle) $(date +%T)"
else say "service already stopped"; fi
wait_idle
[ "$(freekb /mnt/pancake)" -ge 2097152 ] || abort "START: /mnt/pancake has $(freekb /mnt/pancake) KB free (< 2 GB); nothing was deleted"
[ "$(freekb /)" -ge 307200 ] || abort "START: / has $(freekb /) KB free (< 300 MB)"
if [ $B = us ]; then need="$USB/build-us/strata $USB/src-up/terminal/data/expert-profile.bin $USB/src-us/data/expert-profile-pancake.bin"; SRC=$USB/src-us
else need="$RDB/build-updated/strata $RDC $RDU $RDB/src-updated/data/expert-profile-pancake.bin"; SRC=$RDB/src-updated; fi
miss=; for f in $DATA/packs/iq2_xs/tokenizer $DATA/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
               $DATA/models/IQ2_XS/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf $DATA/mtp/rt $HPY $need; do [ -e "$f" ] || miss="$miss $f"; done
[ -z "$miss" ] || abort "PREREQ missing:$miss (bench-updatedAA.sh / bench-updatedRD.sh build the engines and write the configs; nothing is built here)"
"$HPY" -c 'import sys' >/dev/null 2>&1 || abort "PREREQ the harness python $HPY does not run"
say "PREREQ ok ($B): $need"

# ------------------------------------------------------------------ 1. harness: origin port/rdna3-decode (abtest, sweep.py, its registry, spec 220), mtimes reset
SHA=${SWEEP_HARNESS_SHA:-$(timeout 60 git ls-remote $REPO_URL refs/heads/port/rdna3-decode 2>/dev/null | cut -f1)}
HARN=$PM/harness.tgz; rm -f $HARN.part
if [ -n "$SHA" ] && timeout 1200 curl -sfL -o $HARN.part https://codeload.github.com/rocktboy1982/pancake-strata/tar.gz/$SHA && tar tzf $HARN.part >/dev/null 2>&1; then
  mv $HARN.part $HARN; say "SYNC harness: origin port/rdna3-decode $SHA${SWEEP_HARNESS_SHA:+ (pinned by SWEEP_HARNESS_SHA)} -> $HN"
else abort "SYNC harness download failed"; fi
mkdir -p $HN && tar xzmf $HARN --no-same-owner --strip-components=1 -C $HN || abort "SYNC harness tar failed"
for need in tools/pancake/abtest.py tools/pancake/box_chains/sweep.py tools/pancake/box_chains/sweep_candidates.json \
            bench/specs/220-updated-vs-upstream-short.json configs/pancake/flashnext-iq2_xs-plain-upstream.json; do
  [ -f "$HN/$need" ] || abort "SYNC: $need is missing in the synced harness"
done
grep -q ABTEST_PYTHON $HN/tools/pancake/abtest.py || abort "PREREQ harness: $HN/tools/pancake/abtest.py does not read ABTEST_PYTHON"
e=$(cd $HN && PYTHONDONTWRITEBYTECODE=1 "$HPY" -B -c 'import serve.server' 2>&1) || abort "PREREQ harness: $HPY cannot import serve.server from $HN: $(echo "$e" | tail -n 1 | cut -c1-240)"
say "PREREQ harness ok: $HPY imports serve.server from $HN $(date +%T)"

# ------------------------------------------------------------------ 2. the pass-1 spec and the estimate (stop before any start when the run does not fit 2.5 h)
P1=$T/$B-p1-$SLN.json
if [ $B = us ]; then sw spec --branch us --plain $HN/configs/pancake/flashnext-iq2_xs-plain-upstream.json --arm-from $HN/bench/specs/220-updated-vs-upstream-short.json --arm-name updated \
                       --src $SRC --slice $SL --pass2-max $P2MAX --out $P1 > $T/$B-spec-$SLN.txt 2>&1 || abort "SPEC: sweep.py spec failed: $(tail -n 1 $T/$B-spec-$SLN.txt)"
else fx=; for k in ${SWEEP_RD_DROP-STRATA_POOL_FUSE_PHASES}; do fx="$fx --drop-env $k"; done
  for kv in ${SWEEP_RD_SET-STRATA_VERIFY_ECAP=0 STRATA_VERIFY_ECAP_MIN_ROWS=9}; do fx="$fx --set-env $kv"; done
  sw spec --branch rd --plain $RDC --updated-config $RDU $fx --src $SRC --slice $SL --pass2-max $P2MAX --out $P1 > $T/$B-spec-$SLN.txt 2>&1 || abort "SPEC: sweep.py spec failed: $(tail -n 1 $T/$B-spec-$SLN.txt)"; fi
grep -E '^(ARMFIX|SWEEP)' $T/$B-spec-$SLN.txt >> $L
set -- $(grep '^ESTIMATE' $T/$B-spec-$SLN.txt); E1=${2:-0}; E2=${3:-0}
[ "$E1" -gt 0 ] || abort "SPEC: no estimate from sweep.py"
if [ $((E1 + E2)) -gt $MAXS ]; then
  n=${SL#*/}; need=$(( (E1 + E2) * n / MAXS + 1 ))
  abort "BUDGET: this run's estimate $((E1 + E2)) s is over $MAXS s (2.5 h): split it (SWEEP_SLICE=1/$((need * 1)) ... at least $need parts of this slice); nothing was started"
fi
[ -n "${BENCH_BUDGET_S:-}" ] || BUDGET=$((E1 + E2 + 900))
say "BUDGET $BUDGET s (estimate pass 1 $E1 s + pass 2 <= $E2 s + 900 s overhead)"

# ------------------------------------------------------------------ 3. pass 1, triage, pass 2 for the promising flags, the final triage
wait_idle
ab $B-p1-$SLN $((E1 + 600)) $P1 --tier screen --drift-check --retries 0; rc1=$?
sw triage --meta $P1.meta.json --pass1 $O/$B-p1-$SLN.jsonl --out $T/$B-tri1-$SLN.json > $T/$B-tri1-$SLN.txt 2>&1 || say "ERROR triage of pass 1 failed: $(tail -n 1 $T/$B-tri1-$SLN.txt)"
rc2=-
if sw pass2 --spec $P1 --triage $T/$B-tri1-$SLN.json --max $P2MAX --out $T/$B-p2-$SLN.json > $T/$B-p2-$SLN.txt 2>&1; then
  cat $T/$B-p2-$SLN.txt >> $L
  if left_ok "$E2"; then
    wait_idle; ab $B-p2-$SLN $((E2 + 600)) $T/$B-p2-$SLN.json --tier confirm --reps 3 --no-early-stop --warmup 1 --outlier robust --retries 0; rc2=$?
  else say "SKIP pass 2: budget: $(left) s left (< $E2 s); the promising flags stay RETEST"; fi
else cat $T/$B-p2-$SLN.txt >> $L; fi
if [ -f $O/$B-p2-$SLN.jsonl ]; then sw triage --meta $P1.meta.json --pass1 $O/$B-p1-$SLN.jsonl --pass2 $O/$B-p2-$SLN.jsonl --out $O/$B-triage-$SLN.json > $T/$B-tri-$SLN.txt 2>&1
else sw triage --meta $P1.meta.json --pass1 $O/$B-p1-$SLN.jsonl --out $O/$B-triage-$SLN.json > $T/$B-tri-$SLN.txt 2>&1; fi || say "ERROR final triage failed: $(tail -n 1 $T/$B-tri-$SLN.txt)"
sed 's/^/TRIAGE /' $T/$B-tri-$SLN.txt | cut -c1-260 >> $L
sw merge --out $O/$B-triage.json $O/$B-triage-*of*.json 2>> $L && say "TRIAGE file $O/$B-triage.json (slices: $(ls $O/$B-triage-*of*.json 2>/dev/null | wc -l))"
say "SUMMARY branch $B slice $SL: pass 1 exit=$rc1, pass 2 exit=$rc2 (- = not run); box time $(elapsed) s of $BUDGET s; results in $O"
finish
