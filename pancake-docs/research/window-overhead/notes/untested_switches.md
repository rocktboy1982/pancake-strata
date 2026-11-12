# Untested window-overhead switches in Pancake (Strata engine)

Scope: opt-in and gated switches in `src/core/conversation_disk_env.def` whose names carry a window-overhead keyword (MTP, SPEC, WAIT, COPY, ROUTE, DEFER, RING, GRAPH, FUSE). Read-only. Nothing built or run, no GPU.

Sources (worktree copy, newest env table): [conversation_disk_env.def](/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a/src/core/conversation_disk_env.def), [MOD_PLAN.md](../../../../MOD_PLAN.md), [bench/results](../../../../bench/results), [night queue RESULTS.md](/home/dan/night-queue-run/RESULTS.md), [abtest-2026-11-12](/home/dan/abtest-2026-11-12).

Labels: **[measured]** = a result file in the repo or night queue shows the switch varied; **[coded only]** = code and docs only, no A/B result found; **[unverified]** = a claim or mention I could not check against a result file.

Caveat on grep counts: most `bench/results` hits for a switch are environment dumps (run scripts, `base.json`, preset env listings) where the switch is simply set to 1. Those hits are not measurements. A switch is only counted as measured when an arm was varied.

## Key Question 1: which switches target per-window waits, copies, readbacks, or routing

### Takeaway
153 `STRATA_ENV` entries match the keyword filter (Speed 84, KeyIfSet 22, Key 12, Unsure 17, Inert 17, InConfig 1). About 20 of them have a dedicated A/B or screen result. Most of the rest appear only in preset env dumps or nowhere except the `.def` table.

### Cited Findings
- The `.def` table lists 1017 `STRATA_ENV` entries overall; the 153 keyword matches are the window-overhead candidates. [conversation_disk_env.def](/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a/src/core/conversation_disk_env.def)
- Many `bench/results` hits are env dumps, e.g. `"STRATA_WAIT_FUSE": "1"` and `"STRATA_QFUSE": "1"` in [gpuprof2.sh](../../../../bench/results/2026-11-09-7900xt-gpuprof/gpuprof2.sh) and [test-phase base.json](../../../../bench/results/2026-11-09-test-phase/01-deferral/base.json). These show the switch was on in a preset, not that it was measured.

### Inventory of window-overhead candidates (per-window waits, copies, readbacks, routes, deferral, speculation)

| Switch | Class | Status | Evidence | Label |
|---|---|---|---|---|
| STRATA_VERIFY_DEFER_CPU | Key | In fast-hip preset; deferral of CPU rows across the window boundary | Test phase A3: edit +15.8% +-3.9, short n.s.; night queue defer-iq2xs arms ([RESULTS.md](/home/dan/night-queue-run/RESULTS.md)) | [measured] |
| STRATA_VERIFY_DEFER_AUTO (+ _MIN_T) | KeyIfSet | Gate closes deferral on short windows; MOD_PLAN 17.1 "coded, NOT built, NOT measured" (branch dev/p17/defergate) | Night queue defer-iq2xs-03C: auto-closed arm -1.2% to -12.5% vs always-on deferral ([RESULTS.md](/home/dan/night-queue-run/RESULTS.md) lines 163-175). MOD_PLAN line 2623 is stale; the night queue run exists. | [measured] (MOD_PLAN out of date) |
| STRATA_VERIFY_DEFER_CPU_BELOW | Key | Threshold for the rank guard, default 0.1 | Within the defer arms (night queue, abtest 95 files). Rank-guard A4 fold/top2 run in [10-rank-guard-fold.jsonl](../../../../bench/results/2026-11-11-gpu/10-rank-guard-fold.jsonl) | [measured] (only inside defer arms, no isolated sweep) |
| STRATA_VERIFY_DEFER_CPU_TOP | Key | Rank-guard top-k; coded in defer_rank.hpp | Used in top2 arm of 10-rank-guard-fold. MOD_PLAN line 1590: top2 edit -2.6%, short +0.1%. My recomputed row means (n=6 each): top2 edit 158.1 vs base 165.0 (-4.2%), short 114.4 vs 124.4 | [measured] (lossy, negative on edit) |
| STRATA_VERIFY_DEFER_CPU_FOLD | Speed | One pool pass when few rows deferred | 10-rank-guard-fold arms top2-fold1 and top2-fold2. Fold2 edit 164.9 vs base 165.0; short 125.7 vs 124.4. MOD_PLAN line 1753 still says "not built" (stale). | [measured] (flat on edit, n=6) |
| STRATA_ROUTE_PRIOR (turbo, lambda 0.5) | Key | Lossy residency-biased routing | Test phase 06 screen +32.4%; 06c confirm +29.7% +-53.6, INCONCLUSIVE; 2.65% token diff vs 1.8% stock ([test-phase RESULTS.md](../../../../bench/results/2026-11-09-test-phase/RESULTS.md) rows 49, 54) | [measured] (inconclusive, owner decision) |
| STRATA_ROUTE_PRIOR_AUTO (+ _MIN_T) | KeyIfSet | Prior only in lookup windows; MOD_PLAN 17.1 "coded, NOT built, NOT measured" | None found | [coded only] |
| STRATA_ROUTE_PRIOR_DEPTH | Key | Depth-graded bonus (A2); "opt-in; quality gate pending" (MOD_PLAN line 194) | None found | [coded only] |
| STRATA_ROUTE_PRIOR_PROTECT / _KEEP / _KEEP_P / _SCOPE / _MARKS / _ON_PROMPT | KeyIfSet | Turbo guards (F13); planned, not run | None found | [coded only] |
| STRATA_ROUTE_RESIDENT, _MTP, _RANKS | Key | Residency-biased routing after router | Not mentioned in MOD_PLAN or any result | [coded only] |
| STRATA_ROUTE_TAIL_SKIP | Key | Skips missed tail experts; set 0 in preset env | Preset env dumps only | [unverified] |
| STRATA_ROUTE_PUBLISH | Speed | Router publish path; was the non-finite-logit suspect; now forced off under deferral (MOD_PLAN line 166, commit d5b58f669) | Correctness bisect per MOD_PLAN; 10-rank-guard-fold base env. No perf A/B file found. | [unverified] (correctness result cited in MOD_PLAN only) |
| STRATA_ROUTE_FAST | Speed | One-launch router publish (F9) | 25-kern-hip first pass only: route-fast arm n=4, short decode mean 126.1 vs base 116.7 in same pass (base had a 95.9 slow run). Not in confirm pass. MOD_PLAN lines 1760 and 2089 still say "not built" (stale). | [measured] (inconclusive, single noisy pass) |
| STRATA_ROUTER_FUSE | Key | Router GEMV + top-10 + doorbell in one launch (A33); only active without ROUTE_PUBLISH-deferral combos; never acts in fast-hip (MOD_PLAN line 1599) | Test phase 04 screen: +0.4..+4.3% n.s. with base hit by reclaim. Row says "retest the best three"; no retest found. | [measured] (screen only, n.s., probably dead in preset) |
| STRATA_EXPDEC_SWIGLU | Key | Expdec gate/up + SwiGLU + q8_1 fused (item 48) | 25-kern-hip-confirm, 10 reps: 124.0 vs base 123.9 short decode, flat | [measured] (flat) |
| STRATA_EXPDEC_DOWN | Key | Q2_0 down fold, in fast-hip preset since f15ad2b51 | Test phase 04c: +1.7% +-0.5 short, edit +5.4% +-7.3 n.s., token forcing PASS (test-phase RESULTS.md row 29; MOD_PLAN line 2250) | [measured] |
| STRATA_WAIT_FUSE | Speed | Single host-flag poller in combine_cpu (item 46); estimated 0-3% | In preset env dumps. The planned A/B "WAIT_FUSE 1 vs 0, judged on host CPU time" ([Hardware data path technologies.md](../../../../reports/Hardware data path technologies.md), item 46 and item 50) not found. Listed as a suspect for non-finite logits in [Grouped technologies](../../../../reports/Grouped technologies for pancake.md) line 234. | [coded only] |
| STRATA_FUSE_WAIT | Speed | Gate for WAIT_COPY256 on gfx1100 (MOD_PLAN line 1759) | Preset env dumps only | [coded only] |
| STRATA_WAIT_COPY256 | Speed | 256-thread, 16-byte copy plan (F21) | 25-kern-hip copy256 arm, n=2, short decode 124.6 vs base 116.7 in noisy first pass; no confirm arm. MOD_PLAN lines 1759 and 2408 say "not built" (stale). | [measured] (inconclusive, one noisy pass) |
| STRATA_MTP_EAGER | Speed | Eager draft graphs, exact by construction | Test phase 07b screen -0.6% / +15.4%; 06c confirm +1.6% +-5.2 n.s., PASS, kept opt-in (test-phase RESULTS.md row 55) | [measured] (n.s.) |
| STRATA_MTP_HIP_PRIO | Speed | Draft stream on its own HW queue | Test phase 07b/07c/07d: dropped for good (RESULTS.md rows 53, 56, 58) | [measured] (dropped) |
| STRATA_POOL_FUSE_PHASES | Speed | One phase per layer in CPU pool (item 29) | Test phase 03b and night queue (abtest 24 files, nq 40). MOD_PLAN line 1484 +1-4%, measured edit +2.5% +-1.7 in preset row | [measured] |
| STRATA_POOL_FUSE_MIN, _FUSE_PHASES parent | Speed | Pool fusion knobs | bench 12 hits (mostly env dumps) | [unverified] |
| STRATA_COPY_HIP_PRIO | Speed | Coalesced promotion copy priority; MOD_PLAN 1347 "parked: one-line A/B" | None | [coded only] |
| STRATA_GRAPH_PRIO | Speed | Env-only arm for TEST_PLAN 4.1.7 (MOD_PLAN line 2287) | None | [coded only] |
| STRATA_FUSE_HITS | Speed | Tier-5 item (MOD_PLAN line 2283) | None | [coded only] |
| STRATA_GR_FUSED | Speed | HC read fused; in preset env dumps (bench 141) | No isolated A/B; listed as a non-finite suspect (Grouped technologies line 234) | [unverified] |
| STRATA_FUSE_CONVAB | Unsure | GDN conv + a/b projections in one launch | Preset env dumps only | [coded only] |
| STRATA_PREFILL_RING | Speed | Prompt-path ring (192 slots); not a per-window switch | Community RX 7900 XTX and RTX 3080 READMEs (not gfx1100 A/B: [2026-10-08 README](../../../../bench/results/2026-10-08-community-2x-rtx3080-20gb-ud-q4-k-xl-512k-pool/README.md), 384 +~15%) | [measured] (other hardware, not gfx1100) |
| STRATA_STAGER_RING | Speed | Stager ring size (prompt path) | Community RTX 5090 README tuning; preset env dumps on gfx1100 | [measured] (other hardware) |
| STRATA_PREFILL_ROUTED_STREAM (+ _DEPTH) | Speed | Stream only routed experts in prefill (item 1.3 in forks batch) | MOD_PLAN 620 "A/B each on this PC" planned, not run | [coded only] |
| STRATA_SPEC_COST | Speed | Draft cost model; default "1" in NVIDIA table (src/core/nv_defaults.cpp:80) | Night queue prose-var-03D (RESULTS.md lines 445-449). MOD_PLAN 1168: round A worse or flat (112.0 / 98.3 vs 111.7 base) | [measured] (flat or worse on gfx1100) |
| STRATA_SPEC_COST_TABLE | Speed | Cost table from traces (S3) | MOD_PLAN 220: coded, no result | [coded only] |
| STRATA_SPEC_PROB, _COUPLED, _PROB_GATE, _PROB_DT, _MIN_TEMP | Speed | Exact speculative sampling for sampled requests (F6, A21) | MOD_PLAN lines 793, 2086: measured upstream on a 3060 only; A21 "add with PROB/COUPLED", not run here | [coded only] |
| STRATA_SPEC_CUM | Speed | Cumulative-probability draft stop (F8); MOD_PLAN 1770 "not ported", on hold | MOD_PLAN lines 1770, 2088 | [coded only] |
| STRATA_SPEC_ADAPT_EVERY, _SM86, _WARM, STRATA_SPEC_GUMBEL, STRATA_SPEC_DEPTH, STRATA_SPEC_T_CYCLE, STRATA_SPEC_WATCH | Speed | Draft adaptation and sampling knobs | Single-digit env dump hits only | [unverified] |
| STRATA_MTP_BATCH, _BATCH_F16, _BATCH_RING, _CATCHUP_ALL, _FULL_HEAD, _KEEP, _MULTI_COPY, _PREFILL_SYNC, _PRIORITY, _PROXY, _Q8_SM60, _ROOM_MIN, _SHARED_BRANCH, _SIDE_STREAM, _TOP2, _STAGE0, PIPED_MTP | Speed | Batch/draft-stream variants | MTP_STAGE0 3 bench hits; PIPED_MTP guard in MOD_PLAN 2281; rest zero hits | [coded only] (STAGE0 and PIPED_MTP [unverified]) |
| STRATA_KV_RING_ADAPT | InConfig | Adaptive resident K/V ring (17.4) | Night queue kvring-03R and kvring-04P (RESULTS.md rows 42-43, 397-400) | [measured] |
| STRATA_KV_RING_OFF | Speed | KV ring off | Zero hits | [coded only] |
| STRATA_PF_PEER_RING, STRATA_SPLIT_RING, STRATA_SPLIT_WAIT_MS, STRATA_SPLIT_MTP_BATCH, STRATA_GFX12_RING | Speed | Multi-GPU / batch / gfx12 planning | .def note: disk tier needs one GPU; not applicable to a single gfx1100 | [coded only] (out of scope) |
| STRATA_ROUTE_PER_TOKEN, STRATA_ROUTE_PROJ_PER_TOKEN, STRATA_ROUTER_BLOCK | Unsure | Router granularity variants | Zero hits | [coded only] |
| STRATA_DF2_WAITCOMMIT, STRATA_BATCH_DF2_DEFER | Speed | Drafting/commit policy | Zero hits | [coded only] |
| STRATA_PLE_COPY_STREAM, STRATA_SLOT_COPY_BATCHSYNC, STRATA_SLOT_COPY_DIRECT, STRATA_GROUP_COPY, STRATA_HIP_ADAPT_KERNEL_COPY | Speed | Copy-stream and staging variants | Zero hits except HIP_ADAPT_KERNEL_COPY (15 env-dump hits) | [coded only] (ADAPT_KERNEL_COPY [unverified]) |
| STRATA_POOL_UMWAIT, _UMWAIT_C02, _UMWAIT_ESC, _MWAITX | Speed | CPU wait instructions | UMWAIT closed for Linux desktop (MOD_PLAN line 1228) | [coded only] (closed) |
| STRATA_ADAPT_NOWAIT, STRATA_ADAPT_ASYNC_WAIT | Speed | Expert placement waits | 16 and 2 bench hits, unclear if env dumps | [unverified] |
| STRATA_QFUSE_HC, QSA_FUSE_NR, QSA_FUSE_GATE, QFUSE_HC_LOCAL, GDN_FUSE, GDN_TC_FUSE, HEAD_GR, GR_UNFUSED, LFUSE, LFUSE_GATE, LFUSE_PAIR, HIP_SWIGLU_FUSED, CVEC_FUSE, FUSE_DECODE, FUSED_SWIGLU_Q81, PF_FUSED_* (KQ, Q2N, SM75, WS, IQ2S, NATIVE, Q50, TILE, NT) | Speed / Unsure / KeyIfSet | Kernel fusion variants | Mostly zero hits; LFUSE 4, FUSE_DECODE 1, PF_FUSED_TILE 2 | [coded only] |

Inferences:
- The window-overhead switches with real A/B results are: DEFER_CPU and its rank-guard family, POOL_FUSE_PHASES, EXPDEC_DOWN, MTP_EAGER, MTP_HIP_PRIO, ROUTE_PRIOR (turbo), KV_RING_ADAPT, DEFER_AUTO, ROUTER_FUSE (screen only), EXPDEC_SWIGLU, and the one-pass ROUTE_FAST and WAIT_COPY256.
- The clearest uncovered window-overhead candidates are WAIT_FUSE (in preset, no isolated A/B, item 46 still open), FUSE_WAIT and WAIT_COPY256 as a pair (one noisy pass only), and ROUTE_FAST (one noisy pass only).

### Gaps
- Not checked: which switches are default-on in source. Only a sample was read (grep of getenv reads). Most are "== '1'" gated, so default-off. Exceptions seen: SPEC_COST default 1 in the NVIDIA table only; STRATA_STAGER_RING derived from thread count; STRATA_FUSE_WAIT default -1 (meaning not verified). PREFILL_RING default not checked.
- Abtest directory (36 hits on WAIT_COPY256, 39 on ROUTE_FAST) was only partly read. Those hits may be arm configs with env, not results.
- Noisy first pass in 25-kern-hip: base has a 95.9 tok/s short run; "clean" flags were not filtered.

## Key Question 2: is there a result for each window-overhead switch, and where

### Takeaway
Results exist for about 20 of the 153 candidates. The main places are the night queue RESULTS.md, the 2026-11-09 test-phase RESULTS.md, and the 2026-11-11 25-kern-hip and 10-rank-guard-fold jsonl files. MOD_PLAN has several "not built" or "never measured" lines that are now stale.

### Cited Findings
- Night queue: VERIFY_DEFER_CPU, VERIFY_DEFER_AUTO (and _MIN_T), KV_RING_ADAPT, SPEC_COST, and ROUTE_TAIL_SKIP arms in [RESULTS.md](/home/dan/night-queue-run/RESULTS.md).
- Test phase: 03 POOL_FUSE_PHASES, 04 ROUTER_FUSE/EXPDEC_DOWN/MTP screen, 04c EXPDEC_DOWN confirm, 06c turbo (ROUTE_PRIOR), 07b/07c/07d MTP_EAGER and MTP_HIP_PRIO in [test-phase RESULTS.md](../../../../bench/results/2026-11-09-test-phase/RESULTS.md).
- 2026-11-11 kern-hip: base, copy256, route-fast, expdec-swiglu, all3 in [25-kern-hip.jsonl](../../../../bench/results/2026-11-11-gpu/25-kern-hip.jsonl); confirm pass with base, expdec-swiglu, all3 in [25-kern-hip-confirm.jsonl](../../../../bench/results/2026-11-11-gpu/25-kern-hip-confirm.jsonl). Short decode means (tok/s): first pass base 116.7 (n=6), copy256 124.6 (n=2), route-fast 126.1 (n=4), expdec-swiglu 123.9 (n=4), all3 129.0 (n=4). Confirm pass: base 123.9 (n=7), expdec-swiglu 124.0 (n=10), all3 122.0 (n=10, one 96.5 outlier). These are my own row means from the jsonl; no per-arm statistic was computed beyond the mean.
- 2026-11-11 10-rank-guard-fold: [10-rank-guard-fold.jsonl](../../../../bench/results/2026-11-11-gpu/10-rank-guard-fold.jsonl), arms base, top2, top2-fold1, top2-fold2. MOD_PLAN verdict at [line 1590](../../../../MOD_PLAN.md).
- MOD_PLAN status lines that are stale against the results above: line 233 (EXPDEC_DOWN "never measured on gfx1100"; corrected at line 2250), line 1759-1760 (F21 WAIT_COPY256 and F9 ROUTE_FAST "not built"), line 1753 (FOLD "not built"), line 2623 (DEFER_AUTO "NOT measured", night queue has it).

### Inferences
- Any note saying a switch is "never measured" should be checked against the night queue and test-phase files before it is repeated.
- The test-phase RESULTS.md, not MOD_PLAN, is the most reliable place to see which switches were tested.

### Gaps
- MOD_PLAN was not reconciled line by line; only switches with keyword matches were checked.
- Night queue results for the defer arms were read from RESULTS.md tables only; raw jsonl was not re-analysed.

## Key Question 3: default-on switches never measured on gfx1100

### Takeaway
In source, most window-overhead switches are default-off (`getenv(...) == '1'`). The exception is that they are enabled in the fast-hip preset, not in code. STRATA_EXPDEC_DOWN is measured on gfx1100 (test phase 04c, +1.7% +-0.5 short), so MOD_PLAN line 233 is out of date. The clearest default-on-in-preset, never isolated switch is STRATA_WAIT_FUSE (with FUSE_WAIT and GR_FUSED).

### Cited Findings
- EXPDEC_DOWN default in code: 0 (`env_int_or("STRATA_EXPDEC_DOWN", 0)`, [iq_kernels.cu:4257](../../../../src/kernels/cuda/iq_kernels.cu)). It is on through the fast-hip preset (MOD_PLAN line 1749, commit f15ad2b51).
- WAIT_FUSE read as `== '1'`, default off ([verify.cpp:1623](../../../../src/core/verify.cpp)). Set to 1 in preset env dumps.
- ROUTE_PUBLISH read as `== '1'` ([verify.cpp:2517](../../../../src/core/verify.cpp)); forced off under deferral since d5b58f669 (MOD_PLAN line 166).
- SPEC_COST default "1" only in the NVIDIA defaults table ([nv_defaults.cpp:80](../../../../src/core/nv_defaults.cpp)); gfx1100 result is flat or worse (MOD_PLAN line 1168).

### Inferences
- Candidates for a window-overhead study on this GPU, in order of how little evidence exists: WAIT_FUSE (isolated A/B on host CPU time, est. 0-3%), FUSE_WAIT + WAIT_COPY256 (rerun with confirm pass), ROUTE_FAST (confirm pass), ROUTER_FUSE (retest without deferral, as MOD_PLAN 1599 says), GR_FUSED (isolated A/B, listed as a non-finite suspect).
- ROUTE_PUBLISH is a correctness topic before it is a speed topic.

### Gaps
- Default values for FUSE_WAIT (returns -1 when unset), PREFILL_RING, and GR_FUSED were not fully traced.
- No check of whether the preset env used for test-phase 04c is the same as the current fast-hip preset.
