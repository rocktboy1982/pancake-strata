# Pancake 7900 XT - development status

Updated as the work moves (owner's request, 2026-11-09).  Two columns: **Dev** = how much of the code exists and builds
(HIP + CUDA); **Test** = how far it is from decided (measured on a quiet machine, quality-checked, in the preset or rejected).
Order = MOD_PLAN phase 13 (biggest expected gain first).

Last update: 2026-11-11 17:15 (the table below is the 2026-11-09 state; Phase 17 of MOD_PLAN.md is the current plan and the block right below is the current status)

## Current status (2026-11-11): Phase 17, owner priority 1 = TTFT at long context

Working branch: `dev/p17/integration` (pushed; `dev/phase16-academic` plus the agent branches below). Full HIP build of it: OK; new unit tests
(`kv_ring_test`, `route_prior_auto_test`, `conv_cache_test`, `defer_auto_test`, `defer_rank_test`) pass under ctest. No GPU run of any new switch yet:
by day only code is written, the measurements are in the night queue (MOD_PLAN Phase 17, `tools/pancake/night_queue.py` when the agent finishes).

| Item | What | Dev | Test | Where |
| --- | --- | ---: | ---: | --- |
| 17.0 | TTFT harness (`ttft_bench.py`: single prompts 8K-100K, 10-turn session, side requests, sub-agent switch, warm-delta sweep, engine-log attach) and audit (`ttft_audit.py`) | 100% | harness 41 fake-server tests; 0% on a real engine | `dev/p17/ttftbench`, `ttftscen`, `ttftlevers` |
| 17.0 | Lever analysis: cache misses are the only multi-second loss (warm turns re-read ~1% of the prompt, TTFT ~1 s; a lost 54-62K conversation costs 21-25 s per turn) | analysis | - | `pancake-docs/TTFT_LEVERS.md` |
| 17.0 | `STRATA_PARK_REWIND_MAX` (skip the park when resuming from the live conversation's own checkpoint) + `STRATA_PARK_DIVERGENCE_LOG` | 100% | `conv_cache_test` PASS, 21 Python tests; 0% on a real engine | `dev/p17/parkrewind` |
| 17.0 | Prefill levers: chunk plans 32K-128K, equal-tail guard, staged-path floor, long-context fault check (#5539) | in flight | 0% | `dev/p17/prefilllevers` |
| 17.0 | Prefix stability of opencode / Copilot requests (M4): `prefix_check.py` (about 40 divergence kinds, redaction) + opt-in `STRATA_REQUEST_CAPTURE=DIR` | 100% | 28 tests PASS (fake and mock-engine server); no real client captured yet | `dev/p17/prefixcheck` |
| 17.1 | `STRATA_ROUTE_PRIOR_AUTO` (route prior only in lookup windows, via the F13 gate float) | 100% | unit test PASS; 0% on a real engine | `dev/p17/turbogate` |
| 17.1 | `STRATA_VERIFY_DEFER_AUTO` (deferral only in lookup windows); finding: deferral turns off six fused paths for the whole process | 100% | `defer_auto_test` 7,171 checks PASS; 0% on a real engine | `dev/p17/defergate` |
| 17.2 | Prose gap to Avalon: analysis + `prose_gap.py` (Avalon commits more tokens per window on prose; its MTP head is larger) | 100% | 11 fake-server tests; 0% measured | `dev/p17/prosegap` |
| 17.3 | Greedy ablation driver `ablate_greedy.py` + plan of 23 candidates | 100% | 19 fake-engine tests; 0% real | `dev/p17/ablate` |
| 17.4 | `STRATA_KV_RING_ADAPT` (adaptive resident KV ring; credit xyzzing / upstream PR #3919) | 100% | `kv_ring_test` PASS; speed 0% | `dev/p17/kvring` |
| 17.8 | Draft vocabulary from real code (38 languages, 60,895 rows): held-out misses 4.3% -> 1.55%; **no effect on decode or acceptance** | closed | measured (no gain) | `bench/results/2026-11-11-vs-avalon/` |
| review | Correctness review of all the unrun C++ above | in flight | - | `dev/p17/review` |
| night | Night queue runner `night_queue.py` (43 engine starts, quiet-machine guard, resume, RESULTS.md; deferral retest, TTFT baseline, lever arms, KV ring, prose profile; estimate 7 h 26 min, a guess) | 100% | 22 fake-engine tests PASS; never run on a real engine | `dev/p17/nightqueue`, `pancake-docs/NIGHT_QUEUE_RUNNER.md` |

Measured 2026-11-11 (same client, same ISTA files, warm second pass; `bench/results/2026-11-11-vs-avalon/REPORT.md`): long prompts are the
difference (Strata family 2.4-2.8K tok/s at 12-25K tokens vs Avalon 1.5-1.6K; exact `hip` preset 0.8K); decode is a near tie (104-111 tok/s);
Q2_0 = IQ2_XS; turbo -4..-8% (not a default); Avalon cannot run the Q2_0 file. Open: expert deferral (Q2_0 off +7-8% reproduced, IQ2_XS off -3%,
partly overlapped with agent activity): retest at night on a quiet machine.



Mode now: **development first** (owner, 13:15).  Test queues stopped.  All speed numbers measured today are provisional:
other work ran on the machine during them; the test phase repeats every A/B on a quiet machine.

| Prio | Item | MOD_PLAN | Expected gain | Where it is | Dev | Test |
| --- | --- | --- | --- | --- | ---: | ---: |
| P1a | Expert deferral on HIP (non-finite fix) | 8 | up to 1.45x, realistic +10-25% code edit | FIXED: bisect found ROUTE_PUBLISH; now off under deferral (d5b58f669); 512+512 tokens clean | 100% | 10% |
| P1a | Deferral rank guard (STRATA_VERIFY_DEFER_CPU_TOP) | 28 | lossy-small, keeps quality | coded in host pool + GPU copy, builds, probe clean | 100% | 5% |
| P1b | Expert cache judged in ms (simulator) | 14 | decides the policy | tools/pancake/time_ceiling.py: oracle +27..48%, online 2nd-miss +5..16% (code edits, model) | 100% | - |
| P1b | More swaps per window (existing --adapt-swaps) | 14 | +4..9% code edit (model) | nothing to build | 100% | 0% |
| P1b | Admit on k-th miss in W windows (new, STRATA_ADAPT_KMISS) | 14 | +5..16% serial, +45..52% with deferral (model) | in both tiers (kmiss_tier.hpp); must run with --adapt-async 1: the blocking tier stalls the GPU ~5.6 ms per swap round (profile); the kmiss+async probe was interrupted at engine start, not yet measured | 100% | 0% |
| closed | Predictor recall / candidate policies | 27, 30 | - | lookahead recall 19-21% of misses (measured): predicted promotions do not pay; 27 covered by kmiss + conversation seeds | 100% | 100% |
| P1b | Simulator: per-layer GPU/CPU overlap model (--overlap) | 14 | decides 8/28/29 | built: code edit deferral +33%, deferral + kmiss 2/8 +45..52%, item 29 +3% (model, alpha 0.25-0.5) | 100% | - |
| closed | CPU sparsity masks (Berkeley gate mask, TEAL, CATS) | 13 | +10-20%, lossy | REJECTED offline: at 70% sparsity gate mask cos 0.84 (~54% error), oracle 0.95; experts are dense -> no sparse kernel (bench/results/2026-11-09-sparsity) | 100% | 100% |
| P1d | One CPU pool phase per layer (STRATA_POOL_FUSE_PHASES) | 29 | 1-4%, exact | coded, builds (35c512984) | 100% | 0% |
| P1e | Domain / conversation expert seeds | 3-4 | +0.6-1.5 hit points | merged, builds; needs traces | 100% | 0% |
| P2a | turbo-hip (lossy router prior) decision | 7 | +30-50% | preset exists; quality gate pending | 100% | 30% |
| P2b | AcceptMoE depth-graded verify prior (STRATA_ROUTE_PRIOR_DEPTH) | 23 | turbo speed with less quality loss | coded, builds, probe clean | 100% | 0% |
| P2b | BFLA block-sparse prompt attention (feat/bfla-hip) | 19 | long-context TTFT (paper 1.34-1.44x at 128-256K) | e2e measured on Qwen3.6 UD-Q4_K_XL: 128K prompt 386.9 -> 1219.8 sparse (3.15x) -> 771.2 dense tok/s; recall top-1 48/48, KL 0.0014/0.0045; branch unmerged, HIP only | 100% | 10% |
| P2b | SpecFit missed-expert feature | 36 | 0-10% | belongs to DISCO (other session, feat/disco) | - | - |
| P2c | Small exact flags stack | phase 11 | +1.4..3.9% (provisional) | flags exist; A/B won 3/3 (noisy machine) | 100% | 50% |
| P2c | --lookup-max-t 12 | Avalon report | +1% (noise) | flag exists | 100% | 30% |
| done | Matching hipBLASLt table (100401) | Avalon report | +38% prompt (provisional) | in the 7900 XT presets | 100% | 80% |
| done | Pool pre-read (tandem.cpp) | phase 11 | - | loses even on a noisy machine; stays off | 100% | 90% |
| P3a | Turn-boundary checkpoints (existing switches) | 11 | seconds per turn | nothing to build | 100% | 0% |
| P3a | Restore test, preserve_thinking, tool-boundary checkpoints | 35, 31, 16 | seconds per turn | other session (dan-b5) | - | - |
| closed | Decode speed kept after long prompts | 2 | - | NOT the refill: same reply after a 13.8K prompt decodes as fast as after 2K (tools/pancake/decode_curve.py); the drop is prose content + cache warm-up -> items 3-4, 14 | 100% | 100% |
| P3c | Effort defaults, learned thinking stop | 12, 21, 33, 34 | tens of s per turn | other session (dan-b5) | - | - |
| P4 | GPU floor: kernel fusion, window-edge host stalls, parameter sweep | 32, 37 | 1-5% each | deferral profile: GPU still waits for the CPU 31% of kernel time -> fusions wait; the real host stall is the blocking tier's swaps (-> --adapt-async) ; tools/pancake/gpu_profile.py built | 20% | 0% |
| P4 | --mtp-q4 draft head | phase 11 | draft cost | flag exists | 100% | 0% |
| P1 | Shared window trace (STRATA_WINDOW_TRACE + tools/pancake/trace/wtrace.py) | 64 | 0 itself; one trace run answers ~10 offline questions | logger + reader + --dump-routing converter (14a4b3373) | 100% | 0% |
| P1 | Single poller in combine_cpu (STRATA_COMBINE_POLL1) | 46 | 0-3% | coded (93f7a963d) | 100% | 0% |
| P1 | Online swap budget (STRATA_ADAPT_SWAPS_AUTO) | 53 | 0-5% over the best fixed swaps | coded both tiers, controller unit-checked (6bc9530bc) | 100% | 0% |
| P1 | Q2_0 row prefetch (STRATA_Q2_PREFETCH=<bytes>) | 40 | 0-5% of CPU-expert time | coded (c44b1ad7b) | 100% | 0% |
| P1 | Lazy trailing park in the CPU pool (STRATA_POOL_LAZY_PARK) | code study | 0-1% | coded (c44b1ad7b) | 100% | 0% |
| P4 | expdec gate/up + SwiGLU + q8_1 in one launch (STRATA_EXPDEC_SWIGLU) | 48 | ~49 launches/window fewer | coded, bitwise by construction (2aca56777) | 100% | 0% |
| P4 | Router GEMV + top-10 + doorbell in one launch (STRATA_ROUTER_FUSE) | 48 | ~1% short chat (48 launch gaps/window) | coded, shared route code, bitwise by construction (7c0f03228); HC chain was already one kernel | 100% | 0% |
| P1 | Parked workers wait with MWAITX (STRATA_POOL_MWAITX) | code study | frees SMT siblings; wake +0.36 us | coded, micro-checked on the 5700G (4de66f87a) | 100% | 0% |
| meas | waitCPU stamped inside combine_cpu (profile) | 49 | measurement | coded (7bcb77d5e) | 100% | - |
| meas | dispatch 'jobs' split: expert-source sub-timer | - | measurement (jobs = 1.5 ms/window) | coded (c44b1ad7b) | 100% | - |
| P1 | CPU pre-sums missed rows (STRATA_CPU_PRESUM) | 51 | 0.5-1.5%, not bit-identical | coded, tied to the captured graph (cc5499fae) | 100% | 0% |
| P1 | Pre-summed rows to VRAM through ReBAR (STRATA_PRESUM_VRAM) | 52 | 1-2% | coded, HIP fine-grained VRAM (4f829da0e) | 100% | 0% |
| P1 | Tail-row drop | 5 | 0-5% | trace record + offline replay built (0488b80f4, 5c3d8a9fb); dan-9f adds the drafts() call with item 74; engine change only if the replay shows a gain | 60% | 0% |
| Ph14 | Test phase in one command (bench/specs/01-07 + tools/pancake/test_phase.sh) | 14 | - | built (1fbe4e2b1): abtest + quality gate per spec, resumable | 100% | 0% |
| P4 | Infinity Cache warm-up during the CPU wait (STRATA_IC_WARM_MB) | 47 | 0-2 ms/window | coded (269a06ed9) | 100% | 0% |
| P2 | Draft window costs from traces (STRATA_SPEC_COST_TABLE in serve + draft_costs.py) | 43 | 0-5% | coded (663107a56) | 100% | 0% |
| P2 | Draft stream on its own HW queue (STRATA_MTP_HIP_PRIO, existing, off on gfx1100) | 10 | R9700: draft 5.8 -> 3.5 ms | arm added; item 10 closed by code study | 100% | 0% |
| closed | ReplaySSM audit | 20 | - | replay stays (code study) | 100% | 100% |
| P1 | Pool quarter-size tail tasks + down-row prefetch while waiting (basalt ports) | 29-related | 0.5-1.5% | coded (e29bac539) | 100% | 0% |
| P1 | Tier catch-up while idle between requests (STRATA_IDLE_ADAPT) | 17 | 1-3% next turn | coded (43e741bc6) | 100% | 0% |
| P4 | Q2_0 down on the EXPDEC kernel (STRATA_EXPDEC_DOWN, existing) | Q2_0-down | 1-3% short | arm added | 100% | 0% |
| Ph14 | Launch/tile/pool sweep tool (tools/pancake/sweep.py) | 37 | 1-3% | built (7861dc8c1) | 100% | 0% |
| P1 | Per-layer dispatch: 4 quadratic loops -> linear (first_of, miss list, per-expert groups, tail-skip) | code pass (pancake-docs/CODE_OPTIMIZATION.md) | dispatch plan/jobs time | coded, exact by construction, HIP+CUDA build | 100% | 0% |
| Ph14 | Application end to end (agent-session timeline, idle-time use, server -> engine turn info) | 14 | per-turn time | needs the per-turn JSONL log of item 73 (serve/, dan-9f, not started) | 0% | 0% |
| Ph14 | A/B harness with quiet-machine gate + noise monitor (tools/pancake/abtest.py) | 14 | clean numbers | built (f21cc86a1) | 100% | 0% |
| Ph14 | Quality gate as a process (tools/pancake/quality_gate.py, token forcing per lossy arm) | 14 | - | built (c7c44f2d0) | 100% | 0% |
| Ph14 | Regression watch (tools/pancake/regression_watch.py) | 14 | - | built (c7c44f2d0) | 100% | 0% |
| Ph14 | Self-tuning from own sessions (tools/pancake/selftune.py, item 70) | 70 | 1-5% (keeps profiles current) | offline pass built (05e041b9b): traces -> candidate profile -> time model -> abtest spec; dry run: candidate profile +8.6%, + kmiss +23% modeled | 100% | 0% |
| Ph14 | Diagnostic: decode speed along a reply (tools/pancake/decode_curve.py) | 2 | - | built, used to close item 2 | 100% | - |

## Items by session dan-9f (server, agent, loader, draft area) - merged into port/rdna3-decode

| Item | What | Switch / file | Default | Exact? | Dev | Test |
| --- | --- | --- | --- | --- | ---: | ---: |
| 73 | Per-turn JSONL log | `turn_log` / `pancake-turns.jsonl` | on (Flash-Next presets) | - | 100% | 0% |
| 74 | Eager MTP draft steps (all launched at once, chain cut on the host) + `window_trace().drafts()` | `STRATA_MTP_EAGER=1` | off | exact | 100% | 0% (spec 07, identity check) |
| 75 | Persistent adapt worker (one JobThread) | `STRATA_ADAPT_PERSIST=0` restores old | **on** | exact | 100% | 0% (spec 02) |
| 76 | Device spin scheduling | `STRATA_DEVICE_SPIN=1` | off | exact | 100% | 0% (spec 04, log must say "set") |
| 38 | Linux O_DIRECT expert-arena load | `STRATA_UNBUFFERED_LOAD=0` restores buffered | **on** (Linux) | exact | 100% | 0% (test plan 2.10) |
| 35/83 | Restore-equivalence test | `tools/pancake/restore_check.py` | first step of test_phase.sh | gate | 100% | 0% |
| 88 | Agent settings in all Flash-Next presets | env_to_user, effort_position end, grace 512, close retry | on | text | 100% | - |
| status page | `/status`, `/pancake`, `/status.json` (opencode/Gemini session, `214c8cd31`) | serve/web/status.html | on | - | 100% | tests 13/13 |
| 33 | Loop-guard behaviour counters (edits without build/test, reads without edit) | `STRATA_LOOP_BEHAVIOUR`, config `loop_guard_behaviour` | **log** (note opt-in) | exact in log mode | 100% | read counts from real sessions |
| 34 | Rumination signal (always measured) + opt-in thinking close | config `rumination_close` | measure on, close OFF | lossy only when on | 100% | pick a threshold from turn logs |
| 31 | preserve_thinking passthrough + keep_reasoning keeper | `chat_template_kwargs.preserve_thinking`, `STRATA_KEEP_REASONING` | off | exact | 100% | opencode already resends reasoning (22/25 exact reuse) |
| 80 | hipBLASLt descriptor cache + table extender tool | `STRATA_HIPBLASLT_DESC_CACHE` | **on** | exact | 100% | 0% (spec 09-prefill; extender run = GPU ~30 min) |
| 90 + 71 | Park the branch at the pin; auto-pin system+tools prefix | `STRATA_PIN_PARK_TOKENS` (on), `STRATA_AUTO_PIN` (off) | see switch | exact | 100% | 0% (sub-agent switch check) |
| 16 | Tool-boundary checkpoints + FLOPs-per-byte eviction | `STRATA_CACHE_TOOL_BOUNDARY`, `STRATA_CKPT_EVICT_FLOPS` | off | exact vs re-read | 100% | 0% (`restore_check.py --tool-boundary`) |
| 39 | Disk tier for parked conversations | `STRATA_CONVERSATION_DISK_DIR` | off | exact | 100% | 0% (`restore_check.py --disk`) |
| 79 | PLE gather off the critical path; compact PLE with direct I/O | `STRATA_PLE_PIPE`, `STRATA_PLE_COMPACT` | off | exact (CPU byte parity) | 100% | 0% (cold 32K TTFT arms) |
| 86 | RAM-budget conversation cache (opt-in) + PSI guard | `STRATA_CONV_BUDGET`, `STRATA_CONV_KEEP_GIB`, `STRATA_PSI_GUARD` (on) | see switch | exact (timing) | 100% | 0% |
| 93 | Tests labelled cpu/gpu/model, strata_tests target | `ctest -L cpu` 117/0/6, `ctest -L gpu` 124 (parity tolerances untested) | - | - | 100% | cpu done, gpu in test phase |
| 95b | Stubs in *_stub.cpp + check_stubs.py; dead code; keepawake flake | - | - | no behaviour change | 100% | 100% |
| cleanup | STRATA_ARENA_WARM removed (measured worse); native_moe_combine_multi_hits kept (upstream code, unused; removal would only add fork drift) | this session | - | - | 100% | - |

Also from this session: `STRATA_HOST_NATIVE` (CMake, host C++ for this CPU) - built as build-hip-native, spec 08-build.

Development: engine side COMPLETE.  Left, by decision or dependency: 5 (engine change only if the trace replay shows a
gain; needs dan-9f's drafts() call), 18 GDN prefill chunk 64 (parked: cold prompts only, 90.8% of turns hit the cache),
phase-14 agent timeline (needs item 73's per-turn log, dan-9f).  Next: the test phase = tools/pancake/test_phase.sh
(8 spec groups incl. 04b) + tools/pancake/sweep.py, after dan-9f stops builds/GPU jobs.