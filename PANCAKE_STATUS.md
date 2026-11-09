# Pancake 7900 XT - development status

Updated as the work moves (owner's request, 2026-11-09).  Two columns: **Dev** = how much of the code exists and builds
(HIP + CUDA); **Test** = how far it is from decided (measured on a quiet machine, quality-checked, in the preset or rejected).
Order = MOD_PLAN phase 13 (biggest expected gain first).

Last update: 2026-11-09 15:18

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
| P1 | CPU pre-sums missed rows | 51 | 0.5-1.5%, not bit-identical | deferred: item 50 measures the pull cost first (built only if > ~3 us/layer) | 0% | 0% |
| P1 | Tail-row drop | 5 | 0-5% | trace record + offline replay built (0488b80f4, 5c3d8a9fb); dan-9f adds the drafts() call with item 74; engine change only if the replay shows a gain | 60% | 0% |
| Ph14 | Test phase in one command (bench/specs/01-07 + tools/pancake/test_phase.sh) | 14 | - | built (1fbe4e2b1): abtest + quality gate per spec, resumable | 100% | 0% |
| Ph14 | Application end to end (agent-session timeline, idle-time use, server -> engine turn info) | 14 | per-turn time | needs the per-turn JSONL log of item 73 (serve/, dan-9f, not started) | 0% | 0% |
| Ph14 | A/B harness with quiet-machine gate + noise monitor (tools/pancake/abtest.py) | 14 | clean numbers | built (f21cc86a1) | 100% | 0% |
| Ph14 | Quality gate as a process (tools/pancake/quality_gate.py, token forcing per lossy arm) | 14 | - | built (c7c44f2d0) | 100% | 0% |
| Ph14 | Regression watch (tools/pancake/regression_watch.py) | 14 | - | built (c7c44f2d0) | 100% | 0% |
| Ph14 | Self-tuning from own sessions (tools/pancake/selftune.py, item 70) | 70 | 1-5% (keeps profiles current) | offline pass built (05e041b9b): traces -> candidate profile -> time model -> abtest spec; dry run: candidate profile +8.6%, + kmiss +23% modeled | 100% | 0% |
| Ph14 | Diagnostic: decode speed along a reply (tools/pancake/decode_curve.py) | 2 | - | built, used to close item 2 | 100% | - |

Development order from here: engine side done except 51 (after item 50's measurement) and 5 (needs draft confidences
from dan-9f's draft area).  Next: agent-session timeline (phase 14).  Then the test phase = tools/pancake/test_phase.sh
(after the other session stops builds/GPU jobs).  Coordination: dan-9f owns serve/agent items (73, 35, 66, 12, 31, 11,
16, 39, 71, 21, 33, 34, 38, 69, 74-76); this session owns the engine/CPU/tier items.