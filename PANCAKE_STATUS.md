# Pancake 7900 XT - development status

Updated as the work moves (owner's request, 2026-11-09).  Two columns: **Dev** = how much of the code exists and builds
(HIP + CUDA); **Test** = how far it is from decided (measured on a quiet machine, quality-checked, in the preset or rejected).
Order = MOD_PLAN phase 13 (biggest expected gain first).

Last update: 2026-11-09 15:20

Mode now: **development first** (owner, 13:15).  Test queues stopped.  All speed numbers measured today are provisional:
other work ran on the machine during them; the test phase repeats every A/B on a quiet machine.

| Prio | Item | MOD_PLAN | Expected gain | Where it is | Dev | Test |
| --- | --- | --- | --- | --- | ---: | ---: |
| P1a | Expert deferral on HIP (non-finite fix) | 8 | up to 1.45x, realistic +10-25% code edit | FIXED: bisect found ROUTE_PUBLISH; now off under deferral (d5b58f669); 512+512 tokens clean | 100% | 10% |
| P1a | Deferral rank guard (STRATA_VERIFY_DEFER_CPU_TOP) | 28 | lossy-small, keeps quality | coded in host pool + GPU copy, builds, probe clean | 100% | 5% |
| P1b | Expert cache judged in ms (simulator) | 14 | decides the policy | tools/pancake/time_ceiling.py: oracle +27..48%, online 2nd-miss +5..16% (code edits, model) | 100% | - |
| P1b | More swaps per window (existing --adapt-swaps) | 14 | +4..9% code edit (model) | nothing to build | 100% | 0% |
| P1b | Admit on k-th miss in W windows (new, STRATA_ADAPT_KMISS) | 14 | +5..16% code edit (model) | coded, builds (b5f1842f0) | 100% | 0% |
| P1b | Candidate policies in the simulator (Harmonia, MoE-Infinity, ProMoE), predictor recall | 27, 30 | decides further work | not started | 0% | - |
| P1b | Simulator: per-layer GPU/CPU overlap model (to price deferral, rank guard, item 29) | 14 | decides 8/28/29 settings | not started (review by dan-b5: the model sums GPU+CPU) | 0% | - |
| closed | CPU sparsity masks (Berkeley gate mask, TEAL, CATS) | 13 | +10-20%, lossy | REJECTED offline: at 70% sparsity gate mask cos 0.84 (~54% error), oracle 0.95; experts are dense -> no sparse kernel (bench/results/2026-11-09-sparsity) | 100% | 100% |
| P1d | One CPU pool phase per layer (STRATA_POOL_FUSE_PHASES) | 29 | 1-4%, exact | coded, builds (35c512984) | 100% | 0% |
| P1e | Domain / conversation expert seeds | 3-4 | +0.6-1.5 hit points | merged, builds; needs traces | 100% | 0% |
| P2a | turbo-hip (lossy router prior) decision | 7 | +30-50% | preset exists; quality gate pending | 100% | 30% |
| P2b | AcceptMoE depth-graded verify prior / SpecFit | 23, 36 | 0-15% | not started (DISCO itself: other session) | 0% | 0% |
| P2c | Small exact flags stack | phase 11 | +1.4..3.9% (provisional) | flags exist; A/B won 3/3 (noisy machine) | 100% | 50% |
| P2c | --lookup-max-t 12 | Avalon report | +1% (noise) | flag exists | 100% | 30% |
| done | Matching hipBLASLt table (100401) | Avalon report | +38% prompt (provisional) | in the 7900 XT presets | 100% | 80% |
| done | Pool pre-read (tandem.cpp) | phase 11 | - | loses even on a noisy machine; stays off | 100% | 90% |
| P3a | Turn-boundary checkpoints (existing switches) | 11 | seconds per turn | nothing to build | 100% | 0% |
| P3a | Restore test, preserve_thinking, tool-boundary checkpoints | 35, 31, 16 | seconds per turn | other session (dan-b5) | - | - |
| closed | Decode speed kept after long prompts | 2 | - | NOT the refill: same reply after a 13.8K prompt decodes as fast as after 2K (tools/pancake/decode_curve.py); the drop is prose content + cache warm-up -> items 3-4, 14 | 100% | 100% |
| P3c | Effort defaults, learned thinking stop | 12, 21, 33, 34 | tens of s per turn | other session (dan-b5) | - | - |
| P4 | GPU floor: kernel fusion, window-edge host stalls, parameter sweep | 32, 37 | 1-5% each | GPU profile done; code not started | 10% | 0% |
| P4 | --mtp-q4 draft head | phase 11 | draft cost | flag exists | 100% | 0% |
| Ph14 | Application end to end (agent-session timeline, idle-time use, server -> engine turn info) | 14 | per-turn time | after phase 13 | 0% | 0% |
| Ph14 | New processes: A/B harness, quality-gate process, self-tuning, regression watch | 14 | - | queue scripts exist in the scratchpad; repo tool not started | 10% | 0% |

Development order from here: P2b AcceptMoE prior
-> P1b simulator overlap model + policies (27, 30) -> P4 GPU fusions -> phase 14 tools (A/B harness, regression watch).  Then the test phase.
