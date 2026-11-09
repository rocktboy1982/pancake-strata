# Pancake 7900 XT - development status

Updated as the work moves (owner's request, 2026-11-09).  Percent = how far the item is from "decided and in the default
preset, or rejected with a measurement".  Order = MOD_PLAN phase 13 (biggest measured gain first).

Last update: 2026-11-09 13:05

| Prio | Item | MOD_PLAN | Expected gain | Status | % |
| --- | --- | --- | --- | --- | ---: |
| P1a | Expert deferral on HIP (non-finite fix), then rank guard | 8, 28 | up to 1.45x, realistic +10-25% code edit | bisect queued (queue2, 6 runs) | 10% |
| P1b | Expert cache judged in ms (simulator) | 14, 27, 30 | 0-30% code edit | simulator built and run (tools/pancake/time_ceiling.py): oracle +27..48%, online 2nd-miss +5..16% on code edits | 100% |
| P1b | Engine: more swaps per window (existing switch) | 14 | +4..9% code edit (model) | engine A/B queued (queue3: --adapt-swaps 64/96/160); measures the real PCIe contention | 30% |
| P1b | Engine: admit on k-th miss in W windows (new, STRATA_ADAPT_KMISS) | 14 | +5..16% code edit (model) | coded and syntax-checked (b5f1842f0); A/B 2,8 / 2,4 / 3,8 queued (queue4) | 60% |
| P1c | CPU sparsity masks (Berkeley gate mask, TEAL, CATS) | 13 | +10-20%, lossy | not started (offline pass on traces first) | 0% |
| P1d | One CPU pool phase per layer (STRATA_POOL_FUSE_PHASES) | 29 | 1-4%, exact | coded, syntax-checked; build + A/B queued (queue3) | 60% |
| P1e | Domain / conversation expert seeds | 3-4 | +0.6-1.5 hit points | merged and built; needs traces + multi-turn A/B | 60% |
| P2a | turbo-hip (lossy router prior) decision | 7 | +30-50% | speed measured (149-168 tok/s); quality gate pending | 30% |
| P2b | DISCO / SpecFit / AcceptMoE drafting | 15, 36, 23 | 0-15% | DISCO on feat/disco (other session); rest not started | 5% |
| P2c | Small exact flags stack | phase 11 | +1.4..3.9% measured | A/B done (wins 3/3); quality check + preset pending | 80% |
| P2c | --lookup-max-t 12 | Avalon report | +1% (noise) | measured once; retest on a quiet machine | 50% |
| done | Matching hipBLASLt table (100401) | Avalon report | +38% prompt, +15% decode | in the 7900 XT presets, measured | 100% |
| done | Pool pre-read (tandem.cpp) | phase 11 | - | measured: loses (95 vs 107-122 tok/s), stays off | 100% |
| P3a | Turn-boundary checkpoints (STRATA_CACHE_MESSAGE_BOUNDARY + --prompt-cache-tail) | 11 | seconds per turn | test queued (queue2) | 40% |
| P3a | Restore test, preserve_thinking, tool-boundary checkpoints | 35, 31, 16 | seconds per turn | other session (dan-b5) | - |
| P3b | Decode speed kept after long prompts | 2 | ~97 -> ~120 tok/s after 11.7K | lend/chunk sweep queued (queue2) | 20% |
| P3c | Effort defaults, learned thinking stop | 12, 21, 33, 34 | tens of s per turn | other session (dan-b5) | - |
| P4 | GPU floor: kernel fusion, window-edge host stalls, parameter sweep | 32, 37 | 1-5% each | GPU profile done (bench/results/2026-11-09-7900xt-gpuprof); work not started | 30% |
| P4 | --mtp-q4 draft head | phase 11 | draft cost | queued (queue2) | 20% |
| Ph14 | Application end to end (agent-session timeline, idle-time use, server -> engine turn info) | 14 | per-turn time | after phase 13 | 0% |
| Ph14 | New processes: nightly A/B harness, quality-gate process, self-tuning from own sessions, regression watch | 14 | - | after phase 13 (the queue scripts are the seed of the A/B harness) | 5% |

Running now: queue2 (lend sweep, --mtp-q4, deferral bisect, turn checkpoints), then queue3 (swaps sweep, build, item 29 A/B), then queue4 (k-th-miss admission A/B).
