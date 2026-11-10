# Test plan: what each technology really brings (RX 7900 XT, 2026-11-09)

Source of the claims: README.md section 2 (Table 2A, 2C) and section 6 (every switch with its claimed gain).
Rule for every item: measure it alone against the preset it sits in; **keep** it if the gain is larger than its
95% interval; if it shows **no real gain**, run the optimization loop (below) before deciding; only then move on.

## How every test runs

- Harness: `tools/pancake/abtest.py` (fresh server per run, A B B A order, 3 reps, quiet-machine gate, noisy runs
  repeated, 95% intervals); lossy items also `tools/pancake/quality_gate.py` (token forcing); cache/checkpoint items
  also `tools/pancake/restore_check.py`.
- Workloads (the same as the published references): short chat, code edit (2K prompt), 11.7K prompt (read speed +
  decode after it); for long-context items 32K / 64K / 128K prompts.
- Machine: the other sessions pause builds; the engine on 8080 is stopped during the test.
- Result per item: one line in `bench/results/<day>-test-plan/RESULTS.md`: claim, measured, verdict, action.

## The optimization loop (for an item with no real gain)

1. Where does the time go with the item on: engine `decode timing` line (GPU / CPU experts / draft), and
   `tools/pancake/gpu_profile.py` for GPU items.
2. Read the code path with Serena (symbol references, call sites) and graphify (who depends on it) - is it
   actually active in this preset (log line), is something else masking it, is a parameter wrong for gfx1100?
3. Tune its knobs (`tools/pancake/sweep.py`), retest.  At most 2-3 attempts per item.
4. Decide: keep on (gain proven), keep opt-in (gain on other hardware/workloads only), or remove.

## Phase 0 - baselines and the published references (one run each)

| # | Test | Reference to beat or match |
|---|---|---|
| 0.1 | `strata-7900xt-optimization/bench.py` (4 x 300-token decodes, 2 cold ~22K prefills) | Strata 0.1.43 tuned: decode 75-77, prefill 1,500-1,550 tok/s |
| 0.2 | Avalon `bench.py` (5 prompts x 2 passes, 512 tokens, thinking off) | Avalon: 2nd-pass median 100.3, best 115 tok/s |
| 0.3 | `tools/strata_bench.py --long` (chat, 4K/16K/32K prefill, agent session) | Avalon: chat 96.4; prefill 1,180 / 1,771 / 1,805; agent first token 4.1-4.3 s, decode 66-72 |
| 0.4 | code edit, 1,500 tokens, 2 rounds | Avalon 118 (MTP) / 168 (ngram+MTP) tok/s |

Any workload where pancake is below the reference becomes its own item in the loop above (component by component:
prefill, short decode, edit decode, long-context decode, agent turn latency).

## Phase 1 - README Table 2A claims (7900 XT)

| # | Claim (README) | Test | Pass |
|---|---|---|---|
| 1.1 | fast-hip 112-126 tok/s | Phase 0 runs + abtest short/edit | inside the range |
| 1.2 | Q2_0 fast ~135 tok/s | abtest on flashnext-q2_0-fast-hip | >= 130 |
| 1.3 | turbo-hip 149-168 tok/s (lossy) | abtest + quality gate | speed in range, gate not REJECT |
| 1.4 | hipBLASLt 100401 table +38% prompt | arm with the old 100200 table | prompt gain > interval |
| 1.5 | `--pcie-frac 0` +6% chat / +13% edit | arm with the PCIe share on | gain > interval |
| 1.6 | +6% chat / +13% edit over the starting point | fast-hip vs the exact preset | gain > interval |

## Phase 2 - README section 6, group by group (bench/specs/)

| # | Group (README 6x) | Items (spec) | Claimed | Note |
|---|---|---|---|---|
| 2.1 | 6B deferral | `STRATA_VERIFY_DEFER_CPU`, `_TOP=2` (01-deferral) | model +33% edit | lossy-small: quality gate |
| 2.2 | 6B tier | `--adapt-async`, `STRATA_ADAPT_KMISS`, `_SWAPS_AUTO`, `STRATA_IDLE_ADAPT`, `STRATA_ADAPT_PERSIST` (02-tier) | +5-16% edit (model) | kmiss needs async |
| 2.3 | 6D CPU pool | `FUSE_PHASES`, `TAIL4`, `LAZY_PARK`, `Q2_PREFETCH`, `MWAITX` (03-cpu-pool) | 1-4% each | |
| 2.4 | 6C GPU launches | `COMBINE_POLL1`, `EXPDEC_SWIGLU`, `ROUTER_FUSE`, `IC_WARM_MB`, `EXPDEC_DOWN`, `DEVICE_SPIN` (04) | ~1% each | bitwise items: token forcing too |
| 2.5 | 6C pre-sum | `STRATA_CPU_PRESUM`, `+PRESUM_VRAM` (04b) | 0.5-2% | not bit-identical: gate |
| 2.6 | 6A RDNA3 kernels | the ported RDNA3 set on vs off (exact preset) | 87-101 -> ~112 | one arm per kernel family if the group wins less than claimed |
| 2.7 | 6C draft | small-flag stack, `--lookup-max-t 12`, `STRATA_MTP_HIP_PRIO`, `STRATA_MTP_EAGER`, `--mtp-q4` (05, 07) | +1.4-3.9% stack | MTP_EAGER: identity check |
| 2.8 | 6B lossy routing | turbo prior vs depth-graded prior (06) | 7 vs 23: pick one | quality gate decides |
| 2.9 | 6G-6I server | turn-boundary checkpoints, restore check, loop guard on/off on an agent replay | seconds per turn | restore_check first |
| 2.10 | loader (item 38) | `STRATA_UNBUFFERED_LOAD=0/1`: start time, `STRATA_VERIFY_ARENA=1`, PSI | start 28-65 s -> 10-14 s | decode unchanged |
| 2.11 | BFLA (branch) | 32K / 64K / 128K prompt read, needle recall | 3.15x at 128K | branch build |
| 2.13 | prefill (items 80, 79) | `STRATA_HIPBLASLT_DESC_CACHE` on vs 0 (09-prefill), then the extended table from `hipblaslt_extend.py` vs the shipped one on 1.5-3.2K / 4-4.2K / 16K / 32K | exact; prompt read speed | greedy identity + identical "Lt solution=" lines |
| 2.14 | agent signals (33, 34, 31) | no A/B: counters and rumination read from real-session turn logs; a rumination threshold, if picked, gets a lossy arm | - | - |
| 2.15 | cache / checkpoints (16, 39, 90, 71) | `restore_check.py`, `--tool-boundary`, `--disk` (all turns MATCH; disk: >= 2 writes and read-backs; tool retry caches more); sub-agent switch: parent P >= 10K -> sub-agent (same system+tools) -> P restores the parked P (prompt time vs PIN_PARK_TOKENS=-1, tokens vs STRATA_CKPT_REREAD=1) | seconds per turn | exact |
| 2.16 | prefill PLE (79) | cold 32K, `--ple-io direct`, caches dropped: A WAIT_TRACE=1, B + PLE_PIPE=1, C + PLE_COMPACT=1 | TTFT | greedy identity |
| 2.17 | memory (86, 38) | load line "memory pressure during the expert arena load" ~0 holds; one run with STRATA_CONV_BUDGET=ram STRATA_CONV_KEEP_GIB=8 to read the budget | start time, no stalls | - |
| 2.18 | GPU tests (93) | `ctest -L gpu --output-on-failure` (124; first run of rdna parity tolerances and 25 hip_* tests) | - | all pass or tolerances fixed |
| 2.12 | sweep | `tools/pancake/sweep.py` over expdec / pool knobs | 1-3% | after 2.3-2.4 |

dan-9f's branches (39 disk tier, 16/31/71 checkpoints, 33/34 loop guard, 79/80 prefill, 93/95b) join phase 2 as
they merge, each with its own arm.

## Order and time

Phase 0 (~1 h) -> 1 (~1.5 h) -> 2.1, 2.2 (largest modeled gains) -> 2.3-2.5 -> 2.6-2.8 -> 2.9-2.12.
About 1 h per group with 3 reps; one group at a time, results and decisions written after each.

## Phase 3 - candidates from the Strata fork survey (reports/Forkuri Strata câștig viteză.md, 2026-11-10)

| # | Spec | Items | Kind |
|---|---|---|---|
| 3.1 | 10-measure | `STRATA_DUMMY_NODES=10/40` (graph node cost), per-run AnonHugePages / VRAM / GTT / clocks (harness telemetry, done), `strata_bench.py --runs 3 --long` median [min-max] | measurement |
| 3.2 | 11-env-prefill | `STRATA_GDN_PP=2`, `STRATA_PREFILL_LAYER_MAJOR=1`, `STRATA_DENSE_MMQ=1` (after checking the MMQ `% 256` gate), `--vram-reserve-mib` 444 / 1000 | env only, exact |
| 3.3 | 12-env-tier | `--kv-grow`, `--adapt-decay` x `--adapt-swaps` grid, `--spec-min-p 0.7` | env only |
| 3.4 | 13-small-ports | upstream #5655 route_prior fix (first, mandatory), PCIe staging off at `--pcie-frac 0` (strata-slim 0003), DraftPolicy clip (gfx1030), `POOL_GUIDED` (Strata-GLM), Argon AVX2 accumulators / row granularity, rdna3 `1799c0a3a` + `3dbb40516` | small ports, exact |
| 3.5 | 14-larger | routed-only prompt expert streaming (gfx1030 `PF_STREAM_ROUTED`, check `PIN_BY_NEED` first), `VRAM_CALIB` (#5691), fragile-token recheck over turbo-hip (HotRod) with a momoe-style quality gate | medium / large |

Credits: every candidate that is adopted gets its author in README.md section 9 and a row in
pancake-docs/RESEARCH_AND_FINDINGS.md section 4 (commit, what was taken, measured result), in the adopting commit.

Rebase hazard noted: upstream v0.2.13 enables the hipBLASLt table automatically only on gfx1151 - keep `STRATA_HIPBLASLT_TABLE=on` in the -hip presets.
