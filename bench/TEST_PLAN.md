# Test plan: what each technology really brings (RX 7900 XT, 2026-11-09)

Source of the claims: README.md section 2 (Table 2A, 2C) and section 6 (every switch with its claimed gain).
Rule for every item: measure it alone against the preset it sits in; **keep** it if the gain is larger than its
95% interval; if it shows **no real gain**, run the optimization loop (below) before deciding; only then move on.

## How every test runs (tiers, 2026-11-10: rigorous testing only where it is needed)

Why: the night of 11-09/10 spent ~17 h for 4 decisions. Where the time went: a 1-minute load-average wait (45 of
test 04's 70 min, removed); runs during memory compaction / oomd kills while loading the 33 GiB arena (n = 1-2 per
arm, intervals of +-30% to +-265%); the same item re-tested three times (04b / 04d / 04e presum) without a decision;
every item, even env-only and exact ones, got the full 3-rep A B B A treatment plus the quality gate.

| Tier | When | How (`tools/pancake/abtest.py`) | Time per arm |
| --- | --- | --- | --- |
| 0. offline | anything a replay or a byte count can answer (cache policy, drafting headroom, park decisions) | cache_sim, `tail_replay.py`, log replays; no GPU | 0 GPU minutes |
| 1. screen | every candidate, first | `--tier screen`: one A B pass, short + edit, 2 requests per case in the same server; promote if >= +2% on a case and no case below -3%, else drop | ~2-3 min |
| 2. confirm | only screen winners | `--tier confirm` (default): A B B A up to 3 reps with **early stop** (stops once every arm/case is decided or its interval is below 1.5% of base) | ~5-12 min |
| 3. rigorous | only: lossy items, a change of a preset default, anything whose generated text differs from base, long-context claims | confirm + `--no-early-stop` + `quality_gate.py` (token forcing: rejects only large deviations) + **`task_gate.py`** (A55: fixed task set, repeated, decides between close variants; includes the tool-call validity check, so no separate tool-call gate) + 32K/64K/128K cases | 20-60 min (+ ~15-25 min per arm for the task gate at 5 reps; est., first run pending) |

Rules:
- Lossy arms: token forcing first (a REJECT there ends it), then the task gate on the survivors:
  `python tools/pancake/task_gate.py SPEC.json --out bench/results/<day>-<name>` (`--dry-run` checks the spec and the task set
  without a server; the base arm runs twice as the drift check; verdict per arm: BETTER / PASS / INCONCLUSIVE / REJECT, see
  README 7.8). Why a task set and not more KLD: near the baseline, KLD / top-1 do not predict the task score
  ("Displacement Is Not Direction", arXiv 2606.19558). INCONCLUSIVE means more repetitions (`--reps 10-20`) or more
  tasks, not a re-run of the same size.
- Exactness is checked for free: every run hashes the generated text; "identical on every case" = exact, no quality
  gate. Placement changes (GPU vs CPU expert kernels) can change the last bits and the text without being lossy;
  those go to the quality gate only if they are candidates for a default.
- Fail fast: a server that dies before ready is retried once, then the batch aborts with the free-2MiB-block number.
  Fix the machine (`sudo tools/pancake/os_tune.sh compact` / `install`), do not keep running.
- Before a batch: `tools/pancake/os_tune.sh` (free RAM in 2 MiB blocks; abtest warns below 20 GiB, the arena locks 33 GiB), CPU clock at 3.9-4.1 GHz in the per-run
  telemetry (slow runs line up with 3.4-3.7 GHz), no other session running builds, agents or long jobs
  (`pgrep -f test_phase.sh` tells other sessions a test is running).
- One decision per item: if an item comes back "no difference" from confirm, it is opt-in or removed; re-test it
  only after a code change, not to try again.
- Batch arms: put all env-only arms of one area in one spec (one base, many arms) instead of one spec per arm.
- Workloads: short chat, code edit (2K prompt); the 11.7K read case and 32K / 64K / 128K only in the rigorous tier
  or for prefill items.
- Machine: the engine on 8080 is stopped during the test.
- Result per item: one line in `bench/results/<day>-test-plan/RESULTS.md`: claim, tier reached, measured, verdict,
  action.

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

## Phase 3 - candidates from the Strata fork survey (reports/Strata forks speed gains.md, 2026-11-10)

| # | Spec | Items | Kind |
|---|---|---|---|
| 3.1 | 10-measure | `STRATA_DUMMY_NODES=10/40` (graph node cost), per-run AnonHugePages / VRAM / GTT / clocks (harness telemetry, done), `strata_bench.py --runs 3 --long` median [min-max] | measurement |
| 3.2 | 11-env-prefill | `STRATA_GDN_PP=2`, `STRATA_PREFILL_LAYER_MAJOR=1`, `STRATA_DENSE_MMQ=1` (after checking the MMQ `% 256` gate), `--vram-reserve-mib` 444 / 1000 | env only, exact |
| 3.3 | 12-env-tier | `--kv-grow`, `--adapt-decay` x `--adapt-swaps` grid, `--spec-min-p 0.7` | env only |
| 3.4 | 13-small-ports | upstream #5655 route_prior fix (first, mandatory), PCIe staging off at `--pcie-frac 0` (strata-slim 0003), DraftPolicy clip (gfx1030), `POOL_GUIDED` (Strata-GLM), Argon AVX2 accumulators / row granularity, rdna3 `1799c0a3a` + `3dbb40516` | small ports, exact |
| 3.5 | 14-larger | routed-only prompt expert streaming (gfx1030 `PF_STREAM_ROUTED`, check `PIN_BY_NEED` first), `VRAM_CALIB` (#5691), fragile-token recheck over turbo-hip (HotRod) with a momoe-style quality gate | medium / large |

Additions from the active-fork scan (reports/Active Strata forks November.md): spec 11/12 gain env arms
`STRATA_LOOKAHEAD=0`, `STRATA_PREFILL_ROUTED_STREAM=1` x `STRATA_PREFILL_COPY_WAIT_DEDUPE=1`, `STRATA_MMVF_ROWS=1`,
`STRATA_MMVQ_R=2/4` (token forcing), `--spec-min-p 0.7`; spec 13 gains build bundle A (global-vs-flat load hint,
gfx11 lighter barrier, DPP shuffles, sampler loads in flight) and bundle B (ACTQ_SKIP, FUSE_DECODE on HIP,
PARK_REWIND_MAX, POOL_DAG=2); fixes before testing: prompt-cache root < 2048 (ElXreno a18d4668ec), kv_stream bound
(kllmagn e3056ec81b), layer-major group-scratch cap (Spark-NV 8a3475b6d2, before PREFILL_LAYER_MAJOR).

Credits: every candidate that is adopted gets its author in README.md section 9 and a row in
pancake-docs/RESEARCH_AND_FINDINGS.md section 4 (commit, what was taken, measured result), in the adopting commit.

Rebase hazard noted: upstream v0.2.13 enables the hipBLASLt table automatically only on gfx1151 - keep `STRATA_HIPBLASLT_TABLE=on` in the -hip presets.

## Phase 4 - speed ideas from all fork commits (reports/Speed ideas from all forks.md, 2026-11-10)

Plan items F1-F22 are in MOD_PLAN.md Phase 15. The order below is chosen so that each test answers the most: first
the tests that need no GPU, then env-only arms and short probes that decide whether a port is worth writing, then the
ports. Same rules as above (pinned prompts, token forcing for anything lossy, tool-call gate for anything that
touches drafting or routing). All gains quoted in the plan are fork claims until a row here says otherwise.

### 4.0 Offline, no GPU (can run while the GPU is busy; short, low priority `nice -n 19`)

| # | Question | Method | Cost | Unlocks / kills |
| --- | --- | --- | --- | --- |
| 4.0.1 | Would PARK_TRUNCATE have saved the ~162 s of cold 54-61K re-reads? Why were those conversations dropped? | read that session's park / evict log lines in `pancake-turns.jsonl` and the server log; replay the park decisions with a "keep the deepest prefix that fits" rule | 15 min | F1, F3 |
| 4.0.2 | Does opencode send sampled requests (temperature >= 0.7) or greedy ones? | log or inspect one opencode request body | 5 min | F6 (no gain if greedy) |
| 4.0.3 | Is there headroom for a cumulative draft stop on our traffic? | `tools/pancake/tail_replay.py` on recorded window traces | 15 min | F8 (port only if >= 2%) |
| 4.0.4 | How much would frequency-gated admission and prompt-fed tier counts change the hit rate? | cache_sim replay of routing traces | 15 min | F10, admission variants |
| 4.0.5 | How many CPU expert bytes would a sub-2-bit shadow save per token? | byte count from the GGUF tensor table x miss rate from traces | 10 min | F4 (requant only if >= 15%) |

### 4.1 Env-only A/Bs (no code; each answers a question on its own GPU)

| # | GPU | Arms | Unlocks / kills |
| --- | --- | --- | --- |
| 4.1.1 | both | `--prompt-cache-spread` on vs off on a multi-turn agent replay | F2 |
| 4.1.2 | both | F6 arms (`STRATA_SPEC_PROB=1 STRATA_SPEC_COUPLED=1 STRATA_SPEC_PROB_GATE=top`, DT 1.0 / 0.85), only if 4.0.2 says sampled | F6 |
| 4.1.3 | both | `--lookup-max-t 16`, `--lookup-chain 3` on code-edit turns | lookup settings |
| 4.1.4 | 7900 XT | `STRATA_SH_STREAM=1` (xyzzing: +4.0% on gfx1100; our preset sets 0 with no recorded A/B) | preset change |
| 4.1.5 | 7900 XT | `STRATA_PREFILL_AUTO_MAX=24576` at 32K (rxr2: +15% on a 64 GB box); `STRATA_WMMA_GEMM=1` prompt | preset change |
| 4.1.6 | 7900 XT | `GPU_MAX_HW_QUEUES` 4 / 6 / 8; `--aux-cpus auto` | preset change |
| 4.1.7 | 3060 (when installed) | `STRATA_EXPDEC` 1 vs 0, `STRATA_HC_Q8` 1 vs 0 (KhaledElzarw: EXPDEC -3.0%, HC_Q8 +0.9% and lossy on a 3090), `STRATA_PF_FUSED=1` (AdrianBM96: +5.2% prompt on a 3060), `--pcie-frac` 0 / auto / `STRATA_PCIE_BALANCE=1` | 3060 preset; F15 only if PCIe share > 0 wins |
| 4.1.8 | 7900 XT | quality check of `STRATA_PF_HCDOWN=1` (rxr2: doubled first-token KL at 16K for +1.3% prompt) | keep or drop in the preset |

### 4.2 Probes before writing a port (minutes each; decide yes/no)

| # | GPU | Probe | Decides |
| --- | --- | --- | --- |
| 4.2.1 | 7900 XT | unimatrix `empty_grid.cpp`: empty-launch cost vs block count on 42 WGPs; compare with the block counts in our decode graph | F20 |
| 4.2.2 | 7900 XT | per-kernel timing of `route_publish_res` and the top-10 (rocprofv3) | F9 (worth it if >= 2% of GPU time stays) |
| 4.2.3 | both | expert GEMV achieved bandwidth vs peak (IQ2_S / IQ2_XXS gate/up, Q2_0 down) | F12, F14, F16: little headroom above ~85% of peak |
| 4.2.4 | both | CPU pool: Q2_0 unpack cost per row, single thread, old vs imanu86 kernel (microbenchmark, no engine) | F7 |

### 4.3 What transfers between the GPUs

- Transfers (measure once, on either GPU): server and prompt-cache items (F1-F3), drafting and acceptance (F6, F8,
  lookup settings), routing and cache policy (F10, F13), CPU-side kernels (F7, F4). They do not depend on the GPU
  backend; re-check only the size of the gain.
- Must be re-measured on each GPU: every GPU kernel port (F9, F11, F12, F14, F16, F17, F20-F22). Launch cost
  (~14-16 us on gfx1100 vs ~3 us on sm_86), wave32 DPP vs warp intrinsics, WMMA vs mma/IMMA and the different
  VRAM bandwidth change the result. A CUDA win predicts a HIP win only for fusions that remove launches (worth more
  on the 7900 XT) and only after the probe in 4.2.3 shows headroom.

### 4.4 Ports, in order (after 4.0-4.2)

1. F1 PARK_TRUNCATE (+ F3 checkpoint cuts) - exact, both GPUs.
2. F7 Q2_0 AVX2 kernel - exact, both GPUs.
3. F9 router publish (HIP DPP first), F12 sign expansion with EXPDEC_V2 - exact.
4. F14 KhaledElzarw sm86 set on the 3060, then the Q2_0 down fold on gfx1100 - exact.
5. F8 SPEC_CUM (if 4.0.3 >= 2%), F10 ADAPT_PROMPT - exact.
6. F11 window decode attention (CUDA first, then WMMA), F17 / F19 Qwen3.6 attention - rounding-level, KL gate.
7. Lossy arms: F5 VERIFY_ECAP, F13 route-prior guards + stronger turbo vs uniform tail skip at equal KL, F4 shadow
   pack (after 4.0.5 and a requant that fits the disk).
