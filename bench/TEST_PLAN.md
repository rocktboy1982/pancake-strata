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
| 1. screen | every candidate, first | `--tier screen`: one A B pass, short + edit, 2 requests per case in the same server; promote if >= +2% on a case and no case below -3%, else drop. A spec with `"prefill": true` (arms that change prompt reading) also keeps its `long*` cases, one request each, judged on prompt tok/s with the same thresholds | ~2-3 min (+ ~20-60 s per long read) |
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
- Workloads: short chat, code edit (2K prompt); the read cases `long` (11.7K tokens), `long32k`, `long64k` (prompt tok/s) only
  for prefill items (spec key `"prefill": true`, then also in the screen) or the rigorous tier; 128K is not an abtest case.
  Every `long*` request starts with a unique `[run <nonce>-<n>]` line, so prompt reading is always cold (no prompt cache,
  in memory or on disk, can serve a repeat or another arm) and its text hash is not compared between arms; `long32k` /
  `long64k` are ~32000 / ~64000 tokens of `src/**/*.cpp` (deterministic; the engine's token count is printed per run and
  checked against the target) and request 32 tokens only.
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
| 4.0.3 | Is there headroom for a cumulative draft stop on our traffic? | `tools/pancake/trace/tail_replay.py` on a recorded window trace (capture note below) | 15 min | F8 (port only if >= 2%), A22 |
| 4.0.4 | How much would frequency-gated admission and prompt-fed tier counts change the hit rate? | cache_sim replay of routing traces | 15 min | F10, admission variants |
| 4.0.5 | How many CPU expert bytes would a sub-2-bit shadow save per token? | byte count from the GGUF tensor table x miss rate from traces | 10 min | F4 (requant only if >= 15%) |

**4.0.3 capture (one short GPU run, then offline).** The serve loop writes, per decode window, the draft confidences when
the window is built (`window_trace().drafts()` in `src/program/generate.cpp`, only with `STRATA_WINDOW_TRACE` set; the
default path only tests `window_trace().on()`): per draft row the drafter's probability of the token (NaN for lookup
rows) and its source; the window record has rows, accepted, ms; the layer records have every expert's placement, so the
CPU-missed experts each draft row adds (what A22 needs) are read back exactly. Top-1 margin / entropy are not recorded
(the drafter's kernels return only the chosen token's probability). Capture on a short run of each workload (short chat,
code edit), the drafter on (the preset's default MTP; not a lookup-only run):

    STRATA_WINDOW_TRACE=/tmp/w-chat.stwt ./run-pancake.sh iq2_xs fast          # serve; stop it with Ctrl-C after the request
    curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Write a detailed explanation of how a hash map works, with a Python example."}],"max_tokens":400,"temperature":0}' >/dev/null
    python3 tools/pancake/trace/wtrace.py /tmp/w-chat.stwt                      # summary: windows, acceptance, placement
    python3 tools/pancake/trace/tail_replay.py /tmp/w-chat.stwt --us-per-miss 23   # code edit: 60 (time_ceiling.py --overlap)

`tail_replay.py` prints the table of draft rows by cumulative confidence (share accepted, new CPU misses per row, whether
the row pays: A22) and tok/s per stop threshold theta (F8; port only if the best theta is >= +2%). `_EVERY=N` thins the
windows, `_MAX_MB` caps the file (default 2048 MiB; a few hundred windows are about 100-200 MiB at 48 layers).


### 4.1 Env-only A/Bs (no code; each answers a question on its own GPU)

First in line (from 4.0.1): screen `--conversation-cache-mib 2048 --conversation-cache-min-tokens 8192` against the
preset on an agent replay with interleaved short requests (watch free RAM); it replayed as 97% of the lost re-read time.

**Specs for this block (prepared 2026-11-10; base = the generated `pancake-flashnext-iq2_xs-fast-hip.json` at the main
checkout's root, the daily preset; one base + all arms of an area per spec).** Run from the main checkout, engine on
8080 stopped: `python tools/pancake/abtest.py bench/specs/<spec>.json --tier screen --out bench/results/<day>-<name>.jsonl`;
then `--tier confirm` on a copy of the spec trimmed to the promoted arms (plus base).

| Run order | Spec | Arms (besides base) | Cases | Row |
| --- | --- | --- | --- | --- |
| 1 | `11-conv-cache.json` | `cache2048`, `min8192`, `cache2048-min8192` | short, edit | first in line; side effects only, see below |
| 2 | `15-kmiss.json` | `kmiss28` (`STRATA_ADAPT_KMISS=2,8`) | short, edit | 2.2 / A5 |
| 3 | `12-spec-sampling.json` | `f6-dt100`, `f6-dt085` (SPEC_PROB + COUPLED + GATE=top, DT 1.0 / 0.85), run at T 1.0, top_p 0.95 | short, edit | 4.1.2 (4.0.2 answered: opencode is sampled) |
| 4 | `13-lookup.json` | `lookup-t16`, `lookup-chain3` | short, edit | 4.1.3 |
| 5 | `14-hip-env.json` | `sh-stream1`, `hwq4`, `hwq8`, `aux-auto` | short, edit | 4.1.4, 4.1.6 |
| 6 | `14b-hip-prefill.json` | `automax24576`, `wmma-gemm`, `hcdown0` | edit, long32k | 4.1.5, 4.1.8; `"prefill": true`: the screen keeps long32k (one cold ~32K read per arm, prompt tok/s), so the 32K claim is tested in the screen itself; promoted arms then `confirm --reps 2` |
| 7 | `16-3060-env.json` | `expdec0`, `hcq8-0`, `pf-fused1`, `pcie0`, `pcie-auto`, `pcie-balance` | short, edit, long | 4.1.7, only when the 3060 is back; regenerate its base file first (spec note) |

Notes: `12` needs `temperature` in the spec, a new optional abtest key (`temperature`, `top_p`, `seed`; absent =
greedy as before); sampled text differs per run, so its text-hash line is meaningless. `11` is exact and costs
nothing in decode, so its screen only proves "no side effect"; its real verdict comes from the agent replay below.
Lossy arms (`14b`, `15`, `16`) go to the rigorous tier if they promote.

**Agent replay for the cache arms (built 2026-11-10, `tools/pancake/replay_turns.py`; dry runs only so far).**
abtest's cases are independent one-shot requests, so they cannot park a conversation or evict it with a short side
request. `pancake-turns.jsonl` has no prompt text, only per-request sizes and timing (`time`, `prompt_total`, `reused`,
`output_tokens`, `max_new`, `reasoning_tokens`, `prompt_ms`), which is enough to rebuild the traffic shape. The replayer
takes one session (a time window, `sN` from `--list-sessions`, or a request id) and, per arm, starts a fresh server
(abtest's `start_server`, `quiet_gate`, `NoiseMonitor`, `make_config`; the run's engine log and turn log go to the out
directory, not the daily files) and sends the requests as real streamed chat requests in the recorded order:
- conversations: a request of >= 8,192 tokens extends the open conversation it can grow from (last prompt <= it <= last
  + output + 16K); a re-read with `reused == 0` stays in the SAME conversation (it is the cold re-read the arms should
  turn warm; the first proposal would have made each one a new conversation, which no cache can help). Smaller
  requests are side requests, one-message conversations; one with `reused > 0` repeats the text of the latest side
  request of the same size (thinking-off prompts count 2 tokens less), else new text.
- text: lines of the repo's C++ sources, a fixed start per conversation. Agent turn N+1 = turn N's messages + the reply
  the server really returned (reasoning + content, as opencode sends it back; the template keeps the thinking) + a
  `<tool_response>` user message sized so the prompt is the recorded `prompt_total` to within 2 tokens. Counts come from
  the server's own tokenizer and template imported locally (`tools/strata_tokenizer.py`, `serve/frontend.ChatTemplate`,
  `effort_position: end` rendering); each response's `usage.prompt_tokens` corrects any offset for the next request.
- `max_tokens` = the recorded output tokens (`--max-new recorded` for the recorded max_new; agent turns had 32768),
  thinking on/off as recorded, greedy, think time capped at `--max-gap` (2 s).
Per request: prompt tokens, reused (the engine's `prompt N tokens = R reused` line), prompt time, client time to first
token, decode tok/s, wall time and the engine's cache lines in that request's window (parked / skip parking (why) /
restored / evicted). Per arm run: cold re-reads (a non-first agent turn > 8K with reuse < 50%) and their seconds, total
prompt time, total turn time, peak RSS of the server tree, lowest MemAvailable, the auto cache budget line. Verdict vs
the first arm on total prompt time (paired 95% interval with `--reps` >= 2, else deltas). `--synthetic` replaces the log
with a built-in 4.0.1-shaped session (one conversation 40K -> 62K in 18 turns; turns 7-15 each after a group of side
requests: 69-token decodes x3, a 3,058-token probe x6, three ~4.1K probes, 75-token decodes, a thinking off/on pair).

```sh
# from the main checkout, engine on 8080 stopped, nothing else on the GPU (the run stops any Strata server first)
python tools/pancake/replay_turns.py bench/specs/11-conv-cache.json --out /tmp/rp --dry-run      # plan, sizes, duration; no server
python tools/pancake/replay_turns.py bench/specs/11-conv-cache.json --out bench/results/<day>-conv-cache-replay
python tools/pancake/replay_turns.py bench/specs/11-conv-cache.json --out ... --arms base,cache2048,min8192,cache2048-min8192
python tools/pancake/replay_turns.py bench/specs/11-conv-cache.json --out ... --synthetic        # without the log
```

Spec 11's `replay` block names the session (2026-11-09 16:13-16:43), the default pair (`base` vs `cache2048-min8192`)
and an extra `base1280` arm (`--conversation-cache-mib 1280`, the 11-09 auto budget): run it as well when today's base
prints a different auto budget (`conversation cache: N MiB / 4 slots (auto)` at the start of each arm), else base may
not reproduce the misses. Dry runs (2026-11-10, `nice -n 19`, < 1 s of sizing each): the log session is 75 requests
(34 agent turns: A 3 turns 70,082 -> 72,465, B 31 turns 34,261 -> 62,943 with the 7 recorded cold re-reads = 162.7 s;
41 side requests: 20 decodes, 21 probes, 9 repeats), 1.81M prompt tokens, every request built within 2 tokens of its
recorded size; expected ~13.5 min per arm as recorded (11.7 min of requests, 0.3 min of gaps, ~1.5 min start), ~8.7
min if every agent turn is warm; the pair ~27 min. The synthetic session is 48 requests (18 agent, 30 side), 0.98M
prompt tokens, 6.2 min per arm all warm to 9.4 min with every interleaved turn cold. One A B pass per arm is the
first step; `--reps 2` (A B B A) gives the interval.

| # | GPU | Arms | Unlocks / kills |
| --- | --- | --- | --- |
| 4.1.1 | both | `--prompt-cache-spread` on vs off on a multi-turn agent replay | F2 |
| 4.1.2 | both | F6 arms (`STRATA_SPEC_PROB=1 STRATA_SPEC_COUPLED=1 STRATA_SPEC_PROB_GATE=top`, DT 1.0 / 0.85), only if 4.0.2 says sampled | F6 |
| 4.1.3 | both | `--lookup-max-t 16`, `--lookup-chain 3` on code-edit turns | lookup settings |
| 4.1.4 | 7900 XT | `STRATA_SH_STREAM=1` (xyzzing: +4.0% on gfx1100; our preset sets 0 with no recorded A/B) | preset change |
| 4.1.5 | 7900 XT | `STRATA_PREFILL_AUTO_MAX=24576` at 32K (rxr2: +15% on a 64 GB box); `STRATA_WMMA_GEMM=1` prompt; case `long32k` (cold ~32K read, prompt tok/s), spec 14b | preset change |
| 4.1.6 | 7900 XT | `GPU_MAX_HW_QUEUES` 4 / 6 / 8; `--aux-cpus auto` | preset change |
| 4.1.7 | 3060 (when installed) | `STRATA_EXPDEC` 1 vs 0, `STRATA_HC_Q8` 1 vs 0 (KhaledElzarw: EXPDEC -3.0%, HC_Q8 +0.9% and lossy on a 3090), `STRATA_PF_FUSED=1` (AdrianBM96: +5.2% prompt on a 3060), `--pcie-frac` 0 / auto / `STRATA_PCIE_BALANCE=1` | 3060 preset; F15 only if PCIe share > 0 wins |
| 4.1.8 | 7900 XT | quality check of `STRATA_PF_HCDOWN=1` (rxr2: doubled first-token KL at 16K for +1.3% prompt) | keep or drop in the preset |

### 4.2 Probes before writing a port (minutes each; decide yes/no)

| # | GPU | Probe | Decides |
| --- | --- | --- | --- |
| 4.2.1 | 7900 XT | unimatrix `empty_grid.cpp`: empty-launch cost vs block count on 42 WGPs; compare with the block counts in our decode graph | F20 |
| 4.2.2 | 7900 XT | per-kernel timing of `route_publish_res` and the top-10 (rocprofv3). Note: with deferral in the preset the path is `route_multi` + `doorbell_publish` (ROUTE_PUBLISH is off under deferral), so time those two and the gap between them. F9 is coded (`work/f9-route-fast`): run `native_multi_parity` (the ROUTE_FAST lines must say bitwise), then `bench/specs/17-f9-route-fast.json` (doorbell-store isolates the PCIe read, route-fast is the whole change; text identical to base) | F9 (worth it if >= 2% of GPU time stays) |
| 4.2.3 | both | expert GEMV achieved bandwidth vs peak (IQ2_S / IQ2_XXS gate/up, Q2_0 down) | F12, F14, F16: little headroom above ~85% of peak |
| 4.2.4 | both | CPU pool: Q2_0 unpack cost per row, single thread, old vs imanu86 kernel (microbenchmark, no engine) | F7 |

Probes (all in `tools/pancake/probes/`; each file's header says question, build, run, how to read the result; written
offline on 2026-11-10, the Python parsers and the CPU probe were smoke-tested on synthetic input, the GPU probe only
syntax-checked). Run them from the repo root with the GPU idle, one at a time; a trace needs the engine under
`rocprofv3` (it decodes ~1.6x slower, read shares and counts, not tok/s).

- 4.2.1 empty-grid launch cost (7900 XT, ~2 min):
  `hipcc -O2 --offload-arch=gfx1100 tools/pancake/probes/empty_grid.hip.cpp -o /tmp/empty_grid && HIP_VISIBLE_DEVICES=0 /tmp/empty_grid --csv /tmp/sweep.csv`
  (or `cmake -DSTRATA_BUILD_PROBES=ON` on the HIP build, target `empty_grid_probe`; `--step 8` for a 10 s look).
  Then the decode graph's shapes from a trace (step 4.2.2 makes it): `python3 tools/pancake/probes/launch_shapes.py /tmp/prof/run_kernel_trace.csv --sweep /tmp/sweep.csv`
  (`--static` lists literal launch sites without a trace). Close F20 when the "EXCESS" line is ~0.
- 4.2.2 router publish and top-10 (7900 XT, one profiled request):
  `python3 tools/pancake/gpu_profile.py configs/pancake/flashnext-iq2_xs-fast-hip.json /tmp/prof --case short`
  then `python3 tools/pancake/probes/kernel_share.py /tmp/prof/run_kernel_trace.csv` (the `TARGET` line is
  route_publish_res: mean us, share of kernel and wall time, gap after it). F9 only if >= 2% of wall time.
- 4.2.3 expert GEMV bandwidth (both GPUs, same trace as 4.2.2; 3060: `nsys stats --report cuda_gpu_kern_trace --format csv`):
  `python3 tools/pancake/probes/expert_bw.py /tmp/prof/run_kernel_trace.csv --gpu 7900xt --hit-rate 0.6 --floor-us 3`
  (`--gpu 3060` for the 3060; bytes per expert come from `bench/results/2026-11-10-offline/gguf_table.json`). The trace does not
  say how many experts a launch computes, so the table gives GB/s for K = 1..10 experts per launch and the K needed for
  85% of peak; take the real K from the cache hit rate (10 x hit rate).
- 4.2.4 CPU Q2_0 row kernel, legacy vs `STRATA_Q2_FAST` (5700G, ~20 s, no GPU, no engine; do not run during a measurement):
  `cmake -S . -B build-probes -DSTRATA_BUILD_PROBES=ON && cmake --build build-probes --target q2_rows_bench && taskset -c 3 build-probes/q2_rows_bench`
  (or the four-line `g++` recipe in the file header). It prints ns/row for legacy, fast, the dispatcher and a read-only floor,
  in L2 and in DRAM, and checks the outputs are bitwise equal. First numbers on the 5700G (Zen 3, no VNNI, 0.08 s runs, one
  core): down rows 30.5 -> 22.8 ns (x1.33), gate/up rows 106 -> 86 ns (x1.23), both compute-bound (DRAM figures within ~6%
  of the L2 ones; a read-only pass is 8-32 ns), outputs bitwise equal.

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
3. F9 router publish (HIP DPP first), F12 sign expansion with EXPDEC_V2 - exact. F12 is coded alone on
   `work/f12-sign-expand` (EXPDEC_V2 is not in the tree): run `iq_multi_parity`, then
   `expdec_bench 300 -1 20 16/42,17/42,22/42 def,dec:s0,dec:s1` on each GPU (every cell `identical`; keep a format only
   where `dec:s1` has the lower gu_us), then `bench/specs/f12-sign-fast.json` (text identical to base on every case).
4. F14 KhaledElzarw sm86 set on the 3060, then the Q2_0 down fold on gfx1100 - exact.
5. F8 SPEC_CUM (if 4.0.3 >= 2%), F10 ADAPT_PROMPT - exact.
6. F11 window decode attention (CUDA first, then WMMA), F17 / F19 Qwen3.6 attention - rounding-level, KL gate.
7. Lossy arms: F5 VERIFY_ECAP, F13 route-prior guards + stronger turbo vs uniform tail skip at equal KL, F4 shadow
   pack (after 4.0.5 and a requant that fits the disk).
   F5 is coded (work/f5-verify-ecap): bench/specs/11-verify-ecap.json, screen first, then token forcing (decode path)
   and the task gate (tool_call tasks included) on the same spec; no default before all three pass.
