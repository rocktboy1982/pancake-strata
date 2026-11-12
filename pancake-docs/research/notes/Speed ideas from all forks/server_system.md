# Speed ideas outside the GPU kernels: server/agent layer, scheduling, sync, system tuning (all Strata forks, Sep-Nov 2026)

State as of 2026-11-10. Method: `data/fork-kb/forks.sqlite`, `commits` with `upstream_history = 0` and pancake's own
repo left out: 43,685 rows, 21,944 unique SHAs (2026-09-24 .. 2026-11-10). Regex buckets over the messages
(prefix/park/TTFT 1,929, scheduling/slots 2,193, sync 943, threads 281, OS 320, build 275, allocator 118, server
512), deduplicated by subject, with subjects already in pancake's `git log --all` removed; about 2,000 titles read and
40 commits plus one fork doc read in full through `gh api` (44 API calls). READMEs: 4,531 in `readmes`, scanned for
tok/s tables. Pancake status checked against the clone (`src/`, `serve/`, `configs/pancake/flashnext-iq2_xs-fast-hip.json`,
README section 6, `git cat-file` for upstream SHAs). "COV" = already in `reports/Forkuri Strata câștig viteză.md` /
`reports/Forkuri Strata active noiembrie.md` or their notes; those get a mention only.

**Every speed number below was measured by the fork's author on their own hardware and is unverified for pancake**,
unless it says "pancake measured (turn log)". The pancake turn-log numbers come from `pancake-turns.jsonl`
(622 turns, 2026-11-09..10, Flash-Next IQ2_XS, read-only analysis, nothing was run). No build, benchmark or engine run
was done (the GPU was busy in another session).

## Q1. Agent-turn TTFT: prompt/prefix cache reuse, parking, rewind

### Takeaway
The cache-side forks give the biggest end-to-end wins for agent work, and pancake already has most of the basics
(upstream borrow, park-gain, message/tool-boundary checkpoints, auto-pin, disk tier). What it lacks: (a) the Lamina
"fork-point" checkpoint, (b) a no-park rule when a request rewinds its own tail (PARK_REWIND_MAX / skip-tail),
(c) partial parking when a whole chain does not fit (PARK_TRUNCATE), (d) `--prompt-cache-spread` (present but off in the
preset). Our own turn log shows warm turns are cheap (about 0.23 s + 0.91 ms/token) but one agent session had **7
consecutive cold re-reads of a 54-61K conversation, 21-25 s each (~162 s)**: whatever dropped that reuse is worth more
than any kernel change at that point.

### Cited Findings
- **Lamina fork-point checkpoints** (`ab808b133e`, 2026-10-01): two extra checkpoints, at least 256 cells past the reused
  prefix: where the prompt leaves the previous request's, and 4 cells before the last message's `<|im_end|>`. RTX 3060,
  IQ3_S: a 6.1K prompt asked 3 different questions 10.4 s / 9.8 s / **0.5 s**; a transcript growing inside one user
  message 10.9 s then 0.8-0.9 s a turn; 6 agent-benchmark tasks 412 s -> 369 s (-10%). Exact (only where the cut falls).
  Pancake: has `STRATA_CACHE_MESSAGE_BOUNDARY` (only for prefixes of 8,192+ tokens, `serve/server.py:2802`) and
  `STRATA_CACHE_TOOL_BOUNDARY`; no "where the prompt leaves the previous request" cut was found. —
  [eddoursul/Lamina ab808b133e](https://github.com/eddoursul/Lamina/commit/ab808b133e0dfba74bddc9742296f14e7ccb30b7)
- **Lamina borrow from a stashed conversation before its newest checkpoint** (`83f40aee62`, 2026-10-18), an extension of
  upstream `c9e571e0` (which pancake has): 3 subagents after a ~4,000-token system prompt, the 2nd's turn after the 3rd's
  first request cached 4,021 -> 12,372 of 12,975 tokens (IQ3_S 3.00 -> 0.67 s; UD-Q4_K_XL 2.98 -> 0.95 s); all requests
  44,678 -> 36,327 cells read; answers identical. Exact. —
  [eddoursul/Lamina 83f40aee62](https://github.com/eddoursul/Lamina/commit/83f40aee62128e6031ae0c18ae5cd220d8d397f7)
- **Spread checkpoints, up to 16 per conversation** (`13d90a38ec`, 2026-11-03): with only 6 checkpoints by LRU, editing a
  message 5 turns back re-read the whole 43.8K conversation (9.4 s vs 2.2 s); cache_spread_test 130,206 -> 86,218 cells
  re-read. Defaults `--conversation-cache-mib 16384`, no slot limit (4 slots let a 5th parked conversation evict one the
  budget still held). Exact. — [eddoursul/Lamina 13d90a38ec](https://github.com/eddoursul/Lamina/commit/13d90a38ec9a8e443af21e458833b79ef1cb0221)
- **`--prompt-cache-spread` measured**: an edited 82K conversation resumed from 73,610 tokens in **10.2 s instead of
  40.0 s**, same answer (2x Turing). Pancake: the flag exists (upstream `a7d4b754`, `generate.cpp:964`, default off) and
  is **not** in `flashnext-iq2_xs-fast-hip.json`. — [xrip/Strata 56017b1c41](https://github.com/xrip/Strata/commit/56017b1c414cb6f66e271741a4cbb97a32cdb31a)
- **`STRATA_PARK_REWIND_MAX=N`** (`8e4ccbe90d`, 2026-10-12; re-landed by 0xluk `53cecdeee9`, COV): when the client re-sends
  the last reply without its thinking, every turn parked the whole live conversation (1.4-1.6 GB, 280-370 ms at 52K) and
  dropped the copy one request later. Skip the park when the request resumes from the session's own checkpoint and leaves
  <= N tokens behind. i7-13700K + RTX 5090: 12 appended turns on a 52K conversation 10.4 s -> 6.3 s of prompt reading. Exact.
  Pancake: not present (`grep PARK_REWIND_MAX` = 0). — [Xiaomi2008/Strata 8e4ccbe90d](https://github.com/Xiaomi2008/Strata/commit/8e4ccbe90d2e8dbab5c8ddf7f19d2c8159dc33d8)
- **`--conversation-cache-skip-tail N`** (same idea as a flag): gfx1101 + Flash-Next Q2_0, 7 requests with a 4.5K system
  prompt and a client that drops reasoning: 5 parks / 3.36 GB written -> 1 park / 0.67 GB; the skipped turns' prompt stage
  505-693 ms -> 272-338 ms; in a real deployment log 10 of 39 parks (14.5 of 59.2 GB) were this case. Exact. Pancake: absent.
  — [UltimateBoomer/Strata 5d0a9e786c](https://github.com/UltimateBoomer/Strata/commit/5d0a9e786c5d275b3f1f8bb1a48e95a20094b918);
  same fork "park only conversations of 4096 tokens or more" — [9419b689a6](https://github.com/UltimateBoomer/Strata/commit/9419b689a6d1b4ccc92e4e394ab17a0b3e20564b)
- **`--conversation-cache-park-gain N`**: 1,418 parks -> 1, 2,251.9 GB -> 3.26 GB host copies on one long conversation
  (RTX 5090 Laptop), decode -3.4% inside noise. Pancake: **has it** (`generate.cpp:949`, upstream). —
  [DamirLukina88 58ca3f2929](https://github.com/DamirLukina88/D.L.-AI---Strata/commit/58ca3f2929b4e357ce2867b8981a246fdb881ac5).
  Counter-measurement: on a 45K conversation the park costs 351-639 ms **per conversation switch, not per turn** (1-2% of
  a 6-turn run) and gain 4096 did not skip it — [atmJava ca914558db](https://github.com/atmJava/Strata/commit/ca914558dba0ac3f8adff4d6b50ec6eb3630fe19)
- **`STRATA_PARK_TRUNCATE=1`** (`5ef4fc8b4b`): parking refused the whole snapshot when the chain did not fit the byte
  budget or the RAM floor, so the next request re-read from token 0 (225 s for a 160K chain). Now it parks the deepest
  checkpoint prefix that fits and re-reads only the tail; an already-parked entry asked to make room shrinks to one of its
  checkpoints before being dropped. Single-stage only (pancake is single-stage). Exact. Pancake: absent. —
  [TwoAnts/Strata 5ef4fc8b4b](https://github.com/TwoAnts/Strata/commit/5ef4fc8b4b8a97848371ebf76fdcd5e46f710b33)
- **Automatic hash-chained prefix cache** (vLLM-style block hashing in the engine's K/V pages; fnv64 chain, tokens
  verified on hit, refcount pins, LRU eviction; no request key): a second agent on a 13,807-token document mapped 3,444
  cached pages and read 30 tokens, same answer as a fresh read (2x RTX 3090). Exact. Requires wesraph's `--kv-share`
  paged pool (not in pancake); pancake covers the main case (shared system+tools prefix) with `STRATA_AUTO_PIN`. —
  [wesraph/Strata 17abc917c1](https://github.com/wesraph/Strata/commit/17abc917c1308bca6565e77d580675e1df7f3f68)
- **Where the time of a warm re-read goes**: a 461-token re-read = 977 ms `Prefill::run` + 74 ms expert refill + 42 ms first
  window + 10 ms checkpoints; fit 711 ms + 0.728 ms/token (~1,370 tok/s); client TTFT exceeds the engine read by only
  0.05-0.10 s; draft acceptance ~99% within a conversation vs 75% right after a cold read (decode 108-134 vs 66-67 tok/s). —
  [atmJava e2f6600149](https://github.com/atmJava/Strata/commit/e2f660014973fc41cae6e636a7215b1a0086d198),
  [atmJava dcae960123](https://github.com/atmJava/Strata/commit/dcae960123485b22de35ca33b0886a96e6ccda24)
- **Restore speed as the next target**: turn 2 at 80% reuse took 1,586 ms where 469 new tokens are worth ~754 ms; ~830 ms
  went to restoring 260 MB of parked K/V + checkpoint (~300 MB/s, far below PCIe). —
  [phoenixclyde 7d32d9b791](https://github.com/phoenixclyde/Strata/commit/7d32d9b791633f7aa0c9aee62673d36d34dd4db4);
  trace of a 148 ms gap between prefix search and borrow decision — [phoenixclyde 7f6471f8e4](https://github.com/phoenixclyde/Strata/commit/7f6471f8e4138fe15827a649c29ad642b6789f72)
- **Short prompts are the slow ones** per token on small rigs (2.7-3.3 s to first token) —
  [RandoTeam 36fe80faf9](https://github.com/RandoTeam/4gbVramStrata/commit/36fe80faf9b789ffbad403718808e979f1351d93);
  idle conversation park + restore from RAM in 0.56 s for 61K tokens (`STRATA_ELASTIC_IDLE`, 5 min timeout) —
  [ksanislo b6fe7f7d6b](https://github.com/ksanislo/Strata/commit/b6fe7f7d6ba5cb97ccb9ab57322feb724fccd970)
- **strata-poly (Qwen3.6 path) equivalent**: the re-rendered chat template never matched the sampled tokens, so every turn
  re-read the whole prompt (285 s at 45K). `STRATA_Q36_SNAP=N` keeps the GDN state (63 MiB a copy) at N points, last 8
  tokens in their own chunk: TTFT 11.98/1.10/1.06 s (was 11.75/5.97/6.24); 45K: 285.66 -> 2.37 s (GTX 1660 Ti). Exact. —
  [Aaron-xx/Strata 986649d5c6](https://github.com/Aaron-xx/Strata/commit/986649d5c6b1bf6bc0874bfd7bc2d9a7d859fab1)
- **Pancake measured (turn log, 622 turns)**: warm turns (reuse > 0, < 8K read, n=179) fit `prompt_ms = 226 + 0.912 x
  read` (marginal ~1,100 tok/s); < 200 tokens read: median 167 ms; 200-500: 949 ms; 500-1000: 1,127 ms. Engine first token
  minus prompt_ms: median 19 ms. Agent-like turns (max_new >= 8192, n=34): 69.6% of prompt tokens reused, but 7 consecutive
  turns at 54,313-61,565 tokens had `reused = 0` and 21.2-25.0 s prompt each (~162 s), between warm turns at 53K and 62K
  that restored fine. Log park times: 50,876 tokens 462 ms, 55,424 tokens 707 ms; restore of 2,066 tokens 39 ms.

### Inferences
- The 7 cold 54-61K turns match the TwoAnts symptom ("park refused whole when the chain does not fit, next request reads
  from 0"): the log reports `conversation cache sized from the free RAM: 4864 MiB for up to 4 parked conversations`, and a
  55K park is ~700 ms / several GB with GDN checkpoints. Unverified; first step is a grep of that session's log lines for
  refused/evicted parks. If confirmed: `STRATA_PARK_TRUNCATE` port + `STRATA_CONV_BUDGET=ram`/`STRATA_CONV_KEEP_GIB`
  (pancake item 86) are the fix; gain = 20+ s per affected turn.
- Cheapest arm today: add `--prompt-cache-spread` to the Flash-Next presets (flag-only, exact). Expected gain only on
  edited/rewound history (opencode compaction, "undo"), where xrip/Lamina saw 4x shorter re-reads.
- PARK_REWIND_MAX / skip-tail: small port (~25-90 lines in `generate.cpp`). Gain depends on whether opencode re-sends
  reasoning (README says 1.19.5 does, then the rewind case is rare). Check the log for `dropped 1 superseded copy`
  right after each park: our log shows that pattern after a 582-token park, so some turns pay ~110-700 ms for nothing.
- Lamina fork-point cut: medium port; helps parallel sub-agents that share a long context with different tails
  (opencode task tool). Estimated -5..-10% wall on multi-call agent tasks (Lamina's 412 -> 369 s), unverified.
- Our warm-turn fixed cost (~0.23 s) is already 3x lower than atmJava's (0.71 s), so "fixed cost of a small read" is
  not a top target here; restore speed (39 ms for 2K tokens) is fine too.

### Gaps
- Root cause of the 7 cold 54-61K turns not established (log lines for that session not isolated; GPU session busy).
- No fork measured the fork-point cut or spread on an opencode trace; numbers are from synthetic agent benchmarks.
- strata-poly's own re-tokenization behaviour (does pancake's poly path hit Aaron-xx's mismatch?) not checked.

## Q2. Request scheduling, parallel slots, priority

### Takeaway
On every rig that measured it, concurrency is where the aggregate throughput is (2-2.4x at 3-8 streams), but pancake's
presets run one request at a time. For opencode with parallel sub-agents, `--batch N` slots are an untested lever; the
cost is VRAM per slot (20 GB card) and a slower prompt read for the reader while others decode.

### Cited Findings
- 2x RX 7900 XTX, `"parallel": 8`: **70 / 105 / 150 tok/s aggregate at 2 / 4 / 8 requests** (server-side balanced slots,
  no pad rows). — [unimatrix099 fa10a01bd3](https://github.com/unimatrix099/Strata/commit/fa10a01bd3bddaa0684acbca43e3a72b86ca82a9)
- "Three streams reach 136 tok/s where one reaches 63: the headroom is concurrency" (config batch 4, parallel 4). —
  [atmJava 5715656718](https://github.com/atmJava/Strata/commit/5715656718b4ae76749ef09e438803b9efa4adb0)
- `STRATA_PIPED_CONCURRENT` (concurrent prompt read beside decoding slots): a cold 22,700-token read 11.8 -> 14.35 s
  (+21.6%) but the decode stream beside it went from 2.0 chars/s and an 11.64 s worst gap to 47.0 chars/s and 0.70 s;
  called "the largest single win on record" for that deployment. — [atmJava a76cc510cf](https://github.com/atmJava/Strata/commit/a76cc510cf90ac307ea87b69cb5fe3a425242157),
  [atmJava ca914558db](https://github.com/atmJava/Strata/commit/ca914558dba0ac3f8adff4d6b50ec6eb3630fe19); on 2x RTX 3090 the same
  path hangs (a kernel that never ends, also on upstream) — [wesraph e48976a69e](https://github.com/wesraph/Strata/commit/e48976a69e817d2ca475ccf73ecf7ed6f0d840b3);
  staged read: long-read TTFT 47.3 -> 17.2 s, short-read first token 46.2 -> 1.7 s under load — [atmJava e107e1d12f](https://github.com/atmJava/Strata/commit/e107e1d12ffa522cb221c213d8dab4abc582bdb9)
- The server's queue is the slot reservation (`pick_slot` returns None when all are busy); queueing cost 36-43 s per new
  session at full occupancy. — [atmJava ba5011c94b](https://github.com/atmJava/Strata/commit/ba5011c94bf9acd67cd03c1d5b0894f250953229)
- Slot keep across a thinking-budget wrap-up pass (`SLOT_KEEP_S = 5 s`): without it, a 6-agent soak re-read 11.5K/15K
  conversations from 0 in another slot (60 and 160 s without a token). — [ksanislo 85bf7803cf](https://github.com/ksanislo/Strata/commit/85bf7803cf200cf2928623a25fe0b6f7ced80e7a)
- Pipelined per-slot MTP drafts: two proposals per slot 85 / 72 tok/s aggregate at two / three agents (2x RTX 3090);
  the README table shows later-turn first token 1.7 -> 1.4-1.6 s (1 agent), 2.73 -> 1.55 s median (3 agents replay). —
  [wesraph c5675fd26a](https://github.com/wesraph/Strata/commit/c5675fd26af83e2313a03f57e0dea1843927191f), [wesraph README](https://github.com/wesraph/Strata)
- Priority/VIP preemption: background generation parked at a safe boundary into a host capsule and resumed byte-exactly;
  VIP first answer 1.43 s mid-generation vs 313 s FIFO (single RX 7900 XTX). — [evrenesat 47f4449c3d](https://github.com/evrenesat/Strata/commit/47f4449c3dfed979a750f647506160fd9181a133)
- Batch prefill yields to active decode slots at mid-chunk layer boundaries (`STRATA_BATCH_PREFILL_LAYER_YIELD`) —
  [bsorensen110 5bec6f8472](https://github.com/bsorensen110/Strata/commit/5bec6f84722eddfe04fb46150a2161c35962e744)

### Inferences
- Pancake inherits upstream `--batch` (server `batch = 0`, `serve/server.py:1255`), so a `--batch 2` arm is config-only.
  Expected: aggregate tok/s up when opencode runs sub-agents in parallel; single-stream decode likely a bit lower and the
  KV per slot competes with `--kv-resident 32768` and the expert cache on 20 GB. Unmeasured; needs a 2-agent replay.
- ksanislo's slot keep matters only with `--batch`; port with it if batch is adopted (pancake's `reasoning_budget_grace`
  also ends a pass and sends a next one).

### Gaps
- No single-GPU 20 GB measurement of `--batch` on Flash-Next; all concurrency numbers are 2-GPU or 24 GB+ cards.

## Q3. CPU-GPU sync points, host overhead, thread placement, pool parking

### Takeaway
Pancake already has the important pieces (spin-then-sleep pool, host pinning, `--aux-cpus`, MWAITX, DEVICE_SPIN,
MTP stream priority). Two measured items fit our topology: `--aux-cpus auto` (Ben1078's STRATA_HELPER_CPUS twin,
+3.8% on an 8C/16T 7800X3D) and keeping the verify window's final wait a blocking sync (a poll cost ~3%). Several
"obvious" ideas were measured negative by forks: SMT siblings for the pool (-20%), splitting pools across CCDs, futex
parking (-14%), spin-wait device flags on AMD (0%), next-layer expert prefetch on a 7900 XTX (-3-4%).

### Cited Findings
- **Helper threads off the host's CPU**: the host thread is pinned to one CPU and every later thread (32 Stager threads,
  adapt job thread, router lookahead, file-tier I/O) inherited that one-CPU mask: 16.1 M involuntary context switches on the
  host. `STRATA_HELPER_CPUS=auto|list`: decode 47.5 -> 49.3 tok/s (+3.8%, 7800X3D 8C/16T), GPU wait for the host's plan
  1.52 -> 0.51 ms per window. Exact (scheduling only). Pancake: has the upstream equivalent `--aux-cpus off|auto|LIST`
  (`generate.cpp:1290`), default **off**, not in the preset; svgop tested an aux-cpus arm and reverted it (32-thread box,
  "honest negative", COV). — [Ben1078/Strata 5ffd2ccaec](https://github.com/Ben1078/Strata/commit/5ffd2ccaecf4265e246c1ec19d464bdd2ecc2d5c)
- **Blocking sync beats polling** at the verify window's end: a `cudaStreamQuery` poll cost IQ3_S ~3% decode, reverted. —
  [anon761 0d6efec071](https://github.com/anon761/Strata-upstream/commit/0d6efec071159372d6e2127e4130fa85ebe70291)
- **Pool park protocol**: pure futex parking -14% decode, 64-256 us spin -13%, 512 us -5%, 1,024 us -1.2%, >= 2,048 us parity;
  root cause: a core futex-parked across a gap drops to the idle P-state (1.2 GHz vs 2.9 GHz). Idle CPU 24 -> 0 cores at
  parity with 2 ms spin (V100 box, E5-2680 v4). — [noorazman 95a338efe2](https://github.com/noorazman/Strata/commit/95a338efe25ebbf4f3e6c5420ae167fa90e343b8).
  Pancake: spins `kSpinBeforeSleep = 20 ms` before sleeping (`pool.hpp:292`), plus MWAITX opt-in; no change needed.
- **Thread placement negatives**: pool workers restricted to one CCD -13/-14/-20% — [svgop e2acff0c6d](https://github.com/svgop/Strata/commit/e2acff0c6d2d17d0dcc929f6920c35dce34524bb);
  SMT siblings as lane threads -20% (gate+up 0.153 -> 0.308 ms/entry) — [svgop 5e4207bcdd](https://github.com/svgop/Strata/commit/5e4207bcddd4bb983716d8b5a1e38e1373a16be7) (COV).
  Same commit: `DS4_SETTLE_ROUNDS=N`, the post-prompt cache settle drains under the first decode tokens instead of blocking
  TTFT up to 6 s (DeepSeek path).
- **Idle pools spinning on another pool's cores** (layer split): `ExpertPool::nap()` + `_mm_pause` in service loops;
  Q4 one conversation 30.9 -> 32.8 tok/s. Multi-stage only. — [ksanislo 89527ee6fa](https://github.com/ksanislo/Strata/commit/89527ee6fa483acf0ad8b29cacbabd7299774de6);
  service threads that stop spinning after 200 ms idle (power, not speed) — [yqYo1 7c47fe0abf](https://github.com/yqYo1/Strata/commit/7c47fe0abffa64df97532cf0047d40418ac7dc88)
- **Guided self-scheduling** `STRATA_POOL_GUIDED=1`: phase wall 169 -> 135 us, verify window -7%/-3.4% on a hybrid Core
  Ultra 7 255HX (tail from slow E-cores). COV (already in the earlier report). —
  [zsariboga fc5b2a7829](https://github.com/zsariboga/Strata/commit/fc5b2a7829e52052ee9ea716554ca757612d66c8)
- **AMD-specific host findings (7900 XTX, ROCm 10.2)**: a dependent kernel in a HIP graph costs 2.8 us (2.2 us without the
  queue acquire fence, no HIP switch for it); `GPU_MAX_HW_QUEUES=8` made every prompt 10-20% slower; next-layer prefetch
  (RouterLookahead / `STRATA_FS_SLOTS`) 3-4% slower (prediction covers 31-37% of misses; each 1.9 MB copy adds ~14 us to
  the GPU path); `STRATA_PCIE_USAGE=1` + `STRATA_PCIE_KEEP=96` (PCIe-fetched blobs kept in a 96-slot ring, adaptive
  swaps copied inside VRAM) cut the per-window swap wait 1.2-1.7 -> 0.2-0.5 ms, decode 106.15 -> 110.48 tok/s (+4.1%). —
  [android-imdad docs/RX7900XTX.md](https://github.com/android-imdad/Strata-7900xtx/blob/HEAD/docs/RX7900XTX.md),
  [android-imdad 570854aded](https://github.com/android-imdad/Strata-7900xtx/commit/570854aded26742f7ddcc676357e5a19356d4c85)
- **MTP chain stream priority on ROCm**: with default priority the chain shares a HW queue with stage-1 windows; 2x 7900
  XTX half the starts at 81-82 instead of 89-91 tok/s; with priority 6/6 starts at 89.3-90.9; `GPU_MAX_HW_QUEUES=8` had
  the same effect. Pancake: has `STRATA_MTP_HIP_PRIO`. — [unimatrix099 3e1b02cfda](https://github.com/unimatrix099/Strata/commit/3e1b02cfda0056ab292848466481f8a70a248302)
- `hipDeviceScheduleSpin | hipDeviceMapHost` on AMD: no change (53.8 vs 53.7 tok/s). Consistent with pancake's
  `STRATA_DEVICE_SPIN` arm. — [unimatrix099 7f9d8f0845](https://github.com/unimatrix099/Strata/commit/7f9d8f0845ace09dbc042a0de2bf6bbba7408b16)
- Async routing-id D2H (`STRATA_MOE_ASYNC_IDS`, pinned 80 KB copy + event instead of full-stream sync), MTP draft K/V as one
  graph, two-stream DMA: all TTFT-neutral on V100 (<= 0.5 s on 40 s). TTFT at 32K = 71.1 s prefill + 2.3 s HTTP/Python
  tokenize + 162 ms refill + 25 ms first window. — [noorazman 1d9dbe70f1](https://github.com/noorazman/Strata/commit/1d9dbe70f12d8f6254ce718861e7fea92e420913),
  [noorazman 25fbc76161](https://github.com/noorazman/Strata/commit/25fbc7616129b17ccdf2c23463a2a2302ef875af)
- `--no-graphs` +29% / +18% on two small-model engines (COV) — [RandoTeam 4a9539f2e5](https://github.com/RandoTeam/4gbVramStrata/commit/4a9539f2e56b7246ec04d9d5e95f482915192f1a)

### Inferences
- `--aux-cpus auto` on the 5700G (8C/16T, one CCX, the same shape as Ben1078's 7800X3D): flag-only arm, exact; 0-4%
  decode expected. auto uses the host core's SMT sibling and cores without workers; with 7 workers + host on 8 cores only
  the siblings are left, which may contend (svgop's SMT -20% was for pool workers, not helpers).
- Pancake's `STRATA_COMBINE_POLL1` is a poll at a sync point; anon761's -3% from polling is a reason to keep it measured
  per build rather than on by default.
- Pancake preset sets `GPU_MAX_HW_QUEUES=6`; android-imdad saw 8 slow every prompt by 10-20% on gfx1100, unimatrix099 saw 8
  help the MTP chain. A 4/6/8 sweep on prompt and decode is cheap and worth one run.
- `STRATA_PCIE_USAGE`/`PCIE_KEEP` only applies with PCIe fetches on; pancake's preset runs `--pcie-frac 0`, so it is a
  joint arm (pcie-frac 0.35 + keep ring) rather than a drop-in.

### Gaps
- None of the forks profiled HIP host overhead per window on a 5700G-class CPU; whether the host plan wait (0.5-1.5 ms per
  window elsewhere) is visible in pancake's window trace was not checked here.

## Q4. OS/kernel tuning, ROCm/CUDA versions, build flags, allocator

### Takeaway
Huge pages and page-cache control were the busiest system topic; pancake already covers compaction and offers hugetlb.
New and relevant: the THP `defrag=madvise` trap (synchronous compaction in faults; this box is set to `madvise`),
ROCm version as a ±40% prompt lever on gfx1100, and a server-side warm-up replay. **No fork measured LTO, PGO, `-march`
or a different malloc**, so those stay open questions.

### Cited Findings
- **THP defrag trap**: with `transparent_hugepage/defrag` = always/madvise/defer+madvise, a huge-page fault in an advised
  region compacts memory inside the fault; on a host full of page cache a conversation park took 4-14 s (3,968 ms park of
  89K tokens: 12,756 faults, 17 compaction stalls, 8,386 pages of direct reclaim; 9 of 30 parks slow, 71 s total, each
  freezing every slot). Fix: advise only when defrag is `defer` or `never`; `STRATA_CONV_HUGEPAGE=0|1` override. —
  [wesraph 089fe83a88](https://github.com/wesraph/Strata/commit/089fe83a88a195344e655474ffc3912a7f190022).
  This box: `defrag = madvise`, `enabled = madvise`. Pancake's `include/strata/core/conversation_buffer.hpp` has no huge-page
  advice (grep), so the park path is not exposed; the expert arena's MADV_HUGEPAGE is (handled by `os_tune.sh` +
  upstream `6b395578a` gate).
- **Arena THP gate on gfx1100**: never fired in 15/15 production sessions (reads buddyinfo at arena setup only); the forced
  4 KB branch costs +8 s (1K) / +5 s (128K) startup with no steady-state change; `STRATA_COMMIT_BATCH` no measurable
  gfx1100 effect (+5.11% was trajectory-mix confounding); noise floor ±1.3%. —
  [xyzzing 030e7f60be](https://github.com/xyzzing/Strata/commit/030e7f60be2191b5b4420ddae1815b34b07b344f)
- **hugetlb arena**: IQ2_XS solo +4.9% (adopt-grade), pair/concurrent flat. COV; pancake has `os_tune.sh install
  --hugepages`, unmeasured here. — [svgop 8c91289449](https://github.com/svgop/Strata/commit/8c9128944955fb7f7408fb7ae0d5f471893f48ef)
- **tmpfs shadow store**: moving a page-cache-served expert store to `/dev/shm`: +5.5% / +4.2% / +3.5% (solo/pair/conc),
  mechanism 2.6 M minor faults per probe leg on the byte-bound path. Exact. — [svgop 9e889bda3a](https://github.com/svgop/Strata/commit/9e889bda3a838ccf932a25cbd88d6430742beb6e).
  Pancake: its turn log shows `file_blobs = 0` (all experts in VRAM or the mlocked arena), so no page-cache tier to move.
- **Page cache control**: `--drop-file-cache` (MADV_DONTNEED on views + POSIX_FADV_DONTNEED on files; either alone frees
  nothing) returned 55.4 GiB of duplicate page cache after startup — [DamirLukina88 b392dddbb5](https://github.com/DamirLukina88/D.L.-AI---Strata/commit/b392dddbb52152cd4431c32c0a09ee3607a144a5);
  `--protect-ram` (systemd MemoryLow) + `--prewarm` (COV) — [serboox 8836b6cc84](https://github.com/serboox/strata/commit/8836b6cc84914c95859cd64f8d539ceb5a8a1216).
  Pancake loads the arena with O_DIRECT (`STRATA_UNBUFFERED_LOAD` on) and runs under `MemoryMax=54G`, so mostly covered.
- **Server-side warm-up replay**: after READY, a fadvise(WILLNEED) + 4-thread read sweep of exactly what the engine reads,
  then a replay of short + mid prompts through loopback (Qwen 80.1 GiB warm in 8 s + replay 13 s; DS 240 GiB in 46 s +
  19 s); `/health` reports the warm phase. — [svgop cc05319fb5](https://github.com/svgop/Strata/commit/cc05319fb539b208d276dd0fe9e03868fff67e2b).
  Related: decode right after a cold read 66-67 tok/s with 75% acceptance vs 108-134 tok/s / ~99% later in the conversation —
  [atmJava e2f6600149](https://github.com/atmJava/Strata/commit/e2f660014973fc41cae6e636a7215b1a0086d198); prompt-seeded residency
  `STRATA_PROMPT_SEED` and post-prompt page-cache warm-up `STRATA_PROMPT_WARM_MB` (pancake has its own seeding,
  `STRATA_SEED_HOLD/SCALE`) — [mandoof1 122a289eb9](https://github.com/mandoof1/Strata/commit/122a289eb9bce13948f1cf14dfdbc860267c6114)
- **ROCm version**: ROCm 7.14.1 vs 7.9 on 2x RX 7900 XTX: stable, but **prompts ~40% slower**, so 7.9 stayed. —
  [unimatrix099 d71acdec1a](https://github.com/unimatrix099/Strata/commit/d71acdec1ac468e954985aeb2ab3b0658544d303).
  android-imdad's gfx1100 numbers are on ROCm 10.2 nightly (TheRock 10.2.0a20260930); pancake runs ROCm 10.0 (`/opt/rocm/core-10.0`). —
  [android-imdad docs/RX7900XTX.md](https://github.com/android-imdad/Strata-7900xtx/blob/HEAD/docs/RX7900XTX.md)
- **CPU frequency**: `--cpu-max-mhz` caps the CPU clock for thermals (a slowdown knob, COV) — [serboox e7858734db](https://github.com/serboox/strata/commit/e7858734db8b5136f66e59f28ebcd1cf0b77e1e4).
  This box: `amd-pstate-epp`, governor `performance`, EPP `performance` already; `os_tune.sh` does not set or check it.
- **Build flags**: the only CPU-flag commit is BMI2 `pdep` avoidance for Zen 1/2 build machines (IQ1 dots 16-19x faster
  there); Zen 3 has fast pdep, so N/A for the 5700G. — [bsorensen110 084c122106](https://github.com/bsorensen110/Strata/commit/084c1221066df60154ce3f9adb4989cf799bda72).
  android-imdad builds gfx1100 with `-DSTRATA_HIP_EXP_LAYOUTS -DSTRATA_HIP_MMVQ_IL` (kernel variants, not optimization flags).
- **Allocator**: only memory-hygiene commits (malloc's freed memory handed back before parking admission —
  [xrip 972ef575c9](https://github.com/xrip/Strata/commit/972ef575c96d853e8fed5f71e96d51fbd7dc1b71)); no jemalloc/mimalloc/tcmalloc
  measurement found in 21,944 commits.
- **GPU power/DPM**: no commit tuned `power_dpm_force_performance_level` or a compute power profile on AMD (search over all
  messages); this box reports `auto`.

### Inferences
- Add a governor/EPP check line to `os_tune.sh` status (state only; already `performance` here) so a reinstall does not
  silently drop to `powersave`, which noorazman's P-state data says costs up to ~14% when cores idle between phases.
- A ROCm A/B (10.0 vs a 10.2 nightly) is a potentially large prompt-speed lever in either direction (unimatrix099's ±40%);
  only worth it with a rollback path (memory notes the -dev packages were autoremoved once).
- A post-READY warm-up replay in `serve/` (one short + one mid prompt) would move the cold-acceptance/cold-cache penalty
  off the first real agent turn; pancake has `STRATA_ARENA_WARM` / `STRATA_IC_WARM_MB` for the arena, but no request
  replay was found. Gain: first turn after a restart only.
- An AMD GPU DPM `high` / compute-profile arm for short-turn TTFT is untested anywhere; it is a cheap experiment (sysfs) but
  is a system setting the user should apply themselves.

### Gaps
- LTO/PGO/`-march=znver3` for the C++ host code and a malloc swap: no fork data at all.
- `vm.swappiness`, IRQ affinity, `nohz`/`isolcpus`: no commit measured them.

## Q5. Python server overhead and streaming

### Takeaway
The two measurable Python costs (re-encoding the whole conversation each turn, re-decoding the whole answer per token)
were fixed in gputier's fork and are already in pancake via upstream; the residual server overhead on our box is ~19 ms
median between the engine's prompt end and its first token.

### Cited Findings
- Incremental prompt encoding (PromptEncoder: encode only after the longest shared prefix): 235 ms -> 1.5 ms for a 100K
  prompt on a two-core container; 482 ms -> 3.7 ms for a 101K conversation. Exact (tests compare ids token for token). —
  [gputier f29e527856](https://github.com/gputier/StrataGP/commit/f29e527856b85e37bda90626ccc28e002b4989dd),
  [gputier 6be65551f4](https://github.com/gputier/StrataGP/commit/6be65551f482824a9b127aa449db64e73586cc70)
- Streaming detokenizer (incremental UTF-8 instead of re-decoding the answer per token): 4.9 ms/token at 16K tokens -> ~1 us. —
  [gputier 6be65551f4](https://github.com/gputier/StrataGP/commit/6be65551f482824a9b127aa449db64e73586cc70).
  Pancake: has both (`serve/test_prompt_encoder.py`, `serve/test_detok.py`, `PromptEncoder` in `serve/server.py:4731`).
- Before such fixes, TTFT at 32K carried 2.3 s of HTTP/Python tokenize — [noorazman 25fbc76161](https://github.com/noorazman/Strata/commit/25fbc7616129b17ccdf2c23463a2a2302ef875af);
  after, client TTFT exceeds the engine read by 0.05-0.10 s — [atmJava e2f6600149](https://github.com/atmJava/Strata/commit/e2f660014973fc41cae6e636a7215b1a0086d198)
- HTTP robustness: a stalled connection held its thread forever and thread count was unbounded (fixed) —
  [shoftee 90402cf5a0](https://github.com/shoftee/Strata/commit/90402cf5a0a6131451b5755db052e5d94ea38115)
- Pancake measured (turn log): `first_engine_token_s x 1000 - prompt_ms` median 19 ms over 622 turns.

### Inferences
- No server-side speed work is left with a measurable prize; the server's remaining role is cache policy (Q1).

### Gaps
- No fork measured Python 3.13/3.14 free-threading, uvloop or orjson for the server.

## Q6. Fork-level benchmarks: who is fastest overall and why

### Takeaway
On comparable AMD hardware the strongest public numbers are android-imdad's RX 7900 XTX edition (bit-exact kernels +
draft policy + ROCm 10.2) and xyzzing's 7900 XTX line (route prior + gfx11 matrix arms); pancake's 126 tok/s (fast) /
168 tok/s (turbo) on a 7900 XT with IQ2_XS is in the same band. The large multi-GPU numbers (XopMC, svgop) come from
24-32 GB NVIDIA cards and more RAM bandwidth, not from server tricks. Server/agent-layer gains show up as TTFT and
aggregate throughput, not decode tok/s.

### Cited Findings
- **android-imdad/Strata-7900xtx** (RX 7900 XTX 24 GB, Ryzen 7 9800X3D, 62 GiB, ROCm 10.2, IQ3_S), vs upstream v0.2.4.1 same
  day, medians of 3: short-chat decode 82.6 -> 97.5 tok/s (+18%), first token 1.20 -> 0.84 s (-30%), 4K/16K/32K prompt
  +34/+39/+42% (1,365 / 1,868 / 1,928 tok/s), **agent turn first token 3.98 -> 2.99 s (-25%)**, agent decode +5.5%. EXL3
  experts (opt-in, lossy within 0.2% ppl on chat): 115.1 tok/s, agent first token 1.69 s (-56%). Why: Tensile-exact grouped
  gate/up (`STRATA_PF_GU_EXACT`), launch-shape fixes, deterministic decode (`STRATA_IQ_MT_MIN=1`) enabling `--lookup-chain 3`
  drafting, `STRATA_SPEC_CUM=0.5` (+6.4% short chat), 32K exact prompt chunks. Pancake has `STRATA_IQ_MT_MIN`, `--lookup-chain`,
  `STRATA_WAIT_FUSE`, `STRATA_IO_THREADS`; it lacks `STRATA_PF_GU_EXACT`, `STRATA_SPEC_CUM`, `STRATA_PF_LEAN_ISSUE`,
  `STRATA_PF_COPY2`, `STRATA_WAIT_COMBINE`, `STRATA_COMBINE_HITS` (grep = 0; kernel-side items belong to the kernel notes). —
  [android-imdad README](https://github.com/android-imdad/Strata-7900xtx), [93295e0d71](https://github.com/android-imdad/Strata-7900xtx/commit/93295e0d71188c736b70ce994212a6291366709e)
- **xyzzing/Strata** (7900 XTX, 0.1.45 line): decode 102-142 tok/s at 1K (drafter trajectory dependent), ~73 at 32K,
  52-60 at 128K; prefill 2,337-2,421 tok/s at 32K/128K; adopted `STRATA_ROUTE_PRIOR=0.5` (+22-28%, lossy), `STRATA_GDN_CHUNKED`
  (+5.9% prefill), `STRATA_HIP_WMMA`/`SELECT_WMMA` (+22.8% / +8.7% at 128K), tuned hipBLASLt table (up to +82% at 128K).
  Pancake's preset already sets all of these. — [xyzzing README](https://github.com/xyzzing/Strata)
- **xrip/Strata** (2x Turing, Swift 1.5 IQ3_S): live config vs stock upstream: 64K read +56%, 128K +90%, decode +14% (74.3-75.0
  -> 84.7-85.2 tok/s); `--pool-workers 19` (both sockets) vs 0.2.0's 9: decode +14.6% (dual-socket only; pancake's default is
  every physical core minus the host). — [xrip README](https://github.com/xrip/Strata), [xrip 4aa428da86](https://github.com/xrip/Strata/commit/4aa428da868916a52a3314a1dc5284aad3e55ced),
  [xrip 342185bd9f](https://github.com/xrip/Strata/commit/342185bd9f340714d3448bfebff4550984e44143)
- **wesraph/Strata** (2x RTX 3090, multi-agent replay): aggregate 49.9 -> 79.2 tok/s, later-turn first token median
  2.73 -> 1.55 s, p90 5.11 -> 4.43 s; expert cache hit at the end 35-63% -> 96-99%; prefill ~3.6K tok/s for 46-237K first reads. —
  [wesraph README](https://github.com/wesraph/Strata)
- **webzone/strata-gfx906** (dual MI50, agent window): 90.0% of 318,201 prompt tokens reused; a warm 37,026-token prompt
  accepted in 0.453 s; real cold prefill 601.6 tok/s; MTP acceptance 75.4%. Shows how much TTFT comes from reuse. —
  [webzone README](https://github.com/webzone/strata-gfx906)
- **XopMC/Strata-Server** (RTX 4090s, native Swift Flash Next, NVFP4): 173-225 tok/s generation, 3.8K-12.3K tok/s prefill at
  256K-1M contexts on 1-4 GPUs. Not comparable hardware. — [XopMC README](https://github.com/XopMC/Strata-Server)
- **ProTekk**: `kv-grow` adopted +5.3% tg (COV; pancake has kv-grow) — [ProTekk 321a55c367](https://github.com/ProTekk/Strata/commit/321a55c367e2a00d34d4b62a74e2795e8674e9f0)

### Inferences
- Ranked list of ideas for pancake (all unverified on our box; "exact" = output unchanged):

| # | Idea | Fork / commit | Switch | Exact? | Pancake has it? | Est. gain on our box |
|---|---|---|---|---|---|---|
| 1 | Diagnose the 54-61K cold re-reads; port partial parking | TwoAnts `5ef4fc8b4b` | `STRATA_PARK_TRUNCATE=1` | exact | no | 20+ s per affected turn (7 turns ~162 s seen once) |
| 2 | Spread checkpoints back from the end | xrip `56017b1c41`, Lamina `13d90a38ec` | `--prompt-cache-spread` | exact | flag yes, preset no | 4x shorter re-read after an edit/rewind (10.2 vs 40 s at 82K) |
| 3 | No park when a request rewinds its own tail | Xiaomi2008 `8e4ccbe90d`, UltimateBoomer `5d0a9e786c` | `STRATA_PARK_REWIND_MAX=N` / `--conversation-cache-skip-tail N` | exact | no (has park-gain) | 0.1-0.7 s per affected turn |
| 4 | Fork-point + message-end checkpoints | Lamina `ab808b133e` | (code) | exact | partial (8K+ message boundary, tool boundary) | -5..-10% wall on multi-call agent tasks |
| 5 | Borrow before the newest checkpoint | Lamina `83f40aee62` | (code) | exact | base borrow yes (`c9e571e0`) | sub-agent switch 3.0 -> 0.7 s class |
| 6 | Helper threads off the host CPU | Ben1078 `5ffd2ccaec` | `--aux-cpus auto` | exact | yes, off | 0-4% decode |
| 7 | `GPU_MAX_HW_QUEUES` 4/6/8 sweep | android-imdad doc, unimatrix099 `3e1b02cfda` | env | exact | set to 6 | ±10-20% prompt (8 was worse on XTX) |
| 8 | Batch slots for parallel sub-agents | unimatrix099 `fa10a01bd3`, atmJava `5715656718` | `--batch N` | exact per stream | upstream code, unused | aggregate up to ~2x at 2-4 streams; VRAM-limited |
| 9 | Post-READY warm-up replay | svgop `cc05319fb5` | (server) | exact | partial (arena warm) | first turn after restart only |
| 10 | ROCm 10.0 vs 10.2 A/B | unimatrix099 `d71acdec1a`, android-imdad doc | system | exact | n/a | ±40% prompt observed between ROCm versions elsewhere |
| 11 | PCIe keep ring + usage-ordered fetch | android-imdad `570854aded` | `STRATA_PCIE_USAGE=1 STRATA_PCIE_KEEP=96` (+ `--pcie-frac 0.35`) | exact | no | +4.1% decode on XTX; needs PCIe fetch on |
| 12 | strata-poly GDN snapshots for re-rendered templates | Aaron-xx `986649d5c6` | `STRATA_Q36_SNAP=N` | exact | check poly `--prompt-cache` | large only if poly re-reads whole prompts |

- Measured-negative list to avoid repeating: SMT siblings for pool lanes (-20%), CCD-split pools (-13..-20%), futex parking
  (-14%), device spin flags on AMD (0%), next-layer expert prefetch on 7900 XTX (-3-4%), stream-query polling at the window
  end (-3%), `STRATA_COMMIT_BATCH` on gfx1100 (0%), forced 4 KB arena (+5-8 s start), `GPU_MAX_HW_QUEUES=8` for prompts on XTX.

### Gaps
- No fork publishes a 7900 XT (20 GB) + DDR4 Zen 3 table; the nearest AMD tables are 24 GB XTX with Zen 4/5 and DDR5, so
  absolute tok/s are not comparable, only the percentages.
- Several big claims are from the forks' own A/B with 2-3 rounds; android-imdad and svgop report spreads, most others do not.
