# A7 (chunked, cancellable promotions) and A46 (Belady oracle + Belady imitation) - batch 2, 2026-11-10

Branch `work/a7-promotions` (from `origin/dev/batch-2` 99a745af5). MOD_PLAN Phase 16 rows A7 and A46. Starting point:
bench/results/2026-11-10-offline/A6_A17.md (recalibrated time model: A46 > A8 > A7, none worth a GPU screen at the bench's
hit rate; "judge them on real agent traces").

## A7 - chunked, cancellable promotions + reordered landing (ProMoE)

Paper: ProMoE, SJTU 2024, https://arxiv.org/abs/2410.22134 (chunked prefetch that can be pre-empted; execution reordered
cached -> prefetched -> the rest, so copies overlap compute).

### What it does

Opt-in `STRATA_PROMO_CHUNK=<bytes>` (K/M/G suffix allowed; rounded down to 4 KiB, at least 64 KiB; unset or 0 = off).
Where: the asynchronous adaptive tier (`--adapt-async 1`) in the arena mode, serial decode loop (`adapt_tick` in
`src/program/generate.cpp`). In the arena mode a round is: pick swaps (in, out, slot) -> mark the evicted experts
non-resident (CopyBack) -> copy the incoming experts into the slots on the job thread (SwapIn) -> mark them resident.
With the switch:

1. **Chunked copies.** SwapIn copies one chunk per swap per window tick (the same job thread, the same home stream,
   the same event and `a_ready` fence as the old `a_copy_in`), instead of whole 1.4 MB experts at once. A round's DMA is
   spread over windows (less DDR4 read burst against the CPU pool at once: A6's alpha term), and each swap can be
   stopped between two chunks.
2. **Cancellation.** Between windows (never on a drain), a swap still copying is cancelled when the incoming expert was
   **not routed** since the round's copies started and the evicted expert **was** (`promo_should_cancel`, base = the
   routing counts at step 2). The slot still holds the evicted expert's bytes past the incoming bytes written so far,
   so a cancel copies back only those bytes from the evicted expert's arena blob (`promo_cancel`) and the evicted expert is
   resident again. A cancel before the first chunk landed costs nothing (the slot was never written). If the incoming
   expert became resident elsewhere meanwhile (cannot happen with the gates below; checked anyway), the swap is put back
   the same way.
3. **Reordered landing.** Each swap becomes resident (the GPU computes it from the next window) as soon as its own
   last chunk has landed, not when the whole batch has. This is the window-granular form of ProMoE's order
   "VRAM-cached -> finished in flight -> CPU": at every window the GPU has every expert whose copy is complete, the CPU
   the rest. **Not built:** ProMoE's in-window form (the GPU picks up an expert whose copy finishes during the layer) -
   that needs a per-slot ready flag written on the copy stream (`hipStreamWriteValue32` after the last chunk) and read by
   the expert kernels and the CPU pool's split, i.e. a change in `verify.cpp`'s dispatch; out of scope without a GPU
   build. The A6 model gives the whole A7 idea an upper bound of +4.6% (no contention + in-flight reorder) on the main
   trace, so the in-window part is worth at most a few percent.

Exact: placement and scheduling only, no arithmetic changes. Like any tier timing change it is rounding-level different
text (which window first runs an expert on the GPU, which rounds differently from the CPU, moves); the async tier is not
bit-exact run to run already.

Gates (logged at start): the arena (`srcp == &arena_src`), `--adapt-async 1` given, no `--pipeline-windows 2` (that loop's
copies must wait for each card's gap, `a_gap`), no `STRATA_ADAPT_FETCH` (it admits into slots from inside the windows).
Otherwise: `strata serve: STRATA_PROMO_CHUNK=N: off (needs ...)`. On: `strata serve: adaptive tier: promotions in chunks
of N bytes ...`; at the end of a session: `strata serve: chunked promotions (STRATA_PROMO_CHUNK=N): C cancelled, R evicted
experts put back, L slots lost, M MB copied` (L must be 0).

Safety (stream / fence logic kept): the job thread reads only the chunk plan `a_pc_jobs`, which this thread writes only
after `a_ready` has seen the job idle and the home events complete (or at CopyBack, after its own `a_ready`). A slot being
written (incoming chunks or restore chunks) is referenced by no residency entry: the evicted expert left the table at
CopyBack (uploaded before the first chunk is queued) and an entry is written (and uploaded with `res_upload`) only after
the chunk that completes it has been fenced. Restore copies queue on the same stream after any chunk still in flight
(counted as written). A drain (`adapt_tick(true)`, used by `adapt_quiesce` before slots are lent) copies the rest of
every swap at once and waits, so nothing is in flight when a slot changes owner. `save_profile` no longer counts a
cancelled swap's incoming expert as resident.

Default path: with the variable unset, `a_pc` stays empty; the only new statements on the path are an empty-vector test
in SwapIn, a `a_pc.clear()` of an empty vector and a loop in `save_profile` that sets the same entries as before. No log
line changes. Byte-identical by construction (not measured: no build tonight).

### Files

- `include/strata/program/promo_chunk.hpp` (new): parse, per-swap progress, next chunk, cancel rule and arithmetic.
- `tests/program/promo_chunk_test.cpp` (new), registered in `CMakeLists.txt` next to `adaptive_range_test` (and in the
  CPU test list).
- `src/program/generate.cpp`: include; state next to `a_rng`; gate after `a_batch`; `a_pc_copy / a_pc_post /
  a_pc_restore / a_pc_start / a_pc_tick` after `a_post_batch`; CopyBack and SwapIn branches; `a_pc.clear()` at Commit;
  `save_profile`; end-of-session line.
- `bench/specs/22-a7-promo-chunk.json` (new): `async` vs `async-chunk256k`, `async-chunk512k`,
  `async-kmiss28-chunk256k`, short + edit, 3 reps.

### Tests

- `promo_chunk_test` (host-only, header-only), built and run standalone, pinned to one CPU at idle priority while the
  09-prefill screen ran (the coordinator allowed it): `g++ -std=c++20 -Wall -Wextra -O1 -Iinclude
  tests/program/promo_chunk_test.cpp` -> no warnings, `promo_chunk_test: ok`. Covers the parse, the next chunk, the
  cancel rule, whole rounds (1.41 MB in 256 KiB chunks = 6 ticks, chunk 0, chunk > blob, exact multiple), cancels after
  0-7 chunks (the slot ends byte-identical to the evicted blob and exactly the bytes written go back), a cancel with a
  chunk in flight, a landed swap put back, a drain mid-round. Not run through CMake/ctest (no configure tonight).
- `src/program/generate.cpp`: **not compiled** (no engine builds tonight). Re-read for syntax, types and includes:
  `blob_at` (const, `ArenaExpertSource`), `device_slot` (non-const `ExpertCache*` -> `uint8_t*`), `drive.d.usage`
  (`[layer * n_expert + e]`, as `a_pick` reads it), `drive.d.fetch_admit`, `o.pipeline_windows` (resolved before this
  point), `JobThread::post(std::function<void()>)`, `nib::guard_raw`. The coordinator's 02:00 build is the check.

### Spec, expected gain, risk

`python tools/pancake/abtest.py bench/specs/22-a7-promo-chunk.json --tier screen` - compare the chunk arms with the
`async` arm (the preset is the blocking tier; `--adapt-async 1` measured -9.6% against it earlier, A8 row). Check the log
line that says the switch is on (the preset must run the arena). Expected: 0-3% over `async` (A6: A7 bound +4.6% with the
in-window part; this build has only the window-granular part and cancels); at the screen's +-1.7-3 noise it may not
resolve. Risks: a promotion now takes ceil(1.41 MB / chunk) windows to land (6 at 256 KiB): when the round's copies fit a
window anyway (32 x 1.4 MB = 45 MB = 3.7 ms at 12 GB/s, the edit window is 40 ms), chunking delays hits; the cancel rule
can thrash on an expert that alternates. Larger chunks (512 KiB, 1M) trade cancel granularity for landing delay. Not built
or run on the GPU: compile errors are possible (re-read for syntax and types; the coordinator's 02:00 build is the check).

### What is left

- The in-window reorder (per-slot ready flag in the copy stream, read by the VRAM expert kernels and the CPU split).
- A cancel that redirects the slot to a better candidate instead of restoring (costs a whole copy; only worth it if the
  log shows many cancels).
- The pipelined loop (`--pipeline-windows 2`): chunks would have to be queued at each card's gap.

## A46 - Belady oracle and Belady imitation on the simulator

Papers: ADRS / "Barbarians at the Gate", Berkeley 2025, https://arxiv.org/abs/2510.06189 (evolve the policy on a
simulator); Hawkeye, Jain & Lin, ISCA'16 (OPTgen + a PC-indexed predictor of cache-friendly loads); Harmonia,
https://arxiv.org/abs/2503.20507 (admission and migration decided apart).

### What was built

`tools/pancake/cache_learn.py` (new, `--help`), on top of `time_ceiling.py` (`per_layer`, `next_use`, `replay`, `tier`)
and `replay_cache_policy.py` (trace and profile loaders, the engine tier). Same cache model as time_ceiling: per layer the
profile's share of SLOTS, a window served from the slots as they are at its start, hits counted per routed entry (the
engine's "hits / all routed"), admissions between windows. New policies:

- `opt`: Belady with bypass (window-granular; a miss is admitted only if its next use is sooner than the farthest
  resident's). The optimum of distinct misses per window: the unit test checks it against a brute-force DP over every
  admission / eviction choice on 60 random small traces.
- `label`: perfect imitation - admit a miss iff OPTgen says Belady keeps it until its next use (future knowledge);
  eviction averse-then-LRU. The ceiling of any admission predictor on an LRU stack.
- `hawkeye-pc-h8 / -pc-h16 / -pcr-h8 / -glob-h8`: online Hawkeye. OPTgen rebuilds Belady's decisions on the past windows
  (a reuse interval is cached iff every window in it is below the layer's capacity); an access not reused within H
  windows leaves the history and counts as averse; a 3-bit counter per key learns friendly/averse; a miss is admitted iff
  its key is friendly (the admission/migration split: averse misses stay on the CPU, no copy); eviction averse-then-LRU.
  Keys: the expert (the "PC"), expert x recent-use bucket, or per-layer recent-use x gap buckets (no identity).
- `imit`: offline Belady imitation - a logistic model on past-only features of the access (used in the last window,
  uses in the last 4 / 16 windows, log gap since the last use, tokens routed to it now, long-run frequency) fitted to the
  OPTgen labels of the other traces (leave one trace out, class-balanced), admit at p >= the threshold with the best
  mean hit rate on those training traces (`--imit-thr auto`), eviction averse-then-LRU.

Reference policies: `tier` (the engine's preset tier: every window, 32 swaps, decay 0.7), `lru`, `belady` (admit every
miss), `belady-h4` (A6's "oracle, admit if reused <= 4 windows"), `kmiss 2/4` and `kmiss 2/8` (`STRATA_ADAPT_KMISS`).
CPU-side cost per window from A6's calibration: 48.7 us per distinct missed expert + alpha 0.5 x 117 us per promotion.

### Data - no agent traces exist

There is **no routing trace of an agent session** on this box (searched the scratchpads, `bench/`, the repo; the A6 note's
next action 4, "collect agent-session routing traces with conv_trace.py", has not been run, and it needs the GPU). The
only routing trace is the 2026-11-08 trace-bench run (`routing.bin`, sha256 37b87e16..., 219 windows, 6 tokens x k=10 per
layer per window), whose four requests are also the four `*.trace` segment files (read4k 11 windows, short chat 85, code
edit 61, long 62; the A6 sensitivity traces at 86 / 77 / 93% tier hits). So the policies were judged on:
(a) each of the four requests, replayed from the profile's residents, `imit` fitted on the other three;
(b) the whole trace; (c) a **synthetic agent-like trace**: the four requests cut into 8-window turns and played
round-robin (a context switch every 8 windows, as an agent moving between sub-tasks) - synthetic, not a measurement.

### Results

Run 2026-11-10 23:45, `taskset -c 15 chrt -i 0 nice -n 19` (one logical CPU, idle priority, beside the 09-prefill
screen; the coordinator allowed it), about 1 min. Slots 10909 (the preset), profile `expert-profile-pancake.bin` (the A6
one). Raw: `bench/results/2026-11-10-offline/A46_cache_learn.txt` and `.json`. Sanity: the simulator reproduces the
earlier replays exactly - whole trace tier 81.17% / k-th miss 2/4 85.96% / 2/8 86.51% (RESULTS.md 4.0.4: 81.2 / 86.0 /
86.5), edit 76.68 / 86.47 / 86.87 (76.7 / 86.5 / 86.9).

Hit rate (% of routed entries served from VRAM), and in brackets hit points over k-th miss 2/8. `whole` = the 219
windows in order with the cache carried across requests; `interleave8` is SYNTHETIC. `imit` is held out on the four
requests (fitted on the other three) and in-sample on `whole` / `interleave8`; its threshold was picked on the training
traces (0.2 every time):

| policy | read4k | short | edit | long | whole | interleave8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tier | 83.12 (+0.94) | 85.90 (-3.24) | 76.68 (-10.19) | 92.97 (+0.22) | 81.17 (-5.34) | 81.83 (-2.21) |
| lru | 85.35 (+3.17) | 92.32 (+3.17) | 88.40 (+1.54) | 94.16 (+1.41) | 89.46 (+2.95) | 84.81 (+0.77) |
| belady | 89.05 (+6.87) | 95.70 (+6.56) | 94.19 (+7.32) | 96.66 (+3.91) | 94.75 (+8.24) | 93.54 (+9.50) |
| belady-h4 | 87.77 (+5.59) | 93.50 (+4.36) | 93.17 (+6.30) | 95.70 (+2.95) | 93.30 (+6.79) | 92.01 (+7.97) |
| kmiss 2/4 | 81.95 (-0.23) | 88.29 (-0.85) | 86.47 (-0.40) | 92.75 (-0.00) | 85.96 (-0.55) | 83.97 (-0.06) |
| kmiss 2/8 | 82.18 (+0.00) | 89.15 (+0.00) | 86.87 (+0.00) | 92.75 (+0.00) | 86.51 (+0.00) | 84.04 (+0.00) |
| opt | 89.05 (+6.87) | 95.71 (+6.57) | 94.43 (+7.56) | 96.66 (+3.91) | 94.92 (+8.41) | 93.86 (+9.82) |
| label | 87.86 (+5.68) | 94.67 (+5.52) | 93.24 (+6.37) | 96.30 (+3.55) | 94.22 (+7.71) | 92.89 (+8.85) |
| hawkeye-pc-h8 | 85.35 (+3.17) | 89.31 (+0.17) | 87.62 (+0.75) | 91.79 (-0.96) | 85.51 (-1.00) | 82.49 (-1.55) |
| hawkeye-pc-h16 | 85.35 (+3.17) | 91.12 (+1.98) | 88.15 (+1.29) | 93.33 (+0.58) | 87.53 (+1.02) | 84.23 (+0.19) |
| hawkeye-pcr-h8 | 85.35 (+3.17) | 90.02 (+0.87) | 87.81 (+0.94) | 92.12 (-0.63) | 86.46 (-0.05) | 83.39 (-0.65) |
| hawkeye-glob-h8 | 85.30 (+3.12) | 88.45 (-0.70) | 86.50 (-0.37) | 91.48 (-1.27) | 85.80 (-0.71) | 80.22 (-3.82) |
| imit | 85.35 (+3.17) | 92.30 (+3.15) | 88.34 (+1.47) | 94.16 (+1.41) | 89.42 (+2.91) | 84.71 (+0.67) |

Distinct missed experts per window / promotions per window / CPU-side ms per window (48.7 us x miss + 0.5 x 117 us x
promotion; the serial edit window is ~40 ms, the short one ~25 ms):

| policy | read4k | short | edit | long | whole | interleave8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tier | 232 / 27 / 12.9 | 224 / 25 / 12.3 | 732 / 32 / 37.5 | 78 / 12 / 4.5 | 347 / 27 / 18.5 | 334 / 26 / 17.8 |
| lru | 214 / 214 / 22.9 | 123 / 123 / 13.2 | 389 / 389 / 41.6 | 64 / 64 / 6.8 | 196 / 196 / 21.0 | 288 / 288 / 30.9 |
| belady | 159 / 159 / 17.0 | 70 / 70 / 7.5 | 199 / 199 / 21.3 | 37 / 37 / 3.9 | 98 / 98 / 10.5 | 126 / 126 / 13.5 |
| belady-h4 | 173 / 129 / 16.0 | 106 / 42 / 7.6 | 236 / 112 / 18.1 | 48 / 16 / 3.2 | 129 / 53 / 9.4 | 156 / 57 / 10.9 |
| kmiss 2/4 | 236 / 48 / 14.2 | 184 / 46 / 11.6 | 448 / 124 / 29.0 | 79 / 18 / 4.9 | 260 / 71 / 16.8 | 300 / 76 / 19.0 |
| kmiss 2/8 | 234 / 51 / 14.4 | 171 / 52 / 11.4 | 434 / 146 / 29.7 | 78 / 22 / 5.1 | 249 / 82 / 16.9 | 298 / 91 / 19.8 |
| opt | 159 / 53 / 10.8 | 70 / 50 / 6.3 | 191 / 132 / 17.0 | 37 / 18 / 2.9 | 95 / 74 / 8.9 | 120 / 86 / 10.9 |
| label | 176 / 64 / 12.3 | 86 / 63 / 7.8 | 229 / 161 / 20.5 | 41 / 22 / 3.2 | 109 / 83 / 10.1 | 138 / 97 / 12.4 |
| hawkeye-pc-h8 | 214 / 214 / 22.9 | 167 / 117 / 15.0 | 414 / 310 / 38.2 | 90 / 67 / 8.3 | 270 / 136 / 21.1 | 330 / 141 / 24.3 |
| hawkeye-pc-h16 | 214 / 214 / 22.9 | 142 / 117 / 13.7 | 396 / 342 / 39.2 | 74 / 66 / 7.5 | 232 / 155 / 20.4 | 298 / 182 / 25.1 |
| hawkeye-pcr-h8 | 214 / 214 / 22.9 | 157 / 131 / 15.3 | 408 / 339 / 39.7 | 86 / 73 / 8.5 | 253 / 176 / 22.6 | 315 / 206 / 27.4 |
| hawkeye-glob-h8 | 215 / 185 / 21.3 | 180 / 111 / 15.2 | 444 / 293 / 38.7 | 93 / 59 / 8.0 | 264 / 170 / 22.8 | 360 / 192 / 28.7 |
| imit | 214 / 214 / 22.9 | 124 / 124 / 13.2 | 391 / 391 / 41.8 | 64 / 64 / 6.8 | 197 / 197 / 21.1 | 290 / 290 / 31.0 |

What the numbers say:

- **The ceiling is real and large where hits are low.** Belady with bypass (`opt`) is +3.9 to +9.8 hit points over k-th
  miss, and at fewer promotions than k-th miss on the whole trace it would halve the CPU-side time (8.9 vs 16.9 ms per
  window). The synthetic agent-like trace (`interleave8`, context switch every 8 windows) has the largest gap (+9.8).
  Perfect admission on an LRU stack (`label`, future knowledge) keeps most of it: +3.6 to +8.9.
- **No online or held-out policy gets near it.** Best learned results over k-th miss 2/8: `imit` +3.2 (read4k, short),
  +1.5 (edit), +1.4 (long), +0.7 (interleave8); Hawkeye `pc-h16` +2.0 (short), +1.3 (edit), +0.2 (interleave8). Both
  gain hit points only by admitting almost every miss: at the threshold that maximises training hits (0.2) `imit` is LRU
  (same misses and promotions, 2-3x the promotions of k-th miss); Hawkeye converges to "friendly" for most experts
  because at 227 slots per layer OPTgen keeps 84-87% of all reuses (friendly share per request: read4k 0.55, short
  0.87, edit 0.87, long 0.84). With a shorter history (h2 / h4) Hawkeye bypasses more and loses hits (edit 82.1 / 86.3,
  interleave 72.6 / 78.5; a one-off sweep). A class-balanced `imit` at threshold 0.5 bypasses most misses and falls to
  83.9 (short) / 82.7 (edit).
- **In time, k-th miss is still the best online policy.** Every policy that beats it in hits pays for it in
  promotions: CPU-side ms per window, k-th miss 2/8 vs the best learned one: short 11.4 vs 13.2-15.3, edit 29.7 vs
  38.2-41.8, interleave8 19.8 vs 24.3-31.0. The `tier` row (the engine's preset tier) shows why k-th miss was adopted
  (edit: 37.5 ms).
- The features that would be needed are about the **future** (whether this expert comes back within a few windows), and
  the past ones (recency, frequency, tokens now, gap) are already what LRU and k-th miss use. The logistic weights fitted
  on all requests put most weight on long-run frequency (+1.25) and uses in the last 16 windows (+1.07), i.e. "hot
  experts stay hot", which k-th miss already captures.

### Verdict

**Not worth an engine port now.** The A46 rule (a learned policy at >= 5 hit points over k-th miss on held-out agent
traces) is not met: the best held-out gain is +3.2 points, and only by admitting every miss (LRU-like), which costs more
window time than k-th miss under the A6 calibration (alpha 0.5). The >= 5-point region exists only for the oracles
(opt +3.9..+9.8, label +3.6..+8.9). The caveat is the data: there is no real agent trace; the synthetic interleave is
where the oracle gap is largest (+9.8) and also where the learned policies do worst (+0.2..+0.7).

If agent traces later show a learned policy at >= 5 points held out, the port is small: the admission decision is
already a separate step in `include/strata/program/kmiss_tier.hpp` (`KMissTier::select` picks candidates, then pairs
them with victims) and runs on both tiers. A Hawkeye admission would add a per-(layer, expert) 3-bit counter table
(48 x 512 bytes) trained at each round's `observe` from the routing counts (an OPTgen occupancy ring of 16 rounds per
layer), and `select` would take a candidate only if its counter is friendly (same victims). An imitation model would be
the 7-feature dot product per missed expert at `observe`. Both opt-in (`STRATA_ADAPT_ADMIT=hawkeye|imit`), placement
only.

### How to repeat / run on agent traces

```
nice -n 19 python tools/pancake/cache_learn.py data/expert-profile-pancake.bin 10909 routing.bin \
    --cuts 11,96,157 --names read4k,short,edit,long --interleave 8 --json a46.json
# agent traces (when collected, e.g. conv_trace.py OUT/<tag>.trace or --dump-routing on a replay_turns.py run):
nice -n 19 python tools/pancake/cache_learn.py data/expert-profile-pancake.bin 10909 agent1.trace agent2.trace ...
```

Each trace given after the first (or each one without `--cuts`) is a leave-one-out unit for `imit`.

### Files and tests

- `tools/pancake/cache_learn.py` (new), `tools/pancake/test_cache_learn.py` (new); one docstring line in
  `tools/pancake/time_ceiling.py` pointing at it; raw output `bench/results/2026-11-10-offline/A46_cache_learn.{txt,json}`.
- `python tools/pancake/test_cache_learn.py` (pinned, idle priority): 9 of 9 ok - opt equals a brute-force DP optimum on
  60 random small traces; opt never has more distinct misses than any other policy (8 random traces); OPTgen labels by
  hand (1, 2 and 0 slots); Hawkeye's history horizon trains an unreused access averse once; on a hot pair + scan, label and hawkeye-pc beat LRU and promote less; features use the past
  only; the logistic fit separates; the interleave keeps every window in round-robin order; the command line on a
  synthetic 48-layer trace (all keys, hits in [0, 1], opt minimal, JSON written).

### What is left

- Agent-session routing traces (GPU, ~10 min with conv_trace.py; or `--dump-routing` during a replay_turns.py run of the
  opencode session) and a rerun of this tool on them; the decision rule stays ">= 5 hit points over k-th miss, held out".
- Weighted traces (the trace's weight column is zero) would let the heavy-row share enter the cost.
