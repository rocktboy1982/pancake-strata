# A44 - epoch bandit for the mixed opt-ins (STRATA_BANDIT)

Branch `work/a44-bandit` (from `origin/dev/batch-2`). Status: implemented and pushed, **nothing compiled**: a GPU
benchmark ran through all three quiet checks (23:12-23:50), so per the brief the code was reviewed by hand instead.
Needs the 02:00 build, the unit tests below and the screen spec.

## What

Some opt-ins won on code edits and lost on short chat in the A/Bs:

| switch | own env var | measured | exactness vs. the switch off |
| --- | --- | --- | --- |
| `top` (rank guard, item 28 / A4) | `STRATA_VERIFY_DEFER_CPU_TOP=2` (with `STRATA_VERIFY_DEFER_CPU=1`) | test 01: edit +20.1%, short -6.9% | **lossy (small)**: changes which CPU rows join one layer late |
| `kmiss` (k-th miss admission, A5) | `STRATA_ADAPT_KMISS=2,8` | helps edit, not short (02b: +2.0% edit, -6.4% short) | exact up to rounding: placement only (a GPU and a CPU expert round differently) |
| `preread` (pool pre-read) | `STRATA_POOL_PREREAD=1` | mixed | bitwise exact: data movement only |

`STRATA_BANDIT=<switches>` turns these on or off per **window type** with an epoch bandit, instead of fixing them for
the whole process. This follows Athena (ETH SAFARI 2026, https://arxiv.org/abs/2601.17615). Athena is an online
agent that, epoch by epoch, turns a prefetcher and an off-chip predictor on or off and sets how aggressive they are,
rewarded by the change in the epoch's cycle count. Here the "prefetcher / predictor" are our opt-ins and the reward is
the measured round time per committed token.

How it works (core/epoch_bandit.hpp):

- An **arm** is a subset of the selected switches (2^n arms, "off" included).
- Each window type has its own **epochs**. An epoch is `STRATA_BANDIT_EPOCH` (16) counted windows that all run one arm.
  The first `STRATA_BANDIT_SKIP` (2) windows of an epoch are not counted: the previous arm's after-effects (the
  adaptive tier's swaps land in later windows, earlier pre-reads are still in the caches).
- **Cost** of an epoch = sum of round times / sum of committed tokens over its counted windows. A round is the serve
  loop's whole round: adaptive-tier wait, the window, the commit and the next draft (`round0` to the end of the round).
  This is the same quantity the draft policy learns from. Per accepted token, so a switch that changes acceptance (the
  rank guard changes the text) is judged on what the user gets.
- The epoch's cost goes into the arm's **discounted mean**. Every arm of the type decays by `STRATA_BANDIT_DECAY`
  (0.9) per epoch, because sessions drift (the context grows).
- **Rule**: every arm once, in order ("off" first), then epsilon-greedy (`STRATA_BANDIT_RULE=eps:0.1`, the default) or
  UCB1 for a cost (`STRATA_BANDIT_RULE=ucb[:c]`, c = 0.1, scaled by the mean cost so it has no unit; the bonus counts undiscounted epochs, else every arm left alone for a few epochs looks unexplored again). A splitmix64 stream
  seeded by `STRATA_BANDIT_SEED` (1): deterministic for a seed and a sequence of observations.
- **Window types** (`STRATA_BANDIT_TYPE`, known before the window runs):
  - `req[:P]` (default, P = 30): the request's recent lookup share (EWMA 1/8 over its decode windows, 0 at its start)
    >= P% means "copy" (a code edit that repeats its input), else "chat". This is the A4/A5 split, measured online.
  - `src`: this window's draft source (prompt-lookup or chained window vs. MTP / DFlash2 window).
  - `size[:N]` (N = 4): window tokens > N.
  - `ctx[:N]` (N = 4096): window position >= N.
  - `one`: a single type.
  A switch whose effect outlives its window (kmiss: the swaps; preread) is better judged with `req` or `one`; with
  `src` or `size` the type can change window by window.

## Safe runtime toggling (nothing changes inside a window)

The switches were read once (function statics, or a value baked into the captured window graph). The bandit moves them
through `core/live_knobs.hpp` (atomics), written **only by the serve loop, only between windows**: right before
`ver.run` stages the next window, after the previous round's adaptive-tier join.

- **Rank guard**: the window graph bakes `defer_top()` into `copy_rows_by_weight`'s kernel arguments at capture. With
  `STRATA_BANDIT=top` (and deferral with the heavy split and `_TOP` > 0), each Verifier keeps a mapped word `h_top_`.
  `stage_inputs` (once per window, before the launch, before the store fence) writes `defer_top_now()` into it AND into
  the plan sink (`GpuPlanSink::top`). The kernel reads the word when `top_live != nullptr`; the host pool
  (expert_source.cpp heavy split) reads `ps->top`. So GPU and host use one value per window. Without `STRATA_BANDIT=top`
  the word is not allocated, the kernel gets `nullptr`, `sink.top` stays -1 and the host uses `defer_top()`: as before.
- **k-th miss**: `KMissTier::admits()` = `on()` and not paused. Only the two admission sites (blocking tier `adapt()`,
  asynchronous tier `a_pick`) test `admits()`; `observe()` / `end_round()` still test `on()`, so the miss history is
  kept while paused. Placement only. (The asynchronous tier's round can span windows; a pause read mid-round only
  changes which placement rule that round uses: still placement only.)
- **Pool pre-read**: `ArenaExpertSource::warm()` posts nothing while `preread_off`. The router lookahead still runs
  (`warms()` is unchanged), so the "off" arm is "lookahead without posts", not exactly the env-off engine.
- **Not in the bandit**: pipelined windows (`--pipeline-windows`, layer split: windows overlap) reset every switch to
  its configured value; `--batch` windows never run the serial loop. First and merged windows keep the last arm and are
  not counted.

## Exactness

- `top` is lossy (small). `kmiss` is exact up to rounding. `preread` is bitwise exact.
- With `top` or `kmiss` in the bandit, **which arm runs which window depends on timings**, so the text is not
  reproducible run to run (even with a fixed seed: the rewards are measured times).
- `STRATA_BANDIT_EXACT=1` drops the lossy switch (top) from the bandit (it then stays as its own env var says);
  `=2` also drops kmiss: only bitwise-exact arms, and the text is the text with those switches off.
- `STRATA_BANDIT` and `STRATA_BANDIT_EXACT` join the conversation disk tier's key **only when set** (no re-key of
  existing files), so parked states from bandit runs are kept apart from fixed runs.

## How to enable

```sh
STRATA_VERIFY_DEFER_CPU=1 STRATA_VERIFY_DEFER_CPU_TOP=2 STRATA_ADAPT_KMISS=2,8 \
STRATA_BANDIT=top,kmiss STRATA_BANDIT_LOG=1 ./run-pancake.sh
```

Switch names: `top` (alias `rank`), `kmiss`, `preread` (alias `pre`), `all`. A named switch whose own env var is off
(or that this engine cannot toggle) is dropped and said at start. Other knobs: `STRATA_BANDIT_EPOCH`,
`STRATA_BANDIT_SKIP`, `STRATA_BANDIT_RULE`, `STRATA_BANDIT_DECAY`, `STRATA_BANDIT_SEED`, `STRATA_BANDIT_TYPE`,
`STRATA_BANDIT_EXACT`, `STRATA_BANDIT_LOG`.

Logging:

- start: `strata serve: STRATA_BANDIT=top+kmiss: 4 arms (lossy: rank guard), window types by ..., epochs of 16 ...`
  and `strata verify: STRATA_BANDIT=top: the window graphs read the rank guard from a mapped word ...`
- per epoch (`STRATA_BANDIT_LOG=1`): `strata bandit: copy epoch 7: top 21.304 ms/token over 16 windows (mean 21.9)
  -> top (greedy)`
- per request (always with the bandit): `strata bandit: request windows chat:off*112,top*18 (lookup share 0.04)`, and the
  DONE line gets ` bandit=chat:off*112,top*18` (key=value, last). serve/server.py keeps it in the history record and
  the per-turn JSONL log (`"bandit"` key, present only when the engine sent it).

## Files changed

- `include/strata/core/epoch_bandit.hpp` (new): config parse, window typer, the epoch bandit.
- `include/strata/core/live_knobs.hpp` (new): the runtime knobs.
- `src/core/epoch_bandit_test.cpp` (new) + `CMakeLists.txt` (registered next to adapt_async_default_test, label cpu).
- `include/strata/core/defer_rank.hpp`: `defer_top_now()`.
- `include/strata/kernels/verify_kernels.hpp`, `src/kernels/cuda/verify_kernels.cu`: `copy_rows_by_weight(...,
  top_live = nullptr)`; the kernel reads `*top_live` when it is not null.
- `include/strata/core/verify.hpp`, `src/core/verify.cpp`: `h_top_` / `m_top_` (allocated only with STRATA_BANDIT=top,
  deferral heavy split and _TOP > 0), staged in `stage_inputs`, passed to both `copy_rows_by_weight` calls, freed.
- `include/strata/core/expert_source.hpp`, `src/core/expert_source.cpp`: `GpuPlanSink::top`; the heavy split uses it;
  `warm()` checks `preread_off`.
- `include/strata/program/kmiss_tier.hpp`: `admits()`.
- `src/program/generate.cpp`: setup and start line, per-request reset, arm before the window, observation after the
  round, epoch log, per-request line, DONE `bandit=` field, pipelined-request reset; the two k-th-miss admission sites
  use `admits()`.
- `serve/server.py` (+ `serve/test_server.py`): parse `bandit=`, keep it in history and the turn log when present.
- `src/core/conversation_disk_tier.cpp` (+ `tests/core/conversation_disk_tier_test.cpp`): `STRATA_BANDIT`,
  `STRATA_BANDIT_EXACT` in the key when set.
- `bench/specs/24-a44-bandit.json` (new).

Default path: with `STRATA_BANDIT` unset nothing writes the knobs, no word is allocated, the kernel gets `nullptr`, the
DONE line and the history record are unchanged.

## Tests run

- C++: **not compiled, not run.** `wait_quiet.sh` returned 1 three times (abtest 09-prefill screen running), so no
  host compile was allowed. Reviewed by hand for syntax, types, includes, format strings and -Wall/-Wextra warnings.
- Python: `serve/server.py` and `serve/test_server.py` parse (ast syntax check, no test run).
- `bench/specs/24-a44-bandit.json`: valid JSON.

To run at the build (all host-only, label cpu except the server test):

```sh
cmake --build build-hip --target epoch_bandit_test conversation_disk_tier_test -j4
ctest --test-dir build-hip -R 'epoch_bandit_test|conversation_disk_tier_test' --output-on-failure
python -m unittest serve.test_server.CancelledRead -v
```

`epoch_bandit_test` prints one line per rule for the simulated chat/copy session (bandit cost as a fraction of the best
fixed arm and of the per-type oracle); by hand the expected values are about 0.95 and 1.01 (eps) and 0.95 and 1.005
(ucb). The checks are < 0.97 and < 1.03.

## Spec to run

`bench/specs/24-a44-bandit.json`, `--tier screen` first, cases short + edit, 3 reps. Every arm sets
`STRATA_VERIFY_DEFER_CPU=1` (the fast-hip preset has no deferral; the rank guard needs it, and test 01's A4 numbers were
on a deferred base). Fixed arms: `defer` (the bandit's "off"), `defer-top2` (best on edit), `defer-kmiss`. Bandit arms:
`bandit-top`, `bandit-top-kmiss`, `bandit-exact` (kmiss + preread with `STRATA_BANDIT_EXACT=1`).

Promote if `bandit-top` is within noise of `defer` on short AND of `defer-top2` on edit (the per-case best fixed arm on
both). The bandit learns over the process (the abtest warm-up request and earlier reps teach it), so rep 1 is its
slowest; read the `strata bandit:` lines to see which arm each type settled on.

## Expected gain and risk

- Expected: keep A4's edit gain without its short-chat loss. On a mixed session est. +2-5% (MOD_PLAN); on the bench,
  close to the per-case best fixed arm minus the exploration cost (the unit test's simulation: about 1-3% over the
  per-type oracle with 4 arms and epochs of 16).
- Risks:
  - Exploration cost: with eps 0.1 about 1 epoch in 10 runs a random arm; each new arm costs one epoch per type.
  - Short requests may end before a type finishes its exploration (2 arms: 36 windows per type; 4 arms: 72).
  - The `req` typer needs a few lookup windows before a request counts as "copy"; an edit request's first windows run
    the chat arm.
  - Non-reproducible text with lossy / rounding arms (see Exactness). Use `STRATA_BANDIT_EXACT` for exact runs.
  - Kernel change (`copy_rows_by_weight` gets one more pointer argument, null by default): needs the HIP and CUDA
    builds. With the bandit on, the kernel reads one mapped int per block (thread 0 only).
  - The rank-guard path assumes nothing stages a window before the serve loop sets its arm. That holds for the serial
    loop (the chained stages' `prelaunch` runs inside `ver.run`); the pipelined path is excluded.

## What is left

- Build (HIP; CUDA for the kernel signature) and run `epoch_bandit_test`, `conversation_disk_tier_test`,
  `serve/test_server.py` in CI; run the screen spec.
- If the bandit helps: try `STRATA_BANDIT_TYPE=src` and `ucb`, and an epoch of 8 for short sessions.
- Athena also tunes aggressiveness (not only on/off); a later arm set could include `_TOP=1/2/3` or kmiss `k,W` values.
- The async adaptive tier (`--adapt-async`) was not looked at closely: kmiss is paused per call, which is safe
  (placement only) but the reward attribution there is looser.
