# Park rewind and the divergence log (MOD_PLAN 17.0, lever 2 of `pancake-docs/TTFT_LEVERS.md`)

2026-11-11, branch `dev/p17/parkrewind` (from `dev/phase16-academic` `a4500666e`). Coded, NOT built, NOT measured: this was
written while the GPU was busy with benchmarks, so no compiler, engine or server was started. Only small pure-Python tests
may have run (see "What ran" at the end).

## The problem

The 2026-11-11 analysis (`TTFT_LEVERS.md`, sections 1 and 2.2) found in the logs of real traffic on this box:

- 6 of 46 warm agent turns at 34-82K tokens did not continue the live conversation. They fell back to the turn checkpoint,
  read 0.2-1.8K tokens again, and parked a state they never used again. One turn spent 1,417 ms of 2,816 ms on that park
  (log numbers, `TTFT_LEVERS.md` section 0, point 4).
- A park runs inside the next request's prompt time on HIP, because HIP has no async park (`park_async_on()` returns
  false): about 110 ms below 4K tokens, 212 ms at 25K, 0.3-1.4 s at 50-77K tokens (log numbers).
- Nobody knew why the re-rendered reply differed from the generated tokens (measurement M3 in the analysis).

## 1. `STRATA_PARK_REWIND_MAX=N` (opt-in, default off)

Where it is decided: `src/program/generate.cpp`, the park decision of the serve loop, just before the `park_current`
calls (search `rewind_skip`). The rule itself is a pure function, `strata::program::conv_cache::rewind_no_park`
(`include/strata/program/conv_cache.hpp`), covered by `src/program/conv_cache_test.cpp`.

What it does. When a request resumes from one of the live conversation's own checkpoints, and the live conversation is at
most N tokens longer than that checkpoint, the engine does not park the live conversation first. Such a request rewrites
every cell from the checkpoint on, so the copy would hold a branch the client has already left: the last reply, which the
chat template rendered again (typically without its reasoning). The next turn would drop that copy as superseded.

When it does NOT apply (the park happens as before): the request continues the live state exactly (it never parked); a
parked conversation is restored, a checkpoint is borrowed, or a batch slot supplies the start of the prompt; the
checkpoint is one a batch slot keeps; the live conversation has pictures; the checkpoint is not a prefix of the live
tokens (another conversation); more than N tokens would be left behind. N unset or 0 means off, so the default path is
unchanged.

What it costs. The tokens after the checkpoint (the last reply and its header) are not kept in the cache. A client that
returns to exactly that branch later reads them again. For an agent session that moves forward this never happens.

How to choose N. The engine prints `conversation cache: not parked (STRATA_PARK_REWIND_MAX): ... checkpoint at X of Y tokens
(Z left behind)`. Z is the last reply plus its header, so a reasoning reply of several thousand tokens needs a larger N
than a short answer. The fork author used 4096; we have no measurement of our own, so start with 8192 and read Z in the log.

Where it comes from. The idea is 0xluk's `53cecdeee9` (also Xiaomi2008 `8e4ccbe90d` and UltimateBoomer `5d0a9e786c` as
`--conversation-cache-skip-tail`). It was written here from their descriptions and commit message, not copied; credits are
in README section 9 and `RESEARCH_AND_FINDINGS.md` section 4. The differences from the description: the decision is a pure
function with its own unit test, and it also refuses a checkpoint lent by a batch slot (`STRATA_SLOT_CKPT_MOVE`) and a
live conversation with pictures.

Unverified (reading the code only): that the state left in place is exactly what a restore from the checkpoint expects.
This is the same situation as the existing `pin_sibling` path (a request that resumes at a pinned prefix of the live
conversation and is not parked), which has been in use, but nothing here has run.

## 2. `STRATA_PARK_DIVERGENCE_LOG=1` (opt-in, default off; the M3 diagnostic)

See the second half of this note (added with the divergence commit).

## 3. Measurement plan

Question: does the skipped park save time, and how often does the case happen?

1. Baseline first, no new switch. Run `tools/pancake/ttft_bench.py --scenario b --turns 10 --sessions 3` (the 10-turn agent
   session; item 17.0's harness, branch `dev/p17/ttftbench`) against the daily engine. Then run `tools/pancake/ttft_audit.py` on
   the turn log and the engine log. The effect can only appear if the baseline shows turns of class `fallback` (the request
   continued the previous conversation from a checkpoint at or after the previous prompt end) or parks followed by
   "dropped 1 superseded copy". If the baseline has none, there is nothing for this switch to save: record that, stop.
2. Add a side request between turns (a short title or token-count request, as an agent client sends; `ttft_bench` scenario B has
   none, so use the lead's M2 variant or `replay_turns.py` with a spec that interleaves one). It makes the conversation leave
   the live state, which is when the fallback and its park happen.
3. Same session with `STRATA_PARK_REWIND_MAX=8192` (and, to read Z, the engine log). Compare per turn: TTFT of the
   fallback turns (`ttft_bench` per-turn medians), the number of `parked N tokens in X ms` lines in the engine log, and the
   audit's class counts.
4. Exactness: the replies of the two runs must match token for token at temperature 0 with the same seed (a skipped park
   changes no state the request reads).
5. Bar to keep it: at least 2 s saved on each fallback turn, or at least 2% of the total session TTFT (`ttft_bench` prints the
   total over the session). Below that, leave it off and say so here. Whatever the result, record the numbers with the build,
   the model, the context size and the number of sessions.

Effect expected from the logs, not a prediction of gain: a park of 0.2-1.4 s per fallback turn at 25-77K tokens. That is below
the 2 s bar per turn at those sizes, so the 2% of session TTFT bar is the likely one to decide, and only if fallbacks are
frequent.

## What ran

Nothing built. The unit test (`conv_cache_test`) is written and registered already (it is an existing target); build and
run it later:

    cmake --build <build-dir> --target conv_cache_test && <build-dir>/conv_cache_test

The server has to be rebuilt too (`strata serve` in the HIP build) before the engine path can be tried.
