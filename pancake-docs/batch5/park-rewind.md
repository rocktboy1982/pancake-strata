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

Code: `serve/park_divergence.py` (pure helpers and a tracker), hooked into `StrataEngine.generate` in `serve/server.py`
(search `self.divergence`). Tests: `serve/test_park_divergence.py` (21 tests, including `StrataEngine.generate` with a fake
engine pipe). It is in the server, not the engine, because the engine only sees token ids; the server also has the tokenizer.

What it does. The server remembers the last request's prompt plus the tokens the engine streamed back. When the engine
reports `RESUME n` for the next request and the new prompt does not start with all of those tokens, it prints one line:

    [strata] park divergence {"diverge_at":61234,"live_len":61902,"prompt_len":62500,"prev_prompt_len":61200,
      "prev_gen_len":702,"reply_offset":34,"where":"reply-start","resume":58880,"live_ids":[...],"new_ids":[...],
      "live_text":"...","new_text":"...","reply_has_think_close":true,"reasoning_dropped_guess":true}

(one line in the log; wrapped here.) `where` says whether the prompts part inside the old history (`history`: an earlier message
changed, or this is another conversation, such as a side request) or inside the old reply (`reply-start`, `reply-middle`,
`reply-end`). `reasoning_dropped_guess` is a guess, not a measurement: the previous reply had a think-close tag and its first
24 characters are not in the new prompt's version of that reply. The engine's own `resume` is next to it, so a line with
`resume` far below `diverge_at` is a fallback to a checkpoint.

This log contains prompt and reply text. `live_text` and `new_text` are decoded windows of 4 tokens before and 12 tokens after
the divergence, cut to `STRATA_PARK_DIVERGENCE_CHARS` characters (default 48; 0 prints ids only). Look at the file before it is
shared. Nothing is recorded or printed when the switch is off.

Limits: only the plain one-engine path is followed. `--batch`, DeepSeek `--parallel`, picture requests and split serving are
not; the next request after one of those is compared with nothing. The "live" tokens are the prompt plus the tokens streamed
back; if a client stops a reply early the engine may hold a few more. An error inside the log turns the log off and never
changes a request.

How to read it for lever 3: collect the lines of a session with fallbacks (the `fallback` class of `ttft_audit.py`). A cluster of
`reply-start` lines with `reasoning_dropped_guess: true` means the client or template drops the reasoning (`keep_reasoning`
is the lever); `reply-end` lines of one or two tokens mean a whitespace or end-of-turn token rendered differently (a template
fix); `history` lines from a main conversation mean the client edited an earlier message. None of this has been seen yet.

## 3. Measurement plan

Question: does the skipped park save time, and how often does the case happen?

1. Baseline first, no rewind switch (the divergence log may be on: it changes no request). Run `tools/pancake/ttft_bench.py --scenario b --turns 10 --sessions 3` (the 10-turn agent
   session; item 17.0's harness, branch `dev/p17/ttftbench`) against the daily engine. Then run `tools/pancake/ttft_audit.py` on
   the turn log and the engine log. The effect can only appear if the baseline shows turns of class `fallback` (the request
   continued the previous conversation from a checkpoint at or after the previous prompt end) or parks followed by
   "dropped 1 superseded copy". If the baseline has none, there is nothing for this switch to save: record that, stop.
2. Add a side request between turns (a short title or token-count request, as an agent client sends; `ttft_bench` scenario B has
   none, so use the lead's M2 variant or `replay_turns.py` with a spec that interleaves one). It makes the conversation leave
   the live state, which is when the fallback and its park happen.
3. Same session with `STRATA_PARK_REWIND_MAX=8192` (and, to read Z, the engine log). Compare per turn: TTFT of the
   fallback turns (`ttft_bench` per-turn medians), the number of `parked N tokens in X ms` lines in the engine log, and the
   audit's class counts. The divergence lines of the baseline say which fallbacks exist and why.
4. Exactness: the replies of the two runs must match token for token at temperature 0 with the same seed (a skipped park
   changes no state the request reads).
5. Bar to keep it: at least 2 s saved on each fallback turn, or at least 2% of the total session TTFT (`ttft_bench` prints the
   total over the session). Below that, leave it off and say so here. Whatever the result, record the numbers with the build,
   the model, the context size and the number of sessions.

Effect expected from the logs, not a prediction of gain: a park of 0.2-1.4 s per fallback turn at 25-77K tokens. That is below
the 2 s bar per turn at those sizes, so the 2% of session TTFT bar is the likely one to decide, and only if fallbacks are
frequent.

## What ran

Python only, at low priority: `python3 -I -m pytest -q serve/test_park_divergence.py` (21 passed), `serve/test_logprobs.py` and
`serve/test_think_probe.py` (35 passed, they exercise `StrataEngine.generate`). Nothing in C++ was compiled or run. Build and run
the C++ test later (`conv_cache_test` is an existing target):

    cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
          -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DSTRATA_BUILD_TESTS=ON
    cmake --build build-hip --target conv_cache_test -j2 && build-hip/conv_cache_test    # expect PASS
    cmake --build build-hip --target strata -j2                                           # the engine, for the real switch

The divergence log needs no rebuild (the server is Python): `STRATA_PARK_DIVERGENCE_LOG=1`.
