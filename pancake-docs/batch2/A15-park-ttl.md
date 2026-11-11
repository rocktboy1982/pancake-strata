# A15 - TTL / idleness rule for parked sub-agent sessions

Branch `work/a15-park-ttl` (from `origin/dev/batch-2`). Status: coded, opt-in, exact; not built into the engine, not run on a GPU.

## What

When the conversation cache needs room it evicts the oldest parked conversation (A13's disk tier keeps tool-pending
ones in RAM longer). With an agent and its sub-agents that rule is wrong in one common case: the main agent is parked
first (it starts the sub-agent), so it is the OLDEST entry, and the short-lived sub-agent branches parked after it push
it out. The next main-agent turn then re-reads its whole history (54-62K tokens on the 11-09 trace, ~25 s) while the
branches it displaced would never have been resumed.

Continuum (arXiv 2511.02230) gives each paused agent job a time-to-live: it is pinned in the cache for the predicted
duration of its pending tool call and released when the TTL runs out. This item brings that to the parked-conversation
cache as an eviction-order rule:

Rank of a movable entry (the pinned shared prefix of A12 and the guarded donor stay out, as before); lowest rank goes
first, the oldest within a rank:

| rank | entry |
|---|---|
| 0 | idle sub-agent branch: idle >= `STRATA_PARK_TTL_S` and not waiting for a tool result |
| 1 | sub-agent branch not yet idle that long |
| 2 | any other conversation (the main agent's) |
| 3 | last reply ended in a tool call and parked less than `STRATA_PARK_TTL_HOLD_S` ago (Continuum's pin for the predicted tool duration; fixed prediction, 60 s default) |

"Sub-agent branch" = an entry whose live tokens share at least min(`STRATA_PARK_TTL_SHARE`, half its length) leading tokens
(the system prompt + tools; default 512) with a LONGER parked entry of the same steering mode, and whose length is at most
`STRATA_PARK_TTL_SUB_RATIO` (default 0.5) of that entry. That is how the existing code tells them apart without a session id:
a sub-agent starts from the main agent's prefix (the same fact A12's auto-pin and `borrows()` use) and stays short; the
engine has no explicit agent id, and the OpenAI-style requests carry none. A wrong classification only changes which
entry is evicted first, never what any entry contains.

With the disk tier (A13) the evicted branch is demoted to disk by the same `spill` as before. `STRATA_PARK_TRUNCATE`
victim choice uses the same ranking. Tool detection reuses the disk tier's rule (`conversation_ended_in_tool_call`,
`STRATA_TOOL_CALL_TOKEN`, default 248058) but does not need the disk tier to be on.

## Exactness

Cache policy only: `ConversationCache::pick_victim()` chooses the entry; no snapshot, checkpoint or restore path is
touched, so outputs equal the default (the cache returns the same bytes or re-reads the same tokens, as for any eviction
order). Default path: with `STRATA_PARK_TTL_S` unset the code in `pick_victim()` is the old two loops verbatim; the only
addition is a park timestamp pushed per `put` (0, no clock call, while the rule is off). Not lossy.

## How to enable

```
STRATA_PARK_TTL_S=30            # seconds an idle branch may sit before rank 0 (0/unset: off)
STRATA_PARK_TTL_HOLD_S=60       # tool-pending hold, seconds (default 60)
STRATA_PARK_TTL_SHARE=512       # shared leading tokens that make a branch (default 512)
STRATA_PARK_TTL_SUB_RATIO=0.5   # branch length at most this fraction of the longer entry (default 0.5)
```
The engine prints `conversation cache: park TTL on ...` at start. Works with `STRATA_AUTO_PIN`/`STRATA_PIN_PARK_TOKENS` (A12:
the pinned entry itself is never movable) and with `--conversation-disk-dir` (A13). Useful only with `--conversation-cache-mib`
small enough that evictions happen.

## Files changed

- `include/strata/core/conversation_cache.hpp`: `ParkTtl`, `set_park_ttl()`, `ttl_evictions()`, `pick_victim()` (shared by
  `evict_oldest` and `truncate_oldest`), `is_branch()`, parallel `parked_at_` stamps (put / erase / clear).
- `src/program/generate.cpp`: env parsing and `set_park_ttl` after the disk-tier setup.
- `src/core/conversation_cache_test.cpp`: new test block (registered already: `conversation_cache_test`).
- `tools/pancake/replay_turns.py` + `test_replay_turns.py`: opt-in `--shared-system N` (or spec `replay.shared_system`); default unchanged.
- `bench/specs/24-a15-park-ttl.json`, `docs/DETAILS.md` (one table row).

## Tests run

Nothing was compiled or run: `wait_quiet.sh` returned 1 (a benchmark, `abtest.py 09-prefill`, was running) on three
calls in a row, so per the brief the code was only re-read (syntax, types, includes, test expectations traced by hand).
Run at 02:00: `cmake --build <dir> --target conversation_cache_test && ctest -R conversation_cache_test` (header-only
test, no GPU; expect `conversation_cache_test: N checks passed`), and `python -m pytest tools/pancake/test_replay_turns.py`
(`py_compile` of both python files passed).

## Spec to run

1. `tools/pancake/abtest.py bench/specs/24-a15-park-ttl.json --tier screen`: side effects only (decode speed, RAM); the rule never decides anything
   on one-shot requests, expect no difference and identical texts.
2. The real test, the agent replay:
   `python tools/pancake/replay_turns.py bench/specs/24-a15-park-ttl.json --out bench/results/<day>-a15-replay`
   (arms base, ttl30; `--arms base1280,ttl30-1280` pins the 11-09 budget). The spec sets `shared_system: 1024`, because the
   stock replay gives every conversation its own system text, so nothing shares a prefix and there are no branches to rank.
   Read: cold re-reads (count, seconds) of the long conversation, total prompt time, and which entries the `evictions=` park lines
   removed. Do not read the 11-09 trace as a real sub-agent log: its short requests are side probes, not sub-agent sessions;
   it only reproduces the pressure shape (long conversation + short interleaved ones). A real opencode sub-agent session
   (main + Task tool calls) with the A12 auto-pin is the proper trace once one is recorded.

## Expected gain and risk

Gain: when B (the main agent, 55-62K) is parked beside short branches under cache pressure, B stops being the oldest
victim: each saved eviction is one cold re-read (~25 s at 0.41 ms/token). Zero when the cache holds everything (the 7 GiB+ budgets)
or when no entry shares a prefix. Estimate only; nothing measured.

Risks: (1) a branch that does come back after the TTL finds its entry gone (a re-read of a short conversation: cheap);
(2) misclassification of two long conversations of one agent family as main+branch when one is under half the other (the
shorter one is evicted first: LRU-ish, no correctness effect); (3) `is_branch` is O(entries x tokens) per eviction decision (4 slots,
~60K ids: sub-millisecond, only when the rule is on); (4) the hold is a fixed guess of the tool duration, not Continuum's
learned per-tool prediction.

## Left

- Learned per-tool hold (tool name -> measured duration from the turn log) instead of the fixed 60 s.
- An explicit agent/session id from the server (e.g. a request header) would replace the prefix-and-length heuristic.
- A real sub-agent trace for the replay; the GPU runs above.
