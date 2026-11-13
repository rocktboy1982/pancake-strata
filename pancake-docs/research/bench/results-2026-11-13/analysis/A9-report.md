# A9: pre-stage experts while a tool runs (CPU replay of tool gaps)

Date 2026-11-13. CPU only, read-only on the opencode store and on Strata source. Saved by the main session from the agent's final text (the agent could not write this file).

## What A9 is
- Plan item A9 (MOD_PLAN.md ~line 1867): pre-stage the next turn's experts while a tool runs. Expected gain 1-3% of next-turn decode (estimate). Status: coded opt-in as `STRATA_IDLE_ADAPT`.
- Code: `src/program/generate.cpp` ~14928-14959. With `STRATA_IDLE_ADAPT=N` (default 0), after 150 ms with no input, up to N rounds of the blocking adaptive tier run, so routing counts are in VRAM when the next turn starts. Placement only.
- `Verifier::prestage` (`src/core/verify.cpp` ~5905) stages host-side pipelined-window records (PLE token rows), not expert weights. It is a different mechanism.
- No general "stage next turn's experts" path exists in the code.

## Tool durations (opencode store, 4,645 timed tool calls)
- All tools: median 0.885 s, p90 2.42 s, total 4.66 h.
- Excluding `task` (which wraps child sessions): n=4,518, total 3.39 h.
- `task`: n=127, total 1.26 h (includes child model time, excluded from idle time).

## Gaps
- Tool end to next assistant step: median 0.051 s, p90 1.40 s, total 2.56 h. Tool windows are the only idle window.

## Wall base
- Generation time 26.65 h; non-task tool time 3.39 h; wall 30.04 h. Tool share 11.3% (10.4% counting the gaps as idle).

## Staging cost
- The plan gives one figure: a 1.4 ms layer moves ~18 MB (about 1.3 ms per layer at 13.4 GB/s PCIe). The requested 0.5-2 s per layer is 400-1,500x that and is not supported by the plan's bandwidth. Treated as sensitivity only.

## Bound
- Time bound, independent of staging cost: all non-task tool time = 11.3% of wall. Pre-staging cannot hide more than the tool time.
- At the plan's 1.3 ms per layer, hidable time is 0.26% of wall at most (48-unit reading). At the requested values it is 1.7-11.7%, not supported by the hardware.

## Decode-side estimate
- Coverage 20-35% of misses (plan's offline bench trace) x time bound 11.3% = about 2.3-4.0% of wall. The plan's 1-3% sits inside this range.
- Assumptions: tool windows long enough (median 0.9 s vs about 11 ms of copies per window); coverage from bench traces, not agent traces; copy stream does not slow the tool; enough VRAM slots (4,236 in preset); user idle excluded.

## Caveats
- Tool durations are opencode client-side waits, not GPU measurements.
- 75 error tool calls included. Expert size inferred, not measured.
- Store covers 191 sessions on this machine; not a controlled benchmark.

## GPU measurement still needed
1. Per-layer staging cost on the RX 7900 XT: bytes per IQ2_XS expert, copy-stream throughput, per-layer time with the stream active.
2. Replay real tool gaps (sleep for recorded durations, capped) with `STRATA_IDLE_ADAPT` on and off, A/B/B/A with first run discarded, on the night queue (owner rule). Record next-turn first-window copies, decode tok/s, TTFT.
3. Measure coverage of the staged set against the next turn's real misses from the turn log.
4. Check the idle hook does not slow the tool through CPU or PCIe contention.

## Bottom line
- A9 as coded is `STRATA_IDLE_ADAPT`, not a general next-turn expert prefetch.
- Realistic gain from the plan's coverage: about 2-4% of wall. Consistent with the plan's 1-3%. A bound, not a GPU measurement.
