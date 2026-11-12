# Night queue results (night-queue-2026-11-11)

Written 2026-11-12 09:40:07 by tools/pancake/night_queue.py from `state.json`. Numbers only, no interpretation; nothing here is a finding until it is read against the pass-by-pass tables. **Pass 1 of a decode job is the warm-up and is excluded**; only the last 2 passes after it are kept (median over the requests of those passes). Estimated durations are the plan's, not measurements.

Status: done 1; 42 not started.

## Jobs

| job | group | engine | delta (env / args) | status | wall min | est min | stop problem |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ttft-avalon-iq2xs | 2-ttft | avalon-iq2xs | preset as is | done | 2.0 | 20 |  |

## TTFT (tools/pancake/ttft_bench.py)

Scenario A: single prompts, run 1 of each size discarded by the tool, medians of the rest; TTFT is the client's clock, the engine columns are the server's own `timings`. Scenario B: the 10-turn agent session.

**ttft-avalon-iq2xs** (load start 5.54, end 3.26, warning True)

| size | status | prompt tokens | TTFT median s | min | max | engine prompt ms | engine prompt tok/s | kept runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8K | measured | 8157 | 6.382 | 6.263 | 6.426 | 6365.7 | 1281.6 | 3 |
| 32K | failed at run 1: no token streamed | None | None | None | None | None | None | 0 |
| 64K | failed at run 1: ConnectionRefusedError(111, 'Connection ref | None | None | None | None | None | None | 0 |
| 100K | failed at run 1: ConnectionRefusedError(111, 'Connection ref | None | None | None | None | None | None | 0 |

Session: total TTFT over the 10 turns (median of complete sessions) None s; reuse of prompt tokens on follow-up turns None % (source not reported); complete sessions 0/3.

| turn | n | TTFT s | prompt tokens | reused | read | re-read |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | None | None | None | None | None |

