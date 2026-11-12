# Night queue results (night-queue-2026-11-11)

Written 2026-11-12 02:43:00 by tools/pancake/night_queue.py from `state.json`. Numbers only, no interpretation; nothing here is a finding until it is read against the pass-by-pass tables. **Pass 1 of a decode job is the warm-up and is excluded**; only the last 2 passes after it are kept (median over the requests of those passes). Estimated durations are the plan's, not measurements.

Status: done 43; 0 not started.

## Jobs

| job | group | engine | delta (env / args) | status | wall min | est min | stop problem |
| --- | --- | --- | --- | --- | --- | --- | --- |
| defer-iq2xs-01A | 1-defer | pancake-iq2xs | preset as is | done | 2.3 | 8 |  |
| defer-iq2xs-02B | 1-defer | pancake-iq2xs | STRATA_VERIFY_DEFER_CPU=(removed) | done | 4.7 | 8 |  |
| defer-iq2xs-03C | 1-defer | pancake-iq2xs | STRATA_VERIFY_DEFER_AUTO=1, STRATA_VERIFY_DEFER_AUTO_MIN_T=256 | done | 4.9 | 8 |  |
| defer-iq2xs-04C | 1-defer | pancake-iq2xs | STRATA_VERIFY_DEFER_AUTO=1, STRATA_VERIFY_DEFER_AUTO_MIN_T=256 | done | 5.6 | 8 |  |
| defer-iq2xs-05B | 1-defer | pancake-iq2xs | STRATA_VERIFY_DEFER_CPU=(removed) | done | 5.4 | 8 |  |
| defer-iq2xs-06A | 1-defer | pancake-iq2xs | preset as is | done | 4.6 | 8 |  |
| defer-iq2xs-07A | 1-defer | pancake-iq2xs | preset as is | done | 4.4 | 8 |  |
| defer-iq2xs-08B | 1-defer | pancake-iq2xs | STRATA_VERIFY_DEFER_CPU=(removed) | done | 4.7 | 8 |  |
| defer-iq2xs-09C | 1-defer | pancake-iq2xs | STRATA_VERIFY_DEFER_AUTO=1, STRATA_VERIFY_DEFER_AUTO_MIN_T=256 | done | 5.0 | 8 |  |
| defer-q2-01A | 1-defer | pancake-q2 | preset as is | done | 5.9 | 8 |  |
| defer-q2-02B | 1-defer | pancake-q2 | STRATA_VERIFY_DEFER_CPU=(removed) | done | 5.5 | 8 |  |
| defer-q2-03C | 1-defer | pancake-q2 | STRATA_VERIFY_DEFER_AUTO=1, STRATA_VERIFY_DEFER_AUTO_MIN_T=256 | done | 4.5 | 8 |  |
| defer-q2-04C | 1-defer | pancake-q2 | STRATA_VERIFY_DEFER_AUTO=1, STRATA_VERIFY_DEFER_AUTO_MIN_T=256 | done | 4.4 | 8 |  |
| defer-q2-05B | 1-defer | pancake-q2 | STRATA_VERIFY_DEFER_CPU=(removed) | done | 4.1 | 8 |  |
| defer-q2-06A | 1-defer | pancake-q2 | preset as is | done | 4.3 | 8 |  |
| defer-q2-07A | 1-defer | pancake-q2 | preset as is | done | 4.4 | 8 |  |
| defer-q2-08B | 1-defer | pancake-q2 | STRATA_VERIFY_DEFER_CPU=(removed) | done | 4.3 | 8 |  |
| defer-q2-09C | 1-defer | pancake-q2 | STRATA_VERIFY_DEFER_AUTO=1, STRATA_VERIFY_DEFER_AUTO_MIN_T=256 | done | 4.4 | 8 |  |
| ttft-pancake-iq2xs-a | 2-ttft | pancake-iq2xs | preset as is | done | 7.2 | 15 |  |
| ttft-pancake-iq2xs-100k-trace | 2-ttft | pancake-iq2xs | STRATA_TRACE=1 | done | 6.0 | 11 |  |
| ttft-rdna3-iq2xs | 2-ttft | rdna3-iq2xs | preset as is | done | 10.3 | 21 |  |
| ttft-avalon-iq2xs | 2-ttft | avalon-iq2xs | preset as is | done | 3.7 | 20 |  |
| lever-01A | 3-lever | pancake-iq2xs | preset as is | done | 4.7 | 11 |  |
| lever-02B | 3-lever | pancake-iq2xs | --conversation-cache-min-tokens 8192 | done | 5.3 | 11 |  |
| lever-03C | 3-lever | pancake-iq2xs | --conversation-cache-mib 6144 | done | 5.0 | 11 |  |
| lever-04D | 3-lever | pancake-iq2xs | STRATA_PARK_TRUNCATE=1 | done | 4.9 | 11 |  |
| lever-05E | 3-lever | pancake-iq2xs | STRATA_PARK_TRUNCATE=1, --conversation-cache-min-tokens 8192, --conversation-cache-mib 6144 | done | 6.0 | 11 |  |
| lever-06F | 3-lever | pancake-iq2xs | STRATA_PARK_REWIND_MAX=8192, STRATA_PARK_DIVERGENCE_LOG=1 | done | 6.6 | 11 |  |
| lever-07A | 3-lever | pancake-iq2xs | preset as is | done | 5.2 | 11 |  |
| kvring-01S | 4-kvring | pancake-iq2xs | preset as is | done | 7.0 | 13 |  |
| kvring-02T | 4-kvring | pancake-iq2xs | --kv-resident 49152 | done | 7.3 | 13 |  |
| kvring-03R | 4-kvring | pancake-iq2xs | STRATA_KV_RING_ADAPT=1, --kv-resident 49152 | done | 6.2 | 13 |  |
| kvring-04P | 4-kvring | pancake-iq2xs | STRATA_KV_RING_ADAPT=1, --kv-resident 49152, --max-context 65536 | done | 3.9 | 13 |  |
| kvring-05P | 4-kvring | pancake-iq2xs | STRATA_KV_RING_ADAPT=1, --kv-resident 49152, --max-context 65536 | done | 5.4 | 13 |  |
| kvring-06R | 4-kvring | pancake-iq2xs | STRATA_KV_RING_ADAPT=1, --kv-resident 49152 | done | 7.4 | 13 |  |
| kvring-07T | 4-kvring | pancake-iq2xs | --kv-resident 49152 | done | 6.6 | 13 |  |
| kvring-08S | 4-kvring | pancake-iq2xs | preset as is | done | 6.4 | 13 |  |
| prose-base-01A | 5-prose | pancake-iq2xs | preset as is | done | 3.8 | 8 |  |
| prose-avalon | 5-prose | avalon-iq2xs | preset as is | done | 4.7 | 8 |  |
| prose-var-01B | 5-prose | pancake-iq2xs | --spec-min-p 0 | done | 5.0 | 8 |  |
| prose-var-02C | 5-prose | pancake-iq2xs | --spec 2, --spec-min-p 0 | done | 5.2 | 8 |  |
| prose-var-03D | 5-prose | pancake-iq2xs | STRATA_SPEC_COST=1 | done | 5.0 | 8 |  |
| prose-var-04A | 5-prose | pancake-iq2xs | preset as is | done | 5.6 | 8 |  |

## Decode (tok/s, the five cmp_bench prompts, thinking off, greedy)

Per pass. `role` = warm-up (excluded) / kept / extra (not used). Overall, code (js, lru) and prose (story, tips, moe) are medians over the requests of that pass; acceptance = accepted / drafted tokens; prompt tok/s and TTFT are the engine's own (`timings.prompt_per_second`, `prompt_ms`) on these short prompts.

| job | pass | role | overall | code | prose | accept % | prompt tok/s | TTFT s | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| defer-iq2xs-01A | 1 | warm-up (excluded) | 96.9 | 129.5 | 94.5 | 68.1 | 117 | 0.63 |  |
| defer-iq2xs-01A | 2 | extra | 105.2 | 130.7 | 102.1 | 67.3 | 38 | 0.18 |  |
| defer-iq2xs-01A | 3 | kept | 106.9 | 132.8 | 105.5 | 71.1 | 38 | 0.18 |  |
| defer-iq2xs-01A | 4 | kept | 104.5 | 129.6 | 101.0 | 66.2 | 37 | 0.19 |  |
| defer-iq2xs-02B | 1 | warm-up (excluded) | 104.7 | 121.1 | 103.8 | 67.9 | 108 | 0.64 |  |
| defer-iq2xs-02B | 2 | extra | 105.5 | 118.8 | 96.6 | 66.0 | 36 | 0.20 |  |
| defer-iq2xs-02B | 3 | kept | 103.8 | 124.7 | 101.6 | 67.6 | 37 | 0.19 |  |
| defer-iq2xs-02B | 4 | kept | 104.9 | 123.8 | 98.6 | 66.6 | 36 | 0.19 |  |
| defer-iq2xs-03C | 1 | warm-up (excluded) | 84.8 | 102.8 | 74.0 | 67.3 | 112 | 0.65 |  |
| defer-iq2xs-03C | 2 | extra | 90.7 | 98.2 | 90.7 | 69.9 | 36 | 0.20 |  |
| defer-iq2xs-03C | 3 | kept | 96.5 | 106.2 | 96.5 | 67.6 | 36 | 0.19 |  |
| defer-iq2xs-03C | 4 | kept | 93.8 | 119.9 | 81.1 | 69.1 | 36 | 0.19 |  |
| defer-iq2xs-04C | 1 | warm-up (excluded) | 93.6 | 101.8 | 71.7 | 67.5 | 99 | 0.69 |  |
| defer-iq2xs-04C | 2 | extra | 95.8 | 103.4 | 90.7 | 65.4 | 35 | 0.20 |  |
| defer-iq2xs-04C | 3 | kept | 92.8 | 100.3 | 85.2 | 67.3 | 36 | 0.19 |  |
| defer-iq2xs-04C | 4 | kept | 86.8 | 98.1 | 77.3 | 69.2 | 36 | 0.20 |  |
| defer-iq2xs-05B | 1 | warm-up (excluded) | 103.0 | 121.5 | 102.8 | 68.2 | 96 | 0.71 |  |
| defer-iq2xs-05B | 2 | extra | 106.2 | 124.0 | 98.2 | 71.2 | 36 | 0.20 |  |
| defer-iq2xs-05B | 3 | kept | 100.1 | 118.2 | 99.4 | 67.7 | 34 | 0.20 |  |
| defer-iq2xs-05B | 4 | kept | 102.7 | 118.0 | 99.5 | 68.0 | 35 | 0.20 |  |
| defer-iq2xs-06A | 1 | warm-up (excluded) | 100.6 | 123.8 | 98.6 | 67.8 | 118 | 0.62 |  |
| defer-iq2xs-06A | 2 | extra | 103.8 | 132.8 | 101.0 | 66.6 | 37 | 0.19 |  |
| defer-iq2xs-06A | 3 | kept | 104.8 | 130.8 | 103.6 | 66.4 | 38 | 0.19 |  |
| defer-iq2xs-06A | 4 | kept | 104.6 | 131.1 | 103.5 | 66.8 | 37 | 0.19 |  |
| defer-iq2xs-07A | 1 | warm-up (excluded) | 106.8 | 129.4 | 104.3 | 67.7 | 117 | 0.62 |  |
| defer-iq2xs-07A | 2 | extra | 99.9 | 130.1 | 98.2 | 68.0 | 36 | 0.19 |  |
| defer-iq2xs-07A | 3 | kept | 106.3 | 131.4 | 100.1 | 69.2 | 36 | 0.19 |  |
| defer-iq2xs-07A | 4 | kept | 105.1 | 131.2 | 98.7 | 68.0 | 37 | 0.19 |  |
| defer-iq2xs-08B | 1 | warm-up (excluded) | 108.4 | 118.4 | 104.5 | 68.2 | 105 | 0.66 |  |
| defer-iq2xs-08B | 2 | extra | 98.5 | 123.9 | 97.0 | 67.2 | 36 | 0.20 |  |
| defer-iq2xs-08B | 3 | kept | 109.5 | 121.5 | 92.1 | 66.9 | 36 | 0.19 |  |
| defer-iq2xs-08B | 4 | kept | 105.6 | 118.4 | 94.8 | 67.1 | 35 | 0.20 |  |
| defer-iq2xs-09C | 1 | warm-up (excluded) | 94.9 | 103.2 | 84.1 | 67.1 | 110 | 0.63 |  |
| defer-iq2xs-09C | 2 | extra | 94.8 | 105.8 | 94.7 | 67.8 | 35 | 0.20 |  |
| defer-iq2xs-09C | 3 | kept | 97.4 | 110.2 | 91.5 | 69.1 | 35 | 0.20 |  |
| defer-iq2xs-09C | 4 | kept | 81.0 | 89.5 | 73.1 | 67.8 | 35 | 0.20 |  |
| defer-q2-01A | 1 | warm-up (excluded) | 96.2 | 118.3 | 77.3 | 63.7 | 143 | 0.49 |  |
| defer-q2-01A | 2 | extra | 101.6 | 124.8 | 93.5 | 63.7 | 39 | 0.18 |  |
| defer-q2-01A | 3 | kept | 100.7 | 110.3 | 100.7 | 64.0 | 39 | 0.18 |  |
| defer-q2-01A | 4 | kept | 81.8 | 134.2 | 67.7 | 62.9 | 39 | 0.18 |  |
| defer-q2-02B | 1 | warm-up (excluded) | 111.6 | 124.2 | 108.9 | 61.5 | 123 | 0.56 |  |
| defer-q2-02B | 2 | extra | 109.6 | 136.9 | 109.1 | 67.3 | 37 | 0.19 |  |
| defer-q2-02B | 3 | kept | 114.3 | 128.3 | 108.2 | 64.0 | 36 | 0.19 |  |
| defer-q2-02B | 4 | kept | 111.1 | 133.0 | 107.5 | 67.4 | 36 | 0.19 |  |
| defer-q2-03C | 1 | warm-up (excluded) | 99.2 | 122.5 | 98.9 | 63.4 | 123 | 0.56 |  |
| defer-q2-03C | 2 | extra | 103.8 | 128.0 | 101.3 | 63.3 | 36 | 0.20 |  |
| defer-q2-03C | 3 | kept | 104.2 | 126.2 | 100.8 | 66.4 | 36 | 0.19 |  |
| defer-q2-03C | 4 | kept | 101.6 | 123.7 | 100.4 | 62.6 | 36 | 0.20 |  |
| defer-q2-04C | 1 | warm-up (excluded) | 99.4 | 117.2 | 96.7 | 62.0 | 123 | 0.56 |  |
| defer-q2-04C | 2 | extra | 104.4 | 123.0 | 98.8 | 68.4 | 36 | 0.20 |  |
| defer-q2-04C | 3 | kept | 100.0 | 122.6 | 96.5 | 64.0 | 36 | 0.20 |  |
| defer-q2-04C | 4 | kept | 103.1 | 119.2 | 102.2 | 65.3 | 36 | 0.19 |  |
| defer-q2-05B | 1 | warm-up (excluded) | 111.4 | 124.7 | 108.9 | 61.5 | 124 | 0.56 |  |
| defer-q2-05B | 2 | extra | 109.7 | 136.9 | 109.0 | 67.3 | 36 | 0.19 |  |
| defer-q2-05B | 3 | kept | 113.8 | 128.3 | 108.5 | 64.0 | 36 | 0.19 |  |
| defer-q2-05B | 4 | kept | 111.0 | 132.9 | 107.4 | 67.4 | 37 | 0.19 |  |
| defer-q2-06A | 1 | warm-up (excluded) | 104.4 | 138.3 | 99.9 | 63.0 | 141 | 0.51 |  |
| defer-q2-06A | 2 | extra | 106.3 | 141.2 | 105.1 | 65.6 | 37 | 0.19 |  |
| defer-q2-06A | 3 | kept | 109.9 | 133.7 | 102.6 | 65.1 | 37 | 0.19 |  |
| defer-q2-06A | 4 | kept | 104.3 | 140.2 | 102.8 | 65.0 | 38 | 0.19 |  |
| defer-q2-07A | 1 | warm-up (excluded) | 105.0 | 135.8 | 99.0 | 63.2 | 142 | 0.51 |  |
| defer-q2-07A | 2 | extra | 107.7 | 132.4 | 102.4 | 63.8 | 37 | 0.19 |  |
| defer-q2-07A | 3 | kept | 104.4 | 132.9 | 103.2 | 64.8 | 36 | 0.19 |  |
| defer-q2-07A | 4 | kept | 99.3 | 129.2 | 97.8 | 60.3 | 37 | 0.19 |  |
| defer-q2-08B | 1 | warm-up (excluded) | 110.7 | 124.7 | 109.1 | 61.5 | 124 | 0.56 |  |
| defer-q2-08B | 2 | extra | 109.9 | 137.0 | 109.1 | 67.3 | 36 | 0.19 |  |
| defer-q2-08B | 3 | kept | 114.6 | 128.4 | 108.5 | 64.0 | 36 | 0.19 |  |
| defer-q2-08B | 4 | kept | 110.9 | 132.9 | 107.3 | 67.4 | 36 | 0.19 |  |
| defer-q2-09C | 1 | warm-up (excluded) | 99.1 | 124.2 | 99.0 | 63.7 | 123 | 0.56 |  |
| defer-q2-09C | 2 | extra | 100.3 | 126.3 | 99.6 | 63.8 | 36 | 0.20 |  |
| defer-q2-09C | 3 | kept | 99.7 | 123.3 | 99.1 | 64.5 | 36 | 0.20 |  |
| defer-q2-09C | 4 | kept | 101.9 | 126.5 | 98.7 | 64.1 | 35 | 0.20 |  |

### Decode per job (kept passes only)

| job | arm | delta | requests | overall | code | prose | accept % | prompt tok/s | TTFT s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| defer-iq2xs-01A | A | preset as is | 10 | 106.2 | 131.6 | 102.8 | 68.6 | 38 | 0.18 |
| defer-iq2xs-02B | B | STRATA_VERIFY_DEFER_CPU=(removed) | 10 | 104.3 | 123.8 | 100.1 | 67.1 | 36 | 0.19 |
| defer-iq2xs-03C | C | STRATA_VERIFY_DEFER_AUTO=1, STRATA_VERIFY_DEFER_AUTO_MIN_T=256 | 10 | 95.2 | 119.9 | 87.4 | 68.3 | 36 | 0.19 |
| defer-iq2xs-04C | C | STRATA_VERIFY_DEFER_AUTO=1, STRATA_VERIFY_DEFER_AUTO_MIN_T=256 | 10 | 89.8 | 100.3 | 81.2 | 68.3 | 36 | 0.19 |
| defer-iq2xs-05B | B | STRATA_VERIFY_DEFER_CPU=(removed) | 10 | 101.4 | 118.2 | 99.5 | 67.8 | 35 | 0.20 |
| defer-iq2xs-06A | A | preset as is | 10 | 104.7 | 130.8 | 103.5 | 66.6 | 37 | 0.19 |
| defer-iq2xs-07A | A | preset as is | 10 | 105.7 | 131.2 | 99.4 | 68.6 | 37 | 0.19 |
| defer-iq2xs-08B | B | STRATA_VERIFY_DEFER_CPU=(removed) | 10 | 107.5 | 120.0 | 93.4 | 67.0 | 35 | 0.20 |
| defer-iq2xs-09C | C | STRATA_VERIFY_DEFER_AUTO=1, STRATA_VERIFY_DEFER_AUTO_MIN_T=256 | 10 | 86.8 | 101.7 | 77.0 | 68.4 | 35 | 0.20 |
| defer-q2-01A | A | preset as is | 10 | 93.0 | 128.7 | 79.1 | 63.4 | 39 | 0.18 |
| defer-q2-02B | B | STRATA_VERIFY_DEFER_CPU=(removed) | 10 | 112.7 | 131.6 | 107.8 | 65.7 | 36 | 0.19 |
| defer-q2-03C | C | STRATA_VERIFY_DEFER_AUTO=1, STRATA_VERIFY_DEFER_AUTO_MIN_T=256 | 10 | 102.9 | 123.7 | 100.6 | 64.5 | 36 | 0.19 |
| defer-q2-04C | C | STRATA_VERIFY_DEFER_AUTO=1, STRATA_VERIFY_DEFER_AUTO_MIN_T=256 | 10 | 102.7 | 120.0 | 98.2 | 64.6 | 36 | 0.19 |
| defer-q2-05B | B | STRATA_VERIFY_DEFER_CPU=(removed) | 10 | 112.4 | 131.7 | 108.0 | 65.7 | 36 | 0.19 |
| defer-q2-06A | A | preset as is | 10 | 107.1 | 140.2 | 102.7 | 65.1 | 37 | 0.19 |
| defer-q2-07A | A | preset as is | 10 | 103.8 | 132.9 | 98.5 | 62.5 | 37 | 0.19 |
| defer-q2-08B | B | STRATA_VERIFY_DEFER_CPU=(removed) | 10 | 112.8 | 131.6 | 107.9 | 65.7 | 36 | 0.19 |
| defer-q2-09C | C | STRATA_VERIFY_DEFER_AUTO=1, STRATA_VERIFY_DEFER_AUTO_MIN_T=256 | 10 | 100.8 | 126.1 | 98.9 | 64.3 | 36 | 0.20 |

### Per arm: defer-iq2xs

Servers of one arm in run order; the medians pool the kept requests of all its servers; delta against arm A. Order of the servers: A B C C B A A B C.

| arm | meaning | servers | requests | overall | vs A | code | vs A | prose | vs A | accept % | overall per server |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | deferral on (the preset's STRATA_VERIFY_DEFER_CPU=1) | 3 | 30 | 105.3 | +0.0% | 131.5 | +0.0% | 102.2 | +0.0% | 67.9 | 106.2 / 104.7 / 105.7 |
| B | deferral removed (STRATA_VERIFY_DEFER_CPU unset) | 3 | 30 | 104.3 | -0.9% | 122.0 | -7.2% | 99.0 | -3.2% | 67.3 | 104.3 / 101.4 / 107.5 |
| C | gate auto-closed (VERIFY_DEFER_AUTO=1, MIN_T=256: deferral's graphs, every CPU row on time) | 3 | 30 | 92.1 | -12.5% | 102.0 | -22.5% | 81.0 | -20.7% | 68.3 | 95.2 / 89.8 / 86.8 |

### Per arm: defer-q2

Servers of one arm in run order; the medians pool the kept requests of all its servers; delta against arm A. Order of the servers: A B C C B A A B C.

| arm | meaning | servers | requests | overall | vs A | code | vs A | prose | vs A | accept % | overall per server |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | deferral on (the preset's STRATA_VERIFY_DEFER_CPU=1) | 3 | 30 | 103.3 | +0.0% | 134.6 | +0.0% | 99.5 | +0.0% | 63.7 | 93.0 / 107.1 / 103.8 |
| B | deferral removed (STRATA_VERIFY_DEFER_CPU unset) | 3 | 30 | 112.4 | +8.9% | 131.6 | -2.2% | 107.8 | +8.3% | 65.7 | 112.7 / 112.4 / 112.8 |
| C | gate auto-closed (VERIFY_DEFER_AUTO=1, MIN_T=256: deferral's graphs, every CPU row on time) | 3 | 30 | 102.1 | -1.2% | 122.2 | -9.2% | 99.4 | -0.2% | 64.5 | 102.9 / 102.7 / 100.8 |

## TTFT (tools/pancake/ttft_bench.py)

Scenario A: single prompts, run 1 of each size discarded by the tool, medians of the rest; TTFT is the client's clock, the engine columns are the server's own `timings`. Scenario B: the 10-turn agent session.

**ttft-pancake-iq2xs-a** (load start 6.42, end 5.64, warning True)

| size | status | prompt tokens | TTFT median s | min | max | engine prompt ms | engine prompt tok/s | kept runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8K | measured | 8200 | 3.591 | 3.583 | 3.704 | 3570.0 | 2296.6 | 3 |
| 32K | measured | 32419 | 11.623 | 11.551 | 11.623 | 11585.3 | 2798.2 | 3 |
| 64K | measured | 65352 | 22.421 | 22.398 | 22.44 | 22368.9 | 2921.4 | 3 |

Session: total TTFT over the 10 turns (median of complete sessions) 22.983 s; reuse of prompt tokens on follow-up turns 83.5 % (source usage.prompt_tokens_details.cached_tokens); complete sessions 3/3.

| turn | n | TTFT s | prompt tokens | reused | read | re-read |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 2.989 | 3542 | 0 | 3542 | None |
| 2 | 3 | 1.923 | 6496 | 3605 | 2891 | 1 |
| 3 | 3 | 2.166 | 9812 | 6559 | 3253 | 1 |
| 4 | 3 | 2.561 | 14311 | 9872 | 4439 | 1 |
| 5 | 3 | 2.055 | 17350 | 14374 | 2976 | 1 |
| 6 | 3 | 2.16 | 20586 | 17410 | 3176 | 1 |
| 7 | 3 | 2.197 | 23887 | 20649 | 3237 | 1 |
| 8 | 3 | 2.459 | 27780 | 23941 | 3839 | 1 |
| 9 | 3 | 2.194 | 31096 | 27810 | 3286 | 1 |
| 10 | 3 | 2.336 | 34665 | 31141 | 3524 | 1 |

**ttft-pancake-iq2xs-100k-trace** (load start 9.71, end 4.27, warning True)

| size | status | prompt tokens | TTFT median s | min | max | engine prompt ms | engine prompt tok/s | kept runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 100K | measured | 101643 | 36.208 | 36.05 | 36.366 | 36135.7 | 2812.8 | 3 |

**ttft-rdna3-iq2xs** (load start 10.61, end 9.83, warning True)

| size | status | prompt tokens | TTFT median s | min | max | engine prompt ms | engine prompt tok/s | kept runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8K | measured | 8153 | 4.021 | 3.866 | 4.88 | 3999.6 | 2038.5 | 3 |
| 32K | measured | 32540 | 13.743 | 13.591 | 13.785 | 13709.8 | 2373.6 | 3 |
| 64K | measured | 65382 | 25.022 | 25.012 | 25.075 | 24972.3 | 2618.2 | 3 |
| 100K | measured | 102280 | 37.279 | 37.17 | 37.304 | 37211.6 | 2748.6 | 3 |

Session: total TTFT over the 10 turns (median of complete sessions) 24.187 s; reuse of prompt tokens on follow-up turns 83.5 % (source usage.prompt_tokens_details.cached_tokens); complete sessions 3/3.

| turn | n | TTFT s | prompt tokens | reused | read | re-read |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 2.789 | 3503 | 0 | 3503 | None |
| 2 | 3 | 2.035 | 6457 | 3567 | 2890 | 0 |
| 3 | 3 | 2.311 | 9773 | 6521 | 3252 | 1 |
| 4 | 3 | 2.679 | 14275 | 9837 | 4439 | 1 |
| 5 | 3 | 2.152 | 17314 | 14338 | 2976 | 1 |
| 6 | 3 | 2.474 | 20554 | 17378 | 3176 | 0 |
| 7 | 3 | 2.33 | 23852 | 20615 | 3237 | 1 |
| 8 | 3 | 2.633 | 27752 | 23913 | 3839 | 1 |
| 9 | 3 | 2.349 | 31088 | 27802 | 3286 | 1 |
| 10 | 3 | 2.528 | 34655 | 31131 | 3524 | 1 |

**ttft-avalon-iq2xs** (load start 4.79, end 3.82, warning True)

| size | status | prompt tokens | TTFT median s | min | max | engine prompt ms | engine prompt tok/s | kept runs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8K | measured | 8158 | 5.515 | 5.441 | 5.761 | 5490.0 | 1485.8 | 3 |
| 32K | failed at run 1: no token streamed | None | None | None | None | None | None | 0 |
| 64K | failed at run 1: ConnectionRefusedError(111, 'Connection ref | None | None | None | None | None | None | 0 |
| 100K | failed at run 1: ConnectionRefusedError(111, 'Connection ref | None | None | None | None | None | None | 0 |

Session: total TTFT over the 10 turns (median of complete sessions) None s; reuse of prompt tokens on follow-up turns None % (source not reported); complete sessions 0/3.

| turn | n | TTFT s | prompt tokens | reused | read | re-read |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0 | None | None | None | None | None |

## Agent session with side requests (tools/pancake/nq_clients.py session)

Each turn adds about 1,500 words and resends the whole conversation; one short unrelated side request is sent between two turns. TTFT is the client's. Reuse = prompt tokens served from the server's cache.

| job | delta | status | total TTFT s (median session) | median follow-up TTFT s | reuse % follow-up turns | complete sessions | side request TTFT s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lever-01A | preset as is | done | 25.077 | 2.681 | 83.5 | 3/3 | 1.022 |
| lever-02B | --conversation-cache-min-tokens 8192 | done | 27.27 | 2.642 | 78.1 | 3/3 | 0.881 |
| lever-03C | --conversation-cache-mib 6144 | done | 24.894 | 2.584 | 83.5 | 3/3 | 0.993 |
| lever-04D | STRATA_PARK_TRUNCATE=1 | done | 24.926 | 2.589 | 83.5 | 3/3 | 1.007 |
| lever-05E | STRATA_PARK_TRUNCATE=1, --conversation-cache-min-tokens 8192, --conversation-cache-mib 6144 | done | 28.549 | 3.029 | 78.1 | 3/3 | 0.878 |
| lever-06F | STRATA_PARK_REWIND_MAX=8192, STRATA_PARK_DIVERGENCE_LOG=1 | done | 26.348 | 2.674 | 83.5 | 3/3 | 1.034 |
| lever-07A | preset as is | done | 24.839 | 2.576 | 83.5 | 3/3 | 0.996 |

Per turn, lever-01A:

| turn | n | TTFT s | prompt tokens | reused | read | re-read |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 3.027 | 3543 | 0 | 3543 | None |
| 2 | 3 | 2.249 | 6497 | 3606 | 2891 | 1 |
| 3 | 3 | 2.367 | 9811 | 6559 | 3253 | 1 |
| 4 | 3 | 2.745 | 14313 | 9874 | 4439 | 1 |
| 5 | 3 | 2.245 | 17352 | 14376 | 2976 | 1 |
| 6 | 3 | 2.346 | 20592 | 17415 | 3177 | 1 |
| 7 | 3 | 2.394 | 23893 | 20655 | 3238 | 1 |
| 8 | 3 | 2.681 | 27749 | 23910 | 3839 | 1 |
| 9 | 3 | 2.449 | 31044 | 27758 | 3286 | 1 |
| 10 | 3 | 2.731 | 34580 | 31056 | 3524 | 1 |

Post step of lever-01A: `/usr/bin/python3 /home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a/tools/pancake/ttft_audit.py --turns /home/dan/night-queue-run/raw/turns-lever-01A.jsonl --engine-log /home/dan/night-queue-run/raw/engine-lever-01A.log --json /home/dan/night-queue-run/raw/lever-01A-audit.json` -> `raw/lever-01A-post1.txt` (rc 0).

Per turn, lever-02B:

| turn | n | TTFT s | prompt tokens | reused | read | re-read |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 2.947 | 3544 | 0 | 3544 | None |
| 2 | 3 | 3.043 | 6498 | 0 | 6498 | 3608 |
| 3 | 3 | 3.966 | 9814 | 0 | 9814 | 6562 |
| 4 | 3 | 2.642 | 14316 | 9877 | 4439 | 1 |
| 5 | 3 | 2.244 | 17355 | 14379 | 2976 | 1 |
| 6 | 3 | 2.599 | 20587 | 17411 | 3176 | 1 |
| 7 | 3 | 2.266 | 23888 | 20650 | 3238 | 1 |
| 8 | 3 | 2.622 | 27775 | 23935 | 3839 | 1 |
| 9 | 3 | 2.487 | 31077 | 27791 | 3286 | 1 |
| 10 | 3 | 2.475 | 34639 | 31115 | 3524 | 1 |

Post step of lever-02B: `/usr/bin/python3 /home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a/tools/pancake/ttft_audit.py --turns /home/dan/night-queue-run/raw/turns-lever-02B.jsonl --engine-log /home/dan/night-queue-run/raw/engine-lever-02B.log --json /home/dan/night-queue-run/raw/lever-02B-audit.json` -> `raw/lever-02B-post1.txt` (rc 0).

Per turn, lever-03C:

| turn | n | TTFT s | prompt tokens | reused | read | re-read |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 2.959 | 3543 | 0 | 3543 | None |
| 2 | 3 | 2.106 | 6497 | 3606 | 2891 | 1 |
| 3 | 3 | 2.361 | 9813 | 6560 | 3253 | 1 |
| 4 | 3 | 2.738 | 14315 | 9876 | 4439 | 1 |
| 5 | 3 | 2.242 | 17354 | 14378 | 2976 | 1 |
| 6 | 3 | 2.344 | 20593 | 17416 | 3177 | 1 |
| 7 | 3 | 2.423 | 23869 | 20632 | 3237 | 1 |
| 8 | 3 | 2.7 | 27731 | 23892 | 3839 | 1 |
| 9 | 3 | 2.486 | 31043 | 27757 | 3286 | 1 |
| 10 | 3 | 2.584 | 34611 | 31087 | 3524 | 1 |

Post step of lever-03C: `/usr/bin/python3 /home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a/tools/pancake/ttft_audit.py --turns /home/dan/night-queue-run/raw/turns-lever-03C.jsonl --engine-log /home/dan/night-queue-run/raw/engine-lever-03C.log --json /home/dan/night-queue-run/raw/lever-03C-audit.json` -> `raw/lever-03C-post1.txt` (rc 0).

Per turn, lever-04D:

| turn | n | TTFT s | prompt tokens | reused | read | re-read |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 3.094 | 3542 | 0 | 3542 | None |
| 2 | 3 | 2.109 | 6496 | 3605 | 2891 | 1 |
| 3 | 3 | 2.375 | 9812 | 6559 | 3253 | 1 |
| 4 | 3 | 2.733 | 14314 | 9875 | 4439 | 1 |
| 5 | 3 | 2.236 | 17353 | 14377 | 2976 | 1 |
| 6 | 3 | 2.337 | 20586 | 17410 | 3176 | 1 |
| 7 | 3 | 2.45 | 23875 | 20638 | 3237 | 1 |
| 8 | 3 | 2.687 | 27751 | 23912 | 3839 | 1 |
| 9 | 3 | 2.444 | 31073 | 27787 | 3286 | 1 |
| 10 | 3 | 2.589 | 34639 | 31115 | 3524 | 1 |

Post step of lever-04D: `/usr/bin/python3 /home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a/tools/pancake/ttft_audit.py --turns /home/dan/night-queue-run/raw/turns-lever-04D.jsonl --engine-log /home/dan/night-queue-run/raw/engine-lever-04D.log --json /home/dan/night-queue-run/raw/lever-04D-audit.json` -> `raw/lever-04D-post1.txt` (rc 0).

Per turn, lever-05E:

| turn | n | TTFT s | prompt tokens | reused | read | re-read |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 2.965 | 3544 | 0 | 3544 | None |
| 2 | 3 | 3.029 | 6497 | 0 | 6497 | 3607 |
| 3 | 3 | 3.965 | 9813 | 0 | 9813 | 6561 |
| 4 | 3 | 3.233 | 14315 | 9876 | 4439 | 1 |
| 5 | 3 | 2.934 | 17354 | 14378 | 2976 | 1 |
| 6 | 3 | 2.242 | 20594 | 17417 | 3177 | 1 |
| 7 | 3 | 2.273 | 23889 | 20652 | 3237 | 1 |
| 8 | 3 | 2.89 | 27769 | 23930 | 3839 | 1 |
| 9 | 3 | 3.235 | 31087 | 27801 | 3286 | 1 |
| 10 | 3 | 2.83 | 34659 | 31135 | 3524 | 1 |

Post step of lever-05E: `/usr/bin/python3 /home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a/tools/pancake/ttft_audit.py --turns /home/dan/night-queue-run/raw/turns-lever-05E.jsonl --engine-log /home/dan/night-queue-run/raw/engine-lever-05E.log --json /home/dan/night-queue-run/raw/lever-05E-audit.json` -> `raw/lever-05E-post1.txt` (rc 0).

Per turn, lever-06F:

| turn | n | TTFT s | prompt tokens | reused | read | re-read |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 3.71 | 3543 | 0 | 3543 | None |
| 2 | 3 | 2.903 | 6497 | 3606 | 2891 | 1 |
| 3 | 3 | 2.372 | 9813 | 6560 | 3253 | 1 |
| 4 | 3 | 3.042 | 14315 | 9876 | 4439 | 1 |
| 5 | 3 | 2.246 | 17353 | 14378 | 2976 | 1 |
| 6 | 3 | 2.339 | 20582 | 17406 | 3176 | 1 |
| 7 | 3 | 2.384 | 23861 | 20624 | 3237 | 1 |
| 8 | 3 | 2.677 | 27730 | 23891 | 3839 | 1 |
| 9 | 3 | 2.447 | 31028 | 27742 | 3286 | 1 |
| 10 | 3 | 2.576 | 34564 | 31040 | 3524 | 1 |

Post step of lever-06F: `/usr/bin/python3 /home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a/tools/pancake/ttft_audit.py --turns /home/dan/night-queue-run/raw/turns-lever-06F.jsonl --engine-log /home/dan/night-queue-run/raw/engine-lever-06F.log --json /home/dan/night-queue-run/raw/lever-06F-audit.json` -> `raw/lever-06F-post1.txt` (rc 0).

Per turn, lever-07A:

| turn | n | TTFT s | prompt tokens | reused | read | re-read |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3 | 3.001 | 3543 | 0 | 3543 | None |
| 2 | 3 | 2.106 | 6497 | 3606 | 2891 | 1 |
| 3 | 3 | 2.369 | 9809 | 6557 | 3252 | 1 |
| 4 | 3 | 2.728 | 14311 | 9872 | 4439 | 1 |
| 5 | 3 | 2.241 | 17350 | 14374 | 2976 | 1 |
| 6 | 3 | 2.342 | 20590 | 17413 | 3177 | 1 |
| 7 | 3 | 2.392 | 23885 | 20648 | 3237 | 1 |
| 8 | 3 | 2.689 | 27788 | 23948 | 3839 | 1 |
| 9 | 3 | 2.448 | 31100 | 27814 | 3286 | 1 |
| 10 | 3 | 2.576 | 34672 | 31148 | 3524 | 1 |

Post step of lever-07A: `/usr/bin/python3 /home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a/tools/pancake/ttft_audit.py --turns /home/dan/night-queue-run/raw/turns-lever-07A.jsonl --engine-log /home/dan/night-queue-run/raw/engine-lever-07A.log --json /home/dan/night-queue-run/raw/lever-07A-audit.json` -> `raw/lever-07A-post1.txt` (rc 0).

## Depth (single prompts, then a decode at that depth)

Run 1 of each size is discarded; medians of the rest. Acceptance = accepted / drafted over the kept runs.

| job | arm | delta | size | status | prompt tokens | TTFT s | prompt tok/s | decode tok/s | accept % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kvring-01S | S | preset as is | 1K | measured | 1032.5 | 1.127 | 931.0 | 98.4 | 58.9 |
| kvring-01S | S | preset as is | 32K | measured | 32739.0 | 11.651 | 2818.8 | 85.7 | 53.8 |
| kvring-01S | S | preset as is | 64K | measured | 65344.5 | 22.923 | 2857.8 | 76.5 | 68.9 |
| kvring-01S | S | preset as is | 120K | measured | 122661.0 | 42.443 | 2895.9 | 74.0 | 65.8 |
| kvring-02T | T | --kv-resident 49152 | 1K | measured | 1033.0 | 1.125 | 933.6 | 95.6 | 55.1 |
| kvring-02T | T | --kv-resident 49152 | 32K | measured | 32745.0 | 11.666 | 2816.0 | 94.6 | 57.1 |
| kvring-02T | T | --kv-resident 49152 | 64K | measured | 65341.0 | 22.471 | 2914.6 | 91.7 | 59.8 |
| kvring-02T | T | --kv-resident 49152 | 120K | measured | 122659.0 | 42.452 | 2895.0 | 101.0 | 72.3 |
| kvring-03R | R | STRATA_KV_RING_ADAPT=1, --kv-resident 49152 | 1K | measured | 1036.0 | 1.129 | 932.3 | 95.3 | 54.9 |
| kvring-03R | R | STRATA_KV_RING_ADAPT=1, --kv-resident 49152 | 32K | measured | 32739.0 | 11.669 | 2814.4 | 97.3 | 62.3 |
| kvring-03R | R | STRATA_KV_RING_ADAPT=1, --kv-resident 49152 | 64K | measured | 65341.5 | 22.507 | 2909.9 | 93.3 | 61.5 |
| kvring-03R | R | STRATA_KV_RING_ADAPT=1, --kv-resident 49152 | 120K | measured | 122657.0 | 42.594 | 2885.2 | 93.2 | 61.3 |
| kvring-04P | P | STRATA_KV_RING_ADAPT=1, --kv-resident 49152, --max-context 65536 | 1K | measured | 1037.5 | 1.118 | 942.5 | 96.1 | 55.4 |
| kvring-04P | P | STRATA_KV_RING_ADAPT=1, --kv-resident 49152, --max-context 65536 | 32K | measured | 32744.0 | 11.671 | 2814.1 | 91.8 | 61.8 |
| kvring-04P | P | STRATA_KV_RING_ADAPT=1, --kv-resident 49152, --max-context 65536 | 56K | measured | 57202.0 | 19.955 | 2873.7 | 97.1 | 64.1 |
| kvring-05P | P | STRATA_KV_RING_ADAPT=1, --kv-resident 49152, --max-context 65536 | 1K | measured | 1033.0 | 1.151 | 912.0 | 94.3 | 51.6 |
| kvring-05P | P | STRATA_KV_RING_ADAPT=1, --kv-resident 49152, --max-context 65536 | 32K | measured | 32746.5 | 11.667 | 2815.8 | 92.0 | 63.1 |
| kvring-05P | P | STRATA_KV_RING_ADAPT=1, --kv-resident 49152, --max-context 65536 | 56K | measured | 57197.5 | 19.962 | 2872.6 | 95.5 | 61.5 |
| kvring-06R | R | STRATA_KV_RING_ADAPT=1, --kv-resident 49152 | 1K | measured | 1034.5 | 1.121 | 938.0 | 101.6 | 65.4 |
| kvring-06R | R | STRATA_KV_RING_ADAPT=1, --kv-resident 49152 | 32K | measured | 32748.0 | 11.685 | 2811.2 | 96.5 | 63.3 |
| kvring-06R | R | STRATA_KV_RING_ADAPT=1, --kv-resident 49152 | 64K | measured | 65340.5 | 22.603 | 2897.8 | 93.5 | 60.8 |
| kvring-06R | R | STRATA_KV_RING_ADAPT=1, --kv-resident 49152 | 120K | measured | 122658.5 | 42.607 | 2884.6 | 86.8 | 62.2 |
| kvring-07T | T | --kv-resident 49152 | 1K | measured | 1038.5 | 1.121 | 941.7 | 96.0 | 61.0 |
| kvring-07T | T | --kv-resident 49152 | 32K | measured | 32745.5 | 11.68 | 2813.1 | 92.2 | 59.1 |
| kvring-07T | T | --kv-resident 49152 | 64K | measured | 65341.0 | 22.491 | 2912.1 | 93.4 | 56.9 |
| kvring-07T | T | --kv-resident 49152 | 120K | measured | 122660.0 | 42.645 | 2882.0 | 94.4 | 66.3 |
| kvring-08S | S | preset as is | 1K | measured | 1034.5 | 1.123 | 936.9 | 97.4 | 57.6 |
| kvring-08S | S | preset as is | 32K | measured | 32738.5 | 11.671 | 2814.4 | 92.5 | 57.3 |
| kvring-08S | S | preset as is | 64K | measured | 65348.0 | 22.478 | 2914.1 | 93.3 | 63.1 |
| kvring-08S | S | preset as is | 120K | measured | 122660.0 | 42.461 | 2894.6 | 93.5 | 67.1 |

## Prose profile (tools/pancake/prose_gap.py)

Run 1 of each series discarded by the tool; medians of the others. Tokens per window = tokens / (tokens - accepted drafts), an assumption of the tool.

| job | delta | prompt | engine | tok/s median | sd | accept % | tok / window | ms / window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| prose-base-01A | preset as is | story | pancake | 100.6 | 1.8 | 52.8 | 1.84 | 18.4 |
| prose-base-01A | preset as is | tips | pancake | 109.7 | 2.7 | 62.8 | 2.04 | 18.6 |
| prose-base-01A | preset as is | moe | pancake | 106.7 | 1.1 | 58.6 | 1.99 | 18.6 |
| prose-base-01A | preset as is | js | pancake | 149.2 | 7.7 | 78.9 | 3.08 | 20.6 |
| prose-base-01A | preset as is | lru | pancake | 149.0 | 1.1 | 83.7 | 3.12 | 21.0 |
| prose-avalon | preset as is | story | llama | 104.2 | 2.8 | 58.7 | 2.17 | 20.4 |
| prose-avalon | preset as is | tips | llama | 114.6 | 4.7 | 64.8 | 2.29 | 19.8 |
| prose-avalon | preset as is | moe | llama | 115.7 | 2.7 | 67.4 | 2.34 | 20.2 |
| prose-avalon | preset as is | js | llama | 125.6 | 6.7 | 84.6 | 2.88 | 22.7 |
| prose-avalon | preset as is | lru | llama | 132.7 | 4.7 | 91.2 | 2.83 | 21.0 |
| prose-var-01B | --spec-min-p 0 | story | pancake | 95.7 | 3.4 | 33.7 | 2.01 | 21.0 |
| prose-var-01B | --spec-min-p 0 | tips | pancake | 110.2 | 3.3 | 44.9 | 2.34 | 21.1 |
| prose-var-01B | --spec-min-p 0 | moe | pancake | 100.3 | 3.7 | 37.7 | 2.12 | 21.2 |
| prose-var-01B | --spec-min-p 0 | js | pancake | 147.1 | 3.4 | 72.6 | 3.16 | 21.6 |
| prose-var-01B | --spec-min-p 0 | lru | pancake | 152.4 | 2.7 | 77.3 | 3.30 | 21.7 |
| prose-var-02C | --spec 2, --spec-min-p 0 | story | pancake | 95.9 | 1.0 | 59.2 | 1.59 | 16.6 |
| prose-var-02C | --spec 2, --spec-min-p 0 | tips | pancake | 103.5 | 1.6 | 72.9 | 1.72 | 16.6 |
| prose-var-02C | --spec 2, --spec-min-p 0 | moe | pancake | 98.0 | 1.7 | 62.2 | 1.62 | 16.6 |
| prose-var-02C | --spec 2, --spec-min-p 0 | js | pancake | 113.2 | 2.9 | 88.0 | 1.87 | 16.6 |
| prose-var-02C | --spec 2, --spec-min-p 0 | lru | pancake | 114.7 | 1.7 | 88.8 | 1.95 | 17.0 |
| prose-var-03D | STRATA_SPEC_COST=1 | story | pancake | 100.5 | 2.4 | 45.1 | 1.88 | 18.8 |
| prose-var-03D | STRATA_SPEC_COST=1 | tips | pancake | 111.9 | 0.8 | 56.1 | 2.10 | 18.8 |
| prose-var-03D | STRATA_SPEC_COST=1 | moe | pancake | 104.3 | 0.6 | 49.7 | 1.96 | 18.8 |
| prose-var-03D | STRATA_SPEC_COST=1 | js | pancake | 134.2 | 3.4 | 79.0 | 2.56 | 19.1 |
| prose-var-03D | STRATA_SPEC_COST=1 | lru | pancake | 139.1 | 3.4 | 83.6 | 2.69 | 19.3 |
| prose-var-04A | preset as is | story | pancake | 80.8 | 6.8 | 52.8 | 1.84 | 23.1 |
| prose-var-04A | preset as is | tips | pancake | 104.8 | 14.2 | 63.6 | 2.04 | 19.8 |
| prose-var-04A | preset as is | moe | pancake | 100.8 | 3.6 | 63.2 | 2.02 | 19.9 |
| prose-var-04A | preset as is | js | pancake | 132.0 | 23.1 | 76.7 | 3.00 | 23.0 |
| prose-var-04A | preset as is | lru | pancake | 122.0 | 25.5 | 84.8 | 3.14 | 26.7 |

