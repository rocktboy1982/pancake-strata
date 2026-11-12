# Night queue results (night-queue-2026-11-11)

Written 2026-11-12 15:08:56 by tools/pancake/night_queue.py from `state.json`. Numbers only, no interpretation; nothing here is a finding until it is read against the pass-by-pass tables. **Pass 1 of a decode job is the warm-up and is excluded**; only the last 2 passes after it are kept (median over the requests of those passes). Estimated durations are the plan's, not measurements.

Status: done 3; 0 not started.

## Jobs

| job | group | engine | delta (env / args) | status | wall min | est min | stop problem |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ctrl-avalon-tailoff | 5-prose | avalon-tailoff | preset as is | done | 2.8 | 8 |  |
| ctrl-avalon-iq3ks-main | 5-prose | avalon-iq3ks-main | preset as is | done | 5.9 | 8 |  |
| ctrl-pancake-base-rep-01A | 5-prose | pancake-iq2xs | preset as is | done | 5.5 | 8 |  |

## Prose profile (tools/pancake/prose_gap.py)

Run 1 of each series discarded by the tool; medians of the others. Tokens per window = tokens / (tokens - accepted drafts), an assumption of the tool.

| job | delta | prompt | engine | tok/s median | sd | accept % | tok / window | ms / window |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ctrl-avalon-tailoff | preset as is | story | llama | 91.2 | 4.2 | 55.1 | 2.10 | 23.1 |
| ctrl-avalon-tailoff | preset as is | tips | llama | 97.6 | 3.2 | 61.8 | 2.24 | 22.7 |
| ctrl-avalon-tailoff | preset as is | moe | llama | 103.2 | 3.1 | 65.9 | 2.31 | 22.5 |
| ctrl-avalon-tailoff | preset as is | js | llama | 114.1 | 5.0 | 86.0 | 2.80 | 25.4 |
| ctrl-avalon-tailoff | preset as is | lru | llama | 122.7 | 1.6 | 91.2 | 2.81 | 23.0 |
| ctrl-avalon-iq3ks-main | preset as is | story | llama | 86.2 | 2.9 | 57.5 | 2.14 | 24.7 |
| ctrl-avalon-iq3ks-main | preset as is | tips | llama | 93.2 | 3.0 | 63.5 | 2.26 | 24.4 |
| ctrl-avalon-iq3ks-main | preset as is | moe | llama | 92.8 | 2.4 | 65.3 | 2.30 | 24.8 |
| ctrl-avalon-iq3ks-main | preset as is | js | llama | 91.3 | 15.3 | 79.8 | 2.81 | 30.8 |
| ctrl-avalon-iq3ks-main | preset as is | lru | llama | 98.2 | 11.9 | 73.3 | 2.78 | 28.6 |
| ctrl-pancake-base-rep-01A | preset as is | story | pancake | 89.7 | 1.3 | 53.4 | 1.78 | 19.9 |
| ctrl-pancake-base-rep-01A | preset as is | tips | pancake | 100.6 | 2.9 | 65.5 | 2.06 | 20.5 |
| ctrl-pancake-base-rep-01A | preset as is | moe | pancake | 94.7 | 2.2 | 60.3 | 1.92 | 20.2 |
| ctrl-pancake-base-rep-01A | preset as is | js | pancake | 128.9 | 3.4 | 75.1 | 2.99 | 23.8 |
| ctrl-pancake-base-rep-01A | preset as is | lru | pancake | 130.4 | 3.5 | 84.4 | 3.10 | 23.9 |

