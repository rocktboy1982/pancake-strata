# Community benchmark on CPU only

Measured on 2026-11-08 with `tools/strata_bench.py` (suite 1, 3 runs per test, medians with ranges). Speeds are the engine's own timings; first-token times are measured by the client over HTTP. Speed only: no correctness check was run.

## Hardware and software

- GPU and VRAM: CPU only, VRAM not read
- CPU: AMD Ryzen 7 5700G with Radeon Graphics (8 cores, 16 threads); installed RAM: 57.7 GiB; storage: sda     ST2000DM006-2DM164  sata; sdb     INTEL SSDSC2CW120A3 sata; sdc     SanDisk 3.2Gen1     usb; sr0     ATAPI iHAS122 F     sata; nvme0n1 KINGSTON SNVS2000G  nvme
- OS: Linux 7.0.0-43-generic; GPU driver: not read; CUDA (driver supports up to): not read
- Strata: project 0.1.58, commit 77c2ef52; engine 0.1.58
- Background workloads: not recorded (GPU power limit/draw read above)

## Model and configuration

- Model and quantization: qwen3.8-flash-next-iq2_xs
- Context: 131072 tokens; parallel slots: 1
- Settings: the install's own configuration, unchanged; thinking off for the suite; temperature 0

## Method

Run by `python tools/strata_bench.py --long` against the HTTP API of a Strata server (already running); model loading is not timed. One warm-up chat and a 1K prompt are discarded first. Every prompt starts with a unique tag, so the conversation cache cannot help the fresh-prompt tests.

- short chat: the three questions of `tools/ab_engine.py`, up to 100 new tokens, 3 runs
- 4K / 16K prompts: generated word lists sized by a calibration request, 24 new tokens, 3 fresh prompts each
- agent session: system prompt and four tools, then 5 turns that each add about 1500 words of a project file and continue from the model's own earlier answers (the way agent clients resend a conversation), 64 new tokens per turn, 3 sessions; the cache numbers come from the answer's usage (`cached_tokens`) and timings (`prompt_n`)
- concurrency (when run): N requests sent at the same moment, total tokens over the wall time of the burst, latency over all requests

## Results

| Configuration | Actual prompt tokens | Reused tokens | Generated tokens | Runs | Prompt tok/s median and range | Decode tok/s median and range | TTFT seconds median and range |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| short chat | 46 (43-51) | 0 | 100 (100-100) | 9 | not measured | 106.9 (82.2-130.0) | 0.42 (0.38-1.19) |
| 4K prompt, fresh | 4019 (4019-4019) | 0 (0-0) | 24 cap | 3 | 1756.9 (1419.6-1765.7) | not measured | 2.31 (2.29-2.85) |
| 16K prompt, fresh | 16058 (16058-16058) | 0 (0-0) | 24 cap | 3 | 2456.2 (2252.2-2478.0) | not measured | 6.56 (6.50-7.16) |
| 32K prompt, fresh | 32049 (32049-32049) | 0 (0-0) | 24 cap | 3 | 2580.2 (2519.8-2591.3) | not measured | 12.45 (12.40-12.76) |
| agent session, follow-up turns | 13262 (13255-13270) at the last turn | see below | up to 64 | 3 | see below | 117.0 (93.2-145.0) | 2.05 (1.43-3.21) |

Ranges are the lowest and highest of the runs; `Runs` counts requests that returned a token.

### Agent session (conversation cache)

Cache hit over the follow-up turns: 69.4 % (69.4-69.4); earlier-turn tokens read again per session: 5 (5-5) (earlier turns that the cache did not hold; the 12414 (12414-12414) new tokens of the files are read once); first turn (nothing cached) first token: 1.64 s (1.56-2.11).

| Turn (run 1) | Prompt tokens | Reused | Read | Prompt tok/s | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 395.1 | 1.56 | 76.0 |
| 1 | 3513 | 619 | 2894 | 934.2 | 3.121 | 93.2 |
| 2 | 5046 | 3576 | 1470 | 718.4 | 2.067 | 109.5 |
| 3 | 8290 | 5109 | 3181 | 997.0 | 3.213 | 99.8 |
| 4 | 10509 | 8345 | 2164 | 864.7 | 2.525 | 115.3 |
| 5 | 13262 | 10552 | 2710 | 1000.4 | 2.731 | 121.5 |

Per-request data: `results.json`. Engine log lines from the start: `engine-log-excerpt.txt` (when this run started the server).

## Correctness and limitations

Not measured: this is a speed suite. The conversation uses the model's own replies, so token counts differ a little between runs. Machine details come from the server, `nvidia-smi`/`rocm-smi` and the OS; background load is not recorded. Total time of the suite: 133 s.
