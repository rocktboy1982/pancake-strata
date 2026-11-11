# pancake vs Avalon, same client (2026-11-11, RX 7900 XT, quiet machine)

Client: `cmp_bench.py` (one neutral client for both servers: same prompts, greedy, thinking off, OpenAI endpoint timings).
**Run 1 of every case is discarded** (decode: round 1 of 3; prompts: run 1 of 4); medians over the rest.

| | Avalon (llama.cpp fork 8b9a0f4, IQ3_KS hybrid, 256K ctx, MTP+ngram) | pancake (fc77f60c6, fast-hip, Flash-Next IQ2_XS, 128K ctx) | pancake / Avalon |
| --- | ---: | ---: | ---: |
| decode, story (400 words) | 85.9 | 93.5 | 1.09 |
| decode, JS debounce | 108.8 | 128.8 | 1.18 |
| decode, 15 tips | 87.5 | 101.9 | 1.17 |
| decode, MoE explain | 95.2 | 101.8 | 1.07 |
| decode, Python LRU class | 100.1 | 135.1 | 1.35 |
| decode, all prompts median | 95.2 | 104.2 | 1.09 |
| prompt 3.2K tok, tok/s (TTFT) | 978 (3.3 s) | 1705 (1.9 s) | 1.74 |
| prompt 12.7K tok | 1269 (10.0 s) | 2577 (4.9 s) | 2.03 |
| prompt 25.3K tok | 1742 (14.5 s) | 2786 (9.1 s) | 1.60 |

Caveats: the two engines run different files (Avalon's IQ3_KS hybrid is the larger, higher-quality quant; pancake's IQ2_XS is the
smaller one), so this compares speed, not answer quality. Avalon was rebuilt from the updated source (8b9a0f4, 50 commits newer than the
88313b6 baseline; most are CUDA-only): decode 95.2 vs 96.4 before, no gain on HIP. Model files SHA-256 verified before the run.

## Same model on all three engines (Qwen3.8-Flash-Next ISTA GSQ-RCO IQ2_XS, same GGUF files)

Avalon's llama-server loads the same IQ2_XS GGUF the Strata engines use. Avalon ran with its own drafting (its MTP head + ngram,
2 draft tokens), `-c 131072`, KV q8_0/q4_0 and a 4096-token batch (`-b/-ub 9536` ran out of VRAM on the 12.7K prompt with this quant;
4096 fixed it). Strata-rdna3 = `e2151f1b` with its own preset; pancake = `fc77f60c6` fast-hip. Run 1 of every case discarded.

| median | Avalon 8b9a0f4 | Strata-rdna3 | pancake |
| --- | ---: | ---: | ---: |
| decode, all prompts (tok/s) | 110.6 | 106.5 | 104.2 |
| decode, code prompts | 119-132 | 119 | 129-135 |
| decode, prose prompts | 98-111 | 88-107 | 93-102 |
| draft acceptance | 71% | 69% | 69% |
| prompt 3.2K tok (tok/s, TTFT) | 1532 (2.1 s) | 1544 (2.1 s) | 1705 (1.9 s) |
| prompt 12.7K tok | 1581 (8.0 s) | 2360 (5.4 s) | 2577 (4.9 s) |
| prompt 25.3K tok | 1626 (15.6 s) | 2624 (9.6 s) | 2786 (9.1 s) |

On identical weights Avalon decodes as fast as or slightly faster than the Strata engines (differences of a few percent are within
the 10-17 tok/s spread), but its prompt speed stays flat at ~1.5-1.6K tok/s while the Strata engines keep climbing to 2.4-2.8K tok/s
at 12-25K tokens. The first table above (Avalon on its own IQ3_KS hybrid) is therefore mostly a quant difference for decode and an
engine difference for long prompts.

## Strata-rdna3 re-test on the latest upstream (2b19d510, 11 commits newer than e2151f1b)

Rebuilt `build-hip/strata` from `nicejunjie/Strata-rdna3` main (2026-11-11 02:21: prefill grid above 32K, review fixes, megakernel
stall check, refill tuning; mostly multi-GPU / >32K paths). Same preset, same client, run 1 discarded.

| Strata-rdna3 | e2151f1b | 2b19d510 pass 1 (cold) | 2b19d510 pass 2 (warm server) |
| --- | ---: | ---: | ---: |
| decode, all prompts (tok/s) | 106.5 | 90.9 | 106.5 |
| prompt 3.2K / 12.7K / 25.3K (tok/s) | 1544 / 2360 / 2624 | 1547 / 2374 / 2649 | 1623 / 2432 / 2667 |

Pass 1 ran right after Avalon's 92 GB of model files had filled the page cache (server load 95 s instead of 50 s, I/O pressure high)
and its decode climbed over the rounds (MoE prompt 74 -> 85 -> 100 tok/s): a cold-start effect, not a regression. Discarding run 1
is not enough after a heavy file load; run the client twice on the same server and keep the second pass. The new upstream commits
change nothing measurable for 1-GPU decode and add ~0-5% on prompts.
