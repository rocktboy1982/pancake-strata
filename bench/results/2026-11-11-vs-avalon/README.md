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
