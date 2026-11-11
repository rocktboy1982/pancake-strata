# pancake vs Strata-rdna3 vs Avalon on an RX 7900 XT (2026-11-11)

One machine (Ryzen 7 5700G, 57 GB RAM, RX 7900 XT 20 GB, ROCm 10), one neutral client (`cmp_bench.py`) for every engine: same prompts,
greedy decoding, thinking off, the engines' own timings. Each engine ran alone; every server ran the client twice and the **second
pass is kept** (run 1 of each case is discarded as warm-up, and the whole first pass is discarded as a cold start; a cold page cache
after a heavy file load cost one build 15% in its first pass). Raw data and scripts: `raw/`.

Models (the same ISTA GSQ-RCO files for every engine): Qwen3.8-Flash-Next IQ2_XS and Q2_0, 128K context. Avalon was also run on its
own larger IQ3_KS hybrid (256K context). Avalon ran with its MTP head + ngram drafting (2 draft tokens) and a 4096-token batch (its
default 9536 ran out of VRAM on a 12.7K prompt with the smaller quants). Strata-rdna3 = nicejunjie/Strata-rdna3 `2b19d510` with its
own preset. pancake = our fork `fc77f60c6`, three presets: `hip` (exact), `fast-hip` (default), `turbo-hip` (fast + route prior, lossy).

## Results (medians; decode in tok/s; prompts in tok/s of prompt reading)

| engine / preset | model | decode, all | code | prose | 3.2K prompt | 12.7K | 25K | TTFT at 25K |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pancake `hip` (exact) | IQ2_XS | 84.1 | 99.8 | 80.4 | 707 | 788 | 787 | 32.2 s |
| pancake `fast-hip` | IQ2_XS | 104.8 | 132.3 | 98.8 | 1775 | 2612 | 2776 | 9.1 s |
| pancake `turbo-hip` | IQ2_XS | 96.2 | 146.2 | 85.6 | 1780 | 2605 | 2775 | 9.1 s |
| Strata-rdna3 | IQ2_XS | 106.5 | 116.5 | 97.5 | 1623 | 2432 | 2667 | 9.5 s |
| Avalon | IQ2_XS | 108.5 | 126.1 | 102.2 | 1531 | 1583 | 1617 | 15.7 s |
| pancake `hip` (exact) | Q2_0 | 84.7 | 110.2 | 82.0 | 754 | 830 | 827 | 30.7 s |
| pancake `fast-hip` | Q2_0 | 102.7 | 132.9 | 98.5 | 1977 | 2600 | 2772 | 9.1 s |
| pancake `turbo-hip` | Q2_0 | 98.8 | 119.8 | 97.0 | 2044 | 2606 | 2810 | 9.0 s |
| Strata-rdna3 | Q2_0 | 110.2 | 134.2 | 102.2 | 1805 | 2119 | 2204 | 11.5 s |
| Avalon | Q2_0 | does not run in this build (CUDA-graph assert: MUL_MAT_ID with host weights outside a hosted MoE block) | | | | | | |
| Avalon, own IQ3_KS hybrid (larger, higher quality) | IQ3_KS | 95.2 | 105.2 | 87.5 | 978 | 1269 | 1742 | 14.5 s |

Code = JS debounce and Python LRU prompts; prose = story, 15 tips, MoE explanation. Draft acceptance is 64-71% on every row.

## What the numbers say

1. **Long prompts are the real difference.** On identical IQ2_XS weights the Strata-family engines read 12-25K-token prompts at 2.4-2.8K
   tok/s; Avalon stays at 1.5-1.6K. At 25K tokens that is 9 s to first token against 15.7 s. The cause is the RDNA3 prompt kernels
   (WMMA attention, GDN chunking, fused prefill, the hipBLASLt gfx1100 table): pancake's exact `hip` preset, which lacks that switch stack,
   reads prompts at only 0.7-0.8K tok/s, slower than Avalon.
2. **Decode is a near tie.** On IQ2_XS: Avalon 108.5, Strata-rdna3 106.5, pancake 104.8; on Q2_0 Strata-rdna3 110.2 vs pancake 102.7.
   Run-to-run spread is 12-17 tok/s, so gaps under about 5% are noise. pancake is not faster than Strata-rdna3 on decode; its lead over
   it is prompt speed on Q2_0 (+23-26% at 12-25K tokens) and code decode on IQ2_XS (132 vs 117).
3. **Avalon wins prose, the Strata engines win code.** Prose 102 (Avalon) vs 98-99 (pancake) on IQ2_XS; code 132 (pancake) vs 126 (Avalon).
   Draft acceptance is similar, but per window Avalon commits more tokens on prose (2.1-2.3 vs 1.8-2.1) because it always drafts 2 tokens
   while pancake stops a draft chain at probability 0.5; its MTP head is also larger (1.48 GB IQ3_K/IQ4_NL, 65K-token vocabulary vs our
   0.71 GB Q2_0, 40.5K). Unconfirmed: derived from two requests per prompt and from reading code (`pancake-docs/PROSE_GAP_AVALON.md` on
   branch `dev/p17/prosegap`).
4. **Turbo (route prior) is not a default.** vs `fast-hip`: -8% decode on IQ2_XS, -4% on Q2_0, up to +9% on one copy-heavy code prompt
   and -15..-25% on prose, same acceptance, same prompt speed. Text similarity cannot judge its quality: the exact engine already
   differs run to run on every prompt (0 of 6 outputs identical exact vs exact).
5. **Q2_0 vs IQ2_XS: no clear speed difference** on pancake (decode 102.7 vs 104.8; prompts equal at 12-25K; +11% on the 3.2K prompt only).
6. **Avalon's own quant is slower, not the engine:** its IQ3_KS hybrid decodes at 95 tok/s (larger file); on the same file as the others it
   decodes at 108.5.

## Caveats

- Different drafting implementations per engine (Avalon ngram-simple + MTP depth 2; Strata engines MTP depth 4 + suffix drafter +
  lookup windows; pancake `fast-hip` also defers experts to the CPU, so it is not bit-exact with `hip`). All were run with their own defaults.
- Two passes per server, 3 timed runs per prompt size, 10 decode samples per row: treat differences of a few percent as noise.
- The pancake IQ2_XS `fast-hip` and `turbo-hip` rows and the Strata-rdna3 IQ2_XS row come from two-pass runs made earlier the same day
  (`raw/cmp-pf2.json`, `raw/cmp-pt2.json`, `raw/cmp-strata3.json`); every other row comes from the final matrix (`raw/matrix.sh`, `raw/m-*-p2.json`).
- Quality was not measured (the engines write different text for the same prompt); the larger Avalon quant likely answers better.
- An AMD consumer card; Avalon's CUDA path may rank differently on NVIDIA.
