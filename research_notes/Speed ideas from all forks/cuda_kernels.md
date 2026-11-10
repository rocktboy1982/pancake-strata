# CUDA / NVIDIA kernel speed ideas from the commits of all Strata forks

Scope: every Strata-fork commit in `data/fork-kb/forks.sqlite` (43,685 fork-only commits, `upstream_history = 0`, 2026-09-24 to 2026-11-10). A broad regex on the message matched 23,759 rows. A strict CUDA regex (cuda, sm_xx, RTX 30/40/50, mmvq, gemv, graph, tensor core, mma, pinned, pcie, ...) left 5,041 distinct SHAs. I dropped SHAs already cited in `reports/Forkuri Strata câștig viteză.md`, `reports/Forkuri Strata active noiembrie.md` and their `research_notes/` folders (248 SHAs). I then triaged by subject and opened 40 commits with `gh api` (about 45 API calls in total, including 3 file reads from KhaledElzarw/Strata).

Pancake reference points:
- Base is upstream `146df3c46` (v0.2.12, engine 0.1.58).
- RTX 3060 presets: `configs/pancake/flashnext-iq2_xs-fast.json` and `flashnext-q2_0-fast.json`. They set `STRATA_EXPDEC=1`, `STRATA_HC_Q8=1` (lossy), `STRATA_ATTN_DAILY=1`, `STRATA_PROMPT_ATTN_IMMA=1` and `STRATA_PREFILL_CPU_SHARE=auto` (lossy). `--pcie-frac` is not set, so the PCIe share is automatic, i.e. on.
- Upstream's sm_86 table (`src/core/nv_defaults.cpp:27-39`, measured on an A4000 and a 3060) is on by default: ADAPT_NOWAIT, CPU_IQ_ALL_EXACT1, SIDE_MIXER=2, ATTN_DAILY, GR_UP75, GU_SWIGLU, QSA_ROWS, B6_MMVQ_ROWS.
- The IQ2_XS file's experts are gate/up IQ2_S (34 layers), IQ2_XXS (11) and IQ1_M (3). **Every down projection is Q2_0** (`docs/PANCAKE_FINDINGS.md:176`).
- On the 3060, pancake measured 75 tok/s decode with 4,080 experts in VRAM. On the 7900 XT it measured 125-129 tok/s with 10,909 experts (memory file `project_rtx3060-setup.md`).
- The RTX 3060 is out of the box right now (the RX 7900 XT is installed). Every 3060 item is pending the card's return.

All speed numbers below were **measured by the author on the author's hardware**. None of them is verified on our box.

---

## 1. Which forks tuned Strata specifically for sm_86 (RTX 30), and what do they offer pancake on the 3060?

### Takeaway
The best source by far is **KhaledElzarw/Strata** ("STRATA sm86", RTX 3090, 2026-11-07..08). It is forked from **the exact same upstream commit as pancake (`146df3c46`)** and runs the same model family (Flash-Next GSQ-RCO, IQ3_XXS). It ships a set of bit-identical opt-in sm_86 decode switches that compound to +17.3% over its phase-1 config, with a paired 57-turn A/B per step. Earlier reviews only took its CPU-pool items. Their HIP-centric verdict "NOT-APPLICABLE as code" does not hold for the 3060: its GPU kernels port almost 1:1 to pancake's CUDA build. The second source is **AdrianBM96/Strata3060**, a real single-RTX-3060 fork, which earlier reviews did not mention at all. It gives one engine change (`STRATA_FETCH_ADMIT`, +2-7% decode on a 3060) and a careful measured list of what does *not* help on a 3060.

### Cited Findings

**KhaledElzarw/Strata, the sm86 lab (RTX 3090 24 GB, Ryzen 9 9950X, DDR5-5600, Windows 11, Flash-Next IQ3_XXS, 262K context)**
- The fork starts from upstream `146df3c46` (v0.1.58, 2026-11-07), the same base as pancake. Every switch is opt-in and claims bit-identical output, checked by kernel parity benches and a whole-engine forced decode of 1,363 tokens. A step was adopted only with a ≥2% gain and a 95% CI above zero, paired over 57 recorded agent turns. — [README](https://github.com/KhaledElzarw/Strata), [21630968f3](https://github.com/KhaledElzarw/Strata/commit/21630968f38042e391970226829f7d19515f381c) (2026-11-08)
- Per-step decode gains (geometric mean, 95% CI): phase-1 config vs stock +2.2%; phase-2 switches +6.0% (+3.9 to +8.1); HC read kernels +3.7% (+2.1 to +5.5); Q2_0 expert down rows +2.0% (+0.4 to +3.6); tier 5 +4.7% (+2.6 to +6.9). Compounded: +17.3% over phase 1, +19.9% over stock, 162-169 tok/s median decode. — [README](https://github.com/KhaledElzarw/Strata)
- The phase-2 switches and what each gave (NOTES.md):
  - `STRATA_GR_DOWN82=1` (HC down on 80 blocks instead of 41) + `STRATA_GR_UP_WIDE=1` + `STRATA_FETCH_FORK=1` + `STRATA_GRAPH_PRIO=1`: +1.0%.
  - `STRATA_MMVQ_IL_3090=1` (GEMV dispatch tables measured on the 3090): +1.8%.
  - `STRATA_EXP_MODE=9`: expert kernel time -11 to -17%, "but little end to end".
  - `--conversation-cache-park-gain 8192` and `STRATA_CKPT_*`: -13.5% prompt time per turn.
  — [sm86/NOTES.md](https://github.com/KhaledElzarw/Strata/blob/main/sm86/NOTES.md)
- **`STRATA_EXP_MODE=9`** ([9af8f1fa14](https://github.com/KhaledElzarw/Strata/commit/9af8f1fa14222c3c4e78cec688b251b80b419f78), 2026-11-07, +29/-1 in `iq_kernels.cu`):
  - What it does: gate/up for **IQ2_S, IQ2_XXS**, IQ3_S and IQ3_XXS take mode 8's fused kernel. **Down Q2_0** takes mode 8's LDS rows. IQ2_XS gate/up and IQ4_NL down keep mode 0.
  - Author's number: 0.83-0.89 of mode 0's time on all 48 layers, bitwise equal.
  - Why it fits us: our file's gate/up formats are IQ2_S and IQ2_XXS, and every down is Q2_0, so the table covers almost all of our experts.
- **`STRATA_EXP9_DOWN=42:10`**: Q2_0 down rows folded onto 4 lanes ([4d462ccff3](https://github.com/KhaledElzarw/Strata/commit/4d462ccff3e46308ea684ada1a3888e1cc2cd716), 2026-11-08, +90 lines in `iq_kernels.cu`).
  - What it does: `native_down_q2_sub4_kernel` folds the first two butterfly levels into in-lane adds in the same order. It uses 64 rows per block, keeps the q8_1 chunk in shared memory and decodes the weights once per entry.
  - Author's number: down time 0.47-0.96 of mode 8's (0.519-0.776 of mode 0's). Real-turn A/B +2.0% (CI 1.004-1.036) over the POOL_DAG arm, +1.6% (1.000-1.033) over plain base. Bitwise equal on all 30 Q2_0 layers.
  - Our case: our down is Q2_0 in **all 48 layers**, not 30. — [decision 0002](https://github.com/KhaledElzarw/Strata/blob/main/docs/decisions/0002-sm86-vram-expert-kernels.md)
- Same decision doc, Nsight Compute on the 3090: IQ2_XXS / IQ2_XS gate/up are issue-bound (~430 GB/s). Q2_0 down is latency-bound (~360 GB/s). The author also found that `STRATA_EXP9_GU=17:8` (IQ2_XS) "silently ran the unfused CUDA layout", because IQ2_XS is in neither the fused nor the LDS family. A new 16-lane decode-once gate/up kernel (mode 10) was bitwise but slower (IQ2_S 1.148x, IQ2_XXS 1.070x vs mode 8) and was dropped. — [decision 0002](https://github.com/KhaledElzarw/Strata/blob/main/docs/decisions/0002-sm86-vram-expert-kernels.md)
- **`STRATA_EXPDEC=1` was 3.0% slower** on this 3090, in the phase-1 tests. Pancake's 3060 presets set `STRATA_EXPDEC=1`. — [sm86/NOTES.md](https://github.com/KhaledElzarw/Strata/blob/main/sm86/NOTES.md)
- **`STRATA_HC_Q8=1` gave only +0.9%, "not worth its KL of 0.025"**, and the author dropped it. Pancake's 3060 fast presets run it. (Speedy measured +5.5-6.0% for it on its own box; see earlier report `speed_forks.md:84`.) — [sm86/NOTES.md](https://github.com/KhaledElzarw/Strata/blob/main/sm86/NOTES.md)
- **HC read kernels** ([857158c66d](https://github.com/KhaledElzarw/Strata/commit/857158c66dd59298dae78aaa8b2fecb7bcae63ab), 2026-11-08, +297/-21 in `fused_gr.cu`; [8817a1651d](https://github.com/KhaledElzarw/Strata/commit/8817a1651d417d5fa7527e839d94885828faf998), 2026-11-07, +313/-10):
  - What they are: `STRATA_GR_DOWN82` (80 blocks × 136 threads, lane-quarter split, last-block warp_sum). `STRATA_GR_DOWN86` (DOWN82's partition with token-parallel owners reading weight tiles through a **cp.async shared-memory ring**, 3-4 stages). `STRATA_GR_UP_HYB=3` (UP75 from 3 tokens). `STRATA_GR_XN=1` (the up kernel consumes the norm's own xn).
  - Author's number: A/B +2.3% paired (CI 1.015-1.042), hcbench hashes identical.
  - Problem statement: one HC read costs 30.8-60.4 µs at T=1..8 against a ~15 µs DRAM floor; 97 reads per window = 3.81 ms of 18.9 ms. — [decision 0001](https://github.com/KhaledElzarw/Strata/blob/main/docs/decisions/0001-sm86-hc-read-kernels.md)
- **Tier 5** ([20f1fa1a95](https://github.com/KhaledElzarw/Strata/commit/20f1fa1a9504571676ba049a9a8f2414c87454c7), 2026-11-08, +929/-34 in `fused_gr.cu`):
  - What it bundles: `STRATA_DOORBELL_STORE=1` (store the step's doorbell instead of a read-modify-write over PCIe), `STRATA_FUSE_HITS=1` (combine_k10_vec4_hits), `STRATA_ACTQ_SKIP=1`, `STRATA_GR_UP_PL=1`, `STRATA_GR_D87=8` (3-stage DOWN87).
  - Author's number: together +4.0% median / +4.7% geo-mean.
  - `STRATA_GR_RES=1`, one cooperative resident HC kernel, is ~4 µs faster per read alone but was neutral end to end (1.005 [0.990, 1.021]).
  - *Already covered:* ACTQ_SKIP is item 2.5 of `Forkuri Strata active noiembrie`. DOORBELL_STORE and FUSE_HITS exist in pancake (`verify.cpp`). GR_UP_PL, GR_D87, GR_DOWN82/86, GR_UP_HYB, GR_XN, EXP9_DOWN and MMVQ_IL_3090 are **not** in pancake (grep).
- Kept off by the author:
  - `STRATA_HC_PERSIST=1`: 0.5% slower.
  - PCIe share 0.5-0.6: **8-12% slower, "the CPU and PCIe read the same DDR5"**. A share of 0.4 was noise.
  - `STRATA_POOL_DAG=1`: neutral. The POOL_DAG=2 + SMALL_UNR=3 + ROUTE_BELL=2 bundle: +1.1% (CI -0.3 to +2.5). Earlier reports covered these.
  — [README](https://github.com/KhaledElzarw/Strata)
- Where the time goes on the 3090 (Nsight Systems, 484 windows): "the GPU is busy 98% of each window, and launch gaps cost only 0.3 ms per window, so kernels that only save launches do not help". When the expert kernels got 12% faster, "the GPU just waited longer for the CPU". Window breakdown: 18.9 ms = 11.0 ms waiting for the GPU dense chain + 5.0 ms CPU expert step + 1.5 drafting + 1.4 other. — [sm86/NOTES.md](https://github.com/KhaledElzarw/Strata/blob/main/sm86/NOTES.md), [README](https://github.com/KhaledElzarw/Strata)
- Free VRAM was the biggest lever: with VRAM-heavy desktop apps open, 6,377 experts and 126.5-131.5 tok/s; with them closed, 9,724 experts and 150.0-151.2 tok/s. Moving the display to the iGPU freed 1.27 GB more. — [sm86/NOTES.md](https://github.com/KhaledElzarw/Strata/blob/main/sm86/NOTES.md)

**AdrianBM96/Strata3060 (single RTX 3060 12 GB, PCIe Gen4 x16, 2×32 GB DDR4-2133, Linux; Flash-Next IQ2_XS)**
- **`STRATA_FETCH_ADMIT=1`** ([50581c24f2](https://github.com/AdrianBM96/Strata3060/commit/50581c24f2b1d35e8c76e53d1037fcc25672c0f9), 2026-10-04):
  - What it does: a verify window's PCIe share of the misses already crosses the link into staging and is then dropped. With this switch each such blob goes straight into the cache slot of the least-used resident of its layer that the window did not route, so it is a VRAM hit from the next window. "Same bytes over PCIe; this window's result unchanged."
  - Code: verify_kernels `fetch_blobs_admit`/`rebase_ptrs_admit`, an admit2 array at the end of the plan, victim choice in expert_source, guards in generate.cpp.
  - Exactness: exact for the current window; it changes later placement, the same class as the adaptive tier.
- FETCH_ADMIT measured on the 3060 ([3ff40d67e3](https://github.com/AdrianBM96/Strata3060/commit/3ff40d67e3431ad23abe24268472afca59b8d843), 2026-10-04): off→on medians B1 45.25→47.55 (+5.1%), B2 42.75→45.75 (+7.0%), B4 42.4→44.55 (+5.1%). With other adapt settings: +2.3 to +4.2%, two cells -1.2 / -4.7%. Quality: top-1 99.9%, KL 0.0020, ΔNLL "no measurable difference". Cache hit rate 73-78%. Adopted.
- On the 3060, upstream 0.1.39 vs 0.1.38: decode +7.1%, prompt +9.0%. Four new switches did nothing: `STRATA_TOPK_OLD`, `STRATA_IQ_STAGE_GRID`, `STRATA_SH_STREAM`, `STRATA_RING_BYTES` (all noise). — [82c9640da5](https://github.com/AdrianBM96/Strata3060/commit/82c9640da5dfb4848a378843539dc691684e9c30) (2026-10-04)
- `STRATA_PF_FUSED=1` (fused int8 prompt experts): **+5.2% prefill** on the 3060, measured twice at 12.3K and 11.4K tokens, paired output identical. GPU fan at 100% removed thermal slowdown but gave the same tok/s. A 150 W power limit was within noise. — [5b80040b2e](https://github.com/AdrianBM96/Strata3060/commit/5b80040b2e7495e677296b870fc9a3859286a5f4) (2026-10-04)
- On the 3060, `GR_DOWN_MAX4` was +1.2% (noise). AVX-VNNI+gather was within noise for IQ2_XS. — [bd532063a4](https://github.com/AdrianBM96/Strata3060/commit/bd532063a48528bc3a235db4d10a5a100224b833)
- `HIT_GY=16/24` gave no gain. `PROFILE_HEAT_MIN=200000` was enabled for observation. — [aa426de6dc](https://github.com/AdrianBM96/Strata3060/commit/aa426de6dcba55e34853b4cce3191cd41c34c071)
- 3060 GPU profile per B1 window (~2.3 tokens, ~44 ms):
  - GPU-reach wait ~22 ms + per-layer host ~14-19 ms (mostly CPU experts).
  - Per stage on GDN layers, ms/window: hc-read1+router 3.4; q8+qkv gemv 2.4; head 1.8; **waitB (PCIe) 9-10**; VRAM hits 5.2; waitCPU 0.9-2.5; copy+combine 0.5. QSA layers ~5 ms.
  - Graph: 3,149 nodes for a 1-token window, 3,265 for 4 tokens.
  — [ae8ec6d29c](https://github.com/AdrianBM96/Strata3060/commit/ae8ec6d29c1619b8776c213671fd674672f6e005)
- PCIe on that 3060: the link probe reads **11 GB/s** on Gen4 x16. `STRATA_VERIFY_PROFILE` dma + frac 0.30 cut waitB 10.22→8.98 ms (-12%), but net per token was ~1.6% (noise). — [213c383640](https://github.com/AdrianBM96/Strata3060/commit/213c383640101dbb29fd6b244e292f05dd985dd2)

**eddoursul/Lamina (RTX 3090 sm_86 + RTX 5070 Ti; diverged from upstream, "leaving upstream" on 2026-10-29)**
- Batch6 of the earlier review only took its global-address hint `33f63cf1f5`. Its September-October sm_86 kernel commits are new here (sections 2 and 4).
- `native_mmvq`: upstream's ncols > 1 layout timed on sm_86 (RTX 3090) gave no gain over the exact layout. — [77665b73f5](https://github.com/eddoursul/Lamina/commit/77665b73f525e31d9b0b28a236810c706c6cb6d3) (2026-10-14)

**imanu86/Strata-2080Ti**
- `STRATA_HC_DOWN_W4=1`: the staged HC down in 4-warp blocks (80+1 blocks instead of 40+1). In the author's words, with 41 blocks "27 of the 68 SMs of the 2080 Ti stayed idle, **the 3060 made 1.5 waves**". Also `STRATA_HC_UP_FAST=1`, the AMD `gr_up_fast_kernel` ported to CUDA. Both bitwise (gr_parity). Not measured end to end. — [185b57918a](https://github.com/imanu86/Strata-2080Ti/commit/185b57918a6f52e87324313dba936fc401a4f39b) (2026-10-06)

### Inferences
- The 3060 has 28 SMs. DOWN82's 80 blocks × 136 threads at 136 registers use ~18.5K registers per block, so 3 blocks fit per SM (65,536 registers) and 84 slots ≥ 80: **one wave on the 3060**, against ~1.5 waves for the 41-block default. That is the same mechanism imanu86 named. The HC down gain could be relatively larger on the 3060 than on the 82-SM 3090. This is arithmetic, not measured.
- The 3090 dense chain is 11 ms of an 18.9 ms window. The 3060 has 360 GB/s vs 936 GB/s, so the same chain should take ~2-2.5x longer. HC/dense-projection kernel gains should then weigh *more* on the 3060. Expert-kernel gains on the GPU side weigh *less*: 4,080 slots vs 9,500, and the CPU pool on DDR4 is the long pole. This is an inference to verify with `STRATA_DECODE_TIMING` on the 3060.
- `STRATA_MMVQ_IL_3090` tables are tuned for 82 SMs. The right move is to rerun `mmvq_il_parity --bench` on the 3060 and write a `_3060` table. Upstream's sm_86 table has `B6_MMVQ_ROWS=1` worth -1.7 points on the 3060 when removed, so the 3060 does respond to GEMV dispatch tuning.
- Pancake's 3060 presets carry two settings the closest sm_86 lab rejected: `STRATA_EXPDEC=1` (-3.0% on the 3090) and `STRATA_HC_Q8=1` (lossy, +0.9%). Both deserve a fresh A/B on the 3060 before any kernel port. They cost zero code.
- **Cheapest 3060 test:** `STRATA_FETCH_ADMIT` is about 150 lines on the same engine generation, and our 3060 preset runs with the PCIe share on (no `--pcie-frac`). It applies directly and is the cheapest likely +2-7%.

### Gaps
- No 3060 numbers exist for any KhaledElzarw switch. MMVQ_IL tables and DOWN86 ring depth (3 vs 4 stages need ~shared memory per block; the 3060 has 100 KB per SM like the 3090) need a rerun.
- AdrianBM96's FETCH_ADMIT was measured on 0.1.39. Pancake (0.1.58) also has `--adapt-async 1 --adapt-swaps 80` and PIN_BY_NEED, and the commit's guards refuse `ADAPT_NOWAIT`, which pancake's sm_86 table sets. Whether it composes is unknown until built.
- I could not determine the 5700G box's real host→device rate. The 5700G APU is limited to PCIe 3.0, so expect roughly 11-12 GB/s like AdrianBM96's 11 GB/s on Gen4 (unverified).

---

## 2. Which decode GEMV / MMVQ / grouped-expert kernel changes for low-bit formats (IQ2_XS family, Q2_0, IQ3, Q4) are worth porting?

### Takeaway
For our format mix (IQ2_S/IQ2_XXS gate/up, Q2_0 down), the directly relevant new kernels are three:
- KhaledElzarw's EXP_MODE=9 per-format table plus the Q2_0 4-lane down (section 1);
- Lamina's sub-warp grouped kernels (`native_gus80_kernel`, `native_down_sub_kernel`; 3090 expert time -3..-38% by token count, +5.7-8.3% end to end on IQ3_S);
- Lamina's int8 tensor-core path for 2-8-token verify-window dense projections and the head (`native_mma`; +2.2-6.3% end to end, rounding-level, not bitwise).

Several older ones are already upstream and in pancake: noorazman's Q2_0 grouped kernels, decode-once IQ, and Lamina's `iq_dot.cuh`.

### Cited Findings
- **Lamina, rows on sub-warps with each weight decoded once for several entries** ([674bf0caf2](https://github.com/eddoursul/Lamina/commit/674bf0caf236b9276bd82a73b3e6fb948691a6d8), 2026-10-13, +263 `iq_dot.cuh`, +198 `iq_kernels.cu`):
  - What it does: `native_gus80_kernel` covers gate/up rows of 80 calls at n_embd 2560 for IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS and Q2_0. It gives a half-warp a column, decodes a call once for two entries, runs 2 blocks/SM at 64 registers and puts the codebook in shared memory. `native_down_sub_kernel` (n_ff 640) does IQ4_NL down rows on 8 lanes and Q2_0 on 4, up to 4 entries per pass. Bitwise, by folding lane sums in warp_sum's order.
  - Author's numbers (L2 flushed, at 1/2/4/8 tokens): 3090 -3..-19% / -8..-31% / -21..-36% / -30..-38%. CUPTI on the 3090: experts per layer 36.7→32.7 µs at 1 token, 75.3→56.1 at 4, 133.9→94.3 at 8.
  - End to end (IQ3_S model): code +5.9%, thinking +5.7%, after 16K +6.4%, edit +8.3%. UD-Q4_K_XL unchanged.
  - Status in pancake: not present (grep `native_gus80`, `native_down_sub` → none).
- **Lamina, grouped native experts in two kernels** ([cf307ea616](https://github.com/eddoursul/Lamina/commit/cf307ea616073bac53a1dbe459c4111182719fd6), 2026-09-26): gate/up + SwiGLU + q8_1 in one block per group chunk, then down (4 down rows per warp), bitwise. 3-21% faster per call. 3090+5070 Ti end to end: IQ3_XXS code 108.9→111.5 t/s (+2.4%), thinking 93.2→96.2 (+3.2%). Pancake already has the idea: `STRATA_EXPDEC_SWIGLU=1`, MOD_PLAN item 48, `iq_kernels.cu:2639`, and `iq_dot.cuh` via upstream "eddoursul/Strata 8029fa9" (`iq_kernels.cu:327`). ALREADY-HAVE (functionally).
- **Lamina, `native_mma`** ([b4187bc3db](https://github.com/eddoursul/Lamina/commit/b4187bc3dbe8cedc74bd01c3cac1b82595d4ffb8), 2026-10-16, new 436-line `native_mma.cu`):
  - What it does: the verify window's dense projections of 2-8 tokens and the head run on **int8 tensor cores**. 16-row tiles of Q6_K/Q5_K/Q4_K/IQ4_XS blocks go into shared memory a stage ahead by 16-byte cp.async and are decoded once into the A operand of `mma.m16n8k16/k32`. The window's tokens are N (zero-padded to 8). A table measured on the 3090 picks it: Q6_K from 2 tokens, other formats from 5-6.
  - Kernel numbers: Q6_K qkv 30 µs at 2-8 tokens vs 30.7-40.3 for dp4a/MMQ; head on the 3090 343-353 µs vs 361-620.
  - End to end on IQ3_S: code +6.3%, thinking +2.2%, after 16K +3.3%, edit +3.8%.
  - Exactness: within 1e-3 of `native_mmvq` (measured 2-7e-5), so lossy at rounding level, not bitwise.
  - Status in pancake: no `native_mma` in the tree. Pancake's head is IQ4_XS (`iq4xs_head.cu`, exact GEMV, memory file).
- **Lamina, Q6_K dp4a in two fewer ops** ([df47127562](https://github.com/eddoursul/Lamina/commit/df471275629a6860ffc7c047eb30a7dbb847089d), 2026-10-18): "quants less 32" as IADD3 + LOP3 (lo + hi + 96, top bit flipped) instead of `__vsubss4`, which is eight instructions on sm_86. Same bytes. 3090 1-token Q6_K projections -1.2..-6.2% (e.g. 2560→12288 31.8→31.2 µs). Only matters where dense Q6_K tensors exist. Not in pancake (grep `strata_q6_quants` → none).
- **KhaledElzarw `STRATA_MMVQ_IL_3090`** ([ca1edf8786](https://github.com/KhaledElzarw/Strata/commit/ca1edf87869ff1113527d2ec18f93a3c4d7c6c3e), 2026-11-07, +43 in `native_mmvq.cu`): the 2-4, 5-8 and one-column rows-per-warp tables, read off three `mmvq_il_parity --bench` runs (3% rule). Every choice is bitwise `native_mmvq`. +1.8% end to end on the 3090 (NOTES).
- **noorazman, faster grouped and per-hit Q2_0 kernels** ([ba2e800026](https://github.com/noorazman/Strata/commit/ba2e80002637b5544baebc2b73437b2a130f46f0), 2026-09-30): conflict-free word-major shared layout (the old xs_q layout had an 8-way bank conflict), hx summed once per chunk, aligned word loads. Bitwise. Also decode-once IQ weights per column ([29139200c6](https://github.com/noorazman/Strata/commit/29139200c68fb2f29ee2b4c91c01a599df1e75cd)). **ALREADY-HAVE**: `STRATA_OLD_GROUPED` (`s2_expert_grouped.cu:246,530`) and `STRATA_OLD_IQ_MMVQ` (`iq_kernels.cu`) are in pancake.
- **svgop mmvq_il wide** ([db03282c19](https://github.com/svgop/Strata/commit/db03282c19), 2026-10-20): the 5-8-column interleaved matvec with sm_120 tables, bitwise. On an RTX 5090: conc +8.34%, solo +2.63% (sub-bar). Head bench T5 491.6→290.6 µs. It is upstream v0.2.1's, and pancake has `MMVQ_IL` (`mmvq_il_parity.cpp`, `verify.cpp`). ALREADY-HAVE. The sm_86 equivalent is a table (see KhaledElzarw).
- **atmJava (RTX 2080 Ti class, Linux)**: "VRAM hits at 1.52 ms a window - 0.063 ms a layer for 19.83 hits of 1.96 MB, about 620 GB/s… there is no kernel win to take. What is idle is waitCPU." — [541b5d0ead](https://github.com/atmJava/Strata/commit/541b5d0ead20b12a258d595bc72842b5c3fdce01) (2026-10-24). Contrast: KhaledElzarw measured the IQ2 gate/up issue-bound at ~430 GB/s of 936 GB/s on the 3090 ([decision 0002](https://github.com/KhaledElzarw/Strata/blob/main/docs/decisions/0002-sm86-vram-expert-kernels.md)). The two disagree on headroom. Different cards and formats likely explain it.
- **lighttransport `native_tc_mmvq`** ("weights-once tensor-core matvec") is for multi-sequence batch decode (`--seqs N`) on its MiniMax M3 engine. — [e30f6e8f06](https://github.com/lighttransport/Strata/commit/e30f6e8f06e2e8c84a000caaa9a3ce22af776878) (2026-11-03). NOT-APPLICABLE for solo decode.

### Inferences
- Our IQ2_XS file has Q2_0 downs in all 48 layers, against KhaledElzarw's 30. A Q2_0 down kernel change (`EXP9_DOWN=42:10`, or Lamina's 4-lane `native_down_sub_kernel`, which is the same idea) should be worth at least as much here, *if* the down kernels are on the critical path on the 3060.
- On the 3060, the first experiment is a 3-arm env A/B, no code: `STRATA_EXPDEC=1` (current) vs unset vs `STRATA_EXP_MODE=8`. Pancake's tree has mode 8 (`iq_kernels.cu:5694-5711`, "Volta takes mode 8 by default"). EXP_MODE=9's table and EXP9_DOWN are a ~120-line port on the same base.
- The `native_mma` path is lossy at rounding level, so it fits pancake's "fast" preset tier, not the exact one. Lamina gets more from it on Q6_K. Pancake's dense formats on the IQ2_XS pack need checking first: the head is IQ4_XS, the HC is BF16 or Q8.

### Gaps
- I did not open Lamina's `iq_dot.cuh` diff line by line. Whether `native_gus80_kernel`'s IQ2_S/IQ2_XXS paths win at our typical 1.3 entries per group (KhaledElzarw: "~1.3 entries per VRAM group" at T≈3.55) is unmeasured. Lamina's gain grows with tokens per group.
- Pancake's `STRATA_EXPDEC` CUDA kernel vs KhaledElzarw's EXP_MODE=9 has never been compared on any sm_86 card in the same build.

---

## 3. Which attention kernel ideas (decode window, prompt, int8/FP16 tensor cores) could speed pancake up?

### Takeaway
The one big NEW item is **Asofwar/Strata-Turbo's window decode attention**: each K/V chunk is read once for up to 8 queries × 8 heads instead of once per query, then int8-MMA small-window kernels. The author measured +37-83% decode at 130K-200K context on an RTX 5080, lossy at rounding level. It targets long-context decode, which is where pancake drops (75 tok/s short vs much lower after 42K on the 3060). The other attention items (int8 prompt attention, ATTN_DAILY, QSA_SPLIT_ATTN, QSA_ROWS) are already in pancake or upstream.

### Cited Findings
- **`STRATA_ATTN_WINDOW=1`** ([6ebe841e2e](https://github.com/Asofwar/Strata-Turbo/commit/6ebe841e2e), 2026-10-20): "a verify window's dense int8-KV attention through the prompt kernel split over the cells, each K/V chunk read once for up to 8 queries x 8 heads instead of once per query". `STRATA_ATTN_WINDOW_PREC` picks the FP16 modes. RTX 5080 + i9-14900K, Cyber-Tiel-Coder UD-IQ4_XS on the Flash-Next engine path: 200K decode 48.6 → 68.2 tok/s (precision 0) / 88.9 (precision 2); 130K 54.3 → 66.6 / 107.5. Greedy text within run-to-run noise. Default kernels unchanged (same SASS).
- **`STRATA_ATTN_WINDOW_SMALL=1`** ([540f801a9a](https://github.com/Asofwar/Strata-Turbo/commit/540f801a9a), 2026-10-21): windows of up to 4 queries on `decode_attn_window_small_kernel`. It computes only the window's MMA rows with DENSE_IMMA arithmetic, four warps share one double-buffered stream of cells, 2 blocks/SM. Parity at 200K cells T=1..4: 267-342 µs vs 465-474 for WINDOW_PREC=2, error 5e-5..1.7e-4. Together with DENSE_IMMA=3, engine decode after 130K/200K went 112.6/127.6 → 146.4/160.7 tok/s (+30/+26%). — [540f801a9a](https://github.com/Asofwar/Strata-Turbo/commit/540f801a9a), [aac0e4f6d9](https://github.com/Asofwar/Strata-Turbo/commit/aac0e4f6d988f2dfb8f14dbc47cceabffb3f80d3)
- **`STRATA_PROMPT_ATTN_DENSE_IMMA=2/3/4`**, dense prompt attention on INT8 tensor cores with int8 KV ([48c58c0361](https://github.com/Asofwar/Strata-Turbo/commit/48c58c0361), [1fa9bbcdce](https://github.com/Asofwar/Strata-Turbo/commit/1fa9bbcdce), consolidated in [f43c719740](https://github.com/Asofwar/Strata-Turbo/commit/f43c719740), 2026-10-21..22). Per 64K chunk on the 5080: 118-127 ms (FP16ACC=2) → 85-97 (=2) → 80-82 (=3) → 62.2 (=4). Prompt 200K: 76.7 → 53.8 s in steps ([c8808f343f](https://github.com/Asofwar/Strata-Turbo/commit/c8808f343f)). Error vs FP32 8.9e-5 (better than FP16ACC=2's 6.3e-4).
- Asofwar: `--kv k8v4` instead of int8 loses these int8-only kernels. Prompts became 7-9x slower and decode 168.8→107.9 (130K). Keep `--kv int8`. — [d7909eed8d](https://github.com/Asofwar/Strata-Turbo/commit/d7909eed8d) (2026-10-22). Pancake's 3060 presets already use `--kv int8`.
- Pancake's tree has none of `STRATA_ATTN_WINDOW`, `ATTN_WINDOW_PREC`, `ATTN_WINDOW_SMALL`, `PROMPT_ATTN_DENSE_IMMA` (grep). It does have `STRATA_PROMPT_ATTN_IMMA` (svgop's int8-MMA prompt attention bundle, adopted on a 5090 with pair +6.5% / conc +3.9%, [c87d6d7917](https://github.com/svgop/Strata/commit/c87d6d7917491beea4be02548d24a4238d003dd6)), `QSA_SPLIT_ATTN`, `ATTN_DAILY` and `QSA_ROWS`.
- **Volta / Turing tensor-core attention** (jmnargi/Strata-V100 m8n8k4, 1.6-1.9x on the QSA kernel, [4f96da29f5](https://github.com/jmnargi/Strata-V100/commit/4f96da29f5); yqYo1 Turing register-accumulator kernel, [83c209a9f6](https://github.com/yqYo1/Strata/commit/83c209a9f6)): sm_70/sm_75 variants of what sm_80+ already runs. NOT-APPLICABLE to the 3060.
- **xrip `STRATA_QSA_ROWS=1`**: queries in registers with six shuffles per block. 2080 Ti 101→55 µs ([2b7cd07c9a](https://github.com/xrip/Strata/commit/2b7cd07c9a), 2026-10-12). Already upstream and in the sm_86 default table (A4000 -0.5, 3060 -1.2 points when removed, `nv_defaults.cpp:38`). ALREADY-HAVE.
- **dongnh311 (q36 engine)**: decode attention reads the KV cells once per KV head, not once per query head (16K: "10x faster kernel") — [62cc598f3b](https://github.com/dongnh311/Strata/commit/62cc598f3b) (2026-10-16). This is the GQA analogue of Asofwar's idea, for the Qwen3.6 path only.

### Inferences
- Asofwar's gain grows with context because attention's K/V reads grow with context while expert reads do not. At 32-64K on our box the gain should be smaller than +30%, but pancake's measured drop after an 11.7K prompt (125-129 → 105 tok/s on the 7900 XT) shows attention is already visible there. A 3060 A/B at 32K/64K/128K is the right gate.
- Asofwar's "dense int8-KV attention" path must be the one pancake's decode takes on QSA layers. If Flash-Next's QSA top-k selected path is used instead past some length, the window trick needs re-targeting at `qsa_decode_attn.cu`.

### Gaps
- I did not open the Asofwar kernel diffs. Size, exact files and whether they rebase on 0.1.58 are unknown; Strata-Turbo forked around 0.1.45-0.2.x.
- Update 2026-11-10 (diffs opened, MOD_PLAN F11 status): the window kernel is a `SPLIT` instance of the fork's dense int8
  `prompt_attn_dense_kernel`, called from the main engine only where there is no indexer (qwen35moe on the fork's main
  engine), GQA 8 only. It does not rebase: pancake has no dense prompt kernel, Flash-Next always has the QSA indexer
  (<= 2,051 cells a row, attention ~3% of a window), and pancake runs qwen35moe on `src/qwen35` or strata-poly. The second
  inference above is answered: the dense path is not the one Flash-Next takes. The idea was rewritten for the Qwen3.6
  engine (`k_attn_window`, branch work/f11-attn-window).
- No measurements exist below 130K context, nor on Ampere. The 5080 has a different int8 MMA rate than the 3060.

---

## 4. CUDA graphs, launch overhead and fused kernels (router, SwiGLU, norm/HC, head): what do forks show?

### Takeaway
On sm_86 the evidence says **launch overhead is not the bottleneck**:
- 0.3 ms of launch gaps per 18.9 ms window on a 3090 with graphs;
- svgop's "node savings != wall time";
- AdrianBM96's ~3.2K-node window graph on the 3060.

Fusions pay only when they cut serial latency on the dense chain (router, HC read) or bytes. The useful NEW items are small: Lamina's router (+2.5%, `__reduce_max_sync` top-10, the mapped copy in its own blocks, one fence round), and KhaledElzarw's router+doorbell in one launch (1.0-1.8 µs/layer, bundled result below the bar). On HIP, launches are much more expensive, so the same fusions are worth more on the 7900 XT.

### Cited Findings
- KhaledElzarw (3090): "the GPU is busy 98% of each window, and launch gaps cost only 0.3 ms per window, so kernels that only save launches do not help." — [sm86/NOTES.md](https://github.com/KhaledElzarw/Strata/blob/main/sm86/NOTES.md)
- AdrianBM96 (3060): window graph 3,149 nodes at 1 token, 3,265 at 4 tokens, "almost all type0 + ~12 memcpy". — [ae8ec6d29c](https://github.com/AdrianBM96/Strata3060/commit/ae8ec6d29c1619b8776c213671fd674672f6e005)
- svgop (RTX 5090 class): "fusion tranche clean negative (node savings != wall time; LFUSE/FUSE_WAIT/DF_PLE stay off)". — [ee0ccc6afa](https://github.com/svgop/Strata/commit/ee0ccc6afa) (2026-10-15). Its T7 verdict: "overhead-bound (roofline 2.5x current)", with HC kernel fusion estimated at +7.6% end to end. — [977753ef3a](https://github.com/svgop/Strata/commit/977753ef3a7dd50504f29ba18208153e8ed8cf75)
- svgop: PDL (programmatic dependent launch) "+5-22% tg Blackwell, parity-gated". — [7cc372c2db](https://github.com/svgop/Strata/commit/7cc372c2db99bf63005c8758f1b443c659aa12ce). Pancake has `include/strata/kernels/pdl.hpp` (`cudaLaunchKernelEx` with programmaticStreamSerialization). PDL needs sm_90+, so NOT-APPLICABLE to the 3060 or HIP.
- **Lamina router** ([f14591215f](https://github.com/eddoursul/Lamina/commit/f14591215f01afb8c0cf188f5800aba82d5fa9fc), 2026-09-27, +50/-39 in `native_router.cu`). Bitwise. Three changes:
  1. The window input's mapped copy runs in blocks of its own, scheduled first. Previously the first GEMV blocks did it and system-fenced, which stretched the kernel over the 3090's x4 link at ~6 GB/s.
  2. One fence round, from the routing warps only. Each fence cost ~2 µs.
  3. Top-10 via `__reduce_max_sync`/`__reduce_min_sync` on an order-preserving key, replacing a 16-value scan + 5-level shuffle tree per round.
  - Kernel: 17.6→15.6 µs at 1 token, 24.1→21.6 at 4, 35.0→32.4 at 8. In-window 22-30 µs/layer (from 26-42).
  - End to end: IQ3_XXS code +2.5%, thinking +0.9%, after 16K +2.7%.
  - Pancake's `native_router.cu:82,136,193` still uses the shuffle-tree top-k.
- KhaledElzarw `STRATA_ROUTE_BELL=1|2`: route_multi's top-10 and the doorbell in one kernel; =2 adds the REDUX top-10 (`__reduce_max/min_sync`). Parity 47.7M values, 0 differ. Saves 1.0-1.8 µs per layer. `MMVF_ROWS=1` and a 5-rows-per-block GEMV gave no gain. — [0af4328ec8](https://github.com/KhaledElzarw/Strata/commit/0af4328ec89113d00843eb39fdc8fe89bb1f9996) (2026-11-08). The bundle with POOL_DAG=2 + SMALL_UNR=3 measured +1.1% (not adopted). *Earlier report covered ROUTE_BELL as "NOT-APPLICABLE as code" for HIP.* Pancake has its own `STRATA_ROUTER_FUSE=1` (router GEMV's last block routes and publishes, `native_bf16.cu:122,472`, MOD_PLAN item 48), on CUDA too.
- Asofwar `router_top10_fast_kernel` on CUDA: 15.5→8.9 µs per 256×8 call on a 5080, short chat 235.7→246.7 tok/s (another model). — [557d96a838](https://github.com/Asofwar/Strata-Turbo/commit/557d96a838). ALREADY-HAVE (`router_top10.cu` has `router_top10_fast` and `STRATA_ROUTER_OLD`).
- **HC (hyper-connection) read fusions, all bitwise:**
  - KhaledElzarw DOWN82/86, UP_HYB, XN, UP_PL, D87 (section 1). `GR_RES=1` (whole HC read in one cooperative kernel) was neutral.
  - imanu86 HC_DOWN_W4 / HC_UP_FAST (section 1).
  - zsariboga `STRATA_GR_NORM_FAST=1` on CUDA, V100 +0.5-0.6% ([56107b2262](https://github.com/zsariboga/Strata/commit/56107b2262)). svgop adopted it on a 5090 with M1 +5.2% (pooled 6v6, [3dccd89670](https://github.com/svgop/Strata/commit/3dccd89670)). Pancake has `GR_NORM_FAST` in `fused_gr.cu`, opt-in.
  - svgop `GR_DOWN_TG` (upstream v0.2.1) is in pancake.
- **SwiGLU/q8_1 fusion:** Lamina cf307ea616 (section 2); zsariboga's prefill dense IQ4_XS/Q8_0→FP16 with 16-byte stores, swiglu + q8_1 and MoE combine + HC write fused (32K prompt -4.0%, same bits) — [21107e9d05](https://github.com/zsariboga/Strata/commit/21107e9d05) (2026-10-08). Pancake has `STRATA_EXPDEC_SWIGLU`, `GU_SWIGLU` (sm_86 table) and `PF_FUSED`.
- **CUDA stack frame:** a kernel's by-value pointer array took a 128-byte stack frame past the engine's 104-byte limit. The driver then **grew the stack limit at every process's first prompt, mid-request**. Lamina fixed it with a `__grid_constant__` parameter (0 bytes, `cuobjdump -res-usage`). — [95fa8c2321](https://github.com/eddoursul/Lamina/commit/95fa8c2321c7ec3a3fc3fa8b4989e55038fddd0a) (2026-10-18), and [fdf559026f](https://github.com/eddoursul/Lamina/commit/fdf559026f) (stack limit set at the largest frame per GPU).
- **`--no-graphs`** +29%/+18% only on a WDDM 4 GB laptop ([4a9539f2e5](https://github.com/RandoTeam/4gbVramStrata/commit/4a9539f2e5)). *Already covered (batch6, NOT-APPLICABLE on Linux).*
- **HIP launch cost:** "a 48 CUs x 8 launch costs 14-16 us on gfx1100 even when empty". Cutting the PCIe share's copy kernel from 384 to 96 blocks gave decode +6% over eight prompts, same text, on 2× RX 7900 XTX. — [d0c8d6c3e2](https://github.com/unimatrix099/Strata/commit/d0c8d6c3e2c1607447ff342b2aa1fd59c385b1b1) (2026-10-10)

### Inferences
- On the 3060, prefer fusions that shorten the serial dense chain (HC read, router) over pure launch-count cuts. The Lamina router change is small (~90 lines), bitwise and touches `native_router.cu`, which pancake keeps upstream-identical. It is a cheap +1-2.5% candidate. `__reduce_max_sync` exists on sm_80+, so the 3060 has it.
- Pancake's `STRATA_ROUTER_FUSE` and KhaledElzarw's `ROUTE_BELL` overlap. Measure pancake's own first before porting.
- Check pancake's CUDA build for kernels with stack frames above the engine's limit (`cuobjdump -res-usage | grep STACK`). A mid-request stack-limit growth would show as first-prompt latency, not decode tok/s. This is hygiene, not throughput.

### Gaps
- No fork measured CUDA graph on/off on Linux sm_86 for Flash-Next. FastStrata's `--graph-experts` (-7-10%) is from the Qwen3.6 poly engine (memory file), not a fork commit in this KB.

---

## 5. Memory, PCIe copies, pinned memory and streams: which fork changes matter for a 3060 on a DDR4 / PCIe 3.0 box?

### Takeaway
The PCIe share is double-edged on a DDR-bound box:
- the closest sm_86 lab found shares of 0.5-0.6 8-12% slower, because CPU experts and PCIe DMA read the same DRAM;
- JEMostert's adaptive balancing on a 4090 settled at 0.00-0.06 and gave +22% short decode;
- svgop on a 5090 adopted 0.55 (+2.8%).

So on the 3060 the first job is an env A/B of `--pcie-frac 0` vs auto vs `STRATA_PCIE_BALANCE=1`. Only if a non-zero share wins do the PCIe-share refinements pay: FETCH_ADMIT (+2-7% on a 3060), FETCH_FORK, DOORBELL_STORE.

### Cited Findings
- KhaledElzarw: PCIe share 0.5-0.6 is 8-12% slower ("the CPU and PCIe read the same DDR5"); 0.4 is noise. — [README](https://github.com/KhaledElzarw/Strata), [NOTES](https://github.com/KhaledElzarw/Strata/blob/main/sm86/NOTES.md)
- JEMostert, PCIe share balanced per layer while decoding:
  - What it does: the GPU's wait for CPU rows is stored to a mapped word, and the pool steps the share up or down, weighted by expert cost on each side. `STRATA_PCIE_ADAPT=0` disables it.
  - RTX 4090 on PCIe 4.0 x8 (12.9 GB/s), i9-13900K, IQ2_XS: the probe gave 0.35 and the GPU waited ~13 ms of a 27 ms window on copies. Balanced, the share settles at 0.00-0.06. Short / 4.8K code review / essay: 116.5/110.4/105.6 → 142.9/118.1/114.8 tok/s.
  - Also: GDN gate (z) and alpha/beta projections on a side stream beside qkv+conv, `STRATA_GDN_FORK=0` to disable; answer identical with fixed cache and `--pcie-frac 0`.
  — [1a4ff59211](https://github.com/JEMostert/Strata/commit/1a4ff5921114e90e931173d735e08d0c562f8dd3) (2026-10-06)
- Pancake has `STRATA_PCIE_BALANCE` (upstream #1038 port: per-layer quota from measured costs; `expert_source.cpp`, `nv_defaults.hpp`). It has no `PCIE_ADAPT` and no `GDN_FORK` (grep). Pancake also has `STRATA_DF_BRANCH` (upstream 2e4ddf6e: GDN a/b and z on a side branch of the window graph, `verify.cpp`), which overlaps JEMostert's GDN side stream.
- KhaledElzarw `STRATA_FETCH_FORK=1`: pcie_mode 2's fetch_blobs + rebase_ptrs on their own stream, forked after the plan copy and joined before the PCIe groups. Grid default 48 blocks; top priority with `STRATA_GRAPH_PRIO=1`. Identical output. Part of the +1.0% phase-2 step. — [8817a1651d](https://github.com/KhaledElzarw/Strata/commit/8817a1651d417d5fa7527e839d94885828faf998). Pancake has GRAPH_PRIO, not FETCH_FORK.
- AdrianBM96 `STRATA_FETCH_ADMIT`: +5.1/+7.0/+5.1% on a 3060 (section 1). — [50581c24f2](https://github.com/AdrianBM96/Strata3060/commit/50581c24f2b1d35e8c76e53d1037fcc25672c0f9), [3ff40d67e3](https://github.com/AdrianBM96/Strata3060/commit/3ff40d67e3431ad23abe24268472afca59b8d843)
- mandoof1: `STRATA_PCIE_MMAP_AUTO_MAX=<n>` caps the balanced share, and `_AUTO_SLACK=<ms>` adds a margin to its feed side (opt-in). — [3775918cfd](https://github.com/mandoof1/Strata/commit/3775918cfd) (2026-10-20). Not opened; not in pancake.
- svgop: `--pcie-frac 0.55` adopted on its 5090 lane (16-leg pooled M1 +2.8%). — [324da1b2e0](https://github.com/svgop/Strata/commit/324da1b2e0) (2026-10-21). Hardware with much higher DDR5 bandwidth than ours.
- Pinned/mapped memory, prompt path: per-layer expert grouping tables through mapped pinned memory, off the copy engine (+3.9% at 32K, +4.5% at 128K, bit-identical) — [fe609ce2ad](https://github.com/VlDubr/strata-mi50/commit/fe609ce2ad) (2026-09-29, appears in 25 forks: an upstream commit). Most likely ALREADY-HAVE via base v0.2.12 (not grepped by name).
- svgop tmpfs shadow store: moving the Q2_0 shadow experts from disk-backed page cache to `/dev/shm` removed ~2.6M minor faults per leg: M1 +5.5%, pair +4.2%, conc +3.5%. — [9e889bda3a](https://github.com/svgop/Strata/commit/9e889bda3a838ccf932a25cbd88d6430742beb6e) (2026-10-27). It applies only to a page-cache-backed expert store. Pancake's arena is pinned/registered RAM, and its mmap tier is parked (MOD_PLAN 1139), so NOT-APPLICABLE today. The lesson: if the mmap tier is ever made default, back it with tmpfs or hugetlb instead of file pages.
- Lamina, Linux startup: "the expert arena filled with preadv and page-locked layer by layer" — [21970db772](https://github.com/eddoursul/Lamina/commit/21970db772) (2026-10-12). Not opened. This is the startup-memory pattern behind pancake's open oomd lead (arena filled through the page cache: 33 GiB arena + 33 GiB page cache > 57 GB, memory file).
- Asofwar (Windows only): `cudaHostRegister` keeps the arena in RAM but not in the working set, so under pressure Windows trims it and the pool soft-faults ("decode no longer falls to ~20 GB/s"). Fixed with `STRATA_ARENA_WSLOCK=1`. — [2476a46d4b](https://github.com/Asofwar/Strata-Turbo/commit/2476a46d4b). NOT-APPLICABLE on Linux; pancake mlocks.
- noorazman (V100): "63% of wait time is the round-head flag-A visibility lag"; the 3-wait protocol is "load-bearing" — [8f3eeb33e8](https://github.com/noorazman/Strata/commit/8f3eeb33e8). Upstream's verify poll lesson: replacing the window's final blocking sync with a `cudaStreamQuery` poll cost IQ3_S ~3% decode — [0d6efec071](https://github.com/anon761/Strata-upstream/commit/0d6efec071) (upstream, in base).

### Inferences
- The 5700G's DDR4 (~42.5 GB/s ceiling, ~28 GB/s effective per pancake's profile) is far lower than the 9950X's 60-64 GB/s. That points even more strongly than on KhaledElzarw's box toward a small or zero PCIe share on the 3060: every DMA byte competes with the CPU pool.
- If a zero share wins on the 3060, FETCH_ADMIT, FETCH_FORK and DOORBELL_STORE's PCIe part all become moot. Order the work accordingly:
  1. `--pcie-frac 0` / auto / `STRATA_PCIE_BALANCE=1`;
  2. then FETCH_ADMIT only if auto stays non-zero.

### Gaps
- No 3060 measurement of `--pcie-frac 0` vs auto exists in pancake docs (the 7900 XT measured 0 as better). AdrianBM96 kept auto (0.30 from probe) and did not report a 0 arm.

---

## 6. Blackwell / Ada / Volta-specific tricks: anything that still teaches something for Ampere or RDNA3?

### Takeaway
Most Blackwell items are feature-gated: NVFP4, PDL sm_90+, tcgen05 absent on SM12x, sm_120 IQ3_S generic-pointer fault. They do not transfer. What transfers is method:
- per-card dispatch tables measured with the parity benches;
- bit-identity proven at kernel level, because cross-boot greedy text is not a bitwise instrument when expert placement is timing-dependent;
- the warning that NVFP4 was slower than IQ2_XS on a 5090.

### Cited Findings
- svgop: NVFP4 measured 114-116 tok/s vs IQ2_XS 132-139 on the same 5090. "Nothing in the .2 feature set is a speed path for our regime". — [2077db0ce6](https://github.com/svgop/Strata/commit/2077db0ce6) (2026-10-20)
- svgop: "cross-boot greedy text proven NOT a bitwise instrument (stock .2.1 varies at identical env)". — [2a9d335d43](https://github.com/svgop/Strata/commit/2a9d335d430ea8da435b0bf4739ad77e75ed40f7). An A/A control of two identical boots diverged (similarities 0.36-0.84), so svgop proves bit-identity with isolated kernel tests. — [1bc5d0a338](https://github.com/svgop/Strata/commit/1bc5d0a338e0505117a914f89e2e631513763965). This matches pancake's own note that the engine is not run-to-run deterministic with the expert cache on.
- svgop BF16 router gate for DeepSeek-V4.1 (keep the BF16 source instead of promoting to F32, 157 MB/token less read, bit-identical, +2.4-3.2%) — [1bc5d0a338](https://github.com/svgop/Strata/commit/1bc5d0a338e0505117a914f89e2e631513763965). For Flash-Next the router projection is already BF16 (`bf16_gemv_fp32_mmvf`, per KhaledElzarw's 0af4328ec8 message). NOT-APPLICABLE beyond a reminder to check that no dense tensor is promoted to F32 at load on our pack.
- Upstream fix for an sm_120 IQ3_S generic-pointer fault, restoring bit-exactness — [745d2149a2](https://github.com/anon761/Strata-upstream/commit/745d2149a2) (2026-10-03). Lamina skipped its global-address hint for IQ3_S gate/up because it spilled on sm_120 and ran 4-19% slower (batch6 F1, already covered). Lesson: address-space hints are per-arch and per-format; check spills.
- Upstream: the single-matrix mmvq "does not stage the IQ grids (10-20% slower on the RTX 5070 at 3-8 columns)" — [3cc9734edb](https://github.com/anon761/Strata-upstream/commit/3cc9734edb). Staging IQ codebooks in shared memory is arch-dependent; Lamina puts the codebook in shared memory in `gus80`. Measure per card.
- Volko61 (Maxwell): `-Xptxas -dlcm=ca` made the DeepSeek GEMVs cache global loads in L1 (Maxwell sends them to L2): attention 479→50 ms/token. — [bd8e4c1d0a](https://github.com/Volko61/Strata-For-Maxwell/commit/bd8e4c1d0a774cd9e5c306ab3559cca46a861d04) (2026-11-04). sm_70+ caches in L1 by default, so NOT-APPLICABLE to the 3060.
- Asofwar `STRATA_GDN_CHUNKED=1` on 64-127 SMs: a 64-block wide scan in one wave (same bits), recurrence 1.13-1.22x at 8K-32K on a 5080, prompt -0.8-0.9%. — [b4ec40536b](https://github.com/Asofwar/Strata-Turbo/commit/b4ec40536b). Lesson: size grids to one wave on mid-size cards. The 3060 (28 SMs) is below that bracket and needs its own grid choice.
- svgop "TC-hc mma" (HC read on tensor cores, since consumer Blackwell lacks the 2x f16x2 rate): estimated +9-17% M1 — [bfa14e9d1a](https://github.com/svgop/Strata/commit/bfa14e9d1a1f59b95915e8a5b5e2ec1773851197). An estimate, not a measurement.

### Inferences
- The single most transferable Blackwell-era lesson for the 3060: the 3060 has half the SM count of a 3070 and a quarter of a 3090. Grids sized "one block per row group" for 80-SM cards land at fractional waves on 28 SMs. Wave-quantization checks are the cheap first audit: HC down 41 blocks, PCIe copy 48 blocks, GDN scan 64/128 blocks.

### Gaps
- None of the Blackwell forks measured on Ampere. Transfer of percentages is not possible.

---

## 7. Which of these port to HIP for the RX 7900 XT (gfx1100), and in what order should pancake test them?

### Takeaway
Clean HIP carriers:
- FETCH_ADMIT (only if the PCIe share is turned back on; fast-hip runs `--pcie-frac 0`);
- the Q2_0 4-lane down fold (wave32 on RDNA3, same lane arithmetic);
- Lamina's router changes with DPP or ds_swizzle reductions replacing `__reduce_max_sync`;
- the window decode attention via gfx11 WMMA iu8.

CUDA-only as written: DOWN86 (cp.async ring), `native_mma` (mma.sync), REDUX top-k. Each needs a re-design, not a port. HIP launch cost (14-16 µs empty launch on gfx1100) makes launch and grid trimming more valuable on the 7900 XT than on the 3060.

### Cited Findings
- gfx1100 launch cost and the 384→96-block copy kernel (+6% decode) — [d0c8d6c3e2](https://github.com/unimatrix099/Strata/commit/d0c8d6c3e2c1607447ff342b2aa1fd59c385b1b1). That copy kernel serves the PCIe share, which fast-hip does not use.
- Pancake HIP already has (per the presets and the memory file) GR_RDNA*, RDNA_MMVQ_WAVE, WAIT_FUSE, ROUTE_PUBLISH, ROUTER_FUSE, EXPDEC(_SWIGLU/_DOWN), DOORBELL_STORE and FUSE_HITS on HIP (upstream #649), HIP_WMMA prompt attention.
- Lamina's `__isGlobal` hint (gfx1100 flat_load issue) — *already covered* in batch6 F1.

### Candidate table (ordered for the RTX 3060 first; HIP column for the 7900 XT)

| # | Idea | Fork + commit | Date | Switch / flag | Author's claim (author's hardware) | Exact? | In pancake? | Est. gain for pancake (inference) | HIP port |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Re-test two current 3060 preset settings | KhaledElzarw NOTES | 2026-11-08 | drop `STRATA_EXPDEC=1`; drop `STRATA_HC_Q8=1` | EXPDEC -3.0%; HC_Q8 +0.9% with KL 0.025 (3090) | env only; HC_Q8 lossy | both set in 3060 presets | 0-3% decode, and an exact preset | n/a (env) |
| 2 | PCIe share sweep | KhaledElzarw README; JEMostert [1a4ff59211](https://github.com/JEMostert/Strata/commit/1a4ff5921114e90e931173d735e08d0c562f8dd3) | 11-08 / 10-06 | `--pcie-frac 0` vs auto vs `STRATA_PCIE_BALANCE=1` | 0.5-0.6 8-12% slower (3090/DDR5); balanced → 0.00-0.06, short +22% (4090) | exact | auto today | 0-10% on DDR4 (unknown sign) | already measured: 0 best on 7900 XT |
| 3 | `STRATA_FETCH_ADMIT` | AdrianBM96 [50581c24f2](https://github.com/AdrianBM96/Strata3060/commit/50581c24f2b1d35e8c76e53d1037fcc25672c0f9) | 2026-10-04 | `STRATA_FETCH_ADMIT=1` | decode +5.1/+7.0/+5.1% (RTX 3060, DDR4-2133) | exact per window; KL 0.002 | no | +2-7% if the share stays >0 | ports (verify_kernels hipified); moot at `--pcie-frac 0` |
| 4 | Per-format expert layout + Q2_0 4-lane down | KhaledElzarw [9af8f1fa14](https://github.com/KhaledElzarw/Strata/commit/9af8f1fa14222c3c4e78cec688b251b80b419f78), [4d462ccff3](https://github.com/KhaledElzarw/Strata/commit/4d462ccff3e46308ea684ada1a3888e1cc2cd716) | 11-07/08 | `STRATA_EXP_MODE=9`, `STRATA_EXP9_DOWN=42:10` | expert time -11..-17%; down 0.47-0.96x; e2e +1.6-2.0% (3090) | bitwise | mode 8 yes; 9 and EXP9 no | +1-3% (all 48 downs are Q2_0) | Q2_0 fold: yes (wave32); compare with EXPDEC_DOWN |
| 5 | HC read: DOWN82/86, UP_HYB, XN, UP_PL, D87 | KhaledElzarw [8817a1651d](https://github.com/KhaledElzarw/Strata/commit/8817a1651d417d5fa7527e839d94885828faf998), [857158c66d](https://github.com/KhaledElzarw/Strata/commit/857158c66dd59298dae78aaa8b2fecb7bcae63ab), [20f1fa1a95](https://github.com/KhaledElzarw/Strata/commit/20f1fa1a9504571676ba049a9a8f2414c87454c7) | 11-07/08 | `STRATA_GR_DOWN86=1`, `_UP_HYB=3`, `_XN=1`, `_UP_PL=1`, `_D87=8` | +2.3% (HC kernels) + part of +4.7% tier 5 (3090) | bitwise | no (GR_UP_WIDE, GR_UP75 yes) | +2-5%; 41→80 blocks = 1 wave on 28 SMs | DOWN86 cp.async: CUDA-only; partition idea yes; pancake HIP has GR_RDNA |
| 6 | 3060-specific MMVQ_IL tables | KhaledElzarw [ca1edf8786](https://github.com/KhaledElzarw/Strata/commit/ca1edf87869ff1113527d2ec18f93a3c4d7c6c3e) | 2026-11-07 | `STRATA_MMVQ_IL_3090=1` → write `_3060` | +1.8% (3090) | bitwise | no | +1-2% after re-bench | HIP has RDNA_MMVQ_WAVE (own tables) |
| 7 | Router: own copy blocks, one fence, REDUX top-10 | Lamina [f14591215f](https://github.com/eddoursul/Lamina/commit/f14591215f01afb8c0cf188f5800aba82d5fa9fc); KhaledElzarw [0af4328ec8](https://github.com/KhaledElzarw/Strata/commit/0af4328ec89113d00843eb39fdc8fe89bb1f9996) | 09-27 / 11-08 | (Lamina: default); `STRATA_ROUTE_BELL=2` | +0.9-2.7% (3090 x4 link); 1.0-1.8 µs/layer | bitwise | own `STRATA_ROUTER_FUSE` | +0.5-2% | yes, with DPP reductions instead of `__reduce_*_sync` |
| 8 | Sub-warp grouped expert kernels | Lamina [674bf0caf2](https://github.com/eddoursul/Lamina/commit/674bf0caf236b9276bd82a73b3e6fb948691a6d8) | 2026-10-13 | (Lamina default) | experts -3..-38% by T; e2e +5.7-8.3% (3090+5070 Ti, IQ3_S) | bitwise | no | +1-4% (lower: ~1.3 entries/group) | concept ports (wave32) |
| 9 | Window decode attention (+ small-window int8 MMA) | Asofwar [6ebe841e2e](https://github.com/Asofwar/Strata-Turbo/commit/6ebe841e2e), [540f801a9a](https://github.com/Asofwar/Strata-Turbo/commit/540f801a9a) | 10-20/21 | `STRATA_ATTN_WINDOW=1`, `_PREC=2`, `STRATA_ATTN_WINDOW_SMALL=1` | 200K decode 48.6→88.9; 130K 112.6→146.4 (RTX 5080) | rounding-level | no | small at <16K; +10-30% at 64-128K | yes via gfx11 WMMA iu8 / FP16; M-L effort |
| 10 | int8 tensor-core verify projections + head | Lamina [b4187bc3db](https://github.com/eddoursul/Lamina/commit/b4187bc3dbe8cedc74bd01c3cac1b82595d4ffb8) | 2026-10-16 | (table-selected) | e2e +2.2-6.3% (IQ3_S, 3090) | ≤1e-3, not bitwise | no | +1-3% (fast tier only) | redesign on WMMA iu8 |
| 11 | `STRATA_FETCH_FORK` + `GRAPH_PRIO` | KhaledElzarw [8817a1651d](https://github.com/KhaledElzarw/Strata/commit/8817a1651d417d5fa7527e839d94885828faf998) | 2026-11-07 | `STRATA_FETCH_FORK=1` | part of +1.0% step (3090) | bitwise | GRAPH_PRIO yes, FETCH_FORK no | <1% (only with share >0) | ports; moot at frac 0 |
| 12 | Q6_K dp4a IADD3+LOP3 | Lamina [df47127562](https://github.com/eddoursul/Lamina/commit/df471275629a6860ffc7c047eb30a7dbb847089d) | 2026-10-18 | none | Q6_K 1-token projections -1.2..-6.2% (3090) | bitwise | no | ~0-0.5% (few Q6_K tensors?) | SWAR trick ports |
| 13 | Stack-frame hygiene (`__grid_constant__`) | Lamina [95fa8c2321](https://github.com/eddoursul/Lamina/commit/95fa8c2321c7ec3a3fc3fa8b4989e55038fddd0a) | 2026-10-18 | none | removes mid-request stack-limit growth | bitwise | unchecked | first-prompt latency only | n/a |

Already in pancake (not re-proposed): noorazman Q2_0 grouped + decode-once IQ (`STRATA_OLD_GROUPED`, `STRATA_OLD_IQ_MMVQ`); Lamina `iq_dot.cuh`; svgop mmvq_il wide, GR_DOWN_TG, PROMPT_ATTN_IMMA + QSA_SPLIT_ATTN; xrip QSA_ROWS; Asofwar fast router_top10 on CUDA; upstream sm_86 default table; GR_NORM_FAST (opt-in); FUSE_HITS; DOORBELL_STORE; ATTN_DAILY; PF_FUSED (sm_86: AdrianBM96 measured +5.2% prefill, and the pancake 3060 preset does *not* set it; it is only in the HIP presets and the sm_121 table → add `STRATA_PF_FUSED=1` to the 3060 A/B).

Previously covered, not repeated: Lamina 33f63cf1f5 global hint; KhaledElzarw POOL_DAG/ACTQ_SKIP/ROUTE_BELL CPU side; svgop hugepage arenas; Spark-NV LOOKAHEAD; RandoTeam --no-graphs; svgop tail-skip and masked routing (lossy, rejected).

### Inferences
- **Recommended 3060 order** (when the card is back):
  1. Env-only arms: #1, #2 and `STRATA_PF_FUSED=1`.
  2. FETCH_ADMIT (#3) if the share stays non-zero.
  3. The KhaledElzarw kernel set (#4, #5, #6). It is on the same base commit, so cherry-picking `iq_kernels.cu`, `fused_gr.cu` and `native_mmvq.cu` hunks should be close to mechanical. Its parity benches (`fused_gr_bench`, `native_expert_bench`, `mmvq_il_parity`) come with it.
  4. Then router (#7), Lamina grouped (#8), and attention (#9) for long context.
- **7900 XT order:**
  1. The Q2_0 down fold (#4 down part) against pancake's `STRATA_EXPDEC_DOWN`, which was never measured on gfx1100 (MOD_PLAN:221).
  2. Router reductions (#7) with DPP.
  3. Window decode attention on WMMA (#9) for long-context decode.
  - FETCH_ADMIT/FETCH_FORK only if `--pcie-frac` is ever re-opened.
- Expected compounded gain on the 3060 from exact items #1-#8: perhaps +8-15% decode. That reasons from KhaledElzarw's +17.3% on a 3090 and discounts for the 3060's larger CPU share. Speculative until measured.

### Gaps
- None of the opened kernels was test-compiled for gfx1100. `__reduce_max_sync` has no direct HIP equivalent in ROCm 10 as far as I checked (not verified against ROCm 10 headers).
- I opened 40 of ~5,000 candidate commits. Forks with many CUDA subjects that I triaged only by subject: RandoTeam/4gbVramStrata (WDDM/4 GB), Jackwwg83/Strata-DS (DeepSeek fp8), wesraph (multi-user kv-share), lighttransport (GLM/DS/MiniMax engines), yqYo1 (V100/GLM). Their subjects pointed to other models or multi-user serving, but a deeper pass could find more.
