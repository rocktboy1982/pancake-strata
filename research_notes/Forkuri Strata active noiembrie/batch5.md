# Batch 5: active Strata forks, own commits since 2026-11-01 (state 2026-11-10)

Scope: 22 forks (DamirLukina88, svgop, Spark-NV, aktalascloud, yeli19950109, DmitriyW9, kuwacom, Oviing, storm-ace, riccardogiorato, phantomic12, daniel-winner, zhou51320, joja67, yuki-inaho, evrenesat, steddyman, Gold122, dexmac221, jialiang, pecuna). Our tree: pancake-strata, base upstream `146df3c4` (v0.2.12), preset `configs/pancake/flashnext-iq2_xs-fast-hip.json`. All speed numbers below are **author-measured** on the author's own hardware, not ours. Tags: NEW / ALREADY-HAVE / REJECTED-BEFORE / NOT-APPLICABLE.

## Which findings could raise decode tok/s, prompt speed or cut start time on RX 7900 XT + 5700G?

### Takeaway
Nothing here is a big exact kernel win for Flash-Next on gfx1100. Most of the svgop and Spark-NV engine work is already in our base (v0.2.12 smalls, BW-DENSE, PA_TRIM/PA_RPF, EXCHANGE_ROTATE, ADAPT_OVERLAP, PLE sort, GDN_CHUNKED, #5096/#4844). The three things worth acting on are below. (1) Turn off the routing-aware look-ahead when the RAM copy already holds every expert. This is a zero-code A/B today with `STRATA_LOOKAHEAD=0`. (2) Put the 30.5 GiB CPU-expert complement on a reserved 2 MiB hugetlb pool, and make the RAM budget see that pool. Effort M, and the pool reservation is the owner's call. (3) A small exact copy-kernel change (loads before stores). Separately, storm-ace's q36 HIP kernels are a strong NEW item for our Qwen3.6-35B-A3B work on HIP, but not for Flash-Next.

### Cited Findings

**F1. Turn off the routing-aware look-ahead when the RAM copy holds every expert (Spark-NV `0b88bd1a66`). NEW as a policy; exact; zero-code A/B available.**
- What it does: the look-ahead only warms file-tier pages. When the pinned RAM complement holds every expert the GPU cache does not, the patch computes `complement_whole()` in `pin_cache_complement` and detaches `drive.d.lookahead`. `STRATA_LOOKAHEAD=1` keeps it on. The same commit also adds Linux `MADV_RANDOM` plus a `MADV_WILLNEED` row prefetch for the *mmap'd* PLE table. — [commit 0b88bd1a66](https://github.com/Spark-NV/Strata/commit/0b88bd1a66)
- Author-measured on RTX 5090 + Ryzen 7 7800X3D, Flash-Next IQ4_XS, 260K ctx: decode window 17.4 -> 15.8 ms; API bench decode 162.2 -> 182.9 tok/s; prompt unchanged. — [commit 0b88bd1a66](https://github.com/Spark-NV/Strata/commit/0b88bd1a66)
- Our state: `generate.cpp:8943-9065` detaches the look-ahead only when `gpu_holds_all` (every expert in VRAM; P1.8/PR 4972). It does not detach when the RAM complement holds everything. Our 2026-11-10 log (`pancake-flashnext.log`) shows both "routing-aware prefetch on (the next layer's router, file tier)" and "FileExpertSource: mapped pinned cache complement ready: resident 30.50 GiB". So the look-ahead runs and warms GGUF pages that the decode never reads. The CPU-predictor cost is gone already, because `STRATA_LOOKAHEAD_GPU` is on by default (MOD_PLAN 10.2: predictor 30-34 -> 0.6-0.7 ms/window). What is left: a thread reading picks, plus `warm()` read-ahead into page cache on ntfs3, which takes RAM bandwidth and page cache from the CPU pool.
- Suggested A/B: `STRATA_LOOKAHEAD=0` vs unset, on fast-hip, for short chat / 2K code edit / 11.7K prompt; also record `STRATA_IO_STATS=1` warmed counts. If it wins, port the `complement_whole()` gate (effort S, ~25 lines). The PLE `MADV_RANDOM` half is NOT-APPLICABLE for now: we run `--ple-io direct` (PleReader), not mmap, and `ngram.cpp:249-269` already has the Linux WILLNEED hint.

**F2. CPU-expert complement on a reserved 2 MiB hugetlb pool, plus a pool-aware RAM budget (svgop `07530eb7a5`, `511bece612`, `e8484c3663`, `003d78d8da`, `df1a9c9000`). NEW for the complement path (the earlier report covered only the `pinned.cu` arena); exact.**
- What it does: with `STRATA_HUGEPAGE_ARENAS=1`, and when `huge_pool_covers()` (HugePages_Free read at call time) says the pool fits, the complement takes 2 MiB hugetlb pages before registration instead of 4K pages. — [511bece612](https://github.com/svgop/Strata/commit/511bece612), [07530eb7a5](https://github.com/svgop/Strata/commit/07530eb7a5)
- The fix that has to come with it: MemAvailable does not count the reserved pool, so a box with a pool "read as RAM-starved and the complement got CAPPED at 19.80 of 37.97 GiB". The budget now adds `huge_pool_free_bytes()`. A later fix subtracts HugePages_Rsvd. — [e8484c3663](https://github.com/svgop/Strata/commit/e8484c3663), [003d78d8da](https://github.com/svgop/Strata/commit/003d78d8da)
- Boot: one `MADV_POPULATE_WRITE` pass over the pool pages, "-6-12s boot". — [003d78d8da](https://github.com/svgop/Strata/commit/003d78d8da)
- Author-measured, CUDA box (NVML clock work suggests a ~32 GB RTX 50-series card; exact GPU and CPU not confirmed from these commits): GSQ lane solo +7.2% (leg-3), IQ2_XS solo +4.9%, pair/conc flat. — [df1a9c9000](https://github.com/svgop/Strata/commit/df1a9c9000), [8c91289449](https://github.com/svgop/Strata/commit/8c91289449)
- Conflicting evidence from the same author: on 2026-11-02 svgop wrote that the "move-the-bytes-faster family (hugepages/TLB/prefetch-shape/more threads) is closed" (49.5 GB/s, CPU lane at 93-100% of it). That was before the pool-backed route was found to help. — [30a7ef7baa](https://github.com/svgop/Strata/commit/30a7ef7baa)
- Our state: the complement (`expert_source.cpp:3903-3920`) is `posix_memalign` + 4K pages, taken in 1 GiB paced steps that check `available_memory_bytes()` (MemAvailable) against a reserve (`pin_step_fits`). That is exactly the pool-blind budget svgop hit: reserving `vm.nr_hugepages` without the budget fix would *cap our complement*. Box today: `vm.nr_hugepages=0`, THP madvise (tools_benchmarks.md note). MOD_PLAN 1314 parks MADV_POPULATE_WRITE ("load-time only").
- Applicability: 5700G DDR4 with ~30.5 GiB random expert-blob reads. A TLB win is plausible but smaller than on a 7800X3D-class box, and nobody has measured it on Zen 3 desktop. HIP: `hipHostRegister` on hugetlb memory is standard. 57 GB RAM: a ~15,600-page (30.5 GiB) pool is permanent and leaves ~26 GB for the OS, page cache and the PLE reader.
- Port effort M (alloc branch + budget + accounting); owner decision for the boot-time pool. A/B: `vm.nr_hugepages=15700` reserved at boot + `STRATA_HUGEPAGE_ARENAS=1` vs off. Check the startup line for "N GiB in 2 MiB pages", then run short / 2K edit / 11.7K, plus time to ready.

**F3. Faster `copy_rows_from_mapped` (Spark-NV `f95f0bf6cb`). NEW; exact (same bytes); effort S.**
- What it does: each thread issues up to 8 float4 loads over PCIe before its stores, so a row costs one round trip per pass instead of one per float4. Author-measured on RTX 5090: 4.7-5.7 -> 2.4-3.9 us per launch with 1-3 CPU rows. The same commit adds per-window CPU-pool gate/up / quant / down timing and GB/s to `STRATA_DECODE_TIMING=1`. — [f95f0bf6cb](https://github.com/Spark-NV/Strata/commit/f95f0bf6cb)
- Our state: `src/kernels/cuda/elementwise.cu:238-262` has the old loop (`volatile`, one load-store per iteration).
- Expected effect on us: a few us per call times ~48 layers a window, about 0.1-0.3% of a ~35-40 ms window. A cheap exact port; the timing line is useful by itself.
- A/B: build A/B, short chat + 2K edit; token-forcing check.

**F4. Tuning from Spark-NV `f95f0bf6cb` / `71102eadd7`. Config-only A/B.**
- `--spec-min-p 0.7` (was 0.5) together with `--adapt-every 8`. Author-measured, 7800X3D + RTX 5090 IQ4_XS, story/code/16-topics: 174/177/110 -> 179/194/107 tok/s. — [f95f0bf6cb](https://github.com/Spark-NV/Strata/commit/f95f0bf6cb)
- `STRATA_EXCHANGE_ROTATE=1` + adapt every 2 windows: 174/177/110 -> 184/196/124. — [71102eadd7](https://github.com/Spark-NV/Strata/commit/71102eadd7). ALREADY-HAVE: our preset has `STRATA_EXCHANGE_ROTATE=1` and `--adapt-every 1`.
- Our `bench/specs/05-small-flags.json` already tries `--spec-min-p 0.6`; 0.7 extends that sweep. Lossless (it only changes the draft cut).

**F5. Adaptive speculation controller W1 (svgop `ee435ae5ab`). NEW (our `src/spec/controller.cpp` exists but `generate.cpp` does not wire it); exact (sizing only).**
- What it does: sizes the serial verify window by argmax_T E[tokens]/step_ms(T), using the session's MTP depth-acceptance EMAs and a fitted window-cost line. Author-measured: battery +0.9/-0.4/+3.0, "all inside noise"; the claimed +35% applies only to deep-context live sessions and is not shown. — [ee435ae5ab](https://github.com/svgop/Strata/commit/ee435ae5ab)
- Low priority. Effort M. A/B on the 11.7K prompt + long-session follow-ups only.

**F6. storm-ace q36 HIP kernels (Qwen3.6-35B-A3B engine `src/qwen35/q36_engine.cu`). NEW for our q36 HIP work (feat/bfla-hip); NOT-APPLICABLE to Flash-Next.**
- `4fe0284da5`: `k_attn_part_gqa<G>` reads each K/V cell once per GQA group, with LDS-staged coalesced K. Bitwise the same partials; `STRATA_Q36_VERIFY=1` found 0 mismatches. Author-measured on Radeon 8060S (gfx1151, Windows): ~8.9K ctx 25-31 -> 37.0 tok/s; ~35K ctx 11.4-12.5 -> 27.1. — [4fe0284da5](https://github.com/storm-ace/Strata/commit/4fe0284da5)
- `f74984a171`: MTP drafts for *sampled* requests, using the speculative-sampling rule for a one-token draft, so the distribution is exact. Grouped prompt-attention GEMM, bit for bit. Author-measured: 9K 37.0 -> 48.8 tok/s, 36K 27.1 -> 33.2, pp32768 360 -> 389. — [f74984a171](https://github.com/storm-ace/Strata/commit/f74984a171)
- `e4fc07ee57`/`161309c707`: fused flash prompt attention on WMMA (`v_wmma_f32_16x16x16_f16`, wave32; RDNA3 too, but default on only on gfx1151, so other gfx11 chips need `STRATA_Q36_FA=1`), plus hipBLASLt table rows for hidden-2048 shapes. Lossy at the FP16 level: KL 1.2-2.2e-6, top-1 32/32. pp32768 360 -> 517 -> 938 tok/s. — [e4fc07ee57](https://github.com/storm-ace/Strata/commit/e4fc07ee57), [161309c707](https://github.com/storm-ace/Strata/commit/161309c707)
- `48a56ce1ba`: `STRATA_Q36_KV16=1`, FP16 K/V cache; KL 1.9e-6; decode at 64K 20.8-21.6 -> 23.5-25.1. — [48a56ce1ba](https://github.com/storm-ace/Strata/commit/48a56ce1ba)
- Our state: none of `k_attn_part_gqa`, `k_attn_fa_wmma`, sampled-MTP, or `Q36_KV16` is in our `q36_engine.cu`. Our qwen36 daily presets are CUDA `strata-poly`; BFLA on HIP (README 50-60) already gives 2-8.5x on long prompts, so FA-WMMA overlaps the BFLA "dense" mode and needs a head-to-head. The GQA decode kernel and sampled MTP are exact and independent of BFLA. Effort S-M each (one file). A/B on gfx1100: `STRATA_Q36_FA=1`, `STRATA_Q36_OLD_KERNELS=1` vs default, `STRATA_Q36_NO_SPEC_SAMPLE=1`, at 9K/36K ctx.

**Already in our tree (ALREADY-HAVE), checked by grep:**
- svgop "smalls" `3af4b32273`/`4152f114d2` (+13.1/+22.9/+12.6 on their GSQ lane): `STRATA_SMALL_UNR=3` is in `arch_defaults.cpp:58,102`; the grid-stride `copy_i32_from_mapped` is in `elementwise.cu:393-420`. — [3af4b32273](https://github.com/svgop/Strata/commit/3af4b32273)
- svgop BW-DENSE port `7010d3522e` (`S2_SWIGLU_Q8`, `pcie_share_rule`, `GR_V3_PF`): present from upstream. — [7010d3522e](https://github.com/svgop/Strata/commit/7010d3522e)
- svgop prefill trio `047ab209a6` (`PA_TRIM`/`PA_RPF`, +4.7% prefill at 24K): both in `qsa_prompt_attn.cu`. — [047ab209a6](https://github.com/svgop/Strata/commit/047ab209a6)
- Spark-NV `a85c6140fc` upstream opt-ins: `SELECT_TWO_PASS` and `IQ4_W4A8` are present; the fused pool is present as `STRATA_POOL_FUSE_PHASES` (in preset). `--kv k16v8` was not found by grep. — [a85c6140fc](https://github.com/Spark-NV/Strata/commit/a85c6140fc)
- Spark-NV `f9597776dd` `STRATA_ADAPT_OVERLAP(_RESIDENT)` (+9.9% author-measured on 5090): in our tree, and our log shows the adaptive overlap active with the preset's `STRATA_ADAPT_OVERLAP=1`. — [f9597776dd](https://github.com/Spark-NV/Strata/commit/f9597776dd)
- Spark-NV `8a3475b6d2` #5096 empty-DMA-group ordering and #4844 batch range: ours has `raise_flag_b_ordered` (`verify.cpp:4148`) and `a_rng.whole_round` (`generate.cpp:13889`). — [8a3475b6d2](https://github.com/Spark-NV/Strata/commit/8a3475b6d2)
- DmitriyW9 `36fcae2f55` PleReader: no per-call hash map, misses sorted by offset. Ours already has the sorted-miss path (`ple_reader.cpp:478-505`). — [36fcae2f55](https://github.com/DmitriyW9/Strata/commit/36fcae2f55)
- DamirLukina88 upstream 0.2.10-0.2.12 ports (#4677, #4350, Q5_0 dot fix, #5108, #5096) and `STRATA_GDN_CHUNKED` (in our preset): contained in our v0.2.12 base. — [63a139f1b8](https://github.com/DamirLukina88/D.L.-AI---Strata/commit/63a139f1b8), [fd93580c6d](https://github.com/DamirLukina88/D.L.-AI---Strata/commit/fd93580c6d)
- kuwacom `02e8c66374`…`54d4171f71` (HC down/inject BF16, QSA value `__launch_bounds__` 8 CTAs/SM): already covered in `research_notes/Forkuri Strata câștig viteză/amd_rdna_ports.md` and the earlier report (row 3.9). — [docs](https://github.com/kuwacom/Strata-legacy-gpu)

**REJECTED-BEFORE / lossy:**
- svgop `30a7ef7baa` tail-skip rank 7 (`STRATA_ROUTE_TAIL_SKIP=7`). Author-measured +10.2% solo; "golden hash IDENTICAL" on their probe, but the mechanism skips routed experts, so it is lossy in general. Our preset pins `STRATA_ROUTE_TAIL_SKIP=0` under the exactness rule (MOD_PLAN item 24). — [30a7ef7baa](https://github.com/svgop/Strata/commit/30a7ef7baa)
- svgop `9e309adb39`…`93b85b38e7` masked-router expert pruning (+8-11%): the author deleted it after MMLU fell from 92.9% to 78.6%. — [c8286d5a27](https://github.com/svgop/Strata/commit/c8286d5a27)
- svgop `64333c512f` IL-Q8 (dense Q8_0 decode fast path, CUDA `native_mmvq.cu`): CUDA-only, and "GSQ-inert by construction" on their side; our preset runs `STRATA_HC_Q8=0`. NOT-APPLICABLE. — [64333c512f](https://github.com/svgop/Strata/commit/64333c512f)

### Inferences
- F1 is the cheapest first test: one env var, exact, and our log shows the waste happens. On a 57 GB box the page cache the look-ahead fills competes with the complement and with PLE direct reads, so the gain could be larger than the CPU cost alone suggests.
- F2 matters most for what it *warns* about. Anyone who sets `vm.nr_hugepages` for the earlier report's "hugepages per boot" idea will hit the pool-blind budget and cap the complement, unless the budget fix lands first.
- The svgop/Spark-NV prefill numbers (5-7K tok/s at 24-32K) are RTX 5090 numbers. None of their prefill items is new to us.

### Gaps
- No fork measured anything on gfx1100 + Zen 3 DDR4 for F1/F2/F3; all effect sizes on our box are unknown.
- svgop's exact GPU/CPU model is not stated in the commits read (the NVML clock work and "32GB as-built" suggest an RTX 5090-class card).
- I did not verify whether `warm()` actually issues file reads on our complement-whole config (needs `STRATA_IO_STATS=1` on a run).

## Correctness bugs that also exist in our tree (noted in passing)

### Takeaway
Four latent bugs from these forks are present in our tree. None is on the default single-GPU decode path, but two are cheap fixes.

### Cited Findings
- **Idle stats-flush SIGSEGV (svgop `c45eeb2a17`).** `bt_stage_wait0/bt_stage_pool0` are declared empty (`generate.cpp:14572`) and armed only in the single-verifier first-window block (`14944-14948`). The pipelined/cohort first-window block (`15433-15438`) does not arm them, while the flush (`15128-15136`) indexes `[k]`. svgop got three SIGSEGVs on idle after cohort windows. Fix: size at declaration and arm in the cohort block (~10 lines). Reachable only on the pipelined/cohort path (multi-request). — [c45eeb2a17](https://github.com/svgop/Strata/commit/c45eeb2a17)
- **Layer-major prefill row count ignores `STRATA_PF_GROUP_SCRATCH` (Spark-NV `8a3475b6d2`).** `lm_prepare`/`lm_run` use `G = m.T * K` (`prefill.cpp:6798, 7148`), but with group scratch GU/H/Hq hold only `group_cap` rows. Our fast-hip preset sets `STRATA_PF_GROUP_SCRATCH=1`. Layer-major is off on our HIP runs (not in the log), so this is latent until someone sets `STRATA_PREFILL_LAYER_MAJOR=1`. Two-line fix. — [8a3475b6d2](https://github.com/Spark-NV/Strata/commit/8a3475b6d2)
- **Embedding gather beyond grid.y 65,535 (Spark-NV `a6b1e76e7b`).** `iq_embed_rows` launches `n_tok` blocks in grid.y (`iq_kernels.cu:5539`). Safe with our 32K prefill chunks; it breaks if a caller passes over 65,535 tokens. — [a6b1e76e7b](https://github.com/Spark-NV/Strata/commit/a6b1e76e7b)
- **Conversation-cache publish not atomic on allocation failure (aktalascloud `d2fb1b9e4c`, from upstream PR #5530).** Ours still has `serials_.push_back(next_serial_++)` after `entries_.push_back` (`conversation_cache.hpp:547`). — [d2fb1b9e4c](https://github.com/aktalascloud/strata-legacy-gpu/commit/d2fb1b9e4c)
- **JSON response watcher race (aktalascloud `3da17a80fe`).** A client closing right after the last byte is logged as "disconnected". Ours has the unlocked `if gone and not done.is_set():` (`serve/server.py:8642`). — [3da17a80fe](https://github.com/aktalascloud/strata-legacy-gpu/commit/3da17a80fe)
- Checked and **not** in ours: svgop `919b4db31d` slot-release NameError (our `server.py:2141` binds `releasing` correctly). — [919b4db31d](https://github.com/svgop/Strata/commit/919b4db31d)

### Inferences
- The first two are worth taking with any F1/F3 port; the rest are robustness fixes for the server.

### Gaps
- I did not check whether upstream after v0.2.12 already fixed the cohort-flush or layer-major issues.

## One-line verdicts for every fork in the batch

### Takeaway
Of the 22 forks, only Spark-NV, svgop and storm-ace carry anything new for us. The others are app/UI work, other hardware, other models, or upstream syncs we already have.

### Cited Findings
- DamirLukina88/D.L.-AI---Strata (111): Android companion/voice/Home Assistant app + upstream 0.2.10-0.2.12 syncs. Nothing to take (syncs ALREADY-HAVE). — [repo](https://github.com/DamirLukina88/D.L.-AI---Strata)
- svgop/Strata (92): F2 hugepage complement + pool-aware budget (NEW, M), F5 W1 controller (NEW, neutral), cohort-flush crash fix (bug present); smalls/BW-DENSE/PA_TRIM ALREADY-HAVE; tail-skip/pruning lossy. — [repo](https://github.com/svgop/Strata)
- Spark-NV/Strata (47): F1 look-ahead off when the complement is whole (NEW, S, A/B now with `STRATA_LOOKAHEAD=0`), F3 copy-rows kernel (NEW, S), spec-min-p 0.7 A/B, group-cap and embed-grid fixes; AVX-512/NEON/Zen5 kernels NOT-APPLICABLE (5700G has no AVX-512); GUI rest. — [repo](https://github.com/Spark-NV/Strata)
- aktalascloud/strata-legacy-gpu (40): AVX1/P40/DeepSeek (dsv41) work, NOT-APPLICABLE; two server/cache robustness fixes present in ours. — [repo](https://github.com/aktalascloud/strata-legacy-gpu)
- yeli19950109/Strata (22): desktop GUI/CI on Windows. Nothing to take. — [repo](https://github.com/yeli19950109/Strata)
- DmitriyW9/Strata (15): V100+4070S two-GPU split, DP prefill for two GPUs (NOT-APPLICABLE, single GPU), PleReader alloc fix (ALREADY-HAVE). — [repo](https://github.com/DmitriyW9/Strata)
- kuwacom/Strata-legacy-gpu (14): P40 prefill HC BF16 + QSA occupancy, already covered in the earlier report. Nothing new. — [repo](https://github.com/kuwacom/Strata-legacy-gpu)
- Oviing/Strata (9): DiffusionGemma sidecar (MLX/numpy). Nothing to take. — [repo](https://github.com/Oviing/Strata)
- storm-ace/Strata (8): F6 q36 HIP GQA decode / sampled MTP / WMMA FA / KV16. NEW for our Qwen3.6 HIP branch, not for Flash-Next. — [repo](https://github.com/storm-ace/Strata)
- riccardogiorato/Strata (6): RTX 2080 compact prefill, which the author rolled back (`83ecef965d`). Nothing to take. — [repo](https://github.com/riccardogiorato/Strata)
- phantomic12/strata-qwen4exp-generic (5): engram table measurements (DeepSeek V4.1). NOT-APPLICABLE. — [repo](https://github.com/phantomic12/strata-qwen4exp-generic)
- daniel-winner/Strata (4): Monitor "Today" usage card. Nothing to take. — [repo](https://github.com/daniel-winner/Strata)
- zhou51320/Strata (4): dual 2080 Ti configs. Nothing to take. — [repo](https://github.com/zhou51320/Strata)
- joja67/Tuga (3): README only (gfx1200). Nothing to take. — [repo](https://github.com/joja67/Tuga)
- yuki-inaho/Strata (3): 8 GB RTX 2070 diary/Justfile. Nothing to take. — [repo](https://github.com/yuki-inaho/Strata)
- evrenesat/Strata (2): ERR tagging / snapshot refactor. Nothing to take. — [repo](https://github.com/evrenesat/Strata)
- steddyman/Strata (2): gfx906 (Vega 20, wave64) AppImage + two-card defaults. NOT-APPLICABLE. — [repo](https://github.com/steddyman/Strata)
- Gold122/Strata (1): conversation-cache disk tier in `conversation_cache.hpp` (+58/-4); not read in depth, and our tree has its own `conversation_disk_tier.cpp`. Probably nothing to take. — [commit 5dfe6e9662](https://github.com/Gold122/Strata/commit/5dfe6e9662)
- dexmac221/strata-cluster (1): README pointing to Strata Link. Nothing to take. — [repo](https://github.com/dexmac221/strata-cluster)
- jialiang/Strata (1): opt-in projection (control) vector setup. Not a speed item. — [repo](https://github.com/jialiang/Strata)
- pecuna/Strata-3060 (1): backport of engine 0.1.58. Older than our base. Nothing to take. — [repo](https://github.com/pecuna/Strata-3060)

### Inferences
- Priority for pancake: F1 (`STRATA_LOOKAHEAD=0` A/B) -> F3 + the two cheap correctness fixes -> F2 (after an owner decision on the hugetlb pool) -> F6 if the Qwen3.6 HIP line is active.

### Gaps
- Gold122's disk cache was triaged from its file list only.
