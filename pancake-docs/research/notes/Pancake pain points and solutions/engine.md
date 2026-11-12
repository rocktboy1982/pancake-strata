# Pancake engine: performance pain points and the researched technologies that could close them

Scope: engine side only (src/core, src/kernels, src/prefill, src/spec, src/program, pool.cpp, presets, build), RX 7900 XT
(gfx1100, ROCm 10) + Ryzen 7 5700G (AVX2, DDR4) + 57 GB RAM + NVMe on ntfs3, Qwen3.8-Flash-Next ISTA-DASLab GSQ-RCO
(Q2_0 / IQ2_XS), preset fast-hip, single-user opencode. Read-only analysis of existing data (no build, no run, no GPU);
2026-11-09. Ownership: the other session "dan-cd" owns MOD_PLAN items 2, 8/28, 13, 14/27/30, 29 (bench queue until
~13:30); "dan-b5" owns 35/31/16 and 12/21/33/34 per PANCAKE_STATUS.md.

Abbreviations used below: "window" = one speculative verify window (MTP draft + verify); "short" = short chat request;
"edit" = code-edit request (2K prompt); "report X" = /home/dan/reports/*.md of the research series.

Repo root for every path below: `/home/dan/clone-of-d/src/Strata`.

## Q1. Where does decode time go on the 7900 XT, and how much of it is fixable?

### Takeaway
A decode window is 24.8 ms on a short chat (GPU 15.9, CPU experts 5.2, draft 2.2) and 37.4 ms on a code edit (GPU 15.8,
CPU experts 17.4, draft 2.5). The CPU experts run at ~27 GB/s, i.e. ~80% of the measured DDR4 rate, so kernel tuning on
the CPU side has little room: only fewer CPU bytes (fewer misses, sparsity, lossy routing) or more overlap (deferral) can
move the edit window; the GPU floor (15.9 ms) dominates short chats.

### Cited Findings
- fast-hip edit window: 109 windows, 4.70 tokens/window, 37.39 ms = verify 34.73 (GPU-reach wait 15.75 + per-layer host
  17.94, of which CPU 17.41) + draft 2.50; per layer-window 7.44 CPU experts (8.88 entries), 40.47 VRAM hits, PCIe 0.00,
  28.5 swaps/window — [profile-fasthip.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/profile-fasthip.txt)
- fast-hip short window: 224 windows, 2.29 tokens/window, 24.81 ms = verify 22.44 (GPU-reach 15.93 + host 5.59, CPU 5.19)
  + draft 2.24; 2.22 CPU experts and 27.13 VRAM hits per layer-window — [profile-fasthip.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/profile-fasthip.txt)
- Same split stated as the phase-13 table (short 24.8 / GPU 15.9 / CPU 5.2 / draft 2.2; edit 37.4 / 15.8 / 17.4 / 2.5;
  "CPU experts are 47% of the window") — [MOD_PLAN.md phase 13](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)
- GPU stages, short window (ms/window, GDN layers): q8+qkv gemv 1.15, conv 0.33, ab 0.37, z 0.79, rec 0.67, out-proj
  0.87, router+plan+bell 1.05, shared+quant 0.88, VRAM hits 3.25, copy+combine 1.84, head 0.64, hyper-connection reads
  hc0 norm/down/up 0.30/0.53/0.64 + hc1 0.10/0.53/0.63 (=2.73 ms); QSA layers attention 0.74, scores+topk 0.34 —
  [profile-fasthip.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/profile-fasthip.txt)
- rocprofv3, one short request (79 tok/s under the profiler vs ~126 without): combine_cpu_k10_vec4 38.3% of GPU time
  (spin-wait for CPU rows), wait_flag_copy_i32 4.1%, rdna_kq ~10%, GPU experts (expdec_gu + native_down) ~13%,
  rdna_gr_fused 5.6%, bf16_f32_mmvf_multi 5.1%, route_publish_res 2.7%; 395 kernels per token, gaps <50 us average 5.4 us;
  gaps >1 ms total 525 ms at window/request edges (kv_append -> copyBuffer, copyBuffer -> copy_from_mapped) —
  [gpuprof README](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-09-7900xt-gpuprof/README.md)
- Per-kernel stats from the same run: native_down_multi_kernel<42> 5.79% (4224 calls, 37.6 us), expdec_gu_kernel<22>
  5.83%, expdec_gu_kernel<16> 1.60%, rdna_kq_kernel<23,4,3> 8.08% —
  [run_kernel_stats.csv](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-09-7900xt-gpuprof/run_kernel_stats.csv)
- CPU per-expert cost: pool time per missed expert 53-55 us; PCIe probe 13.4 GB/s; RAM read 33 GB/s; one expert
  ~1.3-1.4 MB at ~2 bits, promotion ~115 us vs ~42 us CPU compute — [MOD_PLAN.md phase 10 and item 14](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)
- Decode cache hit rate on a short request after start 92.0%; on 512-token requests after the 11.7K prompt 90.2-92.5%
  — [pancake-flashnext.log](file:///home/dan/clone-of-d/src/Strata/pancake-flashnext.log) (lines ~13960-15243)
- Replay ceilings at 10,909 cached experts (44%): engine tier 79.6%, per-layer LRU 89.5%, Belady 94.7%; tier every
  window x160 85.0% — [MOD_PLAN.md phase 11](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)
- Time-based ceiling (tools/pancake/time_ceiling.py, routing.bin, 10,909 slots): LRU loses (-20% short), oracle admission
  +8..15% short / +27..48% code edit, online 2nd-miss-in-8 +5..16% on code edits, tier x160 +4..9%; DMA contention
  (alpha) decides — [commit c87ef4d27 message / PANCAKE_STATUS.md](file:///home/dan/clone-of-d/src/Strata/PANCAKE_STATUS.md)

### Inferences
- CPU expert bandwidth: edit 7.44 x 48 = 357 experts in 17.41 ms = 48.8 us each; short 2.22 x 48 = 107 in 5.19 ms =
  48.7 us. At ~1.3 MB each that is ~26.7 GB/s, ~80% of the 33 GB/s measured RAM rate. The pool is near the DDR4 roofline:
  item 29 (barriers), item 24 (IQ1_M kernel) and CPU kernel sweeps can only touch the remaining ~20% plus sync overhead.
- Real GPU work is ~57.6% of kernel time (100 - 38.3 combine wait - 4.1 wait_flag). Of that, GPU experts are ~23% and
  native_down<42> alone ~10%.
- native_down_multi_kernel<42> is the down projection of the VRAM-hit experts on the default kernel: the engine log says
  `strata expdec: CUDA0 down type 42 (n_embd 2560, n_ff 640) not covered: the default kernel runs`
  ([pancake-flashnext.log](file:///home/dan/clone-of-d/src/Strata/pancake-flashnext.log), from
  [iq_kernels.cu:4155](file:///home/dan/clone-of-d/src/Strata/src/kernels/cuda/iq_kernels.cu)). The "IQ2_XS" file's
  down tensors are Q2_0 (report Europe F3), so the expdec fast path covers gate/up (types 22, 16) but not down. This is a
  new, unlisted pain point (not in MOD_PLAN items 1-37).
- Fixability ceiling of the GPU floor on short chats: if every GPU piece were 10% faster the short window would drop
  ~1.6 ms (~6%); on edits ~0 until the CPU block shrinks (the GPU already waits ~42% of its time).

### Gaps
- The profiled run was a short request only; no rocprofv3 profile of an edit window or of a post-long-prompt window.
- How much of "GPU-reach wait 15.9 ms" is true GPU work vs launch gaps is inflated by the profiler (79 vs 126 tok/s);
  the 5.4 us gaps x 395 kernels (~2.1 ms/token) is an upper bound, not a measured normal-run figure.

## Q2. Decode pain points and the researched technologies that map to them

### Takeaway
Five decode pain points, in order of time share: (D1) CPU expert critical path, (D2) cache misses / routing, (D3) decode
drop after long prompts, (D4) the GPU floor, (D5) the draft step. D1-D3 are mostly owned by dan-cd; D2's lossy lever
(turbo-hip) is already built and measured (+30-50%) and waits only for a quality decision.

### Cited Findings
D1 - CPU expert critical path (47% of edit window, 21% of short)
- Expert deferral (KTransformers, Tsinghua, SOSP'25): defer low-ranked CPU experts behind the next layer's attention;
  DeepSeek-V3 -26% per-layer time, +33% decode, "up to 45%", tasks within 2 points; not on the last layer; pancake has
  STRATA_VERIFY_DEFER_CPU + _BELOW (weight 0.1) but no rank guard; estimate +10-25% decode after the HIP fix —
  [report Asia, "KTransformers..." and P1](file:///home/dan/reports/Asia%20papers%20for%20Avalon.md); MOD_PLAN items 8, 28
- On the 3060 the same switch measured short +8%, edit -7%, changing the output —
  [PANCAKE_FINDINGS.md section 10](file:///home/dan/clone-of-d/src/Strata/docs/PANCAKE_FINDINGS.md)
- Activation sparsity on CPU experts: Berkeley intra-expert gate mask (arXiv 2605.08575, Qwen3.5 MoE 84-87% sparsity,
  95% of score; -3.4 points at 85%), TEAL (MIT, 40-50% on dense), CATS (Stanford, collapses at 40%), CHESS per-channel
  thresholds; ceiling ~1.25x decode, realistic 10-20%, lossy —
  [report US P1/P2](file:///home/dan/reports/US%20university%20papers%20for%20Avalon.md),
  [report MIT #1-2](file:///home/dan/reports/MIT%20Harvard%20papers%20for%20Avalon.md), MOD_PLAN item 13
- Per-expert barriers in the CPU pool (Tsinghua OSDI'26 "Cloud-Grade SLOs"): run_phase syncs globally between gate/up and
  down; 1-4% decode, exact — [report Asia P4](file:///home/dan/reports/Asia%20papers%20for%20Avalon.md);
  coded as STRATA_POOL_FUSE_PHASES (commit 35c512984, [pool.cpp:1267/1452](file:///home/dan/clone-of-d/src/Strata/src/kernels/cpu/pool.cpp)), A/B in queue3
- CPU pool settings: workers 4/5/6 -> 112.3/117.7/124.1 vs base 126.3-126.4; tasks 8/16/48 -> 121.9/124.9/126.8 (defaults
  best) — [pool-sweep.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/pool-sweep.txt)
- Pool pre-read (tandem.cpp LLAMA_MOE_PREREAD): loses (95 vs 107-122 tok/s), stays off; L3 warm (STRATA_ARENA_WARM)
  worse: 119.8/106.5 vs 126.2/125.3 — [PANCAKE_STATUS.md](file:///home/dan/clone-of-d/src/Strata/PANCAKE_STATUS.md),
  [arena-warm-ab.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/arena-warm-ab.txt)
- IQ1_M rows (3 layers of the IQ2_XS file's gate/up) take the ggml-cpu path; native AVX2 kernel 1-4% on IQ2_XS, 0 on
  Q2_0 — [MOD_PLAN item 24](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md), [report Europe P3](file:///home/dan/reports/Europe%20papers%20for%20Avalon.md)
- T-MAC (USTC) LUT kernels: first a Q2_0 AVX2 micro-benchmark (GB/s vs DDR4) — [report Asia M6](file:///home/dan/reports/Asia%20papers%20for%20Avalon.md)

D2 - cache misses and routing (the number of CPU experts per window)
- Measured: --pcie-frac 0 118.4 vs base 111.7 (+6%), 104.3 vs 92.5 after 11.7K (+13%); + adapt every window 160 swaps
  121.2; rounds B/C best 128.7 (C-best) —
  [roundA.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/roundA.txt),
  [roundC.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/roundC.txt)
- turbo (STRATA_ROUTE_PRIOR 0.5, cache-conditional routing): 149.3 (A), 163.3 (B), 168.6 (C) vs 128.7 C-best; tail skip 7
  129.7/142.8 — [roundA/B/C.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/roundC.txt);
  on the 3060 lambda 0.5: 2.89% mismatch vs stock 1.17-1.30%, 0 large-margin events, REJECT as default (opt-in) —
  [MOD_PLAN 8.9](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md); validated in principle by Cache-Conditional Experts
  (Qualcomm/UvA) — [report Europe P1](file:///home/dan/reports/Europe%20papers%20for%20Avalon.md); upstream
  Strata's router bias: code +37% decode but agent score 5.6 -> 4.5/10 and loops (PR #5596) —
  [report Avalon upgrades, Strata catalogue](file:///home/dan/reports/Avalon%20upgrades%20from%20upstream%20projects.md)
- AcceptMoE (Imperial): depth-graded verify prior, lambda 0 at position 0; 5-15% over fast-hip, lossy —
  [report Europe P2](file:///home/dan/reports/Europe%20papers%20for%20Avalon.md), MOD_PLAN item 23
- Time-cost cache simulator (Baleen/CMU method; SIEVE/S3-FIFO, Harmonia/Athena SAFARI, MoE-Infinity, ProMoE chunked
  cancellable transfers): build only if >=5% decode —
  [report US P3/P4](file:///home/dan/reports/US%20university%20papers%20for%20Avalon.md),
  [report Europe P5](file:///home/dan/reports/Europe%20papers%20for%20Avalon.md),
  [report Asia P6](file:///home/dan/reports/Asia%20papers%20for%20Avalon.md); simulator done: online 2nd-miss
  +5..16% edit, engine "admit on k-th miss" being coded, --adapt-swaps 64/96/160 queued (dan-cd) —
  [PANCAKE_STATUS.md](file:///home/dan/clone-of-d/src/Strata/PANCAKE_STATUS.md)
- Predictors: ProMoE 1.09x avg over LRU (PCIe 4.0); HOBBIT 4.69-8.68% miss penalty cut vs LRU; models had 8-64 experts;
  on PCIe 3.0 + CPU compute "most gains vanish" — [report Asia, expert prediction](file:///home/dan/reports/Asia%20papers%20for%20Avalon.md);
  measured locally: Foresight slots 0.97 (3060), history predictors cover 20-29% of misses, GPU lookahead top-8 24-35% —
  [MOD_PLAN phase 10](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)
- Already rejected on this card: groupA switches 101.2, ADAPT_LAG 87.1, rdna3 profile 105.6, kv-resident 131072 106.2,
  mmap tier 108.6, zh60k vocab 92.5 — [roundA.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/roundA.txt);
  ClassLRU decay replay within 0.5 points — [MOD_PLAN research index](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)
- Domain/conversation seeds (items 3-4, merged): offline +0.6..+1.5 hit points, needs traces —
  [MOD_PLAN "Items 3-4"](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)

D3 - decode drop after long prompts (dan-cd, item 2)
- "the prompt path borrows 9,412 cache slots (12.6 GiB); decode after an 11.7K prompt drops ~126 -> ~105 tok/s while it
  refills" — [MOD_PLAN item 2](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md); log: expert cache 10,733-11,089
  slots (14.4-14.9 GiB) and "the prompt path borrows 9412 CUDA0 cache slots (12.59 GiB)" in every start, 5119 slots
  (6.85 GiB) in the last start (lend sweep) — [pancake-flashnext.log](file:///home/dan/clone-of-d/src/Strata/pancake-flashnext.log)
- P-fasthip-confirm: mean 125.0 vs decode after the 11.7K read 105.2 — [profile-fasthip.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/profile-fasthip.txt)
- Research mapping: none specific; series ranking #5 "exact and measured as a loss" —
  [report Pacific/Africa/LatAm closing table](file:///home/dan/reports/Pacific%20Africa%20Latin%20America%20papers.md);
  related: Avalon "VRAM lending" (10.5, dropped on the 3060: +5% slots, no decode change) — [MOD_PLAN 10.5](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)

D4 - GPU floor (15.9 ms; 64% of a short window)
- Item 1 (rocprofv3 done), item 32 (Gated Residual small-M fusion, part of ~2.7 ms HC reads, 1-5%), item 37 (UFMG
  coordinate-descent sweep, 1-3%), FLUTE LDS bank conflicts in IQ HIP kernels (0-10% of GPU experts) —
  [report Asia P5](file:///home/dan/reports/Asia%20papers%20for%20Avalon.md),
  [report Pacific R4](file:///home/dan/reports/Pacific%20Africa%20Latin%20America%20papers.md),
  [report MIT #13](file:///home/dan/reports/MIT%20Harvard%20papers%20for%20Avalon.md)
- Already ported kernels: decode 87-101 -> 104-108 (kernels) -> ~112 (fusions); GDN megakernel 0.0 ms on one GPU;
  HIP graphs tested without gain — [MOD_PLAN phase 11](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md),
  [report Asia](file:///home/dan/reports/Asia%20papers%20for%20Avalon.md)
- llama.cpp #31751 (RDNA3 J=32 MMQ spill fix, up to +36% on 17-32-token verify batches) is listed for Avalon —
  [report Avalon upgrades "Add now"](file:///home/dan/reports/Avalon%20upgrades%20from%20upstream%20projects.md); pancake's
  verify windows are 1-6 tokens through its own rdna_kq/expdec kernels, not ggml MMQ (log: "window up to 6 tokens").
- Upstream Strata TSUM proposal -0.9% decode time (Issue #5459) — [report Avalon upgrades](file:///home/dan/reports/Avalon%20upgrades%20from%20upstream%20projects.md)

D5 - draft step (2.2-2.5 ms = 9% short, 6.7% edit) and tokens per window
- Item 10: MTP drafting inside the captured verify graph (measure launch overhead of the 2.2 ms first); --mtp-q4 queued;
  basalt 2-bit IQ4_XS proxy already in (short +3%, edit +5% on the 3060) —
  [MOD_PLAN item 10](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md), [PANCAKE_FINDINGS 10b](file:///home/dan/clone-of-d/src/Strata/docs/PANCAKE_FINDINGS.md)
- Acceptance: short 302/423 drafts (71%), edit 400/426 (94%), after the 11.7K prompt 249-291 of 407-498 (55-66%) —
  [pancake-flashnext.log](file:///home/dan/clone-of-d/src/Strata/pancake-flashnext.log); --spec 5/6 flat (111.0/109.2)
  — [roundA.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/roundA.txt)
- DISCO (+entropy feature, MiMo/Entropy Sentinel), SpecFit marginal new-miss cost (Beihang/Monash), SuffixDecoding (CMU):
  0-10% on agent turns, exact (length choice only) —
  [report Asia P3](file:///home/dan/reports/Asia%20papers%20for%20Avalon.md),
  [report Pacific R2/R3](file:///home/dan/reports/Pacific%20Africa%20Latin%20America%20papers.md),
  [report US P8](file:///home/dan/reports/US%20university%20papers%20for%20Avalon.md); DFlash2 115.9
  but 84.3 after long prompts (rejected) — [roundA.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/roundA.txt)
- Longer prompt-lookup window (--lookup-max-t 12/16; upstream +11-14% on edits): +1% here (noise), retest —
  [PANCAKE_STATUS.md](file:///home/dan/clone-of-d/src/Strata/PANCAKE_STATUS.md)

### Inferences
- D1 + D2 together are the edit window's 17.4 ms; the measured or modeled exact levers stack roughly to +15-30% on edits
  (admission +5-16% modeled, swaps +4-9% modeled, item 29 1-4%, seeds ~1 point) and the lossy ones add +10-25%
  (deferral) / +10-20% (sparsity) / +30-50% measured (turbo). Because the CPU is near its DDR4 roofline, all of them work
  by removing or hiding CPU bytes, not by faster CPU math.
- Deferral is the only exact-mechanism overlap lever (lossy output, exact protocol), and it is blocked by the HIP
  non-finite bug (Q5); fixing that bug is the gating step for the highest-ceiling item in every report.
- D4: the new native_down<42> coverage gap is a concrete, exact GPU item: an expdec-style down kernel for Q2_0 at
  n_ff 640 could plausibly cut a third of its 5.8% kernel share, i.e. ~0.3-0.6 ms per short window (~1-2% short-chat
  decode, ~0 on edits). No research report names it; MIT's FLUTE and Pacific's UFMG sweep are the closest methods.
- D5: drafting is 2-5% of a window (3060) / 6.7-9% (7900 XT); even halving it gives 3-5% on short chats.

### Gaps
- No measured A/B yet for admission on k-th miss, --adapt-swaps 64-160, item 29, item 13 masks, deferral after the fix
  (all dan-cd, queued or not started).
- AcceptMoE, SpecFit and DISCO have no evaluation on a 512-expert hybrid GDN model or PCIe 3.0 (reports say so).

## Q3. Prefill and per-turn pain points

### Takeaway
Cold prefill is 1.76K tok/s at 4K, ~1.85K at 11.7K, 2.46K at 16K, 2.58K at 32K, 2.62K at 50K; per agent turn the first
token takes 1.43-3.21 s for 1.5-3.2K-token deltas, of which roughly 0.5-0.9 s is a fixed overhead not explained by the
prefill rate. For an opencode session the per-turn overhead and re-reads weigh more than long-prompt kernels; a 7900 XT
prefill time breakdown does not exist yet (the only breakdown is from the 3060).

### Cited Findings
- strata_bench --long: 4K 1756.9 tok/s (first token 2.31 s), 16K 2456.2 (6.56 s), 32K 2580.2 (12.45 s); chat first token
  0.42 s; agent session 6 turns x 3 runs: cache hit 69.43%, turn first token median 2.05 s (1.43-3.21), e.g. turn 2 read
  1470 tokens in 1.43 s, turn 1 read 2894 in 2.04 s, turn 0 609 tokens 1.56-2.11 s; agent decode 117.0 tok/s (93.2-145);
  needle 50,756 tokens at 2616 tok/s; bench.py ~22K cold 2212-2385 tok/s —
  [2026-11-08-7900xt-final-suites.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-final-suites.txt)
- 11.7K reads in the engine log 1807-1926 tok/s on quiet runs, 1149-1408 tok/s on four disturbed runs —
  [pancake-flashnext.log](file:///home/dan/clone-of-d/src/Strata/pancake-flashnext.log)
- Tiny prompts: "prompt 29 tokens = 22 reused + 7 read in 221 ms", "2073 tokens = 2066 reused + 7 read in 267 ms";
  conversation cache "parked 540 tokens in 119.8 ms", "parked 2584 tokens in 110.9 ms", "restored 2066 tokens
  (checkpoint) in 41.0 ms"; 5120 MiB for up to 4 parked conversations —
  [pancake-flashnext.log](file:///home/dan/clone-of-d/src/Strata/pancake-flashnext.log)
- hipBLASLt table matching the runtime (100401 vs 100200 untuned): prompt 1,345 -> 1,850-1,920 tok/s at 11.7K (+38%),
  decode +15% — [MOD_PLAN phase 13](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md),
  [PANCAKE_STATUS.md](file:///home/dan/clone-of-d/src/Strata/PANCAKE_STATUS.md)
- 3060 prefill breakdown (42K prompt, 42.9 s): expert GEMMs 29% (FP16 path), GDN ~12%, PLE 10.6% (4.3 s on host), QSA
  attention 6.7% + projections 5.5% + select 2.6%; STRATA_PF_FUSED int8 +12% at 5K / -7% at 42K; WMMA worse —
  [MOD_PLAN phase 8](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)
- 7900 XT log: "STRATA_PF_GROUP_SCRATCH=1 is ignored: the prompt path runs the FP16 path or pf_wmma here, not MMQ";
  "prompt experts on the fused int8 kernels (STRATA_PF_FUSED=1, #136)"; "prompt chunk auto: 32768 tokens, a 192-slot
  ring"; "PLE direct read probe: 31707 rows/s" — [pancake-flashnext.log](file:///home/dan/clone-of-d/src/Strata/pancake-flashnext.log)
- Prefill technologies: GDN chunk 64 with blockwise 16x16 inverse + fused prep kernels (gfx11 path works in 16-token
  chunks; 5-11% of prefill est.) — [report MIT #4](file:///home/dan/reports/MIT%20Harvard%20papers%20for%20Avalon.md),
  MOD_PLAN item 18; BFLA (CityU, benchmarks Flash-Next, 1.34-1.44x TTFT at 128-256K, port on feat/bfla-hip) and
  XAttention (20-40% at 131K+, lossy) — [report Asia P9](file:///home/dan/reports/Asia%20papers%20for%20Avalon.md), item 19;
  upstream exact long-prompt QSA scorer +10-13% at 131K / +22% at 240K; PA_FAST + fused prefill + ring 512 +13.8% at
  94K; PLE table resident in RAM +22.5% at 8K (does not fit 57 GB); KVA prefill +28% but decode -11%; MADV_SEQUENTIAL
  follow-up turns 65-76 s -> 23-28 s on RX 6900 XT/32 GB —
  [report Avalon upgrades, Strata catalogue](file:///home/dan/reports/Avalon%20upgrades%20from%20upstream%20projects.md);
  Fiddler (UW) token-count threshold for STRATA_PREFILL_CPU_SHARE (preset 0.6, max 4096) —
  [report US P11](file:///home/dan/reports/US%20university%20papers%20for%20Avalon.md),
  [flashnext-iq2_xs-fast-hip.json](file:///home/dan/clone-of-d/src/Strata/configs/pancake/flashnext-iq2_xs-fast-hip.json)
- Per-turn cache technologies: restore-equivalence test (UNSW, hybrid-state cache off-by-one) item 35; preserve_thinking
  end to end item 31 (if opencode drops earlier `<think>`, the cache re-reads from the first dropped block every turn);
  checkpoints at tool-call/tool-result boundaries + FLOP-saved-per-byte eviction (Marconi, Princeton; FreeToken) item 16;
  turn-boundary checkpoints item 11 (STRATA_CACHE_MESSAGE_BOUNDARY + --prompt-cache-tail; author: first edit -33%) —
  [MOD_PLAN items 11, 16, 31, 35](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md),
  [report Pacific R1](file:///home/dan/reports/Pacific%20Africa%20Latin%20America%20papers.md),
  [report Asia P2](file:///home/dan/reports/Asia%20papers%20for%20Avalon.md); llama.cpp #30137 ref-counted
  checkpoints, #24891/#30745 checkpoint placement (reuse 91 -> 94%) —
  [report Avalon upgrades](file:///home/dan/reports/Avalon%20upgrades%20from%20upstream%20projects.md)
- ContextPilot dedup measured at 1.6% savings on 40 real opencode sessions: keep off —
  [MOD_PLAN series ranking](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)

### Inferences
- Fixed per-turn overhead: turn 2 read 1470 tokens in 1.43 s; at ~1.75K tok/s (the 4K cold rate) that is ~0.84 s of
  reading, leaving ~0.6 s; turn 0 (609 tokens) 1.56-2.11 s implies ~1.2-1.8 s before or around a tiny read. Visible
  pieces: ~0.22-0.27 s floor for a 7-token read, ~0.11 s park, ~0.04 s restore, first window graph capture (183.6 ms for the
  6-token window on first use). Part of the rest is likely the server (template, tokenization, grammar) and the first
  decode window; phase 14's per-turn timeline is the right instrument. This overhead is per turn regardless of tok/s.
- Agent-turn share: with ~500 answer tokens at ~117 tok/s (~4.3 s) plus thinking, 1.4-3.2 s of first-token time is
  roughly 20-40% of a short turn; long cold prompts (>=32K) matter only at session start or after a restart/eviction.
- The int8 fused prompt experts are ON in fast-hip, while on the 3060 they lost 7% at 42K: a chunk-size switch (fused
  below N tokens, FP16 above) is unmeasured on gfx1100.

### Gaps
- No STRATA_PREFILL_TIMING breakdown on the 7900 XT (expert GEMM / GDN / PLE / QSA shares unknown on this card); report
  MIT quotes "GDN 22%" of prefill, the 3060 data say ~12% — source of 22% not found locally.
- The per-turn overhead decomposition (server vs engine) is not measured.
- PLE host time on the 7900 XT and its SSD read share under a long opencode trace are unmeasured (report Asia P7 asks
  for it).

## Q4. Load/startup and memory pressure (VRAM, RAM, disk)

### Takeaway
A restart costs ~30-65 s just to read the 33 GiB expert arena from the ntfs3 partition (0.51-1.18 GiB/s), plus ~10 s of
other setup, and it drops every parked conversation (RAM-only), so the next turn of a long session re-reads its whole
context (12.45 s at 32K). VRAM is full by design (668-880 MiB free), RAM runs inside a 54 GiB cap with 12 GiB kept free,
and the root disk is tight; only MADV/hugepage-style I/O fixes and disk sessions from the research apply.

### Cited Findings
- Startup log (2026-11-09, 13 starts): "loaded 33.02 GiB at" 0.51, 0.63, 0.65, 0.76, 0.92, 1.08, 1.08, 1.08, 1.13, 1.14,
  1.14, 1.16, 1.18 GiB/s; "expert arena read through the file cache (buffered (not Windows))"; "MAP_HUGETLB unavailable
  (needed 16920 2 MiB pages, vm.nr_hugepages=0); transparent huge pages requested"; pin by need "11630 ranges, 18.21 GiB
  pinned"; "494 native projection matrices, 2118.99 MiB of weights, in 4.1 s"; hipBLASLt/FP16 warm "in 2729 ms"; cache
  pre-fill "11089 of 11089 slots ... in 1.9 s (8453 MB/s)"; first verify-window capture 183.6 ms, others 12.6-18.0 ms —
  [pancake-flashnext.log](file:///home/dan/clone-of-d/src/Strata/pancake-flashnext.log)
- VRAM: "expert cache auto: 15.66 GiB free ... -> 10557 slots", "expert cache 11089 slots, 14.85 GiB", "668 MiB of VRAM
  free with everything loaded" (796-880 MiB in other starts); MTP draft layer 835 MiB; KV streaming 98.62% of block reads
  hit VRAM — [pancake-flashnext.log](file:///home/dan/clone-of-d/src/Strata/pancake-flashnext.log)
- RAM: launcher runs in `systemd-run --user --scope -p MemoryMax=54G -p MemorySwapMax=0` with STRATA_RAM_KEEP_GIB=12 and
  waits for 48 GiB available — [tools/pancake/capped.sh](file:///home/dan/clone-of-d/src/Strata/tools/pancake/capped.sh);
  a run without the keep "used the whole 57 GB and killed the desktop session"; the 34 GB arena "often trips systemd-oomd"
  (user@.service 50% / 30 s), mmap tier (RAM_KEEP 20) ~8% slower (69 vs 75 tok/s, 3060) —
  [MOD_PLAN "Next session" and phase 10](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md); mmap tier on the 7900 XT
  108.6 vs 111.7 — [roundA.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/roundA.txt)
- VRAM headroom failures: Foresight 8 slots -> 5.3K prompt failed with "cuBLAS status 14, out of VRAM" (3060) —
  [MOD_PLAN 10.2 step 2](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)
- Idle/sleep: --vram-elastic refused with the pinned RAM complement; --idle-unload is a cold reload (~1 min); basalt
  sleep-in-place (64167ba73, ~815 lines) closes it; FastStrata --conversation-disk 70K conversation after restart 83 s ->
  7.2 s; session files exist by hand (/slots/0?action=save|restore) —
  [MOD_PLAN phase 7](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md); llama.cpp #28092 disk prompt cache (~70 s cold
  prefill at 131K removed), #31886 ROCm staged mmap uploads (load hangs), Avalon registered-RAM arena —
  [report Avalon upgrades](file:///home/dan/reports/Avalon%20upgrades%20from%20upstream%20projects.md)
- Disk: root disk ~7-12 GB free; Flash-Next IQ2_XS shard 1 37 GB + n-gram table 27 GB + MTP 6.5 GB on D; Q2_0 hard-links
  the same n-gram table; DeepSeek V4.1 (239-274 GB) does not fit; Coder variant 29.6 GB competes with Avalon's model
  (~20 GB short) — [MOD_PLAN phase 9 note, item 25](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md),
  [PANCAKE_FINDINGS 10d](file:///home/dan/clone-of-d/src/Strata/docs/PANCAKE_FINDINGS.md)

### Inferences
- Arena read time = 33.02 GiB / (0.51-1.18 GiB/s) = 28-65 s per start; the spread (2.3x) with identical files points
  at page-cache state and concurrent I/O, not the engine. NVMe should do several GB/s; the ntfs3 buffered path is the
  likely limiter (the Strata setup memory note says the ntfs3 mount is required).
- Hugepages (16,920 x 2 MiB) would cut TLB misses for the CPU experts' 33 GiB arena; enabling vm.nr_hugepages is a system
  setting the owner must change and was never measured here. No research report covers it.
- During dan-cd's A/B queue the engine restarts every ~4-5 minutes (log timestamps 09:31, 09:36, 09:40, 09:45, ...),
  so in benchmarking the load time is a real tax; in daily use it is a per-restart cost.

### Gaps
- Total time from "engine started" to "session is up" is not timestamped in the log; only component times are.
- No measurement of arena load from a non-NTFS filesystem or with MADV_SEQUENTIAL/WILLNEED on this box.
- No measurement of major page faults per turn on the 7900 XT (the Avalon report recommends measuring them first).

## Q5. Stability, correctness, determinism and build problems

### Takeaway
Three engine problems block or blur the speed work: VERIFY_DEFER_CPU still produces non-finite logits on HIP (it blocks
the highest-ceiling lever), outputs are not deterministic run to run because GPU-cached and CPU-computed experts round
differently and the adaptive tier moves experts (1.17-1.30% stock-vs-stock mismatch), and run-to-run speed variance is
~20%. Build problems: strata-poly is CUDA-only in practice on HIP; the reported STRATA_PREFILL_MMQ=OFF link failure has no
trace in the logs and a static check of the stubs did not find it. No researched technology addresses these directly.

### Cited Findings
- Deferral on HIP: A-defer-cpu-lossy decode "0.0 13.7 0.0 17.3 0.0 21.9 0.0 20.1 0.0 22.8 0.0 18.0 mean 9.5", decode
  after 11.7K 0.0 — [roundA.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/roundA.txt);
  C-best-defer-lossy "0.0 0.0 68.1 0.0 35.9 68.3 0.0 58.7 0.0 23.8 mean 25.5" with an HTTP 503 traceback before it —
  [roundC.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/roundC.txt)
- "VERIFY_DEFER_CPU gave non-finite logits with COMBINE_CPU (fixed: cfuse off when deferring)" and later "still
  non-finite on HIP with the ported kernels" — [MOD_PLAN phase 11](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md);
  code: `cfuse_on` requires `!defer_` ([verify.cpp:1467-1472](file:///home/dan/clone-of-d/src/Strata/src/core/verify.cpp));
  bisect queued (queue2, 6 runs), 10% — [PANCAKE_STATUS.md](file:///home/dan/clone-of-d/src/Strata/PANCAKE_STATUS.md)
- fast-hip turns on other ported switches that touch the combine/hit path: STRATA_WAIT_FUSE, STRATA_ROUTE_PUBLISH,
  STRATA_GR_FUSED, STRATA_VERIFY_QDEDUP, STRATA_ADAPT_OVERLAP, STRATA_EXPDEC, STRATA_QFUSE —
  [flashnext-iq2_xs-fast-hip.json](file:///home/dan/clone-of-d/src/Strata/configs/pancake/flashnext-iq2_xs-fast-hip.json)
- Determinism: the engine prints "the GPU computes the experts in the cache; it rounds differently from the CPU, so a
  reply can differ slightly from a run without the cache" — [pancake-flashnext.log](file:///home/dan/clone-of-d/src/Strata/pancake-flashnext.log);
  token forcing: stock a/b mismatch 1.17% / 1.30% ("the expert cache is not deterministic") —
  [PANCAKE_FINDINGS 10e](file:///home/dan/clone-of-d/src/Strata/docs/PANCAKE_FINDINGS.md); final preset token forcing
  "INCONCLUSIVE +0.4 pts, 0 large-margin events" — [MOD_PLAN phase 11](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md);
  near-tie splits (la-off's second short run splits at token 133) — [MOD_PLAN 10.2 step 3](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)
- Speed variance: base arms 111.7 / 87.5 / 111.8 / 111.9 / 100.5 (rounds A-C); 11.7K reads 1104-1373 tok/s in disturbed
  arms vs ~1810 normal — [roundB.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/roundB.txt),
  [roundC.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/roundC.txt); decode after the
  11.7K prompt 37.3-114.8 tok/s across log entries — [pancake-flashnext.log](file:///home/dan/clone-of-d/src/Strata/pancake-flashnext.log);
  "run noise of ~20% between a prompt's first and second run" — [MOD_PLAN 10.2](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)
- strata-poly on HIP: `run-pancake.sh` exits with "$MODEL needs strata-poly, which is CUDA-only" on the HIP build —
  [run-pancake.sh:49](file:///home/dan/clone-of-d/src/Strata/run-pancake.sh); CMake declares the target for HIP ("On HIP it
  is the same CUDA-shaped code through the hip_compat headers") — [CMakeLists.txt:3052-3061](file:///home/dan/clone-of-d/src/Strata/CMakeLists.txt);
  build-hip has CMakeFiles/strata-poly.dir and poly test dirs but no strata-poly binary; build-hip/build*.log (13 logs)
  show only `Linking HIP executable strata` targets and no `error:` lines (only discarded-hipError_t warnings, ~190
  repeats of `cudaGetLastError` notes) — [build-hip/](file:///home/dan/clone-of-d/src/Strata/build-hip)
- STRATA_PREFILL_MMQ: option default OFF; build-hip has `STRATA_PREFILL_MMQ:BOOL=ON`; STRATA_PF_FUSED (fast-hip) is
  defined only in the MMQ branch (`STRATA_PREFILL_MMQ=1 STRATA_PREFILL_FUSED=1`) —
  [CMakeLists.txt:94, 2649-2712](file:///home/dan/clone-of-d/src/Strata/CMakeLists.txt),
  [build-hip/CMakeCache.txt](file:///home/dan/clone-of-d/src/Strata/build-hip/CMakeCache.txt); prefill.cpp carries stub
  namespaces for builds without MMQ/fused ("a HIP build without the fused kernels (STRATA_PREFILL_MMQ off) still links") —
  [prefill.cpp:116-162](file:///home/dan/clone-of-d/src/Strata/src/prefill/prefill.cpp)
- Preset switches that do nothing on this build: "STRATA_SELECT_LONG=1 needs the CUDA build with
  STRATA_EXPERIMENTAL_SM60; ignored", "STRATA_PF_GROUP_SCRATCH=1 is ignored" —
  [pancake-flashnext.log](file:///home/dan/clone-of-d/src/Strata/pancake-flashnext.log); head mislabelled
  "experimental native Q5_K head" while the IQ2_XS head is IQ4_XS — [PANCAKE_FINDINGS 10](file:///home/dan/clone-of-d/src/Strata/docs/PANCAKE_FINDINGS.md)
- run-pancake.sh defaults to MODEL=iq2_xs while the plan says "Item 6 closed (Q2_0 is the daily file)" —
  [run-pancake.sh:23](file:///home/dan/clone-of-d/src/Strata/run-pancake.sh), [MOD_PLAN series ranking](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)
- Correctness of cache restore: UNSW found an off-by-one in a hybrid-state cache; item 35 is the precondition for 3, 11,
  16, 31 — [report Pacific R1](file:///home/dan/reports/Pacific%20Africa%20Latin%20America%20papers.md)
- Security of files: ETH "Mind the Gap" GGUF-quantization backdoors -> official files + SHA-256 —
  [report Europe M3](file:///home/dan/reports/Europe%20papers%20for%20Avalon.md)

### Inferences
- Deferral bisect: with cfuse already excluded, the remaining suspects are the other switches on the combine/hit path
  listed above (WAIT_FUSE, ROUTE_PUBLISH, GR_FUSED, QDEDUP, ADAPT_OVERLAP, EXPDEC), because they assume CPU rows have
  landed before the combine reads them; this is a reading of the preset, not a confirmed cause (unconfirmed static
  analysis). The pattern "short runs 0.0, edit runs 14-68 tok/s" suggests the failure depends on window shape or on the
  first request.
- Determinism has two parts: CPU (AVX2, Q8 activations) vs GPU kernels rounding differently, and placement that depends
  on history (--adapt-every 1, 32 swaps, conversation seeds). A deterministic test mode (fixed profile, adapt off) would
  make exactness checks bitwise for A/B of exact kernels; bitwise CPU/GPU parity would be a large kernel project. No
  report in the series covers batch/placement-invariant kernels.
- MMQ=OFF: the static check found every `mmq::`/`fused::` symbol used outside the MMQ sources either stubbed in
  prefill.cpp or header-inline (mmq_ids_table_rows in mmq_ids_margin.hpp; f16_to_f32/set_bounds inside
  `#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)` in gemm.cu). So the reported link failure likely comes from
  elsewhere (e.g. pancake/rdna3 ports or strata_poly_mmq sharing ggml_cuda_host.cu); unconfirmed. Even a linking OFF build
  would silently lose the fast-hip fused int8 prompt experts.
- The ~20% variance means a 1-5% exact item (items 29, 32, 37, the Q2_0 down kernel) cannot be accepted from 2-3 runs;
  the phase-14 load-gated A/B harness and repeated-run distributions (report US M4) are preconditions for the small items.

### Gaps
- No log line or crash dump in the repo shows where the non-finite value first appears (layer, window size).
- No build log for a STRATA_PREFILL_MMQ=OFF configuration or for the strata-poly HIP target exists locally.

## Q6. Gaps where no researched technology helps

### Takeaway
The hardest limits are physical or environmental: DDR4 bandwidth for CPU experts, PCIe 3.0 for promotions, 20 GB VRAM
(44% of expert pairs resident), ntfs3 load speed, the tight root disk, a shared machine's noise, and CPU-vs-GPU rounding.
The research series offers no lever for them beyond moving fewer bytes.

### Cited Findings
- CPU experts at ~49 us each vs 42 us pure-bandwidth estimate; RAM 33 GB/s — [profile-fasthip.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/profile-fasthip.txt),
  [MOD_PLAN item 14](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)
- PCIe 3.0 x16 13.4 GB/s; promotion ~115 us vs ~42 us CPU compute; reuse-aware admission failed on PCIe 3.0 cost;
  --pcie-frac 0 best — [MOD_PLAN phase 10, phase 11](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)
- "None of the papers evaluate PCIe 3.0, AMD GPUs, 512 experts per layer or a GDN hybrid with misses computed on the
  CPU" — [report Asia closing gaps](file:///home/dan/reports/Asia%20papers%20for%20Avalon.md)
- 10,733-11,089 resident slots of 24,576 ranked pairs — [pancake-flashnext.log](file:///home/dan/clone-of-d/src/Strata/pancake-flashnext.log)
- Owner-side hardware levers listed: BIOS iGPU memory 4 -> 2 GB, DDR4-3600 at FCLK 1800; second GPU (phase 12) —
  [MOD_PLAN "Owner's call", phase 12](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)

### Inferences
- No-research gaps: (1) DDR4 roofline of CPU experts (only DDR4-3600/FCLK tuning, owner's call); (2) PCIe 3.0 link;
  (3) VRAM capacity (second GPU, phase 12); (4) ntfs3 arena load (filesystem choice, hugepages: system settings);
  (5) root disk space; (6) CPU-vs-GPU rounding determinism; (7) HIP build hygiene (strata-poly HIP port, MMQ=OFF
  configuration, nodiscard warnings); (8) the Q2_0 down kernel coverage gap (own kernel work; FLUTE/UFMG only as
  methods); (9) the unexplained ~0.5-0.9 s per-turn fixed overhead (needs measurement, then plain engineering);
  (10) machine noise during A/B (process, not research).

### Gaps
- Whether DDR4-3600/FCLK 1800 raises the 33 GB/s figure is unmeasured.

## Q7. Ranking: time share x fixability

### Takeaway
Rank 1 is the CPU-expert block on code edits (D1+D2 exact levers, dan-cd), with the deferral fix as its gate; rank 2 is
the turbo-hip quality decision (largest measured gain, lossy, built); rank 3 the per-turn overhead and cache
correctness; then the post-long-prompt drop (dan-cd), the GPU floor including the newly found Q2_0 down kernel gap, the
draft step, cold prefill, and load/startup.

### Cited Findings
- Series-wide ranking: 1) GPU profile + kernel tuning (1, 32, 37); 2) turbo-hip decision (7); 3) sparsity masks + pool
  barriers (13, 29); 4) per-turn cache with restore test (35, 31, 16, 11); 5) decode cache through long prompts (2);
  6) fix item 8 then rank guard (8, 28); 7) DISCO/SpecFit/AcceptMoE (15, 36, 23); 8) effort/thinking (12, 21, 33, 34);
  9) time-based cache simulator (14, 27, 30, 3-4, 17); 10) tool reliability (26) —
  [report Pacific/Africa/LatAm closing table](file:///home/dan/reports/Pacific%20Africa%20Latin%20America%20papers.md)
- Phase 13 (later, measured-gain order): P1a deferral, P1b simulator, P1d pool phases, P2a turbo decision, P3a/P3b,
  P1c, P2b, P4 — [MOD_PLAN phase 13](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)

### Inferences
Ranking used here: time share of a typical opencode turn (edit-heavy decode, short deltas) x probability the lever lands
x inverse effort. Gains are from the cited measurements or report estimates, not new measurements.

| Rank | Pain point (evidence) | Time share | Levers (source, MOD_PLAN item) | Expected gain | Effort | Risk | Status / owner |
|---|---|---|---|---|---|---|---|
| 1 | D1+D2 CPU experts on edits: 17.4 of 37.4 ms; 357 experts/window at ~27 GB/s | 47% of edit window, 21% short | k-th-miss admission + more swaps (Baleen-style time simulator, 14/27/30); pool per-expert barriers (Tsinghua OSDI'26, 29); seeds (3-4) | +5-16% edit (model), +4-9% swaps, +1-4% (29) | medium | low (exact) | simulator done; engine A/B queued; dan-cd |
| 2 | Same block, lossy: routing toward resident experts | as rank 1 | turbo-hip (Cache-Conditional Experts, 7); AcceptMoE depth-graded prior (23) | +30-50% measured (149-169 tok/s); 5-15% est. | tiny / medium | quality (2.9% mismatch on 3060; upstream agent score drop) | quality gate pending; owner decision |
| 3 | Deferral blocked by HIP non-finite | gates rank-1/2 class lever | bisect, then KTransformers rank guard (8, 28) | +10-25% edit est. (up to 1.45x) | medium | medium (lossy) | bisect queued; dan-cd |
| 4 | Per-turn first token 1.43-3.21 s; ~0.5-0.9 s fixed; 7-token read 0.22-0.27 s; park 0.11 s | ~20-40% of a short agent turn | restore test (UNSW, 35), preserve_thinking (31), tool-boundary checkpoints (Marconi/FreeToken, 16), turn-boundary checkpoints (11), #30137 ref-counted checkpoints; phase-14 timeline | seconds per turn; first edit -33% (author) | small-medium | low | 11 queued (queue2); 35/31/16 dan-b5 |
| 5 | Decode after long prompt 126 -> 105 (prompt path borrows 9,412 slots, 12.6 GiB) | turns after >~10K reads | keep cache through prompts (2) | +20% on those turns | medium | low (exact) | lend sweep queued; dan-cd |
| 6 | Lossy CPU bytes: sparsity | as rank 1 | Berkeley gate mask / TEAL / CATS / CHESS (13) | +10-20% (ceiling 1.25x) | offline 1-2 days, kernel 1-2 weeks | medium-high (quality) | not started; dan-cd |
| 7 | GPU floor 15.9 ms: HC reads 2.73 ms, VRAM-hit experts ~4.3 ms, native_down<42> 5.8% of kernel time (expdec does not cover Q2_0 down), 395 kernels/token, >1 ms host stalls at window edges | 64% of short window, ~0 net on edits | Gated Residual small-M (32), UFMG sweep (37), FLUTE LDS check, own expdec down kernel for type 42 (no item yet), window-edge stall fix | 1-5% each on short chats | small-medium each | low (exact) | profile done, work not started (P4) |
| 8 | Draft step 2.2-2.5 ms; acceptance 55-66% after long prompts | 6.7-9% of window | MTP in the verify graph (10), --mtp-q4, DISCO/SpecFit/SuffixDecoding (15, 36) | 0-10% | small-medium | low (exact) | --mtp-q4 queued; DISCO other session |
| 9 | Cold long prefill: 32K first token 12.45 s; 1.76K tok/s at 4K | session start / restart | GDN chunk 64 (18), BFLA >=64K (19), exact long QSA scorer, PA_FAST + ring 512, Fiddler threshold for PREFILL_CPU_SHARE, fused/FP16 chunk switch | 5-11% (GDN), 1.3-1.4x at 128K+ (BFLA) | medium-large | low-medium | BFLA port in progress; rest not started |
| 10 | Load/startup: arena 28-65 s at 0.51-1.18 GiB/s (ntfs3, no hugepages); restart loses parked conversations | per restart (every ~4-5 min during A/B) | MADV_SEQUENTIAL/WILLNEED (Strata PR #5561), disk sessions (--conversation-disk / #28092), sleep-in-place (basalt), hugepages (owner) | tens of seconds per restart; 83 -> 7.2 s for a 70K resume (FastStrata) | small (madvise) to large | low | not started |
| 11 | Determinism + ~20% speed variance | blurs every <5% A/B | deterministic test mode; load-gated nightly A/B harness, repeated-run distributions (US M4, phase 14) | indirect | small-medium | none | not started |
| 12 | Build: strata-poly CUDA-only on HIP; reported MMQ=OFF link failure unconfirmed; nodiscard warning noise; preset switches ignored on HIP; default model iq2_xs vs daily Q2_0 | maintenance | no research lever | 0 speed | small | low | not started |

### Gaps
- Time shares of ranks 4, 9, 10 in real daily opencode use need the phase-14 agent-session timeline; the agent bench
  (6 turns, 69% cache hit) is the only proxy.
