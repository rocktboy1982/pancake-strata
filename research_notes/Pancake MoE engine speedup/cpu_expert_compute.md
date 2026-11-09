# CPU-side MoE expert FFN compute during batch-1 / small-window decode (CPU+GPU hybrid)

Target used to judge applicability: pancake-strata (C++/HIP), Qwen3.8-Flash-Next 125B MoE (512 experts, top-10, 48 layers, IQ2_XS experts ~2.31 bpw, ~1.37 MiB/expert, n_embd 2560), RX 7900 XT 20 GB + Ryzen 7 5700G (8 Zen3 cores, AVX2/FMA/F16C, no AVX-512/AMX), DDR4-3200 dual channel (51.2 GB/s theoretical, ~33 GB/s measured on expert reads), 80-93% GPU expert-cache hit rate, CPU computes misses for a ~4-row speculative verify window, GPU waits ~5+ ms per window on the CPU. Threads 6-12 measured equal, 4 and 16 slower; expert arena on 2 MiB THP.

Research budget note: about 30 tool calls. Two requested papers (Apto arXiv 2611.01859, Apex arXiv 2610.23006) could not be found by search (see Gaps). The KTransformers SOSP'25 PDF was read in full text; the others mostly from abstracts/HTML.

---

## 1. AVX2 low-bit GEMV kernels for IQ2_XS-class formats: how close to DRAM bandwidth, and what tricks help

### Takeaway
Mainline ggml's AVX2 IQ2_XS/IQ2_XXS dot products are single-row (`nrc == 1`), use scalar 64-bit codebook loads per 8 weights, and decode signs with a shuffle/compare chain. In a 4-row verify window that means each weight block is decoded 4 times. That is the biggest structural inefficiency for this target. Interleaved-row repacking, multi-row/multi-column tiling (decode once, multiply N activations), and the IQK-style kernels in ik_llama.cpp/llamafile are the known fixes. I found no source that gives a %-of-peak number for IQ2_XS token generation on Zen3.

### Cited Findings
- ggml AVX2 `ggml_vec_dot_iq2_xs_q8_K` starts with `assert(nrc == 1)`, so it handles one weight row x one activation column per call. The grid is fetched with scalar table loads packed via `_mm256_set_epi64x(iq2xs_grid[gindex[3]], ...)`, using no gathers. The 9-bit index comes from `_mm256_and_si256(q2_data, m511)`. Signs come from `srli` plus `_mm256_shuffle_epi8` parity, `cmpeq` expansion and `_mm256_sign_epi8`. Products use `_mm256_maddubs_epi16`, scales use `_mm256_madd_epi16` with `2*s+1`, and the loop processes 4x32 weights per iteration with two accumulators. A source comment calls the uint16 view of the register "somewhat hacky, but gives a significant boost in performance" — [llama.cpp ggml-cpu/arch/x86/quants.c](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cpu/arch/x86/quants.c)
- IQ2_XXS AVX2 works the same way: byte-index scalar grid loads, sign lookup via `keven_signs_q2xs[(aux32[1]>>21)&127]`, `_mm256_sign_epi8`, and also `nrc == 1` — [same file](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cpu/arch/x86/quants.c)
- Other x86 k-quant dot products (for example q2_K) also begin with `assert(nrc == 1)` — [same file](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cpu/arch/x86/quants.c)
- IQ2_XS is a 2.31 bpw format — [llama.cpp quantize.cpp listing](https://git.peisongxiao.com/CS348Project/llama.cpp/src/tag/b2579/examples/quantize/quantize.cpp)
- ikawrakow's llamafile PR #394 reports that the same GEMM approach gives "huge performance gains" for IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ3_S. It was benchmarked on a Ryzen 5975WX (Zen3, no AVX-512) to show speedup on non-AVX-512 CPUs. The gains were for prompt processing, and the i-quant part was held back because his versions were incompatible with mainline — [llamafile PR #394 mirror](https://git.tdem.in/Mozilla-Ocho/llamafile/pull/394)
- llamafile 0.8.2 release notes: token generation got faster "in some cases by as much as 1.33x" (k-quants/IQ4_XS), and TG gains were "much more difficult" than PP gains — [Phoronix](https://www.phoronix.com/news/Llamafile-0.8.2-More-AVX2)
- ik_llama.cpp offers row-interleaved repacked types (`_R4` variants, e.g. IQ2_K_R4, IQ4_K_R4) and a `-rtr` load-time repack of CPU-resident tensors. Community forks warn not to use `-rtr` for hybrid MoE with experts on CPU "unless you know what you are doing", because repacked tensors cannot be offloaded/streamed to GPU — [ik_llama.cpp fork README](https://github.com/RodriMora/ik_llama.cpp); [ik_llama.cpp discussion #164](https://github.com/ikawrakow/ik_llama.cpp/discussions/164)
- Forum consensus is that ik_llama.cpp TG is "memory bandwidth bottlenecked" and unaffected by PP kernel changes (Zen5 context) — [Level1Techs](https://forum.level1techs.com/t/deepseek-deep-dive-r1-at-home/225826/463)
- kt-kernel (KTransformers' CPU kernel package) has an AVX2 "LLAMAFILE" backend. It consumes pre-quantized GGUF directly (Q4_K_M, Q4_K, Q5_K, etc.) with no conversion. AVX2 can be forced with `KT_KERNEL_CPU_VARIANT=avx2`, and it is the fallback when AVX-512/AMX is absent. I found no AVX2 throughput numbers — [kt-kernel on PyPI](https://pypi.org/project/kt-kernel/)
- KTransformers' AMX/AVX-512 path uses a tiling-aware layout: block-wise quantization, 64-byte alignment, weights pre-arranged so no runtime transpose/reshape is needed, and Int4 packed into Int8 blocks and unpacked with SIMD. Its decode kernel (AVX-512 on the same layout) gives "up to 1.20x speedup in decode" over AMX when ARI <= 4 — [KTransformers SOSP'25 PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- KTransformers fused MoE operator: Gate projections are merged across experts into one task, Up and Down likewise, and Gate+Up are combined, so an MoE layer is "two fused batches", which lowers threading overhead — [KTransformers SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

### Inferences
- **The 4x decode duplication is the main exact, kernel-level opportunity.** With `nrc==1`, a 4-row window runs the full IQ2_XS decode (4 scalar grid loads + vpinsr chain + sign expansion per 32 weights) four times per weight block. DRAM traffic stays at 1x because the block is L1-resident after the first column, but ALU work is roughly 4x. A multi-column kernel (decode the 32-weight block once into a register, then `sign_epi8`/`maddubs`/`madd` against 1-5 q8 activation columns) cuts per-row decode cost about 3-4x. ik_llama.cpp's IQK `mul_mat` templates (`nrc_y` up to 8) and llamafile tinyBLAS follow this pattern. It is bit-identical if integer accumulation order is preserved per row, and greedy-identical in practice even if float reduction order changes.
- **Rough budget (my calculation, not sourced):** 33 GB/s / 2.3125 bpw ≈ 114 G weights/s ≈ 3.6 G 32-weight groups/s. Eight Zen3 cores at ~4.2 GHz give ~33.6 G cycles/s, so about 9 core-cycles per 32-weight group must cover all rows of the window. The ggml path costs roughly 20-30 uops per 32 weights per row (estimate from the instruction sequence), so 4 rows on the ggml path is plausibly compute-limited. That fits "4 threads slower, 6-12 equal": with >=6 threads the CPU is at the bandwidth ceiling the access pattern allows.
- **Row interleaving (R4/R8):** It processes 4-8 output rows per inner loop, so one q8 activation load and one scale setup are amortized over several rows and partial sums stay in registers. It also turns several concurrent streams into one sequential stream per thread, which is friendlier to the L2 stream prefetcher. Repacking at load time is exact. On this target the arena is CPU-only for missed experts but the same experts may also be uploaded to the GPU, so either keep two layouts, repack on the GPU side, or have the HIP kernel read the interleaved layout.
- **Expected gain on target (estimate):** if the current kernel is ggml-style `nrc==1`, a multi-column IQK-style kernel should move it from partly compute-limited to bandwidth-limited. That could be 1.2-1.5x on the CPU part, more if the window is 5 rows, less if already at ~33 GB/s effective. If the engine already decodes once per window, the remaining headroom is only bandwidth (33 to ~40-45 GB/s, see Q3).

### Gaps
- No source found giving measured GB/s or % of peak for IQ2_XS/IQ2_XXS TG on Zen3 (ggml or ik_llama). ik_llama discussion #164 timed out on fetch.
- No published comparison of prefetch distance, non-temporal loads (pointless for reads on x86; `prefetchnta` is the only relevant variant) or software prefetch for i-quant GEMV on Zen3.
- The exact uop count per 32 weights is not profiled in any source; mine is an estimate.

---

## 2. LUT-based methods (T-MAC, Vec-LUT, bitnet.cpp TL1/TL2, LUT-GEMM, ELUTQ) for 1-5 token GEMV on AVX2

### Takeaway
T-MAC supports AVX2 (`vpshufb` table lookups) and reports up to 4-5x mpGEMV kernel speedup over llama.cpp at 2 bits. Those formats are uniform-grid/bit-plane ("bit-serial") formats, though, not IQ2_XS's E8-lattice codebook. Scalar LUTs degrade with multiple tokens, which Vec-LUT addresses. Using a LUT method would require re-quantizing the experts into a different format (lossy versus the IQ2_XS reference), so it is not an exact drop-in.

### Cited Findings
- T-MAC on AVX2 uses `_mm256_shuffle_epi8` for look-up and `_mm256_avg_epu8` for fast aggregation. It duplicates tables across the two 128-bit lanes to look up 32 int8 indices per instruction, and uses `vpblendvb/vpermd/vpshufb` for table precompute — [T-MAC arXiv 2407.00088v2](https://arxiv.org/html/2407.00088v2)
- T-MAC's x86 test device is a Surface Book 3 (i5-1035G7, 4 cores, 58.2 GB/s). 2-bit mpGEMM speedup is up to 5.3x versus llama.cpp's BLAS path. mpGEMV speedups across four devices are 4.0x, 4.0x, 5.31x and 2.5x (device mapping unclear). There are no x86 end-to-end 2-bit numbers — [T-MAC](https://arxiv.org/html/2407.00088v2)
- T-MAC: "performance is primarily constrained by memory bandwidth" for multi-threaded mpGEMV. Tables are built online from activations, and larger M-tiles improve LUT reuse — [T-MAC](https://arxiv.org/html/2407.00088v2)
- T-MAC headline numbers: BitNet-b1.58-3B at 30 tok/s on 1 core and 71 tok/s on 8 cores on M2-Ultra. Average 3.6x and up to 6.6x kernel speedup over llama.cpp — [T-MAC abstract](https://arxiv.org/abs/2407.00088)
- Vec-LUT (arXiv 2512.06443, Dec 2025, MobiSys'26): scalar LUT methods are good for single-token decode but leave memory bandwidth "≤ 40% of its capacity" for parallel multi-token inputs, because they do repetitive, non-contiguous per-token lookups. Vec-LUT builds one unified table with results for the same index contiguous across tokens (1-to-N lookup), plus a token-contiguous layout and cache-aware streamed lookup. It reports up to 4.2x over T-MAC, bitnet.cpp and llama.cpp. It is integrated into llama.cpp (Cipherxzc/vlut.cpp) — [Vec-LUT arXiv](https://arxiv.org/pdf/2512.06443)

### Inferences
- **Multi-row (2-5 token) on AVX2:** scalar-LUT methods (T-MAC, TL1/TL2) build one table per activation column, so table-build cost and lookups scale with the number of rows, and Vec-LUT says bandwidth utilization falls. The Vec-LUT design fits a 4-row verify window best. Its gains are reported for ternary/ultra-low-bit formats.
- **Format mismatch:** IQ2_XS stores 8-weight groups as 9-bit indices into a 512-entry E8 lattice codebook plus sign bits. LUT methods need weights expressible as sums of bit-planes (uniform grids). Running LUT kernels would mean re-quantizing experts (e.g. to a T-MAC 2-bit uniform format). That is lossy versus the current model output and likely lower quality than IQ2_XS at equal bpw. It does not reduce bytes (same ~2-2.3 bpw), and the target is bandwidth-bound at >=6 threads, so LUT gains would show only if the current kernel is compute-limited. A multi-column IQ2_XS kernel fixes that more cheaply and exactly.
- **Expected gain on target:** low priority. It is lossy (format change), and the gain is at best similar to a well-written multi-column IQ2_XS kernel, because both end at the same DRAM ceiling.

### Gaps
- ELUTQ and LUT-GEMM were not searched in depth within the budget. No AVX2 multi-token numbers were found for bitnet.cpp TL2.
- No LUT kernel for E8-lattice codebook formats (IQ2_XS) was found.

---

## 3. Bandwidth-bound or compute-bound? Roofline and achievable DDR4-3200 bandwidth on Ryzen 5000 APUs

### Takeaway
At 2.3 bpw and 1-5 rows the arithmetic intensity is tiny: about 0.9-4.3 int8 ops per weight byte per row, against Zen3's roughly 1 TOP/s int8 across 8 cores. So a well-written kernel is DRAM-bound. The measured 33 GB/s is ~64% of the 51.2 GB/s theoretical peak. I found no sourced STREAM number for the 5700G at DDR4-3200, so how much headroom remains is uncertain. Cezanne's monolithic die is reported to have better write bandwidth than single-CCD Vermeer.

### Cited Findings
- Dual-channel DDR4-3200 is the native spec for Cezanne (Renoir was DDR4-2933), and the APU tolerates high memory overclocks — [TechPowerUp 5700G review](https://techpowerup.com/review/amd-ryzen-7-5700g/3.html)
- Hexus: Cezanne "almost doubles" the memory write bandwidth of Vermeer (single-CCD Zen3 desktop parts), attributed to its monolithic design with 16 MB L3. Numbers are only in images — [Hexus 5700G memory tests](https://hexus.net/tech/reviews/cpu/148125-amd-ryzen-7-5700g/?page=4)
- Single-CCD Zen 2/3 parts show "half-speed write performance" (an Infinity Fabric write-width limit) — [KitGuru 5800X review](https://www.kitguru.net/components/cpu/luke-hill/amd-ryzen-7-5800x-cpu-review/8/)
- An unstated-config Ryzen 7 5800X3D triad test reports about 31,098 MB/s (not directly comparable) — [search result, OpenBenchmarking](https://openbenchmarking.org/result/2102131-HA-GEDWEBMEM10&export=html)
- Decode speed on CPU is "hard-capped by the memory bandwidth" (community) — [Reddit snapshot](https://reddit.sentinel-team.org/posts/1rm4s3v/snapshots/2026-03-06T19%3A04%3A08.896785Z)
- SpecMoEOff's roofline table (Xeon CPUs): Xeon Gold 6426Y = 2.43 TFLOPS / 357 GB/s, Xeon Gold 5418Y = 1.45 TFLOPS / 197 GB/s. CPU attention is preferred over shipping KV to the GPU whenever CPU memory bandwidth exceeds the CPU-GPU link bandwidth — [SpecMoEOff arXiv 2508.21706v2](https://arxiv.org/html/2508.21706v2)
- KTransformers: on a dual-socket Xeon, per-socket bandwidth is 220 GB/s and cross-socket 125 GB/s. MoE CPU decode is limited by "memory bandwidth constraints and thread synchronization" — [KTransformers SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

### Inferences
- **Roofline (my arithmetic):** for one 2.3125-bpw weight byte there are ~3.46 weights, i.e. ~3.46 MACs per row, or ~14 MACs for 4 rows. At 33 GB/s that is ~0.46 T MAC/s for 4 rows. 8 Zen3 cores x 2 x 256-bit `maddubs` per cycle x 16 MAC x ~4 GHz is ~1 T MAC/s int8 peak, but the codebook decode (scalar loads, inserts, shuffles) is the real compute cost, not the MACs. Conclusion: MAC-wise it is bandwidth-bound, decode-wise it is borderline compute-bound if decode is repeated per row (see Q1).
- **Achievable read bandwidth:** Zen3's Infinity Fabric read path is 32 B/FCLK. At FCLK = MEMCLK = 1600 MHz that equals the 51.2 GB/s DRAM peak, so the fabric is not the limiter. Typical Zen3 desktop AIDA64 read results at DDR4-3200 are commonly quoted in the mid-40s GB/s (unsourced here; verify with a local `likwid-bench`/STREAM read-only test when benchmarks are idle). If 44-47 GB/s read is achievable, the engine's 33 GB/s leaves ~1.3-1.4x headroom. Likely causes of the gap: per-expert scattered 1.37 MiB chunks with ramp-up at each start (prefetcher retraining), TLB (mitigated by THP), and threads idling at barriers between gate/up and down.
- **Memory tuning (exact, zero code):** tightening DRAM timings or running DDR4-3600/3800 with FCLK 1:1 raises the ceiling roughly in proportion (3800/3200 = 1.19x). TechPowerUp's iGPU scaling (6% loss from 3800 to 3200) is consistent with this. This is the cheapest bandwidth gain on this machine, if the 57 GB kit (likely 4 DIMMs or mixed) is stable at higher speed.

### Gaps
- No sourced STREAM/AIDA read GB/s for Ryzen 5700G at DDR4-3200 (chipsandcheese Cezanne article not found in search).
- No measured roofline for IQ2_XS GEMV on any Zen3 part.

---

## 4. Hybrid scheduling systems: KTransformers Expert Deferral, Fiddler, HybriMoE, Apto, Apex, TriMoE, MoE-Lightning, SpecMoEOff

### Takeaway
The single most directly applicable idea is **KTransformers Expert Deferral**. It delays some routed experts' outputs from layer k+1's input to layer k+2's, so CPU expert work overlaps GPU attention. It gave +33% decode on DeepSeek-V3 (BF16) and up to 1.45x overall, at the cost of exactness: the paper reports an average accuracy drop of at most 0.5%. **Fiddler** set the rule this target already follows: compute missed experts on the CPU instead of moving weights over PCIe. **HybriMoE** adds dynamic intra-layer CPU/GPU split, impact-driven prefetch and score-based caching (1.70x decode over kTransformers). **SpecMoEOff** uses speculation to enlarge per-expert work, with experts on the GPU and attention on the CPU, which is the inverse of this target's design.

### Cited Findings
**KTransformers (SOSP'25, Tsinghua MADSys + Approaching.AI)** — [paper PDF](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Shared experts and attention run on the GPU, routed experts on the CPU. After GPU gating, a CPU control thread (i) pushes routed-expert tasks into a lock-free queue and (ii) launches the shared-expert GPU kernels. Background worker threads run the queue, and the control thread waits, merges, and proceeds.
- Without deferral on DeepSeek-V3: CPU utilization 74%, GPU 28%, CPU-GPU overlap only 5% of time, and sync plus activation transfer another 3% idle.
- Expert Deferral: "Immediate" experts feed layer k+1 attention, while "Deferred" experts' outputs are added at layer k+2 (relying on residual connections). Deferring 2 of 8 experts cut layer time 19%. Deferring 3 cut it 26% and raised end-to-end decode +33%. Deferring 4 brought no further gain (CPU saturated). The chosen configuration is 5 immediate + 3 deferred for BF16, and 6 deferred in the quantized model.
- Expert Deferral gives up to 1.45x additional throughput, with average accuracy drop "no more than" 0.5% on HumanEval/MBPP/GSM8K/StrategyQA/LiveBench. It is lossy because outputs change.
- Deferral is not applied during prefill, where it would nearly double memory footprints.
- CUDA Graph: the whole dynamic-shaped hybrid forward is captured in one CUDA graph, with CPU tasks launched via host functions, "eliminating host interruptions". This gives up to 1.23x decode. Average kernel launch is ~16 µs per launch, and llama.cpp disables CUDA graphs to avoid recapture overhead.
- NUMA-aware tensor parallelism gives up to 1.63x decode on 2-socket systems (not relevant to a single-die 5700G).
- Overall: 4.62-19.74x prefill and 1.25-4.09x decode versus existing systems, 2.42-4.09x over Fiddler and 1.25-1.76x over llama.cpp decode.
- The SGLang integration describes Expert Deferral as letting some experts run later so CPU expert work overlaps GPU attention — [LMSYS blog](https://lmsys.org/blog/2025-10-22-KTransformers)

**Fiddler (ICLR'25, UW)**
- When an expert is missing from the GPU, Fiddler copies only activations to the CPU, runs the expert there, and returns outputs, instead of moving weights over PCIe — [Fiddler arXiv v3](https://arxiv.org/html/2402.07033v3)
- Per-expert choice: estimated CPU latency (rises roughly linearly with batch size) versus transfer+GPU latency. Popular experts and non-expert layers go on the GPU — [Moonlight review](https://www.themoonlight.io/review/fiddler-cpu-gpu-orchestration-for-fast-inference-of-mixture-of-experts-models); [Liner](https://liner.com/review/fiddler-cpugpu-orchestration-for-fast-inference-of-mixtureofexperts-models)
- Fiddler's CPU expert kernel uses AVX512_BF16 — [Fiddler arXiv](https://arxiv.org/html/2402.07033v3)

**HybriMoE (DAC'25, PKU)** — [arXiv 2504.05897](https://huggingface.co/papers/2504.05897)
- Three parts: (i) dynamic intra-layer scheduling to balance CPU/GPU load per layer (assigning experts to whichever device finishes first, given unstable activations), (ii) impact-driven inter-layer prefetching, (iii) score-based caching. Built on kTransformers.
- Average 1.33x prefill and 1.70x decode versus a state-of-the-art hybrid framework. Exact (scheduling and caching only).

**SpecMoEOff (arXiv 2508.21706)** — [HTML v2](https://arxiv.org/html/2508.21706v2)
- Uses speculative decoding to enlarge per-expert workload. Experts are streamed to the GPU over PCIe while attention runs on the CPU (a chunked verification attention kernel with Intel MKL). For the draft model, attention runs on the CPU and FFN on the GPU.
- Roofline: batch-1 offloading gives 0.76% GPU utilization on A30. A large batch favors the GPU when b >= n_expert / (n_activate x r_miss).
- Up to 2.5x decode throughput over the state of the art, 1.6-2.9x (avg 2.1x) over MoE-Lightning, and speculation alone 1.45-1.53x. Hardware: A30 + Xeon 6426Y, 4090D + Xeon 5418Y, Mixtral-8x7B FP16 with an EAGLE draft. The paper does not explicitly claim lossless output, but standard verification implies greedy-exact.

**TriMoE (arXiv 2603.01058, DAC'26)** — [arXiv HTML](https://arxiv.org/html/2603.01058)
- Hot experts on the GPU, warm experts on an AMX CPU, cold long-tail experts on DIMM-NDP. Bottleneck-aware greedy makespan scheduler plus prediction-driven relayout. Up to 2.83x over Klotski, Enhanced KTransformers and MoNDE (H100 + Xeon 8470 + 16 DIMM-NDPs). Needs NDP hardware, so not applicable.

**Related speculative + MoE offload work**
- SP-MoE (arXiv 2510.10302): speculative expert prefetching using draft/target structural correspondence, 1.07-3.5x TPOT — [arXiv](https://arxiv.org/pdf/2510.10302)
- MoE-SpeQ (arXiv 2511.14102): quantized-draft speculation plus proactive expert prefetch, up to 2.34x on Phi-MoE — [alphaXiv](https://www.alphaxiv.org/abs/2511.14102)

**APEX (arXiv 2506.03296, not MoE-specific)**: profiles CPU vs GPU subtask times to schedule asynchronous CPU-GPU execution on memory-constrained GPUs. +84-96% throughput on T4 and +11-89% on A10 over vLLM — [arXiv](https://arxiv.org/pdf/2506.03296). Confirmed distinct from the brief's "Apex arXiv 2610.23006", which is a separate MoE expert-assignment paper (see Gaps below).

### Inferences
- **Expert Deferral on the target:** the GPU currently idles ~5 ms per window waiting for CPU rows. Deferring the CPU-missed experts' contribution by one layer (adding them to the residual at layer k+2) would let GPU attention and GPU-hit experts for layer k+1 proceed, and CPU work overlaps nearly fully. This has the highest potential gain (up to removing most of the 5 ms stall, roughly 1.3-1.5x decode, consistent with KTransformers' +33-45%). But it is **lossy**: it changes logits, so speculative-verify acceptance and greedy output are no longer identical to the reference. Natural variants for this engine: (a) defer only the lowest-routing-weight missed experts; (b) opt-in flag; (c) defer within the verify window only for draft rows. All of these remain lossy.
- **Exact overlap without deferral:** split the layer so the GPU starts the next work that does not depend on the CPU result: the shared expert (if the model has one), GPU-hit experts, and the next layer's router/attention for rows already known. For a strict data dependency, the only exact overlap is CPU expert compute running concurrently with GPU-hit expert compute in the same layer, plus pipelining across window rows. HybriMoE's dynamic intra-layer balancing (send some misses to the GPU via PCIe if the CPU queue is longer than transfer time) is exact. The 5700G is limited to PCIe 3.0 (my understanding of Cezanne; verify), so x16 is ~13-14 GB/s effective versus CPU ~33 GB/s. Moving an expert (1.37 MiB, ~0.1 ms) only pays when the CPU queue is long, as Fiddler's latency model predicts.
- **Speculation already enlarges per-expert rows** (the ~4-row window), which is exactly SpecMoEOff's lever. On the CPU side it pays only if the kernel decodes once per weight block (Q1). Otherwise CPU time grows with the number of rows.
- **CUDA Graph analogue:** HIP graphs with host-function nodes could remove per-layer launch and sync overhead (KTransformers +1.23x), if the engine currently synchronizes per layer from the host.

### Gaps
- **Apto (arXiv 2611.01859) and Apex (arXiv 2610.23006): both verified on arXiv (earlier "not found" was a false positive — the IDs were simply too new for search indexing).**
  - **Apto: Self-Adaptive CPU-GPU Hybrid MoE Inference on Shared PCs** — Zihang Zhong, Jiali Wang, Rong Chen; submitted 1 Nov 2026, cs.DC. Self-adaptive hybrid inference for shared PCs: CPU and GPU executors claim experts from opposite ends of a dual-ended queue, so the CPU/GPU work split adapts to execution progress without predicting contention; plus affinity-aware scheduling (preferred experts exposed at each executor's queue end) and cancellable tail racing to mitigate final-expert stragglers. Average 2.96x prefill / 1.52x decode speedup standalone, 5.97x / 3.45x under concurrent CPU workloads, and 2.02x on real coding-agent applications. — [arXiv 2611.01859](https://arxiv.org/abs/2611.01859)
  - **Apex: Efficient CPU-GPU collaborative MoE inference via Adaptive Expert Assignment** — Wenhui Lv, Yunming Liao, Hongli Xu, Yang Xu, Junhao Cheng; submitted 16 Oct 2026, cs.AI. Adaptively assigns GPU-missed experts: selectively transfers high-token-load experts to the GPU while executing low-load experts on the CPU, and adjusts assignments to balance the CPU compute path against the GPU transfer-and-compute path so both finish early. Up to +51.9% prefill speed and +39.5% decode throughput over the strongest baseline across three MoE LLMs, consistent across GPU-memory budgets. — [arXiv 2610.23006](https://arxiv.org/abs/2610.23006)
- MoE-Lightning was not fetched (only cited via SpecMoEOff's comparison). It is a batch-throughput system (CGOPipe, HRM) aimed at large batches, so less relevant to batch-1.
- KTransformers' full evaluation of the quantized-model deferral (6 deferred) accuracy table was not extracted in detail.

---

## 5. Thread scheduling: busy-wait vs blocking, pinning on single-CCD Zen3, SMT, work split, barrier cost

### Takeaway
Production engines spin-poll by default. llama.cpp's threadpool polls up to 128K-256K rounds before falling back to a condition variable, and uses atomic barrier counters. KTransformers uses a lock-free task queue with dynamic sub-task stealing and fuses Gate/Up/Down across experts into two batches per layer to cut synchronization. I found no published per-barrier cost for Zen3. My estimate: about 0.5-2 µs per spin barrier, versus 5-50 µs per futex wake.

### Cited Findings
- llama.cpp PR #8672 ("Threadpool: take 2") added hybrid polling. `--poll` sets the polling level: 0 = none, 1 = poll 128K rounds then wait on a cond var, 2 = 256K rounds. Polling became the default "to match openmp behavior", with relaxed memory order while polling in `ggml_barrier` — [llama.cpp PR #8672](https://github.com/ggml-org/llama.cpp/pull/8672/files)
- llama.cpp #9461 ("skip polling for unused threads") keeps full memory fences on barrier entry/exit. For tests with many small parallel matmuls, "the barriers in between dominate the overhead" — [commit 0226613](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/commit/0226613853133c081b55bb892a41bb5eacc0bc94)
- llama.cpp PR #17748 adds atomic `n_barrier` / `n_barrier_passed` counters, and computes poll rounds from the level with a note that it could be adjusted dynamically — [PR #17748](https://github.com/ggml-org/llama.cpp/pull/17748/files)
- KTransformers: dynamic task scheduling splits large tasks into smaller sub-tasks in a lightweight queue that threads pull from (up to 1.83x in prefill). Its fused MoE reduces each layer to two fused batches to cut threading overhead. Its lock-free queue is fed by a control thread — [KTransformers SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- KTransformers cites "thread synchronization" alongside memory bandwidth as the CPU MoE bottleneck — [same](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)

### Inferences
- **Barrier count on target:** an IQ2_XS expert FFN has two dependent phases (gate+up, then down). With all CPU-missed experts of a layer fused into one phase-1 task and one phase-2 task (KTransformers style), there are 2 barriers per layer, so 96 per window over 48 layers. At ~1 µs per spin barrier that is ~0.1 ms of a ~5 ms window (~2%). With futex/condvar wakes at ~10-30 µs it would be 1-3 ms, which is significant. **Keep the pool spinning during decode** (spin with `_mm_pause`, bounded, then park only between requests).
- **Avoid barrier idle tails:** with only ~1-10 missed experts per layer and 6-12 threads, splitting across experts leaves threads idle (10 experts / 8 threads leaves a tail). Split **within** experts by output-row chunks (e.g. 64-128 rows of the 2560/FF dims per task) in a dynamic queue, so all threads finish together. Also merge phase 2 of layer k with nothing else (strict dependency), but phase 1 of all window rows is already merged by the multi-row kernel.
- **Pinning/SMT:** the 5700G is a single CCX (8 cores share 16 MB L3), so there is no cross-CCD penalty. Measured "6-12 equal, 16 slower" fits a bandwidth wall plus SMT siblings competing for load ports, L1/L2 and the stream prefetchers. Recommended: one thread per physical core (8, or 7 leaving one core for the HIP host thread/driver), pinned with `pthread_setaffinity_np` to the even logical CPUs, so the GPU-feeding control thread is never descheduled. Expected gain is small (<5%) but it reduces tail variance.
- **Persistent per-thread expert affinity:** assigning the same row-slices of each expert to the same core does not help L2 reuse across tokens (1.37 MiB x misses far exceeds 512 KiB L2). Its only benefit is prefetcher locality.

### Gaps
- No measured `ggml_barrier` latency on Zen3 was found. No study compares spin vs block for hybrid MoE decode specifically.

---

## 6. Reducing CPU bytes per expert (exact and lossy), and applicability/expected-gain catalogue for the target

### Takeaway
At ~2.31 bpw, exact byte reduction is basically exhausted. The only exact levers are (a) not reading bytes twice (decode once per window, fused gate/up), (b) more achievable bandwidth (better streaming, memory OC), and (c) fewer CPU bytes overall via better GPU caching/prefetch (HybriMoE-style scoring), which is exact. Lossy levers are Expert Deferral (KTransformers, ≤0.5% avg accuracy drop) and lower-bit CPU-resident copies of cold experts. The latter has no direct published system for this setup; it is closest in spirit to TriMoE's hot/warm/cold tiering and MoE-SpeQ's quantized draft.

### Cited Findings
- Expert Deferral: up to 1.45x throughput, avg accuracy drop ≤0.5% — [KTransformers SOSP'25](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf); ~1.45x / <0.5% also reported by [QbitAI](https://www.qbitai.com/2025/10/344731.html)
- HybriMoE's score-based caching plus impact-driven prefetch: 1.70x decode average (exact) — [HybriMoE](https://huggingface.co/papers/2504.05897)
- SpecMoEOff: expert prediction plus caching of hot experts cuts expert-weight transfer cost by over 80% — [SpecMoEOff abstract](https://arxiv.org/pdf/2508.21706)
- TriMoE: hot/warm/cold expert tiering across devices — [TriMoE](https://arxiv.org/html/2603.01058)
- MoE-SpeQ: a quantized draft drives proactive expert prefetch — [alphaXiv](https://www.alphaxiv.org/abs/2511.14102)

### Inferences — technique catalogue for pancake-strata on 5700G + 7900 XT
(The gain numbers here are my estimates for the target unless a source is given; "CPU part" = the ~5 ms per window the GPU waits on.)

| # | Technique | Mechanism | Exactness | Effort (C++) | Expected gain on target |
|---|---|---|---|---|---|
| 1 | Multi-column IQ2_XS kernel (decode once, dot 1-5 q8 rows) | Removes 4x repeated grid/sign decode under `nrc==1` (ggml AVX2) | Bit-identical per row if int accumulation is kept; otherwise greedy-identical | Medium (1-2 days; model on ik_llama IQK `mul_mat_iq2_xs_q8_K<nrc_y>`) | 1.2-1.5x CPU part if currently decode-repeating; ~0 if already done |
| 2 | Row-interleaved repack (R4/R8) of CPU expert arena | 4-8 rows per inner loop, amortized activation loads, sequential streams | Exact | Medium (repack at load; GPU path keeps original or reads interleaved) | 5-15% CPU part (ik_llama `_R4` evidence is qualitative) |
| 3 | Fused gate+up across all missed experts and all window rows, 2 phases/layer, dynamic row-chunk queue | Fewer barriers, no idle tails (KTransformers fused MoE + dynamic scheduling) | Exact | Low-medium | 5-20% CPU part depending on current barrier count/imbalance |
| 4 | Spin-wait pool, 1 thread/physical core, pinned, keep a core for HIP host thread | Avoid futex wake latency (llama.cpp default polling) | Exact | Low | 2-10% (barriers ~96/window) |
| 5 | Software prefetch of next expert chunk (`prefetcht0`/`prefetchnta` ~2-4 KiB ahead, plus touch the next expert's first lines at the start of the current expert) | Hides prefetcher ramp-up at scattered 1.37 MiB boundaries | Exact | Low | 0-10% (unproven; measure) |
| 6 | Memory tuning: DDR4-3600/3800 at 1:1 FCLK, tighter timings | Raises the bandwidth ceiling | Exact | Zero code | Up to ~1.1-1.19x CPU part (proportional to MT/s) |
| 7 | Exact CPU/GPU overlap: launch GPU-hit experts (and shared expert) before waiting on CPU; HIP graph with host nodes | Hide CPU time behind GPU-resident work in the same layer (KTransformers async scheduler, +1.23x from CUDA graph) | Exact | Medium | Hides only GPU-hit expert time (~fraction of a ms per layer); 5-15% |
| 8 | Dynamic per-layer CPU/GPU split (HybriMoE/Fiddler): when the CPU queue > PCIe transfer time, send some misses to the GPU | Balances the makespan | Exact | Medium-high | Small at PCIe 3.0 (~0.1 ms per 1.37 MiB expert); 0-10% |
| 9 | Better GPU cache policy/prefetch (score-based, impact-driven, draft-guided like SP-MoE/MoE-SpeQ) | Raises the 80-93% hit rate, so fewer CPU bytes | Exact | Medium | Each +1 pt hit rate ≈ -5-8% CPU work at 80-93% hits; potentially largest exact lever |
| 10 | Expert Deferral (defer CPU-missed experts to layer k+2) | Breaks the dependency; CPU overlaps next-layer GPU work | **Lossy** (≤0.5% avg accuracy drop in KTransformers; breaks bit-exact verify) | Medium (residual bookkeeping across layers) | Up to ~1.3-1.45x end-to-end decode (paper: +33% to 1.45x) |
| 11 | Lower-bit CPU-resident copies of cold experts (e.g. IQ1_S/IQ1_M ~1.56-1.75 bpw) | ~25-30% fewer bytes per CPU expert | **Lossy** | Medium (second arena, ~requant) | ~1.3x CPU part; quality risk at 2-bit to 1.5-bit |
| 12 | Skip/approximate lowest-weight missed experts (adaptive top-k for CPU misses) | Fewer CPU experts | **Lossy** | Low | Proportional to skipped share; quality risk |
| 13 | LUT kernels (T-MAC/Vec-LUT) | `vpshufb` lookup instead of decode/multiply | **Lossy** (format change from IQ2_XS) | High | ≤ item 1's gain; not recommended |

- **Priority order for exact gains:** (1) check whether the current CPU kernel repeats decode per row; if yes, item 1. Then items 3+4 (barriers/imbalance), item 9 (hit rate), item 6 (memory OC), then items 2/5/7. Combined exact upside on the CPU part is plausibly 1.3-1.8x. That is bounded by achievable DRAM read bandwidth (~40-47 GB/s if the unsourced mid-40s figure holds) versus the current ~33 GB/s, plus the reduction in missed bytes.
- **Biggest lossy lever:** Expert Deferral. It turns the CPU stall into overlap and fits a "GPU waits 5 ms on CPU" profile, but it gives up the engine's greedy-identical guarantee. Offer it as an opt-in.

### Gaps
- No published system for "lower-bit CPU-resident duplicate of cold experts" in the single-GPU consumer setting was found within budget.
- No source quantifies how IQ1_S vs IQ2_XS expert quality differs for a 512-expert top-10 model.
- Achievable 5700G DDR4-3200 read bandwidth remains unsourced. It should be measured locally once benchmarks are idle.
