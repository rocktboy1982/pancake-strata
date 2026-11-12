# CPU and main-memory path for memory-bandwidth-bound inference (Nov 2022 - Nov 2026), judged for pancake

Scope: techniques on the CPU/DRAM side of bandwidth-bound decode (GEMV/low-bit dot products, prefetch, TLB/huge pages, compression, thread counts, barriers, low-bit SIMD, activation sparsity, CPU/GPU overlap, frequency, arena fill). Platform-independent verdicts; platform specifics (Zen 3 / DDR4 / 5700G) are deferred to a later round and only flagged.

Pancake baseline used for every verdict (from the brief and the two reports): edit window 37.4 ms, CPU experts 17.4 ms; 357 experts/window (7.44/layer x 48), ~49 us each for ~1.3-1.4 MB, i.e. ~27 GB/s against 33 GB/s measured (~80%); thread counts 6-12 give the same result; global barrier between gate/up and down (item 29, another session); 33 GiB expert arena filled through the page cache, oomd risk.

Already covered in /home/dan/reports and not repeated here except as cross-reference: T-MAC (USTC/MSRA, Asia report, verdict "only if the Q2_0 micro-benchmark shows compute-bound"), per-expert barriers (Tsinghua OSDI'26, item 29 / Asia P4), KTransformers deferral (Tsinghua SOSP'25, item 8), Fiddler, HybriMoE, MoE-Lightning, PowerInfer, TEAL/CATS/CHESS (item 13), STRATA_POOL_PREREAD and L3 prefetch (tested, weaker), MADV_SEQUENTIAL, hugepages (16,920 pages, owner decision), DDR4-3600/FCLK (owner decision).

## Q1. How close is pancake to the DRAM ceiling, and what does the literature say about low-bit GEMV kernels (VNNI-style vs LUT vs bit-serial)?

### Takeaway
The best 2025 x86 result for 2-bit GEMV (Intel, "up-convert to int8 + VNNI FMA", AVX2 class client CPUs) lands within 10-20% of the measured-bandwidth roofline, which is exactly where pancake's Q2_0 kernel already sits (~80% of 33 GB/s), with the same design. LUT methods (T-MAC, Vec-LUT) win on edge/ARM and on 1-bit, not on 2-bit x86 decode; there is no kernel-arithmetic lever of size left, only the last ~20% of bandwidth plus fixed overheads.

### Cited Findings
- "Pushing the Envelope of LLM Inference on AI-PC" (Georganas, Kalamkar, Heinecke; **Intel, industry-only, no university**; arXiv 2508.06753, Aug 2025): weights stored at 1/2 bits are expanded to int8 in registers and multiplied with int8 activations via VNNI FMA; the authors argue this beats LUT approaches; 2-bit uses an offline "VNNI4-interleaved" layout so unpack is a few shuffles/shifts/ANDs; prefetching is not discussed — [arXiv 2508.06753](https://arxiv.org/html/2508.06753v1)
- Same paper, numbers: on Core Ultra 9 285K (98 GB/s measured) 2-bit GEMV is within 10-20% of the int8 roofline and within 2-5% of int8 GEMV on large matrices; on Core Ultra 7 255H (75 GB/s) within 10-20%; on Lunar Lake 258V within 20-25% for large shapes, while 1-bit is capped at ~30 GB/s because unpack+compute becomes the bottleneck; 2-bit model up to 2.2x faster than bitnet.cpp; code: libxsmm/tpp-pytorch-extension branch `2bit_gemm` — [arXiv 2508.06753](https://arxiv.org/html/2508.06753v1)
- Vec-LUT (Li, Yin, Wang, Wei, Cao, Liu; MobiSys 2026 per arXiv comments; affiliations not shown on the abstract page — Ting Cao / Yunxin Liu are associated with MSRA/Tsinghua AIR in other work, **unverified here**): scalar LUT kernels build one table per token and access memory non-contiguously, under-using bandwidth in parallel (multi-token) workloads; Vec-LUT builds one LUT shared across parallel tokens with a single 1->N lookup per index, plus an LUT-centric tensor layout and cache-aware streamed lookup; up to 4.2x over SOTA baselines on 5 edge devices and 3 LLMs; code github.com/OpenBitSys/vlut.cpp, integrated into llama.cpp — [arXiv 2512.06443v2](https://arxiv.org/abs/2512.06443v2)
- A secondary overview claims scalar LUT methods often use <40% of available bandwidth in parallel workloads; not found in the paper itself — [search summary of alphaxiv overview](https://www.alphaxiv.org/overview/2512.06443)
- Arm group-wise low-bit GEMV/GEMM kernels (arXiv 2501.00032): batch-1 decode is memory-bound at high thread counts and compute-bound at low thread counts — [arXiv 2501.00032](https://arxiv.org/pdf/2501.00032)
- Repo check (read-only): pancake's Q2_0 CPU row kernel (`src/kernels/cpu/q2_avx2.cpp`, `q2_avx2_rows.inl`) is the up-convert design: `unpack64` turns 16 bytes of 2-bit codes into two 32-byte vectors with AND/shift/unpack, then `dot4` = `vpdpbusd` (AVX-VNNI build, called only where `cpu_avxvnni_ok()`) or `maddubs`+`madd` fallback; `row_multi<NT>` reads each weight row once for up to 4 verify tokens (NT 1..4; >4 is chunked by 4, re-reading a 720-byte row from L1) — [q2_avx2.cpp](../../../../../src/kernels/cpu/q2_avx2.cpp), [q2_avx2_rows.inl](../../../../../src/kernels/cpu/q2_avx2_rows.inl)
- Repo check: the Q2_0 row kernel has **no software prefetch**, while the IQ AVX2 kernel does (`_mm_prefetch(blk + pf, _MM_HINT_T0)` at iq_avx2.cpp:201-202) — [iq_avx2.cpp](../../../../../src/kernels/cpu/iq_avx2.cpp)

### Inferences
- Pancake's ~80% of measured bandwidth matches the Intel paper's 80-90% band for the same kernel family; this independently confirms the reports' claim that kernel arithmetic is not the lever. Expected gain from any kernel rewrite (LUT, bit-serial, Vec-LUT): ~0 on the 2-bit path. Family: F1 subgroup 3 ("cost per byte" / format kernels), alongside the existing T-MAC row; verdict "do not build".
- Vec-LUT's insight (amortize over parallel tokens) is already obtained in pancake by `row_multi<NT>` (one weight read per verify window). Nothing to port.
- Platform flag for the later round: Zen 3 has no AVX-VNNI, so on the 5700G the kernel runs the `maddubs+madd` fallback (2 uops more per 32 bytes). The Intel paper's 1-bit result (unpack-bound at ~30 GB/s on a low-core chip) shows unpack cost can cap bandwidth on weaker cores; the M6 micro-benchmark from the Asia report (single-thread GB/s of the Q2_0 kernel from L2-resident vs DRAM data) answers whether this matters at 6 threads.

### Gaps
- No paper measures 2-bit GEMV on AMD Zen 3/DDR4 desktop; the 10-20% band is Intel Arrow/Lunar Lake only.
- Vec-LUT affiliations and per-bit-width numbers were not visible on the abstract page.

## Q2. Prefetch (hardware vs software), cache blocking, non-temporal accesses: anything to win in the last ~20%?

### Takeaway
The only plausible exact micro-lever is a software prefetch in the Q2_0 row kernel that runs ahead across 4 KB boundaries, because hardware L1/L2 streamers restart at every 4 KB page and each pool task covers only tens of KB; expected 0-5% of CPU-expert time, a few lines of code, risk very low. Non-temporal loads/stores and extra cache blocking do not apply (outputs are tiny, weights are read once per window, activations already stay in L1).

### Cited Findings
- Intel IA-32 optimization manuals: the hardware prefetcher does not cross 4 KB page boundaries; demand loads on the new page are needed before it resumes; the L1 DCU prefetcher is limited to the same 4 KB page as the triggering load — [Intel manual page](https://www.manualsdir.com/manuals/127794/intel-architecture-ia-32.html?page=309); [Intel community quoting the optimization manual](https://community.intel.com/t5/Intel-Moderncode-for-Parallel/Hardware-prefetch-and-shared-multi-core-resources-on-Xeon/m-p/1074005)
- A prefetcher patent describes the restart cost at the next 4 KB region as possibly hundreds of cycles — [USPTO 9430392](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9430392)
- McCalpin (TACC/UT Austin) single-thread read-bandwidth study: the memory-controller prefetcher appears to track one stream per 4 KiB region, and interleaving fetches across several 4 KB pages can raise outstanding prefetches; explicit software prefetch was then tested on top — [McCalpin blog, UT Austin](https://sites.utexas.edu/jdm4372/2010/11/09/optimizing-amd-opteron-memory-bandwidth-part-4-single-thread-read-only) (old hardware; mechanism only)
- Gaze (arXiv 2412.05211) treats 4 KB regions as the standard spatial-prefetch granularity in current designs — [arXiv 2412.05211](https://arxiv.org/html/2412.05211v1)
- Repo: `STRATA_POOL_PREREAD` prefetches predicted next-layer experts with `_MM_HINT_T1` in 64 KiB chunks from parked workers; the grouping report lists L3 prefetch / pre-read as tested and weaker — [pool.cpp:994-1030](../../../../../src/kernels/cpu/pool.cpp); [Grouped technologies](../../../engine-speed/grouped-technologies/report.md)

### Inferences
- Geometry (computed from the code, not measured): a Q2_0 gate/up row is 40 blocks x 18 B = 720 B (n_embd 2560), a down row is 10 x 18 = 180 B (n_ff 640); each pool task is a contiguous row range, so a thread streams a few tens of KB per task and crosses a 4 KB boundary every ~6 gate/up rows. With 2 MiB pages the TLB walk disappears but the streamer restart (a physical-4 KB rule) remains. A `_mm_prefetch(row + D, _MM_HINT_T0)` with D ~ 1-2 KB (env knob, as iq_avx2 already does) is the textbook fix. Expected: 0-5% of the 17.4 ms (0-0.9 ms/window), exact, effort ~1 hour + A/B. Risk: none for correctness; could be neutral (the 20% residual may be DRAM page/bank effects, not prefetch). Family F1 subgroup 3 (cost per byte). This is distinct from POOL_PREREAD (next-layer, predicted, speculative) — it is in-stream, certain, same task.
- Non-temporal: NT stores only matter for large outputs (here: a few KB per expert); NT loads (`movntdqa`) only bypass caches on write-combining memory, not on normal WB arena memory; `prefetchnta` would limit L3 pollution, but per-window expert traffic (~500 MB) has no cross-window L3 reuse anyway. Verdict: no gain, skip.
- Cache blocking: weights are read once per window per expert and activations (int8, 2.5 KB/token) stay in L1; no blocking lever left.

### Gaps
- No paper isolates software vs hardware prefetch for low-bit GEMV on any CPU; the 0-5% figure is an engineering estimate from mechanism.

## Q3. Huge pages / THP / TLB effects and the arena fill (page cache double-buffering, MADV_* hints)

### Takeaway
TLB: pancake already does the right thing (hugetlb 2 MiB, else 2 MiB-aligned anonymous mapping with conditional MADV_HUGEPAGE, guarded against compaction stalls); the literature says huge pages help streaming but only modestly. The real CPU/RAM-path defect is the **arena fill**: buffered `fread` through the page cache plus a bounce-buffer `memcpy` means the 42.9 GB pack is cached while the 33 GiB anonymous arena fills, which is the most likely mechanism behind the oomd kills and also starves THP of free 2 MiB blocks. Direct I/O (or per-chunk `POSIX_FADV_DONTNEED`) is an exact, contained fix, with ServerlessLLM (OSDI'24) as the academic precedent.

### Cited Findings
- SLIDE (Rice, lead; MLSys 2020 — outside the date window, background only): huge pages cut data-TLB page-walk cycles from 7.74% to 0.72% of CPU cycles on a CPU DL workload — [arXiv 1903.03129](https://arxiv.org/pdf/1903.03129)
- vAttention (Microsoft Research India; ASPLOS'25): smaller pages cause TLB thrashing in many applications, but not for LLM inference (GPU context) — [arXiv 2405.04437](https://arxiv.org/pdf/2405.04437)
- Confidential LLM inference on CPU/GPU TEEs (arXiv 2509.18886): huge pages matter; TDX not using 1 GB pages adds TLB accesses; overheads vs a VM with 2 MB THP are 4-10% — [arXiv 2509.18886](https://arxiv.org/pdf/2509.18886)
- Blink (arXiv 2604.07609, 2026): a co-running interferer raises TLB misses only ~1.6x but makes each miss far costlier, driven by the interferer's frequent memory-management calls such as madvise — [arXiv 2604.07609](https://arxiv.org/pdf/2604.07609)
- HugeGPT thesis (Zhang 2024): huge pages help little for weak-locality (random) access — [thesis PDF](https://tianyin.github.io/thesis/ZHANG-THESIS-2024.pdf)
- ServerlessLLM (University of Edinburgh lead; OSDI'24): loader reads checkpoints with direct I/O into a pinned-memory chunk pool, multiple I/O threads per tier and a pipeline across tiers; pinned memory ~1.4x and pipelining a further ~1.5x loading throughput, saturating storage bandwidth; 3.6-8.2x faster cold start than PyTorch/Safetensors — [arXiv 2401.14351](https://arxiv.org/pdf/2401.14351)
- Repo: arena allocation in `core/pinned.cu` tries `MAP_HUGETLB|MAP_HUGE_2MB`, else a 2 MiB-aligned anonymous mapping; MADV_HUGEPAGE is skipped when THP is "always" (#771: a 25 s start became 432 s with compaction) and when free 2 MiB blocks are < half the arena (#2126: 148K failed compactions, 100% system time) — [pinned.cu:432-512](../../../../../src/core/pinned.cu)
- Repo: `load_experts_ranges` fills the arena with per-thread buffered `fopen`/`fread` into a `chunk` buffer then `memcpy` into the arena (comment: "Buffered, not FILE_FLAG_NO_BUFFERING: the cache should still hold what it can"); no `posix_fadvise(DONTNEED)` and no O_DIRECT on this path — [pinned.cu:1173-1260](../../../../../src/core/pinned.cu)
- Repo: the same engine already opens the pack with `O_RDONLY|O_DIRECT` for the file tier probe (#3237) and drops probed ranges with `MADV_DONTNEED` + `POSIX_FADV_DONTNEED`; `conversation_file.cpp` already has an O_DIRECT path with buffered fallback when a filesystem refuses it — [expert_source.cpp:797-897](../../../../../src/core/expert_source.cpp); [conversation_file.cpp:79,443](../../../../../src/core/conversation_file.cpp)
- Pain-points report: arena read 28-65 s for 33.02 GiB at 0.51-1.18 GiB/s, 2.3x variance on identical files attributed to page-cache state; arena build often triggers systemd-oomd — [Pancake pain points](../../../engine-speed/pain-points/report.md)

### Inferences
- Arena fill fix (new, not in the series): read each layer range with O_DIRECT (aligned) straight into the arena (dropping the bounce `memcpy`), or keep buffered reads but issue `posix_fadvise(fd, off, n, POSIX_FADV_DONTNEED)` after each chunk. Expected: page-cache footprint during load drops from up to ~33 GB to ~threads x chunk; removes the double-buffering that pushes the 57 GB box into oomd; makes load time independent of prior cache state (removes the 2.3x variance); frees 2 MiB blocks so THP backing actually succeeds. Effort: small (pattern and fallback already exist in the codebase); risk low (keep buffered fallback if ntfs3 refuses O_DIRECT for a given transfer — the #3237 probe suggests it accepts it, but that is inferred, not confirmed). Family: F9/F10 "harness and infra" row with `MADV_SEQUENTIAL`/hugepages/RAM_KEEP (item 9 cluster); it also unblocks the arena-vs-mmap RAM policy conflict in the grouping report. Zero decode-speed effect, but it is the CPU/RAM-path item with the best value/effort for the oomd problem.
- Optional refinement: prefault the arena with `MADV_POPULATE_WRITE` (Linux >= 5.14) right after MADV_HUGEPAGE, before reads, so faults happen in-kernel in one batch rather than inside the memcpy (engineering idea; no paper, unmeasured).
- TLB at decode: with 2 MiB pages each ~1.4 MB expert spans 1-2 pages (vs ~340 4 KB pages); TLB walks per expert are already negligible if the arena log reports THP/hugetlb. Verify the startup note ("hugetlb 2 MB pages" / "transparent huge pages requested" / "NOT requested ... #2126") in the pancake log before any hugepage work: if it says NOT requested, the arena fill fix above is the cheapest way to get THP back. Expected decode gain from huge pages vs 4 KB: low single digits at most, based on SLIDE/TDX magnitudes; not measured locally.

### Gaps
- No controlled study of THP vs 4 KB pages for CPU LLM decode throughput was found.
- Whether ntfs3 accepts O_DIRECT for large sequential reads on this pack is inferred from the repo probe, not verified.

## Q4. Bandwidth-aware thread counts, barrier/sync costs, NUMA-agnostic desktop tuning

### Takeaway
Literature agrees that decode GEMV saturates DRAM with a few cores and extra threads past saturation do not help and can hurt; pancake's "6-12 threads equal" is the expected signature. The useful consequence is to run the pool at the smallest saturating count (fewer barrier participants, more headroom for the host/GPU-driver threads and less shared-machine noise), not to look for more threads. Barrier work is item 29 (owned elsewhere); NUMA is irrelevant on a single socket.

### Cited Findings
- Arm "llama-roofline" learning path (industry guide): decode speed is bounded by memory bandwidth / bytes per token; adding threads past memory saturation often makes decode slower — [Arm learning path](https://learn.arm.com/learning-paths/cross-platform/llm-decode-roofline/how-decode-uses-memory/)
- ArcLight (arXiv 2603.07770, 2026): decode throughput rises from 6 to 48 threads on a many-core multi-NUMA server, attributed to intra-node bandwidth and cross-NUMA tensor parallelism (server, not desktop) — [arXiv 2603.07770](https://arxiv.org/pdf/2603.07770)
- ProfInfer (Zou, ..., Haibo Chen; MLSys 2026; SJTU affiliation likely given the senior author but not shown on the abstract page): search snippet says decode matmuls stop scaling after ~2 threads on the evaluated device as they hit the bandwidth ceiling; I could not confirm this on the abstract page — [arXiv 2601.20755](https://arxiv.org/abs/2601.20755) (claim unverified)
- KTransformers slides (Tsinghua) list memory bandwidth and thread synchronization as the reasons MoE layers underperform in decode — [ADSL slides](https://adsl-rg.github.io/slides/251223-KTransformer.pdf); [SOSP'25 paper](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- Repo: the pool already places one worker per physical core (comment: two workers per core would "halve the bandwidth the expert kernel is bound by"), pins the host to its own core, has `STRATA_POOL_FUSE_PHASES` (mode 7: a down task spins on `fready_[e]` for its own expert only — the item-29 implementation), `STRATA_POOL_SPIN_US`, and an experimental `STRATA_POOL_UMWAIT` (MSVC-only) — [pool.cpp](../../../../../src/kernels/cpu/pool.cpp)

### Inferences
- With 357 experts x ~1.4 MB = ~500 MB per window split over 48 layers x 2 phases, each phase is ~5 MB, i.e. ~0.19 ms at 27 GB/s; with 12 threads each thread handles ~0.4 MB per phase. Fixed per-phase costs (wake, tail imbalance, barrier) of even 5-10 us per phase are 2.5-5% of phase time; halving participants cuts the tail of a straggler and the cache-line traffic on the epoch/done counters. Action: A/B 6 vs 8 vs 12 workers with FUSE_PHASES on, look at window time variance, not just mean. Expected 0-3%, exact, effort trivial (config), risk none. Family F3 (synchronization subrow, with item 29).
- Desktop NUMA: single socket, one memory controller domain; NUMA split code paths should stay off (they are already gated). No literature item applies.

### Gaps
- No desktop-class (single-socket, 2-channel DDR4/DDR5) study of thread count vs bandwidth for MoE expert GEMV was found; server and mobile results bracket it.

## Q5. Cutting bytes: memory compression (in-DRAM or software) and activation-sparse / weight-sparse CPU kernels

### Takeaway
Hardware compression (DECA near-core decompressor, in-DRAM/HBM dequant) is not available on a desktop. Software lossless entropy coding (EntroLLM, rANS proposals) saves real bytes at 4-8 bits but has little headroom at 2.25 bpw and its decode cost on CPU cores would exceed the bandwidth saved. Weight-sparse AMX kernels (SparAMX) need AMX and unstructured-sparse weights. Activation sparsity remains item 13 (TEAL/CATS/CHESS), already covered.

### Cited Findings
- DECA (Intel with UIUC; arXiv 2505.19349, 2025): on Sapphire Rapids, software decompression of compressed weights is a bottleneck that worsens with higher bandwidth; DECA is a near-core accelerator sharing L2 and L2 TLB, with a new ISA extension (TEPL) for out-of-order invocation; reported 1.6-2.6x next-token time reduction on Llama2-70B / OPT-66B (figure seen via a third-party summary) — [arXiv 2505.19349](https://arxiv.org/pdf/2505.19349); [newx.sg summary](https://newx.sg/paper/detail/74af5a5b-f823-11f0-b84c-00163e10baa7)
- EntroLLM (arXiv 2505.02380, 2025): Huffman coding of quantized weights, parallel decode, no retraining; up to 30% storage savings over uint8 and 65% over uint4, 31.9-146.6% faster inference on memory-limited edge devices (Jetson) — [arXiv 2505.02380](https://arxiv.org/html/2505.02380v4)
- MLX feature request (not a paper): measured entropy ~2.17 bits for 4-bit and ~4.5 bits for 8-bit quantized weights; proposes per-row rANS with fused decode+GEMV — [MLX issue 3043](https://upd.dev/ml-explore/mlx/issues/3043)
- Huff-LLM (arXiv 2502.00922): Huffman-coded weights with a single-cycle hardware decoder, 15-32% less on-chip memory and bandwidth (custom accelerators) — [arXiv 2502.00922](https://export.arxiv.org/pdf/2502.00922)
- SparAMX (Intel Labs lead; arXiv 2502.12444, 2025): unstructured weight sparsity on AMX CPUs for decode; up to 1.42x end-to-end vs stock PyTorch, INT8 kernel up to 1.46x vs a Neural Magic kernel, 1.14x on attention; Llama 3 8B on Xeon Gold 6430L; code in IntelLabs Hardware-Aware-Automated-Machine-Learning — [arXiv 2502.12444](https://arxiv.org/html/2502.12444v1)
- StreamDQ (arXiv 2607.08993) and TriMoE (arXiv 2603.01058) put dequantization / cold experts into HBM or DIMM near-memory processing — [StreamDQ](https://arxiv.org/pdf/2607.08993); [TriMoE](https://arxiv.org/pdf/2603.01058)

### Inferences
- Entropy coding for Q2_0: the codes are already 2 bits (18 B per 64 weights = 2.25 bpw incl. FP16 scale); the 4-bit -> ~2.17-bit entropy observation suggests quantized weights are far from uniform, but a 4-symbol alphabet bounds the saving at <2 bits/code and realistic Gaussian-like code histograms leave perhaps 5-15% (estimate, unmeasured). Huffman/rANS decode on general cores at ~27 GB/s-equivalent weight throughput across 6-12 cores is unlikely (the DECA paper's whole point is that software decompression already bottlenecks at lower ratios). Verdict: do not build; at most an offline histogram of Q2_0 codes per expert to put a number on the ceiling (1 hour, zero risk). Family F1 subgroup 3.
- SparAMX: requires AMX and unstructured weight sparsity on top of quantization; not applicable to Q2_0 on a desktop. Skip.
- In-DRAM / near-memory compression: hardware not present; skip.

### Gaps
- No measured entropy of Q2_0 codes for this model; no CPU software-decompression study at 2-bit.

## Q6. Overlap CPU compute with GPU work, and power/frequency effects on bandwidth

### Takeaway
Overlap is fully covered by item 8 (KTransformers deferral) and its HIP non-finite bug; no new CPU-side overlap paper changes that. On power/frequency, academic work is mobile (memory governors) or GPU DVFS; nothing studies desktop CPU uncore/FCLK for decode, so the DDR4-3600/FCLK question stays an owner-measured experiment.

### Cited Findings
- Mobile DVFS study (arXiv 2507.02135; MLSys 2026 supplement): with other frequencies pinned, the default memory governor is near-optimal for decode, while the CPU governor picks too-low CPU frequencies because CPU utilization is low during GPU-heavy inference — [arXiv 2507.02135](https://arxiv.org/pdf/2507.02135)
- CORE (MLSys 2026): a governor coordinating CPU, GPU and memory frequencies for mobile LLM inference — [MLSys 2026](https://proceedings.mlsys.org/paper_files/paper/2026/hash/136b9a13861308c8948cd308ccd02658-Abstract-Conference.html)
- BiScale/DualScale (arXiv 2602.18755): GPU-side per-iteration DVFS treating decode as bandwidth-bound — [arXiv 2602.18755](https://arxiv.org/html/2602.18755v2)
- Repo: `STRATA_POOL_UMWAIT` is documented as giving parked-worker spin power back to the GPU's share of the package budget on laptops (MSVC x64 only for now) — [pool.cpp:42-73](../../../../../src/kernels/cpu/pool.cpp)

### Inferences
- The mobile finding "CPU governor under-clocks during GPU-heavy inference" has a desktop analogue worth one check: if the cpufreq governor is `schedutil`/`powersave`, the pool's short bursts (~0.2 ms phases) may start at low clocks. With a memory-bound kernel the core clock matters little, but wake latency and the unpack path (Q1, no VNNI on Zen 3) do. Action: A/B `performance` governor vs current, window-time mean and variance. Expected 0-3%, exact, effort trivial, risk none (system setting -> owner). Family F9/F10 (measurement hygiene) and F3.
- The repo's UMWAIT parking is Windows-only; on Linux the parked workers spin with `_mm_pause`. On a desktop dGPU system the CPU and GPU do not share a power budget, so porting UMWAIT has no expected decode gain here. Skip.

### Gaps
- No academic source on desktop CPU uncore / Infinity Fabric frequency vs DRAM bandwidth for LLM decode; FCLK/DDR4-3600 remains unmeasured.

## Summary table for the report writer

| Technique | Paper / institution (lead) | Venue/date | Applies to pancake? | Where | Expected gain | Effort | Risk | Family |
|---|---|---|---|---|---|---|---|---|
| Up-convert + VNNI 2-bit GEMV (confirms ceiling) | AI-PC, Intel (industry) | arXiv Aug 2025 | Already implemented | q2_avx2 | 0 (validation: pancake at the paper's 80-90% band) | 0 | - | F1/3 |
| Vec-LUT shared LUT over tokens | Li et al. (affil. unverified) | MobiSys 2026 | No (row_multi<NT> already amortizes) | - | ~0 | - | - | F1/3 |
| T-MAC / LUT GEMV | USTC + MSRA | EuroSys'25 | Already judged in Asia report | - | ~0 on 2-bit x86 | - | - | F1/3 |
| In-stream software prefetch across 4 KB pages in Q2_0 rows | Intel manuals; McCalpin (UT Austin) | mechanism | Yes, new | q2_avx2_rows.inl `row_multi` | 0-5% of CPU-expert time (0-0.9 ms/edit window) | ~1 h + A/B | very low | F1/3 |
| Non-temporal loads/stores, extra blocking | - | - | No | - | 0 | - | - | - |
| Arena fill with O_DIRECT / FADV_DONTNEED (no page-cache double copy) | ServerlessLLM, Edinburgh | OSDI'24 | Yes, new | pinned.cu `load_experts_ranges` | removes ~33 GB transient page cache, oomd trigger, 2.3x load variance; helps THP succeed | small | low (buffered fallback) | F9/F10 (item 9 cluster), RAM policy conflict |
| MADV_POPULATE_WRITE prefault | engineering | - | Maybe | pinned.cu | load-time only, unmeasured | small | low | F9/F10 |
| Huge pages / THP | SLIDE (Rice), vAttention (MSR India), TDX study | 2020-2025 | Already implemented; verify log note | pinned.cu | low single digits at most | 0 | - | F9/F10 |
| Smallest saturating thread count | Arm guide; ArcLight; ProfInfer (unverified) | 2025-2026 | Yes (config) | pool workers | 0-3%, lower variance | trivial | none | F3 |
| Per-expert barriers | Tsinghua | OSDI'26 | Item 29, other session | pool mode 7 | 1-4% | coded | low | F3 |
| Entropy-coded weights | EntroLLM; Huff-LLM | arXiv 2025 | No (2.25 bpw headroom small, CPU decode too slow) | - | <0 net | - | - | F1/3 |
| Near-core / near-memory decompression | DECA (Intel+UIUC), StreamDQ, TriMoE | 2025-2026 | No (hardware) | - | - | - | - | - |
| Weight-sparse AMX | SparAMX, Intel Labs | arXiv 2025 | No (AMX, unstructured sparsity) | - | - | - | - | F1/1 |
| CPU governor = performance | mobile DVFS study; CORE | MLSys 2026 | Yes (owner setting) | system | 0-3%, lower variance | trivial | none | F9/F10, F3 |
