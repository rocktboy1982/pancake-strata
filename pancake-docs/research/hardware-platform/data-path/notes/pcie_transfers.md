# PCIe / host<->device transfer path for a single-GPU MoE-streaming engine (pancake), university and systems work Nov 2022 - Nov 2026

Scope note. Techniques are judged on their own merits, platform-independent. AM4/AM5/Intel specifics are a later round. Already covered in earlier reports and not repeated here, only cross-referenced: `--pcie-frac 0` and the "compute misses on CPU, copy only for reuse" rule (Fiddler, KTransformers, HybriMoE), ProMoE's chunked, cancellable prefetch planner, predictors, admission at the k-th miss, SLP and MoE-Gen prefill streaming, ZipNN's "GGUF is incompressible", DFloat11, and the generic "zero-copy, ReBAR and SDMA cannot beat the link" verdict (see `pancake-docs/research/engine-speed/moe-speedup/notes/transfer_and_kv.md` in the repo and the F2/F3 families in `Grouped technologies for pancake.md`). This round adds new sources and, more importantly, a code-level reading of where pancake's transfer path actually sits on the critical path today.

Budget note. About 17 web calls plus read-only greps of `/home/dan/clone-of-d/src/Strata` and read-only sysfs. No builds, no GPU runs. All gain figures marked "estimate" are my arithmetic from cited numbers and local profile figures. They are not measurements.

## Q0. What does pancake's transfer path already do, and what is the hardware path? (code and sysfs reading)

### Takeaway
Pancake already has every classic transfer mechanism: a pinned (registered) arena, a separate copy stream, DMA expert uploads, zero-copy mapped aliases, a copy-kernel staging mode, a background promotion-fill thread, and doorbells in mapped host memory. The link is PCIe 3.0 x16 at the CPU root port, Resizable BAR is active, and the IOMMU is in passthrough. The transfer that sits on the critical path every layer is not an expert upload, because those are off at `--pcie-frac 0`. It is the GPU *pulling* the CPU's fp32 result rows out of host-mapped memory inside `combine_cpu_k10_vec4`, plus the PCIe polling that precedes the pull.

### Cited Findings
- PCIe topology (local sysfs, 2026-11-09): root port `00:01.1` runs 8.0 GT/s x16 (max 8.0, Gen3). The card's internal switch (`10:00.0`, `11:00.0`) and the GPU `12:00.0` run 16 GT/s, so the effective host link is **Gen3 x16** — [local: /sys/bus/pci/devices/*/current_link_speed]
- Resizable BAR is active: GPU BAR0 is a **32 GB prefetchable window** (`Region 0: Memory at 7000000000 [size=32G]`), so the CPU can map all 20 GB of VRAM — [local: lspci -vv 12:00.0]
- The IOMMU (AMD-Vi `ivhd0`) is present, but the GPU's IOMMU group type is **`identity`** (passthrough). GPU DMA is not translated through IOVA page tables. THP is in `madvise` mode — [local: /sys/bus/pci/devices/0000:12:00.0/iommu_group/type; /sys/kernel/mm/transparent_hugepage/enabled]
- The expert arena is ordinary memory registered with `cudaHostRegister(..., Mapped|Portable)` (hipified), not `hipHostMalloc`. With `STRATA_NUMA_SPLIT` it is forced to 4 KiB pages (`MADV_NOHUGEPAGE`) — [src/core/expert_source.cpp:3900-3920, 3979, 4022; include/strata/core/expert_source.hpp:1253]
- The PCIe share has three modes. `pcie_mode 0` is DMA uploads on `copy_` (one `cudaMemcpyAsync` per blob), with flag B raised by an ordered host function. `pcie_mode 1` points the grouped GEMV at a mapped device alias of the host blob (zero-copy). `pcie_mode 2` stages with a copy kernel (`fetch_blobs`) — [src/core/verify.cpp:4031-4057, 2585-2597; src/core/expert_source.cpp:5100-5164]
- `STRATA_DMA_BATCH` (a `cudaMemcpyBatchAsync` path) is CUDA 13+ only. "Every other build (HIP, CUDA before 13) ... takes the loop." The ROCm 10 header has no `hipMemcpyBatchAsync`, but it does have `hipStreamWaitValue32` / `hipStreamWriteValue32` and `hipDeviceMallocFinegrained` / `hipDeviceMallocUncached` — [include/strata/core/dma_batch.hpp:1-30; /opt/rocm/core-10.0/include/hip/hip_runtime_api.h:964-970, 3326]
- `STRATA_COPY_HIP_PRIO=1` (opt-in) gives the copy stream a high-priority HIP stream, so it gets its own hardware queue. Its comment notes that sharing a queue made DMA host functions queue behind the window graph — [src/core/verify.cpp:1262-1272]
- Promotions (Foresight swap / adaptive fill) run on a `strata-fill` thread: `cudaMemcpyAsync` per slot plus an event — [src/core/expert_source.cpp:6613-6635]. Staged RAM promotions (`ram_stage`, queue cap `STRATA_RAM_ADAPT_QUEUE`=64) — [src/core/expert_source.cpp:4488-4510]
- The CPU result rows `h_ymiss_` are a mapped, coherent host buffer of `T*K*N*4` bytes (`hipHostMallocMapped|Coherent`) — [src/core/verify.cpp:516-520, 1017]
- `combine_cpu_k10_vec4` reads, for every (token, expert) slot that is not a VRAM hit, a **full fp32 row of N floats from the mapped host buffer** (`p[e] = cpu4[o]`). Under `STRATA_WAIT_FUSE`, thread 0 of *every block* spins on the host flag (`while (*flag < value) strata_spin_pause();`) — [src/kernels/cuda/native_moe.cu:145-200]
- Profile: 38.3% of GPU kernel time is `combine_cpu_k10_vec4` plus 4.1% `wait_flag_copy_i32`. The edit window is 37.4 ms with 357 CPU experts per window (7.44 per layer) — [/home/dan/pancake-docs/research/engine-speed/pain-points/report.md]
- A promotion costs ~115 us per 1.4 MB on PCIe 3.0, against ~42 us CPU compute — [MOD_PLAN.md:496]

### Inferences
- 115 us measured against 1.437 MB / 13.4 GB/s = 107 us means the fixed per-copy overhead is ~8 us (~7%). Large-copy efficiency is already near the ceiling.
- **The 38.3% "spin" in `combine_cpu` is not all spin.** It includes the PCIe pull of the CPU rows after the flag flips. Estimate for an edit window (T≈5, ~2 non-hit slots per token, so ~10 rows of 10,240 B, ~100 KB per layer, possibly up to ~200 KB): at ~12 GB/s that is ~8-17 us of transfer per layer, plus one or two PCIe round trips, so **~0.4-0.8 ms per 37.4 ms window (~1-2%) on the critical path**. This needs measurement: run the combine kernel with the flag pre-set, or use the `stamp()` points, to split spin from pull.
- No published paper evaluates this exact pattern (GPU pulling CPU-computed MoE partial outputs from mapped memory). The levers in Q2 come from the general zero-copy and posted-write literature.

### Gaps
- No local measurement of the GPU's zero-copy read bandwidth from host memory on this Gen3 link, of the PCIe read round-trip latency, or of CPU write-combining bandwidth into the ReBAR window. A 30-minute micro-benchmark (no engine changes) would settle all three.
- Exact count of non-hit slots per token per layer on edit windows: derivable from the existing replay traces, but not done here.

## Q1. DMA engines, copy queues, batching small copies: how efficient are small vs large transfers, and does SDMA tuning help?

### Takeaway
University work in this window adds little on DMA engines themselves. The best 2024-2026 data is AMD-led (ConCCL, DMA-Latte). It shows SDMA control/schedule/sync overhead is up to ~60% of time at ~4 KB and below 20% only above ~1 MB. At 1.4 MB per expert, pancake's uploads are already in the efficient regime (~7% overhead measured). Batching or a better queue setup could save at most that ~7% on promotions, and HIP on ROCm 10 does not expose a batch API anyway. The small transfers that matter (flags, rows) are not done by DMA in pancake, and they should not be.

### Cited Findings
- DMA-Latte (Pati, Aga, Islam, Quach, Kudchadker, Ibrahim; **AMD, industry-led**; arXiv 2511.06605 v1 Nov 2025, v2 Apr 2026). The host CPU builds copy commands into a system-memory queue per SDMA engine and rings a doorbell. The engine fetches, decodes and translates the command, then moves the data. Completion is an atomic on a 64-bit signal the CPU waits on. Non-copy phases (control, schedule, sync) are **up to ~60% of a single copy at ~4 KB and fall below 20% only above ~1 MB**. MI300X has 16 sDMA engines per GPU — [arXiv 2511.06605v2](https://arxiv.org/html/2511.06605v2)
- Same paper: broadcast, swap, back-to-back and "prelaunch" (a DMA poll command, so the engine waits on a memory value and then copies) are "not exposed today across AMD GPU software stack". The runtime exposes only single-source, single-destination copies. The proposed vehicle is an extended `hipMemcpyBatchAsync`. Prelaunch speedups for latency-bound sizes are 1.2-1.9x. In vLLM CPU-KV fetch, the DMA path reached up to 1.3x the throughput of kernel-based fetch, while kernel-based TTFT was ~11% lower — [arXiv 2511.06605v2](https://arxiv.org/html/2511.06605v2)
- ConCCL (Agrawal, Aga, Pati, Islam; **AMD**; arXiv 2412.14335): offloading collectives from CUs to SDMA engines reduces interference with concurrent GEMMs. MI300X has 14 SDMA engines in that paper's description — [arXiv 2412.14335](https://arxiv.org/html/2412.14335v2) (search snippet; full text not read)
- pcie-bench (Neugebauer et al., **Cambridge**, SIGCOMM 2018; background, outside the window): the PCIe host implementation contributes 70-90% of latency on the NIC studied. Performance depends on temporal host state (IOTLB, caches), and IOMMU features have a measurable cost — [Cambridge pcie-bench project](https://www.cl.cam.ac.uk/research/srg/netos/projects/pcie-bench/); [slides](https://www.cl.cam.ac.uk/research/srg/netos/projects/pcie-bench/neugebauer2018understanding-slides.pdf)
- MoE-Infinity (Edinburgh) uses a dedicated thread with pinned memory and DMA. One thread saturates PCIe 4.0 — [arXiv 2401.14361](https://arxiv.org/pdf/2401.14361) (via search snippet)

### Inferences
- Applicability to pancake (family **F2/F3 plumbing**):
  - *Batching promotion copies* (one submission per window instead of ~50 `hipMemcpyAsync`): on HIP this needs either a future `hipMemcpyBatchAsync` or HSA-level `hsa_amd_memory_async_copy` with fewer signals. The bound is the ~8 us per-copy overhead, so ~7% of promotion time, and only if promotions are link-budget-limited. **Expected gain on decode: 0-1%. Effort medium, risk low. Verdict: skip unless a trace shows copy-issue gaps.**
  - *Contiguous layout for promotions*: if the tier picks several experts per layer, sorting the swap list by arena address and coalescing adjacent blobs into one copy would get the same ~7% for free. **Verdict: cheap, tiny.**
  - *`STRATA_COPY_HIP_PRIO=1` as default on gfx1100*: already coded. It matters only when DMA host functions are in the critical path (`pcie-frac > 0`). At `--pcie-frac 0` it affects only promotion latency. **Verdict: a one-line A/B, likely flat.**
  - *Replacing the host-function flag raise (`raise_flag_b_ordered`) with `hipStreamWriteValue32` on the copy stream*: this removes a host-callback hop (typically tens of us in HIP; not measured here) from the PCIe-share path. **Only relevant if `--pcie-frac` ever goes back above 0. Verdict: park.**
- DMA for KB-sized items (flags, rows) would be strictly worse than CPU stores or mapped reads, per the ~60% overhead at 4 KB. Pancake's design of mapped doorbells and rows is correct in principle, and Q2 is about tuning it.

### Gaps
- No measured SDMA overhead on RDNA3 consumer parts (RX 7900 XT has 2 SDMA engines per public specs; not verified in this pass). DMA-Latte numbers are MI300X over xGMI, not PCIe.
- No university-led paper on SDMA scheduling for inference on consumer GPUs was found.

## Q2. Zero-copy / host-mapped access, peer/BAR access and ReBAR: when does reading over PCIe directly beat copying, and what can pancake change?

### Takeaway
The literature agrees: zero-copy reaches ~90%+ of DMA bandwidth only with 128-byte-aligned, coalesced accesses (EMOGI, PyTorch-Direct), and it beats copy-then-compute for data used once (DeepPlan's direct-host-access layers). For pancake's misses, CPU compute (42 us) still beats any PCIe path (≥107 us), as earlier rounds concluded. The new, concrete applicability is **the reverse direction: the CPU result rows**. Two cheap changes are worth testing. (a) **Pre-reduce on the CPU**: send one weighted-sum row per token, optionally bf16, instead of up to 10 fp32 rows. (b) **Push instead of pull**: with ReBAR active, the CPU writes rows and the flag into fine-grained VRAM with posted writes, so the GPU spins and reads locally. Estimated **+0.5-2% on edit windows**, exact or near-exact. Small, but on the one transfer that is on the critical path at every layer.

### Cited Findings
- EMOGI (Min et al., **UIUC**-led with NVIDIA; PVLDB 14, 2021; background). Naive zero-copy cannot saturate PCIe 3.0 x16, which they measured with an FPGA PCIe monitor. Merging accesses into maximum-size requests and forcing 128-byte alignment of warps "nearly fully utilize the PCIe bandwidth". 2.60x average over optimized UVM, and the approach scales to A100/PCIe 4.0 — [arXiv 2006.06890](https://arxiv.org/pdf/2006.06890v1); [PVLDB](https://www.vldb.org/pvldb/vol14/p114-min.pdf)
- PyTorch-Direct / GPU-oriented data communication (Min et al., **UIUC**; arXiv 2103.03330; background): with alignment, zero-copy PCIe bandwidth reaches up to **93% of block-transfer bandwidth** — [arXiv 2103.03330](https://ar5iv.labs.arxiv.org/html/2103.03330) (via search snippet)
- DeepPlan (Jeong, Baek, Ahn; **Ajou University**, lead; EuroSys '23, May 2023). It combines direct-host-access (the GPU computes certain layers straight from host memory, without loading them) with parallel transmission over multiple GPUs, and picks per layer. It reports better latency than pipelined loading and stable p99 on BERT/RoBERTa/GPT-2. Code: Sys-KU/DeepPlan (fork at MachineLearningSystem/DeepPlan) — [paper PDF](https://jeongseob.github.io/assets/papers/jeong_eurosys23.pdf); [AMiner](https://www.aminer.org/pub/64634f5fd68f896efaba1473/fast-and-efficient-model-serving-using-multi-gpus-with-direct-host-access)
- AMD: kernels reading pinned host memory go over PCIe. Pinned allocations are coherent (not cached in L2) by default. `hipHostMallocNonCoherent` allows L2 caching — [ROCm GPU memory](https://rocm.docs.amd.com/en/docs-6.2.1/conceptual/gpu-memory.html) (as cited in the earlier transfer notes)
- Lustig & Martonosi (**Princeton**, HPCA 2013; background): fine-grained CPU-GPU synchronization with full/empty bits per transferred region cuts offload latency against coarse-grained completion — [Princeton listing](https://collaborate.princeton.edu/en/publications/reducing-gpu-offload-latency-via-fine-grained-cpu-gpu-synchroniza/)
- Local: ReBAR active (32 GB BAR0); `combine_cpu_k10_vec4` pulls fp32 rows per non-hit slot from mapped coherent host memory, with a per-block host-flag spin under WAIT_FUSE (see Q0) — [src/kernels/cuda/native_moe.cu:145-200]

### Inferences
Concrete changes for pancake, in pipeline order (family **F3**, overlap and latency hiding; all exact in algorithm):

1. **Measure first (zero code).** Time `combine_cpu_k10_vec4` with the flag already raised (replay one layer's rows) against total kernel time, to split spin from pull. If the pull is under ~3 us per layer, skip items 2-3.
2. **CPU-side pre-reduction of rows** (`expert_pool` writes `sum_{e in CPU} w_e * y_e` per token into one row; the combine reads one host row per token instead of one per non-hit slot). Bytes per layer drop from ~(non-hit slots) × 10 KB to T × 10 KB, or T × 5 KB in bf16, a 2-4x reduction on edit windows. CPU cost is one FMA pass over T×N floats, about nothing against ~48 us per expert. **Expected: −0.2 to −0.6 ms per edit window (~0.5-1.5%), estimate.** Effort small-medium (pool output path, combine kernel variant, and the PCIe-share/deferral paths that also write `h_ymiss_`). Risk: fp32 summation order changes, so results are not bit-identical to the current build. bf16 rows add rounding. Keep fp32 for an "exact" preset, and gate bf16 behind the quality gate.
3. **Push via ReBAR instead of pull.** Allocate the row buffer and the per-layer flag in device memory with `hipExtMallocWithFlags(..., hipDeviceMallocFinegrained)` (or uncached), map it to the CPU through the large BAR, and have pool workers write each finished row with write-combining stores, then `sfence`, then the flag store. The GPU spins on a VRAM flag (an L2/VRAM round trip, not a PCIe round trip) and reads rows from VRAM. Row transfer overlaps with the CPU's remaining experts instead of starting after the last one. **Expected: removes most of the ~0.4-0.8 ms pull plus a fraction of a microsecond of poll latency per layer, ~1-2% on edits (estimate), more if item 2 is skipped.** Effort medium. Risks: WC store ordering across 16 pool threads (needs a release fence per writer and a counter flag, not a single store), CPU write bandwidth to the BAR on Gen3 (unmeasured; if it is under ~6 GB/s, the push could lengthen the CPU phase), and a ROCm requirement that host access to fine-grained VRAM works on gfx1100 consumer parts (to verify).
4. **Spin hygiene under WAIT_FUSE.** Today every block's thread 0 polls a host flag over PCIe. That is ~10 blocks × T tokens of outstanding non-posted reads during the whole CPU phase. Each poll's completion uses downstream link bandwidth, the same direction as H2D promotions. *Hypothesis, unmeasured:* this steals some link capacity from the concurrent promotion DMA and adds snoop traffic on the CPU's memory fabric. Fix: one waiter polls the host flag and republishes it to a VRAM flag that the other blocks poll, or use item 3. **Expected: 0 to a few % on promotion throughput. Test by measuring promotion bandwidth with WAIT_FUSE on and off.**
5. **Zero-copy for misses (pcie_mode 1).** Unchanged verdict: a GEMV reading 1.4 MB over Gen3 costs ≥107 us, against 42 us on the CPU. DeepPlan's direct-host-access wins only where the alternative is copy-then-compute, never where a 2.5x-cheaper CPU path exists. **Keep `--pcie-frac 0`.** If the share is ever re-enabled (faster link), apply EMOGI's rule: the grouped GEMV must issue ≥128 B aligned, coalesced loads from the mapped alias (Q2_0 blocks are small; verify that the access pattern is not 32-64 B scattered).

### Gaps
- No paper measures CPU→VRAM posted-write (ReBAR) bandwidth or latency on consumer AMD; it needs a local probe.
- Whether `hipDeviceMallocFinegrained` VRAM is CPU-mappable through ReBAR on RDNA3 under ROCm 10 was not verified.
- The share of combine time that is pull and not spin is unknown until item 1 is measured. All gains in this section are estimates conditional on that number.

## Q3. Chunked/pipelined transfers, transfer/compute overlap, scheduling and cancellation: what do 2023-2026 systems add beyond what pancake has?

### Takeaway
The newest offloading systems (MoE-Lightning, Pie, Klotski, DALI) all converge on (1) paging or chunking weights so big transfers interleave with small critical ones, (2) a performance model or online feedback that sets how much to transfer without slowing foreground compute, and (3) window-based swap-in/swap-out decisions. Pancake already does (3), adapting every window with a swap budget. The transferable ideas are **Pie's online "performance-transparent" swap-budget controller** and **MoE-Lightning-style paging of promotions so they never collide with the critical row pull**. Both are exact. Expected value: they protect the +5-16% F2 admission gain from DMA contention, not new speedup on their own.

### Cited Findings
- MoE-Lightning (Cao et al., **UC Berkeley**-led; arXiv 2411.11217, Nov 2024; an ASPLOS'25 publication was not confirmed by my search). CGOPipe is a CPU-GPU-I/O pipeline schedule with **paged weights**: weights to transfer are chunked into n pages (n = number of micro-batches), so weight transfers interleave with the much smaller intermediate-result transfers instead of blocking them. A hierarchical roofline-style performance model picks batch, micro-batch and weight placement. Up to 10.3x throughput over offloading baselines on Mixtral 8x7B on one T4 (throughput-oriented, large batch) — [arXiv 2411.11217](https://arxiv.org/html/2411.11217v1)
- Pie (Xu, Mao, Mo, Liu, Stoica; **UC Berkeley**; arXiv 2411.09317, Nov 2024). "Performance-transparent swapping" runs transfers concurrently with foreground compute. "Adaptive expansion" sets the amount of CPU memory used online, so swapping never blocks GPU compute. Up to 1.9x throughput and 2x latency over vLLM, and 9.4x throughput over FlexGen. Evaluated on GH200 (NVLink-C2C, not PCIe) — [arXiv 2411.09317](https://arxiv.org/html/2411.09317)
- Klotski (Fang, Huang et al.; arXiv 2502.06888, Feb 2025; institution not confirmed in this pass). Expert-aware multi-batch pipeline: next-layer loading overlaps the current layer, with a constraint-sensitive I/O-compute planner and a correlation-aware expert prefetcher. Up to 85.12x throughput (baselines not verified) — [arXiv 2502.06888](https://arxiv.org/html/2502.06888v1)
- DALI (Zhu, Li, Wang, Mo, Pei, Song, Liang, Cheng; **CAS Institute of Automation and Shanghai Jiao Tong University**; arXiv 2602.03495, Feb 2026). A 0-1 assignment of experts to CPU or GPU solved greedily, with GPU cost = max(PCIe transfer, compute) and zero transfer for cached experts. Residual-based prefetching (next layer's gate on hidden state plus an offline residual) on a separate stream. **Workload-aware cache replacement that swaps the top-scoring CPU experts for the lowest-scoring GPU ones at sliding-window boundaries (w=4, u=8 for Qwen)**. Built on KTransformers, tested on RTX 3090 with PCIe 4.0 x16 and EPYC 7532. Decode speedups average 3.97x over llama.cpp, 2.16x over KTransformers, 1.48x over MoE-Lightning and 1.32x over HybriMoE. Its cache policy alone is 1.23x over HybriMoE's score cache. No code link — [arXiv 2602.03495](https://arxiv.org/html/2602.03495v1)
- ProMoE's chunked, cancellable transfer planner is already in the F2 table (item 30) — [Grouped technologies for pancake.md](file:///home/dan/reports/Grouped%20technologies%20for%20pancake.md)

### Inferences
- **Pie → an adaptive swap budget (exact, F2).** Pancake runs fixed `--adapt-swaps 64/96/160` (the A/B is queued). A Pie-style controller would raise the per-window swap count while the window time (or the CPU phase, or the combine pull) stays flat, and back off when it rises. That makes the budget self-tuning across short chats, edits and post-prompt refill. The input signals already exist (window timing, DMA events). **Expected: picks the best of the 64/96/160 arms automatically per workload phase, plus avoids the contention loss the grouping report flags ("DMA contention decides how much of the gain remains"). Estimate: 0-5% beyond the best fixed setting. Effort small-medium, risk low.**
- **MoE-Lightning paging → chunked promotions gated by layer phase (exact, F2/F3).** A 1.4 MB promotion takes the link for ~115 us. If one is in flight when the CPU flag flips, the row pull (Q2) shares the downstream link and slows on the critical path. Splitting promotions into ~256 KB pages and issuing them only while the CPU pool is computing (doorbell rung, flag not yet raised), or from the fill thread with a "pause when the layer's flag is near" gate, keeps the pull uncontended. **Expected: only matters once admission raises promotion volume. Protects 1-2% of window time (estimate). Effort medium.** It becomes moot if the Q2 push/pre-reduce shrinks the pull.
- **Cancellation.** A queued promotion whose target layer's next use is already past (or whose victim was just re-referenced) should be dropped before issue. The fill queue in `ForesightSwap::fill_loop` pops FIFO with no staleness check, and `ram_stage` dedups but does not age entries. **Expected: small, exact, cheap.** ProMoE covers the idea; this note just locates it in the code.
- **DALI** mostly confirms pancake's design (window-boundary swaps, GPU cost = max(transfer, compute), separate prefetch stream). Its workload score (tokens routed per expert in the window) is a candidate admission score for the `time_ceiling.py` simulator, not a transfer-path change. Note the hardware difference: PCIe 4.0 and a 3090, so its 1.3-4x factors do not transfer.
- **Klotski and MoE-Lightning** are throughput (batched) designs. Their multi-batch pipelines do not apply to single-stream decode. Only the paging idea carries over.

### Gaps
- No paper in the window evaluates promotion/compute overlap on PCIe 3.0 with CPU-computed misses. All of them use PCIe 4.0+ or NVLink-C2C (Pie).
- Klotski's and FloE's institutions and exact baselines were not verified.

## Q4. Pinned/registered host memory costs, IOMMU effects, and the load path

### Takeaway
On this machine the IOMMU costs nothing (passthrough), and it should stay that way. The recent IOMMU literature (F&S, SOSP'24) shows that strict translation with 4 KiB mappings makes IOTLB misses unavoidable. Pinning cost matters only at startup. The biggest transfer-path win in this family is operational: ServerlessLLM-style loading (chunked, multi-threaded direct I/O straight into the pre-registered arena) for the 28-65 s restart that currently reads 33 GB at 0.5-1.2 GB/s from ntfs3.

### Cited Findings
- Fast & Safe IO Memory Protection (Rubin, S. Agarwal, Cai, R. Agarwal; **university-led**, Cornell group per author list, listed in UIUC's portal; SOSP 2024). With strict IO memory protection, IOTLB misses are unavoidable. F&S instead cuts the cost per miss by using the IOMMU's page-table caches, with a few Linux kernel changes, and "nearly eliminating" protection overhead. Code: xlab-uiuc/vfree — [UIUC listing](https://experts.illinois.edu/en/publications/fast-amp-safe-io-memory-protection/); [code](https://github.com/xlab-uiuc/vfree)
- pcie-bench: IOMMU use has a measurable cost, and PCIe performance depends on IOTLB state (background, 2018) — [Cambridge](https://www.cl.cam.ac.uk/research/srg/netos/projects/pcie-bench/)
- ServerlessLLM (Edinburgh-led; arXiv 2401.14351; the OSDI'24 venue was not confirmed by this pass's search). A loading-optimized checkpoint format, a pinned-memory chunk pool (pinning avoids an extra DRAM copy and lets one thread drive PCIe efficiently), and a multi-tier pipeline with several I/O threads per tier (NVMe, SATA, S3, pinned RAM) that queue chunk offsets to the next tier. Loading is 3.6-8.2x faster than PyTorch/Safetensors — [arXiv 2401.14351](https://arxiv.org/pdf/2401.14351); direct I/O (O_DIRECT) per a secondary summary — [themoonlight review](https://www.themoonlight.io/review/serverlessllm-low-latency-serverless-inference-for-large-language-models)
- Local: IOMMU group type `identity`, THP `madvise`, arena in 4 KiB pages under NUMA split, registered with `cudaHostRegister`; the restart reads the 33.02 GiB arena in 28-65 s (0.51-1.18 GiB/s) from ntfs3 — [local sysfs; expert_source.cpp:3900-3920; Pancake pain points and solutions.md]

### Inferences
- **IOMMU (no change; a guardrail).** Passthrough means no IOVA translation for the GPU's SDMA reads of the arena. If someone enables strict IOMMU (for example `iommu.passthrough=0`, VFIO experiments, or a distro default change), 1.4 MB promotions scattered over a 33 GB, 4 KiB-page arena would take IOTLB misses (~344 pages per expert). pcie-bench and F&S show this is a real cost. **Action: add a startup log line that prints the GPU's iommu_group type and warns if it is not `identity`/`DMA-FQ`.** Effort trivial.
- **Pinning cost.** `hipHostRegister` of 33 GB in 4 KiB pages pins ~8.6M pages at startup (cost not measured here; it is part of the "~10 s setup"). Huge pages (THP `madvise` on the arena when NUMA split is off, or hugetlbfs) would cut registration work and page-table footprint. They are not expected to change steady-state DMA speed with the IOMMU in passthrough. **Verdict: startup-only, owner decision (already listed as hugepages in F10).**
- **ServerlessLLM loading → restart path (F10).** The arena load is filesystem-bound, not PCIe-bound. Reading with O_DIRECT, large aligned chunks and 4-8 I/O threads straight into the already-registered arena, from an ext4/xfs copy rather than ntfs3, should approach the NVMe's sequential rate (drive not identified here; at ~3 GB/s that is ~11 s instead of 28-65 s, estimate). The engine already has pread I/O threads for other paths (`io_pf_`, expert_source.cpp:3037). **Effort small-medium, risk low, exact. Gain: −15 to −50 s per restart, plus less page-cache pressure that triggers systemd-oomd.**

### Gaps
- Measured `hipHostRegister` time for 33 GB on this kernel, with and without THP: not available.
- Which NVMe holds the arena and its sequential read rate: not checked.

## Q5. Compression on the wire (compress in RAM, decompress on GPU): does any recent work change the "not worth it" verdict?

### Takeaway
No. 2025-2026 lossless work (Huff-LLM, DFloat11, ECF8) targets the exponent redundancy of FP16/BF16/FP8 weights, about 15-32% savings, which a 2-bit Q2_0 integer format does not have. The lossy "compress for transfer" line (FloE: sparsity plus ultra-low-bit, 9.3x per expert, 4.4-7.6% quality loss) is the same family as HOBBIT/turbo (F1), not a transfer optimization. With `--pcie-frac 0`, wire compression would help only promotions.

### Cited Findings
- Huff-LLM (Yubeaton et al.; arXiv 2502.00922, Feb 2025; institution not verified here). Per-field Huffman coding of FP16 weights (1-5-5-5 split) cuts on-chip memory and bandwidth by 15-32%. It is a hardware decoder design (about 6% area vs the MAC array). High-entropy weights compress less — [arXiv 2502.00922](https://arxiv.org/pdf/2502.00922)
- DFloat11 (arXiv 2504.11651; NeurIPS 2025 per its GitHub): entropy-coded BF16, ~30% smaller, bit-exact, GPU decode kernel; 1.9-38.8x throughput over CPU-offloading an uncompressed model — [arXiv 2504.11651](https://arxiv.org/pdf/2504.11651)
- ECF8 (arXiv 2510.02676): the same idea for FP8 exponents — [arXiv 2510.02676](https://arxiv.org/html/2510.02676v1)
- FloE (Zhou, Li et al.; ICML 2025): hybrid contextual sparsity plus ultra-low-bit quantization of experts, with inter/intra-expert sparsity predictors to prefetch compressed weights. 9.3x compression per expert on Mixtral-8x7B on an RTX 3090, 4.4-7.6% average quality degradation. The abstract's 48.7x speedup (vs DeepSpeed-MII) and the v2 text's 2.6x disagree, presumably different baselines — [arXiv 2505.05950](https://arxiv.org/abs/2505.05950)
- Statistically-lossless quantization notes that ZipNN/DFloat11-style lossless methods need higher bitrates than quantization — [arXiv 2605.02404](https://arxiv.org/pdf/2605.02404)

### Inferences
- **Q2_0 on the wire (exact):** the only possible redundancy is a non-uniform distribution of the 2-bit codes plus the scales. A Gaussian-ish weight distribution under a symmetric 4-level quantizer could put entropy somewhat under 2 bits, but no source quantifies it for Q2_0 (gap). Even a 10% gain would apply to promotions only, cost a GPU decode kernel before the expert can be used, and need a second layout in RAM. **Verdict: a five-minute offline entropy check (zstd / order-0 Huffman over one Q2_0 down tensor) can close this for good. Do not build anything unless it shows ≥20%.** Family F1 (bytes per expert) only if lossy; exact wire compression has no family slot worth filling.
- **FloE-style lossy transfer** belongs with HOBBIT/turbo/sparsity masks in F1 and should go through the same quality gate. Nothing transfer-specific is gained.

### Gaps
- No published entropy measurement of llama.cpp Q2_0 / IQ2 tensors was found.

## Q6. GPU-initiated transfers: is there anything to gain from the GPU driving its own copies or storage reads?

### Takeaway
GPU-initiated I/O (BaM, NVIDIA SCADA) targets many small random reads from NVMe at GPU-thread granularity. It is NVIDIA-stack and data-center oriented and needs P2P DMA from the SSD into GPU memory. Pancake's NVMe tier is cold and its RAM arena is hot, so the payoff is nil. On AMD, the useful GPU-side primitives are queue-level: `hipStreamWaitValue32/WriteValue32` exist in ROCm 10, while DMA "prelaunch/poll" commands are not exposed yet (DMA-Latte). Verdict: nothing to build now. One minor use (WriteValue for flags) is parked under Q1.

### Cited Findings
- BaM (multi-institution: NVIDIA, UIUC, IBM, Stanford, Buffalo; ASPLOS 2023, pp. 325-339): GPU threads use a user-level library of submission/completion queues in GPU memory and a fine-grained software cache that coalesces requests. 5.3x over CPU-initiated storage access for data analytics; 1.0x and 1.49x on BFS/CC graph analytics while cutting hardware cost up to 21.7x against host-memory serving. Code: github.com/ZaidQureshi/bam — [arXiv 2203.04910](https://arxiv.org/abs/2203.04910); [code](https://github.com/ZaidQureshi/bam)
- NVIDIA SCADA productizes the BaM approach for high volumes of small random reads, complementary to cuFile/GPUDirect Storage for bulk — [StorageReview](https://www.storagereview.com/news/nvidia-scada-puts-storage-control-on-the-gpu-as-cufile-goes-open-source)
- AMD MORI CCO (AMD technical article, Sep 2026; blog, not a paper): a device-side collective object that can issue SDMA all-gathers from inside a kernel — [AMD](https://www.amd.com/en/developer/resources/technical-articles/2026/mori-cco.html)
- DMA-Latte: the DMA "prelaunch" (engine polls a memory value, then copies) is proposed and not exposed in the AMD stack — [arXiv 2511.06605v2](https://arxiv.org/html/2511.06605v2)
- ROCm 10 header exposes `hipStreamWaitValue32` (support queryable via `hipDeviceAttributeCanUseStreamWaitValue`) — [local: /opt/rocm/core-10.0/include/hip/hip_runtime_api.h:585, 3326]

### Inferences
- **NVMe tier:** experts that miss both VRAM and the RAM arena are rare in steady state (the arena holds the full model). A GPU-initiated NVMe path would also need P2P DMA from the SSD into VRAM over the same Gen3 link. **Verdict: skip.**
- **A device-triggered promotion** (the GPU decides "promote expert e" at the router and an SDMA engine starts without a CPU hop) would save the router-to-fill-thread latency, a few tens of us at most. Promotions are not latency-critical, since they serve later windows. **Verdict: skip until AMD exposes DMA poll/prelaunch. Even then it is low value.**

### Gaps
- No university-led GPU-initiated transfer work targeting AMD consumer GPUs was found.

## Q7. Overall verdict for pancake: which transfer-path techniques are worth doing, ranked

### Takeaway
At `--pcie-frac 0` the host-to-device expert path is no longer on the critical path. The transfer that is on it is the device's pull of CPU rows (and the spin before it) at every layer. The only transfer-path items with a plausible measurable decode gain are therefore (1) shrinking or pushing that row transfer and (2) keeping promotions from colliding with it, with a self-tuning swap budget. All are exact or near-exact. Each is worth ~0.5-5%, and they protect the F2 admission gain. Restart-time loading is the biggest absolute win, but it is operational (F10).

### Cited Findings
- Local profile: 38.3% of GPU kernel time is `combine_cpu`, 4.1% `wait_flag_copy_i32` — [Pancake pain points and solutions.md](file:///home/dan/reports/Pancake%20pain%20points%20and%20solutions.md)
- Combine reads fp32 rows per non-hit slot from mapped coherent host memory, with a per-block host-flag spin — [src/kernels/cuda/native_moe.cu:145-200; src/core/verify.cpp:516-520, 1017]
- Pie (Berkeley), MoE-Lightning (Berkeley), EMOGI/PyTorch-Direct (UIUC), DeepPlan (Ajou), F&S (SOSP'24), ServerlessLLM (Edinburgh), DMA-Latte (AMD): see Q1-Q6 for URLs

### Inferences
Ranked table (all figures are estimates unless marked measured):

| # | Technique (source) | Family | Where in pancake | Expected gain | Effort | Risk | Exact? |
|---|---|---|---|---|---|---|---|
| 1 | Measure the spin/pull split inside `combine_cpu`, and promotion bandwidth with WAIT_FUSE on/off (EMOGI-style micro-benchmark) | F9 (measurement) | replay kernel, `stamp()` points | enables 2-4 | 0.5 day | none | — |
| 2 | CPU pre-reduction of result rows per token (fp32; bf16 opt-in) (EMOGI/DeepPlan principle: move fewer bytes over the link on the critical path) | F3 | pool output → `h_ymiss_`, combine kernel variant | 0.5-1.5% on edits | small-medium | low (fp32 order); medium (bf16) | algorithmic yes, not bit-identical |
| 3 | Push rows and flag into fine-grained VRAM via ReBAR (posted WC writes); the GPU spins locally (Lustig & Martonosi fine-grained sync; local ReBAR) | F3 | pool workers, flag/rows allocation | 1-2% on edits (overlaps with #2) | medium | medium (WC ordering, ROCm support) | yes |
| 4 | Single host-flag waiter republishing to a VRAM flag under WAIT_FUSE | F3 | `combine_cpu_k10_vec4` wait | 0-few % of promotion throughput (hypothesis) | small | low | yes |
| 5 | Pie-style online swap-budget controller (performance-transparent swapping) | F2 | adaptive tier / `--adapt-swaps` | 0-5% beyond the best fixed budget | small-medium | low | yes |
| 6 | Paged promotions issued only during the CPU phase; staleness cancellation in the fill queue (MoE-Lightning paging; ProMoE) | F2/F3 | `ForesightSwap::fill_loop`, `ram_stage` | protects ~1-2% window time once admission raises swap volume | medium | low | yes |
| 7 | ServerlessLLM-style arena load (O_DIRECT, chunked, multi-thread, into the registered arena, off ntfs3) | F10 | startup arena read | −15 to −50 s per restart | small-medium | low | yes |
| 8 | IOMMU guardrail log (F&S, pcie-bench) | F10 | startup | protects status quo | trivial | none | yes |
| 9 | Coalesce address-adjacent promotion copies; `STRATA_COPY_HIP_PRIO` A/B (DMA-Latte overhead curve) | F2 plumbing | fill thread, copy stream | 0-1% | trivial-small | low | yes |
| — | Skip: batch-copy API (absent in ROCm 10), wire compression of Q2_0 (unless an entropy check shows ≥20%), zero-copy misses at Gen3, GPU-initiated NVMe/SDMA, FloE-style lossy transfer (belongs to F1) | — | — | ~0 | — | — | — |

- Stacking: #2 and #3 overlap, since both shrink the same pull. Do #2 first because it is simpler, then #3 only if the measurement still shows pull time. #4 is subsumed by #3. #5 and #6 are complements: one sets how much to move, the other sets when. All of them sit under the F2/F3 shared ceiling of ~20 ms per window from the grouping report. Their value grows once turbo, admission or deferral shrink the CPU phase, because then the pull and the spin traffic become a larger share of the window.
- None of the items needs a new paper-scale mechanism. The research mainly says that the pull direction deserves the same care as the push direction (alignment, fewer bytes, posted writes, no contention), and that swap budgets should be controlled online, not fixed.

### Gaps
- Every decode-gain figure in this section depends on measurement #1. If the pull is ≤3 us per layer, items 2-4 drop to ~0.
- No source in Nov 2022 - Nov 2026 evaluates MoE expert streaming on PCIe 3.0, on AMD consumer GPUs, or with CPU-computed misses returning partial outputs over mapped memory. The applicability judgements are first-principles extrapolations from the cited mechanisms.
