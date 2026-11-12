# GPU side of the CPU+GPU hybrid pipeline: synchronization, idle filling, launch/fusion, queues, RDNA3 kernels, measurement (pancake on RX 7900 XT)

Scope: the GPU half of pancake's per-layer hand-off (GPU attention/GDN -> router doorbell -> CPU experts || GPU VRAM-hit experts -> combine). Read-only: repo grep, existing profile files, `lspci`/sysfs. No builds or profiler runs. Literature window is Nov 2022 - Nov 2026. Two older foundational papers are marked as such. Family tags (F1-F10) follow `pancake-docs/research/engine-speed/grouped-technologies/report.md`. Already covered elsewhere and **not repeated here**: Hazy "no bubbles" megakernel and HipKittens (Stanford report), FlashInfer, NanoFlow, Punica, Atom (SUA report), KTransformers deferral, HybriMoE, PowerInfer, DAOP and the Tsinghua per-expert barriers (Asia report), Fiddler and MoE-Lightning (Stanford/SUA), Marlin (Europa), plus the local "tested, no gain" list (HIP graphs, decode GDN megakernel, `hw queues`, global unroll).

## Q1. Where does the GPU time actually go, and what is on the critical path? (grounding from the repo and existing measurements)

### Takeaway
About 98% of `combine_cpu_k10_vec4` time is genuine waiting, not PCIe reading. The host-side window timer also shows that GPU pre-doorbell work and CPU expert work **add up** on edit windows. So GPU kernels *before* the router doorbell (GDN/QSA/hyper-connection/router/shared expert) are on the critical path on both short and edit windows. GPU VRAM-hit expert kernels (expdec_gu, `native_down_multi_kernel<42>`) are mostly hidden behind the CPU share. This partly contradicts the earlier "F8 ~0 on edits" and "Q2_0 down kernel worth 0.3-0.6 ms" statements.

### Cited Findings
- rocprofv3 kernel stats (one short request, profiler slows decode to 79 vs ~126 tok/s): `combine_cpu_k10_vec4<true>`: 4,128 calls, 1,049 ms total, **average 254 µs, minimum 5.6 µs**, maximum 7.83 ms, 38.26% of GPU time. `wait_flag_copy_i32`: 4,224 calls, average 26.4 µs, minimum 5.44 µs, 4.07%. `native_down_multi_kernel<42>`: 4,224 calls, 37.6 µs average, 5.79% — [run_kernel_stats.csv](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-09-7900xt-gpuprof/run_kernel_stats.csv)
- Same profile: 395 kernels per token on the main stream, gaps <50 µs average 5.4 µs (profiler-inflated), >1 ms host stalls at window and request edges (kv_append -> copyBuffer, copyBuffer -> copy_from_mapped) total 525 ms — [gpuprof README](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-09-7900xt-gpuprof/README.md)
- Normal-run (no profiler) window timer, edit window: `37.39 ms/window = ... verify 34.73 (GPU-reach wait 15.75 + per-layer host 17.94 [plan 0.27 ... CPU 17.41]) ... + draft 2.50`. Short window: `24.81 = ... GPU-reach wait 15.93 + per-layer host 5.59 [plan 0.15 ... CPU 5.19] ... + draft 2.24`. GPU stage stamps (ms/window, edit / short): router+plan+bell 1.26 / 1.05, **waitA 0.45 / 0.39**, **VRAM hits 4.26 / 3.25**, waitB 0.20 / 0.20 — [profile-fasthip.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/profile-fasthip.txt)
- Code facts (read-only grep):
  - `combine_cpu_k10_vec4` (native_moe.cu:145) reads the CPU expert rows **directly from mapped host memory** (`const volatile float4* cpu4`, zero-copy).
  - With `STRATA_WAIT_FUSE`, thread 0 of **every block** spins `while (*flag < value) strata_spin_pause();` and then runs `__threadfence_system()`. The grid is `((n4+63)/64, n_tok)` blocks of 64 threads, i.e. 10 blocks per token at n_embd 2560, so **10×n_tok concurrent pollers of one host flag**.
  - `strata_spin_pause()` is `__nanosleep(100)` (include/strata/kernels/dp4a.hpp:47).
  - By contrast, `wait_flag_copy_i32_kernel` launches `<<<1,128>>>` (one poller) (verify_kernels.cu:1777).
  - Mapped buffers come from `hipHostMalloc(Mapped|Coherent)` only when `STRATA_VERIFY_COHERENT=1`, else `cudaHostAllocMapped` (verify.cpp:513).
  - The doorbell ring/publish kernels use volatile stores, plus a HIP-only fence because "on RDNA4 the volatile store alone can sit in L2 until the stream syncs" (elementwise.cu:206-212, #697).
  - A device-side planner exists only for all-resident groups (`resident_plan_kernel`, `device_plan_`, "E-6").
  - The fast-hip preset sets `GPU_MAX_HW_QUEUES=6`, `STRATA_COMBINE_CPU=1`, `STRATA_WAIT_FUSE=1`, `STRATA_ROUTE_PUBLISH=1` — [pancake-flashnext-iq2_xs-fast-hip.json](file:///home/dan/clone-of-d/src/Strata/pancake-flashnext-iq2_xs-fast-hip.json)
- Platform facts read from sysfs/lspci on this machine:
  - The GPU link is 16 GT/s x16 to the card's internal switch, but the **CPU root port 00:01.1 runs at 8 GT/s x16 (PCIe 3.0)**.
  - **BAR0 is 32 GB**, so Resizable BAR is active and the CPU can address all VRAM.
  - Source: local `lspci -vv -s 12:00.0` and `/sys/bus/pci/devices/*/current_link_speed`, 2026-11-09.
- RX 7900 XT (gfx1100): 84 CUs, wave32 or wave64, LDS 128 KiB, **Infinity Cache 80 MiB**, L2 6 MiB, graphics L1 256 KiB, L0 vector 32 KiB, VGPR file 768 KiB — [ROCm GPU specs](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html)

### Inferences
- **The non-wait part of combine is tiny.** The 5.6 µs minimum is the cost of a call when the CPU is already done (zero-copy read + combine). It bounds that part at ≤5.6 µs × 4,128 ≈ 23 ms of 1,049 ms (≈2.2%). So ≥97.8% of combine_cpu time is waiting on the CPU. Making the combine itself faster or moving the rows into VRAM can save at most ~0.27 ms/window (5.6 µs × 48 layers).
- **Critical-path model.** Per layer, the chain is:
  [GPU pre-bell: GDN/QSA + HC + router + publish] -> [CPU experts ‖ (GPU waitA + VRAM hits)] -> combine -> next layer.
  - The host timer's "GPU-reach wait" is the time the host waits for the GPU to reach the doorbell. That equals pre-bell GPU work + max(0, hits − CPU) + combine + hand-off latency.
  - GPU-reach stays at 15.75-15.93 ms while VRAM hits grow from 3.25 to 4.26 ms. That fits hits being hidden behind CPU work.
  - Consequences:
    - (a) Speeding up pre-bell kernels shortens **edit windows too**, ms for ms. The earlier claim "a 10% faster GPU gives ~0 on edits" holds only for the VRAM-hit expert kernels.
    - (b) A faster Q2_0 down kernel (part of "VRAM hits") pays only in layers where GPU hits exceed the CPU share (CPU 5.19 > hits 3.25 even on short windows on average). Its window gain is likely well below the 0.3-0.6 ms estimated in the PP report, until a per-layer distribution shows otherwise.
- **The GPU idle that can be filled is CPU − hits per layer.** That is ≈ (5.19 − 3.25)/48 ≈ 40 µs/layer on short windows and ≈ (17.41 − 4.26)/48 ≈ 275 µs/layer on edits. Only work that does not depend on layer l's output can go there.
- The round-trip plan hand-off (waitA) costs ~0.39-0.45 ms/window in a normal run, not the ~1.27 ms implied by the profiled 26 µs average.

### Gaps
- No per-layer distribution of (CPU share − hit time). Needed to price the hit-expert kernels (Q2_0 down, MonoMoE-style designs) and deferral.
- The normal-run inter-kernel gap on RDNA3 is unknown. The 5.4 µs average is profiler-inflated.
- I did not verify what `__nanosleep` lowers to on HIP/gfx11 (likely `s_sleep`, unconfirmed) or the gfx11 code sequence for volatile/system-scope loads. The LLVM AMDGPUUsage page was too long to reach that section.

## Q2. CPU<->GPU synchronization mechanisms: what exists in the literature/runtime, and what fits pancake

### Takeaway
Pancake already runs the state-of-the-art lightweight mechanism: GPU-side spin on mapped flags plus a doorbell/partial-flag protocol. That is the same idea as the classic full/empty-bit proposal. Runtime alternatives (stream wait-value, host callbacks, blocking events) save no latency on this path. Two cheap, concrete GPU-side tweaks remain:
1. Cut the 10×n_tok host-flag pollers in `combine_cpu` to one poller.
2. Optionally write CPU rows and flags straight into VRAM through the 32 GB BAR.

### Cited Findings
- **Lustig & Martonosi, "Reducing GPU Offload Latency via Fine-Grained CPU-GPU Synchronization"** (Princeton lead, HPCA 2013; foundational, outside the window). It proposes full/empty bits tracking which regions of transferred data are ready, so consumers start before the whole transfer completes — [hgpu listing](https://hgpu.org/?p=8831); [Bohrium summary](https://www.bohrium.com/paper-details/reducing-gpu-offload-latency-via-fine-grained-cpu-gpu-synchronization/813219231578456065-100645)
- **Elmougy, Jia, Ding, Shan, "Diagnosing the Interference on CPU-GPU Synchronization Caused by CPU Sharing in Multi-Tenant GPU Clouds"** (NJIT/Hofstra, IPCCC 2021; outside the window). Polling-based sync suffers more interference than blocking sync when the CPU is shared, up to ~29.79 ms polling latency under vCPU descheduling. The authors propose polling-then-blocking — [Hofstra PDF](https://cs.hofstra.edu/~jshan/papers/gpucpusync.pdf); [slides](https://ele.uri.edu/faculty/weiwei/materials/ipccc21.pdf); [NJIT record](https://researchwith.njit.edu/en/publications/diagnosing-the-interference-on-cpu-gpu-synchronization-caused-by-/)
- **HIP stream memory operations**: `hipStreamWaitValue32` is a **beta** API that enqueues a wait so that later stream work does not run until the condition (Gte/Eq/And/Nor) holds. Support is queried with `hipDeviceAttributeCanUseStreamWaitValue`. One docs version says the pointer should come from `hipMallocSignalMemory` — [HIP 6.2.4 docs](https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.4/reference/hip_runtime_api/modules/stream_memory_operations.html); [HIP 10.0.0 docs](https://rocm.docs.amd.com/projects/HIP/en/docs-10.0.0/reference/hip_runtime_api/modules/stream_memory_operations.html)
- Pancake already uses `cudaLaunchHostFunc` for the peer-expert refill (peer_experts.cpp:395) and `cudaEventBlockingSync` events on snapshot/restore paths (conversation_snapshot.cpp:701). The per-layer path uses GPU spin kernels and doorbells only (repo grep). MOD_PLAN item 28 says: "Hand-off stays the doorbell/partial-flag protocol (no HIP graphs)" — [MOD_PLAN](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)

### Inferences (technique -> verdict)
- **Full/empty bits (Lustig & Martonosi) -> already implemented (F3).** The partial flag (`raise_partial`, `STRATA_VERIFY_DEFER_CPU_BELOW`) and the A/B/CPU flags are this idea. Finer granularity (per-row readiness) adds nothing for combine, because the next layer needs the full hidden state and the post-wait work is ≤5.6 µs.
- **`hipStreamWaitValue32` instead of a spinning kernel -> low value; use only with concurrent GPU work.**
  - What it does: the command processor waits instead of a wave, so no CU is occupied and the stream does not hold a dispatch slot.
  - What it costs: a separate combine launch after the wait resolves, which re-adds the launch latency that WAIT_FUSE removed. Support for host-pinned flags is undocumented and the API is beta.
  - Expected gain: ~0 on its own. It is useful only if Q3's idle-filling work runs on another queue and the spinners prove to get in its way. Effort S, risk medium (beta API, ROCm 10 behaviour unverified). Family F3/F8.
- **Host callbacks (`hipLaunchHostFunc`) and blocking events -> keep out of the per-layer path.** They hand control to a runtime thread and add OS wake-up latency. The IPCCC'21 result says blocking only wins under CPU oversubscription, which pancake avoids by pinning pool threads. No action. Family F3.
- **New, cheap: a single poller in `combine_cpu` (F3, exact).**
  - The problem: with WAIT_FUSE, every one of the 10×n_tok blocks (≈50 at 4.9 tokens/window) polls the same host flag across PCIe. Each poll is a coherent read that the root complex must service from the CPU side, roughly one per ~1 µs per poller with `__nanosleep(100)`. That traffic flows during exactly the window when the CPU pool runs at ~27 of 33 GB/s DDR4.
  - The change: block 0 (or a preceding 1-thread kernel, as `wait_flag_copy_i32` already does) polls the host flag. It then writes a device-memory flag (or bumps a device counter), and the other blocks spin on that flag in L2.
  - Expected gain: unknown, probably 0-3% of CPU expert time on edits if the polling traffic perturbs the pool. Possibly a slightly faster wake-up for the first block.
  - Effort S (one kernel, ~20 lines). Risk low (exact).
  - Test: A/B `STRATA_WAIT_FUSE=1` vs `0` (single-poller `wait_flag_ge`) while reading the host-side "CPU 17.41" per-window figure, not only tok/s.
- **New: CPU writes rows and flags into VRAM via the 32 GB BAR (F3, exact).**
  - The idea: the pool's final down-projection outputs (10 KB fp32 per expert row) go to a VRAM buffer with write-combining stores + `sfence`, then a VRAM flag is set. The GPU polls locally and combine reads locally. Resizable BAR is active here, and the root port is PCIe 3.0 x16.
  - Expected gain: ≤0.27 ms/window (the combine non-wait floor), plus slightly faster flag detection, so ≤~1%.
  - Effort M (BAR mapping of a VRAM buffer via `hipHostRegister`/IPC or KFD; ordering rules).
  - Risk medium: WC ordering, and the RDNA4 #697-type visibility bugs already seen with stores.
  - Priority low; consider it only together with a VRAM-resident flag for the single-poller change.
- **Spin backoff (`__nanosleep(100)`) -> leave as is.** Detection latency is dominated by the PCIe round trip. Two hand-offs × 48 layers × ~1-2 µs ≈ 0.1-0.2 ms/window at most (estimate).

### Gaps
- No published measurement of host-flag polling latency, or of polling traffic interference with CPU DRAM bandwidth, on AMD consumer platforms (Zen 3 APU + RDNA3). I found no dedicated paper; the closest are Lustig & Martonosi and Elmougy et al.
- No source for `hipLaunchHostFunc` or `hipStreamWaitValue32` latency on ROCm 10 / gfx1100.

## Q3. Reducing GPU idle while it waits for CPU results (idle filling, overlap, work stealing)

### Takeaway
The only exact way to *use* the 40-275 µs/layer spin without touching the dependency chain is to prefetch the **next layer's pre-doorbell weights into the 80 MiB Infinity Cache** during the spin, as PRESERVE does with L2 during communication. Q1 shows those kernels are on the critical path on both window types. Deferral (F3) remains the big lever. CPU/GPU work stealing over PCIe was already explored (`--pcie-frac`, PcieAuto) and is capped by the shared DDR4.

### Cited Findings
- **PRESERVE: Prefetching Model Weights and KV-Cache in Distributed LLM Serving** (Yüzügüler, Zhuang, Cavigelli; Huawei Zurich, industry not university; arXiv 2501.08192, Jan 2025, revised May 2025).
  - Mechanism: prefetches weights and KV from HBM into on-chip L2 *while collective communication runs*. This is legal because weights do not depend on the communication result.
  - A graph pass inserts the prefetch ops automatically, and a compile-time L2 budget prevents pollution.
  - Results: up to 1.6× end-to-end on Ascend 910B. Design-space exploration moves the optimal L2 from 8 MB to 104 MB, for a further 1.25× perf/cost.
  - Sources: [arXiv HTML v2](https://arxiv.org/html/2501.08192v2); [abs](https://arxiv.org/abs/2501.08192v1)
- **PreScope** (Yu, Zhang, Dong, Wu, Liao; NUDT lead, Tsinghua co-author; arXiv 2509.23638, Sep 2025, venue not stated).
  - Hybrid CPU+GPU MoE built from a per-layer predictor (LLaPor) and a cross-layer cost model (PreSched) that decides GPU on-demand load vs CPU execution and how many prefetches fit in the "CPU compute gap". Scheduling is event-driven (OnIoDone, OnCpuGpuDone).
  - Transfers use pre-registered GPU buffers with gate/up/down split over 3 streams.
  - Tested on V100 over **PCIe 3.0 x16** and A100 over PCIe 4.0.
  - Results: up to +70.8% throughput and −42.1% decode latency vs HybriMoE, and +129% / −72.3% vs Fiddler. No code link.
  - Source: [arXiv HTML](https://arxiv.org/html/2509.23638v1)
- **Klotski** (Sun Yat-sen lead; arXiv 2502.06888, Feb 2025): multi-batch, expert-aware pipeline that orders computation against I/O to remove bubbles when per-expert compute is shorter than its I/O. Experts are computed only on the GPU (offloading, not CPU execution). It reports up to 85.12× throughput, with a baseline and hardware context not given in the abstract — [arXiv](https://www.arxiv.org/abs/2502.06888)
- Local, from MOD_PLAN/grouping report:
  - `--pcie-frac 0` was chosen because a PCIe promotion costs ~115 µs vs ~42 µs of CPU compute.
  - PcieAuto "finish together" exists with a 0.20-0.25 share.
  - `STRATA_LOOKAHEAD_GPU` (router lookahead) is already the default.
  - Source: [Grouped technologies](file:///home/dan/reports/Grouped%20technologies%20for%20pancake.md)

### Inferences (technique -> verdict)
- **MALL/L2 warm-up of layer l+1 weights during the combine spin (PRESERVE applied to the CPU-wait bubble). Family F3/F8, exact. This is the new item worth testing.**
  - What: while `combine_cpu` for layer l spins, issue streaming loads over layer l+1's pre-bell weights (GDN/QSA projections, HC down/up, router, shared expert) so they are resident in the 80 MiB Infinity Cache when those kernels run. Total weight per layer is well under 80 MiB at Q6/Q8 for n_embd 2560 (estimate; exact size not computed).
  - Implementation options, simplest first:
    - (1) Inside `combine_cpu`, the 63 non-polling threads of each block (or extra blocks) walk a slice of the next layer's weight ranges while thread 0 polls. No new launch is needed, and the work stops when the flag flips.
    - (2) A low-priority prefetch kernel on a second HW queue, launched when the router doorbell rings.
  - Where it pays: only where pre-bell kernels are DRAM-bandwidth-bound and the weights would otherwise miss MALL. With ~4.7 tokens/window and >80 MiB of total dense weights per window, they are most likely streamed from GDDR6 each time.
  - Expected gain: (1 − BW_dram/BW_mall) × the share of pre-bell time spent streaming weights. Pre-bell GPU time is ~12-15 ms/window, so a plausible range is 0-2 ms/window (0-8%) on both short and edit windows. This is an unvalidated estimate.
  - Prerequisite micro-test (no engine change): time one layer's pre-bell kernel sequence cold vs immediately re-run (MALL-warm) in `strata_bench` or a small HIP harness. If warm/cold < 0.85, build it.
  - Effort M. Risks:
    - whether RDNA3 allocates reads into MALL under the default driver policy (unknown);
    - eviction by the VRAM-hit expert reads in between;
    - on short windows the idle is only ~40 µs/layer.
- **Speculative linear pre-compute of next-layer projections on the GPU-only partial sum → reject.**
  - RMSNorm's 1/rms factor is a per-token scalar, so W·(g⊙h) splits linearly into a GPU-part and a CPU-part. But computing W twice doubles weight reads for a memory-bound GEMV.
  - It only makes sense combined with the MALL warm-up above, which already captures most of the benefit.
- **Event-driven CPU/GPU expert placement (PreScope, HybriMoE) → simulator candidate only (F2).**
  - PreScope's PCIe 3.0 V100 setting matches the hardware class. But its gains are vs Fiddler/HybriMoE baselines without pancake's adaptive tier, time simulator and lookahead.
  - The cost-model idea "prefetch only what fits in the CPU compute gap" equals Baleen-style bypass already in `time_ceiling.py`. Add the cost model as a policy in the item-14 simulator; no GPU-side work.
- **GPU work stealing of CPU-miss experts via zero-copy or PCIe → already explored, capped.**
  - The GPU reading expert weights from host memory pulls from the same DDR4 the pool uses (27 of 33 GB/s). The best case is ~+20% of CPU expert time before contention, and promotion cost (115 µs vs 42 µs) already made `--pcie-frac 0` win.
  - Revisit only if F1 (turbo/sparsity) leaves DDR4 headroom. Family F2/F3.
- **Klotski multi-batch pipelining → not applicable.** It is a single-user, batch-1 window with GPU-only expert compute plus disk offload.

### Gaps
- No source for RDNA3 Infinity Cache bandwidth or allocation policy for compute reads (driver-controlled). The GLC/SLC/DLC cache-policy encoding for gfx11 was not found in one search.
- Per-layer pre-bell weight bytes were not computed. Doing so would turn the MALL estimate into a bound.

## Q4. Kernel launch overhead, fusion and megakernels: trade-offs for a CPU-gated window

### Takeaway
University-led persistent-kernel work (CMU MPK, Cornell FlashDMoE) shows 1.0-1.7× (MPK) to 6× (FlashDMoE, multi-GPU) gains, but on NVIDIA and for GPU-only pipelines. In pancake, HIP graphs gave no gain and the decode GDN megakernel gave 0.0 ms. So the CPU-side launch cost is not the bottleneck. The remaining lever is the number of *dependent dispatches in the pre-bell section*, where Q1 places them on the critical path. Targeted fusion there (item 32 Gated Residual small-M, HC reads) is worth more than a whole-layer megakernel.

### Cited Findings
- **MPK: Mirage Persistent Kernel** (CMU lead, Zhihao Jia; co-authors Tsinghua, NVIDIA, Michigan, PKU; arXiv 2512.22219, Dec 2025).
  - Mechanism: compiles tensor programs into SM-level task graphs inside one persistent kernel, with an in-kernel decentralized scheduler.
  - Results: 1.0-1.7× vs SGLang/vLLM on A100/H100/B200; the README claims 1.2-6.7× latency reduction.
  - Code: github.com/mirage-project/mirage (`mpk` branch).
  - Sources: [arXiv](https://arxiv.org/html/2512.22219v1); [PDF](https://arxiv.org/pdf/2512.22219)
- **FlashDMoE / FlashMoE: Fast Distributed MoE in a Single Kernel** (Aimuyo, Oh, Singh; Cornell lead; arXiv 2506.04667, NeurIPS 2025).
  - Mechanism: one persistent kernel fuses expert compute and inter-GPU communication. Thread blocks and warps take actor roles, and work is split into fixed-size tiles.
  - Results: up to 9× GPU utilization, 6× lower latency and 5.7× throughput on 8×H100.
  - Code: github.com/osayamenja/FlashMoE.
  - Sources: [arXiv HTML](https://arxiv.org/html/2506.04667v2); [abs](https://www.arxiv.org/abs/2506.04667)
- **MonoMoE** (Gong, Budhathoki, Kim, Li, Khetan; affiliations not listed, apparently industry; arXiv 2609.04244, Aug 2026).
  - Mechanism: a weight-major persistent megakernel for block-quantized MoE decode. The decode token tile sits on the MMA N dimension, CTAs are partitioned over expert-weight tiles, and routing, top-k, quantization, both projections, activation and reduction are fused. Readiness flags overlap auxiliary work.
  - Results (H200 only): up to 1.54× on the routed-MoE operator vs vLLM Triton grouped GEMM, and up to 18.7% lower TPOT.
  - Code: FlashInfer `csrc/fused_moe/monomoe`.
  - Source: [arXiv](https://arxiv.org/abs/2609.04244)
- Local results:
  - HIP graphs: no gain.
  - Decode GDN megakernel: 0.0 ms ("the window waits on CPU rows").
  - The RDNA3 fusion ports (COMBINE_CPU/WAIT_FUSE/ROUTE_PUBLISH and the HC fused read) moved decode 104-108 -> ~112 tok/s.
  - Source: [MOD_PLAN](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)

### Inferences (technique -> verdict)
- **Whole-layer or whole-model megakernel (MPK, FlashDMoE) → do not build (F8).**
  - On RDNA3 there is no MPK backend. The CPU hand-off inside a persistent kernel is exactly what pancake's WAIT_FUSE spin already does.
  - The GDN megakernel measurement shows no gain while the CPU gates the window.
- **Persistent "post-bell" kernel (waitA + plan copy + VRAM hits + combine as one launch, FlashDMoE actor style) → low.**
  - It saves ~3 dispatches/layer (~144/window), but that segment overlaps the CPU share in most layers (Q1). Gain ≈ the dispatch gaps × layers where hits > CPU.
  - Effort L, risk medium (it hides the stage stamps used for profiling).
- **Targeted fusion of dependent pre-bell kernels → yes, higher than previously rated (F8, exact).**
  - Candidates: item 32 (Gated Residual small-M), the HC norm/down/up chain (hc0/hc1 norm 0.30/0.10, down 0.53, up 0.63-0.64 ms/window on edits per profile-fasthip), and router+plan+bell (1.05-1.26 ms/window).
  - Every removed dependent dispatch and every DRAM round-trip of intermediate activations in this section reduces both short *and* edit windows (Q1 correction).
  - Expected gain: 1-4% per fused chain (estimate). Effort M each.
- **MonoMoE weight-major layout for the VRAM-hit experts / Q2_0 down → only after the per-layer critical-path check.**
  - The weight-major idea (all ≤8 window tokens on the N dimension, one pass over each expert's weights) is the right shape for M=1-8 on wave32 WMMA. But the hit kernels are mostly hidden behind the CPU (Q1).
  - Revisit when F1/F2 shrink the CPU share below the hit time.

### Gaps
- No published RDNA3 (gfx1100) measurement of back-to-back dependent dispatch latency in one HW queue under ROCm 10.
- MPK and FlashDMoE full texts were not read. The numbers come from abstracts and READMEs.

## Q5. Multi-queue / stream scheduling and AMD hardware queue contention

### Takeaway
ROCm's default is 4 hardware queues per device, and the docs recommend ≤4. Pancake runs 6, and the local A/B on "hw queues" came out flat. Streams beyond the limit share HW queues round-robin, which can create false serialization between the drafter, copy and main streams. CU-masked streams sit outside that limit and could isolate spinners or a prefetch kernel. None of this is a big lever. A 4-vs-6 re-check is cheap.

### Cited Findings
- `GPU_MAX_HW_QUEUES`: the maximum hardware queues allocated per device, default 4. When an application creates more HIP streams than this, "HIP runtime reuses the same hardware queues for the new streams in a round-robin manner". The recommended maximum is 4. The limit does not apply to CU-masked streams or the cooperative queue — [ROCm env variables (latest)](https://rocm.docs.amd.com/en/latest/reference/env-variables.html); [ROCm 7.14.1](https://rocm.docs.amd.com/en/docs-7.14.1/reference/environment-variables/index.html); [HIP env vars](https://rocm.docs.amd.com/projects/HIP/en/docs-6.3.1/reference/env_variables.html)
- `hipExtStreamCreateWithCUMask` creates a stream restricted to a CU bit-mask, one bit per CU. The caller is responsible for validity — [HIP stream management 10.0.0](https://rocm.docs.amd.com/projects/HIP/en/docs-10.0.0/reference/hip_runtime_api/modules/stream_management.html)
- Pancake's mtp.cpp creates the drafter stream at the highest priority, with the comment "ROCm hands out hardware queues per priority (GPU_MAX_HW_QUEUES per level)". The verifier creates prioritized copy streams (verify.cpp:1248-1270). The preset sets `GPU_MAX_HW_QUEUES=6`. The "hw queues" A/B was flat — [MOD_PLAN round A](file:///home/dan/clone-of-d/src/Strata/MOD_PLAN.md)

### Inferences (technique -> verdict)
- **`GPU_MAX_HW_QUEUES` 4 vs 6 → cheap re-check, low expected gain (F8/F9).** The docs recommend ≤4, and the earlier flat result means it does not matter today. Re-test only together with any change that adds a concurrent stream (the MALL prefetch kernel), because queue sharing then decides whether the prefetch actually overlaps.
- **CU-masked stream for spinners and prefetch → optional enabler.** Reserve e.g. 2-4 of 84 CUs for the flag-wait and MALL-warm kernels so that they never take CUs from pre-bell work. It is outside the HW-queue limit, so there is no round-robin aliasing. Effort S, gain only as an enabler of Q3.

### Gaps
- No documentation found on how RDNA3's MES maps more user queues than hardware slots (time-slicing cost) on consumer gfx1100.

## Q6. RDNA3 occupancy/wave scheduling and small-batch (M=1-8) GEMV/MoE kernel design, GPU-side dequant

### Takeaway
RDNA3 facts relevant to the M=1-8 kernels: wave32 by default with optional wave64, WMMA wave-wide with A/B fragments replicated across lane halves in wave32, 6 MiB L2 plus 80 MiB MALL. The Q2_0 down gap (5.79% of kernel time) is real in kernel-time terms. By the Q1 critical-path argument its window value is smaller than estimated before. Fusion and bandwidth work in the pre-bell section pays more.

### Cited Findings
- GPUOpen WMMA on RDNA3: WMMA operates wave-wide. In wave32 the A and B fragments must be replicated between lanes 0-15 and 16-31. There are 12 WMMA intrinsics (wave32 and wave64 groups). AMD guidance favours wave32 for divergent or long-running threads and wave64 for ALU-heavy regular work — [GPUOpen WMMA on RDNA3](https://gpuopen.com/learn/wmma_on_rdna3/); [RDNA3 presentation](https://gpuopen.com/presentations/2023/RDNA3_Beyond-the-current-gen-v4.pdf)
- gfx1100 (7900 XT): 84 CUs, LDS 128 KiB, VGPR file 768 KiB, L2 6 MiB, Infinity Cache 80 MiB — [ROCm GPU specs](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html)
- Pancake log: `strata expdec: CUDA0 down type 42 (n_embd 2560, n_ff 640) not covered: the default kernel runs`. `native_down_multi_kernel<42>` takes 5.79% (4,224 calls × 37.6 µs) — [Pancake pain points](file:///home/dan/reports/Pancake%20pain%20points%20and%20solutions.md); [run_kernel_stats.csv](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-09-7900xt-gpuprof/run_kernel_stats.csv)
- MonoMoE's weight-major layout for small-batch quantized MoE decode (Q4 above) — [arXiv](https://arxiv.org/abs/2609.04244)

### Inferences (technique -> verdict)
- **Q2_0 expdec-style down kernel (F8, exact) → keep in the queue, but re-price it.** It reduces kernel time (~1/3 of 5.79% ≈ 2% of GPU kernel time). Its effect on the window happens only in layers where GPU hits outlast the CPU share. Measure the per-layer (hits − CPU) distribution first, using the existing `STRATA_VERIFY_PROFILE` stamps per layer. Effort M, risk low.
- **`combine_cpu` launch shape → fix the polling (Q2), not the compute.** 64-thread blocks × 10 per token is fine for a 5.6 µs combine. Occupancy does not matter for a kernel that waits.
- **Wave priority (`s_setprio`) for spinners → not needed.** The spinners are few and sleep via `__nanosleep`. They do not compete for issue slots in a measurable way (inference, not measured).

### Gaps
- No university paper found on RDNA3-specific small-M quantized GEMV (wave32 + WMMA). The evidence base is NVIDIA-centric (MonoMoE, FlashInfer). HipKittens (Stanford, already covered) targets CDNA/MI-series.

## Q7. Measurement methodology for GPU idle/bubbles

### Takeaway
rocprofv3's `--kernel-trace --stats` mislabels CPU wait as GPU work (spin-in-kernel), inflates gaps (79 vs 126 tok/s), and cannot see the CPU-side critical path. Pancake's own host timer ("GPU-reach wait" vs "per-layer host") is the better critical-path instrument. The missing measurement is per-layer: pre-bell time, hit time, CPU share, and wait.

### Cited Findings
- The profiler run decodes at 79 tok/s vs ~126 without it, so the gap figures overstate a normal run. combine_cpu "spin-waits for the CPU expert rows: CPU time seen from the GPU" — [gpuprof README](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-09-7900xt-gpuprof/README.md)
- The host timer splits verify into GPU-reach wait vs per-layer host, and the GPU stage stamps (waitA, VRAM hits, waitB, waitCPU, copy+combine) come without profiler overhead — [profile-fasthip.txt](file:///home/dan/clone-of-d/src/Strata/bench/results/2026-11-08-7900xt-rounds/profile-fasthip.txt)
- Pancake has `wait_flag_ge_stamped_kernel` (GPU clock on entry/exit) and `STRATA_VERIFY_TRACE` breadcrumbs in mapped memory (verify.cpp:1170-1185; verify_kernels.cu:1387) (repo grep).

### Inferences
- **Use min-duration as the "work" part of wait-fused kernels.** For combine_cpu, min 5.6 µs vs average 254 µs separates work from waiting without new instrumentation. Report GPU busy as total − (calls × (avg − min)) for every kernel that contains a spin.
- **Per-layer critical-path log (F9 instrument).**
  - With WAIT_FUSE on, the stage stamp 23 no longer marks the end of the wait: the "waitCPU 0.12" figure on edits is not the real spin. A stamped variant (t_in/t_out from `wait_flag_ge_stamped`) inside `combine_cpu` thread 0 would give the true per-layer wait.
  - Joined with the host's per-layer CPU time, that yields the (hits − CPU) distribution that Q1, Q4 and Q6 need.
  - Effort S. It belongs to install I7 in the grouping report.
- **Profile edits and after-long-prompt windows (already I7)** with the same min/avg decomposition. On edits the spin should be ~275 µs/layer, and the hit kernels should drop out of the critical path entirely.

### Gaps
- No ROCm 10 documentation was consulted on rocprofv3's overhead per dispatch on gfx1100. The 79-vs-126 tok/s slowdown is the only local calibration.
