# GPU-side decode optimization on AMD RDNA3 (RX 7900 XT, gfx1100) for batch-1 speculative verify (1-8 rows)

Target used to judge applicability: pancake-strata (C++/HIP), Qwen3.8-Flash-Next 125B MoE (48 layers, GDN + sparse QSA attention, 4-stream hyper-connections, 512 experts top-10 + shared, MTP head), IQ2_XS/IQ4_XS/Q8_0. A verify window is a HIP graph of ~1000+ nodes at ~2.4 us per dependent node; window ~24 ms, of which ~5.5 ms is waiting on CPU expert rows; ~112 tok/s. A prior GDN 6-kernel-to-persistent fusion saved ~1.6% on multi-GPU and 0% on one GPU.

Research budget note: this was a 16-tool-call web survey. Two key pages (github.com/cutlerbenjamin1-cmd/qwen38-megakernel and github.com/nicejunjie/Strata-rdna3) timed out on fetch and could not be verified; the TreeWY PDF was not text-extractable. Statements marked "(background knowledge, unsourced)" come from general knowledge and should be verified before being relied on.

## 1. Kernel launch / HIP graph overhead on AMD

### Takeaway
No public RDNA3-specific per-node hipGraph measurement turned up; the engine's own ~2.4 us/dependent node is the best number available and is in line with NVIDIA graph figures (~1.3 us) plus the extra cost that AMD sources report. ROCm 7.2.x includes graph dispatch improvements (batched AQL submission, multi-list launch latency). Upgrading or retesting the ROCm version is a cheap first step. At ~1000 nodes x 2.4 us ≈ 2.4 ms (~10% of a 24 ms window), launch/dependency gaps set the ceiling on what kernel-count reduction can recover, and only on the GPU critical path.

### Cited Findings
- AMD's HIP docs say graphs reduce per-launch overhead by doing setup ahead of time and pay off only for repeated workloads. They describe the HIP graph API as beta, with some CUDA graph features unsupported, and give no microsecond figures. — [ROCm HIP graphs doc](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)
- ROCm 7.2.0 (Jan 2026) HIP runtime: "AQL batch submission approach" for graphs, with handling of varying AQL packet counts for memset nodes. — [ROCm 7.2.0 release notes](https://rocm.docs.amd.com/en/docs-7.2.0/about/release-notes.html)
- ROCm 7.2.1 fixed the batch-dispatch doorbell for AQL packets (potential CPU hang). — [ROCm changelog](https://rocm.docs.amd.com/en/docs-7.2.4/release/changelog.html)
- ROCm 7.2.4 (May 2026): "reduced hipGraphLaunch latency for multi-list graph topologies"; no figures given. — [ROCm 7.2.4 release notes](https://rocm.docs.amd.com/en/docs-7.2.4/about/release-notes.html)
- NVIDIA reference point: a dummy kernel costs ~2.1 us on a stream and ~1.3 us under CUDA graphs. Hazy estimates ~5 us of combined stall per kernel boundary (launch, tail, and load bubbles), which in their example cuts the H100 Llama-1B ceiling from ~1,350 to ~770 forward passes/s. — [Hazy Research "No Bubbles"](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- Secondary source: HIP graphs reportedly cut MI300 latency ~2x on short sequences, versus ~6% for CUDA graphs on H100 for the same kernel. This suggests launch overhead weighs more on AMD (MI300, not RDNA). — [Spheron blog](https://www.spheron.network/blog/triton-on-amd-rocm-vs-nvidia-cuda-same-kernel-different-perf/)
- Triton launch software overhead without graphs is ~100-300 us (~80 us with caching), GPU unspecified. — [arXiv 2511.11581](https://arxiv.org/pdf/2511.11581)
- Fleet (AMD Research) notes that captured graphs bake in batch-specific arguments, so mismatched shapes fall back to eager execution with multi-x latency spikes. — [Fleet, arXiv 2604.15379](https://arxiv.org/html/2604.15379v1)

### Inferences
- Upper bound on launch-overhead savings: if all ~1000 nodes were serially dependent at 2.4 us, gaps would total ~2.4 ms per window (~10% of 24 ms). Any node-count reduction (fusion or persistent kernels) only returns time on the GPU critical path. While the GPU is idle waiting on CPU experts (5.5 ms), shortening GPU work before the wait gives 0%. This matches the observed 0% single-GPU result.
- Cheap experiments, in order: (a) confirm the ROCm runtime includes the 7.2.x graph dispatch changes (memory says ROCm 10.0.0 is installed, which should include them; verify the node gap with rocprofv3 timestamps); (b) remove graph-internal memset/memcpy nodes, which become separate AQL packets or blits; (c) make the graph a single linear chain or explicitly parallel branches, so the runtime does not insert extra barrier-AND packets. (b) and (c) are unsourced suggestions grounded in the AQL batching notes above.
- GPU_MAX_HW_QUEUES and multiple HW queues matter only if parallel graph branches or streams are used (for example to overlap GPU dense work with CPU experts; see section 5). By default HIP maps streams round-robin onto a small number of HW queues (background knowledge, unsourced).

### Gaps
- No measured hipGraph per-node latency on gfx1100 found, nor measurements of direct AQL queue / KFD (hsa_queue) submission versus hipGraph on RDNA3. A local microbenchmark (N empty dependent kernels, graph versus stream versus raw HSA) is the only way to get this.
- No public data on barrier-packet cost or GPU_MAX_HW_QUEUES effects on RDNA3.

## 2. Persistent kernels / megakernels for decode

### Takeaway
Megakernels deliver 1.3-2.5x at batch 1 on small dense models, where launch, tail, and load bubbles are a large share of per-token time. Most of the gain comes from overlapping the next op's weight loads with the current op's tail (shared-memory paging, "productive spin" L2 prefetch), not just from removing launches. On RDNA3 the approach is feasible (cooperative launch exists, or use atomic-counter barriers as Alpindale and Fleet do), but no public RDNA3 LLM megakernel was found. On the target, a whole-window megakernel is high effort, and its gain is capped by the CPU-expert wait unless that wait is overlapped.

### Cited Findings
- Hazy "No Bubbles" (May 2025), Llama-3.2-1B: one interpreter megakernel with 7 instruction types and 13 x 16 KiB shared-memory pages so the next instruction's loads start early, synchronized by global-memory counters (the MLP down-projection is split into 4 counter chunks). Measured: H100 at ~78% of memory bandwidth, ~2.5x vLLM and >1.5x SGLang; B200 under 680 us/forward, >3.5x vLLM. Remaining B200 overhead: activation store/sync/load ~42%, setup ~13%, warp sync ~7%. — [Hazy Research](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
- Mirage Persistent Kernel (MPK, CMU, arXiv 2512.22219) compiles a model into a single megakernel with an in-kernel runtime over an SM-level task graph. The authors report 1.0-1.7x over SGLang/vLLM, single and multi-GPU (author-reported). — [MPK arXiv](https://arxiv.org/html/2512.22219v1); [mirage repo](https://github.com/mirage-project/mirage)
- AutoMegaKernel (arXiv 2606.09682) argues CUDA graphs amortize launches but leave inter-op HBM round trips. — [arXiv 2606.09682](https://arxiv.org/html/2606.09682v1)
- Fleet (AMD Research, arXiv 2604.15379), MI350X, Qwen3-8B bf16: a persistent kernel with one scheduler workgroup per XCD and atomic/event counters (two-level counting, at most one fence per XCD per event), not grid barriers. bs=1: vLLM eager 10.51 ms, chiplet-unaware Mirage-style baseline 7.83 ms, Fleet ~6.8 ms (1.54x over vLLM eager; ~1.15x over the megakernel baseline). The paper is internally inconsistent on L2 hit rates (abstract 12%→54%, table 38.9%→51.0%). It does not evaluate RDNA; the chiplet idea applies only to partitioned-L2 GPUs. — [Fleet](https://arxiv.org/html/2604.15379v1)
- Alpindale, RTX 5090, Qwen3-0.6B megakernel: cooperative `grid.sync()` costs ~3 us each (168/step); a custom atomic barrier ~2.2 us; flag-based partial syncs ~0.2 us. tok/s went 494 → 813 (tuning) → 890 (atomic barriers) → 905 (flags) → 1000 ("productive spin": idle blocks prefetch the next ~23 MB of weights into L2 during attention, the largest single win) → ~1033. Negatives: prefetching during bandwidth-saturated phases (-8.7%), evict-first hints (-8%), bulk prefetch (-5%), more blocks (-2%, more atomics). CUDA graphs gave no measurable difference. — [Alpindale blog](https://hn.jakechampion.name/reader/https://blog.alpindale.net/posts/5090_decode_optimization)
- Lucebox: a Qwen3.5-0.8B megakernel on an RTX 3090 (82 blocks x 512 threads, one persistent dispatch, cooperative grid sync); a separate DFlash speculative decode reports 3.43x over autoregressive. — [lucebox-hub](https://hn.jakechampion.name/reader/https://github.com/Luce-Org/lucebox-hub)
- Kog AI claims a 3,000+ tok/s/request "monokernel" on MI300X. Closed source and unverified (company and secondary coverage only). — [aiweekly](https://aiweekly.co/alerts/amd-mi300x-monokernel-hits-3300-tokens-per-second)
- HipKittens (Stanford Hazy + AMD, arXiv 2511.08083): a tile DSL for AMD whose BF16/FP8 GEMMs and attention are competitive with AITER assembly and 1.3-3.0x over Triton. Validated on CDNA3/CDNA4 only, so not directly usable on RDNA3. — [Hazy HK blog](https://hazyresearch.stanford.edu/blog/2025-11-09-hk); [arXiv](https://arxiv.org/abs/2511.08083v1)
- qwen38-megakernel (github.com/cutlerbenjamin1-cmd/qwen38-megakernel): the repo fetch timed out and a search found no description. Could not verify its "one cooperative kernel per speculative cycle" design. — (unverified)

### Inferences
- Mechanism transfer to RDNA3: Hazy-style counters and Alpindale-style atomic barriers need only global atomics plus a guarantee that all workgroups are co-resident (one workgroup per CU, or WGP, x 84 CUs). hipLaunchCooperativeKernel supports gfx11 (background knowledge, unsourced; verify with hipDeviceAttributeCooperativeLaunch). RDNA3 lacks TMA/async bulk copy, so "page-ahead" loading must be done by issuing global loads early into VGPRs/LDS (LDS 64 KB per workgroup) or by L2 prefetch with dummy loads. The 7900 XT's 80 MB Infinity Cache (MALL) plays the role of the 5090's 96 MB L2 in the "productive spin" trick: idle CUs can touch the next layer's dense weights (attention/GDN projections and the shared expert) while other CUs finish.
- Expected gain on the target: the 1.3-2.5x numbers come from 0.6-8B dense models whose whole token is ~1-10 ms of tiny kernels. Here GPU-resident work per window is roughly 24 - 5.5 = ~18.5 ms across ~1000 nodes, so gaps are at most ~13% of GPU time. A partial approach is realistic: per-layer persistent "super-kernels" for the GDN + QSA + shared-expert dense path, with counter syncs instead of graph edges, plus cross-op L2/MALL prefetch. It could recover ~1-2.4 ms (4-10% of the window), but only after the CPU-expert wait is hidden (section 5). This is consistent with the observed 1.6% from fusing a single GDN layer (one layer type out of 48 layers).
- A full "one kernel per speculative cycle" design is high effort in HIP: the MoE routing is data-dependent, the CPU expert handoff needs device-side polling of a host-written flag (fine-grained host memory), and register pressure is high. RDNA3 wave32 VGPR allocation would limit occupancy (background knowledge).

### Gaps
- No RDNA3 megakernel or persistent LLM decode measurements found. No measured grid-barrier cost on gfx1100.
- Could not read the qwen38-megakernel repo or confirm its numbers.

## 3. RDNA3 low-level: WMMA, dual-issue, dot instructions, caches, low-bit GEMV

### Takeaway
At batch 1 with 1-8 verify rows, the expert and projection matmuls are memory-bound GEMV/skinny GEMM. WMMA (16x16x16) wastes at least half its tile at M≤8, so the relevant tools are dp4a-style integer dot products (v_dot4 / sudot4 on gfx11) with MMVQ-style kernels that read each weight block once for all rows, plus maximizing effective bandwidth (vectorized 128-bit loads, MALL reuse). llama.cpp's RDNA3 work in 2025-26 targeted MMQ (batched) tuning and WMMA-MMQ, with gains concentrated at microbatch 8-16, which is exactly the verify-window regime.

### Cited Findings
- llama.cpp PR #17576 "HIP: enable WMMA-MMQ INT kernels for RDNA 3" (merged late 2025) enabled WMMA int kernels for most quant types on gfx1100. — [PR 17576](https://github.com/ggml-org/llama.cpp/pull/17576)
- llama.cpp PR #26284 "HIP: tune rdna 3 mmq configs" (merged Aug 30, 2026), 7900 XTX: up to ~1.67x on Q2_K_S at microbatch 16; most other results within 1-9%, some slightly slower; IQ1_M ~1.01x. — [PR 26284](https://github.com/ggerganov/llama.cpp/pull/26284)
- llama.cpp PR #24546 "CUDA: size routed MoE MMQ N-tiles from typical expert width on RDNA3" (open as of Jun 2026). — [llama.cpp PRs](https://github.com/ggml-org/llama.cpp/pulls)
- PR #18666 adjusted RDNA3.5 MMQ kernel selection (merged; no numbers seen). — [PR 18666 overview](https://app.semanticdiff.com/gh/ggml-org/llama.cpp/pull/18666/overview)
- Dual issue (VOPD): Chips and Cheese found compilers often fail to pack enough VOPD pairs, so peak FP32 was not reached in wave32 OpenCL; forcing wave64 helped. AMD docs say hardware accepts pairing only when register/opcode/operand constraints hold. — [C&C RX 7600](https://chipsandcheese.com/p/amds-rx-7600-small-rdna-3-appears); [ROCm profiler SOL doc](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/rdna/system-speed-of-light.html)
- Infinity Cache: Navi31 Infinity Cache latency is higher than Navi33's (7900 XTX ~58% longer than 7600). Per-WGP Infinity Cache bandwidth fell versus RDNA2. Power-state effects distort measurement (vector IC latency 199 ns → 150 ns in a higher power state). C&C notes a newer article re-evaluating Navi31 latency. — [C&C RDNA3 microbenchmarks](https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture); [C&C power saving](https://chipsandcheese.com/p/latency-testing-is-hard-rdna-3-power-saving)
- WMMA on RDNA3 supports FP16, BF16, INT8, INT4. — [Wikipedia RDNA 3](https://en.wikipedia.org/wiki/RDNA_3)
- On the 5090, L1-bypass weight loads helped while evict-first hints hurt (-8%). The analogous AMD knobs are glc/slc/nt cache policy bits. — [Alpindale](https://hn.jakechampion.name/reader/https://blog.alpindale.net/posts/5090_decode_optimization)

### Inferences
- For 1-8 verify rows, a "multi-row MMVQ" kernel is the right tool: dequantize each IQ2_XS/IQ4_XS block once into registers and dot it against all R rows' q8_1 activations with v_dot4_i32_iu8 (background knowledge: gfx11 has v_dot4_i32_iu8 and v_dot8_i32_iu4, not the gfx9 sudot4 naming). Weight traffic stays constant as R grows, so verify cost per row falls almost linearly up to the compute limit. Expected: if the engine currently launches per-row or uses a WMMA path that pads M to 16, a multi-row MMVQ could cut those GEMV nodes 1.3-2x (llama.cpp's mmvq handles ncols ≤8 this way; background knowledge, verify in ggml-cuda/mmvq.cu). If it already does this, the gain is ~0.
- IQ2_XS dequant uses codebook (grid) lookups. Keep the grid tables in LDS or constant memory and use DPP/permlane for cross-lane reductions instead of LDS round trips (background knowledge). Measured gain on the target is unknown.
- MALL (80 MB on the 7900 XT, per the task) cannot hold per-window weights (the active set is GBs), but it can hold the GDN recurrent states, KV/QSA indices, hyper-connection streams, and inter-kernel activations. Marking streaming weight loads non-temporal (slc/nt) to avoid evicting those is a cheap experiment. Expect low single-digit %. Alpindale's evict-first result (-8%) shows it can backfire, so A/B test it.
- s_prefetch_data is a gfx12 (RDNA4) instruction, not available on gfx1100 (background knowledge, unsourced). Prefetching on RDNA3 means issuing early loads, or "productive spin" dummy loads into L2/MALL.
- WMMA becomes useful only for batched prefill or for GDN state updates (the 128x128 per-head state matrix times k/v vectors over T≤8 tokens can be cast as small matmuls).

### Gaps
- Could not fetch nicejunjie/Strata-rdna3 (timeout), so its RDNA3 kernel choices and tok/s are unknown.
- No RDNA3 IQ-format MMVQ PR with numbers found for 2026. No WMMA latency measurements found.
- No public LDS bandwidth or dual-issue measurements for GEMV-style integer kernels on gfx1100.

## 4. Gated DeltaNet decode/verify kernels and state handling

### Takeaway
The state of the art for GDN verify over T=2-8 draft tokens is a single fused kernel per layer: read the state once, apply T sequential delta-rule updates in registers/LDS, emit outputs, and either write intermediate states (checkpoints) for rollback or keep the checkpoint frozen and replay or commit the accepted prefix after acceptance (FlashInfer ReplaySSM, "Trees from Marginals"). TreeWY extends the WY/UT transform so one triangular solve verifies a whole draft tree. On the target, GDN state traffic is a modest share. The win is fewer nodes and avoiding writing T full states per layer.

### Cited Findings
- FlashInfer `gated_delta_rule_mtp`: a fused decode kernel optimized for speculative verification (T>1). No BF16 fast path: BF16 state pools should use `gated_delta_rule_decode_pretranspose` (supports T=1 and T>1). The SM100 tcgen05 TF32 specialization requires T=4..8, B=1..256, a contiguous checkpoint, and Q/K normalization; other cases fall back to FP32 MTP. — [FlashInfer gdn_decode docs](https://docs.flashinfer.ai/api/gdn_decode.html); [gated_delta_rule_mtp](https://docs.flashinfer.ai/generated/flashinfer.gdn_decode.gated_delta_rule_mtp.html)
- FlashInfer ReplaySSM: verify leaves checkpoints frozen; once acceptance is known, call `gated_delta_rule_replayssm_commit` once, then overwrite the windows during the next verify (SM100/SM103 only). — [replayssm_commit](https://docs.flashinfer.ai/generated/flashinfer.gdn_decode.gated_delta_rule_replayssm_commit.html)
- FlashInfer `cake_gated_delta_product_mtp`: one fused kernel per call over an FP32 state pool. It reads the state from a pool slot, applies per-token decay and updates, writes back after the last token, computes the gate in-kernel, and skips padding rows marked by negative indices. — [cake_gdp_mtp](https://docs.flashinfer.ai/generated/flashinfer.cake_gdp.cake_gated_delta_product_mtp.html)
- TensorRT-Edge-LLM has a GatedDeltaNet plugin (C++ API). — [TRT-Edge-LLM GDN plugin](https://nvidia.github.io/TensorRT-Edge-LLM/0.10.0/cpp_api/plugins/gatedDeltaNetPlugin.html)
- TreeWY (arXiv 2608.20961): extends DeltaNet's WY/UT transform to the whole draft tree so one triangular solve verifies every node. The accepted node's continuation state is rebuilt from solved values by summing over its ancestors. Numbers not extracted (PDF not readable). — [TreeWY](https://arxiv.org/pdf/2608.20961)
- "Trees from Marginals" (arXiv 2607.06763) includes a fused GDN tree-verification kernel that avoids speculative state updates during verify and replays the accepted path on commit. — [arXiv 2607.06763](https://arxiv.org/pdf/2607.06763)
- LumoTree (listed under "GDN Tree-Scan", arXiv 2609.23900): path-parallel tree verification serving for recurrent hybrids (indexing may be off). — [arXiv 2609.23900](https://arxiv.org/pdf/2609.23900)

### Inferences
- Exactness: replay/commit and per-token checkpointing are both exact (same recurrence order). Chunked WY/UT forms are mathematically equivalent but change float summation order, so they are not bit-exact versus the sequential recurrence. That matters if outputs are compared token-by-token against non-speculative decode.
- On the target (linear chain from the MTP head, T≤8), the fused-verify kernel from FlashInfer is the pattern to port to HIP. One kernel per GDN layer covers conv1d + gates + T-step recurrence + norm + output gate, with states kept FP32 in registers (a 128x128 state per head spread across a workgroup). Write back only (a) the final state and (b) the minimal k/v/g/beta per token needed to replay, instead of T full state checkpoints. Doing the commit replay lazily inside the next window's first GDN kernel avoids an extra node. This is the same direction as the prior "6 kernels → 1 persistent" experiment. Expected incremental gain beyond that experiment is small unless state checkpoint writes are currently T x full-state (then memory traffic per GDN layer drops ~T-fold for the state part).
- Tree verification (TreeWY) matters only if the engine moves from chain MTP drafting to trees. Not applicable today.

### Gaps
- Could not obtain measured speedups for TreeWY, the NVIDIA Blackwell GDN report (arXiv 2607.16831, not searched due to budget), or the SemiAnalysis Qwen3.5 GDN notes.
- No AMD/HIP GDN decode kernels found (fla Triton kernels may run on ROCm via Triton-AMD; not verified).

## 5. Overlapping GPU work with CPU expert computation

### Takeaway
This is the gating issue for the target. With ~5.5 ms/window of GPU idle waiting on CPU expert rows, GPU kernel optimizations show up only once that wait is overlapped. Published hybrid systems (KTransformers Expert Deferral, HybriMoE intra-layer scheduling) restructure the layer so the GPU computes the shared expert, GPU-resident (hot) experts, and next-layer dense work while the CPU computes cold experts. HybriMoE reports 1.70x decode over the prior hybrid state of the art.

### Cited Findings
- KTransformers (SOSP '25): hybrid systems are bottlenecked by CPU compute and CPU-GPU synchronization. It uses AMX CPU kernels plus asynchronous CPU-GPU scheduling, and an "Expert Deferral" mechanism overlaps CPU and GPU work, raising CPU utilization from below 75% to nearly 100%. Hot experts run on GPU, cold on CPU (profiling-based partition, following Fiddler). — [KTransformers paper](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf); [repo](https://github.com/kvcache-ai/ktransformers)
- HybriMoE (built on KTransformers): dynamic intra-layer CPU-GPU scheduling, inter-layer prefetching, and score-based caching. Reports 1.33x prefill and 1.70x decode over the prior hybrid SOTA (author-reported). — summarized in [KTransformers page](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/)
- Alpindale: CUDA graphs gave no gain because per-step stream synchronization serialized decode. A host sync in the critical path dominates launch savings. — [Alpindale](https://hn.jakechampion.name/reader/https://blog.alpindale.net/posts/5090_decode_optimization)

### Inferences
- Mechanism for the HIP engine, without a host round trip per layer: split each MoE layer into (a) the GPU part, which is router, shared expert, GPU-cached experts, and a partial sum, and (b) the CPU part, cold experts. The GPU writes the router top-k to pinned host memory, the CPU polls and computes, and the CPU writes results plus a flag to fine-grained host memory or the device. Inside the graph, a tiny kernel spins on the flag before the combine. Meanwhile the GPU proceeds with independent work. Within one layer that is the shared expert and hot experts. Across layers, the next layer's dense work cannot start before this layer's MoE output under strict data dependence, unless deferral is used.
- Expert deferral (KTransformers): add the CPU experts' contribution one layer late (the next layer's residual), so layer L+1's attention/GDN runs on the GPU while the CPU finishes layer L's cold experts. This is not exact (it changes the model's computation) and must be validated for accuracy. In a verify window it changes outputs relative to the non-speculative path, so acceptance semantics stay consistent only if drafting uses the same approximation.
- Exact alternative: pipeline across the T verify rows or across micro-chunks of experts. Start CPU work on rows whose routing is ready while the GPU computes other experts. Use two graph branches on separate HW queues (requires GPU_MAX_HW_QUEUES ≥ 2) so the spin-wait kernel does not block independent GPU work. A spin kernel occupies a CU, so give it one tiny workgroup.
- Expected gain on the target: if half the 5.5 ms CPU wait can be overlapped with GPU dense work, the window drops ~2.75 ms (~11%), and GPU-side gains from sections 1-4 (an additional ~4-10%) start to count. Without overlap, sections 1-4 yield ~0% on one GPU, as observed. Priority order: (1) CPU/GPU overlap restructuring (exact variant first); (2) multi-row MMVQ for verify rows and hot experts on the GPU; (3) fused GDN verify with lazy commit; (4) per-layer persistent super-kernels with MALL prefetch; (5) a full megakernel only if (1)-(4) leave launch gaps dominant.

### Gaps
- No published hybrid CPU-GPU MoE system on AMD/ROCm with in-graph device-side flag polling was found. No data on hipGraph behaviour with spin-wait kernels on host-coherent memory on RDNA3.
- HybriMoE numbers are secondary and author-reported. The baseline and hardware were not confirmed.
