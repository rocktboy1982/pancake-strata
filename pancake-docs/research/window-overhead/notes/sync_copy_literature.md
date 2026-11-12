# GPU sync and host-copy reduction in speculative-decoding verify windows

Scope: literature up to 2026-11-12. Labels: [paper] = arXiv/conference paper with id or URL; [blog/doc] = vendor, project, or official documentation; [search snippet] = seen only in search results, not fetched; [unverified] = claim not confirmed in a primary source. Search budget was about 17 tool calls, so coverage is partial; gaps are listed per section.

## Key Question 1: Which published systems (2024-2026) report per-step overhead reduction for speculative decoding on GPU, and by how much?

### Takeaway
Few papers isolate the per-window sync/copy cost. The measured wins come mostly from overlapping or restructuring the draft-verify loop (Mirror-SD, AMUSD, PipeSpec) and from fusing sampling and verification into fewer kernels (SonicSampler). Per-window host-overhead numbers are mostly from blogs or snippets, not controlled papers.

### Cited Findings
- Mirror-SD removes the serial draft-then-verify dependency by launching branch-complete rollouts from early-exit signals alongside the target's suffix; reports 2.8x-5.8x wall-time on SpecBench (14B-66B models), averaging a 30% relative gain over EAGLE3 — [paper arXiv:2510.13161](https://arxiv.org/abs/2510.13161). The abstract does not state lossless operation; it says speculative streaming works "without weakening acceptance semantics" — same source.
- AMUSD decouples drafting and verification into asynchronous processes on separate devices, uses a shared draft buffer with a rollback flag and no locks; reports an average 29% gain over standard speculative decoding with identical output quality — [paper arXiv:2410.17375](https://arxiv.org/abs/2410.17375) (figures from search excerpts, [unverified] against full text).
- PipeSpec generalizes this to k models in a hierarchical pipeline with optimistic downstream assumption and rollback; reports 2.25 vs 1.5 tokens per unit time — [paper arXiv:2505.01572](https://arxiv.org/html/2505.01572v1) ([unverified], from excerpt).
- Speculative Aggregation asynchronously verifies drafts across cloud and device and rolls back only on rejection; claims equivalence to synchronous decoding — [paper arXiv:2504.11197](https://arxiv.org/html/2504.11197v2) ([unverified], from excerpt).
- SonicSampler fuses the full sampling pipeline, including speculative verification, into a single batched kernel; reports up to 10x for its top-k and up to 16x over "state-of-the-art baselines" on heterogeneous spec-decode workloads. Baselines and exactness are not stated on the abstract page — [paper arXiv:2607.20475](https://arxiv.org/abs/2607.20475) (submitted 2026-05-24).
- A production evaluation (search snippet) reports scheduling and CUDA-graph launch overheads at 3-12% of runtime, shrinking with batch size, and verification as the dominant cost — [search snippet, chatpaper.com/paper/227911](https://chatpaper.com/paper/227911) ([unverified]).
- Ling-3.0-flash on 4 Blackwell GPUs at batch 1 (search snippet): once host work is hidden, the remaining cost is several hundred 1.5-6 us kernel nodes whose launch floor rivals their arithmetic; profiled step 5.2 ms vs back-computed 4.9 ms, showing profiling distortion — [search snippet, winzheng.com mirror of LMSYS 2026-08-21 post](https://www.lmsys.org/blog/2026-08-21-ling3-flash-spec-decode-blackwell) ([unverified], not fetched).
- SGLang roadmap slides list a "zero-overhead speculative decoding" scheduler item (stated ~10-20% speedup) and a "Spec V2" in progress — [roadmap slides](https://docs.google.com/presentation/d/10KppJL5MolyC2GTcmjNYVDneUQmkwQkA/htmlpresent) ([unverified], intent only).
- SGLang's blog on CUDA graphs (2026-08-17) describes the verify path as the decode runner capturing more than one token per request, with per-runner capture backends; no overhead numbers for verify — [blog/doc, lmsys.org/blog/2026-08-17-advanced-cuda-graph](https://www.lmsys.org/blog/2026-08-17-advanced-cuda-graph). Adaptive-verify work pre-captures a set of verify graphs for different verify lengths — [paper arXiv:2605.00342](https://arxiv.org/pdf/2605.00342) ([search snippet], [unverified]).
- vLLM v0.6.0 moved output processing to async to overlap CPU scheduling with GPU compute — [blog/doc, vllm.ai/blog/perf-update](https://vllm.ai/blog/perf-update). A third-party page claims vLLM 0.19 added zero-bubble async scheduling with speculative decoding — [unverified, autokaam.pages.dev](https://autokaam.pages.dev/news/vllm-v0-19-0-zero-bubble-scheduling-gemma-4/). An Alauda doc says async scheduling is disabled with N-gram drafting because proposer CPU work made it slower overall — [doc](https://docs.alauda.io/ai/2.8/deploy/inference_service/guides/vllm_speculative_decoding.html) ([unverified]).

### Inferences
- The largest documented gains come from removing the draft-to-verify serial dependency (Mirror-SD, AMUSD, PipeSpec), not from shaving one sync. A per-window readback removal alone is a smaller, unmeasured term in these sources.
- Consumer-hardware evidence points the same way: [paper arXiv:2607.17283](https://arxiv.org/abs/2607.17283) (Chordiya, 2026-07-19, Apple silicon) reports a best measured 1.61x at K=6, and says three of five configurations slowed down because the draft framework overhead can exceed a small target's whole decode step, and because the quantized Metal backend verifies serially below batch 8. The full text was not read, so the batch-8 and overhead details are from the abstract only ([unverified] for detail).

### Gaps
- No controlled paper found that isolates the sync-plus-readback time per verify window on CUDA or HIP with and without graph capture.
- No speculative-decoding-specific ablation found for the SGLang overlap scheduler or vLLM async scheduling.
- Mirror-SD and SonicSampler full texts were not read; baselines and exactness are unconfirmed.

## Key Question 2: Which techniques remove host readbacks entirely (device-side accept/reject, device-side draft selection)?

### Takeaway
Device-side acceptance and sampling exists in production kernel libraries and in SonicSampler, but I found no paper that quantifies removing the last host readback in a single-GPU loop. The NVIDIA CUDA mechanisms for device-side waits (stream wait-value and write-value) are documented but not evaluated for speculative decoding.

### Cited Findings
- FlashInfer's chain speculative sampling is a fused GPU kernel for chain-shaped drafts — [library doc, docs.flashinfer.ai](https://docs.flashinfer.ai/generated/flashinfer.sampling.chain_speculative_sampling.html). Community package spec-decode-ops claims fused verify on Ampere+ ([unverified], [HF kernels hub](https://huggingface.co/kernels/phanerozoic/spec-decode-ops)).
- SonicSampler's abstract says verification runs inside the same kernel as sampling ([paper arXiv:2607.20475](https://arxiv.org/abs/2607.20475)).
- CUDA driver API: cuStreamWaitValue32 / cuStreamWriteValue32 (and the batched form) let work queued after the call block on a memory location until a condition holds, so the CPU thread does not block. Pinned or registered host memory must use the device pointer from cuMemHostGetDevicePointer. The docs warn of deadlock on misuse, and batching the calls reduces API and device overhead — [CUDA driver API, Stream Memory Operations](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEMOP.html). This is a host-side enqueue primitive, not a kernel-side one; I did not find a reference for a kernel writing the flag that a later stream op waits on.
- Mapped-memory small readbacks are a documented CUDA pattern, but I found no speculative-decoding study on them in this search.
- Megakernel work replaces kernel-boundary sync with explicit counter-based sync inside one persistent kernel. AutoMegaKernel (AMK) reports counter-based synchronization with formal deadlock checks, and notes grid-wide sync costs are amortized at large model size but can limit small ones ([search snippet, AMK; arXiv id not confirmed]). ForgeMegakernel, a framework for decode megakernels, uses a launch-count-equals-one check ([paper arXiv:2609.12379](https://arxiv.org/pdf/2609.12379), [unverified] from snippet). Mirage Persistent Kernel reports 1.2x-6.7x latency reduction for multi-GPU inference ([project, github.com/tiledsl/mirage](https://github.com/tiledsl/mirage), [unverified]). These are non-speculative decode results; none measured a verify window.

### Inferences
- A device-side accept/reject with the next draft chosen on-device removes the per-window readback; the host then only consumes results asynchronously (see Q3). Exactness is preserved if the device logic is the standard rejection-sampling or greedy-argmax comparison.
- Graph-captured verify (Q1, SGLang) cannot contain data-dependent host branching, so accept length must feed back via device tensors; this is consistent with the device-side approach but is not stated in any source found.

### Gaps
- No paper quantifies the latency of one mapped-memory readback versus a device-side accept for a single window on RDNA or CUDA.
- No HIP equivalent of cuStreamWaitValue32 was checked; HIP documentation was not searched (hipStreamWaitValue32 existence is [unverified] from memory).

## Key Question 3: Which are exact (output-identical for greedy decoding) and which lossy?

### Takeaway
Standard greedy verify (accept while the draft token equals the target argmax) and standard rejection sampling are exact by construction, so the sync/copy changes below are exact if they do not change the decision logic. Mirror-SD, AMUSD, PipeSpec and Speculative Aggregation are claimed to preserve output semantics, but only AMUSD states identical output in the excerpt. Lossy variants are the adaptive or sparse verifications (SSV, adaptive verification such as DSpark), which change the accepted set.

### Cited Findings
- AMUSD reports identical output quality — [paper arXiv:2410.17375](https://arxiv.org/abs/2410.17375) ([unverified] against full text).
- Speculative Aggregation claims equivalence to synchronous decoding — [paper arXiv:2504.11197](https://arxiv.org/html/2504.11197v2) ([unverified]).
- Chordiya reports a two-sample test over about 9,200 tokens per method with chi-square 162.5, dof 200, p = 0.976, supporting distribution equivalence for the tested configs — [paper arXiv:2607.17283](https://arxiv.org/abs/2607.17283).
- SSV (sparse speculative verification) changes attention during verify, so it is lossy-by-design in the search snippet — [paper arXiv:2605.19893](https://arxiv.org/pdf/2605.19893) ([unverified], snippet only).
- Confidence-scheduled adaptive verification in vLLM (DSpark) and MoE adaptive verification trade verify length for acceptance — [blog/doc, vllm.ai/blog/2026-08-14-dspark-adaptive-verification](https://vllm.ai/blog/2026-08-14-dspark-adaptive-verification) ([unverified], not fetched).

### Inferences
- Exact class: device-side greedy compare, FlashInfer/SonicSampler-style fused rejection sampling with the same RNG stream, graph capture of a fixed-shape verify, async readback with rollback-on-reject (AMUSD-style). Verify output is bit-identical only if the kernel reduction order matches; a fused kernel can change floating-point argmax ties, so check on logits near ties.
- Lossy class: adaptive verify length, sparse attention verify, and any change to acceptance thresholds.

### Gaps
- No source found reports bit-exact greedy equality for a fused verify kernel versus the unfused reference across backends.

## Technique catalogue: mapping to per-window cost

Per-window cost is taken as: sync waits + host readbacks + kernel launches + host-side scheduling between draft and verify. Evidence grade in brackets.

- (a) Wait-flag / event sync between kernels: replaces host round trips with device-side ordering. Mechanism: cuStreamWaitValue32 or events so the next kernel waits without CPU involvement [CUDA driver doc, above]. Maps to: removes the host wait on a flag; no speculative-decoding measurement found. Exact.
- (b) Small host readbacks after draft selection via mapped memory: maps the accept length to host without a full synchronize. Mechanism: pinned or mapped memory written by the device. Maps to: one small copy per window. Only the CUDA API is documented here; no measured gain found. Exact (data is unchanged).
- (c) Fusing verify and selection in one kernel: SonicSampler [paper arXiv:2607.20475] and FlashInfer chain sampling [library doc]. Maps to: removes kernel launches and inter-kernel sync. Reported speedups are for sampling, not whole-window latency. Exactness depends on kernel details (see Q3).
- (d) CUDA/HIP graph capture of the window: SGLang verify graphs per verify length [blog/doc, lmsys 2026-08-17; paper arXiv:2605.00342 snippet]; Ling-3.0 kernel-launch floor [search snippet]. Maps to: cuts launch overhead per node. Padding can add cost as a step function [search snippet, vLLM DSpark]. Exact for fixed shapes.
- (e) Asynchronous readback with pipelined next-window drafting: Mirror-SD [paper arXiv:2510.13161], AMUSD [paper arXiv:2410.17375], PipeSpec [paper arXiv:2505.01572], SGLang overlap scheduler [roadmap slides, unverified], vLLM async scheduling [unverified third-party claim]. Maps to: hides host work and readback behind the next window. Mirror-SD and PipeSpec are the strongest reported wall-time gains, but they change the dataflow and need rollback.

## Relevant failure mode from the search (single-GPU shared-device draft/verify)

- llama.cpp issue #23126 (opened 2026-05-16, closed "not planned" 2026-06-30, no maintainer fix): on an AMD Radeon 780M (gfx1103) iGPU, Vulkan only, unified memory, target gemma-4-31B Q6_K plus draft gemma-4-E2B Q5_K_M on the same GPU: draft alone 33-35 t/s, target alone 3-7 t/s, target plus draft 3.2-4.9 t/s, with draft evaluation about 43,000-73,000 ms per run versus about 30 ms. Reporter's hypothesis (unconfirmed): Vulkan queue serialization on UMA, with cross-model waits and cache flushes — [GitHub issue ggml-org/llama.cpp#23126](https://github.com/ggml-org/llama.cpp/issues/23126). Relevant as a cautionary case: cross-model waits on one device can dominate.
- DEV Community write-up on llama.cpp Metal (search snippet, [unverified]): about 87% acceptance but only 0.95x speedup, which the author attributes to strictly sequential draft and verify round trips. This is the author's own analysis, not a measured llama.cpp finding; source URL not confirmed (a dev.to result, [dev.to/rosgluk](https://dev.to/rosgluk/speculative-decoding-20-50-faster-llm-inference-g11), appeared in the same search but its match to this claim was not checked).

## Overall gaps for the owner
- No 2024-2026 paper isolates the per-window sync and copy cost on a single consumer GPU under HIP/ROCm (RX 7900 XT or RDNA iGPU). The closest evidence is the llama.cpp Vulkan issue and the Apple-silicon paper.
- Not checked: HIP graph capture and hipStreamWaitValue32 behavior on gfx1100; vLLM and SGLang source code for their spec-decode proposer paths; Mirror-SD and SonicSampler full texts.
- Several sources above are search snippets and third-party blogs; claims marked [unverified] should not be cited in the paper series until checked against the primary paper or release notes.
