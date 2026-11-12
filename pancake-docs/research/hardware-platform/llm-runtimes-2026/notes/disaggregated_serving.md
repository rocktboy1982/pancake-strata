# Disaggregated Prefill/Decode LLM Serving (state as of November 2026)

Scope: survey, techniques, mechanisms, implementations/code, applicability. Research date 2026-11-08. Each named system is labelled VERIFIED / PARTIALLY ACCURATE / NOT FOUND relative to the pasted AI-written overview's claims.

## Q1. Do the claimed systems exist, and what are they? (llm-d 1.0, vLLM disagg prefill, NVIDIA Dynamo, Mooncake, DistServe, Splitwise, SGLang PD)

### Takeaway
All seven names are real. "llm-d 1.0" exists and is recent: it was released 20 Oct 2026 by the llm-d community, which Red Hat, Google, IBM and others founded in May 2025. It is a Kubernetes-native layer that sits above vLLM/SGLang, not an engine. DistServe (OSDI'24) and Splitwise (ISCA'24) are 2023-2024 research. Mooncake (Moonshot AI, FAST'25 Best Paper) is now a broad KV-transfer/storage substrate used by vLLM, SGLang, Dynamo, TensorRT-LLM and LMCache.

### Cited Findings
**llm-d: VERIFIED (watch for attribution errors)**
- The project was announced 20 May 2025 ("Announcing the llm-d community!"). The authors were Robert Shaw (Red Hat), Clayton Coleman (Google) and Carlos Costa (IBM). It is built on vLLM, Kubernetes and the Inference Gateway (IGW), which extends the Gateway API with inference routing. Its scheduler uses IGW's Endpoint Picker Protocol (EPP) with filters and scorers for disaggregation, prefix-cache awareness and load. P/D disaggregation goes through vLLM's KV Connector API with transports such as NIXL. — [llm-d announce](https://llm-d.ai/blog/llm-d-announce)
- Secondary sources say the founders were Red Hat, IBM, Google Cloud, CoreWeave and NVIDIA. One secondary article attributes v1.0 to "Google" alone and another to IBM Research. That conflict is an artifact of secondary sources: the official post is signed "the llm-d maintainers". — [search summary: xenospectrum](https://xenospectrum.com/en/llm-d-production-inference-operations/), [marktechpost](https://www.marktechpost.com/2026/10/20/llm-d-1-0-open-source-distributed-inference-stack/)
- Release history: v0.2 "Our first well-lit paths" (29 Jul 2025), v0.3 (10 Oct 2025), v0.4 (2 Dec 2025), v0.5 (4 Feb 2026), v0.7 (12 May 2026), v0.8 (24 Jun 2026), v0.9 (28 Aug 2026), v1.0 "State-of-the-art inference, ready for Day 2" (20 Oct 2026). — [llm-d releases tag](https://llm-d.ai/blog/tags/releases)
- v1.0 is the first stable v1 configuration API. It was announced alongside PyTorch Conference NA, with more than 500 contributors from more than 25 organizations. — [llm-d v1.0 post](https://llm-d.ai/blog/llm-d-v1.0-state-of-the-art-inference-ready-for-day-2); [marktechpost via search](https://www.marktechpost.com/2026/10/20/llm-d-1-0-open-source-distributed-inference-stack/)
- v1.0 features (all from the official post):
  - Multi-tenant QoS: centralized `flowControl` with flows keyed by fairness ID and priority band, plus a probabilistic admission controller. The default stays FCFS.
  - EPP hot standby.
  - Session-aware routing that stays sticky to the replica holding a conversation's KV until that replica saturates.
  - Native Responses API state (`previous_response_id`).
  - Topology-aware scheduling via DRA/DRANET.
  - P/D disaggregation: a "DisaggregatedSet" manages prefill and decode replicas as one unit, with revision-aware routing and lockstep in-place upgrades. A router-side coordination layer, consolidated into "llm-d-router", drives each request through encode, prefill and decode.
  - KV transport: NIXL on NVIDIA (contributed via Dynamo) and on AWS (over EFA); MoRI-IO plus NIXL on AMD ROCm; "TPU Sync" on Google TPUs.
  - Engines: vLLM, SGLang, AMD ATOM, and PyTorch/TorchTPU.
  - Hardware: H100/H200/B200/GB200 NVL72/GB300, MI300X/MI355X, TPU 7x, and Intel.
  — [llm-d v1.0 post](https://llm-d.ai/blog/llm-d-v1.0-state-of-the-art-inference-ready-for-day-2)
- Earlier llm-d descriptions (Red Hat/Solo.io, 2025) say a sidecar next to the decode pod coordinates the KV transfer. v1.0 describes the coordination as router-side (llm-d-router). — [Solo.io deep dive](https://www.solo.io/blog/deep-dive-into-llm-d-and-distributed-inference); [llm-d v1.0 post](https://llm-d.ai/blog/llm-d-v1.0-state-of-the-art-inference-ready-for-day-2)

**vLLM disaggregated prefill: VERIFIED (still labelled "experimental" in the docs)**
- Docs list these connectors: ExampleConnector, LMCacheConnectorV1, LMCacheMPConnector (a standalone `lmcache server` shared by many vLLM instances), NixlConnector (fully async send/recv), MooncakeConnector, ECMooncakeConnector (multimodal encoder cache), MoRIIOConnector (ROCm only), MultiConnector, OffloadingConnector (CPU offload) and FlexKVConnectorV1. P2pNcclConnector was not on the current page. — [vLLM docs: disagg_prefill](https://docs.vllm.ai/en/stable/features/disagg_prefill/)
- Configuration is `--kv-transfer-config` JSON with `kv_connector`, `kv_role` (e.g. `kv_both`), `kv_buffer_device` and `kv_connector_extra_config`. NixlConnector can use several NIXL backends, e.g. `"backends":["UCX","GDS"]`. — [vLLM docs](https://docs.vllm.ai/en/stable/features/disagg_prefill/); [search excerpt](https://docs.vllm.ai/en/v0.23.0/features/disagg_prefill/)
- vLLM's own docs say verbatim: "Disaggregated prefill DOES NOT improve throughput." The stated benefits are tuning TTFT and ITL separately and controlling tail ITL. — [vLLM docs](https://docs.vllm.ai/en/stable/features/disagg_prefill/)

**NVIDIA Dynamo: VERIFIED**
- Open-source (GitHub `ai-dynamo/dynamo`, Rust + Python). It was announced at GTC in March 2025, and the docs are now at v1.x (v1.5.1 is the latest referenced). — [Dynamo docs v1.2.0](https://docs.nvidia.com/dynamo/v1.2.0/design-docs/disaggregated-serving.md); [NVIDIA dev blog intro](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models); [GitHub](https://github.com/ai-dynamo/dynamo)
- Request flow:
  - PrefillRouter picks a prefill worker by KV-aware score (cache overlap plus load) or by plain load balancing.
  - The prefill worker returns `disaggregated_params` (transfer metadata), which is injected into the decode request before it is routed to a decode worker.
  - NIXL moves KV directly from prefill VRAM to decode VRAM over the best transport (NVLink, or InfiniBand/UCX). The transfer is non-blocking.
  - Runtime-reconfigurable xPyD: workers register or drain dynamically.
  - Backends differ. SGLang uses `bootstrap_info` and prefill can run in the background, so decode starts during the transfer. vLLM (`kv_transfer_params`) and TRT-LLM (`opaque_state`) run prefill synchronously.
  — [Dynamo disaggregated-serving design doc](https://docs.nvidia.com/dynamo/v1.2.0/design-docs/disaggregated-serving.md)
- The KV-aware router keeps a global radix-tree registry of cached blocks to compute hit rates and balance load. — [Dynamo architecture v0.9.0 (search excerpt)](https://docs.nvidia.com/dynamo/v-0-9-0/design-docs/overall-architecture)

**Mooncake: VERIFIED**
- "KVCache-centric disaggregated architecture", the serving platform for Moonshot AI's Kimi. It splits prefill and decode clusters and pools idle CPU, DRAM and SSD into a disaggregated KV cache. It won the FAST 2025 Best Paper (25 Feb 2025) and joined the PyTorch Ecosystem on 12 Feb 2026. Apache-2.0. — [GitHub kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)
- The arXiv page (2407.00079, v5 revised Oct 2026) adds a KVCache-centric scheduler and prediction-based early rejection under overload. The arXiv abstract page itself names no venue. — [arXiv 2407.00079](https://arxiv.org/abs/2407.00079)

**DistServe: VERIFIED (2024 research)**
- "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving", OSDI 2024 (Peking University / UCSD Hao AI Lab). — [arXiv 2401.09670](https://arxiv.org/abs/2401.09670)

**Splitwise: VERIFIED (2023-2024 research; Microsoft authors)**
- "Splitwise: Efficient generative LLM inference using phase splitting". Authors: Patel, Choukse, Zhang, Shah, Goiri, Maleki, Bianchini. First posted Nov 2023. The arXiv page lists no venue; the ISCA'24 attribution is widely cited but was not confirmed on that page. — [arXiv 2311.18677](https://arxiv.org/abs/2311.18677)

**SGLang PD disaggregation: VERIFIED**
- Separate prefill and decode servers (`--disaggregation-mode prefill|decode`). Transfer backends are Mooncake (RDMA, `mooncake_tcp`, AWS EFA, Intel XPU), NIXL (UCX by default; `SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC` selects others) and Ascend.
- A staging buffer bridges different TP sizes on the two sides (non-MLA models only).
- The router is "SGLang Model Gateway (former Router)": `sglang_router.launch_router --pd-disaggregation --prefill ... --decode ...`.
— [SGLang docs: PD disaggregation](https://docs.sglang.io/advanced_features/pd_disaggregation.html)

### Inferences
- If the pasted overview describes llm-d 1.0 as a new inference *engine*, or attributes it to a single company, that is inaccurate. It is a community-governed, Kubernetes-native distributed-serving control plane that orchestrates vLLM/SGLang. If it claims a "sidecar" design for 1.0, that is partially outdated: v1.0 consolidates coordination into llm-d-router.
- Any claim that vLLM disaggregation boosts raw throughput contradicts vLLM's own docs. The gains are in latency SLOs and goodput, not peak tokens/s.

### Gaps
- I could not fetch the full marktechpost article (HTTP 403). The founding-company list (CoreWeave, NVIDIA) comes from search snippets, not a primary page.
- I did not confirm Splitwise's ISCA'24 venue on a primary page (arXiv lists none).

## Q2. Mechanisms: why split, and how the KV cache moves

### Takeaway
Prefill processes all prompt tokens in parallel, so it is compute-bound. Decode produces one token per step per sequence and must re-read weights and the whole KV cache, so it is memory-bandwidth-bound. Colocating them means a long prefill stalls decode steps, which inflates TPOT, and both metrics must be provisioned for at once. Disaggregation removes this but adds a KV transfer. That transfer is done GPU-to-GPU over RDMA/NVLink via NIXL or the Mooncake Transfer Engine, and it is routed by KV/prefix-aware schedulers.

### Cited Findings
- Dynamo docs: splitting lets each phase use different parallelism, e.g. larger TP for memory-bound decode and smaller TP for compute-bound prefill. Separating long prefills means ongoing decodes are not blocked by them. — [Dynamo design doc](https://docs.nvidia.com/dynamo/v1.2.0/design-docs/disaggregated-serving.md)
- SGLang docs: prefill is compute-heavy and decode memory-heavy. Prefill batches interrupt decode batches. With DP attention, one rank running prefill while another runs decode raises decode latency. — [SGLang PD docs](https://docs.sglang.io/advanced_features/pd_disaggregation.html)
- Hao AI Lab retrospective (3 Nov 2025):
  - Even with chunked prefill, a single large prefill can inflate TPOT by 2-30x under bursty load.
  - Colocation forces the allocator to "provision for the worst case of both metrics simultaneously".
  - DistServe's design: per-phase GPU count and parallelism search, KV transfer across differing parallelism, intra-node phase placement to use NVLink bandwidth, and pull-based scheduling so prefill bursts don't flood decoders.
  - More precisely, decode *attention* is memory-bound while FFN can be compute-bound.
  — [haoailab.com: Disaggregated Inference, 18 Months Later](https://haoailab.com/blogs/distserve-retro)
- KV transfer substrates:
  - **NIXL** (NVIDIA Inference Xfer Library) picks NVLink or InfiniBand/UCX, among others, and does VRAM-to-VRAM non-blocking transfer. — [Dynamo design doc](https://docs.nvidia.com/dynamo/v1.2.0/design-docs/disaggregated-serving.md). In vLLM, NIXL backends include UCX and GDS (GPUDirect Storage). — [vLLM docs excerpt](https://docs.vllm.ai/en/v0.23.0/features/disagg_prefill/)
  - **Mooncake Transfer Engine** supports TCP, RDMA, AWS EFA, NVMe-oF, NVLink (including MNNVL), HIP, CXL and Ascend. It does multi-NIC aggregation, topology-aware path selection and failover. On a 40 GB payload it reaches up to 87 GB/s on 4x200 Gbps RoCE and 190 GB/s on 8x400 Gbps RoCE, about 2.4x and 4.6x faster than TCP. — [Mooncake GitHub](https://github.com/kvcache-ai/Mooncake)
  - **AMD**: MoRI-IO (vLLM MoRIIOConnector, ROCm only); llm-d uses MoRI-IO plus NIXL on ROCm. **AWS**: NIXL over EFA. **Google TPU**: "TPU Sync". — [vLLM docs](https://docs.vllm.ai/en/stable/features/disagg_prefill/); [llm-d v1.0](https://llm-d.ai/blog/llm-d-v1.0-state-of-the-art-inference-ready-for-day-2)
- Transfer latency (practitioner source, not a primary benchmark): "typically 20-80 ms for KV cache of a 4K-token prompt on 70B FP8". The source itself calls this an internal composite. — [search summary of practitioner guides, e.g. learn.traeai](https://learn-traeai.pages.dev/t/ai-engineering/phases/17-infrastructure-and-production/17-disaggregated-prefill-decode)
- Scheduling and routing:
  - Dynamo's KV router uses a global radix tree of cached blocks and scores workers by overlap plus load. — [Dynamo v0.9 architecture](https://docs.nvidia.com/dynamo/v-0-9-0/design-docs/overall-architecture)
  - llm-d's EPP scorers cover prefix cache, load and predicted latency, with session affinity until saturation. Flow control holds requests at the router instead of queueing them on overloaded backends. — [llm-d v1.0](https://llm-d.ai/blog/llm-d-v1.0-state-of-the-art-inference-ready-for-day-2)
  - Mooncake Store gives a cross-node shared KV pool (DRAM + SSD tiers, striping, zero-copy). Dynamo merged an event-driven Mooncake shared-cache index for SGLang HiCache routing (1 Aug 2026). — [Mooncake GitHub](https://github.com/kvcache-ai/Mooncake)
- SGLang's `--disaggregation-decode-allocation-policy prefill_complete` delays reserving decode KV slots until prefill finishes. The default, `early`, reserves them up front. This is a memory-vs-latency knob on the decode side. — [SGLang docs](https://docs.sglang.io/advanced_features/pd_disaggregation.html)

### Inferences
- Rough KV size (my own arithmetic from the public Llama-3.1-70B config: 80 layers, 8 KV heads, head_dim 128, BF16): 2 x 80 x 8 x 128 x 2 B = 320 KiB per token. An 8K-token prompt is then about 2.6 GB, and about half that with an FP8 KV cache. At around 50-90 GB/s RDMA aggregate that is tens of milliseconds, which agrees with the 20-80 ms practitioner figure. Over 10-25 GbE TCP it would be seconds, so fast interconnect is effectively a prerequisite for big dense models. MLA models (DeepSeek, Kimi) have much smaller KV per token, which makes disaggregation cheaper for them.
- Layer-by-layer KV streaming (vLLM worker connectors store and load per layer alongside attention) and SGLang background prefill let transfer overlap compute, so effective TTFT overhead can be smaller than the raw transfer time.

### Gaps
- I found no primary, controlled measurement of NIXL transfer latency per GB on standard hardware.
- I did not retrieve the internals of Dynamo's KV router cost function (weights).

## Q3. Measured gains, and when disaggregation does NOT pay off

### Takeaway
Published gains are large but mostly vendor- or self-reported and "up to":
- DistServe: 7.4x goodput.
- Splitwise: 1.4x throughput at 20% lower cost, or 2.35x throughput at iso-cost.
- Mooncake: up to +525% simulated, +75% requests on real Kimi workloads.
- Dynamo: up to 30x on DeepSeek-R1 on GB200 NVL72, more than 2x on Llama-70B on Hopper.
- llm-d partners: AWS up to +70% tokens/s with P/D plus NIXL over EFA.

Gains concentrate in large MoE models, long prompts, strict TPOT SLOs and multi-node RDMA clusters. On a single node, a small GPU count or short prompts, colocated serving with chunked prefill is usually better.

### Cited Findings
- **DistServe** (OSDI'24): up to 7.4x more requests or 12.6x tighter SLO than state-of-the-art systems, with more than 90% of requests within latency constraints. — [arXiv 2401.09670](https://arxiv.org/abs/2401.09670)
- **Splitwise**: 1.4x higher throughput at 20% lower cost, or 2.35x throughput at the same cost and power. It argues decode can run on cheaper, lower-power hardware. — [arXiv 2311.18677](https://arxiv.org/abs/2311.18677)
- **Mooncake**: up to 525% throughput increase in simulated scenarios while meeting SLOs; 75% more requests for Kimi under real workloads. — [arXiv 2407.00079](https://arxiv.org/abs/2407.00079). Kimi K2 on 128 H200 with PD (Jul 2025): 224k tok/s prefill and 288k tok/s decode. — [Mooncake GitHub](https://github.com/kvcache-ai/Mooncake)
- **NVIDIA Dynamo** (all vendor claims):
  - Up to 30x more requests served for DeepSeek-R1 on GB200 NVL72; more than 2x throughput for Llama 70B on Hopper. — [NVIDIA dev blog](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models); [NVIDIA press release](https://nvidianews.nvidia.com/news/nvidia-dynamo-open-source-library-accelerates-and-scales-ai-reasoning-models)
  - A later NVIDIA blog projects about 6x for DeepSeek-R1 and up to 3x for Llama 70B in the medium-latency regime, using a GPU simulator. This shows the 30x is a peak, regime-specific figure. — [NVIDIA blog: GB200 NVL72 + Dynamo for MoE](https://developer.nvidia.com/blog/how-nvidia-gb200-nvl72-and-nvidia-dynamo-boost-inference-performance-for-moe-models/)
- **SGLang**:
  - DeepSeek-R1 on 96 H100s with PD plus large EP: 52.3k input tok/s and 22.3k output tok/s per node.
  - GB200 NVL72: up to 3.8x prefill and 4.8x decode throughput vs H100.
  - vLLM + llm-d 0.3 (Oct 2025): about 2.2k tok/s per H200 at 32-way EP.
  — [Hao AI Lab retrospective](https://haoailab.com/blogs/distserve-retro)
- **llm-d**:
  - 2025, with prefix/KV-aware routing (not P/D) on 2x8 H100: about 3x lower mean TTFT (Llama-4-Scout, ISL 20k); +50% QPS under a P95 TTFT of 2 s or less; 2x QPS for Llama-3.1-70B. P/D showed only "promising speedups" for 20:1 ISL:OSL workloads, with no numbers. — [llm-d announce](https://llm-d.ai/blog/llm-d-announce)
  - v1.0 partner claims:
    - AWS SageMaker HyperPod: up to 70% more tokens/s vs standard vLLM with P/D plus NIXL over EFA.
    - Google Vertex AI: TTFT down more than 35%, prefix hit rate up from 35% to 70%.
    - Tesla/Red Hat MI300X with Llama-3.1-70B: 3x output tok/s and 2x lower TTFT from prefix-aware routing.
    - IBM: 3,000 concurrent coding agents on 544 H100, about 98% prefix hit rate, about $0.11 per million tokens.
    - Google TPU 7x: median TTFT down more than 85%, 2.4x throughput from prefix routing plus host-memory KV offload.
    - These gains are not broken out per technique.
  — [llm-d v1.0](https://llm-d.ai/blog/llm-d-v1.0-state-of-the-art-inference-ready-for-day-2)
- **When it does not pay off:**
  - vLLM: P/D "DOES NOT improve throughput". — [vLLM docs](https://docs.vllm.ai/en/stable/features/disagg_prefill/)
  - Practitioner guidance (rules of thumb, unofficial): skip it for a single GPU or small node ("nowhere to send the KV cache; colocation with chunked prefill is correct"), for fewer than about 4 GPUs, for prompts and outputs under about 512 tokens (the transfer tax exceeds the savings), and for short uniform prompts where interference barely exists. — [search summary: dev.to](https://dev.to/ji_ai/prefilldecode-disaggregation-stop-serving-llms-on-one-gpu-461h), [BentoML](https://origin.bentoml.com/llm/inference-optimization/prefill-decode-disaggregation), [learn.traeai](https://learn-traeai.pages.dev/t/ai-engineering/phases/17-infrastructure-and-production/17-disaggregated-prefill-decode)
  - llm-d v1.0 itself stresses the operational complexity: prefill and decode must roll forward in lockstep, which is why it added DisaggregatedSet. — [llm-d v1.0](https://llm-d.ai/blog/llm-d-v1.0-state-of-the-art-inference-ready-for-day-2)
  - Adoption was slow in 2024 because it required substantial engine refactoring. — [Hao AI Lab](https://haoailab.com/blogs/distserve-retro)
- **Trends 2025-2026**: heterogeneous prefill and decode hardware, decode-specialized ASICs, NVIDIA Rubin CPX (a prefill/long-context accelerator built around disaggregation), and Attention-FFN Disaggregation (AFD), mostly for MoE (harder for dense models). — [Hao AI Lab](https://haoailab.com/blogs/distserve-retro)

### Inferences
- For a single workstation with one or two consumer GPUs and no RDMA (the user's own setup), P/D disaggregation has no practical benefit. The relevant techniques there are chunked prefill and prefix caching. Prefix-aware routing and KV offload (CPU/SSD tiers via LMCache or Mooncake Store) deliver a large share of the measured gains even without P/D, as the llm-d partner numbers show.
- Many headline numbers bundle P/D with wide expert parallelism, prefix routing and new hardware (GB200), so P/D's isolated contribution is usually unknown.

### Gaps
- I found no rigorous, independent small-cluster crossover study. The thresholds (512 tokens, 4 GPUs) are unofficial heuristics.
- The original papers' hardware: DistServe used A100s in its paper (from memory, not verified here); Splitwise used A100/H100 profiling (likewise unverified here).

## Q4. Code: where it lives

### Takeaway
The main code entry points are vLLM `vllm/distributed/kv_transfer`, the SGLang disaggregation module plus the Model Gateway, `ai-dynamo/dynamo` (Rust runtime and router) together with NIXL, `kvcache-ai/Mooncake`, and the `llm-d` GitHub org (llm-d-router/EPP).

### Cited Findings
- vLLM: implementation under `vllm/distributed/kv_transfer`. The scheduler-side connector schedules transfers; worker-side connectors load and store KV layer by layer. Internal abstractions are Connector, LookupBuffer (`insert`/`drop_select`) and Pipe (`send_tensor`/`recv_tensor`). Examples:
  - `examples/disaggregated/lmcache/disagg_prefill_lmcache_v1/disagg_example_nixl.sh`
  - `examples/disaggregated/mooncake_connector/run_mooncake_connector.sh`
  - `tests/v1/kv_connector/nixl_integration/run_accuracy_test.sh`
  - `examples/disaggregated/flexkv_connector/prefix_caching_flexkv.py`
  — [vLLM docs](https://docs.vllm.ai/en/stable/features/disagg_prefill/)
- Mooncake repo top level: `mooncake-transfer-engine`, `mooncake-store` (plus a Rust `mooncake-store-rs`), `mooncake-ep`, `mooncake-pg`, `mooncake-p2p-store`, `mooncake-integration`, `mooncake-conductor`, and `FAST25-release` (traces). Integrations:
  - vLLM: MooncakeConnector, MooncakeStoreConnector.
  - SGLang: PD transfer, HiCache backend.
  - LMCache.
  - Dynamo.
  - TensorRT-LLM: a Store connector merged into KVCMv2 on 10 Oct 2026.
  — [Mooncake GitHub](https://github.com/kvcache-ai/Mooncake)
- Dynamo: [github.com/ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo); Rust crate `dynamo-llm` (v1.2.1 on docs.rs) — [docs.rs](https://docs.rs/crate/dynamo-llm/1.2.1)
- SGLang: router is `sglang_router` (Model Gateway); env vars `SGLANG_DISAGGREGATION_*` and `SGLANG_MOONCAKE_*`. — [SGLang docs](https://docs.sglang.io/advanced_features/pd_disaggregation.html)
- llm-d: router coordination consolidated into "llm-d-router". The docs site is llm-d.ai; the GitHub org is `llm-d`. — [llm-d v1.0](https://llm-d.ai/blog/llm-d-v1.0-state-of-the-art-inference-ready-for-day-2)

### Inferences
- (Now verified, see the next line.) SGLang source path: `python/sglang/srt/disaggregation/`.
- SGLang `python/sglang/srt/disaggregation/` contains backend subdirectories `ascend/`, `base/`, `common/`, `encoder/`, `fake/`, `mooncake/`, `mori/` and `nixl/`. It also has files such as `decode_hicache_mixin.py`, `decode_kvcache_offload_manager.py`, `kv_events.py` and `role_switch.py`; the listing was partial. — [GitHub sgl-project/sglang disaggregation dir](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/disaggregation)

### Gaps
- I did not browse the source trees directly (llm-d-router repo layout, NIXL repo `ai-dynamo/nixl`, SGLang disaggregation directory). Paths not cited above are unverified.
