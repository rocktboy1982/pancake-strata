# Service-engine decoupled LLM runtimes for heterogeneous accelerators (state as of Nov 2026)

Scope: checking claims from a pasted AI-written overview, plus a map of how the main runtimes support non-NVIDIA accelerators. Research was done on the web only (about 15 search and fetch calls, 2026-11-08). The verdict labels used throughout are VERIFIED, PARTIALLY ACCURATE and NOT FOUND.

Summary of verdicts on the claimed names:

| Claim in overview | Verdict |
|---|---|
| "xLLM-AI / xllm" as the project name or org | PARTIALLY ACCURATE. The canonical repo is `jd-opensource/xllm`, with companion `jd-opensource/xllm-service`. Chinese press calls the product "Oxygen xLLM". I did not find a GitHub org named "xLLM-AI". |
| Built by JD.com | VERIFIED. JD.com is the lead affiliation on the arXiv report, alongside Tsinghua, USTC, BUAA and PKU. |
| Hosted by the OpenAtom Foundation | VERIFIED. Donated in June/July 2026. The exact date differs between sources (see below). |
| Service-layer / engine-layer decoupling | VERIFIED. xLLM-Service and xLLM-Engine are separate layers. |
| Ascend, Moore Threads, Hygon DCU and Cambricon MLU supported | VERIFIED per the current README, which also lists MetaX and Iluvatar. The arXiv report itself evaluates only Ascend 910B/910C. |
| Uses Mooncake for KV cache, with offload/prefetch across VRAM/RAM | VERIFIED. The README says "offloading and prefetching". The paper describes HBM-DRAM-SSD tiers built on Mooncake Store and the Mooncake Transfer Engine. |
| Benchmarks | VERIFIED as the authors' own claims. Up to 1.7x MindIE and 2.2x vLLM-Ascend on Qwen. Nobody has reproduced them independently. |

---

## 1. xLLM: identity, governance, architecture, accelerators, benchmarks

### Takeaway
xLLM is real. It is JD.com's Apache-2.0 inference engine, "specifically optimized for Chinese AI accelerators". It has a decoupled design with xLLM-Service for scheduling, PD/EPD disaggregation and global KV management, and xLLM-Engine for graph mode, multi-stream pipelining and xTensor memory. It was donated to the OpenAtom Foundation in mid-2026. The headline benchmarks were measured only on Ascend 910B/910C against MindIE and vLLM-Ascend.

### Cited Findings
**Identity and governance**
- arXiv:2510.14686, "xLLM Technical Report", v1 dated 2025-10-16 and v2 dated 2026-03-03, cs.DC. Tongxuan Liu plus 51 co-authors, with Ke Zhang among the corresponding authors — [arXiv abs](https://arxiv.org/abs/2510.14686v1); [EmergentMind summary](https://www.emergentmind.com/papers/2510.14686)
- Author affiliations are JD.com, Tsinghua University, USTC, BUAA and Peking University. The corresponding authors use jd.com addresses — [arXiv HTML](https://arxiv.org/html/2510.14686v1)
- The code is at github.com/jd-opensource/xllm and the service component at github.com/jd-opensource/xllm-service, both named in the paper — [arXiv abs](https://arxiv.org/abs/2510.14686v1)
- The README describes xLLM as "an efficient LLM inference framework, specifically optimized for Chinese AI accelerators". The About section calls it a high-performance engine for LLM, VLM, DiT and REC (recommendation) models, and says the project is hosted by the OpenAtom Foundation — [GitHub jd-opensource/xllm](https://github.com/jd-opensource/xllm)
- The README news entry dates the official donation to the OpenAtom Foundation as 2026-07-06 — [GitHub README](https://github.com/jd-opensource/xllm)
- Press accounts date it differently. One says JD announced the donation of "Oxygen xLLM" at an OpenAtom ceremony on 2026-06-24. Another (IT之家, via aggregators) says xLLM was one of 7 projects signed at the 2026 OpenAtom Open Source Ecosystem Conference in Beijing Yizhuang on 2026-06-25. JD transferred the copyright, related patents and trademark to the foundation, and the project entered the foundation's incubation track. The date conflict (June 24/25 against July 6 in the README) probably separates the signing ceremony from the formal completion — [Sina Finance](https://finance.sina.cn/tech/2026-06-26/detail-inietfxt7358071.d.html); [besthub.dev](https://www.besthub.dev/articles/jd-donates-oxygen-xllm-open-source-large-model-inference-engine-boosts-china-s-ai-infrastructure-df69e2faea28); [CSDN](https://adg.csdn.net/6a697517662f9a54cb9593a4.html)
- JD's own claims are production use in JingYan (京言), intelligent customer service and risk control, with ">5x efficiency" and "90% lower machine cost". These are vendor figures with no independent check — [CSDN / press summary](https://damodev.csdn.net/6a4373956b2ae520ee59b4af.html)
- One aggregator says the GitHub repo was "published 2026-03-23". The arXiv paper of Oct 2025 already links the repo, so that date is unreliable — [openflows](https://openflows.pages.dev/currency/currents/xllm)

**Architecture: the service-engine split**
- xLLM-Service covers scheduling. It runs three elastic instance pools (Prefill, Decode and Encode), with policies for online/offline co-location on shared hardware, dynamic PD disaggregation, hybrid Encode-Prefill-Decode (EPD) disaggregation for multimodal requests, and global KV cache management with fault tolerance — [arXiv HTML](https://arxiv.org/html/2510.14686v1)
- xLLM-Engine covers execution. That includes multi-layer pipeline execution, graph mode, memory management through xTensor, speculative decoding and expert-parallel load balancing (EPLB) — [arXiv HTML](https://arxiv.org/html/2510.14686v1)
- The pipeline works at three levels. It overlaps CPU scheduling with accelerator execution, runs dual-stream micro-batches (a compute stream alongside a communication stream for MoE dispatch/combine), and overlaps the matrix (cube) and vector units at operator level. In EPD mode, the vision encoder and the LLM run on separate streams — [arXiv HTML](https://arxiv.org/html/2510.14686v1)
- Adaptive Graph Mode is built on Ascend ACLGraph capture/replay, the Ascend counterpart to CUDA Graph. It uses dimension parameterization so one graph can serve varying shapes, a shared HBM memory pool across graphs, and partial graphs. Modules with simple dynamic shapes are captured and complex ones stay eager. Custom ops such as PagedAttention and AllReduce were reworked to take runtime shape parameters — [arXiv HTML](https://arxiv.org/html/2510.14686v1)
- xTensor stores the KV cache as "logically contiguous, physically discrete" memory. Physical pages are mapped on demand (virtual-memory style), reused after a request finishes, and the next token's pages are pre-mapped asynchronously. Attention operators were adapted to drop the block-table input, and the paper describes a contiguous-address FlashMLA variant. It is an alternative to PagedAttention block tables, similar to vAttention — [arXiv HTML](https://arxiv.org/html/2510.14686v1)

**Supported accelerators**
- The current README lists Ascend NPU (A2 and A3, needs HDK driver 25.2.0+), Cambricon MLU, Moore Threads MUSA (S5000), Hygon DCU (BW1000), MetaX MACA (MXC500) and Iluvatar CoreX (BI150). The capture of the README did not mention NVIDIA CUDA — [GitHub README](https://github.com/jd-opensource/xllm)
- The arXiv paper evaluates only Ascend 910B and 910C. CUDA Graph appears there only as an analogy, and it refers to "domestic accelerators" without naming Cambricon, Hygon or Moore Threads — [arXiv HTML](https://arxiv.org/html/2510.14686v1)
- The repo has an `xllm-kernel` directory for per-backend kernels. Model support announced in the news section includes GLM-4.5/4.6/4.6V/4.7/5/5.3-Flash, DeepSeek-V4 (2026-04-24), MiniMax-M3 (2026-06-13) and VLM-R1, and the topics list includes Qwen — [GitHub README](https://github.com/jd-opensource/xllm)

**Benchmarks (authors' own; baselines vLLM-Ascend v0.10.rc1 and MindIE v2.1.rc1)**
- Qwen3 series, 0.6B to 32B: on 910B, up to 1.9x vLLM-Ascend and 1.7x MindIE. On 910C, up to 2.2x vLLM-Ascend and 1.5x MindIE, under identical TPOT constraints — [arXiv HTML](https://arxiv.org/html/2510.14686v1)
- DeepSeek-R1 on 910B: about 1.7x MindIE and about 12x vLLM-Ascend. On 910C: about 1.4x MindIE — [arXiv HTML](https://arxiv.org/html/2510.14686v1)
- DeepSeek-R1 with PD disaggregation at TPOT 100 ms: 11,351.58 tok/s against MindIE's 8,476.44, about 34% higher — [arXiv HTML](https://arxiv.org/html/2510.14686v1)
- JD business workloads, all against MindIE and vLLM-Ascend:
  - DeepSeek-V3 in the JingYan scenario at TPOT 80 ms: 196.45 tok/s, against 144.40 for MindIE and 21.17 for vLLM-Ascend.
  - Qwen3-32B customer service on 8 accelerators: 3.1x vLLM-Ascend and 1.2x MindIE.
  - Qwen2-7B product understanding: 25% above MindIE and 56% above vLLM-Ascend.

  — [arXiv HTML](https://arxiv.org/html/2510.14686v1)
- Some secondary summaries mix the 1.9x (910B) and 2.2x (910C) figures. The abstract's "up to 2.2x vLLM-Ascend" is the 910C number — [EmergentMind](https://www.emergentmind.com/papers/2510.14686); [arXiv HTML](https://arxiv.org/html/2510.14686v1)

### Inferences
- The overview's "service vs. engine" framing is accurate. In this design the service layer is not just an OpenAI-compatible HTTP front end. It is a cluster scheduler that owns instance pools, PD/EPD roles and the global KV index (ETCD metadata).
- The non-Ascend backends (MLU, DCU, MUSA, MACA, ILU) look like later additions from 2026. The paper's performance claims should not be carried over to them.
- The baselines (vLLM-Ascend 0.10rc1, MindIE 2.1rc1) are from mid-2025. vLLM-Ascend has reached v0.30.0 by Oct 2026, so the gaps are probably smaller today.

### Gaps
- I could not confirm whether a GitHub org named "xLLM-AI" exists, for example as a post-donation mirror. The only verified repos are under jd-opensource. I did not check an OpenAtom-hosted mirror such as AtomGit or gitlink. A gitlink mirror page exists ([gitlink](https://www.gitlink.org.cn/mirrors/xllm)), but I did not inspect it.
- Whether current xLLM still supports NVIDIA CUDA is unclear from the README capture.
- No independent benchmarks of xLLM, and none on non-Ascend hardware.

---

## 2. Mooncake and its role in xLLM

### Takeaway
The overview's claim is essentially correct. xLLM builds hybrid KV cache management on Mooncake: the Mooncake Store design extended to domestic accelerators, plus the Mooncake Transfer Engine for transport, with offloading and prefetching per the README. Mooncake (kvcache-ai, the Moonshot AI / Kimi serving platform) is a KV-centric disaggregated architecture. It has a multi-transport Transfer Engine and a distributed DRAM/SSD KV store, and has become an ecosystem-wide component used by vLLM, SGLang, TensorRT-LLM, LMCache, LMDeploy and xLLM.

### Cited Findings
- The xLLM README news item of 2025-12-05 says xLLM builds hybrid KV cache management on Mooncake, providing global KV cache management with offloading and prefetching. The Acknowledgments credit Mooncake — [GitHub xllm](https://github.com/jd-opensource/xllm)
- The paper says each instance keeps a multi-level HBM-DRAM-SSD pool for KV and image tokens, with the invariant that HBM-resident data is also in DRAM. Globally it "adopts ideas from Mooncake Store", extends them to domestic accelerators, and uses the Mooncake Transfer Engine for transmission. ETCD holds metadata and heartbeat sync of cache events, and the scheduler routes requests by prefix match to maximize KV reuse — [arXiv HTML](https://arxiv.org/html/2510.14686v1)
- The paper does not describe a separate KV prefetch mechanism. The prefetch claim comes from the README rather than the paper — [arXiv HTML](https://arxiv.org/html/2510.14686v1); [GitHub xllm](https://github.com/jd-opensource/xllm)
- Mooncake is the serving platform for Moonshot AI's Kimi. It lets Kimi handle "75% more requests while adhering to SLOs" — [GitHub kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake)
- The Transfer Engine (TE) aggregates multiple RDMA NICs, picks paths by NUMA and device topology, and fails over when a path breaks.
  - Transports: TCP, RDMA, AWS EFA, NVMe-oF, NVLink, HIP, Barex, CXL and Ascend-family transports.
  - Accelerator memory detection: CUDA, MUSA, HIP, MACA, Cambricon MLU and Ascend, with Intel XPU via "TENT".
  - Bandwidth: up to 87 GB/s on 4x200 Gbps RoCE and 190 GB/s on 8x400 Gbps RoCE, about 2.4x and 4.6x faster than TCP.

  — [GitHub Mooncake](https://github.com/kvcache-ai/Mooncake)
- Mooncake Store is a distributed KV cache built on TE, also used for model weights. It has DRAM and SSD/NVMe tiers, replication, eviction, zero-copy transfer, and per-object placement and pinning policies — [GitHub Mooncake](https://github.com/kvcache-ai/Mooncake)
- Integrations:
  - vLLM: MooncakeConnector for PD disaggregation and MooncakeStoreConnector for a shared KV pool.
  - SGLang: PD transfer, HiCache storage backend, Elastic EP, and RDMA weight transfer for RL.
  - TensorRT-LLM: TE for PD transfer, plus a Store connector (KVCMv2) merged 2026-10-10.
  - LMCache: remote KV storage and P2P sharing.
  - LMDeploy: v0.18.0, merged 2026-08-28.
  - xLLM: TE and Store.

  — [GitHub Mooncake](https://github.com/kvcache-ai/Mooncake)
- Hardware partners listed by Mooncake: NVIDIA, Huawei, AMD, Cambricon, Moore Threads, AWS, MetaX, T-Head, Alibaba Cloud, Hygon and Biren. Mooncake won Best Paper at FAST 2025 (2025-02-25) and joined the PyTorch Ecosystem on 2026-02-12 — [GitHub Mooncake](https://github.com/kvcache-ai/Mooncake)

### Inferences
- "Prefetching/offloading across VRAM/RAM" is accurate, and the tiering actually goes further, to SSD as well. The "accelerator-specific optimizations" part is best read as Mooncake TE transports per vendor (HIP, MUSA, MACA, MLU, Ascend) plus xLLM's own Ascend work, rather than special Mooncake features for xLLM.
- Since Mooncake TE has a HIP transport and is integrated with vLLM and SGLang, AMD setups can in principle use the same PD-disaggregation and KV-pool path. That matters for multi-node use, not for a single consumer GPU.

### Gaps
- I did not fetch the Mooncake FAST'25 paper itself. Details such as the KVCache-centric Conductor scheduler and early-rejection policy come from prior knowledge, not from a source fetched here, so they are left out.
- No measured data on Mooncake over HIP or RDNA3 consumer GPUs.

---

## 3. Other heterogeneous-accelerator runtimes

### Takeaway
The dominant pattern in 2026 is a common engine core with out-of-tree hardware plugins, as in vLLM's `vllm.platform_plugins` (vllm-ascend, vllm-kunlun, IBM Spyre, vLLM-Metal). Vendor-native stacks remain alongside it: Huawei MindIE, which has its own LLM engine plus the MindIE Motor/Service serving layer. Several runtimes take a portable-kernel-library approach instead: llama.cpp/ggml with roughly 17 backends, MLC-LLM compiling through TVM, and ONNX Runtime GenAI with its execution providers.

### Cited Findings
**vLLM plugins**
- The Hardware Pluggable RFC (vllm#11162) builds on the "Make vLLM device-agnostic" RFC (#6080). Plugins register a `Platform` subclass (`vllm.platforms.interface.Platform`, enum `PlatformEnum.OOT`) through the `vllm.platform_plugins` entry point. Executor, Worker, ModelRunner, AttentionBackend and Communicator were refactored so that plugins can override them — [vLLM blog: Hardware Plugin](https://vllm.ai/blog/2025-05-12-hardware-plugin)
- A platform plugin is required even if the hardware already supports the PyTorch backend natively — [vLLM forum](https://discuss.vllm.ai/t/what-is-the-best-way-to-support-vllm-on-a-custom-hardware/1743)
- Adopters of the mechanism: Ascend (reference implementation), IBM Spyre and Kunlun XPU (vllm-kunlun) — [vLLM blog](https://vllm.ai/blog/2025-05-12-hardware-plugin); [vllm-kunlun docs](https://vllm-kunlun.readthedocs.io/en/stable/)
- vllm-ascend is a "community maintained hardware plugin". It supports Atlas 800I A2/A3, Atlas 300I Duo and 950PR/950DT products.
  - Latest stable: v0.30.0 (2026-10), after v0.23.0 (2026-08), v0.18.0 (2026-05), v0.13.0 (2026-02), v0.11.0 (2025-12) and the first official release v0.7.3 (2025-05).
  - Prerequisites: Python 3.12, CANN 9.2.1, PyTorch 2.10.0, torch-npu 2.10.0.post7.
  - Release branches track vLLM versions one to one.

  — [GitHub vllm-ascend](https://github.com/vllm-project/vllm-ascend)
- vLLM ROCm support is in-tree. Supported GPUs: MI200 (gfx90a), MI300 (gfx942), MI350 (gfx950), Radeon RX 7900 (gfx1100/1101), RX 9000 (gfx1200/1201), and Ryzen AI MAX / AI 300 (gfx1151/1150). Requires ROCm 6.3 or later, 7.0 or later for MI350.
  - Prebuilt Python 3.12 wheels: `rocm100` (ROCm 10.0, nightly default), `rocm721`, and `rocm700` (stable 0.14 to 0.18).
  - Docker image: `vllm/vllm-openai-rocm` (latest = ROCm 10.0).
  - Build components: Triton and AITER, with CK flash-attention optional.

  — [vLLM GPU install docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)
- The current vLLM install page lists Intel XPU (needs `vllm-xpu-kernels`) and Apple Silicon via vLLM-Metal, a community hardware plugin built on MLX. Gaudi and TPU were not on that page — [vLLM GPU install docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)

**SGLang**
- SGLang has official Ascend NPU docs: Dockerfile.npu, torch-npu, launching with `--attention-backend ascend`, a MindSpore model path, and examples for DeepSeek, Qwen3/3.5 and GLM-5 — [SGLang Ascend docs](https://docs.sglang.ai/platforms/ascend_npu.html); [support matrix](https://docs.sglang.io/platforms/ascend_npu_support.html)
- SGLang has an AMD GPU platform page. sgl-kernel is built with a ROCm-specific setup script and there is a separate AMD quantization matrix. The copy I found was a Hugging Face mirror pinning v0.5.9 — [HF mirror of sglang docs](https://huggingface.co/FasterDFlash/Hanrui/blob/main/sglang/docs/platforms/amd_gpu.md)

**Huawei MindIE**
- MindIE is Huawei Ascend's inference acceleration suite, made up of MindIE LLM, MindIE SD and an acceleration plugin library — [Ascend MindIE overview](https://www.hiascend.com/document/detail/en/mindie/230/quickstart/mindie_what_0001.html)
- MindIE LLM has a Server layer, an LLM Manager and an Engine (Scheduler, Executor, Worker). It supports continuous batching, PagedAttention and FlashDecoding, and compiles an optimized graph for Ascend — [Ascend MindIE LLM docs](https://www.hiascend.com/document/detail/en/mindie/latest/LLMframe/llmdev)
- MindIE Motor is a request-scheduling framework for PD-disaggregated inference that sits on top of MindIE LLM. It is a service-engine split similar to xLLM's. MindIE and CANN versions must match — [Ascend docs](https://www.hiascend.com/document/detail/en/mindie/230/quickstart/mindie_what_0001.html); [MindSpore deployment guide](https://www.mindspore.cn/mindformers/docs/en/master/guide/deployment.html)

**llama.cpp / ggml backends**
- Backends: BLAS, BLIS, CANN (Huawei Ascend), CUDA, HIP (AMD), Hexagon (Qualcomm), IBM zDNN, MUSA (Moore Threads), Metal, OpenCL (Adreno), OpenVINO (in progress), RPC, SYCL (Intel GPU), VirtGPU, Vulkan, WebGPU and ZenDNN (AMD CPU). There is no Hygon-specific backend; DCU would presumably go through HIP — [GitHub llama.cpp](https://github.com/ggml-org/llama.cpp)

**MLC-LLM**
- MLC-LLM is a machine-learning compiler built on TVM, MetaSchedule and TensorIR. MLCEngine is one engine with an OpenAI-compatible API.
  - AMD: Vulkan and ROCm on Linux/Windows, Metal on macOS.
  - NVIDIA: Vulkan and CUDA.
  - Intel: Vulkan.
  - Other targets: WebGPU/WASM, iOS Metal, and Android OpenCL (Adreno/Mali).

  — [GitHub mlc-llm](https://github.com/mlc-ai/mlc-llm)

**ONNX Runtime GenAI**
- Execution providers: CPU, CUDA, DirectML (Windows, cross-vendor), NvTensorRtRtx, OpenVINO (Intel CPU/iGPU/NPU), QNN (Qualcomm Hexagon) and WebGPU. The ROCm EP is marked deprecated in ONNX Runtime — [ORT EP list](https://onnxruntime.ai/docs/execution-providers/); [ORT GenAI acceleration overview (mirror)](https://www.mintlify.com/microsoft/onnxruntime-genai/acceleration/overview)

### Inferences
- For Chinese accelerators the practical options are xLLM (most backends: Ascend, MLU, DCU, MUSA, MACA, ILU), vllm-ascend or SGLang on Ascend, MindIE (Ascend only), and llama.cpp (CANN, MUSA) for small-scale use.
- On Windows, AMD GPUs are reachable through ORT GenAI only via DirectML, now that the ROCm EP is deprecated.

- vLLM's TPU support now runs through the out-of-tree `tpu-inference` hardware plugin, which unifies a JAX path and a PyTorch (Torchax) path under one lowering.
  - Recommended TPUs: v7x, v5e and v6e. Experimental: v3, v4 and v5p.
  - Feature table as of 2026-07-02: chunked prefill, prefix caching, KV cache offload and Eagle3/DFlash speculative decoding pass. Multi-host and hybrid KV cache are untested.
  - Dev builds reached 0.26.0.dev20260731.

  — [GitHub tpu-inference](https://github.com/vllm-project/tpu-inference); [PyPI tpu-inference](https://pypi.org/project/tpu-inference/); [PyPI 0.26.0.dev20260731](https://pypi.org/project/tpu-inference/0.26.0.dev20260731/)

### Gaps
- I did not verify the 2026 status of vLLM for Intel Gaudi (the vllm-gaudi plugin). It was not on the current install page and did not come up in a targeted search.
- No authoritative 2026 SGLang ROCm docs page was fetched; only a mirror.
- No current MLC-LLM activity dates.

---

## 4. Hardware abstraction mechanisms, and AMD ROCm on consumer RDNA3 (gfx1100)

### Takeaway
Abstraction happens at four layers:
- **Platform/plugin classes.** vLLM `Platform` OOT plugins, and xLLM per-vendor backends plus `xllm-kernel`.
- **Per-backend kernel libraries.** ggml backends, AITER/Triton/CK on ROCm, Ascend custom ops.
- **Graph capture.** CUDA Graph, Ascend ACLGraph with xLLM's parameterized partial graphs, and torch.compile-based graph modes.
- **Memory managers.** PagedAttention block tables, xTensor virtual mapping, and Mooncake tiered HBM/DRAM/SSD pools.

gfx1100 (RX 7900 XT/XTX) is an officially supported vLLM ROCm target and appears in AMD's ROCm 7.x matrices.

### Cited Findings
- vLLM abstracts hardware through the `Platform` interface plus pluggable Executor, Worker, ModelRunner, AttentionBackend and Communicator — [vLLM blog](https://vllm.ai/blog/2025-05-12-hardware-plugin)
- xLLM graph capture uses ACLGraph with dimension parameterization, a shared HBM pool and partial graphs. Ops like PagedAttention and AllReduce were rewritten to take runtime shapes. This answers the dynamic-shape limitation that CUDA and ACL graphs share — [arXiv HTML](https://arxiv.org/html/2510.14686v1)
- xLLM's memory manager, xTensor, uses on-demand physical page mapping behind a contiguous virtual KV range, with async pre-mapping — [arXiv HTML](https://arxiv.org/html/2510.14686v1)
- MLC-LLM generates kernels per target by compiling with TVM/TensorIR, rather than using hand-written per-backend libraries — [GitHub mlc-llm](https://github.com/mlc-ai/mlc-llm)
- vLLM lists gfx1100/1101 as supported with ROCm 6.3 or later, and states no Radeon-specific limitations on the install page — [vLLM GPU install docs](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html)
- AMD's ROCm compatibility matrices list gfx1100 as an RDNA3 target in 7.0.x, 7.1.0 and 7.2.2, with footnotes I could not read — [ROCm 7.2.2 matrix](https://rocm.docs.amd.com/en/docs-7.2.2/compatibility/compatibility-matrix.html); [ROCm 7.1.0](https://rocm.docs.amd.com/en/docs-7.1.0/compatibility/compatibility-matrix.html)
- The ROCm 10.0.0 matrix, dated 2026-08-14, lists hipBLASLt 1.4.1. The excerpt I saw showed RDNA3 APU targets (gfx1103, gfx1150-1153) but did not explicitly show gfx1100 — [ROCm 10.0.0 matrix](https://rocm.docs.amd.com/en/docs-10.0.0/compatibility/compatibility-matrix.html)
- Community reports (secondary, partly conflicting):
  - The RX 7900 XTX is called a first-class gfx1100 target through ROCm 7.x, Linux only.
  - One user runs vLLM on Ubuntu 24.04 with ROCm 7.1.1 and a 7900 XTX, but notes that some models are not officially listed for gfx1100.

  — [runaihome guide](https://runaihome.com/blog/rocm-7-ubuntu-local-llm-setup-2026/); [vLLM forum](https://discuss.vllm.ai/t/vllm-combine-7900-xtx-and-w7900/1681)
- On Strix Halo (gfx1151, not gfx1100), Flash Attention needed a custom AOTriton build and rocWMMA, and hipBLASLt benchmarked slower than the alternative. This is unofficial and for different hardware — [whylucian softab](https://whylucian.github.io/softab/docs/rocm-support.html)

### Inferences
- For this user's RX 7900 XT (gfx1100), the supported high-throughput path is vLLM with the `rocm100` wheels or the `vllm/vllm-openai-rocm` image on ROCm 10.0, which is the installed ROCm per local notes. The portable fallback is llama.cpp HIP or Vulkan.
- xLLM, MindIE and vllm-ascend do not apply to RDNA3. xLLM's backend list does not include AMD/HIP, although Hygon DCU is HIP-derived.
- AMD MI-series kernel libraries (AITER, CK flash-attention) target CDNA. RDNA3 likely falls back to Triton attention paths. This is an inference, not confirmed for 2026.

### Gaps
- I found no source describing HIP graph support in vLLM on gfx1100, or which attention backend vLLM selects by default on RDNA3 in 2026.
- gfx1100's explicit status in the ROCm 10.x matrix and its footnotes could not be read from the excerpts.
- Note that the local memory says the RX 7900 XT was "not detected" as of 2026-11-07 (RTX 3060 now installed). Applicability depends on the hardware actually present.
