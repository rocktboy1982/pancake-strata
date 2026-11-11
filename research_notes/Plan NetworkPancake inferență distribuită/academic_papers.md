# Academic literature: distributed LLM inference on heterogeneous, low-end, networked consumer devices (as of 2026-11-10)

Legend: [FULL] = I opened the paper text (partially or fully); [ABS] = only seen in abstract / search snippet / third-party summary, not verified in the paper. Budget-limited: ~20 tool calls, so many papers are [ABS].

## 1. Edge / mobile / home-cluster distributed inference (pipeline, tensor, hybrid)

### Takeaway
prima.cpp is the closest published system to a LAN mesh of old PCs and phones (it ran on laptops, desktops, a phone and a tablet over Wi-Fi). Its main lesson is a pipelined ring with mmap/prefetch plus a heterogeneity-aware scheduler. It does not cover MoE, and its GPU support is CUDA-only. Galaxy, Jupiter and EdgeShard show hybrid or pipeline partitioning gains of 1.5-2.5x on small clusters (abstract numbers). Tensor parallelism over weak links is generally poor, and TPI-LLM argues the opposite only for a tuned low-resource setup.

### Cited Findings
- **prima.cpp: "Fast 30-70B LLM Inference on Heterogeneous and Low-Resource Home Clusters"** (arXiv v1 title "PRIMA.CPP: Speeding Up 70B-Scale LLM Inference on Low-Resource Everyday Home Clusters"; ICLR 2026) [FULL, first ~100k of 140k chars] — [arXiv 2504.08791](https://arxiv.org/abs/2504.08791), [html v2](https://arxiv.org/html/2504.08791v2), [ICLR page](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e991e5587c1daa49bbf9a818b3f02f9a-Abstract-Conference.html)
  - Setup: D1 Mac M1 (2.4 GiB avail RAM), D2 laptop i9 + RTX 3070 8 GiB, D3 desktop i9 + RTX 2080 Ti 11 GiB, D4 Mate 40 Pro phone (Termux, 1.9 GiB RAM); also an Honor Pad (Android) and an old Intel Mac Air. Default cluster D1-D4, about 37 GiB aggregate RAM+VRAM. Linked by a local Wi-Fi router with 320-610 Mbps and 3-7 ms latency. Llama models 8B-70B, mostly Q4K.
  - TPOT (ms/token) llama.cpp vs prima.cpp: 8B 15 vs 15; 30B 202 vs 72; 60B 7,965 vs 468; 70B 10,120 vs 674. exo and dllama: 8B 263 and 459 ms; both OOM at 70B. Claims 5-17x lower TPOT than llama.cpp, exo and dllama; memory pressure under 6% per device. With speculative decoding: 26 tok/s at 32B, 442 ms/token at 70B.
  - Halda scheduler: enumerates valid factors of the layer count, solves a small ILP per candidate (HiGHS), reassigns devices between memory-status sets until convergence, and reports 10-12 ms scheduling latency for 4-32 devices. It drops weak devices or keeps them only as ring relays.
  - Applicability: the direct template for a LAN mesh of old PCs and phones. It confirms layer-pipelined ring plus mmap/disk prefetch lets a model larger than any single node's RAM run, and that weak nodes should sometimes be dropped. Note: with a single device llama.cpp is better.
  - Limits: 4-device Wi-Fi testbed only; CUDA-only GPUs (Vulkan on roadmap, per project page via search snippet); no MoE mention in the part I read; 674 ms/token at 70B without speculative decoding. The first-version claim "15x" was revised to 5-17x in the ICLR version.
- **Galaxy: collaborative edge AI system for in-situ Transformer inference** (IEEE INFOCOM 2024) [ABS] — [arXiv 2405.17245](https://arxiv.org/abs/2405.17245). Hybrid tensor plus sequence parallelism with a heterogeneity and memory-aware planner and tile-based overlap of compute and communication; up to 2.5x end-to-end latency reduction on a prototype. Applicability: only for trusted, nearby devices on a fast LAN; targets BERT-size models and Transformers in general, not tuned for batch-1 decode on weak links.
- **EdgeShard: efficient LLM inference via collaborative edge computing** (IEEE IoT Journal 2025) [ABS] — [arXiv 2405.14371](https://arxiv.org/abs/2405.14371). Joint device selection and model partitioning solved by dynamic programming over edge devices plus cloud; Llama2 testbed; up to 50% latency reduction and 2x throughput. Applicability: DP partitioning formulation is easy to reuse for a pipeline of unequal nodes; the paper's own scope is small.
- **Jupiter: fast and resource-efficient collaborative inference of generative LLMs on edge devices** (INFOCOM 2025) [ABS] — [arXiv 2504.08242](https://arxiv.org/abs/2504.08242). Pipelined design treating prefill and decode differently; claims up to 26.1x lower end-to-end generation latency vs prior methods (baseline-dependent, treat cautiously).
- **HALO: distributed LLM inference over lossy edge networks** (arXiv Jan 2026) [ABS, via search snippet] — [arXiv 2601.11676](https://arxiv.org/abs/2601.11676). Reports 3.41x end-to-end speedup under 5% packet loss on Raspberry Pi 4 testbed. Relevant to Wi-Fi and WAN loss handling; unverified.
- **Petals: collaborative inference and fine-tuning of large models** (ACL 2023 demo; arXiv 2022) [ABS] — [arXiv 2209.01188](https://arxiv.org/abs/2209.01188). BLOOM-176B on consumer GPUs at about 1 step/s over the internet by chaining volunteer servers; clients choose a server chain to minimize time. Applicability: the only internet-scale volunteer pipeline precedent, including fault tolerance via re-routing; limits: dense model, trusts servers with activations, 1 step/s is for sequential generation, no verification.
- **SWARM parallelism** (ICML 2023) [ABS] — [arXiv 2301.11913](https://arxiv.org/abs/2301.11913). Randomized pipelines with rebalancing on heterogeneous, preemptible, slow (under 200 Mb/s) nodes; 1B model trained on 400 preemptible T4s. Training, not inference; lesson is stage rebalancing and stochastic wiring under churn, plus the square-cube argument (bigger hidden size, relatively less communication per compute).
- **TPI-LLM** (tensor parallelism for 70B on low-resource devices) [ABS, only seen as a one-line snippet claiming tensor parallelism can beat pipeline on low-resource devices; I did not verify the paper, ID or numbers].
- **HexGen-2: disaggregated generative inference in heterogeneous environment** (ICLR 2025) [ABS] — [arXiv 2502.07903](https://arxiv.org/abs/2502.07903). Graph partitioning plus max-flow for prefill/decode split and KV-cache transfer on heterogeneous datacenter GPUs; up to 2.0x (avg 1.3x) throughput, or comparable performance at 30% lower price. Not edge; useful for scheduling formulation.
- **Helix: serving LLMs over heterogeneous GPUs and network via max-flow** (ASPLOS 2025) [ABS] — [arXiv 2406.01566](https://arxiv.org/abs/2406.01566). MILP jointly picks model placement and per-request routing as a max-flow on a graph of GPU and network capacities; 24-42 GPU nodes, up to 2.7x throughput; built on vLLM. Latency-reduction numbers differ between arXiv (2.8x/1.3x) and ASPLOS (66%/24%) versions. Applicability: placement and per-request routing model for a mesh with unequal links; datacenter-scale, assumes reliable nodes.

### Inferences
- Pipeline (layer) partitioning over a ring is the proven pattern for batch-1 on weak Wi-Fi/Ethernet because it moves only one hidden vector (a few KB to tens of KB per token) per hop. Tensor parallelism needs two all-reduces per layer, which hurts on 0.3-1 Gbps links with milliseconds of latency.
- Halda/Helix/EdgeShard give three reusable scheduling formulations (ILP, max-flow MILP, DP) for unequal nodes.

### Gaps
- LinguaLinked, Hepti, PipeEdge, DistInfer, Mobile-LLM collaborative papers: I found no results in my searches for LinguaLinked and did not search the others, so they are not covered. Do not cite them without checking.
- No measured WAN tok/s for dense pipelines other than Petals' 1 step/s claim and the WAN paper in section 3.

## 2. MoE-specific (edge offloading, expert parallel, distributed)

### Takeaway
Single-node offloading papers (Fiddler, HOBBIT, ProMoE) hide host-to-GPU weight loads. Distributed edge MoE work (OD-MoE, Prism, WDMoE) is early, small, and mostly abstract-level. ExFlow shows that cross-layer expert affinity can cut all-to-all traffic, but on datacenter GPUs. For batch-1 on slow links, activation-only transfer (send the small hidden vector to where the expert lives, as Fiddler does within a node) beats moving expert weights.

### Cited Findings
- **OD-MoE: on-demand expert loading for cacheless edge-distributed MoE inference** (arXiv Dec 2025) [partial: abstract page only] — [arXiv 2512.03927](https://arxiv.org/abs/2512.03927). Ten-node testbed; a quantized shadow model predicts expert activations several layers ahead (reported 99.94% accuracy); experts loaded just before use and evicted; about 75% of fully-GPU-cached decoding speed with about 1/3 of GPU memory; nodes with under 1 GB GPU memory. Page gave no GPU models or network bandwidth. Applicability: closest published match to "many small nodes, MoE"; the predictor-ahead idea fits a mesh where experts are sharded. Limits: one testbed, not peer-reviewed, network unspecified.
- **Prism: collaborative MoE serving across heterogeneous GPU edge servers** (arXiv Aug 2025) [ABS] — [arXiv 2508.12851](https://arxiv.org/abs/2508.12851). Expert placement to minimize inter-server communication; up to 30.6% latency reduction.
- **WDMoE: wireless distributed MoE** (arXiv Nov 2024) [ABS] — [arXiv 2411.06681](https://arxiv.org/abs/2411.06681). Gating at a base station, experts on mobile devices (Jetson kits), joint expert selection and bandwidth allocation.
- **D2MoE** (arXiv Apr 2025) [ABS] — [arXiv 2504.15299](https://arxiv.org/abs/2504.15299). Single-device bit-width-per-expert and I/O-compute pipelining.
- **OrderMoE** (expert-similarity-driven edge MoE) [ABS, not confirmed: ID 2607.17154 seen only on an aggregator; treat as unverified].
- **Fiddler: CPU-GPU orchestration for fast inference of MoE models** [ABS] — [arXiv 2402.07033](https://arxiv.org/abs/2402.07033). Runs expert layers on CPU and ships only activations; Mixtral-8x7B (over 90 GB) at over 3 tok/s on one 24 GB GPU. Applicability: principle "move activations, not weights" for a node whose RAM holds experts but whose GPU is weak.
- **HOBBIT: mixed-precision expert offloading** [ABS] — [arXiv 2411.01433](https://arxiv.org/abs/2411.01433). Replace cache-miss, low-importance experts with low-precision copies; token-level loading, layer-level prefetch, sequence-level cache; built on llama.cpp; up to 9.93x decode speedup vs offloading baselines. Applicability: llama.cpp-based, edge hardware, directly portable idea.
- **ProMoE: proactive caching** [ABS] — [arXiv 2410.22134](https://arxiv.org/abs/2410.22134). Predicts and prefetches experts; average speedups reported as 2.13x prefill / 2.84x decode (v1) and 2.20x / 2.07x (later version); consumer GPU. Version numbers differ.
- **ExFlow: exploiting inter-layer expert affinity** [ABS] — [arXiv 2401.08383](https://arxiv.org/abs/2401.08383). Context-coherent expert parallelism with one all-to-all instead of two; up to 2.2x throughput; up to 40% less remaining all-to-all data; no fine-tuning needed. Applicability: supports co-activation-based expert placement; tested on GPU clusters, not edge or WAN.
- MoE-Lightning, EdgeMoE, AdapMoE, SwapMoE, DAOP, ExpertFlow, MoE-Infinity, Pre-gated MoE, Lynx: **not verified in this session**, so no claims. Look up before citing.

### Inferences
- With top-k routing at batch 1, each token touches few experts per layer. Sharding experts across nodes needs a round trip per layer per token (latency-bound: layers x RTT), so on LAN (3-7 ms Wi-Fi) it is poor unless the router prediction (OD-MoE style) lets you prefetch or the token is sent only to the owning node(s) pipelined with speculation. Layer pipeline plus per-node expert caches is the safer first design; expert parallel pays off with batched requests.

### Gaps
- No measured numbers for expert parallelism over 1 GbE / Wi-Fi / WAN at batch 1 found in a verified source. OD-MoE bandwidth unknown.
- Wire compression of activations for MoE: not found.

## 3. Speculative decoding over networks, WAN split inference, privacy

### Takeaway
Over high-RTT links, per-token round trips dominate, so speculation helps only modestly (tokens accepted per step about 1.2-1.3 in the one measured WAN study). Splitting by putting the first and last layers on the user's machine gives measurable but not formal privacy.

### Cited Findings
- **Privacy-aware split inference with speculative decoding over WANs** (arXiv Feb 2026) [FULL page] — [arXiv 2602.16760](https://arxiv.org/abs/2602.16760). RTX 3090 holds embedding plus first 2 and last 2 layers and LM head; cloud A100 runs the middle. Public internet via SSH tunnel, RTT 78-85 ms, fp16 activations over WebSocket binary. Mistral 7B and NeMo 12B. Lookahead decoding: 8.7-9.3 tok/s (7B), 7.8-8.7 tok/s (12B); 13-17 tok/s on a lower-latency link; local-only 3090 about 39 tok/s. Transport engineering alone went 1.4 to 7.8 tok/s (HTTP with torch.save to binary WebSocket). Tokens accepted per step about 1.2-1.3 (code 1.43-1.57, creative text 1.06-1.12); break-even about 1.15-1.20. Inversion attack top-1 accuracy: 58.8% with 2 local layers, 44.3% (4), 44.8% (6), 34.8% (8); random 0.003%. Limits: weak attacker (MLP, about 880 training samples, about ±10% variance), no formal guarantees, one model family, greedy only, 70B/405B figures are estimates. Applicability: best available measured WAN figure; shows that RTT, serialization and transport matter more than model compute; 2 local layers do not hide the prompt.
- **DSD: distributed speculative decoding for edge-cloud** (arXiv Dec 2025) [ABS, only via aitopics mirror; arXiv ID not confirmed] — reported up to 1.1x speedup. **PicoSpec** [arXiv 2603.19133](https://arxiv.org/abs/2603.19133) [ABS] criticizes stop-and-wait designs and proposes async pipelining. **FlowSpec** [arXiv 2507.02620](https://arxiv.org/abs/2507.02620) [ABS] pipelines tree-based speculation and notes that links under 1000 Mbps are a bottleneck for distributed inference. **SLED** [arXiv 2506.09397](https://arxiv.org/abs/2506.09397) [ABS] and **GoodSpeed** [arXiv 2512.09963](https://arxiv.org/abs/2512.09963) [ABS] cover edge draft plus central verify.
- Hidden-state inversion: Split-learning LLM inversion work (RevertLM, "Model inversion in split learning for personalized LLMs", [arXiv 2501.05965](https://arxiv.org/abs/2501.05965)) [ABS] reports recovery of inputs from intermediate layers, with attack scores 38-75% across scenarios and over 60% improvement over the prior state of the art. Related 2026 preprints (e.g. [arXiv 2604.24468](https://arxiv.org/abs/2604.24468) survey) [ABS] confirm that intermediate activations retain strong semantic content. Takeaway: assume any node in the middle of the pipeline can reconstruct much of the prompt.

### Inferences
- A mesh on the public internet should not claim privacy from relay nodes; keep the first and last layers (and tokenizer, embeddings, LM head) on the client, and note that this reduces but does not remove leakage. Cryptographic MPC/HE for LLM layers was not surveyed.

### Gaps
- Did not verify DSD paper ID; did not find quantitative bandwidth studies of DSD.
- No survey of MPC/HE or TEE-based private inference was done.

## 4. Verification of untrusted nodes and incentives

### Cited Findings
- **TOPLOC: locality-sensitive hashing for trustless verifiable inference** (ICML 2025) [ABS] — [arXiv 2501.16007](https://arxiv.org/abs/2501.16007), [PMLR](https://proceedings.mlr.press/v267/ong25a.html). Commits to intermediate activations; verification up to 100x faster than inference; about 258 bytes per 32 tokens (Llama-3.1-8B) vs about 262 KB for raw embeddings; claims 100% detection of changed model, prompt or precision in their tests (authors' own evaluation). Argues zero-knowledge proofs have 100x-10,000x prover overhead. Applicability: cheap spot-check of a node's claimed model and precision; limits: tested for whole-model inference by one provider, not intermediate pipeline stages; tolerance to GPU non-determinism across very different hardware (old iGPUs) is not shown.
- **VeriLLM: publicly verifiable decentralized inference** [ABS] — [arXiv 2509.24257](https://arxiv.org/abs/2509.24257). Lightweight re-running plus cryptographic commitments, assumes one honest verifier.
- Incentive systems (tokens, reputation, Petals' points): not covered by verified sources in this session.

### Gaps
- No paper found on verifying a single stage of a layer pipeline. No incentives paper verified.

## Top 10 by usefulness to a LAN-then-internet MoE mesh on old iGPU/GPU PCs and phones
1. prima.cpp / Halda (arXiv 2504.08791) [FULL] — closest system, real numbers, mixed phones and PCs, scheduler ideas.
2. Privacy-aware split inference with speculative decoding over WANs (2602.16760) [FULL] — only measured internet-RTT figures; transport and privacy lessons.
3. OD-MoE (2512.03927) [partial] — distributed MoE with expert prediction on tiny nodes.
4. Petals (2209.01188) [ABS] — internet-scale volunteer pipeline precedent.
5. Helix (2406.01566) [ABS] — placement and routing formulation for unequal nodes and links.
6. HOBBIT (2411.01433) [ABS] — llama.cpp-based MoE expert offloading, mixed precision.
7. Fiddler (2402.07033) [ABS] — activations-not-weights principle for CPU-resident experts.
8. TOPLOC (2501.16007) [ABS] — cheap verification of untrusted nodes.
9. ExFlow (2401.08383) [ABS] — expert affinity for placement; cuts all-to-all.
10. SWARM parallelism (2301.11913) [ABS] — churn-tolerant stage rebalancing on slow heterogeneous nodes.
Runners-up: EdgeShard, Galaxy, Jupiter, ProMoE, Prism, HexGen-2, FlowSpec.

## Cross-cutting lessons (inference, flagged)
- Batch-1 decode over a network is latency-bound: pipeline (1 hop per stage) beats tensor or expert parallel (multiple syncs per layer) on Wi-Fi/WAN. Overlap and speculation help only if acceptance rates exceed about 1.2 tokens/step (per the WAN study).
- Serialization and transport engineering gave a 5.6x gain in the WAN study; do not neglect it.
- Memory-bound weak nodes benefit more from mmap/prefetch and dropping the slowest device than from adding every device (prima.cpp).
- Everything marked [ABS] should be checked in the full paper before it goes into a report as a hard number.
