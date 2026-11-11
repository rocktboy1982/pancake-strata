# Existing open-source distributed / networked LLM inference systems (as of 2026-11-10)

Method note: GitHub star/license/pushed_at figures come from the GitHub REST API queried 2026-11-10 (https://api.github.com/repos/OWNER/REPO), and README facts from WebFetch of the repos. "Measured" = a number from a paper table or a user log. "Claim" = README or marketing text. I did not run anything myself. Paper tables were read through an HTML-to-text extraction (Petals table layout flagged ambiguous in places).

## 1. What measured tok/s do they report on LAN (1 GbE, Wi-Fi) and WAN, for which models?

### Takeaway
Pipeline/layer-split systems are latency-bound, not bandwidth-bound, and real measured numbers are small. llama.cpp RPC buys capacity, not speed (7B decode drops from 92 to 53 tok/s over 10GbE). prima.cpp is the only system benchmarked on a Wi-Fi home cluster (320-610 Mbps, 3-7 ms): 1.5 tok/s on a 70B dense model, 15 tok/s on 8B (one device only). Petals is the only one with real-internet numbers: 0.83 steps/s for BLOOM-176B across two continents. distributed-llama (tensor parallel) works on a wired switch and is poor over Wi-Fi.

### Cited Findings

**prima.cpp (paper, ICLR 2026; Wi-Fi home cluster, measured)**
- Cluster D1-D4: Mac M1 (2.4 GiB avail RAM), laptop i9 + RTX 3070, desktop i9 + 2080 Ti, Huawei Mate 40 Pro (HarmonyOS). Connected via a Wi-Fi router, bandwidth 320-610 Mbps, link latency 3-7 ms. Total RAM+VRAM 37 GiB, too small for a Q4K 70B model. — [arXiv HTML v2](https://arxiv.org/html/2504.08791v2)
- TPOT (ms/token), llama.cpp vs prima.cpp: 8B 15 vs 15; 14B 20 vs 20; 30B 202 vs 72; 45B 328 vs 233; 60B 7965 vs 468; 65B 8807 vs 569; 70B 10120 vs 674 (about 1.5 tok/s). TTFT at 70B: 10806 vs 1793 ms. — [arXiv HTML v2](https://arxiv.org/html/2504.08791v2)
- exo: 8B TPOT 263 ms (TTFT 960); OOM at 70B. dllama: 8B TPOT 459 ms (TTFT 1845); OOM at 70B. exo and dllama were not run on 14-65B. — [arXiv HTML v2](https://arxiv.org/html/2504.08791v2)
- With speculative decoding: 442 ms/token for 70B and 26 tok/s for a 32B model. — [arXiv HTML v2](https://arxiv.org/html/2504.08791v2)
- README table: Qwen-2.5-72B 12227 vs 867 ms; DeepSeek-R1-Distill-Llama-70B 10978 vs 724 ms (llama.cpp vs prima). — [OpenCPIL/prima.cpp](https://github.com/OpenCPIL/prima.cpp)
- Reading: the huge llama.cpp-vs-prima gap at 60B+ is mostly because llama.cpp mmap-thrashes on a single device that cannot hold the model (TPOT above 7 s). It is not a network win. For models that fit on one device prima uses one device (8B and 14B equal llama.cpp). — [arXiv HTML v2](https://arxiv.org/html/2504.08791v2)

**llama.cpp RPC (individual measurements; one hardware pair each)**
- Mac Studio M2 Ultra + DGX Spark, direct 10GbE cable (9.41 Gbps iperf3). Qwen2.5-7B Q4_K_M: local Metal 76 pp / 92 tg tok/s; RPC 318 pp / 53 tg. Qwen2.5-72B Q4_K_M: local 28 pp / 11.1 tg; RPC (31 GB Metal + 14 GB CUDA) 30 pp / 5.9 tg. Author estimates about 16 ms network overhead per token across 80 layers. — [dev.to](https://dev.to/ljkunal/distributed-llm-inference-across-nvidia-blackwell-and-apple-silicon-over-10gbe-2feg); same data at [kjaiswal/llama-cpp-distributed-benchmarks](https://github.com/kjaiswal/llama-cpp-distributed-benchmarks). That README says 1GbE "works but slower" but has no 1GbE numbers. Single author, single commit.
- Jeff Geerling, 4x Mac Studio (M3 Ultra), llama.cpp RPC: performance drops as nodes are added; only a "little speedup" for DeepSeek V3.1 671B; Thunderbolt beats Ethernet on latency even without RDMA. No numbers in text (charts only). — [jeffgeerling.com](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5)
- UNVERIFIED (search-result summaries, pages not opened): a May 2024 r/LocalLLaMA comment says a Mac over Wi-Fi dropped to about 12 tok/s generation vs 48 tok/s wired ([thread](https://redlib.hbubli.cc/r/LocalLLaMA/comments/1cyzi9e/llamacpp_now_supports_distributed_inference/msugpl0)); an April 2026 report of dual RTX 5090 running a 122B MoE at 96 tok/s over 2.5GbE ([popularai.org](https://www.popularai.org/p/10gbe-local-ai-llm) is the related 10GbE write-up, whose conclusion is that RPC buys capacity before speed).

**exo (third-party measured)**
- 4x Mac Studio M3 Ultra over Thunderbolt 5 RDMA: Qwen3-235B 32 tok/s; Kimi K2 Thinking (1T, 32B active) about 30 tok/s. Exo speeds up as nodes are added; README claims 1.8x on 2 devices, 3.2x on 4 (images only). — [Geerling](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5); [exo README](https://github.com/exo-explore/exo)
- On the Wi-Fi home cluster exo got 263 ms/token for 8B. The paper says exo splits layers by memory ratio, overloading slow devices. — [arXiv HTML v2](https://arxiv.org/html/2504.08791v2)
- No exo numbers over plain 1 GbE found from a primary source.

**distributed-llama (b4rtaz; user logs in GitHub Discussions; tensor parallel; CPU)**
- 2x Ryzen 3600 over 1 Gbit switch (in WSL), Llama 3 8B Q40: 2.15 tok/s on 2 nodes (v0.3.1), 2.60 after a later commit, vs 3.69 tok/s on a single node. Distributing was slower than one machine. — [Discussion #41](https://github.com/b4rtaz/distributed-llama/discussions/41)
- 4x Raspberry Pi 5 (8 GB), wired switch, Qwen3-30B-A3B Q40 (MoE): 13.04 tok/s decode, 14.33 eval (v0.16.0). A later 4x Pi 5 16 GB patched run: 15.1 tok/s decode. — [Discussion #255](https://github.com/b4rtaz/distributed-llama/discussions/255)
- 2x OrangePi Plus 16 GB, direct 2.5 GbE, Qwen3-30B-A3B Q40: 7.29 tok/s decode, 11.55 eval. — [Discussion #266](https://github.com/b4rtaz/distributed-llama/discussions/266)
- 8x Snapdragon 8 Gen 2 nodes, Llama-architecture 8B-class (dim 4096, 32 layers, 8 KV heads): 5.08 tok/s decode, 6.67 eval; link type not stated; model not named. — [Discussion #180](https://github.com/b4rtaz/distributed-llama/discussions/180)
- prima.cpp's analysis: on Wi-Fi at 3-7 ms, dllama's 64 all-reduces per token for 8B cost at least 192 ms; at 70B the authors measured about 150 ms per all-reduce and about 30 s/token. — [arXiv HTML v2](https://arxiv.org/html/2504.08791v2)
- More benchmark discussions exist (4x Mac Mini M4 Pro, RTX 3060/3090/4090 Vulkan, Llama 2 70B on 4 GCP VMs); not opened, numbers not captured. — [Discussions](https://github.com/b4rtaz/distributed-llama/discussions)

**Petals (paper; batch 1; sequential steps/s, seq len 128)**
- Llama 2 70B on 3x T4: 2.29 steps/s at 1 Gbit/s (<5 ms) and at 100 Mbit/s (<5 ms); 1.57 at 100 Mbit/s with 100 ms RTT. Offloading: 0.139. — [arXiv 2312.08361 HTML](https://arxiv.org/html/2312.08361v1)
- BLOOM-176B on 3x A100: 1.71 / 1.66 / 1.23 (1 Gbit<5 ms / 100 Mbit<5 ms / 100 Mbit 100 ms). 10x RTX 3090: 1.17 / 1.05 / 0.34. Real world, 14 heterogeneous servers across Europe and North America: 0.83 steps/s. — [same](https://arxiv.org/html/2312.08361v1)
- README claims "up to 6 tokens/sec" for Llama 2 70B and 4 for Falcon 180B with no network conditions (a claim; higher than the paper's measured 2.3). — [Petals README](https://github.com/bigscience-workshop/petals)
- Caveat: paper hardware is T4/A100/3090-class with 4-bit/8-bit weights, not old PCs.

**Parallax, mesh-llm**
- Neither README gives tok/s. Block's mesh-llm post gives no throughput/latency figures. Parallax paper (arXiv 2509.26182) not opened for numbers. — [Parallax](https://github.com/gradientHQ/parallax); [Block blog](https://engineering.block.xyz/blog/buzz-sharing-compute-powered-by-meshllm)

### Inferences
- Over 1 GbE-class LAN expect decode near single-device speed minus a per-token round-trip cost (about 16 ms/token for 80 layers over 10GbE in the dev.to run; likely more on 1 GbE or Wi-Fi). Speed gains only appear when pooling lets the model stay in fast memory instead of swapping from disk.
- Petals data: 100 ms RTT costs about 30% on 3 hops (2.29 to 1.57) and about 3x on 10 hops (1.17 to 0.34); a 10x bandwidth drop costs little. Cost scales with hops x RTT.
- A 70B at about 1-2 tok/s is the realistic ceiling on old-PC clusters over Wi-Fi (prima: 1.5 tok/s).

### Gaps
- No primary-source 1 GbE (vs 10GbE) llama.cpp RPC numbers for a standard model.
- Petals README "6 tok/s" setup not published.
- Parallax and mesh-llm throughput not found.
- exo on 1 GbE or Wi-Fi with models above 8B not found.

## 2. What bytes per token cross the network for pipeline splits, and what does that imply?

### Takeaway
One hidden-state vector per stage boundary per decoded token: hidden_size x bytes_per_element, i.e. roughly 8-32 KiB per hop per token. Bandwidth is irrelevant for single-stream decode; the cost is per-hop latency. Tensor parallelism needs several all-reduces per layer per token, which is why it collapses on Wi-Fi.

### Cited Findings
- Petals: per-token activations are "several kilobytes," so sequential inference is latency-bound, not bandwidth-bound. A 12288-dim token embedding in 16-bit is 24 KiB (their footnote). BLOOM-176B hidden dim 14336. Dynamic blockwise int8 quantization of hidden states halves bandwidth with no noticeable quality effect. — [arXiv 2312.08361 HTML](https://arxiv.org/html/2312.08361v1)
- prima.cpp: each layer window sends one FP32 vector of size e (4e bytes) to the next device, once per window per token. KV cache is FP16 and stays local. GPU devices need an extra RAM-to-VRAM and VRAM-to-RAM copy of 4e bytes per window. — [arXiv HTML v2](https://arxiv.org/html/2504.08791v2)
- Llama-3-8B config in a dllama log: dim 4096, hidden dim 14336, 32 layers, 8 KV heads. — [Discussion #180](https://github.com/b4rtaz/distributed-llama/discussions/180)
- dllama (tensor parallel): 64 all-reduces per token for 8B. — [arXiv HTML v2](https://arxiv.org/html/2504.08791v2)
- llama.cpp RPC README calls the backend a proof of concept, "fragile and insecure." — [RPC README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)
- UNVERIFIED (search summary of a Reddit thread, not opened): RPC traffic hovering around 300 Mbps and latency said to matter more than bandwidth. — [r/LocalLLaMA](https://redlib.hbubli.cc/r/LocalLLaMA/comments/1cyzi9e/llamacpp_now_supports_distributed_inference/mstwc4z)

### Inferences
- My arithmetic (not sourced): Llama-3-8B (dim 4096) one hop = 8 KiB fp16, 16 KiB fp32, about 4 KiB int8. A 70B (dim 8192, from memory, unverified) = 16 KiB fp16 / 32 KiB fp32. At 10 tok/s that is well under 1 Mbit/s per hop.
- Prefill differs: bytes scale with prompt length (2048 tokens x 16 KiB = 32 MiB per hop at dim 8192 fp16), so bandwidth matters for TTFT. The dev.to run saw 4x prefill speedup but decode slowdown with RPC.
- Decode latency per token approx = sum of stage compute + hops x (RTT + packet overhead). A single request leaves stages idle unless multiple requests are pipelined or speculative decoding is used (mesh-llm describes both).

### Gaps
- No source gives a measured packet-size breakdown for llama.cpp RPC (it also sends graph/tensor metadata); I did not read the RPC protocol source.

## 3. Which handle heterogeneous devices (mixed GPU vendors, CPU-only, phones)?

### Takeaway
prima.cpp (CUDA + CPU + Android/HarmonyOS via Termux, with a scheduler) and mesh-llm (Metal, CUDA, ROCm, Vulkan, CPU on three OSes) are broadest. llama.cpp RPC supports any ggml backend per server. distributed-llama is CPU/ARM-focused with experimental Vulkan but needs power-of-two nodes. exo accelerates only on Apple Silicon (MLX). Petals and Parallax assume NVIDIA GPUs (Parallax adds Mac via MLX).

### Cited Findings
- prima.cpp: macOS, Linux, Android, HarmonyOS (Android/HarmonyOS via Termux); GPU support CUDA only (Vulkan and AMD planned); Windows not yet; iOS not mentioned. Halda scheduler picks devices and CPU/GPU layer split under RAM/VRAM limits, drops devices that slow things down. — [OpenCPIL/prima.cpp](https://github.com/OpenCPIL/prima.cpp); [arXiv HTML v2](https://arxiv.org/html/2504.08791v2)
- mesh-llm: Metal on macOS; Linux CPU, ARM64 CPU, ARM64 CUDA, CUDA, Blackwell CUDA, ROCm, Vulkan; Windows CPU, CUDA, ROCm, Vulkan. Rust; QUIC/Iroh transport; Nostr discovery for public meshes, invite tokens for private. — [mesh-llm repo](https://github.com/Mesh-LLM/mesh-llm); [Block blog](https://engineering.block.xyz/blog/buzz-sharing-compute-powered-by-meshllm)
- llama.cpp RPC: server exposes all local ggml devices (or CPU); weights/KV split in proportion to memory, tunable with --tensor-split; RDMA over Linux RoCEv2 and macOS Thunderbolt 5 (macOS 26.2+), negotiated at handshake, TCP fallback. — [RPC README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)
- distributed-llama: Linux/macOS/Windows; ARM and x86-64 AVX2 optimized; experimental Vulkan; Raspberry Pi guide; node count power of two and at most KV heads; quantization limited to q40 weights with q80 buffer, or f32/f32. — [README](https://github.com/b4rtaz/distributed-llama)
- exo: Apple Silicon (macOS 15.1+; RDMA needs 26.2+); Linux CPU only, GPU "under development"; Windows and iOS not mentioned; Mac and Linux nodes will not join the same cluster; mixed versions unsupported. — [exo README](https://github.com/exo-explore/exo). An older undated TechRadar piece said iOS/Android worked but the iOS build was pulled — [TechRadar](https://www.techradar.com/computing/bittorrent-for-llm-exo-software-is-a-distributed-llm-solution-that-can-run-even-on-old-smartphones-and-computers) (possibly outdated). A user found exo's MLX ring backend hangs on CUDA (MLX 0.31.1). — [dev.to](https://dev.to/ljkunal/distributed-llm-inference-across-nvidia-blackwell-and-apple-silicon-over-10gbe-2feg)
- Parallax: GPU backend through SGLang/vLLM, Mac through MLX LM; Lattica P2P; two-phase scheduler (layer placement, then per-request pipeline). — [Parallax repo](https://github.com/gradientHQ/parallax); [paper](https://arxiv.org/html/2509.26182v1)
- Petals: PyTorch/Transformers GPU servers; tested on RTX 3060/2080Ti/3090/A4000/A5000; no phone support found. — [Petals README](https://github.com/bigscience-workshop/petals); [paper](https://arxiv.org/html/2312.08361v1)

### Inferences
- For old PCs with iGPUs, only llama.cpp RPC (Vulkan/SYCL via ggml) and mesh-llm (Vulkan) plausibly use them; no source gives iGPU numbers.
- For phones, prima.cpp (Termux) and distributed-llama (ARM, Snapdragon logs) have the only measured evidence.

### Gaps
- No evidence of any project running natively on iOS today; exo's iOS state unverified.
- No measured data for mixed AMD/NVIDIA/Intel iGPU clusters.
- mesh-llm iGPU/phone behavior unverified.

## 4. Maturity and maintenance (GitHub API, queried 2026-11-10)

### Takeaway
Active: llama.cpp (MIT), exo (Apache-2.0), mesh-llm (Apache-2.0), distributed-llama (MIT). Stale: Petals (last push Sep 2024), Parallax (Jul 2026). prima.cpp's original repo returns 404; an "[Official]" OpenCPIL repo exists with few commits.

### Cited Findings
- llama.cpp: about 133.9k stars, MIT, last push 2026-11-10. — [API](https://api.github.com/repos/ggml-org/llama.cpp)
- exo: about 48.3k stars, Apache-2.0, last push 2026-11-06, 206 open issues / 252 open PRs; latest GitHub release via API v1.0.71 (2026-04-23). README says experimental. — [API](https://api.github.com/repos/exo-explore/exo); [README](https://github.com/exo-explore/exo)
- distributed-llama: about 3.1k stars, MIT, last push 2026-11-09, latest release v0.17.0 (2026-11-07). Qwen3 MoE added Sep 2025, Vulkan Mar 2025. — [API](https://api.github.com/repos/b4rtaz/distributed-llama); [README](https://github.com/b4rtaz/distributed-llama)
- mesh-llm: about 4.0k stars, Apache-2.0, last push 2026-11-10, 132 open issues; self-described "experimental distributed-systems software"; a third-party page says launched 2026-07-11 (unverified); keeps a small llama.cpp patch set. — [API](https://api.github.com/repos/Mesh-LLM/mesh-llm); [README](https://github.com/Mesh-LLM/mesh-llm); [SourcePulse](https://www.sourcepulse.org/projects/27326722)
- Parallax: about 1.4k stars, Apache-2.0, last push 2026-07-01, v0.0.1 Oct 2025. — [API](https://api.github.com/repos/gradientHQ/parallax)
- Petals: about 10.7k stars, MIT, last push 2024-09-07 (over 2 years stale). — [API](https://api.github.com/repos/bigscience-workshop/petals)
- Hivemind (library under Petals): about 2.6k stars, MIT, last push 2026-01-11. — [API](https://api.github.com/repos/learning-at-home/hivemind)
- prima.cpp: Lizonghang/prima.cpp returned 404 (WebFetch and API). OpenCPIL/prima.cpp labeled "[Official]," MIT, about 129 stars, 9 commits, last push 2026-09-06. Paper accepted at ICLR 2026. — [OpenCPIL](https://github.com/OpenCPIL/prima.cpp); [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e991e5587c1daa49bbf9a818b3f02f9a-Abstract-Conference.html)
- llama.cpp RPC: "proof of concept," never run on open networks. — [RPC README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)

### Inferences
- exo's star count outpaces its independently measured evidence; Geerling reported instability and unresponsive developers.
- prima.cpp is a research fork of an older llama.cpp; reusing its ideas is easier than reusing code.

### Gaps
- Whether Lizonghang/prima.cpp moved to OpenCPIL or was removed: unverified.
- exo's current release line may not match the API's "latest release" (v1.0.71, Apr 2026) given Nov 2026 pushes.

## 5. Known problems

### Takeaway
Common failure modes: latency-bound decode, node-count constraints, OOM/disk thrash, discovery blocked on guest/office Wi-Fi, version mismatches, no security on RPC, and trust/privacy on open networks.

### Cited Findings
- llama.cpp RPC: decode slower than local when the model fits; no security; hosts built from the same commit; RDMA point-to-point only. — [dev.to](https://dev.to/ljkunal/distributed-llm-inference-across-nvidia-blackwell-and-apple-silicon-over-10gbe-2feg); [RPC README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)
- exo: RDMA needs full mesh of TB5 cables and matching macOS versions; guest Wi-Fi can block discovery; Linux CPU only. Geerling: cluster needed reboots when RDMA failed; model loads sometimes failed. — [exo README](https://github.com/exo-explore/exo); [Geerling](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5)
- prima.cpp: CUDA only, no Windows, ports 9000/10000 must be open, weak devices can get "No layer is assigned to me, exit," output layer on CPU by default. — [OpenCPIL README](https://github.com/OpenCPIL/prima.cpp)
- distributed-llama: power-of-two nodes, node cap equals KV heads (issue #70), narrow quant support, no network benchmarking per maintainer, 2-node 1 GbE slower than single node. — [README](https://github.com/b4rtaz/distributed-llama); [#41](https://github.com/b4rtaz/distributed-llama/discussions/41); [#180](https://github.com/b4rtaz/distributed-llama/discussions/180)
- mesh-llm: Windows CUDA lacks RTX 50-series native support, public mesh best-effort, prompts visible to node owners, binary not code-signed. — [repo](https://github.com/Mesh-LLM/mesh-llm); [Block blog](https://engineering.block.xyz/blog/buzz-sharing-compute-powered-by-meshllm)
- Petals: public swarm availability depends on volunteers; stale since Sep 2024. — [README](https://github.com/bigscience-workshop/petals)
- prima.cpp paper: vLLM and SGLang failed on the home cluster because they are server-oriented; exo was not run on the phone because it needs root. — [arXiv HTML v2](https://arxiv.org/html/2504.08791v2)

### Inferences
- Data-center frameworks (vLLM/Ray, SGLang) are not built for slow, flaky, heterogeneous links, but only one source (prima.cpp) demonstrates this.

### Gaps
- vLLM/Ray pipeline parallel, MLC/WebLLM multi-device, and Hivemind internals were NOT investigated; no sources collected. Do not rely on this file for them.

## 6. MoE support and efficiency

### Takeaway
distributed-llama has the best user evidence for MoE (Qwen3-30B-A3B at 13 tok/s on 4 Raspberry Pis). exo shows big MoE at about 30 tok/s but only on Thunderbolt 5 RDMA. mesh-llm and Parallax list MoE models. No source I found demonstrates efficient expert-parallel splitting over a slow network.

### Cited Findings
- distributed-llama: Qwen3 MoE on CPU (5 Sep 2025) and Vulkan (16 Sep 2025); Qwen3-30B-A3B Q40 is 17.0 GB; 13.04 tok/s on 4x Pi 5; 7.29 tok/s on 2x OrangePi (2.5 GbE). — [README](https://github.com/b4rtaz/distributed-llama); [#255](https://github.com/b4rtaz/distributed-llama/discussions/255); [#266](https://github.com/b4rtaz/distributed-llama/discussions/266)
- exo: Qwen3-235B 32 tok/s and Kimi K2 Thinking about 30 tok/s on 4x M3 Ultra with RDMA. — [Geerling](https://www.jeffgeerling.com/blog/2025/15-tb-vram-on-mac-studio-rdma-over-thunderbolt-5)
- Parallax model list includes DeepSeek V3.2/R1, Kimi-K2, Qwen3.6-35B-A3B, gpt-oss-120b. — [repo](https://github.com/gradientHQ/parallax)
- Petals README lists Mixtral 8x22B. — [README](https://github.com/bigscience-workshop/petals)
- mesh-llm: README mentions package-backed stages for DeepSeek3 and EXAONE-MoE and an optional Flash-MoE plugin for SSD expert streaming. A third-party blog claimed "expert sharding with zero cross-node traffic" but the README did not confirm cross-node expert distribution (conflicting/unverified). — [repo](https://github.com/Mesh-LLM/mesh-llm); [Wavect blog](https://wavect.io/zh/blog/mesh-llm-distributed-inference-multiple-computers.md)
- prima.cpp: README models all dense; no MoE mention. — [OpenCPIL README](https://github.com/OpenCPIL/prima.cpp)
- Unmeasured projection: 4-8 tok/s for 200B+ MoE across Mac Studio + DGX Spark over 10GbE. — [dev.to](https://dev.to/ljkunal/distributed-llm-inference-across-nvidia-blackwell-and-apple-silicon-over-10gbe-2feg)

### Inferences
- A pipeline split of an MoE still sends one hidden vector per hop per token, same as dense, so network cost per token is unchanged. MoE helps because per-token compute is small while total weights exceed one box; disk/RAM paging (prima's finding) matters more than the network.
- Tensor/expert parallelism with all-to-all per MoE layer would likely be latency-heavy on Wi-Fi; no source tested it.

### Gaps
- No measured numbers for expert-parallel across a LAN.
- Parallax and mesh-llm MoE performance not found.

## 7. Worth reusing or imitating vs avoiding

### Takeaway
Imitate: prima.cpp's Halda-style scheduler (device selection and dropping, RAM/VRAM/disk-aware layer windows, mmap prefetch), mesh-llm's transport and discovery (QUIC, Iroh NAT traversal, invite tokens, stage ordering), llama.cpp RPC for backend breadth, Petals' int8 activation compression and fault-tolerant routing. Avoid: tensor parallelism over Wi-Fi/WAN (dllama), unauthenticated RPC on open networks, Apple-only assumptions (exo) if old PCs are the target.

### Cited Findings
- prima.cpp: piped-ring parallelism overlaps disk I/O, compute and comms; Halda solves an integer program in 10-12 ms for 4-32 devices. — [arXiv HTML v2](https://arxiv.org/html/2504.08791v2)
- mesh-llm: coordinator assigns contiguous layer ranges, starts downstream stages first, publishes the stage-0 route; QUIC end-to-end encrypted including activations; speculative decoding and batching keep the pipeline busy. — [repo](https://github.com/Mesh-LLM/mesh-llm); [Block blog](https://engineering.block.xyz/blog/buzz-sharing-compute-powered-by-meshllm)
- Petals: client-side routing, fault tolerance, load balancing, int8 activation compression. — [paper](https://arxiv.org/html/2312.08361v1)
- Parallax: two-phase scheduler, Lattica P2P. — [paper](https://arxiv.org/html/2509.26182v1)
- exo: LAN auto-discovery, topology-aware parallelism using link latency/bandwidth, OpenAI/Claude/Ollama-compatible APIs. — [README](https://github.com/exo-explore/exo)
- llama.cpp RPC: local tensor cache (-c) avoids re-sending weights; comma-separated --rpc list. — [RPC README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)

### Inferences
- All converge on a hops x latency cost model, so a design should minimize stage count, fall back to one device when the model fits, make the scheduler account for disk/RAM not just compute, compress activations to int8, and pipeline requests or use speculative decoding to hide per-hop latency.
- Forking llama.cpp (prima.cpp, mesh-llm) carries a rebase burden; mesh-llm says it keeps a small patch set awaiting upstream.

### Gaps
- All licenses found are permissive (MIT or Apache-2.0); prima.cpp's MIT comes from the OpenCPIL repo only.
- I read READMEs and papers only, not source code, so wire formats and discovery details are unverified.
