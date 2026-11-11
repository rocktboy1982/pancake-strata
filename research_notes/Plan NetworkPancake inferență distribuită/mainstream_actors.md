# Mainstream local-LLM tools: multi-device / networked inference (as of Nov 2026)

Coverage note: about 16 tool calls. Primary sources were fetched for llama.cpp RPC, exo, Mesh-LLM and the LM Link FAQ. Everything else comes from search snippets or secondary blogs and is marked (unverified) where weak. Not researched, for lack of any hit or call budget: Jan, GPT4All, text-generation-webui, Open WebUI, TGI, llamafile, Foundry Local, Apple PCC, SGLang details, Ray beyond vLLM docs.

## 1. Does mainstream tooling already offer multi-device / networked inference, and how well?

### Takeaway
Yes, in four distinct flavours. (a) Layer or tensor split of one model across machines: llama.cpp RPC, KoboldCpp (wrapping RPC), LocalAI worker mode, exo, Mesh-LLM, MLX distributed. (b) Remote use of a model that sits on one other machine: LM Link, Ollama `:cloud`, Jan/Open WebUI as clients. (c) Request-level routing across nodes: LocalAI federated mode, AI Horde. (d) Datacenter pipeline and tensor parallel: vLLM/Ray. Quality is generally "experimental / proof of concept". The best-engineered consumer option is exo, and it is Apple-centric.

### Cited Findings

#### llama.cpp (RPC)
- The README describes `rpc-server` (ggml-rpc-server) exposing local ggml devices to a remote main host. The main host uses `--rpc host:port,host:port`, and `--tensor-split` overrides the default split, which is proportional to free memory of local and remote devices. Source: [tools/rpc README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)
- The README labels it a proof of concept, "fragile and insecure", and says never to run it on an open network. Source: same README.
- Build requires `-DGGML_RPC=ON` on both sides. The server exposes all accelerators by default, or CPU if there are none. `--device` and `CUDA_VISIBLE_DEVICES` select devices. Source: same README.
- Local tensor cache on the server (`-c`, default `~/.cache/llama.cpp/rpc`) avoids re-sending weights at load. Source: same README.
- RDMA transport was added. It is negotiated in the handshake with TCP fallback, uses libibverbs/RoCEv2 on Linux, and uses Thunderbolt 5 RDMA on macOS 26.2+. `GGML_RPC_NO_RDMA` forces TCP. `GGML_RPC_DEBUG` 0-3 gives timings. Source: same README. This is the 2026-era change I could verify, but the date of the merge is not confirmed.
- Version matching: a benchmark guide says both ends must build from the same commit. Source: [llama.cpp-distributed-benchmarks](https://github.com/kjaiswal/llama-cpp-distributed-benchmarks) (secondary, unverified)
- Speed caveat: RPC is for fitting models that exceed one machine's memory, and can be slower than local for small models. Sources: [Seeed wiki](https://wiki.seeedstudio.com/ai_robotics_distributed_llama_cpp_rpc_jetson/), [Discoverer HPC](https://hpcwithus.discoverer.bg/?p=439). The HPC write-up notes few cores get used on CPU worker nodes.
- Mixed backends work in principle, because each rpc-server loads its own backend (CUDA, Vulkan, Metal, CPU). Vendor mixing is not explicitly demonstrated in what I read. (unverified)
- MoE: no RPC-specific MoE offload found. The single-host MoE work is separate: PR #29887 "GPU cache for MoE experts kept in host memory" is reported merged Oct 2026, with an RTX 5090 benchmark of 67.8 vs 30.8 t/s using `--cpu-moe` plus `--moe-cache-mib`. Source: [developersdigest](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026) (secondary). A "v0.7.0 / 2.2x" claim is uncorroborated: [runaihome](https://runaihome.com/blog/llama-cpp-v0-7-0-moe-cache-gpu-guide-2026/). Open proposals: [expert-heat tiering PR #26563](https://github.com/ggml-org/llama.cpp/pull/26563), [MoE-to-disk RFC #23324](https://github.com/ggml-org/llama.cpp/discussions/23324). Merge status of both is unknown.
- WebGPU backend: not researched, a gap.

#### Ollama
- No native multi-node model splitting found. One analysis says several workers can take several requests but do not cooperate on one request. Source: [DEV: Ollama distributed inference](https://dev.to/runcai/ollama-distributed-inference-what-is-possible-what-is-not-and-when-to-move-beyond-local-serving-2ajl)
- A "native distributed inference in Ollama v4.0" claim appears at [rekcore](https://rekcore.pages.dev/projects/2026-05-06-ollama-v4/). It is uncorroborated and conflicts with the other sources (unverified, likely wrong). Version numbers in aggregators also conflict: Wikipedia says 0.40.14 on Nov 3, 2026, while another source says 0.30.10 in June ([Wikipedia](https://en.wikipedia.org/wiki/Ollama), [aiwiki](https://www.aiwiki.ai/wiki/ollama/raw)).
- Ollama Cloud: `:cloud` model suffix runs on Ollama's servers with the same API. `OLLAMA_NO_CLOUD=1` was added in 0.16.2. Source: [angelo-lima.fr](https://angelo-lima.fr/en/ollama-2026-state-of-the-art-en/), [insiderllm](https://insiderllm.com/guides/ollama-0-17-new-features/)
- Ollama has an MLX-based Apple Silicon preview (about Mar 30, 2026). Source: [ollama blog feed](https://rss-parrot.net/web/feeds/ollama.com.blog)
- Third-party wrappers around Ollama or llama.cpp RPC: SOLLOL (experimental RPC, 13B on 2-3 nodes, "not production-ready") at [PyPI sollol](https://pypi.org/project/sollol/), OLOL at [K2/olol](https://awesome.ecosyste.ms/projects/github.com%2Fk2%2Folol), and ollama-herd at [PyPI](https://pypi.org/project/ollama-herd/0.4.0/) (not read in detail).
- Specific GitHub issues and feature requests: not found, a gap.

#### LM Studio
- LM Link is a preview feature built with Tailscale. It is end-to-end encrypted, and the LM Studio Hub is used only for device discovery. It creates a dedicated tailnet. Linked devices reach only LM Studio/llmster. Known limits: it cannot reuse your existing Tailscale network, devices can appear falsely "disconnected", and duplicate models show as separate entries. Source: [LM Link FAQ](https://lmstudio.ai/docs/lmlink/basics/faq)
- It is remote use of a model on one device. The docs do not describe splitting a model across devices. Source: same FAQ. Remote models appear as local ones, and tools pointed at localhost:1234 work.
- CLI route for headless llmster: `lms login`, then `lms link enable`. Source: [Tailscale blog, Feb 25 2026](https://tailscale.com/blog/lm-link-remote-llm-access)
- Preview pricing was reported as free for 2 users x 5 devices each, with paid tiers later. Source: [Digital Citizen](https://www.digitalcitizen.life/how-to-use-local-models-remotely-with-lm-link-in-lm-studio/) (secondary). An iPhone app reaching LM Studio via LM Link is reported at [letsdatascience](https://letsdatascience.com/news/lm-studio-adds-iphone-to-mac-link-for-local-llms-97175e03) (secondary, unverified).
- Dependence on LM Studio's cloud hub and account is a notable limit.

#### KoboldCpp
- v1.114 (about May 2026) added RPC backend support, so multiple GPUs can be shared over the network. It is incompatible with context shift and fast-forward. Source: [release v1.114](https://newreleases.io/project/github/LostRuins/koboldcpp/release/v1.114)
- "Multiplayer" (v1.79, 2024) is a shared KoboldAI Lite editing session, not distributed compute. Source: [r/LocalLLaMA thread](https://reddit.birdcat.cafe/r/LocalLLaMA/comments/1h3e05z/koboldcpp_179_now_with_shared_multiplayer_ollama) (mirror)
- Horde worker mode is supported as an opt-in. Source: [koboldai.pages.dev](https://koboldai.pages.dev/KoboldCpp) (secondary)
- The latest releases seen are 1.115.2 (Jun 12) and a mention of 1.122.1. Source: [Contabo](https://contabo.com/blog/koboldcpp-vps/) (secondary).

#### LocalAI
- P2P/federated mode uses libp2p (go-libp2p) and edgevpn, with a shared token. Discovery is via DHT, or mDNS only with `LOCALAI_P2P_DISABLE_DHT=true`. Source: [LocalAI docs](https://localai.io/features/distribute/)
- Federated mode routes a whole request to one node. Worker mode shards one model across machines via llama.cpp RPC (`local-ai worker p2p-llama-cpp-rpc`) and works only with llama.cpp-compatible models. Source: same docs. Origin: LocalAI PR #2324, based on llama.cpp PR #6829.
- The docs call it experimental / tech preview. There are `/api/p2p` and `/api/p2p/stats` endpoints. A homelab user reported a federated setup failing because of backend naming. Source: [nerdzblog](https://nerdzblog.pages.dev/2026/deploying-open-llms-04) (anecdote)
- Current 2026 status of the P2P feature: not verified.

#### MLX / exo (Apple)
- RDMA over Thunderbolt 5 is supported from macOS 26.2. Apple's WWDC26 session 233 covers `mlx.distributed_config` and JACCL. Source: [WWDC26 session 233](https://developer.apple.com/videos/play/wwdc2026/233/)
- Qualifying hardware per exo: M4 Pro mini, M4 Max Studio and MacBook Pro, and M3 Ultra Studio. Source: [exo README](https://github.com/exo-explore/exo)
- exo README claims tensor-parallel speedups of up to 1.8x on 2 devices and 3.2x on 4. It has automatic discovery (cluster name, version and build must match), a dashboard at localhost:52415, and an `exo-bench` tool. Linux runs on CPU only, with GPU "under development". Macs and Linux nodes do not join each other. RDMA requires full-mesh Thunderbolt and exact macOS version matching. Windows is not mentioned. Source: [exo README](https://github.com/exo-explore/exo)
- DeepSeek V3.1 671B at 24-26 t/s on a Thunderbolt 5 Mac cluster appears only in secondary sources: [AppleInsider](https://appleinsider.com/articles/25/12/20/ai-calculations-on-mac-cluster-gets-a-big-boost-from-new-rdma-support-on-thunderbolt-5), [Geeky Gadgets](https://www.geeky-gadgets.com/thunderbolt-5-rdma-macs/). The exo README cites 4x M3 Ultra tests by Jeff Geerling without numbers in its text.
- A university report says pipeline mode was the most stable, and tensor+RDMA had intermittent bugs. Source: [UMFT project report](https://www.umft.ro/wp-content/uploads/2026/02/Project-Report_Bio-AI-Synergy_FDI-2025.pdf)
- Older exo (2024-25) marketed iPhone and heterogeneous support with ring memory-weighted partitioning. I found no official iOS install path. Source: [search results on exo](https://github.com/exo-explore/ex-exo) (archived repo). Treat phone support as unverified or abandoned.

#### Mesh-LLM (closest to NetworkPancake)
- Open source, Apache-2.0, "pools GPUs and memory across machines" behind one OpenAI-compatible endpoint (`localhost:9337/v1`). Marked experimental. Source: [Mesh-LLM/mesh-llm](https://github.com/Mesh-LLM/mesh-llm)
- Routing: each request goes to a node that can serve the model, and a model that fits on one node runs locally. Large dense models are split into contiguous layer ranges ("stages"), and peers fetch only the GGUF fragments their stage needs. Source: same repo.
- Discovery uses Nostr for public meshes and invite tokens for private ones. Transport is QUIC with Iroh relays, so it works across NAT and the internet. Source: same repo.
- Built on a patched llama.cpp runtime ("Skippy"). It does not use llama.cpp RPC as far as the page says. MoE models are supported as package-backed stages, with a Flash-MoE SSD expert-streaming plugin. Source: same repo. Whether expert-level distribution exists is not stated.
- Builds exist for macOS Metal, Linux (CPU, CUDA, ROCm, Vulkan, ARM64) and Windows (CPU, CUDA, ROCm, Vulkan). The page has a web console on :3131 and a headless mode. Whether different vendors can mix in one mesh is not stated. Source: same repo.
- It had about 4.0k stars, 132 open issues and 3,117 commits when fetched. No creation date was visible. Source: same repo.
- It also has an experimental "Mixture-of-Agents" mode (`model: "mesh"`).

#### vLLM / Ray / SGLang
- vLLM multi-node: set tensor_parallel_size to GPUs per node and pipeline_parallel_size to the number of nodes. Ray is the default multi-node runtime. Each node needs an identical environment. Tensor parallelism wants fast interconnect such as InfiniBand. Source: [vLLM parallelism docs](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)
- Datacenter GPU oriented, with no phones, iGPUs or consumer mixing story found. Mixed GPU types are not confirmed either way. SGLang was not researched, a gap.

#### AI Horde / KoboldAI Horde
- Volunteer workers poll an API, generate on their own GPUs and earn kudos, which set queue priority. The stack is FOSS and self-hostable. Source: [aihorde.net workers](https://aihorde.net/contribute/workers)
- An undated homepage snapshot shows text at about 497 t/s aggregate, 34 active text workers out of 85 threads, and 135.3M text requests. Source: [stablehorde.net](https://stablehorde.net:443). Each request runs on a single worker, so there is no model sharding. Prompts are visible to workers, so there is no privacy.

#### Jan, GPT4All, Open WebUI, llamafile
- These are secondary-source mentions only. Jan has a Docker headless server and remote providers, and GPT4All has remote providers and a local API server. Open WebUI is a front-end for multiple backends. Llamafile is maintained by Mozilla.ai. Source: [toolhalla](https://toolhalla.ai/blog/lm-studio-vs-jan-vs-gpt4all-2026), [pinggy](https://pinggy.io/amp/blog/top_5_local_llm_tools_and_models/) (SEO blogs, unverified). None of these do multi-node model splitting as far as found.

### Inferences
- The "clients plus remote OpenAI-compatible endpoint" pattern (Jan, Open WebUI, GPT4All, Ollama `OLLAMA_HOST`) means any pancake gateway exposing `/v1` is immediately usable by all of them.
- llama.cpp RPC is the de facto shared substrate (KoboldCpp, LocalAI worker, many wrappers). Mesh-LLM took the other route of a patched llama.cpp with its own stage transport.
- Sharding across a LAN is bandwidth- and latency-bound, so the practical gain is fitting bigger models, not speed. This matches the RPC docs and HPC write-ups.

### Gaps
- Measured t/s numbers for llama.cpp RPC on commodity LAN or Wi-Fi: none found in primary sources.
- Ollama GitHub issues and feature requests, Jan, GPT4All, text-generation-webui, TGI, llamafile, Foundry Local, Apple PCC, SGLang, Ray details: not found or not searched.
- llama.cpp WebGPU backend status and any 2026 RPC PRs other than RDMA: not verified.

## 2. Is there already something equivalent to "NetworkPancake"? What is closest, and what is missing?

### Takeaway
No single tool covers old PCs plus phones, internet mesh, several models and a config UI. Mesh-LLM is the closest on concept: a mesh with discovery over NAT, layer-split stages, a web console and a cross-platform build matrix including Vulkan. exo is the closest on polish but is Apple-only in practice. LocalAI P2P is the closest on mixed workloads (federated plus worker) but is experimental.

### Cited Findings
- Mesh-LLM: see the Mesh-LLM section above. [Mesh-LLM/mesh-llm](https://github.com/Mesh-LLM/mesh-llm)
- exo: automatic discovery, a dashboard, topology-aware parallelism, but Linux is CPU-only and nodes of different platforms do not join. [exo README](https://github.com/exo-explore/exo)
- LocalAI P2P: token-based discovery and both federated and worker modes. [LocalAI docs](https://localai.io/features/distribute/)
- LM Link: cross-device remote use with a UI, but no splitting and a dependency on LM Studio's hub. [FAQ](https://lmstudio.ai/docs/lmlink/basics/faq)
- Neither Mesh-LLM, exo nor LocalAI documents Android/iOS phones as nodes. No source found showing phones as inference nodes in any of them.

### Inferences
Likely missing across all of them (my assessment, not verified in sources):
- phone (Android/iOS) nodes
- old-iGPU-aware scheduling and a capacity-aware config UI
- serving several models at once with per-node assignment
- a bandwidth- or latency-aware placement for Wi-Fi/internet links
- MoE expert-level distribution over the network
- an auth and trust model suitable for untrusted peers

### Gaps
- Mesh-LLM's maturity (creation date, release cadence, real benchmarks, Android support) was not verified. It could have been missed by other researchers.
- I did not search for other mesh projects beyond Mesh-LLM and exo (for example Petals, prima.cpp, distributed-llama, GPUStack). Check whether those are covered by another researcher.

## 3. Reusable components

### Takeaway
Reuse llama.cpp RPC (or Mesh-LLM's staged GGUF-fragment fetch) as the compute transport, and a libp2p/QUIC layer for discovery and NAT traversal. Expose an OpenAI-compatible `/v1` front.

### Cited Findings
- llama.cpp rpc-server: `--rpc` list, `--tensor-split`, server tensor cache (`-c`), optional RDMA, but insecure and requires same-commit builds. [README](https://github.com/ggml-org/llama.cpp/blob/master/tools/rpc/README.md)
- LocalAI discovery: go-libp2p plus edgevpn, token, mDNS-only option. [docs](https://localai.io/features/distribute/)
- Mesh-LLM: Nostr for public meshes and invite tokens for private ones, QUIC with Iroh relays, GGUF fragments fetched per stage. [repo](https://github.com/Mesh-LLM/mesh-llm)
- exo: `exo-bench` and an auto-placement approach that uses link latency and bandwidth. [README](https://github.com/exo-explore/exo)
- LM Link: model of a hub used only for discovery with end-to-end encrypted data. [FAQ](https://lmstudio.ai/docs/lmlink/basics/faq)

### Inferences
- The security gap in llama.cpp RPC means it should only run inside an encrypted overlay (WireGuard, QUIC, libp2p streams), never on an open network.

### Gaps
- License and code quality of reusable pieces other than Mesh-LLM (Apache-2.0) were not checked.

## 4. Recent 2026 news

### Takeaway
2026 headline items: LM Link (Feb), llama.cpp RPC over RDMA plus macOS 26.2 Thunderbolt 5 RDMA, KoboldCpp RPC (about May), Ollama on MLX (Mar) and cloud models, llama.cpp MoE expert cache (Oct), and Mesh-LLM's rise.

### Cited Findings
- LM Link launched in preview around Feb 25, 2026. [Tailscale blog](https://tailscale.com/blog/lm-link-remote-llm-access)
- macOS 26.2 added RDMA over Thunderbolt 5, and WWDC26 covered MLX distributed with JACCL. [WWDC26 233](https://developer.apple.com/videos/play/wwdc2026/233/), [AppleInsider, Dec 2025](https://appleinsider.com/articles/25/12/20/ai-calculations-on-mac-cluster-gets-a-big-boost-from-new-rdma-support-on-thunderbolt-5)
- KoboldCpp 1.114 added RPC. [release](https://newreleases.io/project/github/LostRuins/koboldcpp/release/v1.114)
- Ollama MLX preview (Mar 30) and an MLX performance post (Jun 11). [feed](https://rss-parrot.net/web/feeds/ollama.com.blog)
- llama.cpp PR #29887 MoE expert cache, reported merged Oct 7. [developersdigest](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026)

### Gaps
- Hacker News and r/LocalLLaMA threads were not retrieved. Exact merge dates for the llama.cpp RDMA RPC change were not found.
