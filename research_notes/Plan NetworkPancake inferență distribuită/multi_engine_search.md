# Multi-engine search: projects pooling devices (PCs, iGPU, phones) for distributed LLM inference

Date of research: 2026-11-10. Star counts and push dates come from the GitHub API (api.github.com/repos/...) fetched on that date, so they are verified. Claims from blogs and Reddit mirrors are marked (secondary) or (unverified).

## Engine log (what worked / what blocked)

### Takeaway
GitHub REST API, WebSearch and plain page fetches of GitHub READMEs worked. DuckDuckGo (bot-check challenge), Brave (HTTP 429), and Hacker News Algolia (query returned 0 hits, so not a clear block) did not give usable results. Bing returned irrelevant junk. Google, Hugging Face (only an irrelevant model record) and arXiv (no results returned) were not useful. Reddit was reached only through WebSearch mirrors and secondary summaries, not directly.

### Cited Findings
- DuckDuckGo html endpoint returned a bot-check (duck-selection challenge) page, no results — attempted https://html.duckduckgo.com/html/
- Brave search returned HTTP 429 Too Many Requests — attempted https://search.brave.com/search
- Bing returned only unrelated results (games, movies) for the query — attempted https://www.bing.com/search
- GitHub search API worked but is keyword-sensitive: long multi-word queries often returned `total_count: 0`, short ones (for example `mesh-llm`) returned hits — https://api.github.com/search/repositories?q=mesh-llm
- Hacker News Algolia query returned zero hits for a long query; Show HN items were found instead via WebSearch mirrors — https://hn.jakechampion.name/item/47042729
- Google, Hugging Face listing and arXiv API were attempted but yielded nothing useful (Google not tried successfully; HF returned one unrelated model; arXiv returned no parsed results). Treat these as "no usable result", not proof that nothing exists.

### Inferences
- A future pass should use `gh api` or `curl` with shorter keyword queries, and search Reddit and HN through a real browser (the Chrome tools are available).

### Gaps
- No direct Reddit threads were read. Reddit evidence below is via mirrors (reddit.sentinel-team.org, redlib) and blog summaries.

## Which projects exist, grouped (a) LAN clusters (b) internet/volunteer meshes (c) browser/WebGPU (d) phone-as-worker (e) commercial/crypto

### Takeaway
Most of the genuinely active, popular work is in LAN/mesh pipeline-parallel systems (mesh-llm, exo, Parallax). New to our notes and worth looking at: SwarmLLM/"Pooled" (browser, 640 stars, active this month), dnet, IntelNav, mechnet (phones, Phase 0 only), Kalavai, GPUStack, AI Horde, Synapse and Shard (browser prototypes). No mature iOS/Android worker project was found.

### Cited Findings
Format: name — stars, last push, license, platform, note. All numbers fetched 2026-11-10 from the GitHub API.

(a) LAN clusters / cluster managers
- exo-explore/exo — 48,276 stars; pushed 2026-11-06; Apache-2.0 (already in notes) — [repo](https://github.com/exo-explore/exo)
- dnet (firstbatchxyz/dnet) — 76 stars; pushed 2026-01-06; Apache-2.0; Python; "Distributed LLM Inference for Apple Silicon Clusters"; a search summary says it has automatic node discovery and Thunderbolt detection (secondary) — [repo](https://github.com/firstbatchxyz/dnet). Apple-only and no commits in 10 months, so low priority.
- GPUStack — 6,009 stars; pushed 2026-11-10; Apache-2.0; GPU cluster manager for serving with vLLM/SGLang plus SSH GPU instances — [repo](https://github.com/gpustack/gpustack). It is a cluster manager, not a layer-splitter across weak devices (from description only; I did not read its README).
- Kalavai (kalavai-net/kalavai-client) — 225 stars; pushed 2026-11-09; Apache-2.0; pools spare GPU/CPU across machines via a seed node plus workers (templates for vLLM, llama.cpp, Ray). Needs Docker and root; Linux and Windows workers; self-hosted workers should be on the same network as the seed; README says "early stage", breaking changes before v1.0 — [repo](https://github.com/kalavai-net/kalavai-client)
- LocalAI — 50,285 stars; pushed 2026-11-10; MIT (already in notes) — [repo](https://github.com/mudler/LocalAI)
- KTransformers (kvcache-ai) — 19,659 stars; pushed 2026-11-10; Apache-2.0; heterogeneous CPU/GPU inference/fine-tune optimizations. Single-host, not a network cluster — [repo](https://github.com/kvcache-ai/ktransformers)
- InferHub (Dev-Art-Solutions/InferHub) — 1 star; pushed 2026-11-10; MIT; .NET self-hosted inference mesh with Ollama-compatible API and GPU worker nodes. Tiny and unvetted — [repo](https://github.com/Dev-Art-Solutions/InferHub)
- llama.cpp RPC benchmarks repo (kjaiswal/llama-cpp-distributed-benchmarks) exists (appeared in search; not opened) — [repo](https://github.com/kjaiswal/llama-cpp-distributed-benchmarks)

(b) Internet / volunteer / private meshes
- Mesh-LLM/mesh-llm — 4,012 stars; pushed 2026-11-10; Apache-2.0 (already in notes). README: Rust workspace; "Skippy" does split execution by contiguous layer ranges with per-stage GGUF fragments; "Mesh" does QUIC via Iroh, Nostr discovery for public meshes, invite tokens for private; builds for CPU/CUDA/ROCm/Vulkan on Linux and Windows, Metal on macOS; no documented iOS/Android support; self-described experimental — [repo](https://github.com/Mesh-LLM/mesh-llm)
- Mesh LLM announcement summary: ~16 tok/s Qwen 235B A22B on 2 M3 Ultra nodes over 1 GbE; HN thread 300+ points; main complaints were latency (suits LAN/metro, not global WAN), no end-to-end encryption, unclear incentive model; commenters compared it with exo, AI Horde and cocompute.ai — [DevelopersDigest summary](https://www.developersdigest.tech/blog/mesh-llm-distributed-inference-iroh) (secondary)
- Parallax (GradientHQ/parallax) — 1,428 stars; pushed 2026-07-01; Apache-2.0 (already in notes). Pipeline-parallel across heterogeneous nodes; MLX on Mac, SGLang/vLLM on GPU; Lattica P2P — [repo](https://github.com/GradientHQ/parallax)
- IntelNav (IntelNav/intelnav) — 0 stars; pushed 2026-04-29; Apache-2.0; Rust. Layer ranges on volunteer machines, Kademlia DHT, Noise XX encrypted CBOR activations, Ed25519 identities, Linux only. README itself admits a four-hop chain is slower than a hosted vendor model and downstream peers see activations — [repo](https://github.com/IntelNav/intelnav)
- Hivemind — 2,551 stars; pushed 2026-01-11; MIT (already covered via Petals lineage) — [repo](https://github.com/learning-at-home/hivemind)
- AI Horde (Haidra-Org/AI-Horde) — 1,673 stars; pushed 2026-11-10; AGPL-3.0; crowdsourced cluster for AI art and text generation. It routes whole requests to volunteer workers; it does not split one model across devices — [repo](https://github.com/Haidra-Org/AI-Horde)
- Petals: no source confirmed 2026 maintenance; I found no reliable status information. Volunteer churn and privacy warnings are documented by the project — [Petals paper](https://arxiv.org/pdf/2312.08361)
- Parallax paper on volunteer GPUs over low-bandwidth links (arXiv 2509.26182) — [arXiv](https://arxiv.org/abs/2509.26182)

(c) Browser / WebGPU
- SwarmLLM / "Pooled" (Nehanth/swarmllm) — 640 stars; pushed 2026-11-03; MIT; JavaScript. Layer split across browser tabs, own WGSL kernels (not WebLLM/MLC/llama.cpp), WebRTC activations (about 4-10 KB per token), host keeps embeddings/LM head/sampler, TURN fallback on 443, speculative decoding, `npx @pooled/cli serve` OpenAI/Anthropic-compatible bridge; Chrome macOS is the tested host, iPhone Safari can join with a few layers; models Qwen 27B dense and 35B MoE at Q4_0, plus an experimental 125B MoE that streams experts from SSD — [repo](https://github.com/Nehanth/swarmllm)
- Wavect review of SwarmLLM (author has a commercial interest, treat with care): GB10 ~9 tok/s plain and ~16 with speculative decoding; MacBook+iPhone 7.7 tok/s (demo caption claims 10.7); 16-device topology fell to ~3.8-4.7 tok/s; cross-internet rooms ~3.5-6 tok/s; no verification of remote compute; activations are not private — [review](https://wavect.io/blog/swarmllm-browser-p2p-inference-review-2026/)
- Synapse (tejasphatak/Synapse) — 7 stars; pushed 2026-04-21; no license; "Distributed LLM inference across browsers via WebGPU" — [repo](https://github.com/tejasphatak/Synapse)
- Shard (TrentPierce/Shard) — browsers as WebGPU "scout" nodes, libp2p mesh, "verifier" nodes; author describes it as early stage, not production hardened; no independent benchmarks found (repo page not opened by me; URL from HN mirror) — [HN mirror](https://hn.jakechampion.name/item/47042729)
- AaravMehta-07/distributed-compute — 2 stars; pushed 2026-05-21; browser GPU pooling via WebRTC/WebGPU prototype — GitHub search result (search API), repo not opened
- TU Wien 2026 thesis: web-based distributed LLM with ONNX Runtime Web, WebGPU, protobuf over WebSockets — [thesis](https://repositum.tuwien.at/handle/20.500.12708/227204)

(d) Phone-as-worker
- mechnet (81hast/mechnet) — 0 stars, 6 commits; pushed 2026-06-03; MIT; Python. Laptop plus Android phones in Termux, mDNS discovery, UDP gossip, battery/temperature reporting via termux-api, planned llama.cpp RPC orchestration. README says only Phase 0 (discovery and status) works and inference is not implemented; no benchmarks — [repo](https://github.com/81hast/mechnet)
- Phone as llama.cpp RPC worker (Termux) guide: adds memory, not speed; slower if model already fits desktop VRAM; thermal throttling within 10-15 minutes; Vulkan builds can fail on older Mali GPUs; needs ~4-6 GB free RAM; author concludes "buy RAM". No measured tok/s given — [Kunal Ganglani](https://www.kunalganglani.com/blog/phone-as-gpu-local-llm.md) (secondary blog)
- exo advertised running on old phones in 2024 press — [TechRadar](https://www.techradar.com/computing/bittorrent-for-llm-exo-software-is-a-distributed-llm-solution-that-can-run-even-on-old-smartphones-and-computers) (secondary; I did not verify current iOS/Android status)
- Mesh-LLM: iroh Swift SDK mobile app described as "forthcoming" (secondary); README does not document mobile — see (b)
- I found no maintained, benchmarked iOS or Android worker project.

(e) Commercial / crypto
- io.net (GPU aggregation marketplace), Nosana (Solana-linked inference GPU marketplace), Bittensor (model-output incentive market, not GPU rental), Gensyn (verifiable ML training). A claim that Gensyn RL Swarm paused in Sep 2026 is uncorroborated — [comparison summary](https://metavert.io/depin-vs-decentralized-ai), [BlockEden](https://blockeden.xyz/blog/2026/02/07/decentralized-gpu-networks-2026/) (secondary SEO-style sources)
- Prime Intellect — org repo `PrimeIntellect-ai/prime` is the CLI/SDK, 348 stars, pushed 2026-11-10, MIT — [repo](https://github.com/PrimeIntellect-ai/prime). A $130M Series A claim comes from a single search snippet (unverified).

### Inferences
- Relative to our existing notes, the new items are: SwarmLLM/Pooled, IntelNav, mechnet, dnet, Kalavai, GPUStack, AI Horde, Synapse, Shard, InferHub. (Existing notes already cover Parallax, Mesh-LLM, Hivemind, LocalAI and "Shard" in the MoE sense; I only skimmed them by keyword.)
- Browser-first (SwarmLLM) is the only approach found that already runs on an iPhone without install, but reported speed is similar to or lower than a single decent machine.

### Gaps
- Stars/activity for Shard, AaravMehta distributed-compute and cocompute.ai were not verified from the repo or site.
- Existing notes were only keyword-grepped, not fully read; some "new" items may be partially covered.
- No search surfaced projects with a Romanian or Spanish query (not run); one Chinese query run, results were generic llama.cpp RPC tutorials.

## Lessons: failure modes, complaints, wishes

### Takeaway
Across sources, network latency per token, not bandwidth, is the dominant problem; adding devices rarely adds speed, only capacity; and trust/privacy (activations visible) plus churn (tabs, phones sleeping) are unsolved for public meshes.

### Cited Findings
- RPC over gigabit is "very slow"; a commenter said latency is the biggest issue; llama.cpp multi-GPU may be slow in general, not just RPC; the remote model portion reloads over the network on every server start (secondary summary of r/LocalLLaMA threads) — [WebSearch results via redlib mirror](https://redlib.ssps.io/r/LocalLLaMA/comments/1l8vziy/what_is_the_current_state_of_llamacpp_rpcserver)
- 10GbE benchmark: 4.2x speedup on 7B, slight on 72B; decode slower because each token needs a network round trip (~0.17 ms on 10GbE) (secondary) — [kjaiswal benchmarks](https://github.com/kjaiswal/llama-cpp-distributed-benchmarks) (repo listed in search, numbers come from the search summary; unverified)
- If the model fits on one GPU, adding a phone makes it slower; thermal throttling in 10-15 min — [Kunal Ganglani](https://www.kunalganglani.com/blog/phone-as-gpu-local-llm.md)
- More devices did not mean more speed in SwarmLLM (16-device topology ~3.8-4.7 tok/s); peer loss, sleeping phones, NAT, relays and thermal throttling interrupt generation; no verification of remote compute — [Wavect](https://wavect.io/blog/swarmllm-browser-p2p-inference-review-2026/)
- SwarmLLM README: prefill 861 tok/s vs llama.cpp ~2,400; decode ~1.3x slower than native llama.cpp; prompts visible to everyone in a room; Safari Mac may reload under memory pressure — [repo](https://github.com/Nehanth/swarmllm)
- IntelNav README admits a four-hop chain is slower than a hosted vendor model; no read-only mode (users must host or relay) — [repo](https://github.com/IntelNav/intelnav)
- Mesh LLM HN: top topics were speed vs RAM/NVMe offload, security (use private meshes), incentives unclear; "suits LAN and metro, not global WAN" — [summary](https://www.developersdigest.tech/blog/mesh-llm-distributed-inference-iroh) (secondary)
- exo Reddit sentiment: works for demos; users still use Ollama daily; llama.cpp RPC called simpler and more reliable (secondary blog summarizing Reddit) — [Lobster Pack](https://www.lobsterpack.com/blog/exo-distributed-ai-cluster-macs/)
- Chinese tutorials: llama.cpp RPC needs fast network and explicit thread config for workers, else low utilisation (secondary) — [CSDN](https://gitcode.csdn.net/6a2a841e4168cd04b445a3aa.html)
- Hacker Noon/other essays argue memory is the practical limit: 16 GB laptops fit 8B-class, phones 1B-4B models (secondary) — search snippet, not opened

### Inferences
- "Wish list" implied by complaints (my inference, not quoted): zero-config discovery, verified/trusted peers, battery- and thermal-aware scheduling, honest "will this help or hurt?" estimation before adding a node, and a mode that adds capacity only when the model does not fit locally.
- Speculative decoding and MoE expert placement are the reported mitigations for per-token latency (SwarmLLM uses draft layers; Mesh LLM theoretical ceiling quoted without speculation).

### Gaps
- No primary Reddit URLs were read; wishes are inferred, not quoted.
- No quantified phone-worker tok/s from a real run was found.

## Ranked shortlist: 10 projects to study or fork for NetworkPancake

### Takeaway
Study Mesh-LLM and SwarmLLM first (activity, architecture, honest limitations), then the phone/LAN bits from llama.cpp RPC and mechnet.

### Cited Findings
1. Mesh-LLM — most active (4,012 stars, push today), Apache-2.0, Rust, QUIC/Iroh NAT traversal, per-stage GGUF fragments, Vulkan/CUDA/ROCm/Metal; direct template for layer splitting plus discovery — [repo](https://github.com/Mesh-LLM/mesh-llm)
2. SwarmLLM/Pooled — only working install-free phone/browser participant; WebRTC activation protocol, TURN fallback, speculative decoding, OpenAI-compatible bridge — [repo](https://github.com/Nehanth/swarmllm)
3. exo — largest community (48k stars), auto-discovery UX reference; main caveat is Apple focus (secondary) — [repo](https://github.com/exo-explore/exo)
4. Parallax — Apache-2.0, heterogeneous pipeline scheduling, vLLM/SGLang/MLX backends, scheduling paper (arXiv 2509.26182) — [repo](https://github.com/GradientHQ/parallax)
5. IntelNav — clean Rust design for signed slices, Noise encryption, DHT; shows what a trust model looks like (small, Linux-only) — [repo](https://github.com/IntelNav/intelnav)
6. mechnet — only open design for phones joining via Termux with mDNS, gossip and battery/thermal telemetry; Phase 0, so borrow ideas, do not fork for inference — [repo](https://github.com/81hast/mechnet)
7. llama.cpp RPC (already in notes) — the pragmatic backend that mechnet and many tutorials use; documents the latency limits
8. Kalavai — pool/seed/worker control plane and templates for engines; reference for orchestration and a GUI (needs Docker, so a poor fit for phones) — [repo](https://github.com/kalavai-net/kalavai-client)
9. AI Horde — proven volunteer-worker, kudos/incentive and queue model at scale (1,673 stars, AGPL-3.0 caution) — [repo](https://github.com/Haidra-Org/AI-Horde)
10. KTransformers — CPU/GPU hybrid and MoE offload tricks, relevant to old PCs with iGPU/CPU and MoE experts per node (single-host) — [repo](https://github.com/kvcache-ai/ktransformers)

Honorable mentions: GPUStack (cluster manager), dnet (Apple-only), Synapse and Shard (browser prototypes, very early).

### Inferences
- Ranking is my judgment from README claims and activity, not from running any of them.
- AGPL (AI Horde), unlicensed (Synapse) and tiny repos should be treated as reference reading, not fork bases.

### Gaps
- I did not run or benchmark any project.
- GPUStack, KTransformers and dnet were assessed from API descriptions and secondary summaries, not full READMEs.
