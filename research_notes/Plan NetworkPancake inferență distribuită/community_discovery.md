# Community discovery: projects for pooling PCs and phones to run LLMs (as of 2026-11-10)

Method note: about 20 web searches/fetches. Star counts and "last commit" come from GitHub page snapshots fetched via WebFetch. Most snapshots showed NO commit dates, so "active/abandoned" is only partly verified. Direct Reddit/HN/Play/App Store pages were mostly unreachable (403/DNS); Reddit content came via search snippets and mirrors. Reddit-specific numbers are secondhand and marked as such.

## Q1. Which projects let phones (Android/iOS) participate as workers in a shared model?

### Takeaway
Only a few real options exist. Cake (Rust, iOS and Android listed), prima.cpp (Android via Termux), and DIY llama.cpp `rpc-server` in Termux. exo's current repo does not list iOS. Phone-as-worker gives capacity, not speed, over Wi-Fi.

### Cited Findings
- **Cake (evilsocket/cake)**. Rust (Candle) framework that shards transformer layers across mixed devices. About 3.1k stars, 1,117 commits, no releases listed. The README lists CUDA, Metal, Vulkan and CPU backends on Linux, macOS, Windows, iOS and Android, with Vulkan noted for "Linux (AMD/Intel/Steam Deck)". Workers run `cake run --cluster-key ... --name ...` with no model locally. The master finds them by mDNS, assigns layers by VRAM/compute, and streams zstd-compressed weights to them. It says "experimental code ... changed very quickly". License is FAIR 1.0.0: non-commercial free, commercial needs attribution, business use needs an agreement. No benchmark numbers in the README. — [GitHub](https://github.com/evilsocket/cake); older platform table (Android arm/arm64/x86_64, iOS aarch64 with Metal on A13+; Android CUDA partial) — [docs.rs cake-cli](https://docs.rs/cake-cli); original 2024 post — [r/LocalLLaMA via mirror](https://reddit.birdcat.cafe/r/LocalLLaMA/comments/1e601pj/cake_a_rust_distributed_llm_inference_for_mobile)
- **prima.cpp** (llama.cpp fork, Lizonghang/prima.cpp; the page fetched was the fork tdh-archive/prima.cpp, so the star count shown (0) is meaningless). Supports Linux, macOS, Android and HarmonyOS (Termux). It does NOT list iOS, Windows ("roadmap"), Vulkan or AMD. Only CUDA GPUs are supported. Claims (self-reported): about 11 tok/s on QwQ-32B and 1.5 tok/s on Llama-3-70B on the author's home cluster; per-token latency for 70B 10120 ms (llama.cpp) vs 674 ms (prima.cpp). Test devices had 1.9-9.7 GiB free RAM. Needs gcc, cmake, fio, zmq and HiGHS. Workers form a ring. — [GitHub fork](https://github.com/tdh-archive/prima.cpp)
- **DIY llama.cpp RPC on phone**. Build llama.cpp in Termux with `-DGGML_RPC=ON`, run `rpc-server`, and point the desktop at `--rpc <phone-ip>:50052`. Ollama does not expose RPC, so raw llama.cpp is needed. Author's verdict: phone only adds capacity, no measured gain, and it slows models that already fit in VRAM. Over Wi-Fi the result is neutral to slightly worse. Thermal throttling hits in about 10-15 min, and Vulkan builds can fail on older Mali GPUs. — [kunalganglani.com](https://www.kunalganglani.com/blog/phone-as-gpu-local-llm) (single blog, unverified by me)
- **exo (exo-explore/exo)**. About 48.3k stars, 2,526 commits. The current README lists macOS (Apple silicon, GPU) and Linux (CPU, GPU "under development"). iOS and Vulkan are NOT mentioned. Devices must run the same version, and Macs and Linux machines "don't join each other". RDMA over Thunderbolt 5 is supported on macOS 26.2+. — [GitHub](https://github.com/exo-explore/exo). Contradicted/qualified: a Dec 2025 article claims phones and watches can join — [byteiota](https://byteiota.com/exo-run-ai-clusters-on-phones-laptops-smartwatches/) (secondary source; DeepSeek V3.1 32.5 tok/s on 4 nodes is its claim, unverified). Another explainer says the exo repo was flagged as archived — [inventivehq](https://inventivehq.com/blog/clustering-machines-for-local-ai), but the repo page I fetched showed no archive notice. Conflict unresolved.
- **distributed-llama (b4rtaz)**. Ethernet tensor-parallel; node count is capped by KV heads (power of 2). Only q40 model + q80 buffer is supported. Reported 3.01 tok/s on one Pi 4B 8 GB, and about 14 tok/s for Llama 3.1 8B on 8x Pi 5 (blog claim). No Android/Termux instructions found. — [README](https://cdn.jsdelivr.net/gh/b4rtaz/distributed-llama@main/README.md), [brightcoding blog](https://blog.brightcoding.dev/2025/08/29/turn-your-home-devices-into-an-ai-supercomputer-a-hands-on-guide-to-distributed-llama)
- **Nataris (nataris.ai)**. Android phones as paid inference providers behind an OpenAI-compatible API. Phones serve whole small models (Qwen2.5 0.5B about 5 s latency, Llama 3.2 1B 15-20 s); it is NOT a layer-split cluster. Self-reported: 21 devices, 2,775 jobs. Useful lessons: llama.cpp preallocates the full context cache and crashes phones, battery optimizers silently kill background connections, native SDK must init on the main thread. — [dev.to post](https://dev.to/vishal_sharma_nataris/we-built-a-p2p-ai-inference-network-that-runs-on-android-phones-heres-what-we-learned-3lhb); Play listing [ai.nataris.app](https://play.google.com/store/apps/details?id=ai.nataris.app) (not fetched)
- **Ollama Local AI (Android, com.llmproxy)**. Third-party, unaffiliated with Ollama. The listing advertises pairing multiple Android devices on a LAN to pool memory/compute, and acting as an OpenAI-compatible LAN gateway. Updated Oct 11, 2026 per a search snippet. Mechanism (RPC or not) unconfirmed. — [Play Store](https://play.google.com/store/apps/details?id=com.llmproxy) (page content not retrievable; claims come from search snippets)

### Inferences
- For a pooled cluster, Cake is the only project in this set that credibly targets iOS and Android workers with automatic layer assignment, but its maturity is unproven (no numbers, experimental, restrictive license).
- Phone workers are useful mainly for fitting larger models. Expect no speedup over Wi-Fi.

### Gaps
- No verified user-reported tok/s for Cake on phones. No verified exo iOS client in 2026. No Android/Termux report for distributed-llama.
- Last-commit dates were not visible for any repo; check `git log` or the GitHub API directly.

## Q2. Which work with old PCs / iGPU / Vulkan, and how well?

### Takeaway
llama.cpp RPC (mainline) and Mesh LLM have Vulkan paths; Cake lists Vulkan. exo and prima.cpp do not cover Vulkan/old GPUs. Hard numbers for old PCs over gigabit are scarce.

### Cited Findings
- **Mesh LLM (Mesh-LLM/mesh-llm)**. About 4.0k stars, 3,117 commits. Builds for Linux (CPU, ARM64, CUDA, ROCm, Vulkan), Windows (CPU, CUDA, ROCm, Vulkan) and macOS (Metal). Single OpenAI-compatible endpoint on port 9337. A model that fits on one node runs locally, otherwise it routes to a peer or splits by layer stages (Skippy, built on llama.cpp). QUIC encrypted, public meshes discovered via Nostr, private meshes via invite tokens. Self-described "experimental". No Android/iOS builds mentioned (a Swift package/XCFramework is referenced). — [GitHub](https://github.com/Mesh-LLM/mesh-llm); background [wavect.io](https://wavect.io/blog/mesh-llm-distributed-inference-multiple-computers.md), [developersdigest](https://www.developersdigest.tech/blog/mesh-llm-distributed-inference-iroh)
- **llama.cpp RPC** reported numbers (via a search summary; I could not open the threads): 70B over Wi-Fi about 2.2 tok/s; Qwen3.5-27B Q5 at 13 tok/s over LAN split between RTX 4070 Ti and RX 6800; two RTX 5090 boxes over 2.5GbE gave 96 tok/s on a 122B MoE; a user with four old 12 GB-VRAM laptops said rpc-server was the only thing that worked. Commenters report a notable RPC penalty, also with Vulkan. — [LAN Qwen3.5-27B post](https://localguy-theta.vercel.app/posts/multi-gpu-inference-across-lan-qwen27b-nvidia-amd/), [2025 thread mirror](https://redlib.hbubli.cc/r/LocalLLaMA/comments/1inr5pf/feasibility_of_distributed_cpuonly_llm_inference/mcd6j15), [RPC state thread](https://redlib.ssps.io/r/LocalLLaMA/comments/1l8vziy/what_is_the_current_state_of_llamacpp_rpcserver) (mirror host down when I tried)
- **Parallax (GradientHQ/parallax)**. About 1.4k stars, Apache-2.0, 459 commits, newest dated news 2026/02. Mac uses MLX; Linux GPU uses SGLang/vLLM. Pipeline-parallel across heterogeneous, geographically separate nodes. No CPU-only/Vulkan/Windows confirmation. Paper: scheduler allocates layers by compute and bandwidth. — [GitHub](https://github.com/gradientHQ/parallax), [arXiv 2509.26182](https://arxiv.org/html/2509.26182v1)
- **Kalavai**. Open source pool client (Ray/GPUStack based). README lists Mac M-series, AMD and Intel GPUs as NOT supported (roadmap). Public pool beta launched Jan 2025. Self-reported claims only. — [docs](https://kalavai-net.github.io/kalavai-client/public_llm_pool), [GitHub](https://github.com/kalavai-net/kalavai-client)
- **Hyperspace (hyperspaceai/hyperspace-node)**. libp2p P2P inference node; inference needs a GPU with 4+ GB VRAM; agent/token/points layers are mixed in. Claims (2M+ agents) are self-reported and unverified. — [sourcepulse listing](https://www.sourcepulse.org/projects/1773040), [starlog](https://starlog.pages.dev/articles/llm-engineering/hyperspaceai-agi)
- **Psyche (Nous)** is distributed training, not inference; relevant only as an iroh networking example. — [iroh.computer](https://www.iroh.computer/solutions/nous)

### Inferences
- For iGPU/old PCs, Mesh LLM (Vulkan, Windows) and plain llama.cpp RPC are the practical ones. Gigabit plus pipeline split likely yields single-digit tok/s for mid-size dense models (inference from the reports above, not measured).

### Gaps
- No found report of old PCs/iGPU (e.g. Intel UHD, Vega) on gigabit with named tok/s. The Reddit threads need manual reading.

## Q3. Abandoned vs active; user complaints

### Takeaway
Star counts are known; commit recency mostly isn't. Active-looking: exo, Mesh LLM, Cake, Parallax. Complaints center on latency per token, version lock-in, and experimental status.

### Cited Findings
- exo: 206 open issues, 252 open PRs; same-version requirement; Linux GPU still in development. — [GitHub](https://github.com/exo-explore/exo)
- Cake: 15 open issues; "changed very quickly"; restrictive FAIR license. — [GitHub](https://github.com/evilsocket/cake)
- Mesh LLM: 132 issues, 33 PRs; Blackwell unsupported on native Windows CUDA; HN-style discussion flagged network latency. — [GitHub](https://github.com/Mesh-LLM/mesh-llm), [wavect.io](https://wavect.io/blog/mesh-llm-distributed-inference-multiple-computers.md)
- Parallax: 20 open issues; dated news up to 2026/02. — [GitHub](https://github.com/gradientHQ/parallax)
- Security/fault-tolerance gaps in exo cited by an Oct 2025 analysis (via secondary source). — [byteiota](https://byteiota.com/exo-run-ai-clusters-on-phones-laptops-smartwatches/)

### Gaps
- Last-commit dates, release cadence and the real complaint threads were not retrievable. Treat "active" as unverified.

## Q4. Phone UI candidates (clients for an OpenAI-compatible cluster endpoint)

### Takeaway
Many store apps connect to a self-hosted OpenAI-compatible server; all listings below are developer marketing copy, none tested.

### Cited Findings
- **Chapper** (iOS/iPad/Mac; connects to LM Studio, Ollama, llama.cpp, any OpenAI-compatible endpoint; reportedly $9.99 Pro, no Android) — [App Store](https://apps.apple.com/app/chapper/id6760984679), [review](https://www.promptquorum.com/power-local-llm/chapper-review)
- **Caverno** (iOS, free; Ollama/LM Studio/vLLM/OpenAI-compatible on LAN) — [App Store](https://apps.apple.com/ae/app/caverno/id6760192392)
- **Airgap** (iOS; llama.cpp/Ollama/LM Studio/vLLM) — [mwm.ai](https://mwm.ai/apps/airgap-local-ai-chat/6760871034)
- **MyLLM** (iOS; auto server detection) — [App Store](https://apps.apple.com/us/app/id6760704297)
- **InferrLM** (open source, iOS+Android; custom base URLs plus on-device llama.cpp) — [listing](https://awesome.ecosyste.ms/projects/github.com%2Floomax-labs%2FInferrLM)
- **Atomic Chat** (iOS+Android on-device plus built-in OpenAI-compatible server, per a blog) — [atomic.chat](https://atomic.chat/blog/llm-updates/ollama-alternatives)
- **Ollama Local AI** (Android; see Q1) — [Play](https://play.google.com/store/apps/details?id=com.llmproxy)
- **OlliteRT** turns a phone into an OpenAI-compatible server using Google LiteRT-LM (not llama.cpp) — [review](https://www.promptquorum.com/power-local-llm/ollitert-review)
- The named apps from the brief (PocketPal, MLC Chat, Layla, ChatterUI, Private LLM, LLM Farm, Enclave, Apollo, Off Grid) were NOT researched this pass.

### Gaps
- No ratings, install counts or last-update data verified except the one snippet noted. Android coverage with a remote-server mode is thin in my results (InferrLM, Atomic Chat, Ollama Local AI).

## Q5. Reddit threads on old-PC clusters over LAN

### Takeaway
Only snippets were reachable. Pitfall consensus: per-token network round trips dominate; RPC works but with a penalty; WiFi is poor.

### Cited Findings
- Distributed local LLM experiences thread (Nov 2024) — [mirror](https://redlib.hbubli.cc/r/LocalLLaMA/comments/1gtaiot/distributed_local_llms_experiences/lxn3ja7) (403 on fetch)
- Original llama.cpp distributed announcement (May 2024) — [mirror](https://redlib.hbubli.cc/r/LocalLLaMA/comments/1cyzi9e/llamacpp_now_supports_distributed_inference)
- CPU tensor parallelism thread (Nov 2024) — [mirror](https://redlib.hbubli.cc/r/LocalLLaMA/comments/1gporol/llm_inference_with_tensor_parallelism_on_a_cpu)
- Numbers in Q2 (2.2 tok/s on 70B over Wi-Fi; 13 tok/s 27B on LAN) come from a search summary of these sources; unverified.

### Gaps
- Needs a manual read of those threads for hardware-specific numbers.

## Ranked shortlist: most relevant things we may not have known

1. **Mesh LLM** (github.com/Mesh-LLM/mesh-llm): Vulkan + Windows + Linux builds, OpenAI-compatible endpoint, auto-routing, about 4k stars. Closest to "old PCs with iGPU" plus a ready API. No phone workers.
2. **Cake** (github.com/evilsocket/cake): only project explicitly listing iOS and Android workers plus Vulkan, mDNS discovery, weight streaming. Experimental, restrictive license, no numbers.
3. **prima.cpp**: Android (Termux) workers, designed for low-RAM home devices, claims 70B on a home cluster. CUDA-only GPUs, no Vulkan or iOS.
4. **llama.cpp rpc-server in Termux**: simplest phone worker, mainline, but expect no speedup.
5. **Nataris and its lessons**: a field report on keeping Android workers alive (battery killers, context cache preallocation, thread init). Useful engineering checklist for any phone-worker design.
6. **Phone UI apps**: InferrLM, Atomic Chat (both platforms), Chapper/Caverno/Airgap (iOS), Ollama Local AI (Android).
7. **Parallax**: good scheduler paper (layer allocation by compute and bandwidth); GPU/Mac oriented, not a fit for old PCs.
8. Lower relevance: Kalavai (no AMD/Intel/Mac GPU), Hyperspace (needs 4 GB VRAM GPU, crypto/agent layers), Psyche (training only), exo (Mac-centric, same-version rule).
