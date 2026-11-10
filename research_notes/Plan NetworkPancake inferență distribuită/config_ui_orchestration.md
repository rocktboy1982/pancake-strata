# Configuration UI, device discovery and orchestration for NetworkPancake (as of 2026-11-10)

Research depth: about 15 tool calls. Several GitHub pages returned 404 or truncated text. Items marked [UNVERIFIED] come from my background knowledge or a single weak source.

## 1. How existing systems configure and visualise multiple nodes

### Takeaway
Only exo ships a real cluster dashboard. prima.cpp, distributed-llama and llama.cpp RPC are CLI-and-IP-list systems with no node UI. Petals has an external health monitor (node list, state, throughput). LM Studio LM Link and LocalAI add discovery or mesh networking. The simplest viable NetworkPancake UI is therefore a small coordinator-served page with node list, hardware probe, model placement and health.

### Cited Findings
- exo has a built-in dashboard at http://localhost:52415 for managing the cluster and chatting with models. The README does not mention "tinychat". Its example shows two models loaded on one cluster, each as its own instance. — [exo README](https://github.com/exo-explore/exo)
- exo discovery is automatic for devices on the same network running the same exo version. The README mentions a Zenoh namespace setting for cluster isolation, not mDNS or libp2p. Older exo versions used UDP/Tailscale discovery and tinychat [UNVERIFIED; not in the fetched README]. — [exo README](https://github.com/exo-explore/exo)
- exo API surface: OpenAI `/v1/chat/completions` and `/v1/responses`, Claude `/v1/messages`, Ollama-style `/ollama/api/chat` and `/ollama/api/tags`. Control endpoints: `/instance/previews`, `/instance`, `/instance/await`, `/models`, `/models/search`, `/models/add`, `/state`, `/v1/devices-on-other-versions`. Custom HF models can be pulled via the API. Download paths are set by environment variables, including read-only shared directories. — [exo README](https://github.com/exo-explore/exo)
- exo platform support: macOS on Apple silicon is the main target. Linux runs on CPU only (GPU in development). Windows, iOS and Android are not mentioned. — [exo README](https://github.com/exo-explore/exo)
- exo's topology is a rustworkx directed graph. Nodes are NodeIds. Edges are socket or RDMA connections. It provides cycle detection and a snapshot serialisation, and contains no placement logic itself. — [topology.py](https://raw.githubusercontent.com/exo-explore/exo/main/src/exo/shared/topology.py)
- prima.cpp has no UI. The README fetch failed (404 on main and master), so its CLI flags are not verified here. The paper's profiler and scheduler are covered in section 2. — [prima.cpp paper](https://arxiv.org/abs/2504.08791)
- distributed-llama: one root node plus 2^n − 1 workers. The root is given `--workers ip:port ...`, and workers bind with `--host` and `--port`. Node count is limited to powers of two and to the model's KV head count. `dllama chat` is the only interface described, and I could not confirm whether dllama-api is OpenAI-compatible from the README. It runs on Linux, macOS and Windows, with experimental Vulkan. — [distributed-llama](https://github.com/b4rtaz/distributed-llama)
- llama.cpp rpc-server: built with `-DGGML_RPC=ON`. The client lists servers with `--rpc host:port,host:port`, and `--tensor-split` overrides the default memory-proportional split. Flags `-c` (local tensor cache), `--device` and `-p` apply on the server. The README calls it "fragile and insecure" and says never to expose it on an open network. — [llama.cpp RPC README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/rpc/README.md)
- Upstream rpc-server has no discovery. LocalAI's p2p worker wraps it with DHT and mDNS via edgevpn behind a build flag. Termux phone workers are described in a 2026 blog post, not upstream docs. — [search result summary, LocalAI blame](https://code.morphllm.com/mudler/LocalAI/blame/branch/feat%2Fl4t-diffusers/go.mod); [phone guide](https://www.kunalganglani.com/blog/phone-as-gpu-local-llm)
- Petals health monitor (health.petals.dev) shows a per-server row with requests per second, a block bar, and a state symbol: online, joining (loading blocks), unreachable (NAT or firewall), offline. Data comes from `GET /api/v1/state` (format may change) and `GET /api/v1/is_reachable/<peer_id>`. Servers hosting 10 or more blocks can show a name set with `--public_name`. — [health.petals.dev repo](https://github.com/petals-infra/health.petals.dev); [Petals README](https://github.com/bigscience-workshop/petals)
- Petals is MIT-licensed and "community-run". The README snapshot gives no explicit maintenance status. — [Petals README](https://github.com/bigscience-workshop/petals)
- LM Studio LM Link (preview) lets you load and use models on other machines as if local, over Tailscale mesh. Existing clients keep pointing at localhost:1234. When a model exists on several devices, the REST API uses the preferred device. You cannot push a model from the local machine to the remote one. Limits cited: two users with five devices each (free tier). — [LM Studio LM Link docs](https://lmstudio.ai/docs/lmlink) via [search summary](https://www.dell.com/en-us/blog/lm-studio-s-lm-link-local-ai-everywhere/)

### Inferences
- UI features that matter, from what the working systems expose: node list with state (exo, Petals), per-node resource figures (exo `/state`), instance (model placement) create and delete (exo), model add and search (exo), reachability diagnostics (Petals). Per-node tok/s and layer-assignment visualisation are not documented in any fetched source, so they would be NetworkPancake additions.
- Open WebUI and Ollama federation are not covered here. As background, Open WebUI accepts several OpenAI-compatible connections and aggregates model lists [UNVERIFIED]. It does not shard one model across nodes.

### Gaps
- Open WebUI multi-connection docs (docs page fetch returned 404).
- prima.cpp README flags (404). From memory it uses `--world`, `--rank`, `--master`, `--next`, `--prefetch` [UNVERIFIED].
- LocalAI federated mode details; LM Studio parallel/multi-node behaviour beyond LM Link.
- Whether current exo has a "tinychat" page.

## 2. Hardware profiling and automatic partitioning

### Takeaway
exo uses a simple memory-proportional contiguous split with largest-remainder rounding. prima.cpp's Halda profiles FLOPS, memory throughput, disk and latency, then solves an integer program over window sizes and GPU layer counts. For NetworkPancake, start with the exo rule and add a measured-speed weight; use Halda-style ILP only if needed.

### Cited Findings
- exo pipeline placement: each node's fraction is `ram_available / sum(ram_available)` over the chosen cycle. Layers use floor plus largest-remainder rounding. Every node gets at least one layer (take one from the node with the most layers if a node rounds to zero). Shards are contiguous layer ranges. A node whose share of `resident_size` exceeds its RAM triggers an error. — [placement_utils.py](https://raw.githubusercontent.com/exo-explore/exo/main/src/exo/master/placement_utils.py)
- exo cycle selection: `filter_cycles_by_memory` keeps cycles where all nodes report memory and total available RAM meets the model's need. `get_smallest_cycles` keeps cycles with the fewest nodes. — [placement_utils.py](https://raw.githubusercontent.com/exo-explore/exo/main/src/exo/master/placement_utils.py)
- exo tensor parallelism gives every node all layers with only `device_rank` and `world_size` differing, and does no memory check. The README advertises topology-aware placement using latency and bandwidth. — [placement_utils.py](https://raw.githubusercontent.com/exo-explore/exo/main/src/exo/master/placement_utils.py); [exo README](https://github.com/exo-explore/exo)
- prima.cpp profiler measures per device: FLOPS per backend (CPU, CUDA, Metal) and quantisation (Q4_K, Q5_K, Q6_K, Q8_0, FP16, FP32); practical memory-read throughput (not theoretical); RAM-to-VRAM copy latency (skipped on unified memory); KV-cache copy latency; disk read throughput (sequential on Linux, random elsewhere); network latency to the adjacent device (time to send 4e bytes, e = embedding size). A model profiler counts FLOPs per layer by quantisation. — [prima.cpp paper, HTML v3](https://arxiv.org/html/2504.08791v3)
- Halda formulation: L layers, M devices, window size w_m, GPU layers n_m within the window, W = sum(w_m), k rounds per token with k·W = L. Objective is to minimise TPOT = L·(aᵀw + bᵀn + eᵀc)/(eᵀw) + κ. Constraints: integer, 0 ≤ n_m ≤ w_m ≤ L; RAM and VRAM constraints per device. Latency terms are compute, memory copies, disk reload of overflow beyond RAM, and communication (windows × t_comm). — [prima.cpp paper, HTML v3](https://arxiv.org/html/2504.08791v3)
- Halda algorithm: (1) initialise w proportional to available memory, n = 0; (2) enumerate valid k (factors of L excluding L; at most 11 for L ≤ 100); (3) loop: compute W and k, classify devices into M1 to M4 (macOS no Metal and low RAM; macOS Metal and low RAM; Linux/Android and low RAM; sufficient RAM or slow disk), stop if classes are unchanged, otherwise compute coefficients, solve a fixed-k ILP per k with HiGHS and keep the best. Calibration step moves the slowest-disk overloaded device to M4 if another device has free VRAM. — [prima.cpp paper, HTML v3](https://arxiv.org/html/2504.08791v3)
- Layer count for device m: `l_m = floor(L/W)·w_m + min(w_m, max(0, R − sum_{j<m} min(w_j, R)))`, R = L mod W. Scheduling takes 10 to 12 ms for 4 to 32 devices. — [prima.cpp paper, HTML v3](https://arxiv.org/html/2504.08791v3)
- Pipelined-ring parallelism: each token makes k rounds around the ring, each device processes a window of w_m layers per round and prefetches its next window from disk meanwhile. k = 1 equals plain pipeline parallelism. Devices assigned a single layer are removed (kept as a zero-work relay if they are the only hop). Halda can rerun when the queue is empty because layers are not migrated. — [prima.cpp paper, HTML v3](https://arxiv.org/html/2504.08791v3)
- Reported: 70B at 674 ms/token on four consumer devices; 5 to 17x lower TPOT than llama.cpp, exo and dllama. Authors' own claim, not independently verified. — [arXiv abstract](https://arxiv.org/abs/2504.08791)
- Only the first 100,000 of 186,433 characters of the paper page were read. The last part, including evaluation details, was not.

### Inferences
- Minimal re-implementable algorithm: (a) probe each node for free RAM/VRAM, matmul GFLOPS, memory bandwidth, disk read MB/s, RTT to its ring neighbour; (b) weight_i = free_mem_i (exo) optionally scaled by measured throughput; (c) largest-remainder rounding with a minimum of one layer; (d) reject if any share exceeds free memory; (e) try subsets of nodes ordered by speed and pick the lowest estimated TPOT = sum(layers_i / speed_i) + hops × RTT. Rule (e) is my simplification, not from the sources.
- For a pipeline of heterogeneous nodes, estimated token time is dominated by the slowest stage plus network RTT per hop, so adding a slow phone can lower speed. Halda's removal of single-layer devices reflects this.

### Gaps
- prima.cpp coefficient definitions (a, b, c) were not extracted.
- exo's actual device-FLOPS probe (it ships a static chip table in older versions [UNVERIFIED]) was not confirmed.

## 3. Model distribution and multi-model serving

### Takeaway
Three options exist: each node downloads (exo, distributed-llama's launch script on the root), coordinator pushes (distributed-llama root forwards weights; rpc-server receives tensors over RPC with an optional cache), or shared storage. For NetworkPancake, per-node download of only its layer range, with coordinator-provided URLs and hashes, is simplest and bandwidth-efficient.

### Cited Findings
- distributed-llama: the root loads the model and forwards weights to workers; workers need no model-specific configuration. — [distributed-llama](https://github.com/b4rtaz/distributed-llama)
- llama.cpp RPC: the client sends tensors to each rpc-server. With `-c` the server caches large tensors in `$HOME/.cache/llama.cpp/rpc` (override with `LLAMA_CACHE`) to avoid resending. — [llama.cpp RPC README](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/rpc/README.md)
- exo: models are pulled from Hugging Face via the API, with configurable and read-only shared storage directories. Each model runs as its own instance, so several models can be served at once. — [exo README](https://github.com/exo-explore/exo)
- prima.cpp relies on mmap of local files with prefetch from disk, so each device needs its own copy of the file, or at least its layers [inference; README not read]. — [prima.cpp paper, HTML v3](https://arxiv.org/html/2504.08791v3)
- LM Link cannot copy a model between machines. — [LM Link summary](https://www.dell.com/en-us/blog/lm-studio-s-lm-link-local-ai-everywhere/)

### Inferences
- GGUF is single-file, but a node only needs the tensors for its layers. Use HTTP Range requests against Hugging Face (or coordinator mirror) for the tensor byte ranges listed in the GGUF header [design suggestion, UNVERIFIED feasibility with llama.cpp loaders, which expect a file; sparse-file or re-sharded GGUF via `gguf-split` is the safer route].
- Multi-model: treat each (model, placement) as an "instance" like exo. Nodes with spare memory can host several instances. Route by the `model` field in the OpenAI request.

### Gaps
- No source found on partial-GGUF loading for pipeline shards in llama.cpp.
- NAS mmap over network performance was not researched.

## 4. UI technology and phone participation

### Takeaway
Serve a single HTML page plus REST and WebSocket from the coordinator, as exo's dashboard does. Phones join as workers through a native Android app or Termux; iOS is limited to foreground or client-only roles.

### Cited Findings
- exo's dashboard is served by the node itself at port 52415, alongside OpenAI, Claude and Ollama-compatible endpoints. — [exo README](https://github.com/exo-explore/exo)
- Android workers have been run as rpc-server in Termux, with Vulkan reported unreliable. — [phone guide](https://www.kunalganglani.com/blog/phone-as-gpu-local-llm)
- prima.cpp's cost model treats Android as a platform where the OS swaps memory out. — [prima.cpp paper, HTML v3](https://arxiv.org/html/2504.08791v3)

### Inferences
- A PWA or WebView can show the dashboard on any phone, but a browser tab cannot run native llama.cpp. WebGPU/WASM inference in-browser is possible but slow and memory-limited [UNVERIFIED].
- Android: a foreground service app with a persistent notification, wake lock and a native worker is the practical route [UNVERIFIED general Android knowledge]. iOS: background execution, JIT and memory limits make a continuously running worker unrealistic; use iOS only as a dashboard or chat client [UNVERIFIED].
- Expose the client API as OpenAI-compatible `/v1/chat/completions` and `/v1/models`, which Open WebUI, LM Studio-style clients and most apps accept.

### Gaps
- No primary source on iOS background limits or Android foreground-service constraints was fetched.
- Native app comparison (Flutter, Kotlin, SwiftUI) not researched.

## 5. Minimal coordinator to worker API

### Takeaway
Seven operations cover the control plane. It is my design, informed by exo's `/state` and instance model and by rpc-server's flags, not copied from any source.

### Cited Findings
- exo splits control into cluster state (`/state`), model catalogue (`/models*`), instance lifecycle (`/instance*`) and client inference endpoints. — [exo README](https://github.com/exo-explore/exo)
- Petals exposes a state snapshot and a reachability check. — [health.petals.dev repo](https://github.com/petals-infra/health.petals.dev)

### Inferences (proposed minimal surface)
- `POST /workers/register` (worker to coordinator): id, OS, backends, RAM/VRAM, build version, address.
- `POST /workers/{id}/heartbeat`: free memory, load, temperature or battery, current tok/s.
- `POST /workers/{id}/probe` (coordinator to worker): run benchmark, return FLOPS, memory bandwidth, disk MB/s, RTT to peer.
- `PUT /workers/{id}/assignment`: model id, layer range, next-hop address, quant, KV settings.
- `POST /workers/{id}/models/fetch` and `GET /workers/{id}/models`: download status and hashes.
- `POST /workers/{id}/start` and `/stop`: launch or stop the llama.cpp/rpc process for an instance.
- `GET /state` (UI) and WebSocket `/events`: nodes, instances, progress, per-node tok/s.
- Client side: `/v1/models`, `/v1/chat/completions`.
- Security: bind to LAN or VPN only, with a shared token, because rpc-server is unauthenticated by its own README.

### Gaps
- No standard protocol exists among the surveyed projects for this; the list above is untested.
