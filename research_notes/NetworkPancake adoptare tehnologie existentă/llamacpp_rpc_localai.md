# llama.cpp RPC, LocalAI P2P/federated and KoboldCpp RPC as bases or transports for NetworkPancake (Strata engine)

Date of research: 2026-11-10. About 17 tool calls. Items marked (unverified) rest on secondary or snippet sources only. Not read: the last 40 KB of ggml-rpc.cpp, transport.h (socket options, RDMA code), the LocalAI Distributed Mode page itself, and the LocalAI backend template source.

## 1. llama.cpp RPC: wire protocol, backends, layer assignment, security, limits

### Takeaway
RPC is a ggml-level remote-buffer/remote-graph-execution protocol, not a model-level protocol. The remote end must be a ggml backend, and the main host must run llama.cpp. Its project README calls it a fragile, insecure proof of concept. An external engine like Strata cannot sensibly speak it. The protocol is low-level (tensors and ggml ops) and is version-locked to GGML_OP_COUNT.

### Cited Findings
- Protocol version is 9.0.0 (major/minor/patch defines). `GGML_RPC_MAX_SERVERS` is 16. A static assert ties `GGML_OP_COUNT` == 102 to the protocol, with a note to bump the patch version if the op count changes. — [ggml/include/ggml-rpc.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml-rpc.h)
- The C API is `ggml_backend_rpc_init`, `_buffer_type`, `_get_device_memory`, `_start_server`, `_add_server`, `_reg`. The server takes endpoint, cache dir, thread count and device list. — same file
- The command set (enum rpc_cmd, 0-24) is: ALLOC_BUFFER, GET_ALIGNMENT, GET_MAX_SIZE, BUFFER_GET_BASE, FREE_BUFFER, BUFFER_CLEAR, SET_TENSOR, SET_TENSOR_HASH, GET_TENSOR, COPY_TENSOR, GRAPH_COMPUTE, GET_DEVICE_MEMORY, INIT_TENSOR, GET_ALLOC_SIZE, HELLO (=14), DEVICE_COUNT, GRAPH_RECOMPUTE, MEMSET_TENSOR, SET_TENSOR_2D, GET_TENSOR_2D, COMM_INIT, COMM_ALLREDUCE, COMM_FREE, NONE, COUNT. COMM_* (server-to-server communicator/allreduce) is new compared with older versions. — [ggml/src/ggml-rpc/ggml-rpc.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-rpc/ggml-rpc.cpp)
- Handshake: the client sends HELLO with its connection caps. The server replies with major, minor, patch, a flags byte and its own caps. The client rejects the connection if `major` differs or the server's `minor` is greater than the client's. Patch is only logged. The same exchange does transport auto-negotiation (this is what enables RDMA with TCP fallback). — same file
- Framing is a 1-byte command, then an 8-byte request size, then the input struct, then an optional payload. Responses are an 8-byte size plus data. Structs are packed. — same file
- Graph compute: the client serializes the whole ggml sub-graph (a DFS over tensors, each carrying op, op_params, shapes, strides and src ids) and sends GRAPH_COMPUTE. If the same graph UID was already sent, it sends only GRAPH_RECOMPUTE. The UID cache is capped at 1024 entries. `nb[]` is serialized as uint32, which can truncate strides over 4 GiB. — same file
- Weight cache (`-c`): the server advertises SEND_TENSOR_HASH only if it was started with a cache dir. Hashing applies only when the tensor is larger than 10 MiB (`HASH_THRESHOLD`) and its buffer usage is WEIGHTS. The client sends an FNV-1a hash. On a hit, no data is sent. On a miss, it sends SET_TENSOR with cache_flag and the server stores `cache_dir/<16hex>`. — same file
- README flags and variables: `-c` (cache, default `$HOME/.cache/llama.cpp/rpc`), `--device`, `-p` port, `--rpc host:port,...` on llama-cli/llama-server, `--tensor-split`. Environment variables are `CUDA_VISIBLE_DEVICES`, `LLAMA_CACHE`, `GGML_RPC_NO_RDMA`, `GGML_RPC_DEBUG` (0-3). — [tools/rpc/README.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/rpc/README.md)
- RDMA: Linux uses RoCEv2 through libibverbs. macOS uses Thunderbolt 5 RDMA via librdma and needs macOS 26.2 or later with `rdma_ctl enable` from Recovery. It is used only if both peers support it, with TCP fallback. — README above. The RDMA code itself (transport.h) was not read.
- Backends: RPC is built with `-DGGML_RPC=ON` on both sides. Each server exposes whatever local ggml devices it was built with (CUDA, Metal, CPU shown in the README). Vulkan and SYCL are not named in the README. Because the server is just a ggml backend host, any compiled-in backend should work (inference, unverified). — README above
- Layer assignment: by default weights and KV are spread over local and remote devices in proportion to available memory. `--tensor-split` overrides this. — README above
- Security (README): "proof-of-concept", "fragile and insecure". Never run it on an open network or in a sensitive environment. — README above
- Performance, 10GbE, Mac Studio M2 Ultra + DGX Spark. Qwen2.5-7B Q4_K_M generation drops from 91.8 to 52.7 tok/s with RPC, while prompt processing rises from 76.1 to 317.7. Qwen2.5-72B Q4_K_M generation is 11.1 local vs 5.9 with RPC. — [popularai.org](https://www.popularai.org/p/10gbe-local-ai-llm), a secondary summary with its own measurements.
- 2x RTX 3090 community test: 50 tok/s on one machine, 37 over RPC on 10GbE, 38 on 50GbE, 38 with localhost RPC. This suggests the cost is per-token protocol overhead (latency), not bandwidth. The author labels it community testing. — same page
- Issue #22850 (May 2026, build b9033): Llama-2-7B Q4_0 over a 2.5 GbE link. Generation fell from about 180 tok/s local to about 130 (best) and about 80 (worst). Prompt processing lost more than 33%. Network use was tens of MB/s, so the link is latency-bound. The poster's (LLM-written) analysis says graph metadata is re-sent on every execution, small tensors under 10 MiB skip the hash path, and the request/response is synchronous. The issue was closed the same day by a maintainer for violating the AI-usage policy. No maintainer confirmed any of it, and it is labeled bug-unconfirmed. Treat the numbers as unverified. — [issue 22850](https://github.com/ggml-org/llama.cpp/issues/22850)
- Mixed hardware, 1 GbE and Wi-Fi: I found no reliable measured tok/s figure on 1 GbE or Wi-Fi with named hardware. The kjaiswal repo says 1GbE "works but is slower", with no numbers (per a search snippet, unverified). — [kjaiswal/llama-cpp-distributed-benchmarks](https://github.com/kjaiswal/llama-cpp-distributed-benchmarks)
- The license of llama.cpp is MIT. — [GitHub API](https://api.github.com/repos/ggml-org/llama.cpp), about 134k stars, pushed 2026-11-10.

### Inferences
- External engine: the wire protocol is a remote ggml-tensor-op protocol, with graphs as ggml nodes and a version tied to the ggml op list. A non-ggml engine would have to implement a ggml buffer allocator and execute arbitrary ggml ops (mul_mat, rope, flash-attn, MoE mul_mat_id and so on) to speak it. That is a re-implementation of ggml, so Strata cannot be a drop-in rpc-server replacement. Reading the code, a node must run ggml.
- Because the host decides the layer split and sends every token's graph, the RPC unit is a layer-pipeline stage. Every token pays at least one round trip per remote device, plus synchronous request/response. On Wi-Fi or 1 GbE with 2-10 ms RTT, per-token latency dominates. This is consistent with the 38 tok/s plateau on 10GbE through 50GbE and localhost.
- The README gives no MoE-specific handling over RPC. Expert tensors are ordinary weights, split by layer or tensor-split (inference, see section 3).

### Gaps
- No primary measurement on 1 GbE or Wi-Fi with named hardware.
- Vulkan and SYCL RPC servers were not confirmed in docs.
- transport.h (TCP_NODELAY, RDMA code) and the second half of ggml-rpc.cpp were not read.
- The RDMA PR number and date were not found.

## 2. MoE behavior over RPC and the expert-cache PR

### Takeaway
PR #29887 is verified as merged, but it is a local GPU-vs-host-RAM expert cache (`--moe-cache-mib`) and has nothing to do with RPC or remote experts. I found no RPC-based expert placement or remote expert cache in llama.cpp.

### Cited Findings
- PR #29887 "llama : add a GPU cache for MoE experts kept in host memory", by am17an, merged 2026-10-07. It ports the cache from qvac-fabric. For batches of 32 tokens or fewer, experts in host memory run on the GPU, and an LRU cache keeps experts in VRAM, uploading only misses. Flag: `--moe-cache-mib N`, suggested about 10% of expert size. Reported 1.6x-2.2x token-generation speedup on RTX 4090/5090 with 72-89% hit rates (author benchmarks, model named "Qwen3.8-Flash-Next Q4_0"). — [GitHub API PR 29887](https://api.github.com/repos/ggml-org/llama.cpp/pulls/29887)
- Other competing expert-cache PRs exist (#17044, #21609, #23170, #26563 listed by a Lemmy comment, unverified). A disk-paging RFC is [discussion 23324](https://github.com/ggml-org/llama.cpp/discussions/23324). — [search results](https://www.developersdigest.tech/blog/llama-cpp-moe-offload-n-cpu-moe-vs-moe-cache-2026)
- Existing flags `--cpu-moe` and `--n-cpu-moe` keep experts in host RAM (same blog, secondary).

### Inferences
- Over RPC, MoE experts are ordinary weight tensors placed according to `--tensor-split`/layer assignment. Whole layers, including all their experts, live on one remote node, and activations cross the network only at stage boundaries. No expert-level routing across nodes. This is consistent with Strata's own expert-aware design being something llama.cpp RPC cannot express. (Inference: no source describes it explicitly.)

### Gaps
- No source on MoE-specific RPC tok/s.
- No evidence of any RPC-level expert PR; the search did not exhaustively cover the PR list.

## 3. LocalAI: license, architecture, P2P/federated, node selection, activity

### Takeaway
LocalAI is MIT and very active (about 50k stars, push on 2026-11-10). It has three distribution mechanisms: experimental P2P (federated or llama.cpp-RPC worker), a new PostgreSQL+NATS Distributed Mode, and MLX Distributed. P2P worker mode is llama.cpp-only, but federated mode and Distributed Mode route whole requests to nodes that run any backend.

### Cited Findings
- MIT license, 50,285 stars, 188 open issues, pushed 2026-11-10. — [GitHub API](https://api.github.com/repos/mudler/LocalAI)
- Federated: each request goes to one node, and each node loads the full model. Start with `local-ai run --p2p --federated`. The balancer gives each new TCP connection to the online node with the fewest active connections, with random tie-break. HTTP keep-alive pins a connection to one node. — [P2P docs](https://localai.io/docs/features/distribute/)
- Worker mode (model sharding): weights are split in proportion to memory, it is llama.cpp-only, only a single model is supported, and workers cannot be added after inference has started. Without P2P, point at workers via `LLAMACPP_GRPC_SERVERS="addr:port,..."`. — same docs
- Discovery is a shared token over libp2p (DHT, or mDNS when `LOCALAI_P2P_DISABLE_DHT`) and edgevpn (prior notes). Container deployments need host networking. The page calls it experimental/tech-preview. — same docs
- UI: Operate, Swarm, P2P shows the token and "Add a node". `local-ai explorer` serves a public swarm list, and listing a swarm publishes its token, so anyone can use its workers. — same docs
- Distributed Mode: PostgreSQL for state and node registry, NATS for coordination. Stateless frontends, workers self-register, and a generic worker starts a backend gRPC process on a dynamic port when a model is requested, after which the frontend calls gRPC LoadModel. It requires auth and Postgres (no SQLite). Release 4.10.0 shipped stabilisation fixes. — [Distributed Mode (search result)](https://localai.io/docs/features/distributed-mode/); I read only the search summary, not the page.

### Inferences
- For NetworkPancake, LocalAI federated mode (or Distributed Mode) is a request router. It does not shard a model, so Strata would have to hold a full model per node. That matches Strata's `--serve` per-node design.

### Gaps
- The Distributed Mode page and how it selects nodes (load, VRAM?) were not read.
- Activity of the P2P component specifically (commits to core/p2p) was not checked.

## 4. LocalAI backend gRPC contract and "Strata as a LocalAI backend"

### Takeaway
Yes. It is a language-agnostic gRPC contract defined in `backend/backend.proto` (service `backend.Backend`). A Strata backend needs only a thin shim and a run.sh. This is the cleanest and smallest integration.

### Cited Findings
- File: `backend/backend.proto`, package `backend`, service `Backend`, proto3, with 57 RPCs. Text-relevant ones: Health, Free, LoadModel, Status, GetMetrics, ModelMetadata, Predict, PredictStream, TokenizeString, Detokenize, Embedding, Rerank, Score, TokenClassify. The rest cover audio, image, video, stores, fine-tune, quantization, and a Forward proxy (HTTP passthrough). — [backend.proto](https://raw.githubusercontent.com/mudler/LocalAI/master/backend/backend.proto)
- Unsupported RPCs (e.g. SlotCache, AudioTranscriptionLive, Forward) may return gRPC UNIMPLEMENTED. — same file
- `PredictOptions`: Prompt, Seed, Threads, Tokens, TopK/TopP/MinP, Temperature, Grammar, Messages, Tools/ToolChoice (JSON), Logprobs, Metadata, MainGPU, TensorSplit, ModelIdentity, CorrelationId. `ModelOptions`: Model, ModelFile, ContextSize, Threads, NGPULayers, MainGPU, TensorSplit, Options, Overrides, EngineArgs (JSON), EnvVars, and Proxy. `Reply`: message, tokens, prompt_tokens, timing_*, logprobs, chat_deltas. — same file
- Backends are described as gRPC services that can be written in any language. External backends are attached with `--external-grpc-backends "name:/path/run.sh"` or `name:host:port`. — [advanced docs](https://localai.io/docs/advanced/)
- Packaging: OCI images listed in a YAML gallery (`name`, `uri`, `alias`, `tags`). The directory must contain a top-level `run.sh` (entry point) and optionally `metadata.json`. User backends go in `LOCALAI_BACKENDS_PATH`. Optional Sigstore verification. Starting points are a Python template, the Go `piper` backend and the C++ `llama-cpp` backend. — [backends docs](https://localai.io/docs/backends/)
- The docs page does not document the port or exact run.sh contract (read source). The model YAML selects the backend with `backend: name`.

### Inferences
- Effort estimate (my own, not from a source):
  - MVP, 3-5 days: a small gRPC server (C++ with the Strata process, or a Go/Python shim that spawns `strata-poly --serve` and speaks its GEN/T/DONE line protocol). Implement Health, LoadModel, Free, Status, Predict (collect T lines until DONE), PredictStream (emit T as Reply chunks), TokenizeString (stub or return count). run.sh launches it on the port LocalAI passes. Tested with `--external-grpc-backends`.
  - Production, 2-3 weeks more: OCI image build per CUDA/HIP/CPU variant, gallery YAML, chat templating (LocalAI does templating on the Go side), Tools, Logprobs, cancel and timeout, Metrics, model download via the model gallery, CI.
  - Risks: GEN/T/DONE carries no explicit token IDs, logprobs or usage counts (assumed from the task description, to verify), and the proto evolves frequently (57 methods, the ModelIdentity field was added late).
- This gives NetworkPancake federated multi-node routing, a UI and an OpenAI API for free. It does not give expert-level distribution. P2P worker sharding will still ignore Strata (llama.cpp only).
- Distributing a combined work: LocalAI (MIT) plus llama.cpp (MIT) plus Strata (own license) are compatible. MIT requires keeping the copyright and license notices. Since Strata would be a separate gRPC process, no copyleft linkage issue arises. go-libp2p and edgevpn licenses were not checked (go-libp2p is Apache-2.0/MIT in my recollection, unverified).

### Gaps
- The exact method contract run.sh must follow (listen address, port argument) was not read in the template.
- No example of a third-party non-llama backend merged recently was checked; the Python vllm, diffusers and others show the pattern (advanced docs mention vllm).

## 5. KoboldCpp, Ollama, LM Studio (brief) and the integration comparison

### Takeaway
KoboldCpp reuses llama.cpp RPC and gives no extra engine hook. Ollama and LM Studio offer no documented plug-in point for a custom engine. The smallest integration, in order: (1) LocalAI gRPC backend, (2) an OpenAI-compatible HTTP node behind a router, which needs only a Strata HTTP shim, and (3) last and impractical, an rpc-server replacement.

### Cited Findings
- KoboldCpp v1.114 (about May 2026) added RPC backend support. It is incompatible with context shift and fast-forward. — [release](https://newreleases.io/project/github/LostRuins/koboldcpp/release/v1.114) (from prior notes, not re-read)
- Ollama has runners (a legacy llama.cpp runner, the Go "ollamarunner" and an MLX runner). Reports differ on whether 0.30 drops its ggml fork for direct llama.cpp (secondary sources only). External llama.cpp server can be set through environment variables (per a tutorial). No documented third-party-engine API found. — [martianlee notes](https://martianlee.github.io/kb/2026-03-15-ollama-architecture), [aiweekly](https://aiweekly.co/alerts/ollama-drops-custom-ggml-fork-for-llamacpp) (secondary)

### Inferences
- Effort ranking (my estimates):
  - LocalAI gRPC backend: about 1 week MVP. It reuses a ready router, UI, API and OCI distribution.
  - Own OpenAI-compatible shim plus NetworkPancake router: about 1-2 weeks for a shim with /v1/chat/completions streaming, plus router work that LocalAI federated already does. Most independent.
  - rpc-server replacement: months, because it needs a ggml op executor. Not recommended.
  - Using llama.cpp RPC as transport for Strata: not possible without ggml.
- Slow networks and MoE: RPC pipeline sharding pays a per-token round trip, and measured results show 40-50% generation loss even on 10GbE with Mac+Spark. On 1GbE or Wi-Fi it is expected to be worse (inference, no primary number). Request-level routing (federated) or expert-level offload (Strata) avoids the per-token network cost.

### Gaps
- LM Studio extension points: no results found.
- KoboldCpp RPC source and measurements: not read.
