# NetworkPancake: what the pancake-main (Strata fork) repository offers for a networked model split

Scope: read-only inspection of `C:\Users\iancu\pancake-main` (no builds, nothing run). "Source" below means a file:line in that repo, not a URL. Line numbers were taken from the working tree on 2026-11-10. Anything I could not verify is listed under Gaps.

Orientation: the repo holds several engines. `strata` (src/program/generate.cpp, ~22k lines, Qwen3.8-Flash-Next / its own packed artifact), `strata-poly` (src/poly/, generic GGUF engine, 4 architectures), `strata-dsv41` (DeepSeek V4.1), a Metal engine (metal/), a SYCL/Intel engine (sycl/), and `strata-split` (sFNX, Linux-only). A Python server (serve/server.py) fronts whichever engine over a stdin/stdout line protocol.

---

## 1. How does `strata-poly --serve` work, and is there any HTTP / socket / multi-process code?

### Takeaway
The engine speaks a line protocol on stdin/stdout (no sockets). All HTTP (OpenAI and Anthropic dialects) lives in a separate Python process, `serve/server.py`, which spawns the engine as a child and talks to it through pipes. The repo contains no C++ socket code at all, so any network transport between machines would be new work. The Python server already has two multi-engine wrappers (replicas, split_serve) whose stand-in `Engine` object is the natural attachment point for a remote engine.

### Cited Findings
- Protocol header: `strata-poly --model M.gguf --serve` "the line protocol serve/server.py speaks"; GEN and GENI (prompt with pictures), STOP and QUIT in; RESUME, PP, T, DONE and ERR out. Source: src/poly/main.cpp:4-7.
- stdin reader thread: `std::getline(std::cin, l)`; a line `STOP` sets an atomic stop flag at once (so a STOP is seen while a request runs), anything else is queued; EOF pushes `QUIT`. Source: src/poly/main.cpp:270-295.
- `serve(Session&)` at src/poly/main.cpp:628. On start it prints `INFO context=... expert_slots=... engine=...` and `READY <context> stop` (main.cpp:630-638). Loop at main.cpp:643+:
  - `QUIT` ends the loop (main.cpp:645).
  - `VRAM [reserve_mib]` resizes the expert cache between requests; reply `VRAM reserve_mib=... expert_slots=... prompt_chunk=...` (main.cpp:654-684).
  - `GEN <max_new> [key=value ...] <id,id,...>` and `GENI <max_new> image_pad=<id> <file> <id,id,...>`; parsed by `parse_gen` (main.cpp:552+). Keys include sampling keys and `turn_at=POS` (checkpoint place). Anything else returns `ERR expected: GEN ...` (main.cpp:685-689).
  - Output per request: `RESUME <at>` (prefix reused), `PP <at> <total> <ms> <tok/s>` (prompt progress), `T <token_id>` per generated token (flushed each), then `DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused> <experts from VRAM> <experts> ...` (main.cpp:728-760). Errors: `ERR <text>`; after an exception the engine is `reset()` and the conversation is dropped (main.cpp:741-748).
- The generic `strata` engine (generate.cpp) speaks a larger superset: `BGEN`, `BGENI`, `BCO`, `BSTOP`, `BYIELD` (batch slots), `SAVE <path>`, `RESTORE <path>` (session files), `VRAM`, `STOP`, `QUIT`. Source: src/program/generate.cpp:14052, 15683-15833, 16060-16062. strata-poly implements only GEN/GENI/VRAM/STOP/QUIT.
- HTTP server: `serve/server.py` (11,564 lines). Docstring lists `POST /v1/chat/completions` (OpenAI, stream and non-stream), `POST /v1/messages` (Anthropic), `GET /v1/models`, `/props`, `/slots`, `/health`, `/mcp`; "One sequence at a time behind a FIFO" (serve/server.py:1-7). Routes at 9082 (`/v1/models`), 9176/9236/9249 (chat/messages/responses). `class Server(ThreadingHTTPServer)` at server.py:9894; `main()` with `--host`, `--port` at 10860-10925. `/v1/responses` also exists (server.py:6056).
- Engine child process: `class StrataEngine` (server.py:1243) "`strata --serve` reads `GEN <max_new> <ids>` lines and streams `T <id>` lines, then `DONE ...`"; it detects strata-poly by exe stem (server.py:1294-1296) and adds `turn_at`. `Engine` is a Python `Protocol` (server.py:420) with a `MockEngine` (server.py:441), launched via `--engine mock`.
- Multi-engine wrappers already in Python, each a stand-in for the engine: `serve/replicas.py` (data-parallel replicas, routing by cached prefix hash, 632 lines) and `serve/split.py` (prefill engine + decode engine, session handed over by file `SAVE`/`RESTORE`, 488 lines; docs/SPLIT_SERVE.md says it is measured slower than the layer split on alike cards).
- Network exposure of the HTTP side: `host` default 127.0.0.1; `--host 0.0.0.0 --api-key ...`, multiple keys, CORS, DNS-rebinding host checks. Source: docs/DETAILS.md:640-662; server.py:4970 (`api_key`), 8574-8597.
- Socket grep: a search for `socket(`, `connect(`, `listen(`, `AF_INET`, `WSAStartup`, `sockaddr`, `arpa/inet`, `netinet` across src/, include/, metal/, sycl/ returned no networking hits (the only "socket" hit is CPU socket text in src/kernels/cpu/pool_tasks_test.cpp:254). Other "http" hits in C++ are URLs in comments (not individually checked).
- Class named `RemoteExperts` is NOT network code: it is "a static, profile-filled expert tier on another CUDA device ... results return through the existing pinned CPU rows" (include/strata/core/remote_experts.hpp:15-17; src/core/remote_experts.cpp, 501 lines). Likewise `PeerExperts` (src/core/peer_experts.cpp) is a second-GPU tier.

### Inferences
- A remote stage can be attached in two places: (a) in C++ as an `Engine` subclass behind `next_` (see section 3), or (b) in Python as another `Engine`-protocol object (like `ReplicaEngine`/`SplitEngine`) that forwards the existing text protocol over TCP to a remote `strata-poly --serve`. (b) gives request-level distribution (replicas over LAN, remote prefill) cheaply, but does not split one model's layers across machines.
- Because the HTTP layer is already OpenAI/Anthropic compatible, "NetworkPancake" needs no new HTTP server; it needs an inter-node transport only.

### Gaps
- I did not read serve/server.py's engine start-up (`_start`, server.py:3467) in full, so how an exe path/args are chosen for a remote engine is unverified.
- Not verified which of the ~20 `http` hits in C++ are real clients (e.g. model download helpers); they appear to be comments/URLs.

---

## 2. Backends, hardware requirements, minimum GPU

### Takeaway
Backends in the tree: CUDA (NVIDIA), HIP (AMD; RDNA3 gfx1100 is the tuned target, gfx1201/gfx1030/gfx906 etc. also), Metal (Apple Silicon), SYCL (Intel Arc, experimental), plus CPU kernels with AVX2/AVX-512 (x86). No Vulkan, no OpenCL, no WebGPU. CUDA floor in the supported matrix is sm_75 (RTX 20); Pascal (sm_60/61, GTX 10xx), Volta and even Maxwell (sm_52) build only through the opt-in `STRATA_EXPERIMENTAL_SM60` path, with CUDA 12.x (CUDA 13 dropped them).

### Cited Findings
- CUDA arch policy: default arch list `86-real;89-real;120` (or 120) if unset, and a `native` option; archs below 75 are a `FATAL_ERROR` unless `-DSTRATA_EXPERIMENTAL_SM60=ON`; below 52 always fatal; sm_52 warns "Maxwell... community-tested on one reporter's Quadro M6000". Source: CMakeLists.txt:260-310 (option at 278). `__dp4a` (sm_61+) and `__nanosleep` (sm_70+) are replaced for lower archs in include/strata/kernels/dp4a.hpp (comment lines 7-24, macro at 42), "bit-exact for the first".
- Support matrix: Tesla P100 6.0 and "Tesla P40 / P4, GTX 10 series 6.1" run "the CUDA 12 engine" with BF16 projections through fp32 (cuBLAS has no BF16 GEMM there); RTX 20 (7.5) "supported, the ready-made engine"; V100 7.0 via CUDA 12 engine; "A 2 GB GT 1030 ... cannot hold any of the model". Source: docs/OLDER_GPUS.md:17-40 (table), 36 (GT 1030).
- Multi-GPU docs: "A card older than the RTX 20 series (compute capability below 7.5)... Pascal and Volta cards are admitted when you name them with --gpus ... experimental CUDA 12 engine" (docs/MULTI_GPU.md "Not supported"). Also: a card with less than 8 GB VRAM beside others is refused unless forced (same section).
- Tensor-core / MMA usage: files using wmma/mma: src/kernels/cuda/qsa_prompt_attn.cu (43 hits), iq_kernels.cu (20), qsa_select.cu (19), poly/kernels.cu (4), fused_gr.cu (1), native_qsa_score.cu (1) (grep count, not individually read). The README states tf32 mma in the QSA scorer "has a portable fp32-FMA fallback" (CMakeLists.txt:~268 comment). docs/WINDOWS_CUDA_SM75.md: a GTX 1660 Ti (sm_75, no tensor cores, 6 GB) built with MSVC + CUDA 12.8 and ran `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` UD-IQ2_XXS at 34.3 +/- 14.1 tok/s decode (3 runs, wide interval), 2319 of 10496 experts in VRAM (docs/WINDOWS_CUDA_SM75.md:5-45). Build line there: `-DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared`, targets `strata strata-q36 strata-poly`.
- Turing tensor-core paths are opt-in and "nobody here has an sm_75 card to confirm" (include/strata/core/tc_sm75.hpp:1-13; sm75_series.hpp:4).
- HIP: options `STRATA_ENABLE_HIP` ("wave32 AMD GPUs (gfx1100, gfx1201; gfx1012/gfx1030 unvalidated)", CMakeLists.txt:102) and `STRATA_HIP_GFX906` (wave64 compat layer, :95). A CUDA-to-HIP compat header set lives in include/strata/hip_compat/. PANCAKE_STATUS.md describes this fork as developed on RX 7900 XT (gfx1100); branch main == port/rdna3-decode (per assignment, not re-verified).
- CPU kernels (src/kernels/cpu/): AVX2 (iq_avx2.cpp, q2_avx2.cpp, kq_avx2.cpp, tq_avx2.cpp), AVX-512 (iq_avx512.cpp, tq_avx512.cpp, nvfp4_avx512.cpp, behind a CPUID check, CMakeLists.txt:2047-2052, 2202), AVX1 variants, and `portable.cpp`. strata-poly's CPU experts use ggml-cpu's `vec_dot` (src/poly/cpu_experts.hpp:1-5) and an optional AVX2 multi-token path (cpu_kernels_avx2.cpp, built with `/arch:AVX2` or `-mavx2 -mfma -mf16c`, CMakeLists.txt:3186-3195).
- ARM NEON: a grep for `arm_neon|__ARM_NEON|aarch64` hit src/core/expert_source.cpp, src/core/session.cpp, src/kernels/cpu/expert_layout.cpp, src/kernels/cpu/portable.cpp, include/strata/platform/cpu_relax.hpp. These look like portability shims, not a NEON kernel set (not read closely).
- Metal: separate engine under metal/ (Metal Shading Language kernels in metal/kernels/*.metal, host code metal/src/engine/); docs/MAC.md: needs Apple Silicon, macOS 15+, "Intel Macs cannot run it"; measured on M5 Max 128 GB, M4 mini 32 GB, M1 Max. CI file .github/workflows/metal-build.yml exists. Metal engine implements its own `--serve` (metal/src/engine/window.cpp:210 mentions conversation checkpoints).
- SYCL (Intel Arc): `STRATA_ENABLE_SYCL` OFF by default, "EXPERIMENTAL" (CMakeLists.txt:14; sycl/ tree uses dpct-converted kernels, e.g. sycl/src/kernels/cuda/*.dp.cpp).
- Vulkan/OpenCL: grep for `vulkan|opencl|webgpu` across src, include, metal, sycl, CMakeLists.txt found only sycl/ files (the matches are in SYCL/dpct headers; I did not confirm they are real Vulkan code and none refer to a Vulkan backend of the engine). No Vulkan backend exists.

### Inferences
- A GTX 10xx node is the lowest realistic CUDA node: the build path exists (sm_61, CUDA 12.x, experimental) but throughput and correctness there are community-reported only, and P40/10xx need fp32 BF16 fallbacks.
- For NetworkPancake, a stage that only runs a few layers' dense attention + some experts is lighter than the whole model, which is why even 6 GB sm_75 can serve (it already runs a 35B-A3B at ~34 tok/s with most experts on CPU).

### Gaps
- Cannot say which individual kernels require tensor cores vs dp4a without reading each file; counts above are grep counts only.
- Not verified that the `strata-poly` target itself builds for sm_61 (docs name `strata`, `strata-q36`, `strata-poly` only for sm_75 on Windows).
- Did not verify any Android/iOS-capable build path (see section 7).

---

## 3. Layer structure, hidden-state hand-off, cutting at a layer boundary, KV cache location

### Takeaway
`strata-poly` already implements a pipeline (layer) split across several CUDA devices inside one process: one `Engine` object per card holds layers `[lo_, hi_)`, and the "residual" rows are copied device-to-host into pinned memory and handed to the next engine (`next_`), which continues from them. That is exactly a layer-boundary cut; replacing the host-memory hand-off by a socket is the whole network-split idea. Per-token activation = `streams() * n_embd` fp32; `streams()` is 1 for Qwen3.5/3.6 and Gemma4, and 4 for GLM5-next and DeepSeek4 (hyper-connection streams). KV/recurrent state is per stage and never crosses the boundary during normal decode.

### Cited Findings
- Design statement: "On several cards (Options::layer_split) each card has an engine of its own for a run of the layers, with their weights, caches, state and expert cache, and the last one has the output and the draft block. The first one is the one the caller holds: it runs its layers, hands the rows of the residual on to the next one through host memory, and the next one goes on from them." Source: src/poly/engine.hpp:11-14.
- Option: `std::string layer_split` ("auto" or the first layer of each later card, e.g. "20,30"). Source: src/poly/engine.hpp:109-111. `layer_starts()` computes split points by tensor bytes (src/poly/engine.cpp:263-320); minimum 2 layers per card (engine.cpp:289).
- Chain members: `int lo_, hi_; std::unique_ptr<Engine> next_; bool split_` (src/poly/engine.hpp:541-549). `runs(il)` tells if a layer is this engine's; the draft block lives on the last card (engine.hpp:405-406).
- Hand-off functions: `float* Engine::pass_rows(int n)` allocates pinned host buffer `h_pass_` of `rows * streams() * n_embd * 4` bytes (src/poly/engine.cpp:3249-3260); `void Engine::hand_on(pos, n, h)` does `cudaMemcpyAsync(h, r_, n*streams()*n_embd*sizeof(float), DeviceToHost)`, syncs, then sets `next_->pass_ = h; next_->pass_at_ = pos;` (engine.cpp:3263-3269).
- Single-token / small-window path: `Engine::eval()`: `run(tokens, n); if (next_) { hand_on(pos_, n, pass_rows(n)); next_->eval(tokens, n, want_logits); ... }` (engine.cpp:3271-3278). The next engine reads `pass_` instead of embedding (memcpy from `pass_` at engine.cpp:~3895-3905). The last engine produces logits (output norm, `lin(output_, ...)`, `finish_logits`, D2H copy of `n_vocab` floats, engine.cpp:3280-3300). `logits()`/`logits_of(i)` delegate to the last card (engine.hpp:216, 254).
- Prompt path (`read()`): pipeline over calls on separate threads ("the next one takes a call on a thread of its own, while this one runs the call after it"), `Behind behind; ... hand_on(...); behind.go([...]{ next_->eval(...) })` (engine.cpp:3815-3860). Prompt rows go through pinned `h_res_` (kReadBytes = 512 MiB, runtime.hpp).
- Which `next_->` operations traverse the chain (the interface a remote stage would need): eval, read, window, keep, reset, draft, images, save_state, load_state, save_rows, load_rows, vram_free_mib (engine.cpp, 17 uses), plus accessor delegation in engine.hpp (state_bytes, row_bytes, cached_experts, total_experts, drafts, logits, logits_of, took_record, ...). Counted by grep of `next_->` (17 in engine.cpp, 12 in engine.hpp).
- `streams()`: Qwen35 returns 1 (src/poly/qwen35.hpp:36), Gemma4 1 (gemma4.hpp:34), Glm5Next `kHc`=4 (glm5next.cpp:20,84), DeepSeek4 `kHc`=4 (deepseek4.cpp:18,58).
- n_embd values: Qwen3.8-Flash-Next `H = 2560` (include/strata/core/expert_source.hpp:23-25); Qwen3.6-35B-A3B n_embd is not stated directly; docs/QWEN36.md:30 gives the vision projector output width "2048 here, 2560 there", so 2048 is an inference. GLM5/DeepSeek n_embd not checked. Engine's own comment: GLM costs "64 KiB a token" and Qwen3.6 "8 KiB a token" of pinned residual rows (src/poly/runtime.hpp, comment above kReadBytes) = 4 x n_embd x 4 B = 64 KiB (so n_embd=4096 for GLM), and 1 x n_embd x 4 B = 8 KiB (so n_embd=2048 for Qwen3.6). That second reading confirms 2048 for Qwen3.6.
- So per generated token on a Qwen3.6 boundary: 8 KiB fp32 (4 KiB if sent as fp16/bf16, 1-2 KiB if quantized, which would change numerics). A 16-token verify window = 128 KiB fp32. GLM5-next/DeepSeek4: 64 KiB per token. Prompt prefill: n_tokens x same, e.g. 8,192 tokens x 8 KiB = 64 MiB per Qwen3.6 chunk.
- docs/MULTI_GPU.md "How the split works": "A token crosses from one card to the next once per verify window (a few hundred KB through pinned RAM). No NVLink or peer-to-peer access is needed, cards on x4 or x1 slots work"; prompts "flow through the cards in turn: while a later card reads chunk c, the first card reads chunk c+1" (MULTI_GPU.md, section "How the split works"; this describes the older `strata` engine's split, same shape).
- Pipeline windows (speculative next window before the last card finishes): `--pipeline-windows 2` in the `strata` engine; on by default for RDNA4 only; opt-in on NVIDIA (docs/MULTI_GPU.md "Pipeline windows"). Overlapped hand-off flag `STRATA_SPLIT_OVERLAP=1`, bounded wait `STRATA_SPLIT_WAIT_MS` default 30000 (MULTI_GPU.md "Overlapped hand-off").
- Where KV lives: each engine allocates attention caches for its own layers only, via `alloc_caches()` with virtual-memory growth (kKvGrow=4096 rows, `cover()`; src/poly/runtime.hpp; engine.hpp:336, 487-488). `max_context` default 8192 (engine.hpp:41). KV can be fp16, Q8_0 (`kv_q8`) or Q4_0 (`kv_q4`), both Qwen35-only (engine.hpp:80-84; engine.cpp:1040-1046). Recurrent (GDN) state is per layer and per stage (`recurrent_floats()`, `save_layer`, engine.hpp:327, 383). In the older `strata` engine, "Every card: ... its own session state (the KV cache of the full context)" (MULTI_GPU.md "What each card holds").
- Conversation save/restore already abstracts state vs rows: engine exposes `state_bytes/save_state/load_state` and `row_bytes/save_rows/load_rows`, which sum across the chain (`card_state_bytes() + next_->state_bytes()`, engine.hpp:279-288). `src/poly/conversations.hpp:1-17` states the contract (checkpoint = recurrent state + KV rows). This lets a remote stage keep its own KV and be checkpointed by the stage itself.
- Strata (non-poly) layer code: src/core/layer.cpp (1,656 lines; e.g. `moe_combine` at src/core/layer.cpp:479 per the comment in expert_source.hpp:27); main loop is inside src/program/generate.cpp (21,978 lines, not read in full). The `strata`/Flash-Next engine's stage split code: include/strata/program/split_walk.hpp, split_ring.hpp, split_rules.hpp (VRAM spill walk and ring sizing for a layer split) and generate.cpp.

### Inferences
- Smallest cut point: at the `hand_on()` / `next_->eval()` seam in src/poly/engine.cpp:3263-3278 and 3845-3850. A `RemoteStage` that serializes `(pos, n, rows[n*streams*n_embd])` and returns either logits (last stage) or nothing needs only: eval, read, window, keep, reset, draft, save/load_state/rows, plus a handful of stats accessors. The last stage returns `n_vocab` floats of logits per window row (`logits_of(i)`), i.e. 151,936-ish x 4 B per row if sent raw; better return only top-k or sample on the last stage (not implemented; the Session samples on the first engine, so the remote last stage would have to ship logits or the sampler must move, see Gaps).
- The `Engine` base class has separate virtual methods per architecture (`layer_head`, `layer_tail`, `output_rows`, `route`, ...; engine.hpp:313-397), so stage code is already per-layer modular.

### Gaps
- I did not trace `Engine::window()` / `keep()` / draft handling through `next_` line by line, so speculative-window rollback across a remote boundary (state restore after a rejected guess) is unverified. Note `keep(n)` and `load_state` are chain calls, so rollback exists in-process.
- I did not read `Engine::open()` to see whether `layer_split` stages load only their own layers' dense weights in strata-poly (the older `strata` engine has `STRATA_STAGE_TRIM=1` for this, MULTI_GPU.md). Whether each remote node would need the whole GGUF on disk is therefore unverified; GGUF is mmapped (engine.hpp:1-5: "reads the GGUF in place"), which suggests a stage touches only its own tensors, but not confirmed.
- How the sampler is attached in strata-poly (first card vs last card) not verified: `Session::generate` calls `engine.logits_of(i)` which delegates down the chain (engine.hpp:254), so sampling is on the caller (first) side.
- Exact n_embd for GLM/DeepSeek: not read.

---

## 4. Expert handling: ExpertSource, VRAM cache vs CPU, file source, conversation/disk tier, mem_pressure; could experts be served from a remote node?

### Takeaway
There are two expert paths. The `strata` engine has a clean `ExpertSource` interface (include/strata/core/expert_source.hpp:194+) that returns a pointer to a weight blob (bytes), so a network `ExpertSource` would move weights, 1.38 MB per Flash-Next expert, which is far too heavy per token. `strata-poly` instead computes CPU experts from GGUF bytes in `cpu_experts()`, whose input is only `x` (n_embd floats) plus expert ids/weights and whose output is n_embd floats. That function is the right seam for remote expert compute (activation-in, activation-out), not weight transfer. The repo's own prior research rejected network expert placement as too latency-bound for 40+ layers.

### Cited Findings
- `class ExpertSource` (virtual) at include/strata/core/expert_source.hpp:194-304: `blob(layer, expert)` returns `const uint8_t*` valid until the next call (:203), `blob_stable`, `blob_parts`, `prefetch*`, `pinned`, `device_alias`, `copy_blob(s)`, `warm*`, `begin_layer(layer, ids, k)` (:227). Concrete: `FileExpertSource` (:767) and `ArenaExpertSource` (:1318). Header contract: expert geometry fixed by the artifact `H = 2560, FF = 640, BLOB = 1,382,400` (:23-25).
- Cost model: `struct PcieModel` (expert_source.hpp:443-527) fits per-layer time as CPU `a + c*n_cpu + d*n_pcie` vs GPU `g0 + g*n_pcie` by online least squares and chooses how many missed experts to read over PCIe vs compute on CPU (`pick(nmiss, fixed)`, :507-519). This is a ready-made place to add a third cost (remote), but it models PCIe bandwidth not RTT.
- Expert profile files, adaptive swaps, `--adapt-*`, kmiss tier (include/strata/program/kmiss_tier.hpp), window trace (`STRATA_WINDOW_TRACE`, PANCAKE_STATUS.md) provide the hit-rate/profile data that tells which experts are hot.
- CPU experts in strata-poly: `cpu_experts(Workers&, const CpuExpert* experts, int n, int rows, int gu_type, int d_type, int n_embd, int n_ff, const float* x, float limit, bool gelu, float* out, meanwhile)` computes `sum_i weight_i * down(silu(gate x) * up x)` per token (src/poly/cpu_experts.hpp, near the end; struct `CpuExpert{gate, up, down, weight, row}`). It runs on `Workers` (own thread pool) and uses ggml-cpu `vec_dot`. Experts in VRAM are cached per layer slot and the rest are "computed on the CPU from the mapped file while the GPU does its share" (engine.hpp:1-5); a prompt batch stages missed experts through VRAM (`stage_`, `kStageRows`=32, runtime.hpp).
- Network-looking classes that are not network: `RemoteExperts` (other CUDA device; `begin(layer, x, ids, n_tok, k, kind, ...)` / `finish(out)` at include/strata/core/remote_experts.hpp:36-50) and `PeerExperts`. The `RemoteExperts::begin/finish` shape (send x + ids, return rows through pinned CPU rows) is a split-phase compute offload and would map to RPC, but it is CUDA-only and tied to `ExpertCache`.
- Expert Deferral (`--defer-experts`, Options::graph_experts, engine.hpp:105-108; PANCAKE_STATUS.md P1a): a layer's lower-ranked CPU experts are computed "while the card runs the next layer, and their sum is added at that layer's end". That is the same overlap trick a remote expert call would need. Qwen35-only (engine.cpp:1049).
- File/disk tiers: file-backed expert source exists (`FileExpertSource`; test tests/core/file_expert_source_test.cpp is named in the assignment but I did not open it), `STRATA_UNBUFFERED_LOAD` O_DIRECT arena load on Linux (PANCAKE_STATUS.md item 38), conversation disk tier `STRATA_CONVERSATION_DISK_DIR` (item 39; src/core/conversation_disk_tier.cpp, src/poly/conversation_disk.cpp), RAM-budget cache `STRATA_CONV_BUDGET`, PSI guard `STRATA_PSI_GUARD`.
- `mem_pressure`: include/strata/platform/mem_pressure.hpp:1-40: Linux PSI (`/proc/pressure/memory`); "Other systems than Linux: no PSI files, every call is a no-op".
- Prior decision on networks: "Aviary (SensAI-PT/aviary-hy3) - Multi-machine cluster on colibri; 40 layers of LAN round trips cost more than a whole token - No gain on one PC" (pancake-docs/RESEARCH_AND_FINDINGS.md:258); also "Multi-machine cluster on colibri, activations over the LAN. No gain on one PC; 40 layers of LAN round trips would cost more than a whole token takes now" (docs/PANCAKE_FINDINGS.md:109). Aviary listed "Distributed expert placement across nodes; surveyed, single-machine focus here" (README.md:123; pancake-docs/README.md:113). MOD_PLAN.md:1208, 1219, 1223, 1226 closes MegaScale-Infer, Vidur/LLMServingSim and Mooncake as cluster-only ideas.
- Multi-GPU docs already note: replicas need N copies of experts in RAM or a shared arena (`--shared-expert-arena`, memfd on Linux / named section on Windows; serve/replicas.py header).

### Inferences
- Remote experts are viable only for "bulk" effects: (a) layer-contiguous stage on another box (one hop per stage per window, not per layer), or (b) remote CPU experts for a few layers with deferral to hide the RTT. A per-layer remote expert call costs 1 RTT per layer per window; for a 48-layer model at 1 ms LAN RTT that is ~48 ms added per window, which matches the repo's own rejection.
- If experts were remote at all, the traffic would be activations: n_embd floats in (+ ids/weights), n_embd floats out per token per layer, i.e. 8 KiB each way per token per layer for Qwen3.6 (my arithmetic from n_embd=2048, fp32), not 1.38 MB blobs.

### Gaps
- I did not open tests/core/file_expert_source_test.cpp nor read `FileExpertSource` internals (expert_source.cpp is 7,067 lines).
- I could not find the MoE-layer composition of Qwen3.6 per layer in numbers (top-k per token, n_expert) without reading qwen35.cpp; PANCAKE_STATUS.md says top-10 routing for the Flash-Next model (`router ... top-10`), Qwen3.6 value unverified.
- No measured RTT/bandwidth data for any network in the repo.

---

## 5. Speculative decoding usable to hide network latency?

### Takeaway
Yes, in principle. The engine verifies a window of up to 16 tokens per pass (`DraftPolicy::kMaxT = 16`), so one network round trip per window can carry several tokens. Three drafter families exist: n-gram/suffix repeats ("lookup"), an MTP draft layer, and DFlash2. Hit rates are workload-dependent; the repo has the machinery for choosing the window by cost.

### Cited Findings
- Max window: `static constexpr int kMaxT = 16;` (include/strata/spec/draft_policy.hpp:31). `Options::window` = most tokens `window()` takes (engine.hpp:91; `max_window()` at :252).
- strata-poly drafts: `Session` combines repeats from the conversation (`strata::spec::SuffixDrafter repeats`, src/poly/main.cpp:331-345), an optional `--suffix-drafts` suffix index over all prompts/answers and corpus files (`SuffixIndex`, src/poly/suffix_index.hpp; main.cpp:335-338, 439-455) and the model's own MTP draft block (`engine.drafts()`, `engine.draft(from, i, &p)`; main.cpp:479-492; `Engine::draft_step`, engine.hpp:325). `DraftPolicy::choose(...)` picks window length from observed hits and time (src/spec/draft_policy.cpp; main.cpp:485). In DONE, drafts accepted/offered are reported (main.cpp:754).
- In the `strata` engine: MTP (src/core/mtp.cpp), DFlash2 (src/core/dflash2.cpp, include/strata/core/dflash2.hpp), ngram (src/kernels/ngram.cpp, include/strata/kernels/ngram.hpp), controller (src/spec/controller.cpp), `STRATA_MTP_EAGER`, draft stream on own queue (PANCAKE_STATUS.md items 74, P2). Draft layer is placed on the last card or first (`STRATA_MTP_STAGE0`) in a split (docs/MULTI_GPU.md "Pipeline windows").
- `--pipeline-windows 2`: the first card starts the next verify window under the guess that this one is accepted whole, rolling back if not; "The tokens are the serial loop's, up to CPU rounding" (docs/MULTI_GPU.md "Pipeline windows"). This is the exact mechanism that hides stage-to-stage latency in a layer split.
- Guess quality: README.md Table 2B reports 100.7 tok/s vs 64.6 tok/s for short chat (strata-poly vs strata on RTX 3060) on Qwen3.6-35B-A3B, and code edits 151 vs 40 tok/s, largely from drafting (README.md ~Table 2B; mechanism attribution is my reading).

### Inferences
- Over a LAN, the cost per window is `RTT + stage compute`, so the useful number is accepted tokens per window; code edits (high n-gram hit rate) benefit most, free prose least. A remote design should send whole verify windows (up to 16 x 8 KiB = 128 KiB fp32 per hop for Qwen3.6) and use `--pipeline-windows`-style speculation to overlap hops.
- The MTP draft block lives on the last stage (`draft_` only when `hi_ == n_layer_`, engine.hpp:405); a drafting round trip to a remote last stage would be expensive, so drafts from n-gram/suffix (first node) are the network-friendly source.

### Gaps
- No measured acceptance rates per drafter in this inspection (they are in bench/results and docs, not read).
- Whether a rolled-back window (`keep(n)`) works across a boundary that is remote: unverified (see section 3).

---

## 6. Which models/architectures are supported and what that limits

### Takeaway
`strata-poly` selects the architecture from `general.architecture`: `qwen35moe`, `gemma4`, `glm5-next`/`glm5next`, `deepseek4` (src/poly/engine.cpp:1053-1056). Several optimizations (KV quantization, attn_split, deferral/graph experts) are qwen35moe-only (engine.cpp:1040-1049). The older `strata` engine targets Qwen3.8-Flash-Next and Qwen3.6-35B-A3B; DeepSeek V4.1-Flash has its own engine (`strata-dsv41`).

### Cited Findings
- Arch dispatch: engine.cpp:1035-1056 (`std::make_unique<Glm5Next>() / Qwen35 / Gemma4 / DeepSeek4`), and a throw for others (not read past line 1056).
- Options restricted to qwen35moe: `kv_q8`, `kv_q4`, `attn_split`, `graph_experts` (engine.cpp:1040-1049).
- Docs model list: DeepSeek V4.1-Flash, Qwen3.8-Flash-Next (+ variants Coder, Swift, Gyro, Unsloth packs), Qwen3.6-35B-A3B (docs/MODELS.md:1-12). Flash-Next = 176.9B total (125B MoE + 51B PLE + 4B MTP), 48 layers, 512 experts (README.md Table 2A; docs/SPLIT_ENGINE.md "Both sides need all of this").
- Each architecture class is ~400-600 lines (qwen35.cpp 414, gemma4.cpp 608, glm5next.cpp 565, deepseek4.cpp 570), implementing the `Engine` virtuals (`load_layer`, `layer_head`, `layer_tail`, `route`, `streams`...). No dense (non-MoE) architecture is listed; dense llama-style models are not supported in this tree (I found no such architecture string).
- Model sizes that fit "consumer" nodes: Qwen3.6-35B-A3B UD-IQ2_XXS 11.01 GB (docs/WINDOWS_CUDA_SM75.md:~35); Flash-Next IQ2_XS 39.2 GB RAM+VRAM (docs/MODELS.md:~38); IQ3_S 54.8 GB; DeepSeek V4.1-Flash 46-64 GB RAM and 257 GB disk with a 12 GB GPU (docs/MODELS.md:~48).

### Inferences
- NetworkPancake is MoE-only by construction; a dense model (e.g. 70B) would need new architecture code, so the pitch "run any large model over a cluster" is not supported by this codebase. A split of a 100B+ MoE across 3-5 machines works only for these four families.

### Gaps
- Did not verify gemma4/glm5next completeness (quality/perf status) from docs.

---

## 7. Platform portability (POSIX vs Windows; Android/iOS)

### Takeaway
The C++ engine builds on Windows MSVC (documented on a GTX 1660 Ti, with one MSVC fix needed in kmiss_tier.hpp and some test targets broken) and has an explicit Windows branch in 82 files (`_WIN32`). Several important features are Linux-only (PSI memory pressure, O_DIRECT expert loading, split engine, shared arena via memfd, CPU pinning via Linux affinity). There is no Android/iOS code or build target; Android and iOS have no CUDA/HIP, so only the CPU kernels (x86 AVX2/AVX-512 plus unverified ARM shims) could run there, and nothing in the repo targets them.

### Cited Findings
- `_WIN32` appears in 82 files under src/ and include/ (grep count). POSIX-only header/syscall use found in: src/core/conversation_disk_tier.cpp, conversation_file.cpp, expert_source.cpp, layer.cpp, pinned.cu, verify.cpp, weights.cpp, src/dsv41/ds4_main.cpp and ds4_os.hpp (with ds4_os_apple.h), src/kernels/cpu/pool.cpp, src/kernels/ngram.cpp, src/platform/direct_file.cpp, memory.cpp, src/poly/conversation_disk.cpp, engine.cpp (grep for `unistd.h|sys/mman.h|mmap(|fork(|posix_|O_DIRECT`; most of these are likely guarded by `#ifdef _WIN32`, not individually checked).
- CPU affinity: `pool_affinity_linux.hpp` and `pool_affinity_win.hpp` both exist (src/kernels/cpu/), so thread pinning has both OS implementations.
- Direct I/O: src/platform/direct_file.cpp and include/strata/platform/direct_file.hpp; PANCAKE_STATUS.md item 38 "Linux O_DIRECT expert-arena load".
- Windows MSVC state on this fork: docs/WINDOWS_CUDA_SM75.md:19-27: `strata` fixed in this branch (`std::popcount` replaces `__builtin_popcount` in kmiss_tier.hpp); not fixed: `q5k_head_test` (missing `<string>`), `flag_order_test` (GCC `__atomic_*`), `gemm_w16_parity` (`setenv`), `poly_kv_q4_test` (range-for); these are test targets that "do not block the engine". "`main` does not build `strata` on MSVC" before that branch fix (docs/WINDOWS_CUDA_SM75.md:~58-60). Windows app/launcher exist (apps/desktop with Tauri, START-HERE.bat, SETUP.bat, serve/winjob.py for Job Objects, serve/keepawake.py).
- Windows WDDM: only 8 GiB of the expert arena pinned on a multi-GPU run (docs/MULTI_GPU.md); `src/core/pinned.cu`, `vmm.cpp` handle the driver virtual-memory calls on HIP vs CUDA (CMakeLists strata-poly block, `_strata_poly_driver`).
- Linux-only features: `strata-split` (sFNX): "Linux; CUDA or HIP gfx1201, 2-4 cards", exchange through page-locked host memory (docs/SPLIT_ENGINE.md; CMakeLists.txt:3353 `STRATA_BUILD_SPLIT`; code in src/split/ and third_party/sfnx); PSI guard (include/strata/platform/mem_pressure.hpp:~22); `--shared-expert-arena` memfd (Windows uses a named section, serve/replicas.py header); Docker (Dockerfile, docker/).
- Mac: Metal engine is a separate engine with its own host code (metal/), `src/dsv41/ds4_os_apple.h` exists. `TARGET_OS_IPHONE`/`__ANDROID__`/`android` grep over src, include, metal, sycl and CMakeLists.txt found nothing. The only Android hits in the tree are launcher icons under apps/desktop/src-tauri/icons/android (an icon asset set generated by Tauri), not engine code.
- Tauri desktop app (apps/desktop, Cargo.toml, `gen/schemas` with `macOS-schema.json`, `desktop-schema.json`) is a desktop shell, not mobile (I saw no mobile schema).

### Inferences
- A "Windows + CUDA sm_75 + Linux + Mac Metal" mesh is plausible at the engine level, but the Metal engine and the CUDA engine have different codebases (different Engine classes, state formats), so a mixed-backend pipeline needs the wire format to carry only the residual rows and not backend-specific state. Per-stage KV stays local, so that is possible, but there is no shared stage interface between `strata-poly` and the Metal engine.
- Phones/tablets would need a CPU-only (or Metal/other) stage built from the CPU expert kernels (ggml-cpu `vec_dot` based in strata-poly) plus attention on CPU, which does not exist as a standalone path in this repo (strata-poly requires CUDA/HIP at build: `cuda_runtime.h` in runtime.hpp, and the strata-poly target links CUDA/HIP libs).

### Gaps
- Did not check each POSIX-only file for `#ifdef` guards, so "POSIX-only" is an upper bound.
- Did not verify NEON code quality; hits could be only `__builtin_prefetch`/yield shims.
- Did not check whether CMake supports a CPU-only (no CUDA/HIP) build of strata-poly; the strata-poly link line shows `${_strata_gpu_runtime_target}` unconditionally inside a block I did not trace to its condition.

---

## 8. Existing plan docs on multi-node / network / RPC / distributed / cluster

### Takeaway
No plan item in this repo covers networked inference; every mention is a "surveyed, rejected/closed" row. The only relevant prior art evaluated is Aviary (cluster of machines over LAN), dismissed on latency grounds. The repo's docs describe multi-GPU on one PC (layer split, split engine, replicas, split-serve), which are the closest working designs.

### Cited Findings
- MOD_PLAN.md:834 research row "Runtime architectures 2026 (disaggregated serving, control/execution plane, heterogeneous runtimes ... single-node applicability)"; MOD_PLAN.md:1208 MegaScale-Infer/Comet/UltraMem "cluster, large batch. [closed]"; :1219 Vidur, LLMServingSim "cluster simulators; only their validation method is copied (item 42). [closed]"; :1223 "Chunked-prefill interleave, overlap scheduler ... distributed KV stores ... [parked: several users/agents at once]"; :1226 Mooncake "cluster KV store; only the NVMe-tier idea (phase 7 / item 39). [closed]".
- pancake-docs/RESEARCH_AND_FINDINGS.md:258 and docs/PANCAKE_FINDINGS.md:109 (Aviary, quoted in section 4). pancake-docs/RESEARCH_AND_FINDINGS.md:548-560 repeat the closed rows.
- Reports/research folders that may contain more: reports/LLM runtime architectures 2026.md, research_notes/LLM runtime architectures 2026/, pancake-docs/STRATA_DERIVATIVES_DEEP_RESEARCH.md, research_notes/Active Strata forks November/batch1.md, batch5.md, batch6.md, data/fork-kb/readmes/*.md (11 files matched the broad keyword grep). I did not open them.
- A broad keyword grep (`multi-node|distributed|cluster|RPC|over the network|network split|pipeline.parallel`) matched: bench/results/*community*/README.md (community multi-GPU runs), docs/MULTI_GPU.md, docs/INTEL.md, docs/NVFP4.md, docs/A770_SPEED_SPEC.md, apps/host/docs/dsh-recon.md (these look like ordinary uses of the words), data/fork-kb/readmes/*.md (11 forks' READMEs).
- A directory for this very plan already exists: research_notes/Plan NetworkPancake inferență distribuită/ (created by the coordinator; this file is its first content).
- Multi-GPU on one PC, with measured numbers: docs/MULTI_GPU.md (layer split; 2-4 cards; hand-off through pinned RAM; "pipeline windows"), docs/SPLIT_ENGINE.md (sFNX all-cards split: 2x RTX A6000 UD-Q4_K_XL code edit 195 -> 377-394 tok/s vs layer split; Linux, same-model cards, all experts fit VRAM), docs/SPLIT_SERVE.md, docs/SECOND_GPU.md, docs/AMD_HIP.md (AMD splits, e.g. bench/results/2026-10-09-community-4x-rx-7900-xt).
- The `strata` layer split lays down this resource model per stage: dense weights (trimmed per stage with `STRATA_STAGE_TRIM`), KV, expert cache, plus shared RAM arena of experts (docs/MULTI_GPU.md "What each card holds").

### Inferences
- NetworkPancake would be the first plan item of its kind in this repo, and it will have to argue against the repo's own Aviary conclusion by changing the cost structure (one hop per stage per window, speculative windows, 2-3 stages), not per-layer hops.

### Gaps
- Contents of the unopened research folders and fork-kb READMEs not reviewed.
- I did not open README.md sections 3-10 or docs/AI_REFERENCE.md / HOW_IT_WORKS.md / DETAILS.md in full.

---

## 9. Component map and recommendations (concrete)

### Takeaway
The smallest viable network split is: keep one HTTP front (serve/server.py) and strata-poly's per-card layer pipeline, and turn the in-process `next_` stage hand-off (pinned host rows) into a TCP link carrying `pos, n, residual rows` to a second `strata-poly` process that owns layers `[K, n_layer)` plus the output head and drafter. Two or three stages, whole verify windows per hop, n-gram/suffix drafts on the first stage. Everything else (experts, KV, GGUF reading, conversation cache, HTTP API) is reusable as-is per node.

### Component -> status table

| Component | File:line | Status | Note |
| --- | --- | --- | --- |
| OpenAI/Anthropic HTTP server, streaming, API keys, host binding | serve/server.py:9894, 10860-10925; docs/DETAILS.md:640 | reusable as-is | one process, one sequence at a time |
| Engine line protocol GEN/T/DONE/QUIT/STOP/VRAM | src/poly/main.cpp:4-7, 628-760 | reusable as-is (client side), needs change if remote engine must be one process | stdout/stdin only; no framing for binary |
| Python multi-engine wrappers (Engine protocol) | serve/replicas.py, serve/split.py, server.py:420 | needs change | good hook for remote engines; request-level only |
| Layer-range engine per card (`lo_`, `hi_`, `next_`) | src/poly/engine.hpp:541-549; engine.cpp:1714, 3271-3278 | needs change | in-process, CUDA device ids; remote needs `RemoteStage : Engine` |
| Residual hand-off `hand_on` / `pass_rows` / `pass_` | src/poly/engine.cpp:3249-3269 | needs change | cut point; rows are `n*streams*n_embd` fp32; replace pinned host copy by socket send/recv |
| Last-stage logits delivery (`logits`, `logits_of`) | engine.hpp:216, 254; engine.cpp:3280-3300 | needs change | full-vocab float logits per window row; send top-k or move sampler |
| Per-stage KV and recurrent state; `save/load_state`, `save/load_rows` | engine.hpp:279-288, 327, 383; conversations.hpp:1-17 | reusable as-is per node | each stage keeps its own; checkpoints/parking work per chain |
| Layer-split auto placement by bytes | src/poly/engine.cpp:263-320 | needs change | uses local VRAM; needs per-node capacity input |
| Expert cache/profile/adaptive swaps | engine.hpp (Options), expert_source.hpp, PANCAKE_STATUS.md | reusable as-is per node | each node tunes its own layers |
| CPU experts (ggml-cpu vec_dot, AVX2 multi) | src/poly/cpu_experts.hpp/.cpp, cpu_kernels_avx2.cpp | reusable as-is | good CPU-only-stage candidate |
| `ExpertSource` over network (weights) | include/strata/core/expert_source.hpp:194-304 | missing and not recommended | 1.38 MB per Flash-Next expert; rejected in docs/PANCAKE_FINDINGS.md:109 |
| Remote expert compute RPC (x in, rows out) | `cpu_experts()` signature; remote_experts.hpp:36-50 | missing | per-layer RTT; only with deferral |
| Network transport (sockets, framing, handshake, auth, reconnect) | none | missing | no C++ network code in tree |
| Speculative decoding (n-gram/suffix, MTP, DFlash2) | src/poly/main.cpp:331-345, 439-492; draft_policy.hpp:31; src/core/mtp.cpp | reusable as-is | window up to 16 tokens; MTP sits on the last stage |
| Pipeline windows (speculative next window) | docs/MULTI_GPU.md "Pipeline windows" (older engine) | needs change (poly) / reusable (strata) | not seen in strata-poly engine; hides hop latency |
| Model families | engine.cpp:1035-1056 | limits scope | qwen35moe, gemma4, glm5next, deepseek4 only; MoE only |
| CUDA backend (sm_75+; sm_52/60/61 experimental) | CMakeLists.txt:260-310; docs/OLDER_GPUS.md | reusable as-is | min supported sm_75; GTX 10xx experimental via CUDA 12 |
| HIP backend (gfx1100 tuned; others) | CMakeLists.txt:95-102; include/strata/hip_compat/ | reusable as-is | AMD nodes work with same engine code |
| Metal engine | metal/ | needs change | different codebase, own protocol; mixed-backend stage format missing |
| SYCL (Intel Arc) | sycl/ | not for v1 | experimental |
| Vulkan / OpenCL / CPU-only engine | none | missing | strata-poly links CUDA/HIP unconditionally (unverified) |
| Android / iOS | none | missing | no code or targets |
| Windows MSVC build | docs/WINDOWS_CUDA_SM75.md:19-27 | reusable with caveats | engine builds; some tests do not |
| Linux-only features | mem_pressure.hpp, direct_file, split engine, memfd arena | not required | skip on Windows nodes |
| Conversation cache, disk tier | src/poly/conversations.hpp, conversation_disk.cpp | reusable as-is per node | helps prefix reuse; chain-level save already sums stages |
| Plan docs for multi-node | MOD_PLAN.md:1208-1226; PANCAKE_FINDINGS.md:109 | missing (only rejections) | must be written from scratch |

### Recommendations (smallest viable design in this codebase)

1. Do request-level distribution first (cheap, Python only): extend serve/replicas.py or serve/split.py with a `RemoteEngine` whose pipe is a TCP stream to a remote `strata-poly --serve` (or to a tiny launcher that execs it and relays its stdin/stdout). Reuses protocol GEN/T/DONE unchanged (main.cpp:628-760), prefix-aware routing (replicas.py) and API keys. This does not split one model, but provides a networked fleet.
2. For an actual model split, build a `RemoteStage : Engine` in src/poly (new file) that implements the `next_` chain surface: `eval/read/window/keep/reset/draft`, `save_state/load_state/save_rows/load_rows`, `logits/logits_of`, stats accessors (list in section 3). Sender side serializes `pos, n, rows` from `hand_on()` (engine.cpp:3263); receiver is a second `strata-poly --stage-server --layers K..N` process that sets `pass_`/`pass_at_` (engine.cpp:3263-3269) and runs `eval()`/`read()` as the chain does now. Keep two or three stages only.
3. Transfer format: fp32 rows as the engine already has them (bit-identical to the in-process split), with an opt-in bf16/fp16 mode that is documented as changing output (the repo is strict about "exact" vs "lossy", PANCAKE_STATUS.md). Qwen3.6: 8 KiB/token; Flash-Next: 10 KiB/token (n_embd 2560); GLM/DeepSeek: 64 KiB/token (kHc=4).
4. Hide latency with windows: send whole verify windows (<=16 tokens, draft_policy.hpp:31); add a poly equivalent of `--pipeline-windows` (speculate the next window's first stage while the remote stage verifies) and rely on the existing n-gram/suffix drafters on stage 0; keep the MTP draft block on the last stage only if its RTT can be amortized.
5. Return the minimum from the last stage: either top-k logits plus the sampled token (needs the sampler to move, not currently done) or full `n_vocab` float rows per window row (e.g. a 16-token window x 152k x 4 B is about 9.7 MB, too heavy; verify the actual vocab with the GGUF first).
6. Per-node config: use each node's own `--vram-reserve-mib`, expert profile, conversation cache; make `layer_starts()` (engine.cpp:263-320) accept per-node capacities instead of local CUDA devices.
7. Do not build a network `ExpertSource` for weights; if remote experts are explored at all, make it an activation RPC around `cpu_experts()` with Expert Deferral to hide RTT, and treat it as phase 2.
8. Windows participation: base on the sm_75 MSVC recipe (docs/WINDOWS_CUDA_SM75.md). For sockets use Winsock/POSIX behind one small wrapper; no existing wrapper to reuse. Mobile is out of scope for this codebase.
9. Expect to argue against the repo's own conclusion (Aviary: LAN round trips per layer cost more than a token): the design above has 1 hop per stage per window, not 1 per layer, which is the difference.

### Gaps
- I did not benchmark anything, so no latency/throughput claims for a network split are made; all hop sizes above are arithmetic from tensor shapes, not measurements.
- The `strata` (generate.cpp) and `strata-poly` engines differ; recommendations 2-6 target strata-poly. Whether the `strata` engine's layer split (generate.cpp) would be easier to extend was not assessed, only that it exists and uses pinned host memory (docs/MULTI_GPU.md).
