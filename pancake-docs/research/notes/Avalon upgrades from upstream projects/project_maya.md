# Project Maya (github.com/mw00/project-maya) compared with Avalon, as of 2026-11-09

Method: `gh api` and `gh search issues` against mw00/project-maya. I shallow-cloned main @7bf15ac (2026-11-07, VERSION 1.1.56, 1,074 commits) into the scratchpad for reading only. Nothing was built or run, and no models were downloaded. I listed peasantsmith's models through the HF API. For Avalon and Strata I used the earlier notes in this folder (avalon_baseline.md, strata.md, llamacpp_server_features.md) plus a few greps of the local Avalon checkout (c24c03d). Repo facts: MIT, 2,710 stars, 382 forks, 140 open issues, created 2026-10-06, pushed 2026-11-08. It is not a GitHub fork (`fork: false`). Short links: R = [README](https://github.com/mw00/project-maya/blob/main/README.md), S = [docs/SETTINGS.md](https://github.com/mw00/project-maya/blob/main/docs/SETTINGS.md), C = [CHANGELOG.md](https://github.com/mw00/project-maya/blob/main/CHANGELOG.md).

**Bottom line (read first):**
- **Maya is not a llama.cpp fork.** It is a renamed and extended fork of **Strata** (Niko1221/Strata): a native C++/CUDA/HIP engine with a Python server.
- **Its Qwen-3.8-Flash-Next path is a stale snapshot of Strata's engine (0.1.27).** Users measured it 1.4-1.7x slower in decode than current Strata.
- **Current Strata already loses to Avalon on RDNA3.**
- So running Maya with Qwen3.8-Flash-Next (or GLM-5.3-Flash Maya-XS) on our box will not beat Avalon.
- What is worth porting is mostly on the server/agent side: the loop guard, a soft thinking budget, and tolerance of malformed tool calls. On the engine side, a couple of ideas need measuring first: direct parallel expert reads for experts that RAM cannot hold, and expert skipping by weight share instead of rank.

## Q1. What is Maya's engine built on, and how do its expert placement, movement and auto-tuning work?

### Takeaway
Maya is a fork of **Strata**. It is not based on llama.cpp, buun-llama-cpp or ik_llama.cpp. It uses **ggml only as a pinned library**, for the quant formats, the CPU dot products and the prompt path's MMQ kernels. Each model gets its own native engine: `maya` covers GLM-5.3-Flash and Qwen, `maya-ds4` covers DeepSeek-V4.1 and `maya-mm3` covers MiniMax-M3.1. All of them share Strata's three tiers:
- **VRAM:** an expert cache of the most-used experts, ranked by an expert profile, adaptive, with background promotion.
- **Pinned RAM:** a tier sized from the free RAM. The CPU computes missed experts in place, or sends a measured share over PCIe.
- **SSD:** experts read through mmap or direct I/O.

Auto-tuning works at two levels. At start the engine times one expert on the CPU against the PCIe link and plans the split from that. `--calibrate` then runs 10-15 minutes of real decode and saves the best PCIe share, CPU thread count and SSD read size.

### Cited Findings
**Lineage and build**
- LICENSE reads "Copyright (c) 2026 Niko1221 and the Strata contributors". The CMake project is `project(strata VERSION 0.1.27 ...)`, headers live in `include/strata/`, and environment variables are `STRATA_*` with `MAYA_*` aliases ("Their earlier `STRATA_*` spellings still work"). — [LICENSE](https://github.com/mw00/project-maya/blob/main/LICENSE), [CMakeLists.txt](https://github.com/mw00/project-maya/blob/main/CMakeLists.txt), S
- GLM5-FLASH.md: "The engine Maya started from is not a general inference engine: it is a native implementation of one architecture - Qwen3.8-Flash-Next ('qwen4exp') - with its memory tiers (VRAM expert cache / pinned RAM / mapped SSD)". About two thirds of that machinery was reused for GLM. — [docs/GLM5-FLASH.md](https://github.com/mw00/project-maya/blob/main/docs/GLM5-FLASH.md)
- README credits: "Built on ggml / llama.cpp ... the quantization formats, the CPU dot products and the prompt path's MMQ kernels, built from a pinned commit". The pin is `third_party/ggml/VERSION.txt` = `3cf03257f219` (2026-09-20). EXL3 support follows exllamav3, and the DFlash2 drafter's forward follows Anbeeld's beellama.cpp. — R, [third_party/ggml/VERSION.txt](https://github.com/mw00/project-maya/blob/main/third_party/ggml/VERSION.txt)
- A grep of the whole repo finds no mention of buun, ik_llama, Avalon, freriadoc or thecodacus. The only shared ancestor with Avalon is upstream ggml, plus the ideas Avalon credits to Strata (the ranking format and the preload idea). — local `grep -rli` over the clone; [Avalon README credits](https://github.com/freriadoc/Avalon/blob/main/README.md)
- Authors are Peasant Smith (@mw00, owner) and @needmorevram, who contributed DFlash2, EXL3, KV streaming, checkpoints, layer-split search, the expert profile and the CPU expert kernels. — R "Credits"
- The server is Python: `serve/server.py` (7,212 lines), `loop_guard.py`, `responses.py`, `frontend.py`, `mcp.py` and others, about 32.6k lines in `serve/`. — local `wc -l serve/*.py`

**Tiers and movement**
- README: "keeps the experts your conversation uses most on the GPU, the next ones in RAM and the rest on your NVMe SSD, and moves them as you chat. It measures your GPUs, CPU, RAM and SSD and tunes itself." The engine "pins most of the free RAM (all but about 6 GB)". — R
- At start the engine fills each GPU's free VRAM with experts and sizes the RAM tier from free RAM. It "measures the CPU against the PCIe link at start to decide how many RAM experts the CPU computes itself". `--expert-cache auto` takes free VRAM minus `--vram-reserve-mib` (700 MiB by default). — S "Tuning and settings", "The expert pool's size"
- The CPU lane and the PCIe path round differently (activation blocks of 32 vs 256), so greedy output is not repeatable across starts unless the pool, the split, background promotion (`MAYA_GLM_PROMOTE`) and retiming (`MAYA_GLM_CPU_RETIME`) are fixed. `MAYA_GLM_CPU_LANE=0` makes output repeatable at a cost of 4-10% decode. — S "Repeatable answers"

**Qwen path specifically**
- The most-used experts go to VRAM. When RAM cannot hold the rest, the engine reads them straight from the GGUF: the next-most-used are copied into RAM ("as many as the available RAM allows less 12% of it (at least 6 GB)"), and the remainder are read from the file when routed. "The 52 GB n-gram table is always read from the SSD row by row and never enters RAM." `MAYA_EXPERT_STAGERS` sets the number of reader threads (2/4/8/16), probed once per drive in about 1 s and cached in `~/.config/project-maya/drive-reads.txt`. — S "Qwen-3.8-Flash-Next (CYBER-FROST)"
- C v1.1.38 (#654): "Qwen CYBER-FROST on RAM-short PCs: decode +26%". On Linux, experts that neither VRAM nor RAM holds "are now read straight from the model file, without the system's file cache"; `MAYA_EXPERT_DIRECT=0` turns this off. Measured on a V100 32 GB with 30 GB RAM. — C

**Self-tuning**
- `./maya.sh --calibrate` runs one engine session of about 10-15 minutes. It tries several splits of RAM-tier experts between the CPU and PCIe, fewer CPU threads ("more threads than the memory can feed only wait"), and smaller disk reads when the SSD is involved. A setting is kept only if it is more than 3% faster. Results go into the config's `env` (`MAYA_GLM_PCIE_SHARE`, `MAYA_GLM_CPU_LANE`, `MAYA_GLM_READ_CHUNKS`) and into `~/.config/project-maya/calibration.json`. It uses fresh prompts so the expert tiers are not overfitted. — S "Tuning for your PC"
- A separate speed calibration loads the model twice (with and without the drafter). It reads an 8K-token document, answers sampled questions until about 1,500 tokens are written, and keeps the drafter only if it is more than 5% faster. For Qwen, "The setup measures whether [MTP] is faster on this PC ... and keeps it only then". — S "The speed calibration", "Qwen ... Drafting"

**Drafters, expert skipping and Turbo**
- **DFlash2** is a GLM-only block drafter (Anbeeld's, CC BY-NC-ND 4.0). It reads hidden states at 5 layers and drafts up to 7 tokens per pass, replacing the MTP block. Measured: +44% on 9 GPUs, +19% on one RTX 3090, +9% on an 8060S. It runs only with `--parallel 1`, and on AMD only on Strix/Gorgon Halo. — S "Drafting with DFlash2"
- **DSpark** is DeepSeek-V4.1's and MiniMax's drafter. It appears in benchmark rows, but there is no Qwen equivalent. — R benchmark tables
- **Qwen drafting** uses the MTP block (built from the GGUF's NextN block by `tools/mtp_from_gguf.py`): "up to three tokens a window", `--spec 4` by default, plus prompt-lookup drafts when `--mtp-max-t 1`. — S Qwen section
- **Expert skipping** has two forms:
  - `MAYA_GLM_MISS_SKIP_W` / `_FROM` (GLM, opt-in, changes answers): a routed expert that is *not in VRAM* and carries less than share W of the route's weight is skipped, on layers from `_FROM` on. On an RTX 3090 with Maya-S24 and `_FROM=23`, chat speed went 27.4 → 29.3 / 30.1 / 30.9 t/s at W = 0.12 / 0.14 / 0.16, with KL 0.037 / 0.052 / 0.075. It can be set per request through `strata_tune`.
  - Strata's rank-based `STRATA_ROUTE_TAIL_SKIP=7` is the older form of the same idea.

  — S (MISS_SKIP row); [strata.md](file:///home/dan/research_notes/Avalon%20upgrades%20from%20upstream%20projects/strata.md)
- **Turbo** is one button that "sets the fastest settings for the model (greedy sampling, no thinking, expert skipping, the fastest drafting)", and switching it off restores the previous settings exactly. It is implemented in `serve/server.py` (`turbo_info` / `set_turbo`) and `models.turbo_enable`. — R "Using it", [serve/server.py](https://github.com/mw00/project-maya/blob/main/serve/server.py)

**Several requests at once**
- `--parallel N` (1-8) works for GLM, DeepSeek and MiniMax, not Qwen. On one GPU, one token of each request goes through the model together, so dense weights are read once. Results: V100 x1.41 for 2 requests and x1.35 for 4; with every expert in memory (Strix Halo), x1.67 and x2.02. Each extra request takes its KV and state from the expert pool, and DFlash2 is off when N > 1. — S "Several requests at once"

### Inferences
- Maya's Qwen path is Strata's design, so anything Maya does for Qwen we have already seen in Strata (see strata.md), only older. The truly Maya-only work is in the GLM/DeepSeek/MiniMax engines and the Python server.
- Maya's placement model matches Avalon's "hosted MoE": a GPU cache, the CPU computing misses, and a share of misses sent over PCIe. Maya adds an **explicit SSD tier with direct, parallel reads**, which Avalon (llama.cpp mmap and page cache) does not have. That tier only matters when VRAM plus RAM cannot hold the model, which is our case: 89 GiB IQ3_KS against about 57 GB RAM plus a ~14 GB cache.

### Gaps
- I did not read the Qwen engine's C++ (`src/core/*`) in depth. The above relies on Maya's docs, Strata's docs and the issue measurements.
- The commit history before the rename is not visible in the shallow clone. The exact Strata commit Maya forked from is stated only in #604 ("last synced from Strata 0.1.41, v1.0.13").

## Q2. Maya benchmarks on RX 7900 XT/XTX (gfx1100), and user reports for Qwen-3.8-Flash-Next under Maya

### Takeaway
No benchmark of Qwen-3.8-Flash-Next on Maya on an RX 7900 XT or XTX exists. The Qwen reports that do exist are negative:
- Maya's Qwen engine decodes 1.7x slower than Strata 0.2.7 on the same L40S and weights (78.9 vs 134.7 t/s).
- On AMD gfx1200 it is about 36% slower than Strata (21.3 vs 28.6-29.9 t/s).
- One user saw an order of magnitude slower prefill at 57K tokens (126 vs ~1,900 t/s).

The maintainer declared Qwen unsupported on 2026-10-13, then re-added it (CYBER-FROST at the end of October, UD-IQ2_XS in v1.1.54 on 2026-11-05) without updating the engine. On gfx1100 the only numbers are for GLM and DeepSeek. One RX 7900 XT runs GLM Maya-S at about 15 t/s decode and about 414 t/s prefill.

### Cited Findings
**Qwen-3.8-Flash-Next on Maya**
- **Issue #604 (open, 2026-10-28)**, "Qwen3.8-Flash-Next (CYBER-FROST) decodes ~40% slower on Maya than on Strata 0.2.7 with the same weights: the Qwen engine is still 0.1.27". Test setup: L40S 48 GB with the same Q5_K_M weights. Maya v1.1.34 averaged 78.9 t/s decode (69-84) against Strata 0.2.7 at 134.7 t/s. Draft acceptance, cache hit rate (92.8-94.6% vs 88.4-93.7%) and expert lookups per token were similar, "so the gap is in how fast the engine runs each step". Maya's prefill was *faster* at long prompts: 3,774-3,788 vs 2,742 t/s at 190K. Ruled out as causes: the PLE table reads, the KV format, the expert cache and the draft. — [issue #604](https://github.com/mw00/project-maya/issues/604)
- A comment on #604 (2026-11-08) reproduced the gap on **AMD gfx1200** (RX 9060 16 GB): Maya with q4_0 and MTP-2, tuned, gave 21.3 t/s; base Strata (146df3c4) on the same GGUF gave 28.6-29.9 t/s, about +36%. — [issue #604 comments](https://github.com/mw00/project-maya/issues/604)
- **Issue #898 (open, 2026-11-06)**: RTX 3090 plus 5060 Ti, 32 GB RAM, UD-IQ2_XS, Maya v1.1.55. At about 57-58K prompt tokens: prefill about 126 t/s, decode about 22 t/s. On the same box Strata gave 1,900-2,000 t/s prefill and 72-95 t/s decode at 128-154K. — [issue #898](https://github.com/mw00/project-maya/issues/898)
- **Issue #151 (closed 2026-10-13)**: unsloth UD-IQ3_XXS on gfx1100 HIP gave degenerate output with prompt-independent "eloselo..." text. The maintainer replied: "this is the Qwen3.8-Flash-Next path, which Maya doesn't support. Maya runs GLM-5.3-Flash only" and removed the Qwen/Orca docs (a709524). — [issue #151](https://github.com/mw00/project-maya/issues/151)
- **Qwen support came back later:**
  - CYBER-FROST Q5_K_M (112.5 GB, one file) gained pictures in v1.1.42 (2026-10-30), +26% decode on RAM-short PCs in v1.1.38, and fixes for 8 GB cards in v1.1.49.
  - v1.1.54 (2026-11-05) added "Qwen-3.8-Flash-Next (the base model), unsloth's UD-IQ2_XS GGUF ... NVIDIA from the V100 on, and AMD", plus the n-gram table from Maya's HF repo. "unsloth's other UD quants of it run on the same code but are not listed."
  - The Qwen fixes since then are server-side only: #703 (abandoned prompt read), #706 (unclosed `<parameter=` tag), and a follow-up message that re-read the whole conversation (fixed in v1.1.54).

  — C
- Memory guidance (docs/MEMORY.md): Qwen UD-IQ2_XS is 80 GB on disk, needs 47 GB of memory and 64 GB recommended (VRAM plus RAM). CYBER-FROST Q5_K_M is 113 GB on disk, 84 GB needed and 96 GB recommended. The n-gram table stays on SSD. "24 GB GPU + 64 GB RAM: fits fully GLM Maya-XS, Qwen UD-IQ2_XS". — [docs/MEMORY.md](https://github.com/mw00/project-maya/blob/main/docs/MEMORY.md)
- Qwen defaults in S: KV `int8` (about 14 KB per token over the 12 attention layers), context 32K-256K, sampling temperature 1.0 / top-p 0.95 / top-k 20, a thinking budget of 32,768, and the GGUF's own template verbatim, which always thinks and always adds a fixed system prompt. Reasoning effort maps to `xhigh` (default), `medium`, `low` and `none` (closes `<think>` at once). — S Qwen section
- Qwen gets no `--parallel` (that is a GLM, DeepSeek and MiniMax feature), and no DFlash2 or DSpark (it uses its MTP block). — S

**RDNA3 / gfx1100 numbers in Maya (other models)**
- AMD_MAYA.md: one **RX 7900 XT (20 GB)** running GLM-5.3 Maya-S at 8K context with a 90 GB RAM tier (192 GB RAM), v1.0.11, greedy: "~15 tok/s" answers and "~414 tok/s" prompts at 4K. R9700 plus 7900 XT: about 33 t/s. — [docs/AMD_MAYA.md](https://github.com/mw00/project-maya/blob/main/docs/AMD_MAYA.md)
- Prompt speed for GLM Maya-S-v2 on an RX 7900 XT (ROCm 7.2): chunk 256 gave about 50 t/s, engine-sized chunk about 198, plus tuned hipBLASLt about 253-259, plus `PREFILL_SUB=1024` / `PREFILL_MB=4096` about 413. — docs/AMD_MAYA.md
- README benchmark rows involving gfx1100:
  - RX 7900 XTX with a Radeon 8060S helper, GLM Maya-S24: 27.5 t/s decode and 836 t/s prefill at 114K.
  - 2x 7900 XTX with the 8060S, GLM Maya-M: 56.0 t/s.
  - 1x RX 7900 XTX (ROCm 10.2), DeepSeek Maya-XS with DSpark: 18.4 t/s decode, 395 t/s prefill.
  - 4x RX 7900 XT with `--tp 4`, DeepSeek UD-Q4_K_XL: 35.0 t/s.

  — R "How fast is it?", [issue #757](https://github.com/mw00/project-maya/issues/757)
- Supported AMD hardware: "RX 7900 XT / XTX, Radeon AI PRO R9700 / RX 9070, Strix Halo and Gorgon Halo". ROCm 7 is required; ROCm 6 is refused unless `MAYA_HIP_ALLOW_ROCM6=1`. RDNA3-specific kernels exist for DeepSeek (`DS4_INDEX_TC`, WMMA indexer: +3-10% prefill). — R "What you need", S

**For comparison: Strata and Avalon on RDNA3 (from earlier notes)**
- Strata IQ2_XS on an RX 7900 XTX (engine 0.1.57): prompt 1,051 / 1,385 / 1,381 t/s at 4K / 16K / 32K, decode 80.3 t/s. Avalon on our 7900 XT: about 100 t/s decode and about 1,800 t/s prefill at 16-32K. A user's 7900 XTX on Avalon main: 112.9-122.8 t/s decode and 2,024-2,130 t/s prefill. — [strata.md](file:///home/dan/research_notes/Avalon%20upgrades%20from%20upstream%20projects/strata.md), [avalon_baseline.md](file:///home/dan/research_notes/Avalon%20upgrades%20from%20upstream%20projects/avalon_baseline.md)

### Inferences
- Chain of evidence: Avalon (about 100 t/s on a 7900 XT) is faster than current Strata (about 80 t/s on a larger 7900 XTX), which is 1.36-1.7x faster than Maya's Qwen engine. My rough estimate (an inference, not a measurement) is that Maya with Qwen UD-IQ2_XS on our 7900 XT would decode around **45-60 t/s** with prefill well below 1,800 t/s, and possibly much worse at long context per #898. That is also with a *lower-bit* quant (IQ2_XS against our IQ3_KS).
- GLM-5.3-Flash Maya-XS on our box would sit near the about 15 t/s decode and about 400 t/s prefill that one 7900 XT measured with 192 GB RAM, or likely lower. Our 57 GB RAM plus 20 GB VRAM (77 GB) is below the 88 GB "fits fully" class for Maya-XS (73.4 GB) at 128K, so some experts would come from the SSD. That is 6-7x slower decode than Avalon. GLM-5.3 (321B / 18B active) might be a smarter model, but at that speed it is impractical for an opencode loop with 131-262K prompts.
- Disk: UD-IQ2_XS (80 GB) is larger than the 78 GB free on D, and CYBER-FROST (112.5 GB) does not fit at all. Trying Maya-Qwen would compete with re-downloading Avalon's 89 GiB model for the same space.

### Gaps
- No direct Maya measurement exists on a single 7900 XT with Qwen or with GLM Maya-XS and less than 64 GB RAM. The numbers above are extrapolations.
- The maintainers had not replied with a plan to resync the Qwen engine on #604 or #898 as of 2026-11-09 (none found).

## Q3. Which agent and server features does Maya have that Avalon lacks?

### Takeaway
Avalon's llama-server already has the OpenAI chat API, `/v1/responses`, `/v1/messages` (Anthropic), streamed tool calls, `return_progress`, `--reasoning-budget`, `--cache-ram`, ctx checkpoints and slot save. Maya's real extras are agent-robustness features in its Python server:
1. **Loop guard.** It catches periodic token loops in the thinking and the answer. It also notices when an agent repeats the same tool call with the same result: it adds a note, blocks the call after 3 notes, collapses repeated text in the prompt, and nudges a turn that ends with nothing new.
2. **Soft thinking budget.** It closes the thinking at the next sentence or line end, not mid-token.
3. **Tolerant tool-call parsing.** A malformed call no longer fails the request.
4. **Turn-end markers.** A marker written inside the thinking is treated as text.
5. **Smaller items:** an incremental "prompt memo" for re-sent conversations, a Turbo preset, a cost-savings counter, MCP tools for the web chat, an Ollama-style model API, and per-request `strata_tune` knobs.

### Cited Findings
- **Loop guard (on by default).** Token level: a block of 8-2,048 tokens repeated back to back (4 times for blocks under 32 tokens, 3 times otherwise) over at least 96 tokens counts as a loop. The thinking needs 5 or 4 copies over at least 192 tokens. The response escalates:
  - first loop: the unsent repeats are dropped and a temporary repetition penalty of 1.15 is applied over 2 x the period (at least 128 tokens);
  - 3rd loop in the thinking: the thinking is closed with the model's own end-of-thinking token;
  - 2nd loop in the answer, or a 4th in all: the answer stops.

  The stream holds back at most 64 tokens, and the check costs "a few microseconds a token". — S "The loop guard", [serve/loop_guard.py](https://github.com/mw00/project-maya/blob/main/serve/loop_guard.py)
- **Loop guard across turns.** If the last tool call (same name, arguments and result) is the 3rd of its kind among the last 12 calls since the user's message, a server note follows the result, worded differently at the 3rd, 4th and 5th time. After that the call is *blocked* ("not run"), and the temperature rises by 0.15 per repeat up to 1.5. Results that differ only in numbers or garbled bytes count as the same, except `sleep` waits. "Collapse" removes sentences or tool rounds that already appeared 3 times from the prompt (in a deterministic way, so the cache prefix still holds; "GLM-5.3 had one sentence 543 times in an 88K prompt"). "Kept going": a turn that ends with nothing new gets a short "do not stop here" user note, at most 3 times. Everything is counted in `/metrics` and Prometheus. The rationale cited is "GLM-5.3 ran two commands in turn 25 times over opencode". — S, serve/loop_guard.py docstring
- A related fix: the loop-guard retry overflowed the context, producing stubs that said "reply was cut short" (#727, closed). — [issue #727](https://github.com/mw00/project-maya/issues/727)
- **Thinking budget.** `"thinking_budget"` defaults to 32,768 tokens (8,192 for GLM Maya-XS). "Past it the server closes the thinking at the next sentence or line end and the answer starts". It can be changed without a restart. v1.1.48 notes "no more re-reading after the thinking budget". — S "The thinking level and the thinking budget", C v1.1.48
- **Tool calls.** "Tool calls are streamed as the model writes them, and a long prompt's reading progress too ... (`"return_progress": true`)". Since v1.1.39 (#613) an unparseable tool call no longer fails the request: "the answer goes on, and the log shows the whole call". #706 extended this to Qwen's form with an unclosed `<parameter=` tag (fixed in v1.1.46). #403 stopped long string arguments being held until the closing tag. — R, [issue #706](https://github.com/mw00/project-maya/issues/706), [issue #403](https://github.com/mw00/project-maya/issues/403)
- **Turn-end marker inside the thinking.** A turn-end marker the model writes in its thinking is treated as text, so the thinking goes on. At the engine's own EOS token in the thinking, generation continues with a short server note (at most twice). — S "The loop guard" (last bullet)
- **APIs.**
  - `/v1/responses` covers streaming, function tools, schema-validated output, background tasks with cancel and replay, and `store` / `previous_response_id`. For Codex CLI it handles namespaced tools and `reasoning.encrypted_content` (#726).
  - `/v1/messages` (Anthropic) is present.
  - An Ollama-style `/api/tags|ps|show|chat|generate|pull|delete` with model swapping and `keep_alive`.
  - `"cors_origins"`, `--api-key`, MCP servers for the web chat (`serve/mcp.py`), and file attachments (PDF, Office, EPUB, RTF parsed to text).

  — R "Using it", C v1.1.55, [docs/RESPONSES.md](https://github.com/mw00/project-maya/blob/main/docs/RESPONSES.md)
- **Prompt memo** (`MAYA_PROMPT_MEMO`, on by default since v1.1.55): "the server prepares a re-sent conversation by increment (... the same token ids): a follow-up turn after a 115K-token history takes about 6% less time" (GLM, 3x V100). — C v1.1.55
- **DeepSeek "conversations kept" and restore points** keep a few conversation states so agents with sub-agents stop re-reading their prompts (#375). This is DeepSeek-only. — [issue #375](https://github.com/mw00/project-maya/issues/375), S
- **What Avalon has (local checkout c24c03d):** `/v1/responses` and `/v1/messages` routes in `tools/server/server.cpp:276,280`, `return_progress` in `server-context.cpp`, `--reasoning-budget` / `--reasoning-budget-message` in `common/arg.cpp:3763,3771`. No loop guard: grep for `loop_guard` or tool-repeat logic in `tools/server` returns nothing. Avalon's server also has `--cache-ram`, `--ctx-checkpoints`, `--cache-reuse` and `--slot-save-path`. — local grep; [llamacpp_server_features.md](file:///home/dan/research_notes/Avalon%20upgrades%20from%20upstream%20projects/llamacpp_server_features.md)

### Inferences
- The loop guard targets the exact failure mode we care about (opencode tool loops), and the user's 0.1 temperature makes near-greedy loops more likely. It works on token ids and the message history, so it can be ported as a **proxy in front of llama-server** without touching Avalon's C++. The token-level part does need streaming control (holding back 64 tokens and re-requesting with a penalty), which is easier inside the server.
- Upstream llama.cpp's open PR #27578 (soft wrap-up for the reasoning budget) is the llama.cpp-native version of Maya's "close at the next sentence end".

### Gaps
- I did not measure how often Qwen3.8 on Avalon actually loops in opencode. The value of the loop guard is argued from Maya's GLM and MiniMax anecdotes, not our own logs.

## Q4. Overlap or shared lineage with Avalon, and ranked list of ideas worth porting

### Takeaway
There are no common authors or code. Maya descends from Strata, while Avalon descends from llama.cpp with buun's expert cache and ik_llama's quants. The overlap is conceptual: Avalon credits Strata for the expert-ranking format and the preload idea, and both run tail skipping of missed experts, a GPU expert cache with the CPU computing misses, MTP drafting and q8 KV. Everything below is ranked by value for our single-user opencode workload.

### Cited Findings (basis for the ranking)
- Avalon credits "Strata (ranking format, preload idea)" and runs tail skip of rank ≥7 missed experts by default (`GGML_CUDA_MOE_TAIL_SKIP`). Avalon main (48 commits past our build) adds a registered-RAM expert arena and a hosted-MoE limit computed from link rate. — [avalon_baseline.md](file:///home/dan/research_notes/Avalon%20upgrades%20from%20upstream%20projects/avalon_baseline.md)
- Maya: weight-share miss skip gives +7-13% decode at KL 0.037-0.052 on GLM (RTX 3090). Direct SSD reads give +26% decode on a RAM-short PC (Qwen, V100 plus 30 GB RAM). Drive-probed reader concurrency is +5% on NVMe beside an 8 GB card (C v1.1.43). Prompt memo is -6% on a 115K follow-up. `--calibrate` keeps only gains above 3%. — S, C

### Ranked ideas to port into Avalon

| # | Idea (Maya source) | Expected gain for us | Effort | Risk |
|---|---|---|---|---|
| 1 | **Loop guard**: periodic token-loop detector with drop, penalty, close-think and stop, plus tool-repeat notes and blocking, prompt collapse and keep-going (serve/loop_guard.py, 680 lines of Python, MIT) | No speed gain; fewer stuck opencode sessions and wasted 10K-token loops. Highest practical value | Low-medium: a proxy in Python (reuse the logic as is) or about 500 lines of C++ in server-context | Low: it is a no-op when no loop is seen. Medium for the tool-blocking part, which edits the prompt; start with notes only |
| 2 | **Soft thinking budget**: close `<think>` at the next sentence or line end after N tokens; configurable without a restart | Caps xhigh reasoning latency without hurting answer quality from a mid-sentence cut | Low (or adopt upstream PR #27578) | Low |
| 3 | **Tolerant tool-call parsing and turn-end markers in the thinking**: keep the answer going when a Qwen XML call is malformed; treat quoted end markers in the thinking as text (#613, #706) | Avoids failed requests in long agent turns | Low (in the common/chat Qwen parser) | Low |
| 4 | **Direct, parallel expert reads for experts not in RAM** (O_DIRECT, drive-probed 2-16 readers, prefetch while the CPU prepares the layer; `MAYA_EXPERT_DIRECT`, `MAYA_EXPERT_STAGERS`) | Possibly large if Avalon thrashes the page cache: 89 GiB model against about 57 GB RAM. Maya: +26% decode on a RAM-short PC. Needs measuring against Avalon main's registered-RAM arena first | Medium-high: a new tier in the ggml-backend / moe-cache path | Medium: NTFS3 mount (O_DIRECT support on ntfs3 is uncertain), alignment, and interaction with mmap |
| 5 | **Miss skip by weight share** (skip a missed expert whose route weight < W on layers ≥ F) instead of or in addition to rank ≥7 | +5-13% decode (GLM numbers) at a small KL; Qwen's 10-of-512 routing may react differently | Low: an env knob next to `GGML_CUDA_MOE_TAIL_SKIP` | Medium: changes outputs, so validate with KL or agent evals |
| 6 | **Calibration run** (`--calibrate`-style): sweep CPU threads (DDR4 dual-channel on a 5700G may favour fewer than 16), `--moe-cache-cpu-overlap`, `GGML_SCHED_PREFETCH_EXPERTS`, `-ub`; keep only gains above 3% measured on fresh prompts | A few % to about 10% (unknown); Avalon users report gains from cpu-overlap 2 and prefetch 6 | Low: a script around `bench.py` (cf. upstream `scripts/moe-cache-tune.py`) | Low |
| 7 | **Drafter auto-keep** (measure MTP or ngram on and off on a sustained 1,500-token run, keep if more than 5% faster) | Small: our ngram-simple+MTP choice is already measured (168 vs 118 t/s on edits) | Low | Low |
| 8 | **Turbo preset** (greedy, no thinking, skip on, fastest drafting) as a second systemd or launcher profile, or a per-request alias like the `-quick` model | Faster small tasks (titles, quick edits) | Trivial | Low |
| 9 | **Prompt memo** (incremental tokenization and templating of re-sent history) | About 6% of follow-up TTFT at 115K in Python; in llama.cpp's C++ tokenizer probably less | Medium | Low |
| 10 | **One-GPU multi-request batching** (dense weights read once for N requests) | Only for opencode side requests (title or sub-agent); upstream llama-server already batches slots, but Avalon runs `--parallel 1` | Already available via `--parallel 2`; tuning only | Medium: VRAM taken from the expert cache |

Not portable or not relevant: DFlash2 (a GLM-specific drafter, non-commercial license), DSpark (DeepSeek and MiniMax), EXL3 kernels, Engram / DeepSeek tooling, tensor-parallel multi-GPU, the vision on-demand lending (no images in our workload), the dashboard, Models page and savings counter.

### Verdict: does running Maya beat Avalon for us?
**No.**
- **Qwen path is stale.** Maya's Qwen-3.8-Flash-Next path is Strata engine 0.1.27, measured 1.4-1.7x slower in decode than current Strata on both NVIDIA and AMD (#604) and much slower at long context (#898). Current Strata itself is about 20% slower in decode and about 25% slower in prefill than Avalon on RDNA3. Maya would also force a lower-bit quant (UD-IQ2_XS, 80 GB, which does not fit in our 78 GB free).
- **GLM-5.3 is far slower.** GLM-5.3-Flash Maya-XS (73 GB) would run at about 15 t/s or less on one 7900 XT (Maya's own measurement: about 15 t/s and about 414 t/s prefill, with 192 GB RAM). That is useless for 131-262K-token agent loops.
- **What is worth taking.** The useful parts are Maya's server-side agent safeguards (#1-#3 above) and a measured look at direct SSD expert reads and weight-share skipping (#4-#5).

### Inferences
- Since Maya and Avalon both draw on Strata's ideas, Strata upstream (see strata.md) remains the better source for Qwen-specific engine techniques. Maya's original contributions sit mostly in the server and in the non-Qwen engines.

### Gaps
- I did not verify whether ntfs3 (our D drive) supports O_DIRECT reads; that decides whether idea #4 could work as is.
- There is no evaluation of GLM-5.3-Flash Maya-XS quality against Qwen3.8 IQ3_KS for coding agents.
