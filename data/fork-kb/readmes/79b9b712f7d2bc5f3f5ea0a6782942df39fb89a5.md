<h1 align="center">Strata for Mac</h1>

<p align="center"><b>Run sovereign 78B–125B parameter Mixture-of-Experts (MoE) AI models locally on your Mac or PC</b><br>
Apple Silicon Mac (M1/M2/M3/M4/M5) · Metal GPU Acceleration · Unified Memory · Also supports NVIDIA & AMD · Free and Open Source</p>

<p align="center">
  <a href="https://github.com/shruxx/StrataForMac/archive/refs/heads/main.zip"><b>⬇️ Download Strata for Mac (.zip)</b></a> &nbsp;·&nbsp; 
  <a href="#quickstart">Quickstart</a> &nbsp;·&nbsp; 
  <a href="#which-model-should-i-pick">Models</a> &nbsp;·&nbsp; 
  <a href="docs/MACOS.md">macOS Architecture</a>
</p>

<p align="center"><a href="https://github.com/shruxx/StrataForMac/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/shruxx/StrataForMac/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

Strata runs massive, state-of-the-art open-weight Mixture-of-Experts (MoE) models — such as **[Aleph Alpha's Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1)** (78.1B) and **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** (125B) — directly on your Mac or PC. It chats, writes code, analyses documents, reads images and powers your coding agents. Everything runs 100% locally: **no data ever leaves your device**.

### Why Strata on Apple Silicon Mac?
- **Unified Memory Architecture (UMA):** Unlike traditional PCs constrained by PCIe transfer bottlenecks between CPU RAM and VRAM, Apple Silicon Macs share up to 128 GB+ of unified memory at massive bandwidth (200 to 800 GB/s on Pro, Max, and Ultra chips).
- **Native Metal GPU Acceleration:** Custom Metal compute shaders (`ggml-metal`) execute routing, attention, and feed-forward expert layers directly on Apple GPU cores.
- **Sovereign European & Open Models:** Built-in first-class support for Aleph Alpha's **Kolibri-1** (best-in-class German & English, 262K context, Apache 2.0) and Qwen's specialized **Coder** variants.
- **One-Click Setup:** Automatically detects your Mac's hardware (cores, Unified Memory), chooses the right model configuration, compiles the native Metal engine, and opens the web app.

---

## How fast is it?

"Writes answers" is the generation speed in tokens per second; "reads your prompt" is how fast it ingests long context (code, documents, or chat history). A token is roughly ¾ of a word, so 40–60 tokens/s is significantly faster than comfortable reading speed.

<table>
<tr>
  <th>Apple Silicon Mac (Metal GPU, UMA)</th>
  <th>NVIDIA: RTX 5070 (12 GB), 64 GB RAM</th>
  <th>AMD: RX 9070 XT (16 GB), 47 GB RAM</th>
</tr>
<tr>
<td>

| Model / Size | Writes answers | Reads prompt |
| :--- | ---: | ---: |
| **Kolibri-1 (Q4_K_M)** | ~28–45 tokens/s | 1,400+ tokens/s |
| **Qwen Coder (IQ1_M)** | ~35–55 tokens/s | 1,850+ tokens/s |
| **Qwen Q2_0** | ~40–65 tokens/s | 1,900+ tokens/s |
| **Qwen IQ2_XS** | ~32–50 tokens/s | 1,600+ tokens/s |

</td>
<td>

| Model / Size | Writes answers | Reads prompt |
| :--- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td>
<td>

| Model / Size | Writes answers | Reads prompt |
| :--- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td>
</tr>
</table>

*Measured on Apple Silicon M-series (Pro/Max/Ultra), RTX 5070 and RX 9070 XT. Full breakdown and other cards: [speed of each model](docs/MODELS.md#how-fast-is-each-size) and [community results](docs/COMMUNITY_BENCHMARKS.md).*

---

## What you need

| | |
| :--- | :--- |
| **Apple Silicon Mac** | Any Mac with **M1, M2, M3, M4, or M5** (base, Pro, Max, Ultra) with **Unified Memory**:<br>• **32 GB:** Runs Qwen Coder (`IQ1_M`); Kolibri-1 runs with part of its experts read from the SSD<br>• **48 GB:** Runs **Kolibri-1 Q4_K_M** (part of its experts from the SSD; the runner tunes the GPU share) and Qwen `IQ2_XS` at full speed<br>• **64 GB:** Runs Kolibri-1 and Qwen `IQ3_XXS` with room for large contexts<br>• **96 GB – 128 GB+:** Runs Kolibri-1 with 262K context and Qwen `IQ3_S` |
| **PC (NVIDIA / AMD)** | **NVIDIA** GeForce RTX 20, 30, 40 or 50 series (12 GB+ VRAM) or **AMD** Radeon RX 7900 / 9070 / AI PRO series, with 32–64 GB system RAM |
| **Disk** | ~70–80 GB free space on a fast SSD |
| **System** | macOS 12 (Monterey) or newer (macOS 13+ recommended), Windows 10 / 11 or Linux |

Experimental on PCs, written and tested by community members: [older graphics cards](docs/OLDER_GPUS.md) (Tesla P40 / V100, GTX 10, Radeon VII / MI50), [Intel Arc](docs/INTEL_ARC.md), [processors without AVX2](docs/INSTALL.md#older-cpus-experimental).

Full platform details: [docs/MACOS.md](docs/MACOS.md) and [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

---

## Quickstart

**[⬇️ Download Strata for Mac (.zip)](https://github.com/shruxx/StrataForMac/archive/refs/heads/main.zip)** and unzip it, or clone the repository in Terminal:

### 1. One-click setup on macOS

Open Terminal and run:

```bash
git clone https://github.com/shruxx/StrataForMac.git
cd StrataForMac
./setup.sh
```

*(Or if you downloaded the .zip, open Terminal in the unzipped `StrataForMac` folder and run `./setup.sh`)*.

Setup checks your Mac's CPU, GPU cores, and Unified Memory, recommends the optimal model for your hardware, compiles the native Metal engine, downloads the weights, and launches the server.

To install a specific model directly (e.g. **Kolibri-1**):
```bash
./setup.sh --family kolibri --model Q4_K_M
```

Or for automated / non-interactive installation with recommended defaults:
```bash
./setup.sh --yes
```

### 2. Or let your AI assistant set it up

Using Claude Code, Cursor, Codex, or GitHub Copilot? Paste this prompt:

```text
Set up Strata on this Mac for me: https://github.com/shruxx/StrataForMac - follow docs/AI_SETUP.md in that repository.
```

Your AI assistant will inspect your hardware, choose the best model, run setup, verify the server, and connect your tools. Strata also includes a native [MCP server](docs/MCP_SERVER.md) for direct tool interaction.

### 3. Windows & Linux

- **Windows:** [Download the ZIP](https://github.com/shruxx/StrataForMac/archive/refs/heads/main.zip), unzip it, and double-click **`START-HERE.bat`** (or run `START-HERE.bat --setup`).
- **Linux:** Run **`./setup.sh`** in the unzipped or cloned folder.

---

## Which model should I pick?

Setup automatically suggests the best fit for your RAM. 

| Unified Memory / RAM | Recommended Model | Highlights |
| :--- | :--- | :--- |
| **32 GB** | **[Coder](docs/MODELS.md#coder)** (Qwen) | Specialized coding version with half the experts pruned. High SWE-bench scores; fits 32 GB easily. |
| **48 GB** | **[Kolibri-1](docs/MODELS.md#kolibri-1-aleph-alpha)** (Q4_K_M) | **Aleph Alpha's 78.1B sovereign MoE**. Superb German & English reasoning, 262K context. 47.5 GB: on 48 GB part of its experts are read from the SSD. |
| **64 GB** | **Kolibri-1** or **Qwen IQ2_XS** | High accuracy, full context capacity, fast inference. |
| **96 GB or more** | **Kolibri-1** or **Qwen IQ3_S** | Maximum precision, large context window (up to 262,144 tokens). |

### Featured Models
- **[Kolibri-1 (Aleph Alpha)](docs/MODELS.md#kolibri-1-aleph-alpha)**: Released in October 2026 under the Apache 2.0 license. Features 50 layers with 384 routed experts per layer + 1 shared expert (Top-6 routing, only **3.46B parameters active per token**). Offers state-of-the-art German and English comprehension, deep reasoning, and a 262K context window. Available in `Q4_K_M` (47.5 GB single file).
- **[Qwen3.8-Flash-Next Coder](docs/MODELS.md#coder)**: Coding specialist with pruned experts, tailored for development workflows and code generation. Fits comfortably in 32 GB RAM.
- **[Swift 1.5](docs/MODELS.md#swift-15)**: A fine-tune that reaches answers with condensed thinking phases for faster turnaround.
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs)** (~4-bit, 94 GB, new in 0.1.39) and the experimental **[UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)** (111 GB): the largest versions; part of their experts are read from the SSD.

Model details, download sizes, and benchmarks: [docs/MODELS.md](docs/MODELS.md).

---

## Using it

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The Strata app's Monitor tab next to a coding agent"><br>
<sub>The Strata app's <b>Monitor</b> (left) while a coding agent writes the pagoda garden from the video (right)</sub></p>

- **Web App:** Open `http://127.0.0.1:8080` in your browser. Includes a chat interface, system monitor (GPU, CPU, Memory), and settings.
- **OpenAI-Compatible API:** Connect Cursor, Continue, LibreChat, or custom scripts:
  - Base URL: **`http://127.0.0.1:8080/v1`**
  - API Key: any string (e.g. `sk-strata`)
  - Model: any string (e.g. `kolibri-1` or `qwen3.8`)
- **Anthropic API:** Point Claude Code or Anthropic-compatible apps to:
  - Base URL: **`http://127.0.0.1:8080`** (`ANTHROPIC_BASE_URL=http://127.0.0.1:8080`)
- **Multimodal (Images):** Enable images during setup, then upload pictures directly in the chat interface or send vision requests via the API.
- **Stopping and Restarting:** Close the terminal window to stop the engine. Run `./run-kolibri-q4_k_m.sh` (or `./run-<model>.sh`) to start it again instantly.
- **Codex CLI and other apps that use the OpenAI Responses API:** `http://127.0.0.1:8080/v1/responses` ([setup](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Thinking:** choose **off, low, medium or high** in the chat menu or in your app's "reasoning effort". Off is the fastest; high is best for hard questions.
- **Several models:** with more than one model set up, a list in the web app's header switches between them (the current one stops first, so its memory is free).
- **One request at a time:** by default Strata answers one request and the others wait; `"parallel": 2` in `strata-<model>.json` answers several at once ([BATCHING.md](docs/BATCHING.md), NVIDIA/AMD engine).

Full API reference and advanced settings: [docs/DETAILS.md](docs/DETAILS.md#using-it).

---

## How does it work?

Large MoE models normally require multiple data center GPUs costing tens of thousands of dollars. Strata makes them run smoothly on consumer hardware:

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="MoE routing across graphics card, RAM, and SSD"></p>

1. **Sparsity & Expert Routing:** A model like Kolibri-1 or Qwen consists of dozens to hundreds of specialized sub-networks ("experts"). For each generated token, a routing gate dynamically selects only a tiny fraction of them (e.g. Top-6 experts = only 3.46B parameters active per token in Kolibri-1).
2. **Apple Silicon Unified Memory:** On macOS, CPU and GPU share the same unified memory pool at high bandwidth (up to 400–800 GB/s). All active weights and KV caches reside in unified memory, while Metal compute shaders execute tensor math directly on Apple GPU cores with zero PCIe transfer penalties.
3. **Speculative Decoding:** A lightweight draft mechanism predicts candidate token sequences in advance; the main model validates them in a single parallel step, delivering a 1.6×–1.8× speedup.
4. **Chunked Prefill:** Long prompts (up to 262K tokens) are ingested in parallel chunks for maximum throughput.

Architecture details, benchmarks, and math: [docs/MACOS.md](docs/MACOS.md), [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md), and the [Strata Paper](docs/paper/Strata-Paper.pdf).

---

## Troubleshooting

- **First start takes 1–3 minutes:** On first launch, the model loads weights into memory and prepares GPU pipelines. This is normal; subsequent starts are much faster.
- **Port 8080 already in use:** Another instance is running or another process occupies port 8080. Start on another port with `--port 8081`.
- **System slow or out-of-memory:** If other heavy applications (browsers with many tabs, video editors) consume RAM, close them or select a more compact quantization format.
- More solutions: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) and [docs/MACOS.md](docs/MACOS.md).

The models: [Kolibri-1](https://huggingface.co/Aleph-Alpha/Kolibri-1) by Aleph Alpha (GGUF and llama.cpp patch by
[Hob-forge](https://huggingface.co/Hob-forge/Kolibri-1-GGUF)) and [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
by the Qwen team, compressed by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI
(Swift 1.5) and Unsloth. Strata is [Niko1221/Strata](https://github.com/Niko1221/Strata) with a macOS port; it uses parts of
[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits).
Strata is open source under the [MIT License](LICENSE). A few parts and every model have their own licenses
([which ones](docs/HOW_IT_WORKS.md#license)).

---

## Credits and License

- **Kolibri-1** created and released by **[Aleph Alpha](https://huggingface.co/Aleph-Alpha/Kolibri-1)** under the Apache 2.0 license; GGUF quantization by Hob-forge.
- **Qwen3.8-Flash-Next** created by the **Qwen Team**; quantizations by ISTA-DASLab, UkisAI, and Unsloth.
- Built upon **[llama.cpp / ggml](https://github.com/ggml-org/llama.cpp)**.
- **Strata for Mac** is free and open-source software under the [MIT License](LICENSE).
