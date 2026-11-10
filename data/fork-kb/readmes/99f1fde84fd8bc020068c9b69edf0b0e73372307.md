> ## About this fork: Flash-Next on 2× RTX 3060, PCIe 3.0
>
> This is the `flashnext-pcie3` branch of a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata). It adds 8 small, opt-in patches for running **Qwen3.8-Flash-Next** as two agents on two consumer GPUs that share one copy of the experts in RAM.
> - The patches sit on an older upstream base, not on the current `main`.
> - Everything is turned on by environment variables. With none set, the engine behaves like upstream.
> - Prefill output is bit-identical and decode is unchanged.
>
> **Test box:**
> - Xeon E5-1650 v3 (Haswell, 6c/12t, AVX2), 96 GB DDR4-2133.
> - 2× RTX 3060 12 GB at 130 W, PCIe 3.0 x16, no P2P.
> - Qwen3.8-Flash-Next GSQ-RCO **IQ3_S**; 256K context per agent.
>
> | | Upstream base | This branch |
> |---|---|---|
> | Decode, 1 agent | 53.3 tok/s | 53.0 tok/s (same, within noise) |
> | Decode, 2 agents at once (one per GPU) | ~53 + 53 tok/s | ~53 + 52 tok/s |
> | Decode at ~183K context, 2 agents | 45.8 / 47.7 | 50.5 / 50.3 |
> | Fresh prompt, 2K / 30K / 165K | 552 / 1,080 / 1,005 tok/s | 644 / 1,112 / 1,040 tok/s |
> | **Agentic turns** (~2.5K new tokens on 30-48K context) | 512 tok/s, TTFT 4.97 s | **692 tok/s, TTFT 3.67 s (+35%)** |
> | **Agentic turns at 165-183K** | 449 tok/s | **602 tok/s (+34%)** |
>
> **What the patches do**
> - `STRATA_ARENA_REUSE=1`: a second instance attaches to an already-complete shared expert arena instead of copying ~47 GB again, and GGUF-sourced arenas now qualify.
> - `STRATA_PF_AHEAD=auto`: prefetches the next layer's experts during staged prefill chunks. Before, PCIe sat idle while each layer waited for its routing.
> - `STRATA_REFILL_ASYNC=1`: refills lent expert-cache slots in the background instead of before the first token.
> - `STRATA_PARK_REWIND_MAX=4096`: skips the park/rewind when a request continues the live conversation.
> - The `auto` CPU share no longer counts prefetched experts as GPU work.
>
> Per-commit details and the full A/B are in the merged PR [#2](https://github.com/0xluk/Strata/pull/2), on top of [#1](https://github.com/0xluk/Strata/pull/1).
>
> **Setup that matters more than the patches**
> 1. **All experts in RAM.** With experts mmap'd from disk we got 10-18 tok/s.
> 2. **One instance per GPU, sharing one arena.** Pass `--shared-expert-arena /path/on/tmpfs/flashnext.arena` to both instances, on a tmpfs with `huge=always`. Two agents need ~62 GB of RAM in total.
> 3. **Pin each instance to its own half of the cores.**
>    - Example: `taskset -c 0,1,2,6,7,8` with `--pool-cores 1,2` for instance A, and `taskset -c 3,4,5,9,10,11` with `--pool-cores 4,5` for instance B.
>    - Unpinned, the two instances fought over the cores: 31 + 33 tok/s. Pinned: 51 + 51.
> 4. **Engine flags:** `--kv-resident 20480` (KV streaming, so 256K context works on 12 GB cards), `--kv int8`, `--prefill auto`, `--expert-cache auto`, MTP with `--mtp-draft-vocab data/draft_vocab_en.bin`.
> 5. **Environment:**
>    ```
>    STRATA_ARENA_REUSE=1 STRATA_PREFILL_CPU_SHARE=auto STRATA_PREFILL_CPU_SHARE_MAX=4608
>    STRATA_PF_AHEAD=auto STRATA_REFILL_ASYNC=1 STRATA_PARK_REWIND_MAX=4096
>    STRATA_KV_PREFETCH=1 STRATA_CKPT_POOL=1 STRATA_CKPT_POOL_READY=4
>    ```
>
> **Not worth it on this hardware:**
> - `--mmap-experts` (page faults: 10-18 tok/s);
> - `--slots N` on 12 GB cards;
> - a layer split across both cards (same decode speed; it serves only one agent);
> - a bigger CPU (the expert pool is limited by memory bandwidth).
>
> Roll back by unsetting the variables; each can be dropped on its own.

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Run powerful AI on your own computer.</h3>
<p align="center">Chat, code and work with images, locally.</p>

<p align="center">
  <a href="#models">Models</a> · <a href="#install"><b>Install</b></a> ·
  <a href="#hardware">Hardware</a> · <a href="#tools">Tools</a> ·
  <a href="#help">Help</a> · <a href="#support-strata">Support</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Download Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Support Strata on Buy Me a Coffee"></a>
</p>

<p align="center"><sub><b>English</b> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

**Free, open source, and yours to run.** Use Strata on your desktop or with your favorite coding tools. No account needed.

## Models

We recommend **ISTA-DASLab first**, then **Unsloth** and other providers, where a compatible model fits.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Provider | Choices |
| --- | --- |
| **ISTA-DASLab · recommended** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](docs/MODELS.md#coder) for coding. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Lower memory use but slower. |
| **Other options** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Compare sizes, memory needs and compatibility](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. These are the currently documented packs for this model.

[Model guide and compatibility](docs/QWEN36.md).

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

In the **desktop app** and the **Terminal**. What does not fit in memory is read from your SSD, so it needs a lot of free
disk space.

| Set (IST-DAS) | Download | Free disk needed |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | about 245 GB |
| 2.00 bpw | 257 GB | about 265 GB |
| 2.25 bpw | 274 GB | about 280 GB |

Runs on NVIDIA RTX 30, 40 and 50 series (Windows and Linux), AMD Radeon AI PRO R9700 (Linux), and AMD Ryzen AI Max — Strix Halo
(Windows). Setup downloads the files and the DeepSeek engine, then starts the model.

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Measured with the 2.00 bpw set on Strata 0.2.5 (Strix Halo: 0.2.7). Output is the speed of the answer, prompt is how fast Strata reads your input.

| Hardware | Output | Prompt |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB RAM | 4 to 6 tokens/s | about 110 tokens/s (4K) |
| RTX 3060 12 GB, 46 GB RAM | 9 to 11 tokens/s | about 155 tokens/s (4K) |
| 1x Radeon AI PRO R9700, 251 GB RAM | 21 to 25 tokens/s | 320 to 360 tokens/s (8K) |
| 4x Radeon AI PRO R9700 | 74 to 77 tokens/s | 400 to 450 tokens/s (8K) |
| Ryzen AI Max+ 395 (Strix Halo), 128 GB unified | 19 to 21 tokens/s (code), 11 (chat) | 87 tokens/s (4K), 125 tokens/s (8K) |

[Memory, disk and details](docs/DEEPSEEK41.md) · [Release notes](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

Not sure what fits? [Choose a model for your computer](docs/MODELS.md#pick-by-memory).

## Install

<a id="option-a-the-strata-app"></a>

### Desktop app

**[Download Strata for Windows, Linux or Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Install the app, choose a model, and start chatting. There is one installer for each system. On first start the app
downloads the engine for your graphics card. [Help with the installer](docs/INSTALL.md#the-app).

<sub>Files: `Strata-0.2.7-windows-x64-setup.exe` · `Strata-0.2.7-linux-x86_64.AppImage` · `Strata-0.2.7-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Strata desktop app with a chat and live code preview side by side">
  <br><sub>Chat, files and a live preview in one window.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Terminal app

Prefer the terminal? Install Strata, then run **`strata launch`**.

<details open>
<summary><b>Install commands</b></summary>

**Linux / Mac**

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

On Windows, open a new terminal afterwards. Then:

```sh
strata launch
```

Follow the prompts to download and start a model. [Terminal guide](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Strata terminal app showing model activity, speed, memory and recent requests">
  <br><sub>Watch your model and hardware while it runs.</sub>
</p>

<a id="build-from-source"></a>

[Build from source](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Full install guide](docs/INSTALL.md)

<a id="what-you-need"></a>

## Hardware

| Supported hardware | Guide |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 series** | [NVIDIA setup](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [AMD setup](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimental** | [RX 6000 setup](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel setup](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 or newer** | [Mac setup](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo setup](docs/STRIX_HALO.md) |

<a id="using-it"></a>

## Tools

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, and more through the OpenAI- and Anthropic-compatible APIs.

Run **`strata connect`** to see setup instructions for your tool. [Connection guide](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Help

[All guides](docs) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Report a bug](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [How it works](docs/HOW_IT_WORKS.md) · [Contribute](CONTRIBUTING.md)

## Support Strata

**Help keep local AI open to everyone.** A coffee helps fund development and testing on more hardware.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: support Strata" height="48"></a>

[MIT license](LICENSE) · [Credits and model licenses](docs/HOW_IT_WORKS.md#credits) · [Privacy](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>For AI tools and automated readers: <a href="AGENTS.md">technical reference</a> · <a href="docs/AI_SETUP.md">setup instructions</a>.</sub>

**Sponsorships, hardware partnerships or collaborations:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).

