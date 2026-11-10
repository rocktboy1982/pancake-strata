<h1 align="center">Strata</h1>

**English** · [繁體中文](README.zh-TW.md) · [简体中文](README.zh-CN.md) · [日本語](README.ja.md) · [Deutsch](README.de.md) · [Français](README.fr.md) · [Español](README.es.md) · [Português](README.pt-BR.md)

<p align="center"><b>Run a 125-billion-parameter AI model on your own gaming PC or laptop</b><br>
NVIDIA or AMD graphics card · Windows or Linux · a desktop app · any GGUF model too · free and open source</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** on a normal PC. This is a
large, smart AI model that usually needs a server. It chats, writes code, reads pictures and works with your apps
and coding agents. Nothing leaves your PC.

**It can also run any other GGUF model** (Gemma, Llama, Mistral, a fine-tune of your own) behind the same app and
API, with [llama.cpp](https://github.com/ggml-org/llama.cpp): see [Other models](#other-models-any-gguf).

> **This is a fork** of [Niko1221/Strata](https://github.com/Niko1221/Strata). It adds a desktop app, a profile for
> laptops and 8-12 GB cards, the generic mode for any GGUF, and an activity bar in the web app
> ([what is new](#what-this-fork-adds)). Everything else, and every number measured below, is the original's.

## What this fork adds

| | What it does | Status |
| --- | --- | --- |
| **Desktop app** (`STRATA-GUI.bat`) | install, start and stop models by clicking; shows your GPU and RAM; English or Traditional Chinese | works; tested without a GPU |
| **Low-VRAM profile** (`--low-vram on`) | for NVIDIA cards with 8-12 GB (most gaming laptops): picks the settings that leave the most VRAM for the model | opt-in; **not measured yet** on an 8 GB card |
| **Generic mode** (any GGUF) | runs other models with llama.cpp, behind Strata's web app, Monitor and API | tested against a stand-in llama-server; **not yet with a real model** |
| **Activity bar** | a thin bar under the web app's header whenever Strata works: loading, queued, reading your prompt (with %), writing | works |

"Not measured yet" means exactly that: if you run it on such a PC, please [report the numbers](docs/COMMUNITY_BENCHMARKS.md).

## How fast is it?

Measured by the original author on two ordinary gaming PCs. A token is about ¾ of a word.

- **Writes answers:** how fast the reply appears in a short chat. 60 tokens per second is faster than you can read.
- **Reads your prompt:** how fast it takes in what you send (here a 32K-token document, code or chat history).

<table>
<tr><th>NVIDIA: RTX 5070 (12 GB), Ryzen 5 7600, 64 GB RAM</th><th>AMD: RX 9070 XT (16 GB), Ryzen 9 3900X, 47 GB RAM</th></tr>
<tr><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 94 tokens/s | 2,650 tokens/s |
| **IQ2_XS** | 79 tokens/s | 2,090 tokens/s |
| **IQ3_XXS** | 62 tokens/s | 1,750 tokens/s |
| **IQ3_S** | 53 tokens/s | 1,620 tokens/s |
| **Coder** | 55 tokens/s | 2,180 tokens/s |

</td><td>

| Size | Writes answers | Reads your prompt |
| --- | ---: | ---: |
| **Q2_0** | 60 tokens/s | 1,160 tokens/s |
| **IQ2_XS** | 52 tokens/s | 1,110 tokens/s |
| **Coder** | 44 tokens/s | 1,420 tokens/s |

</td></tr>
</table>

NVIDIA: Q2_0 with engine 0.1.36, the other rows with 0.1.26 (4K answers, 32K prompts). The full tables are in
[DETAILS.md](docs/DETAILS.md#speed-measured). A card with more VRAM is faster: an RTX 3090 (24 GB) should write
about 100-140 tokens per second. A card with less VRAM is slower; how much slower on an 8 GB laptop is not measured
yet. Long chats and other cards: [speed of each model](docs/MODELS.md#how-fast-is-each-size),
[community results](docs/COMMUNITY_BENCHMARKS.md).

## What you need

| | |
| --- | --- |
| **Graphics card** | **NVIDIA** GeForce RTX 20, 30, 40 or 50 series, or **AMD** Radeon RX 7900 XT / XTX, RX 7800 XT / 7700 XT, RX 9060 XT, RX 9070 / 9070 XT, Radeon AI PRO R9700 or RX 6800 / 6900 series. **12 GB of VRAM or more** is what the numbers above were measured on. An NVIDIA card with **8-12 GB** (a laptop RTX 4060 / 4070 / 5060 ...) runs too, slower: use the [low-VRAM profile](#laptops-and-8-12-gb-cards). |
| **RAM** | 32 GB or more. Your RAM decides [which model](#which-model-should-i-pick) fits. 64 GB runs every size. |
| **Disk** | About 80 GB free. Use an SSD if you can: the first start is much faster. |
| **System** | Windows 10 / 11 or Linux, and a current graphics driver from NVIDIA or AMD. |

The installer sets up everything else. Two or three cards can share the model ([multi-GPU](docs/MULTI_GPU.md)).

Experimental, written and tested by community members on their own machines:

- **Older graphics cards** (Tesla P40 / V100, GTX 10, Radeon VII / MI50, RX 6700 XT, RX 5500 XT): [Older GPUs](docs/OLDER_GPUS.md).
- **Intel Arc**, built from source on Linux: [Intel Arc](docs/INTEL_ARC.md).
- **AMD Ryzen AI Max (Strix Halo)**, built from source on Linux: [Strix Halo](docs/STRIX_HALO.md).
- **Older processors without AVX2**: they work, but slowly. [Older CPUs](docs/INSTALL.md#older-cpus-experimental).

The full list: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Install

[Download Strata](https://github.com/chanLik1208-dev/Strata/archive/refs/heads/main.zip) and unzip it (or
`git clone` it). Then pick one of three ways. All of them install the same thing.

### 1. The desktop app (easiest)

**Windows:** double-click **`STRATA-GUI.bat`**. **Linux:** run **`./strata-gui.sh`** (it needs Python's tkinter:
`sudo apt install python3-tk`).

The window shows your graphics card and RAM and recommends a model. Click **Install / apply**. It downloads the model
(about 70 GB), sets it up and starts it. A bar at the bottom moves while it works, and the log shows every step. The
command it runs is shown in the window, so you can see exactly what it does.

The labels follow your system's language (English or Traditional Chinese). The **Language** menu at the bottom
changes it.

### 2. Let your AI set it up

Do you use an AI coding assistant (Claude Code, Cursor, Codex, GitHub Copilot, ...)? Paste this into it:

```text
Set up Strata on this PC for me: https://github.com/chanLik1208-dev/Strata - follow docs/AI_SETUP.md in that repository.
```

It checks your graphics card, RAM and disk and picks the model that fits. Then it installs and starts it and tells
you how to connect your apps. AI tools can also install, start and stop Strata through its
[MCP server](docs/MCP_SERVER.md).

### 3. In the console

**Windows:** double-click **`START-HERE.bat`**. **Linux:** run **`./setup.sh`** in the Strata folder.

It asks a few questions (which model and size, how much context, whether it should read pictures). Press Enter each
time for the recommended answer.

### What happens next

The download is about 70 GB. If it stops, start again: it continues where it left off. Then your browser opens the
Strata app at `http://127.0.0.1:8080`.

> **While the model starts, your PC can be slow or stop responding for 1-3 minutes** (longest the first time).
> Strata loads 35-55 GB into your RAM and locks part of it for the graphics card. This is normal. Wait, and don't
> close the window.

**Next time**, open the desktop app and click **Start**, or run `START-HERE.bat` (`./setup.sh`) again. It starts right
away and downloads nothing twice. `UPDATE.bat` (`./update.sh`) updates Strata without starting it. Updating, Docker,
several cards, where the files go and every option: [docs/INSTALL.md](docs/INSTALL.md).

## Laptops and 8-12 GB cards

Strata keeps the busiest parts of the model on the graphics card and the rest in RAM. On a card with 8-12 GB there is
little room, so every MB counts. **`--low-vram on`** picks the settings that leave the most VRAM for the model:

| Setting | On an 8 GB card | On a 10-12 GB card |
| --- | --- | --- |
| Context | 16K tokens | 32K tokens |
| KV cache (the model's memory of the chat) | 4-bit: half the VRAM | 4-bit |
| Pictures | read on the processor | read on the processor |
| Draft tokens | English and code (~215 MiB less VRAM) | unchanged |
| A card that also drives the screen | 1,500 MiB kept free for the desktop | the same |

- The desktop app turns it on by itself on such a card. In the console: `START-HERE.bat --low-vram on`.
- Anything you set yourself wins (for example `--context 32768`).
- The English draft tokens only make Chinese, Japanese and Korean answers come a little slower. The answers
  themselves stay the same. Add `--draft-vocab cjk` to keep the default.
- You still need **32 GB of RAM**. On a laptop with 32 GB, take the **Coder** model.
- Without the flag nothing changes: the install is the same as the original's.

## Other models (any GGUF)

Strata's own engine is written for Qwen3.8-Flash-Next and runs nothing else. **The generic mode runs any GGUF
model** with llama.cpp's `llama-server` and puts Strata's web app in front of it:

- **Chat**, the live **Monitor** and the **activity bar** work as with Strata's engine.
- **The same API:** `/v1/chat/completions` (OpenAI), `/v1/messages` (Anthropic) and `/v1/responses` go to
  llama-server, which uses the model's own chat template. Tools and thinking work as that template defines them.
- **A model larger than the VRAM still runs.** llama-server keeps what does not fit in RAM. For a mixture-of-experts
  model (Gemma 4 26B-A4B, Qwen3 30B-A3B, gpt-oss) it keeps the experts there, which costs little speed. For an
  ordinary ("dense") model it keeps whole layers, which costs more.

**In the desktop app:** open the tab **Generic: any GGUF model**. Click **Get llama-server** once (about 550 MB for
NVIDIA). Then pick the `.gguf` file and click **Start the GGUF model**.

**In the console:**

```
python tools/get_llama_server.py
python serve/generic.py --model D:\models\my-model.gguf --ctx 32768 --low-vram
```

It is not Strata's engine. It has no expert cache and no draft helper, so a Qwen3.8-Flash-Next model runs faster in
Strata's own mode. Every option: [docs/GENERIC.md](docs/GENERIC.md).

## Which model should I pick?

This is for Strata's own mode. The installer recommends one for your RAM. The same model comes in several sizes,
compressed more or less. Smaller sizes are faster. Larger sizes are a bit smarter.

| Your RAM | Take | Why |
| --- | --- | --- |
| **32 GB** | **Coder** | it fits 32 GB, and it is made for code (with a 24 GB card, Q2_0 and IQ2_XS run too) |
| **48 GB** | **IQ2_XS** (or Q2_0, the fastest) | the larger sizes do not fit |
| **64 GB** | **IQ2_XS** (recommended), or IQ3_XXS / IQ3_S | every size fits; IQ3_S is the best and the slowest |
| **96 GB or more** | **IQ3_S**, or Unsloth's UD-IQ4_XS (~4-bit) | room for the largest sizes with everything else open |

- **[Coder](docs/MODELS.md#coder):** a coding version with half of the experts removed. It reaches 91% of the full
  model's SWE-bench Verified score (measured by its authors) and fits 32 GB of RAM. It is weaker outside code,
  including Chinese and other CJK text (#438). For those, take Q2_0, IQ2_XS or IQ3_S, which keep every expert.
- **[Swift 1.5](docs/MODELS.md#swift-15):** a fine-tune that thinks for a much shorter time before it answers. You
  get the answer sooner, at about the same quality.
- **[Unsloth UD-IQ4_XS](docs/MODELS.md#unsloth-ud-iq4_xs):** Unsloth's ~4-bit version, between IQ3_S and
  UD-Q4_K_XL in quality. A 94 GB download. With less than ~80 GB of RAM, Strata reads part of it from the SSD
  while it answers, so it is slower there (an NVMe SSD helps).
- **[Unsloth UD-Q4_K_XL](docs/MODELS.md#unsloth-ud-q4_k_xl-experimental)** (experimental): the closest to the full
  model. But Strata reads most of it from the SSD while it answers, so it writes only 7-8.5 tokens/s on a 64 GB PC.
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs):** you set it up by hand. It is
  not in the installer's menu.

Sizes, downloads and what fits where: [docs/MODELS.md](docs/MODELS.md). To add another model later, use the desktop
app's **Install / apply**, or run `SETUP.bat` (Linux: `./setup.sh --setup`).

## Using it

<p align="center"><img src="docs/media/runpagoda.png" width="900" alt="The Strata app's Monitor tab next to a coding agent"><br>
<sub>The Strata app's <b>Monitor</b> (left) while a coding agent writes the pagoda garden from the video (right)</sub></p>

- **In the browser:** open `http://127.0.0.1:8080`. It has **Chat**, a live **Monitor** of the model and your
  GPU/CPU/RAM, and **About** with the settings and addresses. The bar under the header shows what Strata is doing:
  amber while it loads or waits in the queue, blue while it reads your prompt (with how far it is), green while it
  writes.
- **Your apps and coding agents:** add an "OpenAI-compatible" provider with the base URL
  **`http://127.0.0.1:8080/v1`**. Any API key and any model name work.
  - Apps that use Anthropic's API: `http://127.0.0.1:8080/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`).
  - Codex CLI and other apps that use the OpenAI Responses API: `/v1/responses`
    ([setup](docs/DETAILS.md#the-responses-api-and-codex-cli)).
- **Thinking:** choose **off, low, medium or high** in the chat menu or in your app's "reasoning effort". Off is the
  fastest. High is best for hard questions.
- **Pictures:** say yes to "Images?" in setup. Then click **Picture** in the chat, or attach pictures in your app.
  AMD cards read pictures on Linux through the processor; on Windows they can't yet.
- **From your phone or another PC:** `START-HERE.bat --setup --host 0.0.0.0 --api-key <secret>`. Always set a key.
- **One request at a time:** by default Strata answers one request, and the others wait. To answer several at once,
  set `"parallel": 2` ([BATCHING.md](docs/BATCHING.md)). On a 12 GB card this makes each answer slower.
- **Long prompts:** Strata reads the first message of a chat in full, about 1 minute per 30,000 tokens. Follow-up
  messages start in seconds.
- **Less motion:** if your system is set to reduce motion, the web app fades instead of sliding.

More: [where your chats are stored](docs/INSTALL.md#where-things-are-stored), [the API](docs/DETAILS.md#using-it).

## Something went wrong?

- **My PC froze the first time Strata started.** This is normal while it loads the model. Wait, and don't close the
  window. Still frozen after 10 minutes? Restart the PC, close other programs and try again, or pick a smaller size.
- **It stopped while downloading or installing.** Start it again. It continues where it stopped.
- **It's very slow and the disk light keeps blinking, or it says "the engine stopped unexpectedly".** Your PC does
  not have enough free RAM. Close other programs (browsers use a lot), or pick a smaller size (Q2_0 or IQ2_XS).
- **It stops with "no VRAM is left for the expert cache" on a laptop.** Use the
  [low-VRAM profile](#laptops-and-8-12-gb-cards), and close other programs that use the graphics card.
- **It says port 8080 is already in use.** Strata is already running. Look for its window, or stop it in the desktop
  app.
- **The generic mode says "llama-server stopped".** Its log is `strata-generic-<model>.log` in the Strata folder.
  Most often the model did not fit: try a smaller `--ctx` or `--low-vram`.

More problems and their fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). For a problem with what this fork
adds, open an [issue here](https://github.com/chanLik1208-dev/Strata/issues). For the engine itself, the original
project's [issues](https://github.com/Niko1221/Strata/issues) are the place. Attach `strata-<model>.log` from the
Strata folder. Found a security problem? Report it privately: [SECURITY.md](SECURITY.md).

## How does it work?

Models like this one usually run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
8-24 GB. Strata makes the model fit by **sharing the work across your whole PC**. Think of a kitchen: the things
you use all the time stay on the counter, and the rest waits in the pantry.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

- **The model is a team of 24,576 small specialists ("experts").** Each word needs only 10 of them.
- **Your graphics card** keeps the few thousand experts that are used most often. **Your RAM** holds all of them,
  and **your processor** works on the rest at the same time. **Your SSD** holds a big lookup table.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

- **Guess, then check:** a small helper guesses the next few words. The big model checks them all at once. You get
  the same answer, 1.6-1.8x sooner.
- **Long texts are read in big pieces** (up to 8,192 tokens at a time), at over 1,000 tokens per second.

The longer explanation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Every part and its numbers:
[the details](docs/DETAILS.md#how-it-works) and the [paper](docs/paper/Strata-Paper.pdf).

## Credits and license

Strata is by [Niko1221](https://github.com/Niko1221/Strata); this fork only adds the parts listed
[above](#what-this-fork-adds). The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by
the Qwen team. It was compressed by [ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
UkisAI (Swift 1.5) and Unsloth. Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp), and
the generic mode runs llama.cpp's own `llama-server`. The web app's motion follows the
[Dynamization](https://github.com/chanLik1208-dev/Dynamization) design pack. All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits).

Strata is open source under the [MIT License](LICENSE). A few parts and every model have their own licenses
([which ones](docs/HOW_IT_WORKS.md#license)).

## Support Strata

Strata is free and open source. If it is useful to you, you can support the original author's work on it:

<p align="center"><a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>
