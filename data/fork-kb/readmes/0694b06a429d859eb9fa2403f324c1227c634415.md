<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/brand/strata-lockup-dark.svg">
    <img src="docs/brand/strata-lockup-light.svg" alt="Strata" width="380">
  </picture>
</p>

<h3 align="center">Run a state-of-the-art model on your own gaming PC</h3>

<p align="center">Support for NVIDIA, AMD, Intel · Windows, Linux, Mac · free and open source</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?include_prereleases&label=release" alt="Latest release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://github.com/Niko1221/Strata/stargazers"><img src="https://img.shields.io/github/stars/Niko1221/Strata?style=flat" alt="GitHub stars"></a>
</p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote, running in the browser"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, a 125-billion-parameter model that
normally needs a server, on a normal PC. It chats, writes code, reads pictures and works with your apps and coding
agents. Your chats and files stay on your PC.

**New here?** Check [what you need](#what-you-need), then [install Strata](#install).

- **Runs on a gaming PC.** 32 GB of RAM and graphics memory together is enough to start.
- **Works with your tools.** It speaks the OpenAI and Anthropic APIs, so Claude Code, Codex CLI, OpenCode and other agents connect to it.
- **Reads pictures.** They are on by default.
- **Free.** MIT license, no account, no cloud.
- **Private.** No telemetry. What Strata downloads, when it uses the internet and how it checks downloads is in [docs/SAFETY.md](docs/SAFETY.md).

Version 0.2.3 · engine 0.1.48

## What you need

**You need at least 32 GB of RAM and graphics memory (VRAM) together.** On a Mac or a Ryzen AI Max, that is the unified
memory. More memory gives a bigger model or a faster one.

- A 32 GB PC with a 12-16 GB card runs the Coder.
- 64 GB of RAM runs every size.

| Your computer | Details |
| --- | --- |
| **NVIDIA** GeForce RTX 20, 30, 40, 50 series (12 GB or more) | Install it from source: [Install](#install). |
| **AMD** Radeon RX 6800 and newer, Radeon AI PRO R9700 (12 GB or more) | [AMD cards](docs/AMD_HIP.md) |
| **Intel** Arc | [Intel Arc](docs/INTEL_ARC.md), [Intel engine](docs/INTEL.md) |
| **Apple Silicon Mac** (M1 or newer, macOS 15 or newer) | [Mac](docs/MAC.md) |
| **AMD Ryzen AI Max (Strix Halo)** and other unified-memory systems | [Strix Halo](docs/STRIX_HALO.md) |
| Older cards and processors | [Older GPUs](docs/OLDER_GPUS.md), [Older CPUs](docs/INSTALL.md#older-cpus-experimental) |

Also needed:

- about 80 GB of free disk (an SSD makes the first start much faster)
- Windows 10 / 11, Linux or macOS
- a current graphics driver

Two or three cards of the same make can share a model ([multi-GPU](docs/MULTI_GPU.md)). Strata never splits a model across
NVIDIA and AMD cards. Full list: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Install

Strata runs from a copy of this repository. The installer checks your PC, gets or builds the engine for your graphics
card, downloads the model you pick and starts serving.

```sh
git clone https://github.com/Niko1221/Strata
cd Strata
./setup.sh --yes                      # Linux
START-HERE.bat --yes                  # Windows
```

The first run installs what it needs (Python packages, and CMake and Ninja if the engine has to be compiled), downloads
the model (about 40-70 GB; an interrupted download continues where it stopped) and starts the server. Its
`--yes` answers the questions the setup would otherwise ask one at a time: which model, how much context, whether to
read pictures, who may connect. Without it you choose each one; Enter takes the recommended answer.

The setup compiles the engine on your PC: 5-15 minutes the first time, with a compiler it installs itself (CMake and
Ninja, and the CUDA toolkit or ROCm for your card). After that it recompiles only when the source changed.

Later runs start the model right away and download nothing twice: run `./setup.sh` (or `START-HERE.bat`) again.
Closing its window stops the model.

> While the model starts, your PC can be slow or stop responding for 1-3 minutes (longest the first time). Strata loads
> 35-55 GB into your RAM and locks part of it for the graphics card. Wait, and don't close the window.

**Mac (Apple Silicon, macOS 15 or newer):** the Metal engine is built on the Mac with cmake, and the server runs it from
a run config; the build and the run steps, and which size fits your Mac, are in [docs/MAC.md](docs/MAC.md).

Everything the setup decided - model, context, pictures, parallel slots, engine flags - is in `strata-<model>.json`, so
the same model starts again without it:

```sh
.venv/bin/python serve/server.py --engine strata --config strata-qwen.json --port 8080
```

Where the files go, what every option does and how to keep the server running: [docs/INSTALL.md](docs/INSTALL.md).

### Let your AI set it up

Paste this into Claude Code, Cursor, Codex, GitHub Copilot or another coding assistant:

```text
Set up Strata on this PC for me: https://github.com/Niko1221/Strata - follow docs/AI_SETUP.md in that repository.
```

It checks your graphics card, RAM and disk, picks the model that fits, installs and starts it, and tells you how to
connect your apps.

## Models

The setup recommends a model for your memory. It is one model in several sizes, compressed more or less.
Smaller sizes are faster. Larger sizes are a bit smarter.

### Low on memory? Try Gyro

[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) (by agentionai) is the smallest way to run this model.

| Size | Memory it needs (RAM + VRAM) | Download | What it is |
| --- | ---: | ---: | --- |
| **Gyro-S** | 27.6 GB | 58.5 GB | For PCs that fit nothing else. Quality is lower than IQ2_XS. |
| **Gyro-M** | 35 GB | 92.0 GB | About IQ3_XXS quality with about 20% less GPU memory. It writes answers 16% (12 GB RTX 3060) to 21% (R9700) slower than Q2_0. |

Gyro runs on NVIDIA and AMD cards (AMD is experimental) on Intel Arc cards on Linux (tested on an Arc A750) and, from 0.2.2, on Apple Silicon Macs (Gyro-S on a 32 GB Mac mini: 9 tok/s, streaming from the SSD; [docs/MAC.md](docs/MAC.md#gyro-s-and-gyro-m-agentionais-trellis-experts)). Details and measurements:
[docs/GYRO.md](docs/GYRO.md). Start it with `./setup.sh --model gyro-s` (`START-HERE.bat --model gyro-s` on Windows).

### The other sizes

The tiers count your RAM plus your VRAM, less 5 GB that Windows and the system keep for themselves. Every other size stays a choice.

| Your memory (RAM + VRAM) | Take | Notes |
| --- | --- | --- |
| **32 GB** | **Gyro-S**, or the **[Coder](docs/MODELS.md#coder)** for code | The Coder needs 32 GB of RAM and keeps half of the experts, so it is weaker outside code |
| **40 GB** | **[Swift 1.5](docs/MODELS.md#swift-15)** IQ2_XS (39.2 GB), or **Gyro-M** (35 GB) | Swift thinks for a shorter time before it answers. Gyro-M is close to IQ3_XXS in quality, but it writes answers 16-21% slower than Q2_0 |
| **48 GB** | **IQ3_XXS** (47.0 GB) | Slower, better |
| **64 GB** | **IQ3_S** (54.8 GB), the original or Swift 1.5 | The slowest, the best: it matches the full model on the published tests |
| **96 GB or more** | Unsloth's **UD-IQ4_XS** (about 72 GB of RAM) to **Q8_0** (about 136 GB), 4 to 8 bit | Closer to the full model. With less RAM, Strata reads part of them from the SSD, which is much slower |

Less RAM than a row says can still work: Strata switches to its
[low-RAM mode](docs/MODELS.md#a-big-graphics-card-and-little-ram) by itself. Expect it to be slower, and much slower
with a small graphics card.

More choices:

- **[Swift 1.5](docs/MODELS.md#swift-15)** is a fine-tune that thinks for a shorter time before it answers, at about the same quality and RAM.
- **[NVFP4](docs/NVFP4.md)** is NVIDIA's 4-bit version for NVIDIA cards. It needs about 80 GB of RAM to keep every expert in RAM.
- **[Qwen3.6-35B-A3B](docs/QWEN36.md)** is a smaller, different model that reads pictures and can answer several requests at once.
- **[OrcaRouter's Uncensored IQ3_XXS](docs/MODELS.md#orcarouter-uncensored-iq3_xxs)** is set up by hand.

Every size, download and what fits where: [docs/MODELS.md](docs/MODELS.md).

## How fast is it?

Based on a 12 GB graphics card (RTX 5070, Ryzen 5 7600, 64 GB RAM).
[speed of each model](docs/MODELS.md#how-fast-is-each-size).

| Model | Answers appear at | Reads your prompt at |
| --- | --- | --- |
| **Q2_0** | `████████████` 94 tokens/s | `████████████` 2,650 tokens/s |
| **IQ2_XS** | `██████████░░` 79 tokens/s | `█████████░░░` 2,090 tokens/s |
| **IQ3_XXS** | `████████░░░░` 62 tokens/s | `████████░░░░` 1,750 tokens/s |
| **IQ3_S** | `███████░░░░░` 53 tokens/s | `███████░░░░░` 1,620 tokens/s |
| **Coder** | `███████░░░░░` 55 tokens/s | `██████████░░` 2,180 tokens/s |

A token is about ¾ of a word, so 60 tokens per second is faster than you can read. "Reads your prompt" is how fast Strata
takes in a long document, code or chat history.

Measured on a 16 GB AMD card (RX 9070 XT, Ryzen 9 3900X, 47 GB RAM):

| Model | Answers appear at | Reads your prompt at |
| --- | --- | --- |
| **Q2_0** | `████████░░░░` 60 tokens/s | `█████░░░░░░░` 1,160 tokens/s |
| **IQ2_XS** | `███████░░░░░` 52 tokens/s | `█████░░░░░░░` 1,110 tokens/s |
| **Coder** | `██████░░░░░░` 44 tokens/s | `██████░░░░░░` 1,420 tokens/s |

The bars use one scale for both tables. A card with more VRAM is faster: an RTX 3090 (24 GB) should write about 100-140
tokens per second. The engine version behind each number is in [DETAILS.md](docs/DETAILS.md#speed-measured).

## Using it

### In the browser

The server's own page, at the address the setup prints, has four views: **Chat**, **Monitor** (the model and your GPU,
CPU and RAM, and the requests it is answering), **Images** (make an image) and **About** (the model and the engine, this
PC, the addresses for your tools, and Settings). The chat picks the thinking level - off, low, medium or high - reads
pictures you attach or paste, shows the answer as it is written, and stops it when you press **Stop**. Chats, the page's
settings and the API key are kept in that browser; the run config on disk keeps the settings of the model and the
engine, and the page's Settings view writes the keys it documents.

By default the server answers on `127.0.0.1` only. For your phone or another PC, start it on an address of this PC with
`--host` and keep an API key: `--api-key` (or `STRATA_API_KEY`, or `"api_key"` in the run config).
`--no-api-key` serves without one, with a warning.

### With your apps and agents

| App type | Base URL |
| --- | --- |
| OpenAI-compatible (most apps and agents) | `http://127.0.0.1:8080/v1` (any API key, any model name) |
| Anthropic API (Claude Code and others) | `http://127.0.0.1:8080`, without `/v1`. These apps add `/v1/messages` themselves. Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` |
| OpenAI Responses API (Codex CLI) | `http://127.0.0.1:8080/v1/responses` ([setup](docs/DETAILS.md#the-responses-api-and-codex-cli)) |

The **About** view shows these addresses with your port in them, ready to copy. The whole API - chat, completions,
Anthropic messages, responses, pictures, the monitor endpoint - is in
[docs/DETAILS.md](docs/DETAILS.md#using-it). An agent with a terminal installs with the commands
in [Install](#install); the steps and exit codes are in [docs/AI_SETUP.md](docs/AI_SETUP.md), and the repository rules
are in [AGENTS.md](AGENTS.md).

### Pictures

Pictures are on by default for models that can read them. Attach or paste a picture in the chat, or send it from your app.
The picture reader runs on the processor when your graphics card has less than 24 GB (a few seconds per picture, nothing
taken from the card's memory) and on the graphics card from 24 GB. The rule is the same for NVIDIA and AMD. Turn pictures
off per model, or choose the reader with `--vision off`, `--vision gpu` or `--vision cpu`.

The numbers behind the rule: on a Radeon AI PRO R9700 the card's reader took 57 ms for a 260-token picture and the
processor's took 3.1 s. On 16 GB cards (Tesla P100 and RTX A4000, Linux) the card's reader cost 3-5% of text speed,
because it holds about 1.4 GB that the model would use otherwise. On a Mac the reader runs on the Mac's GPU whatever its memory (0.45 s for a 260-token picture on an M4 with 32 GB, about 1.2 GB of the shared
memory); the processor reader needs about 50 s per picture there, so it runs only when you ask for it with `--vision cpu`.

### Good to know

- Choose the thinking level (**off, low, medium or high**) in the chat or in your app's "reasoning effort". Off is fastest.
- Strata answers one request at a time by default. For several at once, use `--parallel N` with the setup or
  `"parallel": 2` in the run config ([BATCHING.md](docs/BATCHING.md)). On a 12 GB card this makes each answer slower.
- Strata reads the first message of a chat in full, about 1 minute per 30,000 tokens. Follow-up messages start in seconds.
- Where your chats are stored and the whole API: [docs/INSTALL.md](docs/INSTALL.md#where-things-are-stored), [docs/DETAILS.md](docs/DETAILS.md#using-it).

## More speed on one card

**Two opt-in settings help when one graphics card does the work and your RAM holds every expert**, for example Q2_0 or
IQ3_XXS on a 64 GB PC (both by RubenTorney, #1904, #1905). They change how the expert cache is kept and locked. Open
`strata-<model>.json` in the Strata data folder ([where](docs/INSTALL.md#where-things-are-stored)) and add:

- `"--adapt-async", "1"` to the `"args"` list. The expert cache then updates on a helper thread instead of pausing the
  answer. Measured on an RTX 5070 (Windows, Q2_0): +4-5% decode, +7-10% on code and long documents. Answers can differ
  slightly from the default and stay the same from run to run.
- `"STRATA_PIN_BY_NEED": "1"` in the `"env"` block (Windows). Windows can lock only about half the RAM for the graphics
  card, and this locks the experts the card needs first. Measured on the RTX 5070: +4.4% prompt reading.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Restart Strata after the change. To undo, remove the two entries.

</details>

## How does it work?

Models like this one usually run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. Strata makes the model fit by sharing the work across your whole PC. Picture a library.

- The model is a team of 24,576 small specialists ("experts"), like 24,576 reference books. Each word needs only 10 of them.
- **Your graphics card is the desk.** It is small and fast. The roughly 4,000 books used most often lie open on it.
- **Your RAM is the shelves in the reading room.** All 24,576 books are there. When a word needs a book that is not on the
  desk, your processor reads it from the shelf at once, while the card keeps working.
- **Your SSD is the archive.** It holds a 29 GB lookup table. The model reads a few rows of it per word, so it does not
  need to sit in RAM.

<p align="center"><img src="docs/media/how-it-works.svg" width="860" alt="The model's 24,576 experts: the busiest on the graphics card, all of them in RAM, a lookup table on the SSD"></p>

A small helper also guesses the next few words. The big model checks all of them at once and keeps the right ones. You get
the same answer, 1.6-1.8x sooner. Long texts are read in big pieces, up to 8,192 tokens at a time, at over 1,000 tokens
per second.

<p align="center"><img src="docs/media/guess-and-check.svg" width="860" alt="A small helper guesses the next words; the big model checks them all at once and keeps the right ones"></p>

The longer explanation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). Every part and its numbers:
[the details](docs/DETAILS.md#how-it-works) and the [paper](docs/paper/Strata-Paper.pdf).

## Something went wrong?

- **My PC froze the first time Strata started.** It loads the model into RAM. Wait, and don't close the window. Still
  frozen after 10 minutes? Restart the PC, close other programs and try again, or pick a smaller size.
- **It stopped while downloading or installing.** Run `START-HERE.bat` (or `./setup.sh`) again. It continues
  where it stopped.
- **It is very slow, the disk light keeps blinking, or it says "the engine stopped unexpectedly".** Your PC does not have
  enough free RAM. Close other programs (browsers use a lot), or pick a smaller size such as Q2_0 or IQ2_XS.
- **It says the build tools are missing.** Setup offers to install them; if you would rather see what it found about
  this PC first, run `./setup.sh --check` (`START-HERE.bat --check`) before you start.
- **It says port 8080 is already in use.** Strata is already running. Look for its window.

More problems and fixes: the [troubleshooting table](docs/DETAILS.md#troubleshooting) in the details. Still stuck? Open an
[issue](https://github.com/Niko1221/Strata/issues) and attach `strata-<model>.log` from the Strata folder, with
your graphics card, its driver and what you ran. For a security problem, report it privately:
[SECURITY.md](SECURITY.md).

## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. It was compressed by
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5), agentionai (Gyro)
and Unsloth. Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits). Strata is open source under the [MIT License](LICENSE). A few parts and
every model have their own licenses ([which ones](docs/HOW_IT_WORKS.md#license)).

## Contributing

Found a bug, or want to send a change? This checkout is a personal copy of Strata - reports and changes go to
the original: [Niko1221/Strata](https://github.com/Niko1221/Strata) ([CONTRIBUTING.md](CONTRIBUTING.md)).

## Support Strata

Strata is free and MIT-licensed. I build it and test it on whatever hardware I can get. If it runs well on your PC, a coffee helps
me keep testing new cards and fixing what you report. No pressure. Using it and reporting what you find helps just as much.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>

Interested in sponsoring Strata? Contact me via [GitHub](https://github.com/Niko1221).
