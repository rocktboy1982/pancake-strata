# Strata Dual-GPU

A fork of [Strata](https://github.com/Niko1221/Strata) for PCs with two NVIDIA GPUs. It can run
one model across both cards, or keep one card free for another program while the model runs on
the other.

The dual-GPU code comes from [Strata-DualGPU](https://github.com/Hardin22/Strata-DualGPU),
which was created before upstream added its own multi-GPU support. Upstream now supports
multiple cards, but its low-RAM mode is designed for one card. This fork keeps the low-RAM
mode working across a layer split and overlaps the two stages so that neither card waits for
the other.

The fork is based on Strata 0.1.38. The `main` branch of this repository tracks upstream Strata
0.2.7 without the dual-GPU changes.

## Branches

- `dualgpu` is the default branch: the dual-GPU fork described on this page.
- `main` tracks upstream Strata 0.2.7 without the dual-GPU changes.

The two branches are separate. Use `dualgpu` for a two-card setup. Use `main` only if you need
a feature that exists in upstream 0.2.7 and not in this fork.

## Three ways to use two cards

The right setup depends on what else uses the GPUs. The fork supports all three modes; they
differ only in the card list and the VRAM reserves.

Quick choice:

- The PC is dedicated to the model: use both cards (mode 1).
- ComfyUI, a game or a second model must stay comfortable: run Strata on one card (mode 2).
- Both must run at the same time: use both cards and reserve VRAM on the shared card (mode 3).

A single fast card already runs the model well. The second card adds throughput; it is not
required for the model to work. Mode 2 is a normal setup, not a degraded fallback.

### 1. Both cards for Strata

Use this when the PC is dedicated to the model. Both cards share the layers, the expert cache
and the work of each decoding step. This gives the highest throughput.

```text
cards:        both cards
layer split:  auto
pipeline:     on (default for exactly two GPUs)
best for:     maximum tokens/s, a dedicated model PC
trade-off:    no VRAM left for a second GPU program
```

Start with:

```bat
rem Windows: save the choice, then start normally
START-HERE.bat --setup --gpus 0,1
```

```sh
# Linux
./setup.sh --setup --gpus 0,1
```

The GPU numbers are the ones `nvidia-smi` prints. Setup saves the choice; later starts use it
without asking again.

### 2. One card for Strata, one card free

Use this when the second card must stay available for ComfyUI, a game, a second model or any
other GPU program. Strata runs on one card; the other card is not touched.

```text
cards:        one card
layer split:  off
pipeline:     off (it needs two cards)
best for:     stable latency and no VRAM contention
trade-off:    lower tokens/s than using both cards
```

Start with:

```bat
rem Windows: this start only, on card 1
START-HERE.bat --gpu 1
```

```sh
# Linux
./setup.sh --gpu 1
```

`--gpu N` runs this start on one card. To make it permanent, change `"gpu"` in the
`strata-*.json` config to a single number (`"gpu": 1`). `--gpus` is for a list of two or more
cards; setup rejects a one-entry list.

This is the safest mode when another program needs a card. The model still benefits from the
low-RAM mode: the experts that do not fit in VRAM are copied into system RAM instead of being
read from the SSD.

### 3. Both cards for Strata, with VRAM reserved on one card

Use this when both cards can help the model, but one card must also keep room for another
program. The model uses both cards, and the shared card keeps a fixed amount of VRAM free.

```text
cards:        both cards
layer split:  auto
pipeline:     on
best for:     more speed than one card, while another program still runs
trade-off:    less expert cache on the shared card; the second program may still slow the model
```

Put the card that must keep room for the other program first in the card list, then reserve its
VRAM. The reserve flags follow the position in the list:

- `--vram-reserve-mib N` reserves VRAM on the first card.
- `--vram-reserve-later-mib N` reserves VRAM on the later cards.

The automatic order puts the fastest card last. If the shared card is the slower or larger-VRAM
card, it usually ends up first, so `--vram-reserve-mib` is the flag to use. Check the log line
that prints the final card order.

Example: card 0 is the shared card and is the slower or larger-VRAM card, card 1 is the
dedicated fast card.

```bat
rem Windows: the shared card is first, so the first-card reserve applies to it
START-HERE.bat --setup --gpus 0,1 --vram-reserve-mib 7000
```

```sh
# Linux
./setup.sh --setup --gpus 0,1 --vram-reserve-mib 7000
```

The number is the VRAM the other program needs, plus some headroom. Start with the other
program's normal usage and add 1 to 2 GB. If the model fails to allocate its cache, lower the
reserve; if the other program still runs out of VRAM, raise it.

If the shared card is later in the final order, put `--vram-reserve-later-mib 7000` in the
config's `args` list instead. To pin the order yourself, set `"gpu_order": "as_given"` and
list the shared card first in `"gpu"`.

Shared mode is a compromise. A program that uses the same card heavily can still reduce the
model's speed, because both compete for memory bandwidth and compute. If latency matters more
than throughput, use mode 2 instead.

## Choosing the cards

The cards do not have to be identical. The fork handles different VRAM sizes and different
speeds:

- **Fastest card last.** The last stage runs the output head and the draft layer, so the server
  puts the fastest card last when `layer_split` is `auto`.
- **More VRAM, more layers.** The split search gives a card more layers and a larger expert
  cache when it has more free VRAM. The card order itself follows speed, not VRAM size.
- **Mixed pairs are expected.** A 12 GB + 16 GB pair, a 16 GB + 24 GB pair and two equal cards
  all work. The automatic split is the recommended starting point for every pair.
- **Three or more cards.** A layer split works, but the pipelined two-stage loop is only for
  exactly two cards. With three or more, the stages take turns.

Run this first to see the indexes and the card properties. After setup, the engine is in the
`engine` folder:

```bat
engine\strata.exe --list-gpus
```

```sh
# Linux
./engine/strata --list-gpus
```

Use the numbers it prints in `--gpu`, `--gpus`, `"gpu"` and the reserve flags.

## What this fork changes

- **Low-RAM mode on a layer split.** Each card keeps a cache of its own hottest experts; the
  rest are copied into system RAM. The model does not read experts from the SSD while it
  answers.
- **Pipelined windows.** The first card starts the next decoding step while the second card is
  still checking the current one. This is enabled by default when exactly two GPUs are used.
- **Faster card last.** The last stage also runs the output head and the draft layer, so the
  server orders the cards by speed when the split is automatic.
- **Per-card layer weights.** Each card loads only the dense weights of the layers it runs. The
  freed VRAM is used for that card's expert cache.
- **Asynchronous expert swaps.** Moving experts between VRAM and RAM no longer stops decoding
  while the copy runs.
- **Fewer host waits.** The stages synchronise through device-side flags, the draft chain runs
  in one CUDA graph, and RTX 50 cards use programmatic dependent launch.
- **Defaults for the host.** On hybrid Intel CPUs the expert pool uses the P-cores and part of
  the E-cores; the card that drives the monitors keeps more VRAM free.

The measured effect of each change, and every switch, is documented in
[docs/DUAL_GPU.md](docs/DUAL_GPU.md). Those measurements come from the original author's
machine and are kept as provenance, not as a promise for another pair of cards.

## Requirements

- Two NVIDIA GPUs for the dual-GPU modes. One NVIDIA GPU is enough for the single-card mode.
- 32 GB of system RAM minimum. 64 GB is recommended: it allows a larger expert cache, a larger
  context and more room for other programs.
- Windows 11 for the ready-made engine. Linux is supported by the source build, but this fork
  was not tested on Linux.
- An NVIDIA driver and CUDA runtime compatible with the engine build.
- Enough disk space for the model files. The recommended model is Qwen3.8-Flash-Next
  Swift 1.5 IQ2_XS; larger quantisations need more VRAM and RAM.

AMD GPUs are not supported by this fork.

## Install on Windows

1. Download this repository, or clone it:

   ```sh
   git clone https://github.com/piwi-kiwi/Strata.git
   cd Strata
   ```

2. Double-click `START-HERE.bat`.
3. Answer the setup questions. Setup asks which cards to use. Choose both for mode 1 or 3, one
   for mode 2. Choose the low-RAM mode the setup recommends for the model.
4. When setup finishes, open the address it prints in a browser, or connect an application to
   the OpenAI-compatible API.

Setup downloads the model and the engine. The ready-made Windows engine is published in this
repository's [releases](https://github.com/piwi-kiwi/Strata/releases/latest). The fine-tuned
draft layer is available from the original
[Strata-DualGPU release](https://github.com/Hardin22/Strata-DualGPU/releases/latest). If a
download is interrupted, run `START-HERE.bat` again; it continues where it stopped.

To use an existing Strata installation, unpack this repository next to it and run
`START-HERE.bat`. Strata shares its data folder between copies on the same PC, so the model is
not downloaded again.

## Install on Linux

```sh
git clone https://github.com/piwi-kiwi/Strata.git
cd Strata
./setup.sh
```

The source build compiles the engine for the local CUDA version. This path was not tested on
Linux by the author of the fork. See [docs/INSTALL.md](docs/INSTALL.md) for the general
requirements and troubleshooting.

## Build from source

The engine is a C++/CUDA program. A source build needs CMake, Ninja or Visual Studio, a CUDA
toolkit, and a C++ compiler.

On Windows, the setup script can build the engine. For a manual build, use the same CMake
options as upstream Strata; the upstream build instructions are in
[README.upstream.md](README.upstream.md) and [docs/INSTALL.md](docs/INSTALL.md).

## Configuration

The setup writes a configuration file named `strata-*.json`. The settings that matter for a
two-card run are:

- `"gpu"`: one card number (`1`) or a list of cards (`[1, 0]`). The list is the card order the
  engine sees; the last card is the last stage.
- `"layer_split": "auto"` chooses the split point from the free VRAM and the card speeds. A
  number such as `"18"` is the first layer of the later card's share; use it only when you
  chose that split for this exact order.
- `"gpu_order": "auto"` lets the server put the fastest card last. `"as_given"` keeps the order
  in `"gpu"`.
- `"vram_reserve": "auto"` lets the server reserve VRAM for a card that drives a display.
  `"as_given"` keeps the reserves in the arguments unchanged.

Example: both cards, automatic order and split, with the low-RAM copy.

```json
{
  "gpu": [1, 0],
  "layer_split": "auto",
  "gpu_order": "auto",
  "vram_reserve": "auto",
  "args": ["--resident-experts"]
}
```

The exact file also contains the model paths and the other launch settings written by setup.
Edit only the keys above unless you know what the other settings do.

### Useful engine flags

| Flag | What it does |
| --- | --- |
| `--gpus 0,1` | Use these cards together. The last one is the last stage. |
| `--gpus all` | Use every card that can share the model. |
| `--gpu 1` | Use one card only. |
| `--layer-split auto` | Choose the split from free VRAM and card speed. |
| `--layer-split K` | Put the later card's first layer at K. Use with a fixed order. |
| `--pipeline-windows 2` | Overlap the two stages. Default on exactly two GPUs. |
| `--pipeline-windows 1` | Overlap the prompt windows only. |
| `--pipeline-windows 0` | The stages take turns. |
| `--resident-experts` | Keep the expert copy in system RAM instead of reading from the SSD. |
| `--vram-reserve-mib N` | VRAM kept free on the first card. |
| `--vram-reserve-later-mib N` | VRAM kept free on the later cards. |
| `--no-trim-stage-weights` | Every card keeps the dense weights of all the layers. |
| `--adapt-async 0` | Use the blocking expert-swap tier. |
| `--list-gpus` | Print the visible cards and exit. |

`--resident-budget-gib N` is for the experimental Unsloth UD-Q4_K_XL model only. It sets the
number of GiB of that model's experts kept in RAM. Do not use it for Q2_0 or IQ2_XS; those
models use `--resident-experts` or `--mmap-experts` instead.

### Suggested starting points

These are starting points, not universal rules. The engine log prints the split, the card
order and the reserves it actually used. On Windows, add `--setup` to `START-HERE.bat` to save
a choice; `--gpu N` applies to one start only.

| PC | Mode | Starting flags |
| --- | --- | --- |
| 2 cards, dedicated | both for Strata | `--gpus 0,1` |
| 2 cards, second card free | one card | `--gpu 1` (or the faster card) |
| 2 cards, second card shared | both, reserve on shared card | `--gpus 0,1 --vram-reserve-mib 7000` |
| 32 or 64 GB RAM | both for Strata, low-RAM copy | `--gpus 0,1 --low-ram resident` |
| 64 GB RAM, longer context | both for Strata | `--gpus 0,1 --low-ram resident --context 65536` |
| unequal pair, 12 GB + 16 GB | both for Strata | `--gpus 0,1 --layer-split auto` |
| model fits on one card | one card | `--gpu N` and no layer split |

Pass these flags to `START-HERE.bat --setup` on Windows or `./setup.sh --setup` on Linux.

With 64 GB, the low-RAM mode can keep more experts in RAM than on a 32 GB PC. The model's own
fit check and the engine log say whether the copy fits. If Windows starts swapping, reduce the
context, use a smaller quantisation or leave one card free for the system.

## Tuning and measuring

Change one thing at a time and keep the same prompt and the same context size. Useful signals:

- the engine log's split line: which layers are on which card, and the card order;
- the expert-cache line: how many expert pairs are in VRAM and how many come from RAM;
- the decode speed in the server or the benchmark output;
- Windows Task Manager or `nvidia-smi` for VRAM and GPU use;
- free RAM and page-file use. A full page file is worse than a smaller expert cache.

If one card is idle during decoding, the split is wrong for that pair. Try `--layer-split auto`
again, or sweep a fixed split (`18`, `20`, `22`, ...) with `"gpu_order": "as_given"`. If both
cards are busy but the speed is low, the CPU expert pool or the PCIe link may be the limit.

The pipeline is a throughput feature. It helps most when both cards are dedicated to the model.
It is not a latency guarantee, and it is off when only one card is used.

## Performance

There is no benchmark for every card pair in this README, and there should not be: the result
depends on the cards, the PCIe layout, the CPU, the RAM, the model, the context and the expert
cache.

The original Strata-DualGPU author measured 143 tok/s on code, 102 tok/s on prose and 1,940
tok/s on a 32K prompt with an RTX 5080 + RTX 4060 Ti, 32 GB RAM and Swift 1.5 IQ2_XS. Those
numbers belong to that machine. They are kept in [docs/DUAL_GPU.md](docs/DUAL_GPU.md) to show
what the changes did there. They are not a prediction for a 5070 + 5060 Ti, a 64 GB PC or any
other pair.

Measure your own PC before and after a change. A useful comparison is the same model, the same
context, the same prompt and the same draft settings, with one card and with two.

The output is not bit-for-bit identical to upstream. Experts computed on the CPU and experts
computed on the GPU round differently, so a near-tie between two tokens can choose the other
token. The fork's documentation includes a perplexity comparison with upstream.

## Limitations

- Pipelining requires exactly two GPUs. With three or more cards the split still runs, but
  without pipelining.
- The fork is based on Strata 0.1.38. It does not contain later upstream changes.
- The fork was tested by its original author on one Windows machine with a 5080 + 4060 Ti.
  Other pairs, Linux and AMD GPUs were not tested by that author.
- Shared mode cannot guarantee the speed of either program. Two programs on one card compete
  for the same memory bandwidth.
- The ready-made draft layer in the releases was trained for Swift 1.5 IQ2_XS. With another
  model it still produces correct answers, but it may be slower than the stock draft layer.
- Like upstream with adaptive expert swaps, the same prompt can produce slightly different
  text from one run to the next at near-ties.

## Upstream, credits and license

This repository is a fork of [Hardin22/Strata-DualGPU](https://github.com/Hardin22/Strata-DualGPU),
which is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata).

The Strata engine, server, setup, web application and documentation are the work of Niko1221
and the Strata contributors. The dual-GPU changes and the benchmarks in
[docs/DUAL_GPU.md](docs/DUAL_GPU.md) are from the Strata-DualGPU fork. The upstream README is
kept as [README.upstream.md](README.upstream.md).

Strata is released under the MIT license. See [LICENSE](LICENSE). Model files have their own
licenses; check the model provider's page before redistributing them.
