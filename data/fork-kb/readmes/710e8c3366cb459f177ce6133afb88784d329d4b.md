# Strata · ds-wormhole

<sub><b>English</b> · [Español](README.es.md)</sub>

**DeepSeek-V4.1-Flash (763B) with a 1M-token context on a 64 GB mini PC.** A fork of
[Niko1221/Strata](https://github.com/Niko1221/Strata) tuned for AMD Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S `gfx1151`,
RDNA 3.5, 64 GB of unified LPDDR5X). The working branch is `ds-wormhole`.

> A personal, unofficial fork. Everything from upstream belongs to Niko1221 and the Strata contributors (MIT, see
> `LICENSE`); the DeepSeek engine comes from Project Maya (`third_party/project-maya`). Strata's original README is in
> [`README.strata.md`](README.strata.md).

## What it does

| | |
|---|---|
| Model | DeepSeek-V4.1-Flash-1017 **IQ2_XS** (ISTA-DASLab GSQ-RCO), 40 layers, 384 experts a layer, Engram IQ4_NL and the DSpark Q8_0 drafter |
| Context | **1,048,576 tokens** (FP4 KV, 890 bytes a token), vision and tools |
| Speed (real use) | **~24-27 tok/s** token-weighted (2026-11-11, P075a profile: a 95K-token web session with tools at 27.0 tok/s, a 42-task battery at 24.2 tok/s with 0 typos; the earlier H4d profile: 22.1-25.1) |
| Long context | still above 20 tok/s past 500K (a 703K session: 24.9 tok/s, 872 tool calls, 0 errors) |
| Prompt reading | 257 tok/s on average in long sessions; 340-470 tok/s with short conversations |
| Speculation | DSpark: 2.4-3.2 tokens per verify round, ~100-118 ms a round |
| Memory | ~3,100 of 15,744 routed experts resident (20 %) serve 99.6 % of the picks; the CPU lane serves 0.1-0.4 % |

The model is about 143 GB of experts plus a 111 GB Engram table, so it does not fit in 64 GB. It runs because only part
of the experts stays in memory: a missing one is **substituted under a measured penalty**, or read from the SSD
without going through the page cache. The exact route (no substitutions) runs at ~6 tok/s; production (the P075a
profile) at ~24-27, with a perplexity +6.2 % above the exact route on four human texts (the earlier H4d profile: +9.2 %,
and +32 % on prose).

| Profile (encoder / decoder penalty) | Perplexity vs exact route | Battery tok/s | Typos in 42 answers |
|---|---|---|---|
| H4d (1 / 2), until 2026-11-10 | +9.2 % (story +31.9 %) | 24.3 | 1 |
| **P075a (0.75 / 2), production** | **+6.2 % (story +16.0 %)** | **24.2** | **0** |
| 0.5 / 2 | +3.6 % (story +8.3 %) | 22.9 | 1 |
| 0.5 / 1 | +1.6 % | 19.9 | 0 |

The encoder layers (0-19) carry the quality, especially in prose; the decoder layers (20-39) take the penalty with
little loss and are where it buys speed.

## What ds-wormhole adds to the `strata-dsv41` engine

Everything is opt-in through environment variables; the production values are in
[`configs/ds-wormhole/strata-v41-iq2xs-1m-p075a.example.json`](configs/ds-wormhole/strata-v41-iq2xs-1m-p075a.example.json)
(the previous profile, H4d, is next to it).

**Unified memory (Strix Halo)**
- An expert store in pinned memory that the GPU reads in place (`DS4_STORE_GPU`, `DS4_STORE_PINNED`,
  `DS4_STORE_ZC_FRAC`, `DS4_STORE_MAPREAD`), expert slots in host memory (`DS4_SLOTS_HOST`).
- The memory "switch": the store lends its slots to the prompt read (`DS4_STORE_LEND`), and a chunk's big scratch lives
  in a lent pinned region (`DS4_CHUNK_LEND`, ~1.5 GB).
- The embedding table read from the model's mapping, with no BF16 copy (`DS4_EMBED_MAP`, −1.3 GB).

**Quality with missing experts**
- A miss penalty in the router, per layer (`DS4_ROUTE_MISS_PENALTY`, `DS4_ROUTE_PEN_LAYERS`), also while reading the
  prompt (`DS4_FLOW_PEN_MAX`, `DS4_PICK_MIN`).
- No prompt-read penalty while the engine is cold (`DS4_FLOW_PEN_WARM=N`: the first N tokens read after a start take the
  exact route), which removed a typo the first answer after a restart used to make.
- Near ties rechecked with the exact route under the speculative rule (`DS4_TIE_LOG`, `DS4_TIE_RECHECK`, and
  `DS4_TIE_MID` for ties inside a word, a line break against a candidate that goes on with the word included), with
  counters for draws from the tail and a per-position trace (`DS4_TIE_TRACE`) to find the cause of a typo.
- Experimental: a recheck of rows inside a word that lost their wanted experts (`DS4_MISS_RECHECK`, `DS4_MISS_LOG`;
  measured, not discriminative enough to use).
- A tool-name trie: a call's name can only be one of the declared tools (`DS4_TOOL_TRIE`).

**Speed**
- How many drafts to verify, chosen from the measured cost of a round (`DS4_SPEC_COST`, `DS4_ROUND_LOG`).
- Dense Q5_K GEMVs with the activation sums computed once a block in LDS (`DS4_MV_SUMS`: ~10 % less a round,
  byte-identical results).
- Router lookahead for any expert format (`DS4_AHEAD_ANYFMT`), bounded YOCO (`DS4_YOCO`).

**Server and clients**
- Conversations parked on disk, pictures included (`STRATA_DS4_SLOT_PICS`): they come back in ~1 s without encoding the
  images again.
- Repetition / frequency / presence penalties with llama.cpp's semantics.
- llama.cpp's web UI (b12032) as the chat page, with tools (`web_search`, `fetch_url`, files), MCP, attachments and
  images: [`docs/DS_WORMHOLE_WEB.md`](docs/DS_WORMHOLE_WEB.md).
- DSML tool calls read the way `deepseek-recipe` reads them; OpenAI and Anthropic APIs (Claude Code works).
- Client parameters (sampling belongs to the client, not the server):
  [`docs/DS_WORMHOLE_CLIENTES.md`](docs/DS_WORMHOLE_CLIENTES.md) (in Spanish).

## Local against the cloud

The same Tetris prompt sent to this machine and to DeepSeek's API, with screenshots and scores:
[`docs/DS_WORMHOLE_TETRIS.md`](docs/DS_WORMHOLE_TETRIS.md). Both games work and meet the prompt; the local one is weaker
in graphics (7/10 against 9/10) and took 55 minutes against 10.

## Requirements and build

- Linux (tested on Ubuntu 26.04, kernel 7.0), ROCm 7.2.4, CMake and Ninja.
- BIOS with the iGPU carve-out at its minimum (UMA 2 GB) and a large GTT on the kernel command line:
  `amdgpu.gttsize=57344 ttm.pages_limit=14680064 amdgpu.vm_fragment_size=9 iommu=pt`. Do **not** turn CWSR off
  (`amdgpu.cwsr_enable=0` causes GPU resets). Transparent huge pages without synchronous defrag.
- About 300 GB of disk for the model, the Engram table, the drafter and the vision tower.

```bash
cmake -S . -B build-dsv41-hip -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DSTRATA_BUILD_TESTS=ON -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_COMPILER=/opt/rocm-7.2.4/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm-7.2.4
cmake --build build-dsv41-hip --target strata-dsv41 -j12
```

Start it (Strata's Python server with the example profile, after filling in `<STRATA>`, `<MODELS>` and `<DATA>`):

```bash
.venv/bin/python serve/server.py --engine strata --config strata-v41-iq2xs-1m-p075a.json --host 0.0.0.0 --port 8080
```

Stop it with SIGTERM so it saves the conversation in progress.

## Qwen on Strix Halo

The branch also carries the earlier port of the Qwen engines to `gfx1151` (October 2026): MXFP4 experts, a unified-memory
arena, native Q4_0/Q8_0 MTP, HIP vision, conversations parked on disk (`--conversation-disk-dir`) and the `strata-q36`
engine with WMMA prompt attention, int8 KV, YaRN and the dense 27B models. Those are the `local:` commits in the
history; [`docs/STRIX_HALO.md`](docs/STRIX_HALO.md) is upstream's guide for this machine.

## Upstream fixes brought in (from v0.2.13.1, by cherry-pick)

- The gfx11 dense dequantizer: the old row kernel overflowed its shared staging for Q6_K (wrong rows or NaN); the new
  kernel is the default everywhere (`DS4_DQ_OLD=1` forces the old one only where it is safe), with `tests/dsv41`
  dequantizer tests.
- A router NaN guard (no out-of-range expert ids), the thread-pool generation fence from Project Maya, a fatal GPU error
  ending the engine so the server restarts it (three in ten minutes: 503), and `DS4_IQ_Q8K` (Q8_K activation rounding,
  opt-in; measured neutral here).
- Server: Anthropic context-overflow messages, a write timeout for clients that stop reading, deferred tools (tool
  search) for Claude Code, `errors="replace"` on the engine's output, a failed start that survives a concurrent close
  (#4792), and a picture-format allowlist with Pillow 13.0.0 (GHSA-78g4-xqwv-q3g8).
- Upstream's parser changes for #4701/#5040 are not taken: they broke DSML tool calls.

## Status

Work in progress. Changes are measured with batteries of real tasks (speed, validity and typos), perplexity against the
exact route on human texts, and real sessions, not short benchmarks; nothing is requantized to gain speed.
