# Strata — Apple Silicon (MLX)

A fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) that runs a **small or medium quantized model on a
Mac (Apple Silicon)** in-process through [MLX](https://github.com/ml-explore/mlx), behind Strata's server, web app
and OpenAI / Anthropic API.

It does **not** run Strata's original 125B Qwen3.8-Flash-Next model - that needs 34-50 GB of experts and a discrete
GPU. It runs a model that fits your Mac's unified memory. On a 16 GB Mac the shipped setup is
**Qwen2.5-7B-Instruct-1M-4bit** at up to about **128K context**.

## What you get

- The Strata **web app** on `http://127.0.0.1:8080`: chat, a live **Monitor** and settings.
- An **OpenAI-compatible API** (`/v1/chat/completions`), the **Anthropic Messages API** (`/v1/messages`) and the
  **Responses API** (`/v1/responses`). Point your apps and coding agents at `http://127.0.0.1:8080/v1`.
- **Tool calling**: OpenAI `tools` / `tool_calls`, Anthropic `tool_use`, and MCP servers through the web app.
- **PDF attachments** in the web app: the PDF's text is extracted (`pypdf`) and attached to the conversation;
  documents accumulate across a chat and "this document" means the newest one.
- A **menu bar icon** to open the app and start / stop / restart the model.

## What you need

| | |
| --- | --- |
| Mac | Apple Silicon (M1 / M2 / M3 / M4), macOS |
| Memory | 16 GB is the target (a smaller quantized model also runs on 8 GB) |
| Disk | ~5 GB for the model, plus the Python environment |
| Tools | Xcode Command Line Tools |

## Install and run

```bash
git clone https://github.com/migueldevops-real/Strata-Apple-Silicon-16GB.git
cd Strata-Apple-Silicon-16GB
./setup-macos.sh     # first time: makes .venv, installs mlx-lm and rumps, and starts the model
./run-macos.sh       # later starts
```

The first start downloads the model (~4.3 GB) and the web app opens at `http://127.0.0.1:8080`. Then add the menu
bar icon (starts at every login): `./install-menubar.sh` (`--uninstall` removes it).

## The model and the context

The shipped run config (`strata-qwen25-7b-1m.json`) holds the model, the context and the KV settings:

| key | meaning |
| --- | --- |
| `model` | an `mlx-community/...` id (default `mlx-community/Qwen2.5-7B-Instruct-1M-4bit`) |
| `context` | the context window in tokens (default 131072) |
| `kv_bits` | KV cache bits: `8` (default), `4`, or `null` for fp16 |
| `quantized_kv_start` | quantize the KV only from this step on (default 4096) |
| `prefill_step_size` | prompt tokens processed per step: smaller = lower peak memory |
| `warmup` | run one short generation at startup so the first chat is fast (default `false`; the examples set `true`) |

The model loads at startup (before the web app answers), so the first request does not pay for loading it.  With
`warmup` it also pays MLX's one-time Metal kernel compilation up front.

On a 16 GB Mac, **KV 8-bit at about 128K is the quality-usable ceiling**; `strata-qwen25-7b-1m-256k.json` (256K at
KV 4-bit) fits the memory but we measured it to degrade answers. A long prompt is read once and the KV of the
previous turn is kept, so a follow-up starts in seconds. Every field and the measured limits:
[docs/MACOS_MLX.md](docs/MACOS_MLX.md).

## What is not here

Images / vision, speculative decoding, native batching / parallel requests and the upstream expert cache / MTP /
n-gram / disk sessions are not implemented in the MLX backend. The native C++/CUDA/HIP engine is kept in the tree
but is not built or used on macOS.

## Tests

```bash
.venv/bin/python -m unittest serve.test_mlx_backend serve.test_extract   # the MLX backend (needs mlx-lm / pypdf)
python -m unittest serve.test_server                                     # the server suite (mock engine, no model)
```

## Credits and license

This is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) (MIT): the engine, the server, the web app
and the API are theirs. The MLX backend and the macOS setup are the addition here; the backend-registry shape
follows [jinzy0623/Strata-macOS](https://github.com/jinzy0623/Strata-macOS) (MIT) and it uses
[mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT, Apple). The models are not part of this repository; each
model's own license applies. See [ATTRIBUTION.md](ATTRIBUTION.md).

Strata is open source under the [MIT License](LICENSE).
