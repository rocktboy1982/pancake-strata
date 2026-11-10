# kervia-Strata

[![CPU checks](https://github.com/kerviaHerve/kervia-Strata/actions/workflows/ci.yml/badge.svg)](https://github.com/kerviaHerve/kervia-Strata/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A community fork of **[Niko1221/Strata](https://github.com/Niko1221/Strata)** focused on measurable inference improvements for **two NVIDIA GPUs**. The reference platform is **two RTX 5070 Ti cards, with 16 GB of VRAM each**.

Strata's engine, multi-GPU support, OpenAI-compatible server, vision and tool calling come from upstream. This fork preserves the original history, MIT license and attribution. See [upstream provenance](UPSTREAM.md) and the [original README](README.upstream.md).

## Status and scope

The initial fork adds a reproducible two-GPU configuration profile, a synthetic benchmark client, CPU regression checks and a development roadmap. **It does not yet change CUDA kernels or claim a speed improvement over upstream.**

Our priorities are low latency for coding agents, efficient use of both cards, stable **131,072-token context**, native image input and reliable tool calls. Linux with two NVIDIA cards is the initial validation target. Existing upstream platforms remain in the source tree; they are not all validated by this fork's CI.

## Reference platform

| Component | Reference |
| --- | --- |
| GPUs | 2 × NVIDIA RTX 5070 Ti, 16 GB VRAM per card (32 GB aggregate) |
| Interconnect | PCIe 5.0 x8 and x4; no NVLink |
| CPU / system RAM | Ryzen 9 7900 / 64 GB RAM, separate from GPU memory |
| Model | Qwen3.8-Flash-Next, GSQ-RCO IQ3_XXS, complete expert set |
| Context / KV | 131,072 tokens / INT8, 32,768 resident tokens |
| Execution | Automatic layer split, one active request, speculative decoding |

The model also uses substantial **system RAM**. The historical reference run used about **46.3 GiB of process RSS**, in addition to VRAM. Two 16 GB cards do not make this a 32 GB system-RAM configuration. See [hardware and setup](docs/kervia/DUAL_NVIDIA.md).

## Get started

Use a separate checkout and a free port when another model server is running. Building and loading a model consume significant CPU, RAM and GPU resources.

```bash
git clone https://github.com/kerviaHerve/kervia-Strata.git
cd kervia-Strata
git remote add upstream https://github.com/Niko1221/Strata.git
./setup.sh --check
```

Then follow the [two-NVIDIA setup guide](docs/kervia/DUAL_NVIDIA.md). The guide uses `--build` so the engine comes from this checkout. No fork-specific binary release is published yet; upstream's default prebuilt downloads are still upstream artifacts.

For an installation already prepared by setup, inspect the proposed profile without writing anything:

```bash
python3 tools/kervia_profile.py --config strata-iq3_xxs.json
```

Create a **new** local configuration and start it when the GPUs are available:

```bash
python3 tools/kervia_profile.py --config strata-iq3_xxs.json \
  --output strata-dual-nvidia.local.json
.venv/bin/python serve/server.py --engine strata \
  --config strata-dual-nvidia.local.json --host 127.0.0.1 --port 8080
```

The API base URL is `http://127.0.0.1:8080/v1`, and the profile's model name is `kervia-strata`. The context limit includes input and generated output. Configure authentication before permitting remote access; see [Security](SECURITY.md).

## Measurements

Historical measurements on the reference machine, **before any fork changes**, used three 768-token outputs and uncached synthetic French prompts:

| Backend | Median decode, tokens/s | First token, 32,030 input tokens | First token, 120,029 input tokens |
| --- | ---: | ---: | ---: |
| Upstream Strata, Flash Next IQ3_XXS | 130.92 | 9.18 s | 30.49 s |
| llama.cpp, Flash Next IQ3_XXS | 29.58 | 58.05 s | 234.14 s |
| llama.cpp, Qwen3.8 27B Turbo | 115.33 | 22.92 s | 105.57 s |

These are observations on one machine, not quality scores, single-GPU scaling results or promised performance. The Turbo row uses a different model. The [measurement record](bench/results/kervia/2026-10-07-reference.json) and [method and limitations](docs/kervia/BENCHMARKING.md) explain the workload, cache policy and memory costs. The new English benchmark is a different workload and must establish its own baseline.

## Development

- [Roadmap and acceptance criteria](docs/kervia/ROADMAP.md)
- [Contributing and CPU checks](CONTRIBUTING.md)
- [Benchmark methodology](docs/kervia/BENCHMARKING.md)
- [Upstream multi-GPU architecture](docs/MULTI_GPU.md)
- [Original installation and API documentation](README.upstream.md)
- [Security reporting](SECURITY.md) · [Code of conduct](CODE_OF_CONDUCT.md)

New fork documentation, issues and pull requests use English. Upstream translations are retained with their original content.

## License and credit

[MIT](LICENSE). Copyright and original authorship remain credited to Niko1221 and the Strata contributors. Third-party components retain their own notices. Model weights are downloaded separately and have their own licenses. The inherited GitHub Sponsor link supports **upstream Strata**, not this fork's maintainer.
