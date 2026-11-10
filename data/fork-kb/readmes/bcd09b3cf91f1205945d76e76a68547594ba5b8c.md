# SSD Expert Supply

**English** · [한국어](README.ko.md)

**An engine research project for supplying MoE experts from SSD when they do not all fit in RAM.**

Built on [Strata](https://github.com/Niko1221/Strata), this project investigates expert delivery across SSD, DRAM, and VRAM. The goal is to reduce the time from requesting expert weights to having them ready for computation, and to improve real prefill and decode performance.

The starting point is earlier work on RAM_ADAPT, which adapts RAM expert placement to the conversation, and experiments recording inference time and storage reads. The research extends to file layout, concurrent reads, DRAM buffers, caching, prefetch, and overlap between computation and I/O.

> The project is currently establishing its research foundation. The performance numbers below come from earlier RAM_ADAPT experiments. They are not current performance results for a new engine or a claim of improvement on every machine.

The engine source on this `main` starts from Strata commit `fb58e0db`, dated October 8, 2026. The functionality incorporated in v0.2.7 and the local instrumented v0.2.9 configuration described below have not yet been synchronized to this branch. This change introduces the project through documentation. The earlier installation and usage instructions are preserved in [the upstream README copy](README.upstream.md).

## The problem

A MoE model computes only a subset of experts for each token, but the next token may select different experts. On a PC that cannot hold all weights in RAM and VRAM, the remaining experts must be supplied from SSD.

Sequential SSD bandwidth alone does not determine inference speed. Read size, concurrency, reuse of loaded data, and the time computation spends waiting all matter.

We measure and improve these parts together:

| Area | Research scope |
|---|---|
| File layout and reads | Expert storage units, merging adjacent reads, asynchronous I/O, and concurrent requests |
| DRAM buffers | Allocation, copying, synchronization, and reuse of loaded buffers |
| Expert retention | DRAM/VRAM placement, admission and eviction costs, and cache interference between prefill and decode |
| Prefetch and execution | Read priorities, prediction costs, and overlap between reads, transfers, and computation |

The evaluation centers on **time until requested experts are ready for computation, and actual inference speed**.

## Work so far

### RAM_ADAPT improvement

When the conversation repeatedly uses an expert missing from RAM, RAM_ADAPT replaces a less-used RAM expert to reduce future SSD reads. The improved path copies weights from a preceding read's stage buffer when they are still available, avoiding another file read, and allows up to four pairs to move per layer.

This work was incorporated into Strata v0.2.7 through [PR #3060](https://github.com/Niko1221/Strata/pull/3060). It is available upstream and is not presented as a feature exclusive to this fork.

The following **v0.2.1-based experiments** replayed 120 ShareGPT requests in order, with matched conditions within each model. Each setting was measured in three fresh engine runs.

| Model | RAM_ADAPT off | Before the improvement, `=8` | Stage-buffer improvement, `=1` | Gain over the previous implementation |
|---|---:|---:|---:|---:|
| IQ2_XS decode | 37.70 tok/s | 48.71 tok/s | **51.40 tok/s** | **+5.5%** |
| IQ3_XXS decode | 23.24 tok/s | 33.07 tok/s | **38.71 tok/s** | **+17.1%** |

Environment: RTX 3060 12 GB, 32 GB RAM, Linux, NVMe SSD, 20.5 GiB RAM expert budget, K8V4, MTP 3/T3, and asynchronous adaptation. Every run started from the same stock expert profile, with the model's page cache dropped. Actual VRAM expert slots were 3,769 for IQ2 and 2,791 for IQ3. Comparisons are between settings within the same model.

On the `standard-v1` request replay, additional decode gains over the previous `=8` implementation were +1.6% for IQ2 and +3.9% for IQ3, indistinguishable from run variation. That replay used `--greedy` and is not a conforming `strata bench run` standard result. These measurements do not establish a uniform gain across short requests and long conversations.

Input construction and per-run measurements are available in the [replay kit](https://github.com/SanghunYun/Strata/tree/ram-adapt-replay) and the [PR's measurement description](https://github.com/Niko1221/Strata/pull/3060).

### Observability and reproducibility

- Record prefill and decode sections, and check the residual between total time and the named sections.
- Optional instrumentation records each decode window's time, expert reads, and RAM moves. [PR #3631](https://github.com/Niko1221/Strata/pull/3631)
- Existing experiment tools restart the engine, restore initial expert profiles, replay workloads for each setting, and restore configuration and profiles afterward.

The current local reference engine is Strata v0.2.9 with existing observability instrumentation. Build and deployment smoke checks have passed; a current performance comparison on representative workloads is the next step.

## Next experiments

1. Measure the same current baseline twice to establish variation and the timing breakdown.
2. Separate SSD reads, DRAM copies, buffer waits, and overlap with computation.
3. Build an expert-delivery benchmark that records and replays actual expert read requests.
4. Change one major bottleneck at a time and verify it in both delivery measurements and the real engine.

The initial development target is Linux with NVIDIA GPUs and Qwen3.8-Flash-Next. Performance and compatibility on other hardware and models will be reported after measurement on those configurations.

## Measurement principles

- Match the model, computation settings, and memory budget in before/after comparisons.
- Report prefill and decode separately, and distinguish short and long inputs.
- Distinguish logical expert reads, process storage reads, and device traffic.
- Retain source and binary hashes, effective arguments, actual input lengths, cache state, initial expert placement, and individual run results.
- Report failed requests and regressions. Cache hit rate or sequential read bandwidth alone does not establish an inference improvement.

## Build and run

Public installation and experiment instructions for the research fork are being prepared. For now, see the [upstream Strata README](https://github.com/Niko1221/Strata#readme). Installing upstream Strata does not automatically enable this project's research configuration.

Validated build, run, and benchmark commands will be added here after the public branch and reproduction artifacts are organized and checked in a fresh environment.

## Relationship to Strata and contributions

This is a Strata-derived research project. It builds on the upstream engine and its contributors' work, and distinguishes upstream functionality, earlier contributions, and new changes.

Follow [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md). The contribution guide was brought over from upstream v0.2.9, with its documentation links pointing to that version. Check the [upstream contribution guide](https://github.com/Niko1221/Strata/blob/main/CONTRIBUTING.md) for updates.

- Search existing issues and PRs before upstream submissions, and agree scope with maintainers before substantial features, architecture changes, or default changes.
- Keep new engine features opt-in and verify preservation of the existing path's behavior.
- Separate passed, failed, and not-run checks. Validate affected backends; keep an upstream PR draft when required environments are unavailable.
- Remove private paths, inputs, and credentials from public artifacts, and supply reproducible evidence.

Preserve the upstream [MIT license](LICENSE), copyright notices, and [contributor credit](docs/HOW_IT_WORKS.md#credits), along with the licenses and notices of included third-party code.
