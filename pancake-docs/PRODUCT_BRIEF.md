# pancake-strata: product brief (2026-11-11)

## What it is
pancake is a fork of [Strata](https://github.com/Niko1221/Strata), an inference engine that runs very large mixture-of-experts models
(here Qwen3.8-Flash-Next, 125B parameters, about 6B active per token) on one consumer PC: the experts live in system RAM, the hot ones
in the GPU's memory, and a small draft head (MTP) plus a suffix/prompt-lookup drafter propose several tokens per step so each pass
through the model commits more than one token. pancake tunes that engine for one machine class: an **AMD RX 7900 XT (RDNA3, gfx1100)
with a desktop CPU and about 57 GB of RAM**, serving agent workloads (long prompts, code edits, tool calls) through an OpenAI- and
Anthropic-compatible local API.

## Who it is for
People who want a large model running fully local on a single AMD gaming GPU, and who care most about how fast a long prompt turns
into the first token (agent turns re-read a lot of context) and about steady decode speed for code.

## What it measured on the target machine (same client for every engine, same model files, 2 passes, warm pass kept)
Model: Qwen3.8-Flash-Next IQ2_XS, 128K context. Decode in tokens per second, prompts in tokens per second.

| | pancake `fast-hip` | Strata-rdna3 (a gfx1100 Strata fork) | Avalon (llama.cpp fork) |
| --- | ---: | ---: | ---: |
| decode, mixed prompts | 104.8 | 106.5 | 108.5 |
| decode, code prompts | 132.3 | 116.5 | 126.1 |
| decode, prose prompts | 98.8 | 97.5 | 102.2 |
| prompt of 3.2K tokens | 1775 | 1623 | 1531 |
| prompt of 12.7K tokens | 2612 | 2432 | 1583 |
| prompt of 25K tokens (time to first token) | 2776 (9.1 s) | 2667 (9.5 s) | 1617 (15.7 s) |

- **Long prompts are the clear win:** 1.6-1.7x faster than Avalon at 12-25K tokens, from the RDNA3 prompt kernels (WMMA attention,
  chunked GDN scan, fused prefill, a tuned hipBLASLt table). Without that switch stack (the exact `hip` preset) prompt speed falls to
  about 0.8K tokens per second.
- **Decode is on par with the other engines** (differences under about 5% are noise on this machine); pancake leads on code, trails on prose.
- **Q2_0 and IQ2_XS** quantizations decode at about the same speed.
- Full data, all presets and caveats: `bench/results/2026-11-11-vs-avalon/REPORT.md`.

## Presets
| preset | what it does | status |
| --- | --- | --- |
| `fast-hip` (default) | the measured fast stack, expert deferral to the CPU on; not bit-exact with `hip` | recommended |
| `hip` | exact, without the speed switches | reference only (about 0.8K prompt tok/s) |
| `turbo-hip` | fast plus a router prior that favours resident experts (lossy) | not recommended: -4..-8% decode overall, helps only copy-heavy code |

## What is in progress (MOD_PLAN Phase 17)
1. The prose gap to Avalon (a larger draft head and always drafting two tokens look likely; unconfirmed).
2. A greedy ablation over about 20 coded-but-untested speed items, judged by a paired test with a 2% bar.
3. A route prior that switches on only in copy-heavy windows.
4. A depth-adaptive KV ring for contexts of 64K and up (coded, builds, unit test passes; speed not yet measured).
5. A draft vocabulary built from real code in 38 languages: held-out misses fall from 4.3% to 1.6% of the model's output tokens.

## Honest limits
- One machine, one GPU model; results on NVIDIA or other AMD cards will differ.
- Speed only: answer quality was not compared across engines or quantizations.
- The exact engine is not run-to-run deterministic, so text identity cannot be used as an exactness check.
- Avalon's published CUDA-first changes do not help on this card; Avalon does not run the Q2_0 file in the tested build.

## Credits
Strata (Niko1221 and contributors), Strata-rdna3 (nicejunjie), Avalon (freriadoc), ISTA-DASLab (the quantized model files), and the
other projects listed in README section 9.
