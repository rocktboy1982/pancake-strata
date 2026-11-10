# 🥞 pancake-strata

**A fast local LLM engine, built one layer at a time.**

Why *pancake*? A good stack of pancakes is built one at a time: each one goes on top of the last, warm, and the stack
only gets better as it grows. This project is built the same way. It starts from **[Strata](https://github.com/Niko1221/Strata)**,
a fast mixture-of-experts engine for home GPUs. On top of it go the newest speed techniques the open-source
community has published, one after another: an engine from FastStrata, kernels from llama.cpp, ideas from basalt,
colibri, SGLang and others, and our own measured kernels. Every layer is checked before it goes on the stack: it has
to be **faster on a real benchmark** and keep the **same answers**. The result is a full load of pancakes: a stack of
the latest LLM inference technologies, each adding speed.

The target is single-user agent serving (long, growing prompts and tool calls) on ordinary hardware: an
**RTX 3060 12 GB** (CUDA) and an **RX 7900 XT 20 GB** (HIP), a Ryzen 7 5700G and about 57 GB of RAM.

The long-form documents live in the documentation folder [`pancake-docs/`](pancake-docs/):

- [`pancake-docs/RESEARCH_AND_FINDINGS.md`](pancake-docs/RESEARCH_AND_FINDINGS.md): the full research record, the paper
  catalog, the engineering decisions, and the test and benchmark ledger.
- [`pancake-docs/STRATA_DERIVATIVES_DEEP_RESEARCH.md`](pancake-docs/STRATA_DERIVATIVES_DEEP_RESEARCH.md): deep research
  on every Strata fork and derivative, with a gap analysis.
- [`MOD_PLAN.md`](MOD_PLAN.md): the master plan, item by item, with the "not adopted" ledger.
- [`PANCAKE_STATUS.md`](PANCAKE_STATUS.md): what is built, what is tested, and what comes next.
- [`docs/WINDOWS_CUDA_SM75.md`](docs/WINDOWS_CUDA_SM75.md): the Windows + CUDA 12.8 build on a GTX 1660 Ti (sm_75).

## 2. What `pancake-strata` does compared to Strata and other forks (Measured Results)

Every number below is a real run on the named card: greedy decoding, cold start, quiet machine unless the note says
otherwise. "Exact" means the same tokens as stock decoding, checked token for token. Sources:
[docs/PANCAKE_FINDINGS.md](docs/PANCAKE_FINDINGS.md), `bench/results/`, and MOD_PLAN phase 11.

### Table 2A: RX 7900 XT 20 GB (HIP, gfx1100)

Models: **Qwen3.8-Flash-Next** (176.9B total: 125B MoE + 51B PLE + 4B MTP) and **Qwen3.6-35B-A3B**.

Decode speed on Qwen3.8-Flash-Next (writing speed). The rows do **not** all use the same quant: read the
quant and bits columns before comparing speeds.

| Engine | Quant (expert weights) | Bits per expert weight | tok/s |
| --- | --- | ---: | ---: |
| Strata (stock) | IQ2_XS | ~2.31 | 75-77 |
| Avalon | IQ3_KS / IQ3_K hybrid | ~3.2-3.4 | 95-100 |
| Strata-rdna3 | IQ2_XS | ~2.31 | 109-119 |
| **pancake-strata** (exact / fast-hip) | IQ2_XS | ~2.31 | **112-126** |

Note on the Avalon row: it is **not** the same file.  Avalon (a llama.cpp-based engine) was measured on the same PC
with its own Flash-Next GGUF, a hybrid IQ3_KS / IQ3_K layout (~89 GiB, ~3.2-3.4 bits per expert weight), while the
Strata rows use the IQ2_XS file (~2.3 bits per expert weight).  Decode on this box is bound by reading expert bytes
(CPU-side experts from DDR4, GPU-side from VRAM), so Avalon moves roughly 40-50% more expert bytes per token and gets
a higher-quality model for it.  The row is "what each stack delivers on this card with its own recommended file", not
an engine-vs-engine comparison at equal quantization; that comparison (Avalon on IQ2_XS, or Strata on an IQ3 file) has
not been run.

The pancake-strata gain over its own starting point, measured in the final suites: **+6% on chat** (111.7 to
118.4 tok/s) and **+13% on code edits** (92.5 to 104.3 tok/s) with `--pcie-frac 0`. On the Q2_0 file the same preset
reaches **135 tok/s**. The opt-in lossy `turbo-hip` preset (cache-aware routing, see section 6) reaches **149-168 tok/s**
and passes the token-forcing quality gate with 2.9% of tokens differing from exact decoding (stock run-to-run noise is
1.2-1.3%, with no large-margin errors).

Long-context prompt reading, `feat/bfla-hip` on Qwen3.6-35B-A3B UD-Q4_K_XL (block-sparse prompt attention; "dense" and
"sparse" are the two BFLA modes):

| Prompt | off | dense | sparse | gain |
| --- | ---: | ---: | ---: | ---: |
| 32K, single layer | 1238.7 ms | 581.1 ms | **500.2 ms** | 2.5x |
| 64K, single layer | 5224 ms | 2321 ms | **1090 ms** | 4.8x |
| 128K, single layer | 28287 ms | 9553 ms (3.0x) | **3335 ms** | 8.5x |
| 128K, end-to-end prompt throughput | 386.9 tok/s | 771.2 tok/s (2.0x) | **1219.8 tok/s** | 3.15x |

Quality held: 48/48 top-1 needle-recall agreement with the exact path, KL 0.0014/0.0045.

### Table 2B: RTX 3060 12 GB (CUDA, sm_86)

Qwen3.6-35B-A3B (UD-Q4_K_XL with its MTP layer), stock `strata` vs `strata-poly`:

| Workload | Strata (`strata`) | pancake-strata (`strata-poly`) | Gain |
| --- | ---: | ---: | ---: |
| Short chat, writing speed | 64.6 tok/s | **100.7 tok/s** | x1.56 |
| Code edit (5.3K prompt), writing speed | 40.1 tok/s | **151.1 tok/s** | x3.8 |
| 42K-token prompt: reading | 42.1 s | **32.9 s** | x1.28 |
| 42K-token prompt: writing speed after it | 9.7 tok/s | **71.7 tok/s** | x7.4 |
| Largest context on the 12 GB card | 64K (the 131K setup default runs out of VRAM) | **131K** | |

Qwen3.8-Flash-Next on the same 12 GB card: **50-55 tok/s** short chat, **74 tok/s** code edits, and **41-47 tok/s**
writing after a 42K prompt.

### Table 2C: Feature-by-Feature Comparison against the Ecosystem

| Feature | Strata | FastStrata | Strata-rdna3 | Avalon | basalt | Project Maya | pancake-strata |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Hybrid CPU/GPU MoE execution | Yes (base) | - | - | - | - | - | Yes (extended: k-th-miss tier, deferral, seeds) |
| Second engine for long context (`strata-poly`) | - | Yes (source) | - | - | - | - | Yes (ported + tuned) |
| RDNA3 wave32 kernels (gfx1100) | - | - | Yes (source) | - | - | - | Yes (ported + fused) |
| Specialized output-head kernels (Q5_K / IQ4_XS) | - | - | - | - | Yes (source) | - | Yes (ported, default on) |
| 2-bit MTP draft proxy | - | - | - | - | Yes | - | Yes (ported, default on) |
| Grammar-constrained decoding | - | - | - | - | Yes | - | Yes (ported, `--grammar`) |
| Agent loop guard, soft budget close, tolerant tool calls | - | - | - | - | - | Yes (source) | Yes (ported) |
| Conversation disk cache + suffix replay | - | Yes | - | - | - | - | Yes (in `strata-poly`) |
| hipBLASLt table match + 7900 XT preset tuning | - | - | - | Yes (report) | - | - | Yes (in the `-hip` presets) |
| Block-sparse prompt attention (BFLA), 3.15x at 128K | - | - | - | - | - | - | Yes (`feat/bfla-hip`) |
| Learned stop rule for drafts (DISCO) | - | - | - | - | - | - | Yes (`feat/disco`) |
| Expert-cache admission research (k-th miss, swap budget, deferral rank guard) | - | - | - | - | - | - | Yes |
| Quality gates + 30-script measurement suite | - | - | - | - | - | - | Yes (`tools/pancake/`) |

## 3. Research on Open-Source Projects & Forks (What We Took as Code or Idea, and Licenses)

Before writing a line we surveyed the whole Strata fork ecosystem and the major open-source inference engines. The
rule for everything in the table: code and ideas come in only through a measured A/B and, for anything lossy, a
token-forcing quality gate.

| Project (with URL) | Author / Org | License | Taken as | What `pancake-strata` took (or why rejected) |
| --- | --- | --- | --- | --- |
| [Strata](https://github.com/Niko1221/Strata) | Niko1221 | MIT | Code (base) | The base engine: hybrid CPU/GPU MoE execution, MTP, expert cache, app and terminal. Everything above sits on it. |
| [FastStrata](https://github.com/Jarson1408/FastStrata) | Jarson1408 | MIT | Code | The `strata-poly` engine (engine, CUDA/AVX2 kernels, conversation disk cache, suffix replay), ported and tuned. |
| [PolyStrata](https://github.com/simplexai-labs/PolyStrata) | simplexai-labs | MIT | Code | Same lineage as FastStrata's poly engine; ported together with it. |
| [Strata-rdna3](https://github.com/nicejunjie/Strata-rdna3) | nicejunjie | MIT | Code | All RDNA3 wave32 decode and prefill kernels (`rdna_gr.cu`, `rdna_mmvq.cu`, `route_row.cuh`), hand-ported 1,938 upstream commits after their base. Not ported, with reasons: long-context top-k (upstream `STRATA_SELECT_LONG` covers it), PF_GROUP_RES (upstream group-resident), PA_KLDS (upstream rewrote it as `STRATA_PA_RPF`), RDNA_PAIR (neutral), GDN megakernel (0.0 ms on one GPU). |
| [basalt](https://github.com/jesdga95/basalt) | jesdga95 | MIT | Code | Q5_K/IQ4_XS output-head kernels (`q5k_head`, `head_fmt`), the 2-bit MTP draft proxy (`head_proxy`), the grammar path, idle-sleep ideas. |
| [Project Maya](https://github.com/mw00/project-maya) | mw00 | MIT | Code | Loop guard (`serve/loop_guard.py`), soft reasoning-budget close (#613), tolerant tool-call parser, prompt memo. |
| [Speedy](https://github.com/DevDock-AI/Speedy) | DevDock-AI | MIT | Idea | The "fast" preset switch stack (which switches to stack on HIP and in what order). |
| [colibri](https://github.com/JustVugg/colibri) | JustVugg | Apache-2.0 | Evaluated | Rejected as code; the CACHE_ROUTE discussion informed our routing-cache reasoning. |
| [tandem.cpp](https://github.com/ewhacc/tandem.cpp) / [tandem.cpp-amd](https://github.com/itlab-cl/tandem.cpp-amd) | ewhacc / itlab-cl | MIT | Code | `STRATA_POOL_PREREAD`: idle pool workers pre-read the lookahead's next-layer CPU experts. Ported verbatim. |
| [StrataFast](https://github.com/eddoursul/StrataFast) | eddoursul | MIT | Evaluated | Two-stage prompts (`--prompt-split`, 1.24-1.55x on long prompts) and self-trained draft layer; `--prompt-split` needs a second GPU (parked for Phase 12). |
| [Aviary](https://github.com/SensAI-PT/aviary-hy3) | SensAI-PT | Apache-2.0 | Evaluated | Distributed expert placement across nodes; surveyed, single-machine focus here. |
| [llamAmpere](https://github.com/JakeATX/llamAmpere) | JakeATX | MIT | Idea | Head and kernel specialization ideas for consumer NVIDIA cards; cross-checked our head work. |
| [gpu-unlock](https://github.com/Zagitta/gpu-unlock) | Zagitta | GPL-2.0 | Idea | Power and VRAM tuning context for the 7900 XT presets. |
| [Avalon](https://github.com/freriadoc/Avalon) | freriadoc | MIT | Idea / Evaluated | Its 48-commit delta and our audit gave the gfx1100 hipBLASLt 100401 table match (+38% prompt, in the 7900 XT presets) and `--lookup-max-t 12`. |
| [qwen38-megakernel](https://github.com/cutlerbenjamin1-cmd/qwen38-megakernel) | cutlerbenjamin1-cmd | AGPL-3.0 | Evaluated (ideas only, license) | Megakernel and draft-vocab experiments studied; no code taken (AGPL-3.0), and our own GDN megakernel prototype measured 0.0 ms on one GPU. |
| [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) | ggml-org | MIT | Code | The quantization formats (Q2_0, IQ2_XS, Q5_K, IQ4_XS), GGUF, and the MMVQ kernel lineage the RDNA3 ports build on. |
| [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | ikawrakow | MIT | Idea | DFlash2 (PR 2761) calibrated survival window, now `STRATA_DF2_GATE` / `STRATA_DF2_ROW_MS`. |
| [KTransformers](https://github.com/kvcache-ai/ktransformers) | kvcache-ai | Apache-2.0 | Idea | The rank guard behind `STRATA_VERIFY_DEFER_CPU` (heavy CPU entries kept on time in their own layer, item 28). |
| [SGLang](https://github.com/sgl-project/sglang) | sgl-project | Apache-2.0 | Idea | RadixAttention-style prefix caching (item 11 family). |
| [vLLM](https://github.com/vllm-project/vllm) | vllm-project | Apache-2.0 | Evaluated | Datacenter, batch>1 orientation; not a fit for single-user decode. Its scheduler work informed the overlap model. |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | NVIDIA | Apache-2.0 | Evaluated | Closed datacenter kernels, not portable to HIP/RDNA3; used as a ceiling reference. |
| [FlashInfer](https://github.com/flashinfer-ai/flashinfer) | flashinfer-ai | Apache-2.0 | Evaluated | Attention kernel shapes informed the BFLA port; no direct code. |
| [BitDecoding](https://arxiv.org/abs/2503.18773) | paper (see section 4) | - (paper) | Evaluated | `feat/bitdecoding-hip` ported the decode-attention idea to gfx11 matrix cores (`STRATA_ATTN_WMMA`); measured neutral, not adopted. |
| [ISTA-DASLab](https://huggingface.co/ISTA-DASLab) | ISTA-DASLab (HF) | per model | Evaluated | IQ2_XS / Q2_0 packs for Flash-Next; the default quant recommendation. |
| [Unsloth](https://huggingface.co/unsloth) | Unsloth (HF) | per model | Evaluated | UD-Q4_K_XL / UD-Q3_K_M packs for Qwen3.6-35B-A3B, used in the daily presets. |
| [Qwen](https://huggingface.co/Qwen) | Qwen (Alibaba) | Apache-2.0 (models) | Evaluated | The models themselves: Qwen3.8-Flash-Next and Qwen3.6-35B-A3B; the tech report was checked against code. |

### 3A. Fork knowledge base (`tools/forks/fork_kb.py`)

The fork survey above is not a one-off: `tools/forks/fork_kb.py` keeps an incremental knowledge base of every direct
fork of [Niko1221/Strata](https://github.com/Niko1221/Strata), so the ecosystem can be re-read at any time and new
work in the forks shows up on the next run.

What one run does:

1. Lists every direct fork through the GitHub API (one call per 100 forks).
2. Queries GitHub again only for forks that are new or whose `pushed_at` changed since their last successful scan
   (`FULL=1` rescans all of them). For each one it stores the commits ahead of upstream (`compare` API) and the
   README.
3. Writes `runs/<run>-new-commits.csv` with only the commits first seen in this run, and `runs/<run>-errors.csv`.
   A fork that failed is not marked as scanned, so the next run retries it.

Upstream rewrote its history around early October, so 974 older forks share no commit with it and GitHub's
`compare` answers "No common ancestor". For those the script takes the commits on the fork's branch dated after the
fork was created, then marks as `upstream_history = 1` the ones that are really old upstream commits the fork had
synced (authored by the upstream owner, or present in two or more such forks). Analyses should filter on
`upstream_history = 0`; the new-commits CSV already does.

The knowledge base is committed in this repo under [`data/fork-kb/`](data/fork-kb/) (`KB_DIR` changes the place):

| Path | Contents |
| --- | --- |
| `forks.sqlite.gz` | The committed snapshot of the database (18 MB compressed). The script rewrites it at the end of every run and unpacks it into `forks.sqlite` when that file is missing, so a fresh clone can run an update directly. |
| `forks.sqlite` | Not in git (`.gitignore`); the working copy. Table `runs` (one row per run with counts), `forks` (branch, `pushed_at`, ahead/behind, last scan, last error), `commits` (one row per fork and commit, with `first_seen_run` and `last_seen_run`; a `last_seen_run` older than the fork's last scan means the commit was force-pushed away or merged), `readmes` (README per fork, with the run in which it last changed). |
| `readmes/<blob_sha>.md` | README contents, stored once per distinct content (most forks share upstream's). |
| `runs/` | The per-run CSVs above (the very large first one is gzipped). |

Run and update it (the first run took 70 minutes, including one wait for the GitHub limit of 5,000 calls per hour;
an update only touches forks that changed: the next run rescanned 22 forks in under a minute):

```bash
GITHUB_TOKEN=$(gh auth token) python3 tools/forks/fork_kb.py          # first run, and every update after it
GITHUB_TOKEN=$(gh auth token) FULL=1 python3 tools/forks/fork_kb.py   # rescan every fork
GITHUB_TOKEN=$(gh auth token) LIMIT=20 python3 tools/forks/fork_kb.py # smoke test on 20 forks
git add data/fork-kb && git commit -m "fork-kb: update" -- data/fork-kb   # keep the committed snapshot current
```

Each update adds a new ~18 MB snapshot to the git history, so update when there is something to analyze (for example
weekly), not on every small change.

Other settings: `KB_DIR` (where the knowledge base lives), `WORKERS` (parallel requests, default 5), `UPSTREAM`
(default `Niko1221/Strata`). Run it at low priority (`nice -n 19`) while a benchmark is running.

Analyze it (Python's built-in `sqlite3` is enough; the `sqlite3` command-line tool is optional):

```bash
python3 - <<'PY'
import sqlite3
db = sqlite3.connect("data/fork-kb/forks.sqlite")   # run from the repo root after one update, or gunzip the snapshot
# commits that appeared in the latest run
for r in db.execute("""select repo, substr(date,1,10), substr(message,1,90) from commits
                       where upstream_history = 0 and first_seen_run = (select max(id) from runs)
                       order by date desc"""):
    print(*r, sep="  ")
# the forks with the most unique work
for r in db.execute("select repo, ahead_by, pushed_at from forks where ahead_by > 0 order by ahead_by desc limit 30"):
    print(*r, sep="  ")
PY
```

The new-commits CSV of each run is the input for the next round of fork notes (see the
[November fork report](reports/Active%20Strata%20forks%20November.md)); anything adopted from it follows the
credit rule in section 9.

First results (2026-11-10, report: [Strata fork knowledge base](reports/Strata%20fork%20knowledge%20base.md)):

| Measure | Value |
| --- | ---: |
| Direct forks | 4,550 (21 return 404 on `compare`, retried every run) |
| Forks with their own commits | 407 (340 diverged, 66 of the 974 without a common ancestor, 1 ahead) |
| Own commits in total | 43,685 |
| Forks with own commits since 2026-11-01 / 2026-11-08 | 140 (2,957 commits) / 60 (666 commits) |
| Distinct READMEs | 167 (80 MB knowledge base in total) |

Activity peaked in the week after launch (71 forks, ~20,900 commits, week of 2026-09-28) and has slowed since (110
forks in the last full week). Since 11-01 the commit messages are mostly about the server and agent layer (75 forks),
the expert cache and offload (71), quantization (55), KV and attention (57), CUDA (54), speculative decoding and MTP
(51) and HIP/ROCm (40 forks, 238 commits).

## 4. Academic Research: Papers, Ideas, and Direct URLs

Before building we read. The survey covered **758 arXiv papers across 7 global university regions** (131 US, 97 Asia,
76 Stanford, 69 Europe, 65 MIT/Harvard, 63 Pacific/Africa/LatAm, 56 Canada), plus focused studies on MoE offload,
speculation, attention, and agent control (66 hardware data-path papers, 31 runtime-architecture papers). The full
per-paper URL lists live in the mirrored `reports/*.md` files and in
[`pancake-docs/RESEARCH_AND_FINDINGS.md`](pancake-docs/RESEARCH_AND_FINDINGS.md). The tables below list every paper
that shaped a decision, with its direct URL and what happened to it. Every rejection is also recorded, with its
reason, in the MOD_PLAN "not adopted" ledger.

### 4A. MoE expert offload, caching and routing

| Paper / System | Authors / Venue / Year | Direct URL | Mechanism | Status in `pancake-strata` |
| --- | --- | --- | --- | --- |
| Fiddler | 2024 | [arxiv.org/abs/2402.07033](https://arxiv.org/abs/2402.07033) | CPU executes missed experts, activations over PCIe | Adopted (confirms Strata's hybrid design; context for the pre-sum items 50-52) |
| MoE-Infinity | 2024 | [arxiv.org/abs/2401.14361](https://arxiv.org/abs/2401.14361) | Expert-level cache for personal machines | Adopted (design context for the expert cache) |
| ProMoE | 2024 | [arxiv.org/abs/2410.22134](https://arxiv.org/abs/2410.22134) | MoE routing prediction | Evaluated (informed the lookahead) |
| ExpertFlow | 2024 | [arxiv.org/abs/2410.17954](https://arxiv.org/abs/2410.17954) | Expert caching and pinning | Evaluated |
| FATE | 2025 | [arxiv.org/abs/2502.12224](https://arxiv.org/abs/2502.12224) | Cross-layer gate prediction (97.15%) + shallow-favoring cache | Adopted (router lookahead and cache-priority reasoning) |
| Pre-Attention Expert Prediction | 2025 | [arxiv.org/abs/2511.10676](https://arxiv.org/abs/2511.10676) | Predict experts before attention runs | Evaluated |
| Apto | Zhong, Wang, Chen (2026) | [arxiv.org/abs/2611.01859](https://arxiv.org/abs/2611.01859) | Self-adaptive CPU/GPU dual-ended expert queue, affinity-aware scheduling, cancellable tail racing (2.96x prefill / 1.52x decode) | Evaluated (informs hybrid CPU/GPU queue and tail-straggler design) |
| Apex | Lv et al. (2026) | [arxiv.org/abs/2610.23006](https://arxiv.org/abs/2610.23006) | Adaptive expert assignment: high-token-load missed experts transferred to GPU, low-load run on CPU (+51.9% prefill, +39.5% decode) | Evaluated (informs adaptive CPU vs GPU miss assignment) |
| APEX | 2025 | [arxiv.org/abs/2506.03296](https://arxiv.org/abs/2506.03296) | Profiles CPU vs GPU subtask times to schedule async CPU-GPU execution (+84-96% on T4) | Evaluated (context for our per-layer overlap model) |
| KTransformers | SOSP'25 | [github.com/kvcache-ai/ktransformers](https://github.com/kvcache-ai/ktransformers) | CPU/GPU rank guard for MoE | Adopted (idea behind `STRATA_VERIFY_DEFER_CPU`, item 28) |
| AcceptMoE | 2026 | [arxiv.org/abs/2608.02989](https://arxiv.org/abs/2608.02989) | Depth-graded expert demand per verify row | Adopted (`STRATA_ROUTE_PRIOR_DEPTH`, item 23) |

### 4B. Speculative decoding

| Paper / System | Authors / Venue / Year | Direct URL | Mechanism | Status in `pancake-strata` |
| --- | --- | --- | --- | --- |
| DraftExpert | 2026 | [arxiv.org/abs/2607.24434](https://arxiv.org/abs/2607.24434) | Miss-aware draft windows | Tested: our DraftPolicy is already optimal |
| Limits of Speculation | 2026 | [arxiv.org/abs/2609.22156](https://arxiv.org/abs/2609.22156) | Draft-length saturation | Informed DISCO and the tail-row analysis (item 5) |
| EVICT | 2026 | [arxiv.org/abs/2605.00342](https://arxiv.org/abs/2605.00342) | Draft-tree truncation | Informed DISCO |
| EcoSpec | 2026 | [arxiv.org/abs/2607.12696](https://arxiv.org/abs/2607.12696) | Energy-aware speculation | Informed the tail-row analysis |
| SpecFit | 2025 | [arxiv.org/abs/2610.15417](https://arxiv.org/abs/2610.15417) | Marginal new-miss cost | DISCO feature context (`feat/disco`) |
| SuffixDecoding | CMU, 2024 | [arxiv.org/abs/2411.04975](https://arxiv.org/abs/2411.04975) | Cross-session suffix index | Planned (item 15) |
| SuffixDrafter | 2025 | [arxiv.org/abs/2506.02780](https://arxiv.org/abs/2506.02780) | Suffix drafting | Planned (with SuffixDecoding) |
| DSpine | Tencent, 2026 | [arxiv.org/abs/2609.36173](https://arxiv.org/abs/2609.36173) | Parallel drafter | Rejected: no drafter exists for Qwen3.6-35B; watching for a release |
| Medusa | 2024 | [arxiv.org/abs/2401.10774](https://arxiv.org/abs/2401.10774) | Multi-head drafting | Covered by DraftPolicy's cost-aware window (no port) |
| Sequoia | 2024 | [arxiv.org/abs/2402.12374](https://arxiv.org/abs/2402.12374) | Tree drafting | Covered by DraftPolicy |
| TurboSpec | 2024 | [arxiv.org/abs/2406.14066](https://arxiv.org/abs/2406.14066) | Hardware-aware speculation | Covered by DraftPolicy |

### 4C. Attention and long context

| Paper / System | Authors / Venue / Year | Direct URL | Mechanism | Status in `pancake-strata` |
| --- | --- | --- | --- | --- |
| BFLA (block-sparse flash attention) | ported work | `feat/bfla-hip` (in-repo) | Block-sparse prompt attention | Prototype on branch: 3.15x end-to-end at 128K, 48/48 needle recall |
| XAttention | 2025 | [arxiv.org/abs/2503.16428](https://arxiv.org/abs/2503.16428) | Block-sparse attention, antidiagonal scoring | Planned (item 19, lossy opt-in) |
| SpargeAttn | Tsinghua, 2024 | [arxiv.org/abs/2411.10958](https://arxiv.org/abs/2411.10958) | Training-free sparse attention | Planned (item 19) |
| BitDecoding | 2025 | [arxiv.org/abs/2503.18773](https://arxiv.org/abs/2503.18773) | Low-bit KV, decode on matrix cores | Tested on `feat/bitdecoding-hip`: neutral, not adopted |
| LServe | MIT Han lab, 2025 | [arxiv.org/abs/2502.14866](https://arxiv.org/abs/2502.14866) | Long-context serving with sparse KV | Evaluated (KV design context) |
| ShadowKV | 2024 | [arxiv.org/abs/2410.21465](https://arxiv.org/abs/2410.21465) | Low-rank KV + offload | Evaluated (KV context) |
| DuoAttention | 2024 | [arxiv.org/abs/2410.10819](https://arxiv.org/abs/2410.10819) | Head-level KV eviction | Evaluated (KV context) |
| Quest | 2024 | [arxiv.org/abs/2406.10774](https://arxiv.org/abs/2406.10774) | Query-aware KV sparsity | Evaluated (KV context) |
| Mamba-2 / SSD | 2024 | [arxiv.org/abs/2405.21060](https://arxiv.org/abs/2405.21060) | Chunked linear attention | Adopted (GDN chunk-64 prefill line, item 18) |
| Mamba-3 | 2026 | [arxiv.org/abs/2603.15569](https://arxiv.org/abs/2603.15569) | Linear attention evolution | Adopted (same line) |

### 4D. Quantization and CPU kernels

| Paper / System | Authors / Venue / Year | Direct URL | Mechanism | Status in `pancake-strata` |
| --- | --- | --- | --- | --- |
| T-MAC | Microsoft, EuroSys'25 | [arxiv.org/abs/2407.00088](https://arxiv.org/abs/2407.00088) | LUT mpGEMM for 1-4-bit weights | Rejected for rewrite: our CPU kernel already runs at ~80% of the 33 GB/s DDR4 allows; LUT gains ~0 |
| TEAL | MIT, 2024 | [arxiv.org/abs/2408.14690](https://arxiv.org/abs/2408.14690) | Activation sparsity in FFNs | Tested and closed: Flash-Next experts are dense (gate cos 0.844 at 70%) |
| CATS | Stanford, 2024 | [arxiv.org/abs/2404.08763](https://arxiv.org/abs/2404.08763) | Context-aware activation sparsity | Tested and closed: cos 0.818, failed |
| Berkeley Intra-Expert Sparsity | UC Berkeley, 2025 | [arxiv.org/abs/2605.08575](https://arxiv.org/abs/2605.08575) | Gate-mask sparsity inside experts | Tested and closed (same bench, oracle 0.95 vs gate 0.84) |
| SpAx | 2025 | [arxiv.org/abs/2610.02598](https://arxiv.org/abs/2610.02598) | Sparse-axis kernels | Parked (4th sparsity mask) |
| Scaling Laws for Precision | MIT, 2024 | [arxiv.org/abs/2411.04330](https://arxiv.org/abs/2411.04330) | Bit allocation across weights and KV | Adopted as guidance for quant choices |
| QTIP | 2024 | [arxiv.org/abs/2406.11235](https://arxiv.org/abs/2406.11235) | Trellis quantization with incoherence processing | Rejected: not worth a requant vs IQ2_KT |
| QuIP# | 2024 | [arxiv.org/abs/2402.04396](https://arxiv.org/abs/2402.04396) | Incoherence-processing quantization | Rejected: same reason |

### 4E. Serving, caching and agent control

| Paper / System | Authors / Venue / Year | Direct URL | Mechanism | Status in `pancake-strata` |
| --- | --- | --- | --- | --- |
| RadixAttention / SGLang | 2023 | [arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104) | Radix-tree prefix cache | Adopted (idea; item 11 family) |
| Marconi | Princeton, MLSys'25 | [arxiv.org/abs/2411.19379](https://arxiv.org/abs/2411.19379) | FLOP-per-byte eviction, checkpoint triggers | Planned (item 16) |
| FreeToken | 2026 | [arxiv.org/abs/2608.16157](https://arxiv.org/abs/2608.16157) | Speculation-aware scheduling | Planned (checkpoint-trigger ideas, item 16) |
| KVFlow | 2025 | [arxiv.org/abs/2507.07400](https://arxiv.org/abs/2507.07400) | Expert pre-staging | Coded (item 17, `STRATA_IDLE_ADAPT=N`, `43e741bc6`): when no request line arrives for 150 ms, the blocking tier runs up to N extra rounds at 4x swaps on the idle GPU |
| ADRS / Barbarians at the Gate | UC Berkeley, 2025 | [arxiv.org/abs/2510.06189](https://arxiv.org/abs/2510.06189) | LLM-driven cache-policy search | Planned (item 14, stage 2) |
| SIEVE | NSDI'24 | [github.com/cacheMon/NSDI24-SIEVE](https://github.com/cacheMon/NSDI24-SIEVE) | Simple cache eviction | Candidate (only if >=5% of decode is at stake) |
| Dynasor / Certaindex | Hao AI Lab, 2024 | [arxiv.org/abs/2412.20993](https://arxiv.org/abs/2412.20993) | Thinking early exit | Superseded by DISCO (learned stop) |
| ThinkPrune | 2025 | [arxiv.org/abs/2504.01296](https://arxiv.org/abs/2504.01296) | Reasoning-length pruning | Evaluated (effort budgets cover the need) |
| Plan and Budget | 2025 | [arxiv.org/abs/2505.16122](https://arxiv.org/abs/2505.16122) | Budgeted reasoning | Evaluated (informed `reasoning_budget_by_effort`) |
| CodeSteer | 2025 | [arxiv.org/abs/2502.04350](https://arxiv.org/abs/2502.04350) | Steering between code and text | Evaluated (agent corpus) |
| Mind the Gap | ETH, 2025 | [arxiv.org/abs/2505.23786](https://arxiv.org/abs/2505.23786) | GGUF-quantization backdoors | Adopted as security practice: official files + SHA-256 only |
| Displacement Is Not Direction | Toronto, 2026 | [arxiv.org/abs/2606.19558](https://arxiv.org/abs/2606.19558) | KLD vs top-1 drift measurement | Adopted: revised quality-gate method |

## 5. How `pancake-strata` Works Under the Hood & Why We Chose These Paths

### The anatomy of a decode window

On consumer hardware every generated token is one window with three cost buckets. Profiled on the RX 7900 XT (Qwen3.8-Flash-Next):

- **A code-edit window takes 37.4 ms**: 15.8 ms is the GPU floor (kernels that run no matter what), 17.4 ms is CPU
  expert misses (the Ryzen 7 5700G pulls expert rows from DDR4 at about 80% of the 33 GB/s the memory allows), and
  2.5 ms is the MTP draft step.
- **A short-chat window takes 24.8 ms**: 15.9 ms GPU + 5.2 ms CPU + 2.2 ms draft.

Every optimization in this project targets exactly one of these three buckets. The GPU floor shrinks with kernel
ports and fusion (section 7, items 2-3), the CPU bucket shrinks with cache admission, deferral and overlap
(items 4-5), and the draft bucket shrinks with the head proxy (item 4). Nothing else in the window matters.

### The 7 core architectural choices, and why

1. **Exact-by-default, lossy behind flags.** Every exact switch is checked token for token against stock decoding
   before it ships; lossy switches (`turbo`, deferral, the DF2 gate) are opt-in and must pass a token-forcing quality
   gate. Why: a single-user agent lives on trust, and an exact baseline makes every A/B meaningful. The alternative,
   shipping lossy defaults, would have made every later measurement ambiguous.
2. **Scheduling and overlap over kernel rewrites.** The CPU Q2_0 kernel already runs at ~80% of DDR4 bandwidth, so
   LUT tricks (T-MAC) and activation sparsity (TEAL, CATS, the Berkeley gate mask) have nothing to give; all were
   measured and rejected. Overlap instead attacks the 17.4 ms bucket: k-th-miss admission (`STRATA_ADAPT_KMISS`),
   deferral with a rank guard (`STRATA_VERIFY_DEFER_CPU`), fused pool phases. The overlap model prices deferral at
   +33% on edit windows, deferral + kmiss 2/8 at +45-52%.
3. **GPU router lookahead instead of CPU lookahead threads.** The router GEMV is tiny and the GPU is idle while it
   waits for CPU rows anyway. Moving the lookahead onto the GPU measured +10% decode on mmap'd experts and removed a
   thread from the CPU pool. A CPU lookahead thread (and the `STRATA_ARENA_WARM` variant) measured worse: it evicted
   the L3 the pool needed.
4. **Per-format output heads + a 2-bit draft proxy.** The output head is 540 MB in Q8_0 (337 MB in IQ4_XS), so it is
   a real cost at every draft step. Specialized Q5_K/IQ4_XS head kernels (`q5k_head`, `head_fmt`, basalt ports) plus a
   two-stage proxy pick (`head_proxy`: score all candidates from a 2-bit proxy, rescore only the near-best) make each
   MTP draft step about 12% cheaper with the 59.6K draft vocab.
5. **Dual-engine architecture: `strata` + `strata-poly`.** Stock `strata` keeps an fp32 KV cache and no expert cache
   in its long-context path: 9.7 tok/s after a 42K prompt on the 3060. FastStrata's poly engine, with q8 KV and a
   conversation disk cache, reaches 71.7 tok/s on the same prompt. Keeping both engines lets each model family run on
   its best path instead of forcing one engine to serve both.
6. **BFLA on `qwen35`, QSA on Flash-Next.** The two engines have different attention paths. BFLA block-sparse prompt
   attention ports cleanly onto the qwen35 engine (3.15x end-to-end on 128K prompts, 48/48 needle recall), while
   Flash-Next keeps its own exact sparse-select path. Porting BFLA everywhere would have risked the daily model for a
   gain it cannot offer.
7. **Server-level agent reliability: loop guard, grammar redraw, effort budgets.** opencode session history showed
   turn p99 of 224 s driven by re-reads (~18% of turns) and uncapped thinking, not by tok/s. So the server ports
   Project Maya's loop guard (periodic-block detection, collapse, tool-repeat blocking), adds grammar-constrained
   redraw (`--grammar` + `src/program/grammar/`), effort-graded reasoning budgets with a soft close, and `turn_at`
   checkpoints. Wall-clock per turn is what the user feels.

## 6. Complete Guide to Extra Switches (`STRATA_*` Env Vars, CLI Flags, Server Options & Presets)

Everything below is additive vs upstream Strata v0.2.12. Unless marked lossy, every switch is exact (the same tokens).
The census: 65 new `STRATA_*` variables on the main line, plus the side-branch families and the server-side set.

### 6A. RDNA3/HIP kernels (ported from Strata-rdna3, the gfx1100 half)

| Switch | Defined | Default | Exact/Lossy | Does | Measured |
| --- | --- | --- | --- | --- | --- |
| `STRATA_GR_RDNA` | `rdna_gr.cu` | off | exact | RDNA3 wave32 hyper-connection down/up read kernels | part of the 87-101 to ~112 tok/s port |
| `STRATA_GR_RDNA_Q8` | `rdna_gr.cu`, `native_dense.cpp`, `weights.hpp` | off | lossy | int8 image of hyper-connection projections | in the fast-hip preset |
| `STRATA_GR_RDNA_A8` | `rdna_gr.cu` | off | lossy | int8 activations + int8 dot4 in the Q8 read | in the fast-hip preset |
| `STRATA_GR_RDNA_ONLY` | `rdna_gr.cu` | 0 | bench | 1 = down only, 2 = up only | profiling |
| `STRATA_GR_RDNA_RW` | `rdna_gr.cu` | 2 | tuning | rows per wave | tuning |
| `STRATA_GR_INJ_UP` | `rdna_gr.cu` | off | exact | inject rows' slice dots in their own blocks (no float copy) | in the fast-hip preset |
| `STRATA_GR_INJ_CHECK` | `rdna_gr.cu` | off | debug | inject-row debug buffer | debug |
| `STRATA_GR_FUSED` | `rdna_gr.cu` | off | exact | down+up passes in one launch with a grid barrier | in the fast-hip preset |
| `STRATA_GR_Q32_PL` | `fused_gr.cu` | off | tuning | Q3_2 path variant | tuning |
| `STRATA_RDNA_Q6K` | `rdna_mmvq.cu` | off | exact | gfx1100 multi-column Q6_K decode GEMV | in the fast-hip preset |
| `STRATA_RDNA_MMVQ_WAVE` | `native_mmvq.cu` | unset | exact | wave-shaped multi-row GEMV (preset uses 3) | in the fast-hip preset |
| `STRATA_RDNA_KQ_ALL` | `rdna_mmvq.cu` | off | tuning | allow the KQ kernel below 4 columns | tuning |
| `STRATA_MMVQ_R` | `native_mmvq.cu` | 1 | exact | rows per wave (2/4), shared activation loads | tuning |
| `STRATA_IQ_SIGN_FAST=1` (or a type list, e.g. `22`, `16,22`) | `iq_kernels.cu` (`iq_signed`, MOD_PLAN F12; RX5950XT `b9a2111d3f`, boxwrench in yqYo1 `6f0bdf665b`) | off | exact (bitwise; both GPUs) | the EXPDEC gate/up kernels (`expdec_gu`, and `expdec_gu_swiglu` with `STRATA_EXPDEC_SWIGLU`) put the IQ2_XXS / IQ2_XS / IQ2_S grid signs on with 7 plain 32-bit ops (no multiply) instead of the emulated byte compare + byte subtract (~13 ops and a multiply on HIP); needs `STRATA_EXPDEC=1`. Other kernels keep the old form (native_gu_multi loses waves with it, fork static count) | coded on `work/f12-sign-expand`, not built or measured yet; parity in `iq_multi_parity`, per-format A/B `expdec_bench ... def,dec:s0,dec:s1`, screen `bench/specs/f12-sign-fast.json`. Fork: -29% (IQ2_XXS gate/up) / -35% (IQ2_S down) kernel time on an RX 7900 XT in a GLM engine; est. 0-1.5% decode here |
| `STRATA_PCIE_OFF` | `generate.cpp` | off | tuning | force the PCIe share off | Round A: +6/+13% with `--pcie-frac 0` |

### 6B. Expert cache, routing and deferral

| Switch | Defined | Default | Exact/Lossy | Does | Measured |
| --- | --- | --- | --- | --- | --- |
| `STRATA_ADAPT_KMISS=k,W` | `kmiss_tier.hpp` (item 14) | off | exact | tier admits an expert after k misses in W windows (2,8) | online +5..16% edit (sim); run with `--adapt-async 1` |
| `STRATA_ADAPT_SWAPS_AUTO` | `swap_budget.hpp` (item 53) | off | exact | swaps per round chosen online in [x,96] | built, A/B queued |
| `STRATA_VERIFY_DEFER_CPU` (+`_BELOW=<w>`) | `verify.cpp`, `expert_source.cpp` (item 28, MOD_PLAN A3) | off (on in fast-hip; `_BELOW` 0.1) | **lossy** | a layer's light CPU expert rows (routing weight < w) join the residual one layer late; the heavy ones stay on time | 7900 XT, test phase 01: +15.8% +-3.9 edit, short chat n.s., token forcing PASS; in the fast-hip preset |
| `STRATA_VERIFY_DEFER_CPU_TOP=<n>` | `defer_rank.hpp` (item 28 rank guard, A4) | off | **lossy** | a CPU entry among its token's top n by weight is also kept on time | 01, `_TOP=2` vs fast-hip base: +20.1% +-3.3 edit but -6.9% +-3.5 short chat, token forcing PASS; opt-in |
| `STRATA_VERIFY_DEFER_CPU_FOLD=<n>` | `defer_rank.hpp`, `expert_source.cpp` (A4 fix) | off | exact vs the same `_TOP`/`_BELOW` run | a layer with heavy CPU rows and at most n light experts runs one pool pass instead of two (the same rows still join late) | coded on `work/a4-rank-guard`, screen pending (`bench/specs/10-rank-guard-fold.json`) |
| `STRATA_ROUTE_PRIOR_DEPTH=a` | `route_prior.hpp` (item 23, AcceptMoE-style) | off | lossy | residency bonus graded by window position | coded, probes clean |
| `STRATA_DOMAIN_PROFILES=<dir>` | `expert_seed.hpp`, `server.py` (item 4) | off | exact | per-domain expert profiles seeded per request kind | needs per-domain traces |
| `STRATA_CONV_WORKSET=N` / `_TOKENS=T` | `expert_seed.hpp` (item 3) | off / 32 | exact | keep the last N conversations' routing counts; seed resumed ones | coded on `feat/pattern-cache`, merged |
| `STRATA_SEED_SCALE=S` / `STRATA_SEED_HOLD=R` | `expert_seed.hpp` | 2 / 16 | exact | domain-seed top score / tier rounds | `replay_seeding.py` |
| `STRATA_LOOKAHEAD_GPU` | `generate.cpp` | **on** (=0 to disable) | exact | router lookahead runs on GPU instead of a CPU thread | +10% decode on mmap'd experts |
| `STRATA_ARENA_WARM` | `expert_source.hpp/cpp` | off | exact | lookahead thread pulls the next layer's predicted CPU experts | **measured worse** (L3 prefetch); stays off |

### 6C. Verify/combine data path and draft

| Switch | Defined | Default | Exact/Lossy | Does | Measured |
| --- | --- | --- | --- | --- | --- |
| `STRATA_COMBINE_CPU` | `verify.cpp`, `native_moe.hpp` | off | exact | verify combine reads CPU rows straight from mapped host memory | ported; conflict with deferral fixed |
| `STRATA_WAIT_FUSE` | `verify_kernels.hpp`, `verify.cpp`, `native_moe.cu` | off | exact | wait_flag_ge + mapped copy in one launch | in the fast-hip preset |
| `STRATA_WAIT_COPY256` | `verify_kernels.cu` | off | exact | the fused wait + plan copy (`wait_copy_i32_from_mapped`) with 256 threads and 16-byte reads (4-byte fallback if unaligned, word tail) instead of 128 threads and 4-byte reads; only takes effect where `wait_copy_i32_from_mapped` runs (`STRATA_FUSE_WAIT=1` on gfx11; default on gfx12 / sm75); from android-imdad | fork: host write -> plan on GPU 4.47 -> 3.57 us (7900 XTX); not yet measured here |
| `STRATA_ROUTE_PUBLISH` | `native_router.cu`, `route_row.cuh`, `verify.cpp` | off | exact | top-10 + doorbell publish in one launch | in the fast-hip preset; **off under deferral** (non-finite fix d5b58f669) |
| `STRATA_ROUTER_FUSE` | `native_bf16.cu`, `bf16_gemv.hpp` (item 48) | off | exact | the router GEMV's last block routes and publishes | ~1% short chat |
| `STRATA_COMBINE_POLL1` | `native_moe.cu` (item 46) | off | exact | one block polls the host flag over PCIe, the others spin on VRAM | 0-3% |
| `STRATA_CPU_PRESUM` | `verify.cpp`, `expert_source.hpp` (item 51) | off | not bit-identical | host pre-sums each token's CPU rows | coded (`cc5499fae`) |
| `STRATA_PRESUM_VRAM` | `verify.cpp`, `verify.hpp` (item 52, HIP) | off | not bit-identical | pre-summed rows written straight to VRAM through resizable BAR | coded (`cc5499fae`, `4f829da0e`) |
| `STRATA_EXPDEC_SWIGLU` | `iq_kernels.cu` (item 48) | off | exact | gate/up + SwiGLU + q8_1 in one expdec launch | ~49 launches per window fewer |
| `STRATA_MMVF_AUX` | `native_bf16.cu` | off | exact | the aux row rides in the multi-kernel launch | with COMBINE_CPU |
| `STRATA_VERIFY_PROFILE_SEL=i,j,...` | `verify.cpp` | off | research | selected profile stamps only | profiling |
| `STRATA_DF2_GATE` / `STRATA_DF2_ROW_MS` | `spec/window_rule.hpp` | off / 3.0 | lossy | DFlash2-calibrated survival window instead of fixed confidence | Mac-measured; opt-in |
| `STRATA_MTP_PROXY` | `mtp.cpp`, `mtp.hpp` | **on** (=0 off) | exact | two-stage draft-head proxy pick | basalt lineage; ~12% cheaper draft steps |
| `STRATA_Q5K_HEAD` | `verify.cpp` | **on** (=0 off) | exact | Q5_K/IQ4_XS specialized head kernels | basalt ports; 10b/10c findings |
| `STRATA_IC_WARM_MB=<cap>` | `verify.cpp`, `verify.hpp` (item 47) | off | exact | warms the Infinity Cache with the next layer's pre-doorbell weights during the CPU wait | coded (`269a06ed9`) |
| `STRATA_SPEC_COST_TABLE` | `generate.cpp`, `spec/draft_policy.{hpp,cpp}` (item 43) | off | exact | seeds the draft window cost table from window traces (`tools/pancake/trace/draft_costs.py`) | coded (`663107a56`) |
| `STRATA_MTP_EAGER=1` | `mtp.cpp` (item 74) | off | exact | launches every draft step at once with `--spec-min-p` and cuts the chain on the host (no host round-trip per step); passes the draft counts to `window_trace().drafts()` (item 5) | coded (`5dcd5a83c`) |

### 6D. CPU thread pool and sync

| Switch | Defined | Default | Exact/Lossy | Does | Measured |
| --- | --- | --- | --- | --- | --- |
| `STRATA_POOL_MWAITX=<TSC cycles>` | `pool.cpp` | off (spin) | exact | parked workers wait with Zen MONITORX/MWAITX | wake +0.36 us; frees SMT siblings |
| `STRATA_ADAPT_PERSIST` | `generate.cpp` (item 75, dan-9f) | **on** (=0: old per-window thread) | exact | the serve adaptive step runs on one persistent JobThread instead of a new std::thread per window | arm in bench/specs/02 |
| `STRATA_DEVICE_SPIN=1` | `generate.cpp` main (item 76, dan-9f) | off | exact | sets spin + maphost scheduling on the primary device before its context; logs "set" or "refused" | arm in bench/specs/04 (void if "refused") |
| `STRATA_UNBUFFERED_LOAD` | `expert_source.cpp`, `pinned.cu`, `direct_io.hpp` (item 38, dan-9f) | **on** on Linux (=0: buffered) | exact | the GGUF expert arena loads with O_DIRECT (16 readers) when the page cache cannot keep it; log "expert arena load: mode ..., GiB/s" | start 28-65 s -> ~10-14 s est.; test plan 2.10 |
| `STRATA_POOL_LAZY_PARK` | `pool.cpp` | off | exact | wait_parked at every phase/batch start, park only after idle | 0-1% |
| `STRATA_POOL_FUSE_PHASES` | `pool.cpp`, `pool.hpp` (item 29) | off | exact | mode 7: gate/up + down in one phase per layer | overlap model: +3% |
| `STRATA_POOL_PREREAD` | `pool.hpp`, `expert_source.hpp` (from tandem.cpp) | off | exact | idle workers pre-read the lookahead's next-layer CPU experts | loses even on a noisy machine; stays off |
| `STRATA_Q2_PREFETCH=<bytes>` | `q2_avx2.cpp` (item 40) | off | exact | prefetch Q2_0 row bytes ahead of the dot | 0-5% of CPU-expert time |
| `STRATA_Q2_FAST=1` | `q2_avx2.cpp`, `q2_avx2_rows.inl` (imanu86 `c3fc8f34d3`, MOD_PLAN F7) | off (becomes the default once `q2_fast_parity_test` passes) | exact (bitwise the legacy kernel) | AVX-2 Q2_0 rows: 8-op code unpack instead of ~14-17, d broadcast once per block, 4 rows at a time on short (down) rows; AVX-VNNI copy too; `STRATA_Q2_BITPLANE=1` still wins | imanu86: 1.11-1.45x one thread, Q2_0 layer -17% in a 7-worker pool; not yet measured here |
| `STRATA_Q2_LEGACY=1` | `q2_avx2.cpp` | off | exact | forces the legacy AVX-2 Q2_0 kernel even with `STRATA_Q2_FAST=1` (and after FAST becomes the default): the A/B switch | A/B control |
| `STRATA_IQ2S_PAIR` | `iq_avx2.cpp` | off | exact | IQ2_S decodes both 64-value halves from one table load | **neutral** (2026-11-08) |
| `STRATA_ACT_DUMP_RAW[_MAX]` | `pool.cpp` (item 13) | off | research | dump raw CPU expert activations for the sparsity study | enabled item 13's closure |
| `STRATA_POOL_TAIL4=1` | `pool.cpp` (basalt ports, `e29bac539`) | off | exact | quarter-size tail tasks, plus down-row prefetch while a fused down task waits | basalt `553e9290d` / `4e6841265` ports |

### 6E. `strata-poly` kernels and research

| Switch | Defined | Default | Does |
| --- | --- | --- | --- |
| `STRATA_GDN_COLS` | `src/poly/kernels.cu` | 4 | DeltaNet columns per warp in the GDN prefill; +9-12% prompt |
| `STRATA_GEMM_SMALL_TILED` | `src/poly/kernels.cu` | on | tiled small-GEMM path |
| `STRATA_SOFTMAX_REG` | `src/poly/kernels.cu` | off | register-blocked softmax |
| `STRATA_ROUTE_TRACE=<file>` | `src/poly/engine.cpp` | off | per-call routing trace for `cache_sim`; feeds all replay tools |
| `STRATA_SWAP_HALF=N` | `src/poly/engine.cpp` | off | halve recent-uses every N written tokens |
| `STRATA_CPU_STATS` | `src/poly/cpu_experts.cpp` | off | print what CPU experts read + timing at exit |

### 6F. Window trace and side-branch switches

| Switch | Branch | Does | Measured |
| --- | --- | --- | --- |
| `STRATA_WINDOW_TRACE=<file>` / `_EVERY=N` / `_MAX_MB=M` | main line (`window_trace.hpp`, item 64) | shared window trace: one trace run answers ~10 offline questions | drives `selftune.py` |
| `STRATA_BFLA[_MIN|_ETA|_KMIN|_KMAX|_SINK|_LOCAL|_TAIL|_QGLOBAL|_GFX]` | `feat/bfla-hip` | BFLA block-sparse prompt attention knobs (thresholds, sink/local/tail splits, Q-global, gfx guard) | 3.15x end-to-end at 128K |
| `STRATA_SEL_OVERLAP` | `feat/bfla-hip` | overlap the sparse-select with the previous kernel | part of the BFLA win |
| `STRATA_DISCO` / `STRATA_DISCO_TRACE` | `feat/disco` | learned stop rule for the MTP draft chain (replaces Dynasor, item 21) | replaces fixed draft windows |
| `STRATA_CONTEXT_DEDUP=1` | `feat/contextpilot` | lossy block dedup of repeated context | 1.6% on 40 real sessions; **keep off** |
| `STRATA_ATTN_WMMA=1` | `feat/bitdecoding-hip` | BitDecoding-style decode attention on gfx11 matrix cores | neutral; not adopted |

### 6G. Server env vars (`serve/`)

| Switch | Default | Does |
| --- | --- | --- |
| `STRATA_LOOP_GUARD` (alias `MAYA_LOOP_GUARD`) | on (=0 disables) | Project Maya loop guard: periodic-block detection, penalise/collapse/stop, tool-repeat blocking |
| `STRATA_LOOP_GUARD_COLLAPSE_MAX` | set in code | caps how much the guard collapses a looping reply |
| `STRATA_PROMPT_MEMO` (alias `MAYA_PROMPT_MEMO`) | 1 | memoized tool results for re-sent history |
| `STRATA_DOMAIN_PROFILES` | off | the server classifies each request (`classify_domain`) and passes the domain to the engine seed |
| `STRATA_IDLE_ADAPT=N` | off | item 17 (`43e741bc6`): when no request line arrives for 150 ms, the blocking tier runs up to N extra rounds at 4x swaps on the idle GPU |
| `STRATA_TURN_LOG` | off | item 73 (`1c0f248ae`): per-turn JSONL log at `<REPO>/pancake-turns.jsonl`; on in the Flash-Next presets |
| `STRATA_SPEC_COST_TABLE` | off | item 43 (`663107a56`): draft window costs seeded from window traces (`tools/pancake/trace/draft_costs.py`) |
| `STRATA_LOOP_BEHAVIOUR=off\|log\|note` (+`_EDITS=8`, `_READS=20`) | **log** | item 33 (`c03c905d3`): loop-guard behaviour counters - N edits without a build/test, M reads without an edit; `note` mode (opt-in) adds a note to the prompt |
| `STRATA_KEEP_REASONING` | off | item 31 (`d44dd5eb2`): keep earlier reasoning in the prompt (opencode 1.19.5 already resends it: leave off there) |
| `STRATA_HIPBLASLT_DESC_CACHE` | **on** (=0: per call), N = cap 256 | item 80 (`ca2f1eecc`, `997bb0a1c`): hipBLASLt matmul descriptors cached across prompt GEMMs; exact by construction |
| `STRATA_PIN_PARK_TOKENS=2048` | **on** (-1: old rule) | item 90 (`9ca057669`): parks the branch left at the pin, so a sub-agent switch returns to the parent without a re-read |
| `STRATA_AUTO_PIN=1` | off | item 71 (`35c928729`): auto-pin of the Qwen system + tools prefix (shared by sub-agents) |
| `STRATA_CACHE_TOOL_BOUNDARY=1\|2` (+`_GAP=1024`, `_MAX=2`) | off | item 16 (`527e06429`): checkpoints at tool-call / tool-result boundaries (server and engine); exact vs re-read, chunk geometry changes |
| `STRATA_CKPT_EVICT_FLOPS=<alpha>` | off | item 16: Marconi-style eviction by FLOPs saved per byte |
| `STRATA_CONVERSATION_DISK_DIR` / `--conversation-disk-dir auto\|path` (+`_MAX_GIB=20`, `_MIN_TOKENS=2048`, `_MIN_GAIN=1024`) | off | item 39 (`070fe8512`): parked conversations on disk, survive restarts |
| `STRATA_PLE_PIPE=1` (+`GATHER_THREADS=4`, `SLICES=4`, `EARLY`) | off | item 79 (`e2b3a3dad`, `3b5338412`): PLE gather off the critical path; `STRATA_PLE_COMPACT=1` now also with `--ple-io direct`; `STRATA_PLE_WAIT_TRACE=1` marks the waits |
| `STRATA_CONV_BUDGET=ram` / `STRATA_CONV_KEEP_GIB=N` | off (40% rule stays) | item 86 (`ab4fe164b`, `c1b3fec32`): conversation cache sized from free RAM |
| `STRATA_PSI_GUARD` | **on** (pause at 20%, resume at 10%, 30 s cap) | item 86: pauses loading / skips parking under memory pressure; timing only |
| `STRATA_PARK_TRUNCATE=1` | off | from TwoAnts/Strata `5ef4fc8b4b`: a conversation too large to park (cache budget or RAM floor) parks its deepest checkpoint prefix that fits instead of being dropped, so the next request re-reads only the tail, not from token 0; a parked entry asked to make room is first shrunk to one of its own checkpoints (not below `--conversation-cache-min-tokens`, not beside the disk tier); exact |

### 6H. CLI flags

- **Main engine (`strata`)**: one flag added, `--grammar on|off` (wires `src/program/grammar/`); `--geometry` was
  removed. Everything else the presets use (`--pack --native --ple-gguf --expert-profile --expert-cache --prefill
  --spec --spec-min-p --mtp --mtp-draft-vocab --max-context --kv --kv-resident --adapt-swaps --adapt-async
  --adapt-every --pool-affinity --pcie-frac`) already exists upstream.
- **`strata-poly` (`src/poly/main.cpp`, the whole engine is new)**: `--model --draft-model --max-context --max-new
  --threads --kv-q8 --kv-q4 --kv-q4-keys --kv-whole --attn-split --layer-split --any-node --elastic-prefill
  --elastic-batch --lend-staging --stage-next --stage-ahead --stage-mib --suffix-drafts --suffix-corpus --graph-experts
  --cpu-multi --cpu-share --defer-experts --expert-swaps --expert-profile --weight-cut --prompt-cache
  --prompt-cache-every --prompt-cache-message --conversation-disk[-mib][-min-free-mib] --conversation-cache-*
  --tail-skip --lookup --window --follow --min-p --top-k --top-p --top-probs --temperature --seed --images --image-pad
  --eos-ids --dump-layers --no-draft --no-gpu-experts --no-graphs --serve --vram-reserve-mib --help`.

### 6I. Server config options (`serve/server.py`)

- `"loop_guard"` (bool, default true; also settable per request): the Maya loop guard.
- `"reasoning_budget_by_effort": {level: tokens}` (0 = no cap): effort-graded thinking budgets.
- `"reasoning_budget_grace"`: Maya #613 soft close (finish the sentence, then stop).
- `turn_at`: checkpoint key for `strata-poly` (turn-boundary checkpoints).
- `reasoning_tokens` in the accounting and logs.
- Tolerant tool-call parser: accepts undeclared/malformed tool calls instead of failing the turn
  (`serve/test_tolerant_tool_calls.py`).
- Domain classification for `STRATA_DOMAIN_PROFILES` seeding.
- `"turn_log"` (item 73, `1c0f248ae`): per-turn JSONL log at `<REPO>/pancake-turns.jsonl`; on in the Flash-Next presets.
- Status page (`214c8cd31`): `/status` and `/pancake` in a browser (auto-refresh, hardware, model, switches, tok/s, cache
  and draft rates, last 30 turns), `/status.json` for scripts.
- `loop_guard_*` Prometheus counters.
- `"loop_guard_behaviour"`: off | log | note | {mode, edits, reads, tools} (item 33, default log: counters only, prompt unchanged).
- `"rumination_close"`: threshold | {threshold, checks=3, min_tokens=0} (item 34, `dd8549636`, OFF): closes thinking that
  repeats itself; the rumination signal itself is always measured (`done["rumination"]`, turn log `rumination_max/mean`).
- `chat_template_kwargs.preserve_thinking` passes through on chat / messages / responses; `"keep_reasoning": true|N`
  (item 31, OFF; the prompt is byte-identical when unset).

### 6J. Presets and launcher

`./run-pancake.sh [MODEL] [fast|exact|turbo]` picks the build (CUDA if an NVIDIA card answers, else HIP on gfx1100),
renders the preset, and serves on `PORT` (default 8080). `STRATA_DATA` (default `/home/dan/clone-of-d/Strata-data`)
and `MODELS` (default `/home/dan/clone-of-d/models`) hold the files. Models: `iq2_xs` (Flash-Next IQ2_XS, the daily
model), `q2_0` (Flash-Next Q2_0), `qwen36` (Qwen3.6-35B-A3B UD-Q4_K_XL, `strata-poly`, CUDA only), `qwen36-q3`
(UD-Q3_K_M), `qwen38-35b` (Qwen3.8-35B-A3B-Distill Q4_K_M). Speed levels: `fast` (Speedy's switch stack, default for
Flash-Next), `exact` (plain), `turbo` (IQ2_XS only, lossy: adds `STRATA_ROUTE_PRIOR=0.5`).

The preset set in `configs/pancake/` (one JSON per model, quant, speed level, and GPU):

| Preset | Model / file | GPU | Speed |
| --- | --- | --- | --- |
| `flashnext-iq2_xs` | Flash-Next IQ2_XS | CUDA | exact |
| `flashnext-iq2_xs-fast` | Flash-Next IQ2_XS | CUDA | fast |
| `flashnext-iq2_xs-turbo` | Flash-Next IQ2_XS | CUDA | turbo (lossy) |
| `flashnext-iq2_xs-hip` | Flash-Next IQ2_XS | HIP | exact |
| `flashnext-iq2_xs-fast-hip` | Flash-Next IQ2_XS | HIP | fast (~126 tok/s) |
| `flashnext-iq2_xs-turbo-hip` | Flash-Next IQ2_XS | HIP | turbo (lossy, ~168 tok/s) |
| `flashnext-iq2_xs-rdna3test` | Flash-Next IQ2_XS | HIP | RDNA3 kernel test bench |
| `flashnext-q2_0` | Flash-Next Q2_0 | CUDA | exact |
| `flashnext-q2_0-fast` | Flash-Next Q2_0 | CUDA | fast |
| `flashnext-q2_0-fast-hip` | Flash-Next Q2_0 | HIP | fast (~135 tok/s) |
| `flashnext-q2_0-turbo-hip` | Flash-Next Q2_0 | HIP | turbo (lossy) |
| `qwen36-ud-q4_k_xl` | Qwen3.6-35B-A3B UD-Q4_K_XL | CUDA (`strata-poly`) | - |
| `qwen36-ud-q3_k_m` | Qwen3.6-35B-A3B UD-Q3_K_M | CUDA (`strata-poly`) | - |
| `qwen38-35b-distill-q4_k_m` | Qwen3.8-35B-A3B-Distill Q4_K_M | CUDA (`strata-poly`) | - |

For an always-on setup, `tools/pancake/pancake.service` (item 73/91, `a1683537c`) is a systemd user unit that serves
on port 11437 with `MemoryMax=54G` and `MemorySwapMax=0`, and `Conflicts=strata/avalon/qwen35b` so only one engine
holds the RAM at a time. `NOCAP=1` in `run-pancake.sh` skips the memory-capped scope when the unit manages the process.

## 7. Code Modifications that Diverged `pancake-strata` from Original `Strata` (358 files, +102,103 lines)

Diff vs `Niko1221/Strata` (`upstream/main` = `146df3c46`): 358 files, +102,103 / -363 lines. What follows is the
subsystem walkthrough.

Latest engine/server commits: `b6e19acc2` (engine development complete: items 47, 43, 51, 52, 37, 17 and the basalt
ports coded; items 10 and 20 closed by study), `43e741bc6` (item 17, idle-tier catch-up), `1c0f248ae` (item 73,
per-turn JSONL log), `a1683537c` (item 73/91, systemd user unit), `5dcd5a83c` (item 74, eager MTP draft),
`e29bac539` (basalt CPU pool ports), `269a06ed9` (item 47, Infinity Cache warm-up), `663107a56` (item 43, draft
window costs from traces) and `7861dc8c1` (item 37, the sweep tool).

### 7.1. `src/poly/`: the `strata-poly` engine and its kernels

About 40 new files ported from PolyStrata/FastStrata: the engine (`engine.cpp`), CUDA kernels (`kernels.cu`,
`kernels_dsv4.cu`, `mmq.cu`), AVX2 CPU kernels, model files (`qwen35`, `deepseek4`, `gemma4`, `glm5next`), a
conversation disk cache with suffix replay and index, hybrid CPU/GPU expert execution (`cpu_experts`), and
parity/bench/test files. Own switches: `STRATA_GDN_COLS`, `STRATA_GEMM_SMALL_TILED`, `STRATA_SOFTMAX_REG`,
`STRATA_ROUTE_TRACE`, `STRATA_SWAP_HALF`, `STRATA_CPU_STATS`. Why: stock `strata` at 64K context writes at 9.7 tok/s
on the 3060; `strata-poly` reaches 71.7 on the same prompt.

### 7.2. `src/kernels/cuda/rdna_gr.cu`, `rdna_mmvq.cu`, `route_row.cuh`: the RDNA3 decode and prefill port

Hand-ported from Strata-rdna3 (upstream had moved 1,938 commits since that fork's base): wave32 hyper-connection
read kernels, multi-column Q6_K decode GEMV, wave-shaped MMVQ, one-launch fusions (`STRATA_GR_FUSED`,
`STRATA_WAIT_FUSE`, `STRATA_ROUTE_PUBLISH`), and the CPU-combine path (`STRATA_COMBINE_CPU`). Result on the 7900 XT:
87-101 tok/s before the port, 104-108 after the kernels, ~112 with the fusions; ahead of Strata-rdna3 itself
(109-119), Avalon (95-100) and stock (75-77).

### 7.3. `src/kernels/cuda/q5k_head.cu`, `head_fmt.cu`, `head_proxy.cuh`: specialized output heads and the 2-bit MTP proxy

Basalt ports: specialized Q5_K and IQ4_XS head kernels (the head is 540 MB Q8_0 / 337 MB IQ4_XS), a head-format
dispatcher, and a two-stage draft proxy that scores all MTP candidates from a 2-bit proxy and rescores only the
near-best. `STRATA_Q5K_HEAD` and `STRATA_MTP_PROXY` are on by default; the draft step gets ~12% cheaper.

### 7.4. MoE cache admission, seeding, deferral, depth-graded prior

New headers and their engines: `include/strata/program/kmiss_tier.hpp` (`STRATA_ADAPT_KMISS`, admit on the k-th miss
in W windows), `swap_budget.hpp` (`STRATA_ADAPT_SWAPS_AUTO`, online swap budget), `expert_seed.hpp`
(`STRATA_DOMAIN_PROFILES`, `STRATA_CONV_WORKSET*`, `STRATA_SEED_SCALE/HOLD`: domain and conversation seeding),
`include/strata/core/defer_rank.hpp` (`STRATA_VERIFY_DEFER_CPU/_BELOW/_TOP/_FOLD`, the KTransformers-style rank guard),
`route_prior.hpp` (`STRATA_ROUTE_PRIOR_DEPTH`, AcceptMoE-style depth grading), plus `src/core/verify.cpp` and
`expert_source.cpp` wiring and the GPU lookahead default (`STRATA_LOOKAHEAD_GPU`). Why: replay showed the tier at
79.6% hit vs LRU 89.5% and Belady 94.7% at 44% residency, and the overlap model priced deferral +33% and
deferral + kmiss 2/8 at +45-52% on edit windows.

### 7.5. `src/kernels/cpu/pool.cpp`, `q2_avx2.cpp`, `iq_avx2.cpp`: the CPU pool

MWAITX parking for parked workers (`STRATA_POOL_MWAITX`, wake +0.36 us, frees SMT siblings), lazy trailing park,
fused phases (mode 7: gate/up + down in one phase per layer), Q2_0 row prefetch, IQ2_S pair decode (neutral), a raw
activation dump hook (closed the sparsity question), and NUMA-split-aware planning. The basalt pool ports
(`e29bac539`) add quarter-size tail tasks (`STRATA_POOL_TAIL4=1`, basalt `553e9290d`) and down-row prefetch while a
fused down task waits (basalt `4e6841265`). Why: the CPU Q2_0 kernel already
runs at ~80% of the 33 GB/s DDR4 allows, so only sync and overlap levers remain.

### 7.6. `src/program/grammar/`, `serve/grammar.py`, `serve/loop_guard.py`, `serve/server.py`: the server and agent layer

The grammar engine (`grammar`, `guard`, `schema`, `json_value`) with the engine flag `--grammar` (JSON-schema and
tool-call constraint; fixed bug 89 on the Anthropic/Responses APIs). `loop_guard.py` is a 690-line port of Project
Maya v1.1.56 (MIT): periodic-block detection, penalise/collapse/stop, tool-repeat blocking. `server.py` adds
`reasoning_budget_by_effort` + `reasoning_budget_grace` (soft close), `reasoning_tokens` accounting, a tolerant
tool-call parser, `turn_at` poly checkpoints, domain classification, and the per-turn JSONL log
(`STRATA_TURN_LOG`, item 73, `1c0f248ae`). Why: opencode history showed turn p99 of
224 s driven by re-reads (~18%) and uncapped thinking, not by tok/s.

### 7.7. Side branches

- **`feat/bfla-hip`** (7 commits): BFLA block-sparse prompt attention for the qwen35 engine on gfx11, the
  `STRATA_BFLA*` family, `STRATA_SEL_OVERLAP`, a fused RDNA3 WMMA kernel, and `tools/pancake/bfla_recall.py`.
  Measured: 128K one layer 28.3 s to 9.6 s fused to 3.3 s BFLA; end-to-end 128K prompt 3.15x; needle recall 48/48
  top-1, KL 0.00448. Unmerged, HIP only.
- **`feat/disco`** (1 commit): DISCO, a learned stop rule for the MTP draft chain (`STRATA_DISCO`,
  `src/spec/disco.{hpp,cpp}`, `disco_fit.py`); replaces Dynasor as the early-exit route (item 21).
- **`feat/contextpilot`** (2 commits): `serve/context_dedup.py`, lossy block dedup (`STRATA_CONTEXT_DEDUP`), exact
  cache-aware admission, `dedup_savings.py`. Measured 1.6% on 40 real sessions: keep off.
- **`feat/bitdecoding-hip`** (2 commits): BitDecoding-style decode attention on gfx11 matrix cores
  (`STRATA_ATTN_WMMA`), parity test. Measured neutral: not adopted.
- **`exp/expert-prefetch`** (28 commits): the GPU router lookahead default (+10%), a learned expert profile (+6.7%
  decode), `capped.sh` (memory-capped systemd scope), and the plan-10 measurement record.
- **`feat/pattern-cache`** (63 commits): the item 3-4 seeding switches, Q2_0 7900-XT presets, `STRATA_POOL_PREREAD`
  and `STRATA_IQ2S_PAIR`, and a GDN megakernel prototype (reverted: 0.0 ms on one GPU).

### 7.8. `tools/pancake/`: the measurement suite (31 scripts)

`abtest.py` (quiet-machine interleaved A/B), `quality_gate.py` (token-forcing gate as a process), `task_gate.py` (A55, see below), `quality_ab.sh` /
`quality_hip.sh`, `bench_engines.py`, `gpu_profile.py` (rocprofv3), `decode_curve.py`, `time_ceiling.py` (the cache
judged in ms; serial and `--overlap` model), `belady_ceiling.py`, `tail_rows.py`, `predict_experts.py`,
`replay_seeding.py`, `conv_trace.py`, `learn_profile.py` / `learn_domain_profiles.py`, `make_corpus.py`,
`sparsity_masks.py`, `selftune.py` (item 70 closed loop), `regression_watch.py`, `build_matrix.sh` + `pre-push.sh`
(item 92), `capped.sh`, `test_phase.sh`, `sweep.py` (item 37, `7861dc8c1`: coordinate-descent sweep of the GPU
expert kernels' launch/tile knobs and the CPU pool knobs), `trace/draft_costs.py` (item 43, `663107a56`), and
`pancake.service` (item 73/91, `a1683537c`), plus `trace/{wtrace,tail_replay}.py`. Why: every claim in this README came
out of one of these scripts, and the suite is what makes the stack reproducible.

`task_gate.py` (A55, branch `work/a55-task-gate`, first run pending) is the second half of the quality gate for lossy
arms. Token forcing only rejects arms that deviate a lot; among near-baseline variants KLD and top-1 agreement do not
predict the task score ("Displacement Is Not Direction", arXiv 2606.19558), so a fixed task set decides. It takes the
`abtest.py` spec (base preset plus env / args per arm; optional `tasks`, `reps` (5), `temperature` (0.2), `margin` (0.05),
`validity_margin` (0.03)), starts one fresh server per arm (the base arm twice, first and last, as the drift check) and
sends the 18 tasks of `bench/tasks/default.json` `reps` times each with the same seeds in every arm: 8 code edits (the
answer's python block is run against a test with CPU / memory limits and no network where `unshare -rn` works), 7 tool
calls (JSON schemas; the call must parse, validate and match the expected name and arguments; parallel, multi-turn and
"no call" cases), 3 short QA items. Per arm it reports the success rate with a 95% Wilson interval, per-kind rates, the
tool-call validity rate, time to success (median, seconds per success) and a verdict against base: the paired
per-task difference of the success rate with a 95% bootstrap interval (tasks and repetitions resampled), judged by
non-inferiority: BETTER (lower bound above 0), PASS (lower bound within the margin), REJECT (upper bound below
minus the margin), else INCONCLUSIVE; the tool-call validity rate gets the same test with its own margin and the
worse of the two verdicts counts. Time to success is reported, not judged (speed is `abtest.py`'s job).

```sh
python tools/pancake/task_gate.py bench/specs/taskgate-lossy.json --dry-run --out /tmp/tg   # spec + task set + every checker, no server
python tools/pancake/task_gate.py bench/specs/taskgate-lossy.json --out bench/results/<day>-taskgate [--reps 10] [--only tool_call]
python tools/pancake/task_gate.py --from-runs bench/results/<day>-taskgate [--regrade]     # report again from the stored runs
python -m pytest tools/pancake/test_task_gate.py                                            # statistics + checkers, no server
```

At 18 tasks x 5 reps the interval on the success difference is about +-15 points (synthetic check), so only a large loss
is a REJECT and a close arm comes back INCONCLUSIVE: use `--reps 10-20` for close variants, or add tasks (a JSON file
of the same shape; every task needs a reference answer that must pass and a wrong one that must fail, checked by
`--dry-run`). The shipped set is a smoke-sized stand-in; the BFCL v4 multi-turn / LiveCodeBench subsets and own-repo
checkpoint grading of A55 are not in it yet.

## 8. Daily Use (`Qwen3.8-Flash-Next` & `Qwen3.6-35B-A3B`) and Build & Run (HIP + CUDA)

### Daily use

```sh
./run-pancake.sh                  # Flash-Next IQ2_XS, fast preset, picks CUDA or HIP by itself
./run-pancake.sh iq2_xs fast      # same, explicit
./run-pancake.sh q2_0 fast        # Flash-Next Q2_0, ~135 tok/s on the 7900 XT
./run-pancake.sh iq2_xs turbo     # lossy opt-in: +25% decode on the 3060, ~168 tok/s on the 7900 XT
./run-pancake.sh qwen36           # Qwen3.6-35B-A3B on strata-poly (CUDA only)
./run-pancake.sh qwen36-q3        # the smaller Qwen3.6 file
./run-pancake.sh qwen38-35b       # the Qwen3.8-35B distill, MTP layer borrowed from Qwen3.6
```

### Linux memory tuning (once, at install)

The Flash-Next presets lock a ~33 GiB expert arena in RAM within seconds of starting. On a PC that has been up for a
while, free RAM is fragmented: the kernel stalls in compaction during that lock, memory pressure passes systemd-oomd's
50% limit and the engine is killed while it loads (and below the limit, decoding runs ~25% slower).
The worst part is that compaction by default also migrates mlock'ed pages: the arena is mlock'ed and registered with
the GPU driver, so every moved page makes KFD evict and restore the GPU queues (kernel log `svm_range_restore_work
hogged CPU`); `vm.compact_unevictable_allowed=0` keeps compaction away from it.
`tools/pancake/os_tune.sh` checks this without root and installs the fix; `run-pancake.sh` prints a one-line tip while
it is missing.

```sh
tools/pancake/os_tune.sh                          # state: free RAM in 2 MiB blocks, compaction, huge pages
sudo tools/pancake/os_tune.sh install             # 1 (recommended): compact_unevictable_allowed=0 + compaction_proactiveness=50
sudo tools/pancake/os_tune.sh install --hugepages # 1 + 2 (optional): reserve 16,920 x 2 MiB pages at boot for the arena
sudo tools/pancake/os_tune.sh compact             # one compaction pass now, nothing persistent
```

Option 2 maps the arena from a reserved hugetlb pool (no fragmentation, fewer TLB misses; two Strata forks measured
+5-7% decode, not yet measured here). The pool is taken out of ordinary RAM (33 GiB) from the next boot, so it suits a
PC that mostly runs pancake. The engine counts a pool that holds the whole arena outside its RAM budget.

The server listens on `PORT` (default 8080) with the OpenAI- and Anthropic-compatible API. `STRATA_DATA` holds the
data directory, `MODELS` the model files, `HIP_VISIBLE_DEVICES` picks the ROCm GPU (default 0; GPU 1 is the Vega
iGPU). Flash-Next defaults to the fast preset; `exact` gives the same answers as plain decoding; `turbo` is the
lossy opt-in described in section 6J. For an always-on setup, `tools/pancake/pancake.service` (item 73/91,
`a1683537c`) runs the server as a systemd user unit on port 11437 with `MemoryMax=54G` and `MemorySwapMax=0`, and
`Conflicts=strata/avalon/qwen35b` so only one engine holds the RAM; `NOCAP=1 ./run-pancake.sh` skips the
memory-capped scope when the unit manages the process.

### Build for the RX 7900 XT (HIP, gfx1100)

Optional: `-DSTRATA_HOST_NATIVE=ON` builds Strata's host C++ for this CPU (`-march=native -ffp-contract=off`, device code
untouched); built separately as `build-hip-native` and compared in `bench/specs/08-build.json` before it becomes the default.

Needs a ROCm driver and runtime, HIP headers and compiler, hipBLAS and hipBLASLt development files, CMake 3.24+, a
C++20 host compiler:

```sh
cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip --target strata -j2
```

### Build for the RTX 3060 (CUDA, sm_86)

```sh
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86-real
cmake --build build-cuda --target strata strata-poly poly_gdn_cols_test poly_gemm_small_test -j8
```

### Build on Windows (CUDA 12.8, sm_75)

There is a Windows build with MSVC and CUDA 12.8 for a GTX 1660 Ti (sm_75). It runs `strata-poly` on the Qwen3.6-35B-A3B IQ2_XXS file.
Two changes were needed: one portability line in `kmiss_tier.hpp` (`std::popcount` instead of a GCC builtin) and `-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared` at configure time.
Four test targets do not compile on MSVC yet: `q5k_head_test`, `flag_order_test`, `gemm_w16_parity`, `poly_kv_q4_test`. They do not block the engine.
Details, commands and what was not tested: [`docs/WINDOWS_CUDA_SM75.md`](docs/WINDOWS_CUDA_SM75.md).

### Verification gates

- `pancake-docs/CODE_OPTIMIZATION.md`: the 7-point code-level pass (profiling, algorithms, caching, I/O, async,
  allocations, static analysis) - four quadratic loops removed from the per-layer expert dispatch.
- `tools/pancake/hipblaslt_extend.py` (item 80): extends the gfx1100 hipBLASLt table with the agent-turn GEMM shapes
  (plan / run / merge, 78 cases, GPU ~30 min); then A/B the extended table.
- `ctest -L cpu` (117 pass, 6 skipped) / `ctest -L gpu` (124 incl. rdna_gr/rdna_mmvq parity) and `tools/pancake/build_matrix.sh --tests`
  (item 93); `tools/pancake/check_stubs.py` (item 95b).
- `tools/pancake/restore_check.py` (items 35/83, `--tool-boundary` item 16, `--disk` item 39): a parked and restored conversation must match a cold re-read token
  for token (gate for the cache/checkpoint items); first step of `tools/pancake/test_phase.sh`.
- `tools/pancake/test_phase.sh` + `bench/specs/*.json`: the whole A/B test phase in one command; the plan is
  `bench/TEST_PLAN.md` (every claim of this README tested one by one).

Nothing goes on the stack without these:

```sh
tools/pancake/build_matrix.sh        # builds both backends
tools/pancake/abtest.py              # interleaved A/B on a quiet machine
tools/pancake/quality_gate.py        # token-forcing gate, required for every lossy arm
tools/pancake/test_phase.sh          # the whole test phase in one command, resumable
tools/pancake/regression_watch.py    # watches for regressions across runs
tools/pancake/pre-push.sh            # what to run before pushing
```

An exact switch must decode the same tokens as stock. A lossy switch must pass the quality gate: no large-margin
errors, and a token-difference rate explained against the 1.2-1.3% stock run-to-run noise.

## 9. License & Credits

`pancake-strata` keeps Strata's license: **MIT**, Copyright (c) 2026 Niko1221 and the Strata contributors. See
[LICENSE](LICENSE).

Credits, in the order the stack was built:

- **Niko1221** and the Strata contributors, for the engine everything sits on.
- **Jarson1408** (FastStrata) and **simplexai-labs** (PolyStrata), for the `strata-poly` engine.
- **nicejunjie** (Strata-rdna3), for the RDNA3 kernels this project ported and fused.
- **jesdga95** (basalt), for the head kernels, the draft proxy and the grammar path.
- **mw00** (Project Maya, MIT), for the loop guard, the soft budget close and the tolerant tool calls.
- **android-imdad** (Strata-7900xtx), for the 256-thread, 16-byte plan copy (`STRATA_WAIT_COPY256`, commit caa9b00115).
- **DevDock-AI** (Speedy), for the fast-preset switch stack. **Avalon**, for the hipBLASLt table report.
- **imanu86** (Strata fork), for the faster bit-exact Q2_0 AVX-2 kernel (`STRATA_Q2_FAST`, commit `c3fc8f34d3`).
- **TwoAnts** (Strata fork, MIT), for truncated parking (`STRATA_PARK_TRUNCATE`).
- **RX5950XT** (Strata fork, MIT), for the IQ2/IQ3 sign form without byte compares that `STRATA_IQ_SIGN_FAST` uses (commit
  `b9a2111d3f`, llama.cpp's `ggml_cuda_apply_signs` without its multiply), and **boxwrench** (in yqYo1/Strata, MIT), for
  measuring packed sign expansion on an RX 7900 XT (commit `6f0bdf665b`).
- **ewhacc** and **itlab-cl** (tandem.cpp), for the pool pre-read. **JakeATX** (llamAmpere), **Zagitta**
  (gpu-unlock), **JustVugg** (colibri) and the other forks surveyed along the way.
- **ggml-org** (llama.cpp / ggml, MIT) and **ikawrakow** (ik_llama.cpp, MIT), for the formats and kernels underneath.
- **kvcache-ai** (KTransformers), **sgl-project** (SGLang), **vllm-project** (vLLM), **NVIDIA** (TensorRT-LLM),
  **flashinfer-ai** (FlashInfer), for ideas and reference points.
- The authors of every paper in section 4, and the university groups behind the 758-paper survey.
- **Qwen** (Alibaba) for the models, **ISTA-DASLab** and **Unsloth** for the quant packs.
