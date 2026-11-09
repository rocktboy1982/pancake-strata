# pancake-strata: research and findings

The companion reference to [README.md](README.md) and to the plan in
[../MOD_PLAN.md](../MOD_PLAN.md). It catalogs everything pancake-strata investigated: the
research reports and notes, the academic papers behind them, the open-source projects and
forks that were read or ported, every engineering decision taken (adopted, opt-in,
in-progress, rejected), and every benchmark, profile and test result measured on the box.

Hardware for every number unless a section says otherwise:

| Box | Card | CPU | RAM | Stack |
| --- | --- | --- | --- | --- |
| RTX 3060 era (sections 1-12 of [../docs/PANCAKE_FINDINGS.md](../docs/PANCAKE_FINDINGS.md)) | RTX 3060 12 GB (GA106, sm_86) | Ryzen 7 5700G, 8c/16t, AVX2, 16 MiB L3 | 57 GB DDR4 dual channel | NVIDIA driver 595.99, CUDA 13.1, gcc 15.2, Linux kernel 7.0 |
| RX 7900 XT era (phase 11 onward) | RX 7900 XT 20 GB (gfx1100) | same 5700G | same 57 GB | ROCm 10.0, HIP; PCIe 3.0 forced by the 5700G, ReBAR on (32 GB BAR0), IOMMU passthrough, THP madvise |

Models: Qwen3.8-Flash-Next (125B MoE + 51B n-gram/PLE table + ~4B MTP, about 176.9B
parameters total; the "IQ2_XS" file's experts are IQ2_S/IQ2_XXS/IQ1_M gate/up with Q2_0
down) and Qwen3.6-35B-A3B (Unsloth UD-Q4_K_XL with its MTP layer). Greedy decoding unless
stated. Raw results live in `bench/results/`, one JSON line per run.

---

## 1. Overview and the two-stage verification methodology

Every hypothesis in pancake-strata goes through the same loop before it changes a default:

1. **Research.** A claim is traced to a paper, a fork, or our own profile. The research
   series of 2026-11-09 (15 reports in [../reports/](../reports/), mirrored note by note in
   [../research_notes/](../research_notes/)) covered the open literature in five layers:
   survey, techniques, mechanisms, code, expected gain. Papers are priced against the
   decode-window time budget before any code is written: on the 7900 XT a short-chat window
   is 24.8 ms (GPU 15.9, CPU experts 5.2, draft 2.2) and a code-edit window is 37.4 ms
   (GPU 15.8, CPU experts 17.4, draft 2.5), so a technique that cannot move one of those
   blocks is dropped on paper.
2. **Price.** Gains inside a substitute group do not add. The plan groups items into
   S1-S7 (section 5 below) and caps each group's ceiling: for example the CPU-expert group
   S1 has an ideal-overlap floor of about 20 ms on edits, so its ceiling is lossy +40-60%,
   exact-only +15-30%, and short chat at most 1.27x. A candidate is only built if its
   modeled gain clears the run-to-run noise (about 20% on a busy machine, so A/Bs are
   load-gated and interleaved) and the group's >= 5% rule.
3. **Test for parity.** Exact changes must be bit-identical: kernel tests compare against
   the previous kernels bit for bit (`poly_gdn_cols_test`, `poly_gemm_small_test`,
   `bfla_test` dense vs FP64, `iq4xs_head_test` greedy tokens identical on/off). Lossy
   changes go through token forcing (`tools/pancake/quality_gate.py`): 20 requests over
   code, docs, English, multilingual and agent text, 300 greedy tokens each, scored
   teacher-forced on each arm's own decode path, with two stock arms giving the run-to-run
   noise. Token forcing only rejects large deviations; choice between close variants is
   made by a fixed task set (LiveCodeBench subset + BFCL v4 multi-turn + own-repo
   checkpoint grading), following the Toronto finding "Displacement Is Not Direction"
   (arXiv 2606.19558).
4. **Decide.** Three outcomes only:
   - **Adopted**: in the default path, with the measured numbers in the commit message.
   - **Coded opt-in**: built, off by default, waiting on the test phase or a quality gate.
   - **Not adopted**: logged in the ledger (section 6) with the measured evidence and the
     exact condition that would reopen it. Nothing is silently dropped.

Two rules govern the whole document. First, "measure first, every step": a change stays
only if it is faster on the bench and passes the exactness check. Second, "exact by
default": greedy output equals the base engine token for token; lossy options are opt-in
flags with a start-up warning. A CUDA-only change must never break the HIP build, and both
cards are measured from the same source.

The methodology exists because the box taught it: the edit window is the plain sum of its
parts (GPU 15.8 + CPU 17.4 + draft 2.5 ms, plus about 1.7 ms of sync), which makes it a
scheduling and overlap problem, not a bandwidth problem. That single measurement re-ranked
the whole plan toward deferral, the overlap simulator, and the shared trace, and it caps
the sum of the small exact items at a few percent up to about 10% on edits.

---

## 2. Master index of research reports (`reports/`) and research notes (`research_notes/`)

Fifteen reports (mostly Romanian) and fourteen note folders. Each report is mirrored
note-by-note in `research_notes/`; the plan rows each one produced are in
[../MOD_PLAN.md](../MOD_PLAN.md) phase 11 ("Phase 11 research index", items 13-62) and the
combined items 63-95b.

### 2.1 The 15 reports in `reports/`

| Report | Notes folder | Key findings | Plan items produced |
| --- | --- | --- | --- |
| [Pancake MoE engine speedup](../reports/Pancake%20MoE%20engine%20speedup.md) | [Pancake MoE engine speedup/](../research_notes/Pancake%20MoE%20engine%20speedup/) | Six-thread deep research: CPU experts, caching/prediction, MoE speculative decoding, RDNA3 GPU decode, PCIe/KV, engines and labs. ~60 papers with URLs. | `--pcie-frac 0` (+6/+13%), adaptive tier every window (+16%), Belady ceiling replay (79.6/89.5/94.7%), expert deferral, DraftExpert windows (tested: policy already optimal), tail-row drop (item 5) |
| [Pancake pain points și soluții](../reports/Pancake%20pain%20points%20%C8%99i%20solu%C8%9Bii.md) | [Pancake pain points și soluții/](../research_notes/Pancake%20pain%20points%20%C8%99i%20solu%C8%9Bii/) | Edit window: CPU experts 17.4 of 37.4 ms at ~80% of DDR4; GPU spends 38% of kernel time waiting in `combine_cpu`; opencode history (old backend): turn median 6.2 s, p90 50 s, p99 224 s, re-reads ~18% of turn time, 15 turns cut while thinking; pancake not yet wired to opencode (8080 vs 11437, no unit). | Order of work: 0 wiring + thinking caps, 1 measurements, 2a per-turn exact (35, 31, 11, 16), 2b engine exact, 3 lossy behind revised gates, 4 harness/side branches |
| [Tehnologii grupate pentru pancake](../reports/Tehnologii%20grupate%20pentru%20pancake.md) | (the nine reports) | 10 technology families by mechanism (F1 fewer expert bytes ... F10 harness/ops); F1-F3 are substitutes for the same 17.4 ms of CPU experts; shared infrastructure list. | Substitute groups S1-S7; shared trace + simulator + dump hook + per-turn log + revised gates |
| [Tehnologii drum de date hardware](../reports/Tehnologii%20drum%20de%20date%20hardware.md) | [Tehnologii drum de date hardware/](../research_notes/Tehnologii%20drum%20de%20date%20hardware/) | Edit window ~ sum of parts (scheduling, not bandwidth; overlap floor ~20 ms); CPU Q2_0 at ~80% DDR4; promotions ~93% of Gen3; critical-path transfer is the fp32 row pull in `combine_cpu` (0.4-0.8 ms est. vs <= 0.27 ms GPU bound); pre-doorbell GPU kernels critical in both window types. | Items 38-54 in waves: 0 = 38, 54, 45; 1 = 49, 50, 44; 2 = 46, 40, 41, 48; 3 = 47, 51, 53, 52 last |
| [Capabilități platforme consumer](../reports/Capabilit%C4%83%C8%9Bi%20platforme%20consumer.md) | [Capabilități platforme consumer/](../research_notes/Capabilit%C4%83%C8%9Bi%20platforme%20consumer/) | CPU-compute vs PCIe-promote ratio varies ~10x across platforms (today 1:2.7; Zen 3 B550/X570 ~1:1.2-1.8; LGA1700/1851 DDR5 ~1:2-4; AM5 + Gen5 card ~1:1 or inverted; Strix Halo no promotion). | Items 55-62; item 55 must calibrate on window time; acceptance test: it finds `--pcie-frac 0` on the 7900 XT by itself (static PcieAuto picked 0.20-0.25 and lost) |
| [Pancake cod deep research](../reports/Pancake%20cod%20deep%20research.md) | [Pancake cod deep research/](../research_notes/Pancake%20cod%20deep%20research/) | Five code studies (decode path, prefill path, memory/cache, server/agent, structure/build) via Serena + clangd. No wrong output found; one functional bug (89: grammar missing on Anthropic/Responses). `generate.cpp` main() is 18,361 lines; ~1,360 kernels per window; 872 distinct STRATA_* names. | Items 74-95b with a staged order (stage 0 no GPU: 87, 89 fix, 88, 92, 95b; then env-only GPU session 82, 78, 76; then 91 + 73, 38, 86, 85; then 83, 39, 90, 71; then 75, 74, 79, 80, 94, 95) |
| [LLM runtime architectures 2026](../reports/LLM%20runtime%20architectures%202026.md) | [Arhitecturi runtime LLM 2026/](../research_notes/Arhitecturi%20runtime%20LLM%202026/) | Pasted overview: names real, details inflated (TokenSpeed "zero runtime errors" unsourced; xLLM Ascend-only benchmarks; Exo numbers overstated). P/D disaggregation, chunked-prefill interleave, overlap scheduler, distributed KV stores do not apply at batch 1 on one card. | Adopt items 10-12 of that report only; the rest parked |
| [Avalon upgrades from upstream projects](../reports/Avalon%20upgrades%20from%20upstream%20projects.md) | [Avalon upgrades from upstream projects/](../research_notes/Avalon%20upgrades%20from%20upstream%20projects/) | The gfx1100 hipBLASLt table must match the runtime (preset loaded 100200 on hipBLASLt 100401, untuned; matching table +38% prompt, provisional); longer prompt-lookup windows (`--lookup-max-t 12/16`); Project Maya's loop guard, soft budget close, tolerant tool calls. | hipBLASLt 100401 in the 7900 XT presets (done); `--lookup-max-t 12` A/B; Maya ports into `serve/` |
| [Lucrări Stanford utile pentru Avalon](../reports/Lucr%C4%83ri%20Stanford%20utile%20pentru%20Avalon.md) | [Lucrări Stanford utile pentru Avalon/](../research_notes/Lucr%C4%83ri%20Stanford%20utile%20pentru%20Avalon/) | 76 papers. Per-level thinking cap; multi-entry prefix cache (RadixAttention); CATS; measurement harness. | Per-level caps (done, 5dd620fb9); item 13 (CATS, superseded); item 11 family |
| [Lucrări MIT Harvard pentru Avalon](../reports/Lucr%C4%83ri%20MIT%20Harvard%20pentru%20Avalon.md) | [Lucrări MIT Harvard pentru Avalon/](../research_notes/Lucr%C4%83ri%20MIT%20Harvard%20pentru%20Avalon/) | 65 papers. TEAL over CATS; GDN chunk 64; XAttention; learned thinking stop; Scaling Laws for Precision bit allocation. | Items 13, 18, 19, 21; quant bit-allocation guidance |
| [Lucrări universități SUA pentru Avalon](../reports/Lucr%C4%83ri%20universit%C4%83%C8%9Bi%20SUA%20pentru%20Avalon.md) | [Lucrări universități SUA pentru Avalon/](../research_notes/Lucr%C4%83ri%20universit%C4%83%C8%9Bi%20SUA%20pentru%20Avalon/) | 131 papers (Berkeley, CMU, UW, Princeton, Cornell, UCSD, Duke, Rice, Stevens...). Berkeley gate mask; SuffixDecoding; FreeToken; Marconi; QTIP. | Items 13-17, 20, 22; QTIP/IQ2_KT rejected (not worth a requant) |
| [Lucrări Europa pentru Avalon](../reports/Lucr%C4%83ri%20Europa%20pentru%20Avalon.md) | [Lucrări Europa pentru Avalon/](../research_notes/Lucr%C4%83ri%20Europa%20pentru%20Avalon/) | 69 papers. ISTA-DASLab cards: Q2_0 decodes 93.8 vs IQ2_XS 70.3 tok/s at -0.09 task average (Q2_0 confirmed as daily file); base is 176.9B incl. the 51B n-gram table; "IQ2_XS" experts are IQ2_S/IQ2_XXS/IQ1_M + Q2_0 down; ETH "Mind the Gap" GGUF backdoors. | Items 23-27; daily file decision (item 6 closed); security practice: official files + SHA-256 vs HF |
| [Lucrări Asia pentru Avalon](../reports/Lucr%C4%83ri%20Asia%20pentru%20Avalon.md) | [Lucrări Asia pentru Avalon/](../research_notes/Lucr%C4%83ri%20Asia%20pentru%20Avalon/) | 97 papers. Qwen3.8 tech report checked against our code: MTP depth already `--spec 4` (official 4.07 accepted at 4 steps); drafter has no PLE block and dense SWA attention; PLE via `--ple-io direct`. | Items 28-32; KTransformers rank guard (item 28); Tsinghua OSDI'26 pool barriers (item 29) |
| [Lucrări Canada pentru Avalon](../reports/Lucr%C4%83ri%20Canada%20pentru%20Avalon.md) | [Lucrări Canada pentru Avalon/](../research_notes/Lucr%C4%83ri%20Canada%20pentru%20Avalon/) | 56 papers. "Displacement Is Not Direction" (Toronto): KLD/top-1 stop predicts task score among near-baseline quants, so token forcing only rejects large deviations and a fixed task set picks between close candidates. | Quality-gate revision for items 8, 13, 23; items 33-34; SpAx as 4th mask; Cactus parked |
| [Lucrări Pacific Africa America Latină](../reports/Lucr%C4%83ri%20Pacific%20Africa%20America%20Latin%C4%83.md) | [Lucrări Pacific Africa America Latină/](../research_notes/Lucr%C4%83ri%20Pacific%20Africa%20America%20Latin%C4%83/) | 63 papers. SpecFit marginal new-miss cost (DISCO feature); UNSW restore-equivalence test; UFMG coordinate-descent kernel sweep; UBA top-k entropy signal. | Items 35-37; ends with the series-wide ranking (GPU profile first, then turbo decision, sparsity, per-turn cache...) |

### 2.2 The 14 note folders in `research_notes/`

| Folder | Files | What they hold |
| --- | --- | --- |
| [Pancake MoE engine speedup/](../research_notes/Pancake%20MoE%20engine%20speedup/) | cpu_expert_compute.md, expert_caching_prediction.md, moe_speculative_decoding.md, rdna3_gpu_decode.md, transfer_and_kv.md, engines_labs_companies.md | The six-thread deep research behind the phase 11 report; ~60 papers with URLs, each priced against the window budget |
| [Pancake pain points și soluții/](../research_notes/Pancake%20pain%20points%20%C8%99i%20solu%C8%9Bii/) | engine.md, server_agent_ops.md | Window breakdown, combine_cpu wait share, opencode turn statistics (2,063 turns: median 6.2 s, p90 50 s, p99 224 s) |
| [Pancake cod deep research/](../research_notes/Pancake%20cod%20deep%20research/) | decode_path.md, prefill_path.md, memory_cache.md, server_agent.md, structure_build.md | The five code studies that produced items 74-95b, with file/line evidence |
| [Tehnologii drum de date hardware/](../research_notes/Tehnologii%20drum%20de%20date%20hardware/) | cpu_ram.md, gpu_sync.md, nvme_storage.md, pcie_transfers.md, pipeline_cost_models.md | The data-path round behind items 38-54; 66 papers |
| [Capabilități platforme consumer/](../research_notes/Capabilit%C4%83%C8%9Bi%20platforme%20consumer/) | intel_platforms.md, amd_platforms.md, consumer_gpus.md | The platform matrix behind items 55-62 |
| [Arhitecturi runtime LLM 2026/](../research_notes/Arhitecturi%20runtime%20LLM%202026/) | control_execution_plane.md, disaggregated_serving.md, heterogeneous_runtimes.md, runtime_intelligence.md, single_node_applicability.md | 31 papers; what transfers to a single-node batch-1 engine and what does not |
| [Avalon upgrades from upstream projects/](../research_notes/Avalon%20upgrades%20from%20upstream%20projects/) | avalon_baseline.md, llamacpp_performance.md, llamacpp_server_features.md, project_maya.md, strata.md | Upstream PR watch list (llama.cpp #29887-#31888), Maya feature notes |
| [Lucrări Stanford utile pentru Avalon/](../research_notes/Lucr%C4%83ri%20Stanford%20utile%20pentru%20Avalon/) | agents_nlp.md, attention_kernels.md, decoding_serving.md, quantization_moe.md | 76 papers, four subject files |
| [Lucrări MIT Harvard pentru Avalon/](../research_notes/Lucr%C4%83ri%20MIT%20Harvard%20pentru%20Avalon/) | agents_nlp.md, attention_kernels.md, decoding_serving.md, quantization_moe.md | 65 papers, four subject files |
| [Lucrări universități SUA pentru Avalon/](../research_notes/Lucr%C4%83ri%20universit%C4%83%C8%9Bi%20SUA%20pentru%20Avalon/) | berkeley.md, cmu_uw.md, princeton_cornell.md, other_us.md | 131 papers, four subject files |
| [Lucrări Europa pentru Avalon/](../research_notes/Lucr%C4%83ri%20Europa%20pentru%20Avalon/) | eth_epfl.md, ista_central_europe.md, uk.md, france_south_nordics_israel.md | 69 papers, four subject files |
| [Lucrări Asia pentru Avalon/](../research_notes/Lucr%C4%83ri%20Asia%20pentru%20Avalon/) | tsinghua_pku.md, asian_labs.md, china_other.md, hk_taiwan_singapore.md, korea_japan_india.md | 97 papers, five subject files |
| [Lucrări Canada pentru Avalon/](../research_notes/Lucr%C4%83ri%20Canada%20pentru%20Avalon/) | montreal_toronto.md, waterloo_alberta_ubc.md | 56 papers, two subject files |
| [Lucrări Pacific Africa America Latină/](../research_notes/Lucr%C4%83ri%20Pacific%20Africa%20America%20Latin%C4%83/) | australia_nz.md, africa_latam.md | 63 papers, two subject files |

The URL census across all notes and reports: 1,817 unique URLs, of which 758 are arXiv.
The per-paper lists live in the mirrored `reports/*.md` files above.

---

## 3. Complete academic paper catalog (with direct URLs, findings and decisions)

Every paper below was cited in the MoE engine speedup notes, the pain-points notes, the
runtime architecture notes, or the university sweeps, and each carries an exact decision in
pancake-strata. Decision vocabulary: **Adopted** (in the default path), **Coded opt-in**
(built, off by default), **Prototype branch** (side branch, unmerged), **Planned** (priced,
not built), **Rejected** (with the reason), **Context** (informed reasoning, no code).

### 3.1 CPU expert compute, caching and prediction

| Paper | URL | Authors / venue | Summary | Decision in pancake-strata |
| --- | --- | --- | --- | --- |
| T-MAC | https://arxiv.org/abs/2407.00088 | Microsoft, EuroSys'25 | LUT-based mpGEMM for 1-4-bit weights on CPU | Rejected for rewrite: our Q2_0 kernel already runs at ~80% of the 33 GB/s DDR4, so a LUT gains ~0; only fewer bytes, overlap or routing help |
| Fiddler | https://arxiv.org/html/2402.07033v3 | 2024 | CPU executes missed experts, activations over PCIe | Context: confirms Strata's hybrid design; informed items 50-52 |
| FATE / Fate | https://arxiv.org/abs/2502.12224 | 2025 | Cross-layer gate prediction (97.15%) + shallow-favoring cache | Context: shaped the router lookahead and cache-priority reasoning |
| KTransformers | https://github.com/kvcache-ai/ktransformers | SOSP'25 | Rank guard: keep heavy CPU entries on time in their own layer | Coded opt-in: `STRATA_VERIFY_DEFER_CPU_TOP` (item 28); paper claims up to 1.45x |
| TEAL | https://arxiv.org/html/2408.14690v3 | MIT, FasterDecoding | Activation sparsity via activation-aware thresholds (40-50% on dense) | Rejected (item 13 closed): Flash-Next experts are dense; gate mask cos 0.844 at 70% sparsity |
| CATS | https://arxiv.org/html/2404.08763v4 | Stanford | Context-aware thresholding for MoE sparsity | Rejected (item 13): cos 0.818 at 70%; also superseded by TEAL/gate mask |
| Intra-expert activation sparsity gate mask | https://arxiv.org/abs/2605.08575 | Berkeley | Gate mask for intra-expert sparsity (Qwen3.5 MoE 84-87%) | Rejected (item 13): measured on our Q2_0 file, 11,027 records, gate cos 0.844 at 70% (oracle 0.954); FF-640 experts are dense |
| SpAx | https://arxiv.org/html/2610.02598v1 | 2026 | Fourth sparsity mask variant | Parked with item 13's closure |
| MoE-Infinity | https://arxiv.org/abs/2401.14361v3 | 2024 | Expert cache on personal machines, activation-matrix match | Context: its match idea is covered by the conversation working set (items 3-4) |
| SeqMoE | https://arxiv.org/html/2609.12978 | 2026 | Expert caching baseline in speculative comparison | Context for item 14 |
| SpecMD | https://arxiv.org/html/2602.03921v1 | 2026 | Speculative MoE decoding | Context for items 5/15/23 |
| HOBBIT | https://arxiv.org/html/2411.01433v2 | 2024 | Weight-based precision for offloaded experts | Parked: a variant of turbo; only one re-selection runs in verify.cpp |
| Bevy | https://arxiv.org/abs/2611.04794 | 2026 | MoE serving | Context |
| MoE-Beyond | https://arxiv.org/html/2508.17137v1 | 2025 | Pattern recognition for expert placement | Fed items 3-4 (per-conversation working set, per-domain profiles) |
| ExpertFlow | https://arxiv.org/abs/2410.17954v2 | 2024 | Expert flow scheduling | Rejected as engine work (low at batch 1); its pattern idea fed items 3-4 |
| FineMoE | https://arxiv.org/html/2502.05370v2 | 2025 | Fine-grained expert patterns | Fed items 3-4 |
| ProMoE | https://arxiv.org/html/2410.22134v2 | 2024 | Expert prediction with chunked, cancellable transfers | Its scheduling order goes into the item 14 simulator first; predictor recall measured 19-21%, so no build |
| Pre-Attention Expert Prediction | https://arxiv.org/html/2511.10676v1 | 2025 | Predict experts from pre-attention state | Rejected (item 30 closed): further from routing than the current layer's state |
| Local routing consistency | https://arxiv.org/abs/2505.16056 | 2025 | Routing locality | Context |
| MoE-Sieve | https://arxiv.org/html/2603.24044 | 2026 | Expert sieving | Context |
| Cache-Aware Router | https://arxiv.org/html/2609.04895v1 | 2026 | Router aware of cache state | Context for STRATA_ROUTE_PRIOR (turbo) |
| Speculating Experts | https://arxiv.org/html/2603.19289v1 | 2026 | Experts speculated ahead | Context |
| Apto | https://arxiv.org/abs/2611.01859 | 2026 | CPU/GPU hybrid assignment | Context for the hybrid cache |
| Apex | https://arxiv.org/abs/2610.23006 | 2026 | CPU/GPU hybrid assignment | Context for the hybrid cache |
| DraftExpert | https://arxiv.org/abs/2607.24434 | 2026 | Miss-aware draft windows | Tested: DraftPolicy's cost-aware window is already optimal; closed |
| EVICT | https://arxiv.org/html/2605.00342v2 | 2026 | Tree truncation for spec decoding | Informed DISCO and tail-row analysis (item 5) |
| EcoSpec | https://arxiv.org/html/2607.12696v1 | 2026 | Eco speculative decoding | Informed tail-row analysis |
| Limits of Speculation | https://arxiv.org/html/2609.22156v1 | 2026 | Draft-length saturation | Informed DISCO (feat/disco) |
| TreeWY | https://arxiv.org/abs/2608.20961 | 2026 | Tree-shared GDN state | Parked: not memory-bound here |
| SpecMoEOff | https://arxiv.org/abs/2508.21706 | 2025 | MoE offload + speculation | Context (survey set) |
| SP-MoE | https://arxiv.org/abs/2510.10302 | 2025 | Speculative MoE | Context; its draft-driven prefetch measured ~0 on PCIe 3.0 |
| Specter / MoE-SpeQ | https://arxiv.org/abs/2511.14102 | 2025 | Quantized speculative MoE | Context; feeds DraftPolicy::seed_costs (item 43) |
| MoE-SpAc | https://arxiv.org/abs/2603.09983 | 2026 | Speculative MoE | Context |
| SpecMoE | https://arxiv.org/abs/2604.10152 | 2026 | Speculative MoE | Rejected as engine work: gains at large batch; our drafter is already in VRAM |
| S2-MoE | https://arxiv.org/abs/2608.15018 | 2026 | Speculative MoE | Context |
| MoESD | https://arxiv.org/abs/2505.19645 | 2025 | Speculative MoE | Context |
| DynaSpec | https://arxiv.org/pdf/2510.13847 | 2025 | Dynamic speculative MoE | Context |
| SuffixDecoding | https://arxiv.org/abs/2411.04975 | CMU | Tree over prompt + all earlier outputs, shared by sub-agents | Planned (item 15): cross-session suffix index; simulate with the drafter's --simulate first |
| SuffixDrafter | https://arxiv.org/abs/2506.02780 | 2025 | Suffix drafting | Planned with item 15 |
| FreeToken | https://arxiv.org/abs/2608.16157 | FlashML-org | Checkpoint triggers; PCIe/CPU miss split | Item 22: very likely the same mechanism as `--pcie-frac`; parked pending the code read |
| Marconi | https://arxiv.org/abs/2411.19379 | Princeton, MLSys'25 | Checkpoint triggers + FLOP-saved-per-byte eviction | Planned (item 16): TTFT up to -71% on SWE-bench traces |
| Marconi (follow-up) | https://arxiv.org/abs/2605.08580 | Princeton | Continuation | Same as above |
| KVFlow | https://arxiv.org/html/2507.07400v1 | 2025 | Pre-staging expected state during idle | Planned (item 17): pre-stage the expected expert set while a tool runs |
| RadixAttention / SGLang | https://arxiv.org/abs/2312.07104 | Stanford | Multi-entry prefix cache | Planned (item 11 family / item 71 shared-prefix store) |
| Dynasor / Certaindex | https://arxiv.org/abs/2412.20993 | Hao AI Lab | Thinking early exit | Superseded: replaced by the learned stop (item 21, feat/disco) |
| Barbarians at the Gate / ADRS | https://arxiv.org/abs/2510.06189 | UC Berkeley | OpenEvolve-style policy search for caches | Planned (item 14 stage 2): only if the simulator shows >= 5% |
| SIEVE | https://github.com/cacheMon/NSDI24-SIEVE | NSDI'24 | Simpler cache eviction | Candidate policy, parked behind the >= 5% rule |
| S3-FIFO / Baleen / Hawkeye | (see item 14 in MOD_PLAN) | various | Cache policies | Same: Belady-bounded, one runs at a time |
| Mind the Gap | https://arxiv.org/abs/2505.23786 | ETH | GGUF-quantization backdoors | Adopted as practice: official files + SHA-256 vs Hugging Face |
| Displacement Is Not Direction | https://arxiv.org/abs/2606.19558 | Toronto | KLD/top-1 stop predicts task score | Adopted as method: revised quality gates (token forcing rejects large deviations only) |
| SpecFit | https://arxiv.org/abs/2610.15417 | Monash/Beihang | Marginal new-miss cost per draft position | Prototype branch: DISCO feature (feat/disco, item 36) |
| AcceptMoE | https://arxiv.org/abs/2608.02989 | Imperial | Depth-graded expert demand per verify row | Coded opt-in: `STRATA_ROUTE_PRIOR_DEPTH` (item 23, commit 4f56ce535) |
| WebGPU drafter | https://arxiv.org/abs/2610.39761 | 2026 | Browser-side drafting | Context only |

### 3.2 Quantization, kernels and GPU decode

| Paper | URL | Authors / venue | Summary | Decision in pancake-strata |
| --- | --- | --- | --- | --- |
| Scaling Laws for Precision | https://arxiv.org/abs/2411.04330 | MIT | Post-training quantization scaling laws | Adopted as guidance for bit allocation across quant choices |
| QTIP | https://arxiv.org/abs/2406.11235 | MIT | Extreme 2-bit via trellis | Rejected: no HIP kernel, slow on CPU, ~0.2 bpw gain, new tensor types |
| QuIP# | https://arxiv.org/abs/2402.04396 | ETH | 2-bit lattice quantization | Rejected: same as QTIP (not worth a requant vs IQ2_KT) |
| Medusa | https://arxiv.org/abs/2401.10774 | Princeton | Multi-head drafting | Covered: DraftPolicy's cost-aware window already prices length; no port |
| Hydra | https://arxiv.org/abs/2402.05109 | Notre Dame | Tree drafting | Rejected: each branch copies GDN state and wakes more experts |
| Sequoia | https://arxiv.org/abs/2402.12374v2 | CMU | Draft tree/length choice | Covered by DraftPolicy |
| TurboSpec | https://arxiv.org/abs/2406.14066 | 2024 | Draft length choice | Covered by DraftPolicy |
| EAGLE-3 | (see Asia report) | various | Trained drafter | Rejected: the native MTP head is better here |
| DSpine | https://arxiv.org/abs/2609.36173 | Tencent | Parallel drafter (DFlash family), acceptance 3.77 -> 4.82 on Qwen3-8B | Rejected: needs a drafter trained for the target model; released only for Qwen3-4B/8B; watch for a release for our models |
| DFlash2 / text-DFlash2 | https://github.com/ikawrakow/ik_llama.cpp (PR 2761) | ikawrakow | Calibrated survival window for drafts | Coded opt-in: `STRATA_DF2_GATE` / `STRATA_DF2_ROW_MS`; measured 115.9 short but 84.3 after a long prompt, so off by default |
| Mamba-2 / SSD | https://arxiv.org/abs/2405.21060 | MIT (FLA line) | State-space duality | Context for the GDN chunk-64 prefill item (18) |
| Mamba-3 | https://arxiv.org/abs/2603.15569 | MIT | Next SSM iteration | Context for item 18 |
| Mamba in the Llama | https://arxiv.org/abs/2408.15237 | MIT | Distilled hybrid | Context for item 18 |
| XAttention | https://arxiv.org/html/2502.17421 | Tsinghua | Block-sparse attention with antidiagonal scoring | Superseded: BFLA (CityU) benchmarks Flash-Next itself and won item 19 |
| SpargeAttn | https://arxiv.org/html/2411.10958v3 | Tsinghua | Training-free sparse attention | Context for item 19 |
| Mirage MPK | https://arxiv.org/html/2512.22219v1 | 2025 | Megakernel persistent kernels | Rejected: NVIDIA-only; our GDN megakernel prototype measured 0.0 ms on one GPU and was reverted |
| AutoMegaKernel | https://arxiv.org/html/2606.09682v1 | 2026 | Auto-generated megakernels | Same as above |
| Fleet | https://arxiv.org/html/2604.15379v1 | AMD | Fleet kernels | Context for the GPU floor (item 72) |
| HipKittens | https://arxiv.org/abs/2511.08083v1 | 2025 | HIP kernel primitives | Context; no direct port (CDNA-oriented) |
| PreScope / LayerScope | https://arxiv.org/html/2509.23638v1 | 2025 | Event-driven expert placement | Candidate policy for the item 14 simulator only |
| Klotski | https://www.arxiv.org/abs/2502.06888 | 2025 | Expert scheduling | Rejected: multi-batch |
| MoE-Gen | https://arxiv.org/html/2503.09716v1 | 2025 | Batch generation | Rejected: large batch |
| SLP stream-loading engine | https://arxiv.org/html/2606.10493 | 2026 | Stream loading | Context for items 38-54 |
| DFloat11 | https://arxiv.org/html/2504.11651v3 | 2025 | Lossless weight compression | Parked: 2.25 bpw leaves little entropy |
| EntroLLM | https://arxiv.org/html/2505.02380v4 | 2025 | Entropy-coded weights | Parked with the same trigger |
| Exponent concentration | https://arxiv.org/html/2510.02676v1 | 2025 | FP exponent statistics | Context for the wire-compression rejection |
| LServe | https://arxiv.org/html/2502.14866v2 | MIT Han lab | Long-context serving | Context (KV design) |
| ShadowKV | https://arxiv.org/html/2410.21465v2 | 2024 | KV compression | Rejected: KV lives in 1 of 4 layers, already q8 |
| DuoAttention | https://arxiv.org/abs/2410.10819 | MIT | Streaming heads | Rejected: loses mid-context code and tool output |
| Quest | https://arxiv.org/abs/2406.10774 | 2024 | Decode sparsity | Parked: attention > ~15% of decode at 100-200K is the trigger |
| H2O | https://arxiv.org/abs/2306.14048 | 2023 | KV eviction | Rejected with the whole KV-eviction family |
| TokenSpeed | https://github.com/lightseekorg/tokenspeed | LightSeek | Runtime architecture survey | Context: details flagged as inflated, no direct adoption |

### 3.3 University sweep corpora (counts, not itemized)

The seven university reports plus the runtime and hardware sweeps hold the rest of the 758
arXiv links. Deduped counts: Lucrări universități SUA (131), Lucrări Asia (97), Lucrări
Stanford (76), Lucrări Europa (69), Lucrări MIT Harvard (65), Lucrări Pacific/Africa/LatAm
(63), Lucrări Canada (56), Tehnologii drum de date hardware (66), Arhitecturi runtime LLM
2026 (31). Full per-paper URL lists are in the mirrored `reports/*.md` files (section 2).
The sweep-level decisions that mattered: per-level thinking caps (adopted), GDN chunk 64
(parked, prefill unmeasured), BFLA over XAttention (adopted on feat/bfla-hip), the
restore-equivalence test (planned, item 35), coordinate-descent kernel sweeps (planned,
item 37), and the GGUF security practice (adopted).

---

## 4. Complete open-source project and fork audit

Every project and fork examined, with URL, license, what was adopted, what was rejected,
and measured numbers where they exist.

| Project | URL | License | Adopted | Rejected / notes | Measured numbers |
| --- | --- | --- | --- | --- | --- |
| Strata (base) | https://github.com/Niko1221/Strata | MIT | The base engine, CUDA and HIP builds; upstream `146df3c4` (v0.2.12) | Stock defaults keep `STRATA_ROUTE_TAIL_SKIP=7` (lossy); pancake presets force 0 | Stock Flash-Next IQ2_XS on 3060: decode 54.5 tok/s mean, edit prompt 23.3 s; pancake 69.2 (+27%), 11.0 s |
| FastStrata | https://github.com/Jarson1408/FastStrata | MIT | Qwen3.6-35B options measured on a 3060 Laptop; `docs/OPTIMIZATION_LOG.md` | Nothing rejected | Its measured gains: `--attn-split` +30% at 37K, `--graph-experts` -7..-10% per window, agent turn after 38K 6.4 -> 3.9 s |
| PolyStrata | https://github.com/simplexai-labs/PolyStrata | MIT | The whole `src/poly` engine (strata-poly), ported as `df91582`/`a82999fd` | Nothing rejected | strata-poly vs stock strata on the 3060: short 100.7 vs 53.2 tok/s at 64K; edit 151.1 vs 39.4; 41.8K prompt 35.8 s / 71.7 tok/s vs 42.1 s / 9.7 |
| Strata-rdna3 | https://github.com/nicejunjie/Strata-rdna3 | MIT | All RDNA3 kernels hand-ported (upstream moved 1,938 commits since its base): `rdna_gr.cu`, `rdna_mmvq.cu`, COMBINE_CPU, WAIT_FUSE, ROUTE_PUBLISH; 59.6K draft vocab | Not ported: long-context top-k (= upstream SELECT_LONG), PF_GROUP_RES (= upstream group_resident), PA_KLDS (upstream rewrote as STRATA_PA_RPF), RDNA_PAIR (neutral), GDN megakernel (0.0 ms on one GPU, reverted) | Decode 87-101 -> 104-108 (kernels) -> ~112 tok/s (fusions); Strata-rdna3 itself 109-119; Avalon 95-100; stock 75-77 |
| Speedy | https://github.com/DevDock-AI/Speedy | MIT | The "fast" preset switch stack (prompt scratch #4039, routed stream #4283, IMMA prompt attention, KV prefetch, pin-by-need, HC Q8, adapt async/swaps) | Gains on our card were within noise to +9%, not the +22-33% measured on 10 GB 3080s; forks tuned on one machine need re-measuring | Second pass: after 42K decode 46.8 vs 41.4 (+13%), read 37.19 vs 38.39 s; short answers within noise |
| StrataFast / "Lamina" | https://github.com/eddoursul/StrataFast | MIT | Ideas for the Flash-Next phase: two-stage prompts, self-trained draft layer | Two-stage prompts blocked (needs a second GPU) | Their reports: 1.24-1.55x prompt reads at 32K-128K, 1.3-2.1x answers (their hardware) |
| basalt | https://github.com/jesdga95/basalt | MIT | Head kernels (`q5k_head.cu`, `head_fmt.cu`, `head_proxy.cuh`), grammar engine (`src/program/grammar`), 2-bit MTP draft proxy, idle-sleep idea | basalt's exact two-stage head on the Qwen3.6 Q8_0 head skipped (~1-2%); rewrote its history so nothing merges wholesale | IQ4_XS head GEMV x1.07-1.27; 2-bit draft pick 200/200 exact argmax, 0.113 vs 0.179 ms (x1.59); decode +3-5% |
| colibri | https://github.com/JustVugg/colibri | (see repo) | Idea kept: CACHE_ROUTE (cache-aware routing, lossy opt-in) | Rejected as engine: built for models larger than RAM; its Qwen3.6 decode ~30 tok/s on a 3070 vs strata-poly 100+ on a 3060 | CACHE_ROUTE at lambda 0.5: +25% decode, 2.9% tokens differ (stock noise 1.2-1.3%) = opt-in `turbo` |
| Aviary | https://github.com/SensAI-PT/aviary-hy3 | (see repo) | Nothing | Multi-machine cluster on colibri; 40 layers of LAN round trips cost more than a whole token | No gain on one PC |
| llamAmpere | https://github.com/JakeATX/llamAmpere | MIT | Ideas: adaptive MTP depth (candidate, exact), SJ-KVaRN KV body (candidate, opt-in), `--kv-gpu-tokens` idea | mb-MTP and `--moe-skip` lossy opt-ins parked; trellis weight formats not for this engine | SJ-KVaRN would free ~0.9 GB at 131K (~500 more experts); VRAM does not buy decode here (drafting is 2-5% of a window) |
| Project Maya | https://github.com/mw00/project-maya | MIT | Loop guard (`serve/loop_guard.py`, 690 lines), soft thinking budget close (#613), tolerant tool calls (#613/#706) | Not faster for us (its Qwen engine is a Strata 0.1.27 snapshot); resume-thinking-after-EOS not ported (both turn-end tokens are EOS here) | Server-side only; tests in `serve/test_loop_guard.py`, `serve/test_tolerant_tool_calls.py` |
| gpu-unlock | https://github.com/Zagitta/gpu-unlock | GPL-2.0 | Adopted on the 3060: removes GeForce's half-rate FP16/BF16/TF32 cap with FP32 accumulation | Does not change decode (CPU/memory bound) | cuBLAS 8192^3: FP32 sums 49.7 TFLOPS (locked ~25); prompt read 7.6 -> 6.0 s (-21%) on the arena pair |
| Avalon | (owner's install, source fast-forwarded c24c03d -> 88313b6) | (see repo) | Comparison target; its hipBLASLt table finding adopted | Its Flash-Next model (90 GB) was deleted to make room (owner's choice); `avalon.service` needs it re-downloaded | 7900 XT: Avalon 95-100 tok/s vs pancake ~112 |
| tandem.cpp | https://github.com/ewhacc/tandem.cpp (+ itlab-cl/tandem.cpp-amd) | MIT | `STRATA_POOL_PREREAD` ported verbatim (idle pool workers pre-read the lookahead's next-layer CPU experts) | MTP_DEFER already here; MTP head Q4_K requant not needed (ours is ~2-bit, 787 MB) | A/B pending; loses even on a noisy machine so far (stays off) |
| qwen38-megakernel | (cutlerbenjamin1-cmd) | AGPL | Ideas only (AGPL vs MIT): in-context draft-vocab tokens | Draft-vocab test: full vs en vs zh60k, no gain | zh60k vocab measured worse (92.5 vs 111.7 base) |
| llama.cpp | https://github.com/ggml-org/llama.cpp | MIT | PR watch: #30966 CPU share (same split as `--pcie-frac 0`), #31233 ClassLRU, #31888, #31286 | ClassLRU long-memory scoring replayed: decay 0.7..0.986 within 0.5 points, no gain; upstream expert-cache PRs #29887-#31888 rejected (pancake's tier + pcie-frac already do this; #30594 hurts MTP verify) | `--pcie-frac 0` = their share ~0.95, measured best (+6/+13%) |
| ik_llama.cpp | https://github.com/ikawrakow/ik_llama.cpp | MIT | DFlash2 window rule idea (`STRATA_DF2_GATE`) | DFlash2 measured 115.9 short / 84.3 after long prompt; upstream crash on QFN | Off by default |
| KTransformers | https://github.com/kvcache-ai/ktransformers | AGPL | Rank guard idea behind `STRATA_VERIFY_DEFER_CPU` (item 28) | Needs a 24 GB+ card as a full engine; only the idea ported | Paper claims up to 1.45x |
| SGLang | https://github.com/sgl-project/sglang | Apache-2.0 | Ideas: tree drafts, speculative sampling, GPU token mask (grammar), RadixAttention | Not our engine | Speculative sampling measured upstream on a 3060: default path as fast or faster |
| vLLM | https://github.com/vllm-project/vllm | Apache-2.0 | Nothing (checked and rejected as base) | Keeps the model in VRAM; needs far more VRAM/RAM | n/a |
| TensorRT-LLM | (NVIDIA repos + tech blogs) | (see repo) | Ideas: grammar inside speculative decoding, auto n-gram enablement | Rejected as base | n/a |
| FlashInfer | https://github.com/flashinfer-ai/flashinfer | MIT | Nothing | ROCm port is CDNA only | n/a |
| BitDecoding | https://arxiv.org/abs/2410.03397 + HIP port `feat/bitdecoding-hip` | (see repo) | HIP port of decode attention on gfx11 matrix cores (`STRATA_ATTN_WMMA`) | Not adopted as default: measured neutral | 96.3/84.9 base vs 90.1/95.6 wmma tok/s; prompt 1469.7/1498.3 vs 857.3 (outlier)/1463.4 |
| Rejected engines (one line) | vLLM, SGLang, KTransformers, TensorRT-LLM, LMDeploy, MLC-LLM, ExLlamaV2 (archived), PowerInfer, TGI (archived), LightLLM, NanoFlow, DeepSpeed-MII, TensorFold, jitllm (Go, Apache-2.0), IPEX-LLM, FPGA projects (LUT-LLM, llm-oneapi-fpga) | various | Nothing | Keep the model in VRAM, need far more VRAM/RAM, special model files, other hardware, or are archived | jitllm publishes only ratios, no tok/s; its expert paging is what Strata already does |

---

## 5. Engineering decisions ledger: ranked plan (items 1-95b) and combined improvements (63-73)

### 5.1 The time budget everything is priced against

| Request | Window | GPU | CPU experts | Draft | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| short chat | 24.8 ms | 15.9 | 5.2 | 2.2 | GPU spends ~42% of kernel time waiting for CPU rows (rocprofv3) |
| code edit (2K prompt) | 37.4 ms | 15.8 | 17.4 | 2.5 | CPU experts are 47% of the window; ideal overlap ~20 ms |

### 5.2 Substitute groups S1-S7 (gains inside a group do NOT add)

| Group | Members | Ceiling |
| --- | --- | --- |
| S1 CPU-expert time | 7, 8/28, 23, 5, 14/27/30 (+53, 3-4, 17), 29, 46-52, 40-41 | lossy +40-60% on edits (turbo + one other), exact-only +15-30% on edits (model), short chat <= 1.27x |
| S2 verify expert set | 7 vs 23 (vs OEA/BuddyMoE), pick ONE | 23 sits 5-15% above fast-hip; dropped if 7 ships |
| S3 window length/source | 15, 36, 43, `--spec-min-p`, `--lookup-max-t` | 0-10% combined |
| S4 thinking tokens | 12, 21, 34, effort default | -20-40% thinking on routine turns (est.) plus the p99 tail |
| S5 re-reads | 11, 16, 31, 39, 71 (+3 on resume) | ~15% of turn time (~8% in-session + ~7% sub-agent switches) |
| S6 tool-idle window | 17 vs AsyncFC vs Slipstream compaction | one GPU slot |
| S7 GPU floor | 1, 32/48, 37, 47, Q2_0-down, 10, BitDecoding | pays on short chat now; on edits only after S1 brings the CPU part below ~15.8 ms |

### 5.3 Tiers A-D of the ranked plan (items 1-95b)

**Tier A, prerequisites (no gain of their own):** 73 (wiring: pancake.service on 11437,
per-turn JSONL log, reasoning_tokens in usage), 35 (restore-equivalence test, gates
3/11/16/31/39), 67 (quality/perf harness, gates 7, 8/28, 23, 21), 49+50+44 (measurement
batch: combine_cpu timer, spin vs row pull, per-layer slack), 64 (shared window trace,
CODED 14a4b3373), 42 (overlap simulator, DONE as `time_ceiling.py --overlap`), 8
(non-finite bisect, DONE: cause was STRATA_ROUTE_PUBLISH, fixed d5b58f669).

**Tier A, levers:** 7 (turbo-hip default for some modes, +30-50% measured, lossy, gate
pending), 66 (agent-turn pipeline, 15-25% of agent wall time est.), 12 (thinking caps in
the preset, code done), 8+28 (expert deferral with rank guard, UNBLOCKED, model +33% edits,
+kmiss +45-52%), 38 (Linux direct expert-arena loader, start 28-65 s -> ~10-14 s est.), 2
(closed: no refill cost, decode_curve.py), 14/27/30 (time-based cache simulator, oracle
+27..48% edits), 31 (preserve_thinking), 11 (message-boundary checkpoint), 16 (tool-boundary
checkpoints + Marconi eviction), 69 (memory budget arbiter), 39 (automatic disk tier), 71
(shared-prefix store for sub-agents), 21 (learned thinking stop).

**Tier B (3-10%):** 68 (miss-cost-aware speculation), 23 (AcceptMoE depth prior, CODED
4f56ce535), 15+36 (DISCO + SpecFit features, feat/disco), 65 (overlap package judged by the
simulator), 1 (GPU path profile), 72 (GPU-floor package), 48 (fuse pre-doorbell chains,
CODED as STRATA_EXPDEC_SWIGLU and STRATA_ROUTER_FUSE), 63 (capability probe +
calibration), small exact flag stack (+1.4..+3.9% measured).

**Tier C (1-3%):** 29 (STRATA_POOL_FUSE_PHASES, coded), 3-4 (conversation working set +
domain profiles, merged), 47 (Infinity Cache warm), 53 (STRATA_ADAPT_SWAPS_AUTO, coded),
46 (STRATA_COMBINE_POLL1, coded), 40 (STRATA_Q2_PREFETCH, coded), 43 (DraftPolicy
seed_costs), 10 (MTP draft in graph), 5 (tail-row drop, replay built), 20 (ReplaySSM
audit), Q2_0-down expdec kernel, 51 (CPU pre-sums), 37 (coordinate-descent sweep), 41
(thread count + governor), 17 (pre-stage during tool runs), 18 (GDN chunk 64, parked), 70
(selftune.py, offline pass built), 33/34 (behaviour counters, rumination signal), 25
(Coder variant A/B), 19 (BFLA, measured, branch unmerged).

**Tier D (< 1%, hygiene, hardware-gated):** 52 (ReBAR push), BitDecoding default, 45 (I/O
hygiene), 54/57/55/61/56/58/59/60/62 (platform items, folded into 63), 22 (FreeToken code
read), hygiene batch (MMQ=OFF stubs fixed, test_keepawake, README 176.9B, presence_penalty
0), plus the code-study items 74-95b (MTP overlap, persistent adapt worker, HIP sync
policy, hot-path hygiene, staged-prefill CPU share, PLE gather off critical path,
hipBLASLt descriptor cache, GDN occupancy, prefill instrumentation, restore hooks, seed
virtual miss bits, huge pages, RAM budget sizing, reasoning_tokens DONE 2d5d44709,
preset-only agent settings, grammar routes FIXED bcbdeb5d2, auto-pin branch save,
pancake.service, build matrix DONE 635c2a752, tests build dir, typed flag registry, split
generate.cpp, stub blocks).

### 5.4 Combined improvements (items 63-73)

Each combines existing items because the combination is worth more than its parts or costs
less together. All opt-in; the default path stays byte-identical until a measured A/B
changes a default.

| Item | Combines | Why together | Gain |
| --- | --- | --- | --- |
| 63 Capability probe + calibration | 55, 61, 57, 56, 58, 41, 53, 43, 54, 38's NVMe rate | One start-up probe cached per machine fingerprint keeps derived settings consistent with each other | 0-3% here, 6-13% on other platforms (static PcieAuto lost that much here) |
| 64 Shared window trace | I1, 14/27/30, 42, 43, 36, 15, 23, 5, 28, 3-4, 17 | One trace run replaces ~10 GPU runs; with one GPU, queue time is the scarce resource | 0 itself; unblocks 7/14 decisions and 8/28 pricing |
| 65 Overlap package | 46, 48, 47, 51, 40, 29, judged by 42 | Separately each is 0-3%, under the 20% noise; together one A/B batch and one model | 5-10% on edits, 3-8% short est. (not additive) |
| 66 Agent-turn pipeline | 35, 31, 11, 16, 39, 12, 3, 17, 73's log | One conversation key and one lifecycle instead of five switches that can disagree | ~15% of turn time + tens of seconds per switch + the thinking tail: 15-25% of agent wall time est. |
| 67 Capability-aware harness | I4, I5, nightly A/B, quality gate, regression watch, 63 | The three biggest levers (7, 8/28, 21) are all lossy and blocked on the same gate | 0 itself; unblocks +30-50%, +10-25%, -20-40% |
| 68 Miss-cost-aware speculation | 15, 36, 43, 23, 28, with 63's calibration | One per-position cost vector feeds three decisions | exact part 2-6%, with AcceptMoE 5-15% on edits est. |
| 69 Memory budget arbiter | 38, RAM_KEEP vs arena, parked budget, 39, 2, 61, oomd | Four decisions fight over the same 57 GB / 20 GB; deciding apart is how the arena trips oomd | start 28-65 s -> ~10-14 s est., arena kept (+8%), +20% after long prompts |
| 70 Closed-loop self-tuning | 64, 4, 8.6, 43, 14, 12, 63, 67 | Profile-type gains decay as the workload drifts (8.6 alone gave +6.7%) | 1-5% est., plus no silent decay |
| 71 Shared-prefix store | 11, 16, RadixAttention, prompt reordering | Reorder alone or a store alone gains nothing; together switches become suffix reads | up to ~7% of turn time |
| 72 GPU-floor package | 1, 32/48, 37, 10, Q2_0-down, 47, triggered by 44+64 | Avoids building kernels whose ms vanish into the combine_cpu spin | 5-10% short chat est., later on edits |
| 73 Wiring | pancake.service, per-turn log, reasoning_tokens | No engine gain reaches the user without it | 0 itself; prerequisite for everything per-turn |

### 5.5 Current Dev/Test status (from [../PANCAKE_STATUS.md](../PANCAKE_STATUS.md), 2026-11-09 18:13)

| Prio | Item | MOD_PLAN | Expected gain | Where it is | Dev | Test |
| --- | --- | --- | --- | --- | ---: | ---: |
| P1a | Expert deferral on HIP (non-finite fix) | 8 | up to 1.45x, realistic +10-25% code edit | FIXED: bisect found ROUTE_PUBLISH; now off under deferral (d5b58f669); 512+512 tokens clean | 100% | 10% |
| P1a | Deferral rank guard (STRATA_VERIFY_DEFER_CPU_TOP) | 28 | lossy-small, keeps quality | coded in host pool + GPU copy, builds, probe clean | 100% | 5% |
| P1b | Expert cache judged in ms (simulator) | 14 | decides the policy | tools/pancake/time_ceiling.py: oracle +27..48%, online 2nd-miss +5..16% (code edits, model) | 100% | - |
| P1b | More swaps per window (existing --adapt-swaps) | 14 | +4..9% code edit (model) | nothing to build | 100% | 0% |
| P1b | Admit on k-th miss in W windows (new, STRATA_ADAPT_KMISS) | 14 | +5..16% serial, +45..52% with deferral (model) | in both tiers (kmiss_tier.hpp); must run with --adapt-async 1: the blocking tier stalls the GPU ~5.6 ms per swap round (profile); the kmiss+async probe was interrupted at engine start, not yet measured | 100% | 0% |
| closed | Predictor recall / candidate policies | 27, 30 | - | lookahead recall 19-21% of misses (measured): predicted promotions do not pay; 27 covered by kmiss + conversation seeds | 100% | 100% |
| P1b | Simulator: per-layer GPU/CPU overlap model (--overlap) | 14 | decides 8/28/29 | built: code edit deferral +33%, deferral + kmiss 2/8 +45..52%, item 29 +3% (model, alpha 0.25-0.5) | 100% | - |
| closed | CPU sparsity masks (Berkeley gate mask, TEAL, CATS) | 13 | +10-20%, lossy | REJECTED offline: at 70% sparsity gate mask cos 0.84 (~54% error), oracle 0.95; experts are dense -> no sparse kernel (bench/results/2026-11-09-sparsity) | 100% | 100% |
| P1d | One CPU pool phase per layer (STRATA_POOL_FUSE_PHASES) | 29 | 1-4%, exact | coded, builds (35c512984) | 100% | 0% |
| P1e | Domain / conversation expert seeds | 3-4 | +0.6-1.5 hit points | merged, builds; needs traces | 100% | 0% |
| P2a | turbo-hip (lossy router prior) decision | 7 | +30-50% | preset exists; quality gate pending | 100% | 30% |
| P2b | AcceptMoE depth-graded verify prior (STRATA_ROUTE_PRIOR_DEPTH) | 23 | turbo speed with less quality loss | coded, builds, probe clean | 100% | 0% |
| P2b | BFLA block-sparse prompt attention (feat/bfla-hip) | 19 | long-context TTFT (paper 1.34-1.44x at 128-256K) | e2e measured on Qwen3.6 UD-Q4_K_XL: 128K prompt 386.9 -> 1219.8 sparse (3.15x) -> 771.2 dense tok/s; recall top-1 48/48, KL 0.0014/0.0045; branch unmerged, HIP only | 100% | 10% |
| P2b | SpecFit missed-expert feature | 36 | 0-10% | belongs to DISCO (other session, feat/disco) | - | - |
| P2c | Small exact flags stack | phase 11 | +1.4..3.9% (provisional) | flags exist; A/B won 3/3 (noisy machine) | 100% | 50% |
| P2c | --lookup-max-t 12 | Avalon report | +1% (noise) | flag exists | 100% | 30% |
| done | Matching hipBLASLt table (100401) | Avalon report | +38% prompt (provisional) | in the 7900 XT presets | 100% | 80% |
| done | Pool pre-read (tandem.cpp) | phase 11 | - | loses even on a noisy machine; stays off | 100% | 90% |
| P3a | Turn-boundary checkpoints (existing switches) | 11 | seconds per turn | nothing to build | 100% | 0% |
| P3a | Restore test, preserve_thinking, tool-boundary checkpoints | 35, 31, 16 | seconds per turn | other session (dan-b5) | - | - |
| closed | Decode speed kept after long prompts | 2 | - | NOT the refill: same reply after a 13.8K prompt decodes as fast as after 2K (tools/pancake/decode_curve.py); the drop is prose content + cache warm-up -> items 3-4, 14 | 100% | 100% |
| P3c | Effort defaults, learned thinking stop | 12, 21, 33, 34 | tens of s per turn | other session (dan-b5) | - | - |
| P4 | GPU floor: kernel fusion, window-edge host stalls, parameter sweep | 32, 37 | 1-5% each | deferral profile: GPU still waits for the CPU 31% of kernel time -> fusions wait; the real host stall is the blocking tier's swaps (-> --adapt-async) ; tools/pancake/gpu_profile.py built | 20% | 0% |
| P4 | --mtp-q4 draft head | phase 11 | draft cost | flag exists | 100% | 0% |
| P1 | Shared window trace (STRATA_WINDOW_TRACE + tools/pancake/trace/wtrace.py) | 64 | 0 itself; one trace run answers ~10 offline questions | logger + reader + --dump-routing converter (14a4b3373) | 100% | 0% |
| P1 | Single poller in combine_cpu (STRATA_COMBINE_POLL1) | 46 | 0-3% | coded (93f7a963d) | 100% | 0% |
| P1 | Online swap budget (STRATA_ADAPT_SWAPS_AUTO) | 53 | 0-5% over the best fixed swaps | coded both tiers, controller unit-checked (6bc9530bc) | 100% | 0% |
| P1 | Q2_0 row prefetch (STRATA_Q2_PREFETCH=<bytes>) | 40 | 0-5% of CPU-expert time | coded (c44b1ad7b) | 100% | 0% |
| P1 | Lazy trailing park in the CPU pool (STRATA_POOL_LAZY_PARK) | code study | 0-1% | coded (c44b1ad7b) | 100% | 0% |
| P4 | expdec gate/up + SwiGLU + q8_1 in one launch (STRATA_EXPDEC_SWIGLU) | 48 | ~49 launches/window fewer | coded, bitwise by construction (2aca56777) | 100% | 0% |
| P4 | Router GEMV + top-10 + doorbell in one launch (STRATA_ROUTER_FUSE) | 48 | ~1% short chat (48 launch gaps/window) | coded, shared route code, bitwise by construction (7c0f03228); HC chain was already one kernel | 100% | 0% |
| P1 | Parked workers wait with MWAITX (STRATA_POOL_MWAITX) | code study | frees SMT siblings; wake +0.36 us | coded, micro-checked on the 5700G (4de66f87a) | 100% | 0% |
| meas | waitCPU stamped inside combine_cpu (profile) | 49 | measurement | coded (7bcb77d5e) | 100% | - |
| meas | dispatch 'jobs' split: expert-source sub-timer | - | measurement (jobs = 1.5 ms/window) | coded (c44b1ad7b) | 100% | - |
| P1 | CPU pre-sums missed rows (STRATA_CPU_PRESUM) | 51 | 0.5-1.5%, not bit-identical | coded, tied to the captured graph (cc5499fae) | 100% | 0% |
| P1 | Pre-summed rows to VRAM through ReBAR (STRATA_PRESUM_VRAM) | 52 | 1-2% | coded, HIP fine-grained VRAM (4f829da0e) | 100% | 0% |
| P1 | Tail-row drop | 5 | 0-5% | trace record + offline replay built (0488b80f4, 5c3d8a9fb); dan-9f adds the drafts() call with item 74; engine change only if the replay shows a gain | 60% | 0% |
| Ph14 | Test phase in one command (bench/specs/01-07 + tools/pancake/test_phase.sh) | 14 | - | built (1fbe4e2b1): abtest + quality gate per spec, resumable | 100% | 0% |
| P4 | Infinity Cache warm-up during the CPU wait (STRATA_IC_WARM_MB) | 47 | 0-2 ms/window | coded (269a06ed9) | 100% | 0% |
| P2 | Draft window costs from traces (STRATA_SPEC_COST_TABLE in serve + draft_costs.py) | 43 | 0-5% | coded (663107a56) | 100% | 0% |
| P2 | Draft stream on its own HW queue (STRATA_MTP_HIP_PRIO, existing, off on gfx1100) | 10 | R9700: draft 5.8 -> 3.5 ms | arm added; item 10 closed by code study | 100% | 0% |
| closed | ReplaySSM audit | 20 | - | replay stays (code study) | 100% | 100% |
| P1 | Pool quarter-size tail tasks + down-row prefetch while waiting (basalt ports) | 29-related | 0.5-1.5% | coded (e29bac539) | 100% | 0% |
| P1 | Tier catch-up while idle between requests (STRATA_IDLE_ADAPT) | 17 | 1-3% next turn | coded (43e741bc6) | 100% | 0% |
| P4 | Q2_0 down on the EXPDEC kernel (STRATA_EXPDEC_DOWN, existing) | Q2_0-down | 1-3% short | arm added | 100% | 0% |
| Ph14 | Launch/tile/pool sweep tool (tools/pancake/sweep.py) | 37 | 1-3% | built (7861dc8c1) | 100% | 0% |
| P1 | Per-layer dispatch: 4 quadratic loops -> linear (first_of, miss list, per-expert groups, tail-skip) | code pass (pancake-docs/CODE_OPTIMIZATION.md) | dispatch plan/jobs time | coded, exact by construction, HIP+CUDA build | 100% | 0% |
| Ph14 | Application end to end (agent-session timeline, idle-time use, server -> engine turn info) | 14 | per-turn time | needs the per-turn JSONL log of item 73 (serve/, dan-9f, not started) | 0% | 0% |
| Ph14 | A/B harness with quiet-machine gate + noise monitor (tools/pancake/abtest.py) | 14 | clean numbers | built (f21cc86a1) | 100% | 0% |
| Ph14 | Quality gate as a process (tools/pancake/quality_gate.py, token forcing per lossy arm) | 14 | - | built (c7c44f2d0) | 100% | 0% |
| Ph14 | Regression watch (tools/pancake/regression_watch.py) | 14 | - | built (c7c44f2d0) | 100% | 0% |
| Ph14 | Self-tuning from own sessions (tools/pancake/selftune.py, item 70) | 70 | 1-5% (keeps profiles current) | offline pass built (05e041b9b): traces -> candidate profile -> time model -> abtest spec; dry run: candidate profile +8.6%, + kmiss +23% modeled | 100% | 0% |
| Ph14 | Diagnostic: decode speed along a reply (tools/pancake/decode_curve.py) | 2 | - | built, used to close item 2 | 100% | - |

#### Items by session dan-9f (server, agent, loader, draft area) - merged into port/rdna3-decode

| Item | What | Switch / file | Default | Exact? | Dev | Test |
| --- | --- | --- | --- | --- | ---: | ---: |
| 73 | Per-turn JSONL log | `turn_log` / `pancake-turns.jsonl` | on (Flash-Next presets) | - | 100% | 0% |
| 74 | Eager MTP draft steps (all launched at once, chain cut on the host) + `window_trace().drafts()` | `STRATA_MTP_EAGER=1` | off | exact | 100% | 0% (spec 07, identity check) |
| 75 | Persistent adapt worker (one JobThread) | `STRATA_ADAPT_PERSIST=0` restores old | **on** | exact | 100% | 0% (spec 02) |
| 76 | Device spin scheduling | `STRATA_DEVICE_SPIN=1` | off | exact | 100% | 0% (spec 04, log must say "set") |
| 38 | Linux O_DIRECT expert-arena load | `STRATA_UNBUFFERED_LOAD=0` restores buffered | **on** (Linux) | exact | 100% | 0% (test plan 2.10) |
| 35/83 | Restore-equivalence test | `tools/pancake/restore_check.py` | first step of test_phase.sh | gate | 100% | 0% |
| 88 | Agent settings in all Flash-Next presets | env_to_user, effort_position end, grace 512, close retry | on | text | 100% | - |
| status page | `/status`, `/pancake`, `/status.json` (opencode/Gemini session, `214c8cd31`) | serve/web/status.html | on | - | 100% | tests 13/13 |
| 33 | Loop-guard behaviour counters (edits without build/test, reads without edit) | `STRATA_LOOP_BEHAVIOUR`, config `loop_guard_behaviour` | **log** (note opt-in) | exact in log mode | 100% | read counts from real sessions |
| 34 | Rumination signal (always measured) + opt-in thinking close | config `rumination_close` | measure on, close OFF | lossy only when on | 100% | pick a threshold from turn logs |
| 31 | preserve_thinking passthrough + keep_reasoning keeper | `chat_template_kwargs.preserve_thinking`, `STRATA_KEEP_REASONING` | off | exact | 100% | opencode already resends reasoning (22/25 exact reuse) |
| 80 | hipBLASLt descriptor cache + table extender tool | `STRATA_HIPBLASLT_DESC_CACHE` | **on** | exact | 100% | 0% (spec 09-prefill; extender run = GPU ~30 min) |
| 90 + 71 | Park the branch at the pin; auto-pin system+tools prefix | `STRATA_PIN_PARK_TOKENS` (on), `STRATA_AUTO_PIN` (off) | see switch | exact | 100% | 0% (sub-agent switch check) |
| 16 | Tool-boundary checkpoints + FLOPs-per-byte eviction | `STRATA_CACHE_TOOL_BOUNDARY`, `STRATA_CKPT_EVICT_FLOPS` | off | exact vs re-read | 100% | 0% (`restore_check.py --tool-boundary`) |
| 39 | Disk tier for parked conversations | `STRATA_CONVERSATION_DISK_DIR` | off | exact | 100% | 0% (`restore_check.py --disk`) |
| 79 | PLE gather off the critical path; compact PLE with direct I/O | `STRATA_PLE_PIPE`, `STRATA_PLE_COMPACT` | off | exact (CPU byte parity) | 100% | 0% (cold 32K TTFT arms) |
| 86 | RAM-budget conversation cache (opt-in) + PSI guard | `STRATA_CONV_BUDGET`, `STRATA_CONV_KEEP_GIB`, `STRATA_PSI_GUARD` (on) | see switch | exact (timing) | 100% | 0% |
| 93 | Tests labelled cpu/gpu/model, strata_tests target | `ctest -L cpu` 117/0/6, `ctest -L gpu` 124 (parity tolerances untested) | - | - | 100% | cpu done, gpu in test phase |
| 95b | Stubs in *_stub.cpp + check_stubs.py; dead code; keepawake flake | - | - | no behaviour change | 100% | 100% |
| cleanup | STRATA_ARENA_WARM removed (measured worse); native_moe_combine_multi_hits kept (upstream code, unused; removal would only add fork drift) | this session | - | - | 100% | - |

Test plan: [../bench/TEST_PLAN.md](../bench/TEST_PLAN.md).  Code pass: [CODE_OPTIMIZATION.md](CODE_OPTIMIZATION.md).

## 6. The complete not-adopted / rejected ledger (with exact reasons and reopen triggers)

Everything the research series and the measurement rounds looked at and did not take.
[closed] = decided; [parked: trigger] = reopen when the trigger happens. Sources: MP =
MOD_PLAN round or item, Up = Avalon upgrades report, Stan/MIT/SUA/EU/Asia/Can/Pac =
university reports, PP = pain points, TG = technology families, HW-* = data-path notes.

### 6.1 Measured on this box: no gain, or worse

| Technique | Evidence | Status / reopen trigger |
| --- | --- | --- |
| CPU intra-expert activation sparsity (Berkeley gate mask, TEAL, CATS; + CHESS/SpAx/per-layer thresholds/union sparsity) | STRATA_ACT_DUMP_RAW on the Q2_0 file, 5 layers, 11,027 expert-token records: at 70% sparsity gate mask cos 0.844 (~54% relative error), CATS 0.818, oracle |h| 0.954; FF-640 experts are dense | [closed] Reopen: a model with sparse experts (gate cos >= 0.99 at >= 70%) |
| CPU lookahead thread (RouterLookahead predictor on CPU) | -13% decode (70.5 -> 61.4 tok/s); its 30-34 ms of router GEMVs fight the pool for cores and DDR | [closed] Replaced by the GPU lookahead (+10%) |
| History-only expert predictors (prev-window, EMA frequency, transition dictionary, co-activation, hybrid) | cover 20-29% of misses at ~65% waste; the lookahead is ~2x better | [closed] Reopen: recall@16 >= 80% in a trace run |
| MTP-head router predictor (item 10.3 as its own path) | the GPU lookahead already routes the verify window's draft tokens | [closed] |
| Foresight swap space fed by the lookahead (STRATA_FS_SLOTS) | 0.97x at 4 slots; 8 slots run the 5.3K prompt out of VRAM (cuBLAS status 14) | [parked: only from idle memory, if a lending source appears] |
| VRAM lending (item 10.5) | `--kv-resident 8192` gives +5% slots, decode 68.5 vs 70.5 (noise) | [closed] |
| Per-layer cache budgets (item 7) | cross-layer replay 48.3% vs 48.4% per-layer | [closed] |
| KV body (SJ-KVaRN keys 8-bit) and mb-MTP | VRAM does not buy decode here; drafting is 2-5% of a window | [closed] |
| Cache-aware routing at lambda 2.0 | 9.9% mismatch, 8 large-margin events: REJECT (0.5 stays the opt-in turbo) | [closed] |
| ClassLRU long-memory scoring (llama.cpp #31233) | replayed decay 0.7..0.986 within 0.5 points at 32 swaps/window | [closed] |
| Reuse-aware admission | a PCIe 3.0 promotion (~115 us) costs ~3 CPU computes of the same expert | [closed; k-th-miss admission goes through the item 14 simulator] |
| L3 prefetch (STRATA_ARENA_WARM) | worse; DDR4 already at ~80% | [closed] |
| CPU pool settings sweep | defaults best | [closed] |
| IQ2_S two-half decode (STRATA_IQ2S_PAIR) | neutral; the kernels are compute-bound | [closed] |
| `--spec 3/5/6`, window thresholds | flat; `--spec 4` is also the official 4.07-accepted depth | [closed] |
| Upstream group-A switches all together | 101.2 vs 111.7 base | [parked: one at a time in the item 1 GPU work] |
| Round A worse/flat arms: ADAPT_LAG 87, upstream HC_Q8 107.2, rdna3 expert profile 105.6, rdna3 adapt settings 116.4 (own 121.2 wins), `--kv-resident 131072` 106.2, STRATA_SPEC_COST 112.0/98.3, SELECT_LONG and mtp-window flat | vs 111.7 base | [closed] |
| zh60k draft vocab; full / en vocab | zh60k 92.5 worse; full and en flat | [closed] |
| mmap'd expert tier as default | 108.6 vs 111.7 (and 69 vs 75 on the 3060) | [parked: RAM policy (RAM_KEEP vs oomd) decision] |
| HW queues 4 vs 6 (GPU_MAX_HW_QUEUES) | flat | [parked: a concurrent stream is added] |
| DFlash2 | 115.9 short but 84.3 after a long prompt; upstream crash on QFN | [closed] |
| RDNA_PAIR | neutral, off upstream | [closed] |
| GDN decode megakernel (Strata-rdna3) | 0.0 ms on one GPU: the window waits on CPU rows; prototype reverted on feat/pattern-cache | [parked: second GPU / tensor parallel] |
| DraftExpert-style miss-aware windows | tested: DraftPolicy already optimal | [closed] |
| ContextPilot block dedup (feat/contextpilot) | 1.6% savings on 40 real opencode sessions; non-prefix reuse breaks GDN state | [closed] |
| HIP graphs, global unroll flag, COMPUTE power profile, draft p-min, checkpoint-count tuning, AMD-Ecosystem fork, Vulkan, turbo4/TurboQuant KV | tested with no gain | [parked: only with a new reason, e.g. a profile showing dominant inter-kernel gaps] |
| `--pcie-frac` > 0 / zero-copy misses (pcie_mode 1) | `--pcie-frac 0` measured best (+6/+13%); a 1.4 MB GEMV over Gen3 is >= 107 us vs 42 us on the CPU | [parked: F1 frees DDR4 headroom] |
| Strata gfx1100 switches: int8 WMMA attention, TSUM, expert layouts | neutral or negative on 7900-class cards in Strata's own tests | [closed] |
| VERIFY_DEFER_CPU on HIP (pre-fix) | non-finite logits (#879) with the ported kernels; bisect found STRATA_ROUTE_PUBLISH as the cause, now off under deferral (d5b58f669); post-fix probes clean at 111.8-152.7 tok/s | [parked -> UNBLOCKED; test phase decides] |
| BitDecoding STRATA_ATTN_WMMA | neutral: decode 96.3/84.9 base vs 90.1/95.6 wmma; prompt 1469.7/1498.3 vs 857.3 (outlier)/1463.4 | [parked: ~2-3% of attention confirmed] |
| `--ple-io mmap` | long prompt 42.9 -> 40.4 s but decode not better; PLE stays on direct reads | [closed] |
| STRATA_PF_WMMA prompt expert kernel | slower on both 5.2K and 41.8K prompts | [closed] |
| STRATA_PF_FUSED int8 prompt kernels as default | -7% at 42K but +12% at 5K; agent turns read short deltas | [parked: long cold prompts dominate -> chunk-size switch] |
| Upstream TAIL_SKIP=7 as default | buys 18-22% decode by dropping low-weight experts; token forcing +0.84 pts mismatch (INCONCLUSIVE, clear rate increase) | [closed for default; stays opt-in] |
| Upstream expert-cache PRs #29887, #30637, #30966, #31888, #31286, #30594, #27861 | pancake's tier + `--pcie-frac` already do this; #30594 hurts MTP verify | [closed] |
| PLE-in-RAM | model + 28.8 GB table + 16 GB headroom > 57 GB RAM | [closed] |
| tandem.cpp POOL_PREREAD | loses even on a noisy machine | [parked: quiet-machine A/B in the test phase] |

### 6.2 Needs retraining, fine-tuning or a BF16 requant of the 125B model

Kimi Linear, MoBA, Kimi K2 arch, Jet-Nemotron, stick-breaking (architectures the shipped
weights do not have); PaTH, log-linear attention, KL-guided hybrid distillation; LoLCATs,
Hedgehog linearization (impossible locally for 125B); pre-gated MoE, SiDA-MoE, Read-ME;
ProSparse, TurboSparse, Sparsing Law ReLU-fication, BlockFFN (Flash-Next experts are
SwiGLU); Deja Vu (superseded by CATS, which item 13 then closed); DMC, DMS KV compression;
FoX + ACP; NSA, InfLLM-V2, Sparse Flash Attention, Landmark Attention,
AdaSplash/ASEntmax, DuoAttention, FlashMoBA; Cartridges (backprop through the 125B);
Next-Concept Modeling, Quiet-STaR, SUPO, Think-Prune-Train-Improve; PASTA, Co-LLM,
ThinkPrune, SEAL/TTT for ARC; VeriThinker, LASER, TokenSkip, CodePromptZip; CLLMs, Jacobi
Forcing; EAGLE-3, GRIFFIN, SpecBlock separate drafters (the native MTP head is better
here); SLiM, PATCH, OPTIMA (GGUF has no 2:4 sparsity); Mixture of Pruners / USP pruning
line. All [closed]. Parked on disk space: expert-sensitivity mixed-precision requant (needs
the ~250 GB BF16 model), QuaRot (354 GB BF16), LQER, QERA. GSQ re-quantization at home,
GPTQ-2D, WUSH, QMoE, GPTVQ, FPTQuant, LR-QAT: requant/QAT impossible locally and the daily
files are already GSQ-RCO [closed]. R5: item 25 (Coder) already ships quantized [parked:
if a new base file is needed].

### 6.3 Hardware or platform not available

FP8 KV cache (no FP8 tensor cores on sm_86); FP8 Gated-Residual stream (irrelevant at
batch 1 on RDNA3, only the fusion carries over); FlashAttention-3 (Hopper only);
FlashInfer (ROCm port CDNA only); Mirage MPK, NanoFlow, Punica, Atom (CUDA or batch);
whole-layer/model megakernels MPK, FlashDMoE (NVIDIA; WAIT_FUSE already is the in-kernel
hand-off); persistent post-bell kernel, FlashDMoE actor style [parked: F1/F2 shrink CPU
below hit time]; ThunderKittens / HipKittens direct port, Look Ma No Bubbles megakernel
(NVIDIA or CDNA code); Hexcute, Tilus, K-Search (CUDA-only kernel tools); Flash-LLM,
FP6-LLM (A100 tensor cores; "load compressed, compute dense" is already in pancake);
Marlin, QUIK, GPTQ/SparseGPT/AQLM/PV-Tuning/HIGGS kernels (CUDA-only or superseded by
GSQ); Disaggregated Quantization (NVFP4, prefill below 4K gets slower); BitDecoding as
published (NVIDIA tensor cores; the HIP port stays opt-in); llama.cpp #31802 fused-dequant
MMA FA, #31533, #28714, #28195, #26301, #26592, #27140 (NVIDIA-only / RDNA3.5-only / CUDA
bug, or no gain for 512-expert top-k); hipFile (AMD AIS/GDS: fastpath absent, data on
ntfs3) [parked: newer amdgpu AIS + ext4/xfs + Radeon support]; BaM / GPU-initiated NVMe,
NVIDIA SCADA, device-triggered SDMA promotion (misses come from RAM at ~7x the NVMe rate);
cache_ext eBPF page-cache eviction (not on the stock kernel); DECA, StreamDQ, TriMoE
near-core/in-DRAM decompression (hardware absent); SparAMX (needs AMX and unstructured
weight sparsity); LIA (AMX/CXL); MoNDE, Pimba, "SSD offloading harmful" line (near-data/PIM
hardware); STRATA_POOL_UMWAIT port to Linux (shared power budget only on laptops);
hipMemcpyBatchAsync batched promotion copies (absent in ROCm 10) [parked: ROCm adds it and
promotion volume rises]; LP-GEMM (AVX-512 + prepacked GEMM, ~0 for AVX2 GEMV). Second-GPU
work parked as a group: 8.7 two-stage prompts, tensor parallel, layer split + pipelined
windows, peer experts/RCCL, kfdq idle parking, P/D disaggregation, KV streaming, xLLM-style
heterogeneous split [parked: second GPU installed]. DeepSeek V4.1 on gfx1100 (239-274 GB,
D too small) [parked: disk].

### 6.4 Targets batching, multi-user or multi-GPU throughput; nothing at batch 1

FlexGen (only the placement cost model idea kept); Hydragen, Tokasaurus (need many
sequences with a shared prefix); NEO, Pie (serving), Jenga, PagedAttention, S-LoRA (Pie's
online swap-budget idea is a separate HW-pcie candidate); MegaScale-Infer, Comet, UltraMem
(cluster); MiniMax-01/M1, LongCat-Flash, GLM-4.5, Pangu Pro MoE, ERNIE 4.5, Hunyuan-Large
(other models); MoE-Gen, BatchGen; MoEShard [parked: second GPU]; DSI; KTH priority-aware
scheduling, DeltaZip (multi-tenant); SpecMoE (gains at large batch); ExpertFlow (low at
batch 1); OEA sibling-aware re-routing (measured at batch 16) [parked: idea only]; Klotski,
MoE-Lightning pipelines, MoE-Lens scheduler (only paging and the roofline tool carried
over); PowerInfer-2 planner (phone NVMe) [parked: arena shrunk by RAM_KEEP]; Vidur,
LLMServingSim (cluster simulators; only their validation method is copied, item 42);
PrefixShield, PrefixPlace, Gimbal (multi-tenant, multi-GPU); HiveMind [parked: parallel
sub-agents]; AMESA (service-level mixture); chunked-prefill interleave, overlap scheduler,
PASCAL reasoning-aware scheduling, distributed KV stores [parked: several users/agents at
once]; llama.cpp #30992/#30970/#29738 multi-slot scheduling [parked: second slot for title
requests]; RSA, Scorpio, RL^V, Weaver, Archon, PRM calibration, Reasoning with Sampling (N
samples or many LLM calls per answer); Mooncake (cluster KV store; only the NVMe-tier idea,
phase 7 / item 39); SPRINT (fine-tuning + batch > 1). All [closed] unless marked.

### 6.5 Conflicts with exact-prefix cache, GDN recurrent state or determinism

Draft trees (Staged SD, SpecInfer, Sequoia trees, SGLang tree.cuh, Hydra top-2 tree): each
branch copies GDN state and wakes more experts [parked: watch upstream #30931 only].
Lookahead decoding, parallel Jacobi (wide guesses enlarge the expert union) [closed]. Token
Recycling (longer drafts add new CPU-expert misses) [closed]. SWIFT, CAS-Spec layer-skip
self-speculation (break the GDN state) [closed]. Entropy Gate, Prompt Cache non-prefix
reuse (any change earlier in the prompt invalidates the GDN checkpoint) [closed].
KV-streams (the idea is already the GDN checkpoint) [closed]. Out-of-core streaming RFC
#31469 / #25294 (conflicts with every cache) [closed]. #31497 no MoE cache for the draft
(contradicts the shared MTP cache) [closed]. Grammar during thinking ("Let Me Speak
Freely?": constraints stay after the thinking block) [closed].

### 6.6 Lossy with poor quality evidence or no payoff here

Router residency bias as a default (Strata data shows loops in long agent sessions; 0.5
stays the opt-in turbo) [parked: turbo-hip quality gates]. AdapMoE, BuddyMoE, MoBiLE,
HOBBIT weight-based precision (variants of turbo; only one re-selection runs in
verify.cpp) [parked: if turbo loses to them in the gates]. DAOP (lossy, CPU already
saturated) [closed]. MoE compression study (~8% quality loss unacceptable for an agent)
[closed]. REAP-style rare-expert pruning (drops API/library memory first) [closed]. KVA
approximate late-layer prefill (decode -11% and quality risk) [closed]. StreamingLLM,
DuoAttention streaming heads (lose mid-context code and tool output) [closed]. KV
eviction/compression family: H2O, PyramidKV, SnapKV, KVQuant, XQuant, Palu, xKV, ChunkKV,
KVzip, InfiniGen, Q-Filters, TOVA, OSCAR, KV INT2, ZipCache, MiniCache, Counter-Causal
Surprise, InnerQ, QServe KV4 (KV lives in 1 of 4 layers, already q8; TurboQuant measured
0) [closed]. Dynasor-style early exit (needs a forked probe of the single stream; replaced
by the learned stop, item 21) [closed]. SliceGPT (adds loss on top of 2 bits) [closed].
FloE-style lossy transfer (belongs to F1 and its gates) [closed]. Cactus KL-budget
acceptance (zero under greedy, ~0 at T=0.1) [parked: owner moves to higher temperature].
Relaxed drafting (basalt; draft step is 2-5% of a window) [parked: idea only]. YaRN static
factor 4 (may hurt short texts; 262K native is enough) [parked: contexts > 262K].

### 6.7 Superseded by something already in pancake, or by a better candidate

CATS (superseded by TEAL / Berkeley gate mask, then item 13 closed all three) [closed].
XAttention (superseded by BFLA, which benchmarks Flash-Next itself) [closed]. IQ2_XS as
daily file, item 6 (Q2_0 chosen: 93.8 vs 70.3 tok/s, -0.09 task avg) [closed]. IQ1_M
native AVX2 row kernel, item 24 (0 on Q2_0) [closed]. T-MAC, Vec-LUT, bit-serial LUT GEMV
(Q2_0 kernel at ~80% of DDR4, row_multi already amortizes) [parked: micro-benchmark shows
compute-bound]. Any-Precision, MC-MoE, MxMoE (no bits left below Q2_0) [parked: a >2-bit
file such as Coder ~3.5 bpw]. QTIP / IQ*_KT, CALDERA / LPLR (no HIP kernel, slow on CPU,
~0.2 bpw gain) [closed]. SmoothQuant, AWQ, QServe W4A8, SVDQuant, SqueezeLLM (imatrix +
IQ codebooks + q8 activations already cover it) [closed]. Entropy-coded weights
EntroLLM/Huff-LLM, Q2_0 wire compression (2.25 bpw leaves little; CPU decode costs more
than it saves) [parked: entropy check shows >= 20%]. Non-temporal loads/stores, extra cache
blocking (WB arena, no cross-window L3 reuse) [closed]. HybriMoE, DuoServe-MoE, Fiddler,
CPU-GPU collaborative MoE (covered by `--pcie-frac` / PcieAuto / item 30) [closed].
FreeToken PCIe/CPU split, item 22 (very likely the same as `--pcie-frac`) [parked: code
read shows copied experts are also promoted]. DALI greedy per-expert placement,
PreScope/HybriMoE event placement ("GPU if t_gpu < t_cpu" almost never fires; simulator
candidate only) [closed as engine work]. Draft-driven expert lookahead MoE-SpeQ / SP-MoE as
prefetch (~0 on PCIe 3.0; recall logged only) [closed]. Speculative linear pre-compute of
next-layer projections (doubles weight reads; MALL warm-up, item 47, captures the benefit)
[closed]. GPU work stealing of CPU misses (same DDR4; `--pcie-frac 0` won) [parked: F1
frees DDR4 headroom]. Full/empty bits, per-row readiness (already the partial flag
protocol) [closed]. Spin backoff changes, wave priority s_setprio (detection is
PCIe-bound; spinners sleep already) [closed]. hipStreamWaitValue32, host callbacks /
blocking events instead of the spin (~0 alone, OS wake latency) [parked: concurrent GPU
work on another queue]. CPU writing rows into VRAM via ReBAR (<= 0.27 ms) [parked: combine
pull measurement (item 50) > 3 us/layer]. raise_flag_b_ordered -> hipStreamWriteValue32
(only matters with `--pcie-frac` > 0) [parked: pcie-frac back above 0]. 8.5 adaptive MTP
depth and 8.8 fused FA prompt kernel on main (poly already has the cost rule; attention
6.7% of a 42K prompt) [closed for main engine]. Basalt draft-head screen, K/V append beside
the draft query (draft step 2.5-3.2 ms of 46-130 ms: 1-2% max) [closed]. TurboSpec, Sequoia
length choice, Speculative Decoding + Batching, Su-Pekhimenko adaptive length, TLT,
DISCO-as-new (= DraftPolicy) [closed]. EAGLE / EAGLE-2 / Hydra / Cassandra (= native MTP +
`--spec-min-p`) [closed]. BanditSpec UCB in DraftPolicy (DraftPolicy is already a greedy
bandit) [parked: logs show bad exploration]. MTP head Q4_K requant (tandem.cpp) and
MTP_DEFER (ours is ~2-bit, 787 MB; MTP_DEFER already here) [closed]. Upstream PF_GROUP_RES,
long-context top-k, PA_KLDS from Strata-rdna3 (= upstream group_resident / SELECT_LONG /
PA_RPF) [closed]. #31750 KV-only catch-up, #31037/#31031 trimmed draft head, #31669 GDN
replay, #27489 buffer sharing (already present, or "frees VRAM" which buys nothing here)
[closed]. Maya resume-thinking after an end-of-turn token (both turn-end tokens are EOS;
empty answer handled by #1053) [closed]. Dense 4.x tier / Qwen 4 27B through expert
offload (offload does nothing for a dense model) [parked: if it ships open, choose a dense
engine]. MEM1, Self-Steering/DisCIPL, Graph of Thoughts, KGoT, RAPTOR, Generative Agents,
STOP, ReAct/ToT/MemGPT/Gorilla/RAFT/TinyAgent/Autellix (conceptual or already in opencode)
[closed]. Local-Splitter (pancake is the local model) [closed]. MoESQ presence-penalty
setting (kept as a rule: presence_penalty=0) [closed].

### 6.8 Gain too small for the effort / parked until a measurement says otherwise

Per-layer PcieAuto refinement (share 0.20-0.25) [closed]. Fused attention prompt kernel on
main (attention 6.7% of a 42K prompt) [parked: prefill profile on the 7900 XT says
otherwise]. F7 prefill work: GDN chunk 64 (item 18), BFLA beyond prototype [parked: run 2b
prefill timing]. Quest, TriForce, MagicDec, MagicPIG, TidalDecode, The Sparse Frontier
decode sparsity [parked: attention > ~15% of decode at 100-200K]. Exact QSA scorer (only
above 131K) [parked: contexts > 131K]. Calibrated verify gate STRATA_DF2_GATE (no numbers)
[parked]. Expert predictors as a build, prefetch on top (history 20-29%, ProMoE 1.09x even
on PCIe 4.0) [parked: recall@16 >= 80% in trace run 2a]. Cache policies beyond k-th-miss
admission: SIEVE/S3-FIFO, OpenEvolve, Hawkeye, Harmonia, MoE-Infinity, FlashMoE (+7% vs
LRU) (Belady-bounded, one runs at a time) [parked: item 14 simulator shows >= 5%]. Item 26
semantic oldString constraint (one failure in 180 edits) [parked: failure rate rises].
feat/bitdecoding-hip as default (attention ~3% of decode) [parked: ~2-3% confirmed].
Staged-SD small tree via #30931 (large effort, GDN fork) [parked: upstream lands it].
SSD-lite (Saguaro) precomputed drafts (CPU already busy) [parked: host-gap profile between
verify steps]. Typical acceptance on thinking only (Medusa) [parked]. Draft source as a
bandit [parked: DraftPolicy logs show wrong choices]. Athena epoch bandit for pre-read
[parked: the POOL_PREREAD A/B is mixed]. Delethink resets [parked: > 16K-token thinking
turns]. AsyncFC / Slipstream in tool windows vs expert pre-staging [parked: phase 14
timeline]. io_uring backend for DirectFile (~1.06x drop-in; 16 threads already give QD16)
[parked: PLE waits on the critical path]. Frequency-ordered PLE shard layout [parked: PLE
wait time measured]. ext4/xfs migration, kernel 7.1 "ntfs" driver (no documented read
gain; ntfs-3g avoided) [parked: only for hipFile]. MADV_POPULATE_WRITE prefault, extra THP
work (load-time only, low single digits) [parked: arena load measured after item 38].
Pinning cost via hugetlbfs for hipHostRegister (unmeasured part of ~10 s setup) [parked:
setup timing]. Coalesced promotion copies, STRATA_COPY_HIP_PRIO default (0-1%, likely flat
at pcie-frac 0) [parked: one-line A/B]. MonoMoE weight-major layout for VRAM-hit experts /
Q2_0 down (hit kernels hidden behind the CPU) [parked: per-layer critical path (item 49)
shows hits > CPU]. Expert-level MADV_SEQUENTIAL/WILLNEED (only if experts fault from NVMe;
the arena removes faults) [parked: major-faults per turn > 0 on mmap tier]. Disk prompt
cache #28092 (pancake has session files; automatic tier is item 39) [superseded by item
39]. Sleep-in-place (basalt 64167ba73; only if the GPU is wanted for other work while
idle) [parked]. Upstream #31625 HIP fast-math, #28118 on-device spec checkpoints,
#27210/#27173 deeper MTP, #31271 effort discovery, #30689 engine split (no numbers / deep
drafts touch more experts / rebase hazard) [parked]. SpecCache, Cartograph, AutoCompact,
ArbiGraph, MMLU-Pro tier (local tools are fast / small tool set / minor) [parked: large MCP
catalog]. JRT (doubles prefill) [closed]. MegaBlocks (training-oriented; mul_mat_id covers
batch 1) [closed]. Flash-Decoding split heuristics (already in the vec kernel) [parked:
item 37 sweep]. Markovian Thinker, Flash Inference, Kinetics, Critical Thinking (very long
thinking only / convolutions / context) [closed].

---

## 7. Complete benchmark, profile and test ledger

### 7.1 Hardware and time-budget profile (RX 7900 XT: rocprofv3 and time_ceiling.py)

**Window breakdown** (fast-hip preset, `bench/results/2026-11-08-7900xt-rounds/profile-fasthip.txt`):

| Request | Window | GPU | CPU experts | Draft | Sync |
| --- | ---: | ---: | ---: | ---: | ---: |
| code edit (2K prompt) | 37.4 ms | 15.8 | 17.4 | 2.5 | ~1.7 |
| short chat | 24.8 ms | 15.9 | 5.2 | 2.2 | - |

With perfect overlap the same parts fit in ~20 ms: the edit window is a scheduling/overlap
problem, not a bandwidth problem. The CPU Q2_0 expert kernel already runs at ~80% of the
33 GB/s DDR4 (pool-sweep.txt: pool defaults best), the same 80-90% band Intel reports for
the best 2-bit x86 GEMV, so a LUT rewrite (T-MAC / Vec-LUT) gains ~0.

**Kernel-by-kernel GPU time** (rocprofv3, `bench/results/2026-11-09-7900xt-gpuprof/run_kernel_stats.csv`):

| Kernel | Share of GPU kernel time | Detail |
| --- | ---: | --- |
| combine_cpu_k10_vec4 | 38.3% | 4,128 calls, avg 254 us, min 5.6 us: >= 97.8% of it is waiting for CPU rows |
| native_down_multi_kernel<42> (Q2_0 down, VRAM hits) | 5.8% | 4,224 calls, 37.6 us each; hidden behind the CPU on edits |
| wait_flag_ge_kernel (under deferral) | 31.1% | `bench/results/2026-11-09-gpuprof-defer`: expdec_gu 8.8%, wait_flag_copy 7.0%, 1,111 ms idle launch gaps, 550 ms host stalls of 5,741 ms: under deferral the GPU waits on the CPU (P4) |

**Cache ceilings** (replay at 10,909 cached experts = 44% residency): engine tier 79.6%,
per-layer LRU 89.5%, Belady 94.7%; tier every window x160 swaps 85.0%.

**time_ceiling.py** (serial model, GPU + CPU summed as the engine runs today): LRU -20%
short, oracle admission +8..15% short / +27..48% edit, online 2nd-miss-in-8 +5..16% edit
(built as STRATA_ADAPT_KMISS). Overlap model (`--overlap`, commit 751e09b70; per layer
max(GPU part, CPU part + sync), DMA contention alpha 0.25-0.5, calibrated on measured
windows): code edit: item 29 +3%, deferral +33%, deferral + rank guard (half kept on time)
+24%, deferral + k-th-miss 2/8 +45..52%; short chat: deferral +27%. Model only; validation
to 5-9% error is in the test phase.

**Data-path findings**: with `--pcie-frac 0` the critical-path transfer is not the expert
upload but the fp32 row pull in combine_cpu (a full 2560-float, 10 KB row per missed
(token, expert) slot from mapped host memory; est. 0.4-0.8 ms per edit window against a
<= 0.27 ms GPU-side bound; item 50 measures it before items 51-52 are built). Promotions
already run at ~93% of the Gen3 link (115 us vs 107 us theoretical per expert). The Linux
expert-arena loader reads through the page cache: 33.02 GiB at 0.49-1.18 GiB/s, starts
28-65 s (item 38 targets ~10-14 s via O_DIRECT).

### 7.2 RX 7900 XT (HIP, gfx1100) decode and prefill suites

**Port result** (branch port/rdna3-decode): decode 87-101 -> 104-108 tok/s (RDNA3 kernels)
-> ~112 tok/s (fusions). Comparisons: Strata-rdna3 itself 109-119, Avalon 95-100, stock
Strata 75-77.

**Round A** (load-gated, base 111.7 / 92.5 after an 11.7K prompt):

| Arm | Result |
| --- | --- |
| `--pcie-frac 0` | 118.4 / 104.3 (+6/+13%) |
| + adapt every window, 160 swaps | 121.2 / 100.1 |
| rdna3's adapt settings | 116.4 |
| `--kv-resident 8192` | 114.4 |
| STRATA_SPEC_COST | 112.0 / 98.3 |
| DFlash2 | 115.9 / 84.3 (worse after long prompt) |
| lossy: turbo | 149.3 |
| lossy: tail skip 7 | 129.7 |
| upstream group A switches together | 101.2 (worse) |
| ADAPT_LAG | 87 (worse) |
| zh60k vocab | 92.5 (worse) |
| rdna3 expert profile | 105.6 |
| upstream HC_Q8 | 107.2 |
| `--kv-resident 131072` | 106.2 |
| mmap tier | 108.6 |
| flat | `--spec 5/6`, full vocab, SELECT_LONG, mtp-window, hw queues |

**Rounds B/C and final suites** (2026-11-08 evening, `bench/results/2026-11-08-7900xt-final-suites.txt`):
presets fast-hip (default) / hip (exact) / turbo-hip (commit f0f9c5788); token forcing
INCONCLUSIVE +0.4 pts, 0 large-margin events; `strata_bench --long` and the
strata-7900xt bench beat Avalon and stock Strata. Headline numbers: iq2_xs fast-hip ~126
tok/s, q2_0 fast-hip ~135 tok/s (80.6% same top token as Q8_0), turbo-hip ~168 tok/s.
Prompt reading: 1,850-1,920 tok/s at 11.7K with the matching hipBLASLt table (1,345
without it). Q2_0 vs IQ2_XS on DASLab's card: 93.8 vs 70.3 tok/s decode at -0.09 task
average, so Q2_0 is the daily file (item 6 closed).

**Deferral probes** (`defer_probe.sh`, 512+512 tokens, STRATA_VERIFY_DEFER_CPU=1): the
pre-fix plain probe hit #879 non-finite twice (before the first token and after 3; the
engine re-sent and dropped its cache both times). Post-fix (ROUTE_PUBLISH off, commit
d5b58f669) clean, and so are the depth-prior, rank-guard (top-2) and lookahead-window
(10/16/24) variants at 111.8-152.7 tok/s. The kmiss+async probe was interrupted at engine
start and has not run.

**Lookahead recall** (item 30 closed, `bench/results/2026-11-09-lookahead-recall/`):
STRATA_LOOKAHEAD_STATS=1, fast-hip, short + code edit, K 10/16/24: the next-layer router on
layer l's input covers 19-21% of layer l+1's non-resident experts (precision 13-16%,
3.6-5.8 predicted/layer). At ~117 us of DMA per promotion vs 23-60 us of CPU per miss,
predicted promotions do not pay; predictors 2-3 are further from the routing: not built.

**Prefill path findings** (fast-hip, STRATA_PREFILL_TIMING=1, 41,818-token prompt): 42.9 s
(976 tok/s). GPU timeline: expert GEMMs 29% (gate/up 17.8, down 8.5, dequant 2.9), GDN
~12%, PLE 10.6% (4.3 s on the host), QSA attention 6.7% + projections 5.5% + select 2.6%.
`--ple-io mmap`: long prompt 42.9 -> 40.4 s, PLE host 4.3 -> 3.3 s, decode not better: not
adopted. Staged prefill: 4K reads 1,757 vs 2,580 tok/s at 32K (item 78's target).

### 7.3 BFLA block-sparse flash attention (branch feat/bfla-hip)

**Single-layer kernel benchmarks** (qwen35 engine, gfx11, chunks of 4096):

| Length | hipBLAS + materialised softmax | Fused RDNA3 WMMA dense | Q-from-global dense | BFLA sparse | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32K | 1,238.7 ms | 581.1 ms (max diff 3.7e-5) | 515.2 ms | 500.2 ms (33.4% of blocks) | 2.5x |
| 64K | 5,224 ms | 2,321 ms (max diff 6.4e-5) | 2,066 ms | 1,090 ms (selection 12.9 ms) | 4.8x |
| 128K | 28,287 ms | 9,553 ms (max diff 1.3e-4) | 8,512 ms | 3,335 ms (31.2% of blocks; selection 48.9 ms) | 8.5x |

Over 10 layers at 128K: 283 s -> 96 s (dense) -> 33 s (sparse). Q-from-global was ~11%
faster at every length (candidate default, pending end-to-end). `bfla_test` passes: dense
vs FP64 rel L2 1.3-1.5e-4; sparse-all-blocks bitwise equal to dense.

**End-to-end on Qwen3.6-35B-A3B UD-Q4_K_XL** (7900 XT, feat/bfla-hip build, noisy machine,
runner2.log, `--bench-pp` prompt throughput, 3 runs each):

| Length | BFLA off | BFLA dense | BFLA sparse |
| --- | ---: | ---: | ---: |
| 64K | (died silently once, rerun pending) | 1,054.1 / 1,058.2 / 1,058.9 tok/s | 1,227.6 / 1,284.1 / 1,286.9 tok/s |
| 128K | 387.6 / 386.8 / 386.9 tok/s | 772.2 / 771.2 / 773.8 tok/s (2.0x off) | 1,217.6 / 1,219.8 / 1,220.2 tok/s (3.15x off) |

**Needle recall** (`tools/pancake/bfla_recall.py`, 6 needles, 48 new tokens): prompt TTFT
dense 760.0 vs sparse 1,276.6 tok/s at 64K (1.68x), 388.3 vs 1,204.3 at 128K (3.10x);
forced-vs-dense top-1 agreement 48/48 at both lengths; mean KL 0.00142 (64K) and 0.00448
(128K). The dense continuation itself listed 3 of 6 codes in 48 tokens. The script's
analysis step crashed on an off-by-one (the forced run dumps 49 logit rows, dense 48); the
metrics were recovered offline from the saved logits (`recovered.json` in the run dirs);
the bfla greedy text was not saved, so its recall count needs a rerun.

**Flash-Next tile-shared QSA overlap** (STRATA_SEL_OVERLAP=all, IQ2_XS, 131,071-token
prompt): 16-query tiles read 6.3% of the KV cells at position 0 and 16-32% at 8K-129K; the
shared set holds 42-74% (16-query) / 42-71% (64-query) of the cells. Prefill 572.1 s
(229.1 tok/s). The run then exited 2: `--spec` needs the device residency table
(`--expert-profile`, `--expert-cache` and the token graph), so no generation from this
run. Item 19's tile-shared form is CLOSED: a shared set holds only 42-74% of each query's
QSA cells (falling with length), too lossy for a <= 7-8% prefill ceiling.

### 7.4 RTX 3060 12 GB (CUDA, sm_86) benchmarks

**strata vs strata-poly** (Qwen3.6-35B UD-Q4_K_XL, greedy, cold runs,
`bench/results/2026-11-07-pancake-baseline.jsonl`):

| Case | strata, 16K ctx | strata, 64K ctx | strata-poly, 131K ctx |
| --- | ---: | ---: | ---: |
| short chat, 256 new | 64.6 tok/s | 53.2 tok/s | 100.7 tok/s |
| code edit, 5.3K prompt, 512 new | 40.1 tok/s | 39.4 tok/s | 151.1 tok/s |
| 41.8K prompt: read / decode | - | 42.1 s / 9.7 tok/s | 35.8 s / 71.7 tok/s |

strata keeps fp32 KV (45 KB a token); the 131,072-context setup runs out of VRAM on 12 GB
(an upstream setup bug); at 64K no routed expert fits in VRAM. strata-poly keeps a Q8 cache
(11.7 KB a token) and ~4,500 of 10,496 experts in VRAM. The two engines' greedy texts part
after 25-81 tokens (different kernels, Q8 KV): expected for two engines.

**Where the time goes** (strata-poly, Nsight Systems, 15K-token prompt, 10.4 s): GPU
kernels 9.7 s; gdn_state_reg_kernel 21.6%, gemm_small_kernel 12.9%, MMQ expert products
~30%, attention ~16%. At 42K attention is the largest cost: Q.K^T 4.04 s, P.V 4.13 s,
softmax 3.10 s, rescale + Q8->fp16 0.81 s (about 12 s of 32 s, 38%). Decode: CPU experts
4.1 s of 8.75 s of writing at 33.4 GB/s, close to what dual-channel DDR4 delivers.

**CPU threads**: 6 to 12 threads equal within noise (short 100.3 / edit 150.4 at 6; 98.0 /
154.7 at 8); 4 and 16 slower. Keep one thread per physical core (8).

**Exact kernel changes** (bit-identical, `poly_gdn_cols_test` / `poly_gemm_small_test`):

| Change | Result |
| --- | --- |
| DeltaNet 4 state columns of a head per warp (gdn_state_cols_kernel) | gdn_step x1.38 (1K tokens), x1.44 (4K); same bits for out, state and snapshots; 2 columns x1.27, 8 columns x1.05 |
| Small fp32 GEMM register-tiled 4x4 (gemm_small_tiled_kernel) | router 2048->256: x2.26 (4K tokens), x1.99 (1K); same bits; used only for >= 64 outputs and >= 256 tokens |
| Attention softmax with a thread's scores in registers (attn_softmax_reg_kernel) | read once instead of twice; identical tokens; read 16K 9.58 -> 9.47 s, 42K 32.78 -> 32.30 s (x1.015); STRATA_SOFTMAX_REG=0 restores |

End to end, interleaved old/new/new/old, daily flags (prompt read time): 4K 3.12 -> 2.86 s,
16K 10.74 -> 9.60 s, 42K 35.76 -> 32.89 s, edit 3.97 -> 3.63 s. Exactness rule: with the
expert cache on, strata-poly is not run-to-run deterministic (the cache resizes and swaps
by timing and free VRAM); exactness checks must use a fixed placement
(`--no-gpu-experts --expert-swaps 0`), where greedy tokens are identical old/old, new/new
and old/new.

**Specialized output head kernels** (`head_fmt_test`, 2560 x 248,320; draft pick on 40,525
rows, peaked heads):

| Format | GEMV speedup (1 / 2 columns; 5-8 columns) | Draft pick: exact argmax / time vs GEMV alone |
| --- | --- | --- |
| IQ4_XS | x1.07 / x1.27; x1.24-1.02 | 200/200, 0.113 vs 0.179 ms (x1.59) |
| Q6_K | x1.03 / x1.18; x1.17-1.09 | 200/200, 0.114 vs 0.263 ms (x2.31) |
| Q4_K | x1.02 / ...; x1.19-1.07 | 200/200, 0.114 vs 0.182 ms (x1.60) |

At 3-4 columns native_mmvq_il is faster and stays. The IQ4_XS head GEMV alone: 1 column
1.079 -> 1.006 ms, 2: 1.462 -> 1.152, 5: 2.788 -> 2.250. End to end on IQ2_XS: greedy
tokens identical on/off; the 2-bit MTP proxy (basalt's packing and scoring, near-best
rescored exactly) gave decode short 52.3 -> 54.0 (+3%), edit 72.8 -> 76.8 (+5%), drafts
accepted the same (short 172 of 231-242, edit 456-463 of 466-483), noisy (load ~5).

**basalt 2-bit MTP draft proxy correctness**: 200/200 exact argmax on peaked heads (an
activation = a row plus noise, as a draft head is); on a flat random head no 2-bit copy can
tell the rows apart (basalt's own pick then agrees with the exact argmax about half the
time).

**Flash-Next IQ2_XS on the 3060** (main engine, single cold runs then interleaved pairs):

| Config | short | edit | after 42K: decode / read | read 4K / 16K |
| --- | ---: | ---: | ---: | ---: |
| base (upstream defaults, TAIL_SKIP=7) | 67.0 | 87.7 | 52.8 / 38.5 s | 5.8 / 15.2 s |
| exact (TAIL_SKIP=0) | 54.9 | 72.8 | 43.4 / 38.5 s | 5.9 / 15.2 s |
| exact + Speedy's switches | 54.4 | 74.9 | 47.3 / 37.1 s | 5.5 / 15.7 s |
| exact, mean of 2 | 49.2 | 74.7 | 41.4 / 38.39 s | 16.98 s (16K) |
| + Speedy's switches, mean of 2 | 54.6 | 73.9 | 46.8 / 37.19 s (+13% / -3%) | 18.65 s (16K) |

MTP draft vocabulary (exact config, lossless): default 106K tokens short 53.2 / edit 74.4;
English 40.5K 55.7 / 78.2 (best for English and code, +5%, ~215 MiB less VRAM); zh60k
59.6K 53.2 / 72.7. STRATA_VERIFY_DEFER_CPU (PRs #2689/#2690 as re-ported by Speedy): short
57.7 (+8%), edit 69.0 (-7%), changes the output: opt-in, off by default.

**Against stock Strata** (Flash-Next IQ2_XS, both engines reading experts from the GGUF,
mmap tier, 20 GiB kept free; `bench/prefetch/{stock-mmap,ours-mmap,ours}.jsonl`):

| Run | short 1 (cold) | short 2 | edit 1 | edit 2 | mean | edit prompt (5.3K) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stock, mmap tier | 43.9 | 65.3 | 54.5 | 54.2 | 54.5 | 23.3 s |
| ours, mmap tier | 53.1 | 59.8 | 77.4 | 86.6 | 69.2 (+27%) | 11.0 s |
| ours, RAM arena (12 GiB kept) | 57.9 | 70.2 | 81.9 | 92.1 | 75.5 | 7.6 s |

Code edits gain most (+42% / +60%: drafts, learned profile, GPU lookahead); prompt reading
is 2.1x faster. gpu-unlock measured on top: decode unchanged (74.9 vs 75.5, CPU/memory
bound as predicted), prompt read 7.6 -> 6.0 s (-21%) on the arena pair; cuBLAS 8192^3 FP32
sums 49.7 TFLOPS (locked ~25).

**Learned expert profile** (8.6, `tools/pancake/learn_profile.py`): decode
52.8/63.4/75.8/90.0 -> 57.8/70.8/80.9/91.4 tok/s (+6.7%, every run faster, most on cold
first runs). **GPU lookahead default** (STRATA_LOOKAHEAD_GPU, commit 7ba03d6fe): on the
mmap tier decode 52.0/64.1/72.6/79.2 -> 56.5/67.6/81.0/90.0 tok/s (+10%, every run faster);
same predictions as the CPU thread (k=10 recall/precision 0.576/0.548 vs 0.576/0.553);
predictor thread 30-34 -> 0.6-0.7 ms per window.

**Cache-aware routing (turbo) quality** (token forcing, stock noise 1.17-1.30% mismatch):

| lambda | mismatch | vs stock (90% CI) | large-margin events | verdict | decode |
| --- | ---: | ---: | ---: | --- | ---: |
| 2.0 | 9.90% | +8.67 pts (5.71, 11.95) | 8 | REJECT | 88.1 tok/s (+17%) |
| 0.5 | 2.89% | +1.66 pts (1.00, 2.41) | 0 | REJECT (rate), opt-in only | 94.1 tok/s (+25%) |

**Preset comparison** (single cold runs, decode tok/s, load ~4): Qwen3.6 UD-Q3_K_M (poly)
82.8 / 153.9 / 83.0 after 42K, read 32.5 s; Qwen3.6 UD-Q4_K_XL (poly) ~100 / 120.6 / 66.5,
read 32.6 s; Flash-Next Q2_0 56.8 / 73.3 / 45.4, read 36.8 s; Flash-Next IQ2_XS 52.0 /
75.1 / 42.6, read 39.2 s; Qwen3.8-35B-A3B-Distill Q4_K_M (poly, draft model = Qwen3.6
Q4_K_XL) 97.9 / 151.6 / 74.3, read 32.3 s.

### 7.5 Unit tests and quality gates

**Kernel tests** (bit-for-bit against the previous kernels, plus their times):

| Test | What it proves |
| --- | --- |
| `poly_gdn_cols_test` | DeltaNet 4-column warp kernel: same bits for out, state and snapshots; gdn_step x1.38-1.44 |
| `poly_gemm_small_test` | Register-tiled small GEMM: same bits; x1.99-2.26 on the router shape |
| `q5k_head_test` | basalt's Q5_K head GEMV + fused top-1 + 2-bit proxy, bitwise native_mmvq |
| `iq4xs_head_test` | IQ4_XS head GEMV (full head 2560 x 248,320): greedy tokens identical off/on; 200/200 exact argmax on the draft pick |
| `head_fmt_test` | Per-format traits kernels (IQ4_XS, Q6_K, Q4_K): speedups above, 200/200 picks |
| `bfla_test` | Dense vs FP64 rel L2 1.3-1.5e-4; sparse-all-blocks bitwise equal to dense |
| `expert_seed_test` | Items 3-4 seeding (header-only expert_seed.hpp) |
| `grammar_test`, `grammar_schema_test`, `grammar_guard_test`, `serve/test_grammar.py`, `serve/test_grammar_routes.py` | Grammar engine (GBNF / JSON schema / tool calls); 3 strict json_schema requests + a tool call with an enum argument all valid; 0.3-0.8 ms of grammar code per request, 35-110 us per 50-130 ms window (0.1-0.2%), 0-1 masked redraws per request; routes fixed on all three API paths (bcbdeb5d2) |
| `serve/test_loop_guard.py` | Project Maya loop guard port: periodic-block detection, penalise/collapse/stop, tool-repeat blocking; Maya's module tests + pancake server tests |
| `serve/test_tolerant_tool_calls.py` | Malformed tool calls never raise; unclosed `<parameter=` tag handled |
| `serve/test_request_accounting.py` | reasoning_tokens in usage/logs (2d5d44709) |
| `serve/test_domain_classify.py` | Server domain classification for STRATA_DOMAIN_PROFILES |
| `test_predict_experts.py` | Offline predictor scoring (transition 29% at 65% waste; lookahead ~2x better) |
| `tools/pancake/build_matrix.sh` (item 92, DONE 635c2a752) | Rows hip-mmq, hip-nommq, cuda-86, hip-tests, configure-only: all pass |

**Token-forcing quality gate** (`tools/pancake/quality_gate.py` / `quality_ab.sh`; 20
requests over code, docs, English, multilingual and agent text from `bench/quality/held.jsonl`,
300 greedy tokens each, scored teacher-forced on each arm's own decode path; two stock arms
give the run-to-run noise; mismatch = the arm's argmax differs from the forced token):

| Arm | Mismatch | vs stock (90% CI) | Large-margin events | Verdict |
| --- | ---: | ---: | ---: | --- |
| stock (exact) a / b | 1.17% / 1.30% | - | 0 / 0 | - |
| fast preset (Speedy's switches) | 1.20% | -0.03 pts (-0.25, +0.16) | 0 | PASS (became the default) |
| expert tail skip 7 | 2.07% | +0.84 pts (+0.44, +1.24) | 0 | INCONCLUSIVE (stays opt-in) |
| cache-aware routing lambda 2.0 | 9.90% | +8.67 pts (5.71, 11.95) | 8 | REJECT |
| cache-aware routing lambda 0.5 | 2.89% | +1.66 pts (1.00, 2.41) | 0 | opt-in only (turbo) |
| turbo-hip on 7900 XT | +0.4 pts | - | 0 | INCONCLUSIVE (gate pending) |
| later gate run (quality.log, 10:35) | 1.71% vs stock 1.691/1.935% | -0.103 pts, CI90 [-0.297, 0.08], p=0.54 | - | PASS (which lossy arm was gated is not recorded) |

**Quality-gate revision** (from the Toronto paper): token forcing rejects large deviations
only; choice between close variants by a fixed task set (LiveCodeBench subset + BFCL v4
multi-turn + own-repo checkpoint grading), repeated runs as distributions.

**Decode curve** (item 2 closed, `tools/pancake/decode_curve.py`,
`bench/results/2026-11-09-decode-curve`): the same edit reply after a 13.8K prompt decodes
as fast as after a 2K one, segment by segment; the drop after long prompts is the long
case's prose reply (acceptance 0.62 vs 0.87, slow start while the cache adapts). No refill
fix needed.

**Sparsity study** (item 13 closed, `bench/results/2026-11-09-sparsity/`): STRATA_ACT_DUMP_RAW
on the Q2_0 file, layers 4/12/24/36/44, 11,027 expert-token records (short + 2 code edits),
`tools/pancake/sparsity_masks.py`: at 70% sparsity the gate mask keeps cos 0.844 (~54%
relative error), CATS 0.818, even the oracle |h| mask 0.954; at 50% gate 0.939, oracle
0.989. Flash-Next's experts (FF 640, fine-grained) are dense: the screen (gate cos >= 0.99
at >= 70%) fails, no sparse kernel.

**Self-tuning dry run** (item 70, `tools/pancake/selftune.py`, commit 05e041b9b): traces ->
candidate profile -> time model -> abtest spec; dry run: candidate profile +8.6%, + kmiss
+23% modeled.

**Seeding offline replay** (items 3-4, `bench/prefetch/routing.bin`, 10,909 slots, 32 swaps
every window): plain 81.17%; domain leave-one-out (code) 82.86% (code edit 77.3 -> 78.8,
long 84.7 -> 90.7) but with the prose request seeded in-sample first the held-out edit gets
77.1; oracle domain (own counts) 84.03%. Conversation resumed after another request:
+0.7..+1.0 points on its rest (bound +1.4..+2.7). At 160 swaps the domain seed gains 0.5.
One trace of four requests: not enough to trust the domain part; the conversation part
never went negative.

**Agent-side statistics** (opencode history, old backend, 2,063 Flash-Next turns served on
11437): turn median 6.2 s, p90 50 s, p99 224 s; prompt re-reads ~18% of total turn time (a
lower bound); 15 turns cut at the output cap while thinking (84-340 s each). The tail comes
from re-reads and uncapped thinking, not tok/s. This is why phase 14 (per-turn timeline)
ranks as high as any engine item.

---

## Where this document ends and the others begin

- [README.md](README.md): what pancake-strata is, how to run it, the presets and switches.
- [../MOD_PLAN.md](../MOD_PLAN.md): the live plan, item numbers 1-95b, phases 0-14.
- [../docs/PANCAKE_FINDINGS.md](../docs/PANCAKE_FINDINGS.md): the chronological measurement log.
- [../PANCAKE_STATUS.md](../PANCAKE_STATUS.md): the current Dev/Test board.
- [../reports/](../reports/) and [../research_notes/](../research_notes/): the full research corpus.
