# Speed-focused Strata forks: what can raise decode / prompt speed on pancake (HIP, RX 7900 XT + Ryzen 7 5700G)

State as of 2026-11-10. Sources: GitHub API (`gh api repos/...`: README, commits, file contents) read on 2026-11-10. Every
tok/s number below is **author-measured on the author's hardware** unless it says "pancake measured". Comparison base:
local clone `/home/dan/clone-of-d/src/Strata` (branch `port/rdna3-decode`, HEAD `f15ad2b51`), its `MOD_PLAN.md`, README
section 6, presets `configs/pancake/flashnext-*-fast-hip.json`, and the 2026-11-09 survey `reports/Projects derived from Strata.md`. Status tags: **NEW** (not in our code or ledger), **ALREADY-HAVE** (code present, maybe off), **REJECTED-BEFORE**
(measured/closed in MOD_PLAN), **N/A** (needs hardware we do not have).

Repo fact sheet (all MIT; dates from `pushed_at` / last commit via GitHub API):

| Repo | Last commit | Stars | Focus | Speed content |
| --- | --- | --- | --- | --- |
| [sangharshadhyeta/Strata-HotRod](https://github.com/sangharshadhyeta/Strata-HotRod) | 2026-11-09 | 13 | RTX 3060 12 GB, 30 GB RAM, experts streamed from NVMe | **high** (lossy stand-ins + recheck, learned look-ahead, I/O scheduling) |
| [DevDock-AI/Speedy](https://github.com/DevDock-AI/Speedy) | 2026-11-09 (0.4.0) | 57 | 1-2x RTX 3080 10 GB presets + A/B ledger | medium (mostly 2-GPU / 1M-context; good rejected list) |
| [RubenTorney/strata3090](https://github.com/RubenTorney/strata3090) | 2026-10-13 | 3 | RTX 3090 tuning, ~50 settings A/B'd | medium (Zen 3 AVX2 IQ decode, graph-node cost, config) |
| [sergqwer/strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) | 2026-11-09 | 230 | NVFP4 packs, GPTQ quants, CPU NVFP4 kernels | low for us (Blackwell/FP4; tail-skip quality data) |
| [DeepSeek2333/strata-nvfp4](https://github.com/DeepSeek2333/strata-nvfp4) | 2026-11-10 | 1 | re-merge of sergqwer's fork onto upstream 0.2.13 | none distinct |
| [star1872/TokenRun](https://github.com/star1872/TokenRun) | 2026-11-07 | 7 | packaging + old-PC settings (Haswell/DDR3) | low (settings, DFlash2 acceptance) |
| [ilfarpro/DeepStrata](https://github.com/ilfarpro/DeepStrata) | 2026-11-09 | 0 | DeepSeek-V4.1 UD-IQ2_XXS rebrand, repo hygiene | none (4.7-10 tok/s DS V4.1) |
| [SuperWJ062/Strata-NEXUS](https://github.com/SuperWJ062/Strata-NEXUS) | 2026-10-29 (v0.74) | 7 | Chinese frontend/portable packaging, DS V4.1 | none (11 tok/s DS V4.1 on 4090D) |

## Q1. Per-repo changes, numbers, and port path to HIP/gfx1100

### Takeaway
Only three repos carry engine-side speed work relevant to us: HotRod (lossy confidence-gated expert substitution with a
fragile-token re-check, plus a learned layer look-ahead and NVMe I/O scheduling), strata3090 (a gather-free Zen 3 AVX2
decode for IQ2_S/IQ3_S CPU expert rows, a graph-node cost measurement, and config findings), and Speedy (config ledger;
its new code is 2-GPU / 1M-context). The NVFP4 forks, DeepStrata, NEXUS and TokenRun are packaging or Blackwell-specific.
Most HotRod gains are for an SSD-bound box (30 GB RAM); pancake keeps all experts in 57 GB RAM, so only its lossy routing
ideas and the re-check guard transfer.

### Cited Findings

**Strata-HotRod (RTX 3060 12 GB, i5-12400F, 30 GB RAM, PCIe 3 NVMe, Qwen3.8-Flash-Next IQ3_S 78 GB, upstream 0.1.45 base)**
- Headline (author): 50-70 tok/s decode at 131K context (code 52-55, reasoning 66-71, prose 50-53); 8,554-token prompt read
  in 16.9 s; 45,398-token prompt 67-68 s; resuming a 45K conversation 0.4-0.5 s — [README](https://github.com/sangharshadhyeta/Strata-HotRod)
- **Stand-in experts, confidence rule** (default since 24 Oct): a missing pick is replaced by the best resident expert only
  when `router weight x logit gap <= budget`; heavy picks (rank 0/1) need a stand-in within ~1 logit, light ranks 3-9 may
  go further — [HOTROD-DETAILS](https://github.com/sangharshadhyeta/Strata-HotRod/blob/main/docs/HOTROD-DETAILS.md)
- **Fragile-token re-check** (`STRATA_FRAGILE_RECHECK`, opt-in since 9 Nov): per token the engine sums the stand-ins' moves
  (weight x gap over all layers); when the token's lead over the runner-up is below k x move, the verify window is cut
  before it and the next window decides it with top ranks exact. k is learned (settled 0.03-0.05); ~1 in 5 re-checks had
  flipped a token; only 1-3% of windows were cut. Cost: 42-44 tok/s off vs 31-34 on (262K) — [HOTROD-DETAILS](https://github.com/sangharshadhyeta/Strata-HotRod/blob/main/docs/HOTROD-DETAILS.md)
- **Self-tuning budget with a flip guard** (`STRATA_ROUTE_BUDGET_AUTO`): windows alternate in runs of 8 between budget b and
  b x step; the faster by 2% over 96 windows wins unless its flip rate's 90% lower bound exceeds the base's 90% upper bound.
  "Speed alone once took the budget to 0.95, and the text fell apart". Per-request brake (`STRATA_FRAGILE_BRAKE`): start at
  half budget, halve when >=15 re-checks show >20% flips — [HOTROD-DETAILS](https://github.com/sangharshadhyeta/Strata-HotRod/blob/main/docs/HOTROD-DETAILS.md)
- Stand-in quality/speed table (24 Oct, author): rank rule 31.8/39.8/30.9 tok/s (code/reasoning/prose) clean; fixed budget
  0.25 no re-check 45-51/55-57/40-41 but broken words in 2 of 3 prose answers; self-tuning+re-check+guard+brake 37.3/48.0/29.1,
  27/27 clean — [HOTROD-DETAILS](https://github.com/sangharshadhyeta/Strata-HotRod/blob/main/docs/HOTROD-DETAILS.md)
- **Prompt stand-ins** (`STRATA_PREFILL_STANDIN`, default on, lossy): 76-118-token prompts read in 1.3-1.8 s instead of
  2.3-2.8 s (35-45% faster); 8.5K prompt 15.1 vs 15.8 s — [HOTROD-DETAILS](https://github.com/sangharshadhyeta/Strata-HotRod/blob/main/docs/HOTROD-DETAILS.md)
- **Learned look-ahead** (`STRATA_LA_LEARN`): router score + small live-learned correction, per-layer contest, falls back to
  plain router; recall 1/2/4 layers ahead 85-88/76-79/66-71% vs plain router 77-79/66-69/55-58%; decode about +3% and
  steadier (58.6 vs 54.4-59.0 tok/s) — [README](https://github.com/sangharshadhyeta/Strata-HotRod), [HOTROD-DETAILS](https://github.com/sangharshadhyeta/Strata-HotRod/blob/main/docs/HOTROD-DETAILS.md)
- **Read-ahead I/O scheduling**: 2 I/O threads, next layer first (`STRATA_IO_PF_THREADS`, `STRATA_IO_EDF`) because only
  ~3,000 of 25,000 read-ahead experts were used and 8 threads' reads queued ahead of demand reads: +8% code, +15% reasoning,
  +8% prose; drive wait 3.8-4.7 s -> 2.8-3.0 s — [HOTROD-DETAILS](https://github.com/sangharshadhyeta/Strata-HotRod/blob/main/docs/HOTROD-DETAILS.md)
- `STRATA_PIN_SEGS` (pageable RAM tier page-locked in 1 GiB segments) with `STRATA_PCIE_BALANCE`: 39.7 vs 36.5 tok/s (+9%) —
  [HOTROD-DETAILS](https://github.com/sangharshadhyeta/Strata-HotRod/blob/main/docs/HOTROD-DETAILS.md)
- Opt-in, not measured as wins: `STRATA_MTP_SLOTS`/`STRATA_MTP_ZC` (draft head's experts in slots, rest computed from RAM),
  `STRATA_RING_COST` (look-ahead ring weighted by each layer's CPU cost) — [README](https://github.com/sangharshadhyeta/Strata-HotRod)
- Measured and rejected by the author: `STRATA_ROUTE_TAIL_OPEN=1` (malformed JSON/code), `STRATA_ROUTE_BRAKE_STEP=1` (wrong
  maths answer), `STRATA_IO_GATE=1` (no gain), `STRATA_GDN_LOOP=1` (looped middle GDN layers: -10% decode, 180 vs 175/210
  MMLU-Pro, not significant), Monitor timeline always-on (cost 5-13%) — [HOTROD-DETAILS](https://github.com/sangharshadhyeta/Strata-HotRod/blob/main/docs/HOTROD-DETAILS.md), [README](https://github.com/sangharshadhyeta/Strata-HotRod)
- Author's own decode window breakdown (~55-60 ms/window): GPU-reach wait ~25 ms, CPU experts 10-15 ms, synchronous NVMe
  reads 11-15 ms — [perf-ideas.md](https://github.com/sangharshadhyeta/Strata-HotRod/blob/main/docs/perf-ideas.md)
- Prompt-side wins on that box: `--prefill auto:16384` (8,554-token prompt 38.5 -> 23.4 s, -37%); chunk 32768 no effect;
  layer-major prefill + borrowed layer map + O_DIRECT prompt reads + prompt look-ahead off became LIVE —
  [perf-ideas.md](https://github.com/sangharshadhyeta/Strata-HotRod/blob/main/docs/perf-ideas.md)
- The README states no AMD testing; the learned mask should be turned off "on AMD or Apple" (`STRATA_MASK=0`) — [README](https://github.com/sangharshadhyeta/Strata-HotRod)

**DevDock-AI/Speedy (2x/1x RTX 3080 10 GB, TR PRO 5975WX, 128 GB DDR4-3200 8-ch; Strata v0.2.7.1 base + picks)**
- 1x 3080 (author): Q2_0 152 ±13 tok/s decode, IQ3_S 130, UD-IQ4_XS 107, UD-Q4_K_XL 93; prefill 1,134-2,824 tok/s at 60K.
  Noise floor decode ±4-8% — [README](https://github.com/DevDock-AI/Speedy)
- 0.4.0 (2026-11-09): `STRATA_PW_RES_PATCH` fixes cut-off answers under `--pipeline-windows 2` (route prior read a stale
  residency table): UD-Q4_K_XL decode +25%/+30%; `STRATA_GDN_TC=1` (split tensor-core GDN) prefill +1-4%; route-prior NaN
  guard (= upstream #5655); upstream #5643 turn-2 TTFT 1.44 -> 0.23-0.28 s; `--vram-reserve-mib 444` instead of 700 (700 cost
  single-card prefill 10-17%) — [CHANGELOG](https://github.com/DevDock-AI/Speedy/blob/main/CHANGELOG.md)
- Rejected after measurement (0.4.0): CPU share of short prompt chunks, arena release, HANDOFF_PIPE, GDN_SPLIT, QFUSE,
  EXPDEC_R=2, CPU prefetch distance, q4_0 K/V for 1M, route prior weaker/off (-20/-38% decode), 16K chunks on one card,
  upstream #5595 — [CHANGELOG](https://github.com/DevDock-AI/Speedy/blob/main/CHANGELOG.md)
- Earlier: `STRATA_HC_Q8=1` in every preset: Q2_0 decode 143.2 -> 151.1 (+5.5%), IQ3_S 112.5 -> 119.3 (+6.0%);
  `STRATA_KV_STAGE_REQ` 300K cold prompt 466 -> 1527 t/s at 1M context; `STRATA_SELECT_TWO_PASS` +0.2-2.5% prefill;
  rejected `STRATA_MMQ_DIRECT`, `EXP_MODE=8`, StrataFast prompt split, #2964, #4567, #3919 —
  [CHANGELOG](https://github.com/DevDock-AI/Speedy/blob/main/CHANGELOG.md)
- 2-GPU only: layer split with weight trim (prefill +76-185%), `--pipeline-windows 2` + `STRATA_PIPELINE_PENALTIES=1`
  decode +8-28%, `STRATA_SPLIT_MTP_BATCH=1` +2-9%; `"parallel": 2` costs a single user 15-19% decode —
  [CHANGELOG](https://github.com/DevDock-AI/Speedy/blob/main/CHANGELOG.md), [README](https://github.com/DevDock-AI/Speedy)
- Speedy merged "AMD RX 7900 XTX (RDNA3/HIP) support from Strata-rdna3" on 2026-11-06 (commit ff146b50) — [commits](https://github.com/DevDock-AI/Speedy/commits/main)

**RubenTorney/strata3090 (RTX 3090 24 GB, Qwen3.8-Flash-Next IQ3_S, base v0.1.43)**
- Kept config: 117.0 -> 122.2-123.7 tok/s short context (+4.5% over base); 93.0/90.2/91.5 at a 257,886-token prompt; 213K
  prompt ~2,250 tok/s — [docs/RTX3090.md](https://github.com/RubenTorney/strata3090/blob/main/docs/RTX3090.md)
- `--adapt-async 1` +14% (102.9 -> 117.3); `--kv-grow` +6% (94.5 -> 100.0); `--adapt-swaps 160` vs 80 +1.6%;
  `--adapt-decay 0.97` (0.95: 120.2, 0.99: 113.8); `--pcie-frac 0.35`; `--spec 4 --spec-min-p 0.70` (0.80: -2.5%) —
  [docs/RTX3090.md](https://github.com/RubenTorney/strata3090/blob/main/docs/RTX3090.md)
- **`STRATA_IQ_ZEN3=1`**: AVX2 decode without gather for IQ2_S and IQ3_S gate/up rows (register-held grid indices, gate and
  up in one pass at 1-2 tokens), +0.8% (119.5 -> 120.5); single-token rounding changes (not bit-identical) —
  [docs/RTX3090.md](https://github.com/RubenTorney/strata3090/blob/main/docs/RTX3090.md), [iq_avx2.cpp](https://github.com/RubenTorney/strata3090/blob/main/src/kernels/cpu/iq_avx2.cpp)
- `STRATA_PCIE_BRANCH=1 STRATA_FETCH_BLOCKS=64` (PCIe share on a side stream beside the VRAM hits; 64-block fetch kernel):
  +0.9-1.1% — [docs/RTX3090.md](https://github.com/RubenTorney/strata3090/blob/main/docs/RTX3090.md), [verify.cpp](https://github.com/RubenTorney/strata3090/blob/main/src/core/verify.cpp)
- **Graph node cost**: a 4-token window graph has 2,158 nodes; padding with `STRATA_DUMMY_NODES` gave ~0.86 us/node = 1.9 ms
  per window (7%), the upper bound of what further kernel fusion could return on that card — [docs/RTX3090.md](https://github.com/RubenTorney/strata3090/blob/main/docs/RTX3090.md)
- CPU profile during decode: 63% of CPU samples in the six compute threads' wait loop, 23% in expert kernels; a missing
  expert costs ~80 us on six threads (4.3 GB/s per thread); finer splitting or merging phases estimated 0.2-0.4% at most —
  [docs/RTX3090.md](https://github.com/RubenTorney/strata3090/blob/main/docs/RTX3090.md)
- Tried and not kept (author): `STRATA_GR_SM86`, `GR_ACT_AHEAD`, `--spec 5 --spec-cum-p` (112.6 vs 120.8), personal expert
  profile (no effect), `STRATA_FS_SLOTS=2` (111.1 vs 123.7), `Q2_BITPLANE` (noise), `ADAPT_ASYNC_DEMAND` (119.0 vs 122.5),
  `--mtp-q4` (worse), `LFUSE/GDN_SPLIT/TSUM/VERIFY_QDEDUP/Q8_PACKED` (no effect), `FUSE_HEAD_GR`/`Q6_PACKED` (worse),
  `ROUTE_RESIDENT=0.5` (120.6 vs 123.1), `QFUSE` (95.4 vs 115.1), locked clocks/performance governor (-2.4%) —
  [docs/RTX3090.md](https://github.com/RubenTorney/strata3090/blob/main/docs/RTX3090.md)

**sergqwer/strata-nvfp4 and DeepSeek2333/strata-nvfp4 (RTX 5090, 9950X3D, 128 GB DDR5)**
- Accuracy vs all-Q8_0 (same top token) and decode: NVFP4 GPTQ 93.6% / 177 tok/s, UD-Q4_K_XL 94.1% / 169, IQ3_S 90.0% / 228,
  IQ3_XXS 87.3% / 239 — [sergqwer README](https://github.com/sergqwer/strata-nvfp4); DeepSeek2333 is the same table on a
  0.2.13 re-merge — [DeepSeek2333 README](https://github.com/DeepSeek2333/strata-nvfp4)
- CPU side: NVFP4 rows AVX-512 1.8-3.7x ggml-cpu, AVX2 2.0-2.4x at 4-8 tokens; AVX-VNNI copies of i-quant and Q2_0 kernels
  where present; GPTQ NVFP4 has 33.3% of plain-rounding error — [sergqwer README](https://github.com/sergqwer/strata-nvfp4)
- Route tail skip (`STRATA_ROUTE_TAIL_SKIP=7`): decode +11.5% (193.7 vs 174.0), teacher-forced KL 0.035 (top token same 93.7%),
  "thinking ends sooner (p(`</think>`) early x1.35)" — [sergqwer README](https://github.com/sergqwer/strata-nvfp4)

**star1872/TokenRun (Haswell Xeon + DDR3 + PCIe 3.0)**
- Settings back to 0.1.44.1 rules (`STRATA_PCIE_K_RULE=0`, `STRATA_PCIE_SHARE_MAX=0.60`, `STRATA_NV_CHUNK=0`,
  `STRATA_MMQ_KQ=0`), tail skip off, `STRATA_ADAPT_OVERLAP=1` on Linux: +6.5% over its earlier engine; DFlash2 drafter
  acceptance 0.82 at conf 0.8 but 20.81 tok/s, slower than MTP — [TokenRun README](https://github.com/star1872/TokenRun)

**ilfarpro/DeepStrata and SuperWJ062/Strata-NEXUS**
- DeepStrata: DeepSeek-V4.1 2.00 bpw, RTX 5070/64 GB ~4.7 tok/s, RTX 3060/46 GB 8-10 tok/s; recent commits are docs,
  update-feed key, desktop app removal — [DeepStrata README](https://github.com/ilfarpro/DeepStrata), [commits](https://github.com/ilfarpro/DeepStrata/commits/main)
- NEXUS: DeepSeek-V4.1 2.00bpw on RTX 4090D + 128 GB ~11 tok/s (60% of experts on CPU, 6% in VRAM); commits are versioned
  frontend/packaging releases — [NEXUS README](https://github.com/SuperWJ062/Strata-NEXUS)

**Local status checks (pancake code, 2026-11-10)**
- In our tree already: `STRATA_ROUTE_RESIDENT` (verify.cpp, native_router.cu), `STRATA_PCIE_BALANCE`, `STRATA_IO_PF_THREADS`,
  `STRATA_ROUTE_TAIL_SKIP`, `STRATA_ADAPT_OVERLAP`, `--kv-grow`, `STRATA_DUMMY_NODES`, `STRATA_GR_FAST`, `STRATA_IQ_MT_MIN`,
  `STRATA_Q2_BITPLANE`, `STRATA_HC_Q8`, `STRATA_GDN_TC`, `STRATA_SELECT_TWO_PASS`, `--pipeline-windows` (grep of src/include).
- Not in our tree: `STRATA_FRAGILE_RECHECK`, `ROUTE_BUDGET_AUTO`, `FRAGILE_BRAKE`, `PREFILL_STANDIN`, `LA_LEARN`, `PIN_SEGS`,
  `MTP_SLOTS/ZC`, `RING_COST` (HotRod); `STRATA_IQ_ZEN3`, `PCIE_BRANCH`, `FETCH_BLOCKS` (strata3090); `PW_RES_PATCH`,
  `KV_STAGE_REQ`, `IDX_ELASTIC`, `KV_HOST_ELASTIC`, `ROPE_HOST`, `PIPELINE_PENALTIES` (Speedy); the #5655 route-prior NaN guard
  (no match in src/serve).
- MOD_PLAN Round A (pancake measured, 7900 XT): base 111.7/92.5 tok/s; lossy `turbo` 149.3, tail skip 7 129.7; upstream
  HC_Q8 107.2 (worse); `--pcie-frac 0` 118.4/104.3 is the preset; `fast-hip` preset has no `--kv-grow` and no `--adapt-decay`.

### Port table (pancake: RX 7900 XT gfx1100 HIP, 5700G Zen 3 AVX2, 57 GB DDR4, single user)

| # | Item (source) | Tag | Kind | Port path to HIP | Effort | Suggested A/B (flag, workloads) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Fragile-token re-check + flip-guarded budget search + per-request brake (HotRod) as a **quality guard for our lossy routing** (`turbo-hip` ROUTE_PRIOR, tail skip, ROUTE_RESIDENT) | NEW | lossy-guard, algorithmic | host-side bookkeeping in verify window rule (sum weight x gap per token from router rows we already publish; cut window via existing window-rule hook); no CUDA intrinsics needed; the learned mask part is CUDA-only per author, skip it | M-L | `STRATA_FRAGILE_RECHECK=1` on `turbo-hip` vs `turbo-hip` vs `fast-hip`; short chat, 2K code edit, 11.7K prompt; token forcing + flip rate + tok/s. Goal: keep most of turbo's +25-34% with token-forcing PASS |
| 2 | Confidence rule `weight x logit gap <= budget` instead of fixed lambda/rank rule (HotRod) | NEW (variant of ALREADY-HAVE ROUTE_RESIDENT/ROUTE_PRIOR) | lossy routing | device side: one extra compare in the top-10 route kernel (`route_row.cuh`/`native_router.cu`) using logits already computed | S-M | `STRATA_ROUTE_BUDGET=<b>` sweep 0.1/0.25 vs ROUTE_PRIOR=0.5; same 3 workloads; mismatch % + large-margin events |
| 3 | Prompt stand-ins (HotRod `PREFILL_STANDIN`) | NEW | lossy prefill | prefill routed path: substitute non-resident experts when the rule allows, cutting PCIe uploads of expert weights during chunked prefill | M | `STRATA_PREFILL_STANDIN=1`; 2K code edit and 11.7K prompt TTFT; KL on following decode. Gain likely small here (prompt is GPU/PCIe-stream bound, not NVMe) |
| 4 | Learned look-ahead correction (HotRod `LA_LEARN`) | NEW, but predictor-recall item is REJECTED-BEFORE ("coverage 19-21%; promotions do not pay") | exact | small MLP-free correction on router scores; our lookahead already runs on GPU | M | only if 10.2/10.3 are reopened; offline on a `STRATA_WINDOW_TRACE` first. Low priority: our misses are CPU-compute from RAM, not NVMe |
| 5 | I/O read-ahead scheduling (demand reads never queue behind guesses, 2 threads) (HotRod) | ALREADY-HAVE (`STRATA_IO_PF_THREADS` upstream) / N/A (no NVMe expert streaming on pancake) | exact | none | - | none |
| 6 | `STRATA_IQ_ZEN3` gather-free AVX2 IQ2_S/IQ3_S decode (strata3090) | NEW | near-exact (single-token rounding changes) | pure AVX2 C++ (`iq_avx2.cpp`), no CPU check; drop-in for our CPU pool; **only helps if our CPU-side expert tensors are IQ2_S/IQ3_S** (check `gguf` tensor types of the daily IQ2_XS pack; IQ2_XS rows would need a new variant). Zen 3 has slow `vpgatherdd`, so the idea fits the 5700G | S (if types match), M (write IQ2_XS variant) | `STRATA_IQ_ZEN3=1`; 2K code edit (CPU-miss heavy) and short chat; parity test + tok/s. Related: our `STRATA_IQ2S_PAIR` was neutral |
| 7 | Graph node cost measurement with `STRATA_DUMMY_NODES` (strata3090) | ALREADY-HAVE (switch present), NEW as a measurement | measurement | run on HIP: hipGraph per-node launch cost on ROCm is a known unknown; gives the ceiling for further fusions (EXPDEC_SWIGLU, WAIT_FUSE etc.) | S | `STRATA_DUMMY_NODES=10/40`; short chat; slope = us/node x nodes/window |
| 8 | `--kv-grow` (strata3090: +6% via more VRAM for experts) | ALREADY-HAVE (flag), not in `fast-hip` preset | exact | config | S | add `--kv-grow` to fast-hip; short chat, 2K edit, 11.7K prompt; watch expert slots in log |
| 9 | `--adapt-decay 0.97` (strata3090 best; 0.99 -8%) | ALREADY-HAVE (flag), not set in preset | exact | config | S | `--adapt-decay 0.95/0.97/0.99` with our `--adapt-every 1 --adapt-swaps 160`; 2K edit |
| 10 | `STRATA_PCIE_BRANCH` + `FETCH_BLOCKS=64` (strata3090, +1%) | NEW | exact | host/stream ordering in verify.cpp; HIP streams fine | S | moot while preset uses `--pcie-frac 0`; only test if a PCIe share returns |
| 11 | Route tail skip 7 (upstream; sergqwer data KL 0.035, +11.5%, thinking shorter) | ALREADY-HAVE (pancake measured 129.7 vs 121, lossy, not in preset) | lossy | none | - | could be the second lossy arm under item 1's re-check |
| 12 | `STRATA_HC_Q8` (Speedy +5.5-6% on sm_86) | REJECTED-BEFORE (pancake 107.2 vs 111.7; we use `GR_RDNA_Q8`) | lossy | - | - | - |
| 13 | Route-prior NaN guard (#5655, Speedy 0.4.0) | NEW (correctness for turbo-hip; earlier survey flagged it) | exact | 3-line guard in route_prior kernel | S | unit test with a non-finite router row; no speed effect |
| 14 | `STRATA_PIN_SEGS` (HotRod +9% with PCIE_BALANCE) | NEW | exact | pinned host segments via hipHostRegister; we already use `PIN_BY_NEED` | S-M | relevant only with a PCIe share > 0; skip while `--pcie-frac 0` |
| 15 | `--vram-reserve-mib` lower (Speedy 700 -> 444 gave back 10-17% prefill on 10 GB) | ALREADY-HAVE flag; check preset | exact | config | S | reserve 444 vs current; 11.7K prompt TTFT, watch for OOM |
| 16 | Speedy 2-GPU set: layer split + trim, `--pipeline-windows 2` + `PW_RES_PATCH`, `PIPELINE_PENALTIES`, `SPLIT_MTP_BATCH`, StrataFast prompt split | N/A (phase 12, second GPU) | exact | park with phase 12 | - | - |
| 17 | `KV_STAGE_REQ`, `IDX_ELASTIC`, `KV_HOST_ELASTIC`, `ROPE_HOST` (Speedy 1M tier) | NEW / N/A | exact | only matters at 1M context with streamed K/V | M | not for our 3 workloads |
| 18 | NVFP4 W4A4 packs, FP4 tensor cores, GPTQ NVFP4, AVX-512/AVX-VNNI CPU kernels (sergqwer, DeepSeek2333) | N/A (REJECTED-BEFORE: MOD_PLAN "NVFP4 packs ... need Blackwell") | quant | gfx1100 has no FP4 and the 5700G no AVX-512/AVX-VNNI. Transferable idea only: multi-token CPU expert kernels (2.0-2.4x at 4-8 tokens) — we already have `STRATA_IQ_MT_MIN` multi-token path | - | - |
| 19 | `STRATA_GDN_TC` split tensor-core GDN (Speedy +1-4% prefill) | ALREADY-HAVE (code), CUDA mma | prefill kernel | would need a WMMA (gfx11 `v_wmma`) rewrite; our WMMA prompt attempts (`PF_WMMA`) measured worse | L | not recommended |
| 20 | `STRATA_GDN_LOOP` looped layers (HotRod) | NEW but rejected by author (-10%) | lossy | - | - | do not port |

### Inferences
- The biggest transferable lever is not a kernel: it is HotRod's **re-check + guarded auto-budget** wrapped around a lossy
  routing shortcut. Pancake already owns two lossy shortcuts with large measured gains (turbo-hip +25-34%, tail skip ~+7%)
  that stay opt-in because token forcing was INCONCLUSIVE; a guard that re-decides only the few tokens a substitution could
  flip is exactly what would let one of them become the default. Expected cost from HotRod's data is large on an
  SSD-bound box (-25%); on pancake a cut window costs one extra verify pass of a ~46-130 ms window, so the overhead scales
  with the cut rate (1-3% of windows there) and could be well under 5% — this is an estimate, not measured.
- strata3090's finding that ~50 knobs only gave +4.5% and that graph nodes cost 7% of a window on a 3090 suggests pancake's
  remaining exact headroom on one GPU is small; measuring node cost on ROCm (item 7) tells whether fusion work is worth
  more there.
- `STRATA_IQ_ZEN3` is the only exact CPU-kernel idea aimed at our exact CPU family (Zen 3, slow gather), but it pays only if
  CPU-computed expert rows use IQ2_S/IQ3_S encodings.

### Gaps
- Exact expert tensor types in our daily `packs/iq2_xs` (IQ2_XS vs IQ2_S mix) were not checked here; item 6's value depends on it.
- HotRod's re-check overhead was only measured on an NVMe-bound 3060; no AMD/HIP measurements exist in any of these forks
  (HotRod and strata3090 say AMD untested; Speedy only merged Strata-rdna3's code).
- strata3090 last committed 2026-10-13 on base v0.1.43; its switches may conflict with upstream 0.2.12 code paths.
- Whether DeepSeek2333's "NVFP4 W4A4 on sm_75/86" uses an emulated int8 path (a possibly transferable fused-dequant idea) is
  not described in the README; its code was not read.

## Q2. Which changes are CUDA-only vs portable algorithmic?

### Takeaway
Everything with real value for pancake is portable host/algorithmic code or plain AVX2; the CUDA-only items (NVFP4/FP4
tensor cores, split tensor-core GDN, INT8-IMMA/MMA prompt kernels, gpu-unlock FP16 rate, sm_120 peer tier) are either
Blackwell/Ampere hardware features or already rejected on gfx1100.

### Cited Findings
- NVFP4 W4A4 experts and FP4-Local/QAD models are the sergqwer/DeepSeek2333 core, measured on RTX 5090 — [sergqwer README](https://github.com/sergqwer/strata-nvfp4), [DeepSeek2333 README](https://github.com/DeepSeek2333/strata-nvfp4)
- Speedy's `STRATA_GDN_TC` is a "split tensor-core GDN"; gpu-unlock lifts LHR FP16/BF16 tensor rate 62 -> 106 TFLOPS (+10-12%
  prefill, decode unchanged); RTX 50 (sm_120) peer tier — [CHANGELOG](https://github.com/DevDock-AI/Speedy/blob/main/CHANGELOG.md), [README](https://github.com/DevDock-AI/Speedy)
- HotRod's learned mask is CUDA-only per its README ("turn them off ... on AMD or Apple") — [README](https://github.com/sangharshadhyeta/Strata-HotRod)
- strata3090's `STRATA_IQ_ZEN3` is "Plain AVX2 code, so no CPU check" — [iq_avx2.cpp](https://github.com/RubenTorney/strata3090/blob/main/src/kernels/cpu/iq_avx2.cpp)
- sergqwer's AVX-512 (F/BW/VL/VNNI/VBMI) and AVX-VNNI kernels; AVX2 fallback is "bit for bit ggml-cpu's own AVX2 dot" — [sergqwer README](https://github.com/sergqwer/strata-nvfp4)
- None of the forks inspected uses cp.async or TMA explicitly in its README/changelog claims (not found in docs read).

### Inferences
- Portable: stand-in rule, re-check, budget search, brake, prompt stand-ins, look-ahead correction, I/O scheduling, PCIe side
  branch ordering, config findings (kv-grow, adapt-decay, reserve), graph-node methodology, NaN guard.
- CUDA/hardware-bound: NVFP4/FP4, GDN_TC, IMMA prompt attention, HC_Q8 tuning for sm_86 (rejected on our card anyway),
  AVX-512/AVX-VNNI CPU paths (5700G lacks both).

### Gaps
- I did not grep the forks' .cu files for cp.async/TMA usage; the statement above rests on documentation only.

## Q3. Lossy speed tricks and their quality numbers

### Takeaway
Three lossy families appear: expert substitution by resident stand-ins (HotRod, with the best quality instrumentation),
route tail skip (upstream; sergqwer quantified KL 0.035 and shorter thinking), and looped layers (rejected). Top-2-rank
substitution breaks output; ranks 3-9 under a weight-x-gap budget hold quality on small hand-read sets.

### Cited Findings
- HotRod top-two-rank substitution: any distance or within 3/5/8 logits -> collapse (empty replies, loops); within 2 logits ->
  dropped tokens, broken words, wrong sums; rank 1 within 1 logit -> 49/50 maths (same as exact); ranks 0+1 within 1 logit ->
  47.4 vs 38.6 tok/s but leaked a turn marker at 94K; now "the top two are now always real" — [HOTROD-DETAILS](https://github.com/sangharshadhyeta/Strata-HotRod/blob/main/docs/HOTROD-DETAILS.md)
- HotRod quality with stand-ins on: 47/50 maths and 30/30 coding vs 48/50 and 30/30 exact; MMLU-Pro 87.1% (183/210, stand-ins on);
  re-check on vs off: 6 long answers all clean in both — [README](https://github.com/sangharshadhyeta/Strata-HotRod)
- HotRod ranks 3-9 budget from the first token: 44.7 vs 46.4 tok/s (-4%), 12/12 JSON/code valid; `ROUTE_TAIL_OPEN` (ranks
  3-9 unchecked) produced malformed JSON/code — [HOTROD-DETAILS](https://github.com/sangharshadhyeta/Strata-HotRod/blob/main/docs/HOTROD-DETAILS.md)
- Route tail skip 7: +11.5% decode, KL 0.035, top token same 93.7%, p(`</think>`) early x1.35 — [sergqwer README](https://github.com/sergqwer/strata-nvfp4)
- strata3090 `ROUTE_RESIDENT=0.5`: 120.6 vs 123.1 tok/s (slower on a 24 GB card with 91-94% hit rate) — [docs/RTX3090.md](https://github.com/RubenTorney/strata3090/blob/main/docs/RTX3090.md)
- Speedy: weakening/removing the route prior fixes cut-offs but costs -20/-38% decode; root cause was a stale residency table
  (`PW_RES_PATCH`) — [CHANGELOG](https://github.com/DevDock-AI/Speedy/blob/main/CHANGELOG.md)
- No fork implements top-k reduction (fewer than 10 experts) or early exit; HotRod's `GDN_LOOP` is the opposite (extra
  compute), -10% with no significant gain — [HOTROD-DETAILS](https://github.com/sangharshadhyeta/Strata-HotRod/blob/main/docs/HOTROD-DETAILS.md)

### Inferences
- HotRod's evidence matches pancake's own (`turbo` lambda 2.0 gave 9.90% mismatch, lambda 0.5 2.89%): substitution must
  never touch ranks 0-1 and needs a weight-aware bound. A pancake A/B should log flip rate and large-margin events, not just
  tok/s, and include the "shorter thinking" side effect sergqwer saw for tail skip (it can inflate tok/s-per-answer metrics).
- Lossy gains shrink as the expert hit rate rises (strata3090 at 91-94% hits lost 2% with ROUTE_RESIDENT); pancake holds ~44%
  of pairs in VRAM, so it sits in the regime where substitution pays (as turbo's +25-34% shows).

### Gaps
- No fork reports KL or token-forcing for HotRod's stand-ins; its quality evidence is small hand-read sets and one MMLU-Pro run.
- No quality numbers exist for prompt stand-ins beyond "answers checked read clean".

## Q4. Draft / speculative improvements with acceptance numbers

### Takeaway
No fork brings a new drafter that beats Strata's MTP head on single-GPU decode; the useful data points are tuning results
(spec 4, min-p 0.70 best on a 3090; cumulative-p rules worse; 4-bit MTP head worse) and DFlash2 having high acceptance but
lower throughput, consistent with pancake's own rejection.

### Cited Findings
- strata3090: `--spec 4 --spec-min-p 0.70` kept (0.60 -0.5%, 0.80 -2.5%); `--spec 5 --spec-cum-p 0.45` 112.6 vs 120.8; cum-p 0.60/0.50
  114.7/114.8; `--mtp-q4 all/proj` 120.3/117.9 vs 122.2; draft acceptance 93% on repetitive HTML vs 70-80% on ordinary text —
  [docs/RTX3090.md](https://github.com/RubenTorney/strata3090/blob/main/docs/RTX3090.md)
- TokenRun: DFlash2 drafter acceptance 0.82 at conf 0.8 yet 20.81 tok/s, slower than MTP (conf 0.5: 20.23) — [TokenRun README](https://github.com/star1872/TokenRun)
- HotRod: probabilistic acceptance of drafts on by default (`--no-spec-prob-accept` turns it off); `STRATA_MTP_SLOTS/ZC` opt-in,
  no gain reported — [HOTROD-DETAILS](https://github.com/sangharshadhyeta/Strata-HotRod/blob/main/docs/HOTROD-DETAILS.md), [README](https://github.com/sangharshadhyeta/Strata-HotRod)
- HotRod's re-check interacts with drafting: it cuts the verify window before a fragile token (1-3% of windows) —
  [HOTROD-DETAILS](https://github.com/sangharshadhyeta/Strata-HotRod/blob/main/docs/HOTROD-DETAILS.md)
- Speedy: `STRATA_SPLIT_MTP_BATCH=1` +2-9% decode, 2-GPU only — [CHANGELOG](https://github.com/DevDock-AI/Speedy/blob/main/CHANGELOG.md)

### Inferences
- Status vs pancake: spec-min-p tuning ALREADY-HAVE (`--spec-min-p` in preset; `--spec 5/6` flat in Round A); speculative
  sampling ALREADY-HAVE (`STRATA_SPEC_PROB`); DFlash2 REJECTED-BEFORE (115.9/84.3 in Round A); 4-bit MTP head REJECTED-BEFORE
  (tandem requant note). Nothing NEW in drafting from these forks.
- The 70-80% vs 93% acceptance gap by content type supports pancake's existing cost-based/DISCO window rules rather than a
  new drafter.

### Gaps
- No fork published per-position acceptance curves or tokens-per-window for its MTP setup on Qwen3.8-Flash-Next beyond the
  strata3090 percentages above.
