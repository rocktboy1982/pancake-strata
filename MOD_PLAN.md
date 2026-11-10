# pancake-strata: plan

A fork of [Strata](https://github.com/Niko1221/Strata) tuned for speed on two home cards: an **RTX 3060 12 GB** (CUDA,
sm_86) and an **RX 7900 XT 20 GB** (HIP, gfx1100), with a Ryzen 7 5700G (AVX2, 16 threads) and about 57 GB of RAM.
Features are ported by hand from projects that forked Strata earlier and measured their gains.

Status (2026-11-07): driver up; CUDA sm_86 build of strata, strata-dsv41 and strata-poly. FastStrata's strata-poly engine is ported (branch port/faststrata-poly). First numbers on the RTX 3060 12 GB, Qwen3.6-35B UD-Q4_K_XL, greedy, cold run (bench/results/2026-11-07-pancake-baseline.jsonl, tools/pancake/bench_engines.py):

| Case | strata (16K ctx) | strata (64K ctx) | strata-poly, FastStrata flags (131K ctx) |
| --- | ---: | ---: | ---: |
| short chat, 256 new | 64.6 tok/s | 53.2 tok/s | 100.7 tok/s |
| code edit, 5.3K prompt, 512 new | 40.1 tok/s | 39.4 tok/s | 151.1 tok/s |
| 41.8K prompt: read / 128 new | - | 42.1 s / 9.7 tok/s | 35.8 s / 71.7 tok/s |

strata keeps fp32 KV (45 KB a token): setup's 131072 context runs out of VRAM on 12 GB, and at 64K no expert fits in VRAM. strata-poly: --threads 8 --kv-q8 --attn-split --graph-experts --elastic-prefill 1024 --lend-staging --stage-next --suffix-drafts 256 --cpu-multi (no lossy --tail-skip). The two engines' greedy texts differ after 25-81 tokens (different kernels, Q8 KV), as expected for two engines; the exactness rule applies to base vs mod of one engine.

Since then (commit 1c182934a): DeltaNet 4 columns a warp and a tiled small GEMM, the same bits, prompt reads 9-12%
faster. Every test, profile and rejected idea is in [docs/PANCAKE_FINDINGS.md](docs/PANCAKE_FINDINGS.md); phase 1's
profile is there too (prompt reading is GPU-bound; attention is 38% of a 42K read; CPU experts are ~half of decode).

## Goal

The fastest single-user agent serving (long, growing prompts and tool calls) for three models:

| Model | Engine in Strata | Packs to tune |
| --- | --- | --- |
| Qwen3.8-Flash-Next (125B MoE, ~6B active, 51B n-gram table, 4B MTP) | main engine (`src/core`, `src/program`) | the fastest: Q2_0, IQ2_XS |
| Qwen3.6-35B-A3B | `src/qwen35` | Unsloth UD-Q4_K_XL / Q4_K_M with the MTP layer |
| DeepSeek V4.1-Flash-1017 (latest, 2026-10-17) | `src/dsv41` | ISTA-DAS 2.00 / 2.25 bpw |

Speed counts as: decode tok/s (new text and code edits), prefill tok/s at 4K/16K/32K, and the time to the first token
of an agent turn after a long conversation. Output must stay identical to plain decoding unless a flag says it is lossy.

## Non-goals

- Qwen3.8-27B (dense). Strata's expert offload does nothing for it; llama.cpp or TabbyAPI on the 7900 XT serves it.
- NVFP4 packs and basalt's `.basalt` format: they need Blackwell (sm_120).
- Vulkan, SYCL, Intel and Mac paths: left as upstream has them, not tuned.
- Other engines as the base. Checked and rejected for this hardware: vLLM, SGLang, KTransformers (needs a 24 GB+
  card), TensorRT-LLM, LMDeploy, MLC-LLM, ExLlamaV2 (archived), PowerInfer, TGI (archived), LightLLM, NanoFlow,
  DeepSpeed-MII, TensorFold. They keep the model in VRAM or need far more VRAM or RAM.

## Sources

| Source | Commit | License | What it gives |
| --- | --- | --- | --- |
| [Strata](https://github.com/Niko1221/Strata) (base) | `146df3c4` (v0.2.12) | MIT | the engines, CUDA and HIP builds |
| [FastStrata](https://github.com/Jarson1408/FastStrata) | `df91582` | MIT | Qwen3.6-35B options measured on an RTX 3060 Laptop, `docs/OPTIMIZATION_LOG.md` |
| [PolyStrata](https://github.com/simplexai-labs/PolyStrata) | `a82999fd` | MIT | FastStrata's parent, the `src/poly` engine |
| [basalt](https://github.com/jesdga95/basalt) | `a9468e5c` | MIT | two-stage head, grammar, MTP work, idle sleep |
| [SGLang](https://github.com/sgl-project/sglang) kernels | main | Apache-2.0 | ideas: tree drafts, speculative sampling, GPU token mask |
| TensorRT-LLM tech blogs | - | - | ideas: grammar inside speculative decoding, auto n-gram enablement |

basalt rewrote its history, so it shares no git ancestry with Strata: it forked at Strata `822a5c783` (2026-09-27,
engine 0.1.8). Both diverged by thousands of commits, so nothing is merged; each feature is ported and credited.

## Ground rules

1. **Measure first, every step.** A change stays only if it is faster on the bench below and passes the exactness check.
2. **Exact by default.** Greedy output equals the base engine token for token. Lossy options are opt-in flags with a
   start-up warning (as basalt and FastStrata do).
3. **Both cards.** CUDA and HIP from the same source. A change measured only on one card says so; a CUDA-only change
   must not break the HIP build.
4. **Small commits** with the measured numbers in the message, crediting the source project and commit.

## Bench

- Harness: `~/bench` (existing Python harness and jsonl results), extended for Strata's OpenAI API.
- Cases: short chat (new text), code edit (copy-heavy), 16K and 32K prompts, an agent session replay (growing prompt,
  tool calls), a repeated task in a new conversation (cross-conversation drafts).
- Each case: greedy, 3 runs, interleaved base and mod, median. Record VRAM peak and RAM use.
- Exactness: greedy text of the base and mod engines compared for 900+ tokens per case.

## Status (2026-11-08, branches exp/expert-prefetch -> feat/grammar)

| Item | State | Evidence |
| --- | --- | --- |
| 10.1 decode breakdown | done | CPU experts 38 ms of a 66 ms window; GPU waits 17.5 ms |
| 10.2 router lookahead | done, on by default on the GPU | same picks, CPU cost 30 -> 0.6 ms/window, +10% decode on mmap'd experts |
| 10.2 prefetch consumer (Foresight) | measured, off | 0.97 at 4 slots; 8 slots run out of VRAM on long prompts |
| 10.3 draft-informed prefetch | covered by 10.2 | the lookahead routes the window's draft tokens |
| 10.4 finish together | already in Strata | PcieAuto, share 0.20-0.25 on all 48 layers |
| 10.5 VRAM lending, item 7 per-layer budgets | dropped | +5% cache slots: no decode change; cross-layer swaps 48.3% vs 48.4% |
| 8.6 learned expert profile | done, in the IQ2_XS presets | +6.7% decode, every run faster |
| 8.7 two-stage prompts | blocked | needs a second GPU |
| 8.8 fused attention prompt kernel | low ceiling | attention is 6.7% of a 42K prompt on the main engine |
| 8.9 cache-aware routing | opt-in (`turbo`) | 0.5: +25% decode, 2.9% tokens differ (stock 1.2-1.3%) |
| 8.9 KV body, mb-MTP | dropped | VRAM does not buy decode here; drafting is 2-5% of a window |
| 8.10, phase 5 | blocked | the 7900 XT is not detected |
| phase 3 drafting | in the engine / low ceiling | auto n-gram (DraftPolicy), spec sampling exist; draft step 2-5% |
| phase 6 grammar in spec decoding | done, on in the IQ2_XS presets | tool calls / strict schemas always valid, 0.1-0.2% per window |
| phase 7 cache tiers | mostly in the engine | open: sleep-in-place with the RAM complement (basalt 64167ba73), automatic disk sessions |
| phase 9 Qwen 4 | waiting | no weights published |
| prompt path kernels | measured, FP16 kept | WMMA worse; fused int8 -7% at 42K but +12% at 5K |
| vs stock Strata (upstream 146df3c46) | measured | Flash-Next IQ2_XS, experts read from disk on both: decode 54.5 -> 69.2 tok/s (+27%), 5.3K edit prompt 23.3 -> 11.0 s; ours with the RAM arena 75.5 |

**Next session (left 2026-11-08):**
1. Decide the load default: building the 34 GB expert arena now often trips systemd-oomd (`user@.service`
   memory-pressure limit 50% for 30 s; only the capped scope dies). `STRATA_RAM_KEEP_GIB=20` (mmap tier) loads every
   time but decodes ~8% slower (69 vs 75 tok/s). Either make 20 the default in `capped.sh`, or keep the arena and pin
   it in smaller steps so the stall time stays under the limit.
2. Phase 7 sleep-in-place (basalt 64167ba73, ~815 lines): only if the GPU is wanted for other work while idle.
3. Merge `port/faststrata-poly -> exp/expert-prefetch -> feat/grammar` when wanted (all pushed, none merged).
Stock comparison build: `../Strata-stock` (upstream 146df3c46, `build-cuda/strata`); results in FINDINGS 10f.

## Development plan by expected gain (2026-11-09)

All open items of phases 11-13 plus the new combined items 63-72 (next section), in one order: big user-visible gains first.
Base (fast-hip, 7900 XT): short window 24.8 ms (GPU 15.9, CPU experts 5.2, draft 2.2), edit window 37.4 ms (GPU 15.8, CPU
experts 17.4, draft 2.5); opencode turns (old backend): median 6.2 s, p90 50 s, re-reads ~18% of turn time, thinking ~6.6%.
Line format: `item - name - expected gain - exact/lossy - owner - needs - status`.  "est." = estimate, "meas." = measured
here.  Owners: dan-cd = 2, 8/28, 14/27/30, 29 (13 closed); this = this session (server/agent, feat/disco, feat/bfla-hip,
feat/bitdecoding-hip); unowned = nobody yet (suggestion in brackets).  Tiers: A >= 10% decode or a large per-turn/TTFT win, or
a blocker for one; B 3-10%; C 1-3%; D < 1%, hygiene, docs, hardware-gated.

Substitute groups (gains inside a group do NOT add; the ceiling is the group's, not the sum):
- S1 CPU-expert time (17.4 ms edit / 5.2 ms short; ideal-overlap floor ~20 ms edit window): 7, 8/28, 23, 5, 14/27/30 (+53,
  3-4, 17), 29, 46-52, 40-41.  Ceiling: lossy +40-60% on edits (turbo + one other), exact-only +15-30% on edits (model),
  short chat <= 1.27x.  Turbo and deferral must be measured together.
- S2 verify expert set (one re-selection in verify.cpp): 7 vs 23 (vs OEA/BuddyMoE) - pick ONE; 23 sits 5-15% above fast-hip.
- S3 window length/source: 15, 36, 43, --spec-min-p, --lookup-max-t - 0-10% combined.
- S4 thinking tokens: 12, 21, 34, effort default - -20-40% thinking on routine turns (est.) plus the p99 tail; the cap stays.
- S5 re-reads: 11, 16, 31, 39, 71 (+3 on resume) - ~15% of turn time (~8% in-session + ~7% sub-agent switches).
- S6 tool-idle window (one GPU slot): 17 vs AsyncFC vs Slipstream compaction.
- S7 GPU floor: 1, 32/48, 37, 47, Q2_0-down, 10, BitDecoding - pays on short chat now; on edits only after S1 brings the CPU
  part below ~15.8 ms (until then a faster GPU kernel only lengthens the combine_cpu spin).

**Tier A - prerequisites first (no gain of their own; each unblocks a Tier A lever)**
- 73 (wiring, I9+I3) - pancake.service on 11437 (Conflicts= strata/avalon, ntfs3 mount wait), strata-coder context
  131072, reasoning_tokens in chat usage, per-turn JSONL log on D - 0 itself; no engine gain reaches the user without it -
  exact - this - none - not started (0 opencode sessions on pancake).
- 35 restore-equivalence test (full + partial hit, every boundary type, MTP on) - 0; gates 3, 11, 16, 31, 39 - exact - this -
  none - open.  First PR of item 66.
- 67 quality/perf harness (I4 + I5, capability-tagged) - 0; gates 7, 8/28, 23, 21, 25, effort default - this (runner) +
  dan-cd (A/B queue) - none - open.
- 49 + 50 + 44 measurement batch (one GPU session: combine_cpu timer, spin vs row pull, per-layer slack) - 0; calibrates 42, 49 CODED (7bcb77d5e: combine_cpu stamps waitCPU in-kernel under WAIT_FUSE); dispatch jobs split (expert source sub-timer, c44b1ad7b); 50/44 are runs (test phase).
  decides 51/52, feeds 1/32/48 - exact - unowned [this] - none - open.
- 64 shared window trace (new) - 0; one GPU trace run answers 14/30, 28's weight CDF, 36, 15, 23, 5, 3-4, 17 offline - CODED 2026-11-09 (14a4b3373): STRATA_WINDOW_TRACE logger + tools/pancake/trace/wtrace.py reader/converter; DISCO/SpecFit readers still to do (this).
  dan-cd (logger, = I1) + this (readers) - none - open.
- 42 overlap-aware max-plus simulator - DONE as time_ceiling.py --overlap (751e09b70; serial/defer/guard, alpha DMA contention;
   item 29 +3% modeled); validation to 5-9% error in the test phase - exact -
  dan-cd (item 14 owner; spec here) - 49/50/44, 64 - spec written.
- 8 non-finite bisect (6 runs on WAIT_FUSE, ROUTE_PUBLISH, GR_FUSED, QDEDUP, ADAPT_OVERLAP, EXPDEC) - 0; gates 8/28 - dan-cd -
  none - queued.

**Tier A - levers**
- 7 turbo-hip default for some modes - +30-50% decode meas. (111.7 -> 149-168 tok/s) - lossy (2.9% tokens differ; upstream
   agent score 5.6 -> 4.5) - owner decision - 67 - measured, gate pending.  [S1, S2]
- 66 agent-turn pipeline (new: 35 -> 31/11/16/39 + 12 + 3/17) - 15-25% of agent wall time est. - exact except the caps -
   this - 73, 35 - design.  [S4, S5, S6 as one feature]
- 12 thinking caps in the preset (tail above p99 only, grace 512, reasoning_close_retry) - worst turn 84-340 s -> ~1-2 min,
   ~6.6% of turn time - changes text on the tail only - this - 73 (to read p99) - code done, in no preset.  [S4]
- 8 + 28 expert deferral with the rank guard - +10-25% on edits est. (ceiling 1.45x, KTransformers) - lossy-small - dan-cd -
   42 (done, model), 67 - UNBLOCKED 2026-11-09: non-finite cause = STRATA_ROUTE_PUBLISH, now off under deferral (d5b58f669);
   time_ceiling.py --overlap (751e09b70) models deferral +33% on edits, + k-th-miss 2/8 +45-52% (alpha 0.25-0.5; model, not
   validated).  [S1; with 7 the combined gain is below the sum]
- 38 Linux direct expert-arena loader (O_DIRECT via direct_file.cpp; FADV_DONTNEED stop-gap) - start 28-65 s -> ~10-14 s est.,
   likely removes the oomd trips that force the -8% mmap tier - exact - unowned [this] - none - open.  Part of 69.
- 2 keep the decode cache through long prompts - CLOSED 2026-11-09 (c9d9f014c): decode_curve.py shows no refill cost; the drop
   after long prompts is the prose reply and cache warm-up.
- 14/27/30 time-based cache simulator, then the winning policy - 0-30% on edits; best policy band +5-16% est.; build only at
   >= 5% - exact - dan-cd - 42, 64 - offline.  [S1]
- 31 preserve_thinking end to end - removes the re-read from the first dropped <think> block every turn (share of the 8%
   in-session re-reads) - exact - this - one logged request, 35 - open.  [S5]
- 11 message-boundary checkpoint + --prompt-cache-tail - first edit -33% (author, other hardware) - exact - this - 35 -
   opt-ins exist, arms queued.  [S5]
- 16 tool-boundary checkpoints + Marconi FLOP-saved-per-byte eviction - share of S5 (Marconi: TTFT up to -71% on SWE-bench
    traces) - exact - this (per dan-b5 notes) - 35, 73 log - open.  [S5]
- 69 memory budget arbiter (new: 38 + 39 + 2 + RAM_KEEP + parking budget) - start time above, arena kept (+8% vs mmap
    fallback, meas. 75 vs 69 on the 3060), more parked conversations - exact - this + dan-cd (2) - 63 - design.
- 39 automatic disk tier for parked conversations - resume ~7 s instead of ~83 s at 70K (FastStrata), survives restarts -
    exact - this - 35 - open (D is 91% full: capped tier).  [S5]
- 71 shared-prefix store for sub-agents (new) - up to ~7% of turn time (~103 switches, ~5.5M tokens) - exact - this - 35,
    73 - open.  [S5]
- 21 learned thinking stop - -20-40% thinking on routine turns est. (untested on agents) - lossy (text) - this - I2 hook,
    73 log, 67 - open.  [S4, after 12]

**Tier B (3-10%)**
- 68 miss-cost-aware speculation (new: 15 + 36 + 43 + 23 + 28) - exact part 2-6%, with AcceptMoE 5-15% on edits est. -
    mixed - this (+ dan-cd for 28) - 64, 63 - design.  [S2, S3]
- 23 AcceptMoE depth-graded verify prior - 5-15% over fast-hip est. - lossy - dan-cd - 64 replay, 67 - CODED 2026-11-09 as
   STRATA_ROUTE_PRIOR_DEPTH=a (4f56ce535), opt-in; quality gate pending.
    [S2: dropped if 7 ships]
- 15 + 36 DISCO + SpecFit/entropy features - 0-10% combined - mostly exact - this (feat/disco) - 64 - branch.  [S3]
- 65 overlap package judged by the simulator (new: 46, 47, 48, 51, 40, 29) - 5-10% on edits, 3-8% short est. - exact (51
    not bit-identical) - split, see item - 49/50/44, 42 - design.  [S1 + S7]
- 1 GPU path: rocprofv3 on edit + after 11.7K, group A one at a time, HC reads (~2.7 ms), VRAM-hit phase (3.3 ms) - 3-8% on
    short chat est., ~0 on edits until S1 - exact - unowned - I7 profile - open.  [S7]
- 72 GPU-floor package triggered by the critical path (new: 1, 32/48, 37, 10, Q2_0-down, 47) - 5-10% short chat est. -
    exact - unowned - 44, 64 - design.  [S7]
- 48 (= 32 re-rated) fuse the pre-doorbell chains (HC norm/down/up, router+plan+bell) - 1-4% per chain, critical in both
    window types - exact - dan-cd - 49 - open.  [S7] CODED: expdec gate/up + SwiGLU + q8_1 in one launch (2aca56777, STRATA_EXPDEC_SWIGLU=1); router GEMV + top-10 + doorbell in one launch (7c0f03228, STRATA_ROUTER_FUSE=1); the HC norm/down/up chain was already one kernel (fused_gr_read_multi).
- 63 capability probe + calibration layer (new: 55, 61, 57, 56, 58, 41, 53, 43, 54) - 0-3% here, 6-13% on other platforms
    (static PcieAuto lost that much here) - exact - this (probe) + dan-cd (tier parameters) - none - design.
- small exact flag stack + --lookup-max-t 12 (P2c) - +2-5% (meas. +1.4..+3.9%, +1% retest) - exact - dan-cd queue - 67/I5 -
    queued.  [S3 for lookup]

**Tier C (1-3%)**
- 29 per-expert barriers / STRATA_POOL_FUSE_PHASES - 1-4% - exact - dan-cd - I5 A/B - coded, opt-in.  [S1]
- 3-4 conversation working set + domain profiles - +0.6-1.5 hit points (~1-2% at request start est.) - exact - this (code)
    + dan-cd queue (traces) - 64, 35 for resume - code merged.  [S1]
- 47 warm the Infinity Cache with next-layer pre-doorbell weights during the wait - 0-2 ms/window est. - exact - dan-cd -
    micro-test (build only at warm/cold < 0.85) - open.  [S7] CODED (269a06ed9, STRATA_IC_WARM_MB=<cap>).
- 53 online swap budget + chunked, stale-dropping promotions - 0-5% over the best fixed --adapt-swaps - exact - dan-cd
    (coordinate with dan-cd, touches the tier) - 42 - open.  [S1] CODED (6bc9530bc, STRATA_ADAPT_SWAPS_AUTO=1; ForesightSwap stale-drop skipped: FS off in the presets).
- 46 single poller in combine_cpu - 0-3% - exact - dan-cd - 49 - open.  [S1] CODED (93f7a963d, STRATA_COMBINE_POLL1=1).
- 40 software prefetch in the Q2_0 row kernel - 0-5% of CPU-expert time - exact - dan-cd - none - open.  [S1] CODED (c44b1ad7b, STRATA_Q2_PREFETCH=<bytes>).
- 43 DraftPolicy::seed_costs() from traces - 0-5% - exact - dan-cd - 42, 64 - open.  [S3] CODED (663107a56): serve loop reads STRATA_SPEC_COST_TABLE; tools/pancake/trace/draft_costs.py builds it from window traces.
- 10 MTP draft inside the captured verify graph - 1-3% est. if the 2.2 ms is launch-bound - exact - dan-cd - draft timing
    (2b run) - waiting.  [S7]
    CODE STUDY 2026-11-09 (dan-cd): the draft is already graph-chained (one round graph + step graphs queued at once,
    mtp.cpp chain_launch), so its 2.2-2.5 ms is GPU work, not launches; commit/emit is 0.07 ms.  The real lever found:
    STRATA_MTP_HIP_PRIO (draft stream on its own hardware queue, so the round no longer waits behind the commit graph;
    R9700 draft wall 5.8 -> 3.5 ms) defaults on only for gfx12 - off on the 7900 XT -> arm in bench/specs/07.  Item 20
    (ReplaySSM): the commit replays the GDN recurrence over the kept rows (gdn_step_commit_layers, ~1% of GPU time in the
    profile); snapshots per row would be ~2 MB per GDN layer per row - replay stays; with MTP_HIP_PRIO it overlaps the
    draft.  Both closed by reading. CLOSED by code study (9b0081c43): already graph-chained; STRATA_MTP_HIP_PRIO arm instead.
- 5 tail-row drop - 0-5% est. - exact per plan - dan-cd - 64 replay with real MTP confidences - open.  [S1] Replay built (tools/pancake/trace/tail_replay.py, drafts record in the trace); needs dan-9f's drafts() call (item 74).
- 20 ReplaySSM audit of the GDN replay - 1-3% decode - exact - dan-cd - none - open. CLOSED by code study (9b0081c43): replay stays (per-row snapshots ~2 MB/layer/row).
- Q2_0-down expdec kernel (unnumbered; native_down_multi_kernel<42> = 5.8% of GPU kernel time, hidden on edits) - 1-3% short
    est. - exact - unowned - 49 split - open.  [S7] Exists as STRATA_EXPDEC_DOWN=1 (never measured on gfx1100): arm in bench/specs/04 (dan-cd).
- 51 CPU pre-sums missed rows per token - 0.5-1.5% on edits est. - not bit-identical - dan-cd - 50 - open.  [S1] CODED (cc5499fae, STRATA_CPU_PRESUM=1) + item 52 ReBAR target (4f829da0e, STRATA_PRESUM_VRAM=1).
- 37 coordinate-descent sweep of HIP launch/tile and AVX2 block sizes - 1-3% - exact - dan-cd - 1 - open.  [S7] TOOL BUILT (7861dc8c1, tools/pancake/sweep.py); running it = test phase.
- 41 smallest saturating thread count + performance governor - 0-3% each - exact - owner (governor) / 63 - none - open.
- 17 pre-stage the expected experts while a tool runs - 1-3% of the next turn's decode est. - exact - dan-cd -
    simulator - open.  [S6] CODED engine-only (43e741bc6, STRATA_IDLE_ADAPT=N: tier catch-up between requests).
- 18 GDN prefill chunk 64 - 5-11% of prefill, cold prompts only (90.8% of turns hit the cache) - exact - unowned - I8 prefill
    timing - parked.
- 70 closed-loop self-tuning from own sessions (new) - 1-5% est. (keeps profile-type gains current; 8.6 gave +6.7%) - exact -
    this + dan-cd - 63, 64, 67 - design. CODED offline pass (05e041b9b, tools/pancake/selftune.py): traces -> candidate profile -> time model -> abtest spec.
- 33 behaviour counters (log-only first) - unmeasured (failed runs cost 1.4-4.4x tokens) - exact (log) - this - 73 - open.
- 34 rumination signal into the soft close and 21 - inside S4 - lossy (text) - this - 73 - open.  [S4] CODED log + opt-in close (feat/item-34-rumination); threshold unmeasured.
- 25 ISTA-DASLab Coder variant A/B - quality candidate (95.7% of BF16 on SWE-bench Verified), speed unknown - n/a - owner -
    ~29.6 GB on D, 67 - open.
- 19 [2026-11-09: BFLA kernel on the qwen35 engine done on feat/bfla-hip - 32K layer 1238.7 -> 581.1 ms exact (dense) -> 500.2 ms
   sparse; 64K 5224 -> 2321 -> 1090 ms (4.8x); 128K 28287 -> 9553 -> 3335 ms (8.5x; 10 layers 283 -> 96 -> 33 s). 2026-11-09 end-to-end on Qwen3.6 UD-Q4_K_XL (noisy machine): 128K prompt 386.9 off -> 1219.8 sparse (3.15x) -> 771.2 dense (2.0x) tok/s; 64K sparse 1284.1, dense 1058.2 (the 64K off arm died silently once - rerun); recall probe top-1 agreement 48/48, mean KL 0.0014 (64K) / 0.0045 (128K). Flash-Next tile-shared QSA overlap measured (STRATA_SEL_OVERLAP=all, 131K): 16-query tiles read 6.3-32% of the KV cells by position, shared set holds 42-74%; prefill 229.1 tok/s; generation refused (--spec needs the residency table) - wiring pending.]
- 19 BFLA as one shared QSA selection per query tile - long-context TTFT only (paper 1.34-1.44x at 128-256K; our QSA is
    already sparse) - lossy opt-in - this (feat/bfla-hip) - STRATA_SEL_OVERLAP, I8 - CLOSED 2026-11-09: a shared set holds only
    42-74% of each query's QSA cells (falling with length) - too lossy for a <=7-8% prefill ceiling; not built.

**Tier D (< 1%, hygiene, docs, hardware-gated)**
- 52 CPU pushes rows + flag into VRAM via ReBAR - 1-2% est. but overlaps 51 - parked unless 50 shows > 3 us/layer.
- feat/bitdecoding-hip as default - attention ~3% of decode - this - opt-in until its ~2-3% (of attention) is confirmed.
- 45 I/O hygiene (STRATA_IO_THREADS sweep, filefrag, PLE order only after PLE wait is measured) - unowned.
- 54 IOMMU passthrough guardrail, 57 cheap part (platform log line, STRATA_FORCE_ISA avx512/avx-vnni), 55/61 on this box (0
    here, they matter elsewhere) - all folded into 63.
- 22 code read: FreeToken split = --pcie-frac? - close unless copied experts are also promoted - anyone, no GPU.
- Hygiene: MMQ=OFF stub 2-line fix (want_dm_f16/dm_f16_used), test_keepawake skip off Windows, README 176.9B and daily file
    (Q2_0 vs iq2_xs, owner), strata-iq2_xs.json CUDA paths, presence_penalty = 0 check, oh-my-openagent version pin.
- Hardware-gated, 0 on this box: 56 hybrid P/E affinity, 58 CCD-aware pool, 59 AVX-512 verify windows (code read possible
    now), 60 Strix Halo preset, 62 gfx12 re-measure, rest of 57 (AVX-512 without VBMI, old Q2_0 folders on AVX2).
Closed / not adopted (ledger): 6 (Q2_0 is the daily file), 13 (experts dense, gate cos 0.844 at 70%), 24 (0 on Q2_0), 19's
XAttention form (-> BFLA), 12's Dynasor part (-> 21), feat/contextpilot (1.6%), 9's sleep-in-place (parked; its disk half is
39), 26 (parked: 1 failure in 180 edits); 22 parked pending the code read above.

## Combined improvements (capability-driven evolution, 2026-11-09)

New items that come from combining existing items.  Each says why the combination is worth more than its parts, or costs less
together.  All are opt-in; the default path stays byte-identical until a measured A/B changes a default (CONTRIBUTING).

63. Capability probe + calibration layer ("the engine sets itself up for the box it is on").  Combines 55 (CPU-vs-promote
    ratio), 61 (VRAM-class defaults), 57 cheap part (ISA log line), 56/58 (P/E and CCD topology), 41 (thread count), 53 (swap
    budget start), 43 (draft costs), 54 (IOMMU), 38's NVMe rate.  Mechanism: one start-up probe, cached per machine fingerprint
    (CPU model, DIMMs, GPU, link, driver/ROCm, model file hash) in `<data>/machine-profile.json`: ISA and core topology
    (CPUID 0x1A, CCD map), DDR read bandwidth on pool threads, CPU us/expert per quant (PcieAuto's timer), PCIe gen/width
    (sysfs) and promote us/expert (one timed copy), free VRAM -> pairs per card, NVMe direct-read rate, IOMMU mode.  A
    derivation step turns it into tier size, --pcie-frac candidates, reuse threshold, thread count/affinity, starting
    --adapt-swaps, prefetch on/off, DraftPolicy per-position cost, parking budget; then 55's window-time check of 2-3
    candidates on warm-up windows picks the winner.  Why more than the parts: each item would need its own probe and its own
    noise handling; one probe + one verification step keeps the derived settings consistent with each other (e.g. a smaller
    tier on an 8 GB card also changes the draft cost and the reuse threshold).  Gain: 0-3% here (threads, swaps), 6-13% on
    other platforms (the static PcieAuto choice lost that much here), fewer bad first runs for users.  Effort M-L (staged).
    Risk: ~20% run-to-run noise (use load gate + medians), probe time at start (cache it).  Acceptance: it finds
    --pcie-frac 0 on the 7900 XT by itself, and sane values with cache slots capped to emulate 8/12/16 GB.  Tier B.  Owner:
    this (probe, derivation, log); dan-cd consumes the tier/admission parameters.  First PR: `STRATA_PROBE=1` writes the
    platform line + machine-profile.json (no setting changed), with a unit test on fake sysfs/CPUID input.

64. Shared window trace + one reader ("one trace run, ten answers").  Combines I1, 14/27/30, 42, 43, 36, 15, 23, 5, 28, 3-4,
    17 and phase 14's self-tuning.  Mechanism: an engine-side binary ring logger (per window: positions, accepted?, top-10 +
    weights per layer, resident/CPU/PCIe per pair, per-layer timers from 49), sampled so decode overhead stays < 0.5%,
    spilled per session to D with a size cap; one Python reader (`tools/pancake/trace/`) that the simulator, DraftPolicy cost
    fitter, DISCO trainer, AcceptMoE/tail-row replays and seed learner all use.  Why more than the parts: today each tool has
    its own format (--dump-routing, conv_trace, replay_seeding, predict_experts); with one GPU in the box the scarce resource
    is GPU queue time, and one trace run replaces ~10.  Recorded from real opencode sessions it also makes every offline
    decision match the real workload.  Gain: 0 itself; unblocks Tier A items 7/14 decisions and 8/28 pricing.  Effort M.
    Risk: decode overhead, D space (91% full).  Tier A (prerequisite).  Owner: dan-cd (logger, = I1); this (reader + DISCO/
    SpecFit consumers).  First PR: schema v1 + a converter from today's --dump-routing/conv_trace output, plus the two missing
    fields (draft position, accepted).

65. Overlap package with the simulator as judge.  Combines 46 (single poller), 48 (pre-doorbell fusion), 47 (Infinity Cache
    warm), 51 (CPU pre-sum), 40 (Q2_0 prefetch), 29 (pool barriers), measured by 49/50/44 and priced by 42.  Mechanism: the
    measurement batch calibrates 42; 42 predicts each change's critical-path ms; only changes with a predicted >= 0.3 ms are
    built, in that order; then one same-binary factorial A/B, because they interact (46 frees DDR the pool uses, 47 uses the
    spin slack 46 shortens, 51 shrinks the pull 46 waits on).  Why more than the parts: separately each is 0-3%, under the
    20% noise and not testable alone; together they are one A/B batch and one model.  Gain: 5-10% on edits, 3-8% short est.
    (not additive; floor ~20 ms edit window).  Substitute: if deferral (8/28) ships, it overlaps the same CPU time and this
    package's ceiling shrinks - re-run 42 after 8.  Effort M.  Risk: WC/ordering bugs in 51/52, small effects hidden by noise.
    Tier B.  Owner: 42 dan-cd; 46/47/48/51 unowned [this: no dan-cd area]; 29 dan-cd.  First PR: item 49 (thread-0 entry/exit
    stamps in combine_cpu behind STRATA_TIMING) + a 44 slack-report script.

66. Agent-turn pipeline (one feature instead of five switches).  Combines 35, 31, 11, 16, 39, 12, 3, 17, 73's per-turn log,
    and phase 14.3 (server passes turn state down).  Mechanism: the server knows each boundary (user message, tool call
    issued, tool result, </think>) and sends one "turn plan" with the request: which checkpoints to take (message, tool),
    the conversation key (3's working set, 39's parked state), where the state lives after the turn (RAM if the turn ended in
    a tool call, demote final-answer turns first - Bidaw), the thinking cap for the effort level (12), and an expert pre-stage
    hint for the tool wait (17).  One eviction rule (Marconi FLOP-saved-per-byte) spans RAM and disk; 35 runs in CI for every
    checkpoint kind.  Why more than the parts: one conversation key and one lifecycle instead of five env switches that can
    disagree; a deeper checkpoint makes a shallower one useless on the same turn (S5), so one planner avoids paying twice.
    Gain: ~15% of turn time (re-reads) + tens of seconds per sub-agent switch + the thinking tail (~6.6%): 15-25% of agent
    wall time est.  Effort L (staged).  Risk: restore bugs (hence 35 first), RAM vs oomd (69).  Tier A.  Owner: this (server,
    31, 35, 39); 16 per dan-b5's notes; engine hooks minimal.  First PR: item 35's GPU test + one logged opencode request that
    answers 31.

67. Capability-aware quality + performance harness.  Combines I4 (revised gates), I5 (load-gated A/B), phase 14's nightly A/B,
    quality-gate process and regression watch, 63's machine profile, and the decisions it gates (7, 8/28, 23, 5, 21, 34, 25,
    effort default).  Mechanism: one nightly runner: load gate, same-binary pairs, deterministic mode for exact items and
    distribution mode for F2 items, quality chain (token forcing for large deviations -> LiveCodeBench subset + BFCL v4
    multi-turn + own-repo checkpoint grading), every result tagged with the machine profile; output is a Pareto line (tok/s vs
    task score) per preset; substitutes are run jointly (turbo + deferral, turbo vs AcceptMoE).  Why more than the parts: the
    three biggest levers (7, 8/28, 21) are all lossy and blocked on the same gate; one runner unblocks all three, and profile
    tags make community results comparable.  Gain: 0 itself; unblocks +30-50% (7), +10-25% (8/28), -20-40% thinking (21).
    Effort M-L.  Risk: task-set noise (repeat runs), GPU queue contention with dan-cd.  Tier A (prerequisite).  Owner: this
    (runner, tasks); dan-cd (queue).  First PR: runner with token forcing + a 30-task BFCL v4 multi-turn subset on
    two presets, one JSON verdict line, reusing queue.sh's load gate.

68. Miss-cost-aware speculation.  Combines 15 (DISCO), 36 (SpecFit new-miss cost, entropy), 43 (seed_costs), 23 (AcceptMoE),
    28 (rank guard), with 63's measured CPU us/expert.  Mechanism: the GPU lookahead already routes the draft tokens; per draft
    position compute the NEW CPU-missed experts against the resident map; DraftPolicy picks the window length with cost = new
    misses x CPU us/expert and benefit = acceptance (DISCO entropy features); AcceptMoE's residency bonus is applied only at
    positions >= 1 where that cost is high; 28 reads the same top-2.  Why more than the parts: one per-position cost vector feeds
    three decisions that today would each estimate it; with 63's calibration it ports to boxes where misses are cheap.  Gain:
    exact part 2-6%, with AcceptMoE 5-15% on edits est.  Substitute: if 7 becomes default, drop the AcceptMoE part (S2).
    Effort M.  Risk: lookahead routing differs from verify routing on rejected drafts.  Tier B.  Owner: this (feat/disco; 23
    unowned [this]); dan-cd for 28.  First PR: log per-position new-miss count + top-k entropy into the 64 trace and fit
    DISCO offline; no behaviour change.

69. Memory budget arbiter (RAM + VRAM + NVMe).  Combines 38 (direct loader), the RAM_KEEP vs arena decision, the parked-
    conversation budget (5.4-6.1 GiB today), 39 (disk tier), 2 (prompt lending vs decode cache), 61 (VRAM classes), oomd.
    Mechanism: at start, from 63's probe, one manager splits RAM into locked arena (filled by O_DIRECT, no page-cache double
    copy), parked conversations and oomd headroom, and VRAM into tier vs prompt lending (2's sweep as a curve); parked state
    overflows to the disk tier when NVMe is fast.  Why more than the parts: these four decisions fight over the same 57 GB /
    20 GB; deciding them apart is how the arena trips oomd and the fallback costs 8%.  Gain: start 28-65 s -> ~10-14 s est.,
    the arena reliably kept (+8% vs mmap fallback), more parked conversations (fewer switch re-reads), +20% after long prompts
    via 2.  Effort M.  Risk: ntfs3 + O_DIRECT alignment, D space.  Tier A.  Owner: this (38, 39, policy); dan-cd (2).  First PR:
    38 on Linux (direct_file.cpp, 16 threads) with the FADV_DONTNEED stop-gap behind an env switch and a load-time log line.

70. Closed-loop self-tuning from the owner's own sessions.  Combines 64 (traces from daily use), 4 (domain profiles), 8.6
    (expert profile), 43 (draft costs), 14 (policy parameters), 12 (cap values per effort), 63 (profile change detection), 67
    (promotion gate).  Mechanism: weekly offline job: re-learn profiles, costs, caps from the last week's traces and turn logs ->
    check in the simulator -> nightly A/B -> promote only on a verdict; re-run 63 when the machine fingerprint changes.  Why
    more than the parts: profile-type gains decay as the workload drifts (8.6 alone gave +6.7%); one loop keeps all of them
    current without manual runs.  Gain: 1-5% est., plus no silent decay.  Effort M (after 63/64/67).  Risk: overfitting one
    week (keep a held-out week).  Tier C.  Owner: this (job), dan-cd (engine profiles).  First PR: script that rebuilds
    data/domain-profiles from 64 traces and runs replay_seeding.py against the previous week.

71. Shared-prefix store for sub-agents.  [2026-11-09, feat/item-90-71-shared-prefix: auto-pin of the system+tools prefix for Qwen behind STRATA_AUTO_PIN=1 (exact); the Qwen template already renders tools before the system text, a schema reorder inside <tools> would change the prompt, so it is left out (gate via 67); GPU check pending]  Combines 11 (one system/tool prefix shared across sessions), 16, RadixAttention
    multi-entry store (Stanford), prompt reordering (tool schemas first, agent text after), the oh-my-openagent pin, and
    opencode's compaction prompt.  Mechanism: a server-side reorder (behind a switch) that puts the stable tool schemas first
    so all sub-agents share a long exact prefix; the engine keeps that prefix's GDN/KV checkpoint as a pinned entry next to
    the parked conversations.  Why more than the parts: 19-41K-token agent prompts with no common prefix are re-read cold on a
    one-slot server; reorder alone or a store alone gains nothing, together they turn switches into suffix reads.  Gain: up to
    ~7% of turn time (switches), part of compaction's 2-3%.  Effort M.  Risk: reordering may change model behaviour (gate via
    67), harness version drift.  Tier A (S5 share).  Owner: this.  First PR: measure the common-prefix length across the
    agents' first requests in the opencode logs; prototype the reorder in the server.

72. GPU-floor package triggered by the critical path.  Combines 1 (profile, group A), 32/48 (fusions), 37 (parameter sweep),
    10 (MTP in graph), the Q2_0-down kernel, 47, with 44 + 64 as the trigger.  Mechanism: per window type, 44's slack report
    says whether the GPU or the CPU part sets the window; GPU work starts on short-chat windows now and on edit windows only
    once S1 items bring the CPU part under ~15.8 ms; 37's coordinate descent then tunes the fused kernels together, not one by
    one.  Why more than the parts: avoids building kernels whose ms vanish into the combine_cpu spin, and one sweep covers all
    new kernels.  Gain: 5-10% short chat est., later on edits.  Effort M-L.  Risk: gfx11-only tuning (62 later).  Tier B.
    Owner: unowned [this or a new GPU owner].  First PR: the I7 rocprofv3 run on an edit window and after 11.7K, plus the
    item 10 draft timing, in one GPU session.

Added 2026-11-09 from the decode-path code study (research_notes/Pancake code deep research/decode_path.md; Serena/clangd):
74. [CODED 2026-11-09, opt-in STRATA_MTP_EAGER=1 (exact by construction: all step graphs launched at once, chain cut on the
    host at the first prob < min_p; not with a chain rule); window_trace().drafts() hook added for item 5; GPU A/B in the test
    phase] Overlap the MTP draft with the idle CPU pool (~3 ms/window idle during head, sync, commit, draft): first the exact quick win -
    gate min_p on the device so the draft round and all step graphs launch at once (no per-step host round trip, ~0.3-0.5%);
    item 10 is the larger version.  [this; generate.cpp/mtp.cpp]
75. [CODED 2026-11-09, default on, STRATA_ADAPT_PERSIST=0 restores the per-window thread; builds hip-mmq + cuda-86; A/B in
    the test phase] Persistent adapt worker: --adapt-every 1 spawns a new std::thread every window (generate.cpp ~19396); reuse JobThread
    (~2125).  Exact, 0.1-0.3%.  [this; generate.cpp outside dan-99's tier block - coordinate]
76. [CODED 2026-11-09, opt-in STRATA_DEVICE_SPIN=1 (hipSetDeviceFlags spin|maphost on the primary device before its
    context, result logged); A/B in the test phase] HIP sync policy A/B: the main device never gets hipDeviceScheduleSpin, so blocking syncs (verify.cpp:3872 head sync) use
    HIP's default; A/B spin vs default, 0.2-0.6%.  [this, measurement]
77. Hot-path hygiene: getenv("STRATA_LOGITS_DUMP") every window (verify.cpp:3953), scalar exp in SwiGLU (pool.cpp:1302,
    iq_avx2_rows.inl), activation quant serial on the main thread (expert_source.cpp:5209) - small each.  [dan-99 for files they own]
For dan-99 (their items): deferral (8/28) turns ROUTE_PUBLISH off (verify.cpp:2355) = +1 launch/layer to count in the A/B; pool:
drop the trailing wait_parked in run_phase (pool.cpp:1373), FUSE_PHASES default A/B (29); 51 needs routing weights published on the
ROUTE_PUBLISH path; 40 applies (daily Q2_0 down rows, row_multi has no prefetch; IQ prefetch measured -1.3% ms/round on Zen 3);
row_multi<1> serial FMA chain (0-3% of down, not bit-identical); MWAITX instead of pause-spin for 15 workers (0-3%); fusion
targets for 48/72: swiglu+quant as expdec_gu epilogue (~49 launches/window), router GEMV + route_publish_res, shared-expert
swiglu-quant.  Structure: generate.cpp main() is 18,360 lines with three decode loops (apply_pending/adapt duplicated at
13020/20748, 13172/20782); verify.cpp record_window ~1,480 lines with 6 env-selected post paths; ~45 STRATA_* flags in
function-local statics (no registry) - see the structure study.  Measured: ~1,360 kernels/window (~28/layer).

Added 2026-11-09 from the prefill-path code study (research_notes/Pancake code deep research/prefill_path.md):
78. Agent re-reads take the staged prefill path: the fast-hip preset's STRATA_PREFILL_CPU_SHARE=0.6 + _MAX=4096 make
    stream_all_min() 4096 (arm_cpu_share, prefill.cpp:2970-2998), so every <4096-token delta (agent turns are 1.5-3.2K) gets a
    cudaStreamSynchronize per MoE layer (4928, 48 per chunk), host-side grouping, a new CPU-share thread per layer (5181) and a
    blocking combine join (6110).  4K reads 1757 vs 2580 tok/s at 32K (2026-11-08-7900xt-final-suites.txt).  First step env-only:
    A/B CPU_SHARE=0 and SHARE_MAX=1024 on agent sessions; then a persistent CPU-share worker.  Exact.  [this; feeds 66]
79. PLE (n-gram) gather off the critical path for cold prompts: a 32K prompt is one chunk, its gather (16 direct 4 KiB reads per
    token + single-threaded decode to FP32) only overlaps embedding + layer 0 before the wait at layer 1 (3608-3664); 10.6% of
    the GPU timeline at 42K on the 3060.  Start the gather at request admission, slice it, decode on several threads, and add a
    compact upload that works with --ple-io direct (gather_raw_iq4nl needs mmap, ngram.cpp:714).  Exact.  [this]
80. hipBLASLt: cache descriptors (created/destroyed every GEMM call, gemm.cu:421-423 - host time on the staged critical path)
    and extend the gfx1100 tuning table beyond T=8192 (chunks reach 32768) and below 4096 for 1-3K deltas.  Exact.  [this]
81. GDN occupancy on gfx11: gdn_wmma_scan_kernel (kernels.cu:2006) launches 48 blocks on 84 CUs and walks T/16 chunks serially;
    combine item 18 (chunk 64) with splitting value columns over 2-4 blocks per head in one rewrite (same FP16 class, bits
    change; caution: a V100 chunk-64 attempt was 4-6x slower, bench/results/2026-10-02-gdn-chunk).
82. Prefill instrumentation run (prerequisite): STRATA_PREFILL_TIMING=1 (PfTimer, 27 phases, prefill.cpp:3092) + STRATA_TRACE=1
    were never run on the 7900 XT; add five opt-in marks (PLE wait, staged host time, CPU-share join, issuer waits, chunk-end
    callback); prompt set cold 32K/16K/4000 vs 4200 + an agent session; one rocprofv3 pass.
Smaller (prefill): per-chunk/per-layer thread creation, MTP draft K/V + checkpoints serial after a full sync at chunk end
(6363, generate.cpp:12595), fixed WMMA GEMM tiles and gfx1151-only fused-expert occupancy (37/72), expert ring 192 vs measured
512-1024 optimum on CUDA, dense weights dequantized every chunk on HIP (FP16 weight cache CUDA-only), debug hooks with blocking
syncs inside run_impl (3,224 lines; ~150 env switches); qwen35 engine syncs at every chunk start (BFLA is its lever).
Order: 78 env A/B -> 82 -> 79 -> 80 -> chunk-end fix -> 81 with 18 -> tile/ring sweep.

Added 2026-11-09 from the memory/cache code study (research_notes/Pancake code deep research/memory_cache.md):
Correction to 38: the daily preset (packs/iq2_xs has no experts.bin, STRATA_PIN_BY_NEED=1) loads through load_experts_gguf with
the padded layout (expert_source.cpp:6287), not load_experts_direct; its unbuffered branch is Windows-only (5792-5876), Linux uses
buffered fread with 6 threads (generate.cpp:7712) - measured 33.02 GiB at 0.81-1.08 GiB/s (31-41 s).  Implement 38 there: Linux
branch mirroring Windows (per-thread O_DIRECT fd with EINVAL fallback to buffered, 4 KiB-aligned bounce buffer, EINTR retry, scatter
into padded blob offsets, 16 readers, short read refuses the arena -> existing fallback), experts_unbuffered() uses the Windows
rule file_cache_keeps(avail, arena, need), STRATA_UNBUFFERED_LOAD still overrides; factor the try-direct/leave-direct helper from
conversation_file.cpp:440-452 and give DirectFile::open the same fallback (tmpfs/overlayfs); stop-gap FADV_DONTNEED after each
buffered fread; log registration time; verify with STRATA_VERIFY_ARENA equal buffered vs direct and PSI ~0 during load (~12-15 s
est.).  Likely oomd mechanism: pin-by-need registers and pins ~33 GiB (11592 ranges, 33838 MiB) before reading, then ~33 GiB page
cache is charged to the capped.sh scope (MemoryMax=54G) - confirm with one PSI trace each way.
83. [CODED 2026-11-09: tools/pancake/restore_check.py - restore vs STRATA_CKPT_REREAD, --adapt-swaps 0, exact preset, 4 turns incl.
    tool result and a branch-back; needs the GPU alone - run in the test phase] Item 35 built on existing hooks: STRATA_CKPT_REREAD (re-read instead of restore, compare token for token with --adapt-swaps 0)
    and STRATA_SNAPSHOT_VERIFY; the UNSW off-by-one is already guarded (prefix used only if <= n-1 tokens, generate.cpp:16161-16166;
    saved list never holds the last output, 19419/19611).  [this]
84. Seeds (3-4) are inert under STRATA_ADAPT_KMISS (k-th-miss admits only pairs with >= k real misses; seeding only raises usage,
    a tie-breaker): give seeded pairs virtual miss bits for the hold period.  [dan-99 owns the tier]
85. Huge pages: stop requesting THP at registration time, collapse in the background after load (compaction stalls); pipelined
    per-layer registration on Linux reusing the Windows code - measure first (Windows/5070 decoded ~10% slower).  [this, after 38]
86. [CODE 2026-11-09, branch feat/item-86-mem-budget; GPU start test pending: budget line, PSI line, no oomd.  RAM rule
    (auto on Linux with --ram-keep-gib: available - max(keep, floor) - 512 MiB margin, cap 8192; STRATA_CONV_BUDGET=ram|legacy,
    STRATA_CONV_KEEP_GIB, STRATA_CONV_BUDGET_MARGIN_MIB/CAP_MIB); PSI guard in the arena readers (STRATA_PSI_GUARD=0,
    _PAUSE_PCT 20, _RESUME_PCT 10, _MAX_PAUSE_MS 30000) and parking skipped at some avg10 >= STRATA_PSI_PARK_PCT (20).
    With keep 12 on this box the RAM rule gives ~avail-12.5 GiB (est. 2-3.5 GiB vs 5 GiB today): STRATA_CONV_KEEP_GIB=8 for more]
    69 first slice: after 38, size the conversation cache from a deterministic RAM budget (today 4096-6144 MiB = 40% of whatever is
    free, depends on leftover page cache) and pause loader readers on memory pressure (PSI).  Parked size ~236 MB fixed + 14.2 KB per
    token (fit from 540/2584-token parks; ~1.23 GB at 70K est.).  39 sketch: evict_oldest callback -> session_file_write keeping
    deepest AND root checkpoint, per-conversation index file, same validate/restore path as /slots, size cap, strata-poly key
    invalidation/lock/background writer; ~1-1.5 s resume vs ~70-80 s re-read at 70K est.; gate on 83.
Risks noted (no wrong output found): slot never completes if a ForesightSwap copy fails; async KMISS may count a late swap as a miss
once; session files keep only the deepest checkpoint (restore loses the shared system-prompt root); vague buffered-loader errors.

Added 2026-11-09 from the server/agent code study (research_notes/Pancake code deep research/server_agent.md):
87. [DONE 2026-11-09, 2d5d44709] reasoning_tokens on chat completions: _run already counts it (server.py:7310); add completion_tokens_details.reasoning_tokens
    to the usage block in openai_chunks (7767-7770) - one line.  Part of 73.  [this]
88. [DONE 2026-11-09, 637a71c23: env_to_user, effort_position end, reasoning_budget_grace 512, reasoning_close_retry on every
    Flash-Next preset; per-level caps wait for 73's log] Preset-only agent settings (fast-hip preset has engine args only): effort_position "end" (opencode effort switches re-read the
    whole system+tools prefix today; render_prompt renders 3x when on - cut to one render, 5952), env_to_user (date/folder <env>
    block breaks the shared prefix, ~14K tokens / 8-11 s per new day/folder), item 12 caps; note answer_room already caps thinking
    at 24,000 tokens on every opencode turn (max_tokens 32000, 6520).  [this]
89. [FIXED 2026-11-09, bcbdeb5d2; test serve/test_grammar_routes.py] BUG: grammar-constrained tool calls only on /v1/chat/completions - applied in _openai (9174), missing in _anthropic (9438) and
    _responses_prepare (9369).  Fix by one shared request-building step for the three API paths.  [this]
90. [DONE 2026-11-09, feat/item-90-71-shared-prefix: STRATA_PIN_PARK_TOKENS (default 2048) parks a branch that long past the pin; GPU check pending] Item 71 prerequisite: engine does not save the branch it leaves when a request resumes exactly at the pin (generate.cpp:16463);
    for a sub-agent that branch is the parent conversation -> full re-read on return; add a size threshold before enabling the
    (DeepSeek-only today) auto-pin for Qwen.  [this; generate.cpp session area - coordinate with dan-99]
91. pancake.service (73): capped.sh's systemd-run --scope escapes the unit cgroup (Conflicts=/stop/restart miss the engine) - the unit
    sets MemoryMax itself, NOCAP switch in run-pancake.sh, Conflicts= strata/avalon/qwen35b, port 11437, ntfs3 mount wait; per-turn
    JSONL log on D (no on-disk record today); strata-coder agent points at qwen3.6-35b-a3b, not Flash-Next; preserve_thinking not
    passable by clients (frontend.py:605-614) - log one opencode request to see if it resends reasoning_content (31).  [this]
Server structure: Service 2,443 lines (_run ~810 with nested closures), StrataEngine 2,066, request handler 1,414, three copied API
paths, five config sources; refactor one PR at a time: turn-log module -> shared request builder (fixes 89) -> split _run ->
split Service -> one config schema printed at start.  Python overhead est. 60-300 ms/request (1-5% of a 6.2 s median turn).

Added 2026-11-09 from the structure/build-health study (research_notes/Pancake code deep research/structure_build.md):
92. [DONE 2026-11-09, 635c2a752: tools/pancake/build_matrix.sh, all rows pass; pre-push hook not installed] Build matrix + pre-push check: tools/pancake/build_matrix.sh rows hip-mmq, hip-nommq (the stub break fixed today went unnoticed:
    no build dir has HIP+MMQ off), cuda-86, hip-tests, configure-only set; build dirs on D, install ccache.  S / none.  [this]
93. Tests build dir (STRATA_BUILD_TESTS=ON; all four build dirs have it OFF, so none of the 293 add_test entries is ever built) with a
    labelled CPU-only ctest subset; parity tests for rdna_gr.cu / rdna_mmvq.cu (none today).  S / low.  [this]
94. Typed flag registry: 872 distinct STRATA_* names read in C++ (1,046 getenv sites; 444 undocumented) + 81 in Python; defaults via
    setenv(...,0) (arch_defaults.cpp:199) silently lost by any function-local static read before apply_arch_defaults
    (generate.cpp:3818); inconsistent boolean parsing (STRATA_TSUM =='1' vs STRATA_NO_MULTI_GR !='0').  flags.def X-macro -> config
    struct filled once after arch defaults -> startup [cfg] line -> flag_lint.py; start with the ~51 preset + 48 pancake flags.
    Base for 63, 57, 67, 73 config logging.  M / medium.  [this]
95. Split generate.cpp main() (18,361 lines, 85% of the file; 68.7 s HIP compile; 127 of 143 warnings; 319 env reads): pure moves
    first - Options/usage/startup-defaults block -> startup.cpp.  M / low.  Extract-on-touch rule for Verifier::record_window (1,479)
    and Prefill::run_impl (3,224) - only lift a block when an item edits it.  [this; coordinate with dan-99 on generate.cpp]
95b. Stub blocks (prefill.cpp:116-160 MMQ-off, RDNA CUDA stubs) -> *_stub.cpp + header-parity script.  S / none.
Quick hygiene: test_keepawake fails by design of the test (KeepAwake._loop is state-driven; the mock acquires+releases before the
thread wakes, so ON is never sent; fix the assertion or skip off Windows); cache 3 hot getenvs (fused_gr.cu:1537 per launch,
STRATA_DF_PDL verify.cpp:1537/1542, STRATA_ROUTE_PRIOR 2356); remove dead functions (layer_fused_gr, layer_shared_early,
stage_mark_begin/end, gdn_buffers_zero_state, native_moe_combine_multi_hits, native_mmvq_b6_rows, df2_norm_rope, qsa_hip_wmma_active +
~12 declared-never-defined prototypes); still-read rejected flags STRATA_PF_WMMA, STRATA_ARENA_WARM; one boolean env parser; stale
root pancake-*.json (hipBLASLt 100200 vs template 100401); exclude sycl/ (162K lines, 120/149 files mirror src/) and metal/ from
search/indexing; clangd fails on src/dsv41/ds4_model.cu (invalid AST) - exclude.  strata-poly builds in build-hip but
run-pancake.sh:49 calls it CUDA-only.

Suggested start order (owner's "biggest first", respecting prerequisites): 73, 35, 67 skeleton, 49/50/44 batch, 64 schema
and 63 probe (no GPU risk) in parallel; then 7 decision, 66 stages (31 -> 11 -> 16 -> 39), 12 in preset, 69/38; dan-cd in
parallel: bisect -> 8/28, 42 -> 14, 2; then Tier B (68, 65, 72, 63 derivation), then C/D.

## Phases

### 0. Bring-up (after the reboot)

- Driver 595 loads, `nvidia-smi` sees the 3060; optional gpu-unlock checked in `journalctl`.
- CUDA 13.1 build of Strata for sm_86 on the D drive; HIP build for gfx1100 compiles.
- Download the fastest Flash-Next pack, Qwen3.6-35B UD-Q4 with MTP, DeepSeek 2.00 bpw as disk allows.
- **Done when:** all three models answer on the 3060 and the base numbers are in `bench/results`.

### 1. Baseline and profile

- Base Strata on the 3060: MTP on/off, n-gram on/off, both, at each context length.
- Profile where a decode step goes: head, attention, GPU experts, CPU experts, PCIe copies, drafting, sampling.
- Measure where Flash-Next's n-gram table sits (VRAM, RAM, disk) and what reading it costs.
- **Done when:** a written breakdown per model says which phase below targets the biggest cost.

### 2. FastStrata's measured options, Qwen3.6-35B first

Port from FastStrata's `src/poly` into Strata's `src/qwen35`, in its measured order, then try the same ideas on Flash-Next:

| Option | Gain FastStrata measured (3060 Laptop 6 GB) |
| --- | --- |
| `--attn-split`: decode attention in one pass on the tensor cores | +30% decode at 37K |
| `--graph-experts`: each layer as one CUDA graph | -7 to -10% time per window |
| `--elastic-prefill`, `--lend-staging`, `--stage-next`: wider prompt batches, overlap the next layer's expert copy | agent turn after 38K: 6.4 s -> 3.9 s |
| `--suffix-drafts`: n-gram drafts from every past conversation | repeated task 90 -> 98-107 tok/s |
| `--cpu-multi`: CPU K-quant experts read once per window | +1 to +5% |
| `--kv-q8` | longer context in the same VRAM |

HIP: `--graph-experts` maps to HIP graphs; the tensor-core attention needs a WMMA path for gfx1100 or stays CUDA-only.

### 3. Drafting: MTP and n-gram (the key point)

- From basalt: the 2-bit draft-head screen (cheaper draft scan), the K/V append beside the next draft query,
  `--relaxed-drafting` as a lossy opt-in.
- Auto n-gram: turn suffix/n-gram drafts on when they pay (code edits), off when they do not (TensorRT-LLM's idea).
  Target: keep the old llama.cpp result on code edits (118 -> 175 tok/s with ngram + MTP) or beat it.
- Speculative sampling (SGLang `speculative_sampling.cu`): keep acceptance high at temperature > 0.
- Small draft trees (SGLang `tree.cuh`) against chains. On a MoE with experts in RAM, each branch wakes more experts,
  so this stays only if measured faster.
- **Done when:** each item is measured on the code-edit, chat and agent cases, kept or dropped with its numbers.

### 4. Exact two-stage output head (basalt)

- Port basalt's `q6k_head.cu`, `q5k_head.cu`, `decode/verify/head_split.cpp` to Flash-Next and Qwen3.6-35B. No
  Blackwell intrinsics in them; HIP through the existing hipify path. The 3060's 360 GB/s should make the head a large
  share of a decode step (phase 1 confirms).

### 5. DeepSeek V4.1-Flash on the 7900 XT

- Strata's DeepSeek engine runs on RTX 30 but on AMD only gfx1201 (R9700) and gfx1151 (Strix Halo). Add gfx1100.
- Needs the 7900 XT in the machine to run; until then compile-only.

### 6. Grammar inside speculative decoding

- basalt's grammar engine (`src/program/grammar`: GBNF, JSON schema, tool-call forms; CPU, portable).
- The token mask applied on the GPU (SGLang `apply_token_bitmask_inplace_cuda.cu`), checked per verify window so MTP
  keeps drafting under a grammar.
- **Done when:** tool calls are always valid JSON for the schema and decode speed with a grammar is within a few
  percent of without.

Status 2026-11-08 (branch `feat/grammar`): **done**.  basalt's grammar engine ported (d64768699 GBNF / JSON-schema /
tool-call builders and 02e1006fe the per-request guard; the engine credited to llama.cpp, MIT), `Verifier::resample_row`
(c7813a6c3), the engine's `--grammar on` and request `grammar=` key (011e13d72: after each verify window the first kept pick
the grammar refuses is drawn again under its mask and the window ends there, so MTP keeps drafting), and the server's key
for tool calls and JSON response formats (`serve/grammar.py`).  CPU tests: grammar_test, grammar_schema_test,
grammar_guard_test, serve/test_grammar.py.  On the 3060: 3 strict json_schema requests (enum + required, two of them
off-topic prompts) and a tool call with an enum argument all valid; 0.3-0.8 ms of grammar code per request, 35-110 us per
50-130 ms window (0.1-0.2%), 0-1 masked redraws per request.  The IQ2_XS presets run with `--grammar on`.  Not covered:
`--batch` slots (refused with an error, as basalt's first version), a layer split's head stage (resample_row forwards).

### 7. Cache tiers

- FastStrata's `--conversation-disk` (70K conversation after a restart: 83 s -> 7.2 s), for all three engines.
- One shared cached prefix (system prompt and tools) across agent conversations, as SGLang's RadixAttention does.
- basalt's `--idle-sleep`: free VRAM between requests, wake in seconds instead of a cold load.

Status 2026-11-08: the shared prefix is in the engine (a new conversation that starts inside a parked one borrows its
checkpoint); conversations to disk exist as session files (`slot_save_path`, `/slots/0?action=save|restore`), by hand
rather than FastStrata's automatic `--conversation-disk`; VRAM back between requests exists as `--vram-elastic` +
`POST /v1/vram` (no reload), but the engine refuses it with the pinned RAM complement the daily IQ2_XS preset uses (the
complement holds only the experts the VRAM cache does not), and `--idle-unload` is a cold reload (~1 min).  basalt's
sleep-in-place (64167ba73, ~815 lines: drop and redo the RAM copy's registration, refill on wake) is what closes that gap;
left for after the speed and quality items.

### 8. Additions from the forks and reports read on 2026-11-07 (owner: "add them to the plan and bring them in")

Each lands as its own commit, measured (the bench and, where numbers change, exactness or KL), credited to its source.

| # | Item | Source | Engine | Exact? | Expected |
| --- | --- | --- | --- | --- | --- |
| 8.1 | Flash-Next IQ2_XS running on the 3060 with Speedy's sm_86 presets as the starting config; base numbers | Speedy (DevDock-AI) | main | yes (config) | the plan's first model, finally measured |
| 8.2 | Turn on the upstream opt-ins Speedy measured, A/B each on this PC: `STRATA_PF_GROUP_SCRATCH`/`_HC_REGION` (#4039), `STRATA_PREFILL_ROUTED_STREAM` (#4283), `STRATA_PROMPT_ATTN_IMMA`, `STRATA_KV_PREFETCH` | Speedy, upstream PRs | main | per its PR | prefill +5-33% on the 3080 |
| 8.3 | Port the ones upstream lacks: `STRATA_VERIFY_DEFER_CPU` (#2689/#2690, light CPU expert rows a layer late in verify), `STRATA_PSI_SHED` | Speedy | main | KL measured by the PR | decode GPU-wait cut |
| 8.4 | The 59.6K-token MTP draft vocabulary (English/code + CJK), lossless | Strata-rdna3 | main (and poly's `--draft-vocab`) | yes | draft step ~12% cheaper, decode +4% |
| 8.5 | Adaptive MTP depth (3-4 by measured cost) | llamAmpere | poly | yes | decode |
| 8.6 | A learned expert profile loaded at start | llama.cpp MoE cache PR | poly, main | yes | start-of-session decode |
| 8.7 | Two-stage prompts (`--prompt-split`) | StrataFast | main | check | long prompts 1.24-1.55x there |
| 8.8 | Fused flash-attention prompt kernel (D=256, GQA 8, Q8 cache, causal inline); INT8-IMMA attention variant | own; Speedy's IMMA idea | poly | numerics change: KL | the 38% at 42K |
| 8.9 | Opt-in lossy: cache-aware routing (colibri `CACHE_ROUTE`), SJ-KVaRN-style KV body (keys 8-bit, recent window), mb-MTP | colibri, llamAmpere | poly | no: opt-in, KL | decode / VRAM |
| 8.10 | RDNA3 kernels, tensor parallel, SDMA, idle-queue parking for the 7900 XT | Strata-rdna3 | main, HIP | per kernel parity | 2-3x stock on gfx1100; waits for the card |
| 8.11 | basalt's head kernels (owner's request): **done 2026-11-08** - `q5k_head.cu` ported (Q5_K heads: Q2_0 / IQ3_XXS files) and the same designs written for IQ4_XS (the IQ2_XS head in daily use): exact head GEMV, 2-bit two-stage draft pick | basalt | main | yes | +3-5% decode (draft pick), head x1.07-1.27 in its test |

8.9 measured 2026-11-08, cache-aware routing (`STRATA_ROUTE_PRIOR`, the resident experts' logits get a bonus; changes
the output), token-forcing check against the exact preset (stock noise 1.17-1.30% mismatch, 0 large-margin events) and
decode against the learned-profile preset (75.2 tok/s mean, short x2 + edit x2):

| lambda | mismatch | vs stock (90% CI) | large-margin events | verdict | decode |
| --- | ---: | ---: | ---: | --- | ---: |
| 2.0 | 9.90% | +8.67 pts (5.71, 11.95) | 8 | REJECT | 88.1 tok/s (+17%) |
| 0.5 | 2.89% | +1.66 pts (1.00, 2.41) | 0 | REJECT (rate) | 94.1 tok/s (+25%) |

0.5 changes about one token in 60 more than the engine's own noise, never with a confident wrong pick: an opt-in, not a
default.  `./run-pancake.sh iq2_xs turbo` (configs/pancake/flashnext-iq2_xs-turbo.json) runs it.

Prompt reading on the daily preset, measured 2026-11-08 (STRATA_PREFILL_TIMING=1, the 41,818-token prompt): 42.9 s
(976 tok/s; the 100 s of the RAM-capped trace run read lent experts from the SSD).  GPU timeline: expert GEMMs 29% (gate/up
17.8, down 8.5, dequant 2.9 - the FP16 path, not MMQ, on this card), GDN ~12%, PLE 10.6% (4.3 s on the host), QSA
attention 6.7% + projections 5.5% + select 2.6%.  So 8.8's fused attention kernel can give a few % here (it was 38% on
strata-poly), and the expert GEMMs are the prompt's target.  `--ple-io mmap`: long prompt 42.9 -> 40.4 s, PLE host 4.3 ->
3.3 s, decode not better (edit 75.9/88.6 vs 80.9/91.4): not adopted.
The prompt path's other expert kernels, same prompts (time to first token, 5.2K / 41.8K tokens; FP16 default 6.51 /
42.9 s): `STRATA_PF_WMMA=1` 8.76 / 47.9 s (worse both); `STRATA_PF_FUSED=1` (int8) 7.29 / 40.0 s (+12% / -7%).  Agent turns
mostly read short deltas after the conversation cache, so the fused kernels stay off; a chunk-size switch between the
two would be the next step if long cold prompts matter.

### Priority change (owner, 2026-11-08): Qwen3.8-Flash-Next for daily use

Daily use moves from Qwen3.6 (`strata-poly`) to Qwen3.8-Flash-Next IQ2_XS on the main engine (`run-pancake.sh`).
Main-engine items come first now: 8.2-8.4 (done or measured), basalt's two-stage head (8.11) on the engine's native
Q5_K head (`q5k_head.cu`, the same model family and width), 8.7 two-stage prompts, 8.6 a learned expert profile, and
the Qwen 4 readiness of phase 9. The poly-only items (8.5 adaptive depth: poly already has the cost-based `Guesses`
rule up to 3 guesses; 8.8 its flash-attention kernel) wait.

### 9. Qwen 4 (owner's request, 2026-11-07)

Status, checked on the official pages on 2026-11-07: **no Qwen 4 weights are out.** The Qwen organization on Hugging
Face lists Qwen3.8 (Flash-Next, 27B, 2.4T-A95B) and Qwen3.6 as its newest models. Qwen3.8-Flash-Next's own model card
calls it an experimental preview of the architecture that will underpin Qwen4: 125B total / 6B active, 48 layers in 12
blocks of three Gated DeltaNet + MoE layers and one Qwen Sparse Attention + MoE layer, a 20M-entry n-gram embedding
table (51B parameters) at layer 2, one MTP layer, 262K native context (1M extended). Third-party reports (not
confirmed by Alibaba) name Max / Plus / Flash API tiers and a 27B open-weights tier.

So the work is:

1. **Now:** every Flash-Next item of phase 8 (8.1-8.4, 8.7) runs on the Qwen 4 Flash architecture: the main engine's
   GDN, QSA, n-gram table and MTP paths are what Qwen 4 Flash will need.
2. **When the weights appear:** read the GGUF's architecture name and tensor list against Flash-Next's
   (`strata --families`, the layout guards in `src/core`), add the family to setup (`setup.py` families table, hashes),
   pack it, run the bench and the exactness/KL checks, and add its row to the README. Expected: little new code if the
   architecture matches the preview.
3. **A Qwen 4 27B dense tier**, if it ships open: Strata's expert offload does nothing for a dense model; serve it
   with a dense engine (llamAmpere-style KV and quant tricks are the reference for 12 GB cards) or keep it out of
   scope, decided when its size is known.
4. Watch the Qwen organization on Hugging Face and the Qwen blog for the release.

Disk (2026-11-07): Flash-Next IQ2_XS (shard 1 37 GB + n-gram table 27 GB + MTP 6.5 GB) is already in
`/home/dan/clone-of-d/Strata-data`; `~/Strata-data` on the root disk was a duplicate and is gone; the unused Qwen3.6
UD-IQ3_S file was deleted. DeepSeek V4.1 (239-274 GB) still does not fit.

### 10. CPU, RAM, PCIe and GPU in sync (owner, 2026-11-08)

Decode on the 3060 is a relay: per layer the router picks experts, the GPU runs the cached ones, the CPU reads the
missing ones from RAM (33 GB/s), the GPU waits, then the next layer starts; PCIe is mostly idle inside a step.  Everything
slow starts after the router decides.  Make the parts overlap by knowing earlier which experts come next.  None of these
changes which experts are used: the output stays exact.

| # | Item | Source | Measure first |
| --- | --- | --- | --- |
| 10.1 | Instrument a decode step: GPU idle waiting for the CPU, CPU expert time, PCIe busy, per layer | own (existing STRATA_* timers) | the ceiling on this PC |
| 10.2 | Router lookahead: predict layer L+1's experts from layer L's output; start the CPU's RAM reads and the PCIe copies early | colibri PILOT (idea) | prediction hit rate |
| 10.3 | Draft-informed prefetch: the MTP draft tokens' routing predicts the verify window's experts; copy them into VRAM during drafting | new | prediction hit rate |
| 10.4 | Finish together: when the CPU's share would end later than the GPU's, move one or two missing experts to the GPU | Avalon "hosted MoE" | per-layer finish times |
| 10.5 | Avalon's VRAM lending: unused KV-cache tail and idle compute buffers lent to the expert cache | Avalon | experts in VRAM |

Order: 10.1, then 10.2 / 10.3 by their measured hit rates, then 10.4, 10.5.

Predictors to compare (owner, 2026-11-08), all scored offline on one `--dump-routing` trace before any prefetch is built
(`tools/pancake/predict_experts.py`, branch `exp/expert-prefetch`): recall (predicted & used / used), miss cover (the
share of the cache's misses predicted) and waste (predicted, not held, not used), per budget of experts per layer.
1. Historical: the previous window's experts; decayed frequency (an EMA); both topped up.
2. Next-layer router lookahead (Strata's RouterLookahead; `STRATA_LOOKAHEAD_STATS=1` scores it in the engine).
3. Dictionary: (layer, last 1-4 tokens incl. the draft tokens) -> expert distribution, learned online from every verify
   window (EMA), backed off to shorter keys and then to the global frequency (a Beta prior); top-16 experts per layer per
   key to bound memory.  Strata's suffix index is the model for the lookup.
4. MTP-based: each main layer's router applied to the draft's hidden state.  Only if 1-3 leave enough misses uncovered:
   the MTP block's hidden state is not each layer's input, so it is the weakest-founded and the costliest to build.
5. Hybrid: the best of 1-4 merged by confidence.
Prior art (reported, not yet read here): SP-MoE, MoE-SpeQ, "Speculating Experts" (2026), DoMoE, and an Oct 2026 study
of expert bytes per speculative window.  The aim is speed on this PC, not a novelty claim.

Two more ideas from the owner's reading (2026-11-08), new for this engine and testable on the same trace:
6. Expert co-activation: per layer, which experts fire together (a conditional-probability graph); when one expert is
   predicted or seen, its partners are candidates too.  Scored as predictor 6 and as an add-on to 1-3.
7. Per-layer cache budgets: router entropy differs by layer; give the cache's slots to layers by measured miss cost
   instead of equally.  Scored in the replay simulation (replay_cache_policy.py), then built if it wins.

Already in Strata or in this plan (the owner's lists checked against the code, 2026-11-08): adaptive draft length and
the MTP-vs-suffix choice (DraftPolicy prices each window size and source by measured cost and acceptance), early draft
cut-off (`--spec-min-p`), token-grouped expert execution in a verify window (`native_expert_grouped`; `--cpu-multi` on
poly), fused dequantization in the matrix kernels (native_mmvq, MMQ), INT8 KV cache (`--kv int8`), prefix / conversation
caching and checkpoints, staging buffers that overlap copies with compute (prefill; FastStrata's `--stage-next`), an
adaptive expert tier (usage-based swaps).  FP8 KV: no FP8 tensor cores on sm_86.  Expert-sensitivity mixed-precision
quantization: needs the BF16 model (~250 GB) to quantize from, out of reach on this disk; Avalon's IQ3_KS hybrid is the
evidence it pays.  Age-based KV precision is 8.9 (SJ-KVaRN style).  An adaptive controller over all of it: later, once
each part is measured.

Design rules for 10.2 / 10.3 (owner's input, 2026-11-08, adjusted by measurements):
- Memory: the prefetch window comes only from memory nobody uses now (the KV cache past the current context, idle
  compute buffers: Avalon's lending), never carved out of the expert cache (upstream's Foresight slots lost on every box).
  A hard floor of free KV room: near it, low-confidence prefetches stop (no OOM, no KV offload).
- Confidence: prefetch only experts whose predicted routing weight passes a threshold; weight early draft tokens over
  late ones (token 1 is the most predictable).  The threshold and depth are set from the hit rates measured in 10.1-10.3.
- Streams: transfers on their own copy stream, apart from compute and KV writes; when verification rejects draft token k,
  unstarted transfers for tokens > k are dropped.
- Timing: stage layer N+2 / N+3 while layer N runs; prefetch an expert only when its PCIe time (PCIe 3.0 x16, ~12 GB/s)
  fits before the layer that needs it.

Results, 2026-11-08 (fast preset, IQ2_XS, 4236 cache slots, 4 prompts; `bench/prefetch/`):
- 10.1 done.  A decode window (3.05 tokens) takes 66 ms: verify 63.3 = GPU waiting to reach the CPU's results 17.5 +
  per-layer host 44.4, of which CPU expert compute 37.7.  Per layer and window: 17 VRAM hits, 7.5-12 CPU experts, 1.2-3.3
  PCIe copies; PCIe probe 13.4 GB/s.  The CPU's share is the bottleneck, as the plan assumed.
- 10.2 / 10.3 offline (`predict_experts.py`, tests in `test_predict_experts.py`): of the experts the cache misses, a
  prefetch of 16 per layer covers transition (routing-keyed dictionary) 29% at 65% waste, hybrid 27%, co-activation 26%,
  prev+freq 26%, frequency 25%, prev-window 20%.  History-only predictors are weak.
- The engine's RouterLookahead (`STRATA_LOOKAHEAD_STATS=1`) is about 2x better: its top-8 votes cover 24-35% of the misses
  at 60-69% precision, top-12 33-46%; k=14 recall 58-69%.  Lookahead is the predictor to build on (10.2 first);
  transition is the best add-on to test merged with it.
- Its cost: the predictor thread takes 30-35 ms of CPU per window, on the same cores as the CPU experts (37.7 ms).
  Before 10.2 can pay, the lookahead must get cheaper or move to the GPU; measure decode with it on vs off.
- PCIe budget: at 13.4 GB/s a 1.4 ms layer moves ~18 MB, about 12-15 experts, so a prefetch must stay within ~8-12
  confident experts per layer; the confidence cutoff from the design rules is required, not optional.
- Lookahead cost, measured (arena tier, 2 runs x short/edit, `bench/prefetch/la-*.jsonl`): decode 70.5 tok/s mean
  without the predictor thread, 61.4 with it scoring only (-13%; short 52.8/63.4 -> 43.8/59.2, edit 75.8/90.0 -> 66.5/76.0).
  CPU expert time per window rises 18-33% (the thread's 30-34 ms of router GEMVs compete with the 7 pool workers for
  cores and RAM bandwidth).  So 10.2 starts by moving the lookahead to the GPU: the next layer's router on this layer's
  input is a 512 x n_embd GEMV per token, microseconds there, and it frees the CPU entirely.
- 10.2 step 1 done: the lookahead runs on the GPU (`STRATA_LOOKAHEAD_GPU=1`, commit 7ba03d6fe): the verify window routes
  layer l + 1 on layer l's input and publishes the top-10 before layer l's doorbell.  Same predictions as the CPU thread
  (k=10 recall/precision 0.576/0.548 vs 0.576/0.553, 0.544/0.505 vs 0.535/0.501), predictor thread 30-34 -> 0.6-0.7 ms
  per window, decode with the scorer on 61.4 -> 66.2 tok/s (lookahead off 70.5; the rest is the scorer on the pool's
  path and run noise of ~20% between a prompt's first and second run).  On the arena tier the predictions have no
  consumer yet: step 2 is acting on them (Foresight's swap space first, which measured 0.91 on a 3060 with the CPU
  predictor, i.e. less than the predictor's own 0.87 cost).
- 10.2 step 2, Foresight's swap space fed by the GPU lookahead (`STRATA_FS_SLOTS`): 4 slots per layer, 32 copies per
  window: decode 49.9/63.8 short, 74.4/85.6 edit = 68.4 mean vs 70.5 off (0.97; upstream's 0.91 had the CPU predictor's
  0.87 inside).  The prefetch works (16140 experts served from 8790 copies, 97% of them the lookahead's) but starves:
  486k requests found no free slot, 100k over the copy budget.  8 slots / 64 copies: short 52.6/67.2 (vs 52.8/63.4) but
  the 5.3K-token prompt then failed (cuBLAS status 14, out of VRAM): the swap space is allocated after the cache is sized
  and eats the prompt path's headroom.  So the window has to come from idle memory (design rule 1, 10.5), not on top.
- 10.4 is already in Strata: PcieAuto measures the pool's time per missed expert (53-55 us here) against the link (13.4
  GB/s) and sets the share of every layer's misses the GPU takes over PCIe (0.20-0.25, all 48 layers) so CPU and GPU
  finish together.  Nothing to build; a per-layer refinement is not worth it at a 0.20 share.
- 10.2 step 3: the GPU lookahead is on by default (STRATA_LOOKAHEAD_GPU=0: the CPU thread).  On the mmap'd-expert tier,
  where the lookahead runs by default to warm pages: decode 52.0/64.1/72.6/79.2 -> 56.5/67.6/81.0/90.0 tok/s (+10%,
  every run faster; `bench/prefetch/mmap-*.jsonl`).  Outputs: the arms differ only where the baseline arms differ among
  themselves (near-tie tokens, e.g. la-off's second short run splits at token 133 from every other arm).
- 8.7 (two-stage prompts) needs a second GPU: StrataFast's `--prompt-split` runs each chunk's late layers on another
  card while the main one starts the next chunk.  Blocked on hardware, like 8.10.
- 8.6 done: a profile learned on agent-style requests over source files the bench does not use
  (`tools/pancake/learn_profile.py`, `data/expert-profile-pancake.bin`) against the shipped one: decode
  52.8/63.4/75.8/90.0 -> 57.8/70.8/80.9/91.4 tok/s (+6.7%, every run faster, most on the cold first runs).  The IQ2_XS
  presets use it now; the Q2_0 presets keep the shipped one until measured.
- Phase 3, measured ceiling: the draft step is 2.5-3.2 ms of a 46-130 ms window (2-5%), so basalt's draft-head items (2-bit
  screen of a Q8_0 draft head, K/V append beside the draft query) can give 1-2% at most here.  Speculative sampling at
  temperature > 0 is in the engine (`STRATA_SPEC_PROB`, coupled, Gumbel; measured upstream on a 3060: the default path is
  as fast or faster), and the MTP/suffix choice is DraftPolicy's.  Lower priority than anything on the CPU experts.
- 10.3 is covered by 10.2: the GPU lookahead routes every token of the verify window, and those are the MTP draft's
  tokens; the separate MTP-head router predictor (rated weakest above) is not needed.
- 10.5 does not pay here: `--kv-resident 8192` gives the expert cache the idle KV VRAM a lend would (4188 -> 4406 slots,
  +5%); CPU experts per layer-window fall ~4%, decode 68.5 tok/s vs 70.5 (noise).  Dropped (`bench/prefetch/kv8k.jsonl`).
- Item 7 (per-layer budgets) does not pay: in the replay, swaps across layers give 48.3% of routed entries from the cache
  against 48.4% for the engine's per-layer swaps (`replay_cache_policy.py --cross`, `bench/prefetch/cross-layer.txt`); the
  profile already spreads 62-110 slots per layer.  Dropped.
- Memory: a run without `STRATA_RAM_KEEP_GIB` used the whole 57 GB and killed the desktop session.  Runs now use
  `STRATA_RAM_KEEP_GIB=12` inside `systemd-run --user --scope -p MemoryMax=54G` (owner, 2026-11-08).  The trace above ran
  under a 42G cap, which fell back to mmap'd experts (the arena needs 45 GiB): speed differs, routing does not.


## Phase 11: RX 7900 XT (gfx1100), started 2026-11-08

The owner switched back to the RX 7900 XT. Owner requests this phase (all in scope): resume on the 7900 XT; port Strata-rdna3's
RDNA3 work ("any percent is good": no fusion skipped); try again everything that was dropped on the 3060; make the CPU side
faster without losing the fast preset's gain; research in depth (5 layers: survey -> techniques -> mechanisms -> code ->
expected gain; papers, MIT/Harvard/Stanford/Berkeley/CMU/Tsinghua..., Chinese and Western companies, open-source engines;
memory, caching, prediction, statistics, direct GPU commands); compare with Avalon and stock Strata (numbers from the owner's
repos llama-cpp-7900xt-optimization and strata-7900xt-optimization); keep the RTX 3060 (CUDA) line unchanged.

Research report: [reports/Pancake MoE engine speedup.md](reports/Pancake%20MoE%20engine%20speedup.md) (notes in
research_notes/Pancake MoE engine speedup/).

Done (branch port/rdna3-decode, local):
- HIP build (`build-hip`, gfx1100); run-pancake.sh picks it when no NVIDIA card is present.
- Ported from Strata-rdna3: RDNA hyper-connection read (+Q8, A8, INJ_UP, FUSED), wave32 multi-row GEMVs (RDNA_Q6K, MMVQ_WAVE),
  COMBINE_CPU, WAIT_FUSE, ROUTE_PUBLISH.  rdna_gr.cu / rdna_mmvq.cu build on wave32 HIP only; CUDA gets stubs.  Not ported:
  long-context top-k (= upstream STRATA_SELECT_LONG), PF_GROUP_RES (= upstream group_resident), PA_KLDS (upstream rewrote the
  kernel with STRATA_PA_RPF), RDNA_PAIR (neutral, off upstream), GDN megakernel (0.0 ms on one GPU: the window waits on CPU rows).
- Decode 87-101 -> 104-108 (kernels) -> ~112 tok/s (fusions); Strata-rdna3 itself 109-119; Avalon 95-100; stock Strata 75-77.

Round A (load-gated, base 111.7 / 92.5 after an 11.7K prompt): --pcie-frac 0 118.4 / 104.3; + adapt every window, 160 swaps
121.2 / 100.1; rdna3's adapt settings 116.4; --kv-resident 8192 114.4; STRATA_SPEC_COST 112.0 / 98.3; DFlash2 115.9 / 84.3;
lossy: turbo 149.3, tail skip 7 129.7.  Worse: upstream group A switches together 101.2 (bisecting), ADAPT_LAG 87, zh60k vocab
92.5, rdna3 profile 105.6, upstream HC_Q8 107.2, kv-resident 131072 106.2, mmap tier 108.6.  Flat: --spec 5/6, full vocab,
SELECT_LONG, mtp-window, hw queues.  VERIFY_DEFER_CPU gave non-finite logits with COMBINE_CPU (fixed: cfuse off when deferring).

Replay ceilings at 10,909 cached experts (44%): engine tier 79.6%, per-layer LRU 89.5%, Belady 94.7%; tier every window x160 85.0%.

Done 2026-11-08 evening: rounds B/C, presets fast-hip (default) / hip (exact) / turbo-hip (commit f0f9c5788); token forcing
INCONCLUSIVE +0.4 pts, 0 large-margin events; strata_bench --long and strata-7900xt bench.py beat Avalon and stock Strata
(bench/results/2026-11-08-7900xt-final-suites.txt).  VERIFY_DEFER_CPU still non-finite on HIP with the ported kernels.

Status 2026-11-08 late.  Done: CUDA build check (compiles; GPU run needs the 3060), final comparison (beats Avalon and
stock Strata), and measured with no gain: reuse-aware admission (PCIe 3.0 cost), L3 prefetch (STRATA_ARENA_WARM, worse), CPU pool
settings (defaults best), IQ2_S two-half decode (STRATA_IQ2S_PAIR, neutral; the kernels are compute-bound), window thresholds and
--spec 3/5/6, tandem.cpp's MTP_DEFER (already here).  Added from tandem.cpp: STRATA_POOL_PREREAD (idle pool workers pre-read the
lookahead's next-layer CPU experts); A/B pending with the Q2_0 model test.

### Phase 11 research index (2026-11-08)

| Research | Where | What it produced for the plan |
| --- | --- | --- |
| qwen38-megakernel (cutlerbenjamin1-cmd, AGPL, CUDA dense 27B) | this session, notes in memory | ideas only (AGPL vs MIT): GDN replay already here; in-context draft-vocab tokens -> tested full vs en vs zh60k vocab (no gain) |
| Strata-rdna3 fork (nicejunjie) | MOD_PLAN phase 11 "Done" | RDNA3 kernels + COMBINE_CPU / WAIT_FUSE / ROUTE_PUBLISH ported (87-101 -> ~112 tok/s); megakernel GDN = 0.0 ms on one GPU (reverted, revisit after the GPU profile) |
| Six-thread deep research: CPU experts, caching/prediction, MoE speculative decoding, RDNA3 GPU decode, PCIe/KV, engines/labs/companies (MIT, Harvard, Stanford, Tsinghua, DeepSeek, Moonshot, Alibaba, ByteDance, Huawei, OpenAI, Anthropic, xAI, NASA: none relevant from the last four) | reports/Pancake MoE engine speedup.md, research_notes/Pancake MoE engine speedup/ | --pcie-frac 0 (+6/+13%), adaptive tier every window (+16% total), Belady ceiling replay (79.6 / 89.5 / 94.7%), expert deferral (lossy, broken on HIP), DraftExpert-style miss-aware windows (tested: policy already optimal), tail-row drop (item 5) |
| Avalon / stock Strata numbers (owner's repos llama-cpp-7900xt-optimization, strata-7900xt-optimization) | bench/results/2026-11-08-7900xt-final-suites.txt | final comparison: pancake beats both |
| tandem.cpp (ewhacc, MIT) + tandem.cpp-amd (itlab-cl) | this section | STRATA_POOL_PREREAD ported (A/B pending); MTP_DEFER already here; MTP head Q4_K requant not needed (ours already ~2-bit, 787 MB) |
| Pattern recognition (ExpertFlow, ST-MoE CCT/HT, MoE-Beyond, FineMoE) | this section | items 3-4: per-conversation expert working set, per-domain profiles |
| Stanford engines (Tokasaurus, Hazy Research megakernels, Cohere megakernel 2026) | this section | no new 2026 Stanford engine found; megakernels matter only for the GPU part (item 1) |
| Q2_0 quality (docs/QUANTS.md) | this section | Q2_0 = opt-in presets (134.7-136.1 tok/s; 80.6% same top token as Q8_0) |
| Strategy (where the time goes, ranked steps) | https://claude.ai/artifact/KS4sEzwMLwEGhVEPdi3xyF | the step order below; results log kept there |
| Avalon upgrades report (owner's /home/dan/reports/Avalon upgrades from upstream projects.md) | this section | for pancake: the gfx1100 hipBLASLt table must match the runtime (the preset loaded 100200 on hipBLASLt 100401 -> untuned; 100401 table A/B queued; Strata measured +32/+78/+82% prompt on a 7900 XTX); longer prompt-lookup windows (--lookup-max-t 12/16, +11-14% on edits per Strata) A/B queued; tool-call recovery, env_to_user, reasoning_close_retry already in serve/; for Avalon itself: source fast-forwarded c24c03d -> 88313b6 (48 commits; rollback branch local/c24c03d-baseline), rebuild after the benchmarks, model needs ~20 GB more free on D |
| llama.cpp CPU share (PR #30966, merged 2026-11-07; #31888; #31233 ClassLRU; #31286 open) | this section | already here: Strata's PCIe share is the same split (--pcie-frac 0 = their share ~0.95, measured best); ClassLRU's long-memory scoring (half-life 128 tokens) replayed on our tier: decay 0.7..0.986 within 0.5 points at 32 swaps/window - no gain, not ported |
| Runtime architectures 2026 (disaggregated serving, control/execution plane, heterogeneous runtimes, runtime intelligence, single-node applicability) | reports/LLM runtime architectures 2026.md, research_notes/LLM runtime architectures 2026/ (notes in English) | pasted overview: names real, details inflated (TokenSpeed "zero runtime errors" unsourced; xLLM = jd-opensource/xllm, Ascend-only benchmarks; Exo = agent runtime, 12-53% not "up to 53%"). For pancake: P/D disaggregation, chunked-prefill interleave, overlap scheduler, distributed KV stores do not apply; adopt items 10-12 below |
| Project Maya (mw00/project-maya, MIT, a Strata fork; 2026-11-09) | reports/Avalon upgrades from upstream projects.md, research_notes/Avalon upgrades from upstream projects/project_maya.md | not faster for us (its Qwen engine is a Strata 0.1.27 snapshot); ported loop guard, soft budget close, tolerant tool calls (section "Agent robustness from Project Maya"); SSD direct reads and route-share expert skip left as test ideas |
| Stanford papers 2022-2026 (reports/Stanford papers useful for Avalon.md) | research_notes/Stanford papers useful for Avalon/ | per-level thinking cap (done, 5dd620fb9); multi-entry prefix cache (RadixAttention); CATS (superseded, item 13); measurement harness |
| MIT/Harvard papers 2022-2026 (reports/MIT Harvard papers for Avalon.md) | research_notes/MIT Harvard papers for Avalon/ | TEAL over CATS (item 13); GDN chunk 64 (item 18); XAttention opt-in (item 19); learned thinking stop (item 21); bit allocation guidance (Scaling Laws for Precision) |
| Rest of US universities 2022-2026: Berkeley, CMU, UW, Princeton, Cornell, UCSD, Duke, Rice, Stevens, ... (reports/US university papers for Avalon.md) | research_notes/US university papers for Avalon/ | items 13-17, 20, 22; SuffixDrafter/DraftPolicy already cover TurboSpec/Sequoia-style length choice; QTIP/IQ2_KT not worth a requant; Europe, Asia, Canada, Pacific/Africa/Latin America reports follow |
| European universities 2022-2026: ETH/EPFL, ISTA + Central Europe, UK, France/South/Nordics/Israel (reports/Europe papers for Avalon.md) | research_notes/Europe papers for Avalon/ | daily files are ISTA-DASLab GSQ-RCO (Q2_0 93.8 vs IQ2_XS 70.3 tok/s on DASLab's card, -0.09 task avg: Q2_0 confirmed); base is 176.9B incl. the 51B n-gram table; "IQ2_XS" experts are IQ2_S/IQ2_XXS/IQ1_M + Q2_0 down; items 23-27; cache-conditional routing (Qualcomm/UvA) = STRATA_ROUTE_PRIOR, finish item 7; security: GGUF-quantization backdoors (ETH "Mind the Gap") -> official files + SHA-256 vs HF |
| Asian universities + labs 2022-2026: Tsinghua/PKU, rest of mainland China, HK/Taiwan/Singapore, Korea/Japan/India, Qwen/DeepSeek/Moonshot/ByteDance/MiniMax/Xiaomi (reports/Asia papers for Avalon.md) | research_notes/Asia papers for Avalon/ | Qwen3.8 tech report checked against our code: MTP depth already --spec 4 (official 4.07 accepted at 4 steps), drafter has no PLE block and dense SWA attention (SGLang shortcuts already here), PLE via --ple-io direct (not mmap; Europe report corrected), grammar deferred after </think>; Avalon still at --spec-draft-n-max 2 (A/B 3/4); official thinking sampling T=1.0 noted, owner keeps 0.1; items 28-32 |
| Canadian universities 2022-2026: Mila/McGill/Polytechnique/Concordia, Toronto/Vector, Waterloo, Alberta, UBC, Queen's (reports/Canada papers for Avalon.md) | research_notes/Canada papers for Avalon/ | little for engine speed; "Displacement Is Not Direction" (Toronto): KLD/top-1 stop predicting task score among near-baseline quants -> token forcing only rejects large deviations, a fixed task set picks between close candidates (quality-gate change for items 8, 13, 23); items 33-34; SpAx = 4th mask in item 13; Cactus parked (does nothing under greedy, ~0 at T=0.1) |
| Australia/NZ, Africa, Latin America 2022-2026 (reports/Pacific Africa Latin America papers.md) | research_notes/Pacific Africa Latin America papers/ | little fits: SpecFit marginal new-miss cost (DISCO feature), UNSW restore-equivalence test, UFMG coordinate-descent kernel sweep, UBA top-k entropy signal; items 35-37; the report ends with the series-wide ranking below |
| Pancake pain points (2026-11-09; reports/Pancake pain points and solutions.md) | research_notes/Pancake pain points and solutions/ (engine.md, server_agent_ops.md) | edit windows: CPU experts 17.4 of 37.4 ms at ~80% of DDR4 (only fewer bytes / overlap / routing help); GPU spends 38% of kernel time waiting in combine_cpu; Q2_0 down not covered by expdec (native_down_multi_kernel<42> = 5.8% of GPU kernel time, new); opencode history (old backend): turn median 6.2 s, p90 50 s, p99 224 s, re-reads ~18% of turn time, 15 turns cut while thinking; pancake not yet wired to opencode (8080 vs 11437, no unit); order: 0 wiring + thinking caps, 1 measurements, 2a per-turn exact (35, 31, 11, 16), 2b engine exact (dan-cd), 3 lossy behind the revised gates, 4 harness/side branches |
| Technology families (2026-11-09; reports/Grouped technologies for pancake.md) | the nine reports + this plan | 10 families by mechanism (F1 fewer expert bytes, F2 placement/movement, F3 overlap, F4 drafting/verify, F5 per-turn cache, F6 thinking, F7 prefill, F8 GPU kernels, F9 quality/measurement, F10 harness/ops); F1-F3 are substitutes for the same 17.4 ms of CPU experts (edit-window floor ~20 ms est.); shared infrastructure: routing-trace format + time simulator, activation/hidden-state dump hook (items 13, 21), per-turn JSONL log with reasoning_tokens, revised quality harness, load-gated deterministic A/B, restore test 35, edit-window profile + 7900 XT prefill timing; one trace run answers items 13, 30, 28, 36, 15, 23, 5, 3-4, 17 offline; unowned: 23, 1/32/37, Q2_0 down kernel, 17 |
| Hardware data path (2026-11-09; reports/Hardware data path technologies.md) | research_notes/Hardware data path technologies/ | items 38-54; edit window ~ sum of parts (scheduling, not bandwidth; overlap floor ~20 ms); CPU Q2_0 at ~80% DDR4, promotions ~93% of Gen3; critical-path transfer = fp32 row pull in combine_cpu (0.4-0.8 ms PCIe est. vs <=0.27 ms GPU bound: item 50 decides); pre-doorbell GPU kernels critical in both windows (item 48 up); waves: 0 = 38, 54, 45 (+42/43 spec to dan-cd); 1 = 49, 50, 44; 2 = 46, 40, 41, 48; 3 = 47, 51, 53, 52 last; 39 after 35; gains not additive, few % up to ~10% on edits; item 8 stays the big lever |
| Consumer platform capabilities (2026-11-09; reports/Consumer platform capabilities.md) | research_notes/Consumer platform capabilities/ | items 55-62; CPU:promote ratio varies ~10x across platforms (today 1:2.7; Zen 3 B550/X570 or Rocket Lake ~1:1.2-1.8; LGA1700/1851 DDR5 ~1:2-4; AM5 + Gen5 card ~1:1 or inverted; Strix Halo no promotion) -> item 55 must calibrate on window time, acceptance test: it finds --pcie-frac 0 on the 7900 XT by itself (static PcieAuto picked 0.20-0.25 and lost); stage 1 (this box): 38, 57 cheap part (ISA/path log line, STRATA_FORCE_ISA avx512/avx-vnni), 49/50/44, 55+61 (emulate small cards by capping cache slots), 46/48/40/41/53, 42/43, 35->39/16/31/11, quality gates; stage 2 (needs hardware): 56, 58, 59, 60, 62, rest of 57, calibration re-check on Gen4/Gen5; 57 partly exists (STRATA_VNNI_TIER=1 AVX-512 without VBMI) |
| Pancake code deep research (2026-11-09; reports/Pancake code deep research.md; Serena MCP + clangd) | research_notes/Pancake code deep research/ (decode_path, prefill_path, memory_cache, server_agent, structure_build) | items 74-95b; no wrong output found, one functional bug (89 grammar on Anthropic/Responses); this session order: stage 0 (no GPU) 87, 89 fix, 88, 92, 95b; stage 1 (env-only GPU session) 82, 78, 76, PSI baseline; stage 2: 91 + 73 turn log, 38, 86, 85; stage 3: 83, 39, 90, 71, decision on 31; stage 4: 75, 74, 79, 80, 94, 95, server refactor, 81 with 18; hand-off list for dan-99 in the report |
| Commits of 2026-11-09 in source projects (research_notes/Source project commits today/commits_2026-11-09.md) | upstream Strata (214 since base 146df3c46, nothing to port), basalt, Avalon main, llama.cpp, ik_llama, Maya v1.1.57, forks | nothing speeds Flash-Next on gfx1100; ported: basalt edd09ff3f PLE close() hang (62a58f709), Maya 00655481c + StrataFast 3fbea9882 tool-argument types (2f0cee057); to dan-99: basalt 553e9290d (quarter-size last pool tasks), 4e6841265 (down-row prefetch under FUSE_PHASES, with 29), 1fd53a447 (commit queued behind the MTP draft, item 10); parked: llama.cpp 1945793e0 top-level response_format schema (low value), basalt 6370f0ba3 MTP wait pause/yield (~0%, fold into 74), basalt e7bd2d1ae DraftPolicy cost scale (43 owner); idea for 39: key saved sessions by the lossy routing switches (llamAmpere) so an exact preset never resumes a turbo-made state; Qwen official thinking sampling T=1.0/top_p 0.95/top_k 20/presence 0 (owner keeps T=0.1); DF2_GATE retired upstream -> close its ledger entry |

Development pass done 2026-11-08 night (owner: develop everything first, then test and optimize):
- items 3-4 merged from feat/pattern-cache and built (HIP and CUDA compile; expert_seed_test ok; 7/7 Python tests);
- 7900 XT presets use the matching hipBLASLt table (100401);
- item 11 = existing opt-ins STRATA_CACHE_MESSAGE_BOUNDARY=1 + --prompt-cache-tail (anticipatory boundary with the 32K chunks
  the preset already uses; author: first edit -33%) -> test arms;
- item 12: the thinking budget already exists (reasoning_budget_tokens, #123); Dynasor-style early exit deferred (needs a
  forked probe of the single decode stream, lossy, pays on long math reasoning, not agent coding);
- item 10 (MTP draft inside the verify graph) waits for a draft timing run (launch overhead vs the MTP layer's compute);
- item 8 (VERIFY_DEFER_CPU on HIP) waits for the bisect runs;
- Avalon source 88313b6 built (binary rollback: Avalon/bin-c24c03d-rollback; the model is still missing).
Test phase queue: scratchpad queue.sh order - GPU profile + draft timing first (they steer development), then the A/B arms:
hipBLASLt 100401, --lookup-max-t 12/16, pool pre-read, prompt lending/chunks, the small-gains stack, --mtp-q4, item 11
switches, items 3-4 seeds (needs conv_trace runs), the deferral bisect, then token forcing of the final preset.

Left, ranked (details of the measurements above stay in this section's history):
1. GPU path: the GPU's own work is now 15.9 of a 24.8 ms short window - profile (rocprofv3), then GDN megakernel (rdna3),
   group A switches one at a time, hyper-connection reads (~2.7 ms), the VRAM-hit expert phase (3.3 ms).  Exact.
2. Keep the decode cache through long prompts: the prompt path borrows 9,412 cache slots (12.6 GiB); decode after an 11.7K
   prompt drops ~126 -> ~105 tok/s while it refills.  Exact.
   RESULT 2026-11-09 (closed): tools/pancake/decode_curve.py - the same edit reply after a 13.8K prompt decodes as fast as
   after a 2K one, segment by segment; the drop is the long case's prose reply (acceptance 0.62 vs 0.87, slow start while
   the cache adapts).  No refill fix needed; the start-up belongs to items 3-4 and 14.  bench/results/2026-11-09-decode-curve.
3. Per-conversation expert working set (restore a resumed conversation's hot experts during its prompt).  Exact; needs
   per-session routing traces first.  CODE DONE on feat/pattern-cache (STRATA_CONV_WORKSET), see "Items 3-4" below.
4. Per-domain expert profiles (code / prose / tools, chosen from the prompt).  Exact.  CODE DONE on feat/pattern-cache
   (STRATA_DOMAIN_PROFILES); the offline gain is mixed until per-domain traces exist, see "Items 3-4" below.
5. Tail-row drop: replay with real MTP confidences before building.  Exact.
6. Q2_0 as a 7900 XT preset if its test wins (quality check first; a different quant of the model).
7. Token-forcing check of turbo-hip on the 7900 XT.
8. VERIFY_DEFER_CPU on HIP (non-finite with the ported kernels; lossy opt-in).
9. Phase 7 leftovers: sleep-in-place, automatic disk sessions.
10. MTP drafting inside the captured verify graph (9% of a window): first measure how much of the 2.2 ms is launch overhead.
11. Agent-turn-aware cache policy: anticipatory checkpoints at turn boundaries, one system/tool prompt prefix shared across
    sessions (with the coding agent's conversation working set, item 3).
12. Thinking budget with a natural transition phrase, and Dynasor-style early exit of reasoning (opt-in: changes the text,
    shortens replies rather than speeding tokens).
    2026-11-09: per-level caps (reasoning_budget_by_effort, 5dd620fb9) and Maya's soft close at a sentence end are in;
    the learned stop (item 21) replaces Dynasor as the early-exit route.
Added 2026-11-09 from the university paper series (Stanford, MIT/Harvard, rest of the US; reports/ "... papers for Avalon"):
13. CPU-expert activation sparsity, lossy opt-in: one offline FP32 pass on opencode traces with three masks - Berkeley's
    intra-expert gate mask (arXiv 2605.08575: Qwen3.5 MoE at 84-87% sparsity, 95% of score), TEAL (MIT; holds 40-50% on
    dense) and CATS (Stanford; collapses at 40% on Llama-3-8B) - at 50/70/85%, gated by token forcing.  Build the AVX2
    gather kernel (+ W_down repacked n_ff-major at load) only if the gate mask passes at >= 70%.  Ceiling ~1.25x decode
    (8.9 ms of CPU-bound time above the 15.9 ms GPU floor), realistic 10-20%.
    RESULT 2026-11-09 (closed, offline): STRATA_ACT_DUMP_RAW on the Q2_0 file, layers 4/12/24/36/44, 11,027 expert-token
    records (short + 2 code edits), tools/pancake/sparsity_masks.py: at 70% sparsity the gate mask keeps cos 0.844 (~54%
    relative error), CATS 0.818, even the oracle |h| mask 0.954; at 50% gate 0.939, oracle 0.989.  Flash-Next's experts
    (FF 640, fine-grained) are dense: the screen (gate cos >= 0.99 at >= 70%) fails, no sparse kernel.
    bench/results/2026-11-09-sparsity/.
14. Expert cache, time-based ceiling first: extend the replay with PCIe 3.0 promotion cost (~115 us / 1.4 MB vs ~42 us CPU
    compute), DDR contention and GPU slack; compare tier, LRU-with-promotion and a Belady that skips non-paying promotions
    (Baleen).  Only if >= 5% of decode is on the table: SIEVE/S3-FIFO dynamic segment beside the frequency tier, admission
    priced by that cost; then OpenEvolve policy search on the simulator scored on decode time (Berkeley ADRS) or a
    Hawkeye-style Belady-imitation classifier.  (Replay showed tier 79.6% < LRU 89.5% < Belady 94.7%, promotions free.)
    RESULT 2026-11-09: tools/pancake/time_ceiling.py (bench/results/2026-11-09-time-ceiling/runs.txt).  Serial model
    (GPU + CPU summed, as the engine runs today): LRU -20% short, oracle admission +8..15% short / +27..48% edit, online
    2nd-miss-in-8 +5..16% edit -> built as STRATA_ADAPT_KMISS.  Overlap model (--overlap: per layer GPU pre-routing, then
    VRAM experts beside the CPU pool; calibrated to the measured window; DMA contention alpha 0.25-0.5): code edit -
    item 29 +3%, deferral (item 8) +33%, deferral + rank guard (half kept on time) +24%, deferral + k-th-miss 2/8
    +45..52%; short chat - deferral +27%.  Test phase order follows: deferral, then k-th-miss, then 29.
15. Cross-session suffix index for SuffixDrafter (CMU SuffixDecoding: tree over prompt + all earlier outputs, frequency
    scoring, shared by sub-agents) - simulate with the drafter's --simulate first; check DraftPolicy's learned MTP table
    against Berkeley's measured MTP decay (0.92 / 0.68 / 0.38).
16. Checkpoint triggers at tool-call and tool-result boundaries (FreeToken, Marconi) and Marconi's FLOP-saved-per-byte
    eviction across the parked tiers; a checkpoint after </think> only if the template keeps earlier thinking.
17. Pre-stage the expected expert set while a tool runs (KVFlow applied to experts) - simulator first (L3 prefetch and
    DraftExpert windows failed; this targets the idle tool window, not the decode window).
18. GDN prefill: chunk 64 with blockwise 16x16 inverse + fused prep kernels (MIT GDN/FLA line; 5-11% of prefill).
19. XAttention block-sparse prefill above ~64K, lossy opt-in, gated by code-recall tests (20-40% prefill at 131-262K, est.).
20. ReplaySSM audit of the GDN replay (1-3% decode; up to 8% on speculation only if a snapshot is copied per draft token).
21. Learned thinking stop: a linear probe on hidden states trained offline on logged turns (Thought Calibration, OS-Pruner,
    Learning How Hard to Think; 20-40% fewer thinking tokens on routine turns, untested on agents).
22. FreeToken's PCIe/CPU miss split vs --pcie-frac: read the code, very likely the same mechanism - close unless its
    copied experts are also promoted into the cache.
Added 2026-11-09 from the European report:
23. AcceptMoE-style depth-graded verify prior (Imperial): no bonus at draft position 0 (the committed token stays exact),
    growing cache-residency bonus with depth, weighted by DraftPolicy's measured acceptance; lossy opt-in, token forcing.
    Different from the DraftExpert windows: it changes the verify expert set, not the window length.
24. IQ1_M expert rows (3 layers of the "IQ2_XS" file's gate/up) take the ggml-cpu path in src/kernels/cpu/iq_avx2.cpp:
    add a native AVX2 IQ1_M row kernel (1-4% decode on the IQ2_XS preset, 0 on Q2_0; iq_avx512.cpp's comment is wrong).
25. ISTA-DASLab Coder variant (256 of 512 experts, 29.6 GB resident, SWE-bench Verified 95.7% of BF16, ~3.5 bpw LUT types):
    A/B one evening; compare shard-2 hash first so only ~29.6 GB is fetched; regenerate profile 8.6 and items 3-4 seeds
    (expert ids change).  D space competes with the Avalon model (~20 GB short).
26. Semantic constraint on the edit tool's oldString (must occur in the file) and grammar-constrained tool-call JSON inside
    speculative decoding (ETH type-constrained decoding, PLDI 2025: >50% fewer compile errors) - fewer failed turns.
27. Item 14 simulator gets Harmonia/Athena (SAFARI, online-learned placement) and MoE-Infinity (activation-matrix match)
    as candidate policies; item 13 gets the Polish/Sapienza per-layer thresholds (no new item).
Also: presence_penalty must stay 0 for tool calling (DASLab MoESQ card: SWE-bench 74.7 -> 70.6 with it on).
Added 2026-11-09 from the Asian report:
28. Item 8 rank guard (KTransformers, SOSP'25): STRATA_VERIFY_DEFER_CPU_BELOW (0.1) and "never the last layer" exist; add "always
    immediate when in the token's top-2", measure at verify windows 1/2/4/8 after the HIP non-finite bisect.  Hand-off stays the
    doorbell/partial-flag protocol (no HIP graphs).
29. Per-expert barriers in the CPU pool instead of the global run_phase sync between gate/up and down, SiLU*mul fused into gate/up
    (Tsinghua OSDI'26) - a few percent of CPU expert time.
30. Offline recall test of three next-expert predictors on routing traces (next-layer router on the current hidden state; the
    pre-attention state; the MTP hidden state), recall@10/16/24; use only with PCIe 3.0 promotion cost counted (feeds item 14).
    ProMoE scheduling (chunked, cancellable transfers; cached -> in-flight -> CPU order) goes into the item 14 simulator first.
    RESULT 2026-11-09 (predictor 1 measured, item closed): STRATA_LOOKAHEAD_STATS=1 (exists), fast-hip, short + code edit,
    K 10/16/24: the next-layer router on layer l's input covers 19-21% of layer l+1's non-resident experts (precision
    13-16%, 3.6-5.8 predicted/layer).  At ~117 us of DMA per promotion vs 23-60 us of CPU per miss, predicted promotions do
    not pay; predictors 2-3 (pre-attention state, MTP state) are further from the routing than the current layer's state:
    not built.  Item 27's candidates are covered by k-th-miss admission (online) and the conversation working set
    (MoE-Infinity-style match).  bench/results/2026-11-09-lookahead-recall/.
31. preserve_thinking end to end: server and template honour it; check that opencode resends earlier reasoning, else the
    conversation cache re-reads from the first dropped block every turn.
32. Gated Residual kernel count at batch 1 (SGLang small-M fusion idea) inside the item 1 GPU profile; FP8 irrelevant on RDNA3.
Item 13 also gets CHESS per-channel thresholds, union sparsity over verify windows, transposed down layout.  Item 19: BFLA
(CityU; benchmarks Flash-Next itself, 1.34-1.44x TTFT at 128-256K) is the candidate instead of XAttention - port in progress on
feat/bfla-hip.  Any-Precision for draft passes does not apply (MTP drafter keeps all 512 experts in VRAM, 835 MiB).
Added 2026-11-09 from the Canadian report:
33. Loop guard behaviour counters (UBC/Queen's: failed agent runs cost 1.4-4.4x the tokens, agents do not fail fast): N edits
    without a build/test, M reads/searches without an edit - log-only first, then a note like the tool-repeat note.
34. Rumination signal in thinking (McGill/Mila Thoughtology): rolling 5-gram repetition rate feeding the soft close
    (reasoning_budget_grace) and the learned stop (item 21).
    2026-11-09 CODED (branch feat/item-34-rumination, serve/rumination.py, serve/test_rumination.py; no engine change).
    Phase 1, always on (log only, the reply is unchanged): word 5-grams over the last 256 thinking words, a check every
    32 words from word 128, rate = (grams - distinct) / grams; per request `done["rumination"]` {max, mean, checks,
    closed}, history/turn log `rumination_max`, `rumination_mean`, `rumination_closed`, /metrics `rumination`, Prometheus
    `strata:rumination_measured_total` / `_closed_total`.  Phase 2, opt-in config `"rumination_close": 0.4` (or
    {"threshold", "checks": 3, "min_tokens": 0}): the rate at or above the threshold for K checks in a row -> the soft
    budget's own close (first sentence or line end, at most reasoning_budget_grace tokens, THINK_GRACE when 0; the
    REASONING_WRAP_UP text with the model's end of thinking).  The loop guard's exact-loop handling is untouched.  No
    threshold measured yet: read the turn log's rumination_max on real turns first (the paper gives none).
Quality gates (revised): token forcing rejects large deviations only; choice between close variants by a fixed task set
(LiveCodeBench subset + BFCL v4 multi-turn + own-repo checkpoint grading), repeated runs as distributions.
Added 2026-11-09 from the Pacific/Africa/Latin America report:
35. Restore-equivalence test on the real engine: a full-hit restore of a parked Qwen conversation must match a cold recompute
    token for token (check cache_n/prompt_n off-by-one; UNSW found such a bug in a hybrid-state cache).  Precondition for
    items 3, 11, 16, 31.
36. DISCO (feat/disco) features: per-position marginal NEW missed experts (SpecFit, Beihang/Monash; log the variance first)
    and top-k logprob entropy (UBA Entropy Sentinel), the latter also as a log-only loop-guard signal (items 33-34).
37. UFMG coordinate-descent sweep of HIP launch/tile parameters and AVX2 block sizes, after the item 1 GPU profile (1-3%).
Series-wide ranking (all seven university reports; precondition: the revised quality gates):
  1) GPU profile + kernel tuning (1, 32, 37)  2) turbo-hip decision (7)  3) offline sparsity masks + per-expert pool barriers
  (13, 29)  4) correct, cheap per-turn cache with the restore test (35, 31, 16, 11)  5) keep the decode cache through long
  prompts (2)  6) fix item 8, then the rank guard (8, 28)  7) DISCO / SpecFit / AcceptMoE (15, 36, 23)  8) effort and thinking
  (12, 21, 33, 34)  9) time-based cache simulator with candidate policies (14, 27, 30, 3-4, 17)  10) tool reliability (26,
  presence_penalty 0).  Item 6 closed (Q2_0 is the daily file); item 24 gives 0 on Q2_0.  feat/contextpilot: keep off
  (1.6% dedup savings measured on 40 real opencode sessions).
Added 2026-11-09 from the hardware data-path round (research_notes/Hardware data path technologies/; report follows):
38. Linux direct expert-arena loader: experts_unbuffered() returns false on Linux ("buffered (not Windows)", pinned.cu ~1160) and
    load_experts_direct is _WIN32-only, so the 33 GiB arena is read through the page cache (log: 33.02 GiB at 0.49 GiB/s) and
    competes with the locked arena on 57 GB (likely oomd trigger, 0.51-1.18 GiB/s spread, no free THP blocks).  Port the direct
    loader onto src/platform/direct_file.cpp (16-thread O_DIRECT pread; PLE already reads this way on the same ntfs3 partition).
    Est. start 28-65 s -> ~10-14 s (NVMe at PCIe 3.0 x4).  Stop-gap: POSIX_FADV_DONTNEED after each buffered chunk.
    (ServerlessLLM OSDI'24, fastsafetensors.)
39. Automatic disk tier for parked conversations in strata --serve (conversation_file.cpp has direct 16 MiB writes + fsync +
    atomic rename; today only /slots save/restore by hand; --conversation-disk exists only in strata-poly), restore on start
    (~0.5-1 s for 1.13 GB est.); keep tool-call-ended conversations in RAM, demote final-answer ones first (Bidaw, FAST'26);
    cap the tier size (D is 91% full).  Gate on item 35.
    [CODED 2026-11-09 on feat/item-39-disk-tier, not GPU-tested; CPU test tests/core/conversation_disk_tier_test.cpp]
    Opt-in: engine args `--conversation-disk-dir DIR|auto` (auto = <cwd>/conversation-disk) and `--conversation-disk-max-gib N`
    (default 20, for the whole folder), or the preset's "env": STRATA_CONVERSATION_DISK_DIR / _MAX_GIB.  Presets left alone.
    ConversationCache::set_disk_tier: evict_oldest moves the entry to ConversationDiskTier (src/core/conversation_disk_tier.cpp)
    instead of freeing it, final-answer conversations before tool-call ones (<tool_call> after the reply's <|im_start|>,
    STRATA_TOOL_CALL_TOKEN=248058, -1 = plain oldest-first); a writer thread runs session_file_write keeping the root AND the
    deepest checkpoint, then c-<serial>.idx (token lists, ~4 B/token).  Subfolder per key = model file + session identity +
    lossy env switches (STRATA_ROUTE_PRIOR, ROUTE_TAIL_SKIP, ROUTE_RESIDENT*, ...; STRATA_CONVERSATION_DISK_KEY_ENV adds names),
    key.txt + flock as strata-poly; LRU cap across subfolders; leftovers deleted at open; ENOSPC turns the tier off.  A request
    whose prompt continues a disk entry by > STRATA_CONVERSATION_DISK_MIN_GAIN (1024) tokens past RAM reads it back with /slots
    RESTORE's limits + RAM preflight and parks it, so the parked path validates and takes/borrows it.  QUIT hands what is
    parked to disk.  Shorter than STRATA_CONVERSATION_DISK_MIN_TOKENS (2048) is not written.  One GPU, no --batch.
    GPU test still to run: restore_check.py-style - resumed-from-disk vs STRATA_CKPT_REREAD, token for token, --adapt-swaps 0.
40. Software prefetch 1-2 KB ahead inside the Q2_0 row kernel (q2_avx2_rows.inl row_multi has no _mm_prefetch; hardware
    prefetchers stop at 4 KB pages, a gate/up row is 720 B): env-tunable, exact, 0-5% of CPU-expert time.
41. Smallest saturating CPU thread count with FUSE_PHASES (variance of window time), and the cpufreq "performance" governor
    (owner setting) - 0-3% each.  Q2_0 kernel is already near the bandwidth roofline: no LUT rewrite (T-MAC/Vec-LUT ~0).
42. Overlap-aware simulator model (spec for the item 14 owner): tools/pancake/time_ceiling.py sums GPU and CPU (max(gpu + cpu +
    alpha*dma, dma)), so it cannot price items 8, 28, 29, 22 or reordering; per layer cost = max(GPU part, CPU part + sync) with
    deferral overlapping layer l CPU with layer l+1 GPU; calibrate on window ms and the 38.3% combine_cpu spin, validate to
    5-9% error (Vidur, LLMServingSim2.0) before the >=5% rule; measure alpha (DDR contention CPU vs DMA) locally.  Edit window
    37.4 ms is ~ the plain sum of its parts; ideal overlap ~20 ms.
43. DraftPolicy::seed_costs() from the simulator: cost of each extra draft position = CPU time of the distinct NEW missed
    experts it adds, per context bucket, from routing traces (MoE-SpeQ, SpecFit), replacing the fixed kShape curve measured
    on the RTX 5070.  Exact, 0-5%.
44. Critical-path pass over pancake's per-layer timers: slack per layer and "what if this part were x% faster".
45. I/O hygiene: STRATA_IO_THREADS sweep against the startup PLE probe; filefrag on experts.bin and the PLE shard; reorder PLE
    rows by hit frequency only after PLE wait time is measured.  Not worth it: hipFile / BaM direct NVMe->GPU (absent or
    NVIDIA-only; VRAM fills from RAM anyway), moving models to ext4/xfs (no documented read gain), ntfs-3g (FUSE, avoid).
Added 2026-11-09 from the GPU-sync researcher (data-path round):
Correction: on edit windows the host "GPU-reach wait" (15.75 ms) and CPU experts (17.41 ms) add up; the VRAM-hit expert kernels
run hidden behind the CPU, but kernels BEFORE the router doorbell (GDN, attention, hyper-connection, router, shared expert) are
on the critical path on short and edit windows.  combine_cpu min 5.6 us vs 254 us average: >=97.8% of its 38% is waiting.
46. Single poller in combine_cpu: with STRATA_WAIT_FUSE every block (~50/window) polls the same host flag over PCIe while the
    pool uses ~27 of 33 GB/s DDR4; block 0 polls and sets a VRAM flag for the rest.  Exact, ~20 lines; A/B WAIT_FUSE 1 vs 0 on
    the host "CPU" time, not only tok/s (0-3% est.).
47. Warm the 80 MiB Infinity Cache with the next layer's pre-doorbell weights during the combine_cpu wait (PRESERVE, Huawei
    Zurich).  Micro-test first (one layer's pre-doorbell kernels cold vs warm; build only if warm/cold < 0.85); 0-2 ms/window est.
48. Item 32 re-rated up: fuse the dependent pre-doorbell chains (hyper-connection norm/down/up, router+plan+bell) - on the
    critical path in both window types, ~1-4% per chain.  Q2_0 down kernel (5.8%) is hit work hidden on edits: get the per-layer
    hit-vs-CPU split before building it.
49. Timer inside combine_cpu (thread 0 entry/exit stamps like wait_flag_ge_stamped): with WAIT_FUSE on, "waitCPU" no longer
    measures the real wait; gives the per-layer split items 1, 32, 48 need.
Low value / rejected: hipStreamWaitValue32 or host callbacks instead of the spin, CPU writing rows into VRAM via ReBAR (<=0.27 ms),
megakernels (NVIDIA-only; GDN megakernel 0.0 ms), HW queues 4 vs 6 (re-check only with a concurrent stream; a CU-masked stream
could isolate spin/prefetch kernels).  PreScope = candidate policy for the item 14 simulator.
Added 2026-11-09 from the PCIe/transfers researcher (data-path round):
Finding: with --pcie-frac 0 the transfer on the critical path is not the expert upload but combine_cpu_k10_vec4
(native_moe.cu:145-200) pulling a full fp32 row (2560 floats, 10 KB) per missed (token, expert) slot from mapped host memory
(h_ymiss_, verify.cpp:516-520/1017) - est. 0.4-0.8 ms per 37.4 ms edit window.  Host path: root port Gen3 x16, ReBAR on (32 GB
BAR0), IOMMU passthrough, THP madvise; ROCm 10 has no hipMemcpyBatchAsync (STRATA_DMA_BATCH CUDA-only).
50. Measure first: combine kernel with the flag pre-set (spin vs row pull) and promotion bandwidth with WAIT_FUSE on/off; if the
    pull is < ~3 us/layer, items 51-52 are worth ~0.
51. CPU pre-sums the missed rows per token (one weighted row instead of up to 10; fp32, bf16 optional): 2-4x fewer pull bytes,
    0.5-1.5% on edits est.; fp32 summation order changes (not bit-identical).
52. CPU pushes rows + flag into fine-grained VRAM through ReBAR (write-combining stores), GPU spins locally: 1-2% est., overlaps
    item 51; risks: WC ordering, ROCm 10 support on gfx1100.  (Item 46 single poller covers the per-block host polling.)
53. Online swap budget (Pie, UC Berkeley): raise adapt swaps while window time stays flat instead of fixed --adapt-swaps
    64/96/160 (0-5% over the best fixed); chunked ~256 KB promotions issued only during CPU compute and stale entries dropped
    from ForesightSwap::fill_loop (FIFO, no staleness check) (MoE-Lightning paging).
54. IOMMU guardrail: warn at startup if the GPU IOMMU group stops being passthrough (Fast & Safe IO Memory Protection, SOSP'24).
    (Faster restart = item 38.)
Not worth building (PCIe): batch-copy API (absent; promotions already ~93% of link: 115 vs 107 us), wire compression of Q2_0
(no FP-exponent redundancy at 2 bits; a 5-minute entropy check would settle it), zero-copy reads of missed experts at Gen3
(CPU path 2.5x cheaper), GPU-initiated transfers (BaM, SCADA, AMD DMA prelaunch not exposed), FloE lossy transfer (belongs
with lossy F1).
Added 2026-11-09 from the Intel platform researcher (platform round; matrix in research_notes/Consumer platform capabilities/):
55. Startup calibration of the CPU-vs-promote ratio: measure CPU us per expert and promote us per expert at start and set the
    promotion reuse threshold / --pcie-frac from it (DDR5 desktops ~12-18 us CPU vs ~58 us promote on a PCIe 4.0 card in any Gen5
    slot; Gen5 GPU on Arrow Lake ~1:2; today 42 vs 115 us) instead of one hand-tuned value.  Common work, all platforms.
56. Hybrid P/E cores: default pool affinity is All (E-cores included) while gathered decode runs on P-cores only (CPUID 0x1A),
    so equal task splits make P-cores wait at barriers; make --pool-affinity auto the default on hybrid CPUs or weight the split
    by core class; benchmark p-cores (7 workers, no SMT) vs auto on Arrow Lake.  Planner test already shows 12-16 > 19 workers.
57. ISA dispatch gaps: cpu_avx512_ok() requires F+BW+VL+VNNI+VBMI, so Skylake-X/Cascade Lake HEDT fall back to AVX2 (consider an
    AVX-512 path without VBMI); STRATA_FORCE_ISA accepts only avx2/avx/sse (add avx512 / avx-vnni for A/B); old Q2_0 model
    folders that need AVX-512 run on Intel desktops only on Rocket Lake (document or add an AVX2 path).
Not adopted (Intel): AMX expert kernels (only Sapphire Rapids W; no pancake path; evidence is a vendor-adjacent blog), Intel
iGPU/NPU offload (integrated Arc untested, no shared-memory planning, DPAS kernels need Xe2, no NPU code), NVMe through DMI on
LGA1151/1200 (3.9 GB/s chipset link: keep models on CPU-attached NVMe where possible).
Added 2026-11-09 from the AMD platform researcher (platform round):
58. CCD-aware CPU pool on dual-CCD Zen 4/5: each CCD's Infinity Fabric read link caps at ~64 GB/s, so a single-CCD part gains only
    ~1.2-1.3x over DDR4 for CPU experts; spread pool workers across both CCDs (affinity rule next to item 56).
59. AVX-512 for verify windows: the DeepSeek CPU path measured little at 1 token but 2.8-4.8x at 16 tokens on Zen 4/5
    (ds4_gg_cpu.cpp comments); check that the Qwen expert AVX-512 row kernels exploit multi-token verify windows the same way,
    measure on Zen 4/5 (cpu_avx512_ok passes on every Zen 4/5; AM4 stays AVX2).
60. Unified-memory preset for Strix Halo (gfx1151, ROCm-supported, up to 128 GB, ~103 GB/s per CCD measured): no PCIe
    promotion, shared-DRAM bandwidth is the limit, CPU latency rises under GPU load; existing unified path + gfx1151 Docker build.
Upgrade facts (no code): the 5700G forces PCIe 3.0 on the whole platform; a non-APU Zen 3 on B550/X570 gives PCIe 4.0 x16 to the
GPU and 4.0 x4 to the CPU M.2 (B450/X470 stay 3.0); the board chipset is not identified yet (sudo dmidecode -t baseboard).
Promote est. ~58 us at Gen4, ~30 us at Gen5 vs ~42 us CPU (arithmetic, not measured) -> item 55 calibration matters.
Not adopted (AMD): 8000G APUs as hosts (GPU slot cut to 4.0 x8/x4), X3D cache for expert weights (unproven; L3 prefetch was
worse here), Vega/780M iGPUs (not in ROCm matrices).
Added 2026-11-09 from the consumer-GPU researcher (platform round):
61. VRAM-size-aware defaults: from the 20 GB data point (~4.7 GB non-expert, 10,909 of 24,576 pairs) estimate pairs per card
    (8 GB ~10%, 12 GB ~21%, 16 GB ~33%, 24 GB ~56%, 32 GB ~79%, 48 GB all) and pick tier size / pcie-frac / prefetch defaults by
    card class instead of one preset; x8 cards (RX 7600, RTX 4060 Ti/5060 Ti, Arc B580) on a PCIe 3.0 host (~6.5 GB/s) force
    --pcie-frac 0.  Common work, ties into item 55 calibration.
62. gfx12 (RDNA4) re-measure: pancake *-hip presets and the Strata-rdna3 ports are gfx11-tuned; gfx12 has its own WMMA paths and
    defaults (pcie-frac 0.15 after upstream measured +9.9%/+25.6% at 8K/32K on R9700).  Only when a gfx12 card is available.
Upgrade facts (no code): 7900 XTX 24 GB runs the same gfx1100 binary/presets (+~2,900 pairs); R9700 32 GB +~8,600 pairs but needs
the gfx12 build; RX 9070 XT (640 GB/s, 16 GB) is a step down for VRAM experts; RTX 5090 strongest and enables strata-poly (CUDA).
Promote break-even ~33 GB/s real link: PCIe 4.0 x16 measured ~28 GB/s just short; only Gen5 x16 card + Gen5 host passes on paper
(no local Gen5 measurement) - sweep --pcie-frac on such a host rather than assume.
Not adopted (GPU): RDNA4 FP8/BF8 WMMA and 4:2 sparse SWMMAC (no low-bit FP8 experts here; Flash-Next experts measured dense),
RDNA2 support (no WMMA, no iu8 dot, untested package, not in the ROCm list).
Agent-harness items from the same reports (opencode, outside the engine): low/medium default effort with escalation on a new
external signal, ACE-style delta compaction / Slipstream async compaction in tool-idle windows, Recursive-LM tool for huge
inputs, AsyncFC, GEPA on the compaction prompt.  Quality gates for lossy items: token forcing -> BFCL v4 multi-turn +
LiveCodeBench -> HELMET RAG/re-rank -> Terminal-Bench subset.
(Item 8, VERIFY_DEFER_CPU on HIP, is the highest-ceiling lossy option per both research reports: up to 1.45x in KTransformers.)
Blocked: the 3060 GPU run (card out), two-stage prompts (second GPU), DeepSeek V4.1 on gfx1100 (257 GB, D too small).
Owner's call: merging port/faststrata-poly -> exp/expert-prefetch -> feat/grammar -> port/rdna3-decode into main; BIOS iGPU
memory 4 -> 2 GB; DDR4-3600 at FCLK 1800.

### Not adopted (ledger, 2026-11-09)

Everything the research series and the measurement rounds looked at and did not take, with the reason. [closed] = decided; [parked: trigger] = reopen when the trigger happens. Sources: MP = this plan (round or item), Up = Avalon upgrades report, Stan / MIT / SUA / EU / Asia / Can / Pac = the university reports, PP = pain points, TG = technology families, HW-cpu / HW-gpu / HW-nvme / HW-pcie / HW-pipe = data-path notes.

#### (a) Measured on this box: no gain, or worse
- `--ple-io mmap` (MP 8, 2026-11-08) - long prompt 42.9 -> 40.4 s but decode not better; PLE stays on direct reads. [closed]
- `STRATA_PF_WMMA` prompt expert kernel (MP 8) - slower on both 5.2K and 41.8K prompts. [closed]
- `STRATA_PF_FUSED` int8 prompt kernels as default (MP 8) - -7% at 42K but +12% at 5K; agent turns read short deltas. [parked: long cold prompts dominate -> chunk-size switch]
- CPU-thread RouterLookahead predictor (MP 10.2) - -13% decode (router GEMVs fight the pool for cores and DDR); replaced by the GPU lookahead. [closed]
- History-only expert predictors: prev-window, EMA frequency, transition dictionary, co-activation, hybrid (MP 10.2/10.3 offline) - cover 20-29% of misses at ~65% waste; the lookahead is ~2x better. [closed]
- MTP-head router predictor, 10.3 as its own path (MP 10.3) - the GPU lookahead already routes the draft tokens of the verify window. [closed]
- Foresight swap space fed by the lookahead, `STRATA_FS_SLOTS` (MP 10.2 step 2) - 0.97x at 4 slots; 8 slots run the 5.3K prompt out of VRAM. [parked: only from idle memory, if a lending source appears]
- 10.5 VRAM lending (MP 10.5) - `--kv-resident 8192` gives +5% slots, decode 68.5 vs 70.5 (noise). [closed]
- Item 7 per-layer cache budgets (MP 10, item 7) - cross-layer replay 48.3% vs 48.4% per-layer. [closed]
- 8.9 KV body (SJ-KVaRN keys 8-bit) and mb-MTP (MP 8.9 / status) - VRAM does not buy decode here; drafting is 2-5% of a window. [closed]
- Cache-aware routing at lambda 2.0 (MP 8.9) - 9.9% mismatch, 8 large-margin events: REJECT (0.5 stays the opt-in `turbo`). [closed]
- ClassLRU long-memory scoring (llama.cpp #31233, MP 11 index) - replayed decay 0.7..0.986 within 0.5 points at 32 swaps. [closed]
- Reuse-aware admission (MP 11 late) - a PCIe 3.0 promotion (~115 us) costs ~3 CPU computes of the same expert. [closed; k-th-miss admission goes through the item 14 simulator]
- L3 prefetch `STRATA_ARENA_WARM` (MP 11 late) - worse; DDR4 already at ~80%. [closed]
- CPU pool settings sweep (MP 11 late) - defaults best. [closed]
- IQ2_S two-half decode `STRATA_IQ2S_PAIR` (MP 11 late) - neutral; the kernels are compute-bound. [closed]
- `--spec 3/5/6`, window thresholds (MP 11 round A / late) - flat; `--spec 4` is also the official 4.07-accepted depth. [closed]
- Upstream group-A switches all together (MP 11 round A) - 101.2 vs 111.7. [parked: one at a time in the item 1 GPU work]
- Round A worse/flat vs 111.7 base (MP 11 round A): `ADAPT_LAG` 87, upstream `HC_Q8` 107.2, rdna3 expert profile 105.6 and adapt settings 116.4 (own 121.2), `--kv-resident 131072` 106.2, `STRATA_SPEC_COST` 112.0/98.3, `SELECT_LONG` and mtp-window flat. [closed]
- zh60k draft vocab; full / en vocab (MP 11 round A, qwen38-megakernel idea) - 92.5 worse; full and en flat. [closed]
- mmap'd expert tier as default (MP 11 round A; status) - 108.6 vs 111.7 (and 69 vs 75 on the 3060). [parked: RAM policy (RAM_KEEP vs oomd) decision]
- HW queues 4 vs 6, `GPU_MAX_HW_QUEUES` (MP 11 round A, HW-gpu) - flat. [parked: a concurrent stream is added (MALL warm, CU-masked spinners)]
- DFlash2 (MP 11 round A, Up) - 115.9 short but 84.3 after a long prompt; upstream crash on QFN. [closed]
- RDNA_PAIR (MP 11 not ported) - neutral, off upstream. [closed]
- GDN decode megakernel (Strata-rdna3, MP 11) - 0.0 ms on one GPU, the window waits on CPU rows. [parked: second GPU / tensor parallel]
- DraftExpert-style miss-aware windows (MP 11 index) - tested: DraftPolicy already optimal. [closed]
- Item 13 activation sparsity: Berkeley gate mask, TEAL, CATS (+ CHESS / SpAx / per-layer thresholds / union sparsity as add-ons) (MP item 13 result) - gate cos 0.844 at 70% (oracle 0.954); FF-640 experts are dense; no AVX2 gather kernel. [closed]
- ContextPilot block dedup, `feat/contextpilot` (MP 11, EU, PP) - 1.6% savings on 40 real sessions, and non-prefix reuse breaks GDN state. [closed]
- HIP graphs, global unroll flag, COMPUTE power profile, draft p-min, checkpoint-count tuning, AMD-Ecosystem fork, Vulkan, turbo4/TurboQuant KV (Up skip list, TG) - tested with no gain. [parked: only with a new reason, e.g. a profile showing dominant inter-kernel gaps]
- `--pcie-frac` > 0 / zero-copy misses (pcie_mode 1) (MP round A, HW-pcie) - `--pcie-frac 0` measured best (+6/+13%); a 1.4 MB GEMV over Gen3 is >=107 us vs 42 us on the CPU. [parked: F1 frees DDR4 headroom]
- Strata gfx1100 switches: int8 WMMA attention, TSUM, expert layouts (Up skip) - neutral or negative on 7900-class cards in Strata's own tests. [closed]
- `VERIFY_DEFER_CPU` on HIP (MP item 8) - still non-finite logits with the ported kernels; not rejected, blocked. [parked: 6-run bisect]

#### (b) Needs retraining, fine-tuning or a BF16 requant of the 125B model
- Kimi Linear, MoBA, Kimi K2 arch, Jet-Nemotron, Stick-breaking (Asia, MIT) - architectures the shipped weights do not have. [closed]
- PaTH, Log-Linear attention, KL-guided hybrid distillation (MIT) - retraining. [closed]
- LoLCATs, Hedgehog linearization (Stan) - training; impossible locally for 125B. [closed]
- Pre-gated MoE, SiDA-MoE, Read-ME (Asia, SUA) - trained gates/predictors or a modified model. [closed]
- ProSparse, TurboSparse, Sparsing Law ReLU-fication, BlockFFN (Asia) - need ReLU or pre-training; Flash-Next experts are SwiGLU. [closed]
- Deja Vu (Stan) - trained sparsity predictors; was superseded by CATS, which item 13 then closed. [closed]
- DMC, DMS KV compression (EU); FoX + ACP (Can) - need a model trained that way. [closed]
- NSA, InfLLM-V2 (Asia); Sparse Flash Attention, Landmark Attention, AdaSplash/ASEntmax (EU); DuoAttention, FlashMoBA (MIT) - training/optimization or entmax models. [closed]
- Cartridges (Stan) - backprop through the 125B model. [closed]
- Next-Concept Modeling, Quiet-STaR, SUPO, Think-Prune-Train-Improve (Stan) - specially trained models. [closed]
- PASTA, Co-LLM, ThinkPrune, SEAL / TTT for ARC (MIT) - fine-tuning or RL on the 125B. [closed]
- VeriThinker, LASER, TokenSkip (Asia); CodePromptZip (Can) - training. [closed]
- CLLMs, Jacobi Forcing (SUA) - fine-tuning for parallel decoding. [closed]
- EAGLE-3, GRIFFIN, SpecBlock separate drafters (Asia) - trained drafters; the native MTP head is better here. [closed]
- SLiM, PATCH, OPTIMA (Can) - recalibration on 125B, and GGUF has no 2:4 sparsity. [closed]
- Mixture of Pruners / USP pruning line (Pac) - dense pruning with recovery fine-tuning. [closed]
- Expert-sensitivity mixed-precision requant (MP 10) - needs the ~250 GB BF16 model; D cannot hold it. [parked: disk space for BF16]
- QuaRot (EU) - run path exists, the 354 GB BF16 does not fit. [parked: disk space]
- LQER, QERA (EU) - need BF16. [parked: disk space]
- GSQ re-quantization at home; GPTQ-2D, WUSH, QMoE, GPTVQ, FPTQuant, LR-QAT (EU) - requant/QAT impossible locally, and the daily files are already GSQ-RCO. [closed]
- R5 (Pac) - item 25 (Coder) already ships quantized. [parked: if a new base file is needed]

#### (c) Hardware or platform not available
- FP8 KV cache (MP 10) - no FP8 tensor cores on sm_86. [closed]
- FP8 Gated-Residual stream (Asia) - irrelevant at batch 1 on RDNA3; only the fusion carries over (item 32). [closed]
- FlashAttention-3 (SUA) - Hopper only. [closed]
- FlashInfer (SUA) - ROCm port is CDNA only. [closed]
- Mirage MPK, NanoFlow, Punica, Atom (SUA) - CUDA or batch. [closed]
- Whole-layer/model megakernels MPK, FlashDMoE (HW-gpu, MP 11 low value) - NVIDIA, GPU-only pipelines; WAIT_FUSE already is the in-kernel hand-off. [closed]
- Persistent post-bell kernel, FlashDMoE actor style (HW-gpu) - low: overlaps the CPU share in most layers. [parked: F1/F2 shrink CPU below hit time]
- ThunderKittens / HipKittens direct port, Look Ma No Bubbles megakernel (Stan) - NVIDIA or CDNA code; the lesson gives no new HIP-graphs reason. [closed]
- Hexcute, Tilus, K-Search (Can, SUA) - CUDA-only kernel tools. [closed]
- Flash-LLM, FP6-LLM (Pac) - A100 tensor cores; "load compressed, compute dense" is already in pancake. [closed]
- Marlin, QUIK, GPTQ/SparseGPT/AQLM/PV-Tuning/HIGGS kernels (EU) - CUDA-only or superseded by GSQ. [closed]
- Disaggregated Quantization (EU) - NVFP4, and prefill below 4K gets slower. [closed]
- BitDecoding as published (EU) - NVIDIA tensor cores; the HIP port stays opt-in. [parked: `feat/bitdecoding-hip` confirms its ~2-3%]
- llama.cpp #31802 fused-dequant MMA FA; #31533, #28714, #28195, #26301, #26592; #27140 (Up) - NVIDIA-only / RDNA3.5-only / CUDA bug, or no gain for 512-expert top-k. [closed]
- hipFile (AMD AIS / GDS) (HW-nvme, MP 45) - fastpath absent (no `kfd_ais_rw_file`, data on ntfs3); the RAM arena gains nothing. [parked: newer amdgpu AIS + ext4/xfs + Radeon support]
- BaM / GPU-initiated NVMe, NVIDIA SCADA, device-triggered SDMA promotion (HW-nvme, HW-pcie) - NVIDIA-only or no payoff: misses come from RAM at ~7x the NVMe rate. [closed]
- cache_ext eBPF page-cache eviction (HW-nvme) - not on the stock kernel. [closed]
- DECA, StreamDQ, TriMoE near-core / in-DRAM decompression (HW-cpu) - hardware absent. [closed]
- SparAMX (HW-cpu) - needs AMX and unstructured weight sparsity. [closed]
- LIA (HW-pipe) - AMX/CXL. [closed]
- MoNDE, Pimba, "SSD offloading harmful" line (Asia) - near-data/PIM hardware; NVMe only for capacity. [closed]
- `STRATA_POOL_UMWAIT` port to Linux (HW-cpu) - shared power budget only on laptops; desktop dGPU gains nothing. [closed]
- `hipMemcpyBatchAsync` batched promotion copies (HW-pcie) - absent in ROCm 10. [parked: ROCm adds it, and promotion volume rises]
- LP-GEMM (Pac) - AVX-512 + prepacked GEMM; ~0 for AVX2 GEMV. [closed]
- PLE-in-RAM (Up skip) - model + 28.8 GB table + 16 GB headroom > 57 GB RAM. [closed]
- Second-GPU work: 8.7 two-stage prompts, tensor parallel, layer split + pipelined windows, peer experts/RCCL, kfdq idle parking, P/D disaggregation, KV streaming, xLLM-style heterogeneous split (MP 8.7, 8.10, Phase 12) - one card. [parked: second GPU installed]
- DeepSeek V4.1 on gfx1100 (MP phase 5) - 239-274 GB, D too small. [parked: disk]

#### (d) Targets batching, multi-user or multi-GPU throughput; nothing at batch 1
- FlexGen (Stan) - ~0 at batch 1; only the placement cost model idea kept. [closed]
- Hydragen, Tokasaurus (Stan) - need many sequences with a shared prefix. [closed]
- NEO, Pie (serving), Jenga, PagedAttention, S-LoRA (SUA) - multi-user serving (Pie's online swap-budget idea is a separate HW-pcie candidate). [closed]
- MegaScale-Infer, Comet, UltraMem (Asia) - cluster, large batch. [closed]
- MiniMax-01/M1, LongCat-Flash, GLM-4.5, Pangu Pro MoE, ERNIE 4.5, Hunyuan-Large (Asia) - other models / context only. [closed]
- MoE-Gen, BatchGen (EU) - large batch. [closed]
- MoEShard (EU) - multi-GPU. [parked: second GPU]
- DSI (EU) - several target instances. [closed]
- KTH priority-aware scheduling (EU); DeltaZip (EU) - multi-tenant serving. [closed]
- SpecMoE (Asia) - gains at large batch; pancake's drafter is already in VRAM. [closed]
- ExpertFlow (Asia) - low at batch 1 (its pattern idea feeds items 3-4). [closed]
- OEA sibling-aware re-routing (MIT, TG) - measured at batch 16; small ceiling in a verify window. [parked: idea only]
- Klotski, MoE-Lightning pipelines, MoE-Lens scheduler (HW-pcie, HW-pipe, SUA) - multi-batch; only paging and the roofline tool carried over. [closed]
- PowerInfer-2 planner (HW-pipe) - phone NVMe; NVMe is off the decode path while the arena is resident. [parked: arena shrunk by RAM_KEEP]
- Vidur, LLMServingSim (HW-pipe) - cluster simulators; only their validation method is copied (item 42). [closed]
- PrefixShield, PrefixPlace, Gimbal (Pac) - multi-tenant, multi-GPU. [closed]
- HiveMind (Pac) - matters only with parallel sub-agents on a batch-1 server. [parked: parallel sub-agents]
- AMESA (Pac) - service-level mixture, not in-model MoE. [closed]
- Chunked-prefill interleave, overlap scheduler, PASCAL reasoning-aware scheduling, distributed KV stores (MP runtime architectures, Phase 12) - protect other users' decode. [parked: several users/agents at once]
- llama.cpp #30992 / #30970 / #29738 multi-slot scheduling (Up watch) - only with `-np 2`. [parked: second slot for title requests]
- RSA, Scorpio, RL^V (Can); Weaver, Archon (Stan); PRM calibration, Reasoning with Sampling (MIT) - N samples or many LLM calls per answer. [closed]
- Mooncake (Asia) - cluster KV store; only the NVMe-tier idea (phase 7 / item 39). [closed]
- SPRINT (Stan) - fine-tuning + batch > 1. [closed]

#### (e) Conflicts with exact-prefix cache, GDN recurrent state or determinism
- Draft trees: Staged SD, SpecInfer, Sequoia trees, SGLang `tree.cuh`, Hydra top-2 tree (Stan, SUA, MIT, MP phase 3) - each branch copies GDN state and wakes more experts. [parked: watch upstream #30931 only]
- Lookahead decoding, parallel Jacobi (SUA, EU) - wide guesses enlarge the expert union. [closed]
- Token Recycling (Asia) - longer drafts add new CPU-expert misses. [closed]
- SWIFT, CAS-Spec layer-skip self-speculation (Asia) - break the GDN state. [closed]
- Entropy Gate, Prompt Cache non-prefix reuse (Pac, SUA) - any change earlier in the prompt invalidates the GDN checkpoint. [closed]
- KV-streams (Can) - the idea is already the GDN checkpoint. [closed]
- Out-of-core streaming RFC #31469 / #25294 (Up watch) - conflicts with every cache. [closed]
- #31497 no MoE cache for the draft (Up) - contradicts the shared MTP cache. [closed]
- Upstream expert cache PRs #29887, #30637, #30966, #31888, #31286, #30594, #27861 (Up, MP 11 index) - pancake's tier + `--pcie-frac` already do this; #30594 hurts MTP verify. [closed]
- Grammar during thinking ("Let Me Speak Freely?", Asia, TG) - constraints stay after `</think>`. [closed]

#### (f) Lossy with poor quality evidence or no payoff here
- Router residency bias as a default (Up skip: `ROUTE_PRIOR` / `CACHE_ROUTE`) - Strata data shows loops in long agent sessions; pancake keeps 0.5 as opt-in `turbo` only. [parked: turbo-hip quality gates (item 7)]
- AdapMoE, BuddyMoE, MoBiLE, HOBBIT weight-based precision (Asia, TG) - variants of turbo; only one re-selection runs in `verify.cpp`. [parked: if turbo loses to them in the gates]
- DAOP (Asia) - lossy, CPU already saturated. [closed]
- MoE compression study (Pac) - ~8% quality loss is unacceptable for an agent. [closed]
- REAP-style rare-expert pruning (MIT, Mixture of Parrots) - drops API/library memory first. [closed]
- KVA approximate late-layer prefill (Up) - decode -11% and quality risk. [closed]
- StreamingLLM, DuoAttention streaming heads (MIT) - lose mid-context code and tool output. [closed]
- KV eviction/compression: H2O, PyramidKV, SnapKV, KVQuant, XQuant, Palu, xKV, ChunkKV, KVzip, InfiniGen, Q-Filters, TOVA, OSCAR, KV INT2, ZipCache, MiniCache, Counter-Causal Surprise, InnerQ, QServe KV4 (Stan, SUA, Asia, EU, Can, Pac, MIT) - KV lives in 1 of 4 layers, already q8; TurboQuant measured 0. [closed]
- Dynasor-style early exit (MP item 12) - needs a forked probe of the single stream; replaced by the learned stop (item 21). [closed]
- SliceGPT (EU) - adds loss on top of 2 bits. [closed]
- FloE-style lossy transfer (HW-pcie) - belongs to F1 and its gates; nothing transfer-specific. [closed]
- Cactus KL-budget acceptance (Can) - zero under greedy, ~0 at T=0.1. [parked: owner moves to higher temperature]
- Relaxed drafting (basalt, MP phase 3) - draft step is 2-5% of a window. [parked: idea only]
- YaRN static factor 4 (Asia) - may hurt short texts; 262K native is enough. [parked: contexts > 262K]

#### (g) Superseded by something already in pancake, or by a better candidate
- CATS (Stan) - superseded by TEAL / Berkeley gate mask, then item 13 closed all three. [closed]
- XAttention (MIT, item 19) - superseded by BFLA (benchmarks Flash-Next itself). [closed]
- IQ2_XS as daily file, item 6 (EU, MP) - Q2_0 chosen (GSQ-RCO, 93.8 vs 70.3 tok/s, -0.09 task avg). [closed]
- IQ1_M native AVX2 row kernel, item 24 (EU) - 0 on Q2_0. [closed]
- T-MAC, Vec-LUT, bit-serial LUT GEMV (Asia, HW-cpu) - Q2_0 kernel at ~80% of DDR4, row_multi already amortizes. [parked: micro-benchmark shows compute-bound]
- Any-Precision, MC-MoE, MxMoE (Asia) - no bits left below Q2_0; Any-Precision for draft passes n/a (MTP keeps all experts in VRAM). [parked: a >2-bit file such as Coder ~3.5 bpw]
- QTIP / IQ*_KT, CALDERA / LPLR (SUA, Stan) - no HIP kernel, slow on CPU, ~0.2 bpw gain; new tensor types. [closed]
- SmoothQuant, AWQ, QServe W4A8, SVDQuant, SqueezeLLM (MIT, SUA) - throughput/4-8 bit; imatrix + IQ codebooks + q8 activations already cover it. [closed]
- Entropy-coded weights EntroLLM / Huff-LLM; Q2_0 wire compression (HW-cpu, HW-pcie) - 2.25 bpw leaves little; CPU decode costs more than it saves. [parked: entropy check shows >= 20%]
- Non-temporal loads/stores, extra cache blocking (HW-cpu) - WB arena, no cross-window L3 reuse. [closed]
- HybriMoE, DuoServe-MoE, Fiddler, CPU-GPU collaborative MoE (Asia, Pac, SUA) - covered by `--pcie-frac` / PcieAuto / item 30. [closed]
- FreeToken PCIe/CPU split, item 22 (SUA) - very likely the same as `--pcie-frac`. [parked: code read shows copied experts are also promoted]
- DALI greedy per-expert placement, PreScope/HybriMoE event placement (HW-pipe, HW-gpu) - "GPU if t_gpu < t_cpu" almost never fires; simulator candidate only (<=1-2%). [closed as engine work]
- Draft-driven expert lookahead MoE-SpeQ / SP-MoE as prefetch (HW-pipe) - ~0 on PCIe 3.0; recall logged only. [closed]
- Speculative linear pre-compute of next-layer projections (HW-gpu) - doubles weight reads; MALL warm-up (item 47) captures the benefit. [closed]
- GPU work stealing of CPU misses (HW-gpu) - same DDR4; `--pcie-frac 0` won. [parked: F1 frees DDR4 headroom]
- Full/empty bits, per-row readiness (HW-gpu) - already the partial flag protocol. [closed]
- Spin backoff changes, wave priority `s_setprio` for spinners (HW-gpu) - detection is PCIe-bound; spinners sleep already. [closed]
- `hipStreamWaitValue32`, host callbacks / blocking events instead of the spin (HW-gpu, MP low value) - ~0 alone, OS wake latency. [parked: concurrent GPU work on another queue]
- CPU writing rows into VRAM via ReBAR (MP low value) - <=0.27 ms. [parked: combine pull measurement (item 49) > 3 us/layer]
- `raise_flag_b_ordered` -> `hipStreamWriteValue32` (HW-pcie) - only matters with `--pcie-frac` > 0. [parked: pcie-frac back above 0]
- 8.5 adaptive MTP depth (llamAmpere) and 8.8 fused FA prompt kernel (MP 8) - poly already has the cost rule; attention 6.7% of a 42K prompt. [closed for main engine]
- Basalt draft-head screen, K/V append beside the draft query (MP phase 3) - draft step 2.5-3.2 ms of 46-130 ms: 1-2% max. [closed]
- TurboSpec, Sequoia length choice, Speculative Decoding + Batching, Su-Pekhimenko adaptive length, TLT, DISCO-as-new (SUA, Can, MIT, EU) - = DraftPolicy. [closed]
- EAGLE / EAGLE-2 / Hydra / Cassandra (Can, MIT, Asia) - = native MTP + `--spec-min-p`. [closed]
- BanditSpec UCB in DraftPolicy (HW-pipe) - DraftPolicy is already a greedy bandit; 0-2%. [parked: logs show bad exploration]
- MTP head Q4_K requant (tandem.cpp) and `MTP_DEFER` (MP 11) - ours is ~2-bit (787 MB); MTP_DEFER already here. [closed]
- Upstream `PF_GROUP_RES`, long-context top-k, `PA_KLDS` from Strata-rdna3 (MP 11) - = upstream group_resident / SELECT_LONG / PA_RPF. [closed]
- #31750 KV-only catch-up, #31037/#31031 trimmed draft head, #31669 GDN replay, #27489 buffer sharing (Up) - already present, or "frees VRAM" which buys nothing here. [closed]
- Maya resume-thinking after an end-of-turn token (MP Maya) - both turn-end tokens are EOS; empty answer handled by #1053. [closed]
- Dense 4.x tier / Qwen 4 27B through expert offload (MP 9) - offload does nothing for a dense model. [parked: if it ships open, choose a dense engine]
- MEM1, Self-Steering/DisCIPL, Graph of Thoughts, KGoT, RAPTOR, Generative Agents, STOP, ReAct/ToT/MemGPT/Gorilla/RAFT/TinyAgent/Autellix (MIT, EU, Stan, SUA) - conceptual or already in opencode. [closed]
- Local-Splitter (Pac) - pancake is the local model; minimal edits are already `edit`. [closed]
- MoESQ presence-penalty setting (EU) - kept as a rule (`presence_penalty=0`), the model itself not run. [closed]

#### (h) Gain too small for the effort / parked until a measurement says otherwise
- 10.4 per-layer PcieAuto refinement (MP 10.4) - share 0.20-0.25. [closed]
- 8.8 fused attention prompt kernel on main (MP status) - attention 6.7% of a 42K prompt. [parked: prefill profile on the 7900 XT says otherwise]
- F7 prefill work: GDN chunk 64 (item 18), BFLA beyond prototype (TG skip list) - prefill unmeasured on the 7900 XT. [parked: run 2b prefill timing]
- Quest, TriForce, MagicDec, MagicPIG, TidalDecode, The Sparse Frontier decode sparsity (MIT, SUA, EU) - decode attention is small. [parked: attention > ~15% of decode at 100-200K]
- Exact QSA scorer (Up) - only above 131K. [parked: contexts > 131K]
- Calibrated verify gate `STRATA_DF2_GATE` (Up watch) - no numbers. [parked]
- Expert predictors as a build (item 30), prefetch on top (TG skip list) - history 20-29%, ProMoE 1.09x even on PCIe 4.0. [parked: recall@16 >= 80% in trace run 2a]
- Cache policies beyond k-th-miss admission: SIEVE/S3-FIFO, OpenEvolve, Hawkeye, Harmonia, MoE-Infinity, FlashMoE (+7% vs LRU) (SUA, EU, Can, TG) - Belady-bounded, one runs at a time. [parked: item 14 simulator shows >= 5%]
- Item 26 semantic `oldString` constraint (EU, TG) - one failure in 180 edits. [parked: failure rate rises]
- `feat/bitdecoding-hip` as default (TG) - attention ~3% of decode. [parked: ~2-3% confirmed]
- Staged-SD small tree via #30931 (Stan) - large effort, GDN fork. [parked: upstream lands it]
- SSD-lite (Saguaro) precomputed drafts (Stan, TG) - CPU already busy. [parked: host-gap profile between verify steps]
- Typical acceptance on thinking only (Medusa, SUA) - low priority. [parked]
- Draft source as a bandit (Mila, Can) - only if DraftPolicy logs show wrong choices. [parked]
- Athena epoch bandit for pre-read (EU) - only if the `POOL_PREREAD` A/B is mixed. [parked]
- Delethink resets (Can) - only with > 16K-token thinking turns. [parked]
- AsyncFC / Slipstream in tool windows vs expert pre-staging (TG) - one GPU slot; depends on tool durations. [parked: phase 14 timeline]
- io_uring backend for DirectFile (HW-nvme) - ~1.06x drop-in; 16 threads already give QD16. [parked: PLE waits on the critical path]
- Frequency-ordered PLE shard layout (HW-nvme) - size of gain unknown. [parked: PLE wait time measured]
- ext4/xfs migration, kernel 7.1 "ntfs" driver (HW-nvme, MP 45) - no documented read gain; ntfs-3g (FUSE) avoided. [parked: only for hipFile]
- MADV_POPULATE_WRITE prefault, extra THP work (HW-cpu) - load-time only, low single digits. [parked: arena load measured after item 38]
- Pinning cost via hugetlbfs for `hipHostRegister` (HW-pcie) - unmeasured part of ~10 s setup. [parked: setup timing]
- Coalesced promotion copies, `STRATA_COPY_HIP_PRIO` default (HW-pcie) - 0-1%, likely flat at pcie-frac 0. [parked: one-line A/B]
- MonoMoE weight-major layout for VRAM-hit experts / Q2_0 down (HW-gpu) - hit kernels hidden behind the CPU. [parked: per-layer critical path (item 49) shows hits > CPU]
- Expert-level `MADV_SEQUENTIAL`/`WILLNEED` (Up #5561) - only if experts fault from NVMe; the arena removes faults. [parked: major-faults per turn > 0 on mmap tier]
- Disk prompt cache #28092 (Up) - pancake has session files; automatic tier is item 39. [superseded by item 39]
- Sleep-in-place (basalt 64167ba73, MP phase 7) - only if the GPU is wanted for other work while idle. [parked]
- Upstream #31625 HIP fast-math, #28118 on-device spec checkpoints, #27210/#27173 deeper MTP, #31271 effort discovery, #30689 engine split (Up watch) - no numbers / deep drafts touch more experts / rebase hazard. [parked]
- SpecCache, Cartograph, AutoCompact, ArbiGraph, MMLU-Pro tier (Can, Pac, Asia) - local tools are fast / small tool set / minor. [parked: large MCP catalog]
- JRT (Stan) - doubles prefill. [closed]
- MegaBlocks (Stan) - training-oriented; `mul_mat_id` covers batch 1. [closed]
- Flash-Decoding split heuristics (Stan) - already in the vec kernel; at most a retune for 84 CUs. [parked: item 37 sweep]
- Markovian Thinker, Flash Inference, Kinetics, Critical Thinking (Can, MIT, SUA) - very long thinking only / convolutions / context. [closed]

### Agent robustness from Project Maya (MIT, mw00/project-maya main 7bf15ac, v1.1.56; 2026-11-09)

Server-side only (no engine change, no speed effect until a loop is seen).  Copied and adapted, tests in
`serve/test_loop_guard.py` (Maya's module tests as they are + pancake server tests) and `serve/test_tolerant_tool_calls.py`.

- **Loop guard** (`serve/loop_guard.py`, on by default; `STRATA_LOOP_GUARD=0` / `MAYA_LOOP_GUARD=0`, config
  `"loop_guard": false`, or a request's `"loop_guard": false|true`).  In the token loop: an exact block repeated back to
  back is seen as it starts; the stream holds back at most 64 tokens of a tail that repeats; 1st loop in the answer
  (1st/2nd in the thinking): the repeats not sent are dropped and the engine continues with repetition penalty 1.15 over
  2 x period (GEN `pen_from` so the window counts from the reply's start); 3rd loop in the thinking closes it with
  `</think>`; a loop that comes back in the answer ends it ("stop", a note if no answer text yet).  Tool calls are never
  cut.  Across turns: a note after a tool result repeated 3x with the same result, the call blocked (answered by the
  server in the prompt, thinking steered, slightly hotter) after 3 notes, "keep going" when a looping conversation would
  end the turn with nothing new (3x max), and the prompt of a looping conversation under 32K chars leaves out what the
  model already wrote.  Counters: `/metrics` JSON `loop_guard`, Prometheus `strata:loop_guard_*_total`.
  Precedence: one-token runs stay #606's (`repeat_stop_tokens`, "length"); thinking loops are #728's when
  `reasoning_loop_recovery` is configured; the guard handles everything else.  DeepSeek (DSML) gets notes but no blocking.
- **Soft thinking budget** (Maya #613), opt-in `"reasoning_budget_grace": N` (Maya uses 512): past the #123 budget the
  thinking closes at its first sentence or line end, at most N tokens (and half the remaining room) later.  Default 0
  keeps the exact #123/#984/DeepSeek small-cap close positions the tests pin.
- **Tolerant tool calls** (Maya #613/#706): an unclosed `<parameter=` tag is malformed (pancake waited for a
  `</parameter>` that never came and lost every word after a streamed call); a malformed call never raises (also with
  `"tool_call_recovery": false`): announced -> left unfinished (#211), else text.  Not ported: Maya's resume-the-thinking
  after an end-of-turn token inside it - both Qwen turn-end tokens are the engine's own EOS here, and the empty-answer case
  is already #1053 (`reasoning_close_retry`).

### Items 3-4: seeding the adaptive tier at a request's start (branch feat/pattern-cache, 2026-11-08)

Why seeding and not longer memory: a replay showed the tier's hit rate within 0.5 points for usage decay 0.7..0.986 a
window at 32 swaps, so the gain has to come from putting the right experts in early.  Placement only: the GPU and the
CPU compute an expert the same way, so no output changes.  Both opt-in, default off (the RTX 3060 line is untouched).

- Engine: include/strata/program/expert_seed.hpp (header-only, tests/program/expert_seed_test.cpp), hooked in the
  --serve loop of src/program/generate.cpp (blocking and asynchronous tier; not the batch slots or the pipelined loop).
  STRATA_DOMAIN_PROFILES=<dir> loads <dir>/<name>.bin; a GEN line's domain=<name> picks one; its top M pairs (M = cache
  slots) score S * (1 - rank / M) and stay a floor under the counts for R rounds (STRATA_SEED_SCALE=2, STRATA_SEED_HOLD=16).
  STRATA_CONV_WORKSET=N (1 = 8) keeps the outgoing conversation's routing counts at every conversation switch (keyed by its
  tokens: works with or without the parked K/V) and adds them back, scaled to STRATA_CONV_WORKSET_TOKENS=32 tokens'
  worth, when a prompt continues a kept conversation.  A continuing turn is never seeded; ckpt=0 side requests are skipped.
- Server: serve/server.py classify_domain (code fences / diffs / code-shaped lines -> code; tools offered or a tool turn
  -> tool; else prose), sent as domain=<name> when STRATA_DOMAIN_PROFILES is in the server's environment
  (serve/test_domain_classify.py).
- Tools: tools/pancake/learn_domain_profiles.py (per-domain profiles from collect_profile_traces.py's corpus traces and/or
  DOMAIN=TRACE specs), tools/pancake/replay_seeding.py (offline replay of both, incl. --requests for conv_trace.py runs),
  tools/pancake/conv_trace.py (interleaved multi-turn conversations with --dump-routing; also the live A/B driver).
- Offline (bench/prefetch/routing.bin, 10,909 slots, 32 swaps every window): plain 81.17%; domain leave-one-out (code)
  82.86% (code edit 77.3 -> 78.8, long 84.7 -> 90.7) but with the prose request seeded in-sample first the held-out edit
  gets 77.1; oracle domain (own counts) 84.03%.  Conversation resumed after another request: +0.7..+1.0 points on its
  rest (bound +1.4..+2.7).  At 160 swaps the domain seed gains 0.5.  One trace of four requests: not enough to trust the
  domain part; the conversation part never went negative.
- Next (do not run beside a benchmark): build, collect traces, replay, then the live A/B:

      cmake --build build-hip -j16 --target strata expert_seed_test && build-hip/expert_seed_test
      cmake --build build-cuda -j16 --target strata          # must still compile (the 3060 line)
      sed -e "s#<STRATA_DATA>#/home/dan/clone-of-d/Strata-data#g" -e "s#<REPO>#$PWD#g" -e "s#/build-cuda/#/build-hip/#" \
          configs/pancake/flashnext-iq2_xs-fast-hip.json > pancake-flashnext-iq2_xs-fast-hip.json   # (what run-pancake.sh writes)
      export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib HIP_VISIBLE_DEVICES=0
      .venv/bin/python tools/collect_profile_traces.py pancake-flashnext-iq2_xs-fast-hip.json bench/domain-traces --mode routing
      .venv/bin/python tools/pancake/conv_trace.py pancake-flashnext-iq2_xs-fast-hip.json bench/conv-trace --tag base
      .venv/bin/python tools/pancake/learn_domain_profiles.py --out data/domain-profiles --traces bench/domain-traces --slots 10909
      python3 tools/pancake/replay_seeding.py bench/conv-trace/base.trace data/expert-profile-pancake.bin 10909 \
          --requests bench/conv-trace/base.requests.json --domain-dir data/domain-profiles
      STRATA_CONV_WORKSET=1 STRATA_DOMAIN_PROFILES=$PWD/data/domain-profiles \
          .venv/bin/python tools/pancake/conv_trace.py pancake-flashnext-iq2_xs-fast-hip.json bench/conv-trace --tag seeded --domain-keys

  (compare base vs seeded decode_tok_s per request in <tag>.requests.json; the corpus traces are short decode-only
  replies, so add code-edit traces with long context, e.g. bench_engines.py runs with --dump-routing, as code=TRACE.)

## Phase 13: priority by measured gain (2026-11-09, owner: biggest gains first, then the rest)

Where a decode window's time goes on the 7900 XT (fast-hip, measured 2026-11-08/09):

| Request | Window | GPU | CPU experts | Draft | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| short chat | 24.8 ms | 15.9 | 5.2 | 2.2 | the GPU spends ~42% of its kernel time waiting for CPU rows (rocprofv3, 2026-11-09) |
| code edit (2K prompt) | 37.4 ms | 15.8 | 17.4 | 2.5 | CPU experts are 47% of the window |

Prompt reading: 1,850-1,920 tok/s at 11.7K with the matching hipBLASLt table (1,345 without it).  In agent use a turn is
prompt re-read + thinking tokens + answer tokens, so per-turn cache and thinking length weigh as much as tok/s.

Tier P1 - the CPU expert time (the largest block; every ms is a ms of the window):
| # | Item | Lever | Expected | Exact? |
| --- | --- | --- | --- | --- |
| P1a | 8 + 28: expert deferral on HIP (bisect queued), then the rank guard | overlap CPU experts with the next GPU layer | up to 1.45x (KTransformers); realistic +10-25% on code edits | lossy-small (rank guard) |
| P1b | 14/27/30: time-based cache simulator (PCIe 3.0 cost per promotion), candidate policies, predictor recall | decide whether misses 20.4% -> ~10% (LRU/Belady band) can be bought without paying it back in PCIe | 0-30% on code edits (LRU-band hit rate would halve CPU work); build only at >= 5% | exact |
| P1c | 13: offline sparsity masks (Berkeley gate mask first) | skip 50-85% of an expert's rows on the CPU | +10-20% (ceiling 1.25x) | lossy, quality gated |
| P1d | 29: one phase per layer in the CPU pool (coded, opt-in STRATA_POOL_FUSE_PHASES) | 96 fewer park/wake barriers per window | 1-4% | exact |
| P1e | 3-4: domain / conversation seeds (merged) | warm the right experts at a request start | +0.6-1.5 hit points | exact |

Tier P2 - tokens per window (drafting): 174 of 231 drafts accepted in the profiled run.
| P2a | 7: turbo-hip decision (ROUTE_PRIOR 0.5, 111.7 -> 149-168 tok/s measured) | quality gates decide default-for-some-modes | +30-50% | lossy |
| P2b | 15/36/23: DISCO / SpecFit / AcceptMoE on top of MTP | longer accepted runs, fewer missed experts per draft | 0-15% | mostly exact |
| P2c | stack of small exact flags (measured +1.4..+3.9%) and --lookup-max-t 12 (+1%, retest) | | +2-5% | exact |

Tier P3 - the per-turn cost in agent use (seconds per turn, not ms per token):
| P3a | 35 -> 31/16/11: restore-equivalence test, preserve_thinking, checkpoints at tool boundaries, turn-boundary checkpoint | no re-read of history already processed | seconds per turn; author: first edit -33% | exact |
| P3b | 2: keep the decode cache through long prompts (lend sweep queued) | decode after an 11.7K prompt ~97 -> ~120 tok/s | +20% after long prompts | exact |
| P3c | 12/21/33/34: effort defaults and learned thinking stop (other session) | 20-40% fewer thinking tokens on routine turns | tens of seconds per turn | changes text |

Tier P4 - the GPU floor (15.9 ms): kernel fusion (395 kernels per token, ~5 us gaps), host stalls at window edges,
32/37 parameter sweeps: 1-5% each, done after P1 because the GPU waits for the CPU anyway.

Order of work: P1a, P1b (simulator first, it is offline), P1d A/B, P2a decision, P3a/P3b, P1c, P2b, P4.  Everything else in
phases 7-11 stays below this line.

## Phase 14 (after the capabilities exist): the application end to end, and new processes

Goal: the whole loop (opencode -> serve/server.py -> engine -> tools -> next turn) as fast as possible, not only tok/s.
1. Measure a real agent session as a timeline: per turn prompt re-read, thinking, answer, tool wait, idle GPU.  Rank the
   turn-level losses the way phase 13 ranks the window-level ones.
2. Use idle time: tool-run windows (seconds while a build/test runs) for pre-staging the next turn's experts (item 17),
   prefix warm-up of the expected next prompt, and background compaction.
3. Server/engine split: the server already knows the conversation, the domain and the tool state; pass that down so the
   engine picks seeds, checkpoints and draft sources per turn (one request protocol, not more env switches).
New processes built on these capabilities:
- Automatic A/B harness: queue + load gate + same-binary pairs + result log, run unattended at night; every opt-in switch
  gets a verdict line instead of a manual run.
- Quality gate as a process: token forcing + fixed task set on every candidate preset before it becomes default.
- Self-tuning from the owner's own sessions: routing traces recorded during normal use -> domain profiles, cache policy
  and draft settings re-learned offline weekly, checked by the simulator, promoted by the A/B harness.
- Regression watch: a short benchmark after each build/pull compared with the last good numbers.

## Phase 15: speed ideas from all fork commits (2026-11-10, owner: biggest gains first)

Source: [reports/Speed ideas from all forks.md](reports/Speed%20ideas%20from%20all%20forks.md) and its notes in
`research_notes/Speed ideas from all forks/` (6 areas: HIP kernels, CUDA kernels, MoE offload and CPU, speculative
decoding, attention/KV/quant, server and system). Input: the fork knowledge base `data/fork-kb/` (407 forks, 43,685
own commits, `upstream_history = 0`). Every gain below is the fork author's claim on their hardware or our estimate;
nothing was measured on our box yet. Items already in Phase 3 of bench/TEST_PLAN.md are not repeated.

Order: (1) ideas useful on BOTH GPUs, (2) env-only A/Bs and probes that decide whether a port is worth writing,
(3) exact code ports, (4) lossy arms behind the token-forcing and tool-call quality gates.
"Both" = the idea ports CUDA <-> HIP; the port column says what changes (warp intrinsics -> DPP, mma/IMMA -> WMMA,
launch cost ~14-16 us on gfx1100 vs ~3 us on sm_86, so fusions are worth more on the 7900 XT).

### 15A. Both GPUs, largest expected gains

| ID | Idea | Area | Exact/lossy | Port / note | Expected gain (claim -> our estimate) | Source |
| --- | --- | --- | --- | --- | --- | --- |
| F1 | `STRATA_PARK_TRUNCATE`: park the deepest checkpoint prefix that fits instead of dropping a too-large conversation | server | exact | backend-independent | our turn log: one session lost ~162 s on 7 cold 54-61K re-reads -> large TTFT win on long agent sessions | TwoAnts `5ef4fc8b4b` |
| F2 | `--prompt-cache-spread` on in the presets (flag already in pancake), up to 16 checkpoints per conversation | server | exact | env/preset only | xrip: edited 82K conversation resumed in 10.2 s vs 40 s | xrip; Lamina `13d90a38ec` |
| F3 | Extra checkpoint cuts (where a prompt leaves the previous one, before the last message end) + borrow before a stashed conversation's newest checkpoint | server | exact | backend-independent | Lamina: 6 agent tasks 412 s -> 369 s | Lamina `ab808b133e`, `83f40aee62` |
| F4 | Sub-2-bit shadow expert pack (IQ1_M / IQ2_XXS) read only on CPU misses; VRAM hits stay exact | MoE/CPU | lossy | both GPUs (CPU side); needs a ~25-30 GB requant | svgop (5090): solo +3-8%, concurrency up to +21% -> est. edit +7-10%, chat +3-5% | svgop `25bf1b90f6`, `eeca072bb6`; mandoof1 `85b991da79` |
| F5 | `STRATA_VERIFY_ECAP`: cap the experts read only by draft rows | spec | lossy | backend-independent | jmnargi (2x V100): +2.8% pooled, +11.5% drift-paired -> est. +3-10% on edits; must pass the tool-call gate (a tight cap broke 12/20 tool turns) | jmnargi `24894e676b`, `7c5041c358` |
| F6 | Exact speculative sampling for sampled requests: `STRATA_SPEC_PROB=1 STRATA_SPEC_COUPLED=1 STRATA_SPEC_PROB_GATE=top STRATA_SPEC_PROB_DT=0.85` | spec | exact | env only, both | riceharvest (B70) +5.5%; svgop (5090) +5.8% / +13.4% with DT 0.85 -> est. +3-8% on sampled turns only (check opencode's temperature first) | riceharvest `7389a10f2c`; svgop `0355c3164f` |
| F7 | Faster bit-exact Q2_0 AVX2 kernel (8-op unpack, 4 rows at a time; `STRATA_Q2_LEGACY=1` for A/B) | CPU | exact | both GPUs (CPU path) | imanu86: 1.11-1.45x single thread, Q2_0 layer -17% in a 7-worker pool -> est. edit +1-3%, more on the Q2_0 file | imanu86 `c3fc8f34d3` |
| F8 | `STRATA_SPEC_CUM`: cumulative-probability draft stop (pair with item 5 `tail_replay.py`) | spec | exact | both; ~150 lines | android-imdad (7900 XTX): +1.8% pinned, +6.4% short chat, +2.1% agent -> est. +1-3% | android-imdad `f1bd09212a`, `ef42c2fde9` |
| F9 | Faster router publish: write the step number instead of RMW over PCIe, copy MoE rows to host inside the router GEMV, DPP/warp top-10 | kernels | exact | CUDA warp reduce <-> HIP DPP | ours: `route_publish_res` 17.8 us/call, 2.68% of GPU time; est. +1-2% (7900 XT), smaller on sm_86 | android-imdad `caa9b00115` (`STRATA_ROUTE_FAST`); Ben1078 `23d2fc0450` |
| F10 | `STRATA_ADAPT_PROMPT`: prompt routing feeds the adaptive tier | MoE | exact | backend-independent, ~10 lines | est. +2-5% on the first decode tokens after a long new prompt | zsariboga `4ce19344ff` |
| F11 | Window decode attention reading K/V once per verify window (`STRATA_ATTN_WINDOW`, int8 MMA) | attention | rounding-level | CUDA int8 mma -> gfx11 WMMA | Asofwar (5080): 200K decode 48.6 -> 88.9 tok/s, 130K 112.6 -> 146.4 -> long-context decode on both GPUs | Asofwar/Strata-Turbo |
| F12 | Cheaper IQ2/IQ3 sign expansion in the expert GEMVs (port with EXPDEC_V2) | kernels | exact | HIP measured; CUDA drop-in for the `__vcmpne4/__vsub4` sites | boxwrench (7900 XT, GLM engine): IQ2_XXS gate/up -29%, IQ2_S down -35% kernel time -> est. 0-1.5% decode | yqYo1 `6f0bdf665b`; RX5950XT `b9a2111d3f` |
| F13 | Route-prior guards for turbo (`STRATA_ROUTE_PRIOR_PROTECT/KEEP_P/SCOPE`, `KEEP`) so turbo can run a stronger lambda without breaking tool JSON; compare turbo against a uniform tail skip at equal KL | MoE | lossy | backend-independent | enables a stronger turbo; no speed by itself | zsariboga `8d7042ad97`; mandoof1 `3e0489e118`; lighttransport `2ea0c1e042` |

### 15B. RTX 3060 (CUDA) first, then port to HIP

| ID | Idea | Exact/lossy | Expected gain | Source |
| --- | --- | --- | --- | --- |
| F14 | KhaledElzarw sm86 kernel set: `STRATA_EXP_MODE=9` + `STRATA_EXP9_DOWN=42:10` (Q2_0 down on 4 lanes), GR_DOWN86/UP_HYB/XN/UP_PL/D87, `MMVQ_IL_3090` table (re-tune for 28 SMs); then try the Q2_0 down fold on gfx1100 vs `STRATA_EXPDEC_DOWN` | exact | 3090, same upstream base and model family: +17.3% cumulative over its baseline -> est. 3060 +8-15% (all exact items together, speculative) | KhaledElzarw/Strata |
| F15 | `STRATA_FETCH_ADMIT` (keep PCIe-copied experts in the VRAM cache), only if the PCIe share stays above 0; check it with `ADAPT_NOWAIT` | exact (KL 0.002) | AdrianBM96 (3060): +5.1/+7.0/+5.1% decode | AdrianBM96/Strata3060 |
| F16 | Lamina sub-warp grouped expert kernels; router with separate-block copy, one fence, hardware top-10; int8 verify projections/head | exact / rounding | Lamina (3090): +5.7-8.3% e2e; +0.9-2.7%; +2.2-6.3% | eddoursul/Lamina |
| F17 | Qwen3.6 on strata-poly: RandoTeam `STRATA_FATTN` (prompt flash attention, q8_0 cache) + `STRATA_FDEC` (tensor-core decode attention) = MOD_PLAN 8.8 | fp16 math, KL gate | RandoTeam (4060 Laptop, 118K): 1.62x prompt, 1.60x decode -> est. 3060 prompt +20-35%, decode +10-60% from 4K | RandoTeam/4gbVramStrata `79b1e756`, `67f60c12` |
| F18 | CUDA-only Flash-Next prefill fusions | exact | zsariboga (5090, same IQ2_XS): -4.0% prompt time at 32K | zsariboga `21107e9d` |

### 15C. RX 7900 XT (HIP) only

| ID | Idea | Exact/lossy | Expected gain | Source |
| --- | --- | --- | --- | --- |
| F19 | Qwen3.6 on the 7900 XT: either port RandoTeam's poly HIP work + a gfx11 WMMA flash prompt kernel, or move Qwen3.6 to the in-tree q36 engine with the fork fixes (FP16 K/V, K/V read once per group) | exact / fp16 | storm-ace (gfx1151) 32K prompt 360 -> 938 tok/s; dongnh311 (3060) decode after 16K 12.0 -> 27.5 tok/s | RandoTeam `44b58c6a`...; storm-ace `e4fc07ee`, `4fe0284d`; dongnh311 `62cc598f`; maoyufeng `2f7020f2` |
| F20 | Empty-block launch probe on gfx1100 (unimatrix `empty_grid.cpp`), then resize launch shapes of the decode graph (upstream gfx12 `grid_pad`) | exact | unimatrix (7900 XTX): a 384x256 launch cost 14-16 us empty; 96 blocks gave +6% there -> 0 to several %, unknown until probed | unimatrix099 `d0c8d6c3e2`; mandoof1 `2b46e4b13d` |
| F21 | Plan copy with 256 threads and 16-byte reads (`STRATA_WAIT_COPY256`); `STRATA_ADAPT_COPY2`; `multi_block_head_ops()` on gfx1100 | exact | +0.2-0.4%; 0-0.5%; ~0.1% | android-imdad `caa9b00115` |
| F22 | Port `STRATA_HIP_WMMA_Q4` (gfx12-only) to gfx11 so Q4 KV prompt attention is fast and frees VRAM for experts | exact | VRAM for experts + faster long prompts | upstream / forks (see attention notes) |

### 15D. Measured negatives (do not repeat)

Graph fork/join branches on 7900-class cards -23..-30%; expert kernels inside one persistent launch 2x slower;
`STRATA_DEQUANT_WIDE=1` 8-14% slower; next-layer expert prefetch -3..-4% (and our ARENA_WARM was worse); SMT siblings
as pool threads -20%; pools split across CCDs -13..-20%; futex-only parking -14%; polling the stream at window end -3%;
`GPU_MAX_HW_QUEUES=8` prompts 10-20% slower on a 7900 XTX; 4-bit CPU repack (adds bytes on a byte-bound pool); draft
trees / deeper chains / eager draft launch (-2.73%); DFlash2 (lost on a 3060); more window-size controllers (same family
as STRATA_SPEC_COST, flat here); the chunked WY GDN scan (0.73x on a 3060); renormalizing dropped expert weight (worse
quality); PCIe share on the 7900 XT (off is faster); AVX-512 / VNNI / NUMA work (no hardware).

## Phase 12 (future): a second GPU

The owner plans a second card.  These were judged "does not apply" on one GPU (research reports above) and become the work
list once two cards are in the box.  Each needs its own A/B on the real pair; the gains quoted are other people's hardware.

| Technology | Source | Why it waits for the second card | Expected value with two cards |
| --- | --- | --- | --- |
| Tensor parallel decode (every layer on both cards) | Strata-rdna3 src/core/tp.cpp, --tensor-parallel 2 | needs two cards | rdna3: IQ3_S 2x 7900 XTX 129-170 tok/s vs 112-143 on one |
| Layer split + pipelined windows (stage 0 runs the next window while stage 1 verifies) | upstream --layer-split, --pipeline-windows | needs two cards | overlapped verify windows |
| Two-stage prompts (late layers of a chunk on the second card) | StrataFast --prompt-split, MOD_PLAN 8.7 | needs two cards | prompt reading |
| Peer expert transfer / RCCL, idle-queue parking (kfdq) | Strata-rdna3 peer_experts.cpp, peer_transport.inc, kfdq.cpp | 2-GPU only; RCCL headers needed (librccl.so present) | expert traffic between cards; idle power 98+91 W -> 5+13 W |
| Prefill / decode disaggregation (one card reads prompts, the other decodes) | llm-d 1.0, NVIDIA Dynamo, Mooncake, DistServe, SGLang PD | Strata's own SPLIT_SERVE lost 4-48% on two GPUs; transfer over PCIe 3.0 without RDMA | test only for agent sessions with long prompts while a reply decodes |
| KV streaming between cards (NIXL / Mooncake transfer engine style) | vLLM connectors, Mooncake | no second device | pairs with the item above |
| Chunked-prefill interleave, overlap scheduler, reasoning-aware scheduling (PASCAL) | vLLM V1, SGLang, TensorRT-LLM | protect OTHER users' decode; one user here | only if pancake serves several users or agents at once |
| GDN megakernel (one persistent kernel per GDN layer) | Strata-rdna3 megakernel_gdn.cu | 0.0 ms on one GPU while CPU experts gate the window; measured 1.6% on rdna3's TP config | revisit with TP |
| Heterogeneous-accelerator decoupling (xLLM service/engine split, vLLM platform plugins) | jd-opensource/xllm, vllm-ascend | single AMD card | only for a non-AMD second card |

## Risks

- Code drift: FastStrata's options live in `src/poly`, not in `src/qwen35`; porting may mean rewriting, not copying.
- HIP parity: several gains rely on NVIDIA tensor cores or CUDA graphs.
- The 7900 XT is not detected today; AMD work is compile-only until it is back.
- Disk: the root disk is tight; builds, models and caches go on the D drive (`/home/dan/clone-of-d`).
- Upstream moves fast (thousands of commits a month); rebase on Strata releases, not on every commit.
