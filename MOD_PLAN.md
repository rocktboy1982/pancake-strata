# pancake-strata: plan

A performance fork of [Strata](https://github.com/Niko1221/Strata) for single-user agent serving on one home box.
This file says what is true now, what runs next and what is waiting for a decision. Consolidated on 2026-11-15 from
the old 3,984-line plan and about 200 side documents, after an item-by-item audit against the code, presets, specs and
result archives (the old text is in git history: `git show 5ab3d2db:MOD_PLAN.md`).

Companion files:
- [TEST_LEDGER.md](TEST_LEDGER.md): every test ever run (537+ rows, ids `T-nnn`), with numbers, spec, tier, verdict and
  the archive path; plus the data conflicts found while merging. Cite tests by id.
- [LEADS.md](LEADS.md): every unported idea from forks, upstream PRs, llama.cpp, papers and hardware notes (~600
  leads), with author numbers and whether our tree has it.
- `bench/specs/` (A/B specs), `bench/results/` and `pancake-docs/research/bench/results-*/` (raw evidence),
  `configs/pancake/` (presets), `tools/pancake/` (measurement suite).

## 1. Goal, box and models

Fastest single-user agent serving (long growing prompts, tool calls) with output identical to plain decoding unless a
flag says it is lossy. Speed = decode tok/s (new text and code edits), prompt tok/s at 4K/16K/32K, and time to the
first token of an agent turn after a long conversation.

| | |
|---|---|
| GPU | RX 7900 XT 20 GB, gfx1100, ROCm 10.0 (`/opt/rocm/core-10.0/lib`), hipBLASLt table `tools/hip/gfx1100-hipblaslt-100401.txt` |
| CPU / RAM | Ryzen 7 5700G (Zen 3, AVX2, no AVX-512, no WAITPKG), ~57 GB DDR4 (~33 GB/s; the CPU expert pool is bandwidth-bound) |
| Bus | PCIe 3.0 (forced by the 5700G), ReBAR 32 GB BAR0, IOMMU passthrough, THP madvise |
| Gone | RTX 3060 12 GB removed 2026-11-08: CUDA/sm_86-only items are parked (section 8) |
| Daily model | Qwen3.8-Flash-Next IQ2_XS (GSQ-RCO pack), preset `configs/pancake/flashnext-iq2_xs-fast-hip.json`, `run-pancake.sh` default |
| Opt-in | Flash-Next Q2_0 (`*-q2_0-*-adopted.json`; 134.7-136.1 tok/s, 80.6% top-1 vs Q8_0, T-377; the 15c Q2_0-vs-IQ2_XS redo is still open); Qwen3.6-35B-A3B (`src/qwen35` = strata-q36, and `strata-poly`); DeepSeek V4.1-Flash (`src/dsv41`, owner-deferred: Qwen first) |
| Not goals | Qwen3.8-27B dense, NVFP4 / `.basalt` (Blackwell), Vulkan/SYCL/Intel/Mac tuning, another engine as the base (vLLM, SGLang, KTransformers, TensorRT-LLM, LMDeploy, MLC-LLM, ExLlamaV2, PowerInfer, TGI, LightLLM, NanoFlow, DeepSpeed-MII, TensorFold: they keep the model in VRAM or need far more memory) |

Baseline (HEAD 693cdf63d, spec 43, T-392): short 111.5, edit 112.3, longdec10k 91.7 (noisy), longdec32k 102.8 (noisy)
tok/s. Reference points: stock Strata 54.5 vs ours 75.5 on the 3060 (T-216); Strata-rdna3 109-119 and Avalon 95-100
on this card (T-009); window budget short 24.8 ms = 15.9 GPU / 5.2 CPU / 2.2 other, edit 37.4 ms = 15.8 / 17.4 / 2.5 + 1.7 sync;
decode profile top kernels combine_cpu 25.8%, wait_flag 14.8%, rdna_kq 9.0%, expdec_gu 6.8% (spec 40 profile).

## 2. Rules

Measurement (the full text is in AGENTS.md "Measurement rules"):
1. Paired A B B A runs, run 1 discarded, quiet machine (`abtest.py` waits and refuses beside a running server). Bar:
   >=2% end to end exact, >=3% lossy; TTFT bar >=0.1 s per warm turn or >=2% on long prompts. Screen, then confirm
   (`--block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 9`). Owner rule since 2026-11-13: promote from 0% up
   at screen, decide at confirm; "test, never judge by reading".
2. Text identity is not an exactness proof (stock-vs-stock greedy noise is 1.2-1.3% of tokens; the base arm gave 6
   hashes in 6 runs). Lossy arms need token forcing (`quality_gate.py`, 20 requests x 300 tokens, 2 stock arms; it
   rejects only large deviations) and then the task gate (`task_gate.py`: LiveCodeBench subset + BFCL v4 multi-turn +
   own-repo grading) to choose between close variants.
3. Run `python tools/pancake/spec_lint.py` before any A/B: an arm key the fork does not read is a no-op.
4. Gains inside one substitute group do not add; price an idea against the window budget first (edits are a plain sum,
   so a small exact item caps at a few %). Re-measure fork claims here: Speedy's +22-33% on 3080s was noise..+9% here.
5. Three outcomes only: adopted (numbers in the commit), coded opt-in, not adopted (evidence + what would reopen it).
   Every test goes into TEST_LEDGER.md; nothing is dropped silently.

Code:
6. New features are opt-in; the default path stays byte-identical, and the commit says how that was checked. A default
   changes only after a paired measurement. Every new `STRATA_*` name is classified in
   `src/core/conversation_disk_env.def` (chunk-geometry switches are KeyIfSet in the disk key).
7. HIP is the benchmark platform; a CUDA-only change must not break HIP. Official model files + SHA-256 only;
   `presence_penalty=0`; no `STRATA_AUTO_PIN` for Qwen; `ROUTE_TAIL_SKIP=7` only by owner decision.
8. Credit every adopted idea in README section 9 in the same commit (fork, commit, author numbers).
9. Develop by day, run the GPU queue at night (`tools/pancake/night_queue.py`, `--dry-run` first).
10. Linux box repo on an ntfs3 disk (D): `git config checkout.workers 1`; no `reset --hard`, full checkouts or new
    worktrees there; merge only in the existing worktree.

## 3. What the daily preset runs (fast-hip), with the evidence

| part | setting | evidence |
|---|---|---|
| RDNA ports from Strata-rdna3 | `GR_RDNA`, `GR_RDNA_Q8`, `GR_RDNA_A8`, `GR_INJ_UP`, `GR_FUSED`, `RDNA_Q6K`, `RDNA_MMVQ_WAVE=3`, `COMBINE_CPU`, `WAIT_FUSE`, `QFUSE`, `HEAD_MIX_MULTI` | decode 87-101 -> ~112 tok/s, T-009; round C best -> preset f0f9c5788, T-378 |
| expert tier | `--pcie-frac 0` (T-197; re-confirmed: 0.15/0.3/0.55 all slower, T-268), `--adapt-every 1 --adapt-swaps 32` (T-199), learned profile `data/expert-profile-pancake.bin` (+6.7%, T-194), GPU router lookahead default on (+10%, T-192), `ADAPT_PERSIST` default on |
| CPU pool | `POOL_FUSE_PHASES=1` (edit +2.5% +-1.7, T-224), `EXPDEC=1`, `EXPDEC_DOWN=1` (short +1.7% +-0.5, T-037/T-040) |
| drafting | `--spec 4` (official MTP depth; spec 2 -5%, spec 5/6 flat, T-118/T-149), `--spec-min-p 0.5`, `--lookup-max-t 12` (edit +8.8% +-6.8, quality PASS, T-126/T-127), English draft vocab |
| prompt path | hipBLASLt 100401 table (1,345 -> 1,850-1,920 tok/s, T-096), `PF_GEMM`, `PF_F16=2`, `PF_FUSED=1` (see section 7: no HIP A/B recorded), `PF_PAD`, `PREFILL_AUTO_MAX=32768`, `PREFILL_CPU_SHARE=0.6`/`_MAX=4096`, `PREFILL_RING=192`, `SELECT_WMMA`, `HIP_WMMA=1`, Speedy opt-ins `PF_GROUP_SCRATCH`, `PF_HC_REGION`, `PROMPT_ATTN_IMMA`, `KV_PREFETCH` (taken from Speedy's numbers, no HIP A/B recorded) |
| attention / KV | `--kv int8 --kv-resident 32768 --max-context 131072`, `ATTN_DAILY`, `QSA_ROWS`, `PA_FAST`, `KV_HOST_DMA` |
| deferral | `VERIFY_DEFER_CPU` OFF (owner decision 17.1; contested, section 4) |
| server | grammar on (spec decoding mask 0.1-0.2% of a window, T-315), thinking caps `reasoning_budget_grace: 512` + close retry, `env_to_user`, `effort_position: end`, loop guard, tolerant tool calls, turn log, `AGENT_COUNTERS=1` (log only, T-513) |

Lossy switches in fast-hip (an exact bundle = fast minus these 11): `GDN_CONVL2`, `GR_RDNA_A8`, `GR_RDNA_Q8`, `HC_UPMIX`,
`PA_FAST`, `PF_F16=2`, `PF_HCDOWN`, `PREFILL_AUTO_MAX=32768`, `PREFILL_CPU_SHARE=0.6`, `ROUTE_TAIL_SKIP=0`,
`SELECT_WMMA`. So the daily base is not bit-exact; "exact arm" in a spec means exact relative to this base.

Other presets: `flashnext-iq2_xs-hip.json` (exact reference, not daily); `flashnext-iq2_xs-turbo-hip.json` = fast +
`ROUTE_PRIOR=0.5` + `VERIFY_DEFER_CPU=1`, minus `AGENT_COUNTERS` (lossy; edit +29.7% n.s., token forcing INCONCLUSIVE
+0.83 pts, T-245; all-prose -8%, T-387; not a default); Q2_0 `*-adopted.json` carry FUSE_PHASES, EXPDEC_DOWN,
lookup 12, no deferral. Not in any preset: `ROUTE_PRIOR_AUTO`, `VERIFY_DEFER_AUTO` (-12.5% IQ2_XS, T-262),
`KV_RING_ADAPT`, `PARK_REWIND_MAX`, `PARK_DIVERGENCE_LOG`, `PREFILL_EQUAL_TAIL`, `ADAPT_KMISS`, the cache levers,
`VERIFY_ECAP`, `SUFFIX_INDEX`.

Interactions: `ROUTE_PRIOR` set (even 0) disables ROUTE_PUBLISH / ROUTE_FAST / ROUTER_FUSE; deferral turns off 6
fused paths process-wide (so A35 / ROUTER_FUSE "no-op under deferral" no longer holds with deferral off);
`ADAPT_KMISS` needs `--adapt-async 1`; `*_AUTO` and the fixed switch never together; `KV_RING_ADAPT` under `--serve`
follows `--max-context`; CPU share binds only for deltas <= ~400 tokens; PARK_REWIND helps only when admission is the
miss.

## 4. Decisions waiting for the owner (measured; the data is in)

| # | decision | data | what is missing |
|---|---|---|---|
| D1 | `STRATA_VERIFY_ECAP=1` into fast-hip | edit +33.5% +-14.3, short -3.3% n.s. (spec 56, T-270); token forcing +0.29 pts PASS (T-310). `ECAP=0`: edit +65.2% but +0.95 pts REJECT (T-271, T-311) | tool-call task gate on ecap1; spec 57 stack (ECAP=1 + lookup16 + suffix index + 9-col kernels) |
| D2 | cache levers into fast-hip: `--conversation-cache-mib 2048 --conversation-cache-min-tokens 8192` and/or `STRATA_PARK_TRUNCATE` 1280 | confirm on the live 11-09 replay: -21.6% +-14.8 prompt time (T-183), -23.1% +-11.8 (T-184), zero cold re-reads; screen T-182 | owner choice (RAM use); they overlap, test the pair if both wanted |
| D3 | deferral on IQ2_XS | night retest: off vs on -0.9% overall, code -7.2%, prose -3.2% (T-260); Q2_0: off +8.9% (T-261); first A/B edit +15.8% (T-222) | 5-repeat edit-window A/B on the IQ2_XS preset; until then 17.1 (OFF) stands |
| D4 | A24 `STRATA_SUFFIX_INDEX` | screen: short +26.8% +-34.4 (outlier), edit -0.0% (T-161); offline -2.44% rounds, sub-agents -7.5% (T-424); half the gain is from outputs older than a 30-min pause, so use the persistent `STRATA_SUFFIX_INDEX_FILE` (4 MiB -1.00%, 16 MiB -1.71%) | confirm + quality gate with the file on the daily server |
| D5 | `PF_FUSED=1` / `PF_F16=2` in every HIP preset | the only A/B (T-093, pre-HIP preset): 42K -7% but 5K +12% TTFT | either a HIP A/B (spec) or a recorded owner decision |
| D6 | 9-16 column RDNA K-quant kernels are now the code default (`kRdnaDefaultCols=16`, `kRdnaMaxCols=32`) | parity 192/192; engine +3.7/+3.8% screen, +1.8% +-7.8 / +4.3% +-5.0 confirm (T-081): inside noise | accept as default (it is a kernel default change, measured not worse) or gate behind `STRATA_RDNA_KQ_MAXCOLS` |
| D7 | Phase 18 item 9 merged the window-input copies on the DEFAULT path (`copy_from_mapped_multi`, on unless `STRATA_MTP_MULTI_COPY=0`) | pure copies, `verify_parity` covers both modes; spec 30 never ran | run spec 30 (build-vs-build) or flip the default back |
| D8 | Q2_0 vs IQ2_XS as the daily file | 15 screen +8.6% +-46.3 (T-382); 15c run unusable (T-384) | 15c redo |
| D9 | A29 entropy-coded miss path | plan said "owner approved building it" and also "stay parked"; bound 8-12% < 20% park bar (T-282) | pick one |
| D10 | housekeeping | 107 worktrees (~16 GB); stale upstream/HEAD ref; GPU clock back to auto after pinned runs; disk-tier cap (D 91-94% full); `PARK_REWIND_MAX` default; A41 read-before-edit hook warn-only trial (8/8 tests, not registered); cron `--night-only 23-07 --drop-cache`, `--reclaim-max 3.0`; sudo rules via `visudo`; Avalon 32K+ (shrink its context or drop the comparison: it OOMs at 32K+, T-174); delete `origin/NetworkPancake` (moved out of the repo to `C:\Users\iancu\NetworkPancake`) | owner |

## 5. Run next: coded, never measured (GPU queue, in this order)

1. **Spec 30** (D7) and **spec 57** (D1 stack). Fix the base path in specs 33 and 34 first
   (`pancake-flashnext-iq2_xs-fast-hip.json` does not exist; use `configs/pancake/flashnext-iq2_xs-fast-hip.json`),
   then run spec 33 (MTP bundle: `--mtp-q4 all --mtp-window 4096 --host-core sibling`) and spec 34's `pf-switch-min-t`
   arm (its `mmvq-wave1` arm is answered by T-072, `kv-grow` by T-391).
2. **q36 dense MMQ** for Qwen3.6-35B-A3B: env arm `STRATA_DENSE_MMQ=1` (gemm.cu:1377, HIP; prerequisites 14dd746f /
   55b8c26b in HEAD), prompt 8K/32K. Author (burzumishi, strata-q36): 32K prompt 635 -> 910 tok/s on a 7900 XTX.
   Lossy rounding: token forcing.
3. **EQUAL_TAIL** (`STRATA_PREFILL_EQUAL_TAIL`, `_MIN` 256): est. -1.5..-2.5 s of ~40 s at 100K, ~-1 s at 66.5K, up to
   -2.5 s at 34K. Use prompt sizes 30000, 34000, 60000, 66500, 70000, 100000 (the ttft_bench defaults sit on chunk
   boundaries); plus the #5539 check: 99,997 tokens -> chunks 0/32768/65536/98304 (with EQUAL_TAIL 0/25088/50176/75264),
   ~98,400 tokens, `STRATA_TRACE=1`, a lent line, a reply, no fault.
4. **Never-run specs for coded switches**: 26-stack (F12 `IQ_SIGN_FAST`), 19-turbo-guards (F13), 12-spec-sampling and
   a21-spec-lowtemp (A21/F6 `SPEC_PROB`, FK-01 `SPEC_PROB_DT`), 24-a15 (`PARK_TTL_S`; `pick_victim()` is in
   conversation_cache.hpp), 22-a7 (`PROMO_CHUNK`), 24-a45 (`CALIBRATE`), 16-a19 (`HIDDEN_DUMP`/`THINK_PROBE`, first step
   of the thinking stop; proxy labels cap the saving near 10%), 25-f-env-rerun (lookahead0, routed-dedupe, MMVF_ROWS;
   the 05s screen had no clean runs, T-527), a33-a35-fusion-retest (premise changed: deferral is off), 15-kmiss,
   taskgate-* (point them at `bench/tasks/repo-history.json`, not default.json), 08-build (`STRATA_HOST_NATIVE`).
5. **Coded, no spec yet**: F7 `Q2_FAST` engine A/B (bit-exact, x1.33/x1.23 rows, T-246/T-253); F14 `EXP9_DOWN=42:10`
   (= FK-24, parity in ctest); F22 `HIP_WMMA_Q4` end to end; F11 `ATTN_WINDOW` (q36); `HIP_GLOBAL_WEIGHTS` (flat loads
   13,119 -> 108, T-050); `WAIT_COPY256` confirm (screen +6.7% / +2.4%, T-056; inert below T=6, so check it acts at
   `--spec 4`); FK-02 `--prompt-cache-spread`, FK-03 `--lookup-chain 3`, `--adapt-decay`, FK-10 `ATTN_LANECELL`,
   FK-12 `--batch 2`, FK-27 `PARK_REWIND_MAX`; items 3-4 seeds (`CONV_WORKSET`, `DOMAIN_PROFILES`; offline 81.17 ->
   82.86%, T-398); A13 disk tier GPU test; A14 restore equivalence (`restore_check.py --control`); `SELECT_TWO_PASS` on
   HIP; selftune candidate profile (+4.6% modeled, T-439); `ROUTE_PRIOR_AUTO`; 17.3 ablation; TTFT scenarios C/D/E;
   `prefix_check`; A9 `IDLE_ADAPT` (ceiling <=0.26%, low priority); `STRATA_POOL_SPIN_US` sweep (Juud: 15.1 -> 12.6
   ms/token).
6. **Measure before code (TTFT, M1-M5)**: M1 8/32/64/100K with prompt_ms, TTFB, park, chunk plan, file-tier/KV lines,
   PREFILL_TIMING (each size twice, fresh state or park reported apart); M2 10-turn session x3 (alone; side request +
   sub-agent switch; `--conversation-cache-mib 1280`); M3 divergence; M4 opencode/Copilot request bodies; M5 deltas
   100/300/1000/3000 at 32K/64K. Unexplained so far: a 3.4 s turn after a 34K cold read, 4.8 s / 36 s waits outside
   the timer, the cause of 13% checkpoint fallbacks.

## 6. Code to write (ranked by expected value on this box; details and sources in LEADS.md)

| # | item | expected | exact/lossy |
|---|---|---|---|
| 1 | FK-28 prompt-fed tier counts, redesigned: the existing `set_usage` hook is dead (section 7). Feed `drive.d.usage` from the existing `sp.on_routes` callback in generate.cpp (as ram_adapt2 does, ~14963), weighted like RAM_ADAPT2 (`prompt_w*20*K/n`), env gate + disk_env entry | unmeasured (oracle own-counts +2.9 hit pts offline, T-417) | exact routing input, tier only |
| 2 | 0xluk `PF_AHEAD` + `REFILL_ASYNC` for the staged path of agent deltas < 4,096 tokens | +35% agent prefill (2x 3060, PCIe 3) | exact |
| 3 | llama.cpp #31751 RDNA3 MMQ J=32 spill pragma (check the pinned ggml) | +36% at 17-32-token batches, gfx1100 | exact |
| 4 | android-imdad exact decode bundle (`TOPK_REGKEYS`, `TOPK_SCAN`, `MTP_SELECT_WIDE`, `GDN_NORM128`, `ATTN_MERGE_BOUND`) | +10% chat, +11% agent, KL 0, 7900 XTX | exact |
| 5 | `PF_GU_EXACT` Tensile prompt GEMM for gfx1100 | 26.5 vs 12.9 TFLOPS; 64K prompt 40.1 -> 31.5 s (XTX) | exact |
| 6 | schedule H2D admission copies into CPU-idle phases (attention/GDN) | est. +10-18% | exact |
| 7 | CPU pool bandwidth push (two fused phases per layer, row-chunk split, touch the next expert early); imanu86 IQ2_S/IQ3_S AVX2 decode without per-index shifts (pool on IQ2_S layers 20-33% faster) | est. 33 -> 38-42 GB/s, +5-12% | exact |
| 8 | re-warm or protect the decode tier after a long prompt (decode 92.5 vs 111.7 tok/s after prefill) | up to the gap | exact |
| 9 | launch fusions (`FUSE_SHGATE/COMBINE/SWIGLU_DOWN/PLAN/ROUTEBELL/QGEMV`): launches cost ~14-16 us on gfx1100 | -400 launches/window (R9700) | exact |
| 10 | `EXPDEC_V2` remainder; `MMVQ_WAVE_ROWS` table; small exact bundle (copy_rows loads-first, sampler x8 loads, single-token MMVF_ROWS, lookahead submit after publish) | -20-24% expdec kernel time; +4% MI50; several us per launch | exact |
| 11 | Speedy `KV_STAGE_REQ` + `PF_SEL_BATCH` (long-context loan sizing, VRAM); `TOOL_STEP_SPLIT` + `canonical_tools` (serve only) | 300K cold prompt 466 -> 1527 t/s (2x 3080); turn-2 cache reuse | exact |
| 12 | rdna3 `791cca4b`/`5906629d` (AUTO_MAX 49152 + refill head `2b19d510`), `f557b544` (CPU share retries per layer) | 2-4% at 49K | exact |
| 13 | DALI per-layer per-step CPU/GPU assignment instead of the static A45 probe | +23% over HybriMoE (3090) | exact |
| 14 | QSA long-prompt selection cost (Quest page-level or partial top-k) | after-12K gap 2.18 ms/token | lossy-ish |
| 15 | uniform K experts (`Q36_EXPERTS_USED`, 35B); weight-based miss skip (`SKIP_MINW` / Maya `MISS_SKIP_W` / HOBBIT cumulative-gate, replay offline on a `--dump-routing` run first) | +14-19%; +7-13% | lossy |
| 16 | SpecLA: recover accepted GDN states from the verify-pass factors instead of replaying | unmeasured | exact |
| 17 | Phase 18 items 10 (pre-stage host records), 12 (event wait vs stream sync), 17/18/20 (device accept, fused argmax, branch rollout); A19 stop rule (after 16-a19); V5 (MTP layer ~0.9 GiB to the expert cache); U6; DraftPolicy cost-outlier clip | ceilings in the old plan's Phase 18 notes | exact |

Not packages any more (their parts are separate items above or in LEADS.md): 63 capability probe (`STRATA_PROBE` never
built; became A45), 66 agent-turn pipeline, 68 miss-cost speculation, 69 memory arbiter, 72 GPU-floor package; 94
typed flag registry, 95 split `generate.cpp main()` (hygiene, no speed).

## 7. Defects and stale claims found by the 2026-11-15 audit

Code:
- FK-28 `STRATA_ADAPT_PROMPT` is dead code: `Prefill::set_usage` (prefill.cpp:3009, prefill.hpp:302) has no caller,
  nothing reads the env name, no disk_env entry; it also duplicates `sp.on_routes` and adds unweighted counts. Wire it
  as section 6 item 1 or delete it.
- Default path changed without an A/B: Phase 18 item 9 (D7); RDNA 9-16 column kernels (D6).
- `STRATA_PARK_ASYNC` cannot act on HIP (`park_async_on()` false, conversation_snapshot.cpp:231): parks are synchronous
  inside TTFT.
- Possible fence bug: the staging readers after `wait_flag_ge_or` (verify_kernels.cu ~1619-1650) may lack the
  `__threadfence_system()` the flag-order test needed (test fix: after the spin and after the store, flag_order_test.cu
  26-29; its segfault was a ROCm runtime mismatch, fix `LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib`; failures 57x1 / 101x2
  = one device; line 120 prints "device N ok" even on failure; run the 4-way matrix with `HIP_VISIBLE_DEVICES=0/1`).
- Sampler: `p.temperature <= 0.0f` (sampler.cu:869) does not catch NaN or tiny positive values.
- `gdn_gate_kernel` computes only the first token (elementwise.cu:43; latent, re-check before fixing).
- Upstream #5273 short-chunk guard only on one `give_back` site (prefill.cpp:6215), not 4252/4396/4527/5029.
- Phase 17 review: R1 fixed (24de1106); R2 (`diverge_at` in conv_cache.hpp) and R3 (`max_cells` rule in kv_ring.hpp)
  are not in the code.
- A37 `IQ_LUT_DUP` (d11a9ca09) was never reverted although its gate said no-go (T-060): revert or keep off.
- `STRATA_DF2_GATE` is still read (window_rule.hpp, generate.cpp) although upstream retired it.
- `STRATA_PSI_SHED` (Speedy item 8.3) was never ported though the old plan counted it done.
- Tools: `night_queue.py` marks a job with zero measurements "done", so `--retry-failed` skips it; `abtest.py` text_check
  (:866) should accept any base hash; `bench/quality/set-v2.json` has 28 of 48 items with `cont: null` and
  `quality_gate.py:161` still defaults to `set.json`; all `taskgate-*` specs use `bench/tasks/default.json`.
- Item 38 O_DIRECT loader: the audit found it in HEAD (184e68e83, expert_source.cpp:852, `STRATA_UNBUFFERED_LOAD`), a
  research note says it is not built (expert_source.cpp:6155): re-check.
- Planned but not in code: `STRATA_FORCE_ISA` avx512 / avx-vnni values (item 57), CPU governor setting (item 41);
  `--pool-affinity measure` is a no-op on the 5700G. PCIe 16 GT/s vs Gen3: settle with `lspci -vv`.

Plan claims corrected (all now reflected above): Q2_0 is not the daily file (IQ2_XS is); FK-22 is F9 `ROUTE_FAST`,
coded and dropped at confirm (T-068); FK-24 is F14, coded; `EXPDEC_DOWN` was measured and adopted; the 7 upstream picks
of 2026-11-12 are merged (7529df36 .. 131cca3f); A15 `pick_victim()` exists; ExTV patches 03/06/07/08 were already in;
spec 40 `wmma-attn` was a no-op and `mmvq-wave32` compared mode 1 vs 3 (T-072); U2 is already covered by
`mmq_ids_margin.hpp` (f225844d, PR #5108); ReBAR rows into VRAM (`PRESUM_VRAM`, item 52) and the Athena bandit were
coded and measured with no gain (T-242, T-266); `POOL_PREREAD` loses (T-239); P2c small flag stack -20% short (T-474);
`MTP_HIP_PRIO` no difference (T-141, T-142); item 23 depth prior: screen turbo-depth75 short -0.3%, edit -1.3%, dropped
(`bench/results/2026-11-10-test-phase/06-turbo-depth-screen`); the 7900 XT is not
"compile-only" (measured daily since 2026-11-08); spec 5 is in the do-not-repeat list (T-118, T-160). Grammar commits
cited in the old plan were squashed into 3d3eef53e.

Branches: every `origin/feat/*`, `work/*`, `dev/*`, `exp/*` branch and `optimize-kernel-hip` is in this line. Only
`origin/main` has 3 serve/GUI commits not here (19ffe37a status page em dash, 9d456105 total VRAM, d7defcbb turn record
keeps thinking tokens): merge `origin/main` when convenient.

## 8. Parked (hardware or scope)

- RTX 3060 / CUDA sm_86 only: Speedy sm_86 presets and `HC_Q8=1` (+5.5-6% on 3080s; keep `HC_Q8=0` on HIP), KhaledElzarw
  sm_86 kernels (`GR_DOWN82/86`, `GR_UP_HYB`, `GR_XN`, `GR_UP_PL`, `GR_D87`, `ACTQ_SKIP`, `GR_RES`, `MMVQ_IL_3090`), F17
  `STRATA_FATTN` (in `src/poly/fattn.cu`, needs a CUDA card), `--vram-reserve-mib` 444 (Speedy, 10 GB card), basalt
  head work beyond `q5k_head.cu`.
- Two GPUs (layer split): Speedy `PIPELINE_PENALTIES` / `PW_RES_PATCH` / `ROUTED_STREAM_PP`, ExTV patch 01
  (`--prefill-main`, `--kv-resident-later`, `--prefill-main-small`), upstream #3821 #3819 #3648 #3742 #3306 #4549 #4534
  #4380 #4522 #859, Phase 12 tensor parallel.
- Scope: DeepSeek V4.1 ports (owner: Qwen first); GLM-5.3-Flash / MiniMax-M3.1-Flash (Project Maya, upstream #4319 closed
  "not planned"; MiniMax maybe 5-15 tok/s with a 29 GB deficit, not started); nasone32 EXL3 (format not portable; its
  HIP kernel patterns are a lead); Cactus; ISTA coder (needs 29.6 GB resident); sleep-in-place (basalt 64167ba73).

## 9. Do not repeat (measured and dropped; numbers in TEST_LEDGER.md)

- Decode kernels: wave32 MMVQ mode 1 (T-072), occupancy/waves_per_eu (T-074, spill already at the VGPR limit), matrix-core
  few-column GEMV (T-083, T-084: -1.7% short, -5.4% edit; built, off), wide-tile int8 prompt kernel (T-107, -18..-42%,
  reverted), `WAIT_FUSE` off / fuse-wait (T-077), combine-poll1 (T-064), `VERIFY_DEVICE_PLAN` (T-066), `ROUTE_FAST` and
  `DOORBELL_STORE` (T-068), IQ2_S pair decode (T-008), upstream group-A switches (T-013), upstream `HC_Q8` on HIP (T-012),
  `PF_WMMA` (T-094), graph fork/join, persistent expert launch, DEQUANT_WIDE (T-528), LUT dup (T-060), BitDecoding WMMA
  decode attention (attention ~3% of decode, T-019).
- Drafting: `--spec 2/3/5/6` (T-118, T-149, T-159, T-160), DFlash2 (T-119), `SPEC_COST` (T-120), full / zh60k draft vocab
  (T-121, T-123), real-code draft vocab (T-146, T-151), mtp-window 4096 alone (T-122), `MTP_EAGER` (T-136, T-145, T-159),
  `--mtp-q4` head (T-143), `MTP_HIP_PRIO` (T-140..T-142), EcoSpec (T-148), A22 DISCO chain stop and FK-21 `SPEC_CUM`
  (replay +0.3%, T-455), A26 draft-source selector (T-447), DraftExpert windows (T-115), lookup windows off (T-150).
- Expert tier and cache policy: async tier and `ADAPT_SWAPS_AUTO` (T-221), kmiss on the blocking tier (T-237), rank
  guard top2 / FOLD (T-223, T-247), A44 bandit family (T-266), Foresight swap slots (T-190), CPU lookahead scorer
  (T-188), predictors / co-activation (T-394, T-401), per-layer budgets (T-397), ClassLRU / decay sweeps (T-395, T-451,
  T-452), learned admission A46 (T-408, T-426), FineMoE / PRI / MoE-Infinity (T-431), REAP pruning (T-433), ADAPT_LAG
  (T-198), rdna3 adapt and profile settings (T-203, T-204), mmap'd expert tier (T-200, T-216), arena warm (T-196),
  reuse-aware admission (T-217), dense-expert sparsity (T-219), A36 weight-major (T-045), A7 chunked promotions as
  modeled (+4.6% upper bound, T-427), `--pcie-frac` > 0 (T-268), `--kv-resident 131072` and kv8k lending (T-276,
  T-273), VRAM reserve 512 (T-278), AUTO_PIN pin (T-171), ExTV supersede rule (T-180).
- CPU pool: worker / task counts (T-195, T-202), lazy park, mwaitx, tail4, Q2 prefetch (T-232..T-235), presum and
  presum-vram (T-241..T-244), pool pre-read (T-239), `ACTQ_SKIP` (T-255), long-row grouping (T-254), shadow pack
  sub-2-bit (T-251, T-252), nice 5 (T-201), SMT siblings / CCD split / futex parking (fork-KB, measured worse).
- Prompt / TTFT: prefill CPU share arms (T-108), `WMMA_GEMM` (T-105), `PREFILL_AUTO_MAX=24576` (T-106), desc-cache off
  (T-101), BFLA as shared QSA selection (T-098), ContextPilot dedup (T-165), TTFT min-tokens / cache-mib arms on the
  10-turn session (T-178), `--ple-io mmap` (T-274), item 2 decode-cache through long prompts (T-163).
- Other: `ROUTE_PRIOR` as a default (token forcing REJECT, T-294, T-295), tail skip 7 (T-298), `KV_RING_ADAPT` (T-291),
  `HW_QUEUES` default, `SELECT_LONG` (T-010, T-011), FK-04 aux-cpus / kv-grow (T-391), spec 31 arms (T-389), A20 entropy
  stop host cost (+17% thinking time, T-529), A43 / A38 constrained edits (T-443, T-459), A31 GDN chunk 64 numbers
  (T-103), dedup, MEXT, new compression ISAs (LEADS.md section 0 "Closed").

## 10. Architecture facts worth knowing before touching the code

- Chunk plan: the serve loop cuts at root_at (first `<|im_start|>` after token 0 if >= `--prompt-cache-root` 2048; a
  smaller system prompt is re-read on purpose, ~0.3 s), turn_at (last `<|im_start|>`) and n-1; `CACHE_MESSAGE` /
  `TOOL_BOUNDARY` and `pin=` add cuts; <= `--short-read` 64 tokens go through decode windows; chunk = min(prefill_chunk,
  segment rounded up to 256) via lend -> planned_chunk -> equal_chunk -> request_chunk (generate.cpp ~997, ~1009,
  ~10355; loop in `Prefill::run_impl`; `STRATA_TRACE` prints `prompt chunk` / `lent`). The loan lasts until the segment
  ends.
- Auto chunk: largest 256-multiple with loan <= 90% of slots (85% if less pinned), 128 left; AUTO_MAX clamps
  32768/16384/8192; loan at 32K = 9,413/10,733 slots, 12.59/14.37 GiB, ~5% resident (generate.cpp ~3938).
- With CPU share armed, `stream_all_min` = `CPU_SHARE_MAX` 4096: T >= 4096 streams (192-slot ring, copies overlap
  attention); T < 4096 is staged (per-layer sync on routing, 8-slot ring, 32 with MMQ groups). A warm turn is one staged
  chunk and the copy engine idles through attention (the reason for PF_AHEAD). CPU share only takes experts with <= 8
  tokens (MAXT, expert.hpp:143); share=auto keeps running means (w 0.25), clamp(g/(c+g), 0.05, 0.9), needs 3 ratios,
  probes every 29th layer, HIP off except a single gfx12.
- EQUAL_TAIL (prefill_plan.hpp:182, `_MIN` 256) is skipped when evened chunks would be < 8,192 (kEqualTailMinChunk, :172;
  upstream measured 25 tokens past 8,192 read 25% slower when evened) or below the staged limit; `PREFILL_EQUAL` (#693)
  evens only when the last chunk would stream anyway; EQUAL_TAIL wins on a staged tail.
- Checkpoints: `--prompt-cache 6` (generate.cpp ~961), ~113-118 MiB each, at the turn boundary and chunk ends >= 16,384
  apart; restore only from a prefix (GDN state). Parked conversations: 1,280-6,656 MiB auto (usually 4,352), 4 slots;
  snapshot = 111 MiB + 14.9 KiB/token + 113 MiB per checkpoint. Refill is one wait (~1 s for a 12.6 GiB loan;
  `STRATA_REFILL_BLOCKING` keeps the old form; trace prints `refilled ... in <ms>`).
- Prefix reuse breaks on: client edits; system prompt or tool list change (tools render in the system block -> full
  re-read); effort; a re-render that differs; dropped reasoning (re-renders only the last turn); eviction; loop-guard
  collapse.
- Floor model: 48 layers x 512 experts, top-10, 1.4 MB/expert, 44% resident, routed share 1-exp(-10n/512), floor ~0.67 s,
  PCIe-bound. QSA indexer cost grows n^2, attention ~2,051 cells/query; KV staging ~1,056 B/token/layer over 12 QSA layers.
- Hot path hygiene: no per-window getenv (c44b1ad7b, 7c0f03228), no hot-path allocations, no decode file I/O; the 4
  dispatch loops in `expert_pool_dispatch_multi` are linear (`first_of`, `miss`, `next_of`, `q_of`; greedy check owed).
  Default-on: `ADAPT_PERSIST`, `UNBUFFERED_LOAD`, `HIPBLASLT_DESC_CACHE`, `PIN_PARK_TOKENS`, `PSI_GUARD`. `ARENA_WARM`
  removed; `combine_multi_hits` kept (removal = fork drift).
- Code tools: graphify (`graphify update .`, `graphify-out/` git-excluded), ast-grep 0.51 + MCP, tree-sitter MCP needs
  `mcp<2`.
- Platform: ROCm 10 has no `hipMemcpyBatchAsync`, so `STRATA_DMA_BATCH` is CUDA-only. The Flash-Next "IQ2_XS" file holds
  IQ2_S / IQ2_XXS / IQ1_M gate/up with Q2_0 down (~176.9B parameters); its head is IQ4_XS. Q2_0 and IQ2_XS share one
  n-gram table file (same SHA-256). English 40.5K draft vocab is best for English/code (+5%, ~215 MiB less VRAM); zh60k
  helps Chinese only. KV quant: q8_0 costs < 0.00012 KL; 4-bit adds 0.0016-0.0027 KL and grows 15-30% from 4K to 32K;
  keys are ~2x as sensitive as values.
- strata-poly (Qwen3.6-35B): one CPU thread per physical core (8; SMT adds no bandwidth; poly does not pin threads);
  with the expert cache on it is not run-to-run deterministic (swaps and CPU/GPU rounding part texts after 9-81 tokens),
  so exactness checks need `--no-gpu-experts --expert-swaps 0`; Q8 KV 11.7 KB/token vs strata fp32 45 KB/token; prompt
  reading is GPU-compute bound (attention 38% of a 42K read), decode CPU experts ~half and RAM-bound (33.4 GB/s).
  Reproduce on CUDA: build `strata strata-poly poly_gdn_cols_test poly_gemm_small_test` with
  `-DCMAKE_CUDA_ARCHITECTURES=86-real`; daily poly flags `--max-context 131072 --threads 8 --kv-q8 --attn-split
  --graph-experts --elastic-prefill 1024 --lend-staging --stage-next --suffix-drafts 256 --cpu-multi`.
- gpu-unlock (Zagitta, GPL-2.0, RTX 30 only) lifts FP16/BF16/TF32 MMA with FP32 accumulate (48.3 vs ~25 TFLOPS),
  bit-identical; it helps prompt attention, not INT8 MMQ, CPU experts or decode (T-545). Install: `make install`,
  `gpu_unlock.targets=0000:10:00.0` on the kernel command line, check `journalctl -k -b | grep gpu_unlock`.
- Bench cases: read1k..read16k (16 new tokens), short (34-token chat, 256 new), edit (5.3K code rewrite, 512 new), long
  (41.8K prompt, 128 new); "cold" = first run in a fresh engine (later runs hit the conversation cache).
- The CPU-compute vs PCIe-promote cost per expert varies ~10x across platforms (42 vs 115 us here), so calibration must
  use window time, not a static table (PcieAuto picked 0.20-0.25 and lost). `HSA_ENABLE_SDMA=0` tips conflict with the
  measured SDMA gains here.

## 11. Where things are

- Plan: this file. Tests: TEST_LEDGER.md. Leads: LEADS.md. Agent and build rules: AGENTS.md. Measured results for
  readers: README section 2; switch census: README section 6; credits: README section 9; quants: `docs/QUANTS.md`.
- Raw evidence (kept): `bench/specs/`, `bench/results/`, `pancake-docs/research/bench/` (results archives, run scripts,
  night-queue logs), `bench/prefetch/`, `bench/quality/`, `data/fork-kb/` (fork knowledge base; refresh with
  `GITHUB_TOKEN=... python3 tools/forks/fork_kb.py`, works on Windows too; last scan 0004-20261112).
- Deleted on 2026-11-15 and folded into these three files (read them with `git show 5ab3d2db:<path>`): the old
  MOD_PLAN.md, PANCAKE_STATUS.md, the root `*_OPTIMIZATION_REPORT.md` / `OPTIMIZATION_SUMMARY.md` /
  `REMAINING_ANALYSIS.md`, `docs/PANCAKE_FINDINGS.md`, `pancake-docs/*.md`, `pancake-docs/batch2-5/`,
  `pancake-docs/research/` reports and notes, `reports/`, `research_notes/`.
