# Phase 15 triage and source check (agent `phase15`, 2026-11-11, branch dev/p16/phase15)

What: every item of MOD_PLAN "Phase 15: speed ideas from all fork commits" (15A F1-F13, 15B F14-F18, 15C F19-F22, 15D the
measured negatives) and every NEW item of `research_notes/Active Strata forks November/` (batches 1-6) was checked against
(a) the fork commit it cites (read with `gh api repos/<fork>/commits/<sha>` for the diffs, the fork KB
`data/fork-kb/forks.sqlite` for the messages), (b) this tree, (c) the offline results already in
`bench/results/2026-11-10-offline/`, and (d) our box: Ryzen 7 5700G (Zen 3, AVX2, no AVX-512, **PCIe 3.0**: Cezanne has no
PCIe 4.0; the research notes batches 1-6 assume 4.0), DDR4, RX 7900 XT (gfx1100, wave32, 42 WGPs), `--pcie-frac 0`, MTP `--spec 4`.
Nothing here ran a model, the engine or a GPU. "Covered-by" comes from the brief and from the other agents' batch3 notes and
`git log` as of 2026-11-11 04:50 (kern-cpu, kern-hip, kern-gdn, policy, rankguard, disco-suffix, percache); I read their notes, not
their code, and cite their findings as theirs.

Verdict words: **covered** (another agent owns it or it is already in the tree), **port** (small, exact, CPU-testable: done),
**skip** (fails the >= 2% rule, is inert on our path or was measured negative), **blocked-on-GPU** (nothing more can be proven
on the CPU; the next step is named).

## Result in one paragraph

Of the 22 Phase 15 items, 12 were already coded by earlier sessions (F1, F5, F7, F9, F11-F15, F17, F21, F22; F2 and F6 are
existing flags / env). Other agents now own: F1 (policy, percache), F5 + F13 (rankguard), F7 + F12 (kern-cpu), F9 + F21 (kern-hip),
F11 (kern-gdn), F8 (disco-suffix: coded it as a DISCO model, `bf10834be`), F4 (shadowpack). Never coded and not owned: F10, F16,
F18, F19. F10 fails the plan's own offline gate (<= 3% window time, inert under `STRATA_ADAPT_KMISS`); F16 is already in the tree
in substance (EXPDEC); F18 and F19 need a GPU profile or an owner decision. **What this agent added:**
(1) `tests/core/gpu_rewrite_parity_test.cpp`: a CPU proof that the F14 4-lane Q2_0 down fold keeps the 16-lane tree's bits (4.0 M
random rows, source guard), the only 15B item whose claim can be checked without a GPU;
(2) `bench/specs/25-f-env-rerun.json`: the no-code arms the 2026-11-10 screen never measured (its base had no clean run);
(3) `STRATA_HIP_GLOBAL_WEIGHTS` (compile time, opt-in): the flat-load defect of the EXPDEC kernels, found by two research notes,
reproduced in the gfx1100 ISA and removed by a 25-site cast, device code unchanged when off (kernel wave, needs a GPU A/B);
(4) hipcc ISA evidence on `__syncthreads` (gfx1100); (5) a measured cost of the CPU activation quantize (closes ACTQ_SKIP);
(6) five source-check mismatches (below).

## 15A (both GPUs), 15C (7900 XT)

Columns: fit = sound for gfx1100 + Zen 3 + DDR4 + PCIe 3.0 (what was read, not measured).

| ID | Verdict | Covered-by / state | Source check and evidence |
| --- | --- | --- | --- |
| F1 PARK_TRUNCATE | covered | policy (A15 + F1; its notes: source check MATCH, `conversation_cache_test` 4,897 checks pass, replay -52 s) and percache (`percache_sim.py`); in tree | TwoAnts `5ef4fc8b4b` (cherry-pick of `e26560f2`): park the deepest checkpoint prefix that fits; single stage only; also shrinks an already-parked entry to one of its checkpoints. Matches the port. Offline (4.0.1): alone 39-52 s of the 162.8 s, the two config flags `--conversation-cache-mib 2048 --conversation-cache-min-tokens 8192` 158 s: config first, F1 second. Fit: host RAM policy, no GPU or link dependence. |
| F2 `--prompt-cache-spread` | covered + blocked-on-GPU | percache; flag exists (`generate.cpp:3963`), `conv_cache_test` covers the spread rule | Lamina `13d90a38ec` replaced its own retention (6 checkpoints) by main's spread rule after a 40-turn replay read 130,206 cells vs 86,218. Our tree still has the 6-checkpoint retention plus the flag, so the fork's baseline is not ours. Preset change only; the proof is TEST_PLAN 4.1.1 (agent replay, needs the engine). |
| F3 extra checkpoint cuts | skip | covered by A12 (`STRATA_AUTO_PIN`) / A15; offline 4.0.1: no effect measurable | Lamina `ab808b133e` / `83f40aee62`: the 412 -> 369 s result is six agent-benchmark tasks that ask different questions after one shared prefix (classifier / coder / critic). opencode's turns grow one conversation, which the existing last-turn and root cuts already serve. |
| F4 sub-2-bit shadow pack | covered | shadowpack | svgop `eeca072bb6`/`25bf1b90f6` (+3-8% solo on a 5090). Offline 4.0.5: gate/up IQ1_M -18.2% CPU bytes, +17.6 GB RAM on a 57 GB box that already pins a 33 GiB arena: owner's RAM decision. Not touched. |
| F5 `STRATA_VERIFY_ECAP` | covered (rankguard), blocked-on-GPU (lossy) | coded; `verify_ecap_test` is in compilegate's CPU tests | **Mismatch:** jmnargi `24894e676b` caps the WHOLE window's expert set (token 0 keeps all K picks, the tail keeps the top N-K by weight, dropped picks join a kept expert at weight 0; GPU and CPU alike; 2x V100: +2.8% pooled, +11.5% drift-paired, tool-call gate broke 12 of 20 tool turns at a tight cap). Our port caps only the CPU-missed experts of draft rows, which is our actual cost. Different mechanism: the fork's numbers do not transfer; the screen + token forcing + task gate decide. |
| F6 exact speculative sampling | blocked-on-GPU | env-only; switches present (`STRATA_SPEC_PROB`, `_COUPLED`, `_PROB_GATE`, `_PROB_DT`) | riceharvest `7389a10f2c` (SYCL, B70, +5.5%), svgop `0355c3164f` (+5.8%, +13.4% at DT 0.85). Applies: opencode sends no temperature, the main agent runs at T 1.0 (offline 4.0.2), acceptance 0.69 vs 0.76. Spec `12-spec-sampling.json` exists. |
| F7 Q2_0 AVX2 kernel | covered | kern-cpu (`3bafb6ab9`, notes in its batch3 file) | imanu86 `c3fc8f34d3`. Zen 3 AVX2 is the target ISA. Offline microbench (`q2_rows_bench`): rows x1.33 / x1.23, bit-exact. |
| F8 `STRATA_SPEC_CUM` | covered (coded) | disco-suffix `bf10834be`: `STRATA_SPEC_CUM=s` as a DISCO model on the chain feature, unit tests incl. exactness, replay arms in `disco_fit.py`, screen spec `b3-f8-spec-cum.json`, `ecospec_sim.py` | android-imdad `f1bd09212a`: stop a draft chain when the product of the draft layer's probabilities falls below S; +1.8% on a 7900 XTX at `--spec-min-p 0.5` (104.09 -> 105.94). `ef42c2fde9` adds the suffix-lookup agreement as a feature (replay +3.9%, measured +1.4% / +5.8%). Our ceiling (4.0.3): oracle 1.0% short, 7.9% edit; no window trace with draft records exists yet. **Input for EcoSpec / DISCO that I checked in the fork:** the draft layer is calibrated at depth 1 and overconfident deeper (2nd drafts at p 0.8-0.9 held 57%), so P(path) must be calibrated per depth and probability band (`STRATA_SPEC_CAL`: 8 depths x 8 bands, decayed counts, prior weight 4), not multiplied raw. Nothing coded here (disco-suffix owns the window rule). |
| F9 router publish | covered | kern-hip (`bc3a48c41`, spec 17 on the fast-hip base) | Ben1078 `23d2fc0450`, android-imdad `caa9b00115`. |
| F10 `STRATA_ADAPT_PROMPT` | skip | overlaps A10 (seeding, in tree) | zsariboga `4ce19344ff` is a 71-line patch: a global pointer from prefill into `drive.d.usage`, prompt-chunk counts added with weight V/T. Offline 4.0.4: seeding with the request's own counts is +2.9 hit points = ~0.8 ms of a 24.8 ms window (3%) at best, only until the tier adapts, and inert under `STRATA_ADAPT_KMISS` (item 84: seeds only raise usage, k-th miss needs real misses). No prefill routing dump exists to replay it. Below the 2% gate. |
| F11 `STRATA_ATTN_WINDOW` | covered (kern-gdn), blocked-on-GPU | coded (q36 engine); kern-gdn: `q36_engine.cu` builds for gfx1100, 0 errors | Asofwar's kernel does not rebase (plan: dense int8 prompt kernel missing, Flash-Next has the indexer, GQA 12 not 8); the port is the q36 engine's `k_attn_window`, designed bitwise. Plan's own bound for Flash-Next: ~1% (QSA reads <= 2,051 cells, attention 3% of a window). The q36 engine is the 7900 XT's Qwen3.6 path, so this is F19's route (b). Needs `--dump-full-logits` cmp on a GPU. |
| F12 sign form | covered (kern-cpu), blocked-on-GPU | kern-cpu `f9e0d5644`: the sign step moved into one host+device header (`iq_signs.hpp`), exhaustive host test against the real codebooks (`iq_sign_fast_test`), device code of `iq_kernels.cu` byte-identical (their disassembly check) | RX5950XT `b9a2111d3f` says "not measured on an AMD card ... 17-19 of 112 kernels lose waves (native_gu_multi 16 -> 10-12)", so limiting the port to the EXPDEC gate/up kernels is right. boxwrench's -29% / -35% (`yqYo1 6f0bdf665b`) is a different, packed-sign form in a GLM engine. I first wrote a host proof of the formula (20.1 M cases, formula = the one in the kernel) and dropped it: kern-cpu's test is stronger and its header move would have broken my source guard (`0c930030c`). Expected gain 0-1.5% (plan). |
| F13 route-prior guards | covered (rankguard: source check ok), blocked-on-GPU (lossy) | coded; `route_prior_scope_test` in compilegate's CPU run | zsariboga `8d7042ad97`: `PROTECT` keeps the router's first N picks, `KEEP_P` keeps picks with p >= x of the top pick; port matches. Caution from lighttransport `2ea0c1e042`: residency-aware pruning and cache-aware substitution "do not beat uniform top-3" on their gate; spec 19 compares turbo to a uniform tail skip at equal KL for exactly that reason. |
| F19 Qwen3.6 on the XT | blocked-on-GPU (owner decision) | design only | Route (b), the in-tree q36 engine with F11's `k_attn_window` (exact, supersedes storm-ace's `k_attn_part_gqa`), is the cheap one: strata-poly is CUDA-only and a HIP prompt kernel is a 150-250 line rewrite plus a build. gfx1151 numbers (storm-ace) are not gfx1100 numbers. |
| F20 empty-launch probe | covered | probe exists (`tools/pancake/probes/empty_grid.hip.cpp`, `launch_shapes.py`, CMake `empty_grid_probe`); needs the GPU | unimatrix `d0c8d6c3e2`: a 384 x 256 launch (the author's "48 CUs" are the 7900 XTX's 48 WGPs) costs 14-16 us empty, 352 blocks or fewer 2.7-3.0 us: a cliff, not a slope. A 7900 XT has 42 WGPs, so the cliff is probably at another block count; the probe sweeps it. The fork's concrete fix (PCIe-share copy at 96 blocks) is moot here: `verify.cpp` skips the fetch under `--pcie-frac 0` (`if (!pcie_off_)`). |
| F21 WAIT_COPY256, ADAPT_COPY2 | covered | kern-hip (`424b3fce0`, `hip_wait_copy` test, spec 25-kern-hip) | android-imdad `caa9b00115` (4,198-line diff). `STRATA_ADAPT_COPY2` is not in the tree (0 files); +0.2-0.5% claims. |
| F22 `STRATA_HIP_WMMA_Q4` on gfx11 | blocked-on-GPU, low value | coded | bsorensen110 `19fc0e48` is gfx12 (R9700) and its speed number is gfx1200 only; the gfx11 layout argument in the plan is code reading. The prize is VRAM, not speed: 180 MiB = ~131 expert slots = +1.2% slots, paid with `--kv q4_0` (document PPL +8% at 1K, +12% at 8K). Recommend skip unless Q4 KV is wanted for other reasons. |

## 15B (RTX 3060 first): design notes and CPU-side checks only

| ID | Verdict | State | Check |
| --- | --- | --- | --- |
| F14 EXP_MODE=9, EXP9_DOWN=42:10 | blocked-on-GPU (3060) | coded; **CPU parity test added (`ae2814a3d`, narrowed in `0c930030c`)** | Tree equivalence derived by hand from both kernels and then proven numerically: mode 0's lane t of 16 sums call t (+ call t+16 for t<4), joins, xor 8, 4, 2, 1; the 4-lane kernel gives lane u calls u, u+4, ..., u+16, adds `(a0+a8)+(a4+a12)`, then xor 2, 1: the same additions on the same values. 4,000,000 random rows (zeros, -0, subnormals, cancellation) bitwise equal; a straight 0..19 sum differs on 53% of rows (the test can fail); mutated pairing fails the test. gfx1100: applies on the same terms as `STRATA_EXPDEC_DOWN` (plan), 0-1% expected. |
| F15 FETCH_ADMIT | skip (alias done) | `STRATA_FETCH_ADMIT=1` reads as `STRATA_ADAPT_FETCH=1` | AdrianBM96 `50581c24f2` (+5.1/+7.0/+5.1% on a 3060): in-tree `ADAPT_FETCH` is the same idea. Inert on the 7900 XT presets (`--pcie-frac 0`), off under `--adapt-async` (the 3060 preset has it on): the 3060 A/B must run `--adapt-async 0`. |
| F16 Lamina sub-warp grouped expert kernels | skip | in substance already here | Lamina `674bf0caf2`: rows on sub-warps, each weight decoded once for several entries, Q2_0 down on 4 lanes, IQ4_NL on 8. This tree has the 16-lane `row_dot_*_sub16` family and EXPDEC / EXPDEC_DOWN ("an activation word serves 4 rows", +1.7% measured); the 4-lane Q2_0 down is F14. The fork's numbers are 3090 / 5070 Ti with the L2 flushed. |
| F17 poly FATTN / FDEC | blocked-on-GPU (3060) | coded; `tools/pancake/poly_kl.py --selftest` passes (run here) | Plan's own estimate: FATTN prompt +20-35% at 32-128K, FDEC 0-8% (the preset already runs `--attn-split`). |
| F18 CUDA prefill fusions | blocked-on-GPU | not coded | zsariboga `21107e9d`: dense IQ4_XS / Q8_0 -> FP16 with 16-byte stores (dequant 141 -> 11 ms of a 4.8 s 32K prompt = -2.7%), swiglu + q8_1 and combine + hyper-connection fusions (-30 + -23 ms) are CUDA-only. The dequant kernel is plain and portable; whether our HIP prompt path spends ~3% in a dequant like that is unknown (the 2026-11-09 rocprof run is decode only). Next: one rocprofv3 table of a 32K prompt; port the 16-byte-store dequant only if `dequant` >= 3% of prefill. |

## 15D (measured negatives): nothing proposed repeats one

PF_AHEAD (batch 4) is next-layer expert prefetch for the STAGED PREFILL walk; the negative in 15D is next-layer prefetch in
DECODE (-3..-4%). They are different paths, but the PCIe-3.0 fork (0xluk, 2x RTX 3060) is the only datapoint and it bundles four
patches, so it stays blocked-on-GPU. `STRATA_POOL_DAG` / `CPU_POOL_COST` were not built: see the table below. `--pcie-frac` items are
inert on the XT (15D: "PCIe share on the 7900 XT (off is faster)").

## research_notes/Active Strata forks November/ (items marked NEW there)

| Batch / item | Verdict | Evidence |
| --- | --- | --- |
| 1 `POOL_DAG=1/2` (KhaledElzarw) | skip | In-tree `STRATA_POOL_FUSE_PHASES` mode 7 is the same structure (a down task waits only for its own expert) and is in the preset (+2.5% edit). The fork author did not adopt DAG (bundle 1.011, CI 0.997-1.025). Our pool splits `n * FF` rows into equal tasks over all workers (`ExpertPool::run_split_multi`, `phase_tasks`), so the tail imbalance a longest-first order (dys0re `CPU_POOL_COST`, batch 2) fixes does not exist here. |
| 1 LDS-only barrier (dardeaw `580e251e69`) | blocked-on-GPU, low | **ISA check done (gfx1100, hipcc 10.0).** `__syncthreads()` = `s_waitcnt lgkmcnt(0); s_barrier; buffer_gl0_inv`; a `vmcnt(0)` (drain of loads in flight) is added only when loads are in flight at the barrier. Of 1,022 barriers in `native_mmvq.cu` 927 wait lgkmcnt only, 85 vmcnt(0), 10 vscnt; of 72 in `qsa_decode_attn.cu` 45 lgkmcnt only, 24 vscnt(0) (a wait for global stores), 0 vmcnt(0). The commit's rationale (drain + L0 invalidate) was measured on gfx12; on gfx11 it is weaker (8% of MMVQ barriers). Port is small but the gain is the `buffer_gl0_inv` and the 24 store waits: unproven, needs rocprof. |
| 1 `STRATA_ACTQ_SKIP` | skip | Sound (only `kind == -1` entries build jobs that read `act_multi[t]`, `expert_source.cpp:5330-5436`) but measured small: `act_quant_q8_1_avx2` at H = 2560 costs **0.94 us** per token on this Zen 3 (warm L2). A window quantizes <= T x 48 tokens = 5 x 48 x 0.94 = 0.23 ms (0.6-0.9% of a 25-37 ms window) and only tokens with no CPU entry skip: ~35% at a 90% hit rate -> 0.1-0.3%. Below the 2% gate. |
| 1 prompt-cache root fix (ElXreno `a18d4668ec`) | skip | Bug is present (`generate.cpp:19092-19096`: the scan breaks at the FIRST turn token and sets the root only if it is >= `--prompt-cache-root`), but it bites only for system prompts under 2,048 tokens; opencode's is ~10K. percache's area. |
| 1 kv_stream guard, two-pass restore | blocked-on-GPU / covered | kv_stream bound check is a GPU kernel hardening (CUDA-validated only); two-pass session restore belongs to the disk tier (percache/datapath). |
| 2 `COPY_WAIT_DEDUPE`, `ROUTED_STREAM` | blocked-on-GPU | Switches exist, off. **The 2026-11-10 05s screen that "dropped" them is not a measurement** (see mismatches); re-run in `bench/specs/25-f-env-rerun.json`. |
| 2 `STRATA_FUSE_DECODE` on HIP | skip | 4 of the 7 gates in `layer.cpp` sit inside `if (!w->native_data)` (canonical Q8 images) and the 3 `project_bf16_pair` sites run two `bf16_gemv_fp32_mmvf` calls under `native_bf16_projections`: with `--native` weights nothing fuses. |
| 2 uniform-K expert reduction (serboox) | covered | lossy lever; same family as `STRATA_ROUTE_TAIL_SKIP` / turbo / VERIFY_ECAP, which the task gate already orders. |
| 3 `EXPDEC_V2`, `FUSE_*` launch fusions, one-token `MMVF_ROWS` | blocked-on-GPU, **kernel wave** | Same root defect as batch 6 F1 (flat loads), see below. Fusions: partly in the tree already (`STRATA_COMBINE_CPU`, `STRATA_WAIT_FUSE`, F9's `route_publish_fast` = FUSE_ROUTEBELL). Our decode is CPU-bound (combine_cpu_k10_vec4 = 38% of GPU time), so -300 launches only pays where the GPU stream is the critical path. |
| 3 `DPP_XOR` | skip (have) | `__shfl_xor_sync` is already DPP on gfx11 (`89da64b93`, `STRATA_DPP_SHFL`). |
| 4 `PF_AHEAD`, `REFILL_ASYNC`, `PARK_REWIND_MAX`, `CKPT_MOVE_PARK` | blocked-on-GPU / percache | The only PCIe 3.0 datapoint (0xluk, +35% on agent turns) bundles all four. Our PCIe is 3.0, not 4.0 as the note says, so the copy cost is the fork's, not half of it. |
| 4 `MMVQ_R` table | skip | `STRATA_MMVQ_R=2` on top of `STRATA_RDNA_MMVQ_WAVE=3` swaps the exact-emulation kernel for the plain one (rounding change); the gfx906 table does not transfer to wave32. |
| 5 `LOOKAHEAD` off when the complement is whole (Spark-NV `0b88bd1a66`) | blocked-on-GPU | **Strongest zero-code candidate.** Fork: decode window 17.4 -> 15.8 ms, 162 -> 183 tok/s (5090 + 7800X3D). Our tree detaches the look-ahead only when `gpu_holds_all` (`generate.cpp:9420-9470`), our log shows it on while the 30.5 GiB complement is pinned, so it warms GGUF pages the decode never reads and takes DRAM bandwidth from a bandwidth-bound pool. Arm `lookahead0` is in `25-f-env-rerun.json`; port the `complement_whole()` gate (~25 lines) only if it wins. |
| 5 hugetlb complement, `copy_rows_from_mapped` loads-before-stores | blocked-on-GPU | owner decision (reserved pool, pool-blind RAM budget: reserving `vm.nr_hugepages` without the budget fix caps the complement); the copy kernel gain is ~0.1-0.3%. |
| 6 global-address hint (Lamina `33f63cf1f5`), sampler x8 unroll, `gdn_gate` latent bug | blocked-on-GPU, **kernel wave** | See the flat-load finding below. `gdn_gate_kernel` (`elementwise.cu:43`, `if (i >= h_v) return;` with n_tokens x h_v threads) is a real latent bug for n_tokens > 1; every caller passes 1, so it is unreachable today. |
| 6 temperature < 1e-30 -> greedy (Lamina `6c243c1069`) | not checked | the server sanitizer was not read (`serve/server.py` rejects `temperature` outside 0..2 only). |

## Source-check mismatches (what the plan or the ledger says vs what the source shows)

1. **The 05s screen is not evidence.** MOD_PLAN's "closed" row (small-flag stack, LOOKAHEAD=0, routed-dedupe, MMVF_ROWS, GDN_PP=2,
   spec-min-p 0.7: "meas. screen: dropped") rests on `bench/results/2026-11-10-test-phase/05s-screen.summary.txt`: base "no clean
   runs", every arm n = 2, intervals of +-100 to +-450 tok/s (lookahead0 short 108.7 +- 293.6), and the harness verdict "drop" for
   every arm, including `lookup12`, which 05c then confirmed at +8.8%. Only the stack (-20%) is a result. Re-open the single-flag arms
   (spec `25-f-env-rerun.json`).
2. **F5 is not the fork's mechanism** (whole-window cap on both GPUs vs a CPU-missed-expert cap for draft rows); the fork's gain
   and its tool-call damage do not transfer either way.
3. **PCIe 3.0, not 4.0** (5700G). Research-note batches 1-6 assume 4.0 in several places (PF_AHEAD, REFILL_ASYNC). The 0xluk fork
   (PCIe 3.0) is the closer datapoint, and F15/F17-style link arguments should use 3.0.
4. **`STRATA_FUSE_DECODE` on HIP would be inert** here (above); the research note counted it as a 7-line port with a gain.
5. **LDS-only barrier:** the gfx12 rationale (vmcnt drain + L0 invalidate on every `__syncthreads`) holds only partly on gfx1100
   (ISA counts above); the fork extended its gfx12 guard to gfx11 without a gfx11 number.

## Flat loads in the EXPDEC kernels (kernel wave; evidence from the compile only)

Two research notes (batch 3 F1 `EXPDEC_V2`: "read through `grp_ptr` they compiled to flat loads, and flat loads count in the LDS
counter"; batch 6 F1 Lamina `33f63cf1f5`: `__builtin_assume(__isGlobal(...))`) describe one defect: a weight pointer loaded from a table
is a generic pointer, so the compiler emits `flat_load`, which on RDNA3 counts against both vmcnt and lgkmcnt: every LDS wait then
also waits for the weight reads in flight. 

Reproduced on this tree (hipcc 10.0, `--offload-arch=gfx1100 -O3 -S --cuda-device-only`, `iq_kernels.cu`): 13,119 `flat_load` against
16,593 `global_load`; the EXPDEC gate/up kernels (`expdec_gu_kernel`, `expdec_gu_swiglu_kernel`, IQ2_S 22, IQ3_S 21 at 200, IQ2_XXS 16
at 40-80) and `expdec_down_kernel<20/42,*>` (10-136) issue 6 global loads and 40-200 flat ones, i.e. essentially every weight
read of the preset's kernels (`STRATA_EXPDEC=1`, `STRATA_EXPDEC_DOWN=1`) is flat; `native_gu_multi_kernel<144/145>` 254-267.

Fix prototype (scratch, then committed behind a macro): the 25 sites `(const uint8_t*) grp_ptr[g]` read through a
`const __attribute__((address_space(1))) uint8_t*` instead. Result: flat_load 13,119 -> 108, global_load 16,593 -> 29,538 (the
EXPDEC kernels 200 -> 0 flat, 6 -> 206 global; `native_gu_multi` 254-267 -> 2-3). Of 572 kernels in the object, VGPR count and
occupancy are unchanged for the EXPDEC family except: 3 kernels gain scratch (`expdec_gu_swiglu_kernel<22,1,2,true>` 0 -> 52 B,
VGPR 250 -> 256), 2 lose one wave of occupancy (`<22,1,2,false>` 8 -> 7, VGPR 178 -> 208), 3 gain one. Lamina excluded IQ3_S from
its hint for a spill on sm_120; here the single-entry IQ2_S variants are the ones to watch in a GPU A/B.

This is `STRATA_HIP_GLOBAL_WEIGHTS` in `src/kernels/cuda/iq_kernels.cu` (macro `STRATA_GBLOB`): compile time because it is a code
generation change, off by default, HIP and not gfx906; build the A/B arm with `cmake -DCMAKE_HIP_FLAGS=-DSTRATA_HIP_GLOBAL_WEIGHTS=1`.
It is the same defect `EXPDEC_V2` (this-cute-world `a67b3790d5`) fixes among other things (it reports -20..-25% per kernel on
gfx1201); our profile gives those kernels ~13% of GPU kernel time and the decode window is CPU-bound (combine_cpu 38%), so the
window gain is bounded by a fraction of that: expect low single digits, measured on a GPU only. Check order: (1)
`iq_multi_parity` / `native_grouped_parity` bitwise with the macro on, (2) `expdec_bench` per-kernel times, (3) the screen.
Default path check: the device assembly of `iq_kernels.cu` for gfx1100 with the macro unset is identical to the one of the unmodified
file (`diff`: 18 lines, all the `__hip_cuid_*` symbol, which is a hash of the source text); with the macro set see the counts above
(re-measured on the committed form: flat_load 13,119 -> 108, global_load 16,593 -> 29,538, per-kernel counts equal to the scratch prototype, no compile error).

## Tests run and result

| Command | Result |
| --- | --- |
| `cmake -S . -B build-cpu -G Ninja -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON`; `ninja -j2 gpu_rewrite_parity_test`; `ctest -L cpu -R gpu_rewrite_parity_test` | PASS (0.9 s): F14 4,000,000 random rows bitwise equal to the 16-lane tree, source guard |
| mutation (scratch copy): F14 pairing `(a0+a4)+(a8+a12)` | FAILS, as it must (the first version's F12 mutation `without + c` also failed before that part was dropped) |
| `python3 tools/pancake/poly_kl.py --selftest` | `poly_kl selftest: ok` |
| `hipcc --offload-arch=gfx1100 -S` of a 3-barrier kernel, of `qsa_decode_attn.cu`, `native_mmvq.cu`, `iq_kernels.cu` (scratch, device only) | asm inspected, counts above |
| scratch microbench `act_quant_q8_1_avx2`, H = 2560, best of 7 | 0.935 / 0.937 us per call (5700G) |
| `cmake -S . -B build-hip ... -DCMAKE_HIP_ARCHITECTURES=gfx1100` (configure only) | OK, 132 cpu + 122 gpu tests registered |

Not run: anything that loads a model or touches a GPU; the HIP/CUDA parity binaries; `abtest.py` (the new spec is JSON only).
Note for the CPU build: `src/kernels/cpu/expert.cpp` is compiled with `-mavx512*` in the default CPU configuration, so its generic
`act_quant_q8_1` raises SIGILL on the 5700G; run the `_avx2` entry points (what the engine's `act_quant_any` selects there).

## Risks and next

- The test models transcribe the kernels' arithmetic; the text guard fails if the lines change, the GPU parity runs
  (`native_expert_bench`, `iq_multi_parity`) stay the proof for the compiled code.
- GPU session order suggested by this check: (1) `25-f-env-rerun.json` (lookahead0 first, it has a mechanism), (2) a 32K-prompt rocprofv3 table
  (decides F18 and the prompt-side items), (3) `empty_grid_probe` (F20), (4) kernel wave: global-pointer fix in the EXPDEC kernels,
  then F12/F14 arms.
- Left unbuilt on purpose: F10 (gate), the LDS barrier, FUSE_DECODE on HIP. `STRATA_HIP_GLOBAL_WEIGHTS` touches 25 lines of `iq_kernels.cu`, which kern-cpu (F12 header) and kern-hip (A37) also edit: if it conflicts on merge, redo it with `sed -i 's/(const uint8_t\*) grp_ptr\[g\]/STRATA_GBLOB(grp_ptr[g])/'` plus the macro block after the `tq_lut_i8.cuh` include.

## Commits (branch dev/p16/phase15, on dev/phase16-academic 3f4ba97cf)

| Commit | What |
| --- | --- |
| `ae2814a3d` | tests: host proofs for F12 and F14 (first version) |
| `8ae14de73` | `bench/specs/25-f-env-rerun.json` |
| `0c930030c` | tests: narrowed to F14 (F12 moved to kern-cpu's `iq_sign_fast_test`) |
| `823e77f78` | `STRATA_HIP_GLOBAL_WEIGHTS` in `iq_kernels.cu` |
| (this commit) | this note |
