# Pancake decode critical path: code study (2026-11-09)

Repo /home/dan/clone-of-d/src/Strata, branch port/rdna3-decode (HEAD 658a6f7aa). READ-ONLY study: no edits, builds or GPU runs.
Preset studied: fast-hip (bench/results/2026-11-09-7900xt-gpuprof/gpuprof2.sh): one GPU, --spec 4 --spec-min-p 0.5,
--pcie-frac 0, --adapt-every 1 --adapt-swaps 32, STRATA_COMBINE_CPU=1 WAIT_FUSE=1 ROUTE_PUBLISH=1 EXPDEC=1 GR_FUSED=1
ADAPT_OVERLAP=1 SH_STREAM=0 QDEDUP=1, IQ2_XS pack (gate/up IQ2_XS, down Q2_0 = type 42).
"meas." = a measured number, with the file it comes from. "est." = my estimate.

**Ownership (coordinator, 2026-11-09):** session dan-99 owns items 46, 40, 53, 48, 51, 5 and the 64 logger, and is
editing src/kernels/cpu/*, src/core/verify.cpp, src/kernels/cuda/verify_kernels.cu, native_router.cu,
src/core/expert_source.cpp. Recommendations below that touch those files or items are marked **[dan-99]**: hand them
over, do not open parallel edits. The items marked [free] touch generate.cpp, mtp.cpp, native_moe.cu, pool.hpp or presets only.

## 1. One decode window as implemented (serial serve loop, one GPU)

The pipelined loop (generate.cpp ~18400-18990, `--pipeline-windows`) is off on one GPU (generate.cpp:5699-5741 needs a layer
split). The daily loop is `while (!pl_ran && ...)` at generate.cpp:19016-19554 (inside `main`, 3279-21639).

Threads: M = the main/serve thread (also the verify host thread and pool thread 0, `host_works_`); W = 15 pool workers
(pinned, spin 20 ms before sleeping, pool.hpp:292); A = a new std::thread "strata-adapt" per adapt round; GPU streams:
verify cs_, copy_ (unused at pcie-frac 0), adapt_stream, mtp cs_.

| # | Step | Where | Thread / stream | Waits |
|---|------|-------|-----------------|-------|
| 0 | Window size: choose_window, suffix lookup + DraftPolicy::choose, lookup chain | generate.cpp:19057-19140 | M | none (host only, µs) |
| 1 | keep_sync (MTP upkeep, only with --draft auto) | generate.cpp:18270, 19146 | M | cudaStreamSynchronize(mtp) if pending |
| 2 | Adaptive tier landing: ADAPT_OVERLAP flips host_res and res_upload (sync H2D copy of the residency table) | generate.cpp:19156-19180, 13011, 13020 | M | per-layer AdaptGate inside the window instead of a full wait |
| 3 | RAM_ADAPT (opt-in, off in the preset), adapt_tick (async tier, off) | 19181-19285 | M | - |
| 4 | penalties staging (hist_n>0 only): cudaMemcpy H2D, synchronous | 19288-19300 | M | yes, if penalties |
| 5 | Verifier::run: refresh_ar, capture (cached graph per T), stage_inputs, PLE rows, cudaGraphLaunch(exec_[sel][T]) | verify.cpp:3518-3660 | M → cs_ | none |
| 6 | Per layer l (48x): host spins on h_seq_ (doorbell); GPU runs pre(l): GDN/attention, HC read, router GEMV, route_publish_res (top-10 + x + ids into mapped memory + seq++), shared expert, q8_1 quantize | host loop verify.cpp:3726-3832; graph pre() 1620-2530 | M spins / cs_ | **host waits on device** (doorbell) |
| 6a | M: pool(l) = drive_pool_multi → expert_pool_dispatch_multi: usage count, plan (distinct, residency), publish plan → *h_flagA_* | generate.cpp:1426; expert_source.cpp:4906-5170, verify.cpp:4078 | M | - |
| 6b | GPU: wait_flag_copy_i32 (spins on flagA over PCIe, copies the plan), expdec_gu + swiglu_q8_1 + native_down<42> for the VRAM hits | verify.cpp:2582-2600 (post) | cs_ | **device spins on host** (flagA): meas. avg 26.4 µs/call (profiled) |
| 6c | M: activation quant per token (serial), then pool phases: run_phase(5) gate/up → quant_down_act per expert (serial on M) → run_phase(6) down, each phase = wait_parked + publish + drain + wait_done + wait_parked | expert_source.cpp:5209-5223, 5427-5432; pool.cpp:1365-1383, 1528-1560 | M + W | barriers on M |
| 6d | M: fence, raise *flag = l+1* (h_flag_) | verify.cpp:3796-3829 | M | - |
| 6e | GPU: combine_cpu_k10_vec4<true>: thread 0 of EVERY block spins on h_flag_ (10 blocks x T), then pulls each missed (token,expert) fp32 row (10 KB) from mapped h_ymiss_, adds hits, shared expert (gated), writes bo_ | native_moe.cu:145-200, 286-303; verify.cpp:2665-2670 | cs_ | **device spins on host**: meas. 38.3% of GPU time, avg 254 µs/call (profiled) |
| 7 | Head: final mixer, head mmvq (IQ4_XS head kernel), argmax into mapped m_out_ | verify.cpp:2738-2885 | cs_ | - |
| 8 | M: cudaStreamSynchronize(cs_) (blocking sync; HIP device keeps default scheduling, no ScheduleSpin), then copy_ sync if used, read h_out_ | verify.cpp:3872, 3884 | M | **host waits on device** |
| 9 | M: accept count a, grammar scan (resample_row on a refusal), stop bound | generate.cpp:19331-19380 | M | resample: sync |
| 10 | spawn adapt thread A (every adapt_iv rounds = every window with --adapt-every 1) | generate.cpp:19393-19397 | A: adapt() 13172-13360: choose swaps over 48x512, cudaMemcpyAsync on adapt_stream, event record | runs beside 11-13 |
| 11 | Verifier::commit: commit graph launch, async (commit_done_ event) | verify.cpp:4180-4238 | M → cs_ | none (set_commit_async) |
| 12 | emit tokens: printf "T %d" + fflush per window | generate.cpp:19440-19450 | M | write syscall |
| 13 | MTP draft_long: round graph, then per draft step (min_p>0 path): spin on h_out_[j], launch step graph j+1; final cudaStreamSynchronize(mtp cs_) | mtp.cpp:1781-1960 (else branch ~1880-1940) | M ↔ mtp cs_ | **host round trip per draft step**; meas. draft 2.2-2.5 ms/window (MOD_PLAN base) |
| 14 | pcie_auto_step, join adapt thread A, AdaptAsyncSelf, DraftPolicy::observe | generate.cpp:19478-19554 | M | join A |

Measured base (MOD_PLAN "Development plan", fast-hip, 7900 XT): short window 24.8 ms = GPU 15.9 + CPU experts 5.2 + draft 2.2;
edit window 37.4 ms = GPU 15.8 + CPU experts 17.4 + draft 2.5. Profile (gpuprof README + run_kernel_stats.csv, profiler slows
decode 126 -> 79 tok/s): 117,007 kernels over 86 windows = **~1,360 kernels per window, ~28 per layer**; gaps < 50 µs average
5.4 µs (profiler-inflated).

Serial structure in one sentence: per layer the GPU pre-doorbell chain → host plan → (GPU hits ‖ CPU misses) → CPU flag → GPU
row pull + combine; after layer 47 the head, a blocking sync, commit, then the draft runs on the same GPU with a host round trip
per step, while the CPU pool is idle for the whole head+draft (~3 ms/window).

## 2. Optimization opportunities, ranked by expected gain

Gains are per window, est. unless noted. Exact = output tokens unchanged.

### A. Larger

1. **Item 8/28 deferral (lossy-small) [dan-cd]** - unchanged recommendation; the code path exists (late_add verify.cpp ~1504,
   defer_ok 3004, ROUTE_PUBLISH now off under deferral at verify.cpp:2355). Ceiling +33% edits modeled (time_ceiling --overlap).
   Interaction found here: deferral makes `rpub` false (verify.cpp:2355), so it also costs the ROUTE_PUBLISH fusion
   (one launch more per layer, ~48 launches/window); fold top-10+publish for the deferred case before measuring 8 vs no-8.

2. **Overlap the MTP draft with the CPU pool / start the next window's layer-0 pre work [free, mtp.cpp/generate.cpp]** - today
   CPU workers sit idle for head + sync + commit + draft (~3 ms of a 24.8 ms short window, meas. draft 2.2 ms). Exact options:
   (a) device-side min_p gate: launch round + all step graphs at once (as the min_p<=0 branch, mtp.cpp ~1853-1880) with each step
   kernel reading prob[j-1] from device memory and exiting early when it is below min_p; removes the per-step host round trip
   (launch + wake, est. 30-60 µs per step, 2 steps → 0.3-0.5% window) - exact for the fixed-threshold rule; the chain_rule_ (host
   DraftPolicy) needs a device table form. (b) item 10 (MTP inside the verify graph) remains the bigger version (1-3% est.).
3. **Fusion of the pre-doorbell chain (item 48/72) [dan-99 for verify.cpp]** - ~28 kernels/layer; on short windows every gap
   before the doorbell is critical. Concrete candidates seen in the code/profile: (a) expdec_gu → swiglu_q8_1_entries →
   native_down<42>: SwiGLU+quantize could be the gu kernel's epilogue (4224 extra launches/86 windows = 49/window);
   (b) router GEMV (bf16_f32_mmvf_multi) + route_publish_res; (c) native_swiglu_quantize_q8_1 of the shared expert (4464 calls).
   At ~2 µs real gap each, 48-100 launches → 0.1-0.2 ms/window (0.5-1% short). Exact if arithmetic order is kept.
4. **CPU pool barrier count (item 29, FUSE_PHASES opt-in) [dan-99: src/kernels/cpu/*]** - default path per layer: 2 phases x
   (wait_parked, publish, drain, wait_done, wait_parked) + serial quant_down_act for all experts on M between phases
   (pool.cpp:1528-1560). FUSE_PHASES (mode 7, pool.cpp:1268, 1513-1527) removes one barrier pair and the serial quant (the last
   gate/up task of each expert quantizes it). Also the trailing wait_parked after the LAST phase of a layer is not needed before
   the flag is raised (only before the next begin_batch): move it to the start of the next run_phase (already there as
   "before a phase"), i.e. drop pool.cpp:1373 → saves one barrier/phase on the critical path, exact. Est. 48 x 2 x 2-5 µs =
   0.2-0.5 ms/window (1-2% short). Measure with 49's stamps first.
5. **Row pull + single poller (46/50/51/52) [dan-99]** - combine_cpu: thread 0 of every block (10 x T blocks, grid native_moe.cu:291)
   spins on the same host flag over PCIe, then each wave pulls 10 KB per missed slot from mapped memory. Est. 0.4-0.8 ms/edit
   window (MOD_PLAN 49/50 note). Code note for 51: weights are not published on the ROUTE_PUBLISH path (route_publish_res
   writes x+ids only, verify.cpp:2370); a CPU pre-sum needs w published (10 floats/token) and the down phase re-cut so a task
   owns a row range across all CPU experts of a token (pool.cpp drain mode 6) - not bit-identical.

### B. Medium/small, exact

6. **adapt thread created per window [free, generate.cpp]** - generate.cpp:19396-19397 spawns a std::thread (+ name + pin) every
   adapt_iv rounds; with --adapt-every 1 that is every window, and adapt() itself allocates `swaps/cand/vict` vectors and scans
   48x512 entries (13172-13215). The engine already has a persistent `JobThread` (generate.cpp:2125). Use one long-lived
   worker with a condition variable/eventfd; keep the vectors as members. Est. 30-80 µs/window off the commit/draft overlap and
   less jitter on M (0.1-0.3%). Exact.
7. **Software prefetch in the Q2_0 row kernel (item 40) [dan-99]** - applies to the DAILY preset: the IQ2_XS file's down rows
   are Q2_0 and run through q2_rows_any → q2_avx2_rows.inl row_multi (no _mm_prefetch). The IQ gate/up kernels already
   prefetch (iq_avx2.cpp:178-190, STRATA_IQ_PREFETCH=2048) and that measured -4% gate/up phase, -1.3% ms/round on a Zen 3 5700X3D
   (comment there). Same 2 KB-ahead hint in `rows<NT>` (row r+k). Exact.
8. **Q2_0 nt=1 dependency chain [dan-99]** - row_multi<1> updates acc[0] with two dependent FMAs per block
   (q2_avx2_rows.inl:30-31); split into acc_lo/acc_hi. ~1.4 tokens per expert group on average (native_expert.cpp comment) →
   nt=1 is the common case. Not bit-identical (sum order), deterministic. 0-3% of the down phase est.
9. **Blocking syncs on HIP without ScheduleSpin [free to measure, verify.cpp owner to change]** - the main device never gets
   hipDeviceScheduleSpin (only remote_experts.cpp:88 on CUDA; comment at 365-371 says HIP keeps its default and that a blocking
   sync's wake-up costs more than a small batch). Per window: verify end sync (verify.cpp:3872), mtp draft final sync, sampled
   head sync. Try `hipSetDeviceFlags(hipDeviceScheduleSpin)` at device init (or a spin on cudaStreamQuery only for the head tail,
   which is short). The verify.cpp comment (3864-3868) says a query-poll cost ~3% on IQ3_S (CUDA); re-measure on HIP. Est.
   20-60 µs per sync x 2-3 → 0.2-0.6%. Exact.
10. **SMT/power cost of 15 spinning workers [dan-99: pool.cpp]** - workers spin with _mm_pause for up to 20 ms
    (pool.cpp:1060-1080) through every GPU-only stretch (pre-doorbell, head, draft: most of a short window). Zen 3 has
    MONITORX/MWAITX (user mode, with a timer) - the same shape as the MSVC-only UMWAIT path (pool.cpp:43-75). Could return package
    power/boost to the host thread and the busy cores. 0-3% est., needs measurement; exact.
11. **Per-window env lookup** - verify.cpp:3953 `std::getenv("STRATA_LOGITS_DUMP")` runs every window (not static) [dan-99,
    trivial]. Note: verify.cpp:2356 `getenv("STRATA_ROUTE_PRIOR")` is inside pre(), i.e. capture time only - not hot.
12. **SwiGLU scalar exp** - pool.cpp:1302 (Q2_0 gate/up pack) and iq_avx2_rows.inl:77/342 compute `g/(1+std::exp(-g))*u` one
    element at a time. For the IQ2_XS preset it is one exp per row per token after a 2x740 B dot (~3% of gate/up est.); for a Q2_0
    pack it is a separate scalar pass over 640 x nt. Vectorized exp changes bits. Low priority [dan-99].
13. **Activation quant serial on M** - expert_source.cpp:5209-5223 quantizes each token's x on M before any worker starts
    (T x 2560 floats). Could be the first pool tasks. Tiny (few µs/layer → 0.1-0.3 ms/window est.) [dan-99].
14. **printf + fflush per window** (generate.cpp:19440-19450) - one write syscall per window, ~5-10 µs. Leave it.

### C. Lossy (decisions, not code work)
- Item 7 turbo-hip, item 23 AcceptMoE prior (verify.cpp:2410-2427, coded), item 5 tail-row drop [dan-99]: unchanged, gated by 67.
- STRATA_ROUTE_TAIL_SKIP is 0 in the preset (expert_source.cpp:4886-4897): the free lossy knob next to 5.

### D. Items 53 (ForesightSwap fill loop staleness) and the adaptive tier
- 53 is [dan-99]. Observed: the blocking tier's adapt() picks up to --adapt-swaps by gain and copies whole 1.38 MB blobs on
  adapt_stream concurrently with the next window (ADAPT_OVERLAP gates per layer, verify.cpp:3773-3777 + adapt_gate_wait 3437).
  The copy engine is idle at pcie-frac 0, so the remaining cost is the per-layer gate wait when a copy lands late; the AdaptGate
  wait time is not in the decode timing line - log it (cheap) before tuning swaps.

## 3. Code-structure issues that slow development on this path

- `main` in generate.cpp is 18,360 lines (3279-21639); the serial serve loop body alone is ~540 lines (19016-19554) with
  ~40 captured lambdas. There are three decode loops with copied logic: serve serial (19016), serve pipelined (~18400-18990),
  CLI (~21003-21330) - e.g. DraftPolicy::observe/observe_chain/observe_probs, adapt-thread spawn, window rule appear in each.
  A `DecodeWindow` class (state: p, x, drafts, policy, counters) with run_window()/commit()/draft() would let items 10, 66, 68 be
  done once.
- verify.cpp: record_window is ~1,480 lines (1416-2896) with pre()/post() lambdas of ~900 and ~360 lines; the post() CPU-share
  path has 6 variants of wait+copy+combine (device_plan, wait_fuse, fuse_wait_on, m_bal_, fuse_rows, cfuse) selected by env
  statics - most are dead for the 7900 XT preset. Every new fused kernel must thread through all variants.
- Env flags: the preset sets ~45 STRATA_* variables; many are read in function-local statics deep in hot code (pool, verify,
  native_expert.cpp has ~15). No single registry → no way to print the active configuration or to detect dead flags. A
  `flags.hpp` table (name, default, owner, status) printed at start would cut bisect time (item 8's 6-run bisect).
- Duplicated adapt landing logic: apply_pending exists twice (generate.cpp:13020 and 20748), adapt twice (13172 and 20782).
- expert_pool_dispatch_multi (expert_source.cpp:4906-5470, ~560 lines) mixes plan, PCIe fetch admission, cold-read batches,
  deferral split and timing; the plan part (publish flagA) is on the critical path and is hard to time separately (MOD_PLAN 49).

## 4. Suggested order
1. Measurement (no code risk): 49 stamps [dan-99], draft step timing (g_dt, STRATA_DRAFT_TIMING) for item 2/10, HIP ScheduleSpin
   A/B (item 9).
2. [free] item 6 persistent adapt worker; item 2a device-side min_p draft launch.
3. Hand to dan-99: pool trailing-barrier drop + FUSE_PHASES default A/B (4), Q2_0 prefetch (7), nt=1 chain (8), getenv (11),
   launch fusion list (3).
