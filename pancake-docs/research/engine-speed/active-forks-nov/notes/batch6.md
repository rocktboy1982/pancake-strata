# Batch 6: active Strata forks (own commits since 2026-11-01), state 2026-11-10

Target: pancake-strata on RX 7900 XT (gfx1100, wave32) + Ryzen 7 5700G, Flash-Next 125B IQ2_XS, preset `pancake-flashnext-iq2_xs-fast-hip.json` (`--pcie-frac 0`, `--spec 4` MTP, `STRATA_EXPDEC=1`, no `--spec-split`, no `--pipeline-windows`). Tree checked: /home/dan/clone-of-d/src/Strata at 39f7a3da3. All speed numbers below are **author-measured** on the author's hardware unless marked "[our ISA check]" (a disassembly of our own build-hip objects done for this note, not a speed measurement).

## Which commits in this batch could raise decode / prompt speed or cut start time on our box?

### Takeaway
Two small, exact ports are worth an A/B. (1) **Lamina's global-address hint for expert weight pointers.** Our gfx1100 build still issues `flat_load` for expert weights in the decode gate/up kernel that fast-hip runs (`expdec_gu_swiglu_kernel`) and in `native_gu_multi_kernel`. (2) **rpcribari's DPP/ds_swizzle `__shfl_xor` plus the eight-loads-in-flight sampler/verify loops.** Neither will be a big jump. Each is plausibly 0.5-5% on decode, and both are bit-exact. Everything else in the batch is ALREADY-HAVE, applies only to the q36 (Qwen3.6-35B-A3B) path, is a latent bug fix, or doesn't apply.

### Cited Findings

**F1. Expert weights read as global, not flat memory (NEW, exact, port S)**
- Fork: eddoursul/Lamina, commit 33f63cf1f5 (2026-11-02). The grouped expert kernels reached an expert's weights through a pointer loaded from a table (`grp_ptr[g]`), so the compiler emitted generic loads. `__builtin_assume(__isGlobal(wg/wr))` makes them global loads (LDG). It was added to `native_gus*_kernel` gate/up and `native_down_kernel`, but skipped for IQ3_S (fmt 21) gate/up, where it spilled on sm_120 and ran 4-19% slower — [commit](https://github.com/eddoursul/Lamina/commit/33f63cf1f5)
- Author-measured (expert_grouped_parity --bench, µs, RTX 3090 / 5070 Ti / 5060 Ti): TQP formats 4-9% less time. **IQ2_XS, IQ3_XXS, IQ2_S and IQ4_XS were 0-9% less** on the 5070 Ti and 5060 Ti. K-quants moved ±2-3%. On the 3090, the other i-quants were 1-5% faster at T2-T4 and 1-3% slower at T8. Bitwise parity: 0 of 460 products differ — [commit](https://github.com/eddoursul/Lamina/commit/33f63cf1f5)
- [our ISA check] Disassembly of the gfx1100 code object in `build-hip/.../iq_kernels.cu.o`: 10,364 `flat_load` vs 16,357 `global_load`. The kernels with the most flat loads are `native_gu_multi_kernel<144/145,...>` (254-267 each) and **`expdec_gu_swiglu_kernel<21/22,2,2|4>` (200 each)**. That second one is the kernel `STRATA_EXPDEC=1` in fast-hip uses. `s2_expert_grouped` has 64 flat vs 224 global, and `native_mmvq` has 0 flat.
- Why it can matter on RDNA3 (inference, not measured): a flat load counts against both vmcnt and lgkmcnt, so `s_waitcnt lgkmcnt` waits serialize LDS traffic behind VMEM. It also can't use the SGPR-base `saddr` form, so it takes more VGPRs.
- HIP spelling: `__builtin_assume(__builtin_amdgcn_is_shared(p) == false && __builtin_amdgcn_is_private(p) == false)`, or a cast to `const __attribute__((address_space(1))) uint8_t*`. Verify in the disassembly that the `flat_load` count drops.
- Applicability: gfx1100 yes. Port effort S (a few lines per kernel plus an ISA recount). Watch VGPR/spill counts per format, as the author did for IQ3_S.
- Suggested A/B: compile-time variant, or env-guarded template flag `STRATA_EXP_GLOBAL_HINT=1`, versus base. Run short chat / 2K code edit for decode (EXPDEC path) and 11.7K prompt (prefill GEMM unaffected unless the same pointers are used). Gate on token-forcing parity.

**F2. `__shfl_xor_sync` with a constant mask as DPP / ds_swizzle (NEW, exact, port S)**
- Fork: rpcribari/Strata, commit 9a2cfb6d77 (2026-11-04). In `include/strata/platform/hip_compat/strata_hip.h`, `__shfl_xor_sync` is routed to `strata_shfl_xor`. For a compile-time 32-bit mask below the width, it uses `__builtin_amdgcn_mov_dpp` quad_perm (masks 1, 2) or `__builtin_amdgcn_ds_swizzle` bitmask mode (masks 4, 8, 16). Anything else falls back to `__shfl_xor`. Bit-identical. The author saw ~4 µs per token in the hyper-connection read from ds_bpermute alone (gfx906 trace) — [commit](https://github.com/rpcribari/Strata/commit/9a2cfb6d77)
- Our tree: `strata_hip.h:200` still maps it straight to `__shfl_xor`. Our own `qsa_decode_attn.cu:478` already uses DPP `row_xmask` locally. ROCm 10's `__shfl_xor` goes through `__builtin_amdgcn_wave_shuffle` (or ds_bpermute) (`/opt/rocm/core-10.0/include/hip/amd_detail/amd_warp_functions.h:521`).
- [our ISA check] The gfx1100 objects still contain many `ds_bpermute`: router_top10 42, sampler 82, verify_kernels 42, next to DPP/swizzle the compiler already chose in some places. So the compiler does not convert all of them.
- gfx1100 note (inference): the DPP quad_perm and ds_swizzle bitmask forms are valid in wave32. On gfx10+, `row_xmask:M` DPP (as in our qsa kernel) covers masks 1-15 with no LDS at all, so it may beat ds_swizzle for 4 and 8. Mask 16 needs `v_permlanex16` or ds_swizzle.
- Effort S. A/B: rebuild with the macro change. Workloads short chat / 2K edit decode, plus 11.7K prompt. Expect a small gain (≤1-2%, inference).

**F3. Sampler argmax and verify row_top_prob: eight logit loads in flight (NEW, exact, port S)**
- rpcribari/Strata commit a03833d8f1: `sampler_greedy_kernel` and `row_top_prob_kernel` unrolled ×8 with the same order, the same strict `>` and the same sum order. Author-measured on gfx906: each round of a 248,320-logit row was latency-bound, about 110 µs for the target argmax — [commit](https://github.com/rpcribari/Strata/commit/a03833d8f1)
- Our tree has the old single-load loops: `src/kernels/cuda/sampler.cu:161` (plus similar loops at 355 and 604) and `src/kernels/cuda/verify_kernels.cu:713`.
- Applicability: gfx1100 yes. The gain is smaller than on gfx906 because of higher clocks and better latency hiding, but these kernels run on every verify window (spec 4). Effort S. A/B on short chat decode, greedy and with temperature.

**F4. BF16 router GEMV one-token path through the rows kernel, 2 rows/block (NEW but marginal, exact)**
- rpcribari/Strata commit 6b0f0f8448. Under `STRATA_MMVF_ROWS=1`, a one-token GEMV with ≥64 rows takes the rows kernel, and RPB goes 4→2. This works around a gfx906 command-processor stall seen with ~509-512 blocks. Router GEMV ~46 → ~11 µs, long10k 70.9 → 71.6 tok/s on 2× MI50 (author-measured) — [commit](https://github.com/rpcribari/Strata/commit/6b0f0f8448)
- Our tree: `STRATA_MMVF_ROWS=1` is already a HIP default (`src/core/arch_defaults.cpp:80`, S25), but n_tok==1 still takes the per-row kernel. With MTP most decode steps have n_tok>1, so this applies only to non-draft steps. The 512-block stall is gfx906-specific and not known on gfx1100. Low priority. A/B only if someone touches native_bf16.cu anyway.

**F5. Start-time idea: READY right after the dense weights, experts filled in the background (NEW idea, port L)**
- qwertie/Strataq commit 64cd648a1e: the paged engines (strata-q38/ds41) report READY after the dense weights load, with empty expert pools filled between and during requests (GPU first, then RAM). The adaptive tier takes free pages before swapping. READY in about 9 s instead of 38 s (Qwen3.8 IQ3_S, Strix Halo, author-measured) — [commit](https://github.com/qwertie/Strataq/commit/64cd648a1e). The follow-up 81c91ed684 lets usage, rather than the profile, move experts after the start-up fill — [commit](https://github.com/qwertie/Strataq/commit/81c91ed684)
- Applicability: this is a separate paged engine (new tree layout, `src/common/experts/*`) and not portable code. As an idea for us, decode would run slow, with misses served from a cold arena, until the fill completes. Our MOD_PLAN item 69 / 85 cover start time and THP. Effort L. Only worth it if start time becomes a priority.

**F6. q36 path (Qwen3.6-35B-A3B, our second model) — bahooo22/Strata**
- bbc034d48a: `att_rows = H * kGemmT` instead of `H * min(T,64)`, because split-K runs only for T≤8 and the 64 rows just held VRAM. Also an L2-friendly grid `dim3(nt*H, nc)`, so the q-heads of one KV head are adjacent. At 96k, tg went 18.3 → 23.5 (+28%) and pp 301 → 326, because freed VRAM gave 23 expert slots instead of 17. No gain at 128k. Author-measured, NVIDIA (Windows) — [commit](https://github.com/bahooo22/Strata/commit/bbc034d48a). **Our tree still has the old code** (`src/qwen35/q36_engine.cu:1429`, `:1994`). NEW for q36 only, exact, effort S. A/B with Qwen3.6 at long context.
- 4569b243f7: `k_route_count` + `k_route_scan` merged into one block, with no memset, saving 2 launches per layer in prefill. Bitwise equal. At 96k, pp 562-565 → 567-569 (author-measured) — [commit](https://github.com/bahooo22/Strata/commit/4569b243f7). Our tree has the old pair (`q36_engine.cu:473-477, 2026-2027`). NEW, q36 only, S, tiny.
- 4b679a76b5 / 53a8f8cab5: per-layer `cudaHostAlloc` instead of one 16 GiB allocation, because the big one failed on Windows and fell back to pageable. 96k pp +73%, but tg −10% — [commit](https://github.com/bahooo22/Strata/commit/4b679a76b5). Ours is one allocation (`q36_engine.cu:2929`). On Linux the single pin normally succeeds, so this is NOT-APPLICABLE unless our log shows the q36 arena as pageable.
- 36dddeb104 + fixes 39e8f0a7ee, **590161ccf3 (critical: residual x carried into xsuper)**: q36 super-chunk layer-major prefill — [commit](https://github.com/bahooo22/Strata/commit/36dddeb104), [fix](https://github.com/bahooo22/Strata/commit/590161ccf3). Already listed in the earlier report as PR #5468 (ALREADY-COVERED). Anyone porting it must also take 590161ccf3.
- a36d7f53a6: docs say prefill chunk 2048 gives up to ×3.8 prefill at 64k-96k, because each chunk re-copies all layers' experts — [commit](https://github.com/bahooo22/Strata/commit/a36d7f53a6). This is a config A/B for q36 only.
- The fork's research docs reject: a RoPE table, a CUDA graph with n_cpu_moe, Walsh-Hadamard for q4_0 KV, GDN projection fusion, GDN prefetch d=2, and FA Q-from-global (5b6de8f87f, 744026f9e7, 94265f0f8a, 5e13c9a024, bedae61d29) — [commits](https://github.com/bahooo22/Strata/commits/main). Useful as a "do not try" list for q36.

**F7. Items already in our tree (ALREADY-HAVE)**
- Lamina 219e955a00 HostPin: the host loop pin was lost in Lamina after a refactor — [commit](https://github.com/eddoursul/Lamina/commit/219e955a00). Our tree still pins the host once per session in `SessionLoopScratch::init` (`src/core/session.cpp:550-567`, called at `generate.cpp:9621`). ALREADY-HAVE. It is worth a runtime check of the `strata-host` thread affinity next time the engine runs.
- ksanislo 8aeffd29fa: q36 penalty out-of-bounds with a short draft-vocab row (#3911) — [commit](https://github.com/ksanislo/Strata/commit/8aeffd29fa). Ours already calls `strata::qwen35::apply_penalties` (`q36_engine.cu:3019`).
- RandoTeam bc952e32c0: penalties count the reply, not the prompt (#4590) — [commit](https://github.com/RandoTeam/4gbVramStrata/commit/bc952e32c0). Ours has `STRATA_PENALTY_PROMPT` / reply0 (`generate.cpp:14540, 18238`).
- MyEcoria dbc7598554 / f4c804a2a0 (Flint): host-tier experts read in place over PCIe, then promoted. Decode 27 → 34-38 → 64 tok/s on an RX 9070 for a different model family — [commit](https://github.com/MyEcoria/Strata/commit/dbc7598554). This is the same concept as Strata's PCIe share, which we turned off on purpose (`--pcie-frac 0` measured better). ALREADY-HAVE / REJECTED-BEFORE for our config.
- RandoTeam 4a9539f2e5 `--no-graphs` +29% / +18% on a 4 GB RTX 3050 laptop under Windows WDDM — [commit](https://github.com/RandoTeam/4gbVramStrata/commit/4a9539f2e5). This is WDDM-specific. The flag exists for us (README:397, `--no-token-graph` at `generate.cpp:3807`). A one-run sanity A/B is cheap, but no gain is expected on Linux HIP (NOT-APPLICABLE).

**F8. Correctness fixes that also apply to our tree (latent under our preset)**
- Lamina a9195ed287: `gdn_gate_kernel` launched n_tokens×h_v threads but returned for `i >= h_v`, so only the first token was computed — [commit](https://github.com/eddoursul/Lamina/commit/a9195ed287). **Our `src/kernels/cuda/elementwise.cu:43` has the same `if (i >= h_v) return;`.** It's latent: the only callers pass n_tokens=1 (`src/core/layer.cpp:342`, elementwise_parity). Fix in a few lines (bound by `n_tokens*h_v`, index the head as `i % h_v`).
- Lamina 238316ac09: with `--spec-split`, both token groups wait on a single PCIe-share flag B, so group 0 can pass before its DMA copies land — [commit](https://github.com/eddoursul/Lamina/commit/238316ac09). **Ours has the same single `wait_flag_ge(m_flagB_, ring, cs)` (`src/core/verify.cpp:2675`).** It's latent for us (no `--spec-split`, `--pcie-frac 0`). Port it if spec-split is ever A/B'd.
- rpcribari 7293b8590d: with `--pipeline-windows 2`, `prepare_prefill` sized for 3 rows but got 5, and `hipHostUnregister` blocked under an in-flight window — [commit](https://github.com/rpcribari/Strata/commit/7293b8590d). Ours still calls `mtp.prepare_prefill(e)` at `generate.cpp:11978`. It's latent (no pipeline windows in fast-hip).
- Lamina 6c243c1069: a temperature under 1e-30 (or NaN) should be greedy, because logit/T overflows to NaN — [commit](https://github.com/eddoursul/Lamina/commit/6c243c1069). Not checked in our server-side sanitizer (gap).
- Lamina 0a5baefceb: pin `expert_used_count` to the compiled 10 — [commit](https://github.com/eddoursul/Lamina/commit/0a5baefceb). Our `gguf_reader.hpp:719/769` already validates the range ("the router takes 1..") at run time, so the case differs. NOT-APPLICABLE.

### Inferences
- F1 is the most promising item in the batch. It hits the exact decode gate/up kernel fast-hip uses on every token, the author saw up to 9% kernel-time cuts on IQ2_XS-class formats, and our gfx1100 ISA confirms the precondition (flat loads present). End-to-end gain is bounded by the expert-kernel share of a decode step. With the CPU pool and attention also in the step, a realistic expectation is 1-4% decode (inference).
- F2+F3 together are a "free" exact cleanup worth one build and one A/B round. Each is likely ≤1%.
- No fork in this batch offers a lossy or structural change that would move decode by double digits on our config.

### Gaps
- No gfx1100 measurement exists for F1-F4. All gains are author-measured on NVIDIA or gfx906.
- I didn't check whether the compiler keeps the address-space hint through `S::load` helpers in `expdec_gu_swiglu_kernel`. That needs a recount of `flat_load` after the change.
- Lamina's other 100 commits (server, structured output, MCP, web, Gyro-L trellis formats, Q5_0 experts, Windows) were triaged from subjects only. Gyro-L (TQP5-7) is a new expert quant format that needs its own model files, so it was not evaluated.
- I didn't open qwertie's other engine-scheduling commits (multi-conversation decode, page pools). They target multi-user throughput.

## One-line verdicts for every other fork in the batch

- **eddoursul/Lamina** (103): F1 NEW (port), F8 latent bug fixes (gdn_gate, spec-split flags). The rest is server, structured output, MCP, web, Windows, vision, Gyro-L/Q5_0 formats, prompt-cache park budgets (multi-conversation): nothing to take.
- **qwertie/Strataq** (92): a new paged multi-conversation engine (strata-q38/ds41), memory panel GUI. Only F5 (serve while experts load) is an idea for start time. Nothing to port directly.
- **bahooo22/Strata** (51): q36-only kernel/VRAM fixes (F6). bbc034d48a and 4569b243f7 are NEW for our Qwen3.6 path. Nothing for Flash-Next.
- **RandoTeam/4gbVramStrata** (40): 4 GB / WDDM docs and setup. The penalty fix is ALREADY-HAVE, `--no-graphs` is WDDM-specific. Nothing to take.
- **rpcribari/Strata** (23): F2, F3 NEW (exact HIP), F4 marginal. The gfx906 repack GEMM (PF_RP / DENSE_RP, vendored mx-llama.cpp) and the gfx906 IQ GEMV/MMVQ tuning (77aa02754e, dd0d9ce9e7) are tuned for gfx906 (NOT-APPLICABLE without re-tuning). 7293b8590d is a latent fix for pipeline windows.
- **MyEcoria/Strata** (16): the Flint model-family engine on HIP (RX 9070). In-place host experts is ALREADY-HAVE / REJECTED-BEFORE for our config. The router top-k fix (852fd7bf22, serial top-6 in one thread) is Flint-only, and our router_top10 is already parallel. Nothing to take.
- **ksanislo/Strata** (14): upstream cherry-picks of server, conversation-cache and tool-call fixes. The penalty OOB fix is ALREADY-HAVE. dq_rows_kernel vectorization (9971354a18) is in `src/dsv41` (DeepSeek), NOT-APPLICABLE.
- **okkiv/Strata** (10): Docker/release workflows and DeepSeek V4.1 on RX 7000 packaging: nothing to take.
- **devxplay-forks/Strata** (8): docs plus a timestamp in logs: nothing to take.
- **UltimateBoomer/Strata** (6): per-request engine settings and an MTP acceptance-length readout in the Monitor (could help diagnostics). Nothing for speed.
- **btaens/Strata** (5): lazy load / idle unload, experts kept in RAM between unloads, NUMA-local arena: NOT-APPLICABLE (single always-on engine, single socket).
- **wittech/Strata** (5): H200 support and download tweaks: nothing to take.
- **Ereife2/Strata** (3): EXL3 packs. Docs and benches only. EXL3 was REJECTED-BEFORE (license/architecture, earlier report).
- **SMASH-IT-OUT/Strata** (3): agent docs: nothing to take.
- **quangorn/Strata** (3): personal build rules and an MSVC setenv test fix: nothing to take.
- **PRJXAI/Strata** (2): multi-model lazy router: nothing to take.
- **lask3802/Strata-Link** (2): remote-stage MTP (distributed): NOT-APPLICABLE.
- **DarthCY-K/Strata** (1): sm120 CMake: nothing to take.
- **RX5950XT/Strata** (1): docs on kv-grow / VRAM reserve measurements (NVIDIA). `--kv-grow` and `--vram-reserve-mib` are already queued A/Bs from the earlier report.
- **forestautomation/Strata-Cascade** (1, squashed): VRAM + pinned RAM + SSD tiers and a memory guard for 32 GB machines. NOT-APPLICABLE (our experts fit in RAM).
- **jonathanmajh/Strata** (1): workflows disabled except gfx900 image: nothing to take.
- **scott1028/Strata** (1): quickstart script: nothing to take.
