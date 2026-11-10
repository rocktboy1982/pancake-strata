# Active Strata forks, November 2026: batch 4 (22 forks)

State as of 2026-11-10. Our tree: /home/dan/clone-of-d/src/Strata at 39f7a3da3 (preset `configs/pancake/flashnext-iq2_xs-fast-hip.json`). Each number below was measured by the fork's author on the author's own hardware, never by us. Status labels: NEW, ALREADY-HAVE, REJECTED-BEFORE, ALREADY-FLAGGED (named in an earlier report but not ported), NOT-APPLICABLE.

## Q1. Which own-commits in this batch could raise decode tok/s, prompt speed or TTFT, or cut start time on gfx1100 + 5700G?

### Takeaway
No fork in this batch has a decode kernel we lack. The useful items are on the serve and prefill path. The best one is **0xluk's STRATA_PF_AHEAD**, which prefetches the next layer's experts during staged prefill chunks. It targets the exact path from MOD_PLAN item 78 (agent turns under 4096 tokens run "staged"), and the author measured +35% on agent turns. Two small exact serve fixes come next. **STRATA_PARK_REWIND_MAX** (0xluk) and **STRATA_CKPT_MOVE_PARK** (phoenixclyde) both remove park copies that sit inside the next request's TTFT, about 0.1-0.3 s per agent turn. **STRATA_REFILL_ASYNC** (0xluk) moves the about 143 ms lent-slot refill off the first token. phoenixclyde's per-shape rows-per-wave table is an exact decode A/B. We already have the global switch (STRATA_MMVQ_R) but have never measured it on gfx1100.

### Cited Findings

**F1. 0xluk STRATA_PF_AHEAD: next-layer expert prefetch for staged prefill chunks. NEW. Exact.**
- Commits 896481c943, 09a550bc66, cd90f646ff, 4a3faee927. Files: `src/prefill/prefill.cpp` (+187, +73, +17, +11), `include/strata/prefill/prefill.hpp`, `generate.cpp` +15. Problem it fixes: in the staged walk (chunks below `stream_all_min()`, i.e. agent turns with the CPU share armed), a layer starts its expert copies only after its routing sync, so the copy engine sits idle through every attention half. Fix: in each chunk, record the experts the GPU streamed at layer l+1 (most tokens first), then copy them into an N-slot pool while layer l+1 runs its attention. The pool comes out of the prompt path's existing loan, so it adds no VRAM. The CPU/GPU split reads the residency as it would be without the pool, so the output stays bit-identical. Default `auto` = 96 slots. The window-fit variant (STRATA_PF_AHEAD_FIT) is experimental and was 7-12% slower. With the pool on, STRATA_KV_PREFETCH applies only to streamed chunks (+3% fresh; it cost -4% on staged turns). cd90f646ff also fixes the `auto` CPU share so prefetched experts no longer count as GPU work — [896481c943](https://github.com/0xluk/Strata/commit/896481c943), [09a550bc66](https://github.com/0xluk/Strata/commit/09a550bc66), [cd90f646ff](https://github.com/0xluk/Strata/commit/cd90f646ff), [4a3faee927](https://github.com/0xluk/Strata/commit/4a3faee927)
- Author-measured (Xeon E5-1650 v3 + 2x RTX 3060 12 GB, PCIe 3.0 x16, Flash-Next IQ3_S, combined with REFILL_ASYNC / PARK_REWIND_MAX / ARENA_REUSE): agent turns (~2.5K new tokens on 30-48K context) went from 512 to 692 tok/s, TTFT 4.97 to 3.67 s (+35%). At 165-183K: 449 to 602 tok/s. Fresh 2K/30K/165K prompts: 552/1080/1005 to 644/1112/1040 tok/s. Decode unchanged — [0xluk README](https://github.com/0xluk/Strata/blob/flashnext-pcie3/README.md)
- Our tree: none of these env names exist in src/. The staged walk exists: `stream_all_min_tokens()` is in `include/strata/prefill/prefill.hpp:96`. MOD_PLAN item 78 confirms that our preset (STRATA_PREFILL_CPU_SHARE=0.6, _MAX=4096) sends every agent delta under 4096 tokens down this staged path, which reads 1757 vs 2580 tok/s at 32K.
- Applicability: high. Our agent turns (1.5-3.2K deltas) take the staged path. PCIe 4.0 x16 makes each copy cheaper than on the author's PCIe 3.0, so the gain may be smaller, but the idle copy engine during attention is the same. Caveat: the author's numbers bundle four patches, and the gain from PF_AHEAD alone is in PR #2, which I did not read. Branch `flashnext-pcie3` sits on an older upstream base, so this is a hand port, not a cherry-pick. Effort M (about 290 lines in prefill.cpp, plus the share fix).
- Suggested A/B: STRATA_PF_AHEAD=auto vs unset on the 2K code edit and the 11.7K prompt, with the agent-turn shape (follow-up turns on 30K+ context). Expect no effect on short chat.

**F2. 0xluk STRATA_PARK_REWIND_MAX: skip the park when a request continues the live conversation from one of its own checkpoints. NEW. Exact.**
- Commit 53cecdeee9 (`generate.cpp` +27). A chat template that renders past replies without their reasoning (Qwen does this) makes every agent turn resume from the live conversation's checkpoint before the last generation prompt. Each such turn first parked the whole live conversation (K/V copied to host), and the next turn then dropped that park as superseded. With N > 0, a request that resumes from one of the live conversation's own checkpoints, dropping at most N tokens, skips the park. Requests for another conversation still park. Tradeoff: the dropped tail (the last reply) is not kept — [53cecdeee9](https://github.com/0xluk/Strata/commit/53cecdeee9)
- Author-measured (RTX 3060 host): the park cost 0.3 s at 45K tokens and up to 0.9 s near 200K — [53cecdeee9](https://github.com/0xluk/Strata/commit/53cecdeee9). Independently, phoenixclyde (MI50) measured the synchronous park inside the *next* request's prompt time at 147-197 ms — [b1a79a89ec](https://github.com/phoenixclyde/Strata/commit/b1a79a89ec)
- Our tree: not present. Our park decision (`src/program/generate.cpp` ~16692-16760) has `pin_sibling` (MOD_PLAN 90) and `--conversation-cache-park-gain`, but neither covers this case. The patch applies almost line for line: a lambda `rewind_no_park` plus `&& !rewind_no_park` in the `else if (... park_current(...))` branch.
- Applicability: high for opencode agent sessions on Qwen3.8 (the reasoning is dropped from history). Effort S. A/B: STRATA_PARK_REWIND_MAX=4096 (the author's value) vs unset, measuring TTFT on the 2K code edit inside a multi-turn agent session. Also check the log line "not parked (STRATA_PARK_REWIND_MAX)".

**F3. phoenixclyde STRATA_CKPT_MOVE_PARK, plus async park allowed on HIP. Move part NEW, async-on-HIP part needs checking. Exact.**
- Commit b1a79a89ec lifts the `STRATA_USE_HIP || STRATA_HIP_GFX906` exclusion from `park_async_on()` in `conversation_snapshot.cpp`, so STRATA_PARK_ASYNC can run on HIP. Commit 607152fac7 gives the "move dropped checkpoints into the parked image instead of copying them" logic its own switch, STRATA_CKPT_MOVE_PARK=1. Before, that logic was gated by `park_async_env()`, and on the MI50 the async park "cannot allocate its stage at all", so the move never ran. One 1900-token checkpoint is about 260 MB of K/V, so the move saves tens of ms per park (the author's summary says 92 ms) — [b1a79a89ec](https://github.com/phoenixclyde/Strata/commit/b1a79a89ec), [607152fac7](https://github.com/phoenixclyde/Strata/commit/607152fac7)
- Our tree: `park_async_on()` returns false under STRATA_USE_HIP (`src/core/conversation_snapshot.cpp:231-238`). `on_device()` is also a HIP stub that returns false, so the VRAM-stage path may not be usable as it stands. The park_drops move at `generate.cpp:16728` is gated on `park_async_env()`, which reads only the env and is not HIP-gated. So on our build, setting STRATA_PARK_ASYNC=1 probably already turns the move on while the async copy stays off. That needs a quick log check. If it holds, the move is effectively ALREADY-HAVE behind STRATA_PARK_ASYNC=1, and only the clean separate switch is new.
- Applicability: medium. Each agent-turn park pays this cost. Effort S. A/B: STRATA_PARK_ASYNC=1 (and, if ported, STRATA_CKPT_MOVE_PARK=1) vs unset, measuring TTFT on the turn after a park (2K code edit in an agent session). Do not port the marker-file polling (`/data/strata-*`); it is their ops convenience.

**F4. 0xluk STRATA_REFILL_ASYNC: refill the prompt path's lent expert slots in the background. NEW. Exact (affects timing and residency only).**
- Commit f9b14a4965 (`expert_cache.cpp/.hpp`, `generate.cpp` +57). The lent cache slots are refilled on the refill stream and admitted like an adaptive round's swaps (`apply_pending`) once they land, polled before each decode window. Before, they blocked the first token — [f9b14a4965](https://github.com/0xluk/Strata/commit/f9b14a4965)
- Author-measured: "~143 ms" off the first token on an RTX 3060 (PCIe 3.0) — [f9b14a4965](https://github.com/0xluk/Strata/commit/f9b14a4965)
- Our tree: we have "queued copies and one wait" (perf-review D-4, `generate.cpp:438`, STRATA_REFILL_BLOCKING) and no background admission. On PCIe 4.0 the wait should be about half as long. Risk: the first decode windows run with fewer resident experts, so the first tokens are slower. Effort S-M. A/B: TTFT and first-100-token decode on the 2K code edit and the 11.7K prompt.

**F5. phoenixclyde measured rows-per-wave table for the wave GEMV at ncols > 1 (verify windows). Global switch ALREADY-HAVE, per-shape table NEW. Exact (bitwise).**
- Commit 3d041542b3 (`native_mmvq.cu` +58). `wave_launch_n` hardcoded R = 1. A table measured on 9 shapes x ncols 1..8 x R in {1,2,4} keeps R > 1 only where it beat R = 1 by at least 3%. ncols = 1 stays at R = 1, because R = 4 there cost 0.982x per layer. The run-time override is STRATA_MMVQ_WAVE_ROWS. The per-row reduction order does not change, so the output is bitwise identical (diff = 0 on 9 shapes) — [3d041542b3](https://github.com/phoenixclyde/Strata/commit/3d041542b3)
- Author-measured (MI50 gfx906, wave64): at ncols 6, R = 2 gives 1.6-1.8x and R = 4 gives 1.9x on qkv/gate/q, with the whole layer going from 613.6 to 385.4 us. In production, interleaved, 3 x 480-token greedy decodes: 48.6 vs 46.8 tok/s (+4.0%) — [3d041542b3](https://github.com/phoenixclyde/Strata/commit/3d041542b3)
- Our tree: `STRATA_MMVQ_R=2/4` (global, multi-column only) already exists in `src/kernels/cuda/native_mmvq.cu:2661-2670` (README row 286, "tuning"). I found no gfx1100 measurement of it in MOD_PLAN or bench/results. Note: the R check runs *before* the `g_wave_exact` branch, so STRATA_MMVQ_R=2 on top of the preset's STRATA_RDNA_MMVQ_WAVE=3 swaps the exact-emulation kernel for the plain wave kernel, which changes the rounding against the current preset.
- Applicability: wave32 RDNA3 has a different occupancy and register budget from gfx906 wave64, so the gfx906 table does not transfer. Our own per-shape sweep would be needed. Effort S (env A/B first), M for the table. A/B: STRATA_MMVQ_R=2 and =4 vs unset on short chat and the 2K code edit (MTP verify windows are multi-column).

**F6. Linsche gfx1030: GDN recurrence pair on by default (STRATA_GDN_HEAD=1, STRATA_GDN_PP=2). ALREADY-FLAGGED.**
- Commit 5961fb431f. Author-measured on an RX 6800: `gdn_rec_head` at T=4096 went from 9.51 to 5.25 ms, and recurrence at a 13.6K prompt from 999 to 643 ms (about -2.5% of the prompt). Greedy output identical. CONVL2 and NOY are "not bitwise end to end" and the decode switches showed no gain — [5961fb431f](https://github.com/Linsche/Strata/commit/5961fb431f)
- Our preset has STRATA_GDN_PP=1 (plus CONVL2=1 and NOY=1). STRATA_GDN_PP=2 is already item 1.1 of "Forkuri Strata câștig viteză.md". This is a second, independent RDNA datapoint for the same A/B (11.7K prompt). Linsche's note that CONVL2/NOY are not bitwise is worth recording next to our preset, which has both on.

**F7. Linsche STRATA_PATTN_FLASH: one-pass prompt QSA attention where no tensor-core kernel runs. NOT-APPLICABLE.**
- Commits 4a2e497d09 and ddab195208. Only for k8v4 pools, and only where `qsa_prompt_attn_batch` refuses (gfx103x, no WMMA). Author-measured on an RX 6800: 13.6K prompts 938 to 999/1011 tok/s (+6.5/+8%) — [ddab195208](https://github.com/Linsche/Strata/commit/ddab195208). We run `--kv int8` on gfx1100 with WMMA prompt attention (STRATA_PROMPT_ATTN_IMMA=1, STRATA_HIP_WMMA=1), so this path never runs for us.

**F8. phoenixclyde dp4a hyper-connection down (6319ef6f95, 101ad14947) and dp4a decode attention score + DPP (5c1b3fe6d9). ALREADY-HAVE.**
- Their probe: dp4a gives 73-117% more effective bandwidth on the arithmetic-bound HC/GDN reads on gfx906 — [d202ba007e](https://github.com/phoenixclyde/Strata/commit/d202ba007e). The HC down candidate is gated to T<=6 after a bench cliff. We already have the RDNA3 equivalent: STRATA_GR_RDNA_A8 (int8 activations + `__builtin_amdgcn_sudot4`, `rdna_gr.cu:284`, in the fast-hip preset). Our decode attention already has DPP butterflies (`qsa_decode_attn.cu:470-485`). One idea is left: the T<=6 gate. Check whether GR_RDNA_A8 shows a similar cliff at wider verify windows (spec 4 gives T<=5, so probably irrelevant).

**F9. phoenixclyde VMM on gfx906 (f6fbd61c93). ALREADY-HAVE.** `src/core/vmm.cpp:158-167` already compiles the HIP VMM path for STRATA_USE_HIP on non-gfx906 builds.

**F10. Aaron-xx one MMQ launch per expert (535ce07303, +58% prefill). NOT-APPLICABLE as is.**
- This is in `src/qwen35/q36_engine.cu` (the Qwen3.6-35B-A3B engine), not the Flash-Next path. Author-measured on a GTX 1660 Ti 6 GB: one MMQ grid per layer padded to the busiest expert (max rows 18.5x the mean) gave 231 tok/s, against 365 tok/s with one launch per expert — [535ce07303](https://github.com/Aaron-xx/Strata/commit/535ce07303)
- Our Flash-Next prompt path already launches in 16-expert groups taken in row-count order (`prefill.cpp:5214-5227`, with STRATA_MMQ_GROUP_SORT opt-in), which handles most of the padding. Related finding from phoenixclyde 2c831464bb: at 2K/16K, 90%/99% of the small-batch prefill penalty is *outside* the MMQ kernel (group loop: waits, host table build, tail zeroing, inter-group events) — [2c831464bb](https://github.com/phoenixclyde/Strata/commit/2c831464bb). That agrees with our MOD_PLAN items 78/80 (per-layer sync, host-side grouping, hipBLASLt descriptor churn).

**F11. 0xluk STRATA_ARENA_REUSE=1 (e1246a1134, e2ffeef8da). ALREADY-FLAGGED.** This is upstream PR #3509, already item 1 in "Proiecte derivate din Strata.md" (restarts pay 28-65 s for the 33 GiB arena read on ntfs3). New in this fork: GGUF-sourced arenas also qualify (e2ffeef8da) — [e2ffeef8da](https://github.com/0xluk/Strata/commit/e2ffeef8da). Our arena comes from a GGUF plus pack, so take this follow-up together with #3509.

**F12. lighttransport strata-m3 (MiniMax M3.1 engine, V100). Ideas only. NOT-APPLICABLE as code.**
- This is a separate engine (`src/minimax/m3_engine.cu`). Transferable observations, all author-measured on V100:
  - Resident experts of a window layer on 4 side streams: +3.5%, tokens identical — [efdbb94f82](https://github.com/lighttransport/Strata/commit/efdbb94f82).
  - CPU gate/up started before the GPU experts are queued — [d0008c562b](https://github.com/lighttransport/Strata/commit/d0008c562b).
  - Residency-aware pruning and cache-aware substitution of missed experts "do not beat uniform top-3" on their quality gate — [2ea0c1e042](https://github.com/lighttransport/Strata/commit/2ea0c1e042). This is a caution data point for our lossy turbo preset (STRATA_ROUTE_PRIOR).
  - Confidence-gated draft length (pmin) — [739100bb56](https://github.com/lighttransport/Strata/commit/739100bb56). ALREADY-HAVE as `--spec-min-p 0.5`.

### Inferences
- Suggested order (cheapest exact first): F2 PARK_REWIND_MAX (S), F3 check whether PARK_ASYNC=1 already turns the move on (S, env only), F5 MMVQ_R env A/B (S, env only), F4 REFILL_ASYNC (S-M), F1 PF_AHEAD (M, biggest expected gain on agent turns).
- F1, F2 and F4 fit together: one session of follow-up agent turns on 30K+ context measures all three. The author's +35% is the bundle, not PF_AHEAD alone.
- The serve-side wins (F2/F3/F4) cut TTFT, not decode tok/s. For opencode agent loops, TTFT per turn is the cost the user feels.

### Gaps
- PF_AHEAD alone, on a single GPU with PCIe 4.0: the per-patch A/B is in [0xluk PR #2](https://github.com/0xluk/Strata/pull/2), which I did not open.
- Whether STRATA_PARK_ASYNC=1 on our HIP build really enables the checkpoint move without the async copy: inferred from reading the code, not run.
- STRATA_MMVQ_R has never been measured on gfx1100 (as far as MOD_PLAN and bench/results show).

## Q2. Correctness fixes in these forks that also affect our tree

### Takeaway
None of the fixes I checked applies to our tree. jmnargi's Q5_0 expert-dot fix is already in our `iq_kernels.cu`. Aaron-xx's MMQ grouped-launch corruption only exists in his own q36 grouping patch. jmnargi's compact-walk routing-table upload bug needs a Volta-only branch that our prefill does not have.

### Cited Findings
- jmnargi 2b333319b1: the Q5_0 expert dot applied the -16 offset to the sum of the original activations instead of the quantized ones, roughly doubling the whole-expert error (2.4e-2..3.2e-2 vs 1.25e-2) — [2b333319b1](https://github.com/jmnargi/Strata-V100/commit/2b333319b1). Our `src/kernels/cuda/iq_kernels.cu:444-470` already uses `d5 * d8 * (sumi - 16 * sumu)`. ALREADY-HAVE.
- Aaron-xx df9db65d62: grouped MMQ launches passed the group's row count as `total_rows`, but the q8 activation layout strides by n_ent, so every expert after the first read another expert's activations ("garbled" prompts, 0 tool calls) — [df9db65d62](https://github.com/Aaron-xx/Strata/commit/df9db65d62). This bug was introduced by his own 535ce07303 and is not in our q36 engine (no STRATA_Q36_MMQ_GROUP). Our Flash-Next grouped path passes `total_rows = T*K` / `nr` (`prefill.cpp:5638-5727, 6022`). Lesson for any grouping change we make: `total_rows` must stay the full xq row count.
- jmnargi 141903814c: the routing-table upload sat inside the non-compact branch, so Volta's compact walk crashed past about 2.3K tokens — [141903814c](https://github.com/jmnargi/Strata-V100/commit/141903814c). In our `prefill.cpp` (~5227-5255) the upload is not inside such a branch (`n_order = order.size();` has no closing brace after it). NOT-APPLICABLE.
- cassettesgoboom 5dddd2cd85: the DFlash2 stream selection also covers STRATA_HIP_GFX906 — [5dddd2cd85](https://github.com/cassettesgoboom/strata-gfx906/commit/5dddd2cd85). gfx906 only. We use MTP, and STRATA_USE_HIP already covers gfx1100.

### Inferences
- No action needed on correctness from this batch.

### Gaps
- None.

## Q3. One-line verdict for every fork in the batch

### Takeaway
3 forks carry portable items (0xluk, phoenixclyde, Linsche as a datapoint). The other 19 have nothing to take for our hardware.

### Cited Findings
- **phoenixclyde/Strata** (112 commits, gfx906/MI50): F3 (CKPT_MOVE_PARK / async park on HIP) and F5 (rows-per-wave table) are worth taking. dp4a HC/attention is ALREADY-HAVE. Most commits are CI/governance/licensing/docs. Their probes give useful context: expert read is 1.45 GB/token at 33% of the roof; cold prefill TTFT CV is 71-96%, so a prefill A/B needs n>=6 — [d47556da0f](https://github.com/phoenixclyde/Strata/commit/d47556da0f), [f3a02b0296](https://github.com/phoenixclyde/Strata/commit/f3a02b0296).
- **tyloryang/Strata** (81, Huawei Ascend 910B CANN port): NOT-APPLICABLE, different hardware backend.
- **lighttransport/Strata** (61, MiniMax M3.1 / GLM / A64FX engines on V100 and Fugaku): ideas only (F12), no portable code.
- **sterlenjohnson/Strata** (36, RTX 4060 Ti notes): weight-based CPU-expert skip (STRATA_SKIP_THR) is lossy and the author measured no gain on the production config — [0275e4963d](https://github.com/sterlenjohnson/Strata/commit/0275e4963d). The KV codec test (`--kv-turbo`) is the same class as turbo4/TurboQuant, REJECTED-BEFORE in MOD_PLAN. The rest is docs. The author says "nothing beat the production config" for Flash-Next — [FORK.md](https://github.com/sterlenjohnson/Strata/blob/main/FORK.md). Nothing to take.
- **Aaron-xx/Strata** (22, GTX 1660 Ti, Qwen3.6-35B q36 engine): q36-only prefill work (F10) and the MMQ grouped-launch fix (Q2). Nothing for Flash-Next.
- **rafiki270/Strata** (16): Windows/MSVC build portability, launcher, benchmarks. Nothing to take.
- **0xrjman/Strata** (13): Codex tool-call parsing in serve/ and Docker caching. Nothing for speed.
- **0xluk/Strata** (9, 2x RTX 3060): F1 PF_AHEAD, F2 PARK_REWIND_MAX, F4 REFILL_ASYNC, F11 ARENA_REUSE GGUF follow-up. **The most valuable fork in this batch.**
- **jmnargi/Strata-V100** (8): V100 trellis/HC kernels, sm_70 crash fix, Q5_0 fix (ALREADY-HAVE). STRATA_SPLIT_OWN_PRIMARY targets two-GPU layer splits only (-4..-10% prompt ms on 2x V100) — [5076fbbc5a](https://github.com/jmnargi/Strata-V100/commit/5076fbbc5a). NOT-APPLICABLE (one card; MOD_PLAN parks second-GPU work).
- **cassettesgoboom/strata-gfx906** (6): gfx906 DFlash2 stream fix plus docs. NOT-APPLICABLE.
- **kinoyu-ai/Strata-CacheKV-kinoyu** (5): disk/RAM conversation KV tiers, restart-persistent snapshots, PLE disk page cache. ALREADY-HAVE as our item 39 disk tier (STRATA_CONVERSATION_DISK_*). Its PLE disk cache showed no speedup (USB hits 66 ms vs NVMe 37-40 ms per 4096 rows) — [PLE_DISK_CACHE.md](https://github.com/kinoyu-ai/Strata-CacheKV-kinoyu/blob/main/docs/PLE_DISK_CACHE.md). Nothing to take.
- **Linsche/Strata** (4, RX 6800 gfx1030): F6 GDN_PP=2 datapoint (ALREADY-FLAGGED). F7 PATTN_FLASH is NOT-APPLICABLE (k8v4 + no WMMA).
- **thoives/Strata** (4): devcontainer and "expert tiers" plans. Nothing to take.
- **corbinjurgens/Strata** (3): Windows launch planning and mapped-page release. NOT-APPLICABLE.
- **xszwow/Strata** (3, Ryzen AI Max+ 395, Windows): a 57K draft vocab and deploy docs. The author says "no native engine source changes or new performance measurements" — [60ed2073fd](https://github.com/xszwow/Strata/commit/60ed2073fd). We already use `--mtp-draft-vocab`. Nothing to take.
- **chynggi/Strata** (2): merge notes. Nothing to take.
- **shiosalt/Strata** (2): Muse-Spark-816B model family plan. NOT-APPLICABLE.
- **Dictor/Strata-gfx1030-docker** (1): Docker AVX2 vision encoders. Nothing to take.
- **bearice/Strata** (1): undeclared tool-call fix in serve. Nothing for speed.
- **gputier/StrataGP** (1): docs on DeepSeek stops. Nothing to take.
- **ms-tingliang/Strata** (1): MSVC /bigobj for dsv41. Nothing to take.

### Inferences
- The forks with active engine work in this window are aimed at other hardware (gfx906, Ascend, V100, A64FX, gfx1030, Ada). What crosses over to gfx1100 is serve/prefill orchestration, not kernels.

### Gaps
- I did not open docs-only commits or commits from packaging/UI forks (per the budget rule). Their verdicts rest on commit subjects and descriptions.
