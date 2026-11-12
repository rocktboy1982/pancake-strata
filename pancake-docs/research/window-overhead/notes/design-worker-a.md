# Phase 18 design, Worker A: items 2 to 10

Scope: design only, from code reading. Base: worktree `agent-ad25cdf1dc7ea704a` (branch dev/p17/integration). No repo file edited except this one; no build, no engine run, no GPU, no commit.

Labels: [code file:line] = read in this pass, line checked; [profile] = profile summary or CSV counts; [measured] = result file cited in the untested-switch note; [paper] = literature note; [unverified] = inference, arithmetic on profile numbers with an assumption, or a claim not checked. Percent of wall = ms / 3553 ms decode wall [profile]. The profile summary's percentages are of kernel time (2712 ms) and are not used here except where marked.

Profiled configuration assumed throughout: `configs/pancake/flashnext-iq2_xs-fast-hip.json` with the deferral line removed, as recorded in `notes/profile-environment.md` [profile]: WAIT_FUSE=1, COMBINE_CPU=1 (fast-hip preset, gpu_sync note) [unverified: not dumped at runtime], ROUTE_PUBLISH=1, EXPDEC_DOWN=1, POOL_FUSE_PHASES=1, FUSE_WAIT off, COMBINE_POLL1 off, WAIT_COPY256 off, ROUTE_FAST off, MTP_EAGER off, CLI `--spec 4 --spec-min-p 0.5 --lookup-max-t 12`.

---

## 0. Corrections found while reading (these change the queued arms)

1. **Item 4 as written is almost a no-op while WAIT_FUSE=1.** The plan's copy takes the WAIT_FUSE branch first (`verify.cpp:2827-2828`), so `STRATA_FUSE_WAIT` at `verify.cpp:2830` never runs for the plan copy. In the CPU-share path FUSE_WAIT only reaches the one non-fused group (`verify.cpp:2887-2892`). [code] The arm must be a 2x2 with WAIT_FUSE (see item 4).
2. **WAIT_COPY256 probably never acts on decode windows at `--spec 4`.** `wait_copy_wide` requires `n > 512` words (`verify_kernels.cu:1797-1799`) [code]. The plan size is `plan_i32_ = dst2_off_ + 32` with `cap = T*K` and `K = 10` [code `verify.cpp:1190-1196`]. Working through the formula gives `plan_i32_ = 80*T + 42` [unverified arithmetic; the audit's 1,320 words at T=16 matches]. So T=5 gives 442 words and T=6 gives 522. Window sizes are bounded by `--spec` through `lookup_size_mask` (`generate.cpp:1365-1378`) [code]. At `--spec 4` the windows are probably at most 4 or 5 rows, so the switch is inert there [unverified until the `switch_report` log is read; the code prints "will not act (a plan of N words...)"]. The earlier +6.8% (WAIT_COPY256) and +8.0% (ROUTE_FAST) passes cannot come from these switches, since their ceilings are below 1% (items 6 and 7) [measured] + [unverified].
3. **ROUTE_FAST ceiling is 2.2% of wall, not 2.9%.** route_publish_res is 78.1 ms [profile] = 2.2% of 3553 ms wall; 2.9% is its share of kernel time (2712 ms) [profile].
4. **Item 10's premise is stale.** The draft round writes the step and position rows for all draft steps before the first launch (`mtp.cpp:1821-1835`) [code]. The 17.5 ms gap is host turnaround between seeing `h_out_[j]` (`mtp.cpp:1919-1925` wait_step) and launching step j+1 (`mtp.cpp:1936-1950`) [code].
5. **Spec depth may already be measured flat.** `reports/Grouped technologies for pancake.md:232` says "`--spec 3/5/6` came out flat" [code/notes; the source result file is not cited, so the claim is unverified].
6. **MTP_EAGER class conflict.** The code comment says eager gives "the same drafts as the step-by-step chain" (`mtp.cpp:1859-1862`) [code]. The switch table says "drafting/stop policy: the tokens change" (`conversation_disk_env.def:494`) [code]. Token equality is required either way.
7. **The combine kernel is almost all wait, so item 2 can only change poll traffic.** The gpu_sync note measured a 5.6 us floor per combine call when the CPU is already done, and bounded the non-wait part at 2.2% of combine time on an older run (`research_notes/Hardware data path technologies/gpu_sync.md:30`) [measured, older run]. Applied to the current profile: 4032 calls x 5.6 us = 22.6 ms = 0.6% of wall, so at least 96% of the 670.6 ms (18.9% of wall) is waiting [profile + unverified bound]. The 18.9% is therefore a ceiling for the wait, not a combine-arithmetic opportunity.

---

## Item 2: split the combine spin (STRATA_COMBINE_POLL1)

### Mechanism
- With WAIT_FUSE on, the combine kernel receives the CPU-share flag (`verify.cpp:2944-2946` passes `g_wait_fuse ? m_flag_ : nullptr`) [code].
- `poll_slot_of` returns -1 unless `STRATA_COMBINE_POLL1=1` (`native_moe.cu:267-269`) [code]. With -1, thread 0 of every block spins on the PCIe-mapped flag: `if (poll_slot < 0) while (*flag < value)` (`native_moe.cu:175-177`) [code].
- With POLL1=1 and a free slot (64 slots per stream, `native_moe.cu:151`), one block is elected by CAS and polls the flag over PCIe. The other blocks spin on a VRAM word with `atomicAdd(st+1, 0)` (`native_moe.cu:179-184`) [code]. The last block past the wait resets the slot (`native_moe.cu:186-191`) [code].
- Grid is `((n4+63)/64, n_tok)` (`native_moe.cu:377`) [code]. At N=2560 that is 10 blocks per token [code, gpu_sync note:67], so about 10 x n_tok pollers per combine in the default path.
- The profile splits the combine as 4032 calls, 670.6 ms, median 75 us [profile]. The `t_waited` stamp exists but is only set under profiling (`native_moe.cu:193`) [code].

### Change
None. Switch only: `STRATA_COMBINE_POLL1=1`. Class is Speed (`conversation_disk_env.def:338`) [code].

### Cost and risk
- Lines: 0.
- Exactness: the dependency is the same, so the output must be bit-identical. Only which block reads the flag changes [code]. Confirm with token equality in the arm anyway.
- Risk 1: the spinning blocks now poll a single L2 word with atomics. 10 x n_tok atomics per combine may cost more than the PCIe reads they replace [unverified; the design note expects 0-3%].
- Risk 2: slot exhaustion falls back silently to -1 when more than 64 streams are used [code `native_moe.cu:274-275`]. Not an issue for one verifier.
- Graph capture: the slot state is a device global reset by the last block, so captured graphs are safe [code]. Stream order keeps consecutive launches on one stream clean [code comment].

### Test arm
- Base: the profiled config. Arm: same config with `STRATA_COMBINE_POLL1=1`.
- A B B A, n of at least 8, first run of each block discarded (owner rule).
- Read out: decode wall; `combine_cpu_k10_vec4<true>` total time from rocprof; `wait_flag_ge` and `wait_flag_copy_i32` totals (should not move); token equality against base.

### Decision
Keep the queued arm as it is; no code. Read-out:
- Wall faster by more than 5% and tokens identical: PCIe poll traffic was a real part of the wait. Candidate for a preset switch flip. The lever is then pool bandwidth (item 11, Worker B).
- Within plus or minus 5%: the combine time is pool latency, which the spin cannot remove. Stop work on combine; the next lever is the pool.
- Slower: keep off. The atomics cost is the reason.
Building anything more on the combine is not worth it: the wait is 96% or more of the kernel [unverified].

---

## Item 3: isolate STRATA_WAIT_FUSE (1 vs 0)

### Mechanism
- WAIT_FUSE=1 does two things: the plan wait and copy become one launch (`verify.cpp:2827-2828` -> `wait_flag_copy_i32`, `verify_kernels.cu:1823-1833`) [code]; and the combine polls the CPU flag inside its blocks (`verify.cpp:2944-2946`, `native_moe.cu:175`) [code].
- WAIT_FUSE=0 makes the plan path `wait_flag_ge` (`verify.cpp:2841`) then `copy_i32_from_mapped` (`verify.cpp:2842`) [code]. It also makes the CPU share a separate `wait_flag_ge(m_flag_)` before the combine (`verify.cpp:2883`), and the combine gets a null flag [code].
- Profile counts: wait_flag_copy_i32 4320 calls and combine 4032 calls [profile]. 4320 = 15 x 288 and 4032 = 14 x 288, so 15 MoE groups per window and 288 windows [unverified; the audit's inference].

### Change
None. Switch only.

### Cost and risk
- Node cost of WAIT_FUSE=0: one extra plan node per group (4320) and one extra CPU-share node per fused group (4032) = 8352 nodes [profile counts + code]. At about 2.4 us per dependent kernel (cited by the audit from `verify.cpp:418-420`, not re-read here) that is about 20 ms = 0.6% of wall [unverified]. So WAIT_FUSE=0 starts 0.6% behind; any gain has to be bigger than that.
- The design note expects WAIT_FUSE to be worth 0-3% [unverified]. Its test was to read the host-side "CPU" stage figure, not only tok/s (`gpu_sync.md:71`).
- Non-finite suspicion: `reports/Grouped technologies for pancake.md:234` lists WAIT_FUSE among the suspects for non-finite logits, and the bisect it asks for is not in the repo [code/notes]. The existing guard (`verify.cpp:1624-1626`) concerns deferral only, not WAIT_FUSE. So each arm needs a finite-logit check or a token diff.
- Lines: 0. Exact by class (`conversation_disk_env.def:768`) [code], but the suspicion means the class is not yet trusted.

### Test arm
A 2x2 with item 4: WAIT_FUSE in {1, 0} x FUSE_WAIT in {0, 1}. Use the profiled config as base, A B B A, n of at least 8. Read out wall, combine time, wait totals, token equality, and count of non-finite logits per window if the engine can report it [unverified: no such counter found in this pass].

### Decision
Run once, as part of the 2x2 in item 4, because it is switch-only. Not worth building. Expected effect is small and the node penalty is about 0.6%, so a clear win is unlikely unless poll traffic is large. Only flip the preset if WAIT_FUSE=0 wins by more than 5% and both arms are finite.

---

## Item 4: FUSE_WAIT (with WAIT_FUSE off)

### Mechanism
- `fuse_wait_on()` on HIP: an environment value wins (`verify.cpp:441-447`), otherwise on only for gfx12 (`verify.cpp:453`, `verify.cpp:462`), so gfx1100 is off by default [code; profile-environment note agrees].
- With WAIT_FUSE=0 and FUSE_WAIT=1, the plan copy becomes `wait_copy_i32_from_mapped` (`verify.cpp:2830-2831`) [code]: one launch instead of two.
- With WAIT_FUSE=1, FUSE_WAIT changes only the non-fused CPU-share group, which folds its wait into `copy_rows_from_mapped_after` (`verify.cpp:2887-2892`) [code]. In the profile that is about 288 standalone waits, i.e. about 1 per window [unverified].

### Change
None. Switch only. The plan's arm "FUSE_WAIT 1 vs 0 with WAIT_FUSE off" is the right one; the queued arm must be checked to confirm WAIT_FUSE=0 is set. If WAIT_FUSE=1 stays set in the queued config, it measures close to nothing by construction.

### Cost and risk
- Lines: 0.
- Ceiling with WAIT_FUSE=0: 4320 nodes saved x about 2.4 us = about 10 ms = 0.3% of wall [unverified, the plan's number too].
- Ceiling with WAIT_FUSE=1: 288 nodes folded = about 0.7 ms = 0.02% [unverified].
- Spin is unchanged in both: the fold removes a node, not the pool latency [code: the spin is in the same kernel after the fold].
- Exact by class (`conversation_disk_env.def:403`, class Speed, "graph/stream/doorbell plumbing") [code].

### Test arm
The 2x2 from item 3. Three arms are enough: base (WAIT_FUSE 1, FUSE_WAIT 0), (0, 0), and (0, 1). The fourth (1, 1) tests the 288-node fold and can be dropped.

### Decision
Below the 5% noise by its own ceiling. Do not spend a separate arm. Keep it inside the item 3 2x2, and read it only as a node-count effect (rocprof kernel count should drop by 4320 in the (0,1) arm).

---

## Item 5: speculation depth sweep (--spec 3, 4, 5)

### Mechanism
- `--spec` bounds the window size: lookup window sizes are `1..s_mtp` plus MTP+1, MTP+2 and `--spec` (`generate.cpp:1365-1378`) [code]. The MTP drafter gets `max_t = min(mtp_max_t, spec)` (`generate.cpp:1367`) [code].
- Deeper windows amortise the per-window fixed costs (spin, graph launch, window-end sync) over more tokens, but add positions that bring new CPU experts to the pool (`Grouped technologies for pancake.md:232`) [notes].
- A window of T rows also sets the plan size (item 7), so `--spec 3` and `--spec 4` both leave WAIT_COPY256 inert.

### Change
None. Switch only: `--spec 3`, `--spec 4` (profile), `--spec 5`.

### Cost and risk
- Lines: 0.
- Exactness: greedy verify accepts only tokens the verifier's argmax confirms, so the output is exact only if argmax over T rows equals argmax from the decode path. That is a T-invariance question that the repo's token-forcing check tests [unverified for this build]. Check token equality against `--spec 4` at least.
- Prior evidence: "flat for 3/5/6" [notes, unverified source].

### Test arm
`--spec 3`, `--spec 4`, `--spec 5` on the profiled config, n of 8 each, A B B A, token equality to spec 4.

### Decision
Not worth building. Run only if the owner wants a current-build confirmation of the flat result; the three arms cost machine time and the expected gain is inside the 5% noise.

---

## Item 6: ROUTE_FAST (confirm)

### Mechanism
- route_publish_res runs 4032 times for 78.1 ms, mean 19.4 us [profile]. It is the router's top-10 plus the CPU pool doorbell in one kernel (`route_publish_res` at `native_router.cu:217-221`) [code].
- The ROUTE_FAST kernel is a cheaper one-launch version of the same publish (`native_router.cu:217-225` comment; gate at `native_router.cu:490-492`) [code]. Its comment says every id and weight is route_multi's, and the switch class is "bitwise by design" (`conversation_disk_env.def:670`) [code].

### Change
None. Switch only: `STRATA_ROUTE_FAST=1` with ROUTE_PUBLISH=1 (already in the profiled config) [profile-environment note].

### Cost and risk
- Lines: 0.
- Ceiling: 78.1 ms = 2.2% of wall [profile].
- Prior arms: +8.0% at n=4 (one pass) [measured]. That exceeds the ceiling, so it is noise or a confound (the base had a 95.9 tok/s outlier) [measured, untested-switch note].
- Whether the fast kernel is taken is not logged by the gate (`native_router.cu:490-492` has no `switch_report` call) [code]. Check for the fast kernel name in the trace instead.

### Test arm
ROUTE_FAST 1 vs 0 on the profiled config, n of 8, A B B A. Confirm by rocprof kernel names (route_publish_res count should fall to 0 in the arm).

### Decision
Run once in the night queue: cheap and exact, but the ceiling is 2.2%, so it is probably inside the noise. Do not build anything.

---

## Item 7: WAIT_COPY256 (with FUSE_WAIT)

### Mechanism
- The wide plan copy (256 threads, 16-byte reads) is used only when `n > 512` words (`verify_kernels.cu:1797-1806`) [code]. The narrow path is `wait_flag_copy_i32_kernel` with 128 threads and 4-byte reads (`verify_kernels.cu:1385-1407`) [code].
- The copy is wired to the WAIT_FUSE path too: `wait_flag_copy_i32` calls `wait_copy_wide` (`verify_kernels.cu:1823-1833`) [code]. So in the profiled config (WAIT_FUSE=1) the switch acts through `wait_flag_copy_i32`, not through FUSE_WAIT. The plan's pairing with FUSE_WAIT is therefore wrong for this config.
- Plan size is `80*T + 42` words (item 0.2) [unverified arithmetic], so the switch acts only for T of at least 6.
- Profile: wait_flag_copy_i32 is 4320 calls, 143.1 ms total, median 9.0 us, and the spin is most of it (the audit: 96 calls above 100 us) [profile].

### Change
Optional, only if the owner wants the switch to act at small T: change the `n > 512` gate in `wait_copy_wide` to a 16-byte-alignment check. About 2 lines. Not recommended by itself, because the ceiling is small.

### Cost and risk
- Ceiling: the copy part is a few us per call. At 4 PCIe read rounds (128 threads, 442 words) against 1 round for the wide path, the saving is about 2 to 3 us per call x 4320 = 9 to 13 ms = 0.3 to 0.4% of wall [unverified: PCIe read latency not measured].
- Exact by class (`conversation_disk_env.def:767`) [code].

### Test arm
Check first with the log: a run with `STRATA_WAIT_COPY256=1` prints "will not act (a plan of N words...)" for T below 6 [code `verify_kernels.cu:1800-1806`]. Only if the log shows it acted on enough windows, run WAIT_COPY256 1 vs 0 with n of 8.

### Decision
Not worth building or measuring to a 5% decision. The ceiling is 0.4%. Note in the plan that the earlier +6.8% pass cannot be this switch's effect.

---

## Item 8: MTP_EAGER on the gap-heavy config

### Mechanism
- The chain loop launches step j+1 only after `wait_step(j)` returns (`mtp.cpp:1936-1950`, wait at `mtp.cpp:1948`) [code]. That is the host round trip behind the 106 us median gap (profile).
- The chain stop decision needs the host (`pj >= min_p`), but the tokens flow on the device through `mtp_select` into `Rin_`/`tok_` [code `mtp.cpp:1549` (comment at `mtp.cpp:1557`: "left in Rin_[0] / tok_[0] by mtp_select")]. So the host round trip is only needed for the stop.
- Eager launches every step graph up front (`mtp.cpp:1863-1875`) and cuts on the host afterwards (`mtp.cpp:1898-1903`) [code].
- Profile: 48 mtp_select -> copy pairs, 17.5 ms total, 0.5% of wall; three outliers account for about 12.4 ms [profile] + [unverified: audit inference].

### Change
None. Switch only: `STRATA_MTP_EAGER=1`, class Speed (`conversation_disk_env.def:494`) [code].

### Cost and risk
- Ceiling: 17.5 ms = 0.5% of wall, minus the cost of steps the host would have cut. With `--spec-min-p 0.5`, eager runs every step and throws away the rest [code]. Cost per step is not measured [unverified]; if it is about 0.6 ms (draft about 2.5 ms per window, gpu_sync note:13 for a window of several steps) the wasted work can cancel the gain [unverified].
- Class conflict: "same drafts" (`mtp.cpp:1859-1862`) vs "the tokens change" (`conversation_disk_env.def:494`). Test tokens.
- PLE hazard, unverified: the eager path prefetches PLE rows for the draft tokens only after the GPU run (`mtp.cpp:1894-1902` region, `prefetch_ple(drafts[last])`), while the non-eager path prefetches `drafts[j-1]` after launching step j (`mtp.cpp:1946`) [code]. If the profiled model has PLE (`ple.ready()` at `mtp.cpp` `do_ple`, `mtp.cpp:1843`), the timing of those prefetches changes for every step. The profiled model's PLE state is unknown [unverified].
- Measured before: +1.6% +-5.2, not significant (`untested_switches` note, test-phase 06c) [measured].

### Test arm
MTP_EAGER 1 vs 0 on the profiled config (`--spec-min-p 0.5`), n of 8, A B B A, with token equality and, if the model has PLE, a token diff on a fixed prompt set.

### Decision
Not worth more machine time unless item 1's stream-ID re-profile shows the 17.5 ms pairs are on the critical path. Keep opt-in. Do not build.

---

## Item 9: merge the three window-input copies

### Mechanism
- Window inputs: `copy_i32_from_mapped(tok_, ...)` on `cs` (`verify.cpp:1731`), and step and position on `in_s`, which is `df_side_[1]` when the branch is on (`verify.cpp:1739-1740`, `in_s` set at `verify.cpp:1732-1737`) [code]. The branch flag is `br` at `verify.cpp:1713` [code].
- The copies are kernel nodes in the window's capture (the capture begins at `verify.cpp:3326`; the recording path is the one that runs under it) [code + unverified placement].
- A merged multi-copy exists and is already capture-safe: `copy_from_mapped_multi` (`elementwise.cu:423-...`, used under capture at `mtp.cpp:1537`) [code].

### Change (only if ever built)
- If `br` is off: one `copy_from_mapped_multi` call with three entries on `cs`. About 8 lines.
- If `br` is on: merge 1739 and 1740 into one call on `in_s` (2 entries), leave 1731 on `cs`. About 6 lines.
- The fork and join around side 1 must stay as they are.

### Cost and risk
- Gain: two nodes per window x 288 windows x about 2.4 us = about 1.4 ms = 0.04% of wall [unverified per-node cost]. The copy kernels total 1.4 ms for 384 calls [profile].
- Risk: low. Bit-exact copies. Alignment fast path differs per entry (`elementwise.cu:423-440`) [code].

### Test arm
Not needed to decide.

### Decision
Not worth building. The gain is 0.04%, far below the 5% noise. Note the copy-node count only if rocprof already shows it.

---

## Item 10: pre-stage host records before the window launch

### Mechanism
- Draft step and position records for all steps are staged on the host before the first launch (`mtp.cpp:1821-1834`: `put` for every row, then `h_out_` reset at `mtp.cpp:1835`) [code]. So the copy after `mtp_select` is not waiting for host staging.
- The next window's token inputs depend on this window's accepted tokens, which come from the GPU argmax and the host sync at `verify.cpp:4173` [code]. These cannot be staged early.
- Two candidate sources for the 83-gap `__amd_rocclr_copyBuffer -> copy_i32_from_mapped` pair (16.9 ms) are host-issued device-to-device copies on the same stream, outside a graph: `stage_source_R` (`mtp.cpp:2010-2014`, `cudaMemcpyAsync` D2D on `cs_` when `src_R_ != window_R_`) and the spec copies (`verify.cpp:4214-4216`, H2D on `cs_`, sampled windows only) [code]. Which one matches needs stream IDs [unverified].

### Change
None designed yet. If item 1's stream-tagged trace shows the pair is `stage_source_R`, the candidate change is to have the source write directly into `window_R_`, or to capture the copy into the round graph. That removes a copy, not a wait.

### Cost and risk
- Ceiling: 17.5 ms + 16.9 ms = 34.4 ms = 1.0% of wall [profile]; the realistic share is smaller, because the 17.5 ms pair is draft turnaround (item 8).
- Lines: 0 to 10, depending on the source. Risk low to medium: capture of a D2D copy inside the round graph needs the source to be stable for replays [unverified].

### Test arm
After item 1 names the call site: staged vs not, A B B A.

### Decision
Not worth building now. The premise is stale (records are already staged), the 17.5 ms pair is draft turnaround that item 8 addresses, and the 16.9 ms pair is not yet identified. Revisit only after the stream-tagged trace.

---

## What the queued results would change

- **Item 2 (POLL1 1 vs unset):** a gain above 5% with identical tokens means the poll traffic was a real part of the wait, and the next lever is pool bandwidth (Worker B's item 11). A null result means the combine time is pool latency (about 96% of the kernel, section 0.7) and the combine line of work is closed. A loss keeps it off.
- **Item 4 (FUSE_WAIT 1 vs 0):** the queued arm is only meaningful with WAIT_FUSE=0 set. Read it as a node-count change (4320 nodes, about 0.3%). If it was run with WAIT_FUSE=1, it measures nothing and should be run again as the 2x2 in item 3.

---

## Summary

| Item | Change (lines) | Switch / arm | Ceiling (wall) | Decision |
|---|---|---|---|---|
| 2 combine POLL1 | 0 | `STRATA_COMBINE_POLL1` 1 vs unset, A B B A n>=8, tokens | 18.9% wait; poll-only share unknown | Keep queued arm; no code |
| 3 WAIT_FUSE | 0 | 2x2 with item 4, finite logits | 0.6% node penalty if off; gain 0-3% (unverified) | Run once in the 2x2 |
| 4 FUSE_WAIT | 0 | 2x2 (WAIT_FUSE 0 required) | 0.3% (WAIT_FUSE=0), 0.02% (=1) | Fold into item 3 |
| 5 --spec 3/4/5 | 0 | `--spec` sweep, tokens vs 4 | unknown; prior "flat" | Skip unless owner asks |
| 6 ROUTE_FAST | 0 | 1 vs 0, n>=8, check kernel names | 2.2% | One night arm; probably noise |
| 7 WAIT_COPY256 | 0 (optional ~2) | check log first | 0.3-0.4% | Not worth building |
| 8 MTP_EAGER | 0 | 1 vs 0, tokens, PLE check | 0.5% minus wasted steps | Not worth more time; keep opt-in |
| 9 merge window copies | ~6-8 | none needed | 0.04% | Not worth building |
| 10 pre-stage records | 0-10 | after item 1 | 1.0% (stale premise) | Not worth building now |

No item in 2-10 is worth a code change on its own. The only candidate that could reach the 5% threshold is the combine wait, and only through pool latency (item 11, Worker B). Switch-only arms are cheap and should go to the night queue in this order: item 3 2x2 (with item 4), item 2, item 6.

---

## Gaps

- No GPU run and no runtime dump of STRATA_* switches; the switch states in the profiled run are config-derived [unverified].
- The 1-per-window site of the 288 standalone `wait_flag_ge` calls is not confirmed (the code path at `verify.cpp:2883-2892` fits, but the group condition was not traced) [unverified].
- Per-node launch cost (2.4 us) is the audit's citation, not re-read here [unverified].
- Whether the profiled model uses PLE (item 8) and the window T distribution at `--spec 4` (items 5 and 7) need a run log.
- `sync_copy_literature.md` was not re-read in full; the report's literature summary was used. The literature gives no single-RDNA measurement of these waits [paper, report].
- Stream IDs for the two copy pairs (item 10) need the re-profile in item 1's follow-up.
