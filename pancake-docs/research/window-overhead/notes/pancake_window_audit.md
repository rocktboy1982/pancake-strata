# Pancake speculative window: per-window sync and copy cost audit

Scope: read-only audit of the verify window and MTP drafter code in the worktree
`/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a` (the tree the profiled binary was built from, per `run_agent_info` / `run.log`), mapped against the rocprofv3 kernel trace in `/home/dan/profile-pancake-2026-11-12/`.

Labels: [code] = file:line in the worktree above; [profile] = profile file; [unverified] = inferred, not confirmed by a run or a log.

Source notes that differ from the assignment:
- `src/program/generate.cpp` has no `wait_flag`, `copy_i32_from_mapped` or `mtp_select` call sites (grep count 0). It only sets window policy and calls `Verifier::set_commit_async` [code: src/program/generate.cpp:6027]. The per-window waits and copies live in `src/core/verify.cpp` and `src/core/mtp.cpp`.
- `include/strata/spec/` holds draft policy and window rules (controller.hpp, window_rule.hpp, draft_policy.hpp, ...). It has no kernel launches.
- `src/core/conversation_disk_env.def` exists and classifies each switch (Key / KeyIfSet / Speed / Unsure / Inert). Switch classes are cited below.

---

## 0. Profile facts used throughout

- [profile] summary.txt: decode 3553 ms wall, kernels 2712 ms (76%), 470 kernels per token. Launch gaps <50 us total 699 ms; host stalls >1 ms total 93 ms.
- [profile] Run flags (run_agent_info / run.log command line): `--spec 4 --spec-min-p 0.5 --mtp ... --kv int8 --kv-resident 32768 --lookup-max-t 12 --pool-affinity measure --pcie-frac 0 --adapt-every 1`. No STRATA_* environment is recorded in the profile directory, so every switch state below is [unverified] against the profiled process itself.
- [profile] Trace counts (run_kernel_trace.csv, recounted with Python): wait_flag_ge_kernel 288; wait_flag_copy_i32_kernel 4320 (= 15 x 288); expdec_down_kernel 4320; swiglu_q8_1_entries_kernel 4320; combine_cpu_k10_vec4<true> 4032 (= 14 x 288); route_publish_res 4032; copy_i32_from_mapped_kernel 384; mtp_select_kernel 247 (run summary counts 4032 for a different kernel set, so treat per-window ratios as approximate).
- [profile] The 288-per-window ratios are consistent with 15 MoE layer groups per window, one plan wait+copy per group, 14 groups taking the fused combine wait and one group taking the separate CPU-share wait. This is an inference from the counts [unverified].

### 0.1 Wait kernels are spin time, not compute

- [profile] wait_flag_ge_kernel: 288 calls, total 150.1 ms, median 11.2 us, mean 521 us, max 8.1 ms. 136 calls exceed 100 us and sum to 149.4 ms. The cost is concentrated in a minority of windows.
- [profile] wait_flag_copy_i32_kernel: 4320 calls, total 143.1 ms, median 9.0 us, 96 calls exceed 100 us (105.1 ms). One sampled instance spans 1350 us.
- [code] verify_kernels.cu:1330-1333 (wait_flag_ge_kernel: spin `while (*flag < value)` then `__threadfence_system`); verify_kernels.cu:1396-1407 (wait_flag_copy_i32_kernel: thread 0 spins, `__syncthreads`, then all threads copy `dst[i] = src[i]` from a mapped host pointer).
- Interpretation [code+profile]: these kernels occupy the GPU stream while the host CPU-expert pool produces the flag. Their time is host-pool latency, not launch overhead. Fusing the copy into the wait (STRATA_WAIT_FUSE) removes one graph node and one copy, but does not shorten the spin.

### 0.2 combine_cpu_k10_vec4<true> is mostly wait

- [profile] combine_cpu_k10_vec4<true>: 4032 calls, 670.6 ms (24.25%), median 75 us, mean 166 us, 1823 calls above 100 us.
- [code] native_moe.cu:166-172: the combine takes `flag`, `value`, `poll_slot`. When `poll_slot < 0` (STRATA_WAIT_FUSE path), every block's thread 0 spins on the flag in mapped host memory: "the CPU-share wait in every block".
- [code] native_moe.cu:267-269 (poll_slot_of): returns -1 unless `STRATA_COMBINE_POLL1=1`. So by default every combine block polls the PCIe-mapped flag.
- [code] native_moe.cu:379 launches `combine_cpu_k10_vec4<true>` (shared gate) and 385 `<false>`.
- Inference [unverified]: a large share of the 670 ms top-kernel figure is spin, not arithmetic. The profile cannot separate spin from arithmetic inside one kernel. A stage stamp or a poll-off run would be needed.

---

## 1. Q1: where wait_flag_ge is launched and what the host waits for

### 1.1 Launch sites of wait_flag_ge (host API) in the window path

| Site | Condition | Waits for | Per window? |
|---|---|---|---|
| verify.cpp:1803 `if (ar_on()) wait_flag_ge(m_flag_, 1, cs);` | layer 1, `ple_on`, grp 0, all-resident | the pool's first flag (PLE rows) | conditional on ar_on (all-resident). Profile shows partial residency for most layers, so [unverified] whether this fires |
| verify.cpp:2841 `wait_flag_ge(m_flagA_, ring, cs)` | plan wait, `device_plan_` off, `g_wait_fuse` off, `fuse_wait_on()` off | GPU plan published by pool | not in profiled config (g_wait_fuse on) |
| verify.cpp:2852 `wait_flag_ge(m_flagB_, ...)` | flag B (PCIe share), pcie-frac 0 | skipped when `pcie_off_` | not in profile (`--pcie-frac 0`) |
| verify.cpp:2883 `if (!g_wait_fuse) wait_flag_ge(m_flag_, ...)` | CPU-share wait, `cfuse_on(n)` branch | CPU rows in mapped memory | not in profiled config (g_wait_fuse on) |
| verify.cpp:2892 `wait_flag_ge(m_flag_, ring, cs)` | CPU-share wait, non-fused combine branch (`cfuse_on(n)` false), `fuse_rows` off | CPU rows in mapped memory | yes: the most likely source of the 288 calls [unverified] |
| verify.cpp:1675/1678, 2706-2715 late_add and defer paths | `STRATA_VERIFY_DEFER_CPU*` | deferred CPU rows | off unless deferral switches are set |

- [code] Non-fused CPU-share wait, verify.cpp:2878-2892. `cfuse_on(n)` is `cfuse_off(n) == nullptr` [code: verify.cpp:1627-1644]; `cfuse_off` returns a reason when the combine is off, the window has n <= 1 or n > 8 rows, the shared expert is not BF16, or the CPU expert layout is not native.
- [code] The 288 count matches one non-combined CPU-share wait per window, followed by `copy_rows_from_mapped` in the trace (sample at about 1.4 ms in the trace excerpt) [profile]. Which layer and row-count triggers it is [unverified]; the dispatch note at verify.cpp:2931 prints the reason and would confirm it.

### 1.2 What the host waits for, per window

- [code] Host-side synchronous waits inside a verify window:
  - verify.cpp:4173 `cudaStreamSynchronize(cs_)` after the window graph, the default blocking wait for the window's GPU work (flags A/B/M raised). A window-end sync is a real data dependency, since the host needs the verifier's argmax rows.
  - verify.cpp:4142 and 4144 (chain / overlapped split path): sync of this stage and its copy stream.
  - verify.cpp:4228 and 4234: head sampling sync when a sampled request needs host sampling.
- [code] Host-side draft waits, MtpDrafter::draft:
  - mtp.cpp:1880-1883 and 1921-1924: spin on `h_out_[j] < 0` for each draft step, with a `cudaStreamQuery` every 1024 spins.
  - mtp.cpp:1891 and 1958: `cudaStreamSynchronize(cs_)` after the chain.
- [code] Host-side commit wait: verify.cpp:4480 `set_commit_async(on)` sets `g_commit_async = on && STRATA_COMMIT_SYNC unset`. generate.cpp:6027 calls `set_commit_async(!multi_gpu)`, so on a single GPU the commit is asynchronous: verify.cpp:4520-4531 records `commit_done_` instead of syncing, and `wait_commit` (verify.cpp:~4545-4553) waits on the event later.
- [code] The GPU-side wait for the host (flags) is the spin kernels in 1.1, not a host sync. The host sync is at the window end (4173) and in the draft loop (1891, 1958).

### 1.3 Are these waits needed for correctness?

- Required (true data dependency): the CPU-share rows (`m_ymiss_`) must be in mapped memory before the combine reads them; the GPU plan must be published before the plan copy. Removing the wait would read stale data. Only the latency can be attacked: start CPU experts earlier, publish the plan earlier, or move work so the combine does not depend on the same layer.
- Bookkeeping, avoidable in principle: the separate `wait_flag_ge` plus `copy_rows_from_mapped` pair (verify.cpp:2892-2895) vs. the fused variant. Fusing removes a graph node (the profile notes ~2.4 us per dependent kernel [code: verify.cpp:418-420]); it does not remove spin.
- Not reachable by switch: the plan wait in device-plan mode still waits for the flag (verify.cpp:2822-2823).

---

## 2. Q2: where copy_i32_from_mapped is issued and what host value it reads

### 2.1 Issue sites (copy_i32_from_mapped in the window / draft path)

- [code] verify.cpp:1731 `copy_i32_from_mapped(tok_, m_tok_, T, cs);` window token input.
- [code] verify.cpp:1739 `copy_i32_from_mapped(step_, m_step_, T * kStepCount, in_s);` window step rows.
- [code] verify.cpp:1740 `copy_i32_from_mapped(pos_, m_pos_, MT * (NH+NKV+IQ), in_s);` window positions.
- [code] verify.cpp:3498 `copy_i32_from_mapped(commit_, m_commit_, 2 + MT, cs_);` first node of the commit graph (`commit_exec_`). It reads `h_commit_[0]` (n_keep), `h_commit_[1]` (n_keep - 1), and the position list `h_commit_[2..]`.
- [code] verify.cpp:4489-4491 and 6078-6080: the host writes `h_commit_[0] = n_keep`, `[1] = n_keep - 1`, `[2+t] = last_pos0_ + t`. The commit graph launch is at verify.cpp:4517 and 6082.
- [code] mtp.cpp:1537 `copy_from_mapped_multi(in, 7, cs_)`: the draft round's seven inputs in one launch (STRATA_MTP_MULTI_COPY, default on per the switch class). mtp.cpp:1568 `copy_from_mapped_multi(in, 2, cs_)`: per chain step, step row and position row.
- [code] mtp.cpp:1503-1505 copies in the prefill graph only (prefill is not a per-window cost).
- [code] layer.cpp:1185-1186 copy_i32 of step and pos for a publish-kernel path (not in the profiled hot path [unverified]).

### 2.2 What value does each copy read

- Window inputs (verify.cpp:1731-1740): the host-staged token, step and position records for the window's T rows. These are host decisions made before the window (which drafts are in the window, their cells). Required: the GPU must see them before the layers run.
- Commit copy (verify.cpp:3498): `n_keep` (the accepted count) and the cell positions. This is the accept decision, a true data dependency for the commit (state advance). It is only available after the window's argmax is read, so the commit cannot start before the window ends.
- Draft copies (mtp.cpp:1537, 1568): the MTP's previous residual row index and token, written by the host from `h_out_`.

### 2.3 The profiled copy after mtp_select (48 pairs)

- [profile] mtp_select_kernel -> copy_i32_from_mapped_kernel: 48 pairs, total gap 17.5 ms (matches the 17.5 ms in the assignment). Gap median 106 us, 45 of 48 between 55 and 186 us. Three outliers: 1893 us, 4639 us, 5824 us. The outliers account for about 12.4 ms of the 17.5 ms.
- [profile] The copy kernel itself is cheap: copy_i32_from_mapped_kernel median 3.0 us, total 1.4 ms over 384 calls. The 17.5 ms is the gap, not the copy.
- [unverified] Which copy_i32 call is the one after mtp_select (window input at 1731/1739/1740, commit at 3498, or another) cannot be determined from kernel names alone, since streams interleave in the trace. The timing signature (short gap, then copy) suggests the host turned the drafter's output around into the next window's input or the commit. The three outliers suggest host preemption or a pool wait outside the GPU.
- Host turnaround that a copy waits behind: the drafter's `h_out_` read (mtp.cpp:1880, 1921) and the window launch. This is host bookkeeping, not GPU work.

---

## 3. Q3: which waits are needed for correctness and which are host bookkeeping

| Item | Profile cost | Class | Reason |
|---|---|---|---|
| wait_flag_ge (288), wait_flag_copy_i32 (4320) | 150 ms, 143 ms spin | Correctness (data), latency-bound | CPU rows and GPU plan must be in mapped memory before use. Cost is pool latency. Fusing removes only the graph node. |
| combine_cpu spin (inside 670 ms) | up to ~all of 670 ms [unverified split] | Correctness (data) + latency | The per-block PCIe poll (poll_slot < 0) multiplies PCIe reads. Reducing to one poller (COMBINE_POLL1) changes the poll, not the dependency. |
| window-end cudaStreamSynchronize (verify.cpp:4173) | host stall, part of the 93 ms | Correctness (argmax needed) | Host needs argmax to accept/reject. |
| commit copy_i32 (verify.cpp:3498) | small kernel, host-driven | Correctness (accept count) | n_keep is the host's decision; async commit already removes the host sync on one GPU (set_commit_async). |
| copy_i32 after mtp_select (48 pairs, 17.5 ms) | gap, mostly outliers | Bookkeeping (host turnaround) | The copy is 3 us; the gap is host decision + launch latency. |
| copy_from_mapped_multi (mtp.cpp:1537, 1568) | 1532 copy_from_mapped kernels, 2.7 ms | Bookkeeping | Already merged (MULTI_COPY). Small. |
| kv_append_q8 -> copyBuffer (7 calls, 58 ms) | one outlier of 57.3 ms dominates | Not per-window [profile] | The gap list is [-3, ..., 205, 57317] us; a single 57 ms stall, not steady-state. Likely a one-time event (prefill or pool). Excluded from per-window cost. |

---

## 4. Q4: existing switches (conversation_disk_env.def) that target these waits

Class from `src/core/conversation_disk_env.def`. Only Speed switches are safe for bit-exact comparisons; Key switches change output.

| Switch | Class (def line) | Code | What it targets | Profile state |
|---|---|---|---|---|
| STRATA_WAIT_FUSE | Speed (768) | verify.cpp:1623, 2827-2828, 2883, 2946-2947; verify_kernels.cu:1396, 1823-1830 | Fuses the plan wait with the plan copy (wait_flag_copy_i32) and makes combine poll the CPU flag in every block (poll_slot < 0) | [profile] 4320 wait_flag_copy_i32 calls, so it is on [unverified: no env record] |
| STRATA_FUSE_WAIT | Speed (403) | verify.cpp:441-462 (fuse_wait_on, gfx12 default), 2830 | Wait+copy fusion for the non-WAIT_FUSE path; HIP default only on gfx12 | gfx1100 default off [code]; not in profile (no wait_copy_i32_from_mapped_kernel in trace) |
| STRATA_WAIT_COPY256 | Speed (767) | verify_kernels.cu:1770-1822 (wait_copy_i32_wide_kernel, wait_copy_wide) | 256-thread wide plan copy for plans > 512 words | Plan sizes not measured here [unverified] |
| STRATA_COMBINE_POLL1 | Speed (338) | native_moe.cu:147, 267-269, 379 | One block polls the PCIe flag, others spin on VRAM; cuts PCIe poll traffic | Default off [code]; not confirmed for profile |
| STRATA_VERIFY_DEVICE_PLAN | Speed (763) | verify.cpp:2822-2826 | Device-side plan: skips the plan copy when the GPU planned the group | No copy_i32_unless kernel in trace; [profile] off [unverified] |
| STRATA_VERIFY_DEFER_CPU, _BELOW, _TOP | Key (42-44) | verify.cpp:1675-1678, 2691, 2868 | Defers CPU rows to the next layer (removes in-layer wait) | Key class: output changes; not for bit-exact tests |
| STRATA_VERIFY_DEFER_AUTO, _MIN_T | KeyIfSet (160-161) | verify.cpp | Auto-deferral by window | Output changes |
| STRATA_COMMIT_SYNC | Speed (340) | verify.cpp:4480 | Forces synchronous commit (the opposite of the async default) | Off on single GPU by default [code] |
| STRATA_MTP_EAGER | Speed (494) | mtp.cpp:1859-1863 | Launches all draft step graphs at once (removes the host round trip between steps) | Targets the 106 us gaps; default off [code] |
| STRATA_MTP_MULTI_COPY | Speed (498) | mtp.cpp:1537 | One launch for the seven draft inputs (default on) | Default on [code] |
| STRATA_ROUTE_PUBLISH | Speed (671) | native_router.cu:217; native_bf16.cu:122 | route_publish_res + doorbell in one launch | route_publish_res 4032 calls, 78 ms [profile] |
| STRATA_EXPDEC_DOWN | Key (22) | iq_kernels.cu:4257, 4328 | Enables expdec_down path | Key class; in profile (expdec_down present) |
| STRATA_DF_PDL | Speed (371) | verify.cpp:~1448 (PdlScope) | Programmatic dependent launch | pdl_supported is sm_90+ [code: per comment]; not on gfx1100 [unverified] |
| STRATA_PIPED_MTP | Speed (591) | not traced here | Pipelined MTP drafting | [unverified] whether it applies to the single-GPU decode path |

---

## 5. Kernel-to-code map (the profile's costly kernels)

| Kernel (profile) | Time (profile) | Launch site(s) | Host wait it sits behind |
|---|---|---|---|
| expdec_gu_kernel<22,1,4,false,1> | 217 ms, 8.0%, 3060 calls | iq_kernels.cu:6920 (expdec_gu), 4285-4296 host dispatch, 4475 | none (GPU compute) |
| expdec_gu_kernel<16,...> | 59 ms, 2.2%, 990 calls | same | none |
| expdec_down_kernel<42,4> | 158 ms, 5.8%, 4320 calls | iq_kernels.cu:7006, 4522 | none; followed by swiglu/route per trace |
| wait_flag_ge_kernel | 150 ms, 5.5%, 288 calls | verify.cpp:2892 (likely), 1803, 2841, 2883 | CPU-share flag (pool) |
| wait_flag_copy_i32_kernel | 143 ms, 5.3%, 4320 calls | verify.cpp:2828 (g_wait_fuse) | GPU plan flag (pool) |
| copy_i32_from_mapped_kernel | 1.4 ms total, 384 calls; 17.5 ms in gaps after mtp_select | verify.cpp:1731/1739/1740, 3498; mtp.cpp:1503-1505 (prefill only) | host records staged in mapped memory (window inputs, commit count) |
| combine_cpu_k10_vec4<true> | 671 ms, 24.7%, 4032 calls | native_moe.cu:379-383; verify.cpp:2881-2885, 2944-2947 | CPU rows flag; per-block poll with STRATA_WAIT_FUSE |
| route_publish_res | 78 ms, 2.9%, 4032 calls | native_router.cu:217 | none (router + doorbell) |
| kv_append_q8_kernel<true> -> copyBuffer | 58 ms, 7 calls (one 57 ms gap) | verify.cpp:2215-2244 (kv_append_q8_steps / step) | not per-window: single outlier |

---

## 6. Gaps and what would settle them

- [unverified] Exact STRATA_* environment of the profiled process. Needed to fix which of the WAIT_FUSE / COMBINE_POLL1 / VERIFY_DEVICE_PLAN branches ran. The profile directory records CLI flags only.
- [unverified] The site of the single per-window wait_flag_ge (288 calls). Confirm with the dispatch note at verify.cpp:2931 (reason cfuse off) or a trace of the STRATA_VERIFY_PROFILE stamps.
- [unverified] Which copy_i32 call follows mtp_select in the 48 pairs. Needs a stream-tagged trace (Stream_Id is in the CSV but the pairs here were matched by time only).
- [unverified] Split of combine_cpu time between spin and arithmetic. A run with STRATA_COMBINE_POLL1=1 versus unset, or a stage stamp, would separate them.
- [unverified] Whether STRATA_MTP_EAGER and STRATA_PIPED_MTP apply to this configuration (the profile uses --spec-min-p 0.5, which the EAGER note says it also covers, verify mtp.cpp:1859-1863).
- Not checked: the 48 copies may be commit copies from a second verifier path (verify.cpp:4665, 4867, batch commit), which were not traced to the profile.

Benchmarks were not run (read-only audit). Nothing in the repository was built, checked out, or edited.
