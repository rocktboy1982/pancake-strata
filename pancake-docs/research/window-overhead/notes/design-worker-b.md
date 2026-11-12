# Phase 18 items 11-20: implementation design (Worker B)

Scope: design only. No repo file edited except this one, no build, no engine run, no GPU use, no commit.
Worktree read: `/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-ad25cdf1dc7ea704a` (branch dev/p17/integration). Line numbers below are from that worktree.

Labels: [code file:line] = read in the worktree; [profile] = `pancake-docs/research/bench/profile/summary.txt`; [measured] = a result file cited in `notes/untested_switches.md`; [paper] = abstract or search excerpt only, as the research notes state; [unverified] = not checked by a run or by a primary source.

Each item uses the same five parts: Mechanism, Change, Cost and risk, Test arm, Decision.

---

## Summary

| # | Item | Ceiling (wall) | Exact class | Decision |
|---|---|---|---|---|
| 11 | Pool thread pinning / affinity | 8.3% (293 ms wait kernels) | exact (scheduling) | Test existing flags first (`--host-core`, `--pool-workers`). No new code yet. |
| 12 | Event + device wait replaces window-end sync | 2.6% (93 ms host stalls) | exact | Not worth building alone. A host read of argmax is still needed. Only worth it bundled with 17. |
| 13 | VERIFY_DEVICE_PLAN | <0.5% (copy node only) | exact by class | Not worth building. Keep the switch, no arm unless 1 shows a copy hot spot. |
| 14 | ROUTER_FUSE without deferral | about 0.6-1% (route launch gap) | exact by class (check output) | One check: confirm the switch acts with the "acted" line, then one A/B. Below noise on current evidence. |
| 15 | PIPED_MTP | not on the profiled path | claimed exact (unverified) | Not worth building for this profile. The switch is inert without `--batch-mtp` and layer-split stages. |
| 16 | Device-side wait via `hipStreamWaitValue32` | about 0.3% (node cost); latency unchanged | exact (same dependency) | Design only. HIP symbol exists in headers. Build only as a cleanup or occupancy fix, after 11. |
| 17 | Merge commit into window graph, device-side accept | at most 2.6% plus one launch | exact only if device accept = host accept | Not worth building now. Commit is already async. The next window still needs the host accept. |
| 18 | Fused argmax + accept | near 0 on greedy | exact if tie-breaks match (unverified) | Not worth it on greedy. Sampled-request design only. |
| 19 | Defer CPU rows (VERIFY_DEFER_CPU) | removes an in-layer wait | LOSSY | Do not build or re-run. Already measured flat-to-negative. |
| 20 | Mirror-SD / PipeSpec branch rollouts | claimed 2.8-5.8x, not on this hardware | claimed exact | Design study only. |

Two findings change the plan as written:
1. `--pool-affinity measure` is a no-op on this machine. The Ryzen 7 5700G is not a hybrid CPU, and `measure` returns `All` with a message [code src/program/generate.cpp:2820]. The profiled run used `--pool-affinity measure` [profile run flags, from `run_agent_info`]. So the profiled pool layout was `all`, one worker per physical core [code src/program/generate.cpp:1328]. (The "not a hybrid one" line should be in the profiled run's log; check it.)
2. `STRATA_PIPED_MTP` is only read under `--batch-mtp` with layer-split stages [code src/program/generate.cpp:7194, 7245]. It is not on the single-stream decode path in the profile.

---

## Shared facts used by several items

- Profile: decode 3553 ms wall, kernels 2712 ms (76%), 470 kernels per token, launch gaps <50 us 699 ms, host stalls >1 ms 93 ms [profile].
- `wait_flag_ge_kernel`: 288 calls, 150.1 ms (mean 521 us, median 11.2 us) [profile]. The kernel is `while (*flag < value) strata_spin_pause(); __threadfence_system();` [code src/kernels/cuda/verify_kernels.cu:1330-1333].
- `wait_flag_copy_i32_kernel`: 4320 calls, 143.1 ms [profile]. Thread 0 spins, then all threads copy from mapped host memory [code src/kernels/cuda/verify_kernels.cu:1396-1407].
- `combine_cpu_k10_vec4<true>`: 4032 calls, 670.6 ms, 18.9% of wall [profile]. With `STRATA_WAIT_FUSE` on and `STRATA_COMBINE_POLL1` unset, every block's thread 0 polls the mapped flag [code src/kernels/cuda/native_moe.cu:166-172, 267-269]. How much of the 670.6 ms is spin is [unverified]: no run has split it.
- The profiled switch state, from the config file and code defaults [profile-environment.md]: WAIT_FUSE=1 [config], ROUTE_PUBLISH=1 [config], EXPDEC_DOWN=1 [config], POOL_FUSE_PHASES=1 [config]; FUSE_WAIT, DEFER_CPU, COMBINE_POLL1, WAIT_COPY256, ROUTE_FAST, DEVICE_PLAN, MTP_EAGER off [default]. Note: the engine does not print these switch states, so this table is from config and code, not a runtime dump [unverified].
- Config conflict: `notes/untested_switches.md` says VERIFY_DEFER_CPU is "in fast-hip preset". The file `configs/pancake/flashnext-iq2_xs-fast-hip.json` has no VERIFY/DEFER key. It sets `STRATA_WAIT_FUSE`, `STRATA_ROUTE_PUBLISH` and `--pool-affinity` [code]. The preset loader may add it elsewhere [unverified]. The profile note says the profiled config was the fast-hip file with a deferral line removed; the current file has none. Resolve before any arm.
- Watchdog: `release_gpu_waits` writes `UINT32_MAX` into every host flag word and then waits for the streams to drain [code src/core/verify.cpp:612-625]. Any new wait (item 16) must read the same words, or the #267 release stops working.
- Window count: `wait_flag_ge` is 288 calls, consistent with one window group per window [profile, audit inference]. Treat 288 as approximately the number of windows in the profiled run [unverified].

---

## Item 11. Pool thread pinning and affinity

### Mechanism
- The pool pins workers with `pthread_setaffinity_np` through `pin_thread_to_cpu` [code src/kernels/cpu/pool.cpp:619, 694; src/kernels/cpu/pool_affinity_linux.hpp:60-66].
- The host thread's core is chosen by `--host-core first|last|sibling` or `STRATA_HOST_CORE` [code src/program/generate.cpp:4333-4336; src/kernels/cpu/pool.cpp:113-117, 231-243].
- Worker count default: every physical core except the one the host loop spins on [code src/program/generate.cpp:1302-1306]. On this 8-core, 16-thread 5700G that is 7 workers plus the host thread.
- `--pool-affinity auto|p-cores|measure` only changes the layout on a hybrid CPU. `measure` on a non-hybrid CPU returns `All` [code src/program/generate.cpp:2818-2822]. `auto` on a homogeneous part is expected to give the same layout as `all` [unverified: read `detect_cpu_topology` for the non-hybrid branch].
- Pool workers also spin between windows. `STRATA_POOL_MWAITX` and `STRATA_POOL_UMWAIT` park them instead of spinning, and they are both default off [code src/kernels/cpu/pool.cpp:43-46, 78-80]. The MWAITX comment says the spinning workers sit on the SMT siblings of the host thread.
- The GPU-side time the profile charges to pool latency is the two wait kernels, 293 ms, 8.3% of wall [profile]. Pinning changes only when a worker runs, not how much work it has.

### Change
No code is required to test the hypothesis. Use the existing knobs:
- `--host-core first` (default), `last`, `sibling`.
- `--pool-workers 6`, `7` (default on 8C/16T), `8`, and `14` if SMT siblings are allowed.
- `STRATA_POOL_MWAITX=1` only as a separate arm. The comment says it saves power, not latency.

If the existing knobs show a real effect, a new mechanism is a second step: pin the GPU-driver-facing host thread on the physical core whose sibling is idle, and keep workers off that sibling. That needs a topology test on this 5700G before code.

### Cost and risk
- Cost: arms only. Each run costs a full quiet-machine A B B A block.
- Risk: medium. Pinning fewer workers lowers CPU expert throughput, which can increase the wait the GPU sees. Results must be read as a net effect.
- Exactness: scheduling only. Pool results are order-independent by the pool's own reduction discipline [unverified: not re-read for this item]. Token equality check still required.
- Deadlock: none added.

### Test arm
- Arm A: `--host-core first --pool-workers 7` (current default).
- Arm B: `--host-core last --pool-workers 7`.
- Arm C: `--host-core sibling --pool-workers 7`.
- Arm D: `--pool-workers 6`.
- Per arm: rocprofv3 wait-kernel time (`wait_flag_ge` plus `wait_flag_copy_i32`) and decode tok/s, n=8, A B B A, discard the first run [owner rule, MEMORY feedback_discard-first-run].
- Token diff must be zero for all arms.
- Run after item 1 (environment record) and after the 288 call site is confirmed.

### Decision
Worth testing with existing flags (arms A-D). Expected ceiling is 8.3% of wall, realistic result unknown. Do not write new pinning code until arms A-D show a difference above noise. This item is not a no-op on the 5700G; it is a test of existing knobs. The profile's `--pool-affinity measure` did nothing, so the first measured arm should use `--pool-affinity all` explicitly, to remove the ambiguity from the record.

---

## Item 12. Event plus device-side wait replacing the window-end sync

### Mechanism
- After the window graph is launched, the host blocks in `cudaStreamSynchronize(cs_)` [code src/core/verify.cpp:4173]. The comment there says the wait stays blocking on purpose: "a cudaStreamQuery poll here cost IQ3_S ~3% decode (a core calling the driver beside the expert workers)" [code src/core/verify.cpp:4165-4167].
- The host needs the window's argmax rows (`m_out_`, the mapped `h_out_`) to accept, to emit tokens, and to draft the next window. So the host has to wait for GPU completion of the argmax before it can continue. A device-side wait does not remove that dependency.
- The commit is already asynchronous on a single GPU: `set_commit_async` records `commit_done_` instead of syncing [code src/core/verify.cpp:4520-4531; src/program/generate.cpp:6027]. So the only remaining host sync at the window end is the argmax read.
- Host stalls above 1 ms total 93 ms (2.6% of wall) [profile].

### Change
Replace `cudaStreamSynchronize(cs_)` with `cudaEventRecord(e, cs_)` followed by `cudaEventSynchronize(e)`, or with a device-side wait for the next consumer. Either way the host still blocks until the window is done.

Two variants are possible:
- (a) event with blocking sync: no change in wake-up cost. Expected gain zero.
- (b) the host polls a flag written by the GPU (a mapped word written at the end of the window). This is the case the comment at line 4165 warns about (driver calls beside the expert workers).

Only a change that lets the host do useful work between the window launch and the argmax read would reduce the stall. The only such work is the host-side commit and draft preparation, which needs the argmax first. So no variant removes the stall alone.

### Cost and risk
- Cost: small code change, one file.
- Risk: medium. Event handles need the same lifetime rules as `commit_done_`. The release path (`release_gpu_waits`) waits with `cudaStreamQuery`, and an event-based sync must also honour the 5 s budget [code src/core/verify.cpp:612-625].
- Exactness: exact, only the wake-up path changes.
- Deadlock: a new event wait must be released by the watchdog exactly as the stream sync is.

### Test arm
Only if combined with item 17. Alone: event vs sync, n=8, A B B A, token equality. Expected result: no difference.

### Decision
Not worth building alone. Ceiling 2.6% [profile], realistic gain near zero because the argmax read is a true dependency and the blocking wake-up cost is the same. Revisit only with item 17, where the next window could be queued before the host read.

---

## Item 13. VERIFY_DEVICE_PLAN

### Mechanism
- `device_plan_` is set only when `STRATA_VERIFY_DEVICE_PLAN=1` and the layer group is not all-resident and not `always_publish_` [code src/core/verify.cpp:1457-1458].
- In the partial-residency branch, when `device_plan_` is set, the code does `wait_flag_ge_or(m_flagA_, ...)` and `copy_i32_from_mapped_unless(...)` [code src/core/verify.cpp:2822-2826]. So the wait is kept. Only the plan copy is skipped when the device planned the group (all experts of that group resident).
- The comment says the switch is "neutral on RIBPC 1-2 GPUs: off by default" [code src/core/verify.cpp:1454-1455]. Not measured on gfx1100.
- The profile copy cost: `copy_i32_from_mapped_kernel` total 1.4 ms over 384 calls [profile]. The 17.5 ms gap after `mtp_select` is a gap, not copy time [profile, audit].

### Change
None required. Enable the existing switch for one arm.

### Cost and risk
- Cost: one arm.
- Risk: low-medium. The `always_publish_` guard means the switch is off when windows always publish (pipelined). The code comment at line 1455 says the device table may lag the host's while windows are in flight, so the switch is off in that case.
- Exactness: exact by class (the same plan is computed on device).
- Gain: the wait remains, so only one copy node per device-planned group is removed. The profile has about 384 copy_i32 calls, about 4 us each [profile]. Ceiling under 0.5%.
- Residency: the switch helps only layers with all experts resident. The audit says the profile shows partial residency for most layers [unverified].

### Test arm
`STRATA_VERIFY_DEVICE_PLAN=1` vs unset, n=8, A B B A, token diff zero. Only if item 1 shows the copy count is larger than the 384 in the summary.

### Decision
Not worth building. Ceiling under 0.5%, below the ~5% noise. Keep the switch as is. Run one arm only if the owner wants the record closed.

---

## Item 14. ROUTER_FUSE without deferral

### Mechanism
- `STRATA_ROUTER_FUSE=1` puts the router GEMV's top-10 and doorbell publish in the GEMV's last block [code src/kernels/cuda/native_bf16.cu:122, 504-511].
- It acts only when the publish path is on (`rpub`): `STRATA_ROUTE_PUBLISH=1`, rows 2-8, no `publish_off` reason (no device plan, no always-publish, no route prior, no resident margin above 0, no remote opt, not all-resident), no deferral (`!defer_`), and on HIP no `STRATA_DOORBELL_STORE` [code src/core/verify.cpp:2517, 2528-2558; reasons at 2545-2556, doorbell gate at 2620].
- `STRATA_ROUTE_PUBLISH=1` is in the fast-hip config [config]. `STRATA_DOORBELL_STORE` is off by default [code src/core/verify.cpp:322].
- The earlier screen found +0.4 to +4.3% and "probably dead in preset" [measured, untested notes]. The 2026-11-11 `switch_report` commit says ROUTER_FUSE was a silent no-op under deferral [code include/strata/core/switch_report.hpp:3]. Without deferral, the gate may now be live. Whether it acts in the profiled run is [unverified]: check the once-only "acted" line in the log.
- Profile: `route_publish_res` 78.1 ms, 4032 calls [profile]. The gap `bf16_f32_mmvf_multi_kernel -> route_publish_res` is 22.5 ms over 3455 gaps [profile].

### Change
None, except the "acted" log line already in the code (`switch_report`) must be read. If it acts, its effect is one launch per layer per window removed.

### Cost and risk
- Cost: one arm, n=8.
- Risk: medium. The router top-10 is an exactness-sensitive selection. Expert choice must be bit-identical. The publish path under the fused launch has a documented history of non-finite-logit suspicion [MOD_PLAN 166, 1599; untested notes].
- Exactness: exact by class, but the notes say "check output". Token diff is required, and the top-10 ids must match the unfused route.
- Ceiling: removing one launch per layer-window saves about 22.5 ms of gap plus the publish kernel's share, at most about 0.6-1% of wall [derived from profile]. Below noise.

### Test arm
`STRATA_ROUTER_FUSE=1` vs unset, with `STRATA_ROUTE_PUBLISH=1` in both, no deferral, n=8, A B B A. Token diff zero. Also check the "acted" line in both logs.

### Decision
Not worth a large effort. One check, then one A/B if the "acted" line shows the switch live. The ceiling is about 1% and the risk is in the route selection, so the test must include a token diff, not only tok/s.

---

## Item 15. PIPED_MTP

### Mechanism
- The switch is read in `generate.cpp:7194` into `bin.mtp_piped`, and the pipelined draft path is set up only when `batch_mtp && !stages.empty() && batch_groups_for(true).mtp_piped` [code src/program/generate.cpp:7194, 7245].
- Its comment at 7242 says it is for "--batch-mtp's proposals in the pipelined groups of a layer split, one slot per group" [code src/program/generate.cpp:7242-7246].
- The header note at `verify.cpp:4759` says "commits after the verdict" for `init_deferred_commit` [code src/core/verify.cpp:4759-4764]. That is the deferred commit path for the batched pipeline.
- `conversation_disk_env.def:591` lists PIPED_MTP as class `Speed` with the description "drafting/stop policy: the tokens change, the state of given tokens does not" [code src/core/conversation_disk_env.def:591]. The description says the tokens change, which is not the class Speed means in this table. This is a labelling conflict to fix in the table.
- The profiled run used single-stream `--spec 4` with `--mtp`, no `--batch-mtp` [profile run flags]. So PIPED_MTP is inert there [code, derived].
- AMUSD (the paper with the 29% claim) decouples draft and verify on separate devices [paper, arXiv:2410.17375, excerpt]. That needs a second device or stream with a shared buffer. Not the single-GPU profile.

### Change
None.

### Cost and risk
- Cost: an arm would need `--batch-mtp` and a layer split, which is a different configuration from the profile.
- Risk: high. The description says tokens change. The batched path has its own correctness tests [code src/program/batch_groups_test.cpp].
- Exactness: claimed exact (AMUSD excerpt), unverified for this code.

### Test arm
None on this profile. A separate study would need `--batch-mtp` with a layer split on the same card, which is not the decode path being optimised.

### Decision
Not worth building for this profile. The switch is inert on the single-stream decode the profile measured. Fix the `.def` description (class Speed vs "tokens change").

---

## Item 16. Device-side wait via `hipStreamWaitValue32`

### Mechanism
- The two spin kernels run on the stream: `wait_flag_ge_kernel` (one thread, `<<<1,1>>>`) [code src/kernels/cuda/verify_kernels.cu:1330, 1820-1821] and `wait_flag_copy_i32_kernel` (`<<<1,128>>>`) [code src/kernels/cuda/verify_kernels.cu:1396, 1829]. Both spin on a mapped host word and occupy one CU slot while they spin.
- A stream memory op (`cuStreamWaitValue32` in CUDA, `hipStreamWaitValue32` in HIP) blocks the stream on a memory word without a kernel. Batched ops reduce overhead [paper/doc: CUDA driver API MEMOP page, per the literature notes]. The CUDA docs warn of deadlock on misuse.
- HIP check (requested by the assignment). Headers are installed under `/opt/rocm/core-10.0/include/hip/`:
  - `hipStreamWaitValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags, uint32_t mask)` declared at `hip_runtime_api.h:3326`.
  - `hipStreamWaitValue64` at line 3365; `hipStreamWriteValue32` at 3388; `hipStreamBatchMemOp` at 3430.
  - Flags: `hipStreamWaitValueGte` = 0x0 (line 1033), `hipStreamWaitValueEq`, `hipStreamWaitValueAnd`, `hipStreamWaitValueNor`.
  - Capability query: `hipDeviceAttributeCanUseStreamWaitValue` at line 585, and the doc at 3317 and 3356 says support must be queried with `hipDeviceGetAttribute`.
  - The header marks the API as Beta (doc comment before line 3326) [code, header].
  - Also present in `hip_prof_str.h` (`HIP_API_ID_hipStreamWaitValue32`) and `hip_api_trace.hpp` [header].
  - Not checked: whether `hipDeviceAttributeCanUseStreamWaitValue` returns 1 on gfx1100. That needs a runtime query, which is a GPU run. Not done here, per the rules.
  - Version note: the headers are the `amdrocm-runtime-dev10.0` package (10.0.0-4), while `dpkg` also lists `libamdhip64` 7.1.0 (Ubuntu). Which runtime the engine links was not checked [unverified].
- The wait word is a mapped host word. For a device pointer the code must pass the device address from `hipHostGetDevicePointer` (the same rule as CUDA's pinned/registered memory, per the docs in the literature notes). Not verified for HIP mapped memory on gfx1100 [unverified].
- `release_gpu_waits` writes `UINT32_MAX` into `h_flag_`, `h_flagA_`, `h_flagB_`, `hflag_in_h_`, `h_pflag_` [code src/core/verify.cpp:612-625]. With `hipStreamWaitValueGte` and `UINT32_MAX` the wait is released, so the watchdog would still work if the new wait reads these same words.

### Change
Replace the spin body with a stream wait:
- `hipStreamWaitValue32(stream, flag_dev_ptr, value, hipStreamWaitValueGte, 0xFFFFFFFF)` before the copy.
- Keep a short fence kernel for the acquire side. The existing copy kernel has `__threadfence_system()` after the spin for every wave [code src/kernels/cuda/verify_kernels.cu:1396-1407]. A stream wait does not give that ordering on its own [unverified].
- Keep `wait_flag_copy_i32_kernel` for the copy; only the spin is replaced. That leaves one fence kernel and one copy kernel instead of one spin-copy kernel.

### Cost and risk
- Cost: medium. Needs `#if` branches for CUDA (`cudaStreamWaitValue32`) and HIP, device-pointer plumbing for every flag, and the capability check at start.
- Risk: medium-high.
  - Deadlock: a stream wait with no timeout blocks the stream forever if the host never writes the flag. The watchdog releases it only if it writes the same word (see above). Test the #267 release path with the new wait.
  - Acquire ordering: the copy after the wait must still see host-written data. Test by token equality under WAIT_FUSE.
  - Graph capture: a stream wait inside a captured graph needs graph-node support. Not checked for HIP. The window graph is captured [code src/core/verify.cpp:3498 shows capture for commit]. Check capture support before any change.
  - Beta API: the header calls it Beta.
- Exactness: same dependency, so exact.
- Gain: the nodes removed are about 4608 (288 + 4320) at about 2.4 us per dependent kernel [code comment at verify.cpp:418-420, as cited in audit]. That is about 11 ms, about 0.3% of wall [derived]. The latency the GPU waits for the pool is unchanged, so the 293 ms of wait time stays [derived]. The only other possible benefit is freeing a CU slot while spinning, which matters only if a second stream runs concurrently (the draft or copy stream). Not measured.

### Test arm
Spin kernel vs `hipStreamWaitValue32` plus fence kernel, for one window, then n=8 A B B A. Token equality. A #267 release test (kill the pool mid-window and check the watchdog returns). Capability check first.

### Decision
Design only. Ceiling about 0.3% for node cost. Build only if item 11 shows the spin occupancy matters for the draft or copy stream, or if the owner wants the spin removed for the watchdog's sake. HIP symbol exists in the header; runtime support on gfx1100 is unverified.

---

## Item 17. Merge commit into the window graph with device-side accept

### Mechanism
- The commit is a separate graph: `copy_i32_from_mapped(commit_, m_commit_, 2 + MT, cs_)` is its first node [code src/core/verify.cpp:3498]. The host writes `h_commit_[0] = n_keep`, `[1] = n_keep - 1`, and the positions [code src/core/verify.cpp:4489-4491, 6078-6080].
- The commit graph is launched after the window's argmax is read on the host [code src/core/verify.cpp:4505-4531]. It is asynchronous (`commit_done_` event), so the host does not wait for it on a single GPU.
- The host accept (`n_keep`) comes from the argmax rows the host read at the window end [code src/core/verify.cpp:4173-4180 region]. The next window's inputs (token, step, position copies) are staged from host values [code src/core/verify.cpp:1731, 1739, 1740], and the MTP drafter reads `h_out_` on the host [code src/core/mtp.cpp:1880-1883, 1921-1924].
- So the host needs the accept decision before it can build the next window. The commit is not on that critical path, but the next window is.

### Change
A device-side accept would compute `n_keep` from the argmax rows and the draft tokens in a small kernel inside the window graph, and write `commit_` directly. This removes the host write `h_commit_` and the copy node.

It does not remove the window-end host read (streaming output, stop checks, MTP draft and next-window inputs need it). So the host sync stays.

### Cost and risk
- Cost: high. The accept logic (prefix match of draft and argmax, stop tokens, max length, sampled-request branch) must run on device, or the graph must split on sampled requests. Graph capture of a data-dependent count needs device tensors for every use.
- Risk: high.
  - Exactness: greedy accept is a prefix compare of integer ids. Device and host must use the same argmax tie-break. The code says `argmax_rows` is "bitwise the one-block kernel's" [code src/core/verify.cpp:3155]. A device accept that reads the same `m_out_` ids is exact. Anything that re-derives argmax on device must be checked on ties [unverified].
  - Stop tokens, max length, and lookup-table bookkeeping stay on the host, so the device accept is only partial.
- Ceiling: the window-end sync stall, 93 ms (2.6%) [profile], is the upper bound for the host stalls. The commit copy node is about 3 us each [profile]. The realistic gain is a fraction of the 2.6% because the next-window dependency stays.

### Test arm
Device vs host commit, greedy only, n=8, A B B A. Token equality over all windows. A tie-heavy prompt set for the argmax check.

### Decision
Not worth building now. The commit is already async, so the remaining cost is a copy node and a graph launch, which is small. The large term is the next-window dependency. Revisit only after item 12's test shows the stall can be overlapped.

---

## Item 18. Fused argmax and accept (SonicSampler-style)

### Mechanism
- The greedy path already computes argmax on device: `argmax_rows(head_logits_, T, n_vocab_, arg_scratch_, m_out_, cs)` when `argmax_rows_wanted()` [code src/core/verify.cpp:3155-3156]. The output is the mapped `m_out_`.
- The window-end sync is the only read of `m_out_` on the greedy path [code src/core/verify.cpp:4173].
- The host head-sampling sync at `verify.cpp:4228` and `4234` is inside the sampled-request branch (`judged`, `sample_tokens`) [code src/core/verify.cpp:4222-4238]. Greedy requests do not reach it.
- SonicSampler fuses sampling and speculative verification into one kernel and reports up to 10x for top-k and up to 16x over baselines [paper, arXiv:2607.20475, abstract only]. Its baselines and exactness are not in the abstract. Its gains are for sampling, not for window latency [paper, abstract].

### Change
For greedy: none needed beyond item 17. For sampled requests: a fused argmax + accept kernel for `sample_tokens`. Not in the profiled greedy path.

### Cost and risk
- Cost: medium for the sampled path; zero for greedy.
- Risk: medium. Tie-break and RNG order must match the unfused sampler, or the sampled stream changes. Exactness here means "same distribution and same seed stream", not greedy token identity.
- Ceiling on greedy: near zero. The argmax kernel is small and already one launch [derived].

### Test arm
Sampled requests only, with a fixed seed and a token-stream comparison. Not on the greedy profile.

### Decision
Not worth it on the greedy profile. Design-only for the sampled path. If the owner later runs sampled requests in the window path, measure the 4228/4234 sync first.

---

## Item 19. Defer CPU rows to the next layer (VERIFY_DEFER_CPU)

### Mechanism
- `STRATA_VERIFY_DEFER_CPU=1` (or `check`) defers CPU-expert rows to the next layer [code src/core/verify.cpp:977-979]. `STRATA_VERIFY_DEFER_CPU_BELOW` sets a routing-weight threshold (default 0.1) [code src/core/verify.cpp:983-986].
- It removes the in-layer `wait_flag_ge` for those rows. It changes output, because the rows are applied one layer later [class Key, conversation_disk_env.def:42-44].
- Measured: edit +15.8% ±3.9 but short result not significant; `DEFER_AUTO` was worse than always-on by -1.2% to -12.5% [measured, night queue defer-iq2xs-03C]. Rank-guard top-2 was -4.2% on edit by recomputed means [measured]. The `ROUTE_PUBLISH` path is forced off under deferral after a non-finite-logit bisect [code src/core/verify.cpp:2559-2562 (`!defer_` in the `rpub` gate); MOD_PLAN 166].

### Change
None. Do not build or re-run.

### Cost and risk
- Lossy: output changes. Token diff is always non-zero by design.
- Risk: medium. Non-finite logits were seen under deferral with the publish path.
- Also: deferral turns off three speed switches (`COPY256`, `IC_WARM_MB`, `ROUTER_FUSE`) [code include/strata/core/switch_report.hpp:3; MOD_PLAN 1566], so the deferred arms measured less than they claimed.

### Test arm
None proposed. If the owner wants a lossy arm later, it must report token diff, the non-finite check, and the three switches above under the same config.

### Decision
Not worth building or re-running. Keep it as the lossy class. The window-overhead result in the profile is not a reason to re-open it.

---

## Item 20. Mirror-SD / PipeSpec branch rollouts during verify

### Mechanism (as claimed)
- Mirror-SD launches branch-complete rollouts from early-exit signals alongside the target's suffix, removing the serial draft-then-verify dependency. It reports 2.8x-5.8x wall-time on SpecBench and 30% over EAGLE3 [paper, arXiv:2510.13161, abstract only]. The abstract does not state losslessness, only "without weakening acceptance semantics" [paper, same abstract].
- PipeSpec runs k models in a hierarchical pipeline with an optimistic downstream assumption and rollback, 2.25 vs 1.5 tokens per unit time [paper, arXiv:2505.01572, search excerpt].
- AMUSD decouples draft and verify into separate processes with a shared buffer and rollback flag, 29% average [paper, arXiv:2410.17375, search excerpt].

### Mechanism on Pancake (derived)
- The verify window is GPU-bound at 76% kernel time [profile]. A rollout during verify adds GPU work to a stream already busy. The CPU pool is also busy. So the overlap available on this card is limited to the 1-2 ms gaps and the spin periods of the wait kernels (293 ms) [profile, derived].
- A second draft model (PipeSpec, AMUSD) needs memory for another model and KV cache. The card has 20 GB and the target is large [MEMORY hardware note]. Not checked.
- Rollback on reject: the branch KV cache must be discarded exactly as the verify window discards rejected rows. The commit path (`verify.cpp:3498`) is the only place state advances, so branch state must stay out of it.

### Change
Design only. Nothing to build.

### Cost and risk
- Cost: high. A second model or a branch KV cache, a new scheduler, and a rollback path.
- Risk: high. Exactness requires that the branch state never reaches the committed KV. The papers' losslessness is not established in the abstracts [paper, unverified].
- Hardware: none of the papers measure a single RDNA card. The claimed gains are not transferable without a measurement [derived].

### Test arm
None. A design study would need a published exactness argument for greedy verify and a memory budget on this card.

### Decision
Design study only. Not for the current phase. The literature supports restructuring the draft-verify dependency, but the research notes already say no single-switch gain exists on this card.

---

## Open items and what each needs

1. Item 1 (environment record) comes first for 11, 12, 13, 14, and 16: the switch states of the profiled run are not recorded.
2. Confirm the 288 per-window `wait_flag_ge` site: `verify.cpp:2892` is the most likely non-fused CPU-share wait [audit, unverified]. The dispatch note at `verify.cpp:2931` names the reason.
3. Check the "acted" line for ROUTER_FUSE in a no-deferral run (item 14).
4. Check whether `hipDeviceAttributeCanUseStreamWaitValue` is 1 on gfx1100 (item 16). This needs one runtime query, which is a GPU run and is queued after worker A.
5. Resolve the VERIFY_DEFER_CPU preset conflict (`notes/untested_switches.md` vs the fast-hip config).
6. Fix the PIPED_MTP description in `conversation_disk_env.def:591`.
7. Record the `--pool-affinity measure` result for non-hybrid CPUs in the profiled log (item 11).

## Source list

- Code: `src/kernels/cpu/pool.cpp`, `src/kernels/cpu/pool_affinity_linux.hpp`, `src/program/generate.cpp`, `src/core/verify.cpp`, `src/core/mtp.cpp`, `src/kernels/cuda/verify_kernels.cu`, `src/kernels/cuda/native_moe.cu`, `src/kernels/cuda/native_bf16.cu`, `src/core/conversation_disk_env.def`, `include/strata/core/switch_report.hpp`, `configs/pancake/flashnext-iq2_xs-fast-hip.json`.
- HIP header: `/opt/rocm/core-10.0/include/hip/hip_runtime_api.h` (lines 585, 1032-1037, 3317-3430).
- Profile: `pancake-docs/research/bench/profile/summary.txt`; notes in `pancake-docs/research/window-overhead/notes/`.
- Papers: arXiv:2510.13161, arXiv:2505.01572, arXiv:2410.17375, arXiv:2607.20475 (abstracts or search excerpts only, as the literature notes state).
