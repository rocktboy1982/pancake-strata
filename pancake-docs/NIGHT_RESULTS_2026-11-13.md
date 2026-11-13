# Results and recommendations, 2026-11-13

Scope: the overnight chain (2026-11-12 23:00 to 2026-11-13 08:00), the daytime plan items that followed, the test suites, and the analysis reports. Raw output is archived under `pancake-docs/research/bench/results-2026-11-13/` (p18, fk-env, plan-items, logs, analysis, run-scripts). The plan entries are in `MOD_PLAN.md` under the 2026-11-13 headings. Rule used for every screen and confirm verdict: promote when a case is above 0% with no case below -3%; otherwise drop. A result is "exact" only when the generated text is identical to base.

## Test suites

| suite | result | notes |
| --- | --- | --- |
| CPU (`ctest -L cpu`, 150 tests) | 150 of 150 pass | one failure fixed: `STRATA_WINDOW_TRACE_TARGET` added to the environment registry (`2419711f5`) |
| GPU (`ctest -L gpu`, 134 tests) | 133 of 134 pass | `rdna_mmvq_parity` fixed (explicit stream, `1916b6fd9`); `conversation_copy_streams_test` passes after rebuild (bound raised to 256 MiB, `83a0a1fd5`) |
| GPU, still failing | `hip_flag_order_test` | consistent failure; static analysis points at the test's consumer (no system-scope fence after the flag spin); fix not applied or confirmed on the card; see `analysis/flag-order-report.md` |
| Offline Python | 62 pass, 1 skip | park, ecospec, gate-importance, record-session, task-gate |
| Tool-boundary marks | 12 of 12 pass | A11 |
| Read-before-edit hook (A41) | 8 of 8 pass | hook not yet registered in settings |
| abtest placeholder rendering | pass | regression test added with the fix |

## Speed results (screen = 2 repeats, confirm = 6 repeats; decode tok/s, short prompts unless noted)

| item | spec | arm | screen | confirm | text | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 27 | combine-poll1 | +1.1% | -4.2% | differs | drop |
| 3 | 27 | wait-fuse-off | +1.2% | +4.7% (+-9.1) | exact | candidate, not default |
| 4 | 27 | fuse-wait | +0.9% | +3.3% (+-8.6) | exact | candidate, not default |
| 6 | 17 | route-fast | +1.8% | -2.0% | differs at confirm (identical at screen) | drop |
| 6 | 17 | doorbell-store | +1.6% | -1.6% | differs at confirm | drop |
| 14 | a33 | nodefer+router-fuse | +0.1% | -2.9% (+-6.5) | differs | lossy; quality gate first |
| 14 | a33 | defer-preset | +0.2% | -5.1% | differs | drop |
| FK spec 28 | 28 | spec5 | +1.2% | not run | differs | lossy; quality gate first |
| FK spec 28 | 28 | mtp-eager | -3.6% | not run | identical | drop |
| FK spec 28 | 28 | spec3 | -3.2% | not run | differs | drop |
| FK spec 29 | 29 | device-plan | -5.9% | not run | differs | drop |
| FK spec 31 | 31 | aux-cpus-auto | +2.9% | -3.6% | exact | drop at confirm |
| FK spec 31 | 31 | kv-grow | +2.2% | -4.3% | exact | drop at confirm |
| FK spec 31 | 31 | gdn-pp2, spec-min-p-07, lookup16, sh-stream | mixed | not run | mixed | drop |
| FK spec 32 | 32 | wmma-gemm | +0.2% | not run | differs | lossy; quality gate first |
| FK spec 32 | 32 | prefill-auto-max-24576 | -0.4% | not run | identical | drop |
| A44 | 24 | five bandit and deferral arms | -23.7% to +2.3% | not run | all differ | drop |

Caveats that apply to every row:
- Screen results have two repeats, and most intervals include zero. Screen promotions are a reason to confirm, not a finding.
- The GPU clock was pinned to `high` on 2026-11-13 (after the reboot it had reset to `auto`). Base speed still moved between runs (about 103 to 130 tok/s on short prompts), so only within-run comparisons count.
- Exactness is checked against base in the same run. Item 6 shows that a screen "identical" can become "differs" at confirm, so the exactness check needs repeating on those arms.

## Long-context decode (A49, confirm, two repeats per row)

Short 111.6 tok/s; 10k 97.3; 32k 104.2; 64k 93.2; 128k 95.6. The fixed step from short to 10k is about 13%, and there is no clear slope after it. The first run's prompts were 8 to 11% short because the length constant was wrong (2.85 chars per token, measured 3.2); corrected in `tools/pancake/abtest.py`.

## Other analyses

- **A24 suffix index** (replay of the real opencode store): +2.0% accepted tokens and -2.4% verify rounds overall; sub-agents +6.0% accepted, -7.5% rounds. Needs the GPU screen before any decision.
- **A26 draft-source selector:** no per-request selector beats always-lookup. The per-window longest-match rule is the only candidate above noise (+2.0%), and it is the same figure as A24.
- **A9 pre-stage during tool calls:** bound of about 2 to 4% of wall time; the coded path is `STRATA_IDLE_ADAPT`. Needs the GPU replay.
- **A6 time model:** mean held-out error 3.4 points (inside the 5 to 9 target); the rank-guard arm misses by 8 points and the fitted parameters are not identified by the data.
- **A40 tool share:** 4.7 hours of tool time against an 86.6-hour session span, so 5.4% is an upper bound, not an overlap measurement.
- **A48 2-bit kernel:** 1.23x on one thread, 1.03x at 8 threads; memory-bound, about a 3% ceiling at decode thread count.
- **A53 IOMMU start-up note:** done; silent on this machine's passthrough setup.
- **A21 temperature logging:** the turn log now records the request temperature.
- **A18 effort per task:** logging is on through the daily preset; the new fields appear on the next server start.

## Recommendations

1. **Keep as candidates, not defaults:** `wait-fuse-off` and `fuse-wait` (Phase 18 items 3 and 4). Their output is exact, and their confirm results are positive, but the intervals include zero. Run a longer confirm (9 or more repeats, both prompt types) before any default change.
2. **Quality gates before any further confirm:** `nodefer+router-fuse` (item 14), `wmma-gemm` (spec 32) and `spec5` (spec 28). Each changes the output, so none can be kept on speed alone.
3. **Repeat the exactness check** on the item 6 arms (`route-fast`, `doorbell-store`) to settle whether they are exact. The screen and confirm runs disagree.
4. **Drop from the plan:** `combine-poll1`, the item 6 arms (on speed, pending the exactness check), `defer-preset`, `aux-cpus-auto`, `kv-grow`, `gdn-pp2`, `spec-min-p-07`, `lookup16`, `sh-stream`, `mtp-eager`, `spec3`, `device-plan`, `prefill-auto-max-24576`, and the A44 bandit family.
5. **A51 (sparse decode attention): drop** unless a profile shows attention at 15% or more of decode time. A49 does not show that.
6. **Fix the flag-order test first, then rerun it on the card.** Apply the diff in `analysis/flag-order-report.md` (system-scope fence after the flag spin, in the test consumer), set `LD_LIBRARY_PATH` to the ROCm 10.0 libraries for direct runs, and confirm on each device with `HIP_VISIBLE_DEVICES`.
7. **Make the A41 read-before-edit hook a warn-only trial** before hold mode. It is tested (8 of 8) but not registered; registration needs your approval because it changes Claude Code settings.
8. **Operations before the next long run:**
   - run chkdsk on the D drive from Windows (it was dirty after two reboots; it is mounted with `force` now);
   - pin the GPU clock after every reboot (`power_dpm_force_performance_level` = `high`);
   - rerun the rank guard once memory is compacted (it aborted on fragmentation, and the placeholder bug is fixed);
   - decide on the sudo rules (option 2, for the compaction, clock and mount commands), added by you with `visudo`.
9. **Not yet run:** the A56 trace identity run (flag on versus off, greedy tokens must match), the A55 live task gate, item 9 (needs `build-hip-orig` for the build-versus-build merge), A8, A14, A30, A31, A33, A34, A35, A36, A45, and the FK code ports (FK-22, 23, 24, 26, 28 and the rest in the plan). Each needs its own spec or code work.
10. **Decisions still open for the owner:** A18 (effort-per-task logging check on the next run), A28 (the 58 GB coder model: kept on D, not tested), A29 (research step before any code), the 9router dashboard password, and whether to keep the `claude --resume` session running alongside the GPU work.
