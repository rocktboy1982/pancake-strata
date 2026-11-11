# Expert deferral only in lookup windows (`STRATA_VERIFY_DEFER_AUTO`)

MOD_PLAN Phase 17, item 17.1 note. Branch `dev/p17/defergate` (from `origin/dev/phase16-academic`).

**Status: coded, NOT built, NOT measured.** Benchmarks held the GPU and the CPU while this was written, so nothing was
compiled or run. The C++ was checked by reading it and by an ast-grep (tree-sitter) parse: the edited files have the same
parse-error nodes as before the change, at shifted lines (macro-heavy spots the parser cannot read), and no new ones.
Types and linking are not checked. The unit test is written and registered in CMake; it has not run. Every statement below
about what the engine does is **unconfirmed (read from code)** unless it cites a measurement.

The gate is our own idea: we found no fork or paper that switches expert deferral per verify window by draft source.
The deferral itself comes from Strata PRs #2689/#2690 by lask3802 and the rank guard from KTransformers; both are credited
already. No credits rows were added.

## 1. What the deferral does today

`STRATA_VERIFY_DEFER_CPU=1` (lossy). In a decode verify window, layer l combines its experts without the light CPU rows
and does not wait for them. Those rows join the residual in layer l+1, just before its FFN-side read. The CPU computes
layer l while the GPU runs layer l+1's mixer. The stage's last layer is never deferred.

| piece | where | when it is decided |
| --- | --- | --- |
| switch, `_BELOW` (heavy weight, default 0.1) | `src/core/verify.cpp`, `Verifier::init` | once per process |
| heavy split off (every CPU row late) | `init`: a helper GPU (`pool_gpu_helper`), `STRATA_VERIFY_DEVICE_PLAN=1`, or `_BELOW` >= 1 | once |
| rank guard `_TOP` | `include/strata/core/defer_rank.hpp`, `defer_top()` | once (a static) |
| which windows defer | `Verifier::defer_ok(T)`: not a batch window, not `--spec-split`, not `--remote-expert-opt`, not an all-resident stage, not under a control vector | **at graph capture, per window size T** |
| the deferred graph | `record_window`: post(l) waits for the partial flag, copies the heavy rows (`copy_rows_by_weight`), combines; pre(l+1) runs `late_add` (wait for flag l, copy the light rows, `defer_add_rows`) | baked into the captured graph |
| what the deferral turns off | `record_window`: `cfuse_off` (`STRATA_COMBINE_CPU`, so also `STRATA_WAIT_FUSE`, `STRATA_IC_WARM_MB` and `STRATA_CPU_PRESUM`, which ride in the fused combine), `rpub` (`STRATA_ROUTE_PUBLISH`, so also `STRATA_ROUTER_FUSE`) | tested on the process-wide `defer_` flag, not on the window: off in **every** window graph, deferred or not |
| host side | `Verifier::run`: `heavy_layer(T, l)` sets the plan sink's weights and partial hook per layer; `expert_source.cpp` runs the heavy CPU jobs first, raises the partial flag, then the light ones | per window and layer, at run time |
| rank guard word (A44) | `h_top_` / `m_top_`, staged by `stage_inputs`, read by `copy_rows_by_weight_kernel` (`top_live`) and by the pool (`sink_.top`) | **per window, at run time**; today only with `STRATA_BANDIT=top` |

The `ROUTE_PUBLISH` exclusion came from the 2026-11-09 bisect of non-finite logits (d5b58f669). The `COMBINE_CPU`
exclusion came from non-finite logits on the 7900 XT (a deferred layer's rows are not in the mapped buffer when the fused
combine reads them).

## 2. Can the deferral change per verify window?

**Partly.**

- **Which CPU rows join late: yes, at run time, with no new graph.** The deferral's graphs already read one mapped word
  per window, the A44 rank guard. If a window stages it at k (the model's experts per token, 10 here), every CPU entry is
  among its token's top k, so every CPU row is "heavy" (`defer_heavy_entry`). Then the pool runs all its CPU jobs in the
  first pass and raises the partial flag after them, post(l) waits for all rows and combines them, and `late_add` copies
  only zero rows and adds zero. The numbers of such a window are the exact model's: the same rows in the same combine
  kernels as the non-fused exact path (the fused `COMBINE_CPU` combine is documented as bitwise the launches it
  replaces). The one difference read from the code: adding a zero product turns an exact -0.0 in the residual into +0.0.
- **The graph shape: no.** Deferred or not is fixed at capture, one graph per window size (`exec_[sel_long][T]`,
  `exec_nr_[...]`). The fused paths are off for the whole process. So a window that keeps every row on time still runs
  the deferral's graph. It loses `COMBINE_CPU`, `WAIT_FUSE` and `ROUTE_PUBLISH`, and it pays `late_add` (one wait and
  two small kernels per deferred layer) for nothing. In the 2026-11-09 profile of a deferred decode
  (`bench/results/2026-11-09-gpuprof-defer/summary.txt` on `port/rdna3-decode`), `copy_rows_by_weight_kernel` was 2.1%
  of kernel time and the gaps after `wait_flag_ge -> copy_rows_by_weight` were 119 ms of 5741 ms wall.

So the gate below can make a window exact, but it cannot give that window the speed of the preset without deferral.
Getting that speed back needs a second set of window graphs (section 5).

Which explanation of the prose loss is right decides whether the gate can win. Two readings, neither measured:

1. **The process-wide switch losses cost it** (fused combine, wait fuse, one-launch publish, the late-add launches).
   Small MTP windows have little CPU work to hide, so these fixed costs per layer weigh most there. Then the gate cannot
   win prose back: its exact windows keep those costs and also lose the overlap.
2. **The late rows cost it per window** (the light-row pass, the split's second pool pass, output drift). Then the gate
   wins prose back. Against this reading: in the 2026-11-09 test phase, draft acceptance did not drop with the deferral
   (short 0.70 vs 0.74, edit 0.95 vs 0.95). Also, `_TOP=2`, which keeps more rows on time in the same graphs, was the
   slowest arm on short (-6.9%).

The `auto-closed` arm of the measurement plan (section 7) tells the two apart.

## 3. The gate

    STRATA_VERIFY_DEFER_CPU=1 STRATA_VERIFY_DEFER_AUTO=1     # deferral only in lookup windows
    STRATA_VERIFY_DEFER_AUTO_MIN_T=5                          # optional: only lookup windows of 5 rows or more

Rule (`include/strata/core/defer_auto.hpp`, pure functions; unit test `tests/core/defer_auto_test.cpp`):

| window kind | `defer_window_kind(merged, from_sfx, chain_n, T)` | AUTO off | AUTO on |
| --- | --- | --- | --- |
| merged first window (prompt tokens) | Prompt | as configured | every row on time |
| one row | Plain | as configured | every row on time |
| MTP / DFlash2 drafts | Model | as configured | every row on time |
| MTP drafts + chained lookup tail (`--lookup-chain`) | Chain | as configured | **defers** if T >= max(2, MIN_T) |
| suffix / prompt-lookup copy (the draft policy picked a lookup window) | Lookup | as configured | **defers** if T >= max(2, MIN_T) |

- `defer_window_on(cfg, kind, T)`: the verdict. With AUTO off it is always true, and the loop does not call it.
- `defer_staged_top(auto_on, window_defers, defer_top_now(), k)`: the rank guard `stage_inputs` writes to the mapped word
  and to `sink_.top`. It is `defer_top_now()` unchanged unless AUTO is on and the window does not defer; then it is k.
- The serial serve decode loop (`generate.cpp`) marks each window with `Verifier::set_defer_window(bool)` just before
  `ver.run`. It is a one-run borrow like `set_ecap_window`: `run` clears it on every exit, for this stage and later stages.
  A window nobody marks keeps every row on time. That covers prompt reads through verify windows, `--pipeline-windows`,
  batch windows (they never defer anyway) and `strata generate`.
- Chained windows count as lookup windows, as in the A44 bandit's window type (`from_sfx || chain_n > 0`); they are the
  widest windows. `--lookup-chain` is off in fast-hip.
- With `STRATA_BANDIT=top` the bandit still sets the rank guard inside the windows that defer. The closed windows stage k.
- It needs the heavy split: AUTO is ignored with a log line without `STRATA_VERIFY_DEFER_CPU`, with `_BELOW` >= 1, with a
  helper GPU, or with `STRATA_VERIFY_DEVICE_PLAN=1`. In those cases the deferral acts in every window it can, as before.
- The rank guard's mapped word (64 bytes of pinned memory per stage) is now also kept when AUTO is on.
- Start-up line: "STRATA_VERIFY_DEFER_AUTO=1 (MOD_PLAN 17.1 note, changes the output only through the deferral): ...".
  Per request: "STRATA_VERIFY_DEFER_AUTO: X of the decode loop's Y verify windows deferred their light CPU rows".
- Conversation disk key: both new variables are `KeyIfSet` (`src/core/conversation_disk_env.def`), so the keys of runs
  without them do not change.

**Default path.** With `STRATA_VERIFY_DEFER_AUTO` unset: the mapped-word test in `init` reduces term for term to the A44
test; the bandit start-up line and `top_word` keep their old condition (the word exists then only for the bandit);
`stage_inputs` stages `defer_top_now()` as before; the run guard only stores false; the serve loop's new block and its
log line do not run; the disk key is unchanged. This was checked by reading the code, not by a binary comparison.

## 4. Numbers the idea rests on

Measured 2026-11-11 by the lead on the RX 7900 XT: same neutral client, Qwen3.8-Flash-Next Q2_0, warm second pass, 10
decode samples per row. The raw files are in `bench/results/2026-11-11-vs-avalon/raw/` and the report is `REPORT.md`, both
on `port/rdna3-decode`. The runs without the variable are not in that directory; their numbers come from the lead's brief.

| run | decode | prose | code |
| --- | ---: | ---: | ---: |
| pancake fast-hip, deferral on | 102.7 | 98.5 | 132.9 |
| same, variable removed, run A | 111.0 | 106.2 | 129.4 |
| same, variable removed, run B | 110.3 | 107.7 | 131.2 |
| Strata-rdna3 (no deferral) | 110.2 | 102.2 | 134.2 |

So the deferral costs prose about 8% and gains code at most 1-3%. Earlier, the 2026-11-09 test phase
(`bench/results/2026-11-09-test-phase/01-deferral.summary.txt`, n=3) measured edit +15.8% +- 3.9% and short -1.9% +- 4.8%
(no difference). Another box gave short +8% and edit -7% (lead's note). The deferral changes the output. Run-to-run spread
on prose was 12-17 tok/s, so these are few runs. They motivate the experiment; they do not show that the gate wins. An
IQ2_XS confirmation was running when this was written.

## 5. What a full per-window switch would need (not done)

Two graph sets, one deferred and one exact, chosen per window on the host:

1. Key the window graphs by a defer bit as well: `exec_[2][2][T]` and `exec_nr_` likewise. Then update `capture`,
   `prepare`, `evict_window_graph`, `capture_all` / `pl_launch` (pipelined) and the layer split's `prelaunch` (the next
   stage must take the same bit, as it takes `sel_long_`).
2. In `record_window`, test the window's own `dfr` instead of the process-wide `defer_` in `cfuse_off`, `rpub`, the
   `ROUTER_FUSE` and `IC_WARM` reasons. Then the exact graphs get `COMBINE_CPU`, `WAIT_FUSE` and `ROUTE_PUBLISH` back.
   Key `presum_[T][grp]` (item 51) by the bit too.
3. Host side: `heavy_layer(T, l)` and the plan sink's weights must follow the window's bit. An exact window must never run
   the heavy split, because `ROUTE_PUBLISH`'s `route_publish_res` publishes no weights and the split reads them (the
   2026-11-09 non-finite bisect). The deferral probes (512+512 tokens, the non-finite check; `defer_probe.sh` in
   RESEARCH_AND_FINDINGS, not in the repo) must be rerun for both graph kinds.
4. Cost: up to twice the window graphs (sizes 1..12 at `--lookup-max-t 12`, times the long-select and doorbell variants).
   That means more capture time on first use and more graph memory; `instantiate_evicting` already evicts under pressure.
   Unmeasured.

This touches graph capture and the router publish in the hot path. It is worth doing only if `auto-closed` shows that
the deferral's graphs cost prose by themselves.

## 6. Risks

- **It changes the output** in the windows that defer (lossy, as the deferral is). The closed windows are exact by
  reading, but the engine is not run-to-run deterministic anyway (0 of 6 outputs identical exact vs exact, MOD_PLAN
  Phase 17). So text identity cannot judge it; token forcing and the task gate can.
- **The cache and the adaptive tier learn from deferred windows.** A late row changes the residual of later layers, so
  their routing, so which experts load and stay. The exact windows then run on a cache shaped partly by deferred ones.
- **The draft policy learns window costs by size.** It prices windows per size (`DraftPolicy::cost_ms(t)`), and a lookup
  window of 2-4 rows shares a size with MTP windows. With AUTO and no MIN_T, a size's cost mixes deferred and exact
  windows. `MIN_T=5` avoids that in fast-hip (`--spec 4`): only lookup windows reach 5+ rows when `--lookup-chain` is off.
- **First hardware use of the mapped rank-guard word.** The A44 path (`top_live` in `copy_rows_by_weight_kernel`) is
  compiled and unit-tested, but `bench/results` holds no GPU run of `STRATA_BANDIT=top`.
- **Prompt reads through verify windows no longer defer** under AUTO, unlike plain deferral. That is exact and safer,
  but it may be slower on prompts that are read through windows. The batched prefill path is unaffected.
- **Not checked**: a layer split (each stage stages its own word; `set_defer_window` reaches every stage), DFlash2
  drafting, `STRATA_BANDIT=top` together with AUTO, Metal and SYCL (the SYCL port has its own `verify.cpp` and
  `generate.cpp` and does not get this switch).

## 7. Measurement plan

Preconditions: build HIP (and CUDA, as the files are shared), then
`ctest -R 'defer_auto_test|defer_rank_test|conversation_disk_env_test|conversation_disk_tier_test'`. Then run one short
serve with `STRATA_VERIFY_DEFER_AUTO=1` on the adopted Q2_0 preset. Check the start-up line, and check that the
per-request line shows most windows deferred on an edit and few on a prose reply. Also run one deferral probe
(512+512 tokens, the engine's non-finite check) with AUTO on.

Speed: spec `bench/specs/p17-defer-auto.json` (base = `pancake-flashnext-q2_0-fast-hip-adopted.json`, write it with
`tools/pancake/gen_bases.sh`), paired block harness:

    python3 tools/pancake/abtest.py bench/specs/p17-defer-auto.json --block 3 --warmup 1 --outlier robust --reclaim-max 3

| arm | env | what it measures |
| --- | --- | --- |
| `base` | fast-hip (deferral everywhere) | today |
| `nodefer` | `STRATA_VERIFY_DEFER_CPU` removed | the 11-11 comparison, with the fused paths back |
| `auto` | `_AUTO=1` | deferral in lookup windows of 2+ rows |
| `auto-min5` | `_AUTO=1 _MIN_T=5` | a pure size split at `--spec 4` |
| `auto-closed` | `_AUTO=1 _MIN_T=256` | never defers: the deferral's graphs with every row on time |

Cases `edit` (copy-heavy) and `short` (mostly prose). abtest has no pure-prose case, so repeat the prose check with the
2026-11-11 client (`cmp_bench.py`: story, 15 tips, MoE explanation; second pass kept) on `base`, `nodefer`, `auto-min5`
and `auto-closed`.

Reading it. The bar is >= 2% end to end per content class (Phase 17 rule); `promote.py` asks for a 95% lower bound of 3%
for a lossy item. Keep AUTO only if `edit` stays within noise of `base` and `short` / prose gain >= 2% over `base`.
Compare `auto-closed` with `nodefer` and with `base`:

- `auto-closed` close to `nodefer`: the late rows cost the time (reading 2); the gate can work.
- `auto-closed` close to `base` or slower: the deferral's graphs cost the time (reading 1); the gate cannot win prose.
  The work is section 5, or a per-session preset pair (deferral on for coding sessions, off otherwise).

Quality, only for an arm that passes on speed: `tools/pancake/quality_gate.py` (token forcing, decode path) and
`tools/pancake/task_gate.py --reps 5` (includes code edits and tool calls), against `base` and `nodefer`. A lossy arm
needs both PASS before it can go into a preset.
