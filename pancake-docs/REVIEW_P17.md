# Phase 17 code review (2026-11-11)

Scope: the five opt-in engine and server features merged on `dev/p17/integration` (17b11ef85), reviewed on branch
`dev/p17/review`. The diff is `git diff origin/dev/phase16-academic..HEAD -- '*.cpp' '*.hpp' '*.cu' '*.h' '*.def'
CMakeLists.txt serve/*.py` (16 files, +1,368 lines).

Method: reading the code only. Nothing was built, no engine or server was started, no GPU was used. Run on the CPU:
`serve/test_park_divergence.py` (21 tests, pass), the two regression tests added here, and a pure-Python copy of the
source scan of `conversation_disk_env_test`. Every finding below is unconfirmed static analysis in the sense of
CONTRIBUTING.md. CONFIRMED means "seen on the exact lines", not "reproduced on hardware". PLAUSIBLE means the failure
depends on run-time behaviour that was not checked.

## Verdict

| Feature | Unset = default path unchanged? | Main problem when set |
| --- | --- | --- |
| `STRATA_KV_RING_ADAPT` | Yes. The block is skipped, `o.kv_resident` keeps its value. Only the `--help` text changes | Turns KV streaming on where the cap held the whole context (R3). Under `--serve` the ring depends only on `--max-context`, so the planned night test compares identical servers (R4) |
| `STRATA_ROUTE_PRIOR_AUTO` | Yes. `prior_gate_open` returns `scope_open`, and the gate is allocated under the same condition | Not in the disk-key table: the ctest fails, and a disk image saved with AUTO=1 is reused with AUTO unset (R1) |
| `STRATA_VERIFY_DEFER_AUTO` | Yes. The rank-guard word is allocated under the A44 condition, term for term, and `defer_staged_top` returns `defer_top_now()` | No bug found in the gate. `defer_auto_test` is not labelled `cpu` (R5) |
| `STRATA_PARK_REWIND_MAX` | Yes. `park_rewind_max()` is 0, the lambda returns false at once, and the park condition is the old one | Also skips the park for another conversation that shares a checkpointed prefix. This reverses MOD_PLAN 90's 2048-token rule (R2) |
| `STRATA_PARK_DIVERGENCE_LOG` | Yes. The tracker is `None`, every hook is skipped, and the import has no side effects | A slow memory leak after the tracker turns itself off (R7). Spurious lines after an engine restart (R8) |

Two problems must be fixed before any night run: R1, because `ctest -L cpu` fails on the integration branch, and R4,
because night-queue item 4 as written measures nothing.

## Findings, ranked by severity

| Id | Sev. | File:line | Confidence | Scenario (inputs/state -> wrong result) | Suggested fix |
| --- | --- | --- | --- | --- | --- |
| R1 | High | `src/core/conversation_disk_env.def` (missing lines, next to 156-157); names read at `src/program/generate.cpp:393`, `:2218`, `:4687` | CONFIRMED (scan run: 4 names missing, 0 stale) | `ctest -L cpu`: `conversation_disk_env_test` fails with "ADD A LINE for: STRATA_KV_RING_ADAPT, STRATA_PARK_REWIND_MAX, STRATA_ROUTE_PRIOR_AUTO, STRATA_ROUTE_PRIOR_AUTO_MIN_T". At run time the disk-tier key is built from this table, so `STRATA_ROUTE_PRIOR_AUTO` is not in it. Example: run 1 with `STRATA_ROUTE_PRIOR=0.5 STRATA_ROUTE_PRIOR_AUTO=1` and `--conversation-disk-dir D` evicts a conversation to D. Run 2 with `STRATA_ROUTE_PRIOR=0.5` and no AUTO gets the same key and restores K/V that was computed with other routing. No warning; the run is lossy either way | Add `STRATA_ROUTE_PRIOR_AUTO` and `_AUTO_MIN_T` as `KeyIfSet` (like `STRATA_ROUTE_PRIOR_SCOPE`). Add `STRATA_KV_RING_ADAPT` as `InConfig`: the ring it picks is `c.kv_resident` in `session_identity`, `generate.cpp:12663`. Add `STRATA_PARK_REWIND_MAX` as `Speed`: it is a cache policy, like `STRATA_PARK_ASYNC`. `tools/pancake/test_disk_env_table.py` checks this without a build |
| R2 | Medium | `src/program/generate.cpp:17629-17649`, `include/strata/program/conv_cache.hpp:178-204` | CONFIRMED | `own_checkpoint` is true for every entry of `checks`: after a normal request, all of them are prefixes of `live`. So the only test left is "at most N tokens behind". With `STRATA_PARK_REWIND_MAX=8192` (night-queue value): the main agent's live conversation is a 12,000-token system prompt and tools (a pinned or periodic checkpoint at 12,000) plus 5,000 tokens of its own. A subagent request with the same 12,000-token prefix resumes at 12,000, and 5,000 <= 8,192, so the main conversation is not parked. Its next turn re-reads at least 5,000 tokens (about 2 s at 2.4-2.8K tok/s, from the 11-11 logs in TTFT_LEVERS) instead of a 0.1-0.2 s restore. Alternating agents re-read on every switch. This reverses MOD_PLAN 90 (`generate.cpp:17585-17598`): a branch longer than `STRATA_PIN_PARK_TOKENS` (2048) past the pin "is a conversation of its own" and parks. The output is unchanged (speed only), but the comments at `conv_cache.hpp:197` ("Another conversation's request ... not touched") and `generate.cpp:17626`, and TTFT_LEVERS.md's "loses only the generated tail", are wrong | Identify the motivating case directly. Add `diverge_at` (the common prefix of `ids` and `live`) and `last_prompt_len` (the previous request's prompt length) to `RewindFacts`. Skip only when `diverge_at >= last_prompt_len - slack`, that is, when the prompt parts from `live` inside the last reply. Never skip past `pin_park_tokens` when the checkpoint is a pin |
| R3 | Medium | `include/strata/core/kv_ring.hpp:34-37`, `src/program/generate.cpp:4688`; streaming rule `src/core/layer.cpp:750` | CONFIRMED (the speed and answer effect is from an earlier measurement, not re-run) | `kv_plan` streams only when the ring is below the session's cells. Example: `STRATA_KV_RING_ADAPT=1 --serve --kv-resident 65536 --max-context 49152` (or 49152/49152). Before: the whole K/V in VRAM, no streaming. After: the depth 49152 is below 2 x 65536, so the ring is 32768 and streaming turns on, with the rest of the K/V in pinned RAM. `strata_control/tune_knobs.py:170` records that a `--kv-resident` below the context made answers 22-34% slower and changed them | Keep the cap when it holds the whole session: pass the session's cells (serve: `--max-context`; generate: `--max-context` too) and return `cap` when `max_cells <= cap`. Add a test case |
| R4 | Medium | `src/program/generate.cpp:4688`; `MOD_PLAN.md:2649` (night item 4); `configs/pancake/flashnext-iq2_xs-fast-hip.json` (`--max-context 131072`, `--kv-resident 32768`) | CONFIRMED (code plus config) | Under `--serve` the depth is `--max-context`, so the ring is fixed for the life of the process and does not follow the request. With the preset's 131072 and `--kv-resident 49152`, 131072 >= 2 x 49152, so the ring is 49152 with or without the switch: "on vs off at 1K / 32K / 64K / 128K" runs the same engine twice. With the preset's own 32768 the switch does nothing (cap <= floor). Under `--serve`, `STRATA_KV_RING_ADAPT=1` only ever means "`--kv-resident` floor instead of cap when `--max-context` < 2 x cap" | Measure with `strata generate` (depth = prompt + `--max-new`), or restart the server with `--max-context` set to each depth. Say in `--help` and in the 17.4 note that serve does not adapt per request |
| R5 | Low | `CMakeLists.txt:3563` | CONFIRMED | `defer_auto_test` is missing from `_strata_tests_cpu`, so it gets the `gpu` label, and `ctest -L cpu` (`build_matrix.sh --tests`) never runs it. `route_prior_auto_test` and `kv_ring_test` were added to the list; this one was lost when the defergate branch merged separately | Add `defer_auto_test` after `defer_rank_test` |
| R6 | Low | `src/program/generate.cpp:17650-17654` | CONFIRMED | The line "not parked (STRATA_PARK_REWIND_MAX)" is printed even when no park would have happened: `pin_sibling`, `!park_worth`, a disabled conversation cache, or an empty live state. `ttft_audit.py` and M2 counts would credit the switch with skips it did not cause | Print it only when the else-if branch would have parked: `(!from_live \|\| incoming \|\| slot_source >= 0) && !pin_sibling && park_worth && conversations.enabled()` |
| R7 | Low | `serve/park_divergence.py:181-183` (`token`); `:149-158`, `:185-187` | CONFIRMED (regression test added, marked `expectedFailure`) | After an exception in `resumed()` sets `broken`, `begin()` and `end()` return at once and never clear `_gen`. `token()` does not check `broken`, so every streamed token of every later request is appended to one list, about 36 bytes per token (Python int plus pointer), for the life of the server. This is opt-in and diagnostic only | `token()`: `if self.broken or self._gen is None: return`. Set `self._ids = self._gen = None` wherever `broken` is set |
| R8 | Low | `serve/server.py:3002` (`div.begin(id(self.proc), ids)`) | PLAUSIBLE | CPython reuses the id of a freed object. After an engine restart, the new `Popen` can get the dead one's id, so the first request after the restart is compared with the dead engine's conversation and a spurious divergence line is printed (with `"resume": 0`). Log noise only | Use a key that changes on every start: a start counter, or `(proc.pid, start time)` |
| R9 | Info | `src/program/generate.cpp:17638` | CONFIRMED | `moved_in >= 0` is always false at the decision point. `moved_in` is only assigned later, at `:17758` and `:17949`. Harmless | Drop the term, or say in the comment that it is only a guard |
| R10 | Info | `src/core/conversation_disk_env.def:156-157` | CONFIRMED | `KeyIfSet` keys any set value: `STRATA_VERIFY_DEFER_AUTO=0`, or `_MIN_T` without AUTO, changes the disk key although the engine behaves as if unset. It only causes a miss, and other `KeyIfSet` switches behave the same way | None needed. Measurement scripts should unset these switches, not set them to 0 |
| R11 | Info | `include/strata/program/route_prior_auto.hpp:376-381`, `include/strata/core/defer_auto.hpp:144-149` | CONFIRMED | The two gates classify chained windows differently: route-prior AUTO keeps `Chain` windows exact, while deferral AUTO lets them defer. Both are documented. With both switches on, a `--lookup-chain` window defers but gets no bonus | Keep in mind when reading the 17.3 ablation |

## Verified safe

STRATA_KV_RING_ADAPT
- `o.max_context`, `o.max_new` and `o.tokens` are final at the hook (`generate.cpp:4687`). The only writes are CLI parsing and serve's placeholder tokens at `:4645`, which come before the hook. The hook runs before `qsa_set_kv_resident`, the layer-split search (`chunk_bytes` reads `qsa_kv_resident()`), the session arena and graph capture.
- `split_walk::next_kv_window` starts from the ring and only walks down (20480 / 8192 / 4096). Starting from the ring instead of the cap skips no step, and a failed walk restores the ring (`:6951-6966`).
- The `--kv-grow` gate (`o.kv_resident <= 0`, `:7097`) is unaffected: the ring is > 0 exactly when the cap is > 0.
- The session config fingerprint (`c.kv_resident`, `:12663`), the INFO line and the run manifest all report the ring that is really allocated. The ring is fixed for a process, so no snapshot crosses rings in RAM. On disk the config fingerprint changes with the ring, so a different ring is a miss, not a stale hit.
- The pure rule: no overflow (`cells_needed / 2 >= cap`). A cap at or below the floor is returned unchanged. The floor honours `STRATA_KV_RESIDENT_MIN`.

STRATA_ROUTE_PRIOR_AUTO
- The serve decode loop has one `ver.run` call (`:20637`). The gate is written right before it for every window kind:
  - merged first window -> `Prompt` (closed); a one-row window -> `Plain` (closed);
  - the first window never carries a lookup draft (`prepare_gated_window(initial)` and the non-gated path's `!first_window`);
  - `dfr_lookup` and the cross-session index give `Lookup`; `--lookup-chain` windows give `Chain` (closed).
- Prompt-read windows write `on_prompt` (0 unless set, `:18212`). The pipelined prompt read starts from the closed gate.
- `--batch`, `--pipeline-windows` and `strata generate` get `loop_gates == false`, so the float stays 0 and the bonus is off. This fails safe, and the start-up line says so.
- `PriorGateRequest`'s destructor writes 0 when the request scope ends, error returns included.
- `route_prior_k` returns before writing `ids` and `weights` when the gate is <= 0, so the router's top-10 stands.
- The lambda resolution is the same as `g_route_prior()` in verify.cpp.
- Unset: `prior_gate_open` returns `scope_open`. Setting only `_AUTO=0` adds a log line and calls `route_prior_configure(nullptr, 0, 0)`, which is the same as not calling it.

STRATA_VERIFY_DEFER_AUTO
- `copy_rows_by_weight_kernel` reads `top_live` whenever the pointer is non-null, even when the captured `top` is 0, so the gate acts without `STRATA_VERIFY_DEFER_CPU_TOP`.
- With `top = k`, the rank of every entry is below k on both sides: the GPU kernel and `defer_heavy_entry`, which the pool calls through `ps->top`. So every CPU job runs in the first pass, `post(l)` copies every CPU row on time, and the late addition copies zeros and adds +0.0.
- `sink_.top` and the mapped word are written together in `stage_inputs`, behind a seq_cst fence, before the launch. That is the A44 mechanism, unchanged. No pool work for window N reads `sink_.top` after `run(N)` returns.
- `DeferBorrow` is built before the first return of `run()`, the Metal path included.
- Stage runs nest (`next_->run` is called inside `run`), and the prelaunch helper is joined by `NextLaunch`, which is destroyed first. So every stage stages its own `defer_next_` before the borrow clears it.
- Batch windows never defer (`defer_ok`: `batch_rec_`). `pl_stage`, prompt reads and `strata generate` never call `set_defer_window`, so they stage k and stay exact. `STRATA_VERIFY_DEVICE_PLAN` frees the word, which turns the gate off with a message.
- The epoch bandit is told about the word only when it asked for it (`top_word` condition unchanged).
- Unset: the allocation condition reduces to A44's, term for term. `DeferBorrow` adds a few stores per window and no change to any value.
- `KeyIfSet` for both names is right: an unset switch leaves the disk key as it was.
- Mixed stages are not a concern. The conditions behind `defer_auto().on` are process-wide (env, helper GPU), apart from the per-stage `device_plan_`, which applies equally to every non-resident stage. So checking only the head verifier at `:20629` does not leave a stage that could gate unmarked in practice.

STRATA_PARK_REWIND_MAX (state safety)
- A park never changes the session or the checkpoint chain, apart from `STRATA_PARK_ASYNC` moving checkpoints that are dropped anyway. So after a skipped park the engine is in exactly the state it would be in after a park.
- The checkpoint mount that follows (`checkpoint_restore` at `resume`, stage parts included, `:17905-17941`) is the same path a parked request and `pin_sibling` take. `off_path` drops every checkpoint past `resume`.
- The retained K/V is trimmed by `limit_reuse(read_from)` (`:18024`) whether or not the park ran, so a later park cannot reuse stale cells past `resume`.
- Excluded: borrow, restore, batch-slot sources, lent slot checkpoints, and live conversations with pictures. A layer split goes through the same checkpoint mount. Superseded parked copies of the conversation are dropped at its next park, as before.
- No state is ever resumed that was neither parked nor kept: the cells past `resume` belong to no checkpoint and no parked image after the skip.

STRATA_PARK_DIVERGENCE_LOG
- `begin()` and `resumed()` catch every exception. `end()`, `token()` and `forget()` only build lists of ints. The `finally` hook comes before the STOP and drain, but it cannot raise.
- `decode` failures are caught inside `record()`. Prompt text is printed only with the switch on, and only up to `STRATA_PARK_DIVERGENCE_CHARS`.
- `--parallel` (rid), pictures and `prefill_only` call `forget()`. `--batch` never reaches the hook.
- `id(self.proc)` separates engine runs except for the id-reuse case (R8).

Headers, linkage and CMake
- `#pragma once` in all three new headers. They contain only `inline` functions and namespace-scope `constexpr` values (internal linkage). `defer_auto.hpp`, included by `verify.hpp` and so by nvcc and hipcc, needs only `<cstdlib>` and `<cstring>`.
- Each new test is registered once. The `add_test` override puts its executable in `strata_tests`. `route_prior_auto_test` and `kv_ring_test` carry the `cpu` label.

## Could not judge without running

- Whether an exact window under DEFER_AUTO is bit-identical to a run without deferral. The window graphs stay the deferral's, the process-wide switches stay off, and the zero late-add turns -0.0 into +0.0.
- Whether streaming at the 32768 floor changes answers or speed on the 7900 XT today (R3 relies on the earlier tune_knobs record).
- The visibility timing of the mapped gate float and the rank-guard word: the same mechanism as F13 and A44, never exercised with a per-window change on HIP.
- Whether "exact routing" in a closed AUTO window equals a run without `STRATA_ROUTE_PRIOR`. Setting it turns off ROUTE_PUBLISH, ROUTE_FAST and ROUTER_FUSE for the whole process.
- With `STRATA_PARK_ASYNC`: an earlier asynchronous park still copying while a checkpoint mount overwrites cells. This is the same exposure as the existing park path.
- The real trade between re-reading N tokens and parking plus restoring at 8K-60K (R2).
- Compilation of any of the new C++ (no build was allowed).

## Unit tests

Added on this branch (Python, CPU, run here):
- `tools/pancake/test_disk_env_table.py`: the source scan of `conversation_disk_env_test` without a build, plus the Phase 17 names by name. It FAILS on 17b11ef85 (R1) and passes once the 4 lines are added.
- `serve/test_park_divergence_review.py`: a broken tracker must hold no streamed tokens (R7, `expectedFailure` until the fix; remove the marker with it), and a broken tracker never raises.

Would add, with the fixes (C++, header-only, `cpu` label):
- `kv_ring_test`: `max_cells <= cap` returns the cap (R3), and serve's depth = `--max-context` gives the cap at 131072 / 49152 (R4, to document the behaviour).
- `conv_cache_test`: a sibling conversation (divergence inside the history, at or past a shared pin, branch longer than 2048 tokens) parks; the reasoning-drop case (divergence inside the last reply) skips; `pin_park_tokens` is respected (R2).
- `conversation_disk_env_test` decisions: `STRATA_ROUTE_PRIOR_AUTO` and `_MIN_T` keyed when set, `STRATA_KV_RING_ADAPT` `InConfig`, `STRATA_PARK_REWIND_MAX` out of the key (R1).
- `defer_auto_test`: a three-stage fake chain showing that `set_defer_window` reaches every stage and is cleared after the outer run only (the nesting rule above), once the borrow logic is moved into a small testable helper.
- `serve/test_park_divergence.py`: a restarted engine whose new proc reuses the old object id starts a fresh comparison (R8, after switching to a start counter).
