# Upstream sync 2026-11-12: continuation of the 2026-11-10 batch

Branch target: a fresh `work/upstream-sync` from `dev/p17/integration`. Upstream range: last cutoff
`dbb2d168c` (2026-11-10) .. `upstream/main` = `13eedf55` (2026-11-12): **54 commits** (shown `--no-merges`).
Fork base stays `146df3c46` (v0.2.12); merge-base with `upstream/main` confirmed `146df3c46`.

Scope (same coordinator brief as the 2026-11-10 batch): only bug fixes and performance that touch what pancake
runs on Linux — Flash-Next **Qwen3.8 (qwen35 engine)**, HIP gfx1100 + CUDA sm_86, the AVX2 CPU expert path, `serve/`
and the OpenAI endpoint, conversation/prompt cache, drafting/MTP, IQ2_XS/Q2_0. Skip Mac, Windows, installer/net/lane-15,
App, launcher/setup/wizard, docs-only, picture reader, new-model/Qwen4 geometry.

Counts: 54 non-merge. **Out of scope: 45** (lane-15 net/installer/download wording ~13, App/launcher/setup/wizard ~8,
Windows-only ~3, picture reader ~4, docs-only ~6, DeepSeek/DFlash2-only + Strix Halo/Blackwell/Arm-specific ~6, qwen36
App-estimate ~3, and test-only ~2). **In-scope candidates: 9**, of which 6 are clear bug fixes to take.

## In-scope commits, in upstream order (take list)

| upstream | what | class | take? |
|---|---|---|---|
| `13eedf55` | `serve`: #4677 stop bound on every backend, not only HIP (was HIP-gated; the stop-deadline fix on the other backends) | serve bugfix | **TAKE** `cherry-pick -x` (touches `serve/`; conflicts unlikely — our `serve/` deltas are small) |
| `d7f7d967` | `qwen35`: give MMQ's row-id table the margin llama.cpp's kernels read past its end — the Flash-Next/qwen35 path we run; a latent read-past-end that Strata-GLM flagged for our family | engine/prefill bugfix | **TAKE** (mirrors the fork-KB `%256` MMQ hazard check; resolve conflicts against our `prefill/`/`iq_kernels.cu`) |
| `14dd746f` | `prefill`: same row-id margin for the **dense** MMQ table (`STRATA_DENSE_MMQ=1`) | engine/prefill bugfix | **TAKE** (unblocks the DENSE_MMQ arm of spec 32 safely; the 2026-11-10 doc deferred this "%256 gate check") |
| `697b6d67` | `prefill`: same margin for the DeepSeek MTP draft step's MMQ row-id table | engine bugfix (DFlash2 path we do not run) | OPTIONAL — same family; take only if it cherry-picks clean and keeps the margin family consistent |
| `711a7e93` | `engine`: the stall report asks the prompt path's compute stream whether its work is done | engine diagnostic | **TAKE** (strengthens the timing-adjacency stall report the `gpu_profile.py` capture relies on) |
| `03e687ac` | AMD Linux: report how long the driver paused the GPU (stall report + request line) | AMD/HIP diagnostic | **TAKE** (gfx1100-specific; helps date the bimodal-slow stalls) |
| `bbf97633` | `serve`: the STOP of an early-ending request never raises, whatever the engine pipe holds | serve bugfix | **TAKE** (`serve/` robustness) |
| `92f7e819` | tests: CPU parity tests skip only their bit-equality leg on older-CPU builds | test | **TAKE if we run host CPU ports** — lets `iq_avx2`/`q2_avx2` parity (FK-23) run its bit-leg on the 5700G; test-only, zero engine risk |
| `feaa928f` | resident RAM copy: `STRATA_RESIDENT_PIN` also for the automatic copy; clamp line names no flag | engine mem | REVIEW first — read the diff vs our `expert_source.cpp` resident-copy path; take only if it matches our code shape (we run under `capped.sh` / `pancake.service MemoryMax`, so the automatic-copy path may differ) |

Not-taken in-scope notes: `5f0ae5b5` (fake clock thread) and `b38d9f89`/`2f1f915a`/`92f7e819` are test-only (take only the parity-leg one, above).

## Explicitly NOT taken (out of scope or default-flip), for the record

- **Lane 15 / net / installer / downloads**: `0771e355`, `cadeb1f6`, `465fcbbc`, `54002e83`, `4c062684`, `be9d5f50`, `84069c4f`, `34ae0dcc`, `e77c407b`, `70bfa88d` — the one-network-rule batch; none touches the engine or `serve/` decode path we run.
- **App / launcher / setup / wizard**: `2b9df3fe`, `dffa1049`, `cfe755e7`, `1fea75a4`, `b94c2931`, `a7f384ef` — App/Control/launch; not in pancake's daily engine path.
- **Windows-only**: `5d87c4d8` (kill-on-close job), `2f1f915a`, `f88ea5bb`, `ca9c53c1` (Windows resume wording).
- **Picture reader (any backend)**: `3b3c41e4`, `a9504669`, `b90d38aa`, `b94c2931` — our presets start no picture reader.
- **Docs-only / wording**: `27108a4f`, `c28474dc`, `40334701`, `9947e7d6f`-style — no code.
- **qwen36 App/memory-estimate (#5715)**: `794092c4`, `9e8502b7`, `ebe1fcc9` — Qwen3.6 context estimate text; we run qwen36 only on CUDA `strata-poly` and the change is a message, not engine behaviour. Reconsider if a qwen36 HIP path returns.
- **DeepSeek/DFlash2/Strix Halo/Blackwell/Arm defaults** (same family as the 2026-11-10 skips): `697b6d67` (take optional, above), plus any DFlash2 conf defaults — we do not run DFlash2.

## New switches inherited

None among the take-list except `STRATA_ARENA_FILE_CACHE=0` and `STRATA_BACKTRACE=1` already recorded by the 2026-11-10 batch. `feaa928f` extends `STRATA_RESIDENT_PIN` to the automatic RAM copy (inherit only if that commit is taken).

## Tests for the coordinator (build + suite on the Linux box; nothing run here — box busy / rule 3, static review only)

| commits | test | why |
|---|---|---|
| all taken C++ (`d7f7d967`, `14dd746f`, `697b6d67`, `711a7e93`, `03e687ac`, `feaa928f`) | full HIP gfx1100 build (+ CUDA if matrix runs); `ctest -L cpu` | not built here; MMQ table sizing is a latent-overflow fix, must compile and keep the cpu suite green |
| `d7f7d967`, `14dd746f`, `697b6d67` | parity: default-path tokens byte-identical to `dev/p17/integration`; run the DENSE_MMQ arm with a non-multiple-of-256 row count | the margin must not change numerics on the normal path; it only pads the table |
| `13eedf55`, `bbf97633` | `python -m pytest serve/test_server.py serve/test_security.py -q` | stop-bound + early-STOP robustness on every backend |
| `03e687ac`, `711a7e93` | one decode run under `STRATA_VERIFY_PROFILE=1`; confirm the stall report prints the compute-stream-done check and the AMD pause duration | diagnostics only; no numerics |
| `92f7e819` | `python -m pytest` on the CPU parity tests (bit-equality leg) | gate for FK-23 SDK |

## Method (unchanged from 2026-11-10)

Take only bug fixes + perf that touch the Linux paths above; never take an engine-default flip or a Windows/net/App/launcher
change without a decided A/B (fork policy). `cherry-pick -x` keeps provenance; hand-port partials say so. No force-push.
This batch is docs+branch-prep; the cherry-picks themselves are the next action (conflict-prone engine picks best done beside a build).

## Picks landed on `work/upstream-sync-1112` (from `dev/p17/integration`)

Cherry-picked with `-x` (upstream author preserved). **Not built/tested here (Windows, no ROCm) — build + `ctest -L cpu`
+ parity on the Linux box before merge.**

| ours | upstream | files | note |
|---|---|---|---|
| `7529df36` | `13eedf55` | `include/strata/core/serve_stop_bound.hpp`, `src/program/generate.cpp`, `tests/core/serve_stop_bound_test.cpp` | conflict resolved: kept HEAD's `#endif` + A19 `STRATA_THINK_PROBE` block (upstream side empty), preserving the new unconditional `serve_stop_bound` call; dropped `data/settings-register.json` (DU — we intentionally do not ship it) |
| `2b6f65e9` | `bbf97633` | `serve/server.py`, `serve/test_stop_closed_pipe.py` | clean auto-merge; py_compile ok |
| `daefd63a` | `d7f7d967` | `src/qwen35/q36_engine.cu` | clean auto-merge; the Flash-Next (qwen35) MMQ row-id margin fix |
| `55b8c26b` | `14dd746f` | `src/prefill/gemm.cu` | clean auto-merge; dense-MMQ row-id margin (`STRATA_DENSE_MMQ`) |
| `65da6516` | `711a7e93` | `CMakeLists.txt`, `src/prefill/prefill.cpp`, `src/program/generate.cpp`, `stream_watch.hpp`+test | clean auto-merge; stall report asks the prompt compute stream |
| `f70e3dcd` | `03e687ac` | `CMakeLists.txt`, `src/program/generate.cpp`, `gpu_pause_time.hpp`+test | clean auto-merge; AMD GPU-pause duration in the stall report |
| `131cca3f` | `697b6d67` | `src/prefill/prefill.cpp` | clean auto-merge; DeepSeek MTP draft MMQ row-id margin — completes the margin family even though we do not run DeepSeek/DFlash2 (belt-and-braces; no numerics change) |

Not picked (recorded): `feaa928f` (automatic-RAM-copy `STRATA_RESIDENT_PIN` — REVIEW: our cap path may differ), `92f7e819` (older-CPU parity-leg
skip — take later beside FK-23). Two new test hooks came in and must be registered/kept green: `serve_stop_bound_test`,
`stream_watch_test`, `gpu_pause_time_test`, `test_stop_closed_pipe.py` (add any missing ones to the cpu/tests labels).
