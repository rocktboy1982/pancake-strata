# pancake: structure and build health (2026-11-09)

Repo `/home/dan/clone-of-d/src/Strata`, branch `port/rdna3-decode`, HEAD `658a6f7aa`. Read-only study: no edits,
builds, tests or GPU runs. Tools: Serena (clangd + pylsp) for symbol sizes and references, grep/awk/python scans, the
build caches and logs in `build-hip/`, `build-cuda/`, `build-clangd/`. Scratch lists (env name sets) are in
`/tmp/claude-1000/-home-dan/6a980654-9e98-4e1d-b7d0-6b6bad53f8e1/scratchpad/sb/`.

## 1. Size map

Module totals (lines of .cpp/.cu/.h/.hpp/.cuh/.py/.metal):

| Module | files | lines | Note |
| --- | ---: | ---: | --- |
| src/kernels (cpu + cuda) | 192 | 84,503 | cuda/ has 62 files; HIP builds the same .cu files through hip_compat |
| src/dsv41 | 45 | 36,859 | ds4_model.cu 15,047 lines (clangd fails on it: "invalid AST") |
| src/core | 68 | 35,115 | verify.cpp 5,790, expert_source.cpp 6,803, mtp.cpp 2,133 |
| src/prefill | 44 | 27,762 | prefill.cpp 8,024, kernels.cu 3,459 |
| src/split | 78 | 25,278 | strata-split, OFF in every build here |
| src/program | 11 | 24,107 | generate.cpp 21,703 |
| src/poly | 42 | 19,401 | all pancake-added (FastStrata port) |
| src/qwen35 | 5 | 5,348 | q36_engine.cu 3,972 |
| src/spec, ngram, platform, plan, artifact | 21 | 4,283 | |
| sycl/ | 250 | 161,997 | dpct copy of src; 120 of its 149 src files mirror a src/ file |
| metal/ | 144 | 37,802 | separate engine |
| serve/ | 144 | 47,885 | server.py 11,138; test_server.py 6,640 |
| tests/ | 333 | 71,775 | |

God functions on hot paths (Serena `get_symbols_overview`, body line spans):

| Function | Lines | Where |
| --- | ---: | --- |
| `main` | **18,361** | src/program/generate.cpp:3280 (85% of the file); `Options` class 423 lines at :605, `usage` 288 at :1029 |
| `Prefill::run_impl` | 3,224 | src/prefill/prefill.cpp:3303 |
| `Verifier::record_window` | 1,479 | src/core/verify.cpp:1416 (the decode window) |
| `ds4 serve` | 1,500 | src/dsv41/ds4_main.cpp:513 |
| `Prefill::lm_run` | 1,033 | src/prefill/prefill.cpp:6863 |
| `FileExpertSource::pin_cache_complement` | 713 | src/core/expert_source.cpp:3543 |
| `expert_pool_dispatch_multi` | 563 | src/core/expert_source.cpp:4906 |
| `Verifier::init` / `Verifier::run` | 519 / 480 | src/core/verify.cpp:884 / :3518 |
| `ArenaExpertSource::open` | 451 | src/core/expert_source.cpp:5944 (item 38/69 territory) |
| serve `Service` class / `StrataEngine` class | 2,441 / 2,061 | serve/server.py:4878 / :1173 |
| `make_handler` (HTTP handler) / `Service._run` / `_main` | 1,412 / 811 / 647 | serve/server.py:8080 / :6508 / :10488 |
| `q36 Engine` class | 1,649 | src/qwen35/q36_engine.cu:1146 |

Compile cost (build-hip/.ninja_log, gfx1100 only, Release): native_mmvq.cu 190 s, iq_kernels.cu 85 s,
**generate.cpp 68.7 s**, fused_gr.cu 63 s, iq_avx2.cpp 42 s, prefill.cpp 24 s, expert_source.cpp 24 s. Every change
of a default inside generate.cpp's `main` costs ~69 s + link. Latest HIP build log (build-hip/build-dev.log): 143
warnings, 127 of them in generate.cpp (139 `-Wunused-result`, 4 `-Wunused-variable`: layer.cpp:1144 `n_kv`,
layer.cpp:1501 `hcd`, q36_engine.cu:1885 `D`).

Pancake's own surface since base 146df3c46: `git diff --numstat 146df3c46..HEAD -- src include` = 103 files, +27,255
/ -76. Biggest: poly/engine.cpp 4,204, poly/kernels.cu 3,182, poly/main.cpp 1,264, grammar.cpp 764, rdna_gr.cu 763,
q5k_head.cu 547, head_fmt.cu 473; serve/loop_guard.py 690 (+1,064 test). sycl/ and metal/: 0 pancake commits.

Module boundaries: core (verify/mtp/expert_source/layer) is the decode engine; program/generate.cpp is the CLI +
startup + server loop of `strata` all in one `main`; prefill is a separate library (`strata_prefill`) linked with
optional `strata_mmq`; kernels are shared by strata, strata-poly, strata-q36, dsv41. Boundary leaks: arch defaults are
applied by `setenv()` from core/arch_defaults.cpp:199 (`setenv(k, v, 0)`), called from generate.cpp:3818 and
nv_defaults at :3886; every kernel file then reads the env itself, so "config" crosses every module boundary as
process-global strings.

## 2. Flag / env sprawl

Counts (method: string literals passed to a function whose name contains `env`, excluding set/unset/put and
`_test`/`_parity`/`_bench` files; Python: `environ`/`getenv` near a `STRATA_` literal, non-test):

| Measure | Value |
| --- | ---: |
| distinct `STRATA_*` names read in C++ (src + include) | **872** |
| `getenv("STRATA_…")` call sites (non-test, non-bench) | 1,046 |
| ... of which not visibly cached in a `static`/global (heuristic, 4-line window) | ~440 (196 in generate.cpp, mostly startup) |
| distinct names read in Python (serve, strata_cli, strata_control) | 81 (45 sites in serve/server.py) |
| union C++ + Python | 944 |
| C++-read names mentioned in any doc (docs/, README*, MOD_PLAN, PANCAKE_STATUS, AGENTS) | 428 |
| C++-read names in no doc | 444 |
| ... and also not in configs/, tools/, bench/, scripts, serve | **369** (only discoverable by reading source) |
| names added by pancake commits (`+getenv("STRATA_` in diff since base) | 48 |
| CMake-level `STRATA_*` symbols (options/macros) | 154, a second namespace with the same prefix |

Reads per file (non-test): generate.cpp 319, prefill.cpp 128, verify.cpp 85, expert_source.cpp 46, iq_kernels.cu
36, qsa_select.cu 31, native_expert.cpp 29, native_mmvq.cu 26, prefill/kernels.cu 25, gemm.cu 25, ds4_model.cu 24,
fused_gr.cu 23, q36_engine.cu 21, mtp.cpp 21.

Hot-path reads:
- Mostly cached as function-local `static const bool x = []{ getenv… }();` inside hot functions (record_window has
  ~20 such statics, e.g. verify.cpp:1437 DEC_BATCH, :1471 COMBINE_CPU, :1472 WAIT_FUSE, :2352 ROUTE_PUBLISH, :2415
  ROUTE_PRIOR_DEPTH). Cheap at run time, but each is an invisible flag that cannot be changed or logged after first use.
- Uncached on a hot path (per launch / per window):
  - fused_gr.cu:1537 `qfuse_hc_local()` calls `getenv("STRATA_QFUSE_HC_LOCAL")` on every call; callers (Serena
    find_referencing_symbols): `launch_multi` :1696, `launch_q8_t` :2591, `fused_gr_read_multi_slice` :2826 - per layer
    per window.
  - verify.cpp:1537 and :1542 `getenv("STRATA_DF_PDL")` per window (twice, second for a log line).
  - verify.cpp:2356 `getenv("STRATA_ROUTE_PRIOR")` inside record_window's per-layer condition.
  Cost is a scan of environ (sub-microsecond to ~1 us each), noise next to 25-37 ms windows: a hygiene item, not a
  speed item.
- Ordering hazard: defaults are injected with `setenv(...,0)` (arch_defaults.cpp:199) after the process starts; any
  function-local static evaluated before `apply_arch_defaults` (generate.cpp:3818) or `apply_nv_defaults` (:3886)
  keeps the pre-default value silently. Nothing checks this.
- Parsing differs per site: `STRATA_TSUM` is `v[0]=='1'` (native_bf16.cu:323, native_mmvq.cu:1318, fused_gr.cu:2584,
  iq_kernels.cu:6322/6361); `STRATA_NO_MULTI_GR` is `v[0]!='0'` in mtp.cpp:1349, gr.cu:468, fused_gr.cu:1626 and
  `env_on()` in verify.cpp:323; `STRATA_DEC_BATCH` is "unset or atoi!=0" in 3 copies (verify.cpp:1437,
  expert_source.cpp:5193, shared_expert.cu:248). `STRATA_HEAD_MIX_MULTI` read twice in a row (verify.cpp:400, :406),
  `STRATA_ONE_TOKEN_COMMIT` twice with HIP/CUDA defaults (verify.cpp:501, :507). Vars read in >= 3 files:
  GFX12_DEFAULTS (5), TSUM (4), NO_MULTI_GR (4), UNBUFFERED_LOAD, SM70_TABLE, ROUTE_TAIL_SKIP, PF_SWITCH_MIN_T,
  PF_FUSED, MMQ_KQ, IQ_PREFETCH, GFX1151_DEFAULTS, DEC_BATCH, DBG_NAN (3 each).

Dead / drift:
- Documented but read nowhere (src, include, serve, sycl, metal, tools, scripts): 16 names. Real ones for pancake:
  `STRATA_BFLA`, `STRATA_ATTN_WMMA` (README.md:53, they live on feat/bfla-hip / side branches), `STRATA_PROBE`
  (MOD_PLAN.md:267, planned item 63), `STRATA_KV_RING_ADAPT` (PANCAKE_FINDINGS). The rest are upstream docs
  (A770, Strix, NVFP4: DOORBELL_NOCHECK, PREFILL_X2_MIN_K, PF_ prefix...).
- Flags reported as rejected/slower in FINDINGS/MOD_PLAN and still read: `STRATA_ARENA_WARM`, `STRATA_PF_WMMA`
  ("WMMA worse"). (`STRATA_VERIFY_DEFER_CPU` also matches but is the live item 8 switch.)
- Presets are clean: every `env` key in configs/pancake/*.json (up to 51 in the -hip/turbo presets) is read by the
  code. But run-pancake.sh regenerates `pancake-<preset>.json` at the repo root from the template (run-pancake.sh:53-55);
  the stale root copy `pancake-flashnext-iq2_xs-fast-hip.json` (Nov 8 17:40) points STRATA_HIPBLASLT_TUNING at
  `gfx1100-hipblaslt-100200.txt` while the template has `100401` - one more reason to log the effective config.

"Read but never effective" cannot be proven by grep; candidates are the 369 undocumented names. The registry below
gives the tool to find them (count reads at runtime, report names set in the environment that nobody read).

### Proposal: one typed config registry (ties to items 63 and 57)

- `include/strata/core/flags.def`: an X-macro table, one line per flag:
  `STRATA_FLAG(QFUSE_HC_LOCAL, bool, true, Scope::Kernel, Exact::Yes, "atomic-tail up kernel under QFUSE_HC")`.
  Scopes: Startup, Kernel (read once, hot), Debug, Test, Lossy (lossy flags print the warning ground rule 2 asks for).
- `strata::cfg()` returns a const struct filled ONCE after `apply_arch_defaults`/`apply_nv_defaults` (that fixes the
  setenv ordering hazard: the defaults become the struct's defaults, not setenv calls). Kernel code reads
  `cfg().qfuse_hc_local` - a load, no environ scan.
- Startup log: one line per non-default flag (`[cfg] STRATA_TSUM=1 (arch default gfx12)`), plus the item 57 ISA line
  and the item 63 machine-profile hash, also written to machine-profile.json and to the per-turn JSONL of item 73.
  Unknown `STRATA_*` names in the environment are listed as "set but unknown" (typos, side-branch flags).
- `--print-config` / `--list-flags` (markdown table) generates the docs, so documented == read by construction.
- Python side: serve/runconfig.py gets the same table for its 81 names (or reads a JSON dump of flags.def).
- Lint `tools/pancake/flag_lint.py` (seconds, no build): every `getenv("STRATA_` outside flags.cpp is an error
  except Test/Debug allow-list; every preset key must be in the table.
- Migration: incremental. Step 1 the ~51 preset flags + the 48 pancake-added names + the 3 uncached hot reads;
  upstream's long tail stays on getenv until touched. Keep the env var names (presets, users).

## 3. Arch-path duplication and stub maintenance

- HIP uses the CUDA sources through hip_compat; arch specialisation is inline `#if`: `__HIPCC__`/HIP macros 282
  lines in 51 files, gfx11 463 lines in 70 files, gfx12 464 lines in 75 files, RDNA 275 lines in 54 files,
  `__CUDA_ARCH__` 70 lines in 25 files. Separate RDNA kernels: rdna_gr.cu, rdna_mmvq.cu (pancake, 413e1b968 "HIP-only
  guards with CUDA stubs").
- SYCL: full dpct copy (sycl/src/program/generate.cpp 13,896, verify 4,216, prefill 5,215...), 0 pancake commits,
  untuned by policy (MOD_PLAN non-goals). Metal: separate engine, 0 pancake commits. Both double every grep/clangd
  hit for a hot-path symbol; they should be excluded from search and indexing, not maintained.
- Hand-written stub blocks that must mirror headers: prefill.cpp:116-147 (`#ifndef STRATA_PREFILL_MMQ`, ~35 mmq::
  functions) and :149+ (`#ifndef STRATA_PREFILL_FUSED`, fused:: functions incl. `gfx12_iq2s_default` :153). The
  2026-11-09 link break (`want_dm_f16`/`dm_f16_used` missing since 6e31946cb, PANCAKE_FINDINGS.md:485) is exactly a
  header gaining a function without its stub. Same pattern for RDNA CUDA stubs and for `#if !defined(STRATA_USE_HIP)`
  (29 + 8 + 4 + 3 + 2 guard sites).
- Build dirs and their options (CMakeCache): build-hip HIP+MMQ ON gfx1100 Ninja; build-clangd HIP+MMQ ON; build-cuda
  and build CUDA sm_86 (Unix Makefiles), MMQ OFF. **All four have STRATA_BUILD_TESTS=OFF** and STRATA_BUILD_SPLIT=OFF.
  The HIP MMQ-OFF combination exists in no build dir, so the stub break was invisible until someone tried it.
  strata-poly is a target in build-hip/build.ninja (`build strata-poly:`), but run-pancake.sh:49 treats it as
  CUDA-only; the HIP poly build is not exercised. Note strata_poly_mmq compiles its own copy of
  src/prefill/ggml_cuda_host.cu and ggml quantize.cu (CMakeLists.txt:3080), the same files as strata_mmq: fine while
  no binary links both libraries, a duplicate-symbol trap if one ever does.
- No GPU build CI: .github/workflows has metal-build.yml and terminal/docker workflows only. No ccache on the box;
  nvcc and hipcc are not on PATH (ROCm at /opt/rocm).

### Proposal: build matrix check (local script, later a self-hosted runner)

`tools/pancake/build_matrix.sh`, build dirs on D (`/home/dan/clone-of-d/build-matrix/<name>`), reusing the fetched
llama.cpp via `-DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=<build-hip/_deps/strata_llamacpp-src>`:

| Row | Options | Check | Catches |
| --- | --- | --- | --- |
| hip-mmq | ENABLE_HIP, PREFILL_MMQ=ON, gfx1100 | configure + build `strata strata-poly strata-q36` | daily path, poly on HIP |
| hip-nommq | ENABLE_HIP, PREFILL_MMQ=OFF | configure + link `strata` | stub drift (the 11-09 break) |
| cuda-86 | ENABLE_CUDA, sm_86 | configure + build `strata strata-poly strata-dsv41` | ground rule 3 the other way |
| hip-tests | hip-mmq + BUILD_TESTS=ON | build + `ctest -L` CPU-only subset, then GPU subset by hand | the 293 add_test entries |
| configure-only | DSV41 ON, SPLIT ON, PORTABLE ON, NATIVE_EXPERTS OFF | `cmake` configure only (seconds) | CMake logic breaks |

Make it incremental (Ninja, `ccache` as `CMAKE_{CXX,HIP,CUDA}_COMPILER_LAUNCHER`) and run it from a git pre-push hook
for branches that touch src/ or CMakeLists.txt; full cold cost is dominated by native_mmvq/iq_kernels (~5 min per HIP
dir at -j16), warm runs re-link only. Cheap structural guard for stubs: move each stub block into its own
`*_stub.cpp` next to the header and add a header-parity script that lists functions declared in mmq.hpp/fused.hpp
without a stub (seconds, no compiler).

## 4. Tests

- C++: 97 `*_test.*` and 86 `*_parity*` files in src/ (+20 benches), tests/ has 351 files (core, cuda, hip, prefill,
  dsv41, program, terminal...), CMakeLists.txt has 293 `add_test(`. None are built in any current build dir
  (BUILD_TESTS=OFF everywhere).
- Python: 113 serve/test_*.py with 1,580 `def test_`; tests/ has 221 test_*.py (mostly terminal/desktop/launcher).
  ContextPilot commit: "1604 tests, 1 failure (test_keepawake early_close)" (quoted in
  research_notes/Pancake pain points și soluții/server_agent_ops.md).
- Restore/checkpoint coverage today is component-level: conversation_snapshot_test.cpp (3 ctest variants:
  plain, CKPT_ASYNC, VRAM slots, CMakeLists.txt:877-887), conversation_file/validation/split_failure tests,
  program/message_boundary_test.cpp, poly/conversations_test.cpp (615 lines, pancake), serve/test_ds4_restore_points.py.
  **No test compares decode after a restore with a cold decode** (item 35: full + partial hit, every boundary type,
  MTP on). A ready hook: `STRATA_LOGITS_DUMP=<path>` (verify.cpp:3953) writes each window's logits; the test can run
  prompt P+Q cold, then P, checkpoint, Q via restore, and compare greedy tokens and logits bitwise per boundary kind.
- Hot-path components without a parity test: rdna_gr.cu and rdna_mmvq.cu (no test or parity file references them);
  record_window's window path is covered only by end-to-end greedy checks; grammar has 3 tests
  (CMakeLists.txt:970); q5k_head and head_fmt each have one.
- serve/test_keepawake.py failure, cause from reading the code: `KeepAwake._loop` (serve/keepawake.py:51-68) is
  level-triggered: it waits while `(count > 0) == held`. In
  `test_service_holds_for_the_whole_request_including_early_close` (test_keepawake.py:80) the MockEngine "ok" request
  finishes (acquire + release) before the new thread first takes the lock; it then sees count 0 == held False and
  keeps waiting, so `ON` is never sent and `wait_for(rec.calls == [ON, ES_CONTINUOUS])` (:86) times out after 3 s.
  The feature behaves well (a request shorter than a thread start does not need to hold the PC awake); the test asserts
  an edge that the design coalesces. Fix: assert the second (held-open) half only, or make `acquire()` record a pulse
  the loop must emit; or skip off Windows (MOD_PLAN.md:240 hygiene line). Not run here (read-only).

## 5. Dead code on decode/prefill paths

Header functions with no caller (token count <= 2 in src + include + tests = declaration + definition; confirmed with
Serena `find_referencing_symbols` = 0 for the first four):
- core/layer.cpp: `layer_fused_gr` :553, `layer_shared_early` :555 (its setter `layer_set_shared_early` :554 is used,
  the getter not), `stage_mark_begin`/`stage_mark_end` :1090-1091, `gdn_buffers_zero_state` :246.
- kernels: `native_moe_combine_multi_hits` (native_moe.cu:332), `native_mmvq_b6_rows` (native_mmvq.cu:3357),
  `df2_norm_rope` (dflash2.cu:946), `qsa_hip_wmma_active` (qsa_prompt_attn.cu:2453), `q5k_head_mask_unmapped`,
  `native_qsa_score_set_enabled`, `ple_prefetch_enabled`, `MtpDrafter::record_top2` (mtp.cpp).
- Declared only, never defined or called: `log_row`, `df2_rope_table_cpu`, `df2_attention_chunk`,
  `doorbell_plain_payload`, `doorbell_wait_payload`, `iq_xmx_grouped`, `gu_max_mean`, `grouped_f16` (xmx/sycl
  leftovers in shared headers), `conversation_prefix`, `write_report`, `soft_reserve_default_mib`, `batch_key`.
Total scan: 1,332 header functions in include/strata/{core,prefill,kernels,spec}, 39 flagged. The volume is small
(dozens of lines). The larger "dead" mass is opt-in branches inside live functions (rejected flags such as
PF_WMMA, ARENA_WARM; ~20 function-local static switches in record_window), which only the flag registry with
per-flag "measured, rejected" status can retire safely.

## Ranked structural improvements

Ranked by how much they speed up or de-risk the open MOD_PLAN items (Tier A first).

| # | Improvement | Unblocks / speeds | Effort | Risk | First step |
| --- | --- | --- | --- | --- | --- |
| 1 | Restore-equivalence harness (item 35) on STRATA_LOGITS_DUMP | 35 gates 3, 11, 16, 31, 39, 66, 71 | M | low (test only) | Python test in serve/ that runs strata twice (cold vs checkpoint+restore) at one message boundary and diffs dumped logits; then add tool / think boundaries and MTP on |
| 2 | Build matrix script (hip-mmq, hip-nommq, cuda-86, configure-only rows) + pre-push hook | ground rule 3, every kernel item (1, 32/48, 47, 10, BitDecoding, BFLA) | S | none | write tools/pancake/build_matrix.sh with the 5 rows above, build dirs on D, ccache installed |
| 3 | A tests build dir (BUILD_TESTS=ON) + labelled ctest CPU subset | regression net for 8/28, 23, 29 (verify/expert code) | S | low (disk: D has 73 GB free) | configure build-hip-tests, `ctest -N` to list, tag CPU-only tests with LABELS |
| 4 | Typed flag registry with startup log (flags.def) | 63 (probe), 57 (ISA line), 67 (capability-tagged harness), 73 (per-turn log carries the config), 7/23/28 A/B reproducibility | M (step 1) / L (all) | medium: ordering with apply_arch_defaults, must stay bit-exact | flags.def with the ~51 preset flags + 48 pancake flags + startup "[cfg]" line; flag_lint.py in the matrix |
| 5 | Split generate.cpp `main` (18,361 lines): move Options/usage, arch-default/probe block (:3802-3890) and server loop into separate TUs | 63, 69, 38 (startup work), 73; cuts the 69 s recompile of every startup change | M | low if pure moves (check greedy exactness + flag log identical) | extract `Options` + `usage` + startup defaults into src/program/startup.cpp with no logic change |
| 6 | Stub blocks into `*_stub.cpp` + header-parity script | MMQ-off and CUDA-stub breaks (11-09 class) | S | none | move prefill.cpp:116-160 into prefill/mmq_stub.cpp; script lists mmq.hpp/fused.hpp functions without a stub |
| 7 | Exclude sycl/ and metal/ from search and indexing (.ignore, .clangd, Serena ignored_paths) | every code read: ~200K lines of mirrors out of grep hits | XS | none | add `sycl/` `metal/` to `.ignore` and Serena project.yml ignored_paths |
| 8 | Extract-on-touch rule for record_window (1,479) and run_impl (3,224) | S2 (one re-selection in verify.cpp: 7 vs 23), 46-52 overlap package, 8/28 | L if done at once; S per extraction | high (hot path, exactness) | only when an item edits a block: lift that block into a named function in the same TU, A/B greedy-exact, no other change |
| 9 | server.py: split `Service` (2,441) and `make_handler` (1,412) along the item 66 turn pipeline (turn plan, checkpoint policy, thinking caps) | 66, 12, 31, 11, 16, 39 | M | low-medium (1,580 tests cover it) | new serve/turn_plan.py for 66's planner instead of growing Service._run (811 lines) |

## Quick hygiene

- Fix or skip test_keepawake early_close (cause in section 4) so the Python suite is green for A/B gates.
- Cache the 3 uncached hot getenvs: fused_gr.cu:1537 `qfuse_hc_local`, verify.cpp:1537/1542 `STRATA_DF_PDL`,
  verify.cpp:2356 `STRATA_ROUTE_PRIOR`.
- Remove the 4 unused variables (layer.cpp:1144, :1501, q36_engine.cu:1885) and look at the 139 `-Wunused-result`
  in generate.cpp (unchecked CUDA/HIP calls).
- Delete the ~20 declared-but-undefined header prototypes and the zero-caller getters listed in section 5.
- One parser for boolean flags (`env_on`) instead of `v[0]=='1'` vs `v[0]!='0'` variants (TSUM vs NO_MULTI_GR).
- Delete the stale root `pancake-*.json` copies (regenerated by run-pancake.sh anyway) or print the effective
  hipBLASLt tuning file at start (template 100401 vs stale copy 100200).
- Document or remove the README/MOD_PLAN mentions of side-branch flags (`STRATA_BFLA`, `STRATA_ATTN_WMMA`) on this
  branch.
- clangd cannot parse src/dsv41/ds4_model.cu (Serena "invalid AST"); add a `.clangd` CompileFlags fix or exclude it
  so symbol tools stop failing on project-wide queries.
