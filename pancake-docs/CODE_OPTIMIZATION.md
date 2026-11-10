# Code-level optimization pass (the 7 points), 2026-11-09

The seven usual optimization points applied to pancake's **code**, not only to its features.  Tools: Serena (symbols,
references), graphify (`graphify-out/`: 77,004 nodes, 179,775 edges over 5,174 files, built from `eecadd0d`),
ast-grep (structural rules over the C++/CUDA AST), clang-tidy (`performance-*`, compile database `build-clangd/`),
rocprofv3, the engine's own timers.

| # | Point | Applied to the code | Result |
|---|---|---|---|
| 1 | Profiling | rocprofv3 kernel trace (`tools/pancake/gpu_profile.py`), the engine's decode-timing split (GPU / CPU experts / draft), dispatch sub-timers (`jobs`, `expert source`), `decode_curve.py` | the CPU expert time is 42-47% of a window; GPU kernels ~400 per token with ~5 us gaps |
| 2 | Algorithms / data structures | ast-grep + reading of the hot functions (`expert_pool_dispatch_multi`, `Verifier::run`, the pool's `drain` / `run_phase` / `wait_*`) | **fixed 4 quadratic loops in the per-layer dispatch** (below) |
| 3 | Caching | `getenv` calls on the hot path (ast-grep rule: calls outside a static initializer or lambda, grouped by function) | none left per window: `STRATA_LOGITS_DUMP` and `STRATA_ROUTE_PRIOR` cached earlier (`c44b1ad7b`, `7c0f03228`); the rest are init / capture-time |
| 4 | Database / I/O | no database; the analogue is model I/O: start-up arena load (item 38, O_DIRECT, dan-9f) and the expert source's prefetch/blob lookups (sub-timer) | decode does no file I/O once the arena is resident |
| 5 | Async / parallelism | CPU work overlapped with the GPU: expert deferral (8/28), asynchronous tier + k-th-miss (14), persistent adapt worker (75), draft stream on its own queue (`STRATA_MTP_HIP_PRIO`) | coded, measured in the test phase (largest modeled gain: +33-52% on edits) |
| 6 | Refactoring / allocations | hot functions scanned for per-call heap allocations, `std::function`, locks, string building | none on the hot path (only in error / debug branches); launches fused (router, SwiGLU), pool barriers trimmed (lazy park, fuse phases, MWAITX) |
| 7 | Static analysis | clang-tidy `performance-*` on pool.cpp, expert_source.cpp, q2_avx2.cpp; ast-grep rules | only cold-path findings (string concatenation in errors, enum sizes, push_back at init) |

## Point 2: the four quadratic loops fixed in the per-layer dispatch (src/core/expert_source.cpp)

Every verify window calls `expert_pool_dispatch_multi` once per layer (48 per window).  With a 16-row window the
layer has up to 160 routed entries and up to ~100 distinct experts, and four steps were quadratic:

1. `first_of[i]` (the first entry routed to the same expert): a backward search per entry, O(n^2) -> a per-expert stamp
   and first index, O(n).
2. The distinct CPU experts in first-routed order (`miss`): `std::find` over the list per entry, O(n^2) -> a per-expert
   stamp, O(n).
3. Placement and plan groups (`kind`, the VRAM groups' and PCIe groups' entry lists): for every distinct expert a scan
   of the whole window, O(distinct x n) -> a `next_of` chain (the next entry of the same expert, in increasing order),
   built once in O(n).
4. Tail-skip rank (`STRATA_ROUTE_TAIL_SKIP`): a scan of the distinct list per entry -> an O(1) index (`q_of`).

Same values in the same order (the chains walk entries in increasing index, as the scans did); out-of-range ids keep
the old pairwise rule.  Worst case per layer goes from tens of thousands of comparisons to a few hundred.  HIP and CUDA
build; the effect shows in the dispatch `jobs` / `plan` timers and is part of the test phase (greedy output must be
unchanged: token forcing on the next quality-gate run).

## Compiler

`STRATA_HOST_NATIVE` (CMake, opt-in): Strata's host C++ with `-march=native -ffp-contract=off`; built as
`build-hip-native`, A/B in `bench/specs/08-build.json` (the explicit AVX2/AVX-512 kernels are unchanged; the generic
code around them gains AVX2/FMA/BMI).

## Tools added for this pass

- AST: `ast-grep` CLI (0.51) and the ast-grep MCP server (connected); `mcp-server-tree-sitter` needs `mcp<2`
  (`uv tool install --force mcp-server-tree-sitter --with 'mcp<2'`).
- graphify: `graphify update .` (no LLM) -> `graphify-out/` (git-excluded); query with `graphify query "..."`,
  `graphify explain "<symbol>"`, `graphify path "A" "B"`.
