# Remaining Code Analysis: pancake-strata

## 1. Speculative Decoding (`src/spec/`) — Pure CPU Logic

| File | Size | Role |
|------|------|------|
| `draft_policy.cpp` | 16 KB | MTP/draft token length selection policy (EMA cost model) |
| `disco.cpp` | 11 KB | Learned MTP window rule (logistic regression on draft features) |
| `suffix_index.cpp` | 13 KB | Suffix trie for draft token continuation lookup |
| `suffix_drafter.cpp` | 3.6 KB | Draft token generation from suffix index |
| `controller.cpp` | 3.5 KB | Orchestrates draft/verify rounds |
| `draft_source.cpp` | 1.2 KB | Draft token source abstraction |

**Analysis**: Pure CPU-side logic. No CUDA/HIP. The speculative decoding pipeline:
1. `DraftPolicy::choose()` — selects draft length `t` based on cost model
2. `DiscoPolicy::keep_draft()` — decides if a draft token is kept using learned logistic model
3. `controller.cpp` — orchestrates the draft/verify loop

**Optimization**: None needed — already CPU-optimized. The hot path is integer math + array lookups. No GPU interaction.

## 2. PLE Reader (`src/ngram/`) — PLE (Prefix-Length Encoding) I/O

| File | Size | Key Patterns |
|------|------|--------------|
| `ple_reader.cpp` | 26 KB | 20 `thread` refs, async I/O |
| `ple_reader_test.cpp` | 24 KB | 37 `gather`, 27 `thread` (tests) |

**Analysis**: Async PLE file reader with optional I/O thread (`io_thread` option at line 138). Uses `std::thread` worker for async `io_uring`/`DirectFile` completions. No GPU code.

**Optimization**: 
- Single worker thread — consider thread pool for multiple concurrent readers
- 37 `gather` in test file only (test harness)

## 3. Platform (`src/platform/`) — Cross-Platform Abstractions

| File | Size | Role |
|------|------|------|
| `direct_file.cpp` | 13 KB | 15 `thread` refs — async file I/O (io_uring on Linux, IOCP on Windows) |
| `mem_pressure.cpp` | 13 KB | Memory pressure monitoring (cgroup/virtual memory) |
| `memory.cpp` | 11 KB | Memory allocation tracking |
| `io_priority.cpp` | 2 KB | I/O priority (ionice/IO_PRIORITY_HINT) |

**Analysis**: Pure CPU/OS abstractions. No GPU code. `direct_file.cpp` uses `std::thread` for async completions (line 15 refs). `mem_pressure.cpp` monitors cgroup memory limits for OOM prevention.

## 4. Plan (`src/plan/`) — Minimal

| File | Size |
|------|------|
| `plan_main.cpp` | 3.5 KB |

Tiny entry point. No optimization needed.

## 5. Split (`src/split/`) — Multi-GPU Split

| File | Size |
|------|------|
| `CMakeLists.txt` | 14 KB |

Only CMake config. Actual split logic is in `src/core/` (peer_experts.hpp, expert_source.cpp).

## 5. Serve (`serve/`) — Python Server

**Not analyzed** — Python OpenAI/Anthropic API server, agent loop, tool parsing. No C++ optimization surface. Bottlenecks would be Python GIL, async I/O, or engine calls.

## 6. Bench/Tools — Measurement Suite

**Not analyzed** — Python benchmark harnesses (`abtest.py`, `quality_gate.py`, `night_queue.py`, `spec_lint.py`, `gpu_profile.py`). Not C++ optimization targets.

---

## Summary: No Further C++ Optimization Targets

| Directory | Status | GPU Code? | Optimization Needed? |
|-----------|--------|-----------|---------------------|
| `src/spec/` | Analyzed | No | No (CPU logic) |
| `src/ngram/` | Analyzed | No | No (async I/O) |
| `src/platform/` | Analyzed | No | No (OS abstraction) |
| `src/plan/` | Analyzed | No | No (tiny) |
| `src/split/` | Analyzed | No (in core) | Covered in core |
| `serve/` | Skipped | No (Python) | Python-level |
| `bench/` | Skipped | No (Python) | Python-level |
| `tools/` | Skipped | No (Python) | Python-level |

**All C++ hot paths have been analyzed:**
- `generate.cpp` (host decode) → `optimize-analysis`
- `src/kernels/cuda/` (RDNA3 kernels) → `optimize-kernel-hip`
- `prefill.cpp` (FK-28 hook) → `optimize-prefill`
- `src/poly/` (dense MMQ) → `optimize-poly`
- `src/core/` (mtp, dflash2, verify, expert_source) → `optimize-core`
- `src/kernels/cpu/` (AVX-512/2) → `optimize-core`
- `src/dsv41/` (DeepSeek) → `optimize-core`
- `src/qwen35/` (35B-A3B) → `optimize-core`
- `src/spec/` → No GPU code
- `src/ngram/` → No GPU code
- `src/platform/` → No GPU code

**All C++ optimization targets are complete.**