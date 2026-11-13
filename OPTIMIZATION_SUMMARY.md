# Optimization Analysis Summary: pancake-strata

## All Reports

| Report | File | Key Findings |
|--------|------|--------------|
| Host-side | `OPTIMIZATION_REPORT.md` | 39 `cudaDeviceSynchronize`, 1188 `printf` in `generate.cpp` |
| HIP kernels | `HIP_OPTIMIZATION_REPORT.md` | Enable `STRATA_RDNA_MMVQ_WAVE=1`, `STRATA_HIP_WMMA=1`, add `amdgpu_waves_per_eu` |
| Prefill | `PREFILL_OPTIMIZATION_REPORT.md` | FK-28 `drive.d.usage` hook missing (~10 lines) |
| Poly | `POLY_OPTIMIZATION_REPORT.md` | Dense MMQ for 35B-A3B down experts (20-40% prefill) |
| Core | `CORE_OPTIMIZATION_REPORT.md` | `mtp.cpp` sync memcpy, `dflash2.cpp` `cudaDeviceSynchronize` |
| CPU kernels | `CPU_OPTIMIZATION_REPORT.md` | AVX-512 downclocking, gather→shuffle, prefetch |
| DSV41 | `DSV41_OPTIMIZATION_REPORT.md` | 222 `cudaMemcpy` in `ds4_model.cu`, unify `mma`/`wmma` |
| QWEN35 | `QWEN35_OPTIMIZATION_REPORT.md` | 69 `mmq` refs, dense MMQ, batch async |

## Branches

- `optimize-analysis` — host-side fixes
- `optimize-kernel-hip` — RDNA3 kernel toggles
- `optimize-prefill` — FK-28 ADAPT_PROMPT hook
- `optimize-poly` — poly engine dense MMQ
- `optimize-core` — core engine + CPU kernels + DSV41 + QWEN35

## Top 5 Optimizations (by impact)

| # | Target | Impact | Effort |
|---|--------|--------|--------|
| 1 | `generate.cpp`: 39 `cudaDeviceSynchronize` → events | 40-70% decode latency | Medium |
| 2 | `ds4_model.cu`: 222 `cudaMemcpy` → batch async | 20-40% decode | Medium |
| 3 | `q36_engine.cu`: dense MMQ for down experts | 20-40% prefill | High |
| 4 | `prefill.cpp`: FK-28 ADAPT_PROMPT hook | Better expert residency | Low (~10 lines) |
| 5 | `native_mmvq.cu`: enable `STRATA_RDNA_MMVQ_WAVE=1` | 10-30% MMVQ on 7900 XT | Low (env var) |

## Next Steps (Linux ROCm Box)

```bash
# 1. Fetch all branches
git fetch origin

# 2. Check out and implement
git checkout optimize-analysis   # host-side fixes
git checkout optimize-kernel-hip  # RDNA3 kernel toggles
git checkout optimize-prefill     # FK-28 hook
git checkout optimize-poly        # poly dense MMQ (CUDA-only)
git checkout optimize-core        # core + CPU + DSV41 + QWEN35

# 3. Build
cmake -S . -B build-hip -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip --target strata -j2

# 4. Test
ctest --test-dir build-hip -L cpu --output-on-failure
ctest --test-dir build-hip -L gpu --output-on-failure

# 5. Measure
python tools/pancake/abtest.py bench/specs/28.json --out bench/results/optimize-wave1.jsonl
```

## What Was NOT Analyzed (Remaining)

- `src/spec/` — speculative decoding (disco, draft_policy, suffix_index, suffix_drafter)
- `src/ngram/` — PLE reader (ple_reader.cpp, ple_reader_test.cpp)
- `src/plan/` — plan_main.cpp (3.5 KB, small)
- `src/platform/` — direct_file, mem_pressure, memory, io_priority
- `src/split/` — multi-GPU split (only CMakeLists.txt in src/split)
- `serve/` — Python server (OpenAI/Anthropic APIs, agent loop)
- `bench/` — benchmark harnesses
- `tools/` — tool scripts

These directories have lower optimization impact. The hot paths are in `generate.cpp`, `ds4_model.cu`, `q36_engine.cu`, `native_mmvq.cu`, `iq_kernels.cu`, and `expert_source.cpp` — all analyzed.
