# Windows + CUDA 12.8 on a GTX 1660 Ti (sm_75, 6 GB): what builds, what runs, what was measured

Written from one session on one PC. Everything below was produced there; what was not tested is listed at the end.

**Hardware and tools:** GeForce GTX 1660 Ti 6 GB (Turing, compute capability 7.5, no tensor cores), 13.9 GB RAM, Windows 11,
CUDA 12.8.61, Visual Studio 2022 Build Tools (MSVC 14.44), CMake 4.4.4, Ninja 1.13.2.

## Build

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build-sm75 -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON ^
      -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
cmake --build build-sm75 -j 4 --target strata strata-q36 strata-poly
```

`-DCMAKE_CUDA_RUNTIME_LIBRARY=Shared` is what `setup.py` gets for a CUDA 12 build through `STRATA_EXPERIMENTAL_SM60=ON`
(CMakeLists.txt, near line 222). Without it `strata-q36.exe` fails to link (`LNK2005`, cudart static and shared).

### MSVC problems found

| Target | Error | State |
| --- | --- | --- |
| `strata` | `__builtin_popcount` in `include/strata/program/kmiss_tier.hpp` (GCC/Clang only) | fixed in this branch: `std::popcount` from `<bit>`, same value |
| `q5k_head_test` | `std::to_string` without `<string>` | not fixed |
| `flag_order_test` | `__atomic_load_n`, `__ATOMIC_SEQ_CST` (GCC builtins) | not fixed |
| `gemm_w16_parity` | `setenv` (POSIX) | not fixed |
| `poly_kv_q4_test` | `kv_q4_test.cu:342`, range-for type not deduced | not fixed |

The four test targets do not block the engine. They were not run.

## Model and result

Model: `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, `UD-IQ2_XXS`, 11.01 GB (`general.architecture` = `qwen35moe`). Its tensor types
(IQ2_XXS, IQ3_XXS, IQ4_XS, IQ2_S, Q2_K, Q3_K for experts; Q4_K, Q5_K, Q6_K, Q8_0 elsewhere) all have a kernel in
`src/poly/mmq.cu`. `strata-poly` loaded it with 2319 of 10496 routed experts in VRAM (2.1 GiB), the rest on 8 CPU threads.
The answers were readable and on topic (a short explanation of mixture-of-experts, a correct Python `fibonacci`); two
prompts, greedy, so this shows the output is not broken and nothing more.

**Decode speed**, `tools/pancake/bench_win.py 3 3 3 96`: 3 runs, each a fresh server, 3 warm-up requests not counted, 3
measured per prompt, 96 new tokens, decode timed from the first streamed token to the last. System CPU before each run
4-7%, 9.6 GB RAM free.

| Prompt | Decode (mean, 95% interval, n=3 runs) | Per run | Time to first token |
| --- | ---: | --- | ---: |
| short | 34.3 +/- 14.1 tok/s | 39.8, 34.7, 28.4 | 0.62 s |
| code | 34.2 +/- 12.5 tok/s | 37.6, 28.4, 36.5 | 0.94 s |

The interval is wide: three runs do not separate a 5 tok/s change.

### The first requests are slow

Earlier runs on the same file, same card:

| Situation | Decode |
| --- | ---: |
| 9.3 GB RAM free after closing other programs, first request after start | 8.4 tok/s |
| same, second run | 15.1 tok/s |
| RAM nearly full (other programs open), first run | 2.0 tok/s (CPU experts 683 ms/token) |
| fresh server, 6 requests in a row | 2.5, 6.1, 14.8, 21.3, 24.5, 26.9 tok/s (wall, includes prefill) |

Speed rises about ten times over the first few requests. The likely causes are the file cache and the expert swaps into
VRAM; they were not separated. A number taken from the first requests is not the steady speed, so benchmark after
warm-up (abtest.py does this, `abtest.py:199`).

## Not tested

- The speed on `main` against this branch: `main` does not build `strata` on MSVC, so there is no same-day comparison.
- Long prompts, long context, the app, the `strata launch` path, vision, parallel requests.
- Any test target; the test phase (`tools/pancake/test_phase.sh`); the quality gate.
- A quiet-machine gate: `bench_win.py` only checks system CPU before each run (under 12%), it does not watch during a run.
- CUDA 13.x, other Turing cards, multi-GPU.
- Comparison with the 7900 XT numbers in `PANCAKE_STATUS.md`: different card, different backend.
