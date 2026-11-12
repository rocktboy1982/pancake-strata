# Batch 4, bfla-dense: BFLA's exact fused dense mode first, sparse strictly opt-in and lossy (2026-11-11)

Agent `bfla-dense`, branch `dev/p16b/bfla-dense` (worktree `Strata-p16b-bfla-dense`, from `dev/phase16-academic` 0dd7ef2e8).
MOD_PLAN "Remaining academics" order 9 (16.11 BFLA). No GPU was used and no model was loaded: HIP and CUDA were compiled
only, the tests ran on the CPU.

## What changed

| Commit | What |
| --- | --- |
| `ca3db05f5` | Merge of `feat/bfla-hip` (bf2c03e44). MOD_PLAN.md conflict: ours kept, the branch's plan edits dropped. Code as is. |
| `8b86420aa` | Switch split: `STRATA_BFLA_DENSE=1` (exact) and `STRATA_BFLA_SPARSE=1` (lossy); old `STRATA_BFLA` ignored with a note; parse moved to a pure `bfla_config_from()`; `bfla_tile_plan()` shared by the selection kernel and the tests; once-only `acted` line in the engine. |
| `b47726736` | `bfla_cpu_test` (new, `ctest -L cpu`); `bfla_test` (GPU) built only with `STRATA_BUILD_TESTS`. |
| `7f6e5798e`, `beb18a31f` | `bfla_recall.py`: `--mode dense|sparse`, 320 tokens by default, streamed KL, the gate, the `acted` check, `--reuse-ref`, the RAM cap kept for the engine; `test_bfla_recall.py` (new). |
| `cc4a0bd19` | Sparse tail rule from the prompt's end (paper Eq. 19 / section 3.5), not only inside the last chunk. |

Files: `include/strata/kernels/bfla.hpp`, `src/kernels/cuda/bfla.cu`, `src/kernels/cuda/bfla_test.cu`,
`src/kernels/bfla_cpu_test.cpp` (new), `src/qwen35/q36_engine.cu` (BFLA hunks only), `CMakeLists.txt` (BFLA lines only),
`tools/pancake/bfla_recall.py`, `tools/pancake/test_bfla_recall.py` (new). The merge also brings `docs/BFLA_RDNA3.md`
(not edited, see "For the lead") and the branch's `STRATA_SEL_OVERLAP` debug change in `src/prefill/prefill.cpp`
(off unless that variable is set).

## Switches

| Variable | Effect | Class |
| --- | --- | --- |
| none set | nothing runs: `bfla_ok` stays false, the GEMM path runs as before, no log line | default, unchanged |
| `STRATA_BFLA_DENSE=1` | qwen35 engine prompt chunks with `p0 + T >= STRATA_BFLA_MIN` (32768) use the fused gfx11 kernel over every causal block instead of the hipBLAS GEMMs + materialised softmax | exact (not bitwise) |
| `STRATA_BFLA_SPARSE=1` | BFLA's Ratio-K block selection, then the same kernel over the kept blocks; wins over DENSE if both are set; start-up line says LOSSY | lossy |
| `STRATA_BFLA` (old) | turns nothing on; prints that it was split in two | - |
| `STRATA_BFLA_MIN`, sparse only: `_ETA`, `_KMIN`, `_KMAX`, `_SINK`, `_LOCAL`, `_TAIL`; `STRATA_BFLA_QGLOBAL=1` | as before | - |

Log lines, in the format `tools/pancake/switch_report.py` reads (`^strata switch: (STRATA_[A-Z0-9_]+) (acted|will not act): (.*)$`):

- `strata switch: STRATA_BFLA_<MODE> acted: prompt attention of positions A..B by the fused kernel`, once, on the first
  chunk that took the fused kernel. An arm whose prompt stayed below `STRATA_BFLA_MIN` writes no line and is VOID.
- `strata switch: STRATA_BFLA_<MODE> will not act: needs head dim 256, ...`, at start-up, when the geometry, the GPU or the
  scratch does not fit (the success line stays the engine's own `strata-q36: STRATA_BFLA_<MODE> on`).
- `strata switch: STRATA_BFLA will not act: ignored (split in two): ...` for the old switch (`bfla.cu`).

The lines are printed with `fprintf` here because `include/strata/core/switch_report.hpp` (switch-cpp) is not on this branch;
after the merge they can move to `switch_report::acted()` / `will_not_act()` (same text, same once-only rule).
`bfla_recall.py`'s `acted()` still matches the substring `NAME acted` (tested on the new line and on a `will not act` line).
The engine computes `bfla_config()` at start-up in every run. With nothing set that only reads the environment: no output,
no allocation.

## Source check: BFLA (arXiv 2605.12193 v2, 2026-10-26, HTML body read) against the code

| Point | Paper | Code (`bfla.cu`) | Verdict |
| --- | --- | --- | --- |
| Ratio-K budget (Eq. 9) | K = min(V, max(Kmin, min(Kmax, ceil(eta V)))), k = ceil(K / b), V = last query of the tile + 1; eta 1/3, Kmin 2048, Kmax 32768, b 64 | the same (`bfla_budget_blocks`, now one host/device function) | match (tested at 7 points) |
| Tile | M = 128 token-head rows of one KV group, token-major: 128 / G tokens | the same | match |
| Mandatory blocks inside the quota, ties to the smaller id, sorted list | yes (Eq. 10, 18) | yes (bisection on order keys, ascending emission) | match |
| Head aggregation (Eq. 3) and sharing | q averaged over ALL query heads, k over ALL KV heads; "one block ranking, reused across KV-head groups" (3.3), "one shared score vector per tile, rather than a separate vector for each KV head" (3.6) | phi per (token, KV group) from that group's G heads, mu per KV head, one list **per KV head** | **mismatch, not documented on the branch.** The selection costs HKV x more (2x here, 10-49 ms of a 128K layer). Its quality effect is unknown: per group may be more specific. `docs/BFLA_RDNA3.md` 1.4 misreads the paper here. |
| Channel reduction (Eq. 4) | P(x)_a = (1/sqrt c) sum of c adjacent channels | adjacent pairs summed, no 1/sqrt 2 | equivalent: the next RMS norm removes the scale (except for eps) |
| Hadamard rotation (Eq. 5), FP8 key state with a block scale (Eq. 6) | yes | no rotation, FP32 state | equivalent: relu(phi . mu) does not change under a shared orthonormal rotation; FP32 is more precise than FP8. Documented. |
| eps of N_eps | not given | 1e-6 | - |
| Sink and local blocks | n_sink in tokens (ceil(n_sink / b) blocks); local band j >= floor(f / b_loc) - n_loc + 1 in b_loc units (n_loc counts the own block); no values for BFLA, "we follow the upstream reference configuration of FlashPrefill V2" = 2 sink blocks and a 4-block window of 128-token blocks | sink 2 blocks of 64 (128 tokens); local = 4 blocks of 64 before the own block (5 with it) | **unspecified in the paper.** Read as FPv2's token spans (256 sink tokens, a 512-token window) the code keeps half the sink and ~5/8 of the window. The code's `local` is the paper's n_loc - 1. A sensitivity arm is in the night commands. |
| Protected suffix (Eq. 19) | the last r_tail = 8 groups of M_tail = 128 rows (1024 / G tokens), aligned to M_tail / G-token groups from position 0, computed from the full prompt and not reset at chunk boundaries | before `cc4a0bd19`: only in the `forward()` that computes logits, so a last chunk shorter than the tail (or <= 8 rows, which never take the fused path) left the earlier positions sparse; now from the prompt's end in every chunk. The code protects whole selection tiles from `end - 1024 / G` (>= 128 tokens at G 8; the paper's aligned rule gives 113-128) | fixed; slightly more conservative than the paper |
| N <= Kmin: full attention | yes | yes per tile (k >= visible blocks), and the engine adds `STRATA_BFLA_MIN` (32768): below it the GEMM path runs | stricter |
| Chunked prefill | indexer state kept per layer and request, completed blocks reused | mu recomputed for the whole prefix every chunk (O(P), documented) | different cost, same result |
| Sparse attention (Eq. 11-12) | exact softmax over the retained cells, renormalised, no correction; BF16 Q/K/V, FA3-derived (FlashPrefill V2) kernel on H200 | the same softmax; FP16 operands (gfx11 WMMA f16, like the engine's GEMMs), FP32 accumulation, FP16 probabilities | match in method, other rounding |
| Error bound (Thm 4.1) | omitted mass delta <= (1 - beta) min{1, e^(2 eps_rank) tau(k_free)}; lossy by design | - | the sparse mode stays lossy |
| Models, hardware, claimed gain | Gemma4-E4B-IT, Qwen3.6-27B (dense), Qwen3.8-27B, Qwen3.8-Flash-Next; one H200, batch 1. TTFT vs FA3 (Qwen) or Triton (Gemma): Qwen3.6-27B 1.041x / 1.112x / 1.244x at 32K / 64K / 128K (RULER), 1.224x on LongBench-v2; "about 4x" is Gemma vs Triton; Flash-Next 1.273x at 128K where its QSA is faster (1.319x) | - | no MoE model and no Qwen3.6-35B-A3B in the paper. `docs/BFLA_RDNA3.md` 2 says the authors ran "Qwen3.5/3.6 MoE": not in the paper. |

What our numbers mean against the paper (recorded 2026-11-09, PANCAKE_STATUS P2b, noisy machine, not re-run here): 128K
prompt 386.9 tok/s off, 771.2 dense, 1,219.8 sparse; 64K 1,058.2 dense, 1,284.1 sparse. The paper's baselines are already
fused kernels (FA3), so the like-for-like ratio is sparse / dense = 1.58x at 128K and 1.21x at 64K (paper on a 27B:
1.24x / 1.11x). The "3.15x" is sparse against our materialised GEMM path, which no paper number compares to. The 2.0x
of dense over off is our own kernel, not a paper claim.

## The dense fused kernel's claims, checked against the code

- **Exact, every block**: with `ids == nullptr` `k_attn` walks blocks 0 .. floor(qb / 64) of each workgroup in 16-key
  steps, masks `cell >= P` and, on blocks reaching the first query, `cell - p0 > query`. `bfla_cpu_test` runs the same
  schedule on the host: every row attends exactly its causal cells, at G 4 / 8 / 16, p0 = 0 and after a prefix, T not a
  multiple of the tiles.
- **Online softmax**: exp2 with FP32 row max / sum, alpha rescale per step, 1 / l at the end. With FP32 probabilities the
  host model matches FP64 softmax attention to <= 1.01e-5 max abs (max |out| about 3), so the recurrence itself is exact.
- **Numerics class of the GEMM path** (FP16 operands, FP32 accumulation, FP16 probabilities): true. With the kernel's FP16
  probabilities the model is 1.9e-4 to 3.8e-4 max abs and 9e-5 to 1.3e-4 relative L2 from FP64. The GPU record (bfla_test,
  2026-11-09) gave 3.7e-5 to 1.3e-4 against FP64. One difference, static analysis only: the GEMM path stores *normalised*
  probabilities in FP16, and at 128K a flat row's ~1/131072 = 7.6e-6 is below FP16's normal range (6.1e-5), so it is
  subnormal (or flushed). The fused kernel stores p relative to the running row maximum (<= 1), so at long context it should
  be at least as accurate as the default path. Not bitwise either way (summation order).
- **The resources the kernel claims**: the gfx1100 object (device assembly) has 192 `v_wmma_f32_16x16x16_f16` and no trap,
  no scratch. `k_attn` uses 57,600 B LDS with Q in LDS and 23,808 B with `STRATA_BFLA_QGLOBAL=1` (as documented), 245-249
  VGPRs (Q in LDS) or 209-211 (Q global). Recorded bench: the Q-global variant was 11% faster at 32K-128K (515 / 2,066 /
  8,512 against 581 / 2,321 / 9,553 ms). The default stays Q in LDS until the night bench confirms that.
- **Sparse with every block == dense, bitwise**: the host model gives bitwise equal output for a list of every block, and a
  list with gaps attends exactly its listed cells. `bfla_test` checks the same on the device.

## Tests run

| Check | Result |
| --- | --- |
| `bfla_cpu_test` (new; build-cpu, CPU-only configure; and build-hip, HIP configure) via `ctest -L cpu -R bfla` | pass, 1.0 s (labelled cpu; HIP configure: bfla_cpu_test cpu, bfla_test gpu) |
| Mutations of its host model (mask off by one, mask condition, rescale dropped) | each fails the test (12-15 FAIL lines) |
| Mutations of `bfla_tile_plan` in `bfla.hpp` (restored after each): `qb >= full_from` -> `qb > full_from`; -> `qb + 1 >= full_from`; `k >= n_vis` and `free_q >= nc` both `>` | fail: the `full_from == qb` case, the `full_from == qb + 1` case, the budget 94-of-94 cases (2 / 2 / 5 FAIL lines). Changing only `k >= n_vis` to `>` is an equivalent mutant: `free_q >= nc` is the same condition (`k - (n_vis - nc) >= nc`) |
| `python -m pytest tools/pancake/test_bfla_recall.py` (new, 6 tests; the log line in it is the new `strata switch:` format, plus a `will not act` line) | pass |
| The engine's three line formats against `tools/pancake/switch_report.py` (read-only copy from `dev/phase16-academic`) and the regex above | all three match; `parse()` gives `acted` for `BFLA_DENSE`, `will not act` for `BFLA_SPARSE` and for `BFLA` |
| `bfla_recall.build_prompt` with the real Qwen3.6 tokenizer pack, 4,096 tokens | 4,096 ids, 6 / 6 needles in the decoded prompt |
| HIP gfx1100, the compilegate's recorded commands rewritten to this worktree (`-fsyntax-only` then objects): `bfla.cu`, `bfla_test.cu`, `q36_engine.cu`, `prefill.cpp` | 0 errors. bfla.cu 0 warnings. bfla_test.cu: 60 unchecked-`hipError_t` warnings (the branch's test code, the repo's usual class). q36_engine.cu: only the old sign-compare / unused `D`, `k_add`, `k_gate_rows`. prefill.cpp: only the old nodiscard ones. Objects: bfla.cu 3 s, bfla_test.cu 5 s, q36_engine.cu 13 s |
| After the fix-first changes (switch lines, `bfla_test.cu`'s `host_select` calling `bfla_tile_plan`, new plan cases): `bfla_cpu_test` via `ctest -L cpu -R bfla` in build-cpu (CPU configure); HIP gfx1100 `-fsyntax-only` of `bfla.cu`, `bfla_test.cu`, `q36_engine.cu` through this worktree's recorded build-hip commands | pass; 0 errors, no new warnings (bfla_test.cu still the 60 `hipError_t` ones). `bfla_test.cu` is compiled only, not run (GPU) |
| CUDA sm_86 (nvcc 13.1), same method: `bfla.cu` (stubs + config), `q36_engine.cu` | 0 errors; old warnings only |
| CMake HIP configure of this branch (gfx1100, tests on) | ok: 139 cpu / 129 gpu / 1 model tests |
| Not run: `bfla_test`, any engine run, the link of `strata` | GPU / RAM rules |

## Night GPU commands (for the lead; in order, ~45 min in total without the optional parts)

```bash
cd /home/dan/clone-of-d/src/Strata-p16b-bfla-dense
# build: build-hip is already configured here (HIP gfx1100, tests on, the compilegate's llama.cpp copy)
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 nice -n 19 \
  /home/dan/clone-of-d/src/Strata/.venv/bin/ninja -C build-hip -j2 strata bfla_test bfla_cpu_test
export LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib HIP_VISIBLE_DEVICES=0
unset STRATA_BFLA STRATA_BFLA_DENSE STRATA_BFLA_SPARSE
M=/home/dan/clone-of-d/models/qwen36-UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
TOK=/home/dan/clone-of-d/Strata-data/packs/qwen36-ud-q4_k_xl/tokenizer
BASE=/home/dan/clone-of-d/src/Strata-p16/build-hip/strata          # dev/phase16-academic build (no BFLA code)
R=bench/results/$(date +%F)-bfla; mkdir -p $R

# 1. kernel (no model, ~4 min): correctness, then one attention layer at 8K..128K in the engine's chunk sizes
build-hip/bfla_test | tee $R/bfla_test.txt                         # exit 0 = pass, 1 = fail, 77 = no gfx11
for n in 8192 16384 32768 65536 131072; do build-hip/bfla_test --bench $n 2048 | tee -a $R/bfla_bench-2048.txt; done
build-hip/bfla_test --bench 131072 4096 | tee $R/bfla_bench-4096.txt

# 2. default path byte-identical (model, ~3 min): base, this branch with no switch, base again, a 16K prompt (many
#    prompt chunks; ids inline: the base has no --tokens @FILE, and 16,384 ids fit one 128 KiB argument)
python3 -c "import sys; sys.path.insert(0,'tools/pancake'); import bfla_recall as B; from pathlib import Path; \
  ids,_=B.build_prompt(B.load_tok(Path('$TOK')), 16384, 6, 7); open('$R/p16k.ids','w').write(' '.join(map(str,ids)))"
for x in base:$BASE branch:build-hip/strata base2:$BASE; do
  tools/pancake/capped.sh ${x#*:} --model $M --max-context 20480 --tokens "$(cat $R/p16k.ids)" --n 16 \
    --dump-full-logits $R/def-${x%%:*}.logits > $R/def-${x%%:*}.out 2> $R/def-${x%%:*}.log; done
cmp $R/def-base.logits $R/def-base2.logits && cmp $R/def-base.logits $R/def-branch.logits && echo DEFAULT-IDENTICAL
grep -c STRATA_BFLA $R/def-branch.log                              # must print 0

# 3. quality gate on 320 tokens + one-shot TTFT per arm (model): dense first, sparse on the same reference
for L in 65536 131072; do
  tools/pancake/capped.sh python3 tools/pancake/bfla_recall.py --mode dense  --exe build-hip/strata --model $M \
    --tokenizer $TOK --len $L --new 320 --out $R/recall-$L --env HIP_VISIBLE_DEVICES=0
  tools/pancake/capped.sh python3 tools/pancake/bfla_recall.py --mode sparse --exe build-hip/strata --model $M \
    --tokenizer $TOK --len $L --new 320 --out $R/recall-$L --env HIP_VISIBLE_DEVICES=0 --reuse-ref
done
# results: $R/recall-$L/result-dense.json, result-sparse.json (gate_pass, gate, mean_kl, max_kl, top1_agree,
# n_tokens, recall_*, prompt_s_ref / _test / _forced, acted)

# 4. optional, sparse sensitivity (FlashPrefill V2's sink / window token spans), 64K, same reference
mkdir -p $R/recall-65536-fpv2 && cp $R/recall-65536/{prompt.ids,ref.ids,ref.logits,ref.json} $R/recall-65536-fpv2/
tools/pancake/capped.sh python3 tools/pancake/bfla_recall.py --mode sparse --exe build-hip/strata --model $M \
  --tokenizer $TOK --len 65536 --new 320 --out $R/recall-65536-fpv2 --env HIP_VISIBLE_DEVICES=0 --reuse-ref \
  --test-env STRATA_BFLA_SINK=4 --test-env STRATA_BFLA_LOCAL=8

# 5. optional, TTFT with 3 repetitions per arm (128K: ~31 min; 64K: use --max-context 69632 --bench-pp 65536)
for a in "" STRATA_BFLA_DENSE=1 STRATA_BFLA_SPARSE=1; do
  tools/pancake/capped.sh env $a build-hip/strata --model $M --max-context 135168 --bench-pp 131072 2>&1 | grep -E "bench|acted"; done
```

Time per step (estimates from the recorded prompt speeds): 1 about 4 min. 2 about 3 min. 3 at 128K about 12 min for dense
(off reference 340 s, two dense reads of about 170 s) plus about 4 min for sparse (two reads of about 107 s); at 64K
about 6 min. Disk: about 0.64 GB of logits for each mode directory.

Decision rules:
- Step 1 must exit 0. If the bench's Q-global row is again faster at 32K-128K, make it the kernel default (one line in
  `q_global()`, bfla_test already checks the two variants are bitwise equal).
- Step 2: if base equals base2 bitwise, branch must equal base bitwise. If base differs from base2, the engine is not
  deterministic run to run, and step 3's reference noise sets the floor. `$BASE` must be a build of dev/phase16-academic
  0dd7ef2e8 (the branch's q36_engine.cu differs from it only by the BFLA hunks and `--tokens @FILE`); from an older build a
  difference is inconclusive.
- Step 3, dense (`result-dense.json`): `gate_pass` at 64K and 128K, with mean KL expected near the noise of step 2.
  If it passes, the dense mode is ready for a preset arm (presets-specs) at its measured TTFT.
- Step 3, sparse: `gate_pass` needs mean KL <= 1e-3, top-1 >= 99% on >= 300 tokens and recall >= the reference.
  Pass or fail, sparse stays opt-in. If it fails, compare step 4 and check the per-group selection (mismatch row
  above) before tuning eta.

## Risk and next

- Default path: low. With no switch set the only new work is one environment read at start-up. Step 2 confirms it.
- Dense mode: the kernel ran on the card on 2026-11-09 (bfla_test pass, bench). This branch changes only the switch,
  `k_select`'s use of the shared plan (same arithmetic, sparse only) and the tail rule (sparse only). Its quality gate on
  300+ tokens has not run.
- Sparse mode: lossy. The 48-token numbers (KL 0.0014 / 0.0045) are above the gate. One provenance caveat (static
  analysis): the committed `bfla_recall.py` could not have produced them as is. Its forced logits file has `new + 1`
  rows, and `reshape(new, -1)` raises for the 248,320 vocab. So those numbers came from a modified copy; trust the
  re-run, not them.
- Per-KV-head selection (paper: shared across KV heads) and the sink/local spans are the two method deviations left.
  Neither changes dense mode.

## For the lead (outside my files)

- `docs/BFLA_RDNA3.md` (came with the merge, not edited, docs rule): section 5 still names `STRATA_BFLA=1/sparse/dense`
  (now `STRATA_BFLA_DENSE=1` / `STRATA_BFLA_SPARSE=1`). Section 1.4 says one selection per KV head is the paper's design
  (the paper shares one ranking across KV heads). Section 2 says the authors ran Qwen3.5/3.6 MoE (they did not). Section 5's
  "a start-up warning says the option is lossy" holds for SPARSE only.
- Plan text: the "3.15x" needs its baseline (materialised GEMM path); like-for-like with the paper is sparse / dense 1.58x
  (128K) / 1.21x (64K).
- **At merge, `tools/pancake/kernel_gate.py:544`** (not on this branch): the candidate's label `"STRATA_BFLA=dense"` must become
  `"STRATA_BFLA_DENSE=1"` (the old switch is ignored now).
- **`bench/promotion/switch-exempt.json`** (not on this branch; on `dev/phase16-academic` it is `{"note", "exempt": {"no
  strata switch line yet": [111 names, sorted]}}` and has no BFLA name): `STRATA_BFLA_DENSE` and `STRATA_BFLA_SPARSE` log
  `acted`, so they need no entry. Their knobs print nothing, so an arm that sets one of them is VOID for that key in
  `switch_report.py` unless it is listed. Add to the list (alphabetical place: after `STRATA_BANDIT_LOG`):
  `STRATA_BFLA_ETA`, `STRATA_BFLA_KMAX`, `STRATA_BFLA_KMIN`, `STRATA_BFLA_LOCAL`, `STRATA_BFLA_MIN`, `STRATA_BFLA_QGLOBAL`,
  `STRATA_BFLA_SINK`, `STRATA_BFLA_TAIL` (the night commands above set `_MIN`, `_SINK`, `_LOCAL` through `--test-env`), and
  `STRATA_SEL_OVERLAP` if an arm ever sets it (debug output of `prefill.cpp`, no switch line). A preset arm that sets only
  `STRATA_BFLA_DENSE=1` needs nothing.
- What `bfla_cpu_test` covers: the switches, `bfla_tile_plan` and a host model of the kernel's arithmetic (schedule, mask,
  online softmax). It does not compile `bfla.cu`, so a mutation of the real kernel (a mask or rescale slip in `k_attn`,
  a changed budget in `k_select`) is caught only by `bfla_test` on the GPU (night step 1). `bfla_test`'s `host_select` now
  takes the plan from `bfla_tile_plan` (same function as `k_select`), so the plan itself is tested only by the hand-computed
  cases in `bfla_cpu_test`; `host_select` still checks the scores, ranking and tie rule independently.
- The `switch-cpp` helper (`include/strata/core/switch_report.hpp`) can replace the three `fprintf` calls after the merge.
