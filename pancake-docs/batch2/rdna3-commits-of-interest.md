# Strata-rdna3 (nicejunjie) commits of interest to pancake

Fork base with our tree: `8183c6b67`. rdna3 `main` tip fetched: `f557b544`. Fork-only reach: **908 non-merge
commits** — rdna3 is a large 2-GPU production fork and the bulk is tensor-parallel / dual-PRO-V620 work pancake does
not run. Scope: single-GPU HIP gfx1100 + CUDA sm_86, the prompt/prefill path our fast-hip preset drives
(`STRATA_PF_FUSED/PF_F16/HC_UPMIX/PF_HCDOWN/PREFILL_CPU_SHARE/PREFILL_AUTO_MAX`), decode, the AVX2 CPU path, `serve/`.
Verified: none of the rdna3 opt-in switches below exist in our sources (unported), yet our preset already activates the
code paths they target.

Measured numbers below are rdna3's author's, on a 7900 XTX (and its own 2-GPU rigs), not on our box. These are
"same bits" kernel/geometry changes (bench checksums), so they are exact ports, gated only by our HIP build + the
rdna3 benches (`hip_pf_f16_bench`, `hip_prefill_hc_bench`) + a default-path token replay.

## Take-list, in priority order (single-GPU HIP, author-measured, bit-identical)

| # | rdna3 | what | author number / files | why it matters to us | effort |
|---|---|---|---|---|---|
| 1 | `1799c0a3` | **FP16 pair prefill expert kernel**: 3 blocks/WGP occupancy where LDS+VGPRs allow (`STRATA_PF_F16_OCC/_OCC_D/_OCC_AUTO=0`), IQ2/IQ3 codebooks as FP16 with nibble sign tables (`STRATA_NH_HGRID`), Q2_0 straight to FP16 (`STRATA_NH_Q2FAST`), down instance capped at 6 waves/SIMD (250->154 VGPRs, no scratch) | **prefill expert GEMM -6..-9%** (IQ3/IQ4, 1+2 GPU, 8K/32K); `src/prefill/moe_fused_iq.cu` +226, `tests/hip/pf_f16_bench.cpp` | our IQ2_XS/Q2_0 prompt expert GEMM; preset has `STRATA_PF_F16=2` | M (hipify + bench parity) |
| 2 | `3dbb4051` | **prompt hyper-connection kernels**: `gr_upmix` at 512 threads/256 tokens (512-thread bound -> 127 VGPR vs 237 at 256: 6 waves/SIMD -> 12), half the `w_up` staging (`STRATA_HC_UPMIX_NT=256` restores); `gr_write_norm_rs4` 16-byte accesses + LDS sum-of-squares in reference order (`STRATA_HC_WN4=0` restores) | **HC half-layer -7.4% (16K)..-9.1% (4K)**; upmix -14..-19%; `src/prefill/kernels.cu` +85, `tests/hip/prefill_hc_bench.cpp` | our preset runs `STRATA_HC_UPMIX=1` + `STRATA_PF_HCDOWN=1` | M |
| 3 | `1f4b77d7` | **colskip** in the same FP16 pair kernel: skip a wave's empty 16-token columns (`STRATA_NH_COLSKIP`, default on) | bit-identical; `src/prefill/moe_fused_iq.cu` +25 | rides #1 | S |
| 4 | `791cca4b` + `5906629d` | `STRATA_PREFILL_AUTO_MAX` / `--prefill auto:N` above 32768: keep the 8192 grid up to 49152 (was clamped to 32768); `STRATA_PREFILL_EQUAL_TAIL_MAX` bounds the one-big-chunk short-tail exception | fixes 49-57K prompts read as 49152+tail; geometry (rounding-level) | pairs with spec 32 (`PREFILL_AUTO_MAX=24576`); our `PREFILL_EQUAL_TAIL` came from Phase 17 | S |
| 5 | `f557b544` | **prefill CPU share retries the pool at every layer**, not only at the chunk's start | single-GPU prompt CPU-share scheduling | preset runs `STRATA_PREFILL_CPU_SHARE=0.6` | S-M |
| 6 | `88686782` + `877bf971` | prefill timing diagnostics (`STRATA_PREFILL_TIMING`: the CPU share's thread time + host wait) and drafter GPU profile (`STRATA_MTP_PROFILE=1`) | diagnostics; rdna3 `legacy/WIP cell-HIP.md` style | feeds our `wall clock`-based `gpu_profile.py` record captures / the turn-timing measurement plan | S (host/log) |
| 7 | `7df6538d` | slim writes split-GGUF shards without the routed experts (same names, updated split count) | tooling | lets us build a smaller prompt-only GGUF from our packs | S (tools) |

## Do NOT take (recorded so they do not re-enter)

- **All tensor-parallel / dual-PRO work** (`0f8e76ef`, `6987d7bb`, `2686e751`, `fa5c621d`, `29ab4948`, `5eb29332`,
  `0590d01c`, `d4ee9e6e`, `5077ba88`, `9017ce3f`, `7a36380d`, `551d01e6`, `58bfa2f4`, `b253c7ce`-class refill head, TP
  head split `4b96049f`, DP `da76c449`...): 2-GPU only — maps to phase-12 multi-GPU, not our single 7900 XT.
- `STRATA_MMQ_KQ` on by default (`fea8652e`): the same switch upstream added and reverted; treat with suspicion
  (`ffb94c578`/`88e663439`); not in our preset.
- `STRATA_PF_F16_QUAD` (256-token FP16 expert items): measured **a loss** by rdna3.
- HC read up-pass preload (`877bf971`) was REVERTED (`413c41df`): raised `rdna_gr_fused` 113->143-163 VGPRs, the grid
  barrier needs 168 co-resident blocks and hung beside the pipelined layer split; production layout 86.6 vs ~129 tok/s.
  A direct "do-not-port" and a fill-stall/occupancy data point for our probe P1.
- `STRATA_MEGAKERNEL` single-verifier (`b253c7ce`): correct on one GPU but **0.0 ms** — the window waits ~5.5 ms on CPU
  expert rows, GPU savings hide under it (matches our known "revisit after the CPU bucket shrinks").
- K-quant/top-k/CTAs `c4b18079` (multi-CTA long-context top-k): marginal for our context lengths; leave for later.
- rdna3 configs/README (`3e5dc36c`, `67ff6e7c`, `d5ae2dac`, `e9df7020`, `250e4625`, `b75ef924`, `e1dad2dc`,
  `bd34f079`): their 2-GPU production box; `STRATA_ENGINE_NICE=0` already matches our preset, `GPU_MAX_HW_QUEUES=8` is
  the opposite of our note (8 slowed prompts 10-20% on an XTX).
- `STRATA_ARENA_SHM` (`98d8c36f`): keeps the expert arena in shared memory across engine starts — a real restart-cost
  lever (our 28-65 s arena reads), but a large feature; keep parked behind the restart-cost measurements, not a quick port.

## How to port them (not started; needs the Linux HIP build)

These are exact kernel/geometry ports to `src/prefill/moe_fused_iq.cu`, `src/prefill/kernels.cu`, and the prompt
chunking in `src/program/generate.cpp`. Each lands behind its own `STRATA_*` opt-in (byte-identical default), is
classified in `src/core/conversation_disk_env.def`, and is confirmed by rdna3's benches (`hip_pf_f16_bench`,
`hip_prefill_hc_bench` checksums) plus our default-path token replay and `ctest -L cpu`. None of that runs on this
Windows box, so the ports follow after the probes/specs on the Linux RX 7900 XT, in the order above (start with #2 HC
kernels — smallest delta onto our active `STRATA_HC_UPMIX` path — then #1/#3 expert GEMM).
