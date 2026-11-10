# Memory / I/O / CPU-offload forks of Strata: techniques for the 7900 XT + 5700G box (state 2026-11-10)

Scope: the 10 repos assigned. Target box: RX 7900 XT 20 GB (gfx1100) + Ryzen 7 5700G (8C/16T, AVX2/FMA, DDR4 ~45 GB/s), 57 GB RAM, Flash-Next 125B IQ2_XS (~33 GiB experts in RAM arena, ~14 GiB cached in VRAM), pancake = /home/dan/clone-of-d/src/Strata branch port/rdna3-decode (HEAD f15ad2b51, 2026-11-10).
Tags: NEW = not in pancake; ALREADY-HAVE = switch/code exists in pancake (checked by grep of src/ include/ README.md MOD_PLAN.md); REJECTED-BEFORE = in MOD_PLAN "Not adopted" register or closed items. "Author:" = number measured by the fork author on their hardware, not by us.
Metadata (last push, license) from `gh api repos/<r>` on 2026-11-10. All 10 repos are MIT.

## Per-repo inventory: what each fork changes, author numbers, applicability, port effort, A/B

### Takeaway
None of the 10 forks touches the AVX2 CPU expert kernels for Flash-Next; the memory/I/O work is aimed at models bigger than RAM (SSD streaming) or at 12-16 GB NVIDIA cards, so most of it is either already in pancake (it came from upstream) or not applicable to a box where the whole 33 GiB arena lives in RAM. The few NEW items are small VRAM reclaims (~60-190 MiB more expert cache) and measurement data (momoe's route-prior quality study, strateu's 5700G pool bandwidth).

### Cited Findings

**1. Ribbon46/strata-q4-direct-io** (last push 2026-10-07, MIT, 0 stars, standalone research snapshot of upstream v0.1.40.1, CUDA only, RTX 5080, Windows)
- Purpose: run Unsloth UD-Q4_K_XL on a 16 GB GPU with less RAM than the model: unbuffered direct NVMe expert streaming, Windows file-map suspension, parallel refill, hot-RAM expert promotion; "nothing here has been tested on AMD" — [README](https://github.com/Ribbon46/strata-q4-direct-io)
- Switches (all default off): `STRATA_PREFILL_UNMAP=1` + `STRATA_REFILL_PAR=N` (author: 13k prefill 912 -> 1,719-1,790 tok/s); `STRATA_DECODE_UNMAP=1` (author, Codex A/B: 3.3 -> 9.3 tok/s decode, same binary); `STRATA_RAM_ADAPT_EVERY_WINDOWS/_SWAPS` hot-RAM admission of hottest decoded experts into the pinned RAM tier; `STRATA_LOOKAHEAD_DIRECT=1`; `STRATA_DIRECT_BATCH_BLOBS`, `STRATA_DIRECT_MERGE_GAP_KIB`; `STRATA_FETCH_POOL=1` ("no decode gain measured") — [LOCAL-PORT.md](https://github.com/Ribbon46/strata-q4-direct-io/blob/main/LOCAL-PORT.md)
- Author lookahead A/B (RTX 5080, 26 GiB experts, learned profile): control 17.30/17.25 tok/s; K4 18.12 (+4.7%, 70% hits, 30% wasted); K6 17.55 (+1.5%); K10 16.62 (-3.7%, 51% wasted). Author conclusion: "It works mechanically ... but moves the wait, it does not remove it"; host jobs ~65-67 ms of ~100 ms/window in every arm; "do not use it" — [LOCAL-PORT.md](https://github.com/Ribbon46/strata-q4-direct-io/blob/main/LOCAL-PORT.md)
- Diagnosis behind it: ~166 ms/decode window, ~115 ms host jobs = blocking reads of ~239 MB/token at 5.6-6.3 GB/s SSD — [LOCAL-PORT.md](https://github.com/Ribbon46/strata-q4-direct-io/blob/main/LOCAL-PORT.md)
- Pancake status: `STRATA_RAM_ADAPT*` and `LOOKAHEAD` symbols exist in src/core/expert_source.cpp / generate.cpp (ALREADY-HAVE, upstream code); direct O_DIRECT arena loader is pancake item 38 (ALREADY-PLANNED, covered by the 2026-11-09 survey via Project Maya). CPU-thread RouterLookahead and history predictors are in the Not-adopted register (REJECTED-BEFORE: "-13% decode (router GEMVs fight the pool for cores and DDR)").
- Applicability: low. All experts fit in our RAM, so there are no SSD reads during decode; the SSD-streaming switches do nothing for us. The only transferable lesson is negative: speculative prefetch competing for a bandwidth-bound resource does not help (matches our DDR4 situation).

**2. aflin/momoe ("More MoEs")** (last push 2026-11-07, MIT, 25 stars; standalone; Strata 0.2.9 engine + own ggml-based `momoe-cuda`/`momoe-metal` engines; NVIDIA + Apple only)
- Hardware: PC = Debian 12, RTX 4070 Ti 12 GB, Ryzen 9 5950X, 63 GB RAM; DGX Spark 128 GB — [README](https://github.com/aflin/momoe)
- Strata engine: three upstream defaults turned off after measurement: residency-biased routing (ROUTE_PRIOR), int8 prompt attention on sm_89, 0.2.9 GDN prompt kernels — [README](https://github.com/aflin/momoe)
- Own engines: experts not in the GPU pool are read from SSD past the page cache, copied from RAM, or computed by the CPU while the GPU computes its own; "a trial while the model runs picks the faster way" — [README](https://github.com/aflin/momoe)
- Author: CPU computing missed experts instead of PCIe copies: Ornith-122B 19.2 tok/s vs 11.4 with copies — [README](https://github.com/aflin/momoe)
- Latest branch `cpu-prefetch` (2026-11-07): "the CPU's missed experts read by all its threads before their dot products" (default on, `MOMOE_NO_PREREAD=1` off); tested alternatives `MOMOE_CPU_WILLNEED` (madvise) and `MOMOE_CPU_PREREAD`; merge message claims "models larger than the RAM +13-38%" — [commit 4a4a0dc](https://github.com/aflin/momoe/commit/4a4a0dc), [commit 07aa64d](https://github.com/aflin/momoe/commit/07aa64d)
- Author Flash-Next IQ3_XXS on the 4070 Ti PC: 65.5-76.7 tok/s (with MTP drafting) vs llama.cpp 23.1 (llama.cpp MoE cache 5 GiB) — [README](https://github.com/aflin/momoe)
- Expert cache hit rate in a Claude Code session on Ornith-35B: ~75-80% at both 40K and 100K tokens of conversation — [README](https://github.com/aflin/momoe)
- Helper tier (second GPU as second-level expert cache, each expert in exactly one place, evictions move down main GPU -> helper -> RAM) — [README](https://github.com/aflin/momoe)
- Pancake status: `STRATA_POOL_PREREAD` exists in src/kernels/cpu/pool.cpp and expert_source.cpp (ALREADY-HAVE; only matters when arena pages are not resident). CPU-vs-PCIe trial = pancake's PcieAuto / `--pcie-frac` (ALREADY-HAVE; MOD_PLAN 63 says PcieAuto picks `--pcie-frac 0` on the 7900 XT). Helper tier: NEW but not applicable (single dGPU; the Vega iGPU has no ROCm). ROUTE_PRIOR study: see the Unbiased section below.
- Applicability: momoe's speed claims are for non-AMD hardware; its value to us is the quality data on lossy routing switches.

**3. fragtion/StrataPool** (last push 2026-11-09, MIT, 4 stars, GitHub fork of Niko1221/Strata)
- Latest commits are "sync fork with upstream main", release-manifest tests, Intel Arc setup, README supporters — i.e. an upstream mirror with no engine changes in the last 10 commits — [commits](https://github.com/fragtion/StrataPool/commits/main)
- Applicability: none (no memory/CPU technique found). Gap: did not diff the whole tree; the description is upstream's verbatim.

**4. carteakey/strata-lean** (last push 2026-11-09, MIT, 0 stars, fork of upstream; RTX 4070 12 GB sm_89, 64 GB DDR5-5200, CachyOS)
- Author progression (sustained decode, UD-IQ2_XS): upstream v0.1.45 63.83 -> +sm_89 MMVQ interleave table 69.60 -> +`--kv-grow` 73.53 -> `STRATA_ADAPT_FETCH=1` 70.43 -> `STRATA_HC_Q8=1` 73.00 (2,949 expert slots on GPU, 72.4% hit rate); author caveat: a same-day repeat gave 67.88, run-to-run spread ~2 t/s — [README](https://github.com/carteakey/strata-lean)
- `STRATA_ADAPT_FETCH=1|2` (ported from upstream PR 2964): the PCIe share's fetched misses, which cross into VRAM anyway, are admitted straight into the slot of the layer's least-routed resident expert; author quote of upstream measurement on RTX PRO 5000 IQ3_S, 3,896-slot cache, `--pcie-frac 0.25`: =2 +10.2%, =1 +5.0%, token-forcing mismatch 2.73% vs 2.53% (PASS) — [commit 20c19b8](https://github.com/carteakey/strata-lean/commit/20c19b8)
- Defaults `STRATA_ROUTE_TAIL_SKIP` to OFF "to preserve exact mathematical routing" (upstream default 7; KL 0.003-0.006 drift in verify windows) — [README](https://github.com/carteakey/strata-lean), [commit 3733343](https://github.com/carteakey/strata-lean/commit/3733343)
- Notes `STRATA_PREFILL_CPU_SHARE` (CPU AVX2 FP32 rows of uncached experts during short-chunk prefill) costs mean KL ~0.004 (max 0.026) vs all-GPU prefill — [README](https://github.com/carteakey/strata-lean)
- Rejected: `--mtp-q4 head` (Q4_0 draft head): 69.60 -> 67.88 — [README](https://github.com/carteakey/strata-lean)
- Pancake status: ADAPT_FETCH, HC_Q8, ROUTE_TAIL_SKIP, kv-grow, PREFILL_CPU_SHARE all present (ALREADY-HAVE). ADAPT_FETCH is a no-op for us at `--pcie-frac 0` (there is no PCIe share to admit from) — inference from the commit text. sm_89 MMVQ tables are CUDA-only.

**5. LegendaryFire/strata-slim** (last push 2026-11-10, MIT, 4 stars; Docker packaging of upstream releases, CUDA + ROCm images)
- Patches (off in default images, on in `*-patched`): 0002 dense arena: "Pack the dense projection matrices into one VMM range per GPU instead of one `cudaMalloc` each, so they stop losing VRAM to 2 MiB rounding; the expert cache gets it" (-128 MiB per GPU on UD-Q4_K_XL; CUDA only; `STRATA_DENSE_ARENA=0` off); 0003: "With `--pcie-frac 0`, the verify window keeps no PCIe staging slots (about 61 MiB per GPU on UD-Q4_K_XL), which were never used" (`STRATA_PCIE_STAGING_KEEP=1` keeps them); 0001 skip BF16 HC weights on a layer split with HC_Q8 — [patches/README](https://github.com/LegendaryFire/strata-slim/blob/main/patches/README.md), [commit 06d1fb2](https://github.com/LegendaryFire/strata-slim/commit/06d1fb2)
- Documents `STRATA_STAGE_TRIM=1` (each GPU loads only its own layers' dense weights; multi-GPU) — [README](https://github.com/LegendaryFire/strata-slim)
- Pancake status: `STRATA_DENSE_ARENA` and `STRATA_PCIE_STAGING_KEEP` not found anywhere in pancake (NEW). Pancake verify.cpp:1202 allocates `staging_ = kStagingBlobs * max_blob` unconditionally (kStagingBlobs = 16, include/strata/core/verify.hpp:796) even though the PCIe launch is skipped when `pcie_off_` (verify.cpp:2671, 2687) — so the 0003 waste very likely exists on our `--pcie-frac 0` preset (local code read, not measured). STAGE_TRIM exists (multi-GPU only, irrelevant).
- Applicability: small but free. At ~1.4 MiB per IQ2_XS expert (33 GiB / 24,576 experts, our estimate), 61-190 MiB = roughly 40-130 extra cache slots on top of ~14 GiB (~10,000 slots): +0.4-1.3% slots, i.e. a fraction of a point of hit rate. Port effort: 0003 S (one condition around the staging take), 0002 M (HIP VMM path; pancake has src/core/vmm.cpp, need to check whether dense weights go through it on HIP).

**6. nsbradley88/talus-public** (last push 2026-11-10, MIT, 39 stars; Rust host over Strata/ggml-derived HIP kernels; AMD Strix Halo gfx1151, 128 GB unified memory)
- "Only gfx1151 is tuned" (per-architecture kernel tables keyed by GPU target); five models on one 128 GB machine; v0.1.7 adds 16-bit prompt-path streams and a 4-bit draft head on Flash-Next — [README](https://github.com/nsbradley88/talus-public)
- Applicability: none for CPU experts (unified memory: no CPU/RAM expert tier, no PCIe). gfx1151 tables would need a gfx1100 retune. The 4-bit draft head overlaps REJECTED-BEFORE (2-bit draft head in basalt rejected: draft step 2.5-3.2 ms of a 46-130 ms window) and strata-lean measured Q4 draft head slower on a 4070.

**7. mtnrbq/UnbiasedStrata** (last push 2026-11-10, MIT, 0 stars, fork of robertkarl/UnbiasedStrata which has 231 stars, last push 2026-11-06)
- What it is: upstream engine + llama.cpp tokenizer/chat template/HTTP in one C++ binary, no Python, no installer; NVIDIA only; engine 0.1.58 with "three files changed and one added (retrying reads from slow stores, stall diagnostics)" — [README](https://github.com/mtnrbq/UnbiasedStrata)
- Author: 107 tok/s (fork) vs 78 (upstream with `--pool-workers 15 --ple-io ram`) vs 20 (upstream as its installer wrote it: "127 CPU workers on the container's 32 vCPUs" and the n-gram table read from disk per token); single runs, gap "not explained yet" — [README](https://github.com/mtnrbq/UnbiasedStrata)
- Coder IQ1_M: `STRATA_ADAPT_OVERLAP=1` +2.2% decode at 64K (5 alternating pairs) — [README](https://github.com/mtnrbq/UnbiasedStrata)
- Parent PR #27 (robertkarl, 2026-11-06): `--stage-rows 1` streams decode misses through the pinned staging buffer to the GPU with an adaptive `--cpu-share` on the CPU; `STRATA_WORKER_SPINS=N` (default 20000 ~1 ms; 200 on a laptop); `STRATA_BLOCKING_SYNC=1`. Author (GTX 1060 Max-Q laptop, Qwen3.6-35B-A3B): 20.6 -> 29.3 tok/s; "Performance is WORSE on 3090 Ti but much better on 1060"; left opt-in for old hardware — [PR #27](https://github.com/robertkarl/UnbiasedStrata/pull/27)
- Pancake status: ADAPT_OVERLAP present (ALREADY-HAVE); pool spin/sleep knobs exist (`STRATA_POOL_SPIN_US`, `STRATA_POOL_MWAITX`, pool.cpp:794, 89) (ALREADY-HAVE); blocking sync vs spin = pancake item 76 `STRATA_DEVICE_SPIN` (ALREADY-HAVE, opt-in) and Not-adopted entry "host callbacks / blocking events instead of the spin" (REJECTED-BEFORE/parked). Streaming decode misses over PCIe = what PcieAuto already rejects on the 7900 XT (pcie-frac 0).
- Transferable lesson: thread count matters (worker oversubscription cost the author 4-5x); pancake's thread count is MOD_PLAN item 41.

**8. trustos/strata-locallane** (last push 2026-11-03, MIT, 0 stars; Mac/Metal line)
- Batched Metal decoding with run-time slots, prompt-cache parking, per-slot sampling, checkpoint split hints (`STRATA_SPLIT_HINT`, `STRATA_CKPT_HEAD`), elastic park ledger, opt-in 4-bit V for QSA layers (`STRATA_METAL_KV4=v`) — [README](https://github.com/trustos/strata-locallane), [commits](https://github.com/trustos/strata-locallane/commits/main)
- Applicability: no expert-memory/CPU technique; the checkpoint/parking ideas belong to pancake items 16/39/66 (agent-turn pipeline), outside this note's scope.

**9. gboddaer/strateu** (last push 2026-11-09, MIT, 0 stars; Kolibri-1 78B MoE engine, 384 experts top-6; CUDA + HIP on W7800 gfx1100 and R9700 gfx1201)
- Same CPU as ours: "Ryzen 7 5700G (8 physical cores / 16 threads, 1 NUMA node)" with R9700; Kolibri Q4_K_M decode 15.8 tok/s steady; pool streams "714.2 MB/tok ... 32.4 GB/s" — [AMD_R9700_PORT.md](https://github.com/gboddaer/strateu/blob/main/docs/AMD_R9700_PORT.md)
- Oversubscription on the 5700G: `--prefill`, 8 threads: decode 15.8 tok/s; 24 threads: 7.6 tok/s — [AMD_R9700_PORT.md](https://github.com/gboddaer/strateu/blob/main/docs/AMD_R9700_PORT.md)
- PR #16 on W7800 (gfx1100, Kolibri): exact multi-selection MMVQ (one launch for 2-8 Q4_K/Q6_K experts instead of separate calls), skip-zero-overflow, immutable zero buffer instead of per-layer memset: 12.34 -> 14.69 tok/s (+35.2% paired median, 5/5), complete logits byte-identical — [PR #16](https://github.com/gboddaer/strateu/pull/16)
- Rejected/blocked experiments recorded with evidence: quantization reuse, prefill admission, drain omission, promotion batching, CPU packing, host-copy switches, GPU hit dispatch — [PR #16](https://github.com/gboddaer/strateu/pull/16)
- Campaign report: promotion budget `RES_TOP` 8 -> 122 tok/s (-20%), 24 -> 152 (optimal) (prefill, T4 box) — [campaign report](https://github.com/gboddaer/strateu/blob/main/docs/strateu-campaign-report.md)
- Applicability: Kolibri-specific engine (src/kolibri), not the Flash-Next path. The 32.4 GB/s pool figure is an independent data point for DDR4 on a 5700G (~72% of ~45 GB/s) to compare with pancake's AVX2 pool. The multi-expert single-launch MMVQ is a GPU-side idea; pancake's Flash-Next hit path already uses grouped launches (verify.cpp `grouped(...)`), so likely ALREADY-HAVE in spirit.

**10. BlockYugaLabs/Strata** (last push 2026-10-07, MIT, 6 stars; standalone copy of upstream 0.1.40.x)
- Last commits: `STRATA_PREFILL_CPU_SHARE=auto` hands a small chunk's least-routed experts to the idle CPU pool; SYCL fixes; OrcaRouter IQ3_XXS gated family — [commits](https://github.com/BlockYugaLabs/Strata/commits/main)
- Pancake status: PREFILL_CPU_SHARE present (ALREADY-HAVE, upstream feature; MOD_PLAN 78 already questions its 0.6/4096 preset value for agent re-reads).

### Inferences
- The ecosystem's memory work splits into (a) SSD streaming for models > RAM (Ribbon46, momoe preread, Maya) and (b) VRAM reclaim on small cards (strata-slim, strata-lean). Our box is in neither regime for IQ2_XS: arena fully in RAM, 20 GB card. So the expected gain from this whole batch is ~0-1% decode, mostly from VRAM reclaim.
- Start time: no fork here offers anything beyond what the 2026-11-09 survey already lists (ARENA_REUSE #3509, Maya/item 38 O_DIRECT loader). Ribbon46's parallel refill (`STRATA_REFILL_PAR`) is a prefill-time refill of lent slots, not arena load.

### Gaps
- Did not run full `git diff` against upstream for each repo (several are standalone, not GitHub forks, with re-written history); conclusions rely on README, docs and recent commits.
- No fork published 7900 XT + Flash-Next numbers; none of the author numbers transfers directly.
- The 0003 staging waste in pancake is inferred from a code read (verify.cpp:1202), not measured with `rocm-smi`/engine log.

## Is there a CPU expert kernel faster than ours (AVX2 Q2/IQ2_XS GEMV, fused gate/up/down)?

### Takeaway
No. None of the 10 forks ships a new AVX2 IQ2_XS / Q2_0 expert kernel or a fused gate/up/down CPU kernel for Flash-Next; the only CPU-side work is pre-reading pages of non-resident experts (momoe) and thread/spin tuning, all already in pancake.

### Cited Findings
- momoe's CPU path is ggml's CPU backend in its own engine; its CPU change in Nov 2026 is pre-reading missed experts "by all its threads before their dot products", benefiting "models larger than the RAM" — [commit 4a4a0dc](https://github.com/aflin/momoe/commit/4a4a0dc)
- momoe concedes llama.cpp is ahead where "It computes some experts on the CPU from RAM while the GPU holds the rest" for Flint IQ3_XXS and LFM2.5-24B (not Flash-Next) — [README](https://github.com/aflin/momoe)
- strateu's CPU pool is "plain C++" (Kolibri, Q4_K) and streams 32.4 GB/s on a 5700G; "CPU packing" was tried and rejected — [AMD_R9700_PORT.md](https://github.com/gboddaer/strateu/blob/main/docs/AMD_R9700_PORT.md), [PR #16](https://github.com/gboddaer/strateu/pull/16)
- strata-lean describes upstream CPU prefill share as "AVX2 FP32" rows, unchanged from upstream — [README](https://github.com/carteakey/strata-lean)
- UnbiasedStrata's only CPU-side knobs are worker spin count and blocking sync — [PR #27](https://github.com/robertkarl/UnbiasedStrata/pull/27)

### Inferences
- Pancake already has q2_avx2_rows.inl / iq_avx2_rows.inl row kernels, PREREAD, MWAITX/UMWAIT parking, NUMA pool pick and an expdec SwiGLU fusion on the GPU side; its own register says the Q2_0 kernel runs at ~80% of DDR4 and LUT kernels are parked "Q2_0 kernel at ~80% of DDR4". strateu's 32.4 GB/s on the same CPU (different quant, ~72% of peak) is consistent with pancake being at or above the ecosystem's best on this CPU. Ceiling for any CPU kernel work: ~20% of the CPU-expert time, bounded by DDR4.
- Suggested A/B anyway (cheap, uses existing tools): `kq_cpu_bench`/`kq_window_bench` reporting GB/s for the IQ2_XS rows at 8 vs 16 threads vs pancake's preset, to confirm the "~80% of DDR4" claim holds for IQ2_XS (it was stated for Q2_0).

### Gaps
- No fork published a CPU micro-benchmark for IQ2_XS on AVX2; comparison with pancake is indirect.

## Any expert-cache / placement policy that improves GPU hit rate beyond LRU/frequency?

### Takeaway
Only one exact policy is new relative to plain frequency adaptation — admitting PCIe-fetched misses into the cache (ADAPT_FETCH) — and pancake already has it, with nothing to admit at `--pcie-frac 0`. Everything that raises hit rate a lot does so by changing the routing (ROUTE_PRIOR), which is lossy. The rest is capacity reclaim (strata-slim patches), which pancake does not yet have.

### Cited Findings
- ADAPT_FETCH (upstream PR 2964): admit a fetched miss "routed enough more than the layer's least-routed resident expert"; author-quoted +5.0% (=1) / +10.2% (=2) at `--pcie-frac 0.25` on RTX PRO 5000 — [strata-lean commit 20c19b8](https://github.com/carteakey/strata-lean/commit/20c19b8)
- strata-lean: 2,949 slots on a 12 GB card -> 72.4% hit rate (UD-IQ2_XS) — [README](https://github.com/carteakey/strata-lean)
- momoe: hit rate ~75-80% at 40K and 100K tokens in a Claude Code session (Ornith-35B); helper tier = strict exclusive two-level cache across two GPUs — [README](https://github.com/aflin/momoe)
- Ribbon46: hot-RAM admission promotes hottest decoded experts into the pinned RAM tier (relevant only when RAM < model) — [LOCAL-PORT.md](https://github.com/Ribbon46/strata-q4-direct-io/blob/main/LOCAL-PORT.md)
- strata-slim 0002/0003: -128 MiB and -61 MiB per GPU (UD-Q4_K_XL) returned to the expert cache — [patches/README](https://github.com/LegendaryFire/strata-slim/blob/main/patches/README.md)
- UnbiasedStrata parent PR #27: streamed decode windows must feed the recency list ("without it the miss rate rose to 80%") — [PR #27](https://github.com/robertkarl/UnbiasedStrata/pull/27)
- strateu: promotion budget has an optimum (RES_TOP 8: -20%, 24: optimal) — [campaign report](https://github.com/gboddaer/strateu/blob/main/docs/strateu-campaign-report.md)

### Inferences
- Pancake register already closed ClassLRU long-memory scoring ("replayed decay 0.7..0.986 within 0.5 points at 32 swaps") and history predictors; nothing in these forks beats that. Hit-rate gains for us come from (1) more slots (strata-slim style reclaim: +0.4-1.3% slots est.), (2) the learned profile (pancake 8.6, done, +6.7%), (3) swap budget tuning (`--adapt-swaps`, item 53).
- Suggested A/B for 0003-equivalent (S effort, exact): skip `staging_` allocation in verify.cpp when `pcie_off_`; compare logged cache slot count and decode tok/s, same-binary env toggle, 5 alternating pairs, token forcing must be identical.
- Suggested A/B for dense arena on HIP (M effort, exact): first log the sum of hipMalloc rounding losses for dense weights at load (if < 50 MiB, drop the idea).

### Gaps
- No fork measured a Belady/oracle bound for Flash-Next to tell how far frequency adaptation is from optimal; pancake's simulator (item 42/64 trace) would be needed.

## What does "UnbiasedStrata" change about routing, and does it affect quality or speed?

### Takeaway
Nothing about routing: "Unbiased" is branding ("Strata minus the bullshit") for a Python-free single-binary repackaging; it runs upstream engine 0.1.58 with three files changed (slow-store read retries, stall diagnostics). The routing-bias question is answered instead by momoe's measurement of upstream's residency-biased routing (STRATA_ROUTE_PRIOR): large speedup, clear quality loss, worst in tool use.

### Cited Findings
- UnbiasedStrata description and changes: "Strata minus the bullshit. clone, rsync to prod, compile and go"; engine = upstream 0.1.58 (tag v0.2.12) "with three files changed and one added (retrying reads from slow stores, stall diagnostics)"; server in C++, 0 lines of Python — [README](https://github.com/mtnrbq/UnbiasedStrata), [robertkarl/UnbiasedStrata](https://github.com/robertkarl/UnbiasedStrata)
- Its 107 vs 78 tok/s lead over tuned upstream is single-run and "not explained yet" (compile flags, n-gram table handling) — [README](https://github.com/mtnrbq/UnbiasedStrata)
- ROUTE_PRIOR mechanism: bonus λ added to the router score of every expert already in the GPU cache before re-picking top-10; "The expert it pushed out is never computed" — [momoe ROUTE_PRIOR.md](https://github.com/aflin/momoe/blob/main/docs/ROUTE_PRIOR.md)
- History: upstream 0.2.7 enabled λ=1.5 by default on NVIDIA cards <= 13 GiB; 0.2.8 (2026-10-29) reverted after repetition loops in long agent sessions (#4402) — [momoe ROUTE_PRIOR.md](https://github.com/aflin/momoe/blob/main/docs/ROUTE_PRIOR.md)
- Author speed (RTX 4070 Ti 12 GB, IQ3_XXS, 2,897 cached experts): chat 69.4 -> 106.0 (λ1.0, +53%) -> 123.6 (λ1.5, +78%); after 6,345-token prompt 58.0 -> 88.4 / 84.0 — [momoe ROUTE_PRIOR.md](https://github.com/aflin/momoe/blob/main/docs/ROUTE_PRIOR.md)
- Author quality (4,995 tokens through decode path vs prior off): λ1.0 same-top-token 87.2%, KL 0.144, PPL +5%, agent-transcript PPL +18%; λ1.5 82.5%, KL 0.251, +12% / +25%; λ2.0 74.9%, KL 0.455. For scale IQ3_XXS vs Q2_0 KL 0.094; GPU vs CPU rounding (1,500 vs 2,897 cached) KL 0.025 — [momoe ROUTE_PRIOR.md](https://github.com/aflin/momoe/blob/main/docs/ROUTE_PRIOR.md)
- Author graded tasks (140 tasks x 4, sampling on, thinking off): off 95.7% vs λ1.5 84.3%; tool use 95.0% -> 52.5%; sign test p = 5e-14; λ1.0 not task-checked — [momoe ROUTE_PRIOR.md](https://github.com/aflin/momoe/blob/main/docs/ROUTE_PRIOR.md)
- Route tail skip (`STRATA_ROUTE_TAIL_SKIP=7`) in the same table: decode +27-35%, KL 0.064, graded "95.6% against 91.9%", kept by momoe — [momoe ROUTE_PRIOR.md](https://github.com/aflin/momoe/blob/main/docs/ROUTE_PRIOR.md); strata-lean instead defaults it off for exactness — [commit 3733343](https://github.com/carteakey/strata-lean/commit/3733343)

### Inferences
- Pancake: ROUTE_PRIOR is ALREADY-HAVE as opt-in `turbo` (`STRATA_ROUTE_PRIOR=0.5`, README: turbo-hip 149-168 tok/s vs 135 fast) and "Router residency bias as a default" is REJECTED-BEFORE (parked behind turbo-hip quality gates, item 7). momoe's curve (KL 0.144 at λ1.0, 0.251 at 1.5, roughly growing with λ) suggests λ0.5 sits well below 0.14 but is not free; on a 20 GB card with ~40% of experts cached the bias bites less often than on a 12 GB card with ~12%.
- Suggested A/B for the item 7 gate (exact protocol to borrow): momoe's `tools/quant_quality.py` (KL / same-top-token on a 5K mixed set incl. an agent transcript) plus a 20-task tool-use check, turbo λ0.5 vs exact on the 7900 XT. Ship-gate idea: agent-transcript PPL within +3% and tool-use pass rate within run-to-run spread.
- The momoe tail-skip result ("95.6% against 91.9%") is ambiguous in the source about which arm is which; do not cite it as evidence that tail skip improves quality.

### Gaps
- No measurement of λ=0.5 anywhere in these repos; no AMD measurement of ROUTE_PRIOR.
- UnbiasedStrata's unexplained 37% lead over tuned upstream (single run) was not investigated by its author; cause unknown.
