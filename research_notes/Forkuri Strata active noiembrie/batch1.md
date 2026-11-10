# Forkuri Strata active noiembrie - batch 1 (22 forks, own commits since 2026-11-01, state 2026-11-10)

Target: pancake-strata on RX 7900 XT (gfx1100, HIP) + Ryzen 7 5700G (AVX2, no WAITPKG/AVX-512), Flash-Next 125B IQ2_XS,
preset `pancake-flashnext-iq2_xs-fast-hip.json` (`--pcie-frac 0`, `--spec 4` MTP, `--kv-resident 32768`, `--grammar on`).
Tree checked: /home/dan/clone-of-d/src/Strata (HEAD 39f7a3da3). Status tags: NEW / ALREADY-HAVE / REJECTED-BEFORE / NOT-APPLICABLE.

## Q1: What in this batch could raise decode tok/s, prompt-read speed or cut start time on our box?

### Takeaway
Only three items are new and worth an A/B. (1) Two CPU-pool layouts from KhaledElzarw, `STRATA_POOL_DAG=1/2`. They cut
pool time 7-8% (author-measured) and give the same bits. The author saw no gain on a 9950X, where the pool is ~7% of a
window. On the 5700G the pool is a much larger share, so the gain may be larger here. (2) dardeaw's gfx11 "LDS-only"
barrier in the MMVQ and decode-attention kernels. It is exact and a small port. (3) KhaledElzarw's `STRATA_ACTQ_SKIP`,
also exact and tiny. Everything else in the batch is docs, GUI or packaging, needs CUDA, NVIDIA or several GPUs, is a
DeepSeek-only path, or is something we already have.

### Cited Findings

**F1 - KhaledElzarw/Strata: CPU expert pool as one DAG batch per layer (`STRATA_POOL_DAG=1`, `=2`) - NEW**
- Commits 691fa9d191 (DAG=1) and 287cd207ad (DAG=2). DAG=1 puts each expert's gate/up tasks and then its down tasks in one pool batch. The last gate/up task of an expert quantizes the intermediate and sets a per-expert ready flag, which that expert's down tasks wait on. This removes the second phase and its barrier. DAG=2 starts down tasks in the order gate/up finishes ("early finishers start their down rows first"). Output is bit-identical (replay hash, idcheck sweeps, pool stress) — [691fa9d](https://github.com/KhaledElzarw/Strata/commit/691fa9d191), [287cd20](https://github.com/KhaledElzarw/Strata/commit/287cd207ad)
- Author-measured (Ryzen 9 9950X, 2x64 GB DDR5-5600, RTX 3090, pool replay bench over 600 windows, median pool ms per window): base 5.197, DAG=1 4.834 (-7.0%), DAG=2 4.788 (-7.9%). Software prefetch `STRATA_POOL_PF=1` alone was +10% (worse) — [decision 0005](https://github.com/KhaledElzarw/Strata/blob/main/docs/decisions/0005-sm86-cpu-pool-bandwidth.md)
- Author-measured whole engine: DAG=1 on real turns 0.997 [0.980-1.014], neutral. The DAG=2 + SMALL_UNR=3 + ROUTE_BELL=2 bundle came out 1.011 [0.997-1.025], below the author's 2% bar, so it was not adopted. The pool computes only ~10% of routed experts there and the GPU waits 27-31 us a layer on it (1.3-1.5 ms of an 18.9 ms window). The author found the gap from 49-52 GB/s to the kernels' 57 GB/s "is the per-call structure (two phases, two barriers, short tasks), not the kernels" — [decision 0005](https://github.com/KhaledElzarw/Strata/blob/main/docs/decisions/0005-sm86-cpu-pool-bandwidth.md), [e5f0c4c](https://github.com/KhaledElzarw/Strata/commit/e5f0c4c19a)
- Our tree: no `STRATA_POOL_DAG` in src/. MOD_PLAN "Not adopted": "CPU pool settings sweep - defaults best". That sweep covered settings, not this restructure.
- Exactness: exact. Applicability: high. On the 5700G (8C, DDR4) about 56% of expert pairs run on the CPU, so the pool is a far larger share of the window than on the 9950X. A 7-8% cut in pool time should show up in decode tok/s. Port effort: M (pool.cpp +98/-36 and +50/-6, pool.hpp, a generate.cpp timing print; our pool.cpp carries local changes such as `--pool-affinity measure` and UMWAIT, so the merge is by hand).
- Suggested A/B: `STRATA_POOL_DAG=0/1/2`, together with `STRATA_DECODE_TIMING=1` to read pool time. Workloads: short chat (decode tok/s), 2K code edit, 11.7K prompt (should not change). Also try `STRATA_SMALL_UNR=3`, which we already have.

**F2 - dardeaw/strata-mxfp4-rdna: an LDS-only workgroup barrier on RDNA (gfx1030/gfx110x/gfx115x) in native MMVQ and QSA decode attention - NEW**
- Commit 580e251e69 adds `mmvq_sync_lds()` and `qda_sync_lds()`. They replace `__syncthreads()` where only shared memory is synchronized: `s_waitcnt lgkmcnt(0); s_barrier` on gfx1100, `s_wait_dscnt; s_barrier_signal/wait` on gfx12. The aim is to skip the L0 cache invalidate and the drain of in-flight global loads that `__syncthreads()` emits. It covers 9 MMVQ kernels (Q5_K, Q2_0, Q3_K, IQ4_XS, Q4_K, Q6_K, small, multi) and 2 sites in `attn_chunk_body`. The same commit turns on llama.cpp's v_perm IQ4_NL table lookup for all HIP (it was gfx906-only) and builds the B6 MMVQ bench and the ATTN_DAILY parity test on HIP. The B6 auto table is gfx12-only — [580e251](https://github.com/dardeaw/strata-mxfp4-rdna/commit/580e251e69)
- Author-measured: no per-change number found. The CHANGELOG gives only whole-config numbers on a gfx12 card — [CHANGELOG](https://github.com/dardeaw/strata-mxfp4-rdna/blob/main/CHANGELOG.md)
- Our tree: src/kernels/cuda/native_mmvq.cu has 15 `__syncthreads`, qsa_decode_attn.cu has 10, and neither has an LDS-only barrier. The IQ4_NL v_perm lookup is still `#if defined(STRATA_HIP_GFX906)` (native_mmvq.cu:527). `STRATA_B6_MMVQ_ROWS` exists, default 0.
- Exactness: exact (the arithmetic is the same; it is valid only where the barrier really guards LDS alone, so each site must be checked). Applicability: gfx1100 is named in the guard. The dense projections and decode attention run every token, so the gain is probably small (a few us per layer). Port effort: S (a copy of two inline helpers and the call sites; check each site with the parity tests `qsa_merge_daily_parity` and `mmvq_multi_parity`).
- Suggested A/B: build-time switch (e.g. `-DSTRATA_RDNA_LDS_BARRIER=ON`), or env-select two kernel variants. Workloads: short chat and 2K code edit, decode tok/s. Separately try `STRATA_B6_MMVQ_ROWS=1/2/4` on HIP (it exists in our tree; no gfx11 data anywhere).

**F3 - KhaledElzarw/Strata: `STRATA_ACTQ_SKIP=1`, skip the CPU activation quantize for tokens with no CPU expert - NEW**
- In `expert_pool_dispatch_multi`, a token of the verify window whose 10 entries are all in VRAM (none is -1) is not quantized for the CPU. It is guarded to one GPU with no helper or peer — [20f1fa1](https://github.com/KhaledElzarw/Strata/commit/20f1fa1a95)
- Author-measured: the "tier-5" bundle (DOORBELL_STORE, FUSE_HITS, ACTQ_SKIP, GR_UP_PL, GR_D87) gave +4.0% paired on real agent turns on an RTX 3090. No separate number for ACTQ_SKIP — [20f1fa1](https://github.com/KhaledElzarw/Strata/commit/20f1fa1a95)
- Our tree: `STRATA_DOORBELL_STORE` and `STRATA_FUSE_HITS` already exist (HIP path, upstream #649). `STRATA_ACTQ_SKIP` does not. GR_UP_PL/GR_D87/GR_RES are CUDA HC-read kernels (NOT-APPLICABLE).
- Exact. Applicability: with ~44% of pairs in VRAM, some window tokens have all 10 experts resident. The saving is one H-wide quantize per such token, likely <1%. Effort S (about 10 lines). A/B: `STRATA_ACTQ_SKIP=0/1`, short chat.

**Already have / rejected before / not applicable (speed items seen in this batch)**
- ElXreno 863505f003, "back the resident RAM copy with transparent huge pages". It puts the cudaHostAlloc'd complement on 2 MiB-aligned memory with MADV_HUGEPAGE — [863505f](https://github.com/ElXreno/Strata/commit/863505f003). ALREADY-HAVE for our path: the daily run uses the expert arena, and src/core/pinned.cu already maps it 2 MiB-aligned with MADV_HUGEPAGE (`STRATA_ARENA_THP`). Side note from our own log, not from the fork: pancake-flashnext.log shows 38 starts with "transparent huge pages NOT requested: only 6 GiB of free 2 MiB blocks for a 33 GiB arena". So THP is often skipped on this box because memory is fragmented. Reserving `vm.nr_hugepages` (16920 pages) at boot would make huge pages certain. That is our own A/B, not something to port.
- ElXreno e4cf3b474b / fb9a6525eb `STRATA_POOL_UMWAIT` — [e4cf3b4](https://github.com/ElXreno/Strata/commit/e4cf3b474b). REJECTED-BEFORE (MOD_PLAN ledger: "desktop dGPU gains nothing") and NOT-APPLICABLE (Zen 3 has no WAITPKG). Our pool.cpp already has the switch.
- ElXreno 76985dc813 `STRATA_SELECT_TWO_PASS`: ALREADY-HAVE (src/kernels/cuda/qsa_select.cu, prefill.cpp).
- ElXreno 5d94c4fbbd / 426a28ed9b: learned look-ahead weights, and a GPU look-ahead for the Foresight swap space. REJECTED-BEFORE (ledger: Foresight `STRATA_FS_SLOTS` 0.97x; history predictors closed).
- ElXreno dsv41 commits (102f1d51da AVX2 Q8_K quantizer, 679e4fb598 IQ1_S pdep, addc4fd774 DS4_LANE_FLOW, d4e55e3c40 / eb5dbfbacf huge pages, 5000ccc4a7, e4dbd17730 PDL on sm_90): NOT-APPLICABLE (DeepSeek V4.1 path, blocked here: 257 GB). DS4_LANE_FLOW is the same idea as F1, on the DeepSeek CPU lane.
- KhaledElzarw `STRATA_ROUTE_BELL=1|2` (top-10 + doorbell in one CUDA kernel), `STRATA_MMVQ_IL_3090`, `STRATA_EXP_MODE=9`, `STRATA_EXP9_DOWN`, `STRATA_GR_RES`, bit-exact HC down kernels: CUDA sm_86 kernels. NOT-APPLICABLE as code. ROUTE_BELL came out neutral even on the 3090 (bundle 1.011, CI includes 1) — [0af4328](https://github.com/KhaledElzarw/Strata/commit/0af4328ec8)
- KhaledElzarw 90e168f5e8 / 526670eefd `STRATA_DRAFT_PACK`: DFlash2 only. REJECTED-BEFORE (ledger: DFlash2 closed).
- carteakey 1bd03954ec `STRATA_QSA_FUSE_NR`: ALREADY-HAVE (src/prefill). 20c19b894e `STRATA_ADAPT_FETCH`: ALREADY-HAVE, and inert with `--pcie-frac 0`. 3733343d25 `ROUTE_TAIL_SKIP` off: ALREADY-HAVE (preset sets `STRATA_ROUTE_TAIL_SKIP=0`). 2e7e99de82 `STRATA_HC_Q8=1`: REJECTED-BEFORE (107.2 vs 111.7). All four are in the earlier report "Forkuri Strata câștig viteză".
- ElXreno and ImaginePilot disk conversation tier (af1997da11, dd61ec5384, e0ecd040d4, 3e825e570f): ALREADY-HAVE (MOD_PLAN item 39, src/core/conversation_disk_tier.cpp, `--conversation-disk-max-gib`).
- ElXreno 4f15d7092c restores a session file in two passes, so the K/V is never held in RAM (peak 2x16 MiB). Author-measured: a 198,830-token IQ3_S session, 3.05 GiB, restored in 3.4-3.5 s, next turn 99.99% reused — [4f15d70](https://github.com/ElXreno/Strata/commit/4f15d7092c). Partly NEW: our conversation_file streams in 16 MiB blocks, but no two-pass restore marker was found, so check whether a disk-tier restore still materializes the K/V. It matters only for very long sessions, because `--ram-keep-gib 12` with a 33 GiB arena leaves little headroom. Effort M. A/B: restore an 11.7K-token parked conversation after a restart and watch RSS and TTFT.
- MAHDI-AQ b6c64734aa `STRATA_DYNAMIC_PREFILL`: piped-read chunk size by active decode slots. NOT-APPLICABLE (single user; it does nothing with 0 active decodes).
- dardeaw MXFP4 experts and the FP8 LDS GEMM (69ffa74583, a62189cb21, 7bc75e4efb): gfx12 FP8. NOT-APPLICABLE (MOD_PLAN "Not adopted (GPU)": RDNA4 FP8 WMMA).
- Moeblack TP shards (17b716d173..adf8b2daa1): multi-GPU. NOT-APPLICABLE. Image history and continuation (adc08595aa..9a6f23d16c): dsv41 only.
- Volko61 `DS4_SPLIT_SPEED`, Maxwell builds: NOT-APPLICABLE.
- sheigl SYCL graph fixes: NOT-APPLICABLE.
- zapabob 13df015380 (pin pool for mapped-pack H2D), a149a578d6 (bounded file staging), 6fdb857bec (faster CRC scans of the expert pack), b54d1e4657 (FP16 KV): all in src/qwen35 (Qwen3.6 q36 engine). NOT-APPLICABLE to Flash-Next.

### Inferences
- F1 is the only item that goes after our main bottleneck, the CPU expert pool on DDR4. The author's 2% bar failed on a box where the pool is ~7% of the window. On ours, a ~7% cut in pool time could be worth several percent of decode. This is an estimate, not a measurement.
- F2 and F3 are cheap and exact. Bundle them in one build and A/B as a group, then one at a time if the group shows any gain.

### Gaps
- No author number for F2 (barrier) on any card, and none for F3 alone.
- Not checked whether the gfx1100 `__syncthreads()` in these kernels really emits `buffer_gl0_inv` / `s_waitcnt vmcnt(0)` (needs the ISA from `--save-temps`). If it does not, F2 gains nothing.

## Q2: Correctness fixes in this batch whose bug also exists in our tree

### Takeaway
Four fixes apply to our tree. The prompt-cache root fix (ElXreno a18d4668ec) affects the daily config. The kv_stream
overflow guard (kllmagn) is hardening for our `--kv-resident 32768` + 131K context. Two are latent, in features we
do not run.

### Cited Findings
- **ElXreno a18d4668ec, conversation-cache root at the first turn boundary past `--prompt-cache-root`.** In our generate.cpp:18274-18277 the loop breaks at the FIRST turn token and sets `root_at` only if that index is >= `prompt_cache_root` (default 2048). When the system prompt is shorter than 2048 tokens, no root checkpoint is saved at all. The fix starts the scan at `prompt_cache_root`, so the root lands on the first turn boundary past it — [a18d466](https://github.com/ElXreno/Strata/commit/a18d4668ec). Bug present. Effort S (two loops). Affects re-reads in short-system-prompt chats. A/B: a short-chat sequence with a <2K system prompt, before/after, compare the tokens re-read on turn 2+.
- **kllmagn e3056ec81b, reject incomplete streamed KV selections.** `resolve_kernel` writes `m.miss_block[atomicAdd(&s_nmiss,1)]` with no bound, and if the slots run out it silently sets `page_table[b] = -1` ("never happens with a legal n_slots"). The fix bounds the write and traps before attention — [e3056ec](https://github.com/kllmagn/Strata/commit/e3056ec81b). Same code in our src/kernels/cuda/kv_stream.cu:110 and :146. We stream KV (`--kv-resident 32768`, `--max-context 131072`). Bug present (an out-of-bounds write only if the slot math is ever wrong). Effort S. The fork author validated only CUDA ("HIP ... remain unvalidated").
- **zapabob 9e5f7b2fed, avoid absent KVA stream plans for short chunks.** Three sites in `Prefill::run_impl` read `seq_start[l + 1]` and call `give_back()` without the `stream_all` guard. `seq_start` is only sized on the streaming path — [9e5f7b2](https://github.com/zapabob/Strata/commit/9e5f7b2fed). Same three sites unguarded in our src/prefill/prefill.cpp:4238, :4382, :4513. Latent: reached only with `STRATA_KVA` (research, off) or the matching skip path. Effort S.
- **ElXreno a455cd0159 and 88b394e7c7, `--lookup-max-t` above kWindowMaxT.** `evict_window_graph` caps at kWindowMaxT, so 9..12-row graphs are never evicted. `prepare()` captures the window graph without the window's position, so under `STRATA_SELECT_LONG` it captures the wrong variant. `--batch-mtp` drafter sizing is also wrong — [a455cd0](https://github.com/ElXreno/Strata/commit/a455cd0159), [88b394e](https://github.com/ElXreno/Strata/commit/88b394e7c7). Same code in our verify.cpp:4808, verify.hpp:126, generate.cpp:6910. Latent for us: lookup-max-t defaults to 12 only with `--suffix-draft` > 0, and we run no `--batch-mtp`.
- Not applicable: jess-sol 53016163f3 (mask window cut) targets the ncphri token-mask code (`token_mask.hpp`, MKN replies), which our tree does not have; our `--grammar on` is a different implementation. Mnemosyne fe301cc2b5 / baf8c92237 (#5153) is a DFlash2-only boot with zero draft rows (we use MTP). carteakey 6ae87c7fd2 Q5_0 dot fix is ALREADY-HAVE (iq_kernels.cu:444-470 has `sumi - 16 * sumu`). zapabob 11ba67bec6 is native-cache scope only.

### Inferences
- a18d4668ec is the only fix that can change measured prompt reuse in daily use. Take it with the next prompt-cache work.

### Gaps
- Not measured how often opencode's system prompt falls under 2048 tokens. With the usual long tool prompt it is above, and then the bug does not bite.

## Q3: One-line verdict for every fork in the batch

### Takeaway
3 forks give NEW A/B items: KhaledElzarw (F1, F3), dardeaw (F2), ElXreno (root fix plus the session restore). kllmagn and zapabob give bug
fixes. The rest have nothing to take.

### Cited Findings
- zapabob/Strata (205): GUI, installer and docs for "Hakua", plus a Qwen3.6 (q36) engine. Nothing to take for speed. The prefill fix 9e5f7b2fed is a latent bug in our tree (Q2) — [repo](https://github.com/zapabob/Strata)
- TheHubUser4860938/Strata-Mnemosyne (71): memory/RAG tooling, curation, a DFlash2 snapshot fix. Nothing to take — [repo](https://github.com/TheHubUser4860938/Strata-Mnemosyne)
- ElXreno/Strata (34): the root fix a18d4668ec (Q2) and the two-pass session restore 4f15d7092c (partly NEW). THP, UMWAIT and the disk tier we already have or rejected. The dsv41 items do not apply. The cvec, SearXNG and frontend items are not speed — [repo](https://github.com/ElXreno/Strata)
- KhaledElzarw/Strata (20): F1 `STRATA_POOL_DAG` (NEW, M) and F3 `STRATA_ACTQ_SKIP` (NEW, S). The other items are CUDA sm_86 kernels, or DFlash2 — [repo](https://github.com/KhaledElzarw/Strata)
- Moeblack/Strata (15): multi-GPU GGUF tensor parallel and DeepSeek image/history cache. NOT-APPLICABLE — [repo](https://github.com/Moeblack/Strata)
- Volko61/Strata-For-Maxwell (13): Maxwell/sm_52 builds and a DeepSeek layer split by card speed. NOT-APPLICABLE — [repo](https://github.com/Volko61/Strata-For-Maxwell)
- carteakey/strata-lean (9): all items already have or were rejected (covered in the earlier report) — [repo](https://github.com/carteakey/strata-lean)
- dardeaw/strata-mxfp4-rdna (7): F2, the gfx11 LDS-only barrier plus the HIP IQ4_NL v_perm (NEW, S). MXFP4/FP8 is gfx12 only — [repo](https://github.com/dardeaw/strata-mxfp4-rdna)
- MAHDI-AQ/Strata-AGX (5): multi-slot prefill chunking, a cluster serving suite. Nothing to take — [repo](https://github.com/MAHDI-AQ/Strata-AGX)
- jess-sol/Strata (5): token masks for JSON/tool calls (ncphri #3800/#4222). Not our grammar code. Nothing to take — [repo](https://github.com/jess-sol/Strata)
- ImaginePilot/Strata (4): a disk KV tier. ALREADY-HAVE (item 39) — [repo](https://github.com/ImaginePilot/Strata)
- sheigl/Strata (4): SYCL graph fixes. Nothing to take — [repo](https://github.com/sheigl/Strata)
- SUP3RMASS1VE/Strata-orca (3): local customisations and an Apple core count. Nothing to take.
- sun19970908/Strata (3): `continue_final_message`, an Ada IQ4_W4A8 gate, vision unload. Nothing to take.
- SanghunYun/Strata (2): README for an SSD expert-supply research fork. Docs only, nothing to take.
- rodrigomatta/Strata (2): LFM2.5 support and README. Nothing to take.
- Develophel/Strata (1): Radeon VII (gfx906) scripts and configs, plus a one-line dflash2 change. Nothing to take.
- a00012025/Strata (1): opt-in tool loop guard in the server. Nothing to take for speed.
- fragtion/StrataPool (1): upstream sync. Nothing to take.
- kllmagn/Strata (1): the kv_stream overflow guard. The bug is present in our tree (Q2, S).
- zhgcao/Strata (1): HF_ENDPOINT for DFlash2 downloads. Nothing to take.

### Inferences
- Suggested order: F1 POOL_DAG=2 (biggest possible gain), then the F2+F3 exact bundle, then the a18d4668ec root fix, then the kv_stream guard.

### Gaps
- Commits of the docs/GUI-only forks were not opened, per the budget (triage was from commit subjects).
