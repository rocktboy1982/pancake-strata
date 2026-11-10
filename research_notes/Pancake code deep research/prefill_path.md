# Pancake prefill path: deep code study (2026-11-09)

Repo: /home/dan/clone-of-d/src/Strata, branch port/rdna3-decode. Read-only study: no builds, tests or GPU runs.
Line numbers are 1-based, as of this date. [M] = measured (source cited). [E] = estimate from reading the code.
Daily config studied: `configs/pancake/flashnext-iq2_xs-fast-hip.json` (7900 XT). Its prefill-relevant env:
PF_FUSED=1, PF_F16=2, PF_GEMM=1, PF_PAD=1, PF_GROUP_SCRATCH=1, PF_HC_REGION=1, PF_HCDOWN=1, HC_UPMIX=1,
GDN_CHUNKED=1, GDN_HEAD=1, GDN_PP=1, GDN_CONVL2=1, GDN_NOY=1, HIP_WMMA=1, SELECT_WMMA=1, PROMPT_ATTN_IMMA=1,
KV_PREFETCH=1, KV_HOST_DMA=1, PREFILL_RING=192, STAGER_RING=64, PREFILL_CPU_SHARE=0.6, PREFILL_CPU_SHARE_MAX=4096,
PREFILL_AUTO_MAX=32768, HIPBLASLT_TUNING=tools/hip/gfx1100-hipblaslt-100401.txt. `--prefill auto`, `--kv int8`.

Coordinator updates (2026-11-09): item 19 (tile-shared QSA) is CLOSED: measured overlap was 42-74%. BFLA end to end
on Qwen3.6 at 131K: 387 tok/s with it off, 772 dense, 1219 sparse [M, coordinator].

## 0. Headline findings

1. **There are two prefill paths, and agent turns take the slower one.** `stream_all_min()` (prefill.cpp:265) is
   4096 under this preset, because `arm_cpu_share` (2970-2998) sets `g_stream_min_share = cpu_share_max()` = 4096.
   So every chunk under 4096 tokens takes the **staged path**: MMQ, host grouping, one
   `cudaStreamSynchronize` per MoE layer (4928), CPU share 0.6. Agent-turn deltas are 1.5-3.2K tokens
   (2026-11-08-7900xt-final-suites.txt), so all of them run there. Chunks of 4096+ take the **streamed fused path**
   (int8 WMMA experts, issuer thread, no per-layer host sync). The staged path's settings on the 7900 XT (share 0.6,
   max 4096) were never measured on this card: no 7900 XT A/B for CPU_SHARE exists in bench/results or MOD_PLAN.
2. **The PLE (n-gram) read is likely a large, hidden serial cost on long cold prompts.** With `--prefill auto:32768`
   a 32K prompt is a single chunk. The gather for the whole chunk (16 rows a token, 4 KiB direct reads, then
   single-threaded decode to FP32) can only overlap the embedding and layer 0 (ple_land at layer 1, 3608-3664).
   On the 3060 the PLE was 10.6% of the GPU timeline (4.3 s on the host at 41.8K) [M, MOD_PLAN.md:486]. The compact
   upload (STRATA_PLE_COMPACT) is off, and it cannot work under `--ple-io direct` anyway: `gather_raw_iq4nl`
   (ngram.cpp:714) needs the mmap'ed table.
3. **GDN chunked WMMA on gfx11 launches only 48 workgroups on an 84-CU card.** `gdn_wmma_scan_kernel`
   (kernels.cu:2006, launched at 2166) runs one block per value head, each walking T/16 chunks one after another.
   Item 18 (chunk 64) cuts the serial steps by 4x. Splitting value columns across blocks (2-4 blocks per head, the
   norm moved out) fills the CUs. Both change bits, but stay within the same FP16-operand class.
4. **hipBLASLt descriptors are created and destroyed on every GEMM call** (gemm.cu:421-423, HipLtDescriptors), and
   the tuning table stops at T bucket 8192 although chunks go up to 32768 (gfx1100-hipblaslt-100401.txt).
5. **The instrumentation already exists**: STRATA_PREFILL_TIMING=1 (PfTimer, prefill.cpp:3092-3141, report at
   6440-6460) and STRATA_TRACE (per-part read ms in generate.cpp:18121). Nobody has run them on the 7900 XT. Section 4
   gives the exact run.

## 1. The pipeline for one chunk, as implemented

Entry: `Prefill::run` (prefill.cpp:7934-8021). The layer-major walk (7964) is off for a single stage. So
`run_impl` (3303-6526) runs, then `drain_pipeline`.

Serving side, generate.cpp: before the batched read, the prompt is cut at `{reread_to, root_at, message_at,
turn_at, n-1}` (18068). Each cut is its own `sp.run` with `lend()` before it (18111). A part that fits the decode
windows goes through `read_windows` instead. `refill()` runs at the end (18170). `message_at` is only added with
STRATA_CACHE_MESSAGE_BOUNDARY=1 (18039-18046, message_boundary.hpp: at least 8192 tokens of prefix, tail at most 1024).
So item 11 adds one more `run()` per request, with its fixed costs (section 2, N3).

### Per run (once)
- KVA decline check, PfTimer/PeTimer setup, KV-prefetch stream/events (3346-3363).
- Chunk choice (3389-3395): forced > plan_chunk (layer split only) > `m.T` (the buffers' size: auto up to 32768).
- MMQ row-table iota into the borrowed slots, every prompt (3415).
- PLE gather lambda (3427-3450): `ngram_rows` per token on the host, then `table->gather_batch` (direct reads plus
  decode to FP32) into the pinned `ple_emb_host[buf]`.

### Per chunk (loop at 3466)
1. Embeddings: token ids uploaded (`tok_up`), `embedding_gather_dev` or native gather, then `gr_broadcast` to the 4
   HC streams (3540-3596).
2. PLE: on the first chunk, `std::async` starts the gather (3601). Its result is waited on at layer 1 (`ple_land`,
   3608). Inside ple_land: H2D of T*N*4 bytes (FP32 rows; the compact form is not used), and for a non-last chunk a
   `cudaEventSynchronize` on the other buffer's upload (3643) before the next gather starts (3648).
3. QSA step records: filled on the host per token (3666), uploaded.
4. KV prefetch of the first QSA layer's host prefix on `kv_copy` (3677-3703), for p0 > 0 only.
5. Stream plan (`stream_all`, 3700; T >= stream_all_min, ring > 8): `seq` lists every non-resident expert of every
   layer in id order (3741-3790). The Stager starts host copies of unpinned blobs. An **issuer thread** is created per
   chunk (3902) and keeps `ring` (192) copies ahead on `m.copy`, waiting on each slot's `used` event.
   Below stream_all_min: no plan. The staged walk issues copies per layer, after routing.
6. Layer loop (4039). Per layer:
   - HC read (4096-4220): the q8/r8 to bf16 dequant views when present, then `gr_norm_rs` (skipped when the previous
     write fused the norm, F-2), then the down projection (PF_HCDOWN: down+inject in one WMMA kernel), `gr_silu`, the
     up projection (HC_UPMIX: up plus mix epilogue), inject, mix to `mixed / mixed_bf / mixed_h`.
   - GDN half (4221-4277): qkv and z projections (`native_proj`: dequant weight to FP16 scratch, then
     strata_pf_gemm_f16), alpha/beta (bf16), `gdn_gates`, `gdn_conv` (CONVL2), `gdn_recurrence` -> `gdn_rec_wmma`
     (chunk 16, gfx11, kernels.cu:2133), out projection.
   - QSA half (4279-4664): k, v, q projections, indexer k/q (bf16), norm+rope, KV stage (prefetched or inline),
     `kv_append` (with KV_HOST_DMA: one DMA to the host copy), q prep, indexer append batch, selection loop over
     `sel_batch` = 256 queries (`qsa_block_scores_tc` -> `qsa_block_topk`), `qsa_prompt_attn_batch(_gated)`, gate,
     output projection, then the prefetch of the next QSA layer's KV.
   - MoE half (4665-6146): router GEMM, `route` (top-10), shared expert (deferred after the host sync when
     stream_ahead and staged).
     * **Fused streamed path** (fused_l, 4777-4876): quantize_act(_native), GPU `fused::group`, then batches of at
       most min(kMaxBatch, ring/3) experts: `wait_issued`/`wait_copied` on the last blob of the batch,
       `experts_native` (GU+SwiGLU+down, int8 WMMA, dm16 when allowed), `used` events recorded, slots given back.
       No host sync.
     * **Staged path** (4877-6106): D2H of ids (mapped `grp_host` when present) -> **`cudaStreamSynchronize(m.cs)`
       (4928)** -> host counts per expert -> CPU-share choice (smallest-count non-resident experts, share 0.6) ->
       offsets / slot / src tables -> H2D (or mapped copy) -> **CPU thread via `std::async` per layer (5181)**
       (act-quant of the needed tokens, `run_split_multi_native` on the decode pool) -> GPU quantize_scatter, MMQ
       groups of MMQ_GROUP experts (gather native blobs to group buffers, GU product, SwiGLU, quantize H, down) while
       the staged copies stream through STAGE_GRP slots.
   - Combine (6107-6146): `cpu_fut.get()` (**the host blocks until the CPU experts finish**), H2D of the CPU rows into
     Dm's tail, then `moe_combine(16)`.
   - HC write (6174-6205): `gr_write_norm_rs` fuses the next half's norm.
7. End of chunk: join the issuer (6213). For the last stage with `on_chunk`: **`cudaStreamSynchronize` (6363)**, then
   serve's `on_chunk` (generate.cpp:12595): the MTP draft K/V for the chunk (`draft_kv`, E-9 batched; GPU work issued
   after the sync, so it is serial), the `PP` progress line, periodic and tail checkpoints (GDN/KV state copied out).
8. End of run: sync `m.cs`, `m.copy` and `kv_copy` (6390-6403). The timing report follows.

### Host/device sync points (normal path, no debug env)
| where | line | when | cost type |
|---|---|---|---|
| staged MoE routing readback | 4928 | every MoE layer of a chunk < 4096 tokens (48 per chunk) | GPU idle while the host groups and enqueues |
| CPU share join | 6110 (`cpu_fut.get`) | staged layers with a share | host blocks; the GPU drains if the CPU is slower |
| PLE buffer reuse | 3643 | non-last chunks | usually already done |
| PLE land | 3612 (`ple_next.get`) | layer 1 of each chunk | **GPU waits for the SSD read when the gather is longer than embed + layer 0** |
| chunk end | 6363 | every chunk with on_chunk (always under serve) | MTP draft K/V and checkpoints serialised after it |
| run end | 6390-6403 | every run (every cut segment) | fixed |
| HelpScope / KvDrain destructors | 3398-3405, 3366 | layer split / prefetch only | - |

### Buffers and chunk sizes
- Activations are carved from the borrowed expert-cache slots (carve, 2039-2194): R = T*D*4 (D = 10240: 1.3 GB at
  32K), xn16, mixed (T*N fp32/bf16/fp16), qkv, y/y_h, Kc/Vc/Qf/q, attn, logits, ids/w, Xq/GU/H/Dm (fused layout:
  sized for stream_all_min-1 MMQ rows, guarded at 4762), ple_emb (T*N*4), steps.
- Ring: 192 slots (env); STAGE 8 / STAGE_GRP 32 for staged chunks.
- Chunk: `--prefill auto` with AUTO_MAX 32768. A 32K cold prompt is one chunk, so with one chunk there is no
  cross-chunk overlap (PLE read-ahead, ring warm-up across chunks).

### Overlap that exists today
- Expert DMA vs compute: the issuer thread on `m.copy` (streamed) or the staged lookahead. Ring 192 > 1 layer of
  experts only partly (IQ2_XS: about 460 non-resident of 512 per layer when the cache is cold).
- KV prefix staging of the next QSA layer on `kv_copy` (KV_PREFETCH=1).
- PLE next-chunk gather on a thread (only for chunk >= 2).
- CPU experts vs GPU MMQ in the staged path (until `cpu_fut.get()` at combine).
- No overlap between layers: everything is one compute stream `m.cs`. No second compute stream (e.g. shared expert
  or HC inject in parallel), no graph capture.

## 2. Speed opportunities, ranked by expected gain

Gains are % of prefill wall time for the case named. All are [E] unless marked.

| # | Opportunity (function, file:line) | Change | Exact? | Expected gain | Risk | Plan item |
|---|---|---|---|---|---|---|
| N1 | **PLE gather overlap** (`run_impl` ple_gather 3427, ple_land 3608; `PleTable::gather_batch` ngram.cpp:625) | (a) start the gather at request admission in generate.cpp (the rows depend only on tokens), so it overlaps lend/restore/embedding; (b) for a single big chunk, split the gather into 2-4 slices and let the batched PLE block (sub-batches SB, 4012-4050) wait per slice; (c) decode on 2-4 threads; (d) a compact path for direct mode: the reader's raw rows straight into the pinned buffer and `ple_iq4nl_dequant` on the GPU (1,440 B instead of 10,240 B a token over PCIe) | exact (same rows, same dequant: check that the device dequant matches the host decode bitwise; a parity test exists for compact) | 3060: 10.6% of the timeline at 42K [M]. 7900 XT: 3-10% on cold 16-32K prompts [E] | low; direct-reader queue depth and ntfs3 O_DIRECT | 45 (I/O hygiene says "PLE order only after PLE wait is measured"): new item |
| N2 | **Staged-path settings on the 7900 XT** (cpu_share_max 312, arm_cpu_share 2998, preset env) | A/B on agent deltas (1.5, 3, 4K): CPU_SHARE 0.6/0 x SHARE_MAX 4096/1024 (1024 = the streamed fused path from 1024 tokens up) | share != 0 is not bit-identical to share 0 (CPU rows) | 0-15% on agent re-reads [E]; 4K reads at 1,757 tok/s vs 2,580 at 32K [M, final-suites] | none (env only) | S5 / 66; new |
| N3 | **Fixed costs per `run()` and per cut** (generate.cpp:18068-18160; run_impl per-chunk thread creation 3902, ple async 3601, per-layer `std::async` 5181) | reuse a persistent issuer thread and a CPU-share worker (as HandWorker does for the split); note that item 11's message_at adds a cut; measure STRATA_TRACE "read N tokens (batched) in X ms" per part | exact | 1-3% on short agent deltas [E] | low | 11/16 cost side |
| N4 | **GDN scan occupancy on gfx11** (`gdn_wmma_scan_kernel` kernels.cu:2006, `gdn_rec_wmma` 2133) | grid HV x 2 (or x 4) blocks of 4 (2) waves, each owning 64 (32) value columns; the RMS norm across 128 columns becomes a partial-sum write + a small norm pass (or a cluster-free 2-block exchange through global memory) | same FP16-operand class, other bits for the norm | GDN is ~12% of the timeline [M, 3060]; scan maybe 1.3-1.7x -> 2-5% of prefill [E] | medium; the prep's T/P reuse | new (pairs with 18) |
| 18 | **GDN chunk 64** (same kernels; GW_CK=16 at 1816) | 64-token chunks: blockwise 16x16 inverse of (I+A), 4x fewer serial steps, more WMMA per step | FP16-operand class, not bitwise | 5-11% of prefill per MOD_PLAN [E]; caution: the V100 chunk-64 attempt was 4-6x slower than the recurrence [M, bench/results/2026-10-02-gdn-chunk] - that was latency-bound staging with 4 warps, so the design must keep the state in registers as the gfx11 scan does | medium-high | 18 |
| N5 | **hipBLASLt per-call descriptors + tuning buckets** (gemm.cu:421-423 HipLtDescriptors::init per call; key 178; table tools/hip/gfx1100-hipblaslt-100401.txt) | cache op/layouts in HipLtCachedAlgo (add ldx to the key); add T buckets 1024/2048/3072 and 16384/32768 for the bf16 shapes that reach hipBLASLt (router 512x2560, alpha/beta 48, indexer 128/512, shared gate 1) | exact for the descriptor cache; a new solution id changes bits only if hipBLASLt picks another kernel | descriptors: host time only, matters on the staged path where the host is critical after each sync (~720 GEMM calls/chunk x ~10-30 us = 7-20 ms/chunk [E]); buckets: 0-3% on 16-32K [E] | low | 37 |
| N6 | **Staged-path host bubble** (4877-5300) | the ids D2H already goes to mapped memory; also pre-enqueue the next layer's HC read / norm before the host grouping (they do not depend on the routing tables, only on this layer's MoE output... they do, so only the shared expert and the CPU act-quant can move); cheaper: keep `grp_mapped` on, move the shared expert onto a second stream so it runs during the sync wait | exact (same kernels) | 0.5-2% on agent deltas [E] | low | 48 analogue for prefill |
| N7 | **CPU share join** (6110) | the GPU drains while the host waits for the CPU rows. Record an event and let the combine wait on a host-written flag (as decode's combine_cpu does), so the next layer's attention can be enqueued; or tune the share so the CPU finishes first | exact | 0-5% on staged chunks [E]; only if the CPU is the tail (unmeasured: the "cpu gate" debug line, STRATA_DBG_CPU_GATE=1, shows it) | medium | S1 analogue |
| N8 | **Chunk-end serialisation** (6363 + generate.cpp:12595) | the MTP draft K/V for the last chunk and the checkpoints run after a full sync. Enqueue draft_kv before the sync (it reads R on the same stream), and copy the checkpoint on a side stream with an event | exact | 1-3% of a short read [E] (the MTP draft pass on T rows) | low | 66 |
| 72/37 | **WMMA GEMM tile tuning** (wmma_gemm.cu:684-712 fixed 256x? / 128 tiles; moe_fused_iq occupancy "measured on gfx1151", 1758-1790) | coordinate-descent sweep of BM/BN/BK and blocks/WGP on gfx1100 for the dense shapes (qkv 10240x2560, q 12288x2560, out 2560x6144) and the fused experts | exact for tile changes that keep the k order (check: BK change alters order) | 2-6% [E] | low | 37, 72 |
| N9 | **Dense weight dequant every chunk on HIP** (gemm.cu native: the W16 cache is `#ifndef STRATA_USE_HIP`; bf16 with f16_io converts W per call, 1059-1069) | cache FP16 copies of the most-used dense weights when VRAM allows, or fuse the dequant into the GEMM's B load | exact (same FP16 values) | ~7 ms per chunk [E]: <1% at 32K, ~1% on 2K deltas | VRAM (~6 GB for all; not affordable with the expert cache) | D |
| N10 | **Expert ring vs layer size** (ring_slots 418; PREFILL_RING=192) | the 3060/5070 measurements put the fused optimum at 512-1024 slots (a whole layer); 192 came from gfx12 tables. A/B 192/384/512 on 16-32K | exact (same copies) | 0-8% [E] (5070: 384->512 +11-22% fused [M, comment at 330-339]) | VRAM loan vs cache | 37 |
| 19 | BFLA / tile-shared QSA | item 19 CLOSED (42-74% overlap). BFLA end-to-end measured on Qwen3.6 only (see top). For Flash-Next prompt attention the QSA attention is ~6.7% of the timeline [M, 3060], so the ceiling is small at 32K | lossy opt-in | <=3% at 32K [E] | - | 19 closed |
| 11/16 | conversation cache | not a kernel item: the best prefill is the one not run. Re-reads are ~18% of turn time [M, MOD_PLAN]. Prefill-side cost of each extra checkpoint: one extra cut (N3) + a GDN/KV state copy | exact | S5 group | - | 11, 16, 66 |

Order to do them: N2 (env only, no code) -> instrumentation run (section 4) -> N1 -> N5 descriptors -> N8 -> 18/N4
together (one GDN kernel rewrite) -> 37/72 sweep including N10 -> the rest.

## 3. Structure issues

- **`run_impl` is 3,224 lines** (3303-6526), with nested lambdas holding most of the state (issue_until, wait_copied,
  ple_land, compute, release_to). The MoE half alone is ~1,480 lines (4665-6146) with four expert paths inline
  (fused, MMQ groups, WMMA tiles, FP16 path) plus the peer GPU and the CPU share. `lm_run` (6863-7895, 1,032
  lines) repeats much of it for layer-major. Split into per-half functions with an explicit ChunkCtx struct.
- **Debug and research hooks inside the hot loop**: STRATA_DBG_FEAT/ROUTE_OVERRIDE (4632-4700, with
  `cudaStreamSynchronize` and blocking `cudaMemcpy`), SEL_OVERLAP, IDX_FP16_CHECK, QSA_DUMP, HC_UPMIX_CHECK,
  DUMP_MOE_INPUT, DBG_NAN. Each is a static getenv, so the run-time cost is small, but they make up a large share of
  the function. Move them to a hooks object.
- **getenv sprawl**: ~150 distinct STRATA_* switches are read in prefill.cpp alone (rough count by grep). Several
  are read with plain `std::getenv` inside the per-chunk/per-layer loop without a static: STRATA_TRACE (3474),
  STRATA_PREFILL_DUMP_R (6312), STRATA_DBG_NAN at end of run. These are cheap, but they are inconsistent.
- **Arch paths duplicated**: gfx11 vs gfx12 GDN scans (kernels.cu 1803-2168 vs gdn_wmma12.inc: the same algorithm, two
  fragment layouts), separate CUDA chunked/tc scans (gdn_tc.cuh, gdn_chunk.cu), gfx11/gfx12 fused experts in
  moe_fused_iq.cu, three dispatch functions with separate device caching (`gdn_rec_wmma` 2133, `gdn_fused_impl`
  2184, `fused_ring` per_dev 356). One arch-traits layer (fragment load/store, mma, barrier) would leave one kernel.
- **Defaults in comments**: many measured numbers live only in code comments (ring 384/1024, stream min 1024, share
  rules). The 7900 XT preset overrides several of them without its own measurement record.
- **Hygiene item already noted**: MMQ=OFF stubs (fixed 2026-11-09, MOD_PLAN.md:485).

## 4. Instrumentation to get a 7900 XT prefill time breakdown

Everything needed exists. Proposed run (one GPU session, after the warm start):
1. `STRATA_PREFILL_TIMING=1 STRATA_TRACE=1` added to the preset env (pancake.service or a copy of the json).
   Prompts: cold 32K (one chunk), cold 16K, cold 4K (exactly at the 4096 border: run 4000 and 4200), and the 6-turn
   agent session of strata_bench (staged chunks). Output lines: "strata prefill timing: ... GPU timeline ... wall
   ... host staging" with the 27 phases (kPfNames, 3101), plus the host line (setup / waiting for each chunk / after
   each chunk / PLE), plus STRATA_TRACE's "read N tokens (batched) in X ms" per cut and "prompt chunk" stamps.
2. Gaps the existing timer does not show, and where to add marks (all opt-in, behind the same env):
   - PLE: the time `ple_next.get()` blocks (3612) separately from the upload. `stats_.ms_ple` mixes both: add
     `ms_ple_wait` around the get. Also `PleTable::io_report` (ngram.cpp:736) for reads/IOPS/latency percentiles.
   - Staged path: host time from the sync's return (4930) to the first GPU launch after grouping, as a host stopwatch
     summed per chunk ("host grouping" on the GPU timeline is only the GPU's wait).
   - CPU share: the `cpu_fut.get()` wait time (6110) and `cpu_ms`. STRATA_DBG_CPU_GATE=1 gives per-layer walls
     only in auto mode, and the preset uses a fixed 0.6.
   - Issuer: `iss_ms` is reported in total; add the number of times `wait_copied` actually waited
     (`cudaEventQuery` before the wait) to tell PCIe-bound from compute-bound layers.
   - Chunk end: time inside `on_chunk` split into draft_kv / checkpoint (generate.cpp:12595-12680; `req_times`
     already has evict_ms).
   - hipBLASLt: `STRATA_HIPBLASLT_VERBOSE=1` lists the fallback shapes (untuned buckets).
3. A rocprofv3 `--kernel-trace --stats` pass over one 16K cold read (gpuprof2.sh as in
   bench/results/2026-11-09-7900xt-gpuprof) for the per-kernel shares (gdn_wmma_scan, experts_native, pf gemm,
   qsa_prompt_attn).
4. N2's A/B in the same session: CPU_SHARE=0 and SHARE_MAX=1024 against the preset on the agent session.

## 5. qwen35 engine (q36_engine.cu), brief

`process()` (2434-2480): chunks of `tmax`. Each chunk starts with `cudaStreamSynchronize(st)` (the pinned staging),
then token upload from pageable `toks + i`, `forward(T)`, and for MTP `forward_mtp` on the same stream. There is no
read-ahead across chunks (32 syncs in the file). The BFLA results (top) are the main lever there; a double-buffered
pinned token/pos staging would remove the per-chunk sync (exact, <1%).

## Sources
- prefill.cpp, gemm.cu, wmma_gemm.cu, kernels.cu, gdn_wmma12.inc, ngram.cpp, ple_reader.hpp/.cpp, generate.cpp,
  message_boundary.hpp, q36_engine.cu (paths under src/ and include/).
- MOD_PLAN.md sections "Development plan by expected gain" and "Combined improvements"; MOD_PLAN.md:486 (3060
  timing); docs/PANCAKE_FINDINGS.md section 1-2; bench/results/2026-11-08-7900xt-final-suites.txt,
  2026-11-08-7900xt-rounds/roundA.txt (11.7K read ~1,810 tok/s in every arm), 2026-11-09-7900xt-gpuprof/README.md,
  2026-10-02-gdn-chunk/RESULTS.md, 2026-10-11-gfx1100-qsa-scores/README.md.
