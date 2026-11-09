# Avalon baseline (local build vs upstream llama.cpp vs freriadoc/Avalon main), as of 2026-11-08

> **CRITICAL STATE FINDINGS (read first)**
> 1. **The model directory is gone.** `/home/dan/clone-of-d/Avalon-models/` now holds only `avalon-chat-template.jinja`, `download.log`, `verify.py`; the folder `Qwen3.8-Flash-Next-Avalon-GGUF` (33 GGUF parts + `MTP/` + `moe-cache-ranking.bin`) does not exist (dir mtime 2026-11-08 00:46). A `find` over `/home/dan/clone-of-d` (depth 4) found no `*IQ3KS*.gguf`, no `mtp-Qwen3.8*`, no `moe-cache-ranking.bin`. Partition free space: 78 GB of 758 GB. Re-download is ~89 GiB. — local `ls`/`find`, [download.log](file:///home/dan/clone-of-d/Avalon-models/download.log)
> 2. **avalon.service is inactive and disabled.** journal: restart counter reached 1230 with `start-pre operation timed out` (the ExecStartPre loop waits for part 00033), then "Stopped" at 2026-11-08 03:01. The tail of `avalon-service.log` shows repeated `ROCm error: invalid device function` aborts (consistent with the RTX 3060 period, 2026-11-07). — `journalctl --user -u avalon.service`, [avalon-service.log](file:///home/dan/clone-of-d/Avalon/avalon-service.log)
> 3. **The installed binary is c24c03d, not a91372e.** `build/bin/*` dated 2026-11-03 01:58-02:00; `build-update.log` says `ggml commit: c24c03d`; the a91372e build is kept in `build-bin-a91372e/`. — [build-update.log](file:///home/dan/clone-of-d/Avalon/build-update.log), [build-configure.log](file:///home/dan/clone-of-d/Avalon/build-configure.log)

## Q1. Upstream merge base and how far behind upstream master

### Takeaway
Avalon does not merge upstream; it **cherry-picks/rebases upstream commits linearly** onto its own history. The newest upstream commit in our build is the one for PR #31434 (`opencl: add fwht`), upstream SHA **0ec4027a2a01** (2026-10-31 05:13 UTC). Upstream master is **191 commits ahead** of that (0 behind) as of 2026-11-08 ~16:00 UTC (latest upstream tag b12211). Avalon also **left out** a set of upstream features that existed before the base (its own MoE cache, realtime server, several models).

### Cited Findings
- Local clone is shallow (`.git/shallow`), only 5 commits visible: a91372e (2026-11-01) → c0ff9ec → 2c346ba → d66513f → c24c03d (2026-11-02), all by Frédéric BELVAL. — local `git log`, [freriadoc/Avalon](https://github.com/freriadoc/Avalon)
- Upstream commits appear in Avalon's GitHub history as single-parent commits with new SHAs dated 2026-10-31 16:16 (e.g. Avalon `d0e48436f9` "opencl: add fwht (#31434)", `86020a29eb` "llama : one SET_ROWS case in the loader's op check (ours and #30912 both added it)"); the Avalon SHA `d5987d9df4` does not exist upstream (HTTP 422), confirming rebasing rather than merging. — [Avalon commits](https://github.com/freriadoc/Avalon/commits/main), GitHub API
- PR #31434 merged 2026-10-31T05:13Z as `0ec4027a2a0183de87c007069adddc06e665f7de`; `compare 0ec4027...master` = ahead_by 191, behind_by 0. — [ggml-org/llama.cpp PR #31434](https://github.com/ggml-org/llama.cpp/pull/31434), [compare](https://github.com/ggml-org/llama.cpp/compare/0ec4027a2a0183de87c007069adddc06e665f7de...master)
- Avalon README: "Base: llama.cpp master up to Oct 31, 2026 (merged that day, a few commits left out for speed or overlap)". Example of a left-out commit right below the base: #31380 "chat : strip the think block in the qwen3-coder with reasoning_format none" (upstream 1052f7e1d6) is not in Avalon's sequence. — [Avalon README](https://github.com/freriadoc/Avalon/blob/main/README.md)
- Base fetched into a separate ref only (`refs/upstream/base-0ec4027`, depth 1; no checkout). `git diff --shortstat base a91372e`: 546 files, +40,330 / −37,529. — local git
- Files that exist in upstream base but are **absent in Avalon** (excluding .github/docs): upstream's own `src/llama-moe-cache.cpp/.h`; `ggml-cuda/gated-delta-net-mma.cu/.cuh` (upstream MMA GDN); `ggml-cuda/top-k-mask.cu/.cuh`; server `server-realtime.*`, `server-connect.*`, `server-models-sched.*`, `tools/connect`; models deepseek41, flint, kolibri1, lfm3, limite, mistral-large4, muse-spark, vllm-decision2 (+ their conversion scripts, LFM3/muse-spark templates and mtmd encoders lfm3a/lfm3vl/nemotron-omni/soprano/voxtral-rt); tests llama-context suite, test-fp8, test-backend-tensor, test-speculative-rejection-sampling. — local `git diff --name-status --diff-filter=D`
- Version strings in our build: ggml 0.28.0, llama.cpp 0.8.0-dev (upstream bumped to ggml 0.29.0 / llama.cpp 0.9.0 on 2026-11-04, cca14f64fe / cd959bae99). — [build-update.log](file:///home/dan/clone-of-d/Avalon/build-update.log), [upstream commits](https://github.com/ggml-org/llama.cpp/commits/master)

#### Notable upstream commits since the base (subset of the 191, filtered of opencl/sycl/hexagon/metal/vulkan/ui/ci/mtmd)
MoE cache / MoE kernels (upstream now has its own expert-cache line of work):
- c4fb579695 (11-06) moe-cache: LRFU/MQ-like hybrid "ClassLRU" replaces SLRU; author reports +6% decode on QFN (Qwen3.8-Flash-Next) and +7% on GLM with warm cache, 8% fewer uploads. — [PR #31233](https://github.com/ggml-org/llama.cpp/pull/31233)
- e95426f169 (11-07) llama_moe_cache: run missed experts on CPU (`--moe-cache-cpu-share`), `scripts/moe-cache-tune.py` tuner for `--moe-cache-mib`/cpu-share. — [PR #30966](https://github.com/ggml-org/llama.cpp/pull/30966)
- cf34d31133 (11-08) moe-cache: CPU part no longer waits for cached experts (+7-9% TG on Gemma 4). — [PR #31888](https://github.com/ggml-org/llama.cpp/pull/31888)
- a2b301831d (11-08) speculative: do not apply MoE cache to the draft model. — [PR #31497](https://github.com/ggml-org/llama.cpp/pull/31497)
- 69ee2ecc01 (11-05) CUDA: linearize MoE data layout for MMQ/MMF (tile assignment precomputed in mmid.cu). — [PR #31047](https://github.com/ggml-org/llama.cpp/pull/31047)
- 10d4846a57 sanitize post-bias NaNs in MoE selection (#31005); 016c1d914c highest compiled arch in get_mmvq_mmid_max_batch (#31679). — [upstream log](https://github.com/ggml-org/llama.cpp/commits/master)

Qwen3.8-Flash-Next (qwen4exp) / MTP / speculative:
- 749db7f034 (11-03) qwen4exp: write conv state tail without intermediate cont, "Free 2% on Qwen3.8-Flash-Next generation". — [PR #31443](https://github.com/ggml-org/llama.cpp/pull/31443)
- 040ce199a1 (11-05) graph: MTP conv snapshots of every recurrent model (incl. qwen4exp) in one set_rows (replaces up to 12 copies/layer). — [PR #31660](https://github.com/ggml-org/llama.cpp/pull/31660)
- b74298c109 delta_net_base: refactor conv state (#30878); 0f35d6d7b0 (11-03) spec: save/restore MTP hidden state across checkpoints. — [PR #30141](https://github.com/ggml-org/llama.cpp/pull/30141)
- 2c192444c9 (11-05) fit: account for skipped NextN layers (fixes "--fit on leaves one layer on CPU, performance halves"). — [PR #31569](https://github.com/ggml-org/llama.cpp/pull/31569)
- 2cbca389f5 dflash: device copies of hidden features (#31506).

Server / runtime / CUDA-HIP:
- e152332a15 (11-05) llama-server: reference-counted context checkpoints instead of deep copies. — [PR #30137](https://github.com/ggml-org/llama.cpp/pull/30137)
- 00f113eceb (11-03) CUDA: prevent premature eviction of CUDA graphs about to be used. — [PR #30662](https://github.com/ggml-org/llama.cpp/pull/30662)
- 740251a920 (11-03) CUDA: gather batched tensor reads into mapped host memory (state save of fragmented KV). — [PR #31391](https://github.com/ggml-org/llama.cpp/pull/31391)
- 1b6935b227 flash_attn_ext_rows for multi-seq unified KV (#29510); e56f006ed6 on-device state copy sizes for quantized KV (#30541); 6023692b1e kv shifting fix + per-model build_rope (#31683); ee17f293b6 kv-cache never returns 0 kv_rows (#31570); 0956684529 server: visit parent slots before child slots (#30779); f3586e66bc server: cap shared decision prefixes (#30710); 372dc5ca15 sanitize invalid UTF-8 in PEG tool-call args (#31861); 4402e65401 chat constraint after gpt-oss final channel (#31556); 59695669f1 jinja map('filter') (#24033); 9982004913 SSM_SCAN d_state=64 fast path (#31818); 12ab597986 CPU backend F32 src1 (#31854); 71efc559a0 NCCL multi-node TP (#28967); 15e50990bb/5d7d4aec16 lightning indexer MFMA (CDNA2) and bank-conflict fix. — [upstream log](https://github.com/ggml-org/llama.cpp/commits/master)

### Inferences
- Because Avalon rebases upstream and deliberately dropped upstream's `llama-moe-cache` (replacing it with its own buun-derived `ggml-cuda/moe-cache.cu`), upstream MoE-cache PRs (#31233, #30966, #31888) are **not drop-in**; they target a different implementation and would need to be judged as ideas (e.g. ClassLRU admission vs Avalon's LFU) rather than merged.
- The qwen4exp/MTP upstream fixes (#31443, #31660, #30141, #31569) touch files Avalon heavily rewrote (`src/models/qwen4exp.cpp` +1,088 lines diff, `src/llama-graph.cpp`), so they are likely conflict-prone; #31443 claims +2% on exactly our model and is worth checking whether Avalon's fused kernels already remove that copy.
- Avalon's server is *smaller* than upstream (tools/server: +1,442/−6,377), so upstream server features (realtime, connect, model scheduling router) are intentionally absent.

### Gaps
- Exact list of upstream commits before the base that Avalon skipped could not be enumerated without unshallowing Avalon's history (not done, to stay read-only / disk-light); only file-level absence is listed.
- Whether any of the 191 newer upstream commits were already cherry-picked into Avalon main after Nov 2 was not checked commit-by-commit (Avalon's 48 newer commits contain no "(#NNNNN)" upstream titles, so likely none).

## Q2. What Avalon adds on top of upstream

### Takeaway
Avalon is a single-model (Qwen3.8-Flash-Next, 512 experts/10 active, 48 layers) fork built on buun-llama-cpp's GPU expert cache, with a huge custom HIP/CUDA MoE path ("hosted MoE": GPU computes cached experts while CPU computes missing ones, inside the HIP graph), expert prefetch for prefill, QSA sparse attention for long contexts, fused qwen4exp kernels, IQ3_K/IQ3_KS types from ik_llama.cpp, MTP draft head sharing the expert cache, `-ubd`, and backend greedy sampling. Speculative *types* (ngram-simple, draft-mtp, dflash, eagle3...) are upstream, not Avalon additions.

### Cited Findings
- README positioning: fork "built for one model ... on one consumer GPU with the experts in system RAM"; 98 t/s on RX 7900 XT, PCIe 3.0, Ryzen 7 3800XT; GPU holds ~4.6 GiB dense weights + 9-14 GiB expert cache; 256k context fits under 19.7 GiB with MTP and q8_0/q4_0 KV. — [Avalon README](https://github.com/freriadoc/Avalon/blob/main/README.md)
- "What makes it fast" (README): GPU expert cache (from buun) grown with lent compute buffers and the unused KV tail, 400 MiB calculated reserve; hosted MoE moving 1-2 missing experts to GPU when faster; expert prefetch during prefill on 4 slots borrowed from the cache; QSA sparse attention (GPU-built masks, gathered cells, split kept across speculative steps); MTP draft head with LM head cut to 65k frequent tokens (FR-Spec) sharing the main model's expert cache; fused generation kernels (a linear-attention layer 53 → 39 launches), AVX2 CPU expert kernels, IQ3_K/IQ3_KS. Tail skip of rank ≥7 missed experts on by default (`GGML_CUDA_MOE_TAIL_SKIP=0` disables). — [Avalon README](https://github.com/freriadoc/Avalon/blob/main/README.md)
- Credits: buun-llama-cpp (expert cache, LFU admission, heat profile → shipped `moe-cache-ranking.bin`), thecodacus (expert prefetch), ik_llama.cpp (IQ3_K/IQ3_KS, shared MTP head), Strata (ranking format, preload idea), Unsloth (MTP head), ISTA/AtomicChat/drluoto (quants, FR-Spec list). — [Avalon README](https://github.com/freriadoc/Avalon/blob/main/README.md)
- New files vs upstream base (54 added; non-test): `ggml-cuda/moe-cache.cu` (7,845 lines) + `.cuh`, `ggml-backend-moe-cache.h`, `qwen4-hc.cu` (1,070; hyper-connection kernels), `mk.cu` (939; multi-task kernel), `qsa-select.cu` (+body), `gdn-wmma.cu` (RDNA3 WMMA chunked GDN prefill), fused bodies `fwht-gate`, `fwht-sr`, `gdn-body`, `gdn-gates-body`, `getrows-body`, `idx-key-body`, `norm-gate-body`, `rope-fused`, `ssm-conv-body`, `grid-barrier.cuh`, `mmq-avl-tune.h`, `moe-cache-mmv-tuning.h`, MMQ instances `iq3_k`/`iq3_ks`, `vendors/hip-runtime.h`; scripts `install.sh`, `avalon-server.cmd`, `pack-experts.py`, `check-packed-experts.py`. Largest modified: ggml-cuda.cu (+2,956 diff lines), ggml-backend.cpp (1,921; scheduler borrow/prefetch), mmvq.cu (1,376), qwen4exp.cpp (1,088), ggml-cpu.c (1,053), x86/quants.c (790), top-k.cu (713), common/speculative.cpp (401), common/fit.cpp (330). — local `git diff --stat refs/upstream/base-0ec4027 a91372e`
- New CLI args: `-no-bs/--no-backend-sampling` (and `-bs`), `--moe-cache MODE`, `--moe-cache-expert-parallel N`, `--moe-cache-cpu-overlap auto|N`, `--spec-draft-ubatch-size/-ubd/--ubatch-size-draft N`. — local diff of `common/arg.cpp`
- ~120 Avalon-only env knobs, e.g. `GGML_CUDA_MOE_CACHE_MODE/BUDGET_MB/RESERVE_MB/DEFER/LFU*/PRELOAD/KV_LEND`, `GGML_CUDA_MOE_TAIL_SKIP`, `GGML_SCHED_PREFETCH_EXPERTS`, `GGML_SCHED_BORROW_COMPUTE`, `GGML_MOE_CACHE_LAYER_DEVICE`, `GGML_MOE_SHARE_DRAFT`, `GGML_HIP_COPY_RING/NT`, `GGML_CUDA_GDN_WMMA_MIN`, `GGML_CUDA_MK*`, `LLAMA_QSA_FIXED`, `LLAMA_MTP_KV_ONLY`, `LLAMA_SPEC_TIMING`. — local `git grep` diff
- Speculative types are identical in upstream base and Avalon: dflash, draft-dflash, draft-dspark, draft-eagle3, draft-mtp, draft-simple, ngram-cache, ngram-map-k, ngram-map-k4v, ngram-mod, ngram-simple. — local `git show` of `common/speculative.cpp`
- Build in use: FA K/V combos compiled f16-f16, q4_0-q4_0, q8_0-q8_0, bf16-bf16, **q8_0-q4_0**; hipcc from /opt/rocm/core-10.0; `-march=native -mno-sse4a`; `AMDGPU_TARGETS=gfx1100`, Release, GGML_HIP=ON, GGML_NATIVE=ON. — [build-update.log](file:///home/dan/clone-of-d/Avalon/build-update.log), build/CMakeCache.txt
- Upstream-facing docs: README says tuned on RDNA3 ROCm 7.2.4; CUDA builds (RTX 4090: 141.9 t/s MTP short prompt vs Strata 141.4); the chunked GDN prompt kernel is RDNA3-only; experimental multi-GPU via `GGML_MOE_CACHE_LAYER_DEVICE=1` "has never run on two GPUs". AGENTS.md/CLAUDE.md are upstream's contributor rules unchanged in substance. — [Avalon README](https://github.com/freriadoc/Avalon/blob/main/README.md), local AGENTS.md

### Inferences
- Any "upstream improvement" proposal touching MoE offload, expert caching, MMQ for MoE, GDN prefill or qwen4exp graph should be compared with Avalon's custom kernels first; Avalon has likely already gone further for this one model (e.g. its own hosted MoE vs upstream's `--moe-cache-cpu-share`).
- Areas where Avalon is plain upstream (and so upstream fixes apply more directly): speculative driver types, chat/jinja parsing, server slot logic (but server was trimmed).

### Gaps
- No per-kernel performance attribution found beyond README claims; I did not read moe-cache.cu internals.

## Q3. Avalon repo activity since our build (newer commits, issues, PRs, roadmap)

### Takeaway
freriadoc/Avalon main is **48 commits ahead of c24c03d** (52 ahead of a91372e), pushed 2026-11-08 11:54 UTC. Work since Nov 2: registered-RAM expert arena (default on Linux), multi-GPU expert split and pipeline stages, IQ3_KS via MMA MMQ, link-rate-based hosted-MoE limit, offline allocation planning, and several off-by-default prefill experiments (expert byte ring, grouped MoE, "not working" WIP). 33 open issues / 9 open PRs; no formal roadmap file. A community PR (#41) proposes exactly our `ngram-simple,draft-mtp` launcher default, citing the user's own issue #18.

### Cited Findings
- Repo: created 2026-10-09, not a GitHub fork (no parent), 376 stars, 54 forks, pushed 2026-11-08T11:54:26Z. — [gh repo view freriadoc/Avalon](https://github.com/freriadoc/Avalon)
- compare c24c03d...main: ahead 48, behind 0; 53 files changed (18 in ggml/src/ggml-cuda, src/llama-context/graph/kv-cache/batch, common/speculative.cpp, tools/server/server-context.cpp, new ggml-alloc-plan.h, ggml-sched-ring.h, tests arenatest/expertringtest/prefetchring/rambudgettest). — [compare](https://github.com/freriadoc/Avalon/compare/c24c03d...main)
- Notable newer commits (Avalon SHAs): 213b569 expert tensors registered up to KFD's system memory limit, nothing pinned on the fly; 03c749e "an expert arena in registered RAM, auto by default on Linux"; 517ca1b expert not in arena not proposed for admission; 8c4e475 README RAM cache 16/32 GB numbers; 7990545 hosted MoE limit computed from link copy rate and CPU speed; 8763579/9a2f85d experts split between cards' caches (`GGML_MOE_CACHE_EXPERT_SPLIT=1`); 809e688 MoE cache over several GPUs; a40bb4f/5658177/61e069f/9e9cbb8/6731db3 pipeline stages over two GPUs (`LLAMA_STAGES_FULL_UBATCH_FROM`); 065dacd q8_0/q4_0 sparse FA launched with the right tile; 6cc179c QSA tile mask on GQA tile of 4 on NVIDIA; 98bc077/c10afc1 ggml-alloc offline reservation planning; 853981b quantized embedding rows dequantized on GPU; d52941e prefill dense Q8_0 via cuBLAS on NVIDIA; 7ac9fed MUL_MAT_ID MMQ grid over tiles holding rows; edb71b9/483bc9c IQ3_KS through q8_0 MMA MMQ; ca9af54 cache bands per pool; c8347c9 hosted-MoE server thread sleeps; 5d72de3 borrowed-buffer reservation measures only; 6032e96 MTP draft takes nextn rows ubatch by ubatch with stages; 929c64e link copy rate measured by preload; 5942fb7 prefill expert prefetch via byte ring (`GGML_SCHED_PREFETCH_RING_MB`, off); 7a4f613 CUDA WMMA chunked GDN; d78ce03 MMQ by expert groups (`GGML_CUDA_MMQ_EXPERT_GROUP`, off); b64473f/aa99249/54a77f0/3d685b9/b7eacfc/88313b6 grouped prompt MoE + prompt expert ring (`GGML_CUDA_MOE_GROUPED`, `GGML_CUDA_MOE_RING_PREFILL`, off by default, 54a77f0 marked "WIP ... not working"). — [Avalon commits](https://github.com/freriadoc/Avalon/commits/main)
- README (main) reports a user's 7900 XTX with `--moe-cache-cpu-overlap 2` reaching 2,113-2,130 t/s prefill, and `-ub 19072` + `GGML_SCHED_PREFETCH_EXPERTS=6` 2,024-2,043 t/s prefill, 112.9-122.8 t/s gen (PCIe 4.0). — [Avalon README](https://github.com/freriadoc/Avalon/blob/main/README.md)
- Open PRs: #41 install: launcher defaults to ngram-simple alongside MTP (cshingle; reproduces +71% on code edits 110.6 → 189.2 t/s, prose flat 97 → 98, greedy output identical, accepted length 2.9 → 27-35); #39 Windows CUDA EAGER module loading; #38 Windows/MSVC build fixes; #32/#31 SM80 chunked GDN prefill + benches; #30 fix quantized sparse attention dispatch for GQA 12; #29 Pascal `__nanosleep` guard; #15 gfx1030 first-request stall. — [Avalon PRs](https://github.com/freriadoc/Avalon/pulls), [PR #41](https://github.com/freriadoc/Avalon/pull/41)
- Issue #18 "Research: AValon with speculative decoding" was opened by the user (Dan Iancu / rocktboy1982), linking llama-cpp-7900xt-optimization; cshingle's comment reproduced the ngram-simple finding. — [issue #18](https://github.com/freriadoc/Avalon/issues/18)
- Open issues relevant to our setup: #40 MTP first verification after a short first prompt hangs in `moe_hosted_plan` defer wait (workaround `GGML_CUDA_MOE_CACHE_DEFER=0`, seen on 16de68d, Windows/CUDA); #3 server never finishes loading on ROCm RX 7900 XTX (stuck in moe_cache_begin → hipMemcpyAsync, 14 comments); #24 feature request: n-gram table residency for ngram-* types; #25 benchmark 5900X + 7900 XTX PCIe 3.0 on ROCm 10.1 (8 GB RAM use vs Strata 55 GB; concurrency-2 aggregate 105.7 t/s at 16k); #9 "ROCm error: invalid device function"; #14 RX 9070 XT OOM in prefill; #35 no AVX path; #17 imatrix livelock with expert cache; multi-GPU issues #1, #16, #20, #26, #36. Closed: #13 GPU at 100% idle with -md loaded, #7 draft prefill overflowing n_batch, #5 ctx_dft position after cached prompt, #8 `--vram-reserve-mib` request. — [Avalon issues](https://github.com/freriadoc/Avalon/issues?q=is%3Aissue)
- No ROADMAP/CHANGELOG in repo; the README "Limits" section is the de facto roadmap (multi-GPU untested, per-GPU prefetch would fix x8/x8, Windows installer being tested, kernels not tuned for NVIDIA). — [Avalon README](https://github.com/freriadoc/Avalon/blob/main/README.md)

### Inferences
- Upgrading to Avalon main would bring the registered-RAM expert arena (likely matters for our 57 GB RAM vs 89 GiB model and the "first session slow" symptom), the link-rate hosted-MoE limit and IQ3_KS MMA MMQ (prefill), but also unfinished experiments (off by default) and the open #40 MTP hang risk; it requires a rebuild (not done here).
- Our `ngram-simple,draft-mtp` choice is now being proposed upstream in Avalon (#41), so it is aligned with the project's direction rather than a local hack.

### Gaps
- No measured numbers of Avalon main (post-c24c03d) on a 7900 XT on ROCm 10 were found; README Nov 4 update gives 16/32 GB RAM numbers but I did not extract them.
- Maintainer responses on #41/#40 not yet present (0 comments on #40).

## Q4. Exact runtime configuration in use

### Takeaway
`avalon.service` (systemd --user, port 11437) execs `~/.local/bin/avalon-server`, which runs `build/bin/llama-server` (built from c24c03d) with 256k context, 9536 batch/ubatch, FA, KV q8_0/q4_0, MTP draft head + ngram-simple, custom chat template. Currently non-functional because the model directory is missing and the service is stopped/disabled.

### Cited Findings
- Launcher `/home/dan/.local/bin/avalon-server`:
  ```
  export LD_LIBRARY_PATH="/opt/rocm/core-10.0/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export HIP_VISIBLE_DEVICES=0
  cd "/home/dan/clone-of-d/Avalon-models/Qwen3.8-Flash-Next-Avalon-GGUF"
  exec "/home/dan/clone-of-d/Avalon/build/bin/llama-server" -m Qwen3.8-Flash-Next-Avalon-IQ3KS-00001-of-00033.gguf \
    -c 262144 -b 9536 -ub 9536 -fa on -ctk q8_0 -ctv q4_0 --jinja --parallel 1 \
    -md MTP/mtp-Qwen3.8-Flash-Next-shared-IQ3K-IQ4NL-FR65k.gguf --spec-type ngram-simple,draft-mtp \
    --spec-draft-n-max 2 -ubd 256 -ctkd q8_0 -ctvd q8_0 \
    --chat-template-file /home/dan/clone-of-d/Avalon-models/avalon-chat-template.jinja \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8080}" "$@"
  ```
  Differs from README's run line only by `ngram-simple,` in `--spec-type`, the template file, env and host/port. No `GGML_*` env knobs set (all Avalon defaults, i.e. tail skip on, `--fit` on, expert cache auto). — [launcher](file:///home/dan/.local/bin/avalon-server), [Avalon README Run](https://github.com/freriadoc/Avalon/blob/main/README.md#run)
- Service `/home/dan/.config/systemd/user/avalon.service`: `Environment=PORT=11437`; `Conflicts=strata.service`; ExecStartPre waits up to 180 s for `...-00033-of-00033.gguf`; `Restart=on-failure`, `RestartSec=20`, `TimeoutStopSec=60`; stdout/stderr appended to `/home/dan/clone-of-d/Avalon/avalon-service.log`; `WantedBy=default.target` (currently disabled). — `systemctl --user cat avalon.service`
- Chat template `avalon-chat-template.jinja` (9,115 bytes, 2026-11-02): maps reasoning effort high/max → xhigh (model only knows low/medium/xhigh) and renders later system messages instead of raising (opencode compatibility). — [project_avalon memory](file:///home/dan/.claude/projects/-home-dan/memory/project_avalon.md), [avalon-guide.md §2.4](file:///home/dan/clone-of-d/llama-cpp-7900xt-optimization/docs/avalon-guide.md)
- Local benchmarks (Avalon `bench.py`, 512 tok greedy): bench-ctk-q4 (q4_0 K) 86.2 t/s r1; bench-ctx131k 78.6 t/s r1 (first runs, cold); edit task: default MTP median 118.0 t/s vs ngram-simple+MTP median 168.1 t/s (2nd round 180.9), acceptance 1462/1510. Published guide numbers: ~100 t/s decode short prompts, 1,771/1,805 t/s prefill at 16K/32K, ~4 s warm TTFT; freeing VRAM (q4_0 K, 131k ctx) changed nothing. — [bench-edit-ngram.txt](file:///home/dan/clone-of-d/Avalon/bench-edit-ngram.txt), [bench-edit-default.txt](file:///home/dan/clone-of-d/Avalon/bench-edit-default.txt), [avalon-guide.md](file:///home/dan/clone-of-d/llama-cpp-7900xt-optimization/docs/avalon-guide.md)

### Inferences
- Untried knobs that Avalon users report helping on similar hardware: `--moe-cache-cpu-overlap 2`, larger `-ub` (19072 on a 24 GB XTX; may not fit on 20 GB), `GGML_SCHED_PREFETCH_EXPERTS=6`. These exist in our c24c03d build (args/env present at a91372e).
- Rerunning Avalon's `install.sh` overwrites the launcher (would drop ngram-simple and the template unless PR #41 is merged).

### Gaps
- Could not verify the running state/throughput of c24c03d binary: service is down and model files are missing; no live `/props` check possible.
