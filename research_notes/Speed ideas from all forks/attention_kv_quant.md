# Speed ideas from Strata fork commits: attention, KV cache, long context, prefill and quant formats

Scope and method (2026-11-10): `data/fork-kb/forks.sqlite`, commits with `upstream_history = 0` (43,685 commits, 407 repos,
2026-09-24 .. 2026-11-10). A regex over messages (attention/attn/flash/sparse/kv/gdn/rope/context/prefill/prompt cache/prefix/
chunk/quant/iq*/q*_/mxfp/nvfp4/fp8/w4a4/requant/trellis/...) matched 25,595 commits. After dropping merges, subjects that are
already in pancake's `git log --all` (6,169 subjects), Intel/Apple/POWER-only forks and doc-only noise, 842 + ~150 candidates
were read by subject. About 110 diffs/messages were read with `gh api repos/<repo>/commits/<sha>` (read-only).
"Pancake has it" was checked by grepping `src/`, `include/`, `serve/`, `configs/pancake/*.json` on branch
`port/rdna3-decode` (BFLA lives on `feat/bfla-hip`, not on this branch). All speed numbers are the fork authors' own,
unverified on our box. Nothing was built or run.

Key fact found while checking "already has": pancake runs **Qwen3.6-35B-A3B through `strata-poly`** (preset
`configs/pancake/qwen36-ud-q4_k_xl.json`, `--kv-q8 --attn-split`), ported from FastStrata on 2026-11-07 (commit 53d9b0925),
and `run-pancake.sh` refuses poly on HIP. Pancake's poly has none of RandoTeam/4gbVramStrata's later attention work
(`grep` finds no `STRATA_FATTN`, `STRATA_FDEC`, `attn_flash64`, `STRATA_GDN_MMA`, HIP value split). The separate
`src/qwen35/q36_engine.cu` in the tree (not used by any pancake preset) still has the FP32 K/V cache, the
one-block-per-query-head decode kernel `k_attn_part<<<dim3(nc, nt*H)>>>` and the cuBLAS+materialized-softmax prompt
attention, i.e. none of the q36 fixes listed below. The Flash-Next main engine on HIP, by contrast, already carries
almost every prompt-attention/GDN switch the forks publish (see Q1/Q2).

## Q1. Attention kernels (prompt flash attention, GQA-shared decode, sparse/QSA scoring)

### Takeaway
The biggest untaken attention wins are all on the **Qwen3.6 (poly) path**: RandoTeam's llama.cpp flash-attention prompt
kernel (`STRATA_FATTN`, 1.62x prompt at 118K) and tensor-core decode attention (`STRATA_FDEC`, 1.60x decode at 118K), both
default-on in that fork since 10-28, CUDA Ampere+ only, and exactly the "8.8 fused flash-attention prompt kernel (D=256,
GQA 8, Q8 cache)" item pancake's MOD_PLAN still lists as own work. For Flash-Next (QSA sparse attention) pancake already has
the HIP WMMA prompt attention, IMMA, PA_FAST/PA_TRIM/RPF, ATTN_DAILY and the causal block skip; only small selector
tweaks remain.

### Cited Findings
Qwen3.6 / poly path (Qwen3.6-35B-A3B: 16 query heads, 2 KV heads, head dim 256, GQA 8):
- RandoTeam/4gbVramStrata `79b1e756` (2026-10-18), `STRATA_FATTN=1`: prompt-batch attention for head 256 / GQA 8 through a
  copy of llama.cpp's `fattn-mma-f16` with the causal mask computed in-kernel; under the q8_0 cache it dequantizes key blocks
  into the existing score scratch and merges blocks by log-sum-exp; "No VRAM is added"; Ampere+ NVIDIA only, Turing/HIP keep
  `attend_batch()`; checked against `attend_batch()` and doubles (`poly_fattn_test`). Numerics: different summation order
  (not bitwise). — [RandoTeam 79b1e756](https://github.com/RandoTeam/4gbVramStrata/commit/79b1e756bae98068e3ebff321ea6c5e056694682)
- RandoTeam `67f60c12` (2026-10-18), `STRATA_FDEC=1`: tensor-core (mma m16n8k16) split-K decode attention over q8_0 or fp16
  cache, a block takes 1,024 keys of a group for up to 4 tokens (keys read once per 4 tokens), same bits alone or in a
  verify window; within 1.8e-3 of doubles (old kernel ~1e-6). RTX 3070, one layer at 140K: q8_0 0.45 vs 1.94 ms (1 token),
  0.52 vs 4.91 ms (3 tokens); slower below ~4K context. Model (Cyber-Tiel-Coder 35B-A3B UD-Q4_K_XL, 256K q8_0, RTX 3070):
  decode 26.6 -> 45.6 tok/s at 174K, +44-71% from 72K on, 11/12 vs 11/12 long-context answers. —
  [RandoTeam 67f60c12](https://github.com/RandoTeam/4gbVramStrata/commit/67f60c12ddb5f04b49efaa1a4f7477e88bc877ca)
- RandoTeam `bee07744` (2026-10-28): both made default where they fit (Ampere+, head 256); FDEC only from 4,096 positions.
  RTX PRO 6000 held to 12 GB, Qwen3.6 UD-Q3_K_M: read 7,039 -> 8,555 tok/s and write 188 -> 252 at 32K; 5,837 -> 8,210 and
  106 -> 230 at 100K. — [RandoTeam bee07744](https://github.com/RandoTeam/4gbVramStrata/commit/bee077441b8ae0284de296f28cfe29be0eade564)
- RandoTeam `aa862591` (2026-10-29) community report, RTX 4060 Laptop 8 GB, Qwen3.6-35B-A3B UD-Q3_K_M, 117,977-token
  prompt: FATTN prompt 814 -> 1,318 tok/s (1.62x), FDEC decode 32.1 -> 51.0 tok/s (1.60x), VRAM unchanged; no quality test. —
  [RandoTeam aa862591](https://github.com/RandoTeam/4gbVramStrata/commit/aa8625919a1a2677d34e03ed96c6967ffeae3112)
- Pancake's own record: attention is ~38% of a 42K read on the 3060 in poly, and ggml `fattn` was rejected because it
  "needs a causal mask buffer and fp16 copies of a Q8 cache ... ~130 MB at 42K" (docs/PANCAKE_FINDINGS.md section 5;
  MOD_PLAN item 8.8). RandoTeam's version computes the mask in-kernel and reuses the score scratch, which addresses exactly
  that objection (claim from its commit message, not verified here).
- RandoTeam `372f4ece` (2026-10-23): `attn_flash128_kernel` (128-value heads, groups of 8, scores in registers, 68 KiB smem,
  Ampere+) for Flint; RTX 5090, 29,816-token prompt: attention was 1.9 of 3.4 s of kernels before. `cd87aecd` (10-15):
  cp.async double buffering in `attn_flash64` (1,398 -> 1,366 ms for 29.8K on 5090). `fa6c3f94` (10-09): a verify window's
  tokens read K/V once between them (Gemma 4: 4-token window 4.45 ms/layer vs 1.12 for one token before). —
  [372f4ece](https://github.com/RandoTeam/4gbVramStrata/commit/372f4eced30f6da5b1918647301f7a30395b0114),
  [cd87aecd](https://github.com/RandoTeam/4gbVramStrata/commit/cd87aecde680ac67a642c9a0b62172fde380f891),
  [fa6c3f94](https://github.com/RandoTeam/4gbVramStrata/commit/fa6c3f943e1397b057663f43822f63ef8912ed82)

Qwen3.6 on Strata's own `q36_engine` (in pancake's tree but unused by presets; all bit-preserving or near):
- dongnh311 `62cc598f` (2026-10-16): decode attention one block per (chunk, KV head) for all 8 query heads, warp-coalesced
  K reads; RTX 3060, Ornith-1.5-35B-A3B: kernel 2.37 -> 0.227 ms/layer at 16K; tg 15.7 -> 28.1 tok/s (bench-pp 16000);
  served decode after 16K prompt 12.0 -> 27.5 tok/s; rel err <= 6.3e-6; `STRATA_Q36_ATTN_PER_HEAD=1` = old. —
  [dongnh311 62cc598f](https://github.com/dongnh311/Strata/commit/62cc598f3b0734b1105e68c4b236bf7c519ebf98)
- storm-ace `4fe0284d` (2026-11-03): `k_attn_part_gqa<G>` (G=2/4/8/16), K staged in LDS, same partial bits; gfx1151
  UD-Q4_K_XL: ~8.9K ctx 25-31 -> 37.0 tok/s, ~35K 11.4-12.5 -> 27.1; 22,512 comparisons 0 mismatches. —
  [storm-ace 4fe0284d](https://github.com/storm-ace/Strata/commit/4fe0284da5f6eb0c1d1b38a5f827581878b7ea33)
- storm-ace `e4fc07ee` (2026-11-03): `k_attn_fa_wmma` flash prompt attention on RDNA3/3.5 `v_wmma_f32_16x16x16_f16`
  (wave32, 64 query rows/block, 16-cell K/V LDS tiles, V transposed, online softmax in log2) + 35 hipBLASLt table rows;
  gfx1151 pp32768 360 -> 517 (fused attention) -> 938 tok/s (table rows); KL 2.2e-6; `STRATA_Q36_NO_FA=1` = old. Directly
  reusable design for a gfx1100 Qwen3.6 prompt kernel. — [storm-ace e4fc07ee](https://github.com/storm-ace/Strata/commit/e4fc07ee57c4d212f5720ce1e72bf41b7953b15f)
- maoyufeng1985 `dd653028`/`2f7020f2` (2026-10-29): `STRATA_Q36_ATTN_GQA=auto` (shared shape when the decode reads >= 2048
  cells): P100, ctx 131072 reading 32,768 cells: attention 14.5 -> 4.8 ms/token (3.03x), decode 34.5 -> 51.8 tok/s (+50%);
  byte-identical tokens; -2.6% at 576 cells (why auto). —
  [dd653028](https://github.com/maoyufeng1985/Strata/commit/dd653028c3341f06af872b89627c188640803c52),
  [2f7020f2](https://github.com/maoyufeng1985/Strata/commit/2f7020f29e8d127e19e44220fb8b9fb7355abd61)
- bahooo22 `bbc034d4` (2026-11-03): split-K partials sized for T<=8 (384 -> 48 MiB at 96K) + grid order so the 8 q-heads of a
  KV head run adjacent and share L2; 96K: tg 18.3 -> 23.5 tok/s (+28%, mostly from 6 more VRAM expert slots), pp +8%. —
  [bahooo22 bbc034d4](https://github.com/bahooo22/Strata/commit/bbc034d48a45da320a8e60a41b0609b596cacbce)
- Aaron-xx `69d28a34` (11-07) `STRATA_Q36_ATTN_KSEQ` (default on, one KV head per score slice): GTX 1660 Ti 41K prompt
  269 -> 293 tok/s (+8.8%); `a22cb5a8` `STRATA_Q36_ATTN_S16` (opt-in FP16 score accumulation, lossy): 320.5 -> 364.3 tok/s
  (+13.7%), 23/24 top-1 identical; `24758d5a` frees 64 MiB. —
  [69d28a34](https://github.com/Aaron-xx/Strata/commit/69d28a3428e35cc91d56ad8ad8772cc0d1714184),
  [a22cb5a8](https://github.com/Aaron-xx/Strata/commit/a22cb5a8b38317919184f2cf7226510f67719c49)
- eddoursul/Lamina `0e2db480` (2026-10-30): q36 path flash-decoding (16K: 3,181 -> 248 us/layer), FP16 K/V, mma m16n8k16
  flash prompt attention with cp.async (16K: ~1.24 -> 0.72 s of the prompt), MMQ projections; RTX 3090/5070 Ti class,
  65,536 cells: prompt 1,863 -> 2,817 tok/s, generation after it 44.6 -> 112.5 tok/s. —
  [eddoursul 0e2db480](https://github.com/eddoursul/Lamina/commit/0e2db480db3972547f7b285f9db137ae32ef2ffc)
- Asofwar/Strata-Turbo (main engine's dense Qwen3.6 attention in `qsa_prompt_attn.cu`, CUDA): `cce467a1`
  `STRATA_PROMPT_ATTN_FP16ACC=1|2|3` RTX 5080 TTFT 64K 19.07 -> 11.96 s (1.59x), KL 0.0094-0.0106 vs noise 0.0080;
  `48c58c03` `STRATA_PROMPT_ATTN_DENSE_IMMA=2` (int8 tensor cores, int8 KV) 130K 37.2 -> 30.8 s, 200K 76.7 -> 60.0 s;
  `1fa9bbcd` `=4` a further 64K 12.68 -> 11.87 s; `6ebe841e` `STRATA_ATTN_WINDOW=1` verify-window decode attention, 200K
  decode 48.6 -> 68.2 (exact-precision) / 88.9 tok/s (FP16 modes). None of these four switches exists in pancake. —
  [cce467a1](https://github.com/Asofwar/Strata-Turbo/commit/cce467a14efe69593dfdf435518e6dfca936ef25),
  [48c58c03](https://github.com/Asofwar/Strata-Turbo/commit/48c58c03611da5e4324e87e4b38a508f3307ac0b),
  [1fa9bbcd](https://github.com/Asofwar/Strata-Turbo/commit/1fa9bbcdce47f330ab0c4509b2e9df691ae7b97c),
  [6ebe841e](https://github.com/Asofwar/Strata-Turbo/commit/6ebe841e2ede7ec81812feffe77460075099a786)

Flash-Next (QSA) path:
- unimatrix099 `dcdd5882` (10-09): `STRATA_HIP_WMMA=1` in production, 2x RX 7900 XTX: 16K prompts +20%, 118K read at
  3,368 tok/s. Pancake HIP preset already sets `STRATA_HIP_WMMA=1` (already covered). —
  [unimatrix099 dcdd5882](https://github.com/unimatrix099/Strata/commit/dcdd588239f9917b53cfbbcaf5c14e15a078233e)
- svgop `047ab209` (11-06): PA_TRIM LDS trim + PA_RPF read-ahead from upstream 2970fe74, +4.7% prefill at 24K, bitwise.
  Pancake has `STRATA_PA_TRIM`/`STRATA_PA_RPF` in `qsa_prompt_attn.cu:2297` (already has). —
  [svgop 047ab209](https://github.com/svgop/Strata/commit/047ab209a6ed90cd4aacd461ae639caf65da050e)
- svgop `c87d6d79` (10-24): `QSA_SPLIT_ATTN=1 + PROMPT_ATTN_IMMA=1` adopted on sm_120, pair +6.5%/conc +3.9%, M1 in noise.
  Pancake has both switches; the CUDA preset sets IMMA but not QSA_SPLIT_ATTN. —
  [svgop c87d6d79](https://github.com/svgop/Strata/commit/c87d6d7917491beea4be02548d24a4238d003dd6)
- Linsche `4a2e497d`/`ddab1952` (11-04) `STRATA_PATTN_FLASH=1|2`: one-pass QSA prompt attention for GPUs without WMMA
  (gfx1030), RX 6800 13.6K prompt 938 -> 1,011 tok/s (+8%). Not useful on gfx1100 (has WMMA) or sm_86 (has TC kernels);
  covered in batch4. — [Linsche 4a2e497d](https://github.com/Linsche/Strata/commit/4a2e497d09e6ddc0d22cfb1c3faed793e767aefd)
- bsorensen110 `ea0bd3af` `STRATA_PROMPT_ATTN_RDNA2` (gfx103x only) and `19fc0e48` `STRATA_HIP_WMMA_Q4=1`: Q4_0-KV prompt
  attention on RDNA4 int8 WMMA, parity PASS on gfx1200, 7.2-7.9x the FP32 kernel; gfx12-only as written. —
  [19fc0e48](https://github.com/bsorensen110/Strata/commit/19fc0e48847d89e145fe1d354cc29cf7aa198368)
- bsorensen110 `ebc12652` `STRATA_SELECT_SPLIT2=1`: QSA selection scorer keeps bf16 hi+mid pieces (3 WMMA instead of 6),
  gfx1151 scorer 2x (12.5 -> 6.6 ms at 65K blocks), selections 256/256 at 32K, 253/256 at 262K, +1.3% end-to-end at 98K.
  Not in pancake (the 6-product `block_scores_wmma_kernel` is there). Lossy (rel err ~1.5e-5 in scores). —
  [bsorensen110 ebc12652](https://github.com/bsorensen110/Strata/commit/ebc12652417ba1d825b2729f5e9387a91b237e70)
- xyzzing `33443c9f`/`6d5ca799` (10-18): gfx1100 fp16 rocWMMA selection scorer arm: select phase 1,934 ms vs 3,176 ms for the
  bf16 arm at 128K prefill (1.64x; phase 3.2% vs 5.1% of 128K prefill). Not in pancake (no rocwmma in `src/`). —
  [xyzzing 33443c9f](https://github.com/xyzzing/Strata/commit/33443c9f514313030ab801c85ed9768dfa7136cc),
  [6d5ca799](https://github.com/xyzzing/Strata/commit/6d5ca79964f97d2f5bc1ef1f68c24f4bb91bbc50)
- lighttransport `2e12e24f` (MiniMax MSA): skip key blocks no query of a group can see, exact, 16K block scores 2.0 -> 1.06 s.
  Pancake's QSA WMMA scorer already stops at the group's highest visible block (`if (b0 >= reach || b0 >= hi_nbid) break;`
  in `qsa_select.cu`), so this is already there. `0cf74ba3`/`851cb55c` (FP8 index keys, butterfly sums) are MiniMax-specific. —
  [lighttransport 2e12e24f](https://github.com/lighttransport/Strata/commit/2e12e24f9b1efee8c21c943bc36e332a1cbdd79f)
- xrip `7ecead78` (10-12): decode QSA scorer halving reduction + split top-k (bitwise; 420K, 8 queries 774 -> 255 us).
  Pancake has `STRATA_SCORES_HALVE` and `qsa_block_topk_split` (already has). —
  [xrip 7ecead78](https://github.com/xrip/Strata/commit/7ecead788668a14407726037c593ea73f7659719)

### Inferences
- Poly FATTN on the RTX 3060: if attention is ~38% of a 42K read and the fused kernel makes it ~2.5-3x faster (in line with
  the 1.2x at 32K / 1.4x at 100K / 1.6x at 118K end-to-end figures above), the 42K read would be roughly 1.25-1.35x faster;
  little below ~8K. FDEC: decode +10-30% at 20-60K (typical opencode contexts), up to +50-60% past 100K, nothing below 4K.
  Both are exact-ish only (fp16 tensor-core arithmetic; KL gate needed). Neither runs on HIP as written.
- For the 7900 XT, the Qwen3.6 model currently cannot run at all in pancake (poly blocked on HIP). The q36 engine's fixes
  (GQA decode, storm-ace's gfx11 WMMA flash prompt kernel) or a HIP port of poly + a rocWMMA flash kernel are the two routes;
  storm-ace's kernel was written for RDNA3.5 wave32 `v_wmma_f32_16x16x16_f16`, which gfx1100 also has.
- Flash-Next on gfx1100: remaining attention-side gains are small: fp16 select arm or SPLIT2 ~1-2% of a 64-128K prefill;
  QSA_SPLIT_ATTN on the 3060 preset is a free A/B (+0-6% on concurrent lanes per svgop, sm_120).

### Gaps
- No fork measured FATTN/FDEC on an RTX 3060 (sm_86, 28 SMs, 3 MiB L2); the 4060 Laptop and 3070 are the closest.
- Whether RandoTeam's FATTN really adds zero VRAM at pancake's 131072-context q8 configuration was not verified in code.
- No fork has a gfx1100 port of FATTN/FDEC for poly.

## Q2. GDN / linear-attention layers (hybrid models) in prefill and decode

### Takeaway
Pancake's Flash-Next HIP preset already runs the chunked WY GDN scan on gfx11 matrix cores plus GDN_HEAD/PP/CONVL2/NOY, and
the CUDA build has a tensor-core scan (`STRATA_GDN_TC`, sm_80+). The only notable new GDN item is poly's opt-in tensor-core
GDN prefill (`STRATA_GDN_MMA`, sm_86 listed) for the Qwen3.6 path; upstream evidence says the WY chunked scan is *slower*
on a 28-SM RTX 3060.

### Cited Findings
- YasuhiroIhara `aa1edcea` (2026-10-28): chunked DeltaNet (WY, sergiywith #1372) default on CUDA only for sm_80+ with >= 128 SMs;
  "forced below it, this recurrence is 0.73x of gdn_rec_kh_kernel on an RTX 3060 (28 SMs) and 0.87x on an RTX 5070". HIP
  path untouched. — [YasuhiroIhara aa1edcea](https://github.com/YasuhiroIhara/Strata/commit/aa1edcea4f0a31f3bcd6b83745c5ece1583cc7f1)
- Asofwar `b4ec4053` (10-20): `STRATA_GDN_CHUNKED=1` on 64-127 SMs runs a 64-block scan in one wave; RTX 5080 recurrence
  1.13-1.22x at 8K-32K but prompt time only -0.8-0.9%. — [Asofwar b4ec4053](https://github.com/Asofwar/Strata-Turbo/commit/b4ec40536b7c8e99328554e79df3a203f6c5a7a0)
- eddoursul `50c83e0f` + `236c73ba` (10-13): chunkwise gated-delta-rule on tensor cores, then one kernel with prep and loader
  warps: one layer 2048 tokens 0.517 -> 0.279 ms (1.82-1.87x); 16K prompt IQ3_S 5,460 -> 5,352 ms (-2.0%); not bitwise with
  decode steps (within 4.6e-4). — [50c83e0f](https://github.com/eddoursul/Lamina/commit/50c83e0f8fd1d5c467b24de40d2ada9156b56a86),
  [236c73ba](https://github.com/eddoursul/Lamina/commit/236c73ba414e1ca3ba88dc5448c6c484c0433fc8)
- xyzzing `60f61fca`/`d4f5b686` (10-18): gfx1100 rocWMMA WY GDN arm, 80.1 ms per 8192-token chunk (4.1x vs cols_pipe
  ~330 ms); engine-level not reported. Pancake has its own gfx11 WMMA chunked GDN (`kernels.cu:1803`) and sets
  `STRATA_GDN_CHUNKED=1` in the HIP preset. — [xyzzing d4f5b686](https://github.com/xyzzing/Strata/commit/d4f5b686d4847ea80b64c2f40935a7b4d8da5643)
- Linsche `5961fb43` (11-04): GDN_HEAD=1 + GDN_PP=2 default on gfx1030: recurrence 999 -> 643 ms at 13.6K (~-2.5% of prompt).
  Pancake HIP preset already sets GDN_HEAD=1, GDN_PP=1 (covered). — [Linsche 5961fb43](https://github.com/Linsche/Strata/commit/5961fb431fee7c930e64f5ca1c22ee8eee9427b6)
- RandoTeam `61b41e84` (10-21): poly `STRATA_GDN_MMA=1`, CUDA tensor-core GDN prefill for qwen35moe, SM 86/89/120/121,
  >= 64 contiguous rows, no extra persistent buffer; changes FP accumulation; "some measured long-response TG runs regressed".
  RandoTeam `68c0374d` (10-10): GDN state column kept in registers for a call (same bits): 2,112-token prompt 997 -> 1,115,
  7,709-token 1,444 -> 1,691 tok/s (RTX 4070 Laptop). Pancake's poly has `gdn_state_reg_kernel` (the 68c0374d idea, likely
  present), not GDN_MMA. — [61b41e84](https://github.com/RandoTeam/4gbVramStrata/commit/61b41e84f4919229af21178ec22373d97b1c2d87),
  [68c0374d](https://github.com/RandoTeam/4gbVramStrata/commit/68c0374dc13efcc361fa8c39bb3acbee988885c5)
- Asofwar `3263b41f` (10-22) profile, RTX 5080 Qwen3.6-class UD-IQ4_XS, 200K from scratch: attention 27.3 of 50.1 s, dense
  cuBLAS 6.3, expert MMQ 4.9, GDN recurrence 2.2 s. GDN is small at long context; attention dominates. —
  [Asofwar 3263b41f](https://github.com/Asofwar/Strata-Turbo/commit/3263b41f974292dd2421e324cc5d71fc82019748)

### Inferences
- GDN is ~4% of a 200K prompt (Asofwar) and the chunked scan does not pay on 28 SMs: expected gain for pancake from more GDN
  work is <= 2% on both GPUs. `STRATA_GDN_MMA` on the 3060 poly path is worth one A/B at 8-32K (prefill only; watch decode).

### Gaps
- No gfx1100 end-to-end number for xyzzing's WY rocWMMA arm vs pancake's existing gfx11 chunked kernel.

## Q3. KV cache formats, paging/growth, RoPE and context extension

### Takeaway
For the hybrid models the K/V cache is small (10 attention layers in Qwen3.6, QSA layers in Flash-Next), so KV quantization
pays mainly by giving VRAM back to expert slots, which is where the forks' decode gains come from. Pancake already has
int8/q8/Q4_0(FWHT)/k8v4 KV, `--kv-grow` and poly's on-demand attention cache growth. Lossy sub-4-bit codecs (TurboQuant,
tqt/hq, NVFP4) exist but were covered before or are Blackwell-only.

### Cited Findings
- dongnh311 `848bfa9e` (q36: FP32 -> FP16 K/V, prompt GEMMs read it in place): RTX 3060, 32K context, 40 -> 52 expert slots
  per layer, decode after 16K prompt 24.3-25.2 -> 27.0-27.4 tok/s. `131df968` `--kv q8_0` (53% of FP16 bytes): 256K 0 -> 29
  slots, 27.5 -> 41.9 tok/s; KL 0.024/0.012 vs f16-vs-f16 0.021/0.007 (near-lossless). —
  [848bfa9e](https://github.com/dongnh311/Strata/commit/848bfa9ecca8904d33a8e57820a2e958b486f7db),
  [131df968](https://github.com/dongnh311/Strata/commit/131df96895774bf9d1c8c2ec80153952264bf169)
- Aaron-xx `f7085c83` (10-31): int8 K/V with FP32 row scale in the q36 path: GTX 1660 Ti context 53,248 -> 65,536, tg -6%. —
  [Aaron-xx f7085c83](https://github.com/Aaron-xx/Strata/commit/f7085c8332b70298db68681d829566b83d027325)
- RandoTeam `c234713a` (11-03): poly q8_0 attention caches on Clover (Qwen3.6-class), 4 GB laptop, 32K context: at 24K
  39.0 vs 30.9 tok/s decode and 676 vs 245 tok/s prompt; at 32K 33.9 vs 25.7 and 663 vs 180; 2-4K no difference; 16/16 quality
  suite both. Pancake's Qwen3.6 preset already uses `--kv-q8` (already has). —
  [RandoTeam c234713a](https://github.com/RandoTeam/4gbVramStrata/commit/c234713af0f60aea92dc6255d0daef81d0f313fb)
- RandoTeam `6a991bce` (10-17): poly `--kv q8_0` enabled on HIP (RX 9070 XT gfx1201): Q8_0 at 196,608 tokens vs fp16 at
  131,072, same speed (4.5K prompt 1,843-1,862 vs 1,814-1,822 tok/s). Only matters once poly runs on HIP. —
  [RandoTeam 6a991bce](https://github.com/RandoTeam/4gbVramStrata/commit/6a991bce1989b48aa2f2d8e7e2f894be3ac2a943)
- RandoTeam `4250af80` (10-11): attention caches reserve addresses for the whole context and map VRAM as the conversation
  grows (4,096 rows at a time, VMM); Qwen3.6 at 128K: caches 2,816 MiB would cut the expert cache 2,616 -> 486 MiB and a short
  chat 94.6 -> 77.5 tok/s without it. Pancake's poly has `--kv-whole` (the opt-out), so growth appears present (already has). —
  [RandoTeam 4250af80](https://github.com/RandoTeam/4gbVramStrata/commit/4250af8008286b00c182db00f7bffaa1297366b8)
- RandoTeam `4636dbe3` (10-17): HIP VMM caveats for poly's elastic expert cache (ROCm releases VRAM only when the reservation's
  addresses are freed; `hipMemGetInfo` under-reports free VRAM; `--kv-whole` turned VMM off). Needed reading for a poly HIP
  port. — [RandoTeam 4636dbe3](https://github.com/RandoTeam/4gbVramStrata/commit/4636dbe3957c09a1d35483ca1d0d8c30dc9a3857)
- lighttransport `a4c890c5` (11-03, MiniMax M3 engine): `--kv-host` (K/V in pinned host memory) and `--kv-fp8` (E4M3 with
  per-row scales): needles correct to 255K, 128K decode at 74% of short context with FP8 in VRAM. FP8 KV has no hardware
  benefit on gfx1100/sm_86 over int8 (same byte count). — [lighttransport a4c890c5](https://github.com/lighttransport/Strata/commit/a4c890c5a60c59e18d021d61d6b31451a2c6ade7)
- MyEcoria `e349ab8d` (11-01, Flint): `--kv-quant tqt|hq --kv-bits 2|3|4` (Hadamard + trellis / E8 lattice): RX 9070 decode at
  262,144 context 17.2 -> 37.8 tok/s, but slower than FP16 with a filled 27K context (40-44 vs 48); KL 0.09-0.18/step (lossy). —
  [MyEcoria e349ab8d](https://github.com/MyEcoria/Strata/commit/e349ab8da8454d9cb57f17e5c155fe2543b57524)
- ProTekk `b3afd945` (TurboQuant KV turbo8/4/3tcq/2tcq/1tcq) and vvvDuB `f0448709` (NVFP4 KV, Blackwell) - TurboQuant is
  already in MOD_PLAN/batch4 (already covered); NVFP4 needs FP4 hardware (neither GPU). —
  [ProTekk b3afd945](https://github.com/ProTekk/Strata/commit/b3afd94553ab3d024f919d22442648c3bed6975a),
  [vvvDuB f0448709](https://github.com/vvvDuB/Strata/commit/f04487093d0fc5c510cacbc03191d5a89bab96e7)
- RoPE/context: the RoPE commits found are numerics/memory hygiene, not speed: eddoursul `4ab4bf42` (one angle helper, rope
  scaling CLI, float64 table opt-in), `e6ed2a50` (inverse frequencies once per pair), gputier `ffe982b4` `STRATA_ROPE_F64`,
  samnolak `e8b8b620` `STRATA_ROPE_SHARE` (one RoPE table per device across sessions, memory). TheHubUser4860938 `bd3e1544`
  reports 393,216 with yarn-1.5 beating 524,288/yarn-2 "on every metric" for its memory project (quality claim, not speed).
  (sha prefixes from the KB; diffs not read.)

### Inferences
- On the 3060 the Qwen3.6 preset already has q8 KV; the remaining KV lever is the decode kernel (FDEC) rather than format.
- On Flash-Next both presets use `--kv int8 --kv-resident 32768`; moving to Q4_0 KV would free VRAM for experts but the fast
  Q4 prompt-attention kernel (`STRATA_HIP_WMMA_Q4`) is gfx12-only; porting it to gfx11's int8 WMMA (`v_wmma_i32_16x16x16_iu8`)
  is a medium task with an estimated +0-5% decode (more expert slots) at 64-128K on the 7900 XT; unmeasured.

### Gaps
- How many extra expert slots Q4 KV would free in pancake's 131072-context Flash-Next config was not computed.

## Q4. Prefix / prompt / conversation caching

### Takeaway
Pancake already has the upstream conversation cache, prompt-cache spread, superseded-copy dropping and transient-turn handling;
two small serve-side ideas are not there: Ben1078's trailing-reminder move (`STRATA_TRAILING_REMINDER`) and wesraph's
vLLM-style hash-chained page prefix cache (needs the `--kv-share` page pool pancake does not have).

### Cited Findings
- Ben1078 `8b90ab80` (10-23): agent clients that append a transient reminder-only user turn (Qwen Code named) broke the
  checkpoint every turn (5-19 s re-reads at 18K); `_trailing_reminder()` moves it into the trailing system turn; 18,271 of
  18,822 tokens reused next turn. Default on, `STRATA_TRAILING_REMINDER=0` off. Exact. Not in pancake (`grep` finds no
  `_trailing_reminder`). — [Ben1078 8b90ab80](https://github.com/Ben1078/Strata/commit/8b90ab8077ed943d7b2f0e3e31417608062885e2)
- wesraph `17abc917` (10-26): automatic hash-chained prefix cache on kv-share pages (fnv64 chain, tokens verified on hit, LRU,
  refcount pins); a second agent mapped 3,444 cached pages and read 30 tokens instead of 13,807. Requires wesraph's kv-share
  pool (`--kv-share` absent in pancake). — [wesraph 17abc917](https://github.com/wesraph/Strata/commit/17abc917c1308bca6565e77d580675e1df7f3f68)
- eddoursul `83f40aee` (10-18): a stashed conversation matched before its newest checkpoint lends its start (upstream
  c9e571e0's idea): three subagents after a ~4,000-token system prompt, cached 4,021 -> 12,372 of 12,975 tokens (IQ3_S 3.00 ->
  0.67 s). Whether pancake's prompt cache has this borrow path was not confirmed (generic "borrow" matches many files). —
  [eddoursul 83f40aee](https://github.com/eddoursul/Lamina/commit/83f40aee62128e6031ae0c18ae5cd220d8d397f7)
- quangorn `5150d421` (10-31): drop a parked copy that is a near-prefix (<=10% unshared) of a longer parked one (the OpenCode
  shape: a 76K copy as prefix of a 108K copy, both 1.4-1.7 GiB). Pancake has "superseded" logic (9 files); this exact rule not
  confirmed. — [quangorn 5150d421](https://github.com/quangorn/Strata/commit/5150d421cb73a2d9bcfcbdd698262919192714fd)
- bsorensen110 `4383fcec` (11-07): two-tier L1/L2 disk conversation cache with L2 prefix borrowing (no numbers in message). —
  [bsorensen110 4383fcec](https://github.com/bsorensen110/Strata/commit/4383fcec3b69580108fc9b2c18854fe953f15481)
- RandoTeam `aa862591`: on poly, `--prompt-cache 16 --prompt-cache-every 1024` shrinks a follow-up's re-read from 11,519 to
  5,375 tokens at 118K; harness-shaped request: 82.4 s first turn, 10.4 s per follow-up, 1.0 s identical resend. —
  [RandoTeam aa862591](https://github.com/RandoTeam/4gbVramStrata/commit/aa8625919a1a2677d34e03ed96c6967ffeae3112)

### Inferences
- Prefix reuse is the largest "prefill" win for agent use (seconds per turn), but pancake's cache is already mature. Check
  whether opencode sends a trailing reminder-only turn (if yes, the Ben1078 change is a few-line server.py port with
  multi-second TTFT savings per turn; if no, zero gain). The poly `--prompt-cache-every` setting should be checked in the
  Qwen3.6 preset (it sets none).

### Gaps
- opencode's request shape (trailing transient turns) was not inspected in this task.

## Q5. Prefill GEMM paths, chunking, layer-major and fusions

### Takeaway
Most prefill GEMM work the forks did is already in pancake on HIP (PF_FUSED, PF_GEMM, DENSE_MMQ on HIP, MMQ_KQ, hipBLASLt
tables, PF_F16). Remaining candidates: zsariboga's CUDA-only prefill fusions measured on the same IQ2_XS model (-4% at 32K on a
5090), dense MMQ on CUDA (mixed results; changes bits and costs MTP acceptance), and for Qwen3.6 per-expert MMQ launches
(+58% prefill on a 1660 Ti in the q36 engine; poly has its own MMQ grouping).

### Cited Findings
- zsariboga `21107e9d` (10-08), Flash-Next IQ2_XS, RTX 5090: Q8_0/IQ4_XS dequant with 16-byte stores (141.1 -> 11.2 ms),
  fused swiglu+q8_1 quant (`STRATA_PREFILL_SWIGLU_QUANT`, 43.3 -> 19.6 ms), fused MoE combine + hyper-connection write
  (`STRATA_PREFILL_COMBINE_WRITE`, 234.4 -> 203.0 ms); 32K 4,874.8 -> 4,679.4 ms (-4.0%), 8K -2.8%; first-token logits
  identical; CUDA only. None of the three names is in pancake. —
  [zsariboga 21107e9d](https://github.com/zsariboga/Strata/commit/21107e9d05364385527485fa6294dc5db4903976)
- eddoursul `a1444e33` (10-13): dense projections of >= 2048 rows through llama.cpp MMQ (q8_1 activations): IQ3_S 2K prompt
  -27..-30% time, 16K -16%; KL within noise for IQ3_S, +18% for UD-Q4_K_XL. In pancake `Gemm::native_mmq` is HIP-only
  (`#if defined(STRATA_PREFILL_MMQ) && defined(__HIPCC__)`). — [eddoursul a1444e33](https://github.com/eddoursul/Lamina/commit/a1444e338642bce7d6bd6a82ea1d35ec7c720d82)
- samnolak `51551f05` (11-01): `STRATA_DENSE_MMQ` on CUDA (opt-in, W8A8, not bit-identical). svgop `8731e33c` (10-17) measured
  it on sm_120: prefill 21K -2% / 2.7K +3.1%, but -2.1pp MTP acceptance = -2.8% decode ("prefill-bits changes pay ~-2.5% decode
  acceptance tax"). Asofwar `3263b41f`: dense kernels -1.1 s at 200K but KL 1.6x run-to-run, "not recommended". Conflicting
  evidence. — [samnolak 51551f05](https://github.com/samnolak/Strata/commit/51551f05cd95d203d7c030020f1ec05bb7803e96);
  contradicted by [svgop 8731e33c](https://github.com/svgop/Strata/commit/8731e33c770626fb1d31b1bd05f667fa3681c90f) and
  [Asofwar 3263b41f](https://github.com/Asofwar/Strata-Turbo/commit/3263b41f974292dd2421e324cc5d71fc82019748)
- Asofwar `3263b41f`: `STRATA_MMQ_KQ=1` on a UD-IQ4_XS pack: serve 200K 51.8 -> 49.4 s, 130K 27.1 -> 25.7 s, KL at run-to-run
  level. Pancake has `STRATA_MMQ_KQ`. (already has)
- Aaron-xx `535ce073` (11-06), q36 engine: one MMQ launch per expert instead of one per layer (routing uneven: mean 40 rows,
  max 740-790): GTX 1660 Ti bench-pp 8192 231 -> 366 tok/s (+58%), serve 7,928 fresh tokens 220 -> 350 tok/s. —
  [Aaron-xx 535ce073](https://github.com/Aaron-xx/Strata/commit/535ce07303c17fd5d0c7440c62e5b876b09bb673)
- anon761 `a9603c89` (10-01): Q5_1 down experts through MMQ (UD-Q4_K_XL has Q5_1 down in 43 of 48 layers): dequant+gemms
  590 -> 280 ms per 2048 chunk; 2x3090 prefill 2K 649 -> 737 tok/s. Pancake's main-engine moe_mmq lists Q5_1 under
  `STRATA_MMQ_KQUANTS`; poly uses ggml MMQ (has Q5_1). — [anon761 a9603c89](https://github.com/anon761/Strata-upstream/commit/a9603c890c8d76d7fc344b3bb09e68c79becdf43)
- troytse `9b49593f` (10-14): `STRATA_WMMA_GEMM` beside `STRATA_PF_GEMM` on gfx1151, prompt +5.6..+9.1% (4-32K), tokens identical.
  Pancake has the switch; the HIP preset sets `STRATA_PF_GEMM=1` but not `STRATA_WMMA_GEMM` - free A/B on gfx1100. —
  [troytse 9b49593f](https://github.com/troytse/Strata/commit/9b49593fdff41fb4ac5c7c461353bbcd320fa8b0)
- unimatrix099 `a596fcae`/`1358e4d4` (10-09): `STRATA_PF_FUSED=1` prompts 1.8-2.3x at 16K-121K and `STRATA_PA_FAST=1` +2-3.5% on
  2x RX 7900 XTX. Both in pancake's HIP preset (already has). —
  [a596fcae](https://github.com/unimatrix099/Strata/commit/a596fcaea065da545bad941e4dcde6e797481573)
- Chunk size: hnvcam `d09ad14d` IQ4_XS prefill chunk 512 vs 4096: 3.13x/4.76x at 4K/16K; RandoTeam `7decf8da` poly
  `--prompt-batch 2048` 649 -> 765 tok/s at 8K but only 17 of 421 expert slots left (decode collapses). Pancake uses
  `--prefill auto` with `STRATA_PREFILL_AUTO_MAX=32768` on HIP (already tuned). —
  [hnvcam d09ad14d](https://github.com/hnvcam/Strata/commit/d09ad14d5d8b58b36f9407ce9ce1fabf30c83097),
  [RandoTeam 7decf8da](https://github.com/RandoTeam/4gbVramStrata/commit/7decf8da467d4b7f8eddea1e2dae2235e42f4b21)
- MyEcoria `96b558a8` (Flint, RX 9070): long-prompt path on matrix cores (hipBLAS FP16 dense, batched-GEMM attention, MMQ experts
  in groups of 16, 8-buffer ring): 6.4K prompt 94 -> 1,164 tok/s - shows what poly-on-HIP would need for prompt speed. —
  [MyEcoria 96b558a8](https://github.com/MyEcoria/Strata/commit/96b558a8e4ecc4bd05c3ded5a86ecf9b68aeccce)
- android-imdad `caa9b001` (10-13), RX 7900 XTX Flash-Next IQ3_S, bit-exact: ROUTE_FAST, GDN_NORM128, GRUP_WIDE, TOPK_REGKEYS/SCAN,
  WAIT_COPY256, ADAPT_COPY2, DMA_FLAG_WRITE, DQ_FAST (Q6_K/IQ4_XS dequant by warps) etc. (no per-item numbers in the message).
  Several are in pancake's HIP preset (WAIT_FUSE, ROUTE_PUBLISH); this fork was covered in the earlier amd_rdna_ports notes. —
  [android-imdad caa9b001](https://github.com/android-imdad/Strata-7900xtx/commit/caa9b00115e7f134292cc5be971e35095934f587)

### Inferences
- 3060 Flash-Next prefill: porting zsariboga's three fusions is exact and could give ~-3..-4% at 8-32K (5090 figure; the 3060
  is more bandwidth-bound, so the dequant-store fix may matter more there). CUDA dense MMQ: skip or keep opt-in (net negative
  with MTP per svgop).
- gfx1100: `STRATA_WMMA_GEMM=1` A/B is the only free prefill switch found (+0-9%, gfx1151 figure).

### Gaps
- No fork measured zsariboga's fusions on sm_86.

## Q6. Quant formats and their kernels (bytes read per token)

### Takeaway
Pancake's daily Flash-Next pack is GSQ-RCO IQ2_XS (gate/up IQ2_S / IQ2_XXS / IQ1_M, down type 42 in `native_experts.txt`,
~35.5 GB of experts). The only format in the forks that reads materially fewer expert bytes for the same model is Mirai's QNS
trellis build Flash-Next-S-small (26.5 GiB of experts in RAM, ~-20%), CUDA+AVX2 only; quality vs IQ2_XS is unmeasured.
MXFP4/NVFP4/W4A4 paths target FP4/FP8 hardware (gfx12, sm_120) that neither GPU has. Pancake already has the RDNA3
`v_perm_b32` intrinsics, Q5_0 experts and the IQ4 W4A8 kernel.

### Cited Findings
- alesha-pro `8e75f6f9` (10-20): runs Mirai's Qwen3.8-Flash-Next-S (QNS trellis, routed experts 1-5 bits, 2.4 average,
  backbone/head 5 bits). docs/QNS.md at that commit: RTX 3090, 32K, MTP, one run each: S (33.8 GiB experts) decode 35-81 tok/s,
  6,210-token prompt 4.1 s; S-small (26.5 GiB) 47-106 tok/s, 3.9 s; "Not built or run on AMD (HIP)". ian-morgan99 `21bcc4ef`
  ports the same to 0.2.6. — [alesha-pro 8e75f6f9](https://github.com/alesha-pro/Strata/commit/8e75f6f99b6380b2f12317488578f5f555079ad9);
  [docs/QNS.md](https://github.com/alesha-pro/Strata/blob/8e75f6f99b6380b2f12317488578f5f555079ad9/docs/QNS.md)
- eddoursul `f6d95e04` (11-02) and tctien342 `8a209c27` (10-19): pair-trellis TQP5/6/7 experts (ggml 147-149, 128-value blocks,
  2048-point codebook) with block-Hadamard rotation on CPU (AVX2/AVX-512), verify windows and MMQ overlay; jmnargi `2d541397`
  reads them too. These are for Agention's Gyro-L files (5-7 bit), not a lower-byte format for Flash-Next. —
  [eddoursul f6d95e04](https://github.com/eddoursul/Lamina/commit/f6d95e04eef78d707a8a3b36e5eed412fa0ea0ab),
  [tctien342 8a209c27](https://github.com/tctien342/strata-x/commit/8a209c278912621a213232af79d3943437485a28)
- svgop `4152f114` (11-06): unsloth UD-IQ2_XS (Dynamic 3.1) "overtakes GSQ on pair (+6.8%) and conc (+8.6%) at HALF the bytes",
  MMLU 92.86 vs GSQ 94.29 (sm_120 5090; svgop's GSQ baseline appears to be its IQ3_S-class pack, not our IQ2_XS). Needs Q5_0
  down experts on GPU (eddoursul `dbdf0a51`), which pancake has (`vec_dot_q5_0_q8_1`, MMQ Q5_0 case). The bf16-dense conversion
  (`STRATA_UD_BF16_DENSE`) is not in pancake. — [svgop 4152f114](https://github.com/svgop/Strata/commit/4152f114d284d97043c312a8f0d6c179436fb01b),
  [eddoursul dbdf0a51](https://github.com/eddoursul/Lamina/commit/dbdf0a51513ccb51683fd9110c12be927735e0c7)
- StevenChenSE `7e7e6e15` (10-01): RDNA3/RDNA4 `__byte_perm` -> `v_perm_b32` + SWAR packed-byte ops; R9700 IQ3_XXS 131K decode
  55.0 -> 67.5 tok/s. Pancake has `__builtin_amdgcn_perm` byte_perm in `include/strata/hip_compat/intrinsics.hpp` (already has). —
  [StevenChenSE 7e7e6e15](https://github.com/StevenChenSE/Strata/commit/7e7e6e15d548728c42544ba01d8064f2e3935e2a)
- RX5950XT `b9a2111d` (10-19): `-DSTRATA_HIP_SWAR_SIGNS=1` IQ2/IQ3 grid signs without byte compares (llama.cpp apply_signs form,
  exact): ~25% fewer weighted VALU in 112 kernels but 17-19 lose waves; not measured on AMD. Not in pancake. Low priority. —
  [RX5950XT b9a2111d](https://github.com/RX5950XT/Strata/commit/b9a2111d3f4ece53487ce054e1b23aa3a8939214)
- xrip `78bf7922` (11-07) `STRATA_IQ4_W4A8` on sm_75 (MMQ bits, prompt -2%), sun19970908 `a063c138` (11-09) admits sm_89. Pancake
  has the kernel; `arch_ok` admits sm_86. Irrelevant for the daily IQ2_XS pack (no IQ4 experts); relevant only for IQ3_S-class
  packs with IQ4_NL downs. — [xrip 78bf7922](https://github.com/xrip/Strata/commit/78bf7922d929407fc68287b7ec47843c41037639)
- dardeaw `a62189cb` (11-04) `STRATA_MXFP4_FP8=1`: MXFP4 experts -> E4M3 on gfx12 FP8 WMMA (prefill). gfx1100 has no FP8 WMMA;
  already covered in batch1. — [dardeaw a62189cb](https://github.com/dardeaw/strata-mxfp4-rdna/commit/a62189cb21df75381341bed8c28ad122fe7a7f25)
- serboox `fe057bd1` (11-04): CPU AVX-512 VBMI IQ3_S gate/up rows 1.2-1.4x, bit-identical (cycles/block 43.4 -> 34.5 at 1 token).
  The 5700G (Zen 3) has no AVX-512, so not applicable. — [serboox fe057bd1](https://github.com/serboox/strata/commit/fe057bd143a1ff862628c4a5d1e3f82cc67e4601)
- Cudila `911187e2` (Nemotron-H): W4A4 on FP4 tensor cores (sm_120a), int8-K/FP16-V attention; Blackwell-only. —
  [Cudila 911187e2](https://github.com/Cudila/Strata/commit/911187e2b5f5ad0cdc0b5ff6e08bc74a7c66d78c)
- No fork commit found for ik_llama-style IQ2_KS / IQ2_KT / IQ3_KS kernels in Strata (searched `iq2_k|iq2_kt|iq\d_ks|ik_llama`:
  zero hits outside docs); W4A4 appears only as gfx906/Blackwell build guards (VlDubr `8b62b982`, janus-reith `e231fe93`).

### Inferences
- QNS S-small would cut expert bytes ~20% vs GSQ IQ2_XS (26.5 vs ~33 GiB), which on a bandwidth/PCIe-bound decode is worth up to
  ~+10-20% on the 3060 if quality holds; it needs a full port (format, CUDA kernels, packer) and HIP kernels do not exist, so
  on the 7900 XT it is a large job. Lossy relative to IQ2_XS only in the sense of a different quant; a KL/quality gate is mandatory.
- UD-IQ2_XS (Dynamic 3.1) is the other candidate requant; its bytes vs our GSQ IQ2_XS were not established here.

### Gaps
- No quality comparison (KL/MMLU) between QNS S-small and GSQ-RCO IQ2_XS was found.
- The size of unsloth UD-IQ2_XS for Flash-Next vs our pack was not checked.

## Q7. Ranked list for pancake (estimated, unverified)

### Takeaway
Ranked by expected gain x applicability to pancake's two daily models and two GPUs. The top three are all on the Qwen3.6 path,
because the Flash-Next HIP path already carries the forks' attention/GDN/prefill work.

### Cited Findings
1. Port RandoTeam poly `STRATA_FATTN` + `STRATA_FDEC` (+ `bee07744` defaults) into pancake's poly - 3060/Qwen3.6: est. prompt
   +20-35% at 32-128K, decode +10-60% from 4K up (sources: [79b1e756](https://github.com/RandoTeam/4gbVramStrata/commit/79b1e756bae98068e3ebff321ea6c5e056694682),
   [67f60c12](https://github.com/RandoTeam/4gbVramStrata/commit/67f60c12ddb5f04b49efaa1a4f7477e88bc877ca),
   [aa862591](https://github.com/RandoTeam/4gbVramStrata/commit/aa8625919a1a2677d34e03ed96c6967ffeae3112)). Exactness: fp16
   tensor-core arithmetic, KL gate. HIP: not applicable as written. Replaces MOD_PLAN 8.8's own-kernel plan.
2. Qwen3.6 on the 7900 XT: either poly HIP (RandoTeam `44b58c6a` VMM/cuBLAS shims, `4636dbe3` VMM fixes, `6a991bce` q8 KV on HIP,
   `a70d9747`/`7cd8337d` HIP split value reduction, `5c0bd2c9` 32 cached experts per launch on HIP) plus a rocWMMA flash prompt
   kernel modelled on storm-ace `e4fc07ee`, or switch Qwen3.6 to the q36 engine with dongnh311 `62cc598f`/`848bfa9e`/`131df968`,
   storm-ace `4fe0284d`/`e4fc07ee`, maoyufeng `2f7020f2`, bahooo22 `bbc034d4`, Aaron-xx `69d28a34`/`535ce073`. Gain: enables the
   model on the main GPU (currently impossible); at long context the GQA decode fix alone is 2-2.4x decode in the q36 engine
   (dongnh311, storm-ace). — [RandoTeam 44b58c6a](https://github.com/RandoTeam/4gbVramStrata/commit/44b58c6ad11e7fef7c75e1d73353ee95210641d0),
   [storm-ace e4fc07ee](https://github.com/storm-ace/Strata/commit/e4fc07ee57c4d212f5720ce1e72bf41b7953b15f)
3. poly `STRATA_GDN_MMA` A/B on the 3060 (prefill only; +? unmeasured on sm_86) — [61b41e84](https://github.com/RandoTeam/4gbVramStrata/commit/61b41e84f4919229af21178ec22373d97b1c2d87)
4. Flash-Next 3060: zsariboga's exact prefill fusions (-3..-4% prompt time) and `STRATA_QSA_SPLIT_ATTN=1` A/B —
   [21107e9d](https://github.com/zsariboga/Strata/commit/21107e9d05364385527485fa6294dc5db4903976),
   [c87d6d79](https://github.com/svgop/Strata/commit/c87d6d7917491beea4be02548d24a4238d003dd6)
5. Flash-Next 7900 XT: `STRATA_WMMA_GEMM=1` A/B (+0-9% prompt), fp16 rocWMMA select arm or `STRATA_SELECT_SPLIT2` (~1-2% at
   64-128K prefill) — [9b49593f](https://github.com/troytse/Strata/commit/9b49593fdff41fb4ac5c7c461353bbcd320fa8b0),
   [33443c9f](https://github.com/xyzzing/Strata/commit/33443c9f514313030ab801c85ed9768dfa7136cc),
   [ebc12652](https://github.com/bsorensen110/Strata/commit/ebc12652417ba1d825b2729f5e9387a91b237e70)
6. Serve: trailing-reminder move (if opencode sends such turns) — [8b90ab80](https://github.com/Ben1078/Strata/commit/8b90ab8077ed943d7b2f0e3e31417608062885e2)
7. Requant study: Mirai QNS Flash-Next-S-small (CUDA only, ~-20% expert bytes) vs GSQ IQ2_XS — [8e75f6f9](https://github.com/alesha-pro/Strata/commit/8e75f6f99b6380b2f12317488578f5f555079ad9)
8. Q4 KV prompt attention on gfx11 int8 WMMA (port of gfx12 `STRATA_HIP_WMMA_Q4`) to free VRAM for experts at long context —
   [19fc0e48](https://github.com/bsorensen110/Strata/commit/19fc0e48847d89e145fe1d354cc29cf7aa198368)

Already in pancake (do not redo): HIP_WMMA, PROMPT_ATTN_IMMA, ATTN_DAILY, PA_FAST, PA_TRIM/PA_RPF, PF_FUSED, PF_GEMM, MMQ_KQ,
DENSE_MMQ (HIP), GDN_CHUNKED (gfx11)/GDN_TC (sm_80+)/GDN_HEAD/PP/KEYHEAD, SCORES_HALVE, top-k split, QSA causal block skip,
v_perm intrinsics, Q5_0 experts, IQ4_W4A8, kv q8/int8/q4(FWHT)/k8v4, kv-grow, poly attention cache growth and `--kv-q8`.
Already covered by earlier fork reviews: TurboQuant KV, MXFP4/dardeaw, PATTN_FLASH/Linsche, storm-ace's fork (batch5),
android-imdad's gfx1100 work (amd_rdna_ports).

### Inferences
- Expected order of value: (1) for the 3060 Qwen3.6 user, (2) for using Qwen3.6 at all on the 7900 XT; everything for Flash-Next
  on HIP is in the 1-5% range because pancake already took the forks' big prompt-path switches.

### Gaps
- All gains above are fork-reported on other hardware; none measured on the RX 7900 XT or RTX 3060 in this task (GPU busy).
- Fork READMEs were not mined beyond the QNS doc; commit messages carried the numbers.
