# disk-key: every STRATA_* switch classified for the conversation disk tier's key (round 2, 2026-11-11)

Branch `dev/p16b/disk-key` (from `dev/phase16-academic` 0dd7ef2e8). Owner files: `src/core/conversation_disk_tier.cpp`, its tests.
No GPU, no engine run: CPU-only build, two unit tests.

## What was wrong

The disk tier's key (`conversation_disk_key_env_names`, in `src/core/conversation_disk_tier.cpp`, not in `generate.cpp`: `generate.cpp`
only calls `conversation_disk_key_env`) was a hand-written list of 25 names plus 9 "only when set" names. Nothing said which of the
other 960 `STRATA_*` strings in `src/` and `include/` had been judged and which had been forgotten, so each new kernel switch could
silently share a cache folder with the runs that lack it (percache.md:73, MOD_PLAN 16 F-list item 8 and the top-10 list item 10 all
found one more by hand). The audit below found more than the two the brief named: the master switch `STRATA_DEFAULTS` and two
defaults that are "another summation order" (`STRATA_GR_V3`, `STRATA_PROMPT_ATTN_IMMA`, on by default on sm_120 / cc >= 80) were not in
the key, so `STRATA_DEFAULTS=026` or `STRATA_PROMPT_ATTN_IMMA=0` gave different numerics under the same key.

## What changed

- `src/core/conversation_disk_env.def` (new, the table): one line `STRATA_ENV("STRATA_X", Class, "why")` for each of the **994**
  distinct `"STRATA_..."` strings in `src/` and `include/` (984 in product files, 10 only in tests).
- `conversation_disk_tier.cpp` builds the two key lists from the table (`Key` = always in the key, `KeyIfSet` and `Unsure` = in the key
  when set and not empty) and exposes the table as `conversation_env_switches()` (declared in `conversation_disk_tier.hpp`).
  The key text starts `strata conversation disk 3` (was 2): entries written by the old key are never found, never mismatched.
  All 34 names that were keyed before are still keyed (checked by diff); the always-present list is the same 25 names.
- `tests/core/conversation_disk_env_test.cpp` (new): see "Tests".
- `tests/core/conversation_disk_tier_test.cpp`: the version check only (2 -> 3). The 97 checks are otherwise unchanged and pass.
- `CMakeLists.txt`: the existing `conversation_disk_tier_test` block moved out of the `if(STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP)`
  section (it needs no GPU, and a CPU-only build never ran it) next to `defer_rank_test`, with the new test beside it, and the new test
  added to the `cpu` label list. This is the only edit outside the owned files; it touches only those two blocks and one list entry.
- `generate.cpp` is **not** changed.

## What the key promises (and what it does not)

The key separates runs that *deliberately* compute the saved state differently (other routing, other precision, another summation
order, an approximate prefill). It cannot promise bit equality for the default path: a verify window's CPU-expert rows already round
differently by window size (`native_expert.cpp:438`, which is why `STRATA_IQ_MT_MIN=1` exists), GPU vs CPU experts round differently,
and the prompt chunk geometry changes rounding (`generate.cpp:10263`, KL <= 5e-5 for 32768 vs 16384). Those are placement and
geometry, which the project calls "exact"; they stay out (Speed) except where a note says otherwise.

The question asked of every name: with the same token ids, can the switch change a bit of the state the tier saves (K/V, recurrent
state, tails)? A switch that only changes *which tokens* get produced (sampling, drafting, stop rules, thinking caps) cannot make a
restored state wrong, because the cache is matched by token ids.

Classes: **Key** (always keyed), **KeyIfSet** (value-changing, never keyed before: keyed only when set, so runs without it keep
their key text), **Unsure** (may change the state, no bit-equality claim found or verifiable offline: treated as KeyIfSet, because a
stale hit is silent and a miss costs one prefill), **InConfig** (value-changing but in the config fingerprint), **Speed** (placement,
scheduling, drafting policy, or documented bitwise), **Inert** (diagnostics, test hooks, paths, limits, io, other engines).

## Source check

No paper: the "primary source" of each decision is the code and the project's own documentation. Evidence used, in this order:
the comment at the read site; the README switch table (`exact` / `lossy` column); `docs/DETAILS.md`, `NVIDIA_V100.md`,
`OLDER_GPUS.md`, `AMD_HIP.md`; the arch tables (`arch_defaults.cpp`, `nv_defaults.cpp`, `gfx12_extra.hpp`: "bitwise" vs "bit-changing");
parity tests that compare bits (`gdn_harness`, `qsa_*_parity`, `iq_*_parity`, `batch_rows_parity`, `gr_parity`, ...); and the batch-3
notes (`kern-hip.md`, `percache.md`). A `Speed` decision therefore rests on a **documented** bit-equality claim or on the switch not
touching arithmetic; nothing was re-measured (no GPU). Where no claim exists for an arithmetic variant the answer is `Unsure`.

## Decisions on the switches named in the brief

| switch | class | why |
| --- | --- | --- |
| `STRATA_EXP_MODE`, `STRATA_EXP9_DOWN`, `STRATA_EXP9_GU` (F14) | KeyIfSet | documented bitwise to mode 0 (native_expert_bench on one RTX 3090, `gpu_rewrite_parity_test` for the Q2_0 4-lane sum order) but not gated at run time and not seen on any other card. Cost of keying when set: none. Flip to Speed by editing one word if the owner trusts the bench |
| `STRATA_VERIFY_DEFER_CPU_BELOW/_TOP` | Key | were added by batch-1 (key v2); kept. MOD_PLAN 16 item 8 is closed |
| `STRATA_VERIFY_DEFER_CPU_FOLD` | Speed | only moves a row between pool passes (`defer_rank.hpp`); the existing key test asserts it stays out |
| `STRATA_HIP_GLOBAL_WEIGHTS` | not an environment variable | a compile-time macro (`-DSTRATA_HIP_GLOBAL_WEIGHTS=1`, `iq_kernels.cu`), bitwise by design; it cannot be in an env key. Two builds of one version with and without it share `STRATA_VERSION`, so share a key (see "Open")|
| `STRATA_Q2_FAST` (+ `_LEGACY`) | Speed | `kQ2Fast` runs `corr_form_probe`: it compares the fast kernel with the legacy one with `memcmp` on probe data at start and falls back to legacy on any mismatch, so the switch cannot change a bit that matters |
| `STRATA_EXPDEC_SWIGLU`, `STRATA_ROUTER_FUSE`, `STRATA_EXPDEC`, `_EXPDEC_DOWN`, `STRATA_QFUSE` | Key (kept) | all five are documented exact (README); they were keyed by batch-1 and removing a name from the key is the one change this audit does not make without the owner. If dropped, they are Speed |
| `STRATA_ROUTE_FAST` | Speed | `native_router.cu:223`: "every id and weight is bitwise route_multi's"; `native_multi_parity` passed (kern-hip.md) |
| `STRATA_SPEC_CUM`, `STRATA_DISCO`, `STRATA_SPEC_*`, `STRATA_SUFFIX_INDEX*` | Speed | choose which drafts are tried; every draft is verified, the saved state depends on the token ids only |
| `STRATA_BANDIT` + `_EXACT` | KeyIfSet (were) | its arms include lossy ones chosen by timings |
| `STRATA_BANDIT_MARGIN/_DECAY/_EPOCH/_RULE/_SEED/_SKIP/_TYPE` | KeyIfSet | change which arm (some lossy) is picked; `_LOG` is Inert |
| `STRATA_PARK_*` (TTL, TRUNCATE, ASYNC, ...) | Speed | cache policy: what is parked and when, never a value |

## Findings beyond the brief

1. **`STRATA_DEFAULTS=026`** (`defaults_026.hpp`) turns off the code-resolved defaults `STRATA_GR_V3` (stream-split HC read, "another
   summation order", ON for a discrete sm_120 card over 13 GiB) and `STRATA_PROMPT_ATTN_IMMA` (int8 tensor-core prompt attention,
   "another summation order", ON for every CUDA card of cc >= 80), plus `PLE_AUTO_RAM`. With both variables unset the key was the
   same for both settings. `GR_V3`, `PROMPT_ATTN_IMMA`, `DEFAULTS` and `SM120_PERF` are now KeyIfSet. The master gates
   `GFX12_DEFAULTS`, `GFX1151_DEFAULTS`, `IGPU_DEFAULTS`, `IGPU_PROMPT_SET`, `SM121_PERF`, `NV_KQ_DEFAULT`, `AMD_COMMUNITY_TUNING`
   (which makes `SELECT_SIMT=2`, another summation order, the default on gfx103x) are KeyIfSet for the same reason.
2. The arch tables set several **bit-changing** switches by default through `setenv` (read by the key afterwards): `PF_FUSED`,
   `PF_GEMM`, `HIP_WMMA`, `SELECT_WMMA` (+ `SELECT_ALT_MIN_TOKENS`), `PA_FAST`, `HC_UPMIX`, `PREFILL_NVFP4`, `MMQ_KQ` (first-token KL
   0.02-0.035). None was keyed; a user turning one off (`STRATA_PF_GEMM=0`) kept the key. All KeyIfSet now.
3. `STRATA_CPU_YMM` (the 256-bit CPU expert kernel: "the sum order differs ... last-bit differences"), `STRATA_IQ_MT_MIN`,
   `IQ3S_MT1`, `IQ512_FUSED_MAX/_TYPES` ("different rounding"), `Q2_BITPLANE` ("lossy"), `GDN_CONVL2` (not bitwise on CUDA sm_120),
   `GDN_CHUNKED` / `_MIN` / `GDN_TC*` (other bits; gfx12 turns the chunked scan on by length through `GFX12_DEFAULTS`), `BF16_TC`,
   `DENSE_MMQ`, `PREFILL_CPU_SHARE` (+ `_MAX`: prompt experts on the CPU, mean KL 0.004-0.011, default on gfx12), `PREFILL_HELP`,
   `PREFILL_BF16X2`, `PREFILL_EQUAL`, `PF_SMALL` (decides whether `PF_FUSED` runs, `generate.cpp:11334`), the whole `STRATA_KVA*`
   family (approximate prefill of the late layers) and `STRATA_TAIL_SKIP_MAX_CACHED` (limit of the keyed `ROUTE_TAIL_SKIP`).
4. `STRATA_KV_ROT` changes the saved K/V but is already in the config fingerprint (`session_identity`): class InConfig, not keyed twice.
5. Debug switches that change values are keyed when set (`DBG_ROUTE_OVERRIDE`, `NAN_INJECT`); their value is a path or a count, so a
   different file under the same name is not seen.

## Open (not fixed here; the lead decides)

- **Engine identity.** The config fingerprint holds `STRATA_VERSION` only. `strata-poly`'s own disk key adds the executable's size and
  a hash of its bytes (`src/poly/main.cpp` `disk_key`). With the same version string, a rebuilt engine (new kernels, a compile-time
  macro such as `STRATA_HIP_GLOBAL_WEIGHTS`) reads another build's states. Hashing the executable once at start would close this for
  every build-time difference and for every dev build. Not done: it touches start-up and session identity.
- **CLI arguments that are not in the fingerprint.** `--prefill N` / the auto chunk (rounding: see above), `--adapt-swaps`,
  `--pcie-frac`, `--ram-keep-gib` (placement) and the auto decisions made from free VRAM (`small_card_prompt_path`,
  `pf_fused_override`, the chunk scan) are not seen by an environment key. Chunk geometry is the only one of these that the project
  itself calls a rounding change; it is not covered, apart from the `Unsure` chunk-ceiling switches that name it.
- `metal/`, `sycl/`, `serve/` (Python), `apps/` and `tools/` read other `STRATA_*` names; they are not scanned (the disk tier is built
  only into the CUDA/HIP `strata_engine`). `strata-dsv41` and `strata-poly` have their own stores and keys (their names are Inert here).
- 101 names are `Unsure`: the cheap way to shrink the list is a bit-compare of each on a GPU (`restore_check.py`-style, same prompt,
  switch on/off): every one that compares equal moves to Speed by editing one word.
- Names assembled at run time (`"STRATA_" + x`) would not be seen by the scan; none exists in `src/` or `include/` today (grep).

## Tests

Build: `cmake -S . -B build-cpu -G Ninja -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON
-DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=<pinned llama.cpp-src>`, then `ninja -j2 conversation_disk_tier_test conversation_disk_env_test`
under `systemd-run --user --scope -p MemoryMax=6G`.

- `conversation_disk_tier_test`: **97 checks passed** (the existing test, unchanged but for the key version).
- `conversation_disk_env_test` (new): **4209 checks passed**, 994 switches. It
  1. scans every `.cpp .hpp .h .cu .cuh .inc .inl .c .cc .mm` under `src/` and `include/` for `"STRATA_NAME"` string literals (not the
     table itself) and fails with the file name for every one that has no line in the table, and for every line whose string is gone;
  2. for each line checks the class against the real key: Key changes the key value by value, KeyIfSet / Unsure change it when set
     (and set-but-empty is unset), Speed / Inert / InConfig never change it;
  3. pins the decisions above (F14, the master switch, the other-summation-order defaults, the arch-table bit changers, the old key
     names, FOLD and Q2_FAST out, KV_ROT in the fingerprint).
  Shown to fail: a line `std::getenv("STRATA_ZZZ_NEW_SWITCH")` appended to `src/core/layer.cpp` gives `ADD A LINE for: STRATA_ZZZ_NEW_SWITCH
  (src/core/layer.cpp)`; a stale line gives `REMOVE: STRATA_ZZZ_STALE` (both reverted).
- `ctest -L cpu -R conversation_disk`: both pass (the new test is in the `cpu` label list).
- Not run: the GPU build (the change is host code; `strata_engine` compiles `conversation_disk_tier.cpp` and the new `.def` is next to
  it), the engine, any model.

## Adding a switch

Add one line to `src/core/conversation_disk_env.def`. If the switch can change a bit of the saved state: `KeyIfSet`; if you cannot
tell: `Unsure` and say why; if it only moves work or documented bitwise: `Speed`; otherwise `Inert`. Moving a name into or out of
`Key` changes every key: bump the version line in `conversation_disk_key()`.

## The audited table

Every name below has one line in `src/core/conversation_disk_env.def` (that file is the table the tests read; this section is a copy
for reading). Counts: Key 25, KeyIfSet 95, Unsure 101, InConfig 1, Speed 499, Inert 273.

### Key (25, as before the audit)

| switch | why |
| --- | --- |
| `STRATA_CPU_PRESUM` | host pre-sums CPU rows (not bit-identical) |
| `STRATA_EXPDEC` | decode expert kernels family; documented exact, kept in the key conservatively (batch-1) |
| `STRATA_EXPDEC_DOWN` | documented exact; kept conservatively (batch-1) |
| `STRATA_EXPDEC_SWIGLU` | documented exact; kept conservatively (batch-1) |
| `STRATA_GR_RDNA_A8` | int8 activations + dot4 in the Q8 read (lossy) |
| `STRATA_GR_RDNA_Q8` | int8 HC read on RDNA (lossy) |
| `STRATA_HC_Q8` | int8 image of hyper-connection projections (answers differ) |
| `STRATA_HC_Q8_INJECT` | int8 HC image of the inject rows |
| `STRATA_HC_REQ8` | HC projections loaded as int8 (changes the output) |
| `STRATA_HIP_PROMPT_F16` | FP16 rounding of the prompt path numbers |
| `STRATA_PF_DM16` | down rows to the combine as FP16 |
| `STRATA_PF_F16` | prompt experts on F16 matrix-core kernels (rounding differs from int8) |
| `STRATA_PF_FUSED_F16` | F16 products in the fused prompt experts (rounding level) |
| `STRATA_PRESUM_VRAM` | pre-sums written to VRAM (not bit-identical) |
| `STRATA_QFUSE` | documented exact (arch table); kept conservatively (batch-1) |
| `STRATA_ROUTER_FUSE` | documented exact; kept conservatively (batch-1) |
| `STRATA_ROUTE_PRIOR` | residency-biased routing (lossy) |
| `STRATA_ROUTE_PRIOR_DEPTH` | route-prior bonus graded by window position |
| `STRATA_ROUTE_RESIDENT` | residency-biased routing after the router (lossy) |
| `STRATA_ROUTE_RESIDENT_MTP` | the draft layer routed like ROUTE_RESIDENT |
| `STRATA_ROUTE_RESIDENT_RANKS` | rank band of ROUTE_RESIDENT |
| `STRATA_ROUTE_TAIL_SKIP` | skips missed tail experts (default 7 on CUDA serve; changes the output) |
| `STRATA_VERIFY_DEFER_CPU` | CPU rows deferred a window: lossy |
| `STRATA_VERIFY_DEFER_CPU_BELOW` | deferral threshold |
| `STRATA_VERIFY_DEFER_CPU_TOP` | deferral top-k |

### KeyIfSet (95; value-changing, new to the key except the 9 older ones)

| switch | why |
| --- | --- |
| `STRATA_AMD_COMMUNITY_TUNING` | makes SELECT_SIMT=2 the default on gfx103x (other summation order, KL 0.0256) and loads community GEMM tables |
| `STRATA_BANDIT` | arms chosen by timings, some lossy |
| `STRATA_BANDIT_DECAY` | arm statistics |
| `STRATA_BANDIT_EPOCH` | arm choice cadence |
| `STRATA_BANDIT_EXACT` | which arms the bandit may use |
| `STRATA_BANDIT_MARGIN` | how much a lossy arm must win by |
| `STRATA_BANDIT_RULE` | arm choice rule |
| `STRATA_BANDIT_SEED` | arm choice seed |
| `STRATA_BANDIT_SKIP` | arms excluded |
| `STRATA_BANDIT_TYPE` | arm set |
| `STRATA_BF16_TC` | BF16 projections on FP16 tensor cores (not bitwise to cuBLAS) |
| `STRATA_CPU_YMM` | 256-bit CPU kernel: different sum order (last-bit differences) |
| `STRATA_DBG_ROUTE_OVERRIDE` | debug: replaces the routing of listed layers from a file (value is a path) |
| `STRATA_DEFAULTS` | STRATA_DEFAULTS=026 turns off GR_V3 and PROMPT_ATTN_IMMA defaults (other summation orders), PLE_AUTO_RAM, gfx12 table |
| `STRATA_DENSE_MMQ` | dense prompt projections through int8 MMQ (greedy text can diverge) |
| `STRATA_EXP9_DOWN` | F14 per-format override of EXP_MODE=9 (42:10 = Q2_0 down on 4 lanes) |
| `STRATA_EXP9_GU` | F14 per-format override of EXP_MODE=9 |
| `STRATA_EXP_MODE` | F14 per-format expert layouts: bitwise only per native_expert_bench on one RTX 3090, not gated at run time |
| `STRATA_GDN_CHUNK` | chunked GDN recurrence (opt-in) |
| `STRATA_GDN_CHUNKED` | chunked GDN scan, FP16 operands (other bits); length-gated default on gfx12 |
| `STRATA_GDN_CHUNKED_MIN` | length from which the chunked GDN scan applies (default 30000 on gfx12) |
| `STRATA_GDN_CONVL2` | conv + L2 norm kernel: not bitwise on CUDA sm_120 (#1522) |
| `STRATA_GDN_REC_WARP` | chain-split GDN recurrence (changes the FP32 summation order) |
| `STRATA_GDN_TC` | tensor-core GDN scan (opt-in) |
| `STRATA_GDN_TC_NP` | GDN TC split passes (other bits) |
| `STRATA_GFX1151_DEFAULTS` | gates the gfx1151 table and code defaults (PF_FUSED_Q50, PF_FUSED_KQ routes) |
| `STRATA_GFX12_DEFAULTS` | gates the gfx12 value-changing table (SELECT_WMMA from 28,672 tokens, MMQ_KQ, chunked GDN scan) |
| `STRATA_GR_RDNA` | RDNA HC read kernels, another summation order |
| `STRATA_GR_SPLIT` | K-split HC down: multi-token read no longer bitwise to single-token calls |
| `STRATA_GR_V3` | stream-split HC read, another summation order; default ON on sm_120 >13 GiB |
| `STRATA_HC_Q8_ALL_STAGES` | which stages read the int8 HC image (modifies HC_Q8) |
| `STRATA_HC_UPMIX` | HC up projection + mix in one kernel (bit-changing, KL gated) |
| `STRATA_HIP_WMMA` | prompt attention on matrix cores (bit-changing; default ON on gfx12 Linux) |
| `STRATA_HIP_WMMA_MIN_CAP` | WMMA threshold (kernel choice) |
| `STRATA_HIP_WMMA_MIN_Q` | WMMA threshold (kernel choice) |
| `STRATA_IGPU_DEFAULTS` | gates the iGPU prompt set (PF_GEMM, HIP_WMMA, PA_FAST, SELECT_WMMA, HC_UPMIX: bits change) |
| `STRATA_IGPU_PROMPT_SET` | turns the iGPU prompt set on/off |
| `STRATA_IQ3S_MT1` | changes a lone token's IQ3_S rounding |
| `STRATA_IQ512_FUSED_MAX` | different rounding (N) vs default rows |
| `STRATA_IQ512_FUSED_TYPES` | types of IQ512_FUSED_MAX |
| `STRATA_IQ_MT_MIN` | multi-token CPU IQ kernel from N tokens (different rounding vs ggml one-token dot) |
| `STRATA_KVA` | approximate prefill plan (research): the late layers of a cold prompt are approximated |
| `STRATA_KVA_ANY_MODEL` | allows KVA on any model |
| `STRATA_KVA_GDN_NORM` | KVA GDN norm handling |
| `STRATA_KVA_MIN_PROMPT` | prompt length that triggers KVA |
| `STRATA_KVA_MODE` | KVA approximation mode |
| `STRATA_KVA_ONLY` | which late layers are approximated |
| `STRATA_KVA_ORACLE` | KVA oracle input (replaces late-layer input) |
| `STRATA_KVA_ORACLE_FP32` | KVA oracle precision |
| `STRATA_KVA_PROJECTOR` | KVA projector file |
| `STRATA_KVA_SKIP` | what an approximated chunk skips |
| `STRATA_KVA_SPLIT` | first approximated layer |
| `STRATA_KVA_SUFFIX` | exact tail length |
| `STRATA_MMQ_KQ` | K-quant experts on MMQ (first-token KL ~0.03) |
| `STRATA_NAN_INJECT` | test hook: injects NaN into the state |
| `STRATA_NV_KQ_DEFAULT` | sets MMQ_KQ=1 on NVIDIA (first-token KL 0.024-0.035) |
| `STRATA_PA_FAST` | single FP16 q and p prompt attention (rounding level) |
| `STRATA_PA_V2T` | Turing v2 attention with register staging (changes answers) |
| `STRATA_PF_FUSED` | fused int8 prompt experts (rounds differently from MMQ) |
| `STRATA_PF_FUSED_KQ` | K-quant fused experts (output differs from the FP16 path) |
| `STRATA_PF_FUSED_Q2N` | Q2_0 fused prompt kernel (own bits, MMQ otherwise) |
| `STRATA_PF_FUSED_SM75` | Turing fused kernels (changes answers) |
| `STRATA_PF_FUSED_WS` | F16 products of the fused path (same rounding as F16) |
| `STRATA_PF_GEMM` | prompt projections on WMMA GEMM (bit-changing) |
| `STRATA_PF_HCDOWN` | HC down by WMMA kernel (first-token KL doubled at 16K) |
| `STRATA_PF_SMALL` | turns the small-card prompt path (unfused experts, layer-major walk) on/off; decides whether PF_FUSED (rounds differently from MMQ) runs |
| `STRATA_PF_SWITCH_MIN_T` | chunk length from which the rounding-level prompt switches apply |
| `STRATA_PF_WMMA` | WMMA prompt experts with FP16 activations (not MMQ rounding) |
| `STRATA_PREFILL_BF16X2` | differs from the old path in the BF16 low-part term |
| `STRATA_PREFILL_CPU_SHARE` | prompt experts computed on the CPU pool (mean KL ~0.004-0.011); default on gfx12 |
| `STRATA_PREFILL_CPU_SHARE_MAX` | limit of the CPU share |
| `STRATA_PREFILL_EQUAL` | equal-size chunks: changes rounding |
| `STRATA_PREFILL_HELP` | idle stage computes a share of streamed experts (not bit-identical) |
| `STRATA_PREFILL_HELP_FRAC` | fixes the share of the idle stage in PREFILL_HELP |
| `STRATA_PREFILL_NVFP4` | NVFP4 prompt MMQ with FP4 activations (rounds differently) |
| `STRATA_PROMPT_ATTN_IMMA` | int8 tensor-core prompt attention, another summation order; default ON on cc>=80 |
| `STRATA_PROMPT_ATTN_OLD` | old prompt attention kernel (not bitwise to the current one) |
| `STRATA_PROMPT_ATTN_Q4` | Q4_0-KV prompt attention kernel |
| `STRATA_PROMPT_ATTN_V1` | v1 prompt attention (FP32-level difference) |
| `STRATA_Q2_BITPLANE` | lossy bit-plane Q2_0 dot |
| `STRATA_QSA_SPLIT_ATTN` | decode attention split across blocks (rounding level) |
| `STRATA_ROUTE_PRIOR_KEEP` | route-prior guard |
| `STRATA_ROUTE_PRIOR_KEEP_P` | route-prior guard |
| `STRATA_ROUTE_PRIOR_MARKS` | route-prior guard |
| `STRATA_ROUTE_PRIOR_ON_PROMPT` | bonus kept on while the prompt is read |
| `STRATA_ROUTE_PRIOR_PROTECT` | route-prior guard |
| `STRATA_ROUTE_PRIOR_SCOPE` | route-prior guard |
| `STRATA_SELECT_ALT_MIN_TOKENS` | context length from which SELECT_WMMA applies (gfx12 default 28672) |
| `STRATA_SELECT_OLD` | warp scorer instead of the 3xTF32 tensor-core one (not bitwise) |
| `STRATA_SELECT_SIMT` | FP32 tiled scorer, another summation order |
| `STRATA_SELECT_WMMA` | matrix-core block scorer (not bitwise, cutoff cells flip) |
| `STRATA_SM120_PERF` | gates sm_120 defaults incl. GR_V3 (another summation order) |
| `STRATA_SM121_PERF` | gates the sm_121 table: PF_FUSED, PREFILL_NVFP4=w4a4 (round differently) |
| `STRATA_TAIL_SKIP_MAX_CACHED` | limit of ROUTE_TAIL_SKIP (decides which tails are skipped) |
| `STRATA_VERIFY_ECAP` | caps CPU experts only draft rows need (lossy) |

### Unsure (101; keyed when set)

| switch | why |
| --- | --- |
| `STRATA_ATTN_V3` | decode attention v3 |
| `STRATA_DBG_FORCE_SLOW_LAYER` | debug: forces the unfused path for layer 0 |
| `STRATA_EXPDEC_D20` | EXPDEC tuning (IQ4_NL down on EXPDEC) |
| `STRATA_EXPDEC_DR` | EXPDEC down rows per lane |
| `STRATA_EXPDEC_DRB` | EXPDEC down rows per block |
| `STRATA_EXPDEC_MINB` | EXPDEC launch bounds |
| `STRATA_EXPDEC_MINBLK` | EXPDEC rows per block rule |
| `STRATA_EXPERT_NIBBLE` | 4-bit expert layout (OLDER_GPUS.md:242); no bitwise claim found |
| `STRATA_EXPERT_NIBBLE_FMTS` | formats on the 4-bit expert layout (see EXPERT_NIBBLE) |
| `STRATA_EXPERT_NIBBLE_SKIP` | layers/formats kept off the 4-bit expert layout (see EXPERT_NIBBLE) |
| `STRATA_FORCE_AVX2` | forces the AVX2 CPU kernels; kernel choice decides the sum order |
| `STRATA_FORCE_ISA` | test knob, forces a CPU ISA tier |
| `STRATA_FUSED_SWIGLU_Q81` | fused SwiGLU + q8_1 in the shared expert |
| `STRATA_FUSE_CONVAB` | GDN conv + a/b projections in one launch |
| `STRATA_FUSE_HEAD_GR` | head HC read fused |
| `STRATA_GDN_COMMIT_SPLIT` | GDN commit kernel choice |
| `STRATA_GDN_FUSE` | GDN fusion variant |
| `STRATA_GDN_TC_FUSE` | GDN TC y16 fusion |
| `STRATA_GDN_TC_TILES` | GDN TC tile count |
| `STRATA_GDN_W12_LB` | GDN launch-bounds variant |
| `STRATA_GFX1151_CHUNK` | chunk geometry changes the rounding (generate.cpp:10263, percache.md: restore equals a re-read only for the same geometry; 32768 vs 16384 measured KL <= 5e-5) |
| `STRATA_GFX12_CHUNK` | chunk geometry changes the rounding (generate.cpp:10263, percache.md: restore equals a re-read only for the same geometry; 32768 vs 16384 measured KL <= 5e-5) |
| `STRATA_GFX906_ATTN_QUERY_SWIZZLE` | gfx906 attention variant |
| `STRATA_GFX906_ATTN_REDUCE12` | gfx906 attention reduce |
| `STRATA_GROUPED_V1` | grouped expert kernel generation |
| `STRATA_GR_DOWN_MAX4` | HC down variant |
| `STRATA_GR_DOWN_TG` | HC down thread group |
| `STRATA_GR_NORM_FAST` | HC read norm variant |
| `STRATA_GR_Q32_PL` | GR q32 pipeline variant |
| `STRATA_GR_R2` | v3 read rows-per-warp variants (generic kernels as before) |
| `STRATA_GR_RDNA_RW` | RDNA HC read variant |
| `STRATA_GR_UNFUSED` | unfused HC read in the prompt path |
| `STRATA_HC_SPLIT` | HC read split |
| `STRATA_HIPBLASLT_TABLE` | packaged GEMM solution table on/off |
| `STRATA_HIPBLASLT_TUNING` | GEMM solution table (path value) |
| `STRATA_HIP_ROUTER_FAST` | gfx906 router kernel |
| `STRATA_HIP_ROUTER_OLD` | old router kernel |
| `STRATA_HIP_WMMA_Q4` | Q4_0-KV prompt attention on matrix cores (opt-in port) |
| `STRATA_INDEXER_PER_TOKEN` | indexer per token |
| `STRATA_IQ4_W4A8` | W4A8 (int8 activations) IQ4 prompt experts |
| `STRATA_LM_HIP` | HIP LM head variant |
| `STRATA_MMQ_WS` | MMQ with weight staging (opt-in) |
| `STRATA_MMVQ_IL_120` | IL row table for big 12.0 cards |
| `STRATA_MMVQ_IL_Q3K` | IL kernel for Q3_K |
| `STRATA_MMVQ_IL_T1` | IL kernel for one token |
| `STRATA_MMVQ_IL_WIDE` | wide IL kernels |
| `STRATA_MMVQ_R` | rows per wave; README exact but phase15.md: MMVQ_R=2 + RDNA_MMVQ_WAVE=3 is a rounding change |
| `STRATA_NO_IQ256` | turns off the AVX2 multi-token IQ kernels (sum order vs ggml dot not claimed) |
| `STRATA_NO_IQ4NL` | IQ4_NL multi-token kernel vs ggml single-token dot |
| `STRATA_NO_IQ512` | AVX-512 IQ kernels (16-lane sums) vs ggml dot |
| `STRATA_NO_MMVQ_ROWS2` | rows-2 MMVQ layout off |
| `STRATA_NO_TQ_KERNELS` | TQ CPU kernels off |
| `STRATA_NV_CHUNK` | chunk geometry changes the rounding (generate.cpp:10263, percache.md: restore equals a re-read only for the same geometry; 32768 vs 16384 measured KL <= 5e-5) |
| `STRATA_OLD_GROUPED` | old Q2_0 grouped kernel kept for A/B; no bitwise claim found |
| `STRATA_OLD_Q80_SCALED` | old Q8_0 scaled activation quantizer |
| `STRATA_ONE_TOKEN_COMMIT` | one-token commit path (HIP opt-in) |
| `STRATA_PF_F16_BN` | F16 expert tile width |
| `STRATA_PF_FUSED_IQ2S` | IQ2_S fused kernel |
| `STRATA_PF_FUSED_NATIVE` | native fused kernel |
| `STRATA_PF_FUSED_Q50` | Q5_0 fused kernel (default on for F16 on gfx1151) |
| `STRATA_PF_FUSED_TILE` | fused tile shape |
| `STRATA_PF_Q8D_F16` | F16 q8 down variant |
| `STRATA_PF_WMMA2` | second WMMA expert kernel |
| `STRATA_PREFILL_AUTO_MAX` | chunk geometry changes the rounding (generate.cpp:10263, percache.md: restore equals a re-read only for the same geometry; 32768 vs 16384 measured KL <= 5e-5) |
| `STRATA_PREFILL_HOST_PLAN` | chunk shape (generate.cpp:10454: only the chunk shape, i.e. rounding) |
| `STRATA_PREFILL_MMQ` | int8 MMQ experts for the prompt (on by default on some cards) |
| `STRATA_PREFILL_PLAN_TOKS` | chunk geometry changes the rounding (generate.cpp:10263, percache.md: restore equals a re-read only for the same geometry; 32768 vs 16384 measured KL <= 5e-5) |
| `STRATA_PROMPT_ATTN_Q4_TC` | Q4 tensor-core prompt attention |
| `STRATA_Q2_WARP` | Q2_0 warp kernel |
| `STRATA_Q6P_GRID` | packed Q6 kernel grid |
| `STRATA_Q8_GATHER_WIDE` | wide Q8 KV gather |
| `STRATA_Q8_SM60` | packed Q8_0 planes (Pascal GP100 opt-in) |
| `STRATA_Q8_SM60_BPS` | Q8_SM60 blocks per step |
| `STRATA_QFUSE_HC_LOCAL` | HC q-fuse local variant |
| `STRATA_QIL` | interleaved q8_1 activations choice |
| `STRATA_QSA_CLUSTER` | QSA scorer cluster variant |
| `STRATA_QSA_FUSE_GATE` | QSA gate fusion |
| `STRATA_QSA_WARP` | QSA prompt kernel warp variant |
| `STRATA_QUANT_GATHER` | quantize gather variant |
| `STRATA_ROCBLAS_TUNING` | GEMM solution table: another solution can mean another K order (path value) |
| `STRATA_ROPE_TABLE` | RoPE table source (sincos values) |
| `STRATA_ROUTER_BLOCK` | router block variant |
| `STRATA_ROUTE_PER_TOKEN` | router computed per token vs batched |
| `STRATA_ROUTE_PROJ_PER_TOKEN` | router projection per token |
| `STRATA_S2_SWIGLU_Q8` | fused SwiGLU + q8 in the S2 grouped kernel |
| `STRATA_SCORES_HALVE` | QSA scorer variant |
| `STRATA_SIMT2_TILE` | simt2 scorer tile (simt2 is another summation order) |
| `STRATA_SPLIT_SMALL_OWN` | chunk geometry changes the rounding (generate.cpp:10263) |
| `STRATA_TOPK_CAPACITY_GUARD` | capacity bound of the top-k launch; no bitwise claim found |
| `STRATA_TOPK_SAMPLE` | top-k sampling variant of the selector |
| `STRATA_TOPK_SPLIT` | top-k split |
| `STRATA_TOPK_STREAM` | top-k stream variant |
| `STRATA_TQ_ISA` | TQ CPU kernel ISA tier |
| `STRATA_TQ_ROWCH` | TQ rows per chunk |
| `STRATA_TQ_STAGE` | TQ staging |
| `STRATA_TQ_SUB8` | TQ sub-8 kernels |
| `STRATA_VNNI_TIER` | CPU row-kernel tier (VNNI forms; "same bits" only claimed for NO_AVXVNNI) |
| `STRATA_W4A8_CFG` | W4A8 kernel config |
| `STRATA_WMMA_BF16` | BF16 WMMA GEMM |
| `STRATA_WMMA_GEMM` | WMMA GEMM |
| `STRATA_WS_FAST` | weight-staged fast path (opt-in) |

### InConfig (1)

| switch | why |
| --- | --- |
| `STRATA_KV_ROT` | covered by the config fingerprint (session_identity: c.kv_rot) |

### Speed (499; names without the `STRATA_` prefix, grouped by why)

- **bitwise: documented (docs/README/code comment/parity test)** (155): `ARGMAX_MULTI` `ARGMAX_ROWS` `ARGMAX_ROWS_SM70` `ATTN_DAILY` `ATTN_LANECELL` `ATTN_MERGE_V2` `ATTN_PRE75` `B6_MMVQ_ROWS` `BATCH_ATTN_ROWS` `BATCH_GDN_GROUPS` `BATCH_SELF_COMMIT` `COMBINE_CPU` `COMBINE_POLL1` `COMMIT_BATCH` `CPU_IQ_ALL_EXACT1` `CVEC_FUSE` `DEC_BATCH` `DENSE_F32` `DEQUANT_SIDE` `DEQUANT_WIDE` `DF2_GEMM` `DF2_GEMM4` `DF2_HEAD_IL` `DF_PDL` `DF_PLE` `DOWN_WS` `EMBED_VRAM` `EXPDEC_R` `EXPDEC_RB` `EXPERT_V2` `EXPERT_V2K` `FUSE_DECODE` `GDN_CONV_SERIAL` `GDN_HEAD` `GDN_KEYHEAD` `GDN_NOY` `GDN_PIPELINE` `GDN_PP` `GDN_REC_HEADS` `GDN_SPLIT` `GFX1151_EXP8` `GROUPED_PAIR_MIN_HITS` `GR_FAST` `GR_FUSED` `GR_INJ_UP` `GR_Q32_G` `GR_Q8_PFS` `GR_UP75` `GR_UP_WIDE` `GR_V3_PF` `GU_SWIGLU` `HCD_EXACT` `HC_PERSIST` `HEAD_MIX_MULTI` `HIPBLASLT_DESC_CACHE` `HIP_SWIGLU_FUSED` `IL_PAD` `IQ256_GATHER` `IQ2S_PAIR` `IQ4XS512_MIN` `IQ_GATHER` `IQ_LUT_DUP` `IQ_SIGN_FAST` `IQ_STAGE_GRID` `IQ_STAGE_GRID_MMVQ` `IQ_STAGE_TABLES` `KQ256` `KQ_SPLIT` `LFUSE` `LFUSE_GATE` `LFUSE_PAIR` `LOOKAHEAD_GPU` `MMQ_GROUP_SORT` `MMQ_RESIDENT_SORT_NE` `MMVF_AUX` `MMVF_ROWS` `MMVQ_IL` `MMVQ_IL_89` `MMVQ_IL_PF` `MMVQ_IL_ROWS` `MMVQ_IL_SM80` `MMVQ_WAVE` `MULTI_BLOCK_ARGMAX` `NIB_GU_PROTO` `NORM_ROPE` `NO_AVXVNNI` `NO_IQ128` `NO_IQ4XS512` `NO_IQ512X1` `NO_MULTI_GR` `NO_NORM_ROPE` `NO_NVFP4_256` `NO_NVFP4_512` `NO_Q50_512` `NO_Q8K_AVX1` `NO_Q8K_AVX2` `NO_SUB16_GU` `OLD_IQ_MMVQ` `PATTN_SM60` `PA_RPF` `PA_TRIM` `PA_V1_PREFETCH` `PF_BATCH` `PF_BK64` `PF_FUSED_NT` `PF_PAD` `PF_W16_CACHE_MIB` `PLE_BATCH` `PLE_COMPACT` `PLE_COPY_STREAM` `PLE_PIPE` `Q2_FAST` `Q2_LEGACY` `Q50_512_MIN` `Q5K_HEAD` `Q6_PACKED` `Q8_PACKED` `QFUSE_HC` `QSA_EARLY_EXIT` `QSA_FUSE_NR` `QSA_ROWS` `QSA_SLOTS` `RDNA2_SGEMM` `RDNA_KQ_ALL` `RDNA_MMVQ_WAVE` `RDNA_Q6K` `ROUTE_PUBLISH` `SCORES_DX` `SCORES_MULTI` `SCORES_THREAD` `SELECT_LONG` `SELECT_LONG_AT` `SELECT_TWO_PASS` `SELECT_TWO_PASS_MIN` `SIDE_MIXER` `SM60_PERF` `SM60_TABLE` `SM61_TABLE` `SM70_TABLE` `SM75_EXP` `SM75_IL` `SM75_PERF` `SM86_PERF` `SMALL_UNR` `TC_SM75` `TOPK_ACTIVE_ANY` `TOPK_MULTI` `TOPK_MULTI_N` `TOPK_OLD` `TOPK_REG_MIN_BLOCKS` `TSUM` `VERIFY_QDEDUP` `WAIT_COPY256` `WAIT_FUSE` `XN_F16`
- **placement/tiering of experts (same math, GPU vs CPU)** (79): `ADAPT_ASYNC` `ADAPT_ASYNC_BATCH` `ADAPT_ASYNC_WAIT` `ADAPT_CALLBACK_WARMUP` `ADAPT_CHECKRES` `ADAPT_EVICT_SYNC` `ADAPT_FETCH` `ADAPT_FETCH_MARGIN` `ADAPT_FETCH_MIN` `ADAPT_JOB_CPU` `ADAPT_KMISS` `ADAPT_LAG` `ADAPT_NOWAIT` `ADAPT_OVERLAP` `ADAPT_OVERLAP_RESIDENT` `ADAPT_OVERLAP_SLOW_US` `ADAPT_PERSIST` `ADAPT_PICK_INLINE` `ADAPT_SIGNAL` `ADAPT_SWAPS_AUTO` `CALIBRATE` `CALIBRATE_CACHE` `CALIBRATE_MIN_SHARE` `COLD_OVERLAP` `DISJOINT_ADAPT` `EARLY_REMOTE_CONTEXTS` `EXCHANGE_ROTATE` `FETCH_ADMIT` `FETCH_PVM` `FETCH_THREADS` `FILL_AHEAD` `FS_ADMIT` `FS_AHEAD` `FS_BUDGET` `FS_DEPTH` `FS_SLOTS` `FS_VERIFY` `IDLE_ADAPT` `PARTIAL_PIN` `PARTIAL_PIN_GIB` `PCIE_AUTO_CANON` `PCIE_AUTO_SPLIT` `PCIE_BALANCE` `PCIE_FRAC_DEFAULT` `PCIE_K_RULE` `PCIE_MIN1` `PCIE_OFF` `PCIE_SHARE_MAX` `PCIE_START` `PEER_DIRECT` `PEER_HOT` `PEER_HOT_AT` `PEER_NO_P2P` `PIN_BY_NEED` `PIN_BY_NEED_SKIP` `PIN_GUARD` `PIN_PARK_TOKENS` `PIN_RESERVE_GIB` `PIN_RESERVE_MIB` `PIPE_ADAPT_EVERY` `PROMO_CHUNK` `RAM_ADAPT` `RAM_ADAPT2` `RAM_ADAPT_CACHED_W` `RAM_ADAPT_PER_LAYER` `RAM_ADAPT_PROMPT_W` `RAM_ADAPT_QUEUE` `RAM_ADAPT_SKIP_TOP` `REFILL_BLOCKING` `REFILL_PARTS` `REFILL_SERIAL` `RESIDENT_HEADROOM_GIB` `RESIDENT_PIN` `SPEC_ADAPT_EVERY` `SPEC_ADAPT_SM86` `SPEC_ADAPT_WARM` `STAGE_PIN` `STAGE_TRIM` `VERIFY_COMPLEMENT`
- **drafting/stop policy: the tokens change, the state of given tokens does not** (67): `BATCH_DF2` `BATCH_DF2_CONF` `BATCH_DF2_DEFER` `BATCH_DF2_MAXX` `BATCH_MTP` `BATCH_MTP_SERIAL` `DF2_END_UPDATE` `DF2_FORK` `DF2_GATE` `DF2_HEAD` `DF2_LOG_MIB` `DF2_LOOKUP_SKIP` `DF2_PRIO` `DF2_ROW_MS` `DF2_SHARE_HEAD_OFF` `DF2_SKIP` `DF2_STAGE` `DF2_WAITCOMMIT` `DISCO` `DRAFT_KEEP` `FOLLOW` `KVA_DRAFT_FLOOR` `LOOKAHEAD` `LOOKAHEAD_K` `LOOKUP_CHAIN_FIXED` `LOOKUP_MAX_T` `MTP_BATCH` `MTP_BATCH_F16` `MTP_BATCH_RING` `MTP_CATCHUP_ALL` `MTP_EAGER` `MTP_FULL_HEAD` `MTP_HIP_PRIO` `MTP_KEEP` `MTP_MULTI_COPY` `MTP_PREFILL_SYNC` `MTP_PRIORITY` `MTP_PROXY` `MTP_Q8_SM60` `MTP_ROOM_MIN` `MTP_SHARED_BRANCH` `MTP_SIDE_STREAM` `MTP_STAGE0` `MTP_TOP2` `NO_BATCH_KV_STEP` `PIPED_MTP` `PIPELINE_SPEC_DEPTH` `SERVE_STOP_BOUND` `SPEC_COST` `SPEC_COST_TABLE` `SPEC_COUPLED` `SPEC_CUM` `SPEC_DEPTH` `SPEC_GUMBEL` `SPEC_MIN_TEMP` `SPEC_PROB` `SPEC_PROB_DT` `SPEC_PROB_GATE` `SPEC_T_CYCLE` `SPEC_WATCH` `SUFFIX_INDEX` `SUFFIX_INDEX_FACTOR` `SUFFIX_INDEX_FILE` `SUFFIX_INDEX_MB` `SUFFIX_INDEX_MIN_PROB` `SUFFIX_INDEX_RULE` `THINK_PROBE`
- **cache/threads/park policy** (48): `ADMIT_MERGE_FIRST` `AUX_CPUS` `AUX_STAGER` `CACHE_MESSAGE_BOUNDARY` `CACHE_TOOL_BOUNDARY` `CACHE_TOOL_BOUNDARY_GAP` `CACHE_TOOL_BOUNDARY_MAX` `CKPT_ASYNC` `CKPT_EVICT_FLOPS` `CKPT_GATHER` `CKPT_POOL` `CKPT_POOL_READY` `CKPT_REREAD` `CKPT_THREADS` `CKPT_VRAM_SLOTS` `CONV_BUDGET` `CONV_BUDGET_CAP_MIB` `CONV_BUDGET_MARGIN_MIB` `CONV_KEEP_GIB` `CONV_WORKSET` `CONV_WORKSET_TOKENS` `HOST_CORE` `NUMA_POOL` `NUMA_SPLIT` `PARK_ASYNC` `PARK_TRUNCATE` `PARK_TTL_HOLD_S` `PARK_TTL_S` `PARK_TTL_SHARE` `PARK_TTL_SUB_RATIO` `PARK_VRAM_MIB` `POOL_CORES` `POOL_FUSE_MIN` `POOL_FUSE_PHASES` `POOL_LAZY_PARK` `POOL_MWAITX` `POOL_PREREAD` `POOL_SPIN_US` `POOL_TAIL4` `POOL_UMWAIT` `POOL_UMWAIT_C02` `POOL_UMWAIT_ESC` `SLOT_CKPT_MOVE` `SLOT_COPY_BATCHSYNC` `SLOT_COPY_DIRECT` `SNAPSHOT_FULL_CAPTURE` `TG_FLUSH_US` `TOOL_CALL_TOKEN`
- **multi-GPU / batch / layer-split planning (the disk tier needs one GPU and no --batch)** (46): `BATCH_CONCURRENT_READ` `BATCH_DECODE_SHARE` `BATCH_GROUPS_NV_SERIAL` `BATCH_YIELD_CHUNKS` `GFX12_RING` `KEEP_MAPPING` `LAYER_SPLIT_GATE` `LAYER_SPLIT_PROBE` `PIPED_CONCURRENT` `PIPED_CONCURRENT_MIN` `PIPED_CONCURRENT_PRIO` `PIPED_CONCURRENT_UNPIN` `PIPED_DECODE_MAX_MS` `PIPED_DECODE_MIN_MS` `PIPED_READ_PIECE` `PIPELINE_ADAPT_ASYNC` `PIPELINE_AGREE` `PIPELINE_AUTO` `PIPELINE_AUTO_MIN_STAGES` `PIPELINE_DOOM_SKIP` `PIPELINE_FORCE_MISS` `PIPELINE_LOOKUP_PON` `PIPELINE_PRESTAGE` `PIPELINE_SNAP_OVERLAP` `PIPELINE_SWITCH` `PIPELINE_THETA` `PRIMARY_DEVICE` `PW_WITH_BATCH` `SPILL_OFF` `SPLIT_CHUNK16K` `SPLIT_COVER_B` `SPLIT_EARLY_OFF` `SPLIT_LINK_ABS` `SPLIT_MISS_MS` `SPLIT_MTP_BATCH` `SPLIT_NEAR_TIE_PCT` `SPLIT_OVERLAP` `SPLIT_OWN` `SPLIT_OWN_BUFFERS` `SPLIT_OWN_PLACE` `SPLIT_RING` `SPLIT_SMALL_MAX` `SPLIT_SPEED_SCALE` `SPLIT_UNIFORM` `SPLIT_WAIT_MS` `STACK_ISOLATE`
- **prompt path scheduling/staging/chunk sizing** (36): `GROUP_COPY` `HANDOFF_PIPE` `HANDOFF_PIPE_WORKER` `LM_PREFETCH` `PF_GROUP_CAP` `PF_GROUP_SCRATCH` `PF_HC_REGION` `PF_OCC` `PF_OCC_W` `PF_OWN_PROBE_OFF` `PF_STEP_SYNC` `PREFILL_BORROW` `PREFILL_COPY_EVENT_EVERY` `PREFILL_COPY_WAIT_DEDUPE` `PREFILL_CPU_SHARE_BATCH` `PREFILL_GROUP_GATHER` `PREFILL_HELPER_SOURCE` `PREFILL_ISSUER` `PREFILL_LEND_PCT` `PREFILL_LM_EVEN` `PREFILL_LM_FORCE` `PREFILL_LM_MARGIN_MIB` `PREFILL_LM_R` `PREFILL_LM_TOKENS` `PREFILL_PIPE` `PREFILL_RING` `PREFILL_ROUTED_STREAM` `PREFILL_ROUTED_STREAM_DEPTH` `PREFILL_STREAM_AHEAD` `PREFILL_STREAM_MIN` `PREFILL_WARM` `STAGER_BATCH` `STAGER_RING` `STAGER_SLEEP` `STAGER_SSD` `STAGER_THREADS`
- **graph/stream/doorbell plumbing** (26): `COMMIT_SYNC` `COPY_HIP_PRIO` `DEVICE_SPIN` `DF_BRANCH` `DF_QB_Q4` `DOORBELL_STORE` `DUMMY_NODES` `FUSE_HITS` `FUSE_WAIT` `GRAPH_PRIO` `HEAD_HOST` `HEAD_HOST_OFF` `HIP_ADAPT_KERNEL_COPY` `HIP_QUEUE_PRIME` `IC_WARM_MB` `LAUNCH_PROBE` `SH_FORK_LATE` `SH_STREAM` `TAP_FILE` `TAP_LAYERS` `VERIFY_ALL_RESIDENT` `VERIFY_ASYNC_LAUNCH` `VERIFY_COHERENT` `VERIFY_DEVICE_PLAN` `VERIFY_EARLY_PLE_FLAG` `VERIFY_NODES`
- **KV placement/growth** (12): `KV_GROW` `KV_GROW_FLOOR` `KV_GROW_HOLD` `KV_GROW_INIT` `KV_GROW_STEP` `KV_HOST_DMA` `KV_HOST_REGISTER` `KV_MAIN_OFF` `KV_PREFETCH` `KV_RESIDENT_MIN` `KV_RING_OFF` `KV_STAGE_OWN`
- **multi-GPU peer prompt path (the disk tier needs one GPU; PF_PEER_F16/CPU/SUMS would be value-changing there)** (10): `PF_PEER_COMPACT` `PF_PEER_CPU` `PF_PEER_F16` `PF_PEER_GATHER` `PF_PEER_HOST` `PF_PEER_OUT_PIPE` `PF_PEER_RING` `PF_PEER_STREAM` `PF_PEER_SUMS` `PF_PEER_XSTREAM`
- **expert seeding (placement)** (3): `DOMAIN_PROFILES` `SEED_HOLD` `SEED_SCALE`
- **prefetch hint** (3): `IQ_PREFETCH` `NVFP4_PREFETCH` `Q2_PREFETCH`
- **prompt-path scheduling** (1): `AUTO_NOBORROW`
- **copy batching** (1): `DMA_BATCH`
- **VRAM accounting for chunk sizing** (1): `EMB_REUSE_ACCOUNT`
- **bitwise: "Same bits" (DETAILS.md:1084)** (1): `MMQ_DIRECT`
- **sampling path: tokens change, the state of given tokens does not** (1): `OLD_SAMPLER`
- **sampling penalty counts (same note)** (1): `PENALTY_PROMPT`
- **batch scheduling** (1): `PIPE_BYIELD_NODRAIN`
- **layer-major walk of a prompt (order of work, not arithmetic)** (1): `PREFILL_LAYER_MAJOR`
- **prompt io** (1): `PROMPT_DIRECT`
- **staging ring size** (1): `RING_BYTES`
- **bitwise by design: every id and weight is route_multi's (native_router.cu:223), native_multi_parity passed (kern-hip.md)** (1): `ROUTE_FAST`
- **sampling path (same note)** (1): `SAMPLER_ONE_BLOCK`
- **expert arena placement** (1): `VERIFY_ARENA`
- **only moves a row between pool passes (defer_rank.hpp); asserted by the key test** (1): `VERIFY_DEFER_CPU_FOLD`

### Inert (273; names without the `STRATA_` prefix, grouped by why)

- **diagnostic/test hook** (111): `ACT_DUMP` `ACT_DUMP_COV` `ACT_DUMP_LAYERS` `ACT_DUMP_RAW` `ACT_DUMP_RAW_MAX` `ADAPT_FETCH_VERIFY` `BANDIT_LOG` `BATCH_TELEMETRY` `COLD_TRACE` `COMMIT_TIMING` `CRASH_BT` `CVEC_FUSE_CHECK` `DBG_CPU_GATE` `DBG_FEAT` `DBG_FEAT_LAYERS` `DBG_GDN` `DBG_NAN` `DECODE_TIMING` `DF2_PROFILE` `DF2_RING_LOG` `DF2_RING_STATS` `DF2_TAP_DUMP` `DF2_TIMING` `DISCO_TRACE` `DISPATCH_REPORT` `DISPATCH_STATS` `DRAFT_LOG` `DRAFT_TIMING` `DUMP_FIRST_LOGITS` `DUMP_MOE_INPUT` `DUMP_MOE_LAYER` `FOLLOW_LOGPOS` `FORCE_FAULT` `GFX906_ATTN_QUERY_SWIZZLE_TRACE` `GFX906_ATTN_REDUCE12_TRACE` `GR_INJ_CHECK` `HC_UPMIX_CHECK` `HIDDEN_DUMP` `HIPBLASLT_VERBOSE` `IDX_FP16_CHECK` `INJECT_FATAL` `INJECT_FATAL_ONCE` `IO_STATS` `LOGITS_DUMP` `LOGPOS` `LOGPOS_EXTRA` `LOGPOS_TOPK` `LOOKAHEAD_STATS` `MANIFEST` `MMQ_DIRECT_LOG` `MMVF_CHECK` `MTP_PROFILE` `NAN_GUARD` `P4_TIMING` `P4_TRACE` `PARK_TRACE` `PF_ALLOC_TRACE` `PF_OWN_CHECK` `PF_TILE_DEBUG` `PIPELINE_DEBUG` `PIPELINE_LOG` `PIPELINE_TRACE` `PLE_CHECK` `PLE_WAIT_TRACE` `POOL_TRACE` `PREFILL_DUMP_R` `PREFILL_DUMP_R_ALL` `PREFILL_TIMING` `Q6P_SELFTEST` `QSA_DUMP` `QSA_DUMP_LAST` `QSA_DUMP_LAYER` `QSA_DUMP_STEPS` `ROCBLAS_VERBOSE` `ROUTE_DUMP` `ROUTE_PRIOR_STATS` `ROUTE_STATS` `RSS_TRACE` `SCORES_THREAD_TRACE` `SELECT_TWO_PASS_CHECK` `SNAPSHOT_VERIFY` `SPEC_TRACE` `SPEC_TRACE_NOPICK` `SPLIT_TIMING` `STALL_STACKS` `STATE_HASH` `STATE_HASH_GDN` `TEST_BELL_SLEEP_US` `TEST_CACHE_FAIL` `TEST_CONC_ON_HOST_CPU` `TEST_COPY_FAIL_AT` `TEST_HANDOFF_DROP` `TEST_PAGEABLE` `TEST_PINNED_SHARE` `TEST_PIN_BUDGET_MIB` `TEST_PIN_LEFT_KIB` `TEST_SLOW_COPY_US` `TEST_SYSFS_ROOT` `TEST_UNMAP_GGUF` `TEST_VERIFY_STALL` `TRACE` `TRACE_ADAPT` `VERIFY_DEBUG` `VERIFY_PROFILE` `VERIFY_PROFILE_SEL` `VERIFY_TRACE` `WATCHDOG_IO_S` `WATCHDOG_S` `WINDOW_TRACE` `WINDOW_TRACE_EVERY` `WINDOW_TRACE_MAX_MB`
- **io/memory/limits/paths** (71): `ARENA_FALLBACK_RESIDENT` `ARENA_FILE_CACHE` `ARENA_LOCK` `ARENA_MMAP` `ARENA_PIN_GIB` `ARENA_RAM_FALLBACK` `ARENA_RAM_WAIT_S` `ARENA_RELEASE` `ARENA_THP` `COMMIT_GROWTH` `CONVERSATION_DISK_DIR` `CONVERSATION_DISK_MAX_GIB` `CONVERSATION_DISK_MIN_GAIN` `CONVERSATION_DISK_MIN_TOKENS` `DEFERRED_REGISTER` `DIRECT_LOAD_FAULT` `EMULATE_CC` `EXPERT_NIBBLE_RECHECK` `FILE_CACHE_OLD_RULE` `FILE_RELEASE` `FILE_TIER_HONEST` `FILE_TIER_IDLE` `FILE_TIER_PROBE` `HIP_VMM` `IO_PF_AHEAD` `IO_PF_BUDGET_FRAC` `IO_PF_BUDGET_MB` `IO_PF_GATE` `IO_PF_GATE_HI` `IO_PF_GATE_LO` `IO_PF_GATE_USEFUL` `IO_PF_MINCONF` `IO_PF_SCHED` `IO_PF_STAGE` `IO_PF_TARGET` `IO_PF_THREADS` `IO_PREFETCH` `IO_PREFETCH_DEPTH` `IO_THREADS` `KEEP_IO_PRIORITY` `LOAD_DONTNEED` `MEMORY_LIMIT_MIB` `NO_ARENA_THP` `NO_LARGEPAGES` `OWNED_PRICE` `PAR_COUNTERS` `PLE_AUTO_MMAP` `PLE_AUTO_RAM` `PLE_EARLY` `PLE_GATHER_THREADS` `PLE_PROBE` `PLE_SLICES` `PSI_GUARD` `PSI_MAX_PAUSE_MS` `PSI_PARK_PCT` `PSI_PAUSE_PCT` `PSI_RESUME_PCT` `RAM_KEEP_GIB` `READ_AHEAD` `REMOTE_SPIN` `REMOTE_ZEROCOPY` `SESSION_BUFFERED` `SSD_GBPS` `SSD_KEEPALIVE` `SSD_KEEPALIVE_WINDOW` `STARTUP_UNBUFFERED` `UMA_HEADROOM_GIB` `UMA_REGISTER` `UMA_RESERVE_GIB` `UNBUFFERED_LOAD` `WDDM_BUDGET`
- **other engine (strata-dsv41 / metal): own binary, own slot store** (44): `DENSE_PFGEMM` `DS4_A_MB` `DS4_CPU_LANE` `DS4_DISK_SKIP_AUTO` `DS4_DISK_SKIP_BUDGET` `DS4_DISK_SKIP_FROM` `DS4_DISK_SKIP_SETTLE` `DS4_DISK_SKIP_W` `DS4_DISK_SKIP_WINDOW` `DS4_DSPARK_CUM` `DS4_HIPK` `DS4_LANE_FREE` `DS4_MARKS_INSIDE` `DS4_NO_LIGHT` `DS4_NO_TC` `DS4_PIN_BY_NEED` `DS4_PIN_GB` `DS4_PREFIX_GB` `DS4_PREFIX_KEEP` `DS4_PREFIX_MIN` `DS4_PREFIX_STOP` `DS4_PROMPT_SKIP` `DS4_RAM_SKIP_W` `DS4_RAM_TIER` `DS4_READ_PIECE` `DS4_READ_PIECES` `DS4_READ_ROUNDS` `DS4_RESTORE_POINTS` `DS4_RESTORE_TAIL_AUTO` `DS4_REUSE` `DS4_SLOT_CAP` `DS4_SLOT_DAYS` `DS4_SLOT_DIR` `DS4_SLOT_GB` `DS4_SLOT_KEEP` `DS4_SLOT_MIN` `DS4_STREAM` `DS4_TQ_WORDS` `DS4_X16` `GLM_EXL3_DECODE` `METAL_KEEP_FREE_GB` `METAL_LEAVE_SHARE` `METAL_NEIGHBOURS` `TEST_NAN_STOP_AT`
- **q36 engine path (qwen35 library): no ConversationDiskTier** (15): `ATTN_WINDOW` `Q36_BLAS_F32` `Q36_CHUNK` `Q36_DRAFT_VOCAB` `Q36_FIXED_SLOTS` `Q36_KEEP_PACKED` `Q36_NO_EXT` `Q36_NO_MMQ` `Q36_NO_MTP_CPU` `Q36_OLD_KERNELS` `Q36_PLACE_DEBUG` `Q36_READ_WINDOWS` `Q36_STREAM_ALL` `Q36_VDEBUG` `Q36_VERIFY`
- **strata-poly engine: own disk key** (9): `CPU_STATS` `FATTN` `FDEC` `FDEC_FROM` `GDN_COLS` `GEMM_SMALL_TILED` `ROUTE_TRACE` `SOFTMAX_REG` `SWAP_HALF`
- **test only** (9): `FAST_GDN` `GDN_PARITY_TRACE` `GRP_KEEP_GOING` `GRP_REPS` `PARITY_DUMP` `PARITY_SKIP_Q4` `PLE_GGUF` `SESSION_GOLDEN_PRINT` `TEST_BIG_GDN`
- **file magic, not an environment variable** (2): `HIPBLASLT_TUNING_V1` `ROCBLAS_TUNING_V1`
- **bench only** (1): `BENCH_QMIXED`
- **the key mechanism itself (extra names for the key)** (1): `CONVERSATION_DISK_KEY_ENV`
- **diagnostic** (1): `F16_RANGE`
- **profiling** (1): `GDN_CHUNK_PROF`
- **bench selector** (1): `GR_RDNA_ONLY`
- **calibration dump (read stays exact while capturing)** (1): `KVA_CAPTURE`
- **calibration dump** (1): `KVA_CAPTURE_FP32`
- **parity check run** (1): `KVA_PARITY`
- **test knob** (1): `NO_VBMI`
- **statistics** (1): `QSA_STAT`
- **debug measurement (D-1)** (1): `SEL_OVERLAP`
- **limit** (1): `VRAM_FREE`
