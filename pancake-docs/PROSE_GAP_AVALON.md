# Why does Avalon decode prose faster than pancake? (MOD_PLAN 17.2, analysis note, 2026-11-11)

Status: **read-only analysis plus a harness; nothing new was measured.** No engine was started, nothing was built, the GPU was not used.

How to read the tags in this note:

- **unconfirmed (read from code)**: I read the source (Avalon at `/home/dan/clone-of-d/Avalon`, pancake in this tree) and have not run it.
- **derived**: arithmetic on the saved measurements `bench/results/2026-11-11-vs-avalon/*.json` (branch `port/rdna3-decode`). The assumption is stated each time. A derived number is not a measurement of the thing it names.
- **measured (cited)**: a number someone measured and recorded elsewhere; the source is named.

What I could not do: the task asked for `git log 88313b6..8b9a0f4` in the Avalon repo. The sandbox of this worktree refuses git commands that point at another checkout, so the commit subjects were **not read**. The comparison below is from Avalon's current source (HEAD `8b9a0f4`, as built for the 2026-11-11 run), its README and its launch script `/home/dan/.local/bin/avalon-server`. The 2026-11-11 README already says most of the 50 newer commits are CUDA-only and that decode on HIP did not change (95.2 vs 96.4 tok/s, measured, cited).

## 1. What the saved numbers already say (derived)

Avalon's own log line is `draft acceptance = a (A accepted / G generated), mean len = M`, with `M = 1 + accepted / verify steps` (`tools/server/server-context.cpp:744`, unconfirmed (read from code)). So "mean len" is tokens committed per verify window. The neutral client (`cmp_bench.py`) saved `draft_n_accepted` and `draft_n` for every request on all three engines, so the same quantity can be derived for all of them:

    windows = predicted_n - draft_n_accepted        (assumption: every window commits 1 + its accepted drafts)
    tokens/window = predicted_n / windows           ms/window = predicted_ms / windows

Rounds 2 and 3 pooled (2 requests per prompt and engine; run 1 discarded as in the README), same IQ2_XS GGUF. Avalon 8b9a0f4 / Strata-rdna3 / pancake:

| prompt | decode tok/s | tokens per window | drafted per window | ms per window |
| --- | --- | --- | --- | --- |
| story (prose) | 97.9 / 88.2 / 93.5 | **2.13** / 1.87 / **1.83** | 1.98 / 1.61 / 1.57 | 21.8 / 21.2 / 19.5 |
| 15 tips (prose) | 102.7 / 94.3 / 101.9 | **2.21** / 2.12 / **2.11** | 1.99 / 1.66 / 1.68 | 21.5 / 22.5 / 20.7 |
| MoE explanation (prose) | 110.6 / 106.6 / 101.8 | **2.30** / 2.19 / **2.04** | 1.99 / 1.93 / 1.73 | 20.8 / 20.5 / 20.1 |
| JS debounce (code) | 118.6 / 119.4 / 129.0 | 2.82 / 3.13 / **3.06** | 2.12 / 2.67 / 2.67 | 23.8 / 26.2 / 23.7 |
| Python LRU (code) | 131.9 / 118.9 / 135.0 | 2.81 / 3.34 / **3.27** | 1.99 / 2.80 / 2.67 | 21.3 / 28.1 / 24.2 |

(derived; per-request tokens/window are stable between requests, e.g. story: Avalon 2.06-2.21, pancake 1.81-1.84, while tok/s moves by +-10, which is why the ratio is more informative than the tok/s sd of 12-17.)

What this suggests (derived, not proven):

1. **The gap on prose is in tokens per window, not in time per window.** On prose pancake's window is as cheap as Avalon's or cheaper (19.5-20.7 ms against 20.8-21.8). Avalon commits 0.1-0.3 more tokens per window (story +16%, tips +5%, MoE explanation +13%). On code pancake commits more per window (suffix/lookup windows of ~2.7 drafts against Avalon's ~2.0) and wins.
2. The lead's summary "acceptance 71/69/69%, so not draft quality" does not separate the cases: acceptance per drafted token is not comparable when Avalon always sends 2 drafts and pancake sends 1.6-1.7 on prose (chain cut by `--spec-min-p 0.5`). What differs is accepted drafts per window: story 1.13 (Avalon) against 0.83 (pancake), derived.
3. Strata-rdna3 and pancake, which presumably use the same MTP head (not checked: the Strata-rdna3 run used its own preset and data folder) and a similar draft logic, have almost the same tokens per window on prose (1.83-1.87 story, 2.11-2.12 tips). So the difference is **Avalon against both Strata engines**, not something pancake's changes introduced. This also points away from kernels, threading and expert caching (the Strata engines have the cheaper windows) and towards the drafter.
4. Caveats that matter: n = 2 requests per cell; the three engines generate different text for the same prompt (Avalon runs a lossy tail skip by default, see 3.3, and a different quant path), and text predictability differs, so tokens per window across engines is not a same-text comparison; the window formula is an assumption (pancake's `drafts_offered/accepted` semantics were read in `serve/server.py` only, not checked against the engine); run-to-run noise of a whole series is not known. The harness below records the text hash and the head of the answer so the lead can see whether two engines wrote the same text.

## 2. Ranked explanations (all unconfirmed)

| rank | explanation | evidence | cheapest test | port feasibility |
| --- | --- | --- | --- | --- |
| 1 | **The MTP head is a better drafter in Avalon**: Avalon ships its head as IQ3_K/IQ4_NL, 1.48 GB (`MTP/mtp-...-shared-IQ3K-IQ4NL-FR65k.gguf`), with a 65k-token reduced LM head; pancake's head is Q2_0 round-to-nearest (min-squared-error scales), 0.71 GB experts (`tools/mtp_pack.py`), with a 40.5k-token English draft vocab (`data/draft_vocab_en.bin`, 162,100 bytes / 4, assuming u32 ids). The head is itself a 512-expert MoE layer, so expert precision should matter for its agreement with the model. | Section 1 item 3 (same head, same tokens/window on both Strata engines; Avalon higher). Avalon's README quotes 80.7% teacher-forced agreement for its head; there is no such number for ours. The only draft-head quality arm recorded here (`--mtp-q4`, a 4-bit copy for speed) is not a precision test. | Measure per-position acceptance on both: Avalon logs `acc per pos` at trace verbosity (`SLT_TRC`), pancake's window trace gives the same offline. A same-text teacher-forced agreement run of the two heads (needs the GPU, so later). | Medium to hard. `tools/mtp_rt.py` writes only the Q2_0 expert blob (`BLOB = 3 * (H*FF*18//64)`), so a 4-bit head needs a new expert layout, loader code in `src/core/mtp.cpp` and a head kernel for it (+0.7 GB). A bigger vocabulary alone is a flag (`--mtp-draft-vocab`; MOD_PLAN round A measured "full / en vocab flat" on a mixed set, not on prose). Nothing to port before measuring. |
| 2 | **Draft depth and early stop**: Avalon always drafts 2 (`--spec-draft-n-max 2`, `p_min` default 0, so the second draft is never cut: `common/speculative.cpp` MTP loop cuts only below `p_min`). Pancake drafts up to 4 but cuts the chain after a draft whose own probability is below 0.5 (`--spec-min-p 0.5`, `src/core/mtp.cpp:1936`: step j runs only while the previous step's probability >= min_p). On prose pancake sends 1.57-1.73 drafts per window. | Section 1 (drafted per window). An upper bound for what depth alone can give: to match Avalon's 1.13 accepted per window on the story, the 0.41 extra drafts per window would need 73% acceptance although they were cut at probability < 0.5; at a guessed 40% they add 0.16 accepted per window (tokens/window 1.83 -> ~2.0, +9% tokens per window, before the cost of the extra rows). So depth can explain part of the story gap but not all of it (guess, derived). | A flag-only A/B on prose prompts, per prompt class: base; `--spec-min-p 0`; `--spec 2 --spec-min-p 0` (Avalon's shape); `STRATA_SPEC_COST=1`. Round A already ran `--spec 3/5/6`, thresholds and `STRATA_SPEC_COST` on a mixed set and found them flat (MOD_PLAN 11 round A: base 111.7, SPEC_COST 112.0; after an 11.7K prompt 92.5 vs 98.3), which would hide a prose gain offset by a code loss. | Cheap: flags. If it wins on prose and loses on code, the adaptive version already exists (`DraftPolicy::keep_draft`, `STRATA_SPEC_COST=1`, not on in the HIP preset: it is in `nv_defaults.cpp` for GB10 only); a prose-aware prior is a small change in `src/spec/draft_policy.cpp`. |
| 3 | **The comparison is not like for like**: Avalon runs `GGML_CUDA_MOE_TAIL_SKIP` default 7 (lossy: missed experts routed only at rank >= 7 are skipped; `moe-cache.cu:9677`), pancake's preset sets `STRATA_ROUTE_TAIL_SKIP=0`. The two engines then write different stories, and tokens per window depends on the text. | Source and preset. MOD_PLAN measured pancake with tail skip 7 at 129.7 vs 111.7 (round A, lossy, mixed set). The saved per-window cost (section 1) shows no sign that Avalon's windows are cheaper, so skipping is not what makes Avalon fast per window; but it changes which text is written. I do not know the tail-skip environment of the 2026-11-11 Avalon run (the service sets none; it ran `avalon-server` as shipped). | Run Avalon once with `GGML_CUDA_MOE_TAIL_SKIP=0` (server restart, env only) and compare tokens per window; or run pancake with `STRATA_ROUTE_TAIL_SKIP=7`. Check the answer text hashes in the harness output. | Cheap (env) but it is a control, not a port. |

## 3. Mechanisms in Avalon, one by one

Every row is unconfirmed (read from code). "Port" says what a pancake change would touch.

### 3.1 The ones asked about

| mechanism | what it does, where | pancake equivalent | expected effect on prose decode | port feasibility |
| --- | --- | --- | --- | --- |
| MoE down projection weighted and summed in one MUL_MAT_ID (`LLAMA_MOE_DOWN_SUM`) | `ggml_mul_mat_id_sum` node, `src/llama-graph.cpp:2502-2528`; the CUDA block is in `ggml-cuda.cu:5472`. Only for batches of >= 128 tokens and off unless the env is set, and it needs the prefill ring (`GGML_CUDA_MOE_RING_PREFILL`). | Not needed for decode: decode batches are 1-4 rows. | **None on decode** (prompt-only, off by default). | Not applicable to this question. |
| MTP draft fed from device rows (`LLAMA_MTP_ROWS_CB_SINGLE`, `LLAMA_MTP_ROWS_ASYNC`, `LLAMA_MTP_H_ROWS_DEV`) | The draft takes the target's nextn rows on the device, ubatch by ubatch, instead of through the host output buffer; `src/llama-context.cpp:1801-1810, 2611-2614`, `common/speculative.cpp:1508-1512`. The code comment says it is for a prompt batch of several ubatches (a 390 MB host copy per 9536-token ubatch). | In pancake the MTP chain runs on the GPU from the verify window's hidden rows already (`mtp_draft` graphs). | None on decode (prompt batches; the single-row decode copy is small). | Not applicable. |
| Expert cache preload from `moe-cache-ranking.bin` | Fills the GPU expert cache at start from a shipped ranking, then LRU; `moe-cache.cu` (`moe_cache_preload`, ranking save at `GGML_CUDA_MOE_CACHE_PROFILE_SAVE`); log line "MoE expert cache preloaded". README credits Strata's ranking format. | `--expert-profile data/expert-profile-pancake.bin`, `--adapt-swaps 32 --adapt-every 1`, `STRATA_PIN_BY_NEED=1` in the 7900 XT preset (it is the lineage Avalon took the format from). | Already equivalent; the saved ms/window shows pancake no slower. | Nothing to port. |
| How CPU experts are threaded and synchronised | The GPU MoE block publishes ids and activations to mapped host memory, a polling kernel waits on a mailbox; host threads compute missed experts (`ggml_moe_cache.cpu_experts`) and the GPU copies results in; all inside the HIP graph (`moe-cache.cu:7269-7288`). Threads = physical cores (`moe_hosted_default_threads`, 8 on the 5700G; the log says `n_threads = 8`). Expert weights are `hipHostRegister`ed in place (`:3014-3030`, "CPU per hosted layer -4 to -6%, generation +1.6%" per its comment, not measured by us). | Same lineage as Strata: `STRATA_COMBINE_CPU`, `STRATA_WAIT_FUSE`, `STRATA_ROUTE_PUBLISH`, `STRATA_VERIFY_DEFER_CPU`, registered memory (`src/core/pinned.cu`). | None expected: per-window time is already lower in pancake on prose (section 1). | Nothing to port. |
| Draft depth 2 vs 4 on low-acceptance prose | See rank 2. Avalon: fixed 2 drafts, no early stop (`p_min` 0). Pancake: up to 4, cut at 0.5. | `--spec 4 --spec-min-p 0.5`; `STRATA_SPEC_COST` exists, off on HIP. | Part of the story gap at most (rank 2). | Cheap flags first. |

### 3.2 Cost of a rejected draft window in each engine (the prose angle)

Prose has low copy rate and lower acceptance, so a large share of windows end early. What a rejection costs:

- **Avalon** (unconfirmed (read from code)): the window is always 3 rows (1 + 2 drafts). Both draft steps always run (`p_min` 0, so no early stop and no wasted-draft saving). On rejection the target's recurrent (GDN) state is rebuilt from a per-token stash (`LLAMA_GDN_REPLAY`, default on for this architecture, `src/llama-memory-recurrent.cpp:82`), not by re-running the layers. The 3 verify rows go through the hosted MoE block, which skips missed experts that every row routes only at rank >= 7 (tail skip, default 7). The saved numbers give 20.8-21.8 ms per window on prose (derived).
- **pancake** (unconfirmed (read from code)): the window is `1 + kept drafts` rows (1-5). Drafting is step by step with a host check after each step when `min_p > 0` (the `min_p <= 0` path launches all step graphs at once; `STRATA_MTP_EAGER` does that with `min_p` too, `src/core/mtp.cpp:1860-1905`). GDN state is replayed on device (`gdn_state_replay_kernel`, `src/kernels/cuda/verify_kernels.cu:491`, equivalent to Avalon's replay). `STRATA_VERIFY_DEFER_CPU=1` defers the late CPU expert rows (MOD_PLAN deferral items; I did not read that code path for this note). Tail skip is off in the preset. The saved numbers give 19.5-20.7 ms per window on prose (derived).
- **How much an extra row costs** is the number that decides depth. `src/spec/draft_policy.hpp` says "the cost is not convex: on a card with experts in RAM a window of 3 costs about a window of 2" (a code comment; no measurement for the 7900 XT is recorded in this note). If that holds, a draft is worth sending when `q * a > c / T_window * tokens-per-window`: with ~20 ms windows, ~2 tokens per window and an extra row costing about 1 ms, any draft that holds with chain probability times acceptance above ~10% pays, which is below the 0.5 cut. If an extra row cost 3-4 ms (the 5070 table in `draft_policy.cpp`, `kShape`: +0.35 of one token per extra row) the break-even moves to ~35%, where the 0.5 cut is about right. **So the cost of a row on this card decides rank 2, and it is not measured.** It can be measured without new code: run once with `STRATA_SPEC_COST=1` and read the line `spec cost table` the engine prints at the end (`src/program/generate.cpp:22707`), or collect a window trace and run `tools/pancake/trace/draft_costs.py`.

Adaptive depth that helps prose without hurting code: pancake's code windows get their length from the lookup/suffix drafter (`--lookup-max-t 12`, `DraftPolicy::choose`), not from the MTP depth, so lowering the **MTP** cut for prose does not shorten code windows (unconfirmed (read from code)). `DraftPolicy::keep_draft` already weighs expected tokens per ms against measured row costs and the measured acceptance by draft probability (`pa_ok_/pa_n_`); it was flat on the mixed set. If a prose A/B shows a gain, the cheapest change is to lower `margin_` / the prior for the chain rule only when the recent lookup rate is low (a prose detector), kept behind a switch.

### 3.3 Other differences found, and whether they can explain prose

| mechanism | where in Avalon | pancake equivalent | prose effect | port |
| --- | --- | --- | --- | --- |
| Tail skip default 7, lossy | `moe-cache.cu:9677-9700`, main-model layers only | `STRATA_ROUTE_TAIL_SKIP` (preset 0; MOD_PLAN measured +16% on the mixed set with 7, lossy) | Changes the text and the window cost; confound (rank 3), not a port | Env switch exists |
| ngram-simple as top-priority drafter (12-gram lookup, up to 48 drafts, min 1 hit) | `common/speculative.cpp:3041` (priority order), `common/common.h:367-369`; Avalon is started with `--spec-type ngram-simple,draft-mtp` | suffix drafter + `DraftPolicy` (pancake commits more on code, section 1) | None on prose: its drafted per window stays ~2.0, so it rarely fires | Not applicable |
| Dispatcher that moves one or two missed experts to the GPU when that finishes sooner (kernel copy from registered RAM, CPU time model fitted on one machine, `GGML_MOE_SPLIT_*`) | `moe-cache.cu:7365-7370, 8080-8120` | `--pcie-frac` (preset 0: MOD_PLAN "PCIe share on the 7900 XT: off is faster"), `STRATA_ADAPT_*`, k-th-miss items (A6/A7/A8: "none worth a GPU screen, all < 5%") | Unknown. Avalon's windows are not cheaper, so no evidence it helps decode here | Hard (copy kernel + time model); skip until a cost gap shows |
| MoE cache pregate (predict next layer's experts for MTP verify) | `moe-cache.cu:1841-1850`, `GGML_CUDA_MOE_PREGATE`, **off by default**; its own comment says it now costs 1.7-5% | `STRATA_*` lookahead routing already routes the draft tokens (MOD_PLAN 10.3 closed) | None (off, negative per its author) | Not applicable |
| MTP draft context shares the main model's expert cache | log line "MoE expert cache shared with the draft context", `GGML_MOE_SHARE_DRAFT` | Not checked where pancake keeps the head's 512 experts (0.71 GB `experts.bin`) | Window cost of the draft steps; pancake's windows are already cheaper | Not checked |
| `--moe-cache-cpu-overlap` (CPU share of cached rows) | `common/arg.cpp:2795`, plumbed through `config.overlap_cpu_rows`; I found no use of it in the hosted decode path | `STRATA_PREFILL_CPU_SHARE` (prompts) | None found for decode | Not applicable |
| RAM arena with O_DIRECT, LRU, registered in place | `moe-cache.cu:3129+`, README "RAM cache of experts" | the 33 GiB expert arena, `tools/pancake/os_tune.sh`, `STRATA_PIN_BY_NEED` | Only matters when RAM is short; not here (57 GB) | Nothing to port |
| Fused decode kernels (linear-attention layer 53 -> 39 launches), PDL, graph opt | `ggml-cuda.cu`, `GGML_CUDA_GRAPH_OPT`, `GGML_CUDA_PDL` | pancake's own GDN fusions (`STRATA_GDN_*`, `STRATA_FUSE_CONVAB`, ...) | Pancake's windows are cheaper on prose | Nothing to port |
| Sparse attention (QSA) masks built on the GPU, split kept across spec steps | README "What makes it fast" | `STRATA_QSA_ROWS`, `--kv-resident` | Only at long context; these runs are short | Nothing to port |

## 4. What the lead should run later (free machine, one engine at a time, GPU alone)

Each engine: start it alone, run the series **twice** on the same server and keep the second (the 2026-11-11 README found cold-start climb after a heavy file load).

    # pancake (./run-pancake.sh, port 8080)
    python3 tools/pancake/prose_gap.py --port 8080 --label pancake --prompt-name story --runs 6 --out bench/results/<day>-prose-gap/pancake-story.json
    # Avalon (avalon-server, port 11437; --server-log reads Avalon's own measured 'mean len')
    python3 tools/pancake/prose_gap.py --port 11437 --label avalon --prompt-name story --runs 6 \
        --server-log /home/dan/clone-of-d/Avalon/avalon-service.log --out bench/results/<day>-prose-gap/avalon-story.json
    # offline side by side
    python3 tools/pancake/prose_gap.py --compare bench/results/<day>-prose-gap/avalon-story.json bench/results/<day>-prose-gap/pancake-story.json

Repeat with `--prompt-name tips` and `--prompt-name moe` (the prose prompts), and `--prompt-name js` once as the code control. Then, on pancake only, flag-only variants of the preset, one at a time, each with the same command: `--spec-min-p 0`; `--spec 2 --spec-min-p 0`; `STRATA_SPEC_COST=1` (read `spec cost table` in the engine's stdout for the measured cost of a row). Control for rank 3: Avalon with `GGML_CUDA_MOE_TAIL_SKIP=0` in its environment, or pancake with `STRATA_ROUTE_TAIL_SKIP=7`.

Decision rules to write down before looking (suggested): count a variant only if tokens per window moves by more than the spread of the second-pass requests and tok/s moves in the same direction; a tok/s gap smaller than one series sd is not a result (the harness prints both sds).

## 5. Unverified, in one list

- Commit-level changes between 88313b6 and 8b9a0f4 (git log not readable from this worktree).
- That a window commits `1 + accepted` tokens in pancake's server counters (true for Avalon by its source, assumed for pancake).
- The tail-skip environment and the exact flags of the 2026-11-11 Avalon run beyond the shipped script.
- The cost of one more draft row on the 7900 XT with the current preset.
- That the three engines' texts are equally predictable (they are not the same text).
- Whether `data/draft_vocab_en.bin` is 40.5k u32 ids; whether the head's precision is what moves acceptance (rank 1).
- Credits: nothing here is adopted, so no README section 9 / `RESEARCH_AND_FINDINGS.md` section 4 rows yet. If a head, depth rule or other Avalon idea is adopted later, the rows go in the same commit.
