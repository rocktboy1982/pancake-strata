# MoE expert caching, offload, routing, prefetch and the CPU expert path: speed ideas from all Strata forks

Scope and method (2026-11-10). Source: `data/fork-kb/forks.sqlite`, 43,685 fork-only commits (`upstream_history = 0`,
2026-09-24 .. 2026-11-10). A regex over the messages (expert|cache|evict|admission|offload|prefetch|routing|pcie|pinned|
avx|numa|hugepage|pool|bandwidth|prior|skip|prune|resident|tier|shadow|...) matched 18,322 commits, 7,218 distinct first
lines. These were split into 8 topic lists (cache 493, placement 248, prefetch 97, routing 103, shared expert 56,
PCIe 432, CPU 412, quantization 363), deduplicated by first line, filtered to the earliest copy, and read by hand.
24 diffs were read with `gh api repos/<repo>/commits/<sha>` (about 24 API calls). Commit URLs below are the KB's
`commit_url` (many point at `Niko1221/Strata/commit/<sha>`, because forks share upstream's object network; the
same sha also resolves under the fork's own repo).

Pancake baseline used for the gain estimates (README section 5): an edit window is 37.4 ms, of which **17.4 ms is CPU
expert misses** (about 46%); a short-chat window is 24.8 ms with 5.2 ms CPU (about 21%). The CPU Q2_0 kernel already
runs at about 80% of the 5700G's 33 GB/s DDR4. So, as a rule of thumb: **cutting CPU expert bytes by X% gives at
most about 0.46·X% on edit windows and 0.21·X% on chat windows**, and only if the bytes are on the critical path.
Everything below is an author claim on the author's hardware unless marked as a pancake measurement. Nothing has been
measured on our box. No builds or runs were made for this note.

Pancake expert formats (docs/PANCAKE_FINDINGS.md section 10/10d): the "IQ2_XS" file has IQ2_S gate/up in 34 layers,
IQ2_XXS in 11, IQ1_M in 3, and every down is Q2_0; the Q2_0 file is all Q2_0. Derived blob sizes (n_ff·n_embd =
640·2560, from svgop's shadow table): IQ2_S layer blob 1,510,400 B, canonical Q2_0 blob 1,382,400 B, IQ2_XXS layer blob
about 1,305,600 B (computed from 66 B per 256 IQ2_XXS weights; this one is my arithmetic, not a measurement).

## 1. Which fork commits lower the bytes read per token?

### Takeaway
The strongest new idea is svgop's **shadow store**: CPU misses read a second, lighter quantization while VRAM hits keep
the exact weights. svgop measured the CPU pool as byte-bound and got +3-8% solo and +8-21% under concurrency. For
pancake the Q2_0 shadow is nearly the same as switching to the Q2_0 file, which we already did. The real
bytes-per-token lever for us is a **sub-2-bit shadow (IQ1_M/IQ2_XXS) for misses only**, or the **ISTA Coder
(256 of 512 experts)**, which halves the routed working set. Both are lossy, and both need a new file on a tight disk.

### Cited Findings
- **Shadow expert store, `STRATA_SHADOW_EXPERTS=<q2_0 pack dir>`** (svgop, lossy on misses only). Only the experts of a decode window that miss the GPU cache are computed from a mapped canonical Q2_0 store (1,382,400 B per blob instead of 2,046,400 B for IQ3_S). Hits, the PCIe share, the prompt path and every combine stay on the native pack. When the variable is unset the code is textually identical. Their falsifier measured 75 vs 152 µs per entry and "pool confirmed byte-bound". With 16 threads the Q2_0 rows streamed at about 50 GB/s against 35-40 GB/s for the IQ3_S/IQ2_S native rows (per-layer table in `campaign/shadow-bench/results.txt`). Hardware: an RTX 5090 box, IQ3_S Flash-Next. — [68d58c04d1 (falsifier)](https://github.com/Niko1221/Strata/commit/68d58c04d1c2bf22e3ef2fb3ae9b97aa78de9e64), [25bf1b90f6 (source + notes)](https://github.com/svgop/Strata/commit/25bf1b90f645dd5a44344bb4d04c7042805db95f), [eeca072bb6 (promoted: M1 -0.7% noise, pair +7.7%, conc +8.4%, MMLU 95.71)](https://github.com/Niko1221/Strata/commit/eeca072bb626af35669a010686217e04d85a8d71)
- Clean shadow A/B at the 262K reserve: M1 207.4 vs 201.6 (+2.9%), pair +3.8%, conc 294.1 vs 243.5 (+20.8%). svgop's own conclusion: "the slot count is NOT the conc lever, the shadow's byte-bound CPU path is". Their DeepSeek pack gets no shadow because it already reads at 93-96% of the plain-read ceiling on every type and is already mixed-quant at 1.75-2.31 bpw, so "there is no lighter artifact to swap to". — [5c46e95b23](https://github.com/Niko1221/Strata/commit/5c46e95b231a791d3aca77c43538952f2e1d3da2)
- Placing the shadow on tmpfs (`/dev/shm`) added M1 +5.5%, pair +4.2%, conc +3.5%. The mmap'd store took 2,635,199 minor faults per probe leg, and tmpfs pages are RAM-resident and THP-eligible. Later the store became an engine-level `memfd_create(MFD_HUGETLB|MFD_HUGE_2MB)` filled at about 15 GiB/s in about 2 s, with a fallback to the file mapping (`STRATA_SHADOW_MEMFD=0` turns it off). — [9e889bda3a](https://github.com/Niko1221/Strata/commit/9e889bda3a838ccf932a25cbd88d6430742beb6e), [8aa066b17d](https://github.com/Niko1221/Strata/commit/8aa066b17d5cd5b75aac570261568e6e6b2fe283)
- **`STRATA_ALT_EXPERTS=<pack>`** (mandoof1, lossy). The decode's CPU experts come from a second pack of the same model (for example IQ2_XS beside IQ3_XXS). `_MODE=cold` (the default) uses it only when the primary blob is not in the page cache; `_MODE=all` uses it always. There is a per-job format in the pool (`pool_mixed_fmt_test`). This is the shadow idea aimed at the SSD/page-cache tier. — [85b991da79](https://github.com/Niko1221/Strata/commit/85b991da79303fcd018bcc90f1a763067c93fa1e)
- **Requant recipe for cold layers** (svgop, lossy). "RECIPE-alpha (12 cold layers -> IQ2_S/Q2_0) = +2-4% M1 / +3-6% conc", plus +792 VRAM slots. This was a spec, not shipped. — [0d9d52b210](https://github.com/Niko1221/Strata/commit/0d9d52b2108de33f8585f44adfd326abfbefc647)
- **Masked-router pruned "Coder" pack** (svgop, DeepSeek). 192 of 384 routed experts are kept by setting the non-kept experts' gate bias to -1e30, with no weight copies. Same-hour result: +7.8% M1, +10.0% pair, +11.1% conc. Mechanism: "the routed working set halves, so the VRAM tier covers a far larger share". The pack was deleted the same day because the quality trade "buys nothing over the base". — [9e309adb39](https://github.com/Niko1221/Strata/commit/9e309adb39b7d71c7e7279035b0320bd16a6e65f), [93b85b38e7](https://github.com/Niko1221/Strata/commit/93b85b38e7)
- An ISTA GSQ-RCO **Coder variant of Qwen3.8-Flash-Next (256 of 512 experts)** is supported upstream and in forks. The README of AdrianBM96 lists it at 55 tok/s on a 3060 box and says it "reaches 91% of the full" model. zsariboga lists it as "Coder: IQ1_M". — [VlDubr f07c68c7c2](https://github.com/Niko1221/Strata/commit/f07c68c7c2), AdrianBM96/Strata3060 README (KB `readmes`), zsariboga/Strata README (KB `readmes`)
- Pruning by REAP saliency (GLM 288 -> 224 experts): +1.5% perplexity and +0.033 KL; decode unchanged, prefill 16-31% faster. — [lighttransport af672ebf3e](https://github.com/Niko1221/Strata/commit/af672ebf3ee4a389cfa4d006fc87fb9e2f5ca0e6), [37e0ed8f16](https://github.com/Niko1221/Strata/commit/37e0ed8f16c8c9447941f1d2b3a7343dfc9c29ae)
- Quality cost of lower-bit experts for a whole model: gputier measured Ornith experts in IQ2_XXS at 40.65 vs 28.45 tok/s, but KL 0.4697 vs 0.0423, and dropped it. — [gputier 67d5ff661f](https://github.com/Niko1221/Strata/commit/67d5ff661f)
- EXL3 2.68-bpw experts for Flash-Next are "22% smaller" than IQ3_S (android-imdad 7900 XTX fork, opt-in). Turboderp's 1.66-bpw pack was also benchmarked (Ereife2). — android-imdad/Strata-7900xtx README (KB `readmes`), [Ereife2 2c72fa5fb8](https://github.com/Niko1221/Strata/commit/2c72fa5fb8)
- Compact RAM arena for mixed-size experts (SAGE 1.59 bpw, blobs of 4.3-21.1 MiB). Slots sized to the largest expert held only about 1/3 of the budget's experts, so each slot is now as large as its first expert. — [Jackwwg83 0f40a03dae](https://github.com/Jackwwg83/Strata-DS/commit/0f40a03daed1fc4faa76fa52bc3cbd3b4efd8c7a)
- Activation sparsity does not give block-level byte savings: Nemotron relu² is 92.1% zeros, but only 33.7% of 16-value blocks and 5.9% of 64-value superblocks are all zero. — [Cudila 8dff8dc321](https://github.com/Niko1221/Strata/commit/8dff8dc321075912c901d89ab1bab8b072c3bf80)

### Inferences
- **Q2_0 shadow on the IQ2_XS file: about -8.5% bytes on the 34 IQ2_S layers and none elsewhere** (the 11 IQ2_XXS layers are already smaller than Q2_0). The shadow also needs a second 31.6 GiB store in a 57 GB box that already holds the 31.6 GiB arena. So for pancake the shadow collapses into "run the Q2_0 file", which pancake already measured (+5-9% writing) and which MOD_PLAN takes as the daily file. Status: **effectively ALREADY-HAVE.**
- **Sub-2-bit shadow for misses only (NEW, lossy).** IQ1_M gate/up with Q2_0 down would be about 1,177,600 B per blob (-15% vs Q2_0); IQ1_M down as well would be about 1,075,200 B (-22%). This is my arithmetic from the block sizes (IQ1_M 56 B per 256 weights). Estimated pancake gain if the pool stays byte-bound: **edit +7-10%, chat +3-5%**. The quality risk is real: gputier's whole-model IQ2_XXS KL was 0.47. svgop's design keeps it to misses only and gates it with MMLU, so the quality hit is smaller than a full requant but unknown for IQ1_M. It needs an imatrix requant of the expert tensors (about 25-30 GB on a nearly full disk), a pack builder, and svgop's 6 gated sites in `expert_source.cpp`. RAM works only if the shadow **replaces** the CPU-side native blobs (the native copy is then needed only for VRAM fills and swaps, which could read the GGUF via O_DIRECT).
- **Coder (MOD_PLAN item 25, parked).** svgop's DeepSeek analogue suggests +8-11% from the halved working set alone. On pancake, where the tier hit rate is 79.6% at 44% residency (README 7.4), halving the working set should raise residency toward 88%. The CPU bucket could then fall far more than on svgop's 97.7%-hit 5090, so I expect a larger relative gain, but this is not measured. It costs quality outside code (91% of full per AdrianBM96's README) and a new download.
- Not worth it for pancake: CPU repack to 4-bit codes (more bytes, section 4), EXL3 (needs a new kernel family; 2.68 bpw is more bytes than our 2.06-2.5).

### Gaps
- Whether ISTA publishes an IQ1_M or other sub-2-bit Flash-Next file other than the Coder; the KB has no such pack name. I did not check Hugging Face.
- The Coder's bpw is reported inconsistently: "IQ1_M" (zsariboga README) vs "~3.5 bpw LUT types" (pancake MOD_PLAN item 25). Its bytes per expert, and so its net CPU-bucket effect, is unknown.
- No fork measured a shadow on a DDR4 8-core desktop. svgop's 50 vs 37 GB/s stream gap was with 16 threads on a fast box; the share of that gap that comes from cheaper decode rather than fewer bytes is unknown on Zen 3.

## 2. Cache admission/eviction, placement and prefetch: what is new beyond pancake's kmiss tier, seeding and GPU lookahead?

### Takeaway
Little is new for pancake's regime. Most admission ideas are versions of "don't let one-off experts evict residents"
(pancake has k-th-miss admission) or "admit what crosses PCIe anyway" (inert at `--pcie-frac 0`). Several forks also
confirm that predictive prefetch from RAM does not pay. Two exact, cheap ideas are worth a look: **feeding prompt
routing into the adaptive tier** (`STRATA_ADAPT_PROMPT`) and **byte-aware placement** for the mixed-format IQ2_XS file
(`--light-expert-cpu`).

### Cited Findings
- **`STRATA_ADAPT_PROMPT=V`** (zsariboga, an ADEPT-style idea; exact without route prior). Each prompt chunk adds its routing counts to the adaptive tier's usage, weighted as V decode tokens, so the first adaptation after a prompt fills the cache for that prompt. It is about 10 lines in `prefill.cpp` plus a pointer from `generate.cpp`. No speed number. — [4ce19344ff](https://github.com/Niko1221/Strata/commit/4ce19344ff72c8ca9a1175a01d357c1179a521e8)
- Counter-evidence: RandoTeam tried counting the prompt's lookups and moving the cache on every prompt layer. It "read a 12K and a 29K prompt 3% slower", and the answer after the 12K prompt hit VRAM no more often (35.6% vs 35.1%). Their decode-only follow-the-answer tier (counts halved every 128 tokens, 16 swaps in flight) raised VRAM hits from 20.4% to 49.7% and decode from 28.6 to 33.5 tok/s (GLM 5.3 Flash, RTX 5090). — [RandoTeam 1eabf41cbc](https://github.com/Niko1221/Strata/commit/1eabf41cbc53eb03a558fa3b7ec75727ee929f1a)
- Frequency-gated LRU admission (`--expert-cache-policy=frequency`: 8-bit decaying counts halved every period; admission is refused for cold keys). The author reports "~10-13% decode speedup on short prompts" (RTX 5090, DeepSeek4 Q3 + DSpark). — [Aleks-Che 0561016de8](https://github.com/Aleks-Che/Strata/commit/0561016de87e1962285903f9b547187b4ca787de); same family: `STRATA_GLM_PROMOTE_MIN` "one-off experts no longer evict residents" — [yqYo1 72fb6723da](https://github.com/Niko1221/Strata/commit/72fb6723dacb7129533feea5c873c4d1a7ae2cae)
- **`STRATA_FETCH_ADMIT=1`**: the PCIe share's fetched blobs go straight into the slot of the layer's least-used resident ("Same bytes over PCIe"). — [AdrianBM96 50581c24f2](https://github.com/AdrianBM96/Strata3060/commit/50581c24f2b1d35e8c76e53d1037fcc25672c0f9); upstream's `STRATA_ADAPT_FETCH` is the same idea — [zsariboga 71785ffd7e](https://github.com/Niko1221/Strata/commit/71785ffd7e); and `STRATA_PCIE_USAGE` / `STRATA_PCIE_KEEP=n` (a staging ring the tier copies device to device) — [android-imdad 89ef442349](https://github.com/Niko1221/Strata/commit/89ef44234933c71d138f6f4c4a9a9ead4fa04883)
- **`--light-expert-cpu`** (tctien342, exact placement). With a static `--expert-profile`, experts whose blob is at most `--light-expert-max-mib` (default 2) and that the CPU computes natively get a 0.75 demand weight, so heavy blobs win VRAM and light ones go to the CPU. — [5576b48144](https://github.com/Niko1221/Strata/commit/5576b481449c638517c1835c15bb053bd1443ab5)
- Predictive prefetch from RAM keeps failing in forks. svgop's Fate-class cross-layer prediction (`DS4_AHEAD`) cost -7.5..-8.7%: "no slow tier exists in the RAM-copy config for prediction to pay against (the literature's win is prefetch from SSD/PCIe)". — [svgop 78973fab03](https://github.com/Niko1221/Strata/commit/78973fab0395460c018dd60974ad878edc3c1dff). Reading ahead more than the ten predicted experts is slower — [rostrovsky 845d7670b2](https://github.com/Niko1221/Strata/commit/845d7670b2). atmJava preregistered and rejected staged expert lookahead — [1f642f0bec](https://github.com/Niko1221/Strata/commit/1f642f0bec).
- L3 prefetch of the next layer's predicted CPU experts (`STRATA_L3_PREFETCH=<MB>`, a second thread, MWAITX) was aimed at a 7800X3D's 96 MiB L3; no number in the commit. — [Ben1078 98388242cd](https://github.com/Niko1221/Strata/commit/98388242cd1276bd4defa8b6741bab18b0c83149)
- Learned look-ahead weights with a probability threshold (`STRATA_LOOKAHEAD_WEIGHTS`, `_TAU`, `_B`, plus a feature dump); no number. — [ElXreno 5d94c4fbbd](https://github.com/ElXreno/Strata/commit/5d94c4fbbd61c616c2de67116ad4ccfef99d1acf)
- svgop at 97.7% hits: "no MoE-serving leap in our regime ... miss terms worth only +1-3%". — [714f10bebc](https://github.com/Niko1221/Strata/commit/714f10bebca16cce711135497ccc1fa821300b62)
- Direct-DMA tier promotions (`STRATA_GLM_TIER_DIRECT=1`) copy from the registered GGUF mapping instead of a staging-thread memcpy, "freeing host memory bandwidth for the CPU experts": 18.1 -> 19.0 tok/s (Ryzen 3950X, DDR4-2133, RX 9070 XT). — [lighttransport e5232ca3f2](https://github.com/Niko1221/Strata/commit/e5232ca3f2e926ffb41f9d3ff686efd4c8cc91de)

### Inferences
- Already in pancake: k-th-miss admission (`STRATA_ADAPT_KMISS`, which covers frequency/PROMOTE_MIN), decaying usage counts (`--adapt-decay`), conversation/domain seeding, GPU lookahead, and ADAPT_FETCH (inert at `--pcie-frac 0`). L3 prefetch was **measured worse** as `STRATA_ARENA_WARM`, and lookahead promotions were **closed** (recall 19-21%, item 27/30). Pancake's arena is `cudaHostRegister`ed (`pinned.cu`) and adaptive swaps use `copy_blobs`, so TIER_DIRECT is probably already the case (inferred from the code, not traced).
- `STRATA_ADAPT_PROMPT` (NEW, exact, S effort): it is complementary to `STRATA_CONV_WORKSET` (resumed conversations) because it seeds fresh prompts. RandoTeam's negative result was for moving the cache **during** the prompt. ADAPT_PROMPT only adds counts, and the swaps happen in decode. Expected gain: first 100-300 decode tokens after a long new prompt, perhaps +2-5% on edit turns (a guess). A/B: edit preset with `STRATA_ADAPT_PROMPT=32/128`.
- `--light-expert-cpu` (NEW, exact, S-M): on the IQ2_XS file the blobs differ by at most about 15% (IQ2_S 1.51 MB vs IQ2_XXS about 1.31 MB vs IQ1_M smaller). That gives at most a few % fewer CPU bytes; I estimate 0-2% end to end. It is not useful on the uniform Q2_0 file.

### Gaps
- No fork published hit rates for frequency admission against a k-miss tier on the same trace. Pancake's `replay_seeding.py` / cache_sim could score Aleks-Che's policy offline at zero GPU cost.

## 3. Routing-aware lossy ideas: cache-aware routing guards, expert skipping, fewer experts

### Takeaway
Forks agree on the shape of the curve: skipping or re-routing low-weight **misses** buys large speed at a real quality
cost, and renormalizing the dropped weight makes quality worse. One careful study (lighttransport, MiniMax) found that
miss-targeted dropping is no better than uniform top-k at the same quality. For pancake's `turbo` (`STRATA_ROUTE_PRIOR=0.5`)
the useful new items are **quality guards** (protect the router's top N, keep picks with p ≥ x·p_top, scope gating that
keeps the bonus off inside tool calls and code fences) and a **logit-range-scaled bias**. These could allow a stronger
prior on agent turns without breaking tool calls.

### Cited Findings
- **`STRATA_ROUTE_PRIOR_PROTECT=N`** (the router's first N picks are never displaced), **`STRATA_ROUTE_PRIOR_KEEP_P=x`** (picks with p ≥ x·top are kept), and **`STRATA_ROUTE_PRIOR_SCOPE=answer|think|all`** (the bonus sits behind the same structure gate as CAR: never inside `<tool_call>` or a code fence, nor in windows whose drafts cross a boundary marker; prompt reads are off). All are opt-in. — [zsariboga 8d7042ad97](https://github.com/Niko1221/Strata/commit/8d7042ad97c0e3b1a96667c341cbbd9111c657fb)
- **`STRATA_ROUTE_PRIOR_KEEP=k`** (the route prior acts only below the router's own top k) and **`STRATA_ROUTE_PRIOR_WARM=lw`** (a smaller bonus for page-cache-resident experts outside VRAM, via a residency sweep every `_WARM_MS`). — [mandoof1 3e0489e118](https://github.com/Niko1221/Strata/commit/3e0489e118cd89cd241a365cfab008c746bb8efd)
- Per-layer maps: `STRATA_ROUTE_TAIL_SKIP_MAP` (per-layer tail-skip rank, after Project Maya's TRUNK_KEEP_MAP) and `STRATA_ROUTE_PRIOR_MAP` (per-layer prior bonus). — [zsariboga 48aad94039](https://github.com/Niko1221/Strata/commit/48aad94039abaec3e14d26fd0d5c935632ec4947), [8b1454dda0](https://github.com/Niko1221/Strata/commit/8b1454dda0)
- `--cache-prior L` with the bias scaled by the token's own logit range (`cache_prior·(max-min)`, arXiv 2412.00099 Eq. 9), and `--ser Kmin,t` (ik_llama PR 239 "SER": ranks ≥ Kmin are kept only if p > t·p_top; a dropped pick becomes a weight-0 copy of the first pick). Bit-identical when off. — [gputier 88a9b1a26c](https://github.com/Niko1221/Strata/commit/88a9b1a26c4feea8e6b4e5c1b55dc9d0edaf432c). Their per-layer SER table: one layer at Kmin 6 moves KL by noise; a 17-layer table KL 0.0981 fails; an 8-layer table passes (KL 0.0557) but is "53,4 contre 53,4 tokens/s" and -2.3% after 34K, so they stopped that line. — [93f7c8ec32](https://github.com/Niko1221/Strata/commit/93f7c8ec32a5b0c7eb029ec326d361fe06095e01), [a19b150f90](https://github.com/Niko1221/Strata/commit/a19b150f9080ecf449acee35b9dd345512fad44f)
- **Miss-targeted dropping vs uniform top-k (MiniMax M3.1)**: `missk=`/`missw=` drop only missed experts; `sub=1/2` substitutes the best resident among the next four ranks. Every miss-targeted arm failed the quality gate, while uniform top-3 passed at similar speed. "a dropped missed expert is heavier than the global lightest one", and "a resident stand-in recovers little". — [lighttransport 2ea0c1e042](https://github.com/Niko1221/Strata/commit/2ea0c1e0424e4dbf6d4bdd475ba5920c8e85726f)
- Weight-threshold skipping of misses (Nemotron, 22 routed experts, weights ×5): skip-w 0.1 gives PPL ×1.003 and 43 -> 45 tok/s; 0.2 gives ×1.017 and 73; 0.3 gives ×1.046 and 113; 0.4 gives ×1.091 and 129 (37% of routings dropped). With next-layer prefetch on, 0.4 gives ×1.008 and 85. "Giving the dropped weight to the kept experts (renormalizing) makes it worse". — [Cudila 8dff8dc321](https://github.com/Niko1221/Strata/commit/8dff8dc321075912c901d89ab1bab8b072c3bf80)
- `STRATA_COLD_SKIP_BELOW=<w>` / `_MAX=<n>` skips light **cold** (not-in-page-cache) misses and warms their blobs in the background; `_WARM_MIN` and `STRATA_DEFER_COLD[_LAYERS=k]` (cold CPU experts join the residual up to k layers late). This is the SSD tier; mandoof1's measured gamer-16gb box runs `STRATA_COLD_SKIP_WARM_MIN=0.10` under turbo. — [bbf90ac6f8](https://github.com/Niko1221/Strata/commit/bbf90ac6f8d17b56b81af32360bfeb660e5def49), [32afa967f3](https://github.com/Niko1221/Strata/commit/32afa967f38e66d5fc0c5229cc3cd5f478a97a88), [e6a867bf3b](https://github.com/Niko1221/Strata/commit/e6a867bf3b4940e22f299192816a367112198099)
- `STRATA_GLM_DEFER_ROUTES=n` defers only the n lowest-weight cold routes per layer: n=1 23.4 tok/s (teacher-forced KL 0.221 vs 0.157 undeferred), n=3 24.2 (DDR4-2133 B550). — [lighttransport e5232ca3f2](https://github.com/Niko1221/Strata/commit/e5232ca3f2e926ffb41f9d3ff686efd4c8cc91de)
- Fewer experts per token on DeepSeek: `--experts 4` adopted, +15.1/+17.4/+16.1%, MMLU 90.0 vs 88.57 — [svgop d3c72b3413](https://github.com/Niko1221/Strata/commit/d3c72b34131b0056f56f89a871fae9f5f1d87797). Route-resident 0.25 adopted on another fork (+4.5% tg mean, +7.7% continuation, needle 6/6) — [ProTekk 81346bc5ac](https://github.com/Niko1221/Strata/commit/81346bc5ac0ea003d2b967ac9635fe3af4219072). Skipping small-weight misses (`STRATA_Q36_SKIP_THR=0.10`, `STRATA_NH_SKIP_REL=0.5`) was one of the two large steps on Ornith 122B (3.3 -> about 17 tok/s, together with `--pcie-frac 0.5`), while "Flash-Next does not need them" — [sterlenjohnson e9e68ae0ae](https://github.com/Niko1221/Strata/commit/e9e68ae0aea143944bb35e0024e1cf22e1adbcb4)

### Inferences
- Already covered in earlier reviews: Strata-GLM `SKIP_MINW=0.10` (fast report 2.8), serboox K=9/8 with a KLD gate (November report 4.1), HotRod stand-ins and weight×gap rule, upstream `ROUTE_TAIL_SKIP=7` (pancake: +18-22% decode, closed for default), `ROUTE_RESIDENT` (slower on 3090), PR #5655 NaN fix.
- NEW for pancake's turbo: `PROTECT=1..2` plus `SCOPE=answer` (S effort: `route_prior.cu` keeps ranks < N from the unbiased logits; host-side scope gate from the existing typing state). Expected effect: no speed by itself, but it could let turbo run lambda 0.5 -> 0.7+ inside prose while keeping tool-call JSON exact. With the measured turbo curve (0.5 gave +25% decode, 2.9% tokens differ), a safely stronger prior might add +5-10%. That needs the token-forcing gate.
- The logit-range-scaled bias (gputier) is a better-conditioned variant of our absolute lambda. A sweep on the quality-A/B tooling would show whether it moves the speed/quality frontier.
- `COLD_SKIP` and `ALT_EXPERTS MODE=cold` key on page-cache residency, which pancake does not have (the arena is all in RAM). Not applicable.
- lighttransport's result is a warning for turbo-style ideas in general: on MiniMax, miss-targeted dropping was dominated by uniform top-k. Pancake should A/B turbo against a uniform `TAIL_SKIP` at **equal KL**, not equal speed. The fork data conflict here: Strata-GLM and Cudila report favorable curves, while lighttransport reports an unfavorable one (different models and gates).

### Gaps
- No fork measured PROTECT/KEEP_P/SCOPE effects on speed or on tool-call failure rates. Their benefit is my inference.

## 4. The CPU expert path: kernels, thread pool, memory (AVX2 only, 8 Zen 3 cores, DDR4)

### Takeaway
The kernels are near the bandwidth wall, so most fork kernel work gives 0-5% end to end on our box, and AVX-512/VNNI
work does not apply. One exact item is clearly missing from pancake: the **faster bit-exact Q2_0 AVX2 unpack plus R-row
grouping** (imanu86, `STRATA_Q2_LEGACY` for A/B). It matters because every down is Q2_0, and the Q2_0 file is entirely
Q2_0. The other memory-side items are already in pancake or covered: hugetlb arenas, fault-free placement, prefetch
distance.

### Cited Findings
- **Bit-exact faster Q2_0 AVX2 kernel** (imanu86, default on, `STRATA_Q2_LEGACY=1` keeps the old one). It unpacks the codes with 8 vector ops instead of about 17, broadcasts d once per block for all tokens, and works 4 rows together for one token (2 for two tokens). 570,634 outputs are bit-identical. One thread: 1.11-1.45x. Real pool with 7 workers: "down Q2_0 -4%, gate/up Q2_0 -8%, tutto Q2_0 -17% per strato"; at 131K it is in the noise. — [c3fc8f34d3](https://github.com/Niko1221/Strata/commit/c3fc8f34d37bfcfe3855f21d532f49c3d861771a)
- IQ3_S/IQ2_S AVX2 decode without per-index shifts (upstream PR #930: `HiSpread` table, `sgn_vec_at` single pshufb). Pool on IQ2_S layers -20..-33% per window. — [imanu86 862c18aaea](https://github.com/Niko1221/Strata/commit/862c18aaea2152b205d7a64231c34869a607a0cd)
- One-token exact AVX2 gate/up rows (upstream PR #2215, `STRATA_CPU_SINGLE_EXACT`, default on in that fork) and #1415/#1544 (`STRATA_CPU_IQ_ALL_EXACT1`). — [mandoof1 5eb1c5be9f](https://github.com/Niko1221/Strata/commit/5eb1c5be9f71237ed904a2ea9c50527463fd093a), [39b75623db](https://github.com/Niko1221/Strata/commit/39b75623db)
- Software prefetch 2 KB ahead in both Q2_0 streams (codes and scales). Single thread from DRAM: 311 -> 176 µs per expert at one token, 602 -> 329 at four tokens. Bitwise identical, default on (`STRATA_OLD_CPU_PREFETCH=1` turns it off). There is also opt-in `STRATA_CPU_INT_CORR=1` (AVX-512 only: integer removal of the Q2_0 offset). — [gputier 4ff9e0a194](https://github.com/Niko1221/Strata/commit/4ff9e0a1945c066f0b5fb3bbcf301e30d5123582), [7e1cb22c2b](https://github.com/Niko1221/Strata/commit/7e1cb22c2beaec3097c3d88b5dc766191861d04d)
- AVX2 i-quant row prefetch (`STRATA_IQ_PREFETCH`, default 2048 B, T0 hint: "The non-temporal hint measured worse"). On a Zen 3 5700X3D IQ3_S decode: -4% gate/up phase and -1.3% ms per round. It also caches IQ4_NL scales. — [vvvDuB f975317267](https://github.com/vvvDuB/Strata/commit/f975317267a9b95b4df8503f2fbcdb832c9aee7e); the NTA variant was also neutral on gputier's box (53.1 vs 53.0) — [e56e4b5d5e](https://github.com/Niko1221/Strata/commit/e56e4b5d5e06d5ea96393493fed86933e4f9732f)
- `STRATA_CPU_REPACK_MIB=N` keeps a 4-bit-code CPU copy of the gate/up rows of the most-used CPU experts (146 B per 256 values, bit-exact; "IQ2_XXS measured no faster"). — [zsariboga 3eec27def1](https://github.com/Niko1221/Strata/commit/3eec27def1079b615c38c18a822eff94b69e65e7)
- Pool idle policy: spinning 10 ms before napping (instead of about 0.8 ms) took CPU experts from 15.1 to 12.6 ms per token (Zen 5 laptop, RTX 5090 Laptop). — [Jackwwg83 a83841cb5c](https://github.com/Jackwwg83/Strata-DS/commit/a83841cb5c1cde7e32370aa60dcff04ffb57b762)
- Hugepages: `MADV_HUGEPAGE` on the 4 KB fallback of the expert arena — [gputier 785f0bb4d5](https://github.com/Niko1221/Strata/commit/785f0bb4d5); svgop `STRATA_HUGEPAGE_ARENAS=1` (leg-3 solo +7.2%) — [df1a9c9000](https://github.com/Niko1221/Strata/commit/df1a9c9000); mlock of the ROCm-mapped arena — [Ben1078 bd0e37bb4b](https://github.com/Niko1221/Strata/commit/bd0e37bb4b); and a warning that THP compaction plus swap appear when a 108 GB resident model uses huge-page arenas — [lighttransport 37e0ed8f16](https://github.com/Niko1221/Strata/commit/37e0ed8f16c8c9447941f1d2b3a7343dfc9c29ae)
- NUMA (`STRATA_POOL_PER_NODE`, `STRATA_NUMA_PLACE`, `STRATA_DS41_LANE_NUMA`) and AVX-512/VNNI/VBMI kernels (bsorensen110 q2_0 v2 `aa2284e9e0`, Spark-NV `STRATA_AVX512_EXPERTS`, serboox `STRATA_IQ_VBMI`) are numerous. — [eelgaev 1c9c0ba929](https://github.com/Niko1221/Strata/commit/1c9c0ba929), [bsorensen110 aa2284e9e0](https://github.com/Niko1221/Strata/commit/aa2284e9e0e22235da721442737347ad0f9262fc)
- At the bandwidth wall: atmJava "the expert kernel is already at HBM peak, the idle sits in the CPU pool" — [541b5d0ead](https://github.com/Niko1221/Strata/commit/541b5d0ead); svgop DeepSeek "lane at the DRAM ceiling" — [b23b7472f2](https://github.com/Niko1221/Strata/commit/b23b7472f2)

### Inferences
- **Q2_0 fast unpack (NEW for pancake, exact, S effort):** pancake's `q2_avx2.cpp` still has the 14-op `unpack64` interleave and no `unpack64_fast` or `rows_fast_group` (grep). At about 80% of DDR4 bandwidth most of the 1.11-1.45x single-thread gain disappears, but the R-row grouping overlaps FMA chains, and the pool's tail tasks are compute-visible. Estimate: CPU bucket -2..-6%, so **edit +1-3%, chat +0.5-1%**; more on the Q2_0 file than on IQ2_XS. A/B: `STRATA_Q2_LEGACY=0/1` on the daily preset with `--runs 3`. It must pass the existing parity tool.
- Already in pancake (grep of `src/`): PR #930 (`HiSpread`, `sgn_vec_at` in `iq_avx2.cpp`), AVX2 `STRATA_IQ_PREFETCH` (default 2048), `STRATA_Q2_PREFETCH` (item 40, 0-5%), the one-token exact path (`iq256_gu_rows_exact_one`, `STRATA_CPU_IQ_ALL_EXACT1`), bit-plane Q2_0 (noise), hugetlb/THP arena (`pinned.cu`; svgop's complement path is in the November batch5 F2), and spinning workers (pancake parks by spin or MWAITX, never naps; Jackwwg83's nap fix does not apply).
- Expected losers on the 5700G: CPU_REPACK (it adds about 78% more bytes per repacked row versus IQ2_S's 82 B per 256 on a byte-bound pool), NUMA (single node), and all AVX-512/VNNI work.

### Gaps
- I did not find a fork measurement of the IQ2_S AVX2 path's GB/s on a Zen 3 desktop. That number would tell whether the Q2_0 file's +5-9% comes from bytes or from decode cost.

## 5. PCIe streaming, pinned memory, shared expert, and RAM/VRAM budget tuning

### Takeaway
On CPU-bound offload, forks with weak CPUs gain from sending part of the misses over PCIe. Pancake measured the
opposite on the 7900 XT (`--pcie-frac 0` is +6/+13%), so the PCIe-feeding family is low priority. The shared-expert
work is launch fusion (GPU floor), not bytes. Budget items are mostly already in pancake or covered by earlier reviews.

### Cited Findings
- `--pcie-frac 0.5` was one of the two large steps on Ornith 122B on a 6-core Zen 2 VM (3.3 -> about 17 tok/s with skipping). On that box the CPU pool streamed 97.2 GB in 12.1 s busy (8 GB/s average). — [sterlenjohnson e9e68ae0ae](https://github.com/Niko1221/Strata/commit/e9e68ae0aea143944bb35e0024e1cf22e1adbcb4). svgop adopted `--pcie-frac 0.55` at +2.8% pooled M1 — [324da1b2e0](https://github.com/Niko1221/Strata/commit/324da1b2e0)
- PCIe feeding from the mmap through pinned slots, with per-layer balancing against the pool (`STRATA_PCIE_MMAP`, `_AUTO`, `_AUTO_MAX`, `_SLACK`, `_SPLIT`, `_PREAD`, `_CPUS`, `_NICE`). — [mandoof1 93156454be](https://github.com/Niko1221/Strata/commit/93156454be), [1d5da489df](https://github.com/Niko1221/Strata/commit/1d5da489df)
- Probing the real host->device bandwidth; below 4 GB/s the PCIe share is 0 (x1 links). — [VlDubr d9a497468f](https://github.com/Niko1221/Strata/commit/d9a497468f)
- Shared expert: fused gate/up pair in the router + doorbell launch (`STRATA_FUSE_SHBEL`, `STRATA_FUSE_SHPAIR`, `STRATA_FUSE_SHGATE`) — [this-cute-world 1c0772a666](https://github.com/Niko1221/Strata/commit/1c0772a666), [fb5902f811](https://github.com/Niko1221/Strata/commit/fb5902f811); shared-expert stream fork after the doorbell when CPU experts are in the window (gfx1151 +1.8% / +6.7%) — [VlDubr 5d2fcdccac](https://github.com/Niko1221/Strata/commit/5d2fcdccac); "verify windows read a shared expert once" — [Jackwwg83 8ebc827272](https://github.com/Jackwwg83/Strata-DS/commit/8ebc827272)
- VRAM budget: svgop's 1024 MiB reserve "PRODUCED A HARD STALL" (VRAM at 151-498 MiB free); they went back to 2000 MiB, and "the 597-slot gain is not worth an intermittent stall" — [5c46e95b23](https://github.com/Niko1221/Strata/commit/5c46e95b231a791d3aca77c43538952f2e1d3da2). Sizing the expert cache from the OS's RAM, not the cgroup's cap — [shoftee 48defc12e8](https://github.com/Niko1221/Strata/commit/48defc12e8). Counting clean active file cache in resident admission — [ScrambledEggsOnToast 357e61d54f](https://github.com/Niko1221/Strata/commit/357e61d54f46ab9e59d21055968faf28604ff343)

### Inferences
- Pancake measured PCIe off as faster (+6/+13%, Round A, `STRATA_PCIE_OFF`). The forks that won with PCIe have CPUs that stream 8 GB/s or less, or IQ3_S blobs. The PCIE_MMAP family and FETCH_ADMIT stay parked unless the CPU bucket grows (for example a bigger model).
- Shared-expert fusions reduce launches on the GPU floor (15.8 ms), not expert bytes. They are covered in the November report (3.5 `FUSE_SHGATE` etc.).
- Budget: pancake's reserve tests (fast report 1.6) already cover `--vram-reserve-mib`. svgop's stall at 1 GiB reserve on a 32 GB card argues against cutting ours below about 700 MiB on the 20 GB 7900 XT.

### Gaps
- No fork has a 7900 XT + 5700G PCIe/CPU split curve for Flash-Next. Pancake's own Round A is the only data point.

## Ranked shortlist for pancake (estimates are mine, unmeasured)

| # | Idea | Fork / commit | Exact? | In pancake? | Effort | Est. gain on our box |
|---|---|---|---|---|---|---|
| 1 | Sub-2-bit (IQ1_M/IQ2_XXS) shadow store for CPU misses only, on tmpfs/memfd hugetlb | svgop [25bf1b90f6](https://github.com/svgop/Strata/commit/25bf1b90f645dd5a44344bb4d04c7042805db95f), [8aa066b17d](https://github.com/Niko1221/Strata/commit/8aa066b17d5cd5b75aac570261568e6e6b2fe283); mandoof1 ALT_EXPERTS [85b991da79](https://github.com/Niko1221/Strata/commit/85b991da79303fcd018bcc90f1a763067c93fa1e) | lossy (misses only) | no (Q2_0 shadow ≈ our Q2_0 file) | L (requant + pack + 6 sites) | edit +7-10%, chat +3-5% if byte-bound; quality unknown |
| 2 | ISTA Coder (256/512 experts), or a masked-router subset | svgop [9e309adb39](https://github.com/Niko1221/Strata/commit/9e309adb39b7d71c7e7279035b0320bd16a6e65f); VlDubr [f07c68c7c2](https://github.com/Niko1221/Strata/commit/f07c68c7c2) | lossy (different model) | MOD_PLAN item 25, parked | M (download + profile) | ≥ +8-11% (svgop analogue), likely more at our 80% hit rate |
| 3 | Route-prior guards: PROTECT, KEEP_P, SCOPE (no bonus in tool calls/fences); range-scaled bias | zsariboga [8d7042ad97](https://github.com/Niko1221/Strata/commit/8d7042ad97c0e3b1a96667c341cbbd9111c657fb); gputier [88a9b1a26c](https://github.com/Niko1221/Strata/commit/88a9b1a26c4feea8e6b4e5c1b55dc9d0edaf432c) | lossy-guard | no | S | 0 alone; could allow a stronger turbo (+5-10%) at the same agent quality |
| 4 | Faster bit-exact Q2_0 AVX2 unpack + R-row groups | imanu86 [c3fc8f34d3](https://github.com/Niko1221/Strata/commit/c3fc8f34d37bfcfe3855f21d532f49c3d861771a) | exact | no | S | edit +1-3% (more on the Q2_0 file) |
| 5 | Prompt routing feeds the adaptive tier | zsariboga [4ce19344ff](https://github.com/Niko1221/Strata/commit/4ce19344ff72c8ca9a1175a01d357c1179a521e8) | exact | no | S | +2-5% on the first decode tokens after a new long prompt (guess) |
| 6 | Byte-aware placement (light blobs to CPU) | tctien342 [5576b48144](https://github.com/Niko1221/Strata/commit/5576b481449c638517c1835c15bb053bd1443ab5) | exact | no | S-M | 0-2% (IQ2_XS file only) |
| 7 | Frequency-gated admission (offline replay first) | Aleks-Che [0561016de8](https://github.com/Aleks-Che/Strata/commit/0561016de87e1962285903f9b547187b4ca787de) | exact | partly (KMISS) | S in sim | unknown; replay before coding |

Do not repeat (measured or closed): L3/arena prefetch of predicted experts (pancake `ARENA_WARM` worse; svgop -8%),
lookahead promotions (recall 19-21%), CPU_REPACK (more bytes), PCIe share on the 7900 XT (pancake: off is faster),
COLD_SKIP/ALT_EXPERTS cold mode (SSD tier only), per-layer SER tables (gputier: no speed), miss-targeted dropping vs
uniform top-k (lighttransport: dominated), AVX-512/VNNI/NUMA work (no hardware).
