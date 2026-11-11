# finemoe: FineMoE expert-map prefetch, PRI eviction, MoE-Infinity baseline, A7 / A8 / A46 re-judged (2026-11-11)

Branch `dev/p16/finemoe` (from dev/phase16-academic 3f4ba97cf). MOD_PLAN 16, development order item 1 and section 16.1.
Offline only: no GPU, no engine, no model loaded. **Result: no engine switch.** No replayed policy gains 2% of modeled
decode time over the best existing tier (best online candidate +0.5%, worst sequence -1.2%); FineMoE as published is
-14% on the bench trace.

## What was built

- `tools/pancake/finemoe_replay.py` (new): the FineMoE expert-map store and searcher, the delta prefetch rule, PRI
  eviction, a MoE-Infinity EAM baseline, LRU / LFU / PRI eviction under miss / k-th-miss / no admission, all on the
  same per-layer cache model as `time_ceiling.py` and priced with the A6 model (`time_ceiling.py --validate`, mean
  error 3.4 points). Also prints the A46 rows (cache_learn: opt, label, hawkeye-pc-h16), the A7 bounds and the A8
  swap budgets for every replayed sequence, and the 2% verdict.
- `tools/pancake/time_ceiling.py`: `parse(argv)`, `fit_model(a, base)`, `model_ms(...)` lifted out of `main` and
  `validate_report` so the replayer reuses the calibration instead of copying it. Output byte-identical (below).
- `tools/pancake/cache_learn.py`: `run_opt` / `run_predicted` take an optional `ml` array (misses per window and layer)
  so the A6 model can price them. Default calls unchanged.
- `tools/pancake/test_finemoe_replay.py` (new, 13 tests).
- Raw output: `bench/results/2026-11-11-finemoe/finemoe_replay.{txt,json}`.

How the replay models FineMoE in our engine: one FineMoE iteration = one verify window (about 4.7 tokens, all pass a
layer together); the map of a window is P_l per layer (router weights when the trace has them, else top-k counts);
layer l >= d is predicted from the stored window whose layers 0..l-d match best (cosine, eq. 5), layers < d from a
semantic stand-in; the prefetch set is the delta rule (at least K = top-k = 10); the copies run on the A6 model's
layer clock (preset deferral window 33.76 ms / 48 layers = 0.70 ms per layer, router at +0.33 ms, one copy 117 us,
one serial PCIe channel that also carries the previous window's admissions), a copy that lands after the layer's
experts start is a miss this window; misses are computed by the CPU and admitted between windows (miss = FineMoE's
on-demand load, kmiss = STRATA_ADAPT_KMISS, none = prefetch only). `cover` = the share of the misses-without-prefetch
that the prefetch sets contained (the predictor's recall where it matters).

## Traces

Real agent-session routing traces do not exist yet. Searched bench/, bench/results/, every Strata worktree and
Strata-data: the only routing trace is `bench/prefetch/routing.bin` (main checkout, gitignored: the 2026-11-08
trace-bench trace, 219 windows, 4 requests read4k / short / edit / long cut at 11 / 96 / 157, top-10, **no router
weights**). Everything below is on that trace, per request (cold from the profile), the whole trace, and the
synthetic agent-like interleave of cache_learn (8-window turns, round-robin; marked SYNTHETIC).

    python3 tools/pancake/finemoe_replay.py data/expert-profile-pancake.bin 10909 bench/prefetch/routing.bin \
        --cuts 11,96,157 --names read4k,short,edit,long --interleave 8 --json out.json      # 2 min 36 s here

## Results (modeled decode speed vs the preset tier; hit = routed entries served from VRAM)

| Policy | whole: hits | whole: vs tier | interleave: hits | interleave: vs tier | cover |
| --- | --- | --- | --- | --- | --- |
| tier (preset, the calibration) | 81.2% | 0 | 81.8% | 0 | - |
| kmiss 2/8 (existing opt-in) | 86.5% | +2.7% | 84.0% | -2.0% | - |
| FineMoE faithful (d 3, prev-sem, miss admission, PRI) | 88.0% | -14.1% | 87.1% | -17.1% | 2% |
| FineMoE, ideal DMA | 88.3% | -13.8% | 87.4% | -16.7% | 2% |
| FineMoE + k-th-miss admission | 85.6% | -3.2% | 85.1% | -3.1% | 4% |
| FineMoE prefetch only | 81.9% | -0.6% | 82.7% | +0.2% | 5% |
| FineMoE whole matched map (delta 1) + kmiss | 86.0% | -15.0% | 85.4% | -15.6% | 17-18% |
| FineMoE oracle semantic, ideal DMA | 88.3% | -13.9% | 87.4% | -16.7% | 2% |
| FineMoE d 1 / d 6, Map(T), top-K | 88.0% | -13.5..-14.2% | 87.1% | -16.8..-17.2% | 1-2% |
| no prefetch, kmiss + LRU / LFU / PRI | 86.6 / 84.8 / 85.1% | +2.7 / -2.3 / -1.6% | 84.2 / 84.5 / 84.6% | -1.8 / -2.1 / -1.9% | - |
| no prefetch, miss + LRU / LFU / PRI | 89.6 / 87.4 / 88.0% | -7.1 / -14.8 / -12.9% | 85.0 / 86.8 / 87.1% | -21.1 / -17.0 / -16.1% | - |
| MoE-Infinity EAM online / offline | 84.7 / 86.6% | -23.2 / -18.7% | 80.9 / 84.7% | -31.6 / -23.2% | 0-1% |
| A46 hawkeye-pc-h16 (online) | 87.5% | +0.9% | 84.2% | -7.6% | - |
| A46 opt / label (oracles) | 94.9 / 94.2% | +19.4 / +18.2% | 93.9 / 92.9% | +16.9 / +14.6% | - |

Per request (cold, from the profile): the best online candidate is within +0.5% / -1.2% of the best of tier and
kmiss 2/8 on read4k, short, edit and long; FineMoE with the store prefilled by the other requests (the paper's 70/30
offline setting) changes the hit rate by < 1 point (cover 1-5%). Mean best-match cosine 0.52-0.64.

Why it does not work here (numbers from the replay): the cache already keeps 44% of the experts (10,909 of 24,576) at
81-89% hits, so the experts a past window's map predicts are mostly resident (0-18 prefetches per window against 50-730
misses); the misses are the tail that no stored window contains (cover 0-5%; even the whole matched map 17-18%). The
semantic stand-in is not the limit: a perfect first-d-layer match (oracle) gives the same result. Every-miss admission
(FineMoE's on-demand load) costs 247 copies per window on the whole trace (265 on the interleave); through the DDR4 contention of the A6 model (alpha 0.5)
that is -14%, and some windows need more DMA than the window lasts (rows marked DMA-gated in the txt; the A6 model does
not gate them, so those rows are optimistic).

PRI eviction vs the tiers' LRU / LFU (item 2): with k-th-miss admission (the engine's online tier) PRI is never more
than 0.3 points above LRU and is 1.5 points below it on the whole trace (sequential requests); PRI only wins with
every-miss admission on the interleave, an arm that loses 16% to admission cost anyway. FineMoE's argument against LRU
("layers run sequentially") does not hold for per-layer caches with window recency: every window touches every layer.

A7 / A8 / A46 re-judged (bench trace only, as no agent trace exists):
- **A7 ProMoE**: upper bounds over the tier, both effects (no DMA contention + first use from the in-flight copy):
  whole +4.6%, edit +6.0%, interleave +4.5%, short +3.3%, long +0.5%. Bounds, not gains; < 5% on the whole trace.
  Stays parked.
- **A8 Pie** (fixed swap budget on the blocking tier, the ceiling of an online budget): whole 128 swaps +3.0%, edit
  128 +12.2%, interleave 64/128 +1.4%, short +0.6%, long +0.1%, read4k 8/16 +0.4% (128: -2.7%). The only candidate
  over 2% on the whole trace, driven by the edit request; it needs no code (`--adapt-swaps 64|128`, and the coded
  `STRATA_ADAPT_SWAPS_AUTO`): an env-only screen on edit / agent traffic when the GPU is free.
- **A46 learned admission**: hawkeye-pc-h16 +1.0 hit point over kmiss 2/8 on the whole trace (+0.9% vs tier, -1.7% vs
  kmiss), -7.6% on the interleave; the oracles show +16..19% of room that no online learner reaches. The rule (>= 5
  points over k-th miss, held out) is not met: closed until agent traces exist.

## Source check (primary sources read: arXiv HTML of each)

**FineMoE** (EuroSys'26, arXiv 2502.05370 v2). The plan matches the paper: per-iteration expert maps (gate softmax per
layer, §4.1), semantic cosine on the embedding-layer output for layers 1..d and trajectory cosine on layers 1..l-d for
layer l (eq. 4, 5), delta = clip(1 - score, 0, 1), experts in descending probability until the sum reaches delta, at
least K (eq. 6-8), store capacity 1K with RDY deduplication (§4.4), eviction PRI = 1/(p * freq), LFU-based, recency
ignored (§4.5), prefetch priority p / (l - l_now). Exact (placement only). One correction to the plan: "d = 3 tuned"
is the setup default; the sensitivity study sets d = 3 / 6 / 4 for Mixtral / Qwen1.5-MoE / Phi-3.5-MoE (the replay ran
d 1, 3, 6: no difference). Where our setting breaks its assumptions:
- Miss = blocking on-demand PCIe load in the paper (its objective is the on-demand load time, eq. 1); in our engine a
  miss runs on the CPU pool (48.7 us per expert in the A6 fit) and a copy is optional and costs DDR4 bandwidth (117 us
  of DMA, alpha 0.5). A correct prefetch saves about what its copy costs unless the expert is reused later.
- Hardware: 6x RTX 3090 with NVLink pairs and expert parallelism, PCIe 4.0 at 32 GB/s, 480 GB RAM; ours one RX 7900 XT
  on PCIe 3.0 (~12 GB/s effective) and DDR4: a 1.4 MB IQ2_XS/Q2_0 expert copy is 117 us against 0.70 ms per layer, so
  d = 3 gives room for about 12 copies, and 74% of the faithful prefetches land late (the channel also carries the
  admissions).
- Models fp16, 8-60 experts per layer (Mixtral 2/8 x 32 layers, Qwen1.5-MoE 4/60 x 24, Phi-3.5 2/16 x 32); ours 10/512 x
  48 at ~2 bits: the paper's own bound needs 2LJ = 49,152 maps for 75% similarity; one session has hundreds of windows.
- Iteration = one token in the paper; ours = a speculative verify window (~4.7 tokens), so a map is a union.
- Cache size: the paper's interesting regime is a small cache (down to 6 GB); ours keeps 44% of the experts resident.
- Inputs it needs that our traces lack: embeddings (semantic stage replaced by a stand-in) and router probabilities
  (the bench trace's weight column is zero; maps are top-k counts). `wtrace.read` takes weights from STWT traces.
- Claimed 47% latency and 39% hit rate are upper bounds from that testbed; none transfers (replay above).

**MoE-Infinity** (arXiv 2401.14361, v3 read; baseline only, as the plan says). Request-level EAMs, a collection matched
by cosine, prediction normalised per row with layer proximity, eviction by predicted activation (Algorithm 1), next-layer
prefetch. v3 claims 3.1-16.7x per-token latency (against vLLM, Ollama, DeepSpeed, BrainStorm) on one RTX A5000, PCIe 4.0;
the v2 "2-20x" is a different version (do not mix them). Its premise (batch 1 on a personal machine) matches ours, but
the collection needs many past requests (the deep-research verified a 30-50-request warm-up); with four requests its
eviction follows another request's activations and evicts what the current request uses: -19..-32% in the replay.

**ProMoE** (A7, arXiv 2410.22134 v3): learned predictor + stride prefetch, chunked prefetch (3 chunks = the 3 linear
layers), early preemption, reordered inference, per-layer LRU cache; RTX 4090 on PCIe 4.0 x8 (23.9 GB/s measured);
decode 1.34x over hand-crafted caches in transformers, 1.09x (up to 1.17x) over LRU in llama.cpp (the plan's figure is
right). Breaks: its gains come from hiding blocking loads, which our engine does not have (misses run on the CPU), and it
needs a trained predictor (our next-layer predictors reach 19-21% recall, item 30 closed).

**Pie** (A8, arXiv 2411.09317): KV / parameter swapping for dense-model serving on GH200 with NVLink-C2C (up to 900 GB/s);
adaptive expansion grows the swapped memory while the link is not saturated, swap latency stays below compute and
throughput rises (claims up to 1.9x throughput, 2x latency vs vLLM). Not an expert-placement method: our A8 borrows only
the control rule, on a link ~75x slower, for one user. An online budget cannot beat the best fixed budget (above).

**A46 sources** (ADRS 2510.06189, Hawkeye ISCA'16, Harmonia 2503.20507): unchanged from batch 2
(pancake-docs/batch2/A7-A46-promotions.md); not re-read.

From the deep-research report (wezw1djad): HOBBIT and OD-MoE findings read; neither is built here (OD-MoE skipped by the
plan; HOBBIT is order item 2, another branch). No refuted claim from the plan's list is repeated in these notes.

## Switch

None. The replay shows no gain >= 2% in modeled decode time, so no engine code was written (MOD_PLAN 16 rule). If agent
traces later show `cover` well above 5% and a >= 2% row, the port would be a victim-order option in
`include/strata/program/kmiss_tier.hpp` (`select`'s victim score) plus a per-round map search; not before.

## Tests run

- `python3 -m pytest tools/pancake/test_finemoe_replay.py`: 13 passed (also `python3 tools/pancake/test_finemoe_replay.py`).
  Delta rule by hand, trajectory / prev / oracle search, RDY dedup, PRI / LFU / LRU victim order, FineMoE beating LRU on
  alternating patterns (and lateness with a slow channel), the EAM baseline matching the right request, the eviction-only
  replay within 1 hit point of `time_ceiling.replay` (lru, kmiss) on a Zipf trace, interleave order = cache_learn's,
  cache_learn's `ml` sums, `time_ceiling.parse` / `fit_model`, the loader = `rp.load_trace`, the command line.
- `python3 -m pytest tools/pancake/test_cache_learn.py tools/pancake/test_predict_experts.py`: 14 passed.
- `python3 tools/pancake/trace/wtrace.py --selftest`: ok.
- `time_ceiling.py routing.bin profile 10909 --validate`: byte-identical before / after the refactor (cmp), also with
  `--alpha 0.25 --sdef-us 91` (same md5); mean error still 3.4 points.
- No C++ changed, nothing built.

## Risk

One bench trace (219 windows, 4 requests), not agent sessions; no router weights; a semantic stand-in; the A6 model is an
edit-window model (short chat unvalidated) and does not gate windows on DMA; cache_learn's rows spread promotions evenly.
The conclusion rests on `cover` (0-5%), which is a property of the trace, not of the time model.

## Next

1. When the GPU is free: agent-session routing traces with router weights (conv_trace.py, or `--dump-routing` /
   `STRATA_WINDOW_TRACE` during a replay_turns.py run), then rerun the command above on them; FineMoE is worth a second
   look only if `cover` rises well above 5%.
2. A8: env-only screen of `--adapt-swaps 64` / `128` (and `STRATA_ADAPT_SWAPS_AUTO=1`) on edit / agent traffic.
3. A7 and A46 stay parked / closed until agent traces exist.
