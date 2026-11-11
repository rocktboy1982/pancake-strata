# Batch 3 - policy: A44 bandit, A45 calibration, A15 park TTL, F1 park-truncate, the conversation-cache replay

Branch `dev/p16/policy` (worktree `Strata-p16-policy`, from `dev/phase16-academic` 727eb8a62). 2026-11-11. CPU and compile-only: no GPU,
no engine, no model was started. Every switch below stays opt-in; with its variable unset the code path is the one of the base commit.

## What was done

| item | state before | now |
| --- | --- | --- |
| A45 `STRATA_CALIBRATE` | batch-2 code, never built | `generate.cpp` did not compile: the `calibrate_ratio.hpp` include was missing (21 HIP errors). Fixed (92876e761). Builds. |
| A44 `STRATA_BANDIT` | batch-2 code, never built | builds (HIP compile-only); unit test passes. One change: `STRATA_BANDIT_MARGIN` (see below). |
| A15 `STRATA_PARK_TTL_S` | batch-2 code, never built | builds; unit tests pass. One change: tool-pending entries are evicted shortest first. |
| F1 `STRATA_PARK_TRUNCATE` | unit tests written, never run | builds; the tests ran: 4889 checks passed (4897 with the new ones). |
| replay tool | dry run claimed | dry run run on the 11-09 log: 75 requests, 7 cold re-reads, 162.7 s. One bug fixed in `--shared-system`. |
| offline judge | none | `park_replay_sim` + `tools/pancake/park_sim.py`: the real `ConversationCache` driven by a replay plan. |

## Source check (does the method in the plan and in the code match what the source proposes and measures?)

### A44 - Athena (arXiv 2601.17615, ETH SAFARI, HPCA 2026): MISMATCH in method, the idea is adapted

What Athena is: a hardware agent that coordinates a data prefetcher and an off-chip predictor (OCP). It is a state-aware SARSA agent
with a tabular, hashed Q store (8 planes of 64 x 4 entries, 3 KB). State: four counters of the last epoch (prefetcher accuracy, OCP
accuracy, bandwidth use, prefetch-induced cache pollution). Actions: prefetcher and OCP each on or off (4 actions), plus the prefetcher's
aggressiveness, derived from how far the best Q value is above the others. Reward: a composite, `R = R_corr - R_uncorr`, with cycles, LLC
misses and miss latency on the first side and loads and mispredicted branches on the second, so that a change caused by the workload is
not credited to the action. Tuned values: alpha 0.6, gamma 0.6, epsilon 0.0, epoch 2K retired instructions (microseconds). Evaluated in
ChampSim on 100 memory-intensive traces. Gain: +5.0% over MAB (the paper's name for the stateless multi-armed-bandit baseline) in the
main configuration, +3.6% to +10.3% over prior policies. The paper says that IPC (or cycles) as the only reward is unreliable.

What we built is the paper's MAB baseline, not Athena: arms = subsets of the switches (stateless), reward = the epoch's time per token
(the single-metric reward the paper warns about), epsilon-greedy with eps 0.1 or UCB, epochs of 16 windows (seconds). The two window
types ("copy", "chat" by the request's lookup share) are a hand-made context, which is the useful part. Differences that matter:
1. Athena's actions never change the output. Our `top` arm (rank guard) is lossy (small). Pure cost let measurement noise choose it:
   with no true effect on any arm the bandit ran a lossy arm in 44.5% (eps) and 49.6% (ucb) of the epochs. That is not in the paper.
2. No uncorrelated reward, no state features, no aggressiveness (the plan's "left" list has it).
3. The paper's gain (ChampSim, a different system) is not a prediction for this engine. The plan's "est. +2-5%" is the plan's own estimate.
Verdict: keep, as a small opt-in tool for mixed sessions, with the lossy arm under a margin. Do not quote Athena's numbers for it.

### A45 - DALI cost model (arXiv 2602.03495): PARTIAL match

DALI (EPYC 7532 64 cores, 256 GB DDR4, RTX 3090, PCIe 4.0 x16; DeepSeek-V2-Lite, Qwen3-30B-A3B, Mixtral-8x7B; batches of 1 and up) writes the
CPU-or-GPU choice per layer as a 0-1 integer program, `min max(CPU time, GPU time + transfers)`, solved at run time by a greedy rule, and
says the hardware timing values (`trans_time`, `t_cpu(w)`) come from a warm-up profile. Its headline speedups (7.62x, 3.80x, 2.45x, 2.00x
over llama.cpp, KTransformers, MoE-Lightning, HybriMoE) are prefill figures that come from the dynamic assignment, the residual prefetch and
the cache policy together.
- Matches: the warm-up profile of the CPU lane and the transfer lane, and the balance objective. `calib_derive` computes the balance
  share `f = cpu/(cpu+copy)`. New test: for 2 to 16 missed experts the best integer split is floor or ceil of `m f` and the split saves
  exactly `f` of the CPU-only time (DALI's objective for equal-workload experts).
- Does not match: DALI assigns per layer and per step from the real workload; A45 turns the profile into one static share (this is the
  "static assignment" DALI argues against). "Verified on window time" (the plan's wording) is not built: the probe is not checked against
  a measured window. The threshold 0.35 is chosen so that this box's expected figures give 0 ("must find --pcie-frac 0 alone"): the pass
  criterion is true by construction if the box measures as assumed. Different regime from DALI's (PCIe 4.0, 64 cores).
- Gain: none expected here (share 0). The 6-13% in the plan is the loss of a wrong static share on this box, not a DALI figure.

### A15 - Continuum (arXiv 2511.02230, v7 Sept 2026): PARTIAL match, only the tool hold is Continuum

Continuum pins the KV cache of a request that ended in a tool call, for a TTL chosen per tool by `argmax_tau P(tau, f) * (T*eta +
PrefillReload) - tau` (P from the empirical CDF of the last 100 durations of that tool; T the queueing delay; eta the workload's
"memoryfulness"). It runs on vLLM (A100, H100, B200; Llama-3.1 8B/70B, Gemma-3 12B; a later version also GLM-4.5), reports 1.12x-3.66x lower
delay and up to 8.18x on a company's SWE-agent traffic, and its gain comes mostly from the queueing delay of a busy GPU server. It has no
notion of sub-agents (it lists multi-agent flows as future work) and the paper does not mention "MORI" (the plan's "same group" note is
unverified).
- Continuum, in our code: rank 3 (a conversation whose last reply ended in a tool call is held for `STRATA_PARK_TTL_HOLD_S`). Fixed 60 s
  instead of the learned per-tool TTL. There is no queue in a single-user engine, so only the reload-cost term of Continuum's benefit applies.
- Not Continuum: ranks 0-1 (the sub-agent "branch" guess from a shared prefix and a length ratio) are Strata's own heuristic.
- Bug found with the offline judge (see below): with the hold on, a main agent that waits for its Task tool and sub-agents that wait for
  theirs are all rank 3, and the oldest went first, which is the main agent. The hold cancelled the branch ranking in the case A15 is for.
  Rank 3 now evicts the shortest entry first (the cheapest to re-read; Continuum's benefit is the reload cost). Rule off: unchanged.

### F1 - TwoAnts/Strata 5ef4fc8b4b: MATCH

Read from the commit on GitHub: partial parking walks the chain's checkpoints deepest first and parks the first prefix that passes the byte
and RAM gates; the eviction side shrinks an entry to one of its checkpoints before dropping it; single-stage only; off unless
`STRATA_PARK_TRUNCATE=1`; pinned entries and the guarded donor are never shrunk. Strata's port does the same. Differences, all deliberate and
stated in the code: a landing below `--conversation-cache-min-tokens` is not parked (the fork parks it), and no truncation beside the disk
tier. The fork's number (225 s for a 160K chain) is its own; ours is the modelled 52 s below.

## Switches (unchanged unless said)

- `STRATA_BANDIT=top,kmiss,preread` (+ `_EXACT`, `_RULE`, `_TYPE`, `_EPOCH`, `_SKIP`, `_DECAY`, `_SEED`, `_LOG`) and new `STRATA_BANDIT_MARGIN`
  (default 0.02, 0 = pure cost): an arm of exactness class c (0 exact, 1 rounding, 2 lossy) is chosen only if its cost x (1 + margin c) is the
  lowest, so a lossy arm needs a 4% gain at the default. The logged cost is not charged. It is read only when `STRATA_BANDIT` is set.
- `STRATA_CALIBRATE=1` (+ `_MIN_SHARE`, `_CACHE`); `STRATA_PARK_TTL_S` (+ `_HOLD_S`, `_SHARE`, `_SUB_RATIO`); `STRATA_PARK_TRUNCATE=1`.

## Tests run (all CPU or compile-only; commands from the worktree)

- Configure CPU: `cmake -S . -B build-cpu -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON`
  (`conversation_cache_test` is registered only with `STRATA_BUILD_CONVERSATION_TESTS`; a default build does not run it).
  `ctest -R "conversation|epoch_bandit|calibrate_ratio|park_replay|session_save"`: 8 of 8 passed.
  `conversation_cache_test`: 4897 checks (4889 before this branch's 8 new ones; the F1 block, 24 check calls, and the A15 block, 23, had never run before).
  `epoch_bandit_test`: all passed, with the new margin and short-session tests. `calibrate_ratio_test`: all passed (new objective test).
  `park_replay_sim_test`: all passed (new).
- Configure HIP (compile only): `-G Ninja -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF -DCMAKE_HIP_ARCHITECTURES=gfx1100`; built the
  objects `generate.cpp.o`, `verify.cpp.o`, `expert_source.cpp.o`, `conversation_disk_tier.cpp.o`, `verify_kernels.cu.o` (the `top_live` row-copy
  kernel) and the target `conversation_disk_tier_test` (97 checks passed, includes the `STRATA_BANDIT` key). generate.cpp: 0 errors after the
  include; no warning from A44, A45, A15 or F1 code. Not done: CUDA (no nvcc on this box), the full link of `strata`, any run.
- Python: `tools/pancake/test_replay_turns.py` + `test_park_sim.py`: 12 passed; `serve.test_server.CancelledRead` (the `bandit=` DONE field): 6 passed.
- Replay dry run (`replay_turns.py bench/specs/24-a15-park-ttl.json --dry-run`, real tokenizer, the 11-09 turn log): 75 requests (34 agent turns,
  41 side), conversation B 31 turns 34,261 -> 62,943 tokens, recorded cold re-reads 7 = 162.7 s, first reads 2 = 56.4 s, sizing error 10 tokens
  (it was 1026 before the `--shared-system` fix below). Nothing was replayed against a server.

## Offline replay of the eviction order (`park_replay_sim`, `tools/pancake/park_sim.py`)

`park_replay_sim` runs a trace through the real `ConversationCache` (put, take, borrow, drop_superseded, make_room, truncate, the A15 ranking,
a fake clock from the recording). Snapshot size model from the 11-09 log (111 MiB + 14.9 KiB/token + 113 MiB per checkpoint), all byte counts
scaled by 1/20 so the 60K-token images fit in RAM; prompt time from `replay_turns.est_prompt_s`. Not modelled: RAM floor, disk tier, batch slots,
decode. It reproduces `bench/results/2026-11-10-offline/park_replay.py` (the earlier Python model) within about 3% (S0 7 cold, 162.7 s recorded
vs 168.8 s; S1 saves 52 s vs 51.5; min-tokens 21 vs 21.8; budget 2048 + min-tokens 158 vs 158).

The 11-09 trace (agent prompt time = conversation A and B), budget 1280 MiB (the 11-09 engine's), 4 slots:

| rule | agent prompt s | cold re-reads (s) | note |
| --- | --- | --- | --- |
| default | 224.2 | 7 (168.8) | 7 parks skipped: the snapshot is over the budget |
| F1 | 172.5 | 4 (97.1) | -52 s |
| min-tokens 8192 (flag, no code) | 202.6 | 6 (145.5) | -22 s |
| F1 + min-tokens 8192 | 107.8 | 1 (17.8) | -116 s |
| A15 alone (any setting) | 224.2 | 7 (168.8) | no change; A15 + F1 with a shared prefix and tool-ending turns: 153.0 s (F1 alone 172.5) |
| budget 2048 | 156.9 | 4 (97.1) | -67 s |
| budget 2048 + min-tokens 8192 | 66.7 | 0 | -158 s |

Budget 2048, same trace, with the shared 1024-token system prompt and agent turns ending in a tool call: A15 branches only 133.7 s,
A15 with the hold 87.6 s (cold 1), A15 + F1 74.8 s (cold 0); with no shared prefix and no tool call A15 does nothing (156.9 s).

Reading it:
1. A15 cannot help the 11-09 run by itself. 6 of its 7 cold re-reads follow a "skip parking, snapshot 1286-1574 MiB exceeds the 1280 MiB
   budget" and 1 an eviction (RESULTS.md 4.0.1; the sim agrees); an eviction order cannot make an oversized image fit. The A15 note's sentence that 1280 MiB "is where the side requests evicted B"
   is true for one turn only (54,313 -> 56,264).
2. F1 saves about 52 s of 163 s here, in line with the plan's 39-52 s, and is dominated by two settings that need no code: a 2048 MiB budget
   and `--conversation-cache-min-tokens 8192` (-158 s together). F1 is insurance for the case where the snapshot is larger than any budget.
3. A15's gain needs two things the trace does not show: a shared prefix (so branches exist) or a tool-ending reply (so the hold applies), and
   cache pressure from other parks. The replay spec's `shared_system 1024` builds the first property; it is an assumption, not data.
4. A15 moves the re-reads, it does not remove them. Synthetic main agent (30K -> 42K, tool-ending) with 4 parallel sub-agents (6-10K,
   3 turns, 6 rounds), budget 2048 MiB, `park_replay_sim --synthetic 6,4,3,1024`: default 5 cold re-reads of the main agent (77.9 s,
   agent prompt 90.4 s, all prompts 237.7 s); A15 hold 60: 0 cold re-reads (agent 23.7 s) but all prompts 278.7 s (+41 s), because the
   sub-agents are re-read instead. Whether that is a win depends on whose latency counts. Synthetic: not a measurement.

## Risks and limits

- The sim is a model of generate.cpp's flow (copied, not shared): if the engine's park path changes, the sim drifts. Checkpoint placement is
  periodic (every 16384 tokens) as in the earlier model; `--tail-gap N` adds one near the end. Which of these the engine really has is not known.
- A44: the bandit still compares epochs that see different text. With arm effects of 2-7% and per-window jitter this large the choice is noisy;
  the margin bounds the damage for the lossy arm, it does not make the estimate better. UCB explores more (31% lossy windows at the default margin
  with no effect, against 10% for eps). Text is not reproducible with `top` or `kmiss` in the bandit.
- A45: the probe has never run; the 0.35 threshold is a heuristic; the probe times with the pool idle apart from its own copies.
- A15: the hold is a fixed 60 s; the branch test is a prefix heuristic (a wrong guess only changes the order).
- `rankguard` (dev/p16/rankguard) edits the same rank-guard files as A44 (defer_rank.hpp, verify.cpp): expect a merge to look at.

## Next

1. A GPU run of the A15 replay is only worth it with a real sub-agent trace (a main agent + Task calls); the 11-09 trace has none. First
   run the park_sim table on that trace when it exists (`replay_turns.py ... --dry-run` then `park_sim.py plan.json --sim <binary>`).
2. A44 on the GPU: screen spec `24-a44-bandit.json` with `STRATA_BANDIT_LOG=1`; read how often the lossy arm runs on chat.
3. A45 on the GPU: `24-a45-calibrate.json`; check that the log says `pcie_frac 0.00` and the probe figures are plausible.
4. Learned per-tool hold (Continuum's empirical CDF) from the turn log, if A15 ever shows a gain.
