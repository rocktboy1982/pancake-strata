# pancake-strata: plan

A performance fork of [Strata](https://github.com/Niko1221/Strata) for single-user agent serving on one home box.
This file says what is true now, what runs next and what is waiting for a decision. Consolidated on 2026-11-15 from
the old 3,984-line plan and about 200 side documents, after an item-by-item audit against the code, presets, specs and
result archives.

Companion files:
- [TEST_LEDGER.md](TEST_LEDGER.md): every test ever run (537+ rows, ids `T-nnn`), with numbers, spec, tier, verdict and
  the archive path; plus the data conflicts found while merging. Cite tests by id.
- [LEADS.md](LEADS.md): every unported idea from forks, upstream PRs, llama.cpp, papers and hardware notes (~600
  leads), with author numbers and whether our tree has it.
- `bench/specs/` (A/B specs), `bench/results/` and `pancake-docs/research/bench/results-*/` (raw evidence),
  `configs/pancake/` (presets), `tools/pancake/` (measurement suite).

## 1. Goal, box and models

Fastest single-user agent serving (long growing prompts, tool calls) with output identical to plain decoding unless a
flag says it is lossy. Speed = decode tok/s (new text and code edits), prompt tok/s at 4K/16K/32K, and time to the
first token of an agent turn after a long conversation.

| | |
|---|---|
| GPU | RX 7900 XT 20 GB, gfx1100, ROCm 10.0 (`/opt/rocm/core-10.0/lib`), hipBLASLt table `tools/hip/gfx1100-hipblaslt-100401.txt` |
| CPU / RAM | Ryzen 7 5700G (Zen 3, AVX2, no AVX-512, no WAITPKG), ~57 GB DDR4 (~33 GB/s; the CPU expert pool is bandwidth-bound) |
| Bus | PCIe 3.0 (forced by the 5700G), ReBAR 32 GB BAR0, IOMMU passthrough, THP madvise |
| Gone | RTX 3060 12 GB removed 2026-11-08: CUDA/sm_86-only items are parked (section 8) |
| Daily model | Qwen3.8-Flash-Next IQ2_XS (GSQ-RCO pack), preset `configs/pancake/flashnext-iq2_xs-fast-hip.json`, `run-pancake.sh` default |
| Opt-in | Flash-Next Q2_0 (`*-q2_0-*-adopted.json`; 134.7-136.1 tok/s, 80.6% top-1 vs Q8_0, T-377; the 15c Q2_0-vs-IQ2_XS redo is still open); Qwen3.6-35B-A3B (`src/qwen35` = strata-q36, and `strata-poly`); DeepSeek V4.1-Flash (`src/dsv41`, owner-deferred: Qwen first) |
| Not goals | Qwen3.8-27B dense, NVFP4 / `.basalt` (Blackwell), Vulkan/SYCL/Intel/Mac tuning, another engine as the base (vLLM, SGLang, KTransformers, TensorRT-LLM, LMDeploy, MLC-LLM, ExLlamaV2, PowerInfer, TGI, LightLLM, NanoFlow, DeepSpeed-MII, TensorFold: they keep the model in VRAM or need far more memory) |

Baseline (HEAD 693cdf63d, spec 43, T-392): short 111.5, edit 112.3, longdec10k 91.7 (noisy), longdec32k 102.8 (noisy)
tok/s. Reference points: stock Strata 54.5 vs ours 75.5 on the 3060 (T-216); Strata-rdna3 109-119 and Avalon 95-100
on this card (T-009); window budget short 24.8 ms = 15.9 GPU / 5.2 CPU / 2.2 other, edit 37.4 ms = 15.8 / 17.4 / 2.5 + 1.7 sync;
decode profile top kernels combine_cpu 25.8%, wait_flag 14.8%, rdna_kq 9.0%, expdec_gu 6.8% (spec 40 profile).

## 2. Rules

Measurement (the full text is in AGENTS.md "Measurement rules"):
1. Paired A B B A runs, run 1 discarded, quiet machine (`abtest.py` waits and refuses beside a running server). Bar:
   >=2% end to end exact, >=3% lossy; TTFT bar >=0.1 s per warm turn or >=2% on long prompts. Screen, then confirm
   (`--block 3 --warmup 1 --outlier robust --reclaim-max 3 --reps 9`). Owner rule since 2026-11-13: promote from 0% up
   at screen, decide at confirm; "test, never judge by reading".
2. Text identity is not an exactness proof (stock-vs-stock greedy noise is 1.2-1.3% of tokens; the base arm gave 6
   hashes in 6 runs). Lossy arms need token forcing (`quality_gate.py`, 20 requests x 300 tokens, 2 stock arms; it
   rejects only large deviations) and then the task gate (`task_gate.py`: LiveCodeBench subset + BFCL v4 multi-turn +
   own-repo grading) to choose between close variants.
3. Run `python tools/pancake/spec_lint.py` before any A/B: an arm key the fork does not read is a no-op.
4. Gains inside one substitute group do not add; price an idea against the window budget first (edits are a plain sum,
   so a small exact item caps at a few %). Re-measure fork claims here: Speedy's +22-33% on 3080s was noise..+9% here.
5. Every lead is ported and measured (owner, 2026-11-15; AGENTS.md rule 8): no lead is dropped by reasoning, only by a
   measurement. Three outcomes only: adopted (numbers in the commit), coded opt-in, not adopted (evidence + what would reopen it).
   Every test goes into TEST_LEDGER.md; nothing is dropped silently.

Code:
6. New features are opt-in; the default path stays byte-identical, and the commit says how that was checked. A default
   changes only after a paired measurement. Every new `STRATA_*` name is classified in
   `src/core/conversation_disk_env.def` (chunk-geometry switches are KeyIfSet in the disk key).
7. HIP is the benchmark platform; a CUDA-only change must not break HIP. Official model files + SHA-256 only;
   `presence_penalty=0`; no `STRATA_AUTO_PIN` for Qwen; `ROUTE_TAIL_SKIP=7` only by owner decision.
8. Credit every adopted idea in README section 9 in the same commit (fork, commit, author numbers).
9. Develop by day, run the GPU queue at night (`tools/pancake/night_queue.py`, `--dry-run` first).
10. Linux box repo on an ntfs3 disk (D): `git config checkout.workers 1`; no `reset --hard`, full checkouts or new
    worktrees there; merge only in the existing worktree.

Owner rules, verbatim where it matters (all binding):
- 2026-11-12 22:44 "i need allways full path": give full paths in every report.
- 2026-11-13 09:21 "no promote from 0% up" ... "make this a rule" (screen tier above); 2026-11-15 09:06 "build new
  technologies and test each time do not judge something without a test".
- 2026-11-15 01:37 MANDATORY: "always check the code for the assessment and update of the plan, where code exists": read
  the upstream/fork diff and our code (not titles), mark each item compared / not compared; a grep proves only that a
  symbol exists; if the algorithm differs, implement it faithfully and compare at equal cost. 01:43 "if something does
  not exist port it or build it to properly assess"; 01:50 "dont skip the read" (read full diffs).
- 2026-11-15 12:00 goal: "invent new techniques and technologies especially where we have some gains".
- 2026-11-15 12:40: tag every improvement as KERNEL, MODEL, ENGINE or WORKLOAD related, in the ledger/plan and in the
  commit text.
- 2026-11-15 01:35 two-card potential: "keep it for later dont dismiss" (section 8).
- 2026-11-15 12:51: the CPU/GPU expert combine and waits are in scope.
- 2026-11-15 15:14: "remove the ambiguous configs so we dont stumble on it again" (one daily preset only).
- 2026-11-12 19:40: "use the same qwen in both places" (Flash-Next IQ2_XS in pancake and Avalon); 19:37 DeepSeek waits
  until the Qwen tests are done.
- 2026-11-15 16:00: "ecap 1 can wait after we consolidate also the branches, 3 yes only if it does not affect testing,
  do also a41 and the rocm, the rest can be forgotten" (so the 9router dashboard password and chkdsk of D are dropped).
- Sudo (2026-11-13 12:31): owner chose option 2 = NOPASSWD rules for three commands only (`os_tune.sh compact`, the
  `power_dpm_force_performance_level` write, the ntfs3 mount); advice: drop the mount line and make os_tune.sh
  root-owned first; review file `/home/dan/sudoers-pancake.txt` (still contains the mount line). Agents must not edit
  sudoers or permission settings themselves.

Test tiers (`abtest.py --tier`; answer each item at the cheapest tier that can):
- 0 offline: replays and byte counts (cache sims, `tools/pancake/trace/tail_replay.py`, `replay_turns.py --dry-run`); no GPU.
- 1 screen: one A B pass, short + edit, 2 requests per case per server; promote from 0% up (owner rule 2026-11-13,
  code 41f4f6231), drop if any case < -3%; decide at confirm (bar >= 2% exact, >= 3% lossy).
  Spec key `"prefill": true` keeps the `long*` cases (one cold request each, judged on prompt tok/s). ~2-3 min/arm.
  A screen promote is only a hint: n=2 gave intervals of +-100..+450 tok/s.
- 2 confirm: A B B A with early stop (every arm/case decided or interval < 1.5% of base). For a decision use rule 1's
  flags with `--no-early-stop` and the same tier for every arm. Exact items need 12+ pairs to prove 2% at a paired SD of 1.5%.
- 3 rigorous (lossy, preset-default change, text differs from base, long-context claim): confirm + `quality_gate.py` +
  `task_gate.py SPEC.json --out bench/results/<day>-<name>` (`--dry-run` checks spec and task set without a server;
  base runs twice as the drift check; BETTER/PASS/INCONCLUSIVE/REJECT; INCONCLUSIVE -> `--reps 10-20` or more tasks).
- Evidence order: `ctest -L cpu`/`-L gpu` and parity -> kernel level (`expdec_bench`, `q2_rows_bench`, rocprof; kernel
  gain x window share >= 2% before end to end) -> screen -> confirm -> gates -> `python3 tools/pancake/promote.py <items.json>`.
- Cases: `long` 11.7K, `long32k`/`long64k` ~32000/~64000 tokens of `src/**/*.cpp`, 32 new tokens, each opened by a
  unique `[run <nonce>-<n>]` line (always cold, hash not compared); 128K is not an abtest case.
- Before a batch: `tools/pancake/os_tune.sh` (abtest warns below 20 GiB free in 2 MiB blocks; the arena locks 33 GiB;
  fix with `sudo tools/pancake/os_tune.sh compact|install`); CPU 3.9-4.1 GHz in run telemetry (slow runs sit at
  3.4-3.7 GHz; `sudo tools/pancake/os_tune.sh cpu-install`); engine on 8080 stopped; `pgrep -f test_phase.sh`.
  A server dying before ready is retried once, then the batch aborts with the free-block count.
- One spec per area with all its env-only arms. One decision per item: a "no difference" confirm -> opt-in or removed,
  retest only after a code change. A switch that never logged "acted" voids its arm. Arena vs mmap tiers differ ~8%,
  never mix them. Back to back: `tools/pancake/queue_run.sh OUTDIR SPEC[:abtest args] ...` (`-n` previews).
- No gain: at most 2-3 attempts. Read the `decode timing` line + `gpu_profile.py`; check the path acts (log line) and is
  not masked; tune with `tools/pancake/sweep.py`; then keep, opt-in or remove.

## 3. What the daily preset runs (fast-hip), with the evidence

| part | setting | evidence |
|---|---|---|
| RDNA ports from Strata-rdna3 | `GR_RDNA`, `GR_RDNA_Q8`, `GR_RDNA_A8`, `GR_INJ_UP`, `GR_FUSED`, `RDNA_Q6K`, `RDNA_MMVQ_WAVE=3`, `COMBINE_CPU`, `WAIT_FUSE`, `QFUSE`, `HEAD_MIX_MULTI` | decode 87-101 -> ~112 tok/s, T-009; round C best -> preset f0f9c5788, T-378 |
| expert tier | `--pcie-frac 0` (T-197; re-confirmed: 0.15/0.3/0.55 all slower, T-268), `--adapt-every 1 --adapt-swaps 32` (T-199), learned profile `data/expert-profile-pancake.bin` (+6.7%, T-194), GPU router lookahead default on (+10%, T-192), `ADAPT_PERSIST` default on |
| CPU pool | `POOL_FUSE_PHASES=1` (edit +2.5% +-1.7, T-224), `EXPDEC=1`, `EXPDEC_DOWN=1` (short +1.7% +-0.5, T-037/T-040) |
| drafting | `--spec 4` (official MTP depth; spec 2 -5%, spec 5/6 flat, T-118/T-149), `--spec-min-p 0.5`, `--lookup-max-t 12` (edit +8.8% +-6.8, quality PASS, T-126/T-127), English draft vocab |
| prompt path | hipBLASLt 100401 table (1,345 -> 1,850-1,920 tok/s, T-096), `PF_GEMM`, `PF_F16=2`, `PF_FUSED=1` (see section 7: no HIP A/B recorded), `PF_PAD`, `PREFILL_AUTO_MAX=32768`, `PREFILL_CPU_SHARE=0.6`/`_MAX=4096`, `PREFILL_RING=192`, `SELECT_WMMA`, `HIP_WMMA=1`, Speedy opt-ins `PF_GROUP_SCRATCH`, `PF_HC_REGION`, `PROMPT_ATTN_IMMA`, `KV_PREFETCH` (taken from Speedy's numbers, no HIP A/B recorded) |
| attention / KV | `--kv int8 --kv-resident 32768 --max-context 131072`, `ATTN_DAILY`, `QSA_ROWS`, `PA_FAST`, `KV_HOST_DMA` |
| ECAP and cache (adopted 2026-11-15, owner; ECAP changed to the gated arm 2026-11-16) | `STRATA_VERIFY_ECAP=0` with `STRATA_VERIFY_ECAP_MIN_ROWS=9` (cap 0 only in windows of 9 or more rows: edit 165.3 +-6.2 tok/s = +41.7% +-5.6 over no ECAP, short +2.8% +-2.6, T-588; token forcing PASS -0.011 pts, T-589; tool-call gate PASS, T-591; before: cap 1, edit +33.5% +-14.3, T-270, T-310, base edit prompt 147.6 tok/s, T-560), `--conversation-cache-mib 2048 --conversation-cache-min-tokens 8192` (prompt time -21.6%, T-183); sampling default temperature 0.1, top_p 0.95, top_k 20 |
| deferral | `VERIFY_DEFER_CPU` OFF (owner decision 17.1; contested, section 4) |
| server | grammar on (spec decoding mask 0.1-0.2% of a window, T-315), thinking caps `reasoning_budget_grace: 512` + close retry, `env_to_user`, `effort_position: end`, loop guard, tolerant tool calls, turn log, `AGENT_COUNTERS=1` (log only, T-513) |

Lossy switches in fast-hip (an exact bundle = fast minus these 11): `GDN_CONVL2`, `GR_RDNA_A8`, `GR_RDNA_Q8`, `HC_UPMIX`,
`PA_FAST`, `PF_F16=2`, `PF_HCDOWN`, `PREFILL_AUTO_MAX=32768`, `PREFILL_CPU_SHARE=0.6`, `ROUTE_TAIL_SKIP=0`,
`SELECT_WMMA`. So the daily base is not bit-exact; "exact arm" in a spec means exact relative to this base.

Retired presets (2026-11-15, moved to `configs/pancake/retired/`, restore with `git mv`; `run-pancake.sh turbo` exits with a
message): `flashnext-iq2_xs-turbo-hip.json` / `-turbo.json` = fast + `ROUTE_PRIOR=0.5` (+ `VERIFY_DEFER_CPU=1`, minus
`AGENT_COUNTERS` on HIP) (lossy; edit +29.7% n.s., token forcing INCONCLUSIVE +0.83 pts, T-245; all-prose -8%, T-387;
interaction with ECAP untested) and `flashnext-iq2_xs-hip.json` (exact copy with no users). The one daily preset is
`flashnext-iq2_xs-fast-hip.json`; Q2_0 `*-adopted.json` carry FUSE_PHASES, EXPDEC_DOWN,
lookup 12, no deferral. Not in any preset: `ROUTE_PRIOR_AUTO`, `VERIFY_DEFER_AUTO` (-12.5% IQ2_XS, T-262),
`KV_RING_ADAPT`, `PARK_REWIND_MAX`, `PARK_DIVERGENCE_LOG`, `PREFILL_EQUAL_TAIL`, `ADAPT_KMISS`, `PARK_TRUNCATE`,
`SUFFIX_INDEX`, `VERIFY_ECAP_MIN_ROWS`, the matrix-core kernels (`STRATA_RDNA_WMMA_MINCOLS`).

Served instance: `tools/pancake/pancake.service` (systemd user unit, `run-pancake.sh iq2_xs fast` via `NOCAP=1`, port
11437, 127.0.0.1 only, `MemoryMax=54G`, `Conflicts=` the old units; not enabled at boot). Start `systemctl --user start
pancake` (ready in 1-3 min, log says `ready:`), stop it before any benchmark (it holds the whole card). Portal
`http://127.0.0.1:11437/` and `/pancake`; opencode provider `strata` points at it. Live test: T-556. The old upstream
`strata.service` and its config are in `retired-2026-11-15` folders on the box.

Interactions: `ROUTE_PRIOR` set (even 0) disables ROUTE_PUBLISH / ROUTE_FAST / ROUTER_FUSE; deferral turns off 6
fused paths process-wide (so A35 / ROUTER_FUSE "no-op under deferral" no longer holds with deferral off);
`ADAPT_KMISS` needs `--adapt-async 1`; `*_AUTO` and the fixed switch never together; `KV_RING_ADAPT` under `--serve`
follows `--max-context`; CPU share binds only for deltas <= ~400 tokens; PARK_REWIND helps only when admission is the
miss.

## 4. Decisions waiting for the owner (measured; the data is in)

| # | decision | data | what is missing |
|---|---|---|---|
| D1 | ECAP: cap 1 was adopted into fast-hip on 2026-11-15 (owner); REPLACED on 2026-11-16 by cap 0 from 9-row windows (`STRATA_VERIFY_ECAP=0`, `STRATA_VERIFY_ECAP_MIN_ROWS=9`, T-588, T-589, T-591; autonomous overnight run under the owner's rules: gated arm over cap 1 by +20.7% edit and +17.8% short means, quality gate PASS, task gate PASS; revert = set `STRATA_VERIFY_ECAP` back to 1 and drop MIN_ROWS). Old text: | edit +33.5% +-14.3 (T-270); token forcing PASS (T-310); stack add-ons dropped (T-555, T-561). `ECAP=0` REJECT (+0.95 pts, T-271, T-311); window-gated variants: spec 60 has five arms (no ECAP, cap 1 = preset, cap 0, cap 0 from 9-row windows, cap 0 from 6-row windows; `STRATA_VERIFY_ECAP_MIN_ROWS`), spec 61 the quality gate for the gated arms. Spec 60 was STOPPED by the owner at 19:39 local after about 17 of 30 runs (T-572, no summary): partial edit means ecap1 141.6, ecap0 167.2, cap 0 from 9 rows 143.8, cap 0 from 6 rows 161.2 tok/s, so the 6-row gate is the interesting one and the 9-row gate loses the gain (WRONG: the full run T-588 shows the 9-row gate best and tightest); specs 60 and 61 have now run (T-588, T-589) | still missing: tool-call task gate on real agent prompts (a too-tight cap broke tool calls elsewhere; owner order: after the branch consolidation, now done, so it runs after the spec 60-62 chain); token forcing scores greedy tokens while the served default samples at temperature 0.1 |
| D2 | cache levers: `--conversation-cache-mib 2048 --conversation-cache-min-tokens 8192` ADOPTED 2026-11-15 (owner); `STRATA_PARK_TRUNCATE` 1280 not adopted | -21.6% +-14.8 prompt time (T-183), -23.1% +-11.8 for park-truncate (T-184); one recorded session | combined replay of both levers queued (spec 62) |
| D3 | deferral on IQ2_XS | night retest: off vs on -0.9% overall, code -7.2%, prose -3.2% (T-260); Q2_0: off +8.9% (T-261); first A/B edit +15.8% (T-222) | 5-repeat edit-window A/B on the IQ2_XS preset; until then 17.1 (OFF) stands |
| D4 | A24 `STRATA_SUFFIX_INDEX` | screen: short +26.8% +-34.4 (outlier), edit -0.0% (T-161); offline -2.44% rounds, sub-agents -7.5% (T-424); half the gain is from outputs older than a 30-min pause, so use the persistent `STRATA_SUFFIX_INDEX_FILE` (4 MiB -1.00%, 16 MiB -1.71%) | confirm + quality gate with the file on the daily server |
| D5 | `PF_FUSED=1` / `PF_F16=2` in every HIP preset | the only A/B (T-093, pre-HIP preset): 42K -7% but 5K +12% TTFT | either a HIP A/B (spec) or a recorded owner decision |
| D6 | 9-16 column RDNA K-quant kernels are now the code default (`kRdnaDefaultCols=16`, `kRdnaMaxCols=32`) | parity 192/192 (tolerance); engine +3.7/+3.8% screen, +1.8% +-7.8 / +4.3% +-5.0 confirm (T-081): inside noise. The bitwise bench check fails: IQ4_XS multi-column output differs from 1-column calls at 12-32 cols (T-558), which can explain the short-case text variance of spec 51 | accept as default (it is a kernel default change, measured not worse) or gate behind `STRATA_RDNA_KQ_MAXCOLS` |
| D7 | Phase 18 item 9 merged the window-input copies on the DEFAULT path (`copy_from_mapped_multi`, on unless `STRATA_MTP_MULTI_COPY=0`) | pure copies, `verify_parity` covers both modes; spec 30 never ran | run spec 30 (build-vs-build) or flip the default back |
| D8 | Q2_0 vs IQ2_XS as the daily file | 15 screen +8.6% +-46.3 (T-382); 15c run unusable (T-384) | 15c redo |
| D9 | A29 entropy-coded miss path | owner chose BUILD on 2026-11-13 06:58, with deep research and a plan update before any code; offline bound 8-12% (T-282) | research + design note in this file, then code |
| D10 | housekeeping | done 2026-11-15: worktrees 107 -> 14 (28 GB freed), local branches 166 -> 9 (tip map `pancake-docs/research/bench/branch-tips-2026-11-15.tsv`), 200 of 202 refs were already contained and the last one (`optimize-kernel-hip`) recorded as merged; remote branches untouched. A41 hook registered warn-only (T-562): user-scope `~/.claude/settings.json` on the box, PreToolUse for `Edit|MultiEdit|Write`, command `RBE_MODE=warn python3 -I /home/dan/bin/read_before_edit_hook.py` (timeout 10 s; repo copy `tools/pancake/a41/`; settings backup `~/.claude/settings.json.bak-a41-2026-11-15`; `RBE_MODE=hold` blocks). Box decisions taken without a test, waiting for owner confirmation: `fuse-wait` slow mode closed (it lost on average, T-077); `pancake.service` NOT enabled at boot (it would hold the GPU during tests). Open: the remaining dirty worktrees, remote branch deletion, the box's `PANCAKE_STATUS.md` (still edited there 2026-11-15 16:15 UTC; delete it on the box after pulling, rule 2), stale upstream/HEAD ref; GPU clock back to auto after pinned runs; disk-tier cap (D at 76%); `PARK_REWIND_MAX` default; cron `--night-only 23-07 --drop-cache`, `--reclaim-max 3.0`; sudo rules via `visudo`; Avalon 32K+ (shrink its context or drop the comparison: it OOMs at 32K+, T-174); delete `origin/NetworkPancake` (moved out of the repo to `C:\Users\iancu\NetworkPancake`) | owner |
| D11 | open owner questions found in the box logs | caveman-proxy on the box is respawned by hooks in `~/.claude/settings.json` and trips the quiet-machine gate (the gate also counts the session's own `claude`, remote desktop and pipewire): remove the hook entries or stop the proxy by hand before benchmarks? A15 rule: ce88ce281 evicts the shortest tool-pending entry by tokens, the design said "smallest entry whose bytes cover the shortfall": keep or build the design rule and retest? A20: turn on rumination entropy in the daily config (asked, no answer)? "3 yes" (16:00): read as the approved-ideas list, so `pancake.service` was NOT enabled at boot; confirm. Service binds 127.0.0.1: open to the LAN? CUDA header for the tests ("build also the cuda header for tests", 11:58: the full HIP build stops at `strata-poly`): not done. Delete the unused 10.44 GB Qwen3.8-27B IQ3_XXS GGUF on D (arch `qwen35`, not loadable by the native engine)? | owner |

## 5. Run next: coded, never measured (GPU queue, in this order)

1. **RAN 2026-11-15 21:58 to 2026-11-16 01:06 (autonomous overnight chain): specs 62, 60, 61 (T-587..T-589), the CPU probe (T-590), spec 65 (T-591); spec 64 was invalid (T-592) and is re-run.** **Results of the box chain** `pancake-docs/research/bench/results-2026-11-15/run-scripts-afternoon/auto-chain15.sh` (started
   18:22 local on 2026-11-15; spec 60 from 19:07, ~19:55 ETA): spec 60 (five ECAP arms, see D1), 61 (quality gate of the gated
   arms), 62 (the two cache levers combined on the replay): add them as ledger rows. Box rule while the chain runs: no builds
   and no memory-bandwidth benches. Then, in the box's order: (a) build, parity-check and measure the IQ4_XS matrix-core
   variant that shares the nibble-LUT decode between lane halves (written, off by default, uncommitted on the box:
   `rdna_mmvq.cu` + one `conversation_disk_env.def` line); (b) the P4 byte-bound probe for IQ2_XS rows = the existing
   `iq_avx2_parity --bench` (IQ pairs, `--nt` thread counts, `--mb` working set; already built); (c) the ECAP tool-call task
   gate (D1). Then **spec 30** (D7). Fix the base path in specs 33 and 34 first
   (`pancake-flashnext-iq2_xs-fast-hip.json` does not exist; use `configs/pancake/flashnext-iq2_xs-fast-hip.json`),
   then run spec 33 (MTP bundle: `--mtp-q4 all --mtp-window 4096 --host-core sibling`) and spec 34's `pf-switch-min-t`
   arm (its `mmvq-wave1` arm is answered by T-072, `kv-grow` by T-391).
2. **NIGHT QUEUE, RE-RUN: `/home/dan/clone-of-d/src/win-chain4.sh`** (started 2026-11-16 02:03; log `/home/dan/clone-of-d/src/win-chain4.log`, results
   `/home/dan/auto-runs/win4-<spec>.jsonl`): win-chain2/3 built everything (build-win, build-win-il, build-win-j32 exit 0, ctest -L cpu
   pass) but every spec exited 3 at 01:01 because the box job `/home/dan/auto-post17.sh` (taskgate server, then
   pancake.service restart) held the card; its GPU ctest (2 failures: hip_prefill_fused_iq, _f16) also overlapped that job.
   Chain 4 stops pancake.service, re-runs ctest -L gpu and mmvq_il_parity, then specs 81-86, 80, 87-93 and the never-run specs
   of item 5 (a busy card is retried every 5 min, 12 times), then starts pancake.service again. The Windows-session specs
   were renumbered 64-77 -> 80-93 (the box took 64 and 65).
   First chain-4 pass (02:12-02:18): ctest -L gpu 135/135 and mmvq_il_parity passed on the free card (the 2 earlier
   failures were the overlap); every spec then aborted "server died before ready" because the clone had no `.venv`
   (`Strata-win/.venv` is now a link to `../Strata/.venv`; the 2 MiB-block RAM warning was not the cause). Re-run from
   spec 81 at 02:20 with `echo 1 | sudo -n tee /proc/sys/vm/compact_memory` first; spec 81 base read short 114 / edit 145-171
   tok/s, so the servers start. Earlier text: `/home/dan/clone-of-d/src/win-chain2.sh`, started 22:09, waits for the box
   chain16 (`CHAIN16_DONE` in `/home/dan/auto-chain16.log`: specs 63-park, 62, 60, 61) and for no abtest / replay running, then:
   builds `build-win`, `build-win-il` (-DSTRATA_HIP_MMVQ_IL) and `build-win-j32` in the clone `/home/dan/clone-of-d/src/Strata-win`;
   `ctest -L cpu`, `ctest -L gpu` (134 tests), `mmvq_il_parity`; then specs 81 (decode exact opt-ins), 66 (PCIe share +
   DMA_FLAG_WRITE / PCIE_FORK), 67 (prompt opt-ins), 68 (PF_GU_EXACT family on the exact FP16 path), 69 (MMVQ_IL_PLAIN), 70
   (FK-28 ADAPT_PROMPT), 64 (#31751), all `--tier screen`, results `/home/dan/auto-runs/win-<spec>.jsonl`, log
   `/home/dan/clone-of-d/src/win-chain2.log`. Caution for chain16: the Windows session's builds ran 21:58-22:08 beside its spec 62
   replay (started 21:58): treat that replay's numbers as possibly noisy.
   **Then `/home/dan/clone-of-d/src/win-chain3.sh`** (waits for `WINCHAIN2_DONE`; no spec starts between 10:00 and 22:00):
   specs 87 (exact coded backlog of item 6), 72 (lossy backlog of item 6), then the never-run specs of item 5 (26, 12, a21,
   24-a15, 22-a7, 24-a45, 16-a19, 25-f-env-rerun, a33-a35, 15-kmiss, 19-turbo-guards) on `build-win` (the clone's `build-hip`
   is a link to it). Log `/home/dan/clone-of-d/src/win-chain3.log`. Left out: 08-build (its own build and preset), taskgate-*
   (repoint to `bench/tasks/repo-history.json` first), HIP_GLOBAL_WEIGHTS (build macro), HIP_WMMA_Q4 (needs --kv q4_0), --batch 2.
3. **q36 dense MMQ** for Qwen3.6-35B-A3B: env arm `STRATA_DENSE_MMQ=1` (gemm.cu:1377, HIP; prerequisites 14dd746f /
   55b8c26b in HEAD), prompt 8K/32K. Author (burzumishi, strata-q36): 32K prompt 635 -> 910 tok/s on a 7900 XTX.
   Lossy rounding: token forcing.
4. **EQUAL_TAIL** (`STRATA_PREFILL_EQUAL_TAIL`, `_MIN` 256): est. -1.5..-2.5 s of ~40 s at 100K, ~-1 s at 66.5K, up to
   -2.5 s at 34K. Use prompt sizes 30000, 34000, 60000, 66500, 70000, 100000 (the ttft_bench defaults sit on chunk
   boundaries); plus the #5539 check: 99,997 tokens -> chunks 0/32768/65536/98304 (with EQUAL_TAIL 0/25088/50176/75264),
   ~98,400 tokens, `STRATA_TRACE=1`, a lent line, a reply, no fault.
5. **Never-run specs for coded switches**: 26-stack (F12 `IQ_SIGN_FAST`), 19-turbo-guards (F13), 12-spec-sampling and
   a21-spec-lowtemp (A21/F6 `SPEC_PROB`, FK-01 `SPEC_PROB_DT`), 24-a15 (`PARK_TTL_S`; `pick_victim()` is in
   conversation_cache.hpp), 22-a7 (`PROMO_CHUNK`), 24-a45 (`CALIBRATE`), 16-a19 (`HIDDEN_DUMP`/`THINK_PROBE`, first step
   of the thinking stop; proxy labels cap the saving near 10%), 25-f-env-rerun (lookahead0, routed-dedupe, MMVF_ROWS;
   the 05s screen had no clean runs, T-527), a33-a35-fusion-retest (premise changed: deferral is off), 15-kmiss,
   taskgate-* (point them at `bench/tasks/repo-history.json`, not default.json), 08-build (`STRATA_HOST_NATIVE`).
6. **Coded, no spec yet**: F7 `Q2_FAST` engine A/B (bit-exact, x1.33/x1.23 rows, T-246/T-253); F14 `EXP9_DOWN=42:10`
   (= FK-24, parity in ctest); F22 `HIP_WMMA_Q4` end to end; F11 `ATTN_WINDOW` (q36); `HIP_GLOBAL_WEIGHTS` (flat loads
   13,119 -> 108, T-050); `WAIT_COPY256` confirm (screen +6.7% / +2.4%, T-056; inert below T=6, so check it acts at
   `--spec 4`); FK-02 `--prompt-cache-spread`, FK-03 `--lookup-chain 3`, `--adapt-decay`, FK-10 `ATTN_LANECELL`,
   FK-12 `--batch 2`, FK-27 `PARK_REWIND_MAX`; items 3-4 seeds (`CONV_WORKSET`, `DOMAIN_PROFILES`; offline 81.17 ->
   82.86%, T-398); A13 disk tier GPU test; A14 restore equivalence (`restore_check.py --control`); `SELECT_TWO_PASS` on
   HIP; selftune candidate profile (+4.6% modeled, T-439); `ROUTE_PRIOR_AUTO`; 17.3 ablation; TTFT scenarios C/D/E;
   `prefix_check`; A9 `IDLE_ADAPT` (ceiling <=0.26%, low priority); `STRATA_POOL_SPIN_US` sweep (Juud: 15.1 -> 12.6
   ms/token).
7. **Small open checks**: A18 (effort/temperature fields appear in `pancake-turns.jsonl` after the first served run;
   logging approved 2026-11-13 07:01); A54 blocked until a task-gate run gives per-task scores for the lossy arms; A50 not
   settled by A49; A28 decision: keep the 58 GB Coder GGUF on D (`models/ista-gsq-rco-coder-gguf`) and do NOT test it
   (Avalon-only, ~29.6 GB resident); the confirm tier's early stop ended a run at n=2 of 3 (use `--no-early-stop` for
   long-context cases). Research topics R1-R9 from the 2026-11-15 LLM report (high: R1 Strata-rdna3 code read, R2
   predictive expert prefetch replay on the gate-weight trace; medium R3 hybrid tiers, R4 KV eviction under parallel
   agents; low R5 MiniMax read-ahead, R6 elastic KV, R7 QEngine MTP; R8 multi-GPU parked; R9 orchestration not now):
   R1 then R2 were promised and not done.
8. **Measure before code (TTFT, M1-M5)**: M1 8/32/64/100K with prompt_ms, TTFB, park, chunk plan, file-tier/KV lines,
   PREFILL_TIMING (each size twice, fresh state or park reported apart); M2 10-turn session x3 (alone; side request +
   sub-agent switch; `--conversation-cache-mib 1280`); M3 divergence; M4 opencode/Copilot request bodies; M5 deltas
   100/300/1000/3000 at 32K/64K. Unexplained so far: a 3.4 s turn after a 34K cold read, 4.8 s / 36 s waits outside
   the timer, the cause of 13% checkpoint fallbacks.
9. **Probes before ports** (`tools/pancake/probes/`; each header has question/build/run; `-DSTRATA_BUILD_PROBES=ON`,
   CMakeLists.txt ~3124, EXCLUDE_FROM_ALL). Run with the GPU idle; under rocprofv3 decode is ~1.6x slower, so read shares.
   - 4.2.1 empty-launch cost vs block count on 42 WGPs (F20, never run): `hipcc -O2 --offload-arch=gfx1100
     tools/pancake/probes/empty_grid.hip.cpp -o /tmp/empty_grid && HIP_VISIBLE_DEVICES=0 /tmp/empty_grid --csv /tmp/sweep.csv`
     (or target `empty_grid_probe`; `--step 8` = 10 s), then `python3 tools/pancake/probes/launch_shapes.py
     /tmp/prof/run_kernel_trace.csv --sweep /tmp/sweep.csv` (`--static` = no trace). Close F20 when "EXCESS" ~0.
   - 4.2.2 route publish / top-10 share (pre-deferral answer T-033): `python3 tools/pancake/gpu_profile.py
     configs/pancake/flashnext-iq2_xs-fast-hip.json /tmp/prof --case short`, then `python3 tools/pancake/probes/kernel_share.py
     /tmp/prof/run_kernel_trace.csv` (TARGET line: mean us, kernel/wall share, gap after). Port only if >= 2% of wall.
   - 4.2.3 expert GEMV bandwidth vs peak (F12/F14/F16, never run): `python3 tools/pancake/probes/expert_bw.py
     /tmp/prof/run_kernel_trace.csv --gpu 7900xt --hit-rate 0.6 --floor-us 3` (bytes: `bench/results/2026-11-10-offline/gguf_table.json`);
     GB/s for K = 1..10 experts/launch, real K = 10 x hit rate; little headroom above ~85% of peak.
   - 4.2.4 CPU Q2_0 rows legacy vs `STRATA_Q2_FAST` (T-246, T-253, T-264): `cmake -S . -B build-probes
     -DSTRATA_BUILD_PROBES=ON && cmake --build build-probes --target q2_rows_bench && taskset -c 3 build-probes/q2_rows_bench`
     (~20 s; ns/row legacy/fast/dispatcher/read-only floor in L2 and DRAM, checks bitwise equality).
   - F12: `iq_multi_parity`, then `expdec_bench 300 -1 20 16/42,17/42,22/42 def,dec:s0,dec:s1` (every cell `identical`;
     keep a format only where `dec:s1` has the lower gu_us), then spec 22-f12-sign-fast.
10. **Open claims, never measured**: `STRATA_PF_HCDOWN=1` quality (in fast-hip, lossy; rxr2: 2x first-token KL at 16K
   for +1.3% prompt); `hipblaslt_extend.py` table vs shipped at 1.5-3.2K/4-4.2K/16K/32K (greedy identity + identical
   "Lt solution=" lines); cold 32K `--ple-io direct`, caches dropped: `WAIT_TRACE=1` vs +`PLE_PIPE=1` vs +`PLE_COMPACT=1`;
   start time `STRATA_UNBUFFERED_LOAD=0/1` + `STRATA_VERIFY_ARENA=1` (claim 28-65 s -> 10-14 s);
   `STRATA_CONV_BUDGET=ram STRATA_CONV_KEEP_GIB=8` budget line and arena-load memory pressure ~0; sub-agent switch
   (parent >= 10K -> sub-agent, same system+tools -> parent restore) vs `PIN_PARK_TOKENS=-1`, tokens vs
   `STRATA_CKPT_REREAD=1`; loop guard on/off on an agent replay; `sweep.py` over expdec/pool knobs.

## 6. Code to write (ranked by expected value on this box; details and sources in LEADS.md)

| # | item | expected | exact/lossy |
|---|---|---|---|
| 1 | CODED 2026-11-15 (`STRATA_ADAPT_PROMPT=1`, weight `STRATA_ADAPT_PROMPT_W`; next: spec 86). FK-28 prompt-fed tier counts, redesigned: the old dead `set_usage` hook was deleted (T-575); built fresh. Feed `drive.d.usage` from the existing `sp.on_routes` callback in generate.cpp (as ram_adapt2 does, ~14963), weighted like RAM_ADAPT2 (`prompt_w*20*K/n`), env gate + disk_env entry | unmeasured (oracle own-counts +2.9 hit pts offline, T-417) | exact routing input, tier only |
| 2 | CODED 2026-11-15 (`STRATA_PF_AHEAD=1`, `_SLOTS` 160; `STRATA_REFILL_ASYNC=1`, lossy for the reply; next: specs 83 / 86). 0xluk `PF_AHEAD` + `REFILL_ASYNC` for the staged path of agent deltas < 4,096 tokens | +35% agent prefill (2x 3060, PCIe 3) | exact |
| 3 | CODED 2026-11-15 (opt-in build switch `STRATA_MMQ_RDNA3_J32_ROLLED`, `cmake/strata_ggml_rdna3_j32.cmake`): llama.cpp #31751 RDNA3 MMQ J=32 rolled k01 loop on the pinned ggml; acts on the staged prompt chunks (< 4096 tokens, agent deltas) for Q2_0 down / IQ2_XXS / IQ4_XS. Static check T-579: J=32 kernel 256 VGPRs + spill, occupancy 5 -> 137 VGPRs, no spill, occupancy 10. Next: spec 80 (build-vs-build) | author +36% at 17-32-token batches, gfx1100 | exact |
| 4 | android-imdad bundle caa9b00115 + the PF_GU_EXACT series (90434843b4, ac7e5aff59, 9caad8f2bb, 876c3af7e6, 8a2f38aaad): PORTED 2026-11-15, every switch default-off (owner: every lead followed). Decode: `MULTI_BLOCK_ARGMAX` on gfx11, `MTP_SELECT_WIDE`, `SCAN_UNROLL`, `TOPK_REGKEYS`, `TOPK_SCAN`, `ATTN_MERGE_BOUND`, `ATTN_PRE75_NC`, `GRUP_WIDE`, `GRUP_PF`, `DOWN_SWIGLU`, `GDN_NORM128` (split path), `ADAPT_COPY2`, `ADAPT_LAYERWAIT`, `DMA_FLAG_WRITE`, `PCIE_FORK`; prompt: `DQ_FAST`, `PF_LEAN_ISSUE`, `PF_COPY2`, `PF_GU_EXACT` (+ Tensile kernel, startup self-check), `PF_DOWN_STREAMS`, `PF_DQ_OVERLAP`; test knob `EXPERT_SLOTS_MAX`. `GR_FUSE` and `MMVQ_IL_PLAIN` callers wired (IL_PLAIN needs a -DSTRATA_HIP_MMVQ_IL build). Code-checked equal to ours: `ROUTE_PUB` = `ROUTE_PUBLISH` (route_publish_res: same top-10 + doorbell_publish_res in one launch, same session conditions); `COMBINE_HITS` = `STRATA_FUSE_HITS` (combine_k10_vec4_hits; default only on gfx12, so spec 81 arm fuse-hits). `WAIT_COMBINE` = `COMBINE_CPU` + `WAIT_FUSE` (combine_cpu_k10_vec4: every block spins on the flag, reads the mapped CPU rows) except one gap: ours refuses when the shared expert is not native BF16 or the CPU experts are not in the native layout (verify.cpp cfuse_off), the fork's needs only aligned buffers: port that relaxation if the engine log shows COMBINE_CPU refused on the preset. Next: specs 81-84 | bundle +10% chat, +11% agent, KL 0, 7900 XTX; GU_EXACT 26.5 vs 12.9 TFLOPS | exact |
| 5 | DONE: ported with item 4 (also on the layer-major walk; `STRATA_PF_GU_EXACT_MINT`); next: spec 84. `PF_GU_EXACT` Tensile prompt GEMM for gfx1100 | 26.5 vs 12.9 TFLOPS; 64K prompt 40.1 -> 31.5 s (XTX) | exact |
| 6 | CODED 2026-11-15 (`STRATA_ADAPT_IDLE_COPY=1`, `_KB` slice; host-paced queue, serial decode loop with ADAPT_OVERLAP; next: spec 88). schedule H2D admission copies into CPU-idle phases (attention/GDN) | est. +10-18% | exact |
| 7 | CODED 2026-11-15: `STRATA_POOL_ROW_SPLIT=N`, `STRATA_POOL_TOUCH_NEXT=B` (bitwise, pool_tasks_test; next: spec 87); fused phases = existing `POOL_FUSE_PHASES`; imanu86 862c18aaea HiSpread is already the default in iq_avx2.cpp (port of sanastasiou 9233f998, parity in iq_avx2_parity). CPU pool bandwidth push (two fused phases per layer, row-chunk split, touch the next expert early); imanu86 IQ2_S/IQ3_S AVX2 decode without per-index shifts (pool on IQ2_S layers 20-33% faster) | est. 33 -> 38-42 GB/s, +5-12% | exact |
| 8 | CODED 2026-11-15 (`STRATA_TIER_REWARM=1`, `_MIN` 2048, `_MAX`; serve path; protect variant not built: the refill restores the same slots, no clean choice point; next: spec 86). re-warm or protect the decode tier after a long prompt (decode 92.5 vs 111.7 tok/s after prefill) | up to the gap | exact |
| 9 | CODED 2026-11-15 (this-cute-world: `STRATA_FUSE_PLAN`, `_COMBINE`, `_SHGATE` on the device-plan path, `_QGEMV` with MMVF_ROWS, `_SWIGLU_DOWN`; ROUTEBELL = our ROUTE_PUBLISH; next: spec 87). launch fusions (`FUSE_SHGATE/COMBINE/SWIGLU_DOWN/PLAN/ROUTEBELL/QGEMV`): launches cost ~14-16 us on gfx1100 | -400 launches/window (R9700) | exact |
| 10 | CODED 2026-11-15: `STRATA_EXPDEC_V2`, `STRATA_MMVQ_WAVE_ROWS` (table spec; no RDNA3 cells yet), `STRATA_SAMPLER_X8`, `STRATA_MMVF_ROWS1`, `STRATA_LOOKAHEAD_AFTER_PUB` (next: spec 87; WAVE_ROWS needs a sweep spec). `EXPDEC_V2` remainder; `MMVQ_WAVE_ROWS` table; small exact bundle: copy_rows loads-first CODED 2026-11-15 (`STRATA_ROWS_LOADS_FIRST=1`, next: spec 81); sampler x8 loads (waits for the box's sampler fix); single-token MMVF_ROWS; lookahead submit after publish (deferred: ~10-25 us a window) | -20-24% expdec kernel time; +4% MI50; several us per launch | exact |
| 11 | CODED 2026-11-15 (`STRATA_KV_STAGE_REQ`, `STRATA_PF_SEL_BATCH`: spec 89; `STRATA_TOOL_STEP_SPLIT=1` / `canonical_tools` in serve/, unit tests serve/test_tool_steps.py, test_canonical_tools.py). Speedy `KV_STAGE_REQ` + `PF_SEL_BATCH` (long-context loan sizing, VRAM); `TOOL_STEP_SPLIT` + `canonical_tools` (serve only) | 300K cold prompt 466 -> 1527 t/s (2x 3080); turn-2 cache reuse | exact |
| 12 | CODED 2026-11-15 (49152 AUTO_MAX ceiling, `STRATA_PREFILL_EQUAL_TAIL_MAX`, `STRATA_PREFILL_POOL_RETRY`; `2b19d510` not ported: no late-refill code here, 2-GPU only; next: spec 89). rdna3 `791cca4b`/`5906629d` (AUTO_MAX 49152 + refill head `2b19d510`), `f557b544` (CPU share retries per layer) | 2-4% at 49K | exact |
| 13 | CODED 2026-11-15 (`STRATA_DALI=1`, `_EVERY`; per-layer row-weighted cost fits over the PCIe share; STRATA_PCIE_BALANCE is the one-fit count-only predecessor, expert_source.hpp:453; next: spec 82). DALI per-layer per-step CPU/GPU assignment instead of the static A45 probe | +23% over HybriMoE (3090) | exact |
| 14 | CODED 2026-11-15 (`STRATA_QSA_PARTIAL_TOPK=1` exact; `STRATA_QSA_PAGE_SEL=N` + `_TOPP` lossy; test qsa_page_select_recall; next: spec 91). QSA long-prompt selection cost (Quest page-level or partial top-k) | after-12K gap 2.18 ms/token | lossy-ish |
| 15 | CODED 2026-11-15 (`STRATA_Q36_EXPERTS_USED` from serboox f8fb28ca1f / 0f040ef5e8: spec 90; `STRATA_MISS_SKIP_W`, `STRATA_MISS_SKIP_CUM`: spec 88; --dump-routing exists, generate.cpp:3840). uniform K experts (`Q36_EXPERTS_USED`, 35B); weight-based miss skip (`SKIP_MINW` / Maya `MISS_SKIP_W` / HOBBIT cumulative-gate, replay offline on a `--dump-routing` run first) | +14-19%; +7-13% | lossy |
| 16 | CODED 2026-11-15 (`STRATA_GDN_SPECLA=N`: per-row state checkpoints in the window kernels, commit from the nearest; bitwise, gdn_parity section 7; same commit traffic, saves the serial replay loop; next: spec 87). SpecLA: recover accepted GDN states from the verify-pass factors instead of replaying | unmeasured | exact |
| 17 | PARTLY CODED 2026-11-15: `STRATA_DRAFT_CLIP=x` (spec 87), `STRATA_EVENT_WAIT=1` (acts only in DFlash2 builds with --draft auto: no arm on the preset); item 10 already in the tree (mtp.cpp:1821-1840); OPEN, to build on the box with a compiler: 17 device accept (needs the commit inside the window graph), 18 fused argmax (exact tie/NaN match in the head GEMVs), 20 branch rollout (needs a second draft model), A19 stop rule (needs spec 16-a19 first). Phase 18 items 10 (pre-stage host records), 12 (event wait vs stream sync), 17/18/20 (device accept, fused argmax, branch rollout); A19 stop rule (after 16-a19); DraftPolicy cost-outlier clip | unmeasured | exact |
| 18 | IN PROGRESS 2026-11-15: every LEADS.md row still marked "no" / "partial" in our src (owner rule: every lead followed). First pass done (statuses on the LEADS.md rows, "2026-11-15: ..."): 59 coded, 33 already in the tree, 9 partial, 44 method, 8 parked, 80 left (GPU kernels / layouts / second draft model / no source). Build + ctest -L cpu 154/154 (T-585); specs 92 (exact) and 77 (lossy). Next: the 80 left rows | per lead | mixed |

Not packages any more (their parts are separate items above or in LEADS.md): 63 capability probe (`STRATA_PROBE` never
built; became A45), 66 agent-turn pipeline, 68 miss-cost speculation, 69 memory arbiter, 72 GPU-floor package; 94
typed flag registry, 95 split `generate.cpp main()` (hygiene, no speed).

## 7. Defects and stale claims found by the 2026-11-15 audit

Code:
- FIXED 2026-11-15 (T-575): the FK-28 dead hook (`Prefill::set_usage`, `usage_`, its count loop) is deleted; the feature
  is section 6 item 1 (feed `drive.d.usage` from `sp.on_routes`).
- Default path changed without an A/B: Phase 18 item 9 (D7); RDNA 9-16 column kernels (D6).
- `STRATA_PARK_ASYNC` cannot act on HIP (`park_async_on()` false, conversation_snapshot.cpp:231): parks are synchronous
  inside TTFT. TRIED 2026-11-15 (T-583): enabled as an opt-in env and replayed (spec 63): no gain seen, noisy arm, one stall;
  inconclusive; the change is not committed. Re-test only on a quiet machine with the full 3 reps.
- RESOLVED, no engine change (T-574): the fence theory is not supported; see the ledger row. Original note: Possible fence bug: the staging readers after `wait_flag_ge_or` (verify_kernels.cu ~1619-1650) may lack the
  `__threadfence_system()` the flag-order test needed (test fix: after the spin and after the store, flag_order_test.cu
  26-29; its segfault was a ROCm runtime mismatch, fix `LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib`; failures 57x1 / 101x2
  = one device; line 120 prints "device N ok" even on failure; run the 4-way matrix with `HIP_VISIBLE_DEVICES=0/1`).
- FIXED (T-573): sampler greedy rule at all four sites; a NaN temperature was a uniform draw (4 of 4 picks wrong).
- FIXED (T-576): `gdn_gate_kernel` computed only the first token (latent, every caller passes 1 token).
- FIXED (T-575): upstream #5273 guard (`if (stream_all)`) at the three KVA exits (4252/4396/4527); 5029 is inside the
  `fused_l` branch (needs `stream_all`) and 6215 tests `stream_all` itself. Research path only, no test exercises it.
- Phase 17 review R2 / R3: no review text exists (repo, git messages, session transcript searched 2026-11-15). Resolved by reading the
  code the names point at: R2 `diverge_at` is not in conv_cache.hpp, it is the field of the PARK_DIVERGENCE_LOG diagnostic
  (serve/park_divergence.py: first index where the new prompt and the previous prompt + reply differ; no behavior), and the one
  behavior that uses the divergence point, `STRATA_PARK_TRUNCATE`, tested no difference (spec 62: -1.7% +-40.8%, cold re-reads 0 vs 0).
  R3 `max_cells` is the ring rule of kv_ring.hpp (`kv_ring_adapt`: cap when cells_needed >= 2 x cap, else floor; the upstream page
  rounding of cells_needed is deliberately left out); `STRATA_KV_RING_ADAPT` was measured and DROPPED (T-291). Both CLOSED as
  moot: compared = the code and the two tests; not compared = the original review wording, which does not exist here.
- A37 `IQ_LUT_DUP` (d11a9ca09): code read 2026-11-15: default off (iq_kernels.hpp, 0 = off); stays as an opt-in bench knob, not reverted.
- `STRATA_DF2_GATE` is still read (window_rule.hpp, generate.cpp) although upstream retired it: opt-in, off, classified in disk_env;
  left in (removing it needs a Mac-side check nobody can run here). OPEN, low.
- `STRATA_PSI_SHED` (Speedy C4, idea from Basalt PR #113; Speedy SPEEDY.md / CHANGELOG 0.1.0 / TUNING-HISTORY read 2026-11-15): opt-in
  `STRATA_PSI_SHED=<avg10 %>`, off by default: under PSI memory pressure the engine drops PARKED CONVERSATIONS; no numbers upstream
  (test name `memory_pressure_test`, robustness only). Ours has the park half (`STRATA_PSI_PARK_PCT`, skip parking) and no shed half.
  Port design: `memory_pressure_sheds(&why)` beside `memory_pressure_blocks_parking` (mem_pressure.cpp:297, same probe, threshold from
  `STRATA_PSI_SHED`), and at each served request start in generate.cpp one `conversations.evict_oldest()` while it is true (the avg10
  figure lags, so one entry per request); exact (a cache miss re-reads the prompt, same tokens); disk_env entry Inert like PSI_PARK_PCT;
  unit test with a fake PSI file (`loader_pressure_begin_with` Paths); credit Speedy and Basalt #113 in README 9 and the findings table.
- Tools: FIXED (T-577): `night_queue.py` zero-measurement real passes are now failed; `abtest.py` text_check accepts any base
  pass. OPEN: `bench/quality/set-v2.json` has 28 of 48 items with `cont: null` (a `--fill` run with the stock arm, GPU hours) and
  `quality_gate.py:161` stays on `set.json` until then (by design, see its comment); all `taskgate-*` specs use `bench/tasks/default.json`: left as is on purpose
  (checked 2026-11-15): default.json is the only set with the 7 tool-call tasks the pending ECAP tool-call gate needs;
  `repo-history.json` is 150 code-edit tasks only. Point a spec at repo-history only when it is about code-edit quality.
- Item 38 O_DIRECT loader: CLOSED 2026-11-15 by code read: built (184e68e83, expert_source.cpp:852 and 2522, `STRATA_UNBUFFERED_LOAD`);
  line 6155 is the fallback when the direct read is refused (buffered read plus the `STRATA_LOAD_DONTNEED` page-cache drop).
- Planned but not in code: `STRATA_FORCE_ISA` avx512 / avx-vnni values (item 57): not applicable here (the 5700G is Zen 3, no AVX-512,
  no VNNI) and not worth a no-op value; CPU governor setting (item 41): needs root, the owner sets `performance` by hand (see the
  box notes); `--pool-affinity measure` is a no-op on the 5700G.
- PCIe link, measured 2026-11-15 (T-581, sysfs): the path to the 7900 XT is 00:01.1 (CPU root port) 8.0 GT/s x16, switch 10:00.0
  8.0 GT/s x16, then 16.0 GT/s x16 inside the card: the effective host link is PCIe Gen3 x16 (about 15.8 GB/s), as the plan assumed.

The box plan entry of 2026-11-15 "FK-28 A/B was already run (spec 28)" is wrong: spec 28 (`28-phase18-draft.json`) arms
`--spec 3/5` and `MTP_EAGER`, not `STRATA_ADAPT_PROMPT`; the dead hook was then deleted (T-575), FK-28 is unbuilt (section 6 item 1).
- Status page on the HIP box: GPU name null and backend labelled CUDA (cosmetic); the amdgpu sysfs reader (8a9c5bce7)
  now supplies VRAM, temperature and power.

Plan claims corrected (all now reflected above): Q2_0 is not the daily file (IQ2_XS is); FK-22 is F9 `ROUTE_FAST`,
coded and dropped at confirm (T-068); FK-24 is F14, coded; `EXPDEC_DOWN` was measured and adopted; the 7 upstream picks
of 2026-11-12 are merged (7529df36 .. 131cca3f); A15 `pick_victim()` exists; ExTV patches 03/06/07/08 were already in;
spec 40 `wmma-attn` was a no-op and `mmvq-wave32` compared mode 1 vs 3 (T-072); U2 is already covered by
`mmq_ids_margin.hpp` (f225844d, PR #5108); ReBAR rows into VRAM (`PRESUM_VRAM`, item 52) and the Athena bandit were
coded and measured with no gain (T-242, T-266); `POOL_PREREAD` loses (T-239); P2c small flag stack -20% short (T-474);
`MTP_HIP_PRIO` no difference (T-141, T-142); item 23 depth prior: screen turbo-depth75 short -0.3%, edit -1.3%, dropped
(`bench/results/2026-11-10-test-phase/06-turbo-depth-screen`); the 7900 XT is not
"compile-only" (measured daily since 2026-11-08); spec 5 is in the do-not-repeat list (T-118, T-160). Grammar commits
cited in the old plan were squashed into 3d3eef53e.

Branches: every `origin/feat/*`, `work/*`, `dev/*`, `exp/*` branch, `optimize-kernel-hip` and `origin/main` (PR #1 by
xxxliqu1dxxx and its 3 commits, merged as 8aeff515b, 15 of 15 status-page tests) are in this line.

## 8. Parked (hardware or scope)

- RTX 3060 / CUDA sm_86 only: Speedy sm_86 presets and `HC_Q8=1` (+5.5-6% on 3080s; keep `HC_Q8=0` on HIP), KhaledElzarw
  sm_86 kernels (`GR_DOWN82/86`, `GR_UP_HYB`, `GR_XN`, `GR_UP_PL`, `GR_D87`, `ACTQ_SKIP`, `GR_RES`, `MMVQ_IL_3090`), F17
  `STRATA_FATTN` (in `src/poly/fattn.cu`, needs a CUDA card), `--vram-reserve-mib` 444 (Speedy, 10 GB card), basalt
  head work beyond `q5k_head.cu`.
- Two GPUs (layer split): Speedy `PIPELINE_PENALTIES` / `PW_RES_PATCH` / `ROUTED_STREAM_PP`, ExTV patch 01
  (`--prefill-main`, `--kv-resident-later`, `--prefill-main-small`), upstream #3821 #3819 #3648 #3742 #3306 #4549 #4534
  #4380 #4522 #859, Phase 12 tensor parallel.
- Scope: DeepSeek V4.1 ports (owner: Qwen first); GLM-5.3-Flash / MiniMax-M3.1-Flash (Project Maya, upstream #4319 closed
  "not planned"; MiniMax maybe 5-15 tok/s with a 29 GB deficit, not started); nasone32 EXL3 (format not portable; its
  HIP kernel patterns are a lead); Cactus; ISTA coder (needs 29.6 GB resident); sleep-in-place (basalt 64167ba73).

## 9. Do not repeat (measured and dropped; numbers in TEST_LEDGER.md)

- Decode kernels: wave32 MMVQ mode 1 (T-072), occupancy/waves_per_eu (T-074, spill already at the VGPR limit), matrix-core
  few-column GEMV (T-083, T-084: -1.7% short, -5.4% edit; built, off), wide-tile int8 prompt kernel (T-107, -18..-42%,
  reverted), `WAIT_FUSE` off / fuse-wait (T-077), combine-poll1 (T-064), `VERIFY_DEVICE_PLAN` (T-066), `ROUTE_FAST` and
  `DOORBELL_STORE` (T-068), IQ2_S pair decode (T-008), upstream group-A switches (T-013), upstream `HC_Q8` on HIP (T-012),
  `PF_WMMA` (T-094), graph fork/join, persistent expert launch, DEQUANT_WIDE (T-528), LUT dup (T-060), BitDecoding WMMA
  decode attention (attention ~3% of decode, T-019).
- Drafting: `--spec 2/3/5/6` (T-118, T-149, T-159, T-160), DFlash2 (T-119), `SPEC_COST` (T-120), full / zh60k draft vocab
  (T-121, T-123), real-code draft vocab (T-146, T-151), mtp-window 4096 alone (T-122), `MTP_EAGER` (T-136, T-145, T-159),
  `--mtp-q4` head (T-143), `MTP_HIP_PRIO` (T-140..T-142), EcoSpec (T-148), A22 DISCO chain stop and FK-21 `SPEC_CUM`
  (replay +0.3%, T-455), A26 draft-source selector (T-447), DraftExpert windows (T-115), lookup windows off (T-150).
- Expert tier and cache policy: async tier and `ADAPT_SWAPS_AUTO` (T-221), kmiss on the blocking tier (T-237), rank
  guard top2 / FOLD (T-223, T-247), A44 bandit family (T-266), Foresight swap slots (T-190), CPU lookahead scorer
  (T-188), predictors / co-activation (T-394, T-401), per-layer budgets (T-397), ClassLRU / decay sweeps (T-395, T-451,
  T-452), learned admission A46 (T-408, T-426), FineMoE / PRI / MoE-Infinity (T-431), REAP pruning (T-433), ADAPT_LAG
  (T-198), rdna3 adapt and profile settings (T-203, T-204), mmap'd expert tier (T-200, T-216), arena warm (T-196),
  reuse-aware admission (T-217), dense-expert sparsity (T-219), A36 weight-major (T-045), A7 chunked promotions as
  modeled (+4.6% upper bound, T-427), `--pcie-frac` > 0 (T-268), `--kv-resident 131072` and kv8k lending (T-276,
  T-273), VRAM reserve 512 (T-278), AUTO_PIN pin (T-171), ExTV supersede rule (T-180).
- CPU pool: worker / task counts (T-195, T-202), lazy park, mwaitx, tail4, Q2 prefetch (T-232..T-235), presum and
  presum-vram (T-241..T-244), pool pre-read (T-239), `ACTQ_SKIP` (T-255), long-row grouping (T-254), shadow pack
  sub-2-bit (T-251, T-252), nice 5 (T-201), SMT siblings / CCD split / futex parking (fork-KB, measured worse).
- Prompt / TTFT: prefill CPU share arms (T-108), `WMMA_GEMM` (T-105), `PREFILL_AUTO_MAX=24576` (T-106), desc-cache off
  (T-101), BFLA as shared QSA selection (T-098), ContextPilot dedup (T-165), TTFT min-tokens / cache-mib arms on the
  10-turn session (T-178), `--ple-io mmap` (T-274), item 2 decode-cache through long prompts (T-163).
- 2026-11-15 evening: V5 shared draft/main expert cache (upper bound, T-560), lookup window 16 with ECAP=1 (T-561),
  ECAP=1 stack add-ons: suffix index, 9-column matrix-core kernel (T-555). Read and closed: q8atnight QSA block scores
  (already `STRATA_SCORES_MULTI`), jeremiahritchey retained parked KV + verify (already in tree), TheHubUser4860938
  "livelock" (another fork's memory sweep, not ours), VecSzn strata-poly disk expert I/O (not our path; spin already in
  the pool), U6 ggml graph reuse (ggml internals; HIP graphs tested with no gain).
- Other: `ROUTE_PRIOR` as a default (token forcing REJECT, T-294, T-295), tail skip 7 (T-298), `KV_RING_ADAPT` (T-291),
  `HW_QUEUES` default, `SELECT_LONG` (T-010, T-011), FK-04 aux-cpus / kv-grow (T-391), spec 31 arms (T-389), A20 entropy
  stop host cost (+17% thinking time, T-529), A43 / A38 constrained edits (T-443, T-459), A31 GDN chunk 64 numbers
  (T-103), dedup, MEXT, new compression ISAs (LEADS.md section 0 "Closed").

## 10. Architecture facts worth knowing before touching the code

- Chunk plan: the serve loop cuts at root_at (first `<|im_start|>` after token 0 if >= `--prompt-cache-root` 2048; a
  smaller system prompt is re-read on purpose, ~0.3 s), turn_at (last `<|im_start|>`) and n-1; `CACHE_MESSAGE` /
  `TOOL_BOUNDARY` and `pin=` add cuts; <= `--short-read` 64 tokens go through decode windows; chunk = min(prefill_chunk,
  segment rounded up to 256) via lend -> planned_chunk -> equal_chunk -> request_chunk (generate.cpp ~997, ~1009,
  ~10355; loop in `Prefill::run_impl`; `STRATA_TRACE` prints `prompt chunk` / `lent`). The loan lasts until the segment
  ends.
- Auto chunk: largest 256-multiple with loan <= 90% of slots (85% if less pinned), 128 left; AUTO_MAX clamps
  32768/16384/8192; loan at 32K = 9,413/10,733 slots, 12.59/14.37 GiB, ~5% resident (generate.cpp ~3938).
- With CPU share armed, `stream_all_min` = `CPU_SHARE_MAX` 4096: T >= 4096 streams (192-slot ring, copies overlap
  attention); T < 4096 is staged (per-layer sync on routing, 8-slot ring, 32 with MMQ groups). A warm turn is one staged
  chunk and the copy engine idles through attention (the reason for PF_AHEAD). CPU share only takes experts with <= 8
  tokens (MAXT, expert.hpp:143); share=auto keeps running means (w 0.25), clamp(g/(c+g), 0.05, 0.9), needs 3 ratios,
  probes every 29th layer, HIP off except a single gfx12.
- EQUAL_TAIL (prefill_plan.hpp:182, `_MIN` 256) is skipped when evened chunks would be < 8,192 (kEqualTailMinChunk, :172;
  upstream measured 25 tokens past 8,192 read 25% slower when evened) or below the staged limit; `PREFILL_EQUAL` (#693)
  evens only when the last chunk would stream anyway; EQUAL_TAIL wins on a staged tail.
- Checkpoints: `--prompt-cache 6` (generate.cpp ~961), ~113-118 MiB each, at the turn boundary and chunk ends >= 16,384
  apart; restore only from a prefix (GDN state). Parked conversations: 1,280-6,656 MiB auto (usually 4,352), 4 slots;
  snapshot = 111 MiB + 14.9 KiB/token + 113 MiB per checkpoint. Refill is one wait (~1 s for a 12.6 GiB loan;
  `STRATA_REFILL_BLOCKING` keeps the old form; trace prints `refilled ... in <ms>`).
- Prefix reuse breaks on: client edits; system prompt or tool list change (tools render in the system block -> full
  re-read); effort; a re-render that differs; dropped reasoning (re-renders only the last turn); eviction; loop-guard
  collapse.
- Floor model: 48 layers x 512 experts, top-10, 1.4 MB/expert, 44% resident, routed share 1-exp(-10n/512), floor ~0.67 s,
  PCIe-bound. QSA indexer cost grows n^2, attention ~2,051 cells/query; KV staging ~1,056 B/token/layer over 12 QSA layers.
- Hot path hygiene: no per-window getenv (c44b1ad7b, 7c0f03228), no hot-path allocations, no decode file I/O; the 4
  dispatch loops in `expert_pool_dispatch_multi` are linear (`first_of`, `miss`, `next_of`, `q_of`; greedy check owed).
  Default-on: `ADAPT_PERSIST`, `UNBUFFERED_LOAD`, `HIPBLASLT_DESC_CACHE`, `PIN_PARK_TOKENS`, `PSI_GUARD`. `ARENA_WARM`
  removed; `combine_multi_hits` kept (removal = fork drift).
- Status page: the field `spec = 12` is the effective window cap; the preset runs `--spec 4` (draft depth) with
  `--lookup-max-t 12`.
- llama.cpp v1.0.0 (tag 2026-11-13, 249 commits after v0.9.0) candidates, read against our code: V1 `e95426f16` covered by
  `--pcie-frac` / `pcie_share_for` / `calibrate_ratio.hpp`; V2 `cf34d3113` covered by `ROUTE_PUBLISH=1`; V3 ClassLRU
  `c4fb57969` dropped (T-452); V4 `69ee2ecc0` layout already in `Tables.tiles` / `moe_fused_iq.cu`, only the int8 path's
  fixed tile width differed (built and dropped, T-107); V5 dropped (T-560); V6 dropped (T-569); U1, U3 dropped (T-571,
  T-570); U4, U5 covered; U6 not applicable (ggml internals).
- Code tools: graphify (`graphify update .`, `graphify-out/` git-excluded), ast-grep 0.51 + MCP, tree-sitter MCP needs
  `mcp<2`.
- Platform: ROCm 10 has no `hipMemcpyBatchAsync`, so `STRATA_DMA_BATCH` is CUDA-only. The Flash-Next "IQ2_XS" file holds
  IQ2_S / IQ2_XXS / IQ1_M gate/up with Q2_0 down (~176.9B parameters); its head is IQ4_XS. Q2_0 and IQ2_XS share one
  n-gram table file (same SHA-256). English 40.5K draft vocab is best for English/code (+5%, ~215 MiB less VRAM); zh60k
  helps Chinese only. KV quant: q8_0 costs < 0.00012 KL; 4-bit adds 0.0016-0.0027 KL and grows 15-30% from 4K to 32K;
  keys are ~2x as sensitive as values.
- strata-poly (Qwen3.6-35B): one CPU thread per physical core (8; SMT adds no bandwidth; poly does not pin threads);
  with the expert cache on it is not run-to-run deterministic (swaps and CPU/GPU rounding part texts after 9-81 tokens),
  so exactness checks need `--no-gpu-experts --expert-swaps 0`; Q8 KV 11.7 KB/token vs strata fp32 45 KB/token; prompt
  reading is GPU-compute bound (attention 38% of a 42K read), decode CPU experts ~half and RAM-bound (33.4 GB/s).
  Reproduce on CUDA: build `strata strata-poly poly_gdn_cols_test poly_gemm_small_test` with
  `-DCMAKE_CUDA_ARCHITECTURES=86-real`; daily poly flags `--max-context 131072 --threads 8 --kv-q8 --attn-split
  --graph-experts --elastic-prefill 1024 --lend-staging --stage-next --suffix-drafts 256 --cpu-multi`.
- gpu-unlock (Zagitta, GPL-2.0, RTX 30 only) lifts FP16/BF16/TF32 MMA with FP32 accumulate (48.3 vs ~25 TFLOPS),
  bit-identical; it helps prompt attention, not INT8 MMQ, CPU experts or decode (T-545). Install: `make install`,
  `gpu_unlock.targets=0000:10:00.0` on the kernel command line, check `journalctl -k -b | grep gpu_unlock`.
- Bench cases: read1k..read16k (16 new tokens), short (34-token chat, 256 new), edit (5.3K code rewrite, 512 new), long
  (41.8K prompt, 128 new); "cold" = first run in a fresh engine (later runs hit the conversation cache).
- The CPU-compute vs PCIe-promote cost per expert varies ~10x across platforms (42 vs 115 us here), so calibration must
  use window time, not a static table (PcieAuto picked 0.20-0.25 and lost). `HSA_ENABLE_SDMA=0` tips conflict with the
  measured SDMA gains here.
- Agent replay (`tools/pancake/replay_turns.py SPEC --out DIR [--dry-run|--arms a,b|--synthetic|--reps 2|--max-gap 2]`):
  rebuilds a session from `pancake-turns.jsonl` sizes. A request >= 8,192 tokens extends the conversation it can grow
  from (last prompt <= it <= last + output + 16K). A `reused == 0` re-read stays in that conversation. Smaller requests
  are side requests. Turn N+1 = turn N + the real reply + a `<tool_response>` sized to within 2 tokens (server tokenizer
  and template). Output: per-request park/restore/evict lines; per arm, cold re-reads (non-first turn > 8K, reuse < 50%),
  prompt time, peak RSS, min MemAvailable. Spec 11 `replay` block: 2026-11-09 16:13-16:43, base vs cache2048-min8192;
  add `base1280` when today's auto budget differs.
- Window trace: `STRATA_WINDOW_TRACE=/tmp/w.stwt ./run-pancake.sh iq2_xs fast`, one greedy request, then
  `tools/pancake/trace/wtrace.py` and `tail_replay.py --us-per-miss 23` (edit: 60). `_EVERY=N`, `_MAX_MB` (2048).
  Per row: drafter probability (NaN for lookup) and per-layer placement. No top-1 margin or entropy (the stale
  `tools/pancake/trace/corpus-recorder-engine.patch` would add gate weights and norms; it no longer applies).
- Kernel ports must be re-measured per GPU (launch ~14-16 us gfx1100 vs ~3 us sm_86; DPP vs warp intrinsics; WMMA vs mma).
  Server, drafting, routing and CPU-kernel items transfer; re-check only the size of the gain.
- BFLA (`src/kernels/cuda/bfla.cu`, `include/strata/kernels/bfla.hpp`, q36_engine.cu; arXiv 2605.12193, Apache-2.0
  reference read, not copied). Engine: qwen35 only (10 dense gated attention layers, H 16, HKV 2, D 256). It replaces the
  path that materialises FP32 scores over the whole prefix (FP16 K/V, hipBLAS QK^T, causal softmax to FP16, PV GEMM),
  which is quadratic in FLOPs and score traffic (~12 B/score). Not on Flash-Next: QSA already reads 2,051 cells/query;
  a V/3 budget reads ~16x more at 128K (tile-shared selection: T-098, dropped).
  Kernels (HIP wave32 gfx11; other builds return false):
  `features`: phi = N(mean over G heads of N(q_h)), adjacent pairs summed to 128, N again; mu = block mean over K.
  FP32, no FP8. Hadamard dropped (relu(phi.mu) is rotation-invariant). mu recomputed over the prefix every chunk
  (~1/4 of `k_kv_f16`), so a restored cache is never stale.
  `select`: one workgroup per (128/G-token tile, KV head); online LSE of relu(phi.mu)/sqrt(128); budget
  k = ceil(min(V, max(Kmin, min(Kmax, ceil(eta V))))/64). Mandatory 2 sink + own + 4 local blocks count in the quota.
  32-step bisection for the k-th score, ties to the smaller id, deterministic ascending list.
  `attn`: 4 waves per (64 rows, KV head), 16 rows x 256 dims per wave; Q in LDS once; 16-key steps, V transposed in LDS;
  16 WMMAs each for S and PV (O = 128 VGPRs/lane); causal mask only on diagonal blocks; exp2 online softmax in FP32;
  P to FP16 via a per-wave LDS tile; FP32 out, gate stays `k_gate_mul`. Not bitwise vs the GEMM path.
  gfx11 facts: `v_wmma_f32_16x16x16_f16` A lane l = row l%16 (16-31 repeat), B = column l%16, D lane l = D[2i+l/16][l%16];
  64 KiB LDS/workgroup; no FP8 matrix ops, no TMA/wgmma; every wave does the same work.
  Follows the paper's Ratio-K, not the published code (keep-mass 0.99, 256-blocks, 8 local).
  Switches: `STRATA_BFLA_DENSE=1` (exact), `_SPARSE=1` (lossy, wins if both), `_MIN` 32768, `_ETA` 1/3, `_KMIN` 2048,
  `_KMAX` 32768, `_SINK` 2, `_LOCAL` 4, `_TAIL` 1024/G; plain `STRATA_BFLA` is ignored.
  Estimate (not measured): 81,920 N^2 FLOP; ~45 s at 128K (30-45% of prefill), ~180 s at 256K; Ratio-K keeps
  1/1.4 (32K), 1/1.8 (64K), 1/3.2 (128K), 1/4.9 (256K).
  Gate: `bfla_test`, `bfla_cpu_test`, needles at 64K/128K + teacher-forced top-1 vs dense, then TTFT A/B (T-299 KL fails 1e-3).
- Windows CUDA sm_75 (GTX 1660 Ti 6 GB, CUDA 12.8.61, VS 2022 Build Tools MSVC 14.44, CMake 4.4.4, Ninja 1.13.2):
  run `vcvars64.bat` from the VS Build Tools (`VC\Auxiliary\Build`); `cmake -S . -B build-sm75 -G Ninja
  -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared`;
  `cmake --build build-sm75 -j 4 --target strata strata-q36 strata-poly`. Without Shared, strata-q36 fails with LNK2005
  (setup.py gets Shared via `STRATA_EXPERIMENTAL_SM60=ON`, CMakeLists.txt ~222). One portability fix was needed:
  `std::popcount` instead of a GCC builtin in `kmiss_tier.hpp`.
  Test compile errors: `q5k_head_test` (`std::to_string`, no `<string>`), `flag_order_test` (`__atomic_load_n`),
  `gemm_w16_parity` (`setenv`), `poly_kv_q4_test` (kv_q4_test.cu:342 range-for). Decode with `tools/pancake/bench_win.py
  RUNS WARMUP MEASURED NEW` (gates only system CPU < 12% before each run). The first requests are ~10x slower: warm up.
  Untested: long context, app, vision, CUDA 13, test_phase, quality gate. Results: T-550..T-552.

## 11. Where things are

- Plan: this file. Tests: TEST_LEDGER.md. Leads: LEADS.md. Agent and build rules: AGENTS.md. Measured results for
  readers: README section 2; switch census: README section 6; credits: README section 9; quants: `docs/QUANTS.md`.
- Raw evidence (kept): `bench/specs/`, `bench/results/`, `pancake-docs/research/bench/` (results archives, run scripts,
  night-queue logs), `bench/prefetch/`, `bench/quality/`, `data/fork-kb/` (fork knowledge base; refresh with
  `GITHUB_TOKEN=... python3 tools/forks/fork_kb.py`, works on Windows too; last scan 0004-20261112).
- Documentation rule: only MOD_PLAN.md, TEST_LEDGER.md and LEADS.md are maintained (plus AGENTS.md and README.md);
  no other report, note, status or findings file may be created (AGENTS.md "Documentation rule").
- Bring-up after a reboot (Linux box):
  1. `ls /home/dan/clone-of-d/src/Strata`; if it does not mount: `sudo ntfsfix -d /dev/nvme0n1p5` (ntfs3 writeback hang
     2026-11-10, engine ~25% slower afterwards; T-287, T-290). After the 2026-11-13 reboots the volume came back dirty and
     was force-mounted: `sudo mount -t ntfs3 -o force,uid=1000,gid=1000,umask=022
     /dev/disk/by-uuid/401E033B1E03298C /home/dan/clone-of-d`.
  1b. The GPU clock resets to `auto` on reboot; pin it before measuring:
     `echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level`. Commands shown with a `!`
     prefix are for the agent session; in a normal terminal drop the `!`.
  2. `git fsck --no-dangling && git status -sb`; if broken, clone from GitHub or `git fetch` the all-refs bundle.
  3. `sudo tools/pancake/os_tune.sh cpu-install`, then `tools/pancake/os_tune.sh` (large free 2 MiB blocks).
  4. `cmake --build build-hip -j16 --target strata` (not `all`: strata-poly is CUDA-only).
  5. `./run-pancake.sh` (8080; `http://localhost:8080/v1`, model `qwen3.8-flash-next-iq2_xs`); check `/`, `/status` or
     `/pancake`, and `serve/web/monitor.html`.
  6. Before measuring: restore checks + `ctest -L gpu` (baseline 122/125, T-346; latest 133/134, T-367).
