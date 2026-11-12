# Pancake verify path: where Avalon's speculative advantage could port (MOD_PLAN 17.2)

Scope: read-only. Nothing was built, run, checked out or edited. No git checkout or worktree command was run.

## Sources and tree used

- Main checkout `/home/dan/clone-of-d/src/Strata`, branch `port/rdna3-decode`, HEAD `2ba5a9459`. All `file:line` references below are to this tree unless marked otherwise. `src/program/generate.cpp` here is 21,978 lines.
- `pancake-docs/PROSE_GAP_AVALON.md` is NOT in the main checkout. It exists only in worktree copies. All 7 copies in `.claude/worktrees/*` are byte-identical (md5 `0e4cd97d`). I read `.claude/worktrees/agent-ad25cdf1dc7ea704a/pancake-docs/PROSE_GAP_AVALON.md`. Its line numbers (for example `generate.cpp:22707`) do not match this tree, so I re-anchored every reference to the main checkout.
- `src/core/conversation_disk_env.def` is NOT in the main checkout either. The newest copy is in worktree `agent-ad25cdf1dc7ea704a` (96,061 bytes, 2026-11-11 22:44). Other copies differ (md5 `e652fbd2`, `db36493f`, `f8758320`). Env-table classifications below come from the `ad25` copy and may differ from the main tree. Treat them as a second-hand reading until the file is checked on `port/rdna3-decode`.
- `MOD_PLAN.md` items 17.2 (line 2446) and 17.8 (line 2452) read.
- `configs/pancake/flashnext-iq2_xs-fast-hip.json` read (the daily HIP preset).
- Avalon's source was NOT re-read. Every Avalon claim here is carried from `PROSE_GAP_AVALON.md` and is unverified in this note.

Labels: **[code]** = read from pancake source, with file:line. **[table]** = read from the env table (ad25 copy). **[doc]** = carried from PROSE_GAP_AVALON.md, not checked. **[unverified]** = not checked.

## Q1. Where drafts are proposed and verified, and what sets the draft count

### Takeaway
The daily preset's serve path uses a single-stream round loop: the MTP drafter proposes, the window is chosen by `choose_window`, and one `ver.run` verifies the window. The MTP window is 4 rows (`--spec 4`), which means at most 3 drafts per window. The lookup (suffix) window can be up to 12 rows (`--lookup-max-t 12`). The PROSE_GAP doc says "up to 4 drafts"; that is wrong for this preset.

### Cited findings
- Preset args [code, configs/pancake/flashnext-iq2_xs-fast-hip.json, args line]: `--spec 4`, `--spec-min-p 0.5`, `--mtp <STRATA_DATA>/mtp/rt`, `--mtp-draft-vocab data/draft_vocab_en.bin`, `--adapt-swaps 32`, `--adapt-every 1`, `--lookup-max-t 12`, `--suffix-draft` not given (default 3, see below). The preset does not set `--pipeline-windows`, `--batch` or `--lookup-chain`.
- Flag parsing [code]: `--spec` at `src/program/generate.cpp:3599`; `--mtp` at `:3614`; `--spec-min-p` at `:3621`; `--adapt-every` at `:3617`; `--lookup-max-t` at `:3681`; `--mtp-draft-vocab` at `:3700`; `--adapt-swaps` at `:3760`.
- Defaults [code]: `spec` 0 at `:860` (Metal default 2 at `:858`); `spec_min_p` 0.0 at `:911`; `adapt_every` 4 at `:894`; `adapt_swaps` 96 at `:933`; `suffix_draft` 3 at `:991`; `lookup_max_t` -1 (auto) at `:1004`; `pipeline_windows` -1 (auto) at `:986`.
- Window width derivation [code, generate.cpp:4352-4383]:
  1. `:4352-4356`: `suffix_draft > 0 && spec >= 2 && mtp_max_t == 0` sets `mtp_max_t = spec` (4) and `spec = min(spec+2, 8)` (6).
  2. `:4362-4371`: lookup default only when `lookup_max_t < 0`. The preset passes 12 explicitly, so this is skipped.
  3. `:4372-4375`: `lookup_max_t > 0` sets `spec = max(spec, min(lookup_max_t, kVerifyLongT))`. On HIP this gives `spec = max(6, min(12, 16)) = 12`. `mtp_max_t` stays 4.
  4. `:4379-4383`: `spec_cap = kVerifyLongT` in this case. `spec` 12 is within it.
  5. Result: MTP window `S_mtp = min(mtp_max_t, spec) = 4` rows; lookup window up to 12 rows.
- Window constants [code]: `include/strata/kernels/window_limits.hpp:21` `kVerifyNarrowT = 8`; `:17-23` `kWindowMaxT = STRATA_VERIFY_MAX_T` (default 8). `include/strata/kernels/verify_kernels.hpp:25` `kVerifyMaxT = kWindowMaxT`; `:31` `kVerifyLongT = kVerifyMaxT > 8 ? kVerifyMaxT : 2*kVerifyMaxT`. `CMakeLists.txt:118-131`: `STRATA_WINDOW_MAX_T` default `auto` (comment at `:118`: "HIP 16, CUDA 8"). The HIP build cache has `STRATA_WINDOW_MAX_T=auto` (`build-hip/CMakeCache.txt:844`). So on HIP, kVerifyLongT = 16 [code + CMake comment; the actual built binary is unverified].
- Lookup default [code]: `include/strata/core/lookup_default.hpp:23-28` returns 12 only for an 8-row build (`narrow_window`). On HIP the default is 0. The explicit `--lookup-max-t 12` is therefore needed and present.
- MTP draft cap [code]: `set_max_drafts(S_mtp - 1)` at `generate.cpp:16252` (serve), `:21237` (solo). The MTP drafter computes `max_steps = min(max_t_-1, max_drafts_)` at `src/core/mtp.cpp:1794` and `:1816`. With `S_mtp = 4`, drafts per window are at most **3**.
- Verify call [code]: one `ver.run(T, window, ...)` per round, serve path at `src/program/generate.cpp:19585`; solo path at `:21484` (speculative section begins at `:20842`). The serve block is `if (o.serve) {` at `:10660` through `:20421`.
- Serve window choice [code]: `policy.choose(...)` for lookup-vs-MTP at `:19302` and `:19351-19352`; `WindowRuleIn` filled with `spec_min_p = req_spec_min_p` at about `:19330`; `choose_window` at `include/strata/spec/window_rule.hpp:57-71`.
- Pipelined serve is off on one GPU [code]: `generate.cpp:5744-5763` turns `--pipeline-windows 2` on only for a layer split (`multi_gpu`). The daily preset is single GPU (`gpu: 0`), so the pipelined branch (`:18700-18990`) is not the daily path. [unverified that no other gate enables it; I read the auto rule only].
- Adaptive knobs [code]: `--adapt-every 1` and `--adapt-swaps 32` set the expert-swap cadence and count (`generate.cpp:894`, `:933`, `:13338-13347` swap budget). They do NOT set the verify width. The `STRATA_SPEC_ADAPT_EVERY/WARM` warm-up applies only when `--adapt-every` is not given (`:4394-4404`), so it does not apply to the preset.

### Inferences
- The MTP draft count is 3 at most per round under the preset, and it is capped by `--spec 4`, not by `--spec-min-p`. `--spec-min-p 0.5` only shortens below that.
- A lookup window of up to 12 rows is the widest verify in the preset. Its width is chosen by the lookup policy, not by the MTP.
- The doc's "drafts up to 4, window 1-5 rows" (PROSE_GAP section 1 and 3.2) should read "MTP: 1-4 rows, 0-3 drafts; lookup: up to 12 rows".

### Gaps
- Whether the 7900 XT binary (`build-hip/strata`) was built with `STRATA_WINDOW_MAX_T=16` (kVerifyLongT 16). Not checked; the CMake `auto` rule says 16 on HIP.
- The serve-mode accept loop after `ver.run` (how many tokens a round commits) was not read in this pass. The `1 + accepted` commit formula is [doc], unverified.

## Q2. Acceptance rule, and whether `--spec-min-p 0.5` is a lossy filter

### Takeaway
Pancake's acceptance rule is exact-match against the target's own pick: greedy argmax, or the sampled token. `--spec-min-p` does NOT change which tokens are accepted. It only sets how many drafts are put in the window (a width rule on the drafter's probability). Output is therefore unchanged by `--spec-min-p`; the cost is fewer drafts per window. Lossy switches are elsewhere (see Q4).

### Cited findings
- Verify contract [code]: `include/strata/core/verify.hpp:6-8`: "Token t's argmax is what plain greedy decode would produce after token t, BIT FOR BIT ... So a draft is accepted exactly when greedy decode would have produced it."
- Greedy default [code]: verifier starts greedy at `src/core/verify.cpp:910`; serve request default `req_temperature = 0.0f` at `src/program/generate.cpp:16071`. The struct default `temperature = 1.0f` at `:618` applies to the sampler struct, not the serve request default [unverified how the two interact for every path].
- Sampled acceptance [code]: `include/strata/core/verify.hpp:172-178`: rejection sampling only when a q list is set (`set_spec_q`). `src/core/verify.cpp:3988-3992`: judged path only if `sampled` and `spec_q_` non-null and `T > 1`. Otherwise exact match (`spec_prob.hpp` header lines 1-30: "Under sampling the verify window used to accept a draft only when it EQUALS the token the target sampled").
- STRATA_SPEC_PROB default off [code]: `include/strata/core/coupled_draft.hpp:69-76` (on only when the env is set to a non-"0" value). [table] `STRATA_SPEC_PROB` is class Speed, note "drafting/stop policy: the tokens change, the state of given tokens does not" (ad25 env table line 709).
- `--spec-min-p` gate [code]: CLI default 0.0 (`generate.cpp:911`). Window rule: `include/strata/spec/window_rule.hpp:61-63`: `T = 1; while (T < cap && dprob[T-1] >= spec_min_p) ++T;`. The cut uses the drafter's own probability, not acceptance. Step-by-step chain cut: `src/core/mtp.cpp:1936` `for (j=1; j<max_steps && pj >= min_p; ++j)`. Eager variant (`STRATA_MTP_EAGER`) at `:1864-1905`, marked "(opt-in, exact)" in its comment.
- Policy comment [code]: `include/strata/spec/draft_policy.hpp:1-20`: "It only chooses which drafts to verify: the output is unchanged."
- Lookup vs MTP [code]: `generate.cpp:19351-19352` (serve), `:21405-21406` (solo); `DraftPolicy::choose` compares expected tokens per ms (`draft_policy.hpp` header).

### Inferences
- For greedy requests the preset's output does not depend on `--spec-min-p`, `--spec`, or the lookup policy (by the verify contract). Changing draft depth, head quality, draft vocabulary or lookup policy changes speed, not text. This needs one check: the same prompt with `--spec-min-p 0` and `--spec 2` should give identical text on greedy. [unverified; this is the test to run].
- So `--spec-min-p` is not "a lossy filter the way Avalon is not". It is a throughput rule, same class as Avalon's fixed depth 2.

### Gaps
- The [table] classes spec-policy switches as Speed with the note "the tokens change". That note conflicts with the verify contract for greedy. Either the note means sampled runs (where different draws change the stream), or the table is wrong. Not resolved here.

## Q3. One verify forward per window, and the window sizes

### Takeaway
One verify forward per round. The MTP window (4 rows) and the lookup window (up to 12 rows) are separate widths of the same verify. `--adapt-every` and `--adapt-swaps` do not change the verify width.

### Cited findings
- One `ver.run` per round: `generate.cpp:19585` (serve), `:21484` (solo).
- Window = `x` plus drafts: `generate.cpp:21440-21447` (solo) and serve equivalent around `:19396-19404`: `window[0] = x; window[i] = drafts[i-1]` or lookup tokens.
- The 12-row lookup window is taken where `policy.choose` says it pays (`:19351`, `:21405`).
- Per-window [doc, derived] numbers: prose tokens per window 1.83-2.11 on pancake, 2.04-2.30 on Avalon (PROSE_GAP table section 1). Not re-measured here.

### Gaps
- Whether ver.run's cost grows with T on the 7900 XT (the "cost of one more row" question) is not measured. PROSE_GAP cites `src/spec/draft_policy.hpp` ("a window of 3 costs about a window of 2"); that comment was not read in this pass [unverified].

## Q4. Mechanism-by-mechanism: exact or lossy, and is it already in pancake

Exact = same text as no-change for greedy decode (by the verify contract in Q2). Lossy = changes the computed state or text.

| Avalon mechanism ([doc]) | Pancake equivalent (code) | Already in pancake? | Output effect | Port note |
| --- | --- | --- | --- | --- |
| Draft depth 2, no early stop (`p_min` 0) | `--spec 4 --spec-min-p 0.5`; `choose_window` at `window_rule.hpp:61-63`; `set_max_drafts(3)` at `generate.cpp:16252` | yes (different shape) | exact (speed only) | flag-only A/B: `--spec 2 --spec-min-p 0` |
| MTP head IQ3_K/IQ4_NL 1.48 GB vs pancake Q2_0 0.71 GB [doc] | `--mtp <dir>`, `--mtp-draft-vocab` (`generate.cpp:3614`, `:3700`); head loader `src/core/mtp.cpp` | pancake has its own head | exact (speed only) | expensive: new expert layout, loader, head kernel, +0.7 GB VRAM [doc] |
| Draft vocab 65k reduced head [doc] | `--mtp-draft-vocab data/draft_vocab_en.bin` (preset) | yes, 40.5K English | exact (only limits drafts) | MOD_PLAN 17.8 says output cannot change; offline gain under 1% |
| ngram-simple top-priority drafter [doc] | suffix drafter `--suffix-draft 3` (default, `generate.cpp:991`) and `DraftPolicy` | yes | exact (policy only) | no port |
| Tail skip default 7 [doc] | `STRATA_ROUTE_TAIL_SKIP` read at `generate.cpp:12933`; default on CUDA and gfx1200/1201 only (`:12927-12963`); off on gfx1100 | switch exists, preset sets 0 | **lossy** (table: "changes the output") | do not port as an exact change; confound only |
| Expert cache preload from ranking [doc] | `--expert-profile data/expert-profile-pancake.bin`, `--adapt-swaps 32`, `--adapt-every 1`, `STRATA_PIN_BY_NEED=1` (preset) | yes | exact (placement; table class Speed) | nothing to port |
| CPU experts threaded, mailbox [doc] | `STRATA_COMBINE_CPU`, `STRATA_WAIT_FUSE`, `STRATA_ROUTE_PUBLISH`, `STRATA_VERIFY_DEFER_CPU` (preset) | yes | see next row | nothing to port |
| `STRATA_VERIFY_DEFER_CPU=1` (PRESENT IN PRESET) | `src/core/verify.cpp:974-977`; preset sets `STRATA_VERIFY_DEFER_CPU: "1"` | **yes, ON in the daily preset** | **lossy** (table line 42: "CPU rows deferred a window: lossy") | a baseline issue, see below |
| MoE down-proj in one MUL_MAT_ID [doc] | n/a | prefill only | none on decode | not applicable |
| MTP draft from device rows [doc] | MTP draft graphs already on GPU (`src/core/mtp.cpp`) | yes | none on decode | not applicable |
| MoE cache pregate [doc] | `STRATA_*` lookahead routing [doc] | off in Avalon | none | not applicable |
| Dispatcher moves missed experts to GPU [doc] | `--pcie-frac` (preset 0) | partly | unknown | skip until a cost gap is shown |
| Fused decode kernels, PDL [doc] | `STRATA_GDN_*`, `STRATA_FUSE_CONVAB`, etc. (preset) | yes | exact (table class) | nothing to port |

Other exactness flags in the daily preset [table]: `STRATA_GR_RDNA_A8=1` ("int8 activations + dot4 in the Q8 read (lossy)"), `STRATA_GR_RDNA_Q8=1` ("int8 HC read on RDNA (lossy)"), `STRATA_PF_F16=2` ("prompt experts on F16 matrix-core kernels (rounding differs from int8)"). All three are set in the preset. So the daily baseline is not output-exact in several places, independent of speculation. Not verified against code here.

### Takeaway
Draft-side Avalon ideas (depth, head, vocab, policy) are output-exact for greedy and port as speed-only changes. The lossy Avalon idea (tail skip) is already a switch in pancake and off on gfx1100. The only lossy item relevant to the prose gap that pancake already runs by default is `STRATA_VERIFY_DEFER_CPU=1`, which the PROSE_GAP doc did not flag as lossy.

### Inferences
- The prose gap (tokens per window, PROSE_GAP section 1) is a drafter-quality or depth question, not a verify-exactness question. Both are exact-output levers, so a prose A/B can test them without changing text.
- If `STRATA_VERIFY_DEFER_CPU=1` is lossy as the table says, Avalon's prose text and pancake's prose text differ partly for that reason too. An A/B with it at 0 is a cheap control [unverified cost].

## Corrections to PROSE_GAP_AVALON.md (for the lead)

1. "Pancake drafts up to 4 ... window 1-5 rows": wrong for the preset. MTP drafts are at most 3 (window 4 rows) via `set_max_drafts(S_mtp-1)` (`generate.cpp:16252`; `mtp.cpp:1794, 1816`). Lookup windows go up to 12 (`generate.cpp:4372-4375`).
2. "STRATA_VERIFY_DEFER_CPU=1 defers the late CPU expert rows (... I did not read that code path)": now read. It is in the preset, and the env table classes it lossy. It must be in the exactness accounting.
3. "`generate.cpp:22707` spec cost table": no such line in this tree (the file is 21,978 lines). Not found by search in this pass [unverified location].
4. "Avalon's windows are 3 rows (1 + 2 drafts)": Avalon side not re-read [doc].
5. MOD_PLAN 17.2 (line 2446) says "same acceptance" for pancake and Avalon. PROSE_GAP section 1 derives accepted drafts per window of 1.13 (Avalon) vs 0.83 (pancake) on the story prompt. These conflict. The derived numbers are from the saved bench files, not re-checked.
6. Line refs `src/core/mtp.cpp:1936` (min_p chain cut) and `src/spec/draft_policy.cpp` are consistent with this tree or the doc's own tree; mtp.cpp:1936 matches here.

## Gaps and unverified, in one list

- Avalon's source, its commit range, and its tail-skip/flag state for the 2026-11-11 run (PROSE_GAP section 5).
- Whether `build-hip/strata` on the 7900 XT was built with 16-row windows (CMake auto).
- The serve accept loop after `ver.run` (commit count per round).
- The `ver.run` cost per extra row on gfx1100 (no measurement recorded here).
- The `ad25` env table vs the main tree. Re-read `src/core/conversation_disk_env.def` on `port/rdna3-decode` before quoting classifications.
- Whether `--spec-min-p 0` and `--spec 2` give identical greedy text on this build. This is the one test that confirms the exactness claim of Q2. It needs the GPU and was not run.
- Whether `STRATA_GR_RDNA_A8`, `STRATA_GR_RDNA_Q8` and `STRATA_PF_F16` (set in the preset) change the output on the 7900 XT, as the table says. Not checked.

## Suggested next steps (no GPU run in this pass)

1. Confirm the exact-match claim: greedy text for one prose prompt with `--spec 4 --spec-min-p 0.5` vs `--spec 4 --spec-min-p 0` vs `--spec 2 --spec-min-p 0`. Identical text means the policy flags are exact.
2. Control for `STRATA_VERIFY_DEFER_CPU`: run the same prose prompts with it at 0 and read the tokens-per-window and text hash.
3. Only then consider the costlier ports (head precision, 4-bit head). The draft-side gain is capped by the 3-draft window in the preset.
