# Route prior only in lookup windows (`STRATA_ROUTE_PRIOR_AUTO`)

MOD_PLAN Phase 17, item 17.1. Branch `dev/p17/turbogate` (from `origin/dev/phase16-academic`).

**Status: coded, NOT built, NOT measured.** A benchmark matrix held the GPU and the CPU while this was written, so
nothing was compiled or run. The C++ was checked by reading it and by a tree-sitter parse of the new files (no syntax
errors). Types and linking are not checked. The unit test is written and registered in CMake; it has not run.

The idea is ours: we found no fork or paper that gates the route prior by the draft source. No credits rows were added.

## 1. What the code does today

`STRATA_ROUTE_PRIOR=lambda` ("turbo") is the cache-conditioned routing of arXiv 2412.00099, without training. When a
token's 10 experts are picked, an expert already in VRAM gets `lambda` added to its router logit. The pick uses the
biased logits; the weights stay the router's own. Fewer picked experts miss VRAM, so fewer are copied over PCIe or run
on the CPU. It changes the output (lossy).

Where it lives:

| piece | file | when it is read |
| --- | --- | --- |
| `lambda` | `src/core/verify.cpp`, `g_route_prior()` | once per process (a function-local static) |
| the kernel launch | `src/core/verify.cpp`, the router step, `route_prior_top10(...)` once per MoE layer per verify window | the launch is there whenever `lambda > 0` (and 512 experts x top-10, residency table present) |
| the kernel | `src/kernels/cuda/route_prior.cu`, `route_prior_k` (one source for CUDA and HIP) | every launch |
| `STRATA_ROUTE_PRIOR_DEPTH` (item 23) | `verify.cpp`, a static next to the launch | once; passed as a launch argument |
| `_PROTECT`, `_KEEP_P` (F13) | `generate.cpp`, `prior_guards_init` -> `route_prior_configure` | once at start-up, before any window graph exists |
| `_SCOPE` gate (F13) | `generate.cpp` (decode loop) + `route_prior_scope.hpp` + `route_prior.cu` | **a value read by the kernel at run time** |

### Can the prior differ per verify window?

This was the key question. The answer from the code:

- **`lambda` cannot change per window by itself.** It is read once and passed by value as a kernel argument. The verify
  windows run as captured graphs, and a captured graph keeps the launch arguments it was captured with
  (`route_prior.hpp`: "the kernel's launch arguments are baked into a captured graph"). The same holds for the depth
  grading, `protect` and `keep_p`.
- **A per-window switch already exists and is safe to reuse.** MOD_PLAN F13 added a gate: a mapped (pinned, portable)
  host float whose device pointer is baked into the graphs once. The kernel reads the value itself at run time
  (`*(volatile const float*) gate`). If the value is 0 or less, the kernel returns at once and the router's own top-10,
  already in `ids_` / `w_`, stands: that window is routed exactly. If it is above 0, it scales `lambda`. The serve
  decode loop writes it before each verify window (`prior_gate(open)` just before `ver.run`). It starts closed, is
  closed while the prompt is read (unless `STRATA_ROUTE_PRIOR_ON_PROMPT=1`) and is closed again when a request ends.
- **What stays fixed for the whole process when `STRATA_ROUTE_PRIOR` is set:**
  1. the `route_prior_k` launch is in every window graph. A closed window still pays one small launch per MoE layer
     (it returns after reading the gate);
  2. the one-launch router publishes (`STRATA_ROUTE_PUBLISH`, `STRATA_ROUTE_FAST`, `STRATA_ROUTER_FUSE`) are off for the
     process: `verify.cpp` tests `getenv("STRATA_ROUTE_PRIOR") != nullptr` (so even `STRATA_ROUTE_PRIOR=0` turns them off).
     **For the fast-hip preset this changes nothing**: it sets `STRATA_VERIFY_DEFER_CPU=1`, and deferral already turns
     `ROUTE_PUBLISH` off; `ROUTE_FAST` and `ROUTER_FUSE` are not set. For a preset without deferral, closed windows
     would still lose `ROUTE_PUBLISH`;
  3. the expert cache and the adaptive tier learn from what was routed. Windows that took the bonus shape the cache
     that the exact windows then run on.

So a two-value choice made at window boundaries (bonus on / exact) is possible today through the existing gate. A
different `lambda` per window kind would also be possible through the same float (it scales `lambda`), but nothing
here needs it. Removing the per-window cost of a closed window (item 1) or giving closed windows the publish paths
(item 2) would need two sets of window graphs, or a branch inside the router kernels. That was not done: it touches
graph capture and the router publish, and nothing measured asks for it yet.

## 2. What says "this window is copy-heavy"

The serve decode loop (`generate.cpp`, the serial loop) knows, before each window runs:

- `from_sfx`: the window's drafts are the suffix drafter's copy of earlier context (prompt lookup, `--suffix-draft`,
  on by default with 3 tokens; the A24 cross-session index too). The draft policy
  (`include/strata/spec/draft_policy.hpp`, `DraftPolicy::choose`) takes a lookup window only when its expected
  committed tokens per millisecond, from the measured acceptance per match-length bucket and the measured window costs,
  beat the MTP window's. So a lookup window already carries the policy's own estimate that the copy will be accepted.
- `sfx_match`: the match length. `T`: the window's rows (head + drafts). `chain_n`: lookup tokens chained after the
  MTP's drafts (`--lookup-chain`, off by default and off in fast-hip). `merged`: a first window that carries prompt tokens.

In the fast-hip preset (`--spec 4 --lookup-max-t 12`) MTP windows are at most 4 rows and lookup windows up to 12.

The A44 epoch bandit already uses `from_sfx || chain_n > 0` as its window type, the same signal.

## 3. The gate

    STRATA_ROUTE_PRIOR=0.5 STRATA_ROUTE_PRIOR_AUTO=1           # the bonus only in lookup windows
    STRATA_ROUTE_PRIOR_AUTO_MIN_T=5                             # optional: only lookup windows of 5 rows or more

Rule (`include/strata/program/route_prior_auto.hpp`, pure functions, unit test `tests/program/route_prior_auto_test.cpp`):

| window kind | `prior_window_kind(merged, from_sfx, chain_n, T)` | AUTO off | AUTO on |
| --- | --- | --- | --- |
| merged first window (prompt tokens) | Prompt | as today | exact |
| one row, no drafts | Plain | as today | exact |
| MTP / DFlash2 drafts | Model | as today | exact |
| MTP drafts + chained lookup tail | Chain | as today | exact (its first rows are model drafts) |
| suffix / prompt-lookup copy, at least max(2, MIN_T) rows | Lookup | as today | **bonus** |

- `route_prior_auto_lambda(cfg, lambda, kind, T)` is the bonus a window takes: `lambda` or 0.
- `prior_gate_open(scope_open, cfg, lambda, kind, T)` is the gate's verdict. With AUTO off it returns `scope_open`
  unchanged, so a run without the switch gates exactly as before. With AUTO on it can only close a window.
- With `STRATA_ROUTE_PRIOR_SCOPE` also set, a window takes the bonus only when both say so. Note the tension: scope
  `answer` closes code fences and tool calls for quality, and those are where lookup windows fire most.
- `_PROTECT`, `_KEEP_P` and `_DEPTH` still act inside the windows that take the bonus (they are kernel arguments).
- Same limits as the scope gate: only the serial serve loop writes the float. With `--batch`, `--pipeline-windows`
  or `strata generate` the bonus stays OFF (exact everywhere) and a log line says so.
- Unlike plain turbo, prompt tokens read through verify windows are routed exactly (the gate is closed there). The
  batched prefill path never calls the route-prior kernel, with or without AUTO.
- `STRATA_ROUTE_PRIOR_STATS=1` prints, per request, "route prior auto: the bonus was on in X of Y verify windows".

Start-up log with AUTO on: the F13 guards line (scope "everywhere (the auto gate decides)") and one line
"STRATA_ROUTE_PRIOR_AUTO (MOD_PLAN 17.1, changes the output): the bonus only in lookup (suffix-draft) verify windows of
N rows or more ...".

**Default path.** With `STRATA_ROUTE_PRIOR_AUTO` and `_AUTO_MIN_T` unset, `prior_guards_init` behaves as before (it
returns at once when no guard variable is set; with other guards set, no gate is added unless a scope asks for one), and
the decode loop's verdict is `prior_gate_open(scope_open, off, ...) == scope_open`. Plain turbo
(`STRATA_ROUTE_PRIOR` alone) is untouched. This is shown by reading the code, not by a binary comparison.

## 4. Why (the numbers it rests on)

Measured 2026-11-11 by the lead (same neutral client `bench/results/2026-11-11-vs-avalon/cmp_bench.py`, same IQ2_XS
file, RX 7900 XT, run 1 discarded, warm pass kept): turbo (fast-hip + `STRATA_ROUTE_PRIOR=0.5`) against exact fast-hip:

| prompt | fast-hip | turbo | change |
| --- | --- | --- | --- |
| all decode prompts | 104.8 tok/s | 96.2 tok/s | -8% |
| Python LRU class (code, repeats itself) | 136 | 148 | +9% |
| mixture-of-experts explanation (prose) | 104.0 | 88.5 | -15% |
| 15 tips (prose list) | 97.8 | 72.9 | -25% |

Draft acceptance was the same (68.3% vs 68.2%), prompt speed the same, and turbo's slowest runs fell to 64 tok/s. The
turbo pass's raw files are not in that directory as of this writing; the numbers are from MOD_PLAN Phase 17 and the
lead's brief. An earlier screen (`bench/results/2026-11-10-test-phase/06c-turbo.summary.txt`, base `rdna3test`, n=4)
had edit +29.7% +- 53.6% and short +4.1% +- 10.6%: "no difference" at those intervals. Token forcing of turbo was
INCONCLUSIVE (`06c-turbo.quality.txt`).

So turbo seems to help copy-heavy code and to hurt prose, and a global default loses. These are few runs with wide
spreads; they motivate the experiment, they do not prove the gate will win.

What the code does NOT explain: why prose is slower with the same acceptance. Candidates, none measured: (a) the
adaptive tier learns from biased picks and keeps a worse expert set for exact-ish routing; (b) the per-layer launch;
(c) noise (one warm pass per prompt, pancake's spread on prose was sd ~17 tok/s). If (c), the gate buys nothing on
prose. If (a), the gate removes only part of it, because the lookup windows still feed the tier.

## 5. Risks

- **Quality.** The bonus now lands mostly in code edits, quoted input and tool-call JSON, the places F13's scope
  `answer` closes for safety. A lookup draft is a copy the verifier checks; where the biased verifier disagrees, its
  own pick is emitted. `--grammar on` (in fast-hip) keeps tool-call JSON valid, not its content. Measure AUTO with and
  without `_PROTECT=2 _KEEP_P=0.5`, and with `_DEPTH` (row 0 of each window exact).
- **Which experts load.** The route prior changes which experts run, so the cache state and the adaptive tier shift.
  That can move acceptance and the speed of later, exact windows. Exact windows under AUTO are routed exactly but are
  not the exact preset's windows.
- **Text identity cannot judge quality.** The exact engine is already not run-to-run deterministic (0 of 6 outputs
  identical exact vs exact, MOD_PLAN Phase 17). The right checks are token forcing (`tools/pancake/quality_gate.py`,
  which runs `tools/force_check.py score` with its default `--path decode`; keep it: the gate reads decode windows and
  is closed while a prompt is read, and under forcing the lookup windows still follow the reference text, so AUTO's
  real exposure is measured) and the task gate
  (`tools/pancake/task_gate.py`, `--reps 5`, includes tool-call tasks).
- **A copy-heavy reply is not always a lookup reply.** The LRU prompt has no source to copy; its lookup windows come
  from the reply repeating itself. The part of turbo's +9% that came from MTP windows is lost under AUTO.
- **The gate itself is unmeasured**: one host-memory read per row per MoE layer per window, in open and closed windows.
- Not checked on a multi-GPU split or with DFlash2 drafting; the gate is shared by all stages (portable mapped memory),
  as for the scope gate.

## 6. Measurement plan

Preconditions: build (HIP), `ctest -R 'route_prior_auto_test|route_prior_scope_test'`, then one short serve run with
`STRATA_ROUTE_PRIOR=0.5 STRATA_ROUTE_PRIOR_AUTO=1 STRATA_ROUTE_PRIOR_STATS=1` and check the two start-up lines and that
the per-request line shows the bonus on in most windows of an edit and few of a prose reply.

Speed: spec `bench/specs/p17-route-prior-auto.json` on the fast-hip base, paired block harness, per content class:

    python3 tools/pancake/abtest.py bench/specs/p17-route-prior-auto.json --block 3 --warmup 1 --outlier robust --reclaim-max 3

Arms: `base` (exact fast-hip), `turbo` (0.5 everywhere), `auto` (0.5 in lookup windows), `auto-min5` (only lookup
windows wider than any MTP window), `auto-closed` (`_AUTO_MIN_T=256`: the gate never opens, exact routing with all of
turbo's plumbing: this arm measures the cost of items 1-2 in section 1 alone). Cases: `edit` (copy-heavy) and `short`
(mostly prose, one code example). abtest has no pure-prose case; repeat the prose check with the 11-11 client
(`cmp_bench.py`, prompts "15 tips" and "MoE explanation") on `base` and `auto`, run twice per server, second pass kept.

Reading it: promotion bar >= 2% end to end (Phase 17 rule; `promote.py` asks a 95% lower bound of 3% for a lossy item).
AUTO is worth keeping only if `edit` is >= 2% over `base` and `short` / prose are not below `base` beyond the interval.
`auto-closed` against `base` must be within noise; if it is not, the plumbing costs more than the reading of the code
suggests, and section 1's items 1-2 become the work.

Quality, only for an arm that passes speed: `quality_gate.py` (token forcing on the decode path) and `task_gate.py
--reps 5`, against `base`. A lossy arm needs both PASS before it can be a preset.

Follow-ups, only if AUTO wins: a stronger `lambda` inside lookup windows (the gate float can scale it per window kind),
and the route prior as an A44 bandit switch (same window type).
