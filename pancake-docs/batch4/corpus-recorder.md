# corpus-recorder: one command to record a real agent session for the offline tools (2026-11-11, branch dev/p16b/corpus-recorder)

Every offline verdict so far (FineMoE, HOBBIT, REAP, EcoSpec, A44, A46) rests on one 219-window bench trace
(`bench/prefetch/routing.bin`, 2026-11-08, four requests, **all router weights zero**). The recorder gets a real one.
Python only; no engine C++ was edited (the engine patch is a proposal, below).

## What was built

| File | What |
| --- | --- |
| `tools/pancake/record_session.py` | `plan` (build + print the server command, `--run` starts it), `check` (validate, write `manifest.json`), `convert` (the replay tools' inputs) |
| `tools/pancake/trace/recording.py` | raw window-trace reader, weights verdict, turn-log / dump / hidden-dump checks, the rewrite to `routing.bin` and a cleaned `session.stwt` |
| `tools/pancake/test_record_session.py` | 29 tests on synthetic recordings (written byte for byte as the engine writes them) |
| `pancake-docs/batch4/corpus-recorder-engine.patch` | the engine patch proposal (`git apply --check` clean on dev/phase16-academic) |

```
python tools/pancake/record_session.py plan --preset flashnext-iq2_xs-fast-hip --out DIR [--repo /home/dan/clone-of-d/src/Strata] [--hidden-dump] [--run]
python tools/pancake/record_session.py check DIR [--min-windows 200] [--min-turns 5] [--require-weights]
python tools/pancake/record_session.py convert DIR [--weights auto|valid|none] [--all-windows] [--slots 10909]
```

**plan** starts nothing unless `--run`. It writes `DIR/config.json` (the preset with run-pancake.sh's substitutions and the
recorders on) and `DIR/plan.json`, and prints the command (through `tools/pancake/capped.sh`, `STRATA_RAM_KEEP_GIB=12`, MemoryMax).
Recorders: `STRATA_WINDOW_TRACE=DIR/session.stwt` (+ `_MAX_MB` 4096), the config's `turn_log` = `DIR/turns.jsonl`, `--dump-routing
DIR/routing.bin`, `log` = `DIR/engine.log`, server-side `STRATA_AGENT_COUNTERS=1`; opt-in `--hidden-dump [every]`
(`STRATA_HIDDEN_DUMP`), `--norms`, `--publish-weights`, `--every`, `--max-mb`. The engine's variables go in the config's `env`
(it wins over the process environment, serve/server.py:4535); the server's own (`STRATA_AGENT_COUNTERS`, serve/agent_loop.py:108)
in the command's environment. A directory that holds a recording is refused (the turn log appends, server.py:5387).
Preflight (`[ok]/[WARN]/[FAIL]`; FAIL stops `--run`): exe is `strata` (strata-poly writes no window trace), files named in the
args exist, no `--no-pool`, **no routing-changing switch** (`STRATA_ROUTE_PRIOR`, `_ROUTE_RESIDENT`, `_ROUTE_TAIL_SKIP`,
`_VERIFY_ECAP`: the turbo presets bias the routed ids towards the cache, so a replay of such a trace is circular; they need
`--allow-biased-routing`), disk for the cap, MemAvailable, and whether the doorbell will publish the weights (next section).

**check** reads every file once and writes `DIR/manifest.json`: window counts (decode / merged prompt / incomplete / grouped),
rows, ms, drafts accepted, placement shares and VRAM hit rate, ids sanity, **the weights verdict per layer record**, turn starts,
sha256 and sizes; turn-log stats (turns, sessions, reuse, thinking, malformed lines); routing dump (records, windows, window
count and first 50 windows' ids against the trace); hidden-dump files; the engine log's start-up lines (opened / cap reached /
cannot write). `usable_for` says which tools the recording can feed (ids: time_ceiling / finemoe / cache_learn / replay_seeding /
selftune; router weights: reap gate part / gate_importance / FineMoE maps; placement + ms + drafts: ecospec_sim / draft_costs /
tail_replay / disco_fit; turn log: replay_turns; expert norms: reap `--act-norm`). Exit 2 on a problem (no trace, fewer decode
windows than `--min-windows`, fewer turns than `--min-turns`, trace never opened), 0 otherwise; weights that are not real are a
warning unless `--require-weights`.

**convert** writes `DIR/replay/`: `routing.bin` (the `--dump-routing` format, complete decode windows, one record per token and
layer, with the recorded weights, so `time_ceiling.py`, `finemoe_replay.py`, `cache_learn.py`, `reap_saliency.py`,
`gate_importance.py --dump-routing` read it unchanged), `session.stwt` (cleaned: layer records merged, truncated tail dropped,
only real weights kept, no record 6), `turns.jsonl` (the usable lines sorted, `replay_turns.py --turns`), `act_norm.npy` (only
with record 6), `convert.json` (incl. `cuts` for `finemoe_replay.py --cuts`: turn starts as indices into `routing.bin`'s windows),
and prints one replay command per tool. `--weights auto` keeps weights only if every exported window has real ones (all or
nothing, because `reap_saliency.stats` treats one weightless layer record as an unweighted trace and FineMoE builds maps from
`abs(w)`); `valid` keeps each record that has them; `none` drops them. A record without weights is written as zeros in
`routing.bin`, which the readers take as "none" (`from_dump_routing`).

## Source check: what the engine can record today

1. **Window trace (item 64).** `generate.cpp:1471-1475` writes, per pool call, ids, placement (`d.trace_kind`) and
   `d.plan->route_w`; `verify.cpp:3996` sets `route_w = h_w_ + tb*k` whenever the trace is on. `h_w_` is a zero-initialised mapped
   buffer (`verify.cpp:518-531`) that **only the publishing kernels write**. `doorbell_publish` and `doorbell_publish_value`
   (HIP, STRATA_DOORBELL_STORE) publish the weights, `doorbell_publish_res` and `route_publish_res` do not (the source says it:
   `kernels/elementwise.hpp:138`, `native_router.cu:238`). Which one a window uses is fixed when its
   graph is captured, `verify.cpp:2518-2525` (rpub) and `:2631-2644`:
   - default path (`doorbell_publish_res`, `:2644`): **the trace carries all-zero weights**;
   - `STRATA_VERIFY_DEFER_CPU=1` with `_BELOW` unset or > 0 (the fast presets): `heavy = dfr && defer_heavy_ > 0` (`:1628`), so every
     layer of a window captured with deferral uses `doorbell_publish` (`:2639`): real weights. `defer_ok` (`:3205`) is false for
     a batch window, `--spec-split`, `--remote-expert-opt`, a window whose experts are all in VRAM (`ar_on`), a control vector;
   - `STRATA_ROUTE_FAST=1`: real weights for 2..8-row windows (`native_router_top10_publish_fast`, `:2547`).
   So `weights` in a trace is real for the fast presets in the windows deferral covers, and zero in the exact presets. `check` does
   not trust the flag bit: a layer record is `ok` only if it is finite, non-negative, has no zero row, and is not bit-for-bit the
   previous record's weights (a stale buffer); states `ok / missing / zero / stale / bad` and the share per layer are in the manifest.
   The normalised top-k weights of a row sum to 1 (`native_router.cu`: `selected * inverse_selected_sum`); `check` reports the share
   of rows that do (information, not a validity test, so another model family is not rejected).
   **Patch proposal (engine, not applied): publish the weights whenever the trace is on**, two hunks in `verify.cpp` (after `:2520` add
   `&& !window_trace().on()` to `rpub`; at `:2639` add `|| window_trace().on()` to the `doorbell_publish` condition). The default
   path with the trace off is unchanged; with it on, `doorbell_publish` also copies every row (the `_res` variant skips a layer whose
   routed experts all look resident), so a traced session is slightly slower per window than an untraced one. `open()`
   (`generate.cpp:9615`) runs before the Verifier is configured (`:12347`) and the graphs are captured on first use, so the test in
   `record_window` sees the right value. Until it lands, record with a `-fast` preset (not `-turbo`: ROUTE_PRIOR) or `--publish-weights`
   (adds DEFER_CPU=1: the CPU rows are summed in another order, the fast presets already do).
2. **`--dump-routing`.** `generate.cpp:1480-1488` (`drive_pool_multi_impl`) writes **zeros** for the weights on purpose (the
   multi dispatch fuses them); the one-shot CLI path `drive_pool` (`:1437-1453`) writes true weights but is not the serve path. In
   the plan it is the safety net: ids in the legacy format, no conversion, independent of the window trace. `check` compares its
   first 50 windows with the trace's.
3. **Turn log (item 73).** `server.py:5368` `_write_turn_log`, called at `:7732` per finished request, config key `turn_log`
   or `STRATA_TURN_LOG`; one JSON line, numbers only (prompt_total, reused, output_tokens, reasoning_tokens, effort, drafts,
   hit_rate, ..., `agent` with STRATA_AGENT_COUNTERS=1). `replay_turns.load_turns` (`:85`) needs `prompt_total` and `time`;
   `check`/`convert` use the same rule (a test runs both readers on one file).
4. **`STRATA_HIDDEN_DUMP`.** `think_probe.hpp:21-40`: the LM head's input vector at the thinking's check points, one `hd-*.bin` +
   `.ids` per request. Input of `thinking_probe.py` (A19), **not** an expert norm. `check` validates the SHDM files.
5. **Per-expert output norms ||f_e(x)|| (REAP's second factor): recorded nowhere** (`reap_saliency.py` and `gate_importance.py --npz`
   take them from outside; `STRATA_ACT_DUMP` / `_RAW` dump the experts' *inputs*, `pool.cpp:1514`). The values exist on the host for
   the **CPU-computed entries**: `expert_pool_dispatch_multi` writes entry `i`'s unweighted output into `out + i*H`
   (`expert_source.cpp:5364`, H = 2560, `kernels/cpu/expert.hpp:36`); the GPU combine applies the weights later. Entries served from
   VRAM or over PCIe are never materialised on the host (kind 0/1), so coverage is the CPU share (about 19% of entries at the 81%
   tier hit rate, biased towards the rarely resident experts). With `STRATA_CPU_PRESUM=1` the rows are pre-summed
   (`expert_source.cpp:5437`) and nothing per entry exists. **Patch proposal:** `window_trace.hpp` gets `norms(layer, n_tok, v)` and a
   record type 6 (`u8 6, u8 layer, u8 n_tok, u8 0, n_tok*k f32`, NaN where not computed, buffered with its layer record, dropped with
   the window; old readers refuse the file exactly as they did before record 5), `generate.cpp` `drive_pool_multi_impl` fills it after the pool
   returns (`STRATA_WINDOW_TRACE_NORMS=1`, off by default: one `std::getenv` once), `wtrace.py` reads it into `Layer.norms`.
   A full-coverage calibration run (every expert on the CPU) would need the VRAM cache off; not verified, not planned.
   `recording.py` already reads record 6 and `convert` writes `act_norm.npy` (mean per (layer, expert); an expert never computed on
   the host gets its layer's mean; coverage in `convert.json`). Verified: the patched `window_trace.hpp` compiles
   (`g++ -std=c++17 -Wall -Wextra`), a C++-written trace reads back in `recording.py` and in the patched `wtrace.py`, and the stock
   `wtrace.read` refuses it. The `generate.cpp` / `verify.cpp` hunks were not compiled (HIP build, other agent's file).
6. **Reader bug in `wtrace.read` (not fixed here, `wtrace.py` is used by every tool).** A layer can have two records in a window:
   the verify window runs as two token groups when `--spec-split` is on or T > `kVerifyMaxT` (16 in the HIP build,
   `verify_kernels.hpp:25`; merged prompt windows, never a decode window in the presets). `wtrace.read` does
   `cur.layers[layer] = Layer(...)`, so the later group replaces the earlier one and the first rows are silently lost. `recording.py`
   concatenates them in row order (group 0 first, as the dump writes them). One-line fix for `wtrace.py`: concatenate when the layer is
   already in `cur.layers`. `session.stwt` from `convert` has them merged.
7. **`ms` and turn starts.** A window's `ms` is the time since the previous window's commit, so the first window of a turn also holds
   the idle time and the prompt read, and the file's first window the model load (up to minutes). `check` marks a window as a turn start
   when it is the first, when its position does not continue the previous commit (`disco_fit.py`'s rule) or when it came more than
   1500 ms after it; the manifest lists the count and compares it with the generating requests of the turn log. Cost fitters should use
   a window's `ms` only when it follows its predecessor (`disco_fit` does; `ecospec_sim.py:182`, `draft_costs.py:87` and `tail_replay.py:95` do not by their filters: a note for
   the lead, not changed here). `convert` keeps the recorded values.

## Tests run (all on this box, no GPU, no engine)

- `python -m pytest tools/pancake/test_record_session.py`: 29 passed (5.6 s). They cover the reader against `wtrace.read`, token
  groups, a file cut mid-record, the weights verdicts (real / zero / missing / stale / NaN / negative / zero row), stats and turn
  starts, the export read back by `replay_cache_policy.load_trace`, `wtrace.from_dump_routing`, `reap_saliency`, `finemoe_replay.load`
  and `gate_importance.sort_rows`, the three weights modes, a trace that is not 48 x 512, record 6 and `act_norm`, turn log vs
  `replay_turns`, dump stats and id comparison, hidden dump, `plan` (config, env, args, command, refusals, `--run` runs the server once
  through a stub and then checks, no run without `--run`), `check` (good / zero weights / problems / warnings) and `convert`.
- Real data, read only: the box's `pancake-turns.jsonl` (878 turns, 7 sessions): `turn_stats` and `replay_turns.load_turns` /
  `split_sessions` agree. `bench/prefetch/routing.bin` (219 windows, k 10): rebuilt as a window trace and exported again, the ids,
  layer and k columns are byte-identical to the original. `wtrace.py --selftest` ok. `plan` on the real presets: fast-hip is `ok`
  (deferral publishes weights), exact-hip and q2_0-fast-hip warn that the doorbell publishes none, turbo-hip fails (ROUTE_PRIOR),
  qwen36 fails (strata-poly), and the 6 model/pack files named in the fast-hip args exist.
- Not run: any engine, so whether a real session gives real weights in all layers is unknown until a smoke run; the engine patch
  hunks of `verify.cpp` / `generate.cpp`.

## Risks and limits

- The weights verdict is by content, so it cannot tell a real weight from a copy of another layer's that differs in one bit; stale buffers
  are copies, so this is exact for the failure that exists.
- The turn-start rule misses a turn only if the next request continues the previous position exactly (a byte-identical re-send) within
  1500 ms; a count mismatch with the turn log is a warning.
- `--weights auto` drops all weights if one decode window lacks them (a window without deferral); `valid` keeps the rest, but the
  tools that look at every layer record (reap `stats`) then treat the trace as unweighted. A recording of the fast preset with the patch
  has none of this.
- The routing dump and the trace both stay on the data disk (38 GB free at 96%); the plan warns if the cap does not fit.
- Only the main engine's serial serve loop writes the trace; batch slots, pipelined windows and strata-poly do not.

## Next (the lead, at night)

1. Smoke: `plan --preset flashnext-iq2_xs-fast-hip --out DIR --run`, a few minutes of real use, stop, `check DIR --min-windows 5
   --min-turns 1`. `weights ... carry real weights` tells whether the build needs the patch (or whether `defer_ok` fails for most
   windows).
2. Real session (an hour of the agent's daily work), `check`, `convert`, run the printed replay lines.
3. If the verdict needs weights in every layer or norms: the owner of `verify.cpp` / `generate.cpp` applies
   `corpus-recorder-engine.patch` (`git apply --check` is clean; 4 files) and rebuilds the HIP engine.
