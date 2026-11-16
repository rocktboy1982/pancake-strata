#!/usr/bin/env python3
"""turn_pipeline.py - MOD_PLAN.md section 6 item 23 (warm-turn pipeline), simulator side, offline on the served turn log.

The idea of the item: while a tool runs between two agent turns, the engine is idle.  Two things could be staged into that time:
  prefix pre-read   the tool response is read (prefilled) while the tool still runs, as far as the response is already known;
  tier re-warm      the decode tier is re-warmed (the expert copies a cold start needs) before the next turn arrives.
This tool replays the real turn log (turn_budget.py's input, `pancake-turns.jsonl`) and asks what each stage would take off the time to
first token if it could use ALL of the idle gap before the turn, no more:
    gap_t        = start of turn t minus the end of turn t-1 (turn_budget.gaps), seconds
    pre-read     tokens = min(avail x prompt_read_t, gap_t x 1000 / ms_per_token); saved ms = tokens x ms_per_token, at most prompt_ms_t
                 (avail = the share of the new tokens already known during the tool, --avail; ms_per_token = the log's own
                 least-squares prompt_ms per read token when >= 3 turns read >= 1k tokens, else --ms-per-1k)
    re-warm      the baseline pays --rewarm-ms once per turn that follows a gap >= --rewarm-gap seconds (the tier decayed); staged, it is
                 free when gap_t x 1000 >= rewarm_ms (the cost is a parameter: the log has no field for it; chain 9 / M1 measures it)
The saved ms come off the logged first_engine_token_s.  Nothing here is a measurement of a pipeline: it is the upper bound the
log's own gaps allow, and the premise check of the item (the real log has few gaps above 1 s, T-627).  A turn with no gap (the client
sent the next request at once) saves nothing.

    python tools/pancake/turn_pipeline.py pancake-turns.jsonl [--class agent|all] [--avail 0.5,1.0] [--rewarm-ms 0,200,500]
                                          [--rewarm-gap 30] [--ms-per-1k 400] [--json OUT.json]
    python -m pytest -q tools/pancake/test_turn_pipeline.py
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import turn_budget as TB  # noqa: E402


def turns_with_gaps(rows):
    """[(row, gap seconds before it or None for the first turn / an overlap)] in time order."""
    out = [(rows[0], None)] if rows else []
    for a, b in zip(rows, rows[1:]):
        g = None
        if a.get("time") is not None and b.get("time") is not None:
            g = (b["time"] - (b.get("duration_s") or 0)) - a["time"]
            g = g if g >= 0 else None
        out.append((b, g))
    return out


def ms_per_token(rows, default_per_1k):
    rd = [r for r in rows if (r.get("prompt_read") or 0) >= 1024 and r.get("prompt_ms") is not None]
    s, _ = TB.slope([r["prompt_read"] / 1000 for r in rd], [r["prompt_ms"] for r in rd])
    per_1k = s if s is not None and s > 0 else default_per_1k
    return per_1k / 1000.0, (s is not None and s > 0)


def simulate(rows, avail=0.5, rewarm_ms=0.0, rewarm_gap=30.0, ms_per_1k=400.0):
    """Per-turn first-token seconds before and after staging; returns the summary dict."""
    seq = turns_with_gaps(rows)
    mpt, fitted = ms_per_token(rows, ms_per_1k)
    before, after = [], []
    saved_read = saved_warm = 0.0
    n_read = n_warm = 0
    for r, g in seq:
        ft = r.get("first_engine_token_s")
        if ft is None:
            continue
        base = ft
        cold = rewarm_ms > 0 and g is not None and g >= rewarm_gap
        if cold:
            base += rewarm_ms / 1000.0       # the baseline pays the re-warm on this turn
        b = base
        if g is not None and g > 0:
            read = r.get("prompt_read") or 0
            tok = min(avail * read, g * 1000.0 / mpt) if mpt > 0 else 0.0
            sv = min(tok * mpt, r.get("prompt_ms") or 0.0)
            if sv > 0:
                saved_read += sv / 1000.0
                n_read += 1
            b -= sv / 1000.0
            if cold and g * 1000.0 >= rewarm_ms:
                saved_warm += rewarm_ms / 1000.0
                n_warm += 1
                b -= rewarm_ms / 1000.0
        before.append(base)
        after.append(max(0.0, b))
    q = TB.quantiles
    return {"turns": len(before), "avail": avail, "rewarm_ms": rewarm_ms, "rewarm_gap_s": rewarm_gap,
            "ms_per_read_token": mpt, "ms_per_token_fitted": fitted,
            "before": q(before), "after": q(after), "mean_before": sum(before) / len(before) if before else None,
            "mean_after": sum(after) / len(after) if after else None,
            "read_staged_turns": n_read, "read_saved_s": saved_read, "warm_staged_turns": n_warm, "warm_saved_s": saved_warm}


def report(rep):
    if not rep["turns"]:
        return "no turns with first_engine_token_s"
    b, a = rep["before"], rep["after"]
    mb, ma = rep["mean_before"], rep["mean_after"]
    return (f"  avail {rep['avail']:.2f} rewarm {rep['rewarm_ms']:g} ms (after gaps >= {rep['rewarm_gap_s']:g} s): {rep['turns']} turns, "
            f"first token s p50 {b[1]:.3f} -> {a[1]:.3f}, p90 {b[2]:.3f} -> {a[2]:.3f}, mean {mb:.3f} -> {ma:.3f} "
            f"({(ma / mb - 1) * 100 if mb else 0:+.1f}% mean); pre-read helps {rep['read_staged_turns']} turns ({rep['read_saved_s']:.1f} s), "
            f"re-warm {rep['warm_staged_turns']} turns ({rep['warm_saved_s']:.1f} s)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log")
    ap.add_argument("--class", dest="cls", choices=("agent", "all"), default="agent")
    ap.add_argument("--agent-max-new", type=int, default=TB.BENCH_MAX_NEW)
    ap.add_argument("--avail", default="0.5,1.0")
    ap.add_argument("--rewarm-ms", default="0,200,500")
    ap.add_argument("--rewarm-gap", type=float, default=30.0)
    ap.add_argument("--ms-per-1k", type=float, default=400.0)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    rows = TB.load(a.log)
    if a.cls == "agent":
        rows = [r for r in rows if TB.is_agent(r, a.agent_max_new)]
    print(f"{len(rows)} {a.cls} turns from {a.log}")
    outs = []
    for av in (float(x) for x in a.avail.split(",")):
        for rw in (float(x) for x in a.rewarm_ms.split(",")):
            rep = simulate(rows, av, rw, a.rewarm_gap, a.ms_per_1k)
            outs.append(rep)
            print(report(rep))
    if outs:
        print(f"  ms per read token {outs[0]['ms_per_read_token'] * 1000:.0f} per 1k ("
              f"{'fitted on the log' if outs[0]['ms_per_token_fitted'] else '--ms-per-1k default'}); upper bound: the whole gap is usable")
    if a.json:
        Path(a.json).write_text(json.dumps(outs, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
