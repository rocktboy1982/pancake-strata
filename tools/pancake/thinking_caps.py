#!/usr/bin/env python3
"""thinking_caps.py - MOD_PLAN A17 (and the data side of A20): suggested per-effort thinking caps from the per-turn log.

    python tools/pancake/thinking_caps.py [pancake-turns.jsonl ...] [--min-max-new 8192] [--margin 1.25] [--round 128]
                                          [--min-turns 200] [--grace 512] [--json]

Reads the turn log (item 73; one JSON line per request, numbers only) and, per thinking level (the `effort` key the server
writes since 2026-11-10: low, medium, high, ... ; lines without it are counted as "unknown" and left out of the caps),
prints the distribution of the thinking tokens (p50 / p90 / p99 / max) and a suggested cap:

    cap = ceil(p99 * margin / round) * round         (default margin 1.25, rounded up to a multiple of 128)

so a cap only ever touches the tail above the p99 of what that level really does (s1 budget forcing, arXiv 2501.19393:
a hard cap with the model's own close keeps the answer and cuts the long tails; the grace, `reasoning_budget_grace`, lets
the server finish the sentence).  It also prints what each suggested cap would have cut in the log you gave it (turns hit,
thinking tokens cut, seconds at the turn's own decode speed), and the rumination / entropy figures the log holds
(`rumination_max`, `entropy_mean`: thresholds for "rumination_close" come from these, A20).

Only rows with a big reply cap count by default (--min-max-new 8192: opencode's 32768; the benchmark's 512-token rows and
ad-hoc tests are left out).  A level with fewer than --min-turns thinking turns gets its cap marked PROVISIONAL: with n
turns the p99 is only the maximum when n < 100, so do not apply a provisional cap.

The output ends with the config block to paste (the server reads it as "reasoning_budget_by_effort"; level words are
the ones clients send, the server maps high / xhigh / max onto one template level, so list `high` once):

    "reasoning_budget_by_effort": {"low": 1024, "medium": 1536, "high": 3072}, "reasoning_budget_grace": 512

LOSSY when applied (a turn that thinks past its cap is cut): the tool itself only reads the log.  Opt-in, default off.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

DEFAULT_LOG = Path(__file__).resolve().parents[2] / "pancake-turns.jsonl"


def percentile(xs: list[float], p: float) -> float:
    """Linear-interpolated percentile (the same as numpy's default), 0 for an empty list."""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def suggest_cap(p99: float, margin: float = 1.25, rnd: int = 128) -> int:
    return max(rnd, int(math.ceil(p99 * margin / rnd)) * rnd)


def load_rows(paths) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except ValueError:
                    continue                      # a cut line (the log is appended while the server runs)
                if isinstance(r, dict):
                    rows.append(r)
    return rows


def think_seconds(r: dict, tokens: float) -> float:
    """Seconds `tokens` thinking tokens take at this turn's own decode speed (decode_ms / engine_generated)."""
    g = r.get("engine_generated") or 0
    ms = r.get("decode_ms") or 0
    return tokens / g * ms / 1e3 if g > 0 and ms > 0 else 0.0


def analyse(rows, min_max_new=8192, margin=1.25, rnd=128, min_turns=200, grace=512) -> dict:
    """{"levels": {level: {...}}, "unknown": n, "turns": n, "config": {...}} from the log rows."""
    agent = [r for r in rows if isinstance(r.get("max_new"), (int, float)) and r["max_new"] >= min_max_new]
    unknown = sum(1 for r in agent if "effort" not in r)
    by: dict[str, list[dict]] = {}
    for r in agent:
        lvl = r.get("effort")
        if lvl and (r.get("reasoning_tokens") or 0) > 0:
            by.setdefault(str(lvl), []).append(r)
    levels = {}
    for lvl, rs in sorted(by.items()):
        toks = [float(r["reasoning_tokens"]) for r in rs]
        p99 = percentile(toks, 99)
        cap = suggest_cap(p99, margin, rnd)
        hit = [r for r in rs if r["reasoning_tokens"] > cap + grace]
        cut = sum(r["reasoning_tokens"] - (cap + grace) for r in hit)
        secs = sum(think_seconds(r, r["reasoning_tokens"] - (cap + grace)) for r in hit)
        total = sum(float(r.get("duration_s") or 0) for r in rs)
        lv = {"n": len(rs), "p50": percentile(toks, 50), "p90": percentile(toks, 90), "p99": p99, "max": max(toks),
              "cap": cap, "provisional": len(rs) < min_turns, "turns_hit": len(hit), "tokens_cut": int(cut),
              "seconds_saved": round(secs, 1), "turn_seconds": round(total, 1)}
        rum = [float(r["rumination_max"]) for r in rs if r.get("rumination_max") is not None]
        if rum:
            lv["rumination_max"] = {"n": len(rum), "p50": percentile(rum, 50), "p90": percentile(rum, 90),
                                    "p99": percentile(rum, 99), "max": max(rum)}
        ent = [float(r["entropy_mean"]) for r in rs if r.get("entropy_mean") is not None]
        if ent:
            lv["entropy_mean"] = {"n": len(ent), "p01": percentile(ent, 1), "p10": percentile(ent, 10),
                                  "p50": percentile(ent, 50)}
        levels[lvl] = lv
    config = {k: v["cap"] for k, v in levels.items() if not v["provisional"]}
    return {"turns": len(agent), "unknown": unknown, "levels": levels, "config": config,
            "grace": grace, "margin": margin, "min_turns": min_turns}


def report(a: dict) -> str:
    out = [f"agent turns (reply cap >= the --min-max-new): {a['turns']}; without an effort key (older lines): {a['unknown']}"]
    if not a["levels"]:
        out.append("no thinking turn carries an effort level yet: nothing to suggest (collect turns with the server of "
                   "2026-11-10 or newer)")
    out.append(f"{'level':8s} {'n':>5s} {'p50':>7s} {'p90':>7s} {'p99':>7s} {'max':>7s} {'cap':>6s} {'hit':>4s} "
               f"{'cut tok':>8s} {'saved s':>8s} {'of turn s':>9s}")
    for lvl, v in a["levels"].items():
        out.append(f"{lvl:8s} {v['n']:5d} {v['p50']:7.0f} {v['p90']:7.0f} {v['p99']:7.0f} {v['max']:7.0f} {v['cap']:6d} "
                   f"{v['turns_hit']:4d} {v['tokens_cut']:8d} {v['seconds_saved']:8.1f} {v['turn_seconds']:9.1f}"
                   + ("  PROVISIONAL (n < %d)" % a["min_turns"] if v["provisional"] else ""))
    for lvl, v in a["levels"].items():
        if "rumination_max" in v:
            r = v["rumination_max"]
            out.append(f"  {lvl}: rumination_max n={r['n']} p50 {r['p50']:.3f} p90 {r['p90']:.3f} p99 {r['p99']:.3f} "
                       f"max {r['max']:.3f}  (a rumination_close threshold goes above the p99)")
        if "entropy_mean" in v:
            e = v["entropy_mean"]
            out.append(f"  {lvl}: entropy_mean (nats) n={e['n']} p01 {e['p01']:.3f} p10 {e['p10']:.3f} p50 {e['p50']:.3f}  "
                       "(an entropy_below goes below the p10)")
    if a["config"]:
        out.append("config (levels with enough turns; LOSSY when applied, opt-in):")
        out.append('  "reasoning_budget_by_effort": ' + json.dumps(a["config"]) + f', "reasoning_budget_grace": {a["grace"]}')
    elif a["levels"]:
        out.append("no level has enough turns for a cap yet (all PROVISIONAL): none suggested")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Suggested per-effort thinking caps (p99 of the thinking tokens per level) "
                                             "from the per-turn log.  Read-only.")
    ap.add_argument("logs", nargs="*", help=f"turn log(s), default {DEFAULT_LOG}")
    ap.add_argument("--min-max-new", type=int, default=8192, help="count only rows with at least this reply cap (default 8192)")
    ap.add_argument("--margin", type=float, default=1.25, help="cap = p99 * margin, rounded up (default 1.25)")
    ap.add_argument("--round", type=int, default=128, dest="rnd", help="round the cap up to a multiple of this (default 128)")
    ap.add_argument("--min-turns", type=int, default=200, help="turns a level needs before its cap is not provisional (200)")
    ap.add_argument("--grace", type=int, default=512, help="reasoning_budget_grace the cut estimate assumes (default 512)")
    ap.add_argument("--json", action="store_true", help="print the analysis as JSON")
    args = ap.parse_args(argv)
    paths = args.logs or [str(DEFAULT_LOG)]
    try:
        rows = load_rows(paths)
    except OSError as e:
        print(f"thinking_caps: {e}", file=sys.stderr)
        return 2
    a = analyse(rows, args.min_max_new, args.margin, args.rnd, args.min_turns, args.grace)
    print(json.dumps(a, indent=1) if args.json else report(a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
