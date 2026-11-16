#!/usr/bin/env python3
"""turn_budget.py - pancake-strata MOD_PLAN.md section 6 items 23 (warm-turn pipeline) and 19 (what a hit point is worth): where
the time of the REAL served turns goes, from the turn log the server writes (`pancake-turns.jsonl`, one row per request).  Offline,
no GPU, stdlib only.

    python tools/pancake/turn_budget.py pancake-turns.jsonl [--class agent|bench|all] [--agent-max-new 4096] [--json OUT.json]

The log mixes the owner's agent sessions with the benchmark requests of abtest / ttft_bench, and it has no field that says which is
which.  The classes are therefore a stated heuristic, printed with the counts: `agent` = max_new >= --agent-max-new (an agent
client sends a large cap; abtest sends 128 / 256 / 512 / 1024) or a sampling temperature other than 0 (abtest is greedy);
`bench` = everything else.  Read the counts before the numbers: a class with few turns says little.

Per class it prints: counts and the span; the sum of prompt / decode / other seconds (other = duration - prompt - decode, the part
no engine timer owns: parks, restores, queueing, the HTTP side); quantiles of prompt tokens, reused share, prompt_ms, first token,
decode tok/s and the tier hit rate; the time to first token against the tokens actually read (ms per 1k read tokens, a least-squares
slope over turns that read >= 1k tokens); the decode time per token against the hit rate (ms per token per hit point, a least-squares
slope over turns with >= 64 output tokens: what one more point of resident hits is worth in this traffic); and the idle gaps between
consecutive turns (start of a turn minus the end of the one before; `time` is taken as the end of the request), the free time in
which a tool runs and the tier could be staged (item 23).
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

BENCH_MAX_NEW = 4096


def load(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return sorted(rows, key=lambda r: r.get("time") or 0)


def is_agent(r, agent_max_new=BENCH_MAX_NEW):
    return (r.get("max_new") or 0) >= agent_max_new or (r.get("temperature") not in (None, 0, 0.0))


def quantiles(xs, ps=(0.1, 0.5, 0.9)):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return [xs[int(p * (len(xs) - 1))] for p in ps] + [sum(xs) / len(xs)]


def slope(xs, ys):
    """Least-squares slope of ys on xs and the number of points (None when fewer than 3 or no spread)."""
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pts) < 3:
        return None, len(pts)
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    vx = sum((p[0] - mx) ** 2 for p in pts)
    if vx <= 0:
        return None, len(pts)
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / vx, len(pts)


def gaps(rows):
    """Seconds between the end of a turn and the start of the next one (start = time - duration_s); negative = overlap, skipped."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if a.get("time") is None or b.get("time") is None:
            continue
        g = (b["time"] - (b.get("duration_s") or 0)) - a["time"]
        if g >= 0:
            out.append(g)
    return out


def analyse(rows):
    """The report dict of one class of turns (rows in time order)."""
    n = len(rows)
    rep = {"turns": n}
    if not n:
        return rep
    t = [r["time"] for r in rows if r.get("time") is not None]
    rep["span_hours"] = (max(t) - min(t)) / 3600 if t else 0.0
    tp = sum(r.get("prompt_ms") or 0 for r in rows) / 1000
    td = sum(r.get("decode_ms") or 0 for r in rows) / 1000
    tt = sum(r.get("duration_s") or 0 for r in rows)
    rep["seconds"] = {"prompt": tp, "decode": td, "other": tt - tp - td, "duration": tt}
    rep["prompt_total"] = quantiles([r.get("prompt_total") for r in rows])
    rep["reused_share"] = quantiles([r["reused"] / r["prompt_total"] for r in rows if r.get("prompt_total") and r.get("reused") is not None])
    rep["prompt_ms"] = quantiles([r.get("prompt_ms") for r in rows])
    rep["first_token_s"] = quantiles([r.get("first_engine_token_s") for r in rows])
    dec = [r for r in rows if (r.get("output_tokens") or 0) >= 64]
    rep["decode_turns"] = len(dec)
    rep["decode_tok_s"] = quantiles([r.get("decode_tok_s") for r in dec])
    rep["hit_rate"] = quantiles([r.get("hit_rate") for r in dec])
    rd = [r for r in rows if (r.get("prompt_read") or 0) >= 1024 and r.get("prompt_ms") is not None]
    s, k = slope([r["prompt_read"] / 1000 for r in rd], [r["prompt_ms"] for r in rd])
    rep["prompt_ms_per_1k_read"] = {"slope": s, "points": k}
    d2 = [r for r in dec if r.get("hit_rate") is not None and r.get("decode_ms") is not None and r.get("engine_generated")]
    s, k = slope([r["hit_rate"] * 100 for r in d2], [r["decode_ms"] / r["engine_generated"] for r in d2])
    rep["decode_ms_per_token_per_hit_point"] = {"slope": s, "points": k}
    acc = [r["drafts_accepted"] / r["drafts_offered"] for r in dec if r.get("drafts_offered") and r.get("drafts_accepted") is not None]
    rep["draft_accept_share"] = quantiles(acc)
    g = gaps(rows)
    rep["idle_gaps_s"] = {"count": len(g), "quantiles": quantiles(g),
                          "over_5s": sum(1 for x in g if x > 5), "over_30s": sum(1 for x in g if x > 30),
                          "over_300s": sum(1 for x in g if x > 300),
                          "sum_5_to_300s_hours": sum(x for x in g if 5 < x <= 300) / 3600}
    return rep


def fmt_q(q, nd=2):
    return "n/a" if not q else "p10 %s  p50 %s  p90 %s  mean %s" % tuple(round(x, nd) for x in q)


def report(name, rep):
    lines = [f"== {name}: {rep['turns']} turns" + (f", {rep['span_hours']:.1f} h from first to last" if rep["turns"] else "")]
    if not rep["turns"]:
        return "\n".join(lines)
    s = rep["seconds"]
    lines += [
        f"  seconds: prompt {s['prompt']:.0f}  decode {s['decode']:.0f}  other {s['other']:.0f}  of {s['duration']:.0f} "
        f"({100 * s['prompt'] / max(1e-9, s['duration']):.0f}% / {100 * s['decode'] / max(1e-9, s['duration']):.0f}% / "
        f"{100 * s['other'] / max(1e-9, s['duration']):.0f}%)",
        "  prompt tokens:  " + fmt_q(rep["prompt_total"], 0),
        "  reused share:   " + fmt_q(rep["reused_share"], 3),
        "  prompt_ms:      " + fmt_q(rep["prompt_ms"], 0),
        "  first token s:  " + fmt_q(rep["first_token_s"]),
        f"  decode turns (>= 64 output tokens): {rep['decode_turns']}",
        "  decode tok/s:   " + fmt_q(rep["decode_tok_s"], 1),
        "  hit rate:       " + fmt_q(rep["hit_rate"], 3),
        "  draft accepted: " + fmt_q(rep["draft_accept_share"], 3),
    ]
    p = rep["prompt_ms_per_1k_read"]
    lines.append("  prompt_ms per 1k read tokens: " + ("n/a (%d points)" % p["points"] if p["slope"] is None else f"{p['slope']:.1f} ms ({p['points']} turns)"))
    h = rep["decode_ms_per_token_per_hit_point"]
    lines.append("  decode ms per token per hit point: " + ("n/a (%d points)" % h["points"] if h["slope"] is None else
                 f"{h['slope']:+.3f} ms ({h['points']} turns; negative = more hits, faster)"))
    g = rep["idle_gaps_s"]
    lines.append(f"  idle gaps between turns: {g['count']}; seconds " + fmt_q(g["quantiles"], 1) +
                 f"; over 5 s {g['over_5s']}, over 30 s {g['over_30s']}, over 300 s {g['over_300s']}; "
                 f"the 5-300 s gaps (a tool running) add up to {g['sum_5_to_300s_hours']:.2f} h")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log")
    ap.add_argument("--class", dest="cls", choices=("agent", "bench", "all"), default="all")
    ap.add_argument("--agent-max-new", type=int, default=BENCH_MAX_NEW)
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    rows = load(a.log)
    agent = [r for r in rows if is_agent(r, a.agent_max_new)]
    bench = [r for r in rows if not is_agent(r, a.agent_max_new)]
    classes = {"agent": agent, "bench": bench, "all": rows}
    print(f"{len(rows)} turns in {a.log}; agent-like (max_new >= {a.agent_max_new} or a non-zero temperature): {len(agent)}; the rest "
          f"(abtest / ttft_bench style): {len(bench)}")
    out = {}
    for name in (("agent", "bench", "all") if a.cls == "all" else (a.cls,)):
        out[name] = analyse(classes[name])
        print(report(name, out[name]))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
