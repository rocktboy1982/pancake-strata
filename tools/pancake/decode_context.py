"""decode_context.py - MOD_PLAN A49: how much of a decode window is attention over the cache, from an abtest.py run of
bench/specs/a49-decode-context.json (cases short, longdec10k / 32k / 64k / 128k).

Per request: ms per token = 1000 / decode tok/s at the mean context of its reply (prompt tokens + generated / 2).  A straight
line ms = a + b x context through all clean requests: b x context is the part that grows with the cache (attention over the
KV; a is expert reads, dense layers, drafting), so the attention share at a context is b x context / ms.  The rule of the
plan (A50 / A51 parked "unless attention exceeds ~15% of decode"): share at the longest context >= 15% builds A50 / A51,
< 10% closes them, between is a judgement call.  The draft acceptance of every case is printed because a case whose reply
is easier to draft is faster for a reason that is not attention (a confound the line cannot remove).

    python tools/pancake/decode_context.py bench/results/<day>-a49.jsonl [--arm base]
(Method: the Twilight paper, arXiv 2502.02770, measures batched 32-256 decode on an A100; it gives no share-versus-context
curve for batch 1, so this measurement is ours, not theirs.)
"""
import argparse
import json
import sys

BUILD, CLOSE = 15.0, 10.0   # attention share (%) at the longest context


def requests(lines, arm):
    """(case, context tokens, ms per token, draft acceptance) of every request of the clean runs of `arm`."""
    out = []
    for line in lines:
        r = json.loads(line)
        if r.get("arm") != arm or not r.get("clean"):
            continue
        for row in r["rows"]:
            out.append((row["case"], row["prompt"] + row["gen"] / 2, 1000.0 / row["decode_tps"], row.get("accept")))
    return out


def fit(points):
    """Least-squares line y = a + b x through (x, y) points: (a, b)."""
    n = len(points)
    mx, my = sum(x for x, _ in points) / n, sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x, _ in points)
    b = sum((x - mx) * (y - my) for x, y in points) / sxx
    return my - b * mx, b


def analyse(reqs):
    """{"cases": per case mean context / ms / acceptance, "a", "b", "share": {case: attention share %}, "verdict"}."""
    cases = {}
    for c, x, y, acc in reqs:
        cases.setdefault(c, []).append((x, y, acc))
    if len(cases) < 3:
        raise SystemExit("need at least three contexts (short and two longdec cases) to fit a line")
    mean = lambda v: sum(v) / len(v)   # noqa: E731
    per = {c: {"ctx": mean([x for x, _, _ in v]), "ms": mean([y for _, y, _ in v]), "n": len(v),
               "accept": mean([a for _, _, a in v if a is not None]) if any(a is not None for _, _, a in v) else None}
           for c, v in cases.items()}
    a, b = fit([(x, y) for c in cases for x, y, _ in cases[c]])
    share = {c: 100.0 * max(0.0, b * p["ctx"]) / p["ms"] for c, p in per.items()}
    top = share[max(per, key=lambda c: per[c]["ctx"])]
    verdict = "build A50 / A51" if top >= BUILD else "close A50 / A51" if top < CLOSE else "borderline: decide on the cost"
    return {"cases": per, "a": a, "b": b, "share": share, "top_share": top, "verdict": verdict}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs")
    ap.add_argument("--arm", default="base")
    a = ap.parse_args(argv)
    res = analyse(requests(open(a.runs).read().splitlines(), a.arm))
    print(f"ms per token = {res['a']:.2f} + {res['b'] * 1e3:.3f} us x context")
    for c, p in sorted(res["cases"].items(), key=lambda kv: kv[1]["ctx"]):
        acc = "n/a" if p["accept"] is None else f"{p['accept']:.2f}"
        print(f"  {c:11s} context {p['ctx']:8.0f}  {p['ms']:6.2f} ms/token ({1000 / p['ms']:6.1f} tok/s, n={p['n']})  "
              f"attention {res['share'][c]:5.1f}%  accept {acc}")
    print(f"attention share at the longest context {res['top_share']:.1f}%: {res['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
