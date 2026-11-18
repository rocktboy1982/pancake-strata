"""power_calc.py - how many starts per arm a confirm needs to resolve the standing bars, from the run-to-run spread of past confirms.

    python3 tools/pancake/power_calc.py FILE.jsonl|DIR ... [--bar 2] [--lossy-bar 3] [--json OUT]

Reads only.  For every confirm-tier abtest jsonl (one row per engine start) it takes, per arm and case, the deviation of a start's pass-mean
from the arm's mean over its starts (so real arm effects are NOT in it), and pools the deviations of every file by case family (short, edit / pep8, long* prompt tok/s, longdec* decode tok/s).  Per family it prints
the pooled SD of one paired difference (% of base) and the number of rounds (= starts per arm) n for which the 95% interval
half-width t(n-1) * SD / sqrt(n) falls to the bar (a difference at the bar is then resolved from 0) and to half the bar.
The SD of a paired difference is sqrt(2) x the pooled start-to-start SD; the start-to-start variance is in it, because one start is the unit.
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys

if not hasattr(os, "sysconf"):
    os.sysconf = lambda name: 100
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import abtest as A  # noqa: E402

T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 15: 2.13, 20: 2.09, 30: 2.04}


def tval(df):
    keys = sorted(T975)
    for k in keys:
        if df <= k:
            return T975[k]
    return 1.96


def family(case):
    if case in ("short",):
        return "short"
    if case in ("edit", "pep8"):
        return "edit"
    if case in A.LONG_DEC:
        return "longdec"
    return "long-prompt" if case.startswith("long") else case


def load(path):
    out = []
    with open(path, errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if ln.startswith("{"):
                try:
                    d = json.loads(ln)
                except ValueError:
                    continue
                if "arm" in d and "rows" in d:
                    out.append(d)
    return out


def start_residuals(recs):
    """{case: [(relative deviation of a start's mean from its arm's mean over the starts, in %) x sqrt(n / (n - 1))]} of one file:
    the start-to-start spread of ONE arm, which is the noise of a paired difference (true arm effects are not in it, unlike a
    pooled arm - base difference)."""
    sel = A.select(recs)[0]
    out = {}
    for arm, rs in sel.items():
        vals = {}
        for r in rs:
            for case, v in A.start_means(r).items():
                vals.setdefault(case, []).append(v)
        for case, xs in vals.items():
            n = len(xs)
            if n < 2:
                continue
            m = sum(xs) / n
            if m <= 0:
                continue
            out.setdefault(case, []).extend((x - m) / m * 100 * math.sqrt(n / (n - 1)) for x in xs)
    return out


def rounds_needed(sd, target, nmax=200):
    for n in range(2, nmax):
        if tval(n - 1) * sd / math.sqrt(n) <= target:
            return n
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--bar", type=float, default=2.0)
    ap.add_argument("--lossy-bar", type=float, default=3.0)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True)) if os.path.isdir(p) else [p]
    pool, nfiles = {}, 0
    for f in files:
        try:
            recs = load(f)
        except OSError:
            continue
        if not recs or not any(r.get("opts", {}).get("test_tier") == "confirm" for r in recs):
            continue
        d = start_residuals(recs)
        if d:
            nfiles += 1
        for case, xs in d.items():
            pool.setdefault(family(case), []).extend(xs)
    rep = {}
    print(f"{nfiles} confirm files; start-to-start spread of one arm, pooled by case family; SD % = of one paired (arm - base) start difference")
    print(f"{'family':12} {'n':>5} {'slow':>4} {'SD %':>6} {'rounds for ' + str(a.bar) + '%':>14} {'for ' + str(a.lossy_bar) + '%':>9} {'for half bar':>13}")
    for fam, xs in sorted(pool.items()):
        if len(xs) < 3:
            continue
        keep = [x for x in xs if abs(x) <= A.SLOW_START * 100]   # slow engine starts (abtest.SLOW_START, 8.75%) are repeated by --outlier robust
        slow = len(xs) - len(keep)
        sd = math.sqrt(2) * math.sqrt(sum(x * x for x in keep) / len(keep))   # SD of one paired (arm - base) difference of two starts
        r = {"n": len(xs), "slow": slow, "sd": sd, "rounds_bar": rounds_needed(sd, a.bar), "rounds_lossy": rounds_needed(sd, a.lossy_bar),
             "rounds_half": rounds_needed(sd, a.bar / 2)}
        rep[fam] = r
        print(f"{fam:12} {len(xs):5d} {slow:4d} {sd:6.2f} {str(r['rounds_bar']):>14} {str(r['rounds_lossy']):>9} {str(r['rounds_half']):>13}")
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
