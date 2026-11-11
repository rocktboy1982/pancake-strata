"""regression_watch.py - pancake-strata regression watch (MOD_PLAN phase 14): the default preset after a build or pull,
measured on a quiet machine (tools/pancake/abtest.py's gate and noise monitor) and compared with the last good numbers.

    python tools/pancake/regression_watch.py [--config pancake-flashnext-iq2_xs-fast-hip.json] [--reps 4] [--accept]

Compares decode tok/s (short, edit) and prompt tok/s (long) with bench/results/regression-last-good.json (mean and 95%
interval of the clean runs).  Per case: a drop larger than --tolerance (default 5%) AND larger than the two intervals
together is a REGRESSION (exit 1); a drop larger than the tolerance but inside the noise is SUSPECT (exit 3: run again with
more --reps); a run with no clean data for a case is exit 2 and is never stored.  --accept stores this run as the new last
good (after a deliberate change).  The first run has nothing to compare and stores itself.  Every run is also appended to
bench/results/regression-history.jsonl.

Limit: two runs hours apart on the same preset differ by more than a few percent (a base measured 116.6 tok/s in one group
of the 2026-11-10 test phase and 136.8 in another), and the interval of 4 reps is about +-5% (abtest.ci, t = 3.18): this
catches a 10% regression, not a 5% one.  The drift-free test is an interleaved A/B of the last good build against the new one
(an abtest.py spec with an arm {"exe": "<old build>/strata"}); keep the old build to run it.
"""
import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
GOOD = ROOT / "bench/results/regression-last-good.json"
HISTORY = ROOT / "bench/results/regression-history.jsonl"
CASES = ("short", "edit", "long")


def collect(lines):
    """{case: tok/s of every request} of the clean runs in an abtest.py --out file (decode tok/s; prompt tok/s for long)."""
    import abtest
    vals = {c: [] for c in CASES}
    for line in lines:
        r = json.loads(line)
        if not r.get("clean"):
            continue
        for row in r["rows"]:
            vals[row["case"]].append(row[abtest.metric(row["case"])])
    return {c: v for c, v in vals.items() if v}


def summarize(vals):
    """{case: {mean, half, n}}: half is the 95% interval half-width (None below two runs)."""
    import abtest
    out = {}
    for c, v in vals.items():
        m, h = abtest.ci(v)
        out[c] = {"mean": round(m, 2), "half": None if math.isnan(h) else round(h, 2), "n": len(v)}
    return out


def compare(now, good, tolerance):
    """Per case of the last good record: drop % (positive = slower), the two intervals together in %, and the status:
    ok, suspect (drop above the tolerance, inside the noise) or regression (above both); missing when this run has none."""
    out = {}
    for c, g in good.items():
        n = now.get(c)
        if n is None:
            out[c] = {"status": "missing"}
            continue
        drop = (g["mean"] - n["mean"]) / g["mean"] * 100
        halves = [h for h in (g["half"], n["half"]) if h is not None]
        noise = math.inf if len(halves) < 2 else math.hypot(*halves) / g["mean"] * 100
        status = "regression" if drop > tolerance and drop > noise else "suspect" if drop > tolerance else "ok"
        out[c] = {"drop_pct": round(drop, 1), "noise_pct": None if math.isinf(noise) else round(noise, 1), "status": status}
    return out


def exit_code(result):
    states = {r["status"] for r in result.values()}
    return 1 if "regression" in states else 2 if "missing" in states else 3 if "suspect" in states else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="pancake-flashnext-iq2_xs-fast-hip.json")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--accept", action="store_true")
    a = ap.parse_args(argv)
    day = time.strftime("%Y-%m-%d-%H%M")
    work = ROOT / f"bench/results/regression/{day}"
    work.mkdir(parents=True, exist_ok=True)
    spec = work / "spec.json"
    spec.write_text(json.dumps({"base": a.config, "arms": {"current": {}}, "cases": list(CASES), "reps": a.reps}))
    out = work / "runs.jsonl"
    subprocess.run([str(ROOT / ".venv/bin/python"), str(HERE / "abtest.py"), str(spec), "--out", str(out)], cwd=ROOT, check=False)
    now = summarize(collect(out.read_text().splitlines() if out.exists() else []))
    missing = [c for c in CASES if c not in now]
    if missing:
        print(f"no clean runs for {', '.join(missing)}: nothing compared, nothing stored")
        return 2
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rec = {"t": day, "rev": rev, "config": a.config, "reps": a.reps, "cases": now}
    print("this run:", json.dumps(rec))
    with open(HISTORY, "a") as f:
        f.write(json.dumps(rec) + "\n")
    if not GOOD.exists() or a.accept:
        GOOD.write_text(json.dumps(rec, indent=1))
        print(f"stored as last good ({GOOD})")
        return 0
    good = json.loads(GOOD.read_text())
    res = compare(now, good["cases"], a.tolerance)
    for c, r in res.items():
        if r["status"] == "missing":
            continue
        print(f"  {c:6s} {now[c]['mean']:8.1f} vs last good {good['cases'][c]['mean']:8.1f} ({good['rev']}): "
              f"{-r['drop_pct']:+.1f}% (noise +-{r['noise_pct']}%)  {r['status']}")
    code = exit_code(res)
    print({0: "no regression", 1: f"REGRESSION (more than {a.tolerance}% below the last good and beyond the noise)",
           2: "a case is missing", 3: f"SUSPECT: more than {a.tolerance}% below the last good but inside the noise; run again with more --reps"}[code])
    return code


if __name__ == "__main__":
    sys.exit(main())
