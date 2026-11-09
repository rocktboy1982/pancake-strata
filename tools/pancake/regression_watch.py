"""regression_watch.py - pancake-strata regression watch (MOD_PLAN phase 14): the default preset after a build or pull,
measured on a quiet machine (tools/pancake/abtest.py's gate and noise monitor) and compared with the last good numbers.

    python tools/pancake/regression_watch.py [--config pancake-flashnext-iq2_xs-rdna3test.json] [--reps 2] [--accept]

Compares decode tok/s (short, edit) and prompt tok/s (long) with bench/results/regression-last-good.json; a drop larger
than --tolerance (default 5%) on any of them prints REGRESSION and exits 1.  --accept stores this run as the new last
good (after a deliberate change).  The first run has nothing to compare and stores itself.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOOD = ROOT / "bench/results/regression-last-good.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="pancake-flashnext-iq2_xs-rdna3test.json")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tolerance", type=float, default=5.0)
    ap.add_argument("--accept", action="store_true")
    a = ap.parse_args()
    day = time.strftime("%Y-%m-%d-%H%M")
    work = ROOT / f"bench/results/regression/{day}"
    work.mkdir(parents=True, exist_ok=True)
    spec = work / "spec.json"
    spec.write_text(json.dumps({"base": a.config, "arms": {"current": {}}, "cases": ["short", "edit", "long"], "reps": a.reps}))
    out = work / "runs.jsonl"
    subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "tools/pancake/abtest.py"), str(spec), "--out", str(out)],
                   cwd=ROOT, check=False)
    vals = {"short": [], "edit": [], "long": []}
    for line in out.read_text().splitlines():
        r = json.loads(line)
        if not r.get("clean"):
            continue
        for row in r["rows"]:
            vals[row["case"]].append(row["prompt_tps"] if row["case"] == "long" else row["decode_tps"])
    now = {k: sum(v) / len(v) for k, v in vals.items() if v}
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    rec = {"t": day, "rev": rev, "config": a.config, **{k: round(v, 1) for k, v in now.items()}}
    print("this run:", rec)
    if not GOOD.exists() or a.accept:
        GOOD.write_text(json.dumps(rec, indent=1))
        print(f"stored as last good ({GOOD})")
        return 0
    good = json.loads(GOOD.read_text())
    bad = []
    for k in ("short", "edit", "long"):
        if k in now and k in good:
            d = (now[k] / good[k] - 1) * 100
            print(f"  {k:6s} {now[k]:8.1f} vs last good {good[k]:8.1f} ({good['rev']}): {d:+.1f}%")
            if d < -a.tolerance:
                bad.append(k)
    if bad:
        print(f"REGRESSION on {', '.join(bad)} (more than {a.tolerance}% below the last good)")
        return 1
    print("no regression")
    return 0


if __name__ == "__main__":
    sys.exit(main())
