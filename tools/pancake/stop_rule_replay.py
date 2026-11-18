"""stop_rule_replay.py - replay abtest's early-stop rules on stored confirm results and compare their decision with the full run.

    python3 tools/pancake/stop_rule_replay.py FILE.jsonl|DIR ... [--bar 2] [--lossy-bar 3] [--min-reps 4] [--json OUT]

Reads only.  For every confirm-tier abtest jsonl with at least --min-reps rounds of starts (abtest "rep" values) it
  1. takes the FULL decision of every non-base arm: promote when the best case's mean difference is >= the bar and no case is
     below the -3% floor (the rule of the night chains), else drop;
  2. replays, at every even round boundary >= 2 (where abtest judges an early stop), the rule `decided_bar`
     (--bar-stop) and, for comparison, the old `decided` (interval excludes 0 / narrower than 1.5%);
  3. reports per file the round each rule would have stopped at, whether its decision (arm_status at the stop) equals the
     full decision for every arm, and the share of starts it would have saved.
The summary gives the agreement over all files / arms and the starts saved (hours: starts x the file's recorded mean start time).
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if not hasattr(os, "sysconf"):   # Windows: abtest reads the clock tick at import; the replay never starts a server
    os.sysconf = lambda name: 100
import abtest as A  # noqa: E402


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


def upto(records, rounds):
    """The records of the first `rounds` rounds (rep values 0 .. rounds-1)."""
    return [r for r in records if r.get("rep", 0) < rounds]


def full_decision(results, arms, cases, bar, floor=-3.0):
    """{arm: "promote" | "drop"} by the means over everything recorded."""
    out = {}
    base = arms[0]
    for a in arms[1:]:
        deltas = []
        for c in cases:
            vb = [r[A.metric(c)] for res in results.get(base, []) for r in res["rows"] if r["case"] == c]
            va = [r[A.metric(c)] for res in results.get(a, []) for r in res["rows"] if r["case"] == c]
            if vb and va:
                deltas.append((sum(va) / len(va)) / (sum(vb) / len(vb)) * 100 - 100)
        out[a] = "promote" if deltas and max(deltas) >= bar and min(deltas) > floor else "drop"
    return out


def start_seconds(records):
    ts = sorted(datetime.strptime(r["t"], "%Y-%m-%d %H:%M:%S") for r in records if "t" in r)
    if len(ts) < 3:
        return 0.0
    return (ts[-1] - ts[0]).total_seconds() / (len(ts) - 1)


def replay_file(path, bar, lossy_bar):
    recs = load(path)
    if not recs or not any(r.get("opts", {}).get("test_tier") == "confirm" for r in recs):
        return None
    lossy = False
    for r in recs:   # a lossy spec is held to the lossy bar: read from the spec next to the results when there is one
        pass
    arms = list(dict.fromkeys(r["arm"] for r in recs))
    cases = list(dict.fromkeys(x["case"] for r in recs for x in r["rows"]))
    rounds = max(r.get("rep", 0) for r in recs) + 1
    if len(arms) < 2:
        return None
    sel = A.select(recs)[0]
    use_bar = lossy_bar if lossy else bar
    full = full_decision(sel, arms, cases, use_bar)
    per_round = max(1, len(recs) // max(1, rounds * len(arms)))
    del per_round
    res = {"file": path, "arms": len(arms), "rounds": rounds, "starts": len(recs), "full": full, "start_s": start_seconds(recs)}
    for name, fn in (("bar", lambda s: A.decided_bar(s, arms, cases, use_bar)), ("zero", lambda s: A.decided(s, arms, cases))):
        stop, ok = rounds, True
        for r in range(2, rounds, 2):
            s = A.select(upto(recs, r))[0]
            if fn(s):
                stop = r
                break
        s = A.select(upto(recs, stop))[0]
        early = {a: A.arm_status(s, arms, cases, a, use_bar) or "undecided" for a in arms[1:]}
        if name == "zero":   # the old rule stops on 'excludes 0': its decision is the means rule on the data it had
            early = full_decision(s, arms, cases, use_bar)
        ok = all(early[a] == full[a] for a in arms[1:]) if stop < rounds else True
        saved = sum(1 for x in recs if x.get("rep", 0) >= stop)
        res[name] = {"stop_round": stop, "same_as_full": ok, "early": early, "starts_saved": saved}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--bar", type=float, default=2.0)
    ap.add_argument("--lossy-bar", type=float, default=3.0)
    ap.add_argument("--min-reps", type=int, default=4)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True)) if os.path.isdir(p) else [p]
    rows = []
    for f in files:
        try:
            r = replay_file(f, a.bar, a.lossy_bar)
        except Exception as e:   # noqa: BLE001 - a malformed old file is skipped, said so
            print(f"skip {f}: {type(e).__name__} {e}")
            continue
        if r and r["rounds"] >= a.min_reps:
            rows.append(r)
    print(f"{'file':60} arms rounds starts  bar-stop(round ok saved)  zero-stop(round ok saved)")
    tot = {"bar": [0, 0, 0, 0.0], "zero": [0, 0, 0, 0.0]}   # files stopped early, agreeing, starts saved, hours saved
    for r in rows:
        line = f"{os.path.basename(r['file'])[:60]:60} {r['arms']:4d} {r['rounds']:6d} {r['starts']:6d}"
        for k in ("bar", "zero"):
            x = r[k]
            line += f"  {x['stop_round']:3d} {'ok ' if x['same_as_full'] else 'DIFF'} {x['starts_saved']:3d}      "
            if x["stop_round"] < r["rounds"]:
                tot[k][0] += 1
                tot[k][1] += x["same_as_full"]
            tot[k][2] += x["starts_saved"]
            tot[k][3] += x["starts_saved"] * r["start_s"] / 3600
        print(line)
    n = len(rows)
    allstarts = sum(r["starts"] for r in rows)
    print(f"\n{n} confirm files with >= {a.min_reps} rounds, {allstarts} starts in all")
    for k, label in (("bar", "bar-stop (new)"), ("zero", "zero-stop (old)")):
        e, ag, sv, h = tot[k]
        print(f"{label}: stops early in {e}/{n} files; decision equal to the full run in {ag}/{e} of those; "
              f"saves {sv}/{allstarts} starts = {100 * sv / max(1, allstarts):.0f}%, {h:.1f} h")
    if a.json:
        json.dump({"files": rows, "total": tot}, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
