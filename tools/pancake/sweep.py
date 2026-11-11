"""sweep.py - pancake-strata MOD_PLAN item 37 / A34: coordinate descent over launch / tile / CPU block knobs, with the noise
rule of its source (Verma, Canesche, Pereira, "Enhancing the Power of Polyhedral-Based Optimizations with Coordinate-Based
Hill Climbing", UFMG, arXiv 2609.03114: three runs per point, a candidate replaces the current point only when it is
statistically faster by a Mann-Whitney U test, the search stops when none is).

One knob at a time over its candidate values (a line search per knob, not the paper's +-1 step on every coordinate at
once); a value is kept only if it is >= --min-gain % faster than the current point AND the one-sided Mann-Whitney p is
<= --alpha; passes repeat while a pass kept something.  Every run is one abtest.py server (quiet-machine gate, noisy runs
repeated, engine-log decode rates); a point is --reps runs; samples are decode tok/s per request divided by the start
point's mean for the same case, so short and edit weigh the same.  The kept point is then confirmed against the start
point with fresh runs in A B B A order (--confirm-rounds): a winner picked from ~20 noisy points is biased upward.
The result is printed and written as an env block for the preset.

    python tools/pancake/sweep.py [--knobs default] [--cases short,edit] [--reps 3] --out bench/results/<day>-sweep
Knob sets (--knobs): default = tools/pancake/sweep-knobs-fast-hip.json; or a JSON file {"ENV": ["v1", ...]} (a key starting
with "--" is an engine argument).  The start point is the base config's own value of a knob, else the first value listed
(which must be the engine's default).
Power (simulated, 3% run-to-run spread, alpha 0.05, one comparison): 6 samples per point (--reps 3 x short, edit) detect a
real 3% knob 42% of the time, 5% 85%, 8% 95%; 24 samples (--reps 12) detect 2% 65%, 3% 92%; a 1% knob is out of reach either way
(15% / 28%).  On a flat landscape the old rule (one run per point, `t > best`) left the start in 95% of sweeps and reported
+4.7% for nothing; this rule left it in a third of them and the confirmation accepted none of 30.  Cost: ~25 points x 3 servers.
"""
import argparse
import itertools
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def u_stat(a, b):
    """Mann-Whitney U of a against b: the pairs where a is higher, ties half."""
    return sum((x > y) + 0.5 * (x == y) for x in a for y in b)


def p_faster(new, old, max_perm=20000):
    """One-sided Mann-Whitney p that `new` is higher than `old`: the share of label permutations whose U is at least the
    observed one (exact when C(n+m, n) <= max_perm, else a fixed-seed sample of permutations)."""
    pool, n = list(new) + list(old), len(new)
    obs = u_stat(new, old)
    idx = range(len(pool))
    if math.comb(len(pool), n) <= max_perm:
        splits = itertools.combinations(idx, n)
    else:
        rng = random.Random(0)
        splits = (rng.sample(idx, n) for _ in range(max_perm))
    hit = total = 0
    for s in splits:
        s = set(s)
        hit += u_stat([pool[i] for i in s], [pool[i] for i in idx if i not in s]) >= obs
        total += 1
    return hit / total


def mean(xs):
    return sum(xs) / len(xs)


def faster(new, old, alpha, min_gain):
    """`new` is at least min_gain % higher on average and statistically higher (p <= alpha)."""
    return mean(new) >= mean(old) * (1 + min_gain / 100) and p_faster(new, old) <= alpha


def key(point):
    return tuple(sorted(point.items()))


def search(knobs, start, run_once, reps, alpha, min_gain):
    """Coordinate descent from `start`.  run_once(point) -> [(case, tok/s)] of one server, or None when it failed.
    Returns (best point, {point key: normalised samples}, ref) where ref is the start's mean tok/s per case."""
    ref, hist = {}, {}

    def samples(point):
        k = key(point)
        if k not in hist:
            rows = []
            for _ in range(reps):
                r = run_once(point)
                if r is None:   # a failed server (an invalid knob combination): this point is out
                    rows = None
                    break
                rows += r
            if rows is not None and not ref:   # the first point measured is the start: it sets the per-case scale
                for case in {c for c, _ in rows}:
                    ref[case] = mean([t for c, t in rows if c == case])
            hist[k] = None if rows is None else [t / ref[c] for c, t in rows if c in ref]
        return hist[k]

    point = dict(start)
    if samples(point) is None:
        return point, hist, ref
    changed = True
    while changed:
        changed = False
        for name, values in knobs.items():
            for v in values:
                if v == point[name]:
                    continue
                trial = dict(point, **{name: v})
                s = samples(trial)
                print(f"  {name}={v}: " + ("failed" if s is None else f"{mean(s):.3f} of the start (current {mean(samples(point)):.3f})"),
                      flush=True)
                if s is not None and faster(s, samples(point), alpha, min_gain):
                    point, changed = trial, True
            print(f"{name} -> {point[name]}", flush=True)
    return point, hist, ref


def confirm(start, best, run_once, ref, rounds, alpha, min_gain):
    """Fresh runs of the kept point and the start in A B B A order: (gain %, p, confirmed)."""
    new, old = [], []
    for _ in range(rounds):
        for who in ("best", "start", "start", "best"):
            r = run_once(best if who == "best" else start)
            if r is None:
                return None, None, False
            (new if who == "best" else old).extend(t / ref[c] for c, t in r if c in ref)
    return (mean(new) / mean(old) - 1) * 100, p_faster(new, old), faster(new, old, alpha, min_gain)


def start_point(knobs, cfg):
    """The base config's own value of each knob (env or engine argument), else the first value listed."""
    args = cfg.get("args", [])
    point = {}
    for k, values in knobs.items():
        if k.startswith("--"):
            own = args[args.index(k) + 1] if k in args and args.index(k) + 1 < len(args) else None
        else:
            own = cfg.get("env", {}).get(k)
        point[k] = str(own) if own is not None else values[0]
    return point


def measure_once(base, env, cases, work, tag, max_load, foreign):
    """One server of the point: [(case, tok/s)] of its requests, or None on a failure."""
    import abtest
    arm = {"env": {k: v for k, v in env.items() if not k.startswith("--")},
           "args": {k: v for k, v in env.items() if k.startswith("--")}}
    (work / f"{tag}.spec.json").write_text(json.dumps({"base": base, "arms": {tag: arm}, "cases": cases, "reps": 1}))
    for attempt in range(3):
        abtest.quiet_gate(max_load, foreign, [])
        cfg = str(work / f"{tag}.json")
        c = abtest.make_config(base, arm, cfg)
        res, err = abtest.run_arm(tag, cfg, c, cases, abtest.prompts(), str(work / f"{tag}-a{attempt}.out"), 8080)
        if err:
            print(f"  {tag}: {err}")
            return None
        if res["noise"] <= foreign and res.get("reclaim", 0) <= abtest.RECLAIM_MAX:
            return [(x["case"], x[abtest.metric(x["case"])]) for x in res["rows"]]
        print(f"  {tag}: noisy run (foreign CPU {res['noise']:.0f}%), repeated")
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--knobs", default="default")
    ap.add_argument("--cases", default="short,edit")
    ap.add_argument("--reps", type=int, default=3, help="servers per point (the source uses 3; 1 cannot be significant)")
    ap.add_argument("--min-gain", type=float, default=0.5, help="%% a value must be faster than the current point to be kept")
    ap.add_argument("--alpha", type=float, default=0.05, help="one-sided Mann-Whitney p a kept value (and the confirmation) needs")
    ap.add_argument("--confirm-rounds", type=int, default=3, help="A B B A rounds of kept point vs start; 0 skips")
    ap.add_argument("--max-load", type=float, default=0.0)
    ap.add_argument("--foreign-cpu", type=float, default=25.0)
    ap.add_argument("--base", default="pancake-flashnext-iq2_xs-fast-hip.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    import abtest
    knobs = json.load(open(HERE / "sweep-knobs-fast-hip.json" if a.knobs == "default" else a.knobs))
    cases = a.cases.split(",")
    work = Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    base = str(abtest.ROOT / a.base)
    start = start_point(knobs, json.load(open(base)))
    knobs = {k: list(dict.fromkeys([start[k]] + v)) for k, v in knobs.items()}   # the start value is a candidate, first
    abtest.stop_servers()
    n = [0]

    def run_once(point):
        n[0] += 1
        return measure_once(base, point, cases, work, f"p{n[0]}", a.max_load, a.foreign_cpu)

    print(f"start {start}")
    best, hist, ref = search(knobs, start, run_once, a.reps, a.alpha, a.min_gain)
    result = {"start": start, "best": best, "points": [{"point": dict(k), "samples": s} for k, s in hist.items()]}
    if not ref:
        print("the start point failed: nothing to sweep")
    elif best == start:
        print("no value is significantly faster than the start point")
    elif a.confirm_rounds > 0:
        gain, p, ok = confirm(start, best, run_once, ref, a.confirm_rounds, a.alpha, a.min_gain)
        result["confirm"] = {"gain_pct": gain, "p": p, "confirmed": ok}
        print("confirmation vs the start (fresh A B B A runs): " + ("a server failed" if gain is None else
              f"{gain:+.1f}% p={p:.3f} -> {'CONFIRMED' if ok else 'NOT confirmed: keep the preset'}"))
    (work / "sweep.json").write_text(json.dumps(result, indent=1))
    print("best:", " ".join(f"{k}={v}" for k, v in best.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
