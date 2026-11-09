"""selftune.py - pancake-strata closed-loop self-tuning from the owner's own sessions (MOD_PLAN item 70, phase 14).

One pass of the loop, offline (no GPU):
  1. traces: window traces recorded during normal use (STRATA_WINDOW_TRACE=<file>, item 64) and/or --dump-routing files
  2. a candidate expert profile ranked by how often each (layer, expert) was routed in them, blended with the current
     profile (--blend: weight of the current ranking, so one odd session cannot flip it)
  3. the time model (tools/pancake/time_ceiling.py, serial and overlap) prices current vs candidate profile and the
     tier policies (preset tier, k-th-miss admission) on the same traces - the decision rule of item 14: only a
     candidate that saves >= --min-gain % of the modeled window goes on
  4. it writes the candidate profile and an A/B spec for tools/pancake/abtest.py (+ quality_gate.py is not needed:
     placement only), so the test phase measures it on the real engine before anything changes in a preset.

    python tools/pancake/selftune.py --traces a.stwt b.stwt [--routing old.bin] --profile data/expert-profile-pancake.bin \
        --slots 10909 --out bench/selftune/<day> [--cpu-ms 17.4 --window-ms 37.4] [--blend 0.3] [--min-gain 2]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "trace"))
sys.path.insert(0, str(ROOT / "tools"))
import make_profile as mp  # noqa: E402
import replay_cache_policy as rp  # noqa: E402
import time_ceiling as tc  # noqa: E402
import wtrace  # noqa: E402


def load_windows(traces, routings):
    wins = []
    for p in traces:
        wins += wtrace.to_replay_windows(wtrace.read(p))
    for p in routings:
        wins += wtrace.to_replay_windows(wtrace.from_dump_routing(p))
    return [w for w in wins if len(w) == rp.NL]


def candidate_ranking(windows, current, blend):
    freq = defaultdict(float)
    for w in windows:
        for l, ids in w:
            for e in ids.ravel().tolist():
                freq[(l, int(e))] += 1.0
    total = sum(freq.values()) or 1.0
    n = len(current)
    score = defaultdict(float)
    for (l, e), c in freq.items():
        score[(l, e)] += (1 - blend) * c / total
    for i, (l, e) in enumerate(current):   # the current ranking as a prior: linear weight by rank
        score[(int(l), int(e))] += blend * 2.0 * (n - i) / (n * (n + 1))
    return sorted(score, key=lambda x: -score[x])


def model(windows, init, a, label):
    """Mean modeled window ms (serial, defer) of the preset tier and k-th-miss 2/8 for one starting profile."""
    budgets = np.bincount(init[:, 0], minlength=rp.NL)
    inits = [[int(e) for ll, e in init if ll == l] for l in range(rp.NL)]
    seqs, uses = tc.per_layer(windows)
    runs = [(f"{label} tier", *tc.tier(windows, init, rp.Policy(every=1, swaps=32, decay=0.7))),
            (f"{label} kmiss 2/8", *tc.replay(seqs, uses, budgets, inits, "kmiss", kmiss=(2, 8)))]
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="*", default=[])
    ap.add_argument("--routing", nargs="*", default=[])
    ap.add_argument("--profile", required=True)
    ap.add_argument("--slots", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu-ms", type=float, default=17.4)
    ap.add_argument("--window-ms", type=float, default=37.4)
    ap.add_argument("--gpu-pre-ms", type=float, default=11.5)
    ap.add_argument("--gpu-exp-ms", type=float, default=4.3)
    ap.add_argument("--draft-ms", type=float, default=2.5)
    ap.add_argument("--blend", type=float, default=0.3)
    ap.add_argument("--min-gain", type=float, default=2.0)
    ap.add_argument("--base-config", default="pancake-flashnext-iq2_xs-rdna3test.json")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    windows = load_windows(a.traces, a.routing)
    if len(windows) < 20:
        print(f"only {len(windows)} complete windows in the traces: record more before tuning")
        return 2
    half = len(windows) // 2   # learn on the first half, judge on the second (no in-sample win)
    learn, judge = windows[:half], windows[half:]
    current = mp.read_profile(a.profile)
    ranked = candidate_ranking(learn, current, a.blend)
    cand_path = out / "candidate-profile.bin"
    mp.write_profile(cand_path, ranked)
    cur_init = rp.load_profile(a.profile, a.slots)
    cand_init = rp.load_profile(str(cand_path), a.slots)

    runs = model(judge, cur_init, a, "current") + model(judge, cand_init, a, "candidate")
    nl = rp.NL
    g_pre, g_exp, sync = a.gpu_pre_ms / nl, a.gpu_exp_ms / nl, 0.04
    base_ml = runs[0][4]
    lo, hi = 0.0, 1.0
    for _ in range(50):   # calibrate ms per missed expert on the current preset tier, as time_ceiling --overlap does
        c = (lo + hi) / 2
        w = np.mean([tc.window_ms(r, c, g_pre, g_exp, sync, a.draft_ms, "serial", 0.5) for r in base_ml])
        lo, hi = (c, hi) if w < a.window_ms else (lo, c)
    c = (lo + hi) / 2
    rows = []
    for name, share, miss, promo, ml in runs:
        ser = float(np.mean([tc.window_ms(r, c, g_pre, g_exp, sync, a.draft_ms, "serial", 0.5) for r in ml]))
        dfr = float(np.mean([tc.window_ms(r, c, g_pre, g_exp, 0.0, a.draft_ms, "defer", 0.5) for r in ml]))
        rows.append({"policy": name, "hit": round(float(share), 4), "serial_ms": round(ser, 2), "defer_ms": round(dfr, 2)})
    base = rows[0]["serial_ms"]
    print(f"{len(windows)} windows (learn {len(learn)}, judge {len(judge)}); {c * 1e3:.1f} us per missed expert")
    for r in rows:
        r["gain_pct"] = round((base / r["serial_ms"] - 1) * 100, 2)
        print(f"  {r['policy']:22s} hit {r['hit']:.1%}  serial {r['serial_ms']:6.2f} ms ({r['gain_pct']:+.1f}%)  defer {r['defer_ms']:6.2f} ms")
    best = max(rows[1:], key=lambda r: r["gain_pct"])
    report = {"windows": len(windows), "rows": rows, "best": best, "min_gain": a.min_gain}
    arms = {"current": {}}
    if best["gain_pct"] >= a.min_gain:
        arm = {"env": {}}
        if best["policy"].startswith("candidate"):
            arm["args"] = {"--expert-profile": str(cand_path.resolve())}
        if "kmiss" in best["policy"]:
            arm["env"]["STRATA_ADAPT_KMISS"] = "2,8"
            arm.setdefault("args", {})["--adapt-async"] = "1"
        arms["selftune"] = arm
        print(f"candidate goes to the A/B: {best['policy']} ({best['gain_pct']:+.1f}% modeled)")
    else:
        print(f"no candidate above {a.min_gain}% modeled: nothing to A/B")
    spec = {"base": a.base_config, "arms": arms, "cases": ["short", "edit", "long"], "reps": 3}
    (out / "abtest-spec.json").write_text(json.dumps(spec, indent=1))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(f"written: {cand_path}, {out / 'abtest-spec.json'}, {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
