"""time_model_bootstrap.py - how much of the A6 model's prediction is the trace, not the model.

time_ceiling.py --validate fits the model to ONE 219-window edit trace (c from the measured CPU ms, the deferred-layer cost
from the measured deferral +15.8%) and compares three out-of-sample arms with measurements (mean error 3.4 points).  This
resamples the trace in blocks of --block windows (the windows jump between segments: it resamples the workload mix), refits
c and the deferred-layer cost on every resample exactly as --validate does, and prints the spread of each prediction:
a measured interval that overlaps the spread is consistent with the model, one that does not is a model miss (no trace
noise explains it).

    python tools/pancake/time_model_bootstrap.py TRACE.bin PROFILE.bin 10909 [--boot 40] [--block 20]
Result on bench/prefetch/routing.bin (pancake-docs/batch3/selftune.md): k-th miss 2,8 +2.7% point, resampled -6.3 .. +1.9
(5-95%, so the sign is not settled by the trace); FUSE_PHASES +1.1% (+1.0 .. +1.2) against +2.5 +-1.7; rank guard +12.0% (+11.6 ..
+13.6) against +20.1 +-3.3 measured: a model miss, not trace noise; the fitted deferred-layer cost 91 us (52 .. 103 us).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import chain_model as cm  # noqa: E402
import replay_cache_policy as rp  # noqa: E402
import time_ceiling as tc  # noqa: E402

MEASURED = {"defer": (15.8, 3.9, "fitted"), "kmiss": (2.0, 1.6, "predicted"), "fuse": (2.5, 1.7, "predicted"),
            "guard": (20.1, 3.3, "predicted")}   # bench/results/2026-11-10-test-phase: gain %, 95% half-width, role in --validate
CPU_MS, BENCH_MS, HEAVY_TOP2 = 17.41, 4.70 / 116.6 * 1e3, 0.55   # the edit window the model is fitted to (time_ceiling --validate)


def predictions(windows, init, cpu_ms=CPU_MS, bench_ms=BENCH_MS, fit_gain=15.8):
    """The four --validate numbers on one set of windows: deferral (fitted), k-th miss 2,8 and FUSE_PHASES over deferral,
    rank guard TOP=2 over serial; plus the fitted deferred-layer cost (us)."""
    nl, p0 = rp.NL, cm.PRESET
    budgets = np.bincount(init[:, 0], minlength=nl)
    inits = [[int(e) for ll, e in init if ll == l] for l in range(nl)]
    base = tc.tier(windows, init, rp.Policy(every=1, swaps=32, decay=0.7))
    seqs, uses = tc.per_layer(windows)
    km = tc.replay(seqs, uses, budgets, inits, "kmiss", kmiss=(2, 8))
    p = {**p0, "c": (cpu_ms - nl * p0["ps"]) / base[1].mean()}

    def win(run, mode, heavy=cm.HEAVY, sdef=0.0, cut=0.0):
        return float(np.mean([tc.window_ms2(ml, pr, p, mode, heavy, sdef, cut) for ml, pr in zip(run[3], run[2])]))
    s0 = win(base, "serial")
    extra = bench_ms - s0
    pct = lambda ref, x: ((ref + extra) / (x + extra) - 1) * 100   # noqa: E731
    lo, hi = 0.0, 0.5
    for _ in range(30):   # the deferred-layer cost that makes deferral worth `fit_gain`
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pct(s0, win(base, "defer", sdef=mid)) > fit_gain else (lo, mid)
    sdef = (lo + hi) / 2
    d0 = win(base, "defer", sdef=sdef)
    return {"defer": pct(s0, d0), "kmiss": pct(d0, win(km, "defer", sdef=sdef)),
            "fuse": pct(d0, win(base, "defer", sdef=sdef, cut=p0["ps"] / 2)),
            "guard": pct(s0, win(base, "defer", HEAVY_TOP2, sdef)), "sdef_us": sdef * 1e3}


def bootstrap(windows, init, boot, block, seed=7, **kw):
    """`boot` predictions on block-resampled windows (same length as the trace)."""
    rng = np.random.default_rng(seed)
    n, out = len(windows), []
    for _ in range(boot):
        ws = []
        while len(ws) < n:
            s = int(rng.integers(0, max(1, n - block)))
            ws += windows[s:s + block]
        out.append(predictions(ws[:n], init, **kw))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trace")
    ap.add_argument("profile")
    ap.add_argument("slots", type=int)
    ap.add_argument("--boot", type=int, default=40)
    ap.add_argument("--block", type=int, default=20)
    a = ap.parse_args(argv)
    windows, _ = rp.load_trace(a.trace)
    init = rp.load_profile(a.profile, a.slots)
    point = predictions(windows, init)
    res = bootstrap(windows, init, a.boot, a.block)
    print(f"{len(windows)} windows, {a.boot} resamples of blocks of {a.block}")
    print(f"{'arm':8s} {'point':>7s} {'resampled 5-95%':>20s}   {'measured':>14s}  intervals overlap?")
    for k, (m, h, role) in MEASURED.items():
        v = np.array([r[k] for r in res])
        lo, hi = np.percentile(v, 5), np.percentile(v, 95)
        print(f"{k:8s} {point[k]:+7.1f} {lo:+9.1f} .. {hi:+7.1f}   {m:+6.1f} +-{h:4.1f}  "
              f"{'yes' if m + h >= lo and m - h <= hi else 'NO'} ({role})")
    v = np.array([r["sdef_us"] for r in res])
    print(f"deferred-layer cost fitted to the deferral gain: {point['sdef_us']:.0f} us ({np.percentile(v, 5):.0f} .. {np.percentile(v, 95):.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
