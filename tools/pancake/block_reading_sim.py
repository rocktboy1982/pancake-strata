"""block_reading_sim.py - which reading of a --block run may decide a PROMOTE?  False-PROMOTE and power of three readings of the same data.

A --block N run has k engine starts per arm with N passes each; the passes of a start share that start's effect.  Readings (promote.py's
rule on each: 95% lower bound of the gain >= the need, at least MIN_PAIRS units; the noise model, the slow-start rerun and the outlier rule are
power_sim.py's, fitted on bench/results/2026-11-11-gpu/*.jsonl, the --outlier robust procedure):
  start         promote.py: a start is one pair (the mean of its passes), t over k - 1 degrees of freedom
  pass          abtest's own summary: every pass a pair, t over k x N - 1 df, interval x sqrt(design effect), BLOCK_ICC as abtest assumes
  pass, t(k-1)  the pass-level sd and the same widening, but t over the k independent starts

    python tools/pancake/block_reading_sim.py [--trials 20000] [--seed 1] [--cases short,edit] [--icc 0,0.3,0.6,0.9]
        [--starts 3,4] [--passes 3,4] [--gain 3] [--need 1]

Prints, per case, ICC (the share of a pass's variance that is between starts: the weakest number of the fit, see power_sim.py), slow-start
setting (as fitted / none) and design (k x N): P(PROMOTE) % at a true gain of 0 (false PROMOTE) and at --gain.  Result and decision:
pancake-docs/batch5/promote-block.md.
"""
import argparse
import glob
import math
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abtest  # noqa: E402
import power_sim as ps  # noqa: E402
import promote  # noqa: E402

READINGS = ("start", "pass", "pass, t(k-1)")


def draw(rng, fit, trials, k, passes, gain):
    """(trials, k, passes) of one arm under the robust procedure: slow starts run again, outlier passes are NaN."""
    x, _ = ps.rerun_slow_starts(rng, fit, ps.draw_passes(rng, fit, trials, k, passes, gain), gain)
    flat = x.reshape(trials, -1)
    return np.where(ps.outlier_flags(flat), np.nan, flat).reshape(x.shape)


def promote_rates(rng, fit, k, passes, gain, need, trials, alpha=promote.ALPHA):
    """{reading: P(PROMOTE)} of k starts x passes per arm at a true gain (the other arm at 0), the same draws for every reading."""
    base, arm = draw(rng, fit, trials, k, passes, 0.0), draw(rng, fit, trials, k, passes, gain)
    deff = 1 + (passes - 1) * abtest.BLOCK_ICC
    fb, fa = base.reshape(trials, -1), arm.reshape(trials, -1)
    n = (~np.isnan(fb) & ~np.isnan(fa)).sum(axis=1)
    t_n = np.array([ps.t_quantile(int(v), alpha) if v >= 2 else 1.0 for v in n])
    with warnings.catch_warnings():   # a start whose passes were all dropped is NaN: its pair is left out
        warnings.simplefilter("ignore", RuntimeWarning)
        by_start = (np.nanmean(base, axis=2), np.nanmean(arm, axis=2))
    verdicts = {"start": ps.verdicts(*by_start, need, alpha),
                "pass": ps.verdicts(fb, fa, need, alpha, inflate=math.sqrt(deff)),
                "pass, t(k-1)": ps.verdicts(fb, fa, need, alpha, inflate=math.sqrt(deff) * ps.t_quantile(k, alpha) / t_n)
                if k >= promote.MIN_PAIRS else np.ones(trials)}   # (1 = HOLD: fewer than MIN_PAIRS independent starts)
    return {name: float((v == 0).mean()) for name, v in verdicts.items()}


def report(a):
    attempts = ps.read_attempts(sorted(glob.glob(str(ps.ROOT / ps.DEFAULT_RESULTS))))
    rng = np.random.default_rng(a.seed)
    print(f"block_reading_sim: {a.trials} trials per cell, seed {a.seed}, need {a.need:g}%, BLOCK_ICC {abtest.BLOCK_ICC:g}, "
          f"robust procedure; P(PROMOTE) % at a true gain of 0 / {a.gain:g}%")
    print(f"{'case':6s} {'icc':>4s} {'slow starts':11s} {'k x N':6s}  " + "   ".join(f"{r:>24s}" for r in READINGS))
    for case in a.cases:
        fit0 = ps.fit_case(attempts, case, "none")
        for icc in a.icc:
            for slow in (True, False):
                fit = ps.override(fit0, icc=icc, outlier_prob=None if slow else 0.0)
                for k in a.starts:
                    for m in a.passes:
                        false, power = (promote_rates(rng, fit, k, m, g, a.need, a.trials) for g in (0.0, a.gain))
                        print(f"{case:6s} {icc:4.1f} {'as fitted' if slow else 'none':11s} {k}x{m:<4d}  "
                              + "   ".join(f"{100 * false[r]:10.1f} / {100 * power[r]:<11.1f}" for r in READINGS))
    return 0


def parse(argv=None):
    floats = lambda s: [float(x) for x in s.split(",")]
    ints = lambda s: [int(x) for x in s.split(",")]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cases", type=lambda s: s.split(","), default=["short", "edit"])
    ap.add_argument("--icc", type=floats, default=[0.0, 0.3, 0.6, 0.9])
    ap.add_argument("--starts", type=ints, default=[3, 4])
    ap.add_argument("--passes", type=ints, default=[3, 4])
    ap.add_argument("--gain", type=float, default=3.0, help="true gain %% of the power column")
    ap.add_argument("--need", type=float, default=promote.MIN_GAIN[True], help="MIN_GAIN %%: 1 exact (default), 3 lossy")
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(report(parse()))
