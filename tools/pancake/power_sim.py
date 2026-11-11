"""power_sim.py - how many engine starts does a small gain need?  Monte Carlo of promote.py's rule on the box's real noise.

Fits the noise from real abtest results (default bench/results/2026-11-11-gpu/*.jsonl) and simulates, for a true gain g, how often a
comparison of two arms ends in PROMOTE under promote.py's rule: the 95% lower bound of the paired gain (Student t over the pairs, in % of
the base mean, Bonferroni alpha/M) must reach MIN_GAIN (1% for an exact item, 3% for a lossy one) with at least MIN_PAIRS (3) pairs.
REJECT is the upper bound below MIN_GAIN, HOLD is everything else (promote.py: judge_arm / settle_gain; thresholds imported from it).

    python tools/pancake/power_sim.py [--results GLOB ...] [--cases short,edit] [--gains 0,1,2,3,5,10] [--trials 20000]
        [--old 3,6,10,20,40] [--starts 2,3,4,6] [--passes 4] [--bonferroni 1] [--seed 1]
        [--icc SHARE] [--sd-between PCT] [--sd-within PCT] [--outlier-prob P]      (sensitivity runs: replace the fitted value)

Noise model, per case (fit_case; all relative to the arm's level, so the base figure does not matter):
  start value  = (1 + N(0, sd_between)) * slow factor          one draw per engine start (start-to-start variance)
  pass value   = start value * (1 + N(0, sd_within))           one draw per measured pass (a pass = the case once)
  slow start   = with probability outlier_prob a whole start runs slow, by a factor drawn from the observed slow starts.
sd_within is the robust spread of log(second request / first request) of the same case in one start (the screen tier sends two); the
spread of single-request starts is sd_between^2 + sd_within^2.  Slow starts (more than 3 robust sds and 5% below their arm's median)
are counted apart and kept out of the core spreads.  Two procedures, each fitted on the starts it keeps:
  gate     abtest's default: a start whose foreign CPU or kernel reclaim crossed a threshold is run again, so the data are the "clean"
           starts (fit "recorded"); the discarded starts cost time (the cost column).  Old mode: n pairs = n starts per arm, one pass each.
  robust   abtest --outlier robust: no start is repeated because of a proxy, so every start is a candidate (fit "none"); then two outcome
           rules: a start more than SLOW_START (8.75%) below its arm's reference start (the faster of two starts, the median from
           three on) is run again in its slot (up to `retries` = 2 times; the cost column counts the reruns), and a pass that is a
           robust outlier among its arm's passes (|x - median| > 3.5 x max(1.4826 MAD, 2.5% of the median), at least 3 values) is
           dropped, with its pair.
Designs and analyses (what each reads of the same data):
  old      n starts per arm, one pass each, paired by repetition (abtest default).
  block    k starts per arm x m passes (abtest --block m, k = max(2, ceil(reps / m)) starts per arm).  Analyses:
           start     promote.py: one record per start, the mean of its passes is the unit, k pairs (t with k - 1 degrees of freedom):
                     passes of a start share its start effect, so they do not add pairs.
           pass      abtest's own summary: every pass a pair, the interval widened by sqrt(1 + (m - 1) x ICC) with abtest.BLOCK_ICC (0.45).
           naive     every pass a pair, no widening: what a reader that ignores starts would do.
Limits: starts are independent (pairing by repetition and ABBA drift cancellation are not credited: conservative); the engine's warm-pass
regime (a second identical `edit` request decodes ~35% faster) is not modeled, only relative noise; the data are 60 attempts and 45
usable starts per case, and the split of a pass's variance into between / within starts (--icc) is the weakest part of the fit
(the only data with two requests per start are screens, and their second request is a warm-pass regime): run --icc 0.3 / 0.6 / 0.9 and
--outlier-prob before trusting a verdict near a threshold.
"""
import argparse
import glob
import itertools
import json
import math
import statistics
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abtest  # noqa: E402   (metric(): which engine figure judges a case; the constants of the outlier rule once abtest has them)
import promote  # noqa: E402   (thresholds and the t quantile; importing it reads sysfs only)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = "bench/results/2026-11-11-gpu/*.jsonl"
GAINS = (0, 1, 2, 3, 5, 10)
MIN_GROUP = 3            # attempts of one (file, arm) needed to centre it on its median
OUT_Z, OUT_MIN_PCT = 3.0, 5.0     # slow start of the FIT: below the median by more than OUT_Z robust sds and OUT_MIN_PCT %
ROBUST_Z = getattr(abtest, "ROBUST_Z", 3.5)         # the outlier rule of abtest --outlier robust (modified z-score, Iglewicz & Hoaglin)
ROBUST_FLOOR = getattr(abtest, "ROBUST_FLOOR", 0.025)
ROBUST_MIN = getattr(abtest, "ROBUST_MIN", 3)
SLOW_START = getattr(abtest, "SLOW_START", 0.0875)  # a start this far below its arm's reference start is run again (abtest --outlier robust)
RETRIES = 2                                          # abtest --retries
BLOCK_ICC = getattr(abtest, "BLOCK_ICC", 0.45)       # abtest's assumed share of the variance that passes of one start have in common
START_S, PASS_S = 109, 13  # seconds: engine start + warm-up (122 s attempt median minus its 13 s measurement), one pass (both cases)
NEED_PCT = {"exact": promote.MIN_GAIN[True], "lossy": promote.MIN_GAIN[False]}
CAP = 60                 # largest n / k searched for 80% power


# ------------------------------------------------------------------------------------------------ the noise, from results
GATES = {   # which real attempts stand for the data a procedure keeps (the others were discarded and repeated)
    "recorded": lambda a: a["clean"],                                   # abtest's gates as run (reclaim <= 1.0%, foreign CPU <= 25%)
    "relaxed": lambda a: a.get("reclaim", 0) <= 3.0 and a.get("noise", 0) <= 25.0,   # --reclaim-max 3.0 (the plan's setting for new runs)
    "none": lambda a: True,                                             # --outlier robust: nothing is discarded by a proxy
}


def read_attempts(paths):
    """abtest jsonl files -> records, each with "src" (the file stem)."""
    out = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if line.strip():
                out.append({**json.loads(line), "src": Path(p).stem})
    return out


def pairwise_sd(groups, relative=True):
    """Robust sd of one value from the differences of every pair inside each group (no centring, so a group of 3 is not biased
    low): 1.0485 x the median |difference| (a normal's pair difference has median |d| = 0.6745 sqrt(2) sd).  relative: divide each
    difference by its group's median (a spread in % of the level)."""
    diffs = [abs(x - y) / (statistics.median(g) if relative else 1) for g in groups for x, y in itertools.combinations(g, 2)]
    return 1.0485 * statistics.median(diffs)


def case_groups(attempts, case):
    """{(file, arm): [(attempt, [judged figure of each request of the case in that start]), ...]}, attempts in file order."""
    key, out = abtest.metric(case), {}
    for a in attempts:
        vals = [r[key] for r in a["rows"] if r["case"] == case]
        if vals:
            out.setdefault((a["src"], a["arm"]), []).append((a, vals))
    return out


def fit_case(attempts, case, gates="recorded"):
    """Noise of one case from real attempts.  Start-to-start sd and the slow starts: the starts that pass `gates` (GATES), judged against
    the median of their arm in the same file (groups of >= MIN_GROUP attempts, all attempts counted for the median); the slow starts
    (more than OUT_Z robust sds and OUT_MIN_PCT % below it) are outliers, they do not inflate the core.  Sd inside a start: the
    spread of log(second / first request) within an arm, over all attempts (only the screen tier sends two requests per case)."""
    keep = GATES[gates]
    firsts, ratios, devs = [], [], []
    for g in case_groups(attempts, case).values():
        if len(g) < MIN_GROUP:
            continue
        med = statistics.median(v[0] for _, v in g)
        kept = [(v[0], v[0] / med - 1) for a, v in g if keep(a)]
        devs += [d for _, d in kept]
        if len(kept) >= 2:
            firsts.append(kept)
        ratios.append([math.log(v[1] / v[0]) for _, v in g if len(v) > 1])   # all starts: the noise inside a start is not what gates cut
    if len(devs) < 8:
        raise ValueError(f"case {case}, gates {gates}: only {len(devs)} starts in groups of >= {MIN_GROUP} per arm: not enough to fit")
    cut = -max(OUT_Z * pairwise_sd([[x for x, _ in grp] for grp in firsts]), OUT_MIN_PCT / 100)   # first pass: where "slow" begins
    total = pairwise_sd([[x for x, d in grp if d >= cut] for grp in firsts])   # second pass: the core, without the slow starts
    cut = -max(OUT_Z * total, OUT_MIN_PCT / 100)
    slow = [1 + d for d in devs if d < cut]
    ratios = [r for r in ratios if len(r) >= 2]
    within = pairwise_sd(ratios, relative=False) / math.sqrt(2) if sum(len(r) for r in ratios) >= 6 else total / 2
    within = min(within, total)   # a single-request start has both variances: sd_total is never below sd_within
    return {"case": case, "gates": gates, "starts": len(devs), "pairs": sum(len(r) for r in ratios), "sd_total_pct": total * 100,
            "sd_within_pct": within * 100, "sd_between_pct": math.sqrt(total ** 2 - within ** 2) * 100,
            "outlier_prob": len(slow) / len(devs), "outlier_factors": slow or [0.8]}


def override(fit, between=None, within=None, outlier_prob=None, icc=None):
    """A copy of fit with the sensitivity settings applied: icc = the share of a single pass's variance that is between starts (abtest
    assumes 0.3): it re-splits sd_total into sd_between = sd_total sqrt(icc) and sd_within = sd_total sqrt(1 - icc), which is what the
    data cannot tell well; the explicit sds and the slow-start probability replace the fitted values."""
    f = dict(fit)
    if icc is not None:
        f["sd_between_pct"], f["sd_within_pct"] = f["sd_total_pct"] * math.sqrt(icc), f["sd_total_pct"] * math.sqrt(1 - icc)
    for key, v in (("sd_between_pct", between), ("sd_within_pct", within), ("outlier_prob", outlier_prob)):
        if v is not None:
            f[key] = v
    return f


def discard_rates(attempts):
    """Share of attempts each gate discards (GATES), of all attempts."""
    return {g: sum(not keep(a) for a in attempts) / len(attempts) for g, keep in GATES.items()}


# ------------------------------------------------------------------------------------------------ promote.py's rule
@lru_cache(maxsize=None)
def t_quantile(n, alpha):
    """Two-sided t quantile for n pairs (promote.t_quantile, cached)."""
    return promote.t_quantile(1 - alpha / 2, n - 1)


def verdicts(base, arm, need, alpha=promote.ALPHA, min_pairs=promote.MIN_PAIRS, inflate=1.0):
    """base, arm: (trials, n) figures paired by column (NaN = no run).  Per trial 0 PROMOTE / 1 HOLD / 2 REJECT, as promote.py's
    settle_gain: lower bound >= need -> PROMOTE; upper bound < need -> REJECT; else (or too few pairs) HOLD.  inflate widens the
    interval (abtest's pass-level summary: sqrt of the design effect)."""
    ok = ~np.isnan(base) & ~np.isnan(arm)
    n = ok.sum(axis=1)
    with warnings.catch_warnings():   # a trial with no pair left is all-NaN: its statistics are NaN, and the verdict HOLD
        warnings.simplefilter("ignore", RuntimeWarning)
        bm = np.nanmean(np.where(ok, base, np.nan), axis=1)
        d = np.where(ok, (arm - base) / bm[:, None] * 100, np.nan)
        tq = np.full(len(n), np.inf)
        for k in np.unique(n[n >= 2]):
            tq[n == k] = t_quantile(int(k), alpha)
        mean, sd = np.nanmean(d, axis=1), np.nanstd(d, axis=1, ddof=1)
        half = tq * sd / np.sqrt(np.maximum(n, 1)) * inflate
    lo, hi = mean - half, mean + half
    enough = n >= min_pairs
    return np.where(enough & (lo >= need), 0, np.where(enough & (hi < need), 2, 1))


def shares(v):
    """(P(PROMOTE), P(HOLD), P(REJECT)) of a verdict array."""
    return tuple((v == i).mean() for i in range(3))


# ------------------------------------------------------------------------------------------------ the simulation
def draw_passes(rng, fit, trials, starts, passes, gain_pct):
    """(trials, starts, passes) figures of one arm (base level 100): a start effect (and a slow start) shared by its passes."""
    between, within = fit["sd_between_pct"] / 100, fit["sd_within_pct"] / 100
    start = 100 * (1 + gain_pct / 100) * (1 + rng.normal(0, between, (trials, starts)))
    slow = rng.random((trials, starts)) < fit["outlier_prob"]
    start = np.where(slow, start * rng.choice(fit["outlier_factors"], (trials, starts)), start)
    return start[:, :, None] * (1 + rng.normal(0, within, (trials, starts, passes)))


def outlier_flags(x, z=ROBUST_Z, floor=ROBUST_FLOOR, least=ROBUST_MIN):
    """abtest's robust rule on each row of x: |x - median| > z x max(1.4826 MAD, floor x median); rows of < least values: none."""
    if x.shape[1] < least:
        return np.zeros(x.shape, bool)
    med = np.median(x, axis=1, keepdims=True)
    spread = np.maximum(1.4826 * np.median(np.abs(x - med), axis=1, keepdims=True), floor * np.abs(med))
    return np.abs(x - med) > z * spread


def rerun_slow_starts(rng, fit, x, gain_pct, retries=RETRIES):
    """abtest's slow-start rule on x (trials, starts, passes): a start whose mean is more than SLOW_START below its arm's reference
    start (the faster of two starts, the median from three on) is drawn again, up to `retries` times.  Returns (x, reruns per trial)."""
    reruns = np.zeros(x.shape[0])
    if x.shape[1] < 2:
        return x, reruns
    for _ in range(retries):
        means = x.mean(axis=2)
        ref = means.max(axis=1) if x.shape[1] == 2 else np.median(means, axis=1)
        slow = means < ref[:, None] * (1 - SLOW_START)
        if not slow.any():
            break
        x = np.where(slow[:, :, None], draw_passes(rng, fit, *x.shape, gain_pct), x)
        reruns += slow.sum(axis=1)
    return x, reruns


def simulate(rng, fit, design, size, gain_pct, need, trials, passes=1, robust=False, analysis="start", alpha=promote.ALPHA):
    """(verdict array, mean engine starts per arm) of one cell.  design: old (size = n starts per arm, one pass each) or block
    (size = k starts x `passes`); robust: abtest --outlier robust (rerun slow starts, drop the passes flagged among the arm's passes);
    analysis: start | pass | naive (see the module docstring)."""
    passes = 1 if design == "old" else passes
    arms, reruns = [], 0.0
    for g in (0.0, gain_pct):
        x = draw_passes(rng, fit, trials, size, passes, g)
        if robust:
            x, r = rerun_slow_starts(rng, fit, x, g)
            reruns += r.mean() / 2
            flat = x.reshape(trials, -1)
            x = np.where(outlier_flags(flat), np.nan, flat).reshape(x.shape)
        arms.append(x)
    if analysis == "start":
        with warnings.catch_warnings():   # a start whose passes were all dropped is NaN: its pair is dropped
            warnings.simplefilter("ignore", RuntimeWarning)
            a, b = (np.nanmean(x, axis=2) for x in arms)
        return verdicts(a, b, need, alpha), size + reruns
    a, b = (x.reshape(trials, -1) for x in arms)
    return verdicts(a, b, need, alpha, inflate=math.sqrt(1 + (passes - 1) * BLOCK_ICC) if analysis == "pass" else 1.0), size + reruns


def power(rng, fit, design, size, gain_pct, need, trials, **kw):
    return shares(simulate(rng, fit, design, size, gain_pct, need, trials, **kw)[0])[0]


def min_units(rng, fit, design, gain_pct, need, trials, target=0.8, cap=CAP, **kw):
    """Smallest n (old) or k (block) with P(PROMOTE) >= target (bisection: power only grows with n), or None: within cap never, and
    never when gain <= need (the 95% lower bound clears the need only alpha/2 of the time then)."""
    if gain_pct <= need or power(rng, fit, design, cap, gain_pct, need, trials, **kw) < target:
        return None
    lo, hi = promote.MIN_PAIRS, cap
    while lo < hi:
        mid = (lo + hi) // 2
        if power(rng, fit, design, mid, gain_pct, need, trials, **kw) >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def minutes(design, size, passes=1, discard=0.0, reruns=0.0):
    """Wall minutes of one two-arm comparison: engine starts (+ the reruns the slow-start rule asks for, per arm) x (start + warm-up +
    the passes), and the starts the gates discard on top."""
    return 2 * (size + reruns) * (START_S + (1 if design == "old" else passes) * PASS_S) / (1 - discard) / 60


# ------------------------------------------------------------------------------------------------ report
def label(design, size, passes, analysis=""):
    """Column title: nN = N pairs (old), kKxM = K starts x M passes (block), +p / +n: the pass-level / naive reading."""
    return f"n{size}" if design == "old" else f"k{size}x{passes}" + {"start": "", "pass": "+p", "naive": "+n"}[analysis]


def table(rng, fit, need, a, alpha, title, cols):
    """One P(PROMOTE) table: cols = [(design, size, passes, robust, analysis)], a row per gain."""
    heads = [label(d, s, m, an) for d, s, m, _, an in cols]
    print(f"\n{title}\n  gain%  " + " ".join(f"{h:>8s}" for h in heads))
    for g in a.gains:
        row = [f"{100 * power(rng, fit, d, s, g, need, a.trials, passes=m, robust=r, analysis=an, alpha=alpha):5.1f}" for d, s, m, r, an in cols]
        print(f"  {g:5g}  " + " ".join(f"{x:>8s}" for x in row))


def report(a):
    paths = sorted(p for pat in (a.results or [str(ROOT / DEFAULT_RESULTS)]) for p in glob.glob(pat))
    attempts = read_attempts(paths)
    rng = np.random.default_rng(a.seed)
    dis = discard_rates(attempts)
    alpha = promote.ALPHA / a.bonferroni
    m = a.passes
    print(f"power_sim: {len(attempts)} real attempts from {len(paths)} files; promote.py rule: 95% lower bound (Bonferroni 1/{a.bonferroni}) >= "
          f"{NEED_PCT['exact']:g}% exact / {NEED_PCT['lossy']:g}% lossy, >= {promote.MIN_PAIRS} pairs; {a.trials} trials per cell, seed {a.seed}")
    print("attempts each gate would discard: " + ", ".join(f"{g} {x * 100:.0f}%" for g, x in dis.items())
          + f".  Block columns: k starts per arm x {m} passes; start = promote.py (k pairs), +p = abtest's pass-level summary "
          f"(interval x sqrt(1 + {m - 1} x {BLOCK_ICC:g})), +n = every pass a pair, no widening.  Robust procedure = abtest --outlier robust.")
    for case in a.cases:
        fits = {}
        for gates in ("recorded", "none"):
            f = fits[gates] = override(fit_case(attempts, case, gates), a.sd_between, a.sd_within, a.outlier_prob, a.icc)
            print(f"\n=== case {case} ({abtest.metric(case)}), gates {gates}: {f['starts']} starts, {f['pairs']} first/second pairs; sd between "
                  f"starts {f['sd_between_pct']:.2f}%, within a start {f['sd_within_pct']:.2f}% (one pass per start {f['sd_total_pct']:.2f}%); "
                  f"slow starts {f['outlier_prob'] * 100:.0f}% x {statistics.median(f['outlier_factors']):.2f}")
        for kind, need in NEED_PCT.items():
            print(f"\n----- {case}, {kind} item (need {need:g}%) -----")
            table(rng, fits["recorded"], need, a, alpha, "P(PROMOTE) %, gate procedure (old mode, the clean starts), n pairs",
                  [("old", n, 1, False, "start") for n in a.old])
            table(rng, fits["none"], need, a, alpha, "P(PROMOTE) %, robust procedure (every start kept, outlier rule), old mode, n pairs",
                  [("old", n, 1, True, "start") for n in a.old])
            table(rng, fits["none"], need, a, alpha, f"P(PROMOTE) %, robust procedure, block mode, k starts x {m} passes",
                  [("block", k, m, True, an) for an in ("start", "pass") for k in a.starts] + [("block", a.starts[-1], m, True, "naive")])
        print(f"\nminimum for 80% power, {case} (n pairs / k starts per arm; - = never within {CAP}, or the gain is not above the need);"
              f" minutes = two arms, {dis['recorded'] * 100:.0f}% discards counted for the gate procedure, the slow-start reruns for robust")
        print("  gain%  item   old gate n   old robust n   block start k   block pass-level k   minutes: old gate / old robust / block")
        for g in a.gains:
            for kind, need in NEED_PCT.items():
                if g <= need:
                    continue
                n0 = min_units(rng, fits["recorded"], "old", g, need, a.trials // 2, alpha=alpha)
                n1 = min_units(rng, fits["none"], "old", g, need, a.trials // 2, robust=True, alpha=alpha)
                k0, k1 = (min_units(rng, fits["none"], "block", g, need, a.trials // 2, passes=m, robust=True, analysis=an, alpha=alpha)
                          for an in ("start", "pass"))
                f = lambda x: "-" if x is None else str(x)

                def mins(design, size, discard=0.0, passes=1):
                    if size is None:
                        return "-"
                    kw = {"robust": True, "passes": passes} if discard == 0.0 else {}
                    reruns = simulate(rng, fits["none"], design, size, g, need, 2000, **kw)[1] - size if kw else 0.0
                    return f"{minutes(design, size, passes, discard, reruns):.0f}"
                t = (mins("old", n0, dis["recorded"]), mins("old", n1), mins("block", k0, passes=m))
                print(f"  {g:5g}  {kind:6s} {f(n0):>10s}   {f(n1):>12s}   {f(k0):>13s}   {f(k1):>18s}   {t[0]:>5s} / {t[1]:>5s} / {t[2]:>5s}")
    return 0


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", action="append", help="abtest jsonl glob (repeatable; default: the 2026-11-11 GPU results)")
    ap.add_argument("--cases", type=lambda s: s.split(","), default=["short", "edit"])
    ap.add_argument("--gains", type=lambda s: [float(x) for x in s.split(",")], default=list(GAINS))
    ap.add_argument("--old", type=lambda s: [int(x) for x in s.split(",")], default=[3, 6, 10, 20, 40])
    ap.add_argument("--starts", type=lambda s: [int(x) for x in s.split(",")], default=[2, 3, 4, 6])
    ap.add_argument("--passes", type=int, default=4, help="measured passes per start in block mode")
    ap.add_argument("--trials", type=int, default=20000)
    ap.add_argument("--bonferroni", type=int, default=1, help="M of promote.py: alpha / (arms x claim cases)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--icc", type=float, help="share (0-1) of a single pass's variance that is between starts, instead of the fit's split "
                    "(abtest assumes 0.3); the total sd stays the fitted one")
    ap.add_argument("--sd-between", type=float, help="%% sd between starts, instead of the fit")
    ap.add_argument("--sd-within", type=float, help="%% sd between passes of a start, instead of the fit")
    ap.add_argument("--outlier-prob", type=float, help="probability of a slow start, instead of the fit")
    return ap.parse_args(argv)


if __name__ == "__main__":
    sys.exit(report(parse()))
