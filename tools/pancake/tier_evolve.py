"""tier_evolve.py - pancake-strata MOD_PLAN.md section 6 item 19, upgrade U1: an evolutionary parameter search for the tier laws of
tools/pancake/tier_laws.py.  numpy only (no new dependency), offline, no GPU.

Search: generation 0 is a uniform random sample of the (normalised) parameter box; then a separable CMA-ES (diagonal covariance,
Ros and Hansen 2008: weighted recombination, cumulative step-size adaptation, rank-one and rank-mu diagonal updates) starts at
the weighted mean of the best half and runs --generations - 1 more generations of --population candidates.  Fixed seed (--seed).
Integer parameters are rounded, the box is clipped, evaluations are cached.

Fitness (lower is better) = cache_learn.cpu_ms per window of tier_laws.run on the FIRST HALF of the trace
+ --promo-penalty (ms per promotion per window, default 0.005) x promotions per window.  Every parameter set found is then run on
the SECOND HALF (hold-out, cold start from the profile, the phase rule is fitted on the first half) and printed next to the
fit-half number and next to the hand-set defaults on both halves; the overfit gap = (hold-out - fit-half) of the evolved set minus
the same of the defaults.  An improvement is claimed only when the hold-out cpu ms of the evolved set is below the defaults'.

Families and their parameters (box):
  tier       decay [0.3, 0.97], mingain [0, 6], protect [0, 0.95]                          (defaults 0.7, 1.5, 0)
  kmiss      k int [1, 4], W int [2, 8], protect                                           (2, 8, 0)
  breakeven  H [1, 32] log, margin [0, 1], alpha [0.1, 0.6], protect                       (8, 0.25, 0.3, 0)
  bayes      H [1, 32] log, a0 [1, 16] log, z [0.3, 2], pmax [0.2, 1], d [0.5, 0.98], protect   (16, 4, 1, 0.6, 0.9, 0)
  survival   H int [1, 16], protect                                                         (8, 0)
  ghost      delta [0.005, 0.1] log, quiet int [2, 16], k int [1, 3], W int [2, 8], protect (0.02, 8, 2, 8, 0)
  phase      K one of 4 / 8 / 16 (nearest), H [1, 16], margin [0, 1], protect               (8, 4, 0.25, 0)

    python tools/pancake/tier_evolve.py PROFILE.bin SLOTS TRACE --cuts 11,96,157 --names read4k,short,edit,long --interleave 8
           [--families tier kmiss ...] [--generations 8] [--population 12] [--seed 0] [--promo-penalty 0.005] [--json OUT]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cache_learn as cl  # noqa: E402
import replay_cache_policy as rp  # noqa: E402
import tier_laws as tl  # noqa: E402

# name, lo, hi, kind ("lin", "log", "int", "choice")
FAMILIES = {
    "tier": [("decay", 0.3, 0.97, "lin"), ("mingain", 0.0, 6.0, "lin"), ("protect", 0.0, 0.95, "lin")],
    "kmiss": [("k", 1, 4, "int"), ("W", 2, 8, "int"), ("protect", 0.0, 0.95, "lin")],
    "breakeven": [("H", 1, 32, "log"), ("margin", 0.0, 1.0, "lin"), ("alpha", 0.1, 0.6, "lin"), ("protect", 0.0, 0.95, "lin")],
    "bayes": [("H", 1, 32, "log"), ("a0", 1, 16, "log"), ("z", 0.3, 2.0, "lin"), ("pmax", 0.2, 1.0, "lin"), ("d", 0.5, 0.98, "lin"),
              ("protect", 0.0, 0.95, "lin")],
    "survival": [("H", 1, 16, "int"), ("protect", 0.0, 0.95, "lin")],
    "ghost": [("delta", 0.005, 0.1, "log"), ("quiet", 2, 16, "int"), ("k", 1, 3, "int"), ("W", 2, 8, "int"),
              ("protect", 0.0, 0.95, "lin")],
    "phase": [("K", 4, 16, "choice"), ("H", 1, 16, "lin"), ("margin", 0.0, 1.0, "lin"), ("protect", 0.0, 0.95, "lin")],
}
DEFAULTS = {
    "tier": {"decay": 0.7, "mingain": 1.5, "protect": 0.0},
    "kmiss": {"k": 2, "W": 8, "protect": 0.0},
    "breakeven": {"H": 8, "margin": 0.25, "alpha": 0.3, "protect": 0.0},
    "bayes": {"H": 16, "a0": 4, "z": 1.0, "pmax": 0.6, "d": 0.9, "protect": 0.0},
    "survival": {"H": 8, "protect": 0.0},
    "ghost": {"delta": 0.02, "quiet": 8, "k": 2, "W": 8, "protect": 0.0},
    "phase": {"K": 8, "H": 4, "margin": 0.25, "protect": 0.0},
}


def decode(family, u):
    """Normalised vector u in [0,1]^d -> parameter dict (rounded / snapped)."""
    out = {}
    for x, (name, lo, hi, kind) in zip(np.clip(u, 0.0, 1.0), FAMILIES[family]):
        if kind == "lin":
            v = lo + x * (hi - lo)
        elif kind == "log":
            v = math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo)))
        elif kind == "int":
            v = int(round(lo + x * (hi - lo)))
        else:
            v = min((4, 8, 16), key=lambda c: abs(c - (lo + x * (hi - lo))))
        out[name] = v
    return out


def encode(family, params):
    u = []
    for name, lo, hi, kind in FAMILIES[family]:
        v = params[name]
        if kind == "log":
            u.append((math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)))
        else:
            u.append((v - lo) / (hi - lo))
    return np.array(u)


def build(family, p, ctx):
    """The tier_laws policy for a parameter dict."""
    if family == "tier":
        return tl.Tier(decay=p["decay"], mingain=p["mingain"])
    if family == "kmiss":
        return tl.KMiss(int(p["k"]), min(8, max(int(p["k"]), int(p["W"]))))
    if family == "breakeven":
        return tl.BreakEven(p["H"], p["margin"], p["alpha"])
    if family == "bayes":
        return tl.Bayes(p["H"], p["a0"], p["z"], p["pmax"], p["d"], ctx["ranked"])
    if family == "survival":
        return tl.Survival(int(p["H"]))
    if family == "ghost":
        return tl.Ghost(p["delta"], int(p["quiet"]), int(p["k"]), int(p["W"]))
    if family == "phase":
        return tl.Phase(int(p["K"]), p["H"], p["margin"])
    raise ValueError(family)


def evaluate(family, p, D, turns, init, slots, ctx, penalty, fit_data=None):
    """(fitness, cpu ms per window, promotions per window, hit fraction) of the parameters on trace D; a phase family is fitted
    on fit_data = (D_fit, turns_fit) first."""
    pol = build(family, p, ctx)
    if getattr(pol, "needs_fit", False):
        pol.fit(*(fit_data if fit_data is not None else (D, turns)))
    r = tl.run(D, init, pol, tl.protect_slots(p["protect"], slots), turns, metrics=False)[0]
    ms = cl.cpu_ms(r, 0.5)
    return ms + penalty * r["promo_w"], ms, r["promo_w"], r["hits"]


class SepCMA:
    """Separable CMA-ES on [0,1]^n, minimising; ask() / tell() with the clipped points."""

    def __init__(self, mean, sigma, lam, rng):
        n = len(mean)
        self.n, self.rng, self.lam = n, rng, lam
        self.mean, self.sigma = np.array(mean, float), sigma
        self.mu = lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / np.sum(self.w ** 2)
        m = self.mueff
        self.cs = (m + 2) / (n + m + 5)
        self.damps = 1 + 2 * max(0.0, math.sqrt((m - 1) / (n + 1)) - 1) + self.cs
        self.cc = (4 + m / n) / (n + 4 + 2 * m / n)
        self.c1 = 2 / ((n + 1.3) ** 2 + m) * (n + 1) / 3
        self.cmu = min(1 - self.c1, 2 * (m - 2 + 1 / m) / ((n + 2) ** 2 + m) * (n + 2) / 3)
        self.C = np.ones(n)
        self.ps, self.pc = np.zeros(n), np.zeros(n)
        self.chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.gen = 0

    def ask(self):
        z = self.rng.standard_normal((self.lam, self.n))
        return np.clip(self.mean + self.sigma * np.sqrt(self.C) * z, 0.0, 1.0)

    def tell(self, X, f):
        order = np.argsort(f, kind="stable")[:self.mu]
        Y = (X[order] - self.mean) / self.sigma
        yw = self.w @ Y
        self.mean = np.clip(self.mean + self.sigma * yw, 0.0, 1.0)
        self.gen += 1
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * yw / np.sqrt(self.C)
        hsig = float(np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.gen)) / self.chiN < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * self.pc ** 2 + self.cmu * (self.w @ (Y ** 2))
        self.C = np.maximum(self.C, 1e-8)
        self.sigma = float(np.clip(self.sigma * math.exp(self.cs / self.damps * (np.linalg.norm(self.ps) / self.chiN - 1)),
                                   1e-3, 0.5))


def evolve(fitness, n, generations, population, seed, x0=None):
    """Random generation 0 then separable CMA-ES; fitness(u) -> float (lower is better).  Returns (best u, best f, history of the
    best f per generation, number of evaluations).  The incumbent x0 (the defaults) is evaluated too and kept if nothing beats it."""
    rng = np.random.default_rng(seed)
    cache, evals = {}, 0

    def ev(u):
        nonlocal evals
        key = tuple(np.round(u, 6))
        if key not in cache:
            cache[key] = fitness(u)
            evals += 1
        return cache[key]

    pop = rng.random((population, n))
    if x0 is not None:
        pop[0] = x0
    f = np.array([ev(u) for u in pop])
    best_i = int(np.argmin(f))
    best_u, best_f, hist = pop[best_i].copy(), float(f[best_i]), [float(f[best_i])]
    order = np.argsort(f, kind="stable")[:max(2, population // 2)]
    cma = SepCMA(pop[order].mean(0), 0.3, population, rng)
    for _g in range(1, generations):
        X = cma.ask()
        f = np.array([ev(u) for u in X])
        cma.tell(X, f)
        i = int(np.argmin(f))
        if f[i] < best_f:
            best_u, best_f = X[i].copy(), float(f[i])
        hist.append(best_f)
    return best_u, best_f, hist, evals


def run_family(family, scen, init, slots, ctx, a):
    """One family on one scenario (name, D, turns): fit on the first half, report the second half."""
    name, D, turns = scen
    h = D.shape[0] // 2
    d1, d2 = D[:h], D[h:]
    t1, t2 = tl.slice_turns(turns, 0, h), tl.slice_turns(turns, h, D.shape[0])
    x0 = encode(family, DEFAULTS[family])

    def fit(u):
        return evaluate(family, decode(family, u), d1, t1, init, slots, ctx, a.promo_penalty)[0]

    best_u, best_f, hist, evals = evolve(fit, len(x0), a.generations, a.population, a.seed, x0)
    # the defaults are in generation 0 as its first member, so the evolved fit-half fitness is never worse than the defaults'
    p_best, p_def = decode(family, best_u), decode(family, x0)
    out = {"scenario": name, "family": family, "evals": evals, "history_best_fitness": hist, "params": p_best, "defaults": p_def}
    for tag, p in (("evolved", p_best), ("defaults", p_def)):
        fit_f, fit_ms, fit_pr, fit_h = evaluate(family, p, d1, t1, init, slots, ctx, a.promo_penalty)
        ho_f, ho_ms, ho_pr, ho_h = evaluate(family, p, d2, t2, init, slots, ctx, a.promo_penalty, fit_data=(d1, t1))
        out[tag] = {"fit_half": {"fitness": fit_f, "cpu_ms": fit_ms, "promo_w": fit_pr, "hit": fit_h},
                    "holdout": {"fitness": ho_f, "cpu_ms": ho_ms, "promo_w": ho_pr, "hit": ho_h}}
    ev, df = out["evolved"], out["defaults"]
    out["overfit_gap_ms"] = (ev["holdout"]["cpu_ms"] - ev["fit_half"]["cpu_ms"]) - (df["holdout"]["cpu_ms"] - df["fit_half"]["cpu_ms"])
    out["holdout_gain_ms"] = df["holdout"]["cpu_ms"] - ev["holdout"]["cpu_ms"]       # > 0: the evolved set is better held out
    return out


def fmt(o):
    ev, df = o["evolved"], o["defaults"]
    p = ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in o["params"].items())
    lines = [f"== {o['family']} on {o['scenario']} ({o['evals']} evaluations, best fitness per generation "
             + " ".join(f"{x:.2f}" for x in o["history_best_fitness"]) + ")",
             f"   evolved  {p}",
             f"   fit half     cpu ms/w: evolved {ev['fit_half']['cpu_ms']:6.2f} ({ev['fit_half']['hit'] * 100:5.2f}% hit, "
             f"{ev['fit_half']['promo_w']:6.1f} promo/w)  defaults {df['fit_half']['cpu_ms']:6.2f} ({df['fit_half']['hit'] * 100:5.2f}%, "
             f"{df['fit_half']['promo_w']:6.1f})",
             f"   HOLD-OUT     cpu ms/w: evolved {ev['holdout']['cpu_ms']:6.2f} ({ev['holdout']['hit'] * 100:5.2f}% hit, "
             f"{ev['holdout']['promo_w']:6.1f} promo/w)  defaults {df['holdout']['cpu_ms']:6.2f} ({df['holdout']['hit'] * 100:5.2f}%, "
             f"{df['holdout']['promo_w']:6.1f})",
             f"   overfit gap {o['overfit_gap_ms']:+.2f} ms/w (evolved hold-out minus fit-half, less the defaults'); hold-out gain over "
             f"defaults {o['holdout_gain_ms']:+.2f} ms/w -> " + ("BETTER held out" if o["holdout_gain_ms"] > 0.05 else
                                                                  "NOT better held out")]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("profile")
    ap.add_argument("slots", type=int)
    ap.add_argument("trace")
    ap.add_argument("--cuts", default="")
    ap.add_argument("--names", default="")
    ap.add_argument("--convs", default="")
    ap.add_argument("--interleave", type=int, default=0)
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--population", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--promo-penalty", type=float, default=0.005, help="ms per promotion per window added to the fitness")
    ap.add_argument("--scenarios", nargs="*", default=["whole", "synthetic"], choices=["whole", "synthetic"])
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    init = rp.load_profile(a.profile, a.slots)
    ctx = {"ranked": rp.load_profile(a.profile, 10 ** 9), "seed": None}
    whole = cl.Layered.from_windows(rp.load_trace(a.trace)[0])
    cuts = [int(x) for x in a.cuts.split(",")] if a.cuts else []
    names = a.names.split(",") if a.names else [f"part{i}" for i in range(len(cuts) + 1)]
    convs = a.convs.split(",") if a.convs else names
    scen = []
    if "whole" in a.scenarios:
        scen.append((f"whole ({whole.n} windows)", tl.dense(whole, rp.NE), tl.turns_from_cuts(cuts, whole.n, convs) if cuts else None))
    if "synthetic" in a.scenarios and a.interleave > 0 and cuts:
        edges = [0] + cuts + [whole.n]
        parts = [whole.slice(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        tr, tt = tl.interleave_turns(parts, names, a.interleave)
        scen.append((f"SYNTHETIC interleave {a.interleave}-window turns ({tr.n} windows)", tl.dense(tr, rp.NE), tt))
    print(f"seed {a.seed}, {a.generations} generations x {a.population}, fitness = cpu ms/w + {a.promo_penalty} x promo/w, fit on "
          f"the first half, hold-out = the second half")
    report = []
    for sc in scen:
        for fam in a.families:
            o = run_family(fam, sc, init, a.slots, ctx, a)
            print(fmt(o), flush=True)
            report.append(o)
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
