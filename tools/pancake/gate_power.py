#!/usr/bin/env python3
"""gate_power.py - how often do the two quality gates decide?  Monte Carlo on the gates' own code (no engine, no model).

  tokens   force_check.check (the rate rule + the large-margin event rule) on synthetic arms: P(PASS / INCONCLUSIVE / REJECT) for an arm that is
           identical to stock and for arms that add +0.5 / +1.0 / +1.5 points of mismatches, at the size of the old set (20 requests, 5,322 tokens,
           the real per-request lengths) and of the grown one (48 requests, 20 old + 28 new).
  tasks    task_gate.boot_diff + verdict (95% bootstrap interval of the paired pass-rate difference, margin 5 points): P(PASS or BETTER /
           INCONCLUSIVE / REJECT) for a true loss of 0 / 5 / 10 points, by number of tasks and repetitions, for two difficulty mixes.

    python tools/pancake/gate_power.py tokens [--trials 300] [--new-len 240] [--model ties,iid] [--seed 1]
    python tools/pancake/gate_power.py tasks  [--trials 100] [--tasks 18,100,150] [--reps 3,5,10] [--boots 1000] [--tasks-only] [--seed 1]

(tools/pancake/power_sim.py is another tool: the speed rule of promote.py on abtest's noise.)

Token-forcing noise models (rows are what force_check.load_arm returns: (mismatch, margin, top-2 margin)):
  ties  fitted to the real stock arms (bench/quality/stock-a.tsv and stock-b.tsv, identical config): of 5,322 positions 62 and 69 mismatch and 36
        are the same position (independent arms would share 0.8), so the mismatches are near-tie positions where a run flips a coin: a share T of
        the positions is a tie (T = 2.2%, a mismatch with probability 1/2 in every arm, independently), the rest never mismatches.  The per-
        request counts are over-dispersed (variance 11.8 against 3.1 for a binomial): the tie share of a request is T x a Gamma(shape 1.25)
        weight, the same for every arm.  A lossy arm adds `extra` points of independent mismatches (scaled by the same request weight).  The
        margins of the stock flips are small (median 0.06, 90% 0.19, max 2.2 nats): a margin is drawn from an exponential with mean 0.1, so the
        event rule stays quiet unless an arm adds large-margin events, which these arms do not.
  iid   what pancake-docs/batch3/promotion.md simulated: every position mismatches independently with probability p (the worst case: the same
        positions are NOT shared between arms, no request effect).
Task-gate model: every arm run draws every task independently from its pass rate (no seed pairing: a lossy arm leaves the sampled path at its
first different token, so independent draws are the realistic case); a true loss of d points scales every rate by 1 - d / mean rate.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import force_check as FC  # noqa: E402
import task_gate as TG  # noqa: E402

ROOT = TG.ROOT
TIE_SHARE, TIE_SHAPE, FLIP, MARGIN_MEAN = 0.022, 1.25, 0.5, 0.1
IID_RATE = 0.012                      # the real stock arms: 1.17% and 1.30%
OLD_LENS = [250, 185, 185, 28, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 300, 174, 300, 300, 300]   # bench/quality/set.json
# task difficulty mixes: (share of tasks, pass rate of the base)
MIX_PROMOTION = ((10, 0.95), (3, 0.80), (3, 0.50), (2, 0.10))          # promotion.md: 10 tasks at 95%, 3 at 80%, 3 at 50%, 2 at 10%
MIX_BALANCED = ((2, 0.95), (2, 0.80), (4, 0.50), (2, 0.20))            # a set picked for being informative: most tasks neither always nor never solved


# ------------------------------------------------------------------------------------------------ token forcing
def weights(rng, n):
    return [rng.gammavariate(TIE_SHAPE, 1.0 / TIE_SHAPE) for _ in range(n)]


def tie_mask(rng, lens, w):
    """Which positions are near-ties: a property of the text and the position, so every arm of one trial shares it."""
    return [[rng.random() < min(1.0, TIE_SHARE * wj) for _ in range(n)] for n, wj in zip(lens, w)]


def arm_rows(rng, lens, w, ties, model, extra_pts=0.0):
    """{request: [(mismatch, margin, top2)]} for one arm of the same requests (`w` and `ties` shared by the arms of a trial)."""
    out = {}
    for j, (n, wj) in enumerate(zip(lens, w)):
        rows = []
        for i in range(n):
            if model == "iid":
                p = IID_RATE + extra_pts / 100.0
            else:
                p = (FLIP if ties[j][i] else 0.0) + (extra_pts / 100.0) * wj
            mis = rng.random() < p
            rows.append((mis, rng.expovariate(1.0 / MARGIN_MEAN) if mis else 0.0, 1.0))
        out[f"r{j}"] = rows
    return out


def token_power(lens, model, extra_pts, trials, rng, boots=1000):
    """Shares {"PASS": n, "INCONCLUSIVE": n, "REJECT": n} of force_check.check over `trials` draws: two stock arms and one arm `extra_pts` worse."""
    got = {"PASS": 0, "INCONCLUSIVE": 0, "REJECT": 0}
    for _ in range(trials):
        w = weights(rng, len(lens))
        ties = tie_mask(rng, lens, w)
        stock = [arm_rows(rng, lens, w, ties, model) for _ in range(2)]
        got[FC.check(stock, arm_rows(rng, lens, w, ties, model, extra_pts), boots=boots, seed=rng.randrange(1 << 30))["verdict"]] += 1
    return {k: v / trials for k, v in got.items()}


# ------------------------------------------------------------------------------------------------ task gate
def mix_rates(mix, n):
    """n base pass rates following the mix proportions (the last bucket takes the rounding)."""
    total = sum(c for c, _ in mix)
    rates = []
    for c, p in mix:
        rates += [p] * round(n * c / total)
    return (rates + [mix[0][1]] * n)[:n]


def draw(rng, rates, reps):
    return {f"t{i}": [1 if rng.random() < p else 0 for _ in range(reps)] for i, p in enumerate(rates)}


def task_power(mix, n_tasks, reps, loss_pts, trials, rng, boots=1000, margin=0.05, resample_reps=True):
    """Shares {"PASS": n (PASS or BETTER), "INCONCLUSIVE": n, "REJECT": n} of task_gate.verdict over `trials` draws (resample_reps False =
    task_gate.py --tasks-only: the interval from resampling the tasks alone)."""
    base = mix_rates(mix, n_tasks)
    scale = 1.0 - (loss_pts / 100.0) / (sum(base) / len(base))
    got = {"PASS": 0, "INCONCLUSIVE": 0, "REJECT": 0}
    for _ in range(trials):
        r = TG.boot_diff(draw(rng, [p * scale for p in base], reps), draw(rng, base, reps), boots, rng.randrange(1 << 30), resample_reps=resample_reps)
        v = TG.verdict(r[1], r[2], margin)
        got["PASS" if v in ("PASS", "BETTER") else v] += 1
    return {k: v / trials for k, v in got.items()}


# ------------------------------------------------------------------------------------------------ reports
def cell(s):
    return f"{s['PASS'] * 100:3.0f}/{s['INCONCLUSIVE'] * 100:3.0f}/{s['REJECT'] * 100:3.0f}"


def report_tokens(a):
    rng = random.Random(a.seed)
    sizes = {f"{len(OLD_LENS)} requests (old set)": OLD_LENS, f"{len(OLD_LENS) + 28} requests (20 old + 28 new of {a.new_len} tokens)": OLD_LENS + [a.new_len] * 28}
    lines = []
    for model in a.model:
        lines += [f"\ntoken forcing, model {model}: share of PASS/INCONCLUSIVE/REJECT in %, {a.trials} trials, 90% two-sided = 95% one-sided bootstrap over requests",
                  f"{'set':52s}" + "".join(f"{'+' + str(e) + ' pts':>14s}" for e in a.extra)]
        for name, lens in sizes.items():
            lines.append(f"{name + f' ({sum(lens)} tokens)':52s}" + "".join(f"{cell(token_power(lens, model, e, a.trials, rng, a.boots)):>14s}" for e in a.extra))
    return "\n".join(lines)


def report_tasks(a):
    rng = random.Random(a.seed)
    lines = []
    for name, mix in (("promotion.md mix (56% of tasks at 95%, 17% at 80%, 17% at 50%, 11% at 10%)", MIX_PROMOTION),
                      ("balanced mix (20% of the tasks at 95%, 20% at 80%, 40% at 50%, 20% at 20%)", MIX_BALANCED)):
        lines += [f"\ntask gate{', --tasks-only' if a.tasks_only else ''}, {name}: share of PASS(+BETTER)/INCONCLUSIVE/REJECT in %, {a.trials} trials, margin 5 pts, boots {a.boots}",
                  f"{'tasks x reps':>14s}" + "".join(f"{'loss ' + str(d) + ' pts':>14s}" for d in a.loss)]
        for n in a.tasks:
            for reps in a.reps:
                lines.append(f"{f'{n} x {reps}':>14s}" + "".join(f"{cell(task_power(mix, n, reps, d, a.trials, rng, a.boots, resample_reps=not a.tasks_only)):>14s}" for d in a.loss))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tokens")
    t.add_argument("--trials", type=int, default=300)
    t.add_argument("--new-len", type=int, default=240, help="scored tokens per new request (the generation stops at 300 or at the end of the turn)")
    t.add_argument("--model", type=lambda s: s.split(","), default=["ties", "iid"])
    t.add_argument("--extra", type=lambda s: [float(x) for x in s.split(",")], default=[0.0, 0.5, 1.0, 1.5])
    t.add_argument("--boots", type=int, default=1000)
    t.add_argument("--seed", type=int, default=1)
    k = sub.add_parser("tasks")
    k.add_argument("--trials", type=int, default=100)
    k.add_argument("--tasks", type=lambda s: [int(x) for x in s.split(",")], default=[18, 100, 150])
    k.add_argument("--reps", type=lambda s: [int(x) for x in s.split(",")], default=[3, 5, 10])
    k.add_argument("--loss", type=lambda s: [float(x) for x in s.split(",")], default=[0.0, 5.0, 10.0])
    k.add_argument("--boots", type=int, default=1000)
    k.add_argument("--tasks-only", action="store_true", help="the interval of task_gate.py --tasks-only (resample the tasks, not the repetitions)")
    k.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    print(report_tokens(a) if a.cmd == "tokens" else report_tasks(a))
    return 0


if __name__ == "__main__":
    sys.exit(main())
