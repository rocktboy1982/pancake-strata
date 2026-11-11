#!/usr/bin/env python3
"""ecospec_sim.py - EcoSpec-style cost-aware draft selection, replayed offline on window traces (no GPU, no model).

EcoSpec (Xie, Liu et al., "Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for Mixture-of-Experts",
arXiv 2607.12696, 2026-07) ranks the candidate tokens of a draft TREE by S = P(path) / (new-expert cost + eps), where the
new-expert cost counts the experts a candidate adds to a buffer B (the union of the experts of the tokens already chosen),
and verifies the best gamma of them.  The verification rule is unchanged, so the output is identical.  Measured on
DeepSeek-V3.1 / Qwen3-235B-A22B / GPT-OSS-120B (NVIDIA, HBM), with a trained expert predictor; modest gains
(Qwen3-235B 1.22x -> 1.36x over standard SD).

What this engine can use of it.  There is no draft tree here (a tree verify needs GDN state forks and a wider expert union:
rejected in MOD_PLAN 16.4), only the MTP's chain, so "which gamma tokens" becomes "how many drafts": the prefix length k
that maximises expected tokens per unit cost,

    k* = argmax_k (1 + sum_{j<=k} q_j) / cost(k),        q_j = P(drafts 1..j all hold) = prod of the per-draft scores,

with cost(k) from one of three sources (all replayed on the held-out tail of the trace; the verifier always accepts exactly
what the target would, so the replay only chooses a window length):

  eco-table   cost(k) = DraftPolicy's per-size window cost table, built by trace/draft_costs.py (the STRATA_SPEC_COST_TABLE
              the engine can load today): the same price for every window of that size.
  eco-ema     cost(k) = the one-row window + sum over the drafts of (row ms + c2 x the row's EXPECTED new CPU misses), the
              expectation being DISCO's feature 6 (an EMA of earlier windows' per-row new misses): what the engine knows
              without any predictor.
  eco-oracle  cost(k) = the true cost of this window cut at k rows (the cost model on this window's own per-row new
              misses): EcoSpec with a PERFECT expert predictor, the ceiling of what a predictor could add.

q_j comes from the logistic model disco_fit.py fits on the training windows (features: p, surprise, chain, position,
recent acceptance, expected new misses, context), so the acceptance side is calibrated, not the raw softmax.  The other
arms are the baselines: the rule that wrote the trace, every static length, --spec-min-p, STRATA_SPEC_CUM (F8), DISCO
(its fitted per-position table), and the oracle cut (verify exactly what held: no chain-stop rule can beat it).

Reading the result.  An engine switch for EcoSpec-style selection is worth building only if eco-ema beats the best
baseline by >= 2% of modeled decode time on a REAL trace (this prints that number).  eco-oracle minus eco-ema is what a
trained expert predictor (the paper's component; the plan measured next-layer expert recall of 19-21%) could add.

    python tools/pancake/ecospec_sim.py TRACE.stwt [...] [--holdout 0.25] [--row-ms MS] [--us-per-miss 40] [--l2 1.0]

Collect the trace as for disco_fit.py: STRATA_WINDOW_TRACE=<file> with `--spec-min-p 0` (every draft verified, so every
draft has a label).  Synthetic costs in the tests are made up: they check the mechanism, not the gain.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "trace"))
import disco_fit as D  # noqa: E402
import draft_costs  # noqa: E402
import wtrace  # noqa: E402

GAIN_BAR = 0.02   # an engine switch must beat the best baseline by this much (MOD_PLAN research order: >= 2% of decode time)


def survival(model, r):
    """q_j for each draft j of window r: the chance that drafts 1..j all hold (product of the model's scores)."""
    q, acc = [], 1.0
    for s in D._scores(model, r):
        acc *= s
        q.append(acc)
    return q


class Costs:
    """The three cost sources of the eco arms (see the module doc)."""

    def __init__(self, rounds, cost, tables, max_t):
        self.cost, self.tables, self.max_t = cost, tables, max_t

    def table_ms(self, r, k):
        row = self.tables[draft_costs.bucket(r["pos"] or 0)] or next((t for t in self.tables if t), None)
        if row is None:
            return self.cost.size_ms(k + 1)
        t = min(k + 1, len(row))
        return row[t - 1] + (k + 1 - t) * max(0.0, self.cost.slope)   # beyond the table: the fitted slope per row

    def ema_ms(self, r, k):
        c = self.cost.coef
        if c is None:
            return self.table_ms(r, k)
        ms = self.cost.size_ms(1)
        for _, _, x in r["rows"][:k]:
            ms += c[1] + c[2] * x[6] * 64.0   # x[6] = the row's expected new misses / 64 (DISCO feature 6)
        return ms

    def oracle_ms(self, r, k):
        return self.cost.cut_ms(r, k)


def cut_eco(model, costs, source):
    """The chain rule: the prefix length with the most expected tokens per ms under `source` costs."""
    ms_of = {"table": costs.table_ms, "ema": costs.ema_ms, "oracle": costs.oracle_ms}[source]

    def f(r):
        n = len(r["rows"])
        if n == 0:
            return 0
        q = survival(model, r)
        best_k, best = 0, None
        gain = 1.0
        for k in range(0, n + 1):
            if k > 0:
                gain += q[k - 1]
            v = gain / max(1e-6, ms_of(r, k))
            if best is None or v > best * (1.0 + 1e-9):
                best, best_k = v, k
        return best_k
    return f


def arms(model, costs, rounds_mtp):
    out = [("logged (the rule that wrote the trace)", D.logged), ("oracle cut (verify exactly what held)", D.cut_oracle)]
    max_t = max((r["T"] for r in rounds_mtp), default=1)
    out += [(f"static {n} drafts", D.cut_static(n)) for n in range(0, max_t)]
    out += [(f"--spec-min-p {mp}", D.cut_min_p(mp)) for mp in (0.3, 0.5, 0.7)]
    out += [(f"STRATA_SPEC_CUM {s}", D.cut_cum(s)) for s in (0.1, 0.2, 0.3, 0.5)]
    out += [("DISCO per-position table", D.cut_disco(model))]
    out += [("eco-table (DraftPolicy cost table)", cut_eco(model, costs, "table")),
            ("eco-ema (expected new misses)", cut_eco(model, costs, "ema")),
            ("eco-oracle (a perfect expert predictor)", cut_eco(model, costs, "oracle"))]
    return out


def run(rounds, cost, tables, held, model, out=None):
    """Replay every arm on the held-out MTP windows.  -> {arm: tokens per ms}, and the verdict numbers."""
    out = out if out is not None else sys.stdout
    mtp = [r for r in held if r["mtp"]]
    costs = Costs(rounds, cost, tables, max((r["T"] for r in rounds), default=1))
    base = D.replay(mtp, D.logged, cost)
    res = {}
    print(f"{'arm':44s} {'tok/win':>8s} {'ms/win':>8s} {'tok/ms':>8s} {'vs logged':>9s}", file=out)
    for name, cut in arms(model, costs, mtp):
        v = D.replay(mtp, cut, cost)
        rel = v["tok_per_ms"] / base["tok_per_ms"] - 1.0 if base["tok_per_ms"] > 0 else float("nan")
        res[name] = v["tok_per_ms"]
        print(f"{name:44s} {v['tok_per_window']:8.3f} {v['ms_per_window']:8.2f} {v['tok_per_ms']:8.4f} {rel:+9.1%}", file=out)
    baselines = [k for k in res if not k.startswith("eco-") and not k.startswith("oracle") and not k.startswith("logged")]
    best_name = max(baselines, key=lambda k: res[k])
    best = res[best_name]
    verdict = {"best_baseline": best_name, "best_baseline_tok_per_ms": best,
               "eco_ema_vs_best": res["eco-ema (expected new misses)"] / best - 1.0,
               "eco_table_vs_best": res["eco-table (DraftPolicy cost table)"] / best - 1.0,
               "eco_oracle_vs_best": res["eco-oracle (a perfect expert predictor)"] / best - 1.0,
               "oracle_cut_vs_best": res["oracle cut (verify exactly what held)"] / best - 1.0}
    print(f"\nbest baseline: {best_name} ({best:.4f} tok/ms)", file=out)
    print(f"  eco-table  {verdict['eco_table_vs_best']:+.1%}   eco-ema  {verdict['eco_ema_vs_best']:+.1%}   "
          f"eco-oracle  {verdict['eco_oracle_vs_best']:+.1%}   oracle cut (ceiling of any stop rule)  "
          f"{verdict['oracle_cut_vs_best']:+.1%}", file=out)
    if verdict["eco_ema_vs_best"] >= GAIN_BAR:
        print(f"  eco-ema beats the best baseline by >= {GAIN_BAR:.0%}: an engine switch is worth a screen.", file=out)
    else:
        print(f"  eco-ema is under +{GAIN_BAR:.0%} over the best baseline: no engine switch from this trace "
              f"(a perfect expert predictor would add {verdict['eco_oracle_vs_best'] - verdict['eco_ema_vs_best']:+.1%}).", file=out)
    return res, verdict


def main(argv=None):
    ap = argparse.ArgumentParser(description="EcoSpec-style cost-aware draft length, replayed on window traces.",
                                 epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="window traces (.stwt, STRATA_WINDOW_TRACE)")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--row-ms", type=float, default=None)
    ap.add_argument("--us-per-miss", type=float, default=40.0, help="draft_costs.py's price of one new CPU miss")
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--min-windows", type=int, default=200)
    a = ap.parse_args(argv)
    for p in a.traces:
        if not D.is_stwt(p):
            print(f"ecospec_sim: {p} is not a window trace (STWT)", file=sys.stderr)
            return 2
    rounds, _ = D.read_rounds(a.traces)
    mtp = [r for r in rounds if r["mtp"]]
    if len(mtp) < a.min_windows:
        print(f"ecospec_sim: {len(mtp)} plain MTP windows, need {a.min_windows} (--min-windows)", file=sys.stderr)
        return 2
    cost = D.CostModel(rounds, a.row_ms)
    wins = []
    for p in a.traces:
        wins += [w for w in wtrace.read(p).windows if w.rows and w.ms and w.position is not None and w.accepted is not None]
    tables, _ = draft_costs.build_tables(wins, a.us_per_miss, max_t=D.KMAXT)
    cut = int(len(rounds) * (1.0 - a.holdout))
    train, held = rounds[:cut], rounds[cut:]
    if not [r for r in held if r["mtp"]]:
        held = rounds
    print(f"{len(rounds)} windows, {len(mtp)} plain MTP windows; trained on the first {len(train)}, replayed on the last {len(held)}")
    print(cost.describe())
    try:
        model = D.fit(train, D.REPLAY_FEATURES, a.l2, cost)
    except ValueError as e:
        print(f"ecospec_sim: {e}", file=sys.stderr)
        return 2
    print()
    run(rounds, cost, tables, held, model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
