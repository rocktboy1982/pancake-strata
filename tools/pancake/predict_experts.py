"""predict_experts.py - pancake-strata, plan 10.2/10.3: score expert predictors offline on a --dump-routing trace.

For each verify window and layer the used set is the union of the experts its tokens routed to.  Before the layer runs,
each predictor ranks the experts; the experts the cache does not hold, taken in that order up to `budget`, are what a
prefetch would copy.  Per predictor and budget:
  recall      top-`budget` of the ranking that were used / experts used          (prediction accuracy)
  miss cover  used experts the cache did NOT hold that the prefetch copied / those misses   (what a prefetch saves)
  waste       prefetched experts that were not used / prefetched experts      (PCIe spent for nothing)
"Held" is replay_cache_policy.py's simulation of the engine's adaptive tier, started from PROFILE with SLOTS slots.

Predictors (MOD_PLAN 10, numbers as there):
  prev-window   1: this layer's experts in the previous window
  frequency     1: decayed per-layer frequency (EMA)
  prev+freq     1: prev-window, topped up by frequency
  transition    3 (routing-keyed): experts used at layer L-1 in this window -> layer L, learned online (EMA)
  coact         6: co-activation partners of this layer's previous-window experts, learned online (EMA)
  hybrid        5: transition + prev+freq, each normalised to its maximum
The token-keyed dictionary of 3 needs token ids, which the trace does not record.

    python tools/pancake/predict_experts.py TRACE.bin PROFILE.bin SLOTS [--budget 4 8 16] [--decay 0.8]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay_cache_policy as R  # noqa: E402


class Predictors:
    """Online state shared by all predictors; scores() before a layer runs, observe() after."""

    def __init__(self, nl, ne, decay):
        self.decay = decay
        self.freq = np.zeros((nl, ne), np.float32)
        self.prev = [np.zeros(0, np.int64) for _ in range(nl)]
        self.trans = np.zeros((nl, ne, ne), np.float32)  # [l, expert at l-1, expert at l]
        self.coact = np.zeros((nl, ne, ne), np.float32)  # [l, a, b]: a and b used together at l

    def scores(self, l, below):
        """Score vector per predictor for layer l; `below` = experts used at layer l-1 in this window (or None)."""
        ne = self.freq.shape[1]
        prev = np.zeros(ne, np.float32)
        prev[self.prev[l]] = 1.0
        freq = self.freq[l]
        pf = prev * (freq.max() + 1.0) + freq  # prev-window first, frequency breaks ties and tops up
        trans = self.trans[l, below].sum(0) if below is not None and len(below) else np.zeros(ne, np.float32)
        coact = self.coact[l, self.prev[l]].sum(0)
        hybrid = trans / max(trans.max(), 1e-9) + pf / max(pf.max(), 1e-9)
        return {"prev-window": prev, "frequency": freq, "prev+freq": pf, "transition": trans, "coact": coact,
                "hybrid": hybrid}

    def observe(self, l, act, below):
        self.freq[l] *= self.decay
        self.freq[l, act] += 1.0
        if below is not None and len(below):
            t = self.trans[l]
            t[below] *= self.decay
            t[np.ix_(below, act)] += 1.0
        c = self.coact[l]
        c[act] *= self.decay
        c[np.ix_(act, act)] += 1.0
        self.prev[l] = act


def rank(score):
    """Experts by descending score, zero-score experts dropped (a predictor proposes only what it has evidence for)."""
    order = np.argsort(-score, kind="stable")
    return order[score[order] > 0]


def evaluate(windows, init, budgets, decay=0.8, pol=None, nl=R.NL, ne=R.NE):
    """Run every predictor over the windows; returns (held share, {(name, budget): (recall, miss_cover, waste)})."""
    pol = pol or R.Policy()
    res = np.zeros((nl, ne), bool)
    res[init[:, 0], init[:, 1]] = True
    usage = np.zeros((nl, ne))
    P = Predictors(nl, ne, decay)
    acc = {}  # (name, budget) -> [hit, used, covered, missed, wasted, fetched]
    used_all = miss_all = 0
    for wi, w in enumerate(windows):
        below = None
        for l, ids in w:
            act = np.unique(ids)
            miss = act[~res[l, act]]
            used_all += len(act)
            miss_all += len(miss)
            for name, s in P.scores(l, below).items():
                order = rank(s)
                not_held = order[~res[l, order]]
                for b in budgets:
                    fetch = not_held[:b]
                    a = acc.setdefault((name, b), [0] * 6)
                    a[0] += np.isin(order[:b], act).sum()
                    a[1] += len(act)
                    a[2] += np.isin(fetch, miss).sum()
                    a[3] += len(miss)
                    a[4] += (~np.isin(fetch, act)).sum()
                    a[5] += len(fetch)
            P.observe(l, act, below)
            np.add.at(usage[l], ids.ravel(), 1.0)
            below = act
        if (wi + 1) % pol.every == 0:
            R.adapt_round(res, usage, pol)
            usage *= pol.decay
    out = {k: (a[0] / max(1, a[1]), a[2] / max(1, a[3]), a[4] / max(1, a[5])) for k, a in acc.items()}
    return 1 - miss_all / max(1, used_all), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("profile")
    ap.add_argument("slots", type=int)
    ap.add_argument("--budget", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--decay", type=float, default=0.8)
    a = ap.parse_args()
    windows, k = R.load_trace(a.trace)
    held, out = evaluate(windows, R.load_profile(a.profile, a.slots), a.budget, a.decay)
    print(f"{len(windows)} windows, k={k}; the simulated cache held {held:.1%} of the distinct experts used "
          f"(per window and layer)")
    print(f"{'predictor':12} {'budget':>6} {'recall':>7} {'miss cover':>11} {'waste':>7}")
    for (name, b), (rc, mc, ws) in sorted(out.items(), key=lambda kv: (kv[0][1], -kv[1][1])):
        print(f"{name:12} {b:6d} {rc:7.1%} {mc:11.1%} {ws:7.1%}")


if __name__ == "__main__":
    main()
