#!/usr/bin/env python3
"""window_policy.py - MOD_PLAN.md section 6 item 21 (window policy controller), step 1, offline on a window trace.

Question: how much does a per-window choice of how many draft rows to verify beat the fixed preset, and how much of that can a
controller reach from the acceptance of the last N windows alone?  The decision of the item: build `STRATA_WINDOW_POLICY` only if
the oracle headroom over the best FIXED rule is >= --gate (5%).

Capture (opt-in, one GPU run, nothing when unset): STRATA_WINDOW_TRACE=/tmp/w.stwt on a short / edit / agent replay (the trace
format is tools/pancake/trace/wtrace.py; tail_replay.py documents the capture).  Per window the trace has rows, accepted, wall ms
and every entry's placement, so for each kept-row count k (1 = only the committed token ... rows = the whole window):
    tokens(k) = min(accepted + 1, k)
    ms(k)     = max(0.1, ms - saved(k) x us_per_miss / 1000)     saved(k) = CPU-missed experts only rows >= k needed
(tail_replay.window_effect; the same time model, CPU time per distinct missed expert = --us-per-miss, 23 us short chat, 60 us edit).
The ms(k) side is a MODEL, not a measurement; a policy that wins here is a candidate for a GPU A/B, not a result.

Policies printed (tokens per second of the trace's own wall time, vs the full window = the preset):
    full      every row verified (the preset)
    fixed k   the best constant kept-row count
    theta     tail_replay's confidence rule (windows with draft confidences only), best of --thetas
    causal    kept rows = ceil(mean accepted of the last N windows) + 1 + margin, N in --ns, margin in --margins (what a controller can
              know before the window); the best (N, margin) is reported but it is picked on the same trace, so it is an upper bound
              of the family, not held out; --split halves the trace and picks on the first half, scores on the second
    oracle    per window the k that maximises total tokens / total ms (Dinkelbach iteration): the ceiling of any row-count rule
NOT replayable from a trace: the draft source and depth (`--spec 4` vs `--lookup-chain 3`), DRAFT_CLIP, the lookup length: the
other source's drafts were never drawn.  Those need an A/B arm, not a replay.

    python tools/pancake/window_policy.py TRACE.stwt [--us-per-miss 40] [--ns 1,2,4,8] [--margins 0,1,2] [--gate 0.05] [--split]
    python -m pytest -q tools/pancake/test_window_policy.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "trace"))
import tail_replay  # noqa: E402
import wtrace  # noqa: E402


def usable(tr):
    return [w for w in tr.windows if w.accepted is not None and w.rows and w.rows >= 1 and w.ms]


def tables(wins, us_per_miss):
    """Per window arrays over k = 1..rows: tokens[k-1], ms[k-1]."""
    out = []
    for w in wins:
        ks = np.arange(1, w.rows + 1)
        tok = np.minimum(w.accepted + 1, ks).astype(np.float64)
        ms = np.empty(len(ks))
        for i, k in enumerate(ks):
            saved, _ = tail_replay.window_effect(w, int(k)) if k < w.rows else (0, 0)
            ms[i] = max(0.1, w.ms - saved * us_per_miss / 1e3)
        out.append((tok, ms))
    return out


def rate(tab, ks):
    tok = sum(t[min(k, len(t)) - 1] for (t, _), k in zip(tab, ks))
    ms = sum(m[min(k, len(m)) - 1] for (_, m), k in zip(tab, ks))
    return tok / ms * 1e3


def oracle(tab, iters=50):
    """Per window k maximising total tokens / total ms: Dinkelbach (maximise tokens - lambda ms per window, update lambda)."""
    ks = [len(t) for t, _ in tab]
    lam = rate(tab, ks) / 1e3
    for _ in range(iters):
        ks2 = [int(np.argmax(t - lam * m)) + 1 for t, m in tab]
        new = rate(tab, ks2) / 1e3
        ks = ks2
        if new <= lam + 1e-12:
            break
        lam = new
    return ks, rate(tab, ks)


def fixed_best(tab):
    kmax = max(len(t) for t, _ in tab)
    best = max(range(1, kmax + 1), key=lambda k: rate(tab, [k] * len(tab)))
    return best, rate(tab, [best] * len(tab))


def causal_ks(wins, n, margin):
    """Kept rows from the last n windows' accepted drafts (the first window keeps everything)."""
    ks, hist = [], []
    for w in wins:
        if hist:
            pred = float(np.mean(hist[-n:]))
            ks.append(min(w.rows, int(np.ceil(pred - 1e-9)) + 1 + margin))
        else:
            ks.append(w.rows)
        hist.append(w.accepted)
    return ks


def causal_best(wins, tab, ns, margins):
    grid = {(n, m): rate(tab, causal_ks(wins, n, m)) for n in ns for m in margins}
    key = max(grid, key=grid.get)
    return key, grid[key], grid


def analyse(tr, us_per_miss=40.0, ns=(1, 2, 4, 8), margins=(0, 1, 2), thetas=(0.05, 0.1, 0.2, 0.3, 0.5), split=False):
    wins = usable(tr)
    if not wins:
        return None
    tab = tables(wins, us_per_miss)
    full = rate(tab, [len(t) for t, _ in tab])
    rep = {"windows": len(wins), "full": full}
    rep["fixed_k"], rep["fixed"] = fixed_best(tab)
    _, rep["oracle"] = oracle(tab)
    (rep["causal_n"], rep["causal_margin"]), rep["causal"], rep["causal_grid"] = causal_best(wins, tab, ns, margins)
    dw = [(w, t) for w, t in zip(wins, tab) if w.drafts is not None and len(w.drafts) and not np.isnan(w.drafts[0])]
    if dw:
        best = None
        for th in thetas:
            ks = [min(tail_replay.keep_rows(w.drafts, th), w.rows) for w, _ in dw]
            r = rate([t for _, t in dw], ks)
            if best is None or r > best[1]:
                best = (th, r)
        rep["theta"], rep["theta_rate"] = best
        rep["theta_windows"] = len(dw)
        rep["theta_full"] = rate([t for _, t in dw], [len(t[0]) for _, t in dw])
    if split and len(wins) >= 8:
        h = len(wins) // 2
        key, _, _ = causal_best(wins[:h], tab[:h], ns, margins)
        # score the second half; its history starts cold, like a new session
        rep["causal_heldout"] = {"picked_on_first_half": list(key), "full": rate(tab[h:], [len(t) for t, _ in tab[h:]]),
                                 "rate": rate(tab[h:], causal_ks(wins[h:], *key))}
    best_fixed_rate = max(rep["fixed"], full)
    rep["oracle_headroom"] = rep["oracle"] / best_fixed_rate - 1.0
    return rep


def report(rep, gate):
    if rep is None:
        return "no usable windows (rows, accepted and ms are needed)"
    f = rep["full"]
    pct = lambda x: f"{x:7.1f} tok/s ({x / f * 100 - 100:+.1f}% vs full)"
    lines = [f"{rep['windows']} windows; full window (the preset) {f:.1f} tok/s on the trace's wall time",
             f"  best fixed k = {rep['fixed_k']}: {pct(rep['fixed'])}"]
    if "theta" in rep:
        lines.append(f"  confidence rule theta {rep['theta']:.2f} ({rep['theta_windows']} drafter windows, vs their own full "
                     f"{rep['theta_full']:.1f}): {rep['theta_rate']:.1f} tok/s ({rep['theta_rate'] / rep['theta_full'] * 100 - 100:+.1f}%)")
    lines.append(f"  causal N={rep['causal_n']} margin={rep['causal_margin']} (picked on this trace): {pct(rep['causal'])}")
    if "causal_heldout" in rep:
        h = rep["causal_heldout"]
        lines.append(f"  causal held out (picked on the first half {h['picked_on_first_half']}): second half "
                     f"{h['rate']:.1f} vs full {h['full']:.1f} tok/s ({h['rate'] / h['full'] * 100 - 100:+.1f}%)")
    lines.append(f"  oracle per-window k: {pct(rep['oracle'])}")
    hr = rep["oracle_headroom"]
    lines.append(f"  oracle headroom over the best fixed rule {hr:+.1%}: " +
                 (f"over the {gate:.0%} gate, STRATA_WINDOW_POLICY is worth an A/B arm" if hr >= gate else
                  f"under the {gate:.0%} gate, do not build STRATA_WINDOW_POLICY"))
    lines.append("  (ms per k is the time model, not a measurement; draft source / depth cannot be replayed)")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trace")
    ap.add_argument("--us-per-miss", type=float, default=40.0)
    ap.add_argument("--ns", default="1,2,4,8")
    ap.add_argument("--margins", default="0,1,2")
    ap.add_argument("--thetas", default="0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--gate", type=float, default=0.05)
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    tr = wtrace.read(a.trace)
    rep = analyse(tr, a.us_per_miss, tuple(int(x) for x in a.ns.split(",")), tuple(int(x) for x in a.margins.split(",")),
                  tuple(float(x) for x in a.thetas.split(",")), a.split)
    print(report(rep, a.gate))
    if a.json and rep:
        rep = dict(rep)
        rep["causal_grid"] = {f"{n},{m}": v for (n, m), v in rep["causal_grid"].items()}
        Path(a.json).write_text(json.dumps(rep, indent=1))
    return 0 if rep else 2


if __name__ == "__main__":
    sys.exit(main())
