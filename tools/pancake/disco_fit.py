"""disco_fit.py - pancake-strata: fit DISCO's learned MTP window rule from STRATA_DISCO_TRACE logs.

DISCO (Mamou et al., Intel Labs / Weizmann / Hebrew U, arXiv 2405.04304): after each drafted token a small
classifier predicts whether the target will accept it; the chain stops at the first predicted rejection.  The
engine side is include/strata/spec/disco.hpp.  This script:

  1. reads one or more trace CSVs (one line per drafted token of a plain MTP window, label 1/0/-1 =
     accepted / the first rejected / not reached);
  2. fits a logistic regression (standardised features, L2) on the reached drafts;
  3. picks the threshold by replaying the logged rounds: a round of T rows whose drafts the rule cuts at k keeps
     min(a, k) + 1 tokens at the median measured cost of a k + 1 row window, and the threshold with the most tokens per
     ms wins; the same replay prices every static window length and --spec-min-p, for comparison;
  4. writes the weights file the engine loads with STRATA_DISCO=<file>.

Collect traces with the base rule verifying every draft, so every draft has a label (a rule that stops early leaves
the later drafts unlabelled and the replay cannot widen a window it never saw):

    STRATA_DISCO_TRACE=/path/disco.csv  ./run-pancake.sh ...     # and send requests with "spec_min_p": 0
    .venv/bin/python tools/pancake/disco_fit.py /path/disco.csv -o data/disco-weights.txt

The replay is an offline estimate: a different window moves where the next rounds start, which it ignores, and
round_ms is the verify + commit time of the window (the next window's draft is not in it).  Measure the result with an
A/B run before trusting it.
"""
import argparse
import csv
import math
import random
import statistics
import sys
from pathlib import Path

NF = 9
FEATURE_NAMES = ["p", "surprise", "chain", "pos", "recent", "match", "miss", "dcost", "ctx"]


def read_traces(paths):
    """-> list of rounds: dict(T, a, ms, rows=[(idx, label, feats)])"""
    rounds = []
    for seg_file, path in enumerate(paths):
        with open(path, newline="") as f:
            rd = csv.reader(f)
            cur = None
            last_key = None
            seg = 0
            for row in rd:
                if not row or row[0] == "round":
                    continue
                try:
                    rid, idx, label, T, a = (int(v) for v in row[:5])
                    ms = float(row[5])
                    feats = [float(v) for v in row[6:6 + NF]]
                except ValueError:
                    continue
                if len(feats) != NF:
                    continue
                # a file appended by several processes: round ids restart at 0
                if last_key is not None and (rid < last_key[0] or (rid == last_key[0] and idx <= last_key[1])):
                    seg += 1
                last_key = (rid, idx)
                key = (seg_file, seg, rid)
                if cur is None or cur["key"] != key:
                    cur = {"key": key, "T": T, "a": a, "ms": ms, "rows": []}
                    rounds.append(cur)
                if idx >= 0:
                    cur["rows"].append((idx, label, feats))
    for r in rounds:
        r["rows"].sort(key=lambda x: x[0])
    return rounds


def standardise(X):
    n = len(X)
    mean = [sum(x[i] for x in X) / n for i in range(NF)]
    scale = []
    for i in range(NF):
        v = sum((x[i] - mean[i]) ** 2 for x in X) / max(1, n - 1)
        s = math.sqrt(v)
        scale.append(s if s > 1e-4 else 1.0)
    return mean, scale


def fit_logreg(X, y, mean, scale, l2=1.0, iters=50):
    """Newton / IRLS with an L2 penalty (not on the bias).  numpy when present, plain Python otherwise."""
    Z = [[(x[i] - mean[i]) / scale[i] for i in range(NF)] for x in X]
    try:
        import numpy as np
        A = np.hstack([np.ones((len(Z), 1)), np.array(Z)])
        t = np.array(y, dtype=float)
        w = np.zeros(NF + 1)
        R = l2 * np.eye(NF + 1)
        R[0, 0] = 0.0
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -30, 30)))
            g = A.T @ (p - t) + R @ w
            H = (A * (p * (1 - p))[:, None]).T @ A + R + 1e-9 * np.eye(NF + 1)
            step = np.linalg.solve(H, g)
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        return float(w[0]), [float(v) for v in w[1:]]
    except ImportError:
        b, w = 0.0, [0.0] * NF
        lr = 0.5
        n = len(Z)
        for _ in range(3000):
            gb, gw = 0.0, [l2 * wi / n for wi in w]
            for z, t in zip(Z, y):
                p = 1.0 / (1.0 + math.exp(-max(-30, min(30, b + sum(wi * zi for wi, zi in zip(w, z))))))
                e = (p - t) / n
                gb += e
                for i in range(NF):
                    gw[i] += e * z[i]
            b -= lr * gb
            w = [wi - lr * gi for wi, gi in zip(w, gw)]
        return b, w


def score(model, x):
    b, w, mean, scale = model["bias"], model["w"], model["mean"], model["scale"]
    z = b + sum(w[i] * (x[i] - mean[i]) / scale[i] for i in range(NF))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    n1 = sum(labels)
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg = (i + j + 1) / 2.0
        rank_sum += avg * sum(l for _, l in pairs[i:j])
        i = j
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def cost_table(rounds, max_t=16):
    """median round ms per window rows; unseen sizes from a least-squares line through the seen medians"""
    by_t = {}
    for r in rounds:
        if r["ms"] > 0:
            by_t.setdefault(r["T"], []).append(r["ms"])
    med = {t: statistics.median(v) for t, v in by_t.items() if v}
    if not med:
        return {t: 1.0 + 0.35 * (t - 1) for t in range(1, max_t + 1)}, med
    ts = sorted(med)
    if len(ts) >= 2:
        mt = sum(ts) / len(ts)
        mm = sum(med[t] for t in ts) / len(ts)
        den = sum((t - mt) ** 2 for t in ts)
        slope = sum((t - mt) * (med[t] - mm) for t in ts) / den if den > 0 else 0.0
        icpt = mm - slope * mt
    else:  # one size seen: DraftPolicy's prior shape, one more row = +0.35 of a one-row round
        slope = 0.35 * med[ts[0]] / (1 + 0.35 * (ts[0] - 1))
        icpt = med[ts[0]] - slope * ts[0]
    table = {t: med.get(t, max(1e-3, icpt + slope * t)) for t in range(1, max_t + 1)}
    return table, med


def replay(rounds, cut, cost):
    """cut(round) -> drafts kept k.  -> (tokens, ms, tokens per ms)"""
    tok = ms = 0.0
    for r in rounds:
        k = max(0, min(cut(r), r["T"] - 1))
        tok += min(r["a"], k) + 1
        ms += cost[k + 1]
    return tok, ms, (tok / ms if ms > 0 else 0.0)


def disco_cut(model, thr):
    def f(r):
        k = 0
        for idx, _, x in r["rows"]:
            if idx != k or score(model, x) < thr:
                break
            k += 1
        return k
    return f


def min_p_cut(mp):
    def f(r):
        k = 0
        for idx, _, x in r["rows"]:
            if idx != k or x[0] < mp:
                break
            k += 1
        return k
    return f


def write_weights(path, model, stats):
    with open(path, "w") as f:
        f.write("# DISCO weights for STRATA_DISCO (include/strata/spec/disco.hpp), written by tools/pancake/disco_fit.py\n")
        f.write("# features: " + " ".join(FEATURE_NAMES) + "\n")
        for k, v in stats.items():
            f.write(f"# {k}: {v}\n")
        f.write("disco 1\n")
        f.write(f"features {NF}\n")
        f.write(f"bias {model['bias']:.8g}\n")
        f.write("w " + " ".join(f"{v:.8g}" for v in model["w"]) + "\n")
        f.write("mean " + " ".join(f"{v:.8g}" for v in model["mean"]) + "\n")
        f.write("scale " + " ".join(f"{v:.8g}" for v in model["scale"]) + "\n")
        f.write(f"threshold {model['threshold']:.6g}\n")


def read_weights(path):
    m = {"bias": 0.0, "w": [0.0] * NF, "mean": [0.0] * NF, "scale": [1.0] * NF, "threshold": 0.5}
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].split()
        if not line:
            continue
        k, v = line[0], line[1:]
        if k == "bias":
            m["bias"] = float(v[0])
        elif k in ("w", "mean", "scale"):
            m[k] = [float(x) for x in v[:NF]]
        elif k == "threshold":
            m["threshold"] = float(v[0])
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("traces", nargs="+", help="STRATA_DISCO_TRACE csv files")
    ap.add_argument("-o", "--out", help="weights file to write")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--holdout", type=float, default=0.2, help="share of rounds held out for the reported numbers")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--simulate", help="replay an existing weights file instead of fitting")
    ap.add_argument("--min-rounds", type=int, default=200, help="refuse to fit on fewer rounds")
    a = ap.parse_args(argv)

    rounds = read_traces(a.traces)
    rounds = [r for r in rounds if r["T"] >= 1]
    if len(rounds) < a.min_rounds:
        print(f"disco_fit: {len(rounds)} rounds, need {a.min_rounds} (--min-rounds)", file=sys.stderr)
        return 2
    max_t = max(r["T"] for r in rounds)
    cost, med = cost_table(rounds)
    print(f"{len(rounds)} rounds, widest window {max_t} rows; median ms by rows: "
          + " ".join(f"T{t}:{med[t]:.2f}" for t in sorted(med)))

    rnd = random.Random(a.seed)
    held = [r for r in rounds if rnd.random() < a.holdout]
    held_keys = {id(r) for r in held}
    train = [r for r in rounds if id(r) not in held_keys] or rounds
    if not held:
        held = rounds

    def xy(rs):
        X, y = [], []
        for r in rs:
            for _, lab, x in r["rows"]:
                if lab in (0, 1):
                    X.append(x)
                    y.append(lab)
        return X, y

    if a.simulate:
        model = read_weights(a.simulate)
    else:
        X, y = xy(train)
        if len(set(y)) < 2:
            print("disco_fit: the training drafts are all accepted or all rejected; nothing to learn", file=sys.stderr)
            return 2
        mean, scale = standardise(X)
        b, w = fit_logreg(X, y, mean, scale, l2=a.l2)
        model = {"bias": b, "w": w, "mean": mean, "scale": scale, "threshold": 0.5}
        # threshold: the most tokens per ms on the training rounds
        best = None
        for i in range(1, 40):
            thr = i / 40.0
            _, _, v = replay(train, disco_cut(model, thr), cost)
            if best is None or v > best[1]:
                best = (thr, v)
        model["threshold"] = best[0]

    Xh, yh = xy(held)
    sh = [score(model, x) for x in Xh]
    a_h = auc(sh, yh) if Xh else float("nan")
    a_p = auc([x[0] for x in Xh], yh) if Xh else float("nan")
    base_rate = sum(yh) / len(yh) if yh else float("nan")
    print(f"held-out drafts {len(yh)}, accepted {base_rate:.3f}; AUC classifier {a_h:.3f} vs draft probability alone {a_p:.3f}")
    print("weights (standardised): " + " ".join(f"{n}={v:+.3f}" for n, v in zip(FEATURE_NAMES, model["w"])))

    logged = replay(held, lambda r: r["T"] - 1, cost)
    statics = {L: replay(held, (lambda L: lambda r: L - 1)(L), cost) for L in range(1, max_t + 1)}
    minps = {mp: replay(held, min_p_cut(mp), cost) for mp in (0.3, 0.4, 0.5, 0.6, 0.7)}
    disco = replay(held, disco_cut(model, model["threshold"]), cost)
    best_static = max(statics.items(), key=lambda kv: kv[1][2])
    best_minp = max(minps.items(), key=lambda kv: kv[1][2])
    per_ms = lambda v: 1000.0 * v[2]  # noqa: E731
    print(f"replay on held-out rounds (tokens per second of verify+commit time, offline estimate):")
    print(f"  as logged                 {per_ms(logged):8.1f}")
    for L, v in statics.items():
        print(f"  static {L} rows             {per_ms(v):8.1f}")
    for mp, v in minps.items():
        print(f"  --spec-min-p {mp:.1f}         {per_ms(v):8.1f}")
    print(f"  DISCO thr {model['threshold']:.3f}         {per_ms(disco):8.1f}  "
          f"({100 * (disco[2] / best_static[1][2] - 1):+.1f}% vs best static {best_static[0]}, "
          f"{100 * (disco[2] / best_minp[1][2] - 1):+.1f}% vs best --spec-min-p {best_minp[0]})")

    if a.out and not a.simulate:
        stats = {
            "rounds": len(rounds), "held_out_auc": f"{a_h:.4f}", "auc_p_only": f"{a_p:.4f}",
            "replay_gain_vs_best_static": f"{100 * (disco[2] / best_static[1][2] - 1):+.2f}% (static {best_static[0]} rows)",
            "replay_gain_vs_best_min_p": f"{100 * (disco[2] / best_minp[1][2] - 1):+.2f}% (min_p {best_minp[0]})",
            "traces": " ".join(str(Path(p).name) for p in a.traces),
        }
        write_weights(a.out, model, stats)
        print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
