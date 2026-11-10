"""disco_fit.py - pancake-strata (MOD_PLAN A22): fit DISCO's learned MTP chain stop from window traces or DISCO traces.

DISCO (Mamou et al., Intel Labs / Weizmann / Hebrew U, arXiv 2405.04304): after each drafted token a small classifier
predicts whether the target will accept it; the chain stops at the first predicted rejection.  SpecFit (arXiv 2610.15417)
prices each draft position by what it costs to verify, which on this box is the CPU time of the new missed experts the
row wakes.  The engine side is include/strata/spec/disco.hpp (STRATA_DISCO=<weights>); this script:

  1. reads traces: window traces (STRATA_WINDOW_TRACE=<file>.stwt, item 64; record 4 = draft probabilities and sources,
     record 5 / the layer records = the new CPU misses of every row) and/or DISCO's own CSV (STRATA_DISCO_TRACE);
  2. for a window trace, replays DiscoPolicy's per-draft features exactly as the engine computes them online
     (p, -log p, chain product, position, recent acceptance EMA, the row's expected new CPU misses EMA, context);
     the two a window trace cannot replay (5 lookup match, 7 DraftPolicy's dcost) are fitted at weight 0;
  3. fits a logistic regression (numpy, IRLS, L2; standardised features) on the drafts that were reached;
  4. picks the threshold by replaying the logged windows with the cut applied: a window of T rows cut to k drafts keeps
     min(accepted, k) + 1 tokens and costs its measured time minus what the dropped rows cost, from a least-squares
     model fitted on the trace:  ms = c0 + c1 (rows - 1) + c2 (new CPU misses of the draft rows) + c3 (row 0's misses);
     then a per-position threshold table (coordinate ascent from the best of the global threshold and every static
     length), so each position's threshold carries what that row costs (SpecFit) on top of how likely it holds;
  5. writes the weights file (`disco 2`) the engine loads with STRATA_DISCO=<file>.

    .venv/bin/python tools/pancake/disco_fit.py trace.stwt -o data/disco-weights.txt         # fit + report
    .venv/bin/python tools/pancake/disco_fit.py trace.stwt --simulate data/disco-weights.txt # replay a weights file

--simulate reports, per arm, the expected accepted tokens per window, the window cost and tokens per ms: the logged
windows (the current DraftPolicy / --spec-min-p rule that wrote the trace), DISCO at its threshold and a threshold sweep,
every static chain length, --spec-min-p cuts, and the oracle cut (verify exactly the drafts that held: the ceiling of
any rule that only stops earlier).  DISCO can only cut a logged chain, never extend it, so collect the trace with long
chains: requests with "spec_min_p": 0 (and no STRATA_SPEC_COST), so every draft the MTP made is verified and labelled.

Offline estimate only: a cut window moves where the next windows start, which the replay ignores, and the cost model is a
linear fit.  Measure the result with the A/B spec (bench/specs/24-disco.json) before trusting it.
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "trace"))
import wtrace  # noqa: E402

NF = 9
FEATURE_NAMES = ["p", "surprise", "chain", "pos", "recent", "match", "miss", "dcost", "ctx"]
REPLAY_FEATURES = [0, 1, 2, 3, 4, 6, 8]   # what a window trace lets us replay exactly (5 match, 7 dcost: weight 0)
KMAXT = 16
VERSION = 2


# ---------------------------------------------------------------------------------------------------------------------
# DiscoPolicy's online state, replayed (src/spec/disco.cpp: features(), end_round())
class DiscoState:
    def __init__(self):
        self.recent = 0.7
        self.match_f = 0.0
        self.ctx_f = 0.0
        self.nm = [0.0] * (KMAXT + 1)
        self.nm_n = [0] * (KMAXT + 1)

    def row_miss_expect(self, row):
        for r in range(min(max(row, 0), KMAXT), 0, -1):
            if self.nm_n[r] > 0:
                return self.nm[r]
        return 0.0

    def features(self, idx, p, chain):
        pp = min(max(float(p), 1e-6), 1.0)
        return [pp, -math.log(pp) / 5.0, min(max(chain, 0.0), 1.0), idx / 8.0, self.recent, self.match_f,
                self.row_miss_expect(idx + 1) / 64.0, 0.35, self.ctx_f]

    def end_round(self, T, accepted, mtp_window, probs, row_miss, pos, match=0):
        if mtp_window and probs is not None:
            n = min(max(T - 1, 0), KMAXT - 1)
            for j in range(min(n, accepted + 1)):
                self.recent = 0.9 * self.recent + 0.1 * (1.0 if j < accepted else 0.0)
            if row_miss is not None:
                for row in range(1, min(n, len(row_miss) - 1, KMAXT) + 1):
                    v = max(0, int(row_miss[row]))
                    self.nm[row] = 0.9 * self.nm[row] + 0.1 * v if self.nm_n[row] > 0 else float(v)
                    self.nm_n[row] += 1
        self.match_f = math.log1p(max(0, match)) / 3.0
        self.ctx_f = math.log2(1.0 + max(0, pos)) / 17.0


# ---------------------------------------------------------------------------------------------------------------------
# reading
def is_stwt(path):
    with open(path, "rb") as f:
        return f.read(4) == b"STWT"


def is_mtp_window(w):
    """A plain MTP window: drafts from the MTP only (no lookup, no chained lookup, not DFlash2), not a merged prompt window."""
    return (w.accepted is not None and w.rows is not None and w.rows > 1 and w.drafts is not None
            and w.draft_src is not None and len(w.draft_src) > 0 and bool(np.all(w.draft_src == 0)))


def rounds_from_trace(tr):
    """Window trace -> rounds (every committed window; MTP windows carry the replayed per-draft features).

    round: dict(T, a, ms, pos, mtp, rows=[(idx, label, feats)], miss=[new CPU misses per row 0..T-1] or None, src)"""
    st = DiscoState()
    out = []
    prev = None
    for w in tr.windows:
        if w.rows is None:
            continue
        T = int(w.rows)
        a = None if w.accepted is None else int(w.accepted)
        mtp = is_mtp_window(w)
        rm = wtrace.row_new_miss(w) if w.layers or w.row_miss is not None else None
        rm = None if rm is None or len(rm) == 0 else [int(v) for v in rm]
        probs = None if w.drafts is None else [float(v) for v in w.drafts]
        rows = []
        if mtp:
            chain = 1.0
            for j in range(min(T - 1, KMAXT - 1, len(probs))):
                p = probs[j]
                chain *= min(max(p, 0.0), 1.0)
                label = 1 if j < a else 0 if j == a else -1
                rows.append((j, label, st.features(j, p, chain)))
        # the window's ms is the time since the previous commit: trusted only when it follows its predecessor directly
        # (same request: the position advanced by the previous window's commit) and was not a merged prompt window
        follows = prev is not None and prev[0] is not None and w.position is not None and \
            w.position == prev[0] + (prev[1] if prev[2] is None else prev[2] + 1)
        src = "merged" if a is None else "mtp" if mtp else _src_name(w)
        out.append({"T": T, "a": a if a is not None else T - 1, "ms": float(w.ms or 0.0), "ms_ok": bool(follows and a is not None),
                    "pos": w.position, "mtp": mtp, "rows": rows, "miss": rm, "src": src, "probs": probs})
        st.end_round(T, a if a is not None else 0, mtp, probs, rm, int(w.position or 0))
        prev = (w.position, T, a)
    return out


def _src_name(w):
    if w.drafts is None or w.draft_src is None or len(w.draft_src) == 0:
        return "one-row"
    s = set(int(v) for v in w.draft_src)
    if 2 in s:
        return "lookup"
    if 3 in s:
        return "mtp+chain"
    if 1 in s:
        return "dflash2"
    return "mtp"


def rounds_from_csv(path):
    """STRATA_DISCO_TRACE csv -> rounds (MTP windows only; features as the engine saw them)."""
    rounds = []
    cur, last_key, seg = None, None, 0
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0] == "round":
                continue
            try:
                rid, idx, label, T, a = (int(v) for v in row[:5])
                ms = float(row[5])
                feats = [float(v) for v in row[6:6 + NF]]
                nmiss = int(row[6 + NF]) if len(row) > 6 + NF else -1
            except ValueError:
                continue
            if len(feats) != NF:
                continue
            if last_key is not None and (rid < last_key[0] or (rid == last_key[0] and idx <= last_key[1])):
                seg += 1   # a file appended by several processes: round ids restart at 0
            last_key = (rid, idx)
            key = (seg, rid)
            if cur is None or cur["key"] != key:
                cur = {"key": key, "T": T, "a": a, "ms": ms, "ms_ok": ms > 0, "pos": None, "mtp": True, "rows": [],
                       "miss": None, "src": "mtp", "probs": None, "_nm": {}}
                rounds.append(cur)
            if idx >= 0:
                cur["rows"].append((idx, label, feats))
                if nmiss >= 0:
                    cur["_nm"][idx + 1] = nmiss
    for r in rounds:
        r["rows"].sort(key=lambda x: x[0])
        nm = r.pop("_nm")
        if nm and len(nm) == r["T"] - 1:   # every draft row known; row 0's misses are not in this trace
            r["miss"] = [0] + [nm[i] for i in range(1, r["T"])]
    return rounds


def read_rounds(paths):
    out, kinds = [], set()
    for p in paths:
        if is_stwt(p):
            out += rounds_from_trace(wtrace.read(p))
            kinds.add("stwt")
        else:
            out += rounds_from_csv(p)
            kinds.add("csv")
    return out, kinds


# ---------------------------------------------------------------------------------------------------------------------
# the window cost model
class CostModel:
    """ms = c0 + c1 (T - 1) + c2 (new CPU misses of rows 1..T-1) + c3 (row 0's) on windows whose ms is trusted.
    Without miss counts: the median ms per T, unseen T on a line through the medians."""

    def __init__(self, rounds):
        ok = [r for r in rounds if r["ms_ok"] and r["ms"] > 0]
        self.n = len(ok)
        self.coef = None
        self.med = {}
        by_t = {}
        for r in ok:
            by_t.setdefault(r["T"], []).append(r["ms"])
        self.med = {t: float(np.median(v)) for t, v in by_t.items()}
        if ok:   # drop stalls (a graph capture, an adapt join): over 4x the median of that size
            ok = [r for r in ok if r["ms"] <= 4.0 * self.med[r["T"]]]
        withm = [r for r in ok if r["miss"] is not None and len(r["miss"]) >= r["T"]]
        if len(withm) >= 20:
            A = np.array([[1.0, r["T"] - 1, sum(r["miss"][1:r["T"]]), r["miss"][0]] for r in withm])
            y = np.array([r["ms"] for r in withm])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            coef[1:] = np.maximum(coef[1:], 0.0)   # a row or a miss never makes a window cheaper
            self.coef = coef
            pred = A @ coef
            self.r2 = 1.0 - float(np.sum((y - pred) ** 2)) / max(1e-9, float(np.sum((y - y.mean()) ** 2)))
        ts = sorted(self.med)
        if len(ts) >= 2:
            x = np.array(ts, float)
            m = np.array([self.med[t] for t in ts])
            self.slope, self.icpt = np.polyfit(x, m, 1)
            self.slope = max(self.slope, 0.0)
        elif ts:
            self.slope = 0.35 * self.med[ts[0]] / (1 + 0.35 * (ts[0] - 1))
            self.icpt = self.med[ts[0]] - self.slope * ts[0]
        else:
            self.slope, self.icpt = 0.35, 0.65

    def size_ms(self, t):
        return self.med.get(t, max(1e-3, self.icpt + self.slope * t))

    def cut_ms(self, r, k):
        """The window r verified with only its first k drafts (k + 1 rows)."""
        T = r["T"]
        k = max(0, min(k, T - 1))
        if k == T - 1 and r["ms_ok"] and r["ms"] > 0:
            return r["ms"]
        if self.coef is not None and r["miss"] is not None and len(r["miss"]) >= T and r["ms_ok"] and r["ms"] > 0:
            saved = self.coef[1] * (T - 1 - k) + self.coef[2] * sum(r["miss"][k + 1:T])
            return max(0.3 * r["ms"], r["ms"] - saved)
        if r["ms_ok"] and r["ms"] > 0:   # no miss counts: scale the measured time by the size table
            return r["ms"] * self.size_ms(k + 1) / self.size_ms(T)
        return self.size_ms(k + 1)

    def describe(self):
        s = f"cost model on {self.n} windows with a trusted time; median ms by rows: " + \
            " ".join(f"T{t}:{self.med[t]:.2f}" for t in sorted(self.med))
        if self.coef is not None:
            c = self.coef
            s += (f"\n  ms = {c[0]:.2f} + {c[1]:.3f} x (rows - 1) + {c[2]:.4f} x new CPU misses of the draft rows"
                  f" + {c[3]:.4f} x row 0's  (R^2 {self.r2:.2f})")
        return s


# ---------------------------------------------------------------------------------------------------------------------
# the model
def standardise(X, active):
    mean = np.zeros(NF)
    scale = np.ones(NF)
    if len(X):
        mean[active] = X[:, active].mean(axis=0)
        sd = X[:, active].std(axis=0, ddof=1) if len(X) > 1 else np.zeros(len(active))
        scale[active] = np.where(sd > 1e-4, sd, 1.0)
    return mean, scale


def fit_logreg(X, y, mean, scale, active, l2=1.0, iters=60):
    """Newton / IRLS with an L2 penalty (not on the bias) over the active features; the others keep weight 0."""
    Z = (X[:, active] - mean[active]) / scale[active]
    A = np.hstack([np.ones((len(Z), 1)), Z])
    t = np.asarray(y, float)
    d = A.shape[1]
    w = np.zeros(d)
    R = l2 * np.eye(d)
    R[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(A @ w, -30, 30)))
        g = A.T @ (p - t) + R @ w
        H = (A * (p * (1 - p))[:, None]).T @ A + R + 1e-9 * np.eye(d)
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    full = np.zeros(NF)
    full[active] = w[1:]
    return float(w[0]), full


def score(model, x):
    z = model["bias"] + float(np.dot(model["w"], (np.asarray(x, float) - model["mean"]) / model["scale"]))
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def auc(scores, labels):
    s = np.asarray(scores, float)
    l = np.asarray(labels, int)
    n1 = int(l.sum())
    n0 = len(l) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(ss):   # average ranks over ties
        j = i
        while j < len(ss) and ss[j] == ss[i]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0
        i = j
    return float((ranks[l == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def thr_at(model, idx):
    t = model.get("thresholds")
    if t:
        return t[min(idx, len(t) - 1)]
    return model["threshold"]


def write_weights(path, model, stats):
    with open(path, "w") as f:
        f.write("# DISCO weights for STRATA_DISCO (include/strata/spec/disco.hpp), written by tools/pancake/disco_fit.py\n")
        f.write("# features: " + " ".join(FEATURE_NAMES) + "\n")
        for k, v in stats.items():
            f.write(f"# {k}: {v}\n")
        f.write(f"disco {VERSION}\n")
        f.write(f"features {NF}\n")
        f.write(f"bias {model['bias']:.8g}\n")
        f.write("w " + " ".join(f"{v:.8g}" for v in model["w"]) + "\n")
        f.write("mean " + " ".join(f"{v:.8g}" for v in model["mean"]) + "\n")
        f.write("scale " + " ".join(f"{v:.8g}" for v in model["scale"]) + "\n")
        f.write(f"threshold {model['threshold']:.6g}\n")
        if model.get("thresholds"):
            t = model["thresholds"]
            f.write(f"thresholds {len(t)} " + " ".join(f"{v:.6g}" for v in t) + "\n")


def read_weights(path):
    m = {"bias": 0.0, "w": np.zeros(NF), "mean": np.zeros(NF), "scale": np.ones(NF), "threshold": 0.5}
    ver = None
    with open(path) as f:
        for line in f:
            tok = line.split("#", 1)[0].split()
            if not tok:
                continue
            k, v = tok[0], tok[1:]
            if k == "disco":
                ver = int(v[0])
            elif k == "bias":
                m["bias"] = float(v[0])
            elif k in ("w", "mean", "scale"):
                m[k] = np.array([float(x) for x in v[:NF]])
            elif k == "threshold":
                m["threshold"] = float(v[0])
            elif k == "thresholds":
                m["thresholds"] = [float(x) for x in v[1:1 + int(v[0])]]
    if ver != VERSION:
        raise ValueError(f"{path}: disco {ver} weights; this fitter and the engine read disco {VERSION} (refit)")
    return m


# ---------------------------------------------------------------------------------------------------------------------
# the replay
def cut_disco(model, thr=None):
    """thr None: the model's own rule (its per-position table, else its threshold); a number: that threshold everywhere;
    a list: that per-position table."""
    if thr is None:
        table = model.get("thresholds") or [model["threshold"]]
    elif isinstance(thr, (list, tuple)):
        table = list(thr)
    else:
        table = [thr]

    def f(r):
        k = 0
        for (idx, _, _), s in zip(r["rows"], _scores(model, r)):
            if idx != k or s < table[min(idx, len(table) - 1)]:
                break
            k += 1
        return k
    return f


_UID = [0]


def _scores(model, r):
    """The model's score of every draft of round r, computed once per model (thresholds do not change them)."""
    if "_uid" not in model:
        _UID[0] += 1
        model["_uid"] = _UID[0]
    if r.get("_sk") != model["_uid"]:
        r["_s"] = [score(model, x) for _, _, x in r["rows"]]
        r["_sk"] = model["_uid"]
    return r["_s"]


def cut_static(n):
    return lambda r: n


def cut_min_p(mp):
    def f(r):
        k = 0
        for idx, _, x in r["rows"]:
            if idx != k or x[0] < mp:
                break
            k += 1
        return k
    return f


def cut_oracle(r):
    return r["a"]


def replay(rounds, cut, cost):
    """cut(round) -> drafts kept (MTP windows only; the others keep their logged window).
    -> dict(windows, tokens, ms, tok_per_window, ms_per_window, tok_per_ms, rows)"""
    tok = ms = rows = 0.0
    n = 0
    for r in rounds:
        if r["mtp"]:
            k = max(0, min(int(cut(r)), r["T"] - 1))
            tok += min(r["a"], k) + 1
            ms += cost.cut_ms(r, k)
            rows += k + 1
        else:
            tok += r["a"] + 1
            ms += r["ms"] if r["ms_ok"] and r["ms"] > 0 else cost.size_ms(r["T"])
            rows += r["T"]
        n += 1
    return {"windows": n, "tokens": tok, "ms": ms, "rows": rows, "tok_per_window": tok / max(1, n),
            "ms_per_window": ms / max(1, n), "tok_per_ms": tok / ms if ms > 0 else 0.0}


def logged(r):
    return r["T"] - 1


def report_arms(rounds, cost, model=None, out=None):
    out = out if out is not None else sys.stdout
    mtp = [r for r in rounds if r["mtp"]]
    base = replay(mtp, logged, cost)
    base_all = replay(rounds, logged, cost)
    arms = [("logged (the rule that wrote the trace)", logged), ("oracle cut (verify exactly what held)", cut_oracle)]
    if model is not None:
        arms.append((f"DISCO at threshold {model['threshold']:.3f}", cut_disco(model, model["threshold"])))
        if model.get("thresholds"):
            arms.append(("DISCO per-position table", cut_disco(model)))
    max_t = max((r["T"] for r in mtp), default=1)
    for n in range(0, max_t):
        arms.append((f"static {n} drafts", cut_static(n)))
    for mp in (0.3, 0.5, 0.7, 0.85):
        arms.append((f"--spec-min-p {mp}", cut_min_p(mp)))
    print(f"MTP windows {len(mtp)} of {len(rounds)}; logged: {base['tok_per_window']:.3f} tokens/window, "
          f"{base['ms_per_window']:.2f} ms/window, {base['rows'] / max(1, base['windows']):.2f} rows/window", file=out)
    print(f"{'arm':44s} {'tok/win':>8s} {'ms/win':>8s} {'tok/ms':>8s} {'vs logged':>9s} {'all windows':>11s}", file=out)
    res = {}
    for name, cut in arms:
        v = replay(mtp, cut, cost)
        va = replay(rounds, cut, cost)
        rel = v["tok_per_ms"] / base["tok_per_ms"] - 1.0 if base["tok_per_ms"] > 0 else float("nan")
        rel_all = va["tok_per_ms"] / base_all["tok_per_ms"] - 1.0 if base_all["tok_per_ms"] > 0 else float("nan")
        print(f"{name:44s} {v['tok_per_window']:8.3f} {v['ms_per_window']:8.2f} {v['tok_per_ms']:8.4f} {rel:+9.1%} {rel_all:+11.1%}",
              file=out)
        res[name] = (v, rel, rel_all)
    if model is not None:
        print("threshold sweep (MTP windows): " + "  ".join(
            f"{t:.2f}:{replay(mtp, cut_disco(model, t), cost)['tok_per_ms'] / base['tok_per_ms'] - 1.0:+.1%}"
            for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)), file=out)
    return res


def xy(rounds):
    X, y = [], []
    for r in rounds:
        for _, label, x in r["rows"]:
            if label >= 0:
                X.append(x)
                y.append(label)
    return np.array(X, float).reshape(-1, NF), np.array(y, int)


def fit(rounds, active, l2, cost):
    X, y = xy(rounds)
    if len(set(y.tolist())) < 2:
        raise ValueError("the training drafts are all accepted or all rejected; nothing to learn")
    mean, scale = standardise(X, active)
    b, w = fit_logreg(X, y, mean, scale, active, l2=l2)
    model = {"bias": b, "w": w, "mean": mean, "scale": scale, "threshold": 0.5}
    mtp = [r for r in rounds if r["mtp"]]
    best = None
    for i in range(1, 40):   # the most tokens per ms on the training windows
        thr = i / 40.0
        v = replay(mtp, cut_disco(model, thr), cost)["tok_per_ms"]
        if best is None or v > best[1] + 1e-12:
            best = (thr, v)
    model["threshold"] = best[0]
    # the per-position table: start from the better of the global threshold and the best static length (a static
    # length n is the table 0 x n, 1 after), then improve one position at a time
    depth = max((r["T"] - 1 for r in mtp), default=1)
    depth = max(1, min(depth, 16))
    starts = [[best[0]] * depth] + [[0.0] * n + [1.0] * (depth - n) for n in range(depth + 1)]
    table = max(starts, key=lambda t: replay(mtp, cut_disco(model, t), cost)["tok_per_ms"])
    grid = [0.0] + [i / 40.0 for i in range(1, 40)] + [1.0]
    cur = replay(mtp, cut_disco(model, table), cost)["tok_per_ms"]
    for _ in range(3):
        moved = False
        for pos in range(depth):
            for t in grid:
                trial = table[:pos] + [t] + table[pos + 1:]
                v = replay(mtp, cut_disco(model, trial), cost)["tok_per_ms"]
                if v > cur * (1.0 + 1e-6):
                    table, cur, moved = trial, v, True
        if not moved:
            break
    model["thresholds"] = table
    return model


def main(argv=None):
    ap = argparse.ArgumentParser(description="Fit / replay DISCO's learned MTP chain stop (MOD_PLAN A22).",
                                 epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="window traces (.stwt, STRATA_WINDOW_TRACE) and/or STRATA_DISCO_TRACE csv files")
    ap.add_argument("-o", "--out", help="weights file to write (fit mode)")
    ap.add_argument("--simulate", metavar="WEIGHTS", help="replay an existing weights file instead of fitting")
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--holdout", type=float, default=0.25, help="the last share of the windows, held out for the reported numbers")
    ap.add_argument("--min-windows", type=int, default=200, help="refuse to fit on fewer MTP windows")
    ap.add_argument("--features", choices=["auto", "all", "replay"], default="auto",
                    help="features to fit: all 9 (DISCO csv), the 7 a window trace replays, or auto (replay if any .stwt)")
    a = ap.parse_args(argv)

    rounds, kinds = read_rounds(a.traces)
    mtp = [r for r in rounds if r["mtp"]]
    if not mtp:
        print("disco_fit: no plain MTP windows with draft probabilities in these traces (record 4 needs the drafter on)",
              file=sys.stderr)
        return 2
    cost = CostModel(rounds)
    print(f"{len(rounds)} windows ({', '.join(sorted(kinds))}), {len(mtp)} plain MTP windows, "
          f"{sum(len(r['rows']) for r in mtp)} drafts")
    print(cost.describe())
    if not any(r["miss"] is not None for r in mtp):
        print("  (no per-row miss counts: feature 6 stays 0 and the cost model is the size table)")
    feat = a.features if a.features != "auto" else ("replay" if "stwt" in kinds else "all")
    active = REPLAY_FEATURES if feat == "replay" else list(range(NF))

    cut = int(len(rounds) * (1.0 - a.holdout))
    train, held = rounds[:cut], rounds[cut:]
    if not [r for r in held if r["mtp"]]:
        held = rounds
    if a.simulate:
        model = read_weights(a.simulate)
    else:
        if len(mtp) < a.min_windows:
            print(f"disco_fit: {len(mtp)} MTP windows, need {a.min_windows} (--min-windows)", file=sys.stderr)
            return 2
        try:
            model = fit(train, active, a.l2, cost)
        except ValueError as e:
            print(f"disco_fit: {e}", file=sys.stderr)
            return 2

    Xh, yh = xy(held)
    if len(yh):
        sh = [score(model, x) for x in Xh]
        print(f"held-out drafts {len(yh)} (last {a.holdout:.0%} of the windows), accepted {yh.mean():.3f}; "
              f"AUC classifier {auc(sh, yh):.3f} vs draft probability alone {auc(Xh[:, 0], yh):.3f}")
    print("weights (standardised): " + " ".join(f"{n}={v:+.3f}" for n, v in zip(FEATURE_NAMES, model["w"])))
    print(f"threshold {model['threshold']:.3f}" + (
        "; per-position table " + " ".join(f"{t:.3f}" for t in model["thresholds"]) if model.get("thresholds") else ""))
    print("\nheld-out replay:" if not a.simulate else "\nreplay:")
    res = report_arms(held if not a.simulate else rounds, cost, model)
    if a.out and not a.simulate:
        name = "DISCO per-position table" if model.get("thresholds") else f"DISCO at threshold {model['threshold']:.3f}"
        v, rel, _ = res[name]
        stats = {"traces": " ".join(os.path.basename(p) for p in a.traces), "windows": len(rounds), "mtp_windows": len(mtp),
                 "features": feat, "held_out_gain_vs_logged": f"{rel:+.2%}",
                 "held_out_tok_per_ms": f"{v['tok_per_ms']:.4f}"}
        write_weights(a.out, model, stats)
        print(f"\nwrote {a.out}  (engine: STRATA_DISCO={a.out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
