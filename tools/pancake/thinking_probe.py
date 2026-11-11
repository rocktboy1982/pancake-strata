#!/usr/bin/env python3
"""thinking_probe.py - MOD_PLAN Phase 16 A19: train the learned thinking stop's linear probe on the engine's hidden-state dumps.

Thought Calibration (MIT 2025, https://arxiv.org/abs/2505.18404) and Learning How Hard to Think (ICLR'25) read a linear probe
off the model's hidden state at reasoning-step boundaries and stop the thinking once it says the answer would no longer
change.  The engine side is strata/program/think_probe.hpp:

    STRATA_HIDDEN_DUMP=<dir>[,every=N]          the head's input (n_embd floats) at the thinking's check points -> <dir>/hd-*.bin
                                                 (+ hd-*.ids: every thinking token id), exact
    STRATA_THINK_PROBE=<probe.sprb>,<thr>[,k=K][,min=M]   the probe closes the thinking (lossy; the server's soft close)

Commands
    info  <dump|dir>...                      what the dumps hold (requests, records, natural closes, thinking lengths)
    train <dump|dir>... [options]            labels -> probe -> held-out calibration report -> SPRB weights (--out)
    labels-plan <dump|dir>... --out F.jsonl  the forced-stop replay plan for ground-truth labels (one line per check point)

Labels (see pancake-docs/batch2/A19-thinking-probe.md for the full procedure)
    ground truth (--labels F.jsonl): one JSON line per check point, {"file": "hd-....bin", "think_index": t, "label": 0|1},
        label 1 = the final answer is already correct / unchanged when the thinking is cut after t tokens.  Made by the
        forced-stop replay: for every line of `labels-plan`, send the request's prompt with the thinking prefix ids[:t]
        followed by "\\n</think>\\n\\n" (greedy) and compare the answer with the full run's (the same tool call name and
        arguments, the same final text, or the task checker's verdict for tools/pancake/task_gate.py tasks).
    cheap proxy (default, no replay): only requests whose thinking ended at the model's own </think> (an END record);
        label 1 when the point is within --proxy-tail tokens of that close or past --proxy-frac of the thinking.  It
        learns "the model is about to close" - safe but small savings (about the tail); ground truth is what can find
        the early "already decided" points the papers report (-20-40% thinking on routine turns).

Training: features standardised on the training split; logistic regression (L2, full-batch accelerated gradient) or ridge
(closed form).  The split is by request (file), never by row.  The report: AUC, Brier, log loss, ECE (10 bins) and the
reliability table on held-out requests, then the engine's stop rule replayed on them for a threshold grid (fires,
premature fires = a fire on a label-0 point, thinking saved); the recommended threshold is the lowest one reached walking
down from the highest while the premature rate's Hoeffding upper bound stays under --risk (fixed-sequence testing, which is how
Thought Calibration picks its threshold; the walk ends at the first threshold that fails).  The held-out requests are both
the calibration set and the report set: there is no independent test split.  The export folds the standardisation into w and
b (SPRB v1).

Only numpy.  Tests: python -m pytest tools/pancake/test_thinking_probe.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MAGIC_DUMP = b"SHDM"
MAGIC_PROBE = b"SPRB"
VERSION = 1
HEADER = struct.Struct("<4sIIIQII")        # magic, version, dim, every, start_unix_ms, prompt_tokens, request_seq (32 bytes)
REC_HEAD = struct.Struct("<IiIf")          # think_index, token, flags, score (16 bytes)
PERIODIC, PARAGRAPH, END, FIRED = 1, 2, 4, 8
LOGISTIC, IDENTITY = 0, 1


@dataclass
class Dump:
    path: str
    dim: int
    every: int
    start_ms: int
    prompt_tokens: int
    seq: int
    think_index: np.ndarray            # (n,) int64
    token: np.ndarray                  # (n,) int32
    flags: np.ndarray                  # (n,) uint32
    score: np.ndarray                  # (n,) float32 (NaN: no probe)
    h: np.ndarray                      # (n, dim) float32
    ids: np.ndarray | None = None      # every thinking token id (the .ids companion), or None

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    @property
    def end_index(self) -> int | None:
        """The thinking length at the model's own </think> (the END record), or None (capped / closed / cut off)."""
        e = np.nonzero(self.flags & END)[0]
        return int(self.think_index[e[-1]]) if len(e) else None


# ---------------------------------------------------------------- the dump format
def read_dump(path) -> Dump:
    path = str(path)
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < HEADER.size:
        raise ValueError(f"{path}: too short for a dump header")
    magic, ver, dim, every, start_ms, prompt, seq = HEADER.unpack_from(raw, 0)
    if magic != MAGIC_DUMP:
        raise ValueError(f"{path}: not a hidden-state dump (magic {magic!r})")
    if ver != VERSION:
        raise ValueError(f"{path}: dump version {ver}, this tool reads {VERSION}")
    if dim <= 0:
        raise ValueError(f"{path}: bad dimension {dim}")
    rec = REC_HEAD.size + 4 * dim
    body = len(raw) - HEADER.size
    n = body // rec
    if body % rec:                         # a file cut while the engine wrote it: keep the whole records
        print(f"thinking_probe: {path}: {body % rec} trailing bytes ignored (a partial record)", file=sys.stderr)
    dt = np.dtype([("think_index", "<u4"), ("token", "<i4"), ("flags", "<u4"), ("score", "<f4"), ("h", "<f4", (dim,))])
    arr = np.frombuffer(raw, dtype=dt, count=n, offset=HEADER.size)
    ids = None
    ip = path[:-4] + ".ids" if path.endswith(".bin") else None
    if ip and os.path.exists(ip):
        ids = np.fromfile(ip, dtype="<i4")
    return Dump(path, dim, every, start_ms, prompt, seq, arr["think_index"].astype(np.int64), arr["token"].astype(np.int32),
                arr["flags"].astype(np.uint32), arr["score"].astype(np.float32), np.array(arr["h"], dtype=np.float32), ids)


def write_dump(path, dim, records, every=64, start_ms=0, prompt_tokens=0, seq=0, ids=None) -> None:
    """records: (think_index, token, flags, score, h) tuples.  The engine's writer, for tests and synthetic data."""
    with open(path, "wb") as f:
        f.write(HEADER.pack(MAGIC_DUMP, VERSION, dim, every, start_ms, prompt_tokens, seq))
        for t, tok, fl, sc, h in records:
            h = np.asarray(h, dtype="<f4")
            assert h.shape == (dim,)
            f.write(REC_HEAD.pack(int(t), int(tok), int(fl), float(sc)))
            f.write(h.tobytes())
    if ids is not None:
        np.asarray(ids, dtype="<i4").tofile(str(path)[:-4] + ".ids")


def find_dumps(paths) -> list[str]:
    out = []
    for p in paths:
        p = Path(p)
        out += sorted(str(x) for x in p.glob("hd-*.bin")) if p.is_dir() else [str(p)]
    return out


# ---------------------------------------------------------------- the probe file
def write_probe(path, w, b, link=LOGISTIC) -> None:
    w = np.asarray(w, dtype="<f4").ravel()
    if not np.all(np.isfinite(w)) or not math.isfinite(b):
        raise ValueError("non-finite probe weights")
    with open(path, "wb") as f:
        f.write(struct.pack("<4sIIIf", MAGIC_PROBE, VERSION, len(w), link, float(b)))
        f.write(w.tobytes())


def read_probe(path):
    raw = Path(path).read_bytes()
    magic, ver, dim, link, b = struct.unpack_from("<4sIIIf", raw, 0)
    if magic != MAGIC_PROBE or ver != VERSION:
        raise ValueError(f"{path}: not an SPRB v{VERSION} probe")
    w = np.frombuffer(raw, dtype="<f4", count=dim, offset=20).astype(np.float32)
    return w, float(b), int(link)


def sigmoid(s):
    """1 / (1 + exp(-s)) without overflow warnings."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(s, dtype=np.float64)))


def probe_score(w, b, link, h):
    """The engine's score (think_probe.hpp Probe::score): link(b + w.h), accumulated in float64."""
    s = b + np.asarray(h, dtype=np.float64) @ np.asarray(w, dtype=np.float64)
    return s if link == IDENTITY else sigmoid(s)


# ---------------------------------------------------------------- labels
def proxy_labels(d: Dump, tail=128, frac=0.9):
    """The cheap proxy: None for a request without a natural close; else 1 where the point is within `tail` tokens of the
    model's own </think> or past `frac` of the thinking."""
    end = d.end_index
    if end is None:
        return None
    t = d.think_index
    return ((end - t <= tail) | (t >= frac * end)).astype(np.int8)


def load_label_file(path) -> dict:
    """Ground truth: {(file basename, think_index): 0|1} from JSON lines."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("label") not in (0, 1, True, False):
                raise ValueError(f"{path}:{n}: label must be 0 or 1")
            out[(os.path.basename(r["file"]), int(r["think_index"]))] = int(r["label"])
    return out


def file_labels(d: Dump, table: dict):
    """Labels from the ground-truth table; -1 where the table has no row (those points are left out)."""
    return np.array([table.get((d.name, int(t)), -1) for t in d.think_index], dtype=np.int8)


def replay_plan(dumps) -> list[dict]:
    """One line per check point that is not the model's own close: the thinking prefix to force (ids[:t])."""
    out = []
    for d in dumps:
        if d.ids is None:
            continue
        for t, fl in zip(d.think_index, d.flags):
            if fl & END or t > len(d.ids):
                continue
            out.append({"file": d.name, "think_index": int(t), "start_ms": d.start_ms, "prompt_tokens": d.prompt_tokens,
                        "prefix_ids": [int(x) for x in d.ids[:t]]})
    return out


# ---------------------------------------------------------------- models
@dataclass
class Model:
    w: np.ndarray                      # on standardised features
    b: float
    mean: np.ndarray
    std: np.ndarray
    link: int
    info: dict = field(default_factory=dict)

    def raw(self, X):
        return ((np.asarray(X, np.float64) - self.mean) / self.std) @ self.w + self.b

    def predict(self, X):
        s = self.raw(X)
        return s if self.link == IDENTITY else sigmoid(s)

    def folded(self):
        """(w', b') on the raw vector: w' = w / std, b' = b - sum(w * mean / std)."""
        wf = self.w / self.std
        return wf, float(self.b - np.dot(wf, self.mean))


def standardise(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std > 1e-6, std, 1.0)
    return mean, std


def fit_logreg(X, y, l2=1.0, iters=600, tol=1e-7) -> Model:
    """L2 logistic regression on standardised features by accelerated gradient descent (step 1/L from a power-iteration
    bound on the Hessian).  l2 is per sample (the loss is the mean log loss + l2/2 |w|^2 / n)."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    n, d = X.shape
    mean, std = standardise(X)
    Z = (X - mean) / std
    v = np.random.default_rng(0).standard_normal(d)
    for _ in range(30):                                   # the largest eigenvalue of Z^T Z / n
        v = Z.T @ (Z @ v) / n
        v /= np.linalg.norm(v) + 1e-30
    lam = float(v @ (Z.T @ (Z @ v)) / n)
    reg = l2 / n
    L = 0.25 * (lam + 1.0) + reg                          # (+1: the bias column)
    w = np.zeros(d)
    p0 = min(max(y.mean(), 1e-4), 1 - 1e-4)
    b = math.log(p0 / (1 - p0))
    wv, bv, tk = w.copy(), b, 1.0
    prev = np.inf
    for it in range(iters):
        s = Z @ wv + bv
        p = sigmoid(s)
        g = p - y
        gw = Z.T @ g / n + reg * wv
        gb = g.mean()
        w_new, b_new = wv - gw / L, bv - gb / L
        tk1 = (1 + math.sqrt(1 + 4 * tk * tk)) / 2
        wv = w_new + (tk - 1) / tk1 * (w_new - w)
        bv = b_new + (tk - 1) / tk1 * (b_new - b)
        w, b, tk = w_new, b_new, tk1
        if it % 25 == 0:
            loss = log_loss(y, sigmoid(Z @ w + b)) + 0.5 * reg * float(w @ w)
            if prev - loss < tol:
                break
            prev = loss
    return Model(w, float(b), mean, std, LOGISTIC, {"model": "logreg", "l2": l2, "iters": it + 1})


def fit_ridge(X, y, l2=1.0) -> Model:
    """Ridge regression on standardised features (targets 0/1): closed form, primal or dual, whichever is smaller."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    n, d = X.shape
    mean, std = standardise(X)
    Z = (X - mean) / std
    yc = y - y.mean()
    if d <= n:
        w = np.linalg.solve(Z.T @ Z + l2 * np.eye(d), Z.T @ yc)
    else:
        w = Z.T @ np.linalg.solve(Z @ Z.T + l2 * np.eye(n), yc)
    return Model(w, float(y.mean()), mean, std, IDENTITY, {"model": "ridge", "l2": l2})


# ---------------------------------------------------------------- metrics
def log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, s):
    """Rank AUC (ties count half); None with one class only."""
    y = np.asarray(y)
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if pos == 0 or neg == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = np.asarray(s)[order]
    i = 0
    while i < len(ss):                                    # average ranks of ties
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def calibration(y, p, bins=10):
    """ECE and the reliability table (bins of equal width on [0, 1])."""
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.minimum(np.digitize(p, edges[1:-1]), bins - 1)
    rows, ece = [], 0.0
    for k in range(bins):
        m = idx == k
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(conf - acc)
        rows.append({"bin": f"{edges[k]:.1f}-{edges[k + 1]:.1f}", "n": int(m.sum()), "mean_score": round(conf, 4),
                     "positive_rate": round(acc, 4)})
    return float(ece), rows


def simulate_stop(groups, threshold, k=3, min_tokens=0):
    """The engine's StopRule replayed on each request's check points (in order; the END record is not a check).
    groups: [(think_index, scores, labels, end_index or None)].  -> fires, premature (label 0 at the fire), saved tokens
    and the thinking tokens of the requests with a natural close."""
    fires = premature = 0
    saved = total = 0
    for t, s, lab, fl, end in groups:
        streak, fired_at, fired_label = 0, None, None
        for ti, si, li, fi in zip(t, s, lab, fl):
            if fi & END:
                continue
            streak = streak + 1 if si >= threshold else 0
            if streak >= k and ti >= min_tokens:
                fired_at, fired_label = int(ti), int(li)
                break
        if end is not None:
            total += end
        if fired_at is not None:
            fires += 1
            premature += fired_label == 0
            if end is not None:
                saved += max(0, end - fired_at)
    return {"threshold": float(threshold), "fires": fires, "premature": int(premature), "saved_tokens": int(saved),
            "thinking_tokens": int(total), "saved_share": round(saved / total, 4) if total else None}


def hoeffding_upper(bad, n, delta=0.1):
    return 1.0 if n == 0 else bad / n + math.sqrt(math.log(1 / delta) / (2 * n))


def pick_threshold(sweep, risk=0.1, delta=0.1):
    """Fixed-sequence testing, as Thought Calibration calibrates its threshold (Learn then Test, arXiv 2505.18404): walk the
    sweep from the highest threshold down; a threshold passes when its premature-fire rate's Hoeffding upper bound is <= risk
    (one that never fires has nothing to bound and passes); the first that fails ends the walk.  -> the lowest threshold that
    fired and passed before that, or None.  Taking the lowest passing threshold of the whole sweep instead would pick the
    luckiest of ~50 correlated tests and the bound would not hold."""
    best = None
    for r in sorted(sweep, key=lambda r: -r["threshold"]):
        if r["fires"] and hoeffding_upper(r["premature"], r["fires"], delta) > risk:
            break
        if r["fires"]:
            best = r["threshold"]
    return best


# ---------------------------------------------------------------- training run
def collect(dumps, table=None, tail=128, frac=0.9):
    """-> X, y, group index per row, per-request (dump, mask) kept."""
    Xs, ys, gs, kept = [], [], [], []
    for d in dumps:
        lab = file_labels(d, table) if table is not None else proxy_labels(d, tail, frac)
        if lab is None:
            continue
        m = lab >= 0
        if not m.any():
            continue
        Xs.append(d.h[m])
        ys.append(lab[m])
        gs.append(np.full(int(m.sum()), len(kept)))
        kept.append((d, m, lab))
    if not Xs:
        return None, None, None, []
    return np.concatenate(Xs), np.concatenate(ys).astype(np.int8), np.concatenate(gs), kept


def split_groups(n_groups, holdout, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_groups)
    n_test = max(1, int(round(n_groups * holdout))) if n_groups > 1 else 0
    return set(order[:n_test].tolist()), set(order[n_test:].tolist())


def train(dumps, table=None, model="logreg", l2=1.0, holdout=0.25, seed=0, k=3, min_tokens=0, risk=0.1, delta=0.1,
          tail=128, frac=0.9, thresholds=None):
    X, y, g, kept = collect(dumps, table, tail, frac)
    if X is None:
        raise SystemExit("thinking_probe: no labelled points (proxy labels need requests that closed their own thinking)")
    test_g, train_g = split_groups(len(kept), holdout, seed)
    tr = np.isin(g, list(train_g))
    te = np.isin(g, list(test_g))
    if len(np.unique(y[tr])) < 2:
        raise SystemExit("thinking_probe: the training split has one class only: more requests are needed")
    fit = fit_logreg if model == "logreg" else fit_ridge
    m = fit(X[tr], y[tr], l2=l2)
    rep = {"model": m.info, "requests": len(kept), "train_requests": len(train_g), "test_requests": len(test_g),
           "points": int(len(y)), "positive_share": round(float(y.mean()), 4), "dim": int(X.shape[1]),
           "labels": "file" if table is not None else {"proxy_tail": tail, "proxy_frac": frac}}
    evalset = te if te.any() else tr
    rep["evaluated_on"] = "held-out requests" if te.any() else "the training requests (too few requests to hold any out)"
    s = m.predict(X[evalset])
    rep["auc"] = auc(y[evalset], s)
    if m.link == LOGISTIC:
        rep["brier"] = round(float(np.mean((s - y[evalset]) ** 2)), 5)
        rep["log_loss"] = round(log_loss(y[evalset], s), 5)
        rep["ece"], rep["reliability"] = calibration(y[evalset], s)
        rep["ece"] = round(rep["ece"], 5)
    groups = []
    for gi, (d, msk, lab) in enumerate(kept):
        if (gi in test_g) or not te.any():
            sc = m.predict(d.h)
            groups.append((d.think_index, sc, np.where(lab >= 0, lab, 1), d.flags, d.end_index))
    if thresholds is None:
        thresholds = np.round(np.linspace(0.5, 0.99, 50), 3) if m.link == LOGISTIC else \
            np.quantile(m.predict(X[evalset]), np.linspace(0.5, 0.99, 50))
    sweep = [simulate_stop(groups, t, k, min_tokens) for t in thresholds]
    rep["stop_rule"] = {"k": k, "min_tokens": min_tokens, "sweep": sweep}
    rep["recommended_threshold"] = pick_threshold(sweep, risk, delta)
    rep["risk"], rep["delta"] = risk, delta
    return m, rep


def print_report(rep):
    print(f"requests {rep['requests']} (train {rep['train_requests']}, test {rep['test_requests']}), points {rep['points']}, "
          f"positive share {rep['positive_share']}, dim {rep['dim']}, labels {rep['labels']}")
    print(f"model {rep['model']}; evaluated on {rep['evaluated_on']}")
    line = f"AUC {rep['auc']}" if rep["auc"] is not None else "AUC n/a (one class)"
    if "brier" in rep:
        line += f"  Brier {rep['brier']}  log loss {rep['log_loss']}  ECE {rep['ece']}"
    print(line)
    for r in rep.get("reliability", []):
        print(f"   {r['bin']}: n={r['n']:5d} mean score {r['mean_score']:.3f}  positive rate {r['positive_rate']:.3f}")
    print(f"stop rule k={rep['stop_rule']['k']} min={rep['stop_rule']['min_tokens']}:")
    for r in rep["stop_rule"]["sweep"][::5]:
        print(f"   thr {r['threshold']:.3f}: fires {r['fires']:4d}, premature {r['premature']:4d}, saved {r['saved_tokens']} of "
              f"{r['thinking_tokens']} thinking tokens ({r['saved_share']})")
    print(f"recommended threshold (premature-rate bound <= {rep['risk']} at delta {rep['delta']}): {rep['recommended_threshold']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("info", help="what the dumps hold")
    pi.add_argument("paths", nargs="+")
    pt = sub.add_parser("train", help="train, calibrate and export a probe")
    pt.add_argument("paths", nargs="+")
    pt.add_argument("--labels", help="ground-truth JSONL ({file, think_index, label}); default: the cheap proxy")
    pt.add_argument("--proxy-tail", type=int, default=128)
    pt.add_argument("--proxy-frac", type=float, default=0.9)
    pt.add_argument("--model", choices=("logreg", "ridge"), default="logreg")
    pt.add_argument("--l2", type=float, default=1.0)
    pt.add_argument("--holdout", type=float, default=0.25, help="share of requests held out (by file)")
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--k", type=int, default=3, help="checks in a row (the engine's k=)")
    pt.add_argument("--min", type=int, default=0, dest="min_tokens", help="thinking tokens before a fire (min=)")
    pt.add_argument("--risk", type=float, default=0.1, help="premature-fire rate bound for the recommended threshold")
    pt.add_argument("--delta", type=float, default=0.1)
    pt.add_argument("--out", help="write the SPRB probe here")
    pt.add_argument("--report", help="write the JSON report here")
    pp = sub.add_parser("labels-plan", help="the forced-stop replay plan for ground-truth labels")
    pp.add_argument("paths", nargs="+")
    pp.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    dumps = [read_dump(p) for p in find_dumps(a.paths)]
    if not dumps:
        raise SystemExit("thinking_probe: no dumps found")
    if a.cmd == "info":
        ends = [d.end_index for d in dumps if d.end_index is not None]
        print(f"{len(dumps)} requests, {sum(len(d.think_index) for d in dumps)} records, dim {sorted({d.dim for d in dumps})}, "
              f"every {sorted({d.every for d in dumps})}")
        print(f"natural closes {len(ends)}" + (f": thinking tokens p50 {int(np.median(ends))} max {max(ends)}" if ends else ""))
        fired = [d.name for d in dumps if (d.flags & FIRED).any()]
        print(f"probe fired in {len(fired)} requests; .ids present for {sum(d.ids is not None for d in dumps)}")
        return 0
    if a.cmd == "labels-plan":
        plan = replay_plan(dumps)
        with open(a.out, "w", encoding="utf-8") as f:
            for r in plan:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        print(f"{len(plan)} check points to replay -> {a.out}")
        return 0
    table = load_label_file(a.labels) if a.labels else None
    m, rep = train(dumps, table, a.model, a.l2, a.holdout, a.seed, a.k, a.min_tokens, a.risk, a.delta, a.proxy_tail, a.proxy_frac)
    print_report(rep)
    if a.out:
        wf, bf = m.folded()
        write_probe(a.out, wf, bf, m.link)
        thr = rep["recommended_threshold"]
        print(f"probe -> {a.out}; engine: STRATA_THINK_PROBE={a.out},{thr if thr is not None else '<threshold>'},k={a.k}"
              + (f",min={a.min_tokens}" if a.min_tokens else ""))
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
