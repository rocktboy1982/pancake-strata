#!/usr/bin/env python3
"""fit_lookahead.py - pancake-strata MOD_PLAN.md section 6 item 29: fit the trained per-layer look-ahead predictor.

    python tools/pancake/fit_lookahead.py <dir or trace.bin> --n-embd 2560 --n-layers 48 --n-expert 512 --out predictor.lkw
        [--epochs 6] [--lr 3e-3] [--l2 1e-5] [--holdout 0.2] [--budget 8] [--min-pairs 64] [--seed 0] [--json OUT.json]

Input: the trace STRATA_PRED_DUMP=<dir> wrote (<dir>/trace.bin, RPD2 records; include/strata/program/lookahead_weights.hpp):
per window and layer the MoE input rows, the routed experts and the residency.  A training pair is (the MoE input of layer L - 1,
the experts layer L routed) of the same window and token: exactly what the look-ahead has when it predicts layer L.
Model: per layer L a softmax regression  score = W x + b  (n_expert x n_embd), cross-entropy against the routed set (each routed
expert 1/k), Adam, inputs standardised per dimension and the standardisation folded back into W and b at export.  Offline, numpy only.

Output: an LKW1 file STRATA_LOOKAHEAD_WEIGHTS reads (bf16 W + f32 bias per layer; entry L predicts layer L; entry 0 is zeros).  A layer
with fewer than --min-pairs pairs keeps W = 0 and a bias of the log routing frequency (or zeros with no pair at all).
Printed per layer, on the held-out windows (the last --holdout share of the windows): slot overlap of the predictor's top-k with the
routed set (k = the trace's k), of a popularity baseline (the training frequency), and the share of the NON-RESIDENT routed experts
that the top --budget non-resident experts cover (the number the engine's STRATA_LOOKAHEAD_STATS calls recall).  No result is claimed
without a trace: the numbers come from the trace given.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

LKW_MAGIC = 0x31574B4C
RPD_MAGIC = 0x32445250


def to_bf16(a):
    """float32 array -> uint16 bf16 bits, round to nearest even (NaN kept a NaN)."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    u = a.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    nan = np.isnan(a)
    if nan.any():
        r[nan] = 0x7FC0
    return r


def from_bf16(b):
    return (np.asarray(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def record_bytes(n_tok, k, n_expert, n_embd):
    return 28 + n_tok * n_embd * 2 + n_tok * k * 2 + n_tok * k + (n_expert + 7) // 8


def write_record(f, layer, window, x, ids, res, resmap, n_expert):
    """Append one RPD2 record (the writer the engine implements; here for the tests and for converting other traces)."""
    n_tok, k = ids.shape
    f.write(struct.pack("<7I", RPD_MAGIC, layer, n_tok, k, window & 0xFFFFFFFF, window >> 32, n_expert))
    f.write(to_bf16(x).tobytes())
    f.write(np.asarray(ids, dtype="<i2").tobytes())
    f.write(np.asarray(res, dtype=np.uint8).tobytes())
    bits = np.zeros((n_expert + 7) // 8, dtype=np.uint8)
    for e in np.flatnonzero(resmap):
        bits[e >> 3] |= 1 << (e & 7)
    f.write(bits.tobytes())


def read_trace(path, n_embd):
    """Yield (layer, window, x[n_tok, n_embd] f32, ids[n_tok, k] int, res[n_tok, k] bool, resmap[n_expert] bool)."""
    with open(path, "rb") as f:
        while True:
            h = f.read(28)
            if len(h) < 28:
                return
            magic, layer, n_tok, k, wlo, whi, n_expert = struct.unpack("<7I", h)
            if magic != RPD_MAGIC:
                raise ValueError("%s: bad record magic %#x (is --n-embd right? a trace of another build?)" % (path, magic))
            rest = record_bytes(n_tok, k, n_expert, n_embd) - 28
            buf = f.read(rest)
            if len(buf) < rest:
                return   # a record cut by a killed run: ignored
            o = 0
            x = from_bf16(np.frombuffer(buf, dtype="<u2", count=n_tok * n_embd, offset=o)).reshape(n_tok, n_embd)
            o += n_tok * n_embd * 2
            ids = np.frombuffer(buf, dtype="<i2", count=n_tok * k, offset=o).reshape(n_tok, k).astype(np.int64)
            o += n_tok * k * 2
            res = np.frombuffer(buf, dtype=np.uint8, count=n_tok * k, offset=o).reshape(n_tok, k).astype(bool)
            o += n_tok * k
            bits = np.frombuffer(buf, dtype=np.uint8, count=(n_expert + 7) // 8, offset=o)
            resmap = ((bits[np.arange(n_expert) >> 3] >> (np.arange(n_expert) & 7)) & 1).astype(bool)
            yield layer, wlo | (whi << 32), x, ids, res, resmap


def collect_pairs(path, n_embd, n_layers):
    """-> {layer L: [(window, x of L-1, ids of L, res of L, resmap of L)]}, windows in file order."""
    cur_w, held = None, {}
    pairs = {}

    def flush():
        for L in sorted(held):
            if L >= 1 and (L - 1) in held and held[L - 1][0].shape[0] == held[L][0].shape[0]:
                pairs.setdefault(L, []).append((cur_w, held[L - 1][0], held[L][1], held[L][2], held[L][3]))

    for layer, w, x, ids, res, resmap in read_trace(path, n_embd):
        if layer >= n_layers:
            continue
        if w != cur_w:
            flush()
            cur_w, held = w, {}
        held[layer] = (x, ids, res, resmap)
    flush()
    return pairs


def _targets(ids, n_expert):
    y = np.zeros((ids.shape[0], n_expert), dtype=np.float32)
    for j in range(ids.shape[1]):
        e = ids[:, j]
        ok = (e >= 0) & (e < n_expert)
        y[np.flatnonzero(ok), e[ok]] += 1.0
    s = y.sum(1, keepdims=True)
    return y / np.maximum(s, 1.0)


def _topk(score, k):
    idx = np.argpartition(-score, k - 1, axis=1)[:, :k]
    return idx


def evaluate(W, b, X, ids, res, resmap_per_row, budget, freq):
    """-> (overlap, baseline overlap, non-resident recall); X rows pair with ids rows."""
    n, k = ids.shape
    score = X @ W.T + b
    pred = _topk(score, k)
    hit = sum(int(np.isin(ids[t], pred[t]).sum()) for t in range(n))
    base = _topk(np.tile(freq, (n, 1)), k)
    bhit = sum(int(np.isin(ids[t], base[t]).sum()) for t in range(n))
    # non-resident recall: predicted top-budget among the experts NOT resident in that row's window layer
    miss_total, miss_hit = 0, 0
    for t in range(n):
        masked = score[t].copy()
        masked[resmap_per_row[t]] = -np.inf
        bb = min(budget, int((~resmap_per_row[t]).sum()))
        if bb < 1:
            continue
        top = set(np.argpartition(-masked, bb - 1)[:bb].tolist())
        for j in range(k):
            if not res[t, j]:
                miss_total += 1
                miss_hit += int(ids[t, j] in top)
    return hit / float(n * k), bhit / float(n * k), (miss_hit / miss_total if miss_total else None), miss_total


def fit_layer(Xtr, ytr, n_expert, epochs, lr, l2, rng):
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-6
    Z = (Xtr - mu) / sd
    n, H = Z.shape
    W = np.zeros((n_expert, H), dtype=np.float32)
    b = np.log(ytr.mean(0) + 1e-6).astype(np.float32)
    mW, vW = np.zeros_like(W), np.zeros_like(W)
    mb, vb = np.zeros_like(b), np.zeros_like(b)
    t = 0
    bs = min(256, n)
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, bs):
            i = order[s:s + bs]
            z = Z[i] @ W.T + b
            z -= z.max(1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(1, keepdims=True)
            g = (p - ytr[i]) / len(i)
            gW = g.T @ Z[i] + l2 * W
            gb = g.sum(0)
            t += 1
            for (P, G, M, V) in ((W, gW, mW, vW), (b, gb, mb, vb)):
                M *= 0.9; M += 0.1 * G
                V *= 0.999; V += 0.001 * G * G
                P -= lr * (M / (1 - 0.9 ** t)) / (np.sqrt(V / (1 - 0.999 ** t)) + 1e-8)
    We = (W / sd).astype(np.float32)          # fold the standardisation: W (x - mu) / sd + b = We x + (b - We mu)
    be = (b - We @ mu).astype(np.float32)
    return We, be


def write_lkw(path, n_layers, n_expert, n_embd, layers):
    """layers: {L: (W f32 [n_expert, n_embd], bias f32 [n_expert])}; missing layers are written as zeros."""
    with open(path, "wb") as f:
        f.write(struct.pack("<4I", LKW_MAGIC, n_layers, n_expert, n_embd))
        for L in range(n_layers):
            W, b = layers.get(L, (np.zeros((n_expert, n_embd), np.float32), np.zeros(n_expert, np.float32)))
            f.write(to_bf16(W).astype("<u2").tobytes())
            f.write(np.asarray(b, dtype="<f4").tobytes())


def read_lkw(path):
    with open(path, "rb") as f:
        magic, n_layers, n_expert, n_embd = struct.unpack("<4I", f.read(16))
        if magic != LKW_MAGIC:
            raise ValueError("not an LKW1 file")
        out = {}
        for L in range(n_layers):
            W = from_bf16(np.frombuffer(f.read(n_expert * n_embd * 2), dtype="<u2")).reshape(n_expert, n_embd)
            b = np.frombuffer(f.read(n_expert * 4), dtype="<f4")
            out[L] = (W, b)
    return n_layers, n_expert, n_embd, out


def run(trace, n_embd, n_layers, n_expert, out, epochs=6, lr=3e-3, l2=1e-5, holdout=0.2, budget=8, min_pairs=64, seed=0, quiet=False):
    path = os.path.join(trace, "trace.bin") if os.path.isdir(trace) else trace
    pairs = collect_pairs(path, n_embd, n_layers)
    rng = np.random.default_rng(seed)
    layers, report = {}, {}
    for L in range(1, n_layers):
        pl = pairs.get(L, [])
        if not pl:
            continue
        nwin = len(pl)
        ntest = int(round(nwin * holdout))
        tr, te = pl[:nwin - ntest] if ntest < nwin else pl, pl[nwin - ntest:] if ntest else []
        def cat(items, idx):
            return np.concatenate([it[idx] for it in items], axis=0)
        Xtr, Ytr = cat(tr, 1), cat(tr, 2)
        ytr = _targets(Ytr, n_expert)
        freq = ytr.mean(0)
        if Xtr.shape[0] < min_pairs:
            layers[L] = (np.zeros((n_expert, n_embd), np.float32), np.log(freq + 1e-6).astype(np.float32))
            report[L] = {"pairs": int(Xtr.shape[0]), "fitted": False}
            continue
        W, b = fit_layer(Xtr, ytr, n_expert, epochs, lr, l2, rng)
        layers[L] = (W, b)
        r = {"pairs": int(Xtr.shape[0]), "fitted": True}
        if te:
            Xte, Yte, Rte = cat(te, 1), cat(te, 2), cat(te, 3)
            maps = np.concatenate([np.tile(it[4], (it[1].shape[0], 1)) for it in te], axis=0)
            ov, bov, mr, mt = evaluate(W, b, Xte, Yte, Rte, maps, budget, freq)
            r.update(test_pairs=int(Xte.shape[0]), overlap=ov, baseline_overlap=bov, nonresident_recall=mr, nonresident_slots=mt)
        report[L] = r
    write_lkw(out, n_layers, n_expert, n_embd, layers)
    if not quiet:
        print("layer  pairs  test  overlap  popularity  non-resident recall (budget %d)" % budget)
        for L in sorted(report):
            r = report[L]
            if "overlap" in r:
                mr = "n/a" if r["nonresident_recall"] is None else "%.3f" % r["nonresident_recall"]
                print("%5d %6d %5d   %.3f      %.3f      %s (%d slots)" % (L, r["pairs"], r["test_pairs"], r["overlap"], r["baseline_overlap"], mr, r["nonresident_slots"]))
            else:
                print("%5d %6d   %s" % (L, r["pairs"], "fitted, no held-out windows" if r["fitted"] else "too few pairs: bias only"))
        print("wrote %s (%d layers, %d fitted)" % (out, n_layers, sum(1 for r in report.values() if r["fitted"])))
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--n-embd", type=int, required=True)
    ap.add_argument("--n-layers", type=int, required=True)
    ap.add_argument("--n-expert", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--l2", type=float, default=1e-5)
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--min-pairs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rep = run(a.trace, a.n_embd, a.n_layers, a.n_expert, a.out, a.epochs, a.lr, a.l2, a.holdout, a.budget, a.min_pairs, a.seed)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({str(k): v for k, v in rep.items()}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
