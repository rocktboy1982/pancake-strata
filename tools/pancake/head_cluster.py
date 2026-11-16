"""head_cluster.py - LEADS.md Part J (kindi shortlist output head), offline half only.

The output head is a [vocab, n_embd] matrix; greedy decode needs only argmax(head @ h).  The shortlist idea: cluster the
vocabulary rows, score the K cluster centres first (K x n_embd instead of vocab x n_embd), then exact-score only the rows of
the best `frac` of the clusters and take the argmax among those.  LOSSY: when the true top token sits in a cluster whose
centre scored low, the pick changes.  This tool does the clustering and measures the agreement; the engine side is NOT built
(no kernel gathers a cluster's rows, no STRATA_HEAD_SHORTLIST switch exists).

    python tools/pancake/head_cluster.py cluster ROWS.npy --k 256 --out TABLE.shl [--iters 12] [--seed 0]
    python tools/pancake/head_cluster.py eval TABLE.shl ROWS.npy HIDDEN.npy [--frac 0.13]

ROWS.npy: float [vocab, n_embd] (the dequantized head rows, exported by whoever has the model).  HIDDEN.npy: float
[steps, n_embd] final hidden states of real decode steps.  TABLE.shl (little endian): "SHL1", u32 vocab, u32 n_embd, u32 K,
i32 cluster[vocab], f32 centre[K, n_embd].  eval prints top-1 agreement with the full argmax and the scored-row share.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"SHL1"


def cluster(rows, k, iters=12, seed=0):
    """k-means on the L2-normalised rows (cosine-like), seeded, deterministic.  -> (assign int32 [V], centres f32 [k, D]).
    Empty clusters are re-seeded from the row farthest from its centre.  The centres are the mean RAW row of each cluster
    (what a dot product with the hidden state scores), not the normalised mean."""
    rows = np.asarray(rows, np.float32)
    v = rows.shape[0]
    k = max(1, min(int(k), v))
    unit = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
    rng = np.random.default_rng(seed)
    cen = np.empty((k, unit.shape[1]), np.float32)     # k-means++ seeding on the cosine distance
    cen[0] = unit[int(rng.integers(v))]
    dist = 1.0 - unit @ cen[0]
    for i in range(1, k):
        w = np.maximum(dist, 0.0) ** 2
        cen[i] = unit[int(rng.choice(v, p=w / w.sum())) if w.sum() > 0 else int(rng.integers(v))]
        dist = np.minimum(dist, 1.0 - unit @ cen[i])
    assign = np.zeros(v, np.int32)
    for _ in range(max(1, iters)):
        sim = unit @ cen.T
        assign = sim.argmax(1).astype(np.int32)
        best = sim.max(1)
        for c in range(k):
            m = assign == c
            if m.any():
                cen[c] = unit[m].mean(0)
            else:
                j = int(best.argmin())
                cen[c] = unit[j]
                assign[j] = c
                best[j] = 1.0
        cen /= np.maximum(np.linalg.norm(cen, axis=1, keepdims=True), 1e-12)
    centres = np.zeros((k, rows.shape[1]), np.float32)
    for c in range(k):
        m = assign == c
        centres[c] = rows[m].mean(0) if m.any() else cen[c]
    return assign, centres


def save_table(path, assign, centres):
    v, (k, d) = len(assign), centres.shape
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<III", v, d, k))
        f.write(np.asarray(assign, "<i4").tobytes())
        f.write(np.asarray(centres, "<f4").tobytes())


def load_table(path):
    b = Path(path).read_bytes()
    if b[:4] != MAGIC:
        raise ValueError("not a SHL1 table")
    v, d, k = struct.unpack_from("<III", b, 4)
    off = 16
    assign = np.frombuffer(b, "<i4", v, off).copy()
    centres = np.frombuffer(b, "<f4", k * d, off + 4 * v).reshape(k, d).copy()
    return assign, centres


def shortlist_argmax(h, rows, assign, centres, frac):
    """-> (token, scored_rows): argmax over the rows of the best ceil(frac x K) clusters by centre score (>= 1 cluster)."""
    k = centres.shape[0]
    n = max(1, min(k, int(np.ceil(frac * k - 1e-9))))
    top = np.argsort(-(centres @ h), kind="stable")[:n]
    ids = np.nonzero(np.isin(assign, top))[0]
    return int(ids[(rows[ids] @ h).argmax()]), len(ids)


def agreement(rows, hidden, assign, centres, frac):
    """-> (share of steps whose pick equals the full argmax, mean share of vocabulary rows scored (+ the centres' K rows))."""
    same, scored = 0, 0.0
    v = rows.shape[0]
    for h in hidden:
        t, n = shortlist_argmax(h, rows, assign, centres, frac)
        same += t == int((rows @ h).argmax())
        scored += (n + centres.shape[0]) / v
    return same / len(hidden), scored / len(hidden)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cluster")
    c.add_argument("rows")
    c.add_argument("--k", type=int, default=256)
    c.add_argument("--iters", type=int, default=12)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("table")
    e.add_argument("rows")
    e.add_argument("hidden")
    e.add_argument("--frac", type=float, default=0.13)
    a = ap.parse_args(argv)
    if a.cmd == "cluster":
        assign, centres = cluster(np.load(a.rows), a.k, a.iters, a.seed)
        save_table(a.out, assign, centres)
        sz = np.bincount(assign)
        print(f"{a.out}: vocab {len(assign)}, clusters {centres.shape[0]}, sizes min/max {sz.min()}/{sz.max()}")
    else:
        assign, centres = load_table(a.table)
        same, share = agreement(np.load(a.rows).astype(np.float32), np.load(a.hidden).astype(np.float32), assign, centres, a.frac)
        print(f"frac {a.frac}: top-1 agreement {same * 100:.2f}%, rows scored {share * 100:.1f}% of the vocabulary")


if __name__ == "__main__":
    sys.exit(main())
