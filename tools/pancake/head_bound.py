#!/usr/bin/env python3
"""head_bound.py - MOD_PLAN.md section 6 item 46 (HEAD_BOUND_PRUNE), offline half: how many head rows an EXACT two-stage argmax
could skip with a norm bound, on real hidden states.

The output head is a [vocab, n_embd] matrix W and greedy decode needs argmax_v (W_v . h).  Cauchy-Schwarz gives an upper bound
that costs no dot product: W_v . h <= ub_v = sum over column blocks b of |W_v,b| |h_b| (one block: |W_v| |h|; more blocks are
tighter, never looser).  Stage 1 ranks the rows by ub_v (a table lookup per row, V floats instead of V x n_embd weights),
scores the top `m` rows exactly and takes L = the best of them.  Stage 2 scores only the rows with ub_v >= L.  A row with
ub_v < L cannot beat L, so the pick is the exact argmax (no loss, unlike STRATA_HEAD_SHORTLIST).  This tool measures the share
of rows (= head bytes, one format) the two stages score, per m and per block count, and the oracle floor (L = the true max,
m = 0).  The decision of the item: code the kernel only if the realised share pruned is >= 30% (`--gate`) on real dumps.

Inputs (the same files head_cluster.py uses):
    ROWS.npy    float [vocab, n_embd]: the dequantized head rows (exported by whoever has the model).
    HIDDEN.npy  float [steps, n_embd]: the head's input rows.  Capture: STRATA_HEAD_DUMP=/tmp/h.hdmp on a decode run (verify.cpp
                writes the window's final hidden rows after each window; a dump holds draft rows too), then
                `python tools/pancake/head_bound.py dump2npy /tmp/h.hdmp /tmp/h.npy`.

    python tools/pancake/head_bound.py dump2npy H.hdmp H.npy
    python tools/pancake/head_bound.py eval ROWS.npy H.npy [--blocks 1,4,16] [--ms 16,64,256] [--gate 0.30] [--max-steps 512]
    python -m pytest -q tools/pancake/test_head_bound.py
"""
import argparse
import struct
import sys

import numpy as np

MAGIC = 0x504D4448   # "HDMP"


def read_dump(path):
    """STRATA_HEAD_DUMP file -> float32 [rows, n_embd] (every window's rows in order), and n_vocab."""
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 16:
        raise ValueError("dump too short")
    magic, ver, n_embd, n_vocab = struct.unpack_from("<IIII", raw, 0)
    if magic != MAGIC or ver != 1 or n_embd == 0:
        raise ValueError("not a STRATA_HEAD_DUMP file")
    off, rows = 16, []
    while off + 12 <= len(raw):
        t, _pos = struct.unpack_from("<Iq", raw, off)
        off += 12
        n = t * n_embd * 4
        if off + n > len(raw):
            break   # a window cut short by the end of the run
        rows.append(np.frombuffer(raw, dtype="<f4", count=t * n_embd, offset=off).reshape(t, n_embd))
        off += n
    if not rows:
        return np.zeros((0, n_embd), np.float32), n_vocab
    return np.concatenate(rows).astype(np.float32), n_vocab


def block_norms(rows, blocks):
    """|W_v,b| for `blocks` equal column blocks (the last takes the remainder): float64 [vocab, blocks], and the edges."""
    v, d = rows.shape
    blocks = max(1, min(int(blocks), d))
    edges = [d * i // blocks for i in range(blocks + 1)]
    out = np.empty((v, blocks))
    for b in range(blocks):
        seg = rows[:, edges[b]:edges[b + 1]].astype(np.float64)
        out[:, b] = np.sqrt((seg * seg).sum(1))
    return out, edges


def upper_bounds(wn, edges, h):
    """ub_v = sum_b |W_v,b| |h_b| for one hidden row h."""
    hn = np.array([np.linalg.norm(h[edges[b]:edges[b + 1]].astype(np.float64)) for b in range(len(edges) - 1)])
    return wn @ hn


def two_stage(rows, h, wn, edges, m):
    """One row.  Returns (argmax, scored rows) of the exact two-stage search with `m` first-stage rows (m 0: the oracle floor,
    L = the true max: only rows whose bound reaches it are scored, plus the max itself)."""
    ub = upper_bounds(wn, edges, h)
    if m <= 0:
        logits = rows.astype(np.float64) @ h.astype(np.float64)
        best = int(np.argmax(logits))
        return best, int((ub >= logits[best] - 1e-9 * max(1.0, abs(logits[best]))).sum())
    m = min(m, len(ub))
    first = np.argpartition(-ub, m - 1)[:m]
    s1 = rows[first].astype(np.float64) @ h.astype(np.float64)
    lb = float(s1.max())
    cand = np.nonzero(ub >= lb)[0]
    cand = cand[~np.isin(cand, first)]
    best, bv = int(first[int(np.argmax(s1))]), lb
    if len(cand):
        s2 = rows[cand].astype(np.float64) @ h.astype(np.float64)
        j = int(np.argmax(s2))
        if s2[j] > bv or (s2[j] == bv and int(cand[j]) < best):
            best, bv = int(cand[j]), float(s2[j])
    return best, int(m + len(cand))


def evaluate(rows, hidden, blocks=(1, 4, 16), ms=(16, 64, 256), max_steps=512):
    """-> list of dicts per (blocks, m): scored share mean / p50 / p90, exact (argmax equals the full head's on every row)."""
    rows = np.asarray(rows, np.float32)
    hidden = np.asarray(hidden, np.float32)[:max_steps]
    v = rows.shape[0]
    full = (hidden.astype(np.float64) @ rows.astype(np.float64).T).argmax(1)
    out = []
    for nb in blocks:
        wn, edges = block_norms(rows, nb)
        for m in (0,) + tuple(ms):
            scored, exact = [], True
            for i, h in enumerate(hidden):
                pick, n = two_stage(rows, h, wn, edges, m)
                scored.append(n / v)
                exact = exact and pick == int(full[i])
            s = np.array(scored)
            out.append({"blocks": int(nb), "m": int(m), "scored_mean": float(s.mean()), "scored_p50": float(np.median(s)),
                        "scored_p90": float(np.percentile(s, 90)), "pruned_mean": float(1 - s.mean()), "exact": bool(exact)})
    return out


def report(res, gate):
    lines = ["blocks  m(first stage)  scored mean  p50     p90     pruned   exact"]
    for r in res:
        tag = "oracle floor" if r["m"] == 0 else f"{r['m']:>5}"
        lines.append(f"{r['blocks']:>6}  {tag:>14}  {r['scored_mean']:10.1%}  {r['scored_p50']:6.1%}  {r['scored_p90']:6.1%}  "
                     f"{r['pruned_mean']:6.1%}  {'yes' if r['exact'] else 'NO'}")
    real = [r for r in res if r["m"] > 0]
    best = max(real, key=lambda r: r["pruned_mean"]) if real else None
    if best:
        verdict = (f"over the {gate:.0%} gate: the two-stage kernel is worth coding" if best["pruned_mean"] >= gate
                   else f"under the {gate:.0%} gate: do not code the kernel")
        lines.append(f"best realised (blocks {best['blocks']}, m {best['m']}): {best['pruned_mean']:.1%} of the head rows pruned, {verdict}")
        lines.append("(rows = bytes for one weight format; stage 1's bound table and the gather add cost this does not count)")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump2npy")
    d.add_argument("dump")
    d.add_argument("out")
    e = sub.add_parser("eval")
    e.add_argument("rows")
    e.add_argument("hidden")
    e.add_argument("--blocks", default="1,4,16")
    e.add_argument("--ms", default="16,64,256")
    e.add_argument("--gate", type=float, default=0.30)
    e.add_argument("--max-steps", type=int, default=512)
    a = ap.parse_args(argv)
    if a.cmd == "dump2npy":
        h, nv = read_dump(a.dump)
        np.save(a.out, h)
        print(f"{h.shape[0]} rows of {h.shape[1]} floats (head vocab {nv}) -> {a.out}")
        return 0
    rows, hidden = np.load(a.rows), np.load(a.hidden)
    if rows.ndim != 2 or hidden.ndim != 2 or rows.shape[1] != hidden.shape[1]:
        print("shapes: ROWS [vocab, n_embd] and HIDDEN [steps, n_embd] must share n_embd", file=sys.stderr)
        return 2
    res = evaluate(rows, hidden, tuple(int(x) for x in a.blocks.split(",")), tuple(int(x) for x in a.ms.split(",")), a.max_steps)
    print(report(res, a.gate))
    return 0 if all(r["exact"] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
