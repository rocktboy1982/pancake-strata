#!/usr/bin/env python3
"""rotate_gptq.py - GPTQ in a block-Hadamard-rotated basis for ONE weight matrix (LEADS.md Part L L1; method of SILVERTHRONE32/llama.cpp-rdna3-int4,
MIT, Sachin Mahesh, report sections 3.1-3.2; README.md section 9).  Written here from the report's description, not copied from the fork's
gptq_rotate.py.  Offline, numpy only, one matrix at a time (the engine side is probes/int4_wmma.hip.cpp and include/strata/kernels/int4_rot.hpp).

    python tools/pancake/rotate_gptq.py --weights W.npy --acts X.npy --out Q.npz [--block 256] [--group 32] [--bits 4] [--no-rotate]

W.npy float [rows, K] (a dequantised weight, or a bf16 original), X.npy float [tokens, K] calibration activations in front of that
matrix (the authors: 128 sequences x 512 tokens of WikiText-2, Hessians from forward hooks).  Steps, as the report gives them:
  1. rotate both: W' = W R^T, X' = X R^T per `block` of K, R = H D / sqrt(block), D from the hash s_j = -1 if ((j * 2654435761) mod 2^32) >> 31;
  2. H = 2 X'^T X' / n, dead columns fixed, damping 0.01 of the mean diagonal, Cholesky of the inverse;
  3. GPTQ with error feedback in column blocks of 128, codes in [-2^(bits-1), 2^(bits-1) - 1], one scale per `group` columns chosen at the
     start of the group from the ERROR-COMPENSATED weights by an MSE search over 26 candidates 0.55 .. 1.05 x max|w| / qmax.
Output Q.npz: codes int8 [rows, K], scales float32 [rows, K / group], deq float32 [rows, K] (in the rotated basis), and the report line.
The honest limit: the uniform symmetric grid is the authors' int4; our daily IQ2_XS is a codebook, so this is the INT4 / Q2_0-shaped half of the
idea (LEADS.md Part L L3, L4).  Nothing here was run on a real model tensor."""
import argparse
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "probes"))
from rotation_probe import rotate  # noqa: E402


def _best_scale(w, qmax, cands=np.linspace(0.55, 1.05, 26)):
    """w [rows, group]: per row the scale among cands x max|w| / qmax with the least squared rounding error."""
    mx = np.abs(w).max(axis=1, keepdims=True)
    mx = np.where(mx == 0, 1.0, mx)
    best_e = np.full((w.shape[0], 1), np.inf)
    best_s = mx / qmax
    for c in cands:
        s = c * mx / qmax
        q = np.clip(np.rint(w / s), -qmax - 1, qmax)
        e = ((q * s - w) ** 2).sum(axis=1, keepdims=True)
        better = e < best_e
        best_e = np.where(better, e, best_e)
        best_s = np.where(better, s, best_s)
    return best_s


def gptq(w, x, bits=4, group=32, blocksize=128, damp=0.01):
    """-> (codes int8 [rows, K], scales [rows, K/group], deq [rows, K]); w [rows, K], x [n, K] in the SAME basis."""
    w = np.array(w, dtype=np.float64)
    rows, k = w.shape
    qmax = 2 ** (bits - 1) - 1
    h = 2.0 * (x.T @ x) / x.shape[0]
    dead = np.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    h[np.arange(k), np.arange(k)] += damp * np.mean(np.diag(h))
    hinv = np.linalg.cholesky(np.linalg.inv(h)).T            # upper Cholesky factor of H^-1
    codes = np.zeros((rows, k), dtype=np.int8)
    scales = np.zeros((rows, k // group), dtype=np.float32)
    deq = np.zeros_like(w)
    for i1 in range(0, k, blocksize):
        i2 = min(i1 + blocksize, k)
        w1 = w[:, i1:i2].copy()
        err1 = np.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % group == 0:
                s = _best_scale(w1[:, i:i + group] if i + group <= w1.shape[1] else
                                np.concatenate([w1[:, i:], w[:, i2:col + group]], axis=1), qmax)
                scales[:, col // group] = s[:, 0]
            sc = scales[:, col // group].astype(np.float64)[:, None]
            sc = np.where(sc == 0, 1.0, sc)
            q = np.clip(np.rint(w1[:, i:i + 1] / sc), -qmax - 1, qmax)
            dq = q * sc
            codes[:, col] = q[:, 0].astype(np.int8)
            deq[:, col] = dq[:, 0]
            e = (w1[:, i:i + 1] - dq) / hinv1[i, i]
            w1[:, i:] -= e @ hinv1[i:i + 1, i:]
            err1[:, i:i + 1] = e
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return codes, scales, deq


def rtn(w, bits=4, group=32):
    qmax = 2 ** (bits - 1) - 1
    rows, k = w.shape
    g = w.reshape(rows, k // group, group)
    mx = np.abs(g).max(axis=2, keepdims=True)
    s = np.where(mx == 0, 1.0, mx) / qmax
    return (np.clip(np.rint(g / s), -qmax - 1, qmax) * s).reshape(rows, k)


def out_err(w, deq, x):
    """relative error of the layer output X W^T."""
    ref = x @ w.T
    return float(np.linalg.norm(ref - x @ deq.T) / np.linalg.norm(ref))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--no-rotate", action="store_true")
    a = ap.parse_args(argv)
    w, x = np.load(a.weights).astype(np.float64), np.load(a.acts).astype(np.float64)
    if not a.no_rotate:
        w, x = rotate(w, a.block), rotate(x, a.block)
    codes, scales, deq = gptq(w, x, a.bits, a.group)
    e_g, e_r = out_err(w, deq, x), out_err(w, rtn(w, a.bits, a.group), x)
    np.savez(a.out, codes=codes, scales=scales, deq=deq.astype(np.float32))
    print(f"gptq output error {e_g:.4f} vs round-to-nearest {e_r:.4f} ({'rotated' if not a.no_rotate else 'plain'} basis, {a.bits}-bit, group {a.group})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
