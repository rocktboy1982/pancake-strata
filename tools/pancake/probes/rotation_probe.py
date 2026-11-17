#!/usr/bin/env python3
"""rotation_probe.py - LEADS.md Part L (L1/L4): what does a block-Hadamard rotation do to the error of a low-bit quantiser?

SILVERTHRONE32/llama.cpp-rdna3-int4 (MIT) rotates both operands of every linear layer, R = H_256 D / 16 per 256-block of K (D a
+-1 diagonal from a hash, so converter and backend agree without storing signs), because W'x' = Wx for W' = W R^T, x' = R x.  After
the rotation the 256 values of a block are close to Gaussian, so a 4-bit grid wastes less range on outlier channels; that is what lets
it run the INT4 WMMA path on activations at all.  Its numbers (Qwen3.5-4B, theirs, on the 7900 XT): 8-bit activations in two INT4
passes KLD 0.0241, 4-bit activations one pass KLD 0.0969; clip 0.85 of the block maximum lowered the 4-bit KLD 0.113 -> 0.092.

This probe measures only the ERROR SIDE of that idea on tensors YOU give it, with a round-to-nearest quantiser per group of
`--group` values (symmetric, scale = clip x max|x| / qmax).  It does NOT run GPTQ, does not model the IQ2_XS codebook of the daily
pack, and does not run the model: a smaller error here is a reason to try the idea on the box, never a result.

    python tools/pancake/probes/rotation_probe.py --weights W.npy --acts X.npy [--block 256] [--group 32]
                                                  [--wbits 2,3,4] [--abits 4,8] [--clips 1,0.95,0.9,0.85,0.8] [--json OUT.json]
    python tools/pancake/probes/rotation_probe.py --synthetic [--seed 0]       (the self-check data: Gaussian rows plus outlier channels)

W.npy float [rows, K], X.npy float [tokens, K] (K a multiple of --block); export them from the box (dequantized expert rows, or the
hidden states in front of a projection).  Output: per bit width the relative RMS error of the weights, of the activations and of the
product X W^T, plain vs rotated, the block kurtosis before and after (Gaussian = 3), and the best clip of the sweep.
Offline, numpy only.
"""
import argparse
import json
import sys

import numpy as np


def sign_diag(n):
    """The paper's hash: s_j = -1 if ((j * 2654435761) mod 2^32) >> 31 else +1."""
    j = np.arange(n, dtype=np.uint64)
    h = (j * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)
    return np.where((h >> np.uint64(31)) & np.uint64(1), -1.0, 1.0)


def hadamard(n):
    """Sylvester-Hadamard matrix, n a power of two."""
    assert n > 0 and n & (n - 1) == 0, "block must be a power of two"
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


def rotation(n=256):
    """R = H D / sqrt(n): orthogonal, so R R^T = I."""
    return hadamard(n) * sign_diag(n)[None, :] / np.sqrt(n)


def rotate(x, block=256):
    """x' = R x on every `block` of the last axis (rows of x are vectors; K must divide)."""
    k = x.shape[-1]
    assert k % block == 0, f"K={k} is not a multiple of the block {block}"
    r = rotation(block)
    return (x.reshape(*x.shape[:-1], k // block, block) @ r.T).reshape(x.shape)


def quantize(x, bits, group=32, clip=1.0):
    """Symmetric round-to-nearest per `group` along the last axis: codes in [-2^(b-1), 2^(b-1)-1], scale = clip*max|x|/qmax."""
    k = x.shape[-1]
    assert k % group == 0
    g = x.reshape(*x.shape[:-1], k // group, group)
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(g).max(axis=-1, keepdims=True) * clip / qmax
    scale = np.where(scale == 0, 1.0, scale)
    q = np.clip(np.rint(g / scale), -qmax - 1, qmax)
    return (q * scale).reshape(x.shape)


def rel_err(a, b):
    d = np.linalg.norm(a - b)
    n = np.linalg.norm(a)
    return float(d / n) if n else 0.0


def kurtosis(x, block=256):
    """Mean over blocks of the kurtosis of the block's values (Gaussian = 3)."""
    k = x.shape[-1]
    b = x.reshape(-1, block)
    m = b - b.mean(axis=1, keepdims=True)
    v = (m ** 2).mean(axis=1)
    v = np.where(v == 0, 1.0, v)
    return float(((m ** 4).mean(axis=1) / v ** 2).mean())


def probe(w, x, wbits=(2, 3, 4), abits=(4, 8), clips=(1.0, 0.95, 0.9, 0.85, 0.8), block=256, group=32):
    """-> dict of rows; every figure is a relative RMS error against the float product / tensor."""
    w = np.asarray(w, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    wr, xr = rotate(w, block), rotate(x, block)
    ref = x @ w.T
    out = {"kurtosis": {"weights_plain": kurtosis(w, block), "weights_rot": kurtosis(wr, block),
                        "acts_plain": kurtosis(x, block), "acts_rot": kurtosis(xr, block)},
           "rotation_check": rel_err(ref, xr @ wr.T), "weights": {}, "acts": {}, "product": {}}
    for b in wbits:
        out["weights"][b] = {"plain": rel_err(w, quantize(w, b, group)), "rot": rel_err(wr, quantize(wr, b, group))}
    for b in abits:
        best = {}
        for tag, xx in (("plain", x), ("rot", xr)):
            errs = {c: rel_err(xx, quantize(xx, b, group, c)) for c in clips}
            c_best = min(errs, key=errs.get)
            best[tag] = {"err_clip1": errs.get(1.0, errs[c_best]), "best_clip": c_best, "err_best": errs[c_best]}
        out["acts"][b] = best
    for wb in wbits:
        for ab in abits:
            wp, wq = w, quantize(w, wb, group)
            wrq = quantize(wr, wb, group)
            out["product"][f"w{wb}a{ab}"] = {
                "plain": rel_err(ref, quantize(x, ab, group) @ wq.T),
                "rot": rel_err(ref, quantize(xr, ab, group) @ wrq.T)}
    return out


def synthetic(rows=256, tokens=64, k=1024, seed=0, n_outliers=8, outlier_scale=20.0):
    """Gaussian weights; activations with a few huge channels (the shape that makes 4-bit activations unusable without rotation)."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((rows, k))
    x = rng.standard_normal((tokens, k))
    idx = rng.choice(k, n_outliers, replace=False)
    x[:, idx] *= outlier_scale
    return w, x


def render(res):
    lines = ["kurtosis (Gaussian = 3): " + ", ".join(f"{k}={v:.2f}" for k, v in res["kurtosis"].items()),
             f"rotation check, relative error of X W^T vs (RX)(RW)^T in float: {res['rotation_check']:.2e} (must be ~1e-15)",
             "weights   plain      rot"]
    for b, r in res["weights"].items():
        lines.append(f"  w{b}      {r['plain']:.4f}    {r['rot']:.4f}")
    lines.append("activations   plain (best clip)       rot (best clip)")
    for b, r in res["acts"].items():
        lines.append(f"  a{b}      {r['plain']['err_best']:.4f} (c={r['plain']['best_clip']})      {r['rot']['err_best']:.4f} (c={r['rot']['best_clip']})")
    lines.append("product X W^T   plain     rot     ratio rot/plain")
    for k, r in res["product"].items():
        lines.append(f"  {k}     {r['plain']:.4f}    {r['rot']:.4f}    {r['rot'] / r['plain'] if r['plain'] else float('nan'):.3f}")
    lines.append("proxy only: RTN per group, no GPTQ, no IQ2_XS codebook, no model run; a lower rotated error is a reason to test on the box")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--weights")
    ap.add_argument("--acts")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--group", type=int, default=32)
    ap.add_argument("--wbits", default="2,3,4")
    ap.add_argument("--abits", default="4,8")
    ap.add_argument("--clips", default="1,0.95,0.9,0.85,0.8")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.synthetic:
        w, x = synthetic(seed=a.seed)
    elif a.weights and a.acts:
        w, x = np.load(a.weights), np.load(a.acts)
    else:
        ap.error("give --weights and --acts, or --synthetic")
    ints = lambda s: tuple(int(v) for v in s.split(","))
    res = probe(w, x, ints(a.wbits), ints(a.abits), tuple(float(v) for v in a.clips.split(",")), a.block, a.group)
    print(render(res))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
