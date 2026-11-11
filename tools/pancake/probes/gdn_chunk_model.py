#!/usr/bin/env python3
"""gdn_chunk_model.py - MOD_PLAN A31 numerical model: chunk 16 (today's gfx11 WMMA scan) against the chunk-64 design with a
blockwise 16x16 inverse, both against the FP64 token-by-token gated delta rule.

WHY        The A31 design (pancake-docs/batch2/A37-gpu-kernels.md) replaces T = (I + A)^-1 over 16 tokens by a blockwise inverse over
           64: T_bb = (I + A_bb)^-1 and T_ba = -T_bb * sum_{a <= c < b} A_bc T_ca.  This checks, on the CPU and before any kernel is
           written, that the algebra is exact and that FP16 operands (k, q, T, P, D, state; the kernel's WMMA inputs) with the
           inverse itself done in FP16 stay in the chunk-16 error class.  It says nothing about speed (that is the stage bench).
MODEL      One head, S = 128, unit-norm k and q, beta in (0, 1), log gate <= 0.  The recurrence, with St the dk x dv state:
               St = a (St - beta k (k^T St)) + beta k v^T,  o = St^T q,  a = exp(gate)
           and the chunked form of src/prefill/kernels.cu (gdn_wmma_prep_kernel / gdn_wmma_scan_kernel comment).
RUN        python tools/pancake/probes/gdn_chunk_model.py          (about 2 s)
"""
import numpy as np

S = 128


def f16(x):
    return x.astype(np.float16).astype(np.float64)


def make(n, rng, decay=0.5):
    k = rng.standard_normal((n, S)); k /= np.linalg.norm(k, axis=1, keepdims=True)
    q = rng.standard_normal((n, S)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    v = rng.standard_normal((n, S))
    beta = 1 / (1 + np.exp(-rng.standard_normal(n)))
    gate = -(0.01 + decay * rng.random(n))
    return k, q, v, beta, gate


def sequential(k, q, v, beta, gate):
    st, o = np.zeros((S, S)), np.zeros((len(k), S))
    for t in range(len(k)):
        st = np.exp(gate[t]) * (st - beta[t] * np.outer(k[t], k[t] @ st)) + beta[t] * np.outer(k[t], v[t])
        o[t] = st.T @ q[t]
    return o, st


def inverse(a, mode, blk=16):
    """(I + a)^-1 for strictly lower a: 'exact', 'block' (blockwise, FP64) or 'block16' (blockwise, FP16 operands and results)."""
    n = a.shape[0]
    eye = np.eye(n)
    if mode == "exact":
        return np.linalg.inv(eye + a)
    r = f16 if mode == "block16" else (lambda x: x)
    t = np.zeros((n, n))
    for b in range(n // blk):
        sb = slice(b * blk, (b + 1) * blk)
        t[sb, sb] = r(np.linalg.inv(eye[sb, sb] + a[sb, sb]))
        for c0 in range(b):
            sa = slice(c0 * blk, (c0 + 1) * blk)
            acc = r(sum(r(a[sb, slice(c * blk, (c + 1) * blk)]) @ t[slice(c * blk, (c + 1) * blk), sa] for c in range(c0, b)))
            t[sb, sa] = r(-t[sb, sb] @ acc)
    return t


def chunked(k, q, v, beta, gate, chunk, mode="exact", fp16=True):
    r = f16 if fp16 else (lambda x: x)
    st, o = np.zeros((S, S)), np.zeros((len(k), S))
    for t0 in range(0, len(k), chunk):
        sl = slice(t0, t0 + chunk)
        K, Q, V, b = k[sl], q[sl], v[sl], beta[sl]
        gm = np.cumsum(gate[sl])
        dec = np.exp(gm[:, None] - gm[None, :])
        a = np.tril(b[:, None] * dec * (r(K) @ r(K).T), -1)
        p = np.tril(dec * (r(Q) @ r(K).T))
        y = b[:, None] * (V - np.exp(gm)[:, None] * (r(K) @ r(st)))
        d = r(inverse(a, mode)) @ r(y)
        o[sl] = np.exp(gm)[:, None] * (r(Q) @ r(st)) + r(p) @ r(d)
        st = np.exp(gm[-1]) * st + r(K).T @ r(np.exp(gm[-1] - gm)[:, None] * d)
    return o, st


def compare(n=512, seed=7, decay=0.5):
    """Relative y and state errors against the FP64 sequential recurrence, for each variant."""
    rng = np.random.default_rng(seed)
    x = make(n, rng, decay)
    o0, s0 = sequential(*x)
    out = {}
    for name, chunk, mode, fp in (("c16 fp64", 16, "exact", False), ("c64 block fp64", 64, "block", False),
                                  ("c16 fp16", 16, "exact", True), ("c64 block fp16", 64, "block", True),
                                  ("c64 block16 fp16", 64, "block16", True)):
        o, st = chunked(*x, chunk, mode, fp)
        out[name] = (np.linalg.norm(o - o0) / np.linalg.norm(o0), np.linalg.norm(st - s0) / np.linalg.norm(s0))
    return out


if __name__ == "__main__":
    for decay in (0.5, 0.05):
        print(f"gate decay scale {decay}: relative error against FP64 sequential (y, state)")
        for name, (ey, es) in compare(decay=decay).items():
            print(f"  {name:18s} y {ey:.2e}  state {es:.2e}")
