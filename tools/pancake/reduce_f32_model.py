#!/usr/bin/env python3
"""reduce_f32_model.py - host emulation of STRATA_REDUCE_F32=1 (LEADS.md Part U, lead U2; include/strata/kernels/reduce_f32.hpp): the f32 replacement of each
f64 device kernel against its f64 original, on the host, with the exact operation ORDER of the device code and correctly rounded f32 operations (numpy
float32 is IEEE: mul, add, div and sqrt are each rounded once).

What this proves and what it does not (the engine is judged by exactness, so each kernel says which):
  * silu (elementwise.cu silu_kernel): the f32 form needs expf, and the device expf (ocml) is not the host's.  Two models bracket it: a correctly rounded expf
    (f64 exp rounded to f32) and an expf that is 1 ulp off in either direction (the documented ocml bound).  Over EVERY finite f32 input (--exhaustive; 2^32
    patterns) the f32 result is NOT bit-equal to the f64 original in f32 (counts below), so it is not "exact in f32"; for inputs that are bf16 values (65,280
    finite) and f16 values, the same comparison after a bf16 / f16 store is reported (this is the radiance-rdna3 proof's form: their input was a bf16 value
    and their output a bf16 store).  `exact` is claimed ONLY where the count is 0 for ALL THREE expf models; otherwise the number is a bound, never "exact".
    The device run of the same sweep on the real expf is probes/reduce_f32_parity.hip.cpp (box).
  * the norms and dot sums (gdn_l2, gdn_out_norm, ple gnorm / gate / bf16_gemv, qsa indexer norm): a f32 sum of non-negative squares is not the f64 sum.  These are
    NOT exact and cannot be proven exact by a sweep (the input space is rows); the tool reports, on random and worst-case rows, the largest relative error of the
    inverse norm, the largest ulp distance of the OUTPUT elements, the fraction of output elements whose f32 bits differ, and the analytic bound; the quality
    gate on the box is the judge.  Known domain limit: a square overflows f32 for |x| > 1.8e19 (the f64 original does not); activations are far below.
Commands:  python tools/pancake/reduce_f32_model.py silu [--exhaustive] | norms | all
Tests: python -m pytest -q tools/pancake/test_reduce_f32_model.py
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
F32 = np.float32


# ----------------------------------------------------------------------------------------------------------------- bit helpers
def ordered(a32: np.ndarray) -> np.ndarray:
    """float32 values -> int64 on a monotonic scale, so the difference of two is their distance in ulps (+0 and -0 are 0)."""
    u = a32.view(np.uint32).astype(np.int64)
    return np.where(u >= 0x80000000, -(u - 0x80000000), u)


def ulps(a32, b32) -> np.ndarray:
    return np.abs(ordered(np.asarray(a32, F32)) - ordered(np.asarray(b32, F32)))


def to_bf16_bits(x32: np.ndarray) -> np.ndarray:
    """f32 -> bf16 bits, round to nearest even (finite and infinite inputs; no NaN in these sweeps)."""
    u = x32.view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32)
    return r


# ----------------------------------------------------------------------------------------------------------------- silu
def silu_f64(x32: np.ndarray) -> np.ndarray:
    """The original: v = (double) x; (float)(v / (1 + exp(-v)))."""
    v = x32.astype(np.float64)
    return (v / (1.0 + np.exp(-v))).astype(F32)


def _expf(model: str, arg64: np.ndarray) -> np.ndarray:
    e = np.exp(arg64).astype(F32)   # correctly rounded to within double rounding of the f64 exp
    if model == "cr":
        return e
    # one ulp off = a relative change of 2^-23 (a multiplicative step, so 0 and infinity stay what they are: a "1 ulp" step of 0 would be a denormal)
    if model == "plus":
        return (e.astype(np.float64) * (1.0 + 2.0 ** -23)).astype(F32)
    if model == "minus":
        return (e.astype(np.float64) * (1.0 - 2.0 ** -23)).astype(F32)
    raise ValueError(model)


def silu_f32(x32: np.ndarray, model: str = "cr") -> np.ndarray:
    """reduce_f32.hpp rf32::silu: x < -87 ? x * expf(x) : x / (1 + expf(-x)), every operation rounded to f32."""
    x = np.asarray(x32, F32)
    v = x.astype(np.float64)
    big = x < F32(-87.0)
    e_neg = _expf(model, -v)            # expf(-x)
    e_pos = _expf(model, v)             # expf(x)
    plain = x / (F32(1.0) + e_neg)
    tail = x * e_pos
    return np.where(big, tail, plain).astype(F32)


def silu_stats(x32: np.ndarray, models=("cr", "plus", "minus")) -> dict:
    ref = silu_f64(x32)
    ref16 = to_bf16_bits(ref)
    out = {"n": int(x32.size)}
    for m in models:
        got = silu_f32(x32, m)
        d = ulps(ref, got)
        out[m] = {
            "f32_mismatch": int(np.count_nonzero(ref != got)),
            "max_ulp": int(d.max()) if d.size else 0,
            "bf16_store_mismatch": int(np.count_nonzero(to_bf16_bits(got) != ref16)),
            "f16_store_mismatch": int(np.count_nonzero(got.astype(np.float16).view(np.uint16) != ref.astype(np.float16).view(np.uint16))),
            "max_abs_err": float(np.max(np.abs(got.astype(np.float64) - ref.astype(np.float64)))) if d.size else 0.0,
        }
    return out


def bf16_values() -> np.ndarray:
    """Every finite bf16 value as f32 (65,280 of the 65,536 patterns; the NaN and infinity patterns are left out)."""
    b = np.arange(65536, dtype=np.uint32) << 16
    x = b.view(F32)
    return x[np.isfinite(x)]


def f16_values() -> np.ndarray:
    h = np.arange(65536, dtype=np.uint16).view(np.float16).astype(F32)
    return h[np.isfinite(h)]


def merge_stats(a: dict, b: dict) -> dict:
    out = {"n": a["n"] + b["n"]}
    for m in ("cr", "plus", "minus"):
        out[m] = {k: (max(a[m][k], b[m][k]) if k.startswith("max") else a[m][k] + b[m][k]) for k in a[m]}
    return out


def silu_exhaustive(chunk_bits: int = 24, step: int = 1, progress=False) -> dict:
    """All finite f32 bit patterns (step 1) or every `step`-th one."""
    total = None
    n_chunks = 1 << (32 - chunk_bits)
    for c in range(n_chunks):
        u = (np.arange(1 << chunk_bits, dtype=np.uint64)[::step] + (np.uint64(c) << np.uint64(chunk_bits))).astype(np.uint32)
        x = u.view(F32)
        x = x[np.isfinite(x)]
        s = silu_stats(x)
        total = s if total is None else merge_stats(total, s)
        if progress:
            print(f"chunk {c + 1}/{n_chunks}", file=sys.stderr, flush=True)
    return total


# ----------------------------------------------------------------------------------------------------------------- reductions
def _tree32(vals: list) -> np.float32:
    """The shuffle tree over a warp: lane 0 after offsets 16, 8, 4, 2, 1 (lane l adds lane l + off; a lane past the end adds its own value, unused)."""
    v = [F32(x) for x in vals]
    off = 16
    while off > 0:
        v = [F32(v[i] + (v[i + off] if i + off < 32 else v[i])) for i in range(32)]
        off >>= 1
    return v[0]


def _tree64(vals: list) -> np.float64:
    v = [np.float64(x) for x in vals]
    off = 16
    while off > 0:
        v = [v[i] + (v[i + off] if i + off < 32 else v[i]) for i in range(32)]
        off >>= 1
    return v[0]


def warp_sumsq(x32: np.ndarray, f32: bool):
    """Thread t sums x[t], x[t + 32], ... squares in order, then the warp tree (gdn.cu: blockDim 32)."""
    parts = []
    for t in range(32):
        if f32:
            acc = F32(0)
            for i in range(t, x32.size, 32):
                acc = F32(acc + F32(x32[i] * x32[i]))   # fadd(acc, fmul(x, x))
        else:
            acc = np.float64(0)
            for i in range(t, x32.size, 32):
                acc += np.float64(x32[i]) * np.float64(x32[i])
        parts.append(acc)
    return _tree32(parts) if f32 else _tree64(parts)


def l2_scale(x32: np.ndarray, eps: float, f32: bool) -> np.float32:
    s = warp_sumsq(x32, f32)
    if f32:
        return F32(F32(1.0) / np.sqrt(F32(s + F32(eps))))   # __fdiv_rn(1, __fsqrt_rn(s + eps))
    return F32(1.0 / np.sqrt(s + np.float64(F32(eps))))


def out_norm_scale(o32: np.ndarray, eps: float, f32: bool) -> np.float32:
    s = warp_sumsq(o32, f32)
    S = o32.size
    if f32:
        return F32(F32(1.0) / np.sqrt(F32(F32(s / F32(S)) + F32(eps))))
    return F32(1.0 / np.sqrt(s / np.float64(S) + np.float64(F32(eps))))


def block_sumsq(x32: np.ndarray, threads: int, f32: bool):
    """ple.cu gnorm: thread t strides over the row, a warp tree, per-warp sums, the first warp reduces them (lanes >= nw add 0)."""
    nw = (threads + 31) // 32
    warp_sums = []
    for w in range(nw):
        parts = []
        for lane in range(32):
            t = w * 32 + lane
            acc = F32(0) if f32 else np.float64(0)
            for i in range(t, x32.size, threads):
                acc = F32(acc + F32(x32[i] * x32[i])) if f32 else acc + np.float64(x32[i]) * np.float64(x32[i])
            parts.append(acc)
        warp_sums.append(_tree32(parts) if f32 else _tree64(parts))
    zero = F32(0) if f32 else np.float64(0)
    return _tree32(warp_sums + [zero] * (32 - nw)) if f32 else _tree64(warp_sums + [zero] * (32 - nw))


def gnorm_mean(x32: np.ndarray, threads: int, f32: bool) -> np.float32:
    s = block_sumsq(x32, threads, f32)
    n = x32.size
    return F32(s / F32(n)) if f32 else F32(s / np.float64(n))


def seq_sumsq(x32: np.ndarray, f32: bool):
    """qsa.cu: every thread sums the whole row in index order (dadd/dmul or fadd/fmul)."""
    if f32:
        acc = F32(0)
        for v in x32:
            acc = F32(acc + F32(v * v))
        return acc
    acc = np.float64(0)
    for v in x32:
        acc = acc + np.float64(v) * np.float64(v)
    return acc


def qsa_inv(x32: np.ndarray, eps: float, f32: bool):
    n = x32.size
    ss = seq_sumsq(x32, f32)
    if f32:
        return F32(F32(1.0) / np.sqrt(F32(F32(ss / F32(n)) + F32(eps))))
    return F32(1.0 / np.sqrt(ss / np.float64(n) + np.float64(F32(eps))))


def dot_seq(a32: np.ndarray, b32: np.ndarray, f32: bool):
    """bf16_gemv (the bf16 x bf16 products are exact in f32) and gate_kernel's key . query: sequential in the original per thread; here the bf16 gemv order."""
    if f32:
        acc = F32(0)
        for x, y in zip(a32, b32):
            acc = F32(acc + F32(x * y))
        return acc
    acc = np.float64(0)
    for x, y in zip(a32, b32):
        acc += np.float64(x) * np.float64(y)
    return F32(acc)


def rows(kind: str, rng: np.random.Generator, n: int, count: int) -> list:
    """Random and worst-case rows of n values."""
    out = []
    for _ in range(count):
        out.append(rng.standard_normal(n).astype(F32))
        out.append((rng.standard_normal(n) * np.exp(rng.uniform(-8, 8, n))).astype(F32))   # wide dynamic range
    one = np.ones(n, F32)
    big = np.zeros(n, F32)
    big[0] = F32(1e4)
    big[1:] = F32(1e-3)
    out += [one, one * F32(1e4), one * F32(1e-4), big, big[::-1].copy(), np.where(np.arange(n) % 2 == 0, F32(3e3), F32(1e-2)).astype(F32)]
    return out


def norm_report(seed: int = 0, count: int = 24) -> dict:
    """Per kernel: max relative error of the scale/inverse, max ulp and the mismatch fraction of the output elements, vs the f64 original."""
    rng = np.random.default_rng(seed)
    res = {}
    eps = 1e-6

    def acc_case(name, fn, dim, y_fn):
        rel, ulp_max, diff, tot = 0.0, 0, 0, 0
        for x in rows(name, rng, dim, count):
            a, b = fn(x, False), fn(x, True)
            rel = max(rel, float(abs(np.float64(b) - np.float64(a)) / abs(np.float64(a))))
            ya, yb = y_fn(x, a), y_fn(x, b)
            d = ulps(ya, yb)
            ulp_max = max(ulp_max, int(d.max()))
            diff += int(np.count_nonzero(ya != yb))
            tot += ya.size
        res[name] = {"max_rel_err_scale": rel, "max_ulp_out": ulp_max, "frac_out_differs": diff / tot}

    acc_case("gdn_l2 (cols 128, warp 32)", lambda x, f: l2_scale(x, eps, f), 128, lambda x, s: (x * s).astype(F32))
    acc_case("gdn_out_norm (S 128, warp 32)", lambda x, f: out_norm_scale(x, eps, f), 128, lambda x, s: (x * s).astype(F32))
    acc_case("ple gnorm (n_embd 2560, 256 threads)",
             lambda x, f: F32(F32(1.0) / np.sqrt(F32(gnorm_mean(x, 256, f) + F32(eps)))), 2560, lambda x, s: (x * s).astype(F32))
    acc_case("qsa indexer norm (idx_dim 128, sequential)", lambda x, f: qsa_inv(x, eps, f), 128, lambda x, s: (x * s).astype(F32))
    # bf16 gemv: output is the dot itself
    rel, ulp_max, diff, tot = 0.0, 0, 0, 0
    for _ in range(count):
        x = to_bf16_vals(rng.standard_normal(2560))
        w = to_bf16_vals(rng.standard_normal(2560) * 0.05)
        a, b = dot_seq(x, w, False), dot_seq(x, w, True)
        rel = max(rel, float(abs(np.float64(b) - np.float64(a)) / max(abs(np.float64(a)), 1e-30)))
        d = int(ulps(np.array([a], F32), np.array([b], F32))[0])
        ulp_max = max(ulp_max, d)
        diff += int(a != b)
        tot += 1
    res["ple bf16_gemv (n_in 2560, sequential)"] = {"max_rel_err_scale": rel, "max_ulp_out": ulp_max, "frac_out_differs": diff / tot}
    return res


def to_bf16_vals(x64: np.ndarray) -> np.ndarray:
    x = x64.astype(F32)
    b = to_bf16_bits(x).astype(np.uint32) << 16
    return b.view(F32)


def analytic_bound_ulps(terms_per_thread: int, tree_levels: int = 5) -> float:
    """Worst-case relative error of a non-negative f32 sum = (k + 1) u with k = terms in a chain + tree levels (+1 for the square); the inverse square root
    halves it, then 1.5 ulp for the sqrt, the divide and the multiply; reported in ulps (u = 2^-24 is half an ulp)."""
    k = terms_per_thread + tree_levels + 1
    return (k * 0.5) / 2 + 1.5 + 0.5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("what", choices=("silu", "norms", "all"))
    ap.add_argument("--exhaustive", action="store_true", help="every finite f32 pattern (HOURS in numpy: the device sweep of probes/reduce_f32_parity.hip.cpp is the real one)")
    ap.add_argument("--step", type=int, default=4099, help="without --exhaustive: every STEP-th f32 pattern")
    a = ap.parse_args(argv)
    if a.what in ("silu", "all"):
        print("silu, bf16-valued inputs (65,280):", silu_stats(bf16_values()))
        print("silu, f16-valued inputs:", silu_stats(f16_values()))
        s = silu_exhaustive(step=1 if a.exhaustive else a.step, progress=a.exhaustive)
        print("silu, all finite f32 patterns" if a.exhaustive else f"silu, every {a.step}th f32 pattern", s)
    if a.what in ("norms", "all"):
        for k, v in norm_report().items():
            print(k, v)
        print("analytic bound, gdn (4 terms a thread): %.1f ulp; ple gnorm (10 terms): %.1f ulp; qsa (128 sequential): %.1f ulp"
              % (analytic_bound_ulps(4), analytic_bound_ulps(10), analytic_bound_ulps(128, 0)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
