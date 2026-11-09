"""sparsity_masks.py - pancake-strata MOD_PLAN item 13 (phase 13 P1c): would intra-expert activation sparsity hold?

Input: a raw dump of the CPU experts' activations, recorded by the engine with
    STRATA_ACT_DUMP_RAW=<file> STRATA_ACT_DUMP_LAYERS=4,12,24,36,44 [STRATA_ACT_DUMP_RAW_MAX=200000]
(records: int32 layer, int32 has_gate, int32 n, float gate[n] if has_gate, float h[n]; h = SiLU(gate) * up is the down
projection's input, so skipping an entry of h skips one up row and one down column of the expert).

For each sparsity s (share of h entries skipped) and mask, per layer and overall:
  oracle    keep the largest |h| (the bound no predictor beats; needs up, so it saves only the down columns)
  gate      Berkeley intra-expert gate mask: keep the largest |SiLU(gate)| per token (needs the gate rows only: saves
            the skipped up rows and down columns, ~2/3 of the expert's work at that sparsity)
  cats      CATS: one |SiLU(gate)| threshold per layer, calibrated on the first half of the records for sparsity s,
            measured on the second half (the realized sparsity varies per token)
Metrics: energy kept = sum of kept h^2 / sum h^2, and cos = sqrt(energy kept) (the masked vector is a projection of h).
Down is linear, so the output error of the expert is ||W_down (h - h_masked)||; with W_down's columns of similar norm and
weakly correlated, that is ~ sqrt(1 - energy kept) of the output.  Screen used here: cos >= 0.99 (relative error <= 14%)
at >= 70% sparsity for the gate mask -> build the sparse kernel (MOD_PLAN 13); token forcing decides after.

    python tools/pancake/sparsity_masks.py RAW.bin [--levels 0.5,0.7,0.85] [--max 200000]
    python tools/pancake/sparsity_masks.py --selftest
"""
import argparse
import sys

import numpy as np


def load(path, cap):
    raw = np.fromfile(path, dtype=np.uint8)
    off, recs = 0, {}
    n_read = 0
    while off + 12 <= len(raw) and n_read < cap:
        layer, has_gate, n = np.frombuffer(raw[off:off + 12].tobytes(), dtype=np.int32)
        off += 12
        need = 4 * n * (2 if has_gate else 1)
        if off + need > len(raw):
            break
        v = np.frombuffer(raw[off:off + need].tobytes(), dtype=np.float32)
        off += need
        g = v[:n] if has_gate else None
        h = v[n:] if has_gate else v
        recs.setdefault(int(layer), []).append((g, h))
        n_read += 1
    out = {}
    for l, lst in recs.items():
        H = np.stack([h for _, h in lst]).astype(np.float64)
        G = np.stack([g for g, _ in lst]).astype(np.float64) if all(g is not None for g, _ in lst) else None
        out[l] = (G, H)
    return out, n_read


def silu(x):
    return x / (1.0 + np.exp(-x))


def keep_topk(score, k):
    """Boolean mask keeping the k largest scores per row."""
    idx = np.argpartition(-score, k - 1, axis=1)[:, :k]
    m = np.zeros(score.shape, bool)
    np.put_along_axis(m, idx, True, axis=1)
    return m


def energy(H, mask):
    e = (H * H).sum(1)
    k = (H * H * mask).sum(1)
    return k / np.maximum(e, 1e-30)


def analyze(data, levels):
    rows = []
    for l in sorted(data):
        G, H = data[l]
        n = H.shape[1]
        for s in levels:
            k = max(1, int(round(n * (1 - s))))
            r = {"layer": l, "s": s, "tokens": H.shape[0]}
            r["oracle"] = energy(H, keep_topk(np.abs(H), k)).mean()
            if G is not None:
                a = np.abs(silu(G))
                r["gate"] = energy(H, keep_topk(a, k)).mean()
                half = H.shape[0] // 2
                if half >= 1:
                    thr = np.quantile(a[:half].ravel(), s)
                    m = a[half:] > thr
                    r["cats"] = energy(H[half:], m).mean()
                    r["cats_s"] = 1 - m.mean()
            rows.append(r)
    return rows


def report(rows, levels):
    print(f"{'layer':>5s} {'s':>5s} {'tokens':>7s}   {'oracle cos':>10s} {'gate cos':>9s} {'cats cos':>9s} {'cats real s':>11s}")
    for r in rows:
        f = lambda key: f"{np.sqrt(r[key]):9.4f}" if key in r else f"{'-':>9s}"
        cs = f"{r['cats_s']:11.2f}" if "cats_s" in r else f"{'-':>11s}"
        print(f"{r['layer']:5d} {r['s']:5.2f} {r['tokens']:7d}   {f('oracle'):>10s} {f('gate')} {f('cats')} {cs}")
    print("overall (token-weighted):")
    verdict = None
    for s in levels:
        sel = [r for r in rows if r["s"] == s]
        w = np.array([r["tokens"] for r in sel], float)
        line = f"  s {s:.2f}:"
        for key in ("oracle", "gate", "cats"):
            vals = [r[key] for r in sel if key in r]
            if len(vals) == len(sel) and sel:
                c = np.sqrt(np.average(vals, weights=w))
                line += f"  {key} cos {c:.4f} (rel. err ~{np.sqrt(max(0.0, 1 - c * c)):.1%})"
                if key == "gate" and s >= 0.7 and c >= 0.99:
                    verdict = s if verdict is None else max(verdict, s)
        print(line)
    if any("gate" in r for r in rows):
        print("screen (gate mask cos >= 0.99 at >= 70% sparsity):",
              f"PASS up to s = {verdict:.2f} -> build the sparse kernel" if verdict else "FAIL -> no sparse kernel")
    else:
        print("no gate rows in the dump (gate/up not Q2_0): only the oracle bound; record with the Q2_0 file for the gate mask")


def selftest():
    rng = np.random.default_rng(0)
    T, n = 400, 640
    G = rng.normal(0, 1.0, (T, n))
    U = rng.normal(0, 1.0, (T, n))
    H = silu(G) * U
    data = {7: (G, H)}
    rows = analyze(data, [0.5, 0.7])
    for r in rows:
        assert r["oracle"] >= r["gate"] - 1e-12, "oracle must bound the gate mask"
        assert 0 < r["gate"] <= 1 and 0 < r["oracle"] <= 1
        assert abs(r["cats_s"] - r["s"]) < 0.05, "CATS threshold should realize about the target sparsity"
    # round trip through the file format
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False) as f:
        for t in range(5):
            np.array([7, 1, n], np.int32).tofile(f)
            G[t].astype(np.float32).tofile(f)
            H[t].astype(np.float32).tofile(f)
        path = f.name
    d, cnt = load(path, 100)
    os.unlink(path)
    assert cnt == 5 and d[7][1].shape == (5, n) and np.allclose(d[7][0], G[:5], atol=1e-6)
    print("selftest ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="?")
    ap.add_argument("--levels", default="0.5,0.7,0.85")
    ap.add_argument("--max", type=int, default=200000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if not a.raw:
        ap.error("RAW.bin required")
    levels = [float(x) for x in a.levels.split(",")]
    data, n = load(a.raw, a.max)
    print(f"{n} records, layers {sorted(data)}")
    report(analyze(data, levels), levels)


if __name__ == "__main__":
    sys.exit(main())
