#!/usr/bin/env python3
"""gate_importance.py - MOD_PLAN phase 16 research order 2: is HOBBIT's gate-magnitude importance proxy usable on a
many-expert top-k model?  Offline analysis only (no engine switch, no GPU).

HOBBIT (arXiv 2411.01433, Mixtral-8x7B and Phi-MoE, both top-2) ranks a token's selected experts by their normalised gate
weight and gives each one the cumulative weight of the experts ranked above it, s_i = w_0 + ... + w_{i-1} (s_0 = 0).
s <= T1: high precision, T1 < s <= T2: low precision, s > T2: skipped (the skip inequality is our reading; the paper
only says T2 lets it "bypass" experts).  It reports T1 = 0.6, T2 = 0.9 -> 67% / 30% / 3% on Mixtral, and a Pearson
0.99 between the gate weight and the gated output ||G(x) E(x)||.  Its precision swap does not transfer here (the experts
are already IQ2_XS / Q2_0, there is no lower copy), so only the skip half is a candidate: a lossy skip of the low-importance
CPU-missed entries.  This tool measures, on a routing trace, what the 8-expert top-2 numbers become at top-k:

  correlation  (needs per-entry expert output norms ||E_e(x)||, an .npz with w and out_norm) Pearson(w, w*||E||) as HOBBIT
               computed it, which is inflated because w is on both sides, and Pearson(w, ||E||) and Spearman(w, w*||E||),
               which are not; plus the share of tokens whose order by w equals their order by contribution
  split        the high / low / skip shares at T1 / T2 (default HOBBIT's 0.6 / 0.9) and the skip share per rank
  refit        per loss budget (share of the contribution skipped; without norms the weight mass skipped), the smallest
               T2 on a 0.01 grid that stays within it, and its skip share
  CPU impact   (needs placement: a window trace, kind -1 = CPU) the CPU-missed entries the refit T2 skips and the distinct
               CPU experts per window-layer that would not run at all (every one of their entries skipped); with window
               ms in the trace, the modeled decode saving = removed experts x --us-per-miss / window time.  Rule of the
               plan: an engine switch only if that is >= 2% (--gate).

Inputs (one of):
    TRACE.stwt                 a window trace (STRATA_WINDOW_TRACE; ids, kind, weights per layer)
    --dump-routing FILE        the engine's --dump-routing file (ids + weights; no placement: every entry counted CPU)
    --npz FILE                 arrays w [N, k] (one row per token-layer) and optional out_norm [N, k], ids [N, k],
                               kind [N, k], group [N] (the window-layer of each row; default one group)

    python tools/pancake/gate_importance.py TRACE.stwt [--t1 0.6 --t2 0.9] [--budgets 0.01,0.03,0.05] [--us-per-miss 40]
    python -m pytest tools/pancake/test_gate_importance.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "trace"))


def sort_rows(w, *others):
    """Rows by descending weight (stable), weights normalised to sum 1; `others` follow the same order."""
    w = np.asarray(w, np.float64)
    order = np.argsort(-w, axis=1, kind="stable")
    ws = np.take_along_axis(w, order, axis=1)
    tot = ws.sum(axis=1, keepdims=True)
    ws = np.divide(ws, tot, out=np.zeros_like(ws), where=tot > 0)
    return (ws,) + tuple(None if o is None else np.take_along_axis(np.asarray(o), order, axis=1) for o in others)


def cum_score(ws):
    """HOBBIT's unimportance score per entry of descending, normalised rows: the weight ranked above it."""
    s = np.zeros_like(ws)
    s[:, 1:] = np.cumsum(ws, axis=1)[:, :-1]
    return s


def split(s, t1, t2):
    return s <= t1, (s > t1) & (s <= t2), s > t2


def pearson(a, b):
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    rank = lambda x: np.argsort(np.argsort(np.ravel(x), kind="stable"), kind="stable")
    return pearson(rank(a), rank(b))


def correlations(ws, norm):
    contrib = ws * norm
    same = np.all(np.argsort(-ws, axis=1, kind="stable") == np.argsort(-contrib, axis=1, kind="stable"), axis=1)
    return {"pearson_w_contrib": pearson(ws, contrib), "pearson_w_norm": pearson(ws, norm),
            "spearman_w_contrib": spearman(ws, contrib), "same_order": float(same.mean())}


def loss_share(ws, norm, skip):
    c = ws if norm is None else ws * norm
    tot = c.sum()
    return float(c[skip].sum() / tot) if tot > 0 else 0.0


def refit_t2(ws, norm, budgets, grid=None):
    """Per budget: (the smallest T2 on the grid whose skipped share is within it, its entry skip share)."""
    grid = np.round(np.arange(0.50, 1.0001, 0.01), 2) if grid is None else grid
    s = cum_score(ws)
    loss = [loss_share(ws, norm, s > t) for t in grid]
    out = {}
    for b in budgets:
        ok = [t for t, l in zip(grid, loss) if l <= b]
        t2 = float(min(ok)) if ok else None
        out[b] = (t2, float(np.mean(s > t2)) if t2 is not None else None)
    return out


def cpu_impact(groups, t2):
    """groups: (ids [n, k], kind [n, k] or None, w [n, k]).  Returns (CPU entries, skipped, CPU experts, removed)."""
    cpu_e = skip_e = cpu_x = removed = 0
    for ids, kind, w in groups:
        ws, ids_s, kind_s = sort_rows(w, ids, kind)
        cpu = np.ones(ws.shape, bool) if kind_s is None else kind_s == -1
        skip = cum_score(ws) > t2
        cpu_e += int(cpu.sum())
        skip_e += int((cpu & skip).sum())
        for e in np.unique(ids_s[cpu]):
            m = cpu & (ids_s == e)
            cpu_x += 1
            removed += int(np.all(skip[m]))
    return cpu_e, skip_e, cpu_x, removed


def analyse(w, norm=None, groups=None, window_ms=None, t1=0.6, t2=0.9, budgets=(0.01, 0.03, 0.05), us_per_miss=40.0):
    ws, norm_s = sort_rows(w, norm)
    s = cum_score(ws)
    hi, lo, sk = split(s, t1, t2)
    rep = {"rows": int(ws.shape[0]), "k": int(ws.shape[1]), "t1": t1, "t2": t2,
           "split": (float(hi.mean()), float(lo.mean()), float(sk.mean())),
           "skip_by_rank": [float(v) for v in sk.mean(axis=0)],
           "loss_at_t2": loss_share(ws, norm_s, sk),
           "correlation": correlations(ws, norm_s) if norm_s is not None else None,
           "refit": refit_t2(ws, norm_s, budgets), "cpu": {}}
    if groups:
        for b, (rt2, _) in rep["refit"].items():
            if rt2 is None:
                continue
            ce, se, cx, rm = cpu_impact(groups, rt2)
            saving = rm * us_per_miss / 1e3 / window_ms if window_ms else None
            rep["cpu"][b] = {"t2": rt2, "cpu_entries": ce, "skipped": se, "cpu_experts": cx, "removed": rm,
                             "saving": saving}
    return rep


def from_trace(tr):
    """A wtrace Trace -> (w [N, k], groups, total window ms or None); rows without weights are left out."""
    rows, groups, ms = [], [], 0.0
    for win in tr.windows:
        have = False
        for l in sorted(win.layers):
            lay = win.layers[l]
            if lay.weights is None:
                continue
            have = True
            rows.append(lay.weights)
            groups.append((lay.ids.astype(np.int64), lay.kind, lay.weights))
        if have and win.ms:
            ms += win.ms
    if not rows:
        raise ValueError("the trace has no routing weights (record with --dump-routing or a window trace with weights)")
    return np.concatenate(rows), groups, ms or None


def from_npz(path):
    z = np.load(path)
    w = z["w"]
    norm = z["out_norm"] if "out_norm" in z else None
    groups = None
    if "ids" in z:
        gid = z["group"] if "group" in z else np.zeros(len(w), np.int64)
        kind = z["kind"] if "kind" in z else None
        groups = [(z["ids"][gid == g], None if kind is None else kind[gid == g], w[gid == g]) for g in np.unique(gid)]
    return w, norm, groups


def report(rep, gate):
    k = rep["k"]
    print(f"{rep['rows']} token-layer rows, top-{k}")
    if rep["correlation"]:
        c = rep["correlation"]
        print(f"  correlation: Pearson(w, w*||E||) {c['pearson_w_contrib']:.3f} (HOBBIT's measure, inflated: w on both sides),"
              f" Pearson(w, ||E||) {c['pearson_w_norm']:.3f}, Spearman(w, w*||E||) {c['spearman_w_contrib']:.3f},"
              f" order by w = order by contribution in {c['same_order']:.1%} of rows")
    else:
        print("  correlation: no expert output norms in the input (needs an .npz with out_norm); losses below are weight mass")
    h, l, s = rep["split"]
    print(f"  split at T1 {rep['t1']} / T2 {rep['t2']}: high {h:.1%}, low {l:.1%}, skip {s:.1%}"
          f" (HOBBIT, Mixtral top-2: 67 / 30 / 3); skipped share {rep['loss_at_t2']:.2%}")
    print("  skip share by rank: " + " ".join(f"{v:.0%}" for v in rep["skip_by_rank"]))
    for b, (t2, sh) in rep["refit"].items():
        line = f"  budget {b:.0%}: " + ("no T2 on the grid" if t2 is None else f"T2 {t2:.2f}, skips {sh:.1%} of entries")
        c = rep["cpu"].get(b)
        if c:
            line += (f"; CPU entries skipped {c['skipped']}/{c['cpu_entries']}, CPU experts removed {c['removed']}/"
                     f"{c['cpu_experts']}")
            if c["saving"] is not None:
                line += f", modeled decode saving {c['saving']:.1%} ({'over' if c['saving'] >= gate else 'under'} the {gate:.0%} gate)"
        print(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--dump-routing")
    ap.add_argument("--npz")
    ap.add_argument("--t1", type=float, default=0.6)
    ap.add_argument("--t2", type=float, default=0.9)
    ap.add_argument("--budgets", default="0.01,0.03,0.05")
    ap.add_argument("--us-per-miss", type=float, default=40.0)
    ap.add_argument("--gate", type=float, default=0.02)
    a = ap.parse_args(argv)
    budgets = tuple(float(x) for x in a.budgets.split(","))
    norm = groups = ms = None
    if a.npz:
        w, norm, groups = from_npz(a.npz)
    elif a.dump_routing or a.trace:
        import wtrace
        tr = wtrace.from_dump_routing(a.dump_routing) if a.dump_routing else wtrace.read(a.trace)
        w, groups, ms = from_trace(tr)
    else:
        ap.error("give a TRACE.stwt, --dump-routing FILE or --npz FILE")
    rep = analyse(w, norm, groups, ms, a.t1, a.t2, budgets, a.us_per_miss)
    report(rep, a.gate)
    return rep


if __name__ == "__main__":
    main()
