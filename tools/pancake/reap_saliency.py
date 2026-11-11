#!/usr/bin/env python3
"""reap_saliency.py - MOD_PLAN Phase 16 order 5: REAP expert saliency on recorded routing traces (offline analysis only;
nothing in the model or the engine changes, and nothing is pruned).

REAP (Lasby et al., Cerebras, ICLR 2026, https://arxiv.org/abs/2510.13999) scores expert j of a layer by

    S_j = 1/|X_j| * sum_{x in X_j} g_j(x) * ||f_j(x)||_2,    X_j = the tokens whose top-k holds j,

g_j the router gate normalized over the top-k, f_j the expert's output, and prunes the lowest 25% or 50% of every layer
(calibration 1,024 samples x 2,048 tokens for models up to 110B; the router renormalizes over the experts left).

What a trace holds: the routed ids and, when they were recorded, the gate weights.  --dump-routing writes zero weights
from the serve path (the multi dispatch fuses the router weights; generate.cpp drive_pool_multi_impl); STRATA_WINDOW_TRACE
records them.  ||f_j(x)|| is in no trace: --act-norm takes a measured mean output norm per (layer, expert) ([48, 512]
.npy); without it every norm is 1 and the score is REAP's gate part, the mean gate value when routed.  A per-expert mean
norm equals REAP's per-token product only if gate and norm are uncorrelated.  Without weights the only score left is
routed frequency, the baseline REAP is measured against (and beats); the output says which score ran.

Per prune ratio (uniform per layer, as REAP prunes) it reports
  - the routed entries, and the gate mass when weights exist, that land on pruned experts (routing the pruned model changes);
  - the experts the trace saw fewer than --min-count times (REAP's calibration is ~1,000x this trace);
  - CPU-missed experts per window under the engine's adaptive tier (tools/replay_cache_policy.py, initial residents from
    --profile without the pruned experts), two bounds because the trace holds only the top-k: 'drop' (the pruned entries
    vanish, top-k shrinks: optimistic) and 'miss' (each distinct pruned expert of a layer's window is replaced by one
    expert outside VRAM: pessimistic).

    python tools/pancake/reap_saliency.py TRACE [--profile P.bin] [--slots 10909] [--act-norm N.npy]
                                          [--ratios 0.25,0.5] [--min-count 5] [--json OUT]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "trace"))
sys.path.insert(0, str(HERE.parents[1] / "tools"))
import replay_cache_policy as rp  # noqa: E402
import wtrace  # noqa: E402


def load(path):
    return wtrace.read(path) if str(path).endswith(".stwt") else wtrace.from_dump_routing(path)


def stats(tr):
    """Per (layer, expert): routed entries and the summed gate weight (None when any layer record has no weights)."""
    count = np.zeros((tr.n_layers, tr.n_expert))
    gate = np.zeros((tr.n_layers, tr.n_expert))
    weighted = True
    for w in tr.windows:
        for l, lay in w.layers.items():
            ids = lay.ids.ravel().astype(np.int64)
            np.add.at(count[l], ids, 1.0)
            if lay.weights is None:
                weighted = False
            else:
                np.add.at(gate[l], ids, lay.weights.ravel())
    return count, gate if weighted else None


def saliency(count, gate, act_norm=None):
    """REAP's score where the trace allows it, else routed frequency: (score [L, E], what it is)."""
    if gate is None:
        return count.copy(), "routed frequency (the trace has no gate weights, so this is REAP's baseline, not REAP)"
    norm = np.ones_like(count) if act_norm is None else act_norm
    score = np.where(count > 0, gate / np.maximum(count, 1.0) * norm, 0.0)
    return score, "REAP" + (" (mean gate x measured mean output norm)" if act_norm is not None
                            else " gate part (mean gate when routed; output norms not measured, taken as 1)")


def prune_mask(score, count, ratio):
    """True for the lowest-scoring round(ratio * E) experts of every layer (ties: fewer entries first, then lower id)."""
    L, E = score.shape
    n = int(round(ratio * E))
    mask = np.zeros((L, E), bool)
    ids = np.arange(E)
    for l in range(L):
        mask[l, np.lexsort((ids, count[l], score[l]))[:n]] = True
    return mask


def initial_residents(profile, slots, pruned, count):
    """The tier's starting residents: the profile's ranking (or the trace's own frequency) without pruned experts."""
    if profile:
        pairs = rp.load_profile(profile, 1 << 30).astype(np.int64)
    else:
        order = np.argsort(-count, axis=None, kind="stable")
        pairs = np.stack(np.unravel_index(order, count.shape), axis=1)
    if pruned is not None:
        pairs = pairs[~pruned[pairs[:, 0], pairs[:, 1]]]
    return pairs[:slots]


def replay(windows, init, pruned=None, bound="drop", pol=None):
    """Distinct CPU-missed experts per window and the tier hit rate (served entries / routed entries) through
    replay_cache_policy's adaptive tier.  With `pruned`, entries on pruned experts are removed ('drop') or replaced by
    one non-resident expert per distinct pruned expert of the layer's window ('miss')."""
    pol = pol or rp.Policy()
    res = np.zeros((rp.NL, rp.NE), bool)
    res[init[:, 0], init[:, 1]] = True
    usage = np.zeros((rp.NL, rp.NE))
    misses, hits, routed = [], 0, 0
    for wi, w in enumerate(windows):
        m = 0
        for l, ids in w:
            ids = ids.ravel()
            if pruned is not None:
                p = pruned[l, ids]
                if bound == "miss":
                    m += len(np.unique(ids[p]))
                    routed += int(p.sum())   # the replacements are routed entries that miss
                ids = ids[~p]
            routed += len(ids)
            served = res[l, ids]
            hits += int(served.sum())
            m += len(np.unique(ids[~served]))
            np.add.at(usage[l], ids, 1.0)
        misses.append(m)
        if (wi + 1) % pol.every == 0:
            rp.adapt_round(res, usage, pol)
            usage *= pol.decay
    return np.array(misses, float), hits / max(routed, 1)


def analyse(tr, ratios=(0.25, 0.5), profile=None, slots=10909, act_norm=None, min_count=5, holdout=False):
    """Scores from the calibration windows, effects on the evaluation windows: both are the whole trace (in sample), or
    with `holdout` the even windows score and the odd windows are evaluated."""
    cal = wtrace.Trace(tr.n_layers, tr.n_expert, tr.k, tr.windows[0::2] if holdout else tr.windows)
    ev = wtrace.Trace(tr.n_layers, tr.n_expert, tr.k, tr.windows[1::2] if holdout else tr.windows)
    count, gate = stats(cal)
    ev_count, ev_gate = stats(ev)
    score, what = saliency(count, gate, act_norm)
    windows = wtrace.to_replay_windows(ev)
    base_miss, base_hit = replay(windows, initial_residents(profile, slots, None, count))
    out = {"score": what, "holdout": holdout, "windows": [len(cal.windows), len(windows)],
           "tokens_per_layer": float(count.sum() / tr.n_layers),
           "under_sampled_per_layer": float((count < min_count).sum() / tr.n_layers),
           "never_routed_per_layer": float((count == 0).sum() / tr.n_layers),
           "base": {"misses_per_window": float(base_miss.mean()), "hit_rate": base_hit}, "ratios": {}}
    for r in ratios:
        mask = prune_mask(score, count, r)
        row = {"entries_on_pruned": float(ev_count[mask].sum() / ev_count.sum())}
        if ev_gate is not None:
            row["gate_mass_on_pruned"] = float(ev_gate[mask].sum() / ev_gate.sum())
        for bound in ("drop", "miss"):
            m, h = replay(windows, initial_residents(profile, slots, mask, count), mask, bound)
            row[bound] = {"misses_per_window": float(m.mean()), "hit_rate": h,
                          "misses_vs_base": float(m.sum() / base_miss.sum() - 1.0)}
        out["ratios"][str(r)] = row
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("trace", help="--dump-routing file or a STRATA_WINDOW_TRACE .stwt")
    ap.add_argument("--profile", help="expert profile (STRP) for the tier's initial residents; default: the trace's frequency")
    ap.add_argument("--slots", type=int, default=10909)
    ap.add_argument("--act-norm", help="[layers, experts] .npy of measured mean expert output norms")
    ap.add_argument("--ratios", default="0.25,0.5")
    ap.add_argument("--min-count", type=int, default=5)
    ap.add_argument("--holdout", action="store_true", help="score on the even windows, evaluate on the odd ones")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    tr = load(a.trace)
    out = analyse(tr, [float(x) for x in a.ratios.split(",")], a.profile, a.slots,
                  np.load(a.act_norm) if a.act_norm else None, a.min_count, a.holdout)
    print(f"score: {out['score']}")
    cal, ev = out["windows"]
    print(f"trace: {cal} windows scored, {ev} evaluated ({'held out' if a.holdout else 'the same windows'}); "
          f"{out['tokens_per_layer']:.0f} scored entries per layer; per layer {out['never_routed_per_layer']:.0f} experts "
          f"never routed, {out['under_sampled_per_layer']:.0f} under {a.min_count} entries")
    b = out["base"]
    print(f"no pruning: {b['misses_per_window']:.1f} CPU-missed experts per window, tier hit rate {100 * b['hit_rate']:.1f}%")
    for r, row in out["ratios"].items():
        mass = f", gate mass {100 * row['gate_mass_on_pruned']:.1f}%" if "gate_mass_on_pruned" in row else ""
        print(f"prune {100 * float(r):.0f}% per layer: entries on pruned experts {100 * row['entries_on_pruned']:.1f}%{mass}; "
              + "; ".join(f"{k} bound {row[k]['misses_per_window']:.1f} misses/window ({100 * row[k]['misses_vs_base']:+.1f}%), "
                          f"hit rate {100 * row[k]['hit_rate']:.1f}%" for k in ("drop", "miss")))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
