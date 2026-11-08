"""belady_ceiling.py - pancake-strata: how far the expert cache's hit rate could go at a given VRAM size.

Replays a --dump-routing trace with SLOTS experts resident (per-layer budgets taken from the learned profile, as the
engine places them) under three policies, reporting the share of routed entries served from the cache:
  engine   the adaptive tier as the engine runs it (tools/replay_cache_policy.py)
  lru      per-layer LRU with on-miss admission (llama.cpp --moe-cache-mib style), per-layer budget = engine's
  belady   per-layer Belady (evict the expert whose next use is farthest; admit on miss) - the offline optimum for
           on-miss admission at that budget
The gap engine -> belady is the headroom any predictor / eviction policy (SeqMoE, ExpertFlow, ...) can chase.

    python tools/pancake/belady_ceiling.py TRACE.bin PROFILE.bin SLOTS
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay_cache_policy as rp  # noqa: E402


def per_layer_uses(windows):
    """For each layer: the sequence of distinct-expert sets, one per window (a window reads each expert once)."""
    seqs = [[] for _ in range(rp.NL)]
    for w in windows:
        for l, ids in w:
            seqs[l].append(ids.ravel())
    return seqs


def lru(seq, budget, init):
    res = list(init)[:budget]
    hit = ent = 0
    for ids in seq:
        u = np.unique(ids)
        for e in ids:   # entries
            ent += 1
            hit += e in res
        for e in u:
            if e in res:
                res.remove(e)
                res.append(e)
            else:
                res.append(e)
                if len(res) > budget:
                    res.pop(0)
    return hit, ent


def belady(seq, budget, init):
    n = len(seq)
    # next use index of each expert after window t
    sets = [set(np.unique(s).tolist()) for s in seq]
    res = set(list(init)[:budget])
    hit = ent = 0
    for t, ids in enumerate(seq):
        for e in ids:
            ent += 1
            hit += int(e) in res
        for e in sets[t]:
            if e in res:
                continue
            res.add(e)
            if len(res) > budget:
                def nxt(x):
                    for j in range(t + 1, n):
                        if x in sets[j]:
                            return j
                    return n + 1
                # never evict what this window just used unless forced: evict the farthest next use
                victim = max((x for x in res if x not in sets[t]), key=nxt, default=None)
                if victim is None:
                    victim = max(res, key=nxt)
                res.remove(victim)
    return hit, ent


def main():
    trace, profile, slots = sys.argv[1], sys.argv[2], int(sys.argv[3])
    windows, _k = rp.load_trace(trace)
    init = rp.load_profile(profile, slots)
    eng = rp.simulate(windows, init, rp.Policy())
    seqs = per_layer_uses(windows)
    budgets = np.bincount(init[:, 0], minlength=rp.NL)
    lh = le = bh = be = 0
    for l in range(rp.NL):
        layer_init = [int(e) for ll, e in init if ll == l]
        h, e = lru(seqs[l], int(budgets[l]), layer_init)
        lh += h; le += e
        h, e = belady(seqs[l], int(budgets[l]), layer_init)
        bh += h; be += e
    print(f"slots {slots} ({slots / (rp.NL * rp.NE):.1%} of experts), {len(windows)} windows")
    print(f"  engine adaptive tier : {eng['share']:.1%} of routed entries from the cache")
    print(f"  per-layer LRU        : {lh / le:.1%}")
    print(f"  per-layer Belady     : {bh / be:.1%}   (offline optimum for on-miss admission)")


if __name__ == "__main__":
    main()
