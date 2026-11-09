"""time_ceiling.py - pancake-strata MOD_PLAN item 14 (phase 13 P1b): the expert cache judged in milliseconds, not hits.

Replays a --dump-routing trace under several placement policies and counts, per decode window, what costs time:
  CPU work   distinct experts a layer misses (each is computed by the CPU pool)
  promotions experts copied host -> VRAM (PCIe 3.0 x16 on this box, ~1.4 MB each)
The CPU time per distinct miss is calibrated so that the engine's own policy reproduces the measured CPU-expert time
(--cpu-ms, default 5.2 ms/window for short chat; 17.4 for a code edit).  Promotions are asynchronous DMA: they cost a
window only through contention (--alpha: CPU ms lost per ms of DMA; 0 = free, the sweep prints 0 / 0.25 / 0.5) and when
the DMA does not fit in the window (then it gates the window).  Policies:
  tier       the engine's adaptive tier (every --every windows, --swaps swaps; the fast-hip preset: 1 / 32)
  lru        per-layer LRU, every miss admitted (copied to VRAM for the next use)
  belady     per-layer Belady, every miss admitted, evict the farthest next use (the hit-rate optimum)
  belady-hN  Belady that admits a miss only if the expert is used again within N windows (Baleen-style: skip
             promotions that do not pay) - an oracle, the target
  kmiss k/W  online (what the engine can run): admit an expert after its k-th miss in the last W windows, evict LRU
The decision rule of item 14: build a new policy in the engine only if it saves >= 5% of the window over `tier`.

    python tools/pancake/time_ceiling.py TRACE.bin PROFILE.bin SLOTS [--cpu-ms 5.2] [--gpu-ms 18.1] [--every 1]
                                         [--swaps 32] [--mb 1.4] [--gbps 12] [--horizons 1,2,4,8]
"""
import argparse
import bisect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay_cache_policy as rp  # noqa: E402


def tier(windows, init, pol):
    """The engine's adaptive tier, per window: distinct misses and swaps (rp.simulate, counted per window)."""
    res = np.zeros((rp.NL, rp.NE), bool)
    res[init[:, 0], init[:, 1]] = True
    usage = np.zeros((rp.NL, rp.NE))
    hits = ent = 0
    miss, promo = [], []
    for wi, w in enumerate(windows):
        m = 0
        for l, ids in w:
            r = res[l, ids]
            hits += int(r.sum())
            ent += r.size
            m += len(np.unique(ids[~r]))
            np.add.at(usage[l], ids.ravel(), 1.0)
        p = 0
        if (wi + 1) % pol.every == 0:
            p = rp.adapt_round(res, usage, pol)
            usage *= pol.decay
        miss.append(m)
        promo.append(p)
    return hits / max(1, ent), np.array(miss), np.array(promo)


def per_layer(windows):
    """Per layer: the routed entries of each window and the window indices where each expert is used."""
    seqs = [[] for _ in range(rp.NL)]
    for w in windows:
        for l, ids in w:
            seqs[l].append(ids.ravel())
    uses = []
    for l in range(rp.NL):
        d = {}
        for t, ids in enumerate(seqs[l]):
            for e in np.unique(ids).tolist():
                d.setdefault(e, []).append(t)
        uses.append(d)
    return seqs, uses


def next_use(uses_l, e, t, n):
    lst = uses_l.get(e)
    if not lst:
        return n + 1
    i = bisect.bisect_right(lst, t)
    return lst[i] if i < len(lst) else n + 1


def replay(seqs, uses, budgets, inits, kind, horizon=0, kmiss=(2, 4)):
    """lru / belady / belady with an admission horizon; per window: distinct misses and admissions, all layers."""
    n = len(seqs[0])
    miss = np.zeros(n, int)
    promo = np.zeros(n, int)
    hits = ent = 0
    for l in range(rp.NL):
        b = int(budgets[l])
        if kind in ("lru", "kmiss"):
            res = list(inits[l])[:b]
        else:
            res = set(inits[l][:b])
        recent = {}                       # kmiss: expert -> windows of its recent misses
        ul = uses[l]
        for t, ids in enumerate(seqs[l]):
            u = np.unique(ids).tolist()
            inres = (lambda e: e in res)
            for e in ids.tolist():
                ent += 1
                hits += inres(e)
            missed = [e for e in u if not inres(e)]
            miss[t] += len(missed)
            if kind == "lru":
                for e in u:
                    if e in res:
                        res.remove(e)
                    res.append(e)
                    if len(res) > b:
                        res.pop(0)
                promo[t] += len(missed)
                continue
            if kind == "kmiss":
                # online: admit a miss once it missed in `k` of the last `horizon` windows; evict the least recently used
                k, W = kmiss
                for e in u:
                    if e in res:
                        res.remove(e)
                        res.append(e)
                for e in missed:
                    r = [x for x in recent.get(e, []) if x > t - W] + [t]
                    recent[e] = r
                    if len(r) >= k:
                        cur = set(u)
                        victim = next((x for x in res if x not in cur), res[0])
                        res.remove(victim)
                        res.append(e)
                        promo[t] += 1
                        recent.pop(e, None)
                continue
            cur = set(u)
            for e in missed:
                if horizon and next_use(ul, e, t, n) > t + horizon:
                    continue              # would not be used again soon enough to pay for its copy
                res.add(e)
                promo[t] += 1
                if len(res) > b:
                    cand = [x for x in res if x not in cur] or list(res)
                    res.remove(max(cand, key=lambda x: next_use(ul, x, t, n)))
    return hits / max(1, ent), miss, promo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("profile")
    ap.add_argument("slots", type=int)
    ap.add_argument("--cpu-ms", type=float, default=5.2, help="measured CPU-expert ms per window under the engine policy")
    ap.add_argument("--gpu-ms", type=float, default=18.1, help="GPU + draft ms per window (not affected by placement)")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--swaps", type=int, default=32)
    ap.add_argument("--decay", type=float, default=0.7)
    ap.add_argument("--mb", type=float, default=1.4, help="MB per expert copied")
    ap.add_argument("--gbps", type=float, default=12.0, help="effective host -> VRAM GB/s")
    ap.add_argument("--horizons", default="1,2,4,8")
    a = ap.parse_args()

    windows, _k = rp.load_trace(a.trace)
    init = rp.load_profile(a.profile, a.slots)
    budgets = np.bincount(init[:, 0], minlength=rp.NL)
    inits = [[int(e) for ll, e in init if ll == l] for l in range(rp.NL)]
    seqs, uses = per_layer(windows)

    runs = []
    # the calibration point is always the fast-hip preset's tier (every window, 32 swaps, decay 0.7), where --cpu-ms
    # was measured; --every/--swaps/--decay add the tier under test beside it
    runs.append(("tier every 1 x32 (preset, calibration)",) + tier(windows, init, rp.Policy(every=1, swaps=32, decay=0.7)))
    if (a.every, a.swaps, a.decay) != (1, 32, 0.7):
        pol = rp.Policy(every=a.every, swaps=a.swaps, decay=a.decay)
        runs.append((f"tier every {a.every} x{a.swaps} decay {a.decay}",) + tier(windows, init, pol))
    runs.append(("lru (admit every miss)",) + replay(seqs, uses, budgets, inits, "lru"))
    runs.append(("belady (admit every miss)",) + replay(seqs, uses, budgets, inits, "belady"))
    for h in [int(x) for x in a.horizons.split(",") if x]:
        runs.append((f"belady, admit if reused <= {h} windows",) + replay(seqs, uses, budgets, inits, "belady", h))

    for k, W in ((2, 2), (2, 4), (2, 8), (3, 8)):
        runs.append((f"online: admit on {k}th miss in {W} windows",) + replay(seqs, uses, budgets, inits, "kmiss", kmiss=(k, W)))

    base_miss = runs[0][2].mean()
    c_miss = a.cpu_ms / max(1e-9, base_miss)          # ms of CPU pool time per distinct missed expert
    dma_per = a.mb / 1e3 / a.gbps * 1e3               # ms of DMA per promoted expert
    print(f"{len(windows)} windows, slots {a.slots}; calibration: {base_miss:.1f} distinct misses/window under the tier "
          f"= {a.cpu_ms} ms -> {c_miss * 1e3:.0f} us per missed expert; one promotion = {dma_per * 1e3:.0f} us of DMA")
    print(f"{'policy':40s} {'hits':>6s} {'miss/w':>7s} {'promo/w':>8s} {'CPU ms':>7s} {'DMA ms':>7s}   window ms at alpha 0 / 0.25 / 0.5")
    base = None
    for name, share, miss, promo in runs:
        cpu = miss * c_miss
        dma = promo * dma_per
        outs = []
        for alpha in (0.0, 0.25, 0.5):
            win = np.maximum(a.gpu_ms + cpu + alpha * dma, dma)      # DMA longer than the window gates it
            outs.append(win.mean())
        if base is None:
            base = outs
        rel = " / ".join(f"{o:5.2f} ({(b / o - 1) * 100:+.1f}%)" for o, b in zip(outs, base))
        print(f"{name:40s} {share:6.1%} {miss.mean():7.1f} {promo.mean():8.1f} {cpu.mean():7.2f} {dma.mean():7.2f}   {rel}")


if __name__ == "__main__":
    main()
