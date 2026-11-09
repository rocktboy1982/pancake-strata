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
    ml = np.zeros((len(windows), rp.NL), int)
    for wi, w in enumerate(windows):
        m = 0
        for l, ids in w:
            r = res[l, ids]
            hits += int(r.sum())
            ent += r.size
            dm = len(np.unique(ids[~r]))
            ml[wi, l] = dm
            m += dm
            np.add.at(usage[l], ids.ravel(), 1.0)
        p = 0
        if (wi + 1) % pol.every == 0:
            p = rp.adapt_round(res, usage, pol)
            usage *= pol.decay
        miss.append(m)
        promo.append(p)
    return hits / max(1, ent), np.array(miss), np.array(promo), ml


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
    ml = np.zeros((n, rp.NL), int)
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
            ml[t, l] = len(missed)
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
    return hits / max(1, ent), miss, promo, ml


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
    ap.add_argument("--overlap", action="store_true", help="also price each policy with the layer-by-layer overlap model")
    ap.add_argument("--window-ms", type=float, default=24.8, help="measured window (overlap model calibration)")
    ap.add_argument("--gpu-pre-ms", type=float, default=11.6, help="GPU work before routing, per window (all layers)")
    ap.add_argument("--gpu-exp-ms", type=float, default=4.3, help="GPU work on the VRAM experts, per window")
    ap.add_argument("--draft-ms", type=float, default=2.2)
    ap.add_argument("--sync-us", type=float, default=40.0, help="CPU pool barrier/sync per layer (two phases)")
    ap.add_argument("--alpha", type=float, default=0.25, help="overlap model: CPU ms lost per ms of promotion DMA")
    ap.add_argument("--guard", type=float, default=0.5, help="share of a layer's CPU work kept on time under the guard")
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
    for name, share, miss, promo, _ml in runs:
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

    if a.overlap:
        overlap_report(a, runs)


def window_ms(ml_row, c, g_pre, g_exp, sync, draft, mode, guard, cpu_extra=0.0, dma=0.0):
    """One decode window, layer by layer (event model).  Per layer: the GPU's pre-routing work (attention / GDN, router),
    then the VRAM experts on the GPU while the CPU pool computes the layer's missed experts (c ms each).
      serial  the layer ends when both are done, plus the pool's barrier/sync cost (as the engine runs today)
      defer   STRATA_VERIFY_DEFER_CPU: the GPU does not wait for layer l's CPU rows; they join layer l+1 before its
              experts' read, so layer l's CPU work overlaps layer l+1's GPU work (the last layer is never deferred)
      guard   defer + rank guard / BELOW: the heavy share `guard` of a layer's CPU work (run first) stays in its layer"""
    t = 0.0          # GPU clock
    cpu_free = 0.0   # CPU pool clock
    prev_cpu_end = 0.0
    for l, m in enumerate(ml_row):
        start = t
        route = start + g_pre
        cpu_start = max(route, cpu_free)
        cpu_end = cpu_start + m * c + cpu_extra      # cpu_extra: this layer's share of the DMA contention
        cpu_free = cpu_end
        gpu_done = route + g_exp
        if mode != "serial" and l > 0:
            gpu_done = max(gpu_done, prev_cpu_end)          # layer l-1's late rows enter before this layer's experts
        if mode == "serial" or l == len(ml_row) - 1:
            end = max(gpu_done, cpu_end) + sync
        elif mode == "guard":
            end = max(gpu_done, cpu_start + guard * m * c) + sync
        else:
            end = gpu_done
        prev_cpu_end = cpu_end
        t = end
    return max(t + draft, dma)   # promotions that do not fit in the window gate it


def overlap_report(a, runs):
    nl = rp.NL
    g_pre, g_exp, sync = a.gpu_pre_ms / nl, a.gpu_exp_ms / nl, a.sync_us / 1e3
    base_ml = runs[0][4]
    # calibrate c (ms per distinct missed expert) so that the serial model reproduces the measured window
    lo, hi = 0.0, 1.0
    for _ in range(50):
        c = (lo + hi) / 2
        w = np.mean([window_ms(r, c, g_pre, g_exp, sync, a.draft_ms, "serial", a.guard) for r in base_ml])
        lo, hi = (c, hi) if w < a.window_ms else (lo, c)
    c = (lo + hi) / 2
    print(f"\noverlap model: GPU pre-routing {a.gpu_pre_ms} + experts {a.gpu_exp_ms} ms/window, sync {a.sync_us} us/layer, "
          f"draft {a.draft_ms} ms; c = {c * 1e3:.1f} us per missed expert reproduces the measured {a.window_ms} ms (serial, preset tier)")
    dma_per = a.mb / 1e3 / a.gbps * 1e3
    print(f"promotions: {dma_per * 1e3:.0f} us of DMA each, contention alpha {a.alpha} (CPU ms lost per DMA ms, spread over the layers)")
    print(f"{'policy':40s} {'serial':>8s} {'serial-29':>10s} {'defer':>8s} {'guard':>8s}   (window ms; % vs preset serial)")
    base = None
    for name, share, miss, promo, ml in runs:
        out = []
        for mode, sy in (("serial", sync), ("serial", sync / 2), ("defer", 0.0), ("guard", sync)):
            out.append(np.mean([window_ms(r, c, g_pre, g_exp, sy, a.draft_ms, mode, a.guard,
                                          a.alpha * p * dma_per / nl, p * dma_per) for r, p in zip(ml, promo)]))
        if base is None:
            base = out[0]
        print(f"{name:40s} " + " ".join(f"{o:6.2f}({(base / o - 1) * 100:+4.0f}%)" for o in out))


if __name__ == "__main__":
    main()
