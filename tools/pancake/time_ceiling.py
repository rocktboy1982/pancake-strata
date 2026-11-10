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

--overlap is the 2026-11-09 layer model (kept for comparison; it over-predicts deferral, see below).
--validate is the recalibrated model (A6, 2026-11-10): the engine's real layer chain (window_ms2), fitted to the measured
test-phase points, printing predicted vs measured for every measured arm, the error budget of the old model, and a ranking
of the planned candidates (A7 ProMoE, A8 Pie on the blocking tier, A46 learned policy).  Edit-window numbers: use the
routing trace of an edit-like window (bench/results/2026-11-09-time-ceiling/runs.txt used a 219-window trace at 81.2% hits).
    python tools/pancake/time_ceiling.py TRACE.bin PROFILE.bin 10909 --validate
What the old model got wrong (bench/results/2026-11-10-offline/A6_A17.md): (1) deferral adds layer l's late CPU rows to the
residual BEFORE layer l+1's FFN-side read (verify.cpp late_add), so layer l+1's router and doorbell wait for them and the
pool of l+1 starts after: only the next layer's mixer overlaps the CPU, not its router; (2) BELOW=0.1 keeps the heavy rows
on time (post(l) waits for the partial flag); (3) GPU-reach wait 15.75 ms is all GPU work before the doorbells (11.6 was
too low: the VRAM expert kernels hide behind the CPU wait); (4) promotions cost CPU time (alpha 0.5, the DDR4 share), which
cancels most of the k-th-miss gain; (5) the bench's edit window is ~2.5 ms longer than the profile's (diluting percentages).
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
    ap.add_argument("--validate", action="store_true", help="recalibrated engine-chain model (A6): predicted vs measured")
    ap.add_argument("--pre-ms", type=float, default=15.75, help="validate: GPU-reach wait, all GPU work before the doorbells (profile-fasthip.txt)")
    ap.add_argument("--mu", type=float, default=0.65, help="validate: share of --pre-ms that is the layer's mixer (runs while the CPU computes the previous layer under deferral)")
    ap.add_argument("--e-ms", type=float, default=4.3, help="validate: GPU work on the VRAM experts per window (hidden behind the CPU)")
    ap.add_argument("--ps-us", type=float, default=40.0, help="validate: pool phase barriers per layer (inside the CPU time; FUSE_PHASES halves them)")
    ap.add_argument("--c-us", type=float, default=0.0, help="validate: us CPU per distinct missed expert; 0 = calibrate from --cpu-ms and this trace")
    ap.add_argument("--heavy", type=float, default=0.4, help="validate: share of a layer's CPU time in rows kept on time (BELOW=0.1)")
    ap.add_argument("--heavy-top2", type=float, default=0.55, help="validate: the same with STRATA_VERIFY_DEFER_CPU_TOP=2")
    ap.add_argument("--sdef-us", type=float, default=-1.0, help="validate: per-layer cost of a deferred layer (late add, flag hand-off); <0 = fit to --fit-gain")
    ap.add_argument("--fit-gain", type=float, default=15.8, help="validate: measured edit gain of deferral (%%) the model is fitted to")
    ap.add_argument("--bench-tps", type=float, default=116.6, help="validate: measured base edit decode tok/s (01-deferral)")
    ap.add_argument("--extra-ms", type=float, default=None, help="validate: fixed bench-vs-profile constant (use the main trace's value for other traces)")
    ap.add_argument("--tok-per-win", type=float, default=4.70, help="validate: tokens per window of the edit run")
    a = ap.parse_args()
    if a.validate:   # defaults of the edit window the model is fitted to (profile-fasthip.txt, 109 windows)
        a.alpha = a.alpha if "--alpha" in sys.argv else 0.5
        a.cpu_ms = a.cpu_ms if "--cpu-ms" in sys.argv else 17.41
        a.draft_ms = a.draft_ms if "--draft-ms" in sys.argv else 2.5

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
    if a.validate:
        validate_report(a, runs, windows, init)


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


def calibrate_c(base_ml, g_pre, g_exp, sync, draft, guard, window):
    """ms per distinct missed expert so that the old serial model reproduces the measured window."""
    lo, hi = 0.0, 1.0
    for _ in range(50):
        c = (lo + hi) / 2
        w = np.mean([window_ms(r, c, g_pre, g_exp, sync, draft, "serial", guard) for r in base_ml])
        lo, hi = (c, hi) if w < window else (lo, c)
    return (lo + hi) / 2


def overlap_report(a, runs):
    nl = rp.NL
    g_pre, g_exp, sync = a.gpu_pre_ms / nl, a.gpu_exp_ms / nl, a.sync_us / 1e3
    base_ml = runs[0][4]
    c = calibrate_c(base_ml, g_pre, g_exp, sync, a.draft_ms, a.guard, a.window_ms)
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


def window_ms2(ml_row, promo, p, mode, heavy=0.4, sdef=0.0, ps_cut=0.0, keep=1.0):
    """One decode window, the engine's real layer chain (A6 recalibration; verify.cpp record_window / late_add).
    Per layer l:  bell_l  = the router doorbell: the GPU's mixer M and FFN-side read + router F (together p['pre'] / NL)
                  C_l     = the CPU pool's time: m distinct missed experts * c + pool phase barriers + promotion contention
                  post_l  = the layer's combine: needs the VRAM experts (E, hidden behind the CPU) and the CPU rows
      serial   bell_l = post_{l-1} + M + F; post_l waits for all of C_l (+ combine sync)
      defer    (STRATA_VERIFY_DEFER_CPU, BELOW 0.1) post_l waits only for the heavy share `heavy` of C_l (the partial flag)
               and costs `sdef`; layer l's late rows are added to the residual before layer l+1's FFN-side read, so
               bell_{l+1} = max(post_l + M, end of C_l) + F: only the next mixer overlaps the CPU; its router and the
               pool of l+1 wait for the CPU (the 2026-11-09 model let the router and the pool run during it).
      ps_cut   shortens the pool's phase barriers (STRATA_POOL_FUSE_PHASES: one phase instead of two)
      keep     share of the missed experts that stay CPU work (A7's in-flight bound)"""
    nl = len(ml_row)
    m_ms = p["pre"] * p["mu"] / nl
    f_ms = p["pre"] * (1 - p["mu"]) / nl
    e_ms = p["E"] / nl
    extra = p["alpha"] * promo * p["dma_per"] / nl      # promotion DMA steals DDR4 bandwidth from the pool
    post = cpu_prev = 0.0
    for l, m in enumerate(ml_row):
        c_l = m * keep * p["c"] + (p["ps"] - ps_cut) + extra
        if mode == "serial" or l == 0:
            bell = post + m_ms + f_ms
        else:
            bell = max(post + m_ms, cpu_prev) + f_ms
        cpu_end = bell + c_l
        if mode == "defer" and 0 < l < nl - 1:           # layer 0 (PLE) and the last layer are never deferred
            post = max(bell + e_ms, bell + heavy * c_l) + sdef
        else:
            post = max(bell + e_ms, cpu_end) + p["sync"]
        cpu_prev = cpu_end
    return post + p["draft"]


def validate_report(a, runs, windows, init):
    nl = rp.NL
    pick = lambda prefix: next(r for r in runs if r[0].startswith(prefix))
    base = runs[0]                                      # run = (name, hits, miss/window, promo/window, per-layer misses)
    km28, km38 = pick("online: admit on 2th miss in 8"), pick("online: admit on 3th miss in 8")
    ps = a.ps_us / 1e3
    c = a.c_us / 1e3 if a.c_us > 0 else (a.cpu_ms - nl * ps) / base[2].mean()
    p = dict(pre=a.pre_ms, mu=a.mu, E=a.e_ms, ps=ps, sync=a.sync_us / 1e3, draft=a.draft_ms, c=c, alpha=a.alpha,
             dma_per=a.mb / 1e3 / a.gbps * 1e3)

    def win(run, mode, heavy=a.heavy, sdef=0.0, cut=0.0, alpha=None, keep=1.0):
        q = p if alpha is None else {**p, "alpha": alpha}
        return float(np.mean([window_ms2(r, pr, q, mode, heavy, sdef, cut, keep) for r, pr in zip(run[4], run[3])]))

    s0 = win(base, "serial")
    meas_base = a.tok_per_win / a.bench_tps * 1e3
    extra = meas_base - s0 if a.extra_ms is None else a.extra_ms  # what the profile window lacks vs the bench (host stalls at window edges, ...)
    pct = lambda ref, x: ((ref + extra) / (x + extra) - 1) * 100
    if a.sdef_us >= 0:
        sdef = a.sdef_us / 1e3
    else:
        lo, hi = 0.0, 0.5
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if pct(s0, win(base, "defer", sdef=mid)) > a.fit_gain else (lo, mid)
        sdef = (lo + hi) / 2
    d0 = win(base, "defer", sdef=sdef)
    print("\n=== A6 recalibrated model (engine chain), edit window ===")
    print(f"c = {c * 1e3:.1f} us per distinct missed expert ({nl} layers x {a.ps_us:.0f} us pool barriers inside the {a.cpu_ms} ms CPU); "
          f"pre-doorbell GPU {a.pre_ms} ms (mixer share mu {a.mu}); VRAM experts {a.e_ms} ms hidden; alpha {a.alpha}; "
          f"heavy share {a.heavy}; deferred-layer cost sdef {sdef * 1e3:.0f} us (fitted to +{a.fit_gain}% deferral)")
    print(f"serial model {s0:.2f} ms (profile 37.39); bench base {a.tok_per_win} tok/window / {a.bench_tps} tok/s = {meas_base:.2f} ms "
          f"-> constant {extra:+.2f} ms added to every arm (percentages are on the bench window)")
    # (arm, model window, model % or None = compute, reference, measured %, interval, status)
    arms = [
        ("base (serial, no deferral)", s0, 0.0, "-", 0.0, 0.0, "calibration"),
        ("deferral (BELOW 0.1)", d0, pct(s0, d0), "base", 15.8, 3.9, "FITTED"),
        ("deferral, base of group 02 (136.8 tok/s)", d0, pct(s0, d0), "base", (136.8 / a.bench_tps - 1) * 100, 2.9, "check"),
        ("deferral + rank guard TOP=2", win(base, "defer", a.heavy_top2, sdef), None, "base", 20.1, 3.3, "predicted"),
        ("deferral + KMISS 2,8 (blocking)", win(km28, "defer", sdef=sdef), None, "deferral", 2.0, 1.6, "predicted"),
        ("deferral + KMISS 3,8 (blocking)", win(km38, "defer", sdef=sdef), None, "deferral", None, None, "predicted"),
        ("deferral + FUSE_PHASES", win(base, "defer", sdef=sdef, cut=ps / 2), None, "deferral", 2.5, 1.7, "predicted"),
    ]
    print(f"{'arm':44s} {'model ms':>8s} {'model %':>8s} {'vs':>9s} {'measured':>14s} {'error pts':>9s}  status")
    errs = []
    for name, w_, g, ref, mg, mi, st in arms:
        if g is None:
            g = pct(d0 if ref == "deferral" else s0, w_)
        ms = "n.s." if mg is None else f"{mg:+.1f} +-{mi:.1f}"
        er = "" if mg is None or st == "calibration" else f"{g - mg:+.1f}"
        if st == "predicted" and mg is not None:
            errs.append(abs(g - mg))
        print(f"{name:44s} {w_:8.2f} {g:+7.1f}% {ref:>9s} {ms:>14s} {er:>9s}  {st}")
    if errs:
        print(f"mean |error| of the out-of-sample arms: {np.mean(errs):.1f} points (the 2026-11-09 model: deferral +33 vs +15.8 = 17.2; "
              f"deferral + KMISS +14 vs +2.0 = 12.0; FUSE +3 vs +2.5 = 0.5)")

    # error budget of the deferral prediction, one correction at a time
    base_ml = base[4]
    sync = a.sync_us / 1e3
    gp0, ge0 = 11.5 / nl, 4.3 / nl
    gp1, ge1 = a.pre_ms / nl, a.e_ms / nl
    c0 = calibrate_c(base_ml, gp0, ge0, sync, a.draft_ms, a.guard, 37.4)
    c1 = calibrate_c(base_ml, gp1, ge1, sync, a.draft_ms, a.guard, 37.4)

    def old(gp, ge, cc, mode):
        return float(np.mean([window_ms(r, cc, gp, ge, sync if mode == "serial" else 0.0, a.draft_ms, mode, a.guard) for r in base_ml]))
    steps = [
        ("0 the 2026-11-09 model (GPU pre-doorbell 11.5 ms, router and pool free during the CPU)", (old(gp0, ge0, c0, "serial") / old(gp0, ge0, c0, "defer") - 1) * 100),
        ("1 + GPU pre-doorbell work 15.75 ms (profile), c recalibrated", (old(gp1, ge1, c1, "serial") / old(gp1, ge1, c1, "defer") - 1) * 100),
    ]
    s_a0 = win(base, "serial", alpha=0.0)
    steps.append(("2 + late rows gate the next router and pool (verify.cpp late_add)", (s_a0 / win(base, "defer", heavy=0.0, alpha=0.0) - 1) * 100))
    steps.append((f"3 + heavy rows kept on time (BELOW 0.1, share {a.heavy})", (s_a0 / win(base, "defer", alpha=0.0) - 1) * 100))
    steps.append((f"4 + deferred-layer cost {sdef * 1e3:.0f} us per layer (fitted)", (s_a0 / win(base, "defer", sdef=sdef, alpha=0.0) - 1) * 100))
    steps.append((f"5 + bench window {extra:.1f} ms longer than the profile's, promotion contention alpha {a.alpha}", pct(s0, d0)))
    print("\nwhere the +33% went (deferral on edits; measured +15.8 +-3.9):")
    for nm, g in steps:
        print(f"  {nm:88s} {g:+6.1f}%")

    # candidates
    print("\n=== candidate ranking in the recalibrated model (on top of deferral, edit window; the >= 5% rule of item 14) ===")
    print("sensitivity of the one policy arm we measured (KMISS 2,8) to the DDR contention alpha:")
    for aa in (0.0, 0.25, 0.5, 0.75):
        r28 = pct(win(base, "defer", sdef=sdef, alpha=aa), win(km28, "defer", sdef=sdef, alpha=aa))
        print(f"  alpha {aa}: KMISS 2,8 {r28:+.1f}% over deferral (measured +2.0 +-1.6)")
    d_a0 = win(base, "defer", sdef=sdef, alpha=0.0)
    keep = max(0.0, 1.0 - base[3].mean() / base[2].mean())
    d_keep = win(base, "defer", sdef=sdef, keep=keep)
    d_both = win(base, "defer", sdef=sdef, alpha=0.0, keep=keep)
    print("A7 ProMoE (chunked, cancellable promotions + reordered execution): upper bounds on the preset tier")
    print(f"   no contention (alpha 0; the tier moves {base[3].mean():.1f} promotions/window = {base[3].mean() * p['dma_per']:.1f} ms of DMA): {pct(d0, d_a0):+.1f}%")
    print(f"   in-flight reorder (every promoted expert serves its first use from VRAM, {(1 - keep) * 100:.1f}% fewer CPU misses): {pct(d0, d_keep):+.1f}%")
    print(f"   both: {pct(d0, d_both):+.1f}%   (ProMoE's own claim: 1.09x over LRU on PCIe 4.0; this box is PCIe 3.0)")
    print("A8 Pie, online swap budget on the blocking tier (tier every window; swaps per round):")
    res = []
    for sw in (8, 16, 32, 64, 128):
        r_ = base if sw == 32 else ("",) + tier(windows, init, rp.Policy(every=1, swaps=sw, decay=0.7))
        res.append((sw, r_[1], win(r_, "defer", sdef=sdef)))
    for sw, h_, w_ in res:
        print(f"   swaps {sw:3d}: hit {h_:.1%}  window {w_:.2f} ms  {pct(d0, w_):+.1f}% vs the preset's 32")
    best = min(res, key=lambda x: x[2])
    print(f"   best fixed value {best[0]}: {pct(d0, best[2]):+.1f}%; an online budget can reach at most this (a per-window oracle is not modelled)")
    print("A46 learned / evolved cache policy: the ceiling is the oracle (Belady with an admission horizon); online k-th miss for reference")
    for r_ in runs:
        if r_[0].startswith("belady") or r_[0].startswith("online"):
            print(f"   {r_[0]:44s} hit {r_[1]:.1%}  promo/w {r_[3].mean():5.1f}  {pct(d0, win(r_, 'defer', sdef=sdef)):+6.1f}% over the preset tier")


if __name__ == "__main__":
    main()
