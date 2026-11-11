"""cache_learn.py - pancake-strata MOD_PLAN Phase 16 A46: a Belady oracle and Belady-imitation policies on the expert-cache
simulator (time_ceiling.py / replay_cache_policy.py), judged in hit points against LRU and the k-th-miss admission
(STRATA_ADAPT_KMISS, A5).  Offline only: no GPU, no engine.

Papers: ADRS / "Barbarians at the Gate" (Berkeley 2025, https://arxiv.org/abs/2510.06189: evolve the policy on a
simulator); Hawkeye (Jain & Lin, ISCA'16: OPTgen rebuilds Belady's decisions on the past accesses and a predictor indexed
by the access's "PC" learns which loads are cache-friendly); Harmonia (https://arxiv.org/abs/2503.20507: admission and
migration decided apart).

The cache model is time_ceiling.py's: per layer a fixed number of VRAM slots (the profile's split of SLOTS), a decode
window's routed experts are served from the slots as they are when the window starts (hit = a routed entry, token x k,
whose expert is resident, as the engine counts "hits / all routed"), and admissions/evictions happen between windows.
Policies (all per layer, the same budgets and initial residents):
  tier         the engine's adaptive tier (rp.adapt_round, every window, 32 swaps, decay 0.7: the fast-hip preset)
  lru          admit every miss, evict LRU
  kmiss k/W    admit on the k-th miss in the last W windows, evict LRU (time_ceiling.replay, = STRATA_ADAPT_KMISS)
  belady       admit every miss, evict the farthest next use (time_ceiling.replay)
  opt          Belady with bypass: a miss is admitted only if its next use comes before the farthest resident's (that
               one is evicted).  The optimum of distinct misses per window (checked against a brute-force DP in the
               test); an oracle, the ceiling of every admission + eviction policy
  label        "perfect imitation": admit a miss iff OPTgen says Belady keeps it until its next use (future knowledge),
               evict averse-then-LRU.  The ceiling of an admission predictor on top of LRU eviction
  hawkeye-K-hH online Hawkeye: OPTgen on the past windows labels each reuse friendly/averse, an access not reused
               within H windows leaves the history as averse, a 3-bit counter per key learns it, a miss is admitted
               iff its key's counter is friendly, eviction averse-then-LRU (H = 8 and 16 for pc, 8 otherwise).  Keys:
               pc (= the expert: the "PC" of a load), pcr (expert x recent-use bucket), glob (per layer, recent-use x
               gap buckets, no expert identity: transfers across experts and requests)
  imit         offline imitation: a logistic model on the access's past-only features (uses in the last 1/4/16 windows,
               gap since the last use, tokens routed to it now, long-run frequency) fitted to the OPTgen labels of the
               OTHER traces (leave one trace out, class-balanced), admission at p >= a threshold picked on those
               training traces (--imit-thr auto) or given, eviction averse-then-LRU
CPU-side cost per window (A6's calibration, serial window): 48.7 us per distinct missed expert (the CPU pool) + alpha x
117 us per promotion (1.4 MB at 12 GB/s, DDR4 contention alpha 0.5).  The A46 decision rule (MOD_PLAN 16): describe an
engine port only if a learned (online or held-out) policy beats k-th miss by >= 5 hit points on agent traces.

    python tools/pancake/cache_learn.py PROFILE.bin SLOTS TRACE [TRACE ...] [--cuts 11,96,157] [--interleave 8]
           [--kmiss 2,4 2,8] [--alpha 0.5] [--json OUT.json]

TRACE: --dump-routing files (record = int32 layer, int32 k, k ids, k weights).  --cuts splits the FIRST trace into
requests at these window indices (the 2026-11-08 trace-bench trace: read4k / short chat / code edit / long at 11 / 96 / 157);
each part is a trace of its own (replayed from the profile's residents) and the leave-one-out unit of `imit`.
--interleave N adds a synthetic agent-like trace: the parts cut into N-window turns and played round-robin (a context switch
every N windows, as an agent switching sub-tasks) - synthetic, marked as such.
"""
import argparse
import bisect
import heapq
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay_cache_policy as rp  # noqa: E402
import time_ceiling as tc  # noqa: E402

US_MISS = 48.7      # us of CPU pool per distinct missed expert (A6, both profile windows)
US_DMA = 1.4 / 12.0 * 1e3   # us of DMA per promotion (1.4 MB at 12 GB/s)


# ------------------------------------------------------------------------------------------------ trace plumbing

class Layered:
    """One trace, per layer: seqs[l][t] = routed ids of window t (flat, token x k), uses[l][e] = sorted windows using e,
    cnt[l][t] = {e: entries in window t}."""

    def __init__(self, seqs, nl=None):
        self.nl = len(seqs) if nl is None else nl
        self.seqs = seqs
        self.n = len(seqs[0]) if seqs else 0
        self.uses, self.cnt = [], []
        for l in range(self.nl):
            d, c = {}, []
            for t, ids in enumerate(seqs[l]):
                u, k = np.unique(np.asarray(ids).ravel(), return_counts=True)
                c.append(dict(zip(u.tolist(), k.tolist())))
                for e in u.tolist():
                    d.setdefault(e, []).append(t)
            self.uses.append(d)
            self.cnt.append(c)

    @classmethod
    def from_windows(cls, windows):
        seqs, _uses = tc.per_layer(windows)
        return cls(seqs, rp.NL)

    def next_use(self, l, e, t):
        return tc.next_use(self.uses[l], e, t, self.n)

    def slice(self, a, b):
        return Layered([s[a:b] for s in self.seqs], self.nl)


def interleave(parts, turn):
    """Synthetic agent-like trace: each part cut into `turn`-window pieces, played round-robin."""
    pieces = [[p.slice(i, min(i + turn, p.n)) for i in range(0, p.n, turn)] for p in parts]
    seqs = [[] for _ in range(parts[0].nl)]
    k = 0
    while any(k < len(pc) for pc in pieces):
        for pc in pieces:
            if k < len(pc):
                for l in range(parts[0].nl):
                    seqs[l].extend(pc[k].seqs[l])
        k += 1
    return Layered(seqs, parts[0].nl)


# ------------------------------------------------------------------------------------------------ OPTgen labels

def optgen_labels(tr, budgets):
    """Per layer {(t, e): True/False}: does Belady keep e from its use at window t until its next use?  OPTgen
    (Hawkeye): walking the windows, a reuse interval (p, t] is cached iff the occupancy of every window in it is below the
    layer's capacity; then it occupies them.  The last use of an expert has no reuse: False."""
    labels = []
    for l in range(tr.nl):
        b = int(budgets[l])
        occ = np.zeros(tr.n + 1, int)
        last, lab = {}, {}
        for t in range(tr.n):
            for e in tr.cnt[l][t]:
                p = last.get(e)
                if p is not None:
                    ok = b > 0 and (t - p <= 0 or occ[p + 1:t + 1].max() < b)
                    if ok:
                        occ[p + 1:t + 1] += 1
                    lab[(p, e)] = bool(ok)
                last[e] = t
        for e, p in last.items():
            lab[(p, e)] = False
        labels.append(lab)
    return labels


# ------------------------------------------------------------------------------------------------ features (past only)

NF = 7


def features(tr, l, e, t):
    """The access of expert e at window t of layer l, from windows < t and window t's own routing only."""
    lst = tr.uses[l].get(e, [])
    i = bisect.bisect_left(lst, t)            # uses before t: lst[:i]
    def since(w):
        return i - bisect.bisect_left(lst, t - w)
    gap = (t - lst[i - 1]) if i > 0 else 64
    return np.array([1.0,
                     float(since(1)),
                     since(4) / 4.0,
                     since(16) / 16.0,
                     math.log2(1 + min(gap, 64)) / 6.0,
                     tr.cnt[l][t].get(e, 0) / 6.0,
                     i / (t + 1.0)], dtype=np.float64)


def training_set(tr, budgets, labels=None):
    labels = labels if labels is not None else optgen_labels(tr, budgets)
    X, y = [], []
    for l in range(tr.nl):
        for t in range(tr.n):
            for e in tr.cnt[l][t]:
                X.append(features(tr, l, e, t))
                y.append(1.0 if labels[l][(t, e)] else 0.0)
    return np.array(X).reshape(-1, NF), np.array(y)


def fit_logistic(X, y, iters=400, lr=0.5, l2=1e-3):
    """Plain batch gradient descent (numpy only), class-balanced."""
    w = np.zeros(X.shape[1])
    if len(y) == 0:
        return w
    pos = max(1.0, y.sum())
    neg = max(1.0, len(y) - y.sum())
    sw = np.where(y > 0.5, len(y) / (2 * pos), len(y) / (2 * neg))
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ ((p - y) * sw) / len(y) + l2 * w
        w -= lr * g
    return w


# ------------------------------------------------------------------------------------------------ policies

def stats(hits, ent, miss, promo):
    return {"hits": hits / max(1, ent), "miss_w": float(np.mean(miss)) if len(miss) else 0.0,
            "promo_w": float(np.mean(promo)) if len(promo) else 0.0}


def run_opt(tr, budgets, inits, ml=None):
    """Belady with bypass (window-granular): after window t, its misses are taken soonest-next-use first; each is admitted
    iff a free slot exists or the resident with the farthest next use is used later than it (that one is evicted).
    ml: optional [windows, layers] array filled with the distinct misses per layer (for the A6 time model)."""
    miss = np.zeros(tr.n, int)
    promo = np.zeros(tr.n, int)
    hits = ent = 0
    for l in range(tr.nl):
        b = int(budgets[l])
        res = set(inits[l][:b])
        heap = []                                     # (-next_use, e) lazy: valid iff e resident and nu current
        nu = {}
        for e in res:
            nu[e] = tr.next_use(l, e, -1)
            heapq.heappush(heap, (-nu[e], e))
        for t in range(tr.n):
            c = tr.cnt[l][t]
            for e, k in c.items():
                ent += k
                if e in res:
                    hits += k
            missed = [e for e in c if e not in res]
            miss[t] += len(missed)
            if ml is not None:
                ml[t, l] = len(missed)
            for e in c:                               # every resident used now gets its next use
                if e in res:
                    nu[e] = tr.next_use(l, e, t)
                    heapq.heappush(heap, (-nu[e], e))
            if b <= 0:
                continue
            cand = sorted(((tr.next_use(l, e, t), e) for e in missed))
            for n_e, e in cand:
                if n_e > tr.n:                        # never used again: never worth a slot
                    break
                if len(res) < b:
                    res.add(e)
                else:
                    while heap and (heap[0][1] not in res or -heap[0][0] != nu.get(heap[0][1])):
                        heapq.heappop(heap)
                    if not heap or -heap[0][0] <= n_e:
                        break                         # every resident is needed sooner: bypass (and the later ones too)
                    _, v = heapq.heappop(heap)
                    res.discard(v)
                    res.add(e)
                nu[e] = n_e
                heapq.heappush(heap, (-n_e, e))
                promo[t] += 1
    return stats(hits, ent, miss, promo)


def run_predicted(tr, budgets, inits, predict, learn=None, ml=None):
    """Admission by a predictor, eviction averse-then-LRU (Hawkeye's insertion split onto an LRU stack).
    predict(l, e, t) -> bool (friendly) at an access; learn(l, t) after window t (online learners); ml as in run_opt."""
    miss = np.zeros(tr.n, int)
    promo = np.zeros(tr.n, int)
    hits = ent = 0
    for l in range(tr.nl):
        b = int(budgets[l])
        res = OrderedDict((e, True) for e in inits[l][:b])   # LRU order, value = friendly at its last access
        for t in range(tr.n):
            c = tr.cnt[l][t]
            for e, k in c.items():
                ent += k
                if e in res:
                    hits += k
            missed = [e for e in c if e not in res]
            miss[t] += len(missed)
            if ml is not None:
                ml[t, l] = len(missed)
            for e in c:
                if e in res:
                    res[e] = predict(l, e, t)
                    res.move_to_end(e)
            if b > 0:
                for e in missed:
                    if not predict(l, e, t):
                        continue                      # averse: bypass, the CPU keeps computing it
                    if len(res) >= b:
                        v = next((x for x, f in res.items() if not f and x not in c), None)
                        if v is None:
                            v = next((x for x in res if x not in c), None)
                        if v is None:
                            continue
                        del res[v]
                    res[e] = True
                    promo[t] += 1
            if learn is not None:
                learn(l, t)
    return stats(hits, ent, miss, promo)


class Hawkeye:
    """Online OPTgen + 3-bit counters.  The reuse that ends at window t is labelled when t is reached (no future), and
    trains the key the expert had at its previous use."""

    def __init__(self, tr, budgets, key="pc", init=4, horizon=8):
        self.tr, self.budgets, self.key, self.init, self.h = tr, budgets, key, init, horizon
        self.ctr = {}
        self.occ = [np.zeros(tr.n + 1, int) for _ in range(tr.nl)]
        self.last = [dict() for _ in range(tr.nl)]       # e -> (window of its last use, key then)
        self.expire = [dict() for _ in range(tr.nl)]     # window -> [(e, its use then)]: leaves the history unreused

    def k(self, l, e, t):
        if self.key == "pc":
            return (l, e)
        lst = self.tr.uses[l].get(e, [])
        i = bisect.bisect_left(lst, t)
        r4 = i - bisect.bisect_left(lst, t - 4)
        if self.key == "pcr":
            return (l, e, min(r4, 2))
        gap = (t - lst[i - 1]) if i > 0 else 64
        # per layer (the replay walks layer by layer: a key shared across layers would see later windows of the layers
        # walked before)
        return ("g", l, min(r4, 4), min(int(math.log2(1 + gap)), 6))

    def predict(self, l, e, t):
        return self.ctr.get(self.k(l, e, t), self.init) >= 4

    def learn(self, l, t):
        b = int(self.budgets[l])
        occ, last = self.occ[l], self.last[l]
        for e in self.tr.cnt[l][t]:
            pk = last.get(e)
            if pk is not None:
                p, key = pk
                ok = b > 0 and occ[p + 1:t + 1].max() < b
                if ok:
                    occ[p + 1:t + 1] += 1
                v = self.ctr.get(key, self.init)
                self.ctr[key] = min(7, v + 1) if ok else max(0, v - 1)
            last[e] = (t, self.k(l, e, t))
            if self.h > 0:
                self.expire[l].setdefault(t + self.h, []).append((e, t))
        # Hawkeye's history is finite: an access not reused within `h` windows leaves it and trains its key averse (it
        # would not have paid for its copy); a later reuse then has no previous access in the history
        for e, p in self.expire[l].pop(t, []):
            pk = last.get(e)
            if pk is not None and pk[0] == p:
                v = self.ctr.get(pk[1], self.init)
                self.ctr[pk[1]] = max(0, v - 1)
                del last[e]


def run_all(tr, budgets, inits, init_pairs, kmiss=((2, 4), (2, 8)), imit_w=None, imit_thr=0.5, with_tier=True):
    out = {}
    windows_for_tier = None
    if with_tier:
        # the engine's tier needs windows; rebuild them from the per-layer sequences
        windows_for_tier = [[(l, np.asarray(tr.seqs[l][t]).reshape(-1, 1)) for l in range(tr.nl)] for t in range(tr.n)]
        h, m, p, _ml = tc.tier(windows_for_tier, init_pairs, rp.Policy(every=1, swaps=32, decay=0.7))
        out["tier"] = {"hits": h, "miss_w": float(np.mean(m)), "promo_w": float(np.mean(p))}
    seqs = tr.seqs
    uses = tr.uses
    for kind, label in (("lru", "lru"), ("belady", "belady")):
        h, m, p, _ml = tc.replay(seqs, uses, budgets, inits, kind)
        out[label] = {"hits": h, "miss_w": float(np.mean(m)), "promo_w": float(np.mean(p))}
    h, m, p, _ml = tc.replay(seqs, uses, budgets, inits, "belady", 4)
    out["belady-h4"] = {"hits": h, "miss_w": float(np.mean(m)), "promo_w": float(np.mean(p))}
    for k, W in kmiss:
        h, m, p, _ml = tc.replay(seqs, uses, budgets, inits, "kmiss", kmiss=(k, W))
        out[f"kmiss {k}/{W}"] = {"hits": h, "miss_w": float(np.mean(m)), "promo_w": float(np.mean(p))}
    out["opt"] = run_opt(tr, budgets, inits)
    labels = optgen_labels(tr, budgets)
    out["label"] = run_predicted(tr, budgets, inits, lambda l, e, t: labels[l][(t, e)])
    for key, hz in HAWKEYES:
        hk = Hawkeye(tr, budgets, key, horizon=hz)
        out[f"hawkeye-{key}-h{hz}"] = run_predicted(tr, budgets, inits, hk.predict, hk.learn)
    if imit_w is not None:
        out["imit"] = run_predicted(tr, budgets, inits, imit_predictor(tr, imit_w, imit_thr))
    return out


HAWKEYES = (("pc", 8), ("pc", 16), ("pcr", 8), ("glob", 8))
THRS = (0.2, 0.3, 0.4, 0.5, 0.6)


def imit_predictor(tr, w, thr):
    def pr(l, e, t):
        z = float(features(tr, l, e, t) @ w)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))) >= thr
    return pr


def pick_thr(trains, budgets, inits, w, thrs=THRS):
    """The admission threshold with the best mean hit rate on the training traces (never the held-out one)."""
    best, best_h = thrs[0], -1.0
    for thr in thrs:
        h = float(np.mean([run_predicted(tr, budgets, inits, imit_predictor(tr, w, thr))["hits"] for tr in trains]))
        if h > best_h + 1e-12:
            best, best_h = thr, h
    return best


def cpu_ms(r, alpha):
    return (r["miss_w"] * US_MISS + alpha * r["promo_w"] * US_DMA) / 1e3


def table(name, res, alpha, ref="kmiss 2/8"):
    lines = [f"== {name}", f"{'policy':14s} {'hits':>7s} {'vs ' + ref:>12s} {'miss/w':>8s} {'promo/w':>8s} "
             f"{'CPU-side ms/w':>14s} {'vs ' + ref:>10s}"]
    base = res.get(ref)
    for k, r in res.items():
        dh = (r["hits"] - base["hits"]) * 100 if base else 0.0
        dm = cpu_ms(r, alpha) - cpu_ms(base, alpha) if base else 0.0
        lines.append(f"{k:14s} {r['hits'] * 100:6.2f}% {dh:+11.2f}p {r['miss_w']:8.1f} {r['promo_w']:8.1f} "
                     f"{cpu_ms(r, alpha):14.2f} {dm:+10.2f}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__[__doc__.index("Policies"):])
    ap.add_argument("profile")
    ap.add_argument("slots", type=int)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--cuts", default="", help="split the first trace into requests at these window indices")
    ap.add_argument("--names", default="", help="names of the parts (comma separated)")
    ap.add_argument("--interleave", type=int, default=0, help="add a synthetic round-robin trace with N-window turns")
    ap.add_argument("--kmiss", nargs="*", default=["2,4", "2,8"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--imit-thr", default="auto",
                    help="imit's admission threshold on p(friendly); auto = the best mean hit rate on the training traces")
    ap.add_argument("--no-tier", action="store_true", help="skip the engine tier (slow on long traces)")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    init = rp.load_profile(a.profile, a.slots)
    budgets = np.bincount(init[:, 0], minlength=rp.NL)
    inits = [[int(e) for ll, e in init if ll == l] for l in range(rp.NL)]
    kmiss = tuple(tuple(int(x) for x in s.split(",")) for s in a.kmiss)

    whole = [Layered.from_windows(rp.load_trace(p)[0]) for p in a.traces]
    parts, names = [], []
    if a.cuts:
        cuts = [0] + [int(x) for x in a.cuts.split(",")] + [whole[0].n]
        parts = [whole[0].slice(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]
        names = a.names.split(",") if a.names else [f"part{i}" for i in range(len(parts))]
    for i, p in enumerate(a.traces[1:] if a.cuts else a.traces):
        parts.append(whole[i + 1] if a.cuts else whole[i])
        names.append(Path(p).name)

    # imitation: per part, a model fitted on the other parts' OPTgen labels
    sets = [training_set(p, budgets) for p in parts]
    models, thrs = [], []
    fixed = None if a.imit_thr == "auto" else float(a.imit_thr)
    for i in range(len(parts)):
        X = np.concatenate([s[0] for j, s in enumerate(sets) if j != i] or [np.zeros((0, NF))])
        y = np.concatenate([s[1] for j, s in enumerate(sets) if j != i] or [np.zeros(0)])
        w = fit_logistic(X, y) if len(y) else None
        models.append(w)
        trains = [p for j, p in enumerate(parts) if j != i]
        thrs.append(fixed if fixed is not None or w is None else pick_thr(trains, budgets, inits, w))
    w_all = fit_logistic(np.concatenate([s[0] for s in sets]), np.concatenate([s[1] for s in sets]))
    thr_all = fixed if fixed is not None else pick_thr(parts, budgets, inits, w_all)

    report = {}
    print(f"slots {a.slots}, {len(parts)} traces; CPU-side ms = {US_MISS} us x miss/w + {a.alpha} x {US_DMA:.0f} us x promo/w")
    for name, tr, w, thr in zip(names, parts, models, thrs):
        res = run_all(tr, budgets, inits, init, kmiss, w, thr, not a.no_tier)
        report[name] = res
        print(table(f"{name} ({tr.n} windows; imit fitted on the other traces, threshold {thr})", res, a.alpha))
    if a.cuts:
        res = run_all(whole[0], budgets, inits, init, kmiss, w_all, thr_all, not a.no_tier)
        report["whole"] = res
        print(table(f"whole first trace ({whole[0].n} windows; imit in-sample)", res, a.alpha))
    if a.interleave > 0 and len(parts) > 1:
        tr = interleave(parts, a.interleave)
        res = run_all(tr, budgets, inits, init, kmiss, w_all, thr_all, not a.no_tier)
        report[f"interleave{a.interleave}"] = res
        print(table(f"SYNTHETIC interleave, {a.interleave}-window turns round-robin ({tr.n} windows; imit in-sample)", res,
                    a.alpha))
    print("imitation model (all parts): " + " ".join(f"{x:+.2f}" for x in w_all)
          + "  [bias, used t-1, uses/4w, uses/16w, log gap, tokens now, freq]")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
