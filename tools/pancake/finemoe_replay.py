"""finemoe_replay.py - pancake-strata MOD_PLAN Phase 16, development order item 1 (16.1): FineMoE expert-map prefetch and
PRI eviction replayed offline on recorded routing traces, against the engine's tiers, a MoE-Infinity EAM baseline and the
parked A7 / A8 / A46 candidates, all priced in the A6 time model (time_ceiling.py --validate).  No GPU, no engine.

Sources: FineMoE, EuroSys'26, https://arxiv.org/abs/2502.05370 (v2): expert maps (per iteration, the gate's probability
distribution per layer, §4.1), semantic match for layers 1..d and trajectory match for later layers (eq. 4, 5), prefetch
threshold delta = clip(1 - score, 0, 1) with experts taken in descending probability until their sum reaches delta, at
least K (eq. 6-8), store deduplication by RDY (§4.4), eviction priority PRI = 1 / (p * freq) (§4.5).  MoE-Infinity,
https://arxiv.org/abs/2401.14361 (v3, §4.5-4.7, Algorithm 1): request-level expert activation matrices matched by cosine.

The cache model is time_ceiling.py's: per layer a fixed number of VRAM slots (the profile's split of SLOTS), a routed entry
(token x k) is a hit when its expert is resident, a miss is computed by the CPU pool (it is never loaded on demand: the
engine's tiers copy experts between windows).  One FineMoE "iteration" is one verify window (its tokens pass a layer
together).  What differs from the paper, and how the replay handles it:
  - prefetches go inside the window over PCIe: the time model's layer clock (preset window / 48 layers), a prefetch for
    layer l is issued when the router of layer l-d has run (window start for the semantic layers) and serves this window
    only if its copy (one expert = the model's DMA time, serial channel) lands before layer l's experts; a late copy still
    costs a promotion and stays resident.  --ideal-dma rows let every prefetch land in time (an upper bound).
  - the trace has no embeddings, so the semantic stage is a stand-in: `prev` = cosine of the previous window's whole map
    (the current input is the previous window's output), `oracle` = the current window's own layers 0..d-1 (a perfect
    semantic match: future knowledge), `none` = no prefetch below d (the paper's Map(T)).
  - the bench trace has no router weights (the column is zero): its maps are top-k counts with equal weights.
  - a miss is admitted (copied for the next windows) by `admit`: miss = every miss (FineMoE's on-demand load), kmiss =
    the engine's k-th-miss rule (STRATA_ADAPT_KMISS), none = prefetch only.
  - eviction: pri = FineMoE's min p * (freq + 1) (p from the layer's last searched map, freq = routed entries so far, +1 so
    an expert never routed is ordered by p), lru = least recently routed (the k-th-miss tier), lfu = fewest routed entries
    (the preset tier trades by decayed usage), eam = lowest MoE-Infinity predicted activation.  Ties: the profile's rank.

    python tools/pancake/finemoe_replay.py PROFILE.bin SLOTS TRACE [TRACE ...] [--cuts 11,96,157] [--names a,b,c,d]
           [--interleave 8] [--d 3] [--cap 1000] [--json OUT.json]

TRACE: a window trace (STWT) or a --dump-routing file (tools/pancake/trace/wtrace.py reads both).  The A6 model is
calibrated on the first trace's preset tier (as time_ceiling.py --validate) and priced for every trace with those values.
--cuts splits the first trace into requests (the 2026-11-08 bench trace: read4k / short / edit / long at 11 / 96 / 157);
each part is replayed cold from the profile's residents twice for FineMoE (online store and an offline store filled with the
other parts, the paper's 70/30 setting).  --interleave N adds the synthetic agent-like trace of cache_learn.py.
Decision rule (MOD_PLAN 16, 2026-11-11): an engine switch only if the modeled decode time gains >= 2%.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "trace"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cache_learn as cl  # noqa: E402
import replay_cache_policy as rp  # noqa: E402
import time_ceiling as tc  # noqa: E402
import wtrace  # noqa: E402

NL, NE = rp.NL, rp.NE
EPS = 1e-6


# ------------------------------------------------------------------------------------------------ trace -> expert maps

def load(path):
    """Windows [(layer, ids[n_tok, k])...] and their router weights per layer (None when the trace has none)."""
    tr = wtrace.read(path) if open(path, "rb").read(4) == b"STWT" else wtrace.from_dump_routing(path, NL, NE)
    full = [w for w in tr.windows if len(w.layers) == NL]
    wts = [[w.layers[l].weights for l in sorted(w.layers)] for w in full]
    return wtrace.to_replay_windows(wtrace.Trace(tr.n_layers, tr.n_expert, tr.k, full)), wts, tr.k


def expert_maps(wins, wts=None):
    """counts[t, l, e] = routed entries of window t on expert e of layer l; maps[t, l] = FineMoE's P_l for the window (sums
    to 1): the router weights when the trace has them, else counts / entries."""
    counts = np.zeros((len(wins), NL, NE), np.float32)
    maps = np.zeros_like(counts)
    for t, w in enumerate(wins):
        for j, (l, ids) in enumerate(w):
            counts[t, l] = np.bincount(ids.ravel(), minlength=NE)[:NE]
            wl = None if wts is None else wts[t][j]
            m = counts[t, l] if wl is None else np.bincount(ids.ravel(), weights=np.abs(wl).ravel(), minlength=NE)[:NE]
            maps[t, l] = m / max(float(m.sum()), 1e-30)
    return counts, maps


def interleave_order(lengths, turn):
    """Window order of cache_learn.interleave: each part cut into `turn`-window pieces, played round-robin."""
    starts = np.cumsum([0] + list(lengths))
    out, k = [], 0
    while any(k * turn < n for n in lengths):
        for s, n in zip(starts, lengths):
            out.extend(range(s + k * turn, s + min((k + 1) * turn, n)))
        k += 1
    return out


# ------------------------------------------------------------------------------------------------ predictors

def select(P, score, K, use_delta=True):
    """FineMoE eq. 6-8: experts in descending probability until their summed probability reaches delta = clip(1 - score,
    0, 1), at least K, only experts with p > 0.  use_delta=False: the top K (the paper's Map(T+S) ablation)."""
    order = np.argsort(-P, kind="stable")
    n = K
    if use_delta:
        delta = min(1.0, max(0.0, 1.0 - score))
        n = int(np.searchsorted(np.cumsum(P[order]), delta - 1e-9)) + 1
    return order[:min(max(n, K), int((P > 0).sum()))]


class FineMoE:
    """Expert Map Store + Searcher over the trace's windows.  predict(t) -> per layer (P_l, score, prefetch set, ahead):
    layers l >= d match the trajectory of layers 0..l-d of window t (eq. 5) and are issued d layers ahead; layers l < d use
    the semantic stand-in `sem` and are issued at the window start.  prev[t] = the window before t in its sequence (-1:
    none).  The store holds window indices; at `cap` a new window replaces the most redundant one (RDY, §4.4)."""

    def __init__(self, maps, prev, d=3, sem="prev", cap=1000, K=10, use_delta=True, prefetch=True, store=(), delta=None):
        self.maps, self.prev, self.d, self.sem, self.cap, self.K = maps, prev, d, sem, cap, K
        self.use_delta, self.prefetch = use_delta, prefetch
        self.delta = delta                     # a fixed delta instead of 1 - score (1.0: the whole matched map)
        self.cn2 = np.cumsum((maps.astype(np.float64) ** 2).sum(axis=2), axis=1)   # [T, NL] cumulative squared norms
        self.ids = list(store)[-cap:] if cap > 0 else []

    def scores(self, t, ids):
        """traj[s, j] = cosine of window t and stored window s over layers 0..j; sem[s] (None: no semantic stage)."""
        S = self.maps[ids]
        cd = np.cumsum(np.einsum("sle,le->sl", S, self.maps[t], dtype=np.float64), axis=1)
        traj = cd / np.sqrt(np.maximum(self.cn2[t][None, :] * self.cn2[ids], 1e-30))
        if self.sem == "oracle":
            return S, traj, traj[:, self.d - 1]
        if self.sem != "prev" or self.prev[t] < 0:
            return S, traj, None
        pt, ps = self.prev[t], self.prev[ids]
        sem = np.full(len(ids), -1.0)
        ok = ps >= 0
        if ok.any():
            dot = np.einsum("sle,le->s", self.maps[ps[ok]], self.maps[pt], dtype=np.float64)
            sem[ok] = dot / np.sqrt(np.maximum(self.cn2[pt, -1] * self.cn2[ps[ok], -1], 1e-30))
        return S, traj, sem

    def predict(self, t):
        out = [None] * NL
        if not self.ids:
            return out
        S, traj, sem = self.scores(t, np.array(self.ids))
        for l in range(NL):
            if l >= self.d:
                sc, ahead = traj[:, l - self.d], self.d
            elif sem is not None:
                sc, ahead = sem, None
            else:
                continue
            b = int(np.argmax(sc))
            P = S[b, l]
            sel = float(sc[b]) if self.delta is None else 1.0 - self.delta
            out[l] = (P, float(sc[b]), select(P, sel, self.K, self.use_delta) if self.prefetch else (), ahead)
        return out

    def update(self, t):
        if self.cap <= 0:
            return
        if len(self.ids) < self.cap:
            self.ids.append(t)
            return
        ids = np.array(self.ids)
        _S, traj, sem = self.scores(t, ids)
        rdy = traj[:, -1] if sem is None else (self.d / NL) * sem + ((NL - self.d) / NL) * traj[:, -1]
        self.ids[int(np.argmax(rdy))] = t


class EAM:
    """MoE-Infinity baseline.  Per request an EAM (layers x experts, routed entries); the collection holds the finished
    requests (online) or every other request (offline: the paper prepares it before the run).  Layer l >= 1 matches window
    t's layers 0..l-1 by cosine against the collection (whole flattened EAMs) and takes the best match's row l, normalised;
    its top n experts are prefetched one layer ahead and the cache evicts the lowest predicted expert of the layer (the layer
    factor 1 - l/L of Algorithm 1 is constant inside a layer).  Layer 0 uses the collection's summed row 0."""

    def __init__(self, counts, req, n=10, offline=False, order=None):
        self.counts, self.req, self.n, self.offline = counts, np.asarray(req), n, offline
        self.full = {r: counts[self.req == r].sum(axis=0) for r in set(self.req.tolist())}
        self.done, self.cur = [], {}
        order = list(range(len(req))) if order is None else list(order)
        self.last = {int(self.req[t]): t for t in order}        # the window where each request ends

    def predict(self, t):
        r = int(self.req[t])
        coll = [E for q, E in self.full.items() if q != r] if self.offline else self.done
        out = [None] * NL
        if not coll:
            return out
        C = np.stack(coll).astype(np.float64)
        cd = np.cumsum(np.einsum("cle,le->cl", C, self.counts[t]), axis=1)
        ni = np.sqrt(np.cumsum((self.counts[t].astype(np.float64) ** 2).sum(axis=1)))
        nc = np.sqrt((C ** 2).sum(axis=(1, 2)))
        for l in range(NL):
            if l == 0:
                row, sc, ahead = C[:, 0].sum(axis=0), 0.0, None
            else:
                s = cd[:, l - 1] / np.maximum(nc * ni[l - 1], 1e-30)
                b = int(np.argmax(s))
                row, sc, ahead = C[b, l], float(s[b]), 1
            P = (row / max(float(row.sum()), 1e-30)).astype(np.float32)
            top = np.argsort(-P, kind="stable")[:self.n]
            out[l] = (P, sc, top[P[top] > 0], ahead)
        return out

    def update(self, t):
        r = int(self.req[t])
        self.cur[r] = self.cur.get(r, 0) + self.counts[t]
        if self.last.get(r) == t:
            self.done.append(self.cur.pop(r))


# ------------------------------------------------------------------------------------------------ the cache replay

def keep_rank(init_pairs):
    """Tie-break priority: the profile's rank (higher = kept longer), 0 for experts outside the profile."""
    keep = np.zeros((NL, NE))
    for i, (l, e) in enumerate(init_pairs):
        keep[int(l), int(e)] = len(init_pairs) - i
    return keep


class Cache:
    """Per-layer VRAM slots.  victim(l, excl): the resident expert to evict (not in excl) by `evict`; ties go to the lower
    profile rank.  put(l, e, excl): admit e, evicting when the layer is full; False if it cannot."""

    def __init__(self, budgets, inits, keep, evict):
        self.budgets, self.keep, self.evict = budgets, keep, evict
        self.res = np.zeros((NL, NE), bool)
        for l in range(NL):
            self.res[l, inits[l][:int(budgets[l])]] = True
        self.nres = self.res.sum(axis=1)
        self.freq = np.zeros((NL, NE))          # routed entries so far (LFU, FineMoE's freq)
        self.stamp = np.full((NL, NE), -1.0)    # last window routed or admitted (LRU)
        self.p = np.zeros((NL, NE))             # the layer's last searched map (PRI) or predicted activation (EAM)

    def victim(self, l, excl):
        cand = np.flatnonzero(self.res[l] & ~excl)
        if len(cand) == 0:
            return -1
        if self.evict == "lru":
            key = self.stamp[l, cand]
        elif self.evict == "lfu":
            key = self.freq[l, cand]
        elif self.evict == "pri":
            key = (self.p[l, cand] + EPS) * (self.freq[l, cand] + 1.0)
        else:   # eam
            key = self.p[l, cand]
        tie = cand[key <= key.min()]
        return int(tie[np.argmin(self.keep[l, tie])])

    def put(self, l, e, excl):
        if self.budgets[l] <= 0:
            return False
        if self.nres[l] >= self.budgets[l]:
            v = self.victim(l, excl)
            if v < 0:
                return False
            self.res[l, v] = False
            self.nres[l] -= 1
        self.res[l, e] = True
        self.nres[l] += 1
        return True


def simulate(counts, order, budgets, inits, keep, pred=None, admit="miss", evict="lru", kmiss=(2, 8), timing=None):
    """Replays the windows `order` (indices into counts).  Per window: per layer the predictor's prefetch set is copied in
    (evicting by `evict`, never an expert of the set), the routed entries are served (hit = resident and landed), then the
    misses are admitted by `admit` between windows (never evicting an expert routed in the window).  timing = dict(layer,
    pre, dma) in ms (the A6 model's layer clock), None = every prefetch lands in time.  Returns the time_ceiling run tuple
    without the name (hits, miss/window, promo/window, per-layer misses) and stats: prefetches per window, the share that
    landed late, `cover` = the share of the misses without prefetch that the prefetch sets contained, the mean score."""
    T = len(order)
    ca = Cache(budgets, inits, keep, evict)
    res, freq, stamp, p = ca.res, ca.freq, ca.stamp, ca.p
    put = ca.put
    miss, promo = np.zeros(T, int), np.zeros(T, int)
    ml = np.zeros((T, NL), int)
    hits = ent = n_pref = n_late = n_cold = n_cover = 0
    scores, recent = [], [dict() for _ in range(NL)]
    k_kmiss, w_kmiss = kmiss
    chan_next = 0.0
    for i, t in enumerate(order):
        pr = pred.predict(t) if pred is not None else None
        chan, chan_next = chan_next, 0.0       # the copies of the last window's admissions run first
        landed = {}
        done = []
        for l in range(NL):
            dl = l * timing["layer"] + timing["pre"] if timing else 0.0
            excl = np.zeros(NE, bool)
            c = counts[t, l]
            idx = np.flatnonzero(c)
            cold = idx[~res[l, idx]]           # the misses without this layer's prefetch (statistics only)
            n_cold += len(cold)
            if pr is not None and pr[l] is not None:
                P, sc, S, ahead = pr[l]
                p[l] = P
                scores.append(sc)
                excl[list(S)] = True
                n_cover += int(excl[cold].sum())
                issue = 0.0 if ahead is None or not timing else (l - ahead) * timing["layer"] + timing["pre"]
                for e in S:
                    if res[l, e] or not put(l, e, excl):
                        continue
                    stamp[l, e] = i
                    promo[i] += 1
                    n_pref += 1
                    if timing:
                        chan = max(chan, issue) + timing["dma"]
                        if chan > dl:
                            landed[(l, int(e))] = chan
                            n_late += 1
            ok = res[l, idx]
            if landed:
                ok &= np.array([landed.get((l, int(e)), 0.0) <= dl for e in idx])
            hits += float(c[idx][ok].sum())
            ent += float(c[idx].sum())
            mi = idx[~ok]
            ml[i, l] = len(mi)
            miss[i] += len(mi)
            freq[l, idx] += c[idx]
            stamp[l, idx] = i
            done.append((l, mi, idx))
        for l, mi, idx in done:                # between windows: admit this window's misses
            if admit == "none":
                continue
            cur = np.zeros(NE, bool)
            cur[idx] = True
            for e in mi[np.argsort(-counts[t, l, mi], kind="stable")]:
                if res[l, e]:
                    continue                   # a late prefetch has landed by now
                if admit == "kmiss":
                    r = [x for x in recent[l].get(e, []) if x > i - w_kmiss] + [i]
                    recent[l][e] = r
                    if len(r) < k_kmiss:
                        continue
                    recent[l].pop(e)
                if put(l, e, cur):
                    stamp[l, e] = i
                    promo[i] += 1
                    if timing:
                        chan_next += timing["dma"]
        if pred is not None:
            pred.update(t)
    st = {"prefetch_w": n_pref / max(1, T), "late": n_late / max(1, n_pref), "cover": n_cover / max(1, n_cold),
          "score": float(np.mean(scores)) if scores else None}
    return (hits / max(1.0, ent), miss, promo, ml), st


# ------------------------------------------------------------------------------------------------ pricing and the report

class Model:
    """The A6 model (time_ceiling.fit_model) calibrated on the first trace's preset tier, deferral on (fast-hip)."""

    def __init__(self, trace, profile, slots, base_run):
        self.a = tc.parse([trace, profile, str(slots), "--validate"])
        self.p, self.sdef, self.extra = tc.fit_model(self.a, base_run)
        self.d0 = tc.model_ms(self.p, base_run, "defer", self.a.heavy, self.sdef)
        self.timing = {"layer": self.d0 / NL, "pre": self.p["pre"] / NL, "dma": self.p["dma_per"]}

    def ms(self, run, alpha=None, keep=1.0):
        return tc.model_ms(self.p, run, "defer", self.a.heavy, self.sdef, alpha=alpha, keep=keep) + self.extra

    def gated(self, run):
        """Share of windows whose promotion DMA alone is longer than the preset window (the model does not gate them)."""
        return float(np.mean(run[3] * self.p["dma_per"] > self.d0))


def gain(ref_ms, x_ms):
    return (ref_ms / x_ms - 1) * 100


def fine_rows(d):
    """(row name, FineMoE options, admit, evict, ideal DMA) of the FineMoE variants."""
    return [
        (f"fine d{d} (faithful: prev-sem, miss, pri)", {}, "miss", "pri", False),
        (f"fine d{d} ideal DMA", {}, "miss", "pri", True),
        (f"fine d{d} + kmiss admission", {}, "kmiss", "pri", False),
        (f"fine d{d} prefetch only", {}, "none", "pri", False),
        (f"fine d{d} top-K (no delta)", {"use_delta": False}, "miss", "pri", False),
        (f"fine d{d} whole matched map (delta 1), kmiss", {"delta": 1.0}, "kmiss", "pri", False),
        (f"fine d{d} Map(T) (no semantic)", {"sem": "none"}, "miss", "pri", False),
        (f"fine d{d} oracle semantic, ideal DMA", {"sem": "oracle"}, "miss", "pri", True),
        ("fine d1", {"d": 1}, "miss", "pri", False),
        ("fine d6", {"d": 6}, "miss", "pri", False),
    ]


EVICT = [(f"no prefetch: {adm} + {ev}", adm, ev) for adm in ("miss", "kmiss") for ev in ("lru", "lfu", "pri")]


def a46_rows(seqs, budgets, inits, n):
    """cache_learn's A46 oracle / imitation ceiling / best online learner, with per-layer misses for the time model.
    cache_learn reports mean promotions per window only: they are spread evenly over the windows."""
    lay = cl.Layered(seqs, NL)
    labels = cl.optgen_labels(lay, budgets)
    hk = cl.Hawkeye(lay, budgets, "pc", horizon=16)
    rows = {}
    for nm, run in (("opt (A46 ceiling)", lambda m: cl.run_opt(lay, budgets, inits, ml=m)),
                    ("label (A46 imitation ceiling)",
                     lambda m: cl.run_predicted(lay, budgets, inits, lambda l, e, t: labels[l][(t, e)], ml=m)),
                    ("hawkeye-pc-h16 (A46 online)", lambda m: cl.run_predicted(lay, budgets, inits, hk.predict, hk.learn, ml=m))):
        m = np.zeros((n, NL), int)
        r = run(m)
        rows[nm] = ((r["hits"], m.sum(axis=1), np.full(n, r["promo_w"]), m), {})
    return rows


def row(model, r, st=None):
    run = ("",) + tuple(r)
    return {"hits": float(r[0]), "miss_w": float(np.mean(r[1])), "promo_w": float(np.mean(r[2])), "ms": model.ms(run),
            "gated": model.gated(run), **(st or {})}


def judge(wins, counts, maps, order, prev, req, init, budgets, inits, keep, model, d, cap, K):
    """Every policy on one replayed sequence: {row: dict(hits, miss_w, promo_w, ms, gated, ...)} and the preset tier run."""
    seq = [wins[t] for t in order]
    base = ("tier",) + tc.tier(seq, init, rp.Policy(every=1, swaps=32, decay=0.7))
    rows = {"tier (preset)": (base[1:], {})}
    seqs, uses = tc.per_layer(seq)
    rows["lru"] = (tc.replay(seqs, uses, budgets, inits, "lru"), {})
    rows["kmiss 2/8"] = (tc.replay(seqs, uses, budgets, inits, "kmiss", kmiss=(2, 8)), {})
    rows["belady-h4 (oracle)"] = (tc.replay(seqs, uses, budgets, inits, "belady", 4), {})
    rows.update(a46_rows(seqs, budgets, inits, len(order)))
    for nm, adm, ev in EVICT:
        fm = FineMoE(maps, prev, d, "prev", cap, K, prefetch=False)
        rows[nm] = simulate(counts, order, budgets, inits, keep, fm, adm, ev, timing=model.timing)
    for nm, opt, adm, ev, ideal in fine_rows(d):
        o = {"d": d, "sem": "prev", **opt}
        fm = FineMoE(maps, prev, o.pop("d"), o.pop("sem"), cap, K, **o)
        rows[nm] = simulate(counts, order, budgets, inits, keep, fm, adm, ev, timing=None if ideal else model.timing)
    for nm, off in (("eam online (MoE-Infinity)", False), ("eam offline (MoE-Infinity)", True)):
        rows[nm] = simulate(counts, order, budgets, inits, keep, EAM(counts, req, K, off, order), "miss", "eam",
                            timing=model.timing)
    return {nm: row(model, r, st) for nm, (r, st) in rows.items()}, base


def candidates(model, base, wins_seq, init):
    """A7 (ProMoE) bounds and A8 (Pie) best fixed swap budget on this sequence, as time_ceiling --validate prints them."""
    ref = model.ms(base)
    keep = max(0.0, 1.0 - base[3].mean() / max(1e-9, base[2].mean()))
    a7 = {"no contention": gain(ref, model.ms(base, alpha=0.0)), "in-flight reorder": gain(ref, model.ms(base, keep=keep)),
          "both": gain(ref, model.ms(base, alpha=0.0, keep=keep))}
    a8 = {}
    for sw in (8, 16, 64, 128):
        r = ("",) + tc.tier(wins_seq, init, rp.Policy(every=1, swaps=sw, decay=0.7))
        a8[sw] = {"hits": float(r[1]), "gain": gain(ref, model.ms(r))}
    return {"A7": a7, "A8": a8}


def table(name, res, tier_ms):
    km = res["kmiss 2/8"]["ms"]
    lines = [f"== {name}",
             f"{'policy':42s} {'hits':>7s} {'miss/w':>7s} {'promo/w':>8s} {'pref/w':>7s} {'late':>5s} {'cover':>5s} {'score':>6s} "
             f"{'ms/win':>7s} {'vs tier':>8s} {'vs kmiss':>8s}"]
    for k, r in res.items():
        pw = f"{r['prefetch_w']:7.1f}" if "prefetch_w" in r else f"{'-':>7s}"
        lt = f"{r['late'] * 100:4.0f}%" if "late" in r and r.get("prefetch_w") else f"{'-':>5s}"
        cv = f"{r['cover'] * 100:4.0f}%" if r.get("score") is not None else f"{'-':>5s}"
        sc = f"{r['score']:6.3f}" if r.get("score") is not None else f"{'-':>6s}"
        lines.append(f"{k:42s} {r['hits'] * 100:6.2f}% {r['miss_w']:7.1f} {r['promo_w']:8.1f} {pw} {lt} {cv} {sc} "
                     f"{r['ms']:7.2f} {gain(tier_ms, r['ms']):+7.1f}% {gain(km, r['ms']):+7.1f}%"
                     + ("  DMA-gated" if r["gated"] > 0 else ""))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profile")
    ap.add_argument("slots", type=int)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--cuts", default="", help="split the first trace into requests at these window indices")
    ap.add_argument("--names", default="", help="names of the parts (comma separated)")
    ap.add_argument("--interleave", type=int, default=0, help="add a synthetic round-robin trace with N-window turns")
    ap.add_argument("--d", type=int, default=3, help="prefetch distance (the paper: 3 Mixtral, 6 Qwen1.5-MoE, 4 Phi-3.5)")
    ap.add_argument("--cap", type=int, default=1000, help="Expert Map Store capacity (the paper: 1K)")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    init = rp.load_profile(a.profile, a.slots)
    budgets = np.bincount(init[:, 0], minlength=NL)
    inits = [[int(e) for ll, e in init if ll == l] for l in range(NL)]
    keep = keep_rank(init)
    loaded = [load(p) for p in a.traces]
    K = loaded[0][2]
    wins = [w for ld in loaded for w in ld[0]]
    wts = [w for ld in loaded for w in ld[1]]
    if any(x is None for ws in wts for x in ws):
        wts = None                                            # maps from counts unless every layer record has weights
    counts, maps = expert_maps(wins, wts)
    first = len(loaded[0][0])
    model = Model(a.traces[0], a.profile, a.slots, ("tier",) + tc.tier(loaded[0][0], init, rp.Policy(every=1, swaps=32, decay=0.7)))
    print(f"A6 model on {Path(a.traces[0]).name}: preset window {model.d0:.2f} ms (+{model.extra:.2f} bench constant), "
          f"layer {model.timing['layer']:.3f} ms, router at +{model.timing['pre']:.3f} ms, one copy {model.timing['dma'] * 1e3:.0f} us; "
          f"maps from {'router weights' if wts is not None else 'top-k counts (no weights in the trace)'}; d {a.d}, K {K}, store {a.cap}")

    # sequences: (name, window order, request id per window, request ids of the parts for the offline store)
    req = np.zeros(len(wins), int)
    parts = []
    if a.cuts:
        cuts = [0] + [int(x) for x in a.cuts.split(",")] + [first]
        for i in range(len(cuts) - 1):
            parts.append(list(range(cuts[i], cuts[i + 1])))
    else:
        parts.append(list(range(first)))
    s = first
    for ld in loaded[1:]:
        parts.append(list(range(s, s + len(ld[0]))))
        s += len(ld[0])
    for r, pt in enumerate(parts):
        req[pt] = r
    names = a.names.split(",") if a.names else [f"part{i}" for i in range(len(parts))]
    if not a.cuts:
        names = [Path(p).name for p in a.traces]
    seqs = [(f"{nm} ({len(pt)} windows)", pt) for nm, pt in zip(names, parts)]
    if a.cuts:
        seqs.append((f"whole first trace ({first} windows)", list(range(first))))
    if a.interleave > 0 and len(parts) > 1:
        o = interleave_order([len(pt) for pt in parts], a.interleave)
        seqs.append((f"SYNTHETIC interleave, {a.interleave}-window turns ({len(o)} windows)", o))

    report = {}
    for name, order in seqs:
        prev = np.full(len(wins), -1)
        for x, y in zip(order[:-1], order[1:]):
            prev[y] = x
        res, base = judge(wins, counts, maps, order, prev, req, init, budgets, inits, keep, model, a.d, a.cap, K)
        tier_ms = model.ms(base)
        if len(parts) > 1 and any(order is pt for pt in parts):   # the paper's offline setting: the other parts fill the store
            other = [t for pt in parts if pt is not order for t in pt]
            for x, y in zip(other[:-1], other[1:]):
                prev[y] = x if req[x] == req[y] else -1
            fm = FineMoE(maps, prev, a.d, "prev", a.cap, K, store=other)
            res[f"fine d{a.d} offline store (other parts)"] = row(model, *simulate(counts, order, budgets, inits, keep, fm,
                                                                                    "miss", "pri", timing=model.timing))
        cand = candidates(model, base, [wins[t] for t in order], init)
        report[name] = {"policies": res, "tier_ms": tier_ms, **cand}
        print(table(name, res, tier_ms))
        a7 = cand["A7"]
        print(f"   A7 ProMoE bounds over the tier: no contention {a7['no contention']:+.1f}%, in-flight reorder "
              f"{a7['in-flight reorder']:+.1f}%, both {a7['both']:+.1f}%")
        print("   A8 Pie fixed swap budgets over the preset 32: "
              + ", ".join(f"{sw}: {v['gain']:+.1f}% ({v['hits']:.1%})" for sw, v in cand["A8"].items()))
    verdict(report)
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
    return report


def verdict(report):
    """The 2% rule against the best existing online policy (preset tier or k-th miss 2/8) on every replayed sequence."""
    print("\n== verdict (>= 2% modeled decode gain over the best of tier / kmiss 2/8 needed for an engine switch)")
    best_all = None
    for name, rep in report.items():
        res = rep["policies"]
        ref = min(res["tier (preset)"]["ms"], res["kmiss 2/8"]["ms"])
        online = {k: v for k, v in res.items() if k.startswith(("fine", "no prefetch", "eam"))
                  and "ideal" not in k and "oracle" not in k}
        k_best = min(online, key=lambda k: online[k]["ms"])
        g = gain(ref, online[k_best]["ms"])
        best_all = g if best_all is None else min(best_all, g)
        print(f"   {name:58s} best online candidate: {k_best} {g:+.1f}%")
    print(f"   worst case over the sequences: {best_all:+.1f}% -> " + ("BUILD the engine switch" if best_all >= 2.0 else
                                                                         "no engine switch (below 2%)"))


if __name__ == "__main__":
    main()
