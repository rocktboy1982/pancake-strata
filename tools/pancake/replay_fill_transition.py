"""replay_fill_transition.py - offline twin of STRATA_ADAPT_FILL_FIRST and STRATA_ADAPT_TRANSITION (LEADS.md B143 / B144).

Replays a --dump-routing trace through the adaptive tier (replay_cache_policy.py's simulation of generate.cpp adapt())
and the two statistical rules of include/strata/program/tier_stat_policy.hpp, and compares them with bootstrap
confidence intervals over windows (circular block bootstrap, paired against the baseline).

Variants (a name is a '+'-joined list):
  base       the engine tier: decayed counts, a round every --every windows, profile start
  lru        the same tier with decay --lru-decay (0.05): only the last round counts
  pm         STRATA_ADAPT_PROB_MASS PROXY: the trace's weight column is all zeros, so each entry counts k x r^j / sum r^j
             (j = position in the top-k, r = --pm-r) instead of the real router probability.  Not the real switch.
  ff:F       STRATA_ADAPT_FILL_FIRST=F as the engine runs it: the slots beyond the first F share are placeholders (still hit
             until overwritten), admitted first-miss, no claimed expert evicted while a layer has an empty slot
  thr:N      with ff / ffs: the count a miss needs for the empty slot (default 1.0 = first miss; 2.0 = the tier's own floor)
  ffs:F      like ff but the candidates beyond the empty slots are not blocked
  ffe:F      the owner's literal form: the slots beyond the first F share are truly EMPTY (no hit), filled on the first miss
  tr:W       STRATA_ADAPT_TRANSITION=W
  ff:F+tr:W, ff:F+pm, tr:W+pm, ff:F+tr:W+pm ...

Metric: the share of routed entries the cache serves (the engine's "hits / all routed"), per window; swaps per window is the
expert movement.  Differences against base are in points of share with a 95% interval.

    python tools/pancake/replay_fill_transition.py bench/prefetch/routing.bin data/expert-profile-pancake.bin 10909
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay_cache_policy as R  # noqa: E402


class Stat:
    """tier_stat_policy.hpp in numpy (observe / apply as in the C++ header; same constants)."""

    def __init__(self, nl, ne, adapt_decay, core_n=0, trans_w=0.0, trans_decay=0.95, theta=0.5):
        self.nl, self.ne, self.core_n, self.w, self.td, self.theta = nl, ne, core_n, trans_w, trans_decay, theta
        self.scale = 1.0 / (1.0 - adapt_decay)
        self.last_raw = None
        self.claimed = np.zeros((nl, ne), bool)
        self.was_res = np.zeros((nl, ne), bool)
        self.used = np.zeros((nl, ne), bool)
        self.prev_used = np.zeros((nl, ne), bool)
        self.rounds = 0
        self.s = 0.1
        self.ov_mean = 0.0
        self.g = 1.0
        if trans_w > 0:
            self.T = np.zeros((nl, ne, ne), np.float32)
            self.N = np.zeros((nl, ne), np.float32)
            self.M = np.zeros((nl, ne), np.float32)
            self.Rn = np.zeros(nl, np.float32)

    def observe(self, raw, slot, decay, boundary=False):
        """raw: counts (nl, ne); slot: (nl, ne) int slot index or -1."""
        first = self.last_raw is None
        if first:
            self.last_raw = np.zeros_like(raw)
        self.rounds += 1
        used = raw > self.last_raw * decay + 1e-6
        self.last_raw = raw.copy()
        r = slot >= 0
        if first:
            self.claimed = r & (slot < self.core_n)
        else:
            self.claimed = np.where(~r, False, np.where(~self.was_res | used, True, self.claimed))
        self.was_res = r
        self.used = used
        if self.w > 0:
            self._learn(boundary)
        self.prev_used = used

    def _learn(self, boundary):
        inter = (self.prev_used & self.used).sum(1)
        uni = (self.prev_used | self.used).sum(1)
        ok = uni > 0
        ov = float((inter[ok] / uni[ok]).mean()) if ok.any() else 0.0
        switched = boundary
        if self.rounds > 1:
            if self.ov_mean <= 0.0:
                self.ov_mean = ov
            else:
                if self.theta > 0 and ov < self.theta * self.ov_mean:
                    switched = True
                if not switched:
                    self.ov_mean = 0.8 * self.ov_mean + 0.2 * ov
        self.s = 0.7 * self.s + 0.3 * (1.0 if switched else 0.0)
        if switched and boundary:
            self.s = 1.0
        self.g /= self.td
        g = np.float32(self.g)
        for l in range(self.nl):
            cur = np.flatnonzero(self.used[l])
            if len(cur):
                self.M[l, cur] += g
                self.Rn[l] += g
            if self.rounds > 1 and not switched and len(cur):
                prev = np.flatnonzero(self.prev_used[l])
                if len(prev):
                    self.T[l][np.ix_(prev, cur)] += g
                    self.N[l, prev] += g
        if self.g > 1e6:
            k = np.float32(1.0 / self.g)
            self.T *= k
            self.N *= k
            self.M *= k
            self.Rn *= k
            self.g = 1.0

    def apply_fill(self, u, res_bool, soft=False, thr=1.0):
        """Placeholder semantics (engine): res_bool = resident.  soft: the candidates beyond the empty slots are not blocked
        (the normal rule may still evict a claimed resident for them)."""
        for l in range(self.nl):
            free = res_bool[l] & ~self.claimed[l]
            fn = int(free.sum())
            if fn == 0:
                continue
            cand = np.flatnonzero(~res_bool[l] & (u[l] > 0))
            cand = cand[np.argsort(-u[l, cand], kind="stable")]
            take = 0
            for e in cand:
                if take < fn and u[l, e] >= thr:
                    u[l, e] = max(u[l, e] + 1.0, 2.0)
                    take += 1
                elif not soft:
                    u[l, e] = 0.0
            u[l, free] = -1.0

    def apply_trans(self, u, res_bool):
        s = min(max(self.s, 0.0), 1.0)
        for l in range(self.nl):
            S = np.flatnonzero(self.prev_used[l])
            nn = self.N[l, S]
            has = S[nn > 1e-9]
            rn = float(self.Rn[l])
            if len(has) == 0 and rn <= 1e-9:
                continue
            if len(has):
                acc = (self.T[l, has] / self.N[l, has][:, None]).sum(0)
                ks, km = (1.0 - s) / len(has), (s / rn if rn > 0 else 0.0)
            else:
                acc, ks, km = 0.0, 0.0, 1.0 / rn
            p = np.minimum(1.0, acc * ks + self.M[l] * km)
            u[l] += np.where(res_bool[l], self.w * self.scale * p, 0.0)


def parse(name):
    """'ff:0.5+tr:2+pm' -> dict."""
    cfg = {"pm": False, "ff": 0.0, "ffs": 0.0, "thr": 1.0, "ffe": 0.0, "tr": 0.0, "lru": False}
    for part in name.split("+"):
        if part == "base":
            continue
        if part in ("pm", "lru"):
            cfg[part] = True
        else:
            k, v = part.split(":")
            cfg[k] = float(v)
    return cfg


def simulate(windows, init, slots, pol, cfg, nl=R.NL, ne=R.NE, pm_r=0.7, lru_decay=0.05, k=10):
    decay = lru_decay if cfg["lru"] else pol.decay
    pol = R.Policy(every=pol.every, decay=decay, swaps=pol.swaps, margin=pol.margin, mincand=pol.mincand)
    res = np.zeros((nl, ne), bool)
    res[init[:, 0], init[:, 1]] = True
    cap = res.sum(1)                      # slots per layer
    slotidx = np.full((nl, ne), -1, np.int32)
    slotidx[init[:, 0], init[:, 1]] = np.arange(len(init), dtype=np.int32)
    ffe = cfg["ffe"] > 0
    soft = cfg["ffs"] > 0
    f = cfg["ffe"] if ffe else (cfg["ffs"] if soft else cfg["ff"])
    core_n = int(np.ceil(f * slots - 1e-9)) if f > 0 else 0
    if ffe:                               # literal: only the core is resident at the start
        res[:] = False
        core = init[:core_n]
        res[core[:, 0], core[:, 1]] = True
    stat = None
    if (f > 0 and not ffe) or cfg["tr"] > 0:  # ff / ffs: placeholders
        stat = Stat(nl, ne, decay, core_n=core_n if (f > 0 and not ffe) else 0, trans_w=cfg["tr"])
    pw = None
    if cfg["pm"]:
        pw = pm_r ** np.arange(k)
        pw = pw / pw.sum() * k
    usage = np.zeros((nl, ne))
    hits = np.zeros(len(windows))
    ents = np.zeros(len(windows))
    dh = np.zeros(len(windows))
    dn = np.zeros(len(windows))
    swaps = 0
    for wi, w in enumerate(windows):
        for l, ids in w:
            r = res[l, ids]
            hits[wi] += r.sum()
            ents[wi] += r.size
            un = np.unique(ids)
            dh[wi] += res[l, un].sum()
            dn[wi] += len(un)
            if pw is not None:
                np.add.at(usage[l], ids.ravel(), np.broadcast_to(pw, ids.shape).ravel())
            else:
                np.add.at(usage[l], ids.ravel(), 1.0)
        if (wi + 1) % pol.every == 0:
            raw = usage.copy()
            slot = np.where(res, np.where(slotidx >= 0, slotidx, 1 << 30), -1)   # (only the first round reads the slot order)
            u = raw
            if stat is not None:
                stat.observe(raw, slot, decay)
                u = raw.copy()
                if f > 0 and not ffe:
                    stat.apply_fill(u, res, soft, cfg["thr"])
                if cfg["tr"] > 0:
                    stat.apply_trans(u, res)
            budget = pol.swaps
            if ffe:                       # fill the empty slots first, no eviction on a layer that still has room
                free = cap - res.sum(1)
                fills = []
                for l in range(nl):
                    if free[l] > 0:
                        cand = np.flatnonzero(~res[l] & (u[l] >= 1.0))
                        cand = cand[np.argsort(-u[l, cand], kind="stable")][: int(free[l])]
                        fills += [(u[l, e], l, int(e)) for e in cand]
                fills.sort(key=lambda t: -t[0])
                for _, l, e in fills[:budget]:
                    res[l, e] = True
                    swaps += 1
                budget -= min(budget, len(fills))
                full = (cap - res.sum(1)) <= 0
                uu = u.copy()
                uu[~full] = 0.0           # layers with room: no swap, only fills
                u = uu
            p2 = R.Policy(every=pol.every, decay=pol.decay, swaps=max(budget, 0), margin=pol.margin, mincand=pol.mincand)
            swaps += R.adapt_round(res, u, p2) if budget > 0 else 0
            usage *= pol.decay
    return dict(hits=hits, ents=ents, dh=dh, dn=dn, swaps=swaps / len(windows),
                tstat_rounds=stat.rounds if stat else 0, switch_s=stat.s if stat else 0.0)


def boot_idx(n, block, nboot, rng):
    nb = -(-n // block)
    starts = rng.integers(0, n, size=(nboot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(nboot, -1)[:, :n]


def share(sel, hits, ents):
    return hits[sel].sum(1) / np.maximum(ents[sel].sum(1), 1)


def summarize(results, base="base", nboot=2000, block=8, seed=0, metric="e"):
    rng = np.random.default_rng(seed)
    n = len(results[base]["hits"])
    sel = boot_idx(n, block, nboot, rng)
    hk, nk = ("hits", "ents") if metric == "e" else ("dh", "dn")
    b = share(sel, results[base][hk], results[base][nk])
    out = {}
    for name, r in results.items():
        x = share(sel, r[hk], r[nk])
        d = (x - b) * 100
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[name] = dict(share=r[hk].sum() / r[nk].sum(), diff=(r[hk].sum() / r[nk].sum() - results[base][hk].sum() / results[base][nk].sum()) * 100,
                         lo=lo, hi=hi, swaps=r["swaps"])
    return out


def default_variants(f=0.7, w=2.0):
    return ["base", "lru", "pm", f"ff:{f}", f"ffe:{f}", f"tr:{w}", f"ff:{f}+tr:{w}", f"ff:{f}+pm", f"tr:{w}+pm", f"ff:{f}+tr:{w}+pm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("profile")
    ap.add_argument("slots", type=int)
    ap.add_argument("--variants", nargs="+", default=None)
    ap.add_argument("--sweep", action="store_true", help="also sweep F in 0.3/0.5/0.7/0.85 and W in 0.5/1/2/4/8")
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--nboot", type=int, default=2000)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--pm-r", type=float, default=0.7)
    a = ap.parse_args()
    windows, k = R.load_trace(a.trace)
    init = R.load_profile(a.profile, a.slots)
    pol = R.Policy(every=a.every)
    names = a.variants or default_variants()
    if a.sweep:
        names += [f"ff:{f}" for f in (0.3, 0.5, 0.85)] + [f"ffe:{f}" for f in (0.3, 0.5, 0.85)] + [f"tr:{w}" for w in (0.5, 1, 4, 8)]
    names = list(dict.fromkeys(names))
    print(f"{len(windows)} windows, k={k}, slots {a.slots}, every {a.every}, circular block bootstrap block={a.block} x {a.nboot}; "
          f"trace weight column is all zeros: 'pm' is a position-weight PROXY (r={a.pm_r}), not the real STRATA_ADAPT_PROB_MASS")
    results = {}
    for nm in names:
        results[nm] = simulate(windows, init, a.slots, pol, parse(nm), pm_r=a.pm_r, k=k)
    for metric, label in (("e", "share of routed entries served"), ("d", "share of distinct (window, layer) experts served")):
        s = summarize(results, nboot=a.nboot, block=a.block, metric=metric)
        print(f"\n{label}; difference vs base in points, 95% CI over windows")
        print(f"{'variant':22} {'share':>7} {'diff':>7} {'CI lo':>7} {'CI hi':>7} {'swaps/win':>10}")
        for nm in names:
            r = s[nm]
            print(f"{nm:22} {r['share']:7.4f} {r['diff']:+7.2f} {r['lo']:+7.2f} {r['hi']:+7.2f} {r['swaps']:10.2f}")


if __name__ == "__main__":
    main()
