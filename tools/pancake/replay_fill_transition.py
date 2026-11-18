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
  pmw        the REAL STRATA_ADAPT_PROB_MASS: each entry counts router weight x k (expert_source.cpp); a window without
             weights counts +1 as the engine does.  Needs a trace with nonzero weights: --stwt FILE (a STRATA_WINDOW_TRACE with
             STRATA_WINDOW_TRACE_WEIGHTS=1, matched to the dump by ids) or a dump whose weight column is nonzero
  tr:W       STRATA_ADAPT_TRANSITION=W (its own switch detector, as the engine runs it)
  trb:W      like tr:W, plus the sidecar's true conversation switch fed to the learner as TierStat::boundary() (the engine
             never calls boundary() today: this is the upper bound of a switch signal)
  ff:F+tr:W, ff:F+pm, tr:W+pm, ff:F+tr:W+pmw ...

With --requests REQUESTS.json (conv_trace.py's sidecar: end_record, conv per turn) every window gets its turn and conversation;
a conversation switch is a turn whose conversation differs from the previous turn's.  A third table then gives the share in the
--after windows that follow a switch ("post-switch") and in all other windows, each with its own paired bootstrap.

Metric: the share of routed entries the cache serves (the engine's "hits / all routed"), per window; swaps per window is the
expert movement.  Differences against base are in points of share with a 95% interval.

    python tools/pancake/replay_fill_transition.py bench/prefetch/routing.bin data/expert-profile-pancake.bin 10909
    python tools/pancake/replay_fill_transition.py real4.trace PROFILE 10909 --requests real4.requests.json --stwt real4.stwt
"""
import argparse
import json
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
    cfg = {"pm": False, "pmw": False, "ff": 0.0, "ffs": 0.0, "thr": 1.0, "ffe": 0.0, "tr": 0.0, "trb": False, "lru": False}
    for part in name.split("+"):
        if part == "base":
            continue
        if part in ("pm", "pmw", "lru"):
            cfg[part] = True
        else:
            k, v = part.split(":")
            if k == "trb":
                cfg["tr"], cfg["trb"] = float(v), True
            else:
                cfg[k] = float(v)
    return cfg


def simulate(windows, init, slots, pol, cfg, nl=R.NL, ne=R.NE, pm_r=0.7, lru_decay=0.05, k=10, weights=None, boundaries=()):
    """weights: list parallel to windows, each None or {layer: float array shaped like that layer's ids} (for pmw);
    boundaries: window indexes where a new conversation starts (fed to the learner by trb)."""
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
    bset = set(boundaries)
    bpend = False
    for wi, w in enumerate(windows):
        if wi in bset:
            bpend = True
        ww = weights[wi] if (weights is not None and cfg["pmw"]) else None
        for l, ids in w:
            r = res[l, ids]
            hits[wi] += r.sum()
            ents[wi] += r.size
            un = np.unique(ids)
            dh[wi] += res[l, un].sum()
            dn[wi] += len(un)
            if pw is not None:
                np.add.at(usage[l], ids.ravel(), np.broadcast_to(pw, ids.shape).ravel())
            elif ww is not None and l in ww:     # STRATA_ADAPT_PROB_MASS: w x k where w is a positive number, else 0
                x = ww[l].astype(np.float64)
                np.add.at(usage[l], ids.ravel(), np.where(np.isfinite(x) & (x > 0), x * k, 0.0).ravel())
            else:
                np.add.at(usage[l], ids.ravel(), 1.0)
        if (wi + 1) % pol.every == 0:
            raw = usage.copy()
            slot = np.where(res, np.where(slotidx >= 0, slotidx, 1 << 30), -1)   # (only the first round reads the slot order)
            u = raw
            if stat is not None:
                stat.observe(raw, slot, decay, boundary=bpend and bool(cfg["trb"]))
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
            bpend = False
    return dict(hits=hits, ents=ents, dh=dh, dn=dn, swaps=swaps / len(windows),
                tstat_rounds=stat.rounds if stat else 0, switch_s=stat.s if stat else 0.0)


def load_dump(path, nl=None):
    """Like R.load_trace, but also keeps each complete window's first record index and the weight column.
    Returns (windows [[(layer, ids)]], k, starts [first record of each window], weights [{layer: float32 (n, k)}], total records)."""
    nl = R.NL if nl is None else nl
    raw = np.fromfile(path, dtype=np.int32)
    k = int(raw[1])
    rec = 2 + 2 * k
    raw = raw[: (len(raw) // rec) * rec].reshape(-1, rec)
    layer = raw[:, 0]
    ids = raw[:, 2:2 + k]
    wts = np.ascontiguousarray(raw[:, 2 + k:]).view(np.float32)
    chg = np.flatnonzero(np.diff(layer) != 0) + 1
    starts = np.concatenate([[0], chg])
    ends = np.concatenate([chg, [len(layer)]])
    groups, cur, last = [], None, -1
    for s0, e0 in zip(starts, ends):
        l = int(layer[s0])
        if cur is None or l <= last:
            cur = []
            groups.append(cur)
        cur.append((l, int(s0), int(e0)))
        last = l
    windows, first, weights = [], [], []
    for g in groups:
        if len(g) != nl:
            continue
        windows.append([(l, ids[a:b]) for l, a, b in g])
        weights.append({l: wts[a:b] for l, a, b in g})
        first.append(g[0][1])
    return windows, k, np.array(first, np.int64), weights, len(raw)


def has_real_weights(weights):
    """True when the weight column carries router probabilities: some positive value and not all equal (all 1.0 = unit weights)."""
    vals = np.concatenate([x.ravel() for w in weights[:200] for x in w.values()]) if weights else np.zeros(0, np.float32)
    pos = vals[np.isfinite(vals) & (vals > 0)]
    return bool(len(pos) > 0 and float(pos.max() - pos.min()) > 1e-6)


def _wkey(layers, lids):
    """A match key of one window: a hash of the ids of three layers (first, middle, last)."""
    ls = sorted(layers)
    pick = [ls[0], ls[len(ls) // 2], ls[-1]]
    return tuple(hash(np.ascontiguousarray(lids(l)).astype(np.int16).tobytes()) for l in pick)


def attach_stwt(windows, stwt_path, base_weights=None):
    """Weights for each dump window from a STRATA_WINDOW_TRACE file (STRATA_WINDOW_TRACE_WEIGHTS=1), matched by ids (a window
    of the trace and of the dump hold the same routing).  Returns (weights list, matched, nonzero matched)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "trace"))
    import wtrace  # noqa: E402
    tr = wtrace.read(stwt_path)
    table = {}
    for w in tr.windows:
        if not w.layers or any(lay.weights is None for lay in w.layers.values()):
            continue
        table.setdefault(_wkey(w.layers, lambda l, w=w: w.layers[l].ids), []).append(w)
    out, matched, nonzero = [], 0, 0
    for i, win in enumerate(windows):
        d = dict(win)
        got = None
        cand = table.get(_wkey(d, lambda l, d=d: d[l]))
        if cand:
            w = cand.pop(0)
            if set(w.layers) == set(d) and all(w.layers[l].ids.shape == d[l].shape and (w.layers[l].ids == d[l]).all() for l in d):
                got = {l: w.layers[l].weights for l in d}
        if got is not None:
            matched += 1
            nonzero += int(any(float(np.max(x)) > 0 for x in got.values()))
        out.append(got)
    return out, matched, nonzero


def segment(requests, starts, total):
    """Turn of every window from the sidecar: request i owns the records (end_{i-1}, end_i].  Returns (turn index per window,
    list of turn dicts {turn, conv, domain, first, n, switch}); a switch is a turn whose conversation differs from the previous
    turn's.  Windows after the last end_record (a partial tail) belong to the last turn."""
    ends = np.array([int(r["end_record"]) for r in requests], np.int64)
    turn = np.minimum(np.searchsorted(ends, starts, side="right"), len(requests) - 1)
    turns, prev = [], None
    for i, r in enumerate(requests):
        idx = np.flatnonzero(turn == i)
        turns.append(dict(turn=i, conv=r.get("conv"), domain=r.get("domain"), first=int(idx[0]) if len(idx) else None,
                          n=int(len(idx)), switch=bool(prev is not None and r.get("conv") != prev)))
        prev = r.get("conv")
    return turn, turns


def switch_windows(turns):
    """The first window of every turn that starts a different conversation than the turn before."""
    return [t["first"] for t in turns if t["switch"] and t["first"] is not None]


def subset_masks(n, switches, after):
    """(post-switch mask, other mask): windows within `after` windows from each conversation switch, and the rest."""
    post = np.zeros(n, bool)
    for f in switches:
        post[f:f + after] = True
    return post, ~post


def boot_idx(n, block, nboot, rng):
    nb = -(-n // block)
    starts = rng.integers(0, n, size=(nboot, nb))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(nboot, -1)[:, :n]


def share(sel, hits, ents):
    return hits[sel].sum(1) / np.maximum(ents[sel].sum(1), 1)


def summarize(results, base="base", nboot=2000, block=8, seed=0, metric="e", subset=None):
    """subset: a boolean mask over windows; the bootstrap and the shares then use those windows only (in trace order)."""
    rng = np.random.default_rng(seed)
    pick = np.arange(len(results[base]["hits"])) if subset is None else np.flatnonzero(subset)
    n = len(pick)
    sel = pick[boot_idx(n, block, nboot, rng)]
    hk, nk = ("hits", "ents") if metric == "e" else ("dh", "dn")
    b = share(sel, results[base][hk], results[base][nk])
    out = {}
    for name, r in results.items():
        x = share(sel, r[hk], r[nk])
        d = (x - b) * 100
        lo, hi = np.percentile(d, [2.5, 97.5])
        sh = r[hk][pick].sum() / max(r[nk][pick].sum(), 1)
        sb = results[base][hk][pick].sum() / max(results[base][nk][pick].sum(), 1)
        out[name] = dict(share=sh, diff=(sh - sb) * 100, lo=lo, hi=hi, swaps=r["swaps"], n=n)
    return out


def default_variants(f=0.7, w=2.0, weighted=False, boundaries=False):
    v = ["base", "lru", "pm", f"ff:{f}", f"ffe:{f}", f"tr:{w}", f"ff:{f}+tr:{w}", f"ff:{f}+pm", f"tr:{w}+pm", f"ff:{f}+tr:{w}+pm"]
    if weighted:
        v += ["pmw", f"ff:{f}+pmw", f"tr:{w}+pmw", f"ff:{f}+tr:{w}+pmw"]
    if boundaries:
        v += [f"trb:{w}", f"ff:{f}+trb:{w}"] + ([f"trb:{w}+pmw"] if weighted else [])
    return v


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
    ap.add_argument("--requests", metavar="JSON", help="conv_trace.py sidecar (end_record, conv per turn): marks turns and conversation switches")
    ap.add_argument("--stwt", metavar="FILE", help="STRATA_WINDOW_TRACE file recorded with STRATA_WINDOW_TRACE_WEIGHTS=1: its router weights are matched to the dump's windows by ids (for pmw)")
    ap.add_argument("--after", type=int, default=16, help="windows after a conversation switch that count as post-switch (default 16)")
    ap.add_argument("--json", metavar="OUT", help="also write the tables, turns and weight coverage as JSON")
    a = ap.parse_args()
    windows, k, starts, weights, total = load_dump(a.trace)
    init = R.load_profile(a.profile, a.slots)
    pol = R.Policy(every=a.every)
    wnote = "trace weight column is all zeros (or unit)"
    real = has_real_weights(weights)
    cover = dict(dump_real_weights=real, stwt=a.stwt, matched=None, matched_nonzero=None)
    if real:
        wnote = "the dump's weight column carries router weights"
    if a.stwt:
        w2, m, nz = attach_stwt(windows, a.stwt)
        cover.update(matched=m, matched_nonzero=nz)
        if nz > 0:
            weights, real = [x for x in w2], True
            wnote = f"router weights from {a.stwt}: {m} of {len(windows)} windows matched, {nz} with nonzero weights; unmatched windows count +1"
        else:
            wnote += f"; {a.stwt}: {m} of {len(windows)} windows matched, 0 with nonzero weights (pmw NOT available)"
    cover["pmw_available"] = real
    turns, labels, sw = [], None, []
    if a.requests:
        reqs = json.load(open(a.requests, encoding="utf-8"))
        tix, turns = segment(reqs, starts, total)
        sw = switch_windows(turns)
    names = a.variants or default_variants(weighted=real, boundaries=bool(turns))
    if a.sweep:
        names += [f"ff:{f}" for f in (0.3, 0.5, 0.85)] + [f"ffe:{f}" for f in (0.3, 0.5, 0.85)] + [f"tr:{w}" for w in (0.5, 1, 4, 8)]
        if turns:
            names += [f"trb:{w}" for w in (0.5, 1, 4, 8)]
    names = list(dict.fromkeys(names))
    if any(parse(nm)["pmw"] for nm in names) and not real:
        raise SystemExit("pmw needs router weights: pass --stwt (STRATA_WINDOW_TRACE_WEIGHTS=1 recording) or a dump with nonzero weights")
    print(f"{len(windows)} windows, k={k}, slots {a.slots}, every {a.every}, circular block bootstrap block={a.block} x {a.nboot}; "
          f"{wnote}; 'pm' is a position-weight PROXY (r={a.pm_r}), 'pmw' the real STRATA_ADAPT_PROB_MASS")
    if turns:
        print(f"{len(turns)} turns, {len(sw)} conversation switches (windows {sw}); post-switch = the {a.after} windows after a switch")
        for t in turns:
            print(f"  turn {t['turn']:2d} conv {t['conv']} ({t['domain']}) first window {t['first']} windows {t['n']}{' SWITCH' if t['switch'] else ''}")
    results = {}
    for nm in names:
        results[nm] = simulate(windows, init, a.slots, pol, parse(nm), pm_r=a.pm_r, k=k, weights=weights if real else None, boundaries=sw)
    tables = {}
    jobs = [("e", "share of routed entries served", None), ("d", "share of distinct (window, layer) experts served", None)]
    if turns and sw:
        post, other = subset_masks(len(windows), sw, a.after)
        jobs += [("e", f"share of routed entries served, POST-SWITCH windows ({int(post.sum())})", post),
                 ("e", f"share of routed entries served, other windows ({int(other.sum())})", other)]
    for metric, label, sub in jobs:
        s = summarize(results, nboot=a.nboot, block=a.block, metric=metric, subset=sub)
        tables[label] = s
        print(f"\n{label}; difference vs base in points, 95% CI over windows")
        print(f"{'variant':22} {'share':>7} {'diff':>7} {'CI lo':>7} {'CI hi':>7} {'swaps/win':>10}")
        for nm in names:
            r = s[nm]
            print(f"{nm:22} {r['share']:7.4f} {r['diff']:+7.2f} {r['lo']:+7.2f} {r['hi']:+7.2f} {r['swaps']:10.2f}")
    if a.json:
        def clean(o):
            if isinstance(o, dict):
                return {str(x): clean(y) for x, y in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(y) for y in o]
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            return o
        Path(a.json).write_text(json.dumps(clean(dict(windows=len(windows), k=k, slots=a.slots, every=a.every, weights=cover, turns=turns, switch_windows=sw, tables=tables)), indent=1))


if __name__ == "__main__":
    main()
