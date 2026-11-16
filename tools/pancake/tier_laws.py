"""tier_laws.py - pancake-strata MOD_PLAN.md section 6 items 19 and 26: the expert-placement laws on the offline cache
simulator (cache_learn.Layered, the 2026-11-08 routing trace and the profile).  Offline only: no GPU, no engine.

Cache model = cache_learn.py's: per layer the profile's slot budget, a window's routed experts are served from the
residents as they were when the window started (hit = routed entry, token x k, whose expert is resident), decisions are taken
after the window.  Cost = cache_learn.cpu_ms: 48.7 us per distinct CPU miss + alpha (0.5) x 117 us per promotion.

--protect F (0..1, a comma list runs each): the first ceil(F x SLOTS) pairs of the profile (rank order) are a static core that
is never evicted.  This mirrors the engine switch STRATA_ADAPT_PROTECT=F (include/strata/program/tier_protect.hpp): protected =
slots with index < ceil(F x slots), and the startup fill hands the slots out in profile rank order.  F = 1 is the static cache.
A victim with count 0 is a free slot in effect: "static core plus empty slots that fill on demand" = protect F + the tier.

Policies (policy spec = name[:p1,p2,..]; all honour --protect; swaps per window are capped by --cap, tier 32 like the preset,
the others uncapped like cache_learn's kmiss):
  static           no change
  tier             the engine rule: usage += routed count, a round every window, usage *= decay (0.7) after it, candidate =
                   missed with usage >= 2.0, victim = lowest-usage unprotected resident NOT routed this window, swap while
                   cand >= victim + mingain (1.5, the engine's --adapt-min-gain).  tier:decay,mingain
  tier-any         the same but any unprotected resident may be the victim (cache_learn's tier / rp.adapt_round)
  kmiss:k,W        admit after k misses in the last W windows, evict the least recently used unprotected (STRATA_ADAPT_KMISS)
  breakeven:H,m    law 1: EWMA use rate r (alpha 0.3), admit iff r x H >= theta (theta = 0.5 x 117 / 48.7 = 1.20), swap only if
                   r_cand > r_victim x (1 + m)   (H windows horizon, m margin 0.25)
  bayes:H,a0,z,pmax,d  law 2: posterior mean (a0 x p0 + S) / (a0 + N), S and N the d-decayed uses and windows, prior
                   p0 = pmax x (rank in layer + 1)^-z from the profile order; admit iff post x H >= theta, swap if cand > victim
  layerdecay       law 3: per-layer decay from {0.3, 0.5, 0.7, 0.85, 0.95} picked on the FIRST HALF of the trace, reported on the
                   SECOND HALF against the global 0.7 (hold-out; the table "holdout" only)
  survival:H       law 4: online reuse-gap histogram per layer, state = the gap that preceded the last use (6 buckets), P(reuse
                   within H windows | state, idle age) smoothed to the layer's pooled rate; avoided misses p / (1 - p) >= theta
                   admits, swap if p_cand > p_victim
  conv:w           law 6, conversation-tied (item 26): the tier, and at every turn start the usage is seeded with w x the decayed
                   counts (decay 0.7 per window of that conversation) of that conversation's earlier turns
  ghost:d,q,k,W    ARC-style: per layer a ghost list of the last G (= layer budget) evicted experts; a miss on a ghost is admitted
                   at once and raises the layer's adaptive protect fraction by d, q windows in a row without a ghost hit lower
                   it by d (reported min / mean / max over layers and windows); other misses: k-th miss k/W, evict LRU
  phase:K,H        phase-Markov (hold-out only): per layer K k-means phases of the training half's window count vectors, a
                   transition matrix split into continue / switch (turn structure), predicted demand of the next window =
                   sum P(next phase | phase, ctx) x phase profile, admit iff demand x H >= theta; prints the top-1 next-phase
                   accuracy for continue vs switch (and the stay-in-phase baseline)
  seedfile:w       law 5 (cross-layer / prompt-fed signals) CANNOT be simulated here: there is no prompt data in the trace.  It is
                   the same seeding machinery fed from --seed-file FILE.npz (key "counts", per layer and expert, shape
                   (layers, experts)): usage += w x counts at window 0 and at every turn start.  Whoever has prompt-derived or
                   cross-layer counts can run it; without the file the policy is skipped.

Every policy row also gets (second table): thrash = the share of evictions whose expert is routed again on that layer within
1 / 4 / 16 windows, the net benefit per promotion = (windows the expert was routed while resident, before its eviction or the
trace end) x 48.7 us - 0.5 x 117 us (mean and the share below zero); and the landing-pad estimate: "gap miss/w" = distinct CPU
misses at window t+1 on experts that were victims of round t (the engine marks the victim non-resident at once and the CPU
computes it until the copy lands, generate.cpp adapt()), and "saved P16/P32/P64" = how many of those a pad of 16 / 32 / 64 spare
slots would save if the first P victims of a round stayed resident until their replacement landed (the old slot is the next
pad).  The copies are instant in this model, so the candidates' landing delay is not priced (a lower bound of the gap cost).

Turn / conversation structure: --turns FILE.json (conv_trace.py requests.json: [{end_record, conv, domain, ...}]; end_record is
turned into a window index as replay_seeding.requests_mode does: bisect_left on the first record of every complete window) or
--cuts with --convs a,b,a,c (one label per cut part).  --interleave N adds the SYNTHETIC agent trace of cache_learn (the parts
cut into N-window turns played round-robin; the conversation of a turn = its part).

--validate-conv prints, per turn structure: V1 top-N-per-layer expert-set Jaccard (N = the layer budget) between adjacent turns of
one conversation vs adjacent turns of different conversations; V2 hit-rate and CPU ms lift of the conversation-seeded tier vs plain
tier; V3 Jaccard of the first m = 1..16 windows' top sets vs the previous turn's set after a conversation switch vs after a
same-conversation resume; V4 lead time: for H in 1,2,4,8 the share of CPU misses at window t that were in the set predicted at
t - H (top K = 4 non-resident experts per layer by the conversation's decayed counts, residency of the plain tier) and the
precision of those predictions.

    python tools/pancake/tier_laws.py PROFILE.bin SLOTS TRACE --cuts 11,96,157 --names read4k,short,edit,long
           [--protect 0,0.5,0.75,0.9] [--policies P ...] [--interleave 8] [--turns F.json] [--convs a,b,a,c]
           [--validate-conv] [--seed-file F.npz] [--parts] [--json OUT.json]
"""
import argparse
import bisect
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cache_learn as cl  # noqa: E402
import replay_cache_policy as rp  # noqa: E402

US_MISS, US_DMA = cl.US_MISS, cl.US_DMA
THETA = 0.5 * 117.0 / 48.7          # promotion cost in CPU-miss units (A6 calibration, contention alpha 0.5)
MINCAND = 2.0
DECAYS = (0.3, 0.5, 0.7, 0.85, 0.95)
DEFAULT_POLICIES = ["static", "tier", "tier-any", "kmiss:2,4", "kmiss:2,8", "kmiss:1,1", "kmiss:1,2",
                    "breakeven:4,0.25", "breakeven:8,0.25", "breakeven:16,0.25", "bayes:8,4,1.0,0.6,0.9",
                    "bayes:16,4,1.0,0.6,0.9", "survival:4", "survival:8", "conv:0.5", "conv:1", "conv:2", "ghost:0.02,8,2,8", "ghost:0.05,4,2,8",
                    "phase:4", "phase:8", "phase:16"]


# ------------------------------------------------------------------------------------------------ data

def dense(tr, ne):
    """[windows, layers, experts] int32 routed-entry counts of a cache_learn.Layered."""
    d = np.zeros((tr.n, tr.nl, ne), np.int32)
    for l in range(tr.nl):
        for t, c in enumerate(tr.cnt[l]):
            for e, k in c.items():
                d[t, l, e] = k
    return d


def init_residents(init, nl, ne):
    res = np.zeros((nl, ne), bool)
    res[init[:, 0], init[:, 1]] = True
    return res


def protect_slots(f, slots):
    """ceil(f x slots) clamped to [0, slots], the engine's tier_protect_slots."""
    if not (f > 0) or not math.isfinite(f) or slots <= 0:
        return 0
    if f >= 1.0:
        return slots
    return min(slots, max(0, int(math.ceil(f * slots - 1e-9))))


def protect_mask(init, nl, ne, n_prot):
    m = np.zeros((nl, ne), bool)
    p = init[:n_prot]
    m[p[:, 0], p[:, 1]] = True
    return m


def interleave_turns(parts, names, turn):
    """cache_learn.interleave plus its turn list [(start, end, conv)] (SYNTHETIC)."""
    pieces = [[(p.slice(i, min(i + turn, p.n))) for i in range(0, p.n, turn)] for p in parts]
    seqs = [[] for _ in range(parts[0].nl)]
    turns, k, pos = [], 0, 0
    while any(k < len(pc) for pc in pieces):
        for name, pc in zip(names, pieces):
            if k < len(pc):
                for l in range(parts[0].nl):
                    seqs[l].extend(pc[k].seqs[l])
                turns.append((pos, pos + pc[k].n, name))
                pos += pc[k].n
        k += 1
    return cl.Layered(seqs, parts[0].nl), turns


def turns_from_requests(path, trace, n_windows):
    """[(start, end, conv)] from conv_trace.py's requests.json, replay_seeding.requests_mode's end_record -> window bisect."""
    import replay_seeding as rs
    reqs = json.load(open(path, encoding="utf-8"))
    _, complete = rs.window_records(trace)
    turns, w0 = [], 0
    for r in reqs:
        w1 = min(n_windows, bisect.bisect_left(complete, int(r["end_record"])))
        if w1 > w0:
            turns.append((w0, w1, str(r["conv"])))
        w0 = w1
    return turns


def turns_from_cuts(cuts, n, convs):
    edges = [0] + list(cuts) + [n]
    return [(edges[i], edges[i + 1], convs[i]) for i in range(len(edges) - 1)]


# ------------------------------------------------------------------------------------------------ policies

class Policy:
    """Decides after every window: swaps(t, C, res, prot) -> [(gain, layer, cand, victim)].  turn_start(t, conv) runs before the
    window t is served; observe is folded into swaps()."""
    name = "policy"
    cap = 0           # 0 = uncapped

    def start(self, nl, ne, res, prot, init):
        self.nl, self.ne = nl, ne

    def turn_start(self, t, conv):
        pass

    def extra(self):
        return {}

    def swaps(self, t, C, res, prot):
        return []


def pair_swaps(l, score, cand_mask, vict_mask, ok):
    """Best candidates against worst victims, paired in order; stops at the first pair that does not clear ok(cs, vs)."""
    ci = np.flatnonzero(cand_mask)
    if len(ci) == 0:
        return []
    vi = np.flatnonzero(vict_mask)
    if len(vi) == 0:
        return []
    ci = ci[np.argsort(-score[ci], kind="stable")]
    vi = vi[np.argsort(score[vi], kind="stable")]
    n = min(len(ci), len(vi))
    cs, vs = score[ci[:n]], score[vi[:n]]
    bad = np.flatnonzero(~ok(cs, vs))
    n = int(bad[0]) if len(bad) else n
    return [(float(cs[i] - vs[i]), l, int(ci[i]), int(vi[i])) for i in range(n)]


class Static(Policy):
    name = "static"


class Tier(Policy):
    """The engine's tier (decayed usage), optionally per-layer decay and seeded at turn starts (laws 3, 5, 6)."""

    def __init__(self, decay=0.7, mingain=1.5, exclude_used=True, cap=32, conv_w=0.0, seed=None, seed_w=0.0, name="tier"):
        self.decay = decay
        self.mingain, self.exclude_used, self.cap = mingain, exclude_used, cap
        self.conv_w, self.seed, self.seed_w, self.name = conv_w, seed, seed_w, name

    def start(self, nl, ne, res, prot, init):
        super().start(nl, ne, res, prot, init)
        self.u = np.zeros((nl, ne))
        self.A = {}                    # conversation -> decayed counts of its windows
        self.conv = None
        d = np.asarray(self.decay, float)
        self.d = d.reshape(-1, 1) if d.ndim else float(d)
        if self.seed is not None and self.seed_w > 0:
            self.u += self.seed_w * self.seed

    def turn_start(self, t, conv):
        self.conv = conv
        if t > 0 and self.seed is not None and self.seed_w > 0:
            self.u += self.seed_w * self.seed
        if self.conv_w > 0 and conv in self.A:
            self.u += self.conv_w * self.A[conv]

    def swaps(self, t, C, res, prot):
        self.u += C
        if self.conv_w > 0 and self.conv is not None:
            a = self.A.get(self.conv)
            self.A[self.conv] = (a * self.d if a is not None else 0.0) + C
        out = []
        cand = ~res & (self.u >= MINCAND)
        vict = res & ~prot
        if self.exclude_used:
            vict &= (C == 0)
        mg = self.mingain
        for l in np.flatnonzero(cand.any(1) & vict.any(1)):
            out += pair_swaps(l, self.u[l], cand[l], vict[l], lambda cs, vs: cs >= vs + mg)
        self.u *= self.d
        return out


class KMiss(Policy):
    """cache_learn / time_ceiling kmiss: admit after k misses in the last W windows, evict LRU (unprotected)."""

    def __init__(self, k, W):
        self.k, self.W, self.name = k, W, f"kmiss {k}/{W}"

    def start(self, nl, ne, res, prot, init):
        super().start(nl, ne, res, prot, init)
        self.lists = []
        for l in range(nl):
            self.lists.append([int(e) for e in init[init[:, 0] == l][:, 1]])
        self.recent = [dict() for _ in range(nl)]

    def swaps(self, t, C, res, prot):
        out = []
        for l in range(self.nl):
            u = np.flatnonzero(C[l]).tolist()
            lst = self.lists[l]
            inres = set(lst)
            cur = set(u)
            for e in u:
                if e in inres:
                    lst.remove(e)
                    lst.append(e)
            rec = self.recent[l]
            for e in u:
                if e in inres:
                    continue
                r = [x for x in rec.get(e, []) if x > t - self.W] + [t]
                rec[e] = r
                if len(r) >= self.k:
                    victim = next((x for x in lst if x not in cur and not prot[l, x]), None)
                    if victim is None:
                        victim = next((x for x in lst if not prot[l, x]), None)
                    if victim is None:
                        continue
                    lst.remove(victim)
                    lst.append(e)
                    inres.discard(victim)
                    inres.add(e)
                    out.append((0.0, l, e, victim))
                    rec.pop(e, None)
        return out


class RateRule(Policy):
    """Laws 1 and 2: a per-expert expected use rate per window; admit iff rate x H >= theta."""
    cap = 0

    def __init__(self, H, margin):
        self.H, self.margin = H, margin

    def score(self):
        raise NotImplementedError

    def update(self, C):
        raise NotImplementedError

    def swaps(self, t, C, res, prot):
        self.update(C)
        s = self.score()
        cand = ~res & (s * self.H >= THETA)
        vict = res & ~prot & (C == 0)
        m = self.margin
        out = []
        for l in np.flatnonzero(cand.any(1) & vict.any(1)):
            out += pair_swaps(l, s[l], cand[l], vict[l], lambda cs, vs: cs > vs * (1.0 + m))
        return out


class BreakEven(RateRule):
    def __init__(self, H, margin, alpha=0.3):
        super().__init__(H, margin)
        self.alpha, self.name = alpha, f"breakeven H{H} m{margin:g}"

    def start(self, nl, ne, res, prot, init):
        super().start(nl, ne, res, prot, init)
        self.r = np.zeros((nl, ne))

    def update(self, C):
        self.r += self.alpha * ((C > 0) - self.r)

    def score(self):
        return self.r


def profile_prior(ranked, nl, ne, pmax, z):
    """p0[l, e] = pmax x (rank in layer + 1)^-z from the profile order (all ranked pairs); unranked experts get the last rank."""
    rank = np.full((nl, ne), float(ne))
    cnt = np.zeros(nl, int)
    for l, e in ranked:
        rank[l, e] = cnt[l]
        cnt[l] += 1
    rank = np.where(rank >= ne, np.maximum(cnt[:, None], 1).astype(float), rank)
    return pmax * (rank + 1.0) ** (-z)


class Bayes(RateRule):
    def __init__(self, H, a0, z, pmax, d, ranked):
        super().__init__(H, 0.0)
        self.a0, self.z, self.pmax, self.d, self.ranked = a0, z, pmax, d, ranked
        self.name = f"bayes H{H} a{a0:g} z{z:g} p{pmax:g} d{d:g}"

    def start(self, nl, ne, res, prot, init):
        super().start(nl, ne, res, prot, init)
        self.p0 = profile_prior(self.ranked, nl, ne, self.pmax, self.z)
        self.S = np.zeros((nl, ne))
        self.N = 0.0

    def update(self, C):
        self.S = self.d * self.S + (C > 0)
        self.N = self.d * self.N + 1.0

    def score(self):
        return (self.a0 * self.p0 + self.S) / (self.a0 + self.N)


class Survival(Policy):
    """Law 4: P(next reuse within H windows | the gap that preceded the last use, the idle age), learned online per layer."""
    G, S, M = 16, 6, 2.0

    def __init__(self, H):
        assert 1 <= H <= self.G
        self.H, self.name = H, f"survival H{H}"

    @staticmethod
    def bucket(g):
        return np.where(g <= 1, 1, np.where(g == 2, 2, np.where(g <= 4, 3, np.where(g <= 8, 4, 5))))

    def start(self, nl, ne, res, prot, init):
        super().start(nl, ne, res, prot, init)
        self.last = np.full((nl, ne), -1, int)
        self.st = np.zeros((nl, ne), int)
        self.Nh = np.zeros((nl, self.S, self.G + 2))     # [layer, state, gap 1..G, G+1 = not reused within G]
        self.li = np.arange(nl)[:, None]

    def table(self):
        G, H = self.G, self.H
        cum = np.cumsum(self.Nh, axis=2)                  # cum[..., k] = windows with gap <= k
        a = np.arange(G + 1)
        hi = np.minimum(a + H, G)
        num = cum[:, :, hi] - cum[:, :, a]
        den = cum[:, :, G + 1][:, :, None] - cum[:, :, a]
        pnum, pden = num.sum(1), den.sum(1)
        pool = (pnum + 0.05) / (pden + 1.0)               # layer's pooled P(reuse within H | age)
        return (num + self.M * pool[:, None, :]) / (den + self.M)

    def swaps(self, t, C, res, prot):
        used = C > 0
        gap = t - self.last
        rec = used & (self.last >= 0) & (gap <= self.G + 1)
        li, ei = np.nonzero(rec)
        if len(li):
            g = np.minimum(gap[li, ei], self.G + 1)
            np.add.at(self.Nh, (li, self.st[li, ei], g), 1.0)
        exp = ~used & (self.last >= 0) & (self.last == t - self.G - 1)
        li, ei = np.nonzero(exp)
        if len(li):
            np.add.at(self.Nh, (li, self.st[li, ei], np.full(len(li), self.G + 1)), 1.0)
        self.st = np.where(used, np.where(self.last >= 0, self.bucket(np.maximum(gap, 1)), 0), self.st)
        self.last = np.where(used, t, self.last)
        age = np.minimum(t - self.last, self.G)
        p = self.table()[self.li, self.st, age]
        p = np.where(self.last >= 0, np.minimum(p, 0.99), 0.0)
        gainv = p / (1.0 - p)                              # expected avoided misses (geometric reuse)
        cand = ~res & (gainv >= THETA)
        vict = res & ~prot & (C == 0)
        out = []
        for l in np.flatnonzero(cand.any(1) & vict.any(1)):
            out += pair_swaps(l, p[l], cand[l], vict[l], lambda cs, vs: cs > vs)
        return out


class Ghost(Policy):
    """ARC-style ghost list: per layer the last G (= the layer budget) evicted experts.  A miss on a ghost is admitted at once
    and raises the layer's adaptive protect fraction by delta (the first ceil(f x budget) experts of the layer's profile order
    are not evictable, on top of --protect); `quiet` windows in a row without a ghost hit lower it by delta.  Other misses go
    through the k-th-miss rule (k, W), eviction LRU (unprotected, not routed this window)."""

    def __init__(self, delta=0.02, quiet=8, k=2, W=8, f0=0.0):
        self.delta, self.quiet, self.k, self.W, self.f0 = delta, int(quiet), int(k), int(W), f0
        self.name = f"ghost d{delta:g} q{int(quiet)} k{int(k)}/{int(W)}"

    def start(self, nl, ne, res, prot, init):
        super().start(nl, ne, res, prot, init)
        from collections import OrderedDict
        self.core = [[int(e) for e in init[init[:, 0] == l][:, 1]] for l in range(nl)]
        self.lists = [list(c) for c in self.core]
        self.f = np.full(nl, float(self.f0))
        self.ghost = [OrderedDict() for _ in range(nl)]
        self.recent = [dict() for _ in range(nl)]
        self.run_q = np.zeros(nl, int)
        self.fmin, self.fmax, self.fsum, self.fn = 1.0, 0.0, 0.0, 0

    def swaps(self, t, C, res, prot):
        out = []
        for l in range(self.nl):
            u = np.flatnonzero(C[l]).tolist()
            lst = self.lists[l]
            inres = set(lst)
            cur = set(u)
            for e in u:
                if e in inres:
                    lst.remove(e)
                    lst.append(e)
            b = len(self.core[l])
            adaptive = set(self.core[l][:int(math.ceil(self.f[l] * b - 1e-9))])
            gl, rec, hit = self.ghost[l], self.recent[l], False
            for e in u:
                if e in inres:
                    continue
                if e in gl:
                    hit = True
                    del gl[e]
                    admit = True
                else:
                    r = [x for x in rec.get(e, []) if x > t - self.W] + [t]
                    rec[e] = r
                    admit = len(r) >= self.k
                if not admit:
                    continue
                victim = next((x for x in lst if x not in cur and not prot[l, x] and x not in adaptive), None)
                if victim is None:
                    victim = next((x for x in lst if not prot[l, x] and x not in adaptive), None)
                if victim is None:
                    continue
                lst.remove(victim)
                lst.append(e)
                inres.discard(victim)
                inres.add(e)
                rec.pop(e, None)
                gl[victim] = True
                while len(gl) > b:
                    gl.popitem(last=False)
                out.append((0.0, l, e, victim))
            if hit:
                self.f[l] = min(1.0, self.f[l] + self.delta)
                self.run_q[l] = 0
            else:
                self.run_q[l] += 1
                if self.run_q[l] >= self.quiet:
                    self.f[l] = max(0.0, self.f[l] - self.delta)
                    self.run_q[l] = 0
        self.fmin, self.fmax = min(self.fmin, float(self.f.min())), max(self.fmax, float(self.f.max()))
        self.fsum += float(self.f.mean())
        self.fn += 1
        return out

    def extra(self):
        return {"protect_frac": {"min": self.fmin, "mean": self.fsum / max(1, self.fn), "max": self.fmax}}


class Phase(Policy):
    """Law 7, the phase-Markov matrix: per layer the training windows are clustered into K phases (k-means on the L2-normalised
    per-window expert-count vectors, cosine, numpy only), with a phase profile (the share of the cluster's windows using each
    expert) and a transition matrix T[ctx][k][k'] for ctx = continue (the next window is in the same conversation, also the
    next turn of it) or switch (it starts a turn of another conversation), each row smoothed towards the pooled next-phase
    distribution of its ctx.  After every window: current phase = nearest centroid, predicted demand of the next window
    d = sum_k' T[ctx][k][k'] x profile[k'] (ctx known from the turn structure), admit iff d x H >= theta, swap if d_cand >
    d_victim x (1 + margin).  Needs fit(first half); only reported hold-out (the second half)."""
    needs_fit = True

    def __init__(self, K, H=4, margin=0.25, iters=15):
        self.K, self.H, self.margin, self.iters = K, H, margin, iters
        self.name = f"phase K{K} H{H}"

    @staticmethod
    def _norm(X):
        n = np.linalg.norm(X, axis=-1, keepdims=True)
        return X / np.maximum(n, 1e-12)

    def _kmeans(self, X, K, rng):
        n = len(X)
        cent = [X[rng.integers(n)]]
        for _ in range(1, K):                       # k-means++ on cosine distance
            d = np.min(1.0 - X @ np.array(cent).T, axis=1).clip(0)
            cent.append(X[rng.choice(n, p=d / d.sum())] if d.sum() > 1e-12 else X[rng.integers(n)])
        cent = np.array(cent)
        for _ in range(self.iters):
            lab = np.argmax(X @ cent.T, axis=1)
            for k in range(K):
                m = lab == k
                if m.any():
                    cent[k] = self._norm(X[m].mean(0))
        return cent, np.argmax(X @ cent.T, axis=1)

    def fit(self, Dtr, turns_tr):
        n, nl, ne = Dtr.shape
        rng = np.random.default_rng(0)
        ctx_seq = ctx_sequence(n, turns_tr)
        self.cent, self.prof, self.T = [], [], []
        for l in range(nl):
            X = self._norm(Dtr[:, l, :].astype(float))
            K = min(self.K, n)
            cent, lab = self._kmeans(X, K, rng)
            used = (Dtr[:, l, :] > 0).astype(float)
            glob = used.mean(0)
            prof = np.array([used[lab == k].mean(0) if (lab == k).any() else glob for k in range(K)])
            T = {}
            for ctx in ("continue", "switch"):
                M = np.zeros((K, K))
                for t in range(n - 1):
                    if ctx_seq[t] == ctx:
                        M[lab[t], lab[t + 1]] += 1
                q = (M.sum(0) + 0.5) / (M.sum() + 0.5 * K)
                T[ctx] = (M + 1.0 * q[None, :]) / (M.sum(1, keepdims=True) + 1.0)
            self.cent.append(cent)
            self.prof.append(prof)
            self.T.append(T)
        return self

    def start(self, nl, ne, res, prot, init):
        super().start(nl, ne, res, prot, init)
        self.pend = None
        self.acc = {"continue": [0, 0, 0], "switch": [0, 0, 0]}      # n, top-1 matches, stay-baseline matches
        self.prev = None

    def swaps(self, t, C, res, prot):
        nl, ne = self.nl, self.ne
        X = self._norm(C.astype(float))
        ph = np.array([int(np.argmax(self.cent[l] @ X[l])) for l in range(nl)])
        if self.pend is not None:
            top, ctx_used = self.pend
            a = self.acc[ctx_used]
            a[0] += nl
            a[1] += int((top == ph).sum())
            a[2] += int((self.prev == ph).sum())
        ctx = self.ctx_next
        d = np.zeros((nl, ne))
        top = np.zeros(nl, int)
        for l in range(nl):
            row = self.T[l][ctx][ph[l]]
            d[l] = row @ self.prof[l]
            top[l] = int(np.argmax(row))
        self.pend, self.prev = (top, ctx), ph
        cand = ~res & (d * self.H >= THETA)
        vict = res & ~prot & (C == 0)
        m = self.margin
        out = []
        for l in np.flatnonzero(cand.any(1) & vict.any(1)):
            out += pair_swaps(l, d[l], cand[l], vict[l], lambda cs, vs: cs > vs * (1.0 + m))
        return out

    def extra(self):
        return {"phase_acc": {c: {"n": a[0], "top1": a[1] / max(1, a[0]), "stay_baseline": a[2] / max(1, a[0])}
                              for c, a in self.acc.items()}}


def ctx_sequence(n, turns):
    """ctx[t] = "switch" if window t+1 starts a turn of another conversation than window t's, else "continue"."""
    ctx = ["continue"] * n
    if turns:
        conv = {}
        for s, e, c in turns:
            for t in range(s, min(e, n)):
                conv[t] = c
        for s, e, c in turns:
            if 0 < s < n and conv.get(s - 1) != c:
                ctx[s - 1] = "switch"
    return ctx


def slice_turns(turns, a, b):
    """The turn list of windows [a, b), re-based to 0."""
    if not turns:
        return turns
    return [(max(s, a) - a, min(e, b) - a, c) for s, e, c in turns if min(e, b) > max(s, a)]


# ------------------------------------------------------------------------------------------------ simulator

PADS = (16, 32, 64)
SWAPBACK = (1, 4, 16)


def run(D, init, policy, n_prot, turns=None, record=False):
    """Replay D ([windows, layers, experts] counts) from the profile's residents.  Returns the stats dict (hits fraction,
    miss_w, promo_w, plus "thrash": eviction swap-back rates and the net benefit per promotion, "gap": the CPU misses on the
    previous round's victims and what landing pads would save, and the policy's own extras), per-layer miss / promo arrays,
    and with record the residency at the start of every window."""
    n, nl, ne = D.shape
    res = init_residents(init, nl, ne)
    prot = protect_mask(init, nl, ne, n_prot)
    policy.start(nl, ne, res, prot, init)
    starts = {s: c for s, _e, c in (turns or [])}
    if not turns:
        starts = {0: "all"}
    hits = ent = 0
    miss = np.zeros(n)
    promo = np.zeros(n)
    miss_l = np.zeros(nl)
    promo_l = np.zeros(nl)
    hist = np.zeros((n, nl, ne), bool) if record else None
    evictions = []                 # (round, layer, expert)
    since = {}                     # (layer, expert) -> round of its promotion (still resident)
    promos = []                    # (round promoted, round evicted or None, layer, expert)
    last_victims = {}              # (layer, expert) -> rank in the previous round's swap order
    gap = 0
    saved = {P: 0 for P in PADS}
    conv_now = None
    for t in range(n):
        if t in starts:
            policy.turn_start(t, starts[t])
            conv_now = starts[t]
        nxt = starts.get(t + 1)
        policy.ctx_next = "switch" if (nxt is not None and nxt != conv_now) else "continue"
        C = D[t]
        if record:
            hist[t] = res
        hits += int(C[res].sum())
        ent += int(C.sum())
        mm = (C > 0) & ~res
        miss[t] = mm.sum()
        miss_l += mm.sum(1)
        for (l, v), rk in last_victims.items():
            if mm[l, v]:
                gap += 1
                for P in PADS:
                    if rk < P:
                        saved[P] += 1
        last_victims = {}
        sw = policy.swaps(t, C, res, prot)
        if sw:
            if policy.cap and len(sw) > policy.cap:
                sw.sort(key=lambda x: -x[0])
                sw = sw[:policy.cap]
            for rk, (_g, l, c, v) in enumerate(sw):
                assert res[l, v] and not res[l, c] and not prot[l, v]
                res[l, v] = False
                res[l, c] = True
                promo_l[l] += 1
                evictions.append((t, l, v))
                last_victims[(l, v)] = rk
                if (l, v) in since:
                    promos.append((since.pop((l, v)), t, l, v))
                since[(l, c)] = t
            promo[t] = len(sw)
    for (l, c), tp in since.items():
        promos.append((tp, None, l, c))
    r = cl.stats(hits, ent, miss, promo)
    cum = np.cumsum(D > 0, axis=0, dtype=np.int32) if (evictions or promos) else None
    th = {"evictions": len(evictions), "promotions": len(promos)}
    for N in SWAPBACK:
        k = sum(1 for t, l, v in evictions if cum[min(t + N, n - 1), l, v] - cum[t, l, v] > 0)
        th[f"swapback{N}"] = k / len(evictions) if evictions else 0.0
    nets = []
    for tp, te, l, c in promos:
        avoided = int(cum[(te if te is not None else n - 1), l, c] - cum[tp, l, c])
        nets.append(avoided * US_MISS - 0.5 * 117.0)
    th["net_us_mean"] = float(np.mean(nets)) if nets else 0.0
    th["neg_share"] = float(np.mean([x < 0 for x in nets])) if nets else 0.0
    r["thrash"] = th
    r["gap"] = {"gap_per_w": gap / n, "saved_per_w": {str(P): saved[P] / n for P in PADS}}
    r.update(policy.extra())
    return r, miss_l, promo_l, hist


def parse_policy(spec, ctx):
    name, _, arg = spec.partition(":")
    a = [x for x in arg.split(",") if x != ""]
    f = [float(x) for x in a]
    if name == "static":
        return Static()
    if name == "tier":
        return Tier(decay=f[0] if f else 0.7, mingain=f[1] if len(f) > 1 else 1.5)
    if name == "tier-any":
        return Tier(exclude_used=False, name="tier-any")
    if name == "kmiss":
        return KMiss(int(f[0]), int(f[1]))
    if name == "breakeven":
        return BreakEven(f[0] if f else 8, f[1] if len(f) > 1 else 0.25)
    if name == "bayes":
        v = (f + [8, 4, 1.0, 0.6, 0.9][len(f):]) if f else [8, 4, 1.0, 0.6, 0.9]
        return Bayes(v[0], v[1], v[2], v[3], v[4], ctx["ranked"])
    if name == "survival":
        return Survival(int(f[0]) if f else 8)
    if name == "conv":
        w = f[0] if f else 1.0
        return Tier(conv_w=w, name=f"conv w{w:g}")
    if name == "ghost":
        v = f + [0.02, 8, 2, 8][len(f):]
        return Ghost(v[0], v[1], v[2], v[3])
    if name == "phase":
        return Phase(int(f[0]) if f else 8, f[1] if len(f) > 1 else 4)
    if name == "seedfile":
        if ctx.get("seed") is None:
            return None
        w = f[0] if f else 1.0
        return Tier(seed=ctx["seed"], seed_w=w, conv_w=0.0, name=f"seedfile w{w:g}")
    raise SystemExit(f"unknown policy {spec}")


def pick_layer_decay(D1, init, n_prot):
    """Law 3: per layer the decay (of DECAYS) with the lowest cost on D1 (the first half), the tier run once per decay."""
    nl = D1.shape[1]
    cost = np.zeros((len(DECAYS), nl))
    for i, d in enumerate(DECAYS):
        r, ml, pl, _ = run(D1, init, Tier(decay=d), n_prot)
        cost[i] = ml * US_MISS + 0.5 * pl * US_DMA
    return np.array(DECAYS)[cost.argmin(0)]


def run_holdout(D, init, specs, ctx, n_prot, turns=None):
    """Law 3 / law 7 honest hold-out: decays picked and phases fitted on the first half, every policy reported on the second
    half (cold start from the profile)."""
    h = D.shape[0] // 2
    d1, d2 = D[:h], D[h:]
    t1, t2 = slice_turns(turns, 0, h), slice_turns(turns, h, D.shape[0])
    dec = pick_layer_decay(d1, init, n_prot)
    out = {}
    for spec in specs:
        if spec == "layerdecay":
            continue
        p = parse_policy(spec, ctx)
        if p is None:
            continue
        if getattr(p, "needs_fit", False):
            p.fit(d1, t1)
        out[p.name] = run(d2, init, p, n_prot, t2)[0]
    out["layerdecay"] = run(d2, init, Tier(decay=dec, name="layerdecay"), n_prot, t2)[0]
    return out, {str(d): int((dec == d).sum()) for d in DECAYS}


def table(name, res, alpha):
    ref = next((k for k in res if k == "kmiss 2/8"), None)
    sref = res.get("static")
    lines = [f"== {name}", f"{'policy':26s} {'hit %':>7s} {'miss/w':>8s} {'promo/w':>8s} {'CPU ms/w':>9s} "
             f"{'vs kmiss 2/8':>13s} {'vs static':>10s}"]
    for k, r in res.items():
        ms = cl.cpu_ms(r, alpha)
        a = f"{ms - cl.cpu_ms(res[ref], alpha):+9.2f}" if ref else "      n/a"
        b = f"{ms - cl.cpu_ms(sref, alpha):+9.2f}" if sref else "      n/a"
        lines.append(f"{k:26s} {r['hits'] * 100:6.2f}% {r['miss_w']:8.1f} {r['promo_w']:8.1f} {ms:9.2f} {a:>13s} {b:>10s}")
    if ref:
        d = {k: (r["hits"] - res[ref]["hits"]) * 100 for k, r in res.items()}
        lines.append("   hit points vs kmiss 2/8: " + ", ".join(f"{k} {v:+.2f}" for k, v in d.items()))
    if sref:
        d = {k: (r["hits"] - sref["hits"]) * 100 for k, r in res.items()}
        lines.append("   hit points vs static:    " + ", ".join(f"{k} {v:+.2f}" for k, v in d.items()))
    return "\n".join(lines)


def thrash_table(name, res):
    lines = [f"== {name}: thrash and landing pads",
             f"{'policy':26s} {'evict/w':>8s} {'back<=1':>8s} {'back<=4':>8s} {'back<=16':>9s} {'net us/promo':>13s} {'neg %':>6s} "
             f"{'gap miss/w':>11s} {'saved P16':>10s} {'P32':>7s} {'P64':>7s}"]
    for k, r in res.items():
        th, g = r["thrash"], r["gap"]
        lines.append(f"{k:26s} {r['promo_w']:8.1f} {th['swapback1'] * 100:7.1f}% {th['swapback4'] * 100:7.1f}% "
                     f"{th['swapback16'] * 100:8.1f}% {th['net_us_mean']:13.1f} {th['neg_share'] * 100:5.1f}% "
                     f"{g['gap_per_w']:11.2f} {g['saved_per_w']['16']:10.2f} {g['saved_per_w']['32']:7.2f} "
                     f"{g['saved_per_w']['64']:7.2f}")
        for key in ("protect_frac", "phase_acc"):
            if key in r:
                lines.append(f"{'':26s}   {key}: {r[key]}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ validation V1-V4

def turn_counts(D, s, e):
    return D[s:e].sum(0)                                   # [layers, experts]


def top_sets(cnt, k):
    """Per layer the k experts with the most entries (only experts used at all)."""
    out = []
    for l in range(cnt.shape[0]):
        kk = int(k[l]) if hasattr(k, "__len__") else int(k)
        idx = np.argsort(-cnt[l], kind="stable")[:kk]
        out.append(set(idx[cnt[l, idx] > 0].tolist()))
    return out


def jaccard(a, b):
    """Mean over layers of |a cap b| / |a cup b| (layers where both are empty skipped)."""
    v = [len(x & y) / len(x | y) for x, y in zip(a, b) if x | y]
    return float(np.mean(v)) if v else float("nan")


def validate_conv(D, init, budgets, turns, alpha, label):
    n, nl, ne = D.shape
    res = {"label": label, "turns": len(turns), "convs": sorted({c for _, _, c in turns})}
    tc_ = [turn_counts(D, s, e) for s, e, _ in turns]
    sets = [top_sets(c, budgets) for c in tc_]
    last_of = {}
    same, diff, diff_all = [], [], []
    for i, (_s, _e, c) in enumerate(turns):
        if c in last_of:
            same.append(jaccard(sets[last_of[c]], sets[i]))
        if i > 0 and turns[i - 1][2] != c:
            diff.append(jaccard(sets[i - 1], sets[i]))
        for j in range(i):
            if turns[j][2] != c:
                diff_all.append(jaccard(sets[j], sets[i]))
        last_of[c] = i
    res["V1"] = {"same_conv_adjacent": float(np.mean(same)) if same else None, "n_same": len(same),
                 "other_conv_adjacent": float(np.mean(diff)) if diff else None, "n_other_adjacent": len(diff),
                 "other_conv_all_pairs": float(np.mean(diff_all)) if diff_all else None, "n_other_all": len(diff_all)}
    # V2
    n_prot = 0
    plain = run(D, init, Tier(), n_prot, turns)[0]
    seeded = run(D, init, Tier(conv_w=1.0, name="conv w1"), n_prot, turns)[0]
    res["V2"] = {"plain_hit_pct": plain["hits"] * 100, "conv_hit_pct": seeded["hits"] * 100,
                 "hit_lift_points": (seeded["hits"] - plain["hits"]) * 100, "plain_cpu_ms_w": cl.cpu_ms(plain, alpha),
                 "conv_cpu_ms_w": cl.cpu_ms(seeded, alpha),
                 "cpu_ms_lift": cl.cpu_ms(plain, alpha) - cl.cpu_ms(seeded, alpha)}
    # V3: first m windows of turn i against the previous turn (switch) / the same conversation's last turn (resume)
    v3 = {"switch": {m: [] for m in range(1, 17)}, "resume": {m: [] for m in range(1, 17)}}
    last_of = {}
    for i, (s, e, c) in enumerate(turns):
        for m in range(1, 17):
            mm = min(m, e - s)
            first = turn_counts(D, s, s + mm)
            kk = np.minimum(budgets, (first > 0).sum(1))
            fs = top_sets(first, kk)
            if i > 0 and turns[i - 1][2] != c:
                v3["switch"][m].append(jaccard(fs, top_sets(tc_[i - 1], kk)))
            if c in last_of:
                v3["resume"][m].append(jaccard(fs, top_sets(tc_[last_of[c]], kk)))
        last_of[c] = i
    res["V3"] = {k: {str(m): (float(np.nanmean(v)) if v else None) for m, v in d.items()} for k, d in v3.items()}
    res["V3_n"] = {k: len(d[1]) for k, d in v3.items()}
    # V4: the set predicted H windows ahead from the conversation's decayed counts (non-resident top K = 4 per layer)
    _r, _ml, _pl, hist = run(D, init, Tier(), 0, turns, record=True)
    conv_at = np.full(n, "?", object)
    for s, e, c in turns:
        conv_at[s:e] = c
    A = {c: np.zeros((nl, ne)) for c in set(res["convs"]) | {"?"}}
    K = 4
    Hs = (1, 2, 4, 8)
    agg = {H: [0, 0, 0, 0, 0] for H in Hs}       # missed total, covered, predicted, predicted & missed, predicted & routed
    pending = {}
    for t in range(n):
        C = D[t]
        miss_t = (C > 0) & ~hist[t]
        for pred, H in pending.pop(t, []):
            a = agg[H]
            a[0] += int(miss_t.sum())
            for l in range(nl):
                pl = pred[l]
                if len(pl):
                    a[1] += int(miss_t[l, pl].sum())
                    a[2] += len(pl)
                    a[3] += int(miss_t[l, pl].sum())
                    a[4] += int((C[l, pl] > 0).sum())
        A[conv_at[t]] = A[conv_at[t]] * 0.7 + C
        for H in Hs:
            if t + H < n:
                c = conv_at[t + H]
                sc = np.where(hist[t], -1.0, A[c])               # non-resident only, residency at the time of the prediction
                pred = []
                for l in range(nl):
                    idx = np.argsort(-sc[l], kind="stable")[:K]
                    pred.append(idx[sc[l, idx] > 0])
                pending.setdefault(t + H, []).append((pred, H))
    res["V4"] = {str(H): {"miss_share_covered": a[1] / max(1, a[0]), "precision_missed": a[3] / max(1, a[2]),
                          "precision_routed": a[4] / max(1, a[2]), "misses": a[0], "predicted": a[2]} for H, a in agg.items()}
    return res


def print_validation(v):
    print(f"== V1-V4 validation: {v['label']} ({v['turns']} turns, conversations {', '.join(v['convs'])})")
    a = v["V1"]
    f = lambda x: "n/a" if x is None else f"{x:.3f}"
    print(f"V1 top-N-per-layer Jaccard: same conversation adjacent {f(a['same_conv_adjacent'])} (n {a['n_same']}), "
          f"different conversation adjacent {f(a['other_conv_adjacent'])} (n {a['n_other_adjacent']}), "
          f"different conversation all pairs {f(a['other_conv_all_pairs'])} (n {a['n_other_all']})")
    b = v["V2"]
    print(f"V2 tier plain {b['plain_hit_pct']:.2f}% / {b['plain_cpu_ms_w']:.2f} ms/w, conv-seeded (w 1) {b['conv_hit_pct']:.2f}% / "
          f"{b['conv_cpu_ms_w']:.2f} ms/w: lift {b['hit_lift_points']:+.2f} hit points, {b['cpu_ms_lift']:+.2f} ms/w")
    print("V3 first-m-windows Jaccard vs previous turn's set, m: " + " ".join(f"{m:>5d}" for m in range(1, 17)))
    for k in ("switch", "resume"):
        print(f"   after {k:7s} (n {v['V3_n'][k]:2d})          " + " ".join(f"{f(v['V3'][k][str(m)]):>5s}" for m in range(1, 17)))
    print("V4 lead time (top-4 non-resident per layer by the conversation's decayed counts):")
    for H, a in v["V4"].items():
        print(f"   H {H}: covers {a['miss_share_covered'] * 100:5.1f}% of the CPU misses at t, precision {a['precision_missed'] * 100:5.1f}% "
              f"(predicted expert missed at t), {a['precision_routed'] * 100:5.1f}% (routed at t); misses {a['misses']}, predicted {a['predicted']}")


# ------------------------------------------------------------------------------------------------ main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__[__doc__.index("Policies"):])
    ap.add_argument("profile")
    ap.add_argument("slots", type=int)
    ap.add_argument("trace")
    ap.add_argument("--cuts", default="")
    ap.add_argument("--names", default="")
    ap.add_argument("--convs", default="", help="conversation label per cut part, e.g. a,b,a,c (default: the part names)")
    ap.add_argument("--turns", default="", help="conv_trace.py requests.json: the turn structure of TRACE")
    ap.add_argument("--interleave", type=int, default=0, help="add the SYNTHETIC round-robin trace with N-window turns")
    ap.add_argument("--protect", default="0", help="protected core fractions, comma separated")
    ap.add_argument("--policies", nargs="*", default=None)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--cap", type=int, default=-1, help="swaps per window for every policy (default: tier 32, others none)")
    ap.add_argument("--parts", action="store_true", help="also run every cut part as a trace of its own")
    ap.add_argument("--no-holdout", action="store_true")
    ap.add_argument("--seed-file", default="")
    ap.add_argument("--validate-conv", action="store_true")
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)

    init = rp.load_profile(a.profile, a.slots)
    ranked = rp.load_profile(a.profile, 10 ** 9)
    nl, ne = rp.NL, rp.NE
    budgets = np.bincount(init[:, 0], minlength=nl)
    specs = a.policies if a.policies is not None else DEFAULT_POLICIES
    ctx = {"ranked": ranked, "seed": None}
    if a.seed_file:
        ctx["seed"] = np.load(a.seed_file)["counts"].astype(float)
    elif any(s.startswith("seedfile") for s in specs):
        print("note: seedfile policy skipped (law 5 needs --seed-file FILE.npz with prompt / cross-layer counts)")

    whole = cl.Layered.from_windows(rp.load_trace(a.trace)[0])
    D = dense(whole, ne)
    scen = []     # (name, D, turns)
    cuts = [int(x) for x in a.cuts.split(",")] if a.cuts else []
    names = a.names.split(",") if a.names else [f"part{i}" for i in range(len(cuts) + 1)]
    convs = a.convs.split(",") if a.convs else names
    if a.turns:
        turns = turns_from_requests(a.turns, a.trace, whole.n)
    elif cuts:
        turns = turns_from_cuts(cuts, whole.n, convs)
    else:
        turns = None
    parts = []
    if cuts:
        edges = [0] + cuts + [whole.n]
        parts = [whole.slice(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        if a.parts:
            for nm, p in zip(names, parts):
                scen.append((nm, dense(p, ne), None))
    scen.append((f"whole ({whole.n} windows)", D, turns))
    syn = None
    if a.interleave > 0 and len(parts) > 1:
        tr, tt = interleave_turns(parts, names, a.interleave)
        syn = (f"SYNTHETIC interleave {a.interleave}-window turns ({tr.n} windows)", dense(tr, ne), tt)
        scen.append(syn)

    report = {"slots": a.slots, "alpha": a.alpha, "theta": THETA, "protect": {}}
    for f in [float(x) for x in a.protect.split(",")]:
        n_prot = protect_slots(f, a.slots)
        rep = {}
        print(f"\n######## protect {f:g}: {n_prot} of {a.slots} slots are a static core ########")
        for name, d, tt in scen:
            res = {}
            for spec in specs:
                if spec == "layerdecay":
                    continue
                p = parse_policy(spec, ctx)
                if p is None or getattr(p, "needs_fit", False):   # fitted policies are reported hold-out only
                    continue
                if a.cap >= 0:
                    p.cap = a.cap
                res[p.name] = run(d, init, p, n_prot, tt)[0]
            rep[name] = res
            print(table(f"{name}, protect {f:g}", res, a.alpha))
            print(thrash_table(f"{name}, protect {f:g}", res))
        if not a.no_holdout:
            for hname, hd, htt in [(f"whole ({whole.n} windows)", D, turns)] + ([(syn[0], syn[1], syn[2])] if syn else []):
                ho, hist = run_holdout(hd, init, specs, ctx, n_prot, htt)
                rep[f"holdout {hname}"] = ho
                rep[f"layerdecay_choice {hname}"] = hist
                print(table(f"holdout: second half of {hname}, decay / phases fitted on the first half, protect {f:g}", ho, a.alpha))
                print(thrash_table(f"holdout {hname}, protect {f:g}", ho))
                print("   layerdecay picks (layers per decay): " + str(hist))
        report["protect"][f"{f:g}"] = rep

    if a.validate_conv:
        vs = []
        if turns is not None and len({c for _, _, c in turns}) < len(turns):
            vs.append(validate_conv(D, init, budgets, turns, a.alpha, "real turn structure"))
        elif turns is not None:
            print("\nnote: the real turn structure has no conversation with two turns (V1 same-conversation, V3 resume empty)")
            vs.append(validate_conv(D, init, budgets, turns, a.alpha, "real turn structure (one turn per conversation)"))
        if syn is not None:
            vs.append(validate_conv(syn[1], init, budgets, syn[2], a.alpha, syn[0]))
        for v in vs:
            print_validation(v)
        report["validation"] = vs
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
