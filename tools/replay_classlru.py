#!/usr/bin/env python3
"""Offline replay of upstream llama.cpp's ClassLRU expert cache (ggml-org/llama.cpp c4fb57969, src/llama-moe-cache.cpp)
on a --dump-routing trace, next to this engine's own adaptive tier (tools/replay_cache_policy.py).

    python tools/replay_classlru.py TRACE.bin PROFILE.bin SLOTS [--half-life 128] [--classes 8] [--budget N]

Upstream's policy, as read from the commit (not our own invention):
  * CRF per (layer, expert) = uses decayed with a half-life of H tokens of that layer's clock; kept across evictions.
  * class = min(n_class - 1, floor(log2(CRF))) as of the expert's last use; one LRU list per class over all slots.
  * on a miss the expert is admitted; the victim is the least recently used slot of each class list that is not in the
    current layer's plan (and not filled in this plan), taking the one with the largest age / 2^class.
  * every miss is admitted, so the swap traffic is the number of distinct misses (no per-round budget).

Hit accounting is this engine's (replay_cache_policy.simulate): an entry is a hit when its expert is resident BEFORE the
layer's step; admissions made during the step count as swaps, not hits.  --budget N caps the admissions per window
(misses admitted by descending CRF), the one change made to reach this engine's swap traffic; --classes 1 is a plain LRU.
The slot pool is shared by all layers (upstream groups layers; here one group).
"""
import math
import os
import sys
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_cache_policy as rcp  # noqa: E402


class ClassLRU:
    def __init__(self, nl, ne, init_keys, half_life=128.0, n_class=8):
        self.nl, self.ne, self.hl, self.nc = nl, ne, float(half_life), n_class
        self.crf = np.zeros(nl * ne)
        self.last = np.zeros(nl * ne)           # layer clock at the last use; resident-but-unused slots count from 0
        self.clock = np.zeros(nl)               # tokens this layer has seen
        self.lists = [OrderedDict() for _ in range(n_class)]   # head = least recently used
        self.cls = {}                           # resident key -> its list
        for k in init_keys:
            self.lists[0][int(k)] = None
            self.cls[int(k)] = 0

    def key(self, layer, expert):
        return int(layer) * self.ne + int(expert)

    def decayed(self, key):
        age = self.clock[key // self.ne] - self.last[key]
        return self.crf[key] * 2.0 ** (-age / self.hl)

    def use(self, key):
        self.crf[key] = self.decayed(key) + 1.0
        self.last[key] = self.clock[key // self.ne]

    def class_of(self, key):
        return min(self.nc - 1, int(math.log2(max(self.crf[key], 1.0))))

    def move(self, key, c):
        old = self.cls.get(key)
        if old is not None:
            del self.lists[old][key]
        self.lists[c][key] = None
        self.cls[key] = c

    def victim(self, protected):
        """The slot to give up: per class the oldest slot not protected, the largest age / 2^class wins."""
        best, best_score = None, -1.0
        for c in range(self.nc):
            for k in self.lists[c]:
                if k in protected:
                    continue
                score = (self.clock[k // self.ne] - self.last[k]) / (2.0 ** c)
                if score > best_score:
                    best, best_score = k, score
                break
        return best

    def step(self, layer, ids, budget):
        """One layer's batch ids[T, k]; returns (hits, entries, admissions); budget None = admit every miss."""
        keys = [layer * self.ne + int(e) for e in ids.ravel()]
        hits = sum(1 for k in keys if k in self.cls)
        for k in keys:
            self.use(k)
        plan = set(keys)
        for k in plan:
            if k in self.cls:
                self.move(k, self.class_of(k))
        misses = sorted((k for k in plan if k not in self.cls), key=lambda k: -self.crf[k])
        protected = set(plan)
        swaps = 0
        for k in misses:
            if budget is not None and swaps >= budget:
                break
            v = self.victim(protected)
            if v is None:
                break
            del self.lists[self.cls.pop(v)][v]
            self.move(k, self.class_of(k))
            protected.add(k)
            swaps += 1
        self.clock[layer] += ids.shape[0]
        return hits, len(keys), swaps


def simulate_classlru(windows, init_pairs, half_life=128.0, n_class=8, budget=None, nl=None, ne=None):
    nl = nl or rcp.NL
    ne = ne or rcp.NE
    init_keys = [int(l) * ne + int(e) for l, e in init_pairs]
    cache = ClassLRU(nl, ne, init_keys, half_life, n_class)
    hits = ent = swaps = 0
    for w in windows:
        left = budget
        for l, ids in w:
            h, n, s = cache.step(l, ids, left)
            hits += h
            ent += n
            swaps += s
            if left is not None:
                left -= s
    return dict(share=hits / max(1, ent), swaps=swaps, windows=len(windows), entries=ent, slots=len(cache.cls))


if __name__ == "__main__":
    tr, prof, slots = sys.argv[1], sys.argv[2], int(sys.argv[3])

    def opt(name, default, cast=float):
        return cast(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default

    windows, k = rcp.load_trace(tr)
    init = rcp.load_profile(prof, slots)
    hl, nc, budget = opt("--half-life", 128.0), opt("--classes", 8, int), opt("--budget", 0, int)
    r = simulate_classlru(windows, init, hl, nc, budget or None)
    print("windows %d, k %d | ClassLRU half-life %g classes %d budget %s: share %.4f, swaps %d (%.2f/window), slots %d" %
          (len(windows), k, hl, nc, budget or "unlimited", r["share"], r["swaps"], r["swaps"] / r["windows"], r["slots"]))
