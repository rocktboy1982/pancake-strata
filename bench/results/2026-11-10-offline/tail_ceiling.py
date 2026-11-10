#!/usr/bin/env python3
"""4.0.3 (fallback): ceiling of any draft-stop rule (SPEC_CUM / tail-row drop) from a --dump-routing trace.

tools/pancake/trace/tail_replay.py needs a window trace (.stwt) with drafts records; none was ever recorded
(STRATA_WINDOW_TRACE was never set on a run).  What exists is a routing trace (no acceptance, no confidences).
Bound: an ORACLE that drops exactly the rows the verify window did not accept saves the CPU-missed experts
only those rows needed (tools/pancake/tail_rows.py logic).  The number of rejected rows per window is taken from
the request's measured tokens per window (accepted + 1), so rejected = T - tokens/window.  No real predictor
beats the oracle, and a real one also drops some accepted rows, so this is an upper bound.

    python tail_ceiling.py ROUTING.bin PROFILE.bin SLOTS
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, '/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-a2cec809b8de0e01d/tools')
import replay_cache_policy as rp

trace, profile, slots = sys.argv[1], sys.argv[2], int(sys.argv[3])
windows, _k = rp.load_trace(trace)
init = rp.load_profile(profile, slots)
pol = rp.Policy()
res = np.zeros((rp.NL, rp.NE), bool)
res[init[:, 0], init[:, 1]] = True
usage = np.zeros((rp.NL, rp.NE))
# per window: T, distinct misses, only_last[d] for d = 0..T-1 (cumulative: misses only rows >= T-d need)
per = []
for wi, w in enumerate(windows):
    T = w[0][1].shape[0]
    tot = 0
    only = np.zeros(T)
    for l, ids in w:
        miss = ~res[l, ids]
        for e in np.unique(ids[miss]):
            rows = np.flatnonzero((ids == e).any(axis=1))
            first = rows.min()
            tot += 1
            for d in range(1, T):
                if first >= T - d:
                    only[d] += 1
        np.add.at(usage[l], ids.ravel(), 1.0)
    if (wi + 1) % pol.every == 0:
        rp.adapt_round(res, usage, pol)
        usage *= pol.decay
    per.append((T, tot, only))

# requests of the trace-bench run (cuts from replay_seeding.py): read4k, short chat, code edit, long
segs = [('read4k', 0, 11, None), ('short chat', 11, 96, (24.8, 5.2)), ('code edit', 96, 157, (37.4, 17.4)),
        ('long', 157, 219, (24.8, 5.2))]
tokens = {'read4k': 53, 'short chat': 321, 'code edit': 553, 'long': 169}   # generated tokens per request (r1.txt)
print('%-11s %5s %5s %7s %9s %9s %9s %11s' % ('request', 'win', 'tpw', 'miss/w', 'rej rows', 'saved%miss', 'cpu share', 'oracle %win'))
for name, a, b, tm in segs:
    ws = per[a:b]
    n = len(ws)
    tpw = tokens[name] / n
    miss = sum(x[1] for x in ws)
    saved = 0.0
    rej_rows = 0.0
    for T, tot, only in ws:
        rej = max(0.0, T - tpw)            # expected rows not accepted (>= 0)
        rej_rows += rej
        lo = int(np.floor(rej)); hi = min(T - 1, lo + 1); f = rej - lo
        s_lo = only[lo] if lo >= 1 else 0.0
        s_hi = only[hi] if hi >= 1 else 0.0
        saved += (1 - f) * s_lo + f * s_hi
    frac = saved / miss if miss else 0.0
    share = tm[1] / tm[0] if tm else float('nan')
    print('%-11s %5d %5.1f %7.1f %9.2f %8.1f%% %8.1f%% %10.1f%%' % (
        name, n, tpw, miss / n, rej_rows / n, 100 * frac, 100 * share, 100 * frac * share))
