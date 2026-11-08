"""tail_rows.py - pancake-strata: the ceiling of skipping a verify window's tail rows on the CPU (exact: a dropped row
and the rows after it are simply not accepted; earlier rows never read them).

Replays a --dump-routing trace through the engine's adaptive cache (tools/replay_cache_policy.py) with SLOTS experts
resident, and per window counts the CPU-missed experts (distinct per layer) and how many of them only the last 1, 2 or
3 rows need.  Those are the CPU expert reads a tail-row drop would save in that window.

    python tools/pancake/tail_rows.py TRACE.bin PROFILE.bin SLOTS
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay_cache_policy as rp  # noqa: E402


def main():
    trace, profile, slots = sys.argv[1], sys.argv[2], int(sys.argv[3])
    windows, _k = rp.load_trace(trace)
    init = rp.load_profile(profile, slots)
    pol = rp.Policy()
    res = np.zeros((rp.NL, rp.NE), bool)
    res[init[:, 0], init[:, 1]] = True
    usage = np.zeros((rp.NL, rp.NE))
    by_t = {}
    for wi, w in enumerate(windows):
        T = w[0][1].shape[0]
        st = by_t.setdefault(T, np.zeros(5))   # windows, distinct misses, only-last-1, only-last-2, only-last-3
        st[0] += 1
        for l, ids in w:
            miss = ~res[l, ids]                    # [T, k]
            for e in np.unique(ids[miss]):
                rows = np.flatnonzero((ids == e).any(axis=1))
                first = rows.min()
                st[1] += 1
                for d in (1, 2, 3):
                    if T - d >= 1 and first >= T - d:   # needed only by the last d rows (row 0 is never dropped)
                        st[1 + d] += 1
            np.add.at(usage[l], ids.ravel(), 1.0)
        if (wi + 1) % pol.every == 0:
            rp.adapt_round(res, usage, pol)
            usage *= pol.decay
    tot = sum(by_t.values())
    print(f"slots {slots}: {int(tot[0])} windows, {tot[1] / tot[0]:.1f} distinct CPU experts per window")
    print("rows  windows  CPU experts/window  only last 1  only last 2  only last 3")
    for T in sorted(by_t):
        s = by_t[T]
        if s[0] < 20:
            continue
        print(f"{T:4d}  {int(s[0]):7d}  {s[1] / s[0]:18.1f}  {s[2] / s[1]:10.1%}  {s[3] / s[1]:10.1%}  {s[4] / s[1]:10.1%}")
    print(f" all  {int(tot[0]):7d}  {tot[1] / tot[0]:18.1f}  {tot[2] / tot[1]:10.1%}  {tot[3] / tot[1]:10.1%}  {tot[4] / tot[1]:10.1%}")


if __name__ == "__main__":
    main()
