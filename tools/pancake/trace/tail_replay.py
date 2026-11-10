"""tail_replay.py - MOD_PLAN item 5 (tail-row drop), F8 (STRATA_SPEC_CUM) and A22 (DISCO), offline on a window trace
with draft confidences.

Capture (opt-in, one short GPU run; zero cost when the variable is unset):
    STRATA_WINDOW_TRACE=/tmp/w.stwt  <the usual serve command>        (also _EVERY=N, _MAX_MB=M)
The serve loop writes, per decode window, a drafts record when the window is built (src/program/generate.cpp, before
the verify): for each draft row j = 1..T-1 the drafter's probability of the draft token (NaN for a lookup / chained row,
which have no confidence) and its source (0 MTP, 1 DFlash2, 2 lookup window, 3 chained lookup).  The window record
adds rows, accepted (row j is accepted iff j <= accepted) and wall ms; the layer records carry every entry's placement,
so the CPU-missed experts each row adds (kind -1, distinct per layer, not already needed by an earlier row) are read
back exactly (wtrace.new_cpu_miss).  A trace of a run with the drafter off has no drafts records.

Rule replayed: before the CPU experts of a window run, drop the draft rows from the first position where the running
product of the drafter's confidences falls below THETA (exact: a dropped row and the rows after it are only not
accepted; earlier rows never read them).  Per window:
  saved   CPU-missed experts (distinct per layer, kind -1) that only the dropped rows needed
  lost    accepted drafts beyond the kept rows (the window had accepted them; with the drop they are not)
Priced with the time model: --us-per-miss (CPU time per distinct missed expert; time_ceiling.py --overlap calibrates
it: ~23 us short chat, ~60 us code edit) and the window's own measured ms; net = tokens per second after / before.
Windows whose first draft has no confidence (lookup windows) are not replayed (counted).  A row without one inside a
drafter window (a chained lookup row) counts as confidence 1 (it never stops the chain).

Also prints the table that decides A22 (DISCO) and F8: draft rows bucketed by the cumulative confidence of the chain up
to the row, with the share accepted, the mean new CPU-missed experts the row added, and whether the row pays
(accept rate x ms per token > new misses x us-per-miss).

    python tools/pancake/trace/tail_replay.py TRACE.stwt [--thetas 0.05,0.1,0.2,0.3] [--us-per-miss 40]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wtrace  # noqa: E402


def conf(drafts):
    """Per-row confidence with NaN (no confidence) read as 1."""
    return np.where(np.isnan(drafts), 1.0, drafts)


def keep_rows(drafts, theta):
    """Rows kept (row 0 = the committed token, always kept; row j = draft j)."""
    p = 1.0
    for j, c in enumerate(conf(drafts)):
        p *= float(c)
        if p < theta:
            return j + 1
    return len(drafts) + 1


def window_effect(w, kept):
    saved = 0
    for lay in w.layers.values():
        cpu = lay.kind == -1
        if lay.ids.shape[0] <= kept:
            continue
        keep_set = set(lay.ids[:kept][cpu[:kept]].tolist())
        drop_set = set(lay.ids[kept:][cpu[kept:]].tolist())
        saved += len(drop_set - keep_set)
    lost = max(0, (w.accepted or 0) - (kept - 1))
    return saved, lost


def row_table(wins, us_per_miss, ms_per_tok):
    edges = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0001]
    rows = []   # (cumulative confidence, accepted 0/1, new misses)
    for w in wins:
        cum = np.cumprod(conf(w.drafts))
        nm = wtrace.new_cpu_miss(w)
        for j in range(1, min(len(w.drafts), w.rows - 1) + 1):
            rows.append((cum[j - 1], 1 if j <= w.accepted else 0, nm[j] if j < len(nm) else 0))
    a = np.array(rows, dtype=float)
    print(f"draft rows by cumulative chain confidence ({len(a)} rows; row pays if accept rate x {ms_per_tok:.1f} ms/token"
          f" > new misses x {us_per_miss:g} us):")
    print("  cum conf      rows  accepted  new CPU misses/row  gain ms  cost ms  pays")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if not m.any():
            continue
        acc, miss = a[m, 1].mean(), a[m, 2].mean()
        gain, cost = acc * ms_per_tok, miss * us_per_miss / 1e3
        print(f"  {lo:4.2f}-{min(hi, 1.0):4.2f} {int(m.sum()):7d}  {acc:7.1%}  {miss:18.2f}  {gain:7.2f}  {cost:7.2f}  {'yes' if gain > cost else 'no'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--thetas", default="0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--us-per-miss", type=float, default=40.0)
    a = ap.parse_args()
    tr = wtrace.read(a.trace)
    usable = [w for w in tr.windows if w.drafts is not None and w.accepted is not None and w.rows and w.ms]
    wins = [w for w in usable if len(w.drafts) and not np.isnan(w.drafts[0])]
    if not wins:
        print("no windows with draft confidences in this trace (capture one with STRATA_WINDOW_TRACE=<file> on a run with the drafter on)")
        return 2
    base_tok = sum(w.accepted + 1 for w in wins)
    base_ms = sum(w.ms for w in wins)
    print(f"{len(wins)} windows with drafts ({len(usable) - len(wins)} lookup windows skipped); base {base_tok / base_ms * 1e3:.1f} tok/s"
          f" (trace wall time), accepted {sum(w.accepted for w in wins)} of {sum(w.rows - 1 for w in wins)} offered rows")
    row_table(wins, a.us_per_miss, base_ms / base_tok)
    for th in [float(x) for x in a.thetas.split(",")]:
        tok = ms = saved_all = lost_all = 0
        for w in wins:
            kept = min(keep_rows(w.drafts, th), w.rows)
            saved, lost = window_effect(w, kept)
            saved_all += saved
            lost_all += lost
            tok += w.accepted + 1 - lost
            ms += max(0.1, w.ms - saved * a.us_per_miss / 1e3)
        print(f"  theta {th:4.2f}: saved {saved_all / len(wins):6.1f} misses/window, lost {lost_all / len(wins):.3f} tokens/window"
              f" -> {tok / ms * 1e3:6.1f} tok/s ({(tok / ms) / (base_tok / base_ms) * 100 - 100:+.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
