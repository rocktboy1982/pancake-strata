"""tail_replay.py - MOD_PLAN item 5 (tail-row drop), offline on a window trace with draft confidences.

Rule replayed: before the CPU experts of a window run, drop the draft rows from the first position where the running
product of the drafter's confidences falls below THETA (exact: a dropped row and the rows after it are only not
accepted; earlier rows never read them).  Per window, from the trace (include/strata/core/window_trace.hpp):
  saved   CPU-missed experts (distinct per layer, kind -1) that only the dropped rows needed
  lost    accepted drafts beyond the kept rows (the window had accepted them; with the drop they are not)
Priced with the time model: --us-per-miss (CPU time per distinct missed expert; time_ceiling.py --overlap calibrates
it: ~23 us short chat, ~60 us code edit) and the window's own measured ms; net = tokens per second after / before.
Needs windows with drafts records (written once the draft area calls window_trace().drafts()).

    python tools/pancake/trace/tail_replay.py TRACE.stwt [--thetas 0.05,0.1,0.2,0.3] [--us-per-miss 40]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wtrace  # noqa: E402


def keep_rows(drafts, theta):
    """Rows kept (row 0 = the committed token, always kept; row j = draft j)."""
    p = 1.0
    for j, c in enumerate(drafts):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--thetas", default="0.05,0.1,0.2,0.3,0.5")
    ap.add_argument("--us-per-miss", type=float, default=40.0)
    a = ap.parse_args()
    tr = wtrace.read(a.trace)
    wins = [w for w in tr.windows if w.drafts is not None and w.accepted is not None and w.rows and w.ms]
    if not wins:
        print("no windows with draft confidences in this trace (the draft area must call window_trace().drafts())")
        return 2
    base_tok = sum(w.accepted + 1 for w in wins)
    base_ms = sum(w.ms for w in wins)
    print(f"{len(wins)} windows with drafts; base {base_tok / base_ms * 1e3:.1f} tok/s (trace wall time)")
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
