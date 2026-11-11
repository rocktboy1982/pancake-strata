"""draft_costs.py - MOD_PLAN item 43: DraftPolicy's per-window-size costs from window traces (STRATA_SPEC_COST_TABLE).

DraftPolicy guesses the cost of a window size it has not measured from a fixed curve (kShape, measured on an RTX 5070).
Here, per context bucket (< 6K, < 24K, longer - DraftPolicy's), the cost of window size T is:
  measured  the mean wall ms of the trace's windows with T rows (enough of them), else
  modeled   the nearest measured size's ms plus, per extra row, the CPU time of the distinct NEW missed experts an extra
            row adds on average in this trace (kind -1 entries not already missed by the earlier rows), at --us-per-miss
Prints the table and the engine line:  STRATA_SPEC_COST=1 STRATA_SPEC_COST_TABLE="..."

    python tools/pancake/trace/draft_costs.py TRACE.stwt [...] [--us-per-miss 40] [--max-t 8] [--min-n 20]
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wtrace  # noqa: E402


def bucket(pos):
    return 0 if pos < 6000 else 1 if pos < 24000 else 2


def new_misses_per_row(windows):
    """Mean distinct CPU-missed experts row j adds beyond rows 0..j-1 (all layers), by row index."""
    add = defaultdict(list)
    for w in windows:
        per_row = defaultdict(int)
        if not w.layers and w.row_miss is not None:   # a trace without layer records: the engine's own row counts (record 5)
            for j, v in enumerate(w.row_miss):
                per_row[j] += int(v)
        for lay in w.layers.values():
            seen = set()
            for j in range(lay.ids.shape[0]):
                cpu = lay.ids[j][lay.kind[j] == -1].tolist()
                per_row[j] += len(set(cpu) - seen)
                seen.update(cpu)
        for j, v in per_row.items():
            add[j].append(v)
    return {j: float(np.mean(v)) for j, v in add.items()}


def build_tables(wins, us_per_miss=40.0, max_t=8, min_n=20, out=None):
    """-> (lists, adds): per context bucket a list of the window cost (ms) of sizes 1..max_t (None: no size measured enough
    in that bucket), and the mean new misses per extra row.  Prints the per-bucket lines when `out` is a stream."""
    adds = new_misses_per_row(wins)
    lists = []
    for b in range(3):
        bw = [w for w in wins if bucket(w.position) == b]
        by_t = defaultdict(list)
        for w in bw:
            by_t[w.rows].append(w.ms)
        meas = {t: float(np.median(v)) for t, v in by_t.items() if len(v) >= min_n and 1 <= t <= max_t}
        if not meas:
            if out is not None:
                print(f"bucket {b}: no window size with >= {min_n} windows", file=out)
            lists.append(None)
            continue
        row = []
        for t in range(1, max_t + 1):
            if t in meas:
                row.append(meas[t])
                continue
            u = min(meas, key=lambda x: abs(x - t))   # nearest measured size, then the extra rows' new misses
            lo, hi = (u, t) if u < t else (t, u)
            extra = sum(adds.get(j, 0.0) for j in range(lo, hi)) * us_per_miss / 1e3
            row.append(meas[u] + extra if t > u else max(0.1, meas[u] - extra))
        lists.append(row)
        if out is not None:
            print(f"bucket {b} ({len(bw)} windows): " +
                  "  ".join(f"T{t}={v:.1f}{'' if t in meas else '*'}" for t, v in enumerate(row, 1)), file=out)
    return lists, adds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--us-per-miss", type=float, default=40.0)
    ap.add_argument("--max-t", type=int, default=8)
    ap.add_argument("--min-n", type=int, default=20)
    a = ap.parse_args()
    wins = []
    for p in a.traces:
        wins += [w for w in wtrace.read(p).windows if w.rows and w.ms and w.position is not None and w.accepted is not None]
    if not wins:
        print("no decode windows with timing in these traces")
        return 2
    lists, adds = build_tables(wins, a.us_per_miss, a.max_t, a.min_n, out=sys.stdout)
    lists = [r for r in lists if r is not None]
    if not lists:
        return 2
    table = ";".join(",".join(f"{v:.2f}" for v in r) for r in lists)
    print("(* = modeled)  new misses per extra row:", " ".join(f"r{j}={v:.0f}" for j, v in sorted(adds.items())[:8]))
    print(f'STRATA_SPEC_COST=1 STRATA_SPEC_COST_TABLE="{table}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
