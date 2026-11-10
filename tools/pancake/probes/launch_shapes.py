#!/usr/bin/env python3
"""launch_shapes.py - probe 4.2.1, second half (bench/TEST_PLAN.md): the block counts our decode graph launches.

QUESTION   Which grid shapes (blocks x threads) does one decoded token launch, how often, and how many of them sit on a
           block count that empty_grid.hip.cpp found slow on this card (F20: unimatrix099 d0c8d6c3e2)?
BUILD      nothing (Python 3 only).  Input is a rocprofv3 kernel trace of a decode run:
             python tools/pancake/gpu_profile.py configs/pancake/flashnext-iq2_xs-fast-hip.json /tmp/prof1 --case short
           writes /tmp/prof1/run_kernel_trace.csv (the profiler runs decode ~1.6x slower; shares and counts are what
           counts here).  Or by hand: /opt/rocm/core-10.0/bin/rocprofv3 --kernel-trace --output-format csv -d DIR -o run --
           <engine command>.  Grid_Size is in work-items, blocks = Grid_Size / Workgroup_Size.
RUN        python tools/pancake/probes/launch_shapes.py /tmp/prof1/run_kernel_trace.csv [--tokens N] [--sweep sweep.csv]
                                                        [--top 30] [--whole]
           python tools/pancake/probes/launch_shapes.py --static      (no trace: literal <<<blocks, threads>>> sites
                                                                       in src/, a rough list, no frequencies)
           --tokens N: decoded tokens in the trace (bench.jsonl "new"; default: read OUTDIR/bench.jsonl next to the trace)
           --sweep: the CSV of empty_grid.hip.cpp (--csv); adds the "excess" columns.
READ       1. kernels per token and per window; 2. the 30 most frequent (kernel, blocks x threads) shapes with mean us;
           3. block counts by launches per token, flagged "x42" when blocks % 42 == 0 (whole-card multiples, the XTX
           stall was 48 x 8) and with the empty-launch cost from the sweep; 4. with --sweep: the total EXCESS
           = sum over launches of max(0, empty cost(blocks) - floor), us per token and % of the token's wall time.
           F20 is worth a port only if that excess is a few % of a token (an empty launch is a lower bound: the real
           stall hides behind the previous kernel's tail in a graph); about 0 means close F20.
"""
import argparse
import collections
import csv
import json
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trace_common as tc  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


def load_sweep(path):
    cost = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            cost[int(r["blocks"])] = float(r["empty_us"])
    if not cost:
        sys.exit(f"{path}: empty sweep")
    floor = statistics.median(cost.values())
    return cost, floor


def cost_of(cost, b):
    """Empty-launch cost for b blocks: exact, else the nearest swept count (the sweep may use --step); None above range."""
    if b in cost:
        return cost[b]
    if b > max(cost) or b < 1:
        return None
    return cost[min(cost, key=lambda k: abs(k - b))]


def static_sites():
    pat = re.compile(r"(\w+)\s*(?:<[^<>;]*>)?\s*<<<\s*(\d+)\s*,\s*(\d+)")
    seen = collections.Counter()
    for d in ("src/kernels/cuda", "src/core"):
        for p in sorted((ROOT / d).rglob("*")):
            if p.suffix not in (".cu", ".cpp", ".cuh", ".inc", ".h"):
                continue
            try:
                text = p.read_text(errors="replace")
            except OSError:
                continue
            for m in pat.finditer(text):
                seen[(m.group(1), int(m.group(2)), int(m.group(3)), str(p.relative_to(ROOT)))] += 1
    print("literal launch sites (kernel, blocks, threads, file): grids built from expressions are NOT listed;")
    print("use a trace for the real shapes\n")
    byb = collections.Counter()
    for (k, b, t, f), n in sorted(seen.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print(f"  {b:6d} x {t:<4d} {k}  ({f})" + ("  x42" if b % 42 == 0 else ""))
        byb[b] += n
    print("\nliteral block counts:", ", ".join(f"{b}x{n}" for b, n in sorted(byb.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", nargs="?")
    ap.add_argument("--tokens", type=int)
    ap.add_argument("--sweep")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--whole", action="store_true")
    ap.add_argument("--static", action="store_true")
    a = ap.parse_args()
    if a.static:
        static_sites()
        return 0
    if not a.trace:
        ap.error("trace CSV required (or --static)")
    rows = tc.decode_window(tc.load(a.trace), a.whole)
    if not rows:
        sys.exit("no kernels in the decode window")
    if not any(r["blocks"] for r in rows):
        sys.exit("trace has no Grid_Size / Workgroup_Size columns; rerun rocprofv3 with --kernel-trace")
    tokens = a.tokens
    if tokens is None:
        try:
            tokens = json.loads(open(Path(a.trace).parent / "bench.jsonl").readline())["new"]
        except Exception:
            tokens = 0
    wall_ns = rows[-1]["end"] - rows[0]["start"]
    per = (lambda n: n / tokens) if tokens else (lambda n: n)
    unit = "per token" if tokens else "per window (no --tokens)"
    print(f"{len(rows)} kernels in the decode window, {wall_ns / 1e6:.0f} ms wall"
          + (f", {tokens} tokens: {len(rows) / tokens:.0f} kernels/token, {wall_ns / 1e3 / tokens:.0f} us/token" if tokens else ""))

    shape = collections.defaultdict(list)
    for r in rows:
        shape[(tc.short(r["name"])[:70], r["blocks"], r["wg"])].append(r["dur"])
    print(f"\nmost frequent shapes ({unit}):")
    print("  launches  mean us  blocks x threads  kernel")
    for (k, b, t), d in sorted(shape.items(), key=lambda kv: -len(kv[1]))[: a.top]:
        print(f"  {per(len(d)):8.2f}  {statistics.mean(d) / 1e3:7.1f}  {b:6d} x {t:<5d} {k}" + ("  x42" if b and b % 42 == 0 else ""))

    cost, floor = (load_sweep(a.sweep) if a.sweep else (None, None))
    byb = collections.Counter()
    bydur = collections.defaultdict(list)
    for r in rows:
        byb[r["blocks"]] += 1
        bydur[r["blocks"]].append(r["dur"])
    wrong_wg = sum(1 for r in rows if r["wg"] != 256)
    print(f"\nblock counts ({unit}); {100 * (1 - wrong_wg / len(rows)):.0f}% of launches are 256-thread blocks"
          + (f"; sweep floor {floor:.2f} us" if cost else ""))
    print("  blocks  launches  mean us  flags" + ("  empty us  excess us" if cost else ""))
    excess_total = 0.0
    for b, n in sorted(byb.items(), key=lambda kv: -kv[1])[: a.top * 2]:
        flag = "x42" if b and b % 42 == 0 else ""
        line = f"  {b:6d}  {per(n):8.2f}  {statistics.mean(bydur[b]) / 1e3:7.1f}  {flag:5s}"
        if cost:
            c = cost_of(cost, b)
            line += f"  {c:8.2f}  {max(0.0, c - floor):9.2f}" if c is not None else "       n/a        n/a"
        print(line)
    if cost:
        outside = 0
        for b, n in byb.items():
            c = cost_of(cost, b)
            if c is None:
                outside += n
                continue
            excess_total += n * max(0.0, c - floor)
        tot_us = excess_total
        what = f"{tot_us / tokens:.1f} us per token" if tokens else f"{tot_us:.1f} us in the window"
        print(f"\nEXCESS over the {floor:.2f} us floor, sum over all launches: {what}"
              f" = {100 * tot_us * 1e3 / wall_ns:.2f}% of the decode wall time")
        if outside:
            print(f"  ({outside} launches have more blocks than the sweep covers; rerun empty_grid with a larger --max-blocks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
