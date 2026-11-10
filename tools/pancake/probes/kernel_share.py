#!/usr/bin/env python3
"""kernel_share.py - probe 4.2.2 (bench/TEST_PLAN.md): per-kernel time share of route_publish_res and the top-10, from a
rocprofv3 trace of a decode run.  Decides F9 (the router's publish: worth porting if >= 2% of GPU time stays).

QUESTION   How much of the decode GPU time does route_publish_res take (mean us per call, share of kernel time and of
           the decode wall time), what else is in the top 10, and how long is the idle gap after it (the PCIe round trip
           to the CPU experts is not kernel time)?  The deep-research profile measured ~17.8 us per call; unimatrix's
           top-10 alone is ~3.5 us of work, so most of the rest would be the copy and doorbell.
BUILD      nothing (Python 3 only); needs a trace from the 7900 XT with the GPU free.
RUN        1) the trace, either with the repo's driver (one request through bench_engines.py under rocprofv3):
                python tools/pancake/gpu_profile.py configs/pancake/flashnext-iq2_xs-fast-hip.json /tmp/prof2 --case short
              (writes /tmp/prof2/run_kernel_trace.csv, bench.jsonl, summary.txt; the preset has STRATA_ROUTE_PUBLISH=1,
               so route_publish_res appears; add --env STRATA_ROUTE_PUBLISH=0 for the comparison arm), or by hand:
                /opt/rocm/core-10.0/bin/rocprofv3 --kernel-trace --stats --output-format csv -d /tmp/prof2 -o run -- ENGINE ARGS
              (HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/rocm/core-10.0/lib; the profiler slows decode ~1.6x: read shares)
           2) python tools/pancake/probes/kernel_share.py /tmp/prof2/run_kernel_trace.csv [--tokens N] [--match REGEX]
                                                         [--top 10] [--whole]
              --match: kernel-name regex of the kernel under test (default "route_publish"; with STRATA_ROUTER_FUSE=1 the
              publish moves into the router GEMV, so match "bf16" kernels instead and compare their time with the plain run)
READ       "TARGET" line: calls per token, mean/median/p90 us, share of kernel time and of decode wall time.
           "gap after" = mean idle between the target's end and the next kernel's start (a stall for the host/PCIe).
           Then the top-10 by total time with the same columns.  F9 verdict: if the target is < 2% of the wall time,
           drop F9; if >= 2% and mean us per call is well above ~5-8 us (the estimate after a port), it is worth a try.
"""
import argparse
import collections
import json
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trace_common as tc  # noqa: E402


def pct(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--tokens", type=int, help="decoded tokens in the trace (default: OUTDIR/bench.jsonl 'new')")
    ap.add_argument("--match", default="route_publish")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--whole", action="store_true", help="whole trace, not just the decode phase")
    a = ap.parse_args()
    rows = tc.decode_window(tc.load(a.trace), a.whole)
    if not rows:
        sys.exit("no kernels in the decode window")
    tokens = a.tokens
    if tokens is None:
        try:
            tokens = json.loads(open(Path(a.trace).parent / "bench.jsonl").readline())["new"]
        except Exception:
            tokens = 0
    wall = rows[-1]["end"] - rows[0]["start"]
    busy = sum(r["dur"] for r in rows)
    print(f"decode window: {wall / 1e6:.0f} ms wall, kernels {busy / 1e6:.0f} ms ({busy / wall:.0%} busy), {len(rows)} kernels"
          + (f", {tokens} tokens" if tokens else ""))

    # time by kernel (template arguments dropped for the ranking, so one kernel's instantiations add up)
    base = lambda n: re.sub(r"<.*>", "", tc.short(n))
    tot = collections.Counter()
    calls = collections.Counter()
    durs = collections.defaultdict(list)
    for r in rows:
        k = base(r["name"])
        tot[k] += r["dur"]
        calls[k] += 1
        durs[k].append(r["dur"])

    rx = re.compile(a.match)
    tgt = [r for r in rows if rx.search(r["name"])]
    if tgt:
        d = [r["dur"] for r in tgt]
        gaps = []
        for i, r in enumerate(rows):
            if rx.search(r["name"]) and i + 1 < len(rows):
                gaps.append(max(0, rows[i + 1]["start"] - r["end"]))
        tsum = sum(d)
        print(f"\nTARGET /{a.match}/: {len(tgt)} calls" + (f" ({len(tgt) / tokens:.1f} per token)" if tokens else ""))
        print(f"  us per call: mean {statistics.mean(d) / 1e3:.1f}  median {statistics.median(d) / 1e3:.1f}  p90 {pct(d, 0.9) / 1e3:.1f}"
              f"  max {max(d) / 1e3:.1f}")
        print(f"  share: {100 * tsum / busy:.2f}% of kernel time, {100 * tsum / wall:.2f}% of the decode wall time"
              + (f", {tsum / 1e3 / tokens:.1f} us per token" if tokens else ""))
        if gaps:
            print(f"  gap after it: mean {statistics.mean(gaps) / 1e3:.1f} us, p90 {pct(gaps, 0.9) / 1e3:.1f} us"
                  f"  ({100 * sum(gaps) / wall:.2f}% of the wall time)")
        launch = {(r["blocks"], r["wg"]) for r in tgt}
        print(f"  launch shape(s) blocks x threads: {sorted(launch)}")
    else:
        print(f"\nTARGET /{a.match}/: no kernel matches (is STRATA_ROUTE_PUBLISH=1 and the HIP build in use?). Names seen: "
              + ", ".join(sorted({base(r['name']) for r in rows})[:12]) + " ...")

    print(f"\ntop {a.top} kernels by total time:")
    print("  share(kernel)  share(wall)  total ms     calls  mean us  p90 us  kernel")
    for k, v in tot.most_common(a.top):
        mark = "  <== target" if rx.search(k) else ""
        print(f"  {100 * v / busy:12.1f}%  {100 * v / wall:10.1f}%  {v / 1e6:8.1f}  {calls[k]:8d}  {v / calls[k] / 1e3:7.1f}"
              f"  {pct(durs[k], 0.9) / 1e3:6.1f}  {k[:70]}{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
