#!/usr/bin/env python3
"""timeline_gaps.py - pancake-strata LEADS.md Part T T12: gaps between kernels and the host wait share of a rocprofv3 timeline (T-020 / T-021 follow-up).

T-021 measured wait_flag_ge at 31% of kernel time and T-020 525 ms of 3,638 ms wall in gaps over 1 ms; a kernel list cannot say WHERE the stream waits.
This reads a `--kernel-trace --hip-trace --hip-event-trace --hip-graph-trace` (or `--sys-trace`) capture made by tools/pancake/box_chains/trace-t12.v1.sh
(kernel_trace.csv + hip_api_trace.csv [+ memory_copy_trace.csv], or the rocprofiler-sdk json; same loader as prof_report.py) and reports:
  1. wall (first kernel start to last kernel end), device busy share (union of all kernel intervals), idle share;
  2. gaps between kernels (union intervals) in buckets: <10 us, 10-50 us (launch gaps), 50 us-1 ms, >1 ms (host stalls): count, total ms, share of wall;
  3. host wait: time inside HIP calls that block (default regex Synchronize|WaitEvent|EventQuery|StreamQuery; --wait-regex), per function, share of wall,
     and how much of the gap time lies under such a call (gap_covered) versus not (gap_other = CPU work / launch cost between calls);
  4. the share of kernel busy time taken by kernels matching --wait-kernel (default wait_flag: the T-021 number) and the n largest gaps with the kernel pair around them;
  5. counts of event record / wait calls (hipEventRecord, hipStreamWaitEvent) when --hip-event-trace was on.
    python3 tools/pancake/timeline_gaps.py DIR_OR_FILE [more] [--window-from REGEX --window-to REGEX] [--top 10] [--json OUT.json]
--window-from / --window-to cut the wall to the span from the first kernel matching the first regex to the end of the last kernel matching the second
(decode phase only, as gpu_profile.py does).  Profiling slows the run (~1.6x): read shares and counts, not absolute speed.
Exit 0 ok, 2 no kernels.  stdlib only, read-only except --json.  What is NOT validated on a PC: the real file layouts of ROCm 10.0 (the loader is checked on
fixtures of the documented layout).
Tests: python -m pytest -q tools/pancake/test_timeline_gaps.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prof_report as P  # noqa: E402

BUCKETS = [("<10us", 0, 10e3), ("10-50us", 10e3, 50e3), ("50us-1ms", 50e3, 1e6), (">1ms", 1e6, float("inf"))]
WAIT_RX = r"Synchronize|WaitEvent|EventQuery|StreamQuery"


def union(iv: list) -> list:
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap(a: list, b: list) -> float:
    """total overlap of two sorted disjoint interval lists"""
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            tot += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def analyse(d: dict, wait_rx=WAIT_RX, wait_kernel="wait_flag", w_from=None, w_to=None, top=10) -> dict:
    ks = sorted(d["kernels"], key=lambda k: k["start"])
    if w_from or w_to:
        lo = next((k["start"] for k in ks if not w_from or re.search(w_from, k["name"])), None)
        hi = next((k["end"] for k in reversed(ks) if not w_to or re.search(w_to, k["name"])), None)
        if lo is None or hi is None or hi <= lo:
            return {}
        ks = [k for k in ks if k["start"] >= lo and k["end"] <= hi]
    if not ks:
        return {}
    t0, t1 = min(k["start"] for k in ks), max(k["end"] for k in ks)
    wall = t1 - t0
    busy = union([[k["start"], k["end"]] for k in ks])
    busy_ns = sum(e - s for s, e in busy)
    gaps = []
    for (s0, e0), (s1, e1) in zip(busy, busy[1:]):
        gaps.append({"start": e0, "end": s1, "ns": s1 - e0})
    # kernel pair around each gap
    for g in gaps:
        g["prev"] = max((k for k in ks if k["end"] <= g["start"] + 1), key=lambda k: k["end"], default={}).get("name", "?")
        g["next"] = min((k for k in ks if k["start"] >= g["end"] - 1), key=lambda k: k["start"], default={}).get("name", "?")
    bucket = []
    for name, lo, hi in BUCKETS:
        sel = [g for g in gaps if lo <= g["ns"] < hi]
        bucket.append({"bucket": name, "count": len(sel), "ms": sum(g["ns"] for g in sel) / 1e6, "share": sum(g["ns"] for g in sel) / wall if wall else 0})
    api = [a for a in d["api"] if a["end"] > t0 and a["start"] < t1]
    wrx = re.compile(wait_rx)
    waits = [a for a in api if wrx.search(a["name"]) and not a.get("copy")]
    per_fn: dict = {}
    for a in waits:
        e = per_fn.setdefault(a["name"], {"name": a["name"], "calls": 0, "ns": 0.0})
        e["calls"] += 1
        e["ns"] += min(a["end"], t1) - max(a["start"], t0)
    # host wait per thread (union) then summed over threads is not wall time; use the busiest thread's union
    by_thread: dict = {}
    for a in waits:
        by_thread.setdefault(a["tid"], []).append([max(a["start"], t0), min(a["end"], t1)])
    thread_wait = {t: sum(e - s for s, e in union(iv)) for t, iv in by_thread.items()}
    wait_ns = max(thread_wait.values(), default=0.0)
    wait_union = union([[max(a["start"], t0), min(a["end"], t1)] for a in waits])
    gap_iv = [[g["start"], g["end"]] for g in gaps]
    gap_ns = sum(g["ns"] for g in gaps)
    covered = overlap(gap_iv, wait_union)
    wk = re.compile(wait_kernel) if wait_kernel else None
    k_ns = sum(k["end"] - k["start"] for k in ks)
    wk_ns = sum(k["end"] - k["start"] for k in ks if wk and wk.search(k["name"]))
    return {
        "wall_ms": wall / 1e6, "kernels": len(ks), "busy_share": busy_ns / wall if wall else 0, "idle_share": 1 - busy_ns / wall if wall else 0,
        "buckets": bucket, "gap_ms": gap_ns / 1e6,
        "host_wait_ms": wait_ns / 1e6, "host_wait_share": wait_ns / wall if wall else 0, "host_wait_busiest_thread": max(thread_wait, key=thread_wait.get, default=None),
        "wait_fns": sorted(per_fn.values(), key=lambda e: -e["ns"]),
        "gap_covered_by_wait_ms": covered / 1e6, "gap_other_ms": (gap_ns - covered) / 1e6,
        "wait_kernel_regex": wait_kernel, "wait_kernel_share_of_kernel_time": wk_ns / k_ns if k_ns else 0,
        "event_records": sum(1 for a in api if "EventRecord" in a["name"]), "wait_events": sum(1 for a in api if "StreamWaitEvent" in a["name"]),
        "top_gaps": sorted(gaps, key=lambda g: -g["ns"])[:top],
    }


def render(r: dict) -> str:
    o = [f"wall {r['wall_ms']:.2f} ms, {r['kernels']} kernels, device busy {r['busy_share'] * 100:.1f}%, idle {r['idle_share'] * 100:.1f}%"]
    o.append("gaps between kernels:")
    for b in r["buckets"]:
        o.append(f"  {b['bucket']:>9}: {b['count']:>7} gaps {b['ms']:>9.2f} ms {b['share'] * 100:>6.2f}% of wall")
    o.append(f"host wait (calls matching the wait regex, busiest thread): {r['host_wait_ms']:.2f} ms = {r['host_wait_share'] * 100:.1f}% of wall; "
             f"of the {r['gap_ms']:.2f} ms of gaps {r['gap_covered_by_wait_ms']:.2f} ms lie under a wait call, {r['gap_other_ms']:.2f} ms do not (CPU work / launch)")
    for e in r["wait_fns"][:8]:
        o.append(f"  {e['name'][:40]:40} {e['calls']:>7} calls {e['ns'] / 1e6:>10.2f} ms")
    o.append(f"kernels matching '{r['wait_kernel_regex']}': {r['wait_kernel_share_of_kernel_time'] * 100:.1f}% of kernel time (T-021: 31% for wait_flag_ge)")
    o.append(f"hipEventRecord calls {r['event_records']}, hipStreamWaitEvent calls {r['wait_events']}")
    o.append("largest gaps:")
    for g in r["top_gaps"]:
        o.append(f"  {g['ns'] / 1e3:>10.1f} us  after {g['prev'][:38]} -> before {g['next'][:38]}")
    return "\n".join(o)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--wait-regex", default=WAIT_RX)
    ap.add_argument("--wait-kernel", default="wait_flag")
    ap.add_argument("--window-from")
    ap.add_argument("--window-to")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    d = P.load(a.paths)
    r = analyse(d, a.wait_regex, a.wait_kernel, a.window_from, a.window_to, a.top)
    if not r:
        print("no kernels (need kernel_trace.csv or rocprofiler-sdk json; check the window regexes)", file=sys.stderr)
        return 2
    print(render(r))
    if a.json:
        Path(a.json).write_text(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
