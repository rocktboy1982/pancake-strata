#!/usr/bin/env python3
"""hit_critical.py - MOD_PLAN A36 decision rule: are the VRAM-hit expert kernels on the decode critical path?

QUESTION   A36 (weight-major layout for the VRAM-hit experts / the Q2_0 down, MonoMoE 2026) only pays if the hit kernels'
           time is not hidden behind the wait for the CPU experts.  Per MoE layer the GPU publishes the routing
           (doorbell), runs the hit experts (gate/up, SwiGLU, down) while the CPU computes the misses, then spins in a
           wait kernel until the CPU rows land.  If that wait is longer than its floor, the GPU was early: faster hit
           kernels only lengthen the wait (gain 0).  Only in layers where the wait is at its floor (the CPU was done
           first) is the hit time exposed, and a faster hit kernel can save at most its own saving there.
BUILD      nothing (Python 3 only); needs a rocprofv3 kernel trace of a decode run (the 7900 XT, GPU free).
RUN        1) the trace (one request through bench_engines.py under rocprofv3; the profiler slows decode ~1.6x):
                python tools/pancake/gpu_profile.py configs/pancake/flashnext-iq2_xs-fast-hip.json /tmp/a36 --case short
                python tools/pancake/gpu_profile.py configs/pancake/flashnext-iq2_xs-fast-hip.json /tmp/a36e --case edit
           2) python tools/pancake/probes/hit_critical.py /tmp/a36/run_kernel_trace.csv [--speedup 0.25] [--floor-us AUTO]
                  [--hit REGEX] [--wait REGEX] [--route REGEX] [--whole] [--json OUT]
READ       Per layer segment (from a routing kernel to the next one): H = hit-kernel time, W = wait-kernel time.  A segment
           is "GPU-bound" when W <= floor + eps (floor: the wait kernel's 5th-percentile duration unless --floor-us).
           "exposed hit" = H summed over GPU-bound segments; "ceiling" = speedup x exposed hit, in ms and in % of the decode
           wall time (an upper bound: in a GPU-bound layer the CPU's own margin is unknown).  The gate/up vs down split
           answers the "Q2_0 down" half of A36.
VERDICT    A36 rule: write the weight-major hit kernels only if the ceiling at --speedup 0.25 is >= 2% of the decode wall
           on BOTH the short and the edit trace (below that the run-to-run spread, ~20%, hides an exact gain); otherwise
           keep A36 parked and re-run this after any change that moves CPU time (deferral, k-th-miss admission, swaps).
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trace_common as tc  # noqa: E402

HIT = r"expdec_gu|expdec_down|native_gu|native_down|swiglu_q8_1_entries|swiglu_entries"
DOWN = r"expdec_down|native_down"
WAIT = r"wait_flag_ge|combine_cpu"
ROUTE = r"route_multi|route_publish|doorbell_publish|router_top"


def pct(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))] if s else 0


def segments(rows, hit=HIT, down=DOWN, wait=WAIT, route=ROUTE):
    """Layer segments: a routing kernel opens one (closing the previous); hit kernels add to H (and to D when they are
    down kernels), wait kernels to W.  Kernels before the first routing kernel are ignored.  A segment with no wait
    kernel has W = 0 (nothing waited for: GPU-bound)."""
    rh, rd, rw, rr = re.compile(hit), re.compile(down), re.compile(wait), re.compile(route)
    segs, cur = [], None
    for r in rows:
        n = r["name"]
        if rr.search(n):
            if cur is not None and cur["route_done"] and (cur["h"] or cur["w"] or cur["nwait"]):
                segs.append(cur)
                cur = None
            if cur is None:
                cur = {"h": 0, "d": 0, "w": 0, "nwait": 0, "route_done": False}
            continue
        if cur is None:
            continue
        cur["route_done"] = True
        if rh.search(n):
            cur["h"] += r["dur"]
            if rd.search(n):
                cur["d"] += r["dur"]
        elif rw.search(n):
            cur["w"] += r["dur"]
            cur["nwait"] += 1
    if cur is not None and (cur["h"] or cur["w"] or cur["nwait"]):
        segs.append(cur)
    return segs


def analyze(rows, speedup=0.25, floor_ns=None, eps_ns=2000, **rx):
    segs = segments(rows, **rx)
    if not segs:
        return None
    waits = [s["w"] for s in segs if s["nwait"]]
    floor = floor_ns if floor_ns is not None else pct(waits, 0.05)
    bound = [s for s in segs if s["w"] <= floor + eps_ns]
    wall = rows[-1]["end"] - rows[0]["start"] if rows else 0
    h_all = sum(s["h"] for s in segs)
    d_all = sum(s["d"] for s in segs)
    h_exp = sum(s["h"] for s in bound)
    d_exp = sum(s["d"] for s in bound)
    ceil = speedup * h_exp
    return {
        "segments": len(segs), "gpu_bound": len(bound), "floor_us": floor / 1e3, "wall_ms": wall / 1e6,
        "hit_ms": h_all / 1e6, "down_ms": d_all / 1e6, "wait_ms": sum(s["w"] for s in segs) / 1e6,
        "exposed_hit_ms": h_exp / 1e6, "exposed_down_ms": d_exp / 1e6,
        "exposed_share_of_hit": h_exp / h_all if h_all else 0.0,
        "speedup": speedup, "ceiling_ms": ceil / 1e6, "ceiling_pct_wall": 100.0 * ceil / wall if wall else 0.0,
        "verdict": "build A36 (ceiling >= 2% of the decode wall)" if wall and ceil / wall >= 0.02
                   else "keep A36 parked (ceiling < 2% of the decode wall)",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--speedup", type=float, default=0.25, help="assumed hit-kernel time saved (fraction, default 0.25)")
    ap.add_argument("--floor-us", type=float, default=None, help="wait-kernel floor in us (default: its 5th percentile)")
    ap.add_argument("--eps-us", type=float, default=2.0, help="tolerance above the floor (default 2 us)")
    ap.add_argument("--hit", default=HIT, help=f"hit-kernel regex (default {HIT!r})")
    ap.add_argument("--down", default=DOWN, help=f"the down kernels among them (default {DOWN!r})")
    ap.add_argument("--wait", default=WAIT, help=f"CPU-expert wait regex (default {WAIT!r})")
    ap.add_argument("--route", default=ROUTE, help=f"layer-start regex (default {ROUTE!r})")
    ap.add_argument("--whole", action="store_true", help="whole trace, not just the decode phase")
    ap.add_argument("--json", help="also write the result as JSON here")
    a = ap.parse_args()
    rows = tc.decode_window(tc.load(a.trace), a.whole)
    res = analyze(rows, a.speedup, None if a.floor_us is None else a.floor_us * 1e3, a.eps_us * 1e3,
                  hit=a.hit, down=a.down, wait=a.wait, route=a.route)
    if res is None:
        sys.exit("no layer segments found (check --route / --hit / --wait against the trace's kernel names)")
    print(f"decode wall {res['wall_ms']:.1f} ms, {res['segments']} layer segments, {res['gpu_bound']} GPU-bound "
          f"(wait <= floor {res['floor_us']:.1f} us + {a.eps_us:g} us)")
    print(f"hit kernels {res['hit_ms']:.1f} ms (down {res['down_ms']:.1f} ms), wait kernels {res['wait_ms']:.1f} ms")
    print(f"exposed hit {res['exposed_hit_ms']:.1f} ms ({res['exposed_share_of_hit']:.0%} of hit time; down "
          f"{res['exposed_down_ms']:.1f} ms)")
    print(f"ceiling at {res['speedup']:.0%} faster hit kernels: {res['ceiling_ms']:.1f} ms = {res['ceiling_pct_wall']:.2f}% "
          f"of the decode wall -> {res['verdict']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
