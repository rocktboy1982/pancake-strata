#!/usr/bin/env python3
"""lookahead_ctrl.py - pancake-strata: read the STRATA_LOOKAHEAD_CTRL=1 lines of an engine log (stderr) and print the per-layer table.

    python tools/pancake/lookahead_ctrl.py engine.log [--json OUT.json]

Lines (include/strata/program/lookahead_ctrl.hpp, printed every 100 windows with STRATA_LOOKAHEAD_STATS; the LAST block of a log wins):
  strata lookahead ctrl: windows=W slots=S stale_gpu=a stale_cpu=b control=c prev=d
  strata lookahead ctrl layer L: slots=S stale_gpu=a stale_cpu=b control=c prev=d
Every figure is the share of the TRUE routed slots (all experts, resident or not) found in the predicted set of the same size k:
stale_gpu what the engine acts on; stale_cpu the host BF16 router on the previous layer's input; control the layer's own router on
its own input (must read 1.0000: below that the probe or the router copy is wrong); prev the previous token of the window.
The verdict line says which of the three readings the log gives; it never states a result the log does not carry.
Offline, stdlib only.
"""
import argparse
import json
import re
import sys

SUM_RE = re.compile(r"strata lookahead ctrl: windows=(\d+) slots=(\d+) stale_gpu=(\S+) stale_cpu=(\S+) control=(\S+) prev=(\S+)")
LAY_RE = re.compile(r"strata lookahead ctrl layer (\d+): slots=(\d+) stale_gpu=(\S+) stale_cpu=(\S+) control=(\S+) prev=(\S+)")
KEYS = ("stale_gpu", "stale_cpu", "control", "prev")


def _f(s):
    return None if s == "n/a" else float(s)


def parse(lines):
    """-> {'windows', 'slots', 'total': {key: float|None}, 'layers': {L: {'slots', key: ...}}} of the last block, or None."""
    blocks, cur = [], None
    for ln in lines:
        m = SUM_RE.search(ln)
        if m:
            cur = {"windows": int(m.group(1)), "slots": int(m.group(2)), "total": dict(zip(KEYS, map(_f, m.groups()[2:]))), "layers": {}}
            blocks.append(cur)
            continue
        m = LAY_RE.search(ln)
        if m and cur is not None:
            cur["layers"][int(m.group(1))] = dict(slots=int(m.group(2)), **dict(zip(KEYS, map(_f, m.groups()[2:]))))
    return blocks[-1] if blocks else None


def verdict(b):
    ctl = b["total"]["control"]
    if ctl is None:
        return "no control reading in the log"
    if ctl < 0.999:
        return "CONTROL %.4f < 1: the probe is wrong (a router copy, the input rows, or a routing rule that changes the committed top-k); the stale numbers are not interpretable" % ctl
    return "control 1.0000: the probe reads the routing back, so the stale-gate overlap is a property of the predictor"


def table(b):
    def c(v):
        return "   n/a" if v is None else "%6.3f" % v
    out = ["layer  slots  stale_gpu stale_cpu control   prev"]
    for L in sorted(b["layers"]):
        r = b["layers"][L]
        out.append("%5d %6d  %s    %s  %s %s" % (L, r["slots"], c(r["stale_gpu"]), c(r["stale_cpu"]), c(r["control"]), c(r["prev"])))
    t = b["total"]
    out.append("total %6d  %s    %s  %s %s   (%d windows)" % (b["slots"], c(t["stale_gpu"]), c(t["stale_cpu"]), c(t["control"]), c(t["prev"]), b["windows"]))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    with open(a.log, encoding="utf-8", errors="replace") as f:
        b = parse(f)
    if b is None:
        print("no 'strata lookahead ctrl' lines in %s (run with STRATA_LOOKAHEAD_CTRL=1 and at least 100 windows)" % a.log)
        return 1
    print(table(b))
    print(verdict(b))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(b, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
