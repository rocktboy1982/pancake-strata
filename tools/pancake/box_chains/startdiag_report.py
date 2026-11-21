#!/usr/bin/env python3
"""startdiag_report.py - post-hoc analysis of the T-1006 slow engine starts (stdlib only).

abtest.py prints, per engine start, a `STARTDIAG <ts> prev_run_s=.. gap_s=.. ...` line and, once the engine is ready, a `STARTLOAD <ts> loaded X GiB at Y GiB/s
rate=Y` line; before a start that follows a long run with ABTEST_SETTLE_AFTER_LONG_S set it prints `SETTLE <s> after a <run> s run`.  This script reads the stdout
files of a sweep (files, directories or globs; directories are searched for *.out and *.log), pairs each STARTDIAG with the STARTLOAD after it, and prints

  1. the slow share (load rate < --slow GiB/s, default 0.8) by gap bucket (gap_s < 130, 130-199, >= 200; first start of a process: no gap),
  2. the same by SETTLE on / off (SETTLE on = a SETTLE line came between the previous STARTDIAG of the file and this one), and bucket x settle,
  3. the diag correlations: Pearson r of the load rate with each diag number, and the mean of each number for slow and for fast starts,
  4. S2 (LEADS.md Part S, ABTEST_QUEUEDIAG=1): the `STARTQUEUE <ts> pids=.. gfx=.. cp_queues=.. queues=compute:2,sdma:1 evicted_ms=.. hqds=ok|NA pipes=i0p0:2,..
     shared=0|1|NA` line (engine ready) and the `STOPQUEUE ...` line (just before the server stops) are attached to the start they belong to: the slow share by
     pipe sharing (shared = two or more queues on one CP pipe, needs root's /sys/kernel/debug/kfd/hqds, else NA), by KFD eviction (evicted_ms at stop > 0) and the
     means of evicted_ms and of the compute queue count for slow and fast starts.

Run: python3 startdiag_report.py FILE_OR_DIR_OR_GLOB ...   (--slow 0.8 --min-n 20)"""
import argparse
import glob
import math
import os
import re
import sys

BUCKETS = (("gap < 130 s", 0, 130), ("gap 130-199 s", 130, 200), ("gap >= 200 s", 200, math.inf))
FIELDS = ("prev_run_s", "gap_s", "rd_sect", "wr_sect", "io_ticks_ms", "d_rd_sect", "d_wr_sect", "d_io_ticks_ms", "cached_kb", "mlocked_kb",
          "memavail_kb", "nvme_temp_c", "compact_stall_d")
KV = re.compile(r"([a-z_]+)=(\S+)")
KVS = re.compile(r"([a-z_]+)=(\S+)")
QFIELDS = ("evicted_start_ms", "evicted_stop_ms", "q_compute", "shared")
RATE = re.compile(r"rate=([0-9.]+)")


def num(s):
    try:
        return float(s)
    except ValueError:
        return None


def parse_text(text, src=""):
    """Starts of one file: dicts of the STARTDIAG fields + rate (None when no STARTLOAD / NA) + settle (bool) + src."""
    starts, cur, last, settle = [], None, None, False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("SETTLE "):
            settle = True
        elif line.startswith("STARTDIAG "):
            cur = {k: num(v) for k, v in KV.findall(line) if k in FIELDS}
            cur.update(rate=None, settle=settle, src=src, ts=line.split()[1] if len(line.split()) > 1 else "")
            starts.append(cur)
            last = cur
            settle = False
        elif line.startswith("STARTLOAD ") and cur is not None and cur["rate"] is None:
            m = RATE.search(line)
            cur["rate"] = float(m.group(1)) if m else None
            cur = None
        elif line.startswith(("STARTQUEUE ", "STOPQUEUE ")) and last is not None:   # S2: after the STARTLOAD of the start they belong to
            kv = dict(KVS.findall(line))
            pre = "start" if line.startswith("STARTQUEUE") else "stop"
            last[f"evicted_{pre}_ms"] = num(kv.get("evicted_ms", "NA"))
            if pre == "start":
                last["shared"] = num(kv.get("shared", "NA"))
                m = re.search(r"compute:(\d+)", kv.get("queues", ""))
                last["q_compute"] = float(m.group(1)) if m else None
                last["pipes"] = kv.get("pipes", "NA")
            elif last.get("shared") is None:
                last["shared"] = num(kv.get("shared", "NA"))
    return starts


def files_of(args):
    out = []
    for a in args:
        if os.path.isdir(a):
            for root, _, names in os.walk(a):
                out += [os.path.join(root, n) for n in sorted(names) if n.endswith((".out", ".log"))]
        else:
            out += sorted(glob.glob(a)) or ([a] if os.path.exists(a) else [])
    return sorted(set(out))


def bucket_of(gap):
    if gap is None:
        return "first start (no gap)"
    return next(n for n, lo, hi in BUCKETS if lo <= gap < hi)


def share(rows, slow):
    n = len(rows)
    k = sum(1 for r in rows if r["rate"] < slow)
    return n, k


def fmt_share(rows, slow):
    n, k = share(rows, slow)
    return f"{k:4d} of {n:4d} slow ({100.0 * k / n:5.1f}%)" if n else "   0 of    0"


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def report(starts, slow=0.8, min_n=20, out=print):
    rated = [s for s in starts if s["rate"] is not None]
    out(f"engine starts: {len(starts)} with a STARTDIAG line, {len(rated)} with a load rate; slow = load < {slow:g} GiB/s")
    if not rated:
        return
    out(f"all starts: {fmt_share(rated, slow)}")
    out("\nby gap between consecutive engine starts")
    names = [b[0] for b in BUCKETS] + ["first start (no gap)"]
    for nm in names:
        rows = [r for r in rated if bucket_of(r.get("gap_s")) == nm]
        if rows:
            out(f"  {nm:22s} {fmt_share(rows, slow)}")
    out("\nby SETTLE")
    for lab, on in (("SETTLE on", True), ("SETTLE off", False)):
        rows = [r for r in rated if r["settle"] == on]
        out(f"  {lab:22s} {fmt_share(rows, slow)}")
    out("\ngap bucket x SETTLE (the 130-199 s row with SETTLE vs the 76%% baseline is the first result; >= %d starts wanted per slice)" % min_n)
    for nm in names:
        for lab, on in (("on", True), ("off", False)):
            rows = [r for r in rated if bucket_of(r.get("gap_s")) == nm and r["settle"] == on]
            if rows:
                note = "" if len(rows) >= min_n else f"   (n < {min_n}: not conclusive)"
                out(f"  {nm:22s} settle {lab:3s} {fmt_share(rows, slow)}{note}")
    out("\ndiag correlations (Pearson r with the load rate; mean for slow / fast starts; n = starts with the value)")
    out(f"  {'field':16s} {'n':>5s} {'r':>7s} {'mean slow':>14s} {'mean fast':>14s}")
    for fld in FIELDS + QFIELDS:
        rows = [r for r in rated if r.get(fld) is not None]
        if not rows:
            out(f"  {fld:16s} {0:5d}      NA")
            continue
        r_ = pearson([r[fld] for r in rows], [r["rate"] for r in rows])
        sl = [r[fld] for r in rows if r["rate"] < slow]
        fa = [r[fld] for r in rows if r["rate"] >= slow]
        m = lambda v: f"{sum(v) / len(v):14.1f}" if v else f"{'NA':>14s}"   # noqa: E731
        out(f"  {fld:16s} {len(rows):5d} {('%7.2f' % r_) if r_ is not None else '     NA'} {m(sl)} {m(fa)}")
    if any(r.get("shared") is not None or r.get("evicted_stop_ms") is not None for r in rated):
        out("\nS2 queue layout (slow = load < %g GiB/s; shared needs root's hqds, NA otherwise; evicted = evicted_ms at stop > 0)" % slow)
        groups = (("pipe shared", lambda r: r.get("shared") == 1), ("pipe not shared", lambda r: r.get("shared") == 0),
                  ("pipe unknown (NA)", lambda r: r.get("shared") is None), ("evicted by KFD", lambda r: (r.get("evicted_stop_ms") or 0) > 0),
                  ("not evicted", lambda r: r.get("evicted_stop_ms") == 0))
        for lab, pred in groups:
            rows = [r for r in rated if pred(r)]
            if rows:
                out(f"  {lab:22s} {fmt_share(rows, slow)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="stdout files, directories or globs")
    ap.add_argument("--slow", type=float, default=0.8, help="GiB/s below which a start is slow (default 0.8)")
    ap.add_argument("--min-n", type=int, default=20)
    a = ap.parse_args(argv)
    starts = []
    for f in files_of(a.paths):
        try:
            with open(f, errors="replace") as fh:
                starts += parse_text(fh.read(), f)
        except OSError as e:
            print(f"cannot read {f}: {e}", file=sys.stderr)
    report(starts, a.slow, a.min_n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
