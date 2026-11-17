#!/usr/bin/env python3
"""kernel_axis.py - the second axis of a confirm (LEADS.md Part P lead L10): decode tok/s AND rocprof kernel time.

Method idea from hipfire's autoresearch gate (perf_policy.py / verdict.py: a WIN needs the throughput UP and the kernel
duration DOWN, each by its own one-sided Mann-Whitney U beyond a floor; a gain on one axis only is DEAD; a kernel is EXHAUSTED
after 5 consecutive DEAD / INCONCLUSIVE tests, a WIN resets; a kernel is a candidate only at >= 3% of the decode wall).  Our own
code, no code read or copied.  The Mann-Whitney is tools/pancake/sweep.py's.

Three commands, nothing here starts an engine by itself:

  wrap   python tools/pancake/kernel_axis.py wrap SPEC.json --out DIR
         writes DIR/<arm>.sh (exec rocprofv3 --kernel-trace on the arm's engine binary, one trace directory per start) and
         DIR/spec.json, the spec with every arm's `exe` pointing at its wrapper.  Run abtest.py on DIR/spec.json (the box: a
         profiled engine decodes about 1.6x slower, so the tok/s axis of THAT run is judged against the profiled base, never
         against an unprofiled one).  abtest.py itself is untouched.
  judge  python tools/pancake/kernel_axis.py judge --jsonl RESULT.jsonl --dir DIR --base base --arm ARM --case edit
              [--kernel REGEX] [--floor 1.0] [--alpha 0.1]
         reads the decode_tps of the case from the abtest jsonl and the kernel time per start from the trace CSVs in
         DIR/<arm>/, prints WIN / DEAD / INCONCLUSIVE per axis and overall.
  exhaust python tools/pancake/kernel_axis.py exhaust HISTORY.jsonl [--limit 5]
         HISTORY: one {"kernel": ..., "verdict": "WIN|DEAD|INCONCLUSIVE"} per line, oldest first; prints the kernels that are
         exhausted (the micro-bench queue stops testing them).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from sweep import p_faster  # noqa: E402  (one-sided Mann-Whitney: p that `new` is higher than `old`)

ROCPROF = "/opt/rocm/core-10.0/bin/rocprofv3"
WIN, DEAD, INCONCLUSIVE = "WIN", "DEAD", "INCONCLUSIVE"
MIN_N = 3          # fewer starts on either side: nothing is judged
MIN_SHARE = 3.0    # % of the decode wall a kernel needs to be a candidate


def parse_kernel_trace(path) -> dict[str, list[float]]:
    """rocprofv3 kernel_trace CSV -> {kernel name: [duration in ms]}.  Column names differ between versions
    (Kernel_Name / Start_Timestamp / End_Timestamp; older: KernelName, BeginNs, EndNs): the first of each that is present."""
    out: dict[str, list[float]] = {}
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []

        def pick(*names):
            return next((n for n in names if n in cols), None)

        kn, st, en = pick("Kernel_Name", "KernelName", "Name"), pick("Start_Timestamp", "BeginNs", "Start"), pick("End_Timestamp", "EndNs", "End")
        if not (kn and st and en):
            raise ValueError(f"{path}: not a kernel trace (columns {cols})")
        for row in rd:
            try:
                out.setdefault(row[kn], []).append((float(row[en]) - float(row[st])) / 1e6)
            except (TypeError, ValueError):
                continue
    return out


def kernel_ms(trace: dict[str, list[float]], pattern: str | None) -> float:
    """Total ms of the kernels whose name matches `pattern` (None: every kernel)."""
    rx = re.compile(pattern) if pattern else None
    return sum(sum(v) for k, v in trace.items() if rx is None or rx.search(k))


def share_of_wall(trace: dict[str, list[float]], pattern: str, wall_ms: float) -> float:
    """% of the decode wall the matching kernels take; a candidate needs >= MIN_SHARE."""
    return 100.0 * kernel_ms(trace, pattern) / wall_ms if wall_ms > 0 else 0.0


def axis(new, old, higher_is_better: bool, floor_pct: float, alpha: float) -> str:
    """WIN: the arm is better than the base by at least floor_pct on average AND one-sided Mann-Whitney p <= alpha.
    INCONCLUSIVE with fewer than MIN_N values a side; DEAD otherwise (not better)."""
    if len(new) < MIN_N or len(old) < MIN_N:
        return INCONCLUSIVE
    mn, mo = statistics.fmean(new), statistics.fmean(old)
    if mo == 0:
        return INCONCLUSIVE
    if higher_is_better:
        gain = 100.0 * (mn / mo - 1.0)
        p = p_faster(new, old)
    else:
        gain = 100.0 * (1.0 - mn / mo)
        p = p_faster(old, new)   # the base is higher: the arm's time is lower
    return WIN if gain >= floor_pct and p <= alpha else DEAD


def verdict(base_tps, arm_tps, base_kernel_ms, arm_kernel_ms, floor_pct: float = 1.0, alpha: float = 0.1) -> dict:
    """The conjunctive verdict: WIN only when decode tok/s is up AND the kernel time is down; a gain on one axis only is DEAD
    (a kernel that got faster without the run following is not the bottleneck; a run that got faster with the kernel unchanged
    is noise or something else).  Too few starts on an axis: INCONCLUSIVE."""
    t = axis(arm_tps, base_tps, True, floor_pct, alpha)
    k = axis(arm_kernel_ms, base_kernel_ms, False, floor_pct, alpha)
    if INCONCLUSIVE in (t, k):
        v = INCONCLUSIVE
    else:
        v = WIN if (t == WIN and k == WIN) else DEAD
    return {"tps": t, "kernel": k, "verdict": v}


def exhausted(history: list[dict], limit: int = 5) -> list[str]:
    """Kernels whose last `limit` verdicts, counted back from the newest, are all DEAD or INCONCLUSIVE (a WIN resets)."""
    streak: dict[str, int] = {}
    for h in history:
        k = h["kernel"]
        streak[k] = 0 if h["verdict"] == WIN else streak.get(k, 0) + 1
    return sorted(k for k, n in streak.items() if n >= limit)


def wrap(spec_path, out_dir, rocprof=ROCPROF) -> Path:
    spec = json.load(open(spec_path))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base_exe = spec.get("exe")
    for arm, a in spec["arms"].items():
        exe = a.get("exe") or base_exe
        if not exe:
            raise SystemExit(f"kernel_axis: arm {arm!r} has no exe (and the spec has no top-level exe): the preset's engine would be wrapped, name it")
        sh = out / f"{arm}.sh"
        # one trace directory per engine start (the pid); rocprofv3 writes <dir>/<host>/<pid>_kernel_trace.csv
        sh.write_text("#!/bin/sh\nexec %s --kernel-trace --output-format csv -d %s/%s -- %s \"$@\"\n" % (rocprof, out, arm, exe), encoding="utf-8")
        sh.chmod(0o755)
        a["exe"] = str(sh)
    spec["note"] = (spec.get("note", "") + " [kernel_axis wrap: every arm runs under rocprofv3 --kernel-trace; judge with tools/pancake/kernel_axis.py judge]").strip()
    p = out / "spec.json"
    p.write_text(json.dumps(spec, indent=1), encoding="utf-8")
    return p


def starts_kernel_ms(arm_dir, pattern) -> list[float]:
    return [kernel_ms(parse_kernel_trace(p), pattern) for p in sorted(Path(arm_dir).rglob("*kernel_trace.csv"))]


def jsonl_tps(jsonl, arm, case) -> list[float]:
    vals = []
    for line in open(jsonl):
        r = json.loads(line)
        if r.get("arm") != arm:
            continue
        vals += [x["decode_tps"] for x in r.get("rows", []) if x.get("case") == case]
    return vals


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("wrap")
    w.add_argument("spec")
    w.add_argument("--out", required=True)
    w.add_argument("--rocprof", default=ROCPROF)
    j = sub.add_parser("judge")
    j.add_argument("--jsonl", required=True)
    j.add_argument("--dir", required=True)
    j.add_argument("--base", default="base")
    j.add_argument("--arm", required=True)
    j.add_argument("--case", required=True)
    j.add_argument("--kernel", default=None, help="regex of the kernel names to sum (default: all)")
    j.add_argument("--floor", type=float, default=1.0)
    j.add_argument("--alpha", type=float, default=0.1)
    e = sub.add_parser("exhaust")
    e.add_argument("history")
    e.add_argument("--limit", type=int, default=5)
    a = ap.parse_args(argv)
    if a.cmd == "wrap":
        print(wrap(a.spec, a.out, a.rocprof))
        return 0
    if a.cmd == "judge":
        r = verdict(jsonl_tps(a.jsonl, a.base, a.case), jsonl_tps(a.jsonl, a.arm, a.case),
                    starts_kernel_ms(Path(a.dir) / a.base, a.kernel), starts_kernel_ms(Path(a.dir) / a.arm, a.kernel), a.floor, a.alpha)
        print(json.dumps(r))
        return 0 if r["verdict"] == WIN else 1
    hist = [json.loads(x) for x in open(a.history) if x.strip()]
    for k in exhausted(hist, a.limit):
        print(k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
