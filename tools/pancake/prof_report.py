#!/usr/bin/env python3
"""prof_report.py - pancake-strata LEADS.md Part T T11: rank the kernels of a rocprofv3 run by time / share and show the PMC counters per kernel.

Reads what `tools/pancake/box_chains/prof-t11.v1.sh` (or any `rocprofv3 ... -d DIR` run) wrote under DIR, CSV or JSON, any ROCm 10.x column spelling
(the loader lower-cases column names and accepts Kernel_Name / kernel_name / formatted_kernel_name, Start_Timestamp / start_timestamp):
  kernel_trace.csv            one row per dispatch with Start / End timestamps (ns)           -> time, share, calls, mean, gaps
  counter_collection.csv      one row per (dispatch, counter): Counter_Name, Counter_Value     -> mean / total per kernel and counter
  *_results.json              rocprofiler-sdk json (buffer_records.kernel_dispatch / counter_records + kernel_symbols); best effort, the CSV is the
                              primary path (the JSON layout of 10.x is [unverified] on gfx1100: the parser is checked on a fixture of the documented shape only)

    python3 tools/pancake/prof_report.py DIR_OR_FILE [more ...] [--top 25] [--match REGEX] [--json OUT.json] [--csv OUT.csv] [--counters SQ_WAVES,GRBM_GUI_ACTIVE]
Output: table of kernels by total time (ms, share %, calls, mean us), then per-kernel counters (mean per dispatch).  A kernel replayed for counters runs
alone, so its counter time is NOT the run time: ranking always uses the kernel trace when one is given.  Counters whose values are all zero are listed
as "zero" (a counter that does not exist on gfx1100 reads 0 or is missing: check `rocprofv3-avail list --pmc`).
Exit 0 ok, 2 nothing readable.  stdlib only, read-only except --json / --csv.
Tests: python -m pytest -q tools/pancake/test_prof_report.py   (fixtures; real profiles need the box)
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path


def _lower_rows(path: Path) -> list:
    with open(path, newline="", errors="replace") as f:
        return [{(k or "").strip().lower(): (v or "").strip() for k, v in row.items()} for row in csv.DictReader(f)]


def _first(row: dict, *keys, default=None):
    for k in keys:
        if k in row and row[k] != "":
            return row[k]
    return default


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def name_of(row: dict) -> str:
    return _first(row, "kernel_name", "formatted_kernel_name", "name", default="?")


def from_csv(path: Path) -> dict:
    """{'kernels': [{name, start, end, corr}], 'counters': [{name, counter, value, corr}], 'api': [{name, start, end, tid}]}"""
    out = {"kernels": [], "counters": [], "api": []}
    seen = set()
    for r in _lower_rows(path):
        s, e = _num(_first(r, "start_timestamp", "begin_timestamp", "start")), _num(_first(r, "end_timestamp", "end"))
        corr = _first(r, "correlation_id", "dispatch_id", default="")
        if "counter_name" in r:
            v = _num(r.get("counter_value"), 0.0)
            out["counters"].append({"name": name_of(r), "counter": r["counter_name"], "value": v, "corr": corr or _first(r, "dispatch_id", default="")})
            key = (corr, s, e)
            if s is not None and e is not None and key not in seen:
                seen.add(key)
                out["kernels"].append({"name": name_of(r), "start": s, "end": e, "corr": corr, "from_counters": True})
        elif "kernel_name" in r or "kernel_id" in r:
            if s is not None and e is not None:
                out["kernels"].append({"name": name_of(r), "start": s, "end": e, "corr": corr})
        elif "direction" in r and s is not None and e is not None:
            out["api"].append({"name": "memcpy " + r["direction"], "start": s, "end": e, "tid": "", "copy": True})
        elif "function" in r or ("kind" in r and "operation" in r) or "api_name" in r:
            if s is not None and e is not None:
                out["api"].append({"name": _first(r, "function", "api_name", "operation", default="?"), "start": s, "end": e,
                                   "tid": _first(r, "thread_id", default="")})
    return out


def from_json(path: Path) -> dict:
    out = {"kernels": [], "counters": [], "api": []}
    try:
        data = json.loads(Path(path).read_text(errors="replace"))
    except (OSError, ValueError):
        return out
    tools = data.get("rocprofiler-sdk-tool") if isinstance(data, dict) else None
    for t in (tools if isinstance(tools, list) else []):
        syms = {}
        for s in t.get("kernel_symbols", []) or []:
            syms[s.get("kernel_id")] = s.get("formatted_kernel_name") or s.get("kernel_name") or "?"
        ops = {}
        for ent in ((t.get("strings") or {}).get("buffer_records") or []):
            if isinstance(ent, dict):
                ops[ent.get("kind")] = ent.get("operations") or []
        br = t.get("buffer_records") or {}
        for d in br.get("kernel_dispatch", []) or []:
            info = d.get("dispatch_info") or {}
            nm = syms.get(info.get("kernel_id"), info.get("kernel_name", "?"))
            corr = (d.get("correlation_id") or {}).get("internal", "") if isinstance(d.get("correlation_id"), dict) else d.get("correlation_id", "")
            out["kernels"].append({"name": nm, "start": float(d.get("start_timestamp", 0)), "end": float(d.get("end_timestamp", 0)), "corr": str(corr)})
        for kind in ("hip_api", "hip_api_ext", "marker_api", "hsa_api"):
            for a in br.get(kind, []) or []:
                op = a.get("operation")
                nm = op if isinstance(op, str) else None
                if nm is None:
                    names = ops.get(a.get("kind"), [])
                    nm = names[op] if isinstance(op, int) and 0 <= op < len(names) else f"op{op}"
                out["api"].append({"name": nm, "start": float(a.get("start_timestamp", 0)), "end": float(a.get("end_timestamp", 0)),
                                   "tid": str(a.get("thread_id", ""))})
        for c in br.get("counter_records", []) or []:
            out["counters"].append({"name": str(c.get("dispatch_id", "?")), "counter": c.get("counter_name", "?"), "value": float(c.get("value", 0)), "corr": ""})
    return out


def load(paths) -> dict:
    """Merge every readable file: CSV or JSON; a directory contributes its *.csv and *.json files (recursively)."""
    merged = {"kernels": [], "counters": [], "api": [], "files": []}
    files = []
    for p in map(Path, paths):
        files += sorted(x for x in p.rglob("*") if x.suffix in (".csv", ".json") and x.is_file()) if p.is_dir() else [p]
    for f in files:
        try:
            d = from_json(f) if f.suffix == ".json" else from_csv(f)
        except (OSError, csv.Error, UnicodeError):
            continue
        if any(d.values()):
            merged["files"].append(str(f))
        # a counter CSV repeats the dispatch times: keep them only when no trace file gave kernels
        merged["kernels"] += [k for k in d["kernels"]]
        merged["counters"] += d["counters"]
        merged["api"] += d["api"]
    # drop counter-derived kernel rows when real trace rows exist
    if any(not k.get("from_counters") for k in merged["kernels"]):
        merged["kernels"] = [k for k in merged["kernels"] if not k.get("from_counters")]
    return merged


def rank(kernels: list, match=None) -> list:
    agg = collections.OrderedDict()
    for k in kernels:
        if match and not re.search(match, k["name"]):
            continue
        a = agg.setdefault(k["name"], {"name": k["name"], "calls": 0, "ns": 0.0, "max_ns": 0.0})
        d = max(0.0, k["end"] - k["start"])
        a["calls"] += 1
        a["ns"] += d
        a["max_ns"] = max(a["max_ns"], d)
    total = sum(a["ns"] for a in agg.values()) or 1.0
    rows = sorted(agg.values(), key=lambda a: -a["ns"])
    for a in rows:
        a["share"] = a["ns"] / total
        a["mean_us"] = a["ns"] / a["calls"] / 1000.0
    return rows


def counter_table(counters: list, only=None) -> dict:
    """{kernel: {counter: {'n', 'sum', 'mean'}}}"""
    t: dict = {}
    for c in counters:
        if only and c["counter"] not in only:
            continue
        e = t.setdefault(c["name"], {}).setdefault(c["counter"], {"n": 0, "sum": 0.0})
        e["n"] += 1
        e["sum"] += c["value"]
    for ctrs in t.values():
        for e in ctrs.values():
            e["mean"] = e["sum"] / e["n"] if e["n"] else 0.0
    return t


def render(rows: list, ctab: dict, top: int) -> str:
    out = [f"{'kernel':60} {'ms':>10} {'share%':>7} {'calls':>8} {'mean_us':>9}"]
    for a in rows[:top]:
        out.append(f"{a['name'][:60]:60} {a['ns'] / 1e6:>10.3f} {a['share'] * 100:>7.2f} {a['calls']:>8} {a['mean_us']:>9.2f}")
    if ctab:
        names = sorted({c for v in ctab.values() for c in v})
        out.append("")
        out.append("counters, mean per dispatch (zero = every value 0):")
        out.append(f"{'kernel':60} " + " ".join(f"{n[:16]:>16}" for n in names))
        order = [a["name"] for a in rows if a["name"] in ctab] or sorted(ctab)
        for k in order[:top]:
            cells = []
            for n in names:
                e = ctab[k].get(n)
                cells.append(f"{'-':>16}" if e is None else (f"{'zero':>16}" if e["sum"] == 0 else f"{e['mean']:>16.1f}"))
            out.append(f"{k[:60]:60} " + " ".join(cells))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--match")
    ap.add_argument("--counters", help="comma list of counter names to show (default all)")
    ap.add_argument("--json")
    ap.add_argument("--csv")
    a = ap.parse_args(argv)
    d = load(a.paths)
    if not d["kernels"] and not d["counters"]:
        print("nothing readable (need kernel_trace.csv / counter_collection.csv / rocprofiler-sdk json)", file=sys.stderr)
        return 2
    rows = rank(d["kernels"], a.match)
    ctab = counter_table(d["counters"], set(a.counters.split(",")) if a.counters else None)
    if a.match:
        ctab = {k: v for k, v in ctab.items() if re.search(a.match, k)}
    print(render(rows, ctab, a.top))
    print(f"{len(d['kernels'])} dispatches, {len(rows)} kernels, files: {len(d['files'])}")
    if a.json:
        Path(a.json).write_text(json.dumps({"kernels": rows, "counters": ctab}, indent=1))
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kernel", "ms", "share", "calls", "mean_us"])
            for r in rows:
                w.writerow([r["name"], round(r["ns"] / 1e6, 4), round(r["share"], 5), r["calls"], round(r["mean_us"], 3)])
    return 0


if __name__ == "__main__":
    sys.exit(main())
