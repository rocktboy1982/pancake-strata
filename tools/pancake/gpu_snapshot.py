#!/usr/bin/env python3
"""gpu_snapshot.py - pancake-strata LEADS.md Part T T15: read-only snapshot of the GPU's clocks, power cap, perf level and PCIe link (sysfs + amd-smi / rocm-smi).

Why: (a) MOD_PLAN.md says the bus is PCIe 3.0 (13.4 GB/s h2d probe) while sysfs showed current_link_speed 16.0 GT/s x16 during the 2026-11-21 screen: the
link may train UP under load and idle DOWN (ASPM / DPM pcie levels), so the reading depends on the moment.  Owner fact 2026-11-21: the box runs Gen3 now
and gets a faster (Gen4) link after the current work: all benchmarks of this run are Gen3 and expert-streaming results must be re-measured after the
upgrade.  This script records, with a date stamp and a phase label, BOTH current and max link speed / width, the DPM pcie level, the power cap and
clocks, and (given) the measured host-to-device bandwidth, so before / after the upgrade can be compared (--compare) and the reading reconciled.
(b) energy per token (amd-smi energy counter, 10.2) needs a before / after read: take one snapshot before and one after a run.

    python3 tools/pancake/gpu_snapshot.py --phase idle   [--out DIR] [--samples 1 --interval 1] [--h2d-gbs 13.4 | --h2d-json FILE] [--no-smi]
    python3 tools/pancake/gpu_snapshot.py --phase load   --samples 30 --interval 1      (run while an engine is working: link state is sampled N times)
    python3 tools/pancake/gpu_snapshot.py --compare BEFORE.json AFTER.json
Box recipe (read-only, no GPU compute, safe while the box is busy; check at IDLE and again UNDER LOAD, and once more after the Gen4 upgrade):
    python3 tools/pancake/gpu_snapshot.py --phase idle --out /home/dan/auto-runs/snap
    # bandwidth microbenchmark (later, box idle only, needs a compile): tools/pancake/probes/h2d_bw.hip.cpp, header has the exact commands; its stdout JSON
    # goes to --h2d-json.  Expected: Gen3 x16 ~ 12-13.5 GB/s pinned, Gen4 x16 ~ 24-28 GB/s (W7900 journal: 28.5 GB/s).
--out DIR writes snapshot-<date>-<phase>[.n].json (never overwrites or deletes).  amd-smi: tries `--json` then `--csv`; a non-zero exit with usable output is
accepted (10.x changed the exit codes: partial N/A returns non-zero) and the exit code is recorded; columns / keys are matched by pattern, not by position,
so the 10.x CSV and JSON renames do not break it.  Falls back to rocm-smi.  Missing tools are recorded as null, never an error.
OWNER ACTIONS (root): none (all reads).  `lspci -vv -s <bdf>` LnkSta is world-readable for the link; the DPM pcie table needs no root either.
Exit 0 always for a snapshot; 2 for a --compare input error.  stdlib only.
Tests: python -m pytest -q tools/pancake/test_gpu_snapshot.py   (fake sysfs tree and fixture amd-smi text; the real tools need the box)
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

GEN = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}
LANE_GBS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}   # GB/s per lane after encoding, one direction
ROCM_BIN = os.environ.get("ROCM", "/opt/rocm/core-10.0") + "/bin"
SMI_KEYS = re.compile(r"pcie|link|speed|width|power|cap|clock|clk|sclk|mclk|gfx|energy|perf|profile", re.I)


def _read(p: Path):
    try:
        return p.read_text(errors="replace").strip()
    except OSError:
        return None


def parse_speed(s):
    """'16.0 GT/s PCIe' -> 16.0; None when unreadable."""
    m = re.search(r"([0-9.]+)\s*GT/s", s or "")
    return float(m.group(1)) if m else None


def link_info(speed_s, width_s):
    sp = parse_speed(speed_s)
    try:
        w = int(width_s)
    except (TypeError, ValueError):
        w = None
    gen = GEN.get(sp)
    gbs = round(LANE_GBS[gen] * w, 2) if gen and w else None
    return {"speed": speed_s, "gts": sp, "gen": gen, "width": w, "theoretical_gbs": gbs}


def active_line(text):
    """The line of a pp_dpm_* / pp_power_profile_mode table marked '*'."""
    if not text:
        return None
    for ln in text.splitlines():
        if "*" in ln:
            return ln.strip()
    return None


def sysfs_cards(root: Path) -> list:
    """AMD display devices under root/sys/class/drm (card*/device, vendor 0x1002), one dict each."""
    out = []
    for dev in sorted(glob.glob(str(root / "sys/class/drm/card*/device"))):
        d = Path(dev)
        if re.search(r"card\d+-", str(d.parent.name)) or _read(d / "vendor") not in ("0x1002", "0x1022"):
            continue
        cap = {}
        for hw in sorted(glob.glob(str(d / "hwmon/hwmon*"))):
            for k in ("power1_cap", "power1_cap_max", "power1_cap_min", "power1_average", "power1_input"):
                v = _read(Path(hw) / k)
                if v is not None and v.isdigit():
                    cap[k] = round(int(v) / 1e6, 1)       # microwatt -> W
            break
        out.append({
            "card": d.parent.name, "device_id": _read(d / "device"), "bdf": os.path.basename(os.path.realpath(d)) if d.exists() else None,
            "current": link_info(_read(d / "current_link_speed"), _read(d / "current_link_width")),
            "max": link_info(_read(d / "max_link_speed"), _read(d / "max_link_width")),
            "perf_level": _read(d / "power_dpm_force_performance_level"),
            "power_profile_active": active_line(_read(d / "pp_power_profile_mode")),
            "dpm_sclk": active_line(_read(d / "pp_dpm_sclk")), "dpm_mclk": active_line(_read(d / "pp_dpm_mclk")),
            "dpm_pcie": active_line(_read(d / "pp_dpm_pcie")),
            "power_w": cap,
        })
    return out


def flatten(obj, prefix=""):
    """{dotted.key: scalar} from nested json (lists get an index)."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}{k}."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(flatten(v, f"{prefix}{i}."))
    else:
        out[prefix.rstrip(".")] = obj
    return out


def parse_smi(text: str) -> dict:
    """Interesting fields of amd-smi / rocm-smi output, JSON or CSV (any ROCm 10.x spelling): {key: value}.  {} when nothing parses."""
    text = (text or "").strip()
    if not text:
        return {}
    try:
        flat = flatten(json.loads(text))
        return {k: v for k, v in flat.items() if SMI_KEYS.search(k) and v not in (None, "")}
    except ValueError:
        pass
    # CSV: header line with commas; one row per GPU; keep matching columns (rows keyed gpuN.column)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    start = next((i for i, ln in enumerate(lines) if "," in ln), None)
    if start is None:
        return {}
    out = {}
    try:
        for n, row in enumerate(csv.DictReader(io.StringIO("\n".join(lines[start:])))):
            for k, v in row.items():
                if k and v not in (None, "") and SMI_KEYS.search(k):
                    out[f"gpu{n}.{k.strip()}"] = v.strip()
    except csv.Error:
        return {}
    return out


def run_smi(args: list, timeout=20) -> dict:
    """{cmd, rc, fields, raw}; tries --json then --csv; accepts non-zero exit codes with usable output."""
    exe = shutil.which(args[0]) or (ROCM_BIN + "/" + args[0] if os.path.exists(ROCM_BIN + "/" + args[0]) else None)
    if exe is None:
        return {"cmd": args, "rc": None, "fields": None, "raw": None}
    last = None
    for fmt in ("--json", "--csv"):
        try:
            p = subprocess.run([exe] + args[1:] + [fmt], capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.SubprocessError):
            continue
        fields = parse_smi(p.stdout)
        last = {"cmd": [args[0]] + args[1:] + [fmt], "rc": p.returncode, "fields": fields or None, "raw": (p.stdout or p.stderr)[:2000]}
        if fields:
            return last
    return last or {"cmd": args, "rc": None, "fields": None, "raw": None}


def smi_block() -> dict:
    return {
        "amd-smi static bus": run_smi(["amd-smi", "static", "--bus"]),
        "amd-smi metric": run_smi(["amd-smi", "metric", "--power", "--clock", "--energy", "--pcie"]),
        "rocm-smi": run_smi(["rocm-smi", "--showclocks", "--showpower", "--showmaxpower", "--showperflevel"]),
    }


def reconcile(snap: dict) -> list:
    """Plain statements comparing the readings with MOD_PLAN's 'PCIe 3.0, 13.4 GB/s'."""
    msgs = []
    for c in snap.get("cards", []):
        cur, mx = c["current"], c["max"]
        msgs.append(f"{c['card']}: current {cur['speed']} x{cur['width']} (gen {cur['gen']}), max {mx['speed']} x{mx['width']} (gen {mx['gen']}), "
                    f"DPM pcie {c['dpm_pcie']}")
        if cur["gen"] and mx["gen"] and cur["gen"] < mx["gen"]:
            msgs.append("  link is below its max now: it idles down (or the slot is limited): compare idle and load snapshots")
        if cur["gen"] and cur["gen"] >= 4:
            msgs.append("  sysfs reports Gen4+ while MOD_PLAN assumes Gen3: sysfs may show the device-side max after training, take lspci LnkSta and the "
                        "measured h2d bandwidth as the judge")
    h = snap.get("h2d_gbs")
    if h is not None:
        msgs.append(f"measured h2d {h} GB/s: " + ("Gen3-class (the 13.4 GB/s working assumption holds)" if h < 16 else "Gen4-class" if h < 40 else "faster than Gen4 x16"))
    return msgs


def take(root: Path, phase: str, samples: int, interval: float, smi: bool, h2d=None) -> dict:
    snap = {"date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "phase": phase, "host": os.uname().nodename if hasattr(os, "uname") else "?",
            "cards": sysfs_cards(root), "link_samples": []}
    for i in range(max(0, samples - 1)):
        time.sleep(interval)
        snap["link_samples"].append([{"card": c["card"], "cur": c["current"]["speed"], "w": c["current"]["width"], "dpm_pcie": c["dpm_pcie"]} for c in sysfs_cards(root)])
    if snap["link_samples"]:
        seen = {(s["card"], s["cur"], s["w"]) for row in snap["link_samples"] for s in row}
        snap["link_states_seen"] = sorted(map(str, seen))
    if smi:
        snap["smi"] = smi_block()
    if h2d is not None:
        snap["h2d_gbs"] = h2d
    snap["reconcile"] = reconcile(snap)
    return snap


def compare(a: dict, b: dict) -> str:
    rows = []
    ca = {c["card"]: c for c in a.get("cards", [])}
    for c in b.get("cards", []):
        o = ca.get(c["card"])
        if not o:
            continue
        for label, f in (("current link", lambda x: f"{x['current']['speed']} x{x['current']['width']}"), ("max link", lambda x: f"{x['max']['speed']} x{x['max']['width']}"),
                         ("theoretical GB/s", lambda x: x["current"]["theoretical_gbs"]), ("perf level", lambda x: x["perf_level"]),
                         ("power cap W", lambda x: x["power_w"].get("power1_cap")), ("sclk", lambda x: x["dpm_sclk"]), ("mclk", lambda x: x["dpm_mclk"])):
            rows.append(f"{c['card']:7} {label:18} {str(f(o)):32} -> {str(f(c))}")
    rows.append(f"{'h2d GB/s':26} {str(a.get('h2d_gbs')):32} -> {b.get('h2d_gbs')}")
    rows.append(f"{'date':26} {a.get('date')}   ->   {b.get('date')}  ({a.get('phase')} -> {b.get('phase')})")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", default="idle", help="label: idle | load | gen3 | gen4 ...")
    ap.add_argument("--root", default="/", help="filesystem root (tests use a fake tree)")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--out")
    ap.add_argument("--no-smi", action="store_true")
    ap.add_argument("--h2d-gbs", type=float)
    ap.add_argument("--h2d-json", help="stdout JSON of probes/h2d_bw.hip.cpp (key h2d_gbs)")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    a = ap.parse_args(argv)
    if a.compare:
        try:
            x, y = (json.loads(Path(p).read_text()) for p in a.compare)
        except (OSError, ValueError) as e:
            print(f"cannot read: {e}", file=sys.stderr)
            return 2
        print(compare(x, y))
        return 0
    h2d = a.h2d_gbs
    if a.h2d_json:
        try:
            h2d = float(json.loads(Path(a.h2d_json).read_text().strip().splitlines()[-1])["h2d_gbs"])
        except (OSError, ValueError, KeyError, IndexError):
            print("warning: --h2d-json unreadable, no bandwidth recorded", file=sys.stderr)
    snap = take(Path(a.root), a.phase, a.samples, a.interval, not a.no_smi, h2d)
    text = json.dumps(snap, indent=1)
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        base = Path(a.out) / f"snapshot-{snap['date'][:10]}-{a.phase}"
        n, p = 0, Path(str(base) + ".json")
        while p.exists():
            n += 1
            p = Path(f"{base}.{n}.json")
        p.write_text(text)
        print(f"wrote {p}")
    for m in snap["reconcile"]:
        print(m)
    if not a.out:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
