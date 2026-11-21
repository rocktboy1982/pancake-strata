#!/usr/bin/env python3
"""kernel_resources_obj.py - pancake-strata LEADS.md Part T T9: VGPR / SGPR / LDS / scratch / occupancy per kernel of an ALREADY BUILT gfx1100 code object.

kernel_resources.py compiles sources itself (one translation unit at a time, own flags); this tool reads what a real box BUILD produced, so the numbers are
those of the shipped binary (same flags, same ROCm 10.0 clang):
  A. code-object notes:   llvm-readelf --notes CODEOBJ  (CODEOBJ = the extracted device ELF, see the recipe), fields .vgpr_count .sgpr_count
                          .private_segment_fixed_size .group_segment_fixed_size .vgpr_spill_count .sgpr_spill_count .max_flat_workgroup_size .wavefront_size
  B. build-log remarks:   a build log made with CMAKE_HIP_FLAGS=-Rpass-analysis=kernel-resource-usage (same parser as kernel_resources.py)
Both may be given; a kernel found in both is merged (notes win for counts, remarks add occupancy).  Occupancy not reported by the compiler is MODELLED from
VGPRs (hip_resources.model_waves) and marked "(m)".  Flags: SPILL (scratch > 0 or a spill count), VGPR>N (default 128), LDS>N (default 32768),
LOWOCC (waves < --min-waves, default 4).

    # box recipe (run only when the box is idle; a configure + build is heavy and needs the owner's go):
    sh tools/pancake/box_chains/kres-t9.sh            # default OFF: prints the plan; STRATA_T9_RUN=1 sh ... really runs
    # any host, on saved text:
    python3 tools/pancake/kernel_resources_obj.py --notes notes.txt [--log build.log] [--csv out.csv] [--json out.json] [--vgpr-max 128] [--fail-on-spill]
Exit 0 ok, 1 spill with --fail-on-spill, 2 nothing parsed.  Offline, stdlib only; it never runs a kernel and never writes outside --csv / --json.
Tests: python -m pytest -q tools/pancake/test_kernel_resources_obj.py   (fixtures; real code objects need the box)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hip_resources as H  # noqa: E402
import kernel_resources as K  # noqa: E402

NOTE_KEYS = {".vgpr_count": "vgprs", ".sgpr_count": "sgprs", ".private_segment_fixed_size": "scratch", ".group_segment_fixed_size": "lds",
             ".vgpr_spill_count": "spills_v", ".sgpr_spill_count": "spills_s", ".max_flat_workgroup_size": "max_wg", ".wavefront_size": "wave"}
_KV = re.compile(r"^(\s*)(?:-\s+)?(\.[A-Za-z_]+):\s*(.*?)\s*$")


def parse_notes(text: str) -> dict:
    """{kernel .name: {vgprs, sgprs, scratch, lds, spills_v, spills_s, max_wg, wave}} from `llvm-readelf --notes` (YAML-like AMDGPU metadata).
    A kernel entry is a list item of amdhsa.kernels; the list items of .args are deeper and are skipped by indent."""
    out: dict = {}
    cur = None
    kern_indent = None
    for line in text.splitlines():
        if re.match(r"^\s*amdhsa\.kernels:", line):
            kern_indent = None
            continue
        m = _KV.match(line)
        if not m:
            continue
        indent, key, val = len(m.group(1)), m.group(2), m.group(3)
        if line.lstrip().startswith("- "):
            if kern_indent is None:
                kern_indent = indent
            if indent == kern_indent:
                cur = {}
        if kern_indent is None or cur is None:
            continue
        if indent not in (kern_indent, kern_indent + 2):
            continue                       # deeper keys belong to .args
        if key == ".name":
            out[val.strip("'\"")] = cur
        elif key in NOTE_KEYS:
            try:
                cur[NOTE_KEYS[key]] = int(val, 0)
            except ValueError:
                pass
    return out


def merge(notes: dict, remarks: dict) -> list:
    rows = []
    for name in list(notes) + [n for n in remarks if n not in notes]:
        r = dict(remarks.get(name, {}))
        r.update(notes.get(name, {}))
        if "occupancy" in remarks.get(name, {}):
            r["occupancy"] = remarks[name]["occupancy"]
            r["occ_src"] = "compiler"
        elif "vgprs" in r:
            r["occupancy"] = H.model_waves(r["vgprs"])
            r["occ_src"] = "model"
        r["name"] = name
        r["short"] = K.short_name(name)
        rows.append(r)
    return rows


def flag(r: dict, vgpr_max=128, lds_max=32768, min_waves=4) -> list:
    f = []
    if r.get("scratch", 0) or r.get("spills_v", 0) or r.get("spills_s", 0):
        f.append("SPILL")
    if r.get("vgprs", 0) > vgpr_max:
        f.append(f"VGPR>{vgpr_max}")
    if r.get("lds", 0) > lds_max:
        f.append(f"LDS>{lds_max}")
    if r.get("occupancy") is not None and r["occupancy"] < min_waves:
        f.append("LOWOCC")
    return f


def render(rows: list, **kw) -> str:
    lines = [f"{'kernel':44} {'vgpr':>5} {'sgpr':>5} {'scratch':>8} {'spillV':>6} {'lds':>7} {'waves':>6}  flags"]
    for r in sorted(rows, key=lambda r: (not flag(r, **kw), -r.get("vgprs", 0))):
        occ = r.get("occupancy")
        occs = "-" if occ is None else f"{occ}{'' if r.get('occ_src') == 'compiler' else '(m)'}"
        lines.append(f"{r['short'][:44]:44} {r.get('vgprs', '-'):>5} {r.get('sgprs', '-'):>5} {r.get('scratch', '-'):>8} {r.get('spills_v', '-'):>6} "
                     f"{r.get('lds', '-'):>7} {occs:>6}  {' '.join(flag(r, **kw))}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--notes", action="append", default=[], help="llvm-readelf --notes output file (repeatable)")
    ap.add_argument("--log", action="append", default=[], help="build log with kernel-resource-usage remarks (repeatable)")
    ap.add_argument("--csv")
    ap.add_argument("--json")
    ap.add_argument("--vgpr-max", type=int, default=128)
    ap.add_argument("--lds-max", type=int, default=32768)
    ap.add_argument("--min-waves", type=int, default=4)
    ap.add_argument("--fail-on-spill", action="store_true")
    a = ap.parse_args(argv)
    notes, remarks = {}, {}
    for p in a.notes:
        notes.update(parse_notes(Path(p).read_text(errors="replace")))
    for p in a.log:
        remarks.update(K.parse_remarks(Path(p).read_text(errors="replace")))
    rows = merge(notes, remarks)
    if not rows:
        print("no kernels found (notes need 'amdhsa.kernels:' from llvm-readelf --notes; logs need -Rpass-analysis=kernel-resource-usage)", file=sys.stderr)
        return 2
    kw = dict(vgpr_max=a.vgpr_max, lds_max=a.lds_max, min_waves=a.min_waves)
    print(render(rows, **kw))
    spills = sum(1 for r in rows if "SPILL" in flag(r, **kw))
    print(f"{len(rows)} kernels, {spills} with spill/scratch, {sum(1 for r in rows if 'LOWOCC' in flag(r, **kw))} low occupancy")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "vgprs", "sgprs", "scratch", "spills_v", "spills_s", "lds", "occupancy", "occ_src", "flags"])
            for r in rows:
                w.writerow([r["name"]] + [r.get(k, "") for k in ("vgprs", "sgprs", "scratch", "spills_v", "spills_s", "lds", "occupancy", "occ_src")]
                           + [" ".join(flag(r, **kw))])
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 1 if (a.fail_on_spill and spills) else 0


if __name__ == "__main__":
    sys.exit(main())
