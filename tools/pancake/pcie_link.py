#!/usr/bin/env python3
"""pcie_link.py - is ASPM L1 enabled on the PCIe path of the GPU?  Read only, Linux sysfs (LEADS.md Part O O5; the idea is the caminus authors' note
"the PCIe link stays awake": with L1 on the card's path the link sleeps during a ~4 ms launch and the completion reaches the host ~0.2 ms late on every
token; AGPL, idea only, this is our own code).

For every display-class AMD device (or the one named with --bdf) it walks the device and its upstream bridges up to the root port and prints, per hop,
the negotiated link (current speed / width), whether L0s / L1 / L1.1 / L1.2 ASPM is enabled (`link/*_aspm` in sysfs: kernel 6.x), and the global
policy (/sys/module/pcie_aspm/parameters/policy).  Verdict per device:

  L1-ON    L1 is enabled on at least one hop of the path: the 0.2 ms effect can exist; run the host-tune A/B (MOD_PLAN.md section 6 item 13, `host_tune.sh aspm`,
           policy=performance) and, if it helps and a sysfs write is not wanted, the link-touch probe `tools/pancake/probes/link_touch.hip.cpp`
  L1-OFF   no hop has L1 on: nothing to gain here, `STRATA_LINK_TOUCH` would measure noise
  UNKNOWN  the kernel exposes no `link/l1_aspm` file on the path (older kernel / no ASPM support): read `lspci -vv -s <bdf>` (LnkCtl: ASPM L1 Enabled)

    python3 tools/pancake/pcie_link.py [--bdf 0000:03:00.0] [--root /] [--json]
Tests: python -m pytest -q tools/pancake/test_pcie_link.py  (a fake sysfs tree; the real answer needs the box)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SEP = ":"   # the separator of a bus address in a directory name; tests on Windows (no colon in file names) set "_"
ASPM_FILES = ("l0s_aspm", "l1_aspm", "l1_1_aspm", "l1_2_aspm", "l1_1_pcipm", "l1_2_pcipm")


def _read(p: Path):
    try:
        return p.read_text().strip()
    except OSError:
        return None


def display_devices(root: Path, vendor: str = "0x1002") -> list:
    base = root / "sys/bus/pci/devices"
    out = []
    try:
        names = sorted(os.listdir(base))
    except OSError:
        return out
    for n in names:
        cls, ven = _read(base / n / "class"), _read(base / n / "vendor")
        if cls and cls.startswith("0x03") and (vendor is None or ven == vendor):
            out.append(n)
    return out


def path_to_root(root: Path, bdf: str) -> list:
    """[bdf, parent bridge, ..., root port]: the PCI components of the resolved sysfs path of the device, deepest first."""
    real = os.path.realpath(root / "sys/bus/pci/devices" / bdf)
    comps = [c for c in Path(real).parts if len(c) == 12 and c[4] == SEP and c[7] == SEP and c[10] == "."]
    return list(reversed(comps)) or [bdf]


def hop_info(root: Path, bdf: str) -> dict:
    d = root / "sys/bus/pci/devices" / bdf
    info = {"bdf": bdf, "speed": _read(d / "current_link_speed"), "width": _read(d / "current_link_width"),
            "max_speed": _read(d / "max_link_speed"), "aspm": {}}
    for f in ASPM_FILES:
        v = _read(d / "link" / f)
        if v is not None:
            info["aspm"][f] = v == "1"
    return info


def verdict(hops: list) -> str:
    seen = [h["aspm"].get("l1_aspm") for h in hops if "l1_aspm" in h["aspm"]]
    if not seen:
        return "UNKNOWN"
    return "L1-ON" if any(seen) else "L1-OFF"


def inspect(root: Path, bdf: str) -> dict:
    hops = [hop_info(root, b) for b in path_to_root(root, bdf)]
    return {"device": bdf, "hops": hops, "verdict": verdict(hops)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bdf")
    ap.add_argument("--root", default="/")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root)
    devs = [a.bdf] if a.bdf else display_devices(root)
    if not devs:
        print("pcie_link: no display-class AMD device found under", root / "sys/bus/pci/devices", file=sys.stderr)
        return 2
    policy = _read(root / "sys/module/pcie_aspm/parameters/policy")
    res = [inspect(root, b) for b in devs]
    if a.json:
        print(json.dumps({"policy": policy, "devices": res}, indent=1))
    else:
        print(f"pcie_aspm policy: {policy or 'unreadable'}")
        for r in res:
            print(f"{r['device']}: {r['verdict']}")
            for h in r["hops"]:
                on = ",".join(k for k, v in h["aspm"].items() if v) or ("none" if h["aspm"] else "no link/ files")
                print(f"  {h['bdf']}  link {h['speed'] or '?'} x{h['width'] or '?'} (max {h['max_speed'] or '?'})  aspm on: {on}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
