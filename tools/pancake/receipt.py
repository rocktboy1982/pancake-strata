#!/usr/bin/env python3
"""Run receipt for an A/B (LEADS.md Part P lead L9; method idea from hipfire's "Don't claim a perf win without": record
prompt md5, binary md5, host load, a log line that the lever ran).  Our own implementation, no code copied.

One JSON next to the abtest jsonl (`<out>.receipt.json`): the git head and the paths that differ from it, the sha256 of
the spec, of every arm's engine binary and of every case's prompt, the STRATA_* environment of the arms, and the
machine state at the start (load, free RAM, the GPU's clock and power state read from sysfs).  A result file then says
which bytes produced it, and a clock stuck in a low DPM state or a power cap shows next to the numbers (hipfire: the
first run after idle is 3-7x slower, the first arm of a cell reads 1-6% slow).

    python tools/pancake/receipt.py SPEC.json --out bench/results/<day>-<name>.jsonl [--cases short,edit]

Reads only; a missing file or sysfs path is recorded as null, never an error.  Standalone, not wired into abtest.py: run it beside the A/B (abtest.py had another agent's uncommitted edits).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path) -> dict | None:
    """{"path", "sha256", "bytes"} of a file, None when it cannot be read."""
    try:
        h, n = hashlib.sha256(), 0
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
                n += len(blk)
        return {"path": str(path), "sha256": h.hexdigest(), "bytes": n}
    except OSError:
        return None


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read(path) -> str | None:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return None


def git_state(root: Path = ROOT, cap: int = 40) -> dict:
    def run(*a):
        try:
            r = subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True, timeout=20)
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    dirty = (run("status", "--porcelain", "--untracked-files=no") or "").splitlines()
    return {"head": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_count": len(dirty), "dirty": [d.strip() for d in dirty[:cap]]}


def gpu_state(card: str | None = None, drm_root: str = "/sys/class/drm") -> dict:
    """The card's clocks and power state from sysfs (amdgpu).  `card` defaults to $P18_DRM_CARD, else the first card with a
    pp_dpm_sclk file.  The active DPM line is the one ending in '*'."""
    card = card or os.environ.get("P18_DRM_CARD")
    if card is None:
        try:
            for name in sorted(os.listdir(drm_root)):
                if name.startswith("card") and "-" not in name and os.path.exists(f"{drm_root}/{name}/device/pp_dpm_sclk"):
                    card = name
                    break
        except OSError:
            pass
    if card is None:
        return {"card": None}
    base = f"{drm_root}/{card}/device"

    def active(path):
        txt = _read(path)
        if txt is None:
            return None
        for ln in txt.splitlines():
            if ln.rstrip().endswith("*"):
                return ln.strip()
        return txt.splitlines()[-1].strip() if txt else None

    out = {"card": card, "sclk": active(base + "/pp_dpm_sclk"), "mclk": active(base + "/pp_dpm_mclk"),
           "perf_level": _read(base + "/power_dpm_force_performance_level"),
           "vram_used": _read(base + "/mem_info_vram_used"), "vram_total": _read(base + "/mem_info_vram_total")}
    try:   # hwmon: power cap, average power, edge temperature (microwatts, millidegrees)
        hw = Path(base) / "hwmon"
        for d in sorted(hw.iterdir())[:1]:
            out["power_cap_uw"] = _read(d / "power1_cap")
            out["power_avg_uw"] = _read(d / "power1_average")
            out["temp_edge_mc"] = _read(d / "temp1_input")
    except OSError:
        pass
    return out


def host_state() -> dict:
    out = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        out["loadavg"] = [round(x, 2) for x in os.getloadavg()]
    except (OSError, AttributeError):
        out["loadavg"] = None
    mi = _read("/proc/meminfo")
    if mi:
        for ln in mi.splitlines():
            if ln.startswith("MemAvailable:"):
                out["mem_available_kb"] = int(ln.split()[1])
    return out


def arm_summary(spec: dict, base_exe: str | None = None, root: Path = ROOT) -> dict:
    """Per arm: its engine binary (the arm's `exe`, else `base_exe`), the sha256 of it, and its env / args."""
    arms = {}
    for name, arm in (spec.get("arms") or {}).items():
        exe = arm.get("exe") or base_exe
        if exe and not os.path.isabs(exe):
            exe = str(root / exe)
        arms[name] = {"exe": file_sha256(exe) if exe else None, "env": arm.get("env") or {}, "args": arm.get("args") or {}}
    return arms


def build_receipt(spec_path, spec: dict | None = None, prompts: dict | None = None, argv: list | None = None,
                  base_exe: str | None = None, root: Path = ROOT) -> dict:
    spec = spec if spec is not None else json.load(open(spec_path))
    return {
        "receipt": 1,
        "argv": list(argv) if argv is not None else None,
        "git": git_state(root),
        "spec": file_sha256(spec_path),
        "base": spec.get("base"),
        "cases": {c: {"sha256": text_sha256(t), "chars": len(t)} for c, t in (prompts or {}).items()},
        "arms": arm_summary(spec, base_exe, root),
        "strata_env": {k: v for k, v in sorted(os.environ.items()) if k.startswith("STRATA_")},
        "gpu_start": gpu_state(),
        "host_start": host_state(),
    }


def receipt_path(out_jsonl) -> Path:
    p = Path(out_jsonl)
    return p.with_name(p.name + ".receipt.json")


def write_receipt(out_jsonl, spec_path, **kw) -> Path:
    path = receipt_path(out_jsonl)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_receipt(spec_path, **kw), indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("spec")
    ap.add_argument("--out", required=True, help="the abtest jsonl this receipt belongs to")
    ap.add_argument("--cases", default="", help="comma list: also hash these abtest prompt cases (needs abtest importable)")
    a = ap.parse_args(argv)
    prompts = None
    if a.cases:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import abtest
        prompts = abtest.prompts([c for c in a.cases.split(",") if c])
    p = write_receipt(a.out, a.spec, prompts=prompts, argv=sys.argv)
    print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
