"""Measured launch flags per hardware: the file `tools/pancake/flag_diag.sh --write-recommended` writes and every start reads.

The normal start path (`serve/server_parts/http_api.py: apply_saved_tuning`, run by every start: the Terminal, `runtime start`,
the app's load) calls `apply` before the saved-preset overlay of strata_control/presets.py.  `apply` puts the environment
settings and engine arguments of the profile that matches this machine (graphics architecture, ROCm version, VRAM) on top of the
run configuration, with these rules:

* an explicit user setting always wins: a variable already in the configuration's `env` or in the process environment, an
  argument already in the configuration's `args`, is left alone;
* `STRATA_TUNED_FLAGS=0` in the environment, or `--no-tuned-flags` on the command line, gives plain upstream behaviour;
* only a flag with a measured GAIN enters a profile: exact (bitwise parity) and >= 2 % end to end, or lossy with a passing
  quality gate and >= 3 %.  `accepted` is the one rule; a record that fails it is dropped when the file is read, so a hand-edited
  file cannot smuggle an UNSUPPORTED or NO-GAIN flag in;
* an unknown machine (no profile matches, or no hardware could be read) keeps the plain defaults and says so in one line.

File format (`strata.tuned-flags/1`), shipped default `terminal/data/tuned-flags.json`, user file `$STRATA_TUNED_FLAGS_FILE` or
`~/.strata/tuned-flags.json` (read first):

    {"schema": "strata.tuned-flags/1", "profiles": [
       {"match": {"gfx": "gfx1100", "rocm": ["10.0", "7."], "vram_gib": [16, 24]}, "source": "...",
        "flags": [{"kind": "env", "name": "STRATA_PF_F16", "value": "1", "verdict": "GAIN", "exact": false,
                   "gain_pct": 15.1, "quality_gate": "PASS", "ledger": "T-820"}, ...]}]}

`rocm` entries are version prefixes (an empty or missing list matches any); `vram_gib` is [min, max] of the largest card.
A value may contain `<ROOT>`: the folder that holds the shipped files (the checkout / install root).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

SCHEMA = "strata.tuned-flags/1"
EXACT_MIN_GAIN = 2.0
LOSSY_MIN_GAIN = 3.0
ROOT = Path(__file__).resolve().parents[2]
SHIPPED = Path(__file__).resolve().parents[1] / "data" / "tuned-flags.json"


def accepted(flag: dict[str, Any]) -> bool:
    """The entry rule: a measured GAIN that is exact (>= 2 %) or lossy with a passing quality gate (>= 3 %)."""
    if flag.get("verdict") != "GAIN" or flag.get("kind") not in ("env", "arg") or not flag.get("name"):
        return False
    try:
        gain = float(flag.get("gain_pct"))
    except (TypeError, ValueError):
        return False
    if flag.get("exact") is True:
        return gain >= EXACT_MIN_GAIN
    return flag.get("quality_gate") == "PASS" and gain >= LOSSY_MIN_GAIN


# --------------------------------------------------------------------------- hardware

Runner = Callable[[list[str]], str]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


def detect_hardware(root: str = "/", runner: Runner | None = None) -> dict[str, Any] | None:
    """gfx architecture, ROCm version, VRAM of the largest card, PCIe link, CPU, RAM; None when no AMD card can be read.

    Read-only: rocminfo, /opt/rocm*/.info/version, /sys/class/drm, /proc.  `root` redirects the file reads (tests); `runner`
    replaces the rocminfo call (tests)."""
    run = runner or _run
    base = Path(root)
    gfx = ""
    for m in re.finditer(r"Name:\s+(gfx[0-9a-f]+)", run(["rocminfo"])):
        gfx = m.group(1)
        break
    rocm = ""
    for cand in sorted((base / "opt").glob("rocm*/.info/version")) + sorted((base / "opt").glob("rocm*/core-*/.info/version")):
        v = _read(cand)
        if v:
            rocm = v.split("-")[0]
            break
    if not rocm:
        m = re.search(r"HIP version:\s*([0-9.]+)", run(["hipconfig", "--version"]) or "")
        rocm = m.group(1) if m else ""
    vram, link = 0, ""
    for dev in sorted((base / "sys/class/drm").glob("card[0-9]*/device")):
        if _read(dev / "vendor") != "0x1002":
            continue
        try:
            total = int(_read(dev / "mem_info_vram_total") or 0)
        except ValueError:
            total = 0
        if total > vram:
            vram = total
            speed, width = _read(dev / "current_link_speed"), _read(dev / "current_link_width")
            link = f"{speed} x{width}" if speed or width else ""
    cpu = ""
    for line in _read(base / "proc/cpuinfo").splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    ram_kib = 0
    m = re.search(r"MemTotal:\s+(\d+) kB", _read(base / "proc/meminfo"))
    if m:
        ram_kib = int(m.group(1))
    if not gfx:
        return None
    return {"gfx": gfx, "rocm": rocm, "vram_gib": round(vram / 2**30, 1), "pcie_link": link, "cpu": cpu,
            "ram_gib": round(ram_kib / 2**20, 1)}


def matches(match: dict[str, Any], hw: dict[str, Any]) -> bool:
    if match.get("gfx") and match["gfx"] != hw.get("gfx"):
        return False
    prefixes = match.get("rocm") or []
    if prefixes and not any(str(hw.get("rocm") or "").startswith(p) for p in prefixes):
        return False
    rng = match.get("vram_gib")
    if rng:
        lo, hi = rng
        if not (lo <= float(hw.get("vram_gib") or 0) <= hi):
            return False
    return True


# --------------------------------------------------------------------------- files

def candidate_files(env: dict[str, str] | None = None) -> list[Path]:
    env = os.environ if env is None else env
    user = env.get("STRATA_TUNED_FLAGS_FILE")
    return [Path(user).expanduser() if user else Path.home() / ".strata" / "tuned-flags.json", SHIPPED]


def load_profiles(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return []
    out = []
    for prof in data.get("profiles") or []:
        if isinstance(prof, dict) and isinstance(prof.get("match"), dict):
            flags = [f for f in prof.get("flags") or [] if isinstance(f, dict) and accepted(f)]
            out.append({**prof, "flags": flags})
    return out


def select(hw: dict[str, Any], files: list[Path]) -> tuple[dict[str, Any] | None, Path | None]:
    for path in files:
        for prof in load_profiles(path):
            if matches(prof["match"], hw):
                return prof, path
    return None, None


# --------------------------------------------------------------------------- applying

def disabled(env: dict[str, str] | None = None, argv: list[str] | None = None) -> bool:
    env = os.environ if env is None else env
    argv = sys.argv if argv is None else argv
    return env.get("STRATA_TUNED_FLAGS") == "0" or "--no-tuned-flags" in argv


def _arg_present(args: list[str], name: str) -> bool:
    return any(a == name or a.startswith(name + "=") for a in args)


def apply(cfg: dict[str, Any], *, env: dict[str, str] | None = None, argv: list[str] | None = None,
          hw: dict[str, Any] | None = None, files: list[Path] | None = None, say: Callable[[str], None] | None = None,
          root: str = "/") -> dict[str, Any]:
    """`cfg` with the matching profile's flags added where the user set nothing.  Never raises, never measures."""
    say = say or (lambda line: print(line, flush=True))
    penv = os.environ if env is None else env
    if disabled(penv, argv):
        return cfg
    try:
        hw = hw if hw is not None else detect_hardware(root)
        if hw is None:
            say("[strata] tuned flags: no supported graphics card read here, plain defaults")
            return cfg
        prof, path = select(hw, files if files is not None else candidate_files(penv))
        if prof is None:
            say(f"[strata] tuned flags: no measured profile for {hw['gfx']} / ROCm {hw['rocm'] or '?'} / "
                f"{hw['vram_gib']} GiB, plain defaults")
            return cfg
        out = json.loads(json.dumps(cfg))
        out_env, out_args = dict(out.get("env") or {}), [str(a) for a in out.get("args") or []]
        added: list[str] = []
        for f in prof["flags"]:
            value = str(f.get("value", "")).replace("<ROOT>", str(ROOT))
            if f["kind"] == "env":
                if f["name"] in out_env or f["name"] in penv:
                    continue
                out_env[f["name"]] = value
                added.append(f"{f['name']}={value}")
            else:
                if _arg_present(out_args, f["name"]):
                    continue
                out_args += [f["name"]] if value == "" else [f["name"], value]
                added.append(f"{f['name']} {value}".strip())
        out["env"], out["args"] = out_env, out_args
        say(f"[strata] tuned flags for {hw['gfx']} ({Path(path).name}): " + (", ".join(added) or "all already set by you")
            + "; STRATA_TUNED_FLAGS=0 or --no-tuned-flags turns this off")
        return out
    except Exception as e:  # noqa: BLE001 - a start never fails because of this file
        say(f"[strata] tuned flags could not be applied ({type(e).__name__}), plain defaults")
        return cfg
