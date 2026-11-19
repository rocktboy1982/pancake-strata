#!/usr/bin/env python3
"""Per-hardware recommended flags for the RDNA3 startup script (rdna3/run.py).

The measured flags of this branch (TEST_LEDGER.md of the pancake fork: T-224, T-588/T-589/T-591, T-820, T-096) are not
hard-coded in the engine: they sit in rdna3/recommended-flags/<gfx>[-rocm<M.m>][-vram<GiB>].json and the startup
script exports them before the engine starts.  The file is regenerated from measured verdicts after a benchmark:

    tools/pancake/flag_diag.sh --write-recommended         (flag_diag.py: hardware, per-flag verdicts, this module writes)

Rules (one place, tested in rdna3/test_tuned_flags.py):
  * precedence: the process environment (the user) > the recommended file > the setup's config (rdna3/configs/*.json);
    a recommended argument replaces the value of the same argument in the setup config (the config's value is a default)
    and is appended when the config has none;
  * `--no-tuned-flags` on run.py or STRATA_TUNED_FLAGS=0 gives the plain behaviour of the setup config, byte for byte;
  * only verdicts GAIN enter a written file, and only flags that are exact (bitwise parity) or lossy with a passing
    quality gate; UNSUPPORTED, NO-GAIN, OK (works, no measured gain) and FAIL never do;
  * unknown hardware (no file matches, or no GPU is found): plain defaults and one printed line saying so.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "rdna3" / "recommended-flags"
SCHEMA = 1
# a flag enters a written file only with this verdict; the bars are the standing promotion bars (MOD_PLAN section 2)
BAR_LOSSLESS = 2.0
BAR_LOSSY = 3.0


# ---------------------------------------------------------------------------------------------- hardware
def _read(path, reader):
    try:
        return reader(str(path)).strip()
    except Exception:
        return ""


def _default_reader(path):
    return Path(path).read_text()


def _default_run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def detect_hardware(read=_default_reader, run=_default_run, root="/", env=None):
    """gfx arch, ROCm version, VRAM, PCIe link, CPU, RAM, VMM and hipBLASLt presence.  Read only; every probe is optional
    (read = sysfs/proc reader, run = command runner: both injectable, the tests pass fake ones)."""
    env = os.environ if env is None else env
    hw = {"gfx": "", "rocm": "", "vram_gib": 0, "pcie": "", "cpu": "", "ram_gib": 0, "vmm": False, "hipblaslt": False}
    # the gfx arch: rocm_agent_enumerator lists gfx000 (the CPU) first, then the GPUs; HSA_OVERRIDE_GFX_VERSION is not honoured
    out = run(["rocm_agent_enumerator"]) or run(["rocminfo"])
    arches = [m for m in re.findall(r"gfx[0-9a-f]{3,4}", out) if m != "gfx000"]
    if env.get("HIP_VISIBLE_DEVICES", "").split(",")[0].isdigit() and arches:
        i = int(env["HIP_VISIBLE_DEVICES"].split(",")[0])
        arches = [arches[i]] if i < len(arches) else arches
    hw["gfx"] = arches[0] if arches else ""
    # ROCm version: engine/BUILD.json ("rocm" key, when the build wrote one), else the .info/version file of the install
    ver = ""
    try:
        ver = str(json.loads(read(ROOT / "engine" / "BUILD.json")).get("rocm", ""))
    except Exception:
        ver = ""
    if not ver:
        for p in (env.get("ROCM_PATH", "/opt/rocm") + "/.info/version", "/opt/rocm/.info/version", "/opt/rocm/core-10.0/.info/version"):
            ver = _read(p, read)
            if ver:
                break
    m = re.match(r"(\d+)\.(\d+)", ver)
    hw["rocm"] = f"{m.group(1)}.{m.group(2)}" if m else ""
    # VRAM and PCIe link of the first AMD display device (sysfs)
    base = Path(root) / "sys/class/drm"
    for card in ("card0", "card1", "card2"):
        v = _read(base / card / "device/mem_info_vram_total", read)
        if v.isdigit() and int(v) > (1 << 30):
            hw["vram_gib"] = int(round(int(v) / (1 << 30)))
            sp = _read(base / card / "device/current_link_speed", read)
            wd = _read(base / card / "device/current_link_width", read)
            hw["pcie"] = f"{sp} x{wd}".strip() if sp or wd else ""
            break
    cpu = _read(Path(root) / "proc/cpuinfo", read)
    mm = re.search(r"model name\s*:\s*(.+)", cpu)
    hw["cpu"] = mm.group(1).strip() if mm else ""
    mi = _read(Path(root) / "proc/meminfo", read)
    mt = re.search(r"MemTotal:\s*(\d+)\s*kB", mi)
    hw["ram_gib"] = int(round(int(mt.group(1)) / 1048576)) if mt else 0
    # VMM (virtual memory management API): the HIP runtime library exports hipMemCreate
    for lib in (env.get("ROCM_LIB", ""), env.get("ROCM_PATH", "") + "/lib", "/opt/rocm/lib", "/opt/rocm/core-10.0/lib"):
        so = Path(lib) / "libamdhip64.so" if lib and lib != "/lib" else None
        if so is not None and so.exists():
            hw["vmm"] = "hipMemCreate" in run(["nm", "-D", str(so)])
            hw["hipblaslt"] = any(Path(lib).glob("libhipblaslt.so*"))
            break
    return hw


# ---------------------------------------------------------------------------------------------- files
def file_key(hw):
    """The most specific file names for this hardware, most specific first (a file name never claims more than it checks)."""
    g = hw.get("gfx", "")
    if not g:
        return []
    names = []
    if hw.get("rocm"):
        names.append(f"{g}-rocm{hw['rocm']}")
    names.append(g)
    return names


def load_recommended(hw, directory=DIR):
    """(data, path) of the best matching file, or (None, None).  A file may carry key.vram_gib_min: below it, no match."""
    for n in file_key(hw):
        p = Path(directory) / f"{n}.json"
        if not p.is_file():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("schema") != SCHEMA:
            continue
        key = d.get("key", {})
        if key.get("gfx") and key["gfx"] != hw.get("gfx"):
            continue
        if key.get("rocm") and key["rocm"] != hw.get("rocm"):
            continue
        if key.get("vram_gib_min") and hw.get("vram_gib", 0) < key["vram_gib_min"]:
            continue
        return d, p
    return None, None


def tuned_enabled(argv_off=False, env=None):
    env = os.environ if env is None else env
    return not argv_off and env.get("STRATA_TUNED_FLAGS", "1") != "0"


def apply_to_config(cfg, rec, env=None, root=ROOT):
    """Overlay the recommended env and args on a filled-in server config (in place).  Returns the list of what changed
    as (kind, name, old, new).  The process environment wins; a recommended argument replaces the config's value."""
    env = os.environ if env is None else env
    changed = []
    cenv = cfg.setdefault("env", {})
    for k, v in (rec.get("env") or {}).items():
        v = str(v).replace("@STRATA@", str(root))
        if k in env:
            continue   # the user set it: theirs
        if cenv.get(k) != v:
            changed.append(("env", k, cenv.get(k), v))
            cenv[k] = v
    args = cfg.setdefault("args", [])
    for k, v in (rec.get("args") or {}).items():
        v = str(v).replace("@STRATA@", str(root))
        if k in args:
            i = args.index(k)
            old = args[i + 1] if i + 1 < len(args) and not str(args[i + 1]).startswith("--") else None
            if v == "" or old == v:
                continue
            if old is None:
                args.insert(i + 1, v)
            else:
                args[i + 1] = v
            changed.append(("arg", k, old, v))
            continue
        args.append(k)
        if v != "":
            args.append(v)
        changed.append(("arg", k, None, v))
    return changed


def apply_startup(cfg, no_tuned=False, env=None, hw=None, directory=DIR, say=None):
    """What rdna3/run.py calls.  One printed line either way (say defaults to stderr)."""
    say = say or (lambda s: print(s, file=sys.stderr))
    env = os.environ if env is None else env
    if not tuned_enabled(no_tuned, env):
        say("strata: tuned flags off (--no-tuned-flags / STRATA_TUNED_FLAGS=0): the setup's plain defaults")
        return []
    hw = hw if hw is not None else detect_hardware()
    rec, path = load_recommended(hw, directory)
    if rec is None:
        say("strata: no recommended-flags file for this hardware (gfx=%s rocm=%s vram=%s GiB): plain defaults"
            % (hw.get("gfx") or "?", hw.get("rocm") or "?", hw.get("vram_gib") or "?"))
        return []
    changed = apply_to_config(cfg, rec, env)
    say("strata: tuned flags from %s: %d set (%s); a variable you export yourself, or --no-tuned-flags, wins"
        % (Path(path).name, len(changed), ", ".join(c[1] for c in changed) or "none differ"))
    return changed


# ---------------------------------------------------------------------------------------------- writing
def eligible(entry):
    """True only for a measured GAIN that is exact, or lossy with a passing quality gate."""
    if entry.get("verdict") != "GAIN":
        return False
    if entry.get("exact") is True:
        return float(entry.get("gain_pct", 0)) >= BAR_LOSSLESS
    return bool(entry.get("gate_pass")) and float(entry.get("gain_pct", 0)) >= BAR_LOSSY


def build_recommended(hw, verdicts, base=None):
    """verdicts: {flag: {"env": {K: V} | "args": {K: V}, "verdict", "exact", "gate_pass", "gain_pct", "note"}}.  The
    shipped keys of `base` that were not measured again are kept only when `base` says "keep"; by default the file is
    exactly what the measurements back."""
    env, args, kept = {}, {}, {}
    for flag, e in sorted(verdicts.items()):
        kept[flag] = {k: e.get(k) for k in ("verdict", "exact", "gate_pass", "gain_pct", "note") if k in e}
        if eligible(e):
            env.update({k: str(v) for k, v in (e.get("env") or {}).items()})
            args.update({k: str(v) for k, v in (e.get("args") or {}).items()})
    return {"schema": SCHEMA, "key": {"gfx": hw.get("gfx", ""), "rocm": hw.get("rocm", ""), "vram_gib_min": max(1, int(hw.get("vram_gib", 0)) - 1)},
            "hardware": hw, "env": env, "args": args, "verdicts": kept}


def write_recommended(hw, verdicts, directory=DIR):
    rec = build_recommended(hw, verdicts)
    name = f"{hw['gfx']}-rocm{hw['rocm']}.json" if hw.get("rocm") else f"{hw['gfx']}.json"
    p = Path(directory) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec, indent=1, sort_keys=False) + "\n")
    return p


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="print the hardware and the recommended-flags file that would apply")
    ap.add_argument("--apply-to", help="a server config JSON: write the config with the recommended flags overlaid to --out")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    hw = detect_hardware()
    rec, path = load_recommended(hw)
    print(json.dumps({"hardware": hw, "file": str(path) if path else None}, indent=1))
    if a.apply_to:
        cfg = json.loads(Path(a.apply_to).read_text())
        if rec:
            apply_to_config(cfg, rec, env={})
        Path(a.out or a.apply_to).write_text(json.dumps(cfg, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
