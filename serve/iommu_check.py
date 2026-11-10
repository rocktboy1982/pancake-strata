"""serve/iommu_check.py - one start-up note when the GPU sits behind a TRANSLATED IOMMU (MOD_PLAN A53).

With the IOMMU in translated mode (the default of many distributions: kernel log "iommu: Default domain type: Translated",
group type "DMA" or "DMA-FQ") every DMA of the GPU - expert uploads, reads of pinned host memory over PCIe - goes through
the IOMMU's page tables; passthrough mode (kernel command line `iommu=pt`, group type "identity") skips the translation
(Fast & Safe IO Memory Protection, SOSP'24, describes the cost of the translated modes).  The engine does not care which
mode it is in and nothing here changes it: this module reads sysfs and the kernel command line (cheap, no root) and
names the state, so a slow PCIe path is not a silent surprise.

  state()   -> {"mode": "passthrough" | "translated" | "off" | "unknown", "devices": [(bdf, group type)], "cmdline": [...]}
  note()    -> one sentence for the start-up log when the mode is "translated", else None
  STRATA_IOMMU_CHECK=0 silences the note.  Linux only (any other system: "unknown", no note).

How the mode is read, in order:
  1. /sys/class/iommu has no entries: "off" (no IOMMU is active: no translation, nothing to say);
  2. the display-class PCI devices (class 0x03xxxx) that are not the APU's own graphics: the type file of their IOMMU
     group, /sys/bus/pci/devices/<bdf>/iommu_group/type: "identity" is passthrough, "DMA" and "DMA-FQ" (lazy flush) are
     translated; "unmanaged" (VFIO / a VM owns the group) is "unknown";
  3. no such device found: the kernel command line (`iommu=pt` / `iommu.passthrough=1` -> passthrough, `iommu=off`,
     `amd_iommu=off`, `intel_iommu=off` -> off), else "unknown".
"""
from __future__ import annotations

import os
from pathlib import Path

APU_GPU_IDS = {"0x1636", "0x1638", "0x15e7"}   # Renoir / Cezanne / Barcelo: the APU's own graphics (as tools/pancake/os_tune.sh)
TRANSLATED = {"DMA", "DMA-FQ"}


def _read(p: Path) -> str | None:
    try:
        return p.read_text().strip()
    except OSError:
        return None


def cmdline_tokens(path: str = "/proc/cmdline") -> list[str]:
    raw = _read(Path(path)) or ""
    return raw.split()


def _mode_from_cmdline(tokens: list[str]) -> str | None:
    off = {"iommu=off", "amd_iommu=off", "intel_iommu=off", "iommu=soft"}
    if any(t in off for t in tokens):
        return "off"
    if "iommu=pt" in tokens or "iommu.passthrough=1" in tokens:
        return "passthrough"
    return None


def gpu_devices(sysfs: str = "/sys") -> list[tuple[str, str | None]]:
    """(bdf, IOMMU group type or None) of every display-class PCI device except the APU's own graphics."""
    out = []
    root = Path(sysfs) / "bus/pci/devices"
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return out
    for bdf in names:
        dev = root / bdf
        cls = _read(dev / "class") or ""
        if not cls.startswith("0x03"):
            continue
        if (_read(dev / "device") or "") in APU_GPU_IDS:
            continue
        out.append((bdf, _read(dev / "iommu_group/type")))
    return out


def state(sysfs: str = "/sys", cmdline: str = "/proc/cmdline") -> dict:
    tokens = cmdline_tokens(cmdline)
    res = {"mode": "unknown", "devices": [], "cmdline": [t for t in tokens if "iommu" in t]}
    if not sys_is_linux():
        return res
    try:
        active = [e for e in os.listdir(Path(sysfs) / "class/iommu")]
    except OSError:
        active = None
    if active is not None and not active:
        res["mode"] = "off"
        return res
    devs = gpu_devices(sysfs)
    res["devices"] = devs
    types = [t for _, t in devs if t]
    if types:
        if any(t in TRANSLATED for t in types):
            res["mode"] = "translated"
        elif all(t == "identity" for t in types):
            res["mode"] = "passthrough"
        return res
    res["mode"] = _mode_from_cmdline(tokens) or ("unknown" if active is not None else "off")
    return res


def sys_is_linux() -> bool:
    import sys
    return sys.platform.startswith("linux")


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return env.get("STRATA_IOMMU_CHECK", "1") != "0"


def note(env=None, sysfs: str = "/sys", cmdline: str = "/proc/cmdline") -> str | None:
    """The start-up sentence, only for the translated mode."""
    if not enabled(env):
        return None
    st = state(sysfs, cmdline)
    if st["mode"] != "translated":
        return None
    bad = ", ".join(f"{b} ({t})" for b, t in st["devices"] if t in TRANSLATED)
    return (f"[strata] note: the IOMMU is in translated mode for the GPU {bad}: its DMA (expert uploads, reads of pinned host "
            "memory) is address-translated, which can cost PCIe throughput; kernel command line 'iommu=pt' (passthrough) "
            "avoids it - add it to the boot loader's options and reboot; do not use it on a machine that passes GPUs to "
            "virtual machines. Only a note: nothing is changed (STRATA_IOMMU_CHECK=0 hides it)")


if __name__ == "__main__":
    import json
    print(json.dumps(state(), indent=1))
    print(note() or "no note")
