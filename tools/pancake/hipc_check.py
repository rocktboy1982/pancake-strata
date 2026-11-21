#!/usr/bin/env python3
"""hipc_check.py - compile-only gfx1100 checks with the NEWEST ROCm SDK found on this PC (TEST_LEDGER.md T-nnn, MOD_PLAN.md compile-only note).

Picks the newest SDK among the candidate directories and runs kernel_resources.py (VGPR / spill / LDS table, ISA digest) with that compiler:

    python tools/pancake/hipc_check.py --list                      every SDK found, newest first, the first one is the one chosen
    python tools/pancake/hipc_check.py SRC.cu [SRC.cu ...] [kernel_resources.py options]      e.g. --include <dir> --define X --out F.json
    HIPC_SDK=C:/Program Files/AMD/ROCm/6.2 python tools/pancake/hipc_check.py ...    force one SDK (a ROCm root, or the pip _rocm_sdk_core dir)

Candidates (the first needs no admin): the pip layout `~/rocm-sdk-*/venv/Lib/site-packages/_rocm_sdk_core` (pip install of `rocm[devel]` from
https://stable.repo.amd.com/rocm/whl-next/, version from the sibling rocm_sdk_core-X.dist-info), `C:/Program Files/AMD/ROCm/*` (HIP SDK installer,
version from the directory name), `/opt/rocm*` (box).  HIP_PATH of the environment is overridden for the child so the 6.2 headers never leak into a
10.x compile.  Tests: python -m pytest -q tools/pancake/test_hipc_check.py
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple, Optional

HERE = Path(__file__).resolve().parent
HOME = Path.home()
DEFAULT_PATTERNS = [
    str(HOME / "rocm-sdk-*" / "venv" / "Lib" / "site-packages" / "_rocm_sdk_core"),
    str(HOME / "rocm-sdk-*" / "venv" / "lib" / "python*" / "site-packages" / "_rocm_sdk_core"),
    "C:/Program Files/AMD/ROCm/*",
    "/opt/rocm*",
]


class Sdk(NamedTuple):
    root: Path
    clang: Path
    devlib: Optional[Path]   # --rocm-device-lib-path, only needed for the pip layout
    version: tuple


def _exe(p: Path) -> Optional[Path]:
    for n in ("clang++.exe", "clang++"):
        if (p / n).is_file():
            return p / n
    return None


def _version(root: Path) -> tuple:
    """(major, minor, patch): sibling rocm_sdk_core-X.dist-info (pip), else the last dotted number of the path, else (0,)."""
    if root.parent.is_dir():
        for d in root.parent.glob("rocm_sdk_core-*.dist-info"):
            m = re.match(r"rocm_sdk_core-(\d+(?:\.\d+)*)", d.name)
            if m:
                return tuple(int(x) for x in m.group(1).split("."))
    m = re.findall(r"(\d+(?:\.\d+)+)", root.name)
    return tuple(int(x) for x in m[-1].split(".")) if m else (0,)


def probe(root) -> Optional[Sdk]:
    """-> Sdk when `root` holds a clang++ (pip layout lib/llvm/bin, or HIP SDK / box layout bin or llvm/bin), else None."""
    root = Path(root)
    pip = _exe(root / "lib" / "llvm" / "bin")
    if pip and (root / "lib" / "llvm" / "amdgcn" / "bitcode").is_dir():
        return Sdk(root, pip, root / "lib" / "llvm" / "amdgcn" / "bitcode", _version(root))
    for sub in (("bin",), ("llvm", "bin")):
        c = _exe(root.joinpath(*sub))
        if c:
            return Sdk(root, c, None, _version(root))
    return None


def find_sdks(patterns=None, env=None) -> list:
    """Every usable SDK, newest first.  HIPC_SDK in env wins and is the only entry."""
    env = os.environ if env is None else env
    forced = env.get("HIPC_SDK")
    if forced:
        s = probe(forced)
        if s is None:
            raise SystemExit(f"HIPC_SDK={forced}: no clang++ found there")
        return [s]
    seen, out = set(), []
    for pat in (DEFAULT_PATTERNS if patterns is None else patterns):
        for d in sorted(glob.glob(pat)):
            key = str(Path(d).resolve())
            if key in seen:
                continue
            seen.add(key)
            s = probe(d)
            if s:
                out.append(s)
    return sorted(out, key=lambda s: s.version, reverse=True)


def pick(patterns=None, env=None) -> Sdk:
    sdks = find_sdks(patterns, env)
    if not sdks:
        raise SystemExit("no ROCm SDK found (set HIPC_SDK, or pip install 'rocm[devel]' into ~/rocm-sdk-10x/venv)")
    return sdks[0]


def child_env(sdk: Sdk, env=None) -> dict:
    e = dict(os.environ if env is None else env)
    e["HIP_PATH"] = e["ROCM_PATH"] = str(sdk.root)
    e["HIP_CLANG_PATH"] = e["HIP_DEVICE_LIB_PATH"] = ""
    return e


def command(sdk: Sdk, args: list) -> list:
    cmd = [sys.executable, "-I", str(HERE / "kernel_resources.py"), "--clang", str(sdk.clang), "--rocm-path", str(sdk.root)]
    if sdk.devlib:
        cmd.append(f"--cflag=--rocm-device-lib-path={sdk.devlib}")
    return cmd + list(args)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    sdks = find_sdks()
    if "--list" in argv:
        for i, s in enumerate(sdks):
            print(("* " if i == 0 else "  ") + ".".join(map(str, s.version)).ljust(10), s.clang)
        return 0 if sdks else 1
    if not sdks:
        print("no ROCm SDK found (set HIPC_SDK, or pip install 'rocm[devel]' into ~/rocm-sdk-10x/venv)", file=sys.stderr)
        return 2
    sdk = sdks[0]
    ver = subprocess.run([str(sdk.clang), "--version"], capture_output=True, text=True, env=child_env(sdk)).stdout.splitlines()
    print(f"hipc_check: SDK {'.'.join(map(str, sdk.version))} at {sdk.root}: {ver[0] if ver else '?'}", file=sys.stderr)
    return subprocess.call(command(sdk, argv), env=child_env(sdk))


if __name__ == "__main__":
    sys.exit(main())
