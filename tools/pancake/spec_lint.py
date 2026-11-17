#!/usr/bin/env python3
"""Validate bench/specs/*.json arm references against this repo's engine.

Since:
- What: future spec maintenance must not silently invert an arm's env/flag to something our engine never reads.
- Why: the 2026-11-12 spec audit caught `STRATA_MTP_DRAFT_ALL` armed after our cherry-picks, despite our engine not recognising it (inert arm, engine ignores it = baseline).
- How: scan every bench/specs/*.json, extract every arm's env STRATA_* keys and args --flags, and verify each is recognised by this repo (defined in conversation_disk_env.def, read in src/include, or parsed as a CLI flag in generate.cpp/serve).

Does:
- walks bench/specs/*.json;
- for each arm, checks each STRATA_* env key and each --flag reference against this repo's sources;
- prints NON-PHANTOM wrnings for anything not found (ambiguous cache miss, not a delete);
- exit code: 1 if any phantom found, 0 otherwise, 2 on usage/read errors.

Interface:
```
python tools/pancake/spec_lint.py [--dir bench/specs] [--quiet]
```
Requires `git` on PATH for stale reference checks; exitable no-phantom on empty env.

Reversible:
- read-only diagnostics; deletes no repo state and can be rerun any time.

Caveats:
- a PHANTOM finding is a bisection prompt: a not-found env key may mean "our engine reads it from an unindexed path" (i.e. correct the spec, not the finding);
- acceptable but with real, the intended ink (see NAME), so nothing is written.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Where a STRATA_* key or --flag is truly intended to be found.
ENV_DEF = ROOT / "src" / "core" / "conversation_disk_env.def"
CLI_DIRS = [ROOT / "src" / "program", ROOT / "serve"]


def git_ls() -> list[str]:
    """Files containing sources for the grep. This is the second source of truth."""
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "src", "include", "serve"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        return []
    return [os.path.join(str(ROOT), p) for p in out.stdout.splitlines()]


_TEXT_CACHE: dict = {}


def read_text(path) -> str:
    key = str(path)
    if key not in _TEXT_CACHE:
        try:
            _TEXT_CACHE[key] = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            _TEXT_CACHE[key] = ""
    return _TEXT_CACHE[key]


def check_arm(spec: Path, arm: str, key: str, kind: str, sources: list[str], quiet: bool) -> bool:
    """Return True if `key` is recognised by this repo (not a phantom)."""
    if kind == "env":
        if key == "STRATA_GGML_DIR":
            return True
        if re.search(re.escape(key), read_text(ENV_DEF)):
            return True
    needle = key.lstrip("-") if kind == "args" else key
    for src in sources:
        if needle in read_text(src):
            return True
    return False


def walk_specs(spec_dir: Path) -> list[tuple[Path, str, dict]]:
    for spec_file in sorted(spec_dir.glob("*.json")):
        try:
            data = json.loads(spec_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for arm, body in (data.get("arms") or {}).items():
            yield spec_file, arm, body


def effective_args(preset_args: list, arm_args: dict) -> list:
    """The preset's argv with an arm's `args` applied the way abtest.py does it: a key replaces the flag (and its value),
    "" is a bare switch, null removes."""
    a = [str(x) for x in preset_args]
    for k, v in (arm_args or {}).items():
        if k in a:
            i = a.index(k)
            n = 2 if i + 1 < len(a) and not a[i + 1].startswith("--") else 1
            del a[i:i + n]
        if v is not None:
            a += [k] + ([str(v)] if v != "" else [])
    return a


def flag_value(a: list, flag: str):
    """The value after `flag` in argv `a` (None: the flag is absent; "": a bare switch)."""
    if flag not in a:
        return None
    i = a.index(flag)
    return a[i + 1] if i + 1 < len(a) and not a[i + 1].startswith("--") else ""


def inert_gate(preset_args: list, arm: dict) -> str | None:
    """A reason when the arm turns on a switch that the engine silently leaves off under the arm's own other flags.
    Rule 1 (T-391 found it late): `--kv-grow` / STRATA_KV_GROW=1 needs the whole K/V in VRAM, and the engine says
    "--kv-grow is off" under `--kv-resident N` (generate.cpp, `o.kv_resident <= 0`); an arm that adds it to the daily
    preset (which carries --kv-resident 32768) measures nothing."""
    a = effective_args(preset_args, arm.get("args"))
    env = arm.get("env") or {}
    ev = env.get("STRATA_KV_GROW")
    asked = (ev not in (None, "", "0")) if "STRATA_KV_GROW" in env else ("--kv-grow" in a and "--no-kv-grow" not in a)
    res = flag_value(a, "--kv-resident")
    if asked and res not in (None, "", "0"):
        return f"--kv-grow is off under --kv-resident {res} (the elastic K/V needs the whole K/V in VRAM): the arm is the base"
    return None


def preset_argv(spec_file: Path):
    """The `args`-style argv list of the spec's base preset, or None when it cannot be read."""
    try:
        data = json.loads(spec_file.read_text(encoding="utf-8"))
        base = data.get("base")
        if not base:
            return None
        pp = Path(base) if os.path.isabs(base) else ROOT / base
        preset = json.loads(pp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    for key in ("args", "argv", "engine_args"):
        if isinstance(preset.get(key), list):
            return preset[key]
    return preset if isinstance(preset, list) else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ROOT / "bench" / "specs"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    spec_dir = Path(args.dir)
    if not spec_dir.is_dir():
        print(f"spec_lint: not a directory: {spec_dir}", file=sys.stderr)
        return 2

    sources = git_ls()
    if not sources:
        print("spec_lint: no sources (run inside the repo with git ls-files working)", file=sys.stderr)
        return 2

    missing = 0
    inert = 0
    presets: dict = {}
    for spec_file, arm, body in walk_specs(spec_dir):
        if spec_file not in presets:
            presets[spec_file] = preset_argv(spec_file)
        if presets[spec_file] is not None:
            why = inert_gate(presets[spec_file], body)
            if why:
                inert += 1
                print(f"INERT-GATE   {spec_file.name} arm({arm}): {why}")
        env = body.get("env") or {}
        flags = body.get("args") or {}
        for key in env:
            if not key.startswith("STRATA_"):
                continue
            if not check_arm(spec_file, arm, key, "env", sources, args.quiet):
                missing += 1
                print(f"PHANTOM-ENV  {spec_file.name} arm({arm}) {key}: not in disk_env.def or src/include")
        for flag in flags:
            if not flag.startswith("--"):
                continue
            if not check_arm(spec_file, arm, flag, "args", sources, args.quiet):
                missing += 1
                print(f"PHANTOM-FLAG {spec_file.name} arm({arm}) {flag}: not parsed in generate.cpp/serve")
    if inert and not args.quiet:
        print(f"spec_lint: {inert} arm(s) turn on a switch the engine leaves off under the arm's own flags (warning only)",
              file=sys.stderr)
    if missing:
        if not args.quiet:
            print(f"spec_lint: {missing} phantom arm reference(s) found (see above)", file=sys.stderr)
        return 1
    if not args.quiet:
        print("spec_lint: all arm env/flag references resolve against this repo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
