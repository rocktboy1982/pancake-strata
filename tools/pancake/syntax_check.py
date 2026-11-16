#!/usr/bin/env python3
"""syntax_check.py - compile-only (-fsyntax-only) check of the C++/HIP files a push changes, with the HIP build's own flags (the command ninja would run, `ninja -t commands`).

Why: the pre-push build matrix only CONFIGURES the rows (and links nothing), so a syntax error that merges leave behind (a lost brace,
a real line break inside a string literal: T-586, T-594, T-598) reached origin and broke the daily HIP build.  This compiles just the
changed translation units through build-hip, no object files, no link.

    python3 tools/pancake/syntax_check.py [--range BASE..HEAD] [--build-dir build-hip] [FILE ...]
Exit 0: all compile (or nothing to check, or no build-hip: it says so).  Exit 1: a changed file does not compile.
Tests: python3 -I tools/pancake/test_syntax_check.py
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTS = (".cpp", ".cc", ".cu")   # headers are checked through the .cpp/.cu files that include them


def ninja_of(build):
    """The ninja the build directory was configured with (CMAKE_MAKE_PROGRAM), else the one on PATH."""
    try:
        for l in open(Path(build) / "CMakeCache.txt", encoding="utf-8", errors="replace"):
            if l.startswith("CMAKE_MAKE_PROGRAM:"):
                return l.split("=", 1)[1].strip()
    except OSError:
        pass
    return "ninja"


def syntax_command(cmd):
    """A compile command line turned into a syntax-only one (no -o file, no -MD dependency output)."""
    args = shlex.split(cmd)
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a in ("-o", "-MF", "-MT", "-MQ"):
            skip = True
        elif a not in ("-MD", "-MMD", "-MP", "-c"):
            out.append(a)
    return out + ["-fsyntax-only"]


def compile_commands(ninja, build, files):
    """{file: command} for the files the build compiles: the object target of a file ends in '<path relative to the repo>.o'."""
    r = subprocess.run([ninja, "-C", str(build), "-t", "targets", "all"], capture_output=True, text=True)
    objs = [l.split(":")[0] for l in r.stdout.splitlines() if ".dir/" in l and l.split(":")[0].endswith(".o")]
    found = {}
    for f in files:
        rel = os.path.relpath(f, ROOT)
        for o in objs:
            if o.endswith("/" + rel + ".o"):
                c = subprocess.run([ninja, "-C", str(build), "-t", "commands", o], capture_output=True, text=True).stdout.strip().splitlines()
                if c:
                    found[f] = c[-1]
                break
    return found


def changed_files(rng):
    r = subprocess.run(["git", "diff", "--name-only", "--diff-filter=AM", rng], cwd=ROOT, capture_output=True, text=True)
    return [l for l in r.stdout.split() if l.endswith(EXTS)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--range", default="")
    ap.add_argument("--build-dir", default="build-hip")
    a = ap.parse_args(argv)
    build = ROOT / a.build_dir
    if not (build / "build.ninja").exists():
        print(f"syntax_check: {build}/build.ninja not found, nothing checked")
        return 0
    files = [str((ROOT / f).resolve()) if not os.path.isabs(f) else f for f in a.files] or \
            [str((ROOT / f).resolve()) for f in (changed_files(a.range) if a.range else [])]
    entries = compile_commands(ninja_of(build), build, files)
    todo = [f for f in files if f in entries]
    if not todo:
        print("syntax_check: no changed translation unit in the HIP build")
        return 0
    bad = 0
    for f in todo:
        r = subprocess.run(syntax_command(entries[f]), cwd=build, capture_output=True, text=True)
        if r.returncode != 0:
            bad += 1
            print(f"syntax_check: FAILED {f}", file=sys.stderr)
            print("\n".join([l for l in r.stderr.splitlines() if "error" in l][:12]), file=sys.stderr)
        else:
            print(f"syntax_check: ok {f}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
