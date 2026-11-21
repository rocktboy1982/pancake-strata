#!/usr/bin/env python3
"""hip_repeat_check.py - LEADS.md Part S S4: is the HIP code generation repeatable, and how many VOPD pairs does it form?

Takes the compile_commands.json of a configured HIP build (hipflags-build.v1.sh always exports it), picks HIP translation
units, and for each one compiles the device side (gfx1100 code object, --offload-device-only) TWICE with identical flags and
compares the bytes (sha256).  Optionally (--variant-flags "-mllvm -amdgpu-enable-vopd=false") it compiles a third time with the
flags added and reports the number of v_dual_* instructions (VOPD) in the default and the variant disassembly.
Stdlib only.  Never run on the Windows PC (no ROCm); a box job: it runs clang, so the box must be idle (hipflags-repeat.v1.sh).

  hip_repeat_check.py --compile-commands DIR/compile_commands.json --out OUTDIR [--match REGEX] [--limit N]
                      [--clang PATH] [--objdump PATH] [--variant-flags "..."] [--timeout S]
Exit 0 = every unit repeatable, 1 = at least one unit differs between two identical compiles, 2 = error (nothing is deleted).
Writes OUTDIR/repeat-report.json and prints one line per unit: REPEAT <ok|DIFF> <vopd_default> [<vopd_variant>] <file>.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys

DROP_WITH_ARG = {"-o", "-MF", "-MT", "-MQ"}
DROP_FLAG = {"-MD", "-MMD", "-MP", "-c", "--offload-device-only", "--offload-host-only"}


def split_command(entry):
    if "arguments" in entry:
        return list(entry["arguments"])
    return shlex.split(entry["command"])


def device_command(entry, out, extra, clang=None):
    """The unit's own command line with the output redirected to a device-only gfx code object."""
    args = split_command(entry)
    keep = [args[0]]
    i = 1
    while i < len(args):
        a = args[i]
        if a in DROP_WITH_ARG:
            i += 2
            continue
        if a in DROP_FLAG:
            i += 1
            continue
        keep.append(a)
        i += 1
    if clang:
        keep[0] = clang
    return keep + list(extra) + ["--offload-device-only", "-c", "-o", out]


def is_hip_unit(entry):
    cmd = entry.get("command") or " ".join(entry.get("arguments", []))
    hip = "--offload-arch" in cmd or "-x hip" in cmd or "amdgpu" in cmd or "--cuda-gpu-arch" in cmd
    return hip and entry["file"].endswith((".cu", ".hip", ".cpp"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(cmd, cwd, timeout):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode, p.stdout.decode("utf-8", "replace")


def vopd_count(objdump, obj, timeout):
    """Count v_dual_* instructions in the disassembly (None when no objdump is given)."""
    if not objdump:
        return None
    p = subprocess.run([objdump, "-d", obj], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=timeout)
    if p.returncode != 0:
        return None
    return len(re.findall(r"^\s*v_dual_", p.stdout.decode("utf-8", "replace"), flags=re.M))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--compile-commands", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--match", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--clang", default="")
    ap.add_argument("--objdump", default="")
    ap.add_argument("--variant-flags", default="")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args(argv)
    try:
        with open(a.compile_commands) as f:
            entries = json.load(f)
    except (OSError, ValueError) as e:
        print("ERROR cannot read %s: %s" % (a.compile_commands, e), file=sys.stderr)
        return 2
    rx = re.compile(a.match) if a.match else None
    units = [e for e in entries if is_hip_unit(e) and (rx is None or rx.search(e["file"]))]
    if a.limit > 0:
        units = units[:a.limit]
    if not units:
        print("ERROR no HIP translation unit matched", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    variant = shlex.split(a.variant_flags)
    report, differ = [], 0
    for n, e in enumerate(units):
        cwd = e.get("directory", ".")
        base = os.path.join(a.out, "u%03d" % n)
        o1, o2, o3 = base + ".a1.o", base + ".a2.o", base + ".v.o"
        row = {"file": e["file"], "unit": "u%03d" % n}
        jobs = [(o1, []), (o2, [])] + ([(o3, variant)] if variant else [])
        for out, extra in jobs:
            rc, text = run(device_command(e, out, extra, a.clang or None), cwd, a.timeout)
            if rc != 0:
                last = text.strip().splitlines()[-1:] or [""]
                print("ERROR compile %s exit %d: %s" % (e["file"], rc, last[0]), file=sys.stderr)
                return 2
        row["sha_a1"], row["sha_a2"] = sha(o1), sha(o2)
        row["repeatable"] = row["sha_a1"] == row["sha_a2"]
        row["vopd_default"] = vopd_count(a.objdump, o1, a.timeout)
        if variant:
            row["sha_variant"] = sha(o3)
            row["vopd_variant"] = vopd_count(a.objdump, o3, a.timeout)
        differ += 0 if row["repeatable"] else 1
        report.append(row)
        print("REPEAT %s %s%s %s" % ("ok" if row["repeatable"] else "DIFF", row["vopd_default"],
                                     (" " + str(row["vopd_variant"])) if variant else "", e["file"]))
    with open(os.path.join(a.out, "repeat-report.json"), "w") as f:
        json.dump({"units": report, "differ": differ, "variant_flags": a.variant_flags}, f, indent=1)
    print("REPEATDONE units=%d differ=%d" % (len(report), differ))
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
