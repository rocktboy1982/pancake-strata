#!/usr/bin/env python3
"""Prove that the kernels of a .cu file that a change did NOT mean to touch compile to the same machine code.

    python tools/pancake/kernel_sass_unchanged.py --ref origin/lead/Q3-Q7-Q4 --file src/kernels/cuda/s2_expert_grouped.cu \
        [--arch sm_75] [--allow-new group_resident_ptr_kernel]

It compiles the file as of `--ref` (git show) and the working copy to cubins with nvcc (needs nvcc, and on Windows the MSVC
environment) and compares the SASS (opcodes and encodings) of every kernel; the symbol's file-hash prefix is ignored.  A kernel that
exists only in the new build must match one of the `--allow-new` name fragments.  Exit 0 = every old kernel is identical.
Used for TEST_LEDGER.md T-1047 (STRATA_MTP_SLOTS: moe_group_resident, the default drafter's kernel, must not change).
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile


def sass(nvcc, arch, src, include, tmp, tag):
    cubin = os.path.join(tmp, tag + ".cubin")
    cmd = [nvcc, "-cubin", "-arch=" + arch, "-std=c++17", "-diag-suppress", "20281", "-I" + include, "-o", cubin, src]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = subprocess.run(["cuobjdump", "-sass", cubin], check=True, capture_output=True, text=True).stdout
    funcs, cur = {}, None
    for line in out.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = re.sub(r"\d+_GLOBAL__N__[0-9a-f]+_\d+_\w+?_cu_[0-9a-f]+\d*", "NS", m.group(1))
            funcs[cur] = []
        elif cur:
            funcs[cur].append(re.sub(r"\s+", " ", re.sub(r"/\*[0-9a-f]{4}\*/", "", line)).strip())
    return funcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--arch", default="sm_75")
    ap.add_argument("--nvcc", default="nvcc")
    ap.add_argument("--include", default="include")
    ap.add_argument("--allow-new", action="append", default=[])
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        old_src = os.path.join(tmp, "old_" + os.path.basename(a.file))
        with open(old_src, "wb") as f:
            f.write(subprocess.run(["git", "show", a.ref + ":" + a.file], check=True, capture_output=True).stdout)
        old = sass(a.nvcc, a.arch, old_src, a.include, tmp, "old")
        new = sass(a.nvcc, a.arch, a.file, a.include, tmp, "new")
    changed = [k for k in old if old[k] != new.get(k)]
    added = [k for k in new if k not in old]
    unexpected = [k for k in added if not any(frag in k for frag in a.allow_new)]
    print("kernels: %d before, %d after; changed: %d; new: %d (unexpected: %d)" % (len(old), len(new), len(changed), len(added), len(unexpected)))
    for k in changed:
        print("CHANGED", k)
    for k in unexpected:
        print("UNEXPECTED NEW", k)
    return 1 if changed or unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
