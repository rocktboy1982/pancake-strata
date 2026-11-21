#!/usr/bin/env python3
"""isa_default_check.py - the default path of a switch is unchanged: compile gfx1100 device code of some .cu files at a baseline revision and in the working tree
with a ROCm SDK clang (compile only, no GPU; the Windows PC's ROCm 6.2 SDK, as TEST_LEDGER.md T-854 / T-880) and compare the INSTRUCTION STREAM of every kernel.
A kernel whose name changed because it became a template (name<false>) is matched by its base name; labels are normalised.  Used for the LEADS.md Part U
leads: U2 (src/kernels/cuda/{elementwise,gdn,ple,qsa}.cu: the `F32 = false` instantiation must equal the old kernel) and U5 (src/prefill/moe_fused_iq.cu and
moe_f16_kq.cu with no ablation macro).
  python tools/pancake/isa_default_check.py --rev <baseline> [--files src/kernels/cuda/gdn.cu ...] [--clang PATH] [--ggml-include DIR ...] [--sdk-include DIR]
Exit 0 when every kernel of every file matches, 1 on a difference, 2 when the toolchain is missing.  The two `__builtin_amdgcn_fence(.., "workgroup", "local")`
calls of moe_fused_iq.cu are rewritten to the two-argument form of SDK 6.2's clang in a scratch copy (both sides alike).
Tests: python -m pytest -q tools/pancake/test_reduce_f32_model.py -k isa  (runs only with STRATA_ISA_CHECK=1: it compiles for about two minutes).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_FILES = ["src/kernels/cuda/elementwise.cu", "src/kernels/cuda/gdn.cu", "src/kernels/cuda/ple.cu", "src/kernels/cuda/qsa.cu"]
CLANG_CANDIDATES = ["C:/Program Files/AMD/ROCm/6.2/bin/clang++.exe", "/opt/rocm/llvm/bin/clang++", "/opt/rocm/core-10.0/lib/llvm/bin/clang++"]
SDK_INCLUDES = ["C:/Program Files/AMD/ROCm/6.2/include", "/opt/rocm/include"]


def find(cands):
    for c in cands:
        if c and Path(c).exists():
            return c
    return None


def kernel_streams(asm: str) -> dict:
    """{function symbol: [instruction text]} of an AMDGPU .s, labels (.LBB..) and comments normalised."""
    cur, out = None, {}
    for line in asm.splitlines():
        m = re.match(r"^(_Z\w+):", line)
        if m:
            cur = m.group(1)
            out[cur] = []
            continue
        if cur and re.match(r"^\t[a-z]", line) and not line.startswith("\t."):
            t = re.sub(r"\s*;.*", "", line.strip())
            out[cur].append(re.sub(r"\.L\w+", "L", t))
    return out


def base_name(sym: str) -> str:
    """The function's own identifier of an Itanium-mangled symbol (_ZN<len><ns>..<len><name>E<args> or _Z<len><name><args>); a template argument list is skipped."""
    i = 3 if sym.startswith("_ZN") else 2
    last = sym
    while i < len(sym) and sym[i].isdigit():
        j = i
        while j < len(sym) and sym[j].isdigit():
            j += 1
        n = int(sym[i:j])
        last = sym[j:j + n]
        i = j + n
        if i < len(sym) and sym[i] == "I":   # <template args>E: skip to the matching E
            depth = 0
            while i < len(sym):
                depth += 1 if sym[i] == "I" else (-1 if sym[i] == "E" else 0)
                i += 1
                if depth == 0:
                    break
    return last


def compare(old_asm: str, new_asm: str) -> list:
    """Names of the old kernels whose instruction stream no kernel of the same base name in the new asm reproduces."""
    old, new = kernel_streams(old_asm), kernel_streams(new_asm)
    groups = {}
    for n, v in new.items():
        groups.setdefault(base_name(n), []).append(v)
    return [n for n, v in old.items() if v not in groups.get(base_name(n), [])]


def f64_counts(asm: str) -> list:
    """[(base name, template flag 'true'/'false', instructions, f64 instructions)] of the kernels instantiated with a bool template argument (the U2 `F32` ones)."""
    out = []
    for n, v in kernel_streams(asm).items():
        if "Lb1E" in n or "Lb0E" in n:
            out.append((base_name(n), "true" if "Lb1E" in n else "false", len(v), sum(1 for t in v if "f64" in t.split()[0])))
    return out


def compile_one(clang, root: Path, rel: str, out: Path, sdk, ggml) -> str:
    src = (root / rel).read_text(encoding="utf-8", errors="replace").replace('__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local")', '__builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup")').replace(
        '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local")', '__builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup")')
    tmp = out.with_suffix(".cu")
    tmp.write_text(src, encoding="utf-8", newline="\n")
    cmd = [clang, "-x", "hip", "--offload-arch=gfx1100", "--cuda-device-only", "-S", "-O3", "-std=c++17", "-DSTRATA_USE_HIP=1", "-DSTRATA_WMMA_GFX11=1",
           "-include", str(root / "include/strata/platform/hip_compat/cuda_runtime.h"), "-I", str(root / "include/strata/platform/hip_compat"), "-I", str(root / "include"),
           "-I", str(root / "src"), "-I", str(root / Path(rel).parent), "-I", str(root / "src/kernels/cuda")]
    for g in ggml:
        cmd += ["-I", g]
    if sdk:
        cmd += ["-I", sdk]
    cmd += ["-Wno-everything", str(tmp), "-o", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{rel}: {r.stderr[:600]}")
    return out.read_text(errors="replace")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rev", required=True)
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument("--clang", default=os.environ.get("STRATA_ROCM_CLANG") or find(CLANG_CANDIDATES))
    ap.add_argument("--sdk-include", default=os.environ.get("STRATA_ROCM_INCLUDE") or find(SDK_INCLUDES))
    ap.add_argument("--ggml-include", nargs="*", default=[])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    a = ap.parse_args(argv)
    if not a.clang or not Path(a.clang).exists():
        print("no ROCm clang found (--clang / STRATA_ROCM_CLANG)", file=sys.stderr)
        return 2
    root = Path(a.root)
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base"
        subprocess.run(["git", "-C", str(root), "archive", a.rev, "--format=tar", "-o", str(Path(td) / "base.tar"), "src", "include"], check=True)
        base.mkdir()
        subprocess.run(["tar", "-xf", str(Path(td) / "base.tar"), "-C", str(base)], check=True)
        for rel in a.files:
            old = compile_one(a.clang, base, rel, Path(td) / "old.s", a.sdk_include, a.ggml_include)
            new = compile_one(a.clang, root, rel, Path(td) / "new.s", a.sdk_include, a.ggml_include)
            diff = compare(old, new)
            for name, flag, n_instr, n_f64 in f64_counts(new):
                print(f"    {name}<{flag}>: {n_instr} instructions, {n_f64} f64 instructions")
            print(f"{rel}: {len(kernel_streams(old))} kernels at {a.rev}, {len(kernel_streams(new))} now, {'IDENTICAL default kernels' if not diff else 'DIFFER: ' + ', '.join(diff[:6])}")
            bad += bool(diff)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
