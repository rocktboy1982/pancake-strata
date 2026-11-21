#!/usr/bin/env python3
"""kernel_resources.py - per-kernel GPU resource table and machine-code digest for gfx1100, from the compiler alone (no GPU, no engine).

LEADS.md Part O O1 / O2 (idea from caminus' register-budget gate and its ".text sections must stay unchanged" rule; AGPL: ideas only, this is our own
code).  Two questions the repo could not answer without a box run until now:

  1. Did a change make a hot kernel spill, use scratch, use more LDS or lose occupancy?   (the resource table, `--check`)
  2. Is the code of every kernel the change did NOT mean to touch the same instructions as before, i.e. is the default path byte-identical?
     (the digest column, `--check --digest`; AGENTS.md "new features are opt-in; the unset/default path must stay byte-identical, and say how
     you checked")

One compile per source: `clang++ -x hip --offload-arch=gfx1100 --offload-device-only -S` with `-Rpass-analysis=kernel-resource-usage`.  The
remarks give SGPRs, VGPRs, scratch bytes per lane, spills, LDS and waves per SIMD per kernel; the assembly gives an instruction count and a digest
(comments, label numbers and the module id lines removed, so a change in ANOTHER function does not move this one's digest).

    python tools/pancake/kernel_resources.py SRC.cu [SRC.cu ...] [--out FILE.json] [--update BASELINE.json] [--check BASELINE.json [--digest] [--strict]]
    python tools/pancake/kernel_resources.py --hot --out bench/results/<day>-kernel-resources/hot.json     the default hot-kernel set

--check exits 1 when a kernel gained spills or scratch, gained LDS, lost occupancy, or (with --digest) changed code; --strict also fails on a VGPR/SGPR
count change.  A compile is not a hardware result: the table says what the compiler allocated, never how fast the kernel runs.
Windows: the ROCm 6.2 SDK clang (the flags of TEST_LEDGER.md T-773); Linux box: `--clang /opt/rocm/.../bin/clang++ --rocm-path /opt/rocm/...`.
Tests: python -m pytest -q tools/pancake/test_kernel_resources.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WIN_ROCM = Path("C:/Program Files/AMD/ROCm/6.2")

# the hot decode / prompt kernel sources (MOD_PLAN.md section 1: combine_cpu, wait_flag, rdna_kq, expdec_gu, the QSA attention, the F16 prompt GEMMs)
HOT = ["src/kernels/cuda/native_moe.cu", "src/kernels/cuda/verify_kernels.cu", "src/kernels/cuda/rdna_gr.cu", "src/kernels/cuda/rdna_mmvq.cu", "src/kernels/cuda/native_mmvq.cu",
       "src/kernels/cuda/qsa_decode_attn.cu", "src/kernels/cuda/qsa_decode_attn_wmma.cu", "src/kernels/cuda/native_gdn.cu",
       "src/kernels/cuda/fused_gr.cu", "src/prefill/moe_f16_kq.cu", "src/prefill/moe_fused_iq.cu"]

KEYS = {"SGPRs": "sgprs", "TotalSGPRs": "sgprs", "VGPRs": "vgprs", "AGPRs": "agprs", "ScratchSize [bytes/lane]": "scratch", "Occupancy [waves/SIMD]": "occupancy",
        "SGPRs Spill": "spills_s", "VGPRs Spill": "spills_v", "LDS Size [bytes/block]": "lds"}
REMARK = re.compile(r"remark:\s+(Function Name|[A-Za-z][A-Za-z \[\]/]*?):\s*(\S.*?)\s*\[-Rpass-analysis=kernel-resource-usage\]")


def parse_remarks(text: str) -> dict:
    """{mangled kernel name: {sgprs, vgprs, scratch, occupancy, spills_s, spills_v, lds}} from the compiler's kernel-resource-usage remarks."""
    out: dict = {}
    cur = None
    for line in text.splitlines():
        m = REMARK.search(line)
        if not m:
            continue
        key, val = m.group(1), m.group(2)
        if key == "Function Name":
            cur = out.setdefault(val, {})
        elif cur is not None and key in KEYS:
            try:
                cur[KEYS[key]] = int(val)
            except ValueError:
                pass
    return out


_LBB = re.compile(r"\.LBB\d+_(\d+)")
_FEND = re.compile(r"\.Lfunc_end\d+")


def asm_digests(asm: str) -> dict:
    """{function name: {"instr": instruction count, "sha1": digest of its normalised text}} from AMDGPU assembly.

    Normalised: text after ';' dropped (comments), blank lines dropped, `.LBB<fn>_<bb>` -> `.LBB_<bb>` and `.Lfunc_end<n>` -> `.Lfunc_end`, so a
    change in another function (which renumbers the module's labels) leaves this function's digest alone."""
    out: dict = {}
    name = None
    body: list = []
    for raw in asm.splitlines():
        line = raw.split(";", 1)[0].rstrip()
        if not line.strip():
            continue
        if name is None:
            m = re.match(r"^(_Z[\w$.]*|[A-Za-z_][\w$.]*):$", line)
            if m and not m.group(1).startswith((".L", "__hip_cuid")):
                name, body = m.group(1), []
            continue
        if line.startswith(".Lfunc_end"):
            instr = sum(1 for b in body if b[:1] in ("\t", " ") and b.strip()[:1] not in (".", ""))
            out[name] = {"instr": instr, "sha1": hashlib.sha1("\n".join(body).encode()).hexdigest()[:16]}
            name = None
            continue
        line = _FEND.sub(".Lfunc_end", _LBB.sub(r".LBB_\1", line))
        body.append(line)
    return out


def short_name(mangled: str) -> str:
    """A readable name from an Itanium-mangled kernel name without a demangler: the last length-prefixed identifiers, joined by '::'."""
    parts = []
    i = 2 if mangled.startswith("_Z") else 0
    while i < len(mangled):
        m = re.compile(r"\d+").match(mangled, i)
        if not m:
            i += 1
            continue
        n = int(m.group(0))
        ident = mangled[m.end():m.end() + n]
        if n == 0 or len(ident) < n:
            i = m.end()
            continue
        parts.append(ident)
        i = m.end() + n
        if i >= len(mangled) or not mangled[i].isdigit():   # 'I' (template arguments), 'E' (end of the name) or a parameter type: stop
            break
    parts = [p for p in parts if p != "_GLOBAL__N_1"]
    return "::".join(parts[-2:]) if parts else mangled


def compile_command(src: Path, clang: str, rocm: str | None, asm_out: Path, defs=(), includes=(), cflags=()) -> list:
    inc = ROOT / "include"
    compat = inc / "strata" / "platform" / "hip_compat"
    cmd = [clang, "-x", "hip", "--offload-arch=gfx1100", "--offload-device-only"]
    if rocm:
        cmd.append(f"--rocm-path={rocm}")
    cmd += ["-std=c++20", "-O3", f"-I{compat}", f"-I{inc}", "-include", str(compat / "cuda_runtime.h"), "-DSTRATA_USE_HIP", "-DSTRATA_WMMA_GFX11",
            "-DSTRATA_VERIFY_MAX_T=16", "-D_CRT_SECURE_NO_WARNINGS", "-Wno-everything", "-Rpass-analysis=kernel-resource-usage"]
    cmd += [f"-D{d}" for d in defs] + [f"-I{i}" for i in includes] + list(cflags)
    cmd += ["-S", "-o", str(asm_out), str(src)]
    return cmd


def analyse(src: Path, clang: str, rocm: str | None, tmp: Path, defs=(), includes=(), cflags=()) -> tuple:
    """-> ({kernel: row}, error text or None).  row = resources + {"instr", "sha1", "short"}; a kernel without assembly digest keeps its resources."""
    asm_out = tmp / (src.stem + ".s")
    p = subprocess.run(compile_command(src, clang, rocm, asm_out, defs, includes, cflags), capture_output=True, text=True, errors="replace")
    res = parse_remarks(p.stderr)
    if p.returncode != 0 and not res:
        err = [l for l in p.stderr.splitlines() if "error" in l][:3]
        return {}, "; ".join(err) or f"clang exit {p.returncode}"
    dig = asm_digests(asm_out.read_text(errors="replace")) if asm_out.exists() else {}
    rows = {}
    for k, r in res.items():
        rows[k] = dict(r, short=short_name(k), source=str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src), **dig.get(k, {}))
    return rows, None


REGRESS = ("spills_v", "spills_s", "scratch", "lds")


def check(cur: dict, base: dict, use_digest=False, strict=False) -> tuple:
    """-> (regressions, notes).  A regression: more spills, scratch or LDS than the baseline, fewer waves per SIMD, a changed digest (use_digest), a
    changed VGPR/SGPR count (strict).  A kernel in only one side is a note, never a regression (kernels come and go)."""
    bad, notes = [], []
    for k, b in sorted(base.items()):
        c = cur.get(k)
        nm = b.get("short", k)
        if c is None:
            notes.append(f"gone: {nm}")
            continue
        for f in REGRESS:
            if c.get(f, 0) > b.get(f, 0):
                bad.append(f"{nm}: {f} {b.get(f, 0)} -> {c.get(f, 0)}")
        if c.get("occupancy") is not None and b.get("occupancy") is not None and c["occupancy"] < b["occupancy"]:
            bad.append(f"{nm}: occupancy {b['occupancy']} -> {c['occupancy']} waves/SIMD")
        for f in ("vgprs", "sgprs"):
            if c.get(f) != b.get(f) and c.get(f) is not None and b.get(f) is not None:
                (bad if strict else notes).append(f"{nm}: {f} {b[f]} -> {c[f]}")
        if use_digest and b.get("sha1") and c.get("sha1") and b["sha1"] != c["sha1"]:
            bad.append(f"{nm}: code changed (instr {b.get('instr')} -> {c.get('instr')}, digest {b['sha1']} -> {c['sha1']})")
    for k in sorted(set(cur) - set(base)):
        notes.append(f"new: {cur[k].get('short', k)}")
    return bad, notes


def table(rows: dict) -> str:
    head = f"{'kernel':58} {'VGPR':>4} {'SGPR':>4} {'scr':>4} {'spV':>4} {'spS':>4} {'LDS':>6} {'occ':>3} {'instr':>6}"
    lines = [head]
    for k, r in sorted(rows.items(), key=lambda kv: (kv[1].get("source", ""), kv[1].get("short", kv[0]))):
        lines.append(f"{r.get('short', k)[:58]:58} {r.get('vgprs', 0):4} {r.get('sgprs', 0):4} {r.get('scratch', 0):4} {r.get('spills_v', 0):4} "
                     f"{r.get('spills_s', 0):4} {r.get('lds', 0):6} {r.get('occupancy', 0):3} {r.get('instr', 0):6}")
    return "\n".join(lines)


def default_clang() -> tuple:
    if os.name == "nt" and (WIN_ROCM / "bin" / "clang++.exe").exists():
        return str(WIN_ROCM / "bin" / "clang++.exe"), str(WIN_ROCM)
    return "hipcc", None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("sources", nargs="*")
    ap.add_argument("--hot", action="store_true", help="the default hot-kernel source set")
    ap.add_argument("--clang")
    ap.add_argument("--rocm-path")
    ap.add_argument("--define", action="append", default=[], help="extra -D (repeatable)")
    ap.add_argument("--cflag", action="append", default=[], help="extra compiler flag, written --cflag=-O1 (repeatable; later flags win, so -O1 overrides the default -O3)")
    ap.add_argument("--include", action="append", default=[], help="extra -I (repeatable): iq_kernels.cu needs the ggml include directory, which only the box has")
    ap.add_argument("--out", help="write the table as JSON")
    ap.add_argument("--update", help="write/overwrite this baseline JSON")
    ap.add_argument("--check", help="compare with this baseline JSON")
    ap.add_argument("--digest", action="store_true", help="--check: a changed code digest is a regression (default-path identity)")
    ap.add_argument("--strict", action="store_true", help="--check: a changed VGPR/SGPR count is a regression")
    ap.add_argument("--tmp", default=None, help="directory for the .s files (default: a temp dir)")
    a = ap.parse_args(argv)
    srcs = [ROOT / s for s in (HOT if a.hot else [])] + [Path(s) if Path(s).is_absolute() else ROOT / s for s in a.sources]
    if not srcs:
        ap.error("give sources or --hot")
    clang, rocm = default_clang()
    clang, rocm = a.clang or clang, a.rocm_path or rocm
    import tempfile
    tmp = Path(a.tmp) if a.tmp else Path(tempfile.mkdtemp(prefix="kres-"))
    tmp.mkdir(parents=True, exist_ok=True)
    rows, errors = {}, {}
    for s in srcs:
        r, err = analyse(s, clang, rocm, tmp, a.define, a.include, a.cflag)
        if err:
            errors[str(s)] = err
            print(f"ERROR {s}: {err}", file=sys.stderr)
        rows.update(r)
    print(table(rows))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"kernels": rows, "errors": errors, "clang": clang}, indent=1, sort_keys=True))
    if a.update:
        Path(a.update).write_text(json.dumps({"kernels": rows}, indent=1, sort_keys=True))
    rc = 1 if errors and not rows else 0
    if a.check:
        base = json.loads(Path(a.check).read_text())["kernels"]
        bad, notes = check(rows, base, a.digest, a.strict)
        for n in notes:
            print("note:", n)
        for b in bad:
            print("REGRESSION:", b)
        print(f"check: {len(bad)} regression(s), {len(notes)} note(s)")
        rc = 1 if bad else rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
