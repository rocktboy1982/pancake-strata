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

KEYS = {"SGPRs": "sgprs", "VGPRs": "vgprs", "AGPRs": "agprs", "ScratchSize [bytes/lane]": "scratch", "Occupancy [waves/SIMD]": "occupancy",
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


def analyse(src: Path, clang: str, rocm: str | None, tmp: Path, defs=(), includes=(), cflags=(), prefix=()) -> tuple:
    """-> ({kernel: row}, error text or None).  row = resources + {"instr", "sha1", "short"}; a kernel without assembly digest keeps its resources."""
    asm_out = tmp / (src.stem + ".s")
    p = subprocess.run(list(prefix) + compile_command(src, clang, rocm, asm_out, defs, includes, cflags), capture_output=True, text=True, errors="replace")
    res = parse_remarks(p.stderr)
    if p.returncode != 0 and not res:
        err = [l for l in p.stderr.splitlines() if "error" in l][:3]
        return {}, "; ".join(err) or f"clang exit {p.returncode}"
    dig = asm_digests(asm_out.read_text(errors="replace")) if asm_out.exists() else {}
    rows = {}
    for k, r in res.items():
        rows[k] = dict(r, short=short_name(k), source=str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src), **dig.get(k, {}))
    return rows, None


# ---------------------------------------------------------------------------------------------------------------------------------------------
# U1 (LEADS.md Part U): ISA census.  `--census` counts the instructions of every function and of its innermost loops by class and prints the
# issue-slot ceiling.  Offline: it reads the .s the compile wrote, or any disassembly text (`--census-asm FILE`, llvm-objdump -d or clang -S).
# Without the flag nothing below runs and the table and the digests are byte-identical.
# ---------------------------------------------------------------------------------------------------------------------------------------------
CLASSES = ("VALU_F32", "VALU_F16", "VALU_I32", "VALU_F64", "VOPD", "DOT", "WMMA", "LDS", "VMEM_LD", "VMEM_ST", "SMEM", "WAITCNT", "DELAY", "SALU",
           "BRANCH", "OTHER")
_SUFFIX = re.compile(r"_(e32|e64|dpp|sdwa)$")
_LABEL = re.compile(r"^\s*(?:[0-9a-f]{8,16}\s+)?<?([.\w$@]+)>?:\s*$")
_BLOCK = re.compile(r"^(\.?LBB\d+_\d+|\.?BB\d+_\d+|\.L\w+)$")
_NOLOOP = ("s_endpgm", "s_setpc", "s_swappc", "s_getpc", "s_call")
_TARGET = re.compile(r"([.\w$]+)\s*$")


def classify(mnemonic: str) -> str:
    """The issue class of one gfx11 instruction mnemonic (without its operands)."""
    m = _SUFFIX.sub("", mnemonic.lower())
    if m.startswith("v_dual_"):
        return "VOPD"
    if m.startswith(("v_wmma_", "v_swmmac_")):
        return "WMMA"
    if m.startswith(("v_dot", "v_sudot")):
        return "DOT"
    if m.startswith("v_"):
        if "f64" in m:
            return "VALU_F64"
        if "f32" in m:
            return "VALU_F32"
        if "f16" in m:
            return "VALU_F16"
        return "VALU_I32"
    if m.startswith("ds_"):
        return "LDS"
    if m.startswith(("global_", "buffer_", "flat_", "scratch_", "tbuffer_")):
        return "VMEM_LD" if "_load" in m else "VMEM_ST"
    if m.startswith(("s_load", "s_buffer_load")):
        return "SMEM"
    if m.startswith(("s_waitcnt", "s_wait_")):
        return "WAITCNT"
    if m == "s_delay_alu":
        return "DELAY"
    if m.startswith(("s_branch", "s_cbranch_")) or m.startswith(_NOLOOP):
        return "BRANCH"
    if m.startswith("s_"):
        return "SALU"
    return "OTHER"


def census_functions(asm: str) -> dict:
    """{function: [("label", name) | ("instr", mnemonic, operands)]} from clang -S text or llvm-objdump -d text (comments after ';' and '//' dropped)."""
    out: dict = {}
    cur = None
    for raw in asm.splitlines():
        line = raw.split("//", 1)[0].split(";", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith(".Lfunc_end"):
            cur = None
            continue
        lm = _LABEL.match(line)
        if lm:
            name = lm.group(1)
            if _BLOCK.match(name):
                if cur is not None:
                    cur.append(("label", name.lstrip(".")))
            elif not name.startswith("__hip_cuid"):
                cur = out.setdefault(name, [])
            continue
        if cur is None or line.lstrip().startswith(".") or line[:1] not in ("\t", " "):
            continue
        parts = line.split(None, 1)
        if re.fullmatch(r"[0-9a-f]{4,}:", parts[0]) and len(parts) > 1:   # an address column of some disassemblers
            parts = parts[1].split(None, 1)
        cur.append(("instr", parts[0], parts[1] if len(parts) > 1 else ""))
    return out


def count_classes(mnemonics) -> dict:
    c = {k: 0 for k in CLASSES}
    for mn in mnemonics:
        c[classify(mn)] += 1
    return c


def innermost_loops(items: list) -> tuple:
    """-> ([(label, start, end)], instrs): a backward branch to a label closes the loop [label, branch] over the instruction list; only loops that
    contain no other loop are kept (the hot inner loops)."""
    pos, instrs = {}, []
    for it in items:
        if it[0] == "label":
            pos[it[1]] = len(instrs)
        else:
            instrs.append(it)
    loops = set()
    for j, it in enumerate(instrs):
        if not it[1].startswith(("s_branch", "s_cbranch_")):
            continue
        t = _TARGET.search(it[2])
        name = t.group(1).lstrip(".") if t else None
        if name in pos and pos[name] <= j:
            loops.add((name, pos[name], j))
    keep = [l for l in loops if not any(o != l and l[1] <= o[1] and o[2] <= l[2] for o in loops)]
    return sorted(keep, key=lambda l: (l[1], l[2])), instrs


def issue_ceiling(slots: int, cus: int = 84, clock_ghz: float = 2.0, macs: float = 0.0) -> dict:
    """Wave32 issues one VALU instruction per SIMD per clock: 2 SIMD x 32 lanes = 64 lane-iterations per CU per clock per instruction.
    -> {"iter_per_s": GPU-wide lane-iterations per second at `slots` instructions per iteration, "mac_per_s": x macs (0 when unknown)}."""
    if slots <= 0:
        return {"iter_per_s": 0.0, "mac_per_s": 0.0}
    it = cus * 64 * clock_ghz * 1e9 / slots
    return {"iter_per_s": it, "mac_per_s": it * macs}


def census(asm: str, cus: int = 84, clock_ghz: float = 2.0, macs: float = 0.0, top: int = 3) -> dict:
    """{function: {"total": {class: n}, "instr": n, "loops": [{label, instr, counts, ceiling}] (largest first, at most `top`)}}."""
    res: dict = {}
    for fn, items in census_functions(asm).items():
        loops, instrs = innermost_loops(items)
        row = {"total": count_classes(i[1] for i in instrs), "instr": len(instrs), "loops": []}
        for label, a, b in loops:
            body = instrs[a:b + 1]
            row["loops"].append({"label": label, "instr": len(body), "counts": count_classes(i[1] for i in body),
                                 "ceiling": issue_ceiling(len(body), cus, clock_ghz, macs)})
        row["loops"].sort(key=lambda l: -l["instr"])
        row["loops"] = row["loops"][:top]
        res[fn] = row
    return res


def census_table(cen: dict, macs: float = 0.0) -> str:
    lines = []
    for fn, r in sorted(cen.items()):
        lines.append(f"{short_name(fn)}  ({r['instr']} instructions)")
        rows = [("function", r["total"], r["instr"], None)] + [(f"loop {l['label']}", l["counts"], l["instr"], l["ceiling"]) for l in r["loops"]]
        for tag, cnt, n, ceil in rows:
            used = " ".join(f"{k}={v}" for k, v in cnt.items() if v)
            tail = ""
            if ceil is not None:
                tail = f"  ceiling {ceil['iter_per_s'] / 1e12:.3f} T lane-iter/s"
                if macs:
                    tail += f", {n / macs:.2f} instr/MAC, {ceil['mac_per_s'] / 1e12:.2f} T MAC/s"
            lines.append(f"  {tag:<24} {n:>5}  {used}{tail}")
    return "\n".join(lines)


def repeat_check(src: Path, n: int, clang: str, rocm: str | None, tmp: Path, defs=(), includes=(), cflags=(), prefix=(), run=None) -> dict:
    """U6 (LEADS.md Part U): compile one unit `n` times and compare the per-kernel code digests.  -> {"runs": n, "differ": [kernel, ...], "errors": [...]}.
    `prefix` goes before the compiler command (e.g. `setarch x86_64 -R`: a fixed address-space layout), `run` replaces analyse() in tests.
    The byte-level two-compile check of the whole device object is box_chains/hip_repeat_check.py (S4); this one answers per kernel, N times."""
    run = run or analyse
    first, differ, errors = None, set(), []
    for i in range(n):
        d = tmp / f"rep{i}"
        d.mkdir(parents=True, exist_ok=True)
        rows, err = run(src, clang, rocm, d, defs, includes, cflags, prefix)
        if err:
            errors.append(err)
            continue
        dig = {k: r.get("sha1") for k, r in rows.items()}
        if first is None:
            first = dig
            continue
        differ |= {k for k in set(first) | set(dig) if first.get(k) != dig.get(k)}
    return {"runs": n, "differ": sorted(differ), "errors": errors}


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
    ap.add_argument("--census", action="store_true", help="U1: count the instructions of every kernel and of its inner loops by class, print the issue-slot ceiling")
    ap.add_argument("--census-asm", help="U1: census of this existing disassembly text (clang -S or llvm-objdump -d) instead of a compile; needs no compiler")
    ap.add_argument("--census-cus", type=int, default=84, help="--census: compute units (RX 7900 XT: 84)")
    ap.add_argument("--census-clock-ghz", type=float, default=2.0, help="--census: shader clock used for the ceiling")
    ap.add_argument("--census-macs", type=float, default=0.0, help="--census: MACs per loop iteration per lane, for instr/MAC and the MAC/s ceiling")
    ap.add_argument("--census-top", type=int, default=3, help="--census: inner loops listed per kernel")
    ap.add_argument("--repeat", type=int, default=0, metavar="N", help="U6: compile each source N times and compare the per-kernel digests (exit 1 when they differ)")
    ap.add_argument("--repeat-prefix", default="", help="--repeat: command put before the compiler, e.g. 'setarch x86_64 -R'")
    ap.add_argument("--tmp", default=None, help="directory for the .s files (default: a temp dir)")
    a = ap.parse_args(argv)
    if a.census_asm:
        cen = census(Path(a.census_asm).read_text(errors="replace"), a.census_cus, a.census_clock_ghz, a.census_macs, a.census_top)
        print(census_table(cen, a.census_macs))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps({"census": cen}, indent=1, sort_keys=True))
        return 0
    srcs = [ROOT / s for s in (HOT if a.hot else [])] + [Path(s) if Path(s).is_absolute() else ROOT / s for s in a.sources]
    if not srcs:
        ap.error("give sources or --hot")
    clang, rocm = default_clang()
    clang, rocm = a.clang or clang, a.rocm_path or rocm
    import tempfile
    tmp = Path(a.tmp) if a.tmp else Path(tempfile.mkdtemp(prefix="kres-"))
    tmp.mkdir(parents=True, exist_ok=True)
    if a.repeat:
        import shlex
        bad = 0
        for s in srcs:
            r = repeat_check(s, a.repeat, clang, rocm, tmp, a.define, a.include, a.cflag, shlex.split(a.repeat_prefix))
            print(f"REPEAT {s.name}: {r['runs']} compiles, {len(r['differ'])} kernel(s) differ" + (f", {len(r['errors'])} error(s)" if r["errors"] else ""))
            for k in r["differ"]:
                print("  DIFF", short_name(k))
            bad += bool(r["differ"])
        return 1 if bad else 0
    rows, errors = {}, {}
    for s in srcs:
        r, err = analyse(s, clang, rocm, tmp, a.define, a.include, a.cflag)
        if err:
            errors[str(s)] = err
            print(f"ERROR {s}: {err}", file=sys.stderr)
        rows.update(r)
    print(table(rows))
    if a.census:
        cen = {}
        for s in srcs:
            asm = tmp / (s.stem + ".s")
            if asm.exists():
                cen.update(census(asm.read_text(errors="replace"), a.census_cus, a.census_clock_ghz, a.census_macs, a.census_top))
        print(census_table(cen, a.census_macs))
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
