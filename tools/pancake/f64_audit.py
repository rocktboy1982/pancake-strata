#!/usr/bin/env python3
"""f64_audit.py - static census of double precision in DEVICE code (LEADS.md Part U, lead U2, step 1; the idea is the radiance-rdna3 finding that gfx11 has no
fast FP64, their tanh-in-double epilogue cost 2.19x on one GEMM).  It reads src/**/*.cu, *.cuh, *.hip (and include/**/*.cuh) and lists, with file:line, every
hit of
  double      the type (accumulators, casts `(double)`, `double` locals and arrays)
  libm        tanh( exp( log( pow( sqrt( sin( cos( erf( (a double-argument call is a double call unless the f suffix or a __f intrinsic is used: the tool cannot see the
              argument type, so every unsuffixed call in device code is listed and marked `maybe`; `expf(` `tanhf(` `__expf(` are f32 and are not listed)
  rn-double   __dadd_rn __dmul_rn __ddiv_rn __dsqrt_rn __dfma_rn __double2float_rn
  f64 literal a floating literal with an `.0`-style double suffix-less form is NOT listed (too many false hits); `1.0` against floats is a cast, not an f64 op.
only inside code the tool judges to be device code: a function body whose signature has `__global__` or `__device__` (lambdas and nested blocks inherit),
or a whole `.cu` kernel region guarded `#if STRATA_F16_DEVICE`.  It does NOT know whether the kernel runs on the served path (LEADS.md U.12): that is the
dispatch census (lead U3).  `loop` marks a hit inside a for/while of the function (the cost multiplies); `kernel` names the enclosing function.
  python tools/pancake/f64_audit.py [--root .] [--paths src include] [--json] [--min-score N] [--candidates]
Exit 0 always (it is a report).  Tests: python -m pytest -q tools/pancake/test_f64_audit.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

EXTS = {".cu", ".cuh", ".hip", ".inc"}
LIBM = ("tanh", "exp", "log", "pow", "sqrt", "sin", "cos", "erf", "log1p", "exp2", "rsqrt", "cbrt", "fabs", "floor", "ceil", "fmod")
# unsuffixed libm names that are NOT interesting (exact in f64 and f32 alike, or integer-valued): fabs floor ceil fmod
LIBM_COST = ("tanh", "exp", "log", "pow", "sqrt", "sin", "cos", "erf", "log1p", "exp2", "rsqrt", "cbrt")
RN_DOUBLE = ("__dadd_rn", "__dmul_rn", "__ddiv_rn", "__dsqrt_rn", "__dfma_rn", "__double2float_rn", "__drcp_rn")
CANDIDATES = {
    "elementwise.cu": ("silu_kernel",),
    "gdn.cu": ("gdn_l2_kernel", "gdn_out_norm_kernel"),
    "ple.cu": ("gnorm_kernel", "gate_kernel", "bf16_gemv_kernel", "block_sum"),
    "qsa.cu": ("indexer_key_append_kernel",),
}
WEIGHT = {"double": 1, "libm": 10, "rn-double": 2}

RE_DOUBLE = re.compile(r"\bdouble\b")
RE_LIBM = re.compile(r"(?<![\w.])(?:std::|::)?(" + "|".join(LIBM_COST) + r")\s*\(")
RE_RN = re.compile(r"\b(" + "|".join(RN_DOUBLE) + r")\b")
RE_DEVICE_SIG = re.compile(r"\b(__global__|__device__)\b")


def strip(text: str) -> str:
    """Comments and string / char literals blanked (newlines kept, so line numbers hold)."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def scan_text(text: str, path: str = "") -> list:
    """Hits [{file, line, kind, what, kernel, loop, text}] in device code of one file."""
    src = strip(text)
    lines = src.split("\n")
    orig = text.split("\n")
    stack = []          # frames (is_device, name, is_loop) of the open braces
    hits = []
    sig = ""            # text since the last ';' (outside parentheses), '{' or '}'
    paren = 0
    device_if = 0       # inside `#if STRATA_F16_DEVICE` style regions
    for li, line in enumerate(lines):
        line_no = li + 1
        stripped = line.strip()
        if re.match(r"#\s*if\b.*(STRATA_F16_DEVICE|__HIP_DEVICE_COMPILE__|__CUDA_ARCH__)", stripped):
            device_if += 1
        elif re.match(r"#\s*(endif)\b", stripped) and device_if:
            device_if -= 1
        frames_seen = list(stack)   # every frame open at some point of this line (a one-line function opens and closes its frame here)
        for ch in line:
            if ch == "{":
                s = sig.strip()
                is_dev = bool(RE_DEVICE_SIG.search(s)) and "(" in s and not re.search(r"\bnamespace\b", s)
                name = ""
                if is_dev:
                    m = re.findall(r"([A-Za-z_]\w*)\s*\(", s)
                    name = m[0] if m else ""
                    m2 = re.search(r"([A-Za-z_]\w*)\s*\([^()]*\)\s*(?:const\s*)?$", s)
                    name = m2.group(1) if m2 else name
                is_loop = bool(re.match(r"(for|while)\b", s)) or (s.endswith(")") and bool(re.search(r"(^|[;{}\s])(for|while)\s*\(", s)))
                frame = (is_dev, name, is_loop)
                stack.append(frame)
                frames_seen.append(frame)
                sig = ""
                paren = 0
            elif ch == "}":
                if stack:
                    stack.pop()
                sig = ""
                paren = 0
            elif ch == ";" and paren <= 0:
                sig = ""
            else:
                if ch == "(":
                    paren += 1
                elif ch == ")":
                    paren -= 1
                sig += ch
        sig += "\n"
        dev_frames = [f for f in frames_seen if f[0]]
        if (dev_frames or device_if > 0) and not stripped.startswith("#"):
            kernel = dev_frames[-1][1] if dev_frames else ""
            loop = any(f[2] for f in frames_seen) or bool(re.match(r"\s*(for|while)\b", line))
            found = [("double", m.group(0)) for m in RE_DOUBLE.finditer(line)]
            found += [("libm", m.group(1) + "(") for m in RE_LIBM.finditer(line)]
            found += [("rn-double", m.group(1)) for m in RE_RN.finditer(line)]
            for kind, what in found:
                hits.append({"file": path, "line": line_no, "kind": kind, "what": what, "kernel": kernel or "(guarded region)", "loop": loop,
                             "text": orig[li].strip()[:140]})
    return hits


def scan_paths(root: Path, paths) -> list:
    hits = []
    for p in paths:
        base = (root / p)
        if not base.exists():
            continue
        for f in sorted(base.rglob("*")):
            if f.suffix.lower() in EXTS or f.name.endswith(".hip.cpp"):
                try:
                    text = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                hits.extend(scan_text(text, f.relative_to(root).as_posix()))
    return hits


def summarize(hits: list) -> list:
    """Per (file, kernel): hit counts and a score (libm 10, rn-double 2, double 1, each x3 inside a loop)."""
    agg = {}
    for h in hits:
        k = (h["file"], h["kernel"])
        a = agg.setdefault(k, {"file": h["file"], "kernel": h["kernel"], "double": 0, "libm": 0, "rn-double": 0, "loop_hits": 0, "score": 0, "first_line": h["line"]})
        a[h["kind"]] += 1
        a["score"] += WEIGHT[h["kind"]] * (3 if h["loop"] else 1)
        a["loop_hits"] += 1 if h["loop"] else 0
        a["first_line"] = min(a["first_line"], h["line"])
    out = sorted(agg.values(), key=lambda a: (-a["score"], a["file"], a["first_line"]))
    for a in out:
        cand = CANDIDATES.get(Path(a["file"]).name, ())
        a["u2_candidate"] = a["kernel"] in cand
    return out


def format_text(hits: list, summary: list, min_score: int = 0, candidates: bool = False) -> str:
    out = ["f64_audit: %d hits in %d device functions (static reading; whether a kernel is on the served path is NOT known here)" % (len(hits), len(summary)), ""]
    out.append("%-52s %-34s %5s %5s %5s %5s %6s %s" % ("file:first_line", "kernel", "dbl", "libm", "rn", "loop", "score", "U2"))
    for a in summary:
        if a["score"] < min_score or (candidates and not a["u2_candidate"]):
            continue
        out.append("%-52s %-34s %5d %5d %5d %5d %6d %s" % ("%s:%d" % (a["file"], a["first_line"]), a["kernel"][:34], a["double"], a["libm"], a["rn-double"], a["loop_hits"],
                                                           a["score"], "candidate" if a["u2_candidate"] else ""))
    out.append("")
    out.append("hits (file:line  kind  what  [loop]  kernel):")
    keep = {(a["file"], a["kernel"]) for a in summary if a["score"] >= min_score and (not candidates or a["u2_candidate"])}
    for h in hits:
        if (h["file"], h["kernel"]) in keep:
            out.append("%s:%d  %s  %s  %s  %s" % (h["file"], h["line"], h["kind"], h["what"], "[loop]" if h["loop"] else "", h["kernel"]))
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--paths", nargs="*", default=["src", "include"])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--min-score", type=int, default=0)
    ap.add_argument("--candidates", action="store_true", help="only the kernels LEADS.md U2 names")
    a = ap.parse_args(argv)
    hits = scan_paths(Path(a.root), a.paths)
    summary = summarize(hits)
    if a.json:
        print(json.dumps({"hits": hits, "summary": summary}, indent=1))
    else:
        print(format_text(hits, summary, a.min_score, a.candidates))
    return 0


if __name__ == "__main__":
    sys.exit(main())
