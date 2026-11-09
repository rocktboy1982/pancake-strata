#!/usr/bin/env python3
"""tools/pancake/check_stubs.py - header parity of the no-backend stub files (MOD_PLAN item 95b).

Some builds do not have a backend library and link hand-written "not available" answers instead:

  src/prefill/moe_mmq_stub.cpp    mmq::   (moe_mmq.hpp)                     without strata_mmq (STRATA_PREFILL_MMQ)
  src/prefill/moe_fused_stub.cpp  fused:: (moe_fused.hpp, moe_fused_iq.hpp) without the fused experts (STRATA_PREFILL_FUSED)
  src/prefill/moe_ws_stub.cpp     ws::    (moe_ws.hpp)                      without the weight-stationary experts (STRATA_PREFILL_WS)
  src/kernels/cuda/rdna_stub.cpp  rdna_*  (fused_gr.hpp, native_mmvq.hpp)   CUDA and gfx906 builds (rdna_gr.cu / rdna_mmvq.cu are HIP only)

A header that gains a function without its stub breaks only the build that has no backend, which nobody builds daily
(2026-11-09: fused::want_dm_f16 / dm_f16_used were missing from the stub for weeks).  This script needs no compiler and
runs in seconds.  It fails when

  1. a stub defines a function the headers do not declare with the same return and parameter types (signature drift);
  2. a header function is needed but has no stub: it is called (qualified, e.g. `fused::want_dm_f16(`) from code that is
     built without the backend - every src/ and include/ file except the backend's own sources, tests, parity checks
     and benches, and except the lines inside `#if defined(<backend macro>)` - or, for the RDNA stubs, it is defined
     by the HIP implementation the stub replaces.

Usage: tools/pancake/check_stubs.py [--verbose] [--selftest]      exit 0 ok, 1 a check failed, 2 usage / parse error
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


@dataclass
class Group:
    name: str
    stub: str
    headers: list[str]
    ns: str                      # the innermost namespace name used to qualify calls (mmq, fused, ws, kernels)
    present: list[str]           # macros that mean "the real implementation is in this build"
    impl: list[str] = field(default_factory=list)   # implementation files (excluded from the consumer scan)
    impl_re: str = ""            # more implementation files, by regex on the repo-relative path
    name_re: str = ""            # only header functions whose name matches (the rdna_ subset of fused_gr.hpp)
    qualified: bool = True       # consumers call ns::name(; False: bare name( (unique rdna_* names)
    mirror_impl: bool = False    # every header function the implementation defines needs a stub


GROUPS = [
    Group("mmq", "src/prefill/moe_mmq_stub.cpp", ["include/strata/prefill/moe_mmq.hpp"], "mmq",
          ["STRATA_PREFILL_MMQ"], impl_re=r"^src/prefill/(moe_|mmq_|ggml_cuda_host)"),
    Group("fused", "src/prefill/moe_fused_stub.cpp",
          ["include/strata/prefill/moe_fused.hpp", "include/strata/prefill/moe_fused_iq.hpp"], "fused",
          ["STRATA_PREFILL_FUSED", "STRATA_PREFILL_MMQ"], impl_re=r"^src/prefill/(moe_|mmq_|ggml_cuda_host)"),
    Group("ws", "src/prefill/moe_ws_stub.cpp", ["include/strata/prefill/moe_ws.hpp"], "ws",
          ["STRATA_PREFILL_WS"], impl_re=r"^src/prefill/(moe_|mmq_|ggml_cuda_host)"),
    Group("rdna", "src/kernels/cuda/rdna_stub.cpp",
          ["include/strata/kernels/fused_gr.hpp", "include/strata/kernels/native_mmvq.hpp"], "kernels", [],
          impl=["src/kernels/cuda/rdna_gr.cu", "src/kernels/cuda/rdna_mmvq.cu"], name_re=r"^rdna_",
          qualified=False, mirror_impl=True),
]

CONSUMER_EXT = {".cpp", ".cc", ".cu", ".hpp", ".h", ".cuh", ".inc"}
SKIP_FILE_RE = re.compile(r"(_test|_parity|_bench|_harness)[^/]*$|(^|/)tests?/")
BUILTIN = {"unsigned", "signed", "long", "short", "int", "char", "float", "double", "bool", "void", "const",
           "volatile", "size_t", "int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t",
           "uint64_t"}
DROP_RET = {"inline", "static", "extern", "constexpr", "[[nodiscard]]", "STRATA_API"}
TYPE_START = ("struct ", "class ", "enum ", "union ", "using ", "typedef ", "template", "static_assert", "namespace ",
              "friend ", "extern \"C\"")


def strip_comments(s: str) -> str:
    """Comments and string literals out (newlines kept, so line numbers stay)."""
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if s.startswith("//", i):
            j = s.find("\n", i)
            i = n if j < 0 else j
        elif s.startswith("/*", i):
            j = s.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * s.count("\n", i, j))
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and s[j] != c:
                j += 2 if s[j] == "\\" else 1
            out.append(c + c)
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def drop_preprocessor(s: str) -> str:
    lines, cont = [], False
    for ln in s.split("\n"):
        is_pp = cont or ln.lstrip().startswith("#")
        cont = is_pp and ln.rstrip().endswith("\\")
        lines.append("" if is_pp else ln)
    return "\n".join(lines)


def tokens(s: str) -> list[str]:
    return re.findall(r"\[\[\w+\]\]|[A-Za-z_]\w*|::|\.\.\.|\S", s)


def norm_type(toks: list[str]) -> str:
    out = ""
    for t in toks:
        if out and re.match(r"\w", t) and re.match(r".*\w$", out):
            out += " "
        out += t
    return out.replace("std::size_t", "size_t")


def norm_param(p: str) -> str:
    depth, cut = 0, len(p)
    for i, c in enumerate(p):          # the default value goes
        if c in "(<[{":
            depth += 1
        elif c in ")>]}":
            depth -= 1
        elif c == "=" and depth == 0:
            cut = i
            break
    t = tokens(p[:cut])
    idents = [x for x in t if re.match(r"[A-Za-z_]", x)]
    if len(t) >= 2 and re.match(r"[A-Za-z_]\w*$", t[-1]) and t[-1] not in BUILTIN and len(idents) >= 2 \
            and t[-2] != "::":
        t = t[:-1]                     # the parameter's name goes
    return norm_type(t)


def split_params(s: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for c in s:
        if c in "(<[{":
            depth += 1
        elif c in ")>]}":
            depth -= 1
        if c == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += c
    if cur.strip():
        parts.append(cur)
    ps = [norm_param(p) for p in parts]
    return [] if ps == ["void"] else ps


@dataclass(frozen=True)
class Func:
    ret: str
    name: str                # Class::name for members defined in a stub
    params: tuple[str, ...]
    line: int

    def sig(self) -> str:
        return f"{self.ret} {self.name}({', '.join(self.params)})".strip()


DECL_RE = re.compile(r"^(?P<ret>.*?)(?P<name>(?:\w+::)?~?\w+)\s*\((?P<params>.*)\)\s*(?:const\s*)?(?:noexcept\s*)?"
                     r"(?:override\s*)?$", re.S)


def parse_decl(text: str, line: int) -> Func | None:
    t = " ".join(text.split())
    if not t or t.startswith(TYPE_START) or "operator" in t:
        return None
    m = DECL_RE.match(t)
    if not m:
        return None
    head = m.group("ret")
    if "=" in head or "(" in head:     # a variable with an initializer, a macro call
        return None
    ret = norm_type([x for x in tokens(head) if x not in DROP_RET])
    if not ret and "::" not in m.group("name"):
        return None                    # a call or a constructor outside its class
    return Func(ret, m.group("name"), tuple(split_params(m.group("params"))), line)


def scan(path: Path):
    """(namespace-scope declarations, namespace-scope definitions, class body texts by class name) of one file."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    s = drop_preprocessor(strip_comments(raw))
    decls: list[Func] = []
    defs: list[Func] = []
    classes: dict[str, str] = {}
    i, n, start = 0, len(s), 0
    while i < n:
        c = s[i]
        if c == ";":
            f = parse_decl(s[start:i], s.count("\n", 0, start) + 1)
            if f:
                decls.append(f)
            start = i + 1
        elif c == "{":
            head = " ".join(s[start:i].split())
            if head.startswith("namespace") or head.startswith('extern "C"') or head == "":
                start = i + 1          # transparent
            else:
                depth, j = 1, i + 1
                while j < n and depth:
                    depth += {"{": 1, "}": -1}.get(s[j], 0)
                    j += 1
                if head.startswith(("struct ", "class ", "union ")):
                    m = re.match(r"(?:struct|class|union)\s+(?:alignas\s*\([^)]*\)\s*)?(\w+)", head)
                    if m:
                        classes[m.group(1)] = s[i + 1:j - 1]
                elif not head.startswith(TYPE_START):
                    f = parse_decl(re.sub(r"\)\s*:.*$", ")", head, flags=re.S), s.count("\n", 0, start) + 1)
                    if f:
                        defs.append(f)
                i = j
                # `struct X {...};` and `inline f() {...}` both end here
                k = i
                while k < n and s[k] in " \t\r\n":
                    k += 1
                if k < n and s[k] == ";":
                    i = k
                start = i + (1 if i < n and s[i] == ";" else 0)
                if i < n and s[i] == ";":
                    i += 1
                continue
        elif c == "}":
            start = i + 1              # a namespace closes
        i += 1
    return decls, defs, classes


def guarded_text(text: str, present: list[str]) -> str:
    """The text with the lines inside `#if defined(<present macro>)` (and `#else` of `#ifndef <present>`) blanked."""
    if not present:
        return text
    pres = "|".join(map(re.escape, present))
    pos_re = re.compile(r"^\s*#\s*(?:ifdef\s+(?:%s)\b|(?:el)?if\b(?![^\n]*!\s*defined\s*\(?\s*(?:%s)\b)"
                        r"[^\n]*\bdefined\s*\(?\s*(?:%s)\b)" % (pres, pres, pres))
    neg_re = re.compile(r"^\s*#\s*(?:ifndef\s+(?:%s)\b|(?:el)?if\b[^\n]*!\s*defined\s*\(?\s*(?:%s)\b)" % (pres, pres))
    out, stack = [], []                # stack of (kind, skipping)
    for ln in text.split("\n"):
        st = ln.lstrip()
        skipping = any(sk for _, sk in stack)
        if re.match(r"#\s*if", st):
            kind = "pos" if pos_re.match(ln) else "neg" if neg_re.match(ln) else None
            stack.append((kind, kind == "pos"))
        elif re.match(r"#\s*elif", st) and stack:
            kind = "pos" if pos_re.match(ln) else "neg" if neg_re.match(ln) else stack[-1][0]
            stack[-1] = (kind, kind == "pos")
        elif re.match(r"#\s*else", st) and stack:
            kind, sk = stack[-1]
            stack[-1] = (kind, (not sk) if kind else False)
        elif re.match(r"#\s*endif", st) and stack:
            stack.pop()
        out.append("" if skipping else ln)
    return "\n".join(out)


_STRIPPED: dict[Path, str] = {}


def stripped(p: Path) -> str:
    if p not in _STRIPPED:
        _STRIPPED[p] = strip_comments(p.read_text(encoding="utf-8", errors="replace"))
    return _STRIPPED[p]


def consumer_files(groups: list[Group]) -> list[Path]:
    skip = {g.stub for g in groups} | {h for g in groups for h in g.headers}
    files = []
    for top in ("src", "include"):
        for p in sorted((REPO / top).rglob("*")):
            rel = p.relative_to(REPO).as_posix()
            if p.suffix in CONSUMER_EXT and p.is_file() and rel not in skip and not SKIP_FILE_RE.search(rel):
                files.append(p)
    return files


def check_group(g: Group, consumers: list[Path], stub_text: str | None = None, verbose: bool = False) -> list[str]:
    errors: list[str] = []
    name_ok = re.compile(g.name_re) if g.name_re else None
    hdecl: dict[str, set[tuple[str, tuple[str, ...]]]] = {}
    inline_defs: set[str] = set()
    classes: dict[str, str] = {}
    for h in g.headers:
        d, defs, cl = scan(REPO / h)
        classes.update(cl)
        for f in d:
            if name_ok is None or name_ok.search(f.name):
                hdecl.setdefault(f.name, set()).add((f.ret, f.params))
        inline_defs |= {f.name for f in defs}
    if not hdecl:
        return [f"{g.name}: no function declarations parsed from {', '.join(g.headers)}"]

    stub_path = REPO / g.stub
    if stub_text is not None:
        tmp = stub_path.with_suffix(".selftest.tmp")
        tmp.write_text(stub_text, encoding="utf-8")
        try:
            _, sdefs, _ = scan(tmp)
        finally:
            tmp.unlink()
    else:
        _, sdefs, _ = scan(stub_path)
    stubbed: set[str] = set()
    for f in sdefs:
        if "::" in f.name:             # a member: Context::run, Context::Context
            cls, mem = f.name.split("::", 1)
            body = classes.get(cls)
            if body is None or not re.search(r"(?<![\w:])~?%s\s*\(" % re.escape(mem.lstrip("~")), body):
                errors.append(f"{g.stub}:{f.line}: {f.sig()} - no such member in the headers' class {cls}")
            continue
        stubbed.add(f.name)
        if f.name not in hdecl:
            errors.append(f"{g.stub}:{f.line}: {f.sig()} - not declared in {', '.join(g.headers)}")
        elif (f.ret, f.params) not in hdecl[f.name]:
            want = " | ".join(f"{r} {f.name}({', '.join(p)})" for r, p in sorted(hdecl[f.name]))
            errors.append(f"{g.stub}:{f.line}: {f.sig()} - header declares {want}")

    needed: dict[str, str] = {}
    if g.mirror_impl:
        for rel in g.impl:
            _, idefs, _ = scan(REPO / rel)
            for f in idefs:
                if f.name in hdecl:
                    needed.setdefault(f.name, f"defined by {rel}")
    impl_re = re.compile(g.impl_re) if g.impl_re else None
    names = sorted(set(hdecl) - inline_defs, key=len, reverse=True)
    if names:
        alt = "|".join(map(re.escape, names))
        call = re.compile((r"\b%s::(%s)\s*\(" % (re.escape(g.ns), alt)) if g.qualified else
                          (r"(?<![\w.>])(?:\w+::)*(%s)\s*\(" % alt))
        for p in consumers:
            rel = p.relative_to(REPO).as_posix()
            if rel in g.impl or (impl_re and impl_re.search(rel)):
                continue
            text = stripped(p)
            if not call.search(text):
                continue
            text = guarded_text(text, g.present)
            for m in call.finditer(text):
                needed.setdefault(m.group(1), f"called from {rel}:{text.count(chr(10), 0, m.start()) + 1}")
    for nm, why in sorted(needed.items()):
        if nm not in stubbed:
            sig = " | ".join(f"{r} {nm}({', '.join(p)})" for r, p in sorted(hdecl[nm]))
            errors.append(f"{g.stub}: missing stub for {sig} ({why})")
    if verbose:
        unused = sorted(set(hdecl) - stubbed - set(needed) - inline_defs)
        print(f"  {g.name}: {len(stubbed)} stubs, {len(needed)} needed, {len(hdecl)} header functions"
              + (f"; not needed without the backend: {', '.join(unused)}" if unused else ""))
    return errors


def selftest(consumers: list[Path]) -> int:
    g = next(x for x in GROUPS if x.name == "fused")
    text = (REPO / g.stub).read_text(encoding="utf-8")
    broken = re.sub(r"^void want_dm_f16\(bool\) \{\}.*\n", "", text, flags=re.M)
    drift = text.replace("bool native_supported(int, int)", "bool native_supported(int, long)")
    ok = True
    if broken == text or not any("missing stub" in e and "want_dm_f16" in e for e in check_group(g, consumers, broken)):
        print("selftest FAILED: removing fused::want_dm_f16 from the stub was not reported")
        ok = False
    if drift == text or not any("native_supported" in e for e in check_group(g, consumers, drift)):
        print("selftest FAILED: a changed stub signature was not reported")
        ok = False
    print("selftest ok" if ok else "")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verbose", action="store_true", help="per group: stub / needed / header counts")
    ap.add_argument("--selftest", action="store_true", help="check that a missing and a drifted stub are caught")
    a = ap.parse_args()
    consumers = consumer_files(GROUPS)
    if a.selftest:
        return selftest(consumers)
    errors: list[str] = []
    for g in GROUPS:
        try:
            errors += check_group(g, consumers, verbose=a.verbose)
        except (OSError, ValueError) as e:
            print(f"check_stubs: {g.name}: {e}", file=sys.stderr)
            return 2
    for e in errors:
        print(e)
    print(f"check_stubs: {'FAILED, ' + str(len(errors)) + ' problem(s)' if errors else 'ok'} "
          f"({len(GROUPS)} stub files, {len(consumers)} files scanned)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
