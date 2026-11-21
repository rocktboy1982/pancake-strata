#!/usr/bin/env python3
"""isa_audit.py - pancake-strata LEADS.md Part T T10: instruction-level audit of the gfx1100 disassembly of the hot kernels (offline, reads text only).

Input: `llvm-objdump -d --mcpu=gfx1100 CODEOBJ > dis.txt` (the ROCm llvm-objdump; box recipe: tools/pancake/box_chains/kres-t9.sh does the extraction).
Per function (`<symbol>:` header) it counts:
  instr, valu (v_*), salu (s_*), vopd (v_dual_*) and vopd_cand (single VALU instructions of a type the gfx11 VOPD encoding can pair: v_fmac / fmaak /
  fmamk / mul / add / sub / subrev / mov / cndmask / max / min f32, dot2c, add_nc_u32, lshlrev_b32, and_b32): pairing share = vopd*2 / (vopd*2 + vopd_cand);
  wmma (v_wmma_*), dot (v_dot*), sudot (v_dot4_i32_iu8 / v_dot*_iu*: the signed-by-unsigned int8 dot),
  scratch (scratch_* = spill or stack traffic: any is a finding), vmem (global_/buffer_ loads+stores), lds (ds_*),
  waitcnt (s_waitcnt*), tight_wait (an s_waitcnt vmcnt/lgkmcnt(0) within 1 instruction after the load that feeds it: no latency hiding),
  delay_alu (s_delay_alu: compiler-inserted dependency stalls), barrier (s_barrier),
  bank_cand (HEURISTIC: >= --bank-n ds_ ops with the same address register and offsets that are multiples of 128 bytes, or sub-dword ds_ ops
  (ds_load_u8/u16, ds_store_b8/b16): both are the usual LDS bank-conflict patterns; a candidate, never a proof: confirm with a counter, see prof_report.py).
    python3 tools/pancake/isa_audit.py dis.txt [more.txt ...] [--match REGEX] [--csv OUT.csv] [--json OUT.json] [--bank-n 4] [--top 40]
Exit 0 ok, 2 no function found.  stdlib only; writes only --csv / --json.  A disassembly is not a speed result: it says what the compiler emitted, nothing about time.
Tests: python -m pytest -q tools/pancake/test_isa_audit.py   (fixture disassembly; real code objects need the box)
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

SYM = re.compile(r"^[0-9a-fA-F]+ <([^>]+)>:")
INSN = re.compile(r"^\s+([a-z][a-z0-9_]*)\b(.*)$")
VOPD_OK = re.compile(r"^v_(fmac|fmaak|fmamk|mul|add|sub|subrev|max|min)_f32(_e32)?$|^v_(mov_b32|cndmask_b32|add_nc_u32|lshlrev_b32|and_b32|dot2c_f32_f16|dot2c_f32_bf16)(_e32)?$")
LOADS = re.compile(r"^(global|buffer|flat|ds|scratch|s_load|s_buffer_load)_?(load|read)")
OFFSET = re.compile(r"offset:(0x[0-9a-fA-F]+|\d+)")
COLS = ["name", "instr", "valu", "salu", "vopd", "vopd_cand", "vopd_share", "wmma", "dot", "sudot", "scratch", "vmem", "lds", "waitcnt",
        "tight_wait", "delay_alu", "barrier", "bank_cand"]


def parse(text: str) -> dict:
    """{function: [(mnemonic, operands), ...]}; comments after // are dropped; non-instruction lines are skipped."""
    funcs: dict = collections.OrderedDict()
    cur = None
    for line in text.splitlines():
        m = SYM.match(line)
        if m:
            cur = funcs.setdefault(m.group(1), [])
            continue
        if cur is None or line.lstrip().startswith((";", ".")):
            continue
        line = line.split("//", 1)[0].rstrip()
        m = INSN.match(line)
        if m:
            cur.append((m.group(1), m.group(2).strip()))
    return {k: v for k, v in funcs.items() if v}


def audit(name: str, insns: list, bank_n: int = 4) -> dict:
    r = dict.fromkeys(COLS[1:], 0)
    r["name"] = name
    r["instr"] = len(insns)
    ds_by_base = collections.Counter()
    last_load = None
    for i, (op, args) in enumerate(insns):
        if op.startswith("v_"):
            r["valu"] += 1
            if op.startswith("v_dual_"):
                r["vopd"] += 1
            elif VOPD_OK.match(op):
                r["vopd_cand"] += 1
            if op.startswith("v_wmma"):
                r["wmma"] += 1
            if op.startswith("v_dot"):
                r["dot"] += 1
                if "_iu" in op:
                    r["sudot"] += 1
        elif op.startswith("s_"):
            r["salu"] += 1
        if op.startswith("scratch_"):
            r["scratch"] += 1
        if op.startswith(("global_", "buffer_", "flat_")):
            r["vmem"] += 1
        if op.startswith("ds_"):
            r["lds"] += 1
            if re.search(r"_(u8|u16|i8|i16|b8|b16)$", op):
                ds_by_base["__subdword__"] += 1
            else:
                om = OFFSET.search(args)
                ops = [x.strip().split(" ")[0] for x in args.split(",")]
                base = ops[0 if op.startswith(("ds_store", "ds_write")) else min(1, len(ops) - 1)]
                if om and int(om.group(1), 0) and int(om.group(1), 0) % 128 == 0:
                    ds_by_base[base] += 1
        if op.startswith("s_waitcnt") and not op.startswith("s_waitcnt_"):
            r["waitcnt"] += 1
            if last_load is not None and i - last_load <= 2 and re.search(r"(vmcnt|lgkmcnt)\(0\)", args):
                r["tight_wait"] += 1
        elif op.startswith("s_waitcnt_"):
            r["waitcnt"] += 1
        if LOADS.match(op) and not op.startswith("s_"):
            last_load = i
        if op == "s_delay_alu":
            r["delay_alu"] += 1
        if op == "s_barrier":
            r["barrier"] += 1
    r["bank_cand"] = sum(1 for k, n in ds_by_base.items() if n >= bank_n)
    paired = r["vopd"] * 2
    r["vopd_share"] = round(paired / (paired + r["vopd_cand"]), 3) if (paired + r["vopd_cand"]) else None
    return r


def findings(r: dict) -> list:
    f = []
    if r["scratch"]:
        f.append(f"SPILL({r['scratch']} scratch ops)")
    if r["vopd_cand"] and r["vopd"] == 0:
        f.append("NO-VOPD")
    if r["tight_wait"] > 4:
        f.append(f"TIGHT-WAIT({r['tight_wait']})")
    if r["bank_cand"]:
        f.append(f"BANK?({r['bank_cand']})")
    return f


def render(rows: list, top: int = 40) -> str:
    out = [f"{'function':46} {'instr':>6} {'valu':>5} {'vopd':>5} {'cand':>5} {'share':>6} {'wmma':>5} {'dot':>4} {'iu':>3} {'scr':>4} {'lds':>4} {'wait':>5} {'tight':>5} {'dly':>5}  findings"]
    for r in sorted(rows, key=lambda r: -r["instr"])[:top]:
        sh = "-" if r["vopd_share"] is None else f"{r['vopd_share']:.2f}"
        out.append(f"{r['name'][:46]:46} {r['instr']:>6} {r['valu']:>5} {r['vopd']:>5} {r['vopd_cand']:>5} {sh:>6} {r['wmma']:>5} {r['dot']:>4} {r['sudot']:>3} "
                   f"{r['scratch']:>4} {r['lds']:>4} {r['waitcnt']:>5} {r['tight_wait']:>5} {r['delay_alu']:>5}  {' '.join(findings(r))}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--match", help="regex: only functions whose name matches")
    ap.add_argument("--csv")
    ap.add_argument("--json")
    ap.add_argument("--bank-n", type=int, default=4)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args(argv)
    funcs: dict = {}
    for p in a.files:
        funcs.update(parse(Path(p).read_text(errors="replace")))
    if a.match:
        rx = re.compile(a.match)
        funcs = {k: v for k, v in funcs.items() if rx.search(k)}
    if not funcs:
        print("no function with instructions found (input must be llvm-objdump -d output)", file=sys.stderr)
        return 2
    rows = [audit(k, v, a.bank_n) for k, v in funcs.items()]
    print(render(rows, a.top))
    print(f"{len(rows)} functions, {sum(1 for r in rows if r['scratch'])} with scratch, {sum(r['vopd'] for r in rows)} VOPD instructions in total")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLS)
            for r in rows:
                w.writerow([r[c] if r[c] is not None else "" for c in COLS])
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
