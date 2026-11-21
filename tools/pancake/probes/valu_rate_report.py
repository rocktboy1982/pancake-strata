#!/usr/bin/env python3
"""valu_rate_report.py - host side of the LEADS.md Part U U4 instruction-rate probe (tools/pancake/probes/valu_rate.hip.cpp).

  valu_rate_report.py STDOUT_FILE [--cus 84] [--lds-bytes 128] [--vopd-lanes 128]
      parse the probe's `DEVICE` / `RESULT` lines and print, per mode and occupancy: ops per clock per CU, instructions per
      clock per CU, the theoretical instructions per clock per CU, the percentage of it, and the chip rate in T ops/s at the
      clock the kernel itself measured (clock64 against the event time) next to the theoretical rate on --cus CUs at that clock.
      Then the ratios that decide U4: dot4 / fma, dot2 / fma, mad24 / fma, vopd / fma, LDS bytes per clock.
  valu_rate_report.py --isa FILE.s
      check an assembly listing of the probe (hipcc --offload-arch=gfx1100 -S --cuda-device-only): every mode's kernel
      must contain the mnemonic it was written to measure (no GPU needed).

Theoretical model (gfx11, per CU): two SIMD32 -> 64 lane-instructions per clock for single-issue VALU (fma, add, mul, dot2,
dot4, mad24, perm); a VOPD dual-issue pair counts 2 ops, so the `vopd` row's ceiling is 128 (--vopd-lanes); LDS 128 bytes per
clock per CU (--lds-bytes, an assumption: check against the card's documentation before quoting); WMMA f16 512 flop per clock per CU
(one 16x16x16 tile per wave = 8192 flop).  The percentages are against THESE numbers; a percentage is not a claim about the hardware.
Exit code 0 when the input parses (or the ISA check passes), 1 otherwise.
"""
import argparse
import re
import sys

# mode -> (theoretical instructions per clock per CU, arithmetic unit label)
VALU_INST_PER_CLK = 64.0
THEORY_INST = {m: VALU_INST_PER_CLK for m in ("fma", "add", "mul", "dot2", "dot4", "mad24", "perm")}
# WMMA: instructions are counted per wave in the probe (inst_per_clk_cu = wave instructions): one wave-instruction per
# 16 clocks per SIMD would be 2 SIMDs x 1/16 ... the ceiling is expressed in flop instead (see percent_of_theory)
WMMA_FLOP_PER_CLK_CU = 512.0

ISA_EXPECT = {   # mode number (enum order of valu_rate.hip.cpp) -> (name, mnemonic regex, forbid regex or None)
    0: ("fma", r"\bv_fma_f32\b", r"\bv_dual_"),
    1: ("add", r"\bv_add_f32\b", r"\bv_dual_"),
    2: ("mul", r"\bv_mul_f32\b", r"\bv_dual_"),
    3: ("vopd", r"\bv_dual_(fmac_f32|mul_f32|fma)", None),
    4: ("dot2", r"\bv_dot2_f32_f16\b", None),
    5: ("dot4", r"\bv_dot4_i32_iu8\b", None),
    6: ("mad24", r"\bv_mad_i32_i24\b", None),
    7: ("perm", r"\bv_perm_b32\b", None),
    8: ("lds_r32", r"\bds_load_b32\b", None),
    9: ("lds_r128", r"\bds_load_b128\b", None),
    10: ("lds_w32", r"\bds_store_b32\b", None),
    11: ("lds_w128", r"\bds_store_b128\b", None),
    12: ("wmma_f16", r"\bv_wmma_f32_16x16x16_f16\b", None),
}


def parse_kv(line):
    out = {}
    for m in re.finditer(r'(\w+)=("[^"]*"|\S+)', line):
        out[m.group(1)] = m.group(2).strip('"')
    return out


def parse(text):
    """-> (device dict or None, list of result dicts with numeric fields converted)"""
    dev = None
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("DEVICE "):
            dev = parse_kv(line)
        elif line.startswith("RESULT "):
            kv = parse_kv(line)
            try:
                row = {
                    "mode": kv["mode"],
                    "waves": int(kv["waves_per_simd"]),
                    "unit": kv.get("unit", ""),
                    "ops_per_clk_cu": float(kv["ops_per_clk_cu"]),
                    "inst_per_clk_cu": float(kv["inst_per_clk_cu"]),
                    "ops_per_s": float(kv["ops_per_s"]),
                    "clock_ghz": float(kv["implied_clock_ghz"]),
                    "best_ops_per_clk_cu": float(kv.get("best_ops_per_clk_cu", kv["ops_per_clk_cu"])),
                }
            except (KeyError, ValueError):
                continue
            rows.append(row)
    return dev, rows


def percent_of_theory(row, lds_bytes, vopd_lanes):
    """-> (theoretical rate in the row's own unit per clock per CU, percent).  VALU / dot: instructions; vopd: ops; lds: bytes; wmma: flop."""
    mode = row["mode"]
    if mode in THEORY_INST:
        th = THEORY_INST[mode]
        return th, 100.0 * row["inst_per_clk_cu"] / th
    if mode == "vopd":
        return vopd_lanes, 100.0 * row["inst_per_clk_cu"] / vopd_lanes
    if mode.startswith("lds"):
        return lds_bytes, 100.0 * row["ops_per_clk_cu"] / lds_bytes
    if mode == "wmma_f16":
        return WMMA_FLOP_PER_CLK_CU, 100.0 * row["ops_per_clk_cu"] / WMMA_FLOP_PER_CLK_CU
    return float("nan"), float("nan")


def best_by_mode(rows):
    """the highest-occupancy-independent figure: the best median over the waves-per-SIMD values of each mode"""
    best = {}
    for r in rows:
        if r["mode"] not in best or r["inst_per_clk_cu"] > best[r["mode"]]["inst_per_clk_cu"]:
            best[r["mode"]] = r
    return best


def ratio_lines(rows):
    b = best_by_mode(rows)
    lines = []
    fma = b.get("fma")
    if fma:
        for m in ("dot4", "dot2", "mad24", "add", "mul", "perm", "vopd"):
            if m in b:
                lines.append("%-6s / fma = %.2f (instructions per clock, best occupancy of each)" % (m, b[m]["inst_per_clk_cu"] / fma["inst_per_clk_cu"]))
        if "dot4" in b:
            q = b["dot4"]["inst_per_clk_cu"] / fma["inst_per_clk_cu"]
            if q >= 0.9:
                lines.append("VERDICT dot4: full rate (>= 0.90 of fma): an sudot4 kernel is not slowed by the instruction; rdna_kq / expert decode stay bandwidth questions")
            elif q >= 0.45:
                lines.append("VERDICT dot4: reduced rate (%.2f of fma): count the sudot4 per byte of weights in rdna_kq before calling it bandwidth-bound" % q)
            else:
                lines.append("VERDICT dot4: quarter rate or worse (%.2f of fma): a sudot4 loop is instruction-bound before it is bandwidth-bound" % q)
        if "vopd" in b:
            q = b["vopd"]["inst_per_clk_cu"] / fma["inst_per_clk_cu"]
            lines.append("VERDICT vopd: %s (%.2f of the single-issue fma rate; check --isa for v_dual_ in the loop)" % ("pairs issue" if q >= 1.5 else "no dual issue seen", q))
    for m in ("lds_r32", "lds_r128", "lds_w32", "lds_w128"):
        if m in b:
            lines.append("%-8s %.1f bytes per clock per CU" % (m, b[m]["ops_per_clk_cu"]))
    return lines


def report(text, cus, lds_bytes, vopd_lanes):
    dev, rows = parse(text)
    out = []
    if dev:
        out.append("device %s (%s), %s CUs reported, clock %s MHz reported; the table uses --cus %d" % (dev.get("name", "?"), dev.get("arch", "?"), dev.get("cus", "?"), dev.get("reported_clock_mhz", "?"), cus))
    if not rows:
        out.append("no RESULT lines parsed")
        return "\n".join(out), False
    out.append("%-9s %5s %12s %12s %9s %7s %10s %12s" % ("mode", "W/SIMD", "ops/clk/CU", "inst/clk/CU", "theory", "%", "Tops chip", "Tops theory"))
    for r in rows:
        th, pct = percent_of_theory(r, lds_bytes, vopd_lanes)
        unit_scale = r["ops_per_clk_cu"] / r["inst_per_clk_cu"] if r["inst_per_clk_cu"] else 0.0
        # chip rate measured by the event time; theory = theoretical per-clock-per-CU rate in ops x CUs x measured clock
        th_ops = th * (1.0 if r["mode"].startswith("lds") or r["mode"] == "wmma_f16" else unit_scale) * cus * r["clock_ghz"] * 1e9
        out.append("%-9s %5d %12.2f %12.2f %9.1f %6.1f%% %10.2f %12.2f" % (r["mode"], r["waves"], r["ops_per_clk_cu"], r["inst_per_clk_cu"], th, pct, r["ops_per_s"] / 1e12, th_ops / 1e12))
    out.extend(ratio_lines(rows))
    return "\n".join(out), True


def isa_check(text):
    """-> (report text, ok).  Splits the listing at the rate_kernel instances and looks for each mode's mnemonic in its own function."""
    funcs = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"^(_Z\w*rate_kernelILi(\d+)E\w*):", line)
        if m:
            cur = int(m.group(2))
            funcs[cur] = []
            continue
        if cur is not None:
            if re.match(r"^\s*\.(Lfunc_end|size|end_amdhsa_kernel)", line):
                cur = None
                continue
            funcs[cur].append(line)
    out, ok = [], True
    for mode, (name, want, forbid) in sorted(ISA_EXPECT.items()):
        body = "\n".join(funcs.get(mode, []))
        if not body:
            out.append("%-9s MISSING (no kernel body found)" % name)
            ok = False
            continue
        n = len(re.findall(want, body))
        nd = len(re.findall(r"\bv_dual_", body))
        status = "ok" if n > 0 else "NOT EMITTED"
        if forbid and re.search(forbid, body):
            status = "UNEXPECTED v_dual_ (a VOPD pair formed: the rate is not the single-issue rate)"
        if mode == 3 and nd == 0:
            status = "no v_dual_ in the loop (the compiler did not pair: the vopd row measures single issue)"
        if status not in ("ok",) and mode != 3:
            ok = False
        out.append("%-9s %-9s %d x %s, %d v_dual_" % (name, status, n, want.replace("\\b", ""), nd))
    return "\n".join(out), ok


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stdout_file", nargs="?")
    ap.add_argument("--cus", type=int, default=84)
    ap.add_argument("--lds-bytes", type=float, default=128.0)
    ap.add_argument("--vopd-lanes", type=float, default=128.0)
    ap.add_argument("--isa", help="check an assembly listing instead")
    a = ap.parse_args(argv)
    if a.isa:
        text, ok = isa_check(open(a.isa, encoding="utf-8", errors="replace").read())
        print(text)
        return 0 if ok else 1
    if not a.stdout_file:
        ap.error("a stdout file or --isa FILE.s is required")
    text, ok = report(open(a.stdout_file, encoding="utf-8", errors="replace").read(), a.cus, a.lds_bytes, a.vopd_lanes)
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
