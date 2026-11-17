#!/usr/bin/env python3
"""hip_resources.py - pancake-strata: per-kernel registers, spills, LDS and occupancy of a HIP object, from the compiler's own remarks.

The HIP twin of tools/check_cuda_spills.py's idea: nothing on the build machine says that a kernel spills or sits just over an
occupancy step, and a GPU is needed for a profiler to.  clang writes it, though, when asked:

    clang++ -x hip --offload-arch=gfx1100 --offload-device-only -O2 -S -o /dev/null -Rpass-analysis=kernel-resource-usage \
        <the engine's flags> kernel.cu 2> remarks.txt
    python tools/pancake/hip_resources.py remarks.txt [--csv OUT.csv] [--cliff 8] [--fail-on-spill]

(Windows: the ROCm SDK's clang++ with --rocm-path works; the remarks are on stderr.)  Several files or `-` (stdin) are read.

Per kernel it prints SGPRs, VGPRs, scratch bytes per lane, VGPR / SGPR spills, LDS bytes per block and the occupancy the compiler
reports (waves per SIMD).  Flags:
  SPILL   scratch > 0 or a VGPR / SGPR spill: round trips to memory in the hot loop.
  CLIFF   the kernel is limited by its VGPRs and needs at most --cliff (default 8) fewer to reach the next occupancy step.
          SILVERTHRONE32/llama.cpp-rdna3-int4 (MIT) measured that every variant of its rotation matvec that went from 112 to 128 VGPRs
          lost 1-5% (LEADS.md Part L, L7): a kernel a few VGPRs above a step is where such a loss is cheapest to remove.
The occupancy model (gfx11, wave32: VGPR file 1536 per SIMD lane group, allocation granule 24, at most 16 waves) is NOT assumed: every
kernel's reported occupancy is compared with it, and the summary line counts agreements, kernels held lower by another limit
(LDS, launch bounds) and contradictions (reported above the model's prediction).  Any contradiction makes the CLIFF column
untrustworthy, so it is printed as "model?" and the exit code is 3.  Offline, stdlib only; it never runs a kernel.
Exit 0 = ok, 1 = a spill with --fail-on-spill, 2 = no remarks found, 3 = the occupancy model contradicts the compiler.
"""
import argparse
import csv
import re
import sys

VGPR_FILE = 1536     # gfx11 wave32 per SIMD (checked against the remarks of qsa_decode_attn.cu: 124 VGPRs -> 10 waves)
GRANULE = 24
MAX_WAVES = 16

NAME_RE = re.compile(r"remark:\s+Function Name:\s+(\S+)")
KEY_RES = {
    "sgprs": re.compile(r"remark:\s+SGPRs:\s+(\d+)"),
    "vgprs": re.compile(r"remark:\s+VGPRs:\s+(\d+)"),
    "scratch": re.compile(r"remark:\s+ScratchSize \[bytes/lane\]:\s+(\d+)"),
    "occupancy": re.compile(r"remark:\s+Occupancy \[waves/SIMD\]:\s+(\d+)"),
    "sgpr_spill": re.compile(r"remark:\s+SGPRs Spill:\s+(\d+)"),
    "vgpr_spill": re.compile(r"remark:\s+VGPRs Spill:\s+(\d+)"),
    "lds": re.compile(r"remark:\s+LDS Size \[bytes/block\]:\s+(\d+)"),
}


def parse(lines):
    """-> list of kernels (dicts with name and the KEY_RES keys, ints), in file order; a kernel without a Function Name line is dropped."""
    out, cur = [], None
    for ln in lines:
        m = NAME_RE.search(ln)
        if m:
            cur = {"name": m.group(1)}
            out.append(cur)
            continue
        if cur is None:
            continue
        for k, rx in KEY_RES.items():
            m = rx.search(ln)
            if m:
                cur[k] = int(m.group(1))
                break
    return [k for k in out if "vgprs" in k and "occupancy" in k]


def model_waves(vgprs, vgpr_file=VGPR_FILE, granule=GRANULE, max_waves=MAX_WAVES):
    """Waves per SIMD the VGPR count alone allows."""
    alloc = max(granule, -(-vgprs // granule) * granule)
    return min(max_waves, vgpr_file // alloc)


def classify(k, cliff=8):
    """Adds model, relation ('match' | 'other-limit' | 'contradiction'), spill, cliff (None or (waves, vgprs_to_drop))."""
    pred = model_waves(k["vgprs"])
    rep = k["occupancy"]
    k["model"] = pred
    k["relation"] = "match" if rep == pred else ("other-limit" if rep < pred else "contradiction")
    k["spill"] = bool(k.get("scratch", 0) or k.get("vgpr_spill", 0) or k.get("sgpr_spill", 0))
    k["cliff"] = None
    if k["relation"] == "match" and pred < MAX_WAVES:
        v = k["vgprs"] - 1
        while v > 0 and model_waves(v) <= pred:   # the highest VGPR count that already allows more waves
            v -= 1
        drop = k["vgprs"] - v
        if v > 0 and drop <= cliff:
            k["cliff"] = (model_waves(v), drop)
    return k


def report(kernels, cliff=8):
    ks = [classify(dict(k), cliff) for k in kernels]
    n_match = sum(k["relation"] == "match" for k in ks)
    n_other = sum(k["relation"] == "other-limit" for k in ks)
    n_bad = sum(k["relation"] == "contradiction" for k in ks)
    return ks, {"kernels": len(ks), "match": n_match, "other_limit": n_other, "contradiction": n_bad,
                "spills": sum(k["spill"] for k in ks), "cliffs": sum(k["cliff"] is not None for k in ks)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="+", help="remark text files, or - for stdin")
    ap.add_argument("--csv")
    ap.add_argument("--cliff", type=int, default=8, help="flag a kernel needing at most this many fewer VGPRs for the next step")
    ap.add_argument("--fail-on-spill", action="store_true")
    a = ap.parse_args(argv)
    lines = []
    for f in a.files:
        lines.extend((sys.stdin if f == "-" else open(f, encoding="utf-8", errors="replace")).read().splitlines())
    ks, s = report(parse(lines), a.cliff)
    if not ks:
        print("hip_resources: no kernel-resource-usage remarks found (compile with -Rpass-analysis=kernel-resource-usage)")
        return 2
    print(f"{'kernel':60s} {'VGPR':>5s} {'SGPR':>5s} {'scratch':>7s} {'LDS':>6s} {'occ':>3s} {'model':>5s}  flags")
    for k in sorted(ks, key=lambda k: (-k["vgprs"], k["name"])):
        flags = []
        if k["spill"]:
            flags.append("SPILL")
        if k["relation"] == "contradiction":
            flags.append("model?")
        elif k["cliff"]:
            flags.append(f"CLIFF -{k['cliff'][1]} VGPR -> {k['cliff'][0]} waves")
        if k["relation"] == "other-limit":
            flags.append("held by LDS/launch bounds")
        print(f"{k['name'][:60]:60s} {k['vgprs']:5d} {k.get('sgprs', 0):5d} {k.get('scratch', 0):7d} {k.get('lds', 0):6d} "
              f"{k['occupancy']:3d} {k['model']:5d}  {'; '.join(flags)}")
    print(f"{s['kernels']} kernels: occupancy model agrees on {s['match']}, held lower by another limit on {s['other_limit']}, "
          f"contradicted on {s['contradiction']}; {s['spills']} with spills; {s['cliffs']} within {a.cliff} VGPRs of a step")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["kernel", "vgprs", "sgprs", "scratch", "vgpr_spill", "sgpr_spill", "lds", "occupancy", "model", "relation", "cliff_waves", "cliff_drop"])
            for k in ks:
                w.writerow([k["name"], k["vgprs"], k.get("sgprs", 0), k.get("scratch", 0), k.get("vgpr_spill", 0), k.get("sgpr_spill", 0),
                            k.get("lds", 0), k["occupancy"], k["model"], k["relation"],
                            k["cliff"][0] if k["cliff"] else "", k["cliff"][1] if k["cliff"] else ""])
    if s["contradiction"]:
        return 3
    if a.fail_on_spill and s["spills"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
