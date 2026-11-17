#!/usr/bin/env python3
"""ablate_expdec.py - loads / arithmetic split of the EXPDEC gate/up kernel from three expdec_bench builds (LEADS.md Part O O14; the method is the caminus
authors' "loads-only" / "no arithmetic" ablation builds, our own code).  The kernel has two compile-time ablations (src/kernels/cuda/iq_kernels.cu,
-DSTRATA_KERNEL_ABLATE=N, never in a serving build): 1 no-mac (weights loaded and decoded, no dot product), 2 same-row (every row reads row 0: cache
hits instead of DRAM).  Each build's `expdec_bench` prints one line per cell; this tool reads the three outputs and prints, per cell, gu_us of the default
build and of both ablations and the two shares.

  python3 tools/pancake/ablate_expdec.py build  [--pairs 16/42,17/42,22/42] [--reps 300] [--device 0] --out bench/results/<day>-ablate-expdec   (box: three builds + runs)
  python3 tools/pancake/ablate_expdec.py report DEFAULT.txt NOMAC.txt SAMEROW.txt
Reading: share_mac = (gu_def - gu_nomac) / gu_def is what the multiply-accumulate and the activation reads cost; share_dram = (gu_def - gu_samerow) /
gu_def is what the weight bytes from DRAM cost.  The two overlap (a kernel that is bound by loads hides arithmetic): a small share_mac with a large
share_dram says the kernel is bytes-bound and a decode tile change cannot help; the reverse says decode / MAC bound.  The ablation outputs are wrong on
purpose, only time is read.  Tests: python -m pytest -q tools/pancake/test_ablate_expdec.py
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

LINE = re.compile(r"^(\d+) (\d+) (\w+) (\d+/\d+) (\S+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+)% (identical|DIFFERENT)\s*$")


def parse(text: str) -> dict:
    """{(dev, T, shape, pair, choice): gu_us} from an expdec_bench output."""
    out = {}
    for l in text.splitlines():
        m = LINE.match(l.strip())
        if m:
            out[(int(m[1]), int(m[2]), m[3], m[4], m[5])] = float(m[7])
    return out


def ablate_macro(text: str):
    m = re.search(r"# STRATA_KERNEL_ABLATE=(\d+)", text)
    return int(m[1]) if m else 0


def report(default: dict, nomac: dict, samerow: dict) -> list:
    """Rows (key, gu_def, gu_nomac, gu_samerow, share_mac, share_dram) for the cells present in all three."""
    rows = []
    for k in sorted(set(default) & set(nomac) & set(samerow)):
        d, a, b = default[k], nomac[k], samerow[k]
        rows.append((k, d, a, b, (d - a) / d if d else 0.0, (d - b) / d if d else 0.0))
    return rows


def format_rows(rows: list) -> str:
    head = f"{'dev T shape gu/d choice':34} {'gu_def':>8} {'gu_nomac':>9} {'gu_samerow':>10} {'share_mac':>9} {'share_dram':>10}"
    lines = [head]
    for (dev, T, shape, pair, choice), d, a, b, sm, sd in rows:
        lines.append(f"{f'{dev} {T} {shape} {pair} {choice}':34} {d:8.1f} {a:9.1f} {b:10.1f} {sm * 100:8.1f}% {sd * 100:9.1f}%")
    return "\n".join(lines)


def build_and_run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, macro in (("default", None), ("nomac", 1), ("samerow", 2)):
        bdir = f"build-hip-ablate-{name}"
        cfg = ["cmake", "-S", ".", "-B", bdir, "-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF", "-DCMAKE_HIP_ARCHITECTURES=gfx1100"]
        if macro:
            cfg.append(f"-DCMAKE_HIP_FLAGS=-DSTRATA_KERNEL_ABLATE={macro}")
        for cmd in (cfg, ["cmake", "--build", bdir, "--target", "expdec_bench", "-j2"]):
            if subprocess.run(cmd).returncode != 0:
                print("ablate_expdec: build failed:", " ".join(cmd), file=sys.stderr)
                return 1
        r = subprocess.run([f"{bdir}/expdec_bench", str(args.reps), str(args.device), "20", args.pairs, "def,dec"], capture_output=True, text=True)
        (out / f"{name}.txt").write_text(r.stdout)
        if macro and ablate_macro(r.stdout) != macro:
            print(f"ablate_expdec: the {name} build did not print STRATA_KERNEL_ABLATE={macro} (the flag did not reach the kernels)", file=sys.stderr)
            return 1
    return report_files(out / "default.txt", out / "nomac.txt", out / "samerow.txt")


def report_files(d, n, s) -> int:
    texts = [Path(x).read_text() for x in (d, n, s)]
    if ablate_macro(texts[0]) != 0 or ablate_macro(texts[1]) != 1 or ablate_macro(texts[2]) != 2:
        print("ablate_expdec: the three files are not default / no-mac / same-row builds (the STRATA_KERNEL_ABLATE line)", file=sys.stderr)
        return 2
    rows = report(*[parse(t) for t in texts])
    print(format_rows(rows))
    return 0 if rows else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--pairs", default="16/42,17/42,22/42")
    b.add_argument("--reps", type=int, default=300)
    b.add_argument("--device", type=int, default=0)
    b.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("files", nargs=3)
    a = ap.parse_args(argv)
    return build_and_run(a) if a.cmd == "build" else report_files(*a.files)


if __name__ == "__main__":
    sys.exit(main())
