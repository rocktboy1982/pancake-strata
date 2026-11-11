#!/usr/bin/env python3
"""expert_bw.py - probe 4.2.3 (bench/TEST_PLAN.md): achieved bandwidth of the expert GEMV kernels vs the card's peak.
Decides F12, F14, F16: there is little headroom above ~85% of peak.

QUESTION   How close do the GPU expert kernels (gate/up: IQ2_S / IQ2_XXS / IQ1_M, down: Q2_0) run to the memory
           bandwidth of the card (7900 XT ~800 GB/s, RTX 3060 ~360 GB/s)?
BUILD      nothing (Python 3 only).
RUN        1) a kernel trace of a decode run on the card:
                7900 XT: python tools/pancake/gpu_profile.py configs/pancake/flashnext-iq2_xs-fast-hip.json /tmp/prof3
                         -> /tmp/prof3/run_kernel_trace.csv  (rocprofv3 --kernel-trace --output-format csv)
                3060   : nsys profile -o /tmp/p3060 --stats=false <engine ...>;
                         nsys stats --report cuda_gpu_kern_trace --format csv -o /tmp/p3060 /tmp/p3060.nsys-rep
                         (the Start (ns) / Duration (ns) / Name form is read too)
           2) python tools/pancake/probes/expert_bw.py TRACE.csv [--gpu 7900xt|3060 | --peak GBPS] [--k 1,2,4,10]
                                                       [--hit-rate H] [--floor-us 3] [--table PATH] [--whole]
              --table: the GGUF tensor table (default bench/results/2026-11-10-offline/gguf_table.json on dev/batch-1,
              from bench/results/2026-11-10-offline/gguf_table.py; first model in it, or --model SUBSTRING)
              --k: experts per launch to assume.  The trace does NOT say how many experts a launch computes: the grid is
              sized for the capacity and blocks past the live groups exit at once (iq_kernels.cu launch_gu/launch_down).
              The live count is 10 x (VRAM hit rate) per layer, so pass --hit-rate H (from the engine's cache stats, e.g.
              0.6) and K = 10 x H is marked with "<=="; the table gives GB/s for each K anyway.
              --floor-us: launch + tail time to subtract (a ~3 us dependent launch in a graph), default 0 (raw).
              --gu-re / --down-re: kernel-name regexes if the build names its kernels differently.
READ       One block per class: gate/up (reads gate + up of one expert) and down, split by the quant type taken from the
           kernel's first template argument (ggml type id: 16 IQ2_XXS, 22 IQ2_S, 29 IQ1_M, 42 Q2_0).  Per class: calls,
           median/p10 us, the bytes per expert (from the table: IQ2_S gate/up 524800 B each, IQ2_XXS 422400, IQ1_M 358400,
           Q2_0 down 460800; layer-weighted mean when the type is unknown), then GB/s and % of peak at each K, and "K
           for 85%": how many experts one launch would have to carry to be at 85% of peak.  If the K you really have
           (10 x hit rate) is at or above that, the kernel is bandwidth-bound and F12/F14/F16 have little to gain; if
           the real K is far below it, the kernel is latency-bound (launch, tail, occupancy) and a better kernel may help.
"""
import argparse
import collections
import json
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trace_common as tc  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TABLE = ROOT / "bench/results/2026-11-10-offline/gguf_table.json"
PEAK = {"7900xt": 800.0, "3060": 360.0}
TYPE_ID = {16: "IQ2_XXS", 22: "IQ2_S", 29: "IQ1_M", 42: "Q2_0", 17: "IQ2_XS", 21: "IQ3_S", 18: "IQ3_XXS", 19: "IQ1_S",
           20: "IQ4_NL", 23: "IQ4_XS", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 10: "Q2_K", 11: "Q3_K", 8: "Q8_0"}
GU_RE = r"(^|_)gu(_|$)"
DOWN_RE = r"(^|_)down(_|$)"
NOT_EXPERT = r"dequant|swiglu_q8|swiglu_entries|quantize|hadamard|embed|shared|shexp"


def load_bytes(path, model):
    """-> (gu, down, mix): gu[type] = gate+up bytes of one expert, down[type] = down bytes, mix = layer-weighted means."""
    tab = json.load(open(path))
    keys = [k for k in tab if model in k]
    if not keys:
        sys.exit(f"{path}: no model matching {model!r}; has {list(tab)}")
    ple = tab[keys[0]]["per_layer_expert"]
    gu, down = {}, {}
    cnt = collections.Counter()
    sgu = sdn = 0.0
    for lay in ple.values():
        (gt, gb), (ut, ub), (dt, db) = lay["gate"], lay["up"], lay["down"]
        gu[gt] = gb + ub
        down[dt] = db
        cnt[gt] += 1
        sgu += gb + ub
        sdn += db
    n = len(ple)
    return gu, down, {"gu": sgu / n, "down": sdn / n, "layers": dict(cnt), "model": keys[0]}


def type_of(name):
    m = re.search(r"<\s*(\d+)\b", name)
    return TYPE_ID.get(int(m.group(1))) if m else None


def pct(v, p):
    s = sorted(v)
    return s[min(len(s) - 1, int(p * len(s)))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--gpu", choices=sorted(PEAK), default="7900xt")
    ap.add_argument("--peak", type=float, help="peak GB/s (overrides --gpu)")
    ap.add_argument("--k", default="1,2,3,4,6,8,10")
    ap.add_argument("--hit-rate", type=float)
    ap.add_argument("--floor-us", type=float, default=0.0)
    ap.add_argument("--table", default=str(DEFAULT_TABLE))
    ap.add_argument("--model", default="")
    ap.add_argument("--gu-re", default=GU_RE)
    ap.add_argument("--down-re", default=DOWN_RE)
    ap.add_argument("--whole", action="store_true")
    a = ap.parse_args()
    peak = a.peak or PEAK[a.gpu]
    ks = [int(x) for x in a.k.split(",")]
    kreal = a.hit_rate * 10 if a.hit_rate is not None else None
    gu_b, down_b, mix = load_bytes(a.table, a.model)
    print(f"model {Path(mix['model']).name}; layers by gate/up type {mix['layers']}; peak {peak:.0f} GB/s"
          + (f"; assumed K = {kreal:.1f} experts/launch (10 x hit rate {a.hit_rate})" if kreal else ""))

    rows = tc.decode_window(tc.load(a.trace), a.whole)
    cls = collections.defaultdict(list)   # (class, type) -> durations ns
    for r in rows:
        b = re.sub(r"<.*>", "", tc.short(r["name"]))
        if re.search(NOT_EXPERT, b):
            continue
        if re.search(a.gu_re, b):
            cls[("gate/up", type_of(tc.short(r["name"])))].append(r["dur"])
        elif re.search(a.down_re, b):
            cls[("down", type_of(tc.short(r["name"])))].append(r["dur"])
    if not cls:
        names = collections.Counter(re.sub(r"<.*>", "", tc.short(r["name"])) for r in rows)
        sys.exit("no expert kernel matched; most common kernels:\n  " + "\n  ".join(f"{n} x{c}" for n, c in names.most_common(15)))

    for (kind, ty), d in sorted(cls.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        if ty is None:
            per_expert = mix["gu"] if kind == "gate/up" else mix["down"]
            how = "type unknown: layer-weighted mean"
        else:
            tbl = gu_b if kind == "gate/up" else down_b
            if ty not in tbl:
                print(f"\n{kind} {ty}: not an expert type of this model, skipped ({len(d)} calls)")
                continue
            per_expert = tbl[ty]
            how = ty
        med, p10 = statistics.median(d), pct(d, 0.1)
        eff = lambda t: max(1.0, t - a.floor_us * 1e3)
        print(f"\n{kind:8s} [{how}]  {len(d)} calls, {per_expert / 1e3:.1f} kB per expert; us per call: median {med / 1e3:.1f}"
              f"  p10 {p10 / 1e3:.1f}  mean {statistics.mean(d) / 1e3:.1f}")
        print("    K experts/launch:  " + "".join(f"{k:>10d}" for k in ks))
        for label, t in (("median", med), ("p10 (best case)", p10)):
            gbs = [k * per_expert / eff(t) for k in ks]   # bytes / ns = GB/s
            print(f"    GB/s at {label:<15s}" + "".join(f"{g:10.0f}" for g in gbs))
            print(f"    % of peak         " + "".join(f"{100 * g / peak:9.0f}%" for g in gbs))
        k85 = 0.85 * peak * eff(med) / per_expert
        print(f"    K for 85% of peak at the median time: {k85:.1f} experts per launch")
        if kreal:
            g = kreal * per_expert / eff(med)
            print(f"    <== at K = {kreal:.1f}: {g:.0f} GB/s = {100 * g / peak:.0f}% of peak (median), "
                  f"{100 * kreal * per_expert / eff(p10) / peak:.0f}% (p10)")
    print("\nnote: gate/up reads gate + up of an expert; a K above the live expert count overstates the bandwidth, a K below it"
          " understates it.  Fixed launch/tail cost makes small K look slow: --floor-us 3 removes it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
