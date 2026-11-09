"""gpu_profile.py - pancake-strata: where the GPU's decode time goes, for any preset + overrides (rocprofv3 kernel trace).

Runs the engine of a server config directly (tools/pancake/bench_engines.py drives --serve) under
`rocprofv3 --kernel-trace --stats`, one request, then summarizes the decode phase (from the first to the last
combine/wait kernel): top kernels by time, kernels per token, idle gaps between kernels (launch gaps < 50 us,
host stalls > 1 ms, with the kernel pairs around the largest ones).  The profiler slows decode (~1.6x on the 7900 XT):
read shares and counts, not absolute speed.

    python tools/pancake/gpu_profile.py CONFIG.json OUTDIR [--case short] [--env K=V ...] [--arg "--flag value" ...]
"""
import argparse
import collections
import csv
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROCPROF = "/opt/rocm/core-10.0/bin/rocprofv3"


def run(cfg_path, out, case, envs, extra_args):
    c = json.load(open(cfg_path))
    env = dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/core-10.0/lib", HIP_VISIBLE_DEVICES="0")
    env.update(c.get("env") or {})
    for kv in envs:
        k, v = kv.split("=", 1)
        env[k] = v
    args = list(c["args"])
    for a in extra_args:
        args += shlex.split(a)
    out.mkdir(parents=True, exist_ok=True)
    wrapper = out / "prof-engine.sh"
    engine = c.get("engine") or str(ROOT / "build-hip/strata")
    wrapper.write_text(f"#!/bin/bash\nexec {ROCPROF} --kernel-trace --stats -d {out} -o run --output-format csv -- {engine} \"$@\"\n")
    wrapper.chmod(0o755)
    cmd = [str(ROOT / "tools/pancake/capped.sh"), str(ROOT / ".venv/bin/python"), str(ROOT / "tools/pancake/bench_engines.py"),
           "--name", "prof", "--tokenizer", c["tokenizer"], "--out", str(out / "bench.jsonl"), "--runs", "1", "--only", case,
           "--", str(wrapper), "--serve"] + args
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    (out / "run.log").write_text(r.stdout + r.stderr)
    return out / "run_kernel_trace.csv"


def short(n):
    n = n.replace("strata::kernels::(anonymous namespace)::", "").replace("strata::kernels::", "").replace("void ", "")
    return n.split("(")[0][:60]


def analyze(trace, out):
    rows = list(csv.DictReader(open(trace)))
    rows.sort(key=lambda r: int(r["Start_Timestamp"]))
    marks = [int(r["Start_Timestamp"]) for r in rows if "combine" in r["Kernel_Name"] or "wait_flag" in r["Kernel_Name"]]
    if not marks:
        print("no decode-phase marker kernels found")
        return
    d0, d1 = min(marks), max(marks)
    pf = [int(r["End_Timestamp"]) for r in rows if "prefill" in r["Kernel_Name"] or "prompt_" in r["Kernel_Name"]
          or "pfg::" in r["Kernel_Name"]]
    if pf:   # decode starts after the prompt's last kernel
        d0 = max(d0, max(pf))
        d1 = max(d1, d0)
    R = [r for r in rows if d0 <= int(r["Start_Timestamp"]) <= d1]
    dur = lambda r: int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
    tot = collections.Counter()
    calls = collections.Counter()
    for r in R:
        tot[short(r["Kernel_Name"])] += dur(r)
        calls[short(r["Kernel_Name"])] += 1
    busy = sum(tot.values())
    wall = d1 - d0
    try:
        tokens = json.loads(open(out / "bench.jsonl").readline())["new"]
    except Exception:
        tokens = 0
    lines = [f"decode phase {wall / 1e6:.0f} ms wall, kernels {busy / 1e6:.0f} ms ({busy / wall:.0%}), {len(R)} kernels"
             + (f", {len(R) / tokens:.0f} per token" if tokens else "")]
    lines.append("top kernels:")
    for k, v in tot.most_common(20):
        lines.append(f"  {v / busy * 100:5.1f}%  {v / 1e6:7.1f} ms  {calls[k]:6d}x  {v / calls[k] / 1e3:6.1f} us  {k}")
    gaps = collections.Counter()
    gapn = collections.Counter()
    small = big = 0
    end = int(R[0]["End_Timestamp"])
    for a, b in zip(R, R[1:]):
        end = max(end, int(a["End_Timestamp"]))
        g = int(b["Start_Timestamp"]) - end
        if g <= 0:
            continue
        if g < 50_000:
            small += g
        elif g > 1_000_000:
            big += g
        key = (short(a["Kernel_Name"]), short(b["Kernel_Name"]))
        gaps[key] += g
        gapn[key] += 1
    lines.append(f"idle: launch gaps (<50 us) {small / 1e6:.0f} ms, host stalls (>1 ms) {big / 1e6:.0f} ms of {wall / 1e6:.0f} ms")
    lines.append("largest idle by kernel pair:")
    for k, v in gaps.most_common(10):
        lines.append(f"  {v / 1e6:7.1f} ms {gapn[k]:6d}x avg {v / gapn[k] / 1e3:7.1f} us  {k[0]} -> {k[1]}")
    text = "\n".join(lines)
    (out / "summary.txt").write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("outdir")
    ap.add_argument("--case", default="short")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--arg", action="append", default=[])
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.outdir).resolve()
    trace = out / "run_kernel_trace.csv"
    if not a.analyze_only:
        trace = run(a.config, out, a.case, a.env, a.arg)
    if not trace.exists():
        print(f"no trace ({out / 'run.log'})")
        return 1
    analyze(trace, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
