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


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return "-"


def system_snapshot(out):
    # phase-18 adjacency: the environment a run starts in, to explain bimodal ~25% slow runs.
    # THP state, reserved hugepages, card VRAM/GTT headroom, DPM clock, and a dmesg tail. Read-only,
    # all paths optional (a missing path writes "-"); never fails the profile.
    L = []
    try:
        mi = {}
        with open("/proc/meminfo") as f:
            for ln in f:
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    mi[k] = v.strip()
        L.append("MemTotal %s  MemAvailable %s  AnonHugePages %s" % (mi.get("MemTotal", "-"), mi.get("MemAvailable", "-"),
                                                                     mi.get("AnonHugePages", "-")))
    except OSError:
        L.append("meminfo: unreadable")
    L.append("THP enabled " + _read("/sys/kernel/mm/transparent_hugepage/enabled"))
    L.append("vm.nr_hugepages " + _read("/proc/sys/vm/nr_hugepages"))
    card = os.environ.get("P18_DRM_CARD", "card1")
    base = "/sys/class/drm/%s/device" % card
    L.append("card %s vram_used %s gtt_used %s vram_total %s" % (card, _read(base + "/mem_info_vram_used"),
                                                                 _read(base + "/mem_info_gtt_used"),
                                                                 _read(base + "/mem_info_vram_total")))
    sclk = _read(base + "/pp_dpm_sclk")
    L.append("pp_dpm_sclk " + (sclk.split("\n")[-1].strip() if sclk != "-" else "-"))
    try:
        dm = subprocess.run(["dmesg"], capture_output=True, text=True, timeout=5)
        tail = [ln for ln in dm.stdout.strip().splitlines() if ln][-5:]
        L.append("dmesg tail:\n" + ("\n".join(tail) if tail else "-"))
    except Exception:
        L.append("dmesg: unreadable")
    (out / "system_before.txt").write_text("\n".join(L) + "\n")


def run(cfg_path, out, case, envs, extra_args):
    # the placeholders <REPO>, <STRATA_DATA> and <MODELS> are filled the way abtest.render_preset does
    rendered = open(cfg_path, encoding="utf-8").read().replace("<REPO>", str(ROOT)) \
        .replace("<STRATA_DATA>", "/home/dan/clone-of-d/Strata-data").replace("<MODELS>", "/home/dan/clone-of-d/models")
    c = json.loads(rendered)
    env = dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/core-10.0/lib", HIP_VISIBLE_DEVICES="0")
    env.update(c.get("env") or {})
    for kv in envs:
        k, v = kv.split("=", 1)
        env[k] = v
    args = list(c["args"])
    for a in extra_args:
        args += shlex.split(a)
    out.mkdir(parents=True, exist_ok=True)
    engine = c.get("engine") or str(ROOT / "build-hip/strata")
    (out / "run_environment.json").write_text(json.dumps({
        "config": str(Path(cfg_path).resolve()),
        "engine": engine,
        "args": args,
        "case": case,
        "strata_env": dict(sorted((k, v) for k, v in env.items() if k.startswith("STRATA_"))),
        "overrides": envs,
    }, indent=2) + "\n")
    wrapper = out / "prof-engine.sh"
    wrapper.write_text(f"#!/bin/bash\nexec {ROCPROF} --kernel-trace --stats -d {out} -o run --output-format csv -- {engine} \"$@\"\n")
    wrapper.chmod(0o755)
    system_snapshot(out)
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
    streams = collections.defaultdict(list)
    for r in R:
        stream = r.get("Stream_Id") or r.get("Stream ID") or r.get("Stream") or "unknown"
        streams[stream].append(r)
    # Ensure we have at least one real stream identifier; otherwise per-stream analysis is meaningless
    if not any(s != "unknown" for s in streams):
        print("Warning: No valid Stream_Id found in trace; per-stream gap analysis will be meaningless.", file=sys.stderr)
    small = big = 0
    for stream, stream_rows in streams.items():
        stream_rows.sort(key=lambda r: int(r["Start_Timestamp"]))
        end = int(stream_rows[0]["End_Timestamp"])
        for a, b in zip(stream_rows, stream_rows[1:]):
            end = max(end, int(a["End_Timestamp"]))
            g = int(b["Start_Timestamp"]) - end
            if g <= 0:
                continue
            if g < 50_000:
                small += g
            elif g > 1_000_000:
                big += g
            key = (stream, short(a["Kernel_Name"]), short(b["Kernel_Name"]))
            gaps[key] += g
            gapn[key] += 1
    lines.append(f"per-stream gaps: launch (<50 us) {small / 1e6:.0f} ms, stalls (>1 ms) {big / 1e6:.0f} ms; sums may overlap within {wall / 1e6:.0f} ms wall")
    lines.append("largest idle by kernel pair and stream:")
    for k, v in gaps.most_common(10):
        lines.append(f"  {v / 1e6:7.1f} ms {gapn[k]:6d}x avg {v / gapn[k] / 1e3:7.1f} us  {k[1]} -> {k[2]} (stream {k[0]})")
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
