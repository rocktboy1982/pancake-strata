#!/usr/bin/env python3
"""load_time.py - measure the engine's START time with the direct (O_DIRECT) expert-arena loader against the buffered one
(MOD_PLAN A52 / item 38, commit 184e68e83 = 5c071d364 on dev/batch-2; switch STRATA_UNBUFFERED_LOAD).

abtest.py judges decode speed; this tool judges the load: wall seconds from the server process start to its "ready: http" line,
plus what the load did to the machine.  Per run it records
  start_s        seconds to "ready: http" (polled every 0.5 s)
  load line      the engine's "expert arena load: mode M, N threads, G GiB in S s (R GiB/s); ... P s before the read" (mode = which
                 loader ran: unbuffered = direct, buffered = through the page cache) and its "expert arena read ..." reason line
  psi_some_s /   memory pressure stall seconds during the start (/proc/pressure/memory "total" deltas; "full" is the oomd trigger)
  psi_full_s
  cached_peak_gib  how far the page cache (Cached in /proc/meminfo) grew above its start value: the buffered path copies the
                 GGUF through the cache next to the arena (the double copy), the direct path should not
  avail_min_gib  the lowest MemAvailable seen
  checksum       the arena checksum when STRATA_VERIFY_ARENA=1 is in the arm (it is, by default, in the "verify" arms): every mode must
                 give the same value
Arms (default: cold and warm cache for each mode; --modes picks):
  default   the preset as it is (the engine decides: direct when the page cache cannot keep the arena)
  direct    STRATA_UNBUFFERED_LOAD=1
  buffered  STRATA_UNBUFFERED_LOAD=0
  "cold" = the model files' page cache is advised away (posix_fadvise DONTNEED, no root, only this model's files) before every start;
  "warm" = left as the previous start left it (the usual restart).  Runs are interleaved (default, direct, buffered, default, ...).

    python tools/pancake/load_time.py --out bench/results/<day>-load-time [--base pancake-flashnext-iq2_xs-fast-hip.json]
                                      [--reps 3] [--modes default,direct,buffered] [--cache cold,warm] [--no-verify]
    python tools/pancake/load_time.py --from-rows bench/results/<day>-load-time/rows.jsonl     print the table again

The machine must be quiet and the GPU free (it starts the real server, like abtest.py): run it in the coordinator's window.
Expected (README row): start 28-65 s -> 10-14 s with the direct loader on a cold cache.  Decide: direct is better when its cold start is
shorter than buffered's by more than the run-to-run spread and its psi_full_s / cached_peak_gib are lower; the checksums must match.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

LOAD_RE = re.compile(r"expert arena load: mode (\S+), (\d+) threads, ([\d.]+) GiB in ([\d.]+) s \(([\d.]+) GiB/s\); "
                     r"arena reserved/registered/pinned in ([\d.]+) s before the read")
READ_RE = re.compile(r"expert arena read (unbuffered|through the file cache) \((.*)\)")
SUM_RE = re.compile(r"expert arena checksum ([0-9a-f]{16}) \(([\d.]+) GiB\)")
PSI_RE = re.compile(r"^(some|full) avg10=[\d.]+ avg60=[\d.]+ avg300=[\d.]+ total=(\d+)", re.M)

MODE_ENV = {"default": {}, "direct": {"STRATA_UNBUFFERED_LOAD": "1"}, "buffered": {"STRATA_UNBUFFERED_LOAD": "0"}}


# ------------------------------------------------------------------ pure helpers (tested without a server)
def parse_load(text: str) -> dict:
    """The engine's load lines out of a log text (the last of each kind)."""
    out: dict = {}
    for m in LOAD_RE.finditer(text):
        out.update(mode=m.group(1), threads=int(m.group(2)), gib=float(m.group(3)), load_s=float(m.group(4)),
                   gib_s=float(m.group(5)), register_s=float(m.group(6)))
    for m in READ_RE.finditer(text):
        out.update(read=m.group(1), why=m.group(2))
    for m in SUM_RE.finditer(text):
        out.update(checksum=m.group(1))
    return out


def parse_psi(text: str) -> dict:
    """/proc/pressure/memory -> {"some": total microseconds, "full": ...}."""
    return {m.group(1): int(m.group(2)) for m in PSI_RE.finditer(text)}


def read_psi() -> dict:
    try:
        return parse_psi(Path("/proc/pressure/memory").read_text())
    except OSError:
        return {}


def meminfo_gib() -> dict:
    out = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            if k in ("MemAvailable", "Cached"):
                out[k] = int(v.split()[0]) / 1048576
    except OSError:
        pass
    return out


class MemSampler(threading.Thread):
    """Samples Cached and MemAvailable every 0.5 s: the page-cache growth and the lowest available memory during the start."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()
        self.start_mi = meminfo_gib()
        self.cached_peak = 0.0
        self.avail_min = self.start_mi.get("MemAvailable", float("nan"))

    def run(self):
        while not self.stop_flag.is_set():
            mi = meminfo_gib()
            if "Cached" in mi:
                self.cached_peak = max(self.cached_peak, mi["Cached"] - self.start_mi.get("Cached", mi["Cached"]))
            if "MemAvailable" in mi:
                self.avail_min = min(self.avail_min, mi["MemAvailable"])
            self.stop_flag.wait(0.5)

    def finish(self):
        self.stop_flag.set()
        self.join(timeout=2)


def summarize(rows: list[dict]) -> list[dict]:
    """Per (mode, cache) group: n, mean / min / max start, mean load GiB/s, PSI, page-cache growth, the loader modes seen and
    whether all checksums of the whole batch agree."""
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["mode"], r["cache"]), []).append(r)
    sums = {r["checksum"] for r in rows if r.get("checksum")}
    out = []
    for (mode, cache), rs in groups.items():
        def mean(key):
            xs = [r[key] for r in rs if r.get(key) is not None]
            return statistics.fmean(xs) if xs else None
        starts = [r["start_s"] for r in rs if r.get("start_s") is not None]
        out.append({"mode": mode, "cache": cache, "n": len(rs), "start_mean_s": mean("start_s"),
                    "start_min_s": min(starts) if starts else None, "start_max_s": max(starts) if starts else None,
                    "load_gib_s": mean("gib_s"), "psi_some_s": mean("psi_some_s"), "psi_full_s": mean("psi_full_s"),
                    "cached_peak_gib": mean("cached_peak_gib"), "avail_min_gib": mean("avail_min_gib"),
                    "loader": sorted({r.get("loader") for r in rs if r.get("loader")}),
                    "checksums_agree": len(sums) <= 1})
    return sorted(out, key=lambda g: (g["cache"], g["mode"]))


def format_table(summ: list[dict]) -> str:
    f = lambda x, p=1: "n/a" if x is None else f"{x:.{p}f}"
    lines = [f"{'mode':9s} {'cache':5s} {'n':>2s} {'start mean [min-max] s':>26s} {'load GiB/s':>10s} {'psi some':>8s} {'psi full':>8s} "
             f"{'cache +GiB':>10s} {'avail min':>9s}  loader"]
    for g in summ:
        lines.append(f"{g['mode']:9s} {g['cache']:5s} {g['n']:2d} {f(g['start_mean_s']):>8s} [{f(g['start_min_s'])}-{f(g['start_max_s'])}]".ljust(48)
                     + f" {f(g['load_gib_s']):>10s} {f(g['psi_some_s'], 2):>8s} {f(g['psi_full_s'], 2):>8s} {f(g['cached_peak_gib']):>10s} "
                     f"{f(g['avail_min_gib']):>9s}  {','.join(g['loader']) or '?'}")
    if summ and not summ[0]["checksums_agree"]:
        lines.append("WARNING: the arena checksums differ between runs: a loader read different bytes")
    # the headline: direct vs buffered on a cold cache
    by = {(g["mode"], g["cache"]): g for g in summ}
    d, b = by.get(("direct", "cold")), by.get(("buffered", "cold"))
    if d and b and d["start_mean_s"] and b["start_mean_s"]:
        lines.append(f"cold start: direct {d['start_mean_s']:.1f} s, buffered {b['start_mean_s']:.1f} s "
                     f"({(1 - d['start_mean_s'] / b['start_mean_s']) * 100:+.0f}% shorter with the direct loader)")
    return "\n".join(lines)


def model_paths(cfg: dict) -> list[str]:
    """The model files and folders a config's engine arguments name (for the page-cache advice)."""
    a = cfg.get("args", [])
    out = []
    for flag in ("--native", "--ple-gguf", "--pack", "--model", "--gguf"):
        if flag in a and a.index(flag) + 1 < len(a):
            out.append(a[a.index(flag) + 1])
    # a sharded GGUF: its siblings (name-00001-of-0000N.gguf)
    sib = []
    for p in out:
        m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", p)
        if m:
            sib += [f"{m.group(1)}-{i:05d}-of-{m.group(3)}.gguf" for i in range(1, int(m.group(3)) + 1)]
    return out + sib


# ------------------------------------------------------------------ the measurement
def one_run(base: str, mode: str, cache: str, verify: bool, work: Path, port: int, ready_wait: int) -> dict:
    import abtest
    from serve import cache_drop
    arm = {"env": dict(MODE_ENV[mode])}
    if verify:
        arm["env"]["STRATA_VERIFY_ARENA"] = "1"
    cfg_path = str(work / f"{mode}-{cache}.json")
    cfg = abtest.make_config(base, arm, cfg_path)
    abtest.stop_servers()
    if cache == "cold":
        cache_drop.drop(model_paths(cfg))
    time.sleep(2)
    eng_log = Path(cfg["log"]) if cfg.get("log") else None
    off = eng_log.stat().st_size if eng_log is not None and eng_log.exists() else 0
    log_out = work / f"{mode}-{cache}.server.log"
    psi0 = read_psi()
    sampler = MemSampler()
    sampler.start()
    env = dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/core-10.0/lib", HIP_VISIBLE_DEVICES="0")
    t0 = time.time()
    with open(log_out, "w") as lf:
        srv = subprocess.Popen([str(ROOT / "tools/pancake/capped.sh"), abtest.PY, "serve/server.py", "--engine", "strata", "--config",
                                cfg_path, "--port", str(port)], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env)
    ready = None
    while time.time() - t0 < ready_wait:
        if "ready: http" in log_out.read_text(errors="replace"):
            ready = time.time() - t0
            break
        if srv.poll() is not None:
            break
        time.sleep(0.5)
    sampler.finish()
    psi1 = read_psi()
    text = log_out.read_text(errors="replace")
    if eng_log is not None and eng_log.exists():
        with open(eng_log, "rb") as fh:
            fh.seek(off)
            text += fh.read().decode("utf-8", "replace")
    abtest.stop_servers()
    row = {"mode": mode, "cache": cache, "start_s": None if ready is None else round(ready, 2),
           "cached_peak_gib": round(sampler.cached_peak, 2), "avail_min_gib": round(sampler.avail_min, 2),
           "psi_some_s": round((psi1.get("some", 0) - psi0.get("some", 0)) / 1e6, 3) if psi0 else None,
           "psi_full_s": round((psi1.get("full", 0) - psi0.get("full", 0)) / 1e6, 3) if psi0 else None}
    ld = parse_load(text)
    row.update({k: ld[k] for k in ("gib", "load_s", "gib_s", "register_s", "checksum", "threads") if k in ld})
    row["loader"] = ld.get("mode")
    row["why"] = ld.get("why")
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="result folder (rows.jsonl, logs)")
    ap.add_argument("--base", default="pancake-flashnext-iq2_xs-fast-hip.json")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--modes", default="default,direct,buffered")
    ap.add_argument("--cache", default="cold,warm")
    ap.add_argument("--no-verify", action="store_true", help="no STRATA_VERIFY_ARENA (the checksum costs a read of the whole arena)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ready-wait", type=int, default=600)
    ap.add_argument("--from-rows", help="rows.jsonl of an earlier run: print its table")
    a = ap.parse_args(argv)
    if a.from_rows:
        rows = [json.loads(x) for x in Path(a.from_rows).read_text().splitlines() if x.strip()]
        print(format_table(summarize(rows)))
        return 0
    if not a.out:
        ap.error("--out is required")
    modes = a.modes.split(",")
    caches = a.cache.split(",")
    for m in modes:
        if m not in MODE_ENV:
            ap.error(f"unknown mode {m!r} (default, direct, buffered)")
    work = Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    base = a.base if os.path.isabs(a.base) else str(ROOT / a.base)
    rows = []
    with open(work / "rows.jsonl", "a") as sink:
        for rep in range(a.reps):
            for cache in caches:
                seq = modes if rep % 2 == 0 else modes[::-1]
                for mode in seq:
                    row = one_run(base, mode, cache, not a.no_verify, work, a.port, a.ready_wait)
                    row["rep"] = rep
                    print(json.dumps(row), flush=True)
                    sink.write(json.dumps(row) + "\n")
                    sink.flush()
                    rows.append(row)
    table = format_table(summarize(rows))
    print("\n" + table)
    (work / "REPORT.txt").write_text(table + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
