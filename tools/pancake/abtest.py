"""abtest.py - pancake-strata A/B harness for the 7900 XT (MOD_PLAN phase 14: "automatic A/B harness").

Runs arms of a server config in interleaved order (A B B A ...), each arm a fresh server, and only on a quiet machine:
  - before a run: waits (if --max-load > 0) for the 1-minute load, while memory reclaim runs, and until no foreign process (not this harness, not the
    server's process tree) used more than --foreign-cpu % of a core over a 5 s sample; prints what it waited for;
  - during a run: samples the CPU time of every foreign process; a run whose foreign load averaged above
    --foreign-cpu % of a core is marked noisy, discarded and repeated (at most --retries times per run);
  - per request it reads the engine's own log line (prompt tokens, prompt tok/s, decode tok/s, drafts accepted).
Summary per arm and case: mean, 95% interval (t), difference vs the first arm with its interval, and a verdict
(better / worse / no difference).  Results also go to --out as JSON lines.

Spec (JSON):
  {"base": "pancake-flashnext-iq2_xs-rdna3test.json",
   "arms": {"base": {}, "kmiss": {"env": {"STRATA_ADAPT_KMISS": "2,8"}, "args": {"--adapt-swaps": "96"}}},
   "cases": ["short", "edit", "long"], "reps": 3}
  args: "--flag": "value" sets, "--flag": "" a bare switch, "--flag": null removes.  env: "KEY": null removes.
  Optional top-level "temperature" (default 0 = greedy), "top_p", "seed": sampling of every request, for arms that only
  matter when sampling (speculative sampling, STRATA_SPEC_PROB*).  Sampled text is not reproducible, so the text-hash
  check means nothing there.

    python tools/pancake/abtest.py SPEC.json --out bench/results/<day>-<name>.jsonl [--reps 3] [--max-load 1.0]

Test tiers (bench/TEST_PLAN.md "How every test runs"):
  --tier screen   one A B pass, cases short + edit, 2 requests per case in the same server; verdict per arm:
                  promote (>= +2% on a case, no case below -3%) or drop.  Minutes per arm.  Use it first, always.
  --tier confirm  (default) A B B A ... up to --reps, with early stop: after every full A B B A pair, stop once every
                  arm/case is decided (interval excludes 0) or its interval is below 1.5% of base (no effect worth
                  having).  --no-early-stop runs all reps.
Every run also records a hash of the generated text: an arm whose text is identical to base on every case is exact,
so it needs no quality gate.  A server that dies before ready is retried once; a second death aborts the batch
(memory fragmentation / oomd: see the free-2MiB-block line) instead of continuing with too few runs.
"""
import hashlib
import argparse
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = str(ROOT / ".venv/bin/python")
CLK = os.sysconf("SC_CLK_TCK")
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}


def prompts():
    src = (ROOT / "src/spec/draft_policy.cpp").read_text()
    files = sorted((ROOT / "src/spec").glob("*.cpp"))[:4]
    long_text = "".join(f.read_text()[:20000] for f in files)[:64000]
    return {
        "short": "Write a detailed explanation of how a hash map works, with an example in Python.",
        "edit": "Rewrite the following C++ file, renaming every variable to snake_case. Output only the full code.\n\n" + src[:6000],
        "long": "Summarize what each of these source files does, one paragraph per file.\n\n" + long_text,
    }


# ---- processes: the server's tree vs everything else
def proc_table():
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            with open(f"/proc/{d}/stat") as f:
                st = f.read()
            r = st.rindex(")")
            comm = st[st.index("(") + 1:r]
            f2 = st[r + 2:].split()
            out[int(d)] = (int(f2[1]), comm, int(f2[11]) + int(f2[12]))   # ppid, comm, utime+stime ticks
        except (OSError, ValueError, IndexError):
            pass
    return out


def tree(tab, roots):
    keep = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _, _) in tab.items():
            if ppid in keep and pid not in keep:
                keep.add(pid)
                changed = True
    return keep


def foreign_cpu(seconds, own_roots):
    """Average % of one core used by processes outside `own_roots`' trees over `seconds`, and the top users."""
    a = proc_table()
    time.sleep(seconds)
    b = proc_table()
    own = tree(b, own_roots) | {os.getpid()}
    use = {}
    for pid, (_, comm, t) in b.items():
        if pid in own or pid not in a:
            continue
        d = t - a[pid][2]
        if d > 0:
            use[comm] = use.get(comm, 0) + d
    total = sum(use.values()) / CLK / seconds * 100
    top = sorted(use.items(), key=lambda x: -x[1])[:3]
    reclaim = sum(v for c, v in use.items() if c.startswith(RECLAIM)) / CLK / seconds * 100
    foreign_cpu.reclaim = reclaim   # memory reclaim / compaction threads' share of the same sample
    return total, [(c, round(v / CLK / seconds * 100)) for c, v in top]


# kernel memory reclaim / compaction: little CPU, but a run during it decodes ~25% slower (test phase 03 / 04)
RECLAIM = ("kswapd", "kcompactd")
RECLAIM_MAX = 1.0   # % of a core: above it before a run -> wait; averaged above it during a run -> noisy, repeat


def big_free_gib():
    """GiB of free RAM in blocks of 2 MiB or more (/proc/buddyinfo, order >= 9).  The engine locks a 33 GiB arena in a
    few seconds; with this low the kernel stalls in compaction during the load (slow runs, or systemd-oomd kills the
    engine: 2026-11-10).  tools/pancake/os_tune.sh compacts / installs background compaction."""
    try:
        total = 0
        for line in open("/proc/buddyinfo"):
            f = line.split()
            if f[3] in ("Normal", "DMA32"):
                total += sum(int(n) * (4096 << o) for o, n in enumerate(f[4:]) if o >= 9)
        return total / 2**30
    except (OSError, ValueError, IndexError):
        return -1.0


def quiet_gate(max_load, max_foreign, own_roots, max_wait=3600):
    """max_load <= 0 (default): no load-average wait.  The 1-minute load right after a run is the finished server's own
    16 threads decaying (~2 min to fall below 1, 45 of test 04's 70 min) - the foreign-CPU sample already sees any
    other process, so the load wait only cost time.  Also waits while memory reclaim / compaction is running."""
    t0 = time.time()
    said = False
    big = big_free_gib()
    if 0 <= big < 20:
        print(f"  WARNING: only {big:.0f} GiB of free RAM in 2 MiB blocks (the arena locks 33 GiB): this run may be slow or "
              f"killed while loading - 'sudo tools/pancake/os_tune.sh compact' (or 'install' once)", flush=True)
    while time.time() - t0 < max_wait:
        load = os.getloadavg()[0] if max_load > 0 else 0.0
        fc, top = foreign_cpu(5, own_roots)
        rc = foreign_cpu.reclaim
        if load <= max(max_load, 0) and fc < max_foreign and rc < RECLAIM_MAX:
            if said:
                print(f"  quiet after {time.time() - t0:.0f} s", flush=True)
            return True
        if not said or int(time.time() - t0) % 120 < 6:
            print(f"  waiting for a quiet machine: load {load:.2f}, foreign CPU {fc:.0f}% {top}, reclaim {rc:.1f}%",
                  flush=True)
            said = True
    print("  machine never got quiet: running anyway (the run will likely be marked noisy)", flush=True)
    return False


def _gpu_dir():
    for d in sorted(Path("/sys/class/drm").glob("card*/device")):
        if (d / "pp_dpm_sclk").exists() and list(d.glob("hwmon/hwmon*")):
            return d
    return None


GPU_DIR = _gpu_dir()
IGPU_DIR = next((d for d in sorted(Path("/sys/class/drm").glob("card*/device"))
                 if d != GPU_DIR and (d / "gpu_busy_percent").exists()), None)
CPU_FREQ = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"))
K10 = next((h for h in Path("/sys/class/hwmon").glob("hwmon*") if (h / "name").read_text().strip() == "k10temp"), None)


def host_sample():
    """APU side (Ryzen 7 5700G): the iGPU drives the desktop / remote desktop and shares the package power and the
    DDR4 with the CPU expert threads - mean core clock (MHz), iGPU busy %, iGPU video-engine busy %, fabric clock, Tctl."""
    def num(p, div=1):
        try:
            return int(p.read_text().split()[0]) / div
        except (OSError, ValueError, IndexError):
            return 0.0
    out = {"cpu_mhz": round(sum(num(f, 1000) for f in CPU_FREQ) / max(1, len(CPU_FREQ))),
           "tctl": num(K10 / "temp1_input", 1000) if K10 else 0.0}
    if IGPU_DIR is not None:
        out["igpu_busy"] = num(IGPU_DIR / "gpu_busy_percent")
        out["igpu_vcn"] = num(IGPU_DIR / "vcn_busy_percent")
        try:
            out["fclk"] = next(int("".join(c for c in l.split(":")[1] if c.isdigit()))
                               for l in (IGPU_DIR / "pp_dpm_fclk").read_text().splitlines() if l.strip().endswith("*"))
        except (OSError, StopIteration, ValueError):
            out["fclk"] = 0
    return out


def gpu_sample():
    """GPU telemetry (amdgpu sysfs): current shader / memory clock (MHz), edge / junction / memory temperature (C),
    power (W), PCIe link speed.  Test 04b showed bimodal runs (~25% slower decode AND prompt read, no foreign CPU,
    no reclaim): this tells a clock / thermal / link state apart from a host-side cause."""
    if GPU_DIR is None:
        return {}
    def rd(n):
        try:
            return (GPU_DIR / n).read_text()
        except OSError:
            return ""
    def cur_mhz(n):
        for line in rd(n).splitlines():
            if line.strip().endswith("*"):
                return int("".join(ch for ch in line.split(":")[1] if ch.isdigit()) or 0)
        return 0
    h = next(GPU_DIR.glob("hwmon/hwmon*"))
    def hw(n, div):
        try:
            return int((h / n).read_text()) / div
        except (OSError, ValueError):
            return 0.0
    return {"sclk": cur_mhz("pp_dpm_sclk"), "mclk": cur_mhz("pp_dpm_mclk"), "t_edge": hw("temp1_input", 1000),
            "t_junc": hw("temp2_input", 1000), "t_mem": hw("temp3_input", 1000), "watt": hw("power1_average", 1e6),
            "link": rd("current_link_speed").strip(),
            "vram_mib": int(rd("mem_info_vram_used") or 0) >> 20, "gtt_mib": int(rd("mem_info_gtt_used") or 0) >> 20}


class NoiseMonitor(threading.Thread):
    def __init__(self, own_roots):
        super().__init__(daemon=True)
        self.own_roots, self.samples, self.stop_ev = own_roots, [], threading.Event()
        self.gpu = []

    def gpu_summary(self):
        """Busy samples only (sclk above idle): mean shader / memory clock, max temperatures, mean power, link."""
        busy = [g for g in self.gpu if g and g["sclk"] > 500] or [g for g in self.gpu if g]
        if not busy:
            return {}
        mean = lambda k: round(sum(g[k] for g in busy) / len(busy))
        top = lambda k: round(max(g[k] for g in busy))
        return {"sclk": mean("sclk"), "mclk": mean("mclk"), "t_junc_max": top("t_junc"), "t_mem_max": top("t_mem"),
                "watt": mean("watt"), "link": sorted({g["link"] for g in busy}), "n": len(busy),
                "vram_mib_max": top("vram_mib"), "gtt_mib_max": top("gtt_mib"),
                "cpu_mhz": mean("cpu_mhz"), "tctl_max": top("tctl"),
                **({"igpu_busy": mean("igpu_busy"), "igpu_vcn": mean("igpu_vcn"), "fclk_min": min(g["fclk"] for g in busy)}
                   if "igpu_busy" in busy[0] else {})}   # GTT growth = VRAM evicted to RAM

    def run(self):
        while not self.stop_ev.is_set():
            fc, top = foreign_cpu(3, self.own_roots)
            self.samples.append((fc, top, foreign_cpu.reclaim))
            self.gpu.append({**gpu_sample(), **host_sample()})

    def result(self):
        self.stop_ev.set()
        self.join(timeout=10)
        if not self.samples:
            self.reclaim = 0.0
            return 0.0, []
        avg = sum(s[0] for s in self.samples) / len(self.samples)
        self.reclaim = sum(s[2] for s in self.samples) / len(self.samples)
        worst = max(self.samples, key=lambda s: s[0])
        return avg, worst[1]


# ---- server runs
def make_config(base, arm, path):
    c = json.load(open(base))
    for k, v in (arm.get("env") or {}).items():
        if v is None:
            c["env"].pop(k, None)
        else:
            c["env"][k] = str(v)
    a = c["args"]
    for k, v in (arm.get("args") or {}).items():
        if k in a:
            i = a.index(k)
            n = 2 if i + 1 < len(a) and not a[i + 1].startswith("--") else 1
            del a[i:i + n]
        if v is not None:
            a += [k] + ([str(v)] if v != "" else [])
    if arm.get("drafting"):
        c["drafting"] = arm["drafting"]
    if arm.get("exe"):   # another build of the engine (e.g. build-hip-native/strata)
        exe = arm["exe"] if os.path.isabs(arm["exe"]) else str(ROOT / arm["exe"])
        c["exe"] = exe
    json.dump(c, open(path, "w"), indent=1)
    return c


def stop_servers():
    for pid, (_, comm, _) in proc_table().items():
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if comm.startswith("python") and "server.py --engine strata" in cmd and pid != os.getpid():
            try:
                os.kill(pid, 15)
            except OSError:
                pass
    for _ in range(60):
        if not any(c == "strata" for _, c, _ in proc_table().values()):
            return
        time.sleep(2)


SAMPLING = {"temperature": 0}   # the spec's "temperature" / "top_p" / "seed" keys replace it (see main)


def request(url, text):
    """Returns the generated text."""
    body = {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": 512, **SAMPLING,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    try:
        return r["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


LINE = re.compile(r"prompt (\d+) tokens.*?read in \d+ ms \(([\d.]+) tok/s\), (\d+) generated in \d+ ms \(([\d.]+) tok/s\)"
                  r"(?:, drafts accepted (\d+) of (\d+))?")


def start_server(cfg_path, log_out, port, ready_wait=900):
    """Starts the engine server for a config and waits for its "ready: http" line.  Returns (process, None), or
    (None, "server died before ready" / "server not ready ...") - the caller stops the servers."""
    env = dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/core-10.0/lib", HIP_VISIBLE_DEVICES="0")
    with open(log_out, "w") as lf:
        srv = subprocess.Popen([str(ROOT / "tools/pancake/capped.sh"), PY, "serve/server.py", "--engine", "strata", "--config",
                                cfg_path, "--port", str(port)], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    while time.time() - t0 < ready_wait:
        txt = Path(log_out).read_text(errors="replace")
        if "ready: http" in txt:
            return srv, None
        if srv.poll() is not None or "EngineDied" in txt:
            return None, "server died before ready"
        time.sleep(3)
    return srv, None   # as before: not ready after ready_wait, carry on (the first request waits for it)


def run_arm(name, cfg_path, cfg, cases, P, log_out, port, per_run=1):
    log = cfg.get("log")
    srv, err = start_server(cfg_path, log_out, port)
    if err:
        return None, err
    url = f"http://127.0.0.1:{port}"
    request(url, P["short"])                          # warm-up, not counted
    n0 = len(Path(log).read_text(errors="replace").splitlines()) if log and Path(log).exists() else 0
    mon = NoiseMonitor([srv.pid])
    mon.start()
    texts = {}
    for c in cases:
        for _ in range(per_run):
            texts[c] = hashlib.sha1(request(url, P[c]).encode()).hexdigest()[:16]
    noise, top = mon.result()
    rows = []
    if log and Path(log).exists():
        for line in Path(log).read_text(errors="replace").splitlines()[n0:]:
            m = LINE.search(line)
            if m and "strata serve: prompt" in line:
                acc = int(m[5]) / int(m[6]) if m[5] and int(m[6]) > 0 else None
                rows.append({"prompt": int(m[1]), "prompt_tps": float(m[2]), "gen": int(m[3]), "decode_tps": float(m[4]),
                             "accept": acc})
    stop_servers()
    want = len(cases) * per_run
    if len(rows) > want:   # an extra engine line (a retried / re-read request): the last ones are the cases'
        rows = rows[-want:]
    if len(rows) != want:
        return None, f"expected {want} engine lines, got {len(rows)}"
    for c, r in zip([c for c in cases for _ in range(per_run)], rows):
        r["case"] = c
    return {"rows": rows, "texts": texts, "noise": noise, "noise_top": top, "reclaim": round(mon.reclaim, 2), "gpu": mon.gpu_summary(),
            "big_free_gib": round(big_free_gib(), 1)}, None


def ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, T95.get(n - 1, 2.0) * sd / math.sqrt(n)


def decided(results, arms, cases, floor=1.5):
    """True when every arm/case either has an interval that excludes 0 or one narrower than floor % of base."""
    base = arms[0]
    for c in cases:
        key = "prompt_tps" if c == "long" else "decode_tps"
        vals = {a: [r[key] for res in results.get(a, []) for r in res["rows"] if r["case"] == c] for a in arms}
        if len(vals[base]) < 2:
            return False
        bm = sum(vals[base]) / len(vals[base])
        for a in arms[1:]:
            n = min(len(vals[a]), len(vals[base]))
            if n < 2:
                return False
            dm, dh = ci([x - y for x, y in zip(vals[a][:n], vals[base][:n])])
            if not (abs(dm) > dh or dh / bm * 100 < floor):
                return False
    return True


def screen_verdict(results, arms, cases):
    base = arms[0]
    print("\nscreen verdict (promote: >= +2% on a case and no case below -3%; else drop)")
    for a in arms[1:]:
        deltas = {}
        for c in cases:
            key = "prompt_tps" if c == "long" else "decode_tps"
            b = [r[key] for res in results.get(base, []) for r in res["rows"] if r["case"] == c]
            x = [r[key] for res in results.get(a, []) for r in res["rows"] if r["case"] == c]
            if b and x:
                deltas[c] = (sum(x) / len(x)) / (sum(b) / len(b)) * 100 - 100
        ok = deltas and max(deltas.values()) >= 2.0 and min(deltas.values()) > -3.0
        print(f"  {a:20s} " + "  ".join(f"{c} {d:+.1f}%" for c, d in deltas.items()) + f"  -> {'PROMOTE to confirm' if ok else 'drop'}")


def text_check(results, arms, cases):
    base = arms[0]
    bt = [res.get("texts", {}) for res in results.get(base, [])]
    if not bt:
        return
    print("\ngenerated text vs base (identical = exact: no quality gate needed)")
    for a in arms[1:]:
        at = [res.get("texts", {}) for res in results.get(a, [])]
        if not at:
            continue
        same = all(t.get(c) == bt[0].get(c) for t in at for c in cases)
        print(f"  {a:20s} {'identical on every case -> exact' if same else 'DIFFERENT text -> lossy or placement-dependent: quality gate'}")


def summary(results, arms, cases):
    base = arms[0]
    print("\nsummary (decode tok/s; prompt tok/s for 'long'; mean +- 95% interval over clean runs)")
    for c in cases:
        key = "prompt_tps" if c == "long" else "decode_tps"
        vals = {a: [r[key] for res in results.get(a, []) for r in res["rows"] if r["case"] == c] for a in arms}
        bm, _ = ci(vals[base]) if vals[base] else (float("nan"), 0)
        for a in arms:
            if not vals[a]:
                print(f"  {c:6s} {a:20s} no clean runs")
                continue
            m, h = ci(vals[a])
            line = f"  {c:6s} {a:20s} {m:8.1f} +- {h:5.1f}  (n={len(vals[a])})"
            if a != base and vals[base]:
                n = min(len(vals[a]), len(vals[base]))
                d = [x - y for x, y in zip(vals[a][:n], vals[base][:n])]   # paired by repetition (same ABBA slot)
                dm, dh = ci(d)
                verdict = "better" if dm - dh > 0 else "worse" if dm + dh < 0 else "no difference"
                line += f"   vs {base}: {dm / bm * 100:+5.1f}% +- {dh / bm * 100:4.1f}%  -> {verdict}"
            print(line)
            if c != "long":
                accs = [r["accept"] for res in results.get(a, []) for r in res["rows"] if r["case"] == c and r["accept"]]
                if accs:
                    print(f"  {'':6s} {'':20s} drafts accepted {sum(accs) / len(accs):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int)
    ap.add_argument("--max-load", type=float, default=0.0, help="1-minute load to wait for (0 = off)")
    ap.add_argument("--foreign-cpu", type=float, default=25.0, help="%% of one core: above it a machine is not quiet")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--tier", choices=("screen", "confirm"), default="confirm")
    ap.add_argument("--no-early-stop", action="store_true")
    ap.add_argument("--max-wait", type=int, default=600, help="seconds to wait for a quiet machine before running anyway")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    arms = list(spec["arms"])
    cases = spec.get("cases", ["short", "edit", "long"])
    reps = a.reps or spec.get("reps", 3)
    per_run = 1
    if a.tier == "screen":
        cases = [c for c in cases if c != "long"] or ["short"]
        reps, per_run = 1, 2
    base = str(ROOT / spec["base"]) if not os.path.isabs(spec["base"]) else spec["base"]
    for k in ("temperature", "top_p", "seed"):   # optional sampling, for arms that only matter at T > 0
        if spec.get(k) is not None:
            SAMPLING[k] = spec[k]
    if SAMPLING["temperature"] > 0:
        print(f"sampling: {SAMPLING} (sampled text is not reproducible: the text-hash check below is not meaningful)",
              flush=True)
    P = prompts()
    work =Path(a.out).with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    order = []
    for r in range(reps):   # A B B A, then B A A B, ... : each arm sees early and late slots equally
        seq = arms if r % 2 == 0 else arms[::-1]
        order += [(r, x) for x in seq]
    results = {}
    out = open(a.out, "a")
    stop_servers()
    deaths = 0
    for i, (r, arm) in enumerate(order):
        if (not a.no_early_stop and a.tier == "confirm" and r >= 2 and r % 2 == 0 and order[i - 1][0] != r
                and decided(results, arms, cases)):
            print(f"early stop after {r} reps: every arm/case decided or below 1.5% of base", flush=True)
            break
        for attempt in range(a.retries + 1):
            quiet_gate(a.max_load, a.foreign_cpu, [], a.max_wait)
            cfg_path = str(work / f"{arm}.json")
            cfg = make_config(base, spec["arms"][arm], cfg_path)
            res, err = run_arm(arm, cfg_path, cfg, cases, P, str(work / f"{arm}-r{r}-a{attempt}.out"), a.port, per_run)
            if err:
                print(f"rep {r} {arm}: FAILED ({err})", flush=True)
                if "died" in err:
                    deaths += 1
                    if deaths >= 2:
                        print(f"ABORT: the server died twice before ready (free RAM in 2 MiB blocks: {big_free_gib():.0f} GiB;"
                              f" fix the machine first: 'sudo tools/pancake/os_tune.sh compact'). Partial results below.",
                              flush=True)
                        summary(results, arms, cases)
                        return 2
                    continue   # one retry
                break
            clean = res["noise"] <= a.foreign_cpu and res["reclaim"] <= RECLAIM_MAX
            rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "rep": r, "arm": arm, "attempt": attempt, "clean": clean, **res}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            desc = " ".join(f"{x['case']} {x['decode_tps']:.1f}" + (f"/{x['prompt_tps']:.0f}p" if x["case"] == "long" else "")
                            for x in res["rows"])
            g = res.get("gpu") or {}
            gd = (f"  gpu {g['sclk']}/{g['mclk']} MHz {g['t_junc_max']}C {g['watt']}W gtt {g.get('gtt_mib_max', 0)}M"
                  f" cpu {g.get('cpu_mhz', 0)}MHz {g.get('tctl_max', 0)}C igpu {g.get('igpu_busy', 0)}%/vcn {g.get('igpu_vcn', 0)}%"
                  f" fclk {g.get('fclk_min', 0)}") if g else ""
            print(f"rep {r} {arm:20s} {desc}  foreign CPU {res['noise']:.0f}% {res['noise_top']}{gd}"
                  f"{'' if clean else '  NOISY -> repeat'}", flush=True)
            if clean:
                results.setdefault(arm, []).append(res)
                break
    summary(results, arms, cases)
    text_check(results, arms, cases)
    if a.tier == "screen":
        screen_verdict(results, arms, cases)


if __name__ == "__main__":
    sys.exit(main())
