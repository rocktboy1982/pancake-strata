"""abtest.py - pancake-strata A/B harness for the 7900 XT (MOD_PLAN phase 14: "automatic A/B harness").

Runs arms of a server config in interleaved order (A B B A ...), each arm a fresh server, and only on a quiet machine:
  - before a run: waits until the 1-minute load is below --max-load and no foreign process (not this harness, not the
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

    python tools/pancake/abtest.py SPEC.json --out bench/results/<day>-<name>.jsonl [--reps 3] [--max-load 1.0]
"""
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
    return total, [(c, round(v / CLK / seconds * 100)) for c, v in top]


def quiet_gate(max_load, max_foreign, own_roots, max_wait=3600):
    t0 = time.time()
    said = False
    while time.time() - t0 < max_wait:
        load = os.getloadavg()[0]
        fc, top = foreign_cpu(5, own_roots)
        if load < max_load and fc < max_foreign:
            if said:
                print(f"  quiet after {time.time() - t0:.0f} s", flush=True)
            return True
        if not said or int(time.time() - t0) % 120 < 6:
            print(f"  waiting for a quiet machine: load {load:.2f}, foreign CPU {fc:.0f}% {top}", flush=True)
            said = True
    print("  machine never got quiet: running anyway (the run will likely be marked noisy)", flush=True)
    return False


class NoiseMonitor(threading.Thread):
    def __init__(self, own_roots):
        super().__init__(daemon=True)
        self.own_roots, self.samples, self.stop_ev = own_roots, [], threading.Event()

    def run(self):
        while not self.stop_ev.is_set():
            fc, top = foreign_cpu(3, self.own_roots)
            self.samples.append((fc, top))

    def result(self):
        self.stop_ev.set()
        self.join(timeout=10)
        if not self.samples:
            return 0.0, []
        avg = sum(s[0] for s in self.samples) / len(self.samples)
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


def request(url, text):
    body = {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": 512, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


LINE = re.compile(r"prompt (\d+) tokens.*?read in \d+ ms \(([\d.]+) tok/s\), (\d+) generated in \d+ ms \(([\d.]+) tok/s\)"
                  r"(?:, drafts accepted (\d+) of (\d+))?")


def run_arm(name, cfg_path, cfg, cases, P, log_out, port):
    log = cfg.get("log")
    env = dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/core-10.0/lib", HIP_VISIBLE_DEVICES="0")
    with open(log_out, "w") as lf:
        srv = subprocess.Popen([str(ROOT / "tools/pancake/capped.sh"), PY, "serve/server.py", "--engine", "strata", "--config",
                                cfg_path, "--port", str(port)], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env)
    t0 = time.time()
    while time.time() - t0 < 900:
        txt = Path(log_out).read_text(errors="replace")
        if "ready: http" in txt:
            break
        if srv.poll() is not None or "EngineDied" in txt:
            return None, "server died before ready"
        time.sleep(3)
    url = f"http://127.0.0.1:{port}"
    request(url, P["short"])                          # warm-up, not counted
    n0 = len(Path(log).read_text(errors="replace").splitlines()) if log and Path(log).exists() else 0
    mon = NoiseMonitor([srv.pid])
    mon.start()
    for c in cases:
        request(url, P[c])
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
    if len(rows) != len(cases):
        return None, f"expected {len(cases)} engine lines, got {len(rows)}"
    for c, r in zip(cases, rows):
        r["case"] = c
    return {"rows": rows, "noise": noise, "noise_top": top}, None


def ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, T95.get(n - 1, 2.0) * sd / math.sqrt(n)


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
    ap.add_argument("--max-load", type=float, default=1.0)
    ap.add_argument("--foreign-cpu", type=float, default=25.0, help="% of one core: above it a machine is not quiet")
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    arms = list(spec["arms"])
    cases = spec.get("cases", ["short", "edit", "long"])
    reps = a.reps or spec.get("reps", 3)
    base = str(ROOT / spec["base"]) if not os.path.isabs(spec["base"]) else spec["base"]
    P = prompts()
    work = Path(a.out).with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    order = []
    for r in range(reps):   # A B B A, then B A A B, ... : each arm sees early and late slots equally
        seq = arms if r % 2 == 0 else arms[::-1]
        order += [(r, x) for x in seq]
    results = {}
    out = open(a.out, "a")
    stop_servers()
    for r, arm in order:
        for attempt in range(a.retries + 1):
            quiet_gate(a.max_load, a.foreign_cpu, [])
            cfg_path = str(work / f"{arm}.json")
            cfg = make_config(base, spec["arms"][arm], cfg_path)
            res, err = run_arm(arm, cfg_path, cfg, cases, P, str(work / f"{arm}-r{r}-a{attempt}.out"), a.port)
            if err:
                print(f"rep {r} {arm}: FAILED ({err})", flush=True)
                break
            clean = res["noise"] <= a.foreign_cpu
            rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "rep": r, "arm": arm, "attempt": attempt, "clean": clean, **res}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            desc = " ".join(f"{x['case']} {x['decode_tps']:.1f}" + (f"/{x['prompt_tps']:.0f}p" if x["case"] == "long" else "")
                            for x in res["rows"])
            print(f"rep {r} {arm:20s} {desc}  foreign CPU {res['noise']:.0f}% {res['noise_top']}"
                  f"{'' if clean else '  NOISY -> repeat'}", flush=True)
            if clean:
                results.setdefault(arm, []).append(res)
                break
    summary(results, arms, cases)


if __name__ == "__main__":
    sys.exit(main())
