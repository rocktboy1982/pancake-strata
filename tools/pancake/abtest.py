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
  cases: short, edit (decode tok/s); long (~12K tokens), long32k, long64k (prompt tok/s, see below); longdec32k,
  longdec64k, longdec128k (decode tok/s AFTER a long cold prompt, see below); long1k, long3k (prompt tok/s of a cold
  1000- / 3000-token prompt, A30), longdec10k (decode after a 10K read, A49).  "prefill": true marks a spec whose arms
  change prompt reading (see --tier screen).
  args: "--flag": "value" sets, "--flag": "" a bare switch, "--flag": null removes.  env: "KEY": null removes.
  Optional top-level "temperature" (default 0 = greedy), "top_p", "seed": sampling of every request, for arms that only
  matter when sampling (speculative sampling, STRATA_SPEC_PROB*).  Sampled text is not reproducible, so the text-hash
  check means nothing there.

    python tools/pancake/abtest.py SPEC.json --out bench/results/<day>-<name>.jsonl [--reps 3] [--max-load 1.0]

Test tiers (bench/TEST_PLAN.md "How every test runs"):
  --tier screen   one A B pass, cases short + edit, 2 requests per case in the same server; verdict per arm:
                  promote (>= +2% on a case, no case below -3%) or drop.  Minutes per arm.  Use it first, always.
                  The long* cases are dropped (tens of seconds per read, and the plain screens judge decode), except the longdec*
                  ones (they judge decode; one request each), unless the spec sets "prefill": true:
                  then every case of the spec is kept, the long* ones with ONE request each (short / edit keep two), and
                  the long* cases are compared on prompt tok/s with the same promote / drop thresholds.
  --tier confirm  (default) A B B A ... up to --reps, with early stop: after every full A B B A pair, stop once every
                  arm/case is decided (interval excludes 0) or its interval is below 1.5% of base (no effect worth
                  having).  --no-early-stop runs all reps.
Prefill cases (long, long32k, long64k; metric = prompt tok/s of the engine's own log line): the prompt is source text
from src/ (long: src/spec, ~12K tokens, content unchanged; long32k / long64k: the first non-test src/**/*.cpp files in
sorted order, sized to 32000 / 64000 tokens at LONG_CPT chars per token).  Prompt reading is always COLD: every long*
request starts with a unique line ("[run <batch nonce>-<n>]") at the very first characters of the user message, so no
prompt cache (in memory, or on disk across server restarts) can share more than the few template tokens before it:
repeats and arms never read a warm prompt.  Consequence: the text hash of a long* case is not comparable between arms
and is not checked (short / edit still are).  long32k / long64k request max_tokens 32 (prefill only).  The server's
count_tokens endpoint is asked once per server for the prompt size of each long* case before the first request, and
the engine's reported prompt tokens are printed per run (a warning if more than 8% off the target).  Needs
--max-context above 64K for long64k (the base configs use 131072).
Long-context decode cases (longdec32k, longdec64k, longdec128k; MOD_PLAN F11): the same kind of cold source-text prompt
as long32k / long64k (32000 / 64000 tokens; longdec128k reads 120000 so that the prompt plus the 256-token reply stays
under --max-context 131072 even when LONG_CPT is 8% off), then a 256-token reply; metric = the decode tok/s of that
reply (the engine's own log line), so the attention over the long cache is in every token.  One request per server,
never read warm, text not compared (the unique first line), like the other long* cases.
Every run also records a hash of the generated text: an arm whose text is identical to base on every case is exact,
so it needs no quality gate.  A server that dies before ready is retried once; a second death aborts the batch
(memory fragmentation / oomd: see the free-2MiB-block line) instead of continuing with too few runs.
Faster confirm runs (MOD_PLAN "Testing: why 45-minute windows": about 90% of an attempt is the engine start, and 48% of the
2026-11-11 attempts were discarded by the noise proxies although they decoded as fast as the kept ones).  All off by default
(then the runs, the rows and the summary are as before):
  --block N         N passes (every case once) per engine start; an arm's config is fixed per start.  Starts in A B B A order,
                    max(2, ceil(reps / N)) starts per arm: at least 2, so start-to-start variance stays in the interval (the
                    summary pairs pass k of a start with pass k of the other arm's start in the same round, and widens its
                    intervals by the design effect of passes that share a start, see BLOCK_ICC).  Pass k >= 1 of a
                    start prefixes the short / edit prompt with "[pass k]" so it is read cold like pass 0: the same prompt again in
                    the same server comes from the conversation cache, and the edit then decoded ~35% faster (2026-11-11 screens,
                    second request: 76% -> 88% expert cache hits).  The text hash of a case is the last pass's (the same pass index
                    in every start, so arms stay comparable).
  --warmup W        W throwaway passes per start in place of the single short warm-up request; only the passes after them are
                    measured (with --block, --warmup 1 gives every measured pass the same history).  The throwaway rows go to the
                    attempt's "warmup_rows"; with --block or --warmup every row carries "pass" and "warm" (a full pass ran before
                    it in the same start).
  --outlier robust  no attempt is repeated because a proxy (foreign CPU, reclaim) crossed its threshold ("clean" still records
                    the proxies' verdict).  Judged by the outcome instead: a start more than SLOW_START (8.75%) below its arm's
                    reference start on a case (the faster of two, the median from three on) is a slow engine state - whole
                    starts 10-35% slow, 7 of the 60 attempts of 2026-11-11, with or without proxy noise - and its slot is run
                    again in the same round (up to --retries; with 2 starts per arm no median can tell which one is slow).
                    Then the summary drops only a pass (without --block: the attempt) in which a measured
                    request's metric is a robust outlier among the same arm, case and request slot:
                    |x - median| > ROBUST_Z x max(1.4826 MAD, ROBUST_FLOOR x median), with at least ROBUST_MIN values (the modified
                    z-score of Iglewicz and Hoaglin, 3.5; the 2.5% floor keeps a quiet arm's spread, see ROBUST_FLOOR).  Confirm tier
                    only: a screen has one or two values per arm, too few to judge (replayed on the 2026-11-11 screens it kept a
                    start that decoded 24% slow).  Default: the gate.
Every attempt row also records "tier" (arena / mmap / unknown) with "tier_source" (engine: its start-up lines; capped:
capped.sh's "capped: tier=..." guess; guess: capped.sh's rule on keep_gib and big_free_gib_start; see tier_of), "keep_gib"
(the STRATA_RAM_KEEP_GIB the engine used), "big_free_gib_start" and "opts" (the abtest options), so that promote.py can
refuse mixed-tier or mixed-option comparisons.  select() decides which attempts and requests a summary uses (tools/pancake/
abtest_replay.py feeds recorded jsonl files through it).
Safety (other sessions share this machine and its one GPU): abtest stops only the servers IT started (the Popen it created and the engine
below it, never found by name), so a second abtest's, another session's or a crashed run's leftover server is not touched.  It refuses to
start (exit 3, naming the processes) while another real abtest / restore_check / engine runs - the same anchored test as tools/pancake/
queue_run.sh's BUSY pattern - or while a `serve/server.py --engine strata` is up that it did not start (a leftover is reported, not killed:
stop it yourself).  --force-busy starts anyway (for the fake-engine tests: two fake abtests side by side).
"""
import hashlib
import argparse
import json
import math
import os
import re
import signal
import statistics
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


LONG_CPT = 2.85                                          # chars per token of this repo's C++ (Qwen tokenizer; 'long': 34264 chars = 11978 tokens)
LONG_TARGET = {"long32k": 32000, "long64k": 64000,       # prompt tokens (fit the 131072 context; 32K stays under --kv-resident 32768)
               "long1k": 1000, "long3k": 3000,           # A30: agent-sized prompt deltas (the CPU-share window), prompt tok/s
               "longdec10k": 10000,                      # A49: decode after a 10K read (attention share of the window vs 64K / 128K)
               "longdec32k": 32000, "longdec64k": 64000, "longdec128k": 120000}   # decode after the read (F11)
LONG_DEC = ("longdec10k", "longdec32k", "longdec64k", "longdec128k")   # long* cases judged on the decode tok/s of the reply after the read
MAXTOK = {"long32k": 32, "long64k": 32, "long1k": 32, "long3k": 32, "longdec10k": 256,                # prefill-only cases: no 512-token decode at 32K / 64K context
          "longdec32k": 256, "longdec64k": 256, "longdec128k": 256}
SUMMARIZE = "Summarize what each of these source files does, one paragraph per file.\n\n"


def is_long(c):
    """A prefill case: judged on prompt tok/s, one request per server, unique cold prefix per request."""
    return c.startswith("long")


def metric(c):
    return "prompt_tps" if is_long(c) and c not in LONG_DEC else "decode_tps"


def corpus(chars):
    """Deterministic source text of `chars` characters: src/**/*.cpp (not *_test.cpp) in sorted order, each file cut at
    20000 characters and headed by its path."""
    out, n = [], 0
    for f in sorted((ROOT / "src").rglob("*.cpp")):
        if f.name.endswith("_test.cpp"):
            continue
        part = f"// ==== {f.relative_to(ROOT)}\n" + f.read_text(errors="replace")[:20000] + "\n"
        out.append(part)
        n += len(part)
        if n >= chars:
            return "".join(out)[:chars]
    raise SystemExit(f"abtest: src/ has only {n} characters of non-test .cpp, {chars} needed for the long* cases")


def prompts(cases=()):
    src = (ROOT / "src/spec/draft_policy.cpp").read_text()
    files = sorted((ROOT / "src/spec").glob("*.cpp"))[:4]
    long_text = "".join(f.read_text()[:20000] for f in files)[:64000]
    P = {
        "short": "Write a detailed explanation of how a hash map works, with an example in Python.",
        "edit": "Rewrite the following C++ file, renaming every variable to snake_case. Output only the full code.\n\n" + src[:6000],
        "long": SUMMARIZE + long_text,
    }
    want = [c for c in cases if c in LONG_TARGET]   # built only when a spec asks for them (up to 190K characters)
    if want:
        big = corpus(int(max(LONG_TARGET[c] for c in want) * LONG_CPT))
        for c in want:
            P[c] = SUMMARIZE + big[:int(LONG_TARGET[c] * LONG_CPT)]
    return P


NONCE = f"{os.getpid():x}{time.time_ns() // 1000:x}"
_SEQ = [0]


def case_text(P, c, k=0):
    """The prompt text of one request of case c in pass k of a start.  A long* case gets a unique first line, so its prompt
    is always read cold (never from a prompt cache): the id is never reused within a batch, across arms, reps or servers.
    short / edit: pass 0 as is, pass k >= 1 with a "[pass k]" first line (read cold in the same server, the same text in
    every start)."""
    if not is_long(c):
        return P[c] if k == 0 else f"[pass {k}]\n" + P[c]
    _SEQ[0] += 1
    return f"[run {NONCE}-{_SEQ[0]}]\n" + P[c]


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
    for k, v in (arm.get("config") or {}).items():   # server config keys (reasoning_budget_by_effort, rumination_close ...)
        if v is None:
            c.pop(k, None)
        else:
            c[k] = v
    if arm.get("exe"):   # another build of the engine (e.g. build-hip-native/strata)
        exe = arm["exe"] if os.path.isabs(arm["exe"]) else str(ROOT / arm["exe"])
        c["exe"] = exe
    json.dump(c, open(path, "w"), indent=1)
    return c


OWN = []   # the server processes (Popen) this abtest started: stop_servers() ends these and what runs below them, nothing else
# "Another test or engine is running": the SAME anchored test as tools/pancake/queue_run.sh's BUSY pattern (test_abtest_core pins that it is the
# same string) - a real python running abtest.py / restore_check.py, or the build-hip engine binary - not a shell whose command line only
# mentions them.  SERVER adds the Python server of an engine, the process a crashed run leaves behind.  Both are matched against the command
# line with the arguments joined by single spaces, as pgrep -f does.
BUSY = r"^[^ ]*python[0-9.]* ([^ ]+ )*[^ ]*tools/pancake/(abtest|restore_check)\.py( |$)|^[^ ]*/build-hip/strata( |$)"
SERVER = r"^[^ ]*python[0-9.]* ([^ ]+ )*[^ ]*serve/server\.py ([^ ]+ )*--engine strata( |$)"


def busy_processes():
    """(pid, command line) of the processes that make a start unsafe: a real abtest / restore_check / engine, or a server, that this
    abtest did not start (itself and what it started are skipped)."""
    tab = proc_table()
    mine, found, busy = tree(tab, [os.getpid()]), [], re.compile(f"{BUSY}|{SERVER}")
    for pid in sorted(set(tab) - mine):
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace").rstrip()
        except OSError:
            continue
        if busy.search(cmd):
            found.append((pid, cmd))
    return found


def running(pid):
    try:
        st = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return st[st.rindex(")") + 2] not in "ZX"   # a zombie is over


def stop_servers(wait=120):
    """Ends the servers THIS abtest started and waits until they and the engines below them are gone (at most `wait` s, then the server is
    killed).  A signal goes only to a Popen of ours, never to a process found by name: another abtest's server, another session's server
    and a crashed run's leftover are left alone (before 2026-11-11 it ended every `server.py --engine strata` of the machine and waited
    for any process named strata)."""
    servers = [p for p in OWN if p.poll() is None]
    below = tree(proc_table(), [p.pid for p in servers]) - {p.pid for p in servers} if servers else set()   # the engine exe
    t0 = time.time()
    for p in servers:
        p.terminate()   # the server stops its engine and exits
    for p in servers:
        try:
            p.wait(max(1, wait - (time.time() - t0)))
        except subprocess.TimeoutExpired:
            print(f"  WARNING: server pid {p.pid} still running {wait} s after SIGTERM: killing it", flush=True)
            p.kill()
            p.wait()
    OWN.clear()
    while time.time() - t0 < wait and any(running(pid) for pid in below):
        time.sleep(2)
    for pid in sorted(x for x in below if running(x)):
        print(f"  WARNING: engine pid {pid} of this abtest's server still running {wait} s after it", flush=True)


SAMPLING = {"temperature": 0}   # the spec's "temperature" / "top_p" / "seed" keys replace it (see main)


def request(url, text, max_tokens=512):
    """Returns the generated text."""
    body = {"model": "x", "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, **SAMPLING,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    try:
        return r["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


LINE = re.compile(r"prompt (\d+) tokens.*?read in \d+ ms \(([\d.]+) tok/s\), (\d+) generated in \d+ ms \(([\d.]+) tok/s\)"
                  r"(?:, drafts accepted (\d+) of (\d+))?")
KEEP_LINE = re.compile(r"--ram-keep-gib ([\d.]+):")
CAPPED_LINE = re.compile(r"capped: tier=(\w+) keep_gib=([\d.]+)")   # capped.sh's guess before the start (stderr -> the .out)
ARENA_KEEP_MAX, ARENA_BIG_GIB = 12, 33   # capped.sh's rule: the arena only with keep <= 12 GiB and >= 33 GiB free in 2 MiB blocks


def tier_of(text, keep=None, big=-1.0):
    """(tier, source) of one engine start, from its start-up lines (engine log) and the .out (server narration, capped.sh's
    stderr).  source "engine": what the engine did - "mmap" when it says "experts via mmap" (also after "not enough RAM for
    the expert arena": STRATA_RAM_KEEP_GIB above what the arena leaves, ~8% slower decode) or the server maps them
    (--mmap-experts), "arena" for "expert arena: ..." or the server's "experts loaded: N GiB at R GiB/s" (said after an
    arena load only; its "loading the experts into RAM" line comes for both tiers).  Else "capped": capped.sh's
    "capped: tier=..." line (its guess before the start).  Else "guess": capped.sh's rule on keep and big (big_free_gib
    before the start).  Else ("unknown", "none")."""
    if any(s in text for s in ("experts via mmap", "not enough RAM for the expert arena", "mapping the experts from the model files")):
        return "mmap", "engine"
    if "expert arena:" in text or "experts loaded:" in text:
        return "arena", "engine"
    m = CAPPED_LINE.search(text)
    if m:
        return m[1], "capped"
    if keep is not None and big >= 0:
        return ("arena" if keep <= ARENA_KEEP_MAX and big >= ARENA_BIG_GIB else "mmap"), "guess"
    return "unknown", "none"


def keep_gib(text, env):
    """STRATA_RAM_KEEP_GIB of one engine start: the engine's own "--ram-keep-gib W:" line, else capped.sh's "keep_gib=N",
    else the config's env, else this process's env (capped.sh passes it on), else capped.sh's default 12."""
    m = KEEP_LINE.search(text) or CAPPED_LINE.search(text)
    if m:
        return float(m[1] if m.re is KEEP_LINE else m[2])
    return float(env.get("STRATA_RAM_KEEP_GIB") or os.environ.get("STRATA_RAM_KEEP_GIB") or 12)


def log_lines(log):
    return Path(log).read_text(errors="replace").splitlines() if log and Path(log).exists() else []


def save_engine_log(log, s0, path):
    """Copies this start's lines of the shared engine log (the preset's "log", appended to by every start: the lines after
    the first s0; all of them if the file got shorter) to path, the attempt's <arm>-r<rep>-a<attempt>.engine.log."""
    lines = log_lines(log)
    Path(path).write_text("".join(line + "\n" for line in lines[s0 if s0 <= len(lines) else 0:]))


def start_server(cfg_path, log_out, port, ready_wait=900):
    """Starts the engine server for a config and waits for its "ready: http" line.  Returns (process, None), or
    (None, "server died before ready" / "server not ready ...") - the caller stops the servers."""
    env = dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/core-10.0/lib", HIP_VISIBLE_DEVICES="0")
    with open(log_out, "w") as lf:
        srv = subprocess.Popen([str(ROOT / "tools/pancake/capped.sh"), PY, "serve/server.py", "--engine", "strata", "--config",
                                cfg_path, "--port", str(port)], cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env)
    OWN.append(srv)
    t0 = time.time()
    while time.time() - t0 < ready_wait:
        txt = Path(log_out).read_text(errors="replace")
        if "ready: http" in txt:
            return srv, None
        if srv.poll() is not None or "EngineDied" in txt:
            return None, "server died before ready"
        time.sleep(3)
    return srv, None   # as before: not ready after ready_wait, carry on (the first request waits for it)


def token_check(url, P, cases):
    """Prints the server's own token count (count_tokens: renders and tokenizes, the model does not run) of each long* case
    before it is sent.  Best effort: a server without the endpoint, or a failure, only prints a note."""
    for c in cases:
        if not is_long(c):
            continue
        try:
            body = {"model": "x", "messages": [{"role": "user", "content": P[c]}]}
            req = urllib.request.Request(url + "/v1/messages/count_tokens", json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
            n = json.load(urllib.request.urlopen(req, timeout=300))["input_tokens"]
            tgt = LONG_TARGET.get(c)
            off = f" (target {tgt}{'' if abs(n / tgt - 1) <= 0.08 else ', MORE THAN 8% OFF'})" if tgt else ""
            print(f"  {c}: {n} prompt tokens by count_tokens{off}", flush=True)
        except Exception as e:   # noqa: BLE001 - informational only
            print(f"  {c}: count_tokens unavailable ({type(e).__name__})", flush=True)


def n_requests(c, per_run):
    """Requests per server for case c: a prefill case is read once (minutes each), the others per_run times."""
    return 1 if is_long(c) else per_run


def run_arm(name, cfg_path, cfg, cases, P, log_out, port, per_run=1, passes=1, warmup=0):
    """One engine start: warmup throwaway passes, then `passes` measured passes (a pass = every case once, n_requests each).
    warmup 0: the single short warm-up request instead (as before)."""
    log = cfg.get("log")
    s0, big0 = len(log_lines(log)), big_free_gib()   # the engine log is appended to: this start's lines come after s0
    engine_log = Path(log_out).with_suffix(".engine.log")   # this start's engine lines (promote.py / switch_report.py read it)
    srv, err = start_server(cfg_path, log_out, port)
    if err:
        save_engine_log(log, s0, engine_log)
        return None, err
    url = f"http://127.0.0.1:{port}"
    if not warmup:
        request(url, P["short"])                          # warm-up, not counted
    token_check(url, P, cases)
    lines = log_lines(log)
    n0 = len(lines)
    start_text = "\n".join(lines[s0 if s0 <= n0 else 0:])
    plan = [(k, c) for k in range(warmup + passes) for c in cases for _ in range(n_requests(c, per_run))]
    nwarm = sum(1 for k, _ in plan if k < warmup)
    for k, c in plan[:nwarm]:
        request(url, case_text(P, c, k), MAXTOK.get(c, 512))
    mon = NoiseMonitor([srv.pid])
    mon.start()
    texts = {}
    for k, c in plan[nwarm:]:
        t = request(url, case_text(P, c, k), MAXTOK.get(c, 512))
        if not is_long(c):   # a long* prompt has a unique first line: its text is not comparable between arms
            texts[c] = hashlib.sha1(t.encode()).hexdigest()[:16]
    noise, top = mon.result()
    save_engine_log(log, s0, engine_log)
    rows = []
    for line in log_lines(log)[n0:]:
        m = LINE.search(line)
        # Flash-Next: "strata serve: prompt N tokens ..."; the Qwen3.6 (q36) engine: "strata serve: session S: prompt N tokens ..."
        if m and ("strata serve: prompt" in line or "strata serve: session" in line):
            acc = int(m[5]) / int(m[6]) if m[5] and int(m[6]) > 0 else None
            rows.append({"prompt": int(m[1]), "prompt_tps": float(m[2]), "gen": int(m[3]), "decode_tps": float(m[4]),
                         "accept": acc})
    stop_servers()
    want = len(plan)
    if len(rows) > want:   # an extra engine line (a retried / re-read request): the last ones are the cases'
        rows = rows[-want:]
    if len(rows) != want:
        return None, f"expected {want} engine lines, got {len(rows)}"
    for (k, c), r in zip(plan, rows):
        r["case"] = c
        if passes > 1 or warmup:   # block / warm-up runs only: the old single-pass rows stay as they were
            r["pass"], r["warm"] = k, k > 0
        if is_long(c) and LONG_TARGET.get(c) and abs(r["prompt"] / LONG_TARGET[c] - 1) > 0.08:
            print(f"  WARNING {c}: the engine read {r['prompt']} prompt tokens, target {LONG_TARGET[c]} (more than 8% off: adjust LONG_CPT)",
                  flush=True)
    res = {"rows": rows[nwarm:], "texts": texts, "noise": noise, "noise_top": top, "reclaim": round(mon.reclaim, 2),
           "gpu": mon.gpu_summary(), "big_free_gib": round(big_free_gib(), 1)}
    if warmup:
        res["warmup_rows"] = rows[:nwarm]
    start_text += "\n" + Path(log_out).read_text(errors="replace")
    keep = keep_gib(start_text, cfg.get("env") or {})
    tier, source = tier_of(start_text, keep, big0)
    res.update(tier=tier, tier_source=source, keep_gib=keep, big_free_gib_start=round(big0, 1))
    return res, None


def ci(xs):
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, T95.get(n - 1, 2.0) * sd / math.sqrt(n)


def paired(results, a, b, c):
    """[a - b] of case c's metric, paired by repetition: the same rep (ABBA round), pass and request slot (with every slot
    present: the same pairs as in list order; a rep or pass that one arm lacks is left out instead of shifting the rest).
    An attempt without "rep" counts by its place in the list."""
    def keyed(arm):
        return {(res.get("rep", j), row.get("pass", 0), i): row[metric(c)]
                for j, res in enumerate(results.get(arm, [])) for row, i in request_slots(res) if row["case"] == c}
    ka, kb = keyed(a), keyed(b)
    return [ka[k] - kb[k] for k in sorted(ka.keys() & kb.keys())]


def decided(results, arms, cases, floor=1.5, least=2, deff=1.0):
    """True when every arm/case either has an interval that excludes 0 or one narrower than floor % of base (and at least
    `least` values: the robust outlier rule judges only from ROBUST_MIN on; deff: as in summary)."""
    base = arms[0]
    for c in cases:
        key = metric(c)
        vals = {a: [r[key] for res in results.get(a, []) for r in res["rows"] if r["case"] == c] for a in arms}
        if len(vals[base]) < least:
            return False
        bm = sum(vals[base]) / len(vals[base])
        for a in arms[1:]:
            d = paired(results, a, base, c)
            if len(d) < least:
                return False
            dm, dh = ci(d)
            dh *= math.sqrt(deff)
            if not (abs(dm) > dh or dh / bm * 100 < floor):
                return False
    return True


def schedule(arms, reps, block=0):
    """([(round, arm)], passes per start): engine starts in A B B A order (every arm once per round, odd rounds reversed, so
    each arm sees early and late slots equally).  block 0: one start per rep.  block N: N passes per start and
    max(2, ceil(reps / N)) rounds: at least 2 starts per arm, so start-to-start variance stays in the interval."""
    rounds, passes = (reps, 1) if block <= 0 else (max(2, -(-reps // block)), block)
    return [(r, x) for r in range(rounds) for x in (arms if r % 2 == 0 else arms[::-1])], passes


ROBUST_Z = 3.5        # modified z-score cut (Iglewicz & Hoaglin 1993: |0.6745 (x - median) / MAD| > 3.5)
ROBUST_FLOOR = 0.025  # the spread estimate never below 2.5% of the median: the MAD of 4-6 values often sits in one tight cluster
                      # (base edit 143.3-143.9, MAD 0.3%) while quiet, kept attempts spread more (25-kern-hip-confirm: base edit
                      # 133.4-143.9, SD ~2.6%); with the 3.5 cut a value must be ~9% off its median, and the slow engine states
                      # in the 2026-11-11 data are 10-35% off (short 93-111 and edit 93-111 against 124 / 143)
ROBUST_MIN = 3        # fewer values in a group: nothing is judged (a median of two is no reference)
SLOW_START = ROBUST_Z * ROBUST_FLOOR   # 8.75%: a start this far below its arm's reference start on some case is a slow engine
                      # state (whole start slow, 10-35% in the 2026-11-11 data: 7 of 60 starts); normal starts of an arm differ by
                      # an SD of 0.5% (short) to 1.9% (edit)
BLOCK_ICC = 0.45      # share of the run-to-run variance that the passes of one engine start have in common, slow starts left
                      # out (2026-11-11 screens, two requests per start, 22 starts: short 0.15, edit 0.45; abtest_replay.py
                      # --components): with --block the summary widens its intervals by sqrt(1 + (passes - 1) x BLOCK_ICC), the
                      # design effect of passes clustered in a start.  Revisit it from the first real --block run (same tool)


def design_effect(passes):
    """Variance factor of a mean over `passes` passes per start (1 without --block)."""
    return 1 + (passes - 1) * BLOCK_ICC


def robust_bounds(xs, z=ROBUST_Z, floor=ROBUST_FLOOR, least=ROBUST_MIN):
    """(lo, hi): values outside are robust outliers of xs; None when xs has fewer than `least` values."""
    if len(xs) < least:
        return None
    med = statistics.median(xs)
    s = max(1.4826 * statistics.median(abs(x - med) for x in xs), floor * abs(med))
    return med - z * s, med + z * s


def request_slots(rec):
    """(row, slot) of one attempt's measured rows: slot = the row's index among the rows of its case in its pass.  The
    screen's second request of a case is its own group (same prompt again in the server: a different regime)."""
    seen = {}
    for row in rec["rows"]:
        k = (row.get("pass", 0), row["case"])
        seen[k] = seen.get(k, -1) + 1
        yield row, seen[k]


def robust_keep(used, z=ROBUST_Z, floor=ROBUST_FLOOR, least=ROBUST_MIN):
    """{arm: [attempt]} -> (the same without the outlier passes, [dropped pass]).  A row is judged against the rows of the
    same arm, case and request slot (every start and pass pooled), on its case's metric.  A pass with an outlier row is
    dropped whole (without --block: the attempt): a slow engine state slows every case of a start (25-kern-hip-confirm,
    all3 rep 0: short -5% and edit -29% in the same start)."""
    groups = {}
    for arm, recs in used.items():
        for rec in recs:
            for row, i in request_slots(rec):
                groups.setdefault((arm, row["case"], i), []).append(row[metric(row["case"])])
    bounds = {g: robust_bounds(xs, z, floor, least) for g, xs in groups.items()}
    out, dropped = {}, []
    for arm, recs in used.items():
        out[arm] = []
        for rec in recs:
            marks = []   # (row, its bounds when it is an outlier, else None)
            for row, i in request_slots(rec):
                b = bounds[(arm, row["case"], i)]
                marks.append((row, b if b and not b[0] <= row[metric(row["case"])] <= b[1] else None))
            bad = {row.get("pass", 0) for row, b in marks if b}
            for p in sorted(bad):
                dropped.append({"arm": arm, "rep": rec.get("rep"), "attempt": rec.get("attempt"), "pass": p,
                                "rows": [(row["case"], row[metric(row["case"])], b) for row, b in marks if row.get("pass", 0) == p]})
            out[arm].append({**rec, "rows": [row for row, _ in marks if row.get("pass", 0) not in bad]})
    return out, dropped


def start_means(rec):
    """{case: mean of the attempt's measured rows on the case's metric}."""
    by = {}
    for row in rec["rows"]:
        by.setdefault(row["case"], []).append(row[metric(row["case"])])
    return {c: sum(v) / len(v) for c, v in by.items()}


def slow_starts(starts, cut=SLOW_START):
    """Indexes of the slow starts among one arm's attempts: more than `cut` below the arm's reference on some case - the
    median of the starts from 3 on, else the faster of the two (two starts cannot outvote each other; the slow engine
    states in the data were only ever slow)."""
    means = [start_means(r) for r in starts]
    slow = set()
    for c in {c for m in means for c in m}:
        vals = [m[c] for m in means if c in m]
        if len(vals) >= 2:
            ref = statistics.median(vals) if len(vals) >= 3 else max(vals)
            slow |= {i for i, m in enumerate(means) if c in m and m[c] < (1 - cut) * ref}
    return slow


def slots_of(records):
    """{(arm, rep): [attempts in file order]}."""
    out = {}
    for rec in records:
        out.setdefault((rec["arm"], rec["rep"]), []).append(rec)
    return out


def robust_picks(records):
    """{(arm, rep): the attempt a robust run uses, or None}: per slot the first attempt that is not a slow start against the
    arm's other picks (judged again after every replacement; None: every attempt of the slot so far was slow)."""
    slots = slots_of(records)
    pick = dict.fromkeys(slots, 0)
    changed = True
    while changed:
        changed = False
        for arm in dict.fromkeys(k[0] for k in slots):
            keys = [k for k in slots if k[0] == arm and pick[k] is not None]
            for j in slow_starts([slots[k][pick[k]] for k in keys]):
                k = keys[j]
                pick[k] = pick[k] + 1 if pick[k] + 1 < len(slots[k]) else None
                changed = True
    return {k: None if p is None else slots[k][p] for k, p in pick.items()}


def select(records, outlier=None, foreign_cpu=None, reclaim_max=None):
    """The attempts a summary uses, from attempt records (abtest jsonl rows in file order): ({arm: [record]} in rep order,
    [dropped]).  outlier "gate": per arm and rep the first clean attempt, else the quietest (the old rule).  "robust": per
    arm and rep the first attempt that is not a slow start (robust_picks: a robust run repeats only those), then
    robust_keep; dropped lists the slow starts left out ("pass" None) and the outlier passes.  None: the records' own
    opts.outlier (records older than it: gate).  foreign_cpu and reclaim_max (both): judge "clean" again with these limits
    (a replay of the gate with other thresholds; a stricter limit than the run's can only choose among recorded attempts)."""
    mode = outlier or next((r["opts"]["outlier"] for r in records if "opts" in r), "gate")
    slots = slots_of(records)
    clean = (lambda r: r["clean"]) if foreign_cpu is None else (lambda r: r["noise"] <= foreign_cpu and r.get("reclaim", 0.0) <= reclaim_max)
    picks = robust_picks(records) if mode == "robust" else {
        k: next((r for r in recs if clean(r)), None) or min(recs, key=lambda r: r["noise"]) for k, recs in slots.items()}
    used, slow = {}, []
    for k in sorted(slots, key=lambda k: k[1]):
        if picks[k] is not None:
            used.setdefault(k[0], []).append(picks[k])
        if mode == "robust":   # the attempts before the pick were slow starts
            n = len(slots[k]) if picks[k] is None else next(i for i, r in enumerate(slots[k]) if r is picks[k])
            slow += [{"arm": k[0], "rep": k[1], "attempt": r.get("attempt"), "pass": None,
                      "rows": [(c, v, None) for c, v in start_means(r).items()]} for r in slots[k][:n]]
    if mode != "robust":
        return used, []
    used, dropped = robust_keep(used)
    return used, slow + dropped


def print_selection(used, dropped, robust, passes=1):
    """What the selection did, printed before the summary (whose format promote.py and queue_run.sh rely on): the robust
    rule's drops, the block design effect, and a warning when the attempts used mix memory tiers (arena vs mmap: ~8%)."""
    if robust:
        starts = [d for d in dropped if d["pass"] is None]
        print(f"\noutlier rule robust: {len(starts)} slow starts left out (more than {SLOW_START * 100:g}% below the arm's "
              f"reference start on a case; a robust run repeats them), {len(dropped) - len(starts)} passes dropped (|x - median| > "
              f"{ROBUST_Z} x max(1.4826 MAD, {ROBUST_FLOOR * 100:g}% of the median) within arm, case and request slot, "
              f">= {ROBUST_MIN} values; the pass goes whole)")
        for d in dropped:
            print(f"  {d['arm']:20s} rep {d['rep']} attempt {d['attempt']} "
                  + ("slow start (means)" if d["pass"] is None else f"pass {d['pass']}") + ": " + ", ".join(
                      f"{c} {v:.1f}" + (f" outside [{b[0]:.1f}, {b[1]:.1f}]" if b else "") for c, v, b in d["rows"]))
    if passes > 1:
        print(f"\nblock of {passes} passes per start: intervals widened by sqrt(design effect {design_effect(passes):.2f}) "
              f"(1 + (passes - 1) x {BLOCK_ICC}, the start-shared share of the noise)")
    tiers = {}
    for arm, recs in used.items():
        for rec in recs:
            tiers.setdefault(rec.get("tier", "unknown"), []).append(arm)
    if len(set(tiers) - {"unknown"}) > 1:
        print("\nWARNING: the attempts used mix memory tiers (" + "; ".join(f"{t}: {len(v)} attempts ({', '.join(sorted(set(v)))})"
              for t, v in sorted(tiers.items())) + "): compare same-tier runs only (promote.py rejects mixed-tier pairs)")


def screen_verdict(results, arms, cases):
    base = arms[0]
    print("\nscreen verdict (promote: >= +2% on a case and no case below -3%; else drop)")
    for a in arms[1:]:
        deltas = {}
        for c in cases:
            key = metric(c)
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
    print("\ngenerated text vs base (identical = exact: no quality gate needed; long* cases are not compared:"
          " unique prompt prefix per request)")
    for a in arms[1:]:
        at = [res.get("texts", {}) for res in results.get(a, [])]
        if not at:
            continue
        same = all(t.get(c) == bt[0].get(c) for t in at for c in cases if c in bt[0])
        print(f"  {a:20s} {'identical on every case -> exact' if same else 'DIFFERENT text -> lossy or placement-dependent: quality gate'}")


def summary(results, arms, cases, deff=1.0):
    """deff: design effect of --block (design_effect(passes)): the intervals are widened by its square root."""
    base = arms[0]
    print("\nsummary (decode tok/s; prompt tok/s for long* but longdec*; mean +- 95% interval over clean runs, or the quietest attempt)")
    for c in cases:
        key = metric(c)
        vals = {a: [r[key] for res in results.get(a, []) for r in res["rows"] if r["case"] == c] for a in arms}
        bm, _ = ci(vals[base]) if vals[base] else (float("nan"), 0)
        for a in arms:
            if not vals[a]:
                print(f"  {c:6s} {a:20s} no clean runs")
                continue
            m, h = ci(vals[a])
            h *= math.sqrt(deff)
            line = f"  {c:6s} {a:20s} {m:8.1f} +- {h:5.1f}  (n={len(vals[a])})"
            d = paired(results, a, base, c) if a != base else []   # same ABBA round (with --block, pass k of the round)
            if d:
                dm, dh = ci(d)
                dh *= math.sqrt(deff)
                verdict = "better" if dm - dh > 0 else "worse" if dm + dh < 0 else "no difference"
                line += f"   vs {base}: {dm / bm * 100:+5.1f}% +- {dh / bm * 100:4.1f}%  -> {verdict}"
            print(line)
            if not is_long(c):
                accs = [r["accept"] for res in results.get(a, []) for r in res["rows"] if r["case"] == c and r["accept"]]
                if accs:
                    print(f"  {'':6s} {'':20s} drafts accepted {sum(accs) / len(accs):.2f}")


def main():
    global RECLAIM_MAX
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
    ap.add_argument("--reclaim-max", type=float, default=RECLAIM_MAX,
                    help="%% of one core of kswapd/kcompactd above which a run is noisy and repeated (default 1.0; "
                         "2026-11-11 data, 62 attempts: up to 3 decoded as fast as below 1, above 3 edits were ~8%% slower, "
                         "while 1.0 discarded 48%% of all attempts)")
    ap.add_argument("--block", type=int, default=0,
                    help="N passes per engine start, A B B A blocks, max(2, ceil(reps / N)) starts per arm (0 = one start per rep)")
    ap.add_argument("--warmup", type=int, default=0,
                    help="throwaway passes per start before the measured ones (0 = the single short warm-up request)")
    ap.add_argument("--outlier", choices=("gate", "robust"), default="gate",
                    help="gate: repeat an attempt the noise proxies call noisy (default); robust: keep every attempt, drop only "
                         "a pass in which a request's tok/s is a robust outlier within its arm (median / MAD, see the module doc)")
    ap.add_argument("--force-busy", action="store_true",
                    help="start although another abtest / restore_check / engine runs (only for the fake-engine tests: it stops "
                         "its own servers only, but a real run beside it is disturbed all the same)")
    a = ap.parse_args()
    if a.tier == "screen" and (a.block or a.outlier == "robust"):
        ap.error("--block and --outlier robust are for --tier confirm (a screen has one start and 1-2 values per arm)")
    busy = busy_processes()
    if busy and not a.force_busy:
        print("abtest: another test or engine is already running on this machine: not starting (a second run would disturb it, and "
              "its numbers would be wrong).  Running:\n" + "\n".join(f"  pid {pid}: {cmd[:200]}" for pid, cmd in busy)
              + "\nWait for it, or stop it yourself if it is a leftover of a crashed run (abtest never stops what it did not start).  "
              "--force-busy starts anyway (fake-engine tests only).", file=sys.stderr, flush=True)
        return 3
    if busy:
        print(f"abtest: --force-busy: running beside {len(busy)} other test / engine process(es)", flush=True)
    RECLAIM_MAX = a.reclaim_max
    spec = json.load(open(a.spec))
    arms = list(spec["arms"])
    cases = spec.get("cases", ["short", "edit", "long"])
    reps = a.reps or spec.get("reps", 3)
    per_run = 1
    if a.tier == "screen":
        if not spec.get("prefill"):   # a prefill spec keeps its long* cases (one request each, see n_requests)
            cases = [c for c in cases if not is_long(c) or c in LONG_DEC] or ["short"]
        reps, per_run = 1, 2
    base = str(ROOT / spec["base"]) if not os.path.isabs(spec["base"]) else spec["base"]
    for k in ("temperature", "top_p", "seed"):   # optional sampling, for arms that only matter at T > 0
        if spec.get(k) is not None:
            SAMPLING[k] = spec[k]
    if SAMPLING["temperature"] > 0:
        print(f"sampling: {SAMPLING} (sampled text is not reproducible: the text-hash check below is not meaningful)",
              flush=True)
    P = prompts(cases)
    for c in cases:
        if c not in P:
            raise SystemExit(f"abtest: unknown case {c!r} (short, edit, long, {', '.join(LONG_TARGET)})")
        if is_long(c):
            print(f"case {c}: {len(P[c])} chars, sha1 {hashlib.sha1(P[c].encode()).hexdigest()[:12]}"
                  + (f", target {LONG_TARGET[c]} tokens" if c in LONG_TARGET else "") + " (a unique run line is prepended to every request)",
                  flush=True)
    work =Path(a.out).with_suffix("")
    work.mkdir(parents=True, exist_ok=True)
    order, passes = schedule(arms, reps, a.block)   # A B B A, then B A A B, ... : each arm sees early and late slots equally
    opts = {"test_tier": a.tier, "reps": reps, "block": a.block, "passes": passes, "warmup": a.warmup, "outlier": a.outlier,
            "foreign_cpu": a.foreign_cpu, "reclaim_max": RECLAIM_MAX, "retries": a.retries, "early_stop": not a.no_early_stop,
            "per_run": per_run}
    robust, deff = a.outlier == "robust", design_effect(passes)
    records = []   # every attempt of this batch (the jsonl rows); select() picks what the summary uses
    out = open(a.out, "a")
    deaths = 0

    def next_attempt(rep, arm):   # (a death is not recorded but used its number: the files are named by it)
        return max((x["attempt"] for x in records if x["arm"] == arm and x["rep"] == rep), default=-1) + 1

    begun, i = set(), 0   # rounds begun: early stop is judged when a new even round begins
    while i < len(order):   # (robust: a slow start's slot is put back into the order, see below)
        r, arm = order[i]
        i += 1
        if (not a.no_early_stop and a.tier == "confirm" and r >= 2 and r % 2 == 0 and r not in begun
                and decided(select(records)[0], arms, cases, least=ROBUST_MIN if robust else 2, deff=deff)):
            print(f"early stop after {r} {'rounds of starts' if a.block else 'reps'}: every arm/case decided or below 1.5% of base",
                  flush=True)
            break
        begun.add(r)
        quietest = None   # least noisy attempt: kept when every attempt was noisy, so the base never ends with no runs
        for attempt in range(next_attempt(r, arm), a.retries + 1):
            quiet_gate(a.max_load, a.foreign_cpu, [], a.max_wait)
            cfg_path = str(work / f"{arm}.json")
            cfg = make_config(base, spec["arms"][arm], cfg_path)
            res, err = run_arm(arm, cfg_path, cfg, cases, P, str(work / f"{arm}-r{r}-a{attempt}.out"), a.port, per_run,
                               passes, a.warmup)
            if err:
                print(f"rep {r} {arm}: FAILED ({err})", flush=True)
                if "died" in err:
                    deaths += 1
                    if deaths >= 2:
                        print(f"ABORT: the server died twice before ready (free RAM in 2 MiB blocks: {big_free_gib():.0f} GiB;"
                              f" fix the machine first: 'sudo tools/pancake/os_tune.sh compact'). Partial results below.",
                              flush=True)
                        summary(select(records)[0], arms, cases, deff)
                        return 2
                    continue   # one retry
                break
            clean = res["noise"] <= a.foreign_cpu and res["reclaim"] <= RECLAIM_MAX
            rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "rep": r, "arm": arm, "attempt": attempt, "clean": clean, **res,
                   "opts": opts}
            records.append(rec)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            desc = " ".join(f"{x['case']} {x['decode_tps']:.1f}" + (f"/{x['prompt_tps']:.0f}p {x['prompt']}t" if is_long(x["case"]) else "")
                            for x in res["rows"])
            g = res.get("gpu") or {}
            gd = (f"  gpu {g['sclk']}/{g['mclk']} MHz {g['t_junc_max']}C {g['watt']}W gtt {g.get('gtt_mib_max', 0)}M"
                  f" cpu {g.get('cpu_mhz', 0)}MHz {g.get('tctl_max', 0)}C igpu {g.get('igpu_busy', 0)}%/vcn {g.get('igpu_vcn', 0)}%"
                  f" fclk {g.get('fclk_min', 0)}") if g else ""
            print(f"rep {r} {arm:20s} {desc}  foreign CPU {res['noise']:.0f}% {res['noise_top']}{gd}"
                  f"{'' if clean else '  noisy by the proxies: kept (outlier rule)' if robust else '  NOISY -> repeat'}", flush=True)
            if clean or robust:
                break
            if quietest is None or res["noise"] < quietest["noise"]:
                quietest = res
        else:
            if quietest is not None:   # 2026-11-10 screen 05: caveman-proxy + a foreign python3 left the base with 0 runs -> all arms dropped
                print(f"rep {r} {arm}: every attempt noisy, keeping the quietest ({quietest['noise']:.0f}% foreign CPU)", flush=True)
        if robust:   # a slow start of this arm (this one, or an earlier one this start shows up): its slot runs again, same round
            again = [(rep, x) for (x, rep), p in robust_picks(records).items()
                     if x == arm and p is None and next_attempt(rep, x) <= a.retries and (rep, x) not in order[i:]]
            for rep, _ in again:
                print(f"rep {rep} {arm}: slow start (more than {SLOW_START * 100:g}% below the arm's other starts): run again",
                      flush=True)
            order[i:i] = again
    results, dropped = select(records)   # gate: per arm and rep the first clean attempt, else the quietest (as above)
    print_selection(results, dropped, robust, passes)
    summary(results, arms, cases, deff)
    text_check(results, arms, cases)
    if a.tier == "screen":
        screen_verdict(results, arms, cases)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))   # `kill` ends the servers too (a leftover would stop the next start)
    try:
        sys.exit(main())
    finally:
        stop_servers()   # Ctrl-C, a crash, an abort: no server of this abtest is left behind
