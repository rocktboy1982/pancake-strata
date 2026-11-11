#!/usr/bin/env python3
"""night_queue.py - run a list of benchmark jobs one after another on a quiet machine, unattended (MOD_PLAN Phase 17, night queue).

Standard library only.  One JSON plan (tools/pancake/night_queue_plan.json) lists the jobs.  A job = one engine start (a preset
config with env / args deltas, or an external command such as Avalon's llama-server) + one or more client passes against it
+ optional post-processing commands.  For every job the runner does the same things, in this order:

  1. QUIET-MACHINE GUARD: no other strata / llama-server / build / benchmark process, 1-minute load below --load-max (1.0),
     MemAvailable at least --mem-gib (48), the job's port free.  It waits up to --guard-wait-min minutes; if the machine is
     still not quiet the job is SKIPPED (status "skipped", the reasons in the log and in RESULTS.md), never run noisy.
  2. renders the config (placeholders <REPO> <STRATA_DATA> <MODELS>, build-hip, the job's env / args deltas, per-job log and
     turn-log paths) and starts the engine in its own session (setsid), remembering its PID.
  3. waits for GET /health = 200 (--start-timeout-min), then runs the client pass(es), each in its own session with a timeout.
  4. stops the engine: SIGTERM to the leader PID, then to the remaining members of its session BY PID, SIGKILL only as the last
     resort, waits until the session is gone, the port is free and MemAvailable is back (--mem-gib).  Nothing here uses
     `pkill -f` or kills by name.
  5. runs the post commands (for example tools/pancake/ttft_audit.py on the job's engine log) and writes state.json.

A per-job timeout covers 2-4.  state.json lets `--resume` skip finished jobs.  RESULTS.md is rewritten after every job: one
row per job, every pass with the first pass flagged as warm-up, and a per-arm table for jobs that share a compare group (the
A B B A order of item 1).  No result is interpreted: the tables only carry the numbers.

    python3 tools/pancake/night_queue.py --dry-run                  print every job, command and the order, run nothing
    python3 tools/pancake/night_queue.py                            the real run (needs the whole machine; hours)
    python3 tools/pancake/night_queue.py --resume                   after a stop or a crash: skip the finished jobs
    python3 tools/pancake/night_queue.py --only defer-iq2xs-01A     one job (a name, a glob, comma list; repeatable)
    python3 tools/pancake/night_queue.py --group 1-defer            one group
    python3 tools/pancake/night_queue.py --fake                     the whole loop on a stdlib fake server (seconds, no GPU)
    python3 tools/pancake/night_queue.py --results-only             rewrite RESULTS.md from state.json

Exit codes: 0 every selected job finished, 1 some job failed / timed out / was skipped, 2 plan or usage error, 130 interrupted.
Tests: nice -n 19 python3 -I -m pytest -q tools/pancake/test_night_queue.py   (fake server only)
"""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import http.client
import http.server
import json
import math
import os
import re
import signal
import socket
import socketserver
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_PLAN = HERE / "night_queue_plan.json"
SCHEMA = 1

# processes that make a start unsafe: engines and builds (matched by exact process name, never by a substring of the command line) ...
BUSY_COMM = {"strata", "llama-server", "cc1plus", "cc1", "ninja", "cmake", "hipcc", "ld.lld"}
# ... and our own benchmark / server scripts (matched on the basename of an argument of a python interpreter line)
BUSY_SCRIPTS = {"abtest.py", "ablate_greedy.py", "restore_check.py", "ttft_bench.py", "prose_gap.py", "nq_clients.py", "server.py",
                "queue_run.py", "gpu_profile.py", "task_gate.py", "quality_gate.py"}


class PlanError(Exception):
    pass


# ----------------------------------------------------------------------------------------------------------------- machine probe
class Probe:
    """Everything the guard and the stop logic read from the machine.  Tests replace it."""

    def load1(self) -> float:
        return os.getloadavg()[0]

    def mem_gib(self) -> float:
        try:
            for line in open("/proc/meminfo"):
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576.0
        except OSError:
            pass
        return 0.0

    def processes(self) -> list[dict]:
        """[{pid, ppid, sid, comm, argv}] from /proc."""
        out = []
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                stat = open(f"/proc/{d}/stat").read()
                rest = stat[stat.rindex(")") + 2:].split()
                comm = stat[stat.index("(") + 1:stat.rindex(")")]
                argv = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
                out.append({"pid": int(d), "ppid": int(rest[1]), "sid": int(rest[3]), "comm": comm,
                            "argv": [x.decode("utf-8", "replace") for x in argv if x]})
            except (OSError, ValueError, IndexError):
                continue
        return out

    def port_busy(self, port: int) -> bool:
        s = socket.socket()
        s.settimeout(1.0)
        try:
            return s.connect_ex(("127.0.0.1", port)) == 0
        finally:
            s.close()

    def sleep(self, s: float) -> None:
        time.sleep(s)

    def now(self) -> float:
        return time.monotonic()


def ancestors(procs: list[dict], pid: int) -> set[int]:
    by = {p["pid"]: p for p in procs}
    out, cur = set(), pid
    while cur in by and cur not in out:
        out.add(cur)
        cur = by[cur]["ppid"]
    return out | {pid}


def busy_processes(procs: list[dict], me: int) -> list[str]:
    """The processes that make a start unsafe, as 'pid name' strings.  The queue itself and its ancestors are skipped."""
    skip = ancestors(procs, me)
    found = []
    for p in procs:
        if p["pid"] in skip:
            continue
        argv = p["argv"]
        hit = p["comm"] in BUSY_COMM
        if not hit and argv and os.path.basename(argv[0]).startswith("python"):
            hit = any(os.path.basename(x) in BUSY_SCRIPTS and (os.path.basename(x) != "server.py" or "--engine" in argv)
                      for x in argv[1:4])
        if hit:
            found.append(f"{p['pid']} {p['comm']}")
    return found


class Guard:
    def __init__(self, probe: Probe, load_max: float = 1.0, mem_gib: float = 48.0, enabled: bool = True):
        self.probe, self.load_max, self.mem_gib, self.enabled = probe, load_max, mem_gib, enabled

    def reasons(self, port: int) -> list[str]:
        if not self.enabled:
            return []
        out = []
        busy = busy_processes(self.probe.processes(), os.getpid())
        if busy:
            out.append("other engine / build / benchmark processes: " + ", ".join(busy[:6]) + (" ..." if len(busy) > 6 else ""))
        load = self.probe.load1()
        if load >= self.load_max:
            out.append(f"1-minute load {load:.2f} >= {self.load_max:g}")
        mem = self.probe.mem_gib()
        if mem < self.mem_gib:
            out.append(f"MemAvailable {mem:.1f} GiB < {self.mem_gib:g} GiB")
        if self.probe.port_busy(port):
            out.append(f"port {port} is in use")
        return out

    def wait(self, port: int, wait_s: float, poll_s: float, log=lambda m: None) -> tuple[bool, list[str]]:
        """(quiet, last reasons).  Polls every poll_s for up to wait_s seconds."""
        t0 = self.probe.now()
        last, shown = [], None
        while True:
            last = self.reasons(port)
            if not last:
                return True, []
            if tuple(last) != shown:
                log("guard: not quiet: " + "; ".join(last))
                shown = tuple(last)
            if self.probe.now() - t0 >= wait_s:
                return False, last
            self.probe.sleep(poll_s)


# ----------------------------------------------------------------------------------------------------------------- plan
def subst(obj, mapping: dict):
    """Replace {key} in every string of a JSON value, for the keys in mapping only (other braces are left alone)."""
    if isinstance(obj, str):
        for k, v in mapping.items():
            obj = obj.replace("{" + k + "}", str(v))
        return obj
    if isinstance(obj, list):
        return [subst(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {k: subst(v, mapping) for k, v in obj.items()}
    return obj


def expand_jobs(plan: dict) -> list[dict]:
    """The flat job list.  An entry with "arms" and "sequence" becomes one job per letter of the sequence, in that order:
    name-01A, name-02B, name-03B, name-04A ... (the letters are keys of "arms"; an arm may carry env, args, vars, label)."""
    jobs, seen = [], set()
    for raw in plan.get("jobs", []):
        entries = []
        if "sequence" in raw:
            arms = raw.get("arms") or {}
            for i, ch in enumerate(raw["sequence"], 1):
                if ch not in arms:
                    raise PlanError(f"job {raw.get('name')!r}: sequence letter {ch!r} is not an arm ({sorted(arms)})")
                j = {k: v for k, v in raw.items() if k not in ("arms", "sequence")}
                arm = arms[ch]
                j["name"] = f"{raw['name']}-{i:02d}{ch}"
                j["compare_group"] = raw["name"]
                j["arm"] = ch
                j["arm_label"] = arm.get("label", ch)
                j["env"] = {**(raw.get("env") or {}), **(arm.get("env") or {})}
                j["args"] = {**(raw.get("args") or {}), **(arm.get("args") or {})}
                j["vars"] = {**(raw.get("vars") or {}), **(arm.get("vars") or {})}
                if arm.get("engine"):
                    j["engine"] = arm["engine"]
                entries.append(j)
        else:
            entries.append(dict(raw))
        for j in entries:
            n = j.get("name", "")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", n):
                raise PlanError(f"job name {n!r} must be letters, digits, _ . -")
            if n in seen:
                raise PlanError(f"job {n!r} appears twice")
            seen.add(n)
            if j.get("engine") not in plan.get("engines", {}):
                raise PlanError(f"job {n!r}: engine {j.get('engine')!r} is not in the plan's engines")
            if not (j.get("client") or {}).get("cmd"):
                raise PlanError(f"job {n!r}: client.cmd is missing")
            j.setdefault("env", {})
            j.setdefault("args", {})
            j.setdefault("vars", {})
            j.setdefault("group", "")
            j.setdefault("post", [])
            jobs.append(j)
    return jobs


def client_passes(job: dict) -> list[dict]:
    """[{n, label, extra, est_min}] - client.passes is an int (identical passes) or a list of {label, extra, est_min}."""
    c = job["client"]
    p = c.get("passes", 1)
    est = float(c.get("est_min", 1))
    if isinstance(p, int):
        return [{"n": i, "label": f"pass {i}", "extra": [], "est_min": est} for i in range(1, p + 1)]
    return [{"n": i, "label": e.get("label", f"pass {i}"), "extra": list(e.get("extra", [])), "est_min": float(e.get("est_min", est))}
            for i, e in enumerate(p, 1)]


def job_est_min(job: dict, plan: dict) -> float:
    eng = plan["engines"][job["engine"]]
    return float(job.get("est_min") or (float(eng.get("start_est_min", 4)) + sum(p["est_min"] for p in client_passes(job))
                                       + float(eng.get("stop_est_min", 1))))


def load_plan(path: Path) -> dict:
    try:
        plan = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise PlanError(f"cannot read plan {path}: {e}")
    plan["_jobs"] = expand_jobs(plan)
    return plan


def select_jobs(jobs: list[dict], only: list[str], groups: list[str], skip_groups: list[str]) -> list[dict]:
    pats = [x for o in only for x in o.split(",") if x]
    out = []
    for j in jobs:
        if pats and not any(fnmatch.fnmatchcase(j["name"], p) for p in pats):
            continue
        if groups and j.get("group") not in groups:
            continue
        if j.get("group") in skip_groups:
            continue
        out.append(j)
    return out


# ----------------------------------------------------------------------------------------------------------------- rendering
def apply_delta(cfg: dict, env: dict | None, args: dict | None) -> None:
    """The abtest arm semantics: env value null removes the key; args "--flag": "value" sets, "" is a bare switch, null removes."""
    cfg.setdefault("env", {})
    for k, v in (env or {}).items():
        if v is None:
            cfg["env"].pop(k, None)
        else:
            cfg["env"][k] = str(v)
    a = cfg.setdefault("args", [])
    for k, v in (args or {}).items():
        if k in a:
            i = a.index(k)
            n = 2 if i + 1 < len(a) and not str(a[i + 1]).startswith("--") else 1
            del a[i:i + n]
        if v is not None:
            a += [k] + ([str(v)] if v != "" else [])


def render_config(eng: dict, job: dict, repo: Path, paths: dict) -> dict:
    src = eng.get("template") or eng.get("config")
    p = Path(src) if os.path.isabs(src) else repo / src
    text = p.read_text(encoding="utf-8")
    text = (text.replace("<REPO>", str(repo)).replace("<STRATA_DATA>", paths["data"]).replace("<MODELS>", paths["models"])
            .replace("/build-cuda/", "/build-hip/"))
    for a, b in eng.get("replace", []):
        text = text.replace(a, b)
    cfg = json.loads(text)
    apply_delta(cfg, job.get("env"), job.get("args"))
    return cfg


def process_env(eng: dict, job: dict, mapping: dict) -> dict:
    env = dict(os.environ)
    for k, v in (eng.get("env") or {}).items():
        v = subst(v, mapping)
        if k == "LD_LIBRARY_PATH" and os.environ.get(k):
            v = v + ":" + os.environ[k]
        env[k] = v
    for k, v in (job.get("env") or {}).items():   # the server process too (park_divergence reads its own environment)
        if v is None:
            env.pop(k, None)
        else:
            env[k] = str(v)
    return env


def default_paths(plan: dict, repo: Path) -> dict:
    p = {"data": "/home/dan/clone-of-d/Strata-data", "models": "/home/dan/clone-of-d/models",
         "avalon": "/home/dan/clone-of-d/Avalon", "avalon_models": "/home/dan/clone-of-d/Avalon-models",
         "rdna3": "/home/dan/clone-of-d/src/Strata-rdna3"}
    p.update(plan.get("paths") or {})
    return p


class Rendered:
    """What one job will run: engine argv / cwd / env, the config (not yet written), the client commands, the post commands."""

    def __init__(self):
        self.argv, self.cwd, self.env, self.cfg, self.cfg_path = [], "", {}, None, None
        self.clients, self.post, self.port, self.files = [], [], 0, {}


def render_job(job: dict, plan: dict, q: "Queue") -> Rendered:
    eng = plan["engines"][job["engine"]]
    run_dir, raw = q.run_dir, q.run_dir / "raw"
    port = int(eng.get("port", 8080))
    m = {"repo": str(q.repo), "run_dir": str(run_dir), "py": sys.executable, "venv_py": q.venv_py, "port": port, "job": job["name"],
         **q.paths, **{k: str(v) for k, v in (job.get("vars") or {}).items()}}
    r = Rendered()
    r.port = port
    r.files = {"server_log": str(raw / f"srv-{job['name']}.log"), "engine_log": str(raw / f"engine-{job['name']}.log"),
               "turn_log": str(raw / f"turns-{job['name']}.jsonl"), "config": str(run_dir / "configs" / f"{job['name']}.json")}
    m.update(r.files)
    if eng.get("template") or eng.get("config"):
        r.cfg = render_config(eng, job, q.repo, q.paths)
        for key in ("log", "turn_log"):   # one engine log and one turn log per job (the daily files stay untouched)
            if key in r.cfg:
                r.cfg[key] = r.files["engine_log" if key == "log" else "turn_log"]
        r.cfg_path = r.files["config"]
    r.cwd = subst(eng.get("cwd", "{repo}"), m)
    r.env = process_env(eng, job, m)
    argv = subst(eng["argv"], m)
    if r.cfg_path is None:
        # an external command (Avalon's llama-server): the job's args deltas append to the command line
        for k, v in (job.get("args") or {}).items():
            if k in argv:
                i = argv.index(k)
                del argv[i:i + (2 if i + 1 < len(argv) and not str(argv[i + 1]).startswith("--") else 1)]
            if v is not None:
                argv += [k] + ([str(v)] if v != "" else [])
    else:
        argv = [x.replace("{config}", r.cfg_path) for x in argv]
    if q.fake:
        r.argv = [sys.executable, str(Path(__file__).resolve()), "--fake-server", "--port", str(port)] + (
            ["--fake-config", r.cfg_path] if r.cfg_path else [])
        r.env = dict(r.env)
        for k, v in (job.get("env") or {}).items():   # a config-less (external) engine reads its env deltas from the process
            if v is None:
                r.env.pop(k, None)
            else:
                r.env[k] = str(v)
        if r.cfg_path is None:
            r.env["FAKE_NO_CONFIG"] = "1"
    else:
        r.argv = argv
    c = job["client"]
    for p in client_passes(job):
        mp = {**m, "out": str(raw / f"{job['name']}-p{p['n']}.json"), "pass": p["n"], "label": f"{job['name']}-p{p['n']}",
              "passlabel": p["label"]}
        r.clients.append({"n": p["n"], "label": p["label"], "out": mp["out"], "log": str(raw / f"{job['name']}-p{p['n']}.log"),
                          "argv": subst(c["cmd"], mp) + [subst(x, mp) for x in p["extra"]], "est_min": p["est_min"]})
    for i, cmd in enumerate(job.get("post") or [], 1):
        mp = {**m, "out": str(raw / f"{job['name']}-post{i}.txt")}
        r.post.append({"argv": subst(cmd, mp), "out": mp["out"]})
    return r


def shell_join(argv: list[str]) -> str:
    import shlex
    return " ".join(shlex.quote(a) for a in argv)


# ----------------------------------------------------------------------------------------------------------------- process control
def session_pids(procs: list[dict], sid: int) -> list[int]:
    return [p["pid"] for p in procs if p["sid"] == sid]


def send(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def stop_session(proc: subprocess.Popen, probe: Probe, term_wait: float = 90, member_wait: float = 45, kill_wait: float = 15,
                 poll: float = 0.5) -> dict:
    """Stop an engine started with start_new_session: SIGTERM the leader PID, wait; then every remaining member of its session by
    PID (SIGTERM, wait), SIGKILL by PID only for what is still there.  Returns what it had to do."""
    info = {"leader": proc.pid, "sigterm": 0, "sigkill": 0}
    sid = proc.pid

    def alive() -> list[int]:
        if proc.poll() is None:
            return [proc.pid] + [x for x in session_pids(probe.processes(), sid) if x != proc.pid]
        return session_pids(probe.processes(), sid)

    def wait_gone(limit: float) -> bool:
        t0 = probe.now()
        while alive():
            if probe.now() - t0 >= limit:
                return False
            probe.sleep(poll)
        return True

    if proc.poll() is None:
        send(proc.pid, signal.SIGTERM)
        info["sigterm"] += 1
    t0 = probe.now()
    while proc.poll() is None and probe.now() - t0 < term_wait:
        probe.sleep(poll)
    rest = [x for x in alive()]
    if rest:
        for x in rest:
            send(x, signal.SIGTERM)
            info["sigterm"] += 1
        wait_gone(member_wait)
    rest = alive()
    if rest:
        for x in rest:
            send(x, signal.SIGKILL)
            info["sigkill"] += 1
        wait_gone(kill_wait)
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass
    info["gone"] = not alive()
    return info


def http_ok(port: int, path: str = "/health", timeout: float = 3.0) -> bool:
    try:
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        c.request("GET", path)
        ok = c.getresponse().status == 200
        c.close()
        return ok
    except Exception:                                                     # noqa: BLE001
        return False


PATH_FLAGS = ("--pack", "--native", "--ple-gguf", "--mtp", "--expert-profile", "--mtp-draft-vocab")


def preflight(jobs: list[dict], plan: dict, q: "Queue") -> list[str]:
    """What is missing on disk before any engine is started (a night lost to a wrong path is the worst outcome): the engine binary
    or command, the working directory, and the model / pack / profile files a config names.  One line per problem, deduplicated."""
    seen, out = set(), []
    for j in jobs:
        r = render_job(j, plan, q)
        checks = [("working directory", r.cwd, os.path.isdir)]
        exe = (r.cfg or {}).get("exe") or (r.argv[0] if r.argv and not q.fake else None)
        if exe:
            checks.append(("engine binary", exe, os.path.isfile))
        if r.cfg:
            a = r.cfg.get("args", [])
            for k, v in zip(a, a[1:]):
                if k in PATH_FLAGS and str(v).startswith("/"):
                    checks.append((k, str(v), os.path.exists))
        for p in r.post + r.clients:
            checks.append(("script", p["argv"][1], os.path.isfile))
        for what, path, test in checks:
            if (what, path) not in seen and not test(path):
                seen.add((what, path))
                out.append(f"{what} missing: {path}  (first needed by {j['name']})")
    if not (q.repo / ".venv" / "bin" / "python").exists():
        out.append(f"{q.repo}/.venv/bin/python missing: the Strata server of the pancake / Strata-rdna3 jobs would run with {q.venv_py}")
    return out


# ----------------------------------------------------------------------------------------------------------------- the queue
class Queue:
    def __init__(self, plan: dict, run_dir: Path, args, probe: Probe | None = None):
        self.plan, self.run_dir, self.args = plan, Path(run_dir), args
        self.probe = probe or Probe()
        self.repo = Path(args.repo).resolve()
        self.paths = default_paths(plan, self.repo)
        self.fake = bool(args.fake)
        venv = self.repo / ".venv" / "bin" / "python"
        self.venv_py = str(venv) if venv.exists() else sys.executable
        d = plan.get("defaults") or {}
        self.guard = Guard(self.probe, args.load_max if args.load_max is not None else d.get("load_max", 1.0),
                           args.mem_gib if args.mem_gib is not None else d.get("mem_gib", 48.0),
                           enabled=(not args.no_guard) and (args.guard or not self.fake))
        self.guard_wait_s = (args.guard_wait_min if args.guard_wait_min is not None else d.get("guard_wait_min", 30)) * 60
        self.guard_poll_s = args.guard_poll_s
        self.start_timeout_s = (args.start_timeout_min if args.start_timeout_min is not None else d.get("start_timeout_min", 25)) * 60
        self.defaults = d
        self.state_path = self.run_dir / "state.json"
        self.state = {"schema": SCHEMA, "plan": plan.get("name", ""), "jobs": {}}
        self.health_poll_s = 0.05 if self.fake else 2.0

    # ---- log / state
    def log(self, msg: str) -> None:
        line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            with open(self.run_dir / "queue.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def load_state(self) -> None:
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))

    def finished(self, name: str) -> bool:
        st = (self.state["jobs"].get(name) or {}).get("status")
        return st == "done" or (st in ("failed", "timeout") and not self.args.retry_failed)

    # ---- RAM and port after a stop
    def wait_released(self, port: int, limit_s: float) -> dict:
        t0 = self.probe.now()
        while True:
            ram_ok = self.probe.mem_gib() >= self.guard.mem_gib or self.fake or not self.guard.enabled
            port_free = not self.probe.port_busy(port)
            if (ram_ok and port_free) or self.probe.now() - t0 >= limit_s:
                return {"ram_back": ram_ok, "port_free": port_free, "mem_gib": round(self.probe.mem_gib(), 1),
                        "waited_s": round(self.probe.now() - t0, 1)}
            self.probe.sleep(self.guard_poll_s if self.fake else 3.0)

    # ---- one job
    def run_job(self, job: dict) -> dict:
        name = job["name"]
        rec = {"status": "running", "group": job.get("group"), "engine": job["engine"], "arm": job.get("arm"),
               "arm_label": job.get("arm_label"), "compare_group": job.get("compare_group"), "env": job.get("env"),
               "args": job.get("args"), "client_kind": job["client"].get("kind"), "est_min": job_est_min(job, self.plan),
               "start": time.time(), "passes": [], "post": [], "reason": ""}
        self.state["jobs"][name] = rec
        self.save()
        r = render_job(job, self.plan, self)
        quiet, reasons = self.guard.wait(r.port, self.guard_wait_s, self.guard_poll_s, self.log)
        if not quiet:
            rec.update(status="skipped", reason="machine not quiet after %.0f min: %s" % (self.guard_wait_s / 60, "; ".join(reasons)),
                       end=time.time())
            self.log(f"SKIP {name}: {rec['reason']}")
            self.save()
            return rec
        (self.run_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "configs").mkdir(parents=True, exist_ok=True)
        if r.cfg is not None:
            Path(r.cfg_path).write_text(json.dumps(r.cfg, indent=1), encoding="utf-8")
        timeout_s = float(job.get("timeout_min") or max(30.0, 2.5 * rec["est_min"])) * 60
        deadline = self.probe.now() + timeout_s
        self.log(f"START {name} (est {rec['est_min']:.0f} min, timeout {timeout_s / 60:.0f} min): {shell_join(r.argv)}")
        logf = open(r.files["server_log"], "ab")
        proc = None
        try:
            proc = subprocess.Popen(r.argv, cwd=r.cwd, env=r.env, stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            rec["pid"] = proc.pid
            self.save()
            ready, why = self.wait_ready(proc, r.port, min(self.start_timeout_s, max(1.0, deadline - self.probe.now())))
            if not ready:
                rec.update(status="timeout" if why == "timeout" else "failed", reason=f"engine not ready: {why}")
            else:
                self.run_clients(job, r, rec, deadline)
        except KeyboardInterrupt:
            rec.update(status="interrupted", reason="interrupted")
            raise
        except Exception as e:                                           # noqa: BLE001
            rec.update(status="failed", reason=f"runner error: {e!r}")
        finally:
            if proc is not None:
                rec["stop"] = stop_session(proc, self.probe, term_wait=3 if self.fake else 90, member_wait=3 if self.fake else 45,
                                           kill_wait=3 if self.fake else 15, poll=0.02 if self.fake else 0.5)
                rec["released"] = self.wait_released(r.port, 5 if self.fake else 900)
                if not rec["stop"]["gone"] or not rec["released"]["port_free"]:
                    rec["stop_problem"] = True
            logf.close()
            rec["end"] = time.time()
            rec["wall_s"] = round(rec["end"] - rec["start"], 1)
        if rec["status"] == "running":
            rec["status"] = "done" if rec["passes"] and all(p["rc"] == 0 for p in rec["passes"]) else "failed"
            if rec["status"] == "failed":
                rec["reason"] = rec["reason"] or "a client pass failed"
        for p in r.post:
            self.run_post(p, rec)
        self.log(f"{rec['status'].upper()} {name} in {rec.get('wall_s', 0) / 60:.1f} min" + (f": {rec['reason']}" if rec["reason"] else ""))
        if rec.get("stop_problem"):
            self.log(f"WARNING {name}: the engine session or the port is still there after the stop: {rec.get('stop')} {rec.get('released')}")
        self.save()
        return rec

    def wait_ready(self, proc: subprocess.Popen, port: int, limit_s: float) -> tuple[bool, str]:
        t0 = self.probe.now()
        while True:
            if proc.poll() is not None:
                return False, f"engine exited with code {proc.returncode} before /health"
            if http_ok(port):
                return True, ""
            if self.probe.now() - t0 >= limit_s:
                return False, "timeout"
            self.probe.sleep(self.health_poll_s)

    def run_clients(self, job: dict, r: Rendered, rec: dict, deadline: float) -> None:
        for c in r.clients:
            remaining = deadline - self.probe.now()
            ent = {"n": c["n"], "label": c["label"], "file": os.path.relpath(c["out"], self.run_dir), "rc": None, "wall_s": None}
            rec["passes"].append(ent)
            if remaining <= 0:
                ent["rc"] = -1
                rec.update(status="timeout", reason="job timeout before pass %d" % c["n"])
                return
            limit = min(float(job["client"].get("timeout_min") or 1e9) * 60, remaining)
            t0 = time.time()
            with open(c["log"], "wb") as lf:
                p = subprocess.Popen(c["argv"], cwd=str(self.repo), stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                                     start_new_session=True)
                try:
                    ent["rc"] = p.wait(timeout=limit)
                except subprocess.TimeoutExpired:
                    stop_session(p, self.probe, term_wait=5, member_wait=5, kill_wait=5, poll=0.1)
                    ent["rc"] = -1
                    rec.update(status="timeout", reason=f"pass {c['n']} ({c['label']}) exceeded {limit / 60:.1f} min")
                except KeyboardInterrupt:
                    stop_session(p, self.probe, term_wait=5, member_wait=5, kill_wait=5, poll=0.1)
                    raise
            ent["wall_s"] = round(time.time() - t0, 1)
            self.log(f"  pass {c['n']} ({c['label']}): rc {ent['rc']}, {ent['wall_s']} s")
            self.save()
            if ent["rc"] != 0:
                if rec["status"] == "running":
                    rec.update(status="failed", reason=f"pass {c['n']} ({c['label']}) exited with code {ent['rc']}")
                return

    def run_post(self, p: dict, rec: dict) -> None:
        ent = {"argv": p["argv"], "out": os.path.relpath(p["out"], self.run_dir), "rc": None}
        try:
            with open(p["out"], "wb") as f:
                ent["rc"] = subprocess.run(p["argv"], cwd=str(self.repo), stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                                           timeout=600).returncode
        except Exception as e:                                           # noqa: BLE001
            ent["error"] = repr(e)[:200]
        rec["post"].append(ent)

    # ---- the loop
    def run(self, jobs: list[dict]) -> int:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        todo = [j for j in jobs if not self.finished(j["name"])]
        total = sum(job_est_min(j, self.plan) for j in todo)
        self.log(f"night queue: {len(todo)} of {len(jobs)} selected jobs to run, estimated {fmt_hm(total)}; run dir {self.run_dir}")
        try:
            (self.run_dir / "night_queue.pid").write_text(str(os.getpid()))
        except OSError:
            pass

        def on_term(*_):
            raise KeyboardInterrupt()

        old_term = signal.signal(signal.SIGTERM, on_term)
        rc = 0
        try:
            for j in todo:
                self.run_job(j)
                write_results(self.plan, self.state, self.run_dir)
        except KeyboardInterrupt:
            self.log("interrupted: the engine was stopped; --resume continues with the unfinished jobs")
            rc = 130
        finally:
            signal.signal(signal.SIGTERM, old_term)
        self.save()
        write_results(self.plan, self.state, self.run_dir)
        if rc:
            return rc
        bad = [n for n, v in self.state["jobs"].items() if v["status"] in ("failed", "timeout", "skipped") and any(j["name"] == n for j in jobs)]
        self.log("finished" + (f", not done: {', '.join(bad)}" if bad else ", all jobs done"))
        return 1 if bad else 0


# ----------------------------------------------------------------------------------------------------------------- dry run
def fmt_hm(minutes: float) -> str:
    m = int(round(minutes))
    return f"{m // 60} h {m % 60:02d} min ({m} min)"


def dry_run(plan: dict, jobs: list[dict], q: Queue) -> str:
    out = [f"night queue plan {plan.get('name', '')!r}: {len(jobs)} jobs" + (" (fake engine)" if q.fake else ""),
           f"run dir: {q.run_dir}", f"guard: load < {q.guard.load_max:g}, MemAvailable >= {q.guard.mem_gib:g} GiB, wait up to "
           f"{q.guard_wait_s / 60:g} min, port free, no other strata / llama-server / build / benchmark process", ""]
    tot, per_group = 0.0, {}
    for i, j in enumerate(jobs, 1):
        r = render_job(j, plan, q)
        est = job_est_min(j, plan)
        tot += est
        per_group[j.get("group", "")] = per_group.get(j.get("group", ""), 0) + est
        out.append(f"[{i:02d}] {j['name']}  group={j.get('group')}  engine={j['engine']}  est {est:.0f} min")
        if j.get("arm_label"):
            out.append(f"     arm {j['arm']}: {j['arm_label']}")
        out.append("     env delta:  " + (json.dumps(j.get("env")) if j.get("env") else "-"))
        out.append("     args delta: " + (json.dumps(j.get("args")) if j.get("args") else "-"))
        if r.cfg_path:
            out.append(f"     config: {r.cfg_path} (written at the start of the job)")
        out.append(f"     start:  (cd {r.cwd} && setsid {shell_join(r.argv)})")
        for c in r.clients:
            out.append(f"     client pass {c['n']} ({c['label']}): {shell_join(c['argv'])}")
        for p in r.post:
            out.append(f"     post: {shell_join(p['argv'])}")
        out.append("")
    if not q.fake:
        pf = preflight(jobs, plan, q)
        out.append("preflight: " + ("everything the jobs need exists" if not pf else f"{len(pf)} problem(s), a real run would stop here:"))
        out += ["  " + x for x in pf] + [""]
    out.append("estimated total: " + fmt_hm(tot) + "  (per group: " + ", ".join(f"{g or '-'} {fmt_hm(m)}" for g, m in per_group.items()) + ")")
    return "\n".join(out)


# ----------------------------------------------------------------------------------------------------------------- RESULTS.md
def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def f1(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def pct(a, b):
    return "-" if a is None or not b else f"{100.0 * (a - b) / b:+.1f}%"


def md_row(cells) -> str:
    return "| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in cells) + " |"


def md_table(head, rows) -> list[str]:
    return [md_row(head), md_row(["---"] * len(head))] + [md_row(r) for r in rows] + [""]


def pass_roles(rec: dict, job_client: dict) -> dict:
    """{pass number: 'warm-up' | 'kept' | 'extra'}: pass 1 is the warm-up for the decode client (client.warmup_passes, default 1),
    the last client.keep_last (default 2) of the others are kept; for other clients every pass is kept."""
    nums = [p["n"] for p in rec.get("passes", [])]
    kind = job_client.get("kind")
    warm = int(job_client.get("warmup_passes", 1 if kind == "decode" else 0))
    keep = int(job_client.get("keep_last", 2 if kind == "decode" else 10 ** 6))
    rest = nums[warm:]
    kept = set(rest[-keep:]) if keep < len(rest) else set(rest)
    return {n: ("warm-up" if i < warm else "kept" if n in kept else "extra") for i, n in enumerate(nums)}


def read_json(run_dir: Path, rel: str):
    try:
        return json.loads((run_dir / rel).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def decode_stats(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("tps") is not None]
    acc, dr = sum(r.get("accepted") or 0 for r in ok), sum(r.get("drafted") or 0 for r in ok)
    return {"n": len(ok), "all": med(r["tps"] for r in ok), "code": med(r["tps"] for r in ok if r["class"] == "code"),
            "prose": med(r["tps"] for r in ok if r["class"] == "prose"), "accept": 100.0 * acc / dr if dr else None,
            "pps": med(r.get("prompt_tps") for r in ok), "ttft": med((r["prompt_ms"] / 1000.0) if r.get("prompt_ms") is not None else None for r in ok)}


def job_title(v: dict) -> str:
    d = [f"{k}={'(removed)' if x is None else x}" for k, x in (v.get("env") or {}).items()] + [f"{k} {x}".strip() for k, x in (v.get("args") or {}).items()]
    return ", ".join(d) or "preset as is"


def write_results(plan: dict, state: dict, run_dir: Path) -> Path:
    jobs_by_name = {j["name"]: j for j in plan["_jobs"]}
    order = [j["name"] for j in plan["_jobs"] if j["name"] in state["jobs"]]
    L = [f"# Night queue results ({plan.get('name', '')})", "",
         f"Written {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by tools/pancake/night_queue.py from `state.json`. Numbers only, no "
         "interpretation; nothing here is a finding until it is read against the pass-by-pass tables. **Pass 1 of a decode job is the warm-up "
         "and is excluded**; only the last 2 passes after it are kept (median over the requests of those passes). Estimated durations are "
         "the plan's, not measurements.", ""]
    cnt = {}
    for n in order:
        cnt[state["jobs"][n]["status"]] = cnt.get(state["jobs"][n]["status"], 0) + 1
    L += ["Status: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())) + f"; {len(plan['_jobs']) - len(order)} not started.", ""]
    L += ["## Jobs", ""]
    rows = []
    for n in order:
        v = state["jobs"][n]
        rows.append([n, v.get("group"), v.get("engine"), job_title(v), v["status"] + (f": {v['reason'][:120]}" if v.get("reason") else ""),
                     f1((v.get("wall_s") or 0) / 60.0), f1(v.get("est_min"), 0), "yes" if v.get("stop_problem") else ""])
    L += md_table(["job", "group", "engine", "delta (env / args)", "status", "wall min", "est min", "stop problem"], rows)

    # decode jobs
    dec = [n for n in order if state["jobs"][n].get("client_kind") == "decode"]
    if dec:
        L += ["## Decode (tok/s, the five cmp_bench prompts, thinking off, greedy)", "",
              "Per pass. `role` = warm-up (excluded) / kept / extra (not used). Overall, code (js, lru) and prose (story, tips, moe) are "
              "medians over the requests of that pass; acceptance = accepted / drafted tokens; prompt tok/s and TTFT are the engine's own "
              "(`timings.prompt_per_second`, `prompt_ms`) on these short prompts.", ""]
        rows = []
        per_job = {}
        for n in dec:
            v, cl = state["jobs"][n], jobs_by_name[n]["client"]
            roles = pass_roles(v, cl)
            kept_rows = []
            for p in v["passes"]:
                d = read_json(run_dir, p["file"])
                rws = (d or {}).get("rows") or []
                s = decode_stats(rws)
                role = roles.get(p["n"], "")
                if role == "kept":
                    kept_rows += rws
                rows.append([n, p["n"], "warm-up (excluded)" if role == "warm-up" else role, f1(s["all"]), f1(s["code"]), f1(s["prose"]),
                             f1(s["accept"]), f1(s["pps"], 0), f1(s["ttft"], 2), "rc %s" % p["rc"] if p["rc"] else ""])
            per_job[n] = decode_stats(kept_rows)
        L += md_table(["job", "pass", "role", "overall", "code", "prose", "accept %", "prompt tok/s", "TTFT s", "note"], rows)
        L += ["### Decode per job (kept passes only)", ""]
        rows = [[n, state["jobs"][n].get("arm") or "", job_title(state["jobs"][n]), s["n"], f1(s["all"]), f1(s["code"]), f1(s["prose"]),
                 f1(s["accept"]), f1(s["pps"], 0), f1(s["ttft"], 2)] for n, s in per_job.items()]
        L += md_table(["job", "arm", "delta", "requests", "overall", "code", "prose", "accept %", "prompt tok/s", "TTFT s"], rows)
        groups = {}
        for n in dec:
            g = state["jobs"][n].get("compare_group")
            if g:
                groups.setdefault(g, []).append(n)
        for g, names in groups.items():
            arms = {}
            for n in names:
                arms.setdefault(state["jobs"][n].get("arm"), []).append(n)
            ref = next(iter(arms))
            L += [f"### Per arm: {g}", "",
                  f"Servers of one arm in run order; the medians pool the kept requests of all its servers; delta against arm {ref}. "
                  "Order of the servers: " + " ".join(f"{state['jobs'][n].get('arm')}" for n in names) + ".", ""]
            pooled = {}
            for a, ns in arms.items():
                rr = []
                for n in ns:
                    roles = pass_roles(state["jobs"][n], jobs_by_name[n]["client"])
                    for p in state["jobs"][n]["passes"]:
                        if roles.get(p["n"]) == "kept":
                            rr += ((read_json(run_dir, p["file"]) or {}).get("rows") or [])
                pooled[a] = decode_stats(rr)
            rows = []
            for a, ns in arms.items():
                s, r0 = pooled[a], pooled[ref]
                lab = state["jobs"][ns[0]].get("arm_label") or ""
                rows.append([a, lab, len(ns), s["n"], f1(s["all"]), pct(s["all"], r0["all"]), f1(s["code"]), pct(s["code"], r0["code"]),
                             f1(s["prose"]), pct(s["prose"], r0["prose"]), f1(s["accept"]),
                             " / ".join(f1(per_job[n]["all"]) for n in ns)])
            L += md_table(["arm", "meaning", "servers", "requests", "overall", "vs " + ref, "code", "vs " + ref, "prose", "vs " + ref,
                           "accept %", "overall per server"], rows)

    # ttft_bench jobs
    tt = [n for n in order if state["jobs"][n].get("client_kind") == "ttft"]
    if tt:
        L += ["## TTFT (tools/pancake/ttft_bench.py)", "", "Scenario A: single prompts, run 1 of each size discarded by the tool, medians of the "
              "rest; TTFT is the client's clock, the engine columns are the server's own `timings`. Scenario B: the 10-turn agent session.", ""]
        for n in tt:
            for p in state["jobs"][n]["passes"]:
                d = read_json(run_dir, p["file"])
                if not d:
                    L += [f"**{n}**: no result file ({p['file']}).", ""]
                    continue
                L += [f"**{n}** (load start {d.get('load', {}).get('start')}, end {d.get('load', {}).get('end')}, warning "
                      f"{d.get('load', {}).get('warning')})", ""]
                a = d.get("scenario_a")
                if a:
                    rows = []
                    for e in a["sizes"]:
                        s = e.get("summary") or {}
                        t = s.get("ttft_s") or {}
                        rows.append([e["label"], e["status"][:60], s.get("prompt_tokens"), t.get("median"), t.get("min"), t.get("max"),
                                     s.get("engine_prompt_ms"), s.get("engine_prompt_tps"), s.get("n")])
                    L += md_table(["size", "status", "prompt tokens", "TTFT median s", "min", "max", "engine prompt ms", "engine prompt tok/s", "kept runs"], rows)
                b = d.get("scenario_b")
                if b:
                    L += session_table(b["summary"])
    ses = [n for n in order if state["jobs"][n].get("client_kind") == "session"]
    if ses:
        L += ["## Agent session with side requests (tools/pancake/nq_clients.py session)", "",
              "Each turn adds about 1,500 words and resends the whole conversation; one short unrelated side request is sent between two "
              "turns. TTFT is the client's. Reuse = prompt tokens served from the server's cache.", ""]
        rows = []
        for n in ses:
            v = state["jobs"][n]
            for p in v["passes"]:
                d = read_json(run_dir, p["file"])
                if not d:
                    rows.append([n, job_title(v), v["status"], "-", "-", "-", "-", "-"])
                    continue
                s = d["summary"]
                rows.append([n, job_title(v), v["status"], s.get("total_ttft_s"), s.get("median_follow_up_ttft_s"), s.get("reuse_pct_follow_up_turns"),
                             f"{s.get('complete_sessions')}/{s.get('sessions')}", (s.get("side_requests") or {}).get("median_ttft_s")])
        L += md_table(["job", "delta", "status", "total TTFT s (median session)", "median follow-up TTFT s", "reuse % follow-up turns",
                       "complete sessions", "side request TTFT s"], rows)
        for n in ses:
            for p in state["jobs"][n]["passes"]:
                d = read_json(run_dir, p["file"])
                if d:
                    L += [f"Per turn, {n}:", ""] + session_table(d["summary"], only_turns=True)
            for post in state["jobs"][n].get("post", []):
                L += [f"Post step of {n}: `{shell_join(post['argv'])}` -> `{post['out']}` (rc {post['rc']}).", ""]

    dp = [n for n in order if state["jobs"][n].get("client_kind") == "depth"]
    if dp:
        L += ["## Depth (single prompts, then a decode at that depth)", "", "Run 1 of each size is discarded; medians of the rest. Acceptance "
              "= accepted / drafted over the kept runs.", ""]
        rows = []
        for n in dp:
            v = state["jobs"][n]
            for p in v["passes"]:
                d = read_json(run_dir, p["file"])
                for e in (d or {}).get("sizes", []):
                    s = e.get("summary") or {}
                    rows.append([n, v.get("arm") or "", job_title(v), e["label"], e["status"][:50], s.get("prompt_tokens"), s.get("ttft_s"),
                                 s.get("prompt_tps"), s.get("decode_tps"), s.get("accept_pct")])
        L += md_table(["job", "arm", "delta", "size", "status", "prompt tokens", "TTFT s", "prompt tok/s", "decode tok/s", "accept %"], rows)

    pr = [n for n in order if state["jobs"][n].get("client_kind") == "prose"]
    if pr:
        L += ["## Prose profile (tools/pancake/prose_gap.py)", "", "Run 1 of each series discarded by the tool; medians of the others. "
              "Tokens per window = tokens / (tokens - accepted drafts), an assumption of the tool.", ""]
        rows = []
        for n in pr:
            v = state["jobs"][n]
            for p in v["passes"]:
                d = read_json(run_dir, p["file"])
                s = (d or {}).get("summary") or {}
                g = lambda k, f="median": ((s.get(k) or {}).get(f))                       # noqa: E731
                rows.append([n, job_title(v), p["label"], d.get("engine_kind") if d else "-", f1(g("tps")), f1(g("tps", "sd")),
                             f1(g("accept_pct")), f1(g("tok_per_window"), 2), f1(g("ms_per_window"), 1)])
        L += md_table(["job", "delta", "prompt", "engine", "tok/s median", "sd", "accept %", "tok / window", "ms / window"], rows)

    bad = [n for n in order if state["jobs"][n]["status"] not in ("done",)]
    if bad:
        L += ["## Not done", ""] + md_table(["job", "status", "reason"], [[n, state["jobs"][n]["status"], state["jobs"][n].get("reason", "")] for n in bad])
    p = run_dir / "RESULTS.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


def session_table(s: dict, only_turns: bool = False) -> list[str]:
    L = []
    if not only_turns:
        L += [f"Session: total TTFT over the 10 turns (median of complete sessions) {s.get('total_ttft_s')} s; reuse of prompt tokens on follow-up "
              f"turns {s.get('reuse_pct_follow_up_turns')} % (source {', '.join(s.get('reuse_source') or []) or 'not reported'}); complete sessions "
              f"{s.get('complete_sessions')}/{s.get('sessions')}.", ""]
    rows = [[t["turn"], t.get("n"), t.get("ttft_s"), t.get("prompt_tokens"), t.get("reused_tokens"), t.get("read_tokens"), t.get("reread_tokens")]
            for t in s.get("per_turn", [])]
    return L + md_table(["turn", "n", "TTFT s", "prompt tokens", "reused", "read", "re-read"], rows)


# ----------------------------------------------------------------------------------------------------------------- fake server
class FakeHandler(http.server.BaseHTTPRequestHandler):
    """A stdlib stand-in for an engine: /health, /v1/chat/completions (stream and not).  Decode speed depends on the config's env so
    the loop's tables have something to show: deferral removed = prose +8% and code -3%; the first requests of a process are 10%
    slower (a warm-up); a tiny deterministic jitter.  Not an engine and not a measurement of anything."""

    protocol_version = "HTTP/1.1"
    env: dict = {}
    count = 0
    last_prompt = ""
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/health", "/api/health"):
            self._send(200, b'{"status":"ok"}')
        elif self.path.startswith("/v1/models"):
            self._send(200, b'{"data":[{"id":"fake"}]}')
        else:
            self._send(404, b"{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        text = "\n".join(str(m.get("content")) for m in body.get("messages", []))
        cls = "code" if re.search(r"JavaScript|Python class|debounce|LRU", text) else "prose"
        with FakeHandler.lock:
            FakeHandler.count += 1
            k = FakeHandler.count
            prev = FakeHandler.last_prompt
            FakeHandler.last_prompt = text
        env = FakeHandler.env
        base = 130.0 if cls == "code" else 100.0
        if "STRATA_VERIFY_DEFER_CPU" not in env:
            base *= 0.97 if cls == "code" else 1.08
        if k <= 5:
            base *= 0.9
        base *= 1.0 + 0.002 * ((k * 7) % 5 - 2)   # a tiny deterministic jitter
        ntok = min(int(body.get("max_tokens") or 64), 512)
        words = max(1, len(text.split()))
        prompt_n = int(words * 1.06) + 20
        common = len(os.path.commonprefix([prev, text]))
        cache_n = int(common / max(1, len(text)) * prompt_n) if prev and text.startswith(prev[:max(1, len(prev) // 2)]) else 0
        read = prompt_n - cache_n
        drafted = ntok * 2 // 3
        timings = {"predicted_n": ntok, "predicted_per_second": round(base, 3), "predicted_ms": 1000.0 * ntok / base,
                   "draft_n": drafted, "draft_n_accepted": int(drafted * 0.7), "prompt_n": read, "cache_n": cache_n,
                   "prompt_per_second": 2500.0, "prompt_ms": read / 2500.0 * 1000.0}
        usage = {"prompt_tokens": prompt_n, "completion_tokens": ntok, "total_tokens": prompt_n + ntok,
                 "prompt_tokens_details": {"cached_tokens": cache_n}}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            chunks = [{"choices": [{"delta": {"role": "assistant"}}]},
                      {"choices": [{"delta": {"content": "ok "}}]},
                      {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": usage, "timings": timings}]
            for c in chunks:
                self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
        else:
            self._send(200, json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                                        "usage": usage, "timings": timings}).encode())


def fake_server_main(port: int, config: str) -> int:
    if config and os.path.exists(config):
        FakeHandler.env = (json.loads(Path(config).read_text(encoding="utf-8")).get("env") or {})
    else:
        FakeHandler.env = {k: v for k, v in os.environ.items() if k.startswith("STRATA_")}
        if os.environ.get("FAKE_NO_CONFIG") and "STRATA_VERIFY_DEFER_CPU" not in FakeHandler.env:
            FakeHandler.env["STRATA_VERIFY_DEFER_CPU"] = "1"

    class S(socketserver.ThreadingMixIn, http.server.HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    srv = S(("127.0.0.1", port), FakeHandler)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=srv.shutdown, daemon=True).start())
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


# ----------------------------------------------------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default=str(DEFAULT_PLAN))
    ap.add_argument("--run-dir", default=None, help="state, logs, raw results, RESULTS.md (default bench/results/night-queue)")
    ap.add_argument("--repo", default=str(REPO), help="the checkout that has build-hip and .venv (default: this one)")
    ap.add_argument("--resume", action="store_true", help="continue in --run-dir: skip the finished jobs (done, failed, timeout)")
    ap.add_argument("--retry-failed", action="store_true", help="with --resume: run failed / timed-out jobs again")
    ap.add_argument("--fresh", action="store_true", help="move an existing run dir with a state file aside and start again")
    ap.add_argument("--dry-run", action="store_true", help="print every job, command and the order; run and write nothing")
    ap.add_argument("--only", action="append", default=[], help="job name, glob or comma list (repeatable)")
    ap.add_argument("--group", action="append", default=[], help="only the jobs of this group (repeatable)")
    ap.add_argument("--skip-group", action="append", default=[], help="leave this group out (repeatable)")
    ap.add_argument("--results-only", action="store_true", help="rewrite RESULTS.md from state.json and exit")
    ap.add_argument("--fake", action="store_true", help="a stdlib fake server instead of the engine (no GPU, no guard unless --guard)")
    ap.add_argument("--guard", action="store_true", help="keep the quiet-machine guard in --fake mode")
    ap.add_argument("--no-guard", action="store_true", help="turn the guard off (not for the night run)")
    ap.add_argument("--load-max", type=float, default=None, help="1-minute load must be below this (default 1.0)")
    ap.add_argument("--mem-gib", type=float, default=None, help="MemAvailable must be at least this many GiB (default 48)")
    ap.add_argument("--guard-wait-min", type=float, default=None, help="minutes to wait for a quiet machine before skipping a job (default 30)")
    ap.add_argument("--guard-poll-s", type=float, default=10.0)
    ap.add_argument("--start-timeout-min", type=float, default=None, help="minutes to wait for /health (default 25)")
    ap.add_argument("--skip-preflight", action="store_true", help="start even if a binary, model file or script is missing")
    ap.add_argument("--list", action="store_true", help="one line per job and exit")
    ap.add_argument("--fake-server", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--fake-config", default="", help=argparse.SUPPRESS)
    ap.add_argument("--port", type=int, default=0, help=argparse.SUPPRESS)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.fake_server:
        return fake_server_main(a.port, a.fake_config)
    try:
        plan = load_plan(Path(a.plan))
    except PlanError as e:
        print(f"plan error: {e}", file=sys.stderr)
        return 2
    run_dir = Path(a.run_dir) if a.run_dir else Path(a.repo) / "bench" / "results" / "night-queue"
    jobs = select_jobs(plan["_jobs"], a.only, a.group, a.skip_group)
    if not jobs:
        print("no job selected", file=sys.stderr)
        return 2
    q = Queue(plan, run_dir, a)
    if a.list:
        for i, j in enumerate(jobs, 1):
            print(f"{i:02d} {j['name']:<34} {j.get('group', ''):<10} {j['engine']:<16} est {job_est_min(j, plan):>4.0f} min")
        print("estimated total: " + fmt_hm(sum(job_est_min(j, plan) for j in jobs)))
        return 0
    if a.dry_run:
        print(dry_run(plan, jobs, q))
        return 0
    if a.results_only:
        q.load_state()
        print(write_results(plan, q.state, run_dir))
        return 0
    if (run_dir / "state.json").exists():
        if a.resume:
            q.load_state()
        elif a.fresh:
            run_dir.rename(run_dir.with_name(run_dir.name + ".old-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        else:
            print(f"{run_dir}/state.json exists: use --resume to continue, --fresh to start again, or another --run-dir", file=sys.stderr)
            return 2
    elif a.resume:
        print(f"--resume: no state file in {run_dir}", file=sys.stderr)
        return 2
    if not a.fake and not a.skip_preflight:
        pf = preflight([j for j in jobs if not q.finished(j["name"])], plan, q)
        if pf:
            print("preflight failed (nothing was started; --skip-preflight overrides):\n  " + "\n  ".join(pf), file=sys.stderr)
            return 2
    return q.run(jobs)


if __name__ == "__main__":
    sys.exit(main())
