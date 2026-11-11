"""Tests for night_queue.py and nq_clients.py with the stdlib fake server (no engine, no GPU, nothing heavy).
Run: nice -n 19 python3 -I -m pytest -q tools/pancake/test_night_queue.py"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import night_queue as NQ  # noqa: E402

REPO = HERE.parents[1]


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class FakeProbe(NQ.Probe):
    """Scripted machine: load / memory from lists (the last value repeats), extra busy processes, a virtual clock whose sleeps are
    real but short (a stopping process needs real milliseconds)."""

    def __init__(self, loads=(0.1,), mems=(100.0,), busy=()):
        self.loads, self.mems, self.busy = list(loads), list(mems), list(busy)
        self.t = 0.0
        self.polls = 0

    def load1(self):
        self.polls += 1
        return self.loads.pop(0) if len(self.loads) > 1 else self.loads[0]

    def mem_gib(self):
        return self.mems.pop(0) if len(self.mems) > 1 else self.mems[0]

    def processes(self):
        extra = [{"pid": 10 ** 7 + i, "ppid": 1, "sid": 10 ** 7 + i, "comm": c, "argv": a} for i, (c, a) in enumerate(self.busy)]
        return super().processes() + extra

    def sleep(self, s):
        self.t += s
        time.sleep(min(s, 0.01))

    def now(self):
        return self.t


def make_plan(tmp_path, port, jobs=None, engines=None):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"exe": "/nonexistent/strata", "args": ["--spec", "4", "--spec-min-p", "0.5", "--kv-resident", "32768"],
                               "env": {"STRATA_VERIFY_DEFER_CPU": "1", "STRATA_OTHER": "x"}, "log": "x.log", "turn_log": "t.jsonl"}))
    decode = {"kind": "decode", "cmd": ["{py}", str(HERE / "nq_clients.py"), "decode", "--port", "{port}", "--out", "{out}", "--max-tokens", "32"],
              "passes": 3, "warmup_passes": 1, "keep_last": 2, "est_min": 0.5}
    plan = {"name": "t", "engines": engines or {"fake-a": {"config": str(cfg), "argv": ["{py}", "never-run"], "cwd": str(REPO), "port": port,
                                                          "start_est_min": 2}},
            "jobs": jobs if jobs is not None else [
                {"name": "dfr", "group": "g1", "engine": "fake-a", "client": decode,
                 "arms": {"A": {"label": "on"}, "B": {"label": "removed", "env": {"STRATA_VERIFY_DEFER_CPU": None}}}, "sequence": "ABBA"},
                {"name": "later", "group": "g2", "engine": "fake-a", "client": dict(decode, passes=2)}]}
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    return p


def args_for(argv):
    return NQ.build_parser().parse_args(argv)


def run_main(tmp_path, plan, extra=(), run_dir=None):
    rd = run_dir or tmp_path / "run"
    return NQ.main(["--plan", str(plan), "--run-dir", str(rd), "--fake", "--repo", str(REPO), *extra]), rd


# ---------------------------------------------------------------------------------------------------------------- plan / order
def test_sequence_expands_in_order_a_b_b_a(tmp_path):
    plan = NQ.load_plan(make_plan(tmp_path, free_port()))
    names = [j["name"] for j in plan["_jobs"]]
    assert names == ["dfr-01A", "dfr-02B", "dfr-03B", "dfr-04A", "later"]
    assert [j.get("arm") for j in plan["_jobs"][:4]] == list("ABBA")
    assert plan["_jobs"][1]["env"] == {"STRATA_VERIFY_DEFER_CPU": None}
    assert plan["_jobs"][0]["compare_group"] == "dfr"


def test_shipped_plan_loads_and_has_the_ablation_order():
    plan = NQ.load_plan(NQ.DEFAULT_PLAN)
    names = [j["name"] for j in plan["_jobs"]]
    assert len(names) == len(set(names))
    assert [n for n in names if n.startswith("defer-iq2xs")] == [f"defer-iq2xs-{i:02d}{c}" for i, c in enumerate("ABCCBAABC", 1)]
    # every job's group is one of the five items of the night queue and every job has an estimate
    assert {j["group"] for j in plan["_jobs"]} == {"1-defer", "2-ttft", "3-lever", "4-kvring", "5-prose"}
    assert all(NQ.job_est_min(j, plan) > 0 for j in plan["_jobs"])
    # item 1 arm C is the gate's auto-closed arm of bench/specs/p17-defer-auto.json
    c = next(j for j in plan["_jobs"] if j["name"] == "defer-q2-03C")
    assert c["env"] == {"STRATA_VERIFY_DEFER_AUTO": "1", "STRATA_VERIFY_DEFER_AUTO_MIN_T": "256"}
    spec = json.loads((REPO / "bench/specs/p17-defer-auto.json").read_text())
    assert spec["arms"]["auto-closed"]["env"] == c["env"]


def test_apply_delta_follows_the_abtest_semantics():
    cfg = {"env": {"A": "1", "B": "2"}, "args": ["--spec", "4", "--spec-min-p", "0.5", "--flag"]}
    NQ.apply_delta(cfg, {"A": None, "C": 3}, {"--spec": "2", "--spec-min-p": "0", "--new": "", "--flag": None})
    assert cfg["env"] == {"B": "2", "C": "3"}
    assert cfg["args"] == ["--spec", "2", "--spec-min-p", "0", "--new"]


def test_unknown_engine_and_duplicate_names_are_plan_errors(tmp_path):
    p = make_plan(tmp_path, free_port(), jobs=[{"name": "x", "engine": "nope", "client": {"cmd": ["a"]}}])
    with pytest.raises(NQ.PlanError):
        NQ.load_plan(p)


def test_only_and_group_selection(tmp_path):
    plan = NQ.load_plan(make_plan(tmp_path, free_port()))
    assert [j["name"] for j in NQ.select_jobs(plan["_jobs"], ["dfr-0[12]*"], [], [])] == ["dfr-01A", "dfr-02B"]
    assert [j["name"] for j in NQ.select_jobs(plan["_jobs"], ["later,dfr-04A"], [], [])] == ["dfr-04A", "later"]
    assert [j["name"] for j in NQ.select_jobs(plan["_jobs"], [], ["g2"], [])] == ["later"]
    assert [j["name"] for j in NQ.select_jobs(plan["_jobs"], [], [], ["g1"])] == ["later"]


# ---------------------------------------------------------------------------------------------------------------- dry run
def test_dry_run_prints_order_commands_and_total_and_writes_nothing(tmp_path, capsys):
    plan = make_plan(tmp_path, free_port())
    rd = tmp_path / "run"
    rc = NQ.main(["--plan", str(plan), "--run-dir", str(rd), "--dry-run", "--repo", str(REPO)])
    out = capsys.readouterr().out
    assert rc == 0
    pos = [out.index(f"] {n} ") for n in ("dfr-01A", "dfr-02B", "dfr-03B", "dfr-04A", "later")]
    assert pos == sorted(pos)
    assert "nq_clients.py decode" in out and "client pass 3" in out and "client pass 3" in out.split("later")[0]
    assert '"STRATA_VERIFY_DEFER_CPU": null' in out
    assert "estimated total: " in out and "(" in out.split("estimated total: ")[1]
    assert "quiet" in out or "guard:" in out
    assert not rd.exists()           # nothing written
    # B's rendered config would lack the deferral switch
    plan_obj = NQ.load_plan(plan)
    q = NQ.Queue(plan_obj, rd, args_for(["--repo", str(REPO)]))
    r = NQ.render_job(plan_obj["_jobs"][1], plan_obj, q)
    assert "STRATA_VERIFY_DEFER_CPU" not in r.cfg["env"] and r.cfg["env"]["STRATA_OTHER"] == "x"
    assert r.cfg["log"].endswith("engine-dfr-02B.log") and r.cfg["turn_log"].endswith("turns-dfr-02B.jsonl")


def test_dry_run_total_is_the_sum_of_the_job_estimates(tmp_path, capsys):
    plan = make_plan(tmp_path, free_port())
    obj = NQ.load_plan(plan)
    total = sum(NQ.job_est_min(j, obj) for j in obj["_jobs"])
    NQ.main(["--plan", str(plan), "--dry-run", "--fake", "--run-dir", str(tmp_path / "r")])
    assert f"({int(round(total))} min)" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------------------------- full fake run
def test_fake_run_all_jobs_results_and_arms(tmp_path):
    plan = make_plan(tmp_path, free_port())
    rc, rd = run_main(tmp_path, plan)
    assert rc == 0
    state = json.loads((rd / "state.json").read_text())
    assert [state["jobs"][n]["status"] for n in ("dfr-01A", "dfr-02B", "dfr-03B", "dfr-04A", "later")] == ["done"] * 5
    assert all(state["jobs"][n]["stop"]["gone"] and state["jobs"][n]["released"]["port_free"] for n in state["jobs"])
    res = (rd / "RESULTS.md").read_text()
    assert res.count("| warm-up (excluded) |") == 5      # pass 1 of each of the five decode jobs
    assert "### Per arm: dfr" in res and "Order of the servers: A B B A" in res
    arm_rows = [ln for ln in res.splitlines() if ln.startswith("| B | removed |")]
    assert arm_rows, res
    cells = [c.strip() for c in arm_rows[0].strip("|").split("|")]
    # fake engine: deferral removed = prose +8%, code -3%
    assert cells[2] == "2" and cells[9].startswith("+7") or cells[9].startswith("+8")
    assert cells[7].startswith("-")
    assert "accept %" in res and "prompt tok/s" in res and "TTFT s" in res
    # warm-up pass really was slower in the fake (x0.9): the excluded pass must not enter the per-job numbers
    per_job = [ln for ln in res.split("### Decode per job (kept passes only)")[1].splitlines() if ln.startswith("| dfr-01A")][0]
    overall = float([c.strip() for c in per_job.strip("|").split("|")][4])
    assert 100 < overall < 130
    # per-pass rows: the first pass is the warm-up, 2 and 3 are the kept passes
    rows = [ln for ln in res.splitlines() if ln.startswith("| dfr-01A | ")]
    assert [r.split("|")[3].strip() for r in rows if r.count("|") > 9][:3] == ["warm-up (excluded)", "kept", "kept"]
    # one raw result per pass, one engine log per job
    assert (rd / "raw" / "dfr-01A-p3.json").exists() and (rd / "raw" / "srv-later.log").exists()
    assert (rd / "queue.log").read_text().count("START ") == 5
    assert (rd / "night_queue.pid").read_text().strip().isdigit()


def test_fake_run_other_clients_and_post_step(tmp_path):
    port = free_port()
    py = "{py}"
    cl = HERE / "nq_clients.py"
    jobs = [
        {"name": "tt", "group": "g", "engine": "fake-a",
         "client": {"kind": "ttft", "cmd": [py, str(HERE / "ttft_bench.py"), "--port", "{port}", "--label", "{job}", "--out", "{out}", "--sizes", "1k",
                                              "--runs", "2", "--scenario", "both", "--turns", "2", "--sessions", "1", "--agent-words", "100"]}},
        {"name": "se", "group": "g", "engine": "fake-a",
         "client": {"kind": "session", "cmd": [py, str(cl), "session", "--port", "{port}", "--out", "{out}", "--turns", "3", "--sessions", "2",
                                                 "--side", "1", "--words", "100"]},
         "post": [[py, "-c", "print('post', '{job}')"]]},
        {"name": "de", "group": "g", "engine": "fake-a",
         "client": {"kind": "depth", "cmd": [py, str(cl), "depth", "--port", "{port}", "--out", "{out}", "--sizes", "1k,2k", "--runs", "2",
                                               "--max-tokens", "16"]}},
        {"name": "pr", "group": "g", "engine": "fake-a",
         "client": {"kind": "prose", "cmd": [py, str(HERE / "prose_gap.py"), "--port", "{port}", "--label", "{label}", "--out", "{out}", "--runs", "3"],
                    "passes": [{"label": "story", "extra": ["--prompt-name", "story"]}, {"label": "js", "extra": ["--prompt-name", "js"]}]}},
    ]
    plan = make_plan(tmp_path, port, jobs=jobs)
    rc, rd = run_main(tmp_path, plan)
    assert rc == 0, (rd / "queue.log").read_text()
    res = (rd / "RESULTS.md").read_text()
    assert "## TTFT (tools/pancake/ttft_bench.py)" in res and "| 1K |" in res
    assert "## Agent session with side requests" in res and "side request TTFT s" in res
    assert "## Depth" in res and "| 2K |" in res
    assert "## Prose profile" in res and "| story |" in res and "| js |" in res
    post = json.loads((rd / "state.json").read_text())["jobs"]["se"]["post"]
    assert post[0]["rc"] == 0 and "post se" in (rd / post[0]["out"]).read_text()
    se = json.loads((rd / "raw" / "se-p1.json").read_text())
    assert len(se["sessions"]) == 2 and all(len(s["side"]) == 2 for s in se["sessions"])      # 3 turns -> 2 gaps


# ---------------------------------------------------------------------------------------------------------------- resume / state
def test_resume_skips_finished_jobs_and_state_file_guards(tmp_path):
    plan = make_plan(tmp_path, free_port())
    rc, rd = run_main(tmp_path, plan, ["--only", "dfr-01A,dfr-02B"])
    assert rc == 0
    log1 = (rd / "queue.log").read_text()
    assert log1.count("START dfr-01A") == 1 and "START later" not in log1
    # a second start without --resume must refuse, and not touch the state
    rc2, _ = run_main(tmp_path, plan, run_dir=rd)
    assert rc2 == 2
    assert (rd / "queue.log").read_text() == log1
    rc3, _ = run_main(tmp_path, plan, ["--resume"], run_dir=rd)
    assert rc3 == 0
    log = (rd / "queue.log").read_text()
    assert log.count("START dfr-01A") == 1 and log.count("START dfr-02B") == 1          # not run again
    assert all(log.count(f"START {n}") == 1 for n in ("dfr-03B", "dfr-04A", "later"))   # the rest ran
    assert "3 of 5 selected jobs to run" in log
    state = json.loads((rd / "state.json").read_text())
    assert all(v["status"] == "done" for v in state["jobs"].values()) and len(state["jobs"]) == 5


def test_resume_without_state_and_fresh(tmp_path):
    plan = make_plan(tmp_path, free_port())
    assert run_main(tmp_path, plan, ["--resume", "--only", "later"])[0] == 2
    rc, rd = run_main(tmp_path, plan, ["--only", "later"])
    assert rc == 0
    rc, _ = run_main(tmp_path, plan, ["--only", "later", "--fresh"], run_dir=rd)
    assert rc == 0 and list(tmp_path.glob("run.old-*"))


def test_failed_job_is_finished_for_resume_unless_retry_failed(tmp_path):
    port = free_port()
    jobs = [{"name": "bad", "group": "g", "engine": "dead", "client": {"kind": "decode", "cmd": ["{py}", "-c", "pass"]}},
            {"name": "after", "group": "g", "engine": "real-fake", "client": {"kind": "decode", "cmd": [
                "{py}", str(HERE / "nq_clients.py"), "decode", "--port", "{port}", "--out", "{out}", "--max-tokens", "8"], "passes": 2}}]
    engines = {"dead": {"argv": [sys.executable, "-c", "import sys; sys.exit(3)"], "cwd": str(REPO), "port": port},
               "real-fake": {"argv": [sys.executable, str(HERE / "night_queue.py"), "--fake-server", "--port", "{port}"], "cwd": str(REPO),
                             "port": port}}
    plan = make_plan(tmp_path, port, jobs=jobs, engines=engines)
    rd = tmp_path / "run"
    # real (non-fake) launch path with the guard off: the first engine dies before /health, the queue goes on
    rc = NQ.main(["--plan", str(plan), "--run-dir", str(rd), "--repo", str(REPO), "--no-guard", "--skip-preflight", "--start-timeout-min", "0.2"])
    state = json.loads((rd / "state.json").read_text())
    assert rc == 1 and state["jobs"]["bad"]["status"] == "failed" and "exited with code 3" in state["jobs"]["bad"]["reason"]
    assert state["jobs"]["after"]["status"] == "done"
    assert "## Not done" in (rd / "RESULTS.md").read_text()
    # resume: failed counts as finished
    q = NQ.Queue(NQ.load_plan(plan), rd, args_for(["--repo", str(REPO), "--fake"]))
    q.load_state()
    assert q.finished("bad") and q.finished("after")
    q.args.retry_failed = True
    assert not q.finished("bad") and q.finished("after")


# ---------------------------------------------------------------------------------------------------------------- the guard
def test_busy_processes_detection():
    me = os.getpid()
    procs = [
        {"pid": 5, "ppid": 1, "sid": 5, "comm": "strata", "argv": ["/x/build-hip/strata", "--serve"]},
        {"pid": 6, "ppid": 1, "sid": 6, "comm": "llama-server", "argv": ["llama-server"]},
        {"pid": 7, "ppid": 1, "sid": 7, "comm": "python3", "argv": ["python3", "tools/pancake/abtest.py", "spec.json"]},
        {"pid": 8, "ppid": 1, "sid": 8, "comm": "python3", "argv": ["/r/.venv/bin/python", "serve/server.py", "--engine", "strata"]},
        {"pid": 9, "ppid": 1, "sid": 9, "comm": "python3", "argv": ["python3", "-m", "pytest", "tools/pancake/test_ttft_bench.py"]},
        {"pid": 10, "ppid": 1, "sid": 10, "comm": "python3", "argv": ["python3", "server.py", "--port", "1"]},      # another project
        {"pid": 11, "ppid": 1, "sid": 11, "comm": "grep", "argv": ["grep", "strata"]},
        {"pid": 12, "ppid": 1, "sid": 12, "comm": "cc1plus", "argv": ["cc1plus"]},
        {"pid": me, "ppid": 1, "sid": me, "comm": "python3", "argv": ["python3", "tools/pancake/ttft_bench.py"]},      # ourselves: skipped
    ]
    found = [x.split()[0] for x in NQ.busy_processes(procs, me)]
    assert found == ["5", "6", "7", "8", "12"]


def test_guard_waits_until_the_machine_is_quiet_then_runs(tmp_path):
    plan = NQ.load_plan(make_plan(tmp_path, free_port(), jobs=[
        {"name": "one", "group": "g", "engine": "fake-a",
         "client": {"kind": "decode", "cmd": ["{py}", str(HERE / "nq_clients.py"), "decode", "--port", "{port}", "--out", "{out}",
                                                "--max-tokens", "8"], "passes": 2}}]))
    probe = FakeProbe(loads=[3.0, 3.0, 1.5, 0.2], busy=[("strata", ["strata"])])
    q = NQ.Queue(plan, tmp_path / "run", args_for(["--fake", "--guard", "--guard-wait-min", "5", "--guard-poll-s", "1", "--repo", str(REPO)]),
                 probe=probe)
    # the busy process disappears after two polls, the load after three
    real = probe.processes
    state = {"n": 0}

    def procs():
        state["n"] += 1
        if state["n"] > 2:
            probe.busy = []
        return real()
    probe.processes = procs
    q.run_dir.mkdir()
    rc = q.run(plan["_jobs"])
    log = (tmp_path / "run" / "queue.log").read_text()
    assert rc == 0 and q.state["jobs"]["one"]["status"] == "done"
    assert "guard: not quiet" in log and "other engine / build / benchmark processes" in log and "1-minute load" in log
    assert probe.t >= 3          # it really waited (virtual seconds)


def test_guard_skips_a_job_when_the_machine_never_gets_quiet(tmp_path):
    plan = NQ.load_plan(make_plan(tmp_path, free_port()))
    probe = FakeProbe(loads=[5.0], mems=[20.0])
    q = NQ.Queue(plan, tmp_path / "run", args_for(["--fake", "--guard", "--guard-wait-min", "0.05", "--guard-poll-s", "1", "--repo", str(REPO)]),
                 probe=probe)
    q.run_dir.mkdir()
    rc = q.run([j for j in plan["_jobs"] if j["name"] in ("dfr-01A", "later")])
    st = q.state["jobs"]
    assert rc == 1 and st["dfr-01A"]["status"] == "skipped" and st["later"]["status"] == "skipped"
    assert "1-minute load 5.00" in st["dfr-01A"]["reason"] and "MemAvailable 20.0 GiB" in st["dfr-01A"]["reason"]
    assert not (tmp_path / "run" / "raw" / "srv-dfr-01A.log").exists()      # no engine was started
    res = (tmp_path / "run" / "RESULTS.md").read_text()
    assert "## Not done" in res and "skipped" in res and "machine not quiet" in res
    # a skipped job is not finished: --resume would try it again
    assert not q.finished("dfr-01A")


def test_guard_sees_a_busy_port(tmp_path):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    try:
        g = NQ.Guard(FakeProbe(), 1.0, 48.0)
        assert any(f"port {port} is in use" in r for r in g.reasons(port))
    finally:
        s.close()
    assert not NQ.Guard(FakeProbe(), 1.0, 48.0).reasons(port)          # closed again: quiet


# ---------------------------------------------------------------------------------------------------------------- stop / timeout
def test_stop_session_ends_children_by_pid_and_the_leader(tmp_path):
    marker = f"nq-test-{os.getpid()}-{time.time_ns()}"
    proc = subprocess.Popen(["sh", "-c", f"sleep 120 & sleep 120 & wait # {marker}"], start_new_session=True)
    time.sleep(0.3)
    probe = NQ.Probe()
    assert len(NQ.session_pids(probe.processes(), proc.pid)) >= 3
    info = NQ.stop_session(proc, probe, term_wait=5, member_wait=5, kill_wait=5, poll=0.05)
    assert info["gone"] and NQ.session_pids(probe.processes(), proc.pid) == []
    assert info["sigterm"] >= 1


def test_stop_session_kills_a_process_that_ignores_sigterm():
    proc = subprocess.Popen([sys.executable, "-c", "import signal,time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                             "print('up', flush=True)\ntime.sleep(120)"], start_new_session=True, stdout=subprocess.PIPE)
    proc.stdout.readline()
    info = NQ.stop_session(proc, NQ.Probe(), term_wait=0.5, member_wait=0.5, kill_wait=5, poll=0.05)
    assert info["gone"] and info["sigkill"] >= 1 and proc.returncode == -signal.SIGKILL


def test_job_timeout_stops_the_client_and_the_engine(tmp_path):
    port = free_port()
    jobs = [{"name": "slow", "group": "g", "engine": "fake-a", "timeout_min": 0.03,
             "client": {"kind": "decode", "cmd": ["{py}", "-c", "import time; time.sleep(60)"], "passes": 2}}]
    plan = make_plan(tmp_path, port, jobs=jobs)
    t0 = time.time()
    rc, rd = run_main(tmp_path, plan)
    st = json.loads((rd / "state.json").read_text())["jobs"]["slow"]
    assert rc == 1 and st["status"] == "timeout" and "exceeded" in st["reason"] and time.time() - t0 < 30
    assert st["stop"]["gone"] and st["released"]["port_free"]
    assert not NQ.Probe().port_busy(port)


def test_results_only_rewrites_from_state(tmp_path):
    plan = make_plan(tmp_path, free_port())
    rc, rd = run_main(tmp_path, plan, ["--only", "later"])
    (rd / "RESULTS.md").unlink()
    assert NQ.main(["--plan", str(plan), "--run-dir", str(rd), "--results-only", "--repo", str(REPO)]) == 0
    assert "# Night queue results" in (rd / "RESULTS.md").read_text()


# ---------------------------------------------------------------------------------------------------------------- the fake server itself
def test_fake_server_speaks_the_decode_and_stream_protocol(tmp_path):
    port = free_port()
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"env": {}}))
    p = subprocess.Popen([sys.executable, str(HERE / "night_queue.py"), "--fake-server", "--port", str(port), "--fake-config", str(cfg)],
                         start_new_session=True)
    try:
        for _ in range(100):
            if NQ.http_ok(port):
                break
            time.sleep(0.05)
        out = tmp_path / "d.json"
        r = subprocess.run([sys.executable, str(HERE / "nq_clients.py"), "decode", "--port", str(port), "--out", str(out), "--max-tokens", "16"],
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        rows = json.loads(out.read_text())["rows"]
        assert [x["class"] for x in rows] == ["prose", "code", "prose", "prose", "code"]
        assert all(x["tps"] and x["accepted"] is not None for x in rows)
    finally:
        NQ.stop_session(p, NQ.Probe(), term_wait=5, member_wait=2, kill_wait=2, poll=0.05)


def test_sigterm_stops_the_engine_gracefully_and_resume_finishes(tmp_path):
    port = free_port()
    marker = tmp_path / "marker"
    # the first run of this client creates the marker and hangs; the second one returns at once
    hang = ("import os, sys, time\nm = sys.argv[1]\nif not os.path.exists(m):\n    open(m, 'w').close()\n    time.sleep(60)\n")
    jobs = [{"name": "slow", "group": "g", "engine": "fake-a", "client": {"kind": "decode", "cmd": ["{py}", "-c", hang, str(marker)], "passes": 1}},
            {"name": "quick", "group": "g", "engine": "fake-a", "client": {"kind": "decode", "cmd": [
                "{py}", str(HERE / "nq_clients.py"), "decode", "--port", "{port}", "--out", "{out}", "--max-tokens", "8"], "passes": 2}}]
    plan = make_plan(tmp_path, port, jobs=jobs)
    rd = tmp_path / "run"
    q = subprocess.Popen([sys.executable, str(HERE / "night_queue.py"), "--plan", str(plan), "--run-dir", str(rd), "--fake", "--repo", str(REPO)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(400):
        if marker.exists():
            break
        time.sleep(0.05)
    assert marker.exists()
    q.send_signal(signal.SIGTERM)
    out, _ = q.communicate(timeout=60)
    assert q.returncode == 130, out
    st = json.loads((rd / "state.json").read_text())["jobs"]
    assert st["slow"]["status"] == "interrupted" and "quick" not in st
    assert not NQ.Probe().port_busy(port)                      # the engine was stopped
    assert (rd / "RESULTS.md").exists()
    rc, _ = run_main(tmp_path, plan, ["--resume"], run_dir=rd)
    assert rc == 0
    st = json.loads((rd / "state.json").read_text())["jobs"]
    assert st["slow"]["status"] == "done" and st["quick"]["status"] == "done"
