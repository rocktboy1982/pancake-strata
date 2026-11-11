"""abtest.py and queue_run.sh end to end against the fake engine (tools/pancake/probes/fake_engine.py): real processes, real HTTP, real
log files, no GPU, seconds per test.  What it checks: the old mode's contract (order of runs, record schema, text parity, early stop,
screen tier, server death and retry, the gates keep a slow start) and, when abtest has the flags, block mode, --warmup, the robust
outlier rule and the tier record; queue_run.sh: dry run, two specs, resume, arguments, busy refusal, night window, drop-cache.

Each test builds on a fake repo root (a copy of abtest.py / queue_run.sh with ROOT = a temp dir, see make_fake_root) and compresses time
(FAKE_TIME_SCALE, default 0.05).  The tests under test come from this checkout; point them at another build with
    ABTEST_UNDER_TEST=/path/abtest.py QUEUE_RUN_UNDER_TEST=/path/queue_run.sh python -m pytest tools/pancake/test_abtest_e2e.py

SAFETY: the abtest under test stops only the servers it started (the old stop_servers() ended every "server.py --engine strata" of the machine)
and refuses to start (exit 3) while another test or engine runs; the tests of that change (side by side, refusal, leftover, SIGTERM) skip
for an abtest without --force-busy.  Even so the tests refuse to run (skip) while a real abtest / engine / queue runs
(fake_engine.real_run_active()): a fake run must never share the machine with a real one.

Run: python -m pytest tools/pancake/test_abtest_e2e.py"""
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "probes"))
import fake_engine as F  # noqa: E402
import promote  # noqa: E402

ABTEST = os.environ.get("ABTEST_UNDER_TEST") or str(HERE / "abtest.py")
QUEUE_RUN = os.environ.get("QUEUE_RUN_UNDER_TEST") or str(HERE / "queue_run.sh")
SCALE = os.environ.get("FAKE_TIME_SCALE", "0.05")
QUIET = ["--foreign-cpu", "1000000", "--reclaim-max", "1000000", "--max-wait", "1"]   # the host's load is not what these tests test
LOW_NOISE = {"sd_start_pct": 0.1, "sd_request_pct": 0.1}
ROW_KEYS = {"prompt", "prompt_tps", "gen", "decode_tps", "accept", "case"}
RECORD_KEYS = {"t", "rep", "arm", "attempt", "clean", "rows", "texts", "noise", "noise_top", "reclaim", "gpu", "big_free_gib"}


# ------------------------------------------------------------------------------------------------------------ helpers
def refuse_when_real_run():
    busy = F.real_run_active()
    if busy:
        pytest.skip(f"a real abtest / engine / queue is running ({busy}): the fake run could stop its server")


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    refuse_when_real_run()
    return F.make_fake_root(tmp_path_factory.mktemp("fakeroot"), abtest=ABTEST, queue_run=QUEUE_RUN, scale=float(SCALE))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def config(root, name, **fake):
    """A base config for the fake server (its own engine log), with the profile `fake`."""
    path = root / "configs" / f"{name}.json"
    path.write_text(json.dumps({"exe": str(root / "bin/strata"), "args": ["--spec", "4"], "env": {"STRATA_RAM_KEEP_GIB": "12"},
                                "log": str(root / "logs" / f"{name}.log"), "fake": {"startup_s": 2.0, **fake}}))
    return path


def spec(root, name, base, arms, cases=("short", "edit"), reps=2, **extra):
    path = root / "configs" / f"{name}-spec.json"
    path.write_text(json.dumps({"base": str(base), "arms": arms, "cases": list(cases), "reps": reps, **extra}))
    return path


def abtest_cmd(root, spec_path, out, flags, port):
    return [str(root / ".venv/bin/python"), "tools/pancake/abtest.py", str(spec_path), "--out", str(out),
            "--port", str(port or free_port()), *QUIET, *flags]


def run_abtest(root, spec_path, out, *flags, port=None, timeout=240, guard=True):
    """guard=False: no check for a real run (a test that has started a fake process of its own, which the check would take for one)."""
    if guard:
        refuse_when_real_run()
    return subprocess.run(abtest_cmd(root, spec_path, out, flags, port), cwd=root, capture_output=True, text=True, timeout=timeout,
                          env={**os.environ, "FAKE_TIME_SCALE": SCALE})


def start_abtest(root, spec_path, out, *flags, port=None):
    """abtest in the background (stdout and stderr in one pipe): for a test that needs something to happen while it runs."""
    refuse_when_real_run()
    return subprocess.Popen(abtest_cmd(root, spec_path, out, flags, port), cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            env={**os.environ, "FAKE_TIME_SCALE": SCALE})


def servers_of(fragment):
    """pids of the python `serve/server.py --engine strata` processes whose command line has `fragment` (a unique config path)."""
    found = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{d}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            comm = Path(f"/proc/{d}/comm").read_text().strip()
        except OSError:
            continue
        if comm.startswith("python") and "serve/server.py --engine strata" in cmd and fragment in cmd:
            found.append(int(d))
    return found


def wait_for(cond, seconds=30):
    t0 = time.time()
    while time.time() - t0 < seconds:
        if cond():
            return True
        time.sleep(0.05)
    return False


def records(out):
    return [json.loads(line) for line in Path(out).read_text().splitlines() if line.strip()]


def supports(root, flag):
    r = subprocess.run([str(root / ".venv/bin/python"), "tools/pancake/abtest.py", "--help"], cwd=root, capture_output=True, text=True, timeout=60)
    return flag in r.stdout


def need(root, flag):
    if not supports(root, flag):
        pytest.skip(f"this abtest.py has no {flag} (harness-core's block mode / warm-up / outlier rule has not landed in this checkout)")


def summary_n(stdout, case, arm):
    """(mean, n) of the summary line `  <case> <arm>  <mean> +- <half>  (n=<n>)`."""
    m = re.search(rf"^\s+{case}\s+{arm}\s+([\d.]+) \+-\s+[\d.nan]+\s+\(n=(\d+)\)", stdout, re.M)
    assert m, f"no summary line for {case} {arm} in:\n{stdout[-1500:]}"
    return float(m[1]), int(m[2])


def values(recs, arm, case, key="decode_tps"):
    return [r[key] for rec in recs if rec["arm"] == arm for r in rec["rows"] if r["case"] == case]


def engine_lines(root, name):
    return [ln for ln in (root / "logs" / f"{name}.log").read_text().splitlines() if ln.startswith("strata serve: prompt")]


# ------------------------------------------------------------------------------------------------------------ old mode
def test_old_mode_order_schema_text_and_gain(root):
    base = config(root, "old", **LOW_NOISE)
    arms = {"base": {}, "fast": {"env": {"FAKE_GAIN_PCT": "10"}}, "salted": {"env": {"FAKE_TEXT_SALT": "x"}}}
    out = root / "old.jsonl"
    r = run_abtest(root, spec(root, "old", base, arms, reps=2), out, "--no-early-stop")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    recs = records(out)
    assert [(x["rep"], x["arm"]) for x in recs] == [(0, "base"), (0, "fast"), (0, "salted"), (1, "salted"), (1, "fast"), (1, "base")]   # A B C, C B A
    for rec in recs:
        assert RECORD_KEYS <= set(rec) and rec["attempt"] == 0 and rec["clean"] is True
        assert [x["case"] for x in rec["rows"]] == ["short", "edit"] and all(ROW_KEYS <= set(x) for x in rec["rows"])
        assert not any("pass" in x or "warm" in x for x in rec["rows"])           # the old rows carry no block fields
        assert all(x["gen"] == 512 and x["decode_tps"] > 100 for x in rec["rows"])   # the cold warm-up request is not in the rows
    base_short, fast_short = values(recs, "base", "short"), values(recs, "fast", "short")
    assert 1.07 < (sum(fast_short) / 2) / (sum(base_short) / 2) < 1.13
    texts = {x["arm"]: x["texts"] for x in recs}
    assert texts["fast"] == texts["base"] and texts["salted"] != texts["base"] and set(texts["base"]) == {"short", "edit"}
    assert re.search(r"fast\s+identical on every case -> exact", r.stdout) and re.search(r"salted\s+DIFFERENT text", r.stdout)
    assert summary_n(r.stdout, "short", "base")[1] == 2
    assert len(engine_lines(root, "old")) == 6 * 3          # per start: the warm-up request (not counted) + short + edit
    runs, discarded, problems = promote.read_runs(out)       # the file promote.py reads
    assert sorted(runs) == ["base", "fast", "salted"] and discarded == 0 and not problems
    gain = promote.gain_ci(promote.case_pairs(runs["base"], runs["fast"], "short"))
    assert gain["n"] == 2 and 7 < gain["gain"] < 13


def test_screen_tier_asks_every_case_twice_and_the_second_edit_is_faster(root):
    base = config(root, "screen", **LOW_NOISE)
    out = root / "screen.jsonl"
    r = run_abtest(root, spec(root, "screen", base, {"base": {}, "fast": {"env": {"FAKE_GAIN_PCT": "10"}}}, reps=5), out, "--tier", "screen")
    assert r.returncode == 0, r.stdout[-2000:]
    recs = records(out)
    assert [(x["rep"], x["arm"]) for x in recs] == [(0, "base"), (0, "fast")]            # one A B pass
    rows = recs[0]["rows"]
    assert [x["case"] for x in rows] == ["short", "short", "edit", "edit"]
    assert 1.25 < rows[3]["decode_tps"] / rows[2]["decode_tps"] < 1.45                    # the repeated edit prompt: the fake's 35% (the engine's lookup draft)
    assert re.search(r"fast\s+short \+\d+\.\d%\s+edit \+\d+\.\d%\s+-> PROMOTE to confirm", r.stdout)


def test_early_stop_after_two_decided_reps(root):
    base = config(root, "early", **LOW_NOISE)
    out = root / "early.jsonl"
    r = run_abtest(root, spec(root, "early", base, {"base": {}, "fast": {"env": {"FAKE_GAIN_PCT": "10"}}}, reps=6), out)
    assert r.returncode == 0 and "early stop after 2 reps" in r.stdout, r.stdout[-2000:]
    assert len(records(out)) == 4


def test_a_server_that_dies_once_is_retried_and_twice_aborts(root):
    one = config(root, "die1", die_starts=[0], **LOW_NOISE)
    out = root / "die1.jsonl"
    r = run_abtest(root, spec(root, "die1", one, {"base": {}, "fast": {}}, reps=1), out)
    assert r.returncode == 0 and r.stdout.count("FAILED (server died before ready)") == 1, r.stdout[-2000:]
    recs = records(out)
    assert [(x["arm"], x["attempt"]) for x in recs] == [("base", 1), ("fast", 0)]          # the first start died, its retry is attempt 1
    two = config(root, "die2", die=True, **LOW_NOISE)
    out2 = root / "die2.jsonl"
    r = run_abtest(root, spec(root, "die2", two, {"base": {}, "fast": {}}, reps=2), out2)
    assert r.returncode == 2 and "ABORT: the server died twice before ready" in r.stdout and not out2.read_text().strip()


def test_the_gate_keeps_a_slow_start_that_no_proxy_flagged(root):
    """Start 4 of A B B A (base, fast, fast, base, base, ...) runs at 60%: nothing in the old rules sees it (no foreign CPU), so it stays."""
    base = config(root, "slow", slow_starts=[4], slow_factor=0.6, **LOW_NOISE)
    out = root / "slow.jsonl"
    r = run_abtest(root, spec(root, "slow", base, {"base": {}, "fast": {"env": {"FAKE_GAIN_PCT": "10"}}}, reps=4), out, "--no-early-stop")
    assert r.returncode == 0, r.stdout[-2000:]
    recs = records(out)
    assert all(x["clean"] for x in recs) and len(recs) == 8
    slow = [x for x in recs if x["rep"] == 2 and x["arm"] == "base"][0]
    assert slow["rows"][0]["decode_tps"] < 0.7 * 121.5 and values(recs, "base", "short")[2] == slow["rows"][0]["decode_tps"]
    assert summary_n(r.stdout, "short", "base")[1] == 4                                      # ... and counts in the mean


# ------------------------------------------------------------------------------------------------------------ new modes
def test_block_mode_passes_per_start_in_abba_order_and_cold_passes(root):
    need(root, "--block")
    base = config(root, "block", **LOW_NOISE)
    out = root / "block.jsonl"
    r = run_abtest(root, spec(root, "block", base, {"base": {}, "fast": {"env": {"FAKE_GAIN_PCT": "10"}}}, reps=6), out, "--block", "3", "--no-early-stop")
    assert r.returncode == 0, r.stdout[-2000:]
    recs = records(out)
    assert [(x["rep"], x["arm"]) for x in recs] == [(0, "base"), (0, "fast"), (1, "fast"), (1, "base")]   # max(2, ceil(6 / 3)) starts per arm
    for rec in recs:
        assert [(x["pass"], x["case"]) for x in rec["rows"]] == [(k, c) for k in range(3) for c in ("short", "edit")]
        assert [x["warm"] for x in rec["rows"]] == [False, False, True, True, True, True]
    assert len(engine_lines(root, "block")) == 4 * (1 + 6)                                   # per start: the short warm-up + 3 passes x 2 cases
    edit = [x["decode_tps"] for x in recs[0]["rows"] if x["case"] == "edit"]
    assert edit[1] / edit[0] < 1.15 and edit[2] / edit[0] < 1.15                             # "[pass k]" keeps every pass cold: no 35% repeat gain
    assert summary_n(r.stdout, "edit", "base")[1] == 6


def test_promote_reads_a_block_start_as_one_pair_so_three_starts_per_arm_are_needed(root):
    """promote.py pairs by (arm, rep) and averages the rows of a case inside a record: one pair per start.  Its MIN_PAIRS is 3, abtest's
    default is max(2, ceil(reps / N)) starts per arm, so --block 3 --reps 6 can never PROMOTE (2 pairs): ask for reps >= 3 x N."""
    need(root, "--block")
    arms = {"base": {}, "fast": {"env": {"FAKE_GAIN_PCT": "10"}}}
    found = {}
    for reps in (6, 9):
        base = config(root, f"pairs{reps}", **LOW_NOISE)
        out = root / f"pairs{reps}.jsonl"
        assert run_abtest(root, spec(root, f"pairs{reps}", base, arms, reps=reps), out, "--block", "3", "--no-early-stop").returncode == 0
        runs, _, _ = promote.read_runs(out)
        found[reps] = promote.case_pairs(runs["base"], runs["fast"], "short")
        rows = [r["decode_tps"] for r in records(out)[0]["rows"] if r["case"] == "short"]
        assert promote.case_value(next(iter(runs[records(out)[0]["arm"]].values())), "short") == pytest.approx(sum(rows) / len(rows))
    assert len(found[6]) == 2 < promote.MIN_PAIRS <= len(found[9]) == 3


def test_warmup_passes_are_thrown_away(root):
    need(root, "--warmup")
    base = config(root, "warm", cold_pct=[40.0, 10.0], **LOW_NOISE)
    out = root / "warm.jsonl"
    r = run_abtest(root, spec(root, "warm", base, {"base": {}, "fast": {}}, reps=2), out, "--block", "2", "--warmup", "1", "--no-early-stop")
    assert r.returncode == 0, r.stdout[-2000:]
    rec = records(out)[0]
    assert [(x["pass"], x["case"]) for x in rec["warmup_rows"]] == [(0, "short"), (0, "edit")]
    assert [x["pass"] for x in rec["rows"]] == [1, 1, 2, 2] and all(x["warm"] for x in rec["rows"])
    assert rec["rows"][0]["decode_tps"] > 0.97 * 121.5 and rec["warmup_rows"][0]["decode_tps"] < 0.7 * 121.5   # cold first request, not measured
    assert len(engine_lines(root, "warm")) == 4 * (1 + 2) * 2                                # no single short warm-up request in its place


def test_robust_rule_runs_a_slow_start_again_where_the_gate_keeps_it(root):
    need(root, "--outlier")
    arms = {"base": {}, "fast": {"env": {"FAKE_GAIN_PCT": "10"}}}
    runs = {}
    for name, flags in (("gate", ()), ("robust", ("--outlier", "robust"))):
        base = config(root, f"robust-{name}", slow_starts=[4], slow_factor=0.6, **LOW_NOISE)   # its own engine log: start 4 is this run's
        runs[name] = run_abtest(root, spec(root, f"robust-{name}", base, arms, reps=4), root / f"robust-{name}.jsonl", "--no-early-stop", *flags)
    gate, robust = runs["gate"], runs["robust"]
    assert gate.returncode == 0 and robust.returncode == 0, robust.stdout[-2000:]
    (m_gate, n_gate), (m_robust, n_robust) = summary_n(gate.stdout, "short", "base"), summary_n(robust.stdout, "short", "base")
    assert n_gate == 4 and n_robust == 4 and m_robust > m_gate + 10                           # the gate averages the 60% start in; robust replaces it
    recs = records(root / "robust-robust.jsonl")
    assert len(recs) == 9 and [(x["rep"], x["arm"], x["attempt"]) for x in recs if x["rep"] == 2 and x["arm"] == "base"] == [(2, "base", 0), (2, "base", 1)]
    assert all(x["clean"] for x in recs) and "1 slow starts left out" in robust.stdout       # no proxy flagged it: the outcome did


def test_robust_rule_tells_the_slow_start_apart_with_two_starts_per_arm(root):
    need(root, "--outlier")
    base = config(root, "robust-two", slow_starts=[3], slow_factor=0.6, **LOW_NOISE)         # base's second start (A B | B A)
    r = run_abtest(root, spec(root, "robust-two", base, {"base": {}, "fast": {"env": {"FAKE_GAIN_PCT": "10"}}}, reps=4), root / "robust-two.jsonl",
                   "--block", "2", "--no-early-stop", "--outlier", "robust")
    assert r.returncode == 0, r.stdout[-2000:]
    recs = records(root / "robust-two.jsonl")
    assert [(x["rep"], x["arm"], x["attempt"]) for x in recs] == [(0, "base", 0), (0, "fast", 0), (1, "fast", 0), (1, "base", 0), (1, "base", 1)]
    assert "1 slow starts left out" in r.stdout and summary_n(r.stdout, "short", "base") == (pytest.approx(121.5 * 1.02, rel=0.03), 4)   # 2 starts x 2 passes


def test_tier_is_recorded_from_the_engine_line_and_capped_sh(root):
    need(root, "--block")        # the tier record landed with the block / warm-up / outlier change
    out = root / "tier.jsonl"
    r = run_abtest(root, spec(root, "tier", config(root, "tier-a", tier="arena", **LOW_NOISE), {"base": {}}, reps=1), out)
    assert r.returncode == 0, r.stdout[-2000:]
    assert (records(out)[0]["tier"], records(out)[0]["tier_source"]) == ("arena", "engine")
    out2 = root / "tier2.jsonl"
    run_abtest(root, spec(root, "tier2", config(root, "tier-b", **LOW_NOISE), {"base": {}}, reps=1), out2)
    assert (records(out2)[0]["tier"], records(out2)[0]["tier_source"]) == ("mmap", "capped")   # no engine line: capped.sh's guess


def test_every_start_keeps_its_engine_log_with_the_switch_lines(root):
    need(root, "--block")
    base = config(root, "sw", **LOW_NOISE)
    arms = {"base": {}, "on": {"env": {"STRATA_FAKE_ON": "1", "STRATA_FAKE_QUIET": "1", "FAKE_SWITCH_SILENT": "STRATA_FAKE_QUIET"}}}
    out = root / "sw.jsonl"
    r = run_abtest(root, spec(root, "sw", base, arms, reps=1), out)
    assert r.returncode == 0, r.stdout[-2000:]
    work = out.with_suffix("")
    log_on, log_base = (work / "on-r0-a0.engine.log").read_text(), (work / "base-r0-a0.engine.log").read_text()
    assert "strata switch: STRATA_FAKE_ON acted:" in log_on and "STRATA_FAKE_QUIET" not in log_on     # a silent no-op says nothing
    assert "STRATA_FAKE" not in log_base and "engine started" in log_base                            # only this start's lines


# ------------------------------------------------------------------------------------------------------------ stop only what abtest started
def slow(seconds):
    """A fake profile whose every request takes `seconds` of real time (the fake's sleeps are scaled by FAKE_TIME_SCALE)."""
    return {"latency_s": seconds / float(SCALE), **LOW_NOISE}


def test_two_abtests_side_by_side_stop_only_their_own_server(root):
    """The old stop_servers() ended every `server.py --engine strata` of the machine: B's first stop killed A's server in mid-run."""
    need(root, "--force-busy")
    out_a, out_b = root / "side-a.jsonl", root / "side-b.jsonl"
    frag_a, frag_b = str(out_a.with_suffix("")), str(out_b.with_suffix(""))
    a = start_abtest(root, spec(root, "side-a", config(root, "side-a", **slow(2.0)), {"base": {}}, reps=1), out_a, "--force-busy")
    try:
        assert wait_for(lambda: servers_of(frag_a)), "A's server did not come up"
        arms_b = {"base": {}, "fast": {"env": {"FAKE_GAIN_PCT": "10"}}}
        b = run_abtest(root, spec(root, "side-b", config(root, "side-b", **LOW_NOISE), arms_b, reps=1), out_b, "--force-busy", guard=False)   # A is ours: no "real run"
        overlapped = bool(servers_of(frag_a))                   # A was still being served when B had started and stopped two servers
        out_a_text = a.communicate(timeout=120)[0]
    finally:
        if a.poll() is None:   # SIGTERM, not kill: abtest then ends its own server (a killed one would leave it for the next test)
            a.terminate()
            a.communicate(timeout=60)
    assert a.returncode == 0 and "FAILED" not in out_a_text, "A lost its server to B?\n" + out_a_text[-1500:]
    assert len(records(out_a)) == 1 and [x["case"] for x in records(out_a)[0]["rows"]] == ["short", "edit"]
    assert len(engine_lines(root, "side-a")) == 3               # warm-up + short + edit, none lost to B
    assert b.returncode == 0 and "--force-busy: running beside" in b.stdout and "FAILED" not in b.stdout, b.stdout[-1500:]
    assert [x["arm"] for x in records(out_b)] == ["base", "fast"]
    assert overlapped, "the runs did not overlap: nothing was tested (the machine is too slow for the 6 s window?)"
    assert servers_of(frag_a) == [] and servers_of(frag_b) == []   # each ended its own server


@pytest.mark.parametrize("n, argv0", [("abtest", "python3 tools/pancake/abtest.py"), ("restore", "python3 tools/pancake/restore_check.py"),
                                      ("engine", "/opt/x/build-hip/strata")])
def test_abtest_refuses_to_start_while_another_test_or_engine_runs(root, n, argv0):
    need(root, "--force-busy")
    sp, out = spec(root, f"busy-{n}", config(root, f"busy-{n}", **LOW_NOISE), {"base": {}}, reps=1), root / f"busy-{n}.jsonl"
    decoy = subprocess.Popen(["bash", "-c", f"exec -a '{argv0}' sleep 60"])   # argv[0] only: its command line is that of the real thing
    try:
        time.sleep(0.3)
        r = run_abtest(root, sp, out)
        assert r.returncode == 3 and "another test or engine is already running" in r.stderr, r.stdout + r.stderr
        assert f"pid {decoy.pid}: {argv0} 60" in r.stderr and "--force-busy" in r.stderr
        assert not out.exists() and not out.with_suffix("").exists() and not (root / "logs" / f"busy-{n}.log").exists()   # nothing was started
        forced = run_abtest(root, sp, out, "--force-busy")
        assert forced.returncode == 0 and "--force-busy: running beside 1 other" in forced.stdout, forced.stdout[-1500:]
        assert len(records(out)) == 1 and decoy.poll() is None   # it ran, and the other process was left alone
    finally:
        decoy.kill()
        decoy.wait()


def test_a_shell_that_only_mentions_abtest_does_not_stop_a_start(root):
    need(root, "--force-busy")
    decoy = subprocess.Popen(["bash", "-c", "sleep 60; true  # python3 tools/pancake/abtest.py"])
    try:
        time.sleep(0.3)
        out = root / "mention.jsonl"
        r = run_abtest(root, spec(root, "mention", config(root, "mention", **LOW_NOISE), {"base": {}}, reps=1), out)
        assert r.returncode == 0 and "force-busy" not in r.stdout and len(records(out)) == 1, r.stdout + r.stderr
    finally:
        decoy.kill()
        decoy.wait()


def test_a_leftover_server_is_reported_and_never_killed(root):
    """A `server.py --engine strata` that abtest did not start (a crashed run's, or another session's): reported with its pid, left running."""
    need(root, "--force-busy")
    refuse_when_real_run()
    stale_cfg = config(root, "stale", **LOW_NOISE)
    stale = subprocess.Popen([str(root / ".venv/bin/python"), "serve/server.py", "--engine", "strata", "--config", str(stale_cfg),
                              "--port", str(free_port())], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             env={**os.environ, "FAKE_TIME_SCALE": SCALE})
    try:
        assert wait_for(lambda: servers_of(str(stale_cfg))), "the leftover fake server did not come up"
        sp, out = spec(root, "leftover", config(root, "leftover", **LOW_NOISE), {"base": {}}, reps=1), root / "leftover.jsonl"
        r = run_abtest(root, sp, out, guard=False)
        assert r.returncode == 3 and f"pid {stale.pid}: " in r.stderr and "serve/server.py --engine strata" in r.stderr, r.stdout + r.stderr
        assert stale.poll() is None and not out.exists()
        forced = run_abtest(root, sp, out, "--force-busy", guard=False)   # its own port: the leftover is no obstacle to the fake
        assert forced.returncode == 0 and len(records(out)) == 1, forced.stdout[-1500:]
        assert stale.poll() is None and servers_of(str(stale_cfg)) == [stale.pid]   # abtest ended its own server and left this one
    finally:
        stale.kill()
        stale.wait()


def test_sigterm_ends_abtest_and_the_server_it_started(root):
    """A killed abtest must not leave its server behind: the next start would refuse (a leftover), and before it was killed by name."""
    need(root, "--force-busy")
    out = root / "term.jsonl"
    a = start_abtest(root, spec(root, "term", config(root, "term", **slow(2.0)), {"base": {}}, reps=1), out)
    frag = str(out.with_suffix(""))
    try:
        assert wait_for(lambda: servers_of(frag)), "the server did not come up"
        a.send_signal(signal.SIGTERM)
        a.communicate(timeout=60)
        assert a.returncode == 128 + signal.SIGTERM
        assert servers_of(frag) == []
    finally:
        if a.poll() is None:   # SIGTERM, not kill: abtest then ends its own server (a killed one would leave it for the next test)
            a.terminate()
            a.communicate(timeout=60)


# ------------------------------------------------------------------------------------------------------------ queue_run.sh
def queue(root, out, *items, flags=(), env=None, timeout=300):
    refuse_when_real_run()
    e = {**os.environ, "FAKE_TIME_SCALE": SCALE, "MIN_GIB": "1", **(env or {})}
    return subprocess.run(["bash", str(root / "tools/pancake/queue_run.sh"), *flags, str(out), *items], cwd=root, capture_output=True, text=True,
                          env=e, timeout=timeout)


def queue_items(root, tag):
    """Two fake specs for the queue; the second asks for the screen tier.  Each item carries its own port and the quiet flags."""
    base = config(root, f"q-{tag}", **LOW_NOISE)
    arms = {"base": {}, "fast": {"env": {"FAKE_GAIN_PCT": "10"}}}
    quiet = " ".join(f"{k}={v}" for k, v in zip(QUIET[::2], QUIET[1::2]))
    one, two = spec(root, f"q1-{tag}", base, arms, reps=2), spec(root, f"q2-{tag}", base, arms, reps=1)
    return [f"{one}:--no-early-stop --port={free_port()} {quiet}", f"{two}:--tier=screen --port={free_port()} {quiet}"], one, two


def queue_env(root):
    """The busy check looks at every process of the machine: a decoy-proof pattern where the queue allows one, else skip on a match."""
    if "BUSY_PATTERN" in (root / "tools/pancake/queue_run.sh").read_text():
        return {"BUSY_PATTERN": "no-such-process-" + str(os.getpid())}
    if subprocess.run(["pgrep", "-f", "tools/pancake/abtest.py|restore_check.py|bin/strata "], capture_output=True).returncode == 0:
        pytest.skip("something on this machine matches queue_run.sh's busy pattern (a watcher shell or a real run); it would stop the queue")
    return {}


def test_queue_dry_run_prints_the_commands_and_runs_nothing(root, tmp_path):
    items, one, two = queue_items(root, "dry")
    r = queue(root, tmp_path / "out", *items, "no-such-spec.json", flags=["-n"], env=queue_env(root))
    assert r.returncode == 0, r.stdout + r.stderr
    lines = [ln for ln in r.stdout.splitlines() if "would run:" in ln]
    assert len(lines) == 2 and str(one) in lines[0] and "--no-early-stop" in lines[0] and "--tier screen" in lines[1]
    assert "no-such-spec: no such spec, skipped" in r.stdout and "queue finished" in r.stdout
    assert not list((tmp_path / "out").glob("*.jsonl")) and not list((tmp_path / "out").glob("*.summary.txt"))


def test_queue_runs_two_specs_then_skips_them_on_a_second_call(root, tmp_path):
    items, one, two = queue_items(root, "real")
    out, env = tmp_path / "out", queue_env(root)
    r = queue(root, out, *items, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    n1, n2 = one.stem, two.stem
    assert f"{n1}: exit 0, 4 run lines" in r.stdout and f"{n2}: exit 0, 2 run lines" in r.stdout and "queue finished" in r.stdout
    assert len(records(out / f"{n1}.jsonl")) == 4 and len(records(out / f"{n2}.jsonl")) == 2
    assert (out / f"{n1}.summary.txt").read_text().startswith("summary")
    assert "screen verdict" in (out / f"{n2}.log").read_text() and "screen verdict" not in (out / f"{n1}.log").read_text()   # ":--tier=screen" got through
    if (out / "queue.summary.txt").exists():
        assert re.search(rf"^{n1} exit=0 runs=4 minutes=", (out / "queue.summary.txt").read_text(), re.M)
    before = {p.name: p.read_text() for p in out.glob("*.jsonl")}
    again = queue(root, out, *items, env=env)
    assert again.returncode == 0 and again.stdout.count("done before, skipped") == 2 and ": start" not in again.stdout
    assert {p.name: p.read_text() for p in out.glob("*.jsonl")} == before                    # resumable: nothing was run again


def test_queue_stops_while_another_test_runs(root, tmp_path):
    if "BUSY_PATTERN" not in (root / "tools/pancake/queue_run.sh").read_text():
        pytest.skip("this queue_run.sh has a fixed busy pattern (BUSY_PATTERN comes with tier-tools' change)")
    items, _, _ = queue_items(root, "busy")
    decoy = subprocess.Popen(["bash", "-c", "exec -a fake-busy-decoy-4711 sleep 60"])
    try:
        time.sleep(0.3)
        r = queue(root, tmp_path / "out", *items, env={"BUSY_PATTERN": "fake-busy-decoy-4711"})
    finally:
        decoy.kill()
    assert r.returncode == 3 and "another test or engine is running: queue stopped" in r.stdout
    assert not list((tmp_path / "out").glob("*.jsonl"))


@pytest.mark.xfail(strict=False, reason="queue_run.sh's busy check is pgrep -f 'tools/pancake/abtest.py|restore_check.py|bin/strata ': it matches any command "
                   "line that only contains that text (a `bash -c 'until ! pgrep -f tools/pancake/abtest.py ...'` watcher never ends: it matches itself), "
                   "so such a shell stops the queue with exit 3 although nothing runs; anchoring the pattern at argv[0] python fixes it")
def test_queue_busy_check_ignores_a_shell_that_only_mentions_abtest(root, tmp_path):
    items, _, _ = queue_items(root, "mention")
    decoy = subprocess.Popen(["bash", "-c", "sleep 60; true  # tools/pancake/abtest.py"])
    try:
        time.sleep(0.3)
        r = queue(root, tmp_path / "out", *items)            # the script's own busy pattern (no override)
    finally:
        decoy.kill()
    assert r.returncode == 0, r.stdout


def test_queue_night_window_and_drop_cache(root, tmp_path):
    text = (root / "tools/pancake/queue_run.sh").read_text()
    if "--night-only" not in text or "--drop-cache" not in text:
        pytest.skip("this queue_run.sh has no --night-only / --drop-cache (tier-tools' change)")
    items, one, _ = queue_items(root, "night")
    hour = time.localtime().tm_hour
    elsewhere = f"{(hour + 2) % 24:02d}-{(hour + 3) % 24:02d}"                              # a window that does not contain now
    r = queue(root, tmp_path / "a", *items, flags=["--night-only", elsewhere], env=queue_env(root))
    assert r.returncode == 5 and "outside the night window" in r.stdout and not list((tmp_path / "a").glob("*.jsonl"))
    marker = tmp_path / "dropped"
    script = tmp_path / "drop.py"
    script.write_text(f"import pathlib\np = pathlib.Path({str(marker)!r})\np.write_text(p.read_text() + 'x' if p.exists() else 'x')\n")
    env = {**queue_env(root), "DROP_CACHE": str(script)}
    r = queue(root, tmp_path / "b", *items, flags=["--night-only", f"{hour:02d}-{hour:02d}", "--drop-cache"], env=env)   # equal hours: all day
    assert r.returncode == 0, r.stdout + r.stderr
    assert marker.read_text() == "x"                                                          # once, before the first spec, not before each
    assert len(records(tmp_path / "b" / f"{one.stem}.jsonl")) == 4
    dry = queue(root, tmp_path / "c", *items, flags=["-n", "--drop-cache"], env=env)
    assert "would run:" in dry.stdout and "drop.py" in dry.stdout
