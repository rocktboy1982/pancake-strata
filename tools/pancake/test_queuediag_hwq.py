"""Tests of LEADS.md Part S S1 + S2: specs 253 / 254 (GPU_MAX_HW_QUEUES 1 and 2), the bench-hwq.v1.sh chain file, and the ABTEST_QUEUEDIAG=1 KFD queue-layout
lines of abtest.py (fake /sys files) with their reader in box_chains/startdiag_report.py.  Host side only: no GPU, no ROCm, no engine.
Run: python -m pytest -q tools/pancake/test_queuediag_hwq.py"""
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CHAINS = HERE / "box_chains"
sys.path.insert(0, str(HERE))
import abtest  # noqa: E402
import spec_lint  # noqa: E402

SPECS = [ROOT / "bench/specs/253-hwq-decode.json", ROOT / "bench/specs/254-hwq-long64k.json"]


# ------------------------------------------------------------------------------------------------ S1: specs and chain
@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.name)
def test_hwq_specs_differ_only_in_the_hw_queue_env(path):
    s = json.loads(path.read_text(encoding="utf-8"))
    assert list(s["arms"]) == ["base", "hwq1", "hwq2"]
    assert s["arms"]["base"] == {}                                        # the preset's own GPU_MAX_HW_QUEUES (6)
    assert s["arms"]["hwq1"] == {"env": {"GPU_MAX_HW_QUEUES": "1"}}
    assert s["arms"]["hwq2"] == {"env": {"GPU_MAX_HW_QUEUES": "2"}}      # no args, no exe, no other env key
    assert s["reps"] == 5 and s["lossy"] is False
    preset = json.loads((ROOT / s["base"]).read_text(encoding="utf-8"))
    assert preset["env"]["GPU_MAX_HW_QUEUES"] == "6"
    assert spec_lint.predict_problems(s) == []


def test_hwq_spec_cases():
    a, b = (json.loads(p.read_text(encoding="utf-8")) for p in SPECS)
    assert a["cases"] == ["short", "edit"] and not a.get("prefill")
    assert b["cases"] == ["long64k"] and b["prefill"] is True
    assert "long64k" in abtest.LONG_TARGET


def test_hwq_specs_are_lint_clean(tmp_path):
    for p in SPECS:
        shutil.copy(p, tmp_path / p.name)
    r = subprocess.run([sys.executable, str(HERE / "spec_lint.py"), "--dir", str(tmp_path), "--require-predict-from", "253"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "253-hwq" not in r.stdout and "254-hwq" not in r.stdout


def test_chain_file_marker_busy_and_syntax():
    p = CHAINS / "bench-hwq.v1.sh"
    t = p.read_text(encoding="utf-8")
    assert re.search(r"WIN[A-Z]*_DONE", t).group(0) == "WINBENCHHWQ_DONE"                     # what bench-sched.sh reads as the marker
    assert t.rstrip().endswith("finish") and "echo WINBENCHHWQ_DONE >> $L" in t and "bench-hwq.v1.log" in t
    assert "253-hwq-decode" in t and "254-hwq-long64k" in t and "ABTEST_QUEUEDIAG" in t
    assert "sudo" not in t.replace("no sudo", "") and "rm -rf" not in t and "ssh" not in t.split("# NOT RUN")[1].split("\nS=")[1]   # nothing deleted, no ssh inside
    sched = (CHAINS / "bench-sched.sh").read_text(encoding="utf-8")
    pat = re.search(r'pgrep -f "([^"]*\[b\]ench-\(updated[^"]*)"', sched).group(1)               # the scheduler's own busy patterns must not see this script
    assert not re.search(pat, "sh ./bench-hwq.v1.sh")
    mine = re.search(r'busy\(\) \{ pgrep -f "([^"]*)"', t).group(1)
    assert not re.search(mine, "sh ./bench-hwq.v1.sh")                                          # its own busy() does not wait for itself
    sh = shutil.which("sh")
    if sh:
        assert subprocess.run([sh, "-n", str(p)], capture_output=True).returncode == 0


# ------------------------------------------------------------------------------------------------ S2: fake sysfs
def fake_kfd(tmp_path, evicted=350, hqds=None, pid=4242):
    k = tmp_path / "kfd"
    n0, n1 = k / "topology/nodes/0", k / "topology/nodes/1"
    n0.mkdir(parents=True)
    n1.mkdir(parents=True)
    (n0 / "gpu_id").write_text("0\n")
    (n0 / "properties").write_text("cpu_cores_count 16\n")
    (n1 / "gpu_id").write_text("48821\n")
    (n1 / "properties").write_text("simd_count 168\nnum_cp_queues 24\ngfx_target_version 110000\n")
    pr = k / f"proc/{pid}"
    for qid, typ in ((1, 0), (2, 0), (3, 1)):
        q = pr / f"queues/{qid}"
        q.mkdir(parents=True)
        (q / "type").write_text(f"{typ}\n")
        (q / "gpuid").write_text("48821\n")
        (q / "size").write_text("262144\n")
    (pr / "stats_48821").mkdir()
    (pr / "stats_48821/evicted_ms").write_text(f"{evicted}\n")
    h = tmp_path / "hqds"
    if hqds is not None:
        h.write_text(hqds)
    return str(k), str(h)


HQDS_SHARED = "Node 1, gpu_id bd35:\n  Inst 0, CP Pipe 0, Queue 2\n    CP_MQD_BASE_ADDR: 0x1\n  Inst 0, CP Pipe 0, Queue 3\n  Inst 0, CP Pipe 1, Queue 0\n  SDMA Engine 0, RLC 0\n"
HQDS_SPLIT = "Node 1:\n  Inst 0, CP Pipe 0, Queue 2\n  Inst 0, CP Pipe 1, Queue 0\n"
TAB = {4242: (1, "python", 0), 4300: (4242, "strata", 0), 9: (1, "other", 0)}


def test_parse_hqds_counts_queues_per_pipe():
    assert abtest.parse_hqds(HQDS_SHARED) == {(0, 0): 2, (0, 1): 1}
    assert abtest.parse_hqds("CP Pipe 3, Queue 1\n") == {(0, 3): 1}
    assert abtest.parse_hqds("") == {} and abtest.parse_hqds(None) == {}


def test_queue_snapshot_and_line_shared_pipe(tmp_path):
    root, hq = fake_kfd(tmp_path, hqds=HQDS_SHARED)
    s = abtest.queue_snapshot([4242], root, hq, tab=TAB)
    assert (s["gfx"], s["cp_queues"], s["pids"], s["types"], s["evicted_ms"], s["shared"]) == ("gfx110000", "24", [4242], {"compute": 2, "sdma": 1}, 350, 1)
    line = abtest.queuediag_line("STARTQUEUE", s, 0)
    assert line.startswith("STARTQUEUE ") and " gfx=gfx110000 cp_queues=24 queues=compute:2,sdma:1 evicted_ms=350 hqds=ok pipes=i0p0:2,i0p1:1 shared=1" in line
    assert "pids=4242" in line


def test_queue_snapshot_not_shared_and_no_root(tmp_path):
    root, hq = fake_kfd(tmp_path, evicted=0, hqds=HQDS_SPLIT)
    assert abtest.queue_snapshot([4242], root, hq, tab=TAB)["shared"] == 0
    root, hq = fake_kfd(tmp_path / "b", hqds=None)                                               # hqds file absent / unreadable: the non-root case
    s = abtest.queue_snapshot([4242], root, hq, tab=TAB)
    line = abtest.queuediag_line("STOPQUEUE", s, 0)
    assert s["pipes"] is None and s["shared"] is None and s["evicted_ms"] == 350
    assert "hqds=NA pipes=NA shared=NA" in line and "evicted_ms=350" in line


def test_queue_snapshot_without_kfd_is_all_na(tmp_path):
    s = abtest.queue_snapshot([1], str(tmp_path / "nokfd"), str(tmp_path / "nohqds"), tab={1: (0, "x", 0)})
    line = abtest.queuediag_line("STARTQUEUE", s, 0)
    assert "pids=NA gfx=NA cp_queues=NA queues=NA evicted_ms=NA hqds=NA pipes=NA shared=NA" in line


def test_queuediag_default_off_and_never_raises(tmp_path):
    out = []
    abtest.queuediag("STARTQUEUE", [1], out.append, env={})
    abtest.queuediag("STARTQUEUE", [1], out.append, env={"ABTEST_QUEUEDIAG": "0"})
    assert out == []
    abtest.queuediag("STARTQUEUE", [1], out.append, env={"ABTEST_QUEUEDIAG": "1"}, snapshot=lambda p: 1 / 0)
    assert len(out) == 1 and "failed (ZeroDivisionError): carrying on" in out[0]
    abtest.queuediag("STOPQUEUE", [1], lambda s: (_ for _ in ()).throw(OSError("say broke")), env={"ABTEST_QUEUEDIAG": "1"},
                     snapshot=lambda p: 1 / 0)                                                     # even a broken `say` stops nothing
    abtest.queuediag("STARTQUEUE", [1], out.append, env={"ABTEST_QUEUEDIAG": "1"}, snapshot=lambda p: abtest.queue_snapshot(p, str(tmp_path / "n"), str(tmp_path / "m"), tab={}))
    assert len(out) == 2 and out[1].startswith("STARTQUEUE ")


def test_hooks_are_in_start_and_stop_and_the_reader_is_read_only():
    text = (HERE / "abtest.py").read_text(encoding="utf-8")
    body = text[text.index("def start_server"):text.index("def token_check")]
    assert body.index('queuediag("STARTQUEUE"') > body.index("diag_load(") and body.index("subprocess.Popen") < body.index('queuediag("STARTQUEUE"')
    stop = text[text.index("def stop_servers"):text.index("SAMPLING = ")]
    assert stop.index('queuediag("STOPQUEUE"') < stop.index("p.terminate()")                        # read before the server is ended
    block = text[text.index("# ---- S2 (LEADS.md Part S)"):text.index("def start_server")]
    assert "subprocess" not in block.replace("# ", "") .split("def queuediag_on")[1] and "sudo -" not in block.split("def queuediag_on")[1]
    assert 'get("ABTEST_QUEUEDIAG", "0") == "1"' in block                                          # default off


# ------------------------------------------------------------------------------------------------ S2: the report
def _report_mod():
    spec = importlib.util.spec_from_file_location("startdiag_report_q", CHAINS / "startdiag_report.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_report_attaches_queue_lines_and_splits_slow_share(tmp_path):
    rp = _report_mod()

    def start(rate, shared, ev_stop, ev_start=0, comp=2):
        sh = "NA" if shared is None else shared
        return ("STARTDIAG 2026-11-21T01:00:00 prev_run_s=100 gap_s=100 rd_sect=1 wr_sect=2 io_ticks_ms=3 d_rd_sect=1 d_wr_sect=0 d_io_ticks_ms=5 cached_kb=10 mlocked_kb=20 "
                "memavail_kb=30 nvme_temp_c=40 compact_stall_d=0\n"
                f"STARTLOAD 2026-11-21T01:01:00 loaded 33.0 GiB at {rate} GiB/s rate={rate}\n"
                f"STARTQUEUE 2026-11-21T01:01:01 pids=1 gfx=gfx110000 cp_queues=24 queues=compute:{comp},sdma:1 evicted_ms={ev_start} hqds={'NA' if shared is None else 'ok'} "
                f"pipes=i0p0:2 shared={sh}\n"
                f"STOPQUEUE 2026-11-21T01:09:00 pids=1 gfx=gfx110000 cp_queues=24 queues=compute:{comp} evicted_ms={ev_stop} hqds=NA pipes=NA shared=NA\n")
    text = start(0.4, 1, 900) + start(0.5, 1, 0) + start(1.2, 0, 0) + start(1.3, 0, 0) + start(1.1, None, 0) + "STARTQUEUE broken line\n"
    (tmp_path / "x.out").write_text(text)
    starts = rp.parse_text(text)
    assert len(starts) == 5 and starts[0]["shared"] == 1 and starts[0]["evicted_stop_ms"] == 900 and starts[0]["q_compute"] == 2 and starts[4]["shared"] is None
    out = []
    rp.report(starts, 0.8, 20, out.append)
    t = "\n".join(out)
    assert "S2 queue layout" in t
    assert re.search(r"pipe shared +\s*2 of +2 slow", t) and re.search(r"pipe not shared +\s*0 of +2 slow", t) and re.search(r"evicted by KFD +\s*1 of +1 slow", t)
    assert "evicted_stop_ms" in t and "shared" in t
    assert subprocess.run([sys.executable, str(CHAINS / "startdiag_report.py"), str(tmp_path / "x.out")], capture_output=True, text=True).returncode == 0


def test_report_without_queue_lines_is_unchanged(tmp_path):
    rp = _report_mod()
    text = ("STARTDIAG 2026-11-21T01:00:00 prev_run_s=100 gap_s=100 rd_sect=1 wr_sect=2 io_ticks_ms=3 d_rd_sect=1 d_wr_sect=0 d_io_ticks_ms=5 cached_kb=10 mlocked_kb=20 "
            "memavail_kb=30 nvme_temp_c=40 compact_stall_d=0\nSTARTLOAD 2026-11-21T01:01:00 loaded 33.0 GiB at 1.2 GiB/s rate=1.2\n")
    out = []
    rp.report(rp.parse_text(text), 0.8, 20, out.append)
    assert "S2 queue layout" not in "\n".join(out)
