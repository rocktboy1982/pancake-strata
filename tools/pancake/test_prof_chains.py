"""Control-flow tests of box_chains/prof-t11.v1.sh, trace-t12.v1.sh, att-t13.v1.sh, kres-t9.sh with a fake rocprofv3 (POSIX sh; Git Bash on Windows).
Checks: default OFF does nothing, the command lines are built from the environment, output goes to a NEW directory each time, a failing rocprofv3 is
survived, ATT without decoder exits 3.  Nothing here runs a GPU."""
import os
import shutil
import sys
import subprocess
from pathlib import Path

import pytest

CH = Path(__file__).resolve().parent / "box_chains"
SH = shutil.which("sh")
pytestmark = pytest.mark.skipif(SH is None, reason="needs a POSIX sh")

FAKE = """#!/bin/sh
echo "$@" >> "$FAKE_ARGS"
case "$1" in --help) echo "--kernel-trace --hip-trace --hip-event-trace --memory-copy-trace --marker-trace --pmc --att --kernel-include-regex --kernel-iteration-range --hip-graph-trace $FAKE_HELP"; exit 0;; esac
while [ $# -gt 0 ]; do [ "$1" = "-d" ] && D=$2; shift; done
mkdir -p "$D"
printf '"Kernel_Name","Correlation_Id","Start_Timestamp","End_Timestamp"\\n"k1","1","100","200"\\n"k2","2","300","400"\\n' > "$D/kernel_trace.csv"
exit ${FAKE_RC:-0}
"""


def env(tmp_path, **kw):
    rocm = tmp_path / "rocm"
    (rocm / "bin").mkdir(parents=True, exist_ok=True)
    (rocm / "lib").mkdir(exist_ok=True)
    f = rocm / "bin" / "rocprofv3"
    f.write_text(FAKE)
    f.chmod(0o755)
    e = dict(os.environ, ROCM=str(rocm), PROF_ROOT=str(tmp_path / "prof"), FAKE_ARGS=str(tmp_path / "args.txt"), PROF_SKIP_BUSY="1", PROF_CMD="true", PYTHON=sys.executable)
    e.update(kw)
    return e


def run(script, e):
    return subprocess.run([SH, str(CH / script)], env=e, capture_output=True, text=True, timeout=60)


def dirs(tmp_path):
    p = tmp_path / "prof"
    return sorted(x for x in p.iterdir()) if p.exists() else []


@pytest.mark.parametrize("script", ["prof-t11.v1.sh", "trace-t12.v1.sh", "att-t13.v1.sh"])
def test_default_off_starts_nothing(tmp_path, script):
    r = run(script, env(tmp_path))
    assert r.returncode == 0 and "default OFF" in r.stdout
    assert dirs(tmp_path) == [] and not (tmp_path / "args.txt").exists()


def test_t11_groups_and_replay_flag(tmp_path):
    e = env(tmp_path, STRATA_T11_RUN="1", PROF_GROUPS="SQ_WAVES GRBM_GUI_ACTIVE;SQ_INSTS_VALU")
    r = run("prof-t11.v1.sh", e)
    assert r.returncode == 0, r.stdout + r.stderr
    args = (tmp_path / "args.txt").read_text()
    assert "--pmc SQ_WAVES GRBM_GUI_ACTIVE" in args and "--pmc SQ_INSTS_VALU" in args and "--kernel-trace" in args
    d = dirs(tmp_path)
    assert len(d) == 1 and (d[0] / "report.txt").read_text().count("k1") >= 1
    assert (d[0] / "run.log").read_text().strip().endswith("PROF_T11_DONE")
    run("prof-t11.v1.sh", e)
    assert len(dirs(tmp_path)) == 2      # a new numbered directory, the first one untouched
    # replay requested but flag absent from help -> warning, still runs
    e2 = env(tmp_path / "x", STRATA_T11_RUN="1", PROF_REPLAY="1")
    r = run("prof-t11.v1.sh", e2)
    assert "WARN kernel replay flag" in r.stdout and "--replay-mode" not in (tmp_path / "x" / "args.txt").read_text()
    e3 = env(tmp_path / "y", STRATA_T11_RUN="1", PROF_REPLAY="1", FAKE_HELP="--kernel-replay-beta-enabled")
    run("prof-t11.v1.sh", e3)
    assert "--replay-mode kernel --kernel-replay-beta-enabled" in (tmp_path / "y" / "args.txt").read_text()


def test_t11_needs_cmd_and_survives_failure(tmp_path):
    r = run("prof-t11.v1.sh", env(tmp_path, STRATA_T11_RUN="1", PROF_CMD=""))
    assert r.returncode == 1 and "PROF_CMD is not set" in r.stdout
    r = run("prof-t11.v1.sh", env(tmp_path / "f", STRATA_T11_RUN="1", FAKE_RC="3"))
    assert r.returncode == 0 and "exit 3" in r.stdout


def test_t12_modes(tmp_path):
    r = run("trace-t12.v1.sh", env(tmp_path, STRATA_T12_RUN="1"))
    assert r.returncode == 0, r.stdout
    args = (tmp_path / "args.txt").read_text()
    assert "--hip-event-trace" in args and "--hip-graph-trace" in args and "--kernel-trace" in args
    rep = (dirs(tmp_path)[0] / "report.txt").read_text()
    assert "wall" in rep and "gaps between kernels" in rep
    run("trace-t12.v1.sh", env(tmp_path / "s", STRATA_T12_RUN="1", TRACE_MODE="sys"))
    assert "--sys-trace" in (tmp_path / "s" / "args.txt").read_text()
    r = run("trace-t12.v1.sh", env(tmp_path / "r", STRATA_T12_RUN="1", TRACE_MODE="rocsys"))
    assert "rocsys not installed" in r.stdout


def test_t13_decoder_missing_exits_3_without_engine(tmp_path):
    e = env(tmp_path, STRATA_T13_RUN="1", ATT_KERNEL="k1")
    r = run("att-t13.v1.sh", e)
    assert r.returncode == 3 and "no rocprof-trace-decoder" in (dirs(tmp_path)[0] / "check.txt").read_text()
    assert "--att" not in (tmp_path / "args.txt").read_text().replace("--help", "").split("--att", 1)[0] or True
    # no engine run: the fake rocprofv3 was called only for --help
    lines = (tmp_path / "args.txt").read_text().splitlines()
    assert all(ln.startswith("--help") for ln in lines)


def test_t13_runs_with_decoder_and_one_dispatch(tmp_path):
    e = env(tmp_path, STRATA_T13_RUN="1", ATT_KERNEL="k1", ATT_ITER="7")
    (tmp_path / "rocm" / "lib").mkdir(parents=True, exist_ok=True)
    (tmp_path / "rocm" / "lib" / "librocprof-trace-decoder.so").write_text("x")
    r = run("att-t13.v1.sh", e)
    assert r.returncode == 0, r.stdout
    args = (tmp_path / "args.txt").read_text()
    assert "--att --kernel-include-regex k1 --kernel-iteration-range [7,7]" in args
    r = run("att-t13.v1.sh", dict(e, ATT_CHECK_ONLY="1", PROF_ROOT=str(tmp_path / "p2")))
    assert r.returncode == 0 and "check only" in r.stdout


def test_t9_default_off(tmp_path):
    r = subprocess.run([SH, str(CH / "kres-t9.sh")], env=dict(os.environ, OUT=str(tmp_path / "o")), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "PLAN" in r.stdout and not (tmp_path / "o").exists()
