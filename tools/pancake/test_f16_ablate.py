"""Tests of tools/pancake/f16_ablate.py (LEADS.md Part U, lead U5, TEST_LEDGER.md T-1070): the report arithmetic on a synthetic run, the spec, the macro
table against src/prefill/moe_f16_ablate.cuh, and the default-unchanged rule (the ablation blocks stripped from the three kernel sources give the
text of the commit before them).  No GPU, no box."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import f16_ablate as fa  # noqa: E402

ROOT = HERE.parent.parent
BASELINE_REV = "7b1836b60"   # port/rdna3-decode before the U5 commit


def write_run(path: Path, shares: dict, base_tps=1000.0, late=None, case="long", extra_cases=()):
    """One record per arm, one row of `case`; an arm's time is base * (1 - share) (shares are of the base time)."""
    lines = []
    arms = {"base": base_tps, "base-late": late if late else base_tps}
    for arm, s in shares.items():
        arms[arm] = base_tps / (1.0 - s)
    for arm, tps in arms.items():
        rows = [{"prompt": 14000, "prompt_tps": tps, "gen": 8, "decode_tps": 1.0, "case": case}]
        lines.append(json.dumps({"rep": 0, "arm": arm, "clean": True, "rows": rows}))
    path.write_text("\n".join(lines) + "\n")


def test_variants_table():
    full = fa.variants("full")
    names = [n for n, _, _ in full]
    assert names[0] == "base" and full[0][1] == []
    assert names[1:8] == [n for n, _, _ in fa.LADDER]
    assert names[8:] == [f"cum{k}" for k in range(2, 8)]
    assert [n for n, _, _ in fa.variants("singles")] == names[:8]
    # cumulative arms add one macro at a time in ladder order
    for k in range(2, 8):
        macros = dict((n, m) for n, m, _ in full)[f"cum{k}"]
        assert macros == [m for _, m, _ in fa.LADDER[:k]]


def test_macro_names_match_header_and_sources():
    head = (ROOT / "src/prefill/moe_f16_ablate.cuh").read_text()
    for _, macro, _ in fa.LADDER:
        assert macro in head, macro
    # the guard: the macros need STRATA_BENCH_ABLATION
    assert re.search(r"#if !defined\(STRATA_BENCH_ABLATION\)\s*\n#error", head)
    # every macro the sources test is a ladder macro (or the alias)
    known = {m for _, m, _ in fa.LADDER} | {"STRATA_ABL_NO_STAGE_LOAD", "STRATA_ABL_ANY"}
    for f in ("moe_f16_common.cuh", "moe_fused_iq.cu", "moe_f16_kq.cu"):
        for m in re.findall(r"STRATA_ABL_[A-Z_]+", (ROOT / "src/prefill" / f).read_text()):
            assert m in known, (f, m)
    # the CMake hook is the only way to define them, and is empty by default
    cm = (ROOT / "CMakeLists.txt").read_text()
    assert 'set(STRATA_BENCH_ABLATION_DEFS "" CACHE STRING' in cm
    assert "STRATA_BENCH_ABLATION ${STRATA_BENCH_ABLATION_DEFS}" in cm


def test_ladder_arithmetic(tmp_path):
    shares = {"noload": 0.10, "nodeq": 0.06, "nowiden": 0.03, "nostore": 0.04, "noread": 0.05, "nowmma": 0.07, "noepi": 0.02,
              # cumulative: cum1 = noload (0.10), then +0.06, +0.03, +0.04, +0.05, +0.07, +0.02 with a little overlap lost
              "cum2": 0.15, "cum3": 0.18, "cum4": 0.21, "cum5": 0.25, "cum6": 0.31, "cum7": 0.33}
    p = tmp_path / "run.jsonl"
    write_run(p, shares)
    r = fa.ladder(fa.load(p), "long")
    assert r["verdict"] == "OPEN" and abs(r["best"] - 0.10) < 1e-9
    assert not r["drifted"] and r["drift"] == 0.0
    single = {x["arm"]: x["share"] for x in r["singles"]}
    for k, v in single.items():
        assert abs(v - shares[k]) < 1e-9, k
    steps = [x["step"] for x in r["steps"]]
    assert [round(s, 6) for s in steps] == [0.10, 0.05, 0.03, 0.03, 0.04, 0.06, 0.02]
    assert abs(sum(steps) - 0.33) < 1e-9
    assert abs(r["floor"] - 0.67) < 1e-9 and abs(r["kernel_total"] - 0.33) < 1e-9
    g = {x["arm"]: x["gain_if_free"] for x in r["singles"]}
    assert abs(g["noload"] - 0.10 / 0.90) < 1e-9
    assert not r["arms_missing"]
    text = fa.format_report([r])
    assert "verdict OPEN" in text and "noload" in text and "floor" in text


def test_closed_when_no_stage_reaches_the_bar(tmp_path):
    shares = {k: 0.01 for k, _, _ in fa.LADDER}
    shares.update({f"cum{k}": 0.01 * k for k in range(2, 8)})
    p = tmp_path / "run.jsonl"
    write_run(p, shares)
    r = fa.ladder(fa.load(p), "long")
    assert r["verdict"] == "CLOSED" and r["best"] < fa.BAR


def test_drift_and_missing(tmp_path):
    p = tmp_path / "run.jsonl"
    write_run(p, {"noload": 0.2}, base_tps=1000.0, late=900.0)   # base-late 10% slower
    r = fa.ladder(fa.load(p), "long")
    assert r["drifted"] and abs(r["drift"] - 0.1) < 1e-9
    assert "nodeq" in r["arms_missing"] and "cum7" in r["arms_missing"]
    assert "DRIFTED" in fa.format_report([r])
    q = tmp_path / "nobase.jsonl"
    q.write_text(json.dumps({"arm": "noload", "clean": True, "rows": [{"prompt_tps": 5, "case": "long"}]}) + "\n")
    assert "error" in fa.ladder(fa.load(q), "long")


def test_unclean_and_garbage_lines_skipped(tmp_path):
    p = tmp_path / "run.jsonl"
    good = {"arm": "base", "clean": True, "rows": [{"prompt_tps": 100.0, "case": "long"}]}
    bad = {"arm": "base", "clean": False, "rows": [{"prompt_tps": 1.0, "case": "long"}]}
    p.write_text("# comment\n" + json.dumps(good) + "\nnot json {\n" + json.dumps(bad) + "\n")
    assert fa.load(p)[("base", "long")] == [100.0]


def test_report_merges_two_runs(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_run(a, {"noload": 0.10})
    write_run(b, {"noload": 0.20})
    res = fa.report([str(a), str(b)])
    assert [x["run"] for x in res] == [str(a), str(b), "all runs"]
    single = {x["arm"]: x["share"] for x in res[2]["singles"]}
    # mean tps: base 1000, noload mean of 1000/0.9 and 1000/0.8
    t_arm = 1.0 / ((1000 / 0.9 + 1000 / 0.8) / 2)
    assert abs(single["noload"] - (1e-3 - t_arm) / 1e-3) < 1e-9


def test_spec_is_the_committed_one():
    spec = fa.spec("/mnt/pancake/ablate-f16/bin", "full")
    committed = json.loads((ROOT / "bench/specs/276-f16-ablation-ladder.json").read_text())
    assert committed == spec
    assert spec["drift_check"] and spec["prefill"]
    assert len(spec["arms"]) == 14
    assert all(a["env"] == {"STRATA_PF_F16": "1"} for a in spec["arms"].values())
    assert spec["arms"]["base"]["exe"] == "/mnt/pancake/ablate-f16/bin/base/strata"
    assert "long64k" in spec["cases"]


def _strip(text: str) -> str:
    """The sources with every STRATA_ABL_* conditional block replaced by its #else branch (or nothing)."""
    out, stack = [], []   # stack of (is_ablation, in_else)
    for line in text.replace("\r\n", "\n").split("\n"):
        s = line.strip()
        m = re.match(r"#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", s)
        if m:
            kw, rest = m.group(1), m.group(2)
            if kw in ("if", "ifdef", "ifndef"):
                stack.append([bool(re.search(r"STRATA_ABL_", rest)), False])
                if stack[-1][0]:
                    continue
            elif kw == "else" and stack and stack[-1][0]:
                stack[-1][1] = True
                continue
            elif kw == "endif":
                top = stack.pop()
                if top[0]:
                    continue
        # a line is kept when every enclosing ablation block is in its #else branch
        if all((not a) or e for a, e in stack):
            out.append(line)
    res = "\n".join(out)
    return res.replace('\n#include "moe_f16_ablate.cuh"   // bench-only ablation macros (LEADS.md Part U U5); nothing unless -DSTRATA_BENCH_ABLATION and a STRATA_ABL_* macro\n', "")


@pytest.mark.parametrize("name", ["moe_f16_common.cuh", "moe_fused_iq.cu", "moe_f16_kq.cu"])
def test_default_source_unchanged(name):
    rel = f"src/prefill/{name}"
    try:
        base = subprocess.run(["git", "show", f"{BASELINE_REV}:{rel}"], cwd=ROOT, capture_output=True, check=True).stdout.decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip(f"{BASELINE_REV} not reachable here")
    now = (ROOT / rel).read_text()
    assert _strip(now) == base.replace("\r\n", "\n")
    assert "STRATA_ABL_" in now   # the macros are really there


def test_strip_helper_keeps_else_branch():
    t = "a\n#if defined(STRATA_ABL_X)\nb\n#else\nc\n#endif\nd\n#if FOO\ne\n#endif\n"
    assert _strip(t) == "a\nc\nd\n#if FOO\ne\n#endif\n"


# ---- the box driver (never run here): syntax and the dry-run plan
import os  # noqa: E402
import shutil  # noqa: E402

DRIVER = ROOT / "tools/pancake/box_chains/bench-ablate-f16.sh"


def _bash():
    """A working POSIX bash (not the WSL stub of System32): Git's, else the PATH one that answers."""
    for cand in ("C:/Program Files/Git/bin/bash.exe", "C:/Program Files/Git/usr/bin/bash.exe", shutil.which("bash") or ""):
        if cand and Path(cand).exists():
            try:
                r = subprocess.run([cand, "-c", "echo ok"], capture_output=True, text=True, timeout=30)
            except (OSError, subprocess.SubprocessError):
                continue
            if r.stdout.strip() == "ok":
                return cand
    return None


@pytest.mark.skipif(_bash() is None, reason="no working bash")
def test_driver_syntax_and_dry_run():
    bash = _bash()
    assert subprocess.run([bash, "-n", DRIVER.as_posix()]).returncode == 0
    env = dict(os.environ, ABL_DRY="1", ABL_PY3=sys.executable, ABL_RUNS="2", ABL_TAG="t")
    out = subprocess.run([bash, DRIVER.as_posix()], env=env, capture_output=True, text=True, check=True).stdout.splitlines()
    builds = [x for x in out if x.startswith("BUILD ")]
    assert len(builds) == 14 and builds[0].startswith("BUILD base DEFS=[]")
    assert sum(1 for x in out if x.startswith("RUN ")) == 2
    assert any("DEFS=[STRATA_ABL_NO_GLOBAL_LOAD;STRATA_ABL_NO_DEQUANT]" in x for x in out)
    env["ABL_SET"] = "singles"
    out = subprocess.run([bash, DRIVER.as_posix()], env=env, capture_output=True, text=True, check=True).stdout.splitlines()
    assert len([x for x in out if x.startswith("BUILD ")]) == 8


def test_driver_is_idle_only_and_deletes_nothing():
    text = DRIVER.read_text()
    assert not re.search(r"(^|[;&|(\s])(rm|rmdir|unlink)\s", re.sub(r"#.*", "", text)), "the driver must not delete"
    assert "quick_idle" in text and "run-queue-us" in text and "bench-sweep" in text   # busy() sees the queue
    assert "never overwritten" in text and "ABL_STOP_SERVICE" in text
