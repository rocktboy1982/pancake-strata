"""Host-side tests of LEADS.md Part S S4 / Part T T8 (STRATA_HIP_VOPD, STRATA_HIP_EXTRA_FLAGS): the CMake option parser (script mode),
the compile-command rewriting and repeat verdict of hip_repeat_check.py (fake compiler), and the shell syntax and safety lines of the
box scripts.  Nothing here compiles HIP or touches a GPU."""
import json
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CH = ROOT / "tools/pancake/box_chains"
sys.path.insert(0, str(CH))
import hip_repeat_check as H  # noqa: E402


@pytest.mark.skipif(shutil.which("cmake") is None, reason="no cmake")
def test_cmake_option_parser():
    p = subprocess.run(["cmake", "-P", str(ROOT / "cmake/strata_hip_flags_test.cmake")], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "strata_hip_flags: ok" in p.stdout + p.stderr


def test_cmake_defaults_are_off():
    s = (ROOT / "CMakeLists.txt").read_text()
    assert 'set(STRATA_HIP_VOPD "default" CACHE' in s
    assert 'set(STRATA_HIP_EXTRA_FLAGS "" CACHE' in s
    assert 'set(STRATA_HIP_EXTRA_FLAGS_SCOPE "mmq" CACHE' in s


def test_device_command_rewrite():
    e = {"directory": "/b", "file": "/s/rdna_kq.cu",
         "command": "/opt/clang++ -x hip --offload-arch=gfx1100 -O3 -MD -MT a.o -MF a.o.d -o a.o -c /s/rdna_kq.cu"}
    cmd = H.device_command(e, "/o/x.o", ["-mllvm", "-amdgpu-enable-vopd=false"], clang="/c")
    assert cmd[0] == "/c"
    assert "-MD" not in cmd and "-MF" not in cmd and "a.o" not in cmd
    assert cmd[-4:] == ["--offload-device-only", "-c", "-o", "/o/x.o"]
    assert "-amdgpu-enable-vopd=false" in cmd and "/s/rdna_kq.cu" in cmd


def _fake_clang(tmp, deterministic):
    """A python script standing in for clang: writes 'dual=N salt' to the -o file (N = 0 with the VOPD-off flag)."""
    f = tmp / "fakeclang.py"
    f.write_text("import sys, time\n"
                 "o = sys.argv[sys.argv.index('-o') + 1]\n"
                 "dual = 0 if '-amdgpu-enable-vopd=false' in sys.argv else 3\n"
                 "salt = '' if %r else str(time.time_ns())\n"
                 "open(o, 'w').write('dual=%%d %%s' %% (dual, salt))\n" % deterministic)
    return f


def _cc(tmp, fc):
    cc = tmp / "cc.json"
    cc.write_text(json.dumps([
        {"directory": str(tmp), "file": "k.cu",
         "arguments": [sys.executable, str(fc), "-x", "hip", "--offload-arch=gfx1100", "-o", "k.o", "-c", "k.cu"]},
        {"directory": str(tmp), "file": "host.cpp", "arguments": ["g++", "-o", "h.o", "-c", "host.cpp"]}]))
    return cc


@pytest.mark.parametrize("deterministic,code", [(True, 0), (False, 1)])
def test_repeat_verdict(tmp_path, deterministic, code):
    cc = _cc(tmp_path, _fake_clang(tmp_path, deterministic))
    assert H.main(["--compile-commands", str(cc), "--out", str(tmp_path / "out")]) == code
    rep = json.loads((tmp_path / "out/repeat-report.json").read_text())
    assert len(rep["units"]) == 1 and rep["units"][0]["repeatable"] is (code == 0)   # the host unit is not picked


def test_vopd_census_with_variant(tmp_path, monkeypatch):
    cc = _cc(tmp_path, _fake_clang(tmp_path, True))
    real = subprocess.run

    def fake_run(cmd, **kw):
        if cmd[0] == "OBJDUMP":      # the disassembly of the fake object: 'dual=N' means N v_dual lines
            n = int(Path(cmd[2]).read_text().split()[0].split("=")[1])
            body = "  s_nop 0" + chr(10) + ("  v_dual_fmac_f32 v0" + chr(10)) * n
            return subprocess.CompletedProcess(cmd, 0, stdout=body.encode())
        return real(cmd, **kw)
    monkeypatch.setattr(H.subprocess, "run", fake_run)
    rc = H.main(["--compile-commands", str(cc), "--out", str(tmp_path / "o"), "--objdump", "OBJDUMP",
                 "--variant-flags", "-mllvm -amdgpu-enable-vopd=false"])
    assert rc == 0
    u = json.loads((tmp_path / "o/repeat-report.json").read_text())["units"][0]
    assert (u["vopd_default"], u["vopd_variant"]) == (3, 0)


def test_no_unit_is_an_error(tmp_path):
    cc = tmp_path / "cc.json"
    cc.write_text("[]")
    assert H.main(["--compile-commands", str(cc), "--out", str(tmp_path / "o")]) == 2


@pytest.mark.skipif(shutil.which("sh") is None, reason="no sh")
@pytest.mark.parametrize("name", ["hipflags-build.v1.sh", "hipflags-repeat.v1.sh", "hipflags-probe.v1.sh"])
def test_shell_syntax(name):
    assert subprocess.run(["sh", "-n", str(CH / name)]).returncode == 0


def test_build_script_rules():
    s = (CH / "hipflags-build.v1.sh").read_text()
    assert "rm -" not in s                                       # never deletes
    assert 'fail "build failed' in s and "ERROR" in s            # a failed build stops with ERROR
    assert "/mnt/pancake" in s and "strata.prev" in s            # outside /, previous binary kept
    assert "HF_VOPD=${HF_VOPD:-default}" in s and "HF_EXTRA=${HF_EXTRA:-}" in s   # defaults = unchanged build
    for pat in ("abtest", "bench-", "strata --serve", "serve/server.py", "quality_gate", "task_gate", "ninja", "clang", "ctest", "replay_turns"):
        for n in ("hipflags-build.v1.sh", "hipflags-repeat.v1.sh", "hipflags-probe.v1.sh", "hip_repeat_check.py"):
            assert pat not in n                                  # no file name carries a busy pattern of the running queue


def test_specs_reference_hipflags_binaries():
    for n in ("260-hip-vopd-build-ab.json", "261-hip-extra-flags-build-ab.json"):
        sp = json.loads((ROOT / "bench/specs" / n).read_text())
        assert all(a["exe"].startswith("/mnt/pancake/hipflags/build-") for a in sp["arms"].values())
        assert "default" in sp["arms"] and (ROOT / sp["base"]).exists()
