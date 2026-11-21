"""Tests of tools/pancake/box_chains/rocm102-install.v1.sh on a PC: every box command (pgrep, apt-get, apt-cache, dpkg, ldconfig, df, uname, hipcc,
rocminfo) is a stub in a temp dir, SUDO is overridden through ROCM102_SUDO.  Proved: syntax, the default prints the plan and writes nothing, the busy
refusal, the owner path (sudo not passwordless: exit 3, nothing installed), no deletion / removal command in the code, no kernel / DKMS / amdgpu package in
the install list, every privileged call goes through $SUDO, the dry-run guard, the disk guard, the unchanged-state guard, the happy path.
Needs a POSIX sh (Git Bash on Windows).  python -m pytest -q tools/pancake/test_rocm102_install.py"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "box_chains" / "rocm102-install.v1.sh"
SH = shutil.which("sh")
pytestmark = pytest.mark.skipif(not SH, reason="needs a POSIX sh")

DRY_OK = ("amdrocm-sysdeps10.2 amdrocm-base10.2 amdrocm-llvm10.2 amdrocm-runtime10.2 amdrocm-core10.2-gfx1100 amdrocm10.2-gfx1100 "
          "amdrocm-llvm-dev10.2")


def code_lines():
    return [ln for ln in SCRIPT.read_text().splitlines() if not ln.lstrip().startswith("#")]


def p(path):
    return Path(path).as_posix()


class Box:
    def __init__(self, tmp):
        self.tmp = Path(tmp)
        self.bin = self.tmp / "bin"
        self.S = self.tmp / "S"
        self.root = self.tmp / "opt-rocm"
        self.bin.mkdir()
        self.S.mkdir()
        (self.root / "core-10.0" / "bin").mkdir(parents=True)
        h = self.root / "core-10.0" / "bin" / "hipcc"
        h.write_text("#!/bin/sh\necho 'HIP version: 7.15.26333-0000000'; echo 'AMD clang version 23'\n")
        h.chmod(0o755)
        self.calls = self.tmp / "calls.log"
        self.stub("pgrep", 'echo "pgrep $*" >> "$CALLS"; [ -f "$BOX_TMP/busy" ] && exit 0; exit 1')
        self.stub("apt-cache", r'''echo "apt-cache $*" >> "$CALLS"
case "$1" in
policy) echo "amdrocm10.2-gfx1100:"; echo "  Installed: (none)"; echo "  Candidate: 10.2.0-2";;
show) echo "Package: x"; echo "Installed-Size: 1000"; echo "Size: 1048576";;
esac''')
        self.stub("apt-get", r'''echo "apt-get $*" >> "$CALLS"
case " $* " in
*" -s "*) for x in $DRYLIST; do echo "Inst $x (10.2.0-2 AMD ROCm:stable [amd64])"; done; [ -n "$DRYREMV" ] && echo "Remv $DRYREMV [1]"; exit 0;;
*" install "*) mkdir -p "$ROCM102_ROOT/core-10.2/bin"
  printf '#!/bin/sh\necho "HIP version: 7.17.0"\n' > "$ROCM102_ROOT/core-10.2/bin/hipcc"
  printf '#!/bin/sh\necho "  Name: gfx1100"\n' > "$ROCM102_ROOT/core-10.2/bin/rocminfo"
  chmod +x "$ROCM102_ROOT/core-10.2/bin/hipcc" "$ROCM102_ROOT/core-10.2/bin/rocminfo"; exit 0;;
esac''')
        self.stub("dpkg", 'printf "h1\\nh2\\nh3\\nh4\\nh5\\nii amdrocm-base10.0 10.0.0-4\\nii amdgpu-core 7.2\\n"')
        self.stub("ldconfig", 'echo "librocm 3 libs"; echo "librocm2"')
        self.stub("uname", 'echo 7.0.0-43-generic')
        self.stub("df", f'printf "Filesystem 1K-blocks Used Available\\nx 1 1 %s\\n" "$FREE_KIB"')
        self.stub("wget", "exit 1")

    def stub(self, name, body):
        f = self.bin / name
        f.write_text("#!/bin/sh\n" + body + "\n")
        f.chmod(0o755)

    def run(self, env=None, **kw):
        e = dict(os.environ)
        e.update({"PATH": p(self.bin) + os.pathsep + e["PATH"], "CALLS": p(self.calls), "BOX_TMP": p(self.tmp), "ROCM102_S": p(self.S),
                  "ROCM102_ROOT": p(self.root), "ROCM102_APT_SRC": p(self.tmp), "ROCM102_SUDO": "env", "DRYLIST": DRY_OK,
                  "FREE_KIB": str(100 * 1048576), "ROCM102_INSTALL": "1"})
        e.update(env or {})
        return subprocess.run([SH, str(SCRIPT)], capture_output=True, text=True, env=e, timeout=60, **kw)

    def log(self):
        f = self.S / "rocm102-install.log"
        return f.read_text() if f.exists() else ""

    def called(self):
        return self.calls.read_text() if self.calls.exists() else ""


@pytest.fixture
def box(tmp_path):
    return Box(tmp_path)


def test_syntax():
    assert subprocess.run([SH, "-n", str(SCRIPT)]).returncode == 0


def test_default_prints_plan_and_does_nothing(box):
    r = box.run(env={"ROCM102_INSTALL": "0"})
    assert r.returncode == 0
    assert "PLAN" in r.stdout and "amdrocm10.2-gfx1100" in r.stdout and "core-10.2" in r.stdout
    assert box.log() == "" and box.called() == ""
    r2 = box.run(env={"ROCM102_INSTALL": ""})
    assert r2.returncode == 0 and "PLAN" in r2.stdout


def test_busy_refusal(box):
    (box.tmp / "busy").write_text("1")
    r = box.run()
    assert r.returncode == 1
    log = box.log()
    assert "ERROR a measured job runs" in log and "ROCM102_INSTALL_DONE" in log and "START busy_check" in log
    assert "apt-get" not in box.called()


def test_busy_pattern_is_the_chains_pattern_and_misses_itself():
    src = SCRIPT.read_text()
    pat = re.search(r"BUSY_RE='([^']*)'", src).group(1)
    for s in ["abtest.py", "bench-sweep.sh", "strata --serve", "serve/server.py", "quality_gate.py", "task_gate.py", "ninja", "clang", "ctest",
              "replay_turns.py", "win-chainT.sh"]:
        assert re.search(pat, s), s
    assert not re.search(pat, "sh ./rocm102-install.v1.sh")


def test_sudo_not_passwordless_prints_owner_commands(box):
    r = box.run(env={"ROCM102_SUDO": "false"})
    assert r.returncode == 3
    assert "apt-get install" in r.stdout and "amdrocm10.2-gfx1100" in r.stdout and "!" in r.stdout
    assert "ROCM102_INSTALL_NEEDS_OWNER" in box.log()
    assert "install" not in box.called()


def test_no_deletion_in_code():
    pos = r"(^\s*|[;&|(]\s*)"
    bad = re.compile(pos + r"(rm|rmdir|unlink|shred|mv|truncate|modprobe|insmod|reboot|systemctl|ln)\s|apt(-get)?\s+(remove|purge|autoremove|autoclean|clean)|--purge|dpkg\s+-[rP]\b")
    for ln in code_lines():
        assert not bad.search(ln), ln


def test_package_list_has_no_kernel_or_driver():
    pk = re.search(r'^PKGS="([^"]*)"', SCRIPT.read_text(), re.M).group(1).split()
    assert pk and all(re.match(r"amdrocm.*10\.2", x) for x in pk), pk
    assert not [x for x in pk if re.search(r"dkms|linux-|amdgpu|firmware|kernel|10\.0", x)]


def test_every_privileged_call_goes_through_sudo_variable():
    lines = code_lines()
    priv = [ln for ln in lines if re.search(r"apt-get (install|update)|\btee\b|mkdir --parents", ln) and "-s " not in ln]
    assert priv
    for ln in priv:
        assert "$SUDO" in ln or ln.strip().startswith("echo") or "cat <<" in ln or "tee -a" in ln or "wget" in ln, ln
    assert "SUDO=${ROCM102_SUDO:-sudo -n}" in SCRIPT.read_text()
    # no bare sudo command in the code (outside the SUDO assignment and the printed owner text)
    for ln in lines:
        if re.match(r"\s*sudo\s", ln) or re.search(r"[;|&]\s*sudo\s", ln):
            raise AssertionError(ln)


def test_happy_path(box):
    r = box.run()
    log = box.log()
    assert r.returncode == 0, (r.stdout, r.stderr, log)
    for s in ["busy_check", "sudo_check", "pre_state", "repo", "apt_update", "dry_run", "disk_check", "install", "verify", "print_env"]:
        assert f"START {s} " in log and f"END {s} " in log, s
    assert log.rstrip().endswith(re.search(r"ROCM102_INSTALL_DONE \S+", log).group(0))
    assert "ERROR" not in log
    assert "export PATH=" in r.stdout and "core-10.2/bin" in r.stdout
    assert re.search(r"apt-get install -y --no-install-recommends amdrocm10.2-gfx1100", box.called())
    assert (box.S / "rocm102-pre-state.txt").exists()


def test_dry_run_with_kernel_package_stops(box):
    r = box.run(env={"DRYLIST": DRY_OK + " amdgpu-dkms"})
    assert r.returncode == 1
    assert "ERROR dry run touches" in box.log() and "ROCM102_INSTALL_DONE" in box.log()
    assert not re.search(r"apt-get .*install -y", box.called())


def test_dry_run_with_removal_stops(box):
    r = box.run(env={"DRYREMV": "amdrocm-base10.0"})
    assert r.returncode == 1 and "would REMOVE" in box.log()
    assert not re.search(r"apt-get .*install -y", box.called())


def test_disk_too_small_stops(box):
    r = box.run(env={"FREE_KIB": str(1048576)})   # 1 GiB free: below install + margin
    assert r.returncode == 1
    assert "ERROR disk too small" in box.log() and "ROCM102_INSTALL_DONE" in box.log()
    assert not re.search(r"apt-get .*install -y", box.called())
