"""Tests for hipc_check.py SDK selection with fake directories (no compiler needed)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hipc_check as H  # noqa: E402


def fake_installer(base: Path, name: str) -> Path:
    r = base / "AMD" / "ROCm" / name
    (r / "bin").mkdir(parents=True)
    (r / "bin" / "clang++.exe").write_text("")
    return r


def fake_pip(base: Path, tag: str, ver: str) -> Path:
    sp = base / f"rocm-sdk-{tag}" / "venv" / "Lib" / "site-packages"
    r = sp / "_rocm_sdk_core"
    (r / "lib" / "llvm" / "bin").mkdir(parents=True)
    (r / "lib" / "llvm" / "bin" / "clang++.exe").write_text("")
    (r / "lib" / "llvm" / "amdgcn" / "bitcode").mkdir(parents=True)
    (sp / f"rocm_sdk_core-{ver}.dist-info").mkdir()
    return r


def pats(base: Path):
    return [str(base / "rocm-sdk-*" / "venv" / "Lib" / "site-packages" / "_rocm_sdk_core"), str(base / "AMD" / "ROCm" / "*")]


def test_newest_wins(tmp_path):
    fake_installer(tmp_path, "6.2")
    fake_pip(tmp_path, "10x", "10.2.0")
    fake_installer(tmp_path, "7.1")
    s = H.pick(pats(tmp_path), env={})
    assert s.version == (10, 2, 0) and s.devlib is not None
    assert [x.version for x in H.find_sdks(pats(tmp_path), env={})] == [(10, 2, 0), (7, 1), (6, 2)]


def test_version_compare_is_numeric(tmp_path):
    fake_installer(tmp_path, "6.10")
    fake_installer(tmp_path, "6.9")
    assert H.pick(pats(tmp_path), env={}).version == (6, 10)


def test_env_override_forces_older(tmp_path):
    old = fake_installer(tmp_path, "6.2")
    fake_pip(tmp_path, "10x", "10.2.0")
    s = H.pick(pats(tmp_path), env={"HIPC_SDK": str(old)})
    assert s.version == (6, 2) and s.root == old


def test_env_override_bad_dir(tmp_path):
    with pytest.raises(SystemExit):
        H.find_sdks(pats(tmp_path), env={"HIPC_SDK": str(tmp_path / "nope")})


def test_none_found(tmp_path):
    with pytest.raises(SystemExit):
        H.pick(pats(tmp_path), env={})


def test_dir_without_clang_ignored(tmp_path):
    (tmp_path / "AMD" / "ROCm" / "9.9" / "bin").mkdir(parents=True)
    fake_installer(tmp_path, "6.2")
    assert H.pick(pats(tmp_path), env={}).version == (6, 2)


def test_command_and_env(tmp_path):
    fake_pip(tmp_path, "10x", "10.2.0")
    s = H.pick(pats(tmp_path), env={})
    cmd = H.command(s, ["a.cu"])
    assert "--clang" in cmd and any(c.startswith("--cflag=--rocm-device-lib-path=") for c in cmd) and cmd[-1] == "a.cu"
    e = H.child_env(s, {"HIP_PATH": "C:/old", "X": "1"})
    assert e["HIP_PATH"] == str(s.root) and e["X"] == "1" and e["HIP_CLANG_PATH"] == ""
