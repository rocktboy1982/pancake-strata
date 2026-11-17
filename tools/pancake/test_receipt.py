"""Tests for receipt.py (LEADS.md Part P L9): hashes, git state, the GPU sysfs reader on a fake tree, the receipt file.
No GPU, no server.   python -m pytest tools/pancake/test_receipt.py"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import receipt as R  # noqa: E402


def test_file_sha256(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    r = R.file_sha256(f)
    assert r["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" and r["bytes"] == 3
    assert R.file_sha256(tmp_path / "missing") is None


def test_text_sha256_is_utf8():
    assert R.text_sha256("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    assert R.text_sha256("é") != R.text_sha256("e")


def test_git_state_outside_repo(tmp_path):
    g = R.git_state(tmp_path)
    assert g["head"] is None and g["dirty_count"] == 0


def test_git_state_dirty(tmp_path):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "f.txt").write_text("1")
    git("add", "f.txt")
    git("commit", "-qm", "x")
    (tmp_path / "f.txt").write_text("2")
    g = R.git_state(tmp_path)
    assert len(g["head"]) == 40 and g["dirty_count"] == 1 and "f.txt" in g["dirty"][0]


def test_gpu_state_fake_sysfs(tmp_path):
    dev = tmp_path / "card1" / "device"
    (dev / "hwmon" / "hwmon3").mkdir(parents=True)
    (dev / "pp_dpm_sclk").write_text("0: 500Mhz\n1: 1500Mhz\n2: 2400Mhz *\n")
    (dev / "pp_dpm_mclk").write_text("0: 96Mhz *\n1: 1249Mhz\n")
    (dev / "power_dpm_force_performance_level").write_text("auto\n")
    (dev / "hwmon" / "hwmon3" / "power1_cap").write_text("315000000\n")
    (tmp_path / "card1-DP-1").mkdir()
    g = R.gpu_state(drm_root=str(tmp_path))
    assert g["card"] == "card1" and g["sclk"] == "2: 2400Mhz *" and g["mclk"] == "0: 96Mhz *"
    assert g["perf_level"] == "auto" and g["power_cap_uw"] == "315000000"
    assert g["vram_used"] is None            # a missing file is null, not an error


def test_gpu_state_no_card(tmp_path):
    assert R.gpu_state(drm_root=str(tmp_path)) == {"card": None}


def test_receipt_file(tmp_path, monkeypatch):
    exe = tmp_path / "strata"
    exe.write_bytes(b"\x7fELF")
    spec = {"base": "configs/x.json", "arms": {"a": {"exe": str(exe)}, "b": {"exe": str(exe), "env": {"STRATA_X": "1"}}}}
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    monkeypatch.setenv("STRATA_FOO", "bar")
    out = tmp_path / "run.jsonl"
    p = R.write_receipt(out, sp, prompts={"short": "hello", "edit": "x\n\n\ny"}, argv=["abtest", "spec"])
    assert p.name == "run.jsonl.receipt.json" and p.exists()
    r = json.loads(p.read_text())
    assert r["receipt"] == 1 and r["spec"]["sha256"] == R.file_sha256(sp)["sha256"]
    assert r["cases"]["short"] == {"sha256": R.text_sha256("hello"), "chars": 5}
    assert r["arms"]["a"]["exe"]["sha256"] == r["arms"]["b"]["exe"]["sha256"] and r["arms"]["b"]["env"] == {"STRATA_X": "1"}
    assert r["strata_env"]["STRATA_FOO"] == "bar" and "loadavg" in r["host_start"]
    assert r["argv"] == ["abtest", "spec"]


def test_arm_without_exe_uses_base(tmp_path):
    exe = tmp_path / "e"
    exe.write_bytes(b"1")
    arms = R.arm_summary({"arms": {"a": {}, "b": {"exe": str(tmp_path / "gone")}}}, base_exe=str(exe))
    assert arms["a"]["exe"]["bytes"] == 1 and arms["b"]["exe"] is None
