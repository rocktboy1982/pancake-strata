"""Tests for pcie_link.py on a fake sysfs tree (symlinks are not used: the device directories are nested the way the resolved path is)."""
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pcie_link as P  # noqa: E402

GPU, BR, RP = "0000_03_00.0", "0000_02_00.0", "0000_00_01.1"
P.SEP = "_"   # Windows file names have no colon; the code reads the separator from P.SEP


def make(root: Path, l1_on: set, with_files=True):
    """root port 00:01.1 > bridge 02:00.0 > gpu 03:00.0, as one nested directory (so realpath yields the whole chain); the bus listing is a second tree
    that points at it by a copy of the three files that matter (no symlinks: Windows tests)."""
    (root / "sys/module/pcie_aspm/parameters").mkdir(parents=True)
    (root / "sys/module/pcie_aspm/parameters/policy").write_text("[default] performance powersave powersupersave\n")
    for b, speed in ((RP, "8.0 GT/s PCIe"), (BR, "8.0 GT/s PCIe"), (GPU, "8.0 GT/s PCIe")):
        d = root / "sys/bus/pci/devices" / b
        (d / "link").mkdir(parents=True)
        (d / "class").write_text("0x030000\n" if b == GPU else "0x060400\n")
        (d / "vendor").write_text("0x1002\n")
        (d / "current_link_speed").write_text(speed + "\n")
        (d / "current_link_width").write_text("16\n")
        (d / "max_link_speed").write_text("16.0 GT/s PCIe\n")
        if with_files:
            for f in P.ASPM_FILES:
                (d / "link" / f).write_text("1\n" if (f == "l1_aspm" and b in l1_on) else "0\n")


def test_display_devices_and_verdicts(tmp_path):
    make(tmp_path, {BR})
    assert P.display_devices(tmp_path) == [GPU]
    assert P.display_devices(tmp_path, vendor="0x10de") == []
    hops = [P.hop_info(tmp_path, b) for b in (GPU, BR, RP)]
    assert P.verdict(hops) == "L1-ON"                      # one hop on is enough
    assert hops[0]["speed"].startswith("8.0") and hops[0]["width"] == "16"
    make_off = tmp_path / "off"
    make(make_off, set())
    assert P.verdict([P.hop_info(make_off, b) for b in (GPU, BR, RP)]) == "L1-OFF"


def test_unknown_without_link_files(tmp_path):
    make(tmp_path, set(), with_files=False)
    assert P.verdict([P.hop_info(tmp_path, GPU)]) == "UNKNOWN"


def test_cli_text_and_json(tmp_path):
    make(tmp_path, {GPU})
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert P.main(["--root", str(tmp_path)]) == 0
    assert "pcie_aspm policy: [default] performance" in out.getvalue() and f"{GPU}: " in out.getvalue()
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        P.main(["--root", str(tmp_path), "--json", "--bdf", GPU])
    j = json.loads(out.getvalue())
    assert j["devices"][0]["verdict"] == "L1-ON" and j["devices"][0]["hops"][0]["aspm"]["l1_aspm"] is True


def test_no_device_exit_2(tmp_path):
    with contextlib.redirect_stderr(io.StringIO()):
        assert P.main(["--root", str(tmp_path)]) == 2
