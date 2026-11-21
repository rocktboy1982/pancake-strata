"""Tests for gpu_snapshot.py (T15) on a fake sysfs tree and fixture amd-smi text (the real tools need the box)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpu_snapshot as G  # noqa: E402


def fake_root(tmp_path, cur="8.0 GT/s PCIe", mx="16.0 GT/s PCIe"):
    d = tmp_path / "sys/class/drm/card1/device"
    (d / "hwmon/hwmon3").mkdir(parents=True)
    for k, v in {"vendor": "0x1002", "device": "0x744c", "current_link_speed": cur, "max_link_speed": mx, "current_link_width": "16", "max_link_width": "16",
                 "power_dpm_force_performance_level": "high",
                 "pp_power_profile_mode": "NUM        MODE_NAME\n  0 BOOTUP_DEFAULT *\n  5      COMPUTE\n",
                 "pp_dpm_sclk": "0: 500Mhz\n1: 2400Mhz *\n", "pp_dpm_mclk": "0: 96Mhz\n1: 1249Mhz *\n", "pp_dpm_pcie": "0: 2.5GT/s, x1\n1: 16.0GT/s, x16 *\n"}.items():
        (d / k).write_text(v + "\n")
    for k, v in {"power1_cap": "257000000", "power1_cap_max": "290000000", "power1_cap_min": "231000000"}.items():
        (d / "hwmon/hwmon3" / k).write_text(v)
    (tmp_path / "sys/class/drm/card1-DP-1").mkdir()      # connector entry is skipped
    return tmp_path


def test_link_info():
    assert G.link_info("16.0 GT/s PCIe", "16") == {"speed": "16.0 GT/s PCIe", "gts": 16.0, "gen": 4, "width": 16, "theoretical_gbs": 31.5}
    assert G.link_info("8.0 GT/s PCIe", "16")["theoretical_gbs"] == 15.76
    assert G.link_info(None, None)["gen"] is None


def test_sysfs_snapshot(tmp_path):
    cards = G.sysfs_cards(fake_root(tmp_path))
    assert len(cards) == 1
    c = cards[0]
    assert c["current"]["gen"] == 3 and c["max"]["gen"] == 4 and c["perf_level"] == "high"
    assert "BOOTUP_DEFAULT" in c["power_profile_active"] and "2400" in c["dpm_sclk"] and "x16" in c["dpm_pcie"]
    assert c["power_w"] == {"power1_cap": 257.0, "power1_cap_max": 290.0, "power1_cap_min": 231.0}


def test_reconcile_messages(tmp_path):
    snap = G.take(fake_root(tmp_path), "idle", 1, 0, False, 13.4)
    txt = "\n".join(snap["reconcile"])
    assert "below its max" in txt and "Gen3-class" in txt and snap["phase"] == "idle" and snap["date"]
    snap = G.take(fake_root(tmp_path / "b", cur="16.0 GT/s PCIe"), "load", 2, 0, False, 26.0)
    txt = "\n".join(snap["reconcile"])
    assert "Gen4" in txt and snap["link_states_seen"]


def test_parse_smi_json_both_shapes():
    a = json.dumps([{"gpu": 0, "bus": {"pcie_width": 16, "pcie_speed": {"value": 8, "unit": "GT/s"}, "bdf": "0000:03:00.0"}}])
    b = json.dumps({"gpu_data": [{"gpu": 0, "power": {"socket_power": 120, "power_cap": 257}, "energy": {"total_energy_consumption": 5}}]})
    fa, fb = G.parse_smi(a), G.parse_smi(b)
    assert fa["0.bus.pcie_width"] == 16 and "0.bus.bdf" not in fa
    assert any(k.endswith("power_cap") for k in fb) and any("energy" in k for k in fb)


def test_parse_smi_csv_old_and_new_columns():
    old = "gpu,pcie_width,pcie_speed,power_cap\n0,16,8,257\n"
    new = "GPU,PCIE_WIDTH,PCIE_MAX_SPEED,SOCKET_POWER,COUNTER\n0,16,16,121,x\n"
    assert G.parse_smi(old)["gpu0.pcie_width"] == "16"
    n = G.parse_smi("WARNING: partial\n" + new)
    assert n["gpu0.PCIE_MAX_SPEED"] == "16" and "gpu0.COUNTER" not in n
    assert G.parse_smi("") == {} and G.parse_smi("no csv here") == {}


def test_cli_out_never_overwrites(tmp_path, capsys):
    root = fake_root(tmp_path / "r")
    out = tmp_path / "o"
    for _ in range(2):
        assert G.main(["--root", str(root), "--no-smi", "--out", str(out), "--phase", "idle"]) == 0
    files = sorted(p.name for p in out.glob("*.json"))
    assert len(files) == 2 and files[0].startswith("snapshot-")


def test_compare_and_h2d_json(tmp_path, capsys):
    root = fake_root(tmp_path / "r")
    h = tmp_path / "h.json"
    h.write_text('noise\n{"h2d_gbs":13.1,"d2h_gbs":12.9,"mib":256,"reps":20}\n')
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    for p, extra in ((a, ["--h2d-json", str(h)]), (b, ["--h2d-gbs", "26.5", "--phase", "gen4"])):
        G.main(["--root", str(root), "--no-smi", *extra])
        text = capsys.readouterr().out
        p.write_text(text[text.index("{"):])
    assert json.loads(a.read_text())["h2d_gbs"] == 13.1
    assert G.main(["--compare", str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "13.1" in out and "26.5" in out and "idle -> gen4" in out
    assert G.main(["--compare", str(a), str(tmp_path / "missing.json")]) == 2
