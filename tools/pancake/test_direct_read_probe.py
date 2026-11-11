"""A52: direct_read_probe.py on small temporary files.  No model, no GPU.

Run: python -m pytest tools/pancake/test_direct_read_probe.py"""
import json
import os
import sys
import zlib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import direct_read_probe as P  # noqa: E402

MIB = P.MIB


def make_file(tmp_path, mib, name="m.bin"):
    path = tmp_path / name
    path.write_bytes(os.urandom(mib * MIB))
    return str(path)


def file_crc(path, offset, length):
    with open(path, "rb") as f:
        f.seek(offset)
        return zlib.crc32(f.read(length))


def test_parse_arms():
    assert P.parse_arms("buffered:6,direct:16") == [("buffered", 6), ("direct", 16)]
    for bad in ("direct", "direct:0", "mmap:4", "direct:x", ""):
        with pytest.raises(ValueError):
            P.parse_arms(bad)


def test_project_seconds():
    assert P.project_seconds(33.0, 3.0) == pytest.approx(11.0)
    assert P.project_seconds(33.0, 0.0) == float("inf")


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("threads", [1, 4, 32])
def test_read_window_returns_the_files_bytes(tmp_path, direct, threads):
    path = make_file(tmp_path, 6)
    # an offset and a length that are not a multiple of the chunk: the last piece is short
    off, length = 1 * MIB, 3 * MIB + 8192
    seconds, crc, ran = P.read_window(path, off, length, MIB, threads, direct)
    assert crc == file_crc(path, off, length)
    assert seconds >= 0
    assert ran in ("direct", "buffered") and (direct or ran == "buffered")


def test_read_window_past_the_end_is_an_error(tmp_path):
    path = make_file(tmp_path, 2)
    with pytest.raises(EOFError):
        P.read_window(path, 1 * MIB, 4 * MIB, MIB, 2, False)


def test_unaligned_window_is_refused(tmp_path):
    path = make_file(tmp_path, 2)
    with pytest.raises(AssertionError):
        P.read_window(path, 100, MIB, MIB, 1, True)


def test_resident_share_of_a_file_just_written(tmp_path):
    path = make_file(tmp_path, 2)
    fd = os.open(path, os.O_RDONLY)
    try:
        share = P.mincore_resident(fd, 0, 2 * MIB)
    finally:
        os.close(fd)
    assert share is None or 0.9 <= share <= 1.0     # tmpfs / page cache keeps what was written


def test_device_info_is_a_dict_or_none(tmp_path):
    info = P.device_info(str(tmp_path))
    assert info is None or {"disk", "model", "link"} <= set(info)


def test_cap_is_enforced(tmp_path):
    path = make_file(tmp_path, 1)
    with pytest.raises(SystemExit) as e:
        P.main([path, "--window-mib", "300", "--arms", "buffered:2,direct:2"])
    assert "cap" in str(e.value)


def test_window_beyond_the_file_is_refused(tmp_path):
    path = make_file(tmp_path, 2)
    with pytest.raises(SystemExit) as e:
        P.main([path, "--offset-mib", "1", "--window-mib", "2", "--arms", "buffered:1"])
    assert "windows end" in str(e.value)


def test_main_reports_one_row_per_arm(tmp_path, capsys):
    path = make_file(tmp_path, 6)
    out = tmp_path / "rows.json"
    P.main([path, "--offset-mib", "0", "--window-mib", "2", "--chunk-mib", "1", "--arms", "buffered:2,direct:2",
            "--arena-gib", "33", "--json", str(out)])
    data = json.loads(out.read_text())
    assert [r["arm"] for r in data["rows"]] == ["buffered:2", "direct:2"]
    for i, r in enumerate(data["rows"]):          # the arms read consecutive windows
        assert r["crc32"] == "%08x" % file_crc(path, i * 2 * MIB, 2 * MIB)
        assert r["gib_s"] > 0 and r["arena_s"] == pytest.approx(33 / r["gib_s"], abs=0.06)
    assert "GiB/s" in capsys.readouterr().out
