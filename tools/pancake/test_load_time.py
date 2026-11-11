"""A52: load_time.py pure helpers (log parsing, PSI, summary table, model paths).  No server, no GPU.

Run: python -m pytest tools/pancake/test_load_time.py   (or python tools/pancake/test_load_time.py)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_time as L  # noqa: E402

LOG = """strata generate: expert arena read unbuffered (MemAvailable 18.2 GiB < arena 33.0 GiB + need)
strata generate: expert arena load: mode unbuffered, 16 threads, 33.02 GiB in 11.42 s (2.89 GiB/s); arena reserved/registered/pinned in 3.10 s before the read
strata generate: expert arena checksum 0123456789abcdef (33.02 GiB)
ready: http://127.0.0.1:8080
"""


def test_parse_load_lines():
    d = L.parse_load(LOG)
    assert d["mode"] == "unbuffered" and d["threads"] == 16 and d["gib"] == 33.02
    assert d["load_s"] == 11.42 and d["gib_s"] == 2.89 and d["register_s"] == 3.10
    assert d["read"] == "unbuffered" and d["why"].startswith("MemAvailable")
    assert d["checksum"] == "0123456789abcdef"
    assert L.parse_load("nothing here") == {}


def test_parse_psi():
    t = ("some avg10=0.00 avg60=0.10 avg300=0.50 total=123456\n"
         "full avg10=0.00 avg60=0.00 avg300=0.20 total=7890\n")
    assert L.parse_psi(t) == {"some": 123456, "full": 7890}
    assert L.parse_psi("") == {}


def rows():
    out = []
    for rep, (d, b) in enumerate([(12.0, 40.0), (14.0, 44.0), (13.0, 42.0)]):
        out.append({"mode": "direct", "cache": "cold", "start_s": d, "gib_s": 2.9, "psi_some_s": 0.1, "psi_full_s": 0.0,
                    "cached_peak_gib": 0.5, "avail_min_gib": 20.0, "loader": "unbuffered", "checksum": "aa"})
        out.append({"mode": "buffered", "cache": "cold", "start_s": b, "gib_s": 0.8, "psi_some_s": 9.0, "psi_full_s": 4.0,
                    "cached_peak_gib": 30.0, "avail_min_gib": 3.0, "loader": "buffered", "checksum": "aa"})
    return out


def test_summary_and_table():
    s = L.summarize(rows())
    by = {(g["mode"], g["cache"]): g for g in s}
    assert by[("direct", "cold")]["start_mean_s"] == 13.0 and by[("buffered", "cold")]["start_mean_s"] == 42.0
    assert by[("direct", "cold")]["start_min_s"] == 12.0 and by[("direct", "cold")]["start_max_s"] == 14.0
    assert by[("buffered", "cold")]["cached_peak_gib"] == 30.0 and by[("direct", "cold")]["checksums_agree"]
    t = L.format_table(s)
    assert "direct" in t and "buffered" in t and "cold start: direct 13.0 s, buffered 42.0 s" in t and "+69% shorter" in t
    assert "WARNING" not in t


def test_checksum_mismatch_is_flagged_and_missing_values_do_not_crash():
    r = rows()
    r[1]["checksum"] = "bb"
    r.append({"mode": "default", "cache": "warm", "start_s": None})
    s = L.summarize(r)
    assert not s[0]["checksums_agree"]
    assert "WARNING: the arena checksums differ" in L.format_table(s)


def test_model_paths_include_the_shards():
    cfg = {"args": ["--pack", "/p/packs/iq2_xs", "--native", "/m/Q-00001-of-00002.gguf", "--ple-gguf", "/m/Q-00002-of-00002.gguf"]}
    p = L.model_paths(cfg)
    assert "/p/packs/iq2_xs" in p and "/m/Q-00001-of-00002.gguf" in p and "/m/Q-00002-of-00002.gguf" in p


def test_mode_env_values():
    assert L.MODE_ENV["direct"] == {"STRATA_UNBUFFERED_LOAD": "1"}
    assert L.MODE_ENV["buffered"] == {"STRATA_UNBUFFERED_LOAD": "0"}
    assert L.MODE_ENV["default"] == {}


def test_meminfo_reads_this_machine():
    mi = L.meminfo_gib()
    assert mi == {} or ("MemAvailable" in mi and mi["MemAvailable"] > 0)



def test_wait_for_ram_returns_at_once_when_enough_and_gives_up_at_the_timeout():
    saved = L.meminfo_gib
    try:
        L.meminfo_gib = lambda: {"MemAvailable": 50.0}
        assert L.wait_for_ram(48.0, timeout=5, poll=0.01) < 0.5
        L.meminfo_gib = lambda: {"MemAvailable": 10.0}
        assert 0.05 <= L.wait_for_ram(48.0, timeout=0.1, poll=0.01) < 1.0   # never enough: stops at the timeout
        L.meminfo_gib = lambda: {}
        assert L.wait_for_ram(48.0, timeout=5, poll=0.01) < 0.5            # no /proc/meminfo: do not wait on nothing
    finally:
        L.meminfo_gib = saved


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
