"""Tests for state_cost.py on SYNTHETIC log lines written from the engine's format strings (generate.cpp), not from a real run."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import state_cost as sc  # noqa: E402

LOG = """\
strata serve: conversation cache sized from the free RAM: 2048 MiB
strata serve: conversation cache: skip parking (70 tokens, below the 8192 minimum; parked=0)
strata serve: conversation cache: skip parking (900 tokens, below the 8192 minimum; parked=1)
strata trace: checkpoint at 33000: sync 2.0 ms, save 70.5 ms (1 stage(s))
strata trace: checkpoint at 33000: 1 evicted in 3.5 ms
strata serve: conversation cache: parked 70904 tokens in 706.0 ms; parked=1 bytes=1555000000 evictions=0 snapshot_bytes=1555000000 reused_kv_bytes=500000000
strata serve: conversation cache: parked 3088 tokens (live) in 100.0 ms; parked=2 bytes=1800000000 evictions=0 snapshot_bytes=283115520 reused_kv_bytes=0
strata serve: conversation cache: skipped 80000 tokens in 50.0 ms; parked=2 bytes=1800000000 evictions=1 snapshot_bytes=1700000000 reused_kv_bytes=0
strata serve: conversation cache: restored 52628 tokens (checkpoint) in 81.0 ms; parked=1 bytes=1555000000
strata serve: conversation cache: borrowed 1000 tokens from a parked conversation, which stays parked, in 12.0 ms; parked=1 bytes=1 borrowed=1
strata serve: conversation disk: disk-restored 40000 tokens (live, 900 MiB) in 900.0 ms (read pass 800.0 ms)
strata serve: conversation disk: read back a 50000-token match (1000 MiB) in 1100.0 ms; parked=1
unrelated line 12 ms
"""


def test_derived_sizes_match_the_documented_numbers():
    assert sc.gdn_state_bytes() == 117669888            # 36 layers x 817,152 floats x 4 = 112.2 MiB (the "~118 MB" of --prompt-cache help)
    assert sc.kv_bytes_per_token() == 14208             # 12 layers x 1184 B (T-164's fit: 14.2 KB/token)


def test_parse_counts_each_event_kind():
    ev = sc.parse_log(LOG.splitlines())
    assert len(ev["parks"]) == 2 and len(ev["skipped_parks"]) == 1
    assert ev["parks"][0] == {"tokens": 70904, "ms": 706.0, "snapshot_bytes": 1555000000, "reused_kv_bytes": 500000000}
    assert len(ev["restores"]) == 1 and ev["restores"][0]["kind"] == "checkpoint"
    assert len(ev["borrows"]) == 1 and ev["borrows"][0]["ms"] == 12.0
    assert ev["skip_reasons"] == {"N tokens, below the N minimum; parked=N": 2}
    assert ev["checkpoints"] == [{"at": 33000, "sync_ms": 2.0, "save_ms": 70.5}]
    assert ev["ckpt_evictions"][0]["ms"] == 3.5
    assert ev["disk_restores"][0]["bytes"] == 900 << 20 and ev["disk_reads"][0]["ms"] == 1100.0


def test_summary_sums_and_shares():
    ev = sc.parse_log(LOG.splitlines())
    turns = [{"duration_s": 10.0, "prompt_ms": 2000.0}, {"duration_s": 30.0, "prompt_ms": 3000.0}]
    s = sc.summarize(ev, turns)
    assert s["parks"]["ms"] == 806.0 and s["parks"]["n"] == 2
    assert s["parks"]["copied_upper_bytes"] == 1555000000 + 283115520 - 500000000
    assert s["restores"]["ms"] == 81.0 and s["restores"]["from_checkpoint"] == 1
    assert s["checkpoints"]["bytes_derived"] == sc.gdn_state_bytes()
    want = 806.0 + 50.0 + 81.0 + 12.0 + 1100.0 + 900.0 + 70.5 + 2.0 + 3.5
    assert abs(s["state_ms"] - want) < 1e-9
    assert abs(s["turns"]["state_share_of_duration"] - want / 1000.0 / 40.0) < 1e-12
    assert abs(s["turns"]["state_share_of_prompt"] - want / 1000.0 / 5.0) < 1e-12
    assert abs(s["turns"]["state_ms_per_request"] - want / 2) < 1e-9


def test_no_checkpoint_lines_says_so_and_no_division_by_zero():
    ev = sc.parse_log(["strata serve: conversation cache: parked 100 tokens in 5.0 ms; parked=1 bytes=9 evictions=0 snapshot_bytes=9"])
    s = sc.summarize(ev, [])
    assert s["checkpoints"]["ms_in_log"] is False
    assert "not in the log" in sc.render(s)
    s0 = sc.summarize(sc.parse_log([]), None)
    assert s0["state_ms"] == 0.0 and "turns" not in s0
    sc.render(s0)


def test_old_format_park_without_bytes_is_flagged():
    ev = sc.parse_log(["strata serve: conversation cache: parked 100 tokens in 5.0 ms; parked=1 bytes=9 evictions=0"])
    s = sc.summarize(ev)
    assert s["parks"]["bytes_in_log"] is False and "older log format" in sc.render(s)


def test_truncated_park_line():
    line = ("strata serve: conversation cache: parked 1280 tokens (truncated from 50000, STRATA_PARK_TRUNCATE) in 30.0 ms; "
            "parked=3 bytes=7 evictions=1 truncations=1 snapshot_bytes=123456")
    ev = sc.parse_log([line])
    assert ev["parks"][0]["tokens"] == 1280 and ev["parks"][0]["snapshot_bytes"] == 123456


def test_main_with_files(tmp_path, capsys):
    lg = tmp_path / "e.log"
    lg.write_text(LOG, encoding="utf-8")
    tl = tmp_path / "t.jsonl"
    tl.write_text(json.dumps({"duration_s": 10.0, "prompt_ms": 2000.0}) + "\nnot json\n", encoding="utf-8")
    out = tmp_path / "o.json"
    assert sc.main([str(lg), "--turns", str(tl), "--json", str(out)]) == 0
    assert "turn log:     1 requests" in capsys.readouterr().out
    assert json.loads(out.read_text())["parks"]["n"] == 2


def test_replay_park_estimate():
    lines = [
        "rep 0 arm base (attempt 0)",
        "  [  1/75] agent    A#0   69995 tok (plan  70082) reused      0 prompt  40.3 s ttft  40.2 s decode  79.5 tok/s wall  57.7 s [parked 70 (226 MiB); parked 70904 (1783 MiB)]",
        "  [  2/75] side    s0#0      67 tok (plan     65) reused      0 prompt   0.7 s ttft   0.7 s decode  52.0 tok/s wall   0.8 s []",
        "  [  3/75] agent    A#1   71664 tok (plan  71788) reused  71391 prompt   1.7 s ttft   1.7 s decode  90.1 tok/s wall   4.7 s [skip (70 tokens, below the 8192 minimum; parked=0)]",
        "  [  4/75] agent    B#24  57763 tok (plan  57763) reused      0 prompt  25.9 s ttft  25.9 s decode 106.8 tok/s wall  33.4 s [parked 463 (231 MiB)]  COLD RE-READ",
        "rep 0 base: prompt 100.0 s (agent 80.0), turn 200.0 s, cold re-reads 3 (80.6 s), min avail 13.0 GiB, peak RSS 41.61 GiB, foreign CPU 4%",
    ]
    r = sc.replay_park_estimate(lines, rate_gib_s=1.0)
    assert len(r) == 1 and r[0]["parks"] == 3 and r[0]["mib"] == 2240 and r[0]["requests"] == 4
    assert abs(r[0]["est_park_s"] - 2240 / 1024.0) < 1e-9
    assert abs(r[0]["share_of_prompt"] - (2240 / 1024.0) / 100.0) < 1e-9 and abs(r[0]["share_of_turn"] - (2240 / 1024.0) / 200.0) < 1e-9
