"""ttft_audit.py on tiny synthetic logs.  No model, no GPU, no server.

Run: python -m pytest tools/pancake/test_ttft_audit.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ttft_audit as A  # noqa: E402


def rec(t, prompt, reused, gen, ms, first=None):
    return {"time": t, "prompt_tokens": prompt, "reused": reused, "prompt_read": prompt - reused,
            "engine_generated": gen, "output_tokens": gen, "prompt_ms": ms,
            "first_engine_token_s": (ms + 17) / 1000 if first is None else first}


def classes(recs, **kw):
    return [c for _, c in A.classify(recs, **kw)]


def test_live_fallback_lost_cold():
    recs = [
        rec(1, 40000, 0, 500, 15000),            # a new long conversation: cold
        rec(2, 40800, 40499, 300, 900),          # continues prompt + output (40,500) up to n - 1: live
        rec(3, 42000, 40795, 200, 1600),         # back at the turn checkpoint (40,800 - 5): fallback
        rec(4, 70, 0, 400, 700),                 # a short side request: cold
        rec(5, 43000, 0, 300, 17000),            # the long conversation again, read from 0: lost
        rec(6, 3000, 0, 1, 2000),                # a probe far smaller than the conversation: cold
    ]
    assert classes(recs) == ["cold", "live", "fallback", "cold", "lost", "cold"]


def test_switch_after_a_side_request_and_benchmark_repeats():
    # the agent's turn after a side request restores its parked copy: more reused than the side request's live state
    recs = [rec(1, 40000, 0, 500, 15000), rec(2, 70, 0, 400, 700), rec(3, 40600, 40499, 300, 1200)]
    assert classes(recs) == ["cold", "cold", "switch"]
    # fresh benchmark prompts of the same size (a unique first line): not a lost conversation
    bench = [rec(1, 25360, 0, 24, 9100), rec(2, 25360, 0, 24, 9100), rec(3, 25359, 0, 24, 9100)]
    assert classes(bench) == ["cold", "cold", "cold"]
    # too long after the earlier request: not the next turn
    late = [rec(1, 40000, 0, 500, 15000), rec(2000, 40800, 0, 300, 15000)]
    assert classes(late) == ["cold", "cold"]


def test_lost_needs_a_long_conversation_and_bounded_growth():
    recs = [rec(1, 5000, 0, 100, 3000), rec(2, 5200, 0, 100, 3000)]
    assert classes(recs) == ["cold", "cold"]                          # below --min-reused
    assert classes(recs, min_reused=4096) == ["cold", "lost"]
    far = [rec(1, 40000, 0, 100, 15000), rec(2, 90000, 0, 100, 40000)]
    assert classes(far) == ["cold", "cold"]                           # grew past output + max_growth


def test_summary_bands_and_shares():
    recs = [rec(1, 40000, 0, 500, 15000), rec(2, 40530, 40499, 300, 375), rec(3, 41330, 40829, 300, 1000)]
    s = A.summarize_turns(A.classify(recs))
    assert s["classes"]["live"]["n"] == 2
    assert s["classes"]["live"]["tokens_read"] == 31 + 501
    bands = {tuple(b["read"]): b for b in s["live_bands"]}
    assert bands[(1, 40)]["median_ms"] == 375 and bands[(301, 800)]["median_ms"] == 1000
    assert s["outside_timer_ms"]["p50"] == 17.0


def test_engine_log_lines():
    lines = [
        "strata serve: conversation cache: parked 25383 tokens in 218.0 ms; parked=1 bytes=1 evictions=0 snapshot_bytes=1 reused_kv_bytes=0",
        "strata serve: conversation cache: parked 67711 tokens in 1417.1 ms; parked=4 bytes=1 evictions=1 snapshot_bytes=1 reused_kv_bytes=0",
        "strata serve: conversation cache: skip parking (snapshot 1574 MiB exceeds available budget)",
        "strata serve: conversation cache: restored 72644 tokens (live) in 131.8 ms",
        "strata serve: conversation cache: restored 2066 tokens (checkpoint) in 44.0 ms",
        "strata serve: file tier I/O this request: the OS read 53686.9 MB from storage (432 major faults) for 26117.7 MB of expert reads",
        "strata serve: file tier I/O this request: the OS read 15.2 MB from storage (0 major faults) for 0.0 MB of expert reads",
        "unrelated line",
    ]
    e = A.summarize_engine_log(lines)
    assert e["parks"] == 2 and e["skips"] == 1 and e["restores"] == 2
    assert e["skip_reasons"] == ["snapshot 1574 MiB exceeds available budget"]
    assert e["restore_ms_16k_plus"] == [131.8]
    assert e["requests_with_storage_reads_over_1gb"] == 1 and e["storage_read_gb_max"] == 52.4
    assert [b["n"] for b in e["park_bands"]] == [1, 1]


def test_main_end_to_end(tmp_path, capsys):
    turns = tmp_path / "turns.jsonl"
    turns.write_text("\n".join(json.dumps(r) for r in [rec(1, 40000, 0, 500, 15000), rec(2, 40530, 40499, 300, 375)])
                     + "\nnot json\n")
    log = tmp_path / "engine.log"
    log.write_text("strata serve: conversation cache: parked 40499 tokens in 300.0 ms; parked=1\n")
    out = tmp_path / "s.json"
    assert A.main(["--turns", str(turns), "--engine-log", str(log), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "live" in text and "1 parks" in text
    assert json.loads(out.read_text())["turns"]["classes"]["live"]["n"] == 1
