"""Tests for thinking_caps.py (MOD_PLAN A17): percentile, cap rule, grouping by effort, provisional levels, cut estimate.
python -m pytest tools/pancake/test_thinking_caps.py   (or python tools/pancake/test_thinking_caps.py)"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import thinking_caps as TC  # noqa: E402


def row(effort, think, max_new=32768, gen=None, decode_ms=None, dur=10.0, **extra):
    r = {"max_new": max_new, "reasoning_tokens": think, "engine_generated": gen if gen is not None else think + 100,
         "decode_ms": decode_ms if decode_ms is not None else (think + 100) * 10.0, "duration_s": dur}
    if effort != "unset":
        r["effort"] = effort
    r.update(extra)
    return r


def test_percentile_matches_linear_interpolation():
    assert TC.percentile([], 99) == 0.0
    assert TC.percentile([5], 99) == 5.0
    assert TC.percentile([1, 2, 3, 4], 50) == 2.5
    assert abs(TC.percentile(list(range(1, 101)), 99) - 99.01) < 1e-9


def test_cap_rule_rounds_up_with_margin():
    assert TC.suggest_cap(1061, 1.25, 128) == 1408            # 1326.25 -> 1408
    assert TC.suggest_cap(1, 1.25, 128) == 128                 # never below one round step
    assert TC.suggest_cap(1024, 1.0, 128) == 1024


def test_groups_by_effort_and_leaves_out_the_rest():
    rows = [row("low", t) for t in range(100, 300)] + [row("medium", t) for t in range(500, 700)]
    rows += [row("unset", 400)] * 7                              # an older line: counted, not capped
    rows += [row(None, 0)] * 5                                   # thinking off
    rows += [row("low", 9999, max_new=512)] * 3                  # a benchmark row: left out
    rows += [row("high", 0)]                                     # a level with no thinking turn: no entry
    a = TC.analyse(rows)
    assert set(a["levels"]) == {"low", "medium"}
    assert a["unknown"] == 7
    low, med = a["levels"]["low"], a["levels"]["medium"]
    assert low["n"] == 200 and med["n"] == 200 and not low["provisional"]
    assert low["max"] == 299 and low["cap"] == TC.suggest_cap(low["p99"])
    assert a["config"] == {"low": low["cap"], "medium": med["cap"]}


def test_provisional_level_gets_no_config_entry():
    a = TC.analyse([row("low", t) for t in range(100, 150)])
    assert a["levels"]["low"]["provisional"] and a["config"] == {}
    assert "PROVISIONAL" in TC.report(a) and "none suggested" in TC.report(a)
    a = TC.analyse([row("low", t) for t in range(100, 150)], min_turns=10)
    assert not a["levels"]["low"]["provisional"] and "low" in a["config"]


def test_cut_estimate_counts_only_the_tail_above_cap_plus_grace():
    rows = [row("medium", 300) for _ in range(99)] + [row("medium", 5000, gen=5100, decode_ms=51000.0)]
    a = TC.analyse(rows, margin=1.0, min_turns=1)
    m = a["levels"]["medium"]
    # p99 of 99 x 300 and one 5000 = 300 + 0.01*4700 -> 347 -> cap 384; the one tail turn thinks 5000 > 384 + 512
    assert m["cap"] == 384 and m["turns_hit"] == 1
    assert m["tokens_cut"] == 5000 - (384 + 512)
    assert abs(m["seconds_saved"] - round((5000 - 896) / 5100 * 51.0, 1)) < 0.11


def test_rumination_and_entropy_figures_are_reported_when_present():
    rows = [row("high", 400, rumination_max=0.1 * (i % 5), entropy_mean=0.5 + i % 3) for i in range(30)]
    a = TC.analyse(rows, min_turns=1)
    h = a["levels"]["high"]
    assert h["rumination_max"]["n"] == 30 and h["entropy_mean"]["n"] == 30
    text = TC.report(a)
    assert "rumination_max" in text and "entropy_mean" in text


def test_cli_reads_files_skips_cut_lines_and_prints_the_config():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "turns.jsonl"
        lines = [json.dumps(row("low", t)) for t in range(100, 400)]
        p.write_text("\n".join(lines) + '\n{"max_new": 32768, "reaso\n')       # the last line is cut
        assert TC.main([str(p), "--min-turns", "100"]) == 0
        assert TC.main([str(p), "--json"]) == 0
        assert TC.main([str(Path(d) / "missing.jsonl")]) == 2
        assert len(TC.load_rows([str(p)])) == 300


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
