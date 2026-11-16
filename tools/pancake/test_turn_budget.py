"""tools/pancake/turn_budget.py: classes, least-squares slopes, idle gaps (MOD_PLAN.md section 6 items 19 and 23)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import turn_budget as tb  # noqa: E402


def row(t, dur=2.0, **kw):
    r = {"time": t, "duration_s": dur, "prompt_ms": 500.0, "decode_ms": 1000.0, "prompt_total": 1000, "prompt_read": 1000,
         "reused": 0, "output_tokens": 100, "engine_generated": 100, "max_new": 512, "temperature": 0, "hit_rate": 0.9,
         "drafts_offered": 10, "drafts_accepted": 7, "first_engine_token_s": 0.6, "decode_tok_s": 100.0}
    r.update(kw)
    return r


def test_classes():
    assert tb.is_agent(row(0, max_new=32768))
    assert tb.is_agent(row(0, temperature=0.2))
    assert not tb.is_agent(row(0, max_new=512, temperature=0))
    assert not tb.is_agent(row(0, max_new=512, temperature=None))


def test_slope_known_answer():
    s, k = tb.slope([1, 2, 3, 4], [2, 4, 6, 8])
    assert abs(s - 2.0) < 1e-9 and k == 4
    assert tb.slope([1, 1, 1], [1, 2, 3])[0] is None and tb.slope([1, 2], [1, 2])[0] is None


def test_gaps_are_start_minus_previous_end():
    rows = [row(100.0), row(110.0, dur=2.0), row(111.0, dur=5.0)]   # starts at 108 and 106: gaps 8 and (106 - 110 < 0, skipped)
    assert tb.gaps(rows) == [8.0]


def test_hit_point_slope_and_budget_split():
    # decode time per token falls 0.5 ms per hit point: 50 ms at 80%, 45 ms at 90% (decode_ms / engine_generated)
    rows = [row(i * 10.0, hit_rate=h, decode_ms=100 * (100 - 0.5 * (h * 100 - 80) * 0 - (h * 100 - 80) * 0.5) * 1.0)
            for i, h in enumerate([0.80, 0.85, 0.90, 0.95])]
    rep = tb.analyse(rows)
    assert rep["turns"] == 4 and rep["decode_ms_per_token_per_hit_point"]["slope"] < 0
    s = rep["seconds"]
    assert abs(s["prompt"] - 2.0) < 1e-9 and abs(s["duration"] - 8.0) < 1e-9
    assert tb.analyse([])["turns"] == 0


def test_main_runs_on_a_file(tmp_path, capsys):
    import json
    p = tmp_path / "turns.jsonl"
    p.write_text("\n".join(json.dumps(row(i * 5.0, max_new=32768 if i % 2 else 512)) for i in range(6)) + "\nnot json\n")
    assert tb.main([str(p)]) == 0
    out = capsys.readouterr().out
    assert "6 turns" in out and "== agent: 3 turns" in out and "== bench: 3 turns" in out
