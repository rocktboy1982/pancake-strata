"""Unit tests for mid_edit_bench.py with the fake streaming server of test_ttft_bench.py (stdlib only, no engine, no GPU).
Run: python3 -I -m pytest -q tools/pancake/test_mid_edit_bench.py"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mid_edit_bench as M  # noqa: E402
import test_ttft_bench as TT  # noqa: E402


def test_edit_paragraph_changes_one_paragraph_of_the_same_size():
    rng = random.Random(1)
    paras = M.make_paragraphs(20, 30, rng)
    edited, idx = M.edit_paragraph(paras, 0.5, rng)
    assert idx == 10
    assert [i for i, (a, b) in enumerate(zip(paras, edited)) if a != b] == [10]
    assert len(edited[10].split()) == len(paras[10].split())
    assert edited[:10] == paras[:10]


def test_edit_fraction_is_clamped_inside_the_text():
    rng = random.Random(1)
    paras = M.make_paragraphs(10, 5, rng)
    assert M.edit_paragraph(paras, 0.001, rng)[1] == 1       # never the first paragraph (keeps a shared head)
    assert M.edit_paragraph(paras, 0.999, rng)[1] == 9


def test_edit_token_position_scales_with_the_words_before():
    paras = ["a b c d"] * 10                 # 4 words each
    assert M.edit_token_position(1000, paras, 0) == 0
    assert M.edit_token_position(1000, paras, 5) == 500
    assert M.edit_token_position(1000, [], 0) == 0


def test_run_reports_reuse_below_the_edit_with_a_prefix_cache_server():
    f = TT.Fake(first_delay=0.0, tail_delay=0.0)
    try:
        cl = TT.client(f)
        res = M.run(cl, 600, [0.25, 0.5, 0.75], 2, 30, 8, seed=7, log=lambda *_: None)
    finally:
        f.close()
    assert res["status"] == "measured" and len(res["rows"]) == 6
    s = {x["fraction"]: x for x in res["summary"]}
    assert s[0.25]["reused_tokens"] < s[0.5]["reused_tokens"] < s[0.75]["reused_tokens"]
    for r in res["rows"]:
        assert r["ok"] and r["reused_tokens"] <= r["edit_token"] + 100   # the fake reuses the common prefix (its system + role words are not in the estimate)
        assert r["reused_vs_edit"] is not None and 0.5 < r["reused_vs_edit"] <= 1.6


def test_compare_text_lists_both_files():
    a = {"scenario": {"summary": [{"fraction": 0.5, "reused_tokens": 100.0, "read_tokens": 900.0, "ttft_s": 2.0}]}}
    b = {"scenario": {"summary": [{"fraction": 0.5, "reused_tokens": 400.0, "read_tokens": 600.0, "ttft_s": 1.5}]}}
    t = M.compare_text(a, b)
    assert "0.750" in t and "400" in t and "100" in t


def test_main_writes_the_json(tmp_path):
    f = TT.Fake(first_delay=0.0, tail_delay=0.0)
    out = tmp_path / "m.json"
    try:
        rc = M.main(["--port", str(f.port), "--tokens", "600", "--rounds", "1", "--para-words", "30", "--out", str(out),
                     "--seed", "3"])
    finally:
        f.close()
    assert rc == 0
    d = json.loads(out.read_text())
    assert d["tool"] == "mid_edit_bench" and d["scenario"]["summary"]
    assert M.main(["--fractions", "1.5"]) == 2
