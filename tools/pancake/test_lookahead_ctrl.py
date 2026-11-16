"""Tests for lookahead_ctrl.py on SYNTHETIC lines written from the format of include/strata/program/lookahead_ctrl.hpp (no engine run)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lookahead_ctrl as lc  # noqa: E402

OK = """\
strata lookahead stats: 100 windows, 4700 layers scored ...
strata lookahead ctrl: windows=100 slots=6 stale_gpu=0.6667 stale_cpu=n/a control=1.0000 prev=n/a
strata lookahead ctrl layer 1: slots=3 stale_gpu=0.3333 stale_cpu=n/a control=1.0000 prev=0.2500
strata lookahead ctrl layer 2: slots=3 stale_gpu=1.0000 stale_cpu=n/a control=1.0000 prev=n/a
strata lookahead ctrl: windows=200 slots=12 stale_gpu=0.5000 stale_cpu=0.4000 control=1.0000 prev=0.3000
strata lookahead ctrl layer 1: slots=6 stale_gpu=0.5000 stale_cpu=0.4000 control=1.0000 prev=0.3000
"""


def test_last_block_wins_and_layers_belong_to_their_block():
    b = lc.parse(OK.splitlines())
    assert b["windows"] == 200 and b["slots"] == 12
    assert b["total"] == {"stale_gpu": 0.5, "stale_cpu": 0.4, "control": 1.0, "prev": 0.3}
    assert list(b["layers"]) == [1] and b["layers"][1]["slots"] == 6


def test_na_is_none_and_no_lines_is_none():
    b = lc.parse(OK.splitlines()[:4])
    assert b["total"]["stale_cpu"] is None and b["layers"][2]["prev"] is None
    assert lc.parse(["nothing here"]) is None


def test_verdict_flags_a_control_under_one():
    bad = "strata lookahead ctrl: windows=100 slots=10 stale_gpu=0.2000 stale_cpu=0.2100 control=0.8800 prev=0.1000"
    assert "probe is wrong" in lc.verdict(lc.parse([bad]))
    assert "reads the routing back" in lc.verdict(lc.parse(OK.splitlines()))
    nc = "strata lookahead ctrl: windows=100 slots=10 stale_gpu=0.2000 stale_cpu=n/a control=n/a prev=n/a"
    assert "no control" in lc.verdict(lc.parse([nc]))


def test_table_prints_every_layer_and_the_total(tmp_path):
    log = tmp_path / "e.log"
    log.write_text(OK)
    b = lc.parse(OK.splitlines())
    t = lc.table(b)
    assert t.splitlines()[0].startswith("layer") and "total" in t.splitlines()[-1]
    assert lc.main([str(log)]) == 0
    assert lc.main([str(tmp_path / "missing_blocks.log")] if False else [str(log), "--json", str(tmp_path / "o.json")]) == 0
    assert (tmp_path / "o.json").exists()
