"""tools/pancake/stack_select.py: greedy stack building (MOD_PLAN.md section 6 item 24)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stack_select as ss  # noqa: E402

SUMMARY = """  short  stack                   100.0 +-   1.0  (n=5)
  short  stack+B                 103.0 +-   1.0  (n=5)   vs base:  +3.0% +-  1.0%  -> better
  edit   stack                   150.0 +-   1.0  (n=5)
  edit   stack+B                 151.5 +-   1.0  (n=5)   vs base:  +1.0% +-  1.0%  -> no difference
"""


def test_merge_conflict_and_args():
    st = {"env": {"A": "1"}, "args": ["--x"]}
    assert ss.merge(st, {"env": {"A": "2"}}) is None
    m = ss.merge(st, {"env": {"A": "1", "B": "1"}, "args": ["--x", "--y"]})
    assert m == {"env": {"A": "1", "B": "1"}, "args": ["--x", "--y"]}


def test_decide():
    assert ss.decide({"short": 3.0, "edit": 1.0}, 1.0) == (True, "mean +2.0%")
    assert ss.decide({"short": 3.0, "edit": -2.0}, 1.0)[0] is False   # a case below -1%
    assert ss.decide({"short": 0.2}, 1.0)[0] is False
    assert ss.decide({}, 1.0)[0] is False


def test_parse_step():
    assert ss.parse_step(SUMMARY, "stack+B") == {"short": 3.0, "edit": 1.0}


def test_flow(tmp_path):
    w = tmp_path / "w.json"
    w.write_text(json.dumps([{"arm": "A", "env": {"K": "1"}, "gain": 2}, {"arm": "B", "env": {"K": "2"}, "gain": 1},
                             {"arm": "C", "env": {"L": "1"}, "gain": 0.5, "lossy": True}]))
    s = tmp_path / "s.json"
    ss.main(["init", str(s), str(w), "--base", "p.json", "--exe", "/x/strata"])
    assert ss.main(["next", str(s), "--out-dir", str(tmp_path)]) == 0           # step 1: stack vs stack+A
    spec = json.loads((tmp_path / "stack-s-step01.json").read_text())
    assert list(spec["arms"]) == ["stack", "stack+A"] and spec["arms"]["stack+A"]["env"] == {"K": "1"}
    (tmp_path / "r1").write_text(SUMMARY.replace("stack+B", "stack+A"))
    ss.main(["record", str(s), str(tmp_path / "r1")])                            # A kept
    ss.main(["next", str(s), "--out-dir", str(tmp_path)])                       # B conflicts with A (K): skipped, C next
    st = json.loads(s.read_text())
    assert st["kept"] == ["A"] and st["history"][1] == {"arm": "B", "kept": False, "why": "env conflict"}
    spec = json.loads((tmp_path / "stack-s-step03.json").read_text())
    assert spec["lossy"] is True and spec["arms"]["stack+C"]["env"] == {"K": "1", "L": "1"}
