"""tools/pancake/promote_confirm.py: screen PROMOTE arms -> confirm specs (no GPU, no engine)."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import promote_confirm as pc  # noqa: E402

STDOUT = """summary ...
generated text vs base (identical = exact ...)
  fast-a               identical on every case -> exact
  fast-b               DIFFERENT text -> lossy or placement-dependent: quality gate
screen verdict (promote: > 0% on a case and no case below -3%; else drop)
  fast-a               short +0.4%  edit +3.0%  -> PROMOTE to confirm
  fast-b               short +0.1%  edit +1.0%  -> PROMOTE to confirm
  slow                 short -4.8%  edit +1.0%  -> drop
"""
SPEC = {"base": "configs/p.json", "cases": ["short", "edit"], "arms": {
    "base": {"exe": "/x/strata"}, "fast-a": {"exe": "/x/strata", "env": {"A": "1"}},
    "fast-b": {"exe": "/x/strata", "env": {"B": "1"}}, "slow": {"exe": "/x/strata", "env": {"C": "1"}}}}


def test_promoted_parses_verdict_and_text_lines():
    arms, diff = pc.promoted(STDOUT)
    assert arms == ["fast-a", "fast-b"]
    assert diff == {"fast-b"}


def test_build_groups_and_marks_lossy():
    g = list(pc.build([(STDOUT, SPEC, "87-x")]).values())
    assert len(g) == 1
    assert list(g[0]["arms"]) == ["base", "fast-a", "fast-b"]
    assert g[0]["lossy"] is True and g[0]["reps"] == 9


def test_build_exact_only_stays_exact_and_nothing_promoted_writes_nothing():
    exact = STDOUT.replace("fast-b               DIFFERENT text", "fast-b               identical text")
    g = list(pc.build([(exact, SPEC, "87-x")]).values())
    assert g[0]["lossy"] is False
    assert pc.build([("screen verdict\n  slow  short -9%  -> drop\n", SPEC, "87-x")]) == {}


def test_main_writes_one_spec_per_group(capsys):
    with tempfile.TemporaryDirectory() as d:
        out, spec = Path(d) / "r.stdout", Path(d) / "87-x.json"
        out.write_text(STDOUT)
        spec.write_text(json.dumps(SPEC))
        other = dict(SPEC, base="configs/q.json")
        spec2 = Path(d) / "90-y.json"
        spec2.write_text(json.dumps(other))
        assert pc.main(["--pair", str(out), str(spec), "--pair", str(out), str(spec2), "--out-dir", d, "--prefix", "107-auto"]) == 0
        files = sorted(Path(d).glob("107-auto-*.json"))
        assert len(files) == 2
        assert "promote_confirm.py" in json.loads(files[0].read_text())["note"]
