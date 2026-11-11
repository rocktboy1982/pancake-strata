"""Tests for build_task_set.py: the context analysis, the mined cases, the mutations, the checker text, a whole build on a throw-away git
repository (all three families, the history one from a real commit), the selection, and a sample of the committed task set.  No model, no GPU.
Run: python -m pytest tools/pancake/test_build_task_set.py   (or python tools/pancake/test_build_task_set.py)"""
import argparse
import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_task_set as B  # noqa: E402
import task_gate as TG  # noqa: E402

MODULE = '''
import re
import json
import numpy as np
from . import sibling

LIMIT = 10
WORD = re.compile(r"\\w+")

def helper(x):
    return x + LIMIT

def clamp(x, hi=LIMIT, *, name: "str" = "x"):
    """Clamp x to hi."""
    out = [helper(i) for i in range(3) if i]
    return min(x, hi) if x > 0 else 0

def reads_file(p):
    return open(p).read()

def uses_numpy(x):
    return np.zeros(x)

def uses_sibling(x):
    return sibling.go(x)

def uses_unknown(x):
    return mystery(x)

if LIMIT:
    GATED = 1

def uses_gated(x):
    return x + GATED
'''


def test_referenced_globals_sees_defaults_comprehensions_and_nested_scopes():
    src = ast.unparse(B.Module("m.py", MODULE).funcs["clamp"])
    assert B.referenced_globals(src) >= {"LIMIT", "helper", "min", "range"}
    assert "x" not in B.referenced_globals(src) and "out" not in B.referenced_globals(src)
    nested = "def f(a):\n    def g(b):\n        return b + K\n    return g(a) + len(L)\n"
    assert B.referenced_globals(nested) == {"K", "len", "L"}
    assert B.referenced_globals("K = 3\nBIG = K * 2") == set()


def test_impurity():
    f = lambda s: B.impurity(ast.parse(s).body[0])
    assert f("def f(p):\n    return open(p).read()") == "uses open"
    assert f("def f():\n    global X\n    X = 1") == "global"
    assert f("def f(p):\n    p.unlink()") == "uses .unlink"
    assert f("def f(xs):\n    return sorted(xs)") is None


def test_context_for_collects_the_needed_definitions_in_file_order():
    mod = B.Module("pkg/m.py", MODULE)
    fn = ast.unparse(mod.funcs["clamp"])
    setup, why = B.context_for(mod, fn, "clamp")
    assert why is None
    assert setup.index("LIMIT = 10") < setup.index("def helper") and "WORD" not in setup
    assert B.context_for(mod, ast.unparse(mod.funcs["uses_numpy"]), "uses_numpy")[1] == "np: imports numpy"
    assert "imports from ." in B.context_for(mod, ast.unparse(mod.funcs["uses_sibling"]), "uses_sibling")[1]
    assert "unresolved name mystery" in B.context_for(mod, ast.unparse(mod.funcs["uses_unknown"]), "uses_unknown")[1]
    assert "not a plain definition" in B.context_for(mod, ast.unparse(mod.funcs["uses_gated"]), "uses_gated")[1]
    assert "uses open" not in (B.context_for(mod, ast.unparse(mod.funcs["clamp"]), "clamp")[0] or "")


def test_eligible():
    ok = ast.parse("def f(a):\n    if a:\n        return 1\n    return 2\n").body[0]
    assert B.eligible(ok)
    assert not B.eligible(ast.parse("def f(a):\n    return a\n").body[0])
    assert not B.eligible(ast.parse("@cache\ndef f(a):\n    if a:\n        return 1\n    return 2\n").body[0])
    assert not B.eligible(ast.parse("def f(p):\n    if p:\n        return 1\n    return open(p)\n").body[0])


def test_literal_call_and_mining_resolve_by_import():
    by_stem = {"m": ["pkg/m.py"], "other": ["a/other.py", "b/other.py"]}
    assert B.resolve_module("pkg.m", "tests/test_m.py", by_stem) == "pkg/m.py"
    assert B.resolve_module("other", "tests/test_x.py", by_stem) is None                     # ambiguous, not next to the test
    assert B.resolve_module("other", "a/test_x.py", by_stem) == "a/other.py"                 # the one beside the test
    test = ("import m as M\nfrom pkg.m import clamp as c2\nimport json\nx = 5\n"
            "def test_a():\n    assert M.clamp(5, hi=3) == 3\n    assert M.clamp(x) == 5\n    assert c2(-1) == 0\n    assert json.loads('1') == 1\n    M.clamp(5, hi=3)\n")
    got = B.mine_calls("tests/test_m.py", test, by_stem)
    assert got[("pkg/m.py", "clamp")] == [{"args": ["5"], "kwargs": {"hi": "3"}}, {"args": ["-1"], "kwargs": {}}]
    assert set(got) == {("pkg/m.py", "clamp")}


def test_mutations_change_the_source_and_are_counted():
    fn = ast.parse("def f(xs, n):\n    out = 0\n    for x in xs:\n        if x > n and x != 3:\n            out += x * 2\n        else:\n"
                   "            break\n    return min(out, 100)\n").body[0]
    n = B.mutation_sites(fn)
    assert n >= 8
    base = ast.unparse(fn)
    labels = set()
    for site in range(n):
        label, new = B.mutate(fn, site)
        labels.add(label.split(" ")[0])
        assert ast.unparse(new) != base, label
        ast.parse(ast.unparse(new))
    assert {"comparison", "operator", "negated", "call", "off-by-one", "break"} <= labels
    assert ast.unparse(fn) == base                       # the original is never touched
    assert B.mutate(ast.parse("def f(a):\n    if not a:\n        return 1\n    return 2\n").body[0], 0)[0] in ("negated condition", "dropped `not`")


def test_record_marks_what_is_not_stable_or_not_literal():
    cases = [{"args": ["3"], "kwargs": {}}]
    assert B.record("", "def f(a):\n    return a * 2", "f", cases) == [["ok", "6"]]
    assert B.record("", "def f(a):\n    raise ValueError(a)", "f", cases) == [["raises", "ValueError"]]
    assert B.record("", "def f(a):\n    return object()", "f", cases) == [None]
    assert B.record("", "def f(a):\n    class E(Exception):\n        pass\n    raise E()", "f", cases) == [None]
    assert B.record("", "def f(a):\n    while True:\n        pass", "f", cases, timeout=1) is None
    assert B.stable_cases("import os", "def f(a):\n    return os.getpid()", "f", cases) == []
    assert B.stable_cases("", "def f(a):\n    return a * 2", "f", cases) == [(cases[0], ["ok", "6"])]


def test_checker_text_accepts_the_reference_and_rejects_a_broken_one():
    cases = [{"args": ["3"], "kwargs": {}}, {"args": ["'x'"], "kwargs": {}}, {"args": ["2.5"], "kwargs": {"k": "2"}}]
    ref = "def f(a, k=1):\n    if isinstance(a, str):\n        raise ValueError(a)\n    return {'v': [a * k, (a,)]}"
    kept = B.stable_cases("", ref, "f", cases)
    assert [r[0] for _, r in kept] == ["ok", "raises", "ok"]
    test = B.build_test("f", kept)
    assert TG.run_python(ref + "\n\n" + test)[0] == 0
    assert TG.run_python(ref.replace("a * k", "a + k") + "\n\n" + test)[0] != 0
    assert TG.run_python(ref.replace("raise ValueError(a)", "return 0") + "\n\n" + test)[0] != 0
    assert TG.run_python(ref.replace("(a,)", "[a]") + "\n\n" + test)[0] != 0                  # a list is not a tuple
    assert TG.run_python(ref.replace("{'v':", "{'w':") + "\n\n" + test)[0] != 0
    same = ref.replace("a * k", "(a * k) * (1 + 1e-12)")                                      # float noise is tolerated
    assert TG.run_python(same + "\n\n" + test)[0] == 0
    assert B.failing(kept, [["ok", "7"], ["raises", "ValueError"], None]) == [0, 2]
    assert B.failing(kept, None) == []


def test_select_is_round_robin_and_respects_the_caps():
    def t(i, fam, path, fn):
        return {"id": f"{fam}-{i}", "meta": {"family": fam, "path": path, "function": fn}}
    tasks = [t(i, fam, f"f{i % 3}.py", f"fn{i}") for i in range(12) for fam in ("history", "bugfix", "implement")]
    out = B.select(tasks, 9, per_file=3, per_function=2, seed=1)
    assert len(out) == 9 and {x["meta"]["family"] for x in out[:3]} == {"history", "bugfix", "implement"}
    assert max(sum(1 for x in out if x["meta"]["path"] == p) for p in ("f0.py", "f1.py", "f2.py")) <= 3
    same_fn = [t(i, "bugfix", "a.py", "same") for i in range(5)]
    assert len(B.select(same_fn, 5, per_file=9, per_function=2, seed=1)) == 2
    assert [x["id"] for x in B.select(tasks, 9, 3, 2, 1)] == [x["id"] for x in out]            # deterministic


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True, capture_output=True)


def make_repo(root):
    root = Path(root)
    (root / "tools").mkdir()
    old = ('import re\n\nLIMIT = 4\n\n\ndef clamp(x, hi=LIMIT):\n    """Clamp x between 0 and hi."""\n    if x < 0:\n        return 0\n'
           '    if x > hi:\n        return hi\n    total = 0\n    for i in range(x):\n        total += 1\n    return total\n\n\n'
           'def words(text):\n    """The words of a text, lower case, in order."""\n    out = []\n    for w in re.findall(r"[a-z]+", text.lower()):\n'
           '        out.append(w)\n    return out\n')
    (root / "tools" / "m.py").write_text(old)
    (root / "tools" / "test_m.py").write_text(
        "import m\n\n\ndef test_clamp():\n    assert m.clamp(2) == 2\n    assert m.clamp(-1) == 0\n    assert m.clamp(9) == 4\n    assert m.clamp(3, hi=2) == 2\n\n\n"
        "def test_words():\n    assert m.words('A b') == ['a', 'b']\n    assert m.words('') == []\n    assert m.words('x-y') == ['x', 'y']\n")
    git(root, "init", "-q")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "add m")
    (root / "tools" / "m.py").write_text(old.replace("if x < 0:\n        return 0", "if x <= 0:\n        return 1"))
    git(root, "commit", "-q", "-am", "clamp: zero and below give 1")
    return root


def args_for(root, **kw):
    a = dict(root=str(root), rev="HEAD", seed=1, jobs=2, families=list(B.FAMILIES), per_file=9, per_function=3, max_history=20,
             mutants_tried=12, mutants_per_function=2, count=50)
    a.update(kw)
    return argparse.Namespace(**a)


def test_build_on_a_throw_away_repository_makes_valid_tasks_of_every_family():
    with tempfile.TemporaryDirectory() as d:
        root = make_repo(d)
        rev, tasks, cnt = B.build(args_for(root))
        fams = {t["meta"]["family"] for t in tasks}
        assert fams == set(B.FAMILIES), (fams, dict(cnt))
        assert cnt["emitted"] == len(tasks) and cnt["validated"] >= len(tasks) and cnt["recorded"] == 2
        assert TG.validate_tasks(tasks) == [] and TG.selftest(tasks) == []
        hist = next(t for t in tasks if t["meta"]["family"] == "history")
        assert hist["meta"]["function"] == "clamp" and hist["meta"]["commits"] == ["clamp: zero and below give 1"]
        assert "if x < 0" in hist["prompt"] and "clamp: zero and below give 1" in hist["prompt"] and "if x <= 0" in hist["reference"]["content"]
        bug = next(t for t in tasks if t["meta"]["family"] == "bugfix" and t["meta"]["function"] == "clamp")
        assert "LIMIT = 4" in bug["expect"]["setup"] and bug["meta"]["hint"] in ("example", "none")
        impl = next(t for t in tasks if t["meta"]["family"] == "implement")
        assert "NotImplementedError" in impl["prompt"] and "NotImplementedError" in impl["wrong"]["content"]
        # a wrong answer written by hand fails, the reference and an equivalent rewrite pass
        t = hist
        other = t["reference"]["content"].replace("return 1", "return 2 - 1")
        assert TG.grade(t, {"content": other})["ok"] and not TG.grade(t, {"content": t["wrong"]["content"]})["ok"]
        # --check on a file written the way main() writes it
        path = Path(d) / "set.json"
        json.dump({"name": "x", "version": 1, "tasks": tasks}, open(path, "w"))
        assert B.check(str(path), str(root)) == 0
        tasks[0]["reference"]["content"] = tasks[0]["wrong"]["content"]
        json.dump({"name": "x", "version": 1, "tasks": tasks}, open(path, "w"))
        assert B.check(str(path), str(root)) > 0


def test_setup_runs_before_the_answer_and_is_optional():
    task = {"id": "t", "kind": "code_edit", "prompt": "p", "expect": {"setup": "K = 5", "test": "assert f() == 5"},
            "reference": {"content": "```python\ndef f(a=0):\n    return K\n```"}, "wrong": {"content": "```python\ndef f(a=0):\n    return 4\n```"}}
    assert TG.validate_tasks([task]) == [] and TG.selftest([task]) == []
    assert TG.grade({**task, "expect": {"test": "assert f() == 5"}}, {"content": task["reference"]["content"]})["ok"] is False    # no setup: K undefined
    assert TG.validate_tasks([{**task, "expect": {"setup": 3, "test": "x"}}])


def test_the_committed_task_set_is_valid_and_has_at_least_100_tasks():
    path = B.ROOT / "bench/tasks/repo-history.json"
    if not path.exists():
        return
    ts, tasks = TG.load_tasks(path)
    assert len(tasks) >= 100 and TG.validate_tasks(tasks) == []
    assert {t["meta"]["family"] for t in tasks} == set(B.FAMILIES)
    assert len({t["id"] for t in tasks}) == len(tasks)
    sample = tasks[::max(1, len(tasks) // 12)]
    assert TG.selftest(sample) == []
    assert "funnel" in ts and ts["funnel"]["validated"] >= len(tasks)


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
