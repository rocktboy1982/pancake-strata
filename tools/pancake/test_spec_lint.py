"""Unit tests for spec_lint.py: the phantom arm check (every STRATA_* env key and --flag of every arm in bench/specs must be
recognised by this repo's sources), on a synthetic source tree and synthetic specs.
Static only: no engine, no GPU, no git (git_ls is replaced by a fixed list).
Run: python -m pytest tools/pancake/test_spec_lint.py   (or python tools/pancake/test_spec_lint.py)"""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec_lint as S  # noqa: E402


class Tree:
    """A fake repo: an env .def file, one source file that parses --good-flag, and a spec directory.  Redirects spec_lint to it."""

    def __init__(self, root):
        self.root = Path(root)
        self.env_def = self.root / "env.def"
        self.env_def.write_text("X(STRATA_KNOWN_KNOB)\n", encoding="utf-8")
        self.src = self.root / "generate.cpp"
        self.src.write_text('if (a == "--good-flag") {}\nconst char* k = "STRATA_IN_SOURCE";\n', encoding="utf-8")
        self.specs = self.root / "specs"
        self.specs.mkdir()

    def spec(self, name, arms):
        (self.specs / name).write_text(json.dumps({"base": "p.json", "arms": arms}), encoding="utf-8")

    def __enter__(self):
        self.saved = (S.ENV_DEF, S.git_ls, dict(S._TEXT_CACHE))
        S.ENV_DEF = self.env_def
        S.git_ls = lambda: [str(self.src)]
        S._TEXT_CACHE.clear()
        return self

    def __exit__(self, *a):
        S.ENV_DEF, S.git_ls = self.saved[0], self.saved[1]
        S._TEXT_CACHE.clear()
        S._TEXT_CACHE.update(self.saved[2])


def tree():
    return tempfile.TemporaryDirectory()


def run_main(t, *extra):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = S.main(["--dir", str(t.specs), *extra])
    return code, out.getvalue(), err.getvalue()


def test_check_arm_recognises_env_from_the_def_and_from_sources():
    with tree() as d, Tree(d):
        assert S.check_arm(Path("s.json"), "a", "STRATA_KNOWN_KNOB", "env", [str(Path(d) / "generate.cpp")], True)
        assert S.check_arm(Path("s.json"), "a", "STRATA_IN_SOURCE", "env", [str(Path(d) / "generate.cpp")], True)
        assert S.check_arm(Path("s.json"), "a", "STRATA_GGML_DIR", "env", [], True)                 # the one built-in exception
        assert not S.check_arm(Path("s.json"), "a", "STRATA_NOWHERE", "env", [str(Path(d) / "generate.cpp")], True)


def test_check_arm_flags_are_matched_without_the_dashes():
    with tree() as d, Tree(d):
        srcs = [str(Path(d) / "generate.cpp")]
        assert S.check_arm(Path("s.json"), "a", "--good-flag", "args", srcs, True)
        assert not S.check_arm(Path("s.json"), "a", "--bad-flag", "args", srcs, True)


def test_walk_specs_yields_every_arm_and_skips_unreadable_files():
    with tree() as d, Tree(d) as t:
        t.spec("a.json", {"base": {}, "x": {"env": {"STRATA_KNOWN_KNOB": "1"}}})
        (t.specs / "broken.json").write_text("{not json", encoding="utf-8")
        (t.specs / "plan.json").write_text(json.dumps({"runner": "bench_win_ab"}), encoding="utf-8")   # no arms: nothing yielded
        got = [(f.name, arm) for f, arm, _ in S.walk_specs(t.specs)]
        assert got == [("a.json", "base"), ("a.json", "x")]


def test_clean_specs_exit_0():
    with tree() as d, Tree(d) as t:
        t.spec("a.json", {"base": {}, "x": {"env": {"STRATA_KNOWN_KNOB": "1", "OTHER_VAR": "1"}, "args": {"--good-flag": "2", "plain": 1}}})
        code, out, _ = run_main(t)
        assert code == 0 and "PHANTOM" not in out


def test_phantom_env_and_flag_exit_1_and_are_named():
    with tree() as d, Tree(d) as t:
        t.spec("a.json", {"base": {}, "x": {"env": {"STRATA_NOWHERE": "1"}, "args": {"--bad-flag": "2"}}})
        code, out, err = run_main(t)
        assert code == 1
        assert "PHANTOM-ENV  a.json arm(x) STRATA_NOWHERE" in out and "PHANTOM-FLAG a.json arm(x) --bad-flag" in out
        assert "2 phantom" in err
        code, out, err = run_main(t, "--quiet")
        assert code == 1 and err == ""                                                           # quiet: findings only, no summary


def test_usage_errors_exit_2():
    with tree() as d, Tree(d) as t:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            assert S.main(["--dir", str(t.root / "missing")]) == 2                              # not a directory
            S.git_ls = lambda: []
            assert S.main(["--dir", str(t.specs)]) == 2                                          # no sources


PRESET = ["--kv", "int8", "--kv-resident", "32768", "--max-context", "131072"]


def test_inert_gate_kv_grow_under_kv_resident():
    assert "--kv-resident 32768" in S.inert_gate(PRESET, {"args": {"--kv-grow": ""}})
    assert "--kv-resident" in S.inert_gate(PRESET, {"env": {"STRATA_KV_GROW": "1"}})
    assert S.inert_gate(PRESET, {}) is None                                                    # the base arm
    assert S.inert_gate(PRESET, {"args": {"--kv-grow": "", "--kv-resident": None}}) is None    # ring removed: the switch can act
    assert S.inert_gate(PRESET, {"args": {"--kv-grow": "", "--kv-resident": "0"}}) is None
    assert S.inert_gate(PRESET, {"args": {"--kv-grow": ""}, "env": {"STRATA_KV_GROW": "0"}}) is None   # the env wins, as in the engine
    assert S.inert_gate(PRESET, {"args": {"--kv-grow": "", "--no-kv-grow": ""}}) is None


def test_inert_gate_is_a_warning_not_a_failure():
    with tree() as d, Tree(d) as t:
        t.src.write_text(t.src.read_text(encoding="utf-8") + 'if (a == "--kv-grow") {}' + chr(10), encoding="utf-8")
        preset = Path(d) / "p.json"
        preset.write_text(json.dumps({"args": PRESET}), encoding="utf-8")
        (t.specs / "a.json").write_text(json.dumps({"base": str(preset), "arms": {"base": {}, "g": {"args": {"--kv-grow": ""}}}}), encoding="utf-8")
        code, out, err = run_main(t)
        assert code == 0 and "INERT-GATE   a.json arm(g)" in out and "INERT-GATE   a.json arm(base)" not in out
        assert "1 arm(s)" in err


def test_real_specs_walk_without_crashing():
    n = 0
    for _spec, arm, body in S.walk_specs(S.ROOT / "bench/specs"):
        assert isinstance(arm, str) and isinstance(body, dict)
        n += 1
    assert n > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
