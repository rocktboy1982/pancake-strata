"""Unit tests for spec_lint.py: the schema read from the harness source, the merge rules, and synthetic good and bad specs.
Static only: no engine, no GPU, and the harness is never imported (its source is read as text and with ast).
Run: python -m pytest tools/pancake/test_spec_lint.py   (or python tools/pancake/test_spec_lint.py)"""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import spec_lint as S  # noqa: E402

BASE = "configs/pancake/flashnext-iq2_xs-fast-hip.json"     # a repo template: exe build-hip/strata, env, args (--max-context)
SCHEMA = S.Schema(S.ROOT)


def lint_text(text, schema=SCHEMA):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "spec.json"
        p.write_text(text)
        return S.lint_spec(p, schema)


def lint(spec, schema=SCHEMA):
    return lint_text(json.dumps(spec), schema)


def good(**over):
    spec = {"base": BASE, "arms": {"base": {}, "kmiss": {"env": {"STRATA_ADAPT_KMISS": "2,8"},
                                                         "args": {"--adapt-swaps": "96"}}},
            "cases": ["short", "edit"], "reps": 2}
    spec.update(over)
    return spec


def errors(items):
    return [m for lvl, _, m in items if lvl == S.ERROR]


def warnings(items):
    return [m for lvl, _, m in items if lvl == S.WARN]


def has(items, level, *words):
    """True when some problem of this level has every word in its "where: message" text."""
    return any(lvl == level and all(w in f"{where}: {m}" for w in words) for lvl, where, m in items)


def test_good_spec_is_clean():
    items = lint(good())
    assert errors(items) == [] and warnings(items) == [], items


def test_schema_is_read_from_the_harness():
    assert {"short", "edit", "long", "long64k", "longdec128k"} <= SCHEMA.cases
    assert SCHEMA.arm_keys == {"args", "config", "drafting", "env", "exe", "request"}
    assert SCHEMA.request_keys == {"thinking", "reasoning_effort", "max_tokens", "temperature"}
    assert {"session", "turns", "arms", "add_arms", "reps", "shared_system"} <= SCHEMA.replay_keys
    assert {"base", "arms", "cases", "reps", "temperature", "top_p", "seed", "prefill", "tasks", "margin"} <= SCHEMA.top_keys
    assert {"strata", "strata-q36", "strata-poly"} <= SCHEMA.targets
    assert SCHEMA.env_known("STRATA_ADAPT_KMISS") and SCHEMA.env_known("STRATA_SPEC_MIN_TEMP")   # include/ readers
    assert not SCHEMA.env_known("STRATA_ADAPT_KMISSS")
    assert "--adapt-swaps" in SCHEMA.flags_of("strata") and "--adapt-swap" not in SCHEMA.flags_of("strata")
    assert "--threads" in SCHEMA.flags_of("strata-poly") and "--threads" not in SCHEMA.flags_of("strata")
    assert SCHEMA.flags_of("strata-unknown") is None
    assert (SCHEMA.root / SCHEMA.default_tasks).exists() and (SCHEMA.root / SCHEMA.default_quality_set).exists()


def test_merge_mirrors_make_config():
    base = {"env": {"A": "1", "B": "2"}, "args": ["--spec", "4", "--kv", "int8"]}
    arm = {"env": {"A": None, "C": 3}, "args": {"--spec": None, "--kv": "", "--new": "x"},
           "drafting": "mtp", "config": {"port": None, "k": 1}}
    cfg = S.merged(base, arm)
    assert cfg["env"] == {"B": "2", "C": "3"}
    assert cfg["args"] == ["--kv", "--new", "x"]
    assert cfg["drafting"] == "mtp" and cfg["k"] == 1 and "port" not in cfg
    assert base["env"] == {"A": "1", "B": "2"}                       # the base is not changed


def test_bad_json_and_duplicate_keys():
    assert has(lint_text("{ not json"), S.ERROR, "not readable JSON")
    dup = ('{"base": "%s", "arms": {"base": {}, "kmiss": {"env": {"STRATA_ADAPT_KMISS": "2,8"}},'
           ' "kmiss": {"env": {"STRATA_ADAPT_KMISS": "3,8"}}}}' % BASE)
    assert has(lint_text(dup), S.ERROR, "duplicate key 'kmiss'")


def test_required_keys():
    assert has(lint({"arms": {"base": {}}}), S.ERROR, '"base" is required')
    assert has(lint({"base": BASE}), S.ERROR, '"arms" is required')
    assert has(lint({"base": BASE, "arms": {}}), S.ERROR, '"arms" is required')


def test_unknown_top_level_key_and_case():
    spec = good()
    spec["case"] = spec.pop("cases")
    assert has(lint(spec), S.ERROR, "unknown top-level key 'case'")
    assert has(lint(good(cases=["short", "longdec999k"])), S.ERROR, "unknown case 'longdec999k'")


def test_unknown_arm_key_and_arm_name():
    spec = good()
    spec["arms"]["kmiss"]["enviroment"] = {}
    assert has(lint(spec), S.ERROR, "unknown arm key 'enviroment'")
    spec = good()
    spec["arms"]["a/b"] = {"env": {"STRATA_ADAPT_KMISS": "2,8"}}
    assert has(lint(spec), S.ERROR, "file name")


def test_env_and_flag_typos_are_errors():
    spec = good()
    spec["arms"]["kmiss"]["env"] = {"STRATA_ADAPT_KMISSS": "2,8"}
    assert has(lint(spec), S.ERROR, "env STRATA_ADAPT_KMISSS is not read")
    spec = good()
    spec["arms"]["kmiss"]["args"] = {"--adapt-swap": "96"}
    assert has(lint(spec), S.ERROR, "flag --adapt-swap is not accepted by the strata parser")


def test_flags_follow_the_arm_exe():
    poly = good(arms={"base": {}, "p": {"exe": "build-hip/strata-poly", "args": {"--threads": "4"}}})
    assert not errors(lint(poly)) and not has(lint(poly), S.WARN, "no flag parser")   # found from the arm's exe
    bad_poly = good(arms={"base": {}, "p": {"exe": "build-hip/strata-poly", "args": {"--threadz": "4"}}})
    assert has(lint(bad_poly), S.ERROR, "flag --threadz is not accepted by the strata-poly parser")
    wrong = good(arms={"base": {}, "p": {"args": {"--threads": "4"}}})
    assert has(lint(wrong), S.ERROR, "flag --threads is not accepted by the strata parser")
    assert has(lint(good(arms={"base": {}, "x": {"exe": "build-hip/strata-foo"}})), S.ERROR, "not an engine target")


def test_value_types():
    assert has(lint(good(arms={"base": {}, "r": {"request": ["high"]}})), S.ERROR, '"request" must be an object')
    spec = good()
    spec["arms"]["kmiss"]["env"] = {"STRATA_ADAPT_KMISS": True}
    assert has(lint(spec), S.ERROR, "env STRATA_ADAPT_KMISS: the value must be")
    spec = good()
    spec["arms"]["kmiss"]["args"] = {"--adapt-swaps": True}
    assert has(lint(spec), S.ERROR, "flag --adapt-swaps: the value must be")
    assert has(lint(good(reps=0)), S.ERROR, "reps must be")
    req = good(arms={"base": {}, "r": {"request": {"thinking": "yes", "temprature": 0.2}}})
    assert has(lint(req), S.ERROR, "request thinking must be") and has(lint(req), S.ERROR, "request key 'temprature'")


def test_arms_that_change_nothing():
    spec = good(arms={"base": {}, "same": {"env": {"STRATA_EXPDEC": "1"}}})     # the base already sets it to "1"
    assert has(lint(spec), S.WARN, "arm same: changes nothing against the base config")
    spec = good(arms={"base": {}, "absent": {"env": {"STRATA_SPEC_PROB": None}}})   # removing a key the base lacks
    assert has(lint(spec), S.WARN, "arm absent: changes nothing against the base config")
    spec = good(arms={"base": {}, "a": {"env": {"STRATA_ADAPT_KMISS": "2,8"}},
                      "b": {"env": {"STRATA_ADAPT_KMISS": "2,8"}}})
    assert has(lint(spec), S.WARN, "arm b: changes nothing against arm a")


def test_base_config_problems():
    assert has(lint(good(base="configs/pancake/nope.json")), S.ERROR, "config configs/pancake/nope.json not found")
    items = lint(good(base="pancake-flashnext-iq2_xs-fast-hip.json"))             # generated, not here: a note only
    assert not errors(items)
    if not (S.ROOT / "pancake-flashnext-iq2_xs-fast-hip.json").exists():          # present after gen_bases.sh / a test run
        assert has(items, S.INFO, "is generated and not in this checkout")


def test_env_missing_from_the_base_only_fails_when_an_arm_sets_env():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "noenv.json"
        base.write_text(json.dumps({"exe": "build-hip/strata", "args": ["--spec", "4"]}))
        no_env = good(base=str(base), arms={"base": {}, "x": {"args": {"--spec": "2"}}})
        assert not errors(lint(no_env))
        bad = good(base=str(base), arms={"base": {}, "x": {"env": {"STRATA_EXPDEC": "0"}}})
        assert has(lint(bad), S.ERROR, 'has no "env" object')


def test_max_context_must_fit_the_long_cases():
    with tempfile.TemporaryDirectory() as d:
        cfg = json.loads((SCHEMA.root / BASE).read_text())
        i = cfg["args"].index("--max-context")
        cfg["args"][i + 1] = "65536"
        base = Path(d) / "small.json"
        base.write_text(json.dumps(cfg))
        assert has(lint(good(base=str(base), cases=["long64k"], arms={"base": {}})), S.WARN, "case long64k needs about")
        assert not warnings(lint(good(base=str(base), cases=["short"], arms={"base": {}})))
    assert not warnings(lint(good(cases=["longdec128k"], arms={"base": {}})))          # 120K + 8% + reply fits 131072


def test_replay_block():
    spec = good(replay={"arms": ["base", "ghost"], "sesion": "x", "add_arms": {"kmiss": {}},
                        "turns": "no-such-turns.jsonl", "reps": 0})
    items = lint(spec)
    assert has(items, S.ERROR, "unknown replay key 'sesion'")
    assert has(items, S.ERROR, "replay arm 'ghost' is neither")
    assert has(items, S.ERROR, "add_arms 'kmiss' has the same name as a spec arm")
    assert has(items, S.ERROR, "reps must be an integer")
    assert has(items, S.WARN, "turn log no-such-turns.jsonl is not here")


def test_task_set_and_request_arms():
    req = {"base": BASE, "arms": {"base": {}, "xhigh": {"request": {"thinking": True, "reasoning_effort": "xhigh"}}}}
    assert not errors(lint(req))                                                  # default task set exists
    assert has(lint(dict(req, tasks="bench/tasks/missing.json")), S.ERROR, "task set bench/tasks/missing.json is missing")


def test_exe_outside_the_checkout_is_a_warning():
    spec = good(arms={"base": {}, "o": {"exe": "/nonexistent-elsewhere/build-hip/strata"}})
    items = lint(spec)
    assert not errors(items) and has(items, S.WARN, "lies outside this checkout")


def test_real_specs_lint_without_crashing():
    assert not [i for i in S.lint_harness(SCHEMA) if i[0] == S.ERROR]
    for p in sorted((S.ROOT / "bench/specs").glob("*.json")):
        assert isinstance(S.lint_spec(p, SCHEMA), list), p


def test_cli_exit_status():
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
        ok, bad = Path(d) / "ok.json", Path(d) / "bad.json"
        ok.write_text(json.dumps(good()))
        bad.write_text(json.dumps(good(cases=["nope"])))
        assert S.main([str(ok)]) == 0
        assert S.main([str(bad)]) == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
