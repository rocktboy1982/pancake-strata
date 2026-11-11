"""test_switch_report.py - switch_report.py and promote.py's switch and tier checks on synthetic engine logs and abtest jsonl (no GPU,
no model): a good arm, a void arm, a partly acting one, results without a kept log, an exempt switch, mixed tiers, old rows without a tier.

    python tools/pancake/test_switch_report.py      (or: python -m pytest tools/pancake/test_switch_report.py)
"""
import ast
import contextlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import promote as P  # noqa: E402
import switch_report as S  # noqa: E402
from test_promote import GOOD, item, verdict, why, world  # noqa: E402

SWITCH = "STRATA_WAIT_COPY256"
ACTED = f"strata switch: {SWITCH} acted: verify plan copied with 256 threads"
NO_GAIN = {"x": {"short": [0.0] * 4, "edit": [0.2, 0.3, 0.1, 0.2]}}


def switch_spec(w, stem, env=None, arm="x"):
    """The spec of World.spec with an env overlay on arm."""
    path = w.root / w.spec(stem)
    spec = json.loads(path.read_text())
    spec["arms"][arm]["env"] = env or {SWITCH: "1"}
    path.write_text(json.dumps(spec))
    return f"bench/specs/{stem}.json"


def keep_logs(w, stem, arm, lines, reps=4, attempt=0, suffix=S.LOG_SUFFIX):
    """abtest's per-attempt engine log: <results>/<stem>/<arm>-r<rep>-a<attempt>.engine.log; lines: one text, or one per rep."""
    d = w.out / stem
    d.mkdir(exist_ok=True)
    for rep in range(reps):
        text = lines if isinstance(lines, str) else lines[rep]
        (d / f"{arm}-r{rep}-a{attempt}{suffix}").write_text("[strata] engine started\n" + text + "\n[strata] ready\n")


def tag_tier(w, stem, tiers):
    """Adds a tier field to the rows of the jsonl: tiers {arm: tier}; an arm that is not named keeps its rows without one."""
    path = w.out / f"{stem}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("\n".join(json.dumps({**r, **({"tier": tiers[r["arm"]]} if r["arm"] in tiers else {})}) for r in rows) + "\n")


# ------------------------------------------------------------------------------------------------ the reader
def test_parse_reads_both_line_kinds_under_any_prefix_and_name_form():
    text = "\n".join(["[strata] 12:00 something else", "strata switch: STRATA_A acted: one", "[strata] strata switch: B=1 acted: two",
                      "strata switch: C will not act: needs STRATA_D=1", "strata switch: strata switch is not a line"])
    assert S.parse(text) == {"A": ("acted", "one"), "B": ("acted", "two"), "C": ("will not act", "needs STRATA_D=1")}


def test_acted_beats_will_not_act_whatever_the_order():
    both = S.parse("strata switch: A will not act: no\nstrata switch: A acted: yes\nstrata switch: A will not act: again")
    assert both == {"A": ("acted", "yes")}


def test_an_arm_asks_for_the_strata_keys_that_are_on():
    env = {"STRATA_A": "1", "STRATA_B": "0", "STRATA_C": None, "STRATA_D": "", "STRATA_E": "2,8", "GPU_MAX_HW_QUEUES": "8", "STRATA_F": 1}
    assert S.switches(env) == ["STRATA_A", "STRATA_E", "STRATA_F"] and S.switches(None) == []


def test_status_over_the_runs_of_an_arm():
    env, ok, none, refuse = {"STRATA_A": "1"}, "strata switch: A acted: x", "nothing", "strata switch: A will not act: no doorbell"
    assert S.arm_status(env, [ok, ok], {}) == {"STRATA_A": (S.ACTED, "x")}
    assert S.arm_status(env, [ok, none], {})["STRATA_A"][0] == S.PARTIAL
    assert S.arm_status(env, [ok, None], {})["STRATA_A"] == (S.PARTIAL, "acted in 1 of 2 runs, 1 without an engine log")
    assert S.arm_status(env, [none, none], {})["STRATA_A"][0] == S.VOID
    assert S.arm_status(env, [none, refuse], {})["STRATA_A"] == (S.VOID, "will not act: no doorbell")
    assert S.arm_status(env, [None, None], {})["STRATA_A"][0] == S.NO_LOG
    assert S.arm_status(env, [none], {"A": "no line yet"}) == {"STRATA_A": (S.EXEMPT, "no line yet")}


def test_the_repo_exemption_file_is_well_formed_and_keeps_the_switches_that_log_checked():
    exempt = S.load_exempt(P.ROOT)
    assert len(exempt) > 50 and all(k == k.upper() and not k.startswith("STRATA_") for k in exempt)
    logging = {"WAIT_COPY256", "ROUTE_FAST", "ROUTER_FUSE", "IC_WARM_MB", "EXPDEC_SWIGLU", "HIP_GLOBAL_WEIGHTS"}   # switch_report.hpp
    assert not logging & set(exempt)
    assert S.load_exempt(P.ROOT, {"switch_exempt": {"later": ["STRATA_NEW"]}})["NEW"] == "later"


def test_the_reader_only_reads():
    tree = ast.parse(Path(S.__file__).read_text())
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not imported & {"subprocess", "os", "shutil", "pty"}, imported
    calls = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "") for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not calls & {"system", "popen", "Popen", "run", "write_text", "write_bytes", "mkdir", "unlink", "rename", "touch", "write", "open"}, calls


def test_command_line_reports_every_arm_and_exits_1_for_a_void_switch():
    with world() as w:
        w.runs("s", GOOD)
        spec = switch_spec(w, "s")
        keep_logs(w, "s", "x", ACTED, reps=2)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert S.main([str(w.out / "s"), "--spec", str(w.root / spec)]) == 0
        assert f"ACTED    {SWITCH}" in out.getvalue() and "arm base: 0 engine starts, 0 switches" in out.getvalue()
        keep_logs(w, "s", "x", "nothing here", reps=2, attempt=1)       # a later engine start that never acted
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert S.main([str(w.out / "s"), "--spec", str(w.root / spec)]) == 1
        assert f"PARTIAL  {SWITCH}" in out.getvalue()


# ------------------------------------------------------------------------------------------------ promote: the switch rule
def test_arm_whose_switch_acted_in_every_run_is_judged_as_before():
    with world() as w:
        w.runs("s", GOOD)
        spec = switch_spec(w, "s")
        keep_logs(w, "s", "x", ACTED)
        v, r = verdict(w, item(spec, claim=["edit"]))
        assert v == "PROMOTE", why(r)
        assert f"switch {SWITCH} acted in 4 of 4 runs: verify plan copied" in why(r)


def test_void_arm_is_held_not_rejected_and_nothing_else_is_judged():
    with world() as w:
        w.runs("s", NO_GAIN)                                    # the numbers alone would be REJECT: no gain
        spec = switch_spec(w, "s")
        keep_logs(w, "s", "x", f"strata switch: {SWITCH} will not act: inert under deferral")
        v, r = verdict(w, item(spec, claim=["edit"]))
        assert v == "HOLD" and not r["arms"][0].reject, why(r)
        assert f"switch never acted: the arm measured nothing ({SWITCH}: will not act: inert under deferral)" in why(r)
        assert "claim " not in why(r) and "parity" not in why(r)       # the base's numbers are not shown as the arm's
        assert "HOLD     it" in P.render([r])


def test_void_arm_with_a_silent_switch_says_where_to_register_a_switch_that_logs_nothing_yet():
    with world() as w:
        w.runs("s", GOOD)
        spec = switch_spec(w, "s")
        keep_logs(w, "s", "x", "[strata] nothing about switches")
        v, r = verdict(w, item(spec, claim=["edit"]))
        assert v == "HOLD" and "no line in 4 engine logs" in why(r) and "switch-exempt.json" in why(r)


def test_arm_whose_switch_acted_in_some_runs_only_is_held_with_its_numbers_shown():
    with world() as w:
        w.runs("s", GOOD)
        spec = switch_spec(w, "s")
        keep_logs(w, "s", "x", [ACTED, ACTED, "silent", "silent"])
        v, r = verdict(w, item(spec, claim=["edit"]))
        assert v == "HOLD" and f"switch {SWITCH} acted in 2 of 4 runs" in why(r) and "claim  edit" in why(r), why(r)


def test_result_without_a_kept_engine_log_is_unverified_not_void():
    with world() as w:
        w.runs("s", GOOD)
        v, r = verdict(w, item(switch_spec(w, "s"), claim=["edit"]))
        assert v == "HOLD" and f"switch {SWITCH} not verified: no <arm>-r<rep>-a<attempt>.engine.log kept for any of the 4 runs" in why(r), why(r)
        assert "never acted" not in why(r)


def test_the_server_narration_file_is_not_an_engine_log():
    with world() as w:                                         # abtest has always kept <arm>-r<rep>-a<attempt>.out: no engine lines in it
        w.runs("s", GOOD)
        spec = switch_spec(w, "s")
        keep_logs(w, "s", "x", "[strata] still starting (26 s) - please wait ...", suffix=".out")
        v, r = verdict(w, item(spec, claim=["edit"]))
        assert v == "HOLD" and "not verified" in why(r) and "never acted" not in why(r), why(r)


def test_only_the_attempts_abtest_used_are_read():
    with world() as w:
        w.runs("s", GOOD, noisy=(("x", 0),))                    # rep 0: attempt 0 noisy ...
        with (w.out / "s.jsonl").open("a") as f:                # ... and repeated as attempt 1, clean
            rec = next(json.loads(x) for x in (w.out / "s.jsonl").read_text().splitlines() if json.loads(x)["arm"] == "x")
            f.write(json.dumps({**rec, "attempt": 1, "clean": True, "noise": 1.0}) + "\n")
        spec = switch_spec(w, "s")
        keep_logs(w, "s", "x", ACTED, reps=4)
        keep_logs(w, "s", "x", "the discarded attempt never acted", reps=1, attempt=0)       # overwrites rep 0 attempt 0
        keep_logs(w, "s", "x", ACTED, reps=1, attempt=1)                                      # the attempt abtest used for rep 0
        v, r = verdict(w, item(spec, claim=["edit"]))
        assert f"switch {SWITCH} acted in 4 of 4 runs" in why(r) and v == "PROMOTE", why(r)
        keep_logs(w, "s", "x", "silent", reps=1, attempt=1)                                   # now the used attempt is the silent one
        assert verdict(w, item(spec, claim=["edit"]))[0] == "HOLD"


def test_exempt_switch_is_not_asked_for_a_line_but_is_listed():
    with world() as w:
        w.runs("s", GOOD)
        spec = switch_spec(w, "s", {"STRATA_QUIET": "1", "GPU_MAX_HW_QUEUES": "8", "STRATA_OFF_ONE": "0"})
        (w.root / "bench/promotion").mkdir(parents=True)
        (w.root / S.EXEMPT_FILE).write_text(json.dumps({"exempt": {"prints no line yet": ["STRATA_QUIET"]}}))
        v, r = verdict(w, item(spec, claim=["edit"]))
        assert v == "PROMOTE" and "switch not checked (prints no line yet): STRATA_QUIET" in why(r), why(r)


def test_item_can_exempt_its_own_switch_and_a_malformed_list_is_refused():
    with world() as w:
        w.runs("s", GOOD)
        spec = switch_spec(w, "s")
        v, r = verdict(w, item(spec, claim=["edit"], switch_exempt={"results predate the log": [SWITCH]}))
        assert v == "PROMOTE" and "switch not checked (results predate the log)" in why(r)
        try:
            P.evaluate(item(spec, switch_exempt={"reason": SWITCH}), w.root)
        except ValueError as e:
            assert "switch_exempt" in str(e)
        else:
            raise AssertionError("a string instead of a list of names was accepted")


def test_arm_without_env_or_with_only_off_values_has_nothing_to_verify():
    with world() as w:
        w.runs("s", GOOD)
        v, r = verdict(w, item(w.spec("s"), claim=["edit"]))                       # no env at all: the tests of promote.py
        assert v == "PROMOTE" and "switch" not in why(r)


# ------------------------------------------------------------------------------------------------ promote: the tier rule
def test_tier_of_reads_a_string_nothing_or_unknown():
    assert P.tier_of({"tier": "arena"}) == "arena" and P.tier_of({}) is None and P.tier_of({"tier": "unknown"}) is None
    assert P.tier_of({"tier": {"b": 2, "a": 1}}) == P.tier_of({"tier": {"a": 1, "b": 2}}) != P.tier_of({"tier": "arena"})


def test_arms_on_one_tier_are_judged_and_say_so():
    with world() as w:
        w.runs("s", GOOD)
        tag_tier(w, "s", {"base": "arena", "x": "arena"})
        v, r = verdict(w, item(w.spec("s"), claim=["edit"]))
        assert v == "PROMOTE" and "tier arena on every run" in why(r) and not r["arms"][0].warn, why(r)


def test_arms_on_different_tiers_are_held_and_not_judged():
    with world() as w:
        w.runs("s", GOOD)
        tag_tier(w, "s", {"base": "arena", "x": "mmap"})
        v, r = verdict(w, item(w.spec("s"), claim=["edit"]))
        assert v == "HOLD" and not r["arms"][0].reject, why(r)
        assert "tiers differ: base on arena, x on mmap" in why(r) and "repeat both arms on one tier" in why(r)
        assert "claim " not in why(r)


def test_the_vs_arm_counts_for_the_tier_rule():
    with world() as w:
        w.runs("s", {"y": {"short": [0.0] * 4, "edit": [1.0] * 4}, "x": {"short": [0.1, -0.1, 0.1, -0.1], "edit": [5.0, 5.4, 4.6, 5.0]}})
        spec = w.spec("s", arms=("base", "y", "x"))
        tag_tier(w, "s", {"base": "arena", "y": "mmap", "x": "arena"})                 # x and the base agree, the arm it is measured against not
        v, r = verdict(w, item(spec, claim=["edit"], arms=["x"], vs="y"))
        assert v == "HOLD" and "tiers differ: base on arena, x on arena, y on mmap" in why(r), why(r)


def test_one_run_on_another_tier_inside_an_arm_is_caught_too():
    with world() as w:
        w.runs("s", GOOD)
        path = w.out / "s.jsonl"
        rows = [json.loads(x) for x in path.read_text().splitlines()]
        path.write_text("\n".join(json.dumps({**r, "tier": "mmap" if (r["arm"], r["rep"]) == ("x", 3) else "arena"}) for r in rows) + "\n")
        v, r = verdict(w, item(w.spec("s"), claim=["edit"]))
        assert v == "HOLD" and "x on arena/mmap" in why(r), why(r)


def test_rows_without_a_tier_are_judged_and_warned_about_once_per_file():
    with world() as w:
        w.runs("s", GOOD)
        spec = w.spec("s", arms=("base", "x"))
        items = [item(spec, claim=["edit"], name="a"), item(spec, claim=["edit"], name="b")]
        results = [P.evaluate(i, w.root) for i in items]
        assert [r["verdict"] for r in results] == ["PROMOTE", "PROMOTE"]            # unknown is a warning, not a hold
        text = P.render(results)
        assert text.count("WARNING: s.jsonl: 8 of 8 runs carry no tier") == 1, text


def test_part_of_the_rows_without_a_tier_still_warns():
    with world() as w:
        w.runs("s", GOOD)
        tag_tier(w, "s", {"x": "arena"})                                           # the base rows are the old ones
        v, r = verdict(w, item(w.spec("s"), claim=["edit"]))
        assert v == "PROMOTE" and "WARNING: s.jsonl: 4 of 8 runs carry no tier" in P.render([r])


def test_main_prints_the_warning_and_void_and_tier_checks_never_write():
    with world() as w:
        w.runs("s", NO_GAIN)
        f = w.root / "items.json"
        f.write_text(json.dumps({"items": [item(switch_spec(w, "s"), claim=["edit"])]}))
        keep_logs(w, "s", "x", "silent")
        before = sorted(str(p) for p in w.root.rglob("*"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert P.main([str(f), "--root", str(w.root)]) == 0
        assert "0 PROMOTE, 1 HOLD, 0 REJECT" in out.getvalue() and "WARNING: s.jsonl" in out.getvalue()
        assert sorted(str(p) for p in w.root.rglob("*")) == before


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
