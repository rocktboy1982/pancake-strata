"""Tests of tools/pancake/ablate_greedy.py (MOD_PLAN 17.3).

Part 1 needs no engine at all (delta merging, plan checks, the bundle state machine, the dry run).
Part 2 runs the whole loop on the FAKE engine (tools/pancake/probes/fake_engine.py: real abtest.py processes against a fake repo root, no
GPU, no model): one made-up plan with 4 candidates (one winner, one loser, one substitute pair), then a plan with an exact and a lossy
candidate.  The fake numbers are synthetic: these tests check the loop and the bookkeeping, not any speed.

SAFETY: the fake run starts abtest processes and fake servers.  The tests skip while a real abtest / engine / queue runs on the machine
(fake_engine.real_run_active()), exactly as test_abtest_e2e.py does; ABLATE_FAKE_ALLOW_BUSY=1 runs them anyway (abtest --force-busy,
a free port, nice them: a few seconds of one core).

Run: nice -n 19 python3 -I -m pytest -q tools/pancake/test_ablate_greedy.py"""
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "probes"))
import ablate_greedy as A  # noqa: E402
import fake_engine as F  # noqa: E402


def cand(name, env=None, args=None, lossy=False, subs=(), gain=0.0, **kw):
    if env is None and args is None:
        env = {f"STRATA_FAKE_{name.upper().replace('-', '_')}": "1"}
    c = {"name": name, "env": env or {}, "args": args or {}, "lossy": lossy,
         "substitutes_of": list(subs), "fake": {"gain_pct": gain}}
    c.update(kw)
    return c


def plan_of(cands, **kw):
    p = {"name": "t", "base": "ignored-in-fake.json", "cases": ["short", "edit"], "reps": 6, "block": 2, "warmup": 1, "min_gain": 2.0,
         "max_rounds": 6, "candidates": cands, "fake_profile": {"sd_start_pct": 0.3, "sd_request_pct": 0.3}}
    p.update(kw)
    return A.check_plan(p)


# ------------------------------------------------------------------------------------------------ part 1: no engine
def test_merge_later_wins_and_none_removes():
    d = A.merge({"env": {"A": "1", "B": "2"}, "args": {"--spec": "4"}}, {"env": {"B": None, "C": 3}, "args": {"--spec": "5", "--x": ""}})
    assert d == {"env": {"A": "1", "B": None, "C": "3"}, "args": {"--spec": "5", "--x": ""}}
    assert A.bundle_delta({}, []) == {}


def test_spec_arms_are_base_plus_candidate_and_carry_the_bundle():
    p = plan_of([cand("a", env={"STRATA_A": "1"}), cand("b", args={"--spec": "5"}), cand("c", env={"STRATA_C": "1"})],
                bundle_start={"env": {"STRATA_START": "1"}})
    spec = A.build_spec(p, [p["candidates"][0]], p["candidates"][1:])
    assert list(spec["arms"]) == ["base", "b", "c"]                       # the reference arm first, as promote and abtest need
    assert spec["arms"]["base"] == {"env": {"STRATA_START": "1", "STRATA_A": "1"}}
    assert spec["arms"]["b"] == {"env": {"STRATA_START": "1", "STRATA_A": "1"}, "args": {"--spec": "5"}}
    assert spec["base"] == "ignored-in-fake.json" and spec["cases"] == ["short", "edit"] and spec["reps"] == 6


@pytest.mark.parametrize("bad, text", [
    ({"candidates": [cand("a"), cand("a")]}, "twice"),
    ({"candidates": [cand("base")]}, "not be 'base'"),
    ({"candidates": [cand("a", subs=["zzz"])]}, "not another candidate"),
    ({"candidates": [cand("a", env={"STRATA_X": "1"}), cand("b", env={"STRATA_X": "2"})]}, "silently overwrite"),
    ({"candidates": [{"name": "a", "env": {}, "args": {}}]}, "changes nothing"),
    ({"candidates": [cand("a")], "lossy_policy": "yes"}, "lossy_policy"),
    ({"candidates": [cand("a")], "claim": ["long"]}, "claim case"),
])
def test_plan_errors(bad, text):
    with pytest.raises(A.PlanError, match=text):
        plan_of(bad.pop("candidates"), **bad)


def test_same_key_is_fine_between_substitutes_or_with_one_value():
    plan_of([cand("a", env={"STRATA_X": "1"}), cand("b", env={"STRATA_X": "1"})])
    plan_of([cand("a", env={"STRATA_X": "1"}, subs=["b"]), cand("b", env={"STRATA_X": "2"})])


def test_pool_skips_disabled_and_applies_the_lossy_policy_and_filters():
    cs = [cand("a"), cand("b", lossy=True), cand("c", disabled="inert here"), cand("d", group="x")]
    p = plan_of(cs)
    assert [c["name"] for c in A.pool_of(p)] == ["a", "b", "d"]
    assert [c["name"] for c in A.pool_of(plan_of(cs, lossy_policy="skip"))] == ["a", "d"]
    assert [c["name"] for c in A.pool_of(p, only={"a", "d"}, exclude={"d"})] == ["a"]
    assert [c["name"] for c in A.pool_of(p, group="x")] == ["d"]


def res(verdict, lo, **kw):
    return {"verdict": verdict, "gain_lo": lo, "cases": {}, "reject": kw.get("reject", []), "hold": kw.get("hold", []), "lines": [],
            "void": kw.get("void", False)}


def test_state_accepts_the_best_lower_bound_drops_substitutes_benches_losers_and_retests_on_interaction():
    p = plan_of([cand("a"), cand("b", subs=["a"]), cand("c"), cand("d", interacts_with=["c"]), cand("e")])
    st = A.State(p, A.pool_of(p))
    tested = st.eligible()
    results = {"a": res("PROMOTE", 3.0), "b": res("PROMOTE", 5.0), "c": res("PROMOTE", 4.0), "d": res("REJECT", -1.0, reject=["no gain"]),
               "e": res("HOLD", 0.5, hold=["gain open on short"])}
    pick = st.choose(results, tested)
    assert pick["name"] == "b"                                           # highest lower bound among the clearing ones
    n_before = len(st.bundle)
    st.accept(pick, 1)
    st.note_round(1, tested, results, n_before)
    assert [c["name"] for c in st.bundle] == ["b"]
    assert st.out["a"]["state"] == "dropped" and "substitute of b" in st.out["a"]["why"]   # a declared b, but substitution is symmetric
    assert st.out["d"]["state"] == "benched" and "e" not in st.out        # undecided stays in the pool
    assert [c["name"] for c in st.eligible()] == ["c", "e"]               # d is benched, a is gone, b is in the bundle
    pick = st.choose({"c": res("PROMOTE", 4.0), "e": res("HOLD", 1.0, hold=["x"])}, st.eligible())
    n_before = len(st.bundle)
    st.accept(pick, 2)                                                    # c accepted: d interacts with c -> tested again
    assert {c["name"] for c in st.eligible()} == {"d", "e"}


def test_exact_first_prefers_an_exact_candidate_even_with_a_lower_bound_below_a_lossy_one():
    p = plan_of([cand("ex"), cand("lo", lossy=True)])
    st = A.State(p, A.pool_of(p))
    results = {"ex": res("PROMOTE", 2.5), "lo": res("PROMOTE", 9.0)}
    assert st.choose(results, st.eligible())["name"] == "ex"
    st.plan["lossy_policy"] = "mixed"
    assert st.choose(results, st.eligible())["name"] == "lo"


def test_void_or_regressing_arms_are_benched():
    assert A.regressed(res("HOLD", 0.0, void=True))
    assert A.regressed(res("HOLD", 0.0, hold=["regression on short: -4.0%"]))
    assert not A.regressed(res("HOLD", 0.0, hold=["gain open on short"]))


def test_dry_run_prints_plan_names_spec_and_order_and_starts_nothing(capsys):
    plan = {"name": "t", "base": "x.json", "cases": ["short"], "reps": 6, "block": 2, "warmup": 1,
            "candidates": [cand("a", env={"STRATA_COMBINE_POLL1": "1"}), cand("b", args={"--spec": "5"}), cand("c", env={"STRATA_NO_SUCH_SWITCH_XYZ": "1"}),
                           cand("d", disabled="inert")]}
    p = A.check_plan(plan)
    assert A.dry_run(p, A.pool_of(p), A.ROOT) == 0
    out = capsys.readouterr().out
    assert "round 1 spec" in out and "round 1 run order (12 engine starts, 2 pass(es) per start)" in out   # 4 arms x max(2, ceil(6 / 2)) = 3 starts
    assert "FOUND" in out and "MISSING env  STRATA_NO_SUCH_SWITCH_XYZ" in out and "DISABLED" in out
    assert "r0:base" in out


def test_starts_per_arm_follows_abtest_schedule():
    assert A.starts_per_arm(plan_of([cand("a")], reps=9, block=3)) == 3
    assert A.starts_per_arm(plan_of([cand("a")], reps=6, block=3)) == 2
    assert A.starts_per_arm(plan_of([cand("a")], tier="screen")) == 1


# ------------------------------------------------------------------------------------------------ part 2: fake engine
@pytest.fixture(scope="module", autouse=False)
def fake_ok():
    busy = F.real_run_active()
    if busy and not os.environ.get("ABLATE_FAKE_ALLOW_BUSY"):
        pytest.skip(f"a real abtest / engine / queue is running ({busy}): set ABLATE_FAKE_ALLOW_BUSY=1 to run the fake loop beside it")
    return bool(busy)


def run_fake(tmp_path, plan, extra=()):
    pf = tmp_path / "plan.json"
    pf.write_text(json.dumps(plan))
    out = tmp_path / "out"
    flags = ["--fake", "--out-dir", str(out), *extra]
    if os.environ.get("ABLATE_FAKE_ALLOW_BUSY") and F.real_run_active():
        flags.append("--force-busy")
    rc = A.main([str(pf), *flags])
    return rc, out


def test_fake_loop_picks_the_winner_then_the_better_of_a_substitute_pair_and_leaves_the_loser(tmp_path, fake_ok):
    cands = [cand("winner", gain=15.0), cand("loser", gain=-3.0), cand("pair-a", gain=9.0, subs=["pair-b"]), cand("pair-b", gain=4.0, subs=["pair-a"])]
    rc, out = run_fake(tmp_path, {"name": "t4", "base": "x.json", "cases": ["short", "edit"], "reps": 6, "block": 2, "warmup": 1, "min_gain": 2.0,
                                  "max_rounds": 6, "candidates": cands, "fake_profile": {"sd_start_pct": 0.3, "sd_request_pct": 0.3}})
    assert rc == 0
    state = json.loads((out / "state.json").read_text())
    rounds = [r for r in state["rounds"] if "results" in r]
    assert [r["accepted"] for r in rounds] == ["winner", "pair-a"]                        # the bundle, in order
    assert rounds[0]["tested"] == ["winner", "loser", "pair-a", "pair-b"]
    assert rounds[1]["tested"] == ["pair-a", "pair-b"]                                    # the loser was benched, the winner is in the base
    assert rounds[0]["results"]["winner"]["verdict"] == "PROMOTE"
    assert rounds[0]["results"]["loser"]["verdict"] != "PROMOTE"
    assert [d["name"] for d in rounds[1]["dropped"]] == ["pair-b"]                        # substitute dropped, noted, never run again
    assert state["status"] == "pool empty"
    spec2 = json.loads((out / "round2.spec.json").read_text())
    assert list(spec2["arms"]) == ["base", "pair-a", "pair-b"]
    assert "STRATA_FAKE_WINNER" in spec2["arms"]["base"]["env"]                          # the base arm of round 2 IS the round-1 bundle
    assert "STRATA_FAKE_PAIR_A" in spec2["arms"]["pair-a"]["env"] and "STRATA_FAKE_PAIR_A" not in spec2["arms"]["base"]["env"]
    gains = rounds[0]["results"]["winner"]["cases"]
    assert gains["short"]["lo"] > 2.0 and gains["edit"]["n"] >= 3                         # paired gain with its interval and pair count
    md = (out / "RESULTS.md").read_text()
    assert "FAKE ENGINE" in md and "dropped without a test: pair-b" in md and "## Interactions" in md and "pair-a vs pair-b" in md
    assert "| loser |" in md and "benched" in md
    delta = json.loads((out / "final-delta.json").read_text())
    assert delta["bundle"] == ["winner", "pair-a"]
    preset = json.loads((out / "final-preset.json").read_text())                          # the fake base preset with the bundle applied
    assert preset["env"]["STRATA_FAKE_WINNER"] == "1" and preset["env"]["STRATA_FAKE_PAIR_A"] == "1" and "STRATA_FAKE_PAIR_B" not in preset["env"]
    assert state["final"]["results"]["bundle"]["cases"]["short"]["gain"] > 10.0           # the stack against the start preset (synthetic +24% there)
    assert not (out / "quality-gate-spec.json").exists()                                   # nothing lossy was accepted


def test_fake_lossy_item_is_flagged_and_routed_to_the_quality_gate_not_promoted(tmp_path, fake_ok):
    cands = [cand("exact-one", gain=5.0), cand("lossy-one", gain=8.0, lossy=True)]
    rc, out = run_fake(tmp_path, {"name": "tl", "base": "x.json", "cases": ["short"], "reps": 6, "block": 2, "warmup": 1, "min_gain": 2.0,
                                  "min_gain_lossy": 3.0, "max_rounds": 4, "candidates": cands,
                                  "fake_profile": {"sd_start_pct": 0.3, "sd_request_pct": 0.3}})
    assert rc == 0
    state = json.loads((out / "state.json").read_text())
    rounds = [r for r in state["rounds"] if "results" in r]
    assert [r["accepted"] for r in rounds] == ["exact-one", "lossy-one"]                   # exact-first, although the lossy one is faster
    delta = json.loads((out / "final-delta.json").read_text())
    assert "STRATA_FAKE_LOSSY_ONE" in delta["delta"]["env"] and "STRATA_FAKE_LOSSY_ONE" not in delta["exact_only_delta"]["env"]
    qg = json.loads((out / "quality-gate-spec.json").read_text())
    assert list(qg["arms"]) == ["base", "lossy-one"] and "STRATA_FAKE_EXACT_ONE" in qg["arms"]["base"]["env"]
    exact_preset = json.loads((out / "final-preset-exact.json").read_text())
    assert "STRATA_FAKE_LOSSY_ONE" not in exact_preset["env"] and "STRATA_FAKE_EXACT_ONE" in exact_preset["env"]
    assert list(state["final"]["results"]) == ["bundle", "exact-only"]
    md = (out / "RESULTS.md").read_text()
    assert "quality-gate queue" in md and "NOT promoted by this tool" in md and "lossy-one" in md


def test_fake_run_refuses_to_reuse_a_started_out_dir_without_resume(tmp_path, fake_ok):
    plan = {"name": "tr", "base": "x.json", "cases": ["short"], "reps": 4, "block": 2, "warmup": 0, "candidates": [cand("only", gain=6.0)],
            "fake_profile": {"sd_start_pct": 0.3, "sd_request_pct": 0.3}, "final_check": False}
    rc, out = run_fake(tmp_path, plan, ["--max-rounds", "1"])
    assert rc == 0 and (out / "state.json").exists()
    assert A.main([str(tmp_path / "plan.json"), "--fake", "--out-dir", str(out)]) == 2     # state.json is there: --resume or another dir
