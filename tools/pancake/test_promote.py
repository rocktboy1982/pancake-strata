"""test_promote.py - promote.py on synthetic result directories (no GPU, no model, no git): the statistics, every verdict, the
parity and gate readers, the cherry-pick lines, and the promise that the tool only reads.

    python tools/pancake/test_promote.py      (or: python -m pytest tools/pancake/test_promote.py)
"""
import ast
import contextlib
import io
import json
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import promote as P  # noqa: E402

DAY = "2026-11-12-test-phase"
CASES = ("short", "edit")


class World:
    """A repo root in a temp dir: bench/specs and bench/results/<DAY> with the files the test tools write."""

    def __init__(self, root):
        self.root = Path(root)
        self.out = self.root / "bench/results" / DAY
        self.out.mkdir(parents=True)
        (self.root / "bench/specs").mkdir(parents=True)

    def spec(self, stem, arms=("base", "x"), cases=CASES, **extra):
        (self.root / f"bench/specs/{stem}.json").write_text(json.dumps({"base": "p.json", "arms": {a: {} for a in arms}, "cases": list(cases), **extra}))
        return f"bench/specs/{stem}.json"

    def runs(self, stem, diffs, texts=None, noisy=(), reps=None, base_value=100.0, ref="base"):
        """abtest jsonl.  diffs {arm: {case: [gain in % of base per rep]}}; the reference arm (ref) is added at base_value in every rep, so
        a gain of d % is a paired difference of exactly d.  noisy: (arm, rep) pairs written with clean false.  texts {(arm, case): hash}."""
        arms = [ref] + [a for a in diffs if a != ref]
        n = reps or len(next(iter(next(iter(diffs.values())).values())))
        lines = []
        for rep in range(n):
            for arm in arms:
                rows = [{"case": c, "decode_tps": base_value + (0 if arm == ref else diffs[arm][c][rep]), "prompt_tps": 1.0, "prompt": 1,
                         "gen": 1, "accept": None} for c in CASES]
                t = {c: (texts or {}).get((arm, c), f"text-{c}") for c in CASES}
                lines.append(json.dumps({"t": "x", "rep": rep, "arm": arm, "attempt": 0, "clean": (arm, rep) not in noisy, "rows": rows, "texts": t,
                                         "noise": 80.0 if (arm, rep) in noisy else 1.0, "reclaim": 0.0}))
        (self.out / f"{stem}.jsonl").write_text("\n".join(lines) + "\n")

    def quality(self, stem, verdicts, stock=(1.8, 1.9)):
        d = self.out / f"{stem}-quality"
        d.mkdir(exist_ok=True)
        (d / "verdicts.json").write_text(json.dumps(verdicts))
        for arm in verdicts:
            (d / f"{arm}.verdict.json").write_text(json.dumps({
                "requests": 20, "tokens_cand": 5322, "mismatch_rate_stock_arms_pct": list(stock), "mismatch_rate_cand_pct": 1.9, "diff_pts": 0.05,
                "diff_ci90_pts": [-0.2, 0.3], "events_cand": 0, "events_stock_arms": [0, 0]}))

    def taskgate(self, stem, arms, warnings=()):
        d = self.out / f"{stem}-taskgate"
        d.mkdir(exist_ok=True)
        (d / "summary.json").write_text(json.dumps({"margin": 0.05, "warnings": list(warnings), "arms": arms}))

    def replay(self, stem, runs):
        d = self.out / f"{stem}-replay"
        d.mkdir(exist_ok=True)
        arms = list(runs)
        (d / "summary.json").write_text(json.dumps({"arms": arms, "reps": len(runs[arms[0]]), "runs": {
            a: [{"rep": r, "arm": a, "clean": True, "totals": {"prompt_s": v}} for r, v in enumerate(vs)] for a, vs in runs.items()}}))

    def file(self, name, text):
        (self.out / name).write_text(text)


@contextlib.contextmanager
def world():
    with tempfile.TemporaryDirectory() as d:
        yield World(d)


def item(spec, **kw):
    return {"name": "it", "spec": spec, "exact": True, "branch": "dev/p16/test", "commits": ["abc1234", "def5678"], **kw}


def verdict(w, it):
    r = P.evaluate(it, w.root)
    return r["verdict"], r


def why(r):
    return " | ".join(x for a in r["arms"] for x in a.reject + a.hold + a.lines) + " | ".join(r.get("why", []))


GOOD = {"x": {"short": [0.1, -0.1, 0.1, -0.1], "edit": [5.0, 5.4, 4.6, 5.0]}}      # +5% on edit, short unchanged


# ------------------------------------------------------------------------------------------------ statistics
def test_t_quantile_matches_the_tables():
    for p, df, want in [(0.975, 1, 12.7062), (0.975, 3, 3.1824), (0.975, 10, 2.2281), (0.975, 30, 2.0423), (0.9875, 3, 4.1765), (0.995, 5, 4.0321)]:
        assert abs(P.t_quantile(p, df) - want) < 5e-4, (p, df)


def test_gain_ci_by_hand():
    g = P.gain_ci([(100.0, 102.0), (100.0, 108.0), (100.0, 100.0), (100.0, 106.0)])        # diffs 2, 8, 0, 6
    assert abs(g["gain"] - 4.0) < 1e-9 and abs(g["sd"] - 3.6515) < 1e-3
    assert abs(g["lo"] - (4.0 - 3.1824 * 3.6515 / 2)) < 2e-3 and abs(g["hi"] - (4.0 + 3.1824 * 3.6515 / 2)) < 2e-3
    lower = P.gain_ci([(100.0, 90.0), (100.0, 91.0), (100.0, 89.0)], lower_is_better=True)  # a smaller number is the gain
    assert abs(lower["gain"] - 10.0) < 1e-9
    assert P.gain_ci([(100.0, 101.0)])["lo"] == -float("inf")                             # one pair says nothing


def test_pairs_needed_is_the_smallest_n():
    n = P.pairs_needed(4.0, 1.0)
    assert P.t_quantile(0.975, n - 1) * 4.0 / n ** 0.5 <= 1.0 < P.t_quantile(0.975, n - 2) * 4.0 / (n - 1) ** 0.5
    assert P.pairs_needed(40.0, 0.1) is None


def test_reader_keeps_what_abtest_kept():
    def rec(rep, attempt, clean, noise):
        return json.dumps({"rep": rep, "arm": "a", "attempt": attempt, "clean": clean, "noise": noise, "rows": []})
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "r.jsonl"
        f.write_text("\n".join([rec(0, 0, False, 60), rec(0, 1, True, 3), rec(0, 2, True, 1),       # the first clean attempt wins
                                rec(1, 0, False, 90), rec(1, 1, False, 40), rec(1, 2, False, 70)]) + "\n")   # all noisy: the quietest
        runs, discarded, problems = P.read_runs(f)
        assert runs["a"][0]["attempt"] == 1 and runs["a"][1]["attempt"] == 1 and discarded == 4 and not problems
        f.write_text(f.read_text() + rec(0, 0, False, 60) + "\n")
        assert "appears twice" in P.read_runs(f)[2][0]


def test_reader_agrees_with_abtest_summary_on_a_real_result():
    d = P.ROOT / "bench/results/2026-11-10-test-phase"
    if not (d / "05c-lookup12.jsonl").is_file():
        return
    runs, _, _ = P.read_runs(d / "05c-lookup12.jsonl")
    summary = (d / "05c-lookup12.summary.txt").read_text()
    for case in ("short", "edit"):
        want = re.search(rf"{case}\s+lookup12 .*vs base:\s+([+-][\d.]+)% \+-\s+([\d.]+)%", summary)
        g = P.gain_ci(P.case_pairs(runs["base"], runs["lookup12"], case))
        assert abs(g["gain"] - float(want[1])) < 0.1 and abs((g["hi"] - g["lo"]) / 2 - float(want[2])) < 0.15, (case, g, want[0])


# ------------------------------------------------------------------------------------------------ verdicts, exact items
def test_exact_item_with_a_clear_gain_is_promoted_and_prints_the_cherry_pick():
    with world() as w:
        w.runs("s", GOOD)
        v, r = verdict(w, item(w.spec("s"), claim=["edit"]))
        assert v == "PROMOTE", why(r)
        text = P.render([r])
        assert "git cherry-pick -x abc1234 def5678" in text and "port/rdna3-decode" in text and "PROMOTE" in text
        assert "parity text identical" in text


def test_gain_that_the_noise_cannot_decide_is_held_with_the_pairs_needed():
    with world() as w:
        w.runs("s", {"x": {"short": [0.0] * 4, "edit": [2.0, 8.0, 0.0, 6.0]}})
        v, r = verdict(w, item(w.spec("s"), claim=["edit"]))
        assert v == "HOLD" and f"about {P.pairs_needed(3.6515, 3.0)} pairs" in why(r), why(r)


def test_no_gain_worth_having_is_rejected():
    with world() as w:
        w.runs("s", {"x": {"short": [0.0] * 4, "edit": [0.2, 0.3, 0.1, 0.2]}})
        v, r = verdict(w, item(w.spec("s"), claim=["edit"]))
        assert v == "REJECT" and "no gain worth having" in why(r)


def test_exact_item_whose_text_differs_is_rejected_and_a_lossy_one_is_not_asked():
    with world() as w:
        w.runs("s", GOOD, texts={("x", "edit"): "other"})
        v, r = verdict(w, item(w.spec("s"), claim=["edit"]))
        assert v == "REJECT" and "declared exact but the text differs" in why(r)
        w.quality("s", {"x": "PASS"})
        w.taskgate("s", {"x": {"verdict": "PASS", "success_diff": {"diff": 0, "lo": -0.02, "hi": 0.02, "tasks": 18}}})
        assert verdict(w, item(w.spec("s"), claim=["edit"], exact=False))[0] == "PROMOTE"


def test_fewer_than_three_pairs_never_promote():
    with world() as w:
        w.runs("s", {"x": {"short": [0.0, 0.0], "edit": [5.0, 5.0]}})
        v, r = verdict(w, item(w.spec("s"), claim=["edit"]))
        assert v == "HOLD" and "fewer than 3 clean pairs" in why(r)
        assert verdict(w, item(w.spec("s"), claim=["edit"], min_pairs=2))[0] == "PROMOTE"      # the item may lower it; the report says so
        assert "item override: min_pairs" in P.render([verdict(w, item(w.spec("s"), claim=["edit"], min_pairs=2))[1]])


def test_noisy_runs_hold_a_promotion():
    with world() as w:
        w.runs("s", GOOD, noisy=[("base", 1)])
        v, r = verdict(w, item(w.spec("s"), claim=["edit"]))
        assert v == "HOLD" and "noisy" in why(r) and "quiet machine" in why(r)


def test_regression_on_another_case_holds_a_gain():
    with world() as w:
        w.runs("s", {"x": {"short": [-8.0, -8.5, -7.5, -8.0], "edit": [5.0, 5.4, 4.6, 5.0]}})
        v, r = verdict(w, item(w.spec("s"), claim=["edit"]))
        assert v == "HOLD" and "regression on short" in why(r)
        w.runs("s", {"x": {"short": [-3.2, -6.2, -0.2, -3.2], "edit": [5.0, 5.4, 4.6, 5.0]}})   # not certain, but the point estimate is -3.2%
        v, r = verdict(w, item(w.spec("s"), claim=["edit"]))
        assert v == "HOLD" and "regression on short" in why(r)


def test_claim_declared_in_advance_avoids_the_multiple_comparison_penalty():
    data = {"x": {"short": [0.0] * 4, "edit": [2.7, 1.5, 2.3, 1.5]}}                           # +2.0% +- 0.95 at 95%, +- 1.25 at 97.5% each
    with world() as w:
        w.runs("s", data)
        assert verdict(w, item(w.spec("s"), claim=["edit"]))[0] == "PROMOTE"
        v, r = verdict(w, item(w.spec("s")))                                                   # claim = both cases: Bonferroni over 2
        assert v == "HOLD" and "gain open on edit" in why(r)


def test_stale_results_and_missing_results_and_appended_reruns_hold():
    with world() as w:
        w.runs("s", GOOD)
        w.file("rev.txt", "abc1234\n")
        assert verdict(w, item(w.spec("s"), claim=["edit"], tested_rev="abc1234def"))[0] == "PROMOTE"
        v, r = verdict(w, item(w.spec("s"), claim=["edit"], tested_rev="9999999"))
        assert v == "HOLD" and "stale" in why(r)
        f = w.out / "s.jsonl"
        f.write_text(f.read_text() + f.read_text().splitlines()[0] + "\n")
        v, r = verdict(w, item(w.spec("s"), claim=["edit"]))
        assert v == "HOLD" and "appears twice" in why(r)
        v, r = verdict(w, item(w.spec("nothing"), claim=["edit"]))
        assert v == "HOLD" and "not run" in why(r)


def test_vs_names_the_arm_the_gain_is_measured_against():
    with world() as w:
        w.spec("s", arms=("base", "mid", "x"))
        w.runs("s", {"mid": {"short": [0.0] * 4, "edit": [5.0, 5.0, 5.0, 5.0]}, "x": {"short": [0.0] * 4, "edit": [5.2, 5.0, 4.8, 5.0]}})
        assert verdict(w, item("bench/specs/s.json", arms=["x"], claim=["edit"]))[0] == "PROMOTE"            # against base: +5%
        assert verdict(w, item("bench/specs/s.json", arms=["x"], claim=["edit"], vs="mid"))[0] == "REJECT"                                                                                 # on top of mid: +0%


def test_best_arm_decides_the_item():
    with world() as w:
        w.spec("s", arms=("base", "weak", "x"))
        w.runs("s", {"weak": {"short": [0.0] * 4, "edit": [0.2, 0.3, 0.1, 0.2]}, "x": GOOD["x"]})
        v, r = verdict(w, item("bench/specs/s.json", claim=["edit"]))
        assert v == "PROMOTE" and r["best"].name == "x" and [a.verdict() for a in r["arms"]] == ["REJECT", "PROMOTE"]
        assert 'winning arm x: {}' in P.render([r])


# ------------------------------------------------------------------------------------------------ lossy items and their gates
def lossy_world(w, diffs=None):
    w.runs("s", diffs or {"x": {"short": [0.1, -0.1, 0.1, -0.1], "edit": [8.0, 8.4, 7.6, 8.0]}})
    return item(w.spec("s"), exact=False, claim=["edit"])


def test_lossy_item_needs_both_gates():
    with world() as w:
        it = lossy_world(w)
        v, r = verdict(w, it)
        assert v == "HOLD" and "token forcing not run" in why(r) and "task gate not run" in why(r)
        w.quality("s", {"x": "PASS"})
        w.taskgate("s", {"x": {"verdict": "BETTER", "success_diff": {"diff": 0.05, "lo": 0.01, "hi": 0.09, "tasks": 18}, "validity_verdict": "PASS"}})
        v, r = verdict(w, it)
        assert v == "PROMOTE", why(r)
        assert "token forcing PASS" in why(r) and "task gate BETTER" in why(r)


def test_lossy_threshold_is_higher_than_the_exact_one():
    data = {"x": {"short": [0.0] * 4, "edit": [2.1, 2.0, 1.9, 2.0]}}                            # +2.0%: enough for an exact item, not a lossy one
    with world() as w:
        w.runs("s", data)
        assert verdict(w, item(w.spec("s"), claim=["edit"]))[0] == "PROMOTE"
        v, r = verdict(w, item(w.spec("s"), claim=["edit"], exact=False))
        assert v == "REJECT" and "below 3%" in why(r)


def test_token_forcing_verdicts():
    with world() as w:
        it = lossy_world(w)
        w.taskgate("s", {"x": {"verdict": "PASS", "success_diff": {"diff": 0, "lo": -0.02, "hi": 0.02, "tasks": 18}}})
        w.quality("s", {"x": "REJECT"})
        v, r = verdict(w, it)
        assert v == "REJECT" and "token forcing REJECT" in why(r)


def test_token_forcing_inconclusive_unscored_and_noisy_stock_hold():
    for case in ("INCONCLUSIVE", "not scored"):
        with world() as w:
            it = lossy_world(w)
            w.taskgate("s", {"x": {"verdict": "PASS", "success_diff": {"diff": 0, "lo": -0.02, "hi": 0.02, "tasks": 18}}})
            w.quality("s", {"x": case})
            v, r = verdict(w, it)
            assert v == "HOLD" and f"token forcing {case}" in why(r)
            if case == "INCONCLUSIVE":
                assert "4x the requests" in why(r)
    with world() as w:
        it = lossy_world(w)
        w.taskgate("s", {"x": {"verdict": "PASS", "success_diff": {"diff": 0, "lo": -0.02, "hi": 0.02, "tasks": 18}}})
        w.quality("s", {"x": "PASS"}, stock=(1.2, 2.1))                                        # the two stock runs differ by 0.9 pts > 0.5
        v, r = verdict(w, it)
        assert v == "HOLD" and "stock runs differ" in why(r)


def test_task_gate_verdicts():
    bad = {"verdict": "REJECT", "success_diff": {"diff": -0.2, "lo": -0.3, "hi": -0.1, "tasks": 18}}
    unsure = {"verdict": "INCONCLUSIVE", "success_diff": {"diff": -0.05, "lo": -0.15, "hi": 0.05, "tasks": 18}}
    ok = {"verdict": "PASS", "success_diff": {"diff": 0, "lo": -0.02, "hi": 0.02, "tasks": 18}}
    for e, want in ((bad, "REJECT"), (unsure, "HOLD"), (ok, "PROMOTE")):
        with world() as w:
            it = lossy_world(w)
            w.quality("s", {"x": "PASS"})
            w.taskgate("s", {"x": e})
            assert verdict(w, it)[0] == want
    with world() as w:
        it = lossy_world(w)
        w.quality("s", {"x": "PASS"})
        w.taskgate("s", {"x": ok}, warnings=["the two base runs differ (+9.0 pts [+1.0, +17.0]): the server or the machine is not stable"])
        v, r = verdict(w, it)
        assert v == "HOLD" and "base runs differ" in why(r)


def test_gates_can_be_narrowed_by_the_item():
    with world() as w:
        it = lossy_world(w)
        w.quality("s", {"x": "PASS"})
        assert verdict(w, it)[0] == "HOLD"                                                    # the task gate is missing
        assert verdict(w, {**it, "gates": ["token_forcing"]})[0] == "PROMOTE"


# ------------------------------------------------------------------------------------------------ parity
def test_restore_check_parity():
    with world() as w:
        w.runs("s", GOOD)
        it = item(w.spec("s"), claim=["edit"], parity=["text", {"restore": "restore-check.txt"}])
        assert verdict(w, it)[0] == "HOLD" and "not found" in why(verdict(w, it)[1])
        w.file("restore-check.txt", "restore first: MATCH\nreread first: MATCH\nrestore follow-up: MATCH\nrestore check exit 0\n")
        assert verdict(w, it)[0] == "PROMOTE"
        w.file("restore-check.txt", "restore first: MATCH\nrestore follow-up: DIFFERENT\nrestore check exit 1\n")
        v, r = verdict(w, it)
        assert v == "REJECT" and "DIFFERENT" in why(r)
        w.file("restore-check.txt", "restore check exit 2\n")
        assert verdict(w, it)[0] == "HOLD"


def test_ctest_parity():
    out = ("  1/3 Test  #1: native_multi_parity ..........   Passed    0.40 sec\n  2/3 Test  #2: iq_multi_parity ...........***Failed    0.10 sec\n"
           "ctest -L gpu exit 8\n")
    with world() as w:
        w.runs("s", GOOD)
        w.file("ctest-gpu.txt", out)
        def mk(tests):
            return item(w.spec("s"), claim=["edit"], parity=["text", {"ctest": "ctest-gpu.txt", "tests": tests}])
        assert verdict(w, mk(["native_multi_parity"]))[0] == "PROMOTE"
        v, r = verdict(w, mk(["native_multi_parity", "iq_multi_parity"]))
        assert v == "REJECT" and "failed iq_multi_parity" in why(r)
        v, r = verdict(w, mk(["no_such_test"]))
        assert v == "HOLD" and "no_such_test" in why(r)
        v, r = verdict(w, item(w.spec("s"), claim=["edit"], parity=["text", {"ctest": "ctest-gpu.txt"}]))          # no names: every test counts
        assert v == "REJECT" and "failed iq_multi_parity" in why(r)
        w.file("ctest-gpu.txt", "ctest -L gpu exit 8\n")                                                         # nothing listed: the exit line decides
        v, r = verdict(w, item(w.spec("s"), claim=["edit"], parity=["text", {"ctest": "ctest-gpu.txt"}]))
        assert v == "REJECT" and "exit 8" in why(r)
        w.file("ctest-gpu.txt", "ctest -L gpu exit 0\n")
        assert verdict(w, item(w.spec("s"), claim=["edit"], parity=["text", {"ctest": "ctest-gpu.txt"}]))[0] == "PROMOTE"


def test_sampled_spec_has_no_text_parity_unless_waived():
    with world() as w:
        w.runs("s", GOOD)
        w.spec("s", temperature=0.1)
        v, r = verdict(w, item("bench/specs/s.json", claim=["edit"]))
        assert v == "HOLD" and "sampled spec" in why(r)
        v, r = verdict(w, item("bench/specs/s.json", claim=["edit"], parity=[{"waived": "exact in distribution"}]))
        assert v == "PROMOTE" and "parity waived by the item: exact in distribution" in why(r)


def test_unstable_base_text_holds():
    with world() as w:
        w.runs("s", GOOD)
        f = w.out / "s.jsonl"
        recs = [json.loads(x) for x in f.read_text().splitlines()]
        next(r for r in recs if r["arm"] == "base" and r["rep"] == 2)["texts"]["edit"] = "changed"
        f.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        v, r = verdict(w, item(w.spec("s"), claim=["edit"]))
        assert v == "HOLD" and "not the same on every run" in why(r)


# ------------------------------------------------------------------------------------------------ replay and task gate items
def test_replay_item():
    with world() as w:
        w.spec("s", arms=("flags", "x"))
        w.runs("s", {"x": {"short": [0.0] * 3, "edit": [0.0] * 3}}, reps=3, ref="flags")
        w.file("restore-check.txt", "restore first: MATCH\nrestore check exit 0\n")
        w.replay("s", {"flags": [100.0, 102.0, 98.0], "x": [60.0, 61.0, 59.0]})
        it = item("bench/specs/s.json", arms=["x"], decides="replay", parity=["text", {"restore": "restore-check.txt"}])
        v, r = verdict(w, it)
        assert v == "PROMOTE", why(r)
        assert "replay prompt s" in why(r) and "side   edit" in why(r) and "parity restore 1 turns MATCH" in why(r)
        w.replay("s", {"flags": [100.0, 102.0, 98.0], "x": [100.2, 101.9, 98.1]})              # the arm reads as long as the reference
        assert verdict(w, it)[0] == "REJECT"


def test_replay_side_effect_regression_holds():
    with world() as w:
        w.spec("s", arms=("base", "x"))
        w.runs("s", {"x": {"short": [-6.0, -6.2, -5.8, -6.0], "edit": [0.0] * 4}})
        w.replay("s", {"base": [100.0, 102.0, 98.0], "x": [60.0, 61.0, 59.0]})
        v, r = verdict(w, item("bench/specs/s.json", arms=["x"], decides="replay", parity=[{"waived": "n/a"}]))
        assert v == "HOLD" and "regression on short (abtest side effect check)" in why(r)


def test_taskgate_item_judges_time_to_success():
    def summary(lo, hi, verdict_):
        return {"x": {"verdict": verdict_, "success_diff": {"diff": 0, "lo": -0.02, "hi": 0.02, "tasks": 18},
                      "auc_diff": {"diff": (lo + hi) / 2, "lo": lo, "hi": hi}}}
    for lo, hi, v_, want in ((0.04, 0.08, "PASS", "PROMOTE"), (0.01, 0.10, "PASS", "HOLD"), (0.04, 0.08, "REJECT", "REJECT"), (-0.03, 0.01, "PASS", "REJECT")):
        with world() as w:
            w.spec("tg", arms=("medium", "x"), cases=())
            w.taskgate("tg", summary(lo, hi, v_))
            v, r = verdict(w, item("bench/specs/tg.json", exact=False, arms=["x"], decides="taskgate"))
            assert v == want, (lo, hi, v_, why(r))
    with world() as w:
        w.spec("tg", arms=("medium", "x"), cases=())
        w.taskgate("tg", {"x": {"verdict": "PASS", "success_diff": {"diff": 0, "lo": -0.02, "hi": 0.02, "tasks": 18}}})
        v, r = verdict(w, item("bench/specs/tg.json", exact=False, arms=["x"], decides="taskgate"))
        assert v == "HOLD" and "no time-to-success curve" in why(r)


# ------------------------------------------------------------------------------------------------ items file, commands, safety
def test_cherry_pick_lines_only_for_promote_in_listed_order():
    with world() as w:
        w.runs("s", GOOD)
        good = item(w.spec("s"), claim=["edit"], name="good", commits=["1111111", "2222222", "3333333"], note="check it")
        bad = item("bench/specs/s.json", claim=["edit"], name="bad", exact=False, commits=["4444444"])
        none = item("bench/specs/s.json", claim=["edit"], name="none", commits=[])
        text = P.render([P.evaluate(i, w.root) for i in (good, bad, none)])
        assert "git cherry-pick -x 1111111 2222222 3333333" in text and "4444444" not in text and "# good (arm x): check it" in text
        assert "# no commits listed" in text and text.count("git cherry-pick") == 1 and "2 PROMOTE, 1 HOLD, 0 REJECT" in text


def test_malformed_items_are_refused_before_anything_is_printed():
    with world() as w:
        w.runs("s", GOOD)
        spec = w.spec("s")
        for bad in (item(spec, commits=["abc1234; rm -rf ~"]), item(spec, branch="x y"), item(spec, exact="yes"), item(spec, arms=["nope"]),
                    item(spec, claim=["long"]), item(spec, vs="nope"), {"name": "n", "spec": spec},
                    item(w.spec("only-base", arms=("base",)))):
            try:
                P.evaluate(bad, w.root)
            except ValueError:
                continue
            raise AssertionError(f"accepted {bad}")
        f = w.root / "items.json"
        f.write_text(json.dumps({"items": [item(spec, commits=["$(reboot)"])]}))
        err, out = io.StringIO(), io.StringIO()
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            rc = P.main([str(f), "--root", str(w.root)])
        assert rc == 2 and out.getvalue() == "" and "hex hashes" in err.getvalue()


def test_main_end_to_end_with_only_and_without_writing_anything():
    with world() as w:
        w.runs("s", GOOD)
        w.runs("t", {"x": {"short": [0.0] * 4, "edit": [0.2, 0.3, 0.1, 0.2]}})
        f = w.root / "items.json"
        f.write_text(json.dumps({"items": [item(w.spec("s"), claim=["edit"], name="a"), item(w.spec("t"), claim=["edit"], name="b")]}))
        before = sorted(str(p) for p in w.root.rglob("*"))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            assert P.main([str(f), "--root", str(w.root)]) == 0
        assert "1 PROMOTE, 0 HOLD, 1 REJECT" in out.getvalue()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            P.main([str(f), "--root", str(w.root), "--only", "b"])
        assert "REJECT   b" in out.getvalue() and "a  [" not in out.getvalue()
        assert sorted(str(p) for p in w.root.rglob("*")) == before                              # it only reads


def test_the_tool_cannot_run_git_or_write():
    tree = ast.parse(Path(P.__file__).read_text())
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not imported & {"subprocess", "os", "shutil", "pty"}, imported
    calls = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "") for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert not calls & {"system", "popen", "Popen", "run", "write_text", "write_bytes", "mkdir", "unlink", "rename", "touch", "write"}, calls
    opens = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "open"]
    assert not opens, "promote.py reads through Path.read_text only"


def test_noise_report_lists_pairs_needed():
    with world() as w:
        w.runs("s", {"x": {"short": [1.0, -1.0, 2.0, -2.0], "edit": [5.0, 5.4, 4.6, 5.0]}})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            P.main(["--noise", str(w.out / "s.jsonl")])
        text = out.getvalue()
        assert "short" in text and "x SD" in text and "pairs for 1%/3%" in text


# ------------------------------------------------------------------------------------------------ the catalog and the specs
def test_the_catalog_matches_the_specs():
    cat = json.loads((P.ROOT / "bench/promotion/phase16.json").read_text())["items"]
    names = [i["name"] for i in cat]
    assert len(names) == len(set(names)) and len(names) >= 30
    for i in cat:
        P.validate(i, P.ROOT)                       # arms, claim cases, vs, hashes, branch, exact flag
        if i.get("decides") == "taskgate":
            assert not i["exact"]


def test_every_spec_is_valid_json_in_the_abtest_format():
    keys = {"env", "args", "drafting", "config", "exe", "request"}
    for f in sorted((P.ROOT / "bench/specs").glob("*.json")):
        s = json.loads(f.read_text())
        assert "base" in s and s["arms"], f.name
        for arm, v in s["arms"].items():
            assert set(v) <= keys, (f.name, arm)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
