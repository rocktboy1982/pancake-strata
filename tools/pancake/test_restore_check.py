"""Tests for restore_check.py's pure parts (no engine, no GPU): the scripted conversation's structure, the strict comparison,
the refusal of a run in which the cache never served a hit, and the noise-aware decision rule (--control, --teacher-forced) on
the two real 11-11 GPU runs and on synthetic ones."""
import contextlib
import http.server
import io
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import restore_check as rc  # noqa: E402


def ans(turn, content="x", cached=100, reasoning="", calls=None):
    return {"turn": turn, "content": content, "reasoning": reasoning, "tool_calls": calls or [],
            "usage": {"prompt_tokens_details": {"cached_tokens": cached}}}


class Script(unittest.TestCase):
    def setUp(self):
        self.t = dict(rc.turns())

    def test_labels(self):
        self.assertEqual(list(self.t), ["first", "after-tool-result", "follow-up", "branch-back", "tool-retry", "repeat"])

    def test_every_turn_continues_or_branches_an_earlier_one(self):
        first = self.t["first"]
        for label in ("after-tool-result", "follow-up", "branch-back", "tool-retry"):
            self.assertEqual(self.t[label][:len(first)], first, label)       # all share the system + first user message
        atr = self.t["after-tool-result"]
        self.assertEqual(self.t["follow-up"][:len(atr)], atr)                # a continuation of the parked state
        self.assertEqual(self.t["branch-back"][:len(atr)], atr)              # a branch at the tool result's end
        self.assertNotEqual(self.t["branch-back"][len(atr)], self.t["follow-up"][len(atr)])

    def test_tool_retry_shares_the_call_and_has_another_result(self):
        atr, retry = self.t["after-tool-result"], self.t["tool-retry"]
        self.assertEqual(retry[:len(atr) - 1], atr[:len(atr) - 1])           # up to and including the assistant's call
        self.assertNotEqual(retry[-1]["content"], atr[-1]["content"])        # not the same tool result
        self.assertEqual(retry[-1]["tool_call_id"], atr[-1]["tool_call_id"])

    def test_repeat_is_the_previous_prompt_exactly(self):
        # a complete hit: the restored state covers the whole prompt (the UNSW off-by-one case)
        self.assertEqual(self.t["repeat"], self.t["tool-retry"])

    def test_the_result_is_long_enough_to_be_checkpointed(self):
        # the engine's tool-boundary mark needs >= 1024 tokens before it (TOOL_MARKS_LEAST) and a gap (the script sets 16)
        self.assertGreater(len(self.t["after-tool-result"][0]["content"]), 1000)


class Compare(unittest.TestCase):
    def arms(self, **over):
        names = ["first", "after-tool-result", "follow-up", "branch-back", "tool-retry", "repeat"]
        reread = [ans(n) for n in names]
        restore = [dict(a) for a in reread]
        for i, k in over.items():
            restore[int(i)] = k
        return {"restore": restore}, reread

    def run_compare(self, arms, reread):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            return rc.compare(arms, reread)

    def test_all_equal(self):
        self.assertEqual(self.run_compare(*self.arms()), [])

    def test_a_different_answer_is_reported_with_its_turn(self):
        diffs = self.run_compare(*self.arms(**{"2": ans("follow-up", content="y")}))
        self.assertEqual([(d["arm"], d["turn"]) for d in diffs], [("restore", "follow-up")])

    def test_reasoning_and_tool_calls_count(self):
        diffs = self.run_compare(*self.arms(**{"1": ans("after-tool-result", reasoning="think"),
                                               "3": ans("branch-back", calls=[{"id": "c"}])}))
        self.assertEqual([d["turn"] for d in diffs], ["after-tool-result", "branch-back"])

    def test_repeat_must_equal_tool_retry_inside_an_arm(self):
        arms, reread = self.arms()
        arms["restore"][4] = ans("tool-retry", content="same as reread")
        arms["restore"][5] = ans("repeat", content="same as reread")
        reread[4] = ans("tool-retry", content="same as reread")
        reread[5] = ans("repeat", content="same as reread")
        self.assertEqual(self.run_compare(arms, reread), [])
        # the restore arm answers the repeat differently from its own tool-retry while the re-read arm agrees with both
        arms["restore"][5] = ans("repeat", content="other")
        reread_ok = [dict(a) for a in reread]
        diffs = self.run_compare(arms, reread_ok)
        self.assertIn(("restore", "repeat"), [(d["arm"], d["turn"]) for d in diffs])
        self.assertEqual(sum(1 for d in diffs if "tool-retry" in d), 1)       # the intra-arm check fired once


class Exercised(unittest.TestCase):
    def test_a_cache_that_never_hit_is_refused(self):
        a = [ans("first", cached=0), ans("after-tool-result", cached=0), ans("follow-up", cached=1200)]
        self.assertEqual(rc.never_hit(a), ["after-tool-result"])              # "first" is cold by design

    def test_missing_usage_counts_as_no_hit(self):
        a = [{"turn": "repeat", "content": "", "reasoning": "", "tool_calls": [], "usage": {}}]
        self.assertEqual(rc.never_hit(a), ["repeat"])

    def test_all_hits(self):
        self.assertEqual(rc.never_hit([ans(t) for t in rc.HIT_TURNS]), [])


TOKDIR = Path(__import__("os").environ.get("STRATA_TEST_TOKENIZER",
                                           "/home/dan/clone-of-d/Strata-data/packs/iq2_xs/tokenizer"))


@unittest.skipUnless((TOKDIR / "chat_template.jinja").exists() and (TOKDIR / "vocab.json").exists(),
                     "needs a pack's tokenizer folder (STRATA_TEST_TOKENIZER)")
class RealTemplate(unittest.TestCase):
    """The scripted conversation, rendered by the pack's own template and tokenizer as the server does: the call boundary
    must be a tool-boundary mark (>= 1024 tokens in) that both prompts of the retry share, or --tool-boundary tests nothing."""

    def test_the_retry_resumes_at_a_mark(self):
        import json
        sys.path.insert(0, str(rc.ROOT))
        sys.path.insert(0, str(rc.ROOT / "tools"))
        import strata_tokenizer as ST
        from serve import frontend as F
        from serve.server import qwen_tool_marks
        vocab = json.loads((TOKDIR / "vocab.json").read_text())
        tokens = [None] * len(vocab)
        for tk, i in vocab.items():
            tokens[i] = tk
        tcfg = json.loads((TOKDIR / "tokenizer.json").read_text())
        tok = ST.Tokenizer(tokens, (TOKDIR / "merges.txt").read_text().split("\n"),
                           json.loads((TOKDIR / "token_type.json").read_text()), tcfg.get("pre") or "qwen35",
                           tcfg.get("special_ids") or None)
        tpl = F.ChatTemplate(TOKDIR / "chat_template.jinja")
        turns = dict(rc.turns())
        ids = {}
        for k in ("after-tool-result", "tool-retry"):
            msgs, tools, kw = F.openai_to_messages({"messages": turns[k], "tools": rc.TOOLS})
            kw = {a: b for a, b in (kw or {}).items() if a != "enable_thinking"}
            ids[k] = tok.encode(tpl.render(msgs, tools=tools, add_generation_prompt=True, enable_thinking=True, **kw),
                                parse_special=True)
        marks = {k: qwen_tool_marks(v, vocab["<|im_start|>"], vocab["<tool_response>"]) for k, v in ids.items()}
        call_mark = marks["tool-retry"][0]
        self.assertGreaterEqual(call_mark, 1024)
        self.assertEqual(marks["after-tool-result"][0], call_mark)
        self.assertEqual(ids["after-tool-result"][:call_mark], ids["tool-retry"][:call_mark])


NAMES = ["first", "after-tool-result", "follow-up", "branch-back", "tool-retry", "repeat"]
GPU = rc.ROOT / "bench/results/2026-11-11-gpu"
NEW, BASE = GPU / "restore-check.json", GPU / "restore-check-BASEBINARY.json"


def said(turn, text, cached=100):
    """An answer whose whole text is `text` (all of it in the reasoning)."""
    return {"turn": turn, "content": "", "reasoning": text, "tool_calls": [],
            "usage": {"prompt_tokens_details": {"cached_tokens": 0 if turn == "first" else cached}}}


def reread_arm():
    # 120 pieces a turn; "repeat" is the same prompt as "tool-retry", so it says the same
    return [said(n, " ".join(f"{'tool-retry' if n == 'repeat' else n}{i % 7}" for i in range(120))) for n in NAMES]


def split_at(answer, at):
    """The answer with its first difference from the original at text unit `at`."""
    u = rc.UNIT.findall(answer["reasoning"])
    return {**answer, "reasoning": "".join(u[:at]) + "\x02" + "".join(u[at:])}


def leaving(positions):
    """An arm that leaves the synthetic re-read arm at positions[i] in turn i (None: never)."""
    return [a if p is None else split_at(a, p) for a, p in zip(reread_arm(), positions)]


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kw)


class Positions(unittest.TestCase):
    def test_split_point(self):
        a = said("x", "one two three four")
        self.assertEqual(rc.split_point(a, a), (None, len(rc.units(a))))
        self.assertEqual(rc.split_point(a, split_at(a, 3)), (3, 4))
        self.assertEqual(rc.split_point(a, said("x", "one two")), (3, 4))
        with mock.patch.object(rc, "units", side_effect=[["a", "b"], ["a", "b", "c"]]):   # one stops where the other goes on
            self.assertEqual(rc.split_point(None, None), (2, 3))

    def test_tool_calls_and_content_count(self):
        a, b = said("x", "same"), said("x", "same")
        b["tool_calls"] = [{"id": "c"}]
        self.assertIsNotNone(rc.split_point(a, b)[0])
        b = {**a, "content": "more"}
        self.assertIsNotNone(rc.split_point(a, b)[0])

    def test_rate_test(self):
        none = [(None, 100)] * 5
        self.assertEqual(rc.rate_test(none, none), 1.0)                                   # nobody diverged
        self.assertLess(rc.rate_test([(3, 4)] * 5, none), 1e-6)                           # five early splits, a quiet control
        self.assertGreater(rc.rate_test(none, [(3, 4)] * 5), 0.99)                        # the control is the noisy one
        same = [(30, 31), (None, 100), (50, 51), (20, 21), (70, 71)]
        self.assertGreater(rc.rate_test(same, same), 0.4)                                 # equal rates: nothing to see

    def test_a_noisy_control_cannot_resolve_an_early_bug(self):
        self.assertTrue(rc.resolves([(None, 100)] * 5))
        self.assertTrue(rc.resolves([(98, 99), (19, 20), (23, 24), (2, 3), (51, 52)]))     # the real control's shape
        self.assertFalse(rc.resolves([(1, 2)] * 5))                                       # it splits earlier than the bug would
        self.assertFalse(rc.resolves([]))

    def test_the_turn_majority_rule(self):
        late = [(60, 61)] * 5
        self.assertTrue(rc.turn_majority([(3, 4)] * 5, late))
        self.assertFalse(rc.turn_majority(late, late))                                    # ties are not earlier
        self.assertEqual(rc.turns_earlier([(3, 4), (None, 9), (5, 6)], [(None, 9), (None, 9), (5, 6)]), 1)


class SyntheticRuns(unittest.TestCase):
    CONTROL = [26, 60, 19, 23, 40, 51]            # where a second re-read leaves the first one (shaped like the real data)

    def report(self, positions, control=CONTROL, **arms):
        return quiet(rc.noise_report, {"restore": leaving(positions), **{k: leaving(v) for k, v in arms.items()}},
                     reread_arm(), leaving(control) if control else None)

    def test_a_cache_as_noisy_as_a_second_reread_is_consistent_with_noise(self):
        rep = self.report([26, 50, 30, 24, 8, 8])
        self.assertEqual((rep["exit"], rep["arms"]["restore"]["verdict"]), (0, "pass"))

    def test_a_restore_bug_that_diverges_at_token_3_in_every_turn_fails(self):
        rep = self.report([26, 3, 3, 3, 3, 3])
        self.assertEqual((rep["exit"], rep["arms"]["restore"]["verdict"]), (1, "fail"))
        self.assertLess(rep["arms"]["restore"]["position"]["p"], 0.01)

    def test_one_bad_arm_fails_the_run(self):
        rep = self.report([26, 50, 30, 24, 8, 8], disk=[26, 3, 3, 3, 3, 3])
        self.assertEqual((rep["exit"], rep["arms"]["restore"]["verdict"], rep["arms"]["disk"]["verdict"]), (1, "pass", "fail"))

    def test_a_noise_floor_too_high_is_inconclusive_not_a_pass(self):
        # the control already leaves the re-read arm at the second text unit of every turn: a bug at the fourth is invisible
        rep = self.report([26, 1, 1, 1, 1, 1], control=[1] * 6)
        self.assertEqual((rep["exit"], rep["arms"]["restore"]["verdict"]), (3, "inconclusive"))
        rep = self.report([26, 3, 3, 3, 3, 3], control=[1] * 6)
        self.assertEqual(rep["exit"], 3)

    def test_the_exact_evidence_fails_a_run_the_positions_would_pass(self):
        arm = leaving([26, 50, 30, 24, 8, 70])                      # its repeat answers differently from its own tool-retry
        rep = quiet(rc.noise_report, {"restore": arm}, reread_arm(), leaving(self.CONTROL))
        self.assertEqual(rep["arms"]["restore"]["exact"], "DIFFERENT")
        self.assertEqual(rep["arms"]["restore"]["position"]["verdict"], "pass")
        self.assertEqual((rep["exit"], rep["arms"]["restore"]["verdict"]), (1, "fail"))

    def test_the_rereads_own_repeat_is_only_a_noise_sample(self):
        control = leaving(self.CONTROL)
        control[5] = split_at(control[5], 2)                        # the control's repeat differs from its tool-retry
        rep = quiet(rc.noise_report, {"restore": leaving([26, 50, 30, 24, 8, 8])}, reread_arm(), control)
        self.assertEqual(rep["exit"], 0)

    def test_needs_a_control_or_logprobs(self):
        with self.assertRaises(ValueError):
            rc.noise_report({"restore": reread_arm()}, reread_arm())


@unittest.skipUnless(NEW.exists() and BASE.exists(), "the 11-11 GPU run files are not in this checkout")
class RealRuns(unittest.TestCase):
    """bench/results/2026-11-11-gpu: the new binary's run and the base binary's run, no control arm in either.  Each run's
    cache arms are judged against its own re-read arm with the OTHER run's re-read arm as the control (an independent
    engine start; it also carries whatever the two binaries differ by, which can only make the control noisier)."""

    def setUp(self):
        self.new, self.base = rc.load_run(NEW), rc.load_run(BASE)
        self.pairs = {"new binary": (self.new[0], self.new[1], self.base[1]),
                      "base binary": (self.base[0], self.base[1], self.new[1])}

    def test_the_strict_comparison_said_different_on_every_turn_including_the_cold_one(self):
        for label, (arms, reread, _) in self.pairs.items():
            diffs = quiet(rc.compare, arms, reread)
            self.assertIn(("restore", "first"), [(d["arm"], d["turn"]) for d in diffs], label)
            self.assertGreaterEqual(len(diffs), 12, label)

    def test_both_runs_are_consistent_with_noise_against_the_control(self):
        for label, (arms, reread, control) in self.pairs.items():
            rep = quiet(rc.noise_report, arms, reread, control)
            self.assertEqual(rep["exit"], 0, label)
            for name in arms:
                self.assertEqual(rep["arms"][name]["verdict"], "pass", (label, name))
                self.assertGreater(rep["arms"][name]["position"]["p"], 0.3, (label, name))

    def test_the_exact_evidence_is_reported_apart_and_matches(self):
        for label, (arms, reread, control) in self.pairs.items():
            rep = quiet(rc.noise_report, arms, reread, control)
            for name in arms:
                self.assertEqual(rep["arms"][name]["exact"], "MATCH", (label, name))
            self.assertEqual(rc.repeat_vs_retry(reread), "DIFFERENT")        # the re-read arm's own pair is noise
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc.noise_report(*self.pairs["new binary"])
        self.assertIn("restore repeat == tool-retry (exact evidence): MATCH", out.getvalue())

    def test_the_control_can_resolve_an_early_bug_in_these_runs(self):
        for arms, reread, control in self.pairs.values():
            self.assertTrue(rc.resolves(rc.splits(control, reread, rc.HIT_TURNS)))

    def test_the_same_runs_with_a_bug_planted_fail(self):
        for label, (arms, reread, control) in self.pairs.items():
            broken = {name: [split_at(r, 3) if r["turn"] in rc.HIT_TURNS else r for r in answers]
                      for name, answers in arms.items()}
            rep = quiet(rc.noise_report, broken, reread, control)
            self.assertEqual(rep["exit"], 1, label)
            self.assertEqual((rep["arms"]["restore"]["exact"], rep["arms"]["restore"]["verdict"]), ("MATCH", "fail"))   # the positions

    def test_a_control_that_splits_at_once_makes_them_inconclusive(self):
        arms, reread, _ = self.pairs["new binary"]
        rep = quiet(rc.noise_report, arms, reread, [split_at(r, 1) for r in reread])
        self.assertEqual(rep["exit"], 3)

    def test_the_fitted_noise_is_about_a_percent_a_unit(self):
        h, events, watched = rc.estimate_hazard([[*self.new[0].values(), self.new[1]], [*self.base[0].values(), self.base[1]]])
        self.assertTrue(0.005 < h < 0.03, h)
        self.assertGreater(events, 20)


class Cli(unittest.TestCase):
    """--analyze on saved files: the exit codes 0 / 1 / 2 / 3 and the strict form without a control."""

    def main(self, run, *flags):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "run.json"
            f.write_text(json.dumps(run))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return rc.main(["--analyze", str(f), *flags])

    def run_of(self, restore, control=None):
        return {"restore": leaving(restore), "reread": reread_arm(), **({"control": leaving(control)} if control else {})}

    def test_exit_codes(self):
        noisy = [26, 60, 19, 23, 40, 51]
        self.assertEqual(self.main(self.run_of([26, 50, 30, 24, 8, 8], noisy)), 0)
        self.assertEqual(self.main(self.run_of([26, 3, 3, 3, 3, 3], noisy)), 1)
        self.assertEqual(self.main(self.run_of([26, 1, 1, 1, 1, 1], [1] * 6)), 3)
        never = self.run_of([26, 50, 30, 24, 8, 8], noisy)
        never["restore"][2]["usage"] = {"prompt_tokens_details": {"cached_tokens": 0}}   # follow-up never hit the cache
        self.assertEqual(self.main(never), 2)

    def test_without_a_control_the_strict_comparison_stays(self):
        self.assertEqual(self.main(self.run_of([None] * 6)), 0)
        self.assertEqual(self.main(self.run_of([26, 50, 30, 24, 8, 8])), 1)

    def test_control_from_another_file(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.json", Path(d) / "b.json"
            a.write_text(json.dumps(self.run_of([26, 50, 30, 24, 8, 8])))
            b.write_text(json.dumps({"reread": leaving([26, 60, 19, 23, 40, 51])}))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(rc.main(["--analyze", str(a), "--control-from", str(b)]), 0)

    def test_out_holds_the_noise_report(self):
        with tempfile.TemporaryDirectory() as d:
            f, out = Path(d) / "run.json", Path(d) / "out.json"
            f.write_text(json.dumps(self.run_of([26, 3, 3, 3, 3, 3], [26, 60, 19, 23, 40, 51])))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(rc.main(["--analyze", str(f), "--out", str(out)]), 1)
            saved = json.loads(out.read_text())
            self.assertEqual((saved["noise"]["exit"], saved["noise"]["arms"]["restore"]["verdict"]), (1, "fail"))
            self.assertIn("control", saved)


def entries(n, dev=None, flip=None):
    """n tokens at logprob -0.1 (+ dev[i]) that list one alternative 3 nats lower; flip = (i, own token, other token, gap nats):
    token i is `own`, and `other` is listed `gap` nats below it."""
    out = []
    for i in range(n):
        tok, lp = f"t{i}", -0.1 + (dev or {}).get(i, 0.0)
        top = [{"token": tok, "logprob": lp}, {"token": "alt", "logprob": lp - 3.0}]
        if flip and flip[0] == i:
            tok, top = flip[1], [{"token": flip[1], "logprob": lp}, {"token": flip[2], "logprob": lp - flip[3]}]
        out.append({"token": tok, "logprob": lp, "top_logprobs": top})
    return out


def lp_arm(**per_turn):
    """A run on the six turns whose turn named `x_y` (x-y) has the given entries, 40 clean tokens otherwise."""
    return [{**said(n, "tool-retry" if n == "repeat" else n), "logprobs": {"content": per_turn.get(n.replace("-", "_"), entries(40))}}
            for n in NAMES]


class Logprobs(unittest.TestCase):
    def test_compare_counts_the_shared_tokens_and_their_deviation(self):
        a, b = {"logprobs": {"content": entries(40)}}, {"logprobs": {"content": entries(40, dev={5: 0.2})}}
        self.assertEqual(rc.lp_compare(a, a), {"tokens": 40, "dev": 0.0, "gap": 0.0})
        self.assertAlmostEqual(rc.lp_compare(a, b)["dev"], 0.2)

    def test_reasoning_entries_come_before_the_content_ones(self):
        a = {"logprobs": {"reasoning_content": entries(3), "content": entries(4)}}
        self.assertEqual(len(rc.lp_entries(a)), 7)
        self.assertEqual(rc.lp_entries({}), [])

    def test_a_flip_is_explained_when_both_tokens_were_near_ties(self):
        a = {"logprobs": {"content": entries(40, flip=(7, "x", "y", 0.01))}}
        b = {"logprobs": {"content": entries(40, flip=(7, "y", "x", 0.02))}}
        got = rc.lp_compare(a, b)
        self.assertEqual(got["tokens"], 7)
        self.assertAlmostEqual(got["gap"], 0.02)
        far = {"logprobs": {"content": entries(40, flip=(7, "y", "x", 2.0))}}
        self.assertAlmostEqual(rc.lp_compare(a, far)["gap"], 2.0)                    # the other run was sure of its token

    def test_the_gap_is_a_lower_bound_when_the_other_token_is_not_listed(self):
        e = {"token": "a", "logprob": -0.1, "top_logprobs": [{"token": "a", "logprob": -0.1}, {"token": "b", "logprob": -2.1}]}
        self.assertAlmostEqual(rc.flip_gap(e, "zzz"), 2.0)
        self.assertIsNone(rc.flip_gap({"token": "a", "logprob": -0.1, "top_logprobs": []}, "b"))

    def check(self, arm, control=None, **kw):
        return rc.lp_check(arm, lp_arm(), control, rc.HIT_TURNS, **kw)

    def test_a_clean_cache_arm_passes(self):
        got = self.check(lp_arm())
        self.assertEqual((got["verdict"], got["tokens"]), ("pass", 200))

    def test_a_deviation_over_the_tolerance_fails_and_names_the_turn(self):
        got = self.check(lp_arm(follow_up=entries(40, dev={5: 0.2})))
        self.assertEqual((got["verdict"], got["bad"]), ("fail", ["follow-up"]))

    def test_the_control_widens_the_tolerance_to_what_the_engine_does_on_its_own(self):
        arm = lp_arm(follow_up=entries(40, dev={5: 0.2}))
        self.assertEqual(self.check(arm, lp_arm(first=entries(40, dev={3: 0.15})))["verdict"], "pass")   # 2 x 0.15 > 0.2
        self.assertEqual(self.check(arm, lp_arm(first=entries(40, dev={3: 0.05})))["verdict"], "fail")
        self.assertEqual(self.check(arm, lp_tol=0.5)["verdict"], "pass")

    def test_a_flip_that_one_run_was_sure_of_fails(self):
        arm = lp_arm(branch_back=entries(40, flip=(9, "y", "x", 2.0)))
        reread = lp_arm(branch_back=entries(40, flip=(9, "x", "y", 0.01)))
        got = rc.lp_check(arm, reread, None, rc.HIT_TURNS)
        self.assertEqual((got["verdict"], got["bad"]), ("fail", ["branch-back"]))
        near = lp_arm(branch_back=entries(40, flip=(9, "y", "x", 0.01)))
        self.assertEqual(rc.lp_check(near, reread, None, rc.HIT_TURNS)["verdict"], "pass")

    def test_too_few_shared_tokens_is_inconclusive(self):
        early = {n: entries(40, flip=(2, "y", "x", 0.01)) for n in ("after_tool_result", "follow_up", "branch_back", "tool_retry", "repeat")}
        got = rc.lp_check(lp_arm(**early), lp_arm(**{k: entries(40, flip=(2, "x", "y", 0.01)) for k in early}), None, rc.HIT_TURNS)
        self.assertEqual((got["verdict"], got["tokens"]), ("inconclusive", 10))

    def test_the_logprob_verdict_decides_when_it_ran(self):
        # positions say a bug (every hit turn splits at 3), but the logprobs of the shared tokens and the flips are near-ties
        arm = [{**a, "logprobs": {"content": entries(40, flip=(3, "y", "x", 0.01))}} for a in leaving([26, 3, 3, 3, 3, 3])]
        reread = [{**a, "logprobs": {"content": entries(40, flip=(3, "x", "y", 0.01))}} for a in reread_arm()]
        rep = quiet(rc.noise_report, {"restore": arm}, reread, leaving([26, 60, 19, 23, 40, 51]))
        self.assertEqual(rep["arms"]["restore"]["position"]["verdict"], "fail")
        self.assertEqual(rep["arms"]["restore"]["logprob"]["verdict"], "inconclusive")   # 3 shared tokens a turn: too few
        self.assertEqual(rep["exit"], 3)
        arm = [{**a, "logprobs": {"content": entries(40, flip=(3, "y", "x", 3.0))}} for a in leaving([26, 3, 3, 3, 3, 3])]
        rep = quiet(rc.noise_report, {"restore": arm}, reread, leaving([26, 60, 19, 23, 40, 51]))
        self.assertEqual((rep["arms"]["restore"]["logprob"]["verdict"], rep["exit"]), ("fail", 1))

    def test_a_run_with_logprobs_and_no_control_is_judged_on_the_floor_alone(self):
        rep = quiet(rc.noise_report, {"restore": lp_arm()}, lp_arm())
        self.assertEqual((rep["exit"], rep["arms"]["restore"]["logprob"]["tol"]), (0, rc.LP_TOL))


class ChatRequest(unittest.TestCase):
    """chat() against a stand-in for the server's /v1/chat/completions."""

    def serve(self, reply):
        seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        return srv.server_address[1], seen

    REPLY = {"choices": [{"message": {"content": "hi", "reasoning_content": "hm"},
                          "logprobs": {"content": [{"token": "hi", "logprob": -0.5, "bytes": [104, 105],
                                                    "top_logprobs": [{"token": "hi", "logprob": -0.5, "bytes": [104]}]}],
                                       "refusal": None,
                                       "reasoning_content": [{"token": "hm", "logprob": -1.5, "bytes": [104], "top_logprobs": []}]}}],
             "usage": {}}

    def test_without_logprobs_nothing_changes(self):
        port, seen = self.serve(self.REPLY)
        got = rc.chat(port, [{"role": "user", "content": "x"}], 8)
        self.assertNotIn("logprobs", seen[0])
        self.assertNotIn("top_logprobs", seen[0])
        self.assertEqual(sorted(got), ["content", "reasoning", "tool_calls", "usage"])

    def test_teacher_forced_asks_for_top_logprobs_and_keeps_the_tokens_without_their_bytes(self):
        port, seen = self.serve(self.REPLY)
        got = rc.chat(port, [{"role": "user", "content": "x"}], 8, rc.TOP_LOGPROBS)
        self.assertEqual((seen[0]["logprobs"], seen[0]["top_logprobs"]), (True, rc.TOP_LOGPROBS))
        self.assertEqual(got["logprobs"]["content"], [{"token": "hi", "logprob": -0.5, "top_logprobs": [{"token": "hi", "logprob": -0.5}]}])
        self.assertEqual(got["logprobs"]["reasoning_content"], [{"token": "hm", "logprob": -1.5, "top_logprobs": []}])
        self.assertEqual(len(rc.lp_entries(got)), 2)


class Simulation(unittest.TestCase):
    """The error rates of the position rule on the noise model fitted to the 11-11 runs (h = 0.0122 per text unit, the five
    hit turns' lengths): numbers quoted in pancake-docs/batch4/equiv-gate.md.  Seeded, so exact; the bounds leave room."""
    H, LENGTHS, TRIALS = 0.0122, [112, 175, 128, 145, 143], 1500

    def rates(self, bug, h=H):
        return rc.simulate(h, self.LENGTHS, bug, self.TRIALS)

    def test_an_equivalent_cache_rarely_fails(self):
        got = self.rates(None)
        self.assertLess(got["rate_fail"], 0.05)

    def test_the_turn_majority_rule_fails_an_equivalent_cache_far_too_often(self):
        self.assertGreater(self.rates(None)["majority_fail"], 0.10)

    def test_a_bug_at_token_3_or_a_much_higher_divergence_rate_is_caught(self):
        self.assertGreater(self.rates(("at", rc.BUG_POS))["rate_fail"], 0.97)
        self.assertGreater(self.rates(("rate", 11))["rate_fail"], 0.90)
        self.assertGreater(self.rates(("rate", 4))["rate_fail"], 0.40)                  # 4x is only caught about half the time

    def test_a_very_noisy_engine_is_inconclusive_not_a_pass(self):
        got = self.rates(("at", rc.BUG_POS), h=0.4)
        self.assertGreater(got["rate_inconclusive"] + got["rate_fail"], 0.9)
        self.assertGreater(got["rate_inconclusive"], 0.4)

    def test_simulate_is_seeded(self):
        self.assertEqual(rc.simulate(self.H, self.LENGTHS, None, 200), rc.simulate(self.H, self.LENGTHS, None, 200))


@unittest.skipUnless(NEW.exists() and BASE.exists(), "the 11-11 GPU run files are not in this checkout")
class SimulateCli(unittest.TestCase):
    def test_the_table_is_printed_for_the_real_runs(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(rc.main(["--simulate", str(NEW), str(BASE)]), 0)
        self.assertIn("noise model fitted to 2 saved run(s)", out.getvalue())
        self.assertEqual(out.getvalue().count("no bug"), 3)


if __name__ == "__main__":
    unittest.main()
