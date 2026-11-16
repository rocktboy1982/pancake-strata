"""STRATA_KNOB_TRIAL (serve/knob_trial.py): the decision rule against the cases of src/program/knob_trial_test.cpp, the
runner, and the server's two hooks through the HTTP API with a fake engine (no GPU or model pack).

    python -m unittest serve.test_knob_trial -v
"""
import json
import unittest
import urllib.request
from pathlib import Path

from serve import knob_trial as K
from serve import server as S
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, serve
from serve.test_server import ThinkingEngine


def run(rate_of, hit=0.9, max_windows=40):
    t = K.KnobTrial()
    for i in range(max_windows):
        if t.verdict != K.RUNNING:
            break
        a = t.next_arm()
        t.add(a, 8, 8.0 / rate_of(a, i), hit)
    return t


class Rule(unittest.TestCase):
    def test_flat_20_percent_win(self):
        w = run(lambda a, i: 120.0 if a == K.CANDIDATE else 100.0)
        self.assertEqual(w.verdict, K.WIN)
        self.assertEqual(len(w.ratios), 3)
        self.assertAlmostEqual(w.median_ratio(), 1.2)
        self.assertEqual((w.windows, w.settle_windows), (8, 2))

    def test_five_percent_is_under_the_margin_and_exactly_ten_wins(self):
        s = run(lambda a, i: 105.0 if a == K.CANDIDATE else 100.0)
        self.assertEqual((s.verdict, len(s.ratios), s.reason), (K.LOSE, 3, "median ratio below the margin"))
        self.assertEqual(run(lambda a, i: 110.0 if a == K.CANDIDATE else 100.0).verdict, K.WIN)

    def test_drift_cancels(self):
        d = run(lambda a, i: (1.2 if a == K.CANDIDATE else 1.0) * 100.0 * (1.0 - 0.01 * i))
        self.assertEqual(d.verdict, K.WIN)
        self.assertLess(abs(d.median_ratio() - 1.2), 0.01)
        dn = run(lambda a, i: 100.0 * (1.0 - 0.02 * i))
        self.assertEqual(dn.verdict, K.LOSE)
        self.assertLess(abs(dn.median_ratio() - 1.0), 0.02)

    def test_two_lost_rounds_stop_early_and_a_20x_slower_window_is_cut(self):
        n = [0]

        def lost(a, i):
            if a != K.CANDIDATE:
                return 100.0
            n[0] += 1
            return 95.0
        l3 = run(lost)
        self.assertEqual((l3.verdict, len(l3.ratios), l3.reason), (K.LOSE, 2, "more than half of the rounds lost"))
        c = run(lambda a, i: 5.0 if a == K.CANDIDATE else 100.0)
        self.assertEqual((c.verdict, c.ratios, c.windows, c.reason),
                         (K.LOSE, [], 3, "a candidate window ran apart from the incumbent band"))

    def test_settle_and_undecided_and_wrong_arm(self):
        st = K.KnobTrial()
        st.add(K.INCUMBENT, 8, 0.08, 0.50)
        st.add(K.INCUMBENT, 8, 0.08, 0.80)
        self.assertEqual((st.next_arm(), st.settle_windows), (K.INCUMBENT, 2))
        st.add(K.INCUMBENT, 8, 0.08, 0.81)
        self.assertEqual(st.next_arm(), K.CANDIDATE)
        su = K.KnobTrial()
        su.add(K.INCUMBENT, 8, 0.08)
        su.add(K.INCUMBENT, 8, 0.08)
        self.assertEqual(su.next_arm(), K.CANDIDATE)
        u = K.KnobTrial()
        u.add(K.INCUMBENT, 8, 0.08, 0.9)
        u.add(K.INCUMBENT, 3, 0.03, 0.9)
        self.assertEqual(u.verdict, K.UNDECIDED)
        u.add(K.INCUMBENT, 8, 0.08, 0.9)
        self.assertEqual(u.verdict, K.UNDECIDED)
        wa = K.KnobTrial()
        wa.add(K.CANDIDATE, 8, 0.08, 0.9)
        self.assertEqual((wa.next_arm(), wa.settle_windows), (K.INCUMBENT, 0))

    def test_log_lines_equal_the_cpp_ones(self):
        w = run(lambda a, i: 120.0 if a == K.CANDIDATE else 100.0)
        self.assertEqual(w.line("adapt-swaps", "32", "64"),
                         "strata knob trial: adapt-swaps 32 -> 64: WIN ratio 1.200 (rounds 1.200 1.200 1.200); "
                         "8 windows of 8 tokens, 2 to settle")
        c = run(lambda a, i: 5.0 if a == K.CANDIDATE else 100.0)
        self.assertEqual(c.line("k", "a", "b"), "strata knob trial: k a -> b: LOSE; 3 windows of 8 tokens, 2 to settle; "
                                               "a candidate window ran apart from the incumbent band")


class Runner(unittest.TestCase):
    def test_candidates_are_validated_and_unset_gives_none(self):
        self.assertEqual(K.parse_candidates('[{"spec_min_p": 0.4, "bogus": 1}, {"pcie_frac": 3}, 5, {"aux_cpus": 1}]'),
                         [{"spec_min_p": 0.4}, {"aux_cpus": 1}])
        self.assertEqual(K.parse_candidates("not json"), K.DEFAULT_CANDIDATES)
        self.assertEqual(K.parse_candidates(None), K.DEFAULT_CANDIDATES)
        self.assertIsNone(K.from_env({}))
        self.assertIsNone(K.from_env({"STRATA_KNOB_TRIAL": "0"}))
        self.assertIsNotNone(K.from_env({"STRATA_KNOB_TRIAL": "1"}))

    def test_a_request_with_its_own_tune_is_left_alone_and_not_counted(self):
        r = K.KnobTrialRunner([{"spec_min_p": 0.3}], min_tokens=8, log=lambda s: None)
        smp = {"strata_tune": {"pcie_frac": 0.5}}
        got, arm = r.assign(smp)
        self.assertIs(got, smp)
        self.assertIsNone(arm)
        self.assertIsNone(r.feed(arm, 100, 50.0))
        self.assertIsNone(r.trial.windows if r.trial else None)

    def test_short_cancelled_or_speedless_requests_are_not_windows(self):
        r = K.KnobTrialRunner([{"spec_min_p": 0.3}], min_tokens=8, log=lambda s: None)
        _, arm = r.assign({})
        self.assertEqual(arm, K.INCUMBENT)
        for tok, rate in ((3, 50.0), (100, None), (100, 0.0)):
            self.assertIsNone(r.feed(arm, tok, rate))
        self.assertEqual(r.trial.windows, 0)

    def test_a_win_is_adopted_and_a_loss_is_not(self):
        lines = []
        r = K.KnobTrialRunner([{"spec_min_p": 0.3}, {"spec_min_p": 0.9}], min_tokens=8, log=lines.append)
        rates = {0.3: 130.0, 0.9: 80.0}
        for _ in range(40):
            smp, arm = r.assign({})
            tune = (smp or {}).get("strata_tune") or {}
            rate = rates.get(tune.get("spec_min_p"), 100.0)
            if tune.get("spec_min_p") == 0.9 and "spec_min_p" in r.adopted:
                rate = 80.0
            r.feed(arm, 64, rate)
            if not r.active():
                break
        self.assertEqual(r.adopted, {"spec_min_p": 0.3})
        self.assertEqual(len(lines), 2)
        self.assertIn("WIN ratio 1.3", lines[0])
        self.assertIn("LOSE", lines[1])
        smp, arm = r.assign({"temperature": 0.0})
        self.assertEqual((smp["strata_tune"], arm), ({"spec_min_p": 0.3}, None))   # adopted for later requests, no trial left
        self.assertEqual(smp["temperature"], 0.0)


class TrialEngine(ThinkingEngine):
    """Records the sampling of every request; its decode speed depends on the strata_tune it was sent."""
    RATE = {None: 100.0, 0.3: 125.0}

    def __init__(self, tok):
        super().__init__(tok)
        self.samplings = []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.samplings.append(dict(sampling or {}))
        tune = (sampling or {}).get("strata_tune") or {}
        rate = self.RATE.get(tune.get("spec_min_p"), 100.0)
        n = 0
        stopped = False
        try:
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                n += 1
                stopped = t in self.tok.encode("<|im_end|>", parse_special=True)
                yield t
        finally:
            self.last = dict(prompt_tokens=len(ids), generated=n, reused=0, prompt_read=len(ids), finish="stop" if stopped else "cancel",
                             prompt_ms=10.0, decode_ms=1000.0 * n / rate, drafts_offered=0, drafts_accepted=0,
                             hits=9, lookups=10)


class Hooks(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = TrialEngine(self.tok)
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).with_name("chat_template.jinja")))
        self.svc.loop_guard = False
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.saved = S.KNOB_TRIAL

    def tearDown(self):
        S.KNOB_TRIAL = self.saved
        self.httpd.shutdown()
        self.httpd.server_close()

    def chat(self, extra=None):
        body = {"model": "m", "messages": [{"role": "user", "content": "2+2?"}], "max_tokens": 400, **(extra or {})}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())

    def test_unset_nothing_is_sent_and_nothing_changes(self):
        S.KNOB_TRIAL = None
        self.chat()
        self.chat()
        self.assertTrue(all("strata_tune" not in s for s in self.engine.samplings))

    def test_trial_alternates_settings_across_requests_and_adopts_the_winner(self):
        lines = []
        S.KNOB_TRIAL = K.KnobTrialRunner([{"spec_min_p": 0.3}], min_tokens=8, log=lines.append)
        for _ in range(8):                       # 2 settle + C I C I C I
            self.chat()
        tunes = [(s.get("strata_tune") or {}).get("spec_min_p") for s in self.engine.samplings]
        self.assertEqual(tunes, [None, None, 0.3, None, 0.3, None, 0.3, None])
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("WIN ratio 1.25", lines[0])
        self.assertEqual(S.KNOB_TRIAL.adopted, {"spec_min_p": 0.3})
        self.chat()                              # adopted: every later request without its own tune carries it
        self.assertEqual(self.engine.samplings[-1]["strata_tune"], {"spec_min_p": 0.3})

    def test_a_request_with_its_own_strata_tune_is_not_a_window(self):
        S.KNOB_TRIAL = K.KnobTrialRunner([{"spec_min_p": 0.3}], min_tokens=8, log=lambda s: None)
        self.chat({"strata_tune": {"pcie_frac": 0.5}})
        self.assertEqual(self.engine.samplings[-1]["strata_tune"], {"pcie_frac": 0.5})
        self.assertIsNone(S.KNOB_TRIAL.trial)    # no window was assigned or counted


if __name__ == "__main__":
    unittest.main()
