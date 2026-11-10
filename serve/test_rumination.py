"""serve/test_rumination.py - pancake item 34: the thinking's rolling 5-gram repetition rate (serve/rumination.py),
logged per request, and the opt-in "rumination_close" that closes the thinking with the soft budget's wrap-up.

    python -m unittest serve.test_rumination -v
"""
from __future__ import annotations

import collections
import json
import math
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import rumination as R  # noqa: E402
from serve import server as S  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import IM_END, REASONING_WRAP_UP, Service, think_boundary  # noqa: E402
from serve.test_loop_guard import Tok  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Thinking that goes back to the same few ideas with other words between them: no block repeats back to back (the
# loop guard's exact loops), but its 5-grams come back again and again.
IDEAS = ["wait, let me check whether the cache key includes the session id",
         "maybe the problem is that the cache key does not include the session id",
         "let me reconsider the cache key and whether it includes the session id",
         "actually, i think the cache key includes the session id after all"]
FILLERS = ["Hmm.", "Okay.", "So,", "Right.", "Then", "Well,", "Now", "Again,", "Yes,", "No,", "But"]


def ruminating(n=60):
    return "".join(f"{FILLERS[i % len(FILLERS)]} {IDEAS[(i * 3 + i // 4) % len(IDEAS)]} (try {i}).\n"
                   for i in range(n))


def fresh(n=60):
    """Thinking that keeps moving: every sentence has new words."""
    return "".join(f"Step {i}: compute term {i} as {i * 37 % 101} plus {i * i} giving w{i}x{i * 7} for "
                   f"branch b{i} under rule r{i % 97}{i}.\n" for i in range(n))


class Meter(unittest.TestCase):
    def test_fresh_text_scores_zero_and_rumination_scores_high(self):
        a, b = R.RuminationMeter(), R.RuminationMeter()
        a.feed(fresh())
        b.feed(ruminating())
        self.assertGreater(a.checks, 0)
        self.assertEqual(a.rate_max, 0.0)
        self.assertGreater(b.rate_max, 0.4)
        self.assertGreater(b.summary()["mean"], 0.3)

    def test_piecewise_feed_equals_whole(self):
        text = ruminating()
        whole, parts = R.RuminationMeter(), R.RuminationMeter()
        whole.feed(text)
        for k in range(0, len(text), 3):             # pieces that split words, as tokens do
            parts.feed(text[k:k + 3])
        self.assertEqual((whole.words, whole.checks, whole.rate_max, whole.rate_sum),
                         (parts.words, parts.checks, parts.rate_max, parts.rate_sum))

    def test_rate_counts_only_the_window(self):
        m = R.RuminationMeter(window=64, every=8, min_words=8)
        m.feed("alpha beta gamma delta epsilon " * 4)   # a repeat inside the window
        self.assertGreater(m.rate, 0)
        m.feed(" ".join(f"w{i}" for i in range(100)) + " ")   # the repeat leaves the 64-word window
        self.assertEqual(m.rate, 0.0)
        self.assertEqual(len(m.grams), 64 - R.N + 1)

    def test_no_check_on_short_thinking(self):
        m = R.RuminationMeter()
        m.feed("the same five words here " * 20)     # 100 words < MIN_WORDS
        self.assertIsNone(m.summary())

    def test_high_needs_checks_in_a_row(self):
        m = R.RuminationMeter(threshold=0.3, checks=3)
        m.feed(fresh(30))
        self.assertFalse(m.high)
        words = iter(ruminating(100).split())
        while m.streak < 2:
            m.feed(next(words) + " ")                # (one word at a time: at most one check a call)
        self.assertFalse(m.high)
        while not m.high:
            m.feed(next(words) + " ")
        self.assertEqual(m.streak, 3)
        self.assertFalse(R.RuminationMeter().high)   # (never without a threshold)

    def test_close_setting(self):
        self.assertIsNone(R.close_setting(None))
        self.assertIsNone(R.close_setting(0))
        self.assertIsNone(R.close_setting(False))
        self.assertEqual(R.close_setting(0.4), {"threshold": 0.4, "checks": R.CHECKS, "min_tokens": 0})
        self.assertEqual(R.close_setting({"threshold": 1, "checks": 5, "min_tokens": 2000}),
                         {"threshold": 1.0, "checks": 5, "min_tokens": 2000})
        for bad in (True, 1.5, -0.1, "0.4", [0.4], {"checks": 2}, {"threshold": 0.4, "checks": 0},
                    {"threshold": 0.4, "min_tokens": -1}, {"threshold": 0.4, "window": 9}):
            with self.assertRaises(ValueError, msg=repr(bad)):
                R.close_setting(bad)


class ThinkingEngine:
    """Writes `thoughts` (continued from wherever the prompt got to), then </think>, `answer` and the end; once the
    prompt holds the end of thinking (the server's wrap-up), `answer` and the end.  Records every call's prompt."""

    def __init__(self, tok, thoughts, answer="The answer."):
        self.tok, self.max_context, self.made = tok, 1 << 20, 0
        enc = lambda s: tok.encode(s, parse_special=True)   # noqa: E731
        self.thoughts, self.answer, self.end = enc(thoughts), enc(answer), enc(IM_END)
        self.think_end = enc("</think>")[0]
        self.base, self.calls = None, []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.calls.append(list(ids))
        if self.base is None:
            self.base = list(ids)
        written = ids[len(self.base):]
        if self.think_end in written:
            out = self.answer + self.end
        else:
            out = self.thoughts[len(written):] + [self.think_end] + self.answer + self.end
        for i, t in enumerate(out):
            if i >= max_new or cancel.is_set():
                return
            self.made += 1
            yield t


def run(svc, max_new=20000):
    reasoning, content, done = "", "", None
    for kind, val in svc.run(svc.tok.encode("go"), True, None, max_new, {}, threading.Event()):
        if kind == "event" and val.kind == "reasoning":
            reasoning += val.text
        elif kind == "event" and val.kind == "content":
            content += val.text
        elif kind == "done":
            done = val
    return reasoning, content, done


class Server(unittest.TestCase):
    def service(self, thoughts, close=None):
        tok = Tok()
        eng = ThinkingEngine(tok, thoughts)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.answer_room = False
        svc.rumination_close = R.close_setting(close)
        return svc, eng

    def test_log_only_by_default(self):
        """Phase 1: measured and logged; the thinking runs to its own end, the engine is called once."""
        thoughts = ruminating(40)
        svc, eng = self.service(thoughts)
        with tempfile.TemporaryDirectory() as d:
            svc.turn_log = str(Path(d) / "turns.jsonl")
            reasoning, content, done = run(svc)
            line = json.loads(Path(svc.turn_log).read_text().splitlines()[-1])
        self.assertEqual(len(eng.calls), 1)
        self.assertEqual(reasoning.strip(), thoughts.strip())
        self.assertEqual(content.strip(), "The answer.")
        rum = done["rumination"]
        self.assertGreater(rum["max"], 0.4)
        self.assertFalse(rum["closed"])
        self.assertEqual(svc.history[-1]["rumination_max"], rum["max"])
        self.assertEqual(svc.history[-1]["rumination_mean"], rum["mean"])
        self.assertEqual((line["rumination_max"], line["rumination_closed"]), (rum["max"], False))
        m = svc.metrics()["rumination"]
        self.assertEqual((m["measured"], m["closed"], m["close"]), (1, 0, None))
        self.assertIn("strata:rumination_measured_total 1\n", svc.prometheus())

    def test_short_thinking_has_no_rate(self):
        svc, _ = self.service("A short thought.\n", close=0.3)
        _, _, done = run(svc)
        self.assertNotIn("rumination", done)
        self.assertIsNone(svc.history[-1]["rumination_max"])
        self.assertEqual(svc.metrics()["rumination"]["measured"], 0)

    def test_close_wraps_up_at_a_sentence_end(self):
        """Phase 2 (opt-in): the rate stays high for K checks -> the soft budget's wrap-up at the next line end."""
        thoughts = ruminating(100)
        svc, eng = self.service(thoughts, close=0.3)
        reasoning, content, done = run(svc)
        self.assertEqual(len(eng.calls), 2)
        p1, p2 = eng.calls
        said = svc.tok.decode(p2[len(p1):])
        self.assertTrue(said.endswith(REASONING_WRAP_UP), said[-120:])
        kept = said[:-len(REASONING_WRAP_UP)]
        self.assertTrue(thoughts.startswith(kept))
        self.assertTrue(think_boundary(kept), kept[-40:])
        self.assertLess(len(kept), len(thoughts) // 2)      # closed well before the thinking's own end
        self.assertEqual(content.strip(), "The answer.")
        self.assertTrue(done["rumination"]["closed"])
        self.assertTrue(svc.history[-1]["rumination_closed"])
        m = svc.metrics()["rumination"]
        self.assertEqual((m["closed"], m["close"]["threshold"]), (1, 0.3))
        self.assertIn("strata:rumination_closed_total 1\n", svc.prometheus())

    def test_close_waits_at_most_the_grace(self):
        """Thinking with no sentence or line end: the close comes reasoning_budget_grace tokens after the signal."""
        thoughts = ruminating(100).replace(".\n", " ;; ")
        svc, eng = self.service(thoughts, close=0.3)
        svc.reasoning_budget_grace = 40
        run(svc)
        self.assertEqual(len(eng.calls), 2)
        said = svc.tok.decode(eng.calls[1][len(eng.calls[0]):])
        self.assertTrue(said.endswith(REASONING_WRAP_UP))

    def test_fresh_thinking_is_not_closed(self):
        thoughts = fresh(30)
        svc, eng = self.service(thoughts, close=0.3)
        reasoning, _, done = run(svc)
        self.assertEqual(len(eng.calls), 1)
        self.assertEqual(reasoning.strip(), thoughts.strip())
        self.assertEqual((done["rumination"]["max"], done["rumination"]["closed"]), (0.0, False))

    def test_min_tokens_holds_the_close(self):
        thoughts = ruminating(40)
        svc, eng = self.service(thoughts, close={"threshold": 0.3, "min_tokens": 10 ** 6})
        run(svc)
        self.assertEqual(len(eng.calls), 1)

    def test_status_page_shows_the_switch(self):
        svc, _ = self.service("x", close=0.35)
        self.assertEqual(svc.pancake_status()["switches"]["server"]["rumination_close"],
                         {"threshold": 0.35, "checks": R.CHECKS, "min_tokens": 0})


# ---- A20: the top-K entropy of the thinking (opt-in "rumination_entropy"), the second signal of the close
SURE = "LP -0.010 7:-0.010 8:-6.0 9:-7.0"        # one token takes almost all the mass: low entropy
UNSURE = "LP -1.386 7:-1.386 8:-1.386 9:-1.386 10:-1.386"    # four equal choices: ln 4 = 1.386


class Entropy(unittest.TestCase):
    def test_lp_entropy(self):
        self.assertAlmostEqual(R.lp_entropy(UNSURE), math.log(4), places=3)
        self.assertLess(R.lp_entropy(SURE), 0.1)
        self.assertEqual(R.lp_entropy("LP -0.5"), 0.0)          # no list: the chosen token alone
        for bad in ("LP nan", "LP -0.5 7:nan", "LP", "T 5", "LP x 7:-1", "LP -1 7:-inf"):
            self.assertIsNone(R.lp_entropy(bad), bad)

    def test_meter_rolls_and_summarises(self):
        m = R.EntropyMeter(window=4)
        self.assertIsNone(m.summary())
        for h in (1.0, 1.0, 1.0):
            m.add(h)
        self.assertFalse(m.low(9.0))                             # window not full yet
        m.add(1.0)
        self.assertTrue(m.low(1.0))
        self.assertFalse(m.low(0.5))
        for h in (0.0, 0.0, 0.0, 0.0):
            m.add(h)
        self.assertTrue(m.low(0.01))
        self.assertEqual(m.summary(), {"mean": 0.5, "min_window": 0.0, "tokens": 8})
        self.assertTrue(m.feed_line(SURE))
        self.assertFalse(m.feed_line("LP nan"))

    def test_settings(self):
        self.assertIsNone(R.entropy_setting(None))
        self.assertIsNone(R.entropy_setting(0))
        self.assertEqual(R.entropy_setting(5), 5)
        for bad in (True, 1, 21, 2.5, "5", [5]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                R.entropy_setting(bad)
        self.assertEqual(R.close_setting({"threshold": 0.3, "entropy_below": 0.5})["entropy_below"], 0.5)
        self.assertNotIn("entropy_below", R.close_setting(0.3))
        for bad in (0, -1, True, "0.5"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                R.close_setting({"threshold": 0.3, "entropy_below": bad})


class LPEngine(ThinkingEngine):
    """ThinkingEngine that writes an LP line before each token, as StrataEngine does when the request asked for
    logprobs (sampling logprobs / top_logprobs); the same line for every token."""

    line = SURE

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.lp_lines, self.sampling_seen = collections.deque(), []

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.sampling_seen.append(dict(sampling or {}))
        want = S.top_logprobs(sampling or {}) is not None
        for t in super().generate(ids, max_new, sampling, cancel, embeddings):
            if want:
                self.lp_lines.append((t, self.line))
            yield t


class UnsureEngine(LPEngine):
    line = UNSURE


class ServerEntropy(unittest.TestCase):
    def service(self, thoughts, engine_cls=LPEngine, close=None, entropy=4):
        tok = Tok()
        eng = engine_cls(tok, thoughts)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.answer_room = False
        svc.rumination_close = R.close_setting(close)
        svc.rumination_entropy = R.entropy_setting(entropy)
        return svc, eng

    def run(self, svc, sampling=None, patched=True):
        done = None
        with mock.patch.object(S, "StrataEngine", LPEngine if patched else S.StrataEngine):
            for kind, val in svc.run(svc.tok.encode("go"), True, None, 20000, sampling or {}, threading.Event()):
                if kind == "done":
                    done = val
        return done

    def test_off_by_default_nothing_is_asked_or_logged(self):
        svc, eng = self.service(ruminating(20), entropy=None)
        done = self.run(svc)
        self.assertNotIn("logprobs", eng.sampling_seen[0])
        self.assertNotIn("thinking_entropy", done)
        self.assertNotIn("entropy_mean", svc.history[-1])

    def test_measured_and_logged_with_the_switch(self):
        svc, eng = self.service(ruminating(20), entropy=4)
        with tempfile.TemporaryDirectory() as d:
            svc.turn_log = str(Path(d) / "turns.jsonl")
            done = self.run(svc)
            line = json.loads(Path(svc.turn_log).read_text().splitlines()[-1])
        self.assertEqual((eng.sampling_seen[0]["logprobs"], eng.sampling_seen[0]["top_logprobs"]), (True, 4))
        ent = done["thinking_entropy"]
        self.assertLess(ent["mean"], 0.1)
        self.assertGreater(ent["tokens"], 100)
        self.assertEqual(svc.history[-1]["entropy_mean"], ent["mean"])
        self.assertEqual((line["entropy_mean"], line["entropy_min_window"]), (ent["mean"], ent["min_window"]))

    def test_a_client_asking_for_logprobs_keeps_its_lines(self):
        svc, eng = self.service(ruminating(20), entropy=4)
        done = self.run(svc, {"logprobs": True, "top_logprobs": 2})
        self.assertEqual(eng.sampling_seen[0]["top_logprobs"], 2)       # the client's own K
        self.assertNotIn("thinking_entropy", done)
        self.assertTrue(eng.lp_lines)                                    # not drained by the meter

    def test_not_for_another_engine_class(self):
        svc, eng = self.service(ruminating(20), entropy=4)
        done = self.run(svc, patched=False)                              # a MockEngine-like engine is not StrataEngine
        self.assertNotIn("logprobs", eng.sampling_seen[0])
        self.assertNotIn("thinking_entropy", done)

    def test_close_needs_low_entropy_when_entropy_below_is_set(self):
        close = {"threshold": 0.3, "entropy_below": 0.5}
        svc, eng = self.service(ruminating(100), LPEngine, close=close)          # repeats AND sure of itself: closes
        self.run(svc)
        self.assertEqual(len(eng.calls), 2)
        svc, eng = self.service(ruminating(100), UnsureEngine, close=close)      # repeats but unsure (ln 4): kept open
        done = self.run(svc)
        self.assertEqual(len(eng.calls), 1)
        self.assertFalse(done["rumination"]["closed"])
        self.assertGreater(done["thinking_entropy"]["min_window"], 1.0)

    def test_entropy_below_without_a_meter_never_closes(self):
        svc, eng = self.service(ruminating(100), LPEngine, close={"threshold": 0.3, "entropy_below": 0.5}, entropy=None)
        self.run(svc)
        self.assertEqual(len(eng.calls), 1)

    def test_status_page_shows_the_switch(self):
        svc, _ = self.service("x", entropy=6)
        self.assertEqual(svc.pancake_status()["switches"]["server"]["rumination_entropy"], 6)


if __name__ == "__main__":
    unittest.main()
