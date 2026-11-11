"""serve/park_divergence.py (MOD_PLAN 17.0, M3): the divergence line of STRATA_PARK_DIVERGENCE_LOG."""
import json
import unittest

from serve import park_divergence as pd


def dec(ids):
    return "".join(chr(97 + (t % 26)) for t in ids)   # a toy tokenizer: id -> a letter


class Pure(unittest.TestCase):
    def test_off_by_default(self):
        self.assertFalse(pd.enabled({}))
        self.assertFalse(pd.enabled({"STRATA_PARK_DIVERGENCE_LOG": "0"}))
        self.assertTrue(pd.enabled({"STRATA_PARK_DIVERGENCE_LOG": "1"}))
        self.assertIsNone(pd.tracker_from_env({}))
        self.assertIsNotNone(pd.tracker_from_env({"STRATA_PARK_DIVERGENCE_LOG": "1"}))

    def test_chars_env(self):
        self.assertEqual(pd.text_chars({}), pd.DEFAULT_CHARS)
        self.assertEqual(pd.text_chars({"STRATA_PARK_DIVERGENCE_CHARS": "0"}), 0)
        self.assertEqual(pd.text_chars({"STRATA_PARK_DIVERGENCE_CHARS": "-3"}), 0)
        self.assertEqual(pd.text_chars({"STRATA_PARK_DIVERGENCE_CHARS": "abc"}), pd.DEFAULT_CHARS)

    def test_first_divergence(self):
        self.assertEqual(pd.first_divergence([1, 2, 3], [1, 2, 9]), 2)
        self.assertEqual(pd.first_divergence([1, 2, 3], [1, 2, 3, 4]), 3)   # live is a prefix: no divergence
        self.assertEqual(pd.first_divergence([1, 2, 3], [1, 2]), 2)         # prompt is a prefix of live
        self.assertEqual(pd.first_divergence([], [1]), 0)

    def test_extension_is_not_logged(self):
        self.assertIsNone(pd.record([1, 2, 3], [1, 2, 3, 4, 5], 2, 3, dec))

    def test_record_in_the_reply(self):
        prompt = list(range(100, 140))                 # 40 tokens
        reply = [3, 4, 5, 6, 7, 8, 9, 10]              # 8 generated
        live = prompt + reply
        new = prompt + [3, 4, 99, 6, 7, 8, 9, 10] + [200, 201]   # the reply rendered again, one token different
        rec = pd.record(live, new, len(prompt), 40, dec)
        self.assertEqual(rec["diverge_at"], 42)
        self.assertEqual(rec["reply_offset"], 2)
        self.assertEqual(rec["where"], "reply-start")
        self.assertEqual(rec["prev_gen_len"], 8)
        self.assertEqual(rec["live_len"], 48)
        self.assertEqual(rec["prompt_len"], 50)
        self.assertEqual(rec["resume"], 40)
        self.assertEqual(rec["live_ids"], live[38:42 + 12])
        self.assertEqual(rec["new_ids"][4], 99)
        self.assertIn("live_text", rec)
        self.assertNotEqual(rec["live_text"], rec["new_text"])

    def test_where_classes(self):
        self.assertEqual(pd.classify(10, 100, 150, 160), "history")
        self.assertEqual(pd.classify(100, 100, 150, 160), "reply-start")
        self.assertEqual(pd.classify(102, 100, 150, 160), "reply-start")
        self.assertEqual(pd.classify(120, 100, 150, 160), "reply-middle")
        self.assertEqual(pd.classify(148, 100, 150, 160), "reply-end")
        self.assertEqual(pd.classify(120, 100, 150, 120), "prompt-is-prefix")

    def test_ids_only_when_chars_zero(self):
        rec = pd.record([1, 2, 3, 4], [1, 2, 9, 9], 2, 0, dec, chars=0)
        self.assertNotIn("live_text", rec)
        self.assertNotIn("new_text", rec)
        self.assertIn("live_ids", rec)

    def test_text_is_cut(self):
        live = list(range(300))
        new = list(range(100)) + [999] * 300
        rec = pd.record(live, new, 50, 0, lambda ids: "x" * 500, chars=20)
        self.assertEqual(len(rec["live_text"]), 23)
        self.assertTrue(rec["live_text"].endswith("..."))

    def test_decoder_failure_keeps_ids(self):
        def bad(ids):
            raise ValueError("no")
        rec = pd.record([1, 2, 3], [1, 9], 1, 0, bad)
        self.assertEqual(rec["text_error"], "ValueError")
        self.assertNotIn("live_text", rec)

    def test_reasoning_guess(self):
        tok = {1: "<think>", 2: "the plan is to read", 3: "</think>", 4: "answer", 5: "<think>\n\n</think>"}
        dec2 = lambda ids: "".join(tok.get(t, "?") for t in ids)
        prompt = [10, 11, 12]
        live = prompt + [2, 3, 4]                      # reply: reasoning, </think>, answer
        dropped = prompt + [5, 4] + [30]               # rendered again without the reasoning
        rec = pd.record(live, dropped, 3, 0, dec2)
        self.assertTrue(rec["reply_has_think_close"])
        self.assertTrue(rec["reasoning_dropped_guess"])
        kept = prompt + [2, 3, 4, 77]                  # the reply rendered as generated, one extra token
        self.assertIsNone(pd.record(live, kept, 3, 0, dec2))

    def test_format_line_is_one_json_line(self):
        rec = pd.record([1, 2, 3], [1, 9], 1, 0, dec)
        line = pd.format_line(rec)
        self.assertTrue(line.startswith("[strata] park divergence "))
        self.assertNotIn("\n", line)
        self.assertEqual(json.loads(line.split("park divergence ", 1)[1])["diverge_at"], 1)


class Tracker(unittest.TestCase):
    def make(self):
        out = []
        return pd.DivergenceTracker(decode=dec, emit=out.append), out

    def run_request(self, tr, key, ids, reply, reused):
        tr.begin(key, ids)
        got = tr.resumed(reused)
        for t in reply:
            tr.token(t)
        tr.end()
        return got

    def test_first_request_logs_nothing(self):
        tr, out = self.make()
        self.assertIsNone(self.run_request(tr, 1, [1, 2, 3], [4, 5], 0))
        self.assertEqual(out, [])

    def test_continuation_logs_nothing_and_divergence_logs_once(self):
        tr, out = self.make()
        self.run_request(tr, 1, [1, 2, 3], [4, 5], 0)
        self.assertIsNone(self.run_request(tr, 1, [1, 2, 3, 4, 5, 6], [7], 4))     # extends prompt + reply
        self.assertEqual(out, [])
        rec = self.run_request(tr, 1, [1, 2, 3, 4, 5, 6, 8], [9], 6)               # the reply 7 became 8
        self.assertEqual(rec["diverge_at"], 6)
        self.assertEqual(rec["where"], "reply-start")
        self.assertEqual(len(out), 1)
        self.assertEqual(tr.logged, 1)

    def test_a_new_engine_run_is_not_compared(self):
        tr, out = self.make()
        self.run_request(tr, 1, [1, 2, 3], [4, 5], 0)
        self.assertIsNone(self.run_request(tr, 2, [9, 9], [], 0))                    # another process: its state is empty
        self.assertEqual(out, [])

    def test_forget_clears_the_comparison(self):
        tr, out = self.make()
        self.run_request(tr, 1, [1, 2, 3], [4, 5], 0)
        tr.forget()
        self.assertIsNone(self.run_request(tr, 1, [7, 7, 7], [], 0))
        self.assertEqual(out, [])

    def test_side_request_then_back(self):
        tr, out = self.make()
        self.run_request(tr, 1, list(range(50)), [60, 61], 0)
        self.run_request(tr, 1, [900, 901, 902], [903], 0)                           # a side request: another conversation
        self.assertEqual(len(out), 1)
        self.assertEqual(json.loads(out[0].split("park divergence ", 1)[1])["where"], "history")
        self.run_request(tr, 1, list(range(50)) + [60, 61, 62], [63], 50)            # back: compared with the side request
        self.assertEqual(len(out), 2)

    def test_an_error_turns_it_off_and_never_raises(self):
        tr, out = self.make()
        tr.decode = None
        tr.emit = lambda line: (_ for _ in ()).throw(RuntimeError("log full"))
        self.run_request(tr, 1, [1, 2, 3], [4], 0)
        self.assertIsNone(self.run_request(tr, 1, [1, 9], [], 0))
        self.assertTrue(tr.broken)
        tr.begin(1, [1])                                                             # still callable, still silent
        tr.token(1)
        tr.end()

    def test_attach_decoder(self):
        class Eng:
            divergence = pd.DivergenceTracker()

        class Tok:
            def decode(self, ids):
                return "T" + str(len(ids))
        eng = Eng()
        pd.attach_decoder(eng, Tok())
        self.assertEqual(eng.divergence.decode([1, 2]), "T2")
        pd.attach_decoder(object(), Tok())                                           # an engine without a tracker: nothing happens


class EnginePath(unittest.TestCase):
    """StrataEngine.generate feeds the tracker: the prompt, the RESUME line and the streamed tokens."""

    def engine(self, tracker):
        import io
        import queue
        import types
        from serve.server import StrataEngine
        e = StrataEngine.__new__(StrataEngine)
        e.proc = types.SimpleNamespace(stdin=io.StringIO(), poll=lambda: None)
        e.lines, e.can_stop, e.max_context, e.silence_s = queue.Queue(), False, 262144, 0
        e.divergence = tracker
        return e

    def run_once(self, e, ids, *lines):
        import threading
        for line in lines:
            e.lines.put(line)
        return [t for t in e.generate(ids, 8, {}, threading.Event()) if t is not None]

    def test_a_turn_that_rerenders_the_reply_is_logged(self):
        out = []
        e = self.engine(pd.DivergenceTracker(decode=dec, emit=out.append))
        self.assertEqual(self.run_once(e, [1, 2, 3], "RESUME 0", "T 10", "T 11", "DONE 3 3 0 2 stop 0 0 0"), [10, 11])
        self.assertEqual(out, [])
        # the next prompt carries the reply as 10, 12 instead of 10, 11
        self.run_once(e, [1, 2, 3, 10, 12, 4], "RESUME 3", "T 13", "DONE 6 3 0 1 stop 0 0 0")
        self.assertEqual(len(out), 1)
        rec = json.loads(out[0].split("park divergence ", 1)[1])
        self.assertEqual((rec["diverge_at"], rec["live_len"], rec["resume"], rec["where"]), (4, 5, 3, "reply-start"))
        # an exact continuation of 1,2,3,10,12,4 + 13 logs nothing more
        self.run_once(e, [1, 2, 3, 10, 12, 4, 13, 5], "RESUME 7", "T 14", "DONE 8 1 0 1 stop 0 0 0")
        self.assertEqual(len(out), 1)

    def test_off_is_no_tracker_and_nothing_printed(self):
        e = self.engine(None)
        self.assertEqual(self.run_once(e, [1, 2], "RESUME 0", "T 7", "DONE 2 2 0 1 stop 0 0 0"), [7])
        self.assertEqual(self.run_once(e, [9, 9], "RESUME 0", "T 8", "DONE 2 2 0 1 stop 0 0 0"), [8])

    def test_picture_requests_are_not_followed(self):
        import threading
        out = []
        e = self.engine(pd.DivergenceTracker(decode=dec, emit=out.append))
        self.run_once(e, [1, 2, 3], "RESUME 0", "T 10", "DONE 3 3 0 1 stop 0 0 0")
        for line in ("RESUME 0", "T 5", "DONE 3 3 0 1 stop 0 0 0"):
            e.lines.put(line)
        list(e.generate([7, 7, 7], 8, {}, threading.Event(), embeddings="/tmp/none.emb"))
        self.run_once(e, [1, 2, 3, 10, 11], "RESUME 3", "T 1", "DONE 5 2 0 1 stop 0 0 0")
        self.assertEqual(out, [])                                                     # the picture turn broke the chain


if __name__ == "__main__":
    unittest.main()
