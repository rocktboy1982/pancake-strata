"""serve/test_think_probe.py - MOD_PLAN A19: the engine's learned thinking stop (STRATA_THINK_PROBE prints `TPROBE <n> <score>`,
read into engine.think_probe_hit) closes the thinking the way the soft budget does: at the next sentence or line end, with
the wrap-up and </think>, once.  Without the line nothing changes.

    python -m unittest serve.test_think_probe -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import IM_END, REASONING_WRAP_UP, Service, think_boundary  # noqa: E402
from serve.test_loop_guard import Tok  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def thoughts(n=60):
    return "".join(f"Step {i}: check the value of term {i} against the bound.\n" for i in range(n))


class ProbeEngine:
    """Writes `thoughts`, then </think>, the answer and the end; once the prompt holds the end of thinking (the server's
    wrap-up), the answer and the end.  `fire_at`: after that many tokens of a pass it sets think_probe_hit, as the real
    engine's reader does for a TPROBE line (reset at every pass, as StrataEngine.generate does)."""

    def __init__(self, tok, text, fire_at=None, answer="The answer."):
        self.tok, self.max_context, self.fire_at = tok, 1 << 20, fire_at
        enc = lambda s: tok.encode(s, parse_special=True)   # noqa: E731
        self.thoughts, self.answer, self.end = enc(text), enc(answer), enc(IM_END)
        self.think_end = enc("</think>")[0]
        self.base, self.calls = None, []
        self.think_probe_hit = None

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.think_probe_hit = None
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
            if self.fire_at is not None and i == self.fire_at and self.think_end not in written:
                self.think_probe_hit = (i, 0.97)
            yield t


def run(svc, max_new=20000):
    reasoning, content = "", ""
    for kind, val in svc.run(svc.tok.encode("go"), True, None, max_new, {}, threading.Event()):
        if kind == "event" and val.kind == "reasoning":
            reasoning += val.text
        elif kind == "event" and val.kind == "content":
            content += val.text
    return reasoning, content


class ThinkProbeClose(unittest.TestCase):
    def service(self, text, fire_at=None):
        tok = Tok()
        eng = ProbeEngine(tok, text, fire_at)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        svc.answer_room = False
        return svc, eng

    def test_no_line_no_change(self):
        text = thoughts(40)
        svc, eng = self.service(text)
        with tempfile.TemporaryDirectory() as d:
            svc.turn_log = str(Path(d) / "turns.jsonl")
            reasoning, content = run(svc)
            line = json.loads(Path(svc.turn_log).read_text().splitlines()[-1])
        self.assertEqual(len(eng.calls), 1)
        self.assertEqual(reasoning.strip(), text.strip())
        self.assertEqual(content.strip(), "The answer.")
        self.assertFalse(line["think_probe_closed"])

    def test_line_closes_at_a_sentence_end(self):
        text = thoughts(60)
        svc, eng = self.service(text, fire_at=50)
        with tempfile.TemporaryDirectory() as d:
            svc.turn_log = str(Path(d) / "turns.jsonl")
            reasoning, content = run(svc)
            line = json.loads(Path(svc.turn_log).read_text().splitlines()[-1])
        self.assertEqual(len(eng.calls), 2)
        p1, p2 = eng.calls
        said = svc.tok.decode(p2[len(p1):])
        self.assertTrue(said.endswith(REASONING_WRAP_UP), said[-120:])
        kept = said[:-len(REASONING_WRAP_UP)]
        self.assertTrue(text.startswith(kept))
        self.assertTrue(think_boundary(kept), kept[-40:])
        self.assertLess(len(kept), len(text) // 2)
        self.assertEqual(content.strip(), "The answer.")
        self.assertTrue(svc.history[-1]["think_probe_closed"])
        self.assertTrue(line["think_probe_closed"])

    def test_close_waits_at_most_the_grace(self):
        text = thoughts(60).replace(".\n", " ;; ")      # no sentence or line end anywhere
        svc, eng = self.service(text, fire_at=30)
        svc.reasoning_budget_grace = 25
        run(svc)
        self.assertEqual(len(eng.calls), 2)
        p1, p2 = eng.calls
        said = svc.tok.decode(p2[len(p1):])
        self.assertTrue(said.endswith(REASONING_WRAP_UP))
        kept = svc.tok.encode(said[:-len(REASONING_WRAP_UP)], parse_special=True)
        self.assertLessEqual(len(kept), 30 + 25 + 2)   # the signal, then at most the grace (and the one in flight)


if __name__ == "__main__":
    unittest.main()
