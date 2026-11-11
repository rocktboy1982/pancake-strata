"""Tests for restore_check.py's pure parts (no server, no GPU): the scripted conversation's structure, the comparison, and
the refusal of a run in which the cache never served a hit."""
import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
