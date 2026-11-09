"""MOD_PLAN 16, STRATA_CACHE_TOOL_BOUNDARY: the server sends Strata's own engine the tool-call and tool-result boundaries
of a Qwen prompt (its GEN line's marks=), computed from the prompt's ids as the template wrote them - no GPU, no engine
(a mock one).  The engine decides which of them it checkpoints (include/strata/program/message_boundary.hpp).

    python -m unittest serve.test_tool_boundary_marks -v
"""
from __future__ import annotations

import os
import sys
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (ByteTokenizer, MockEngine, Service, StrataEngine, qwen_tool_marks,  # noqa: E402
                          tool_boundary_level)

ON = "STRATA_CACHE_TOOL_BOUNDARY"
SYSTEM = "You are a coding agent working in a C++ repository. Answer briefly. " * 30   # ~2,000 byte-tokens
FILE = "".join(f"int f{i}(int x) {{ return x + {i}; }}\n" for i in range(60))
THINK = "I should read the file first, then answer. " * 10


class ToolTokenizer(ByteTokenizer):
    """The byte tokenizer with the Qwen template's reasoning and tool markers as specials matched everywhere (GGUF type 4)."""
    SPECIALS = ByteTokenizer.SPECIALS + ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>",
                                         "</tool_response>"]
    ALWAYS = ("<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>")


def call(i, path="src/a.cpp", reasoning=""):
    return {"role": "assistant", "content": "", "reasoning_content": reasoning,
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": "read_file", "arguments": {"path": path}}}]}


def result(i, text=FILE):
    return {"role": "tool", "tool_call_id": f"c{i}", "content": text}


TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
                                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]


class Pure(unittest.TestCase):
    """qwen_tool_marks() on ids built by hand: T = <|im_start|>, R = <tool_response>, E = </think>."""
    T, R, E, U = 1000, 1001, 1002, 7

    def test_call_and_result_boundaries(self):
        T, R, U = self.T, self.R, self.U
        ids = [T] + [U] * 2000 + [T, U, U] + [U] * 300 + [T, U, 9, R] + [U] * 500 + [T, U, U]
        call_at = ids.index(R) - 3
        res_at = len(ids) - 3
        self.assertEqual(qwen_tool_marks(ids, T, R), [call_at, res_at])
        self.assertEqual(qwen_tool_marks(ids, T, R, least=call_at + 1), [res_at])
        self.assertEqual(qwen_tool_marks(ids, None, R), [])
        self.assertEqual(qwen_tool_marks(ids, T, None), [])
        self.assertEqual(qwen_tool_marks([T] + [U] * 3000 + [T, U], T, R), [])   # no tool result: nothing

    def test_a_tool_response_deep_inside_a_turn_is_not_a_boundary(self):
        T, R, U = self.T, self.R, self.U
        ids = [T] + [U] * 2000 + [T] + [U] * 20 + [R] + [U] * 10 + [T, U]
        self.assertEqual(qwen_tool_marks(ids, T, R), [])

    def test_think_end_only_with_the_id_and_a_long_block_before_the_last_turn(self):
        T, R, E, U = self.T, self.R, self.E, self.U
        ids = ([T] + [U] * 2000 + [T] + [U] * 400 + [E] + [U] * 50 + [T, U, 9, R] + [U] * 200 +
               [T] + [U] * 10 + [E] + [U] * 20 + [T, U, U, E])
        e1 = ids.index(E)
        self.assertNotIn(e1 + 1, qwen_tool_marks(ids, T, R))
        marks = qwen_tool_marks(ids, T, R, E)
        self.assertIn(e1 + 1, marks)                                   # a 400-token block: marked
        self.assertEqual([p for p in marks if ids[p - 1] == E], [e1 + 1])   # the short one and the generation prompt's: not

    def test_level(self):
        self.assertEqual(tool_boundary_level({}), 0)
        self.assertEqual(tool_boundary_level({ON: "1"}), 1)
        self.assertEqual(tool_boundary_level({ON: "2"}), 2)
        self.assertEqual(tool_boundary_level({ON: "9"}), 2)
        self.assertEqual(tool_boundary_level({ON: "x"}), 0)
        self.assertEqual(tool_boundary_level({ON: ""}), 0)


class Prepare(unittest.TestCase):
    """The same through Service.prepare with the real chat template."""

    def setUp(self):
        self.tok = ToolTokenizer()
        self.engine = MockEngine(self.tok, "ok", max_context=1 << 20)
        self.svc = Service(self.engine, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.T, self.R, self.E = self.svc.qwen_tool_ids()

    def prep(self, messages, env=None, tools=TOOLS, **kw):
        req = {"temperature": 0, "_restore": {"marks": [5]}}   # a client's own never reaches the engine
        with unittest.mock.patch.dict(os.environ, env or {}, clear=False):
            if ON not in (env or {}):
                os.environ.pop(ON, None)
            ids, _, _ = self.svc.prepare(messages, tools, kw, max_new=16, req=req)
        return ids, req

    def chat(self, reasoning=""):
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "What does f3 return? Read it."},
                call(1, reasoning=reasoning), result(1)]

    def test_ids_resolved(self):
        self.assertEqual((self.T, self.R, self.E), tuple(256 + ToolTokenizer.SPECIALS.index(s)
                                                         for s in ("<|im_start|>", "<tool_response>", "</think>")))

    def test_off_by_default(self):
        _, req = self.prep(self.chat())
        self.assertNotIn("_restore", req)
        self.assertNotIn("marks=", StrataEngine.sampling_keys(req))

    def test_marks_at_the_call_and_the_result(self):
        ids, req = self.prep(self.chat(), {ON: "1"})
        marks = req["_restore"]["marks"]
        turns = [i for i, t in enumerate(ids) if t == self.T]
        resp = ids.index(self.R)
        call_at = max(i for i in turns if i < resp)                   # <|im_start|>user\n<tool_response>
        self.assertEqual(marks, [call_at, turns[-1]])                  # and the generation prompt's <|im_start|>
        self.assertEqual(self.tok.decode(ids[call_at:resp + 1]), "<|im_start|>user\n<tool_response>")
        self.assertTrue(self.tok.decode(ids[:call_at]).endswith("</tool_call><|im_end|>\n"))
        keys = StrataEngine.sampling_keys(req).split()
        self.assertIn("marks=" + ",".join(map(str, marks)), keys)
        self.assertFalse([k for k in keys if k.startswith(("tail=", "shared="))])

    def test_a_retried_step_shares_the_prompt_up_to_the_call_boundary(self):
        ids, req = self.prep(self.chat(), {ON: "1"})
        call_at = req["_restore"]["marks"][0]
        retry = self.chat()[:-1] + [result(1, "error: file is busy, try again\n")]
        ids2, _ = self.prep(retry, {ON: "1"})
        n = 0
        while n < min(len(ids), len(ids2)) and ids[n] == ids2[n]:
            n += 1
        self.assertGreater(n, call_at)                                 # resumable at the mark
        self.assertLess(n, req["_restore"]["marks"][1])

    def test_parallel_results_and_several_steps(self):
        msgs = self.chat() + [call(2, "src/b.cpp"), result(2), result(3, "second result\n"),
                              {"role": "assistant", "content": "f3 returns x + 3."},
                              {"role": "user", "content": "thanks"}]
        ids, req = self.prep(msgs, {ON: "1"})
        marks = req["_restore"]["marks"]
        self.assertEqual(len(marks), 4)                                # two calls, two results (one grouped turn)
        for p in marks:
            self.assertEqual(ids[p], self.T)
        self.assertEqual(sum(1 for t in ids if t == self.R), 3)

    def test_think_end_with_level_2_only_for_kept_thinking(self):
        msgs = self.chat(reasoning=THINK) + [{"role": "assistant", "content": "x + 3", "reasoning_content": "short"},
                                             {"role": "user", "content": "and f4?"}]
        ids, req = self.prep(msgs, {ON: "1"})
        self.assertFalse([p for p in req["_restore"]["marks"] if ids[p - 1] == self.E])
        ids, req = self.prep(msgs, {ON: "2"})
        ends = [p for p in req["_restore"]["marks"] if ids[p - 1] == self.E]
        self.assertEqual(len(ends), 1)                                 # the long block; not "short"
        self.assertTrue(self.tok.decode(ids[:ends[0]]).endswith(THINK.strip() + "\n</think>"))
        # preserve_thinking false: the template drops earlier thinking, so nothing is marked there
        ids, req = self.prep(msgs, {ON: "2"}, preserve_thinking=False)
        self.assertFalse([p for p in req.get("_restore", {}).get("marks", []) if ids[p - 1] == self.E])

    def test_not_for_the_split_engine_nor_strata_poly_nor_pictures(self):
        for attr in ("split_engine", "poly_engine"):
            with self.subTest(engine=attr):
                setattr(self.engine, attr, True)
                try:
                    _, req = self.prep(self.chat(), {ON: "1"})
                    self.assertNotIn("_restore", req)
                finally:
                    delattr(self.engine, attr)
        req = {}
        with unittest.mock.patch.dict(os.environ, {ON: "1"}):
            self.svc.set_restore_points(req, self.prep(self.chat(), {ON: "1"})[0], True)
        self.assertNotIn("_restore", req)

    def test_short_prompts_get_none(self):
        msgs = [{"role": "user", "content": "hi"}, call(1), result(1, "ok")]
        _, req = self.prep(msgs, {ON: "1"}, tools=None)
        self.assertNotIn("_restore", req)


if __name__ == "__main__":
    unittest.main()
