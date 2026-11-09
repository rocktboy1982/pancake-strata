"""serve/test_context_dedup.py - the ContextPilot port (no process, no GPU):

- serve/context_dedup.py, block dedup (LOSSY opt-in): what is replaced, what never is, and that it is prefix-stable
  (a conversation that grows by appending keeps every earlier message's rewritten text: the conversation cache hits);
- Service._deduped / the request and config switches (off by default: the messages are the very same list);
- STRATA_ADMIT_CACHE_AWARE (exact): new --batch requests are admitted by the tokens they still have to read.

    python -m unittest serve.test_context_dedup -v
"""
from __future__ import annotations

import random
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server  # noqa: E402
from serve import context_dedup as cd  # noqa: E402
from serve.test_admit_order import engine_without_process  # noqa: E402


def code_file(name: str, n: int, start: int = 1, prefix: str = "opencode") -> str:
    """A file as a read tool shows it, with line numbers (opencode `00012| `, Claude Code `    12→`)."""
    lines = [f"def {name}_{i}(x):  # line {i} of {name}, padded so a block is long enough" for i in range(n)]
    if prefix == "opencode":
        return "\n".join(f"{start + i:05d}| {t}" for i, t in enumerate(lines))
    return "\n".join(f"{start + i:6d}→{t}" for i, t in enumerate(lines))


def convo(file_text: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a coding agent.\n" * 3},
        {"role": "user", "content": "read a.py twice"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read", "arguments": {}}}]},
        {"role": "tool", "content": file_text},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read", "arguments": {}}}]},
        {"role": "tool", "content": file_text},
    ]


class DedupTest(unittest.TestCase):
    def test_a_tool_result_sent_twice_becomes_one_reference(self):
        f = code_file("a", 120)
        out, res = cd.dedup_messages(convo(f))
        self.assertEqual(out[3]["content"], f)                     # the first copy stays whole
        self.assertIn("identical to the earlier read result", out[5]["content"])
        self.assertLess(len(out[5]["content"]), 120)
        self.assertEqual(res.results_deduped, 1)
        self.assertGreater(res.chars_saved, len(f) - 200)

    def test_moved_lines_are_replaced_block_by_block(self):
        """The same code read again at other line numbers (an edit above it): the line-number prefix is ignored."""
        first = code_file("a", 120)
        again = code_file("b", 6) + "\n" + code_file("a", 120, start=7)
        msgs = convo(first)
        msgs[5] = dict(msgs[5], content=again)
        out, res = cd.dedup_messages(msgs)
        self.assertEqual(out[3]["content"], first)
        self.assertGreater(res.blocks_deduped, 0)
        self.assertIn("identical to earlier read result", out[5]["content"])
        self.assertIn("def b_0", out[5]["content"])                # the new lines stay
        self.assertLess(len(out[5]["content"]), len(again) // 2)

    def test_claude_code_line_prefix(self):
        first = code_file("a", 120, prefix="cc")
        again = code_file("a", 120, start=31, prefix="cc")
        msgs = convo(first)
        msgs[5] = dict(msgs[5], content=again)
        out, res = cd.dedup_messages(msgs)
        self.assertGreater(res.blocks_deduped, 0)

    def test_system_and_assistant_turns_are_never_rewritten(self):
        f = code_file("a", 120)
        msgs = [{"role": "system", "content": f},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": f},             # the model quoted it: the cache holds it as written
                {"role": "user", "content": f}]
        out, res = cd.dedup_messages(msgs)
        self.assertEqual(out[0]["content"], f)
        self.assertEqual(out[2]["content"], f)
        self.assertNotEqual(out[3]["content"], f)
        self.assertIn("identical to earlier system prompt", out[3]["content"])

    def test_short_and_unique_content_is_untouched(self):
        msgs = convo(code_file("a", 2))
        out, res = cd.dedup_messages(msgs)
        self.assertEqual(out, msgs)
        self.assertEqual(res.chars_saved, 0)
        unique = [{"role": "user", "content": code_file("a", 80)}, {"role": "tool", "content": code_file("z", 80)}]
        out, res = cd.dedup_messages(unique)
        self.assertEqual(out, unique)

    def test_input_not_changed(self):
        msgs = convo(code_file("a", 120))
        snapshot = [dict(m) for m in msgs]
        cd.dedup_messages(msgs)
        self.assertEqual(msgs, snapshot)

    def test_json_wrapped_result_keeps_its_wrapper(self):
        import json
        f = code_file("a", 120)
        msgs = convo(f)
        msgs[5] = dict(msgs[5], content=json.dumps({"path": "a.py", "content": code_file("q", 5) + "\n" + f}))
        out, res = cd.dedup_messages(msgs)
        obj = json.loads(out[5]["content"])
        self.assertEqual(obj["path"], "a.py")
        self.assertIn("identical to earlier", obj["content"])

    def test_image_parts_are_kept(self):
        f = code_file("a", 120)
        msgs = convo(f)
        msgs[5] = dict(msgs[5], content=[{"type": "image", "source": "x.png"},
                                         {"type": "text", "text": code_file("q", 5) + "\n" + f}])
        out, res = cd.dedup_messages(msgs)
        self.assertEqual(out[5]["content"][0], {"type": "image", "source": "x.png"})
        self.assertIn("identical to earlier", out[5]["content"][1]["text"])

    def test_prefix_stable_as_the_conversation_grows(self):
        """Message k's rewritten text depends only on messages 0..k: dedup(history[:n]) is a prefix of dedup(history).
        That is what keeps the engine's exact prefix cache (the only exact reuse for the GDN layers) hitting."""
        rng = random.Random(7)
        files = {n: code_file(n, rng.randint(20, 90)) for n in "abcdefg"}
        history = [{"role": "system", "content": "agent rules\n" * 30 + files["g"]}]
        for turn in range(40):
            kind = rng.random()
            name = rng.choice("abcdefg")
            if kind < 0.4:
                history.append({"role": "assistant", "content": "",
                                "tool_calls": [{"function": {"name": "read", "arguments": {"p": name}}}]})
                body = files[name]
                if rng.random() < 0.5:                       # read again after an edit: lines moved, one line new
                    lines = body.split("\n")
                    lines.insert(rng.randint(0, len(lines)), f"00000| new line {turn} {'x' * 50}")
                    body = "\n".join(lines)
                history.append({"role": "tool", "content": body})
            elif kind < 0.7:
                history.append({"role": "user", "content": f"turn {turn}: look at this\n" + files[name]})
            else:
                history.append({"role": "assistant", "content": f"answer {turn}\n" + files[name][:400]})
        full, res = cd.dedup_messages(history)
        self.assertGreater(res.chars_saved, 0)
        for n in range(1, len(history) + 1):
            part, _ = cd.dedup_messages(history[:n])
            self.assertEqual(part, full[:n], f"prefix of {n} messages changed")

    def test_setting(self):
        self.assertFalse(cd.dedup_setting({}, env={}))
        self.assertTrue(cd.dedup_setting({"context_dedup": True}, env={}))
        self.assertFalse(cd.dedup_setting({"context_dedup": True}, env={"STRATA_CONTEXT_DEDUP": "0"}))
        self.assertTrue(cd.dedup_setting({}, env={"STRATA_CONTEXT_DEDUP": "1"}))
        with self.assertRaises(ValueError):
            cd.dedup_setting({"context_dedup": "yes"}, env={})


class ServiceDedupTest(unittest.TestCase):
    def service(self, on: bool):
        svc = server.Service.__new__(server.Service)
        svc.status_lock = threading.Lock()
        svc.context_dedup = on
        svc.context_dedup_counts = {"requests": 0, "blocks": 0, "results": 0, "chars_saved": 0}
        return svc

    def test_off_by_default_returns_the_same_list(self):
        svc = self.service(False)
        msgs = convo(code_file("a", 120))
        self.assertIs(svc._deduped(msgs, {}), msgs)
        self.assertIs(svc._deduped(msgs, None), msgs)

    def test_on_and_request_override(self):
        svc = self.service(True)
        msgs = convo(code_file("a", 120))
        out = svc._deduped(msgs, {})
        self.assertNotEqual(out, msgs)
        self.assertEqual(svc.context_dedup_counts["results"], 1)
        self.assertIs(svc._deduped(msgs, {"context_dedup": False}), msgs)
        self.assertNotEqual(self.service(False)._deduped(msgs, {"context_dedup": True}), msgs)
        svc._deduped(msgs, {}, count=False)                     # a token count is not counted
        self.assertEqual(svc.context_dedup_counts["requests"], 1)

    def test_service_default_is_off(self):
        self.assertIn("self.context_dedup = False", Path(server.__file__).read_text())


class CacheAwareAdmitTest(unittest.TestCase):
    def setUp(self):
        self.e = engine_without_process(4)
        self.e.slot_held = [[], list(range(50000)), [], []]

    def entry(self, plen, prompt, t):
        return [plen, t, True, prompt, None]

    def test_least_uncached_first(self):
        now = time.monotonic()
        cold = self.entry(30000, list(range(100000, 130000)), now - 2)        # arrived first, nothing cached
        warm = self.entry(52000, list(range(50000)) + [7, 8] * 1000, now - 1)   # 50000 of 52000 held by slot 1
        self.e.wait_lens = [cold, warm]
        with mock.patch.object(server, "ADMIT_CACHE_AWARE", True), mock.patch.object(server, "ADMIT_MAX_WAIT_S", 60.0), \
                mock.patch.object(server, "ADMIT_SHORT_FIRST", 0):
            self.assertIs(self.e._admit_next(), warm)
            self.assertEqual(self.e._cached_prefix(warm[3]), 50000)
            self.e.slot_busy[1] = True                                 # a busy slot's stream is being overwritten
            self.assertIs(self.e._admit_next(), cold)

    def test_waited_too_long_goes_first(self):
        now = time.monotonic()
        cold = self.entry(30000, list(range(100000, 130000)), now - 5)
        warm = self.entry(52000, list(range(50000)) + [7, 8] * 1000, now - 1)
        self.e.wait_lens = [cold, warm]
        with mock.patch.object(server, "ADMIT_CACHE_AWARE", True), mock.patch.object(server, "ADMIT_MAX_WAIT_S", 3.0), \
                mock.patch.object(server, "ADMIT_SHORT_FIRST", 0):
            self.assertIs(self.e._admit_next(), cold)

    def test_with_short_first_short_means_uncached(self):
        now = time.monotonic()
        chat = self.entry(3000, list(range(200000, 203000)), now - 2)
        warm = self.entry(52000, list(range(50000)) + [7, 8] * 1000, now - 3)  # 2000 uncached: "short" too, earlier
        self.e.wait_lens = [chat, warm]
        with mock.patch.object(server, "ADMIT_CACHE_AWARE", True), mock.patch.object(server, "ADMIT_MAX_WAIT_S", 60.0), \
                mock.patch.object(server, "ADMIT_SHORT_FIRST", 8192):
            self.assertIs(self.e._admit_next(), warm)
        with mock.patch.object(server, "ADMIT_CACHE_AWARE", False), mock.patch.object(server, "ADMIT_SHORT_FIRST", 8192), \
                mock.patch.object(server, "ADMIT_MAX_WAIT_S", 60.0):
            self.assertIs(self.e._admit_next(), chat)                  # as before without the switch

    def test_off_by_default(self):
        if "STRATA_ADMIT_CACHE_AWARE" not in __import__("os").environ:
            self.assertFalse(server.ADMIT_CACHE_AWARE)
        with mock.patch.object(server, "ADMIT_SHORT_FIRST", 0), mock.patch.object(server, "ADMIT_CACHE_AWARE", False):
            self.assertFalse(server.StrataEngine._admit_ordered())


if __name__ == "__main__":
    unittest.main()
