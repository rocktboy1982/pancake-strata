"""serve/test_agent_loop.py - MOD_PLAN A38, A41, A42, A43 (serve/agent_loop.py): the per-request behaviour counters, the
oldString check, the test-before-done nudge and the externalized tool results.  Mock engines, no GPU.

    python -m unittest serve.test_agent_loop -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import agent_loop as AL  # noqa: E402
from serve import loop_guard as LG  # noqa: E402
from serve.frontend import ChatTemplate, openai_to_messages  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402
from serve.test_loop_guard import ScriptEngine, Server, Tok, call_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = "def parse(x):\n    items = x.split(',')\n    return [i.strip() for i in items]\n"


def numbered(text: str, style: str = "opencode") -> str:
    """A read tool's output: the file's lines with line numbers (opencode's two forms, Claude Code's)."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if style == "opencode":
        body = "\n".join(f"{i + 1:05d}| {ln}" for i, ln in enumerate(lines))
        return f"<file>\n{body}\n\n(End of file - total {len(lines)} lines)\n</file>"
    if style == "new":
        body = "\n".join(f"{i + 1}: {ln}" for i, ln in enumerate(lines))
        return f"<path>/w/p.py</path>\n<type>file</type>\n<content>\n{body}\n</content>"
    return "\n".join(f"{i + 1:6d}→{ln}" for i, ln in enumerate(lines))


class Conv:
    """A Chat Completions conversation: rounds of (name, arguments, result), one call a round."""

    def __init__(self, user="make the parser accept trailing commas"):
        self.msgs = [{"role": "user", "content": user}]
        self.n = 0
        self.names = set()

    def call(self, name, args, result="ok"):
        cid = f"c{self.n}"
        self.n += 1
        self.names.add(name)
        self.msgs.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": cid, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]})
        self.msgs.append({"role": "tool", "tool_call_id": cid, "content": result})
        return self

    def read(self, path, text=SRC, style="opencode"):
        return self.call("read", {"filePath": path}, numbered(text, style))

    def edit(self, path, old, new="x", result="Edit applied successfully."):
        return self.call("edit", {"filePath": path, "oldString": old, "newString": new}, result)

    def bash(self, cmd, result="ok"):
        return self.call("bash", {"command": cmd}, result)

    def user(self, text):
        self.msgs.append({"role": "user", "content": text})
        return self

    def req(self):
        names = sorted(self.names | {"bash", "edit", "read"})
        return {"messages": [dict(m) for m in self.msgs],
                "tools": [{"type": "function", "function": {"name": n, "parameters": {}}} for n in names]}

    def template(self):
        return openai_to_messages(self.req())[0]

    def counters(self):
        return AL.counters(self.template())


class Counters(unittest.TestCase):
    def test_no_tool_call_no_counters(self):
        self.assertIsNone(AL.counters(Conv().template()))
        self.assertIsNone(AL.counters([]))

    def test_kinds_and_streaks_are_exact(self):
        c = (Conv().read("a.py").read("b.py").edit("a.py", "items").edit("b.py", "x")
             .bash("pytest -q").edit("a.py", "i.strip").read("c.py").read("d.py").read("e.py")).counters()
        self.assertEqual((c["tool_calls"], c["rounds"], c["edits"], c["checks"], c["reads"], c["other"]),
                         (9, 9, 3, 1, 5, 0))
        self.assertEqual((c["edits_since_check"], c["reads_since_edit"]), (1, 3))
        self.assertEqual((c["max_edits_since_check"], c["max_reads_since_edit"]), (2, 3))
        self.assertEqual(c["files_edited"], 2)
        self.assertEqual(c["ends_unchecked"], 1)

    def test_it_agrees_with_the_item_33_streaks(self):
        conv = Conv()
        for i in range(5):
            conv.read("a.py").edit("a.py", "items")
        conv.bash("make -j4").edit("a.py", "items").edit("a.py", "items")
        msgs = conv.template()
        b, c = LG.behaviour(msgs, 100, 100), AL.counters(msgs)
        self.assertEqual((c["edits_since_check"], c["reads_since_edit"]), (b.edits, b.reads))

    def test_a_check_ends_the_streak_and_the_turn_starts_over(self):
        c = Conv().edit("a.py", "items").bash("python -m pytest tests").counters()
        self.assertEqual((c["edits_since_check"], c["ends_unchecked"]), (0, 0))
        conv = Conv().edit("a.py", "items").edit("a.py", "i.strip")
        self.assertEqual(conv.counters()["edits"], 2)
        conv.user("now the docs").read("README.md")                # a new user message: the turn's counts start over
        c = conv.counters()
        self.assertEqual((c["edits"], c["reads"], c["tool_calls"], c["edits_since_check"]), (0, 1, 1, 0))
        conv2 = Conv().edit("a.py", "items").user("continue").edit("a.py", "i.strip")   # (a bare "continue" does not)
        self.assertEqual(conv2.counters()["edits"], 2)

    def test_doc_edits_do_not_trigger_the_finish_nudge(self):
        c = Conv().read("README.md", "# t\n").edit("README.md", "# t", "# title").counters()
        self.assertEqual((c["edits_since_check"], c["code_edits_since_check"], c["ends_unchecked"]), (1, 0, 0))
        c = Conv().read("a.py").edit("a.py", "items").edit("NOTES.md", "x").counters()
        self.assertEqual((c["code_edits_since_check"], c["ends_unchecked"]), (1, 1))

    def test_a_request_that_does_not_end_with_a_tool_result_is_not_mid_loop(self):
        conv = Conv().read("a.py").edit("a.py", "items")
        conv.msgs.append({"role": "assistant", "content": "done"})
        self.assertEqual(AL.counters(conv.template())["ends_unchecked"], 0)

    def test_edits_on_a_file_that_was_never_read(self):
        c = Conv().edit("a.py", "items").read("b.py").edit("b.py", "items").counters()
        self.assertEqual(c["edits_unread"], 1)
        # a write created the file: an edit after it is not "unread"
        c = Conv().call("write", {"filePath": "n.py", "content": SRC}).edit("n.py", "items").counters()
        self.assertEqual(c["edits_unread"], 0)
        # the read may come from an earlier user turn of the same conversation
        c = Conv().read("a.py").user("again").edit("a.py", "items").counters()
        self.assertEqual(c["edits_unread"], 0)

    def test_claude_code_names_and_arguments(self):
        conv = Conv()
        conv.call("Read", {"file_path": "/w/a.py"}, numbered(SRC, "cc"))
        conv.call("Edit", {"file_path": "/w/a.py", "old_string": "items = x.split(',')", "new_string": "y"})
        conv.call("Edit", {"file_path": "/w/a.py", "old_string": "nope nope", "new_string": "y"})
        c = conv.counters()
        self.assertEqual((c["edits"], c["edit_old_exact"], c["edit_old_miss"], c["edits_unread"]), (2, 1, 1, 0))


class OldString(unittest.TestCase):
    def test_states(self):
        k = [AL.known_text(numbered(SRC))]
        self.assertEqual(AL.edit_old_state("items = x.split(',')", k), "exact")
        self.assertEqual(AL.edit_old_state("    items = x.split(',')\n    return [i.strip()", k), "exact")
        self.assertEqual(AL.edit_old_state("items  =  x.split(',')", k), "loose")           # only white space differs
        self.assertEqual(AL.edit_old_state("items = x.split(';')", k), "miss")
        self.assertEqual(AL.edit_old_state("items", []), "unknown")                          # never read

    def test_line_numbers_are_not_part_of_the_file(self):
        for style in ("opencode", "new", "cc"):
            text = AL.known_text(numbered(SRC, style))
            self.assertIn("def parse(x):\n    items = x.split(',')\n    return", text, style)
            self.assertNotIn("00001", text)
        # a multi-line oldString crosses the line numbers of the read output
        conv = Conv().read("a.py", style="opencode").edit("a.py", "def parse(x):\n    items = x.split(',')")
        self.assertEqual(conv.counters()["edit_old_exact"], 1)

    def test_miss_loose_unknown_counted_and_the_failure_seen(self):
        fail = "Error: Could not find oldString in the file. It must match exactly, including whitespace."
        c = (Conv().read("a.py").edit("a.py", "return [i.strip() for i in items]").edit("a.py", "return  [i.strip()")
             .edit("a.py", "return [j.strip() for j in items]", result=fail).edit("b.py", "anything")).counters()
        self.assertEqual((c["edit_old_exact"], c["edit_old_loose"], c["edit_old_miss"], c["edit_old_unknown"]),
                         (1, 1, 1, 1))
        self.assertEqual(c["edit_failed"], 1)
        self.assertEqual((c["last_round_old_miss"], c["last_round_failed"]), (0, 0))        # the last round was b.py
        c = Conv().read("a.py").edit("a.py", "return [j.strip() for j in items]", result=fail).counters()
        self.assertEqual((c["last_round_old_miss"], c["last_round_failed"]), (1, 1))

    def test_text_the_conversation_wrote_into_the_file_is_known(self):
        c = (Conv().read("a.py").edit("a.py", "items = x.split(',')", "parts = x.rstrip(',').split(',')")
             .edit("a.py", "parts = x.rstrip(',').split(',')", "z")).counters()
        self.assertEqual((c["edit_old_exact"], c["edit_old_miss"]), (2, 0))
        # a failed edit wrote nothing
        fail = "Error: Could not find oldString in the file."
        c = (Conv().read("a.py").edit("a.py", "items = x.split(',')", "NEWTEXT", result=fail)
             .edit("a.py", "NEWTEXT")).counters()
        self.assertEqual(c["edit_old_miss"], 1)
        # a write replaces what was read before
        c = (Conv().read("a.py").call("write", {"filePath": "a.py", "content": "print(1)\n"})
             .edit("a.py", "items = x.split(',')")).counters()
        self.assertEqual(c["edit_old_miss"], 1)

    def test_a_file_never_read_stays_unknown(self):
        c = Conv().edit("a.py", "items").edit("a.py", "other text").counters()
        self.assertEqual((c["edit_old_unknown"], c["edit_old_miss"], c["edits_unread"]), (2, 0, 2))

    def test_multiedit(self):
        conv = Conv().read("a.py")
        conv.call("multiedit", {"filePath": "a.py", "edits": [{"oldString": "items", "newString": "q"},
                                                              {"oldString": "not there", "newString": "q"}]})
        c = conv.counters()
        self.assertEqual((c["edits"], c["edit_old_exact"], c["edit_old_miss"]), (1, 1, 1))

    def test_settings(self):
        self.assertFalse(AL.settings({}).active)
        s = AL.settings({"STRATA_EDIT_OLDSTRING_CHECK": "note", "STRATA_TEST_BEFORE_DONE": "1",
                         "STRATA_AGENT_COUNTERS": "1", "STRATA_EXTERNALIZE_TOKENS": "2000",
                         "STRATA_EXTERNALIZE_DIR": "/tmp/x-ext"})
        self.assertEqual((s.oldstring, s.finish_nudge, s.counters, s.externalize_tokens, s.externalize_dir),
                         ("note", True, True, 2000, "/tmp/x-ext"))
        self.assertTrue(s.active and s.needs_counters)
        self.assertEqual(AL.settings({"STRATA_EDIT_OLDSTRING_CHECK": "1"}).oldstring, "log")
        self.assertEqual(AL.settings({"STRATA_EXTERNALIZE_TOKENS": "900"}).needs_counters, False)
        for bad in ({"STRATA_EDIT_OLDSTRING_CHECK": "maybe"}, {"STRATA_EXTERNALIZE_TOKENS": "-1"},
                    {"STRATA_EXTERNALIZE_TOKENS": "500"}):                  # (head+tail 800 >= 500)
            with self.assertRaises(ValueError):
                AL.settings(bad)


def big(n_lines: int, tag: str = "line") -> str:
    return "\n".join(f"{tag} {i:05d} some output of the command, a bit long to be a log line" for i in range(n_lines))


class Externalize(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.cfg = AL.settings({"STRATA_EXTERNALIZE_TOKENS": "1000", "STRATA_EXTERNALIZE_DIR": self.dir.name,
                                "STRATA_EXTERNALIZE_HEAD_TOKENS": "100", "STRATA_EXTERNALIZE_TAIL_TOKENS": "100"})

    def tearDown(self):
        self.dir.cleanup()

    def test_a_long_result_is_stored_and_excerpted(self):
        text = big(600)
        conv = Conv().bash("git diff", text).bash("ls", "a b c")
        msgs = conv.template()
        ext = AL.externalize(msgs, self.cfg)
        self.assertEqual((ext.count, ext.new_files), (1, 1))
        got = ext.messages[2]["content"]
        self.assertTrue(got.startswith("line 00000"))
        self.assertTrue(got.rstrip().endswith("line 00599 some output of the command, a bit long to be a log line"))
        self.assertIn(AL.EXTERNALIZED, got)
        self.assertLess(len(got), len(text) // 5)
        path = Path(got.split("saved in the file ")[1].split(" - read it")[0])
        self.assertEqual(path.read_text(), text)                        # the whole result, exactly
        self.assertEqual(oct(path.parent.stat().st_mode & 0o777), "0o700")
        # the client's messages and the short result are untouched
        self.assertEqual(msgs[2]["content"], text)
        self.assertEqual(ext.messages[4]["content"], "a b c")
        self.assertEqual(ext.chars_saved, len(text) - len(got))

    def test_deterministic_and_idempotent_so_the_prefix_stays_byte_stable(self):
        conv = Conv().bash("cat build.log", big(700))
        first = AL.externalize(conv.template(), self.cfg)
        conv.bash("make").read("a.py")                                    # the next request of the conversation
        second = AL.externalize(conv.template(), self.cfg)
        self.assertEqual(second.new_files, 0)                             # the file exists: nothing written again
        self.assertEqual(first.messages[2]["content"], second.messages[2]["content"])
        again = AL.externalize(second.messages, self.cfg)                 # an excerpt is never cut again
        self.assertEqual(again.count, 0)

    def test_what_is_left_alone(self):
        conv = Conv().read("a.py", big(900)).call("bash", {"command": "ls"}, big(900))
        ext = AL.externalize(conv.template(), self.cfg)
        self.assertEqual(ext.count, 1)                                    # a file read is a file: skipped by default
        self.assertEqual(ext.messages[2]["content"], conv.template()[2]["content"])
        cfg = AL.settings({"STRATA_EXTERNALIZE_TOKENS": "1000", "STRATA_EXTERNALIZE_DIR": self.dir.name,
                           "STRATA_EXTERNALIZE_SKIP": "", "STRATA_EXTERNALIZE_HEAD_TOKENS": "100",
                           "STRATA_EXTERNALIZE_TAIL_TOKENS": "100"})
        self.assertEqual(AL.externalize(conv.template(), cfg).count, 2)
        # at the limit, not over it
        short = Conv().bash("ls", "x" * (1000 * AL.CHARS_PER_TOKEN))
        self.assertEqual(AL.externalize(short.template(), self.cfg).count, 0)
        # off: the same list back
        msgs = conv.template()
        self.assertIs(AL.externalize(msgs, AL.Settings()).messages, msgs)

    def test_an_unwritable_directory_leaves_the_result(self):
        blocker = Path(self.dir.name) / "file"
        blocker.write_text("x")
        cfg = self.cfg._replace(externalize_dir=str(blocker / "sub"))      # (a path below a file)
        msgs = Conv().bash("cat x", big(700)).template()
        ext = AL.externalize(msgs, cfg)
        self.assertEqual((ext.count, ext.messages[2]["content"]), (0, msgs[2]["content"]))


class ServerSide(unittest.TestCase):
    """The wiring in Service.prepare / run / the turn log, against the mock engines."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def svc(self, engine, **env):
        svc = Server().service(engine)
        svc.agent_loop = AL.settings(env)
        return svc

    def prepare(self, svc, conv):
        msgs, tools, kw = openai_to_messages(conv.req())
        req = {}
        ids, thinking, max_new = svc.prepare(msgs, tools, kw, 2000, req=req)
        return req, ids, thinking, max_new, tools

    def test_default_is_untouched(self):
        svc = self.svc(ScriptEngine(Tok(), "done"))
        self.assertFalse(svc.agent_loop.active)
        req, *_ = self.prepare(svc, Conv().read("a.py").edit("a.py", "items"))
        self.assertFalse([k for k in req if k.startswith("_agent") or k == "_loop_finish"])
        self.assertNotIn("agent_loop", svc.metrics())
        self.assertNotIn("agent_loop", svc.prometheus())

    def test_counters_go_to_the_turn_log_only_when_asked(self):
        for on in (True, False):
            svc = self.svc(ScriptEngine(Tok(), "done"), STRATA_AGENT_COUNTERS="1" if on else "")
            svc.turn_log = str(Path(self.tmp.name) / f"turns-{on}.jsonl")
            svc.agent_loop = svc.agent_loop._replace(oldstring="log")      # (counted either way)
            conv = Conv().read("a.py").edit("a.py", "nope nope").edit("b.py", "x")
            req, ids, thinking, max_new, tools = self.prepare(svc, conv)
            list(svc.run(ids, thinking, tools, max_new, req, threading.Event()))
            rec = json.loads(Path(svc.turn_log).read_text().splitlines()[-1])
            if on:
                self.assertEqual((rec["agent"]["edits"], rec["agent"]["edits_unread"], rec["agent"]["edit_old_miss"],
                                  rec["agent"]["reads"]), (2, 1, 1, 1))
                self.assertEqual(svc.metrics()["agent_loop"]["oldstring_flagged"], 0)   # (the last round was b.py)
            else:
                self.assertNotIn("agent", rec)

    def test_oldstring_log_counts_and_note_mode_writes_one_note(self):
        fail = "Error: Could not find oldString in the file. It must match exactly."
        conv = Conv().read("a.py").edit("a.py", "return [j for j in k]", result=fail)
        svc = self.svc(ScriptEngine(Tok(), "x"), STRATA_EDIT_OLDSTRING_CHECK="log")
        seen = []
        encode = svc.encode_prompt
        svc.encode_prompt = lambda messages, *a, **k: seen.append(messages) or encode(messages, *a, **k)
        self.prepare(svc, conv)
        self.assertEqual(seen[-1][-1]["role"], "tool")                 # log: the prompt as sent
        self.assertEqual(svc.agent_loop_counts["oldstring_flagged"], 1)
        self.assertIn("strata:agent_loop_oldstring_flagged_total 1", svc.prometheus())
        svc = self.svc(ScriptEngine(Tok(), "x"), STRATA_EDIT_OLDSTRING_CHECK="note")
        seen = []
        encode = svc.encode_prompt
        svc.encode_prompt = lambda messages, *a, **k: seen.append(messages) or encode(messages, *a, **k)
        self.prepare(svc, conv)
        self.assertEqual(seen[-1][-1], {"role": "user", "content": AL.oldstring_note({})})
        self.assertEqual(svc.agent_loop_counts["oldstring_notes"], 1)
        # an edit that worked (the old text was found) gets no note
        ok = Conv().read("a.py").edit("a.py", "items = x.split(',')")
        seen.clear()
        self.prepare(svc, ok)
        self.assertEqual(seen[-1][-1]["role"], "tool")

    def test_finish_nudge_continues_the_answer_once(self):
        # the model ends its turn right after an edit: it is told once to check, and runs the tests
        eng = ScriptEngine(Tok(), "Done - the parser accepts trailing commas.", call_text("pytest -q"))
        svc = self.svc(eng, STRATA_TEST_BEFORE_DONE="1")
        conv = Conv().read("a.py").edit("a.py", "items = x.split(',')")
        req, ids, thinking, max_new, tools = self.prepare(svc, conv)
        self.assertIn("_loop_finish", req)
        events = [x for k, x in svc.run(ids, thinking, tools, max_new, req, threading.Event()) if k == "event"]
        calls = [e.call.arguments["command"] for e in events if e.kind == "tool_call"]
        self.assertEqual(calls, ["pytest -q"])
        self.assertEqual(len(eng.prompts), 2)
        said = svc.tok.decode(eng.prompts[1][len(eng.prompts[0]):])
        self.assertIn(AL.FINISH_NOTE, said)
        self.assertEqual(eng.prompts[1][:len(ids)], list(ids))             # the request's prompt is the prefix
        self.assertEqual(svc.agent_loop_counts["finish_nudges"], 1)
        # a second end of turn in the same answer is not nudged again
        eng = ScriptEngine(Tok(), "Done - the parser accepts trailing commas.", "Still done.")
        svc = self.svc(eng, STRATA_TEST_BEFORE_DONE="1")
        req, ids, thinking, max_new, tools = self.prepare(svc, conv)
        list(svc.run(ids, thinking, tools, max_new, req, threading.Event()))
        self.assertEqual((len(eng.prompts), svc.agent_loop_counts["finish_nudges"]), (2, 1))

    def test_no_finish_nudge_when_checked_or_off_or_not_mid_loop(self):
        eng = ScriptEngine(Tok(), "Done.")
        svc = self.svc(eng, STRATA_TEST_BEFORE_DONE="1")
        for conv in (Conv().read("a.py").edit("a.py", "items").bash("pytest -q"),       # tested after the edit
                     Conv().read("a.py").read("b.py"),                                  # no edit
                     Conv().read("README.md", "# t\n").edit("README.md", "# t")):       # docs only
            req, *_ = self.prepare(svc, conv)
            self.assertNotIn("_loop_finish", req)
        off = self.svc(ScriptEngine(Tok(), "Done."))
        req, *_ = self.prepare(off, Conv().read("a.py").edit("a.py", "items"))
        self.assertNotIn("_loop_finish", req)
        # an answer that makes a call is not nudged
        eng = ScriptEngine(Tok(), call_text("pytest -q"))
        svc = self.svc(eng, STRATA_TEST_BEFORE_DONE="1")
        req, ids, thinking, max_new, tools = self.prepare(svc, Conv().read("a.py").edit("a.py", "items"))
        list(svc.run(ids, thinking, tools, max_new, req, threading.Event()))
        self.assertEqual((len(eng.prompts), svc.agent_loop_counts["finish_nudges"]), (1, 0))

    def test_externalize_in_the_prompt_only(self):
        d = str(Path(self.tmp.name) / "ext")
        svc = self.svc(ScriptEngine(Tok(), "x"), STRATA_EXTERNALIZE_TOKENS="1000", STRATA_EXTERNALIZE_DIR=d,
                       STRATA_EXTERNALIZE_HEAD_TOKENS="100", STRATA_EXTERNALIZE_TAIL_TOKENS="100")
        seen = []
        encode = svc.encode_prompt
        svc.encode_prompt = lambda messages, *a, **k: seen.append(messages) or encode(messages, *a, **k)
        conv = Conv().bash("cat build.log", big(700))
        msgs, tools, kw = openai_to_messages(conv.req())
        original = msgs[2]["content"]
        svc.prepare(msgs, tools, kw, 2000, req={})
        self.assertIn(AL.EXTERNALIZED, seen[-1][2]["content"])
        self.assertEqual(msgs[2]["content"], original)                    # the caller's list is not changed
        self.assertEqual(svc.agent_loop_counts["externalized"], 1)
        self.assertEqual(len(list(Path(d).iterdir())), 1)
        self.assertEqual(svc.metrics()["agent_loop"]["externalize_tokens"], 1000)
        svc.prepare(openai_to_messages(conv.req())[0], tools, kw, 2000, req={})
        self.assertEqual(svc.agent_loop_counts["externalized"], 1)        # the second request wrote nothing new
        self.assertEqual(svc.agent_loop_counts["externalized_files"], 2)


if __name__ == "__main__":
    unittest.main()
