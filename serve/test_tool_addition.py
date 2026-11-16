"""STRATA_TOOL_ADDITION=1 and STRATA_TOOLSEARCH_PASSTHROUGH=1 (both opt-in; LEADS.md Part H: upstream #5952 and fork
adambenhassen 54ef684d05).

Claude Code sends some tools (MCP servers that connect after the session started) as `tool_addition` blocks in a
`role: "system"` message inside `messages`, not in the top-level `tools`; and with tool search on it runs tools it never
declared by name.  Off (the default), both are exactly as before: the block is left out of the prompt and a call for an
undeclared name stays text.

    python -m unittest serve.test_tool_addition -v
"""
import copy
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import OutputParser, anthropic_to_messages, anthropic_tools  # noqa: E402

MCP = "mcp__docs__batch"
READ = {"name": "Read", "description": "Read a file.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}}}
DEFINITION = {"name": MCP, "description": "Batch edits.",
              "input_schema": {"type": "object", "properties": {"source": {"type": "string"},
                                                                "dry_run": {"type": "boolean"}}}}
SEARCH = {"name": "ToolSearch", "description": "Load deferred tools.", "input_schema": {"type": "object"}}


def addition(definition=DEFINITION):
    return {"type": "tool_addition", "tool": {"type": "tool_definition", "definition": definition},
            "cache_control": {"type": "ephemeral", "ttl": "1h"}}


def request(*blocks, tools=(READ,)):
    return {"max_tokens": 512, "tools": list(tools),
            "messages": [{"role": "user", "content": "hi"},
                         {"role": "system", "content": list(blocks)},
                         {"role": "user", "content": "use the batch tool"}]}


def listed(text):
    return [json.loads(t.split("\n</tools>")[0]) for t in text.split("<tools>\n")[1:]]


def call(name, *params):
    inner = "".join("<parameter=%s>\n%s\n</parameter>\n" % p for p in params)
    return "<tool_call>\n<function=%s>\n%s</function>\n</tool_call>" % (name, inner)


ON = {"STRATA_TOOL_ADDITION": "1"}


class ToolAddition(unittest.TestCase):
    def test_off_by_default_nothing_changes(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("STRATA_TOOL_ADDITION", None)
            req = request(addition())
            messages, tools, _ = anthropic_to_messages(req)
            self.assertEqual([m["content"] for m in messages], ["hi", "use the batch tool"])
            self.assertEqual([t["name"] for t in tools], ["Read"])
            self.assertEqual([t["name"] for t in anthropic_tools(req)[0]], ["Read"])

    def test_the_definition_is_in_the_prompt_where_it_was_added(self):
        with mock.patch.dict(os.environ, ON):
            messages, tools, _ = anthropic_to_messages(request(addition()))
        self.assertEqual([t["name"] for t in tools], ["Read"])          # the prompt's start does not change
        self.assertEqual([m["role"] for m in messages], ["user", "user", "user"])
        self.assertEqual(listed(messages[1]["content"]),
                         [{"name": MCP, "description": "Batch edits.", "parameters": DEFINITION["input_schema"]}])

    def test_the_added_tool_is_callable_and_the_request_is_not_changed(self):
        req = request(addition())
        before = copy.deepcopy(req)
        with mock.patch.dict(os.environ, ON):
            tools, deferred = anthropic_tools(req)
        self.assertEqual([t["name"] for t in tools], ["Read", MCP])
        self.assertFalse(deferred)
        self.assertEqual(req, before)

    def test_a_listed_tool_or_a_repeat_adds_nothing(self):
        with mock.patch.dict(os.environ, ON):
            messages, _, _ = anthropic_to_messages(request(addition(), addition(), addition({**READ})))
            names = [t["name"] for t in anthropic_tools(request(addition(), addition({**READ})))[0]]
        self.assertEqual([t["name"] for t in listed(messages[1]["content"])], [MCP])
        self.assertEqual(names, ["Read", MCP])

    def test_text_beside_an_addition_is_kept(self):
        with mock.patch.dict(os.environ, ON):
            messages, _, _ = anthropic_to_messages(request({"type": "text", "text": "hook context"}, addition()))
        self.assertTrue(messages[1]["content"].startswith("hook context<tools>\n"))

    def test_an_addition_without_a_definition_is_left_out(self):
        with mock.patch.dict(os.environ, ON):
            for broken in ({"type": "tool_addition"}, {"type": "tool_addition", "tool": "x"},
                           addition({"description": "no name"}), addition({"name": MCP, "input_schema": "x"})):
                messages, _, _ = anthropic_to_messages(request(broken))
                self.assertEqual([m["content"] for m in messages], ["hi", "use the batch tool"])
                self.assertEqual([t["name"] for t in anthropic_tools(request(broken))[0]], ["Read"])

    def test_the_parser_reads_the_added_schema(self):
        with mock.patch.dict(os.environ, ON):
            tools, deferred = anthropic_tools(request(addition()))
        parser = OutputParser(thinking=False, tools=tools, stream_tools=True, declared_only=True, deferred=deferred)
        text = call(MCP, ("source", "42"), ("dry_run", "true"))
        calls = [e.call for e in parser.feed(text) + parser.finish("stop") if e.kind == "tool_call"]
        self.assertEqual([(c.name, c.arguments) for c in calls], [(MCP, {"source": "42", "dry_run": True})])


def parse(text, tools):
    p = OutputParser(thinking=False, tools=tools, stream_tools=True, declared_only=True)
    evs = p.feed(text) + p.finish("stop")
    return [e.call for e in evs if e.kind == "tool_call"], p


def tool(name):
    return {"name": name, "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}


UPDATE = call("TaskUpdate", ("taskId", "2"), ("status", "completed"))


class ToolSearchPassthrough(unittest.TestCase):
    def test_off_by_default_the_undeclared_call_stays_text(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("STRATA_TOOLSEARCH_PASSTHROUGH", None)
            calls, p = parse(UPDATE, [tool("ToolSearch"), tool("Bash")])
        self.assertEqual(calls, [])
        self.assertEqual(p.unread_names, ["TaskUpdate"])

    def test_on_with_toolsearch_declared_the_call_is_a_call(self):
        with mock.patch.dict(os.environ, {"STRATA_TOOLSEARCH_PASSTHROUGH": "1"}):
            calls, p = parse(UPDATE, [tool("ToolSearch"), tool("Bash")])
        self.assertEqual([c.name for c in calls], ["TaskUpdate"])
        self.assertEqual(calls[0].arguments, {"taskId": 2, "status": "completed"})   # no schema: "2" reads as a number
        self.assertEqual(p.unread, 0)

    def test_on_without_toolsearch_it_stays_text(self):
        with mock.patch.dict(os.environ, {"STRATA_TOOLSEARCH_PASSTHROUGH": "1"}):
            calls, p = parse(UPDATE, [tool("Bash")])
        self.assertEqual(calls, [])
        self.assertEqual(p.unread, 1)

    def test_on_a_malformed_name_stays_text(self):
        with mock.patch.dict(os.environ, {"STRATA_TOOLSEARCH_PASSTHROUGH": "1"}):
            calls, _ = parse(UPDATE.replace("TaskUpdate", "Task/Update!"), [tool("ToolSearch")])
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
