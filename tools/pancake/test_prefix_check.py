"""prefix_check.py on synthetic captures.  No model, no GPU, no server.

Run: nice -n 19 python3 -I -m pytest -q tools/pancake/test_prefix_check.py"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prefix_check as P  # noqa: E402

SYSTEM = "You are a coding agent.\nToday's date: 2026-11-11\nWorking directory: /work/project\n" + "Rules. " * 300
TOOLS = [{"type": "function", "function": {"name": n, "description": f"the {n} tool",
                                           "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                          "required": ["path"]}}}
         for n in ("read", "edit", "bash", "grep")]
CALL = {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"path": "a.py", "limit": 50}'}}
THINK = "The user wants a.py; I will read it first. " * 20


def body(messages, tools=None, system=SYSTEM, **kw):
    msgs = ([{"role": "system", "content": system}] if system is not None else []) + messages
    out = {"model": "m", "messages": msgs, "stream": True}
    if tools is not None:
        out["tools"] = tools
    out.update(kw)
    return out


def rec(request, reply=None, session="s1", path="/v1/chat/completions"):
    r = {"path": path, "session": session, "request": request}
    if reply is not None:
        r["response"] = {"message": reply}
    return r


def reply_openai(reasoning=THINK, call=CALL, text=None):
    m = {"role": "assistant", "content": text}
    if reasoning:
        m["reasoning_content"] = reasoning
    if call:
        m["tool_calls"] = [call]
    return m


def resend(reply, **over):
    """The reply as a client resends it in the next request."""
    m = copy.deepcopy(reply)
    for k, v in over.items():
        if v is None:
            m.pop(k, None)
        else:
            m[k] = v
    return m


USER = {"role": "user", "content": "please read a.py and fix the bug"}
RESULT = {"role": "tool", "tool_call_id": "call_1", "content": "def f():\n    return 1\n" * 40}


def pair(a_req, reply, b_req, **kw):
    reqs, results = P.run([rec(a_req, reply), rec(b_req)], **kw)
    assert len(results) == 1
    return results[0][0]


def kinds(res):
    return [d["kind"] for d in res["divergences"]]


def base_pair(**over):
    reply = reply_openai()
    a = body([USER], TOOLS)
    b = body([USER, resend(reply, **over.pop("reply_over", {})), RESULT], over.pop("tools", TOOLS), **over)
    return a, reply, b


def test_identical_prefix_pair():
    a, reply, b = base_pair()
    res = pair(a, reply, b)
    assert res["verdict"] == "reusable"
    assert res["first_divergence"] is None and res["divergences_total"] == 0 and res["render_neutral"] == []
    assert res["tools"]["identical"] and res["system"]["identical"]
    assert res["messages"] == {"expected": 2, "sent": 3, "new_roles": ["tool"], "reply_checked": True}
    assert res["reasoning"]["reply_reasoning_resent"] is True
    p = res["prefix"]
    assert p["shared_chars"] == p["ideal_chars"] and p["share"] == p["ideal_share"] and 0.7 < p["share"] < 1
    # the share a perfect client gives is everything but the new tool result
    assert p["total_chars"] - p["ideal_chars"] == len(RESULT["content"])


def test_reasoning_dropped():
    a, reply, b = base_pair(reply_over={"reasoning_content": None})
    res = pair(a, reply, b)
    f = res["first_divergence"]
    assert res["verdict"] == "breaks"
    assert f["kind"] == "reasoning_dropped" and f["index"] == 2 and f["role"] == "assistant" and f["field"] == "reasoning"
    assert f["chars_expected"] == len(THINK) and f["chars_sent"] == 0
    assert res["reasoning"]["reply_reasoning_resent"] is False
    # everything before the reply is shared; the tool result, after the break, is not
    p = res["prefix"]
    assert p["shared_chars"] < p["ideal_chars"] and p["shared_chars"] >= len(SYSTEM) + len(USER["content"])


def test_reasoning_under_another_key_is_dropped_for_the_server():
    a, reply, b = base_pair(reply_over={"reasoning_content": None, "reasoning": THINK})
    res = pair(a, reply, b)
    assert res["first_divergence"]["kind"] == "reasoning_other_key"
    assert res["reasoning"]["sent_history"]["with_reasoning_under_other_key"] == 1


def test_no_reasoning_made_none_lost():
    reply = reply_openai(reasoning=None)
    a = body([USER], TOOLS)
    b = body([USER, resend(reply), RESULT], TOOLS)
    res = pair(a, reply, b)
    assert res["verdict"] == "reusable" and res["reasoning"]["reply_reasoning_chars"] == 0


def test_tool_order_changed():
    a, reply, b = base_pair(tools=[TOOLS[1], TOOLS[0], TOOLS[2], TOOLS[3]])
    res = pair(a, reply, b)
    f = res["first_divergence"]
    assert f["kind"] == "tools_reordered" and f["field"] == "tools"
    assert res["tools"]["same_set"] and not res["tools"]["identical"] and res["tools"]["first_difference_at_tool"] == 0
    assert res["prefix"]["shared_chars"] == 0                      # the tool list starts the rendered prompt


def test_tool_added_removed_and_schema():
    a, reply, b = base_pair(tools=TOOLS + [{"type": "function", "function": {"name": "web", "parameters": {}}}])
    assert pair(a, reply, b)["first_divergence"]["kind"] == "tools_added"
    a, reply, b = base_pair(tools=TOOLS[:3])
    assert pair(a, reply, b)["first_divergence"]["kind"] == "tools_removed"
    changed = copy.deepcopy(TOOLS)
    changed[2]["function"]["description"] = "the bash tool, now with a timeout note"
    a, reply, b = base_pair(tools=changed)
    res = pair(a, reply, b)
    assert res["first_divergence"]["kind"] == "tool_schema_changed" and res["tools"]["changed_tools"] == ["bash"]
    # same content, another key order inside one tool: bytes differ
    reordered = copy.deepcopy(TOOLS)
    fn = reordered[1]["function"]
    reordered[1]["function"] = {"parameters": fn["parameters"], "description": fn["description"], "name": fn["name"]}
    a, reply, b = base_pair(tools=reordered)
    assert pair(a, reply, b)["first_divergence"]["kind"] == "tool_schema_key_order"


def test_system_timestamp_changed():
    a, reply, b = base_pair(system=SYSTEM.replace("2026-11-11", "2026-11-12"))
    res = pair(a, reply, b, snippets=40)
    f = res["first_divergence"]
    assert f["kind"] == "system_timestamp" and res["verdict"] == "breaks"
    assert res["system"]["kind"] == "system_timestamp" and not res["system"]["identical"]
    assert res["system"]["first_diff_char"] == SYSTEM.index("2026-11-11") + 9      # the last digit of the day
    assert "2026-11-11" in res["system"]["snippet_expected"] and "2026-11-12" in res["system"]["snippet_sent"]
    # the tool list is rendered before the system prompt, so the tools are still shared
    assert res["prefix"]["shared_chars"] > 0
    # by default the report holds no text from the capture
    plain = pair(a, reply, b)
    assert "snippet_expected" not in json.dumps(plain)


def test_system_edge_whitespace_and_id_and_billing():
    a, reply, b = base_pair(system=SYSTEM + "\n\n")
    res = pair(a, reply, b)
    assert res["verdict"] == "reusable" and [d["kind"] for d in res["render_neutral"]] == ["system_edge_whitespace"]
    a, reply, b = base_pair(system=SYSTEM.replace("Rules.", "Rules! ", 1))
    assert pair(a, reply, b)["first_divergence"]["kind"] == "system_changed"
    sysa = "x-anthropic-billing-header: cc_version=2.1.170.bf4; cch=b145e;\n" + SYSTEM
    sysb = "x-anthropic-billing-header: cc_version=2.1.170.bf4; cch=77aa1;\n" + SYSTEM
    a, reply, b = base_pair(system=sysb)
    a = body([USER], TOOLS, system=sysa)
    res = pair(a, reply, b)
    assert res["verdict"] == "reusable" and [d["kind"] for d in res["render_neutral"]] == ["system_billing_stamp"]


def test_tool_result_rerendered_in_history():
    r1 = reply_openai()
    a = body([USER, resend(r1), RESULT], TOOLS)
    r2 = {"role": "assistant", "content": "done"}
    nxt = {"role": "user", "content": "thanks"}
    # whitespace inside the result (tabs for spaces): breaks
    changed = dict(RESULT, content=RESULT["content"].replace("    ", "\t"))
    res = pair(a, r2, body([USER, resend(r1), changed, resend(r2), nxt], TOOLS))
    f = res["first_divergence"]
    assert f["kind"] == "tool_result_reformatted" and f["index"] == 3 and f["role"] == "tool"
    # truncated result: changed, and the shared prefix inside the result is counted
    cut = dict(RESULT, content=RESULT["content"][:100] + "...[truncated]")
    res = pair(a, r2, body([USER, resend(r1), cut, resend(r2), nxt], TOOLS))
    f = res["first_divergence"]
    assert f["kind"] == "tool_result_changed" and f["common_prefix_chars"] == 100
    # a trailing newline only: the template trims it
    trimmed = dict(RESULT, content=RESULT["content"].rstrip("\n"))
    res = pair(a, r2, body([USER, resend(r1), trimmed, resend(r2), nxt], TOOLS))
    assert res["verdict"] == "reusable" and [d["kind"] for d in res["render_neutral"]] == ["tool_result_edge_whitespace"]


def test_tool_arguments():
    ws = {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"path":"a.py","limit":50}'}}
    a, reply, b = base_pair(reply_over={"tool_calls": [ws]})
    res = pair(a, reply, b)
    assert res["verdict"] == "reusable" and [d["kind"] for d in res["render_neutral"]] == ["tool_args_whitespace"]
    ko = {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"limit": 50, "path": "a.py"}'}}
    a, reply, b = base_pair(reply_over={"tool_calls": [ko]})
    assert pair(a, reply, b)["first_divergence"]["kind"] == "tool_args_key_order"
    other = {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"path": "b.py"}'}}
    a, reply, b = base_pair(reply_over={"tool_calls": [other]})
    assert pair(a, reply, b)["first_divergence"]["kind"] == "tool_args_changed"
    newid = {"id": "call_zzz", "type": "function", "function": CALL["function"]}
    a, reply, b = base_pair(reply_over={"tool_calls": [newid]})
    res = pair(a, reply, b)
    assert res["verdict"] == "reusable" and [d["kind"] for d in res["render_neutral"]] == ["tool_call_id_changed"]


def test_reply_text_changed_and_content_shape():
    reply = reply_openai(call=None, text="Here is the fix.\n\nIt was an off-by-one.")
    a = body([USER], TOOLS)
    nxt = {"role": "user", "content": "thanks"}
    b = body([USER, resend(reply, content="Here is the fix.\n\nIt was an off by one."), nxt], TOOLS)
    assert pair(a, reply, b)["first_divergence"]["kind"] == "text_changed"
    b = body([USER, resend(reply, content="Here is the fix.\r\n\r\nIt was an off-by-one."), nxt], TOOLS)
    assert pair(a, reply, b)["first_divergence"]["kind"] == "text_inner_whitespace"
    parts = [{"type": "text", "text": reply["content"]}]
    b = body([USER, resend(reply, content=parts), nxt], TOOLS)
    res = pair(a, reply, b)
    assert res["verdict"] == "unclear" and res["first_divergence"]["kind"] == "content_shape"


def test_history_shorter_and_not_a_continuation():
    a, reply, b = base_pair()
    res = pair(b, reply_openai(call=None, text="done"), body([USER], TOOLS))      # the history lost messages
    assert res["first_divergence"]["kind"] == "history_shorter"
    res = pair(a, reply, a)                                                        # the same request again (a retry)
    assert res["first_divergence"]["kind"] == "not_a_continuation"


def test_no_response_in_capture_checks_the_history_only():
    a, reply, b = base_pair()
    reqs, results = P.run([rec(a), rec(b)])
    res = results[0][0]
    assert res["verdict"] == "reusable" and res["messages"]["reply_checked"] is False
    assert res["messages"]["new_roles"] == ["tool"]
    # a changed earlier message is still found
    b2 = copy.deepcopy(b)
    b2["messages"][1]["content"] = "please read b.py and fix the bug"
    _, results = P.run([rec(a), rec(b2)])                                # the first message changed: auto-link finds no parent
    assert len(results) == 0 and len(results.orphans) == 1
    res = P.run([rec(a), rec(b2)], link="consecutive")[1][0][0]
    assert res["first_divergence"]["kind"] == "text_changed" and res["first_divergence"]["index"] == 1


def anthropic_body(messages, system=None, tools=None, **kw):
    out = {"model": "m", "max_tokens": 4096, "stream": True, "messages": messages}
    out["system"] = system if system is not None else [{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}]
    out["tools"] = tools if tools is not None else [{"name": n, "description": f"the {n} tool", "input_schema": {"type": "object"}}
                                                    for n in ("read", "edit", "bash")]
    out.update(kw)
    return out


A_USER = {"role": "user", "content": [{"type": "text", "text": "read a.py", "cache_control": {"type": "ephemeral"}}]}
A_REPLY = {"role": "assistant", "content": [{"type": "thinking", "thinking": THINK, "signature": "sig"},
                                            {"type": "text", "text": "I will read it."},
                                            {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "a.py", "limit": 50}}]}
A_RESULT = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "x = 1\n" * 30}]}]}


def test_anthropic_style_pair():
    a = anthropic_body([A_USER])
    b = anthropic_body([A_USER, A_REPLY, A_RESULT])
    res = pair_a(a, A_REPLY, b)
    assert res["style"] == "anthropic" and res["verdict"] == "reusable" and res["tools"]["identical"]
    assert res["reasoning"]["reply_reasoning_resent"] is True
    # the cache_control marker moving to the newest block is not a difference
    moved = copy.deepcopy(b)
    moved["messages"][0]["content"][0].pop("cache_control")
    moved["messages"][2]["content"][0]["cache_control"] = {"type": "ephemeral"}
    assert pair_a(a, A_REPLY, moved)["verdict"] == "reusable"
    # thinking block dropped from the resent assistant turn
    dropped = copy.deepcopy(b)
    dropped["messages"][1]["content"] = [blk for blk in dropped["messages"][1]["content"] if blk["type"] != "thinking"]
    res = pair_a(a, A_REPLY, dropped)
    assert res["first_divergence"]["kind"] == "reasoning_dropped" and res["first_divergence"]["index"] == 1
    # tool input key order
    ko = copy.deepcopy(b)
    ko["messages"][1]["content"][2]["input"] = {"limit": 50, "path": "a.py"}
    assert pair_a(a, A_REPLY, ko)["first_divergence"]["kind"] == "tool_args_key_order"
    # system block changed
    sysb = [{"type": "text", "text": SYSTEM.replace("2026-11-11", "2026-11-12")}]
    res = pair_a(a, A_REPLY, anthropic_body([A_USER, A_REPLY, A_RESULT], system=sysb))
    assert res["first_divergence"]["kind"] == "system_timestamp"
    # a tool result re-rendered in the history of the third request
    reply2 = {"role": "assistant", "content": [{"type": "text", "text": "done"}]}
    nxt = {"role": "user", "content": "thanks"}
    third = anthropic_body([A_USER, A_REPLY, A_RESULT, reply2, nxt])
    assert pair_a(b, reply2, third)["verdict"] == "reusable"
    rr = copy.deepcopy(third)
    rr["messages"][2]["content"][0]["content"] = "x = 1\n" * 29
    res = pair_a(b, reply2, rr)
    assert res["first_divergence"]["kind"] == "tool_result_changed" and res["first_divergence"]["index"] == 2


def pair_a(a_req, reply, b_req, **kw):
    reqs, results = P.run([rec(a_req, reply, path="/v1/messages"), rec(b_req, path="/v1/messages")], **kw)
    assert len(results) == 1
    return results[0][0]


def test_redaction_keeps_the_comparison_valid():
    secret = "SECRET_TOKEN_9f3a"
    user = {"role": "user", "content": f"please use {secret} to fix the bug"}
    cases = []
    reply = reply_openai()
    cases.append((body([user], TOOLS), reply, body([user, resend(reply), RESULT], TOOLS)))                      # reusable
    cases.append((body([user], TOOLS), reply, body([user, resend(reply, reasoning_content=None), RESULT], TOOLS)))   # dropped
    cases.append((body([user], TOOLS), reply, body([user, resend(reply), RESULT], [TOOLS[1], TOOLS[0], TOOLS[2], TOOLS[3]])))
    cases.append((body([user], TOOLS), reply, body([user, resend(reply), RESULT], TOOLS, system=SYSTEM.replace("2026-11-11", "2026-11-12"))))
    ko = {"id": "call_1", "type": "function", "function": {"name": "read", "arguments": '{"limit": 50, "path": "a.py"}'}}
    cases.append((body([user], TOOLS), reply, body([user, resend(reply, tool_calls=[ko]), RESULT], TOOLS)))     # args key order
    cases.append((body([user], TOOLS), reply, body([user, resend(reply), dict(RESULT, content=RESULT["content"].rstrip())], TOOLS)))   # neutral
    cases.append((body([user], TOOLS), reply, body([user, resend(reply), dict(RESULT, content=RESULT["content"].replace("    ", "\t"))], TOOLS)))
    for a, rp, b in cases:
        raw = [rec(a, rp), rec(b)]
        red = [P.redact_record(r) for r in raw]
        text = json.dumps(red)
        assert secret not in text and "off-by-one" not in text and "a.py" not in text and "def f()" not in text
        assert "read" in text                                               # tool names stay
        want = pair(a, rp, b)
        got = P.run(red)[1][0][0]
        assert got["verdict"] == want["verdict"], (want["first_divergence"], got["first_divergence"])
        wk = want["first_divergence"] and want["first_divergence"]["kind"]
        gk = got["first_divergence"] and got["first_divergence"]["kind"]
        if wk == "system_timestamp":
            assert gk == "system_changed"                                   # the changed text cannot be looked at any more
        else:
            assert gk == wk
        assert [d["kind"] for d in got["render_neutral"]] == [d["kind"] for d in want["render_neutral"]]
        # shared prefix: exact in a raw capture, to 512 characters in a redacted one
        assert abs(got["prefix"]["shared_chars"] - want["prefix"]["shared_chars"]) <= 512
        assert got["prefix"]["total_chars"] == want["prefix"]["total_chars"]
    assert P.redact_record(rec(body([user], TOOLS)))["redacted"] is True


def test_redacted_anthropic():
    a = anthropic_body([A_USER])
    b = anthropic_body([A_USER, A_REPLY, A_RESULT])
    dropped = copy.deepcopy(b)
    dropped["messages"][1]["content"] = [blk for blk in dropped["messages"][1]["content"] if blk["type"] != "thinking"]
    for bad, kind in ((b, None), (dropped, "reasoning_dropped")):
        red = [P.redact_record(rec(a, A_REPLY, path="/v1/messages")), P.redact_record(rec(bad, path="/v1/messages"))]
        assert "a.py" not in json.dumps(red) and "read it" not in json.dumps(red)
        res = P.run(red)[1][0][0]
        assert (res["first_divergence"] or {}).get("kind") == kind


def test_pairing_skips_interleaved_side_requests_and_sessions():
    reply = reply_openai()
    a = body([USER], TOOLS)
    side = body([{"role": "user", "content": "write a title"}], None, system="Title writer.")
    b = body([USER, resend(reply), RESULT], TOOLS)
    recs = [rec(a, reply, session=None), rec(side, session=None), rec(b, session=None)]
    reqs, results = P.run(recs)
    pairs = [(r["earlier"], r["later"]) for r, *_ in results]
    assert pairs == [(1, 3)]                                           # request 3 continues request 1
    assert any("skipped 1" in how for *_, how in results)
    assert [r.n + 1 for r in results.orphans] == [2]                   # the side request continues nothing
    reqs, results = P.run(recs, link="consecutive")
    assert [(r["earlier"], r["later"]) for r, *_ in results] == [(1, 2), (2, 3)]
    # two sessions are never paired with each other
    recs = [rec(a, reply, session="x"), rec(b, session="y")]
    assert P.run(recs)[1] == []


def test_loading_formats(tmp_path):
    a, reply, b = base_pair()
    f1 = tmp_path / "cap.jsonl"
    f1.write_text(json.dumps(rec(a, reply)) + "\n\n" + json.dumps(rec(b)) + "\n")
    assert len(P.load([str(f1)])) == 2
    f2 = tmp_path / "bare.json"                                       # bare bodies in one JSON array
    f2.write_text(json.dumps([a, b]))
    recs = P.load([str(f2)])
    assert len(recs) == 2 and recs[0]["request"] == a
    f3 = tmp_path / "monitor.json"                                    # API monitor records: the truncated one is skipped
    f3.write_text(json.dumps([{"path": "/v1/chat/completions", "input": json.dumps(a, indent=2), "input_truncated": False},
                              {"path": "/v1/chat/completions", "input": json.dumps(b)[:50], "input_truncated": True}]))
    assert len(P.load([str(f3)])) == 1


def test_cli_report_and_json(tmp_path, capsys):
    a, reply, b = base_pair(reply_over={"reasoning_content": None})
    cap = tmp_path / "cap.jsonl"
    cap.write_text(json.dumps(rec(a, reply)) + "\n" + json.dumps(rec(b)) + "\n")
    out_json, out_red = tmp_path / "r.json", tmp_path / "red.jsonl"
    assert P.main([str(cap), "--json", str(out_json), "--redact", str(out_red)]) == 0
    text = capsys.readouterr().out
    assert "PREFIX BREAKS" in text and "reasoning_dropped" in text and "keep_reasoning" in text
    data = json.loads(out_json.read_text())
    assert data["verdicts"]["breaks"] == 1 and data["pairs"][0]["first_divergence"]["kind"] == "reasoning_dropped"
    assert P.main([str(out_red)]) == 0                                # the redacted file gives the same verdict
    assert "reasoning_dropped" in capsys.readouterr().out
    assert P.main([str(out_red), "--no-report"]) == 0
