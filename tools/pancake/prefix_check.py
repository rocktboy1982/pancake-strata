#!/usr/bin/env python3
"""prefix_check.py - does a chat client resend a byte-identical prefix between consecutive agent turns?  (MOD_PLAN 17.0,
measurement M4 of pancake-docs/TTFT_LEVERS.md; how to use it: pancake-docs/PREFIX_CHECK.md).  Stdlib only; reads files,
runs no model, no server, no GPU, sends nothing anywhere.

    python tools/pancake/prefix_check.py CAPTURE [CAPTURE ...] [--link auto|consecutive] [--json OUT]
                                         [--redact OUT] [--snippets N] [--session KEY]

CAPTURE: a JSONL (one record per line) or JSON (a list, or one record) file.  A record is one of
  * a capture line of the server (serve/request_capture.py, STRATA_REQUEST_CAPTURE=DIR):
    {"request": <body>, "response": {"message": <assistant message>}, "session": ..., "path": ...};
  * a bare request body: an OpenAI /v1/chat/completions body or an Anthropic /v1/messages body (no reply known: the reply
    of the earlier request is then taken from the next request's own history, so only the replay of history is checked);
  * a record of the server's API monitor (GET /api/requests?id=...: "input" is the body as text, "path"); a record whose
    input was cut at 256 KB (input_truncated) is skipped.

For each pair of requests of one session (request N and the next one) it asks: does the messages array of request N+1
start with the messages of request N plus the assistant reply the client received?  It compares field by field, in the
order the chat template renders them (tool list, system prompt, then each message: reasoning, text, tool calls, tool
results), and reports
  * the first divergence that changes the rendered prompt: message index, role, field and kind (see EFFECTS);
  * differences that do NOT change it (the template trims whitespace at the edges of every text; the server parses tool
    arguments as JSON; call ids are not rendered) - listed apart as "render-neutral";
  * the tool list (names, order, byte-level), the system prompt (a timestamp-like or an id-like change is named),
    whether the reasoning of the reply (and of earlier assistant turns) comes back;
  * an estimate of the shared prefix in characters, and the share a perfect client would give.

Rendering rules used (read from serve/chat_template.jinja and serve/frontend.py, not measured): every text is trimmed; the
tool list is rendered BEFORE the system prompt, with json.dumps in the key order the client sent; tool-call arguments are
parsed as JSON and rendered in their key order; only `reasoning_content` (OpenAI) / thinking blocks (Anthropic) are read
as reasoning; call ids are not rendered.  The server also has its own fixes (stable order for a repeated tool set,
a pinned Claude Code billing stamp, opencode's <env> block moved to the first user message), which this tool does not
apply: it shows what the CLIENT sent.

--redact OUT writes a copy in which every message text, system text, tool-call argument, tool result and reply is replaced
by a marker with its length and hashes (of the text, of the text without edge whitespace, without any whitespace, of its
JSON re-dump, and of every 512-character prefix), and each tool by its name and hashes.  The comparison of a redacted
file gives the same verdicts (the prefix length is then known to 512 characters).  Role names, call ids, tool names and
numbers stay.  Share a capture only after that, and look at the result first.

PRIVACY: a capture holds the user's prompts and code.  This tool reads files you name and writes only the files you name
(--json, --redact).  By default the report holds no text from the capture; --snippets N adds N characters around a
difference, to the terminal and to the JSON.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys

STEP = 512                                  # prefix-hash granularity of a redacted text, in characters
MAX_DIVERGENCES = 12
BILLING = "x-anthropic-billing-header:"

# kind -> (does it change the rendered prompt?  yes / maybe / no, what it means, what fixes it)
EFFECTS = {
    "tools_reordered": ("yes", "the same tools in another order; the template renders the tool list before the system prompt, so the prompt leaves the cache at the tools: everything is read again",
                        "client: a stable tool order.  Server: _stable_tool_order puts a repeated set back in the order it first came in (so a loss only where the server never saw the first order, e.g. after a restart)"),
    "tools_added": ("yes", "a tool appeared; the prompt changes in the tool block, at its start", "client: a fixed tool list (pin the MCP servers, no per-request tools); no server setting fixes it"),
    "tools_removed": ("yes", "a tool disappeared; the prompt changes in the tool block", "client: a fixed tool list; no server setting fixes it"),
    "tools_changed": ("yes", "tools were added and removed", "client: a fixed tool list; no server setting fixes it"),
    "tool_schema_changed": ("yes", "a tool's description or schema differs; rendered with json.dumps, so any byte counts", "client: stable tool descriptions"),
    "tool_schema_key_order": ("yes", "a tool's JSON has the same content in another key order; rendered in the sent order", "client: stable key order (or a server-side sort, not implemented)"),
    "system_timestamp": ("yes", "a date or time inside the system prompt changed; the system block starts the prompt, so everything after it is read again", "opencode: env_to_user (on in fast-hip) moves its <env> block to the first user message; other clients: take the date out of the system prompt in the client's settings"),
    "system_volatile_id": ("yes", "an id-like token (hex, uuid, counter) in the system prompt changes per request", "client: remove it; Claude Code's billing stamp is already pinned by the server"),
    "system_billing_stamp": ("no", "Claude Code's x-anthropic-billing-header changed; the server pins it (pin_billing_stamp)", "none needed"),
    "system_changed": ("yes", "the system prompt text differs", "client: a stable system prompt"),
    "system_inner_whitespace": ("yes", "the system prompt differs in inner whitespace", "client: a stable system prompt"),
    "system_edge_whitespace": ("no", "the system prompt differs only in whitespace at its ends; the template trims it", "none needed"),
    "system_blocks_changed": ("yes", "the number of system blocks or messages changed", "client: a stable system prompt"),
    "reasoning_dropped": ("yes", "the client resent the assistant reply without its reasoning; the engine's cache holds the thinking tokens, the re-rendered turn has an empty think block, so the prompt leaves the cache at that turn (a fallback to the turn checkpoint: that turn is read again)",
                          "server: keep_reasoning (config or STRATA_KEEP_REASONING=1) puts the kept reasoning back"),
    "reasoning_other_key": ("yes", "the client sends the reasoning under a key the server does not read (reasoning / reasoning_text instead of reasoning_content); same effect as dropped",
                            "server: keep_reasoning; or accept the other key (not implemented)"),
    "reasoning_changed": ("yes", "the reasoning the client resends is not what the model wrote", "none known; keep_reasoning only fills in reasoning the client did not send"),
    "reasoning_added": ("yes", "the client sends reasoning where the model produced none", "none known"),
    "reasoning_edge_whitespace": ("no", "reasoning differs only in whitespace at its ends; the template trims it", "none needed"),
    "text_changed": ("yes", "the text of a message differs from the earlier one (or from the reply the client got): the client edited, summarised or re-encoded it", "none server-side; client behaviour"),
    "text_inner_whitespace": ("yes", "a text differs only in whitespace inside it (line endings, indentation, trailing spaces per line)", "none server-side; client behaviour"),
    "text_edge_whitespace": ("no", "a text differs only in whitespace at its ends; the template trims it", "none needed"),
    "tool_args_key_order": ("yes", "tool-call arguments have the same content in another key order; the template renders them in key order", "client: resend the arguments as received"),
    "tool_args_changed": ("yes", "tool-call arguments differ in content", "client: resend the arguments as received"),
    "tool_args_whitespace": ("no", "tool-call arguments differ only in JSON whitespace; the server parses them", "none needed"),
    "tool_call_id_changed": ("no", "a tool-call id differs; ids are not rendered, but keep_reasoning looks a turn up by its ids first (then by a hash of text and calls)", "none needed for the prefix"),
    "tool_call_name_changed": ("yes", "a tool-call name differs", "client behaviour"),
    "tool_call_added": ("yes", "the client resent more tool calls than the model made", "client behaviour"),
    "tool_call_removed": ("yes", "the client resent fewer tool calls than the model made", "client behaviour"),
    "tool_result_changed": ("yes", "a tool result differs from the earlier one (pruned, truncated, re-rendered)", "client behaviour (e.g. opencode prunes old tool outputs)"),
    "tool_result_reformatted": ("yes", "a tool result differs only in inner whitespace or formatting", "client behaviour"),
    "tool_result_edge_whitespace": ("no", "a tool result differs only in whitespace at its ends; the template trims it", "none needed"),
    "tool_result_id_changed": ("no", "a tool result's call id differs; not rendered (it only orders results)", "none needed for the prefix"),
    "tool_result_added": ("yes", "an extra tool result", "client behaviour"),
    "tool_result_removed": ("yes", "a tool result is missing", "client behaviour"),
    "media_changed": ("yes", "an image or other non-text part differs", "client behaviour"),
    "content_shape": ("maybe", "content is a string in one request and a list of parts (or null) in the other; the server joins text parts, so probably neutral - check with a replay", "none needed if neutral"),
    "extra_fields": ("no", "message fields the server does not read differ", "none needed"),
    "role": ("yes", "the role of a message differs", "client behaviour"),
    "history_shorter": ("yes", "the new request has fewer messages than the earlier one plus its reply: history was compacted, pruned or this is another conversation", "client behaviour (compaction); nothing server-side"),
    "not_a_continuation": ("yes", "the new request has no message after the earlier request's messages (a retry or regenerate, or another conversation)", "none"),
}


# ------------------------------------------------------------------------------------------------ text info
def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", "surrogatepass")).hexdigest()[:16]


class T:
    """A text as the comparison needs it: length and hashes (exact, edge-stripped, whitespace-free, JSON re-dump in sent
    key order and with sorted keys, prefix hashes every STEP characters); `raw` is the text itself unless it was redacted."""
    __slots__ = ("n", "sha", "st", "ws", "norm", "srt", "pre", "raw")

    def __init__(self, n, sha, st, ws, norm, srt, pre, raw):
        self.n, self.sha, self.st, self.ws, self.norm, self.srt, self.pre, self.raw = n, sha, st, ws, norm, srt, pre, raw

    def same(self, o: "T") -> bool:
        return self.n == o.n and self.sha == o.sha


MARK = re.compile(r"^\[\[redacted n=(\d+) sha=(\w+) st=(\w+) ws=(\w+)(?: norm=(\w+) srt=(\w+))?(?: pre=([0-9a-f,]*))?\]\]$")


def _json_hashes(s: str):
    if s.lstrip()[:1] in ("{", "["):
        try:
            v = json.loads(s)
        except ValueError:
            return None, None
        return _h(json.dumps(v, ensure_ascii=False)), _h(json.dumps(v, ensure_ascii=False, sort_keys=True))
    return None, None


def _prefixes(s: str) -> list[str]:
    acc, out = hashlib.sha256(), []
    for k in range(STEP, len(s) + 1, STEP):
        acc.update(s[k - STEP:k].encode("utf-8", "surrogatepass"))
        out.append(acc.copy().hexdigest()[:8])
    return out


def from_raw(s: str) -> T:
    norm, srt = _json_hashes(s)
    return T(len(s), _h(s), _h(s.strip()), _h("".join(s.split())), norm, srt, _prefixes(s), s)


EMPTY = from_raw("")


def redact_text(s: str) -> str:
    t = from_raw(s)
    out = f"[[redacted n={t.n} sha={t.sha} st={t.st} ws={t.ws}"
    if t.norm:
        out += f" norm={t.norm} srt={t.srt}"
    return out + (f" pre={','.join(t.pre)}" if t.pre else "") + "]]"


def info(x) -> T:
    """T of a string (a redaction marker is read back), of None (empty), or of any JSON value (its dump)."""
    if x is None:
        return EMPTY
    if isinstance(x, str):
        m = MARK.match(x)
        if m:
            n, sha, st, ws, norm, srt, pre = m.groups()
            return T(int(n), sha, st, ws, norm, srt, pre.split(",") if pre else [], None)
        return from_raw(x)
    return from_raw(json.dumps(x, ensure_ascii=False))


def common_prefix(a: T, b: T) -> tuple[int, bool]:
    """(characters the two texts share from the start, exact?).  Redacted texts: to the STEP, from the prefix hashes."""
    if a.raw is not None and b.raw is not None:
        n = min(a.n, b.n)
        lo, hi = 0, n
        while lo < hi:                                   # binary search on slices: O(n log n) at worst, no Python loop per char
            mid = (lo + hi + 1) // 2
            if a.raw[:mid] == b.raw[:mid]:
                lo = mid
            else:
                hi = mid - 1
        return lo, True
    if a.same(b):
        return a.n, True
    k = 0
    while k < min(len(a.pre), len(b.pre)) and a.pre[k] == b.pre[k]:
        k += 1
    return min(k * STEP, a.n, b.n), False


# ------------------------------------------------------------------------------------------------ redaction
KEEP_KEYS = {"role", "type", "id", "tool_call_id", "tool_use_id", "name", "index", "is_error"}


def redact_value(v, keep_keys=True):
    if isinstance(v, str):
        return redact_text(v)
    if isinstance(v, list):
        return [redact_value(x, keep_keys) for x in v]
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if keep_keys and k in KEEP_KEYS and isinstance(x, (str, int, bool)) and not isinstance(x, dict):
                out[k] = x
            elif k in ("input", "arguments"):
                out[k] = redact_value(x, False)
            else:
                out[k] = redact_value(x, keep_keys)
        return out
    return v


def redact_message(m):
    if not isinstance(m, dict):
        return m
    out = {}
    for k, v in m.items():
        if k in KEEP_KEYS and isinstance(v, (str, int, bool)):
            out[k] = v
        elif k == "tool_calls" and isinstance(v, list):
            out[k] = []
            for c in v:
                if not isinstance(c, dict):
                    continue
                fn = c.get("function") if isinstance(c.get("function"), dict) else {}
                call = {kk: vv for kk, vv in c.items() if kk in ("id", "type", "index") and isinstance(vv, (str, int))}
                call["function"] = {"name": fn.get("name"), "arguments": redact_value(fn.get("arguments"), False)}
                out[k].append(call)
        else:
            out[k] = redact_value(v)
    return out


def _tool_name(t):
    if not isinstance(t, dict):
        return None
    fn = t.get("function") if isinstance(t.get("function"), dict) else t
    return fn.get("name")


def rendered_tool(t, style):
    """The object the template renders for one tool (serve/frontend.py: openai_to_messages / _anthropic_tool)."""
    if not isinstance(t, dict):
        return t
    if style == "anthropic":
        return {"name": t.get("name"), "description": t.get("description", ""), "parameters": t.get("input_schema", {})}
    return t["function"] if t.get("type") == "function" and isinstance(t.get("function"), dict) else t


def redact_tool(t, style):
    if isinstance(t, dict) and "__redacted_tool__" in t:
        return t
    r = rendered_tool(t, style)
    exact = json.dumps(r, ensure_ascii=False)
    return {"type": t.get("type") if isinstance(t, dict) else None, "name": _tool_name(t),
            "__redacted_tool__": {"n": len(exact), "sha": _h(exact),
                                  "srt": _h(json.dumps(r, ensure_ascii=False, sort_keys=True))}}


def redact_record(rec: dict) -> dict:
    rec = copy.deepcopy(rec)
    style = style_of(rec)
    if isinstance(rec.get("session"), str):
        rec["session"] = _h(rec["session"])[:12]
    if isinstance(rec.get("client"), dict):
        rec["client"] = {k: (v if k == "user-agent" else _h(str(v))[:12]) for k, v in rec["client"].items()}
    req = rec.get("request")
    if isinstance(req, dict):
        for k in ("messages",):
            if isinstance(req.get(k), list):
                req[k] = [redact_message(m) for m in req[k]]
        if "system" in req:
            req["system"] = redact_value(req["system"])
        if isinstance(req.get("tools"), list):
            req["tools"] = [redact_tool(t, style) for t in req["tools"]]
        for k in ("user", "metadata", "prompt_cache_key", "safety_identifier", "tool_choice"):
            if k in req:
                req[k] = redact_value(req[k], False)
    resp = rec.get("response")
    if isinstance(resp, dict) and isinstance(resp.get("message"), dict):
        resp["message"] = redact_message(resp["message"])
    rec["redacted"] = True
    return rec


# ------------------------------------------------------------------------------------------------ loading
def _records_of(obj):
    if isinstance(obj, list):
        for x in obj:
            yield from _records_of(x)
    elif isinstance(obj, dict):
        if "request" in obj and isinstance(obj["request"], dict):
            yield obj
        elif isinstance(obj.get("messages"), list):
            yield {"request": obj}
        elif isinstance(obj.get("input"), str) and obj.get("path"):          # an API monitor record
            if obj.get("input_truncated"):
                return
            try:
                body = json.loads(obj["input"])
            except ValueError:
                return
            if isinstance(body, dict):
                yield {"request": body, "path": obj.get("path"), "t": obj.get("started_at")}
        elif isinstance(obj.get("requests"), list):
            yield from _records_of(obj["requests"])


def load(paths) -> list[dict]:
    recs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            text = f.read()
        try:
            recs.extend(_records_of(json.loads(text)))
            continue
        except ValueError:
            pass
        for n, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.extend(_records_of(json.loads(line)))
            except ValueError as e:
                raise SystemExit(f"{p}:{n}: not JSON ({e})")
    return recs


def style_of(rec: dict) -> str:
    path = rec.get("path") or ""
    if "/messages" in path:
        return "anthropic"
    if "chat/completions" in path:
        return "openai"
    req = rec.get("request") or {}
    if "system" in req and not any(isinstance(m, dict) and m.get("role") == "system" for m in req.get("messages") or []):
        return "anthropic"
    for m in req.get("messages") or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result", "thinking") for b in c):
            return "anthropic"
    return "openai"


# ------------------------------------------------------------------------------------------------ unified model
RANK = {"role": 0, "reasoning": 1, "text": 2, "result": 3, "call": 4, "media": 5, "extra": 6, "shape": 7}
KEY = re.compile(r"^(role|reasoning|text|call|result|media|extra|shape)(\d*)(?:\.(\w+))?$")


def order_of(key: str):
    m = KEY.match(key)
    return (RANK[m.group(1)], int(m.group(2) or 0), m.group(3) or "")


def family(key: str) -> str:
    m = KEY.match(key)
    return m.group(1) + ("." + m.group(3) if m.group(3) else "")


class Msg:
    __slots__ = ("idx", "role", "parts", "reasoning_other", "fp")

    def __init__(self, idx, role, parts, reasoning_other=None):
        self.idx, self.role, self.parts, self.reasoning_other = idx, role, parts, reasoning_other
        self.fp = tuple(sorted((k, t.ws) for k, t in parts.items() if k != "shape"))


def join_texts(texts) -> T:
    """One T for a list of text pieces (a redacted piece cannot be joined: several of them are compared as a list)."""
    if not texts:
        return EMPTY
    if len(texts) == 1:
        return info(texts[0])
    if any(isinstance(t, str) and MARK.match(t) for t in texts):
        return info(json.dumps(texts))
    return info("".join(t if isinstance(t, str) else json.dumps(t) for t in texts))


def content_of(value):
    """(text pieces, other parts, shape) of an OpenAI-style content value."""
    texts, media = [], []
    if isinstance(value, str):
        texts.append(value)
        shape = "str"
    elif value is None:
        shape = "null"
    elif isinstance(value, list):
        shape = "parts"
        for p in value:
            if isinstance(p, str):
                texts.append(p)
            elif isinstance(p, dict) and p.get("type") in ("text", "input_text", "output_text") and isinstance(p.get("text"), str):
                texts.append(p["text"])
            else:
                media.append({k: v for k, v in p.items() if k != "cache_control"} if isinstance(p, dict) else p)
    else:
        shape = "other"
        media.append(value)
    if not media and all(info(t).n == 0 for t in texts):
        shape = "empty"                                  # null, "" and [] render the same
    return texts, media, shape


def _args_text(a):
    if a is None:
        return ""
    return a if isinstance(a, str) else json.dumps(a, ensure_ascii=False)


def unify_openai(m: dict, idx: int) -> Msg:
    role = m.get("role")
    parts = {"role": info(str(role)), "reasoning": info(m.get("reasoning_content"))}
    other = next((k for k in ("reasoning", "reasoning_text", "thinking", "reasoning_details") if m.get(k)), None)
    texts, media, shape = content_of(m.get("content"))
    if role == "tool":
        parts["result0.id"] = info(m.get("tool_call_id"))
        parts["result0.text"] = join_texts(texts)
    else:
        parts["text"] = join_texts(texts)
    for i, p in enumerate(media):
        parts[f"media{i}"] = info(p)
    parts["shape"] = info("absent" if "content" not in m else shape)
    for k, c in enumerate(m.get("tool_calls") or []):
        c = c if isinstance(c, dict) else {}
        fn = c.get("function") if isinstance(c.get("function"), dict) else c
        parts[f"call{k}.id"] = info(c.get("id"))
        parts[f"call{k}.name"] = info(fn.get("name"))
        parts[f"call{k}.args"] = info(_args_text(fn.get("arguments")))
    extra = sorted(set(m) - {"role", "content", "reasoning_content", "reasoning", "reasoning_text", "thinking", "reasoning_details",
                             "tool_calls", "tool_call_id", "name", "refusal", "annotations", "audio", "cache_control"})
    if extra:
        parts["extra"] = info(",".join(extra))
    return Msg(idx, role, parts, other)


def unify_anthropic(m: dict, idx: int) -> Msg:
    role = m.get("role")
    parts = {"role": info(str(role)), "reasoning": EMPTY}
    content = m.get("content")
    if not isinstance(content, list):
        texts, media, shape = content_of(content)
        parts["text"] = join_texts(texts)
        return Msg(idx, role, parts)
    thinking, texts, calls, results, media = [], [], [], [], []
    for b in content:
        if not isinstance(b, dict):
            texts.append(b)
            continue
        kind = b.get("type")
        if kind == "text":
            texts.append(b.get("text", ""))
        elif kind == "thinking":
            thinking.append(b.get("thinking", ""))
        elif kind == "tool_use":
            calls.append(b)
        elif kind == "tool_result":
            results.append(b)
        elif kind != "redacted_thinking":                # the server reads thinking blocks only
            media.append({k: v for k, v in b.items() if k != "cache_control"})
    parts["reasoning"] = join_texts(thinking)
    parts["text"] = join_texts(texts)
    for k, b in enumerate(calls):
        parts[f"call{k}.id"] = info(b.get("id"))
        parts[f"call{k}.name"] = info(b.get("name"))
        parts[f"call{k}.args"] = info(_args_text(b.get("input")))
    for k, b in enumerate(results):
        parts[f"result{k}.id"] = info(b.get("tool_use_id"))
        rt, rm, _ = content_of(b.get("content"))
        parts[f"result{k}.text"] = join_texts(rt)
        if rm:
            parts[f"media{k}"] = info(rm)
        if b.get("is_error"):
            parts[f"result{k}.err"] = info("1")
    for i, b in enumerate(media):
        parts[f"media{len(results) + i}"] = info(b)
    return Msg(idx, role, parts)


class Req:
    """One request in the unified model."""

    def __init__(self, rec: dict, n: int):
        self.n, self.rec = n, rec
        self.style = style_of(rec)
        body = rec.get("request") or {}
        meta = body.get("metadata")
        self.session = rec.get("session") or body.get("prompt_cache_key") or \
            (meta.get("user_id") if isinstance(meta, dict) else None) or body.get("user") or None
        if isinstance(self.session, dict):
            self.session = json.dumps(self.session, sort_keys=True)
        self.client = (rec.get("client") or {}).get("user-agent")
        self.tools = []
        for t in body.get("tools") or []:
            if isinstance(t, dict) and "__redacted_tool__" in t:
                red = t["__redacted_tool__"]
                self.tools.append((t.get("name"), red["n"], red["sha"], red["srt"]))
            else:
                r = rendered_tool(t, self.style)
                exact = json.dumps(r, ensure_ascii=False)
                self.tools.append((_tool_name(t), len(exact), _h(exact), _h(json.dumps(r, ensure_ascii=False, sort_keys=True))))
        raw_messages = [m for m in body.get("messages") or [] if isinstance(m, dict)]
        self.system, k = [], 0
        if self.style == "anthropic":
            s = body.get("system")
            if isinstance(s, str):
                self.system = [info(s)]
            elif isinstance(s, list):
                self.system = [info(b.get("text") if isinstance(b, dict) else b) for b in s]
            unify = unify_anthropic
        else:
            while k < len(raw_messages) and raw_messages[k].get("role") in ("system", "developer"):
                self.system.append(join_texts(content_of(raw_messages[k].get("content"))[0]))
                k += 1
            unify = unify_openai
        self.messages = [unify(m, i) for i, m in enumerate(raw_messages) if i >= k]
        reply = (rec.get("response") or {}).get("message") if isinstance(rec.get("response"), dict) else None
        self.reply = unify(reply, -1) if isinstance(reply, dict) else None


# ------------------------------------------------------------------------------------------------ comparing a pair
def _div(kind, pos, role=None, index=None, field=None, a=None, b=None, extra=None):
    d = {"kind": kind, "breaks": EFFECTS[kind][0], "index": index, "role": role, "field": field, "pos": pos}
    if a is not None and b is not None:
        d["chars_expected"], d["chars_sent"] = a.n, b.n
        d["common_prefix_chars"], d["prefix_exact"] = common_prefix(a, b)
    if extra:
        d.update(extra)
    return d


TS = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}|\b(?:mon|tue|wed|thu|fri|sat|sun)[a-z]*\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b|\d{9,}", re.I)
ID = re.compile(r"^[0-9a-f\-]{5,}$|^\d+$", re.I)


def classify_system(a: T, b: T, snippets: int):
    """(kind, extra) for two differing system blocks."""
    if a.st == b.st:
        return "system_edge_whitespace", {}
    extra = {}
    if a.raw is None or b.raw is None:
        return ("system_inner_whitespace" if a.ws == b.ws else "system_changed"), {"note": "redacted: the changed text cannot be classified further"}
    if a.ws == b.ws:
        return "system_inner_whitespace", {}
    i, _ = common_prefix(a, b)
    j = 0
    while j < min(a.n, b.n) - i and a.raw[-1 - j] == b.raw[-1 - j]:
        j += 1
    am, bm = a.raw[i:a.n - j], b.raw[i:b.n - j]
    l = i
    while l > 0 and not a.raw[l - 1].isspace():          # the whole token around the change (2026-11-1[1] -> 2026-11-1[2])
        l -= 1
    ra, rb = a.n - j, b.n - j
    while ra < a.n and not a.raw[ra].isspace():
        ra += 1
    while rb < b.n and not b.raw[rb].isspace():
        rb += 1
    ta, tb = a.raw[l:ra], b.raw[l:rb]
    if snippets:
        extra["snippet_expected"], extra["snippet_sent"] = a.raw[max(0, i - snippets // 3):i + snippets], b.raw[max(0, i - snippets // 3):i + snippets]
    if a.raw.startswith(BILLING) and b.raw.startswith(BILLING) and i < 200:
        return "system_billing_stamp", extra
    if len(am) <= 80 and len(bm) <= 80:
        if TS.search(ta) and TS.search(tb) and len(ta) <= 80 and len(tb) <= 80:
            return "system_timestamp", extra
        if ID.match(am.strip() or "x") and ID.match(bm.strip() or "x"):
            return "system_volatile_id", extra
    return "system_changed", extra


def diff_tools(ta, tb):
    """(summary dict, divergence or None).  Tools are (name, n, exact hash, sorted hash)."""
    names_a, names_b = [t[0] for t in ta], [t[0] for t in tb]
    summary = {"count_expected": len(ta), "count_sent": len(tb), "names_equal": names_a == names_b,
               "same_set": sorted(map(str, names_a)) == sorted(map(str, names_b)),
               "identical": len(ta) == len(tb) and all(x[1:3] == y[1:3] for x, y in zip(ta, tb))}
    if summary["identical"]:
        return summary, None
    i = next((k for k in range(min(len(ta), len(tb))) if ta[k][:3] != tb[k][:3]), min(len(ta), len(tb)))
    pos = (0, i, 0)
    if summary["names_equal"]:
        changed = [names_a[k] for k in range(len(ta)) if ta[k][2] != tb[k][2]]
        summary["changed_tools"] = changed[:10]
        kind = "tool_schema_key_order" if all(ta[k][3] == tb[k][3] for k in range(len(ta)) if ta[k][2] != tb[k][2]) else "tool_schema_changed"
    elif summary["same_set"]:
        kind = "tools_reordered"
    else:
        added, removed = sorted(set(map(str, names_b)) - set(map(str, names_a))), sorted(set(map(str, names_a)) - set(map(str, names_b)))
        summary["added"], summary["removed"] = added[:10], removed[:10]
        kind = "tools_changed" if added and removed else "tools_added" if added else "tools_removed" if removed else "tool_schema_changed"
    summary["first_difference_at_tool"] = i
    summary["kind"] = kind
    return summary, _div(kind, pos, field="tools", extra={"tool_index": i})


def diff_message(x: Msg, y: Msg):
    out = []
    if x.parts["role"].raw != y.parts["role"].raw:
        return [_div("role", (2, y.idx, 0), y.role, y.idx, "role")]
    for key in sorted(set(x.parts) | set(y.parts), key=order_of):
        a, b = x.parts.get(key), y.parts.get(key)
        fam = family(key)
        pos = (2, y.idx, order_of(key)[0], order_of(key)[1], order_of(key)[2])
        if a is None and fam in ("text", "reasoning"):
            a = EMPTY
        if b is None and fam in ("text", "reasoning"):
            b = EMPTY
        mk = lambda kind, a=a, b=b: _div(kind, pos, y.role, y.idx, key, a, b)
        if a is None:
            kind = {"call.id": "tool_call_added", "call.name": "tool_call_added", "call.args": "tool_call_added",
                    "result.id": "tool_result_added", "result.text": "tool_result_added", "result.err": "tool_result_changed",
                    "media": "media_changed", "extra": "extra_fields", "text": "text_changed", "shape": "content_shape"}.get(fam, "extra_fields")
            if fam.startswith("call") and not key.endswith(".id"):
                continue                                 # one divergence per call, reported on its id
            if fam.startswith("result") and not key.endswith(".id"):
                continue
            out.append(_div(kind, pos, y.role, y.idx, key))
            continue
        if b is None:
            kind = {"call.id": "tool_call_removed", "result.id": "tool_result_removed", "media": "media_changed", "extra": "extra_fields",
                    "shape": "content_shape"}.get(fam)
            if kind:
                out.append(_div(kind, pos, y.role, y.idx, key))
            continue
        if a.same(b):
            continue
        if fam == "text":
            out.append(mk("text_edge_whitespace" if a.st == b.st else "text_inner_whitespace" if a.ws == b.ws else "text_changed"))
        elif fam == "reasoning":
            if a.n and not b.n:
                out.append(mk("reasoning_other_key" if y.reasoning_other else "reasoning_dropped"))
            elif b.n and not a.n:
                out.append(mk("reasoning_added"))
            else:
                out.append(mk("reasoning_edge_whitespace" if a.st == b.st else "reasoning_changed"))
        elif fam == "call.args":
            if a.st == b.st or (a.norm is not None and a.norm == b.norm):
                out.append(mk("tool_args_whitespace"))
            elif a.srt is not None and a.srt == b.srt:
                out.append(mk("tool_args_key_order"))
            else:
                out.append(mk("tool_args_changed"))
        elif fam == "call.id":
            out.append(mk("tool_call_id_changed"))
        elif fam == "call.name":
            out.append(mk("tool_call_name_changed"))
        elif fam == "result.text":
            out.append(mk("tool_result_edge_whitespace" if a.st == b.st else "tool_result_reformatted" if a.ws == b.ws else "tool_result_changed"))
        elif fam == "result.id":
            out.append(mk("tool_result_id_changed"))
        elif fam == "result.err":
            out.append(mk("tool_result_changed"))
        elif fam == "media":
            out.append(mk("media_changed"))
        elif fam == "shape":
            out.append(mk("content_shape"))
        else:
            out.append(mk("extra_fields"))
    return out


def components(r: Req):
    """B's text components in render order: (position, chars)."""
    comps = [((0, i, 0), t[1]) for i, t in enumerate(r.tools)]
    comps += [((1, j, 0), t.n) for j, t in enumerate(r.system)]
    for m in r.messages:
        for key, t in m.parts.items():
            if family(key) in ("role", "shape", "extra", "call.id", "result.id"):
                continue                                     # not rendered
            o = order_of(key)
            comps.append(((2, m.idx, o[0], o[1], o[2]), t.n))
    return comps


def reasoning_report(a: Req, b: Req, reply: Msg | None):
    def stats(r):
        asst = [m for m in r.messages if m.role == "assistant"]
        return {"assistant_messages": len(asst), "with_reasoning": sum(1 for m in asst if m.parts["reasoning"].n),
                "with_reasoning_under_other_key": sum(1 for m in asst if m.reasoning_other and not m.parts["reasoning"].n)}
    out = {"expected_history": stats(a), "sent_history": stats(b)}
    if reply is not None:
        out["reply_reasoning_chars"] = reply.parts["reasoning"].n
        nxt = b.messages[len(a.messages)] if len(b.messages) > len(a.messages) else None
        out["reply_resent_reasoning_chars"] = None if nxt is None else nxt.parts["reasoning"].n
        out["reply_reasoning_resent"] = None if nxt is None or not reply.parts["reasoning"].n else bool(nxt.parts["reasoning"].n)
    return out


def compare(a: Req, b: Req, snippets: int = 0) -> dict:
    divs = []
    tools_summary, d = diff_tools(a.tools, b.tools)
    if d:
        divs.append(d)
    system = {"blocks_expected": len(a.system), "blocks_sent": len(b.system), "identical": True,
              "chars_expected": sum(t.n for t in a.system), "chars_sent": sum(t.n for t in b.system)}
    if len(a.system) != len(b.system):
        system["identical"] = False
        divs.append(_div("system_blocks_changed", (1, 0, 0), field="system"))
    else:
        for j, (x, y) in enumerate(zip(a.system, b.system)):
            if x.same(y):
                continue
            system["identical"] = False
            kind, extra = classify_system(x, y, snippets)
            cp, exact = common_prefix(x, y)
            system.update(kind=kind, first_diff_char=sum(t.n for t in a.system[:j]) + cp, first_diff_exact=exact,
                          first_diff_share_of_system=round(cp / max(1, x.n), 4), **{k: v for k, v in extra.items() if k.startswith("snippet")})
            divs.append(_div(kind, (1, j, 0), field="system", a=x, b=y, extra={k: v for k, v in extra.items() if k == "note"}))
            break
    # messages: the earlier request's messages + the reply it got must be a prefix of the new request's messages
    expected = list(a.messages)
    reply_checked = a.reply is not None
    if reply_checked:
        expected.append(a.reply)
    n_sent = len(b.messages)
    shorter = False
    for k, x in enumerate(expected):
        if k >= n_sent and k == len(a.messages) and reply_checked:       # B is A again, without the reply
            break
        if k >= n_sent:
            shorter = True
            divs.append(_div("history_shorter", (2, (b.messages[-1].idx + 1 if b.messages else 0), 0), field="messages",
                             extra={"expected_messages": len(expected), "sent_messages": n_sent}))
            break
        y = b.messages[k]
        divs.extend(diff_message(Msg(y.idx, x.role, x.parts, x.reasoning_other), y))   # the reply sits at the next index
    if n_sent < len(a.messages) + 2 and not shorter:     # history, the reply as resent, and something new
        divs.append(_div("not_a_continuation", (2, (b.messages[-1].idx + 1 if b.messages else 0), 0), field="messages"))
    new_roles = [m.role for m in b.messages[len(a.messages) + 1:]]
    divs.sort(key=lambda d: d["pos"])
    first = next((d for d in divs if d["breaks"] in ("yes", "maybe")), None)
    neutral = [d for d in divs if d["breaks"] == "no"]
    breaking = [d for d in divs if d["breaks"] != "no"]
    # shared prefix, in characters of the new request, in render order
    comps = components(b)
    total = sum(n for _, n in comps)
    reply_idx = b.messages[len(a.messages)].idx if len(b.messages) > len(a.messages) else None
    ceiling = total if reply_idx is None else sum(n for p, n in comps if p[0] < 2 or p[1] <= reply_idx)
    if first is None:
        shared = ceiling
    else:
        shared = sum(n for p, n in comps if p < first["pos"]) + (first["common_prefix_chars"] if "chars_sent" in first else 0)
    reasoning = reasoning_report(a, b, a.reply)
    verdict = "reusable" if first is None else ("unclear" if first["breaks"] == "maybe" else "breaks")
    return {"verdict": verdict, "style": b.style, "session": b.session, "client": b.client,
            "messages": {"expected": len(expected), "sent": n_sent, "new_roles": new_roles, "reply_checked": reply_checked},
            "tools": tools_summary, "system": system, "reasoning": reasoning,
            "first_divergence": _public(first), "divergences": [_public(d) for d in breaking[:MAX_DIVERGENCES]],
            "divergences_total": len(breaking), "render_neutral": [_public(d) for d in neutral[:MAX_DIVERGENCES]],
            "prefix": {"shared_chars": shared, "total_chars": total, "share": round(shared / total, 4) if total else None,
                       "ideal_chars": ceiling, "ideal_share": round(ceiling / total, 4) if total else None}}


def _public(d):
    if d is None:
        return None
    out = {k: v for k, v in d.items() if k != "pos"}
    return out


# ------------------------------------------------------------------------------------------------ pairing
def lead_same(a: Req, b: Req) -> int:
    n = 0
    for x, y in zip(a.messages, b.messages):
        if x.fp != y.fp:
            break
        n += 1
    return n


def pairs_of(reqs: list[Req], link: str):
    """[(earlier, later, how)] - for every request, the request it continues."""
    groups: dict = {}
    for r in reqs:
        groups.setdefault(r.session, []).append(r)
    out, orphans = [], []
    for key, g in groups.items():
        for j in range(1, len(g)):
            if link == "consecutive":
                out.append((g[j - 1], g[j], "consecutive"))
                continue
            best, score = g[j - 1], -1
            for i in range(j - 1, -1, -1):
                s = lead_same(g[i], g[j])
                if s > score:
                    best, score = g[i], s
            if score <= 0:
                orphans.append(g[j])                     # no leading message in common with any earlier request
            else:
                skipped = j - 1 - g.index(best)
                out.append((best, g[j], "best prefix" + (f" (skipped {skipped} interleaved request(s))" if skipped else "")))
    out.sort(key=lambda p: p[1].n)
    return out, orphans


# ------------------------------------------------------------------------------------------------ report
def summary_lines(res: dict, a: Req, b: Req, how: str) -> list[str]:
    m, t, s, rs, p = res["messages"], res["tools"], res["system"], res["reasoning"], res["prefix"]
    head = f"request {a.n + 1} -> {b.n + 1}  [{res['style']}{', ' + res['client'] if res.get('client') else ''}; session {str(res['session'])[:16] if res['session'] else 'none'}; {how}]"
    word = {"reusable": "PREFIX REUSABLE", "breaks": "PREFIX BREAKS", "unclear": "UNCLEAR"}[res["verdict"]]
    out = [f"{head}  ->  {word}"]
    out.append(f"  messages  : earlier {len(a.messages)} + reply{'' if m['reply_checked'] else ' (not in the capture)'} -> sent {m['sent']}"
               + (f"; new: {', '.join(m['new_roles'])}" if m["new_roles"] else ""))
    if t["identical"]:
        out.append(f"  tools     : identical ({t['count_sent']} tools, same bytes, same order)" if t["count_sent"] else "  tools     : none sent")
    else:
        out.append(f"  tools     : DIFFERENT ({t['kind']}); {t['count_expected']} -> {t['count_sent']} tools; first difference at tool {t.get('first_difference_at_tool')}"
                   + (f"; changed: {', '.join(map(str, t['changed_tools']))}" if t.get("changed_tools") else "")
                   + (f"; added: {', '.join(t['added'])}" if t.get("added") else "") + (f"; removed: {', '.join(t['removed'])}" if t.get("removed") else ""))
    if s["identical"]:
        out.append(f"  system    : identical ({s['chars_sent']:,} chars)")
    else:
        out.append(f"  system    : DIFFERENT {s.get('kind', 'blocks')} (first difference at char {s.get('first_diff_char', 0):,} of {s['chars_sent']:,}"
                   f"{'' if s.get('first_diff_exact', True) else ', to 512'})")
        if s.get("snippet_expected") is not None:
            out.append(f"    expected: {s['snippet_expected']!r}\n    sent    : {s['snippet_sent']!r}")
    r = rs
    if "reply_reasoning_chars" in r:
        if not r["reply_reasoning_chars"]:
            out.append("  reasoning : the reply had none (thinking off or none shown), nothing to resend")
        elif r["reply_reasoning_resent"] is None:
            out.append(f"  reasoning : the reply had {r['reply_reasoning_chars']:,} chars; the new request has no message after the history")
        else:
            out.append(f"  reasoning : the reply had {r['reply_reasoning_chars']:,} chars; resent: {'YES (' + format(r['reply_resent_reasoning_chars'], ',') + ' chars)' if r['reply_reasoning_resent'] else 'NO - dropped'}")
    h = r["sent_history"]
    out.append(f"  history   : {h['with_reasoning']} of {h['assistant_messages']} assistant messages carry reasoning"
               + (f" ({h['with_reasoning_under_other_key']} under another key)" if h["with_reasoning_under_other_key"] else ""))
    f = res["first_divergence"]
    if f:
        where = f"message {f['index']} ({f['role']}), {f['field']}" if f.get("index") is not None else f["field"]
        out.append(f"  first break: {f['kind']} at {where}"
                   + (f"; {f['chars_expected']:,} chars expected, {f['chars_sent']:,} sent, shared {f['common_prefix_chars']:,}{'' if f.get('prefix_exact', True) else ' (to 512)'}" if "chars_sent" in f else "")
                   + f"  [{EFFECTS[f['kind']][0]}]")
        if f.get("snippet_expected") is not None:
            out.append(f"    expected: {f['snippet_expected']!r}   sent: {f['snippet_sent']!r}")
        more = res["divergences_total"] - 1
        if more:
            out.append(f"  further breaking differences: {more} (kinds: {', '.join(sorted({d['kind'] for d in res['divergences'][1:]}))})")
    if res["render_neutral"]:
        out.append(f"  neutral   : {', '.join(sorted({d['kind'] for d in res['render_neutral']}))} (the render is identical)")
    out.append(f"  shared prefix: {p['shared_chars']:,} of {p['total_chars']:,} chars = {100 * (p['share'] or 0):.1f}%  (a perfect client: {100 * (p['ideal_share'] or 0):.1f}%)")
    return out


class Results(list):
    """The (result, earlier, later, how) tuples; `.orphans` are the requests that continue no earlier one."""
    orphans: list = []


def run(recs: list[dict], link="auto", snippets=0, only_session=None):
    reqs = [Req(r, i) for i, r in enumerate(recs)]
    reqs = [r for r in reqs if r.messages and (only_session is None or r.session == only_session)]
    results = Results()
    pairs, results.orphans = pairs_of(reqs, link)
    for a, b, how in pairs:
        res = compare(a, b, snippets)
        res.update(earlier=a.n + 1, later=b.n + 1, linked_by=how)
        results.append((res, a, b, how))
    return reqs, results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("captures", nargs="+", help="JSONL/JSON capture file(s)")
    ap.add_argument("--link", choices=("auto", "consecutive"), default="auto",
                    help="auto: pair each request with the earlier one of its session that shares the most leading messages "
                         "(skips interleaved side requests); consecutive: strictly the previous request of the session")
    ap.add_argument("--session", help="only requests with this session key")
    ap.add_argument("--json", metavar="OUT", help="write the full result as JSON")
    ap.add_argument("--redact", metavar="OUT", help="write a copy of the capture with texts replaced by length + hashes")
    ap.add_argument("--snippets", type=int, default=0, metavar="N", help="show N characters of text around a system-prompt difference (default 0: no text from the capture in the report)")
    ap.add_argument("--no-report", action="store_true", help="with --redact: write the copy and print nothing else")
    a = ap.parse_args(argv)
    recs = load(a.captures)
    if not recs:
        print("no chat requests found in the input", file=sys.stderr)
        return 2
    if a.redact:
        with open(a.redact, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(redact_record(r), ensure_ascii=False) + "\n")
        print(f"wrote {len(recs)} redacted record(s) to {a.redact}")
        if a.no_report:
            return 0
    reqs, results = run(recs, a.link, a.snippets, a.session)
    print(f"{len(recs)} request(s), {len(reqs)} with messages, {len(results)} pair(s)\n")
    if results.orphans:
        print(f"{len(results.orphans)} request(s) share no leading message with any earlier request of their session (requests "
              f"{', '.join(str(r.n + 1) for r in results.orphans[:12])}{' ...' if len(results.orphans) > 12 else ''}): new conversations or side requests - "
              f"or the FIRST message changes on every turn; --link consecutive compares strictly consecutive requests\n")
    if not results:
        print("nothing to compare: a pair needs two requests of one session (see --link, --session)")
    for res, x, y, how in results:
        print("\n".join(summary_lines(res, x, y, how)))
        print()
    kinds: dict = {}
    for res, *_ in results:
        for d in res["divergences"][:1]:
            kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
    verdicts = {v: sum(1 for r, *_ in results if r["verdict"] == v) for v in ("reusable", "unclear", "breaks")}
    if results:
        shares = [r["prefix"]["share"] for r, *_ in results if r["prefix"]["share"] is not None]
        print(f"summary: {verdicts['reusable']} reusable, {verdicts['unclear']} unclear, {verdicts['breaks']} break the prefix"
              + (f"; mean shared prefix {100 * sum(shares) / len(shares):.1f}% of characters" if shares else ""))
        for kind, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
            effect, means, fix = EFFECTS[kind]
            print(f"  first break {kind} x{n}: {means}\n    fix: {fix}")
        print("note: effects are read from serve/chat_template.jinja and serve/frontend.py, not measured; the server's own fixes "
              "(stable tool order, billing stamp, env_to_user) are not applied here.")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"requests": len(recs), "pairs": [res for res, *_ in results], "verdicts": verdicts,
                       "first_break_kinds": kinds}, f, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
