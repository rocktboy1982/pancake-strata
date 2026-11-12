"""request_capture - opt-in record of the raw chat requests a client sends and the replies it got (MOD_PLAN 17.0, M4).

Opt-in: STRATA_REQUEST_CAPTURE=DIR.  Unset (the default) nothing is recorded, nothing is printed and no code of this
module runs on the request path.

Why it exists: the conversation cache only helps when the client resends a BYTE-IDENTICAL prefix on the next turn.  The
turn log (STRATA_TURN_LOG) holds numbers only, and the API monitor (--api-monitor) keeps the last 100 requests in
memory, cut at 256 KB each (a 55K-token agent prompt is larger).  Neither can answer "what did the client send?", so
this writes one JSON line per /v1/chat/completions or /v1/messages request to a file in DIR:

    {"ts": "2026-11-11T20:15:01", "t": 1762888501.2, "path": "/v1/chat/completions", "session": "<header value or null>",
     "client": {"user-agent": "..."}, "request": {...the body as parsed, key order kept...},
     "response": {"message": {...the assistant message in the API's own wire format...}, "finish_reason": "...",
                  "usage": {...}}, "http_status": 200}

Read it with tools/pancake/prefix_check.py (pancake-docs/PREFIX_CHECK.md).

PRIVACY: the request body is the user's prompts, code and tool output, and the response is the model's answer.  The file
is the FULL text.  It goes to the local directory the operator names and nowhere else: this module opens no socket and
sends nothing.  The directory is created private (mode 0700), the file 0600.  No Authorization header, API key or cookie
is recorded: only the User-Agent and any header whose name contains "session", "conversation" or "thread" (a client's
conversation id, used to tell sessions apart).  Delete the file when the measurement is done.  To share a capture, run
`prefix_check.py --redact` on it first: that replaces every message text by a length and hashes.

Bounded: STRATA_REQUEST_CAPTURE_MAX_MB (default 512) is the most the module writes; past it the capture stops with one
line in the log (the root disk of the dev box is small).  An error (disk full, a permission) turns the capture off with
one line; it never fails or slows a request beyond the write.
"""
from __future__ import annotations

import json
import os
import threading
import time

ENV_DIR = "STRATA_REQUEST_CAPTURE"
ENV_MAX_MB = "STRATA_REQUEST_CAPTURE_MAX_MB"
DEFAULT_MAX_MB = 512
PATHS = ("/v1/chat/completions", "/v1/messages")
SESSION_WORDS = ("session", "conversation", "thread")


def from_env(env=None):
    """A RequestCapture for STRATA_REQUEST_CAPTURE=DIR, or None (unset or empty: off)."""
    env = os.environ if env is None else env
    directory = (env.get(ENV_DIR) or "").strip()
    if not directory:
        return None
    try:
        max_mb = float(env.get(ENV_MAX_MB) or DEFAULT_MAX_MB)
    except ValueError:
        max_mb = DEFAULT_MAX_MB
    return RequestCapture(os.path.abspath(os.path.expanduser(directory)), max_bytes=int(max_mb * (1 << 20)))


def _client_headers(headers) -> tuple[dict, str | None]:
    """(the few headers worth keeping, the session key or None).  Never an Authorization header, key or cookie."""
    keep, session = {}, None
    try:
        items = list(headers.items()) if headers is not None else []
    except Exception:                                    # a headers object this module does not know: record none
        items = []
    for name, value in items:
        low = str(name).lower()
        if low == "user-agent":
            keep[low] = str(value)[:200]
        elif any(w in low for w in SESSION_WORDS) and "auth" not in low and "cookie" not in low:
            keep[low] = str(value)[:200]
            session = session or str(value)[:200]
    return keep, session


class RequestCapture:
    def __init__(self, directory: str, max_bytes: int = DEFAULT_MAX_MB << 20):
        self.directory = directory
        self.max_bytes = max_bytes
        self.written = 0
        self.off = False
        self._lock = threading.Lock()
        os.makedirs(directory, mode=0o700, exist_ok=True)
        self.path = os.path.join(directory, time.strftime("requests-%Y%m%d-%H%M%S") + f"-{os.getpid()}.jsonl")

    # ---------------------------------------------------------------------------------------------- per request
    def begin(self, path: str, req, headers=None):
        """The entry for one request, or None (a path that is not a chat route, the capture is off).  The body is
        serialised NOW, before the server edits it (shared defaults, grammar), so the line shows what the client sent."""
        if self.off or path not in PATHS or not isinstance(req, dict):
            return None
        try:
            request_json = json.dumps(req, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            return None
        client, session = _client_headers(headers)
        return {"t": time.time(), "path": path, "session": session, "client": client, "request_json": request_json,
                "text": [], "reasoning": [], "calls": {}, "blocks": [], "finish": None, "usage": None,
                "http_status": None, "stream": bool(req.get("stream"))}

    def observe(self, entry: dict, api: str, item) -> None:
        """One streamed item of the reply (an OpenAI chunk, or an Anthropic (event name, event) pair)."""
        try:
            if item is None:
                return
            if api == "openai":
                self._openai(entry, item)
            elif api == "anthropic":
                self._anthropic(entry, item)
        except Exception:                                # a shape this module did not expect never reaches the request
            entry["observe_error"] = True

    @staticmethod
    def _openai(entry: dict, chunk: dict) -> None:
        if chunk.get("usage"):
            entry["usage"] = chunk["usage"]
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                entry["text"].append(delta["content"])
            if delta.get("reasoning_content"):
                entry["reasoning"].append(delta["reasoning_content"])
            for call in delta.get("tool_calls") or []:
                slot = entry["calls"].setdefault(call.get("index", len(entry["calls"])),
                                                 {"id": None, "name": None, "arguments": []})
                if call.get("id"):
                    slot["id"] = call["id"]
                fn = call.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"].append(fn["arguments"])
            if choice.get("finish_reason"):
                entry["finish"] = choice["finish_reason"]

    @staticmethod
    def _anthropic(entry: dict, item) -> None:
        name, event = item
        blocks = entry["blocks"]
        if name == "content_block_start":
            blocks.append(dict(event.get("content_block") or {}))
            if blocks[-1].get("type") == "tool_use":
                blocks[-1]["_json"] = []
        elif name == "content_block_delta" and blocks:
            delta, block = event.get("delta") or {}, blocks[-1]
            if delta.get("type") == "text_delta":
                block["text"] = block.get("text", "") + delta.get("text", "")
            elif delta.get("type") == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
            elif delta.get("type") == "input_json_delta":
                block.setdefault("_json", []).append(delta.get("partial_json", ""))
        elif name == "message_delta":
            entry["finish"] = (event.get("delta") or {}).get("stop_reason") or entry["finish"]
            entry["usage"] = event.get("usage") or entry["usage"]

    def response_message(self, entry: dict) -> dict | None:
        """The assistant message the client received, in the wire format of the route it called (what the client will
        resend as that turn): OpenAI {"role", "content", "reasoning_content", "tool_calls"}; Anthropic {"role",
        "content": [blocks]}.  A tool call's arguments stay the exact text that was streamed."""
        if entry["path"] == "/v1/messages":
            if not entry["blocks"]:
                return None
            blocks = []
            for b in entry["blocks"]:
                b = dict(b)
                if b.get("type") == "tool_use":
                    raw = "".join(b.pop("_json", []))
                    try:
                        b["input"] = json.loads(raw) if raw else b.get("input", {})
                    except ValueError:
                        b["input"] = {"_unparsed_arguments": raw}
                blocks.append(b)
            return {"role": "assistant", "content": blocks}
        if not (entry["text"] or entry["reasoning"] or entry["calls"]):
            return None
        msg = {"role": "assistant", "content": "".join(entry["text"]) or None}
        if entry["reasoning"]:
            msg["reasoning_content"] = "".join(entry["reasoning"])
        if entry["calls"]:
            msg["tool_calls"] = [{"id": c["id"], "type": "function",
                                  "function": {"name": c["name"], "arguments": "".join(c["arguments"])}}
                                 for _, c in sorted(entry["calls"].items())]
        return msg

    def finish(self, entry: dict) -> None:
        """Write the line (thread-safe).  A failure turns the capture off and says so once."""
        if self.off:
            return
        try:
            response = {"message": self.response_message(entry), "finish_reason": entry["finish"],
                        "usage": entry["usage"]}
            ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(entry["t"]))
            head = json.dumps({"ts": ts, "t": round(entry["t"], 3), "path": entry["path"], "session": entry["session"],
                               "client": entry["client"], "stream": entry["stream"]}, ensure_ascii=False)
            tail = json.dumps({"response": response, "http_status": entry["http_status"]}, ensure_ascii=False)
            # the request goes in as the text serialised at begin(): its key order is the client's
            line = head[:-1] + ',"request":' + entry["request_json"] + "," + tail[1:] + "\n"
            data = line.encode("utf-8")
            with self._lock:
                if self.written + len(data) > self.max_bytes:
                    self.off = True
                    print(f"[strata] request capture stopped: {self.path} reached {self.max_bytes >> 20} MiB "
                          f"({ENV_MAX_MB} raises the limit)", flush=True)
                    return
                fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "ab") as f:             # (buffered: a large line is written whole)
                    f.write(data)
                self.written += len(data)
        except (OSError, TypeError, ValueError) as e:
            self.off = True
            print(f"[strata] request capture off: {self.path}: {e}", flush=True)
