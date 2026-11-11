"""park_divergence - why did a request not continue the engine's last conversation? (MOD_PLAN 17.0, measurement M3)

Opt-in: STRATA_PARK_DIVERGENCE_LOG=1.  Off (the default) nothing is recorded and nothing is printed.

The engine continues its live conversation only when the new prompt starts with ALL of its tokens (the last prompt plus
the reply it generated).  When it does not, the engine falls back to a checkpoint, a parked conversation or a full read,
and the analysis in pancake-docs/TTFT_LEVERS.md found that 6 of 46 warm agent turns did this for a reason nobody knew.
This module sees what the server sends and what the engine answers, so it can say where the two token streams part.

For every request after the first, when the new prompt does not extend the previous request's prompt + reply, one line
is printed:

    [strata] park divergence {"diverge_at": 61234, "live_len": 61902, "prompt_len": 62500, ...}

Fields (all token counts or indexes into the token stream of the request):
  diverge_at       first index where the new prompt and the previous prompt + reply differ
  live_len         length of the previous prompt + reply (what the engine holds, as far as the server can tell)
  prompt_len       length of the new prompt
  prev_prompt_len  length of the previous prompt;  prev_gen_len the tokens the engine generated for it
  reply_offset     diverge_at - prev_prompt_len: negative = inside the previous prompt (an earlier message changed, or this is
                   another conversation); 0 or more = inside the previous reply (it was rendered again differently)
  where            "history", "reply-start", "reply-middle", "reply-end" or "prompt-is-prefix"
  resume           the tokens the engine says it reused (its RESUME line)
  live_ids / new_ids, live_text / new_text   a short window around diverge_at on both sides, as ids and decoded text
  reply_has_think_close, reasoning_dropped_guess   whether the previous reply contained a think-close tag and a guess (not a
                   measurement) that the new prompt carries that reply without its reasoning

PRIVACY: live_text and new_text are decoded prompt or reply text.  They are cut to STRATA_PARK_DIVERGENCE_CHARS characters
(default 48; 0 = ids only, no text at all).  Do not share this log without looking at it.

Limits: the server tracks only the one-engine, one-conversation-at-a-time path (not --batch, not DeepSeek --parallel, not
image requests, not split serving); after any of those the next request is compared with nothing.  The "live" tokens are
what the server sent plus the token ids the engine streamed back; the engine's own state may be a few tokens shorter if a
reply was stopped early.  The line is a diagnostic: it never changes a request, and an error in it turns it off.
"""
from __future__ import annotations

import json
import os
import sys

ENV_ON = "STRATA_PARK_DIVERGENCE_LOG"
ENV_CHARS = "STRATA_PARK_DIVERGENCE_CHARS"
DEFAULT_CHARS = 48
WINDOW_BEFORE = 4     # tokens shown before diverge_at
WINDOW_AFTER = 12     # tokens shown from diverge_at on
THINK_CLOSE = "</think>"


def enabled(env=None) -> bool:
    v = (os.environ if env is None else env).get(ENV_ON, "")
    return v.strip().lower() in ("1", "true", "yes", "on")


def text_chars(env=None) -> int:
    v = (os.environ if env is None else env).get(ENV_CHARS, "")
    try:
        return max(0, int(v)) if v.strip() else DEFAULT_CHARS
    except ValueError:
        return DEFAULT_CHARS


def first_divergence(live, ids) -> int:
    """Length of the common prefix of two token lists: the index of the first token where they differ, or the shorter
    length when one is a prefix of the other."""
    n = min(len(live), len(ids))
    i = 0
    while i < n and live[i] == ids[i]:
        i += 1
    return i


def _window(tokens, at: int):
    lo = max(0, at - WINDOW_BEFORE)
    return tokens[lo:at + WINDOW_AFTER]


def _clip(text: str, chars: int) -> str:
    return text if len(text) <= chars else text[:chars] + "..."


def classify(d: int, prev_prompt_len: int, live_len: int, prompt_len: int) -> str:
    """Where the divergence sits.  `reply-start` is the first 3 tokens of the reply (the think block's opening: a template
    that renders the reply without its reasoning parts here), `reply-end` the last 3 (a trailing newline or end-of-turn
    token), `prompt-is-prefix` a new prompt that is shorter than, and a prefix of, the previous prompt + reply."""
    if d >= prompt_len and prompt_len < live_len:
        return "prompt-is-prefix"
    if d < prev_prompt_len:
        return "history"
    if d - prev_prompt_len <= 2:
        return "reply-start"
    if d >= live_len - 3:
        return "reply-end"
    return "reply-middle"


def record(live, ids, prev_prompt_len: int, resume, decode=None, chars: int = DEFAULT_CHARS):
    """The log record for a request whose prompt `ids` does not start with `live`, or None when it does (the engine
    continues the live state: nothing to explain).  `decode` turns a list of ids into text (None: ids only)."""
    d = first_divergence(live, ids)
    if d >= len(live):
        return None
    rec = {"diverge_at": d, "live_len": len(live), "prompt_len": len(ids), "prev_prompt_len": prev_prompt_len,
           "prev_gen_len": max(0, len(live) - prev_prompt_len), "reply_offset": d - prev_prompt_len,
           "where": classify(d, prev_prompt_len, len(live), len(ids)), "resume": resume,
           "live_ids": _window(live, d), "new_ids": _window(ids, d)}
    if decode is not None and chars > 0:
        try:
            rec["live_text"] = _clip(decode(rec["live_ids"]), chars)
            rec["new_text"] = _clip(decode(rec["new_ids"]), chars)
            reply = live[prev_prompt_len:]
            if reply:
                old = decode(reply)
                rec["reply_has_think_close"] = THINK_CLOSE in old
                if rec["reply_has_think_close"]:
                    head = old.strip()[:24]
                    region = decode(ids[prev_prompt_len:prev_prompt_len + len(reply) + 8])
                    rec["reasoning_dropped_guess"] = bool(head) and head not in region
        except Exception as e:    # a tokenizer that cannot decode a window: the ids are still there
            rec["text_error"] = type(e).__name__
            rec.pop("live_text", None)
            rec.pop("new_text", None)
    return rec


def format_line(rec: dict) -> str:
    return "[strata] park divergence " + json.dumps(rec, ensure_ascii=False, separators=(",", ":"))


class DivergenceTracker:
    """Follows one engine's requests: begin() when a request is about to be sent, resumed() when the engine's RESUME line
    arrives, token() for every streamed token, end() when the request is over.  Not thread safe on purpose: the server
    sends one request at a time to the engine this tracks (the other paths do not call it)."""

    def __init__(self, decode=None, emit=None, chars: int = DEFAULT_CHARS):
        self.decode = decode
        self.emit = emit if emit is not None else (lambda line: print(line, file=sys.stderr, flush=True))
        self.chars = chars
        self.live = None          # the previous prompt + reply, or None when unknown
        self.key = None           # which engine run it belongs to
        self.prev_prompt_len = 0
        self._ids = None
        self._gen = None
        self._prev = None
        self._prev_prompt_len = 0
        self.broken = False
        self.logged = 0

    def begin(self, key, ids) -> None:
        if self.broken:
            return
        try:
            self._prev = self.live if self.key == key else None
            self._prev_prompt_len = self.prev_prompt_len
            self._ids, self._gen = list(ids), []
            self.key, self.live = key, None    # unknown until this request has ended
        except Exception:
            self.broken = True

    def forget(self) -> None:
        """A request this tracker cannot follow (batch, pictures, split): the next one is compared with nothing."""
        self.live, self.key, self._ids, self._gen = None, None, None, None

    def resumed(self, reused):
        """The engine reported how many prompt tokens it reuses.  Prints the line when this request does not continue the
        previous one; returns the record (or None)."""
        if self.broken or self._prev is None or self._ids is None:
            return None
        try:
            rec = record(self._prev, self._ids, self._prev_prompt_len, reused, self.decode, self.chars)
            if rec is not None:
                self.logged += 1
                self.emit(format_line(rec))
            return rec
        except Exception:
            self.broken = True
            return None
        finally:
            self._prev = None

    def token(self, t) -> None:
        if self._gen is not None:
            self._gen.append(int(t))

    def end(self) -> None:
        if self.broken or self._ids is None:
            return
        self.live = self._ids + (self._gen or [])
        self.prev_prompt_len = len(self._ids)
        self._ids = self._gen = None


def tracker_from_env(env=None):
    """The tracker when STRATA_PARK_DIVERGENCE_LOG is on, else None (the server then does nothing at all)."""
    return DivergenceTracker(chars=text_chars(env)) if enabled(env) else None


def attach_decoder(engine, tokenizer) -> None:
    """Give the engine's tracker (if it has one) the service's tokenizer for the text windows."""
    tr = getattr(engine, "divergence", None)
    if tr is not None and tr.decode is None and tokenizer is not None and hasattr(tokenizer, "decode"):
        tr.decode = lambda ids: tokenizer.decode(list(ids))
