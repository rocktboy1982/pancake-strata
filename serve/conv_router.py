"""STRATA_CONV_ROUTE=off|shadow|on (opt-in; unset, empty or "off" = the conversation cache is never retuned) and
STRATA_TUNE=1 (opt-in; unset = no POST /v1/tune endpoint, 404 as before).

Port of Strata PR #5516 by rwkeyes (head e929c26411; serve/conv_router.py + the TUNE line; LEADS.md "Strata POST /v1/tune + --conv-route").
Decide, per request, whether the prompt/conversation cache should be on, and switch the running engine with a
`TUNE prompt_cache=N` line when the traffic shape changes.  The upstream measurement (Arc Pro B70, in LEADS.md) is a
single-shot TTFT -42% and +10.1% prefill on code with the cache off; on our box it is unmeasured.

The decision is deliberately simple and biased to KEEPING the cache:
  - any request with more than one user turn, tools, or a prefix seen recently  -> cache ON (a repeat is coming)
  - a run of single-turn, never-seen prompts (batch_after in a row)             -> cache OFF
  - one conversational request switches it back immediately
A wrong "off" costs one full prefill on the next repeat; a wrong "on" costs the parking share on a fresh request.
The asymmetry is why the counter is a run and not a single request.

Nothing in Router talks to the engine: `decide()` is pure and keeps only counters and a small prefix history, so it is
testable without a GPU.  `tune_line()` / `parse_tuned()` are the two halves of the engine protocol
(src/program/generate.cpp, the TUNE handler).

Knobs (read once, first use): STRATA_CONV_ROUTE (mode), STRATA_CONV_ROUTE_BATCH_AFTER (default 4),
STRATA_CONV_ROUTE_CACHE_ON (the prompt_cache value restored in chat mode, default 6 = the engine's default),
STRATA_TUNE=1 (POST /v1/tune).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

CHAT = "chat"        # cache on
BATCH = "batch"      # cache off

_CFG: dict | None = None


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, "").strip() or default))
    except ValueError:
        return default


def config() -> dict:
    """{"mode": off|shadow|on, "batch_after": int, "cache_on": int, "tune": bool}; read once."""
    global _CFG
    if _CFG is None:
        mode = os.environ.get("STRATA_CONV_ROUTE", "").strip().lower()
        _CFG = {"mode": mode if mode in ("shadow", "on") else "off",          # a typo must not enable it
                "batch_after": _int_env("STRATA_CONV_ROUTE_BATCH_AFTER", 4),
                "cache_on": _int_env("STRATA_CONV_ROUTE_CACHE_ON", 6),
                "tune": os.environ.get("STRATA_TUNE", "").strip() == "1"}
    return _CFG


@dataclass
class Decision:
    mode: str
    reason: str
    changed: bool                  # the engine should be retuned before this request runs
    wanted_cache: int              # the prompt_cache value to apply (0 = off)

    def __str__(self) -> str:
        return f"{self.mode} ({self.reason})"


@dataclass
class Router:
    """Hysteresis: `batch_after` consecutive single-shot requests are needed to switch the cache off."""

    cache_on: int = 6
    batch_after: int = 4
    repeat_window: int = 64         # how many recent prefixes count as "seen recently"
    mode: str = CHAT
    consecutive_single: int = 0
    seen: dict = field(default_factory=dict)
    stats: dict = field(default_factory=lambda: {"requests": 0, "chat": 0, "batch": 0,
                                                 "switches_on": 0, "switches_off": 0, "repeats": 0})

    @staticmethod
    def user_turns(messages) -> int:
        return sum(1 for m in (messages or []) if isinstance(m, dict) and (m.get("role") or "").lower() == "user")

    @staticmethod
    def prefix_key(messages, chars: int = 1024) -> str:
        """The first user turn, trimmed and hashed: an agent appends, so the first turn is what stays stable."""
        for m in (messages or []):
            if isinstance(m, dict) and (m.get("role") or "").lower() == "user":
                text = m.get("content")
                if not isinstance(text, str):
                    text = " ".join(p.get("text", "") for p in text if isinstance(p, dict)) \
                        if isinstance(text, list) else ""
                return hashlib.blake2b(" ".join(text.split())[:chars].encode(), digest_size=8).hexdigest()
        return ""

    def decide(self, messages, tools=None) -> Decision:
        self.stats["requests"] += 1
        turns = self.user_turns(messages)
        key = self.prefix_key(messages)
        repeat = bool(key) and key in self.seen
        if key:
            self.seen[key] = self.seen.get(key, 0) + 1
            if len(self.seen) > self.repeat_window:
                del self.seen[next(iter(self.seen))]
        if repeat:
            self.stats["repeats"] += 1
        if turns > 1 or bool(tools) or repeat:
            self.consecutive_single = 0
            why = "repeat prefix" if repeat else ("tools" if tools else f"{turns} user turns")
            return self._to(CHAT, why)
        self.consecutive_single += 1
        if self.mode == BATCH:
            self.stats["batch"] += 1
            return Decision(BATCH, "single-shot (already batch)", False, 0)
        if self.consecutive_single >= self.batch_after:
            return self._to(BATCH, f"{self.consecutive_single} single-shot requests in a row")
        self.stats["chat"] += 1
        return Decision(CHAT, f"single-shot {self.consecutive_single}/{self.batch_after}", False, self.cache_on)

    def _to(self, mode: str, reason: str) -> Decision:
        if mode == self.mode:
            self.stats["chat" if mode == CHAT else "batch"] += 1
            return Decision(mode, reason, False, self.cache_on if mode == CHAT else 0)
        self.mode = mode
        self.stats["switches_on" if mode == CHAT else "switches_off"] += 1
        self.stats["chat" if mode == CHAT else "batch"] += 1
        if mode == CHAT:
            return Decision(CHAT, f"back to chat: {reason}", True, self.cache_on)
        return Decision(BATCH, f"batch: {reason}", True, 0)

    def revert(self) -> None:
        """A refused tune: the engine still has the previous value, so the mode rolls back and the next decision
        retries."""
        self.mode = CHAT if self.mode == BATCH else BATCH
        self.consecutive_single = 0

    def report(self) -> dict:
        return {"mode": self.mode, "consecutive_single": self.consecutive_single,
                "batch_after": self.batch_after, "cache_on": self.cache_on, **self.stats}


def tune_line(settings: dict) -> str:
    """The engine line for `settings` ({"prompt_cache": 0, "spec_min_p": 0.4}); ValueError when empty."""
    parts = []
    for k, v in settings.items():
        key = str(k).strip().lstrip("-").replace("-", "_")
        if isinstance(v, bool):
            v = int(v)
        if not key or any(c.isspace() for c in key) or any(c.isspace() for c in str(v)):
            raise ValueError(f"tune: bad setting {k!r}={v!r}")
        parts.append(f"{key}={v}")
    if not parts:
        raise ValueError("tune: needs at least one setting")
    return "TUNE " + " ".join(parts)


def parse_tuned(line: str) -> dict:
    """`TUNED k=v ... [refused: k=v(why) ...]` -> {"applied": {k: v}, "refused": ["k=v", ...]}."""
    head, _, tail = line.strip()[5:].strip().partition("refused:")
    applied: dict = {}
    for kv in head.split():
        k, _, v = kv.partition("=")
        if not k or not v:
            continue
        v = v.split("(", 1)[0]                       # "prefill_chunk=512(capped at the startup chunk)"
        try:
            applied[k] = int(v) if v.lstrip("-").isdigit() else float(v)
        except ValueError:
            applied[k] = v
    refused = []
    for t in tail.split(" "):
        t = t.split("(", 1)[0]
        if "=" in t:
            refused.append(t)
    return {"applied": applied, "refused": refused}
