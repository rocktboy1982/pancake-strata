"""STRATA_TOOL_SNIP=N (opt-in, lossy for the prompt text; unset, empty or 0 = every tool result exactly as sent).

ContextSniper-style tool-output filtering before the result enters the context (paper "ContextSniper", LEADS.md
Part E: about half the tokens on SWE-bench Lite).  The paper's learned selector is not available here, so this is
the deterministic rule from its description: a tool result longer than N characters keeps its first and last lines
and every line that looks like a diagnosis (error, exception, traceback, failure, assertion, warning, a file:line
reference), with one marker line per omitted run that says how many lines were left out.

Deterministic on purpose: the same result gives the same text in every later request, so the prompt prefix the
engine cached stays the same.  Images in a result are kept untouched; only text items are filtered.

Knobs (read once, first use): STRATA_TOOL_SNIP=N (character threshold), STRATA_TOOL_SNIP_HEAD (lines kept at the
start, default 40), STRATA_TOOL_SNIP_TAIL (lines kept at the end, default 40), STRATA_TOOL_SNIP_KEEP (cap on the
diagnosis lines kept from the middle, default 200).
"""
from __future__ import annotations

import os
import re

_CFG: tuple[int, int, int, int] | None = None

# Lines worth keeping from the middle of a long result: compiler, test and runtime diagnostics.
_DIAG = re.compile(r"error|exception|traceback|fail|assert|warning|panic|fatal|denied|not found|"
                   r"\b[\w./\\-]+\.\w+:\d+", re.IGNORECASE)


def _int_env(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, "").strip() or default))
    except ValueError:
        return default


def config() -> tuple[int, int, int, int]:
    """(threshold, head, tail, keep); threshold 0 = off."""
    global _CFG
    if _CFG is None:
        _CFG = (_int_env("STRATA_TOOL_SNIP", 0), _int_env("STRATA_TOOL_SNIP_HEAD", 40),
                _int_env("STRATA_TOOL_SNIP_TAIL", 40), _int_env("STRATA_TOOL_SNIP_KEEP", 200))
    return _CFG


def snip_text(text: str, threshold: int, head: int, tail: int, keep: int) -> str:
    """The filtered text, or `text` itself when it is not longer than `threshold` or nothing would be left out."""
    if threshold <= 0 or len(text) <= threshold:
        return text
    lines = text.split("\n")
    n = len(lines)
    if n <= head + tail:
        return text
    kept = set(range(head)) | set(range(n - tail, n))
    mid = [i for i in range(head, n - tail) if _DIAG.search(lines[i])]
    kept.update(mid[:keep])
    if len(kept) >= n:
        return text
    out, gap = [], 0
    for i in range(n):
        if i in kept:
            if gap:
                out.append(f"[... {gap} lines omitted (STRATA_TOOL_SNIP) ...]")
                gap = 0
            out.append(lines[i])
        else:
            gap += 1
    if gap:
        out.append(f"[... {gap} lines omitted (STRATA_TOOL_SNIP) ...]")
    return "\n".join(out)


def snip_tool_results(messages: list[dict]) -> None:
    """Filters, in place, the content of every role "tool" message (a string, or the template's list form whose
    text items are filtered one by one).  Off (the default): returns at once, nothing is touched."""
    threshold, head, tail, keep = config()
    if threshold <= 0:
        return
    for m in messages:
        if m.get("role") != "tool":
            continue
        c = m.get("content")
        if isinstance(c, str):
            m["content"] = snip_text(c, threshold, head, tail, keep)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    part["text"] = snip_text(part["text"], threshold, head, tail, keep)
