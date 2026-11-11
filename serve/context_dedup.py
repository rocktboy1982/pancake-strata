"""serve/context_dedup.py - ContextPilot's block de-duplication for agent prompts (LOSSY opt-in).

Ported from ContextPilot (University of Edinburgh, MLSys 2026, arXiv 2511.03475;
github.com/EfficientContext/ContextPilot b4c80327, contextpilot/dedup/block_dedup.py and the whole-result dedup of
opencode-plugin/src/index.ts; Apache-2.0, Copyright the ContextPilot authors).  Changes for pancake: it works on the
server's template messages (after frontend.openai_to_messages / anthropic_to_messages, where tool-call ids are already
gone, so a tool result's name comes from its position after the assistant turn that called it); assistant turns are
sources only and never rewritten (the engine's conversation cache holds the reply as the model wrote it: a rewritten
reply would no longer match it); the line-number prefix also accepts Claude Code's `   12→`; whole-result dedup is
stateless (an earlier identical tool result in the same request, not a per-session table).

What it does: a later copy of a block of lines (content-defined chunks, 3-40 lines, cut where a line's hash is 0 mod
13) that already appeared earlier in the same request - in the system prompt, a user turn, an assistant turn or a
tool result - is replaced by a one-line reference to the earlier copy.  A whole tool result identical to an earlier
one becomes one reference line.  Fewer prompt tokens to read, for every layer (GDN and attention alike).

LOSSY: the model sees a reference instead of the text (a re-read file whose lines moved loses its line numbers on the
blocks that are replaced).  Off by default.

Prefix-stable by construction: what message k becomes depends only on messages 0..k (references point backward and
the decision is a pure function of what came before), so a conversation that grows by appending keeps every earlier
message's rewritten text, and the engine's conversation cache (exact prefix reuse, the only reuse that is exact for
the Gated DeltaNet layers) still hits.  serve/test_context_dedup.py checks this."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

MIN_BLOCK_CHARS = 40        # a chunk shorter than this (stripped) is never replaced
MIN_CONTENT_CHARS = 200     # a message text shorter than this is never rewritten
CHUNK_MODULUS = 13
CHUNK_MIN_LINES = 3
CHUNK_MAX_LINES = 40

# line-number prefixes: opencode's read tool ("00012| "), cat -n style ("    12|"), Claude Code's Read ("    12→")
_LINE_NUM_PREFIX_RE = re.compile(r"^\s*\d+[|→]")
_TEXT_PARTS = ("text", "input_text", "output_text")


@dataclass
class DedupResult:
    blocks_deduped: int = 0
    blocks_total: int = 0
    results_deduped: int = 0        # whole tool results replaced
    chars_before: int = 0
    chars_after: int = 0
    messages_changed: list = field(default_factory=list)

    @property
    def chars_saved(self) -> int:
        return self.chars_before - self.chars_after


def _strip_line_prefix(line: str) -> str:
    return _LINE_NUM_PREFIX_RE.sub("", line)


def content_defined_chunks(text: str, chunk_modulus: int = CHUNK_MODULUS) -> list[str]:
    """ContextPilot's chunking: cut after a line whose (prefix-stripped) md5 is 0 mod `chunk_modulus` once the chunk has
    CHUNK_MIN_LINES lines, or at CHUNK_MAX_LINES.  The same lines give the same cuts wherever they sit."""
    lines = text.split("\n")
    if len(lines) <= CHUNK_MIN_LINES:
        return [text]
    blocks, current = [], []
    for line in lines:
        current.append(line)
        h = int.from_bytes(hashlib.md5(_strip_line_prefix(line).strip().encode("utf-8", "replace")).digest()[:4],
                           "little")
        if (h % chunk_modulus == 0 and len(current) >= CHUNK_MIN_LINES) or len(current) >= CHUNK_MAX_LINES:
            blocks.append("\n".join(current))
            current = []
    if current:
        if blocks and len(current) < CHUNK_MIN_LINES:
            blocks[-1] += "\n" + "\n".join(current)
        else:
            blocks.append("\n".join(current))
    return blocks


def _hash_block(block: str) -> str:
    lines = block.strip().split("\n")
    return hashlib.sha256("\n".join(_strip_line_prefix(x) for x in lines).encode("utf-8", "replace")).hexdigest()[:20]


def _reference(block: str, origin: str) -> str:
    first = block.strip().split("\n")[0][:80]
    return f'[... "{first}" — identical to earlier {origin}, see above ...]'


class _Table:
    """The blocks seen so far in this request: hash -> (message index, label of where it was)."""

    def __init__(self, min_block_chars: int, chunk_modulus: int):
        self.seen: dict[str, tuple[int, str]] = {}
        self.whole: dict[str, tuple[int, str]] = {}
        self.min_block_chars = min_block_chars
        self.chunk_modulus = chunk_modulus

    def register(self, text: str, idx: int, label: str, result: DedupResult) -> None:
        for block in content_defined_chunks(text, self.chunk_modulus):
            if len(block.strip()) >= self.min_block_chars:
                result.blocks_total += 1
                self.seen.setdefault(_hash_block(block), (idx, label))

    def dedup(self, text: str, idx: int, label: str, result: DedupResult) -> str | None:
        """`text` with its blocks seen in an EARLIER message replaced; None when nothing was replaced (its blocks are
        registered either way)."""
        blocks = content_defined_chunks(text, self.chunk_modulus)
        if len(blocks) < 2:
            self.register(text, idx, label, result)
            return None
        out, changed = [], 0
        for block in blocks:
            if len(block.strip()) < self.min_block_chars:
                out.append(block)
                continue
            h = _hash_block(block)
            result.blocks_total += 1
            hit = self.seen.get(h)
            if hit is not None and hit[0] != idx:
                ref = _reference(block, hit[1])
                if len(ref) < len(block):
                    out.append(ref)
                    changed += 1
                    continue
            self.seen.setdefault(h, (idx, label))
            out.append(block)
        if not changed:
            return None
        result.blocks_deduped += changed
        return "\n".join(out)


def _json_text(content: str) -> tuple[str | None, str | None]:
    """A JSON-wrapped tool result ({"content": "...", ...}): its main text field and the key."""
    s = content.strip()
    if not s.startswith("{"):
        return None, None
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return None, None
    if isinstance(obj, dict):
        for key in ("content", "output", "result", "text", "stdout"):
            v = obj.get(key)
            if isinstance(v, str) and len(v) >= MIN_CONTENT_CHARS:
                return v, key
    return None, None


def _rewrap_json(original: str, key: str, text: str) -> str:
    try:
        obj = json.loads(original)
        obj[key] = text
        return json.dumps(obj, ensure_ascii=False)
    except (ValueError, TypeError):
        return original


def _texts(content):
    """(getter list): the text pieces of a message content (a string, or the template's list of parts)."""
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        return [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in _TEXT_PARTS
                and isinstance(p.get("text"), str)]
    return []


def _tool_labels(messages: list[dict]) -> dict[int, str]:
    """index of a tool message -> "<name> result", by position after the assistant turn whose calls it answers
    (the template pairs them by position too; the ids are gone by now)."""
    labels, calls, k = {}, [], 0
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant":
            calls = [((c.get("function") or {}).get("name") or "tool") for c in (m.get("tool_calls") or [])
                     if isinstance(c, dict)]
            k = 0
        elif role == "tool":
            name = calls[k] if k < len(calls) else (m.get("name") or "tool")
            labels[i] = f"{name} result"
            k += 1
    return labels


def dedup_messages(messages: list[dict], min_block_chars: int = MIN_BLOCK_CHARS,
                   min_content_chars: int = MIN_CONTENT_CHARS,
                   chunk_modulus: int = CHUNK_MODULUS) -> tuple[list[dict], DedupResult]:
    """The template messages with repeated blocks replaced, and what was done.  The input is not changed (the
    messages that change are copies).  System and assistant turns are sources only; user turns and tool results are
    rewritten."""
    result = DedupResult()
    if not isinstance(messages, list) or not messages:
        return messages, result
    table = _Table(min_block_chars, chunk_modulus)
    labels = _tool_labels(messages)
    out = list(messages)
    for idx, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role in ("system", "developer") or role == "assistant":
            label = "system prompt" if role != "assistant" else "assistant message"
            for t in _texts(content):
                table.register(t, idx, label, result)
            continue
        if role not in ("user", "tool"):
            continue
        label = labels.get(idx, "tool result") if role == "tool" else "user message"
        if role == "tool" and isinstance(content, str) and len(content) >= min_content_chars:
            h = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()[:20]
            hit = table.whole.get(h)
            if hit is not None:
                ref = f"[Duplicate — identical to the earlier {hit[1]} above. Refer to it there.]"
                result.results_deduped += 1
                result.chars_before += len(content)
                result.chars_after += len(ref)
                result.messages_changed.append(idx)
                out[idx] = dict(m, content=ref)
                continue
            table.whole[h] = (idx, label)
        if isinstance(content, str):
            if len(content) < min_content_chars:
                table.register(content, idx, label, result)
                continue
            target, key = _json_text(content) if role == "tool" else (None, None)
            new = table.dedup(target if target is not None else content, idx, label, result)
            if new is None:
                continue
            new_content = _rewrap_json(content, key, new) if target is not None else new
            result.chars_before += len(content)
            result.chars_after += len(new_content)
            result.messages_changed.append(idx)
            out[idx] = dict(m, content=new_content)
        elif isinstance(content, list):
            parts, changed = [], False
            for p in content:
                t = p.get("text") if isinstance(p, dict) and p.get("type") in _TEXT_PARTS else None
                if not isinstance(t, str):
                    parts.append(p)
                    continue
                if len(t) < min_content_chars:
                    table.register(t, idx, label, result)
                    parts.append(p)
                    continue
                new = table.dedup(t, idx, label, result)
                if new is None:
                    parts.append(p)
                    continue
                result.chars_before += len(t)
                result.chars_after += len(new)
                parts.append(dict(p, text=new))
                changed = True
            if changed:
                result.messages_changed.append(idx)
                out[idx] = dict(m, content=parts)
    return out, result


def dedup_setting(cfg: dict | None, env=None) -> bool:
    """STRATA_CONTEXT_DEDUP (1/on/true) or the config's "context_dedup": true.  The environment, when set, wins."""
    import os
    env = os.environ if env is None else env
    v = env.get("STRATA_CONTEXT_DEDUP")
    if v is not None and str(v).strip():
        return str(v).strip().lower() in ("1", "on", "true", "yes")
    v = (cfg or {}).get("context_dedup", False)
    if not isinstance(v, bool):
        raise ValueError(f"config \"context_dedup\" must be true or false, not {v!r}")
    return v
