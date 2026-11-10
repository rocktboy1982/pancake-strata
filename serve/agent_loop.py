"""serve/agent_loop.py - MOD_PLAN Phase 16 items A38, A41, A42, A43 (agent-loop helpers for opencode-style traffic).

Everything here works on the template messages of one request (serve/frontend.py: openai_to_messages /
anthropic_to_messages): the conversation the client resends, with the assistant's tool calls and the tool results.  The
server never sees file contents except through those tool results, so every check below is a statement about "what
the conversation shows", not about the disk.  Nothing here runs unless one of the opt-in switches below is set; the
default request path is untouched.

    A38  counters(messages)   exact per-request behaviour counters (edits without a test, reads without an edit, the
                              edit calls whose file was never read, ...), written into the turn log under "agent"
                              when STRATA_AGENT_COUNTERS=1.  Log only.
    A41  finish_note()        a short user-role note the server puts in front of the model when it is about to end its
                              turn after code edits with no build/test/run since (STRATA_TEST_BEFORE_DONE=1).  The
                              server (serve/server.py _run, next to the loop guard's keep-going) continues the answer
                              once; the client's conversation and the cached prefix are untouched.  LOSSY (the answer
                              grows by the model's reaction to the note).  Read-before-edit needs no server help:
                              opencode's edit and write tools already fail without a prior read (their tool text says
                              so), and `counters()["edits_unread"]` shows whether a client lets it through.
    A42  externalize()        tool results above N tokens are stored in a file and replaced, in the prompt only, by a
                              head and tail excerpt plus the path (STRATA_EXTERNALIZE_TOKENS=N).  LOSSY, deterministic
                              (same result -> same excerpt, same path), so the prompt prefix stays byte-stable.
    A43  edit_old_state()     an edit call whose oldString is in none of the file's earlier read results (nor in text
                              the conversation itself wrote into the file) is flagged: counted in the turn log and
                              /metrics (STRATA_EDIT_OLDSTRING_CHECK=log), plus a note after the failed result in
                              "note" mode.  Constrained decoding is not built (see pancake-docs/batch2).

Settings (environment only, all default off): STRATA_AGENT_COUNTERS=1, STRATA_TEST_BEFORE_DONE=1,
STRATA_EDIT_OLDSTRING_CHECK=off|log|note, STRATA_EXTERNALIZE_TOKENS=N (0: off), STRATA_EXTERNALIZE_DIR=path,
STRATA_EXTERNALIZE_HEAD_TOKENS=N (default 400), STRATA_EXTERNALIZE_TAIL_TOKENS=N (default 400),
STRATA_EXTERNALIZE_SKIP=tool,tool (default "read,read_file": a file read is already a file).
"""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import NamedTuple

from serve import loop_guard as LG

OLDSTRING_MODES = ("off", "log", "note")
CHARS_PER_TOKEN = 4                      # the estimate behind every "tokens" setting here (no tokenizer call)
DOC_SUFFIXES = (".md", ".markdown", ".txt", ".rst", ".adoc", ".org", ".csv", ".tsv", ".log")
READ_FILE_TOOLS = ("read", "read_file", "view", "view_file")
PATH_KEYS = ("filePath", "file_path", "path", "file", "filename", "target_file")
OLD_KEYS = ("oldString", "old_string", "old_str", "old_text", "oldText")
NEW_KEYS = ("newString", "new_string", "new_str", "new_text", "newText")
CONTENT_KEYS = ("content", "file_text", "text")
# what an edit tool says when the old text was not found (opencode, Claude Code, str_replace_editor) or not unique
FAIL_MARKERS = ("could not find oldstring", "found multiple matches", "string to replace not found",
                "no match found", "did not appear verbatim", "oldstring not found")
EXTERNALIZED = "[Output too long for the prompt."     # the first words of an excerpt (never externalized twice)


class Settings(NamedTuple):
    counters: bool = False               # A38: the "agent" counters in the turn log
    finish_nudge: bool = False           # A41
    oldstring: str = "off"               # A43: off | log | note
    externalize_tokens: int = 0          # A42: 0 = off
    externalize_dir: str = ""
    head_tokens: int = 400
    tail_tokens: int = 400
    skip_tools: tuple = ("read", "read_file")

    @property
    def active(self) -> bool:
        """Does any step run?  (False: the request path is exactly what it was.)"""
        return bool(self.counters or self.finish_nudge or self.oldstring != "off" or self.externalize_tokens)

    @property
    def needs_counters(self) -> bool:
        return bool(self.counters or self.finish_nudge or self.oldstring != "off")


def _flag(v) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "on")


def _count(v, name: str, default: int) -> int:
    if v is None or str(v).strip() == "":
        return default
    s = str(v).strip()
    if not s.isdigit():
        raise ValueError(f"{name} must be a whole number >= 0, not {v!r}")
    return int(s)


def settings(env=None) -> Settings:
    """The switches from the environment.  ValueError on a value it cannot read (the server stops at the start)."""
    env = os.environ if env is None else env
    mode = str(env.get("STRATA_EDIT_OLDSTRING_CHECK") or "off").strip().lower()
    mode = {"0": "off", "false": "off", "no": "off", "1": "log", "on": "log", "true": "log"}.get(mode, mode)
    if mode not in OLDSTRING_MODES:
        raise ValueError(f"STRATA_EDIT_OLDSTRING_CHECK must be one of {', '.join(OLDSTRING_MODES)}, not {mode!r}")
    tokens = _count(env.get("STRATA_EXTERNALIZE_TOKENS"), "STRATA_EXTERNALIZE_TOKENS", 0)
    head = _count(env.get("STRATA_EXTERNALIZE_HEAD_TOKENS"), "STRATA_EXTERNALIZE_HEAD_TOKENS", 400)
    tail = _count(env.get("STRATA_EXTERNALIZE_TAIL_TOKENS"), "STRATA_EXTERNALIZE_TAIL_TOKENS", 400)
    if tokens and head + tail >= tokens:
        raise ValueError(f"STRATA_EXTERNALIZE_HEAD_TOKENS + _TAIL_TOKENS ({head}+{tail}) must be below "
                         f"STRATA_EXTERNALIZE_TOKENS ({tokens}): the excerpt would be as long as the result")
    skip = env.get("STRATA_EXTERNALIZE_SKIP")
    skip_tools = tuple(t.strip().lower() for t in skip.split(",") if t.strip()) if skip is not None \
        else ("read", "read_file")
    directory = str(env.get("STRATA_EXTERNALIZE_DIR") or "").strip() or \
        os.path.join(tempfile.gettempdir(), "strata-externalized")
    return Settings(_flag(env.get("STRATA_AGENT_COUNTERS")), _flag(env.get("STRATA_TEST_BEFORE_DONE")), mode, tokens,
                    os.path.abspath(os.path.expanduser(directory)), head, tail, skip_tools)


# ---------------------------------------------------------------------------------------------------- reading calls
def _args(call: dict) -> dict:
    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    return LG._arguments(fn.get("arguments")) if isinstance(fn, dict) else {}


def _name(call: dict):
    fn = call.get("function") if isinstance(call.get("function"), dict) else call
    return fn.get("name") if isinstance(fn, dict) else None


def path_of(args: dict) -> str | None:
    for k in PATH_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            return os.path.normpath(v.strip())
    return None


def edit_pairs(args: dict) -> list[tuple[str, str]]:
    """The (old, new) texts of an edit call: its own, or those of a multi-edit's `edits` list.  [] for a write."""
    out = []
    items = args.get("edits") if isinstance(args.get("edits"), list) else [args]
    for it in items:
        if not isinstance(it, dict):
            continue
        old = next((it[k] for k in OLD_KEYS if isinstance(it.get(k), str)), None)
        new = next((it[k] for k in NEW_KEYS if isinstance(it.get(k), str)), "")
        if old is not None:
            out.append((old, new))
    return out


def written_text(args: dict) -> str | None:
    """The whole content a write call puts in a file (None for an edit)."""
    if edit_pairs(args):
        return None
    return next((args[k] for k in CONTENT_KEYS if isinstance(args.get(k), str)), None)


# A read tool numbers its lines: opencode "00001| text" (older) or "1: text" (newer), Claude Code "     1\ttext" or
# "     1→text".  The model's oldString has none of it.
_LINE_NO = re.compile(r"^[ \t]*\d{1,7}(?:\|(?: |$)|: |:$|\t|→)", re.M)
_WRAP = re.compile(r"^[ \t]*</?(?:file|path|type|content|system-reminder)>[^\n]*\n?", re.M)


def known_text(read_result: str) -> str:
    """A read result as the file's text: line numbers and the tool's wrapper lines taken off."""
    return _LINE_NO.sub("", _WRAP.sub("", read_result))


def _loose(s: str) -> str:
    return " ".join(s.split())


def edit_old_state(old: str, known: list[str]) -> str:
    """"exact": `old` is a substring of one of the file's known texts; "loose": only after white space is collapsed (an
    edit tool with fuzzy matching still finds it); "miss": in none of them; "unknown": no text of that file known."""
    if not known:
        return "unknown"
    if any(old in t for t in known):
        return "exact"
    lo = _loose(old)
    if lo and any(lo in _loose(t) for t in known):
        return "loose"
    return "miss"


def _failed(result: str) -> bool:
    low = result[:600].lower()
    return any(m in low for m in FAIL_MARKERS)


# ---------------------------------------------------------------------------------------------------- A38 counters
def counters(messages: list[dict], tools: dict | None = None) -> dict | None:
    """Exact behaviour counters for one request, from its messages.  The turn is what follows the user's last message
    (a bare "continue" does not start it over, as in loop_guard).  None when the request has no tool call in its turn.

      tool_calls, rounds, edits, checks, reads, other      calls of the turn by kind (loop_guard.call_kind)
      edits_since_check, reads_since_edit                  the streaks at the request's end (as item 33)
      max_edits_since_check, max_reads_since_edit          the longest streaks of the turn
      code_edits_since_check                               edit streak without doc files (.md .txt ...): A41's trigger
      files_edited                                         distinct files named by the turn's edit and write calls
      edits_unread                                         edit calls (with an oldString) on a file no read result
                                                           in the whole conversation shows and no write created
      edit_old_exact / _loose / _miss / _unknown           A43: the oldString against the file's known texts
      edit_failed                                          edit results that say the text was not found / not unique
      last_round_old_miss, last_round_failed               the same two for the model's last round only
      ends_unchecked                                       1: the request ends with tool results and
                                                           code_edits_since_check > 0 (the model is mid-loop)
    """
    out = dict.fromkeys((
        "tool_calls", "rounds", "edits", "checks", "reads", "other", "edits_since_check", "reads_since_edit",
        "max_edits_since_check", "max_reads_since_edit", "code_edits_since_check", "files_edited", "edits_unread",
        "edit_old_exact", "edit_old_loose", "edit_old_miss", "edit_old_unknown", "edit_failed",
        "last_round_old_miss", "last_round_failed", "ends_unchecked"), 0)
    read_files: set[str] = set()                   # the whole conversation
    known: dict[str, list[str]] = {}
    edited: set[str] = set()
    pending: list[dict] = []
    kinds: dict[str, str] = {}                     # call id -> kind of the pending calls
    ended_with_tool = False
    saw = False
    for m in messages:
        role = m.get("role")
        if role != "tool":
            ended_with_tool = False
        if role == "user" and not LG.go_on(m):
            out.update(dict.fromkeys(out, 0))      # a new turn: the per-turn counts start over (what was read stays)
            edited.clear()
            saw = False
        elif role == "assistant" and m.get("tool_calls"):
            pending = [c for c in m["tool_calls"] if isinstance(c, dict)]
            out["rounds"] += 1
            out["last_round_old_miss"] = out["last_round_failed"] = 0
            saw = True
            for c in pending:
                kind = LG.call_kind(_name(c), _args(c), tools)
                kinds[id(c)] = kind
                out["tool_calls"] += 1
                if kind == LG.EDIT:
                    out["edits"] += 1
                    out["edits_since_check"] += 1
                    out["reads_since_edit"] = 0
                    p = path_of(_args(c))
                    if p:
                        edited.add(p)
                    if not (p and p.lower().endswith(DOC_SUFFIXES)):
                        out["code_edits_since_check"] += 1
                elif kind == LG.CHECK:
                    out["checks"] += 1
                    out["edits_since_check"] = out["code_edits_since_check"] = 0
                elif kind == LG.READ:
                    out["reads"] += 1
                    out["reads_since_edit"] += 1
                else:
                    out["other"] += 1
                out["max_edits_since_check"] = max(out["max_edits_since_check"], out["edits_since_check"])
                out["max_reads_since_edit"] = max(out["max_reads_since_edit"], out["reads_since_edit"])
        elif role == "tool" and pending:
            ended_with_tool = True
            cid = m.get("_call_id")
            i = next((j for j, c in enumerate(pending) if cid is not None and c.get("_call_id") == cid), 0)
            call = pending.pop(i)
            result = LG._text(m.get("content"))
            name = str(_name(call) or "").strip().lower()
            args = _args(call)
            kind = kinds.pop(id(call), LG.OTHER)
            p = path_of(args)
            if kind == LG.READ and name in READ_FILE_TOOLS and p:
                read_files.add(p)
                known.setdefault(p, []).append(known_text(result))
            elif kind == LG.EDIT and p:
                pairs = edit_pairs(args)
                failed = _failed(result)
                if pairs:
                    if p not in read_files and p not in known:
                        out["edits_unread"] += 1
                    if failed:
                        out["edit_failed"] += 1
                        out["last_round_failed"] += 1
                    for old, new in pairs:
                        state = edit_old_state(old, known.get(p, []))
                        out["edit_old_" + state] += 1
                        if state == "miss":
                            out["last_round_old_miss"] += 1
                    if not failed and p in known:      # (a file never read stays "unknown": its other text is not ours)
                        for old, new in pairs:
                            known[p].append(new)
                else:
                    whole = written_text(args)
                    if whole is not None and not failed:
                        known[p] = [whole]         # (the write replaced the file: what was read before is gone)
    out["files_edited"] = len(edited)
    if not saw:
        return None
    out["ends_unchecked"] = int(ended_with_tool and out["code_edits_since_check"] > 0)
    return out


# ---------------------------------------------------------------------------------------------------- A41, A43 notes
FINISH_NOTE = ("[The server, not the user: you changed code in this turn and have not run a build, a test or the "
               "program since. Before you finish, run the check that fits (the tests for what you changed, or the "
               "build) and read what it prints; if none applies, say in one sentence why.]")
FINISH_STEER = "I changed code and have not checked it yet. The check that fits is"


def oldstring_note(c: dict) -> str:
    """A43 "note" mode: after the failed edit result of the model's last round."""
    return ("[The server, not the user: the oldString of your last edit is not in any text of that file that you "
            "have read. Read the lines you want to change again and copy oldString from the output exactly, "
            "without line numbers.]")


def wants_oldstring_note(c: dict | None, mode: str) -> bool:
    return bool(c) and mode == "note" and c["last_round_old_miss"] > 0 and c["last_round_failed"] > 0


# ---------------------------------------------------------------------------------------------------- A42 externalize
class External(NamedTuple):
    messages: list
    count: int                           # results replaced in this request
    new_files: int                       # files written by this request (the others already existed)
    chars_saved: int


def est_tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def excerpt(text: str, path: str, head_tokens: int, tail_tokens: int) -> str:
    """The head and the tail of `text` (cut at line ends where one is near) around a line that says where the whole
    is.  Deterministic: the same text always gives the same string."""
    h, t = head_tokens * CHARS_PER_TOKEN, tail_tokens * CHARS_PER_TOKEN
    head, tail = text[:h], text[len(text) - t:] if t else ""
    cut = head.rfind("\n")
    if cut >= h // 2:
        head = head[:cut + 1]
    cut = tail.find("\n")
    if 0 <= cut < len(tail) - t // 2:
        tail = tail[cut + 1:]
    omitted = len(text) - len(head) - len(tail)
    mark = (f"{EXTERNALIZED} {omitted:,} of {len(text):,} characters (~{est_tokens(text):,} tokens) are left out "
            f"here. The whole output is saved in the file {path} - read it in ranges (offset and limit) or search "
            "it with grep instead of running the command again.]")
    return head + ("" if head.endswith("\n") or not head else "\n") + mark + "\n" + tail


def externalize(messages: list[dict], cfg: Settings) -> External:
    """A42 (LOSSY, opt-in): every tool result of more than cfg.externalize_tokens (estimated: 4 characters a token) is
    written to cfg.externalize_dir/<sha256 prefix>-<tool>.txt and replaced, in the prompt only, by excerpt().  The
    client's own conversation is not touched, and as the excerpt and the path depend on the result alone, every later
    request of the conversation carries the same bytes: the cached prefix stays valid.  Results of the tools in
    cfg.skip_tools (a file read), results with pictures and excerpts already made are left as they are.  A file
    that cannot be written leaves its result as it is."""
    limit = cfg.externalize_tokens
    if not limit or not messages:
        return External(messages, 0, 0, 0)
    out, count, new_files, saved = list(messages), 0, 0, 0
    names: dict[str, str] = {}                       # call id -> tool name
    pending: list[dict] = []
    for i, m in enumerate(messages):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            pending = [c for c in m["tool_calls"] if isinstance(c, dict)]
            for c in pending:
                if c.get("_call_id") is not None:
                    names[c["_call_id"]] = str(_name(c) or "")
        elif m.get("role") == "tool":
            cid = m.get("_call_id")
            tool = names.get(cid) if cid is not None else None
            if tool is None and pending:
                tool = str(_name(pending[0]) or "")
            if pending:
                pending = [c for c in pending if c.get("_call_id") != cid] if cid is not None else pending[1:]
            content = m.get("content")
            if not isinstance(content, str) or est_tokens(content) <= limit or content.startswith(EXTERNALIZED):
                continue
            if (tool or "").strip().lower() in cfg.skip_tools:
                continue
            digest = hashlib.sha256(content.encode("utf-8", "replace")).hexdigest()[:20]
            safe = re.sub(r"[^A-Za-z0-9_.-]", "_", tool or "tool")[:40]
            path = os.path.join(cfg.externalize_dir, f"{digest}-{safe}.txt")
            try:
                if not (os.path.isfile(path) and os.path.getsize(path) == len(content.encode("utf-8", "replace"))):
                    os.makedirs(cfg.externalize_dir, mode=0o700, exist_ok=True)
                    tmp = f"{path}.{os.getpid()}.tmp"
                    Path(tmp).write_bytes(content.encode("utf-8", "replace"))
                    os.replace(tmp, path)
                    new_files += 1
            except OSError:
                continue
            short = excerpt(content, path, cfg.head_tokens, cfg.tail_tokens)
            if len(short) >= len(content):
                continue
            nm = dict(m)
            nm["content"] = short
            out[i] = nm
            count += 1
            saved += len(content) - len(short)
    return External(out, count, new_files, saved)
