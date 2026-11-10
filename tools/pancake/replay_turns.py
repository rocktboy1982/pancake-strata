#!/usr/bin/env python3
"""replay_turns.py - replays a recorded agent session as real HTTP chat traffic, one fresh server per arm of an
abtest-style spec, so that the conversation-cache arms can be judged (bench/TEST_PLAN.md 4.1, "Agent replay for the
cache arms").  abtest.py's cases are independent one-shot requests: they never park a long conversation and never
evict it with a short side request.  This replays the traffic shape that did (2026-11-09 16:13-16:43: 7 cold
54-62K re-reads of one agent conversation while short decodes and 3-4K prefill probes ran in between).

    python tools/pancake/replay_turns.py SPEC.json --out DIR [--session WINDOW|sN|REQUEST_ID] [--scale 1.0]
           [--reps N] [--arms base,cache2048-min8192] [--synthetic] [--dry-run]

Input: the per-turn log (`pancake-turns.jsonl`, item 73; numbers only, no text): per request its start time,
prompt_total, reused, output_tokens, max_new, reasoning_tokens (thinking on/off), prompt_ms, duration_s.
  --session "2026-11-09T16:13/16:43" (a time window, local time), "s3" (the 4th session when the log is split at
  gaps > --session-gap s; --list-sessions prints them) or a request id (the session that holds it).  Default: the
  spec's "replay": {"session": ...}.  --synthetic: a built-in session with the 4.0.1 shape instead (one conversation
  growing from 40K to 62K tokens, 7 warm turns, then 9 turns each after a group of side requests, then 2 warm turns).

How the traffic is rebuilt (the text is synthetic and deterministic; only the sizes, the order and the reuse
structure come from the log):
  - conversations: a request of >= --conv-min tokens extends the open conversation whose last prompt it can grow
    from (last <= prompt <= last + its output + --max-growth), the largest such; otherwise it starts a new one.  A
    re-read with reused == 0 is therefore still the SAME conversation (it is the cold re-read the cache arms should
    turn warm), not a new one.
  - every agent turn sends the whole conversation: turn N+1 = turn N's messages + the assistant reply the server
    actually returned for turn N (reasoning_content + content, as an agent client sends it back; the template keeps
    the thinking in the history) + one user message "<tool_response>...</tool_response>" whose size makes the prompt
    exactly the recorded prompt_total.  So the engine sees the same prefix growth, the same reuse and the same
    snapshot sizes (the 1280 MiB budget was missed by 0.5-4.8% on 11-09: sizes are matched to the token).
  - side requests (< --conv-min tokens): separate one-message conversations; one with reused > 0 repeats the text
    of the latest side request of the same size (thinking-off prompts count 2 tokens less), else new text.
  - text: lines of the repo's own C++ sources (src/), a fixed start per conversation; control-token-like lines are
    left out.  Token counts come from the server's own tokenizer and chat template, imported locally
    (tools/strata_tokenizer.py + serve/frontend.ChatTemplate, as the server's encode_prompt; "effort_position": "end"
    rendering for thinking-off requests); each response's usage.prompt_tokens corrects any offset for the next
    request.  Without the tokenizer: ~3.2 characters per token, corrected the same way.
  - max_tokens = the recorded output_tokens (--max-new recorded: the recorded max_new; agent turns had 32768),
    thinking on/off as recorded, greedy unless the spec sets "temperature"; think time between requests is the
    recorded gap capped at --max-gap s.

  - --shared-system N (opt-in; default each conversation its own system text, as above): every conversation, the
    agent's and the side requests, starts with the same ~N-token system prompt, so the parked conversations share a
    prefix the way a main agent and its sub-agents do (MOD_PLAN A15's park TTL classifies a short entry that shares
    its prefix with a longer one as a sub-agent branch).  The sizes still match the recorded prompts.

Per request: prompt tokens, tokens reused (the engine's "prompt N tokens = R reused + ..." line), prompt-read time,
time to first token (client, streamed), decode tok/s, wall time, and the engine's conversation-cache lines in that
request's window (parked / skip parking (why) / restored / evicted).  Per arm run: cold re-reads (a non-first agent
turn > 8K tokens with reuse < 50%) and their seconds, total prompt time, total turn time, peak RSS of the server tree,
lowest MemAvailable.  Verdict vs the first arm on total prompt time: paired 95% interval over --reps, else deltas.

--dry-run builds the whole request plan (and checks the sizing with placeholder replies), prints it with the expected
duration, and starts nothing.  The real run stops any running Strata server first (abtest.stop_servers): engine on
8080 stopped, nothing else on the GPU.
"""
import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import abtest  # noqa: E402

ROOT = abtest.ROOT
SHARED_SYSTEM = 0          # --shared-system N: every conversation starts with the same ~N-token system prompt (0: off)
COLD_MIN = 8192            # a cold re-read: a non-first agent turn above this many tokens ...
COLD_REUSE = 0.5           # ... with less than this share of its prompt reused
PLACEHOLDER = "(continued)"   # an assistant reply cut inside its thinking has no content; an empty one is dropped by
#                               the server's template (#843), which would break the prefix the engine holds
TOL = 2                    # tokens: how close a synthesized prompt must come to its recorded size


# ------------------------------------------------------------------------------------------------ the turn log
def load_turns(path):
    rows = []
    for line in open(path, errors="replace"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("prompt_total") is not None and r.get("time") is not None:
            rows.append(r)
    rows.sort(key=lambda r: r["time"])
    return rows


def split_sessions(rows, gap_s):
    out, cur = [], []
    for r in rows:
        if cur and r["time"] - (cur[-1]["time"] + (cur[-1].get("duration_s") or 0)) > gap_s:
            out.append(cur)
            cur = []
        cur.append(r)
    return out + ([cur] if cur else [])


def hm(t):
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


WINDOW = re.compile(r"(\d{4}-\d{2}-\d{2})[T ](\d{1,2}:\d{2})(?::\d{2})?\s*(?:/|-|\.\.)\s*(?:(\d{4}-\d{2}-\d{2})[T ])?"
                    r"(\d{1,2}:\d{2})(?::\d{2})?")


def select_session(rows, sel, gap_s):
    m = WINDOW.fullmatch(sel.strip())
    if m:
        t0 = datetime.strptime(f"{m[1]} {m[2]}", "%Y-%m-%d %H:%M").timestamp()
        t1 = datetime.strptime(f"{m[3] or m[1]} {m[4]}", "%Y-%m-%d %H:%M").timestamp() + 60
        got = [r for r in rows if t0 <= r["time"] < t1]
        if not got:
            raise SystemExit(f"no request in {sel}")
        return got, f"{hm(got[0]['time'])} - {hm(got[-1]['time'])} (window {sel})"
    ss = split_sessions(rows, gap_s)
    if re.fullmatch(r"s?\d+", sel):
        k = int(sel.lstrip("s"))
        if k >= len(ss):
            raise SystemExit(f"session {sel}: the log has {len(ss)} sessions (--list-sessions)")
        return ss[k], f"s{k}: {hm(ss[k][0]['time'])} - {hm(ss[k][-1]['time'])}"
    for k, s in enumerate(ss):
        if any(str(r.get("id")) == sel for r in s):
            return s, f"s{k} (holds request {sel}): {hm(s[0]['time'])} - {hm(s[-1]['time'])}"
    raise SystemExit(f"--session {sel!r}: not a window (2026-11-09T16:13/16:43), a session (s3) or a request id")


def thinking_of(r):
    return bool(r.get("effort")) or (r.get("reasoning_tokens") or 0) > 0


def conv_name(k):
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[k] if k < 26 else f"C{k}"


REC_KEYS = ("id", "time", "duration_s", "finish", "prompt_total", "reused", "prompt_read", "prompt_ms", "output_tokens",
            "first_engine_token_s", "decode_tok_s", "max_new", "reasoning_tokens", "effort")


def plan_from_rows(rows, conv_min, max_growth, max_gap, max_new_mode, think_off_extra=2):
    """The recorded requests -> the replay plan (see the module doc for the rules)."""
    plan, convs, side_last, nside, prev_end = [], [], {}, 0, None
    for r in rows:
        tot, reused, think = int(r["prompt_total"]), int(r.get("reused") or 0), thinking_of(r)
        out = max(1, int(r.get("output_tokens") or 0))
        gap = 0.0 if prev_end is None else max(0.0, r["time"] - prev_end)
        prev_end = r["time"] + (r.get("duration_s") or 0)
        cap = int(r.get("max_new") or out) if max_new_mode == "recorded" else out
        e = {"i": len(plan), "target": tot, "max_new": cap, "thinking": think, "gap_s": round(min(gap, max_gap), 2),
             "rec": {k: r.get(k) for k in REC_KEYS}}
        if tot >= conv_min:
            cands = [c for c in convs if c["last"] <= tot <= c["last"] + c["out"] + max_growth]
            c = max(cands, key=lambda c: c["last"]) if cands else None
            first = c is None
            if first:
                c = {"name": conv_name(len(convs)), "last": tot, "out": out, "turns": 0}
                convs.append(c)
            e.update(kind="agent", conv=c["name"], turn=c["turns"], first=first)
            c.update(turns=c["turns"] + 1, last=tot, out=out)
        else:
            key = tot - (0 if think else think_off_extra)
            repeat = reused > 0 and key in side_last
            if not repeat:
                side_last[key] = f"s{nside}"
                nside += 1
            e.update(kind="side", conv=side_last[key], turn=0, first=not repeat, repeat=repeat,
                     sub="probe" if (r.get("max_new") or out) <= 2 else "decode")
        plan.append(e)
    return plan


def plan_synthetic(gap_s=1.0):
    """The 4.0.1 shape without the log: conversation S grows 40,000 -> 62,000 tokens in 18 turns (thinking on, 320-760
    output tokens); turns 7-15 each follow a group of side requests patterned on the 11-09 traffic, turns 0-6 and
    16-17 run back to back (warm on any arm)."""
    plan, nside = [], 0

    def add(**e):
        e.setdefault("gap_s", gap_s)
        e.setdefault("rec", None)
        plan.append(dict(i=len(plan), **e))

    def side(target, max_new, think, sub, key=None):
        nonlocal nside
        if key is None:
            key, nside = f"s{nside}", nside + 1
        add(kind="side", conv=key, turn=0, first=True, repeat=False, sub=sub, target=target, max_new=max_new,
            thinking=think)
        return key

    def again(key, target, max_new, think, sub):
        add(kind="side", conv=key, turn=0, first=False, repeat=True, sub=sub, target=target, max_new=max_new,
            thinking=think)

    def group(g):
        if g == 1:      # 3 x a 69-token decode, max_new 400, the same text (1 cold, 2 warm)
            k = side(69, 400, True, "decode")
            again(k, 69, 400, True, "decode"); again(k, 69, 400, True, "decode")
        elif g == 2:    # a 3,058-token prefill probe (max_new 1) and 5 repeats of it
            k = side(3058, 1, True, "probe")
            for _ in range(5):
                again(k, 3058, 1, True, "probe")
        elif g == 3:    # 3 different ~4.1K prefill probes
            for t in (4084, 4088, 4093):
                side(t, 1, True, "probe")
        elif g == 4:    # 2 different 75-token decodes, max_new 1000
            side(75, 1000, True, "decode"); side(75, 1000, True, "decode")
        else:           # a 73-token thinking-off decode, then the same messages with thinking on (71 tokens)
            k = side(73, 160, False, "decode")
            again(k, 71, 400, True, "decode")

    totals = [40000 + round(i * 22000 / 17) for i in range(18)]
    outs = [480, 320, 760, 400, 560]
    groups = {7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 12: 1, 13: 2, 14: 3, 15: 4}
    for n, tot in enumerate(totals):
        if n in groups:
            group(groups[n])
        add(kind="agent", conv="S", turn=n, first=n == 0, target=tot, max_new=outs[n % len(outs)], thinking=True)
    return plan


# ------------------------------------------------------------------------------------------------ text and tokens
class Corpus:
    """The repo's C++ sources as one deterministic text; take() cuts a slice of about N tokens."""

    def __init__(self, root, max_chars=6_000_000):
        parts, n = [], 0
        files = sorted(p for ext in ("cpp", "h", "cu", "cuh", "hpp") for p in (Path(root) / "src").rglob("*." + ext))
        for f in files:
            try:
                t = f.read_text(errors="replace").replace("\r", "")
            except OSError:
                continue
            # a control token's or a thinking marker's text would be tokenized differently by the server (#537)
            t = "".join(l for l in t.splitlines(True) if "<|" not in l and "think>" not in l and "<tool_" not in l)
            parts.append(t if t.endswith("\n") else t + "\n")
            n += len(t)
            if n >= max_chars:
                break
        text = "".join(parts)
        if len(text) < 200_000:          # no sources: seeded pseudo-words
            words, h = [], b"replay"
            while sum(map(len, words)) < 1_000_000:
                h = hashlib.sha256(h).digest()
                words += ["w%x%s" % (b, " " if b % 9 else "\n") for b in h]
            text = "".join(words)
        self.text, self.n, self.cpt = text, len(text), 3.2

    def start(self, key):
        return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) % self.n

    def slice(self, pos, nchars):
        pos %= self.n
        out = self.text[pos:pos + nchars]
        while len(out) < nchars:
            out += self.text[:nchars - len(out)]
        return out

    def take(self, pos, ntok, count):
        """About ntok tokens of text from pos (count: text -> tokens); returns (text, pos after it)."""
        if ntok <= 0:
            return "", pos
        est, text, m = int(ntok * self.cpt) + 1, "", 0
        for _ in range(6):
            text = self.slice(pos, est)
            m = count(text)
            if abs(m - ntok) <= 1:
                break
            step = int((ntok - m) * self.cpt) or (1 if ntok > m else -1)
            est = max(1, est + step)
        if m > 50:
            self.cpt = 0.7 * self.cpt + 0.3 * len(text) / m
        return text, pos + len(text)


class Counter:
    """Prompt token counts as the server makes them: its tokenizer and chat template imported locally, the prompt
    rendered as serve/server.py render_prompt does ("effort_position": "end": a thinking-off request is the default
    history plus the template's empty thinking block).  Falls back to characters / 3.2 when they cannot be loaded."""

    def __init__(self, tokdir, effort_end=True):
        self.effort_end, self.exact, self.why = effort_end, False, ""
        try:
            for p in (str(ROOT), str(ROOT / "tools")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            import strata_tokenizer as ST
            from serve.frontend import ChatTemplate
            tp = Path(tokdir)
            vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
            tokens = [None] * len(vocab)
            for t, i in vocab.items():
                tokens[i] = t
            merges = (tp / "merges.txt").read_text(encoding="utf-8").split("\n")
            types = json.loads((tp / "token_type.json").read_text())
            try:
                tcfg = json.loads((tp / "tokenizer.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                tcfg = {}
            self.tok = ST.Tokenizer(tokens, merges, types, tcfg.get("pre") or "qwen35", tcfg.get("special_ids") or None)
            self.pe = ST.PromptEncoder(self.tok, keep=16)
            self.T = ChatTemplate(tp / "chat_template.jinja")
            self.exact, self.why = True, f"local tokenizer + chat template ({tp})"
        except Exception as e:  # noqa: BLE001 - any failure: the estimate, said once
            self.why = f"estimate, ~3.2 characters per token ({type(e).__name__}: {e})"

    def block(self, text):
        return len(self.tok.encode(text)) if self.exact else math.ceil(len(text) / 3.2)

    def render(self, msgs, thinking):
        if thinking:
            return self.T.render(msgs, add_generation_prompt=True, enable_thinking=True)
        if not self.effort_end:
            return self.T.render(msgs, add_generation_prompt=True, enable_thinking=False)
        history = self.T.render(msgs, add_generation_prompt=False)
        full = self.T.render(msgs, add_generation_prompt=True, enable_thinking=False)
        mine = self.T.render(msgs, add_generation_prompt=False, enable_thinking=False)
        return history + full[len(mine):] if full.startswith(mine) else full

    def prompt(self, msgs, thinking):
        if self.exact:
            return len(self.pe.encode(self.render(msgs, thinking)))
        chars = sum(len(m.get("content") or "") + len(m.get("reasoning_content") or "") for m in msgs)
        return 45 + 6 * len(msgs) + math.ceil(chars / 3.2) + (3 if thinking else 5)


class Conversations:
    """The messages of every replayed conversation, extended turn by turn and sized to the recorded prompts."""

    def __init__(self, counter, corpus):
        self.C, self.corpus = counter, corpus
        corpus.cpt = 3.2                       # the same slices in every run, given the same replies
        self.msgs, self.pos, self.reply = {}, {}, {}
        self.offset = {True: 0, False: 0}      # the server's count minus the local one (should stay 0)
        self.offset_seen = set()
        self.shared = None                     # --shared-system: the one system text every conversation starts with
        if SHARED_SYSTEM > 0:
            self.shared, _ = corpus.take(corpus.start("shared-system"), SHARED_SYSTEM, self.C.block)

    def fit(self, prefix, wrap, target, thinking, key):
        goal = target - self.offset[thinking]
        n = self.C.prompt(prefix + [wrap("")], thinking)
        pos = self.pos.setdefault(key, self.corpus.start(key))
        text, end = self.corpus.take(pos, goal - n, self.C.block)
        msgs = prefix + [wrap(text)]
        for _ in range(8):
            msgs = prefix + [wrap(text)]
            n = self.C.prompt(msgs, thinking)
            d = goal - n
            if abs(d) <= TOL or (d < 0 and not text):
                break
            if d > 0:
                extra, end = self.corpus.take(end, d, self.C.block)
                text += extra
            else:
                cut = min(len(text), max(1, int(-d * self.corpus.cpt * 0.9)))
                text, end = text[:-cut], end - cut
        self.pos[key] = end
        return msgs, n + self.offset[thinking]

    def build(self, e):
        """The messages of plan entry e; returns (messages, expected server prompt tokens)."""
        k, think, target = e["conv"], e["thinking"], e["target"]
        if e["kind"] == "side":
            if k in self.msgs:
                msgs = self.msgs[k]
                return msgs, self.C.prompt(msgs, think) + self.offset[think]
            lead = [{"role": "system", "content": self.shared}] if self.shared else []
            msgs, n = self.fit(lead, lambda t: {"role": "user", "content": f"{k}: {t}"}, target, think, k)
            self.msgs[k] = msgs
            return msgs, n
        if e["first"] or k not in self.msgs:
            system = {"role": "system", "content": self.shared or
                      f"You are a coding agent working in a C++ repository (replayed "
                      f"conversation {k}). Read what the tools return and answer briefly."}
            msgs, n = self.fit([system], lambda t: {"role": "user", "content": f"Task {k}: review these sources and "
                                                    f"list what to change.\n\n{t}"}, target, think, k)
        else:
            reasoning, content = self.reply.get(k, ("", ""))
            a = {"role": "assistant", "content": content if content.strip() else PLACEHOLDER}
            if reasoning.strip():
                a["reasoning_content"] = reasoning
            msgs, n = self.fit(self.msgs[k] + [a],
                               lambda t: {"role": "user", "content": f"<tool_response>\n{t}\n</tool_response>"},
                               target, think, k)
        self.msgs[k] = msgs
        return msgs, n

    def answered(self, e, reasoning, content, server_prompt, local_prompt):
        if e["kind"] == "agent":
            self.reply[e["conv"]] = (reasoning or "", content or "")
        if server_prompt is not None and local_prompt is not None:
            d = int(server_prompt) - int(local_prompt - self.offset[e["thinking"]])
            if d != self.offset[e["thinking"]] and e["thinking"] not in self.offset_seen:
                print(f"  token count: the server counts {d:+d} vs the local count (thinking "
                      f"{'on' if e['thinking'] else 'off'}); corrected from here on", flush=True)
                self.offset_seen.add(e["thinking"])
            self.offset[e["thinking"]] = d

    def placeholder_reply(self, e):
        """Dry run: a reply of the planned size (its reasoning) in place of the model's."""
        if e["kind"] == "agent":
            key = "reply-" + e["conv"]
            pos = self.pos.setdefault(key, self.corpus.start(key))
            text, self.pos[key] = self.corpus.take(pos, max(1, e["max_new"] - 4), self.C.block)
            self.reply[e["conv"]] = (text, PLACEHOLDER) if e["thinking"] else ("", text)


# ------------------------------------------------------------------------------------------------ estimates
def est_prompt_s(read):
    """Prompt time model from the 11-09 log: cold 54-61K at 0.39-0.42 ms/token, ~4K at ~0.64 ms/token, ~0.2 s floor."""
    return 0.2 + max(0, read) * (0.41 if read >= 8192 else 0.64) / 1000


def est_request_s(e, read):
    return est_prompt_s(read) + e["max_new"] / (90.0 if e["thinking"] else 110.0)


def growth_reads(plan):
    """Per entry: the tokens a warm engine would read (an agent turn: the growth over its previous turn plus the
    template's few; a repeated side text: ~5; anything new: all of it)."""
    last, reads = {}, []
    for e in plan:
        k = e["conv"]
        if e["kind"] == "agent" and k in last:
            reads.append(max(5, e["target"] - last[k]))
        elif e["kind"] == "side" and e.get("repeat"):
            reads.append(5)
        else:
            reads.append(e["target"])
        last[k] = e["target"] + e["max_new"]
    return reads


def cold_reread(kind, first, prompt, reused):
    return kind == "agent" and not first and (prompt or 0) > COLD_MIN and (reused or 0) < COLD_REUSE * (prompt or 0)


# ------------------------------------------------------------------------------------------------ server I/O
def chat(url, msgs, max_tokens, thinking, sampling, timeout=1800):
    body = {"model": "x", "messages": msgs, "max_tokens": int(max_tokens), "stream": True, **sampling,
            "chat_template_kwargs": {"enable_thinking": bool(thinking)}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    first, content, reasoning, usage, timings, finish = None, [], [], None, None, None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ch = json.loads(data)
            except ValueError:
                continue
            if ch.get("error"):
                raise RuntimeError(f"server error: {ch['error']}")
            for c in ch.get("choices") or []:
                d = c.get("delta") or {}
                if (d.get("content") or d.get("reasoning_content")) and first is None:
                    first = time.time() - t0
                content.append(d.get("content") or "")
                reasoning.append(d.get("reasoning_content") or "")
                finish = c.get("finish_reason") or finish
            usage = ch.get("usage") or usage
            timings = ch.get("timings") or timings
    wall = time.time() - t0
    return {"content": "".join(content), "reasoning": "".join(reasoning), "usage": usage or {}, "timings": timings,
            "finish": finish, "ttft_s": round(first if first is not None else wall, 3), "wall_s": round(wall, 3)}


PROMPT_LINE = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+)(?: of \d+)? read in (\d+) ms "
                         r"\(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\)")
PARKED = re.compile(r"conversation cache: (parked|skipped) (\d+) tokens.*?evictions=(\d+)(?:.*?snapshot_bytes=(\d+))?")
RESTORED = re.compile(r"conversation cache: restored (\d+) tokens \(([^)]*)\) in ([\d.]+) ms")
SKIP = re.compile(r"conversation cache: skip parking \((.*)\)\s*$")
EVICTED = re.compile(r"conversation cache: evicted (\d+) parked")
BUDGET = re.compile(r"conversation cache (?:sized from the free RAM: (\d+) MiB for up to (\d+)|off)")


def parse_engine(text):
    """The engine lines of one request's window: its prompt line(s) and the conversation-cache events."""
    out = {"prompt_lines": [], "events": [], "evictions": None}
    for line in text.splitlines():
        m = PROMPT_LINE.search(line)
        if m:
            out["prompt_lines"].append({"prompt": int(m[1]), "reused": int(m[2]), "read": int(m[3]),
                                        "read_ms": int(m[4]), "gen": int(m[6]), "gen_ms": int(m[7]),
                                        "decode_tps": float(m[8])})
            continue
        if "conversation cache" not in line and "conversation disk" not in line:
            continue
        if m := PARKED.search(line):
            out["events"].append(f"{m[1]} {m[2]}" + (f" ({int(m[4]) >> 20} MiB)" if m[4] else ""))
            out["evictions"] = int(m[3])
        elif m := RESTORED.search(line):
            out["events"].append(f"restored {m[1]} ({m[2]}, {float(m[3]):.0f} ms)")
        elif m := SKIP.search(line):
            out["events"].append(f"skip ({m[1]})")
        elif m := EVICTED.search(line):
            out["events"].append(f"evicted {m[1]}")
        elif "dropped" in line and "superseded" in line:
            out["events"].append("superseded")
        elif "conversation disk" in line:
            out["events"].append("disk: " + line.split("conversation disk:", 1)[-1].strip()[:80])
    return out


def read_from(path, offset):
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def size_of(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def turn_rows(path):
    try:
        return [json.loads(l) for l in open(path, errors="replace") if l.strip()]
    except (OSError, ValueError):
        return []


def meminfo_avail():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


class MemSampler(threading.Thread):
    """Every second: the resident memory of the server's process tree (peak) and MemAvailable (lowest)."""

    def __init__(self, root_pid):
        super().__init__(daemon=True)
        self.root, self.stop_ev, self.peak_rss, self.min_avail = root_pid, threading.Event(), 0, None
        self.page = os.sysconf("SC_PAGE_SIZE")

    def run(self):
        while not self.stop_ev.is_set():
            rss = 0
            for pid in abtest.tree(abtest.proc_table(), [self.root]):
                try:
                    rss += int(open(f"/proc/{pid}/statm").read().split()[1]) * self.page
                except (OSError, ValueError, IndexError):
                    pass
            self.peak_rss = max(self.peak_rss, rss)
            av = meminfo_avail()
            if av is not None:
                self.min_avail = av if self.min_avail is None else min(self.min_avail, av)
            self.stop_ev.wait(1.0)

    def result(self):
        self.stop_ev.set()
        self.join(timeout=5)
        return round(self.peak_rss / 2**30, 2), (round(self.min_avail / 2**30, 2) if self.min_avail is not None else None)


# ------------------------------------------------------------------------------------------------ one arm
def run_arm(arm, rep, cfg_path, plan, counter, corpus, a, out_dir, sampling):
    tag = f"{arm}-r{rep}"
    eng_log, tlog, srv_out = out_dir / f"{tag}.engine.log", out_dir / f"{tag}.turns.jsonl", out_dir / f"{tag}.server.out"
    cfg = json.load(open(cfg_path))
    cfg["log"], cfg["turn_log"] = str(eng_log), str(tlog)        # this run's own logs, not the daily ones
    json.dump(cfg, open(cfg_path, "w"), indent=1)
    os.environ["STRATA_TURN_LOG"] = str(tlog)                     # it wins over the config's turn_log
    for p in (eng_log, tlog):
        if p.exists():
            p.unlink()
    srv, err = abtest.start_server(cfg_path, str(srv_out), a.port)
    if err:
        abtest.stop_servers()
        return None, err
    url = f"http://127.0.0.1:{a.port}"

    def budget_line():          # the engine may load on the first request: read after it, and again at the end
        m = BUDGET.search(read_from(eng_log, 0))
        return f"{m[1]} MiB / {m[2]} slots (auto)" if m and m[1] else "off" if m else "set by flag (no auto line)"

    mon, mem = abtest.NoiseMonitor([srv.pid]), MemSampler(srv.pid)
    mon.start()
    mem.start()
    convs, rows, t_start = Conversations(counter, corpus), [], time.time()
    try:
        for e in plan:
            time.sleep(e["gap_s"])
            msgs, local = convs.build(e)
            off, n_turns = size_of(eng_log), len(turn_rows(tlog))
            row = {"arm": arm, "rep": rep, **{k: e[k] for k in ("i", "kind", "conv", "turn", "first", "target",
                                                                "max_new", "thinking")}, "local_prompt": local}
            try:
                res = chat(url, msgs, e["max_new"], e["thinking"], sampling, a.timeout)
            except (urllib.error.URLError, OSError, RuntimeError, ValueError) as ex:
                row["error"] = f"{type(ex).__name__}: {ex}"
                rows.append(row)
                print(f"  [{e['i'] + 1}/{len(plan)}] {e['kind']} {e['conv']}: ERROR {row['error']}", flush=True)
                if srv.poll() is not None:
                    print("  the server died: this arm run stops here", flush=True)
                    break
                continue
            t_wait = time.time()
            while len(turn_rows(tlog)) <= n_turns and time.time() - t_wait < 10:
                time.sleep(0.1)
            tr = (turn_rows(tlog)[n_turns:] or [{}])[-1]
            eng = parse_engine(read_from(eng_log, off))
            pl = eng["prompt_lines"][0] if eng["prompt_lines"] else {}
            u = res["usage"]
            prompt = u.get("prompt_tokens") or tr.get("prompt_total") or pl.get("prompt")
            reused = pl.get("reused", tr.get("reused", (u.get("prompt_tokens_details") or {}).get("cached_tokens")))
            prompt_ms = tr.get("prompt_ms")
            if prompt_ms is None and eng["prompt_lines"]:
                prompt_ms = sum(p["read_ms"] for p in eng["prompt_lines"])
            convs.answered(e, res["reasoning"], res["content"], u.get("prompt_tokens"), local)
            row.update(prompt_tokens=prompt, reused=reused, read=pl.get("read", tr.get("prompt_read")),
                       prompt_ms=prompt_ms, ttft_s=res["ttft_s"], first_engine_token_s=tr.get("first_engine_token_s"),
                       decode_tps=tr.get("decode_tok_s") or pl.get("decode_tps"), gen=u.get("completion_tokens"),
                       wall_s=res["wall_s"], finish=res["finish"], cache=eng["events"], evictions=eng["evictions"],
                       engine_lines=len(eng["prompt_lines"]), avail_gib=round((meminfo_avail() or 0) / 2**30, 1),
                       cold_reread=cold_reread(e["kind"], e["first"], prompt, reused))
            rows.append(row)
            if len(rows) == 1:
                print(f"  conversation cache: {budget_line()}", flush=True)
            flag = "  COLD RE-READ" if row["cold_reread"] else ""
            print(f"  [{e['i'] + 1:3d}/{len(plan)}] {e['kind']:5s} {e['conv']:>4s}#{e['turn']:<2d} {prompt or 0:6d} tok "
                  f"(plan {e['target']:6d}) reused {reused if reused is not None else '-':>6} "
                  f"prompt {(prompt_ms or 0) / 1000:5.1f} s ttft {res['ttft_s']:5.1f} s "
                  f"decode {row['decode_tps'] or 0:5.1f} tok/s wall {res['wall_s']:5.1f} s "
                  f"[{'; '.join(eng['events'])[:120]}]{flag}", flush=True)
    finally:
        noise, top = mon.result()
        peak_rss, min_avail = mem.result()
        abtest.stop_servers()
    tot = totals(rows)
    tot.update(run_s=round(time.time() - t_start, 1), peak_rss_gib=peak_rss, min_avail_gib=min_avail)
    return {"rows": rows, "totals": tot, "budget": budget_line(), "noise": noise, "noise_top": top,
            "reclaim": round(mon.reclaim, 2), "gpu": mon.gpu_summary(), "big_free_gib": round(abtest.big_free_gib(), 1)}, None


def totals(rows):
    ok = [r for r in rows if not r.get("error")]
    secs = lambda rs: round(sum((r.get("prompt_ms") or 0) for r in rs) / 1000, 2)   # noqa: E731
    agent = [r for r in ok if r["kind"] == "agent"]
    side = [r for r in ok if r["kind"] == "side"]
    rere = [r for r in agent if r.get("cold_reread")]
    first = [r for r in agent if r["first"]]
    ev = [e for r in ok for e in (r.get("cache") or [])]
    skips = {}
    for e in ev:
        if e.startswith("skip ("):
            why = re.sub(r"\d+", "N", e[6:-1])
            skips[why] = skips.get(why, 0) + 1
    return {"requests": len(rows), "errors": len(rows) - len(ok), "prompt_s": secs(ok), "prompt_s_agent": secs(agent),
            "prompt_s_side": secs(side), "turn_s": round(sum(r.get("wall_s") or 0 for r in ok), 1),
            "cold_rereads": len(rere), "cold_reread_s": secs(rere), "first_cold": len(first), "first_cold_s": secs(first),
            "parked": sum(e.startswith("parked") for e in ev), "restored": sum(e.startswith("restored") for e in ev),
            "skip_park": skips, "evictions": max([r["evictions"] for r in ok if r.get("evictions") is not None] or [0])}


def recorded_totals(plan):
    rs = [e for e in plan if e.get("rec")]
    if not rs:
        return None
    secs = lambda es: round(sum((e["rec"].get("prompt_ms") or 0) for e in es) / 1000, 1)   # noqa: E731
    agent = [e for e in rs if e["kind"] == "agent"]
    rere = [e for e in agent if cold_reread("agent", e["first"], e["rec"]["prompt_total"], e["rec"]["reused"])]
    return {"prompt_s": secs(rs), "prompt_s_agent": secs(agent), "prompt_s_side": secs([e for e in rs if e["kind"] == "side"]),
            "turn_s": round(sum(e["rec"].get("duration_s") or 0 for e in rs), 1), "cold_rereads": len(rere),
            "cold_reread_s": secs(rere), "first_cold": sum(e["first"] for e in agent),
            "first_cold_s": secs([e for e in agent if e["first"]])}


# ------------------------------------------------------------------------------------------------ reports
def print_plan(plan, label, counter, corpus, a, n_arms, reps):
    agent = [e for e in plan if e["kind"] == "agent"]
    side = [e for e in plan if e["kind"] == "side"]
    print(f"replay plan: {label}")
    print(f"  {len(plan)} requests: {len(agent)} agent turns, {len(side)} side requests "
          f"({sum(e['sub'] == 'decode' for e in side)} decodes, {sum(e['sub'] == 'probe' for e in side)} prefill probes, "
          f"{sum(bool(e.get('repeat')) for e in side)} of them repeat an earlier text); token counts: {counter.why}")
    for k in dict.fromkeys(e["conv"] for e in agent):
        es = [e for e in agent if e["conv"] == k]
        rec = [e for e in es if e.get("rec") and cold_reread("agent", e["first"], e["rec"]["prompt_total"], e["rec"]["reused"])]
        print(f"  conversation {k}: {len(es)} turns, {es[0]['target']:,} -> {es[-1]['target']:,} tokens"
              + (f"; recorded cold re-reads {len(rec)} ({sum(e['rec']['prompt_ms'] or 0 for e in rec) / 1000:.1f} s of prompt)"
                 if es[0].get("rec") else ""))
    if side:
        sz = sorted(e["target"] for e in side)
        print(f"  side requests: {sz[0]:,}-{sz[-1]:,} tokens, max_new {min(e['max_new'] for e in side)}-"
              f"{max(e['max_new'] for e in side)}")
    print(f"  prompt tokens sent: {sum(e['target'] for e in plan):,} (agent {sum(e['target'] for e in agent):,}, "
          f"side {sum(e['target'] for e in side):,}); output cap total {sum(e['max_new'] for e in plan):,}"
          + (f"; --scale {a.scale}" if a.scale != 1 else ""))
    # build every request as the replay would (placeholder replies): checks the sizing without a server
    convs, worst, sized, t0 = Conversations(counter, corpus), 0, [], time.time()
    for e in plan:
        msgs, n = convs.build(e)
        sized.append(n)
        worst = max(worst, abs(n - e["target"]))
        convs.placeholder_reply(e)
    print(f"  sizing check ({'exact counts' if counter.exact else 'estimate'}, placeholder replies): "
          f"max |built - recorded| = {worst} tokens over {len(plan)} requests ({time.time() - t0:.1f} s)")
    reads = growth_reads(plan)
    print("\n    #   gap   kind   conv turn  prompt  built  max_new think  rec.reused rec.prompt_s rec.wall_s  note")
    for e, n, rd in zip(plan, sized, reads):
        r = e.get("rec") or {}
        note = []
        if e["kind"] == "agent" and e["first"]:
            note.append("new conversation (cold)")
        if r and cold_reread("agent" if e["kind"] == "agent" else "side", e["first"], r.get("prompt_total"), r.get("reused")):
            note.append("recorded COLD RE-READ")
        if e.get("repeat"):
            note.append("repeat text")
        rec_cols = (f"{str(r.get('reused', '-')):>8s} {(r.get('prompt_ms') or 0) / 1000:10.1f} "
                    f"{r.get('duration_s') or 0:10.1f}" if r else f"{'-':>8s} {'-':>10s} {'-':>10s}")
        print(f"  {e['i']:3d} {e['gap_s']:5.1f} {e['kind'] if e['kind'] == 'agent' else e['sub']:6s} {e['conv']:>4s} "
              f"{e['turn']:4d} {e['target']:7d} {n:6d} {e['max_new']:7d} {'on ' if e['thinking'] else 'off'}   "
              f"{rec_cols}  {', '.join(note)}")
    gaps = sum(e["gap_s"] for e in plan)
    start = a.start_s
    rec = recorded_totals(plan)
    warm = sum(est_request_s(e, rd) for e, rd in zip(plan, reads))
    print()
    if rec:
        print(f"  recorded: prompt {rec['prompt_s']:.1f} s (agent {rec['prompt_s_agent']:.1f}, side {rec['prompt_s_side']:.1f}), "
              f"turn time {rec['turn_s']:.1f} s, cold re-reads {rec['cold_rereads']} ({rec['cold_reread_s']:.1f} s), "
              f"first reads {rec['first_cold']} ({rec['first_cold_s']:.1f} s)")
        base_s = rec["turn_s"] + gaps + start
        print(f"  expected per arm: as recorded {rec['turn_s'] / 60:.1f} min of requests + {gaps / 60:.1f} min of gaps (capped "
              f"at {a.max_gap:g} s) + ~{start / 60:.1f} min server start = {base_s / 60:.1f} min; with every agent turn "
              f"warm (model) {(warm + gaps + start) / 60:.1f} min")
    else:
        cold = list(reads)
        for j, e in enumerate(plan):            # worst case: every agent turn that follows a side request read cold
            if e["kind"] == "agent" and j and plan[j - 1]["kind"] == "side":
                cold[j] = e["target"]
        cold_s = sum(est_request_s(e, rd) for e, rd in zip(plan, cold))
        base_s = cold_s + gaps + start
        print(f"  expected per arm (model: cold 0.41 ms/token >= 8K, 0.64 below, 90-110 tok/s decode): every agent turn "
              f"warm {(warm + gaps + start) / 60:.1f} min, every interleaved agent turn cold {base_s / 60:.1f} min "
              f"(prompt time {sum(est_prompt_s(r) for r in reads):.0f} vs {sum(est_prompt_s(r) for r in cold):.0f} s; "
              f"gaps {gaps:.0f} s, server start ~{start:.0f} s)")
    print(f"  expected total: {n_arms} arm(s) x {reps} rep(s) ~ {n_arms * reps * base_s / 60:.0f} min (upper estimate)")


def report(results, arms, plan):
    rec = recorded_totals(plan)
    print("\nper arm run (prompt time = the server's prompt_ms, restores included)")
    print(f"  {'arm':22s} rep prompt_s  agent_s  side_s  turn_s  cold re-reads (s)  first(s)   parked/restored  "
          f"skip-park  min avail GiB  peak RSS GiB  budget")
    if rec:
        print(f"  {'(recorded 11-09)':22s}   - {rec['prompt_s']:8.1f} {rec['prompt_s_agent']:8.1f} {rec['prompt_s_side']:7.1f} "
              f"{rec['turn_s']:7.1f}  {rec['cold_rereads']:3d} ({rec['cold_reread_s']:6.1f})   {rec['first_cold']} "
              f"({rec['first_cold_s']:.1f})")
    for arm in arms:
        for res in results.get(arm, []):
            t = res["totals"]
            print(f"  {arm:22s} {res['rep']:3d} {t['prompt_s']:8.1f} {t['prompt_s_agent']:8.1f} {t['prompt_s_side']:7.1f} "
                  f"{t['turn_s']:7.1f}  {t['cold_rereads']:3d} ({t['cold_reread_s']:6.1f})   {t['first_cold']} "
                  f"({t['first_cold_s']:.1f})   {t['parked']:4d}/{t['restored']:<4d}       {sum(t['skip_park'].values()):4d}"
                  f"       {t['min_avail_gib'] or 0:6.1f}        {t['peak_rss_gib']:6.1f}   {res['budget']}"
                  f"{'' if res['clean'] else '  NOISY'}{'  ERRORS ' + str(t['errors']) if t['errors'] else ''}")
    base = arms[0]
    if not results.get(base):
        return
    print(f"\nverdict vs {base} (total prompt time; paired by rep, 95% interval when reps >= 2)")
    bvals = [r["totals"]["prompt_s"] for r in results[base]]
    bm = sum(bvals) / len(bvals)
    for arm in arms[1:]:
        vals = [r["totals"]["prompt_s"] for r in results.get(arm, [])]
        if not vals:
            print(f"  {arm:22s} no runs")
            continue
        n = min(len(vals), len(bvals))
        d = [x - y for x, y in zip(vals[:n], bvals[:n])]
        cr = [r["totals"]["cold_rereads"] for r in results[arm]]
        if n >= 2:
            dm, dh = abtest.ci(d)
            v = "better (less prompt time)" if dm + dh < 0 else "worse" if dm - dh > 0 else "no difference"
            print(f"  {arm:22s} {dm:+7.1f} s +- {dh:5.1f} ({dm / bm * 100:+5.1f}% +- {dh / bm * 100:4.1f}%)  -> {v};"
                  f"  cold re-reads {cr} vs {[r['totals']['cold_rereads'] for r in results[base]]}")
        else:
            print(f"  {arm:22s} {d[0]:+7.1f} s ({d[0] / bm * 100:+5.1f}%) prompt time, turn time "
                  f"{results[arm][0]['totals']['turn_s'] - results[base][0]['totals']['turn_s']:+.1f} s, cold re-reads "
                  f"{cr[0]} vs {results[base][0]['totals']['cold_rereads']}  (one rep: deltas only)")


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("spec")
    ap.add_argument("--out", required=True, help="directory for the plan, per-run logs and results")
    ap.add_argument("--session", help="time window '2026-11-09T16:13/16:43', 'sN' or a request id (default: the "
                                      "spec's replay.session)")
    ap.add_argument("--synthetic", action="store_true", help="the built-in 4.0.1-shaped session instead of the log")
    ap.add_argument("--turns", help="the per-turn log (default: the spec's replay.turns, else pancake-turns.jsonl)")
    ap.add_argument("--list-sessions", action="store_true")
    ap.add_argument("--session-gap", type=float, default=600, help="s of silence that splits the log into sessions")
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies every prompt size and output cap (a quick "
                                                             "smoke run; below 1 the snapshots no longer meet the "
                                                             "11-09 budget the same way: verdicts at 1.0 only)")
    ap.add_argument("--reps", type=int, help="replays per arm (A B B A order; default: spec replay.reps, else 1)")
    ap.add_argument("--arms", help="comma list (default: spec replay.arms, else every arm); 'all' = every arm and "
                                   "replay.add_arms")
    ap.add_argument("--base", help="the base preset (default: the spec's base)")
    ap.add_argument("--tokenizer", help="the tokenizer directory (default: the base preset's 'tokenizer')")
    ap.add_argument("--conv-min", type=int, default=8192, help="tokens: a request this large belongs to a conversation")
    ap.add_argument("--max-growth", type=int, default=16384, help="tokens a conversation may grow by between turns "
                                                                  "beyond its last output")
    ap.add_argument("--max-gap", type=float, default=2.0, help="s: think time between requests is capped at this")
    ap.add_argument("--max-new", choices=("output", "recorded"), default="output",
                    help="output (default): max_tokens = the recorded output tokens; recorded: the recorded max_new")
    ap.add_argument("--shared-system", type=int, help="tokens: every conversation (the agent's and the side requests) "
                                                      "starts with the same system prompt of about this size, so they "
                                                      "share a prefix as a main agent and its sub-agents do (default: "
                                                      "spec replay.shared_system, else 0 = each its own, as before; "
                                                      "needed to see MOD_PLAN A15's park TTL)")
    ap.add_argument("--start-s", type=float, default=90, help="s per server start, for the duration estimate")
    ap.add_argument("--dry-run", action="store_true", help="build and print the plan; no server")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--foreign-cpu", type=float, default=25.0)
    ap.add_argument("--max-wait", type=int, default=600)
    ap.add_argument("--retries", type=int, default=1, help="a noisy or failed arm run is repeated this many times")
    a = ap.parse_args()

    spec = json.load(open(a.spec))
    rp = spec.get("replay") or {}
    all_arms = {**spec["arms"], **(rp.get("add_arms") or {})}
    if a.arms == "all":
        arms = list(all_arms)
    else:
        arms = a.arms.split(",") if a.arms else list(rp.get("arms") or spec["arms"])
    bad = [x for x in arms if x not in all_arms]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; the spec has {list(all_arms)}")
    reps = a.reps or rp.get("reps", 1)
    global SHARED_SYSTEM
    SHARED_SYSTEM = max(0, a.shared_system if a.shared_system is not None else int(rp.get("shared_system", 0)))
    base_cfg = a.base or spec["base"]
    base_cfg = base_cfg if os.path.isabs(base_cfg) else str(ROOT / base_cfg)
    try:
        bc = json.load(open(base_cfg))
    except (OSError, ValueError) as e:
        if not a.dry_run:
            raise SystemExit(f"base preset {base_cfg}: {e}")
        print(f"note: base preset {base_cfg} not readable ({e}); the dry run goes on without it")
        bc = {}
    tokdir = a.tokenizer or bc.get("tokenizer") or str(ROOT / "pack/full/tokenizer")
    counter = Counter(tokdir, effort_end=bc.get("effort_position", "end") == "end")
    corpus = Corpus(ROOT)

    if a.synthetic:
        plan = plan_synthetic(gap_s=min(1.0, a.max_gap))
        label = "synthetic (4.0.1 shape: 40K -> 62K with interleaved side requests)"
    else:
        tpath = a.turns or rp.get("turns") or str(ROOT / "pancake-turns.jsonl")
        tpath = tpath if os.path.isabs(tpath) else str(ROOT / tpath)
        rows = load_turns(tpath)
        if a.list_sessions:
            for k, s in enumerate(split_sessions(rows, a.session_gap)):
                big = sum(r["prompt_total"] >= a.conv_min for r in s)
                print(f"s{k}: {hm(s[0]['time'])} - {hm(s[-1]['time'])}  {len(s)} requests, {big} >= {a.conv_min} tokens, "
                      f"largest {max(r['prompt_total'] for r in s):,}")
            return 0
        sel = a.session or rp.get("session")
        if not sel:
            raise SystemExit("name a --session (or --synthetic); --list-sessions shows them")
        got, label = select_session(rows, sel, a.session_gap)
        plan = plan_from_rows(got, a.conv_min, a.max_growth, a.max_gap, a.max_new)
        label += f", from {tpath}"
    if a.scale != 1:
        global COLD_MIN
        COLD_MIN = max(256, round(COLD_MIN * a.scale))      # a smoke run's "long" turn shrinks with it
        for e in plan:
            e["target"] = max(16, round(e["target"] * a.scale))
            e["max_new"] = max(1, round(e["max_new"] * a.scale))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump({"label": label, "arms": arms, "reps": reps, "plan": plan}, open(out / "plan.json", "w"), indent=1)
    if a.dry_run:
        print_plan(plan, label, counter, corpus, a, len(arms), reps)
        print(f"\narms: {arms}, reps {reps}; plan written to {out / 'plan.json'}; dry run: no server started")
        return 0

    print_plan(plan, label, counter, corpus, a, len(arms), reps)
    sampling = dict(abtest.SAMPLING)
    for k in ("temperature", "top_p", "seed"):
        if spec.get(k) is not None:
            sampling[k] = spec[k]
    order = []
    for r in range(reps):                       # A B B A ...: each arm sees early and late slots equally
        order += [(r, x) for x in (arms if r % 2 == 0 else arms[::-1])]
    results, runs_f, rows_f, deaths = {}, open(out / "runs.jsonl", "a"), open(out / "replay.jsonl", "a"), 0
    abtest.stop_servers()
    for r, arm in order:
        for attempt in range(a.retries + 1):
            abtest.quiet_gate(0.0, a.foreign_cpu, [], a.max_wait)
            cfg_path = str(out / f"{arm}-r{r}.json")
            abtest.make_config(base_cfg, all_arms[arm], cfg_path)
            print(f"\nrep {r} arm {arm} (attempt {attempt})", flush=True)
            res, err = run_arm(arm, r, cfg_path, plan, counter, corpus, a, out, sampling)
            if err:
                print(f"rep {r} {arm}: FAILED ({err})", flush=True)
                deaths += "died" in err
                if deaths >= 2:
                    print(f"ABORT: the server died twice before ready (free RAM in 2 MiB blocks: "
                          f"{abtest.big_free_gib():.0f} GiB; 'sudo tools/pancake/os_tune.sh compact')", flush=True)
                    report(results, arms, plan)
                    return 2
                continue
            res["clean"] = res["noise"] <= a.foreign_cpu and res["reclaim"] <= abtest.RECLAIM_MAX
            res["rep"], res["arm"], res["attempt"] = r, arm, attempt
            for row in res["rows"]:
                rows_f.write(json.dumps(row) + "\n")
            rows_f.flush()
            runs_f.write(json.dumps({k: v for k, v in res.items() if k != "rows"}) + "\n")
            runs_f.flush()
            t = res["totals"]
            print(f"rep {r} {arm}: prompt {t['prompt_s']:.1f} s (agent {t['prompt_s_agent']:.1f}), turn {t['turn_s']:.1f} s, "
                  f"cold re-reads {t['cold_rereads']} ({t['cold_reread_s']:.1f} s), min avail {t['min_avail_gib']} GiB, "
                  f"peak RSS {t['peak_rss_gib']} GiB, foreign CPU {res['noise']:.0f}%"
                  f"{'' if res['clean'] else '  NOISY'}", flush=True)
            if res["clean"] or attempt == a.retries:
                results.setdefault(arm, []).append(res)
                break
    report(results, arms, plan)
    json.dump({"label": label, "arms": arms, "reps": reps,
               "runs": {k: [{x: y for x, y in v.items() if x != "rows"} for v in vs] for k, vs in results.items()},
               "recorded": recorded_totals(plan)}, open(out / "summary.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
