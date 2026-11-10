#!/usr/bin/env python3
"""suffix_sim.py - A24: replays recorded agent turns and counts the draft tokens a cross-session suffix index with
frequency scoring (SuffixDecoding, Oliaro et al., arXiv 2411.04975) would propose and get accepted, next to the
per-request --lookup drafter the engine has now.  No model, no GPU, no server.

    python tools/pancake/suffix_sim.py [--db ~/.local/share/opencode/opencode.db] [--model REGEX] [--k 11]
           [--mib 64] [--min-prob 0.1] [--factor 1.0] [--limit N] [--reset-gap MIN] [--json OUT] [--tokenizer DIR]
    python tools/pancake/suffix_sim.py --self-test

Input: opencode's own session store (SQLite, read-only), which holds the text of every agent turn the engine served:
the user messages, each assistant step's reasoning, text and tool calls, and each tool's output.  The per-turn log
replay_turns.py reads (pancake-turns.jsonl) has numbers only, so it cannot feed a drafter.  Only assistant messages of
models matching --model are replayed (default: the daily Qwen3.8-Flash-Next served by the strata provider); a session
with a parent is a sub-agent's.  Every step of an assistant message (a step-start part) is one request.

How a request is rebuilt (an approximation of the server's chat template, the system prompt and the tool definitions
are not stored and left out): the prompt is the session so far -
    <|im_start|>user\\n{text}<|im_end|>\\n
    <|im_start|>assistant\\n<think>\\n{reasoning}\\n</think>\\n\\n{text}<tool_call>\\n<function=NAME>\\n<parameter=K>\\nV\\n</parameter>\\n...
    <|im_end|>\\n<|im_start|>user\\n<tool_response>\\n{output}\\n</tool_response><|im_end|>\\n
and the output (what the model wrote, the tokens the drafters guess) is the step's reasoning, text and tool calls.
Requests run in the order they started, one at a time (the engine serves one at a time).

The index holds at most --mib MiB as the engine sizes it (a power of two of tokens at 40 bytes a token; 64 MiB ->
1,048,576 tokens), the oldest output evicted first.

The decode is simulated as suffix_drafter_test --simulate does: at each round a source drafts up to --k tokens (the
--lookup window minus its first row: 11 for --lookup-max-t 12), the round takes the drafted prefix that equals the
recorded output plus one model token (the verify's bonus).  The recorded output is what a greedy decode wrote, so a
draft is accepted exactly where it equals it.  Arms:
  lookup        the engine's --lookup drafter, exactly (SuffixDrafter: trigram -> 4 newest places over this request's
                prompt + output, longest match >= 3 back to 64 tokens, newest on ties, the whole continuation)
  index         the cross-session suffix index alone: previous OUTPUTS of every session (main agent and sub-agents),
                each after a lead of its prompt's last 64 tokens (matched into, not indexed); chains of the places
                that end with the same 24 / 8 / 3 tokens, the deepest match (back to 64 tokens) among the 64 newest
                places of the longest K that matches, then a vote a token at a time: the token most of the places
                go on with, its share of them is its probability, the draft ends where the product falls under
                --min-prob or after --factor x the match length tokens (the paper's MAX_SPEC = alpha * p)
  fallback      lookup, and the index only where lookup has no draft (STRATA_SUFFIX_INDEX=1's rule in the engine)
  longest       the draft with the longer match of the two (propose_from_sources' rule)
Per arm: rounds, tokens per round (the ceiling speed-up when a window costs what a one-token step costs), draft tokens
offered and accepted, acceptance; the index's own gain = accepted tokens the combined arm adds over lookup.
"""
import argparse
import bisect
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = Path.home() / ".local/share/opencode/opencode.db"
DEFAULT_TOK = Path("/home/dan/clone-of-d/Strata-data/packs/iq2_xs/tokenizer")
BYTES_PER_TOKEN = 40           # the engine's SuffixIndex: ring 4 + 3 levels x (chain 4 + head 8)
LEVELS = (24, 8, 3)            # the K of each chain, longest first
DEPTH = 64                     # a match is compared back this far at most
WALK = 64                      # places looked at on a chain
LEAD = 64                      # prompt tokens before an output in the index (matched into, not indexed)
MARK = -1


def cap_tokens(mib):
    """The engine's ring: the largest power of two of tokens within the cap at 40 bytes a token, 4096 at least."""
    cap = 4096
    while cap * 2 * BYTES_PER_TOKEN <= mib * (1 << 20):
        cap *= 2
    return cap


# ------------------------------------------------------------------------------------------------ the drafters
class Lookup:
    """The engine's SuffixDrafter (src/spec/suffix_drafter.cpp) over one session's history, queried as of a place:
    every trigram's places are kept, the query takes the 4 newest at or before the end (the C++ table's WAYS), the
    end's own place included, as the C++ code has just inserted it."""
    WAYS, MIN, MAX = 4, 3, 64

    def __init__(self):
        self.h = []
        self.tri = {}

    def extend(self, toks):
        h, tri = self.h, self.tri
        for t in toks:
            h.append(t)
            e = len(h) - 1
            if e >= 2:
                tri.setdefault((h[e - 2], h[e - 1], t), []).append(e)

    def propose(self, n, k):
        """Draft for the history's first n tokens (n <= len(h)): (tokens, match)."""
        if n < 4 or k <= 0:
            return [], 0
        h, cur = self.h, n - 1
        ps = self.tri.get((h[cur - 2], h[cur - 1], h[cur]))
        if not ps:
            return [], 0
        j = bisect.bisect_right(ps, cur)
        best_len, best_end = 0, 0
        for p in reversed(ps[max(0, j - self.WAYS):j]):          # newest first
            if p >= cur:
                continue
            ln = 0
            while ln < self.MAX and ln <= p and h[p - ln] == h[cur - ln]:
                ln += 1
            if ln > best_len:
                best_len, best_end = ln, p
        if best_len < self.MIN:
            return [], 0
        return h[best_end + 1:min(cur + 1, best_end + 1 + k)], best_len


class Index:
    """The cross-session suffix index (include/strata/spec/suffix_index.hpp), in Python: a ring of the previous
    outputs, three hash chains, frequency-voted drafts.  Queries take the pattern (the context's last tokens)."""

    def __init__(self, cap_tokens, min_prob=0.1, factor=1.0):
        self.cap, self.min_prob, self.factor = cap_tokens, min_prob, factor
        self.ring = []                              # absolute places; a place below low() is evicted
        self.head = [dict() for _ in LEVELS]
        self.back = [[] for _ in LEVELS]
        self.run = 0

    def low(self):
        return max(0, len(self.ring) - self.cap)

    def insert(self, lead, out):
        """One finished output, after a lead of its prompt's last tokens: matched into, and only the lead's last place
        (whose continuation is the output's first token) is a draft's place."""
        ring = self.ring
        if ring and ring[-1] != MARK:
            self._put(MARK, False)
        lead = lead[-LEAD:]
        for j, t in enumerate(lead):
            self._put(t, j == len(lead) - 1)
        for t in out:
            self._put(t, True)

    def _put(self, t, indexed):
        ring = self.ring
        at = len(ring)
        ring.append(t)
        self.run = 0 if t == MARK else self.run + 1
        for li, k in enumerate(LEVELS):
            b = self.back[li]
            if not indexed or self.run < k:
                b.append(-1)
                continue
            key = tuple(ring[at - k + 1:at + 1])
            hd = self.head[li]
            b.append(hd.get(key, -1))
            hd[key] = at

    def propose(self, pat, k):
        """Draft after `pat` (the context's last <= DEPTH tokens): (tokens, match, probabilities)."""
        if k <= 0 or len(pat) < LEVELS[-1]:
            return [], 0, []
        ring, low, n_pat = self.ring, self.low(), len(pat)
        places, match = [], 0
        for li, kk in enumerate(LEVELS):
            if n_pat < kk:
                continue
            p = self.head[li].get(tuple(pat[n_pat - kk:]), -1)
            walked = 0
            b = self.back[li]
            while p >= 0 and walked < WALK and p >= low:
                walked += 1
                most = min(n_pat, DEPTH, p - low + 1)
                m = kk                               # the chain's key matched kk tokens exactly
                while m < most and ring[p - m] == pat[n_pat - 1 - m]:
                    m += 1
                if m > match:
                    match, places = m, [p]
                elif m == match:
                    places.append(p)
                p = b[p]
            if places:
                break
        if not places:
            return [], 0, []
        k = min(k, max(1, int(self.factor * match)))
        out, probs, sure, end = [], [], 1.0, len(ring)
        for i in range(k):
            nxt = {}
            live = []
            for q in places:
                x = q + 1 + i
                if x >= end or ring[x] == MARK:
                    continue
                live.append(q)
                nxt[ring[x]] = nxt.get(ring[x], 0) + 1
            if not live:
                break
            # the most frequent; of equals, the one the newest place has (places are newest first)
            best, best_n = None, 0
            for q in live:
                t = ring[q + 1 + i]
                if nxt[t] > best_n:
                    best, best_n = t, nxt[t]
            sure *= best_n / len(live)
            if sure < self.min_prob:
                break
            out.append(best)
            probs.append(sure)
            places = [q for q in live if ring[q + 1 + i] == best]
        return out, (match if out else 0), probs


# ------------------------------------------------------------------------------------------------ the turns
def load_turns(db, model_re, limit=0):
    """[(time, session, is_sub, prompt_delta_text_pieces, output_pieces)], pieces = [str | ('special', name)]."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    sess = {sid: parent for sid, parent in con.execute("select id, parent_id from session")}
    msgs = {}
    for mid, sid, t, d in con.execute("select id, session_id, time_created, data from message"):
        m = json.loads(d)
        msgs[mid] = (sid, t, m.get("role"), f"{m.get('providerID')}/{m.get('modelID')}")
    parts = {}
    for mid, t, pid, d in con.execute("select message_id, time_created, id, data from part"):
        if mid in msgs:
            parts.setdefault(mid, []).append((t, pid, json.loads(d)))
    con.close()
    rx = re.compile(model_re)
    by_sess = {}
    for mid, (sid, t, role, model) in msgs.items():
        by_sess.setdefault(sid, []).append((t, mid, role, model))
    reqs = []
    for sid, ms in by_sess.items():
        ms.sort()
        pending = []                                  # prompt pieces not yet seen by a request of this session
        for t, mid, role, model in ms:
            ps = sorted(parts.get(mid, []), key=lambda x: (x[0], x[1]))
            if role == "user":
                txt = "".join(p.get("text", "") for _, _, p in ps if p.get("type") == "text")
                pending += [("special", "<|im_start|>"), "user\n" + txt, ("special", "<|im_end|>"), "\n"]
                continue
            steps, cur = [], None
            for pt, _, p in ps:
                ty = p.get("type")
                if ty == "step-start" or cur is None:
                    cur = {"t": pt, "think": [], "text": [], "tools": []}
                    steps.append(cur)
                    if ty == "step-start":
                        continue
                if ty == "reasoning":
                    cur["think"].append(p.get("text", ""))
                elif ty == "text":
                    cur["text"].append(p.get("text", ""))
                elif ty == "tool":
                    st = p.get("state", {})
                    cur["tools"].append((p.get("tool", ""), st.get("input") or {}, str(st.get("output", ""))))
            for s in steps:
                if not (s["think"] or s["text"] or s["tools"]):
                    continue
                out = []
                think = "".join(s["think"])
                body = think + "\n</think>\n\n" + "".join(s["text"])
                for name, args, _ in s["tools"]:
                    body += f"<tool_call>\n<function={name}>\n"
                    for kk, v in (args.items() if isinstance(args, dict) else []):
                        body += f"<parameter={kk}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
                    body += "</function>\n</tool_call>"
                out = [body, ("special", "<|im_end|>")]
                head = [("special", "<|im_start|>"), "assistant\n<think>\n"]
                if rx.search(model):
                    reqs.append((s["t"], sid, sess.get(sid) is not None, pending + head, out))
                    pending = []
                else:                                  # another model's turn: history only
                    pending += head + out
                pending += ["\n"]
                for _, _, res in s["tools"]:
                    pending += [("special", "<|im_start|>"), "user\n<tool_response>\n" + res + "\n</tool_response>",
                                ("special", "<|im_end|>"), "\n"]
    reqs.sort(key=lambda r: r[0])
    return reqs[:limit] if limit else reqs


class Tok:
    """The server's tokenizer (tools/strata_tokenizer.py over a pack's tokenizer/); else a word-level stand-in."""

    def __init__(self, tokdir):
        self.why = ""
        try:
            sys.path.insert(0, str(ROOT / "tools"))
            import strata_tokenizer as ST
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
            self.special = {s: vocab[s] for s in ("<|im_start|>", "<|im_end|>") if s in vocab}
            self.why = f"server tokenizer ({tp})"
        except Exception as e:  # noqa: BLE001 - any failure: the stand-in, said once
            self.tok, self.special, self.ids = None, {}, {}
            self.why = f"word-level stand-in ({type(e).__name__}: {e})"
        self.rx = re.compile(r"\w+|\s+|[^\w\s]")

    def encode(self, pieces):
        out = []
        for p in pieces:
            if isinstance(p, tuple):
                out.append(self.special.get(p[1], 1_000_000 + len(p[1])))
            elif self.tok is not None:
                out += self.tok.encode(p)
            else:
                out += [self.ids.setdefault(w, len(self.ids)) for w in self.rx.findall(p)]
        return out


# ------------------------------------------------------------------------------------------------ the replay
ARMS = ("lookup", "index", "fallback", "longest")


def accept(draft, out, i):
    a = 0
    while a < len(draft) and i + a < len(out) and draft[a] == out[i + a]:
        a += 1
    return a


def simulate(reqs_tok, k=11, cap_tokens=1 << 21, min_prob=0.1, factor=1.0, times=None, reset_gap_s=0.0):
    """reqs_tok: [(session, is_sub, prompt_delta_ids, output_ids)] in order.  Returns per-arm totals.  With `times`
    (request start, seconds) and reset_gap_s > 0 the index starts empty after every pause longer than that (the
    engine's index lives in memory: a pause stands in for a server restart)."""
    look = {}
    idx = Index(cap_tokens, min_prob, factor)
    zero = lambda: {"tokens": 0, "rounds": 0, "windows": 0, "offered": 0, "accepted": 0}  # noqa: E731
    tot = {g: {a: zero() for a in ARMS} for g in ("all", "main", "sub")}
    src = {"index_only_windows": 0, "index_only_accepted": 0, "index_proposals": 0, "index_sure_sum": 0.0,
           "index_resets": 0}
    for ri, (sid, sub, delta, out) in enumerate(reqs_tok):
        if reset_gap_s > 0 and times is not None and ri > 0 and times[ri] - times[ri - 1] > reset_gap_s:
            idx = Index(cap_tokens, min_prob, factor)
            src["index_resets"] += 1
        L = look.setdefault(sid, Lookup())
        L.extend(delta)
        base = len(L.h)
        L.extend(out)                               # queried as of each place below
        lead = L.h[max(0, base - LEAD):base]
        cache_l, cache_i = {}, {}

        def q_look(i):
            if i not in cache_l:
                cache_l[i] = L.propose(base + i, k)
            return cache_l[i]

        def q_idx(i):
            if i not in cache_i:
                n = base + i
                d, m, pr = idx.propose(L.h[max(0, n - DEPTH):n], k)
                cache_i[i] = (d, m)
                if d:
                    src["index_proposals"] += 1
                    src["index_sure_sum"] += sum(pr)
            return cache_i[i]

        for arm in ARMS:
            i, n_out = 0, len(out)
            st = [tot["all"][arm], tot["sub" if sub else "main"][arm]]
            while i < n_out:
                if arm == "lookup":
                    d, _ = q_look(i)
                elif arm == "index":
                    d, _ = q_idx(i)
                else:
                    dl, ml = q_look(i)
                    if arm == "fallback" and dl:
                        d = dl
                    else:
                        di, mi = q_idx(i)
                        d = di if (di and (not dl or mi > ml)) else dl
                        if arm == "fallback" and di and not dl:
                            src["index_only_windows"] += 1
                            src["index_only_accepted"] += accept(di, out, i)
                a = accept(d, out, i)
                for s in st:
                    s["rounds"] += 1
                    s["windows"] += 1 if d else 0
                    s["offered"] += len(d)
                    s["accepted"] += a
                i += a + 1
            for s in st:
                s["tokens"] += n_out
        idx.insert(lead, out)                       # the finished output joins the shared index
    return tot, src


def report(tot, src, out=sys.stdout):
    def line(g, a, s):
        tpr = s["tokens"] / s["rounds"] if s["rounds"] else 0
        acc = s["accepted"] / s["offered"] if s["offered"] else 0
        per_k = 1000.0 * s["accepted"] / s["tokens"] if s["tokens"] else 0
        print(f"  {g:5s} {a:9s} tokens {s['tokens']:9d} rounds {s['rounds']:9d} tok/round {tpr:5.3f} windows "
              f"{s['windows']:8d} offered {s['offered']:9d} accepted {s['accepted']:8d} ({acc:5.1%}) "
              f"accepted/1k tokens {per_k:6.1f}", file=out)
    for g in ("all", "main", "sub"):
        for a in ARMS:
            line(g, a, tot[g][a])
    base = tot["all"]["lookup"]
    for a in ("fallback", "longest"):
        c = tot["all"][a]
        if base["rounds"]:
            print(f"  {a}: +{c['accepted'] - base['accepted']} accepted draft tokens over lookup "
                  f"({(c['accepted'] - base['accepted']) / max(1, base['accepted']):+.1%}), rounds "
                  f"{(c['rounds'] - base['rounds']) / base['rounds']:+.2%}", file=out)
    if src.get("index_resets"):
        print(f"  index reset {src['index_resets']} times (pauses above --reset-gap)", file=out)
    if src["index_only_windows"]:
        print(f"  windows where only the index drafts (fallback): {src['index_only_windows']}, accepted "
              f"{src['index_only_accepted']} ({src['index_only_accepted'] / src['index_only_windows']:.2f} a window)",
              file=out)


# ------------------------------------------------------------------------------------------------ self-test
def self_test():
    # lookup: an exact repeat inside the history drafts its continuation
    L = Lookup()
    L.extend([1, 2, 3, 4, 5, 6, 9, 9, 2, 3, 4])
    d, m = L.propose(len(L.h), 5)
    assert d == [5, 6, 9, 9, 2] and m == 3, (d, m)
    # index: two outputs agree after "7 8 9": the vote takes their token, the third disagrees at 1/3 -> still drafted
    idx = Index(1 << 16, min_prob=0.1, factor=4.0)
    idx.insert([], [7, 8, 9, 10, 11, 12])
    idx.insert([], [7, 8, 9, 10, 11, 13])
    idx.insert([], [7, 8, 9, 20, 21, 22])
    d, m, pr = idx.propose([5, 7, 8, 9], 4)
    assert d[:3] == [10, 11, 13] or d[:3] == [10, 11, 12], d
    assert abs(pr[0] - 2 / 3) < 1e-9, pr
    # eviction: a tiny ring forgets the first output
    idx = Index(8, min_prob=0.1, factor=4.0)
    idx.insert([], [1, 2, 3, 4, 5])
    idx.insert([], [6, 7, 8, 9, 10, 11, 12])
    assert idx.propose([1, 2, 3], 2)[0] == [], "evicted output still drafts"
    assert idx.propose([7, 8, 9], 2)[0] == [10, 11]
    # the lead is matched into but never drafted from
    idx = Index(1 << 16, min_prob=0.1, factor=4.0)
    idx.insert([40, 41, 42, 43], [50, 51, 52])
    assert idx.propose([41, 42, 43], 3)[0] == [50, 51, 52]
    assert idx.propose([40, 41, 42], 3)[0] == [], "a lead place drafted"
    # a full replay on synthetic turns: the index helps where sessions repeat one another's outputs
    reqs = []
    for s in range(6):
        reqs.append((f"s{s}", s > 0, [100 + s, 101, 102], [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]))
    tot, _ = simulate(reqs, k=8, cap_tokens=1 << 16, min_prob=0.1, factor=4.0)
    assert tot["all"]["index"]["accepted"] > 0 and tot["all"]["lookup"]["accepted"] == 0, tot["all"]
    assert tot["all"]["fallback"]["accepted"] == tot["all"]["index"]["accepted"]
    print("suffix_sim self-test: ok")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--model", default=r"^strata/qwen3\.8-flash-next", help="provider/model regex of the replayed turns")
    ap.add_argument("--k", type=int, default=11, help="draft tokens a window may take (window - 1)")
    ap.add_argument("--mib", type=float, default=64, help="index memory cap (MiB, 40 bytes a token)")
    ap.add_argument("--min-prob", type=float, default=0.1, help="a draft ends where its probability falls under this")
    ap.add_argument("--factor", type=float, default=1.0, help="a draft is at most factor x match tokens long")
    ap.add_argument("--limit", type=int, default=0, help="the first N requests only")
    ap.add_argument("--reset-gap", type=float, default=0, help="minutes: the index starts empty after a pause longer "
                    "than this between two requests (a stand-in for a server restart; 0: never)")
    ap.add_argument("--tokenizer", default=os.environ.get("STRATA_TOKENIZER", str(DEFAULT_TOK)))
    ap.add_argument("--json", help="write the totals here")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    t0 = time.time()
    reqs = load_turns(a.db, a.model, a.limit)
    tok = Tok(a.tokenizer)
    rt = [(sid, sub, tok.encode(delta), tok.encode(out)) for _, sid, sub, delta, out in reqs]
    n_sess = len({r[0] for r in rt})
    n_sub = len({r[0] for r in rt if r[1]})
    print(f"suffix_sim: {len(rt)} requests in {n_sess} sessions ({n_sub} sub-agent sessions), output "
          f"{sum(len(r[3]) for r in rt)} tokens, prompt deltas {sum(len(r[2]) for r in rt)} tokens; {tok.why}; "
          f"loaded in {time.time() - t0:.1f}s")
    cap = cap_tokens(a.mib)
    t1 = time.time()
    times = [r[0] / 1000.0 for r in reqs]          # opencode stores milliseconds
    tot, src = simulate(rt, a.k, cap, a.min_prob, a.factor, times, a.reset_gap * 60.0)
    print(f"suffix_sim: k {a.k}, index cap {cap} tokens ({a.mib:g} MiB), min-prob {a.min_prob}, factor {a.factor}; "
          f"replayed in {time.time() - t1:.1f}s")
    report(tot, src)
    if a.json:
        Path(a.json).write_text(json.dumps({"args": vars(a), "totals": tot, "sources": src}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
