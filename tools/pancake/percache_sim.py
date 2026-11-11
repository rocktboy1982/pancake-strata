#!/usr/bin/env python3
"""percache_sim.py - offline model of the per-turn cache levers (MOD_PLAN 16.2: A11 tool-boundary checkpoints and FLOP
eviction, A12 pinned prefix, A13 disk tier; items 16, 31, 39, 71, 90) on the real opencode history.  No model, no GPU,
no server.

    python tools/pancake/percache_sim.py [--db ~/.local/share/opencode/opencode.db] [--model REGEX]
           [--arms base,flags,flags+disk,pin,pin+flags,keep-all] [--json OUT] [--no-micro]
    python tools/pancake/percache_sim.py --self-test

Input: opencode's own session store (read-only).  Every assistant message of the served model is one request of the
engine; its `tokens` field holds what the Strata server reported for it: `cache.read` = tokens reused (usage cached_tokens),
`input` = tokens read, `output`.  So the history already says, per request, how much of the prompt was read again.  The
text is never used.  Three parts:

1. classify: the tokens read beyond the new content (prompt growth since the session's previous request) are the
   "re-read"; each is labelled by what happened before it (first request of a session, compaction or prune shrank the
   prompt, another session's request ran in between = a switch, a user message arrived, a long idle gap, or nothing the
   store shows).  Seconds = tokens x 0.375 ms (the marginal prefill rate fitted on the 2026-11-09 cold reads, R2 0.99).
2. park model: the requests replayed in global time order through the engine's conversation cache as the code has it
   (one live state; parked snapshots under a MiB budget and a slot count, oldest out first; a snapshot's size
   111 MiB + 14.9 KiB/token + 113 MiB/checkpoint as in bench/results/2026-11-10-offline/park_replay.py).  A request
   continues its own session's state when it has one (parked, on disk, or live), else it starts from the family root
   (the system prompt + tools: the shared prefix of the agent kind) when a live or parked state of that family still holds
   it, else from 0.  Arms are the engine's switches: the 4.0.1 flags, the disk tier (A13), the pinned prefix with its
   rule that a parked entry holding a pin is never evicted (A12/90/71), and an unlimited store as the ceiling.
3. micro: the checkpoint chain of one 120K conversation under the engine's retention (--prompt-cache 6): least recent use
   vs Marconi's FLOP rule vs spread, with and without the tool-boundary checkpoints, for edits at random places and at
   tool-result boundaries.  Expected tokens read again.

What this cannot say (printed with the numbers): the request text is not in the store, so a re-read inside a session
without a recorded cause stays "other"; the history mixes engine settings of 2026-10-15..18 (the conversation cache did
less then), so the base arm is checked against the recorded reuse and its agreement is printed; a family's shared prefix
is estimated from repeated `reused` values.  Gains are modelled seconds of prefill, as a share of the summed request time.
"""
import argparse
import collections
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

DB_DEFAULT = os.path.expanduser("~/.local/share/opencode/opencode.db")
MS_PER_TOKEN = 0.375            # marginal prefill: ms = 1583 + 0.375 x tokens on the 11-09 cold reads (replay_turns log)
PROMPT_CACHE = 6                # --prompt-cache default: checkpoints one conversation keeps
EVERY = 16384                   # --prompt-cache-every default
MIB_BASE, KIB_PER_TOKEN, MIB_PER_CKPT = 111.0, 14.9, 113.0   # snapshot size model (park_replay.py)
COLD_FRACTION = 0.5             # a continuing request that reused less than this share of the state is a miss


# --------------------------------------------------------------------------------------------------- the history
def connect(db):
    try:
        return sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)


def load_steps(db, model_re="qwen3.8-flash-next"):
    """Requests of the served model in time order: dicts with sid, agent, sub, created, done (ms), total, reused, input, out;
    plus per-session event times (user messages, compaction parts, pruned tool parts)."""
    con = connect(db)
    parent = {sid: p for sid, p in con.execute("select id, parent_id from session")}
    msgsess, users, steps = {}, collections.defaultdict(list), []
    rx = re.compile(model_re)
    for mid, sid, t, d in con.execute("select id, session_id, time_created, data from message"):
        m = json.loads(d)
        msgsess[mid] = sid
        if m.get("role") == "user":
            users[sid].append((m.get("time") or {}).get("created", t))
        elif m.get("role") == "assistant" and rx.search(str(m.get("modelID", ""))):
            tk = m.get("tokens") or {}
            reused = (tk.get("cache") or {}).get("read", 0)
            total = tk.get("input", 0) + reused
            tm = m.get("time") or {}
            if total <= 0 or not tm.get("completed"):
                continue
            steps.append(dict(sid=sid, agent=str(m.get("agent") or m.get("mode") or "?"), sub=parent.get(sid) is not None,
                              created=tm.get("created", t), done=tm["completed"], total=total, reused=reused,
                              input=tk.get("input", 0), out=tk.get("output", 0)))
    comp, prune = collections.defaultdict(list), collections.defaultdict(list)
    for mid, t, d in con.execute("select message_id, time_created, data from part"):
        sid = msgsess.get(mid)
        if sid is None:
            continue
        if '"compaction"' in d[:60]:
            comp[sid].append(t)
        elif '"compacted"' in d:
            ct = ((json.loads(d).get("state") or {}).get("time") or {}).get("compacted")
            if ct:
                prune[sid].append(ct)
    con.close()
    steps.sort(key=lambda s: s["created"])
    return steps, dict(users=users, comp=comp, prune=prune)


def annotate(steps, events, idle_s=300.0):
    """kind (first / cont / shrink), necessary new tokens, excess (re-read) and cause for every request."""
    prev_of, last = {}, {}
    for i, s in enumerate(steps):
        s["i"] = i
        prev_of[i] = last.get(s["sid"])
        last[s["sid"]] = i
    for i, s in enumerate(steps):
        j = prev_of[i]
        p = steps[j] if j is not None else None
        if p is None:
            s.update(kind="first", proc_prev=0, need=s["total"], excess=0)
        else:
            proc = p["total"] + p["out"]
            if s["total"] >= proc - 2:
                s.update(kind="cont", proc_prev=proc, need=s["total"] - proc)
            else:
                s.update(kind="shrink", proc_prev=proc, need=0)
            s["excess"] = max(0, s["input"] - s["need"])
        lo, hi = ((p["done"], s["created"]) if p else (0, s["created"]))
        win = lambda ts: sum(1 for t in ts if lo <= t <= hi)   # noqa: E731
        s["users"], s["comps"], s["prunes"] = (win(events["users"][s["sid"]]), win(events["comp"][s["sid"]]),
                                               win(events["prune"][s["sid"]]))
        s["switch"] = p is not None and any(steps[k]["sid"] != s["sid"] for k in range(j + 1, i))
        s["gap_s"] = (s["created"] - p["done"]) / 1000.0 if p else 0.0
        s["cause"] = cause_of(s, idle_s)
    return steps


def cause_of(s, idle_s=300.0):
    if s["kind"] == "first":
        return "first request of a session"
    if s["kind"] == "shrink":
        return "compaction or prune (prompt shrank)" if (s["comps"] or s["prunes"]) else "prompt shrank, no event in the store"
    if s["comps"]:
        return "after a compaction part"
    if s["switch"]:
        return "switch (another session ran between)"
    if s["prunes"]:
        return "after a prune"
    if s["users"]:
        return "after a new user message"
    if s["gap_s"] > idle_s:
        return "idle gap"
    return "other (no event in the store)"


def classify(steps, min_excess=1000):
    busy = sum(s["done"] - s["created"] for s in steps) / 1000.0
    tok, cnt = collections.Counter(), collections.Counter()
    for s in steps:
        c = "first request of a session" if s["kind"] == "first" else s["cause"]
        e = s["input"] if s["kind"] == "first" else s["excess"]
        if e > min_excess or s["kind"] == "first":
            tok[c] += e
            cnt[c] += 1
    return dict(busy_s=busy, input=sum(s["input"] for s in steps), reread=dict(tok), steps=dict(cnt),
                sessions=len({s["sid"] for s in steps}), n=len(steps))


# --------------------------------------------------------------------------------------------------- the park model
def family_roots(steps):
    """agent kind -> estimated shared prefix tokens (the system prompt + tools).  The `reused` value that two or more
    sessions of the kind show on their first two requests; else 0.97 of the smallest first request."""
    by_sess = collections.defaultdict(list)
    for s in steps:
        by_sess[s["sid"]].append(s)
    first = collections.defaultdict(list)
    cand = collections.defaultdict(lambda: collections.defaultdict(set))
    for sid, L in by_sess.items():
        a = L[0]["agent"]
        first[a].append(L[0]["total"])
        for s in L[:2]:
            if s["reused"] >= 2048 and s["reused"] <= 0.98 * s["total"]:
                cand[a][s["reused"]].add(sid)
    roots = {}
    for a, tots in first.items():
        shared = [v for v, ss in cand[a].items() if len(ss) >= 2]
        roots[a] = max(shared) if shared else int(0.97 * min(tots))
    return roots


def snapshot_mib(tokens, ckpts):
    return MIB_BASE + KIB_PER_TOKEN / 1024.0 * tokens + MIB_PER_CKPT * ckpts


def n_checkpoints(length, root, resume):
    """root + the turn boundary + one every EVERY tokens from where the request resumed, capped like --prompt-cache."""
    n = (1 if root and length > root else 0) + 1 + max(0, (length - resume) // EVERY)
    return min(n, PROMPT_CACHE)


class Arm:
    def __init__(self, name, budget=1280.0, slots=4, min_tokens=0, disk=False, disk_gib=20.0, pin=False, keep_all=False,
                 disk_gbps=1.5, copy_s=0.9):
        self.name, self.budget, self.slots, self.min_tokens = name, budget, slots, min_tokens
        self.disk, self.disk_gib, self.pin, self.keep_all = disk, disk_gib, pin, keep_all
        self.disk_gbps, self.copy_s = disk_gbps, copy_s


ARMS = {
    "base": lambda: Arm("base"),
    "flags": lambda: Arm("flags", budget=2048.0, min_tokens=8192),
    "flags+disk": lambda: Arm("flags+disk", budget=2048.0, min_tokens=8192, disk=True),
    "pin": lambda: Arm("pin", pin=True),
    "pin+flags": lambda: Arm("pin+flags", budget=2048.0, min_tokens=8192, pin=True),
    "keep-all": lambda: Arm("keep-all", budget=1e9, slots=10 ** 6, min_tokens=0),
}


def simulate(steps, arm, roots):
    """Replay the requests through the conversation cache; returns per-request sim tokens read, restore seconds and the
    agreement with the recorded cold/hit outcome of the continuing requests."""
    live = None                     # dict(sid, fam, L, ck, pinned)
    parked = []                     # dicts as live + size, stamp
    disk = collections.OrderedDict()   # sid -> dict (L, size)
    clock = 0
    res = []

    def root_ok(fam):
        if live and live["fam"] == fam:
            return True
        return any(e["fam"] == fam for e in parked)

    def park(e):
        nonlocal clock
        if e["L"] < arm.min_tokens:
            return
        e["size"] = snapshot_mib(e["L"], e["ck"])
        immovable = lambda x: arm.pin and x["pinned"]   # noqa: E731  (conversation_cache.hpp: a pinned entry never leaves)
        if arm.keep_all:
            clock += 1
            parked[:] = [x for x in parked if x["sid"] != e["sid"]] + [dict(e, stamp=clock)]
            return
        if e["size"] > arm.budget:
            return
        parked[:] = [x for x in parked if x["sid"] != e["sid"]]
        while sum(x["size"] for x in parked) + e["size"] > arm.budget or len(parked) >= arm.slots:
            movable = [x for x in parked if not immovable(x)]
            if not movable:
                return                                  # only pinned entries left: the new image is not parked
            v = min(movable, key=lambda x: x["stamp"])
            parked.remove(v)
            if arm.disk and v["L"] >= 2048:
                disk[v["sid"]] = dict(L=v["L"], size=v["size"], fam=v["fam"])
                cap = arm.disk_gib * 1024.0
                while sum(d["size"] for d in disk.values()) > cap:
                    disk.popitem(last=False)
        clock += 1
        parked.append(dict(e, stamp=clock))

    for s in steps:
        fam, n = s["agent"], s["total"]
        root = roots.get(fam, 0)
        own, src, restore_s = None, None, 0.0
        if live and live["sid"] == s["sid"]:
            own, src = live, "live"
        else:
            e = next((x for x in parked if x["sid"] == s["sid"]), None)
            if e is not None:
                own, src, restore_s = e, "parked", arm.copy_s * e["size"] / 1229.0
            elif s["sid"] in disk:
                own, src = disk[s["sid"]], "disk"
                restore_s = (disk[s["sid"]]["size"] / 1024.0) / arm.disk_gbps + arm.copy_s
        if s["kind"] == "cont" and own is not None:
            resume = min(own["L"], s["proc_prev"])
        else:                       # no state of its own: the family root when a live or parked state still holds it
            resume = root if (root and root < n and root_ok(fam)) else 0
            src, restore_s = ("root" if resume else "cold"), (arm.copy_s * 0.1 if resume else 0.0)
        # park the outgoing live state, then drop what this request takes
        if live and live["sid"] != s["sid"]:
            park(live)
        if src == "parked":
            parked[:] = [x for x in parked if x["sid"] != s["sid"]]
        pinned = bool(arm.pin and root and n > root)
        live = dict(sid=s["sid"], fam=fam, L=n + s["out"], ck=n_checkpoints(n, root, resume), pinned=pinned)
        s["sim_resume"], s["sim_src"], s["sim_restore_s"] = resume, src, restore_s
        s["sim_read"] = max(0, n - resume)
    # agreement: continuing requests, miss = reused < COLD_FRACTION of the state it could have continued
    agree = tot = 0
    for s in steps:
        if s["kind"] != "cont" or s["proc_prev"] < 8192:
            continue
        rec_miss = s["reused"] < COLD_FRACTION * s["proc_prev"]
        sim_miss = s["sim_resume"] < COLD_FRACTION * s["proc_prev"]
        tot += 1
        agree += rec_miss == sim_miss
    return dict(agree=agree, n=tot)


def arm_report(steps, roots, names):
    base = None
    out = {}
    busy = sum(s["done"] - s["created"] for s in steps) / 1000.0
    for nm in names:
        arm = ARMS[nm]()
        st = [dict(s) for s in steps]
        ag = simulate(st, arm, roots)
        # the sim is judged on the requests it models (continuing, own state known); the rest of each request's read
        # (first, shrink, and recorded partial resumes) is common to every arm and cancels in the deltas
        read = sum(s["sim_read"] for s in st if s["kind"] == "cont")
        rest = sum(s["sim_restore_s"] for s in st if s["kind"] == "cont")
        first = sum(s["sim_read"] for s in st if s["kind"] != "cont")
        out[nm] = dict(read_tokens=read, restore_s=rest, first_shrink_read=first, agree=ag)
        if base is None:
            base = out[nm]
    for nm, o in out.items():
        d_tok = base["read_tokens"] - o["read_tokens"]
        o["saved_s"] = d_tok * MS_PER_TOKEN / 1000.0 - (o["restore_s"] - base["restore_s"])
        o["saved_pct"] = 100.0 * o["saved_s"] / busy if busy else 0.0
    return out


# --------------------------------------------------------------------------------------------------- the chain micro model
def micro(policy, marks, n_final=120000, step_tokens=1500, alpha=1.0, trials=400, seed=1, cold=False):
    """One conversation retained under --prompt-cache 6.  cold=False: grown by agent steps (a tool result of `step_tokens`
    + a ~300-token reply each, one turn checkpoint a request).  cold=True: one request reads the whole history from
    the system prompt's end (a re-read after a switch, a prune or an eviction).  `marks`: the A11 checkpoints at tool
    boundaries (tool_checkpoint_marks: >= 1024 tokens past the deepest held point, >= 64 before the next, at most 2 a
    request).  Returns the mean tokens read again for an edit at a uniformly random place, at a random tool-result
    boundary, and within the last 8K."""
    import random
    ROOT = 11000
    chain = [dict(L=ROOT, used=0)]
    clock, pos, boundaries = 1, ROOT, []
    rng = random.Random(seed)

    def reread(a, b):
        return (b - a) * (1.0 + 0.5 * (a + b) / 32768.0)

    def evict():
        while len(chain) > PROMPT_CACHE:
            newest = max(range(len(chain)), key=lambda k: chain[k]["used"])
            cand = [i for i in range(1, len(chain)) if i != newest]
            if policy == "lru":
                v = min(cand, key=lambda i: chain[i]["used"])
            elif policy == "flops":
                srt = sorted(c["L"] for c in chain)

                def worth(i):
                    return reread(max([x for x in srt if x < chain[i]["L"]] or [0]), chain[i]["L"])
                wmax = max(worth(i) for i in cand) or 1.0
                lo, hi = min(chain[i]["used"] for i in cand), max(chain[i]["used"] for i in cand)

                def score(i):
                    return ((chain[i]["used"] - lo) / (hi - lo) if hi > lo else 1.0) + alpha * worth(i) / wmax
                v = min(cand, key=score)
            else:   # spread: the item whose removal leaves the smallest gap for its distance from the newest
                ls = sorted(chain, key=lambda c: c["L"])
                best, v = None, cand[0]
                for i in cand:
                    k = ls.index(chain[i])
                    if k == 0 or k == len(ls) - 1:
                        continue
                    gap = (ls[k + 1]["L"] - ls[k - 1]["L"]) / max(1, ls[-1]["L"] - ls[k - 1]["L"])
                    if best is None or gap < best:
                        best, v = gap, i
            chain.pop(v)

    def add(L):
        nonlocal clock
        for c in chain:
            if c["L"] == L:
                c["used"] = clock
                clock += 1
                return
        chain.append(dict(L=L, used=clock))
        clock += 1
        evict()

    def request(read_from, to):
        """The checkpoints one request makes while reading [read_from, to): periodic ones, the marks, the turn boundary."""
        for p_ in range(read_from + EVERY, to, EVERY):
            add(p_)
        if marks:
            held = sorted(c["L"] for c in chain) + [to - 3]
            picked = []
            for b in sorted((x for x in boundaries if read_from < x < to - 1), reverse=True):
                below = max([h for h in held + picked if h < b] or [0])
                above = min([h for h in held + picked if h > b] or [to - 1])
                if b - below >= 1024 and above - b >= 64 and len(picked) < min(2, PROMPT_CACHE - 3):
                    picked.append(b)
            for b in sorted(picked):
                add(b)
        add(to - 3)

    def grow(to):
        nonlocal pos
        while pos < to:
            boundaries.append(pos + 300)                  # the tool call ends 300 tokens into the step, the result follows
            pos += step_tokens
            boundaries.append(pos - step_tokens + 300 + 1)

    if cold:
        grow(n_final)
        boundaries = sorted(set(boundaries))
        request(ROOT, pos)
    else:
        prev = ROOT
        while pos < n_final:
            grow(pos + step_tokens)
            request(prev, pos)
            prev = pos
    n = pos

    def cost(d):
        return n - max([c["L"] for c in chain if c["L"] <= d] or [0])
    uni = [cost(rng.randrange(1, n)) for _ in range(trials)]
    tool = [cost(rng.choice(boundaries)) for _ in range(trials)]
    near = [cost(n - rng.randrange(1, 8192)) for _ in range(trials)]
    mean = lambda x: sum(x) / len(x)   # noqa: E731
    return mean(uni), mean(tool), mean(near)


def micro_table(cold):
    rows = []
    for pol, marks, alpha in (("lru", False, 0), ("lru", True, 0), ("flops", False, 1.0), ("flops", True, 1.0),
                              ("flops", True, 2.0), ("spread", False, 0), ("spread", True, 0)):
        u, t, nr = micro(pol, marks, alpha=alpha, cold=cold)
        rows.append((f"{pol}{'' if pol != 'flops' else ' a=%g' % alpha}{' +tool marks' if marks else ''}", u, t, nr))
    return rows


# --------------------------------------------------------------------------------------------------- report
def report(steps, roots, arms, do_micro=True, out=sys.stdout):
    c = classify(steps)
    busy, tot_in = c["busy_s"], c["input"]
    P = lambda *a: print(*a, file=out)   # noqa: E731
    P(f"requests {c['n']} in {c['sessions']} sessions; summed request time {busy:,.0f} s; tokens read {tot_in:,}")
    ex = sum(c["reread"].values())
    P(f"re-read (read beyond new content; first requests counted whole): {ex:,} tokens = {ex * MS_PER_TOKEN / 1000:,.0f} s = "
      f"{100 * ex * MS_PER_TOKEN / 1000 / busy:.1f}% of request time")
    P(f"{'cause':44s} {'steps':>5s} {'tokens':>11s} {'s':>8s} {'% busy':>7s}")
    for k, v in sorted(c["reread"].items(), key=lambda kv: -kv[1]):
        sec = v * MS_PER_TOKEN / 1000
        P(f"{k:44s} {c['steps'][k]:5d} {v:11,d} {sec:8.0f} {100 * sec / busy:6.1f}%")
    P("family shared prefixes (estimated):", {k: v for k, v in sorted(roots.items())})
    res = arm_report(steps, roots, arms)
    P(f"\npark model (continuing requests only; base reads {res[arms[0]]['read_tokens']:,} tokens)")
    P(f"{'arm':12s} {'read tokens':>12s} {'restore s':>9s} {'saved s':>8s} {'% busy':>7s}  agreement with the recorded hit/miss")
    for nm in arms:
        o = res[nm]
        P(f"{nm:12s} {o['read_tokens']:12,d} {o['restore_s']:9.0f} {o['saved_s']:8.0f} {o['saved_pct']:6.2f}%  "
          f"{o['agree']['agree']}/{o['agree']['n']}")
    if do_micro:
        for cold, what in ((False, "grown by agent steps (a turn checkpoint each request)"),
                           (True, "after ONE cold read of the whole history")):
            P(f"\nchain micro model, 120K conversation, --prompt-cache 6, {what}: tokens read again per edit")
            P(f"{'policy':26s} {'random place':>13s} {'tool boundary':>14s} {'last 8K':>9s}")
            for nm, u, t, nr in micro_table(cold):
                P(f"{nm:26s} {u:13,.0f} {t:14,.0f} {nr:9,.0f}")
    return dict(classify=c, arms=res)


# --------------------------------------------------------------------------------------------------- self test
def _mk(sid, agent, t, total, reused, out, sub=False, done_after=5000):
    return dict(sid=sid, agent=agent, sub=sub, created=t, done=t + done_after, total=total, reused=reused,
                input=total - reused, out=out)


def self_test():
    ev = dict(users=collections.defaultdict(list), comp=collections.defaultdict(list), prune=collections.defaultdict(list))
    # session A (agent a): first, two continuing; session B (agent b) runs between A's second and third request
    steps = [_mk("A", "a", 0, 20000, 0, 300), _mk("A", "a", 10_000, 20800, 20299, 300), _mk("B", "b", 20_000, 9000, 0, 200),
             _mk("A", "a", 400_000, 21600, 0, 300)]
    annotate(steps, ev)
    kinds = [s["kind"] for s in steps]
    assert kinds == ["first", "cont", "first", "cont"], kinds
    assert steps[1]["excess"] <= 1 and steps[3]["excess"] == 21100, steps[3]
    assert steps[3]["cause"] == "switch (another session ran between)", steps[3]["cause"]
    roots = family_roots(steps)
    assert roots["a"] == int(0.97 * 20000), roots
    # base parks A (20.8K tokens is ~0.5 GiB: fits 1280 MiB), so A's third request is a hit
    st = [dict(s) for s in steps]
    simulate(st, ARMS["base"](), roots)
    assert st[3]["sim_src"] == "parked" and st[3]["sim_resume"] == st[3]["proc_prev"] == 21100, st[3]
    res = arm_report(steps, roots, ["base", "keep-all"])
    assert res["keep-all"]["saved_s"] == 0.0, res        # base already keeps it: nothing left to save
    # a snapshot over the budget is not parked: the same request is then a cold read, the 2048 MiB arm keeps it
    big = [_mk("A", "a", 0, 60000, 0, 300), _mk("B", "b", 10_000, 9000, 0, 200), _mk("A", "a", 20_000, 61000, 0, 300)]
    annotate(big, ev)
    st = [dict(s) for s in big]
    simulate(st, ARMS["base"](), family_roots(big))
    assert st[2]["sim_src"] in ("cold", "root"), st[2]
    st = [dict(s) for s in big]
    simulate(st, ARMS["flags"](), family_roots(big))
    assert st[2]["sim_src"] == "parked" and st[2]["sim_resume"] == 60300, st[2]
    # a pinned entry is immovable: with the pin arm and slots exhausted by pinned entries, a new image is not parked
    pin = [_mk("A", "a", 0, 20000, 0, 100), _mk("B", "b", 1, 20000, 0, 100), _mk("C", "c", 2, 20000, 0, 100),
           _mk("D", "d", 3, 20000, 0, 100), _mk("E", "e", 4, 20000, 0, 100), _mk("F", "f", 5, 20000, 0, 100),
           _mk("A", "a", 6, 20200, 0, 100)]
    annotate(pin, ev)
    for nm, want in (("base", "cold"), ("pin", "parked")):
        st = [dict(s) for s in pin]
        simulate(st, ARMS[nm](), family_roots(pin))
        # base: oldest-first leaves A out of 4 slots by F -> not parked; pin: A (first parked, pinned) stays, F is skipped
        assert (st[-1]["sim_src"] in ("cold", "root")) == (want == "cold"), (nm, st[-1]["sim_src"])
    # the chain micro model: in steady growth the turn checkpoint is always within 1024 tokens of any tool boundary, so the
    # marks never qualify and change nothing; after one cold read they help edits near the end and cost the older ones
    s0, s1 = micro("lru", False, trials=200), micro("lru", True, trials=200)
    assert s0 == s1, (s0, s1)
    c0, c1 = micro("lru", False, trials=200, cold=True), micro("lru", True, trials=200, cold=True)
    assert c1[2] <= c0[2] and c1[0] >= c0[0], (c0, c1)
    assert micro("spread", False, trials=200)[0] < micro("lru", False, trials=200)[0]
    print("percache_sim self-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--model", default="qwen3.8-flash-next")
    ap.add_argument("--arms", default="base,flags,flags+disk,pin,pin+flags,keep-all")
    ap.add_argument("--json", help="write the numbers here")
    ap.add_argument("--no-micro", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    steps, ev = load_steps(a.db, a.model)
    if not steps:
        print("no requests of that model in the store", file=sys.stderr)
        return 2
    annotate(steps, ev)
    arms = a.arms.split(",")
    for nm in arms:
        if nm not in ARMS:
            print(f"unknown arm {nm}; known: {', '.join(ARMS)}", file=sys.stderr)
            return 2
    roots = family_roots(steps)
    out = report(steps, roots, arms, do_micro=not a.no_micro)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
