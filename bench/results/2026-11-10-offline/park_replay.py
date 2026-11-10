#!/usr/bin/env python3
"""4.0.1: replay of the conversation cache (budget, slots, oldest-first eviction) on the real request
sequence of the 2026-11-09 16:13 engine run, under several park policies.  Pure arithmetic, no GPU.

Inputs (read only): the engine log segment of that run (--log, --first-line/--last-line) and nothing else.
Model of the snapshot size, fitted on the log's own park / skip lines (see RESULTS.md):
    size_MiB(tokens, ckpts) = 111 + 14.9/1024 * tokens + 113 * ckpts
Periodic checkpoints sit every 16384 tokens from where a request resumed (--prompt-cache-every default).
"""
import argparse, re, sys, json

ap = argparse.ArgumentParser()
ap.add_argument('--log', default='/home/dan/clone-of-d/src/Strata/pancake-flashnext.log')
ap.add_argument('--first-line', type=int, default=16769)
ap.add_argument('--last-line', type=int, default=17630)
ap.add_argument('--ms-per-tok', type=float, default=0.41, help='cold prefill cost of a long prompt (log: 0.39-0.42)')
ap.add_argument('--restore-s', type=float, default=0.9, help='time to copy a ~1.2 GiB parked snapshot back')
a = ap.parse_args()

lines = open(a.log, errors='replace').read().split('\n')[a.first_line - 1:a.last_line]
reqs = []
for l in lines:
    m = re.search(r'prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms.* (\d+) generated in (\d+) ms.* (\d+) checkpoints', l)
    if m:
        reqs.append(dict(tot=int(m[1]), reuse=int(m[2]), read=int(m[3]), ms=int(m[4]), gen=int(m[5]), ck=int(m[7])))
# conversation A = the first 3 long requests (70-72K) ends before the 34K one; B = every later request >= 34K
seq = []
for r in reqs:
    if r['tot'] >= 30000:
        r['conv'] = 'A' if r['tot'] >= 70000 else 'B'
    else:
        r['conv'] = 's%d' % r['tot']
    seq.append(r)
# conversation A: only its first turn matters for the replay; keep the whole sequence so the cache starts like the log

def size_mib(tok, ck):
    return 111 + 14.9 / 1024 * tok + 113 * ck

def run(budget=1280, slots=4, truncate=False, min_tokens=0, tail_gap=None):
    cache = {}          # conv -> dict(tok, ck, pos, size, t)
    live = None         # (conv, tok_after, ck, positions)
    agent_s = 0.0; cold = []; log = []
    t = 0
    for r in seq:
        t += 1
        conv = r['conv']; agent = conv in ('A', 'B')
        tot = r['tot']
        # the request's own parked copy is taken out first (generate.cpp: `incoming`), then the live session is parked
        # with that copy's bytes held: it must fit beside it, else it is not parked (the incoming copy is never evicted)
        incoming = cache.pop(conv, None) if not (live and live[0] == conv) else None
        held = incoming['size'] if incoming else 0
        if live and live[0] != conv:
            pc, ptok, pck, ppos = live
            if ptok >= min_tokens:
                room = budget - held
                size = size_mib(ptok, pck)
                keep_tok, keep_ck, keep_pos = ptok, pck, ppos
                if size > room and truncate:
                    # deepest checkpoint prefix that fits (the tail checkpoints are dropped)
                    cand = sorted(set(list(ppos) + ([ptok - tail_gap] if tail_gap and ptok - tail_gap > 0 else [])))
                    for k in range(len(cand), 0, -1):
                        c = cand[k - 1]
                        s_ = size_mib(c, k)
                        if s_ <= room:
                            keep_tok, keep_ck, keep_pos, size = c, k, cand[:k], s_
                            break
                if size <= room:
                    cache.pop(pc, None)
                    while cache and (sum(e['size'] for e in cache.values()) + size > room or len(cache) + (1 if incoming else 0) >= slots):
                        victim = min(cache, key=lambda k: cache[k]['t'])
                        del cache[victim]
                    cache[pc] = dict(tok=keep_tok, ck=keep_ck, pos=keep_pos, size=size, t=t)
                # else: skipped (too big)
            live = None
        # serve the request
        if live and live[0] == conv:                       # warm from the live session
            read = tot - live[1]
            read = max(read, 0)
            cost = read * a.ms_per_tok / 1000
            pos = live[3]; ck = min(6, live[2] + 1)
        elif incoming:                                    # restore the parked copy taken above
            e = incoming
            p = min(e['tok'], tot)
            read = tot - p
            cost = a.restore_s * (e['size'] / 1257) + read * a.ms_per_tok / 1000
            pos = list(e['pos']) + [p + 16384 * i for i in range(1, read // 16384 + 1)]
            ck = min(6, e['ck'] + 1)
        else:                                             # cold
            read = tot
            cost = read * a.ms_per_tok / 1000
            pos = [16384 * i for i in range(1, tot // 16384 + 1)]
            ck = max(1, len(pos)) if agent else 1
            if agent and conv == 'B':
                cold.append((tot, round(cost, 1)))
        if agent and conv == 'B':
            agent_s += cost
        log.append((conv, tot, read, round(cost, 1)))
        live = (conv, tot + r['gen'], ck, pos)
    return agent_s, cold, log

if __name__ == '__main__':
    # actual cost of conversation B in the log
    act = sum(r['ms'] for r in seq if r['conv'] == 'B') / 1000
    act_cold = [(r['tot'], round(r['ms'] / 1000, 1)) for r in seq if r['conv'] == 'B' and r['reuse'] == 0]
    print('conv B: %d requests, actual prompt time %.1f s, cold re-reads %d (%s)' % (
        sum(1 for r in seq if r['conv'] == 'B'), act, len(act_cold), act_cold))
    base = None
    out = {}
    for name, kw in [('S0 baseline (1280 MiB, 4 slots)', {}),
                     ('S1 PARK_TRUNCATE only', dict(truncate=True)),
                     ('S2 --conversation-cache-min-tokens 8192 only', dict(min_tokens=8192)),
                     ('S3 PARK_TRUNCATE + min-tokens 8192', dict(truncate=True, min_tokens=8192)),
                     ('S1b PARK_TRUNCATE only, a tail checkpoint 2K tokens before the end exists', dict(truncate=True, tail_gap=2048)),
                     ('S3b PARK_TRUNCATE + min-tokens 8192, tail checkpoint 2K before the end', dict(truncate=True, min_tokens=8192, tail_gap=2048)),
                     ('S4 budget 2048 MiB only', dict(budget=2048)),
                     ('S5 budget 2048 MiB + min-tokens 8192', dict(budget=2048, min_tokens=8192)),
                     ('S6 budget 1600 MiB + min-tokens 8192', dict(budget=1600, min_tokens=8192))]:
        s, cold, log = run(**kw)
        if base is None: base = s
        print('%-48s conv-B prompt time %6.1f s  cold re-reads %d  saved vs S0 %6.1f s' % (name, s, len(cold), base - s))
        out[name] = dict(prompt_s=round(s, 1), cold=len(cold), saved_s=round(base - s, 1))
    json.dump(out, open('/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-a2cec809b8de0e01d/bench/results/2026-11-10-offline/park_replay.json', 'w'), indent=1)
