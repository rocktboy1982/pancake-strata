"""turn_thinking.py - A17: the thinking tokens and thinking time per agent turn, from the per-turn log (item 73).

    python tools/pancake/turn_thinking.py [pancake-turns.jsonl] [--max-new 32768]

Agent turns = the rows whose reply cap is opencode's (max_new 32768); the benchmark rows (max_new 512) and ad-hoc tests are
left out.  The log has no effort-level field (see bench/results/2026-11-10-offline/A6_A17.md for the one-line change), so the
distribution is for whatever level the client sent.  Thinking time = decode_ms * reasoning_tokens / engine_generated (the
same speed for thinking and answering tokens).  Cap scenarios: thinking above cap + reasoning_budget_grace (512) is cut.
"""
import json
import sys
import time

import numpy as np

path = next((x for x in sys.argv[1:] if not x.startswith("--")), "/home/dan/clone-of-d/src/Strata/pancake-turns.jsonl")
MAX_NEW = int(sys.argv[sys.argv.index("--max-new") + 1]) if "--max-new" in sys.argv else 32768
rows = [json.loads(l) for l in open(path)]
print('rows', len(rows))
ag = [r for r in rows if r['max_new'] == MAX_NEW]
print('agent turns (max_new 32768):', len(ag), time.strftime('%F %H:%M', time.localtime(ag[0]['time'])), '->', time.strftime('%F %H:%M', time.localtime(ag[-1]['time'])))


def q(x, p):
    return float(np.percentile(x, p))


def stats(name, x):
    x = np.array(x, float)
    print(f'{name:34s} n={len(x):3d} mean {x.mean():8.1f} p50 {q(x,50):8.1f} p90 {q(x,90):8.1f} p99 {q(x,99):8.1f} max {x.max():8.1f}')


stats('reasoning tokens / turn (all)', [r['reasoning_tokens'] for r in ag])
th = [r for r in ag if r['reasoning_tokens'] > 0]
stats('reasoning tokens (turns that think)', [r['reasoning_tokens'] for r in th])
print('turns that think', len(th), 'of', len(ag))
stats('output tokens / turn', [r['output_tokens'] for r in ag])
stats('turn duration s', [r['duration_s'] for r in ag])
# time in thinking: decode time of the turn * share of generated tokens that are thinking
def think_s(r):
    g = max(1, r['engine_generated'])
    return r['decode_ms'] / 1e3 * min(1.0, r['reasoning_tokens'] / g)
ts = [think_s(r) for r in ag]
stats('thinking s / turn (est)', ts)
tot = sum(r['duration_s'] for r in ag)
print(f'thinking time {sum(ts):.1f} s of {tot:.1f} s total turn time = {sum(ts)/tot*100:.1f}%')
dec = [r['decode_tok_s'] for r in ag if r['decode_ms'] > 2000]
stats('decode tok/s (turns > 2 s decode)', dec)
print('finish', {f: sum(1 for r in ag if r['finish'] == f) for f in set(r['finish'] for r in ag)})
print('top thinking turns: reasoning, gen, decode_ms, dur, finish, prompt, reused')
for r in sorted(ag, key=lambda r: -r['reasoning_tokens'])[:8]:
    print(r['reasoning_tokens'], r['engine_generated'], r['decode_ms'], r['duration_s'], r['finish'], r['prompt_tokens'], r['reused'], r.get('rumination_max'))
# cap scenarios: cap C tokens (+512 grace): thinking beyond C+grace is removed
for grace in (512,):
    print('--- cap scenarios (thinking tokens above cap+grace are cut; time saved at the turn\'s own decode speed)')
    for cap in (512, 1024, 1500, 2048, 3072, 4096, 8192):
        saved_t = saved_tok = n = 0
        for r in ag:
            over = r['reasoning_tokens'] - (cap + grace)
            if over > 0:
                n += 1
                saved_tok += over
                saved_t += over / max(1, r['engine_generated']) * r['decode_ms'] / 1e3
        print(f'cap {cap:5d}: turns hit {n:2d}, tokens cut {saved_tok:6d}, time saved {saved_t:6.1f} s = {saved_t/tot*100:4.1f}% of turn time, {saved_t/len(ag):.2f} s/turn')
# every thinking-capable row: 1000-token ad hoc tests excluded
other = [r for r in rows if r['max_new'] not in (512, 32768)]
print('other rows', len(other), 'reasoning tokens max', max(r['reasoning_tokens'] for r in other))
print('benchmark rows (max_new 512):', sum(1 for r in rows if r['max_new'] == 512), 'with reasoning', sum(1 for r in rows if r['max_new'] == 512 and r['reasoning_tokens'] > 0))
# time-ordered distribution of the thinking tokens (first and second session?)
print('reasoning per turn in order:', [r['reasoning_tokens'] for r in ag])
print('duration per turn in order:', [r['duration_s'] for r in ag])
print('prompt-read per turn (prompt_ms s):', [round(r['prompt_ms'] / 1e3, 1) for r in ag])
