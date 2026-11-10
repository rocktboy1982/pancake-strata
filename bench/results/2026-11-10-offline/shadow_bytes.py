#!/usr/bin/env python3
"""4.0.5: CPU expert bytes per decoded token, now and with a shadow pack at lower bit width.

Bytes per expert per layer come from the GGUF tensor table (gguf_table.json, header only).  Which experts the
CPU reads is the replay of the --dump-routing trace through the engine's adaptive tier (tools/replay_cache_policy.py,
10909 resident slots): every distinct (layer, expert) a window routes that is not resident is computed by the CPU
(the same count tail_rows.py / time_ceiling.py use).  A shadow tensor is used only where it is smaller than the
tensor the layer already has (min), as a real shadow pack would be built.

    python shadow_bytes.py ROUTING.bin PROFILE.bin GGUF_TABLE.json
"""
import sys, json
import numpy as np
sys.path.insert(0, '/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-a2cec809b8de0e01d/tools')
import replay_cache_policy as rp

trace, profile, table = sys.argv[1:4]
tab = json.load(open(table))
lay = list(tab.values())[0]['per_layer_expert']
W = 2560 * 640                                  # weights per expert per tensor (gate, up, down all 2560 x 640)
BPW = {'IQ1_M': 56 / 256, 'IQ2_XXS': 66 / 256, 'IQ2_XS': 74 / 256, 'IQ2_S': 82 / 256, 'Q2_0': 18 / 64}   # bytes per weight
def shadow_bytes(kind):                         # bytes per expert per layer, per tensor, under a shadow variant
    out = {}
    for l, d in lay.items():
        t = {}
        for part in ('gate', 'up', 'down'):
            cur = d[part][1]
            sh = {'all-IQ1_M': W * BPW['IQ1_M'], 'gateup-IQ1_M': W * BPW['IQ1_M'] if part != 'down' else cur,
                  'all-IQ2_XXS': W * BPW['IQ2_XXS'], 'gateup-IQ2_XXS': W * BPW['IQ2_XXS'] if part != 'down' else cur,
                  'cur': cur}[kind]
            t[part] = min(cur, sh)
        out[int(l)] = sum(t.values())
    return out
variants = ['cur', 'gateup-IQ2_XXS', 'all-IQ2_XXS', 'gateup-IQ1_M', 'all-IQ1_M']
per_layer_bytes = {v: shadow_bytes(v) for v in variants}
print('bytes per expert (gate+up+down), mean over the 48 layers: ' +
      ', '.join('%s %.0f KB' % (v, np.mean(list(per_layer_bytes[v].values())) / 1000) for v in variants))

windows, _k = rp.load_trace(trace)
init = rp.load_profile(profile, 10909)
pol = rp.Policy()
res = np.zeros((rp.NL, rp.NE), bool)
res[init[:, 0], init[:, 1]] = True
usage = np.zeros((rp.NL, rp.NE))
miss_layer = []                                   # per window: distinct misses per layer
for wi, w in enumerate(windows):
    m = np.zeros(rp.NL)
    for l, ids in w:
        miss = ~res[l, ids]
        m[l] = len(np.unique(ids[miss]))
        np.add.at(usage[l], ids.ravel(), 1.0)
    miss_layer.append(m)
    if (wi + 1) % pol.every == 0:
        rp.adapt_round(res, usage, pol)
        usage *= pol.decay
miss_layer = np.array(miss_layer)
segs = [('read4k', 0, 11, 53), ('short chat', 11, 96, 321), ('code edit', 96, 157, 553), ('long', 157, 219, 169), ('ALL', 0, 219, 53 + 321 + 553 + 169)]
print('%-11s %8s %9s | ' % ('request', 'miss/win', 'tok/win') + ' | '.join('%-14s' % v for v in variants))
res_tab = {}
for name, a, b, toks in segs:
    m = miss_layer[a:b]
    n = b - a
    row = []
    base = None
    for v in variants:
        by = sum(m[:, l].sum() * per_layer_bytes[v][l] for l in range(rp.NL))
        mb_tok = by / toks / 1e6
        if base is None: base = by
        row.append((mb_tok, 100 * (1 - by / base)))
    res_tab[name] = row
    print('%-11s %8.1f %9.2f | ' % (name, m.sum() / n, toks / n) + ' | '.join('%6.1f MB %+5.1f%%' % (r[0], -r[1]) for r in row))
json.dump({k: [list(x) for x in v] for k, v in res_tab.items()},
          open('/home/dan/clone-of-d/src/Strata/.claude/worktrees/agent-a2cec809b8de0e01d/bench/results/2026-11-10-offline/shadow_bytes.json', 'w'), indent=1)
