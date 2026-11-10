#!/usr/bin/env python3
"""Read ONLY the header (metadata + tensor-info table) of GGUF files; never touches tensor data.
Prints bytes per tensor class (expert gate/up/down, shared expert, attention, ...) and per quant type.

    python gguf_table.py FILE.gguf [FILE2.gguf ...]  [--json out.json]
"""
import struct, sys, json, re, collections

# ggml type id -> (name, block elems, block bytes)
TYPES = {0: ('F32', 1, 4), 1: ('F16', 1, 2), 2: ('Q4_0', 32, 18), 3: ('Q4_1', 32, 20), 6: ('Q5_0', 32, 22), 7: ('Q5_1', 32, 24),
         8: ('Q8_0', 32, 34), 9: ('Q8_1', 32, 36), 10: ('Q2_K', 256, 84), 11: ('Q3_K', 256, 110), 12: ('Q4_K', 256, 144),
         13: ('Q5_K', 256, 176), 14: ('Q6_K', 256, 210), 15: ('Q8_K', 256, 292), 16: ('IQ2_XXS', 256, 66), 17: ('IQ2_XS', 256, 74),
         18: ('IQ3_XXS', 256, 98), 19: ('IQ1_S', 256, 50), 20: ('IQ4_NL', 32, 18), 21: ('IQ3_S', 256, 110), 22: ('IQ2_S', 256, 82),
         23: ('IQ4_XS', 256, 136), 24: ('I8', 1, 1), 25: ('I16', 1, 2), 26: ('I32', 1, 4), 27: ('I64', 1, 8), 28: ('F64', 1, 8),
         29: ('IQ1_M', 256, 56), 30: ('BF16', 1, 2), 34: ('TQ1_0', 256, 54), 35: ('TQ2_0', 256, 66), 39: ('MXFP4', 32, 17), 42: ('Q2_0', 64, 18)}
SCALAR = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}


class R:
    def __init__(self, f): self.f = f
    def u(self, fmt):
        n = struct.calcsize('<' + fmt)
        return struct.unpack('<' + fmt, self.f.read(n))
    def s(self):
        n, = self.u('Q'); return self.f.read(n).decode('utf8', 'replace')


def skip_value(r, t):
    if t in SCALAR: r.u(SCALAR[t]); return
    if t == 8: r.s(); return
    if t == 9:
        et, = r.u('I'); n, = r.u('Q')
        for _ in range(n): skip_value(r, et)
        return
    raise ValueError('bad kv type %d' % t)


def read(path):
    with open(path, 'rb', buffering=1 << 20) as f:
        r = R(f)
        assert f.read(4) == b'GGUF'
        ver, = r.u('I'); nt, = r.u('Q'); nkv, = r.u('Q')
        kv = {}
        for _ in range(nkv):
            k = r.s(); t, = r.u('I')
            if t in SCALAR or t == 8:
                v = r.s() if t == 8 else r.u(SCALAR[t])[0]
                if k.endswith(('expert_count', 'expert_used_count', 'block_count', 'embedding_length', 'expert_feed_forward_length',
                               'expert_shared_feed_forward_length', 'architecture', 'name')) or 'file_type' in k: kv[k] = v
            else:
                skip_value(r, t)
        tens = []
        for _ in range(nt):
            name = r.s(); nd, = r.u('I'); dims = r.u('%dQ' % nd); ty, = r.u('I'); off, = r.u('Q')
            tens.append((name, dims, ty, off))
        return ver, kv, tens


def nbytes(dims, ty):
    nm, be, bb = TYPES[ty]
    n = 1
    for d in dims: n *= d
    return n // be * bb


def klass(name):
    if re.search(r'ffn_(gate|up|down)_exps', name): return 'expert_' + re.search(r'(gate|up|down)', name)[1]
    if re.search(r'ffn_(gate|up|down)_shexp', name): return 'shared_expert'
    if re.search(r'ffn_gate_inp', name): return 'router'
    if re.search(r'attn|ssm|delta|gdn', name): return 'attention/ssm'
    if 'token_embd' in name or name.startswith('output'): return 'embed/output'
    return 'other'


if __name__ == '__main__':
    out = {}
    jp = sys.argv[sys.argv.index('--json') + 1] if '--json' in sys.argv else None
    for p in [a for a in sys.argv[1:] if not a.startswith('--') and a != jp]:
        ver, kv, tens = read(p)
        print('==', p.rsplit('/', 1)[-1], 'gguf v%d' % ver, len(tens), 'tensors')
        print('  kv:', {k: v for k, v in kv.items()})
        by = collections.defaultdict(lambda: [0, 0, collections.Counter()])
        for name, dims, ty, off in tens:
            c = klass(name); b = nbytes(dims, ty)
            by[c][0] += 1; by[c][1] += b; by[c][2][TYPES[ty][0]] += 1
        for c, (n, b, ty) in sorted(by.items(), key=lambda x: -x[1][1]):
            print('  %-16s %5d tensors %9.2f GiB  types %s' % (c, n, b / 2 ** 30, dict(ty)))
        out[p] = {c: dict(n=n, bytes=b, types=dict(ty)) for c, (n, b, ty) in by.items()}
        layers = collections.defaultdict(dict)
        for name, dims, ty, off in tens:
            m = re.match(r'blk\.(\d+)\.ffn_(gate|up|down)_exps', name)
            if m: layers[int(m[1])][m[2]] = (TYPES[ty][0], nbytes(dims, ty) // dims[-1])   # bytes per expert
        out[p]['per_layer_expert'] = {str(k): v for k, v in sorted(layers.items())}
        # one expert-layer example
        ex = [t for t in tens if re.search(r'blk\.\d+\.ffn_(gate|up|down)_exps', t[0])][:3]
        for name, dims, ty, off in ex:
            print('  e.g.', name, dims, TYPES[ty][0], '%.1f MiB' % (nbytes(dims, ty) / 2 ** 20))
    if jp:
        json.dump(out, open(jp, 'w'), indent=1)
