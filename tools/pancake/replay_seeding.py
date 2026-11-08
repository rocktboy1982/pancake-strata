"""replay_seeding.py - pancake-strata: does seeding the adaptive tier at a request's start pay off?  (MOD_PLAN phase 11,
"Left, ranked" items 3 and 4.)

Replays a --dump-routing trace through the engine's adaptive tier (tools/replay_cache_policy.py: usage += 1 per routed
entry, a round every window, best candidates against worst residents, `--swaps` per round, usage *= decay) with the
trace cut into requests, and at a request's start adds a SEED to the tier's usage counts, the way the engine does with
STRATA_DOMAIN_PROFILES (item 4) and STRATA_CONV_WORKSET (item 3) (include/strata/program/expert_seed.hpp):

  domain   the seed is a per-domain profile's ranking (tools/pancake/learn_domain_profiles.py writes them; here the
           profile is learned from the trace's OTHER requests of the same domain, leave-one-out), turned into scores
           S * (1 - rank / M) for the top M pairs (M = the cache's slots)
  conv     a conversation resumes: its first part ran, another request ran, then its second part runs seeded from the
           first part's routing counts (scaled to T tokens' worth)
  oracle   the request's own counts as the seed: the bound of any working-set restore

Seeding modes (the engine's STRATA_SEED_HOLD): hold=0 adds the seed once (then it decays with the counts); hold=R keeps
usage >= seed for the first R rounds (max), then lets it decay.

    python tools/pancake/replay_seeding.py TRACE.bin PROFILE.bin SLOTS --segments 11:code,96:prose,157:code,219:code
                                           [--swaps 32 --every 1 --decay 0.7]

Segments: END:DOMAIN for each request in order (window indices, exclusive ends).  For bench/prefetch/routing.bin (the
trace-bench run: read4k, short, edit, long) the cuts are 11, 96, 157, 219 - where consecutive windows' expert sets stop
overlapping (layer-mean Jaccard 0.04-0.09 against 0.27 on average); `--auto-segments` prints those cuts.

Other modes:
    --split-out DIR            each segment as its own trace (input for learn_domain_profiles.py)
    --domain-dir DIR [--scale S --hold R]   seed every segment from DIR/<domain>.bin, as the engine reads the files
    --requests FILE [--domain-dir DIR]      a conv_trace.py run: the trace cut at each request's end record and replayed
                               under the engine's own rules (no seed on a continuing turn, kept conversation counts on a
                               resumed one, else the domain profile): plain / conv / domain / both

Results (2026-11-08, bench/prefetch/routing.bin, data/expert-profile-pancake.bin, 10,909 slots, 32 swaps every window,
decay 0.7; share of routed entries from VRAM, per request read4k / short / edit / long):
    plain tier                               81.17  | 83.1 85.6 77.3 84.7
    domain, leave-one-out (code only), S2 R16 82.86 | 83.3 85.7 78.8 90.7
    domain files (prose in-sample), S2 R16   82.82  | 84.2 87.6 77.1 92.1   (the held-out edit loses 0.2 after a prose seed)
    domain, own counts (oracle), S16 R16     84.03  | 88.3 88.6 79.8 87.8
    conversation resumed (second half after another request, counts x32, added once): edit 79.9 -> 80.6 (oracle 81.3),
        long 90.5 -> 91.4 (93.2), short 86.2 -> 87.0 (88.4); --requests with long as read4k's next turn: 84.7 -> 85.7
    at 160 swaps a window: plain 85.03, domain S2 R16 85.50
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay_cache_policy as rp  # noqa: E402

NL, NE = rp.NL, rp.NE


def counts_of(windows):
    c = np.zeros((NL, NE))
    for w in windows:
        for l, ids in w:
            np.add.at(c[l], ids.ravel(), 1.0)
    return c


def tokens_of(windows):
    return sum(len(w[0][1]) for w in windows)


def rank_table(counts, base_pairs=None):
    """Global rank of every (layer, expert): by counts descending (make_profile_mix.py's order), then the base profile's
    order for the pairs the counts never saw - the same table learn_domain_profiles.py writes."""
    flat = counts.ravel()
    seen = np.flatnonzero(flat > 0)
    order = list(seen[np.argsort(-flat[seen], kind="stable")])
    have = set(order)
    if base_pairs is not None:
        for l, e in base_pairs:
            i = int(l) * NE + int(e)
            if i not in have:
                order.append(i)
                have.add(i)
    for i in range(NL * NE):
        if i not in have:
            order.append(i)
    rank = np.empty(NL * NE, np.int64)
    rank[np.asarray(order)] = np.arange(NL * NE)
    return rank.reshape(NL, NE)


def seed_from_rank(rank, top, scale):
    """The engine's domain seed (expert_seed.hpp DomainProfiles): scale * (1 - rank / top) for rank < top, else 0."""
    return np.where(rank < top, scale * (1.0 - rank / float(top)), 0.0)


def seed_from_counts(counts, n_tokens, tokens_worth):
    """The engine's conversation seed (expert_seed.hpp WorkingSets): the counts scaled to `tokens_worth` tokens."""
    return counts * (tokens_worth / max(1, n_tokens))


def simulate(segments, init, pol, seeds, hold):
    """segments: list of window lists, in order; seeds: per segment None, an (NL, NE) seed (held `hold` rounds), or a
    (seed, hold) pair.  Returns per-segment (hits, entries)."""
    res = np.zeros((NL, NE), bool)
    res[init[:, 0], init[:, 1]] = True
    usage = np.zeros((NL, NE))
    out = []
    hold0 = hold
    for seg, seed in zip(segments, seeds):
        hold = hold0
        if isinstance(seed, tuple):
            seed, hold = seed
        if seed is not None:
            usage += seed if hold == 0 else 0.0
        h = e = 0
        for wi, w in enumerate(seg):
            for l, ids in w:
                r = res[l, ids]
                h += int(r.sum())
                e += r.size
                np.add.at(usage[l], ids.ravel(), 1.0)
            if seed is not None and hold > 0 and wi < hold:
                np.maximum(usage, seed, out=usage)
            if (wi + 1) % pol.every == 0:
                rp.adapt_round(res, usage, pol)
                usage *= pol.decay
        out.append((h, e))
    return out


def auto_segments(windows, cut=0.12):
    prev, cuts = None, []
    for i, w in enumerate(windows):
        s = [set(ids.ravel().tolist()) for _, ids in w]
        if prev is not None:
            j = np.mean([len(a & b) / max(1, len(a | b)) for a, b in zip(s, prev)])
            if j < cut:
                cuts.append((i, round(float(j), 3)))
        prev = s
    return cuts


def window_records(trace):
    """(records, the first record of every complete window load_trace keeps)."""
    raw = np.fromfile(trace, dtype=np.int32)
    k = int(raw[1])
    rec = 2 + 2 * k
    raw = raw[: (len(raw) // rec) * rec].reshape(-1, rec)
    layer = raw[:, 0]
    # a window starts at a record whose layer does not rise over the previous layer group's
    starts = [i for i in range(len(layer)) if i == 0 or (layer[i] != layer[i - 1] and layer[i] <= layer[i - 1])]
    complete = [s for j, s in enumerate(starts)
                if len(set(layer[s:(starts[j + 1] if j + 1 < len(starts) else len(layer))].tolist())) == NL]
    return raw, complete


def split_out(trace, segments, outdir):
    """The trace's records cut at the segments' window boundaries (the same windows load_trace finds; incomplete
    windows go with the request they fall in)."""
    raw, complete = window_records(trace)
    layer = raw[:, 0]
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    s0 = 0
    for i, spec in enumerate(segments.split(",")):
        end, dom = spec.split(":")
        end = int(end)
        r0 = complete[s0] if s0 < len(complete) else len(layer)
        r1 = complete[end] if end < len(complete) else len(layer)
        p = out / f"{i}-{dom}.trace"
        raw[r0:r1].tofile(p)
        print(f"{p}: windows {s0}-{end}, {r1 - r0} records")
        s0 = end


def domain_seeds(domain_dir, slots, scale):
    """{domain: seed} from DIR/<domain>.bin, as the engine builds them (expert_seed.hpp add_domain)."""
    out = {}
    for f in sorted(Path(domain_dir).glob("*.bin")):
        pairs = rp.load_profile(str(f), NL * NE)
        rank = np.full((NL, NE), NL * NE, np.int64)
        rank[pairs[:, 0], pairs[:, 1]] = np.arange(len(pairs))
        out[f.stem] = seed_from_rank(rank, slots, scale)
    return out


def engine_seeds(reqs, segs, k, conv_slots, conv_tokens, dseeds, dhold):
    """Per request the seed the engine's ExpertSeeder gives it (expert_seed.hpp begin_request): none when the request
    continues the previous one's conversation; the kept counts of its conversation (STRATA_CONV_WORKSET); else its
    domain's profile (STRATA_DOMAIN_PROFILES); else none."""
    kept, order, prev, cur, out = {}, [], None, None, []
    for r, seg in zip(reqs, segs):
        conv = r["conv"]
        seed = None
        if conv == prev:
            pass
        else:
            if prev is not None and conv_slots > 0 and cur is not None and cur.sum() > 0:
                kept[prev] = cur
                if prev in order:
                    order.remove(prev)
                order.append(prev)
                while len(order) > conv_slots:
                    kept.pop(order.pop(0))
            if conv_slots > 0 and conv in kept:
                c = kept[conv]
                seed = (c * (conv_tokens * k * NL / c.sum()), 0)
                cur = c.copy()
            else:
                cur = np.zeros((NL, NE))
                if r.get("domain") in dseeds:
                    seed = (dseeds[r["domain"]], dhold)
        cur = cur + counts_of(seg)
        prev = conv
        out.append(seed)
    return out


def requests_mode(a, windows, k, init, pol):
    """--requests FILE (tools/pancake/conv_trace.py): the trace cut at each request's end record, replayed plain, with
    the conversation working set, with the domain profiles (--domain-dir), and with both - the engine's rules."""
    import bisect
    import json
    reqs = json.load(open(a.requests, encoding="utf-8"))
    _, complete = window_records(a.trace)
    segs, w0 = [], 0
    for r in reqs:
        w1 = bisect.bisect_left(complete, int(r["end_record"]))
        segs.append(windows[w0:w1])
        w0 = w1
    print("requests:", ", ".join(f"{r['conv']}:{r.get('domain', '?')} {len(s)}w" for r, s in zip(reqs, segs)))
    dseeds = domain_seeds(a.domain_dir, a.slots, a.scale) if a.domain_dir else {}

    def run(name, conv_slots, ds):
        seeds = engine_seeds(reqs, segs, k, conv_slots, a.conv_tokens, ds, a.hold)
        res = simulate(segs, init, pol, seeds, 0)
        h, e = sum(x for x, _ in res), sum(y for _, y in res)
        which = [i for i, s in enumerate(seeds) if s is not None]
        hs, es = sum(res[i][0] for i in which), sum(res[i][1] for i in which)
        per = " ".join(f"{100.0 * x / max(1, y):5.1f}" for x, y in res)
        print(f"{name:<22} all {100.0 * h / max(1, e):5.2f}  seeded requests {100.0 * hs / max(1, es):5.2f} "
              f"({len(which)})  | {per}")
        return res

    run("plain", 0, {})
    run(f"conv x{a.conv_tokens:g}", 8, {})
    if dseeds:
        run(f"domain S{a.scale:g} h{a.hold}", 0, dseeds)
        run("conv + domain", 8, dseeds)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trace")
    ap.add_argument("profile")
    ap.add_argument("slots", type=int)
    ap.add_argument("--segments", default="11:code,96:prose,157:code,219:code")
    ap.add_argument("--auto-segments", action="store_true")
    ap.add_argument("--swaps", type=int, default=32)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--decay", type=float, default=0.7)
    ap.add_argument("--split-out", help="write each segment as its own trace, DIR/<i>-<domain>.trace, and stop "
                                        "(input for learn_domain_profiles.py)")
    ap.add_argument("--domain-dir", help="seed each request from DIR/<domain>.bin (learn_domain_profiles.py's files, "
                                         "as the engine reads them) instead of the leave-one-out profiles")
    ap.add_argument("--scale", type=float, default=2.0, help="with --domain-dir: STRATA_SEED_SCALE")
    ap.add_argument("--hold", type=int, default=16, help="with --domain-dir: STRATA_SEED_HOLD")
    ap.add_argument("--requests", help="conv_trace.py's <tag>.requests.json: replay the engine's rules request by request")
    ap.add_argument("--conv-tokens", type=float, default=32.0, help="with --requests: STRATA_CONV_WORKSET_TOKENS")
    a = ap.parse_args()
    windows, k = rp.load_trace(a.trace)
    if a.auto_segments:
        print("cuts (window, layer-mean Jaccard with the previous window):", auto_segments(windows))
        return
    if a.split_out:
        split_out(a.trace, a.segments, a.split_out)
        return
    init = rp.load_profile(a.profile, a.slots)
    base_pairs = rp.load_profile(a.profile, NL * NE)
    pol = rp.Policy(every=a.every, decay=a.decay, swaps=a.swaps, margin=1.5, mincand=2.0)
    if a.requests:
        print(f"{len(windows)} windows, k {k}, {a.slots} slots, swaps {a.swaps}/round every {a.every}, decay {a.decay}")
        requests_mode(a, windows, k, init, pol)
        return
    segs, doms, s0 = [], [], 0
    for spec in a.segments.split(","):
        end, dom = spec.split(":")
        segs.append(windows[s0:int(end)])
        doms.append(dom)
        s0 = int(end)
    print(f"{len(windows)} windows, k {k}, {a.slots} slots, swaps {a.swaps}/round every {a.every}, decay {a.decay}")
    print("requests:", ", ".join(f"{d} {len(s)}w/{tokens_of(s)}t" for d, s in zip(doms, segs)))

    def pct(r):
        return 100.0 * sum(h for h, _ in r) / max(1, sum(e for _, e in r))

    def row(name, res, which):
        per = " ".join(f"{100.0 * h / max(1, e):5.1f}" for h, e in res)
        sel = [res[i] for i in which]
        return f"{name:<34} all {pct(res):5.2f}  seeded {pct(sel):5.2f}  | per request {per}"

    plain = simulate(segs, init, pol, [None] * len(segs), 0)
    print(row("plain tier", plain, range(len(segs))))
    if a.domain_dir:   # the engine's path end to end: profile files -> rank seeds
        ds = domain_seeds(a.domain_dir, a.slots, a.scale)
        seeds = [ds.get(d) for d in doms]
        which = [i for i, s in enumerate(seeds) if s is not None]
        print(row(f"{a.domain_dir} S {a.scale:g} hold {a.hold}", simulate(segs, init, pol, seeds, a.hold), which))
        return

    # ---- item 4: domain seeds, leave-one-out (the profile of request i's domain from the other requests of it)
    print("\n== item 4: domain seed at each request's start (profile from the OTHER requests of its domain)")
    for top_frac in (0.5, 1.0):
        top = int(a.slots * top_frac)
        for scale in (4.0, 8.0, 16.0, 32.0):
            for hold in (0, 4, 16):
                seeds, which = [], []
                for i, d in enumerate(doms):
                    others = [segs[j] for j in range(len(segs)) if j != i and doms[j] == d]
                    if not others:
                        seeds.append(None)
                        continue
                    c = sum(counts_of(s) / max(1, tokens_of(s)) for s in others)   # each request counts once
                    seeds.append(seed_from_rank(rank_table(c, base_pairs), top, scale))
                    which.append(i)
                r = simulate(segs, init, pol, seeds, hold)
                print(row(f"top {top} S {scale:g} hold {hold}", r, which))
    # the request's own counts as a domain profile: what a perfect domain classifier + profile could reach
    for scale, hold in ((16.0, 0), (16.0, 16), (32.0, 16)):
        seeds = [seed_from_rank(rank_table(counts_of(s), base_pairs), a.slots, scale) for s in segs]
        r = simulate(segs, init, pol, seeds, hold)
        print(row(f"oracle rank S {scale:g} hold {hold}", r, range(len(segs))))

    # ---- item 3: a conversation resumes after another request
    print("\n== item 3: conversation resumes (A1, other request, A2 seeded from A1's counts)")
    for ai, oi in ((2, 1), (3, 1), (1, 2)):   # (resumed request, the one between)
        A, O = segs[ai], segs[oi]
        h = len(A) // 2
        order = [segs[0], A[:h], O, A[h:]]
        base = simulate(order, init, pol, [None] * 4, 0)
        print(f"-- {doms[ai]} request {ai} split at {h} windows, request {oi} ({doms[oi]}) between")
        print(row("  no seed", base, [3]))
        c1, t1 = counts_of(A[:h]), tokens_of(A[:h])
        for worth in (8, 32, 128):
            for hold in (0, 4, 16):
                r = simulate(order, init, pol, [None, None, None, seed_from_counts(c1, t1, worth)], hold)
                print(row(f"  counts x{worth} tokens hold {hold}", r, [3]))
        for scale, hold in ((16.0, 0), (16.0, 16)):
            r = simulate(order, init, pol, [None, None, None, seed_from_rank(rank_table(c1, base_pairs), a.slots, scale)],
                         hold)
            print(row(f"  rank S {scale:g} hold {hold}", r, [3]))
        r = simulate(order, init, pol, [None, None, None, seed_from_counts(counts_of(A[h:]), tokens_of(A[h:]), 32)], 16)
        print(row("  oracle (A2's own counts) x32 h16", r, [3]))


if __name__ == "__main__":
    main()
