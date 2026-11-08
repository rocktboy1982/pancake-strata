"""learn_domain_profiles.py - pancake-strata: one expert profile per kind of request (code / prose / tool ...) from
routing traces, for the engine's STRATA_DOMAIN_PROFILES=<dir> (MOD_PLAN phase 11, "Left, ranked" item 4;
include/strata/program/expert_seed.hpp).  Each output is <dir>/<domain>.bin in the --expert-profile format
(tools/make_profile.py), so the engine reads it with the same reader; the server names the domain per request
(serve/server.py classify_domain: code, tool, prose).

    python tools/pancake/learn_domain_profiles.py --out data/domain-profiles \\
        [--traces OUTDIR --corpus bench/profile_corpus.json]       (tools/collect_profile_traces.py's c<NN>.trace files)
        [code=edit1.trace code=edit2.trace:2 tool=agent.trace ...]  (any --dump-routing trace, DOMAIN=TRACE[:WEIGHT])
        [--map code=code,data tool=tool]   (corpus labels -> domains; every other label is prose)
        [--base data/expert-profile-pancake.bin] [--slots 10909]

As tools/make_profile_mix.py: every trace becomes a distribution (its counts / its total) times its weight, so each
trace counts once however long it is; the domain's ranking is the weighted sum, most routed first.  The pairs no trace
of the domain routed follow in the --base profile's order (the static placement the engine starts from), so a domain
profile only reorders what its traces saw.  With --slots it prints how many of each domain's top SLOTS pairs the base's
top SLOTS lack: the swaps the seed asks of the adaptive tier (at 32 a window, the windows it takes).

Measured offline (tools/pancake/replay_seeding.py, bench/prefetch/routing.bin, leave-one-out code profile, 10,909 slots,
32 swaps a window): 81.2% -> 82.9% of routed entries from VRAM, the code-edit request 77.3 -> 78.8%.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import make_profile as mp  # noqa: E402

DEFAULT_MAP = {"code": "code", "data": "code", "tool": "tool"}


def parse_spec(spec):
    """DOMAIN=TRACE[:WEIGHT] -> (domain, path, weight)."""
    dom, _, rest = spec.partition("=")
    path, weight = rest, 1.0
    head, sep, tail = rest.rpartition(":")
    if sep:
        try:
            path, weight = head, float(tail)
        except ValueError:
            pass
    return dom, path, weight


def domain_ranking(traces, base_pairs, n_expert):
    """traces: [(path, weight)] -> ranked (layer, expert) pairs (see the docstring)."""
    score = defaultdict(float)
    for path, weight in traces:
        freq = mp.read_trace(path, n_expert)
        total = float(sum(freq.values()))
        if total <= 0:
            print(f"  {path}: empty trace, skipped")
            continue
        for p, c in freq.items():
            score[p] += weight * c / total
    ranked = [p for p, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))]
    seen = set(ranked)
    for p in base_pairs:
        if p not in seen:
            ranked.append(p)
            seen.add(p)
    for e in range(n_expert):          # anything the base did not rank either: make_profile.py's fill
        for layer in range(mp.N_LAYER):
            if (layer, e) not in seen:
                ranked.append((layer, e))
    return ranked, len(score)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("specs", nargs="*", help="DOMAIN=TRACE[:WEIGHT]")
    ap.add_argument("--out", required=True, help="the directory for <domain>.bin (STRATA_DOMAIN_PROFILES)")
    ap.add_argument("--traces", help="tools/collect_profile_traces.py's output directory (c<NN>.trace)")
    ap.add_argument("--corpus", default=str(Path(__file__).resolve().parents[2] / "bench" / "profile_corpus.json"))
    ap.add_argument("--map", nargs="*", default=None, help="DOMAIN=label,label ... (default code=code,data tool=tool)")
    ap.add_argument("--base", default=str(Path(__file__).resolve().parents[2] / "data" / "expert-profile-pancake.bin"))
    ap.add_argument("--slots", type=int, default=0)
    ap.add_argument("--n-expert", type=int, default=mp.N_EXPERT)
    a = ap.parse_args()
    ne = a.n_expert
    label_map = dict(DEFAULT_MAP)
    if a.map:
        label_map = {}
        for m in a.map:
            dom, _, labels = m.partition("=")
            for lab in labels.split(","):
                label_map[lab] = dom
    groups = defaultdict(list)
    if a.traces:
        corpus = json.load(open(a.corpus, encoding="utf-8"))
        for n, c in enumerate(corpus):
            t = Path(a.traces) / f"c{n:02d}.trace"
            if t.exists() and t.stat().st_size > 0:
                groups[label_map.get(c["d"], "prose")].append((str(t), 1.0))
    for spec in a.specs:
        dom, path, w = parse_spec(spec)
        if not dom or not path:
            raise SystemExit(f"{spec}: expected DOMAIN=TRACE[:WEIGHT]")
        groups[dom].append((path, w))
    if not groups:
        raise SystemExit("no traces: give --traces DIR and/or DOMAIN=TRACE specs")
    base = mp.read_profile(a.base, ne) if a.base and Path(a.base).exists() else []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for dom, traces in sorted(groups.items()):
        ranked, n_seen = domain_ranking(traces, base, ne)
        path = out / f"{dom}.bin"
        mp.write_profile(str(path), ranked, ne)
        assert mp.read_profile(str(path), ne) == ranked, "the profile did not survive the round trip"
        msg = f"{path}: {len(traces)} traces, {n_seen} pairs routed"
        if a.slots > 0 and base:
            new = len(set(ranked[:a.slots]) - set(base[:a.slots]))
            msg += f"; {new} of its top {a.slots} are not in the base's top {a.slots} ({new / 32:.0f} windows at 32 swaps)"
        print(msg)


if __name__ == "__main__":
    main()
