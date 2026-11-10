#!/usr/bin/env python3
"""force_check.py: the token-forcing check (round 0217), after SpecPilot (arXiv 2611.02501, Li et al., Apache-2.0, "token forcing").

A candidate engine/build/setting is judged on a FIXED text: the engine scores prompt + continuation teacher-forced (every position's head logits come
out of the verify windows with the head in FP32: STRATA_LOGPOS / STRATA_LOGPOS_TOPK, the prompt cache off), and the checker counts where the scorer's
argmax is not the forced token.  Nothing here depends on a drafter; it serves any class N change (Q5_0 experts, tree verify, a new kernel ...).

  force_check.py gen   CFG SET.json --corpus held.jsonl [--kinds code,json] [--per-kind 4] [--offset 8] [--new 300] [--env K=V ...]
        one engine with CFG's normal decode flags; greedy continuations of the corpus prompts (rows with "ids") -> SET.json
  force_check.py score CFG SET.json OUT.tsv [--path decode|prompt] [--env K=V ...] [--arg "--flag v" ...]
        --path decode (default): one engine with CFG's own flags (drafter, spec windows, adaptive tiers ...), prompt cache off; every request decodes the
        forced continuation: the verify windows take the forced token in place of the argmax (STRATA_FOLLOW), so drafts are accepted when they match it,
        and the FP32 head logits of every valid row come out (STRATA_FOLLOW_LOGPOS, top-2).  --path prompt: the whole text is read as a prompt through the
        windows (STRATA_LOGPOS, --short-read), no decode features.  OUT.tsv.meta.json = per item line ranges.
        --env STRATA_FORCE_FAULT=7:0.3 is the test fault (adds 0.3 to every 7th token's logit in the scored rows).
  force_check.py check --stock A.tsv B.tsv [...] --cand C.tsv [--delta D] [--json out.json]
        the two rules of the paper against the stock arms (>= 2 stock arms: the noise): exit 0 pass, 1 reject, 2 inconclusive
  force_check.py ear     --stock A.tsv --cand C.tsv [--topk K] [--json out.json]
         A56: the expected acceptance rate EAR (SLQ, IST-DASLab) and the "same top token" rate of CAND against the STOCK arm A on the same
         forced positions.  Needs no engine.  check ... --ear adds the same numbers (stock arm 1 vs the candidate, and stock 1 vs stock 2 as noise).
  force_check.py selftest      the statistics on synthetic rows (no engine)

EAR (A56): per position, p = the reference arm's next-token distribution renormalised over ITS top-K ids, q = the other arm's probabilities
at those ids, renormalised; EAR_pos = sum_k min(p_k, q_k), EAR = the mean over all scored positions (SLQ's src/metrics/evaluation.py:
compute_kl_and_ear, K = 10 there).  It is the chance that a draft from the reference's distribution survives rejection sampling against the
other's, so it predicts the speculative acceptance of an approximation better than the 0/1 "same top token" rate.  The TSV holds only the top-K
ids that STRATA_LOGPOS_TOPK asked for (`score --topk K`; the default stays 2, which makes EAR a top-2 figure): an id of the reference's list
that the other arm's list lacks gets that list's smallest listed probability (an upper bound), and the count of such substitutions is printed.
An "arm" = one scored TSV.  Which engine produced the continuation and which scored it is the caller's choice:
  - a change of the scorer (a different expert format, kernel, KV type): same SET.json, `score` with the candidate build/flags;
  - a change of the decode path (tree verify, a drafter): `gen` with the candidate, `score` with the STOCK scorer (the paper's way).
Stock arms = the same procedure with the stock build, at least twice (a second `gen`, or a second `score`).

Rules (paper):  mismatch = the scorer's argmax is not the forced token; margin = log p(argmax) - log p(forced token).
  rate rule : candidate mismatch rate within +0.5 points of stock at 95% (one-sided bootstrap over requests, paired on the same positions).
              REJECT if the 95% lower bound of (cand - stock) > 0.5 points; PASS if the 95% upper bound <= 0.5 points; else INCONCLUSIVE (more tokens).
  event rule: delta = the largest margin among stock's first departures per request, rounded UP to a multiple of 1/8; a large-margin event is a
              mismatch with margin > 2 delta.  REJECT if the candidate has more events than stock by an exact one-sided binomial test (p < 0.05,
              exposure-weighted).  (The paper also audits events stock reproduces on its own paths; that audit is not done here, events are counted raw.)
"""
import argparse, json, math, os, random, signal, subprocess, sys, time, pathlib

SCORER_DROP_VALUE = {"--draft", "--draft-model", "--draft-fallback", "--mtp-draft-vocab", "--spec", "--draft-conf", "--draft-cap", "--mtp", "--spec-min-p",
                     "--prompt-cache", "--max-context", "--prefill", "--short-read", "--prompt-cache-every", "--conversation-cache-mib", "--suffix-draft"}
SCORER_DROP_FLAG = {"--serve", "--vision"}


def mem_available_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return float("inf")


# ------------------------------------------------------------------------------------------------ engine
class Engine:
    def __init__(self, cfg, args, env_extra, log):
        env = dict(os.environ)
        for k in list(env):
            if k.startswith("STRATA_"):
                env.pop(k)
        env.update(cfg.get("env", {}))
        env.update(env_extra)
        if cfg.get("lib_dirs"):
            env["LD_LIBRARY_PATH"] = ":".join(cfg["lib_dirs"] + [env.get("LD_LIBRARY_PATH", "")])
        self.errf = open(log, "wb")
        self.p = subprocess.Popen([cfg["exe"], "--serve"] + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errf, text=True,
                                  cwd=cfg.get("cwd"), env=env, bufsize=1, start_new_session=True)

    def ready(self):
        for line in self.p.stdout:
            if line.startswith("READY"):
                return True
        return False

    def gen(self, ids, n):
        self.p.stdin.write("GEN %d %s\n" % (n, ",".join(map(str, ids))))
        self.p.stdin.flush()
        toks = []
        for line in self.p.stdout:
            if line.startswith("T "):
                toks.append(int(line.split()[1]))
            elif line.startswith("DONE"):
                return toks, line.split()
            elif line.startswith("ERR"):
                raise SystemExit("engine: " + line.strip())
        raise SystemExit("the engine closed its output")

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=300)
        except Exception:
            try:
                os.killpg(self.p.pid, signal.SIGKILL)   # only our own session
            except Exception:
                self.p.kill()
        self.errf.close()


def fresh_profile(cfg, args, tag):
    """cfg "profile_src": a fresh copy of the expert profile per engine start (the engine may rewrite it) -> args with the copy's path"""
    src = cfg.get("profile_src")
    if not src or "--expert-profile" not in args:
        return args
    import shutil
    dst = os.path.abspath(tag) + ".profile.bin"
    shutil.copyfile(src, dst)
    args = list(args)
    args[args.index("--expert-profile") + 1] = dst
    return args


def scorer_args(cfg, total_tokens, extra):
    base, out, i = list(cfg["args"]), [], 0
    while i < len(base):
        x = base[i]
        if x in SCORER_DROP_FLAG:
            i += 1
        elif x in SCORER_DROP_VALUE:
            i += 2
        else:
            out.append(x)
            i += 1
    return out + ["--prompt-cache", "0", "--max-context", str(total_tokens + 512), "--short-read", str(total_tokens + 64), "--prefill", "512"] + extra


def kv_env(lst):
    d = {}
    for kv in lst or []:
        k, _, v = kv.partition("=")
        d[k] = v
    return d


# ------------------------------------------------------------------------------------------------ gen / score
def cmd_gen(a):
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    rows = [json.loads(l) for l in open(a.corpus) if l.strip()]
    rows = [r for r in rows if r.get("split", "held") == "held"]
    kinds = a.kinds.split(",") if a.kinds else sorted({r["kind"] for r in rows})
    sel = []
    for k in kinds:
        sel += [r for r in rows if r["kind"] == k][a.offset:a.offset + a.per_kind]
    if mem_available_gb() < a.min_free_gb:
        raise SystemExit("only %.0f GB available (< %d): not starting an engine" % (mem_available_gb(), a.min_free_gb))
    eng = Engine(cfg, fresh_profile(cfg, list(cfg["args"]), a.out), kv_env(a.env), a.out + ".engine.log")
    items = []
    try:
        if not eng.ready():
            raise SystemExit("the engine ended before READY, see %s.engine.log" % a.out)
        for r in sel:
            toks, f = eng.gen(r["ids"], a.new)
            items.append({"id": r["id"], "kind": r["kind"], "prompt": r["ids"], "cont": toks})
            print("%s: %d prompt, %d forced tokens" % (r["id"], len(r["ids"]), len(toks)), flush=True)
    finally:
        eng.close()
    json.dump({"config": a.config, "env": a.env or [], "items": items}, open(a.out, "w"))
    return 0


def cmd_score(a):
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    st = json.load(open(a.set))
    items = st["items"]
    total = max(len(i["prompt"]) + len(i["cont"]) for i in items)
    out = pathlib.Path(a.out).resolve()
    if out.exists():
        out.unlink()
    env = kv_env(a.env)
    extra = [x for s in (a.arg or []) for x in s.split()]
    if a.path == "decode":
        # the real decode/verify windows (drafter, spec windows, adaptive tiers, whatever the config has) with the forced continuation standing in
        # for the argmax (STRATA_FOLLOW, one line per request in decode order); rows come from STRATA_FOLLOW_LOGPOS
        follow = str(out) + ".follow.txt"
        with open(follow, "w") as ff:
            for it in items:
                ff.write(",".join(map(str, it["cont"])) + chr(10))
        env.update({"STRATA_FOLLOW": follow, "STRATA_FOLLOW_LOGPOS": str(out), "STRATA_LOGPOS_TOPK": str(a.topk)})
        args = list(cfg["args"])
        if "--prompt-cache" in args:
            i = args.index("--prompt-cache")
            del args[i:i + 2]
        args += ["--prompt-cache", "0"] + extra
    else:
        env.update({"STRATA_LOGPOS": str(out), "STRATA_LOGPOS_TOPK": str(a.topk), "STRATA_ROUTE_TAIL_SKIP": env.get("STRATA_ROUTE_TAIL_SKIP", "0")})
        args = scorer_args(cfg, total, extra)
    if mem_available_gb() < a.min_free_gb:
        raise SystemExit("only %.0f GB available (< %d): not starting an engine" % (mem_available_gb(), a.min_free_gb))
    args = fresh_profile(cfg, args, str(out))
    eng = Engine(cfg, args, env, str(out) + ".engine.log")

    def nlines():
        return sum(1 for _ in open(out)) if out.exists() else 0
    meta = []
    try:
        if not eng.ready():
            raise SystemExit("the engine ended before READY, see %s.engine.log" % out)
        for it in items:
            ids = it["prompt"] + it["cont"]
            n0 = nlines()
            if a.path == "decode":
                toks, f = eng.gen(it["prompt"], len(it["cont"]))
                if toks != it["cont"]:
                    print("%s: WARNING the engine emitted %d tokens, %d differ from the forced continuation" % (
                        it["id"], len(toks), sum(1 for x, y in zip(toks, it["cont"]) if x != y) + abs(len(toks) - len(it["cont"]))), flush=True)
            else:
                toks, f = eng.gen(ids, 1)
            n1 = nlines()
            if n1 - n0 < len(it["cont"]) // 2:
                raise SystemExit("%s: only %d positions scored (the prompt took the batched path? raise --short-read; decode path: STRATA_FOLLOW not in this build?)" % (it["id"], n1 - n0))
            meta.append({"id": it["id"], "kind": it["kind"], "first_line": n0, "end_line": n1, "prompt_len": len(it["prompt"]), "ids": ids})
            print("%s: %d positions" % (it["id"], n1 - n0), flush=True)
    finally:
        eng.close()
    json.dump({"set": a.set, "config": a.config, "path": a.path, "env": a.env or [], "args": args, "items": meta}, open(str(out) + ".meta.json", "w"))
    return 0


# ------------------------------------------------------------------------------------------------ check
def load_arm(tsv):
    """-> {request id: [(is_mismatch, margin, top2_margin), ...]} over the continuation positions"""
    meta = json.load(open(tsv + ".meta.json"))
    lines = open(tsv).read().splitlines()
    out = {}
    for it in meta["items"]:
        ids, P = it["ids"], it["prompt_len"]
        rows = []
        for ln in lines[it["first_line"]:it["end_line"]]:
            f = ln.split("\t")
            pos, tgt = int(f[0]), int(f[1])
            if pos < P - 1 or pos + 1 >= len(ids):
                continue
            if ids[pos + 1] != tgt:
                raise SystemExit("%s: line for position %d scores token %d, the set has %d" % (it["id"], pos, tgt, ids[pos + 1]))
            lp_t, am, lp_am = float(f[2]), int(f[3]), float(f[4])
            lp2 = None
            for x in f[8:10]:
                i, _, v = x.partition(":")
                if int(i) != am:
                    lp2 = float(v)
                    break
            rows.append((am != tgt, max(0.0, lp_am - lp_t), (lp_am - lp2) if lp2 is not None else float("nan")))
        out[it["id"]] = rows
    return out


# ------------------------------------------------------------------------------------------------ EAR (A56)
def load_dists(tsv):
    """-> {request id: {position: (argmax id, {id: probability})}}: the listed ids of every scored continuation position (the forced token
    and the argmax from the fixed columns, the STRATA_LOGPOS_TOPK list from column 8 on)."""
    meta = json.load(open(tsv + ".meta.json"))
    lines = open(tsv).read().splitlines()
    out = {}
    for it in meta["items"]:
        ids, P = it["ids"], it["prompt_len"]
        rows = {}
        for ln in lines[it["first_line"]:it["end_line"]]:
            f = ln.split("\t")
            pos = int(f[0])
            if pos < P - 1 or pos + 1 >= len(ids):
                continue
            d = {int(f[1]): math.exp(float(f[2])), int(f[3]): math.exp(float(f[4]))}
            for x in f[8:]:
                i, _, v = x.partition(":")
                if v:
                    d[int(i)] = math.exp(float(v))
            rows[pos] = (int(f[3]), d)
        out[it["id"]] = rows
    return out


def ear_position(ref, cand, k):
    """One position: (EAR_pos, substitutions).  ref / cand = {id: probability} of the listed ids.  S = the reference's k most likely listed ids;
    p, q renormalised over S; an id of S that cand does not list gets cand's smallest listed probability (an upper bound)."""
    top = sorted(ref, key=ref.get, reverse=True)[:k]
    floor = min(cand.values())
    miss = sum(1 for v in top if v not in cand)
    p = [ref[v] for v in top]
    q = [cand.get(v, floor) for v in top]
    sp, sq = sum(p), sum(q)
    if sp <= 0 or sq <= 0:
        return 0.0, miss
    return sum(min(a / sp, b / sq) for a, b in zip(p, q)), miss


def ear_compare(ref, cand, k=10, seed=1, boots=2000):
    """ref, cand: load_dists() results over the same set.  -> dict: EAR and the same-top-token rate, token-weighted over the positions both
    arms scored, with a 95% bootstrap interval over requests."""
    per = []   # per request: (positions, sum EAR, same-top count, substitutions)
    for rid, rrows in ref.items():
        crows = cand.get(rid)
        if not crows:
            continue
        n = e = same = miss = 0
        for pos, (am_r, dr) in rrows.items():
            if pos not in crows:
                continue
            am_c, dc = crows[pos]
            ep, m = ear_position(dr, dc, k)
            n += 1
            e += ep
            same += am_r == am_c
            miss += m
        if n:
            per.append((n, e, same, miss))
    if not per:
        raise SystemExit("ear: no position scored by both arms")
    tot = sum(x[0] for x in per)

    def agg(sel):
        t = sum(per[j][0] for j in sel)
        return sum(per[j][1] for j in sel) / t, sum(per[j][2] for j in sel) / t
    ear, same = agg(range(len(per)))
    rng = random.Random(seed)
    es, ss = [], []
    for _ in range(boots):
        a, b = agg([rng.randrange(len(per)) for _ in per])
        es.append(a)
        ss.append(b)
    es.sort()
    ss.sort()
    lo, hi = int(0.025 * boots), int(0.975 * boots) - 1
    return {"requests": len(per), "positions": tot, "topk": k, "ear": round(ear, 5), "ear_ci95": [round(es[lo], 5), round(es[hi], 5)],
            "same_top_token": round(same, 5), "same_top_token_ci95": [round(ss[lo], 5), round(ss[hi], 5)],
            "substituted_ids": sum(x[3] for x in per)}


def cmd_ear(a):
    r = ear_compare(load_dists(a.stock), load_dists(a.cand), a.topk)
    print(json.dumps(r, indent=1))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    return 0


def ceil8(x):
    return max(0.125, math.ceil(x * 8.0 - 1e-12) / 8.0)


def binom_tail_ge(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def check(stock, cand, delta=None, seed=1, boots=4000):
    """stock: list of arms, cand: one arm (each {request: rows}); the requests are the same. -> dict"""
    ids = [i for i in cand if all(i in s for s in stock)]
    # delta: largest margin at stock's first departure per request (every stock arm), rounded up to 1/8
    if delta is None:
        firsts = []
        for s in stock:
            for i in ids:
                for mm, mg, _ in s[i]:
                    if mm:
                        firsts.append(mg)
                        break
        delta = ceil8(max(firsts)) if firsts else 0.125

    def counts(arm):
        return [(len(arm[i]), sum(1 for r in arm[i] if r[0]), sum(1 for r in arm[i] if r[0] and r[1] > 2 * delta)) for i in ids]
    sc = [counts(s) for s in stock]
    cc = counts(cand)
    n_s = [sum(c[0] for c in a) for a in sc]
    n_c = sum(c[0] for c in cc)

    def rate(cs):
        return 100.0 * sum(c[1] for c in cs) / max(1, sum(c[0] for c in cs))
    r_stock = [rate(a) for a in sc]
    r_cand = rate(cc)
    r_s = sum(r_stock) / len(r_stock)
    rng = random.Random(seed)
    diffs = []
    for _ in range(boots):
        pick = [rng.randrange(len(ids)) for _ in ids]
        dn = sum(cc[j][0] for j in pick)
        dc = 100.0 * sum(cc[j][1] for j in pick) / max(1, dn)
        ds = sum(100.0 * sum(a[j][1] for j in pick) / max(1, sum(a[j][0] for j in pick)) for a in sc) / len(sc)
        diffs.append(dc - ds)
    diffs.sort()
    lo, hi = diffs[int(0.05 * boots)], diffs[int(0.95 * boots) - 1]
    if lo > 0.5:
        rate_v = "REJECT"
    elif hi <= 0.5:
        rate_v = "PASS"
    else:
        rate_v = "INCONCLUSIVE"
    ev_c = sum(c[2] for c in cc)
    ev_s = sum(sum(c[2] for c in a) for a in sc)
    N_s = sum(n_s)
    p0 = n_c / (n_c + N_s)
    n = ev_c + ev_s
    p = binom_tail_ge(ev_c, n, p0) if n > 0 else 1.0
    event_v = "REJECT" if p < 0.05 else "PASS"
    # stock-vs-stock spread (the noise): per-arm rates and large-margin events
    return {"requests": len(ids), "tokens_cand": n_c, "tokens_stock_arms": n_s, "delta": delta, "mismatch_rate_stock_arms_pct": [round(x, 3) for x in r_stock],
            "mismatch_rate_cand_pct": round(r_cand, 3), "diff_pts": round(r_cand - r_s, 3), "diff_ci90_pts": [round(lo, 3), round(hi, 3)],
            "rate_rule": rate_v, "events_cand": ev_c, "events_stock_arms": [sum(c[2] for c in a) for a in sc], "events_p": round(p, 5),
            "event_rule": event_v, "verdict": "REJECT" if "REJECT" in (rate_v, event_v) else ("PASS" if rate_v == "PASS" else "INCONCLUSIVE")}


def cmd_check(a):
    stock = [load_arm(t) for t in a.stock]
    cand = load_arm(a.cand)
    if len(stock) < 2:
        print("warning: one stock arm only: the noise between stock runs is not measured", file=sys.stderr)
    r = check(stock, cand, a.delta)
    if a.ear:   # A56: the candidate vs stock arm 1 (and stock 1 vs stock 2: the noise), next to the mismatch rates
        d0 = load_dists(a.stock[0])
        r["ear"] = ear_compare(d0, load_dists(a.cand), a.topk)
        if len(a.stock) > 1:
            r["ear_stock1_vs_stock2"] = ear_compare(d0, load_dists(a.stock[1]), a.topk)
    print(json.dumps(r, indent=1))
    if a.json:
        json.dump(r, open(a.json, "w"), indent=1)
    return {"PASS": 0, "REJECT": 1, "INCONCLUSIVE": 2}[r["verdict"]]


def cmd_selftest(a):
    rng = random.Random(7)

    def arm(p_mis, p_big, nreq=60, n=300):
        d = {}
        for q in range(nreq):
            rows = []
            for _ in range(n):
                if rng.random() < p_mis:
                    big = rng.random() < p_big
                    rows.append((True, rng.uniform(1.0, 3.0) if big else rng.uniform(0.0, 0.4), 1.0))
                else:
                    rows.append((False, 0.0, 2.0))
            d["r%d" % q] = rows
        return d
    s1, s2, s3 = arm(0.007, 0.0), arm(0.007, 0.0), arm(0.007, 0.0)
    ok = check([s1, s2], s3)
    bad = check([s1, s2], arm(0.02, 0.0))
    ev = check([s1, s2], arm(0.0075, 0.5))
    print("stock vs stock:", ok["verdict"], ok["diff_pts"], ok["diff_ci90_pts"], "events p", ok["events_p"])
    print("rate +1.3 pts:", bad["verdict"], bad["diff_pts"], bad["diff_ci90_pts"])
    print("more large-margin events:", ev["verdict"], ev["events_cand"], ev["events_stock_arms"], ev["events_p"])
    assert ok["verdict"] != "REJECT" and bad["verdict"] == "REJECT" and ev["event_rule"] == "REJECT", "selftest failed"
    print("selftest ok")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen"); g.add_argument("config"); g.add_argument("out"); g.add_argument("--corpus", required=True)
    g.add_argument("--kinds", default=""); g.add_argument("--per-kind", type=int, default=4); g.add_argument("--offset", type=int, default=8)
    g.add_argument("--new", type=int, default=300); g.add_argument("--env", action="append"); g.add_argument("--min-free-gb", type=int, default=20)
    s = sp.add_parser("score"); s.add_argument("config"); s.add_argument("set"); s.add_argument("out")
    s.add_argument("--path", choices=["decode", "prompt"], default="decode", help="decode: the real verify windows with the forced tokens (STRATA_FOLLOW); prompt: the text read as a prompt (STRATA_LOGPOS)")
    s.add_argument("--env", action="append"); s.add_argument("--arg", action="append"); s.add_argument("--min-free-gb", type=int, default=20)
    s.add_argument("--topk", type=int, default=2, help="top-K ids and log-probabilities per position in the TSV (2: the gate's need; 10: EAR as SLQ defines it)")
    c = sp.add_parser("check"); c.add_argument("--stock", nargs="+", required=True); c.add_argument("--cand", required=True)
    c.add_argument("--delta", type=float); c.add_argument("--json")
    c.add_argument("--ear", action="store_true", help="also print EAR and the same-top-token rate (A56)")
    c.add_argument("--topk", type=int, default=10, help="with --ear: the K of EAR (the TSV must hold that many ids; see score --topk)")
    e = sp.add_parser("ear"); e.add_argument("--stock", required=True); e.add_argument("--cand", required=True)
    e.add_argument("--topk", type=int, default=10); e.add_argument("--json")
    sp.add_parser("selftest")
    a = ap.parse_args()
    return {"gen": cmd_gen, "score": cmd_score, "check": cmd_check, "ear": cmd_ear, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
