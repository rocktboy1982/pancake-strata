"""restore_check.py - the restore-equivalence test (MOD_PLAN items 35 / 83).

A conversation that resumes from the conversation cache (a parked state or a message-boundary checkpoint) must answer
token for token like the same conversation read from position 0.  The engine has the switch for the comparison:
STRATA_CKPT_REREAD=1 reads a checkpoint's tokens again instead of restoring it.  With the VRAM expert set fixed
(--adapt-swaps 0) and greedy decoding, both runs must give the same text; any difference is something the checkpoint
or the restore missed (the UNSW finding on hybrid-state caches: a full-hit restore off by one token).

Each run starts a fresh server (the preset expanded as run-pancake.sh does, abtest.make_config overrides, its own port), sends the same scripted agent conversation turn by turn
- a long system prompt, a user turn, a tool call and its result, a follow-up user turn, then a branch back to an earlier
point (a partial hit) - and records every greedy answer.  Arm "restore" uses the cache as served; arm "reread" sets
STRATA_CKPT_REREAD=1.  STRATA_SNAPSHOT_VERIFY=1 is set on both (the engine also checks the drafter's snapshot).

--disk adds arm "disk" (MOD_PLAN item 39, the disk tier for parked conversations): one parking slot, the disk tier on
in a fresh folder with no size thresholds, two side conversations after every turn (the second one's park evicts the main
conversation to disk) and a server restart before "follow-up" (QUIT hands what is parked to disk; the new engine finds it).  So every
main turn after the first resumes from a state read back from disk; its answers must match the "reread" arm, and the
engine log must show the tier writing and reading back.

    python tools/pancake/restore_check.py [--preset flashnext-iq2_xs-hip] [--port 18437] [--max-tokens 96] [--disk] [--tool-boundary]
                                          [--control] [--teacher-forced] [--out FILE.json]
    python tools/pancake/restore_check.py --analyze FILE.json [--control-from OTHER.json]     (offline, no engine)
    python tools/pancake/restore_check.py --simulate FILE.json [...]                          (offline, no engine)
    (--tool-boundary: MOD_PLAN 16, every arm with STRATA_CACHE_TOOL_BOUNDARY=1; "tool-retry" resumes at the call)

Turns (each a continuation or a branch of an earlier one): first (cold), after-tool-result (the parked state, a full
continuation), follow-up (the template re-renders the last reply: a partial hit at the turn checkpoint), branch-back (a
partial hit deeper), tool-retry (the same call, another result: the tool-boundary checkpoint with --tool-boundary) and
repeat (the previous prompt sent again: a COMPLETE hit - the UNSW case, where the restored state covers the whole prompt
and the engine must still read its last token).  Greedy answers must match the re-read arm token for token, the
"repeat" answer must also equal the "tool-retry" answer inside every arm, and an arm in which the cache never served a
hit (cached_tokens 0 where a hit is expected) is refused, not passed: a cache that does nothing matches a re-read
trivially.  With --tool-boundary the engine log must also show a tool-boundary checkpoint (STRATA_TRACE=1 is set for it).

Without --control / --teacher-forced the comparison is the strict one above: exit 0 when every turn matches, 1 on a
difference, 2 on a setup error or a cache that was never exercised.  The stock engine is not reproducible token for token
(README: 1.2-1.3% of the tokens differ between runs; the 11-11 GPU run said DIFFERENT on every turn, `first` included, on the
base binary too), so that strict form can only fail.  The noise-aware form measures the noise in the same run:

  --control        a second, independent re-read arm.  For each turn that must hit the cache, the first position where the
                   cache arm differs from the re-read arm is compared with where the control differs from it.  The cache
                   arm is flagged when it diverges faster per watched position than the control does (one-sided exact test
                   of two divergence rates, level 0.05; simulate() gives its error rates).  If the control diverges so early
                   that a cache arm leaving the re-read arm after three matching text units in every turn would not be
                   flagged, exit 3.
  --teacher-forced the server's logprobs (top 5) on every request of every arm.  Up to the first divergence two runs
                   have the same context, so the logprobs of those tokens are teacher-forced for free: a cache arm must stay
                   within max(0.05 nats, twice the control's own deviation) on the shared tokens and at the flip (both
                   tokens of a split near-tied in both runs).  With it the logprob verdict decides; the position rule is
                   printed.  (The server has no way to force a continuation or return prompt logprobs; STRATA_FOLLOW in the
                   engine does, but its lines map to engine requests, not chat calls: reasoning_close_retry adds some.)
  --analyze F      offline: judge a saved --out file (--control-from G: use G's re-read arm as the control).
  --simulate F...  offline: error rates of the rule on a noise model fitted to saved files.

The exact evidence is reported apart from the comparison: inside a cache arm (restore, disk) the complete-hit "repeat" answer
must equal its own "tool-retry" answer token for token; a difference there is a real difference (exit 1).  The same pair inside
the re-read arms is only a noise sample.

Exit 0 consistent with noise (or with the strict form: every turn matches), 1 a real difference, 2 a setup error or a cache
that was never exercised, 3 inconclusive (noise floor too high to tell).  Needs the GPU for itself: run it only when no other
engine serves (it refuses when something listens on the port) and outside a benchmark.
"""
import argparse
import itertools
import json
import math
import os
import random
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SYSTEM = ("You are a coding agent working in a C++ repository. Answer briefly. " * 60).strip()   # a long, cached prefix
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
                                            "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                                                           "required": ["path"]}}}]


def turns():
    """The scripted conversation: (label, messages) per request, each one a continuation of an earlier one."""
    base = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "What does the function draft() in src/core/mtp.cpp return? Read the file."}]
    call = {"role": "assistant", "content": "",
            "tool_calls": [{"id": "c1", "type": "function",
                            "function": {"name": "read_file", "arguments": json.dumps({"path": "src/core/mtp.cpp"})}}]}
    result = {"role": "tool", "tool_call_id": "c1",
              "content": (ROOT / "src/core/mtp.cpp").read_text(errors="replace")[:6000]}
    follow = {"role": "user", "content": "Summarize the min_p handling in two sentences."}
    branch = {"role": "user", "content": "Instead, list three function names you saw."}
    # MOD_PLAN 16: the same call answered with another result (a retried step).  With STRATA_CACHE_TOOL_BOUNDARY the
    # engine resumes it at the tool-call boundary checkpoint taken in "after-tool-result" (--tool-boundary)
    retry = {"role": "tool", "tool_call_id": "c1",
             "content": (ROOT / "src/core/mtp.cpp").read_text(errors="replace")[6000:9000]}
    return [("first", base),
            ("after-tool-result", base + [call, result]),
            ("follow-up", base + [call, result, {"role": "assistant", "content": "It returns true on success."}, follow]),
            ("branch-back", base + [call, result, branch]),
            ("tool-retry", base + [call, retry]),
            ("repeat", base + [call, retry])]   # the previous prompt again: a complete hit (the UNSW off-by-one case)


SIDE_SYSTEM = ("You are a poet. Answer in one short line. " * 40).strip()   # shares no prefix with SYSTEM


def side_turn(k):
    return [{"role": "system", "content": SIDE_SYSTEM}, {"role": "user", "content": f"Write a line about the number {k}."}]


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_ready(port, proc, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(3)
    return False


def chat(port, messages, max_tokens, top_logprobs=0):
    body = {"model": "m", "messages": messages, "tools": TOOLS, "max_tokens": max_tokens, "temperature": 0,
            "reasoning_effort": "low", "stream": False}
    if top_logprobs:
        body.update({"logprobs": True, "top_logprobs": top_logprobs})
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    m = out["choices"][0]["message"]
    answer = {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
              "tool_calls": m.get("tool_calls") or [], "usage": out.get("usage", {})}
    lp = out["choices"][0].get("logprobs") or {}
    if top_logprobs:   # the server's entries without their bytes: token, logprob, the listed alternatives
        answer["logprobs"] = {k: [{"token": e["token"], "logprob": e["logprob"],
                                   "top_logprobs": [{"token": t["token"], "logprob": t["logprob"]}
                                                    for t in e.get("top_logprobs") or []]}
                                  for e in lp.get(k) or []] for k in ("reasoning_content", "content")}
    return answer


def expand_preset(preset, out):
    """configs/pancake/<preset>.json with run-pancake.sh's placeholders filled (HIP build)."""
    data = os.environ.get("STRATA_DATA", "/home/dan/clone-of-d/Strata-data")
    models = os.environ.get("MODELS", "/home/dan/clone-of-d/models")
    raw = (ROOT / "configs/pancake" / f"{preset}.json").read_text()
    raw = raw.replace("<STRATA_DATA>", data).replace("<REPO>", str(ROOT)).replace("<MODELS>", models)
    raw = raw.replace("/build-cuda/", "/build-hip/")
    Path(out).write_text(raw)


def stop_server(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)   # the server's Ctrl+C path: QUIT to the engine, then exit
        proc.wait(timeout=120)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def engine_log(name):
    """The engine's own stderr of this arm: the preset's `log` is one shared file, so the evidence checks (tool-boundary
    checkpoint, disk writes and read-backs) would count another arm's lines or none; per run so a rerun cannot inflate them."""
    return ROOT / f"restore-check-{name}.engine.{os.getpid()}.log"


def run_arm(name, env_extra, a, args_extra=None, side=False, restart_before=None):
    sys.path.insert(0, str(ROOT / "tools/pancake"))
    import abtest                                        # make_config: the same arm overrides abtest uses
    base = ROOT / f"restore-check-base.json"
    expand_preset(a.preset, base)
    cfg_path = ROOT / f"restore-check-{name}.json"
    cfg = abtest.make_config(str(base), {"env": {"STRATA_SNAPSHOT_VERIFY": "1", **env_extra},
                                         "args": {"--adapt-swaps": "0", **(args_extra or {})}}, str(cfg_path))
    cfg_path.write_text(json.dumps({**cfg, "turn_log": None, "log": str(engine_log(name))}, indent=1))   # keep the daily turn log clean
    log_out = ROOT / f"restore-check-{name}.log"
    env = dict(os.environ, LD_LIBRARY_PATH="/opt/rocm/core-10.0/lib", HIP_VISIBLE_DEVICES="0")
    # (STRATA_CACHE_TOOL_BOUNDARY is read by the server too: a config's env reaches only the engine)
    env.update({k: v for k, v in env_extra.items() if k.startswith("STRATA_CACHE_TOOL_BOUNDARY")})
    lf = open(log_out, "w")

    def start():
        p = subprocess.Popen([str(ROOT / "tools/pancake/capped.sh"), str(ROOT / ".venv/bin/python"), "serve/server.py",
                              "--engine", "strata", "--config", str(cfg_path), "--port", str(a.port)],
                             cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        if not wait_ready(a.port, p):
            print(f"[{name}] server did not come up (see {log_out})", file=sys.stderr)
            stop_server(p)
            return None
        return p

    proc = start()
    try:
        if proc is None:
            return None
        answers = []
        for k, (label, msgs) in enumerate(turns()):
            if label == restart_before:   # QUIT hands what is parked to disk; the next engine must find it there
                stop_server(proc)
                proc = start()
                if proc is None:
                    return None
            ans = chat(a.port, msgs, a.max_tokens, TOP_LOGPROBS if a.teacher_forced else 0)
            answers.append({"turn": label, **ans})
            print(f"[{name}] {label}: cached={ans['usage'].get('prompt_tokens_details', {}).get('cached_tokens')} "
                  f"out={ans['usage'].get('completion_tokens')}", flush=True)
            if side:
                # two side conversations: the first parks the main one (one slot), the second's park evicts it - to
                # disk - so the next main turn finds it on disk only (a fetch runs before the outgoing state is parked)
                chat(a.port, side_turn(2 * k), 8)
                chat(a.port, side_turn(2 * k + 1), 8)
        return answers
    finally:
        if proc is not None:
            stop_server(proc)
        lf.close()


HIT_TURNS = ("after-tool-result", "follow-up", "branch-back", "tool-retry", "repeat")   # turns that must reuse something


def cached_tokens(answer):
    return int(((answer.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0)


def never_hit(answers):
    """The turns of an arm that were expected to resume from the cache and report no cached token: a restore check on
    them proves nothing (a cache that does nothing matches a re-read)."""
    return [r["turn"] for r in answers if r["turn"] in HIT_TURNS and cached_tokens(r) <= 0]


def same_answer(a, b):
    return a["content"] == b["content"] and a["reasoning"] == b["reasoning"] and a["tool_calls"] == b["tool_calls"]


def compare(arms, reread):
    """The differences: each arm's answer vs the re-read arm's, and (inside every arm, re-read included) the complete-hit
    "repeat" vs the "tool-retry" it repeats.  A list of dicts, empty when everything matches."""
    diffs = []
    for arm, answers in arms.items():
        for r, q in zip(answers, reread):
            same = same_answer(r, q)
            print(f"{arm} {r['turn']}: {'MATCH' if same else 'DIFFERENT'}")
            if not same:
                diffs.append({"arm": arm, "turn": r["turn"], arm: r, "reread": q})
    for arm, answers in {**arms, "reread": reread}.items():
        by = {r["turn"]: r for r in answers}
        if "repeat" in by and "tool-retry" in by:
            same = same_answer(by["repeat"], by["tool-retry"])
            print(f"{arm} repeat vs tool-retry: {'MATCH' if same else 'DIFFERENT'}")
            if not same:
                diffs.append({"arm": arm, "turn": "repeat", "tool-retry": by["tool-retry"], "repeat": by["repeat"]})
    return diffs



# ---- the noise-aware form (--control, --teacher-forced) -------------------------------------------------------------

ALPHA = 0.05            # one-sided level of the divergence-rate test
BUG_POS = 3             # a run must be able to flag a cache arm that splits from the re-read arm here (text unit) in every turn
TOP_LOGPROBS = 5        # --teacher-forced: alternatives listed per token (the flip check needs both tokens of a split listed)
LP_TOL = 0.05           # nats, UNCALIBRATED (pancake-docs/batch4/equiv-gate.md): below this a logprob difference is float noise
LP_NOISE_FACTOR = 2.0   # a cache arm may deviate this many times as far as the control did
MIN_LP_TOKENS = 32      # shared tokens over the hit turns below which the logprob check cannot clear an arm
UNIT = re.compile(r"\w+|\s+|[^\w\s]")   # the saved runs hold text, not token ids: positions count these pieces (about one token each)


def by_turn(answers):
    return {r["turn"]: r for r in answers}


def units(answer):
    """The whole answer (reasoning, content, tool calls) cut into words, spaces and marks."""
    return UNIT.findall(answer["reasoning"] + "\x01" + answer["content"] + "\x01"
                        + json.dumps(answer["tool_calls"], sort_keys=True))


def split_point(a, b):
    """(position, exposure) of two answers: the index of the first piece where they differ (None: they never do) and how many
    positions were watched, position + 1 when they differ, else the shorter length.  One that stops where the other goes on
    differs there."""
    ua, ub = units(a), units(b)
    n = min(len(ua), len(ub))
    for i in range(n):
        if ua[i] != ub[i]:
            return i, i + 1
    return (None, n) if len(ua) == len(ub) else (n, n + 1)


def splits(answers, reread, turns_):
    x, y = by_turn(answers), by_turn(reread)
    return [split_point(x[t], y[t]) for t in turns_ if t in x and t in y]


def binom_tail(n, k, p):
    """P(X >= k) for X ~ Binomial(n, p)."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def rate_test(cache, control):
    """One-sided exact p-value that the cache arm leaves the re-read arm faster than the control does.  Both lists hold the
    (position, exposure) of every turn: a stretch of watched positions with at most one divergence, so the two arms'
    divergences are compared per watched position (the conditional test of two Poisson rates)."""
    events = [sum(p is not None for p, _ in s) for s in (cache, control)]
    watched = [sum(e for _, e in s) for s in (cache, control)]
    n = sum(events)
    return binom_tail(n, events[0], watched[0] / sum(watched)) if n else 1.0


def resolves(control, alpha=ALPHA):
    """Would this run have flagged a cache arm that splits from the re-read arm at BUG_POS in every turn?  If even that does
    not beat the control's own noise, a pass says nothing: the noise floor is too high."""
    return rate_test([(BUG_POS, BUG_POS + 1)] * len(control), control) <= alpha


def turns_earlier(cache, control):
    """In how many turns the cache arm leaves the re-read arm before the control does."""
    return sum(1 for (a, _), (c, _) in zip(cache, control) if a is not None and (c is None or a < c))


def turn_majority(cache, control):
    """The rule first tried: fail when the cache arm is earlier than the control in most turns (kept to measure it, simulate())."""
    return turns_earlier(cache, control) > len(cache) / 2


def repeat_vs_retry(answers):
    """The exact evidence: the complete-hit "repeat" answer against the "tool-retry" answer it repeats, inside one arm."""
    by = by_turn(answers)
    if "repeat" in by and "tool-retry" in by:
        return "MATCH" if same_answer(by["repeat"], by["tool-retry"]) else "DIFFERENT"
    return None


def position_check(arm, reread, control, turns_, alpha=ALPHA):
    cache, ctl = splits(arm, reread, turns_), splits(control, reread, turns_)
    p = rate_test(cache, ctl)
    return {"verdict": "fail" if p <= alpha else "pass" if resolves(ctl, alpha) else "inconclusive", "p": p,
            "earlier": turns_earlier(cache, ctl), "turns": len(cache), "positions": [s[0] for s in cache]}


def lp_entries(answer):
    lp = answer.get("logprobs") or {}
    return (lp.get("reasoning_content") or []) + (lp.get("content") or [])


def flip_gap(entry, other):
    """By how many nats the run's own token beat `other` in the run's own distribution: a lower bound when `other` is not
    among the listed alternatives, None when nothing is listed."""
    top = entry.get("top_logprobs") or []
    for t in top:
        if t["token"] == other:
            return entry["logprob"] - t["logprob"]
    return entry["logprob"] - min(t["logprob"] for t in top) if top else None


def lp_compare(a, b):
    """Two answers on the tokens they share (the same context, so teacher-forced by construction): how many, the largest
    logprob difference of the chosen token, and at the first token where they differ the larger of the two runs' margins
    over the other run's token (a split that float noise caused is a near-tie in both runs)."""
    ea, eb = lp_entries(a), lp_entries(b)
    n = min(len(ea), len(eb))
    k = next((i for i in range(n) if ea[i]["token"] != eb[i]["token"]), n)
    dev = max((abs(ea[i]["logprob"] - eb[i]["logprob"]) for i in range(k)), default=0.0)
    gaps = [] if k == n else [flip_gap(ea[k], eb[k]["token"]), flip_gap(eb[k], ea[k]["token"])]
    return {"tokens": k, "dev": dev, "gap": max((g for g in gaps if g is not None), default=0.0)}


def worst(cmp_):
    return max(cmp_["dev"], cmp_["gap"])


def lp_check(arm, reread, control, turns_, lp_tol=LP_TOL):
    """A cache arm stays within max(lp_tol, LP_NOISE_FACTOR x the control's own worst deviation from the re-read arm)."""
    x, y = by_turn(arm), by_turn(reread)
    seen = {t: lp_compare(x[t], y[t]) for t in turns_ if t in x and t in y}
    noise = 0.0
    if control is not None:
        c = by_turn(control)
        noise = max((worst(lp_compare(c[t], y[t])) for t in y if t in c), default=0.0)
    tol = max(lp_tol, LP_NOISE_FACTOR * noise)
    bad = [t for t, s in seen.items() if worst(s) > tol]
    tokens = sum(s["tokens"] for s in seen.values())
    return {"verdict": "fail" if bad else "pass" if tokens >= MIN_LP_TOKENS else "inconclusive", "tol": tol, "noise": noise,
            "tokens": tokens, "bad": bad, "worst": max((worst(s) for s in seen.values()), default=0.0)}


WORDS = {"pass": "consistent with noise", "fail": "REAL DIFFERENCE", "inconclusive": "INCONCLUSIVE"}


def noise_report(arms, reread, control=None, alpha=ALPHA, lp_tol=LP_TOL):
    """Every cache arm against the re-read arm, noise-aware, printed and returned: per arm the exact evidence, the position
    rule (needs the control) and the logprob check (needs --teacher-forced answers) and a verdict (the logprob check
    decides when it ran), and "exit": 1 when any arm has a real difference, else 3 when any is inconclusive, else 0."""
    turns_ = [t for t in HIT_TURNS if t in by_turn(reread)]
    use_lp = any("logprobs" in r for r in reread)
    if control is None and not use_lp:
        raise ValueError("a noise-aware comparison needs a control arm or logprobs")

    def show(s, names):
        return ", ".join(f"{t} {'-' if p is None else p}" for t, (p, _) in zip(names, s))
    print("first difference from reread (text unit, - = none); the hit turns must resume from the cache:")
    if "first" in by_turn(reread):   # a cold prompt: whatever differs here is the noise itself
        cold = {n: splits(r, reread, ["first"]) for n, r in {**arms, **({"control": control} if control is not None else {})}.items()}
        print("  first (cold, no cache involved): " + ", ".join(f"{n} {'-' if s[0][0] is None else s[0][0]}"
                                                                for n, s in cold.items() if s))
    ctl = splits(control, reread, turns_) if control is not None else None
    if ctl is not None:
        print(f"  control vs reread: {show(ctl, turns_)}")
    out = {}
    for name, answers in arms.items():
        res = {"exact": repeat_vs_retry(answers)}
        print(f"{name} vs reread: {show(splits(answers, reread, turns_), turns_)}")
        print(f"  {name} repeat == tool-retry (exact evidence): {res['exact'] or 'not run'}")
        if control is not None:
            res["position"] = p = position_check(answers, reread, control, turns_, alpha)
            note = ("" if p["verdict"] != "inconclusive" else
                    f"; the control splits so early that a cache arm splitting at unit {BUG_POS} in every turn would not be flagged"
                    " (noise floor too high)")
            print(f"  {name} position rule: earlier than the control in {p['earlier']} of {p['turns']} turns, divergence-rate "
                  f"test p = {p['p']:.3f} (level {alpha}): {WORDS[p['verdict']]}{note}")
        if use_lp:
            res["logprob"] = lp = lp_check(answers, reread, control, turns_, lp_tol)
            print(f"  {name} logprobs: {lp['tokens']} shared tokens, worst deviation {lp['worst']:.3f} nats, tolerance "
                  f"{lp['tol']:.3f} (control {lp['noise']:.3f}), over it in {lp['bad'] or 'no turn'}: {WORDS[lp['verdict']]}"
                  f"{'; too few shared tokens to clear it' if lp['verdict'] == 'inconclusive' else ''}")
        deciding = res.get("logprob") or res["position"]
        res["verdict"] = "fail" if res["exact"] == "DIFFERENT" else deciding["verdict"]
        print(f"{name}: {WORDS[res['verdict']]}")
        out[name] = res
    for name, answers in (("reread", reread), ("control", control)):
        if answers is not None and repeat_vs_retry(answers):
            print(f"{name} repeat vs tool-retry: {repeat_vs_retry(answers)} (the same prompt twice in one engine run: a noise "
                  "sample, not a check)")
    verdicts = {r["verdict"] for r in out.values()}
    return {"arms": out, "exit": 1 if "fail" in verdicts else 3 if "inconclusive" in verdicts else 0}


# ---- the error rates of the position rule (--simulate) ---------------------------------------------------------------

def estimate_hazard(run_sets):
    """Divergences per watched position between two runs of the same turn, over all pairs of runs inside each set."""
    events = watched = 0
    for runs in run_sets:
        for ra, rb in itertools.combinations(runs, 2):
            for p, e in splits(ra, rb, [r["turn"] for r in rb]):
                events += p is not None
                watched += e
    return events / watched, events, watched


def first_success(rng, p):
    """0-based index of the first success of a Bernoulli(p) trial per position."""
    return math.inf if p <= 0 else 0 if p >= 1 else int(math.log(1.0 - rng.random()) / math.log(1.0 - p))


def noise_turn(rng, h, length, bug=None):
    """One turn of the noise model: a cache arm A, the re-read arm R and the control C that do not differ in distribution.
    A near-tie comes up at each position with probability 2h; every run still sharing the prefix takes one of the two
    candidates by a fair coin, so a pair splits with probability h per position and, of three runs, A is the odd one out as
    often as C is.  bug: ("at", n) A splits from R at position n; ("rate", k) A's divergence rate from R is k times as high.
    Returns the (position, exposure) of A-R and of C-R, as split_point gives them."""
    x = y = math.inf
    pos = 0
    while pos < length and (x == math.inf or y == math.inf):
        both = x == math.inf and y == math.inf
        t = pos + first_success(rng, 2 * h if both else h)
        if t >= length:
            break
        pos = t + 1
        if not both:
            x, y = (t, y) if x == math.inf else (x, t)
            continue
        u = rng.randrange(4)   # nobody splits, both split, only A splits, only C splits
        x = t if u in (1, 2) else x
        y = t if u in (1, 3) else y
    if bug:
        kind, v = bug
        b = v if kind == "at" else first_success(rng, (v - 1) * h)
        x = min(x, b) if b < length else x
    return [(None, length) if t == math.inf else (t, t + 1) for t in (x, y)]


def simulate(h, lengths, bug=None, trials=2000, seed=1, alpha=ALPHA):
    """How often the divergence-rate rule fails / is inconclusive on `trials` simulated runs (one turn per entry of lengths),
    and how often the turn-majority rule fails."""
    rng = random.Random(seed)
    fail = inconclusive = majority = 0
    for _ in range(trials):
        pairs = [noise_turn(rng, h, n, bug) for n in lengths]
        cache, ctl = [p[0] for p in pairs], [p[1] for p in pairs]
        p = rate_test(cache, ctl)
        fail += p <= alpha
        inconclusive += p > alpha and not resolves(ctl, alpha)
        majority += turn_majority(cache, ctl)
    return {"rate_fail": fail / trials, "rate_inconclusive": inconclusive / trials, "majority_fail": majority / trials}


def load_run(path):
    """(cache arms, reread, control or None) of a saved --out file."""
    d = json.loads(Path(path).read_text())
    return {k: d[k] for k in ("restore", "disk") if k in d}, d["reread"], d.get("control")


def simulate_report(files, trials=4000):
    runs = [load_run(f) for f in files]
    h, events, watched = estimate_hazard([[*arms.values(), reread, *([control] if control else [])]
                                          for arms, reread, control in runs])
    lengths = [len(units(r)) for r in runs[0][1] if r["turn"] in HIT_TURNS]
    print(f"noise model fitted to {len(files)} saved run(s): two runs of a turn diverge at h = {h:.4f} per text unit "
          f"({events} divergences in {watched} watched positions); hit-turn lengths {lengths}")
    print(f"{'h':>7}  {'cache arm':34} {'rate rule: fail':>15} {'inconclusive':>12} {'turn majority: fail':>20}")
    for factor in (0.5, 1, 2):
        for label, bug in (("as good as a re-read (no bug)", None), (f"splits at unit {BUG_POS} in every turn", ("at", BUG_POS)),
                           ("4x the divergence rate", ("rate", 4)), ("11x the divergence rate", ("rate", 11))):
            s = simulate(h * factor, lengths, bug, trials)
            print(f"{h * factor:7.4f}  {label:34} {s['rate_fail']:15.3f} {s['rate_inconclusive']:12.3f} {s['majority_fail']:20.3f}")
    return 0


def judge(a, arms, reread, control, idle):
    """What a run says, as an exit code (and --out): 2 when a cache was never exercised, else the noise-aware verdict when
    there is a control or logprobs, else the strict comparison."""
    for arm, turns_ in idle.items():
        if turns_:
            print(f"[{arm}] no cached token on {', '.join(turns_)}: the cache was never exercised there; the result "
                  f"would prove nothing", file=sys.stderr)
    noisy = control is not None or any("logprobs" in r for r in reread)
    diffs = [] if noisy else compare(arms, reread)
    report = noise_report(arms, reread, control, a.alpha, a.lp_tol) if noisy else None
    if a.out:
        out = {**arms, "reread": reread, **({"control": control} if control is not None else {}), "diffs": diffs,
               **({"noise": report} if report else {}), **({"never_hit": idle} if any(idle.values()) else {})}
        Path(a.out).write_text(json.dumps(out, indent=1))
    if any(idle.values()):
        return 2
    return report["exit"] if noisy else 1 if diffs else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", default="flashnext-iq2_xs-hip",
                    help="configs/pancake preset (the exact HIP preset keeps decoding deterministic)")
    ap.add_argument("--port", type=int, default=18437)
    ap.add_argument("--max-tokens", type=int, default=96,
                    help="greedy tokens compared per turn (the UNSW paper compared 64-256)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tool-boundary", action="store_true",
                    help="MOD_PLAN 16: both arms with STRATA_CACHE_TOOL_BOUNDARY=1 (gap 16, so the short scripted call "
                         "is checkpointed)")
    ap.add_argument("--disk", action="store_true", help="also arm 'disk': every resume read back from the disk tier")
    ap.add_argument("--control", action="store_true",
                    help="also arm 'control': a second independent re-read arm, so the engine's own run-to-run noise is "
                         "measured in the same run and the verdict is noise-aware (exit 3: noise floor too high)")
    ap.add_argument("--teacher-forced", action="store_true",
                    help="ask the server for logprobs (top %d) on every request and compare them on the tokens two runs "
                         "share (same context); with it the logprob verdict decides" % TOP_LOGPROBS)
    ap.add_argument("--alpha", type=float, default=ALPHA, help="level of the divergence-rate test (default %(default)s)")
    ap.add_argument("--lp-tol", type=float, default=LP_TOL,
                    help="--teacher-forced: nats below which a logprob difference is float noise (default %(default)s, "
                         "uncalibrated; re-judge a saved run with --analyze to try another)")
    ap.add_argument("--analyze", metavar="FILE", help="offline: judge a saved --out file, no engine")
    ap.add_argument("--control-from", metavar="FILE",
                    help="--analyze: take FILE's re-read arm as the control (another engine start, so it also carries "
                         "whatever the two binaries differ by)")
    ap.add_argument("--simulate", nargs="+", metavar="FILE",
                    help="offline: error rates of the position rule on a noise model fitted to saved --out files")
    a = ap.parse_args(argv)
    if a.simulate:
        return simulate_report(a.simulate)
    if a.analyze:
        arms, reread, control = load_run(a.analyze)
        if a.control_from:
            control = load_run(a.control_from)[1]
        return judge(a, arms, reread, control, {arm: never_hit(answers) for arm, answers in arms.items()})
    if not port_free(a.port):
        print(f"port {a.port} is in use: refusing (one engine at a time on the GPU)", file=sys.stderr)
        return 2
    # (STRATA_TRACE makes the engine say where it took a tool-boundary checkpoint, so the run can prove it exercised one)
    tb = ({"STRATA_CACHE_TOOL_BOUNDARY": "1", "STRATA_CACHE_TOOL_BOUNDARY_GAP": "16", "STRATA_TRACE": "1"}
          if a.tool_boundary else {})
    restore = run_arm("restore", dict(tb), a)            # all arms: --adapt-swaps 0 (fixed VRAM expert set)
    reread = run_arm("reread", {"STRATA_CKPT_REREAD": "1", **tb}, a)
    control = run_arm("control", {"STRATA_CKPT_REREAD": "1", **tb}, a) if a.control else None
    if restore is None or reread is None or (a.control and control is None):
        return 2
    arms = {"restore": restore}
    if a.disk:
        import shutil
        disk_dir = ROOT / "restore-check-disk"
        shutil.rmtree(disk_dir, ignore_errors=True)
        disk = run_arm("disk", {**tb, "STRATA_CONVERSATION_DISK_DIR": str(disk_dir), "STRATA_CONVERSATION_DISK_MAX_GIB": "8",
                                "STRATA_CONVERSATION_DISK_MIN_TOKENS": "0", "STRATA_CONVERSATION_DISK_MIN_GAIN": "0"},
                       a, args_extra={"--conversation-cache-slots": "1"}, side=True, restart_before="follow-up")
        shutil.rmtree(disk_dir, ignore_errors=True)
        if disk is None:
            return 2
        log = engine_log("disk").read_text(errors="replace")
        wrote, back = log.count("conversation disk: wrote"), log.count("conversation disk: read back a")
        print(f"[disk] engine log: {wrote} writes, {back} reads back")
        if wrote < 2 or back < 2:
            print("[disk] the disk tier was not exercised (expected >= 2 writes and >= 2 reads back)", file=sys.stderr)
            return 2
        arms["disk"] = disk
    idle = {arm: never_hit(answers) for arm, answers in arms.items()}
    if a.tool_boundary:
        marks = engine_log("restore").read_text(errors="replace").count("tool boundary checkpoint at")
        print(f"[restore] engine log: {marks} tool-boundary checkpoint(s) taken")
        if marks < 1:
            print("[restore] no tool-boundary checkpoint was taken: the retry resumed from an older one, so --tool-boundary "
                  "tested nothing (check STRATA_CACHE_TOOL_BOUNDARY reached the engine and the GAP)", file=sys.stderr)
            idle["restore"] = idle.get("restore", []) + ["tool-boundary"]
    return judge(a, arms, reread, control, idle)


if __name__ == "__main__":
    sys.exit(main())
