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
                                          [--out FILE.json]
    (--tool-boundary: MOD_PLAN 16, every arm with STRATA_CACHE_TOOL_BOUNDARY=1; "tool-retry" resumes at the call)

Turns (each a continuation or a branch of an earlier one): first (cold), after-tool-result (the parked state, a full
continuation), follow-up (the template re-renders the last reply: a partial hit at the turn checkpoint), branch-back (a
partial hit deeper), tool-retry (the same call, another result: the tool-boundary checkpoint with --tool-boundary) and
repeat (the previous prompt sent again: a COMPLETE hit - the UNSW case, where the restored state covers the whole prompt
and the engine must still read its last token).  Greedy answers must match the re-read arm token for token, the
"repeat" answer must also equal the "tool-retry" answer inside every arm, and an arm in which the cache never served a
hit (cached_tokens 0 where a hit is expected) is refused, not passed: a cache that does nothing matches a re-read
trivially.  With --tool-boundary the engine log must also show a tool-boundary checkpoint (STRATA_TRACE=1 is set for it).

Exit 0 when every turn matches, 1 on a difference, 2 on a setup error or a cache that was never exercised.  Needs the
GPU for itself: run it only when no other engine serves (it refuses when something listens on the port) and outside a
benchmark.
"""
import argparse
import json
import os
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


def chat(port, messages, max_tokens):
    body = {"model": "m", "messages": messages, "tools": TOOLS, "max_tokens": max_tokens, "temperature": 0,
            "reasoning_effort": "low", "stream": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.loads(r.read())
    m = out["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "",
            "tool_calls": m.get("tool_calls") or [], "usage": out.get("usage", {})}


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


def run_arm(name, env_extra, a, args_extra=None, side=False, restart_before=None):
    sys.path.insert(0, str(ROOT / "tools/pancake"))
    import abtest                                        # make_config: the same arm overrides abtest uses
    base = ROOT / f"restore-check-base.json"
    expand_preset(a.preset, base)
    cfg_path = ROOT / f"restore-check-{name}.json"
    cfg = abtest.make_config(str(base), {"env": {"STRATA_SNAPSHOT_VERIFY": "1", **env_extra},
                                         "args": {"--adapt-swaps": "0", **(args_extra or {})}}, str(cfg_path))
    cfg_path.write_text(json.dumps({**cfg, "turn_log": None}, indent=1))   # keep the daily turn log clean
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
            ans = chat(a.port, msgs, a.max_tokens)
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


def main():
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
    a = ap.parse_args()
    if not port_free(a.port):
        print(f"port {a.port} is in use: refusing (one engine at a time on the GPU)", file=sys.stderr)
        return 2
    # (STRATA_TRACE makes the engine say where it took a tool-boundary checkpoint, so the run can prove it exercised one)
    tb = ({"STRATA_CACHE_TOOL_BOUNDARY": "1", "STRATA_CACHE_TOOL_BOUNDARY_GAP": "16", "STRATA_TRACE": "1"}
          if a.tool_boundary else {})
    restore = run_arm("restore", dict(tb), a)            # all arms: --adapt-swaps 0 (fixed VRAM expert set)
    reread = run_arm("reread", {"STRATA_CKPT_REREAD": "1", **tb}, a)
    if restore is None or reread is None:
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
        log = (ROOT / "restore-check-disk.log").read_text(errors="replace")
        wrote, back = log.count("conversation disk: wrote"), log.count("conversation disk: read back a")
        print(f"[disk] engine log: {wrote} writes, {back} reads back")
        if wrote < 2 or back < 2:
            print("[disk] the disk tier was not exercised (expected >= 2 writes and >= 2 reads back)", file=sys.stderr)
            return 2
        arms["disk"] = disk
    idle = {arm: never_hit(answers) for arm, answers in arms.items()}
    for arm, turns_ in idle.items():
        if turns_:
            print(f"[{arm}] no cached token on {', '.join(turns_)}: the cache was never exercised there; the result "
                  f"would prove nothing", file=sys.stderr)
    if a.tool_boundary:
        marks = (ROOT / "restore-check-restore.log").read_text(errors="replace").count("tool boundary checkpoint at")
        print(f"[restore] engine log: {marks} tool-boundary checkpoint(s) taken")
        if marks < 1:
            print("[restore] no tool-boundary checkpoint was taken: the retry resumed from an older one, so --tool-boundary "
                  "tested nothing (check STRATA_CACHE_TOOL_BOUNDARY reached the engine and the GAP)", file=sys.stderr)
            idle["restore"] = idle.get("restore", []) + ["tool-boundary"]
    diffs = compare(arms, reread)
    if any(idle.values()):
        if a.out:
            Path(a.out).write_text(json.dumps({**arms, "reread": reread, "diffs": diffs, "never_hit": idle}, indent=1))
        return 2
    if a.out:
        Path(a.out).write_text(json.dumps({**arms, "reread": reread, "diffs": diffs}, indent=1))
    return 1 if diffs else 0


if __name__ == "__main__":
    sys.exit(main())
